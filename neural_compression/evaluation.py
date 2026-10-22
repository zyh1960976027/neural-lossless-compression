"""独立分块概率评估；计入每块首 token 的均匀编码成本。"""
import math
import tempfile
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .model import autocast_context, load_checkpoint
from .data import (
    TokenBlockDataset,
    LosslessTokenizer,
    tokenize_file,
    require_baseline_tokenizer,
)
from .io import sha256_file


def block_loss(logits, targets, mask):
    # Fixed shape avoids the variable-size GPU indexing synchronization.
    loss = F.cross_entropy(
        logits.flatten(0, 1).float(), targets.flatten(), reduction="none"
    )
    return (loss * mask.flatten()).sum(), mask.sum()


@torch.inference_mode()
def validate(model, dataset, raw_bytes, batch_size, device, workers=0):
    model.eval()
    total = torch.zeros((), dtype=torch.float64, device=device)
    targets_seen = 0
    started = time.perf_counter()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    for values, targets, masks in loader:
        targets_seen += int(masks.sum())
        values, targets, masks = [
            v.to(device, non_blocking=True) for v in (values, targets, masks)
        ]
        with autocast_context(device):
            logits = model(values)
        nll, _ = block_loss(logits, targets, masks)
        total += nll.double()
        del logits, nll
    expected = dataset.num_tokens - dataset.num_blocks
    if targets_seen != expected:
        raise AssertionError("Validation crossed a block boundary")
    model_nll = float(total)
    seed_nll = dataset.num_blocks * math.log(model.embedding.num_embeddings)
    bpb = (model_nll + seed_nll) / (raw_bytes * math.log(2)) if raw_bytes else 0.0
    if not math.isfinite(bpb):
        raise FloatingPointError("Non-finite validation result")
    return dict(
        bpb=bpb,
        model_nll=model_nll,
        uniform_seed_nll=seed_nll,
        raw_bytes=raw_bytes,
        tokens=dataset.num_tokens,
        predicted_tokens=targets_seen,
        num_blocks=dataset.num_blocks,
        elapsed_seconds=time.perf_counter() - started,
    )


def evaluate(
    input_path,
    checkpoint,
    tokenizer,
    *,
    batch_size=72,
    device="cuda:0",
    workers=0,
    tokenizer_threads=24
):
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(device)
    adapter = require_baseline_tokenizer(
        LosslessTokenizer(tokenizer, threads=tokenizer_threads)
    )
    model, state = load_checkpoint(checkpoint)
    if state["tokenizer_sha256"] != adapter.sha256:
        raise ValueError("分词器与模型检查点不配套")
    model.to(device)
    with tempfile.TemporaryDirectory(prefix="nlc-evaluate-") as temporary:
        cache = Path(temporary) / "input.tokens"
        metadata = tokenize_file(adapter, input_path, output=cache)
        metrics = validate(
            model,
            TokenBlockDataset(cache),
            metadata["raw_bytes"],
            batch_size,
            device,
            workers,
        )
    metrics.update(
        operation="evaluate",
        metric="model_probability_bpb",
        input=str(Path(input_path).resolve()),
        input_sha256=metadata["original_sha256"],
        checkpoint_sha256=sha256_file(checkpoint),
        tokenizer_sha256=adapter.sha256,
        first_token_policy="uniform_per_independent_block",
        block_size=2048,
    )
    return metrics

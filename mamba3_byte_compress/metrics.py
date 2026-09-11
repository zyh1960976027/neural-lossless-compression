"""NLL and bits-per-byte metrics for byte probability models."""

import math
import time

import torch
import torch.nn.functional as F


def autocast_context(device, enabled=True):
    device = torch.device(device)
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=bool(enabled and device.type == "cuda"),
    )


def masked_nll(logits, targets, masks):
    """Return summed FP32 cross-entropy and the number of valid byte targets."""

    flat_masks = masks.reshape(-1)
    valid_tokens = int(flat_masks.sum().item())
    if valid_tokens == 0:
        return 0.0, 0

    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="none",
    )
    return losses[flat_masks].sum().item(), valid_tokens


@torch.no_grad()
def evaluate_probability_model(model, dataloader, device, use_bf16=True):
    """Evaluate file-level NLL/BPB with the frozen baseline definition."""

    dataset = dataloader.dataset
    if not hasattr(dataset, "raw_bytes") or not hasattr(dataset, "predicted_bytes"):
        raise TypeError("Probability evaluation requires ByteEvaluationDataset metadata.")

    was_training = model.training
    model.eval()
    predicted_nll = 0.0
    valid_tokens = 0
    started = time.perf_counter()
    try:
        for batch in dataloader:
            inputs = batch["input_tokens"].to(device, non_blocking=True)
            targets = batch["target_tokens"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            with autocast_context(device, enabled=use_bf16):
                logits = model(inputs)
            batch_nll, batch_valid = masked_nll(logits, targets, masks)
            predicted_nll += batch_nll
            valid_tokens += batch_valid
    finally:
        if was_training:
            model.train()

    if valid_tokens != dataset.predicted_bytes:
        raise AssertionError(
            f"Evaluation covered {valid_tokens} predicted bytes; "
            f"expected {dataset.predicted_bytes}."
        )

    first_byte_nll = math.log(256.0) if dataset.raw_bytes else 0.0
    total_nll = predicted_nll + first_byte_nll
    nll_per_byte = total_nll / dataset.raw_bytes if dataset.raw_bytes else 0.0
    return {
        "nll_nats_total": total_nll,
        "nll_nats_predicted": predicted_nll,
        "nll_nats_uniform_first_byte": first_byte_nll,
        "nll_nats_per_byte": nll_per_byte,
        "bpb": nll_per_byte / math.log(2.0),
        "num_bytes": int(dataset.raw_bytes),
        "predicted_bytes": int(valid_tokens),
        "sequence_length": int(dataset.sequence_length),
        "context_reset_each_block": True,
        "elapsed_seconds": time.perf_counter() - started,
    }

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
def evaluate_probability_model(model, dataloader, device, use_bf16=True, *,
                               measure_performance=False):
    """Evaluate NLL/BPB, optionally profiling forward time and CUDA memory.

    Profiling synchronizes CUDA around each forward pass. It includes first-call
    compilation, but excludes data loading/transfers and loss computation.
    KB and GB use decimal units; throughput counts valid predicted bytes only.
    """

    dataset = dataloader.dataset
    if not hasattr(dataset, "raw_bytes") or not hasattr(dataset, "predicted_bytes"):
        raise TypeError("Probability evaluation requires ByteEvaluationDataset metadata.")

    device = torch.device(device)
    profile_cuda = measure_performance and device.type == "cuda"
    if profile_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    inference_seconds = 0.0
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
                if measure_performance:
                    if profile_cuda:
                        torch.cuda.synchronize(device)
                    inference_started = time.perf_counter()
                logits = model(inputs)
                if measure_performance:
                    if profile_cuda:
                        torch.cuda.synchronize(device)
                    inference_seconds += time.perf_counter() - inference_started
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
    result = {
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
    if measure_performance:
        result.update({
            "inference_seconds": inference_seconds,
            "inference_kb_per_second": (
                valid_tokens / 1000.0 / inference_seconds
                if inference_seconds > 0.0 else 0.0
            ),
            "peak_gpu_memory_gb": (
                torch.cuda.max_memory_allocated(device) / 1_000_000_000.0
                if profile_cuda else None
            ),
            "peak_gpu_reserved_memory_gb": (
                torch.cuda.max_memory_reserved(device) / 1_000_000_000.0
                if profile_cuda else None
            ),
        })
    return result

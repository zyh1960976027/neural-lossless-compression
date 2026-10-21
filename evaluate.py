"""Evaluate a raw byte file with a trained Mamba-3 checkpoint."""

import argparse
import csv
from itertools import islice
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from neural_compression.config import FORMAL_TRAINING_CONFIG
from neural_compression.data import ByteEvaluationDataset
from neural_compression.experiments import (
    DEFAULT_CHECKPOINT,
    TEST_EXPERIMENTS_ROOT,
    create_experiment_directory,
    validate_experiment_name,
)
from neural_compression.metrics import autocast_context, evaluate_probability_model
from neural_compression.trainer import load_mamba3_checkpoint


DEFAULT_SEQUENCE_LENGTH = 2048
SUMMARY_HEADERS = ("BPB", "Inference (KB/s)", "Peak GPU memory (GB)")


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise ValueError("must be a positive integer")
    return value


def _nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise ValueError("must be a nonnegative integer")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", required=True, help="Raw byte file to evaluate")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                        help="Project weights; default: checkpoints/best.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--name", type=validate_experiment_name,
                        help="Test name; default: checkpoint filename without extension. A timestamp is always prepended.")
    parser.add_argument("--batch-size", type=_positive_int, default=None,
                        help="Default: checkpoint validation batch size, or 16")
    parser.add_argument("--sequence-length", type=_positive_int, default=DEFAULT_SEQUENCE_LENGTH,
                        help="Default: 2048, independent of checkpoint configuration")
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--warmup-batches", type=_nonnegative_int, default=0,
                        help="Forward-only warmup batches excluded from reported inference time")
    return parser.parse_args(argv)


def _cuda_device(device):
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Mamba-3 evaluation requires an available CUDA device.")
    index = torch.cuda.current_device() if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index is unavailable: {index}")
    device = torch.device("cuda", index)
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device must support BF16.")
    return device


def evaluate_file(test_file, checkpoint=DEFAULT_CHECKPOINT, *, device="cuda:0", batch_size=None,
                  sequence_length=DEFAULT_SEQUENCE_LENGTH, num_workers=0, warmup_batches=0):
    """Load project weights and return file-level metrics plus run metadata."""
    test_file = Path(test_file).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    for label, path in (("test", test_file), ("checkpoint", checkpoint_path)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} file: {path}")
    num_workers = _nonnegative_int(num_workers)
    warmup_batches = _nonnegative_int(warmup_batches)
    if batch_size is not None:
        batch_size = _positive_int(batch_size)
    if sequence_length is not None:
        sequence_length = _positive_int(sequence_length)
    device = _cuda_device(device)
    model_load_started = time.perf_counter()
    try:
        model, loaded = load_mamba3_checkpoint(checkpoint_path)
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("Invalid project checkpoint: expected model weights and metadata.") from error
    training_config = loaded.get("training_config", {})
    if not isinstance(training_config, dict):
        raise ValueError("Checkpoint training_config must be a dictionary.")
    batch_size = _positive_int(
        batch_size if batch_size is not None else training_config.get(
            "validation_batch_size", FORMAL_TRAINING_CONFIG["validation_batch_size"]
        )
    )
    sequence_length = _positive_int(
        DEFAULT_SEQUENCE_LENGTH if sequence_length is None else sequence_length
    )
    metadata = {
        "test_file": str(test_file),
        "checkpoint": str(checkpoint_path),
        "model_name": loaded["model_name"],
        "checkpoint_kind": loaded.get("checkpoint_kind"),
        "global_update": loaded.get("global_update"),
        "device": str(device),
        "precision": "bf16_amp_fp32_master",
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    del loaded  # Release optimizer and other training state before inference.
    model.to(device).eval()
    model_load_peak_allocated_gb = None
    model_load_peak_reserved_gb = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        model_load_peak_allocated_gb = torch.cuda.max_memory_allocated(device) / 1_000_000_000.0
        model_load_peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1_000_000_000.0
    model_load_seconds = time.perf_counter() - model_load_started
    dataset = ByteEvaluationDataset(test_file, sequence_length)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    warmup_count = min(warmup_batches, len(loader))
    warmup_seconds = 0.0
    warmup_peak_allocated_gb = None
    warmup_peak_reserved_gb = None
    if warmup_count:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.inference_mode(), autocast_context(device, enabled=True):
            for batch in islice(loader, warmup_count):
                model(batch["input_tokens"].to(device, non_blocking=True))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            warmup_peak_allocated_gb = torch.cuda.max_memory_allocated(device) / 1_000_000_000.0
            warmup_peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1_000_000_000.0
        warmup_seconds = time.perf_counter() - started
    with torch.inference_mode():
        result = evaluate_probability_model(
            model, loader, device, use_bf16=True, measure_performance=True
        )
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise FloatingPointError("Evaluation produced non-finite metrics.")
    result.update(metadata, warmup_batches_requested=warmup_batches,
                  warmup_batches_completed=warmup_count,
                  model_load_seconds=model_load_seconds,
                  warmup_seconds=warmup_seconds,
                  model_load_peak_gpu_memory_gb=model_load_peak_allocated_gb,
                  model_load_peak_gpu_reserved_memory_gb=model_load_peak_reserved_gb,
                  warmup_peak_gpu_memory_gb=warmup_peak_allocated_gb,
                  warmup_peak_gpu_reserved_memory_gb=warmup_peak_reserved_gb)
    return result


def format_summary(result):
    """Render the evaluation summary without adding a table dependency."""
    headers = SUMMARY_HEADERS
    peak_memory = result["peak_gpu_memory_gb"]
    values = (
        f"{result['bpb']:.6f}",
        f"{result['inference_kb_per_second']:.3f}",
        f"{peak_memory:.6f}" if peak_memory is not None else "N/A",
    )
    widths = [max(len(header), len(value)) for header, value in zip(headers, values)]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def row(cells):
        return "| " + " | ".join(
            cell.rjust(width) for cell, width in zip(cells, widths)
        ) + " |"

    return "\n".join((border, row(headers), border, row(values), border))


def main(argv=None):
    args = parse_args(argv)
    test_name = validate_experiment_name(
        args.name if args.name is not None else Path(args.checkpoint).stem
    )
    experiment_dir = create_experiment_directory(TEST_EXPERIMENTS_ROOT, test_name)
    try:
        result = evaluate_file(
            args.test_file, args.checkpoint, device=args.device,
            batch_size=args.batch_size, sequence_length=args.sequence_length,
            num_workers=args.num_workers, warmup_batches=args.warmup_batches,
        )
    except BaseException:
        # Failed evaluations must not leave an empty experiment behind.
        experiment_dir.rmdir()
        raise
    result.update({
        "test_name": test_name,
        "experiment_name": experiment_dir.name,
        "experiment_dir": str(experiment_dir),
    })
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    (experiment_dir / "results.json").write_text(serialized + "\n", encoding="utf-8")
    with (experiment_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUMMARY_HEADERS)
        writer.writerow((result["bpb"], result["inference_kb_per_second"],
                         result["peak_gpu_memory_gb"]))
    print(serialized, flush=True)
    # Keep stdout valid JSON for callers that already parse it.
    print(format_summary(result), file=sys.stderr, flush=True)
    print(f"Results saved to: {experiment_dir}", file=sys.stderr, flush=True)
    return result


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, RuntimeError, FloatingPointError) as error:
        raise SystemExit(f"Evaluation failed: {error}") from error

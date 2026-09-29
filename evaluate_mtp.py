"""Compare baseline and MTP-4 byte probabilities on one raw validation file."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evaluate import _cuda_device, _nonnegative_int, _positive_int
from mamba3_byte_compress.config import FORMAL_TRAINING_CONFIG
from mamba3_byte_compress.data import ByteEvaluationDataset
from mamba3_byte_compress.experiments import (
    PROJECT_ROOT,
    TEST_EXPERIMENTS_ROOT,
    create_experiment_directory,
    validate_experiment_name,
)
from mamba3_byte_compress.metrics import evaluate_probability_model
from mamba3_byte_compress.model import BASELINE_MAMBA3_CONFIG, parameter_counts
from mamba3_byte_compress.mtp import evaluate_mtp_probability_model, load_mtp4_checkpoint
from mamba3_byte_compress.trainer import load_mamba3_checkpoint


SUMMARY_HEADERS = (
    "Model", "Parameters", "Head1 BPB", "Head2 BPB", "Head3 BPB", "Head4 BPB"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--mtp-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint",
                        default=str(PROJECT_ROOT / "checkpoints" / "best-3M.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--sequence-length", type=_positive_int, default=2048)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=0)
    parser.add_argument("--name", type=validate_experiment_name,
                        default="mtp4_comparison")
    return parser.parse_args(argv)


@torch.no_grad()
def evaluate_comparison(test_file, mtp_checkpoint, *, baseline_checkpoint=None,
                        device="cuda:0", batch_size=None, sequence_length=2048,
                        num_workers=0):
    test_file = Path(test_file).resolve()
    mtp_checkpoint = Path(mtp_checkpoint).resolve()
    baseline_checkpoint = Path(
        baseline_checkpoint or PROJECT_ROOT / "checkpoints" / "best-3M.pt"
    ).resolve()
    for label, path in (
        ("test", test_file), ("MTP checkpoint", mtp_checkpoint),
        ("baseline checkpoint", baseline_checkpoint),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} file: {path}")
    sequence_length = _positive_int(sequence_length)
    batch_size = None if batch_size is None else _positive_int(batch_size)
    num_workers = _nonnegative_int(num_workers)
    device = _cuda_device(device)

    baseline_model, baseline_metadata = load_mamba3_checkpoint(baseline_checkpoint)
    if baseline_metadata["model_name"] != BASELINE_MAMBA3_CONFIG["name"]:
        raise ValueError("Comparison requires the two-layer baseline checkpoint.")
    mtp_model, mtp_metadata = load_mtp4_checkpoint(mtp_checkpoint)
    training_config = mtp_metadata.get("training_config", {})
    if not isinstance(training_config, dict):
        raise ValueError("MTP checkpoint training config must be a dictionary.")
    batch_size = _positive_int(batch_size or training_config.get(
        "validation_batch_size", FORMAL_TRAINING_CONFIG["validation_batch_size"]
    ))
    dataset = ByteEvaluationDataset(test_file, sequence_length)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    baseline_params = parameter_counts(baseline_model)["total_trainable_params"]
    mtp_params = parameter_counts(mtp_model)["total_trainable_params"]
    del baseline_metadata
    baseline_model.to(device)
    baseline_metrics = evaluate_probability_model(
        baseline_model, loader, device, use_bf16=True
    )
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    mtp_model.to(device)
    mtp_metrics = evaluate_mtp_probability_model(
        mtp_model, loader, device, use_bf16=True
    )
    del mtp_model
    if baseline_metrics["predicted_bytes"] != mtp_metrics["head1_predicted_bytes"]:
        raise AssertionError("Baseline and Head1 target counts differ.")
    baseline_bpb = (
        baseline_metrics["nll_nats_predicted"]
        / (baseline_metrics["predicted_bytes"] * math.log(2.0))
        if baseline_metrics["predicted_bytes"] else None
    )
    training_elapsed_seconds = mtp_metadata.get("training_elapsed_seconds")
    peak_training_gpu_memory_gb = mtp_metadata.get("peak_training_gpu_memory_gb")
    training_time_source = "checkpoint"
    summary_path = mtp_checkpoint.parent.parent / "experiment_summary.json"
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as handle:
            training_summary = json.load(handle)
        final = training_summary.get("final_results", {})
        recorded_paths = {
            str(Path(final[key]).resolve())
            for key in ("best_checkpoint", "latest_checkpoint") if final.get(key)
        }
        if str(mtp_checkpoint) in recorded_paths:
            training_elapsed_seconds = final.get(
                "training_elapsed_seconds", training_elapsed_seconds
            )
            peak_training_gpu_memory_gb = final.get(
                "peak_training_gpu_memory_gb", peak_training_gpu_memory_gb
            )
            training_time_source = "experiment_summary"
    result = {
        "test_file": str(test_file),
        "num_bytes": int(dataset.raw_bytes),
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": str(device),
        "bpb_definition": "predicted-target negative log2 probability; no first-byte cost",
        "context_reset_each_block": True,
        "baseline": {
            "checkpoint": str(baseline_checkpoint),
            "parameters": baseline_params,
            "head1_bpb": baseline_bpb,
            "head1_predicted_bytes": baseline_metrics["predicted_bytes"],
        },
        "mtp4": {
            "checkpoint": str(mtp_checkpoint),
            "parameters": mtp_params,
            "global_update": mtp_metadata.get("global_update"),
            "training_elapsed_seconds": training_elapsed_seconds,
            "training_time_source": training_time_source,
            "peak_training_gpu_memory_gb": peak_training_gpu_memory_gb,
            "global_batch_size": training_config.get("global_batch_size"),
            "micro_batch_size": training_config.get("micro_batch_size"),
            "data_fingerprint": mtp_metadata.get("data_fingerprint"),
            **mtp_metrics,
        },
    }
    for value in (baseline_bpb, *(result["mtp4"][f"head{i}_bpb"] for i in range(1, 5))):
        if value is not None and not math.isfinite(value):
            raise FloatingPointError("Comparison produced non-finite BPB.")
    return result


def summary_rows(result):
    baseline = result["baseline"]
    mtp = result["mtp4"]
    return [
        ("Baseline", baseline["parameters"], baseline["head1_bpb"], None, None, None),
        ("MTP-4", mtp["parameters"], *(mtp[f"head{i}_bpb"] for i in range(1, 5))),
    ]


def main(argv=None):
    args = parse_args(argv)
    result = evaluate_comparison(
        args.test_file, args.mtp_checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
        device=args.device, batch_size=args.batch_size,
        sequence_length=args.sequence_length, num_workers=args.num_workers,
    )
    experiment_dir = create_experiment_directory(TEST_EXPERIMENTS_ROOT, args.name)
    result.update(
        experiment_name=experiment_dir.name,
        experiment_dir=str(experiment_dir),
    )
    (experiment_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (experiment_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUMMARY_HEADERS)
        writer.writerows(summary_rows(result))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    print(" | ".join(SUMMARY_HEADERS), file=sys.stderr)
    for row in summary_rows(result):
        print(" | ".join("" if value is None else str(value) for value in row), file=sys.stderr)
    print(f"Results saved to: {experiment_dir}", file=sys.stderr)
    return result


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, RuntimeError, FloatingPointError) as error:
        raise SystemExit(f"MTP evaluation failed: {error}") from error

"""Run reproducible forward and v5 codec benchmarks for both project models."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mamba3_byte_compress.codec import BLOCK_SIZE, sha256_file
from mamba3_byte_compress.experiments import (
    TEST_EXPERIMENTS_ROOT, create_experiment_directory, validate_experiment_name,
)
from mamba3_byte_compress.model import (
    analytic_parameter_counts, autoregressive_macs, parameter_counts,
)
from mamba3_byte_compress.trainer import load_mamba3_checkpoint
from scripts.benchmark_codec_batched import extract_slice


FULL_BYTES = 1_000_000_000
PREFLIGHT_BYTES = 10_000_000
DEFAULT_MODELS = (
    ("baseline_3m", ROOT / "checkpoints" / "best-3M.pt"),
    ("formal_4l", ROOT / "experiments" / "train" /
     "20260912_180146_mamba3_siso_4l" / "checkpoints" / "best.pt"),
)
SUMMARY_FIELDS = (
    "model", "actual_bpb", "codec_theoretical_bpb", "forward_bpb",
    "inference_kb_per_second", "encode_kb_per_second", "decode_kb_per_second",
    "inference_peak_gpu_memory_gb", "encode_peak_gpu_memory_gb",
    "decode_peak_gpu_memory_gb", "peak_gpu_memory_gb",
    "backbone_trainable_params", "total_trainable_params",
    "macs_per_predicted_byte", "gmacs_per_full_block", "total_codec_macs",
    "compressed_bytes", "inference_predicted_bytes", "predicted_bytes", "uniform_bytes",
    "inference_seconds", "inference_model_load_seconds", "inference_warmup_seconds",
    "encode_seconds", "decode_seconds",
    "encode_model_load_seconds", "decode_model_load_seconds",
    "encode_warmup_seconds", "decode_warmup_seconds",
    "inference_peak_reserved_gb", "encode_peak_reserved_gb", "decode_peak_reserved_gb",
    "inference_model_load_peak_gb", "inference_warmup_peak_gb",
    "source_sha256", "checkpoint_sha256", "gpu", "cuda", "torch", "mamba_ssm",
    "constriction", "checkpoint", "source",
)
SOURCE_FILES = (
    "evaluate.py", "mamba3_byte_compress/metrics.py", "mamba3_byte_compress/model.py",
    "mamba3_byte_compress/data.py", "mamba3_byte_compress/trainer.py",
    "mamba3_byte_compress/config.py", "mamba3_byte_compress/codec_step_adapter.py",
    "mamba3_byte_compress/codec.py", "mamba3_byte_compress/codec_batched_v4.py",
    "mamba3_byte_compress/codec_graph.py", "scripts/benchmark_codec_batched.py",
    "scripts/benchmark_enwik9_full.py",
)


def write_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False,
                  ensure_ascii=False)
        handle.write("\n")


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def complexity_for_checkpoint(path, predicted_bytes):
    path = Path(path).resolve(strict=True)
    model, metadata = load_mamba3_checkpoint(path)
    config = metadata["model_config"]
    actual = parameter_counts(model)
    analytic = analytic_parameter_counts(config)
    if actual != analytic:
        raise AssertionError(f"Checkpoint parameter counts disagree: {actual} != {analytic}")
    tensor_count = sum(tensor.numel() for tensor in metadata["model"].values())
    if tensor_count != actual["total_trainable_params"]:
        raise AssertionError(
            f"Checkpoint tensor count disagrees with loaded model: {tensor_count} != {actual}"
        )
    macs = autoregressive_macs(
        config, block_size=BLOCK_SIZE, predicted_bytes=predicted_bytes,
    )
    result = {
        "model_name": metadata["model_name"],
        "checkpoint_kind": metadata.get("checkpoint_kind"),
        "global_update": metadata.get("global_update"),
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "parameter_counts": actual,
        "checkpoint_parameter_tensors": tensor_count,
        "macs": macs,
    }
    del model, metadata
    return result


def validate_source(source, mode):
    source = Path(source).resolve(strict=True)
    size = source.stat().st_size
    if mode == "full" and size != FULL_BYTES:
        raise ValueError(f"Full enwik9 must be exactly {FULL_BYTES} bytes; found {size}.")
    if mode == "preflight" and size < PREFLIGHT_BYTES:
        raise ValueError(f"Preflight source must contain at least {PREFLIGHT_BYTES} bytes.")
    return source


def validate_rate_metrics(inference, encode, decode):
    checks = (
        (encode["actual_bpb"], 8 * encode["compressed_bytes"] / encode["num_bytes"],
         "actual_bpb"),
        (encode["encode_kb_per_second"],
         encode["num_bytes"] / 1000.0 / encode["encode_seconds"], "encode speed"),
        (decode["decode_kb_per_second"],
         decode["num_bytes"] / 1000.0 / decode["decode_seconds"], "decode speed"),
        (inference["inference_kb_per_second"],
         inference["predicted_bytes"] / 1000.0 / inference["inference_seconds"],
         "inference speed"),
    )
    for actual, expected, label in checks:
        if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-12):
            raise ValueError(f"Invalid {label}: {actual} != {expected}")


def aggregate_model(label, model_dir, raw, complexity, *, source=None,
                    source_sha256=None, codec_batch_size=1024):
    model_dir = Path(model_dir)
    inference = read_json(model_dir / "inference.json")
    codec = read_json(model_dir / "codec" / "results.json")
    encode, decode, verification = codec["encode"], codec["decode"], codec["verification"]
    validate_rate_metrics(inference, encode, decode)
    if not (verification.get("files_identical") and verification.get("sha256_matches")):
        raise ValueError(f"{label}: codec verification did not pass.")
    if verification["original_sha256"] != verification["decoded_sha256"]:
        raise ValueError(f"{label}: decoded checksum does not match input.")
    if not (encode.get("file_format") == decode.get("file_format") == "v5"):
        raise ValueError(f"{label}: codec result is not v5.")
    if not (encode.get("execution") == decode.get("execution") == "graph"
            and encode.get("graph_enabled") and decode.get("graph_enabled")):
        raise ValueError(f"{label}: CUDA Graph inference was not active.")
    if encode.get("batch_size") != codec_batch_size or decode.get("batch_size") != codec_batch_size:
        raise ValueError(f"{label}: codec batch size differs from requested value.")
    packed = next((model_dir / "codec").glob("*.m3rc"), None)
    decoded_file = model_dir / "codec" / "decoded.bin"
    if packed is None or packed.stat().st_size != encode["compressed_bytes"]:
        raise ValueError(f"{label}: compressed file size differs from recorded value.")
    if not decoded_file.is_file() or decoded_file.stat().st_size != encode["num_bytes"]:
        raise ValueError(f"{label}: decoded file size differs from recorded value.")
    if encode["num_bytes"] != Path(raw).stat().st_size or decode["num_bytes"] != encode["num_bytes"]:
        raise ValueError(f"{label}: byte counts do not match the experiment input.")
    if inference["num_bytes"] != encode["num_bytes"] or inference["predicted_bytes"] != encode["num_bytes"] - 1:
        raise ValueError(f"{label}: forward evaluation did not cover the complete input.")
    expected_blocks = math.ceil(encode["num_bytes"] / BLOCK_SIZE)
    if encode["uniform_bytes"] != expected_blocks:
        raise ValueError(f"{label}: unexpected number of uniform block seeds.")
    expected_predicted = encode["num_bytes"] - expected_blocks
    if encode["predicted_bytes"] != expected_predicted:
        raise ValueError(f"{label}: unexpected number of model-predicted bytes.")
    if complexity["macs"]["predicted_bytes"] != expected_predicted:
        raise ValueError(f"{label}: MAC byte count does not match codec byte count.")

    inference_peaks = [
        inference.get("model_load_peak_gpu_memory_gb"),
        inference.get("peak_gpu_memory_gb"), inference.get("warmup_peak_gpu_memory_gb")
    ]
    inference_peak = max(value for value in inference_peaks if value is not None)
    encode_peak = encode["run_peak_allocated_bytes"] / 1_000_000_000.0
    decode_peak = decode["run_peak_allocated_bytes"] / 1_000_000_000.0
    inference_reserved = max(
        value for value in (
            inference.get("peak_gpu_reserved_memory_gb"),
            inference.get("warmup_peak_gpu_reserved_memory_gb"),
            inference.get("model_load_peak_gpu_reserved_memory_gb"),
        ) if value is not None
    )
    encode_reserved = encode["run_peak_reserved_bytes"] / 1_000_000_000.0
    decode_reserved = decode["run_peak_reserved_bytes"] / 1_000_000_000.0
    overall_peak = max(inference_peak, encode_peak, decode_peak)
    params = complexity["parameter_counts"]
    macs = complexity["macs"]
    runtime = encode.get("runtime", {})
    row = {
        "model": label,
        "actual_bpb": encode["actual_bpb"],
        "codec_theoretical_bpb": encode["theoretical_nll_bpb"],
        "forward_bpb": inference["bpb"],
        "inference_kb_per_second": inference["inference_kb_per_second"],
        "encode_kb_per_second": encode["encode_kb_per_second"],
        "decode_kb_per_second": decode["decode_kb_per_second"],
        "inference_peak_gpu_memory_gb": inference_peak,
        "encode_peak_gpu_memory_gb": encode_peak,
        "decode_peak_gpu_memory_gb": decode_peak,
        "peak_gpu_memory_gb": overall_peak,
        "backbone_trainable_params": params["backbone_trainable_params"],
        "total_trainable_params": params["total_trainable_params"],
        "macs_per_predicted_byte": macs["macs_per_predicted_byte"],
        "gmacs_per_full_block": macs["gmacs_per_full_block"],
        "total_codec_macs": macs["total_macs"],
        "compressed_bytes": encode["compressed_bytes"],
        "inference_predicted_bytes": inference["predicted_bytes"],
        "predicted_bytes": encode["predicted_bytes"],
        "uniform_bytes": encode["uniform_bytes"],
        "inference_seconds": inference["inference_seconds"],
        "inference_model_load_seconds": inference.get("model_load_seconds"),
        "inference_warmup_seconds": inference.get("warmup_seconds"),
        "encode_seconds": encode["encode_seconds"],
        "decode_seconds": decode["decode_seconds"],
        "encode_model_load_seconds": encode.get("model_load_seconds"),
        "decode_model_load_seconds": decode.get("model_load_seconds"),
        "encode_warmup_seconds": encode.get("warmup_seconds"),
        "decode_warmup_seconds": decode.get("warmup_seconds"),
        "inference_peak_reserved_gb": inference_reserved,
        "encode_peak_reserved_gb": encode_reserved,
        "decode_peak_reserved_gb": decode_reserved,
        "inference_model_load_peak_gb": inference.get("model_load_peak_gpu_memory_gb"),
        "inference_warmup_peak_gb": inference.get("warmup_peak_gpu_memory_gb"),
        "source_sha256": source_sha256,
        "checkpoint_sha256": complexity["checkpoint_sha256"],
        "gpu": runtime.get("gpu"),
        "cuda": runtime.get("cuda"),
        "torch": runtime.get("torch"),
        "mamba_ssm": runtime.get("mamba_ssm"),
        "constriction": runtime.get("constriction"),
        "checkpoint": complexity["checkpoint"],
        "source": str(source) if source is not None else None,
    }
    return {"summary": row, "complexity": complexity, "inference": inference,
            "encode": encode, "decode": decode, "verification": verification}


def write_summary(output, models):
    rows = [value["summary"] for value in models.values()]
    if not (output / "summary.csv").exists():
        with (output / "summary.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# 完整 enwik9 双模型编解码实验", "",
        "实际 BPB 由完整压缩文件大小计算。推理速度为预热后的整块前向速度；编码和解码速度为端到端速度。", "",
        "| 模型 | 实际 BPB | 理论 BPB | 前向 BPB | 推理 KB/s | 编码 KB/s | 解码 KB/s | 峰值显存 GB | 参数量 | MAC/预测字节 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['actual_bpb']:.6f} | "
            f"{row['codec_theoretical_bpb']:.6f} | {row['forward_bpb']:.6f} | "
            f"{row['inference_kb_per_second']:.3f} | {row['encode_kb_per_second']:.3f} | "
            f"{row['decode_kb_per_second']:.3f} | {row['peak_gpu_memory_gb']:.6f} | "
            f"{row['total_trainable_params']} | {row['macs_per_predicted_byte']} |"
        )
    lines.extend([
        "", "MACs 包含模型线性投影、输出头、循环状态衰减、两次状态注入和一次状态读出；其他逐元素运算与区间编码不计入。",
        "峰值显存为 PyTorch 分配量，不包含 CUDA 上下文、第三方库直接分配量和其他进程。", "",
    ])
    if not (output / "report.md").exists():
        with (output / "report.md").open("x", encoding="utf-8") as handle:
            handle.write("\n".join(lines))


def run_inference_stage(args):
    import evaluate

    result = evaluate.evaluate_file(
        args.raw, args.checkpoint, device=args.device,
        batch_size=args.inference_batch_size, sequence_length=2048,
        num_workers=args.num_workers, warmup_batches=2,
    )
    result.update(process_id=os.getpid(), python=platform.python_version(),
                  stage="forward_inference")
    write_json(args.stage_output, result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


def run_logged(command, log_path):
    started = time.perf_counter()
    with Path(log_path).open("x", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - started


def next_log_path(directory, stem):
    directory = Path(directory)
    path = directory / f"{stem}.log"
    if not path.exists():
        return path
    attempt = 1
    while (directory / f"{stem}.resume{attempt}.log").exists():
        attempt += 1
    return directory / f"{stem}.resume{attempt}.log"


def archive_interrupted_json(path):
    """Move a partial stage record aside without deleting it."""
    path = Path(path)
    archive = path.parent / "incomplete"
    archive.mkdir(exist_ok=True)
    attempt = 1
    while (archive / f"{path.stem}.attempt{attempt}{path.suffix}").exists():
        attempt += 1
    target = archive / f"{path.stem}.attempt{attempt}{path.suffix}"
    path.rename(target)
    return target


def completed_inference(path, *, sample_bytes, checkpoint, batch_size):
    path = Path(path)
    if not path.exists():
        return False
    try:
        result = read_json(path)
    except json.JSONDecodeError:
        archive_interrupted_json(path)
        return False
    required = ("num_bytes", "checkpoint", "batch_size", "sequence_length",
                "bpb", "inference_seconds", "inference_kb_per_second",
                "warmup_batches_completed", "peak_gpu_memory_gb")
    if not isinstance(result, dict) or any(key not in result for key in required):
        archive_interrupted_json(path)
        return False
    if (result["num_bytes"] != sample_bytes or result["checkpoint"] != str(checkpoint)
            or result["batch_size"] != batch_size or result["sequence_length"] != 2048
            or result["warmup_batches_completed"] != 2):
        raise ValueError("Saved inference stage has a different data or model configuration.")
    if not all(math.isfinite(result[key]) for key in
               ("bpb", "inference_seconds", "inference_kb_per_second",
                "peak_gpu_memory_gb")):
        raise ValueError("Saved inference stage has non-finite metrics.")
    return True


def run_experiment(args):
    mode = args.mode
    source = validate_source(args.source, mode)
    sample_bytes = FULL_BYTES if mode == "full" else PREFLIGHT_BYTES
    name = validate_experiment_name(args.name or f"enwik9_{mode}")
    if args.resume_dir:
        output = Path(args.resume_dir).resolve(strict=True)
        manifest = read_json(output / "run_manifest.json")
        if manifest["mode"] != mode or manifest["sample_bytes"] != sample_bytes:
            raise ValueError("Resume directory belongs to a different experiment configuration.")
    else:
        output = create_experiment_directory(TEST_EXPERIMENTS_ROOT, name)
        manifest = None

    if mode == "full":
        free = shutil.disk_usage(output).free
        if not args.resume_dir and free < 5_000_000_000:
            raise OSError(f"At least 5 GB free space is required; found {free} bytes.")
        raw = source
    else:
        raw = output / "enwik9_first_10mb.bin"
        if raw.exists():
            if raw.stat().st_size != sample_bytes:
                raise ValueError("Existing preflight sample has the wrong size.")
        else:
            extract_slice(source, raw, 0, sample_bytes)

    source_digest = sha256_file(source)
    input_digest = source_digest if mode == "full" else sha256_file(raw)
    blocks = math.ceil(sample_bytes / BLOCK_SIZE)
    predicted_bytes = sample_bytes - blocks
    checkpoints = (
        ("baseline_3m", Path(args.baseline_checkpoint).resolve(strict=True)),
        ("formal_4l", Path(args.formal_checkpoint).resolve(strict=True)),
    )
    complexities = {
        label: complexity_for_checkpoint(checkpoint, predicted_bytes)
        for label, checkpoint in checkpoints
    }
    current_manifest = {
        "schema_version": 1,
        "mode": mode,
        "sample_bytes": sample_bytes,
        "source": str(source),
        "experiment_input": str(raw),
        "source_sha256": source_digest,
        "experiment_input_sha256": input_digest,
        "device": args.device,
        "sequence_length": 2048,
        "inference_batch_size": args.inference_batch_size,
        "inference_warmup_batches": 2,
        "codec_batch_size": args.codec_batch_size,
        "codec_format": "v5",
        "codec_execution": "graph",
        "models": complexities,
        "source_sha256_by_file": {
            name: sha256_file(ROOT / name) for name in SOURCE_FILES
        },
    }
    if manifest is None:
        write_json(output / "run_manifest.json", current_manifest)
    elif manifest != current_manifest:
        raise ValueError("Current files or arguments do not match the saved run manifest.")

    for label, checkpoint in checkpoints:
        model_dir = output / label
        model_dir.mkdir(exist_ok=True)
        complexity_path = model_dir / "complexity.json"
        if complexity_path.exists():
            try:
                existing_complexity = read_json(complexity_path)
            except json.JSONDecodeError:
                archive_interrupted_json(complexity_path)
                existing_complexity = None
            if existing_complexity is not None and existing_complexity != complexities[label]:
                raise ValueError(f"{label}: saved complexity result changed.")
        if not complexity_path.exists():
            write_json(complexity_path, complexities[label])
        inference_path = model_dir / "inference.json"
        if not completed_inference(
                inference_path, sample_bytes=sample_bytes, checkpoint=checkpoint,
                batch_size=args.inference_batch_size):
            command = [
                sys.executable, str(Path(__file__).resolve()), "--internal-inference",
                "--raw", str(raw), "--checkpoint", str(checkpoint),
                "--stage-output", str(inference_path), "--device", args.device,
                "--inference-batch-size", str(args.inference_batch_size),
                "--num-workers", str(args.num_workers),
            ]
            seconds = run_logged(command, next_log_path(model_dir, "inference"))
            print(f"{label}: forward inference complete in {seconds:.1f}s", flush=True)
        else:
            print(f"{label}: forward inference already complete", flush=True)

        codec_dir = model_dir / "codec"
        command = [
            sys.executable, str(ROOT / "scripts" / "benchmark_codec_batched.py"),
            "--output", str(codec_dir), "--source", str(raw),
            "--checkpoint", str(checkpoint), "--device", args.device,
            "--batch-size", str(args.codec_batch_size), "--sample-bytes", str(sample_bytes),
            "--format", "v5", "--execution", "graph", "--direct-source",
        ]
        if codec_dir.exists():
            command.append("--resume")
        seconds = run_logged(command, next_log_path(model_dir, "codec_run"))
        print(f"{label}: codec complete in {seconds:.1f}s", flush=True)

    if sha256_file(raw) != input_digest or sha256_file(source) != source_digest:
        raise ValueError("Experiment input or source changed during the run.")

    models = {
        label: aggregate_model(label, output / label, raw, complexities[label],
                               source=source, source_sha256=source_digest,
                               codec_batch_size=args.codec_batch_size)
        for label, _ in checkpoints
    }
    acceptance = {
        "both_models_complete": set(models) == {"baseline_3m", "formal_4l"},
        "source_sha256": source_digest,
        "all_roundtrips_verified": all(
            model["verification"]["files_identical"]
            and model["verification"]["sha256_matches"] for model in models.values()
        ),
        "all_metrics_finite": all(
            math.isfinite(value)
            for model in models.values() for value in model["summary"].values()
            if isinstance(value, float)
        ),
    }
    if not all(value for key, value in acceptance.items() if key != "source_sha256"):
        raise ValueError(f"Experiment acceptance failed: {acceptance}")
    results_path = output / "results.json"
    if not results_path.exists():
        write_json(results_path, {"manifest": current_manifest, "models": models,
                                  "acceptance": acceptance})
    if not (output / "summary.csv").exists() or not (output / "report.md").exists():
        write_summary(output, models)
    print(json.dumps({"experiment_dir": str(output), "acceptance": acceptance},
                     indent=2, ensure_ascii=False), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("preflight", "full"))
    parser.add_argument("--source", type=Path, default=ROOT / "datasets" / "raw" / "enwik9")
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_MODELS[0][1])
    parser.add_argument("--formal-checkpoint", type=Path, default=DEFAULT_MODELS[1][1])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--codec-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--name")
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--internal-inference", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--raw", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stage-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.inference_batch_size <= 0 or args.codec_batch_size <= 0 or args.num_workers < 0:
        parser.error("Batch sizes must be positive and num-workers must be nonnegative.")
    if args.internal_inference:
        if not all((args.raw, args.checkpoint, args.stage_output)):
            parser.error("Internal inference requires raw, checkpoint and stage-output.")
    elif args.mode is None:
        parser.error("mode is required")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.internal_inference:
        return run_inference_stage(args)
    return run_experiment(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, RuntimeError, FloatingPointError,
            subprocess.CalledProcessError) as error:
        raise SystemExit(f"Full enwik9 benchmark failed: {error}") from error

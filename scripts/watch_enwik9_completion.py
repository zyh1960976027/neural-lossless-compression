"""Wait for an enwik9 experiment and independently verify its final artifacts."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time


MODELS = ("baseline_3m", "formal_4l")
CHUNK_BYTES = 4 * 1024 * 1024


def process_alive(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1][0]
        if state in ("Z", "X"):
            return False
    except (FileNotFoundError, PermissionError, IndexError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def compare_and_hash(source, restored):
    source_hash = hashlib.sha256()
    restored_hash = hashlib.sha256()
    compared = 0
    with source.open("rb") as original, restored.open("rb") as decoded:
        while True:
            left = original.read(CHUNK_BYTES)
            right = decoded.read(CHUNK_BYTES)
            if left != right:
                raise ValueError(f"Byte mismatch at or after offset {compared}: {restored}")
            if not left:
                break
            source_hash.update(left)
            restored_hash.update(right)
            compared += len(left)
    return compared, source_hash.hexdigest(), restored_hash.hexdigest()


def check_close(actual, expected, label):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-10):
        raise ValueError(f"{label}: {actual} != {expected}")


def verify_run(directory):
    directory = Path(directory).resolve(strict=True)
    with (directory / "run_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    with (directory / "results.json").open(encoding="utf-8") as handle:
        results = json.load(handle)
    source = Path(manifest["experiment_input"]).resolve(strict=True)
    source_bytes = source.stat().st_size
    if source_bytes != manifest["sample_bytes"]:
        raise ValueError("Experiment input size changed")
    if set(results["models"]) != set(MODELS):
        raise ValueError("Both model results are required")
    acceptance = results["acceptance"]
    if not all(acceptance[key] for key in (
            "both_models_complete", "all_roundtrips_verified", "all_metrics_finite")):
        raise ValueError(f"Experiment acceptance failed: {acceptance}")

    checked = {}
    for label in MODELS:
        model = results["models"][label]
        row = model["summary"]
        encode, decode, inference = model["encode"], model["decode"], model["inference"]
        packed = list((directory / label / "codec").glob("*.m3rc"))
        if len(packed) != 1:
            raise ValueError(f"{label}: expected exactly one compressed file")
        compressed_bytes = packed[0].stat().st_size
        restored = directory / label / "codec" / "decoded.bin"
        if restored.stat().st_size != source_bytes:
            raise ValueError(f"{label}: restored file has the wrong size")
        compared, original_sha, restored_sha = compare_and_hash(source, restored)
        if (compared != source_bytes or original_sha != restored_sha
                or original_sha != manifest["experiment_input_sha256"]):
            raise ValueError(f"{label}: SHA256 or byte comparison failed")
        if compressed_bytes != row["compressed_bytes"]:
            raise ValueError(f"{label}: compressed file size changed")
        check_close(row["actual_bpb"], 8 * compressed_bytes / source_bytes,
                    f"{label} actual BPB")
        check_close(row["inference_kb_per_second"],
                    inference["predicted_bytes"] / 1000 / inference["inference_seconds"],
                    f"{label} inference speed")
        check_close(row["encode_kb_per_second"],
                    source_bytes / 1000 / encode["encode_seconds"],
                    f"{label} encode speed")
        check_close(row["decode_kb_per_second"],
                    source_bytes / 1000 / decode["decode_seconds"],
                    f"{label} decode speed")
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{label}: non-finite {key}")
        checked[label] = {
            "compressed_bytes": compressed_bytes,
            "actual_bpb": row["actual_bpb"],
            "inference_kb_per_second": row["inference_kb_per_second"],
            "encode_kb_per_second": row["encode_kb_per_second"],
            "decode_kb_per_second": row["decode_kb_per_second"],
            "peak_gpu_memory_gb": row["peak_gpu_memory_gb"],
            "restored_bytes": compared,
            "sha256": original_sha,
        }
    return {"status": "verified", "directory": str(directory), "models": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pid", type=int, help="Benchmark process to monitor")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    directory = args.directory.resolve(strict=True)
    required = ("results.json", "summary.csv", "report.md")
    while args.pid is not None and not all((directory / name).exists() for name in required):
        if not process_alive(args.pid):
            result = {"status": "failed", "directory": str(directory),
                      "error": "Benchmark exited before writing all final artifacts"}
            break
        time.sleep(args.interval)
    else:
        try:
            result = verify_run(directory)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            result = {"status": "failed", "directory": str(directory),
                      "error": str(error)}
    result["checked_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    with (directory / "completion_check.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

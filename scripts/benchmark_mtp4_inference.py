"""Reproducible MTP-4 sequence/CUDA Graph codec comparison."""

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODES = ("sequence", "graph")


def read_result(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def validate_result(result, *, execution, batch_size, sample_bytes, checkpoint_sha256):
    verification = result["verification"]
    if verification["sample_bytes"] != sample_bytes:
        raise ValueError("Saved run uses a different sample size.")
    if result["encode"]["batch_size"] != batch_size:
        raise ValueError("Saved run uses a different batch size.")
    if result["encode"]["execution"] != execution:
        raise ValueError("Saved run uses a different execution mode.")
    if result["encode"]["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Saved run uses a different checkpoint.")
    if not all(verification[key] for key in
               ("files_identical", "sha256_matches", "independent_processes")):
        raise ValueError("Saved run did not pass independent roundtrip verification.")


def run_case(args, name, execution, batch_size, sample_bytes, *, cold=False):
    from mamba3_byte_compress.codec import sha256_file

    output = args.output / name
    result_path = output / "results.json"
    checkpoint_sha256 = sha256_file(args.checkpoint)
    existing = read_result(result_path)
    if existing is not None:
        validate_result(
            existing, execution=execution, batch_size=batch_size,
            sample_bytes=sample_bytes, checkpoint_sha256=checkpoint_sha256,
        )
        return existing
    command = [
        sys.executable, str(ROOT / "scripts" / "benchmark_codec_batched.py"),
        "--output", str(output), "--source", str(args.source),
        "--checkpoint", str(args.checkpoint), "--sample-bytes", str(sample_bytes),
        "--source-offset", "0", "--batch-size", str(batch_size),
        "--format", "v6", "--execution", execution,
    ]
    environment = os.environ.copy()
    if cold:
        command.append("--cold")
        environment["TRITON_CACHE_DIR"] = str(output / "triton-cache")
    log_path = args.output / f"{name}.log"
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run(
            command, cwd=ROOT, env=environment, stdout=log,
            stderr=subprocess.STDOUT, check=True,
        )
    result = json.loads(result_path.read_text())
    validate_result(
        result, execution=execution, batch_size=batch_size,
        sample_bytes=sample_bytes, checkpoint_sha256=checkpoint_sha256,
    )
    print(f"{name} complete", flush=True)
    return result


def summarize_runs(runs, sample_bytes):
    summary = {}
    for execution, values in runs.items():
        hashes = {value["verification"]["compressed_sha256"] for value in values}
        if len(hashes) != 1:
            raise AssertionError(f"{execution}: repeated encodes are not deterministic.")
        summary[execution] = {}
        for operation in ("encode", "decode"):
            seconds = [value[operation][f"{operation}_seconds"] for value in values]
            summary[execution][operation] = {
                "repeats": len(seconds),
                "median_seconds": statistics.median(seconds),
                "min_seconds": min(seconds),
                "max_seconds": max(seconds),
                "median_kb_per_second": sample_bytes / 1000 / statistics.median(seconds),
                "peak_allocated_bytes": max(
                    value[operation]["run_peak_allocated_bytes"] for value in values
                ),
                "peak_reserved_bytes": max(
                    value[operation]["run_peak_reserved_bytes"] for value in values
                ),
                "compressed_bytes": values[0][operation]["compressed_bytes"],
                "model_load_seconds": statistics.median(
                    value[operation]["model_load_seconds"] for value in values
                ),
                "warmup_seconds": statistics.median(
                    value[operation]["warmup_seconds"] for value in values
                ),
                "graph_prepare_seconds": statistics.median(
                    value[operation].get("graph_prepare_seconds", 0.0) for value in values
                ),
            }
    return summary


def select_best(screen):
    best = {}
    for execution in MODES:
        candidates = [
            (batch_size,
             result["encode"]["encode_seconds"] + result["decode"]["decode_seconds"])
            for (mode, batch_size), result in screen.items() if mode == execution
        ]
        if not candidates:
            raise RuntimeError(f"No successful {execution} screening run.")
        best[execution] = min(candidates, key=lambda item: item[1])[0]
    common = max(
        batch_size for batch_size in (128, 512, 1024)
        if all((execution, batch_size) in screen for execution in MODES)
    )
    return best, common


def write_report(args, screen, best, common, formal, cold):
    from mamba3_byte_compress.codec import sha256_file

    common_summary = summarize_runs(
        {mode: formal[(mode, common)] for mode in MODES}, args.formal_bytes
    )
    stable = all(
        common_summary["graph"][operation]["max_seconds"]
        < common_summary["sequence"][operation]["min_seconds"]
        for operation in ("encode", "decode")
    )
    bitstream_identical = (
        formal[("sequence", common)][0]["verification"]["compressed_sha256"]
        == formal[("graph", common)][0]["verification"]["compressed_sha256"]
    )
    if not bitstream_identical:
        raise AssertionError("Sequence and Graph v6 bitstreams differ.")
    microbenchmark = json.loads(
        (args.output / "short_sequence_microbenchmark.json").read_text()
    )

    screen_rows = []
    for (mode, batch_size), result in sorted(screen.items(), key=lambda item: (item[0][1], item[0][0])):
        screen_rows.append({
            "execution": mode, "batch_size": batch_size,
            "encode_kb_per_second": result["encode"]["encode_kb_per_second"],
            "decode_kb_per_second": result["decode"]["decode_kb_per_second"],
            "compressed_bytes": result["encode"]["compressed_bytes"],
            "peak_allocated_bytes": max(
                result["encode"]["run_peak_allocated_bytes"],
                result["decode"]["run_peak_allocated_bytes"],
            ),
        })
    with (args.output / "screening.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(screen_rows[0]))
        writer.writeheader()
        writer.writerows(screen_rows)

    comparison_rows = []
    for mode in MODES:
        for operation in ("encode", "decode"):
            row = {"execution": mode, "operation": operation, "batch_size": common}
            row.update(common_summary[mode][operation])
            comparison_rows.append(row)
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    runtime = formal[("graph", common)][0]["encode"]["runtime"]
    manifest = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).splitlines(),
        "source": str(args.source),
        "source_sha256": sha256_file(args.source),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "screen_bytes": args.screen_bytes,
        "formal_bytes": args.formal_bytes,
        "screen_batches": [128, 512, 1024],
        "best_batch_sizes": best,
        "common_batch_size": common,
        "runtime": runtime,
        "source_sha256_by_file": {
            name: sha256_file(ROOT / name)
            for name in (
                "codec.py", "mamba3_byte_compress/codec.py",
                "mamba3_byte_compress/codec_graph.py",
                "mamba3_byte_compress/codec_mtp4.py",
                "mamba3_byte_compress/codec_sequence_adapter.py",
                "scripts/benchmark_codec_batched.py",
                "scripts/benchmark_mtp4_inference.py",
                "scripts/profile_mtp4_short_sequence.py",
                "tests/test_codec_mtp4.py",
            )
        },
    }
    result = {
        "stable_speedup": stable,
        "bitstream_identical": bitstream_identical,
        "common_batch_summary": common_summary,
        "best_batch_sizes": best,
        "common_batch_size": common,
        "cold": cold,
        "short_sequence_microbenchmark": microbenchmark,
        "manifest": manifest,
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )

    enc = common_summary
    encode_speedup = enc["sequence"]["encode"]["median_seconds"] / enc["graph"]["encode"]["median_seconds"]
    decode_speedup = enc["sequence"]["decode"]["median_seconds"] / enc["graph"]["decode"]["median_seconds"]
    lines = [
        "# MTP-4 CUDA Graph 编解码性能实验", "",
        f"编码从 {enc['sequence']['encode']['median_kb_per_second']:.2f} KB/s 提升到 "
        f"{enc['graph']['encode']['median_kb_per_second']:.2f} KB/s，{encode_speedup:.3f} 倍；"
        f"解码从 {enc['sequence']['decode']['median_kb_per_second']:.2f} KB/s 提升到 "
        f"{enc['graph']['decode']['median_kb_per_second']:.2f} KB/s，{decode_speedup:.3f} 倍。"
        f"压缩大小均为 {enc['graph']['encode']['compressed_bytes']:,} 字节，独立进程逐字节、SHA256 与重复码流检查通过。", "",
        f"设备为 {runtime['gpu']}，PyTorch {runtime['torch']}，CUDA {runtime['cuda']}；"
        f"固定使用 enwik9 前 {args.formal_bytes:,} 字节、相同权重、BF16 和 batch {common}。", "",
        "| 路径 | 操作 | 中位吞吐 KB/s | 中位时间 s | 范围 s | 峰值显存 GB |", 
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for mode, label in (("sequence", "原序列路径"), ("graph", "CUDA Graph")):
        for operation, operation_label in (("encode", "编码"), ("decode", "解码")):
            row = enc[mode][operation]
            lines.append(
                f"| {label} | {operation_label} | {row['median_kb_per_second']:.2f} | "
                f"{row['median_seconds']:.3f} | {row['min_seconds']:.3f}–{row['max_seconds']:.3f} | "
                f"{row['peak_allocated_bytes'] / 1e9:.3f} |"
            )
    lines.extend([
        "", f"稳定收益判定：{'通过' if stable else '未通过'}。判定要求 Graph 三次测试的最慢耗时仍小于基线最快耗时。", "",
        "## Batch 短测", "",
        "| 路径 | Batch | 编码 KB/s | 解码 KB/s | 峰值显存 GB |", 
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in screen_rows:
        lines.append(
            f"| {row['execution']} | {row['batch_size']} | {row['encode_kb_per_second']:.2f} | "
            f"{row['decode_kb_per_second']:.2f} | {row['peak_allocated_bytes'] / 1e9:.3f} |"
        )
    lines.extend(["", "## 首次运行", ""])
    for mode in MODES:
        value = cold[mode]
        for operation in ("encode", "decode"):
            row = value[operation]
            lines.append(
                f"- {mode} {operation}：完整进程 {row['process_seconds']:.3f} s，"
                f"模型加载 {row['model_load_seconds']:.3f} s，正式编解码 {row[f'{operation}_seconds']:.3f} s，"
                f"图准备 {row.get('graph_prepare_seconds', 0.0):.3f} s。"
            )
    lines.extend([
        "", "## 四字节短序列内核", "",
        f"batch {microbenchmark['batch_size']} 下，现有长度 4 SISO 路径为 "
        f"{microbenchmark['sequence4_graph_ms']:.3f} ms/组；把四次完整单步递推捕获到一张 Graph 为 "
        f"{microbenchmark['four_step_graph_ms']:.3f} ms/组，后者慢 "
        f"{(microbenchmark['four_step_over_sequence_ratio'] - 1) * 100:.1f}%。Profiler 中 SISO 内核仍是最大单项，"
        "但现有实现已经在每层的一次调用内完成四个位置的状态更新。", "",
        "CUDA Graph 只改变调度和缓冲区复用，模型权重、四头语义、Mamba 循环状态更新及 Range Coding 均未改变。"
        "现有 SISO 内核已经在每层的一次调用内处理四个位置；16 是矩阵计算 tile，未按四倍浪费处理。"
        "进一步改写该内核会增加数值兼容和维护风险，本次未保留没有端到端证据支持的替代内核。", "",
        "## 复现", "", "```bash",
        "conda activate mamba3",
        f"python scripts/benchmark_mtp4_inference.py --output {shlex.quote(str(args.output))}",
        "```", "",
    ])
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"stable_speedup": stable, "summary": common_summary}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/raw/enwik9")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "experiments/train/20260918_175712_mamba3_siso_3m_mtp4/checkpoints/best.pt",
    )
    parser.add_argument("--screen-bytes", type=int, default=3_000_000)
    parser.add_argument("--formal-bytes", type=int, default=10_000_000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.source = args.source.resolve(strict=True)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.repeats < 3:
        parser.error("--repeats must be at least 3.")

    screen = {}
    for batch_size in (128, 512, 1024):
        for execution in MODES:
            name = f"screen_{execution}_b{batch_size}"
            try:
                screen[(execution, batch_size)] = run_case(
                    args, name, execution, batch_size, args.screen_bytes
                )
            except subprocess.CalledProcessError as error:
                failure = {
                    "execution": execution, "batch_size": batch_size,
                    "returncode": error.returncode, "log": f"{name}.log",
                }
                (args.output / f"{name}.failure.json").write_text(
                    json.dumps(failure, indent=2) + "\n"
                )
    best, common = select_best(screen)

    cases = {(execution, common) for execution in MODES}
    cases.update((execution, best[execution]) for execution in MODES)
    formal = {}
    for execution, batch_size in sorted(cases):
        label = "baseline" if execution == "sequence" else "graph"
        formal[(execution, batch_size)] = [
            run_case(
                args, f"{label}_b{batch_size}_run{index}", execution,
                batch_size, args.formal_bytes,
            )
            for index in range(1, args.repeats + 1)
        ]
    cold = {
        execution: run_case(
            args, f"cold_fresh_{execution}_b{common}", execution, common,
            args.formal_bytes, cold=True,
        )
        for execution in MODES
    }
    microbenchmark = args.output / "short_sequence_microbenchmark.json"
    if not microbenchmark.exists():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "profile_mtp4_short_sequence.py"),
                "--output", str(microbenchmark),
                "--checkpoint", str(args.checkpoint),
                "--batch-size", str(common),
            ],
            cwd=ROOT,
            check=True,
        )
    write_report(args, screen, best, common, formal, cold)


if __name__ == "__main__":
    main()

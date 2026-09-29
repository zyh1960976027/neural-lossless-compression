"""Summarize instrumented MTP-4 codec runs and a CUDA kernel profile."""

import argparse
import csv
import json
from pathlib import Path


PHASE_LABELS = {
    "model_load": "模型加载",
    "warmup": "预热与执行准备",
    "codec": "正式编解码",
    "process_other": "进程启动、导入与收尾",
    "input_read_hash_pack": "读取、SHA256 与批数据整理",
    "container_read_and_allocate": "容器读取与输出数组分配",
    "state_reset": "循环状态清零",
    "predictor": "预测器（H2D、GPU 推理、D2H 与概率检查）",
    "probability_cpu": "CPU 概率选择与统计",
    "range_coding": "Range Coding（区间编码）",
    "container_write": "帧封装与压缩文件写入",
    "output_write_hash": "恢复文件写入与 SHA256",
    "other": "正式编解码其他开销",
}


def kernel_label(name):
    if name == "mamba3_siso_fwd_kernel":
        return "Mamba-3 SISO 四位置递推"
    if name == "memcpy128":
        return "GPU 内部 128-bit 数据搬运"
    if "FillFunctor<c10::BFloat16>" in name:
        return "BF16 张量填充"
    if "FillFunctor<float>" in name:
        return "FP32 张量填充"
    if "cutlass_75_tensorop" in name:
        return "CUTLASS BF16 矩阵乘"
    if "cutlass_80_tensorop" in name and "relu" in name:
        return "CUTLASS BF16 矩阵乘与 ReLU"
    if name.startswith("ampere_bf16"):
        return "Ampere BF16 矩阵乘"
    if "direct_copy_kernel_cuda" in name and "BFloat16" in name:
        return "BF16 直接复制"
    if "bfloat16_copy_kernel_cuda" in name:
        return "FP32/BF16 类型转换"
    if "BinaryFunctor<float" in name and "MulFunctor" in name:
        return "FP32 逐元素乘"
    if "CUDAFunctor_add<float>" in name:
        return "FP32 逐元素加"
    if name == "angle_dt_fwd_kernel":
        return "角度与时间步更新"
    if "pow_tensor_scalar" in name:
        return "FP32 逐元素幂"
    if "BinaryFunctor<c10::BFloat16" in name and "MulFunctor" in name:
        return "BF16 逐元素乘"
    if "direct_copy_kernel_cuda" in name:
        return "GPU 直接复制"
    return name


def load_run(path):
    result = json.loads((path / "results.json").read_text())
    if not result["verification"]["sha256_matches"]:
        raise ValueError(f"Unverified run: {path}")
    for operation in ("encode", "decode"):
        if "phase_seconds" not in result[operation]:
            raise ValueError(f"Run lacks --phase-timing data: {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--kernel-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-replays", type=int, default=8)
    args = parser.parse_args()
    if args.profile_replays <= 0:
        parser.error("--profile-replays must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        "sequence": load_run(args.sequence_dir),
        "graph": load_run(args.graph_dir),
    }
    timing_rows = []
    summary = {"runs": {}, "kernel_profile": {}}
    for mode, run in runs.items():
        summary["runs"][mode] = {}
        for operation in ("encode", "decode"):
            result = run[operation]
            codec_seconds = result[f"{operation}_seconds"]
            process_seconds = result["process_seconds"]
            load_seconds = result["model_load_seconds"]
            warmup_seconds = result["warmup_seconds"]
            process_other = max(
                0.0, process_seconds - load_seconds - warmup_seconds - codec_seconds
            )
            process_phases = {
                "model_load": load_seconds,
                "warmup": warmup_seconds,
                "codec": codec_seconds,
                "process_other": process_other,
            }
            item = {
                "process_seconds": process_seconds,
                "process_phases": process_phases,
                "codec_seconds": codec_seconds,
                "codec_phases": result["phase_seconds"],
                "graph_prepare_seconds": result.get("graph_prepare_seconds", 0.0),
                "kb_per_second": result[f"{operation}_kb_per_second"],
                "compressed_bytes": result["compressed_bytes"],
            }
            summary["runs"][mode][operation] = item
            for phase, seconds in process_phases.items():
                timing_rows.append({
                    "mode": mode, "operation": operation, "level": "process",
                    "phase": phase, "label": PHASE_LABELS[phase],
                    "seconds": seconds,
                    "share_percent": 100.0 * seconds / process_seconds,
                })
            for phase, seconds in result["phase_seconds"].items():
                timing_rows.append({
                    "mode": mode, "operation": operation, "level": "codec",
                    "phase": phase, "label": PHASE_LABELS.get(phase, phase),
                    "seconds": seconds,
                    "share_percent": 100.0 * seconds / codec_seconds,
                })

    profile = json.loads(args.kernel_profile.read_text())
    events = profile["top_cuda_events_for_sequence_graph"]
    listed_us = sum(event["device_time_us"] for event in events)
    kernel_rows = []
    for rank, event in enumerate(events, start=1):
        kernel_rows.append({
            "rank": rank,
            "label": kernel_label(event["name"]),
            "raw_name": event["name"],
            "total_device_time_us": event["device_time_us"],
            "ms_per_four_byte_group": (
                event["device_time_us"] / args.profile_replays / 1000.0
            ),
            "launches_per_four_byte_group": event["count"] / args.profile_replays,
            "share_of_listed_device_time_percent": (
                100.0 * event["device_time_us"] / listed_us
            ),
        })
    summary["kernel_profile"] = {
        "batch_size": profile["batch_size"],
        "profile_replays": args.profile_replays,
        "sequence4_graph_ms": profile["sequence4_graph_ms"],
        "listed_device_time_us": listed_us,
        "events": kernel_rows,
        "note": "Percentages use the listed top CUDA events, not end-to-end wall time.",
    }

    json_path = args.output_dir / "end_to_end_timing.json"
    csv_path = args.output_dir / "end_to_end_timing.csv"
    kernel_csv_path = args.output_dir / "gpu_kernel_ranking.csv"
    report_path = args.output_dir / "timing_and_kernels.md"
    for path in (json_path, csv_path, kernel_csv_path, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with csv_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=timing_rows[0])
        writer.writeheader()
        writer.writerows(timing_rows)
    with kernel_csv_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=kernel_rows[0])
        writer.writeheader()
        writer.writerows(kernel_rows)

    lines = [
        "# MTP-4 端到端耗时分解与 GPU 内核排行",
        "",
        "数据为 enwik9 前 10,000,000 字节、batch 1024 的单次插桩运行。插桩仅用于定位，正式吞吐率仍以未插桩的三次中位数为准。",
        "",
        "## 完整进程耗时",
        "",
        "| 路径 | 操作 | 完整进程 s | 模型加载 s | 预热/准备 s | 其中 Graph 捕获 s | 正式编解码 s | 进程其他 s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    operation_labels = {"encode": "编码", "decode": "解码"}
    for mode in ("sequence", "graph"):
        for operation in ("encode", "decode"):
            item = summary["runs"][mode][operation]
            phases = item["process_phases"]
            lines.append(
                f"| {mode} | {operation_labels[operation]} | {item['process_seconds']:.3f} | "
                f"{phases['model_load']:.3f} | {phases['warmup']:.3f} | "
                f"{item['graph_prepare_seconds']:.3f} | {phases['codec']:.3f} | "
                f"{phases['process_other']:.3f} |"
            )
    lines += [
        "",
        "Graph 捕获时间包含在“预热/准备”中；正式编解码不包含模型加载与 Graph 捕获。",
        "",
        "## 正式编解码内部耗时",
        "",
        "| 路径 | 操作 | 阶段 | 秒 | 正式编解码占比 |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in timing_rows:
        if row["level"] == "codec":
            lines.append(
                f"| {row['mode']} | {operation_labels[row['operation']]} | "
                f"{row['label']} | {row['seconds']:.3f} | {row['share_percent']:.2f}% |"
            )
    lines += [
        "",
        "预测器阶段按端到端墙钟计时，包含输入复制、GPU 推理、概率复制回主机和有限值检查，因此可以与 CPU 区间编码等阶段直接相加。",
        "",
        "## 最耗时 GPU 内核",
        "",
        f"以下为 batch {profile['batch_size']} 的四字节 Graph，连续 {args.profile_replays} 次重放的 CUDA profiler 汇总；单组时间已除以重放次数。占比仅针对列出的前 15 个 CUDA 事件。",
        "",
        "| 排名 | 内核类别 | 每四字节组 ms | 每组启动次数 | 列表内占比 |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in kernel_rows:
        lines.append(
            f"| {row['rank']} | {row['label']} | "
            f"{row['ms_per_four_byte_group']:.3f} | "
            f"{row['launches_per_four_byte_group']:.1f} | "
            f"{row['share_of_listed_device_time_percent']:.2f}% |"
        )
    lines += [
        "",
        "原始完整 CUDA 内核名保存在 `gpu_kernel_ranking.csv`，便于用 Nsight 或 PyTorch profiler 交叉核对。",
        "",
    ]
    report_path.write_text("\n".join(lines))
    print(json.dumps({
        "json": str(json_path), "timing_csv": str(csv_path),
        "kernel_csv": str(kernel_csv_path), "report": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()

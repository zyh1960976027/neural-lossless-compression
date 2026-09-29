"""Validate and summarize the 100 MB, B=1024 codec bottleneck experiment."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mamba3_byte_compress.model import (
    BASELINE_MAMBA3_CONFIG,
    MTP4_MAMBA3_CONFIG,
    analytic_parameter_counts,
    autoregressive_macs,
    mtp4_sequence_macs,
)

SYSTEMS = {
    "autoregressive": {"label": "自回归 v5", "format": "v5"},
    "mtp4": {"label": "MTP-4 v6", "format": "v6"},
}
OPERATIONS = {"encode": "编码", "decode": "解码"}
PHASES = {
    "predictor": "预测器（上传、GPU 推理、回传）",
    "probability_cpu": "CPU 概率选择与统计",
    "range_coding": "区间编码（含概率量化）",
    "step_input": "逐位置输入整理",
    "state_reset": "循环状态重置",
    "input_read_hash_pack": "读取、校验和批数据整理",
    "container_read_and_allocate": "读取压缩帧并分配数组",
    "container_write": "压缩帧写入",
    "output_write_hash": "恢复文件写入与校验",
    "other": "其余开销",
}


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def median(values):
    return statistics.median(values)


def kernel_label(name):
    lower = name.lower()
    if "mamba3_siso_fwd_kernel" in lower or "mamba3_step" in lower:
        return "Mamba 状态更新"
    if "memcpy" in lower or "copy_kernel" in lower:
        return "数据搬运"
    if "fill" in lower or "zero" in lower:
        return "张量填充或清零"
    if "rotary" in lower or "angle_dt" in lower:
        return "位置与时间状态更新"
    if "gemm" in lower or "cutlass" in lower or "matmul" in lower:
        return "矩阵乘"
    return name[:100]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    summaries = {}
    phase_rows = []
    input_hashes = set()
    for system, config in SYSTEMS.items():
        run_paths = sorted(
            root.glob(f"{system}-baseline-*/results.json"),
            key=lambda path: int(path.parent.name.rsplit("-", 1)[1]),
        )
        if len(run_paths) not in (3, 5):
            raise AssertionError(f"{system}: expected three or five baseline runs")
        if [int(path.parent.name.rsplit("-", 1)[1]) for path in run_paths] != list(
            range(1, len(run_paths) + 1)
        ):
            raise AssertionError(f"{system}: baseline run numbers are not consecutive")
        runs = [read_json(path) for path in run_paths]
        phase_run = read_json(root / f"{system}-phase" / "results.json")
        gpu = read_json(root / f"{system}-gpu" / "profile.json")
        bitstreams = set()
        checkpoint_hashes = set()
        for run in [*runs, phase_run]:
            verification = run["verification"]
            if not all(verification[key] for key in (
                "files_identical", "sha256_matches", "independent_processes"
            )):
                raise AssertionError(f"{system}: roundtrip verification failed")
            if verification["sample_bytes"] != 100_000_000 or verification["source_offset"] != 0:
                raise AssertionError(f"{system}: input segment differs")
            input_hashes.add(verification["original_sha256"])
            for operation in OPERATIONS:
                result = run[operation]
                if (
                    result["batch_size"] != 1024
                    or result["batch_sizes"] != [1024] * 47 + [701]
                    or result["file_format"] != config["format"]
                    or result["execution"] != "graph"
                    or result["num_bytes"] != 100_000_000
                ):
                    raise AssertionError(f"{system} {operation}: codec settings differ")
                checkpoint_hashes.add(result["checkpoint_sha256"])
            bitstreams.add(verification["compressed_sha256"])
        if len(checkpoint_hashes) != 1 or gpu["checkpoint_sha256"] not in checkpoint_hashes:
            raise AssertionError(f"{system}: checkpoint mismatch")
        if len(bitstreams) != 1:
            raise AssertionError(f"{system}: repeated or instrumented bitstreams differ")
        if gpu["batch_size"] != 1024 or gpu["byte_positions"] != 128:
            raise AssertionError(f"{system}: GPU profile has different span")

        operations = {}
        for operation in OPERATIONS:
            key = f"{operation}_seconds"
            baseline_seconds = [run[operation][key] for run in runs]
            baseline_median = median(baseline_seconds)
            range_fraction = (
                (max(baseline_seconds) - min(baseline_seconds)) / baseline_median
            )
            timed = phase_run[operation]
            phase_seconds = timed.get("phase_seconds")
            if not phase_seconds:
                raise AssertionError(f"{system} {operation}: phase timing missing")
            phase_sum = sum(phase_seconds.values())
            if abs(phase_sum - timed[key]) > max(0.001, timed[key] * 0.001):
                raise AssertionError(f"{system} {operation}: phase times do not add up")
            overhead_fraction = (timed[key] - baseline_median) / baseline_median
            ordered_phases = sorted(
                phase_seconds.items(), key=lambda item: item[1], reverse=True
            )
            for phase, seconds in ordered_phases:
                phase_rows.append({
                    "system": system,
                    "operation": operation,
                    "phase": phase,
                    "label": PHASES.get(phase, phase),
                    "seconds": seconds,
                    "share_percent": 100.0 * seconds / timed[key],
                })
            operations[operation] = {
                "baseline_seconds": baseline_seconds,
                "baseline_median_seconds": baseline_median,
                "baseline_range_fraction": range_fraction,
                "baseline_median_kb_per_second": 100_000 / baseline_median,
                "instrumented_seconds": timed[key],
                "instrumentation_overhead_fraction": overhead_fraction,
                "phase_seconds": phase_seconds,
                "dominant_phase": ordered_phases[0][0],
                "model_load_seconds": timed["model_load_seconds"],
                "warmup_seconds": timed["warmup_seconds"],
                "codec_peak_allocated_bytes": timed["codec_peak_allocated_bytes"],
                "run_peak_allocated_bytes": timed["run_peak_allocated_bytes"],
                "run_peak_reserved_bytes": timed["run_peak_reserved_bytes"],
            }
        encoded = runs[0]["encode"]
        ar_macs_per_predicted_byte = None
        if system == "autoregressive":
            model_config = BASELINE_MAMBA3_CONFIG
            ar_macs = autoregressive_macs(
                model_config, predicted_bytes=encoded["predicted_bytes"]
            )
            macs_per_byte = ar_macs["total_macs"] / encoded["num_bytes"]
            ar_macs_per_predicted_byte = ar_macs["macs_per_predicted_byte"]
        else:
            model_config = MTP4_MAMBA3_CONFIG
            macs_per_byte = mtp4_sequence_macs(
                model_config,
                state_tokens=encoded["state_tokens_processed"],
                prediction_groups=encoded["num_prediction_groups"],
                input_bytes=encoded["num_bytes"],
            )["macs_per_input_byte"]
        summaries[system] = {
            "label": config["label"],
            "checkpoint_sha256": next(iter(checkpoint_hashes)),
            "compressed_sha256": next(iter(bitstreams)),
            "compressed_bytes": encoded["compressed_bytes"],
            "actual_bpb": encoded["actual_bpb"],
            "num_model_steps": encoded["num_model_steps"],
            "parameter_count": analytic_parameter_counts(model_config)["total_trainable_params"],
            "macs_per_input_byte": macs_per_byte,
            "macs_per_predicted_byte": ar_macs_per_predicted_byte,
            "operations": operations,
            "gpu_profile": gpu,
            "needs_more_baseline_repeats": any(
                value["baseline_range_fraction"] > 0.05
                for value in operations.values()
            ),
            "needs_lower_timing_sampling": any(
                abs(value["instrumentation_overhead_fraction"]) > 0.05
                for value in operations.values()
            ),
        }
    if len(input_hashes) != 1:
        raise AssertionError("The systems did not use the same original bytes")

    output = {
        "sample_bytes": 100_000_000,
        "batch_size": 1024,
        "batch_sizes": [1024] * 47 + [701],
        "original_sha256": next(iter(input_hashes)),
        "systems": summaries,
        "timing_note": (
            "CPU wall phases add to codec wall time. CUDA events are isolated "
            "microbenchmarks and must not be added to wall phases."
        ),
    }
    summary_path = root / "summary.json"
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (root / "phase_breakdown.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(phase_rows[0]))
        writer.writeheader()
        writer.writerows(phase_rows)

    lines = [
        "# batch_size=1024 编解码速度瓶颈",
        "",
        "两套系统均使用 enwik9 前 100,000,000 字节和 RTX 3090；48,829 个独立块分为 47 批 1024 块及一批 701 块。编码和解码均由独立进程执行，恢复文件逐字节及 SHA256 校验通过，同一系统的重复基线与插桩码流一致。",
        "",
        "## 无插桩端到端结果",
        "",
        "| 系统 | 操作 | 正式重复耗时 s | 中位速度 KB/s | BPB | 压缩文件 MB | 正式阶段显存 GB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for system, item in summaries.items():
        for operation, values in item["operations"].items():
            seconds = "、".join(f"{value:.3f}" for value in values["baseline_seconds"])
            lines.append(
                f"| {item['label']} | {OPERATIONS[operation]} | {seconds} | "
                f"{values['baseline_median_kb_per_second']:.2f} | "
                f"{item['actual_bpb']:.4f} | "
                f"{item['compressed_bytes'] / 1e6:.2f} | "
                f"{values['codec_peak_allocated_bytes'] / 1e9:.3f} |"
            )
    lines += [
        "",
        "速度按原始字节数除以正式编解码耗时计算，不含模型加载及提前的 Graph 准备。显存为 PyTorch 已分配张量峰值，不等于整卡占用；完整运行峰值及预留量保留在 summary.json。",
        "",
        "## 插桩分段",
        "",
        "| 系统 | 操作 | 分段 | 秒 | 正式阶段占比 |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in phase_rows:
        lines.append(
            f"| {SYSTEMS[row['system']]['label']} | {OPERATIONS[row['operation']]} | "
            f"{row['label']} | {row['seconds']:.3f} | {row['share_percent']:.2f}% |"
        )
    lines += ["", "预测器阶段包括输入传输、GPU 推理、概率回传及相关主机开销。区间编码阶段包含 Constriction 内部的概率量化。"]
    for system, item in summaries.items():
        lines += ["", f"### {item['label']}"]
        for operation, values in item["operations"].items():
            lines.append(
                f"{OPERATIONS[operation]}插桩总时间 {values['instrumented_seconds']:.3f} 秒，"
                f"相对无插桩中位数偏差 {100 * values['instrumentation_overhead_fraction']:+.2f}%；"
                f"模型加载 {values['model_load_seconds']:.3f} 秒，预热与 Graph 准备 "
                f"{values['warmup_seconds']:.3f} 秒。"
            )
        gpu = item["gpu_profile"]
        events = gpu["isolated_cuda_event_ms_for_128_byte_positions"]
        lines.append(
            "稳定状态下 128 字节跨度的独立 CUDA Event 中位数："
            f"输入传输 {events['h2d']['median']:.3f} ms，"
            f"Graph 执行 {events['graph']['median']:.3f} ms，"
            f"概率回传 {events['d2h']['median']:.3f} ms；"
            f"完整预测器调用的墙钟时间 {gpu['public_predictor_wall_seconds'] * 1000:.3f} ms。"
        )
        lines.append("GPU 跟踪中设备时间最高的事件：")
        count = 0
        for event in gpu["top_cuda_events"]:
            name = event["name"]
            if name == "codec_predictor_step" or name.startswith("aten::"):
                continue
            lines.append(
                f"- {kernel_label(name)}：{event['device_time_us'] / 1000:.3f} ms，"
                f"{event['count']} 次。"
            )
            count += 1
            if count == 5:
                break
    lines += [
        "",
        "CUDA Event（GPU 事件计时）来自隔离的短时测试；Graph 重放的事件时间包含可能的主机提交间隙。GPU 跟踪也会引入额外开销，这些设备时间与端到端墙钟分段不能相加。原始 GPU 事件名、调用次数和 Chrome trace 保存在各系统的 gpu 目录。",
        "",
        "## 瓶颈判定",
        "",
    ]
    for item in summaries.values():
        encode = item["operations"]["encode"]
        decode = item["operations"]["decode"]
        encode_share = 100 * encode["phase_seconds"]["predictor"] / encode["instrumented_seconds"]
        decode_share = 100 * decode["phase_seconds"]["predictor"] / decode["instrumented_seconds"]
        lines.append(
            f"{item['label']}：预测器占编码时间 {encode_share:.1f}%、"
            f"解码时间 {decode_share:.1f}%，是两端的最大瓶颈。"
        )
        top_kernel = next(
            (
                event for event in item["gpu_profile"]["top_cuda_events"]
                if event["name"] != "codec_predictor_step"
                and not event["name"].startswith("aten::")
            ),
            None,
        )
        if top_kernel:
            lines.append(
                f"短时 GPU 跟踪中，该系统耗时最高的内核为 "
                f"{kernel_label(top_kernel['name'])}。"
            )
    ar_encode = summaries["autoregressive"]["operations"]["encode"]["phase_seconds"]
    mtp_encode = summaries["mtp4"]["operations"]["encode"]["phase_seconds"]
    ar_events = summaries["autoregressive"]["gpu_profile"][
        "isolated_cuda_event_ms_for_128_byte_positions"
    ]
    mtp_events = summaries["mtp4"]["gpu_profile"][
        "isolated_cuda_event_ms_for_128_byte_positions"
    ]
    lines.append(
        f"编码插桩中，MTP-4 的预测器比自回归少用 "
        f"{ar_encode['predictor'] - mtp_encode['predictor']:.2f} 秒，"
        f"CPU 概率处理多用 "
        f"{mtp_encode['probability_cpu'] - ar_encode['probability_cpu']:.2f} 秒。"
        f"相同 128 字节输入跨度下，Graph 重放分别为 "
        f"{ar_events['graph']['median']:.1f} 和 "
        f"{mtp_events['graph']['median']:.1f} ms；"
        "MTP-4 虽将模型调用次数降为四分之一，四字节续算仍需更新对应状态，"
        "所以 Graph 阶段用时没有按四倍下降。"
    )
    lines += [
        "",
        "## 模型规模与可复核结果",
        "",
        "| 系统 | 参数量 | MAC/输入字节 | 模型调用次数 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in summaries.values():
        lines.append(
            f"| {item['label']} | {item['parameter_count']:,} | "
            f"{item['macs_per_input_byte']:,.0f} | {item['num_model_steps']:,} |"
        )
    lines += [
        "",
        "MAC 表示一次乘加计算，表中按原始输入字节平均。",
        f"自回归按被预测字节计为 {summaries['autoregressive']['macs_per_predicted_byte']:,} MAC/字节；表中统一按原始输入字节计，因此摊入了每块首字节的均匀编码。",
        f"原始文件 SHA256：{output['original_sha256']}。",
        "",
        "复现：使用 benchmark_codec_batched.py，分别指定 --format v5/v6、--execution graph、--sample-bytes 100000000、--batch-size 1024；插桩运行另加 --phase-timing。短时跟踪使用 profile_codec_graph_bottleneck.py，汇总使用 summarize_codec_bottleneck.py。两套系统使用各自 checkpoint，完整路径和指纹见 summary.json。",
        "",
    ]
    with (root / "report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(json.dumps({
        "summary": str(summary_path),
        "phase_csv": str(root / "phase_breakdown.csv"),
        "report": str(root / "report.md"),
        "needs_more_baseline_repeats": {
            key: value["needs_more_baseline_repeats"]
            for key, value in summaries.items()
        },
        "needs_lower_timing_sampling": {
            key: value["needs_lower_timing_sampling"]
            for key, value in summaries.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Controlled BPE vocabulary sweep with read-only historical references."""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import statistics
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from .token_experiment import ROOT, announce, git_info, prepare
from .token_pipeline import (checked_inputs, code_fingerprint, gate, inference_export,
                             preflight, roundtrip_job, run_job)
from .token_training import TRAINING
from .tokenization import digest_file, load_json, save_json

VOCAB_SIZES = (8192, 16384)
STAGES = ("prepare", "preflight", "tune", "train", "evaluate", "report", "all")
ARCHIVES = {name: ROOT / "docs" / "实验数据" / f"sentencepiece-{name}.json"
            for name in ("results", "performance")}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--vocab-sizes", nargs="+", type=int, choices=VOCAB_SIZES, default=list(VOCAB_SIZES))
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/raw/enwik9")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if len(set(args.vocab_sizes)) != len(args.vocab_sizes):
        parser.error("Vocabulary sizes must not be repeated")
    args.vocab_sizes = sorted(args.vocab_sizes)
    args.run_dir = args.run_dir.resolve()
    args.source = args.source.resolve()
    return args


def groups(args):
    return [SimpleNamespace(run_dir=args.run_dir / f"v{vocab}", source=args.source,
                            device=args.device, vocab_size=vocab, algorithms=("bpe",),
                            algorithm="bpe", experiment="bpe-vocab-scale")
            for vocab in args.vocab_sizes]


def bind_run(args):
    historical = load_json(ARCHIVES["results"])
    source_sha = digest_file(args.source)
    if args.source.stat().st_size != 1_000_000_000 or source_sha != historical["manifest"]["source_sha256"]:
        raise ValueError("Source must match the archived enwik9 experiment")
    settings = dict(experiment="bpe-vocab-scale", vocab_sizes=args.vocab_sizes,
                    algorithm="bpe", source=str(args.source), source_sha256=source_sha,
                    code_sha256=code_fingerprint(), device=args.device,
                    effective_batch_size=72, block_size=2048, codec_batch_size=1024,
                    tokenizer_threads=24, workers=0, training_config=TRAINING,
                    historical_archives={k: dict(path=str(p), sha256=digest_file(p))
                                         for k, p in ARCHIVES.items()})
    settings = json.loads(json.dumps(settings))
    path = args.run_dir / "sweep_manifest.json"
    if path.exists():
        previous = load_json(path)
        if any(previous.get(k) != v for k, v in settings.items()):
            raise ValueError("Sweep configuration, archives or implementation changed; use a new run directory")
    else:
        save_json(path, dict(settings, created_git=git_info(), created_at=time.time()))


def environment(args):
    path = args.run_dir / "environment.json"
    if path.exists():
        return
    import importlib.metadata
    import torch
    if not (Path(sys.prefix) / "conda-meta/history").exists():
        raise RuntimeError("Activate the project's Conda environment before running")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The BPE experiment requires an accessible BF16 CUDA GPU")
    packages = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
                      if d.metadata.get("Name"))
    conda = (os.environ.get("CONDA_EXE") or shutil.which("conda") or
             str(Path(os.environ.get("MAMBA3_CONDA_ROOT", str(Path.home() / "anaconda3"))) / "bin/conda"))
    explicit = subprocess.check_output([conda,
                                        "list", "--explicit", "--prefix", sys.prefix], text=True)
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                      "--format=csv,noheader"], text=True).strip()
    save_json(path, dict(python=sys.version, executable=sys.executable, conda_prefix=sys.prefix,
                        packages=packages, conda_explicit=explicit, gpu_driver=driver,
                        torch=torch.__version__, cuda=torch.version.cuda,
                        gpu=torch.cuda.get_device_name(torch.device(args.device)), git=git_info()))


def tune_group(group):
    checked_inputs(group)
    gate(group, "preflight")
    candidates = []
    for micro in sorted((n for n in range(1, 73) if 72 % n == 0), reverse=True):
        announce(f"BPE {group.vocab_size}: probe effective=72 micro={micro}")
        record = run_job(group, group.run_dir / "probes" / f"micro{micro}",
                         dict(kind="probe", directory=str(group.run_dir / "bpe"),
                              batch_size=micro, effective_batch_size=72, workers=0))
        candidates.append(record)
        if record["safe"]:
            result = dict(verified=True, batch_size=72, effective_batch_size=72,
                          micro_batch_size=micro, workers=0, candidates=candidates,
                          manifest_sha256=digest_file(group.run_dir / "manifest.json"),
                          code_sha256=code_fingerprint(), git=git_info())
            save_json(group.run_dir / "tuning.json", result)
            return result
    raise RuntimeError(f"BPE {group.vocab_size}: no micro batch retains 1 GiB headroom")


def train_group(group):
    checked_inputs(group)
    gate(group, "preflight")
    gate(group, "tuning")
    command = [sys.executable, "-u", str(ROOT / "scripts/benchmark_sentencepiece.py"), "train",
               "--run-dir", str(group.run_dir), "--algorithm", "bpe", "--device", group.device]
    with (group.run_dir / "bpe/training.log").open("a", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = load_json(group.run_dir / "bpe/training/summary.json")
    if not summary["converged"]:
        raise RuntimeError("Training interrupted before convergence")


def export_group(group):
    checked_inputs(group)
    gate(group, "preflight")
    gate(group, "tuning")
    summary = load_json(group.run_dir / "bpe/training/summary.json")
    if not summary["converged"] or summary["vocab_size"] != group.vocab_size:
        raise ValueError("Matching converged training is required before evaluation")
    target = group.run_dir / "evaluation/bpe/inference.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    inference_export(group.run_dir / "bpe/training/best.pt", target, "bpe")
    return target


def evaluate_case(group, split, checkpoint, *, repeat=None, profile=False):
    raw = group.source if split == "full" else group.run_dir / f"splits/{split}.bin"
    metadata = (group.run_dir / "bpe/full_reversibility.json" if split == "full" else
                group.run_dir / f"bpe/{split}.tokens.json")
    count = load_json(metadata)["tokens"]
    name = "profile" if profile else f"repeat{repeat}" if repeat is not None else "single"
    directory = group.run_dir / "evaluation/bpe" / split / name
    result = roundtrip_job(group, directory, "bpe", checkpoint, raw, count, profile=profile, batch=1024)
    package_bytes = checkpoint.stat().st_size
    result.update(vocab_size=group.vocab_size, split=split, profile=profile, repeat=repeat,
                  inference_package_bytes=package_bytes,
                  including_weights_bpb=8 * (result["encode"]["compressed_bytes"] + package_bytes) / raw.stat().st_size)
    save_json(directory / "roundtrip.json", result)
    return result


def evaluate_groups(args):
    selected = groups(args)
    checkpoints = {g.vocab_size: export_group(g) for g in selected}
    for group in selected:
        announce(f"BPE {group.vocab_size}: validation profile")
        evaluate_case(group, "validation", checkpoints[group.vocab_size], profile=True)
    for repeat in range(1, 4):
        order = selected if repeat % 2 else list(reversed(selected))
        for group in order:
            announce(f"BPE {group.vocab_size}: validation repeat {repeat}/3")
            evaluate_case(group, "validation", checkpoints[group.vocab_size], repeat=repeat)
            report(args)
    for group in selected:
        for split in ("test", "full"):
            announce(f"BPE {group.vocab_size}: {split}")
            evaluate_case(group, split, checkpoints[group.vocab_size])
            report(args)


def collect_row(group, split, trials, training, historical_runtime):
    if len({t["encode"]["compressed_sha256"] for t in trials}) != 1:
        raise ValueError("Repeated encoding produced different streams")
    first = trials[0]
    enc = first["encode"]
    verified = all(t["verified"] and t["decode"]["files_identical"]
                   and t["decode"]["decoded_sha256"] == t["raw_sha256"] for t in trials)
    if not verified:
        raise ValueError("An evaluation failed lossless verification")
    row = dict(vocab_size=group.vocab_size, algorithm="bpe", split=split,
               original_bytes=enc["num_bytes"], num_tokens=enc["num_tokens"],
               bytes_per_token=enc["num_bytes"] / enc["num_tokens"],
               actual_bpb=enc["actual_bpb"], including_weights_bpb=first["including_weights_bpb"],
               compressed_bytes=enc["compressed_bytes"], compressed_sha256=enc["compressed_sha256"],
               raw_sha256=first["raw_sha256"], checkpoint_sha256=first["checkpoint_sha256"],
               inference_package_bytes=first["inference_package_bytes"],
               parameter_counts=training["parameter_counts"], repeats=len(trials), verified=verified,
               effective_batch_size=training["effective_batch_size"], micro_batch_size=training["micro_batch_size"],
               training_epochs=training["epoch"], training_seconds=training["elapsed_seconds"],
               best_validation_bpb=training["best_bpb"], runtime=enc["runtime"],
               runtime_matches_historical=all(enc["runtime"].get(k) == v for k, v in historical_runtime.items()))
    for op in ("encode", "decode"):
        times = [t[op][f"{op}_seconds"] for t in trials]
        row[f"{op}_seconds"] = dict(samples=times, median=statistics.median(times), min=min(times), max=max(times))
        row[f"{op}_kb_per_second"] = enc["num_bytes"] / 1000 / statistics.median(times)
        for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
            row[f"{op}_{field}"] = max(t[op][field] for t in trials)
        row[f"{op}_model_load_seconds"] = [t[op]["model_load_seconds"] for t in trials]
        row[f"{op}_warmup_seconds"] = [t[op]["warmup_seconds"] for t in trials]
        row[f"{op}_process_seconds"] = [t[op]["process_seconds"] for t in trials]
    return row


def report(args):
    historical = load_json(ARCHIVES["results"])
    performance = load_json(ARCHIVES["performance"])
    baseline = [dict(row, vocab_size=4096, measurement="historical_old_implementation")
                for row in historical["results"] if row["algorithm"] == "bpe"]
    optimized = next(row for row in performance["benchmark"]
                     if row["algorithm"] == "bpe" and row["variant"] == "optimized")
    package_size = baseline[0]["inference_package_bytes"]
    raw_sha = next(row["raw_sha256"] for row in performance["runs"]
                   if row["algorithm"] == "bpe" and row["variant"] == "optimized")
    baseline.append(dict(optimized, vocab_size=4096, split="validation", raw_sha256=raw_sha,
                         original_bytes=performance["raw_bytes"],
                         including_weights_bpb=optimized["actual_bpb"] + 8 * package_size / performance["raw_bytes"],
                         measurement="historical_optimized_implementation"))
    rows, profiles, training_records = [], {}, {}
    for group in groups(args):
        path = group.run_dir / "bpe/training/summary.json"
        if not path.exists():
            continue
        training = load_json(path)
        training_records[str(group.vocab_size)] = training
        profile = group.run_dir / "evaluation/bpe/validation/profile/roundtrip.json"
        if profile.exists():
            profiles[str(group.vocab_size)] = load_json(profile)
        for split in ("validation", "test", "full"):
            parent = group.run_dir / "evaluation/bpe" / split
            paths = ([parent / f"repeat{n}/roundtrip.json" for n in range(1, 4)]
                     if split == "validation" else [parent / "single/roundtrip.json"])
            if not all(p.exists() for p in paths):
                continue
            trials = [load_json(p) for p in paths]
            row = collect_row(group, split, trials, training, performance["runtime"])
            reference = next(b for b in baseline if b["split"] == split)
            if row["raw_sha256"] != reference["raw_sha256"]:
                raise ValueError("Evaluation bytes do not match the historical reference")
            row["bpb_reduction_percent_vs_4096"] = 100 * (1 - row["actual_bpb"] / reference["actual_bpb"])
            comparable = split == "validation" and row["runtime_matches_historical"]
            row["speed_comparison"] = "historical_same_runtime" if comparable else "reference_only"
            for op in ("encode", "decode"):
                row[f"{op}_throughput_change_percent_vs_4096"] = (
                    100 * (row[f"{op}_kb_per_second"] / reference[f"{op}_kb_per_second"] - 1)
                    if comparable else None)
            rows.append(row)
    complete = (len(rows) == 3 * len(args.vocab_sizes) and len(profiles) == len(args.vocab_sizes)
                and all(t["converged"] for t in training_records.values()))
    result = dict(complete=complete, results=rows, profiles=profiles, training=training_records,
                  baseline_4096=baseline, historical_runtime=performance["runtime"], baseline_retested=False,
                  manifest=load_json(args.run_dir / "sweep_manifest.json"),
                  limitations=["4096 未同期复测", "每个词表仅使用一个训练种子",
                               "全量历史速度来自优化前实现，仅作参考"], git=git_info())
    save_json(args.run_dir / "results.json", result)
    if rows:
        with (args.run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()} for row in rows)
    lines = ["# BPE 词表规模对照实验", "", "仅新测 8192、16384；4096 引用历史记录，未重新训练或测试。",
             "890 MB 为独立主测试；完整 1 GB 包含训练和验证内容，仅作补充。",
             "实际 BPB 包含分词器和完整容器，含权重 BPB 再计入独立推理包。", "",
             "| 来源 | 词表 | 范围 | 实际 BPB | 含权重 BPB | 编码 KB/s | 解码 KB/s |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in baseline + rows:
        lines.append(f"| {'历史' if row['vocab_size'] == 4096 else '本次'} | {row['vocab_size']} | {row['split']} | "
                     f"{row['actual_bpb']:.6f} | {row['including_weights_bpb']:.6f} | "
                     f"{row['encode_kb_per_second']:.2f} | {row['decode_kb_per_second']:.2f} |")
    lines += ["", "## 比较与限制", "",
              "10 MB 速度使用三次无分阶段计时干扰的独立进程结果中位数。编码含实际分词、文件读写及内部校验；"
              "解码含原文字节还原及内部校验。加载、预热和外部逐字节比较单列。",
              "4096 的 890 MB 和 1 GB 速度来自优化前实现，不能据此归因词表变化；"
              "10 MB 的历史优化后速度也未同期复测。硬件或依赖不一致时不计算速度改善百分比。",
              "每组只训练一个种子，不能据此断言跨训练种子的稳定收益。"]
    for row in rows:
        lines += ["", f"词表 {row['vocab_size']}，{row['split']}：相对 4096 的实际 BPB 降低 "
                  f"{row['bpb_reduction_percent_vs_4096']:.2f}%；平均每 token 对应 {row['bytes_per_token']:.3f} 字节。"]
        if row["speed_comparison"] == "historical_same_runtime":
            lines += [f"编码吞吐变化 {row['encode_throughput_change_percent_vs_4096']:+.2f}%，"
                      f"解码吞吐变化 {row['decode_throughput_change_percent_vs_4096']:+.2f}%。"]
        for op in ("encode", "decode"):
            t = row[f"{op}_seconds"]
            lines += [f"{op} 耗时：中位数 {t['median']:.3f} 秒，范围 [{t['min']:.3f}, {t['max']:.3f}] 秒。"]
    lines += ["", "## 训练状态", ""]
    for vocab, training in training_records.items():
        lines += [f"词表 {vocab}：{training['epoch']} 轮，有效批大小 {training['effective_batch_size']}，"
                  f"微批大小 {training['micro_batch_size']}，训练 {training['elapsed_seconds']/3600:.2f} 小时；"
                  f"停止原因 {training['stop_reason']}。",
                  f"[训练曲线](v{vocab}/bpe/training/bpb_vs_steps.png) · "
                  f"[训练记录](v{vocab}/bpe/training/history.json)"]
    lines += ["", "状态：" + ("全部完成，所有恢复结果均通过逐字节比较与 SHA256 校验。" if complete else
                                  "实验尚未完成；缺失结果不使用估计值填充。"), ""]
    (args.run_dir / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")
    return complete


def main(argv=None):
    args = parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / ".sweep.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage != "report" and git_info()["dirty"]:
            raise RuntimeError("Commit implementation changes before running experiment stages")
        bind_run(args)
        if args.stage == "report":
            report(args)
            return
        stages = ("prepare", "preflight", "tune", "train", "evaluate") if args.stage == "all" else (args.stage,)
        status_path = args.run_dir / "status.json"
        stage = "environment"
        try:
            environment(args)
            for stage in stages:
                save_json(status_path, dict(status="running", stage=stage, pid=os.getpid(), updated_at=time.time()))
                announce(f"BPE vocabulary sweep: {stage}")
                if stage == "evaluate":
                    evaluate_groups(args)
                else:
                    operation = {"prepare": prepare, "preflight": preflight, "tune": tune_group, "train": train_group}[stage]
                    for group in groups(args):
                        save_json(status_path, dict(status="running", stage=stage, vocab_size=group.vocab_size,
                                                   pid=os.getpid(), updated_at=time.time()))
                        operation(group)
                report(args)
            complete = report(args)
            if args.stage == "all" and not complete:
                raise RuntimeError("Sweep finished without all required results")
            save_json(status_path, dict(status="complete" if complete else "stage_complete", stage=args.stage,
                                       updated_at=time.time(), pid=os.getpid()))
        except BaseException as error:
            save_json(status_path, dict(status="failed", stage=stage, error=f"{type(error).__name__}: {error}",
                                       pid=os.getpid(), updated_at=time.time()))
            report(args)
            raise

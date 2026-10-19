"""GPU workers, preflight gates, training selection and full experiment reports."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from .tokenization import LosslessTokenizer, digest_file, load_json, save_json
from .token_experiment import ALGORITHMS, ROOT, announce, git_info, prepare, require_manifest


def code_fingerprint():
    files = sorted((ROOT / "mamba3_byte_compress").glob("*.py"))
    files += [ROOT / "requirements-codec.txt", ROOT / "scripts/benchmark_sentencepiece.py"]
    return hashlib.sha256("".join(digest_file(path) for path in files).encode()).hexdigest()


def gate(args, name):
    result = load_json(args.run_dir / f"{name}.json")
    if not result.get("verified"):
        raise ValueError(f"Stage {name} has not passed")
    if name != "prepared" and result.get("code_sha256") != code_fingerprint():
        raise ValueError(f"Stage {name} used different code; rerun it")
    if result["manifest_sha256"] != digest_file(args.run_dir / "manifest.json"):
        raise ValueError("Experiment manifest changed")
    return result


def checked_inputs(args):
    gate(args, "prepared")
    manifest = require_manifest(args)
    if digest_file(manifest["source"]) != manifest["source_sha256"]:
        raise ValueError("Source file changed")
    if digest_file(manifest["byte_checkpoint"]) != manifest["byte_checkpoint_sha256"]:
        raise ValueError("Byte checkpoint changed")
    for algorithm in ALGORITHMS:
        directory = args.run_dir / algorithm
        model_hash = digest_file(directory / "tokenizer.model")
        for split in ("train", "validation", "test"):
            meta = load_json(str(directory / f"{split}.tokens") + ".json")
            if (not meta["verified"] or meta["tokenizer_sha256"] != model_hash
                    or meta["token_sha256"] != digest_file(directory / f"{split}.tokens")
                    or meta["original_sha256"] != digest_file(args.run_dir / "splits" / f"{split}.bin")):
                raise ValueError("Prepared token cache/source changed")
        full = load_json(directory / "full_reversibility.json")
        if not full["verified"] or full["tokenizer_sha256"] != model_hash or full["original_sha256"] != manifest["source_sha256"]:
            raise ValueError("Full reversibility check is stale")
    return manifest


def run_job(args, directory, job):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    job = dict(job, device=args.device, code_sha256=code_fingerprint())
    for key in ("checkpoint", "raw", "tokenizer"):
        if job.get(key):
            job[f"{key}_sha256"] = digest_file(job[key])
    specification = json.dumps(job, sort_keys=True)
    signature = hashlib.sha256(specification.encode()).hexdigest()
    result_path = directory / "result.json"
    if result_path.exists():
        result = load_json(result_path)
        if result.get("job_sha256") == signature:
            artifact = job.get("packed") if job.get("operation") == "encode" else job.get("decoded")
            field = "compressed_sha256" if job.get("operation") == "encode" else "decoded_sha256"
            if artifact and (not Path(artifact).exists() or result.get(field) != digest_file(artifact)):
                raise ValueError("Completed codec artifact changed")
            return result
    # Preserve incomplete or old-run artifacts, without silently reusing them.
    for path in (result_path, Path(job["packed"]) if job.get("operation") == "encode" else
                 Path(job["decoded"]) if job.get("operation") == "decode" else None):
        if path is not None and path.exists():
            path.rename(path.with_name(path.name + f".incomplete.{time.time_ns()}"))
    save_json(directory / "job.json", job)
    log_path = directory / "worker.log"
    with log_path.open("a") as log:
        result = subprocess.run([sys.executable, "-u", str(ROOT / "scripts/benchmark_sentencepiece.py"),
                                 "worker", "--run-dir", str(args.run_dir), "--job", str(directory / "job.json")],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Worker failed ({result.returncode}); inspect {log_path}")
    result = load_json(result_path)
    if result["job_sha256"] != signature:
        raise ValueError("Worker result signature mismatch")
    return result


def is_cuda_out_of_memory(error):
    import torch
    return (isinstance(error, torch.cuda.OutOfMemoryError)
            or isinstance(error, RuntimeError)
            and str(error).strip().lower().startswith("triton error [cuda]: out of memory"))


def worker(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    job = load_json(args.job)
    signature = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    if job["code_sha256"] != code_fingerprint():
        raise ValueError("Code changed before worker start")
    for key in ("checkpoint", "raw", "tokenizer"):
        if job.get(key) and digest_file(job[key]) != job[f"{key}_sha256"]:
            raise ValueError(f"Worker {key} changed")
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Run GPU stages outside the device-isolated sandbox")
    if job["kind"] == "probe":
        from .token_training import probe_batch
        try:
            result = probe_batch(job["directory"], job["batch_size"], job["device"], job["workers"])
        except RuntimeError as error:
            if not is_cuda_out_of_memory(error):
                raise
            result = dict(safe=False, oom=True, error=str(error),
                          batch_size=job["batch_size"], workers=job["workers"])
    elif job["kind"] == "codec":
        from . import codec as c
        from . import token_codec as tc
        started = time.perf_counter()
        token_mode = job["algorithm"] != "byte"
        if token_mode:
            predictor = tc.load_token_predictor(job["checkpoint"], job["device"], job["batch_size"])
        else:
            predictor = c.load_predictor(job["checkpoint"], job["device"], batch_size=job["batch_size"])
        predictor.synchronize()
        load_seconds = time.perf_counter() - started
        prepare_seconds = tc.warmup(predictor, job["token_count"])
        if job.get("profile"):
            predictor.profile_timings = dict(upload=0., model_probability=0., download=0.)
        torch.cuda.reset_peak_memory_stats(predictor.device)
        if job["operation"] == "encode":
            if token_mode:
                tokenizer = LosslessTokenizer(job["tokenizer"], job["threads"])
                result = tc.encode(job["raw"], job["packed"], predictor, tokenizer, profile=job.get("profile", False))
            else:
                result = c.encode_file(job["raw"], job["packed"], predictor, file_format="v5")
                result["compressed_sha256"] = digest_file(job["packed"])
        else:
            result = (tc.decode(job["packed"], job["decoded"], predictor, profile=job.get("profile", False))
                      if token_mode else c.decode_file(job["packed"], job["decoded"], predictor))
            if not c.verify_files(job["raw"], job["decoded"]):
                raise AssertionError("Independent-process decoding differs from original bytes")
            result.update(files_identical=True, decoded_sha256=digest_file(job["decoded"]))
        result.update(model_load_seconds=load_seconds, warmup_seconds=prepare_seconds,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(predictor.device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(predictor.device),
                      prediction_phases_seconds=getattr(predictor, "profile_timings", None),
                      gpu=torch.cuda.get_device_name(predictor.device), process_id=os.getpid())
    else:
        raise ValueError("Unknown worker kind")
    result.update(job_sha256=signature, git=git_info())
    save_json(args.job.parent / "result.json", result)


def synthetic_preflight():
    import constriction
    family = constriction.stream.model.Categorical(perfect=False)
    rng = np.random.default_rng(2026)
    results = []
    for vocab in (256, 4096):
        for label in ("uniform", "random", "peaked", "zeros"):
            p = rng.random((1024, vocab), dtype=np.float32)
            if label == "uniform":
                p.fill(1.)
            elif label in ("peaked", "zeros"):
                p.fill(1e-30 if label == "peaked" else 0.)
                p[:, 0] = 1.
            p /= p.sum(axis=1, keepdims=True)
            symbols = rng.integers(0, vocab, 1024, dtype=np.int32)
            symbols[:4] = [0, 255, min(256, vocab-1), vocab-1]
            times, reference = [], None
            for _ in range(3):
                tick = time.perf_counter()
                encoder = constriction.stream.queue.RangeEncoder()
                encoder.encode(symbols, family, p)
                compressed = encoder.get_compressed()
                decoder = constriction.stream.queue.RangeDecoder(compressed)
                recovered = decoder.decode(family, p)
                times.append(time.perf_counter()-tick)
                np.testing.assert_array_equal(symbols, recovered)
                if reference is not None:
                    np.testing.assert_array_equal(reference, compressed)
                reference = compressed
            results.append(dict(vocab_size=vocab, distribution=label, seconds=times,
                                median_seconds=statistics.median(times), verified=True))
    return results


def roundtrip_job(args, directory, algorithm, checkpoint, raw, tokens, *, profile=False, batch=1024):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    job = dict(kind="codec", algorithm=algorithm, checkpoint=str(checkpoint), raw=str(raw),
               packed=str(directory / "compressed.m3rc"), decoded=str(directory / "decoded.bin"),
               token_count=tokens, batch_size=batch, profile=profile)
    if algorithm != "byte":
        source = args.run_dir / algorithm
        job.update(tokenizer=str(source / "tokenizer.model"), threads=load_json(source / "threads.json")["threads"])
    encoded = run_job(args, directory / "encode", dict(job, operation="encode"))
    decoded = run_job(args, directory / "decode", dict(job, operation="decode"))
    result = dict(encode=encoded, decode=decoded, verified=decoded["files_identical"],
                  checkpoint_sha256=digest_file(checkpoint), raw_sha256=digest_file(raw),
                  packed=job["packed"], decoded=job["decoded"], code_sha256=code_fingerprint())
    save_json(directory / "roundtrip.json", result)
    return result


def preflight(args):
    manifest = checked_inputs(args)
    from .model import build_mamba3_token_model
    from .trainer import seed_everything, _atomic_torch_save
    import torch
    result = dict(synthetic=synthetic_preflight(), models={})
    for algorithm in ALGORITHMS:
        directory = args.run_dir / algorithm
        path = directory / "preflight.pt"
        if not path.exists():
            seed_everything(2026)
            model, config = build_mamba3_token_model()
            with torch.no_grad():
                model.lm_head.weight.normal_(std=0.02)
            _atomic_torch_save(dict(model=model.state_dict(), model_config=config, vocab_size=4096,
                                   tokenizer_sha256=digest_file(directory / "tokenizer.model")), path)
            del model
        trials = []
        for repeat in range(3):
            announce(f"GPU preflight {algorithm}, repeat {repeat+1}/3")
            trials.append(roundtrip_job(args, directory / "preflight" / f"repeat{repeat+1}", algorithm, path,
                          args.run_dir / "splits/validation.bin",
                          load_json(str(directory / "validation.tokens") + ".json")["tokens"], profile=True))
        hashes = [trial["encode"]["compressed_sha256"] for trial in trials]
        if len(set(hashes)) != 1:
            raise AssertionError("Repeated GPU encoding produced different streams")
        result["models"][algorithm] = trials
    result.update(verified=True, manifest_sha256=digest_file(args.run_dir / "manifest.json"),
                  code_sha256=code_fingerprint(), git=git_info(), completed_at=time.time())
    save_json(args.run_dir / "preflight.json", result)


def tune(args):
    gate(args, "preflight")
    candidates = []
    batch = 16
    # Separate processes release all CUDA allocations after every attempted size.
    while batch <= 1024:
        announce(f"Training batch probe B={batch}")
        record = run_job(args, args.run_dir / "probes" / f"bpe_b{batch}_w0",
                         dict(kind="probe", directory=str(args.run_dir / "bpe"), batch_size=batch, workers=0))
        candidates.append(record)
        if not record["safe"]:
            break
        batch *= 2
    safe = [r for r in candidates if r["safe"]]
    if not safe:
        raise RuntimeError("No tested batch fits with 1 GiB headroom")
    lower = max(r["batch_size"] for r in safe)
    if batch > lower + 8:
        for value in range(lower+8, min(batch, 1025), 8):
            record = run_job(args, args.run_dir / "probes" / f"bpe_b{value}_w0",
                             dict(kind="probe", directory=str(args.run_dir / "bpe"), batch_size=value, workers=0))
            candidates.append(record)
            if not record["safe"]:
                break
    ranked = sorted((r for r in candidates if r["safe"]), key=lambda row: row["tokens_per_second"], reverse=True)
    pair = None
    for candidate in ranked:
        other = run_job(args, args.run_dir / "probes" / f"unigram_b{candidate['batch_size']}_w0",
                        dict(kind="probe", directory=str(args.run_dir / "unigram"),
                             batch_size=candidate["batch_size"], workers=0))
        if other["safe"]:
            pair = candidate, other
            break
    if pair is None:
        raise RuntimeError("No common stable training batch")
    batch = pair[0]["batch_size"]
    worker_options = [(min(p["tokens_per_second"] for p in pair), 0, pair)]
    for workers in (2, 4):
        records = [run_job(args, args.run_dir / "probes" / f"{algorithm}_b{batch}_w{workers}",
                          dict(kind="probe", directory=str(args.run_dir / algorithm), batch_size=batch, workers=workers))
                   for algorithm in ALGORITHMS]
        if all(record["safe"] for record in records):
            worker_options.append((min(r["tokens_per_second"] for r in records), workers, records))
    _, workers, selected = max(worker_options, key=lambda item: item[0])
    save_json(args.run_dir / "tuning.json", dict(verified=True, batch_size=batch, workers=workers,
              selected=selected, candidates=candidates, manifest_sha256=digest_file(args.run_dir / "manifest.json"),
              code_sha256=code_fingerprint(), git=git_info()))
    announce(f"Selected training batch={batch}, data workers={workers}")


def training(args):
    checked_inputs(args)
    gate(args, "preflight")
    tuning = gate(args, "tuning")
    # Train each model in a separate process, avoiding stale allocations between models.
    for algorithm in ([args.algorithm] if args.algorithm else ALGORITHMS):
        if args.algorithm:
            from .token_training import train
            summary = train(args.run_dir / algorithm, tuning["batch_size"], args.device, tuning["workers"])
            if not summary["converged"]:
                raise RuntimeError("Training was interrupted before convergence; resume this stage")
        else:
            announce(f"Training {algorithm} until validation plateau")
            path = args.run_dir / algorithm / "training.log"
            with path.open("a") as log:
                subprocess.run([sys.executable, "-u", str(ROOT / "scripts/benchmark_sentencepiece.py"), "train",
                                "--run-dir", str(args.run_dir), "--algorithm", algorithm, "--device", args.device],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def inference_export(checkpoint, target, algorithm):
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    export = {key: state[key] for key in ("model", "model_config")}
    export.update(vocab_size=256 if algorithm == "byte" else 4096)
    export["source_checkpoint_sha256"] = digest_file(checkpoint)
    if algorithm != "byte":
        export["tokenizer_sha256"] = state["tokenizer_sha256"]
    else:
        export["model_name"] = state["model_name"]
    if target.exists():
        existing = torch.load(target, map_location="cpu", weights_only=False)
        if existing.get("source_checkpoint_sha256") == export["source_checkpoint_sha256"]:
            return target.stat().st_size
    from .trainer import _atomic_torch_save
    _atomic_torch_save(export, target)
    return target.stat().st_size


def evaluate(args):
    manifest = checked_inputs(args)
    gate(args, "preflight")
    for algorithm in ALGORITHMS:
        if not load_json(args.run_dir / algorithm / "training/summary.json")["converged"]:
            raise ValueError(f"{algorithm} has not reached its convergence criterion")
    for algorithm in ("byte", *ALGORITHMS):
        checkpoint = (Path(manifest["byte_checkpoint"]) if algorithm == "byte" else
                      args.run_dir / algorithm / "training/best.pt")
        target_dir = args.run_dir / "evaluation" / algorithm
        target_dir.mkdir(parents=True, exist_ok=True)
        package_bytes = inference_export(checkpoint, target_dir / "inference.pt", algorithm)
        # The actual coding checkpoint is the standalone inference package, so
        # the advertised package alone satisfies the container fingerprint.
        checkpoint = target_dir / "inference.pt"
        for split, raw in (("validation", args.run_dir / "splits/validation.bin"),
                           ("test", args.run_dir / "splits/test.bin"), ("full", Path(manifest["source"]))):
            if algorithm == "byte":
                tokens = raw.stat().st_size
            elif split == "full":
                tokens = load_json(args.run_dir / algorithm / "full_reversibility.json")["tokens"]
            else:
                tokens = load_json(str(args.run_dir / algorithm / f"{split}.tokens") + ".json")["tokens"]
            announce(f"Final evaluation {algorithm} {split}")
            result = roundtrip_job(args, target_dir / split, algorithm, checkpoint, raw, tokens)
            result["inference_package_bytes"] = package_bytes
            result["including_weights_bpb"] = (result["encode"]["compressed_bytes"] + package_bytes) * 8 / raw.stat().st_size
            save_json(target_dir / split / "roundtrip.json", result)
            report(args)


def report(args):
    rows = []
    training_records = {}
    evaluations = {}
    for algorithm in ALGORITHMS:
        path = args.run_dir / algorithm / "training/summary.json"
        if path.exists():
            training_records[algorithm] = load_json(path)
    for algorithm in ("byte", *ALGORITHMS):
        for split in ("test", "full"):
            path = args.run_dir / "evaluation" / algorithm / split / "roundtrip.json"
            if not path.exists():
                continue
            result = load_json(path)
            if "including_weights_bpb" not in result:
                continue
            enc, dec = result["encode"], result["decode"]
            evaluations[f"{algorithm}/{split}"] = result
            row = dict(algorithm=algorithm, split=split, original_bytes=enc["num_bytes"],
                       actual_bpb=enc["actual_bpb"], including_weights_bpb=result["including_weights_bpb"],
                       encode_kb_per_second=enc["encode_kb_per_second"], decode_kb_per_second=dec["decode_kb_per_second"],
                       compressed_bytes=enc["compressed_bytes"], inference_package_bytes=result["inference_package_bytes"],
                       verified=result["verified"], checkpoint_sha256=result["checkpoint_sha256"],
                       raw_sha256=result["raw_sha256"], backbone_params=3275212,
                       total_params=3455436 if algorithm == "byte" else 6158796,
                       training_source="historical_checkpoint" if algorithm == "byte" else "trained_from_scratch")
            if algorithm != "byte":
                training_summary = load_json(args.run_dir / algorithm / "training/summary.json")
                row.update(training_epochs=training_summary["epoch"], training_seconds=training_summary["elapsed_seconds"],
                           training_predicted_tokens=training_summary["predicted_tokens_seen"],
                           training_stop_reason=training_summary["stop_reason"],
                           best_validation_bpb=training_summary["best_bpb"], converged=training_summary["converged"])
            rows.append(row)
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (args.run_dir / "summary.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    complete = len(rows) == 6 and all(row["verified"] for row in rows)
    historical = dict(algorithm="byte", split="historical_full", original_bytes=1_000_000_000,
                      actual_bpb=1.337071088, encode_kb_per_second=361.7320167044504,
                      decode_kb_per_second=371.8051155889493,
                      source="experiments/test/20260916_160926_enwik9_full/summary.csv")
    stages = {name: load_json(args.run_dir / f"{name}.json") for name in ("prepared", "preflight", "tuning")
              if (args.run_dir / f"{name}.json").exists()}
    save_json(args.run_dir / "results.json", dict(complete=complete, results=rows, historical=historical,
              training=training_records, evaluations=evaluations, stages=stages,
              manifest=require_manifest(args), git=git_info(), code_sha256=code_fingerprint()))
    lines = ["# 三种表示方式的无损压缩实验", "", "主测试为独立的后 890 MB；完整 1 GB 仅作历史可比性补充。",
             "实际 BPB 包含分词器和整个压缩文件；含权重 BPB 再加入独立推理包。速度包含分词及文件读写，单位为 KB/s。",
             "", "| 范围 | 表示 | 实际 BPB | 含权重 BPB | 编码 KB/s | 解码 KB/s | 无损校验 |",
             "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        lines.append(f"| {row['split']} | {row['algorithm']} | {row['actual_bpb']:.6f} | {row['including_weights_bpb']:.6f} | "
                     f"{row['encode_kb_per_second']:.2f} | {row['decode_kb_per_second']:.2f} | 通过 |")
    lines += ["", "历史完整 1 GB 字节结果：BPB 1.337071088，编码 361.73 KB/s，解码 371.81 KB/s。",
              "字节组复用历史权重；两种子词模型按验证集早停。原始记录、训练曲线和校验信息保存在各模型目录。",
              "三组骨干均为 3,275,212 个参数；字节组总参数为 3,455,436，子词组分别为 6,158,796。",
              "验证 BPB 用于判断收敛；实际 BPB 来自完整压缩文件，两者口径不同。"]
    for algorithm, record in training_records.items():
        lines += ["", f"{algorithm}：完成 {record['epoch']} 轮，累计训练 {record['predicted_tokens_seen']:,} 个条件预测位置，"
                  f"用时 {record['elapsed_seconds']/3600:.2f} 小时，最佳验证 BPB {record['best_bpb']}；"
                  f"训练 batch {record['batch_size']}，停止原因 {record['stop_reason']}。",
                  f"[验证曲线]({algorithm}/training/bpb_vs_steps.png) · [完整训练记录]({algorithm}/training/history.json)"]
    lines += ["", "完整 JSON 保存预检耗时、环境、权重指纹及逐字节比较和 SHA256 证明。",
              "", "状态：" + ("全部完成。" if complete else "实验尚未全部完成；缺失结果不会用估算值填充。")]
    (args.run_dir / "report.md").write_text("\n".join(lines) + "\n")
    return complete


def dispatch(args):
    stages = {"preflight": preflight, "tune": tune, "train": training,
              "evaluate": evaluate, "report": report, "worker": worker}
    if args.stage == "all":
        if git_info()["dirty"]:
            raise RuntimeError("Commit the implementation before starting the formal pipeline")
        prepare(args)
        preflight(args)
        tune(args)
        training(args)
        evaluate(args)
        if not report(args):
            raise AssertionError("Incomplete experiment")
        announce("All three systems have completed both evaluation ranges")
    else:
        stages[args.stage](args)

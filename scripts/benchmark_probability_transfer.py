"""Paired BPE transport benchmark; FP32, FP16 or GPU integer quantization."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neural_compression.tokenization import digest_file, load_json, save_json


def fingerprint():
    paths = sorted((ROOT / "neural_compression").glob("*.py")) + [Path(__file__)]
    crate = ROOT / "native/integer_range"
    paths += sorted(crate.glob("*.toml")) + sorted(crate.glob("*.lock")) + sorted(crate.glob("src/*.rs"))
    native = ROOT / "neural_compression/_integer_range.so"
    if native.exists():
        paths.append(native)
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def experiment(vocab):
    return ROOT / ("experiments/tokenizers/enwik9_sp4096_20261019" if vocab == 4096
                   else f"experiments/tokenizers/bpe_vocab_scale_20261020/v{vocab}")


def worker(job_path):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from neural_compression import token_codec as tc
    from neural_compression.codec import verify_files
    from neural_compression.tokenization import LosslessTokenizer
    from neural_compression.token_experiment import git_info
    job = load_json(job_path)
    if fingerprint() != job["code_sha256"]:
        raise ValueError("Implementation changed; use a new experiment directory")
    for name in ("raw", "checkpoint", "tokenizer"):
        if digest_file(job[name]) != job[name + "_sha256"]:
            raise ValueError(f"Changed experiment input: {name}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; no performance measurement was performed")
    torch.set_num_threads(1)
    started = time.perf_counter()
    if job["operation"] == "decode":
        predictor = tc.load_token_decoder(job["checkpoint"], job["packed"], job["device"])
        if predictor.probability_transfer != job["probability_transfer"]:
            raise ValueError("Job and container transfer modes differ")
    else:
        predictor = tc.load_token_predictor(job["checkpoint"], job["device"], batch_size=1024,
                                            probability_transfer=job["probability_transfer"])
    predictor.synchronize()
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    tokenizer = LosslessTokenizer(job["tokenizer"], threads=24) if job["operation"] == "encode" else None
    tokenizer_load_seconds = time.perf_counter() - started
    warmup_seconds = tc.warmup(predictor, job["token_count"])
    torch.cuda.reset_peak_memory_stats(predictor.device)
    if job["operation"] == "encode":
        result = tc.encode(job["raw"], job["packed"], predictor, tokenizer)
        # The preserved baseline must still match the pre-existing FP32 stream.
        if job["probability_transfer"] == "fp32" and result["compressed_sha256"] != job["reference_sha256"]:
            raise AssertionError("FP32 baseline no longer matches the archived implementation")
    else:
        result = tc.decode(job["packed"], job["decoded"], predictor)
        started = time.perf_counter()
        if not verify_files(job["raw"], job["decoded"]) or digest_file(job["decoded"]) != job["raw_sha256"]:
            raise AssertionError("Lossless byte comparison or SHA256 verification failed")
        result.update(files_identical=True, external_verification_seconds=time.perf_counter()-started)
    result.update(job=job, model_load_seconds=load_seconds, tokenizer_load_seconds=tokenizer_load_seconds,
                  warmup_seconds=warmup_seconds, git=git_info(), process_id=os.getpid(),
                  gpu=torch.cuda.get_device_name(predictor.device), torch=torch.__version__, cuda=torch.version.cuda,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(predictor.device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(predictor.device))
    save_json(job_path.parent / "result.json", result)


def run_job(directory, job):
    directory.mkdir(parents=True, exist_ok=True)
    spec, result = directory / "job.json", directory / "result.json"
    if result.exists():
        if load_json(result)["job"] != job:
            raise ValueError("Existing result has different inputs; choose a new output directory")
        return
    if spec.exists():
        if load_json(spec) != job:
            raise ValueError("Existing job differs; choose a new output directory")
        # Retry only the artifact owned by this exact unfinished job.
        Path(job["packed"] if job["operation"] == "encode" else job["decoded"]).unlink(missing_ok=True)
    save_json(spec, job)
    with (directory / "worker.log").open("a", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--job", str(spec)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def summarize(output):
    manifest = load_json(output / "manifest.json")
    rows = []
    for vocab in manifest["vocab_sizes"]:
        for mode in manifest["probability_transfer"]:
            trials = []
            for repeat in range(1, manifest["repeats"]+1):
                directory = output / str(vocab) / mode / f"repeat{repeat}"
                enc, dec = [load_json(directory / op / "result.json") for op in ("encode", "decode")]
                for record in (enc, dec):
                    if (record["job"]["code_sha256"] != manifest["code_sha256"]
                            or record["job"]["raw_sha256"] != manifest["raw_sha256"]):
                        raise ValueError("Mixed implementation or input fingerprints")
                if not dec["files_identical"] or not dec["checksum_verified"]:
                    raise AssertionError("Unverified decoding")
                # Recheck retained artifacts independently of worker results.
                if (digest_file(enc["job"]["packed"]) != enc["compressed_sha256"]
                        or digest_file(dec["job"]["decoded"]) != manifest["raw_sha256"]):
                    raise AssertionError("Changed benchmark output")
                trials.append((enc, dec))
            if len({enc["compressed_sha256"] for enc, _ in trials}) != 1:
                raise AssertionError("Repeated encoding is not deterministic")
            first = trials[0][0]
            row = dict(vocab_size=vocab, probability_transfer=mode, verified=True,
                       actual_bpb=first["actual_bpb"], compressed_bytes=first["compressed_bytes"],
                       compressed_sha256=first["compressed_sha256"], repeats=len(trials))
            for index, operation in enumerate(("encode", "decode")):
                seconds = [trial[index][operation + "_seconds"] for trial in trials]
                transferred = {trial[index]["probability_transfer_bytes"] for trial in trials}
                if len(transferred) != 1:
                    raise AssertionError("Inconsistent probability transfer counts")
                row.update({operation + "_seconds": statistics.median(seconds),
                            operation + "_min_seconds": min(seconds), operation + "_max_seconds": max(seconds),
                            operation + "_kb_per_second": 10_000 / statistics.median(seconds),
                            operation + "_probability_transfer_bytes": transferred.pop()})
            rows.append(row)
    decisions = []
    for vocab in manifest["vocab_sizes"]:
        by_mode = {r["probability_transfer"]: r for r in rows if r["vocab_size"] == vocab}
        if "fp32" not in by_mode:
            continue
        candidate_mode = "gpu_int" if "gpu_int" in by_mode else "fp16"
        if candidate_mode not in by_mode:
            continue
        base, candidate = by_mode["fp32"], by_mode[candidate_mode]
        growth = candidate["actual_bpb"] / base["actual_bpb"] - 1
        decision = dict(vocab_size=vocab, candidate_mode=candidate_mode, bpb_increase_percent=100*growth)
        improvements = []
        for op in ("encode", "decode"):
            if candidate_mode == "fp16" and candidate[op + "_probability_transfer_bytes"] * 2 != base[op + "_probability_transfer_bytes"]:
                raise AssertionError("FP16 did not halve the probability payload")
            if candidate_mode == "gpu_int":
                expected_num, expected_den = (2, vocab) if op == "encode" else (vocab+1, vocab)
                if (candidate[op + "_probability_transfer_bytes"] * expected_den !=
                        base[op + "_probability_transfer_bytes"] * expected_num):
                    raise AssertionError("Unexpected integer transfer payload")
            decision[op + "_speedup_percent"] = 100*(base[op + "_seconds"]/candidate[op + "_seconds"]-1)
            # Conservative adoption gate: all three candidate times beat all baseline times.
            clear = candidate[op + "_max_seconds"] < base[op + "_min_seconds"]
            decision[op + "_clearly_faster"] = clear
            improvements.append(clear)
        decision["accepted"] = growth <= .01 and all(improvements)
        decisions.append(decision)
    accepted = manifest["repeats"] >= 3 and len(decisions) == 3 and all(d["accepted"] for d in decisions)
    summary = dict(manifest=manifest, results=rows, comparisons=decisions,
                   fp16_default_eligible=accepted and all(d["candidate_mode"] == "fp16" for d in decisions),
                   gpu_int_default_eligible=accepted and all(d["candidate_mode"] == "gpu_int" for d in decisions),
                   acceptance="BPB <= +1%; both timing ranges strictly faster")
    save_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("benchmark", "worker", "summary"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--vocab-sizes", nargs="+", type=int, choices=(4096, 8192, 16384), default=[4096, 8192, 16384])
    parser.add_argument("--probability-transfer", nargs="+", choices=("fp32", "fp16", "gpu_int"), default=["fp32", "fp16"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.stage == "worker":
        if args.job is None:
            parser.error("--job is required")
        worker(args.job)
        return
    if args.output is None:
        parser.error("--output is required")
    output = args.output.resolve()
    if args.stage == "summary":
        summarize(output)
        return
    if args.repeats < 1 or len(set(args.vocab_sizes)) != len(args.vocab_sizes) or len(set(args.probability_transfer)) != len(args.probability_transfer):
        parser.error("positive repeats and unique vocabularies/modes required")
    raw = experiment(4096) / "splits/validation.bin"
    if raw.stat().st_size != 10_000_000:
        raise ValueError("Expected exactly 10 MB")
    raw_sha = digest_file(raw)
    with (ROOT / "datasets/raw/enwik9").open("rb") as source:
        source.seek(100_000_000)
        if hashlib.sha256(source.read(10_000_000)).hexdigest() != raw_sha:
            raise ValueError("Validation file does not match the required enwik9 byte range")
    inputs = {}
    for vocab in args.vocab_sizes:
        exp = experiment(vocab)
        if digest_file(exp / "splits/validation.bin") != raw_sha:
            raise ValueError("Vocabulary groups do not share the same validation bytes")
        checkpoint, tokenizer = exp / "evaluation/bpe/inference.pt", exp / "bpe/tokenizer.model"
        reference = (exp / "evaluation/bpe/validation/encode/result.json" if vocab == 4096
                     else exp / "evaluation/bpe/validation/repeat1/encode/result.json")
        inputs[str(vocab)] = dict(checkpoint=str(checkpoint), tokenizer=str(tokenizer),
                                 checkpoint_sha256=digest_file(checkpoint), tokenizer_sha256=digest_file(tokenizer),
                                 token_count=load_json(exp / "bpe/validation.tokens.json")["tokens"],
                                 reference_sha256=load_json(reference)["compressed_sha256"])
    manifest = dict(code_sha256=fingerprint(), raw=str(raw), raw_sha256=raw_sha,
                    offset_bytes=100_000_000, num_bytes=10_000_000, vocab_sizes=args.vocab_sizes,
                    probability_transfer=args.probability_transfer, repeats=args.repeats, device=args.device,
                    batch_size=1024, tokenizer_threads=24, torch_threads=1, inputs=inputs)
    path = output / "manifest.json"
    if path.exists() and load_json(path) != manifest:
        raise ValueError("Experiment differs; use a new output directory")
    save_json(path, manifest)
    for repeat in range(args.repeats):
        shift = repeat % len(args.vocab_sizes)
        order = args.vocab_sizes[shift:] + args.vocab_sizes[:shift]
        modes = args.probability_transfer if repeat % 2 == 0 else list(reversed(args.probability_transfer))
        for vocab in order:
            for mode in modes:
                directory = output / str(vocab) / mode / f"repeat{repeat+1}"
                common = dict(inputs[str(vocab)], vocab_size=vocab, probability_transfer=mode,
                              code_sha256=manifest["code_sha256"], raw=str(raw), raw_sha256=raw_sha,
                              packed=str(directory / "compressed.m3rc"), decoded=str(directory / "decoded.bin"),
                              device=args.device, repeat=repeat+1)
                for operation in ("encode", "decode"):
                    print(f"BPE {vocab} {mode} repeat {repeat+1}/{args.repeats}: {operation}", flush=True)
                    run_job(directory / operation, dict(common, operation=operation))
    summarize(output)


if __name__ == "__main__":
    main()

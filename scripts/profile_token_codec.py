"""Independent-process, lossless A/B profiling of the trained subword codecs."""
import argparse
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


def fingerprint():
    paths = [ROOT / "mamba3_byte_compress" / name for name in
             ("token_codec.py", "codec_graph.py", "tokenization.py")]
    paths.append(Path(__file__))
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def worker(job_path):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from mamba3_byte_compress import token_codec as tc
    from mamba3_byte_compress import codec as c
    from mamba3_byte_compress.tokenization import LosslessTokenizer, digest_file, save_json
    from mamba3_byte_compress.token_experiment import git_info
    job = json.loads(job_path.read_text())
    if job["code_sha256"] != fingerprint():
        raise ValueError("Implementation changed before worker startup")
    for field in ("raw", "checkpoint", "tokenizer"):
        if digest_file(job[field]) != job[field + "_sha256"]:
            raise ValueError(f"Changed {field}")
    torch.set_num_threads(1)
    start = time.perf_counter()
    predictor = tc.load_token_predictor(job["checkpoint"], "cuda:0", 1024,
                                        device_checks=job["variant"] == "optimized")
    predictor.synchronize()
    load_seconds = time.perf_counter() - start
    warmup_seconds = tc.warmup(predictor, job["token_count"])
    if job["profile"]:
        predictor.profile_timings = dict(upload=0., model_probability=0., download=0.)
    torch.cuda.reset_peak_memory_stats()
    optimized = job["variant"] == "optimized"
    if job["operation"] == "encode":
        tokenizer = LosslessTokenizer(job["tokenizer"], job["threads"])
        result = tc.encode(job["raw"], job["packed"], predictor, tokenizer,
                           profile=job["profile"], optimized=optimized)
        if result["compressed_sha256"] != job["expected_compressed_sha256"]:
            raise AssertionError("Compressed stream differs from the original completed experiment")
    else:
        result = tc.decode(job["packed"], job["decoded"], predictor,
                           profile=job["profile"], optimized=optimized)
        start = time.perf_counter()
        if not c.verify_files(job["raw"], job["decoded"]):
            raise AssertionError("Decoded bytes differ from original")
        if digest_file(job["decoded"]) != job["raw_sha256"]:
            raise AssertionError("Decoded SHA256 differs from original")
        result.update(files_identical=True, external_verification_seconds=time.perf_counter()-start)
    result.update(job=job, model_load_seconds=load_seconds, warmup_seconds=warmup_seconds,
                  probability_validation_location="gpu" if optimized else "cpu_encode_only",
                  predictor_phases_seconds=getattr(predictor, "profile_timings", None),
                  predictor_timing_mode=("cuda_events_overlapping_cpu" if optimized and job["operation"] == "encode"
                                         else "synchronized_host_wall"),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  git=git_info(), process_id=os.getpid())
    save_json(job_path.parent / "result.json", result)


def run_job(directory, job):
    from mamba3_byte_compress.tokenization import save_json
    directory.mkdir(parents=True, exist_ok=True)
    spec = directory / "job.json"
    result = directory / "result.json"
    if result.exists():
        record = json.loads(result.read_text())
        if record["job"] != job:
            raise ValueError(f"Use a new output directory; previous job differs: {directory}")
        return record
    save_json(spec, job)
    with (directory / "worker.log").open("a") as log:
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                        "--job", str(spec)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(result.read_text())


def summarize(output):
    from mamba3_byte_compress.tokenization import save_json
    groups = {}
    for path in sorted(output.glob("*/*/repeat*/encode/result.json")):
        enc = json.loads(path.read_text())
        dp = path.parent.parent / "decode/result.json"
        if not dp.exists():
            continue
        dec = json.loads(dp.read_text())
        key = (enc["job"]["algorithm"], enc["job"]["variant"])
        groups.setdefault(key, []).append((enc, dec))
    rows = []
    for (algorithm, variant), trials in groups.items():
        row = dict(algorithm=algorithm, variant=variant, repeats=len(trials),
                   verified=all(d["files_identical"] for _, d in trials),
                   compressed_sha256=trials[0][0]["compressed_sha256"],
                   actual_bpb=trials[0][0]["actual_bpb"])
        if len({e["compressed_sha256"] for e, _ in trials}) != 1:
            raise AssertionError("Repeated coding streams differ")
        for operation, index in (("encode", 0), ("decode", 1)):
            values = [trial[index][operation + "_seconds"] for trial in trials]
            row[operation + "_seconds"] = dict(median=statistics.median(values),
                                               min=min(values), max=max(values), samples=values)
            row[operation + "_kb_per_second"] = statistics.median(
                trial[index][operation + "_kb_per_second"] for trial in trials)
            if trials[0][index]["job"]["profile"]:
                row[operation + "_phases"] = trials[0][index]["phases_seconds"]
                row[operation + "_predictor_phases"] = trials[0][index]["predictor_phases_seconds"]
        rows.append(row)
    for algorithm in {r["algorithm"] for r in rows}:
        pair = [r for r in rows if r["algorithm"] == algorithm]
        if len(pair) == 2 and pair[0]["compressed_sha256"] != pair[1]["compressed_sha256"]:
            raise AssertionError("Optimization changed the compressed stream")
    save_json(output / "summary.json", dict(results=rows, code_sha256=fingerprint()))
    print(json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("profile", "benchmark", "summary", "worker"))
    parser.add_argument("--experiment", type=Path,
                        default=ROOT / "experiments/tokenizers/enwik9_sp4096_20261019")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--algorithms", nargs="+", choices=("bpe", "unigram"), default=["bpe", "unigram"])
    parser.add_argument("--variants", nargs="+", choices=("baseline", "optimized"), default=["baseline", "optimized"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    if args.stage == "worker":
        worker(args.job)
        return
    if args.output is None:
        parser.error("--output is required")
    args.output = args.output.resolve()
    if args.stage == "summary":
        summarize(args.output)
        return
    from mamba3_byte_compress.tokenization import digest_file
    exp = args.experiment.resolve()
    raw = exp / "splits/validation.bin"
    for algorithm in args.algorithms:
        for repeat in range(args.repeats):
            # Alternate ordering to reduce drift between variants.
            variants = args.variants if repeat % 2 == 0 else list(reversed(args.variants))
            for variant in variants:
                directory = args.output / algorithm / variant / f"repeat{repeat+1}"
                checkpoint = exp / "evaluation" / algorithm / "inference.pt"
                tokenizer = exp / algorithm / "tokenizer.model"
                meta = json.loads((exp / algorithm / "validation.tokens.json").read_text())
                threads = json.loads((exp / algorithm / "threads.json").read_text())["threads"]
                job = dict(algorithm=algorithm, variant=variant, profile=args.stage == "profile",
                           code_sha256=fingerprint(), raw=str(raw), checkpoint=str(checkpoint),
                           tokenizer=str(tokenizer), token_count=meta["tokens"], threads=threads,
                           packed=str(directory / "compressed.m3rc"), decoded=str(directory / "decoded.bin"))
                reference = json.loads((exp / "evaluation" / algorithm / "validation/encode/result.json").read_text())
                job["expected_compressed_sha256"] = reference["compressed_sha256"]
                for field in ("raw", "checkpoint", "tokenizer"):
                    job[field + "_sha256"] = digest_file(job[field])
                print(f"{algorithm} {variant} repeat {repeat+1}/{args.repeats}", flush=True)
                for operation in ("encode", "decode"):
                    run_job(directory / operation, dict(job, operation=operation))
    summarize(args.output)


if __name__ == "__main__":
    main()

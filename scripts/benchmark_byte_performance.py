"""Independent-process paired byte codec benchmark with original stream proof."""
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
    names = ("codec.py", "codec_batched_v4.py", "codec_graph.py")
    return hashlib.sha256(b"".join((ROOT / "neural_compression" / n).read_bytes()
                                   for n in names) + Path(__file__).read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def worker(path):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from neural_compression import codec as c
    from neural_compression.codec_batched_v4 import encode_batched_v4_file, decode_batched_v4_file
    from neural_compression.token_experiment import git_info
    job = json.loads(path.read_text())
    assert fingerprint() == job["code_sha256"], "Implementation changed"
    for name in ("raw", "checkpoint", "reference"):
        assert c.sha256_file(job[name]) == job[name + "_sha256"], name
    torch.set_num_threads(1)
    started = time.perf_counter()
    p = c.load_predictor(job["checkpoint"], batch_size=1024)
    p.synchronize()
    load = time.perf_counter() - started
    started = time.perf_counter()
    blocks = (Path(job["raw"]).stat().st_size + 2047) // 2048
    with c._coding_mode(p):
        for size in sorted({min(blocks, 1024), blocks % 1024} - {0}):
            p.reset(size)
            p.step(np.zeros(size, dtype=np.int32))
            if job["variant"] == "optimized":
                p.prepare_prefetch()
    p.synchronize()
    warmup = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    opt = job["variant"] == "optimized"
    if job["operation"] == "encode":
        result = encode_batched_v4_file(job["raw"], job["packed"], p, optimized=opt)
        result["compressed_sha256"] = c.sha256_file(job["packed"])
        result["reference_bytes_identical"] = Path(job["packed"]).read_bytes() == Path(job["reference"]).read_bytes()
        assert result["reference_bytes_identical"]
    else:
        result = decode_batched_v4_file(job["packed"], job["decoded"], p, optimized=opt)
        started = time.perf_counter()
        result["files_identical"] = c.verify_files(job["raw"], job["decoded"])
        assert result["files_identical"]
        assert result["decoded_sha256"] == job["raw_sha256"]
        result["external_verification_seconds"] = time.perf_counter() - started
    result.update(job=job, git=git_info(), model_load_seconds=load, warmup_seconds=warmup,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    save(path.parent / "result.json", result)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    if args.job:
        worker(args.job)
        return
    if not args.output or args.repeats < 1:
        parser.error("--output and positive --repeats required")
    exp = ROOT / "experiments/tokenizers/enwik9_sp4096_20261019"
    base = dict(raw=str(exp / "splits/validation.bin"),
                checkpoint=str(exp / "evaluation/byte/inference.pt"),
                reference=str(exp / "evaluation/byte/validation/compressed.m3rc"),
                code_sha256=fingerprint())
    from neural_compression.codec import sha256_file
    for name in ("raw", "checkpoint", "reference"):
        base[name + "_sha256"] = sha256_file(base[name])
    results = []
    for repeat in range(1, args.repeats + 1):
        order = ("baseline", "optimized") if repeat % 2 else ("optimized", "baseline")
        for variant in order:
            folder = args.output.resolve() / variant / f"repeat{repeat}"
            print(variant, repeat, flush=True)
            for op in ("encode", "decode"):
                directory = folder / op
                directory.mkdir(parents=True, exist_ok=True)
                job = dict(base, variant=variant, operation=op, repeat=repeat,
                           packed=str(folder / "compressed.m3rc"), decoded=str(folder / "decoded.bin"))
                spec = directory / "job.json"
                result = directory / "result.json"
                if result.exists():
                    assert json.loads(result.read_text())["job"] == job, "Use a new output directory"
                else:
                    save(spec, job)
                    with (directory / "worker.log").open("w") as log:
                        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--job", str(spec)],
                                       stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, check=True)
                results.append(json.loads(result.read_text()))
    rows = []
    for variant in ("baseline", "optimized"):
        row = dict(variant=variant, repeats=args.repeats)
        for op in ("encode", "decode"):
            ds = [r for r in results if r["job"]["variant"] == variant and r["job"]["operation"] == op]
            times = [r[op + "_seconds"] for r in ds]
            row[op + "_seconds"] = dict(samples=times, median=statistics.median(times), min=min(times), max=max(times))
            row[op + "_kb_per_second"] = 10000 / statistics.median(times)
        rows.append(row)
    summary = dict(code_sha256=fingerprint(), configuration=base, results=rows, runs=results)
    save(args.output / "summary.json", summary)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()

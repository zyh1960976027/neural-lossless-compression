"""当前 BPE 4096 GPU 整数基线：预热后独立进程编解码及三次中位数。"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neural_compression.io import load_json, save_json, sha256_file, git_info
from neural_compression.config import DEFAULT_CHECKPOINT, DEFAULT_TOKENIZER


def worker(job_path):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from neural_compression import codec
    from neural_compression.data import LosslessTokenizer
    from neural_compression.io import verify_files

    job = load_json(job_path)
    for name in ("input", "checkpoint", "tokenizer"):
        if sha256_file(job[name]) != job[name + "_sha256"]:
            raise ValueError(f"测速输入已变化：{name}")
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("基线测速需要真实 CUDA GPU")
    if job["operation"] == "encode":
        predictor = codec.load_predictor(
            job["checkpoint"], job["device"], job["batch_size"]
        )
    else:
        predictor = codec.load_decoder(
            job["checkpoint"], job["compressed"], job["device"]
        )
    warmup = codec.warmup(predictor, job["tokens"])
    if job["operation"] == "encode":
        result = codec.encode(
            job["input"],
            job["compressed"],
            predictor,
            LosslessTokenizer(job["tokenizer"], threads=24),
        )
    else:
        result = codec.decode(job["compressed"], job["decoded"], predictor)
        result["files_identical"] = verify_files(job["input"], job["decoded"])
    result.update(
        warmup_seconds=warmup,
        process_id=os.getpid(),
        job=job,
        graph=predictor.execution_metrics(),
    )
    save_json(Path(job_path).with_name("result.json"), result)


def benchmark(args):
    from neural_compression.data import (
        LosslessTokenizer,
        tokenize_file,
        require_baseline_tokenizer,
    )

    if args.repeats < 1:
        raise ValueError("重复次数必须为正")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = {
        name: str(getattr(args, name).resolve())
        for name in ("input", "checkpoint", "tokenizer")
    }
    hashes = {name + "_sha256": sha256_file(value) for name, value in inputs.items()}
    adapter = require_baseline_tokenizer(LosslessTokenizer(args.tokenizer, threads=24))
    tokenization = tokenize_file(adapter, args.input)
    jobs = dict(
        **inputs,
        **hashes,
        tokens=tokenization["tokens"],
        batch_size=args.batch_size,
        device=args.device,
    )
    trials = []
    for trial in range(1, args.repeats + 1):
        results = []
        for operation in ("encode", "decode"):
            directory = output / f"trial{trial}" / operation
            directory.mkdir(parents=True)
            job = dict(
                jobs,
                operation=operation,
                trial=trial,
                compressed=str(output / f"trial{trial}/compressed.m3rc"),
                decoded=str(output / f"trial{trial}/decoded.bin"),
            )
            path = directory / "job.json"
            save_json(path, job)
            with (directory / "worker.log").open("x") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        str(path),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            result = load_json(directory / "result.json")
            results.append(result)
            print(
                f'{trial}/{args.repeats} {operation}: {result[operation + "_seconds"]:.3f} s',
                flush=True,
            )
        trials.append(dict(zip(("encode", "decode"), results)))
    streams = {t["encode"]["compressed_sha256"] for t in trials}
    if len(streams) != 1 or not all(t["decode"]["files_identical"] for t in trials):
        raise AssertionError("重复编码或逐字节恢复校验失败")
    result = dict(
        inputs=jobs,
        git=git_info(),
        repeats=args.repeats,
        trials=trials,
        compressed_sha256=streams.pop(),
        verified=True,
    )
    for op in ("encode", "decode"):
        result[op + "_kb_per_second"] = statistics.median(
            t[op][op + "_kb_per_second"] for t in trials
        )
    if args.reference:
        reference = load_json(args.reference)
        for key in (
            "input_sha256",
            "checkpoint_sha256",
            "tokenizer_sha256",
            "batch_size",
            "tokens",
        ):
            if reference["inputs"][key] != result["inputs"][key]:
                raise ValueError("对照输入与基线不一致：" + key)
        if reference["compressed_sha256"] != result["compressed_sha256"]:
            raise AssertionError("重构前后码流不一致")
        result["regression_percent"] = {
            op: 100
            * (1 - result[op + "_kb_per_second"] / reference[op + "_kb_per_second"])
            for op in ("encode", "decode")
        }
    save_json(output / "summary.json", result)
    if args.reference and max(result["regression_percent"].values()) > 5:
        raise AssertionError("中位吞吐回退超过 5%，请查看原始记录并排查")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.input is None or args.output_dir is None:
        parser.error("需要 --input 和 --output-dir")
    else:
        benchmark(args)


if __name__ == "__main__":
    main()

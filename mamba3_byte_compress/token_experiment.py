"""Resumable experiment stages; artifact hashes bind every dependent stage."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from .tokenization import (LosslessTokenizer, digest_file, load_json, raw_batches,
                           save_json, tokenize_file, train_tokenizer)

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ("bpe", "unigram")
SPLITS = {"train": (0, 100_000_000), "validation": (100_000_000, 10_000_000),
          "test": (110_000_000, 890_000_000)}


def announce(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def git_info():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=normal"))}


def require_manifest(args):
    return load_json(args.run_dir / "manifest.json")


def checked_result(path, **expected):
    result = load_json(path)
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Stale artifact {path}: {key} mismatch")
    return result


def extract(source, target, offset, count):
    if target.exists():
        if target.stat().st_size != count:
            raise ValueError(f"Wrong split size: {target}")
        return
    temporary = target.with_suffix(".tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(source, "rb") as src, temporary.open("wb") as dst:
        src.seek(offset)
        while count:
            chunk = src.read(min(count, 8 * 1024 * 1024))
            if not chunk:
                raise ValueError("Truncated source")
            dst.write(chunk)
            count -= len(chunk)
    os.replace(temporary, target)


def tokenizer_edge_checks(tokenizer):
    rng = np.random.default_rng(2026)
    samples = [b"", b"\x00", bytes(range(256)), b" \t\n\r\n  ",
               "a▁b中文😀".encode(), b"\xff\xc3\xe4\xbd", b"x" * 65539]
    samples += [rng.integers(0, 256, n, dtype=np.uint8).tobytes()
                for n in (1, 2047, 2048, 2049, 4097)]
    for raw in samples:
        tokens = tokenizer.encode(raw)
        if tokenizer.decode(tokens) != raw:
            raise AssertionError("Tokenizer edge-case roundtrip failed")
        if len(tokens) > 2048:
            decoded = b"".join(tokenizer.decode(tokens[i:i+2048]) for i in range(0, len(tokens), 2048))
            if decoded != raw:
                raise AssertionError("Token block boundary lost bytes")
    return len(samples)


def tune_threads(tokenizer, validation):
    rows = [raw for batch in raw_batches(validation) for raw in batch]
    tokenizer.threads = 1
    reference = tokenizer.encode_batch(rows)
    fingerprint = hashlib.sha256(b"".join(ids.astype("<u2").tobytes() for ids in reference)).hexdigest()
    measurements = []
    available = len(os.sched_getaffinity(0))
    for threads in [t for t in (1, 2, 4, 8, 12, 24) if t <= available]:
        tokenizer.threads = threads
        times = []
        for _ in range(3):
            start = time.perf_counter()
            encoded = tokenizer.encode_batch(rows)
            times.append(time.perf_counter()-start)
            digest = hashlib.sha256(b"".join(ids.astype("<u2").tobytes() for ids in encoded)).hexdigest()
            if digest != fingerprint:
                raise AssertionError("Thread count changed tokenization")
        measurements.append({"threads": threads, "seconds": times, "median_seconds": statistics.median(times)})
    best = min(measurements, key=lambda row: row["median_seconds"])["threads"]
    tokenizer.threads = best
    return {"threads": best, "measurements": measurements, "token_sha256": fingerprint}


def prepare(args):
    args.run_dir.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    if source.stat().st_size != 1_000_000_000:
        raise ValueError("Formal experiment requires the exact 1 GB enwik9 file")
    manifest_path = args.run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = checked_result(manifest_path, source=str(source), source_sha256=digest_file(source),
                                  byte_checkpoint_sha256=digest_file(args.byte_checkpoint))
    else:
        manifest = dict(version=1, source=str(source), source_sha256=digest_file(source),
                        byte_checkpoint=str(args.byte_checkpoint.resolve()),
                        byte_checkpoint_sha256=digest_file(args.byte_checkpoint),
                        splits=SPLITS, vocab_size=4096, block_size=2048, codec_batch_size=1024,
                        created_git=git_info(), created_at=time.time())
        save_json(manifest_path, manifest)
    for split, (offset, count) in SPLITS.items():
        extract(source, args.run_dir / "splits" / f"{split}.bin", offset, count)
    for algorithm in ALGORITHMS:
        directory = args.run_dir / algorithm
        directory.mkdir(exist_ok=True)
        model = directory / "tokenizer.model"
        if not model.exists():
            announce(f"Training SentencePiece {algorithm}")
            metadata = train_tokenizer(source, model, algorithm, threads=min(24, len(os.sched_getaffinity(0))))
            save_json(directory / "tokenizer.json", metadata)
        checked_result(directory / "tokenizer.json", sha256=digest_file(model), algorithm=algorithm,
                       vocab_size=4096, train_bytes=100_000_000)
        tokenizer = LosslessTokenizer(model)
        edge_cases = tokenizer_edge_checks(tokenizer)
        thread_path = directory / "threads.json"
        if not thread_path.exists():
            announce(f"Tuning {algorithm} SentencePiece threads")
            result = tune_threads(tokenizer, args.run_dir / "splits" / "validation.bin")
            result["tokenizer_sha256"] = tokenizer.sha256
            save_json(thread_path, result)
        tokenizer.threads = checked_result(thread_path, tokenizer_sha256=tokenizer.sha256)["threads"]
        for split, (_, count) in SPLITS.items():
            cache = directory / f"{split}.tokens"
            metadata_path = Path(str(cache) + ".json")
            if not metadata_path.exists():
                announce(f"Verifying and caching {algorithm} {split} ({count} bytes)")
                tokenize_file(tokenizer, args.run_dir / "splits" / f"{split}.bin", output=cache)
            metadata = checked_result(metadata_path, tokenizer_sha256=tokenizer.sha256, raw_bytes=count,
                                      verified=True, token_sha256=digest_file(cache))
            if metadata["original_sha256"] != digest_file(args.run_dir / "splits" / f"{split}.bin"):
                raise ValueError("Split source mismatch")
        full_path = directory / "full_reversibility.json"
        if not full_path.exists():
            announce(f"Verifying {algorithm} full 1 GB stream")
            full = tokenize_file(tokenizer, source)
            full["edge_cases"] = edge_cases
            save_json(full_path, full)
        checked_result(full_path, tokenizer_sha256=tokenizer.sha256,
                       original_sha256=manifest["source_sha256"], verified=True)
    save_json(args.run_dir / "prepared.json", {"verified": True, "manifest_sha256": digest_file(manifest_path),
                                               "git": git_info(), "completed_at": time.time()})
    announce("All tokenizers verified, including the complete enwik9 file")


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["prepare", "preflight", "tune", "train", "evaluate", "report", "all", "worker"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/raw/enwik9")
    parser.add_argument("--byte-checkpoint", type=Path, default=ROOT / "checkpoints/best-3M.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--algorithm", choices=ALGORITHMS)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    if args.stage == "prepare":
        prepare(args)
    else:
        from .token_pipeline import dispatch
        dispatch(args)

"""One enwik9 prefix encode and separate-process decode, with configurable size/batch."""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def next_log_path(output, operation):
    path = output / f"{operation}.log"
    if not path.exists():
        return path
    attempt = 1
    while (output / f"{operation}.resume{attempt}.log").exists():
        attempt += 1
    return output / f"{operation}.resume{attempt}.log"


def completed_stage(output, operation, artifact, sample_bytes):
    """Return saved metrics for a resumable completed stage, or None."""
    output, artifact = Path(output), Path(artifact)
    metrics_path = output / f"{operation}.json"
    if metrics_path.is_file() and artifact.is_file():
        existing = json.loads(metrics_path.read_text())
        if existing.get("num_bytes") != sample_bytes:
            raise ValueError(f"Existing {operation} result does not match sample size.")
        digest_field = "compressed_sha256" if operation == "encode" else "decoded_sha256"
        if digest_field in existing:
            from mamba3_byte_compress.codec import sha256_file

            if sha256_file(artifact) != existing[digest_field]:
                raise ValueError(f"Existing {operation} artifact checksum mismatch.")
        return existing
    if metrics_path.exists() or artifact.exists():
        raise FileExistsError(
            f"Incomplete {operation} stage has conflicting output: {metrics_path} or {artifact}"
        )
    return None


def archive_incomplete_stage(output, operation, artifact):
    """Preserve interrupted outputs before retrying the exact stage."""
    output, artifact = Path(output), Path(artifact)
    paths = [output / f"{operation}.json", artifact]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return []
    archive = output / "incomplete"
    archive.mkdir(exist_ok=True)
    attempt = 1
    while any((archive / f"{operation}.attempt{attempt}.{path.name}").exists()
              for path in existing):
        attempt += 1
    moved = []
    for path in existing:
        target = archive / f"{operation}.attempt{attempt}.{path.name}"
        path.rename(target)
        moved.append(target)
    return moved


def sample_paths(args):
    label = (f"{args.sample_bytes // 1_000_000}mb" if args.sample_bytes % 1_000_000 == 0
             else f"{args.sample_bytes}bytes")
    end = args.source_offset + args.sample_bytes
    raw = (args.source.resolve() if args.direct_source
           else args.output / f"enwik9_{args.source_offset}_{end}.bin")
    return (raw,
            args.output / f"enwik9_{args.source_offset}_{end}_{label}.m3rc")


def extract_slice(source, target, offset, count, *, chunk_bytes=8 * 1024 * 1024):
    """Copy one byte range without materializing the complete sample in RAM."""
    source, target = Path(source), Path(target)
    if offset < 0 or count <= 0 or chunk_bytes <= 0:
        raise ValueError("Invalid source range or copy chunk size.")
    remaining = count
    created = False
    try:
        with source.open("rb") as input_handle:
            with target.open("xb") as output_handle:
                created = True
                input_handle.seek(offset)
                while remaining:
                    data = input_handle.read(min(remaining, chunk_bytes))
                    if not data:
                        raise ValueError(f"enwik9 source must contain at least {offset + count} bytes.")
                    output_handle.write(data)
                    remaining -= len(data)
    except BaseException:
        if created and target.exists():
            target.unlink()
        raise


def stage(args):
    import numpy as np
    import torch
    from mamba3_byte_compress import codec as c

    # Keep host-side entropy coding measurements comparable across runs.
    torch.set_num_threads(1)
    raw, packed = sample_paths(args)
    inference = {
        "v3": "batched-step",
        "v4": "batched-step-v4",
        "v5": "batched-step-v5",
        "v6": "mtp4-batched-sequence",
    }[args.format]
    settings = ({"inference": inference, "batch_size": args.batch_size}
                if args.stage == "encode" else c.read_codec_settings(packed))
    if args.format == "v6":
        settings["execution"] = args.execution
    started = time.perf_counter()
    predictor = c.load_predictor(args.checkpoint, args.device, **settings)
    predictor.collect_phase_timings = args.phase_timing
    if args.format == "v5":
        predictor.use_graph = args.execution == "graph"
    predictor.synchronize()
    load_seconds = time.perf_counter() - started

    def memory_snapshot():
        return {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(predictor.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(predictor.device),
            "end_allocated_bytes": torch.cuda.memory_allocated(predictor.device),
            "end_reserved_bytes": torch.cuda.memory_reserved(predictor.device),
        }

    memory_phases = {"model_load": memory_snapshot()}
    print(f"{args.stage}: model loaded in {load_seconds:.3f}s", flush=True)
    blocks = (args.sample_bytes + c.BLOCK_SIZE - 1) // c.BLOCK_SIZE
    groups = [min(settings["batch_size"], blocks - offset)
              for offset in range(0, blocks, settings["batch_size"])]
    torch.cuda.reset_peak_memory_stats(predictor.device)
    started = time.perf_counter()
    with c._coding_mode(predictor):
        for size in (() if args.cold else dict.fromkeys(groups)):
            predictor.reset(size)
            if args.format == "v6":
                predictor.start(np.zeros(size, dtype=np.uint8))
                predictor.advance(np.zeros((size, 4), dtype=np.uint8))
            else:
                for _ in range(2):
                    predictor.step(np.zeros(size, dtype=np.uint8))
        if not args.cold:
            predictor.reset(groups[0])
        predictor.synchronize()
    warmup_seconds = time.perf_counter() - started
    memory_phases["warmup"] = memory_snapshot()
    torch.cuda.reset_peak_memory_stats(predictor.device)
    print(f"{args.stage}: warmup {warmup_seconds:.3f}s; starting timed codec", flush=True)
    if args.stage == "encode":
        result = c.encode_file(raw, packed, predictor, file_format=args.format)
        result["compressed_sha256"] = c.sha256_file(packed)
    else:
        result = c.decode_file(packed, args.output / "decoded.bin", predictor)
    memory_phases["codec"] = memory_snapshot()
    result.update(
        codec_peak_allocated_bytes=memory_phases["codec"]["peak_allocated_bytes"],
        codec_peak_reserved_bytes=memory_phases["codec"]["peak_reserved_bytes"],
        run_peak_allocated_bytes=max(p["peak_allocated_bytes"] for p in memory_phases.values()),
        run_peak_reserved_bytes=max(p["peak_reserved_bytes"] for p in memory_phases.values()),
        cuda_memory_phases=memory_phases,
        cuda_memory_scope="PyTorch CUDA allocator; includes resident weights/state and cached reservations; "
                          "excludes CUDA context, non-PyTorch allocations and other processes; "
                          "no empty_cache between phases",
    )
    result.update(model_load_seconds=load_seconds, warmup_seconds=warmup_seconds,
                  warmup_steps_per_shape=0 if args.cold else 2, cold=args.cold, execution=args.execution, batch_sizes=groups, process_id=os.getpid(),
                  python=platform.python_version(), platform=platform.platform(),
                  torch_cpu_threads=torch.get_num_threads(), runtime=predictor.runtime,
                  checkpoint_sha256=predictor.checkpoint_sha256,
                  cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    write_json(args.output / f"{args.stage}.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/raw/enwik9")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best-3M.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Positive uint32; choose a batch that fits memory; remaining blocks limit the actual batch")
    parser.add_argument("--sample-bytes", type=int, default=1_000_000,
                        help="Bytes from the start of enwik9 (default 1000000; 10 MB = 10000000)")
    parser.add_argument("--source-offset", type=int, default=0,
                        help="Zero-based byte offset in the source file")
    parser.add_argument("--direct-source", action="store_true",
                        help="Encode the source directly; requires offset 0 and sample size equal to file size")
    parser.add_argument("--format", choices=("v3", "v4", "v5", "v6"), default="v5",
                        help="v6 uses batched MTP-4 sequence continuation; v3-v5 remain available")
    parser.add_argument("--execution", choices=("auto", "graph", "buffered", "sequence"), default="graph",
                        help="v5: graph/buffered; v6: auto/graph/sequence")
    parser.add_argument("--cold", action="store_true", help="Include preparation in codec time")
    parser.add_argument("--phase-timing", action="store_true",
                        help="Collect instrumented MTP-4 end-to-end phase timings")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an incomplete output directory after validating completed stages")
    parser.add_argument("--stage", choices=("encode", "decode"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    from mamba3_byte_compress import codec as c

    c.validate_batch_size(args.batch_size)
    if args.format == "v6" and args.execution == "buffered":
        parser.error("--format v6 supports auto, graph or sequence execution.")
    if args.format != "v6" and args.execution in ("auto", "sequence"):
        parser.error("--execution auto/sequence is only valid with --format v6.")
    if args.sample_bytes <= 0 or args.source_offset < 0:
        parser.error("--sample-bytes must be positive and --source-offset nonnegative.")
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    if args.stage:
        stage(args)
        return
    if args.resume:
        if not args.output.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {args.output}")
        completed = args.output / "results.json"
        if (completed.is_file() and (args.output / "results.csv").is_file()
                and (args.output / "verification.json").is_file()):
            result = json.loads(completed.read_text())
            if not result.get("verification", {}).get("sha256_matches"):
                raise ValueError("Existing results.json does not contain a successful verification.")
            raw, packed = sample_paths(args)
            decoded_path = args.output / "decoded.bin"
            if not (raw.is_file() and packed.is_file() and decoded_path.is_file()):
                raise FileNotFoundError("Completed run is missing an input, compressed, or decoded file.")
            verification = result["verification"]
            if c.sha256_file(raw) != verification["original_sha256"]:
                raise ValueError("Completed run input checksum mismatch.")
            if c.sha256_file(packed) != verification["compressed_sha256"]:
                raise ValueError("Completed run compressed checksum mismatch.")
            if c.sha256_file(decoded_path) != verification["decoded_sha256"]:
                raise ValueError("Completed run decoded checksum mismatch.")
            c.verify_files(raw, decoded_path)
            print(json.dumps(result["verification"], indent=2), flush=True)
            return
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    raw, packed = sample_paths(args)
    if args.direct_source:
        if args.source_offset != 0 or args.source.stat().st_size != args.sample_bytes:
            raise ValueError("--direct-source requires offset 0 and sample size equal to source size.")
    else:
        if args.resume and raw.is_file():
            if raw.stat().st_size != args.sample_bytes:
                raise ValueError("Existing sample size does not match the resume configuration.")
        else:
            extract_slice(args.source, raw, args.source_offset, args.sample_bytes)
    process_seconds = {}
    for operation in ("encode", "decode"):
        artifact = packed if operation == "encode" else args.output / "decoded.bin"
        if args.resume:
            try:
                existing = completed_stage(args.output, operation, artifact, args.sample_bytes)
            except FileExistsError:
                moved = archive_incomplete_stage(args.output, operation, artifact)
                print(f"{operation}: archived interrupted outputs: {moved}", flush=True)
                existing = None
        else:
            existing = None
        if existing is not None:
            process_seconds[operation] = existing.get("process_seconds")
            print(f"{operation} already complete", flush=True)
            continue
        command = [sys.executable, str(Path(__file__).resolve()), "--stage", operation,
                   "--output", str(args.output), "--checkpoint", str(args.checkpoint.resolve()),
                   "--device", args.device, "--batch-size", str(args.batch_size),
                   "--sample-bytes", str(args.sample_bytes),
                   "--source-offset", str(args.source_offset), "--format", args.format,
                   "--execution", args.execution, "--source", str(args.source)]
        if args.direct_source:
            command.append("--direct-source")
        if args.cold:
            command.append("--cold")
        if args.phase_timing:
            command.append("--phase-timing")
        environment = os.environ.copy()
        cache_root = environment.get("TRITON_CACHE_DIR")
        if args.cold and cache_root:
            environment["TRITON_CACHE_DIR"] = str(Path(cache_root) / operation)
        process_start = time.perf_counter()
        if args.resume:
            command.append("--resume")
        with next_log_path(args.output, operation).open("x") as log:
            subprocess.run(
                command, cwd=ROOT, env=environment,
                stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        process_seconds[operation] = time.perf_counter() - process_start
        print(f"{operation} complete", flush=True)
    encoded = json.loads((args.output / "encode.json").read_text())
    decoded = json.loads((args.output / "decode.json").read_text())
    encoded["process_seconds"] = process_seconds["encode"]
    decoded["process_seconds"] = process_seconds["decode"]
    verification = dict(files_identical=c.verify_files(raw, args.output / "decoded.bin"),
                        original_sha256=c.sha256_file(raw),
                        decoded_sha256=c.sha256_file(args.output / "decoded.bin"),
                        compressed_sha256=c.sha256_file(packed),
                        independent_processes=encoded["process_id"] != decoded["process_id"],
                        source=str(args.source.resolve()), source_offset=args.source_offset,
                        sample_bytes=args.sample_bytes, file_format=args.format)
    verification["sha256_matches"] = (verification["original_sha256"] == verification["decoded_sha256"]
                                      == encoded["original_sha256"] == decoded["decoded_sha256"])
    if not verification["sha256_matches"]:
        raise ValueError("Independent SHA256 verification failed.")
    sources = ("codec.py", "mamba3_byte_compress/codec.py", "mamba3_byte_compress/codec_batched.py",
               "mamba3_byte_compress/codec_batched_v4.py",
               "mamba3_byte_compress/codec_step_adapter.py", "mamba3_byte_compress/codec_graph.py",
               "mamba3_byte_compress/codec_mtp4.py",
               "mamba3_byte_compress/codec_sequence_adapter.py",
               "mamba3_byte_compress/model.py", "mamba3_byte_compress/trainer.py",
               "mamba3_byte_compress/metrics.py", "mamba3_byte_compress/config.py",
               "scripts/benchmark_codec_batched.py",
               "scripts/benchmark_mtp4_inference.py",
               "scripts/profile_mtp4_short_sequence.py")
    verification["source_sha256"] = {name: c.sha256_file(ROOT / name) for name in sources}
    if not (args.output / "verification.json").exists():
        write_json(args.output / "verification.json", verification)
    if not (args.output / "results.json").exists():
        write_json(args.output / "results.json", dict(encode=encoded, decode=decoded, verification=verification))
    fields = ["operation", "num_bytes", "batch_size", "compressed_bytes", "range_payload_bytes",
              "theoretical_nll_bpb", "actual_bpb", "payload_bpb", "seconds", "kb_per_second",
              "model_load_seconds", "warmup_seconds", "codec_peak_allocated_bytes",
              "codec_peak_reserved_bytes", "run_peak_allocated_bytes", "run_peak_reserved_bytes"]
    csv_path = args.output / "results.csv"
    if csv_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {csv_path}")
    with csv_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for operation, result in (("encode", encoded), ("decode", decoded)):
            row = {field: result.get(field, "") for field in fields}
            row.update(operation=operation, seconds=result[f"{operation}_seconds"],
                       kb_per_second=result[f"{operation}_kb_per_second"])
            writer.writerow(row)
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Profile one steady B=1024 CUDA Graph segment of the byte codec."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import statistics

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def event_ms(action, count):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(count):
        action()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("autoregressive", "mtp4"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.batch_size != 1024:
        parser.error("This comparison uses exactly 1024 blocks.")
    args.output.mkdir(parents=True, exist_ok=False)

    from mamba3_byte_compress import codec as c

    torch.set_num_threads(1)
    required = args.batch_size * c.BLOCK_SIZE
    with args.input.open("rb") as source:
        data = source.read(required)
    if len(data) != required:
        raise ValueError("Input lacks one full 1024-block batch.")
    values = np.frombuffer(data, dtype=np.uint8).reshape(args.batch_size, c.BLOCK_SIZE)
    inference = (
        "batched-step-v5"
        if args.system == "autoregressive"
        else "mtp4-batched-sequence"
    )
    predictor = c.load_predictor(
        args.checkpoint, inference=inference,
        batch_size=args.batch_size, execution="graph"
        if args.system == "mtp4" else None,
    )
    if args.system == "autoregressive":
        tokens = [values[:, position] for position in range(129, 257)]
        warm = lambda: [
            predictor.step(values[:, position])
            for position in range(0, 129)
        ]
        public_step = predictor.step
        graph = lambda: predictor._graph.replay()
        device_tokens = lambda: predictor._token
        host_tokens = lambda: predictor._host_token
        device_probabilities = lambda: predictor._probabilities
    else:
        tokens = [values[:, position:position + 4] for position in range(129, 257, 4)]
        def warm():
            predictor.start(values[:, 0])
            for context in range(0, 128, 4):
                predictor.advance(values[:, context + 1:context + 5])
        public_step = predictor.advance
        graph = lambda: predictor._advance_graph.replay()
        device_tokens = lambda: predictor._tokens
        host_tokens = lambda: predictor._host_tokens
        device_probabilities = lambda: predictor._advance_probabilities

    def reset_and_warm():
        predictor.reset(args.batch_size)
        warm()
        predictor.synchronize()

    with c._coding_mode(predictor):
        reset_and_warm()
        start = time.perf_counter()
        for token in tokens:
            public_step(token)
        predictor.synchronize()
        public_wall_seconds = time.perf_counter() - start

        event_samples = {"h2d": [], "graph": [], "d2h": []}
        blocking_d2h_wall = []
        for _ in range(3):
            reset_and_warm()
            host = host_tokens()
            device = device_tokens()
            for token in tokens:
                host.numpy()[:] = token
            event_samples["h2d"].append(
                event_ms(lambda: device.copy_(host, non_blocking=True), len(tokens))
            )
            event_samples["graph"].append(event_ms(graph, len(tokens)))
            gpu_output = device_probabilities()
            pinned_output = predictor._host_probabilities
            event_samples["d2h"].append(
                event_ms(
                    lambda: pinned_output.copy_(gpu_output, non_blocking=True),
                    len(tokens),
                )
            )
            predictor.synchronize()
            started = time.perf_counter()
            for _ in tokens:
                pinned_output.copy_(gpu_output, non_blocking=False)
            blocking_d2h_wall.append(time.perf_counter() - started)

        reset_and_warm()
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(activities=activities) as profile:
            for token in tokens:
                with torch.profiler.record_function("codec_predictor_step"):
                    public_step(token)
            predictor.synchronize()

    trace_json = args.output / "trace.json"
    profile.export_chrome_trace(str(trace_json))
    with trace_json.open("rb") as source, gzip.open(args.output / "trace.json.gz", "wb") as target:
        shutil.copyfileobj(source, target)
    trace_json.unlink()
    top_cuda = [
        {
            "name": event.key,
            "device_time_us": float(event.device_time_total),
            "count": int(event.count),
        }
        for event in sorted(
            profile.key_averages(),
            key=lambda event: event.device_time_total,
            reverse=True,
        )
        if event.device_time_total > 0
    ][:25]
    result = {
        "system": args.system,
        "batch_size": args.batch_size,
        "byte_positions": 128,
        "model_calls": len(tokens),
        "source": str(args.input.resolve()),
        "source_first_batch_sha256": hashlib.sha256(data).hexdigest(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "runtime": predictor.runtime,
        "public_predictor_wall_seconds": public_wall_seconds,
        "isolated_cuda_event_ms_for_128_byte_positions": {
            key: {"median": statistics.median(samples), "samples": samples}
            for key, samples in event_samples.items()
        },
        "isolated_blocking_d2h_wall_seconds": {
            "median": statistics.median(blocking_d2h_wall),
            "samples": blocking_d2h_wall,
        },
        "top_cuda_events": top_cuda,
        "note": (
            "Isolated CUDA events and the public-call wall clock have different "
            "measurement scopes and must not be summed. The trace covers one "
            "steady 128-byte span without Range Coding."
        ),
    }
    with (args.output / "profile.json").open("x") as target:
        json.dump(result, target, indent=2, ensure_ascii=False)
        target.write("\n")
    print(json.dumps({
        "system": args.system,
        "public_predictor_wall_seconds": public_wall_seconds,
        "event_medians_ms": {
            key: value["median"]
            for key, value in result["isolated_cuda_event_ms_for_128_byte_positions"].items()
        },
        "top_cuda": top_cuda[:5],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

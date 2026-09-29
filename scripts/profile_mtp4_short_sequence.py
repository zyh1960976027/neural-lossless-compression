"""Compare the installed length-4 SISO path with four captured recurrent steps."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def event_milliseconds(replay, repeats):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "experiments/train/20260918_175712_mamba3_siso_3m_mtp4/checkpoints/best.pt",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")

    from mamba3_byte_compress import codec as c
    from mamba3_byte_compress.codec_graph import BufferedMambaStep
    from mamba3_byte_compress.metrics import autocast_context

    predictor = c.load_predictor(
        args.checkpoint, inference="mtp4-batched-sequence",
        batch_size=args.batch_size, execution="graph",
    )
    zeros = np.zeros((args.batch_size, 4), dtype=np.uint8)
    predictor.reset(args.batch_size)
    predictor.start(np.zeros(args.batch_size, dtype=np.uint8))
    predictor.advance(zeros)
    sequence_ms = event_milliseconds(
        predictor._advance_graph.replay, args.repeats
    )

    model = predictor.model
    states = [
        block.mixer.allocate_inference_cache(
            args.batch_size, c.BLOCK_SIZE, dtype=torch.bfloat16
        )
        for block in model.blocks
    ]
    steps = [BufferedMambaStep(block.mixer) for block in model.blocks]
    tokens = torch.zeros(
        (args.batch_size, 4), dtype=torch.long, device=predictor.device
    )

    def compute_four_steps():
        with autocast_context(predictor.device):
            for position in range(4):
                x = model.embedding(tokens[:, position])
                for block, layer_states, step in zip(model.blocks, states, steps):
                    mixed = step(block.norm1(x), *layer_states)
                    x = x + mixed
                    x = x + block.mlp(block.norm2(x))
            hidden = model.final_norm(x)
            logits = torch.nn.functional.linear(hidden, predictor._head_weight)
            return logits.view(args.batch_size, 4, 256).float().softmax(dim=-1)

    with torch.inference_mode():
        for _ in range(3):
            compute_four_steps()
        predictor.synchronize()
        for layer_states in states:
            for state in layer_states:
                state.zero_()
        four_step_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(four_step_graph):
            four_step_probabilities = compute_four_steps()
        del four_step_probabilities
        predictor.synchronize()
        four_step_ms = event_milliseconds(four_step_graph.replay, args.repeats)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        for _ in range(8):
            predictor._advance_graph.replay()
        predictor.synchronize()
    top_cuda = [
        {
            "name": event.key,
            "device_time_us": event.device_time_total,
            "count": event.count,
        }
        for event in sorted(
            profile.key_averages(), key=lambda item: item.device_time_total,
            reverse=True,
        )
        if event.device_time_total > 0
    ][:15]
    result = {
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "sequence4_graph_ms": sequence_ms,
        "four_step_graph_ms": four_step_ms,
        "four_step_over_sequence_ratio": four_step_ms / sequence_ms,
        "candidate_retained": False,
        "reason": "Four captured recurrent steps are slower than the installed length-4 SISO path.",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "runtime": predictor.runtime,
        "top_cuda_events_for_sequence_graph": top_cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

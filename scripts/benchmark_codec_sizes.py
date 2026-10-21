"""Measure container, probability quantization and range coding costs unchanged."""

import argparse
import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time
import zlib

import constriction

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neural_compression import codec

PRECISION = 24
TOTAL = 1 << PRECISION
INITIAL_RANGE = (1 << 64) - 1


def quantized_count(symbol, entropy_model):
    """Read the exact frequency via an isolated one-symbol range update.

    Constriction 0.4.2: new_range = (old_range >> 24) * frequency.
    Initially even frequency=1 leaves range > 2**32, so no renormalization
    occurs. This avoids reimplementing its floating-point CDF conversion.
    """
    probe = constriction.stream.queue.RangeEncoder()
    assert probe.pos() == (0, (0, INITIAL_RANGE))
    probe.encode(int(symbol), entropy_model)
    position, (_, width) = probe.pos()
    count, remainder = divmod(width, INITIAL_RANGE >> PRECISION)
    assert position == 0 and remainder == 0 and 1 <= count < TOTAL
    return count


class Observer:
    """Forward original predictions unchanged; duplicate only entropy coding.

    The duplicate coder lets us check its payload against the actual file,
    and measure integer range-rounding separately from CDF quantization.
    No predictions or histories are shared from encoding into decoding.
    """

    def __init__(self, predictor, data, label):
        self.predictor = predictor
        self.data = data
        self.label = label
        self.stage = "encode"
        self.index = 1
        self.last_log = time.monotonic()
        self.frames = []
        self.shadow = None
        self.start_frame("uniform_first_byte")
        self.record(data[0], codec._uniform(), 8.0)
        self.end_frame()

    def __getattr__(self, name):
        return getattr(self.predictor, name)

    def start_frame(self, kind):
        self.shadow = constriction.stream.queue.RangeEncoder()
        self.frame = dict(kind=kind, symbols=0, model_bits=0.0,
                          quantized_bits=0.0, range_rounding_bits=0.0)

    def record(self, symbol, entropy_model, model_bits):
        count = quantized_count(symbol, entropy_model)
        width = self.shadow.pos()[1][1]
        kept = (width >> PRECISION) << PRECISION
        # Stable evaluation of log2(width / kept), a very small positive cost.
        rounding = math.log1p((width - kept) / kept) / math.log(2)
        self.shadow.encode(int(symbol), entropy_model)
        self.frame["symbols"] += 1
        self.frame["model_bits"] += model_bits
        self.frame["quantized_bits"] += PRECISION - math.log2(count)
        self.frame["range_rounding_bits"] += rounding

    def end_frame(self):
        payload = self.shadow.get_compressed().astype("<u4", copy=False).tobytes()
        self.frame.update(payload_bytes=len(payload), payload_sha256=hashlib.sha256(payload).hexdigest())
        self.frame["termination_alignment_bits"] = (
            8 * len(payload) - self.frame["quantized_bits"] - self.frame["range_rounding_bits"]
        )
        self.frames.append(self.frame)
        self.shadow = None

    def __call__(self, history):
        probabilities, log_probs = self.predictor(history)
        if self.stage == "encode":
            if (self.index - 1) % codec.BLOCK_SIZE == 0:
                self.start_frame("prediction_block")
            symbol = self.data[self.index]
            self.record(symbol, codec._categorical(probabilities), -float(log_probs[symbol]) / math.log(2))
            if self.index % codec.BLOCK_SIZE == 0 or self.index == len(self.data) - 1:
                self.end_frame()
        self.index += 1
        if time.monotonic() - self.last_log >= 20:
            print(f"{self.label}: {self.stage} {self.index}/{len(self.data)} bytes", flush=True)
            self.last_log = time.monotonic()
        return probabilities, log_probs


def parse_container(path):
    with path.open("rb") as f:
        magic, json_size = codec.HEADER.unpack(f.read(codec.HEADER.size))
        assert magic == codec.MAGIC
        metadata = json.loads(f.read(json_size))
        n = metadata["num_bytes"]
        frames = []
        for _ in range(1 + (max(0, n - 1) + codec.BLOCK_SIZE - 1) // codec.BLOCK_SIZE):
            length, checksum = codec.FRAME.unpack(f.read(codec.FRAME.size))
            payload = f.read(length)
            assert len(payload) == length and zlib.crc32(payload) == checksum
            frames.append(dict(payload_bytes=length, payload_sha256=hashlib.sha256(payload).hexdigest()))
        digest = f.read(32)
        assert len(digest) == 32 and not f.read(1)
    fixed = codec.HEADER.size + json_size + 32
    frame_headers = codec.FRAME.size * len(frames)
    payload_bytes = sum(frame["payload_bytes"] for frame in frames)
    assert path.stat().st_size == fixed + frame_headers + payload_bytes
    return dict(fixed_prefix_bytes=codec.HEADER.size, json_metadata_bytes=json_size,
                file_sha256_bytes=32, fixed_file_overhead_bytes=fixed,
                frame_header_bytes=frame_headers, frames=frames)


def run_sample(raw_path, directory, predictor, label, reference=None):
    data = raw_path.read_bytes()
    observed = Observer(predictor, data, label)
    packed, decoded = directory / "compressed.m3rc", directory / "decoded.bin"
    print(f"{label}: encode start ({len(data)} bytes)", flush=True)
    result = codec.encode_file(raw_path, packed, observed, file_format="v1")
    assert observed.index == len(data) and observed.shadow is None
    layout = parse_container(packed)
    for actual, measured in zip(layout.pop("frames"), observed.frames, strict=True):
        assert all(actual[key] == measured[key] for key in actual)
    observed.stage, observed.index = "decode", 1
    print(f"{label}: decode start", flush=True)
    result.update(codec.decode_file(packed, decoded, observed))
    result["files_identical"] = codec.verify_files(raw_path, decoded)
    result.update(layout)
    result.update(label=label, input=str(raw_path.resolve()), frames=observed.frames)
    for key in ("model_bits", "quantized_bits", "range_rounding_bits", "termination_alignment_bits"):
        result[key] = math.fsum(frame[key] for frame in observed.frames)
    assert math.isclose(result["model_bits"] / len(data), result["theoretical_nll_bpb"], abs_tol=1e-10)
    result["quantization_delta_bits"] = result["quantized_bits"] - result["model_bits"]
    result["actual_minus_theoretical_bpb"] = result["actual_bpb"] - result["theoretical_nll_bpb"]
    for key in ("quantization_delta_bits", "range_rounding_bits", "termination_alignment_bits"):
        result[key.removesuffix("_bits") + "_bpb"] = result[key] / len(data)
    result["fixed_file_overhead_bpb"] = result["fixed_file_overhead_bytes"] * 8 / len(data)
    result["frame_header_bpb"] = result["frame_header_bytes"] * 8 / len(data)
    components = [result[key] for key in (
        "fixed_file_overhead_bpb", "frame_header_bpb", "quantization_delta_bpb",
        "range_rounding_bpb", "termination_alignment_bpb")]
    assert math.isclose(sum(components), result["actual_minus_theoretical_bpb"], abs_tol=1e-10)
    if reference:
        result["matches_previous_compressed_file"] = codec.verify_files(reference, packed)
    (directory / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in (
        "label", "num_bytes", "theoretical_nll_bpb", "actual_bpb", "compressed_bytes",
        "range_payload_bytes", "container_overhead_bytes", "actual_minus_theoretical_bpb", "files_identical")}), flush=True)
    return result


def write_summary(root, rows, metadata):
    (root / "summary.json").write_text(json.dumps(dict(metadata=metadata, results=rows), indent=2, allow_nan=False) + "\n")
    columns = ["label", "num_bytes", "theoretical_nll_bpb", "compressed_bytes", "actual_bpb",
               "range_payload_bytes", "container_overhead_bytes", "actual_minus_theoretical_bpb",
               "files_identical", "fixed_file_overhead_bytes", "frame_header_bytes", "payload_bpb",
               "quantization_delta_bpb", "range_rounding_bpb", "termination_alignment_bpb"]
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/raw/enwik9")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best-3M.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if version("constriction") != "0.4.2":
        raise RuntimeError("This diagnostic requires Constriction 0.4.2.")
    with args.source.open("rb") as source:
        data = source.read(100000)
    if len(data) != 100000:
        raise ValueError("Source must contain at least 100000 bytes.")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(source=str(args.source.resolve()), source_offset=0, source_size=args.source.stat().st_size,
                    source_prefix_100k_sha256=hashlib.sha256(data).hexdigest(), unit="1 KB = 1000 bytes",
                    checkpoint=str(args.checkpoint.resolve()), started=datetime.now().isoformat(),
                    coding_unchanged=True, encode_timing_includes_diagnostics=True,
                    diagnostic_source="https://docs.rs/constriction/latest/src/constriction/stream/queue.rs.html")
    predictor = codec.load_predictor(args.checkpoint, args.device)
    metadata["checkpoint_sha256"] = predictor.checkpoint_sha256
    rows = []
    for size in (2000, 10000, 50000, 100000):
        label = f"{size // 1000}KB"
        directory = args.output / label
        directory.mkdir()
        raw = directory / "original.bin"
        raw.write_bytes(data[:size])
        rows.append(run_sample(raw, directory, predictor, label))
        write_summary(args.output, rows, metadata)
    previous = ROOT / "experiments/test/codec_smoke_20260914"
    directory = args.output / "previous_2066B"
    directory.mkdir()
    run_sample(previous / "original.bin", directory, predictor, "previous_2066B",
               reference=previous / "compressed.m3rc")
    metadata["completed"] = datetime.now().isoformat()
    write_summary(args.output, rows, metadata)


if __name__ == "__main__":
    main()

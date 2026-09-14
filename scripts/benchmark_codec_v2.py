"""Compare the original 100 KB experiment with compact/shared-stream coding."""

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mamba3_byte_compress import codec
from benchmark_codec_sizes import Observer, parse_container


class SharedObserver(Observer):
    def __init__(self, predictor, data):
        self.predictor, self.data = predictor, data
        self.label, self.stage, self.index = "v2", "encode", 1
        self.last_log = time.monotonic()
        self.frames = []
        self.start_frame("shared_stream")
        self.record(data[0], codec._uniform(), 8.0)

    def __call__(self, history):
        probabilities, log_probs = self.predictor(history)
        if self.stage == "encode":
            symbol = self.data[self.index]
            self.record(symbol, codec._categorical(probabilities), -float(log_probs[symbol]) / math.log(2))
            if self.index == len(self.data) - 1:
                self.end_frame()
        self.index += 1
        if time.monotonic() - self.last_log >= 20:
            print(f"v2: {self.stage} {self.index}/{len(self.data)} bytes", flush=True)
            self.last_log = time.monotonic()
        return probabilities, log_probs


def compact_only(baseline_path, output_path, predictor, baseline):
    """Repack existing frames without predicting or entropy-encoding again."""
    layout = parse_container(baseline_path)  # Validates every original frame.
    original = baseline_path.read_bytes()
    offset = codec.HEADER.size + layout["json_metadata_bytes"]
    frame_body, digest = original[offset:-32], original[-32:]
    assert digest.hex() == baseline["original_sha256"]
    with codec._new_output(output_path) as handle:
        codec._write_compact_header(handle, baseline["num_bytes"], predictor, "v2-framed")
        writer = codec._BodyWriter(handle)
        writer.write(frame_body)
        handle.write(codec.COMPACT_TRAILER.pack(writer.crc, digest))
    with output_path.open("rb") as handle:
        metadata = codec._read_header(handle, predictor)
        body, checked_digest = codec._read_compact_body(handle)
        assert body.read() == frame_body and checked_digest == digest
        assert metadata["num_bytes"] == baseline["num_bytes"]
    return {
        **{key: baseline[key] for key in (
            "num_bytes", "theoretical_nll_bpb", "nll_nats_total", "range_payload_bytes",
            "original_sha256", "checkpoint_sha256", "runtime", "input",
        )},
        "label": "compact_header_only", "file_format": "v2-framed",
        "compressed_bytes": output_path.stat().st_size,
        "container_overhead_bytes": output_path.stat().st_size - baseline["range_payload_bytes"],
        "actual_bpb": output_path.stat().st_size * 8 / baseline["num_bytes"],
        "fixed_file_overhead_bytes": codec.COMPACT_HEADER.size + codec.CRC32.size + codec.COMPACT_TRAILER.size,
        "frame_header_bytes": baseline["frame_header_bytes"],
        "frame_bytes_identical_to_baseline": True,
        "full_roundtrip_rerun": False,
        "validation": "Byte-identical original frames, original length and SHA256; compact header/body CRC validated.",
    }


def verify_prediction_code(root):
    before = ast.parse((root / "codec_before.py").read_text())
    after = ast.parse((ROOT / "mamba3_byte_compress/codec.py").read_text())
    names = ("PrefixPredictor", "load_predictor", "_coding_mode", "_categorical", "_uniform")
    for name in names:
        left = next(node for node in before.body if getattr(node, "name", None) == name)
        right = next(node for node in after.body if getattr(node, "name", None) == name)
        assert ast.dump(left) == ast.dump(right), name
    for name, digest in json.loads((root / "unchanged_code_hashes.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best-3M.pt")
    args = parser.parse_args()
    root = args.output
    verify_prediction_code(root)
    baseline_root = ROOT / "experiments/test/codec_sizes_20260914/100KB"
    baseline = json.loads((baseline_root / "results.json").read_text())
    raw = baseline_root / "original.bin"
    data = raw.read_bytes()
    assert len(data) == 100000 and hashlib.sha256(data).hexdigest() == baseline["original_sha256"]
    predictor = codec.load_predictor(args.checkpoint)
    assert predictor.checkpoint_sha256 == baseline["checkpoint_sha256"]
    assert predictor.runtime == baseline["runtime"]
    compact = compact_only(baseline_root / "compressed.m3rc", root / "compact_only.m3rc", predictor, baseline)
    print("Compact-header-only file:", compact["compressed_bytes"], "bytes", flush=True)
    observed = SharedObserver(predictor, data)
    result = codec.encode_file(raw, root / "optimized.m3rc", observed)
    assert observed.index == len(data) and len(observed.frames) == 1
    payload = (root / "optimized.m3rc").read_bytes()[codec.COMPACT_HEADER.size + codec.CRC32.size:-codec.COMPACT_TRAILER.size]
    measured = observed.frames[0]
    assert hashlib.sha256(payload).hexdigest() == measured["payload_sha256"]
    assert result["theoretical_nll_bpb"] == baseline["theoretical_nll_bpb"]
    assert math.isclose(measured["quantized_bits"], baseline["quantized_bits"], abs_tol=1e-5)
    # Persist encoder metrics before the relatively long decoding pass.
    (root / "encode_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Optimized encoding complete:", result["compressed_bytes"], "bytes; NLL exactly unchanged", flush=True)
    observed.stage, observed.index = "decode", 1
    result.update(codec.decode_file(root / "optimized.m3rc", root / "decoded.bin", observed))
    result["files_identical"] = codec.verify_files(raw, root / "decoded.bin")
    assert result["original_sha256"] == result["decoded_sha256"] == codec.sha256_file(root / "decoded.bin")
    result.update(label="compact_shared_stream", input=str(raw), checkpoint=str(args.checkpoint),
                  exact_nll_match=True, probability_code_unchanged=True, diagnostics=measured)
    n = len(data)
    result["actual_minus_theoretical_bpb"] = result["actual_bpb"] - result["theoretical_nll_bpb"]
    result["quantization_delta_bpb"] = (measured["quantized_bits"] - measured["model_bits"]) / n
    result["range_rounding_bpb"] = measured["range_rounding_bits"] / n
    result["termination_alignment_bpb"] = measured["termination_alignment_bits"] / n
    result["container_bpb"] = result["container_overhead_bytes"] * 8 / n
    assert math.isclose(sum(result[k] for k in ("quantization_delta_bpb", "range_rounding_bpb",
                                               "termination_alignment_bpb", "container_bpb")),
                        result["actual_minus_theoretical_bpb"], abs_tol=1e-10)
    comparison = [dict(baseline, label="baseline_v1"), compact, result]
    savings = dict(
        header_bytes=baseline["compressed_bytes"] - compact["compressed_bytes"],
        shared_stream_bytes=compact["compressed_bytes"] - result["compressed_bytes"],
        frame_header_bytes=baseline["frame_header_bytes"] - result["frame_header_bytes"],
        range_payload_bytes=baseline["range_payload_bytes"] - result["range_payload_bytes"],
        total_bytes=baseline["compressed_bytes"] - result["compressed_bytes"],
    )
    savings.update({key.replace("_bytes", "_bpb"): value * 8 / n for key, value in list(savings.items())})
    (root / "results.json").write_text(json.dumps(dict(comparison=comparison, savings=savings), indent=2) + "\n")
    columns = ("label", "theoretical_nll_bpb", "range_payload_bytes", "container_overhead_bytes", "compressed_bytes", "actual_bpb")
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparison)
    verify_prediction_code(root)
    print(json.dumps(dict(optimized=result, savings=savings), indent=2), flush=True)


if __name__ == "__main__":
    main()

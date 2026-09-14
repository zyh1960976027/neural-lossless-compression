"""A correctness-first, autoregressive Constriction range codec.

Blocks follow ByteEvaluationDataset: one uniform file byte, then groups of
2048 targets, each seeded by the preceding byte with no other carried context.
Both ends recompute the same zero-padded prefix through the existing model.
"""

from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import struct
import time
import zlib

import constriction
import numpy as np
import torch

from .metrics import autocast_context


BLOCK_SIZE = 2048
MAGIC = b"M3RC\r\n\x1a\n"
HEADER = struct.Struct("<8sI")
FRAME = struct.Struct("<II")  # payload byte length, CRC32
MAX_HEADER_BYTES = 65536
MAX_FRAME_BYTES = 65536
PROFILE = "padded-prefix-2048-bf16-fp32-master-cpu-f64-softmax-v1"
MAGIC_V2 = b"M3RC\x02\r\n\x1a"
COMPACT_HEADER = struct.Struct("<8sBBQ32s16s")
CRC32 = struct.Struct("<I")
COMPACT_TRAILER = struct.Struct("<I32s")
PROFILE_ID = 1  # PROFILE above, including 2048 targets and the existing CDF.
FILE_FORMATS = ("v1", "v2-framed", "v2")


def _runtime_fingerprint(predictor):
    serialized = json.dumps(predictor.runtime, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).digest()[:16]


def _write_compact_header(handle, num_bytes, predictor, file_format):
    header = COMPACT_HEADER.pack(
        MAGIC_V2, PROFILE_ID, int(file_format == "v2"), num_bytes,
        bytes.fromhex(predictor.checkpoint_sha256), _runtime_fingerprint(predictor),
    )
    handle.write(header)
    handle.write(CRC32.pack(zlib.crc32(header)))


class _BodyWriter:
    """Checksum only the body, excluding the compact header and trailer."""

    def __init__(self, handle):
        self.handle = handle
        self.crc = 0

    def write(self, data):
        self.handle.write(data)
        self.crc = zlib.crc32(data, self.crc)


def _read_compact_header(handle, predictor):
    header = MAGIC_V2 + _read_exact(handle, COMPACT_HEADER.size - len(MAGIC_V2))
    if CRC32.unpack(_read_exact(handle, CRC32.size))[0] != zlib.crc32(header):
        raise ValueError("Compact header CRC32 mismatch.")
    _, profile, mode, num_bytes, checkpoint, runtime = COMPACT_HEADER.unpack(header)
    if profile != PROFILE_ID or mode not in (0, 1):
        raise ValueError("Unsupported compact profile or stream mode.")
    if checkpoint.hex() != predictor.checkpoint_sha256:
        raise ValueError("Incompatible compressed file: checkpoint_sha256 mismatch.")
    if runtime != _runtime_fingerprint(predictor):
        raise ValueError("Incompatible compressed file: runtime mismatch.")
    return {**_metadata(num_bytes, predictor), "version": 2,
            "file_format": "v2" if mode else "v2-framed"}


def _read_compact_body(handle):
    # Constriction's Python decoder consumes an in-memory uint32 array. The
    # shared stream is buffered once; model history remains bounded at 2048.
    remaining = handle.read()
    if len(remaining) < COMPACT_TRAILER.size:
        raise ValueError("Truncated compact file trailer.")
    body = remaining[:-COMPACT_TRAILER.size]
    checksum, digest = COMPACT_TRAILER.unpack(remaining[-COMPACT_TRAILER.size:])
    if zlib.crc32(body) != checksum:
        raise ValueError("Compressed body CRC32 mismatch.")
    return io.BytesIO(body), digest


def _format_metrics(file_format, num_bytes):
    streams = (1 + (max(0, num_bytes - 1) + BLOCK_SIZE - 1) // BLOCK_SIZE) if num_bytes else 0
    return {
        "file_format": file_format,
        "format_version": 1 if file_format == "v1" else 2,
        "num_range_streams": int(bool(num_bytes)) if file_format == "v2" else streams,
        "frame_header_bytes": 0 if file_format == "v2" else streams * FRAME.size,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime(device):
    return {
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "constriction": version("constriction"),
        "device_type": device.type,
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "mamba_ssm": version("mamba_ssm") if device.type == "cuda" else None,
        "triton": version("triton") if device.type == "cuda" else None,
    }


class PrefixPredictor:
    """Use only known bytes; identical shapes and operations on both ends.

    CPU models are supported for tests. Production Mamba-3 uses CUDA/BF16.
    The fingerprint identifies the external checkpoint, which is not embedded.
    """

    def __init__(self, model, device, checkpoint_sha256):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.checkpoint_sha256 = checkpoint_sha256
        self.runtime = _runtime(self.device)

    @torch.inference_mode()
    def __call__(self, history):
        if not 1 <= len(history) <= BLOCK_SIZE:
            raise ValueError("Expected 1..2048 known bytes in the current block.")
        inputs = torch.zeros((1, BLOCK_SIZE), dtype=torch.long, device=self.device)
        inputs[0, :len(history)] = torch.tensor(list(history), device=self.device)
        with autocast_context(self.device):
            logits = self.model(inputs)
        if logits.shape != (1, BLOCK_SIZE, 256):
            raise ValueError("The model must output [1, 2048, 256] logits.")
        last = logits[0, len(history) - 1].detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(last).all():
            raise FloatingPointError("The model produced non-finite logits.")
        log_probs = last.log_softmax(dim=-1).numpy()
        return np.exp(log_probs), log_probs

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def load_predictor(checkpoint, device="cuda:0"):
    """Load the existing project checkpoint without changing its architecture."""
    checkpoint = Path(checkpoint).resolve(strict=True)
    # Must precede CUDA initialization for deterministic cuBLAS operations.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Mamba-3 coding requires an available CUDA device.")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Mamba-3 coding requires BF16 support.")
    from .trainer import load_mamba3_checkpoint

    model, metadata = load_mamba3_checkpoint(checkpoint)
    del metadata
    return PrefixPredictor(model, device, sha256_file(checkpoint))


@contextmanager
def _coding_mode(predictor):
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    was_training = predictor.model.training
    try:
        predictor.model.eval()
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode(), torch.backends.cudnn.flags(
            benchmark=False, deterministic=True, allow_tf32=False,
        ):
            yield
    finally:
        predictor.model.train(was_training)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@contextmanager
def _new_output(path):
    """Never overwrite a file, and remove this call's incomplete output."""
    path = Path(path)
    handle = path.open("xb")
    try:
        with handle:
            yield handle
    except BaseException:
        path.unlink()
        raise


def _read_exact(handle, count):
    data = handle.read(count)
    if len(data) != count:
        raise ValueError("Truncated input or compressed file.")
    return data


def _categorical(probabilities):
    # Explicit on both ends: defaults changed between Constriction versions.
    # Constriction gives even zero/underflowed probabilities a nonzero mass.
    return constriction.stream.model.Categorical(probabilities, perfect=False)


def _uniform():
    return _categorical(np.full(256, 1.0 / 256, dtype=np.float64))


def _write_frame(handle, encoder):
    payload = encoder.get_compressed().astype("<u4", copy=False).tobytes()
    handle.write(FRAME.pack(len(payload), zlib.crc32(payload)))
    handle.write(payload)
    return len(payload)


def _read_frame(handle):
    size, checksum = FRAME.unpack(_read_exact(handle, FRAME.size))
    if size == 0 or size % 4 or size > MAX_FRAME_BYTES:
        raise ValueError("Invalid range payload length.")
    payload = _read_exact(handle, size)
    if zlib.crc32(payload) != checksum:
        raise ValueError("Range payload CRC32 mismatch.")
    words = np.frombuffer(payload, dtype="<u4").astype(np.uint32, copy=True)
    return constriction.stream.queue.RangeDecoder(words), size


def _metadata(num_bytes, predictor):
    return {
        "version": 1,
        "num_bytes": num_bytes,
        "block_size": BLOCK_SIZE,
        "profile": PROFILE,
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "runtime": predictor.runtime,
    }


def _read_header(handle, predictor):
    magic = _read_exact(handle, len(MAGIC))
    if magic == MAGIC_V2:
        return _read_compact_header(handle, predictor)
    if magic != MAGIC:
        raise ValueError("Not a Mamba-3 range file (bad magic).")
    length = CRC32.unpack(_read_exact(handle, CRC32.size))[0]
    if not 0 < length <= MAX_HEADER_BYTES:
        raise ValueError("Invalid compressed header length.")
    try:
        metadata = json.loads(_read_exact(handle, length))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid compressed header JSON.") from error
    if not isinstance(metadata, dict):
        raise ValueError("Invalid compressed metadata.")
    num_bytes = metadata.get("num_bytes")
    if type(num_bytes) is not int or not 0 <= num_bytes < 2**64:
        raise ValueError("Invalid original byte count.")
    expected = _metadata(num_bytes, predictor)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Incompatible compressed file: {key} mismatch.")
    return {**metadata, "file_format": "v1"}


def _metrics(num_bytes, compressed_bytes, payload_bytes, seconds, operation):
    return {
        "num_bytes": num_bytes,
        "predicted_bytes": max(0, num_bytes - 1),
        "num_prediction_blocks": (max(0, num_bytes - 1) + BLOCK_SIZE - 1) // BLOCK_SIZE,
        "block_size": BLOCK_SIZE,
        "uniform_first_byte_only": True,
        "compressed_bytes": compressed_bytes,
        "range_payload_bytes": payload_bytes,
        "container_overhead_bytes": compressed_bytes - payload_bytes,
        "actual_bpb": 8 * compressed_bytes / num_bytes if num_bytes else None,
        "payload_bpb": 8 * payload_bytes / num_bytes if num_bytes else None,
        f"{operation}_seconds": seconds,
        f"{operation}_kb_per_second": num_bytes / 1000 / seconds if seconds > 0 else 0.0,
    }


def encode_file(input_path, output_path, predictor, *, file_format="v2"):
    """Write a real .m3rc file; return BPB and end-to-end coding throughput."""
    input_path, output_path = Path(input_path), Path(output_path)
    if file_format not in FILE_FORMATS:
        raise ValueError(f"Unknown file format: {file_format}")
    shared = file_format == "v2"
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files.")
    predictor.synchronize()
    started = time.perf_counter()
    digest = hashlib.sha256()
    payload_bytes = 0
    nll = 0.0
    with _coding_mode(predictor), input_path.open("rb") as source, _new_output(output_path) as target:
        num_bytes = os.fstat(source.fileno()).st_size
        if file_format == "v1":
            header = json.dumps(_metadata(num_bytes, predictor), sort_keys=True).encode("utf-8")
            target.write(HEADER.pack(MAGIC, len(header)))
            target.write(header)
            body = target
        else:
            _write_compact_header(target, num_bytes, predictor, file_format)
            body = _BodyWriter(target)
        encoder = constriction.stream.queue.RangeEncoder()
        if num_bytes:
            first = _read_exact(source, 1)
            digest.update(first)
            encoder.encode(first[0], _uniform())
            if not shared:
                payload_bytes += _write_frame(body, encoder)
            previous = first[0]
            nll = math.log(256)
        remaining = max(0, num_bytes - 1)
        while remaining:
            block = _read_exact(source, min(BLOCK_SIZE, remaining))
            digest.update(block)
            history = bytearray([previous])
            if not shared:
                encoder = constriction.stream.queue.RangeEncoder()
            for symbol in block:
                probabilities, log_probs = predictor(history)
                encoder.encode(symbol, _categorical(probabilities))
                nll -= float(log_probs[symbol])
                history.append(symbol)
            if not shared:
                payload_bytes += _write_frame(body, encoder)
            previous = block[-1]
            remaining -= len(block)
        if source.read(1):
            raise ValueError("Input file size changed during encoding.")
        if not math.isfinite(nll):
            raise FloatingPointError("Non-finite theoretical NLL.")
        if shared:
            payload = encoder.get_compressed().astype("<u4", copy=False).tobytes()
            body.write(payload)
            payload_bytes = len(payload)
        if file_format == "v1":
            target.write(digest.digest())
        else:
            target.write(COMPACT_TRAILER.pack(body.crc, digest.digest()))
    predictor.synchronize()
    result = _metrics(num_bytes, output_path.stat().st_size, payload_bytes,
                      time.perf_counter() - started, "encode")
    result.update({
        "theoretical_nll_bpb": nll / (num_bytes * math.log(2)) if num_bytes else 0.0,
        "nll_nats_total": nll,
        "nll_nats_uniform_first_byte": math.log(256) if num_bytes else 0.0,
        "original_sha256": digest.hexdigest(),
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "runtime": predictor.runtime,
        "profile": PROFILE,
    })
    result.update(_format_metrics(file_format, num_bytes))
    return result


def decode_file(input_path, output_path, predictor):
    """Decode using only the compressed file, checkpoint and decoded history."""
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files.")
    predictor.synchronize()
    started = time.perf_counter()
    digest = hashlib.sha256()
    payload_bytes = 0
    with _coding_mode(predictor), input_path.open("rb") as source:
        metadata = _read_header(source, predictor)
        num_bytes = metadata["num_bytes"]
        file_format = metadata["file_format"]
        shared = file_format == "v2"
        if file_format == "v1":
            body, expected_digest = source, None
        else:
            body, expected_digest = _read_compact_body(source)
        if shared:
            payload = body.read()
            if len(payload) % 4 or bool(payload) != bool(num_bytes):
                raise ValueError("Invalid shared Range payload length.")
            payload_bytes = len(payload)
            if num_bytes:
                words = np.frombuffer(payload, dtype="<u4").astype(np.uint32, copy=True)
                decoder = constriction.stream.queue.RangeDecoder(words)
        with _new_output(output_path) as target:
            if num_bytes:
                if not shared:
                    decoder, size = _read_frame(body)
                    payload_bytes += size
                previous = int(decoder.decode(_uniform()))
                first = bytes([previous])
                target.write(first)
                digest.update(first)
            remaining = max(0, num_bytes - 1)
            while remaining:
                if not shared:
                    decoder, size = _read_frame(body)
                    payload_bytes += size
                count = min(BLOCK_SIZE, remaining)
                history = bytearray([previous])
                for _ in range(count):
                    probabilities, _ = predictor(history)
                    symbol = int(decoder.decode(_categorical(probabilities)))
                    history.append(symbol)
                block = bytes(history[1:])
                target.write(block)
                digest.update(block)
                previous = block[-1]
                remaining -= count
            if file_format == "v1":
                expected_digest = _read_exact(body, 32)
            if expected_digest != digest.digest():
                raise ValueError("Decoded SHA256 mismatch: corrupt data or different model probabilities.")
            if body.read(1):
                raise ValueError("Unexpected trailing data in compressed file.")
    predictor.synchronize()
    result = _metrics(num_bytes, input_path.stat().st_size, payload_bytes,
                      time.perf_counter() - started, "decode")
    result.update(decoded_sha256=digest.hexdigest(), checksum_verified=True)
    result.update(_format_metrics(file_format, num_bytes))
    return result


def verify_files(original_path, decoded_path):
    """Compare every byte, independently of the container's hash check."""
    with Path(original_path).open("rb") as original, Path(decoded_path).open("rb") as decoded:
        while True:
            left, right = original.read(1024 * 1024), decoded.read(1024 * 1024)
            if left != right:
                raise ValueError("Original and decoded files differ.")
            if not left:
                return True


def roundtrip_file(input_path, compressed_path, decoded_path, predictor, *, file_format="v2"):
    paths = [Path(path).resolve() for path in (input_path, compressed_path, decoded_path)]
    if len(set(paths)) != 3:
        raise ValueError("Original, compressed and decoded paths must be distinct.")
    for path in paths[1:]:
        if path.exists():
            raise FileExistsError(path)
    encoded = encode_file(paths[0], paths[1], predictor, file_format=file_format)
    decoded = decode_file(paths[1], paths[2], predictor)
    return {**encoded, **decoded, "files_identical": verify_files(paths[0], paths[2])}

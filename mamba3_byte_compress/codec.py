"""A correctness-first, autoregressive Constriction range codec.

The batched path batches independent 2048-byte blocks with uniform first
bytes. Legacy step and padded-prefix profiles retain their original semantics.
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
STEP_PROFILE = "recurrent-step-2048-zero-state-bf16-fp32-master-kv-bf16-cpu-f64-softmax-v1"
BATCHED_PROFILE = "batched-step-independent-2048-uniform-seed-bf16-fp32-master-kv-bf16-cpu-f64-v1"
BATCHED_V4_PROFILE = "batched-step-group-stream-2048-uniform-seed-bf16-fp32-master-kv-bf16-gpu-softmax-f32-v1"
MAGIC_V2 = b"M3RC\x02\r\n\x1a"
COMPACT_HEADER = struct.Struct("<8sBBQ32s16s")
CRC32 = struct.Struct("<I")
COMPACT_TRAILER = struct.Struct("<I32s")
PROFILE_ID = 1  # PROFILE above, including 2048 targets and the existing CDF.
STEP_PROFILE_ID = 2
BATCHED_PROFILE_ID = 3
BATCHED_V4_PROFILE_ID = 4
BATCHED_V5_PROFILE_ID = 5
BATCHED_V5_PROFILE = "batched-step-group-stream-2048-buffered-cudagraph-bf16-gpu-f32-v1"
DEFAULT_BATCH_SIZE = 128
MAX_BATCH_SIZE = 2**32 - 1  # The v3 header stores an unsigned 32-bit value.
PROFILES = {PROFILE_ID: PROFILE, STEP_PROFILE_ID: STEP_PROFILE,
            BATCHED_PROFILE_ID: BATCHED_PROFILE, BATCHED_V4_PROFILE_ID: BATCHED_V4_PROFILE, BATCHED_V5_PROFILE_ID: BATCHED_V5_PROFILE}
PROFILE_IDS = {profile: number for number, profile in PROFILES.items()}
INFERENCE_PROFILES = {"prefix": PROFILE, "step": STEP_PROFILE,
                      "batched-step": BATCHED_PROFILE,
                      "batched-step-v4": BATCHED_V4_PROFILE, "batched-step-v5": BATCHED_V5_PROFILE}
FILE_FORMATS = ("v1", "v2-framed", "v2", "v3", "v4", "v5")
MAGIC_V5 = b"M3RC\x05\r\n\x1a"
MAGIC_V3 = b"M3RC\x03\r\n\x1a"
MAGIC_V4 = b"M3RC\x04\r\n\x1a"
BATCHED_HEADER = struct.Struct("<8sBBQI32s16s")


def validate_batch_size(batch_size):
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be an integer in 1..{MAX_BATCH_SIZE} (uint32).")
    return batch_size


def _runtime_fingerprint(predictor):
    serialized = json.dumps(predictor.runtime, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).digest()[:16]


def _write_compact_header(handle, num_bytes, predictor, file_format):
    header = COMPACT_HEADER.pack(
        MAGIC_V2, PROFILE_IDS[predictor.profile], int(file_format == "v2"), num_bytes,
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


def _read_compact_header(handle, predictor=None):
    header = MAGIC_V2 + _read_exact(handle, COMPACT_HEADER.size - len(MAGIC_V2))
    if CRC32.unpack(_read_exact(handle, CRC32.size))[0] != zlib.crc32(header):
        raise ValueError("Compact header CRC32 mismatch.")
    _, profile, mode, num_bytes, checkpoint, runtime = COMPACT_HEADER.unpack(header)
    if profile not in (PROFILE_ID, STEP_PROFILE_ID) or mode not in (0, 1):
        raise ValueError("Unsupported compact profile or stream mode.")
    if predictor is None:
        return {"profile": PROFILES[profile]}
    if PROFILES[profile] != predictor.profile:
        raise ValueError("Incompatible compressed file: profile mismatch.")
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

    profile = PROFILE

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


class StepPredictor(PrefixPredictor):
    """Stateful next-byte predictions using the model's existing Mamba step.

    ``step(symbol)`` consumes one known byte and returns the next-byte distribution.
    States are zeroed before the next call after 2048 updates. ``reset()`` starts
    a new file/block. ``__call__`` adapts the codec's sequential history interface;
    histories must grow by one byte, with length 1 starting a fresh block.
    """

    profile = STEP_PROFILE

    def __init__(self, model, device, checkpoint_sha256):
        super().__init__(model, device, checkpoint_sha256)
        if self.device.type == "cuda":
            for package in ("nvidia-cutlass-dsl", "quack-kernels", "cuda-bindings"):
                self.runtime[package] = version(package)
        cache_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.states = [
            block.mixer.allocate_inference_cache(1, BLOCK_SIZE, dtype=cache_dtype)
            for block in self.model.blocks
        ]
        self._token = torch.empty((1,), dtype=torch.long, device=self.device)
        self.steps_in_block = 0

    @torch.inference_mode()
    def reset(self):
        for layer_states in self.states:
            for state in layer_states:
                state.zero_()
        self.steps_in_block = 0

    @torch.inference_mode()
    def step(self, symbol):
        if not isinstance(symbol, (int, np.integer)) or not 0 <= symbol <= 255:
            raise ValueError("Expected a byte value in 0..255.")
        if self.steps_in_block == BLOCK_SIZE:
            self.reset()
        self._token.fill_(int(symbol))
        with autocast_context(self.device):
            x = self.model.embedding(self._token)
            for block, states in zip(self.model.blocks, self.states):
                # Mamba3.step expects [batch, d_model] and updates all four
                # cache tensors in place (angle, SSM, K, V).
                mixed, *_ = block.mixer.step(block.norm1(x), *states)
                x = x + mixed
                x = x + block.mlp(block.norm2(x))
            logits = self.model.lm_head(self.model.final_norm(x)).float()
        if logits.shape != (1, 256):
            raise ValueError("The step model must output [1, 256] logits.")
        last = logits[0].detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(last).all():
            raise FloatingPointError("The model produced non-finite logits.")
        log_probs = last.log_softmax(dim=-1).numpy()
        self.steps_in_block += 1
        return np.exp(log_probs), log_probs

    def __call__(self, history):
        if not 1 <= len(history) <= BLOCK_SIZE:
            raise ValueError("Expected 1..2048 known bytes in the current block.")
        if len(history) == 1:
            self.reset()
        elif len(history) != self.steps_in_block + 1:
            raise ValueError("Step histories must grow by one byte; reset at each block.")
        return self.step(history[-1])


class BatchedStepPredictor(PrefixPredictor):
    """One independent recurrent stream per slot; at most 2047 steps per block.

    reset(B) starts a group of B blocks, where B <= the file's batch_size.
    Finished slots receive zero inputs and keep their position until group end.
    """

    profile = BATCHED_PROFILE

    def __init__(self, model, device, checkpoint_sha256, batch_size=DEFAULT_BATCH_SIZE):
        self.batch_size = validate_batch_size(batch_size)
        super().__init__(model, device, checkpoint_sha256)
        if self.device.type == "cuda":
            for package in ("nvidia-cutlass-dsl", "quack-kernels", "cuda-bindings"):
                self.runtime[package] = version(package)
        self.active_batch_size = 0
        self.states = []
        self.steps_in_block = 0
        if self.device.type == "cuda":
            from .codec_step_adapter import AlignedMambaStep

            self._layer_steps = [AlignedMambaStep(block.mixer) for block in self.model.blocks]
        else:
            self._layer_steps = [block.mixer.step for block in self.model.blocks]

    @torch.inference_mode()
    def reset(self, batch_size=None):
        if batch_size is None:
            batch_size = self.active_batch_size or self.batch_size
        validate_batch_size(batch_size)
        if batch_size > self.batch_size:
            raise ValueError("Active batch exceeds the configured batch_size.")
        if batch_size != self.active_batch_size:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
            self.states = [block.mixer.allocate_inference_cache(batch_size, BLOCK_SIZE, dtype=dtype)
                           for block in self.model.blocks]
            self._token = torch.empty(batch_size, dtype=torch.long, device=self.device)
            self._host_token = torch.empty(batch_size, dtype=torch.long,
                                           pin_memory=self.device.type == "cuda")
            self.active_batch_size = batch_size
        else:
            for layer_states in self.states:
                for state in layer_states:
                    state.zero_()
        self.steps_in_block = 0

    @torch.inference_mode()
    def step(self, tokens, *, active=None):
        tokens = np.asarray(tokens)
        if (not self.active_batch_size or tokens.shape != (self.active_batch_size,)
                or not np.issubdtype(tokens.dtype, np.integer)
                or np.any(tokens < 0) or np.any(tokens > 255)):
            raise ValueError("reset(B) then supply B byte values in 0..255.")
        if self.steps_in_block >= BLOCK_SIZE - 1:
            raise ValueError("An independent block allows 2047 predictions; reset the batch.")
        if active is not None:
            active = np.asarray(active, dtype=bool)
            if active.shape != tokens.shape:
                raise ValueError("Active mask must match the batch shape.")
        self._host_token.numpy()[:] = tokens
        self._token.copy_(self._host_token, non_blocking=True)
        with autocast_context(self.device):
            x = self.model.embedding(self._token)
            for block, states, step in zip(self.model.blocks, self.states, self._layer_steps):
                mixed, *_ = step(block.norm1(x), *states)
                x = x + mixed
                x = x + block.mlp(block.norm2(x))
            logits = self.model.lm_head(self.model.final_norm(x)).float()
        if logits.shape != (self.active_batch_size, 256):
            raise ValueError("The batched model must output [B, 256] logits.")
        last = logits.detach().to(device="cpu", dtype=torch.float64)
        checked = last if active is None else last[torch.from_numpy(active)]
        if not torch.isfinite(checked).all():
            raise FloatingPointError("The model produced non-finite logits.")
        log_probs = last.log_softmax(dim=-1).numpy()
        self.steps_in_block += 1
        return np.exp(log_probs), log_probs

    def __call__(self, tokens):
        return self.step(tokens)


class BatchedStepV4Predictor(BatchedStepPredictor):
    """Batched step predictor with GPU softmax and reusable pinned FP32 output."""

    profile = BATCHED_V4_PROFILE

    @torch.inference_mode()
    def reset(self, batch_size=None):
        previous = self.active_batch_size
        super().reset(batch_size)
        if self.active_batch_size != previous:
            self._host_probabilities = torch.empty(
                (self.active_batch_size, 256), dtype=torch.float32,
                pin_memory=self.device.type == "cuda")

    def _compute(self):
        with autocast_context(self.device):
            x = self.model.embedding(self._token)
            for block, states, step in zip(self.model.blocks, self.states, self._layer_steps):
                mixed, *_ = step(block.norm1(x), *states)
                x = x + mixed
                x = x + block.mlp(block.norm2(x))
            logits = self.model.lm_head(self.model.final_norm(x)).float()
        if logits.shape != (self.active_batch_size, 256):
            raise ValueError("The batched model must output [B, 256] logits.")
        return logits.softmax(dim=-1)

    @torch.inference_mode()
    def step(self, tokens, *, active=None):
        tokens = np.asarray(tokens)
        if (not self.active_batch_size or tokens.shape != (self.active_batch_size,)
                or not np.issubdtype(tokens.dtype, np.integer)
                or np.any(tokens < 0) or np.any(tokens > 255)):
            raise ValueError("reset(B) then supply B byte values in 0..255.")
        if self.steps_in_block >= BLOCK_SIZE - 1:
            raise ValueError("An independent block allows 2047 predictions; reset the batch.")
        if active is not None:
            active = np.asarray(active, dtype=bool)
            if active.shape != tokens.shape:
                raise ValueError("Active mask must match the batch shape.")
        self._host_token.numpy()[:] = tokens
        self._token.copy_(self._host_token, non_blocking=True)
        probabilities = self._compute()
        self._host_probabilities.copy_(probabilities, non_blocking=False)
        result = self._host_probabilities.numpy()
        self.steps_in_block += 1
        return result, None


def load_predictor(checkpoint, device="cuda:0", *, inference="batched-step-v5",
                   batch_size=DEFAULT_BATCH_SIZE):
    """Load the existing project checkpoint without changing its architecture."""
    if inference not in INFERENCE_PROFILES:
        raise ValueError(f"Unknown inference mode: {inference}")
    validate_batch_size(batch_size)
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

    if inference in ("step", "batched-step", "batched-step-v4", "batched-step-v5"):
        from mamba_ssm.modules.mamba3 import mamba3_step_fn

        if mamba3_step_fn is None:
            raise RuntimeError("Mamba step requires nvidia-cutlass-dsl and quack-kernels.")
    model, metadata = load_mamba3_checkpoint(checkpoint)
    del metadata
    if inference == "batched-step":
        return BatchedStepPredictor(model, device, sha256_file(checkpoint), batch_size)
    if inference == "batched-step-v5":
        from .codec_graph import BatchedStepV5Predictor

        return BatchedStepV5Predictor(model, device, sha256_file(checkpoint), batch_size)
    if inference == "batched-step-v4":
        return BatchedStepV4Predictor(model, device, sha256_file(checkpoint), batch_size)
    predictor_type = StepPredictor if inference == "step" else PrefixPredictor
    return predictor_type(model, device, sha256_file(checkpoint))


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
        "profile": predictor.profile,
        "checkpoint_sha256": predictor.checkpoint_sha256,
        "runtime": predictor.runtime,
    }


def _read_header(handle, predictor=None):
    magic = _read_exact(handle, len(MAGIC))
    if magic == MAGIC_V3:
        from .codec_batched import read_v3_header

        return read_v3_header(handle, predictor)
    if magic in (MAGIC_V4, MAGIC_V5):
        from .codec_batched_v4 import read_v4_header

        return read_v4_header(handle, predictor, version=5 if magic == MAGIC_V5 else 4)
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
    if metadata.get("profile") not in (PROFILE, STEP_PROFILE):
        raise ValueError("Unsupported probability profile.")
    if predictor is None:
        return metadata
    expected = _metadata(num_bytes, predictor)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Incompatible compressed file: {key} mismatch.")
    return {**metadata, "file_format": "v1"}


def read_inference_mode(path):
    """Select a predictor from the header; decode still validates all metadata."""
    with Path(path).open("rb") as handle:
        profile = _read_header(handle)["profile"]
    return next(mode for mode, value in INFERENCE_PROFILES.items() if value == profile)


def read_codec_settings(path):
    """Read the prediction mode and fixed batch grouping before loading weights."""
    with Path(path).open("rb") as handle:
        metadata = _read_header(handle)
    settings = {"inference": next(mode for mode, profile in INFERENCE_PROFILES.items()
                                   if profile == metadata["profile"])}
    if "batch_size" in metadata:
        settings["batch_size"] = metadata["batch_size"]
    return settings


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


def encode_file(input_path, output_path, predictor, *, file_format=None):
    """Write a real .m3rc file; return BPB and end-to-end coding throughput."""
    defaults = {BATCHED_PROFILE: "v3", BATCHED_V4_PROFILE: "v4", BATCHED_V5_PROFILE: "v5"}
    file_format = file_format or defaults.get(predictor.profile, "v2")
    expected = {"v3": BATCHED_PROFILE, "v4": BATCHED_V4_PROFILE, "v5": BATCHED_V5_PROFILE}
    if file_format in expected and predictor.profile != expected[file_format]:
        raise ValueError(f"{file_format} requires its matching batched-step predictor.")
    if file_format not in expected and predictor.profile in expected.values():
        raise ValueError("Batched-step predictors require their matching v3, v4 or v5 format.")
    if file_format == "v3":
        from .codec_batched import encode_batched_file

        return encode_batched_file(input_path, output_path, predictor)
    if file_format in ("v4", "v5"):
        from .codec_batched_v4 import encode_batched_v4_file

        return encode_batched_v4_file(input_path, output_path, predictor)
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
        "profile": predictor.profile,
    })
    result.update(_format_metrics(file_format, num_bytes))
    return result


def decode_file(input_path, output_path, predictor):
    """Decode using only the compressed file, checkpoint and decoded history."""
    if predictor.profile == BATCHED_PROFILE:
        from .codec_batched import decode_batched_file

        return decode_batched_file(input_path, output_path, predictor)
    if predictor.profile in (BATCHED_V4_PROFILE, BATCHED_V5_PROFILE):
        from .codec_batched_v4 import decode_batched_v4_file

        return decode_batched_v4_file(input_path, output_path, predictor)
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
    result.update(decoded_sha256=digest.hexdigest(), checksum_verified=True,
                  profile=predictor.profile)
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


def roundtrip_file(input_path, compressed_path, decoded_path, predictor, *, file_format=None):
    paths = [Path(path).resolve() for path in (input_path, compressed_path, decoded_path)]
    if len(set(paths)) != 3:
        raise ValueError("Original, compressed and decoded paths must be distinct.")
    for path in paths[1:]:
        if path.exists():
            raise FileExistsError(path)
    encoded = encode_file(paths[0], paths[1], predictor, file_format=file_format)
    decoded = decode_file(paths[1], paths[2], predictor)
    return {**encoded, **decoded, "files_identical": verify_files(paths[0], paths[2])}

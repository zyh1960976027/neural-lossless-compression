"""v4 codec: one Range stream per batch group and batched probability models."""

import contextlib
import hashlib
import math
import os
from pathlib import Path
import time
import zlib

import constriction
import numpy as np

from . import codec as c


def read_v4_header(handle, predictor=None, *, version=4):
    magic, profile_id, expected_profile = _format_config(version)
    header = magic + c._read_exact(handle, c.BATCHED_HEADER.size - len(magic))
    if c.CRC32.unpack(c._read_exact(handle, c.CRC32.size))[0] != zlib.crc32(header):
        raise ValueError(f"v{version} header CRC32 mismatch.")
    _, profile, flags, count, batch_size, checkpoint, runtime = c.BATCHED_HEADER.unpack(header)
    if profile != profile_id or flags != 0:
        raise ValueError(f"Unsupported v{version} profile or flags.")
    c.validate_batch_size(batch_size)
    if predictor is not None:
        if predictor.profile != expected_profile:
            raise ValueError("Incompatible compressed file: profile mismatch.")
        if predictor.batch_size != batch_size:
            raise ValueError("Incompatible compressed file: batch_size mismatch.")
        if checkpoint.hex() != predictor.checkpoint_sha256:
            raise ValueError("Incompatible compressed file: checkpoint_sha256 mismatch.")
        if runtime != c._runtime_fingerprint(predictor):
            raise ValueError("Incompatible compressed file: runtime mismatch.")
    return dict(version=version, file_format=f"v{version}", profile=expected_profile,
                num_bytes=count, block_size=c.BLOCK_SIZE, batch_size=batch_size)


def _format_config(version):
    if version == 5:
        return c.MAGIC_V5, c.BATCHED_V5_PROFILE_ID, c.BATCHED_V5_PROFILE
    return c.MAGIC_V4, c.BATCHED_V4_PROFILE_ID, c.BATCHED_V4_PROFILE


def _write_header(handle, count, predictor):
    magic, profile_id, _ = _format_config(5 if predictor.profile == c.BATCHED_V5_PROFILE else 4)
    header = c.BATCHED_HEADER.pack(
        magic, profile_id, 0, count,
        c.validate_batch_size(predictor.batch_size),
        bytes.fromhex(predictor.checkpoint_sha256), c._runtime_fingerprint(predictor))
    handle.write(header)
    handle.write(c.CRC32.pack(zlib.crc32(header)))


def _group_lengths(remaining, batch_size):
    count = min(remaining, batch_size * c.BLOCK_SIZE)
    return np.minimum(c.BLOCK_SIZE, count - np.arange(0, count, c.BLOCK_SIZE))


def _read_group_frame(handle, max_symbols):
    size, checksum = c.FRAME.unpack(c._read_exact(handle, c.FRAME.size))
    if size == 0 or size % 4 or size > max_symbols * 4 + 64:
        raise ValueError("Invalid group Range payload length.")
    payload = c._read_exact(handle, size)
    if zlib.crc32(payload) != checksum:
        raise ValueError("Group Range payload CRC32 mismatch.")
    words = np.frombuffer(payload, dtype="<u4").astype(np.uint32, copy=True)
    return constriction.stream.queue.RangeDecoder(words), size


def _metrics(count, packed_size, payload_size, seconds, operation, predictor, steps):
    blocks = (count + c.BLOCK_SIZE - 1) // c.BLOCK_SIZE
    batches = (blocks + predictor.batch_size - 1) // predictor.batch_size
    result = c._metrics(count, packed_size, payload_size, seconds, operation)
    version = 5 if predictor.profile == c.BATCHED_V5_PROFILE else 4
    result.update(file_format=f"v{version}", format_version=version, profile=predictor.profile,
                  batch_size=predictor.batch_size, num_blocks=blocks, num_batches=batches,
                  num_prediction_blocks=blocks, num_model_steps=steps,
                  predicted_bytes=count - blocks, uniform_bytes=blocks,
                  uniform_first_byte_only=False, num_range_streams=batches,
                  frame_header_bytes=batches * c.FRAME.size)
    if hasattr(predictor, "execution_metrics"):
        result.update(predictor.execution_metrics())
    return result


def _active_rows(probabilities, active):
    return probabilities if bool(np.all(active)) else probabilities[active]


def encode_batched_v4_file(input_path, output_path, predictor, *, optimized=True):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files.")
    predictor.synchronize()
    started = time.perf_counter()
    digest = hashlib.sha256()
    payload_bytes = steps = 0
    model_nll = 0.0
    family = constriction.stream.model.Categorical(perfect=False)
    uniform = c._uniform()
    with c._coding_mode(predictor), input_path.open("rb") as source, c._new_output(output_path) as target, contextlib.ExitStack() as cleanup:
        count = os.fstat(source.fileno()).st_size
        _write_header(target, count, predictor)
        body = c._BodyWriter(target)
        remaining = count
        while remaining:
            lengths = _group_lengths(remaining, predictor.batch_size)
            data = c._read_exact(source, int(lengths.sum()))
            digest.update(data)
            values = np.zeros((len(lengths), int(lengths.max())), dtype=np.uint8)
            offset = 0
            for lane, length in enumerate(lengths):
                values[lane, :length] = np.frombuffer(
                    data, dtype=np.uint8, count=int(length), offset=offset)
                offset += int(length)
            encoder = constriction.stream.queue.RangeEncoder()
            encoder.encode(values[:, 0].astype(np.int32), uniform)
            predictor.reset(len(lengths))
            from .codec_graph import BatchedStepV5Predictor
            prefetch = (optimized and isinstance(predictor, BatchedStepV5Predictor)
                        and predictor.device.type == "cuda" and predictor.use_graph)
            pending = predictor.iter_prefetched(values, lengths) if prefetch else None
            if pending is not None:
                cleanup.callback(pending.close)
            for position in range(1, values.shape[1]):
                active = lengths > position
                tokens = np.where(active, values[:, position - 1], 0)
                probabilities, _ = next(pending) if prefetch else predictor.step(tokens, active=active)
                symbols = values[active, position].astype(np.int32)
                active_probabilities = (probabilities[:int(active.sum())] if optimized
                                        else _active_rows(probabilities, active))
                encoder.encode(symbols, family, active_probabilities)
                selected = active_probabilities[np.arange(len(symbols)), symbols]
                if np.any(selected <= 0):
                    raise FloatingPointError("A coded symbol has zero floating-point probability.")
                model_nll -= float(np.log(selected.astype(np.float64)).sum())
                steps += 1
            if pending is not None:
                pending.close()
            payload_bytes += c._write_frame(body, encoder)
            remaining -= len(data)
        if source.read(1):
            raise ValueError("Input file size changed during encoding.")
        target.write(c.COMPACT_TRAILER.pack(body.crc, digest.digest()))
    predictor.synchronize()
    result = _metrics(count, output_path.stat().st_size, payload_bytes,
                      time.perf_counter() - started, "encode", predictor, steps)
    uniform_nll = result["uniform_bytes"] * math.log(256)
    nll = model_nll + uniform_nll
    result.update(theoretical_nll_bpb=nll / (count * math.log(2)) if count else 0.0,
                  nll_nats_total=nll, nll_nats_model=model_nll,
                  nll_nats_uniform_seeds=uniform_nll, original_sha256=digest.hexdigest(),
                  checkpoint_sha256=predictor.checkpoint_sha256, runtime=predictor.runtime)
    return result


def decode_batched_v4_file(input_path, output_path, predictor, *, optimized=True):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files.")
    predictor.synchronize()
    started = time.perf_counter()
    digest = hashlib.sha256()
    payload_bytes = steps = 0
    family = constriction.stream.model.Categorical(perfect=False)
    uniform = c._uniform()
    with c._coding_mode(predictor), input_path.open("rb") as source:
        metadata = c._read_header(source, predictor)
        count = metadata["num_bytes"]
        body, expected_digest = c._read_compact_body(source)
        with c._new_output(output_path) as target:
            remaining = count
            while remaining:
                lengths = _group_lengths(remaining, predictor.batch_size)
                max_symbols = int(lengths.sum())
                decoder, size = _read_group_frame(body, max_symbols)
                payload_bytes += size
                values = np.zeros((len(lengths), int(lengths.max())), dtype=np.uint8)
                values[:, 0] = decoder.decode(uniform, len(lengths)).astype(np.uint8)
                predictor.reset(len(lengths))
                for position in range(1, values.shape[1]):
                    active = lengths > position
                    tokens = np.where(active, values[:, position - 1], 0)
                    probabilities, _ = predictor.step(tokens, active=active)
                    active_probabilities = (probabilities[:int(active.sum())] if optimized
                                            else _active_rows(probabilities, active))
                    values[active, position] = decoder.decode(family, active_probabilities).astype(np.uint8)
                    steps += 1
                for lane, length in enumerate(lengths):
                    data = values[lane, :length].tobytes()
                    target.write(data)
                    digest.update(data)
                remaining -= int(lengths.sum())
            if body.read(1):
                raise ValueError("Unexpected trailing data in compressed file.")
            if digest.digest() != expected_digest:
                raise ValueError("Decoded SHA256 mismatch: corrupt data or different probabilities.")
    predictor.synchronize()
    result = _metrics(count, input_path.stat().st_size, payload_bytes,
                      time.perf_counter() - started, "decode", predictor, steps)
    result.update(decoded_sha256=digest.hexdigest(), checksum_verified=True)
    return result

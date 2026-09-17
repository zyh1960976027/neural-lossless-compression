"""v3 container: independent 2048-byte Range streams with batched Mamba state."""

import hashlib
import math
import os
from pathlib import Path
import time
import zlib

import constriction
import numpy as np

from . import codec as c


def read_v3_header(handle, predictor=None):
    # The dispatcher has already consumed the magic.
    header = c.MAGIC_V3 + c._read_exact(handle, c.BATCHED_HEADER.size - len(c.MAGIC_V3))
    if c.CRC32.unpack(c._read_exact(handle, c.CRC32.size))[0] != zlib.crc32(header):
        raise ValueError("v3 header CRC32 mismatch.")
    _, profile, flags, count, batch_size, checkpoint, runtime = c.BATCHED_HEADER.unpack(header)
    if profile != c.BATCHED_PROFILE_ID or flags != 0:
        raise ValueError("Unsupported v3 profile or flags.")
    c.validate_batch_size(batch_size)
    if predictor is not None:
        if predictor.profile != c.BATCHED_PROFILE:
            raise ValueError("Incompatible compressed file: profile mismatch.")
        if predictor.batch_size != batch_size:
            raise ValueError("Incompatible compressed file: batch_size mismatch.")
        if checkpoint.hex() != predictor.checkpoint_sha256:
            raise ValueError("Incompatible compressed file: checkpoint_sha256 mismatch.")
        if runtime != c._runtime_fingerprint(predictor):
            raise ValueError("Incompatible compressed file: runtime mismatch.")
    return dict(version=3, file_format="v3", profile=c.BATCHED_PROFILE,
                num_bytes=count, block_size=c.BLOCK_SIZE, batch_size=batch_size)


def _write_header(handle, count, predictor):
    header = c.BATCHED_HEADER.pack(c.MAGIC_V3, c.BATCHED_PROFILE_ID, 0, count,
                                  c.validate_batch_size(predictor.batch_size),
                                  bytes.fromhex(predictor.checkpoint_sha256),
                                  c._runtime_fingerprint(predictor))
    handle.write(header)
    handle.write(c.CRC32.pack(zlib.crc32(header)))


def _group_lengths(remaining, batch_size):
    count = min(remaining, batch_size * c.BLOCK_SIZE)
    return np.minimum(c.BLOCK_SIZE, count - np.arange(0, count, c.BLOCK_SIZE))


def _metrics(count, packed_size, payload_size, seconds, operation, predictor, steps):
    blocks = (count + c.BLOCK_SIZE - 1) // c.BLOCK_SIZE
    result = c._metrics(count, packed_size, payload_size, seconds, operation)
    result.update(file_format="v3", format_version=3, profile=predictor.profile,
                  batch_size=predictor.batch_size, num_blocks=blocks,
                  num_batches=(blocks + predictor.batch_size - 1) // predictor.batch_size,
                  num_prediction_blocks=blocks, num_model_steps=steps,
                  predicted_bytes=count - blocks, uniform_bytes=blocks,
                  uniform_first_byte_only=False, num_range_streams=blocks,
                  frame_header_bytes=blocks * c.FRAME.size)
    return result


def encode_batched_file(input_path, output_path, predictor):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files.")
    predictor.synchronize()
    started = time.perf_counter()
    digest = hashlib.sha256()
    payload_bytes = steps = 0
    model_nll = 0.0
    with c._coding_mode(predictor), input_path.open("rb") as source, c._new_output(output_path) as target:
        count = os.fstat(source.fileno()).st_size
        _write_header(target, count, predictor)
        body = c._BodyWriter(target)
        remaining = count
        uniform = c._uniform()
        while remaining:
            lengths = _group_lengths(remaining, predictor.batch_size)
            data = c._read_exact(source, int(lengths.sum()))
            digest.update(data)
            values = np.zeros((len(lengths), int(lengths.max())), dtype=np.uint8)
            offset = 0
            encoders = []
            for lane, length in enumerate(lengths):
                values[lane, :length] = np.frombuffer(data, dtype=np.uint8, count=int(length), offset=offset)
                offset += int(length)
                encoder = constriction.stream.queue.RangeEncoder()
                encoder.encode(int(values[lane, 0]), uniform)
                encoders.append(encoder)
            predictor.reset(len(lengths))
            for position in range(1, values.shape[1]):
                active = lengths > position
                tokens = np.where(active, values[:, position - 1], 0)
                probabilities, log_probs = predictor.step(tokens, active=active)
                steps += 1
                for lane in np.flatnonzero(active):
                    symbol = int(values[lane, position])
                    encoders[lane].encode(symbol, c._categorical(probabilities[lane]))
                    model_nll -= float(log_probs[lane, symbol])
            for encoder in encoders:
                payload_bytes += c._write_frame(body, encoder)
            remaining -= len(data)
        if source.read(1):
            raise ValueError("Input file size changed during encoding.")
        if not math.isfinite(model_nll):
            raise FloatingPointError("Non-finite theoretical NLL.")
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


def decode_batched_file(input_path, output_path, predictor):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files.")
    predictor.synchronize()
    started = time.perf_counter()
    digest = hashlib.sha256()
    payload_bytes = steps = 0
    with c._coding_mode(predictor), input_path.open("rb") as source:
        metadata = c._read_header(source, predictor)
        count = metadata["num_bytes"]
        body, expected_digest = c._read_compact_body(source)
        with c._new_output(output_path) as target:
            remaining = count
            uniform = c._uniform()
            while remaining:
                lengths = _group_lengths(remaining, predictor.batch_size)
                values = np.zeros((len(lengths), int(lengths.max())), dtype=np.uint8)
                decoders = []
                for lane in range(len(lengths)):
                    decoder, size = c._read_frame(body)
                    payload_bytes += size
                    values[lane, 0] = int(decoder.decode(uniform))
                    decoders.append(decoder)
                predictor.reset(len(lengths))
                for position in range(1, values.shape[1]):
                    active = lengths > position
                    tokens = np.where(active, values[:, position - 1], 0)
                    probabilities, _ = predictor.step(tokens, active=active)
                    steps += 1
                    for lane in np.flatnonzero(active):
                        values[lane, position] = int(decoders[lane].decode(c._categorical(probabilities[lane])))
                for lane, length in enumerate(lengths):
                    data = values[lane, :length].tobytes()
                    target.write(data)
                    digest.update(data)
                remaining -= int(lengths.sum())
            if body.read(1):
                raise ValueError("Unexpected trailing data in compressed file.")
            if digest.digest() != expected_digest:
                raise ValueError("Decoded SHA256 mismatch: corrupt data or different model probabilities.")
    predictor.synchronize()
    result = _metrics(count, input_path.stat().st_size, payload_bytes,
                      time.perf_counter() - started, "decode", predictor, steps)
    result.update(decoded_sha256=digest.hexdigest(), checksum_verified=True)
    return result

"""v6 MTP-4 codec with batched four-token state continuation."""

import hashlib
import math
import os
from pathlib import Path
import time
import zlib

import constriction
import numpy as np

from . import codec as c
from .codec_batched_v4 import _group_lengths, _read_group_frame


def _phase_start(timings):
    return time.perf_counter() if timings is not None else None


def _phase_stop(timings, name, started):
    if timings is not None:
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - started


def read_v6_header(handle, predictor=None):
    header = c.MAGIC_V6 + c._read_exact(
        handle, c.BATCHED_HEADER.size - len(c.MAGIC_V6)
    )
    if c.CRC32.unpack(c._read_exact(handle, c.CRC32.size))[0] != zlib.crc32(header):
        raise ValueError("v6 header CRC32 mismatch.")
    _, profile, flags, count, batch_size, checkpoint, runtime = (
        c.BATCHED_HEADER.unpack(header)
    )
    if profile != c.MTP4_PROFILE_ID or flags != 0:
        raise ValueError("Unsupported v6 profile or flags.")
    c.validate_batch_size(batch_size)
    if predictor is not None:
        if predictor.profile != c.MTP4_PROFILE:
            raise ValueError("Incompatible compressed file: profile mismatch.")
        if predictor.batch_size != batch_size:
            raise ValueError("Incompatible compressed file: batch_size mismatch.")
        if checkpoint.hex() != predictor.checkpoint_sha256:
            raise ValueError("Incompatible compressed file: checkpoint_sha256 mismatch.")
        if runtime != c._runtime_fingerprint(predictor):
            raise ValueError("Incompatible compressed file: runtime mismatch.")
    return {
        "version": 6,
        "file_format": "v6",
        "profile": c.MTP4_PROFILE,
        "num_bytes": count,
        "block_size": c.BLOCK_SIZE,
        "batch_size": batch_size,
    }


def _write_header(handle, count, predictor):
    header = c.BATCHED_HEADER.pack(
        c.MAGIC_V6,
        c.MTP4_PROFILE_ID,
        0,
        count,
        c.validate_batch_size(predictor.batch_size),
        bytes.fromhex(predictor.checkpoint_sha256),
        c._runtime_fingerprint(predictor),
    )
    handle.write(header)
    handle.write(c.CRC32.pack(zlib.crc32(header)))


def _prediction_groups(lengths):
    return int(np.maximum(0, (lengths.astype(np.int64) - 1 + 3) // 4).sum())


def _metrics(
    count,
    packed_size,
    payload_size,
    seconds,
    operation,
    predictor,
    *,
    forwards,
    prediction_groups,
    state_tokens,
    head_counts,
):
    blocks = (count + c.BLOCK_SIZE - 1) // c.BLOCK_SIZE
    batches = (blocks + predictor.batch_size - 1) // predictor.batch_size
    result = c._metrics(count, packed_size, payload_size, seconds, operation)
    result.update(
        file_format="v6",
        format_version=6,
        profile=predictor.profile,
        batch_size=predictor.batch_size,
        num_blocks=blocks,
        num_batches=batches,
        num_prediction_blocks=blocks,
        num_prediction_groups=prediction_groups,
        num_model_forwards=forwards,
        num_model_steps=forwards,
        state_tokens_processed=state_tokens,
        predicted_bytes=count - blocks,
        uniform_bytes=blocks,
        uniform_first_byte_only=False,
        num_range_streams=batches,
        frame_header_bytes=batches * c.FRAME.size,
    )
    for head, head_count in enumerate(head_counts, start=1):
        result[f"head{head}_predicted_bytes"] = head_count
    if hasattr(predictor, "execution_metrics"):
        result.update(predictor.execution_metrics())
    return result


def _values(data, lengths):
    values = np.zeros((len(lengths), int(lengths.max())), dtype=np.uint8)
    offset = 0
    for lane, length in enumerate(lengths):
        values[lane, :length] = np.frombuffer(
            data, dtype=np.uint8, count=int(length), offset=offset
        )
        offset += int(length)
    return values


def _encode_predictions(
    encoder, family, values, lengths, predictor, head_nll, timings=None
):
    forwards = state_tokens = 0
    head_counts = [0, 0, 0, 0]
    active = lengths > 1
    if not np.any(active):
        return forwards, state_tokens, head_counts
    phase = _phase_start(timings)
    probabilities = predictor.start(values[:, 0], active=active)
    _phase_stop(timings, "predictor", phase)
    forwards += 1
    state_tokens += int(active.sum())
    context = 0
    while np.any(lengths > context + 1):
        for head in range(4):
            position = context + head + 1
            active = lengths > position
            if not np.any(active):
                continue
            phase = _phase_start(timings)
            symbols = values[active, position].astype(np.int32)
            selected_probabilities = probabilities[active, head]
            _phase_stop(timings, "probability_cpu", phase)
            phase = _phase_start(timings)
            encoder.encode(symbols, family, selected_probabilities)
            _phase_stop(timings, "range_coding", phase)
            phase = _phase_start(timings)
            selected = selected_probabilities[np.arange(len(symbols)), symbols]
            if np.any(selected <= 0):
                raise FloatingPointError(
                    "An MTP-coded symbol has zero floating-point probability."
                )
            head_nll[head] -= float(np.log(selected.astype(np.float64)).sum())
            head_counts[head] += len(symbols)
            _phase_stop(timings, "probability_cpu", phase)
        context += 4
        active = lengths > context + 1
        if np.any(active):
            tokens = values[:, context - 3:context + 1]
            if tokens.shape != (len(lengths), 4):
                raise AssertionError("MTP continuation group is not four bytes.")
            phase = _phase_start(timings)
            probabilities = predictor.advance(tokens, active=active)
            _phase_stop(timings, "predictor", phase)
            forwards += 1
            state_tokens += 4 * int(active.sum())
    return forwards, state_tokens, head_counts


def encode_mtp4_file(input_path, output_path, predictor):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files.")
    predictor.synchronize()
    started = time.perf_counter()
    digest = hashlib.sha256()
    payload_bytes = forwards = prediction_groups = state_tokens = 0
    head_nll = [0.0, 0.0, 0.0, 0.0]
    head_counts = [0, 0, 0, 0]
    family = constriction.stream.model.Categorical(perfect=False)
    uniform = c._uniform()
    timings = {} if getattr(predictor, "collect_phase_timings", False) else None
    with (
        c._coding_mode(predictor),
        input_path.open("rb") as source,
        c._new_output(output_path) as target,
    ):
        count = os.fstat(source.fileno()).st_size
        _write_header(target, count, predictor)
        body = c._BodyWriter(target)
        remaining = count
        while remaining:
            lengths = _group_lengths(remaining, predictor.batch_size)
            phase = _phase_start(timings)
            data = c._read_exact(source, int(lengths.sum()))
            digest.update(data)
            values = _values(data, lengths)
            _phase_stop(timings, "input_read_hash_pack", phase)
            phase = _phase_start(timings)
            encoder = constriction.stream.queue.RangeEncoder()
            encoder.encode(values[:, 0].astype(np.int32), uniform)
            _phase_stop(timings, "range_coding", phase)
            phase = _phase_start(timings)
            predictor.reset(len(lengths))
            if timings is not None:
                predictor.synchronize()
            _phase_stop(timings, "state_reset", phase)
            group_forwards, group_state_tokens, group_head_counts = (
                _encode_predictions(
                    encoder, family, values, lengths, predictor, head_nll, timings
                )
            )
            forwards += group_forwards
            state_tokens += group_state_tokens
            prediction_groups += _prediction_groups(lengths)
            head_counts = [
                total + added
                for total, added in zip(head_counts, group_head_counts)
            ]
            phase = _phase_start(timings)
            payload_bytes += c._write_frame(body, encoder)
            _phase_stop(timings, "container_write", phase)
            remaining -= len(data)
        if source.read(1):
            raise ValueError("Input file size changed during encoding.")
        phase = _phase_start(timings)
        target.write(c.COMPACT_TRAILER.pack(body.crc, digest.digest()))
        _phase_stop(timings, "container_write", phase)
    predictor.synchronize()
    result = _metrics(
        count,
        output_path.stat().st_size,
        payload_bytes,
        time.perf_counter() - started,
        "encode",
        predictor,
        forwards=forwards,
        prediction_groups=prediction_groups,
        state_tokens=state_tokens,
        head_counts=head_counts,
    )
    uniform_nll = result["uniform_bytes"] * math.log(256)
    model_nll = sum(head_nll)
    nll = model_nll + uniform_nll
    result.update(
        theoretical_nll_bpb=nll / (count * math.log(2)) if count else 0.0,
        nll_nats_total=nll,
        nll_nats_model=model_nll,
        nll_nats_uniform_seeds=uniform_nll,
        original_sha256=digest.hexdigest(),
        checkpoint_sha256=predictor.checkpoint_sha256,
        runtime=predictor.runtime,
    )
    for head, value in enumerate(head_nll, start=1):
        result[f"head{head}_nll_nats"] = value
    if timings is not None:
        accounted = sum(timings.values())
        timings["other"] = max(0.0, result["encode_seconds"] - accounted)
        result["phase_seconds"] = timings
    return result


def _decode_predictions(decoder, family, values, lengths, predictor, timings=None):
    forwards = state_tokens = 0
    head_counts = [0, 0, 0, 0]
    active = lengths > 1
    if not np.any(active):
        return forwards, state_tokens, head_counts
    phase = _phase_start(timings)
    probabilities = predictor.start(values[:, 0], active=active)
    _phase_stop(timings, "predictor", phase)
    forwards += 1
    state_tokens += int(active.sum())
    context = 0
    while np.any(lengths > context + 1):
        for head in range(4):
            position = context + head + 1
            active = lengths > position
            if not np.any(active):
                continue
            phase = _phase_start(timings)
            selected_probabilities = probabilities[active, head]
            _phase_stop(timings, "probability_cpu", phase)
            phase = _phase_start(timings)
            decoded = decoder.decode(family, selected_probabilities)
            _phase_stop(timings, "range_coding", phase)
            phase = _phase_start(timings)
            values[active, position] = decoded.astype(np.uint8)
            head_counts[head] += int(active.sum())
            _phase_stop(timings, "probability_cpu", phase)
        context += 4
        active = lengths > context + 1
        if np.any(active):
            phase = _phase_start(timings)
            probabilities = predictor.advance(
                values[:, context - 3:context + 1], active=active
            )
            _phase_stop(timings, "predictor", phase)
            forwards += 1
            state_tokens += 4 * int(active.sum())
    return forwards, state_tokens, head_counts


def decode_mtp4_file(input_path, output_path, predictor):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files.")
    predictor.synchronize()
    started = time.perf_counter()
    digest = hashlib.sha256()
    payload_bytes = forwards = prediction_groups = state_tokens = 0
    head_counts = [0, 0, 0, 0]
    family = constriction.stream.model.Categorical(perfect=False)
    uniform = c._uniform()
    timings = {} if getattr(predictor, "collect_phase_timings", False) else None
    with c._coding_mode(predictor), input_path.open("rb") as source:
        metadata = c._read_header(source, predictor)
        count = metadata["num_bytes"]
        body, expected_digest = c._read_compact_body(source)
        with c._new_output(output_path) as target:
            remaining = count
            while remaining:
                lengths = _group_lengths(remaining, predictor.batch_size)
                phase = _phase_start(timings)
                decoder, size = _read_group_frame(body, int(lengths.sum()))
                payload_bytes += size
                values = np.zeros(
                    (len(lengths), int(lengths.max())), dtype=np.uint8
                )
                _phase_stop(timings, "container_read_and_allocate", phase)
                phase = _phase_start(timings)
                values[:, 0] = decoder.decode(uniform, len(lengths)).astype(np.uint8)
                _phase_stop(timings, "range_coding", phase)
                phase = _phase_start(timings)
                predictor.reset(len(lengths))
                if timings is not None:
                    predictor.synchronize()
                _phase_stop(timings, "state_reset", phase)
                group_forwards, group_state_tokens, group_head_counts = (
                    _decode_predictions(
                        decoder, family, values, lengths, predictor, timings
                    )
                )
                forwards += group_forwards
                state_tokens += group_state_tokens
                prediction_groups += _prediction_groups(lengths)
                head_counts = [
                    total + added
                    for total, added in zip(head_counts, group_head_counts)
                ]
                for lane, length in enumerate(lengths):
                    phase = _phase_start(timings)
                    data = values[lane, :length].tobytes()
                    target.write(data)
                    digest.update(data)
                    _phase_stop(timings, "output_write_hash", phase)
                remaining -= int(lengths.sum())
            if body.read(1):
                raise ValueError("Unexpected trailing data in compressed file.")
            if digest.digest() != expected_digest:
                raise ValueError(
                    "Decoded SHA256 mismatch: corrupt data or different probabilities."
                )
    predictor.synchronize()
    result = _metrics(
        count,
        input_path.stat().st_size,
        payload_bytes,
        time.perf_counter() - started,
        "decode",
        predictor,
        forwards=forwards,
        prediction_groups=prediction_groups,
        state_tokens=state_tokens,
        head_counts=head_counts,
    )
    result.update(decoded_sha256=digest.hexdigest(), checksum_verified=True)
    if timings is not None:
        accounted = sum(timings.values())
        timings["other"] = max(0.0, result["decode_seconds"] - accounted)
        result["phase_seconds"] = timings
    return result

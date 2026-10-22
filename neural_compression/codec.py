"""Streaming lossless token container, with an embedded SentencePiece model."""
import base64
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
import zlib
import numpy as np
import torch
from . import io as c
from .inference import Predictor, load_predictor, _coding_mode
from .data import LosslessTokenizer, token_blocks, require_baseline_tokenizer

MAGIC_GPU_INT = b"M3SP\x03\r\n\x1a"
PREFIX = struct.Struct("<8sI")
FRAME = struct.Struct("<II")
TRAILER = struct.Struct("<QQI32s")
MAX_HEADER = 8 * 1024 * 1024


def _transfer_metrics(predictor, initial_bytes):
    return dict(
        probability_transfer="gpu_int",
        probability_transfer_bytes=getattr(predictor, "probability_transfer_bytes", 0)
        - initial_bytes,
    )


def warmup(predictor, token_count):
    blocks = (token_count + 2047) // 2048
    sizes = {min(blocks, predictor.batch_size), blocks % predictor.batch_size}
    started = time.perf_counter()
    with _coding_mode(predictor):
        for size in sorted(sizes - {0}):
            predictor.reset(size)
            predictor.step(np.zeros(size, dtype=np.int32))
            if (
                isinstance(predictor, Predictor)
                and predictor.vocab_size > 256
                and predictor.use_graph
            ):
                predictor.prepare_prefetch()
    predictor.synchronize()
    return time.perf_counter() - started


def _metrics(operation, raw_bytes, tokens, packed_bytes, payload, elapsed, phases):
    return {
        "operation": operation,
        "num_bytes": raw_bytes,
        "num_tokens": tokens,
        "compressed_bytes": packed_bytes,
        "payload_bytes": payload,
        "actual_bpb": 8 * packed_bytes / raw_bytes if raw_bytes else None,
        "payload_bpb": 8 * payload / raw_bytes if raw_bytes else None,
        f"{operation}_seconds": elapsed,
        f"{operation}_kb_per_second": raw_bytes / 1000 / elapsed if raw_bytes else 0,
        f"{operation}_tokens_per_second": tokens / elapsed if tokens else 0,
        "phases_seconds": phases,
    }


def encode(
    input_path, output_path, predictor, tokenizer, *, profile=False, optimized=True
):
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Input and output must differ")
    if (
        tokenizer.sha256 != predictor.tokenizer_sha256
        or tokenizer.vocab_size != predictor.vocab_size
    ):
        raise ValueError("Checkpoint tokenizer mismatch")
    require_baseline_tokenizer(tokenizer)
    transfer = "gpu_int"
    from .entropy import IntegerEncoder, QUANTIZATION, TOTAL, uniform_cdf

    initial_transfer_bytes = getattr(predictor, "probability_transfer_bytes", 0)
    predictor.synchronize()
    started = time.perf_counter()
    raw_size = Path(input_path).stat().st_size
    header = dict(
        version=3,
        profile="token-graph-gpu-int-v3",
        block_size=2048,
        batch_size=predictor.batch_size,
        vocab_size=tokenizer.vocab_size,
        raw_bytes=raw_size,
        threads=tokenizer.threads,
        tokenizer=base64.b64encode(tokenizer.model_bytes).decode(),
        tokenizer_sha256=tokenizer.sha256,
        checkpoint_sha256=predictor.checkpoint_sha256,
        runtime=predictor.runtime,
    )
    header.update(
        version=3,
        profile="token-graph-gpu-int-v3",
        probability_transfer=transfer,
        quantization=QUANTIZATION,
        frequency_precision=24,
    )
    encoded_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    initial_cdf = uniform_cdf(tokenizer.vocab_size)
    digest = hashlib.sha256()
    tokens = payload_total = body_crc = decoded_size = 0
    (nll, zero_mass) = (0.0, 0)
    phases = dict(
        tokenize=0.0,
        restore_and_hash=0.0,
        state_reset=0.0,
        predict_and_transfer=0.0,
        probability_selection=0.0,
        probability_validation=0.0,
        quantize_and_range=0.0,
        diagnostics=0.0,
        framing_and_io=0.0,
        original_hash=0.0,
    )
    with _coding_mode(predictor), c._new_output(
        output_path
    ) as target, contextlib.ExitStack() as cleanup:
        magic = MAGIC_GPU_INT
        target.write(PREFIX.pack(magic, len(encoded_header)))
        target.write(encoded_header)
        target.write(c.CRC32.pack(zlib.crc32(encoded_header)))
        iterator = iter(token_blocks(tokenizer, input_path, predictor.batch_size))
        while True:
            tick = time.perf_counter()
            try:
                (values, lengths) = next(iterator)
            except StopIteration:
                phases["tokenize"] += time.perf_counter() - tick
                break
            phases["tokenize"] += time.perf_counter() - tick
            tick = time.perf_counter() if profile else 0
            for (row, length) in zip(values, lengths):
                raw = tokenizer.decode(row[:length])
                digest.update(raw)
                decoded_size += len(raw)
            if profile:
                phases["restore_and_hash"] += time.perf_counter() - tick
            count = int(lengths.sum())
            tokens += count
            tick = time.perf_counter() if profile else 0
            predictor.reset(len(lengths))
            if profile:
                phases["state_reset"] += time.perf_counter() - tick
            encoder = IntegerEncoder()
            cleanup.callback(encoder.close)
            first = values[:, 0].astype(np.int32)
            encoder.encode(
                np.stack(
                    (initial_cdf[first], initial_cdf[first + 1] - initial_cdf[first]),
                    axis=-1,
                )
            )
            nll += -np.log(
                (initial_cdf[first + 1] - initial_cdf[first]).astype(np.float64) / TOTAL
            ).sum()
            prefetch = (
                optimized
                and isinstance(predictor, Predictor)
                and (predictor.device.type == "cuda")
                and predictor.use_graph
            )
            pending = predictor.iter_prefetched(values, lengths) if prefetch else None
            if pending is not None:
                cleanup.callback(pending.close)
            for position in range(1, int(lengths.max())):
                active = lengths > position
                previous = np.where(active, values[:, position - 1], 0)
                tick = time.perf_counter() if profile else 0
                if prefetch:
                    (probabilities, _) = next(pending)
                else:
                    (probabilities, _) = predictor.step(
                        previous, active=active, targets=values[:, position]
                    )
                if profile:
                    phases["predict_and_transfer"] += time.perf_counter() - tick
                tick = time.perf_counter() if profile else 0
                p = (
                    probabilities[: int(active.sum())]
                    if optimized
                    else np.ascontiguousarray(probabilities[active])
                )
                symbols = values[active, position].astype(np.int32)
                if profile:
                    phases["probability_selection"] += time.perf_counter() - tick
                    tick = time.perf_counter()
                if profile:
                    phases["probability_validation"] += time.perf_counter() - tick
                tick = time.perf_counter() if profile else 0
                encoder.encode(p)
                if profile:
                    phases["quantize_and_range"] += time.perf_counter() - tick
                tick = time.perf_counter() if profile else 0
                chosen = p[:, 1].astype(np.float64) / TOTAL
                zero_mass += int(np.count_nonzero(chosen == 0))
                nll -= np.log(chosen[chosen > 0].astype(np.float64)).sum()
                if profile:
                    phases["diagnostics"] += time.perf_counter() - tick
            if pending is not None:
                pending.close()
            tick = time.perf_counter() if profile else 0
            payload = encoder.get_compressed().astype("<u4", copy=False).tobytes()
            encoder.close()
            frame = (
                struct.pack("<I", count)
                + FRAME.pack(len(payload), zlib.crc32(payload))
                + payload
            )
            body_crc = zlib.crc32(frame, body_crc)
            target.write(frame)
            payload_total += len(payload)
            if profile:
                phases["framing_and_io"] += time.perf_counter() - tick
        tick = time.perf_counter() if profile else 0
        if decoded_size != raw_size or digest.hexdigest() != c.sha256_file(input_path):
            raise ValueError(
                "Input changed or tokenizer did not preserve original bytes"
            )
        if profile:
            phases["original_hash"] += time.perf_counter() - tick
        target.write(struct.pack("<I", 0))
        target.write(TRAILER.pack(tokens, raw_size, body_crc, digest.digest()))
    predictor.synchronize()
    result = _metrics(
        "encode",
        raw_size,
        tokens,
        Path(output_path).stat().st_size,
        payload_total,
        time.perf_counter() - started,
        phases,
    )
    if profile:
        phases["other"] = max(0.0, result["encode_seconds"] - sum(phases.values()))
    result.update(
        original_sha256=digest.hexdigest(),
        compressed_sha256=c.sha256_file(output_path),
        theoretical_nll_bpb=float(nll / (raw_size * math.log(2)))
        if raw_size and (not zero_mass)
        else None,
        floating_zero_probability_symbols=zero_mass,
        checkpoint_sha256=predictor.checkpoint_sha256,
        tokenizer_sha256=tokenizer.sha256,
        runtime=predictor.runtime,
        batch_size=predictor.batch_size,
    )
    result.update(_transfer_metrics(predictor, initial_transfer_bytes))
    result.update(
        quantization=QUANTIZATION,
        nll_probability_basis="integer_frequency/2**24",
        floating_zero_probability_symbols=None,
        integer_zero_frequency_symbols=zero_mass,
    )
    return result


def decode(input_path, output_path, predictor, *, profile=False, optimized=True):
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Input and output must differ")
    initial_transfer_bytes = getattr(predictor, "probability_transfer_bytes", 0)
    predictor.synchronize()
    started = time.perf_counter()
    digest = hashlib.sha256()
    tokens = raw_bytes = payload_total = body_crc = 0
    short_group = False
    phases = dict(
        detokenize=0.0,
        state_reset=0.0,
        predict_and_transfer=0.0,
        probability_selection=0.0,
        quantize_and_range=0.0,
        framing_and_io=0.0,
    )
    with _coding_mode(predictor), open(input_path, "rb") as source:
        (header, tokenizer) = read_header(source, predictor)
        from .entropy import IntegerDecoder, uniform_cdf

        initial_cdf = uniform_cdf(tokenizer.vocab_size)
        with c._new_output(output_path) as target, contextlib.ExitStack() as cleanup:
            while True:
                tick = time.perf_counter() if profile else 0
                raw_count = c._read_exact(source, 4)
                count = struct.unpack("<I", raw_count)[0]
                if count == 0:
                    break
                if (
                    short_group
                    or count > predictor.batch_size * 2048
                    or tokens + count > header["raw_bytes"]
                ):
                    raise ValueError("Invalid token group length")
                short_group = count < predictor.batch_size * 2048
                frame_header = c._read_exact(source, FRAME.size)
                (size, crc) = FRAME.unpack(frame_header)
                if not size or size % 4 or size > count * 8 + 64:
                    raise ValueError("Invalid Range payload size")
                payload = c._read_exact(source, size)
                if zlib.crc32(payload) != crc:
                    raise ValueError("Range payload CRC mismatch")
                body_crc = zlib.crc32(raw_count + frame_header + payload, body_crc)
                decoder = IntegerDecoder(np.frombuffer(payload, dtype="<u4"))
                cleanup.callback(decoder.close)
                b = (count + 2047) // 2048
                lengths = np.minimum(2048, count - np.arange(b) * 2048)
                values = np.zeros((b, 2048), dtype=np.uint16)
                values[:, 0] = decoder.decode(
                    np.tile(initial_cdf, (b, 1)), trusted=True
                )
                if profile:
                    phases["framing_and_io"] += time.perf_counter() - tick
                    tick = time.perf_counter()
                predictor.reset(b)
                if profile:
                    phases["state_reset"] += time.perf_counter() - tick
                for position in range(1, int(lengths.max())):
                    active = lengths > position
                    tick = time.perf_counter() if profile else 0
                    (p, _) = predictor.step(
                        np.where(active, values[:, position - 1], 0), active=active
                    )
                    if profile:
                        phases["predict_and_transfer"] += time.perf_counter() - tick
                    tick = time.perf_counter() if profile else 0
                    selected = (
                        p[: int(active.sum())]
                        if optimized
                        else np.ascontiguousarray(p[active])
                    )
                    if profile:
                        phases["probability_selection"] += time.perf_counter() - tick
                        tick = time.perf_counter()
                    values[active, position] = decoder.decode(selected, trusted=True)
                    if profile:
                        phases["quantize_and_range"] += time.perf_counter() - tick
                decoder.close()
                tick = time.perf_counter()
                for (row, length) in zip(values, lengths):
                    data = tokenizer.decode(row[:length])
                    raw_bytes += len(data)
                    if raw_bytes > header["raw_bytes"]:
                        raise ValueError("Decoded bytes exceed advertised size")
                    target.write(data)
                    digest.update(data)
                phases["detokenize"] += time.perf_counter() - tick
                tokens += count
                payload_total += size
            (
                expected_tokens,
                expected_bytes,
                expected_crc,
                expected_sha,
            ) = TRAILER.unpack(c._read_exact(source, TRAILER.size))
            if (
                (tokens, raw_bytes, body_crc, digest.digest())
                != (expected_tokens, expected_bytes, expected_crc, expected_sha)
                or raw_bytes != header["raw_bytes"]
                or source.read(1)
            ):
                raise ValueError(
                    "Token container checksum, length or trailing data mismatch"
                )
    predictor.synchronize()
    result = _metrics(
        "decode",
        raw_bytes,
        tokens,
        Path(input_path).stat().st_size,
        payload_total,
        time.perf_counter() - started,
        phases,
    )
    if profile:
        phases["other"] = max(0.0, result["decode_seconds"] - sum(phases.values()))
    result.update(
        decoded_sha256=digest.hexdigest(),
        checksum_verified=True,
        checkpoint_sha256=predictor.checkpoint_sha256,
        runtime=predictor.runtime,
    )
    result.update(_transfer_metrics(predictor, initial_transfer_bytes))
    return result


def read_header(handle, predictor=None):
    magic, length = PREFIX.unpack(c._read_exact(handle, PREFIX.size))
    if magic != MAGIC_GPU_INT or not 1 <= length <= MAX_HEADER:
        raise ValueError("仅支持当前 GPU 整数量化码流；旧格式请检出历史标签")
    raw = c._read_exact(handle, length)
    if zlib.crc32(raw) != c.CRC32.unpack(c._read_exact(handle, 4))[0]:
        raise ValueError("Header CRC mismatch")
    header = json.loads(raw)
    from .entropy import QUANTIZATION

    if (
        (
            header["version"],
            header["profile"],
            header.get("probability_transfer"),
            header["block_size"],
        )
        != (3, "token-graph-gpu-int-v3", "gpu_int", 2048)
        or header.get("quantization") != QUANTIZATION
        or header.get("frequency_precision") != 24
    ):
        raise ValueError("Unsupported integer coding profile")
    c.validate_batch_size(header["batch_size"])
    if not isinstance(header["raw_bytes"], int) or header["raw_bytes"] < 0:
        raise ValueError("Invalid original size")
    tokenizer = LosslessTokenizer(
        base64.b64decode(header["tokenizer"], validate=True), threads=header["threads"]
    )
    require_baseline_tokenizer(tokenizer)
    if (
        tokenizer.sha256 != header["tokenizer_sha256"]
        or tokenizer.vocab_size != 4096
        or header["vocab_size"] != 4096
    ):
        raise ValueError("Tokenizer fingerprint or vocabulary mismatch")
    if predictor is not None:
        if (
            header["checkpoint_sha256"] != predictor.checkpoint_sha256
            or header["runtime"] != predictor.runtime
            or header["batch_size"] != predictor.batch_size
            or header["vocab_size"] != predictor.vocab_size
            or header["tokenizer_sha256"] != predictor.tokenizer_sha256
        ):
            raise ValueError(
                "Incompatible checkpoint, tokenizer, runtime or batch size"
            )
    return header, tokenizer


def read_settings(input_path):
    with open(input_path, "rb") as source:
        header, _ = read_header(source)
    return dict(batch_size=header["batch_size"])


def load_decoder(checkpoint, input_path, device="cuda:0", *, use_graph=True):
    return load_predictor(
        checkpoint, device, use_graph=use_graph, **read_settings(input_path)
    )

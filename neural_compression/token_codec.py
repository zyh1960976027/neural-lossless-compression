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

import constriction
import numpy as np
import torch

from . import codec as c
from .codec_graph import BatchedStepV5Predictor
from .tokenization import LosslessTokenizer, token_blocks

MAGIC = b"M3SP\x01\r\n\x1a"
MAGIC_FP16 = b"M3SP\x02\r\n\x1a"
MAGIC_GPU_INT = b"M3SP\x03\r\n\x1a"
PREFIX = struct.Struct("<8sI")
FRAME = struct.Struct("<II")
TRAILER = struct.Struct("<QQI32s")
MAX_HEADER = 8 * 1024 * 1024


def load_token_predictor(checkpoint, device="cuda:0", batch_size=1024, use_graph=True,
                         device_checks=True, probability_transfer="fp32"):
    if probability_transfer not in ("fp32", "fp16", "gpu_int"):
        raise ValueError("probability_transfer must be fp32, fp16 or gpu_int")
    if probability_transfer == "gpu_int":
        from .integer_range import _library
        _library()
    from importlib.metadata import version
    from .model import build_mamba3_token_model
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, config = build_mamba3_token_model(state["vocab_size"])
    if state["model_config"] != config:
        raise ValueError("Unsupported token checkpoint configuration")
    model.load_state_dict(state["model"], strict=True)
    predictor = BatchedStepV5Predictor(model, device, c.sha256_file(checkpoint),
                                      batch_size=batch_size, use_graph=use_graph,
                                      check_probabilities=device_checks,
                                      probability_transfer=probability_transfer)
    predictor.tokenizer_sha256 = state["tokenizer_sha256"]
    predictor.runtime.update(sentencepiece=version("sentencepiece"), protobuf=version("protobuf"),
                             token_adapter_version=1)
    return predictor


def read_header(handle, predictor=None):
    magic, length = PREFIX.unpack(c._read_exact(handle, PREFIX.size))
    if magic not in (MAGIC, MAGIC_FP16, MAGIC_GPU_INT) or not 1 <= length <= MAX_HEADER:
        raise ValueError("Invalid token container header")
    raw = c._read_exact(handle, length)
    if zlib.crc32(raw) != c.CRC32.unpack(c._read_exact(handle, 4))[0]:
        raise ValueError("Header CRC mismatch")
    header = json.loads(raw)
    transfer = header.get("probability_transfer", "fp32")
    expected = {MAGIC: (1, "token-graph-v1", "fp32"),
                MAGIC_FP16: (2, "token-graph-fp16-v2", "fp16"),
                MAGIC_GPU_INT: (3, "token-graph-gpu-int-v3", "gpu_int")}[magic]
    if (header["version"], header["profile"], transfer) != expected or header["block_size"] != 2048:
        raise ValueError("Unsupported token coding profile")
    if transfer == "gpu_int":
        from .integer_range import QUANTIZATION
        if header.get("quantization") != QUANTIZATION or header.get("frequency_precision") != 24:
            raise ValueError("Unsupported integer quantization profile")
    c.validate_batch_size(header["batch_size"])
    tokenizer = LosslessTokenizer(base64.b64decode(header["tokenizer"], validate=True),
                                  threads=header["threads"])
    if tokenizer.sha256 != header["tokenizer_sha256"] or tokenizer.vocab_size != header["vocab_size"]:
        raise ValueError("Tokenizer fingerprint or vocabulary mismatch")
    if predictor is not None:
        if transfer != getattr(predictor, "probability_transfer", "fp32"):
            raise ValueError("Probability transfer mode does not match the container")
        if (header["checkpoint_sha256"] != predictor.checkpoint_sha256
                or header["runtime"] != predictor.runtime
                or header["batch_size"] != predictor.batch_size
                or header["vocab_size"] != predictor.vocab_size
                or header["tokenizer_sha256"] != predictor.tokenizer_sha256):
            raise ValueError("Incompatible checkpoint, tokenizer, runtime or batch size")
    return header, tokenizer


def read_token_settings(input_path):
    """Read the transfer mode and grouping before allocating a decoder predictor."""
    with open(input_path, "rb") as source:
        header, _ = read_header(source)
    return dict(batch_size=header["batch_size"],
                probability_transfer=header.get("probability_transfer", "fp32"))


def load_token_decoder(checkpoint, input_path, device="cuda:0", *, use_graph=True, device_checks=True):
    return load_token_predictor(checkpoint, device, use_graph=use_graph,
                                device_checks=device_checks, **read_token_settings(input_path))


def _transfer_metrics(predictor, initial_bytes):
    return dict(probability_transfer=getattr(predictor, "probability_transfer", "fp32"),
                probability_transfer_bytes=getattr(predictor, "probability_transfer_bytes", 0) - initial_bytes)


def warmup(predictor, token_count):
    blocks = (token_count + 2047) // 2048
    sizes = {min(blocks, predictor.batch_size), blocks % predictor.batch_size}
    started = time.perf_counter()
    with c._coding_mode(predictor):
        for size in sorted(sizes - {0}):
            predictor.reset(size)
            predictor.step(np.zeros(size, dtype=np.int32))
            if (isinstance(predictor, BatchedStepV5Predictor) and predictor.vocab_size > 256
                    and predictor.use_graph):
                predictor.prepare_prefetch()
    predictor.synchronize()
    return time.perf_counter() - started


def _metrics(operation, raw_bytes, tokens, packed_bytes, payload, elapsed, phases):
    return {"operation": operation, "num_bytes": raw_bytes, "num_tokens": tokens,
            "compressed_bytes": packed_bytes, "payload_bytes": payload,
            "actual_bpb": 8 * packed_bytes / raw_bytes if raw_bytes else None,
            "payload_bpb": 8 * payload / raw_bytes if raw_bytes else None,
            f"{operation}_seconds": elapsed,
            f"{operation}_kb_per_second": raw_bytes / 1000 / elapsed if raw_bytes else 0,
            f"{operation}_tokens_per_second": tokens / elapsed if tokens else 0,
            "phases_seconds": phases}


def _validate_probabilities(probabilities):
    # Equivalent to finite/nonnegative/positive row-sum checks for categorical
    # distributions. Reduction buffers have B entries, not B*vocabulary entries.
    smallest = probabilities.min(axis=1)
    largest = probabilities.max(axis=1)
    if (not np.isfinite(smallest).all() or not np.isfinite(largest).all()
            or np.any(smallest < 0) or np.any(largest <= 0)):
        raise ValueError("Invalid predicted probabilities")


def encode(input_path, output_path, predictor, tokenizer, *, profile=False, optimized=True):
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Input and output must differ")
    if tokenizer.sha256 != predictor.tokenizer_sha256 or tokenizer.vocab_size != predictor.vocab_size:
        raise ValueError("Checkpoint tokenizer mismatch")
    transfer = getattr(predictor, "probability_transfer", "fp32")
    if transfer not in ("fp32", "fp16", "gpu_int"):
        raise ValueError("Unsupported probability transfer mode")
    integer = transfer == "gpu_int"
    if integer:
        from .integer_range import IntegerEncoder, QUANTIZATION, TOTAL, uniform_cdf
    initial_transfer_bytes = getattr(predictor, "probability_transfer_bytes", 0)
    predictor.synchronize()
    started = time.perf_counter()
    raw_size = Path(input_path).stat().st_size
    header = dict(version=1, profile="token-graph-v1", block_size=2048,
                  batch_size=predictor.batch_size, vocab_size=tokenizer.vocab_size,
                  raw_bytes=raw_size, threads=tokenizer.threads,
                  tokenizer=base64.b64encode(tokenizer.model_bytes).decode(),
                  tokenizer_sha256=tokenizer.sha256, checkpoint_sha256=predictor.checkpoint_sha256,
                  runtime=predictor.runtime)
    if transfer == "fp16":
        header.update(version=2, profile="token-graph-fp16-v2", probability_transfer=transfer)
    elif integer:
        header.update(version=3, profile="token-graph-gpu-int-v3", probability_transfer=transfer,
                      quantization=QUANTIZATION, frequency_precision=24)
    encoded_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    uniform = c._categorical(np.full(tokenizer.vocab_size, 1 / tokenizer.vocab_size))
    family = constriction.stream.model.Categorical(perfect=False)
    if integer:
        initial_cdf = uniform_cdf(tokenizer.vocab_size)
    digest = hashlib.sha256()
    tokens = payload_total = body_crc = decoded_size = 0
    nll, zero_mass = 0., 0
    phases = dict(tokenize=0., restore_and_hash=0., state_reset=0., predict_and_transfer=0.,
                  probability_selection=0., probability_validation=0., quantize_and_range=0.,
                  diagnostics=0., framing_and_io=0., original_hash=0.)
    with c._coding_mode(predictor), c._new_output(output_path) as target, contextlib.ExitStack() as cleanup:
        magic = {"fp32": MAGIC, "fp16": MAGIC_FP16, "gpu_int": MAGIC_GPU_INT}[transfer]
        target.write(PREFIX.pack(magic, len(encoded_header)))
        target.write(encoded_header)
        target.write(c.CRC32.pack(zlib.crc32(encoded_header)))
        iterator = iter(token_blocks(tokenizer, input_path, predictor.batch_size))
        while True:
            tick = time.perf_counter()
            try:
                values, lengths = next(iterator)
            except StopIteration:
                phases["tokenize"] += time.perf_counter() - tick
                break
            phases["tokenize"] += time.perf_counter() - tick
            tick = time.perf_counter() if profile else 0
            for row, length in zip(values, lengths):
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
            encoder = IntegerEncoder() if integer else constriction.stream.queue.RangeEncoder()
            if integer:
                cleanup.callback(encoder.close)
                first = values[:, 0].astype(np.int32)
                encoder.encode(np.stack((initial_cdf[first], initial_cdf[first+1]-initial_cdf[first]), axis=-1))
            else:
                encoder.encode(values[:, 0].astype(np.int32), uniform)
            nll += (-np.log((initial_cdf[first+1]-initial_cdf[first]).astype(np.float64)/TOTAL).sum()
                    if integer else len(lengths) * math.log(tokenizer.vocab_size))
            prefetch = (optimized and isinstance(predictor, BatchedStepV5Predictor)
                        and predictor.device.type == "cuda" and predictor.use_graph)
            pending = predictor.iter_prefetched(values, lengths) if prefetch else None
            if pending is not None:
                cleanup.callback(pending.close)
            for position in range(1, int(lengths.max())):
                active = lengths > position
                previous = np.where(active, values[:, position-1], 0)
                tick = time.perf_counter() if profile else 0
                if prefetch:
                    probabilities, _ = next(pending)
                elif integer:
                    probabilities, _ = predictor.step(previous, active=active, targets=values[:, position])
                else:
                    probabilities, _ = predictor.step(previous, active=active)
                if profile:
                    phases["predict_and_transfer"] += time.perf_counter() - tick
                tick = time.perf_counter() if profile else 0
                # Independent blocks are full except the final row. Active rows
                # always form a prefix, so this view keeps the pinned buffer.
                p = probabilities[:int(active.sum())] if optimized else np.ascontiguousarray(probabilities[active])
                symbols = values[active, position].astype(np.int32)
                if profile:
                    phases["probability_selection"] += time.perf_counter() - tick
                    tick = time.perf_counter()
                if integer:
                    pass  # GPU validates model probabilities; native encoder validates integer intervals.
                elif optimized and isinstance(predictor, BatchedStepV5Predictor) and predictor.check_probabilities:
                    pass  # The predictor checked the exact buffer before returning it.
                elif optimized:
                    _validate_probabilities(p)
                elif not np.isfinite(p).all() or np.any(p < 0) or np.any(p.sum(axis=1) <= 0):
                    raise ValueError("Invalid predicted probabilities")
                if profile:
                    phases["probability_validation"] += time.perf_counter() - tick
                tick = time.perf_counter() if profile else 0
                if integer:
                    encoder.encode(p)
                else:
                    encoder.encode(symbols, family, p)
                if profile:
                    phases["quantize_and_range"] += time.perf_counter() - tick
                tick = time.perf_counter() if profile else 0
                chosen = p[:, 1].astype(np.float64) / TOTAL if integer else p[np.arange(len(symbols)), symbols]
                zero_mass += int(np.count_nonzero(chosen == 0))
                nll -= np.log(chosen[chosen > 0].astype(np.float64)).sum()
                if profile:
                    phases["diagnostics"] += time.perf_counter() - tick
            if pending is not None:
                pending.close()
            tick = time.perf_counter() if profile else 0
            payload = encoder.get_compressed().astype("<u4", copy=False).tobytes()
            if integer:
                encoder.close()
            # The count explicitly terminates each group; a zero count terminates the body.
            frame = struct.pack("<I", count) + FRAME.pack(len(payload), zlib.crc32(payload)) + payload
            body_crc = zlib.crc32(frame, body_crc)
            target.write(frame)
            payload_total += len(payload)
            if profile:
                phases["framing_and_io"] += time.perf_counter() - tick
        tick = time.perf_counter() if profile else 0
        if decoded_size != raw_size or digest.hexdigest() != c.sha256_file(input_path):
            raise ValueError("Input changed or tokenizer did not preserve original bytes")
        if profile:
            phases["original_hash"] += time.perf_counter() - tick
        target.write(struct.pack("<I", 0))
        target.write(TRAILER.pack(tokens, raw_size, body_crc, digest.digest()))
    predictor.synchronize()
    result = _metrics("encode", raw_size, tokens, Path(output_path).stat().st_size,
                      payload_total, time.perf_counter()-started, phases)
    if profile:
        phases["other"] = max(0., result["encode_seconds"] - sum(phases.values()))
    result.update(original_sha256=digest.hexdigest(), compressed_sha256=c.sha256_file(output_path),
                  theoretical_nll_bpb=float(nll / (raw_size * math.log(2))) if raw_size and not zero_mass else None,
                  floating_zero_probability_symbols=zero_mass,
                  checkpoint_sha256=predictor.checkpoint_sha256, tokenizer_sha256=tokenizer.sha256,
                  runtime=predictor.runtime, batch_size=predictor.batch_size)
    result.update(_transfer_metrics(predictor, initial_transfer_bytes))
    if integer:
        result.update(quantization=QUANTIZATION, nll_probability_basis="integer_frequency/2**24",
                      floating_zero_probability_symbols=None, integer_zero_frequency_symbols=zero_mass)
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
    phases = dict(detokenize=0., state_reset=0., predict_and_transfer=0.,
                  probability_selection=0., quantize_and_range=0., framing_and_io=0.)
    with c._coding_mode(predictor), open(input_path, "rb") as source:
        header, tokenizer = read_header(source, predictor)
        integer = header.get("probability_transfer") == "gpu_int"
        if integer:
            from .integer_range import IntegerDecoder, uniform_cdf
            initial_cdf = uniform_cdf(tokenizer.vocab_size)
        uniform = c._categorical(np.full(tokenizer.vocab_size, 1 / tokenizer.vocab_size))
        family = constriction.stream.model.Categorical(perfect=False)
        with c._new_output(output_path) as target, contextlib.ExitStack() as cleanup:
            while True:
                tick = time.perf_counter() if profile else 0
                raw_count = c._read_exact(source, 4)
                count = struct.unpack("<I", raw_count)[0]
                if count == 0:
                    break
                if short_group or count > predictor.batch_size * 2048 or tokens + count > header["raw_bytes"]:
                    raise ValueError("Invalid token group length")
                short_group = count < predictor.batch_size * 2048
                frame_header = c._read_exact(source, FRAME.size)
                size, crc = FRAME.unpack(frame_header)
                if not size or size % 4 or size > count * 8 + 64:
                    raise ValueError("Invalid Range payload size")
                payload = c._read_exact(source, size)
                if zlib.crc32(payload) != crc:
                    raise ValueError("Range payload CRC mismatch")
                body_crc = zlib.crc32(raw_count + frame_header + payload, body_crc)
                decoder = (IntegerDecoder if integer else constriction.stream.queue.RangeDecoder)(
                    np.frombuffer(payload, dtype="<u4"))
                if integer:
                    cleanup.callback(decoder.close)
                b = (count + 2047) // 2048
                lengths = np.minimum(2048, count - np.arange(b) * 2048)
                values = np.zeros((b, 2048), dtype=np.uint16)
                values[:, 0] = (decoder.decode(np.tile(initial_cdf, (b, 1)), trusted=True) if integer
                                else decoder.decode(uniform, b))
                if profile:
                    phases["framing_and_io"] += time.perf_counter() - tick
                    tick = time.perf_counter()
                predictor.reset(b)
                if profile:
                    phases["state_reset"] += time.perf_counter() - tick
                for position in range(1, int(lengths.max())):
                    active = lengths > position
                    tick = time.perf_counter() if profile else 0
                    p, _ = predictor.step(np.where(active, values[:, position-1], 0), active=active)
                    if profile:
                        phases["predict_and_transfer"] += time.perf_counter() - tick
                    tick = time.perf_counter() if profile else 0
                    selected = p[:int(active.sum())] if optimized else np.ascontiguousarray(p[active])
                    if profile:
                        phases["probability_selection"] += time.perf_counter() - tick
                        tick = time.perf_counter()
                    values[active, position] = (decoder.decode(selected, trusted=True) if integer
                                                else decoder.decode(family, selected))
                    if profile:
                        phases["quantize_and_range"] += time.perf_counter() - tick
                if integer:
                    decoder.close()
                tick = time.perf_counter()
                for row, length in zip(values, lengths):
                    data = tokenizer.decode(row[:length])
                    raw_bytes += len(data)
                    if raw_bytes > header["raw_bytes"]:
                        raise ValueError("Decoded bytes exceed advertised size")
                    target.write(data)
                    digest.update(data)
                phases["detokenize"] += time.perf_counter() - tick
                tokens += count
                payload_total += size
            expected_tokens, expected_bytes, expected_crc, expected_sha = TRAILER.unpack(c._read_exact(source, TRAILER.size))
            if ((tokens, raw_bytes, body_crc, digest.digest()) !=
                    (expected_tokens, expected_bytes, expected_crc, expected_sha)
                    or raw_bytes != header["raw_bytes"] or source.read(1)):
                raise ValueError("Token container checksum, length or trailing data mismatch")
    predictor.synchronize()
    result = _metrics("decode", raw_bytes, tokens, Path(input_path).stat().st_size,
                      payload_total, time.perf_counter()-started, phases)
    if profile:
        phases["other"] = max(0., result["decode_seconds"] - sum(phases.values()))
    result.update(decoded_sha256=digest.hexdigest(), checksum_verified=True,
                  checkpoint_sha256=predictor.checkpoint_sha256, runtime=predictor.runtime)
    result.update(_transfer_metrics(predictor, initial_transfer_bytes))
    return result

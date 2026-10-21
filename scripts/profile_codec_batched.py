"""Profile the production batched v3 codec without changing its public API.

The parent process extracts one byte range, runs uninstrumented baselines in
independent child processes, runs a bit-identical instrumented implementation,
and writes a Chinese experiment report.  Hidden child-process arguments are an
implementation detail; the documented interface is the six options in
``build_parser`` before ``--stage``.
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import types


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def extract_slice(source, target, offset, count):
    source, target = Path(source), Path(target)
    if offset < 0 or count <= 0:
        raise ValueError("offset must be nonnegative and count must be positive")
    handle = target.open("xb")
    try:
        with source.open("rb") as src, handle:
            src.seek(offset)
            remaining = count
            while remaining:
                chunk = src.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("source does not contain the requested byte range")
                handle.write(chunk)
                remaining -= len(chunk)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return {"offset_bytes": offset, "num_bytes": count, "sha256": hash_file(target)}


def summarize_values(values):
    values = [float(value) for value in values]
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "coefficient_of_variation": statistics.pstdev(values) / mean if mean else 0.0,
    }


def timing_rows(operation, total_seconds, components, num_bytes, steps):
    rows = []
    for name, seconds in components.items():
        rows.append({
            "operation": operation,
            "component": name,
            "seconds": seconds,
            "percent": 100 * seconds / total_seconds if total_seconds else 0.0,
            "microseconds_per_step": seconds * 1e6 / steps if steps else 0.0,
            "nanoseconds_per_byte": seconds * 1e9 / num_bytes if num_bytes else 0.0,
        })
    return rows


class ProfileStats:
    """Collect additive host wall time plus separately sampled CUDA event time."""

    def __init__(self, operation, gpu_sample_every=1, range_sample_every=64):
        self.operation = operation
        self.primary = defaultdict(float)
        self.primary_calls = defaultdict(int)
        self.predictor_detail = defaultdict(float)
        self.predictor_detail_calls = defaultdict(int)
        self.sampled_range = defaultdict(float)
        self.sampled_range_calls = defaultdict(int)
        self.gpu = defaultdict(lambda: defaultdict(float))
        self.gpu_samples = defaultdict(lambda: defaultdict(int))
        self.steps_by_batch = defaultdict(int)
        self.gpu_sample_every = gpu_sample_every
        self.range_sample_every = range_sample_every
        self.step_index = 0
        self.position_index = 0
        self.predicted_symbols = 0
        self.sampled_symbols = 0
        self.range_calibration_samples = []
        self.total_seconds = 0.0

    @contextmanager
    def timed(self, name, *, detail=False):
        started = time.perf_counter()
        try:
            yield
        finally:
            target = self.predictor_detail if detail else self.primary
            counts = self.predictor_detail_calls if detail else self.primary_calls
            target[name] += time.perf_counter() - started
            counts[name] += 1

    def start_step(self, batch_size):
        self.step_index += 1
        self.steps_by_batch[batch_size] += 1
        return (self.step_index - 1) % self.gpu_sample_every == 0

    def sample_range_position(self, symbols):
        self.position_index += 1
        self.predicted_symbols += symbols
        sampled = (self.position_index - 1) % self.range_sample_every == 0
        if sampled:
            self.sampled_symbols += symbols
        return sampled

    def add_range_sample(self, name, seconds):
        self.sampled_range[name] += seconds
        self.sampled_range_calls[name] += 1

    def add_gpu(self, name, batch_size, milliseconds):
        self.gpu[name][batch_size] += milliseconds / 1000
        self.gpu_samples[name][batch_size] += 1

    def capture_range_calibration(self, probabilities, symbols):
        if len(self.range_calibration_samples) < 16 and self.position_index % 512 == 1:
            self.range_calibration_samples.append((probabilities.copy(), symbols.copy()))
            self.sampled_symbols += len(symbols)

    def calibrate_batched_range(self):
        """Estimate fused quantization cost against an equal-sized fixed-model coder."""
        import constriction
        import numpy as np

        if not self.range_calibration_samples:
            return {}
        family = constriction.stream.model.Categorical(perfect=False)
        uniform = constriction.stream.model.Categorical(
            np.full(256, 1 / 256, dtype=np.float32), perfect=False)
        repetitions = 5
        dynamic_total = uniform_total = 0.0
        for _ in range(repetitions):
            dynamic_encoder = constriction.stream.queue.RangeEncoder()
            uniform_encoder = constriction.stream.queue.RangeEncoder()
            point = time.perf_counter()
            for probabilities, symbols in self.range_calibration_samples:
                dynamic_encoder.encode(symbols, family, probabilities)
            dynamic_total += time.perf_counter() - point
            point = time.perf_counter()
            for _, symbols in self.range_calibration_samples:
                uniform_encoder.encode(symbols, uniform)
            uniform_total += time.perf_counter() - point
            if self.operation == "decode":
                dynamic_decoder = constriction.stream.queue.RangeDecoder(
                    dynamic_encoder.get_compressed())
                uniform_decoder = constriction.stream.queue.RangeDecoder(
                    uniform_encoder.get_compressed())
                point = time.perf_counter()
                for probabilities, _ in self.range_calibration_samples:
                    dynamic_decoder.decode(family, probabilities)
                dynamic_total += time.perf_counter() - point
                point = time.perf_counter()
                for _, symbols in self.range_calibration_samples:
                    uniform_decoder.decode(uniform, len(symbols))
                uniform_total += time.perf_counter() - point
        if self.operation == "decode":
            repetitions *= 2
        dynamic_seconds = dynamic_total / repetitions
        uniform_seconds = uniform_total / repetitions
        range_fraction = min(1.0, uniform_seconds / dynamic_seconds) if dynamic_seconds else 0.0
        return {
            "sample_positions": len(self.range_calibration_samples),
            "sample_symbols": sum(len(symbols) for _, symbols in self.range_calibration_samples),
            "repetitions": 5,
            "dynamic_seconds_per_pass": dynamic_seconds,
            "uniform_seconds_per_pass": uniform_seconds,
            "estimated_range_fraction": range_fraction,
            "estimated_quantization_fraction": 1.0 - range_fraction,
            "method": "batched dynamic categorical coder versus equal-sized fixed uniform coder",
        }

    def finish(self, total_seconds):
        self.total_seconds = total_seconds
        attributed = sum(self.primary.values())
        residual = total_seconds - attributed
        if residual < -max(1e-6, total_seconds * 1e-6):
            raise AssertionError("profile components overlap")
        self.primary["unattributed"] += max(0.0, residual)
        return self.as_dict()

    def estimated_gpu_seconds(self):
        result = {}
        for name, by_batch in self.gpu.items():
            total = 0.0
            for batch_size, observed in by_batch.items():
                samples = self.gpu_samples[name][batch_size]
                total += observed / samples * self.steps_by_batch[batch_size]
            result[name] = total
        return result

    def estimated_range_seconds(self):
        if not self.sampled_symbols:
            return {}
        scale = self.predicted_symbols / self.sampled_symbols
        return {name: seconds * scale for name, seconds in self.sampled_range.items()}

    def as_dict(self):
        return {
            "total_seconds": self.total_seconds,
            "components": dict(self.primary),
            "component_calls": dict(self.primary_calls),
            "predictor_wall_detail": dict(self.predictor_detail),
            "predictor_wall_calls": dict(self.predictor_detail_calls),
            "cuda_event_seconds_estimated": self.estimated_gpu_seconds(),
            "cuda_event_sample_every": self.gpu_sample_every,
            "cuda_event_samples_by_batch": {
                name: dict(values) for name, values in self.gpu_samples.items()
            },
            "steps_by_batch": dict(self.steps_by_batch),
            "range_sample_every": self.range_sample_every,
            "range_sampled_symbols": self.sampled_symbols,
            "range_total_symbols": self.predicted_symbols,
            "range_subcomponents_estimated": self.estimated_range_seconds(),
            "range_sample_calls": dict(self.sampled_range_calls),
        }


def install_measured_step(predictor, stats):
    """Replace one predictor instance's step with a value-identical measured copy."""
    import numpy as np
    import torch
    from neural_compression import codec as c
    from neural_compression.metrics import autocast_context

    predictor._profile_stats = stats
    predictor._profile_events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

    @torch.inference_mode()
    def measured_step(self, tokens, *, active=None):
        profile = self._profile_stats
        batch_size = self.active_batch_size
        sampled_gpu = profile.start_step(batch_size)
        events = self._profile_events
        with profile.timed("input_validation_and_host_buffer", detail=True):
            tokens = np.asarray(tokens)
            if (not batch_size or tokens.shape != (batch_size,)
                    or not np.issubdtype(tokens.dtype, np.integer)
                    or np.any(tokens < 0) or np.any(tokens > 255)):
                raise ValueError("reset(B) then supply B byte values in 0..255.")
            if self.steps_in_block >= c.BLOCK_SIZE - 1:
                raise ValueError("An independent block allows 2047 predictions; reset the batch.")
            if active is not None:
                active = np.asarray(active, dtype=bool)
                if active.shape != tokens.shape:
                    raise ValueError("Active mask must match the batch shape.")
            self._host_token.numpy()[:] = tokens
        if sampled_gpu:
            events[0].record()
        with profile.timed("h2d_enqueue", detail=True):
            self._token.copy_(self._host_token, non_blocking=True)
        if sampled_gpu:
            events[1].record()
        with profile.timed("model_launch", detail=True):
            with autocast_context(self.device):
                x = self.model.embedding(self._token)
                for block, states, step in zip(self.model.blocks, self.states, self._layer_steps):
                    mixed, *_ = step(block.norm1(x), *states)
                    x = x + mixed
                    x = x + block.mlp(block.norm2(x))
                logits = self.model.lm_head(self.model.final_norm(x)).float()
        if sampled_gpu:
            events[2].record()
        if logits.shape != (batch_size, 256):
            raise ValueError("The batched model must output [B, 256] logits.")
        with profile.timed("d2h_and_wait", detail=True):
            last = logits.detach().to(device="cpu", dtype=torch.float64)
        if sampled_gpu:
            events[3].record()
            events[3].synchronize()
            profile.add_gpu("h2d", batch_size, events[0].elapsed_time(events[1]))
            profile.add_gpu("model", batch_size, events[1].elapsed_time(events[2]))
            profile.add_gpu("d2h", batch_size, events[2].elapsed_time(events[3]))
        with profile.timed("cpu_probability", detail=True):
            checked = last if active is None else last[torch.from_numpy(active)]
            if not torch.isfinite(checked).all():
                raise FloatingPointError("The model produced non-finite logits.")
            log_probs = last.log_softmax(dim=-1).numpy()
            probabilities = np.exp(log_probs)
        self.steps_in_block += 1
        return probabilities, log_probs

    predictor.step = types.MethodType(measured_step, predictor)
    return predictor


def install_measured_v4_step(predictor, stats):
    """Measure the optimized GPU-softmax predictor without changing its values."""
    import numpy as np
    import torch
    from neural_compression import codec as c
    from neural_compression.metrics import autocast_context

    predictor._profile_stats = stats
    predictor._profile_events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]

    @torch.inference_mode()
    def measured_step(self, tokens, *, active=None):
        profile = self._profile_stats
        batch_size = self.active_batch_size
        sampled_gpu = profile.start_step(batch_size)
        events = self._profile_events
        with profile.timed("input_validation_and_host_buffer", detail=True):
            tokens = np.asarray(tokens)
            if (not batch_size or tokens.shape != (batch_size,)
                    or not np.issubdtype(tokens.dtype, np.integer)
                    or np.any(tokens < 0) or np.any(tokens > 255)):
                raise ValueError("reset(B) then supply B byte values in 0..255.")
            if self.steps_in_block >= c.BLOCK_SIZE - 1:
                raise ValueError("An independent block allows 2047 predictions; reset the batch.")
            if active is not None:
                active = np.asarray(active, dtype=bool)
                if active.shape != tokens.shape:
                    raise ValueError("Active mask must match the batch shape.")
            self._host_token.numpy()[:] = tokens
        if sampled_gpu:
            events[0].record()
        with profile.timed("h2d_enqueue", detail=True):
            self._token.copy_(self._host_token, non_blocking=True)
        if sampled_gpu:
            events[1].record()
        with profile.timed("model_launch", detail=True):
            with autocast_context(self.device):
                x = self.model.embedding(self._token)
                for block, states, step in zip(self.model.blocks, self.states, self._layer_steps):
                    mixed, *_ = step(block.norm1(x), *states)
                    x = x + mixed
                    x = x + block.mlp(block.norm2(x))
                logits = self.model.lm_head(self.model.final_norm(x)).float()
        if sampled_gpu:
            events[2].record()
        with profile.timed("gpu_probability_launch", detail=True):
            probabilities = logits.softmax(dim=-1)
        if sampled_gpu:
            events[3].record()
        with profile.timed("d2h_and_wait", detail=True):
            self._host_probabilities.copy_(probabilities, non_blocking=False)
            result = self._host_probabilities.numpy()
        if sampled_gpu:
            events[4].record()
            events[4].synchronize()
            profile.add_gpu("h2d", batch_size, events[0].elapsed_time(events[1]))
            profile.add_gpu("model", batch_size, events[1].elapsed_time(events[2]))
            profile.add_gpu("probability", batch_size, events[2].elapsed_time(events[3]))
            profile.add_gpu("d2h", batch_size, events[3].elapsed_time(events[4]))
        self.steps_in_block += 1
        return result, None

    predictor.step = types.MethodType(measured_step, predictor)
    return predictor


def profile_encode_v4(input_path, output_path, predictor, stats):
    import constriction
    import numpy as np
    from neural_compression import codec as c
    from neural_compression import codec_batched_v4 as batched

    input_path, output_path = Path(input_path), Path(output_path)
    predictor.synchronize()
    started = time.perf_counter()
    digest = hashlib.sha256()
    payload_bytes = steps = 0
    model_nll = 0.0
    family = constriction.stream.model.Categorical(perfect=False)
    uniform = c._uniform()
    with c._coding_mode(predictor), input_path.open("rb") as source, c._new_output(output_path) as target:
        with stats.timed("container_setup"):
            count = os.fstat(source.fileno()).st_size
            batched._write_header(target, count, predictor)
            body = c._BodyWriter(target)
            remaining = count
        while remaining:
            with stats.timed("input_read_and_hash"):
                lengths = batched._group_lengths(remaining, predictor.batch_size)
                data = c._read_exact(source, int(lengths.sum()))
                digest.update(data)
            with stats.timed("block_buffer_prepare"):
                values = np.zeros((len(lengths), int(lengths.max())), dtype=np.uint8)
                offset = 0
                for lane, length in enumerate(lengths):
                    values[lane, :length] = np.frombuffer(
                        data, dtype=np.uint8, count=int(length), offset=offset)
                    offset += int(length)
            with stats.timed("range_stream_seed"):
                encoder = constriction.stream.queue.RangeEncoder()
                encoder.encode(values[:, 0].astype(np.int32), uniform)
            with stats.timed("state_reset"):
                predictor.reset(len(lengths))
            for position in range(1, values.shape[1]):
                with stats.timed("token_and_mask_prepare"):
                    active = lengths > position
                    tokens = np.where(active, values[:, position - 1], 0)
                with stats.timed("predictor_step"):
                    probabilities, _ = predictor.step(tokens, active=active)
                with stats.timed("probability_table_and_symbols"):
                    symbols = values[active, position].astype(np.int32)
                    active_probabilities = batched._active_rows(probabilities, active)
                stats.position_index += 1
                stats.predicted_symbols += len(symbols)
                stats.capture_range_calibration(active_probabilities, symbols)
                with stats.timed("probability_quantization_and_range_encode"):
                    encoder.encode(symbols, family, active_probabilities)
                with stats.timed("nll_metric"):
                    selected = active_probabilities[np.arange(len(symbols)), symbols]
                    if np.any(selected <= 0):
                        raise FloatingPointError("A coded symbol has zero floating-point probability.")
                    model_nll -= float(np.log(selected.astype(np.float64)).sum())
                steps += 1
            with stats.timed("range_finalize_crc_and_write"):
                payload_bytes += c._write_frame(body, encoder)
            remaining -= len(data)
        with stats.timed("trailer_and_integrity"):
            if source.read(1):
                raise ValueError("Input file size changed during encoding.")
            target.write(c.COMPACT_TRAILER.pack(body.crc, digest.digest()))
    with stats.timed("final_gpu_sync"):
        predictor.synchronize()
    total_seconds = time.perf_counter() - started
    breakdown = stats.finish(total_seconds)
    calibration = stats.calibrate_batched_range()
    breakdown["batched_range_calibration"] = calibration
    combined = breakdown["components"]["probability_quantization_and_range_encode"]
    breakdown["range_subcomponents_estimated"] = {
        "probability_quantization": combined * calibration.get("estimated_quantization_fraction", 0),
        "range_encode": combined * calibration.get("estimated_range_fraction", 1),
    }
    result = batched._metrics(count, output_path.stat().st_size, payload_bytes,
                              total_seconds, "encode", predictor, steps)
    uniform_nll = result["uniform_bytes"] * math.log(256)
    nll = model_nll + uniform_nll
    result.update(theoretical_nll_bpb=nll / (count * math.log(2)) if count else 0.0,
                  nll_nats_total=nll, nll_nats_model=model_nll,
                  nll_nats_uniform_seeds=uniform_nll, original_sha256=digest.hexdigest(),
                  checkpoint_sha256=predictor.checkpoint_sha256, runtime=predictor.runtime,
                  profile_breakdown=breakdown)
    return result


def profile_decode_v4(input_path, output_path, predictor, stats):
    import constriction
    import numpy as np
    from neural_compression import codec as c
    from neural_compression import codec_batched_v4 as batched

    input_path, output_path = Path(input_path), Path(output_path)
    predictor.synchronize()
    started = time.perf_counter()
    digest = hashlib.sha256()
    payload_bytes = steps = 0
    family = constriction.stream.model.Categorical(perfect=False)
    uniform = c._uniform()
    with c._coding_mode(predictor), input_path.open("rb") as source:
        with stats.timed("header_body_read_and_crc"):
            metadata = c._read_header(source, predictor)
            count = metadata["num_bytes"]
            body, expected_digest = c._read_compact_body(source)
            remaining = count
        with c._new_output(output_path) as target:
            while remaining:
                lengths = batched._group_lengths(remaining, predictor.batch_size)
                with stats.timed("frame_read_crc_and_seed"):
                    decoder, size = batched._read_group_frame(body, int(lengths.sum()))
                    payload_bytes += size
                    values = np.zeros((len(lengths), int(lengths.max())), dtype=np.uint8)
                    values[:, 0] = decoder.decode(uniform, len(lengths)).astype(np.uint8)
                with stats.timed("state_reset"):
                    predictor.reset(len(lengths))
                for position in range(1, values.shape[1]):
                    with stats.timed("token_and_mask_prepare"):
                        active = lengths > position
                        tokens = np.where(active, values[:, position - 1], 0)
                    with stats.timed("predictor_step"):
                        probabilities, _ = predictor.step(tokens, active=active)
                    with stats.timed("probability_table_and_symbols"):
                        active_probabilities = batched._active_rows(probabilities, active)
                    with stats.timed("probability_quantization_and_range_decode"):
                        symbols = decoder.decode(family, active_probabilities).astype(np.uint8)
                    values[active, position] = symbols
                    stats.position_index += 1
                    stats.predicted_symbols += len(symbols)
                    stats.capture_range_calibration(active_probabilities, symbols.astype(np.int32))
                    steps += 1
                with stats.timed("output_assemble_write_and_hash"):
                    for lane, length in enumerate(lengths):
                        data = values[lane, :length].tobytes()
                        target.write(data)
                        digest.update(data)
                remaining -= int(lengths.sum())
            with stats.timed("trailer_and_integrity"):
                if body.read(1):
                    raise ValueError("Unexpected trailing data in compressed file.")
                if digest.digest() != expected_digest:
                    raise ValueError("Decoded SHA256 mismatch: corrupt data or different probabilities.")
    with stats.timed("final_gpu_sync"):
        predictor.synchronize()
    total_seconds = time.perf_counter() - started
    breakdown = stats.finish(total_seconds)
    calibration = stats.calibrate_batched_range()
    breakdown["batched_range_calibration"] = calibration
    combined = breakdown["components"]["probability_quantization_and_range_decode"]
    breakdown["range_subcomponents_estimated"] = {
        "probability_quantization": combined * calibration.get("estimated_quantization_fraction", 0),
        "range_decode": combined * calibration.get("estimated_range_fraction", 1),
    }
    result = batched._metrics(count, input_path.stat().st_size, payload_bytes,
                              total_seconds, "decode", predictor, steps)
    result.update(decoded_sha256=digest.hexdigest(), checksum_verified=True,
                  profile_breakdown=breakdown)
    return result


def profile_encode(input_path, output_path, predictor, stats):
    import constriction
    import numpy as np
    from neural_compression import codec as c
    from neural_compression import codec_batched as batched

    input_path, output_path = Path(input_path), Path(output_path)
    predictor.synchronize()
    started = time.perf_counter()
    digest = hashlib.sha256()
    payload_bytes = steps = 0
    model_nll = 0.0
    with c._coding_mode(predictor), input_path.open("rb") as source, c._new_output(output_path) as target:
        with stats.timed("container_setup"):
            count = os.fstat(source.fileno()).st_size
            batched._write_header(target, count, predictor)
            body = c._BodyWriter(target)
            remaining = count
            uniform = c._uniform()
        while remaining:
            with stats.timed("input_read_and_hash"):
                lengths = batched._group_lengths(remaining, predictor.batch_size)
                data = c._read_exact(source, int(lengths.sum()))
                digest.update(data)
            with stats.timed("block_buffer_prepare"):
                values = np.zeros((len(lengths), int(lengths.max())), dtype=np.uint8)
                offset = 0
                for lane, length in enumerate(lengths):
                    values[lane, :length] = np.frombuffer(
                        data, dtype=np.uint8, count=int(length), offset=offset)
                    offset += int(length)
            with stats.timed("range_stream_seed"):
                encoders = []
                for lane, length in enumerate(lengths):
                    encoder = constriction.stream.queue.RangeEncoder()
                    encoder.encode(int(values[lane, 0]), uniform)
                    encoders.append(encoder)
            with stats.timed("state_reset"):
                predictor.reset(len(lengths))
            for position in range(1, values.shape[1]):
                with stats.timed("token_and_mask_prepare"):
                    active = lengths > position
                    tokens = np.where(active, values[:, position - 1], 0)
                with stats.timed("predictor_step"):
                    probabilities, log_probs = predictor.step(tokens, active=active)
                steps += 1
                lanes = np.flatnonzero(active)
                sampled = stats.sample_range_position(len(lanes))
                with stats.timed("probability_quantization_and_range_encode"):
                    for lane in lanes:
                        symbol = int(values[lane, position])
                        if sampled:
                            point = time.perf_counter()
                            model = c._categorical(probabilities[lane])
                            stats.add_range_sample("probability_quantization", time.perf_counter() - point)
                            point = time.perf_counter()
                            encoders[lane].encode(symbol, model)
                            stats.add_range_sample("range_encode", time.perf_counter() - point)
                        else:
                            encoders[lane].encode(symbol, c._categorical(probabilities[lane]))
                        model_nll -= float(log_probs[lane, symbol])
            with stats.timed("range_finalize_crc_and_write"):
                for encoder in encoders:
                    payload_bytes += c._write_frame(body, encoder)
            remaining -= len(data)
        with stats.timed("trailer_and_integrity"):
            if source.read(1):
                raise ValueError("Input file size changed during encoding.")
            if not math.isfinite(model_nll):
                raise FloatingPointError("Non-finite theoretical NLL.")
            target.write(c.COMPACT_TRAILER.pack(body.crc, digest.digest()))
    with stats.timed("final_gpu_sync"):
        predictor.synchronize()
    total_seconds = time.perf_counter() - started
    breakdown = stats.finish(total_seconds)
    result = batched._metrics(count, output_path.stat().st_size, payload_bytes,
                              total_seconds, "encode", predictor, steps)
    uniform_nll = result["uniform_bytes"] * math.log(256)
    nll = model_nll + uniform_nll
    result.update(theoretical_nll_bpb=nll / (count * math.log(2)) if count else 0.0,
                  nll_nats_total=nll, nll_nats_model=model_nll,
                  nll_nats_uniform_seeds=uniform_nll, original_sha256=digest.hexdigest(),
                  checkpoint_sha256=predictor.checkpoint_sha256, runtime=predictor.runtime,
                  profile_breakdown=breakdown)
    return result


def profile_decode(input_path, output_path, predictor, stats):
    import numpy as np
    from neural_compression import codec as c
    from neural_compression import codec_batched as batched

    input_path, output_path = Path(input_path), Path(output_path)
    predictor.synchronize()
    started = time.perf_counter()
    digest = hashlib.sha256()
    payload_bytes = steps = 0
    with c._coding_mode(predictor), input_path.open("rb") as source:
        with stats.timed("header_body_read_and_crc"):
            metadata = c._read_header(source, predictor)
            count = metadata["num_bytes"]
            body, expected_digest = c._read_compact_body(source)
            remaining = count
            uniform = c._uniform()
        with c._new_output(output_path) as target:
            while remaining:
                lengths = batched._group_lengths(remaining, predictor.batch_size)
                with stats.timed("block_buffer_prepare"):
                    values = np.zeros((len(lengths), int(lengths.max())), dtype=np.uint8)
                with stats.timed("frame_read_crc_and_seed"):
                    decoders = []
                    for lane in range(len(lengths)):
                        decoder, size = c._read_frame(body)
                        payload_bytes += size
                        values[lane, 0] = int(decoder.decode(uniform))
                        decoders.append(decoder)
                with stats.timed("state_reset"):
                    predictor.reset(len(lengths))
                for position in range(1, values.shape[1]):
                    with stats.timed("token_and_mask_prepare"):
                        active = lengths > position
                        tokens = np.where(active, values[:, position - 1], 0)
                    with stats.timed("predictor_step"):
                        probabilities, _ = predictor.step(tokens, active=active)
                    steps += 1
                    lanes = np.flatnonzero(active)
                    sampled = stats.sample_range_position(len(lanes))
                    with stats.timed("probability_quantization_and_range_decode"):
                        for lane in lanes:
                            if sampled:
                                point = time.perf_counter()
                                model = c._categorical(probabilities[lane])
                                stats.add_range_sample("probability_quantization", time.perf_counter() - point)
                                point = time.perf_counter()
                                value = decoders[lane].decode(model)
                                stats.add_range_sample("range_decode", time.perf_counter() - point)
                            else:
                                value = decoders[lane].decode(c._categorical(probabilities[lane]))
                            values[lane, position] = int(value)
                with stats.timed("output_assemble_write_and_hash"):
                    for lane, length in enumerate(lengths):
                        data = values[lane, :length].tobytes()
                        target.write(data)
                        digest.update(data)
                remaining -= int(lengths.sum())
            with stats.timed("trailer_and_integrity"):
                if body.read(1):
                    raise ValueError("Unexpected trailing data in compressed file.")
                if digest.digest() != expected_digest:
                    raise ValueError("Decoded SHA256 mismatch: corrupt data or different probabilities.")
    with stats.timed("final_gpu_sync"):
        predictor.synchronize()
    total_seconds = time.perf_counter() - started
    breakdown = stats.finish(total_seconds)
    result = batched._metrics(count, input_path.stat().st_size, payload_bytes,
                              total_seconds, "decode", predictor, steps)
    result.update(decoded_sha256=digest.hexdigest(), checksum_verified=True,
                  profile_breakdown=breakdown)
    return result


def groups_for(num_bytes, batch_size, block_size=2048):
    blocks = (num_bytes + block_size - 1) // block_size
    return [min(batch_size, blocks - offset) for offset in range(0, blocks, batch_size)]


def load_and_warm(args):
    import numpy as np
    import torch
    from neural_compression import codec as c

    torch.set_num_threads(1)
    settings_started = time.perf_counter()
    inference = "batched-step-v4" if args.format == "v4" else "batched-step"
    settings = ({"inference": inference, "batch_size": args.batch_size}
                if args.operation != "decode" else c.read_codec_settings(args.packed))
    settings_seconds = time.perf_counter() - settings_started
    started = time.perf_counter()
    predictor = c.load_predictor(args.checkpoint, args.device, **settings)
    predictor.synchronize()
    load_seconds = time.perf_counter() - started
    num_bytes = Path(args.raw).stat().st_size
    groups = groups_for(num_bytes, settings["batch_size"], c.BLOCK_SIZE)
    started = time.perf_counter()
    with c._coding_mode(predictor):
        for size in dict.fromkeys(groups):
            predictor.reset(size)
            for _ in range(2):
                predictor.step(np.zeros(size, dtype=np.uint8))
        predictor.reset(groups[0])
        predictor.synchronize()
    warmup_seconds = time.perf_counter() - started
    return predictor, {
        "settings_seconds": settings_seconds,
        "model_load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "warmup_steps_per_shape": 2,
        "batch_sizes": groups,
        "torch_cpu_threads": torch.get_num_threads(),
    }


def memory_snapshot(predictor):
    import torch
    return {
        "allocated_bytes": torch.cuda.memory_allocated(predictor.device),
        "reserved_bytes": torch.cuda.memory_reserved(predictor.device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(predictor.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(predictor.device),
    }


def run_codec_stage(args):
    import torch
    from neural_compression import codec as c

    stage_started = time.perf_counter()
    predictor, setup = load_and_warm(args)
    torch.cuda.reset_peak_memory_stats(predictor.device)
    if args.stage == "baseline":
        if args.operation == "encode":
            result = c.encode_file(args.raw, args.packed, predictor, file_format=args.format)
        else:
            result = c.decode_file(args.packed, args.decoded, predictor)
    else:
        stats = ProfileStats(args.operation, args.gpu_sample_every, args.range_sample_every)
        if args.format == "v4":
            install_measured_v4_step(predictor, stats)
            if args.operation == "encode":
                result = profile_encode_v4(args.raw, args.packed, predictor, stats)
            else:
                result = profile_decode_v4(args.packed, args.decoded, predictor, stats)
        else:
            install_measured_step(predictor, stats)
            if args.operation == "encode":
                result = profile_encode(args.raw, args.packed, predictor, stats)
            else:
                result = profile_decode(args.packed, args.decoded, predictor, stats)
    result.update(setup)
    result.update(
        stage=args.stage,
        operation=args.operation,
        stage_wall_seconds=time.perf_counter() - stage_started,
        process_id=os.getpid(),
        python=platform.python_version(),
        platform=platform.platform(),
        runtime=predictor.runtime,
        checkpoint_sha256=predictor.checkpoint_sha256,
        cuda_memory=memory_snapshot(predictor),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
    write_json(args.metrics, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


def run_trace_stage(args):
    import numpy as np
    import torch
    from neural_compression import codec as c

    args.operation = "encode"
    predictor, setup = load_and_warm(args)
    count = args.batch_size * c.BLOCK_SIZE
    with Path(args.raw).open("rb") as handle:
        data = handle.read(count)
    if len(data) != count:
        raise ValueError("trace input must contain one full batch group")
    values = np.frombuffer(data, dtype=np.uint8).reshape(args.batch_size, c.BLOCK_SIZE)
    predictor.reset(args.batch_size)
    with c._coding_mode(predictor):
        for position in range(1, 129):
            predictor.step(values[:, position - 1])
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(
                activities=activities, record_shapes=True, profile_memory=True) as profile:
            for position in range(129, 193):
                with torch.profiler.record_function("batched_predictor_step"):
                    predictor.step(values[:, position - 1])
        predictor.synchronize()
    temporary = Path(args.trace).with_suffix(".json")
    profile.export_chrome_trace(str(temporary))
    with temporary.open("rb") as source, gzip.open(args.trace, "xb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
    temporary.unlink()
    averages = profile.key_averages()

    def event_value(event, *names):
        for name in names:
            value = getattr(event, name, None)
            if value is not None:
                return float(value)
        return 0.0

    rows = [{
        "name": event.key,
        "self_cpu_time_us": event_value(event, "self_cpu_time_total"),
        "cpu_time_us": event_value(event, "cpu_time_total"),
        "self_cuda_time_us": event_value(event, "self_device_time_total", "self_cuda_time_total"),
        "cuda_time_us": event_value(event, "device_time_total", "cuda_time_total"),
        "calls": int(event.count),
    } for event in averages]
    summary = {
        "batch_size": args.batch_size,
        "warmup_positions_before_trace": 128,
        "traced_positions": 64,
        "setup": setup,
        "top_self_cuda": sorted(rows, key=lambda row: row["self_cuda_time_us"], reverse=True)[:20],
        "top_self_cpu": sorted(rows, key=lambda row: row["self_cpu_time_us"], reverse=True)[:20],
    }
    write_json(args.metrics, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def child_command(args, *, stage, operation, raw, packed, decoded, metrics,
                  batch_size, gpu_sample_every=1, trace=None):
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--stage", stage, "--operation", operation,
        "--raw", str(raw), "--packed", str(packed), "--decoded", str(decoded),
        "--metrics", str(metrics), "--checkpoint", str(args.checkpoint.resolve()),
        "--device", args.device, "--batch-size", str(batch_size),
        "--format", args.format,
        "--gpu-sample-every", str(gpu_sample_every),
        "--range-sample-every", "64",
    ]
    if trace is not None:
        command.extend(["--trace", str(trace)])
    return command


def run_child(command, log_path):
    started = time.perf_counter()
    with Path(log_path).open("x", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - started


def run_one_child(args, *, stage, operation, raw, packed, decoded, metrics,
                  log, batch_size, gpu_sample_every=1, trace=None):
    command = child_command(
        args, stage=stage, operation=operation, raw=raw, packed=packed,
        decoded=decoded, metrics=metrics, batch_size=batch_size,
        gpu_sample_every=gpu_sample_every, trace=trace)
    process_seconds = run_child(command, log)
    result = json.loads(Path(metrics).read_text(encoding="utf-8"))
    result["subprocess_wall_seconds"] = process_seconds
    return result


def metric_seconds(result, operation):
    return result[f"{operation}_seconds"]


def render_report(results):
    baseline = results["baseline_summary"]
    detailed = results["detailed"]
    verification = results["verification"]
    lines = [
        "# Batched codec 10 MB 速度瓶颈分析",
        "",
        ("数据为 enwik9 的 `[100,000,000, 110,000,000)`，共 10,000,000 bytes。"
         "正式配置为 batch size 1024、2048-byte 独立块；编码和解码均使用独立进程。"),
        "",
        "## 无插桩端到端基线",
        "",
        "| 操作 | codec 中位数秒 | 最小秒 | 最大秒 | 变异系数 | 中位吞吐 KB/s | 完整进程中位数秒 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for operation in ("encode", "decode"):
        item = baseline[operation]
        lines.append(
            f"| {operation} | {item['codec']['median']:.6f} | {item['codec']['min']:.6f} | "
            f"{item['codec']['max']:.6f} | {item['codec']['coefficient_of_variation']:.4%} | "
            f"{item['kb_per_second_median']:.3f} | {item['process']['median']:.6f} |")
    lines.extend(["", "## 完整分项墙钟时间", ""])
    component_names = {
        "container_setup": "文件打开、v3 头部与容器初始化",
        "input_read_and_hash": "输入读取与 SHA256",
        "block_buffer_prepare": "分块、缓冲区与尾部补齐",
        "range_stream_seed": "Range 流创建与块首 byte",
        "frame_read_crc_and_seed": "帧读取、CRC 与块首 byte",
        "state_reset": "Mamba 状态创建或清零",
        "token_and_mask_prepare": "每步输入与有效块掩码",
        "predictor_step": "批量模型预测全过程",
        "probability_quantization_and_range_encode": "概率量化、Range 编码与 NLL",
        "probability_quantization_and_range_decode": "概率量化与 Range 解码",
        "range_finalize_crc_and_write": "Range 收尾、帧 CRC 与写入",
        "header_body_read_and_crc": "头部、主体读取与 CRC",
        "output_assemble_write_and_hash": "恢复数据整理、写入与 SHA256",
        "trailer_and_integrity": "尾部及完整性校验",
        "final_gpu_sync": "最终 GPU 同步",
        "unattributed": "未归因及上下文管理开销",
    }
    for operation in ("encode", "decode"):
        profile = detailed[operation]["profile_breakdown"]
        lines.extend([
            f"### {operation}", "",
            "| 环节 | 秒 | 占 profile 总时间 |",
            "| --- | ---: | ---: |",
        ])
        for name, seconds in sorted(profile["components"].items(), key=lambda item: item[1], reverse=True):
            lines.append(
                f"| {component_names.get(name, name)} | {seconds:.6f} | "
                f"{100 * seconds / profile['total_seconds']:.2f}% |")
        overhead = results["profile_overhead"][operation]
        lines.extend([
            "",
            f"插桩总时间为 {profile['total_seconds']:.6f} 秒，相对无插桩中位数偏差 {overhead:.2%}。",
            "",
        ])
    lines.extend([
        "## Predictor 内部设备时间", "",
        "CUDA Event 时间与上面的 CPU 墙钟时间属于不同口径，不直接相加。",
        "",
        "| 操作 | H2D 秒 | GPU 模型秒 | D2H 秒 | CPU 概率生成秒 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for operation in ("encode", "decode"):
        profile = detailed[operation]["profile_breakdown"]
        gpu = profile["cuda_event_seconds_estimated"]
        cpu_probability = profile["predictor_wall_detail"].get("cpu_probability", 0.0)
        lines.append(
            f"| {operation} | {gpu.get('h2d', 0):.6f} | {gpu.get('model', 0):.6f} | "
            f"{gpu.get('d2h', 0):.6f} | {cpu_probability:.6f} |")
    lines.extend([
        "", "## 概率量化与 Range Coding 抽样拆分", "",
        "主分项表中的 Range 环节是完整计时；下表按固定间隔抽样后外推，只用于解释其内部构成。",
        "",
        "| 操作 | 抽样 byte 数 | 总预测 byte 数 | 概率量化估计秒 | Range 编解码估计秒 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for operation in ("encode", "decode"):
        profile = detailed[operation]["profile_breakdown"]
        estimated = profile["range_subcomponents_estimated"]
        coder_key = f"range_{operation}"
        lines.append(
            f"| {operation} | {profile['range_sampled_symbols']} | {profile['range_total_symbols']} | "
            f"{estimated.get('probability_quantization', 0):.6f} | "
            f"{estimated.get(coder_key, 0):.6f} |")
    lines.extend([
        "", "## Batch 缩放短对照", "",
        "每项只处理一个完整 group，均执行 2047 个位置 step；数据量随 batch 成比例变化。",
        "",
        "| Batch | 操作 | bytes | 秒 | KB/s | 每 step 毫秒 | GPU 模型秒 | Range 环节秒 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in results["scaling"]:
        lines.append(
            f"| {row['batch_size']} | {row['operation']} | {row['num_bytes']} | "
            f"{row['seconds']:.6f} | {row['kb_per_second']:.3f} | "
            f"{row['milliseconds_per_step']:.6f} | {row['gpu_model_seconds']:.6f} | "
            f"{row['range_seconds']:.6f} |")
    trace = results["trace_summary"]
    lines.extend([
        "", "## 64-step 模型跟踪", "",
        "该跟踪仅覆盖稳定状态下连续 64 个 B=1024 预测位置，不参与完整流程的加总。",
        "",
        "| CUDA 自耗时最高的算子 | 微秒 | 调用次数 |",
        "| --- | ---: | ---: |",
    ])
    for row in trace["top_self_cuda"][:10]:
        lines.append(f"| `{row['name']}` | {row['self_cuda_time_us']:.3f} | {row['calls']} |")
    bottlenecks = {}
    for operation in ("encode", "decode"):
        components = detailed[operation]["profile_breakdown"]["components"]
        considered = {key: value for key, value in components.items() if key != "unattributed"}
        bottlenecks[operation] = max(considered, key=considered.get)
    lines.extend([
        "", "## 结论", "",
        (f"编码占时最大的互斥环节是“{component_names.get(bottlenecks['encode'], bottlenecks['encode'])}”；"
         f"解码是“{component_names.get(bottlenecks['decode'], bottlenecks['decode'])}”。"),
        "应结合 CUDA Event、Range 抽样拆分、短 batch 对照和 trace 中的热点算子解释原因，"
        "不根据单次总耗时猜测。",
        "", "## 正确性", "",
        f"- 三次基线码流一致：{verification['baseline_compressed_identical']}。",
        f"- 所有恢复文件与输入一致：{verification['all_decoded_identical']}。",
        f"- 插桩码流与生产码流一致：{verification['profile_compressed_identical']}。",
        f"- 输入 SHA256：`{verification['sample_sha256']}`。",
        "",
    ])
    return "\n".join(lines)


def write_csv_files(output, results):
    timing_fields = ["operation", "component", "seconds", "percent",
                     "microseconds_per_step", "nanoseconds_per_byte"]
    with (output / "timings.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=timing_fields)
        writer.writeheader()
        for operation in ("encode", "decode"):
            detail = results["detailed"][operation]
            profile = detail["profile_breakdown"]
            writer.writerows(timing_rows(
                operation, profile["total_seconds"], profile["components"],
                detail["num_bytes"], detail["num_model_steps"]))
    scaling_fields = ["batch_size", "operation", "num_bytes", "seconds", "kb_per_second",
                      "milliseconds_per_step", "gpu_model_seconds", "range_seconds"]
    with (output / "scaling.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scaling_fields)
        writer.writeheader()
        writer.writerows(results["scaling"])


def source_hashes():
    names = (
        "codec.py", "neural_compression/codec.py", "neural_compression/codec_batched.py",
        "neural_compression/codec_batched_v4.py",
        "neural_compression/codec_step_adapter.py", "neural_compression/model.py",
        "scripts/profile_codec_batched.py",
    )
    return {name: hash_file(ROOT / name) for name in names}


def run_parent(args):
    from neural_compression import codec as c

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = output / f"enwik9_{args.offset_bytes}_{args.offset_bytes + args.sample_bytes}.bin"
    sample = extract_slice(args.source, raw, args.offset_bytes, args.sample_bytes)
    canonical_packed = output / "sample.m3rc"
    canonical_decoded = output / "decoded.bin"
    baseline = {"encode": [], "decode": []}
    compressed_hashes = []
    decoded_hashes = []
    with tempfile.TemporaryDirectory(prefix="m3rc-profile-baseline-") as temporary_name:
        temporary = Path(temporary_name)
        for index in range(1, args.repeats + 1):
            packed = canonical_packed if index == 1 else temporary / f"run{index}.m3rc"
            decoded = canonical_decoded if index == 1 else temporary / f"run{index}.decoded"
            encode_metrics = temporary / f"encode{index}.json"
            decode_metrics = temporary / f"decode{index}.json"
            encoded = run_one_child(
                args, stage="baseline", operation="encode", raw=raw, packed=packed,
                decoded=decoded, metrics=encode_metrics,
                log=output / f"baseline_encode_{index}.log", batch_size=args.batch_size)
            decoded_result = run_one_child(
                args, stage="baseline", operation="decode", raw=raw, packed=packed,
                decoded=decoded, metrics=decode_metrics,
                log=output / f"baseline_decode_{index}.log", batch_size=args.batch_size)
            baseline["encode"].append(encoded)
            baseline["decode"].append(decoded_result)
            compressed_hashes.append(hash_file(packed))
            decoded_hashes.append(hash_file(decoded))
    sample_hash = sample["sha256"]
    if any(value != sample_hash for value in decoded_hashes):
        raise ValueError("a baseline decode differs from the input slice")
    if len(set(compressed_hashes)) != 1:
        raise ValueError("baseline encodes produced different compressed streams")

    baseline_summary = {}
    for operation in ("encode", "decode"):
        runs = baseline[operation]
        baseline_summary[operation] = {
            "codec": summarize_values(metric_seconds(run, operation) for run in runs),
            "process": summarize_values(run["subprocess_wall_seconds"] for run in runs),
            "model_load": summarize_values(run["model_load_seconds"] for run in runs),
            "warmup": summarize_values(run["warmup_seconds"] for run in runs),
            "kb_per_second_median": statistics.median(
                run[f"{operation}_kb_per_second"] for run in runs),
        }

    detailed = {}
    profile_hash = None
    profile_decoded_hash = None
    profile_overhead = {}
    with tempfile.TemporaryDirectory(prefix="m3rc-profile-detailed-") as temporary_name:
        temporary = Path(temporary_name)
        for operation in ("encode", "decode"):
            sample_every = 1
            attempt = 1
            while True:
                packed = temporary / f"profile_encode_{attempt}.m3rc" if operation == "encode" else canonical_packed
                decoded = temporary / f"profile_decode_{attempt}.bin"
                metrics = temporary / f"profile_{operation}_{attempt}.json"
                result = run_one_child(
                    args, stage="profile", operation=operation, raw=raw, packed=packed,
                    decoded=decoded, metrics=metrics,
                    log=output / f"profile_{operation}_{attempt}.log",
                    batch_size=args.batch_size, gpu_sample_every=sample_every)
                baseline_median = baseline_summary[operation]["codec"]["median"]
                overhead = metric_seconds(result, operation) / baseline_median - 1
                if overhead <= 0.05 or sample_every == 16:
                    detailed[operation] = result
                    profile_overhead[operation] = overhead
                    if operation == "encode":
                        profile_hash = hash_file(packed)
                    else:
                        profile_decoded_hash = hash_file(decoded)
                    break
                sample_every = 16
                attempt += 1

    if profile_hash != compressed_hashes[0]:
        raise ValueError("instrumented encode differs from production output")
    if profile_decoded_hash != sample_hash:
        raise ValueError("instrumented decode differs from the input slice")

    scaling = []
    if not args.skip_scaling:
        with tempfile.TemporaryDirectory(prefix="m3rc-profile-scaling-") as temporary_name:
            temporary = Path(temporary_name)
            with raw.open("rb") as source:
                full_slice = source.read()
            for batch_size in (512, 1024, 2048):
                count = batch_size * c.BLOCK_SIZE
                subset = temporary / f"b{batch_size}.bin"
                subset.write_bytes(full_slice[:count])
                packed = temporary / f"b{batch_size}.m3rc"
                decoded = temporary / f"b{batch_size}.decoded"
                for operation in ("encode", "decode"):
                    metrics = temporary / f"b{batch_size}_{operation}.json"
                    result = run_one_child(
                        args, stage="profile", operation=operation, raw=subset, packed=packed,
                        decoded=decoded, metrics=metrics,
                        log=output / f"scaling_b{batch_size}_{operation}.log",
                        batch_size=batch_size, gpu_sample_every=16)
                    seconds = metric_seconds(result, operation)
                    breakdown = result["profile_breakdown"]
                    range_key = f"probability_quantization_and_range_{operation}"
                    scaling.append({
                        "batch_size": batch_size,
                        "operation": operation,
                        "num_bytes": count,
                        "seconds": seconds,
                        "kb_per_second": result[f"{operation}_kb_per_second"],
                        "milliseconds_per_step": seconds * 1000 / result["num_model_steps"],
                        "gpu_model_seconds": breakdown["cuda_event_seconds_estimated"].get("model", 0.0),
                        "range_seconds": breakdown["components"].get(range_key, 0.0),
                    })
                if hash_file(decoded) != hash_file(subset):
                    raise ValueError(f"scaling decode failed for batch size {batch_size}")

    trace_summary = {"top_self_cuda": [], "top_self_cpu": [], "skipped": True}
    if not args.skip_trace:
        trace_metrics = output / "trace_summary.json"
        run_one_child(
            args, stage="trace", operation="encode", raw=raw, packed=canonical_packed,
            decoded=canonical_decoded, metrics=trace_metrics,
            log=output / "trace.log", batch_size=args.batch_size, gpu_sample_every=16,
            trace=output / "trace.json.gz")
        trace_summary = json.loads(trace_metrics.read_text(encoding="utf-8"))
        trace_metrics.unlink()

    verification = {
        **sample,
        "source": str(args.source.resolve()),
        "sample_sha256": sample_hash,
        "compressed_sha256": compressed_hashes[0],
        "decoded_sha256": decoded_hashes[0],
        "baseline_compressed_identical": len(set(compressed_hashes)) == 1,
        "all_decoded_identical": all(value == sample_hash for value in decoded_hashes),
        "profile_compressed_identical": profile_hash == compressed_hashes[0],
        "profile_decoded_identical": profile_decoded_hash == sample_hash,
        "profile_metrics_identical": all(
            detailed["encode"].get(key) == baseline["encode"][0].get(key)
            for key in ("theoretical_nll_bpb", "actual_bpb", "payload_bpb",
                        "compressed_bytes", "range_payload_bytes")),
        "independent_processes": all(
            encoded["process_id"] != decoded["process_id"]
            for encoded, decoded in zip(baseline["encode"], baseline["decode"])),
        "source_hashes": source_hashes(),
    }
    results = {
        "configuration": {
            "source": str(args.source.resolve()),
            "offset_bytes": args.offset_bytes,
            "sample_bytes": args.sample_bytes,
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "checkpoint": str(args.checkpoint.resolve()),
            "device": args.device,
            "format": args.format,
            "groups": groups_for(args.sample_bytes, args.batch_size),
        },
        "baseline_runs": baseline,
        "baseline_summary": baseline_summary,
        "detailed": detailed,
        "profile_overhead": profile_overhead,
        "scaling": scaling,
        "trace_summary": trace_summary,
        "verification": verification,
    }
    write_json(output / "verification.json", verification)
    write_json(output / "results.json", results)
    write_csv_files(output, results)
    with (output / "实验报告.md").open("x", encoding="utf-8") as handle:
        handle.write(render_report(results))
    print(json.dumps(verification, indent=2, ensure_ascii=False), flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/raw/enwik9")
    parser.add_argument("--offset-bytes", type=int, default=100_000_000)
    parser.add_argument("--sample-bytes", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--format", choices=("v3", "v4"), default="v3")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument("--skip-trace", action="store_true")
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best-3M.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=("baseline", "profile", "trace"), help=argparse.SUPPRESS)
    parser.add_argument("--operation", choices=("encode", "decode"), help=argparse.SUPPRESS)
    parser.add_argument("--raw", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--packed", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--decoded", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--metrics", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--trace", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-sample-every", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--range-sample-every", type=int, default=64, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.repeats <= 0:
        raise ValueError("batch size and repeats must be positive")
    if args.gpu_sample_every <= 0 or args.range_sample_every <= 0:
        raise ValueError("sampling intervals must be positive")
    if args.stage:
        required = (args.raw, args.packed, args.decoded, args.metrics, args.operation)
        if any(value is None for value in required):
            raise ValueError("child stage arguments are incomplete")
        if args.stage == "trace":
            if args.trace is None:
                raise ValueError("trace stage requires --trace")
            run_trace_stage(args)
        else:
            run_codec_stage(args)
        return
    if args.output is None:
        raise ValueError("--output is required")
    run_parent(args)


if __name__ == "__main__":
    main()

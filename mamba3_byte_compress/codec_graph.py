"""Fixed-buffer SISO inference and bounded, shape-specific CUDA graphs.

This adapter follows the installed Mamba3.step arithmetic, retaining its native
rotary and state-update kernels. Only storage and execution scheduling change.
"""

from collections import OrderedDict
import time

import numpy as np
import torch

from . import codec as c
from .codec_step_adapter import _aligned_rows
from .metrics import autocast_context


def probability_validity(probabilities):
    smallest, largest = torch.aminmax(probabilities, dim=-1)
    return (torch.isfinite(smallest) & torch.isfinite(largest)
            & (smallest >= 0) & (largest > 0)).all()


class BufferedMambaStep:
    def __init__(self, mixer):
        if mixer.is_mimo or mixer.is_outproj_norm:
            raise ValueError("v5 supports SISO without output projection normalization.")
        self.mixer = mixer
        self.buffers = {}

    def aligned(self, name, value):
        if name not in self.buffers:
            self.buffers[name] = _aligned_rows(torch.empty_like(value))
        target = self.buffers[name]
        target.copy_(value)
        return target

    def __call__(self, u, angle_state, ssm_state, k_state, v_state):
        from mamba_ssm.modules.mamba3 import (
            apply_rotary_qk_inference_fwd, mamba3_step_fn,
        )

        m = self.mixer
        z, x, b, q, dt, a, trap, angles = torch.split(
            m.in_proj(u), [m.d_inner, m.d_inner, m.d_state * m.num_bc_heads,
                           m.d_state * m.num_bc_heads, m.nheads, m.nheads,
                           m.nheads, m.num_rope_angles], dim=-1)
        dt, b, q, x, z, trap, a, angles = m._preprocess(a, dt, b, q, x, z, trap, angles)
        dt, x, z, trap, a = [self.aligned(name, value) for name, value in
                              (("dt", dt), ("x", x), ("z", z), ("trap", trap), ("a", a))]
        q, b, next_angle = apply_rotary_qk_inference_fwd(
            q=q, k=b, angle_state=angle_state, angle_proj=angles, dt=dt,
            bias_q=m.C_bias.permute(1, 0, 2), bias_k=m.B_bias.permute(1, 0, 2),
            conjugate=False, inplace=False, rotate_pairwise=True)
        if "ones" not in self.buffers:
            self.buffers["ones"] = torch.ones(
                (1, m.nheads, m.headdim), device=x.device, dtype=x.dtype)
            self.buffers["out"] = torch.empty_like(x)
        ones, y = self.buffers["ones"], self.buffers["out"]
        mamba3_step_fn(ssm_state, k_state, v_state, a, b, q, m.D, x, dt, trap,
                       ones, outproj=ones, state_out=None, out=y, z=z, zproj=ones,
                       tile_D=64, num_warps=4)
        out = m.out_proj(y.reshape(u.shape[0], -1).to(x.dtype))
        angle_state.copy_(next_angle)
        k_state.copy_(b)
        v_state.copy_(x)
        return out


class BatchedStepV5Predictor(c.BatchedStepV4Predictor):
    profile = c.BATCHED_V5_PROFILE

    def __init__(self, *args, use_graph=True, check_probabilities=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_graph = use_graph
        self.check_probabilities = check_probabilities
        self._contexts = OrderedDict()
        self.graph_prepare_seconds = 0.0
        self.graph_cache_hits = 0
        self.graph_capture_count = 0

    def _compute(self):
        with autocast_context(self.device):
            x = self.model.embedding(self._token)
            for block, states, step in zip(self.model.blocks, self.states, self._buffered_steps):
                mixed = step(block.norm1(x), *states)
                x = x + mixed
                x = x + block.mlp(block.norm2(x))
            logits = self.model.lm_head(self.model.final_norm(x)).float()
        probabilities = logits.softmax(dim=-1)
        if self.check_probabilities:
            self._probabilities_valid = probability_validity(probabilities)
        return probabilities

    def _clear_states(self):
        for states in self.states:
            for state in states:
                state.zero_()
        self.steps_in_block = 0

    @torch.inference_mode()
    def reset(self, batch_size=None):
        size = batch_size if batch_size is not None else self.active_batch_size or self.batch_size
        c.validate_batch_size(size)
        if size > self.batch_size:
            raise ValueError("Active batch exceeds the configured batch_size.")
        if size in self._contexts:
            self.graph_cache_hits += 1
            context = self._contexts.pop(size)
            self._contexts[size] = context
            self.__dict__.update(context)
            self._clear_states()
            return
        if len(self._contexts) == 2:
            self._contexts.popitem(last=False)
        # Force allocation of a new set of state addresses for this graph.
        self.active_batch_size = 0
        super().reset(size)
        self._buffered_steps = [BufferedMambaStep(block.mixer) for block in self.model.blocks]
        self._graph = None
        self._probabilities = None
        self._probabilities_valid = None
        self._host_valid = torch.empty((), dtype=torch.bool, pin_memory=self.device.type == "cuda")
        self._prefetch_slots = None
        if self.use_graph:
            if self.device.type != "cuda":
                raise RuntimeError("CUDA Graph inference requires CUDA.")
            started = time.perf_counter()
            # Compilation and lazy allocations must finish before capture. The
            # capture stream and the caller explicitly wait for each other.
            with c._coding_mode(self), torch.cuda.device(self.device):
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    self._token.zero_()
                    for _ in range(3):
                        self._compute()
                torch.cuda.current_stream(self.device).wait_stream(stream)
                self.synchronize()
                self._clear_states()
                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph, stream=stream):
                    self._probabilities = self._compute()
                self.synchronize()
                self._clear_states()
            self.graph_prepare_seconds += time.perf_counter() - started
            self.graph_capture_count += 1
        keys = ("active_batch_size", "states", "_token", "_host_token",
                "_host_probabilities", "_buffered_steps", "_graph", "_probabilities", "_prefetch_slots",
                "_probabilities_valid", "_host_valid")
        self._contexts[size] = {key: getattr(self, key) for key in keys}

    def prepare_prefetch(self):
        """Two owned pinned buffers; no tensor arithmetic or probability changes."""
        if self.device.type != "cuda" or not self.use_graph:
            raise ValueError("Prefetch requires CUDA graph inference")
        if self._prefetch_slots is None:
            self._prefetch_slots = [dict(
                tokens=torch.empty_like(self._host_token, pin_memory=True),
                probabilities=torch.empty_like(self._host_probabilities, pin_memory=True),
                valid=torch.empty((), dtype=torch.bool, pin_memory=True),
                ready=torch.cuda.Event(),
                events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]) for _ in range(2)]
            self._contexts[self.active_batch_size]["_prefetch_slots"] = self._prefetch_slots

    @torch.inference_mode()
    def _enqueue_prefetch(self, tokens, slot):
        timing = getattr(self, "profile_timings", None)
        stream = torch.cuda.current_stream(self.device)
        if timing is not None:
            slot["events"][0].record(stream)
        slot["tokens"].numpy()[:] = tokens
        self._token.copy_(slot["tokens"], non_blocking=True)
        if timing is not None:
            slot["events"][1].record(stream)
        self._graph.replay()
        if timing is not None:
            slot["events"][2].record(stream)
        # All submissions share the current stream: the graph's reusable device
        # output is copied before the next replay can overwrite it.
        slot["probabilities"].copy_(self._probabilities, non_blocking=True)
        if self.check_probabilities:
            slot["valid"].copy_(self._probabilities_valid, non_blocking=True)
        if timing is not None:
            slot["events"][3].record(stream)
        slot["ready"].record(stream)
        self.steps_in_block += 1

    def iter_prefetched(self, values, lengths):
        """Predict known encoder inputs ahead by one step; yield borrowed arrays.

        A yielded array is valid until the consumer advances the iterator. Each
        pinned input/output slot is reused only after its transfer completes and
        the consumer finishes reading it. Recurrent state stays on one stream.
        """
        values, lengths = np.asarray(values), np.asarray(lengths)
        if (self.steps_in_block or values.ndim != 2 or values.shape[0] != self.active_batch_size
                or lengths.shape != (self.active_batch_size,)
                or not np.issubdtype(values.dtype, np.integer)
                or not np.issubdtype(lengths.dtype, np.integer)
                or np.any(lengths < 1) or np.any(lengths > min(values.shape[1], c.BLOCK_SIZE))
                or np.any(values < 0) or np.any(values >= self.vocab_size)):
            raise ValueError("Prefetch requires a fresh block and valid token arrays")
        self.prepare_prefetch()
        limit = int(lengths.max())
        if limit <= 1:
            return
        def submit(position, slot):
            self._enqueue_prefetch(np.where(lengths > position, values[:, position-1], 0), slot)
        submit(1, self._prefetch_slots[0])
        try:
            for position in range(1, limit):
                slot = self._prefetch_slots[(position-1) % 2]
                slot["ready"].synchronize()
                if self.check_probabilities and not slot["valid"].item():
                    raise ValueError("Invalid predicted probabilities")
                timing = getattr(self, "profile_timings", None)
                if timing is not None:
                    events = slot["events"]
                    for i, name in enumerate(("upload", "model_probability", "download")):
                        timing[name] += events[i].elapsed_time(events[i+1]) / 1000
                if position+1 < limit:
                    submit(position+1, self._prefetch_slots[position % 2])
                yield slot["probabilities"].numpy(), None
        finally:
            # Also drain queued work if a consumer rejects a distribution.
            self.synchronize()

    @torch.inference_mode()
    def step(self, tokens, *, active=None):
        tokens = np.asarray(tokens)
        if (not self.active_batch_size or tokens.shape != (self.active_batch_size,)
                or not np.issubdtype(tokens.dtype, np.integer)
                or np.any(tokens < 0) or np.any(tokens >= self.vocab_size)):
            raise ValueError("reset(B) then supply B valid token IDs.")
        if self.steps_in_block >= c.BLOCK_SIZE - 1:
            raise ValueError("An independent block allows 2047 predictions; reset the batch.")
        if active is not None and np.asarray(active, dtype=bool).shape != tokens.shape:
            raise ValueError("Active mask must match the batch shape.")
        timing = getattr(self, "profile_timings", None)
        if timing is not None:
            self.synchronize()
            started = time.perf_counter()
        self._host_token.numpy()[:] = tokens
        self._token.copy_(self._host_token, non_blocking=True)
        if timing is not None:
            self.synchronize()
            timing["upload"] += time.perf_counter() - started
            started = time.perf_counter()
        if self.use_graph:
            self._graph.replay()
            probabilities = self._probabilities
        else:
            probabilities = self._compute()
        if timing is not None:
            self.synchronize()
            timing["model_probability"] += time.perf_counter() - started
            started = time.perf_counter()
        if self.check_probabilities:
            self._host_valid.copy_(self._probabilities_valid, non_blocking=True)
        self._host_probabilities.copy_(probabilities, non_blocking=False)
        if self.check_probabilities and not self._host_valid.item():
            raise ValueError("Invalid predicted probabilities")
        if timing is not None:
            timing["download"] += time.perf_counter() - started
        self.steps_in_block += 1
        return self._host_probabilities.numpy(), None

    def execution_metrics(self):
        return dict(graph_enabled=self.use_graph, graph_prepare_seconds=self.graph_prepare_seconds,
                    graph_cache_hits=self.graph_cache_hits, graph_capture_count=self.graph_capture_count,
                    graph_cached_batch_sizes=list(self._contexts))

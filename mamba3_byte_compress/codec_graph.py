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

    def __init__(self, *args, use_graph=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_graph = use_graph
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
        return logits.softmax(dim=-1)

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
                "_host_probabilities", "_buffered_steps", "_graph", "_probabilities")
        self._contexts[size] = {key: getattr(self, key) for key in keys}

    @torch.inference_mode()
    def step(self, tokens, *, active=None):
        tokens = np.asarray(tokens)
        if (not self.active_batch_size or tokens.shape != (self.active_batch_size,)
                or not np.issubdtype(tokens.dtype, np.integer)
                or np.any(tokens < 0) or np.any(tokens > 255)):
            raise ValueError("reset(B) then supply B byte values in 0..255.")
        if self.steps_in_block >= c.BLOCK_SIZE - 1:
            raise ValueError("An independent block allows 2047 predictions; reset the batch.")
        if active is not None and np.asarray(active, dtype=bool).shape != tokens.shape:
            raise ValueError("Active mask must match the batch shape.")
        self._host_token.numpy()[:] = tokens
        self._token.copy_(self._host_token, non_blocking=True)
        if self.use_graph:
            self._graph.replay()
            probabilities = self._probabilities
        else:
            probabilities = self._compute()
        self._host_probabilities.copy_(probabilities, non_blocking=False)
        self.steps_in_block += 1
        return self._host_probabilities.numpy(), None

    def execution_metrics(self):
        return dict(graph_enabled=self.use_graph, graph_prepare_seconds=self.graph_prepare_seconds,
                    graph_cache_hits=self.graph_cache_hits, graph_capture_count=self.graph_capture_count,
                    graph_cached_batch_sizes=list(self._contexts))


class MTP4GraphPredictor(c.MTP4BatchedPredictor):
    """MTP-4 continuation with stable buffers and one graph per input length."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._contexts = OrderedDict()
        self.graph_prepare_seconds = 0.0
        self.graph_cache_hits = 0
        self.graph_capture_count = 0

    def _clear_states(self):
        for states in self.states:
            for state in states:
                state.zero_()
        self.state_tokens_in_block = 0

    def _graph_compute(self, length):
        return super()._compute(
            self._tokens[:, :length], None, check_finite=False
        )

    @torch.inference_mode()
    def reset(self, batch_size=None):
        size = (
            batch_size
            if batch_size is not None
            else self.active_batch_size or self.batch_size
        )
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

        # Each graph retains tensor addresses, so every batch shape owns states,
        # token/probability buffers and both length-specific graphs.
        self.active_batch_size = 0
        super().reset(size)
        started = time.perf_counter()
        with c._coding_mode(self), torch.cuda.device(self.device):
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                self._tokens.zero_()
                for _ in range(3):
                    self._graph_compute(1)
                    self._clear_states()
                    self._graph_compute(4)
                    self._clear_states()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self.synchronize()

            self._clear_states()
            self._start_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._start_graph, stream=stream):
                self._start_probabilities = self._graph_compute(1)
            self.synchronize()

            self._clear_states()
            self._advance_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._advance_graph, stream=stream):
                self._advance_probabilities = self._graph_compute(4)
            self.synchronize()
            self._clear_states()

        self.graph_prepare_seconds += time.perf_counter() - started
        self.graph_capture_count += 2
        keys = (
            "active_batch_size", "states", "_tokens", "_host_tokens",
            "_host_probabilities", "_start_graph", "_start_probabilities",
            "_advance_graph", "_advance_probabilities",
        )
        self._contexts[size] = {key: getattr(self, key) for key in keys}

    def _copy_graph_probabilities(self, probabilities, active):
        self._host_probabilities.copy_(probabilities, non_blocking=False)
        result = self._host_probabilities.numpy()
        checked = result if active is None else result[active]
        if not np.isfinite(checked).all():
            raise FloatingPointError("The MTP-4 graph produced non-finite probabilities.")
        return result

    @torch.inference_mode()
    def start(self, seed_tokens, *, active=None):
        seed_tokens = np.asarray(seed_tokens)
        if (
            not self.active_batch_size
            or seed_tokens.shape != (self.active_batch_size,)
            or not np.issubdtype(seed_tokens.dtype, np.integer)
            or np.any(seed_tokens < 0)
            or np.any(seed_tokens > 255)
        ):
            raise ValueError("reset(B) then supply B seed byte values in 0..255.")
        if self.state_tokens_in_block != 0:
            raise ValueError("start() may only be called once after reset().")
        active = self._validate_active(active)
        self._host_tokens[:, 0].numpy()[:] = seed_tokens
        self._tokens[:, 0].copy_(self._host_tokens[:, 0], non_blocking=True)
        self._start_graph.replay()
        result = self._copy_graph_probabilities(
            self._start_probabilities, active
        )
        self.state_tokens_in_block = 1
        return result

    @torch.inference_mode()
    def advance(self, tokens, *, active=None):
        tokens = np.asarray(tokens)
        if (
            not self.active_batch_size
            or tokens.shape != (self.active_batch_size, 4)
            or not np.issubdtype(tokens.dtype, np.integer)
            or np.any(tokens < 0)
            or np.any(tokens > 255)
        ):
            raise ValueError("advance() requires B groups of four byte values.")
        if self.state_tokens_in_block == 0:
            raise ValueError("start() must precede advance().")
        if self.state_tokens_in_block + 4 > c.BLOCK_SIZE:
            raise ValueError("MTP continuation exceeds the 2048-byte block.")
        active = self._validate_active(active)
        self._host_tokens.numpy()[:] = tokens
        self._tokens.copy_(self._host_tokens, non_blocking=True)
        self._advance_graph.replay()
        result = self._copy_graph_probabilities(
            self._advance_probabilities, active
        )
        self.state_tokens_in_block += 4
        return result

    def execution_metrics(self):
        return {
            "graph_enabled": True,
            "graph_prepare_seconds": self.graph_prepare_seconds,
            "graph_cache_hits": self.graph_cache_hits,
            "graph_capture_count": self.graph_capture_count,
            "graph_cached_batch_sizes": list(self._contexts),
        }

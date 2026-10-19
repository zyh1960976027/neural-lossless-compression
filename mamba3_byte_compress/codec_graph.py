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
                or np.any(tokens < 0) or np.any(tokens >= self.vocab_size)):
            raise ValueError("reset(B) then supply B valid token IDs.")
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

"""Short-sequence continuation for the installed Mamba-3 SISO kernel."""

import torch
import torch.nn.functional as F
from einops import rearrange


CONTINUATION_KERNEL_PROFILE = (
    "chunk16_angle-w2-s2-rnone_siso-w4-s1-r256"
)
# Triton's dot path requires a token tile of at least 16. Lengths 1 and 4 are
# masked within this smallest supported tile instead of the training tile 64.
CONTINUATION_CHUNK_SIZE = 16
_KERNELS_PINNED = False


def pin_continuation_kernels():
    """Pin Triton launch configs so separate codec processes agree bit-for-bit."""
    global _KERNELS_PINNED
    if _KERNELS_PINNED:
        return CONTINUATION_KERNEL_PROFILE

    from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd_kernel
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd import (
        mamba3_siso_fwd_kernel,
    )

    requested = (
        (angle_dt_fwd_kernel, 2, 2, None, "angle"),
        (mamba3_siso_fwd_kernel, 4, 1, 256, "siso"),
    )
    for kernel, warps, stages, maxnreg, name in requested:
        matches = [
            config
            for config in kernel.configs
            if config.num_warps == warps
            and config.num_stages == stages
            and config.maxnreg == maxnreg
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Installed Mamba-3/Triton does not provide the fixed {name} "
                "continuation kernel configuration."
            )
        kernel.configs = matches
        kernel.cache.clear()
    _KERNELS_PINNED = True
    return CONTINUATION_KERNEL_PROFILE


class Mamba3SequenceContinuation:
    """Run a short sequence from existing recurrent states and update them."""

    def __init__(self, mixer):
        if mixer.is_mimo or mixer.is_outproj_norm:
            raise ValueError(
                "MTP continuation requires SISO without output projection normalization."
            )
        if mixer.num_bc_heads != 1 or mixer.mimo_rank != 1:
            raise ValueError("MTP continuation requires one SISO B/C group.")
        pin_continuation_kernels()
        self.mixer = mixer

    def __getattr__(self, name):
        return getattr(self.mixer, name)

    def __call__(self, inputs, angle_state, ssm_state, k_state, v_state):
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import (
            mamba3_siso_combined,
        )

        if inputs.ndim != 3 or inputs.shape[-1] != self.mixer.d_model:
            raise ValueError("Continuation inputs must have shape [batch, length, d_model].")
        if inputs.shape[1] not in (1, 4):
            raise ValueError("MTP continuation only accepts sequence lengths 1 and 4.")

        mixer = self.mixer
        projected = mixer.in_proj(inputs)
        z, x, b, q, dd_dt, dd_a, trap, angles = torch.split(
            projected,
            [
                mixer.d_inner,
                mixer.d_inner,
                mixer.d_state * mixer.num_bc_heads,
                mixer.d_state * mixer.num_bc_heads,
                mixer.nheads,
                mixer.nheads,
                mixer.nheads,
                mixer.num_rope_angles,
            ],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", p=mixer.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=mixer.headdim)
        b = rearrange(b, "b l (r g n) -> b l r g n", r=1, g=1)
        q = rearrange(q, "b l (r g n) -> b l r g n", r=1, g=1)
        trap = rearrange(trap, "b l h -> b h l")

        a = torch.clamp(-F.softplus(dd_a.float()), max=-mixer.A_floor)
        dt = F.softplus(dd_dt + mixer.dt_bias)
        adt = rearrange(a * dt, "b l h -> b h l")
        dt = rearrange(dt, "b l h -> b h l")
        angles = angles.unsqueeze(-2).expand(
            -1, -1, mixer.nheads, -1
        ).float()
        b = mixer.B_norm(b).squeeze(2)
        q = mixer.C_norm(q).squeeze(2)

        result = mamba3_siso_combined(
            Q=q,
            K=b,
            V=x,
            ADT=adt,
            DT=dt,
            Trap=trap,
            Q_bias=mixer.C_bias.squeeze(1),
            K_bias=mixer.B_bias.squeeze(1),
            Angles=angles,
            D=mixer.D,
            Z=z,
            Input_States=(angle_state, ssm_state, k_state.squeeze(1), v_state),
            chunk_size=CONTINUATION_CHUNK_SIZE,
            return_final_states=True,
        )
        y, final_angle, final_ssm, final_k, final_v = result
        angle_state.copy_(final_angle)
        ssm_state.copy_(final_ssm)
        k_state.copy_(final_k.unsqueeze(1))
        v_state.copy_(final_v)
        y = rearrange(y, "b l h p -> b l (h p)")
        return mixer.out_proj(y.to(x.dtype))

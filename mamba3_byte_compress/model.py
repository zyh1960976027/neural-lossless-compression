"""Mamba-3 SISO byte models for the depth experiment and frozen baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba3

from .config import MODEL_NAME


VOCAB_SIZE = 256

BASELINE_MAMBA3_CONFIG = {
    "status": "ready",
    "family": "mamba3_siso",
    "label": "Mamba-3 SISO",
    "n_layer": 2,
    "d_model": 352,
    "expand": 2,
    "d_state": 128,
    "head_dim": 64,
    "n_groups": 1,
    "rope_fraction": 0.5,
    "chunk_size": 64,
    "is_outproj_norm": False,
    "ffn_hidden": 736,
    "lm_head_init": "zeros",
    "expected_backbone_params": 3_275_212,
    "expected_total_trainable_params": 3_455_436,
    "name": "mamba3_siso_3m",
}

FORMAL_MAMBA3_CONFIG = {
    **BASELINE_MAMBA3_CONFIG,
    "n_layer": 4,
    "expected_backbone_params": 6_550_072,
    "expected_total_trainable_params": 6_730_296,
    "name": MODEL_NAME,
}

SUPPORTED_MAMBA3_CONFIGS = {
    config["name"]: config
    for config in (BASELINE_MAMBA3_CONFIG, FORMAL_MAMBA3_CONFIG)
}


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(*normalized_shape))

    def forward(self, x):
        inv_rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return self.weight * x * inv_rms


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_size):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_size, bias=False)
        self.up = nn.Linear(d_model, hidden_size, bias=False)
        self.down = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Mamba3ByteBlock(nn.Module):
    def __init__(self, model_config, layer_id):
        super().__init__()
        d_model = int(model_config["d_model"])
        self.norm1 = RMSNorm(d_model)
        self.mixer = Mamba3(
            d_model=d_model,
            d_state=int(model_config["d_state"]),
            expand=int(model_config["expand"]),
            headdim=int(model_config["head_dim"]),
            ngroups=int(model_config["n_groups"]),
            rope_fraction=float(model_config["rope_fraction"]),
            is_outproj_norm=bool(model_config["is_outproj_norm"]),
            is_mimo=False,
            chunk_size=int(model_config["chunk_size"]),
            dropout=0.0,
            layer_idx=layer_id,
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, int(model_config["ffn_hidden"]))

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Mamba3ByteLM(nn.Module):
    def __init__(self, vocab_size, model_config):
        super().__init__()
        d_model = int(model_config["d_model"])
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                Mamba3ByteBlock(model_config, layer_id)
                for layer_id in range(int(model_config["n_layer"]))
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if model_config.get("lm_head_init") != "zeros":
            raise ValueError("The fixed experiment requires lm_head_init='zeros'.")
        nn.init.zeros_(self.lm_head.weight)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x)).float()


def analytic_parameter_counts(model_config, vocab_size=VOCAB_SIZE):
    if model_config.get("family") != "mamba3_siso":
        raise ValueError("Only the Mamba-3 SISO model family is supported.")

    d_model = int(model_config["d_model"])
    n_layer = int(model_config["n_layer"])
    d_inner = int(model_config["expand"]) * d_model
    head_dim = int(model_config["head_dim"])
    if d_inner % head_dim:
        raise ValueError("Mamba-3 d_inner must be divisible by head_dim.")

    n_heads = d_inner // head_dim
    d_state = int(model_config["d_state"])
    n_groups = int(model_config["n_groups"])
    rope_width = int(d_state * float(model_config["rope_fraction"]))
    rope_width -= rope_width % 2
    rope_angles = rope_width // 2
    d_in_proj = 2 * d_inner + 2 * d_state * n_groups + 3 * n_heads + rope_angles
    mixer = (
        d_model * d_in_proj
        + n_heads
        + 2 * n_heads * d_state
        + 2 * d_state
        + n_heads
        + d_inner * d_model
    )
    if model_config["is_outproj_norm"]:
        mixer += d_inner
    swiglu = 3 * d_model * int(model_config["ffn_hidden"])
    block = mixer + swiglu + 2 * d_model
    backbone = n_layer * block + d_model
    total = backbone + 2 * int(vocab_size) * d_model
    return {
        "backbone_trainable_params": backbone,
        "total_trainable_params": total,
    }


def parameter_counts(model):
    backbone = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("embedding.")
        and not name.startswith("lm_head.")
    )
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "backbone_trainable_params": backbone,
        "total_trainable_params": total,
    }


def build_mamba3_byte_model(
    model_config=None,
    vocab_size=VOCAB_SIZE,
    verify_counts=True,
):
    if int(vocab_size) != VOCAB_SIZE:
        raise ValueError("The frozen byte baseline requires vocab_size=256.")
    model_config = (
        dict(FORMAL_MAMBA3_CONFIG)
        if model_config is None
        else dict(model_config)
    )
    model = Mamba3ByteLM(vocab_size, model_config)

    if verify_counts:
        actual = parameter_counts(model)
        analytic = analytic_parameter_counts(model_config, vocab_size)
        expected = {
            "backbone_trainable_params": int(
                model_config["expected_backbone_params"]
            ),
            "total_trainable_params": int(
                model_config["expected_total_trainable_params"]
            ),
        }
        if actual != analytic or actual != expected:
            raise AssertionError(
                f"Parameter-count mismatch: actual={actual}, "
                f"analytic={analytic}, expected={expected}."
            )
    return model

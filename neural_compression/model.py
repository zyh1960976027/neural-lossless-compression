"""固定的两层 Mamba-3 BPE 4096 模型及检查点加载。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import MODEL_CONFIG, VOCAB_SIZE


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


class MambaBlock(nn.Module):
    def __init__(self, model_config, layer_id):
        super().__init__()
        d_model = int(model_config["d_model"])
        self.norm1 = RMSNorm(d_model)
        from mamba_ssm import Mamba3

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


class CompressionModel(nn.Module):
    def __init__(self, vocab_size, model_config):
        super().__init__()
        d_model = int(model_config["d_model"])
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(model_config, layer_id)
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
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "backbone_trainable_params": backbone,
        "total_trainable_params": total,
    }


def build_model():
    config = dict(MODEL_CONFIG)
    model = CompressionModel(VOCAB_SIZE, config)
    if parameter_counts(model) != analytic_parameter_counts(config, VOCAB_SIZE):
        raise AssertionError("Model parameter count mismatch")
    return model, config


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["vocab_size"] != VOCAB_SIZE or state["model_config"] != MODEL_CONFIG:
        raise ValueError("需要当前两层 Mamba-3、BPE 4096 配置的检查点")
    model, _ = build_model()
    model.load_state_dict(state["model"], strict=True)
    return model, state


def autocast_context(device, enabled=True):
    device = torch.device(device)
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=bool(enabled and device.type == "cuda"),
    )

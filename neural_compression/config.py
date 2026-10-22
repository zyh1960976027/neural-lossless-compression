"""当前主线的路径、模型和训练配置。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/bpe4096/inference.pt"
DEFAULT_TOKENIZER = PROJECT_ROOT / "checkpoints/bpe4096/tokenizer.model"
DEFAULT_SOURCE = PROJECT_ROOT / "datasets/raw/enwik9"
VOCAB_SIZE = 4096
BLOCK_SIZE = 2048
DEFAULT_BATCH_SIZE = 1024
TRAIN_BATCH_SIZE = 72
TRAINING = dict(
    seed=2026,
    peak_lr=0.0006,
    min_lr=0.000006,
    betas=(0.9, 0.99),
    eps=1e-8,
    max_grad_norm=5.0,
    threshold_bpb=0.001,
    reduce_patience=5,
    stop_patience=15,
    progress_seconds=43200,
    checkpoint_seconds=900,
)
MODEL_CONFIG = {
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
    "expected_backbone_params": 3275212,
    "expected_total_trainable_params": 6158796,
    "name": "mamba3_siso_2l_v4096",
    "vocab_size": 4096,
}

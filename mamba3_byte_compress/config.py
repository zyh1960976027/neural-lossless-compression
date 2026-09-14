"""Frozen configuration for the formal Mamba-3 byte baseline."""

from copy import deepcopy


MODEL_NAME = "mamba3_siso_4l"

FORMAL_TRAINING_CONFIG = {
    "optimizer": "adam",
    "peak_lr": 0.0006,
    "betas": (0.9, 0.99),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "max_grad_norm": 5.0,
    "precision": "bf16_amp_fp32_master",
    "total_updates": 50_000,
    "warmup_updates": 2_000,
    "min_lr_ratio": 0.1,
    "global_batch_size": 32,
    "micro_batch_size": 32,
    "sequence_length": 2_048,
    "validation_batch_size": 16,
    "validation_interval": 2_000,
    "checkpoint_interval": 2_000,
    "seed": 2_026,
}


def validate_training_config(config):
    required = set(FORMAL_TRAINING_CONFIG)
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing training settings: {', '.join(missing)}")
    if config["optimizer"] != "adam":
        raise ValueError("The frozen baseline requires the Adam optimizer.")
    if config["precision"] != "bf16_amp_fp32_master":
        raise ValueError("The frozen baseline requires BF16 AMP with FP32 parameters.")

    total_updates = int(config["total_updates"])
    warmup_updates = int(config["warmup_updates"])
    global_batch_size = int(config["global_batch_size"])
    micro_batch_size = int(config["micro_batch_size"])
    if total_updates <= 0 or warmup_updates < 0 or warmup_updates >= total_updates:
        raise ValueError("Invalid warmup/total update schedule.")
    if global_batch_size <= 0 or micro_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if global_batch_size % micro_batch_size:
        raise ValueError("micro_batch_size must divide global_batch_size exactly.")
    for key in (
        "sequence_length",
        "validation_batch_size",
        "validation_interval",
        "checkpoint_interval",
    ):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    return config


def make_training_config(**overrides):
    """Return an isolated baseline config, optionally shortened for tests."""

    config = deepcopy(FORMAL_TRAINING_CONFIG)
    config.update(overrides)
    return validate_training_config(config)

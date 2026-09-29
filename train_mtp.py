"""Train the two-layer Mamba-3 byte model with four future-byte heads."""

import argparse
import json
from pathlib import Path

import torch

from mamba3_byte_compress.config import FORMAL_TRAINING_CONFIG, make_training_config
from mamba3_byte_compress.experiments import (
    TRAIN_EXPERIMENTS_ROOT,
    create_experiment_directory,
    validate_experiment_name,
)
from mamba3_byte_compress.model import MTP4_MAMBA3_CONFIG, build_mamba3_mtp4_model
from mamba3_byte_compress.mtp import MTP4ByteTrainer
from mamba3_byte_compress.trainer import seed_everything


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--train-starts", required=True)
    parser.add_argument("--name", type=validate_experiment_name,
                        default=MTP4_MAMBA3_CONFIG["name"])
    parser.add_argument("--run-dir", help="Existing experiment directory for --resume")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--micro-batch-size", type=int,
                        default=FORMAL_TRAINING_CONFIG["micro_batch_size"])
    parser.add_argument("--total-updates", type=int,
                        default=FORMAL_TRAINING_CONFIG["total_updates"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", nargs="?", const="latest")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("MTP-4 training requires a CUDA device.")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("MTP-4 training requires BF16 support.")
    training_config = make_training_config(
        micro_batch_size=args.micro_batch_size,
        total_updates=args.total_updates,
    )
    seed_everything(int(training_config["seed"]))
    model = build_mamba3_mtp4_model()
    if args.resume is None:
        if args.run_dir is not None:
            raise ValueError("--run-dir requires --resume.")
        run_dir = create_experiment_directory(TRAIN_EXPERIMENTS_ROOT, args.name)
    else:
        if args.run_dir is None:
            raise ValueError("--resume requires --run-dir.")
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Missing experiment directory: {run_dir}")
    trainer = MTP4ByteTrainer(
        model=model,
        train_file=args.train_file,
        validation_file=args.validation_file,
        train_starts_file=args.train_starts,
        run_dir=run_dir,
        device=device,
        training_config=training_config,
        num_workers=args.num_workers,
        allow_training_extension=args.total_updates > FORMAL_TRAINING_CONFIG["total_updates"],
    )
    if args.resume is not None:
        trainer.resume(None if args.resume == "latest" else args.resume)
    summary = trainer.train()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()

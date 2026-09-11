"""CLI for the formal Mamba-3 SISO byte baseline training run."""

import argparse
import json
from pathlib import Path

import torch

from mamba3_byte_compress.config import FORMAL_TRAINING_CONFIG, make_training_config
from mamba3_byte_compress.model import build_mamba3_byte_model
from mamba3_byte_compress.trainer import ByteTrainer, seed_everything


def parse_args(argv=None):
    parser = argparse.ArgumentParser("Mamba-3 SISO byte baseline training")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--train-starts", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=FORMAL_TRAINING_CONFIG["micro_batch_size"],
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The formal Mamba-3 baseline requires a CUDA device.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device must support BF16.")

    training_config = make_training_config(
        micro_batch_size=args.micro_batch_size
    )
    seed_everything(int(training_config["seed"]))
    model = build_mamba3_byte_model()

    run_dir = Path(args.run_dir).resolve()
    if args.resume is None and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. Use --resume or a new directory."
        )
    trainer = ByteTrainer(
        model=model,
        train_file=args.train_file,
        validation_file=args.validation_file,
        train_starts_file=args.train_starts,
        run_dir=run_dir,
        device=device,
        training_config=training_config,
        num_workers=args.num_workers,
    )
    if args.resume is not None:
        resume_path = None if args.resume == "latest" else args.resume
        trainer.resume(resume_path)

    summary = trainer.train()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

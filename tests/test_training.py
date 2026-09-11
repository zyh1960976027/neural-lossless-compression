import json
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train
from mamba3_byte_compress.experiments import TRAIN_EXPERIMENTS_ROOT
from mamba3_byte_compress.config import FORMAL_TRAINING_CONFIG, make_training_config
from mamba3_byte_compress.metrics import autocast_context
from mamba3_byte_compress.model import FORMAL_MAMBA3_CONFIG, build_mamba3_byte_model
from mamba3_byte_compress.trainer import (
    ByteTrainer,
    WarmupCosineScheduler,
    load_model_checkpoint,
    seed_everything,
)


LEGACY_COMMIT = "a5cd9baa24ed2b3959af5ce4572644d6076c69a9"
LEGACY_ROOT = Path(
    os.environ.get(
        "OMNIZIP_BASELINE_ROOT", "/media/ivc-17/New Volume/zyh/omnizip"
    )
)


class TinyByteLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 8)
        self.backbone = torch.nn.Linear(8, 8, bias=False)
        self.lm_head = torch.nn.Linear(8, 256, bias=False)

    def forward(self, inputs):
        hidden = torch.tanh(self.backbone(self.embedding(inputs)))
        return self.lm_head(hidden).float()


class TrainingTest(unittest.TestCase):
    def test_cli_defaults_to_training_experiments_directory(self):
        args = train.parse_args([
            "--train-file", "train.bin", "--validation-file", "validation.bin",
            "--train-starts", "starts.npy",
        ])
        self.assertEqual(Path(args.run_dir).parent, TRAIN_EXPERIMENTS_ROOT)

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _fixture(self, total_updates, global_batch_size, sequence_length):
        train_file = self.root / "train.bin"
        validation_file = self.root / "validation.bin"
        starts_file = self.root / "starts.npy"
        train_file.write_bytes(bytes([7]) * (sequence_length + 1))
        validation_file.write_bytes(bytes([7]) * (sequence_length + 1))
        np.save(
            starts_file,
            np.zeros(total_updates * global_batch_size, dtype=np.int64),
            allow_pickle=False,
        )
        return train_file, validation_file, starts_file

    def _tiny_config(self, total_updates=4):
        return make_training_config(
            peak_lr=0.03,
            total_updates=total_updates,
            warmup_updates=1,
            global_batch_size=4,
            micro_batch_size=2,
            sequence_length=4,
            validation_batch_size=2,
            validation_interval=1,
            checkpoint_interval=1,
        )

    def _trainer(self, model, run_dir, config, trainer_class=ByteTrainer):
        files = self._fixture(
            config["total_updates"],
            config["global_batch_size"],
            config["sequence_length"],
        )
        return trainer_class(
            model=model,
            train_file=files[0],
            validation_file=files[1],
            train_starts_file=files[2],
            run_dir=run_dir,
            device="cpu",
            training_config=config,
            model_name="tiny",
            model_config={"family": "tiny"},
            num_workers=0,
        )

    @unittest.skipUnless(LEGACY_ROOT.is_dir(), "frozen OmniZip baseline unavailable")
    def test_formal_training_config_matches_frozen_baseline(self):
        resolved_commit = subprocess.check_output(
            [
                "git",
                "-C",
                str(LEGACY_ROOT),
                "rev-parse",
                "byte-backbone-baseline-v1^{commit}",
            ],
            text=True,
        ).strip()
        self.assertEqual(resolved_commit, LEGACY_COMMIT)
        subprocess.run(
            [
                "git",
                "-C",
                str(LEGACY_ROOT),
                "diff",
                "--quiet",
                "byte-backbone-baseline-v1",
                "--",
                "byte_experiment/training.py",
                "byte_experiment/train_cli.py",
                "configs/byte_backbone/enwik9_rwkv7_mamba3.json",
            ],
            check=True,
        )
        old_config = json.loads(
            (
                LEGACY_ROOT
                / "configs/byte_backbone/enwik9_rwkv7_mamba3.json"
            ).read_text(encoding="utf-8")
        )
        old_training = old_config["training"]
        expected = {
            "optimizer": old_training["optimizer"],
            "peak_lr": 0.0006,
            "betas": tuple(old_training["betas"]),
            "eps": old_training["eps"],
            "weight_decay": old_training["weight_decay"],
            "max_grad_norm": old_training["max_grad_norm"],
            "precision": old_training["precision"],
            "total_updates": old_training["formal_updates"],
            "warmup_updates": old_training["formal_warmup_updates"],
            "min_lr_ratio": old_training["min_lr_ratio"],
            "global_batch_size": old_training["global_batch_size"],
            "micro_batch_size": 32,
            "sequence_length": old_config["data"]["sequence_length"],
            "validation_batch_size": old_config["evaluation"]["batch_size"],
            "validation_interval": old_training["validation_interval"],
            "checkpoint_interval": old_training["checkpoint_interval"],
            "seed": old_config["data"]["seed"],
        }
        self.assertEqual(FORMAL_TRAINING_CONFIG, expected)

    def test_warmup_cosine_schedule_matches_baseline(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=1e-3)
        scheduler = WarmupCosineScheduler(
            optimizer,
            peak_lr=1e-3,
            total_updates=10,
            warmup_updates=2,
            min_lr_ratio=0.1,
        )
        values = [scheduler.lr_for_update(index) for index in range(10)]
        self.assertEqual(values[:2], [5e-4, 1e-3])
        self.assertTrue(all(left >= right for left, right in zip(values[1:], values[2:])))
        self.assertTrue(math.isclose(values[-1], 1e-4, rel_tol=1e-12))

    def test_forward_backward_update_loss_and_checkpoints(self):
        config = self._tiny_config(total_updates=4)
        seed_everything(config["seed"])
        trainer = self._trainer(TinyByteLM(), self.root / "train", config)
        batch = next(iter(trainer._training_loader()))
        inputs = batch["input_tokens"]
        targets = batch["target_tokens"]
        with torch.no_grad():
            initial_loss = F.cross_entropy(
                trainer.model(inputs).reshape(-1, 256), targets.reshape(-1)
            ).item()

        summary = trainer.train()
        with torch.no_grad():
            final_loss = F.cross_entropy(
                trainer.model(inputs).reshape(-1, 256), targets.reshape(-1)
            ).item()

        self.assertLess(final_loss, initial_loss)
        self.assertEqual(summary["global_update"], 4)
        self.assertEqual(summary["tokens_seen"], 64)
        self.assertEqual(summary["best_update"], 4)
        self.assertEqual(trainer.accumulation_steps, 2)
        self.assertTrue(trainer.latest_checkpoint_path.is_file())
        self.assertTrue(trainer.best_checkpoint_path.is_file())
        latest = torch.load(
            trainer.latest_checkpoint_path, map_location="cpu", weights_only=False
        )
        best = torch.load(
            trainer.best_checkpoint_path, map_location="cpu", weights_only=False
        )
        self.assertEqual(latest["checkpoint_kind"], "latest")
        self.assertEqual(best["checkpoint_kind"], "best")
        self.assertEqual(best["global_update"], 4)
        self.assertEqual(latest["window_sample_index"], 16)
        self.assertEqual(latest["scheduler"]["last_update"], 3)
        self.assertIn("rng_state", latest)
        loaded_model = TinyByteLM()
        loaded = load_model_checkpoint(
            loaded_model,
            trainer.best_checkpoint_path,
            expected_model_name="tiny",
            expected_model_config={"family": "tiny"},
            require_best=True,
        )
        for name, value in loaded_model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded["model"][name]), name)

    def test_resume_matches_uninterrupted_training(self):
        config = self._tiny_config(total_updates=4)

        seed_everything(config["seed"])
        uninterrupted = self._trainer(
            TinyByteLM(), self.root / "uninterrupted", config
        )
        uninterrupted.train()
        expected_state = {
            name: value.detach().clone()
            for name, value in uninterrupted.model.state_dict().items()
        }

        class StopAtUpdateTwo(ByteTrainer):
            def save_checkpoint(self, kind, validation_metrics=None):
                path = super().save_checkpoint(kind, validation_metrics)
                if kind == "latest" and self.global_update == 2:
                    raise RuntimeError("simulated interruption")
                return path

        seed_everything(config["seed"])
        interrupted = self._trainer(
            TinyByteLM(),
            self.root / "interrupted",
            config,
            trainer_class=StopAtUpdateTwo,
        )
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            interrupted.train()

        seed_everything(9_999)
        resumed = self._trainer(TinyByteLM(), self.root / "interrupted", config)
        checkpoint = resumed.resume()
        self.assertEqual(checkpoint["global_update"], 2)
        resumed.train()

        self.assertEqual(resumed.global_update, uninterrupted.global_update)
        self.assertEqual(resumed.tokens_seen, uninterrupted.tokens_seen)
        for name, value in resumed.model.state_dict().items():
            self.assertTrue(torch.equal(value, expected_state[name]), name)

    def test_fixed_seed_is_reproducible(self):
        config = self._tiny_config(total_updates=2)
        states = []
        for index in range(2):
            seed_everything(config["seed"])
            trainer = self._trainer(
                TinyByteLM(), self.root / f"reproducible_{index}", config
            )
            trainer.train()
            states.append(
                {
                    name: value.detach().clone()
                    for name, value in trainer.model.state_dict().items()
                }
            )
        for name in states[0]:
            self.assertTrue(torch.equal(states[0][name], states[1][name]), name)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by Mamba-3")
    def test_mamba3_short_smoke_training(self):
        config = make_training_config(
            total_updates=1,
            warmup_updates=0,
            global_batch_size=1,
            micro_batch_size=1,
            sequence_length=64,
            validation_batch_size=1,
            validation_interval=1,
            checkpoint_interval=1,
        )
        files = self._fixture(1, 1, 64)
        seed_everything(config["seed"])
        model = build_mamba3_byte_model()
        trainer = ByteTrainer(
            model=model,
            train_file=files[0],
            validation_file=files[1],
            train_starts_file=files[2],
            run_dir=self.root / "mamba_smoke",
            device="cuda:0",
            training_config=config,
            model_config=FORMAL_MAMBA3_CONFIG,
            num_workers=0,
        )
        tokens = torch.full((1, 64), 7, dtype=torch.long, device="cuda")
        with torch.inference_mode(), autocast_context("cuda:0", enabled=True):
            initial_loss = F.cross_entropy(
                trainer.model(tokens).reshape(-1, 256).float(), tokens.reshape(-1)
            ).item()

        summary = trainer.train()
        with torch.inference_mode(), autocast_context("cuda:0", enabled=True):
            final_loss = F.cross_entropy(
                trainer.model(tokens).reshape(-1, 256).float(), tokens.reshape(-1)
            ).item()

        self.assertLess(final_loss, initial_loss)
        self.assertEqual(summary["global_update"], 1)
        self.assertTrue(trainer.latest_checkpoint_path.is_file())
        self.assertTrue(trainer.best_checkpoint_path.is_file())


if __name__ == "__main__":
    unittest.main()

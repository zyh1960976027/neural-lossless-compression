"""Minimal training and checkpoint flow for the frozen byte baseline."""

import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import MODEL_NAME, make_training_config
from .data import ByteEvaluationDataset, ByteTrainingDataset
from .metrics import autocast_context, evaluate_probability_model
from .model import FORMAL_MAMBA3_CONFIG, parameter_counts


class WarmupCosineScheduler:
    def __init__(self, optimizer, peak_lr, total_updates, warmup_updates, min_lr_ratio):
        if total_updates <= 0 or warmup_updates < 0 or warmup_updates >= total_updates:
            raise ValueError("Invalid warmup/total update schedule.")
        self.optimizer = optimizer
        self.peak_lr = float(peak_lr)
        self.total_updates = int(total_updates)
        self.warmup_updates = int(warmup_updates)
        self.min_lr_ratio = float(min_lr_ratio)
        self.last_update = -1

    def lr_for_update(self, update_index):
        update_index = int(update_index)
        if self.warmup_updates and update_index < self.warmup_updates:
            return self.peak_lr * (update_index + 1) / self.warmup_updates
        decay_updates = self.total_updates - self.warmup_updates
        decay_index = update_index - self.warmup_updates
        progress = min(max(decay_index / max(1, decay_updates - 1), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        ratio = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        return self.peak_lr * ratio

    def step(self, update_index):
        learning_rate = self.lr_for_update(update_index)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.last_update = int(update_index)
        return learning_rate

    def state_dict(self):
        return {
            "peak_lr": self.peak_lr,
            "total_updates": self.total_updates,
            "warmup_updates": self.warmup_updates,
            "min_lr_ratio": self.min_lr_ratio,
            "last_update": self.last_update,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ("peak_lr", "total_updates", "warmup_updates", "min_lr_ratio"):
            if state[key] != expected[key]:
                raise ValueError(f"Scheduler checkpoint mismatch for {key}.")
        self.last_update = int(state["last_update"])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _atomic_torch_save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def _atomic_json_dump(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _data_fingerprint(train_file, validation_file, train_starts_file):
    paths = {
        "train": Path(train_file).resolve(),
        "validation": Path(validation_file).resolve(),
        "training_windows": Path(train_starts_file).resolve(),
    }
    fingerprint = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} file: {path}")
        fingerprint[name] = {
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return fingerprint


class ByteTrainer:
    def __init__(
        self,
        *,
        model,
        train_file,
        validation_file,
        train_starts_file,
        run_dir,
        device,
        training_config=None,
        model_name=MODEL_NAME,
        model_config=None,
        num_workers=0,
    ):
        self.model = model
        self.model_name = model_name
        self.model_config = dict(
            FORMAL_MAMBA3_CONFIG if model_config is None else model_config
        )
        self.config = make_training_config(**(training_config or {}))
        self.train_file = Path(train_file).resolve()
        self.validation_file = Path(validation_file).resolve()
        self.train_starts_file = Path(train_starts_file).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.num_workers = int(num_workers)

        self.global_batch_size = int(self.config["global_batch_size"])
        self.micro_batch_size = int(self.config["micro_batch_size"])
        self.accumulation_steps = self.global_batch_size // self.micro_batch_size
        self.total_updates = int(self.config["total_updates"])
        self.sequence_length = int(self.config["sequence_length"])
        required_samples = self.total_updates * self.global_batch_size
        starts = np.load(self.train_starts_file, mmap_mode="r", allow_pickle=False)
        if starts.dtype != np.int64 or starts.ndim != 1:
            raise ValueError("Training starts must be a one-dimensional int64 array.")
        if required_samples > len(starts):
            raise ValueError("Training-window file does not contain enough samples.")
        self.data_fingerprint = _data_fingerprint(
            self.train_file, self.validation_file, self.train_starts_file
        )

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=float(self.config["peak_lr"]),
            betas=tuple(float(value) for value in self.config["betas"]),
            eps=float(self.config["eps"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            peak_lr=float(self.config["peak_lr"]),
            total_updates=self.total_updates,
            warmup_updates=int(self.config["warmup_updates"]),
            min_lr_ratio=float(self.config["min_lr_ratio"]),
        )
        self.global_update = 0
        self.tokens_seen = 0
        self.best_bpb = math.inf
        self.best_update = None
        self.last_validation_update = None

    @property
    def latest_checkpoint_path(self):
        return self.run_dir / "checkpoints" / "latest.pt"

    @property
    def best_checkpoint_path(self):
        return self.run_dir / "checkpoints" / "best.pt"

    def _checkpoint(self, kind, validation_metrics=None):
        return {
            "schema_version": 1,
            "checkpoint_kind": kind,
            "model_name": self.model_name,
            "model_config": self.model_config,
            "phase": "formal",
            "training_config": self.config,
            "precision": self.config["precision"],
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_update": self.global_update,
            "tokens_seen": self.tokens_seen,
            "window_sample_index": self.global_update * self.global_batch_size,
            "data_fingerprint": self.data_fingerprint,
            "micro_batch_size": self.micro_batch_size,
            "accumulation_steps": self.accumulation_steps,
            "best_bpb": self.best_bpb,
            "best_update": self.best_update,
            "validation_metrics": validation_metrics,
            "rng_state": capture_rng_state(),
            "parameter_counts": parameter_counts(self.model),
        }

    def save_checkpoint(self, kind, validation_metrics=None):
        if kind not in {"best", "latest"}:
            raise ValueError("Checkpoint kind must be 'best' or 'latest'.")
        path = self.best_checkpoint_path if kind == "best" else self.latest_checkpoint_path
        _atomic_torch_save(self._checkpoint(kind, validation_metrics), path)
        return path

    def resume(self, path=None):
        path = self.latest_checkpoint_path if path is None else Path(path)
        checkpoint = _torch_load(path, map_location="cpu")
        for key, expected in (
            ("model_name", self.model_name),
            ("model_config", self.model_config),
            ("phase", "formal"),
            ("training_config", self.config),
            ("data_fingerprint", self.data_fingerprint),
            ("micro_batch_size", self.micro_batch_size),
            ("accumulation_steps", self.accumulation_steps),
        ):
            if checkpoint.get(key) != expected:
                raise ValueError(f"Checkpoint {key} does not match the requested run.")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_update = int(checkpoint["global_update"])
        self.tokens_seen = int(checkpoint["tokens_seen"])
        expected_sample_index = self.global_update * self.global_batch_size
        if int(checkpoint["window_sample_index"]) != expected_sample_index:
            raise ValueError("Checkpoint training-window position is inconsistent.")
        self.best_bpb = float(checkpoint["best_bpb"])
        self.best_update = checkpoint["best_update"]
        restore_rng_state(checkpoint["rng_state"])
        return checkpoint

    def _training_loader(self):
        sample_offset = self.global_update * self.global_batch_size
        sample_count = (self.total_updates - self.global_update) * self.global_batch_size
        dataset = ByteTrainingDataset(
            raw_path=self.train_file,
            starts_path=self.train_starts_file,
            sequence_length=self.sequence_length,
            sample_offset=sample_offset,
            sample_count=sample_count,
        )
        return DataLoader(
            dataset,
            batch_size=self.micro_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.num_workers > 0,
        )

    def _validation_loader(self):
        dataset = ByteEvaluationDataset(self.validation_file, self.sequence_length)
        return DataLoader(
            dataset,
            batch_size=int(self.config["validation_batch_size"]),
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.num_workers > 0,
        )

    def validate(self):
        metrics = evaluate_probability_model(
            self.model,
            self._validation_loader(),
            self.device,
            use_bf16=True,
        )
        metrics.update(
            {
                "split": "validation",
                "global_update": self.global_update,
                "tokens_seen": self.tokens_seen,
            }
        )
        self.last_validation_update = self.global_update
        _atomic_json_dump(
            metrics, self.run_dir / f"validation_update_{self.global_update:06d}.json"
        )
        if metrics["bpb"] < self.best_bpb:
            self.best_bpb = metrics["bpb"]
            self.best_update = self.global_update
            self.save_checkpoint("best", validation_metrics=metrics)
        return metrics

    def _write_log(self, record):
        with open(self.run_dir / "train.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def train(self):
        if self.global_update >= self.total_updates:
            raise ValueError("Checkpoint has already reached the requested update budget.")
        loader_iterator = iter(self._training_loader())
        validation_interval = int(self.config["validation_interval"])
        checkpoint_interval = int(self.config["checkpoint_interval"])
        self.model.train()

        while self.global_update < self.total_updates:
            learning_rate = self.scheduler.step(self.global_update)
            self.optimizer.zero_grad(set_to_none=True)
            update_nll = 0.0
            for _ in range(self.accumulation_steps):
                batch = next(loader_iterator)
                inputs = batch["input_tokens"].to(self.device, non_blocking=True)
                targets = batch["target_tokens"].to(self.device, non_blocking=True)
                with autocast_context(self.device, enabled=True):
                    logits = self.model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    targets.reshape(-1),
                    reduction="mean",
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at update {self.global_update}."
                    )
                (loss / self.accumulation_steps).backward()
                update_nll += float(loss.detach()) / self.accumulation_steps

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float(self.config["max_grad_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at update {self.global_update}."
                )
            self.optimizer.step()
            self.global_update += 1
            self.tokens_seen = (
                self.global_update * self.global_batch_size * self.sequence_length
            )

            if self.global_update == 1 or self.global_update % 100 == 0:
                record = {
                    "global_update": self.global_update,
                    "tokens_seen": self.tokens_seen,
                    "nll_nats_per_byte_batch": update_nll,
                    "bpb_batch": update_nll / math.log(2.0),
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm),
                }
                self._write_log(record)
                print(json.dumps(record, sort_keys=True), flush=True)

            if self.global_update % validation_interval == 0:
                metrics = self.validate()
                print(json.dumps(metrics, sort_keys=True), flush=True)
                self.model.train()
            if self.global_update % checkpoint_interval == 0:
                self.save_checkpoint("latest")

        if self.last_validation_update != self.global_update:
            metrics = self.validate()
            print(json.dumps(metrics, sort_keys=True), flush=True)
        self.save_checkpoint("latest")
        summary = {
            "model_name": self.model_name,
            "peak_lr": float(self.config["peak_lr"]),
            "micro_batch_size": self.micro_batch_size,
            "accumulation_steps": self.accumulation_steps,
            "global_update": self.global_update,
            "tokens_seen": self.tokens_seen,
            "best_bpb": self.best_bpb,
            "best_update": self.best_update,
            "best_checkpoint": str(self.best_checkpoint_path),
            "latest_checkpoint": str(self.latest_checkpoint_path),
        }
        _atomic_json_dump(summary, self.run_dir / "training_summary.json")
        return summary


def load_model_checkpoint(
    model,
    checkpoint_path,
    *,
    expected_model_name=MODEL_NAME,
    expected_model_config=None,
    require_best=False,
):
    checkpoint = _torch_load(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_name") != expected_model_name:
        raise ValueError("Checkpoint model name does not match the requested model.")
    expected_model_config = dict(
        FORMAL_MAMBA3_CONFIG
        if expected_model_config is None
        else expected_model_config
    )
    if checkpoint.get("model_config") != expected_model_config:
        raise ValueError("Checkpoint model config does not match the requested model.")
    if require_best and checkpoint.get("checkpoint_kind") != "best":
        raise ValueError("A best-validation checkpoint is required.")
    if require_best and checkpoint.get("phase") != "formal":
        raise ValueError("A formal-training checkpoint is required.")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint

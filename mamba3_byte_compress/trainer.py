"""Minimal training and checkpoint flow for the frozen byte baseline."""

import hashlib
import io
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
from .model import (
    FORMAL_MAMBA3_CONFIG,
    SUPPORTED_MAMBA3_CONFIGS,
    build_mamba3_byte_model,
    parameter_counts,
)


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


def _plot_bpb_history(training_metrics, validation_metrics, path):
    """Write a compact linear BPB curve for visual convergence checks."""
    from PIL import Image, ImageDraw, ImageFont

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1600, 960
    left, top, right, bottom = 145, 100, 70, 135
    plot_left, plot_top = left, top
    plot_right, plot_bottom = width - right, height - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 17)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        font = small_font = title_font = ImageFont.load_default()

    training_points = [
        (int(record["global_update"]), float(record["bpb_batch"]))
        for record in training_metrics
    ]
    validation_points = [
        (int(record["global_update"]), float(record["bpb"]))
        for record in validation_metrics
    ]
    # Validation BPB is the convergence signal. Mixing the much larger early
    # batch BPB into the same y-axis would visually flatten its late plateau.
    plotted_points = validation_points if validation_points else training_points
    series_name = "Validation BPB" if validation_points else "Training batch BPB"
    if not plotted_points:
        raise ValueError("Cannot plot an empty BPB history.")
    x_min = min(step for step, _ in plotted_points)
    x_max = max(step for step, _ in plotted_points)
    y_min = min(bpb for _, bpb in plotted_points)
    y_max = max(bpb for _, bpb in plotted_points)
    if x_min == x_max:
        x_min, x_max = 0, max(1, x_max)
    if y_min == y_max:
        padding = max(abs(y_min) * 0.05, 0.1)
    else:
        padding = (y_max - y_min) * 0.08
    y_min -= padding
    y_max += padding

    def chart_point(point):
        step, bpb = point
        x = plot_left + (step - x_min) / (x_max - x_min) * (plot_right - plot_left)
        y = plot_bottom - (bpb - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
        return round(x), round(y)

    for tick in range(6):
        fraction = tick / 5
        x = round(plot_left + fraction * (plot_right - plot_left))
        y = round(plot_bottom - fraction * (plot_bottom - plot_top))
        draw.line((x, plot_top, x, plot_bottom), fill="#e3e7eb", width=1)
        draw.line((plot_left, y, plot_right, y), fill="#e3e7eb", width=1)
        step_label = f"{round(x_min + fraction * (x_max - x_min)):,}"
        bpb_label = f"{y_min + fraction * (y_max - y_min):.4f}"
        draw.text((x - 28, plot_bottom + 14), step_label, fill="#4a5560", font=small_font)
        draw.text((plot_left - 80, y - 9), bpb_label, fill="#4a5560", font=small_font)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#202830", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#202830", width=2)

    if not validation_points:
        points = [chart_point(point) for point in plotted_points]
        if len(points) > 1:
            draw.line(points, fill="#8fb9df", width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#8fb9df")
    if validation_points:
        points = [chart_point(point) for point in validation_points]
        if len(points) > 1:
            draw.line(points, fill="#d1495b", width=5)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#d1495b")
        best_point = min(validation_points, key=lambda point: point[1])
        best_x, best_y = chart_point(best_point)
        draw.ellipse(
            (best_x - 11, best_y - 11, best_x + 11, best_y + 11),
            fill="#2a9d8f",
            outline="white",
            width=3,
        )
        recent_values = [bpb for _, bpb in validation_points[-5:]]
        recent_change = recent_values[0] - recent_values[-1]
        draw.text(
            (plot_right - 310, plot_top + 12),
            f"Recent BPB improvement: {recent_change:.6f}",
            fill="#202830",
            font=small_font,
        )

    draw.text(
        (width // 2 - 145, 38),
        "Validation BPB vs. Training Steps",
        fill="#202830",
        font=title_font,
    )
    draw.text((width // 2 - 55, height - 55), "Training step", fill="#202830", font=font)
    draw.text((25, plot_top - 40), "Bits per byte (lower is better)", fill="#202830", font=font)
    legend_y = height - 95
    series_color = "#d1495b" if validation_points else "#8fb9df"
    draw.line((plot_left, legend_y, plot_left + 35, legend_y), fill=series_color, width=5)
    draw.text((plot_left + 45, legend_y - 10), series_name, fill="#202830", font=font)
    if validation_points:
        best_bpb = min(bpb for _, bpb in validation_points)
        draw.ellipse(
            (plot_left + 235, legend_y - 7, plot_left + 249, legend_y + 7),
            fill="#2a9d8f",
        )
        draw.text(
            (plot_left + 260, legend_y - 10),
            f"Best validation BPB: {best_bpb:.6f}",
            fill="#202830",
            font=font,
        )

    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG")
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


def _npy_prefix_fingerprint(path, count):
    """Fingerprint a prefix using the same on-disk representation as np.save."""
    starts = np.load(path, mmap_mode="r", allow_pickle=False)
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(starts[:count], dtype=np.int64), allow_pickle=False)
    payload = buffer.getvalue()
    return {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


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
        allow_training_extension=False,
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
        self.allow_training_extension = bool(allow_training_extension)
        self.training_extension = None

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
        self.experiment_data_path = self.run_dir / "experiment_data.json"
        self.experiment_summary_path = self.run_dir / "experiment_summary.json"
        self.bpb_plot_path = self.run_dir / "bpb_vs_steps.png"
        self.training_metrics = []
        self.validation_metrics = []
        if self.experiment_data_path.is_file():
            with self.experiment_data_path.open(encoding="utf-8") as handle:
                saved_data = json.load(handle)
            training_metrics = saved_data.get("training_metrics")
            validation_metrics = saved_data.get("validation_metrics")
            if not isinstance(training_metrics, list) or not isinstance(
                validation_metrics, list
            ):
                raise ValueError("Invalid experiment_data.json metric history.")
            self.training_metrics = training_metrics
            self.validation_metrics = validation_metrics

    @property
    def latest_checkpoint_path(self):
        return self.run_dir / "checkpoints" / "latest.pt"

    @property
    def best_checkpoint_path(self):
        return self.run_dir / "checkpoints" / "best.pt"

    def _checkpoint(self, kind, validation_metrics=None):
        checkpoint = {
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
        if self.training_extension is not None:
            checkpoint["training_extension"] = self.training_extension
        return checkpoint

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
            ("micro_batch_size", self.micro_batch_size),
            ("accumulation_steps", self.accumulation_steps),
        ):
            if checkpoint.get(key) != expected:
                raise ValueError(f"Checkpoint {key} does not match the requested run.")
        checkpoint_config = checkpoint.get("training_config")
        checkpoint_fingerprint = checkpoint.get("data_fingerprint")
        extending = checkpoint_config != self.config
        if extending:
            old_config = dict(checkpoint_config or {})
            new_config = dict(self.config)
            old_total = old_config.pop("total_updates", None)
            new_total = new_config.pop("total_updates", None)
            if (
                not self.allow_training_extension
                or old_config != new_config
                or not isinstance(old_total, int)
                or new_total <= old_total
                or int(checkpoint.get("global_update", -1)) > old_total
            ):
                raise ValueError(
                    "Checkpoint training_config does not match the requested run."
                )
            for name in ("train", "validation"):
                if checkpoint_fingerprint.get(name) != self.data_fingerprint.get(name):
                    raise ValueError(
                        f"Checkpoint data_fingerprint for {name} does not match."
                    )
            old_windows = checkpoint_fingerprint.get("training_windows")
            old_window_count = old_total * int(checkpoint_config["global_batch_size"])
            if (
                old_window_count <= 0
                or _npy_prefix_fingerprint(self.train_starts_file, old_window_count)
                != old_windows
            ):
                raise ValueError(
                    "Extended training windows do not preserve the checkpoint prefix."
                )
            self.training_extension = {
                "from_total_updates": old_total,
                "to_total_updates": new_total,
                "scheduler_policy": "preserve_original_then_hold_minimum_lr",
                "original_scheduler_total_updates": int(
                    checkpoint["scheduler"]["total_updates"]
                ),
            }
        elif checkpoint_fingerprint != self.data_fingerprint:
            raise ValueError(
                "Checkpoint data_fingerprint does not match the requested run."
            )
        elif checkpoint.get("training_extension") is not None:
            self.training_extension = dict(checkpoint["training_extension"])
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.allow_training_extension:
            self.scheduler.total_updates = int(checkpoint["scheduler"]["total_updates"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.global_update = int(checkpoint["global_update"])
        self.tokens_seen = int(checkpoint["tokens_seen"])
        expected_sample_index = self.global_update * self.global_batch_size
        if int(checkpoint["window_sample_index"]) != expected_sample_index:
            raise ValueError("Checkpoint training-window position is inconsistent.")
        self.best_bpb = float(checkpoint["best_bpb"])
        self.best_update = checkpoint["best_update"]
        self.training_metrics = [
            record
            for record in self.training_metrics
            if int(record["global_update"]) <= self.global_update
        ]
        self.validation_metrics = [
            record
            for record in self.validation_metrics
            if int(record["global_update"]) <= self.global_update
        ]
        self.last_validation_update = (
            int(self.validation_metrics[-1]["global_update"])
            if self.validation_metrics
            else None
        )
        self._write_experiment_data()
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
        self.validation_metrics.append(metrics)
        self._write_experiment_data()
        if metrics["bpb"] < self.best_bpb:
            self.best_bpb = metrics["bpb"]
            self.best_update = self.global_update
            self.save_checkpoint("best", validation_metrics=metrics)
        return metrics

    def _write_experiment_data(self):
        _atomic_json_dump(
            {
                "schema_version": 1,
                "training_metrics": self.training_metrics,
                "validation_metrics": self.validation_metrics,
            },
            self.experiment_data_path,
        )

    def _write_log(self, record):
        self.training_metrics.append(record)
        self._write_experiment_data()

    def _batch_loss(self, logits, targets):
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
            reduction="mean",
        )
        return loss, {}

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
            update_components = {}
            for _ in range(self.accumulation_steps):
                batch = next(loader_iterator)
                inputs = batch["input_tokens"].to(self.device, non_blocking=True)
                targets = batch["target_tokens"].to(self.device, non_blocking=True)
                with autocast_context(self.device, enabled=True):
                    logits = self.model(inputs)
                loss, components = self._batch_loss(logits, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at update {self.global_update}."
                    )
                (loss / self.accumulation_steps).backward()
                update_nll += float(loss.detach()) / self.accumulation_steps
                for name, value in components.items():
                    update_components[name] = (
                        update_components.get(name, 0.0)
                        + float(value.detach()) / self.accumulation_steps
                    )

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
                record.update(update_components)
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
        final_results = {
            "global_update": self.global_update,
            "tokens_seen": self.tokens_seen,
            "best_bpb": self.best_bpb,
            "best_update": self.best_update,
            "final_validation": self.validation_metrics[-1],
            "best_checkpoint": str(self.best_checkpoint_path),
            "latest_checkpoint": str(self.latest_checkpoint_path),
        }
        summary = {
            "schema_version": 1,
            "experiment_name": self.run_dir.name,
            "experiment_dir": str(self.run_dir),
            "configuration": {
                "model_name": self.model_name,
                "model_config": self.model_config,
                "training_config": self.config,
                "train_file": str(self.train_file),
                "validation_file": str(self.validation_file),
                "train_starts_file": str(self.train_starts_file),
                "data_fingerprint": self.data_fingerprint,
                "device": str(self.device),
                "num_workers": self.num_workers,
                "micro_batch_size": self.micro_batch_size,
                "accumulation_steps": self.accumulation_steps,
                "parameter_counts": parameter_counts(self.model),
            },
            "final_results": final_results,
        }
        summary = json.loads(json.dumps(summary, allow_nan=False))
        self._write_experiment_data()
        _atomic_json_dump(summary, self.experiment_summary_path)
        _plot_bpb_history(
            self.training_metrics, self.validation_metrics, self.bpb_plot_path
        )
        return summary


def _validate_checkpoint_metadata(
    checkpoint, expected_model_name, expected_model_config, require_best=False,
):
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid project checkpoint: expected a dictionary.")
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


def load_model_checkpoint(
    model,
    checkpoint_path,
    *,
    expected_model_name=MODEL_NAME,
    expected_model_config=None,
    require_best=False,
):
    checkpoint = _torch_load(checkpoint_path, map_location="cpu")
    _validate_checkpoint_metadata(
        checkpoint, expected_model_name, expected_model_config, require_best,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint


def load_mamba3_checkpoint(checkpoint_path):
    """Build either supported depth from checkpoint metadata and load strictly."""
    checkpoint = _torch_load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid project checkpoint: expected a dictionary.")
    model_name = checkpoint.get("model_name")
    if not isinstance(model_name, str) or model_name not in SUPPORTED_MAMBA3_CONFIGS:
        raise ValueError("Checkpoint model name is not a supported Mamba-3 model.")
    model_config = SUPPORTED_MAMBA3_CONFIGS[model_name]
    _validate_checkpoint_metadata(checkpoint, model_name, model_config)
    model = build_mamba3_byte_model(model_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint

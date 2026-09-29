"""MTP-4 training and probability-only evaluation for the two-layer byte model."""

import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .metrics import autocast_context, masked_nll
from .model import MTP4_MAMBA3_CONFIG, build_mamba3_mtp4_model
from .trainer import (
    ByteTrainer,
    _atomic_json_dump,
    _torch_load,
    _validate_checkpoint_metadata,
)


WEIGHTED_MTP_LOSS_WEIGHTS = (1.0, 0.5, 0.25, 0.125)


def _plot_mtp_series(records, series, path, title, y_label):
    """Render one complete MTP metric history without optional plot packages."""
    from PIL import Image, ImageDraw, ImageFont

    points_by_series = [
        [
            (int(record["global_update"]), float(record[key]))
            for record in records if key in record
        ]
        for _, key, _ in series
    ]
    if not records or any(not points for points in points_by_series):
        raise ValueError("Cannot plot an incomplete MTP metric history.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1600, 960
    left, top, right, bottom = 135, 95, 65, 175
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 19)
        small = ImageFont.truetype("DejaVuSans.ttf", 16)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        font = small = title_font = ImageFont.load_default()

    all_points = [point for points in points_by_series for point in points]
    x_min = min(step for step, _ in all_points)
    x_max = max(step for step, _ in all_points)
    y_min = min(value for _, value in all_points)
    y_max = max(value for _, value in all_points)
    if x_min == x_max:
        x_min, x_max = 0, max(1, x_max)
    padding = max((y_max - y_min) * 0.06, 1e-6)
    y_min -= padding
    y_max += padding

    def chart_point(point):
        step, value = point
        x = left + (step - x_min) / (x_max - x_min) * (width - left - right)
        y = height - bottom - (value - y_min) / (y_max - y_min) * (
            height - top - bottom
        )
        return round(x), round(y)

    for tick in range(6):
        fraction = tick / 5
        x = round(left + fraction * (width - left - right))
        y = round(height - bottom - fraction * (height - top - bottom))
        draw.line((x, top, x, height - bottom), fill="#e3e7eb")
        draw.line((left, y, width - right, y), fill="#e3e7eb")
        draw.text(
            (x - 30, height - bottom + 14),
            f"{round(x_min + fraction * (x_max - x_min)):,}",
            fill="#4a5560", font=small,
        )
        draw.text(
            (25, y - 9), f"{y_min + fraction * (y_max - y_min):.3f}",
            fill="#4a5560", font=small,
        )
    draw.line((left, top, left, height - bottom), fill="#202830", width=2)
    draw.line(
        (left, height - bottom, width - right, height - bottom),
        fill="#202830", width=2,
    )
    for (label, _, color), points in zip(series, points_by_series):
        draw.line([chart_point(point) for point in points], fill=color, width=4)
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((width - (title_box[2] - title_box[0])) // 2, 35),
        title, fill="#202830", font=title_font,
    )
    x_label = "Training step"
    x_label_box = draw.textbbox((0, 0), x_label, font=font)
    draw.text(
        ((width - (x_label_box[2] - x_label_box[0])) // 2, height - 108),
        x_label, fill="#202830", font=font,
    )
    draw.text((25, 55), y_label, fill="#202830", font=font)
    legend_x = left
    legend_y = height - 55
    for label, _, color in series:
        draw.line((legend_x, legend_y, legend_x + 32, legend_y), fill=color, width=4)
        draw.text((legend_x + 40, legend_y - 9), label, fill="#202830", font=small)
        legend_x += 245
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, path)


def mtp_head_losses(logits, targets):
    """Return one mean cross-entropy per prediction distance."""
    if logits.ndim != 4 or logits.shape[-2:] != (4, 256):
        raise ValueError("MTP logits must have shape [batch, time, 4, 256].")
    if targets.shape != logits.shape[:2] or targets.shape[1] < 4:
        raise ValueError("MTP training requires at least four target positions.")
    length = targets.shape[1]
    return tuple(
        F.cross_entropy(
            logits[:, :length - offset, offset, :].reshape(-1, 256).float(),
            targets[:, offset:].reshape(-1),
        )
        for offset in range(4)
    )


@torch.no_grad()
def evaluate_mtp_probability_model(model, dataloader, device, use_bf16=True):
    """Compute each head's predicted-byte BPB without coding or first-byte cost."""
    dataset = dataloader.dataset
    if not hasattr(dataset, "predicted_bytes"):
        raise TypeError("MTP evaluation requires ByteEvaluationDataset metadata.")
    device = torch.device(device)
    nll = [0.0] * 4
    counts = [0] * 4
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    try:
        for batch in dataloader:
            inputs = batch["input_tokens"].to(device, non_blocking=True)
            targets = batch["target_tokens"].to(device, non_blocking=True)
            masks = batch["masks"].to(device, non_blocking=True)
            with autocast_context(device, enabled=use_bf16):
                logits = model(inputs)
            if logits.ndim != 4 or logits.shape[-2:] != (4, 256):
                raise ValueError("MTP logits must have shape [batch, time, 4, 256].")
            length = targets.shape[1]
            for offset in range(min(4, length)):
                batch_nll, batch_count = masked_nll(
                    logits[:, :length - offset, offset, :],
                    targets[:, offset:],
                    masks[:, offset:],
                )
                nll[offset] += batch_nll
                counts[offset] += batch_count
    finally:
        if was_training:
            model.train()
    if counts[0] != dataset.predicted_bytes:
        raise AssertionError("MTP Head1 did not cover every next-byte target.")
    head_bpb = [
        loss / (count * math.log(2.0)) if count else None
        for loss, count in zip(nll, counts)
    ]
    result = {
        "bpb": sum(head_bpb) / 4 if all(value is not None for value in head_bpb) else None,
        "num_bytes": int(dataset.raw_bytes),
        "sequence_length": int(dataset.sequence_length),
        "context_reset_each_block": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    for index, (loss, count, bpb) in enumerate(zip(nll, counts, head_bpb), start=1):
        result[f"head{index}_nll_nats"] = loss
        result[f"head{index}_predicted_bytes"] = count
        result[f"head{index}_bpb"] = bpb
    return result


class MTP4ByteTrainer(ByteTrainer):
    """Reuse baseline data, optimizer, scheduler and checkpoints with MTP loss."""

    def __init__(self, **kwargs):
        kwargs.setdefault("model_name", MTP4_MAMBA3_CONFIG["name"])
        kwargs.setdefault("model_config", MTP4_MAMBA3_CONFIG)
        super().__init__(**kwargs)
        self.training_elapsed_seconds = 0.0
        self.peak_training_gpu_memory_gb = None
        self._train_started = None

    def _batch_loss(self, logits, targets):
        losses = mtp_head_losses(logits, targets)
        total = sum(losses) / 4
        components = {f"loss{index}": loss for index, loss in enumerate(losses, start=1)}
        components["total_loss"] = total
        return total, components

    def validate(self):
        metrics = evaluate_mtp_probability_model(
            self.model, self._validation_loader(), self.device, use_bf16=True
        )
        if metrics["bpb"] is None or not math.isfinite(metrics["bpb"]):
            raise ValueError("MTP validation needs at least one target for each head.")
        metrics.update(
            split="validation", global_update=self.global_update,
            tokens_seen=self.tokens_seen,
        )
        self.last_validation_update = self.global_update
        self.validation_metrics.append(metrics)
        self._write_experiment_data()
        if metrics["bpb"] < self.best_bpb:
            self.best_bpb = metrics["bpb"]
            self.best_update = self.global_update
            self.save_checkpoint("best", validation_metrics=metrics)
        return metrics

    def _current_training_seconds(self):
        elapsed = 0.0 if self._train_started is None else time.perf_counter() - self._train_started
        return self.training_elapsed_seconds + elapsed

    def _current_peak_memory_gb(self):
        peak = self.peak_training_gpu_memory_gb
        if self.device.type == "cuda":
            current = torch.cuda.max_memory_allocated(self.device) / 1_000_000_000.0
            peak = current if peak is None else max(peak, current)
        return peak

    def _checkpoint(self, kind, validation_metrics=None):
        checkpoint = super()._checkpoint(kind, validation_metrics)
        checkpoint.update(
            selection_metric="mean_head_bpb",
            training_elapsed_seconds=self._current_training_seconds(),
            peak_training_gpu_memory_gb=self._current_peak_memory_gb(),
        )
        return checkpoint

    def resume(self, path=None):
        checkpoint = super().resume(path)
        if checkpoint.get("selection_metric") != "mean_head_bpb":
            raise ValueError("MTP checkpoint selection metric does not match.")
        self.training_elapsed_seconds = float(checkpoint["training_elapsed_seconds"])
        self.peak_training_gpu_memory_gb = checkpoint["peak_training_gpu_memory_gb"]
        return checkpoint

    def train(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._train_started = time.perf_counter()
        try:
            summary = super().train()
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.peak_training_gpu_memory_gb = self._current_peak_memory_gb()
            self.training_elapsed_seconds = self._current_training_seconds()
            self._train_started = None
        summary["final_results"].update(
            training_elapsed_seconds=self.training_elapsed_seconds,
            peak_training_gpu_memory_gb=self.peak_training_gpu_memory_gb,
            selection_metric="mean_head_bpb",
        )
        _atomic_json_dump(summary, self.experiment_summary_path)
        return summary


class WeightedMTP4ByteTrainer(MTP4ByteTrainer):
    """MTP-4 trainer with one fixed distance-decay loss weighting."""

    def __init__(self, *, scheduler_total_updates=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_weights = WEIGHTED_MTP_LOSS_WEIGHTS
        if scheduler_total_updates is not None:
            scheduler_total_updates = int(scheduler_total_updates)
            if scheduler_total_updates <= int(self.config["warmup_updates"]):
                raise ValueError("Weighted MTP scheduler horizon must exceed warmup.")
            self.scheduler.total_updates = scheduler_total_updates

    def _batch_loss(self, logits, targets):
        losses = mtp_head_losses(logits, targets)
        denominator = sum(self.loss_weights)
        total = sum(
            weight * loss for weight, loss in zip(self.loss_weights, losses)
        ) / denominator
        components = {
            f"loss{index}": loss for index, loss in enumerate(losses, start=1)
        }
        components["weighted_total_loss"] = total
        components["total_loss"] = total
        return total, components

    def _checkpoint(self, kind, validation_metrics=None):
        checkpoint = super()._checkpoint(kind, validation_metrics)
        checkpoint.update(
            loss_scheme="distance_decay_weighted_mtp",
            loss_weights=list(self.loss_weights),
        )
        return checkpoint

    def resume(self, path=None):
        checkpoint = super().resume(path)
        if checkpoint.get("loss_scheme") != "distance_decay_weighted_mtp":
            raise ValueError("Checkpoint is not a weighted MTP experiment.")
        if checkpoint.get("loss_weights") != list(self.loss_weights):
            raise ValueError("Weighted MTP checkpoint loss weights do not match.")
        return checkpoint

    def train(self):
        summary = super().train()
        summary["configuration"].update(
            loss_scheme="distance_decay_weighted_mtp",
            loss_weights=list(self.loss_weights),
            scheduler_total_updates=self.scheduler.total_updates,
        )
        summary["final_results"]["loss_scheme"] = "distance_decay_weighted_mtp"
        _plot_mtp_series(
            self.validation_metrics,
            (
                ("Head1", "head1_bpb", "#1f77b4"),
                ("Head2", "head2_bpb", "#ff7f0e"),
                ("Head3", "head3_bpb", "#2ca02c"),
                ("Head4", "head4_bpb", "#9467bd"),
                ("Four-head mean", "bpb", "#d62728"),
            ),
            self.run_dir / "bpb_heads_vs_steps.png",
            "Weighted MTP-4 validation BPB",
            "Bits per byte",
        )
        _plot_mtp_series(
            self.training_metrics,
            (
                ("loss1", "loss1", "#1f77b4"),
                ("loss2", "loss2", "#ff7f0e"),
                ("loss3", "loss3", "#2ca02c"),
                ("loss4", "loss4", "#9467bd"),
                ("weighted total", "weighted_total_loss", "#d62728"),
            ),
            self.run_dir / "loss_heads_vs_steps.png",
            "Weighted MTP-4 training losses",
            "Cross-entropy (nats/byte)",
        )
        _atomic_json_dump(summary, self.experiment_summary_path)
        return summary


def load_mtp4_checkpoint(path):
    """Strictly load a four-head checkpoint, rejecting baseline weights."""
    checkpoint = _torch_load(path, map_location="cpu")
    _validate_checkpoint_metadata(
        checkpoint, MTP4_MAMBA3_CONFIG["name"], MTP4_MAMBA3_CONFIG
    )
    model = build_mamba3_mtp4_model()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint

"""Independent 2048-token training blocks, epoch validation and resumable plateaus."""

import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .metrics import autocast_context
from .model import build_mamba3_token_model, parameter_counts
from .tokenization import digest_file, load_json, save_json
from .trainer import (capture_rng_state, restore_rng_state, seed_everything,
                      _atomic_torch_save, _plot_bpb_history)

BLOCK_SIZE = 2048
TRAINING = dict(seed=2026, peak_lr=0.0006, min_lr=0.000006,
                betas=(0.9, 0.99), eps=1e-8, max_grad_norm=5.0,
                threshold_bpb=0.001, reduce_patience=5, stop_patience=15,
                progress_seconds=43200, checkpoint_seconds=900)


class TokenBlockDataset(Dataset):
    def __init__(self, path, block_size=BLOCK_SIZE):
        self.path = str(path)
        self.block_size = int(block_size)
        if Path(path).stat().st_size % 2:
            raise ValueError("Malformed uint16 token cache")
        self.num_tokens = Path(path).stat().st_size // 2
        self.num_blocks = (self.num_tokens + self.block_size - 1) // self.block_size
        self._tokens = None

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._tokens is None:
            self._tokens = np.memmap(self.path, mode="r", dtype="<u2")
        start = index * self.block_size
        valid = min(self.block_size, self.num_tokens - start)
        values = np.zeros(self.block_size, dtype=np.int64)
        values[:valid] = self._tokens[start:start+valid]
        target = np.zeros_like(values)
        target[:max(valid-1, 0)] = values[1:valid]
        mask = np.arange(self.block_size) < valid-1
        return torch.from_numpy(values), torch.from_numpy(target), torch.from_numpy(mask)


def block_loss(logits, targets, mask):
    # Fixed shape avoids the variable-size GPU indexing synchronization.
    loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), reduction="none")
    return (loss * mask.flatten()).sum(), mask.sum()


class PlateauControl:
    def __init__(self, threshold=0.001, reduce_patience=5, stop_patience=15):
        self.threshold = float(threshold)
        self.reduce_patience = int(reduce_patience)
        self.stop_patience = int(stop_patience)
        self.reference = None
        self.best = None
        self.bad_epochs = 0

    def update(self, bpb):
        if not math.isfinite(bpb):
            raise ValueError("Non-finite validation BPB")
        best = self.best is None or bpb < self.best
        if best:
            self.best = bpb
        if self.reference is None or self.reference - bpb >= self.threshold:
            self.reference = bpb
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return {"best": best, "reduce": self.bad_epochs > 0 and self.bad_epochs % self.reduce_patience == 0,
                "stop": self.bad_epochs >= self.stop_patience}

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, state):
        for key in ("threshold", "reduce_patience", "stop_patience"):
            if state[key] != getattr(self, key):
                raise ValueError("Convergence settings changed on resume")
        self.__dict__.update(state)


@torch.inference_mode()
def validate(model, dataset, raw_bytes, batch_size, device, workers=0):
    model.eval()
    total = torch.zeros((), dtype=torch.float64, device=device)
    targets_seen = 0
    started = time.perf_counter()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    for values, targets, masks in loader:
        targets_seen += int(masks.sum())
        values, targets, masks = [v.to(device, non_blocking=True) for v in (values, targets, masks)]
        with autocast_context(device):
            logits = model(values)
        nll, _ = block_loss(logits, targets, masks)
        total += nll.double()
        del logits, nll
    expected = dataset.num_tokens - dataset.num_blocks
    if targets_seen != expected:
        raise AssertionError("Validation crossed a block boundary")
    model_nll = float(total)
    seed_nll = dataset.num_blocks * math.log(model.embedding.num_embeddings)
    bpb = (model_nll + seed_nll) / (raw_bytes * math.log(2)) if raw_bytes else 0.
    if not math.isfinite(bpb):
        raise FloatingPointError("Non-finite validation result")
    return dict(bpb=bpb, model_nll=model_nll, uniform_seed_nll=seed_nll,
                raw_bytes=raw_bytes, tokens=dataset.num_tokens, predicted_tokens=targets_seen,
                num_blocks=dataset.num_blocks, elapsed_seconds=time.perf_counter()-started)


def optimizer_for(model):
    return torch.optim.Adam(model.parameters(), lr=TRAINING["peak_lr"],
                            betas=TRAINING["betas"], eps=TRAINING["eps"], weight_decay=0.)


def probe_batch(directory, batch_size, device, workers=0):
    """A fresh worker process is used for each candidate, including OOM failures."""
    seed_everything(TRAINING["seed"])
    device = torch.device(device)
    torch.cuda.set_device(device)
    free_before, total = torch.cuda.mem_get_info(device)
    model, _ = build_mamba3_token_model()
    model.to(device).train()
    optimizer = optimizer_for(model)
    dataset = TokenBlockDataset(Path(directory) / "train.tokens")
    loader = iter(DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=True,
                             shuffle=False, drop_last=True))
    times = []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(8):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        values, targets, masks = next(loader)
        values, targets, masks = [v.to(device, non_blocking=True) for v in (values, targets, masks)]
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            logits = model(values)
        nll, count = block_loss(logits, targets, masks)
        loss = nll / count
        if not torch.isfinite(loss):
            raise FloatingPointError("Probe loss is not finite")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["max_grad_norm"])
        if not torch.isfinite(grad):
            raise FloatingPointError("Probe gradient is not finite")
        optimizer.step()
        del logits, nll, loss, count
        torch.cuda.synchronize(device)
        if step >= 3:
            times.append(time.perf_counter()-started)
    # Include real validation allocations, while keeping the probe small.
    validation = TokenBlockDataset(Path(directory) / "validation.tokens")
    values, targets, masks = next(iter(DataLoader(validation, batch_size=batch_size, pin_memory=True)))
    with torch.no_grad(), autocast_context(device):
        model.eval()
        logits = model(values.to(device))
        val, _ = block_loss(logits, targets.to(device), masks.to(device))
        if not torch.isfinite(val):
            raise FloatingPointError("Probe validation is not finite")
    torch.cuda.synchronize(device)
    reserved = torch.cuda.max_memory_reserved(device)
    allocated = torch.cuda.max_memory_allocated(device)
    median = float(np.median(times))
    return dict(batch_size=batch_size, workers=workers, step_seconds=times,
                median_step_seconds=median, tokens_per_second=batch_size * 2047 / median,
                peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                free_before_bytes=free_before, device_bytes=total,
                safe=reserved <= free_before - 1024**3)


def train(directory, batch_size, device="cuda:0", workers=4):
    directory = Path(directory)
    training_dir = directory / "training"
    training_dir.mkdir(exist_ok=True)
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    seed_everything(TRAINING["seed"])
    train_data = TokenBlockDataset(directory / "train.tokens")
    validation_data = TokenBlockDataset(directory / "validation.tokens")
    raw_validation_bytes = load_json(str(directory / "validation.tokens") + ".json")["raw_bytes"]
    fingerprint = {name: digest_file(directory / name) for name in
                   ("tokenizer.model", "train.tokens", "validation.tokens")}
    model, model_config = build_mamba3_token_model()
    model.to(device)
    optimizer = optimizer_for(model)
    control = PlateauControl(TRAINING["threshold_bpb"], TRAINING["reduce_patience"], TRAINING["stop_patience"])
    epoch = position = updates = tokens_seen = 0
    elapsed_before = 0.
    history = []
    train_history = []
    latest = training_dir / "latest.pt"
    best_path = training_dir / "best.pt"
    completed = False
    if latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        if (checkpoint["fingerprint"] != fingerprint or checkpoint["batch_size"] != batch_size
                or checkpoint["training_config"] != TRAINING or checkpoint["model_config"] != model_config):
            raise ValueError("Training inputs/configuration changed on resume")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        control.load_state_dict(checkpoint["control"])
        epoch, position = checkpoint["epoch"], checkpoint["position"]
        updates, tokens_seen = checkpoint["updates"], checkpoint["tokens_seen"]
        elapsed_before = checkpoint["elapsed_seconds"]
        history, train_history = checkpoint["history"], checkpoint["train_history"]
        completed = checkpoint.get("converged", False)
        restore_rng_state(checkpoint["rng_state"])
    started = time.perf_counter()
    last_save = last_progress = started
    stopping = [False]
    old_handlers = {sig: signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
                    for sig in (signal.SIGTERM, signal.SIGINT)}

    def elapsed():
        return elapsed_before + time.perf_counter()-started

    def checkpoint(converged=False):
        _atomic_torch_save(dict(
            schema_version=1, model=model.state_dict(), model_config=model_config, vocab_size=4096,
            tokenizer_sha256=fingerprint["tokenizer.model"], fingerprint=fingerprint,
            optimizer=optimizer.state_dict(), control=control.state_dict(), epoch=epoch, position=position,
            updates=updates, tokens_seen=tokens_seen, elapsed_seconds=elapsed(),
            rng_state=capture_rng_state(), training_config=TRAINING, batch_size=batch_size,
            workers=workers, history=history, train_history=train_history, converged=converged), latest)

    def status(converged=False):
        result = dict(converged=converged, epoch=epoch, position=position, updates=updates,
                      predicted_tokens_seen=tokens_seen, elapsed_seconds=elapsed(), batch_size=batch_size,
                      workers=workers, best_bpb=control.best, bad_epochs=control.bad_epochs,
                      learning_rate=optimizer.param_groups[0]["lr"], parameter_counts=parameter_counts(model),
                      fingerprint=fingerprint, training_config=TRAINING,
                      stop_reason="validation_plateau" if converged else "running_or_interrupted")
        save_json(training_dir / "summary.json", result)
        save_json(training_dir / "history.json", {"validation": history, "training": train_history})
        if history:
            _plot_bpb_history([], history, training_dir / "bpb_vs_steps.png")
        return result

    try:
        while not completed and not stopping[0]:
            order = np.random.default_rng(TRAINING["seed"] + epoch).permutation(len(train_data))
            loader = DataLoader(Subset(train_data, order[position:].tolist()), batch_size=batch_size,
                                num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
            model.train()
            for values, targets, masks in loader:
                valid_count = int(masks.sum())
                if epoch == 0:
                    lr = TRAINING["peak_lr"] * min(1., (position + len(values)) / len(train_data))
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                if valid_count:
                    values, targets, masks = [v.to(device, non_blocking=True) for v in (values, targets, masks)]
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(device):
                        logits = model(values)
                    nll, count = block_loss(logits, targets, masks)
                    loss = nll / count
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss")
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["max_grad_norm"])
                    if not torch.isfinite(grad):
                        raise FloatingPointError("Non-finite training gradient")
                    optimizer.step()
                    updates += 1
                    tokens_seen += valid_count
                    if updates == 1 or updates % 100 == 0:
                        record = dict(update=updates, epoch=epoch+1, nll_per_token=float(loss),
                                      predicted_tokens_seen=tokens_seen, elapsed_seconds=elapsed())
                        train_history.append(record)
                        print(record, flush=True)
                    del logits, nll, loss, count
                position += len(values)
                now = time.perf_counter()
                if now-last_save >= TRAINING["checkpoint_seconds"] or stopping[0]:
                    checkpoint()
                    status()
                    last_save = now
                if now-last_progress >= TRAINING["progress_seconds"]:
                    print({"progress_12h": status()}, flush=True)
                    last_progress = now
                if stopping[0]:
                    break
            if stopping[0]:
                break
            metrics = validate(model, validation_data, raw_validation_bytes, batch_size, device, workers)
            metrics.update(epoch=epoch+1, global_update=updates, elapsed_training_seconds=elapsed())
            history.append(metrics)
            decision = control.update(metrics["bpb"])
            if decision["best"]:
                _atomic_torch_save(dict(model=model.state_dict(), model_config=model_config,
                    vocab_size=4096, tokenizer_sha256=fingerprint["tokenizer.model"],
                    epoch=epoch+1, updates=updates, validation=metrics), best_path)
            if decision["reduce"]:
                for group in optimizer.param_groups:
                    group["lr"] = max(TRAINING["min_lr"], group["lr"] * 0.5)
            epoch += 1
            position = 0
            completed = decision["stop"]
            checkpoint(completed)
            print({"validation": metrics, "control": control.state_dict()}, flush=True)
            status(completed)
            last_save = time.perf_counter()
        checkpoint(completed)
        return status(completed)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)

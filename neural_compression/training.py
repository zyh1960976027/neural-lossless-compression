"""BPE 主线训练、梯度累积、早停和可恢复检查点。"""
import fcntl
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .config import TRAINING
from .model import autocast_context, build_model, parameter_counts
from .data import (
    TokenBlockDataset,
    LosslessTokenizer,
    prepare_training,
    require_baseline_tokenizer,
)
from .evaluation import block_loss, validate
from .io import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    _atomic_torch_save,
    _plot_bpb_history,
    sha256_file as digest_file,
    load_json,
    save_json,
    git_info,
)


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
        return {
            "best": best,
            "reduce": self.bad_epochs > 0
            and self.bad_epochs % self.reduce_patience == 0,
            "stop": self.bad_epochs >= self.stop_patience,
        }

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, state):
        for key in ("threshold", "reduce_patience", "stop_patience"):
            if state[key] != getattr(self, key):
                raise ValueError("Convergence settings changed on resume")
        self.__dict__.update(state)


def optimizer_for(model):
    return torch.optim.Adam(
        model.parameters(),
        lr=TRAINING["peak_lr"],
        betas=TRAINING["betas"],
        eps=TRAINING["eps"],
        weight_decay=0.0,
    )


def accumulated_update(model, optimizer, batch, micro_batch_size, device):
    """One optimizer update, normalized by all valid targets including a tail."""
    values, targets, masks = batch
    if micro_batch_size < 1:
        raise ValueError("micro_batch_size must be positive")
    valid_count = int(masks.sum())
    optimizer.zero_grad(set_to_none=True)
    if not valid_count:
        return 0.0, 0
    total_nll = torch.zeros((), device=device)
    for start in range(0, len(values), micro_batch_size):
        stop = start + micro_batch_size
        x, y, mask = [v[start:stop].to(device, non_blocking=True) for v in batch]
        if not bool(mask.any()):
            continue
        with autocast_context(device):
            logits = model(x)
        nll, _ = block_loss(logits, y, mask)
        loss = nll / valid_count
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        total_nll += nll.detach()
        del logits, nll, loss
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["max_grad_norm"])
    if not torch.isfinite(grad):
        raise FloatingPointError("Non-finite training gradient")
    optimizer.step()
    return float(total_nll / valid_count), valid_count


def _train(
    directory,
    batch_size,
    device="cuda:0",
    workers=0,
    micro_batch_size=None,
    max_epochs=None,
):
    directory = Path(directory)
    micro_batch_size = batch_size if micro_batch_size is None else micro_batch_size
    if micro_batch_size < 1 or batch_size % micro_batch_size:
        raise ValueError("Micro batch must be a positive divisor of effective batch")
    vocab_size = LosslessTokenizer(directory / "tokenizer.model").vocab_size
    training_dir = directory / "checkpoints"
    training_dir.mkdir(exist_ok=True)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    seed_everything(TRAINING["seed"])
    train_data = TokenBlockDataset(directory / "data/train.tokens")
    validation_data = TokenBlockDataset(directory / "data/validation.tokens")
    raw_validation_bytes = load_json(
        str(directory / "data/validation.tokens") + ".json"
    )["raw_bytes"]
    fingerprint = {
        name: digest_file(directory / name)
        for name in (
            "tokenizer.model",
            "data/train.tokens",
            "data/validation.tokens",
            "config.json",
            "prepared.json",
        )
    }
    model, model_config = build_model()
    model.to(device)
    optimizer = optimizer_for(model)
    control = PlateauControl(
        TRAINING["threshold_bpb"],
        TRAINING["reduce_patience"],
        TRAINING["stop_patience"],
    )
    epoch = position = updates = tokens_seen = 0
    elapsed_before = 0.0
    history = []
    train_history = []
    latest = training_dir / "latest.pt"
    best_path = training_dir / "best.pt"
    completed = False
    if latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        if (
            checkpoint["fingerprint"] != fingerprint
            or checkpoint["batch_size"] != batch_size
            or checkpoint.get("micro_batch_size", batch_size) != micro_batch_size
            or checkpoint["training_config"] != TRAINING
            or checkpoint["model_config"] != model_config
        ):
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
    old_handlers = {
        sig: signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
        for sig in (signal.SIGTERM, signal.SIGINT)
    }

    def elapsed():
        return elapsed_before + time.perf_counter() - started

    def checkpoint(converged=False):
        _atomic_torch_save(
            dict(
                schema_version=2,
                model=model.state_dict(),
                model_config=model_config,
                vocab_size=vocab_size,
                tokenizer_sha256=fingerprint["tokenizer.model"],
                fingerprint=fingerprint,
                optimizer=optimizer.state_dict(),
                control=control.state_dict(),
                epoch=epoch,
                position=position,
                updates=updates,
                tokens_seen=tokens_seen,
                elapsed_seconds=elapsed(),
                rng_state=capture_rng_state(),
                training_config=TRAINING,
                batch_size=batch_size,
                micro_batch_size=micro_batch_size,
                effective_batch_size=batch_size,
                workers=workers,
                history=history,
                train_history=train_history,
                converged=converged,
            ),
            latest,
        )

    def status(converged=False):
        result = dict(
            converged=converged,
            epoch=epoch,
            position=position,
            updates=updates,
            predicted_tokens_seen=tokens_seen,
            elapsed_seconds=elapsed(),
            batch_size=batch_size,
            workers=workers,
            best_bpb=control.best,
            bad_epochs=control.bad_epochs,
            vocab_size=vocab_size,
            micro_batch_size=micro_batch_size,
            effective_batch_size=batch_size,
            learning_rate=optimizer.param_groups[0]["lr"],
            parameter_counts=parameter_counts(model),
            fingerprint=fingerprint,
            training_config=TRAINING,
            stop_reason="validation_plateau"
            if converged
            else (
                "epoch_limit"
                if max_epochs is not None and epoch >= max_epochs
                else "running_or_interrupted"
            ),
        )
        save_json(directory / "summary.json", result)
        save_json(
            directory / "history.json",
            {"validation": history, "training": train_history},
        )
        if history:
            _plot_bpb_history([], history, directory / "bpb_vs_steps.png")
        return result

    try:
        while (
            not completed
            and not stopping[0]
            and (max_epochs is None or epoch < max_epochs)
        ):
            order = np.random.default_rng(TRAINING["seed"] + epoch).permutation(
                len(train_data)
            )
            loader = DataLoader(
                Subset(train_data, order[position:].tolist()),
                batch_size=batch_size,
                num_workers=workers,
                pin_memory=device.type == "cuda",
                persistent_workers=workers > 0,
            )
            model.train()
            for values, targets, masks in loader:
                valid_count = int(masks.sum())
                if epoch == 0:
                    lr = TRAINING["peak_lr"] * min(
                        1.0, (position + len(values)) / len(train_data)
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                if valid_count:
                    loss, valid_count = accumulated_update(
                        model,
                        optimizer,
                        (values, targets, masks),
                        micro_batch_size,
                        device,
                    )
                    updates += 1
                    tokens_seen += valid_count
                    if updates == 1 or updates % 100 == 0:
                        record = dict(
                            update=updates,
                            epoch=epoch + 1,
                            nll_per_token=float(loss),
                            predicted_tokens_seen=tokens_seen,
                            elapsed_seconds=elapsed(),
                        )
                        train_history.append(record)
                        print(record, flush=True)
                position += len(values)
                now = time.perf_counter()
                if now - last_save >= TRAINING["checkpoint_seconds"] or stopping[0]:
                    checkpoint()
                    status()
                    last_save = now
                if now - last_progress >= TRAINING["progress_seconds"]:
                    print({"progress_12h": status()}, flush=True)
                    last_progress = now
                if stopping[0]:
                    break
            if stopping[0]:
                break
            metrics = validate(
                model,
                validation_data,
                raw_validation_bytes,
                micro_batch_size,
                device,
                workers,
            )
            metrics.update(
                epoch=epoch + 1,
                global_update=updates,
                elapsed_training_seconds=elapsed(),
            )
            history.append(metrics)
            decision = control.update(metrics["bpb"])
            if decision["best"]:
                _atomic_torch_save(
                    dict(
                        model=model.state_dict(),
                        model_config=model_config,
                        vocab_size=vocab_size,
                        tokenizer_sha256=fingerprint["tokenizer.model"],
                        epoch=epoch + 1,
                        updates=updates,
                        validation=metrics,
                    ),
                    best_path,
                )
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
        result = status(completed)
        source = best_path if best_path.exists() else latest
        result["inference"] = export_inference(source, directory / "inference.pt")
        save_json(directory / "summary.json", result)
        return result
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def export_inference(source, target):
    state = torch.load(source, map_location="cpu", weights_only=False)
    package = {
        key: state[key]
        for key in ("model", "model_config", "vocab_size", "tokenizer_sha256")
    }
    package.update(
        source_checkpoint_sha256=digest_file(source),
        validation=state.get("validation"),
        schema_version=1,
    )
    _atomic_torch_save(package, target)
    return dict(
        path=str(Path(target).resolve()),
        bytes=Path(target).stat().st_size,
        sha256=digest_file(target),
        source_checkpoint_sha256=package["source_checkpoint_sha256"],
    )


def preflight(directory, batch_size, micro_batch_size, device):
    directory = Path(directory)
    identity = dict(
        config=digest_file(directory / "config.json"),
        prepared=digest_file(directory / "prepared.json"),
    )
    path = directory / "preflight.json"
    if path.exists():
        if load_json(path)["identity"] != identity:
            raise ValueError("训练预检配置发生变化")
        return load_json(path)
    device = torch.device(device)
    seed_everything(TRAINING["seed"])
    model, _ = build_model()
    model.to(device).train()
    optimizer = optimizer_for(model)
    data = TokenBlockDataset(directory / "data/train.tokens")
    batch = next(iter(DataLoader(data, batch_size=batch_size, num_workers=0)))
    started = time.perf_counter()
    loss, count = accumulated_update(model, optimizer, batch, micro_batch_size, device)
    validation = TokenBlockDataset(directory / "data/validation.tokens")
    values, targets, mask = next(
        iter(DataLoader(validation, batch_size=micro_batch_size))
    )
    with torch.inference_mode(), autocast_context(device):
        model.eval()
        nll, _ = block_loss(
            model(values.to(device)), targets.to(device), mask.to(device)
        )
    if not torch.isfinite(nll) or not count:
        raise FloatingPointError("训练或验证预检失败")
    result = dict(
        identity=identity,
        passed=True,
        train_nll_per_token=loss,
        predicted_tokens=count,
        validation_nll=float(nll),
        elapsed_seconds=time.perf_counter() - started,
    )
    save_json(path, result)
    return result


def train(
    run_dir,
    *,
    source=None,
    train_file=None,
    validation_file=None,
    tokenizer=None,
    batch_size=72,
    micro_batch_size=None,
    device="cuda:0",
    workers=0,
    tokenizer_threads=24,
    max_epochs=None
):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if max_epochs is not None and max_epochs < 1:
        raise ValueError("最大轮数必须为正整数")
    micro_batch_size = batch_size if micro_batch_size is None else micro_batch_size
    directory = Path(run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("该实验目录已有训练进程运行") from error
        prepare_training(
            directory,
            source=source,
            train_file=train_file,
            validation_file=validation_file,
            tokenizer=tokenizer,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            device=device,
            workers=workers,
            tokenizer_threads=tokenizer_threads,
        )
        save_json(directory / "provenance.json", dict(git=git_info()))
        preflight(directory, batch_size, micro_batch_size, device)
        return _train(
            directory, batch_size, device, workers, micro_batch_size, max_epochs
        )

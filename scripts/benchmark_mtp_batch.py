"""Short isolated MTP-4 batch sweep; writes metrics only, never checkpoints."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import torch
    from torch.utils.data import DataLoader
    from mamba3_byte_compress.config import make_training_config
    from mamba3_byte_compress.data import ByteTrainingDataset
    from mamba3_byte_compress.metrics import autocast_context
    from mamba3_byte_compress.model import build_mamba3_mtp4_model
    from mamba3_byte_compress.mtp import MTP4ByteTrainer
    from mamba3_byte_compress.trainer import WarmupCosineScheduler, seed_everything

    b = args.worker_batch
    config = make_training_config(global_batch_size=b, micro_batch_size=b)
    result = dict(batch_size=b, micro_batch_size=b, accumulation_steps=1,
                  config=config, status='error', steps=[], initialization='same seed, fresh model and Adam')
    samples = []
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                raw = subprocess.check_output(['nvidia-smi', '--id=0',
                    '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True)
                util, memory = map(float, raw.strip().split(','))
                samples.append((time.perf_counter(), util, memory))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            stop.wait(0.5)
    thread = threading.Thread(target=monitor, daemon=True)
    measured_start = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        device = torch.device('cuda:0')
        result['gpu'] = torch.cuda.get_device_name(device)
        seed_everything(config['seed'])
        model = build_mamba3_mtp4_model().to(device).train()
        result['parameter_count'] = sum(p.numel() for p in model.parameters())
        assert result['parameter_count'] == 3725772
        optimizer = torch.optim.Adam(model.parameters(), lr=config['peak_lr'],
            betas=config['betas'], eps=config['eps'], weight_decay=config['weight_decay'])
        scheduler = WarmupCosineScheduler(optimizer, config['peak_lr'], config['total_updates'],
            config['warmup_updates'], config['min_lr_ratio'])
        dataset = ByteTrainingDataset(ROOT / args.train_file, ROOT / args.train_starts,
            config['sequence_length'], sample_count=(args.warmup + args.steps) * b)
        loader = DataLoader(dataset, batch_size=b, shuffle=False, drop_last=True,
            num_workers=4, pin_memory=True, persistent_workers=True)
        iterator = iter(loader)
        thread.start()
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(args.warmup + args.steps):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            if step == args.warmup:
                measured_start = started
            lr = scheduler.step(step)
            optimizer.zero_grad(set_to_none=True)
            batch = next(iterator)
            inputs = batch['input_tokens'].to(device, non_blocking=True)
            targets = batch['target_tokens'].to(device, non_blocking=True)
            with autocast_context(device, enabled=True):
                logits = model(inputs)
            loss, components = MTP4ByteTrainer._batch_loss(None, logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss')
            loss.backward()
            values = {name: float(value.detach()) for name, value in components.items()}
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            if not torch.isfinite(norm):
                raise FloatingPointError('Non-finite gradient')
            optimizer.step()
            torch.cuda.synchronize(device)
            result['steps'].append(dict(step=step + 1, seconds=time.perf_counter()-started,
                lr=lr, **values))
        measured_end = time.perf_counter()
        measured = result['steps'][args.warmup:]
        result.update(status='success', mean_step_seconds=statistics.mean(x['seconds'] for x in measured),
            first10_loss=statistics.mean(x['total_loss'] for x in result['steps'][:10]),
            last10_loss=statistics.mean(x['total_loss'] for x in result['steps'][-10:]))
        result['bytes_per_second'] = b * config['sequence_length'] / result['mean_step_seconds']
        active = [x for x in samples if measured_start <= x[0] <= measured_end]
        result['gpu_utilization_mean_percent'] = statistics.mean(x[1] for x in active) if active else None
        result['gpu_monitor_samples'] = len(active)
    except torch.OutOfMemoryError as exc:
        result.update(status='oom', error=str(exc))
    except Exception as exc:
        # Triton kernels can report CUDA OOM as a plain RuntimeError.
        status = 'oom' if isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower() else 'error'
        result.update(status=status, error=repr(exc))
    finally:
        stop.set()
        if thread.is_alive():
            thread.join(timeout=3)
        if torch.cuda.is_initialized():
            result['peak_allocated_gb'] = torch.cuda.max_memory_allocated() / 1e9
            result['peak_reserved_gb'] = torch.cuda.max_memory_reserved() / 1e9
        result['device_peak_used_mib_sampled'] = max((x[2] for x in samples), default=None)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('steps','config')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', type=int, nargs='+', default=[32, 64, 128, 256])
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--train-file', default='datasets/processed/enwik9_train_100m.bin')
    parser.add_argument('--train-starts', default='datasets/processed/train_windows.npy')
    parser.add_argument('--output', default=None)
    parser.add_argument('--worker-batch', type=int)
    args = parser.parse_args()
    if args.steps < 10 or args.warmup < 0:
        parser.error('steps must be >=10 and warmup >=0')
    if args.worker_batch:
        return worker(args)
    output = Path(args.output or ROOT / 'experiments/test' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_mtp4_batch'))
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for batch in args.batches:
        path = output / f'batch_{batch}.json'
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker-batch', str(batch),
            '--steps', str(args.steps), '--warmup', str(args.warmup), '--output', str(path),
            '--train-file', args.train_file, '--train-starts', args.train_starts], check=True, cwd=ROOT)
        results.append(json.loads(path.read_text()))
        (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'Results: {output}', flush=True)


if __name__ == '__main__':
    main()

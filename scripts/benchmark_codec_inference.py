"""Compare v4, fixed-buffer eager v5 and CUDA Graph v5 on the same byte range."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LABELS = {'baseline': '原始 v4', 'buffered': '缓冲区复用', 'graph': '缓冲区复用与图重放'}
OPERATIONS = {'encode': '编码', 'decode': '解码'}
MODES = {'baseline': ('v4', 'graph'), 'buffered': ('v5', 'buffered'), 'graph': ('v5', 'graph')}


def run_case(args, mode, index, cold=False):
    name = f'{mode}_{"cold" if cold else index}'
    output = args.output / name
    result_path = output / 'results.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        expected = dict(source_offset=args.offset_bytes, sample_bytes=args.sample_bytes,
                        file_format=MODES[mode][0])
        if any(result['verification'][k] != v for k, v in expected.items()):
            raise ValueError(f'Cannot resume different settings in {output}')
        if result['encode']['batch_size'] != args.batch_size or result['encode']['execution'] != MODES[mode][1]:
            raise ValueError(f'Cannot resume different inference in {output}')
        from mamba3_byte_compress.codec import sha256_file
        if result['encode']['checkpoint_sha256'] != sha256_file(args.checkpoint):
            raise ValueError('Cannot resume a different checkpoint')
        return result
    command = [sys.executable, str(ROOT / 'scripts/benchmark_codec_batched.py'),
               '--output', str(output), '--source', str(args.source),
               '--source-offset', str(args.offset_bytes), '--sample-bytes', str(args.sample_bytes),
               '--batch-size', str(args.batch_size), '--checkpoint', str(args.checkpoint),
               '--format', MODES[mode][0], '--execution', MODES[mode][1]]
    if cold:
        command.append('--cold')
    with (args.output / f'{name}.log').open('x') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'{name} complete', flush=True)
    return json.loads(result_path.read_text())


def model_measurement(args):
    import numpy as np
    import torch
    from mamba3_byte_compress import codec as c
    torch.set_num_threads(1)
    result = {}
    blocks = (args.sample_bytes + 2047) // 2048
    shapes = sorted({min(args.batch_size, blocks), blocks % args.batch_size} - {0})
    for mode in MODES:
        p = c.load_predictor(args.checkpoint, inference=f'batched-step-{MODES[mode][0]}', batch_size=args.batch_size)
        if mode == 'buffered':
            p.use_graph = False
        measurements = {}
        with c._coding_mode(p):
            for size in shapes:
                p.reset(size)
                for _ in range(3):
                    p.step(np.zeros(size, dtype=np.uint8))
                samples = []
                for _ in range(3):
                    p.reset(size)
                    p.synchronize()
                    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record()
                    for _ in range(64):
                        if mode == 'graph':
                            p._graph.replay()
                        else:
                            p._compute()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) / 64)
                measurements[size] = dict(milliseconds_per_step=statistics.median(samples), samples=samples)
        result[mode] = measurements
        del p
        torch.cuda.empty_cache()
    return result


def overlap(runs):
    for operation in ('encode', 'decode'):
        field = f'{operation}_seconds'
        baseline = [r[operation][field] for r in runs['baseline']]
        graph = [r[operation][field] for r in runs['graph']]
        if max(min(baseline), min(graph)) <= min(max(baseline), max(graph)):
            return True
    return False


def report(args, runs, cold, model):
    summary = {}
    rows = []
    for mode, values in runs.items():
        summary[mode] = {}
        hashes = {r['verification']['compressed_sha256'] for r in values}
        if len(hashes) != 1:
            raise AssertionError(f'{mode}: repeated encodes differ')
        for r in values:
            if not all(r['verification'][k] for k in ('files_identical', 'sha256_matches', 'independent_processes')):
                raise AssertionError(f'{mode}: roundtrip verification failed')
        for operation in ('encode', 'decode'):
            times = [r[operation][f'{operation}_seconds'] for r in values]
            row = dict(mode=mode, operation=operation, repeats=len(times), median_seconds=statistics.median(times),
                       min_seconds=min(times), max_seconds=max(times),
                       throughput_kb_s=args.sample_bytes / 1000 / statistics.median(times),
                       process_seconds=statistics.median(r[operation]['process_seconds'] for r in values),
                       warmup_seconds=statistics.median(r[operation]['warmup_seconds'] for r in values),
                       peak_allocated_mib=max(r[operation]['run_peak_allocated_bytes'] for r in values) / 2**20,
                       peak_reserved_mib=max(r[operation]['run_peak_reserved_bytes'] for r in values) / 2**20,
                       actual_bpb=values[0][operation]['actual_bpb'])
            if operation == 'encode':
                row['theoretical_nll_bpb'] = values[0][operation]['theoretical_nll_bpb']
            summary[mode][operation] = row
            rows.append(row)
    accepted = all(summary['graph'][op]['max_seconds'] < summary['baseline'][op]['min_seconds']
                   for op in ('encode', 'decode'))
    eager_graph_same = (runs['buffered'][0]['verification']['compressed_sha256'] ==
                        runs['graph'][0]['verification']['compressed_sha256'])
    if not eager_graph_same:
        raise AssertionError('Buffered eager and graph bitstreams differ')
    result = dict(summary=summary, stable_speedup=accepted, eager_graph_bitstream_identical=eager_graph_same,
                  model_timing=model, cold=cold,
                  settings=dict(source=str(args.source), offset_bytes=args.offset_bytes,
                                sample_bytes=args.sample_bytes, batch_size=args.batch_size), runs=runs)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    with (args.output / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    runtime = runs['baseline'][0]['encode']['runtime']
    lines = ['# 模型推理执行优化实验', '',
             f"设备：{runtime['gpu']}；PyTorch {runtime['torch']}；CUDA {runtime['cuda']}；权重：{args.checkpoint.name}。", '',
             f'数据区间：enwik9 [{args.offset_bytes:,}, {args.offset_bytes + args.sample_bytes:,})；'
             f'批大小 {args.batch_size}，独立块 2048 字节，CPU 单线程。', '',
             '采用固定缓冲区复用及 CUDA Graph。后者预先记录 GPU 运算并逐步重放；模型结构、权重和状态更新内核保持不变。', '',
             '## 无插桩端到端结果', '',
             '| 路径 | 操作 | 中位时间 s | 范围 s | 吞吐 KB/s | 完整进程 s | 峰值已分配 MiB |',
             '| --- | --- | ---: | --- | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {LABELS[row['mode']]} | {OPERATIONS[row['operation']]} | {row['median_seconds']:.3f} | "
                     f"{row['min_seconds']:.3f}–{row['max_seconds']:.3f} | {row['throughput_kb_s']:.2f} | "
                     f"{row['process_seconds']:.3f} | {row['peak_allocated_mib']:.1f} |")
    lines += ['', '端到端时间不含模型加载和提前准备；完整进程时间包含导入、加载、准备和结果写入。'
              '显存为 PyTorch 管理的峰值，包含模型和图缓存，不含其他进程和 CUDA 上下文。', '']
    for op in ('encode', 'decode'):
        ratio = summary['baseline'][op]['median_seconds'] / summary['graph'][op]['median_seconds']
        lines.append(f'- {OPERATIONS[op]}加速比：{ratio:.3f} 倍。')
    lines += [f'- 稳定收益验收：{"通过" if accepted else "未通过"}；要求两端优化后最大耗时低于基线最小耗时。',
              '- 所有独立进程解码、逐字节比较和 SHA256 校验通过；各路径重复编码一致。',
              '- 缓冲区复用与图重放的完整 v5 码流一致。', '', '## 压缩率', '']
    for mode in MODES:
        row = summary[mode]['encode']
        lines.append(f"- {LABELS[mode]}：实际 {row['actual_bpb']:.8f} bit/byte，理论 {row['theoretical_nll_bpb']:.8f} bit/byte。")
    lines += ['', '## GPU 推理区间计时', '',
              '每种实际批大小预热后测量连续 64 步，重复三次取中位数。CUDA Event 包围模型计算与 GPU 概率生成，'
              '不含输入上传、概率下载和 CPU 区间编码；该区间也包含 GPU 等待主机提交运算的空隙，不能解释为内核纯计算时间。'
              '这里连续提交运算，而正式编解码每步还需回传概率并处理字节，因此两种计时不能直接相减来估算其他组件耗时。', '',
              '| 路径 | 批大小 | 每步毫秒 |', '| --- | ---: | ---: |']
    for mode, shapes in model.items():
        for size, values in shapes.items():
            lines.append(f"| {LABELS[mode]} | {size} | {values['milliseconds_per_step']:.4f} |")
    lines += ['', '## 首次运行及准备成本', '']
    for mode, value in cold.items():
        for op in ('encode', 'decode'):
            r = value[op]
            lines.append(f"- {LABELS[mode]}{OPERATIONS[op]}：首次编解码 {r[f'{op}_seconds']:.3f} s，完整进程 {r['process_seconds']:.3f} s，"
                         f"图准备 {r.get('graph_prepare_seconds', 0):.3f} s。")
    lines += ['', '首次运行在新进程中跳过预热，编解码时间包含实际发生的内核准备及图捕获；磁盘上的编译缓存仍保留。',
              '', '## 产物', '', '各运行子目录保存编码、解码日志与指标、原文、码流、恢复文件及源码指纹；'
              '`results.json` 保存原始结果，`comparison.csv` 保存端到端对照。', '']
    lines += ['## 收益来源与后续瓶颈', '',
              '仅复用缓冲区的收益较小，主要改善来自图重放减少每一步的重复调度。'
              'Mamba 循环状态更新仍使用原内核；本轮没有改变模型结构、权重或计算精度。'
              '进一步优化应重新跟踪剩余模型计算，重点评估状态更新内核，本轮不据此承诺额外加速幅度。', '']
    (args.output / '实验报告.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(stable_speedup=accepted, summary=summary), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT / 'datasets/raw/enwik9')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/best-3M.pt')
    parser.add_argument('--offset-bytes', type=int, default=100_000_000)
    parser.add_argument('--sample-bytes', type=int, default=10_000_000)
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {mode: [] for mode in MODES}
    for i in range(1, 4):
        for mode in MODES:
            runs[mode].append(run_case(args, mode, i))
    if overlap(runs):
        for i in range(4, 6):
            for mode in MODES:
                runs[mode].append(run_case(args, mode, i))
    cold = {mode: run_case(args, mode, 0, cold=True) for mode in ('baseline', 'graph')}
    model = model_measurement(args)
    report(args, runs, cold, model)


if __name__ == '__main__':
    main()

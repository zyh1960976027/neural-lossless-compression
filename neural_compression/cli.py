"""Command-line entry for the project's token-based lossless codec."""

import argparse
import json
from pathlib import Path
import sys

import torch

from .io import validate_batch_size, verify_files
from .config import DEFAULT_CHECKPOINT, DEFAULT_TOKENIZER
from .codec import decode, encode, load_decoder, load_predictor
from .data import LosslessTokenizer, require_baseline_tokenizer


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="神经网络无损压缩；默认使用 BPE 4096 与 GPU 概率整数量化。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("encode", "压缩文件"),
        ("decode", "解压文件"),
        ("roundtrip", "压缩、解压并逐字节比较"),
    ):
        command = commands.add_parser(
            name,
            help=help_text,
            description=help_text,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        command.add_argument("--input", "-i", type=Path, required=True, help="输入文件")
        if name == "roundtrip":
            command.add_argument(
                "--compressed", type=Path, required=True, help="压缩文件输出路径"
            )
            command.add_argument("--decoded", type=Path, required=True, help="恢复文件输出路径")
        else:
            command.add_argument(
                "--output", "-o", type=Path, required=True, help="输出文件"
            )
        command.add_argument(
            "--checkpoint",
            "-c",
            type=Path,
            default=DEFAULT_CHECKPOINT,
            help="配套的子词模型权重",
        )
        command.add_argument("--device", default="cuda:0", help="推理设备")
        command.add_argument(
            "--cpu-threads", type=positive_int, default=1, help="PyTorch CPU 线程数"
        )
        command.add_argument("--results", type=Path, help="另存 JSON 指标；已有文件不会覆盖")
        if name != "decode":
            command.add_argument(
                "--tokenizer",
                "-t",
                type=Path,
                default=DEFAULT_TOKENIZER,
                help="与权重配套的分词器",
            )
            command.add_argument(
                "--batch-size", type=positive_int, default=1024, help="同时处理的块数"
            )
            command.add_argument(
                "--tokenizer-threads", type=positive_int, default=24, help="分词线程数"
            )
    training = commands.add_parser("train", help="准备 BPE 数据并训练或续训")
    inputs = training.add_mutually_exclusive_group()
    inputs.add_argument("--source", type=Path, help="使用前 100 MB 训练、随后 10 MB 验证的 enwik9")
    inputs.add_argument("--train-file", type=Path, help="完整自定义训练文件，须同时提供验证文件")
    training.add_argument("--validation-file", type=Path, help="完整自定义验证文件")
    training.add_argument(
        "--run-dir", type=Path, required=True, help="保存数据准备和训练产物的独立目录"
    )
    training.add_argument("--tokenizer", type=Path, help="复用已有 BPE 4096 分词器；默认只从训练集学习")
    training.add_argument(
        "--batch-size", type=positive_int, default=72, help="每次更新的有效批大小"
    )
    training.add_argument(
        "--micro-batch-size", type=positive_int, help="微批大小，默认等于有效批大小"
    )
    training.add_argument("--workers", type=int, default=0, help="数据加载进程数")
    training.add_argument("--device", default="cuda:0")
    training.add_argument("--cpu-threads", type=positive_int, default=1)
    training.add_argument("--tokenizer-threads", type=positive_int, default=24)
    training.add_argument(
        "--max-epochs", type=positive_int, help="可选的累计轮数上限，用于短训练；默认按早停收敛"
    )
    evaluation = commands.add_parser("evaluate", help="评估独立分块的模型概率 BPB")
    evaluation.add_argument("--input", "-i", type=Path, required=True)
    evaluation.add_argument("--checkpoint", "-c", type=Path, default=DEFAULT_CHECKPOINT)
    evaluation.add_argument("--tokenizer", "-t", type=Path, default=DEFAULT_TOKENIZER)
    evaluation.add_argument("--batch-size", type=positive_int, default=72)
    evaluation.add_argument("--workers", type=int, default=0)
    evaluation.add_argument("--device", default="cuda:0")
    evaluation.add_argument("--cpu-threads", type=positive_int, default=1)
    evaluation.add_argument("--tokenizer-threads", type=positive_int, default=24)
    evaluation.add_argument("--results", type=Path)
    return parser.parse_args(argv)


def prepare_paths(args):
    sources = [args.input, args.checkpoint]
    if args.command != "decode":
        validate_batch_size(args.batch_size)
        sources.append(args.tokenizer)
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(f"找不到输入、权重或分词器文件：{path}")
    outputs = (
        []
        if args.command == "evaluate"
        else [args.compressed, args.decoded]
        if args.command == "roundtrip"
        else [args.output]
    )
    if args.results is not None:
        outputs.append(args.results)
    resolved = [path.resolve() for path in outputs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("压缩、恢复和报告的输出路径必须不同")
    if set(resolved).intersection(path.resolve() for path in sources):
        raise ValueError("输出路径必须与输入、权重和分词器不同")
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    if any(
        left in right.parents
        for left in resolved
        for right in resolved
        if left != right
    ):
        raise ValueError("一个输出文件不能同时作为另一个输出文件的目录")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    if args.command == "train":
        from .training import train

        result = train(
            args.run_dir,
            source=args.source,
            train_file=args.train_file,
            validation_file=args.validation_file,
            tokenizer=args.tokenizer,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            device=args.device,
            workers=args.workers,
            tokenizer_threads=args.tokenizer_threads,
            max_epochs=args.max_epochs,
        )
        print(
            json.dumps(
                result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            ),
            flush=True,
        )
        return result
    prepare_paths(args)
    if args.command == "evaluate":
        if args.workers < 0:
            raise ValueError("数据加载进程数不能为负")
        from .evaluation import evaluate

        result = evaluate(
            args.input,
            args.checkpoint,
            args.tokenizer,
            batch_size=args.batch_size,
            device=args.device,
            workers=args.workers,
            tokenizer_threads=args.tokenizer_threads,
        )
    elif args.command == "decode":
        predictor = load_decoder(args.checkpoint, args.input, args.device)
        result = decode(args.input, args.output, predictor)
    else:
        tokenizer = require_baseline_tokenizer(
            LosslessTokenizer(args.tokenizer, threads=args.tokenizer_threads)
        )
        predictor = load_predictor(
            args.checkpoint, args.device, batch_size=args.batch_size,
        )
        packed = args.compressed if args.command == "roundtrip" else args.output
        encoded = encode(args.input, packed, predictor, tokenizer)
        if args.command == "roundtrip":
            decoded = decode(packed, args.decoded, predictor)
            result = dict(
                encode=encoded,
                decode=decoded,
                files_identical=verify_files(args.input, args.decoded),
            )
        else:
            result = encoded
    result.update(
        command=args.command,
        input=str(args.input.resolve()),
        checkpoint=str(args.checkpoint.resolve()),
    )
    if args.command == "roundtrip":
        result.update(
            compressed=str(args.compressed.resolve()),
            decoded=str(args.decoded.resolve()),
        )
    elif args.command != "evaluate":
        result["output"] = str(args.output.resolve())
    serialized = json.dumps(
        result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    )
    if args.results is not None:
        with args.results.open("x", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    print(serialized, flush=True)
    return result


def entrypoint():
    try:
        main()
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        FloatingPointError,
    ) as error:
        print(f"运行失败：{error}", file=sys.stderr, flush=True)
        return 1
    return 0

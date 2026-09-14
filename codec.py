"""Encode, decode or verify files with the Mamba-3 range codec."""

import argparse
import json
from pathlib import Path
import sys

from mamba3_byte_compress.codec import (
    decode_file, encode_file, load_predictor, roundtrip_file,
)
from mamba3_byte_compress.experiments import DEFAULT_CHECKPOINT


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("encode", "decode", "roundtrip"):
        command = commands.add_parser(name)
        command.add_argument("--input", required=True)
        if name == "roundtrip":
            command.add_argument("--compressed", required=True)
            command.add_argument("--decoded", required=True)
        else:
            command.add_argument("--output", required=True)
        command.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
        command.add_argument("--device", default="cuda:0")
        if name != "decode":
            command.add_argument("--format", choices=("v1", "v2-framed", "v2"), default="v2",
                                 help="Default v2: compact header and a shared Range stream")
        command.add_argument("--results", help="Optional JSON report (must not exist)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    outputs = ([args.compressed, args.decoded] if args.command == "roundtrip" else [args.output])
    if args.results:
        outputs.append(args.results)
    paths = [Path(path).resolve() for path in outputs]
    if len(set(paths)) != len(paths):
        raise ValueError("Output and report paths must be distinct.")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    predictor = load_predictor(args.checkpoint, args.device)
    if args.command == "roundtrip":
        result = roundtrip_file(args.input, args.compressed, args.decoded, predictor, file_format=args.format)
    else:
        operation = encode_file if args.command == "encode" else decode_file
        kwargs = {"file_format": args.format} if args.command == "encode" else {}
        result = operation(args.input, args.output, predictor, **kwargs)
    result.update(command=args.command, input=str(Path(args.input).resolve()),
                  checkpoint=str(Path(args.checkpoint).resolve()))
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.results:
        with Path(args.results).open("x", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    print(serialized, flush=True)
    keys = ("theoretical_nll_bpb", "actual_bpb", "encode_kb_per_second", "decode_kb_per_second",
            "files_identical", "checksum_verified")
    for key in keys:
        if key in result:
            print(f"{key}: {result[key]}", file=sys.stderr, flush=True)
    return result


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, RuntimeError, FloatingPointError) as error:
        raise SystemExit(f"Codec failed: {error}") from error

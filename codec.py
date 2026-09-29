"""Encode, decode or verify files with the Mamba-3 range codec."""

import argparse
import json
from pathlib import Path
import sys

from mamba3_byte_compress.codec import (
    decode_file, encode_file, load_predictor, read_codec_settings, roundtrip_file,
    validate_batch_size,
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
        command.add_argument(
            "--execution", choices=("auto", "sequence", "graph"),
            help="MTP-4 execution: auto (default), sequence baseline, or CUDA Graph",
        )
        if name != "decode":
            command.add_argument("--inference",
                                 choices=("mtp4-batched-sequence", "batched-step-v5",
                                          "batched-step-v4", "batched-step", "step", "prefix"),
                                 help="Default batched-step-v5: fixed buffers and CUDA Graph inference")
            command.add_argument("--format", choices=("v1", "v2-framed", "v2", "v3", "v4", "v5", "v6"),
                                 help="Default v5; v6 selects MTP-4 continuation")
            command.add_argument("--batch-size", type=int,
                                 help="Batched-step slots: positive uint32, limited by available memory (default 128)")
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
    if args.command == "decode":
        settings = read_codec_settings(args.input)
        if settings["inference"] == "mtp4-batched-sequence":
            settings["execution"] = args.execution or "auto"
        elif args.execution is not None:
            raise ValueError("--execution is only configurable for MTP-4 files.")
    else:
        inferred = {"v1": "step", "v2-framed": "step", "v2": "step",
                    "v3": "batched-step", "v4": "batched-step-v4",
                    "v5": "batched-step-v5", "v6": "mtp4-batched-sequence"}
        inference = args.inference or inferred.get(args.format, "batched-step-v5")
        default_formats = {"batched-step-v5": "v5", "batched-step-v4": "v4", "batched-step": "v3",
                           "mtp4-batched-sequence": "v6", "step": "v2", "prefix": "v2"}
        args.format = args.format or default_formats[inference]
        compatible = ((inference == "batched-step-v5" and args.format == "v5")
                      or (inference == "mtp4-batched-sequence" and args.format == "v6")
                      or (inference == "batched-step-v4" and args.format == "v4")
                      or (inference == "batched-step" and args.format == "v3")
                      or (inference in ("step", "prefix") and args.format in ("v1", "v2-framed", "v2")))
        if not compatible:
            raise ValueError("The selected format and inference mode are incompatible.")
        settings = {"inference": inference}
        if inference in ("batched-step", "batched-step-v4", "batched-step-v5",
                         "mtp4-batched-sequence"):
            settings["batch_size"] = validate_batch_size(128 if args.batch_size is None else args.batch_size)
        elif args.batch_size is not None:
            raise ValueError("--batch-size requires batched-step.")
        if inference == "mtp4-batched-sequence":
            settings["execution"] = args.execution or "auto"
        elif args.execution is not None:
            raise ValueError("--execution is only configurable for MTP-4 inference.")
    predictor = load_predictor(args.checkpoint, args.device, **settings)
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

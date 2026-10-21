# Mamba-3 Lossless Compression

`baseline/byte` maintains the 256-symbol byte-model comparison with the optimized v5 codec at bdd731c. The project mainline is `main`, which uses BPE 4096 and GPU integer quantization. Use the byte checkpoints and commands documented below.

Mamba-3 predicts byte probabilities, and range coding uses those probabilities
to compress files losslessly. The project also compares byte models with
SentencePiece BPE and Unigram models.

Read the [Chinese project guide](项目说明.md) for setup and usage, or the
[directory guide](docs/目录说明.md) for file responsibilities and document links.
Chinese documentation uses Chinese filenames; this conventional README is an
English entry page.

## Setup

Run commands from the project root in a configured Mamba-3/PyTorch environment
with a CUDA GPU supporting BF16 and a trained project checkpoint:

```bash
conda activate mamba3
python -m pip install -r requirements-codec.txt
```

The requirements file contains additional codec dependencies. See the
[environment guide](docs/使用指南/运行环境.md) for the local Conda setup.

## Encode

```bash
python codec.py encode \
  --input /path/to/input.bin \
  --output /path/to/input.m3rc \
  --checkpoint checkpoints/best-3M.pt
```

## Decode

```bash
python codec.py decode \
  --input /path/to/input.m3rc \
  --output /path/to/decoded.bin \
  --checkpoint checkpoints/best-3M.pt
cmp /path/to/input.bin /path/to/decoded.bin
```

Both operations use `codec.py` with different subcommands. Decoding requires
the same checkpoint and a compatible runtime; inference settings are read from
the compressed file. Output parent directories must exist, and existing files
are never overwritten. The default device is `cuda:0`; encoding defaults to
v5 with batch size 128. Omitting `--checkpoint` selects `checkpoints/best.pt`.

Use `python codec.py --help` or the
[codec guide](docs/使用指南/无损编解码.md) for roundtrip verification and more options.
Training and probability evaluation use `train.py` and `evaluate.py`; see the
[training and evaluation guide](docs/使用指南/训练与评估.md).

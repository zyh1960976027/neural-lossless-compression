# Mamba-3 BPE Lossless Compression

The `main` branch uses a lossless SentencePiece **BPE tokenizer with 4096
entries**, a one-dimensional Mamba-3 sequence model, and a range coder. The
current performance baseline quantizes probabilities into integer frequencies
on the **GPU** (`gpu_int`); the pinned Constriction 0.4.2 range-coding core runs
on the CPU. Decoding reproduces the same probabilities and verifies the restored
bytes with checksums and SHA256.

Read the [Chinese project guide](项目说明.md) for setup and usage, or the
[directory guide](docs/目录说明.md) for file responsibilities and document links.
Chinese documentation uses Chinese filenames; this conventional README is an
English entry page.

## Branches

| Branch | Purpose |
| --- | --- |
| `main` | BPE 4096 with GPU integer probability quantization |
| `experiment/unigram` | Unigram 4096 comparison with the optimized token codec |
| `baseline/byte` | Byte model with the optimized v5 codec |
| `experiment/mtp` | Multi-token prediction experiments |
| `experiment/model-scale` | Model depth and parameter-scale experiments |

Previous branch tips are retained as `archive/*` tags. See the
[version-control and collaboration guide](docs/使用指南/版本管理与协作.md)
for the branch history, GitHub setup proposal, and review workflow.

## Setup

Run commands from the project root in a configured Mamba-3/PyTorch environment
with a CUDA GPU supporting BF16 and a trained project checkpoint:

```bash
conda activate mamba3
python -m pip install -r requirements-codec.txt
python scripts/build_integer_range.py
```

The requirements file contains additional codec dependencies. Building the
integer interface requires Linux and Rust/Cargo >= 1.75. Prepare a matching
BPE 4096 tokenizer and inference checkpoint separately; datasets, checkpoints,
and generated experiment files are excluded from Git. See the
[environment guide](docs/使用指南/运行环境.md) for the local Conda setup.

## BPE 4096 encode and decode

Run this example from the project root after replacing the checkpoint,
tokenizer, and file paths:

```python
from mamba3_byte_compress.token_codec import (
    encode, decode, load_token_predictor, load_token_decoder,
)
from mamba3_byte_compress.tokenization import LosslessTokenizer
from mamba3_byte_compress.codec import verify_files

checkpoint = "/path/to/bpe4096/inference.pt"
tokenizer = LosslessTokenizer("/path/to/bpe4096/tokenizer.model", threads=24)
predictor = load_token_predictor(checkpoint, probability_transfer="gpu_int")
encode("input.bin", "compressed.m3rc", predictor, tokenizer)
# In separate processes, only the checkpoint and compressed file are needed.
decoder = load_token_decoder(checkpoint, "compressed.m3rc")
decode("compressed.m3rc", "decoded.bin", decoder)
verify_files("input.bin", "decoded.bin")
```

Decoding reads the batch size and transfer mode from the compressed file, which
also embeds the tokenizer. Use the same checkpoint and a compatible runtime.
Output parent directories must exist, and existing files are never overwritten.
The default device is `cuda:0`; token blocks contain 2048 tokens and the default
batch size is 1024. The Python API retains its FP32 default for compatibility,
so select `gpu_int` explicitly for the main baseline.

See the [BPE guide](docs/使用指南/BPE概率传输.md) and
[GPU integer quantization report](docs/实验报告/BPEGPU整数量化/实验报告.md)
for details. The existing `codec.py`, `train.py`, and `evaluate.py` commands
remain byte-model tools; BPE/Unigram training uses
`scripts/benchmark_sentencepiece.py`, as described in the
[token experiment guide](docs/实验方案/子词与字节压缩对照实验.md).

## Validation

```bash
python -m unittest discover -s tests -v
```

Real GPU checks are opt-in; see the relevant guides and tests for their runtime
requirements. The archived GPU quantization experiment verified all 18
encode/decode pairs byte for byte.

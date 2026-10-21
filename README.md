# Mamba-3 Unigram Lossless Compression

`experiment/unigram` maintains the lossless SentencePiece Unigram comparison
with a vocabulary of 4096, a Mamba-3 sequence model, and Constriction range
coding. It starts from the verified token-codec performance snapshot bdd731c.
The project mainline is `main`, which uses BPE 4096 and GPU integer quantization.

Prepare the Mamba-3/PyTorch CUDA environment and install
`requirements-codec.txt`. Use the Unigram tokenizer and matching Unigram
inference checkpoint; BPE and byte checkpoints are not interchangeable.
Datasets, weights, and full experiment outputs are managed outside Git.

```python
from mamba3_byte_compress.token_codec import load_token_predictor, encode, decode
from mamba3_byte_compress.tokenization import LosslessTokenizer
from mamba3_byte_compress.codec import verify_files

checkpoint = "/path/to/unigram4096/inference.pt"
tokenizer = LosslessTokenizer("/path/to/unigram4096/tokenizer.model", threads=24)
predictor = load_token_predictor(checkpoint)
encode("input.bin", "compressed.m3rc", predictor, tokenizer)
decode("compressed.m3rc", "decoded.bin", predictor)
verify_files("input.bin", "decoded.bin")
```

This snapshot uses FP32 probabilities and the optimized token container. See
the [experiment guide](docs/实验方案/子词与字节压缩对照实验.md),
[performance report](docs/实验报告/子词模型编解码性能.md), and
[project directory guide](docs/目录说明.md). Training is available through
`scripts/benchmark_sentencepiece.py train --algorithm unigram --run-dir <run>`
after the required preparation, preflight, and tuning stages.

Run `python -m unittest discover -s tests -v` in the configured environment.
The root `codec.py`, `train.py`, and `evaluate.py` remain byte-model tools.

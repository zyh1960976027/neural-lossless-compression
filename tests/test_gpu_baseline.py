"""真实基线权重的边界、重复编码、损坏检测及独立推理包检查。"""
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from neural_compression import codec
from neural_compression.config import (
    DEFAULT_CHECKPOINT,
    DEFAULT_TOKENIZER,
    MODEL_CONFIG,
)
from neural_compression.data import LosslessTokenizer
from neural_compression.io import verify_files


@unittest.skipUnless(
    os.environ.get("MAMBA_CODEC_GPU_TEST") == "1", "opt-in real CUDA baseline"
)
class BaselineGPUVerification(unittest.TestCase):
    def test_real_model_boundaries_repeat_corruption_and_overwrite(self):
        torch.set_num_threads(1)
        tokenizer = LosslessTokenizer(DEFAULT_TOKENIZER, threads=24)
        predictor = codec.load_predictor(DEFAULT_CHECKPOINT, batch_size=2)
        self.assertEqual(sum(p.numel() for p in predictor.model.parameters()), 6158796)
        self.assertEqual(predictor.model.embedding.num_embeddings, 4096)
        rng = np.random.default_rng(20261021)
        samples = [
            b"",
            bytes(range(256)),
            b"\xff" * 2048,
            b"\xff" * 2049,
            b"\xff" * 4096,
            b"\xff" * 4097,
            rng.integers(0, 256, 1003, dtype=np.uint8).tobytes(),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for number, raw in enumerate(samples):
                source = root / f"{number}.raw"
                source.write_bytes(raw)
                streams = []
                for repeat in (1, 2):
                    packed = root / f"{number}.{repeat}.m3rc"
                    output = root / f"{number}.{repeat}.decoded"
                    encoded = codec.encode(source, packed, predictor, tokenizer)
                    restored = codec.decode(packed, output, predictor)
                    self.assertTrue(verify_files(source, output))
                    self.assertTrue(restored["checksum_verified"])
                    streams.append(packed.read_bytes())
                    with self.assertRaises(FileExistsError):
                        codec.encode(source, packed, predictor, tokenizer)
                    self.assertEqual(packed.read_bytes(), streams[-1])
                    with self.assertRaises(FileExistsError):
                        codec.decode(packed, output, predictor)
                self.assertEqual(*streams)
            original = packed.read_bytes()
            for index, damaged in enumerate(
                (original[:-1], original + b"extra", original[:-4] + b"xxxx")
            ):
                packed.write_bytes(damaged)
                output = root / f"corrupted{index}"
                with self.assertRaises(ValueError):
                    codec.decode(packed, output, predictor)
                self.assertFalse(output.exists())

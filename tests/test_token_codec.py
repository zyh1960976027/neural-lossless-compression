from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from mamba3_byte_compress import token_codec as tc
from mamba3_byte_compress.tokenization import LosslessTokenizer
from test_tokenization import toy_model


class ControlledPredictor:
    def __init__(self, tokenizer, batch_size=2):
        self.model = torch.nn.Identity()
        self.device = torch.device("cpu")
        self.batch_size = batch_size
        self.vocab_size = tokenizer.vocab_size
        self.tokenizer_sha256 = tokenizer.sha256
        self.checkpoint_sha256 = "0" * 64
        self.runtime = {"test": "controlled"}
        self.reset(batch_size)

    def synchronize(self):
        pass

    def reset(self, batch_size):
        self.states = np.zeros(batch_size, dtype=np.int64)
        self.steps = 0

    def step(self, tokens, active=None):
        self.states = (self.states + tokens) % self.vocab_size
        self.steps += 1
        if self.steps > 2047:
            raise AssertionError("Crossed a block reset")
        p = np.full((len(tokens), self.vocab_size), 0.1 / self.vocab_size, dtype=np.float32)
        p[np.arange(len(tokens)), self.states] += .9
        return p, None


class TokenCodecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = LosslessTokenizer(toy_model())

    def test_roundtrip_boundaries_and_repeatability(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for size in (0, 1, 2048, 2049, 4099):
                raw = directory / f"{size}.raw"
                raw.write_bytes(rng.integers(0, 256, size, dtype=np.uint8).tobytes())
                for repeat in (1, 2):
                    packed = directory / f"{size}.{repeat}.packed"
                    restored = directory / f"{size}.{repeat}.restored"
                    encoded = tc.encode(raw, packed, ControlledPredictor(self.tokenizer), self.tokenizer)
                    decoded = tc.decode(packed, restored, ControlledPredictor(self.tokenizer))
                    self.assertEqual(restored.read_bytes(), raw.read_bytes())
                    self.assertEqual(encoded["original_sha256"], decoded["decoded_sha256"])
                    if size:
                        self.assertEqual(encoded["actual_bpb"], packed.stat().st_size * 8 / size)
                self.assertEqual((directory / f"{size}.1.packed").read_bytes(), (directory / f"{size}.2.packed").read_bytes())

    def test_corruption_and_wrong_checkpoint_remove_output(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw, packed, output = [directory / name for name in ("raw", "packed", "output")]
            raw.write_bytes(bytes(range(256)))
            tc.encode(raw, packed, ControlledPredictor(self.tokenizer), self.tokenizer)
            original = packed.read_bytes()
            for data in (original[:-1], original + b"extra", original[:-4]+b"xxxx"):
                packed.write_bytes(data)
                with self.assertRaises(ValueError):
                    tc.decode(packed, output, ControlledPredictor(self.tokenizer))
                self.assertFalse(output.exists())
            packed.write_bytes(original)
            predictor = ControlledPredictor(self.tokenizer)
            predictor.checkpoint_sha256 = "1" * 64
            with self.assertRaises(ValueError):
                tc.decode(packed, output, predictor)
            self.assertFalse(output.exists())

    def test_4096_quantization_extremes(self):
        from mamba3_byte_compress.token_pipeline import synthetic_preflight
        rows = synthetic_preflight()
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["verified"] for row in rows))

    def test_nonfinite_probability_rejected(self):
        class Invalid(ControlledPredictor):
            def step(self, tokens, active=None):
                p, logp = super().step(tokens, active)
                p[0, 0] = np.nan
                return p, logp
        with tempfile.TemporaryDirectory() as directory:
            raw, packed = Path(directory) / "raw", Path(directory) / "packed"
            raw.write_bytes(bytes(range(256)))
            with self.assertRaises(ValueError):
                tc.encode(raw, packed, Invalid(self.tokenizer), self.tokenizer)
            self.assertFalse(packed.exists())


if __name__ == "__main__":
    unittest.main()

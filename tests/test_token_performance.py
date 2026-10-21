"""Bitstream compatibility and validation regressions for the fast codec path."""
from pathlib import Path
import tempfile
import unittest
import numpy as np

from neural_compression import token_codec as tc
from neural_compression.tokenization import LosslessTokenizer
from test_token_codec import ControlledPredictor
from test_tokenization import toy_model


class TokenPerformanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = LosslessTokenizer(toy_model())

    def test_identical_streams_and_cross_decode_at_tail_boundaries(self):
        rng = np.random.default_rng(2026)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for size in (0, 1, 2048, 2049, 4095, 8195):
                source = root / f"raw{size}"
                source.write_bytes(rng.integers(0, 256, size, dtype=np.uint8).tobytes())
                streams = []
                for optimized in (False, True):
                    packed = root / f"{size}.{optimized}.packed"
                    enc = tc.encode(source, packed, ControlledPredictor(self.tokenizer, 3),
                                    self.tokenizer, optimized=optimized, profile=True)
                    streams.append(packed.read_bytes())
                    restored = root / f"{size}.{optimized}.decoded"
                    dec = tc.decode(packed, restored, ControlledPredictor(self.tokenizer, 3),
                                    optimized=not optimized, profile=True)
                    self.assertEqual(restored.read_bytes(), source.read_bytes())
                    self.assertEqual(enc["original_sha256"], dec["decoded_sha256"])
                    for metric, operation in ((enc, "encode"), (dec, "decode")):
                        self.assertAlmostEqual(sum(metric["phases_seconds"].values()),
                                               metric[operation + "_seconds"], places=5)
                self.assertEqual(*streams)

    def test_fast_validation_retains_invalid_distribution_rejection(self):
        for defect in ("nan", "positive_inf", "negative", "negative_inf", "zero_row"):
            class Invalid(ControlledPredictor):
                def step(self, tokens, active=None):
                    p, logp = super().step(tokens, active)
                    if defect == "zero_row":
                        p[0] = 0
                    else:
                        p[0, 1] = dict(nan=np.nan, positive_inf=np.inf,
                                       negative=-0.1, negative_inf=-np.inf)[defect]
                    return p, logp
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                raw, packed = Path(tmp) / "raw", Path(tmp) / "packed"
                raw.write_bytes(bytes(range(256)))
                with self.assertRaises(ValueError):
                    tc.encode(raw, packed, Invalid(self.tokenizer), self.tokenizer, optimized=True)
                self.assertFalse(packed.exists())


if __name__ == "__main__":
    unittest.main()

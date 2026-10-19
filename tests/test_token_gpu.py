"""Opt-in real Mamba-3 tests: MAMBA_CODEC_GPU_TEST=1."""
import hashlib
import os
import unittest

import numpy as np
import torch

from mamba3_byte_compress import codec as c
from mamba3_byte_compress.codec_graph import BatchedStepV5Predictor
from mamba3_byte_compress.model import build_mamba3_token_model


@unittest.skipUnless(os.environ.get("MAMBA_CODEC_GPU_TEST") == "1", "opt-in real CUDA test")
class TokenGpuTest(unittest.TestCase):
    def test_4096_graph_replay_high_tokens_and_reset(self):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(2026)
        model, _ = build_mamba3_token_model()
        with torch.no_grad():
            model.lm_head.weight.normal_(std=.02)
        predictor = BatchedStepV5Predictor(model, "cuda:0", "0"*64, batch_size=3)
        reference = None
        with c._coding_mode(predictor):
            for repeat in range(2):
                predictor.reset(3)
                digest = hashlib.sha256()
                for position in range(2047):
                    p, _ = predictor.step(np.asarray([4095, 256, position % 4096]))
                    self.assertEqual(p.shape, (3, 4096))
                    self.assertTrue(np.isfinite(p).all())
                    np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-6)
                    digest.update(p.tobytes())
                if reference is not None:
                    self.assertEqual(digest.hexdigest(), reference)
                reference = digest.hexdigest()
                with self.assertRaises(ValueError):
                    predictor.step(np.zeros(3, dtype=np.int32))
                predictor.reset(1)
                predictor.step(np.asarray([4095]))
        self.assertEqual(predictor.graph_capture_count, 2)


if __name__ == "__main__":
    unittest.main()

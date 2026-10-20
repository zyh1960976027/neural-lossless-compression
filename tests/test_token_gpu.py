"""Opt-in real Mamba-3 tests: MAMBA_CODEC_GPU_TEST=1."""
import hashlib
import os
import unittest

import numpy as np
import torch

from mamba3_byte_compress import codec as c
from mamba3_byte_compress.codec_graph import BatchedStepV5Predictor, probability_validity
from mamba3_byte_compress.model import build_mamba3_token_model


@unittest.skipUnless(os.environ.get("MAMBA_CODEC_GPU_TEST") == "1", "opt-in real CUDA test")
class TokenGpuTest(unittest.TestCase):
    def test_device_probability_checks_reject_invalid_rows(self):
        for value in (float("nan"), float("inf"), -0.1, -float("inf")):
            p = torch.full((3, 4096), 1/4096, device="cuda")
            p[1, 10] = value
            self.assertFalse(probability_validity(p).item())
        p.fill_(0)
        self.assertFalse(probability_validity(p).item())
        p[:, 4095] = 1
        self.assertTrue(probability_validity(p).item())

    def test_prefetch_matches_serial_probabilities_and_drains_on_close(self):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(2026)
        model, _ = build_mamba3_token_model()
        with torch.no_grad():
            model.lm_head.weight.normal_(std=.02)
        predictor = BatchedStepV5Predictor(model, "cuda:0", "0"*64, batch_size=3, check_probabilities=True)
        rng = np.random.default_rng(2026)
        with c._coding_mode(predictor):
            for lengths in (np.array([2048, 2048, 7]), np.array([19]), np.array([19, 19, 1])):
                values = rng.integers(0, 4096, (len(lengths), lengths.max()), dtype=np.uint16)
                predictor.reset(len(lengths))
                serial = hashlib.sha256()
                for pos in range(1, int(lengths.max())):
                    p, _ = predictor.step(np.where(lengths > pos, values[:, pos-1], 0))
                    serial.update(p.tobytes())
                predictor.reset(len(lengths))
                parallel = hashlib.sha256()
                for p, _ in predictor.iter_prefetched(values, lengths):
                    parallel.update(p.tobytes())
                self.assertEqual(serial.digest(), parallel.digest())
                self.assertEqual(predictor.steps_in_block, int(lengths.max())-1)
                predictor.reset(len(lengths))
                iterator = predictor.iter_prefetched(values, lengths)
                next(iterator)
                iterator.close()
                predictor.reset(len(lengths))
                p, _ = predictor.step(values[:, 0])
                self.assertTrue(np.isfinite(p).all())
            with torch.no_grad():
                model.lm_head.weight.fill_(float("nan"))
            predictor.reset(1)
            with self.assertRaises(ValueError):
                predictor.step(np.array([0]))
            predictor.reset(1)
            with self.assertRaises(ValueError):
                next(predictor.iter_prefetched(np.zeros((1, 3), dtype=np.uint16), np.array([3])))

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

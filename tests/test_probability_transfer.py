"""Transport rounding, container identity, reusable buffers and real CUDA parity."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import constriction
import numpy as np
import torch

from neural_compression import token_codec as tc
from neural_compression import codec as c
from neural_compression.codec_graph import BatchedStepV5Predictor
from neural_compression.tokenization import LosslessTokenizer
from test_token_codec import ControlledPredictor
from test_tokenization import toy_model


class RoundedPredictor(ControlledPredictor):
    probability_transfer = "fp16"

    def step(self, tokens, active=None):
        p, _ = super().step(tokens, active)
        return p.astype(np.float16).astype(np.float32), None


def tiny_model():
    model = torch.nn.Module()
    model.embedding = torch.nn.Embedding(320, 16)
    model.blocks = torch.nn.ModuleList()
    model.final_norm = torch.nn.Identity()
    model.lm_head = torch.nn.Linear(16, 320)
    return model


class ProbabilityTransferTest(unittest.TestCase):
    def test_real_conversion_buffers_and_cache_on_cpu(self):
        torch.set_num_threads(1)
        model = tiny_model()
        fp32 = BatchedStepV5Predictor(model, "cpu", "0" * 64, 3, use_graph=False)
        fp16 = BatchedStepV5Predictor(model, "cpu", "0" * 64, 3, use_graph=False,
                                     probability_transfer="fp16", check_probabilities=True)
        with c._coding_mode(fp16):
            for size in (3, 1, 3, 2):
                fp32.reset(size)
                fp16.reset(size)
                address = fp16._host_probabilities.data_ptr()
                for position in range(3):
                    tokens = np.full(size, position, dtype=np.int32)
                    expected = fp32.step(tokens)[0].astype(np.float16).astype(np.float32)
                    actual = fp16.step(tokens)[0]
                    np.testing.assert_array_equal(expected, actual)
                    self.assertEqual(actual.dtype, np.float32)
                    self.assertEqual(fp16._host_transfer.dtype, torch.float16)
                    self.assertEqual(fp16._host_probabilities.data_ptr(), address)
        with self.assertRaises(AttributeError):
            fp16.probability_transfer = "fp32"
        with self.assertRaises(ValueError):
            BatchedStepV5Predictor(model, "cpu", "0" * 64, probability_transfer="bad")
        with torch.no_grad(), c._coding_mode(fp16):
            model.lm_head.weight.fill_(float("nan"))
            fp16.reset(1)
            with self.assertRaisesRegex(ValueError, "Invalid predicted"):
                fp16.step([0])

    def test_extreme_distributions_all_vocabularies(self):
        rng = np.random.default_rng(2026)
        family = constriction.stream.model.Categorical(perfect=False)
        for vocab in (4096, 8192, 16384):
            p = rng.random((4, vocab), dtype=np.float32)
            p /= p.sum(axis=1, keepdims=True)
            p[0] = 1 / vocab
            p[1] = 0
            p[1, 0] = 1
            p[2, -1] = 1e-12
            q = p.astype(np.float16).astype(np.float32)
            self.assertEqual(q[2, -1], 0)
            symbols = np.array([vocab-1, vocab-1, vocab-1, 256], dtype=np.int32)
            encoder = constriction.stream.queue.RangeEncoder()
            encoder.encode(symbols, family, q)
            decoder = constriction.stream.queue.RangeDecoder(encoder.get_compressed())
            np.testing.assert_array_equal(decoder.decode(family, q), symbols)

    def test_formats_boundaries_repeats_and_rejection(self):
        tokenizer = LosslessTokenizer(toy_model())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 0xff always uses byte fallback: byte and token boundaries coincide.
            for size in (0, 1, 2048, 2049, 4096, 4097):
                raw = root / f"raw{size}"
                raw.write_bytes(b"\xff" * size)
                for mode, cls in (("fp32", ControlledPredictor), ("fp16", RoundedPredictor)):
                    streams = []
                    for repeat in range(2):
                        packed = root / f"packed{size}.{mode}.{repeat}"
                        restored = root / f"restored{size}.{mode}.{repeat}"
                        tc.encode(raw, packed, cls(tokenizer, 2), tokenizer)
                        settings = tc.read_token_settings(packed)
                        self.assertEqual(settings, dict(batch_size=2, probability_transfer=mode))
                        with patch.object(tc, "load_token_predictor") as load:
                            tc.load_token_decoder("checkpoint", packed)
                            self.assertEqual(load.call_args.kwargs["probability_transfer"], mode)
                        result = tc.decode(packed, restored, cls(tokenizer, 2))
                        self.assertTrue(result["checksum_verified"])
                        self.assertEqual(raw.read_bytes(), restored.read_bytes())
                        streams.append(packed.read_bytes())
                    self.assertEqual(*streams)
                    wrong_cls = ControlledPredictor if mode == "fp16" else RoundedPredictor
                    with self.assertRaisesRegex(ValueError, "transfer mode"):
                        tc.decode(packed, root / "wrong", wrong_cls(tokenizer, 2))
                    self.assertFalse((root / "wrong").exists())
                    packed.write_bytes(streams[0][:-1])
                    with self.assertRaises(ValueError):
                        tc.decode(packed, root / "broken", cls(tokenizer, 2))
                    self.assertFalse((root / "broken").exists())


@unittest.skipUnless(os.environ.get("MAMBA_CODEC_GPU_TEST") == "1", "opt-in real CUDA test")
class TransferGPUVerification(unittest.TestCase):
    def test_graph_rounding_prefetch_full_block_and_cleanup(self):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(1)
        from neural_compression.model import build_mamba3_token_model
        torch.manual_seed(2026)
        model, _ = build_mamba3_token_model()
        with torch.no_grad():
            model.lm_head.weight.normal_(std=.02)
        make = lambda mode: BatchedStepV5Predictor(
            model, "cuda:0", "0" * 64, 3, probability_transfer=mode, check_probabilities=True)
        original, rounded = make("fp32"), make("fp16")
        rng = np.random.default_rng(2026)
        with c._coding_mode(rounded):
            for lengths in (np.array([2048, 2048, 7]), np.array([19]), np.array([19, 19, 1])):
                size, steps = len(lengths), int(lengths.max()) - 1
                values = rng.integers(0, 4096, (size, steps+1), dtype=np.int32)
                original.reset(size)
                rounded.reset(size)
                self.assertEqual(rounded._probabilities.dtype, torch.float16)
                self.assertTrue(rounded._host_transfer.is_pinned())
                baseline_bytes = original.probability_transfer_bytes
                rounded_bytes = rounded.probability_transfer_bytes
                expected = []
                for position in range(1, steps+1):
                    tokens = np.where(lengths > position, values[:, position-1], 0)
                    raw = original.step(tokens)[0]
                    actual = rounded.step(tokens)[0]
                    np.testing.assert_array_equal(raw.astype(np.float16).astype(np.float32), actual)
                    expected.append(actual.copy())
                self.assertEqual(original.probability_transfer_bytes-baseline_bytes, steps*size*4096*4)
                self.assertEqual(rounded.probability_transfer_bytes-rounded_bytes, steps*size*4096*2)
                rounded.reset(size)
                before = rounded.probability_transfer_bytes
                for reference, (actual, _) in zip(expected, rounded.iter_prefetched(values, lengths)):
                    np.testing.assert_array_equal(reference, actual)
                self.assertEqual(rounded.probability_transfer_bytes-before, steps*size*4096*2)
                rounded.reset(size)
                iterator = rounded.iter_prefetched(values, lengths)
                next(iterator)
                iterator.close()
                rounded.reset(size)
                np.testing.assert_array_equal(rounded.step(np.where(lengths > 1, values[:, 0], 0))[0], expected[0])
            with torch.no_grad():
                model.lm_head.weight.fill_(float("nan"))
            rounded.reset(1)
            iterator = rounded.iter_prefetched(np.zeros((1, 2), dtype=np.int32), np.array([2]))
            with self.assertRaisesRegex(ValueError, "Invalid predicted"):
                next(iterator)
            iterator.close()


if __name__ == "__main__":
    unittest.main()

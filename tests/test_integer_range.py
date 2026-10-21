"""Integer CDF contract, CPU entropy interface, containers and GPU transfers."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import zlib

import constriction
import numpy as np
import torch

from neural_compression import codec as c, token_codec as tc
from neural_compression.codec_graph import BatchedStepV5Predictor
from neural_compression.integer_range import (
    IntegerEncoder, IntegerDecoder, TOTAL, quantize_cdf, select_intervals, uniform_cdf,
)
from neural_compression.tokenization import LosslessTokenizer
from test_token_codec import ControlledPredictor
from test_tokenization import toy_model


class IntegerPredictor(ControlledPredictor):
    probability_transfer = "gpu_int"

    def step(self, tokens, active=None, targets=None):
        probabilities, _ = super().step(tokens, active)
        cdf = quantize_cdf(torch.from_numpy(probabilities)).numpy()
        return (cdf if targets is None else select_intervals(cdf, targets)), None


class IntegerRangeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_extremes_and_random_distributions(self):
        rng = np.random.default_rng(20261021)
        for vocab in (4096, 8192, 16384):
            probabilities = rng.random((64, vocab), dtype=np.float32)
            probabilities[0] = 1 / vocab
            probabilities[1:4] = 0
            probabilities[1:4, 0] = 1
            probabilities[2, -1] = 1e-40
            probabilities[3, vocab//2] = 1e-10
            cdfs = quantize_cdf(torch.from_numpy(probabilities)).numpy()
            self.assertTrue((cdfs[:, 0] == 0).all())
            self.assertTrue((cdfs[:, -1] == TOTAL).all())
            self.assertTrue((np.diff(cdfs, axis=1) >= 1).all())
            symbols = rng.integers(0, vocab, len(cdfs), dtype=np.int32)
            symbols[:4] = vocab-1  # Includes exact zero and subnormal model probabilities.
            encoder = IntegerEncoder()
            encoder.encode(select_intervals(cdfs, symbols))
            packed = encoder.get_compressed()
            np.testing.assert_array_equal(packed, encoder.get_compressed())
            decoder = IntegerDecoder(packed)
            np.testing.assert_array_equal(symbols, decoder.decode(cdfs))
            encoder.close()
            decoder.close()
            with self.assertRaises(ValueError):
                encoder.get_compressed()
            with self.assertRaises(ValueError):
                decoder.decode(cdfs)

    def test_reuses_constriction_range_engine_and_validates_input(self):
        vocab = 4096
        symbols = np.random.default_rng(0).integers(0, vocab, 10000, dtype=np.int32)
        encoder = IntegerEncoder()
        with self.assertRaises(ValueError):
            encoder.encode(np.array([[0, 0]], dtype=np.int32))
        with self.assertRaises(ValueError):
            encoder.encode(np.array([[-1, 1]], dtype=np.int32))
        width = TOTAL // vocab
        encoder.encode(np.stack((symbols * width, np.full_like(symbols, width)), axis=-1))
        baseline = constriction.stream.queue.RangeEncoder()
        uniform = constriction.stream.model.Categorical(np.full(vocab, 1/vocab), perfect=False)
        baseline.encode(symbols, uniform)
        np.testing.assert_array_equal(encoder.get_compressed(), baseline.get_compressed())
        decoder = IntegerDecoder(encoder.get_compressed())
        for invalid in ([[1, TOTAL//2, TOTAL]], [[0, 0, TOTAL]], [[0, TOTAL, TOTAL-1]]):
            with self.assertRaises(ValueError):
                decoder.decode(np.array(invalid, dtype=np.int32))
        encoder.close()
        decoder.close()

    def test_container_boundaries_and_format_validation(self):
        tokenizer = LosslessTokenizer(toy_model())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for size in (0, 1, 2048, 2049, 4096, 4097):
                raw = root / "raw"
                raw.write_bytes(b"\xff" * size)
                streams = []
                for repeat in range(2):
                    packed, output = root / f"packed{size}.{repeat}", root / f"output{size}.{repeat}"
                    tc.encode(raw, packed, IntegerPredictor(tokenizer), tokenizer)
                    result = tc.decode(packed, output, IntegerPredictor(tokenizer))
                    self.assertTrue(result["checksum_verified"])
                    self.assertEqual(raw.read_bytes(), output.read_bytes())
                    self.assertEqual(tc.read_token_settings(packed)["probability_transfer"], "gpu_int")
                    streams.append(packed.read_bytes())
                self.assertEqual(*streams)
                with self.assertRaisesRegex(ValueError, "transfer mode"):
                    tc.decode(packed, root / "bad", ControlledPredictor(tokenizer))
                self.assertFalse((root / "bad").exists())
                packed.write_bytes(streams[0][:-1])
                with self.assertRaises(ValueError):
                    tc.decode(packed, root / "bad", IntegerPredictor(tokenizer))
                self.assertFalse((root / "bad").exists())
            magic, length = tc.PREFIX.unpack(streams[0][:tc.PREFIX.size])
            header = json.loads(streams[0][tc.PREFIX.size:tc.PREFIX.size+length])
            header["frequency_precision"] = 23
            encoded = json.dumps(header).encode()
            packed.write_bytes(tc.PREFIX.pack(magic, len(encoded)) + encoded + c.CRC32.pack(zlib.crc32(encoded)))
            with self.assertRaisesRegex(ValueError, "quantization"):
                tc.read_token_settings(packed)

    def test_predictor_cpu_target_and_reset(self):
        model = torch.nn.Module()
        model.embedding = torch.nn.Embedding(320, 16)
        model.blocks = torch.nn.ModuleList()
        model.final_norm = torch.nn.Identity()
        model.lm_head = torch.nn.Linear(16, 320)
        predictor = BatchedStepV5Predictor(model, "cpu", "0"*64, 3, use_graph=False,
                                          probability_transfer="gpu_int")
        with c._coding_mode(predictor):
            for size in (3, 1, 3, 2):
                tokens = np.arange(size, dtype=np.int32)
                predictor.reset(size)
                cdfs = predictor.step(tokens)[0].copy()
                predictor.reset(size)
                actual = predictor.step(tokens, targets=tokens)[0]
                np.testing.assert_array_equal(actual, select_intervals(cdfs, tokens))
            with torch.no_grad():
                model.lm_head.weight.fill_(float("nan"))
            predictor.reset(1)
            with self.assertRaisesRegex(ValueError, "Invalid predicted"):
                predictor.step([0], targets=[0])


@unittest.skipUnless(os.environ.get("MAMBA_CODEC_GPU_TEST") == "1", "opt-in real CUDA test")
class IntegerGPUVerification(unittest.TestCase):
    def test_gpu_zero_and_tiny_probabilities(self):
        torch.set_num_threads(1)
        for vocab in (4096, 8192, 16384):
            p = torch.zeros((4, vocab), dtype=torch.float32, device="cuda:0")
            p[0] = 1 / vocab
            p[1:, 0] = 1
            p[2, -1] = 1e-40
            p[3, -1] = 1e-10
            cdf = quantize_cdf(p).cpu().numpy()
            self.assertTrue((np.diff(cdf, axis=1) >= 1).all())
            symbols = np.full(4, vocab-1, dtype=np.int32)
            encoder = IntegerEncoder()
            encoder.encode(select_intervals(cdf, symbols))
            decoder = IntegerDecoder(encoder.get_compressed())
            np.testing.assert_array_equal(symbols, decoder.decode(cdf))
            encoder.close()
            decoder.close()

    def test_full_block_serial_prefetch_intervals_and_cleanup(self):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(1)
        rng = np.random.default_rng(11)
        for vocab in (4096, 8192, 16384):
            model = torch.nn.Module()
            model.embedding = torch.nn.Embedding(vocab, 16)
            model.blocks = torch.nn.ModuleList()
            model.final_norm = torch.nn.Identity()
            model.lm_head = torch.nn.Linear(16, vocab)
            predictor = BatchedStepV5Predictor(model, "cuda:0", "0"*64, 3,
                                              probability_transfer="gpu_int")
            with c._coding_mode(predictor):
                for lengths in (np.array([2048, 2048, 7]), np.array([19]), np.array([19, 19, 1])):
                    size, steps = len(lengths), int(lengths.max())-1
                    values = rng.integers(0, vocab, (size, steps+1), dtype=np.int32)
                    predictor.reset(size)
                    self.assertTrue(predictor._host_transfer.is_pinned())
                    before = predictor.probability_transfer_bytes
                    expected = []
                    for position in range(1, steps+1):
                        tokens = np.where(lengths > position, values[:, position-1], 0)
                        cdf = predictor.step(tokens)[0]
                        self.assertTrue((np.diff(cdf, axis=1) > 0).all())
                        expected.append(select_intervals(cdf, values[:, position]))
                    self.assertEqual(predictor.probability_transfer_bytes-before, steps*size*(vocab+1)*4)
                    predictor.reset(size)
                    before = predictor.probability_transfer_bytes
                    for reference, (actual, _) in zip(expected, predictor.iter_prefetched(values, lengths)):
                        np.testing.assert_array_equal(reference, actual)
                    self.assertEqual(predictor.probability_transfer_bytes-before, steps*size*8)
                    predictor.reset(size)
                    for position, reference in enumerate(expected, 1):
                        previous = np.where(lengths > position, values[:, position-1], 0)
                        np.testing.assert_array_equal(reference, predictor.step(previous, targets=values[:, position])[0])
                    predictor.reset(size)
                    pending = predictor.iter_prefetched(values, lengths)
                    next(pending)
                    pending.close()
                    predictor.reset(size)
                    np.testing.assert_array_equal(expected[0], predictor.step(
                        np.where(lengths > 1, values[:, 0], 0), targets=values[:, 1])[0])
                with torch.no_grad():
                    model.lm_head.weight.fill_(float("nan"))
                predictor.reset(1)
                pending = predictor.iter_prefetched(np.zeros((1, 2), dtype=np.int32), np.array([2]))
                with self.assertRaisesRegex(ValueError, "Invalid predicted"):
                    next(pending)
                pending.close()


if __name__ == "__main__":
    unittest.main()

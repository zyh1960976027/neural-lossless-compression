"""Lightweight checks for batched state and v3/v4 codec roundtrips."""

import contextlib
import io
from pathlib import Path
import tempfile
import zlib
import unittest
from unittest.mock import patch

import numpy as np
import torch

import codec as cli
from neural_compression.codec import (
    BLOCK_SIZE, BATCHED_HEADER, CRC32, BATCHED_PROFILE, BATCHED_PROFILE_ID,
    BATCHED_V4_PROFILE, BATCHED_V4_PROFILE_ID, BatchedStepPredictor,
    BatchedStepV4Predictor, PrefixPredictor, decode_file, encode_file,
    read_codec_settings, validate_batch_size,
)


class ZeroMLP(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class StatefulMixer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype):
        return tuple(torch.zeros(batch_size, 4, dtype=dtype) for _ in range(4))

    def step(self, x, angle, ssm, k, v):
        if x.ndim != 2 or x.shape[1] != 4:
            raise AssertionError("Only one byte may enter the mixer.")
        self.calls.append((x.shape[0], int(angle[0, 0])))
        angle.add_(1)
        ssm.mul_(0.875).add_(x)
        k.copy_(x)
        v.copy_(ssm)
        return ssm * 0.25, angle, ssm, k, v

    def forward(self, *args, **kwargs):
        raise AssertionError("Step codec must not invoke a mixer full forward.")


class StatefulModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 4)
        block = torch.nn.Module()
        block.norm1 = torch.nn.Identity()
        block.norm2 = torch.nn.Identity()
        block.mixer = StatefulMixer()
        block.mlp = ZeroMLP()
        self.blocks = torch.nn.ModuleList([block])
        self.final_norm = torch.nn.Identity()
        self.lm_head = torch.nn.Linear(4, 256, bias=False)
        with torch.no_grad():
            values = torch.arange(1024).reshape(256, 4).float()
            self.embedding.weight.copy_(values.remainder(31) / 31)
            self.lm_head.weight.copy_(values.remainder(17) / 17)

    def forward(self, *args, **kwargs):
        raise AssertionError("Step codec must not invoke a model full forward.")


class UniformModel(torch.nn.Module):
    def forward(self, tokens):
        return torch.zeros(*tokens.shape, 256)


def batched_predictor(batch_size=2):
    return BatchedStepPredictor(StatefulModel(), "cpu", "0" * 64, batch_size)


def batched_v4_predictor(batch_size=2):
    return BatchedStepV4Predictor(StatefulModel(), "cpu", "0" * 64, batch_size)


class BatchedCodecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_independent_slots_and_reset(self):
        from neural_compression.codec_step_adapter import _aligned_rows

        # The real 3M checkpoint has 11 heads: padding must only change storage.
        for tensor in (torch.arange(22).reshape(2, 11).float(),
                       torch.arange(22).reshape(2, 11).bfloat16(),
                       torch.arange(3000).reshape(2, 1500)[:, 1:705].reshape(2, 11, 64).bfloat16()):
            aligned = _aligned_rows(tensor)
            self.assertTrue(torch.equal(aligned, tensor))
            self.assertTrue(all(s * aligned.element_size() % 16 == 0 for s in aligned.stride()[:-1]))
        predictor = batched_predictor()
        predictor.reset(2)
        first = predictor.step([7, 19])[0]
        second = predictor.step([8, 20])[0]
        predictor.reset(2)
        np.testing.assert_array_equal(first, predictor.step([7, 19])[0])
        np.testing.assert_array_equal(second, predictor.step([8, 20])[0])
        predictor.reset(2)
        predictor.step([200, 19])
        changed = predictor.step([201, 20])[0]
        np.testing.assert_array_equal(second[1], changed[1])
        self.assertFalse(np.array_equal(second[0], changed[0]))
        predictor.reset(1)
        self.assertTrue(all(torch.count_nonzero(state) == 0
                            for layer in predictor.states for state in layer))

    def test_cli_roundtrip_and_fast_boundaries(self):
        data = (bytes(range(256)) * 17)[:4098]
        raw, packed, decoded = (self.root / name for name in ("raw", "packed", "decoded"))
        raw.write_bytes(data)
        predictors = []

        def load_fresh(*args, **kwargs):
            predictor = batched_v4_predictor(kwargs["batch_size"])
            predictors.append(predictor)
            return predictor

        with patch.object(cli, "load_predictor", side_effect=load_fresh) as load, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = cli.main(["encode", "--input", str(raw), "--output", str(packed), "--batch-size", "2", "--format", "v4"])
            self.assertEqual(load.call_args.kwargs, {"inference": "batched-step-v4", "batch_size": 2})
            self.assertEqual(result["profile"], BATCHED_V4_PROFILE)
            self.assertEqual(result["file_format"], "v4")
            self.assertEqual(result["num_range_streams"], 2)
            self.assertEqual(result["predicted_bytes"], 4095)
            self.assertEqual(result["uniform_bytes"], 3)
            self.assertEqual(result["num_model_steps"], 2048)
            raw.unlink()
            cli.main(["decode", "--input", str(packed), "--output", str(decoded)])
            self.assertEqual(load.call_args.kwargs, {"inference": "batched-step-v4", "batch_size": 2})
        self.assertEqual(decoded.read_bytes(), data)
        self.assertIsNot(predictors[0], predictors[1])
        for predictor in predictors:
            self.assertEqual(predictor.model.blocks[0].mixer.calls,
                             [(2, i) for i in range(BLOCK_SIZE - 1)] + [(1, 0)])
        for version, factory in (("v3", batched_predictor), ("v4", batched_v4_predictor)):
            for size in (0, 1):
                raw = self.root / f"{version}.raw{size}"
                packed = self.root / f"{version}.packed{size}"
                decoded = self.root / f"{version}.decoded{size}"
                raw.write_bytes(b"x" * size)
                result = encode_file(raw, packed, factory())
                self.assertEqual(result["num_model_steps"], 0)
                decode_file(packed, decoded, factory())
                self.assertEqual(decoded.read_bytes(), raw.read_bytes())

    def test_configuration_and_corrupt_parameters(self):
        raw, packed, decoded = (self.root / name for name in ("raw", "packed", "decoded"))
        raw.write_bytes(b"x")
        encode_file(raw, packed, batched_predictor())
        self.assertEqual(read_codec_settings(packed), {"inference": "batched-step", "batch_size": 2})
        original = packed.read_bytes()
        fields = list(BATCHED_HEADER.unpack_from(original))
        self.assertEqual(fields[1], BATCHED_PROFILE_ID)
        with self.assertRaisesRegex(ValueError, "batch_size mismatch"):
            decode_file(packed, decoded, batched_predictor(1))
        self.assertFalse(decoded.exists())
        for field, value in ((4, 0), (1, 255)):
            altered = fields.copy()
            altered[field] = value
            header = BATCHED_HEADER.pack(*altered)
            packed.write_bytes(header + CRC32.pack(zlib.crc32(header)) + original[len(header) + 4:])
            with self.assertRaises(ValueError):
                read_codec_settings(packed)
        packed.write_bytes(original[:18] + bytes([original[18] ^ 1]) + original[19:])
        with self.assertRaisesRegex(ValueError, "CRC32 mismatch"):
            read_codec_settings(packed)
        for invalid in (-1, 2**32, True):
            with self.assertRaises(ValueError):
                validate_batch_size(invalid)
        large = self.root / "large_batch"
        large_decoded = self.root / "large_decoded"
        encode_file(raw, large, batched_predictor(1024))
        self.assertEqual(read_codec_settings(large), {"inference": "batched-step", "batch_size": 1024})
        decode_file(large, large_decoded, batched_predictor(1024))
        self.assertEqual(large_decoded.read_bytes(), raw.read_bytes())
        legacy = self.root / "legacy"
        encode_file(raw, legacy, PrefixPredictor(UniformModel(), "cpu", "0" * 64))
        self.assertEqual(read_codec_settings(legacy), {"inference": "prefix"})
        with self.assertRaisesRegex(ValueError, "profile mismatch"):
            decode_file(legacy, decoded, batched_predictor())
        self.assertFalse(decoded.exists())

    def test_v4_group_stream_roundtrip_and_settings(self):
        data = (bytes(range(256)) * 17)[:4098]
        raw, packed, decoded = (self.root / name for name in ("v4.raw", "v4.m3rc", "v4.decoded"))
        raw.write_bytes(data)
        result = encode_file(raw, packed, batched_v4_predictor(2))
        self.assertEqual(result["file_format"], "v4")
        self.assertEqual(result["format_version"], 4)
        self.assertEqual(result["num_blocks"], 3)
        self.assertEqual(result["num_batches"], 2)
        self.assertEqual(result["num_range_streams"], 2)
        self.assertEqual(result["frame_header_bytes"], 16)
        self.assertEqual(read_codec_settings(packed), {"inference": "batched-step-v4", "batch_size": 2})
        fields = BATCHED_HEADER.unpack_from(packed.read_bytes())
        self.assertEqual(fields[1], BATCHED_V4_PROFILE_ID)
        with self.assertRaisesRegex(ValueError, "batch_size mismatch"):
            decode_file(packed, decoded, batched_v4_predictor(1))
        self.assertFalse(decoded.exists())
        decode_file(packed, decoded, batched_v4_predictor(2))
        self.assertEqual(decoded.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()

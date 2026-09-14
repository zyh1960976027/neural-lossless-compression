import contextlib
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib
import torch
from torch.utils.data import DataLoader

import codec as cli
from mamba3_byte_compress.codec import (
    BLOCK_SIZE, FRAME, HEADER, COMPACT_HEADER, COMPACT_TRAILER, CRC32,
    PrefixPredictor, decode_file, encode_file,
    roundtrip_file, sha256_file, verify_files,
)
from mamba3_byte_compress.data import ByteEvaluationDataset
from mamba3_byte_compress.metrics import evaluate_probability_model


class UniformModel(torch.nn.Module):
    def forward(self, inputs):
        if self.training or torch.is_grad_enabled():
            raise AssertionError("Coding must use eval mode with gradients disabled.")
        return torch.zeros(256).expand(*inputs.shape, 256)


class HistoryModel(torch.nn.Module):
    """A causal distribution that depends on all bytes since the block reset."""
    def forward(self, inputs):
        centers = inputs.cumsum(dim=1).remainder(256).float()
        classes = torch.arange(256).float()
        return -(centers.unsqueeze(-1) - classes).abs() / 32


class RecordingPredictor(PrefixPredictor):
    def __init__(self, model):
        super().__init__(model, "cpu", "0" * 64)
        self.histories = []

    def __call__(self, history):
        self.histories.append(bytes(history))
        return super().__call__(history)


class CodecTest(unittest.TestCase):
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
        self.raw = self.root / "original.bin"
        self.packed = self.root / "compressed.m3rc"
        self.decoded = self.root / "decoded.bin"
        self.predictor = PrefixPredictor(UniformModel(), "cpu", "0" * 64)

    def test_uniform_boundaries_and_all_byte_values(self):
        for size in (0, 1, 2, 2047, 2048, 2049, 2050, 4097, 4106):
            with self.subTest(size=size):
                self.raw.write_bytes((bytes(range(256)) * 17)[:size])
                packed, decoded = self.root / f"{size}.m3rc", self.root / f"{size}.bin"
                result = roundtrip_file(self.raw, packed, decoded, self.predictor)
                self.assertEqual(decoded.read_bytes(), self.raw.read_bytes())
                self.assertTrue(result["files_identical"])
                self.assertEqual(result["original_sha256"], sha256_file(decoded))
                self.assertAlmostEqual(result["theoretical_nll_bpb"], 0 if size == 0 else 8, places=10)
                self.assertEqual(result["num_prediction_blocks"], (max(0, size - 1) + 2047) // 2048)
                self.assertAlmostEqual(result["nll_nats_uniform_first_byte"], math.log(256) if size else 0)
                if size:
                    self.assertEqual(result["actual_bpb"], packed.stat().st_size * 8 / size)
                    self.assertAlmostEqual(result["encode_kb_per_second"], size / 1000 / result["encode_seconds"])
                    self.assertAlmostEqual(result["decode_kb_per_second"], size / 1000 / result["decode_seconds"])
                else:
                    self.assertIsNone(result["actual_bpb"])
                    self.assertEqual(result["encode_kb_per_second"], 0)

    def test_history_reset_alignment_and_existing_evaluation_nll(self):
        data = (b"A range coder predicts bytes.\x00\xff" * 70)[:2053]
        self.raw.write_bytes(data)
        encoder_predictor = RecordingPredictor(HistoryModel())
        decoder_predictor = RecordingPredictor(HistoryModel())
        result = encode_file(self.raw, self.packed, encoder_predictor)
        decode_file(self.packed, self.decoded, decoder_predictor)
        self.assertEqual(encoder_predictor.histories, decoder_predictor.histories)
        self.assertEqual(encoder_predictor.histories[0], data[:1])
        self.assertEqual(encoder_predictor.histories[2047], data[:2048])
        self.assertEqual(encoder_predictor.histories[2048], data[2048:2049])
        self.assertEqual(encoder_predictor.histories[-1], data[2048:-1])
        self.assertTrue(verify_files(self.raw, self.decoded))
        evaluation = evaluate_probability_model(
            HistoryModel(), DataLoader(ByteEvaluationDataset(self.raw, BLOCK_SIZE), batch_size=1),
            "cpu", use_bf16=False,
        )
        self.assertAlmostEqual(result["theoretical_nll_bpb"], evaluation["bpb"], places=5)

    def test_repeated_encoding_produces_identical_bytes(self):
        self.raw.write_bytes(bytes(range(128)))
        predictor = PrefixPredictor(HistoryModel(), "cpu", "0" * 64)
        encode_file(self.raw, self.packed, predictor)
        second = self.root / "second.m3rc"
        encode_file(self.raw, second, predictor)
        self.assertEqual(self.packed.read_bytes(), second.read_bytes())

    def test_tiny_probabilities_still_encode_every_symbol(self):
        class ExtremeModel(UniformModel):
            def forward(self, inputs):
                logits = torch.full((256,), -10000.0)
                logits[0] = 10000
                return logits.expand(*inputs.shape, 256)

        self.raw.write_bytes(b"\x00\xff\x01\x80")
        predictor = PrefixPredictor(ExtremeModel(), "cpu", "0" * 64)
        result = roundtrip_file(self.raw, self.packed, self.decoded, predictor)
        self.assertTrue(math.isfinite(result["theoretical_nll_bpb"]))
        self.assertTrue(result["files_identical"])

    def test_nonfinite_logits_remove_partial_output(self):
        class BrokenModel(UniformModel):
            def forward(self, inputs):
                return torch.full((256,), float("nan")).expand(*inputs.shape, 256)

        self.raw.write_bytes(b"abc")
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            encode_file(self.raw, self.packed, PrefixPredictor(BrokenModel(), "cpu", "0" * 64))
        self.assertFalse(self.packed.exists())

    def test_bad_headers_truncation_payload_hash_and_trailing_data(self):
        self.raw.write_bytes(b"range coding\x00\xff")
        encode_file(self.raw, self.packed, self.predictor, file_format="v1")
        good = self.packed.read_bytes()
        _, header_length = HEADER.unpack(good[:HEADER.size])
        frame_start = HEADER.size + header_length
        corrupt_payload = bytearray(good)
        corrupt_payload[frame_start + FRAME.size] ^= 1
        corrupt_hash = bytearray(good)
        corrupt_hash[-1] ^= 1
        bad_length = bytearray(good)
        struct.pack_into("<I", bad_length, frame_start, 3)
        bad_json = good[:HEADER.size] + b"!" + good[HEADER.size + 1:]
        variants = [b"", good[:11], good[:-1], b"badmagic" + good[8:],
                    good + b"extra", corrupt_payload, corrupt_hash, bad_length, bad_json]
        for index, data in enumerate(variants):
            with self.subTest(index=index):
                bad = self.root / f"bad{index}.m3rc"
                bad.write_bytes(data)
                with self.assertRaises(ValueError):
                    decode_file(bad, self.decoded, self.predictor)
                self.assertFalse(self.decoded.exists())

    def test_weight_runtime_and_probability_mismatch_rejected(self):
        self.raw.write_bytes(bytes(range(64)))
        encode_file(self.raw, self.packed, self.predictor)
        wrong_weights = PrefixPredictor(UniformModel(), "cpu", "1" * 64)
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256 mismatch"):
            decode_file(self.packed, self.decoded, wrong_weights)
        wrong_runtime = PrefixPredictor(UniformModel(), "cpu", "0" * 64)
        wrong_runtime.runtime["torch"] = "different"
        with self.assertRaisesRegex(ValueError, "runtime mismatch"):
            decode_file(self.packed, self.decoded, wrong_runtime)
        wrong_probs = PrefixPredictor(HistoryModel(), "cpu", "0" * 64)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            decode_file(self.packed, self.decoded, wrong_probs)
        self.assertFalse(self.decoded.exists())

    def test_outputs_never_overwrite_inputs_or_existing_files(self):
        self.raw.write_bytes(b"original")
        with self.assertRaises(ValueError):
            encode_file(self.raw, self.raw, self.predictor)
        self.packed.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            encode_file(self.raw, self.packed, self.predictor)
        self.assertEqual(self.packed.read_bytes(), b"keep")
        self.decoded.write_bytes(b"other")
        with self.assertRaisesRegex(ValueError, "differ"):
            verify_files(self.raw, self.decoded)

    def test_cli_roundtrip_and_standalone_decode_json(self):
        self.raw.write_bytes(b"abc\x00\xff")
        report = self.root / "results.json"
        with patch.object(cli, "load_predictor", return_value=self.predictor), \
             contextlib.redirect_stdout(io.StringIO()) as stdout, \
             contextlib.redirect_stderr(io.StringIO()):
            result = cli.main(["roundtrip", "--input", str(self.raw), "--compressed", str(self.packed),
                               "--decoded", str(self.decoded), "--results", str(report)])
        self.assertEqual(json.loads(stdout.getvalue()), result)
        self.assertEqual(json.loads(report.read_text()), result)
        self.assertTrue(result["files_identical"])
        self.raw.unlink()  # Standalone decoding has no access to the original.
        other = self.root / "standalone.bin"
        with patch.object(cli, "load_predictor", return_value=self.predictor), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cli.main(["decode", "--input", str(self.packed), "--output", str(other)])
        self.assertEqual(other.read_bytes(), b"abc\x00\xff")

    def test_all_formats_keep_probabilities_and_compact_frames_identical(self):
        self.raw.write_bytes((b"known history\x00\xff" * 150)[:2053])
        results, packed, histories = {}, {}, {}
        for file_format in ("v1", "v2-framed", "v2"):
            predictor = RecordingPredictor(HistoryModel())
            packed[file_format] = self.root / f"{file_format}.m3rc"
            result = roundtrip_file(self.raw, packed[file_format], self.root / f"{file_format}.bin",
                                    predictor, file_format=file_format)
            results[file_format] = result
            histories[file_format] = predictor.histories
            self.assertTrue(result["files_identical"])
            self.assertEqual(result["file_format"], file_format)
        self.assertEqual(histories["v1"], histories["v2-framed"])
        self.assertEqual(histories["v1"], histories["v2"])
        self.assertEqual(results["v1"]["theoretical_nll_bpb"], results["v2"]["theoretical_nll_bpb"])
        legacy = packed["v1"].read_bytes()
        length = HEADER.unpack(legacy[:HEADER.size])[1]
        compact = packed["v2-framed"].read_bytes()
        self.assertEqual(legacy[HEADER.size + length:-32],
                         compact[COMPACT_HEADER.size + CRC32.size:-COMPACT_TRAILER.size])
        self.assertEqual(results["v2"]["container_overhead_bytes"], 106)
        self.assertEqual(results["v2"]["num_range_streams"], 1)
        self.assertEqual(results["v1"]["num_range_streams"], 3)
        self.assertLess(results["v2"]["compressed_bytes"], results["v1"]["compressed_bytes"])

    def test_compact_corruption_and_unsupported_profile(self):
        self.raw.write_bytes(b"known bytes\x00\xff")
        for file_format in ("v2", "v2-framed"):
            good_path = self.root / f"{file_format}.m3rc"
            encode_file(self.raw, good_path, self.predictor, file_format=file_format)
            good = good_path.read_bytes()
            variants = [good[:69], good[:-1], good + b"extra"]
            for index in (8, COMPACT_HEADER.size, COMPACT_HEADER.size + CRC32.size, len(good) - 1):
                damaged = bytearray(good)
                damaged[index] ^= 1
                variants.append(damaged)
            for index in (8, 9):
                damaged = bytearray(good)
                damaged[index] = 255  # Unknown profile or stream mode, with valid CRC.
                CRC32.pack_into(damaged, COMPACT_HEADER.size, zlib.crc32(damaged[:COMPACT_HEADER.size]))
                variants.append(damaged)
            for index, damaged in enumerate(variants):
                bad = self.root / f"bad-{file_format}-{index}.m3rc"
                bad.write_bytes(damaged)
                with self.subTest(file_format=file_format, index=index), self.assertRaises(ValueError):
                    decode_file(bad, self.decoded, self.predictor)
                self.assertFalse(self.decoded.exists())

    def test_legacy_and_compact_framed_empty_and_single_byte(self):
        for file_format in ("v1", "v2-framed"):
            for data in (b"", b"\xff"):
                self.raw.write_bytes(data)
                label = f"{file_format}-{len(data)}"
                result = roundtrip_file(self.raw, self.root / f"{label}.m3rc", self.root / label,
                                        self.predictor, file_format=file_format)
                self.assertTrue(result["files_identical"])


if __name__ == "__main__":
    unittest.main()

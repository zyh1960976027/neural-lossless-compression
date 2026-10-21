import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from neural_compression import cli, token_codec
from neural_compression.experiments import DEFAULT_TOKEN_CHECKPOINT, DEFAULT_TOKENIZER
from neural_compression.tokenization import LosslessTokenizer
from test_integer_range import IntegerPredictor
from test_tokenization import toy_model


class CommandLineTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw.bin"
        self.raw.write_bytes(bytes(range(256)) * 9 + b"\x00\xff\n")
        self.checkpoint = self.root / "inference.pt"
        self.checkpoint.write_bytes(b"controlled predictor checkpoint")
        self.model = self.root / "tokenizer.model"
        self.model.write_bytes(toy_model())
        self.tokenizer = LosslessTokenizer(self.model)

    def arguments(self, command, *extra):
        args = [command, "--input", str(self.raw), "--checkpoint", str(self.checkpoint)]
        if command != "decode":
            args += ["--tokenizer", str(self.model), "--batch-size", "2", "--tokenizer-threads", "1"]
        return args + list(extra)

    def predictor(self, checkpoint, device="cuda:0", *, batch_size, probability_transfer, **kwargs):
        self.assertEqual(Path(checkpoint), self.checkpoint)
        self.assertEqual(probability_transfer, "gpu_int")
        return IntegerPredictor(self.tokenizer, batch_size=batch_size)

    def test_main_defaults_select_bpe4096_gpu_integer(self):
        args = cli.parse_args(["encode", "-i", "raw", "-o", "packed"])
        self.assertEqual(args.checkpoint, DEFAULT_TOKEN_CHECKPOINT)
        self.assertEqual(args.tokenizer, DEFAULT_TOKENIZER)
        self.assertEqual(args.probability_transfer, "gpu_int")
        self.assertEqual(args.batch_size, 1024)

    def test_real_integer_encode_and_independent_decode_read_header_settings(self):
        packed = self.root / "new" / "packed.m3rc"
        decoded = self.root / "restored" / "decoded.bin"
        report = self.root / "metrics" / "encode.json"
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(cli, "load_token_predictor", side_effect=self.predictor) as encode_load:
            encoded = cli.main(self.arguments("encode", "-o", str(packed), "--results", str(report)))
        self.assertEqual(encoded["probability_transfer"], "gpu_int")
        self.assertEqual(json.loads(report.read_text()), encoded)
        self.assertEqual(token_codec.read_token_settings(packed),
                         dict(batch_size=2, probability_transfer="gpu_int"))
        encode_load.assert_called_once()

        # Decoding must work without the external tokenizer or encoding options.
        self.model.unlink()
        args = ["decode", "-i", str(packed), "-o", str(decoded), "-c", str(self.checkpoint)]
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(token_codec, "load_token_predictor", side_effect=self.predictor) as decode_load:
            result = cli.main(args)
        self.assertEqual(decoded.read_bytes(), self.raw.read_bytes())
        self.assertTrue(result["checksum_verified"])
        self.assertEqual(decode_load.call_args.kwargs["batch_size"], 2)
        self.assertEqual(decode_load.call_args.kwargs["probability_transfer"], "gpu_int")

    def test_roundtrip_verifies_every_byte_and_keeps_both_metric_sets(self):
        packed = self.root / "roundtrip" / "packed"
        decoded = self.root / "roundtrip" / "decoded"
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(cli, "load_token_predictor", side_effect=self.predictor):
            result = cli.main(self.arguments("roundtrip", "--compressed", str(packed),
                                             "--decoded", str(decoded)))
        self.assertTrue(result["files_identical"])
        self.assertTrue(result["decode"]["checksum_verified"])
        self.assertIn("encode_seconds", result["encode"])
        self.assertIn("decode_seconds", result["decode"])
        self.assertEqual(decoded.read_bytes(), self.raw.read_bytes())

    def test_existing_files_aliases_and_reports_are_rejected_before_loading(self):
        packed = self.root / "packed"
        packed.write_bytes(b"keep")
        report = self.root / "report"
        report.write_bytes(b"keep report")
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "absent")
        cases = [self.arguments("encode", "-o", str(path))
                 for path in (packed, self.raw, self.checkpoint, self.model, dangling)]
        cases += [self.arguments("encode", "-o", str(self.root / "new"), "--results", str(report)),
                  self.arguments("encode", "-o", str(self.root / "same"),
                                 "--results", str(self.root / "same")),
                  self.arguments("roundtrip", "--compressed", str(self.root / "nested"),
                                 "--decoded", str(self.root / "nested" / "file"))]
        original = {path: path.read_bytes() for path in (self.raw, self.checkpoint, self.model, packed, report)}
        with patch.object(cli, "load_token_predictor") as load:
            for args in cases:
                with self.subTest(args=args), self.assertRaises((ValueError, FileExistsError)):
                    cli.main(args)
            load.assert_not_called()
        self.assertTrue(dangling.is_symlink())
        self.assertEqual({path: path.read_bytes() for path in original}, original)

    def test_bad_batch_and_missing_inputs_fail_without_loading(self):
        with patch.object(cli, "load_token_predictor") as load:
            args = self.arguments("encode", "-o", str(self.root / "new"), "--batch-size", str(2**32))
            with self.assertRaises(ValueError):
                cli.main(args)
            self.checkpoint.unlink()
            with self.assertRaises(FileNotFoundError):
                cli.main(self.arguments("encode", "-o", str(self.root / "new")))
            load.assert_not_called()
        self.assertFalse((self.root / "new").exists())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_args(["encode", "-i", "raw", "-o", "packed", "--batch-size", "0"])

    def test_both_launchers_and_clean_error_exit(self):
        project = Path(__file__).resolve().parents[1]
        helped = subprocess.run([sys.executable, "-m", "neural_compression", "--help"],
                                cwd=project, text=True, capture_output=True)
        self.assertEqual(helped.returncode, 0, helped.stderr)
        self.assertIn("roundtrip", helped.stdout)
        failed = subprocess.run([sys.executable, str(project / "run.py"), "encode", "-i", str(self.raw),
                                 "-o", str(self.root / "new"), "-c", str(self.root / "missing")],
                                cwd=project, text=True, capture_output=True)
        self.assertEqual(failed.returncode, 1)
        self.assertIn("运行失败", failed.stderr)
        self.assertNotIn("Traceback", failed.stderr)

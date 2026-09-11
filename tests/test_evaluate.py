import contextlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

import evaluate
from mamba3_byte_compress.config import MODEL_NAME
from mamba3_byte_compress.data import ByteEvaluationDataset
from mamba3_byte_compress.metrics import evaluate_probability_model
from mamba3_byte_compress.model import FORMAL_MAMBA3_CONFIG, build_mamba3_byte_model
from mamba3_byte_compress.trainer import load_model_checkpoint


class UniformByteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(256))

    def forward(self, inputs):
        if self.training or torch.is_grad_enabled():
            raise AssertionError("Evaluation must disable training and gradients.")
        return self.logits.expand(*inputs.shape, 256)


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "test.bin"
        self.raw.write_bytes(bytes([0, 255, 1, 2, 0, 3]))
        self.checkpoint = self.root / "best.pt"
        self.payload = {
            "model_name": MODEL_NAME,
            "model_config": dict(FORMAL_MAMBA3_CONFIG),
            "model": UniformByteModel().state_dict(),
            "checkpoint_kind": "best",
            "global_update": 123,
            "training_config": {"sequence_length": 4, "validation_batch_size": 2},
        }
        self.save_checkpoint()
        builder = patch.object(evaluate, "build_mamba3_byte_model", UniformByteModel)
        device = patch.object(evaluate, "_cuda_device", return_value=torch.device("cpu"))
        builder.start()
        device.start()
        self.addCleanup(builder.stop)
        self.addCleanup(device.stop)

    def save_checkpoint(self):
        torch.save(self.payload, self.checkpoint)

    def run_evaluation(self, **kwargs):
        return evaluate.evaluate_file(self.raw, self.checkpoint, **kwargs)

    def test_checkpoint_defaults_and_uniform_bpb(self):
        result = self.run_evaluation()
        self.assertAlmostEqual(result["bpb"], 8.0, places=6)
        self.assertEqual(result["num_bytes"], 6)
        self.assertEqual(result["predicted_bytes"], 5)
        self.assertEqual(result["sequence_length"], 4)
        self.assertEqual(result["batch_size"], 2)
        self.assertEqual(result["global_update"], 123)
        self.assertTrue(result["context_reset_each_block"])

    def test_best_and_latest_load_actual_weights(self):
        for kind in ("best", "latest"):
            with self.subTest(kind=kind):
                self.payload["checkpoint_kind"] = kind
                logits = torch.arange(256, dtype=torch.float32) / 32
                self.payload["model"] = {"logits": logits}
                self.save_checkpoint()
                result = self.run_evaluation()
                targets = torch.tensor(list(self.raw.read_bytes())[1:])
                expected = (-logits.log_softmax(0)[targets].sum().item() + math.log(256))
                expected /= 6 * math.log(2)
                self.assertAlmostEqual(result["bpb"], expected, places=6)
                self.assertEqual(result["checkpoint_kind"], kind)

    def test_missing_config_defaults_and_explicit_overrides(self):
        del self.payload["training_config"]
        self.save_checkpoint()
        result = self.run_evaluation()
        self.assertEqual((result["batch_size"], result["sequence_length"]), (16, 2048))
        result = self.run_evaluation(batch_size=1, sequence_length=2)
        self.assertEqual((result["batch_size"], result["sequence_length"]), (1, 2))
        self.assertAlmostEqual(result["bpb"], 8.0, places=6)

    def test_empty_single_byte_and_block_boundaries(self):
        for size in (0, 1, 4, 5, 6, 9, 10):
            with self.subTest(size=size):
                self.raw.write_bytes(bytes(range(size)))
                result = self.run_evaluation()
                self.assertEqual(result["num_bytes"], size)
                self.assertEqual(result["predicted_bytes"], max(0, size - 1))
                self.assertAlmostEqual(result["bpb"], 8.0 if size else 0.0, places=6)

    def test_json_output_matches_stdout(self):
        output = self.root / "results" / "metrics.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = evaluate.main([
                "--test-file", str(self.raw), "--checkpoint", str(self.checkpoint),
                "--output", str(output), "--batch-size", "1", "--sequence-length", "2",
            ])
        self.assertEqual(json.loads(stdout.getvalue()), result)
        self.assertEqual(json.loads(output.read_text()), result)
        self.assertEqual(result["batch_size"], 1)
        self.assertEqual(result["sequence_length"], 2)

    def test_invalid_cli_parameters(self):
        base = ["--test-file", str(self.raw), "--checkpoint", str(self.checkpoint)]
        for option, value in (("--batch-size", "0"), ("--sequence-length", "-1"),
                              ("--num-workers", "-1"), ("--batch-size", "bad")):
            with self.subTest(option=option, value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    evaluate.parse_args(base + [option, value])

    def test_invalid_api_parameters_and_missing_files(self):
        for kwargs in ({"batch_size": 0}, {"sequence_length": 0}, {"num_workers": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_evaluation(**kwargs)
        self.raw.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing test file"):
            self.run_evaluation()
        self.raw.write_bytes(b"abc")
        self.checkpoint.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing checkpoint file"):
            self.run_evaluation()

    def test_output_cannot_overwrite_inputs(self):
        for path in (self.raw, self.checkpoint):
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                evaluate.main(["--test-file", str(self.raw), "--checkpoint", str(self.checkpoint),
                               "--output", str(path)])
            self.assertEqual(path.read_bytes(), original)

    def test_mismatched_checkpoint_metadata_and_weights(self):
        for key, value, message in (
            ("model_name", "wrong", "model name"),
            ("model_config", {}, "model config"),
            ("model", {}, "Missing key"),
        ):
            with self.subTest(key=key):
                original = self.payload[key]
                self.payload[key] = value
                self.save_checkpoint()
                with self.assertRaisesRegex((ValueError, RuntimeError), message):
                    self.run_evaluation()
                self.payload[key] = original

    def test_nonfinite_metrics_are_rejected(self):
        self.payload["model"]["logits"].fill_(float("nan"))
        self.save_checkpoint()
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            self.run_evaluation()


class CudaDeviceTest(unittest.TestCase):
    def test_cpu_and_unavailable_cuda_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            evaluate._cuda_device("cpu")
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                evaluate._cuda_device("cuda:0")

    def test_bf16_checked_on_selected_device(self):
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "device_count", return_value=2), \
             patch.object(torch.cuda, "device") as selected, \
             patch.object(torch.cuda, "is_bf16_supported", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "BF16"):
                evaluate._cuda_device("cuda:1")
            selected.assert_called_once_with(torch.device("cuda:1"))
            with self.assertRaisesRegex(ValueError, "index"):
                evaluate._cuda_device("cuda:2")


REAL_CHECKPOINT = Path(os.environ.get(
    "OMNIZIP_MAMBA3_CHECKPOINT",
    "/media/ivc-17/New Volume/zyh/omnizip/runs/byte_backbone/formal_mamba3/checkpoints/best.pt",
))


@unittest.skipUnless(torch.cuda.is_available() and REAL_CHECKPOINT.is_file(),
                     "Real Mamba-3 checkpoint and CUDA are required")
class RealCheckpointEvaluationTest(unittest.TestCase):
    def test_cli_matches_direct_evaluation_and_batch_sizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "test.bin"
            raw.write_bytes((b"The byte model predicts the next byte.\n" * 120)[:4106])
            results = []
            for batch_size in (1, 2):
                output = root / f"batch_{batch_size}.json"
                completed = subprocess.run([
                    sys.executable, "evaluate.py", "--test-file", str(raw),
                    "--checkpoint", str(REAL_CHECKPOINT), "--output", str(output),
                    "--batch-size", str(batch_size), "--sequence-length", "2048",
                ], cwd=Path(__file__).resolve().parents[1], check=True,
                    capture_output=True, text=True)
                result = json.loads(completed.stdout)
                self.assertEqual(result, json.loads(output.read_text()))
                self.assertTrue(math.isfinite(result["bpb"]))
                self.assertEqual(result["num_bytes"], 4106)
                self.assertEqual(result["predicted_bytes"], 4105)
                results.append(result)
            model = build_mamba3_byte_model()
            load_model_checkpoint(model, REAL_CHECKPOINT)
            model.cuda().eval()
            for result in results:
                direct = evaluate_probability_model(
                    model, DataLoader(ByteEvaluationDataset(raw, 2048),
                                      batch_size=result["batch_size"]),
                    "cuda:0", use_bf16=True,
                )
                for key in direct.keys() - {"elapsed_seconds"}:
                    self.assertEqual(result[key], direct[key], key)
            # BF16 kernels round differently with batch shape. Each shape must
            # match the existing evaluator exactly; cross-shape BPB allows 0.001.
            self.assertLessEqual(abs(results[0]["bpb"] - results[1]["bpb"]), 0.001)
            print(json.dumps({"real_checkpoint_bpb": results[1]["bpb"],
                              "batch_1_bpb": results[0]["bpb"],
                              "num_bytes": results[1]["num_bytes"]}))


if __name__ == "__main__":
    unittest.main()

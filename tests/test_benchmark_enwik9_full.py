"""CPU-only checks for the complete enwik9 experiment harness."""

import json
import math
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.benchmark_codec_batched import (
    archive_incomplete_stage, completed_stage, extract_slice, next_log_path,
)
from scripts import benchmark_enwik9_full as full


class FullEnwik9HarnessTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_streaming_slice_and_resume_stage_validation(self):
        source = self.root / "source"
        target = self.root / "slice"
        source.write_bytes(bytes(range(64)))
        extract_slice(source, target, 7, 19, chunk_bytes=3)
        self.assertEqual(target.read_bytes(), bytes(range(7, 26)))
        with self.assertRaises(FileExistsError):
            extract_slice(source, target, 0, 5)
        self.assertEqual(target.read_bytes(), bytes(range(7, 26)))

        output = self.root / "run"
        output.mkdir()
        artifact = output / "packed"
        artifact.write_bytes(b"packed")
        (output / "encode.json").write_text(json.dumps({"num_bytes": 19}))
        self.assertEqual(completed_stage(output, "encode", artifact, 19)["num_bytes"], 19)
        with self.assertRaisesRegex(ValueError, "sample size"):
            completed_stage(output, "encode", artifact, 20)
        artifact.unlink()
        with self.assertRaises(FileExistsError):
            completed_stage(output, "encode", artifact, 19)
        archived = archive_incomplete_stage(output, "encode", artifact)
        self.assertEqual(len(archived), 1)
        self.assertFalse((output / "encode.json").exists())
        self.assertEqual(json.loads(archived[0].read_text())["num_bytes"], 19)

    def test_resume_log_names_do_not_overwrite(self):
        first = next_log_path(self.root, "encode")
        self.assertEqual(first.name, "encode.log")
        first.write_text("first")
        second = next_log_path(self.root, "encode")
        self.assertEqual(second.name, "encode.resume1.log")

    def test_source_size_modes(self):
        source = self.root / "enwik9"
        source.write_bytes(b"x" * 20)
        with patch.object(full, "FULL_BYTES", 20), patch.object(full, "PREFLIGHT_BYTES", 10):
            self.assertEqual(full.validate_source(source, "full"), source.resolve())
            self.assertEqual(full.validate_source(source, "preflight"), source.resolve())
            source.write_bytes(b"x" * 19)
            with self.assertRaisesRegex(ValueError, "exactly"):
                full.validate_source(source, "full")

    def test_metric_recalculation_and_aggregation(self):
        raw = self.root / "raw"
        raw.write_bytes(b"0123456789")
        model_dir = self.root / "model"
        codec_dir = model_dir / "codec"
        codec_dir.mkdir(parents=True)
        (codec_dir / "packed.m3rc").write_bytes(b"xx")
        (codec_dir / "decoded.bin").write_bytes(raw.read_bytes())
        inference = {
            "bpb": 1.25, "num_bytes": 10, "predicted_bytes": 9,
            "inference_seconds": 0.003,
            "inference_kb_per_second": 3.0, "peak_gpu_memory_gb": 2.0,
            "warmup_peak_gpu_memory_gb": 2.5,
            "peak_gpu_reserved_memory_gb": 2.7,
        }
        encode = {
            "num_bytes": 10, "compressed_bytes": 2, "actual_bpb": 1.6,
            "theoretical_nll_bpb": 1.4, "encode_seconds": 0.005,
            "encode_kb_per_second": 2.0, "run_peak_allocated_bytes": 3_000_000_000,
            "run_peak_reserved_bytes": 4_000_000_000,
            "uniform_bytes": 1, "predicted_bytes": 9, "file_format": "v5",
            "execution": "graph", "graph_enabled": True, "batch_size": 1024,
        }
        decode = {
            "num_bytes": 10, "decode_seconds": 0.004,
            "decode_kb_per_second": 2.5, "run_peak_allocated_bytes": 2_000_000_000,
            "run_peak_reserved_bytes": 3_000_000_000, "file_format": "v5",
            "execution": "graph", "graph_enabled": True, "batch_size": 1024,
        }
        verification = {"files_identical": True, "sha256_matches": True,
                        "original_sha256": "same", "decoded_sha256": "same"}
        (model_dir / "inference.json").write_text(json.dumps(inference))
        (codec_dir / "results.json").write_text(json.dumps(
            {"encode": encode, "decode": decode, "verification": verification}
        ))
        complexity = {
            "checkpoint": "/checkpoint.pt", "checkpoint_sha256": "hash",
            "parameter_counts": {
                "backbone_trainable_params": 90, "total_trainable_params": 100,
            },
            "macs": {
                "predicted_bytes": 9, "macs_per_predicted_byte": 7,
                "gmacs_per_full_block": 0.1, "total_macs": 63,
            },
        }
        result = full.aggregate_model("model", model_dir, raw, complexity)
        row = result["summary"]
        self.assertEqual(row["actual_bpb"], 1.6)
        self.assertEqual(row["peak_gpu_memory_gb"], 3.0)
        self.assertEqual(row["total_codec_macs"], 63)
        self.assertTrue(all(math.isfinite(value) for value in row.values()
                            if isinstance(value, float)))
        full.write_summary(self.root, {"model": result})
        self.assertIn("actual_bpb", (self.root / "summary.csv").read_text())
        self.assertIn("model", (self.root / "report.md").read_text())

    def test_interrupted_run_resumes_without_repeating_inference(self):
        source = self.root / "source"
        source.write_bytes(b"0123456789")
        baseline = self.root / "baseline.pt"
        formal = self.root / "formal.pt"
        baseline.write_bytes(b"baseline")
        formal.write_bytes(b"formal")
        experiment_root = self.root / "experiments"
        operations = []

        def fake_complexity(path, predicted_bytes):
            return {"checkpoint": str(path), "checkpoint_sha256": "fixed",
                    "macs": {"predicted_bytes": predicted_bytes}}

        def fake_run(command, log_path):
            operation = "inference" if "--internal-inference" in command else "codec"
            operations.append(operation)
            if operation == "inference":
                path = Path(command[command.index("--stage-output") + 1])
                checkpoint = command[command.index("--checkpoint") + 1]
                path.write_text(json.dumps({
                    "num_bytes": 10, "checkpoint": checkpoint, "batch_size": 128,
                    "sequence_length": 2048, "bpb": 1.0,
                    "inference_seconds": 1.0, "inference_kb_per_second": 1.0,
                    "warmup_batches_completed": 2, "peak_gpu_memory_gb": 1.0,
                }))
            else:
                path = Path(command[command.index("--output") + 1])
                path.mkdir(exist_ok=True)
                (path / "results.json").write_text("{}")
            return 1.0

        def fake_aggregate(label, *args, **kwargs):
            return {"summary": {"model": label, "actual_bpb": 1.0},
                    "verification": {"files_identical": True,
                                     "sha256_matches": True}}

        def fake_summary(output, models):
            (output / "summary.csv").write_text("model\n")

        common = ["preflight", "--source", str(source),
                  "--baseline-checkpoint", str(baseline),
                  "--formal-checkpoint", str(formal)]
        with patch.object(full, "PREFLIGHT_BYTES", 10), \
             patch.object(full, "TEST_EXPERIMENTS_ROOT", experiment_root), \
             patch.object(full, "complexity_for_checkpoint", side_effect=fake_complexity), \
             patch.object(full, "run_logged", side_effect=fake_run), \
             patch.object(full, "aggregate_model", side_effect=fake_aggregate), \
             patch.object(full, "write_summary", side_effect=fake_summary), \
             contextlib.redirect_stdout(io.StringIO()):
            full.run_experiment(full.parse_args(common))
            self.assertEqual(operations, ["inference", "codec", "inference", "codec"])
            run_dir = next(experiment_root.iterdir())
            operations.clear()
            full.run_experiment(full.parse_args(common + ["--resume-dir", str(run_dir)]))
            self.assertEqual(operations, ["codec", "codec"])


if __name__ == "__main__":
    unittest.main()

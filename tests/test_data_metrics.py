import importlib
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from mamba3_byte_compress.data import ByteEvaluationDataset, ByteTrainingDataset
from mamba3_byte_compress.metrics import evaluate_probability_model, masked_nll
from mamba3_byte_compress import metrics


LEGACY_COMMIT = "a5cd9baa24ed2b3959af5ce4572644d6076c69a9"
LEGACY_ROOT = Path(
    os.environ.get(
        "OMNIZIP_BASELINE_ROOT", "/media/ivc-17/New Volume/zyh/omnizip"
    )
)


class UniformByteModel(torch.nn.Module):
    def forward(self, inputs):
        return torch.zeros((*inputs.shape, 256), dtype=torch.float32)


class DistanceByteModel(torch.nn.Module):
    def forward(self, inputs):
        classes = torch.arange(256, device=inputs.device, dtype=torch.float32)
        return -(classes - inputs.unsqueeze(-1).float()).abs() / 32.0


class ByteDataAndMetricsTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _raw_file(self, values):
        path = self.root / "fixture.bin"
        path.write_bytes(bytes(values))
        return path

    def _starts_file(self, values):
        path = self.root / "starts.npy"
        np.save(path, np.asarray(values, dtype=np.int64), allow_pickle=False)
        return path

    def test_training_input_target_alignment(self):
        raw_path = self._raw_file([10, 20, 30, 40])
        starts_path = self._starts_file([0])
        dataset = ByteTrainingDataset(raw_path, starts_path, sequence_length=3)

        self.assertEqual(dataset[0]["input_tokens"].tolist(), [10, 20, 30])
        self.assertEqual(dataset[0]["target_tokens"].tolist(), [20, 30, 40])

    def test_evaluation_tail_padding_mask(self):
        raw_path = self._raw_file([10, 20, 30, 40])
        dataset = ByteEvaluationDataset(raw_path, sequence_length=2)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["input_tokens"].tolist(), [10, 20])
        self.assertEqual(dataset[0]["target_tokens"].tolist(), [20, 30])
        self.assertEqual(dataset[0]["masks"].tolist(), [True, True])
        self.assertEqual(dataset[1]["input_tokens"].tolist(), [30, 0])
        self.assertEqual(dataset[1]["target_tokens"].tolist(), [40, 0])
        self.assertEqual(dataset[1]["masks"].tolist(), [True, False])

    def test_zero_and_255_are_ordinary_bytes(self):
        raw_path = self._raw_file([0, 255, 0, 255])
        starts_path = self._starts_file([0])
        training = ByteTrainingDataset(raw_path, starts_path, sequence_length=3)
        evaluation = ByteEvaluationDataset(raw_path, sequence_length=3)

        self.assertEqual(training[0]["input_tokens"].tolist(), [0, 255, 0])
        self.assertEqual(training[0]["target_tokens"].tolist(), [255, 0, 255])
        self.assertEqual(evaluation[0]["input_tokens"].tolist(), [0, 255, 0])
        self.assertEqual(evaluation[0]["target_tokens"].tolist(), [255, 0, 255])
        self.assertEqual(evaluation[0]["masks"].tolist(), [True, True, True])

    def test_masked_nll_ignores_padding(self):
        logits = torch.zeros((1, 2, 256), dtype=torch.float32)
        logits[0, 1, 17] = 100.0
        targets = torch.tensor([[3, 99]], dtype=torch.long)
        masks = torch.tensor([[True, False]])

        nll, valid = masked_nll(logits, targets, masks)
        self.assertEqual(valid, 1)
        self.assertTrue(math.isclose(nll, math.log(256.0), rel_tol=1e-7))

    def test_uniform_256_class_model_is_eight_bpb(self):
        raw_path = self._raw_file([0, 255, 7, 0, 9, 2])
        dataset = ByteEvaluationDataset(raw_path, sequence_length=4)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        model = UniformByteModel()
        model.train()
        result = evaluate_probability_model(
            model, loader, device=torch.device("cpu"), use_bf16=False
        )

        self.assertTrue(math.isclose(result["bpb"], 8.0, rel_tol=1e-7))
        self.assertEqual(result["num_bytes"], 6)
        self.assertEqual(result["predicted_bytes"], 5)
        self.assertEqual(result["nll_nats_uniform_first_byte"], math.log(256.0))
        self.assertTrue(model.training)

    def test_single_file_byte_uses_uniform_first_byte_cost(self):
        raw_path = self._raw_file([255])
        dataset = ByteEvaluationDataset(raw_path, sequence_length=4)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        result = evaluate_probability_model(
            UniformByteModel(), loader, device=torch.device("cpu"), use_bf16=False
        )

        self.assertEqual(result["predicted_bytes"], 0)
        self.assertEqual(result["nll_nats_total"], math.log(256.0))
        self.assertTrue(math.isclose(result["bpb"], 8.0, rel_tol=1e-7))

    def test_forward_timing_excludes_other_work_and_padding_from_throughput(self):
        dataset = ByteEvaluationDataset(self._raw_file([0, 1, 2, 3, 4, 5]), 4)
        # Two forwards take 0.25 + 0.75 seconds; all evaluation takes 12 seconds.
        with patch.object(metrics.time, "perf_counter",
                          side_effect=[10.0, 12.0, 12.25, 20.0, 20.75, 22.0]):
            result = evaluate_probability_model(
                UniformByteModel(), DataLoader(dataset, batch_size=1), "cpu",
                use_bf16=False, measure_performance=True,
            )
        self.assertEqual(result["inference_seconds"], 1.0)
        self.assertEqual(result["elapsed_seconds"], 12.0)
        # Only five actual predictions count, not six file bytes or eight slots.
        self.assertEqual(result["inference_kb_per_second"], 0.005)
        self.assertIsNone(result["peak_gpu_memory_gb"])
        self.assertAlmostEqual(result["bpb"], 8.0, places=6)

    def test_cuda_profile_synchronizes_and_resets_selected_device_peak(self):
        dataset = ByteEvaluationDataset(self._raw_file([0, 1, 2]), 2)
        device = torch.device("cuda:1")
        events = []

        class CheckedModel(UniformByteModel):
            def forward(self, inputs):
                events.append("forward")
                return super().forward(inputs)

        def clock():
            events.append("clock")
            return float(len(events))

        # Exercise profiling control flow on CPU, with the CUDA boundary mocked.
        with patch.object(torch.Tensor, "to", lambda tensor, *args, **kwargs: tensor), \
             patch.object(torch.cuda, "synchronize", side_effect=lambda dev: events.append("sync")) as sync, \
             patch.object(torch.cuda, "reset_peak_memory_stats") as reset, \
             patch.object(torch.cuda, "max_memory_allocated", return_value=2_500_000_000) as peak, \
             patch.object(metrics.time, "perf_counter", side_effect=clock):
            result = evaluate_probability_model(
                CheckedModel(), DataLoader(dataset, batch_size=1), device,
                use_bf16=False, measure_performance=True,
            )
        forward_index = events.index("forward")
        self.assertEqual(events[forward_index - 2:forward_index + 3],
                         ["sync", "clock", "forward", "sync", "clock"])
        self.assertEqual(sync.call_args_list, [call(device)] * 3)
        reset.assert_called_once_with(device)
        peak.assert_called_once_with(device)
        self.assertEqual(result["peak_gpu_memory_gb"], 2.5)

    def test_default_evaluation_does_not_profile_or_reset_cuda_peak(self):
        dataset = ByteEvaluationDataset(self._raw_file([0, 1]), 2)
        with patch.object(torch.cuda, "reset_peak_memory_stats") as reset:
            result = evaluate_probability_model(
                UniformByteModel(), DataLoader(dataset), "cpu", use_bf16=False,
            )
        reset.assert_not_called()
        self.assertNotIn("inference_seconds", result)
        self.assertNotIn("peak_gpu_memory_gb", result)

    @unittest.skipUnless(LEGACY_ROOT.is_dir(), "frozen OmniZip baseline is unavailable")
    def test_frozen_baseline_data_and_metrics_parity(self):
        resolved_commit = subprocess.check_output(
            [
                "git",
                "-C",
                str(LEGACY_ROOT),
                "rev-parse",
                "byte-backbone-baseline-v1^{commit}",
            ],
            text=True,
        ).strip()
        self.assertEqual(resolved_commit, LEGACY_COMMIT)
        subprocess.run(
            [
                "git",
                "-C",
                str(LEGACY_ROOT),
                "diff",
                "--quiet",
                "byte-backbone-baseline-v1",
                "--",
                "byte_experiment/data.py",
                "byte_experiment/metrics.py",
            ],
            check=True,
        )

        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(LEGACY_ROOT))
        try:
            legacy_data = importlib.import_module("byte_experiment.data")
            legacy_metrics = importlib.import_module("byte_experiment.metrics")
        finally:
            sys.path.remove(str(LEGACY_ROOT))
            sys.dont_write_bytecode = previous_dont_write_bytecode

        raw_path = self._raw_file([0, 10, 20, 255, 30, 40, 50])
        starts_path = self._starts_file([0, 2])
        new_training = ByteTrainingDataset(raw_path, starts_path, sequence_length=3)
        old_training = legacy_data.ByteTrainingDataset(
            raw_path, starts_path, sequence_length=3
        )
        for index in range(len(new_training)):
            for key in ("input_tokens", "target_tokens"):
                self.assertTrue(
                    torch.equal(new_training[index][key], old_training[index][key])
                )

        new_evaluation = ByteEvaluationDataset(raw_path, sequence_length=4)
        old_evaluation = legacy_data.ByteEvaluationDataset(
            raw_path, sequence_length=4
        )
        self.assertEqual(len(new_evaluation), len(old_evaluation))
        for index in range(len(new_evaluation)):
            for key in ("input_tokens", "target_tokens", "masks"):
                self.assertTrue(
                    torch.equal(new_evaluation[index][key], old_evaluation[index][key])
                )

        new_result = evaluate_probability_model(
            DistanceByteModel(),
            DataLoader(new_evaluation, batch_size=2, shuffle=False),
            device=torch.device("cpu"),
            use_bf16=False,
        )
        old_result = legacy_metrics.evaluate_probability_model(
            DistanceByteModel(),
            DataLoader(old_evaluation, batch_size=2, shuffle=False),
            device=torch.device("cpu"),
            use_bf16=False,
        )
        for key in new_result.keys() - {"elapsed_seconds"}:
            if isinstance(new_result[key], float):
                self.assertAlmostEqual(new_result[key], old_result[key], places=12)
            else:
                self.assertEqual(new_result[key], old_result[key])


if __name__ == "__main__":
    unittest.main()

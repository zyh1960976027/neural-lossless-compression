import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import evaluate_mtp
import train_mtp
import train_mtp_weighted
from mamba3_byte_compress.config import make_training_config
from mamba3_byte_compress.data import ByteEvaluationDataset
from mamba3_byte_compress.metrics import evaluate_probability_model
from mamba3_byte_compress.model import (
    MTP4_MAMBA3_CONFIG,
    analytic_parameter_counts,
    build_mamba3_mtp4_model,
    parameter_counts,
)
from mamba3_byte_compress.mtp import (
    MTP4ByteTrainer,
    WEIGHTED_MTP_LOSS_WEIGHTS,
    WeightedMTP4ByteTrainer,
    evaluate_mtp_probability_model,
    load_mtp4_checkpoint,
    mtp_head_losses,
)


class CountingFutureModel(torch.nn.Module):
    def forward(self, inputs):
        return torch.stack([
            F.one_hot((inputs + offset + 1) % 256, 256).float() * 12
            for offset in range(4)
        ], dim=-2)


class CountingNextModel(torch.nn.Module):
    def forward(self, inputs):
        return F.one_hot((inputs + 1) % 256, 256).float() * 12


class TinyMTP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 8)
        self.lm_head = torch.nn.Linear(8, 256, bias=False)
        self.future_heads = torch.nn.ModuleList(
            torch.nn.Linear(8, 256, bias=False) for _ in range(3)
        )

    def forward(self, inputs):
        hidden = self.embedding(inputs)
        return torch.stack([
            self.lm_head(hidden), *(head(hidden) for head in self.future_heads)
        ], dim=-2)


class MTPModelTest(unittest.TestCase):
    def test_four_heads_have_expected_parameters_and_zero_initialization(self):
        model = build_mamba3_mtp4_model()
        self.assertEqual(len(model.blocks), 2)
        self.assertEqual(len(model.future_heads), 3)
        self.assertTrue(all(head.bias is None for head in model.future_heads))
        self.assertEqual(parameter_counts(model), {
            "backbone_trainable_params": 3_275_212,
            "total_trainable_params": 3_725_772,
        })
        self.assertEqual(analytic_parameter_counts(MTP4_MAMBA3_CONFIG),
                         parameter_counts(model))
        self.assertEqual(MTP4_MAMBA3_CONFIG["prediction_heads"], 4)
        self.assertEqual(torch.count_nonzero(model.lm_head.weight).item(), 0)
        self.assertTrue(all(torch.count_nonzero(head.weight).item() == 0
                            for head in model.future_heads))
        model.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(2)])
        self.assertEqual(tuple(model(torch.arange(6).unsqueeze(0)).shape),
                         (1, 6, 4, 256))

    def test_each_loss_uses_the_matching_future_byte(self):
        targets = torch.tensor([[11, 12, 13, 14, 15, 16]])
        logits = torch.zeros(1, 6, 4, 256, requires_grad=True)
        with torch.no_grad():
            for offset in range(4):
                for position in range(6 - offset):
                    logits[0, position, offset, targets[0, position + offset]] = 10
        losses = mtp_head_losses(logits, targets)
        self.assertEqual(len(losses), 4)
        self.assertTrue(all(loss.item() < 0.02 for loss in losses))
        (sum(losses) / 4).backward()
        self.assertTrue(all(logits.grad[:, :, offset, :].abs().sum().item() > 0
                            for offset in range(4)))
        with self.assertRaisesRegex(ValueError, "at least four"):
            mtp_head_losses(logits[:, :3], targets[:, :3])


class MTPProbabilityTest(unittest.TestCase):
    def test_block_boundaries_counts_and_baseline_head1_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bytes.bin"
            path.write_bytes(bytes(range(16)))
            loader = DataLoader(ByteEvaluationDataset(path, 8), batch_size=2)
            result = evaluate_mtp_probability_model(
                CountingFutureModel(), loader, "cpu", use_bf16=False
            )
            baseline = evaluate_probability_model(
                CountingNextModel(), loader, "cpu", use_bf16=False
            )
            self.assertEqual(
                [result[f"head{i}_predicted_bytes"] for i in range(1, 5)],
                [15, 13, 11, 9],
            )
            self.assertTrue(all(result[f"head{i}_bpb"] < 0.01 for i in range(1, 5)))
            baseline_predicted_bpb = baseline["nll_nats_predicted"] / (15 * math.log(2))
            self.assertAlmostEqual(result["head1_bpb"], baseline_predicted_bpb, places=8)
            self.assertAlmostEqual(result["bpb"], sum(
                result[f"head{i}_bpb"] for i in range(1, 5)
            ) / 4)

    def test_short_file_has_no_invented_future_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bytes.bin"
            path.write_bytes(b"abc")
            loader = DataLoader(ByteEvaluationDataset(path, 8), batch_size=1)
            result = evaluate_mtp_probability_model(
                CountingFutureModel(), loader, "cpu", use_bf16=False
            )
            self.assertEqual(
                [result[f"head{i}_predicted_bytes"] for i in range(1, 5)],
                [2, 1, 0, 0],
            )
            self.assertIsNone(result["head3_bpb"])
            self.assertIsNone(result["bpb"])


class MTPTrainingTest(unittest.TestCase):
    def test_weighted_loss_uses_the_fixed_distance_decay_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_file = root / "train.bin"
            validation_file = root / "validation.bin"
            starts_file = root / "starts.npy"
            train_file.write_bytes(bytes([7]) * 16)
            validation_file.write_bytes(bytes([7]) * 12)
            np.save(starts_file, np.zeros(4, dtype=np.int64))
            config = make_training_config(
                total_updates=2, warmup_updates=0,
                global_batch_size=2, micro_batch_size=1,
                sequence_length=4, validation_batch_size=1,
                validation_interval=1, checkpoint_interval=1,
            )
            trainer = WeightedMTP4ByteTrainer(
                model=TinyMTP(), train_file=train_file,
                validation_file=validation_file, train_starts_file=starts_file,
                run_dir=root / "run", device="cpu", training_config=config,
                model_name="tiny_mtp", model_config={"family": "tiny_mtp"},
                num_workers=0, scheduler_total_updates=1,
            )
            targets = torch.tensor([[1, 2, 3, 4]])
            logits = torch.randn(1, 4, 4, 256, requires_grad=True)
            total, components = trainer._batch_loss(logits, targets)
            expected = sum(
                weight * components[f"loss{index}"]
                for index, weight in enumerate(WEIGHTED_MTP_LOSS_WEIGHTS, start=1)
            ) / sum(WEIGHTED_MTP_LOSS_WEIGHTS)
            self.assertTrue(torch.equal(total, expected))
            self.assertTrue(torch.equal(total, components["weighted_total_loss"]))

            summary = trainer.train()
            latest = torch.load(
                root / "run" / "checkpoints" / "latest.pt",
                map_location="cpu", weights_only=False,
            )
            self.assertEqual(latest["loss_weights"], [1.0, 0.5, 0.25, 0.125])
            self.assertEqual(latest["scheduler"]["total_updates"], 1)
            self.assertEqual(summary["configuration"]["scheduler_total_updates"], 1)
            self.assertTrue((root / "run" / "bpb_heads_vs_steps.png").is_file())
            self.assertTrue((root / "run" / "loss_heads_vs_steps.png").is_file())
            for record in trainer.training_metrics:
                self.assertAlmostEqual(
                    record["weighted_total_loss"], record["total_loss"]
                )

    def test_training_logs_all_losses_and_resumes_with_recorded_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_file = root / "train.bin"
            validation_file = root / "validation.bin"
            starts_file = root / "starts.npy"
            train_file.write_bytes(bytes([7]) * 8)
            validation_file.write_bytes(bytes([7]) * 12)
            np.save(starts_file, np.zeros(4, dtype=np.int64))
            config = make_training_config(
                total_updates=2, warmup_updates=1,
                global_batch_size=2, micro_batch_size=1,
                sequence_length=4, validation_batch_size=1,
                validation_interval=1, checkpoint_interval=1,
            )
            kwargs = dict(
                train_file=train_file, validation_file=validation_file,
                train_starts_file=starts_file, run_dir=root / "run",
                device="cpu", training_config=config,
                model_name="tiny_mtp", model_config={"family": "tiny_mtp"},
                num_workers=0,
            )
            trainer = MTP4ByteTrainer(model=TinyMTP(), **kwargs)
            summary = trainer.train()
            self.assertEqual(trainer.global_update, 2)
            self.assertEqual(len(trainer.validation_metrics), 2)
            self.assertEqual(summary["final_results"]["selection_metric"], "mean_head_bpb")
            self.assertGreater(summary["final_results"]["training_elapsed_seconds"], 0)
            self.assertIsNone(summary["final_results"]["peak_training_gpu_memory_gb"])
            for record in trainer.training_metrics:
                self.assertTrue(all(key in record for key in (
                    "loss1", "loss2", "loss3", "loss4", "total_loss"
                )))
                self.assertAlmostEqual(record["total_loss"], sum(
                    record[f"loss{i}"] for i in range(1, 5)
                ) / 4, delta=1e-6)
            self.assertAlmostEqual(trainer.best_bpb, min(
                record["bpb"] for record in trainer.validation_metrics
            ))
            resumed = MTP4ByteTrainer(model=TinyMTP(), **kwargs)
            checkpoint = resumed.resume()
            self.assertEqual(resumed.global_update, 2)
            self.assertEqual(checkpoint["selection_metric"], "mean_head_bpb")
            self.assertGreater(resumed.training_elapsed_seconds, 0)
            self.assertTrue((root / "run" / "experiment_summary.json").is_file())
            self.assertTrue((root / "run" / "checkpoints" / "best.pt").is_file())

    def test_resume_can_strictly_extend_budget_and_window_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_file = root / "train.bin"
            validation_file = root / "validation.bin"
            starts_file = root / "starts.npy"
            train_file.write_bytes(bytes([7]) * 16)
            validation_file.write_bytes(bytes([7]) * 12)
            np.save(starts_file, np.zeros(2, dtype=np.int64))
            original_config = make_training_config(
                total_updates=1, warmup_updates=0,
                global_batch_size=2, micro_batch_size=1,
                sequence_length=4, validation_batch_size=1,
                validation_interval=1, checkpoint_interval=1,
            )
            kwargs = dict(
                train_file=train_file, validation_file=validation_file,
                train_starts_file=starts_file, run_dir=root / "run",
                device="cpu", model_name="tiny_mtp",
                model_config={"family": "tiny_mtp"}, num_workers=0,
            )
            MTP4ByteTrainer(
                model=TinyMTP(), training_config=original_config, **kwargs
            ).train()

            np.save(starts_file, np.zeros(4, dtype=np.int64))
            extended_config = make_training_config(
                **{**original_config, "total_updates": 2}
            )
            resumed = MTP4ByteTrainer(
                model=TinyMTP(), training_config=extended_config,
                allow_training_extension=True, **kwargs
            )
            resumed.resume()
            self.assertEqual(resumed.global_update, 1)
            self.assertEqual(resumed.scheduler.total_updates, 1)
            self.assertEqual(
                resumed.training_extension["scheduler_policy"],
                "preserve_original_then_hold_minimum_lr",
            )
            resumed.train()
            self.assertEqual(resumed.global_update, 2)

    def test_checkpoint_loader_rejects_baseline_model_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            model = build_mamba3_mtp4_model()
            torch.save({
                "model_name": MTP4_MAMBA3_CONFIG["name"],
                "model_config": dict(MTP4_MAMBA3_CONFIG),
                "model": model.state_dict(),
            }, path)
            loaded, metadata = load_mtp4_checkpoint(path)
            self.assertEqual(parameter_counts(loaded), parameter_counts(model))
            self.assertEqual(metadata["model_name"], MTP4_MAMBA3_CONFIG["name"])
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["model_name"] = "mamba3_siso_3m"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "model name"):
                load_mtp4_checkpoint(path)


class MTPCLITest(unittest.TestCase):
    def test_comparison_evaluates_same_file_and_head1_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "validation.bin"
            baseline = root / "baseline.pt"
            mtp = root / "best.pt"
            raw.write_bytes(bytes(range(16)))
            baseline.touch()
            mtp.touch()
            with patch.object(evaluate_mtp, "_cuda_device", return_value=torch.device("cpu")), \
                 patch.object(evaluate_mtp, "load_mamba3_checkpoint", return_value=(
                     CountingNextModel(), {"model_name": "mamba3_siso_3m"}
                 )), \
                 patch.object(evaluate_mtp, "load_mtp4_checkpoint", return_value=(
                     CountingFutureModel(), {"training_config": {"validation_batch_size": 2}}
                 )):
                result = evaluate_mtp.evaluate_comparison(
                    raw, mtp, baseline_checkpoint=baseline,
                    sequence_length=8, device="cpu"
                )
            self.assertEqual(result["batch_size"], 2)
            self.assertEqual(result["baseline"]["head1_predicted_bytes"], 15)
            self.assertEqual(result["mtp4"]["head1_predicted_bytes"], 15)
            self.assertAlmostEqual(result["baseline"]["head1_bpb"],
                                   result["mtp4"]["head1_bpb"], places=8)

    def test_entry_point_defaults_and_summary_mapping(self):
        args = train_mtp.parse_args([
            "--train-file", "train.bin", "--validation-file", "validation.bin",
            "--train-starts", "starts.npy",
        ])
        self.assertEqual(args.name, MTP4_MAMBA3_CONFIG["name"])
        self.assertEqual(args.micro_batch_size, 32)
        weighted_args = train_mtp_weighted.parse_args([
            "--train-file", "train.bin", "--validation-file", "validation.bin",
            "--train-starts", "starts.npy",
        ])
        self.assertEqual(weighted_args.name, "mamba3_siso_3m_mtp4_weighted")
        args = evaluate_mtp.parse_args([
            "--test-file", "validation.bin", "--mtp-checkpoint", "best.pt"
        ])
        self.assertEqual(args.sequence_length, 2048)
        self.assertEqual(args.name, "mtp4_comparison")
        rows = evaluate_mtp.summary_rows({
            "baseline": {"parameters": 3, "head1_bpb": 1.0},
            "mtp4": {"parameters": 4, **{
                f"head{i}_bpb": float(i) for i in range(1, 5)
            }},
        })
        self.assertEqual(rows[0], ("Baseline", 3, 1.0, None, None, None))
        self.assertEqual(rows[1], ("MTP-4", 4, 1.0, 2.0, 3.0, 4.0))

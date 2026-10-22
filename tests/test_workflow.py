"""当前主线的数据准备、续训、导出与入口约束。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from neural_compression import cli, data, training
from neural_compression.config import MODEL_CONFIG, DEFAULT_SOURCE
from neural_compression.io import load_json
from test_tokenization import toy_model


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(4096, 2)
        self.lm_head = torch.nn.Linear(2, 4096, bias=False)

    def forward(self, values):
        return self.lm_head(self.embedding(values)).float()


class WorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixtures(self, root):
        train, val, tokenizer = [
            root / name for name in ("train.raw", "validation.raw", "source.model")
        ]
        train.write_bytes(b"\xff" * 4099)
        val.write_bytes(b"\xfe" * 2051)
        tokenizer.write_bytes(toy_model())
        return dict(
            train_file=train,
            validation_file=val,
            tokenizer=tokenizer,
            batch_size=2,
            micro_batch_size=1,
            tokenizer_threads=1,
            device="cpu",
        )

    def test_command_defaults_and_removed_mode(self):
        args = cli.parse_args(["train", "--run-dir", "new"])
        self.assertEqual((args.batch_size, args.workers, args.source), (72, 0, None))
        for command in ("encode", "decode", "roundtrip", "evaluate"):
            with self.assertRaises(SystemExit):
                cli.parse_args([command, "--probability-transfer", "fp32"])

    def test_input_exclusivity_and_default_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixtures(root)
            with self.assertRaisesRegex(ValueError, "互斥"):
                data.training_inputs(
                    source=args["train_file"],
                    train_file=args["train_file"],
                    validation_file=args["validation_file"],
                )
            with self.assertRaisesRegex(ValueError, "同时"):
                data.training_inputs(train_file=args["train_file"])
            with self.assertRaisesRegex(ValueError, "110,000,000"):
                data.training_inputs(source=args["train_file"])
            source = root / "enwik9"
            with source.open("wb") as handle:
                handle.truncate(110_000_000)
            with patch("neural_compression.config.DEFAULT_SOURCE", source):
                mode, paths = data.training_inputs()
            self.assertEqual((mode, paths), ("enwik9", [source]))
            target = root / "slice"
            args["train_file"].write_bytes(bytes(range(100)))
            data._copy_range(args["train_file"], target, 10, 25)
            self.assertEqual(target.read_bytes(), bytes(range(10, 35)))

    def test_custom_complete_files_cache_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixtures(root)
            run = root / "run"
            first = data.prepare_training(run, **args)
            again = data.prepare_training(run, **args)
            self.assertEqual(first, again)
            self.assertEqual(first["datasets"]["train"]["raw_bytes"], 4099)
            self.assertEqual(first["datasets"]["validation"]["raw_bytes"], 2051)
            self.assertFalse((run / "data/test.tokens").exists())
            args["train_file"].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "变更"):
                data.prepare_training(run, **args)

    def test_changed_tokenizer_and_cache_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixtures(root)
            run = root / "run"
            data.prepare_training(run, **args)
            cache = run / "data/train.tokens"
            original = cache.read_bytes()
            cache.write_bytes(original[:-2])
            with self.assertRaisesRegex(ValueError, "缓存"):
                data.prepare_training(run, **args)
            cache.write_bytes(original)
            (run / "tokenizer.model").write_bytes(toy_model() + b"\0")
            with self.assertRaises(Exception):
                data.prepare_training(run, **args)

    def test_new_tokenizer_rejects_insufficient_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixtures(root)
            args.pop("tokenizer")
            args["train_file"].write_bytes(b"abc abc\n" * 30)
            with self.assertRaisesRegex(ValueError, "4096"):
                data.prepare_training(root / "run", **args)
            self.assertFalse((root / "run/tokenizer.model").exists())

    def test_short_training_resume_and_independent_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.fixtures(root)
            run = root / "run"
            with patch(
                "neural_compression.training.build_model",
                side_effect=lambda: (TinyModel(), dict(MODEL_CONFIG)),
            ):
                first = training.train(run, max_epochs=1, **args)
                latest = torch.load(run / "checkpoints/latest.pt", weights_only=False)
                self.assertEqual(first["epoch"], 1)
                self.assertEqual(latest["position"], 0)
                result = training.train(run, max_epochs=2, **args)
                self.assertEqual(result["epoch"], 2)
                self.assertGreater(result["updates"], first["updates"])
                self.assertFalse(result["converged"])
                self.assertEqual(result["stop_reason"], "epoch_limit")
                package = torch.load(run / "inference.pt", weights_only=False)
                self.assertNotIn("optimizer", package)
                self.assertEqual(package["vocab_size"], 4096)
                self.assertEqual(len(package["source_checkpoint_sha256"]), 64)
                self.assertTrue((run / "bpb_vs_steps.png").is_file())
                self.assertTrue(load_json(run / "preflight.json")["passed"])
                args["batch_size"] = 1
                with self.assertRaisesRegex(ValueError, "变更"):
                    training.train(run, max_epochs=3, **args)

    def test_output_collisions_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root / "input"
            inp.write_bytes(b"a")
            checkpoint = root / "checkpoint"
            checkpoint.write_bytes(b"b")
            tok = root / "tokenizer"
            tok.write_bytes(b"c")
            out = root / "output"
            out.symlink_to(inp)
            args = cli.parse_args(
                [
                    "encode",
                    "-i",
                    str(inp),
                    "-o",
                    str(out),
                    "-c",
                    str(checkpoint),
                    "-t",
                    str(tok),
                ]
            )
            with self.assertRaises(ValueError):
                cli.prepare_paths(args)
            out.unlink()
            out.write_bytes(b"old")
            with self.assertRaises(FileExistsError):
                cli.prepare_paths(args)

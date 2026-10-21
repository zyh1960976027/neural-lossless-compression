import copy
import hashlib
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sentencepiece import sentencepiece_model_pb2 as pb
import torch

from neural_compression import bpe_vocab_experiment as sweep
from neural_compression import token_training as training
from neural_compression import token_codec as tc
from neural_compression import codec as c
from neural_compression.codec_graph import BatchedStepV5Predictor
from neural_compression.model import build_mamba3_token_model, parameter_counts
from neural_compression.token_pipeline import inference_export, synthetic_preflight
from neural_compression.tokenization import LosslessTokenizer, load_json, save_json
from test_tokenization import toy_model
from test_token_codec import ControlledPredictor


def large_tokenizer(vocab):
    proto = pb.ModelProto()
    proto.ParseFromString(toy_model())
    while len(proto.pieces) < vocab - 1:
        piece = proto.pieces.add()
        piece.piece = f"<unused{len(proto.pieces)}>"
        piece.type = pb.ModelProto.SentencePiece.NORMAL
        piece.score = -10.
    piece = proto.pieces.add()
    piece.piece = "Z"
    piece.type = pb.ModelProto.SentencePiece.NORMAL
    piece.score = 0.
    proto.trainer_spec.vocab_size = vocab
    return LosslessTokenizer(proto.SerializeToString())


class TinyModel(torch.nn.Module):
    def __init__(self, vocab=16):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, 4)
        self.lm_head = torch.nn.Linear(4, vocab, bias=False)

    def forward(self, x):
        return self.lm_head(self.embedding(x))


class BpeVocabularyTest(unittest.TestCase):
    def test_evaluation_schedule_alternates_only_requested_vocabularies(self):
        args = sweep.parse_args(["evaluate", "--run-dir", "/tmp/bpe-test"])
        calls = []
        with patch.object(sweep, "export_group", side_effect=lambda g: Path(str(g.vocab_size))), \
             patch.object(sweep, "evaluate_case", side_effect=lambda g, s, c, **k: calls.append((g.vocab_size, s, k))), \
             patch.object(sweep, "report"):
            sweep.evaluate_groups(args)
        self.assertEqual([v for v, s, k in calls if "repeat" in k],
                         [8192, 16384, 16384, 8192, 8192, 16384])
        self.assertEqual(sum(k.get("profile", False) for _, _, k in calls), 2)
        self.assertEqual([(v, s) for v, s, k in calls if s != "validation"],
                         [(8192, "test"), (8192, "full"), (16384, "test"), (16384, "full")])

    def test_run_binding_rejects_changed_vocabulary_and_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = sweep.parse_args(["prepare", "--run-dir", temporary, "--source", str(Path(temporary)/"source")])
            with args.source.open("wb") as handle:
                handle.truncate(1_000_000_000)
            historical = load_json(sweep.ARCHIVES["results"])
            with patch.object(sweep, "digest_file", return_value=historical["manifest"]["source_sha256"]), \
                 patch.object(sweep, "code_fingerprint", return_value="first-code"):
                sweep.bind_run(args)
                sweep.bind_run(args)
                args.vocab_sizes = [8192]
                with self.assertRaisesRegex(ValueError, "configuration"):
                    sweep.bind_run(args)
                args.vocab_sizes = [8192, 16384]
                with patch.object(sweep, "code_fingerprint", return_value="changed-code"), \
                     self.assertRaisesRegex(ValueError, "configuration"):
                    sweep.bind_run(args)

    def test_incomplete_report_only_reads_bpe_history_and_marks_missing_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = sweep.parse_args(["report", "--run-dir", temporary])
            save_json(args.run_dir / "sweep_manifest.json", dict(vocab_sizes=args.vocab_sizes))
            with patch.object(sweep, "run_job", side_effect=AssertionError("History must not be retested")):
                self.assertFalse(sweep.report(args))
            result = load_json(args.run_dir / "results.json")
            self.assertFalse(result["baseline_retested"])
            self.assertEqual(result["results"], [])
            self.assertEqual(len(result["baseline_4096"]), 3)
            self.assertTrue(all(r["algorithm"] == "bpe" for r in result["baseline_4096"]))
            self.assertIn("实验尚未完成", (args.run_dir / "实验报告.md").read_text())

    def test_completed_report_checks_reference_ranges_runtime_and_repetition(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = sweep.parse_args(["report", "--run-dir", temporary])
            save_json(args.run_dir / "sweep_manifest.json", dict(vocab_sizes=args.vocab_sizes))
            sweep.report(args)
            references = {r["split"]: r for r in load_json(args.run_dir / "results.json")["baseline_4096"]}
            runtime = load_json(sweep.ARCHIVES["performance"])["runtime"]
            for group in sweep.groups(args):
                save_json(group.run_dir / "bpe/training/summary.json", dict(converged=True,
                          parameter_counts=dict(total_trainable_params=9042380), effective_batch_size=72,
                          micro_batch_size=12, epoch=20, elapsed_seconds=1200., best_bpb=1.3,
                          stop_reason="validation_plateau"))
                for split in ("validation", "test", "full"):
                    ref = references[split]
                    encoded = dict(compressed_sha256="packed", num_bytes=ref["original_bytes"],
                                   num_tokens=ref["original_bytes"]//3, actual_bpb=1.2, compressed_bytes=1500000,
                                   runtime=runtime, peak_allocated_bytes=100, peak_reserved_bytes=200,
                                   model_load_seconds=1., warmup_seconds=1., process_seconds=3., encode_seconds=1.)
                    decoded = dict(encoded, decode_seconds=1., files_identical=True, decoded_sha256=ref["raw_sha256"])
                    trial = dict(verified=True, encode=encoded, decode=decoded, raw_sha256=ref["raw_sha256"],
                                 checkpoint_sha256="checkpoint", inference_package_bytes=100000,
                                 including_weights_bpb=1.3)
                    names = ["profile", "repeat1", "repeat2", "repeat3"] if split == "validation" else ["single"]
                    for name in names:
                        save_json(group.run_dir / f"evaluation/bpe/{split}/{name}/roundtrip.json", trial)
            self.assertTrue(sweep.report(args))
            result = load_json(args.run_dir / "results.json")
            self.assertEqual(len(result["results"]), 6)
            for row in result["results"]:
                self.assertEqual(row["speed_comparison"] == "historical_same_runtime", row["split"] == "validation")
                self.assertEqual(row["repeats"], 3 if row["split"] == "validation" else 1)
            path = args.run_dir / "v8192/evaluation/bpe/validation/repeat2/roundtrip.json"
            stale = load_json(path)
            stale["encode"]["compressed_sha256"] = "changed"
            save_json(path, stale)
            with self.assertRaisesRegex(ValueError, "different streams"):
                sweep.report(args)

    def test_entry_only_selects_new_bpe_groups_in_fixed_order(self):
        args = sweep.parse_args(["prepare", "--run-dir", "/tmp/bpe-test", "--vocab-sizes", "16384", "8192"])
        self.assertEqual([g.vocab_size for g in sweep.groups(args)], [8192, 16384])
        self.assertTrue(all(g.algorithms == ("bpe",) for g in sweep.groups(args)))
        for values in (["4096"], ["8192", "8192"]):
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                sweep.parse_args(["all", "--run-dir", "/tmp/bpe-test", "--vocab-sizes", *values])

    def test_accumulation_matches_full_update_and_normalizes_tail(self):
        torch.manual_seed(2026)
        values = torch.randint(0, 16, (5, 7))
        targets = torch.randint(0, 16, (5, 7))
        masks = torch.ones(5, 7, dtype=torch.bool)
        masks[-1, 2:] = False
        masks[-2] = False
        model = TinyModel()
        reference = copy.deepcopy(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        other = torch.optim.Adam(reference.parameters(), lr=.001)
        loss, count = training.accumulated_update(model, optimizer, (values, targets, masks), 2, torch.device("cpu"))
        expected, n = training.accumulated_update(reference, other, (values, targets, masks), 5, torch.device("cpu"))
        self.assertEqual(count, n)
        self.assertEqual(count, 23)
        self.assertAlmostEqual(loss, expected, places=6)
        for left, right in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(left, right, atol=2e-7, rtol=1e-5)
        with patch.object(optimizer, "step") as step:
            loss, count = training.accumulated_update(model, optimizer,
                (values, targets, torch.zeros_like(masks)), 2, torch.device("cpu"))
            self.assertEqual((loss, count), (0., 0))
            step.assert_not_called()

    def test_resume_matches_uninterrupted_accumulation_and_rejects_micro_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            dirs = [Path(temporary) / name for name in ("full", "resumed")]
            tokenizer = LosslessTokenizer(toy_model())
            for directory in dirs:
                directory.mkdir()
                (directory / "tokenizer.model").write_bytes(tokenizer.model_bytes)
                (np.arange(4099, dtype=np.uint16) % tokenizer.vocab_size).astype("<u2").tofile(directory / "train.tokens")
                (np.arange(25, dtype=np.uint16) % tokenizer.vocab_size).astype("<u2").tofile(directory / "validation.tokens")
                save_json(directory / "validation.tokens.json", dict(raw_bytes=50))
            def builder(vocab):
                return TinyModel(vocab), dict(vocab_size=vocab)
            settings = dict(training.TRAINING, stop_patience=2, threshold_bpb=1000., checkpoint_seconds=0.)
            with patch.object(training, "build_mamba3_token_model", side_effect=builder), \
                 patch.object(training, "TRAINING", settings), patch("torch.cuda.set_device"), \
                 patch.object(training, "_plot_bpb_history"), patch("builtins.print"):
                training.train(dirs[0], 2, "cpu", 0, micro_batch_size=1)
                original_update = training.accumulated_update
                def interrupt(*args, **kwargs):
                    result = original_update(*args, **kwargs)
                    signal.raise_signal(signal.SIGTERM)
                    return result
                with patch.object(training, "accumulated_update", side_effect=interrupt):
                    partial = training.train(dirs[1], 2, "cpu", 0, micro_batch_size=1)
                self.assertFalse(partial["converged"])
                self.assertEqual(partial["position"], 2)
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    training.train(dirs[1], 2, "cpu", 0, micro_batch_size=2)
                training.train(dirs[1], 2, "cpu", 0, micro_batch_size=1)
            states = [torch.load(d / "training/latest.pt", weights_only=False) for d in dirs]
            self.assertEqual(states[0]["updates"], states[1]["updates"])
            self.assertEqual(states[0]["tokens_seen"], states[1]["tokens_seen"])
            self.assertEqual(states[0]["control"], states[1]["control"])
            for name in states[0]["model"]:
                torch.testing.assert_close(states[0]["model"][name], states[1]["model"][name], atol=0, rtol=0)

    def test_large_models_export_with_actual_vocab_and_exact_parameter_counts(self):
        from neural_compression import model as model_module
        with tempfile.TemporaryDirectory() as temporary:
            # Full model construction is already covered by test_model. Check exports
            # separately using tiny weights and the real vocabulary-bearing metadata.
            for vocab, expected in ((8192, 9042380), (16384, 14809548)):
                counts = model_module.analytic_parameter_counts(model_module.BASELINE_MAMBA3_CONFIG, vocab)
                self.assertEqual(counts["total_trainable_params"], expected)
                source, target = Path(temporary) / f"{vocab}.pt", Path(temporary) / f"{vocab}.inference.pt"
                model = TinyModel(vocab)
                config = dict(name=f"test_v{vocab}", vocab_size=vocab)
                torch.save(dict(model=model.state_dict(), model_config=config, vocab_size=vocab,
                                tokenizer_sha256="1" * 64, optimizer={"must_be_removed": True}), source)
                inference_export(source, target, "bpe")
                state = torch.load(target, weights_only=False)
                self.assertEqual(state["vocab_size"], vocab)
                self.assertNotIn("optimizer", state)
                with patch.object(model_module, "build_mamba3_token_model", return_value=(TinyModel(vocab), config)), \
                     patch.object(tc, "BatchedStepV5Predictor") as predictor:
                    tc.load_token_predictor(target, "cpu", 2)
                    self.assertEqual(predictor.call_args.args[0].embedding.num_embeddings, vocab)

    def test_large_vocab_highest_id_roundtrip_tail_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for vocab in sweep.VOCAB_SIZES:
                tokenizer = large_tokenizer(vocab)
                self.assertEqual(int(tokenizer.encode(b"Z")[0]), vocab - 1)
                for size in (0, 1, 2049, 4099):
                    raw, packed, decoded = [root / f"{vocab}.{size}.{suffix}" for suffix in ("raw", "packed", "decoded")]
                    raw.write_bytes(b"Z" * size)
                    tc.encode(raw, packed, ControlledPredictor(tokenizer), tokenizer)
                    tc.decode(packed, decoded, ControlledPredictor(tokenizer))
                    self.assertEqual(raw.read_bytes(), decoded.read_bytes())
                broken = root / f"{vocab}.broken"
                data = packed.read_bytes()
                packed.write_bytes(data[:-3])
                with self.assertRaises(ValueError):
                    tc.decode(packed, broken, ControlledPredictor(tokenizer))
                self.assertFalse(broken.exists())

    def test_large_probability_quantization_including_zero_mass(self):
        rows = synthetic_preflight(sweep.VOCAB_SIZES)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["verified"] for row in rows))


@unittest.skipUnless(os.environ.get("MAMBA_CODEC_GPU_TEST") == "1" and torch.cuda.is_available(), "opt-in real CUDA test")
class BpeVocabularyGpuTest(unittest.TestCase):
    def test_both_vocabularies_full_recurrence_high_ids_prefetch_and_reset(self):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        for vocab in sweep.VOCAB_SIZES:
            torch.manual_seed(2026)
            model, _ = build_mamba3_token_model(vocab)
            with torch.no_grad():
                model.lm_head.weight.normal_(std=.02)
            predictor = BatchedStepV5Predictor(model, "cuda:0", "0" * 64, 3, check_probabilities=True)
            rng = np.random.default_rng(2026)
            lengths = np.array([2048, 2048, 7])
            values = rng.integers(0, vocab, (3, 2048), dtype=np.uint16)
            values[0, 0] = vocab - 1
            with c._coding_mode(predictor):
                predictor.reset(3)
                serial = hashlib.sha256()
                for position in range(1, 2048):
                    p, _ = predictor.step(np.where(lengths > position, values[:, position-1], 0))
                    self.assertEqual(p.shape, (3, vocab))
                    serial.update(p.tobytes())
                with self.assertRaises(ValueError):
                    predictor.step(np.zeros(3, dtype=np.int32))
                predictor.reset(3)
                parallel = hashlib.sha256()
                for p, _ in predictor.iter_prefetched(values, lengths):
                    parallel.update(p.tobytes())
                self.assertEqual(serial.digest(), parallel.digest())
                for size in (1, 2, 3):
                    predictor.reset(size)
                    predictor.step(np.full(size, vocab-1))
                predictor.reset(3)
                iterator = predictor.iter_prefetched(values, lengths)
                next(iterator)
                iterator.close()
                predictor.reset(1)
                predictor.step(np.array([vocab-1]))
                with torch.no_grad():
                    model.lm_head.weight.fill_(float("nan"))
                predictor.reset(1)
                with self.assertRaises(ValueError):
                    predictor.step(np.array([vocab-1]))
            del predictor, model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()

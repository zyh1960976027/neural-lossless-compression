import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np
import torch

import codec as cli
from mamba3_byte_compress import codec as c


class FakeMTP4Predictor:
    profile = c.MTP4_PROFILE

    def __init__(self, batch_size=2):
        self.batch_size = batch_size
        self.checkpoint_sha256 = "0" * 64
        self.runtime = {"fake_mtp4": 1}
        self.model = torch.nn.Identity()
        self.active_batch_size = 0
        self.calls = []

    def synchronize(self):
        pass

    def reset(self, batch_size=None):
        self.active_batch_size = batch_size or self.batch_size
        self.state = np.zeros(self.active_batch_size, dtype=np.int64)
        self.calls.append(("reset", self.active_batch_size))

    def _probabilities(self):
        result = np.ones((self.active_batch_size, 4, 256), dtype=np.float32)
        for lane in range(self.active_batch_size):
            for head in range(4):
                result[lane, head, (self.state[lane] + head + 1) % 256] = 9
        result /= result.sum(axis=-1, keepdims=True)
        return result

    def start(self, seeds, *, active=None):
        seeds = np.asarray(seeds, dtype=np.int64)
        self.state[:] = seeds
        self.calls.append((
            "start", seeds.tolist(), np.asarray(active, dtype=bool).tolist()
        ))
        return self._probabilities()

    def advance(self, tokens, *, active=None):
        tokens = np.asarray(tokens, dtype=np.int64)
        for column in range(4):
            self.state[:] = (self.state * 17 + tokens[:, column]) % 256
        self.calls.append((
            "advance", tokens.tolist(), np.asarray(active, dtype=bool).tolist()
        ))
        return self._probabilities()


class MTP4CodecTest(unittest.TestCase):
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

    def roundtrip(self, count, batch_size=2):
        raw = self.root / f"{count}.raw"
        packed = self.root / f"{count}.m3rc"
        decoded = self.root / f"{count}.decoded"
        raw.write_bytes(bytes(index % 256 for index in range(count)))
        encoder = FakeMTP4Predictor(batch_size)
        result = c.encode_file(raw, packed, encoder, file_format="v6")
        decoder = FakeMTP4Predictor(batch_size)
        decoded_result = c.decode_file(packed, decoded, decoder)
        self.assertEqual(decoded.read_bytes(), raw.read_bytes())
        self.assertEqual(encoder.calls, decoder.calls)
        self.assertEqual(c.read_codec_settings(packed), {
            "inference": "mtp4-batched-sequence", "batch_size": batch_size,
        })
        return result, decoded_result, packed

    def test_boundaries_roundtrip_and_metrics(self):
        large_result = None
        for count in (0, 1, 2, 4, 5, 6, 2047, 2048, 2049, 4098, 8193):
            with self.subTest(count=count):
                result, decoded, _ = self.roundtrip(count)
                self.assertEqual(result["file_format"], "v6")
                self.assertEqual(result["format_version"], 6)
                self.assertEqual(result["head1_predicted_bytes"]
                                 + result["head2_predicted_bytes"]
                                 + result["head3_predicted_bytes"]
                                 + result["head4_predicted_bytes"],
                                 result["predicted_bytes"])
                self.assertEqual(result["num_model_forwards"],
                                 decoded["num_model_forwards"])
                if count == 4098:
                    large_result = result
        result = large_result
        self.assertEqual(result["num_blocks"], 3)
        self.assertEqual(result["num_batches"], 2)
        self.assertEqual(result["num_prediction_groups"], 1025)
        self.assertEqual(result["num_model_forwards"], 513)
        self.assertEqual(result["state_tokens_processed"], 4091)
        self.assertEqual(
            [result[f"head{i}_predicted_bytes"] for i in range(1, 5)],
            [1025, 1024, 1024, 1022],
        )

    def test_all_byte_values_and_repeated_encoding(self):
        raw = self.root / "all.raw"
        first = self.root / "first.m3rc"
        second = self.root / "second.m3rc"
        decoded = self.root / "decoded"
        raw.write_bytes(bytes(range(256)) * 9)
        c.encode_file(raw, first, FakeMTP4Predictor(), file_format="v6")
        c.encode_file(raw, second, FakeMTP4Predictor(), file_format="v6")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        c.decode_file(first, decoded, FakeMTP4Predictor())
        self.assertEqual(decoded.read_bytes(), raw.read_bytes())

    def test_optional_phase_timing_is_complete(self):
        raw = self.root / "timed.raw"
        packed = self.root / "timed.m3rc"
        decoded = self.root / "timed.decoded"
        raw.write_bytes(bytes(range(251)) * 17)
        encoder = FakeMTP4Predictor()
        encoder.collect_phase_timings = True
        encode_result = c.encode_file(raw, packed, encoder, file_format="v6")
        decoder = FakeMTP4Predictor()
        decoder.collect_phase_timings = True
        decode_result = c.decode_file(packed, decoded, decoder)
        self.assertEqual(decoded.read_bytes(), raw.read_bytes())
        self.assertEqual(
            set(encode_result["phase_seconds"]),
            {
                "input_read_hash_pack", "state_reset", "predictor",
                "probability_cpu", "range_coding", "container_write", "other",
            },
        )
        self.assertEqual(
            set(decode_result["phase_seconds"]),
            {
                "container_read_and_allocate", "state_reset", "predictor",
                "probability_cpu", "range_coding", "output_write_hash", "other",
            },
        )
        for operation, result in (("encode", encode_result), ("decode", decode_result)):
            self.assertAlmostEqual(
                sum(result["phase_seconds"].values()),
                result[f"{operation}_seconds"],
                places=9,
            )

    def test_header_payload_digest_and_configuration_corruption(self):
        raw = self.root / "raw"
        packed = self.root / "packed"
        decoded = self.root / "decoded"
        raw.write_bytes(b"MTP four-token continuation" * 3)
        c.encode_file(raw, packed, FakeMTP4Predictor(), file_format="v6")
        original = packed.read_bytes()
        with self.assertRaisesRegex(ValueError, "batch_size mismatch"):
            c.decode_file(packed, decoded, FakeMTP4Predictor(1))
        self.assertFalse(decoded.exists())
        variants = []
        flags = bytearray(original)
        flags[9] = 1
        flags[c.BATCHED_HEADER.size:c.BATCHED_HEADER.size + c.CRC32.size] = (
            c.CRC32.pack(zlib.crc32(flags[:c.BATCHED_HEADER.size]))
        )
        variants.append(flags)
        payload = bytearray(original)
        payload[c.BATCHED_HEADER.size + c.CRC32.size + c.FRAME.size] ^= 1
        variants.extend((payload, original[:-1], original + b"x"))
        digest = bytearray(original)
        digest[-1] ^= 1
        variants.append(digest)
        for index, data in enumerate(variants):
            bad = self.root / f"bad-{index}"
            bad.write_bytes(data)
            with self.subTest(index=index), self.assertRaises(ValueError):
                c.decode_file(bad, decoded, FakeMTP4Predictor())
            self.assertFalse(decoded.exists())

    def test_cli_selects_v6_and_rejects_incompatible_pairs(self):
        with patch.object(cli, "load_predictor") as load, \
             patch.object(cli, "encode_file", return_value={}), \
             contextlib.redirect_stdout(io.StringIO()):
            cli.main([
                "encode", "--input", "input.bin", "--output", str(self.root / "out"),
                "--format", "v6", "--batch-size", "7",
            ])
        self.assertEqual(load.call_args.kwargs, {
            "inference": "mtp4-batched-sequence", "batch_size": 7,
            "execution": "auto",
        })
        with patch.object(cli, "load_predictor") as load, \
             patch.object(cli, "encode_file", return_value={}), \
             contextlib.redirect_stdout(io.StringIO()):
            cli.main([
                "encode", "--input", "input.bin", "--output", str(self.root / "seq"),
                "--format", "v6", "--batch-size", "7", "--execution", "sequence",
            ])
        self.assertEqual(load.call_args.kwargs["execution"], "sequence")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            cli.main([
                "encode", "--input", "input.bin", "--output", str(self.root / "wrong"),
                "--inference", "mtp4-batched-sequence", "--format", "v5",
            ])


@unittest.skipUnless(
    os.environ.get("MAMBA_MTP_CODEC_GPU_TEST") == "1",
    "opt-in real MTP-4 CUDA test",
)
class MTP4GPUVerification(unittest.TestCase):
    equal_checkpoint = (
        "experiments/train/20260918_175712_mamba3_siso_3m_mtp4/"
        "checkpoints/best.pt"
    )
    weighted_checkpoint = (
        "experiments/train/20260921_160009_mamba3_siso_3m_mtp4_weighted/"
        "checkpoints/best.pt"
    )

    def test_equal_and_weighted_checkpoint_roundtrips(self):
        for checkpoint in (self.equal_checkpoint, self.weighted_checkpoint):
            with self.subTest(checkpoint=checkpoint), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                raw, packed, decoded = (root / name for name in ("raw", "packed", "decoded"))
                raw.write_bytes(bytes(range(37)))
                predictor = c.load_predictor(
                    checkpoint, inference="mtp4-batched-sequence", batch_size=2
                )
                result = c.roundtrip_file(
                    raw, packed, decoded, predictor, file_format="v6"
                )
                self.assertTrue(result["files_identical"])

    def test_continuation_matches_step_with_bf16_tolerance(self):
        from mamba3_byte_compress.metrics import autocast_context

        predictor = c.load_predictor(
            self.equal_checkpoint, inference="mtp4-batched-sequence", batch_size=1
        )
        model = predictor.model
        with c._coding_mode(predictor):
            predictor.reset(1)
            predictor.start(np.array([3], dtype=np.uint8))
            fused = predictor.advance(
                np.array([[17, 29, 41, 53]], dtype=np.uint8)
            ).copy()
            fused_states = [
                [state.clone() for state in layer] for layer in predictor.states
            ]
            step_states = [
                [
                    state.clone().zero_()
                    for state in block.mixer.allocate_inference_cache(
                        1, c.BLOCK_SIZE, dtype=torch.bfloat16
                    )
                ]
                for block in model.blocks
            ]
            for token in (3, 17, 29, 41, 53):
                value = torch.tensor([token], device=predictor.device)
                with autocast_context(predictor.device):
                    x = model.embedding(value)
                    for block, states in zip(model.blocks, step_states):
                        mixed, *_ = block.mixer.step(block.norm1(x), *states)
                        x = x + mixed
                        x = x + block.mlp(block.norm2(x))
                    hidden = model.final_norm(x)
                    logits = torch.stack(
                        [
                            model.lm_head(hidden),
                            *(head(hidden) for head in model.future_heads),
                        ],
                        dim=1,
                    ).float()
            reference = logits.softmax(dim=-1).cpu().numpy()
        probability_error = np.abs(fused - reference)
        self.assertLessEqual(float(probability_error.max()), 5e-2)
        self.assertLessEqual(float(probability_error.mean()), 5e-4)
        for fused_layer, step_layer in zip(fused_states, step_states):
            angle_error = torch.atan2(
                torch.sin(fused_layer[0] - step_layer[0]),
                torch.cos(fused_layer[0] - step_layer[0]),
            ).abs().max()
            self.assertLessEqual(float(angle_error), 1e-1)
            for fused_state, step_state in zip(fused_layer[1:], step_layer[1:]):
                self.assertLessEqual(
                    float((fused_state.float() - step_state.float()).abs().max()),
                    1e-1,
                )

    def test_graph_matches_sequence_for_full_block_and_batch_switches(self):
        from mamba3_byte_compress.codec_graph import MTP4GraphPredictor

        sequence = c.load_predictor(
            self.equal_checkpoint, inference="mtp4-batched-sequence",
            batch_size=2, execution="sequence",
        )
        graph = MTP4GraphPredictor(
            sequence.model, sequence.device, sequence.checkpoint_sha256, 2
        )

        def states_equal():
            return all(
                torch.equal(left, right)
                for left_layer, right_layer in zip(sequence.states, graph.states)
                for left, right in zip(left_layer, right_layer)
            )

        with c._coding_mode(sequence):
            sequence.reset(2)
            graph.reset(2)
            seeds = np.array([3, 241], dtype=np.uint8)
            np.testing.assert_array_equal(
                sequence.start(seeds), graph.start(seeds)
            )
            self.assertTrue(states_equal())

            # One seed plus 511 groups advances state through byte 2045 and
            # produces all four distributions needed for a 2048-byte block.
            for group in range(511):
                tokens = (
                    np.arange(8, dtype=np.int64).reshape(2, 4) * 17
                    + group * 31
                ) % 256
                active = np.array([True, group < 510])
                np.testing.assert_array_equal(
                    sequence.advance(tokens, active=active),
                    graph.advance(tokens, active=active),
                )
                self.assertTrue(states_equal())

            for size in (1, 2, 1):
                sequence.reset(size)
                graph.reset(size)
                seeds = np.arange(size, dtype=np.uint8) + 11
                np.testing.assert_array_equal(
                    sequence.start(seeds), graph.start(seeds)
                )
                self.assertTrue(states_equal())
            self.assertLessEqual(len(graph._contexts), 2)


if __name__ == "__main__":
    unittest.main()

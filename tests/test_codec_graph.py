"""Container regression plus opt-in real CUDA graph/state verification."""
import os
from pathlib import Path
import tempfile
import unittest
import zlib

import numpy as np
import torch

from mamba3_byte_compress import codec as c
try:
    from .test_codec_step import StatefulModel
except ImportError:  # unittest discovery adds tests/ directly to sys.path.
    from test_codec_step import StatefulModel


class ContainerPredictor(c.BatchedStepV4Predictor):
    profile = c.BATCHED_V5_PROFILE


class V5ContainerTest(unittest.TestCase):
    def test_boundaries_and_header(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for count in (0, 1, 2048, 2049, 4098, 8193):
                with self.subTest(count=count):
                    raw, packed, decoded = [root / f'{count}.{suffix}' for suffix in ('raw', 'm3rc', 'out')]
                    raw.write_bytes(bytes(i % 256 for i in range(count)))
                    p = ContainerPredictor(StatefulModel(), 'cpu', '0' * 64, 2)
                    c.encode_file(raw, packed, p)
                    self.assertEqual(c.read_codec_settings(packed), dict(inference='batched-step-v5', batch_size=2))
                    q = ContainerPredictor(StatefulModel(), 'cpu', '0' * 64, 2)
                    c.decode_file(packed, decoded, q)
                    self.assertEqual(raw.read_bytes(), decoded.read_bytes())
                    with self.assertRaisesRegex(ValueError, 'profile mismatch'):
                        c.decode_file(packed, root / 'wrong', c.BatchedStepV4Predictor(StatefulModel(), 'cpu', '0' * 64, 2))
                    corrupt = bytearray(packed.read_bytes())
                    corrupt[9] ^= 1
                    packed.write_bytes(corrupt)
                    with self.assertRaisesRegex(ValueError, 'CRC32'):
                        c.read_codec_settings(packed)

    def test_cli_selects_v5(self):
        from unittest.mock import patch
        import codec as cli
        for args in ([], ['--format', 'v5'], ['--inference', 'batched-step-v5']):
            with patch.object(cli, 'load_predictor') as load, patch.object(cli, 'encode_file', return_value={}):
                cli.main(['encode', '--input', 'input.bin', '--output', '/tmp/v5-cli-test-unused', *args])
                self.assertEqual(load.call_args.kwargs['inference'], 'batched-step-v5')

    def test_corruption_rejected_without_output(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, packed, decoded = (root / name for name in ('raw', 'packed', 'decoded'))
            raw.write_bytes(b'known byte history')
            make = lambda: ContainerPredictor(StatefulModel(), 'cpu', '0' * 64, 2)
            c.encode_file(raw, packed, make())
            original = packed.read_bytes()
            flags = bytearray(original)
            flags[9] = 1
            flags[c.BATCHED_HEADER.size:c.BATCHED_HEADER.size + 4] = c.CRC32.pack(
                zlib.crc32(flags[:c.BATCHED_HEADER.size]))
            payload = bytearray(original)
            payload[c.BATCHED_HEADER.size + 4 + c.FRAME.size] ^= 1
            digest = bytearray(original)
            digest[-1] ^= 1
            for data in (original[:-1], original + b'x', flags, payload, digest):
                with self.subTest(data=bytes(data).hex()[:24]):
                    packed.write_bytes(data)
                    with self.assertRaises(ValueError):
                        c.decode_file(packed, decoded, make())
                    self.assertFalse(decoded.exists())


@unittest.skipUnless(os.environ.get('MAMBA_CODEC_GPU_TEST') == '1', 'opt-in CUDA/checkpoint test')
class GraphGPUVerification(unittest.TestCase):
    def test_full_recurrence_reset_shapes_and_eviction(self):
        torch.set_num_threads(1)
        p = c.load_predictor('checkpoints/best-3M.pt', inference='batched-step-v5', batch_size=1024)
        from mamba3_byte_compress.codec_graph import BatchedStepV5Predictor
        eager = BatchedStepV5Predictor(p.model, p.device, p.checkpoint_sha256, 1024, use_graph=False)
        reference = c.BatchedStepV4Predictor(p.model, p.device, p.checkpoint_sha256, 1024)
        max_error = 0.0
        with c._coding_mode(p):
            for size, steps in ((1024, 2047), (787, 2047), (1024, 8), (1, 8), (2, 8), (787, 8)):
                p.reset(size)
                eager.reset(size)
                reference.reset(size)
                self.assertLessEqual(len(p._contexts), 2)
                for layer in p.states:
                    self.assertTrue(all(torch.count_nonzero(v).item() == 0 for v in layer))
                for position in range(steps):
                    tokens = ((np.arange(size) * 17 + position * 31) % 256).astype(np.uint8)
                    a = p.step(tokens)[0].copy()
                    b = eager.step(tokens)[0]
                    np.testing.assert_array_equal(a, b)
                    r = reference.step(tokens)[0]
                    max_error = max(max_error, float(np.max(np.abs(a-r))))
                    if position in (0, steps - 1):
                        self.assertTrue(all(torch.equal(a,b) for aa,bb in zip(p.states,eager.states) for a,b in zip(aa,bb)))
                if steps == 2047:
                    with self.assertRaisesRegex(ValueError, '2047'):
                        p.step(tokens)
            p.reset(2)
            for invalid in ([256, 1], [-1, 1], [1.5, 2.5], [1]):
                with self.assertRaises(ValueError):
                    p.step(invalid)
            with self.assertRaises(ValueError):
                p.step([1, 2], active=[True])
        print('GPU graph/eager probability and state equality: passed; v4 max probability error:', max_error)


if __name__ == '__main__':
    unittest.main()

"""Old/new byte paths retain identical containers across group/tail boundaries."""
from pathlib import Path
import tempfile
import unittest
import torch

from neural_compression.codec_batched_v4 import encode_batched_v4_file, decode_batched_v4_file
from test_codec_graph import ContainerPredictor
from test_codec_step import StatefulModel


class BytePerformanceTest(unittest.TestCase):
    def test_identical_streams_and_cross_decode(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for size in (0, 1, 2048, 2049, 4095, 8195):
                raw = root / "raw"
                raw.write_bytes(bytes(i % 256 for i in range(size)))
                streams = []
                for optimized in (False, True):
                    make = lambda: ContainerPredictor(StatefulModel(), "cpu", "0" * 64, 2)
                    packed = root / f"{size}.{optimized}.packed"
                    decoded = root / f"{size}.{optimized}.decoded"
                    encode_batched_v4_file(raw, packed, make(), optimized=optimized)
                    streams.append(packed.read_bytes())
                    decode_batched_v4_file(packed, decoded, make(), optimized=not optimized)
                    self.assertEqual(raw.read_bytes(), decoded.read_bytes())
                self.assertEqual(*streams)

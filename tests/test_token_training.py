from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from neural_compression.data import TokenBlockDataset
from neural_compression.training import PlateauControl
from neural_compression.evaluation import block_loss, validate


class BlockTrainingTest(unittest.TestCase):
    def test_independent_targets_and_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens"
            values = np.arange(4097, dtype=np.uint16) % 4096
            values.astype("<u2").tofile(path)
            data = TokenBlockDataset(path)
            self.assertEqual(len(data), 3)
            for block in range(2):
                inputs, targets, mask = data[block]
                self.assertEqual(int(mask.sum()), 2047)
                self.assertFalse(bool(mask[-1]))
                self.assertEqual(int(targets[2046]), int(values[block * 2048 + 2047]))
                self.assertEqual(int(inputs[0]), int(values[block * 2048]))
            self.assertEqual(int(data[2][2].sum()), 0)

    def test_loss_mask_blocks_cross_boundary_gradient(self):
        logits = torch.zeros(1, 4, 8, requires_grad=True)
        nll, count = block_loss(
            logits,
            torch.ones(1, 4, dtype=torch.long),
            torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
        )
        nll.backward()
        self.assertEqual(int(count), 3)
        self.assertEqual(float(logits.grad[0, 3].abs().sum()), 0.0)

    def test_plateau_accumulates_small_improvements_and_resumes(self):
        control = PlateauControl()
        self.assertTrue(control.update(2.0)["best"])
        control.update(1.9996)
        control.update(1.9992)
        self.assertEqual(control.bad_epochs, 2)
        control.update(1.9988)
        self.assertEqual(control.bad_epochs, 0)
        reductions = []
        for i in range(1, 16):
            event = control.update(1.9988)
            if event["reduce"]:
                reductions.append(i)
            self.assertEqual(event["stop"], i == 15)
            resumed = PlateauControl()
            resumed.load_state_dict(control.state_dict())
            control = resumed
        self.assertEqual(reductions, [5, 10, 15])

    def test_uniform_bpb_includes_each_block_seed(self):
        class Uniform(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(4096, 1)

            def forward(self, x):
                return torch.zeros((*x.shape, 4096))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens"
            np.arange(9, dtype="<u2").tofile(path)
            result = validate(
                Uniform(),
                TokenBlockDataset(path, block_size=4),
                18,
                2,
                torch.device("cpu"),
            )
            self.assertEqual(result["predicted_tokens"], 6)
            self.assertAlmostEqual(result["bpb"], 6.0, places=5)


if __name__ == "__main__":
    unittest.main()

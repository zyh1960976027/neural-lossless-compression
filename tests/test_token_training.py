from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from mamba3_byte_compress.token_training import TokenBlockDataset, PlateauControl, block_loss, validate


class BlockTrainingTest(unittest.TestCase):
    def test_probe_recognizes_triton_oom_without_hiding_other_failures(self):
        from mamba3_byte_compress.token_pipeline import is_cuda_out_of_memory
        self.assertTrue(is_cuda_out_of_memory(torch.cuda.OutOfMemoryError("allocation failed")))
        self.assertTrue(is_cuda_out_of_memory(RuntimeError("Triton Error [CUDA]: out of memory")))
        self.assertFalse(is_cuda_out_of_memory(RuntimeError("CUDA error: an illegal memory access was encountered")))
        self.assertFalse(is_cuda_out_of_memory(RuntimeError("Non-finite training loss")))

    @unittest.skipUnless(Path("checkpoints/best-3M.pt").exists(), "local byte checkpoint unavailable")
    def test_inference_package_is_a_loadable_checkpoint(self):
        from mamba3_byte_compress.token_pipeline import inference_export
        from mamba3_byte_compress.trainer import load_mamba3_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "inference.pt"
            size = inference_export(Path("checkpoints/best-3M.pt"), target, "byte")
            self.assertEqual(size, target.stat().st_size)
            model, state = load_mamba3_checkpoint(target)
            self.assertEqual(model.embedding.num_embeddings, 256)
            self.assertNotIn("optimizer", state)
            self.assertEqual(len(state["source_checkpoint_sha256"]), 64)

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
                self.assertEqual(int(targets[2046]), int(values[block*2048+2047]))
                self.assertEqual(int(inputs[0]), int(values[block*2048]))
            self.assertEqual(int(data[2][2].sum()), 0)

    def test_loss_mask_blocks_cross_boundary_gradient(self):
        logits = torch.zeros(1, 4, 8, requires_grad=True)
        nll, count = block_loss(logits, torch.ones(1, 4, dtype=torch.long), torch.tensor([[1, 1, 1, 0]], dtype=torch.bool))
        nll.backward()
        self.assertEqual(int(count), 3)
        self.assertEqual(float(logits.grad[0, 3].abs().sum()), 0.)

    def test_plateau_accumulates_small_improvements_and_resumes(self):
        control = PlateauControl()
        self.assertTrue(control.update(2.)["best"])
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
            result = validate(Uniform(), TokenBlockDataset(path, block_size=4), 18, 2, torch.device("cpu"))
            self.assertEqual(result["predicted_tokens"], 6)
            self.assertAlmostEqual(result["bpb"], 6., places=5)


if __name__ == "__main__":
    unittest.main()

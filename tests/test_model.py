import importlib
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from model import (
    FORMAL_MAMBA3_CONFIG,
    VOCAB_SIZE,
    Mamba3ByteLM,
    RMSNorm,
    SwiGLU,
    analytic_parameter_counts,
    build_mamba3_byte_model,
    parameter_counts,
)


LEGACY_COMMIT = "a5cd9baa24ed2b3959af5ce4572644d6076c69a9"
LEGACY_ROOT = Path(
    os.environ.get(
        "OMNIZIP_BASELINE_ROOT", "/media/ivc-17/New Volume/zyh/omnizip"
    )
)
CHECKPOINT_PATH = Path(
    os.environ.get(
        "OMNIZIP_MAMBA3_CHECKPOINT",
        str(LEGACY_ROOT / "runs/byte_backbone/formal_mamba3/checkpoints/best.pt"),
    )
)


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_frozen_legacy_models():
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
    if resolved_commit != LEGACY_COMMIT:
        raise AssertionError(
            f"Unexpected frozen baseline commit: {resolved_commit}"
        )
    subprocess.run(
        [
            "git",
            "-C",
            str(LEGACY_ROOT),
            "diff",
            "--quiet",
            "byte-backbone-baseline-v1",
            "--",
            "byte_experiment/models.py",
            "models/common.py",
        ],
        check=True,
    )

    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(LEGACY_ROOT))
    try:
        return importlib.import_module("byte_experiment.models")
    finally:
        sys.path.remove(str(LEGACY_ROOT))
        sys.dont_write_bytecode = previous_dont_write_bytecode


@unittest.skipUnless(LEGACY_ROOT.is_dir(), "frozen OmniZip baseline is unavailable")
class Mamba3ByteModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = _torch_load(CHECKPOINT_PATH)
        cls.model_config = cls.checkpoint["model_config"]
        cls.state_dict = cls.checkpoint["model"]

    def test_formal_configuration_and_structure(self):
        self.assertEqual(VOCAB_SIZE, 256)
        self.assertEqual(FORMAL_MAMBA3_CONFIG, self.model_config)

        model = build_mamba3_byte_model(self.model_config)
        self.assertEqual(len(model.blocks), 2)
        self.assertEqual(model.embedding.num_embeddings, 256)
        self.assertEqual(model.embedding.embedding_dim, 352)
        self.assertEqual(model.lm_head.in_features, 352)
        self.assertEqual(model.lm_head.out_features, 256)
        self.assertIsNone(model.lm_head.bias)
        self.assertNotEqual(
            model.embedding.weight.data_ptr(), model.lm_head.weight.data_ptr()
        )
        self.assertTrue(torch.count_nonzero(model.lm_head.weight).item() == 0)
        self.assertTrue(all(isinstance(block.norm1, RMSNorm) for block in model.blocks))
        self.assertTrue(all(isinstance(block.mlp, SwiGLU) for block in model.blocks))
        self.assertTrue(all(block.norm1.eps == 1e-8 for block in model.blocks))
        self.assertEqual(model.final_norm.eps, 1e-8)
        self.assertTrue(all(block.mixer.is_mimo is False for block in model.blocks))
        self.assertTrue(all(block.mixer.chunk_size == 64 for block in model.blocks))
        self.assertTrue(all(block.mlp.gate.out_features == 736 for block in model.blocks))

    def test_parameter_counts(self):
        expected = {
            "backbone_trainable_params": 3_275_212,
            "total_trainable_params": 3_455_436,
        }
        model = build_mamba3_byte_model(self.model_config)
        self.assertEqual(analytic_parameter_counts(self.model_config), expected)
        self.assertEqual(parameter_counts(model), expected)
        self.assertEqual(sum(value.numel() for value in self.state_dict.values()), 3_455_436)

    def test_old_checkpoint_loads_strictly_without_key_mapping(self):
        model = build_mamba3_byte_model(self.model_config)
        incompatible = model.load_state_dict(self.state_dict, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(list(model.state_dict()), list(self.state_dict))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by Mamba-3")
    def test_frozen_model_logits_match(self):
        legacy_models = _load_frozen_legacy_models()
        old_model = legacy_models.build_byte_model(self.model_config)
        new_model = build_mamba3_byte_model(self.model_config)
        old_model.load_state_dict(self.state_dict, strict=True)
        new_model.load_state_dict(self.state_dict, strict=True)
        old_model.cuda().eval()
        new_model.cuda().eval()
        inputs = torch.tensor(
            [[0, 255] + [index % 256 for index in range(62)]],
            device="cuda",
            dtype=torch.long,
        )

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            old_logits = old_model(inputs)
            new_logits = new_model(inputs)

        self.assertEqual(tuple(new_logits.shape), (1, 64, 256))
        self.assertEqual(new_logits.dtype, torch.float32)
        self.assertEqual(tuple(old_logits.shape), tuple(new_logits.shape))
        self.assertEqual((old_logits - new_logits).abs().max().item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by Mamba-3")
    def test_future_changes_do_not_affect_prefix_logits(self):
        model = build_mamba3_byte_model(self.model_config)
        model.load_state_dict(self.state_dict, strict=True)
        model.cuda().eval()
        prefix_length = 64
        first = torch.arange(128, device="cuda", dtype=torch.long).unsqueeze(0)
        second = first.clone()
        second[:, prefix_length:] = torch.arange(
            255, 255 - (128 - prefix_length), -1, device="cuda", dtype=torch.long
        )

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            first_logits = model(first)
            second_logits = model(second)

        prefix_max_abs_diff = (
            first_logits[:, :prefix_length] - second_logits[:, :prefix_length]
        ).abs().max().item()
        suffix_max_abs_diff = (
            first_logits[:, prefix_length:] - second_logits[:, prefix_length:]
        ).abs().max().item()
        self.assertEqual(prefix_max_abs_diff, 0.0)
        self.assertGreater(suffix_max_abs_diff, 0.0)


if __name__ == "__main__":
    unittest.main()

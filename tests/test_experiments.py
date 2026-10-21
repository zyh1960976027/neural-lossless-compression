import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from neural_compression import experiments


class ExperimentDirectoryTest(unittest.TestCase):
    def test_timestamp_collision_preserves_existing_results(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(experiments, "datetime") as clock:
            clock.now.return_value.astimezone.return_value.strftime.return_value = "20260911_160000"
            root = Path(temporary) / "experiments" / "test"
            first = experiments.create_experiment_directory(root, "debug")
            saved = first / "results.json"
            saved.write_text("original result")
            second = experiments.create_experiment_directory(root, "debug")
            self.assertEqual(first.name, "20260911_160000_debug")
            self.assertEqual(second.name, first.name + "_2")
            self.assertEqual(saved.read_text(), "original result")
            self.assertEqual(list(second.iterdir()), [])

    def test_chinese_and_spaced_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = experiments.create_experiment_directory(temporary, "  小样本 debug  ")
            self.assertRegex(
                directory.name, r"^\d{8}_\d{6}_小样本 debug$"
            )
            self.assertTrue(directory.is_dir())


if __name__ == "__main__":
    unittest.main()

"""CPU-only tests for the batched codec profiling harness."""

from pathlib import Path
import tempfile
import unittest

from scripts.profile_codec_batched import (
    extract_slice, summarize_values, timing_rows, write_json,
)


class ProfileCodecBatchedHelpersTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_extract_slice_uses_half_open_byte_range(self):
        source = self.root / "source"
        target = self.root / "target"
        source.write_bytes(bytes(range(32)))
        result = extract_slice(source, target, 10, 7)
        self.assertEqual(target.read_bytes(), bytes(range(10, 17)))
        self.assertEqual(result["offset_bytes"], 10)
        self.assertEqual(result["num_bytes"], 7)
        self.assertEqual(len(result["sha256"]), 64)

    def test_extract_slice_rejects_short_source_without_partial_output(self):
        source = self.root / "source"
        target = self.root / "target"
        source.write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "requested byte range"):
            extract_slice(source, target, 3, 10)
        self.assertFalse(target.exists())

    def test_summary_and_timing_rows(self):
        summary = summarize_values([3, 1, 2])
        self.assertEqual(summary["median"], 2)
        self.assertEqual(summary["min"], 1)
        self.assertEqual(summary["max"], 3)
        rows = timing_rows("encode", 4, {"model": 3, "range": 1}, 100, 20)
        self.assertEqual([row["percent"] for row in rows], [75, 25])
        self.assertEqual(rows[0]["microseconds_per_step"], 150000)
        self.assertEqual(rows[1]["nanoseconds_per_byte"], 10000000)

    def test_json_serialization_is_strict_and_utf8(self):
        path = self.root / "result.json"
        write_json(path, {"结论": "一致", "seconds": 1.25})
        self.assertIn('"结论": "一致"', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

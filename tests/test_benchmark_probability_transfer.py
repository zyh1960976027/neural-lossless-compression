"""Acceptance gates must reject noisy speedups and altered retained artifacts."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark_probability_transfer import summarize
from mamba3_byte_compress.tokenization import digest_file, save_json


class TransferBenchmarkTest(unittest.TestCase):
    def fixture(self, output, *, noisy=False, growth=0, candidate="fp16"):
        raw = output / "raw"
        raw.write_bytes(b"original")
        sha = digest_file(raw)
        save_json(output / "manifest.json", dict(vocab_sizes=[4096, 8192, 16384],
                  probability_transfer=["fp32", candidate], repeats=3,
                  raw_sha256=sha, code_sha256="code"))
        for vocab in (4096, 8192, 16384):
            for mode in ("fp32", candidate):
                for repeat in range(1, 4):
                    directory = output / str(vocab) / mode / f"repeat{repeat}"
                    directory.mkdir(parents=True)
                    packed, decoded = directory / "packed", directory / "decoded"
                    packed.write_bytes(mode.encode())
                    decoded.write_bytes(raw.read_bytes())
                    seconds = 10 + repeat if mode == "fp32" else 8 + (repeat if noisy else 0)
                    common = dict(actual_bpb=1 + (growth if mode == candidate else 0),
                                  compressed_bytes=4, compressed_sha256=digest_file(packed),
                                  probability_transfer_bytes=200 if mode == "fp32" else 100,
                                  files_identical=True, checksum_verified=True,
                                  job=dict(code_sha256="code", raw_sha256=sha,
                                           packed=str(packed), decoded=str(decoded)))
                    for op in ("encode", "decode"):
                        record = dict(common, **{op + "_seconds": seconds})
                        if candidate == "gpu_int":
                            record["probability_transfer_bytes"] = (vocab*400 if mode == "fp32"
                                                                    else (800 if op == "encode" else (vocab+1)*400))
                        save_json(directory / op / "result.json", record)

    def test_integer_mode_acceptance_and_payload_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, growth=.005, candidate="gpu_int")
            with redirect_stdout(io.StringIO()):
                summary = summarize(root)
            self.assertTrue(summary["gpu_int_default_eligible"])
            self.assertFalse(summary["fp16_default_eligible"])

    def test_acceptance_requires_both_speed_and_bpb(self):
        for noisy, growth, accepted in ((False, .005, True), (True, 0, False), (False, .02, False)):
            with self.subTest(noisy=noisy, growth=growth), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root, noisy=noisy, growth=growth)
                with redirect_stdout(io.StringIO()):
                    summary = summarize(root)
                self.assertEqual(summary["fp16_default_eligible"], accepted)

    def test_changed_output_is_not_certified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / "4096/fp16/repeat1/decoded").write_bytes(b"corrupt")
            with self.assertRaisesRegex(AssertionError, "Changed benchmark output"):
                summarize(root)


if __name__ == "__main__":
    unittest.main()

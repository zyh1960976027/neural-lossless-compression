import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import sentencepiece as spm

from neural_compression.tokenization import LosslessTokenizer, raw_segments, tokenize_file


def toy_model(algorithm="bpe"):
    output = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(["Hello  world\n中文\tother text" for _ in range(30)]),
        model_writer=output, model_type=algorithm, vocab_size=320, hard_vocab_limit=False,
        byte_fallback=True, normalization_rule_name="identity", add_dummy_prefix=False,
        remove_extra_whitespaces=False, bos_id=-1, eos_id=-1, pad_id=-1,
        num_threads=1, minloglevel=2)
    return output.getvalue()


class TokenizationTest(unittest.TestCase):
    def test_arbitrary_bytes_and_threads(self):
        rng = np.random.default_rng(42)
        samples = [bytes(range(256)), b"", b" \n\r\t \0", "中文▁😀".encode()]
        samples += [rng.integers(0, 256, 2049, dtype=np.uint8).tobytes() for _ in range(8)]
        for algorithm in ("bpe", "unigram"):
            tokenizer = LosslessTokenizer(toy_model(algorithm))
            baseline = tokenizer.encode_batch(samples)
            tokenizer.threads = 4
            other = tokenizer.encode_batch(samples)
            for raw, tokens, parallel in zip(samples, baseline, other):
                np.testing.assert_array_equal(tokens, parallel)
                self.assertEqual(tokenizer.decode(tokens), raw)
                self.assertEqual(b"".join(tokenizer.decode(tokens[i:i+2048]) for i in range(0, len(tokens), 2048)), raw)

    def test_stream_boundaries_and_cache(self):
        raw = b"a" * 65535 + "中文字".encode() + b"\n" + b"x" * 65536 + b"\xff\xe4\xbd"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw"
            source.write_bytes(raw)
            self.assertEqual(b"".join(raw_segments(source)), raw)
            for segment in list(raw_segments(source))[:-1]:
                segment.decode("utf8")
            self.assertEqual(b"".join(raw_segments(source, 65536, 100)), raw[65536:65636])
            tokenizer = LosslessTokenizer(toy_model())
            cache = Path(directory) / "tokens"
            result = tokenize_file(tokenizer, source, output=cache)
            self.assertEqual(tokenizer.decode(np.fromfile(cache, dtype="<u2")), raw)
            self.assertEqual(result["original_sha256"], result["decoded_sha256"])


if __name__ == "__main__":
    unittest.main()

"""Lossless SentencePiece byte adapter and deterministic streaming corpus I/O."""

import hashlib
import io
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

VERSION = 1
MAX_PIECE_INPUT_BYTES = 65536
SPECIAL = re.compile("([\u2581\udc80-\udcff])")


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_json(path):
    return json.loads(Path(path).read_text())


def _incomplete_suffix(data):
    for width in range(1, min(3, len(data)) + 1):
        first = data[-width]
        expected = (2 if 0xC2 <= first <= 0xDF else
                    3 if 0xE0 <= first <= 0xEF else
                    4 if 0xF0 <= first <= 0xF4 else 0)
        if expected > width and all(0x80 <= b <= 0xBF for b in data[len(data)-width+1:]):
            return width
    return 0


def raw_segments(path, offset=0, size=None):
    """Keep line endings, bound memory, and avoid splitting valid UTF-8 chars."""
    if offset < 0 or (size is not None and size < 0):
        raise ValueError("Negative input range")
    with open(path, "rb") as handle:
        handle.seek(offset)
        remaining = os.fstat(handle.fileno()).st_size - offset if size is None else size
        if remaining < 0:
            raise ValueError("Input offset outside file")
        pending = b""
        while remaining:
            chunk = handle.readline(min(MAX_PIECE_INPUT_BYTES - len(pending), remaining))
            if not chunk:
                raise ValueError("Truncated input range")
            remaining -= len(chunk)
            data = pending + chunk
            cut = _incomplete_suffix(data) if remaining and not data.endswith(b"\n") else 0
            pending = data[-cut:] if cut else b""
            data = data[:-cut] if cut else data
            if data:
                yield data
        if pending:
            yield pending


def raw_batches(path, offset=0, size=None, batch_bytes=1024 * 1024):
    batch, count = [], 0
    for segment in raw_segments(path, offset, size):
        batch.append(segment)
        count += len(segment)
        if count >= batch_bytes:
            yield batch
            batch, count = [], 0
    if batch:
        yield batch


class LosslessTokenizer:
    def __init__(self, model, threads=1):
        self.model_bytes = Path(model).read_bytes() if isinstance(model, (str, Path)) else bytes(model)
        self.sha256 = hashlib.sha256(self.model_bytes).hexdigest()
        self.sp = spm.SentencePieceProcessor(model_proto=self.model_bytes)
        self.threads = int(threads)
        if self.threads < 1:
            raise ValueError("threads must be positive")
        proto = pb.ModelProto()
        proto.ParseFromString(self.model_bytes)
        n = proto.normalizer_spec
        t = proto.trainer_spec
        if (n.name != "identity" or n.add_dummy_prefix or n.remove_extra_whitespaces
                or not t.byte_fallback or t.bos_id != -1 or t.eos_id != -1):
            raise ValueError("Tokenizer does not implement the lossless profile")
        self.vocab_size = len(proto.pieces)
        self.byte_ids, self.byte_pieces = {}, []
        for index, piece in enumerate(proto.pieces):
            if piece.type == pb.ModelProto.SentencePiece.BYTE:
                value = int(piece.piece[3:5], 16)
                self.byte_ids[value] = index
                self.byte_pieces.append(bytes([value]))
            elif piece.type in (pb.ModelProto.SentencePiece.NORMAL,
                                pb.ModelProto.SentencePiece.USER_DEFINED):
                self.byte_pieces.append(piece.piece.replace("▁", " ").encode("utf-8"))
            else:
                self.byte_pieces.append(None)
        if len(self.byte_ids) != 256:
            raise ValueError("All 256 byte fallback pieces are required")

    def encode_batch(self, raw_rows):
        texts, recipes = [], []
        for raw in raw_rows:
            recipe = []
            for part in SPECIAL.split(raw.decode("utf-8", "surrogateescape")):
                if not part:
                    continue
                if SPECIAL.fullmatch(part):
                    recipe.append([self.byte_ids[b] for b in part.encode("utf-8", "surrogateescape")])
                else:
                    recipe.append(len(texts))
                    texts.append(part)
            recipes.append(recipe)
        encoded = self.sp.encode(texts, out_type=int, num_threads=self.threads,
                                 enable_sampling=False) if texts else []
        rows = []
        for recipe in recipes:
            ids = [token for part in recipe for token in (encoded[part] if isinstance(part, int) else part)]
            if self.sp.unk_id() in ids:
                raise ValueError("Unknown piece would lose bytes")
            rows.append(np.asarray(ids, dtype=np.uint16))
        return rows

    def encode(self, raw):
        return self.encode_batch([raw])[0]

    def decode(self, tokens):
        try:
            pieces = [self.byte_pieces[int(token)] for token in tokens
                      if 0 <= int(token) < self.vocab_size]
            if len(pieces) != len(tokens) or any(piece is None for piece in pieces):
                raise ValueError("Invalid token in compressed data")
            return b"".join(pieces)
        except (IndexError, TypeError) as error:
            raise ValueError("Invalid token in compressed data") from error


def train_tokenizer(raw_path, output, algorithm, vocab_size=4096, threads=16,
                    train_bytes=100_000_000):
    if algorithm not in ("bpe", "unigram"):
        raise ValueError("Expected bpe or unigram")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    def sentences():
        for raw in raw_segments(raw_path, size=train_bytes):
            for part in SPECIAL.split(raw.decode("utf-8", "surrogateescape")):
                if part and not SPECIAL.fullmatch(part):
                    yield part
    stream = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=sentences(), model_writer=stream, model_type=algorithm,
        vocab_size=vocab_size, hard_vocab_limit=True, character_coverage=0.9995,
        byte_fallback=True, normalization_rule_name="identity", add_dummy_prefix=False,
        remove_extra_whitespaces=False, bos_id=-1, eos_id=-1, pad_id=-1,
        num_threads=threads, input_sentence_size=0, shuffle_input_sentence=False,
        max_sentence_length=MAX_PIECE_INPUT_BYTES + 16, minloglevel=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_bytes(stream.getvalue())
    LosslessTokenizer(temporary)
    os.replace(temporary, output)
    return {"algorithm": algorithm, "vocab_size": vocab_size,
            "sha256": digest_file(output), "sentencepiece": spm.__version__,
            "threads": threads, "train_bytes": train_bytes, "adapter_version": VERSION}


def tokenize_file(tokenizer, raw_path, *, offset=0, size=None, output=None):
    """Verify every byte while optionally writing a little-endian uint16 cache."""
    import contextlib
    started = time.perf_counter()
    original, restored = hashlib.sha256(), hashlib.sha256()
    count = byte_count = 0
    target = Path(output) if output is not None else None
    temporary = target.with_suffix(target.suffix + ".tmp") if target else None
    if target:
        target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (temporary.open("wb") if temporary else contextlib.nullcontext()) as handle:
            for batch in raw_batches(raw_path, offset, size):
                rows = tokenizer.encode_batch(batch)
                for raw, tokens in zip(batch, rows):
                    decoded = tokenizer.decode(tokens)
                    if raw != decoded:
                        raise AssertionError(f"Lossy tokenizer at byte {byte_count}")
                    original.update(raw)
                    restored.update(decoded)
                    byte_count += len(raw)
                    count += len(tokens)
                    if handle:
                        handle.write(tokens.astype("<u2", copy=False).tobytes())
        if temporary:
            os.replace(temporary, target)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    result = dict(raw_bytes=byte_count, tokens=count, offset=offset,
                  original_sha256=original.hexdigest(), decoded_sha256=restored.hexdigest(),
                  tokenizer_sha256=tokenizer.sha256, verified=True,
                  elapsed_seconds=time.perf_counter()-started, threads=tokenizer.threads)
    if target:
        result.update(token_file=str(target.resolve()), token_sha256=digest_file(target))
        save_json(str(target) + ".json", result)
    return result


def token_blocks(tokenizer, raw_path, batch_size, *, block_size=2048):
    """Stream raw input into groups of independent token blocks, without a cache."""
    pending = np.empty(0, dtype=np.uint16)
    capacity = batch_size * block_size
    for raw in raw_batches(raw_path):
        rows = tokenizer.encode_batch(raw)
        values = np.concatenate([pending, *rows])
        offset = 0
        while len(values) - offset >= capacity:
            yield values[offset:offset+capacity].reshape(batch_size, block_size), np.full(batch_size, block_size)
            offset += capacity
        pending = values[offset:].copy()
    if len(pending):
        n = (len(pending) + block_size - 1) // block_size
        values = np.zeros((n, block_size), dtype=np.uint16)
        values.ravel()[:len(pending)] = pending
        lengths = np.minimum(block_size, len(pending) - np.arange(n) * block_size)
        yield values, lengths

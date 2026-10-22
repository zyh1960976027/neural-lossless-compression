"""Lossless SentencePiece byte adapter and deterministic streaming corpus I/O."""

import hashlib
import io as buffer_io
import json
import os
from pathlib import Path
import re
import time

import torch
from torch.utils.data import Dataset
import numpy as np
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from .io import sha256_file as digest_file, save_json, load_json
from .config import BLOCK_SIZE, VOCAB_SIZE

VERSION = 1
MAX_PIECE_INPUT_BYTES = 65536
SPECIAL = re.compile("([\u2581\udc80-\udcff])")


def _incomplete_suffix(data):
    for width in range(1, min(3, len(data)) + 1):
        first = data[-width]
        expected = (
            2
            if 0xC2 <= first <= 0xDF
            else 3
            if 0xE0 <= first <= 0xEF
            else 4
            if 0xF0 <= first <= 0xF4
            else 0
        )
        if expected > width and all(
            0x80 <= b <= 0xBF for b in data[len(data) - width + 1 :]
        ):
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
            chunk = handle.readline(
                min(MAX_PIECE_INPUT_BYTES - len(pending), remaining)
            )
            if not chunk:
                raise ValueError("Truncated input range")
            remaining -= len(chunk)
            data = pending + chunk
            cut = (
                _incomplete_suffix(data)
                if remaining and not data.endswith(b"\n")
                else 0
            )
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
        self.model_bytes = (
            Path(model).read_bytes() if isinstance(model, (str, Path)) else bytes(model)
        )
        self.sha256 = hashlib.sha256(self.model_bytes).hexdigest()
        self.sp = spm.SentencePieceProcessor(model_proto=self.model_bytes)
        self.threads = int(threads)
        if self.threads < 1:
            raise ValueError("threads must be positive")
        proto = pb.ModelProto()
        proto.ParseFromString(self.model_bytes)
        n = proto.normalizer_spec
        t = proto.trainer_spec
        if (
            n.name != "identity"
            or n.add_dummy_prefix
            or n.remove_extra_whitespaces
            or not t.byte_fallback
            or t.bos_id != -1
            or t.eos_id != -1
        ):
            raise ValueError("Tokenizer does not implement the lossless profile")
        self.vocab_size = len(proto.pieces)
        self.byte_ids, self.byte_pieces = {}, []
        for index, piece in enumerate(proto.pieces):
            if piece.type == pb.ModelProto.SentencePiece.BYTE:
                value = int(piece.piece[3:5], 16)
                self.byte_ids[value] = index
                self.byte_pieces.append(bytes([value]))
            elif piece.type in (
                pb.ModelProto.SentencePiece.NORMAL,
                pb.ModelProto.SentencePiece.USER_DEFINED,
            ):
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
                    recipe.append(
                        [
                            self.byte_ids[b]
                            for b in part.encode("utf-8", "surrogateescape")
                        ]
                    )
                else:
                    recipe.append(len(texts))
                    texts.append(part)
            recipes.append(recipe)
        encoded = (
            self.sp.encode(
                texts, out_type=int, num_threads=self.threads, enable_sampling=False
            )
            if texts
            else []
        )
        rows = []
        for recipe in recipes:
            ids = [
                token
                for part in recipe
                for token in (encoded[part] if isinstance(part, int) else part)
            ]
            if self.sp.unk_id() in ids:
                raise ValueError("Unknown piece would lose bytes")
            rows.append(np.asarray(ids, dtype=np.uint16))
        return rows

    def encode(self, raw):
        return self.encode_batch([raw])[0]

    def decode(self, tokens):
        try:
            pieces = [
                self.byte_pieces[int(token)]
                for token in tokens
                if 0 <= int(token) < self.vocab_size
            ]
            if len(pieces) != len(tokens) or any(piece is None for piece in pieces):
                raise ValueError("Invalid token in compressed data")
            return b"".join(pieces)
        except (IndexError, TypeError) as error:
            raise ValueError("Invalid token in compressed data") from error


def train_tokenizer(raw_path, output, threads=16):
    algorithm = "bpe"
    vocab_size = VOCAB_SIZE
    train_bytes = Path(raw_path).stat().st_size
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)

    def sentences():
        for raw in raw_segments(raw_path, size=train_bytes):
            for part in SPECIAL.split(raw.decode("utf-8", "surrogateescape")):
                if part and not SPECIAL.fullmatch(part):
                    yield part

    stream = buffer_io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=sentences(),
        model_writer=stream,
        model_type=algorithm,
        vocab_size=vocab_size,
        hard_vocab_limit=True,
        character_coverage=0.9995,
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        num_threads=threads,
        input_sentence_size=0,
        shuffle_input_sentence=False,
        max_sentence_length=MAX_PIECE_INPUT_BYTES + 16,
        minloglevel=1,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_bytes(stream.getvalue())
    LosslessTokenizer(temporary)
    os.replace(temporary, output)
    return {
        "algorithm": algorithm,
        "vocab_size": vocab_size,
        "sha256": digest_file(output),
        "sentencepiece": spm.__version__,
        "threads": threads,
        "train_bytes": train_bytes,
        "adapter_version": VERSION,
    }


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
        with (
            temporary.open("wb") if temporary else contextlib.nullcontext()
        ) as handle:
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
    result = dict(
        raw_bytes=byte_count,
        tokens=count,
        offset=offset,
        original_sha256=original.hexdigest(),
        decoded_sha256=restored.hexdigest(),
        tokenizer_sha256=tokenizer.sha256,
        verified=True,
        elapsed_seconds=time.perf_counter() - started,
        threads=tokenizer.threads,
    )
    if target:
        result.update(
            token_file=str(target.resolve()), token_sha256=digest_file(target)
        )
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
            yield values[offset : offset + capacity].reshape(
                batch_size, block_size
            ), np.full(batch_size, block_size)
            offset += capacity
        pending = values[offset:].copy()
    if len(pending):
        n = (len(pending) + block_size - 1) // block_size
        values = np.zeros((n, block_size), dtype=np.uint16)
        values.ravel()[: len(pending)] = pending
        lengths = np.minimum(block_size, len(pending) - np.arange(n) * block_size)
        yield values, lengths


class TokenBlockDataset(Dataset):
    def __init__(self, path, block_size=BLOCK_SIZE):
        self.path = str(path)
        self.block_size = int(block_size)
        if Path(path).stat().st_size % 2:
            raise ValueError("Malformed uint16 token cache")
        self.num_tokens = Path(path).stat().st_size // 2
        self.num_blocks = (self.num_tokens + self.block_size - 1) // self.block_size
        self._tokens = None

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._tokens is None:
            self._tokens = np.memmap(self.path, mode="r", dtype="<u2")
        start = index * self.block_size
        valid = min(self.block_size, self.num_tokens - start)
        values = np.zeros(self.block_size, dtype=np.int64)
        values[:valid] = self._tokens[start : start + valid]
        target = np.zeros_like(values)
        target[: max(valid - 1, 0)] = values[1:valid]
        mask = np.arange(self.block_size) < valid - 1
        return (
            torch.from_numpy(values),
            torch.from_numpy(target),
            torch.from_numpy(mask),
        )


def _copy_range(source, target, offset, size):
    temporary = target.with_suffix(".tmp")
    digest = hashlib.sha256()
    with open(source, "rb") as reader, temporary.open("wb") as writer:
        reader.seek(offset)
        remaining = size
        while remaining:
            chunk = reader.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError("enwik9 不足以提供前 100 MB 训练与随后 10 MB 验证数据")
            writer.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
    os.replace(temporary, target)
    return digest.hexdigest()


def training_inputs(source=None, train_file=None, validation_file=None):
    from .config import DEFAULT_SOURCE

    if source is not None and (train_file is not None or validation_file is not None):
        raise ValueError("--source 与 --train-file/--validation-file 互斥")
    if (train_file is None) != (validation_file is None):
        raise ValueError("--train-file 和 --validation-file 必须同时提供")
    if train_file is not None:
        paths = [Path(train_file).resolve(), Path(validation_file).resolve()]
        if paths[0] == paths[1]:
            raise ValueError("训练集和验证集必须为不同文件")
        mode = "custom"
    else:
        paths = [Path(source or DEFAULT_SOURCE).resolve()]
        mode = "enwik9"
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"数据文件不存在或为空：{path}")
    if mode == "enwik9" and paths[0].stat().st_size < 110_000_000:
        raise ValueError("enwik9 输入至少需要 110,000,000 字节；小文件请使用自定义训练/验证参数")
    return mode, paths


def prepare_training(
    run_dir,
    *,
    source=None,
    train_file=None,
    validation_file=None,
    tokenizer=None,
    batch_size=72,
    micro_batch_size=None,
    workers=0,
    device="cuda:0",
    tokenizer_threads=24,
):
    """准备 BPE 与两个缓存，并在续跑前核对输入、分词器和配置。调用方持有实验锁。"""
    from .config import MODEL_CONFIG, TRAINING
    from .io import validate_batch_size

    validate_batch_size(batch_size)
    micro_batch_size = batch_size if micro_batch_size is None else micro_batch_size
    if (
        micro_batch_size < 1
        or micro_batch_size > batch_size
        or batch_size % micro_batch_size
    ):
        raise ValueError("微批大小必须是有效批大小的正整数约数")
    if workers < 0 or tokenizer_threads < 1:
        raise ValueError("数据加载进程数必须非负，分词线程数必须为正")
    mode, paths = training_inputs(source, train_file, validation_file)
    directory = Path(run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    supplied = Path(tokenizer).resolve() if tokenizer is not None else None
    if supplied is not None:
        require_baseline_tokenizer(LosslessTokenizer(supplied))
    request = dict(
        schema_version=1,
        mode=mode,
        inputs=[
            dict(path=str(p), bytes=p.stat().st_size, sha256=digest_file(p))
            for p in paths
        ],
        split_bytes=[100_000_000, 10_000_000] if mode == "enwik9" else None,
        tokenizer_source=dict(path=str(supplied), sha256=digest_file(supplied))
        if supplied
        else None,
        model_config=MODEL_CONFIG,
        training_config=TRAINING,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        workers=workers,
        device=str(torch.device(device)),
        tokenizer_threads=tokenizer_threads,
    )
    # JSON converts tuples to lists; normalize before comparing resumed configuration.
    request = json.loads(json.dumps(request))
    config_path = directory / "config.json"
    if config_path.exists():
        if load_json(config_path) != request:
            raise ValueError("输入数据、分词器或配置已变更，拒绝复用实验目录；请使用新目录")
    else:
        owned = [p for p in directory.iterdir() if p.name != ".run.lock"]
        if owned:
            raise ValueError("实验目录已有其他材料且没有匹配配置；请使用新目录")
        save_json(config_path, request)
    data_dir = directory / "data"
    data_dir.mkdir(exist_ok=True)
    prepared_path = directory / "prepared.json"
    prepared = load_json(prepared_path) if prepared_path.exists() else None
    raw_paths = paths
    if mode == "enwik9":
        raw_paths = [data_dir / "train.bin", data_dir / "validation.bin"]
        for path, offset, size in zip(
            raw_paths, (0, 100_000_000), (100_000_000, 10_000_000)
        ):
            if not path.exists():
                _copy_range(paths[0], path, offset, size)
            if path.stat().st_size != size:
                raise ValueError(f"训练切片大小已变化：{path}")
    tokenizer_path = directory / "tokenizer.model"
    if not tokenizer_path.exists():
        if supplied:
            temporary = tokenizer_path.with_suffix(".tmp")
            temporary.write_bytes(supplied.read_bytes())
            os.replace(temporary, tokenizer_path)
        else:
            try:
                train_tokenizer(raw_paths[0], tokenizer_path, threads=tokenizer_threads)
            except RuntimeError as error:
                raise ValueError(
                    "BPE 训练失败：训练语料必须足以构建完整 4096 词表；可通过 --tokenizer 复用配套分词器。"
                    + str(error)
                ) from error
    adapter = LosslessTokenizer(tokenizer_path, threads=tokenizer_threads)
    require_baseline_tokenizer(adapter)
    if supplied and adapter.sha256 != digest_file(supplied):
        raise ValueError("实验目录中的分词器与指定分词器不一致")
    if prepared and adapter.sha256 != prepared["tokenizer_sha256"]:
        raise ValueError("实验目录中的分词器已变化，拒绝续训")
    records = {}
    for name, raw in zip(("train", "validation"), raw_paths):
        cache = data_dir / f"{name}.tokens"
        metadata = Path(str(cache) + ".json")
        expected_raw = digest_file(raw)
        if cache.exists() and metadata.exists():
            record = load_json(metadata)
            if (
                record["original_sha256"] != expected_raw
                or record["tokenizer_sha256"] != adapter.sha256
                or record["token_sha256"] != digest_file(cache)
                or not record["verified"]
            ):
                raise ValueError(f"{name} 数据缓存已变化，拒绝复用")
        elif prepared:
            raise ValueError(f"{name} 缓存缺失，拒绝改变已有训练输入")
        else:
            record = tokenize_file(adapter, raw, output=cache)
        if record["tokens"] <= (record["tokens"] + BLOCK_SIZE - 1) // BLOCK_SIZE:
            raise ValueError(f"{name} 数据不足以提供条件预测位置")
        if prepared and record != prepared["datasets"][name]:
            raise ValueError(f"{name} 缓存记录已变更，拒绝续训")
        records[name] = record
    result = dict(
        schema_version=1,
        tokenizer_sha256=adapter.sha256,
        vocab_size=VOCAB_SIZE,
        datasets=records,
    )
    if not prepared:
        save_json(prepared_path, result)
    return result


def require_baseline_tokenizer(tokenizer):
    proto = pb.ModelProto()
    proto.ParseFromString(tokenizer.model_bytes)
    if (
        tokenizer.vocab_size != VOCAB_SIZE
        or proto.trainer_spec.model_type != pb.TrainerSpec.BPE
    ):
        raise ValueError("当前主线需要 BPE 4096 分词器（最大编号 4095），不会自动缩小词表")
    return tokenizer

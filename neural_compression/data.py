"""Byte-level training and evaluation datasets."""

import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class _LazyByteMemmap:
    """Open a raw byte file lazily so DataLoader workers can pickle the dataset."""

    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self.size = os.path.getsize(self.path)
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = np.memmap(self.path, mode="r", dtype=np.uint8)
        return self._data

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_data"] = None
        return state


class ByteTrainingDataset(Dataset):
    """Read fixed byte windows selected by a one-dimensional int64 start array."""

    def __init__(
        self,
        raw_path,
        starts_path,
        sequence_length,
        sample_offset=0,
        sample_count=None,
    ):
        self.raw = _LazyByteMemmap(raw_path)
        self.starts_path = str(Path(starts_path).resolve())
        self.starts = np.load(self.starts_path, mmap_mode="r", allow_pickle=False)
        if self.starts.dtype != np.int64 or self.starts.ndim != 1:
            raise ValueError("Training starts must be a one-dimensional int64 array.")

        self.sequence_length = int(sequence_length)
        self.sample_offset = int(sample_offset)
        available = len(self.starts) - self.sample_offset
        self.sample_count = available if sample_count is None else int(sample_count)
        if (
            self.sample_offset < 0
            or self.sample_count < 0
            or self.sample_count > available
        ):
            raise ValueError("Invalid training sample range.")

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        if index < 0 or index >= self.sample_count:
            raise IndexError(index)

        start = int(self.starts[self.sample_offset + index])
        stop = start + self.sequence_length + 1
        if start < 0 or stop > self.raw.size:
            raise ValueError(f"Training window [{start}, {stop}) is outside the raw file.")

        window = np.array(self.raw.data[start:stop], dtype=np.int64, copy=True)
        return {
            "input_tokens": torch.from_numpy(window[:-1]),
            "target_tokens": torch.from_numpy(window[1:]),
        }


class ByteEvaluationDataset(Dataset):
    """Read every next-byte target in contiguous, independently reset blocks."""

    def __init__(self, raw_path, sequence_length):
        self.raw = _LazyByteMemmap(raw_path)
        self.sequence_length = int(sequence_length)
        self.raw_bytes = self.raw.size
        self.predicted_bytes = max(0, self.raw_bytes - 1)
        self.num_blocks = math.ceil(self.predicted_bytes / self.sequence_length)

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, index):
        if index < 0 or index >= self.num_blocks:
            raise IndexError(index)

        target_start = 1 + index * self.sequence_length
        valid = min(self.sequence_length, self.raw_bytes - target_start)
        source = np.array(
            self.raw.data[target_start - 1 : target_start + valid],
            dtype=np.int64,
            copy=True,
        )

        inputs = np.zeros(self.sequence_length, dtype=np.int64)
        targets = np.zeros(self.sequence_length, dtype=np.int64)
        masks = np.zeros(self.sequence_length, dtype=np.bool_)
        inputs[:valid] = source[:valid]
        targets[:valid] = source[1 : valid + 1]
        masks[:valid] = True
        return {
            "input_tokens": torch.from_numpy(inputs),
            "target_tokens": torch.from_numpy(targets),
            "masks": torch.from_numpy(masks),
        }

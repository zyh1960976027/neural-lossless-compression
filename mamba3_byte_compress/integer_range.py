"""Batch integer intervals/CDFs for the CPU Constriction range coder.

CDFs from the GPU must be strictly increasing from zero to TOTAL. The public
decoder validates them by default; the internal GPU path can skip the full scan.
"""
import ctypes as ct
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

TOTAL = 1 << 24
QUANTIZATION = "cdf-f64-leaky24-v1"


def quantize_cdf(probabilities):
    """FP32 model probabilities -> FP64 prefix sum -> strictly positive u24 CDF.

    One unit per symbol guarantees representability, including zero model mass.
    The same inclusive prefix sum supplies normalization and both encoder ends.
    Arithmetic and operation ordering are part of the container profile.
    """
    size = probabilities.shape[-1]
    if not 2 <= size < TOTAL:
        raise ValueError("Unsupported integer vocabulary size")
    cumulative = probabilities.cumsum(dim=-1, dtype=torch.float64)
    scaled = cumulative * ((TOTAL - size) / cumulative[..., -1:])
    ends = scaled.clamp(0, TOTAL-size).to(torch.int32)
    ends = ends + torch.arange(1, size+1, device=ends.device, dtype=torch.int32)
    ends[..., -1] = TOTAL
    return torch.cat((torch.zeros_like(ends[..., :1]), ends), dim=-1)


@lru_cache(maxsize=1)
def _library():
    path = Path(__file__).with_name("_integer_range.so")
    if not path.exists():
        raise RuntimeError("Build the integer range interface: python scripts/build_integer_range.py")
    lib = ct.CDLL(str(path))
    pointer = ct.POINTER(ct.c_uint32)
    signatures = {
        "ir_abi_version": ([], ct.c_uint32),
        "ir_encoder_new": ([], ct.c_void_p),
        "ir_encoder_free": ([ct.c_void_p], None),
        "ir_encode": ([ct.c_void_p, pointer, ct.c_size_t], ct.c_int),
        "ir_compressed_size": ([ct.c_void_p], ct.c_size_t),
        "ir_compressed_copy": ([ct.c_void_p, pointer, ct.c_size_t], ct.c_int),
        "ir_decoder_new": ([pointer, ct.c_size_t], ct.c_void_p),
        "ir_decoder_free": ([ct.c_void_p], None),
        "ir_decode": ([ct.c_void_p, pointer, ct.c_size_t, ct.c_size_t, pointer], ct.c_int),
    }
    for name, (args, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
    if lib.ir_abi_version() != 1:
        raise RuntimeError("Incompatible integer range ABI")
    return lib


def _array(values, ndim):
    values = np.asarray(values)
    if values.ndim != ndim or values.dtype not in (np.dtype("int32"), np.dtype("uint32")):
        raise ValueError("Expected a 32-bit integer array")
    return np.ascontiguousarray(values).view(np.uint32)


def _ptr(values):
    return values.ctypes.data_as(ct.POINTER(ct.c_uint32))


class IntegerEncoder:
    def __init__(self):
        self.lib = _library()
        self.handle = self.lib.ir_encoder_new()

    def close(self):
        if self.handle:
            self.lib.ir_encoder_free(self.handle)
            self.handle = None

    def __del__(self):
        if getattr(self, "handle", None):
            self.close()

    def encode(self, intervals):
        intervals = _array(intervals, 2)
        if not self.handle or intervals.shape[1] != 2:
            raise ValueError("Expected an open encoder and (B, 2) intervals")
        if self.lib.ir_encode(self.handle, _ptr(intervals), len(intervals)):
            raise ValueError("Invalid integer encoding interval")

    def get_compressed(self):
        if not self.handle:
            raise ValueError("Encoder is closed")
        data = np.empty(self.lib.ir_compressed_size(self.handle), dtype=np.uint32)
        if self.lib.ir_compressed_copy(self.handle, _ptr(data), len(data)):
            raise ValueError("Cannot finalize range encoder")
        return data


class IntegerDecoder:
    def __init__(self, data):
        self.lib = _library()
        data = _array(data, 1)
        if not len(data):
            raise ValueError("Empty integer range payload")
        self.handle = self.lib.ir_decoder_new(_ptr(data), len(data))
        if not self.handle:
            raise ValueError("Invalid integer range payload")

    def close(self):
        if self.handle:
            self.lib.ir_decoder_free(self.handle)
            self.handle = None

    def __del__(self):
        if getattr(self, "handle", None):
            self.close()

    def decode(self, cdfs, *, trusted=False):
        cdfs = _array(cdfs, 2)
        if not self.handle or cdfs.shape[1] < 3:
            raise ValueError("Expected an open decoder and (B, V+1) CDFs")
        if not trusted and (np.any(cdfs[:, 0] != 0) or np.any(cdfs[:, -1] != TOTAL)
                            or np.any(cdfs[:, 1:] <= cdfs[:, :-1])):
            raise ValueError("Invalid integer cumulative distribution")
        result = np.empty(len(cdfs), dtype=np.uint32)
        if self.lib.ir_decode(self.handle, _ptr(cdfs), len(cdfs), cdfs.shape[1], _ptr(result)):
            raise ValueError("Invalid integer range stream or cumulative distribution")
        return result


def uniform_cdf(vocab_size):
    return (np.arange(vocab_size+1, dtype=np.int64) * TOTAL // vocab_size).astype(np.int32)


def select_intervals(cdfs, symbols):
    rows = np.arange(len(symbols))
    left, right = cdfs[rows, symbols], cdfs[rows, symbols+1]
    return np.stack((left, right-left), axis=-1)

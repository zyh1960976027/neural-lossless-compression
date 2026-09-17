"""Memory-layout adapter for the installed Mamba-3 CuTe batched step.

The kernel declares 16-byte aligned outer strides. With 11 heads, its native
preprocessing produces [B,11] tensors and projection views that violate that
contract for B > 1. Copies below preserve shape, dtype and every tensor value;
padding is storage only, never an extra head or token. No package is patched.
"""

import math

import torch


def _aligned_rows(tensor):
    alignment = 16 // tensor.element_size()
    if (tensor.stride(-1) == 1 and tensor.data_ptr() % 16 == 0
            and all(stride % alignment == 0 for stride in tensor.stride()[:-1])):
        return tensor
    row_size = math.prod(tensor.shape[1:])
    row_stride = (row_size + alignment - 1) // alignment * alignment
    storage = torch.empty((tensor.shape[0], row_stride), device=tensor.device, dtype=tensor.dtype)
    result = storage[:, :row_size].view(tensor.shape)
    result.copy_(tensor)
    return result


class AlignedMambaStep:
    """Delegate the native step unchanged, adapting its preprocessing layout."""

    def __init__(self, mixer):
        self.mixer = mixer

    def __getattr__(self, name):
        return getattr(self.mixer, name)

    def _preprocess(self, *args):
        dt, b, c, x, z, trap, a, angles = self.mixer._preprocess(*args)
        return (_aligned_rows(dt), b, c, _aligned_rows(x), _aligned_rows(z),
                _aligned_rows(trap), _aligned_rows(a), angles)

    def __call__(self, inputs, *states):
        return type(self.mixer).step(self, inputs, *states)

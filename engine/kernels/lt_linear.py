"""Native BF16 linear using PyTorch 2.5.1's cuBLASLt preference."""

import torch
from torch.nn.functional import linear


def lt_linear(x, weight):
    previous = torch.backends.cuda.preferred_blas_library()
    torch.backends.cuda.preferred_blas_library('cublaslt')
    try:
        return linear(x, weight)
    finally:
        torch.backends.cuda.preferred_blas_library(previous)

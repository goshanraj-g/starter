"""BF16 small-batch matrix-vector product with parallel FP32 K reductions."""

import torch
import triton
import triton.language as tl


@triton.jit
def _matvec_partial(X, W, PART, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    split = tl.program_id(1)
    row = tl.arange(0, BM)
    col = tl.program_id(0) * BN + tl.arange(0, BN)
    k = split * BK + tl.arange(0, BK)
    inputs = tl.load(X + row[:, None] * K + k[None, :],
                     (row[:, None] < M) & (k[None, :] < K), 0).to(tl.float32)
    weights = tl.load(W + col[:, None] * K + k[None, :],
                      (col[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
    result = tl.sum(inputs[:, None, :] * weights[None, :, :], 2)
    tl.store(PART + (split * M + row[:, None]) * N + col[None, :],
             result, (row[:, None] < M) & (col[None, :] < N))


@triton.jit
def _finish(PART, OUT, SIZE: tl.constexpr, SPLIT: tl.constexpr,
            SPLIT_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, SPLIT_BLOCK)
    values = tl.load(PART + split[:, None] * SIZE + col[None, :],
                     (split[:, None] < SPLIT) & (col[None, :] < SIZE), 0)
    tl.store(OUT + col, tl.sum(values, 0), col < SIZE)


def matvec(x, weight, block_k=1024):
    batch = x.numel() // x.shape[-1]
    n, k = weight.shape
    split = triton.cdiv(k, block_k)
    block_n = 16 if batch == 1 else 4
    partial = torch.empty((split, batch, n), dtype=torch.float32, device=x.device)
    output = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
    _matvec_partial[(triton.cdiv(n, block_n), split)](
        x, weight, partial, batch, n, k, triton.next_power_of_2(batch), block_n,
        block_k, num_warps=8, enable_fp_fusion=False,
    )
    _finish[(triton.cdiv(batch * n, 256),)](
        partial, output, batch * n, split, triton.next_power_of_2(split), 256,
        num_warps=4,
    )
    return output

"""Prefill pointwise fusion, preserving the worked norm and native SiLU casts."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _residual_norm(X, BRANCH, W, RESIDUAL, NORMAL, row_stride, n_cols, eps,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offsets = row * row_stride + cols
    x = tl.load(X + offsets, mask, 0).to(tl.float32)
    branch = tl.load(BRANCH + offsets, mask, 0).to(tl.float32)
    residual = (x + branch).to(tl.bfloat16)
    value = residual.to(tl.float32)
    # Same runtime divisor/epsilon, block shape, warp count and compiler
    # fusion setting as the original worked FusedRMSNorm kernel.
    variance = tl.sum(value * value, axis=0) / n_cols
    normed = value * tl.math.rsqrt(variance + eps)
    weight = tl.load(W + cols, mask, 0)
    tl.store(RESIDUAL + offsets, residual, mask)
    tl.store(NORMAL + offsets, normed.to(NORMAL.dtype.element_ty) * weight, mask)


def residual_norm(x, branch, norm):
    residual = torch.empty_like(x)
    normal = torch.empty_like(x)
    columns = x.shape[-1]
    block = triton.next_power_of_2(columns)
    _residual_norm[(x.numel() // columns,)](
        x, branch, norm.weight, residual, normal, columns, columns,
        norm.variance_epsilon, block,
        num_warps=max(4, min(16, block // 256)),
    )
    return residual, normal


@triton.jit
def _swiglu(GATE, UP, OUT, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(GATE + index, index < SIZE, 0).to(tl.float32)
    up = tl.load(UP + index, index < SIZE, 0).to(tl.float32)
    activated = libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(tl.bfloat16)
    tl.store(OUT + index, activated.to(tl.float32) * up, index < SIZE)


def swiglu(gate, up):
    output = torch.empty_like(gate)
    _swiglu[(triton.cdiv(gate.numel(), 256),)](
        gate, up, output, gate.numel(), 256,
        num_warps=4, enable_fp_fusion=False,
    )
    return output

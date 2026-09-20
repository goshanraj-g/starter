"""BF16 gate/up GEMM with FP32 split reduction and fused exact SwiGLU."""

from functools import partial
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from kernels.linear import _linear
from kernels.pointwise import swiglu
from kernels.select_linears import graph_time


@triton.jit
def _finish(PART, OUT, ROWS: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = index // 9728
    col = index % 9728
    split = tl.arange(0, SPLIT)
    offset = (split[:, None] * ROWS + row[None, :]) * 19456 + col[None, :]
    gate = tl.sum(tl.load(PART + offset, row[None, :] < ROWS, 0), 0)
    up = tl.sum(tl.load(PART + offset + 9728, row[None, :] < ROWS, 0), 0)
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    activated = libdevice.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(tl.bfloat16)
    tl.store(OUT + index, activated.to(tl.float32) * up, row < ROWS)


def fused_gate_up(x, weight, split=4):
    rows = x.numel() // 2560
    workspace = torch.empty((split, rows, 19456), dtype=torch.float32, device=x.device)
    output = torch.empty((*x.shape[:-1], 9728), dtype=x.dtype, device=x.device)
    _linear[(triton.cdiv(rows, 16), triton.cdiv(19456, 64), split)](
        x, weight, workspace, rows, 19456, 2560, split, 16, 64, 64,
        num_warps=4, num_stages=3,
    )
    _finish[(triton.cdiv(rows * 9728, 256),)](
        workspace, output, rows, split, 256, num_warps=4, enable_fp_fusion=False,
    )
    return output


def separate_gate_up(x, weight, projection):
    return swiglu(projection(x, weight))


@torch.inference_mode()
def select_gate_up(model, batch, projection):
    baseline = partial(separate_gate_up, projection=projection)
    if batch > 32:
        return baseline
    weights = [layer.mlp.gate_up_weight for layer in model.model.layers]
    x = torch.randn((batch, 1, 2560), dtype=weights[0].dtype, device=weights[0].device)
    reference = baseline(x, weights[0])
    native_ms = graph_time(baseline, x, weights)
    best, best_ms = baseline, native_ms
    for split in (2, 4, 8):
        candidate = partial(fused_gate_up, split=split)
        actual = candidate(x, weights[0])
        if not torch.allclose(actual, reference, rtol=0.016, atol=0.002):
            continue
        elapsed = graph_time(candidate, x, weights)
        if elapsed < best_ms:
            best, best_ms = candidate, elapsed
    return best if best_ms < native_ms * 0.98 else baseline

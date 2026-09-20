"""Causal full-prefix attention for two-token exact verification."""

import torch
import triton
import triton.language as tl
from kernels.attention import _combine


@triton.jit
def _verify_partial(Q, K, V, LENGTH, PART, STATS, capacity,
                 SPLIT: tl.constexpr, SCALE: tl.constexpr, BN: tl.constexpr):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    row = tl.arange(0, 16)
    d = tl.arange(0, 128)
    token_index = row // 4
    head = kv_head * 4 + row % 4
    q = tl.load(Q + ((batch * 2 + token_index[:, None]) * 32 + head[:, None]) * 128 + d[None, :],
                row[:, None] < 8, 0)
    length = tl.load(LENGTH + batch)
    chunk = ((capacity + BN * SPLIT - 1) // (BN * SPLIT)) * BN
    begin = split * chunk
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.full((16,), 0, tl.float32)
    acc = tl.full((16, 128), 0, tl.float32)
    if begin < length:
        for start in range(begin, begin + chunk, BN):
            token = start + tl.arange(0, BN)
            key = tl.load(K + (batch * capacity + token[None, :]) * 1024
                          + kv_head * 128 + d[:, None], token[None, :] < length, 0)
            score = tl.dot(q, key) * SCALE
            visible = (token[None, :] < length) & (token[None, :] <= length - 2 + token_index[:, None])
            score = tl.where(visible, score, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(score, 1))
            alpha = tl.where(next_maximum > -float("inf"), tl.exp2(maximum - next_maximum), 0.)
            probability = tl.where(visible, tl.exp2(score - next_maximum[:, None]), 0.)
            value = tl.load(V + (batch * capacity + token[:, None]) * 1024
                            + kv_head * 128 + d[None, :], token[:, None] < length, 0)
            acc = tl.dot(probability.to(tl.bfloat16), value, acc * alpha[:, None])
            denominator = denominator * alpha + tl.sum(probability, 1)
            maximum = next_maximum
    part = ((batch * 2 + token_index) * 32 + head) * SPLIT + split
    tl.store(PART + part[:, None] * 128 + d[None, :], acc, row[:, None] < 8)
    tl.store(STATS + part * 2, maximum, row < 8)
    tl.store(STATS + part * 2 + 1, denominator, row < 8)


def verify_attention(q, key, value, cu_query, cu_key, capacity, lengths, scale, split=8, block_n=64):
    batch = q.shape[0]
    partial = torch.empty((batch, 2, 32, split, 128), device=q.device, dtype=torch.float32)
    stats = torch.empty((batch, 2, 32, split, 2), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    _verify_partial[(batch, 8, split)](
        q, key, value, lengths, partial, stats, capacity,
        split, scale * 1.4426950408889634, block_n, num_warps=4, num_stages=2,
    )
    _combine[(batch * 2 * 32,)](partial, stats, out, split, num_warps=4)
    return out

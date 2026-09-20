"""Per-head normalization, rotary and KV store without changing projections."""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rotary(QIN, KIN, VIN, QW, KW, COS, SIN, QOUT, KCACHE, VCACHE,
                 n_cols, qeps, keps,
            TOKENS: tl.constexpr, CAPACITY: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    batch = row // TOKENS
    token = row % TOKENS
    col = tl.arange(0, 128)
    other = (col + 64) % 128
    if head < 32:
        source = QIN + row * 4096 + head * 128
        gain = QW
        eps = qeps
    else:
        source = KIN + row * 1024 + (head - 32) * 128
        gain = KW
        eps = keps
    x = tl.load(source + col).to(tl.float32)
    rotated = tl.load(source + other).to(tl.float32)
    # Match the worked RMSNorm kernel: FP32 reduction, BF16 normalized
    # value, then gain and another BF16 rounding boundary.
    variance = tl.sum(x * x, axis=0) / n_cols
    scale = tl.math.rsqrt(variance + eps)
    normal = (x * scale).to(tl.bfloat16)
    other_normal = (rotated * scale).to(tl.bfloat16)
    weight = tl.load(gain + col)
    other_weight = tl.load(gain + other)
    x = (normal * weight).to(tl.bfloat16).to(tl.float32)
    rotated = (other_normal * other_weight).to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(col < 64, -rotated, rotated)
    cosine = tl.load(COS + token * 128 + col).to(tl.float32)
    sine = tl.load(SIN + token * 128 + col).to(tl.float32)
    left = (x * cosine).to(tl.bfloat16).to(tl.float32)
    right = (rotated * sine).to(tl.bfloat16).to(tl.float32)
    result = (left + right).to(tl.bfloat16)
    if head < 32:
        tl.store(QOUT + row * 4096 + head * 128 + col, result)
    else:
        destination = (batch * CAPACITY + token) * 1024 + (head - 32) * 128 + col
        tl.store(KCACHE + destination, result)
        value = tl.load(VIN + row * 1024 + (head - 32) * 128 + col)
        tl.store(VCACHE + destination, value)


def norm_rotary_store(query, key, value, attn, position_embeddings, key_cache, value_cache):
    batch, length = query.shape[:2]
    output = torch.empty_like(query)
    cosine, sine = position_embeddings
    _norm_rotary[(batch * length, 40)](
        query, key, value, attn.q_norm.weight, attn.k_norm.weight,
        cosine, sine, output, key_cache, value_cache, 128,
        attn.q_norm.variance_epsilon, attn.k_norm.variance_epsilon,
        length, key_cache.shape[1], num_warps=4,
    )
    return output

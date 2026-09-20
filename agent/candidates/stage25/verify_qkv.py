"""BF16 Q/K norm, rotary, and token-major KV write for two-token verification."""

import torch
import triton
import triton.language as tl


@triton.jit
def _verify_qkv(QKV, QW, KW, COS, SIN, POS, Q, K, V,
                  CAPACITY: tl.constexpr, QEPS: tl.constexpr, KEPS: tl.constexpr):
    row = tl.program_id(0)
    batch = row // 2
    token = row % 2
    head = tl.program_id(1)
    col = tl.arange(0, 128)
    other = (col + 64) % 128
    # Packed rows: 32 query heads, 8 key heads, 8 value heads.
    source = QKV + row * 6144 + head * 128
    x = tl.load(source + col).to(tl.float32)
    rotated_x = tl.load(source + other).to(tl.float32)
    if head < 32:
        weight = tl.load(QW + col).to(tl.float32)
        rotated_weight = tl.load(QW + other).to(tl.float32)
        eps = QEPS
    else:
        weight = tl.load(KW + col).to(tl.float32)
        rotated_weight = tl.load(KW + other).to(tl.float32)
        eps = KEPS
    scale = tl.math.rsqrt(tl.sum(x * x, 0) / 128 + eps)
    # Reference norm rounds both its normalized input and its weighted output.
    normal = (x * scale).to(tl.bfloat16).to(tl.float32)
    rotated = (rotated_x * scale).to(tl.bfloat16).to(tl.float32)
    normal = (normal * weight).to(tl.bfloat16).to(tl.float32)
    rotated = (rotated * rotated_weight).to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(col < 64, -rotated, rotated)
    cosine = tl.load(COS + row * 128 + col).to(tl.float32)
    sine = tl.load(SIN + row * 128 + col).to(tl.float32)
    # Native eager RoPE materializes each BF16 product before adding them.
    left = (normal * cosine).to(tl.bfloat16).to(tl.float32)
    right = (rotated * sine).to(tl.bfloat16).to(tl.float32)
    result = (left + right).to(tl.bfloat16)
    if head < 32:
        tl.store(Q + row * 4096 + head * 128 + col, result)
    else:
        position = tl.load(POS + batch) + token
        destination = (batch * CAPACITY + position) * 1024 + (head - 32) * 128 + col
        tl.store(K + destination, result)
        value = tl.load(QKV + row * 6144 + 5120 + (head - 32) * 128 + col)
        tl.store(V + destination, value)


def verify_qkv(qkv, attn, position_embeddings, position, key, value, capacity):
    """Contiguous [B,2,6144] BF16 input; writes both verification slots."""
    batch = qkv.shape[0]
    query = torch.empty((batch, 2, 32, 128), dtype=qkv.dtype, device=qkv.device)
    cosine, sine = position_embeddings
    _verify_qkv[(batch * 2, 40)](
        qkv, attn.q_norm.weight, attn.k_norm.weight, cosine, sine,
        position, query, key, value, capacity,
        attn.q_norm.variance_epsilon, attn.k_norm.variance_epsilon,
        num_warps=4, enable_fp_fusion=False,
    )
    return query

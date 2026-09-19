"""Native GQA prefill; original projections and BF16 boundaries."""

import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


def prefill(model, input_ids, cache):
    if cache.length != 0:
        raise ValueError("Causal prefill requires a reset cache")
    base = model.model
    batch, length = input_ids.shape
    x = base.embed_tokens(input_ids)
    positions = torch.arange(length, device=input_ids.device)
    position_embeddings = base.rotary_emb(x, positions.unsqueeze(0))
    for index, layer in enumerate(base.layers):
        attn = layer.self_attn
        n = layer.input_layernorm(x)
        shape = (batch, length, -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(n).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(n).view(shape)).transpose(1, 2)
        v = attn.v_proj(n).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        # Native dense FlashAttention accepts [B,T,H,D], with Hq/Hkv=4.
        # All prompt queries and keys start at position zero, so is_causal=True
        # is the exact mask. No padded or uninitialized cache slots are passed.
        a = torch.ops.aten._flash_attention_forward(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            None, None, length, length, 0.0, True, False,
            scale=attn.scaling,
        )[0]
        cache.update(k, v, index)
        x = x + attn.o_proj(a.reshape(batch, length, -1))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    cache.length = length
    return model.lm_head(base.norm(x[:, -1:, :]))

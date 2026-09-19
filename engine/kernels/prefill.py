"""Packed/fused prefill using the shared BF16 Q/K epilogue."""

import torch
from torch.nn.functional import linear
from kernels.qkv_epilogue import qkv_epilogue


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
        packed = linear(layer.input_layernorm(x), attn.qkv_weight)
        q = qkv_epilogue(
            packed, attn, position_embeddings, positions,
            cache.key_tokens[index], cache.value_tokens[index], cache.capacity,
        )
        a = torch.ops.aten._flash_attention_forward(
            q, cache.key_tokens[index][:, :length], cache.value_tokens[index][:, :length],
            None, None, length, length, 0.0, True, False,
            scale=attn.scaling,
        )[0]
        x = x + attn.o_proj(a.reshape(batch, length, -1))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    cache.length = length
    return model.lm_head(base.norm(x[:, -1:, :]))

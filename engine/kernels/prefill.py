"""Original prefill projections/norms with exact BF16 rotary/cache fusion."""

import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from kernels.prefill_rotary import rotary_store
from kernels.prefill_pointwise import residual_norm, swiglu


def checked_residual_norm(x, branch, norm, check):
    residual, normal = residual_norm(x, branch, norm)
    if check:
        reference = x + branch
        if not (torch.equal(residual, reference)
                and torch.equal(normal, norm(reference))):
            raise RuntimeError("Prefill residual/norm fusion differs from native BF16 output")
    return residual, normal


def prefill(model, input_ids, cache, check_rotary=False, blas_backend=None):
    if cache.length != 0:
        raise ValueError("Causal prefill requires a reset cache")
    base = model.model
    batch, length = input_ids.shape
    x = base.embed_tokens(input_ids)
    positions = torch.arange(length, device=input_ids.device)
    position_embeddings = base.rotary_emb(x, positions.unsqueeze(0))
    previous_backend = torch.backends.cuda.preferred_blas_library()
    if blas_backend is not None:
        torch.backends.cuda.preferred_blas_library(blas_backend)
    try:
        n = base.layers[0].input_layernorm(x)
        for index, layer in enumerate(base.layers):
            attn = layer.self_attn
            shape = (batch, length, -1, attn.head_dim)
            q = attn.q_norm(attn.q_proj(n).view(shape))
            k = attn.k_norm(attn.k_proj(n).view(shape))
            v = attn.v_proj(n).view(shape)
            rotated = rotary_store(q, k, v, position_embeddings,
                                   cache.key_tokens[index], cache.value_tokens[index])
            if check_rotary:
                # Called in eager warmup only, never inside CUDA graph capture.
                reference_q, reference_k = apply_rotary_pos_emb(
                    q.transpose(1, 2), k.transpose(1, 2), *position_embeddings,
                )
                if not (torch.equal(rotated, reference_q.transpose(1, 2))
                        and torch.equal(cache.key_tokens[index][:, :length],
                                        reference_k.transpose(1, 2))
                        and torch.equal(cache.value_tokens[index][:, :length], v)):
                    raise RuntimeError("Prefill rotary/cache fusion differs from native BF16 output")
            a = torch.ops.aten._flash_attention_forward(
                rotated, cache.key_tokens[index][:, :length], cache.value_tokens[index][:, :length],
                None, None, length, length, 0.0, True, False,
                scale=attn.scaling,
            )[0]
            x, n = checked_residual_norm(
                x, attn.o_proj(a.reshape(batch, length, -1)),
                layer.post_attention_layernorm, check_rotary,
            )
            mlp = layer.mlp
            gate, up = mlp.gate_proj(n), mlp.up_proj(n)
            activated = swiglu(gate, up)
            if check_rotary and not torch.equal(activated, mlp.act_fn(gate) * up):
                raise RuntimeError("Prefill SwiGLU fusion differs from native BF16 output")
            branch = mlp.down_proj(activated)
            if index + 1 < len(base.layers):
                x, n = checked_residual_norm(
                    x, branch, base.layers[index + 1].input_layernorm, check_rotary,
                )
            else:
                x = x + branch
        cache.length = length
        return model.lm_head(base.norm(x[:, -1:, :]))
    finally:
        if blas_backend is not None:
            torch.backends.cuda.preferred_blas_library(previous_backend)

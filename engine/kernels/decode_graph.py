"""A single-token native Qwen step captured with PyTorch 2.5.1 CUDA graphs.

KV storage, token IDs and absolute position have stable CUDA addresses. Only
their contents change between replays. Host conversion is the caller's job.
"""

import torch
from torch.nn.functional import linear
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


class GraphCache:
    def __init__(self, storage, position):
        self.keys = storage.keys
        self.values = storage.values
        self.flat_keys = [key.flatten(0, 1) for key in storage.key_tokens]
        self.flat_values = [value.flatten(0, 1) for value in storage.value_tokens]
        self.position = position

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        key = self.keys[layer_idx]
        value = self.values[layer_idx]
        key.index_copy_(2, self.position, key_states)
        value.index_copy_(2, self.position, value_states)
        # Valid lengths are supplied separately to the native FlashAttention op.
        return key, value


class DecodeGraph:
    @torch.inference_mode()
    def __init__(self, model, storage, first_token, first_position):
        self.model = model
        self.tokens = torch.empty_like(first_token)
        self.position = torch.empty(1, dtype=torch.int64, device=first_token.device)
        batch = first_token.shape[0]
        self.capacity = storage.capacity
        self.cu_query = torch.arange(batch + 1, dtype=torch.int32, device=first_token.device)
        self.cu_key = self.cu_query * storage.capacity
        self.valid_lengths = torch.empty(batch, dtype=torch.int32, device=first_token.device)
        self.cache = GraphCache(storage, self.position)
        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        current_stream = torch.cuda.current_stream()
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            # Initialize cuBLAS/SDPA and allocator paths before capture. Each
            # warmup overwrites the same next slot; prompt slots stay intact.
            for _ in range(3):
                self.reset(first_token, first_position)
                self.step()
        current_stream.wait_stream(stream)
        self.reset(first_token, first_position)
        stream.wait_stream(current_stream)
        with torch.cuda.graph(self.graph, stream=stream):
            self.step()
        current_stream.wait_stream(stream)
        self.reset(first_token, first_position)

    def reset(self, first_token, first_position):
        self.tokens.copy_(first_token)
        self.position.fill_(first_position)
        self.valid_lengths.fill_(first_position + 1)

    def step(self):
        base = self.model.model
        x = base.embed_tokens(self.tokens)
        position_ids = self.position.unsqueeze(0)
        position_embeddings = base.rotary_emb(x, position_ids)
        batch = self.tokens.shape[0]
        for layer_idx, layer in enumerate(base.layers):
            attn = layer.self_attn
            n = layer.input_layernorm(x)
            head_shape = (batch, 1, -1, attn.head_dim)
            q_width = attn.q_proj.out_features
            kv_width = attn.k_proj.out_features
            q, k, v = linear(n, attn.qkv_weight).split(
                (q_width, kv_width, kv_width), dim=-1,
            )
            q = attn.q_norm(q.reshape(head_shape)).transpose(1, 2)
            k = attn.k_norm(k.reshape(head_shape)).transpose(1, 2)
            v = v.reshape(head_shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
            self.cache.update(k, v, layer_idx)
            # PyTorch 2.5.1's variable-length FlashAttention entry point accepts
            # GQA directly. cu_key describes reserved batch segments; seqused_k
            # limits each segment to the device-side initialized prefix. These
            # int32 CUDA tensors keep the operator safe for graph replay.
            a = torch.ops.aten._flash_attention_forward(
                q.squeeze(2),
                self.cache.flat_keys[layer_idx],
                self.cache.flat_values[layer_idx],
                self.cu_query, self.cu_key,
                1, self.capacity, 0.0, False, False,
                scale=attn.scaling, seqused_k=self.valid_lengths,
            )[0]
            x = x + attn.o_proj(a.reshape(batch, 1, -1))
            mlp = layer.mlp
            gate, up = linear(
                layer.post_attention_layernorm(x), mlp.gate_up_weight,
            ).chunk(2, dim=-1)
            x = x + mlp.down_proj(mlp.act_fn(gate) * up)
        logits = self.model.lm_head(base.norm(x))
        self.tokens.copy_(logits[:, -1, :].argmax(dim=-1, keepdim=True))
        self.position.add_(1)
        self.valid_lengths.add_(1)

"""A single-token native Qwen step captured with PyTorch 2.5.1 CUDA graphs.

KV storage, token IDs and absolute position have stable CUDA addresses. Only
their contents change between replays. Host conversion is the caller's job.
"""

import torch


class GraphCache:
    def __init__(self, storage, position):
        self.keys = storage.keys
        self.values = storage.values
        self.position = position

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        key = self.keys[layer_idx]
        value = self.values[layer_idx]
        key.index_copy_(2, self.position, key_states)
        value.index_copy_(2, self.position, value_states)
        # Fixed capacity is safe only with DecodeGraph's valid-prefix mask.
        return key, value


class DecodeGraph:
    @torch.inference_mode()
    def __init__(self, model, storage, first_token, first_position):
        self.model = model
        self.tokens = torch.empty_like(first_token)
        self.position = torch.empty(1, dtype=torch.int64, device=first_token.device)
        self.key_positions = torch.arange(storage.capacity, device=first_token.device)
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

    def step(self):
        base = self.model.model
        x = base.embed_tokens(self.tokens)
        position_ids = self.position.unsqueeze(0)
        position_embeddings = base.rotary_emb(x, position_ids)
        attention_mask = (self.key_positions <= self.position).view(1, 1, 1, -1)
        for layer in base.layers:
            x = layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=self.cache,
                use_cache=True,
                cache_position=self.position,
                position_embeddings=position_embeddings,
            )[0]
        logits = self.model.lm_head(base.norm(x))
        self.tokens.copy_(logits[:, -1, :].argmax(dim=-1, keepdim=True))
        self.position.add_(1)

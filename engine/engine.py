"""Native causal prefill and a CUDA-graph single-token decode loop."""

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from kernels.cache import PrefixCache
from kernels.decode_graph import DecodeGraph
from kernels.rmsnorm import FusedRMSNorm
from kernels.projections import pack_projections


@torch.inference_mode()
def qwen_forward(model, input_ids, cache):
    base = model.model
    x = base.embed_tokens(input_ids)
    length = input_ids.shape[1]
    end = cache.length + length
    positions = torch.arange(cache.length, end, device=input_ids.device)
    position_ids = positions.unsqueeze(0)
    position_embeddings = base.rotary_emb(x, position_ids)
    if cache.length == 0:
        # Native empty-cache causal prefill keeps SDPA's fast causal dispatch.
        # An explicit prefill mask caused a measured 1.28x TTFT regression.
        active_cache = DynamicCache()
        attention_mask = None
    else:
        active_cache = cache
        # Boolean SDPA masks use True for visible keys. The static storage
        # exposes only initialized prefixes, including an offset multi-token call.
        keys = torch.arange(end, device=input_ids.device)
        attention_mask = keys[None, None, None, :] <= positions[None, None, :, None]
    for layer in base.layers:
        x = layer(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=active_cache,
            use_cache=True,
            cache_position=positions,
            position_embeddings=position_embeddings,
        )[0]
    if active_cache is not cache:
        for layer_idx, (key, value) in enumerate(zip(active_cache.key_cache, active_cache.value_cache)):
            cache.update(key, value, layer_idx)
    cache.length = end
    x = base.norm(x)
    return model.lm_head(x[:, -1:, :])


class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to("cuda:0")
        )
        base = self.model.model
        base.norm = FusedRMSNorm(base.norm)
        for layer in base.layers:
            layer.input_layernorm = FusedRMSNorm(layer.input_layernorm)
            layer.post_attention_layernorm = FusedRMSNorm(layer.post_attention_layernorm)
            layer.self_attn.q_norm = FusedRMSNorm(layer.self_attn.q_norm)
            layer.self_attn.k_norm = FusedRMSNorm(layer.self_attn.k_norm)
        pack_projections(self.model)
        self.cache = None
        self.cache_shape = None
        self.decoder = None

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yields a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Every sequence has the same length.
        Never stops at end-of-sequence tokens.
        """
        with torch.inference_mode():
            if max_new_tokens <= 0:
                return
            current = torch.tensor(input_ids, dtype=torch.int64, device="cuda:0")
            batch, prompt_length = current.shape
            shape = (batch, prompt_length + max_new_tokens)
            if shape != self.cache_shape:
                # Release an old shape before allocating its replacement.
                self.decoder = None
                self.cache = None
                self.cache = PrefixCache(
                    self.model.config, batch, shape[1], current.device,
                    self.model.dtype,
                )
                self.cache_shape = shape
            self.cache.reset()
            logits = qwen_forward(self.model, current, self.cache)
            current = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            yield current[:, 0].tolist()
            if max_new_tokens == 1:
                return
            if self.decoder is None:
                # Each workload supplies an untimed warmup of the same shape.
                self.decoder = DecodeGraph(self.model, self.cache, current, prompt_length)
            self.decoder.reset(current, prompt_length)
            for _ in range(max_new_tokens - 1):
                self.decoder.graph.replay()
                yield self.decoder.tokens[:, 0].tolist()

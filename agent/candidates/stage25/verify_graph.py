"""Two-token exact verification with independent per-sequence cache lengths."""

from functools import partial
import torch
from torch.nn.functional import linear

from kernels.decode_graph import GraphCache
from kernels.pointwise import residual_norm, swiglu
from kernels.select_attention import measure
from kernels.verify_attention import verify_attention
from kernels.verify_qkv import verify_qkv
from kernels.verify_state import commit, propose


def native_attention(q, key, value, cu_query, cu_key, capacity, lengths, scale):
    return torch.ops.aten._flash_attention_forward(
        q.flatten(0, 1), key, value, cu_query, cu_key, 2, capacity, 0.0, True, False,
        scale=scale, seqused_k=lengths,
    )[0].view_as(q)


def select_attention(decoder):
    cache = decoder.cache
    q = torch.randn((decoder.batch, 2, 32, 128), device=decoder.tokens.device,
                    dtype=cache.flat_keys[0].dtype)
    lengths = torch.full_like(decoder.valid_lengths, decoder.prompt_length + 2)
    scale = decoder.model.model.layers[0].self_attn.scaling
    common = (decoder.cu_query, decoder.cu_key, decoder.capacity, lengths, scale)
    args = (q, cache.flat_keys, cache.flat_values, *common)
    native_ms = measure(native_attention, *args)
    best, best_ms = native_attention, native_ms
    for operation in (partial(verify_attention, split=8),
                      partial(verify_attention, split=16, block_n=32)):
        valid = True
        for length in sorted({2, max(2, decoder.prompt_length // 2),
                              decoder.prompt_length + 2}):
            lengths.fill_(length)
            for index in (0, len(cache.flat_keys) - 1):
                test = (q, cache.flat_keys[index], cache.flat_values[index], *common)
                if not torch.allclose(operation(*test), native_attention(*test),
                                      rtol=0.016, atol=0.002):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            lengths.fill_(decoder.prompt_length + 2)
            elapsed = measure(operation, *args)
            if elapsed < best_ms:
                best, best_ms = operation, elapsed
    return best if best_ms < native_ms * .9 else native_attention


class VerifyGraph:
    @torch.inference_mode()
    def __init__(self, model, storage, prompt, first, output_length):
        self.model = model
        self.batch, self.prompt_length = prompt.shape
        self.output_length = output_length
        self.capacity = storage.capacity
        device = prompt.device
        self.tokens = torch.empty((self.batch, 2), dtype=torch.int64, device=device)
        self.position = torch.empty(self.batch, dtype=torch.int64, device=device)
        self.valid_lengths = torch.empty(self.batch, dtype=torch.int32, device=device)
        self.count = torch.empty_like(self.valid_lengths)
        self.output = torch.empty((self.batch, output_length), dtype=torch.int64, device=device)
        self.history = torch.empty((self.batch, self.prompt_length + output_length),
                                   dtype=torch.int64, device=device)
        self.offsets = torch.arange(2, device=device).unsqueeze(0)
        cumulative = torch.arange(self.batch + 1, dtype=torch.int32, device=device)
        self.cu_query = cumulative * 2
        self.cu_key = cumulative * self.capacity
        self.cache = GraphCache(storage, self.position)
        self.attention = native_attention
        self.graphs = {}
        stream = torch.cuda.Stream()
        current = torch.cuda.current_stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.reset(prompt, first)
                self.step()
        current.wait_stream(stream)
        self.attention = select_attention(self)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            self.reset(prompt, first)
            self.step()
        current.wait_stream(stream)
        for passes in (1, 2, 4, 8):
            self.reset(prompt, first)
            stream.wait_stream(current)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(passes):
                    self.step()
            current.wait_stream(stream)
            self.graphs[passes] = graph
        self.reset(prompt, first)

    def reset(self, prompt, first):
        self.tokens.copy_(first.expand(-1, 2))
        self.position.fill_(self.prompt_length)
        self.valid_lengths.fill_(self.prompt_length + 2)
        self.count.fill_(1)
        self.output[:, :1].copy_(first)
        self.history[:, :self.prompt_length].copy_(prompt)
        self.history[:, self.prompt_length:self.prompt_length + 1].copy_(first)
        propose(self)

    def step(self):
        base = self.model.model
        x = base.embed_tokens(self.tokens)
        positions = self.position[:, None] + self.offsets
        embeddings = base.rotary_emb(x, positions)
        n = base.layers[0].input_layernorm(x)
        for index, layer in enumerate(base.layers):
            attn = layer.self_attn
            q = verify_qkv(linear(n, attn.qkv_weight), attn, embeddings,
                           self.position, self.cache.flat_keys[index],
                           self.cache.flat_values[index], self.capacity)
            a = self.attention(q, self.cache.flat_keys[index], self.cache.flat_values[index],
                               self.cu_query, self.cu_key, self.capacity,
                               self.valid_lengths, attn.scaling)
            x, n = residual_norm(x, linear(a.reshape(self.batch, 2, 4096), attn.o_proj.weight),
                                 layer.post_attention_layernorm)
            activated = swiglu(linear(n, layer.mlp.gate_up_weight))
            norm = (base.layers[index + 1].input_layernorm
                    if index + 1 < len(base.layers) else base.norm)
            x, n = residual_norm(x, linear(activated, layer.mlp.down_proj.weight), norm)
        predictions = linear(n, self.model.lm_head.weight).argmax(-1)
        commit(self, predictions)


def group_size(remaining):
    maximum = max(1, remaining // 2)
    return next(size for size in (8, 4, 2, 1) if size <= maximum)


def stream_verify(decoder, first, max_new_tokens):
    available = 1
    decoder.graphs[group_size(max_new_tokens - available)].replay()
    yield first
    while available < max_new_tokens:
        complete = min(decoder.count.tolist())
        tokens = decoder.output[:, available:complete].T.tolist()
        available = complete
        if available < max_new_tokens:
            decoder.graphs[group_size(max_new_tokens - available)].replay()
        yield from tokens

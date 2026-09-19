"""Shared KV storage; eager updates expose only the initialized prefix.

GraphCache uses these same buffers at full capacity with an explicit mask.
"""

import torch


class PrefixCache:
    def __init__(self, config, batch, capacity, device, dtype):
        shape = (batch, config.num_key_value_heads, capacity, config.head_dim)
        self.keys = [
            torch.zeros(shape, device=device, dtype=dtype)
            for _ in range(config.num_hidden_layers)
        ]
        # Masked values must be finite: an uninitialized NaN could contaminate
        # attention even with a zero softmax weight. Zero once during allocation.
        self.values = [torch.zeros_like(key) for key in self.keys]
        self.capacity = capacity
        self.length = 0

    def reset(self):
        # Every exposed slot is overwritten before use; unused capacity stays hidden.
        self.length = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        end = self.length + key_states.shape[-2]
        if end > self.capacity:
            raise ValueError("KV cache capacity exceeded")
        key = self.keys[layer_idx]
        value = self.values[layer_idx]
        key[:, :, self.length:end, :].copy_(key_states)
        value[:, :, self.length:end, :].copy_(value_states)
        return key[:, :, :end, :], value[:, :, :end, :]

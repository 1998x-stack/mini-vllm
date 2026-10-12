"""Compare reserved token slots for short active requests."""

import torch

from mini_vllm.cache.contiguous import ContiguousKVCache
from mini_vllm.cache.paged import PagedKVCache

contiguous = ContiguousKVCache(num_layers=1, max_model_len=32, num_kv_heads=1, head_dim=2)
paged = PagedKVCache(num_layers=1, num_blocks=32, block_size=4, num_kv_heads=1, head_dim=2)
for cache in [contiguous, paged]:
    for request_id, length in [("A", 3), ("B", 6), ("C", 1)]:
        cache.ensure_capacity(request_id, length)
        for position in range(length):
            value = torch.ones(1, 2) * position
            cache.write(0, request_id, position, value, value)
    print(type(cache).__name__, cache.stats())

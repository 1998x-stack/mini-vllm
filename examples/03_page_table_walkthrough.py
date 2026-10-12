import torch

from mini_vllm.cache.paged import PagedKVCache

cache = PagedKVCache(num_layers=1, num_blocks=6, block_size=4, num_kv_heads=1, head_dim=2)
request_id = "A"
assert cache.ensure_capacity(request_id, 10)
print("block table:", cache.block_table(request_id))
for position in range(10):
    key = torch.tensor([[position, position + 0.1]])
    value = torch.tensor([[position + 100, position + 100.1]])
    cache.write(0, request_id, position, key, value)
    physical, offset = cache.slot_mapping(request_id, position)
    print(f"position={position:02d} -> logical={position // cache.block_size} physical={physical} offset={offset}")
keys, values = cache.read_prefix(0, request_id, 10)
print("gathered keys shape:", tuple(keys.shape))
print("before free:", cache.stats())
cache.free(request_id)
print("after free:", cache.stats())

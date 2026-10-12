import torch

from mini_vllm.cache.contiguous import ContiguousKVCache
from mini_vllm.cache.paged import PagedKVCache


def test_contiguous_cache_roundtrip() -> None:
    cache = ContiguousKVCache(num_layers=2, max_model_len=8, num_kv_heads=2, head_dim=4)
    assert cache.ensure_capacity("r", 3)
    for pos in range(3):
        k = torch.full((2, 4), float(pos)); v = torch.full((2, 4), float(pos + 10)); cache.write(1, "r", pos, k, v)
    keys, values = cache.read_prefix(1, "r", 3)
    assert keys.shape == (3, 2, 4)
    assert torch.equal(keys[2], torch.full((2, 4), 2.0))
    assert torch.equal(values[0], torch.full((2, 4), 10.0))


def test_paged_slot_mapping_and_roundtrip() -> None:
    cache = PagedKVCache(num_layers=1, num_blocks=4, block_size=2, num_kv_heads=1, head_dim=2)
    assert cache.ensure_capacity("r", 3)
    table = cache.block_table("r")
    assert len(table) == 2
    assert cache.slot_mapping("r", 0) == (table[0], 0)
    assert cache.slot_mapping("r", 1) == (table[0], 1)
    assert cache.slot_mapping("r", 2) == (table[1], 0)
    for pos in range(3):
        k = torch.tensor([[float(pos), float(pos + 1)]])
        cache.write(0, "r", pos, k, k + 10)
    keys, values = cache.read_prefix(0, "r", 3)
    assert torch.equal(keys[2], torch.tensor([[2.0, 3.0]]))
    assert torch.equal(values[1], torch.tensor([[11.0, 12.0]]))


def test_paged_blocks_are_recycled() -> None:
    cache = PagedKVCache(num_layers=1, num_blocks=2, block_size=2, num_kv_heads=1, head_dim=1)
    assert cache.ensure_capacity("A", 4)
    assert not cache.ensure_capacity("B", 1)
    cache.free("A")
    assert cache.ensure_capacity("B", 1)
    assert cache.stats()["used_blocks"] == 1

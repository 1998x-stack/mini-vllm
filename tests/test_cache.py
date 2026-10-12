import torch

from mini_vllm.cache.contiguous import ContiguousKVCache


def test_contiguous_cache_roundtrip() -> None:
    cache = ContiguousKVCache(num_layers=1, max_model_len=8, num_kv_heads=1, head_dim=2)
    assert cache.ensure_capacity("r", 3)
    for pos in range(3):
        k = torch.tensor([[float(pos), float(pos + 1)]])
        v = k + 10
        cache.write(0, "r", pos, k, v)
    keys, values = cache.read_prefix(0, "r", 3)
    assert torch.equal(keys[2], torch.tensor([[2.0, 3.0]]))
    assert torch.equal(values[0], torch.tensor([[10.0, 11.0]]))

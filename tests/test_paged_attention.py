import torch

from mini_vllm.attention import (
    PagedAttentionConfig,
    ReferencePagedAttention,
    VectorizedPagedAttention,
)
from mini_vllm.cache.contiguous import ContiguousKVCache
from mini_vllm.cache.paged import PagedKVCache
from mini_vllm.config import TinyModelConfig
from mini_vllm.model.tiny_transformer import TinyDecoderLM


def _fragmented_cache() -> PagedKVCache:
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=8,
        block_size=2,
        num_kv_heads=2,
        head_dim=3,
    )
    # Deliberately non-contiguous physical mapping. The test owns this internal
    # setup so the addressing invariant is visible rather than accidental.
    cache._block_tables["R"] = [5, 1, 7]
    cache._used_tokens["R"] = 0
    for position in range(5):
        base = float(position * 10)
        key = torch.arange(6, dtype=torch.float32).reshape(2, 3) + base
        value = key + 100.0
        cache.write(0, "R", position, key, value)
    # Unused tail slot must remain invisible when logical length is five.
    cache.keys[0, 7, 1].fill_(9999.0)
    cache.values[0, 7, 1].fill_(9999.0)
    return cache


def test_vectorized_prefix_read_matches_reference_for_fragmented_partial_tail():
    cache = _fragmented_cache()
    ref_k, ref_v = cache.read_prefix(0, "R", 5)
    vec_k, vec_v = cache.read_prefix_vectorized(0, "R", 5)
    assert torch.equal(vec_k, ref_k)
    assert torch.equal(vec_v, ref_v)
    assert not torch.any(vec_k == 9999.0)
    assert not torch.any(vec_v == 9999.0)


def test_vectorized_attention_matches_reference_with_gqa():
    cache = _fragmented_cache()
    q = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10.0
    reference = ReferencePagedAttention()
    vectorized = VectorizedPagedAttention()

    ref = reference.attend(
        q,
        cache=cache,
        layer_idx=0,
        request_id="R",
        length=5,
        n_heads=4,
        n_kv_heads=2,
        head_dim=3,
    )
    got = vectorized.attend(
        q,
        cache=cache,
        layer_idx=0,
        request_id="R",
        length=5,
        n_heads=4,
        n_kv_heads=2,
        head_dim=3,
    )
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-6)
    assert vectorized.stats.vectorized_calls == 1
    assert vectorized.stats.fallbacks == 0


def test_vectorized_backend_falls_back_for_nonpaged_cache():
    cache = ContiguousKVCache(
        num_layers=1,
        max_model_len=8,
        num_kv_heads=1,
        head_dim=2,
    )
    assert cache.ensure_capacity("R", 2)
    cache.write(0, "R", 0, torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]))
    cache.write(0, "R", 1, torch.tensor([[2.0, 3.0]]), torch.tensor([[4.0, 5.0]]))
    backend = VectorizedPagedAttention(
        PagedAttentionConfig(backend="vectorized", fallback_to_reference=True)
    )
    out = backend.attend(
        torch.tensor([[0.5, 1.0]]),
        cache=cache,
        layer_idx=0,
        request_id="R",
        length=2,
        n_heads=1,
        n_kv_heads=1,
        head_dim=2,
    )
    assert out.shape == (1, 2)
    assert backend.stats.fallbacks == 1
    assert backend.events[-1].mode == "fallback-reference"


def test_vectorized_backend_strict_mode_rejects_nonpaged_cache():
    cache = ContiguousKVCache(
        num_layers=1,
        max_model_len=4,
        num_kv_heads=1,
        head_dim=2,
    )
    assert cache.ensure_capacity("R", 1)
    cache.write(0, "R", 0, torch.zeros((1, 2)), torch.zeros((1, 2)))
    backend = VectorizedPagedAttention(
        PagedAttentionConfig(backend="vectorized", fallback_to_reference=False)
    )
    try:
        backend.attend(
            torch.zeros((1, 2)),
            cache=cache,
            layer_idx=0,
            request_id="R",
            length=1,
            n_heads=1,
            n_kv_heads=1,
            head_dim=2,
        )
    except RuntimeError as exc:
        assert "does not support vectorized paged reads" in str(exc)
    else:
        raise AssertionError("strict vectorized backend should reject contiguous cache")


def test_tiny_decoder_reference_and_vectorized_attention_match_logits():
    config = TinyModelConfig(
        vocab_size=24,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=32,
        max_position_embeddings=32,
        seed=23,
    )
    reference_model = TinyDecoderLM(
        config,
        attention_backend=ReferencePagedAttention(),
    )
    vector_model = TinyDecoderLM(
        config,
        attention_backend=VectorizedPagedAttention(),
    )
    ref_cache = PagedKVCache(
        num_layers=2,
        num_blocks=16,
        block_size=2,
        num_kv_heads=2,
        head_dim=4,
    )
    vec_cache = PagedKVCache(
        num_layers=2,
        num_blocks=16,
        block_size=2,
        num_kv_heads=2,
        head_dim=4,
    )
    assert ref_cache.ensure_capacity("R", 5)
    assert vec_cache.ensure_capacity("R", 5)
    token_ids = (1, 2, 3, 4, 5)
    ref = reference_model.forward_chunk_logits(
        request_id="R", token_ids=token_ids, start_pos=0, cache=ref_cache
    )
    got = vector_model.forward_chunk_logits(
        request_id="R", token_ids=token_ids, start_pos=0, cache=vec_cache
    )
    assert len(got) == len(ref)
    for actual, expected in zip(got, ref, strict=True):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

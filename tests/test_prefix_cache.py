import torch

from mini_vllm import CacheConfig, EngineConfig, Request, TinyModelConfig, create_engine
from mini_vllm.cache.prefix import PrefixCachingPagedKVCache


def tiny_model(seed: int = 17) -> TinyModelConfig:
    return TinyModelConfig(
        vocab_size=32,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        d_ff=32,
        max_position_embeddings=64,
        seed=seed,
    )


def test_warm_request_reuses_full_prefix_blocks_and_preserves_output() -> None:
    engine = create_engine(
        "v0.5",
        engine_config=EngineConfig(
            max_num_seqs=1,
            max_model_len=32,
            max_num_batched_tokens=8,
        ),
        model_config=tiny_model(),
        cache_config=CacheConfig(block_size=2, num_blocks=16),
    )
    assert isinstance(engine.cache, PrefixCachingPagedKVCache)

    prompt = [1, 2, 3, 4, 5]
    engine.add_request(Request("cold", prompt.copy(), max_new_tokens=4))
    engine.run()
    cold_output = engine.get_request("cold").output_token_ids.copy()

    # Finished requests release active ownership, but immutable full prompt
    # blocks remain indexed for future reuse.
    assert engine.cache.stats()["cached_blocks"] >= 2
    assert engine.cache.stats()["active_block_refs"] == 0

    warm = Request("warm", prompt.copy(), max_new_tokens=4)
    engine.add_request(warm)
    assert warm.num_computed_tokens == 4
    assert engine.cache.stats()["prefix_hit_tokens"] >= 4

    warm_trace = engine.run()
    assert warm_trace[0].work[0].start_pos == 4
    assert engine.get_request("warm").output_token_ids == cold_output


def test_shared_prefix_blocks_are_refcounted_and_tail_is_private() -> None:
    engine = create_engine(
        "v0.5",
        engine_config=EngineConfig(
            max_num_seqs=2,
            max_model_len=32,
            max_num_batched_tokens=4,
        ),
        model_config=tiny_model(seed=23),
        cache_config=CacheConfig(block_size=2, num_blocks=16),
    )
    cache = engine.cache
    assert isinstance(cache, PrefixCachingPagedKVCache)

    first = Request("A", [1, 2, 3, 4, 5], max_new_tokens=3)
    engine.add_request(first)

    # First step computes exactly two full prompt blocks and publishes them.
    engine.step()
    assert first.num_computed_tokens == 4
    shared_table = cache.block_table("A")
    assert len(shared_table) == 2
    snapshot = cache.keys[:, shared_table[0]].clone()

    second = Request("B", [1, 2, 3, 4, 9], max_new_tokens=2)
    engine.add_request(second)
    assert second.num_computed_tokens == 4
    assert cache.block_table("B") == shared_table
    assert all(cache.block_refcount(block) == 2 for block in shared_table)

    # B's non-full suffix is not shared. It is allocated as a private tail,
    # while shared prefix K/V remains byte-for-byte unchanged.
    engine.step()
    b_table = cache.block_table("B")
    assert len(b_table) == 3
    assert b_table[:2] == shared_table
    assert cache.block_refcount(b_table[2]) == 1
    assert not cache.is_cached_block(b_table[2])
    assert torch.equal(cache.keys[:, shared_table[0]], snapshot)


def test_exact_prompt_hit_keeps_work_for_next_token_logits() -> None:
    engine = create_engine(
        "v0.5",
        engine_config=EngineConfig(
            max_num_seqs=1,
            max_model_len=16,
            max_num_batched_tokens=8,
        ),
        model_config=tiny_model(seed=31),
        cache_config=CacheConfig(block_size=2, num_blocks=8),
    )
    prompt = [2, 4, 6, 8]
    engine.add_request(Request("seed", prompt.copy(), max_new_tokens=1))
    engine.run()

    warm = Request("warm", prompt.copy(), max_new_tokens=1)
    engine.add_request(warm)

    # Prompt length is exactly two blocks, but KV cache has no logits. The
    # final full block is therefore intentionally recomputed.
    assert warm.num_computed_tokens == 2
    step = engine.step()
    assert step.work[0].start_pos == 2
    assert step.work[0].end_pos == 4
    assert "warm" in step.sampled_token_ids


def test_unreferenced_cached_blocks_are_evicted_under_pressure() -> None:
    cache = PrefixCachingPagedKVCache(
        num_layers=1,
        num_blocks=2,
        block_size=2,
        num_kv_heads=1,
        head_dim=2,
    )
    key = torch.tensor([[1.0, 2.0]])
    value = torch.tensor([[3.0, 4.0]])

    assert cache.prepare_request("A", [1, 2, 3]) == 0
    assert cache.ensure_capacity("A", 2)
    cache.write(0, "A", 0, key, value)
    cache.write(0, "A", 1, key, value)
    cache.commit_work("A", start_pos=0, token_ids=(1, 2), prompt_length=3)
    cached_block = cache.block_table("A")[0]
    assert cache.is_cached_block(cached_block)
    cache.free("A")
    assert cache.block_refcount(cached_block) == 0
    assert cache.stats()["cached_blocks"] == 1

    # B needs both physical blocks. The one-block free list is insufficient,
    # so v0.5 evicts the unreferenced cached block before allocating.
    assert cache.prepare_request("B", [7, 8, 9]) == 0
    assert cache.ensure_capacity("B", 3)
    assert cache.stats()["prefix_evictions"] == 1
    assert cache.stats()["cached_blocks"] == 0
    assert len(cache.block_table("B")) == 2

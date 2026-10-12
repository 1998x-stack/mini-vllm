from mini_vllm import CacheConfig, EngineConfig, Request, TinyModelConfig, create_engine
from mini_vllm.cache.paged import PagedKVCache
from mini_vllm.scheduler.budgeted import BudgetedContinuousScheduler


def model_config(seed: int = 7) -> TinyModelConfig:
    return TinyModelConfig(
        vocab_size=32,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        d_ff=32,
        max_position_embeddings=32,
        seed=seed,
    )


def test_v04_composition_and_chunked_paged_growth() -> None:
    engine = create_engine(
        "v0.4",
        engine_config=EngineConfig(
            max_num_seqs=1,
            max_model_len=16,
            max_num_batched_tokens=2,
        ),
        model_config=model_config(),
        cache_config=CacheConfig(block_size=2, num_blocks=16),
    )
    assert isinstance(engine.scheduler, BudgetedContinuousScheduler)
    assert isinstance(engine.cache, PagedKVCache)

    engine.add_request(Request("A", [1, 2, 3, 4, 5], max_new_tokens=2))
    step0 = engine.step()
    assert step0.cache_stats["used_blocks"] == 1
    assert step0.sampled_token_ids == {}
    step1 = engine.step()
    assert step1.cache_stats["used_blocks"] == 2
    assert step1.sampled_token_ids == {}
    step2 = engine.step()
    assert step2.cache_stats["used_blocks"] == 3
    assert "A" in step2.sampled_token_ids


def test_chunked_and_unchunked_execution_produce_same_tokens() -> None:
    cache_config = CacheConfig(block_size=2, num_blocks=32)
    unchunked = create_engine(
        "v0.3",
        engine_config=EngineConfig(
            max_num_seqs=1,
            max_model_len=32,
            max_num_batched_tokens=32,
        ),
        model_config=model_config(seed=77),
        cache_config=cache_config,
    )
    chunked = create_engine(
        "v0.4",
        engine_config=EngineConfig(
            max_num_seqs=1,
            max_model_len=32,
            max_num_batched_tokens=2,
        ),
        model_config=model_config(seed=77),
        cache_config=cache_config,
    )
    prompt = [1, 7, 4, 9, 3]
    unchunked.add_request(Request("r", prompt.copy(), max_new_tokens=4))
    chunked.add_request(Request("r", prompt.copy(), max_new_tokens=4))

    unchunked.run()
    chunked.run()

    assert chunked.get_request("r").output_token_ids == unchunked.get_request("r").output_token_ids

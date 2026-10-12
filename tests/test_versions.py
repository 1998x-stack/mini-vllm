from mini_vllm import CacheConfig, EngineConfig, TinyModelConfig, create_engine
from mini_vllm.cache.contiguous import ContiguousKVCache
from mini_vllm.scheduler.static import StaticBatchScheduler


def test_v01_composition() -> None:
    engine = create_engine(
        "v0.1",
        engine_config=EngineConfig(max_num_seqs=2, max_model_len=16),
        model_config=TinyModelConfig(vocab_size=24, d_model=16, n_heads=4, n_kv_heads=2, n_layers=1, d_ff=32, max_position_embeddings=32),
        cache_config=CacheConfig(),
    )
    assert isinstance(engine.scheduler, StaticBatchScheduler)
    assert isinstance(engine.cache, ContiguousKVCache)

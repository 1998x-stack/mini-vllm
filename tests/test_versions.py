from mini_vllm import CacheConfig, EngineConfig, TinyModelConfig, create_engine
from mini_vllm.cache.contiguous import ContiguousKVCache
from mini_vllm.scheduler.continuous import ContinuousBatchScheduler
from mini_vllm.scheduler.static import StaticBatchScheduler


def cfg():
    return EngineConfig(max_num_seqs=2, max_model_len=16), TinyModelConfig(vocab_size=24, d_model=16, n_heads=4, n_kv_heads=2, n_layers=1, d_ff=32, max_position_embeddings=32), CacheConfig()


def test_version_composition() -> None:
    e, m, c = cfg()
    a = create_engine("v0.1", engine_config=e, model_config=m, cache_config=c)
    b = create_engine("v0.2", engine_config=e, model_config=m, cache_config=c)
    assert isinstance(a.scheduler, StaticBatchScheduler)
    assert isinstance(a.cache, ContiguousKVCache)
    assert isinstance(b.scheduler, ContinuousBatchScheduler)
    assert isinstance(b.cache, ContiguousKVCache)

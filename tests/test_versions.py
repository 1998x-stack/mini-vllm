from mini_vllm import CacheConfig, EngineConfig, Request, TinyModelConfig, create_engine
from mini_vllm.cache.contiguous import ContiguousKVCache
from mini_vllm.cache.paged import PagedKVCache
from mini_vllm.cache.prefix import PrefixCachingPagedKVCache
from mini_vllm.scheduler.continuous import ContinuousBatchScheduler
from mini_vllm.scheduler.budgeted import BudgetedContinuousScheduler
from mini_vllm.scheduler.static import StaticBatchScheduler
from mini_vllm.scheduler.preemptive import PreemptiveBudgetedScheduler


def configs():
    return (
        EngineConfig(max_num_seqs=2, max_model_len=16),
        TinyModelConfig(
            vocab_size=24,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            n_layers=1,
            d_ff=32,
            max_position_embeddings=32,
        ),
        CacheConfig(block_size=2, num_blocks=16),
    )


def test_version_composition() -> None:
    e, m, c = configs()
    v01 = create_engine("v0.1", engine_config=e, model_config=m, cache_config=c)
    v02 = create_engine("v0.2", engine_config=e, model_config=m, cache_config=c)
    v03 = create_engine("v0.3", engine_config=e, model_config=m, cache_config=c)
    v04 = create_engine("v0.4", engine_config=e, model_config=m, cache_config=c)
    v05 = create_engine("v0.5", engine_config=e, model_config=m, cache_config=c)
    v06 = create_engine("v0.6", engine_config=e, model_config=m, cache_config=c)

    assert isinstance(v01.scheduler, StaticBatchScheduler)
    assert isinstance(v01.cache, ContiguousKVCache)
    assert isinstance(v02.scheduler, ContinuousBatchScheduler)
    assert isinstance(v02.cache, ContiguousKVCache)
    assert isinstance(v03.scheduler, ContinuousBatchScheduler)
    assert isinstance(v03.cache, PagedKVCache)
    assert isinstance(v04.scheduler, BudgetedContinuousScheduler)
    assert isinstance(v04.cache, PagedKVCache)
    assert isinstance(v05.scheduler, BudgetedContinuousScheduler)
    assert isinstance(v05.cache, PrefixCachingPagedKVCache)
    assert isinstance(v06.scheduler, PreemptiveBudgetedScheduler)
    assert isinstance(v06.cache, PrefixCachingPagedKVCache)


def test_v03_end_to_end_finishes_and_frees_cache() -> None:
    e, m, c = configs()
    engine = create_engine("v0.3", engine_config=e, model_config=m, cache_config=c)
    engine.add_request(Request("A", [1, 2, 3], max_new_tokens=2))
    engine.add_request(Request("B", [4], max_new_tokens=3))
    trace = engine.run()
    assert trace
    assert engine.get_request("A").finished
    assert engine.get_request("B").finished
    assert engine.cache.stats()["used_blocks"] == 0

from mini_vllm import CacheConfig, EngineConfig, Request, TinyModelConfig, create_engine
from mini_vllm.cache.transactional import TransactionalPrefixCachingPagedKVCache
from mini_vllm.execution import LocalExecutor, Worker
from mini_vllm.speculative.engine import SpeculativeEngine


def configs():
    return (
        EngineConfig(max_num_seqs=2, max_model_len=32, max_num_batched_tokens=8, max_speculative_tokens=2),
        TinyModelConfig(vocab_size=24, d_model=16, n_heads=4, n_kv_heads=2, n_layers=1, d_ff=32, max_position_embeddings=64, seed=7),
        CacheConfig(block_size=2, num_blocks=32),
    )


def test_v08_composition_exposes_local_executor_worker_boundary() -> None:
    e, m, c = configs()
    engine = create_engine("v0.8", engine_config=e, model_config=m, cache_config=c)
    assert isinstance(engine, SpeculativeEngine)
    assert isinstance(engine.executor, LocalExecutor)
    assert isinstance(engine.executor.worker, Worker)
    assert isinstance(engine.cache, TransactionalPrefixCachingPagedKVCache)
    assert engine.version == "v0.8"
    assert not hasattr(engine.executor.worker, "scheduler")
    assert not hasattr(engine.executor.worker, "waiting")
    assert not hasattr(engine.executor.worker, "running")


def test_v07_and_v08_have_identical_deterministic_output() -> None:
    e, m, c = configs()
    v07 = create_engine("v0.7", engine_config=e, model_config=m, cache_config=c)
    v08 = create_engine("v0.8", engine_config=e, model_config=m, cache_config=c)
    r07 = Request("same", [1, 2, 3, 4], max_new_tokens=6)
    r08 = Request("same", [1, 2, 3, 4], max_new_tokens=6)
    v07.add_request(r07)
    v08.add_request(r08)
    v07.run()
    v08.run()
    assert r08.output_token_ids == r07.output_token_ids
    assert r08.finish_reason == r07.finish_reason


def test_v08_normal_and_speculative_forward_both_cross_executor() -> None:
    e, m, c = configs()
    engine = create_engine("v0.8", engine_config=e, model_config=m, cache_config=c)
    request = Request("A", [1, 2], max_new_tokens=5)
    engine.add_request(request)
    engine.run()
    assert engine.executor.execute_calls > 0
    assert engine.executor.speculative_execute_calls > 0

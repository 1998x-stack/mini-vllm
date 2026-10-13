from mini_vllm import AsyncEngine, CacheConfig, EngineConfig, TinyModelConfig, create_engine
from mini_vllm.distributed import DistributedExecutor


def test_v010_wraps_v09_distributed_core_without_moving_policy() -> None:
    engine = create_engine(
        "v0.10",
        engine_config=EngineConfig(max_num_seqs=2, max_model_len=32, max_num_batched_tokens=8),
        model_config=TinyModelConfig(vocab_size=24, d_model=16, n_heads=4, n_kv_heads=2, n_layers=1, d_ff=32, max_position_embeddings=64),
        cache_config=CacheConfig(block_size=2, num_blocks=32),
        world_size=2,
        max_queue_size=3,
    )
    assert isinstance(engine, AsyncEngine)
    assert isinstance(engine.core.executor, DistributedExecutor)
    assert engine.core.executor.world_size == 2
    assert engine.version == "v0.10"
    assert engine.core.version == "v0.9"
    assert not hasattr(engine, "scheduler")
    assert not hasattr(engine, "waiting")
    assert not hasattr(engine, "running")

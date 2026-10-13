import asyncio

from mini_vllm import (
    CacheConfig,
    CompileConfig,
    EngineConfig,
    Request,
    TinyModelConfig,
    TokenEvent,
    create_engine,
)
from mini_vllm.model.compile_runner import CompileAwareVocabParallelTinyRunner


def configs():
    return (
        EngineConfig(max_num_seqs=2, max_model_len=32, max_num_batched_tokens=8, max_speculative_tokens=2),
        TinyModelConfig(vocab_size=24, d_model=16, n_heads=4, n_kv_heads=2, n_layers=1, d_ff=32, max_position_embeddings=64, seed=17),
        CacheConfig(block_size=2, num_blocks=32),
    )


async def collect(version: str, *, world_size: int = 2):
    e, m, c = configs()
    engine = create_engine(
        version,
        engine_config=e,
        model_config=m,
        cache_config=c,
        world_size=world_size,
        max_queue_size=4,
        max_speculative_tokens=2,
        compile_config=CompileConfig(backend="eager") if version == "v0.11" else None,
    )
    handle = engine.submit(Request("R", [1, 2, 3, 4], max_new_tokens=5))
    tokens = []
    async for event in handle:
        if isinstance(event, TokenEvent):
            tokens.append(event.token_id)
    await engine.wait_idle()
    return engine, tokens


def test_v011_async_output_matches_v010_reference():
    async def scenario():
        _, reference = await collect("v0.10")
        _, compiled = await collect("v0.11")
        assert compiled == reference
    asyncio.run(scenario())


def test_v011_world_size_one_and_two_match():
    async def scenario():
        _, one = await collect("v0.11", world_size=1)
        _, two = await collect("v0.11", world_size=2)
        assert one == two
    asyncio.run(scenario())


def test_v011_rank_local_compile_caches_are_observable_and_reused():
    async def scenario():
        engine, _ = await collect("v0.11")
        group = engine.core.executor.worker_group
        caches = []
        for ranked in group.ranked_workers:
            runner = ranked.worker.model_runner
            assert isinstance(runner, CompileAwareVocabParallelTinyRunner)
            caches.append(runner.compile_cache)
        assert len({id(cache) for cache in caches}) == group.world_size
        for cache in caches:
            assert cache.stats.compile_attempts >= 1
            assert cache.stats.compiled_variants >= 1
            assert cache.stats.cache_hits >= 1
            assert cache.events
    asyncio.run(scenario())

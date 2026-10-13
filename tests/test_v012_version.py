import asyncio

from mini_vllm import (
    CacheConfig,
    CompileConfig,
    EngineConfig,
    PagedAttentionConfig,
    Request,
    TinyModelConfig,
    TokenEvent,
    VectorizedPagedAttention,
    create_engine,
)
from mini_vllm.model.compile_runner import CompileAwareVocabParallelTinyRunner


def configs():
    return (
        EngineConfig(
            max_num_seqs=2,
            max_model_len=48,
            max_num_batched_tokens=8,
            max_speculative_tokens=2,
        ),
        TinyModelConfig(
            vocab_size=24,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            n_layers=1,
            d_ff=32,
            max_position_embeddings=64,
            seed=31,
        ),
        CacheConfig(block_size=2, num_blocks=32),
    )


async def collect(
    version: str,
    *,
    world_size: int = 2,
    backend: str | None = None,
):
    e, m, c = configs()
    kwargs = dict(
        engine_config=e,
        model_config=m,
        cache_config=c,
        world_size=world_size,
        max_queue_size=4,
        max_speculative_tokens=2,
    )
    if version in {"v0.11", "v0.12"}:
        kwargs["compile_config"] = CompileConfig(backend="eager")
    if version == "v0.12":
        kwargs["attention_config"] = PagedAttentionConfig(
            backend=backend or "vectorized"
        )
    engine = create_engine(version, **kwargs)
    handle = engine.submit(Request("R", [1, 2, 3, 4, 5], max_new_tokens=6))
    tokens = []
    async for event in handle:
        if isinstance(event, TokenEvent):
            tokens.append(event.token_id)
    await engine.wait_idle()
    return engine, tokens


def test_v012_vectorized_output_matches_v011_reference():
    async def scenario():
        _, reference = await collect("v0.11")
        _, vectorized = await collect("v0.12")
        assert vectorized == reference

    asyncio.run(scenario())


def test_v012_reference_and_vectorized_backends_match():
    async def scenario():
        _, reference = await collect("v0.12", backend="reference")
        _, vectorized = await collect("v0.12", backend="vectorized")
        assert vectorized == reference

    asyncio.run(scenario())


def test_v012_world_size_one_and_two_match():
    async def scenario():
        _, one = await collect("v0.12", world_size=1)
        _, two = await collect("v0.12", world_size=2)
        assert one == two

    asyncio.run(scenario())


def test_v012_rank_local_attention_and_compile_state_are_observable():
    async def scenario():
        engine, _ = await collect("v0.12")
        group = engine.core.executor.worker_group
        backends = []
        for ranked in group.ranked_workers:
            runner = ranked.worker.model_runner
            assert isinstance(runner, CompileAwareVocabParallelTinyRunner)
            backend = runner.model.attention_backend
            assert isinstance(backend, VectorizedPagedAttention)
            assert backend.stats.vectorized_calls > 0
            assert backend.stats.fallbacks == 0
            assert backend.events
            assert runner.compile_cache.stats.compile_attempts >= 1
            backends.append(backend)
        assert len({id(backend) for backend in backends}) == group.world_size

    asyncio.run(scenario())


def test_v012_shared_prefix_warm_request_uses_vectorized_attention_safely():
    async def scenario():
        e, m, c = configs()
        engine = create_engine(
            "v0.12",
            engine_config=e,
            model_config=m,
            cache_config=c,
            world_size=2,
            max_queue_size=4,
            max_speculative_tokens=1,
            compile_config=CompileConfig(backend="eager"),
            attention_config=PagedAttentionConfig(backend="vectorized"),
        )

        async def run(request_id: str):
            handle = engine.submit(
                Request(request_id, [1, 2, 3, 4, 5], max_new_tokens=3)
            )
            tokens = []
            async for event in handle:
                if isinstance(event, TokenEvent):
                    tokens.append(event.token_id)
            return tokens

        cold = await run("cold")
        warm = await run("warm")
        await engine.wait_idle()
        assert warm == cold
        stats = engine.core.executor.cache_stats()
        assert stats["prefix_hit_tokens"] > 0
        for ranked in engine.core.executor.worker_group.ranked_workers:
            backend = ranked.worker.model_runner.model.attention_backend
            assert backend.stats.vectorized_calls > 0
            assert backend.stats.fallbacks == 0

    asyncio.run(scenario())

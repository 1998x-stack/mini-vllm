import asyncio

import pytest

from mini_vllm import (
    CacheConfig,
    CompileConfig,
    EngineConfig,
    PagedAttentionConfig,
    Request,
    RuntimeConfig,
    TinyModelConfig,
    TokenEvent,
    create_engine,
)
from mini_vllm.versions.v1_0 import IntegratedAsyncEngine


def configs():
    return (
        EngineConfig(
            max_num_seqs=2,
            max_model_len=48,
            max_num_batched_tokens=4,
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
            seed=41,
        ),
        CacheConfig(block_size=2, num_blocks=32),
    )


async def collect(version: str):
    e, m, c = configs()
    engine = create_engine(
        version,
        engine_config=e,
        model_config=m,
        cache_config=c,
        world_size=2,
        max_queue_size=4,
        max_speculative_tokens=2,
        compile_config=CompileConfig(backend="eager"),
        attention_config=PagedAttentionConfig(backend="vectorized"),
        fairness_bypass_limit=2,
    )
    handle = engine.submit(Request("R", [1, 2, 3, 4, 5], max_new_tokens=5))
    tokens = []
    async for event in handle:
        if isinstance(event, TokenEvent):
            tokens.append(event.token_id)
    await engine.wait_idle()
    return engine, tokens


def test_default_builder_is_integrated_v1_runtime() -> None:
    engine = create_engine()
    assert isinstance(engine, IntegratedAsyncEngine)
    assert engine.version == "v1.0"
    assert engine.core.version == "v1.0"


def test_v1_deterministic_output_matches_v013() -> None:
    async def scenario():
        _, previous = await collect("v0.13")
        _, integrated = await collect("v1.0")
        assert integrated == previous

    asyncio.run(scenario())


def test_runtime_config_facade_constructs_integrated_runtime() -> None:
    e, m, c = configs()
    config = RuntimeConfig(
        engine=e,
        model=m,
        cache=c,
        compile=CompileConfig(backend="eager"),
        attention=PagedAttentionConfig(backend="vectorized"),
        world_size=1,
        max_queue_size=3,
        fairness_bypass_limit=3,
    )
    engine = create_engine(runtime_config=config)
    assert engine.max_queue_size == 3
    snapshot = engine.snapshot()
    assert snapshot.core.execution.world_size == 1
    assert snapshot.core.scheduler.name == "PrefixAwareBudgetedScheduler"

    with pytest.raises(ValueError, match="runtime_config cannot be combined"):
        create_engine(runtime_config=config, world_size=2)


def test_snapshot_is_side_effect_free_and_exposes_execution_paths() -> None:
    async def scenario():
        engine, tokens = await collect("v1.0")
        first = engine.snapshot()
        second = engine.snapshot()
        assert first == second
        assert first.pending_request_ids == ()
        assert not first.driver_running
        assert first.core.last_step is not None
        assert first.core.last_step.committed_token_ids
        assert first.core.requests[0].output_tokens == tuple(tokens)
        assert first.core.cache_stats["cached_blocks"] >= 1
        assert first.core.execution.world_size == 2
        assert len(first.core.execution.ranks) == 2
        for rank in first.core.execution.ranks:
            assert rank.attention_backend == "VectorizedPagedAttention"
            assert rank.attention_vectorized_calls > 0
            assert rank.compile_attempts >= 1
        assert first.core.scheduler.admissions >= 1
        assert first.core.scheduler.reconciliations >= 1
        assert first.core.recent_steps
        assert any(step.speculation for step in first.core.recent_steps)

    asyncio.run(scenario())


def test_integrated_snapshot_explains_warm_prefix_admission() -> None:
    async def scenario():
        e, m, c = configs()
        engine = create_engine(
            "v1.0",
            engine_config=e,
            model_config=m,
            cache_config=c,
            world_size=1,
            max_queue_size=4,
            max_speculative_tokens=1,
            compile_config=CompileConfig(backend="eager"),
            attention_config=PagedAttentionConfig(backend="vectorized"),
        )

        async def run(request_id: str, prompt: list[int]):
            handle = engine.submit(Request(request_id, prompt, max_new_tokens=1))
            await handle.wait_terminal()
            await engine.wait_idle()

        shared = [1, 2, 3, 4, 5, 6, 7]
        await run("seed", shared)

        cold = Request("cold", [9, 10, 11, 12, 13, 14, 15], max_new_tokens=1)
        warm = Request("warm", shared, max_new_tokens=1)
        cold_handle = engine.submit(cold)
        warm_handle = engine.submit(warm)
        await asyncio.gather(
            cold_handle.wait_terminal(),
            warm_handle.wait_terminal(),
        )
        await engine.wait_idle()

        snapshot = engine.snapshot()
        admissions = [
            event for event in snapshot.core.scheduler.recent_events
            if event.kind == "admit" and event.request_id in {"cold", "warm"}
        ]
        assert admissions
        warm_event = next(event for event in admissions if event.request_id == "warm")
        cold_event = next(event for event in admissions if event.request_id == "cold")
        assert warm_event.reusable_tokens > cold_event.reusable_tokens
        assert warm_event.expected_new_prefill_tokens < cold_event.expected_new_prefill_tokens
        assert snapshot.core.scheduler.warm_preferred_admissions >= 1

    asyncio.run(scenario())


def test_historical_builders_remain_available() -> None:
    assert create_engine("v0.1").version == "v0.1"
    assert create_engine("v0.6").version == "v0.6"
    assert create_engine("v0.13").version == "v0.13"

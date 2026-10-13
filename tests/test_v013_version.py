import asyncio

from mini_vllm import (
    CacheConfig,
    CompileConfig,
    EngineConfig,
    PagedAttentionConfig,
    PrefixAwareBudgetedScheduler,
    Request,
    TinyModelConfig,
    TokenEvent,
    create_engine,
)


def configs(*, max_num_seqs: int = 2):
    return (
        EngineConfig(
            max_num_seqs=max_num_seqs,
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


async def collect(version: str, *, world_size: int = 2):
    e, m, c = configs()
    kwargs = dict(
        engine_config=e,
        model_config=m,
        cache_config=c,
        world_size=world_size,
        max_queue_size=4,
        max_speculative_tokens=2,
        compile_config=CompileConfig(backend="eager"),
        attention_config=PagedAttentionConfig(backend="vectorized"),
    )
    engine = create_engine(version, **kwargs)
    handle = engine.submit(Request("R", [1, 2, 3, 4, 5], max_new_tokens=6))
    tokens = []
    async for event in handle:
        if isinstance(event, TokenEvent):
            tokens.append(event.token_id)
    await engine.wait_idle()
    return engine, tokens


def test_v013_async_output_matches_v012():
    async def scenario():
        _, reference = await collect("v0.12")
        _, prefix_aware = await collect("v0.13")
        assert prefix_aware == reference

    asyncio.run(scenario())


def test_v013_world_size_one_and_two_match():
    async def scenario():
        _, one = await collect("v0.13", world_size=1)
        _, two = await collect("v0.13", world_size=2)
        assert one == two

    asyncio.run(scenario())


def test_v013_waiting_request_does_not_own_kv_before_admission():
    e, m, c = configs(max_num_seqs=1)
    engine = create_engine(
        "v0.13",
        engine_config=e,
        model_config=m,
        cache_config=c,
        world_size=2,
        max_queue_size=4,
        max_speculative_tokens=1,
        compile_config=CompileConfig(backend="eager"),
    )
    core = engine.core
    core.add_request(Request("active", [7, 8, 9, 10, 11], max_new_tokens=3))
    core.step()
    core.add_request(Request("waiting", [1, 2, 3, 4, 5], max_new_tokens=1))

    for ranked in core.executor.worker_group.ranked_workers:
        assert ranked.worker.cache.block_table("waiting") == ()
    assert core.get_request("waiting").num_computed_tokens == 0


def test_v013_warm_prefix_changes_admission_order() -> None:
    e, m, c = configs(max_num_seqs=1)
    engine = create_engine(
        "v0.13",
        engine_config=e,
        model_config=m,
        cache_config=c,
        world_size=1,
        max_queue_size=4,
        max_speculative_tokens=1,
        compile_config=CompileConfig(backend="eager"),
    )
    core = engine.core
    assert isinstance(core.scheduler, PrefixAwareBudgetedScheduler)

    prompt = [1, 2, 3, 4, 5]
    core.add_request(Request("seed", prompt, max_new_tokens=1))
    core.run()

    core.add_request(Request("cold", [9, 10, 11, 12, 13], max_new_tokens=1))
    core.add_request(Request("warm", prompt, max_new_tokens=1))
    step = core.step()

    assert step.admitted_request_ids == ("warm",)
    admission = [event for event in core.scheduler.events if event.kind == "admit"][-1]
    assert admission.request_id == "warm"
    assert admission.reusable_tokens == 4
    assert admission.expected_new_prefill_tokens == 1
    assert core.scheduler.stats.warm_preferred_admissions >= 1

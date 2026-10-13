import asyncio

import pytest

from mini_vllm import (
    AsyncEngine,
    CacheConfig,
    CancelledEvent,
    EngineConfig,
    ErrorEvent,
    FinishedEvent,
    Request,
    TinyModelConfig,
    TokenEvent,
    create_engine,
)


def configs():
    return (
        EngineConfig(max_num_seqs=2, max_model_len=48, max_num_batched_tokens=8, max_speculative_tokens=1),
        TinyModelConfig(vocab_size=24, d_model=16, n_heads=4, n_kv_heads=2, n_layers=1, d_ff=32, max_position_embeddings=64, seed=7),
        CacheConfig(block_size=2, num_blocks=32),
    )


def collect_sync_output(request_id: str, prompt: list[int], max_new_tokens: int) -> list[int]:
    e, m, c = configs()
    engine = create_engine("v0.9", engine_config=e, model_config=m, cache_config=c, world_size=2, max_speculative_tokens=1)
    request = Request(request_id, prompt, max_new_tokens=max_new_tokens)
    engine.add_request(request)
    engine.run()
    return request.output_token_ids


def test_v010_async_stream_matches_v09_and_has_one_terminal_event() -> None:
    expected = collect_sync_output("sync", [1, 2, 3], 6)

    async def scenario() -> None:
        e, m, c = configs()
        engine = create_engine("v0.10", engine_config=e, model_config=m, cache_config=c, world_size=2, max_speculative_tokens=1, max_queue_size=2)
        assert isinstance(engine, AsyncEngine)
        handle = engine.submit(Request("async", [1, 2, 3], max_new_tokens=6))
        events = [event async for event in handle]
        await engine.wait_idle()

        tokens = [event.token_id for event in events if isinstance(event, TokenEvent)]
        terminals = [event for event in events if isinstance(event, (FinishedEvent, CancelledEvent, ErrorEvent))]
        assert tokens == expected
        assert len(terminals) == 1
        assert isinstance(terminals[0], FinishedEvent)
        assert terminals[0].output_token_ids == tuple(expected)
        assert not engine.driver_running

    asyncio.run(scenario())


def test_duplicate_request_id_is_rejected_before_second_submission() -> None:
    async def scenario() -> None:
        e, m, c = configs()
        engine = create_engine("v0.10", engine_config=e, model_config=m, cache_config=c)
        first = engine.submit(Request("dup", [1], max_new_tokens=2))
        with pytest.raises(ValueError, match="duplicate request_id"):
            engine.submit(Request("dup", [2], max_new_tokens=2))
        first.cancel()
        events = [event async for event in first]
        assert len(events) == 1 and isinstance(events[0], CancelledEvent)
        await engine.wait_idle()

    asyncio.run(scenario())


def test_cancel_before_admission_allocates_no_kv() -> None:
    async def scenario() -> None:
        e, m, c = configs()
        engine = create_engine("v0.10", engine_config=e, model_config=m, cache_config=c, max_queue_size=2)
        handle = engine.submit(Request("queued", [1, 2, 3], max_new_tokens=5))
        assert handle.cancel()
        events = [event async for event in handle]
        await engine.wait_idle()
        assert len(events) == 1 and isinstance(events[0], CancelledEvent)
        with pytest.raises(KeyError):
            engine.core.get_request("queued")
        assert engine.core.executor.cache_stats()["active_block_refs"] == 0

    asyncio.run(scenario())


def test_active_cancel_stops_scheduler_and_releases_kv() -> None:
    async def scenario() -> None:
        e, m, c = configs()
        engine = create_engine("v0.10", engine_config=e, model_config=m, cache_config=c, max_queue_size=2, max_speculative_tokens=1)
        handle = engine.submit(Request("active", [1, 2], max_new_tokens=12))

        first = await handle.__anext__()
        assert isinstance(first, TokenEvent)
        assert handle.cancel()
        rest = [event async for event in handle]
        await engine.wait_idle()

        terminals = [event for event in rest if isinstance(event, (FinishedEvent, CancelledEvent, ErrorEvent))]
        assert len(terminals) == 1 and isinstance(terminals[0], CancelledEvent)
        request = engine.core.get_request("active")
        assert request.finish_reason == "cancelled"
        assert all(r.request_id != "active" for r in engine.core.scheduler.running)
        assert all(r.request_id != "active" for r in engine.core.scheduler.waiting)
        assert engine.core.executor.cache_stats()["active_block_refs"] == 0

    asyncio.run(scenario())


def test_bounded_queue_applies_backpressure_without_dropping_tokens() -> None:
    expected = collect_sync_output("sync-backpressure", [3, 4], 5)

    async def scenario() -> None:
        e, m, c = configs()
        engine = create_engine("v0.10", engine_config=e, model_config=m, cache_config=c, max_queue_size=1, max_speculative_tokens=1)
        handle = engine.submit(Request("slow", [3, 4], max_new_tokens=5))

        for _ in range(12):
            await asyncio.sleep(0)
        step_before = engine.core._step
        for _ in range(12):
            await asyncio.sleep(0)
        assert engine.core._step == step_before
        assert handle.queue_size == 1

        events = [event async for event in handle]
        await engine.wait_idle()
        tokens = [event.token_id for event in events if isinstance(event, TokenEvent)]
        assert tokens == expected
        assert isinstance(events[-1], FinishedEvent)

    asyncio.run(scenario())


def test_core_failure_becomes_error_terminal_and_releases_resources() -> None:
    async def scenario() -> None:
        e, m, c = configs()
        engine = create_engine("v0.10", engine_config=e, model_config=m, cache_config=c, max_queue_size=2)

        def boom(_request):
            raise RuntimeError("synthetic executor failure")

        engine.core.executor.execute = boom
        handle = engine.submit(Request("fail", [1, 2], max_new_tokens=4))
        events = [event async for event in handle]
        await engine.wait_idle()

        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)
        assert events[0].error_type == "RuntimeError"
        assert "synthetic executor failure" in events[0].message
        assert engine.core.get_request("fail").finish_reason == "error"
        assert engine.core.executor.cache_stats()["active_block_refs"] == 0

    asyncio.run(scenario())

import asyncio
import multiprocessing as mp
import os
import pickle

import pytest

from mini_vllm import (
    CacheConfig,
    CompileConfig,
    EngineConfig,
    MultiprocessConfig,
    PagedAttentionConfig,
    Request,
    RuntimeConfig,
    TinyModelConfig,
    TokenEvent,
    WorkerProcessDied,
    WorkerRemoteError,
    create_engine,
)
from mini_vllm.execution import ExecutionItem, ExecutionRequest
from mini_vllm.multiprocess.protocol import TransportCommand
from mini_vllm.versions.v1_1 import MultiprocessAsyncEngine


FAST_START_METHOD = "fork" if "fork" in mp.get_all_start_methods() else "spawn"


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


def build(version="v1.1", *, world_size=1, start_method=FAST_START_METHOD):
    e, m, c = configs()
    return create_engine(
        version,
        engine_config=e,
        model_config=m,
        cache_config=c,
        world_size=world_size,
        max_queue_size=4,
        max_speculative_tokens=2,
        compile_config=CompileConfig(backend="eager"),
        attention_config=PagedAttentionConfig(backend="vectorized"),
        fairness_bypass_limit=2,
        **(
            {"multiprocess_config": MultiprocessConfig(start_method=start_method)}
            if version == "v1.1"
            else {}
        ),
    )


async def collect(engine, request_id="R", prompt=None, max_new_tokens=5):
    handle = engine.submit(
        Request(
            request_id,
            prompt or [1, 2, 3, 4, 5],
            max_new_tokens=max_new_tokens,
        )
    )
    tokens = []
    async for event in handle:
        if isinstance(event, TokenEvent):
            tokens.append(event.token_id)
    await engine.wait_idle()
    return tokens


def test_transport_protocol_is_pickle_serializable() -> None:
    payload = TransportCommand(
        correlation_id=7,
        operation="execute",
        payload=ExecutionRequest((ExecutionItem("R", (1, 2), 0),)),
    )
    assert pickle.loads(pickle.dumps(payload)) == payload


def test_default_builder_starts_real_worker_processes_and_snapshot_is_read_only() -> None:
    engine = create_engine(
        "v1.1",
        runtime_config=RuntimeConfig(
            world_size=1,
            multiprocess=MultiprocessConfig(start_method=FAST_START_METHOD),
        ),
    )
    try:
        assert isinstance(engine, MultiprocessAsyncEngine)
        assert engine.version == "v1.1"
        pid = engine.core.executor.worker_pids[0]
        assert pid > 0 and pid != os.getpid()
        assert engine.core.executor.worker_alive == (True,)

        first = engine.snapshot()
        second = engine.snapshot()
        assert first == second
        rank = first.core.execution.ranks[0]
        assert first.core.execution.boundary == "multiprocess-ipc"
        assert first.core.execution.healthy
        assert rank.boundary == "multiprocess-ipc"
        assert rank.process_id == pid
        assert rank.worker_state == "ready"
    finally:
        engine.close()
    assert engine.core.executor.worker_alive == (False,)


def test_v11_world_size_one_and_two_match_v10_deterministic_output() -> None:
    async def scenario():
        for world_size in (1, 2):
            reference = build("v1.0", world_size=world_size)
            expected = await collect(reference)

            remote = build("v1.1", world_size=world_size)
            try:
                actual = await collect(remote)
                snapshot = remote.snapshot()
                assert actual == expected
                assert snapshot.core.execution.world_size == world_size
                assert all(
                    rank.process_id != os.getpid()
                    for rank in snapshot.core.execution.ranks
                )
                assert all(
                    rank.transport_commands_sent > 0
                    and rank.transport_commands_sent == rank.transport_commands_completed
                    for rank in snapshot.core.execution.ranks
                )
                assert any(step.speculation for step in snapshot.core.recent_steps)
            finally:
                await remote.aclose()

    asyncio.run(scenario())


def test_prefix_reuse_and_cache_lifecycle_cross_process_boundary() -> None:
    async def scenario():
        engine = build("v1.1", world_size=1)
        try:
            shared = [1, 2, 3, 4, 5, 6, 7]
            await collect(engine, "seed", shared, 1)
            await collect(engine, "warm", shared, 1)
            snapshot = engine.snapshot()
            warm_events = [
                event
                for event in snapshot.core.scheduler.recent_events
                if event.request_id == "warm" and event.kind == "reconcile"
            ]
            assert warm_events
            assert warm_events[-1].actual_reuse_tokens > 0
            assert snapshot.core.cache_stats["prefix_hit_tokens"] > 0
            assert snapshot.core.execution.ranks[0].attention_vectorized_calls > 0
        finally:
            await engine.aclose()

    asyncio.run(scenario())


def test_child_reported_error_is_structured_and_process_survives() -> None:
    engine = build("v1.1", world_size=1)
    executor = engine.core.executor
    try:
        with pytest.raises(WorkerRemoteError, match="worker rank 0 reported"):
            executor.checkpoint("missing", 0)
        assert executor.worker_alive == (True,)
        assert executor.healthy
        snapshot = executor.transport_snapshot()[0]
        assert snapshot.remote_errors >= 1
        assert snapshot.transport_failures == 0
    finally:
        engine.close()


def test_unexpected_worker_death_is_detected_and_shutdown_reaps_process() -> None:
    engine = build("v1.1", world_size=1)
    executor = engine.core.executor
    client = executor._clients[0]
    pid = executor.worker_pids[0]
    client.process.terminate()
    client.process.join(2.0)
    try:
        with pytest.raises(WorkerProcessDied, match="worker rank 0 exited unexpectedly"):
            executor.cache_stats()
        assert not executor.healthy
    finally:
        engine.close()
    assert not executor.worker_alive[0]
    assert pid != os.getpid()

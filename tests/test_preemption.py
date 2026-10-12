import torch
import pytest

from mini_vllm import CacheConfig, EngineConfig, Request, RequestStatus, TinyModelConfig, create_engine
from mini_vllm.scheduler.preemptive import PreemptiveBudgetedScheduler


def model_config(seed: int = 17) -> TinyModelConfig:
    return TinyModelConfig(
        vocab_size=32,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        d_ff=32,
        max_position_embeddings=64,
        seed=seed,
    )


def engine_config() -> EngineConfig:
    return EngineConfig(
        max_num_seqs=2,
        max_model_len=20,
        max_num_batched_tokens=4,
    )


def test_victim_selection_is_latest_admitted_and_resume_waits_for_finish() -> None:
    scheduler = PreemptiveBudgetedScheduler(max_num_seqs=3, max_num_batched_tokens=8)
    a = Request("A", [1], 1)
    b = Request("B", [2], 1)
    c = Request("C", [3], 1)
    for request in (a, b, c):
        scheduler.add_request(request)

    scheduler.schedule()
    victim = scheduler.select_preemption_victim(a)
    assert victim is c

    c.mark_preempted()
    scheduler.on_request_preempted(c)
    assert c.status is RequestStatus.PREEMPTED

    # The anti-thrashing guard keeps C out until one survivor finishes.
    decision = scheduler.schedule()
    assert "C" not in decision.admitted_request_ids

    b.mark_finished("test")
    scheduler.on_request_finished(b)
    decision = scheduler.schedule()
    assert "C" in decision.admitted_request_ids
    assert c.status is RequestStatus.RUNNING


def _run_pair(num_blocks: int):
    engine = create_engine(
        "v0.6",
        engine_config=engine_config(),
        model_config=model_config(),
        cache_config=CacheConfig(block_size=2, num_blocks=num_blocks),
    )
    engine.add_request(Request("A", [1, 2, 3], max_new_tokens=2))
    engine.add_request(Request("B", [4, 5, 6], max_new_tokens=1))
    trace = engine.run(max_steps=32)
    return engine, trace


def test_pressure_preempts_replays_and_matches_nonpreempted_outputs() -> None:
    pressured, trace = _run_pair(num_blocks=3)
    roomy, _ = _run_pair(num_blocks=8)

    assert pressured.preemption_events == 1
    assert any(step.preempted_request_ids == ("A",) for step in trace)
    assert pressured.get_request("A").preemption_count == 1
    assert pressured.get_request("A").finished
    assert pressured.get_request("B").finished

    # Logical output history survives KV invalidation/recomputation.
    assert pressured.get_request("A").output_token_ids == roomy.get_request("A").output_token_ids
    assert pressured.get_request("B").output_token_ids == roomy.get_request("B").output_token_ids

    # Both active ownership and private state are gone after completion.
    assert pressured.cache.stats()["active_block_refs"] == 0


def test_preempting_one_owner_does_not_corrupt_shared_prefix() -> None:
    engine = create_engine(
        "v0.6",
        engine_config=engine_config(),
        model_config=model_config(seed=23),
        cache_config=CacheConfig(block_size=2, num_blocks=3),
    )

    # Seed an immutable [1, 2] prefix block.
    engine.add_request(Request("seed", [1, 2, 9], max_new_tokens=1))
    engine.run()

    engine.add_request(Request("A", [1, 2, 3], max_new_tokens=4))
    engine.add_request(Request("B", [1, 2, 4], max_new_tokens=3))
    shared_block = engine.cache.block_table("A")[0]
    assert engine.cache.block_table("B")[0] == shared_block
    assert engine.cache.block_refcount(shared_block) == 2
    snapshot = engine.cache.keys[:, shared_block].clone()

    # Two steps fill the private tail block.  The third decode needs another
    # physical block; B (latest admitted) is preempted so A can grow.
    engine.step()
    engine.step()
    pressure_step = engine.step()

    assert pressure_step.preempted_request_ids == ("B",)
    assert engine.get_request("B").status is RequestStatus.PREEMPTED
    assert engine.cache.block_table("B") == ()
    assert engine.get_request("B").num_computed_tokens == 0

    # A still references exactly the same immutable prefix block.
    assert engine.get_request("A").status is RequestStatus.RUNNING
    assert engine.cache.block_refcount(shared_block) == 1
    assert torch.equal(engine.cache.keys[:, shared_block], snapshot)

    # Once A eventually finishes, the anti-thrashing guard allows B to resume.
    engine.run(max_steps=32)
    assert engine.get_request("B").finished
    assert engine.get_request("B").preemption_count == 1


def test_impossible_single_request_allocation_fails_explicitly() -> None:
    engine = create_engine(
        "v0.6",
        engine_config=EngineConfig(
            max_num_seqs=1,
            max_model_len=16,
            max_num_batched_tokens=8,
        ),
        model_config=model_config(),
        cache_config=CacheConfig(block_size=2, num_blocks=1),
    )
    engine.add_request(Request("too-large", [1, 2, 3], max_new_tokens=1))

    with pytest.raises(RuntimeError, match="no eligible preemption victim"):
        engine.step()

import pytest

from mini_vllm.cache.noop import NoOpKVCache
from mini_vllm.config import EngineConfig
from mini_vllm.core.engine import Engine
from mini_vllm.core.request import Request
from mini_vllm.core.sampler import Sampler
from mini_vllm.model.runner import ScriptedModelRunner
from mini_vllm.scheduler.base import make_work_item
from mini_vllm.scheduler.budgeted import BudgetedContinuousScheduler


def make_engine(*, budget: int = 3, max_num_seqs: int = 3) -> Engine:
    return Engine(
        config=EngineConfig(
            max_num_seqs=max_num_seqs,
            max_model_len=64,
            max_num_batched_tokens=budget,
        ),
        scheduler=BudgetedContinuousScheduler(max_num_seqs, budget),
        cache=NoOpKVCache(),
        model_runner=ScriptedModelRunner(vocab_size=64),
        sampler=Sampler(),
        version="test-v0.4",
    )


def test_budget_is_never_exceeded() -> None:
    engine = make_engine(budget=4)
    engine.add_request(Request("A", list(range(1, 10)), 2))
    engine.add_request(Request("B", [20, 21, 22], 3))

    trace = engine.run()
    assert trace
    assert all(step.num_scheduled_tokens <= 4 for step in trace)
    assert all(sum(w.num_tokens for w in step.work) == step.num_scheduled_tokens for step in trace)


def test_long_prompt_prefills_across_multiple_steps_without_early_sampling() -> None:
    engine = make_engine(budget=2, max_num_seqs=1)
    engine.add_request(Request("A", [1, 2, 3, 4, 5], 2))

    first = engine.step()
    assert first.num_scheduled_tokens == 2
    assert first.sampled_token_ids == {}
    assert engine.get_request("A").num_computed_tokens == 2

    second = engine.step()
    assert second.num_scheduled_tokens == 2
    assert second.sampled_token_ids == {}
    assert engine.get_request("A").num_computed_tokens == 4

    third = engine.step()
    assert [(w.phase, w.num_tokens) for w in third.work] == [("prefill", 1)]
    assert "A" in third.sampled_token_ids
    assert engine.get_request("A").num_computed_tokens == 5
    assert len(engine.get_request("A").output_token_ids) == 1


def test_decode_runs_before_long_prefill_and_shares_budget() -> None:
    engine = make_engine(budget=3, max_num_seqs=2)
    engine.add_request(Request("decode", [1], 3))

    first = engine.step()
    assert "decode" in first.sampled_token_ids

    engine.add_request(Request("prefill", [10, 11, 12, 13, 14, 15], 2))
    mixed = engine.step()

    assert [(w.request_id, w.phase, w.num_tokens) for w in mixed.work] == [
        ("decode", "decode", 1),
        ("prefill", "prefill", 2),
    ]
    assert mixed.num_decode_tokens == 1
    assert mixed.num_prefill_tokens == 2
    assert mixed.num_scheduled_tokens == 3
    assert "decode" in mixed.sampled_token_ids
    assert "prefill" not in mixed.sampled_token_ids


def test_invalid_budget_and_chunk_size_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        EngineConfig(max_num_batched_tokens=0)
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        BudgetedContinuousScheduler(max_num_seqs=1, max_num_batched_tokens=0)

    request = Request("A", [1, 2], 1)
    with pytest.raises(ValueError, match="max_tokens"):
        make_work_item(request, max_tokens=0)

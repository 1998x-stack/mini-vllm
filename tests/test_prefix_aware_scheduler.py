from mini_vllm.cache.base import PrefixReuseHint
from mini_vllm.cache.noop import NoOpKVCache
from mini_vllm.cache.prefix import PrefixCachingPagedKVCache
from mini_vllm.config import EngineConfig
from mini_vllm.core.engine import Engine
from mini_vllm.core.request import Request
from mini_vllm.core.sampler import Sampler
from mini_vllm.model.runner import ScriptedModelRunner
from mini_vllm.scheduler.prefix_aware import PrefixAwareBudgetedScheduler


def make_prefix_cache() -> PrefixCachingPagedKVCache:
    return PrefixCachingPagedKVCache(
        num_layers=1,
        num_blocks=8,
        block_size=2,
        num_kv_heads=1,
        head_dim=2,
    )


def test_prefix_reuse_estimate_is_side_effect_free_and_matches_attach() -> None:
    cache = make_prefix_cache()
    prompt = [1, 2, 3, 4, 5]
    assert cache.prepare_request("seed", prompt) == 0
    assert cache.ensure_capacity("seed", len(prompt))
    cache.commit_work(
        "seed",
        start_pos=0,
        token_ids=tuple(prompt),
        prompt_length=len(prompt),
    )
    cache.free("seed")

    before = cache.stats().copy()
    hint = cache.estimate_prefix_reuse(prompt)
    after = cache.stats().copy()

    assert hint == PrefixReuseHint(reusable_tokens=4, reusable_full_blocks=2, prompt_tokens=5)
    assert before == after
    assert cache.block_table("probe") == ()
    assert cache.prepare_request("warm", prompt) == hint.reusable_tokens


def test_warm_request_is_preferred_under_tight_admission() -> None:
    scheduler = PrefixAwareBudgetedScheduler(
        max_num_seqs=1,
        max_num_batched_tokens=4,
        fairness_bypass_limit=2,
    )
    cold = Request("cold", list(range(1, 9)), 1)
    warm = Request("warm", list(range(11, 19)), 1)
    scheduler.add_request(cold)
    scheduler.add_request(warm)
    scheduler.update_prefix_reuse_hint(cold, PrefixReuseHint(0, 0, 8))
    scheduler.update_prefix_reuse_hint(warm, PrefixReuseHint(6, 3, 8))

    decision = scheduler.schedule()
    assert decision.admitted_request_ids == ["warm"]
    assert scheduler.events[0].expected_new_prefill_tokens == 2
    assert scheduler.stats.warm_preferred_admissions == 1


def test_equal_costs_preserve_stable_arrival_order() -> None:
    scheduler = PrefixAwareBudgetedScheduler(1, 4)
    first = Request("first", [1, 2, 3, 4], 1)
    second = Request("second", [5, 6, 7, 8], 1)
    scheduler.add_request(first)
    scheduler.add_request(second)
    scheduler.update_prefix_reuse_hint(first, PrefixReuseHint(0, 0, 4))
    scheduler.update_prefix_reuse_hint(second, PrefixReuseHint(0, 0, 4))

    assert scheduler.schedule().admitted_request_ids == ["first"]


def test_cold_request_gains_fairness_protection_after_bypasses() -> None:
    scheduler = PrefixAwareBudgetedScheduler(
        max_num_seqs=1,
        max_num_batched_tokens=4,
        fairness_bypass_limit=2,
    )
    cold = Request("cold", list(range(1, 9)), 1)
    scheduler.add_request(cold)
    scheduler.update_prefix_reuse_hint(cold, PrefixReuseHint(0, 0, 8))

    for index in range(2):
        warm = Request(f"warm-{index}", list(range(11, 19)), 1)
        scheduler.add_request(warm)
        scheduler.update_prefix_reuse_hint(warm, PrefixReuseHint(6, 3, 8))
        decision = scheduler.schedule()
        assert decision.admitted_request_ids == [warm.request_id]
        warm.mark_finished("length")
        scheduler.on_request_finished(warm)

    warm = Request("warm-2", list(range(11, 19)), 1)
    scheduler.add_request(warm)
    scheduler.update_prefix_reuse_hint(warm, PrefixReuseHint(6, 3, 8))
    decision = scheduler.schedule()
    assert decision.admitted_request_ids == ["cold"]
    assert scheduler.events[-1].fairness_protected
    assert scheduler.stats.fairness_overrides >= 1


class OptimisticNoOpCache(NoOpKVCache):
    def estimate_prefix_reuse(self, prompt_token_ids: list[int]) -> PrefixReuseHint:
        return PrefixReuseHint(6, 3, len(prompt_token_ids))

    def prepare_request(self, request_id: str, prompt_token_ids: list[int]) -> int:
        return 0


def test_stale_optimistic_hint_cannot_advance_computed_frontier() -> None:
    scheduler = PrefixAwareBudgetedScheduler(1, 4)
    engine = Engine(
        config=EngineConfig(max_num_seqs=1, max_model_len=32, max_num_batched_tokens=4),
        scheduler=scheduler,
        cache=OptimisticNoOpCache(),
        model_runner=ScriptedModelRunner(vocab_size=32),
        sampler=Sampler(),
        version="v0.13-test",
    )
    request = Request("R", list(range(1, 9)), 1)
    engine.add_request(request)

    step = engine.step()
    assert step.work[0].start_pos == 0
    assert step.work[0].num_tokens == 4
    assert request.num_computed_tokens == 4
    assert scheduler.stats.stale_hint_reconciliations == 1
    reconcile = [event for event in scheduler.events if event.kind == "reconcile"][-1]
    assert reconcile.reusable_tokens == 6
    assert reconcile.actual_reuse_tokens == 0

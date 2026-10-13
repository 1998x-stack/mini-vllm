from __future__ import annotations

from dataclasses import dataclass, field

from mini_vllm.core.work import EngineStepOutput


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    request_id: str
    status: str
    prompt_tokens: int
    output_tokens: tuple[int, ...]
    num_computed_tokens: int
    preemption_count: int
    finish_reason: str | None


@dataclass(frozen=True, slots=True)
class SchedulingObservation:
    kind: str
    request_id: str
    raw_prefill_tokens: int
    reusable_tokens: int
    expected_new_prefill_tokens: int
    bypass_count: int
    fairness_protected: bool
    actual_reuse_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    name: str
    waiting_request_ids: tuple[str, ...]
    running_request_ids: tuple[str, ...]
    hint_refreshes: int = 0
    admissions: int = 0
    warm_preferred_admissions: int = 0
    fairness_overrides: int = 0
    reconciliations: int = 0
    stale_hint_reconciliations: int = 0
    recent_events: tuple[SchedulingObservation, ...] = ()


@dataclass(frozen=True, slots=True)
class RankExecutionSnapshot:
    rank: int
    world_size: int
    vocab_range: tuple[int, int] | None
    attention_backend: str
    attention_total_calls: int
    attention_reference_calls: int
    attention_vectorized_calls: int
    attention_fallbacks: int
    compile_enabled: bool
    compile_backend: str
    compile_attempts: int
    compile_cache_hits: int
    compile_cache_misses: int
    compile_fallbacks: int
    eager_projection_calls: int


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot:
    executor: str
    world_size: int
    execute_calls: int
    speculative_execute_calls: int
    ranks: tuple[RankExecutionSnapshot, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    version: str
    next_step: int
    unfinished_request_ids: tuple[str, ...]
    requests: tuple[RequestSnapshot, ...]
    scheduler: SchedulerSnapshot
    execution: ExecutionSnapshot
    cache_stats: dict[str, int | float] = field(default_factory=dict)
    preemption_events: int = 0
    last_step: EngineStepOutput | None = None
    recent_steps: tuple[EngineStepOutput, ...] = ()


@dataclass(frozen=True, slots=True)
class AsyncRuntimeSnapshot:
    version: str
    driver_running: bool
    pending_request_ids: tuple[str, ...]
    cancellation_requested_ids: tuple[str, ...]
    core: RuntimeSnapshot

    @property
    def last_step(self) -> EngineStepOutput | None:
        return self.core.last_step

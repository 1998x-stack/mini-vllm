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
    boundary: str = "in-process"
    process_id: int | None = None
    worker_state: str = "local"
    transport_commands_sent: int = 0
    transport_commands_completed: int = 0
    remote_errors: int = 0
    transport_failures: int = 0
    artifact_identity: str | None = None
    tokenizer_identity: str | None = None
    artifact_architecture: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot:
    executor: str
    world_size: int
    execute_calls: int
    speculative_execute_calls: int
    ranks: tuple[RankExecutionSnapshot, ...] = ()
    boundary: str = "in-process"
    healthy: bool = True


@dataclass(frozen=True, slots=True)
class ModelArtifactSnapshot:
    identity: str
    architecture: str
    format_version: int
    tokenizer_identity: str
    tokenizer_compatibility_id: str
    vocab_size: int
    weight_dtype: str
    runtime_dtype: str
    device: str
    rank_identity_agreement: bool


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
    model_artifact: ModelArtifactSnapshot | None = None


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


@dataclass(frozen=True, slots=True)
class NetworkSnapshot:
    lifecycle_state: str
    host: str | None
    port: int | None
    health: bool
    ready: bool
    active_connections: int
    inflight_requests: int
    accepted_requests: int
    rejected_requests: int
    completed_requests: int
    cancelled_requests: int
    failed_requests: int
    disconnect_cancellations: int

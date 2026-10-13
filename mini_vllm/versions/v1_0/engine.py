from __future__ import annotations

from collections import deque

from mini_vllm.async_runtime import AsyncEngine
from mini_vllm.attention import PagedAttentionConfig, build_paged_attention_backend
from mini_vllm.cache.transactional import TransactionalPrefixCachingPagedKVCache
from mini_vllm.compile_runtime import CompileConfig
from mini_vllm.config import CacheConfig, EngineConfig, TinyModelConfig
from mini_vllm.core.request import Request
from mini_vllm.core.sampler import Sampler
from mini_vllm.core.work import EngineStepOutput
from mini_vllm.distributed import (
    DistributedExecutor,
    RankConfig,
    RankedWorker,
    ReferenceCollective,
    WorkerGroup,
)
from mini_vllm.execution import Worker
from mini_vllm.model.compile_runner import CompileAwareVocabParallelTinyRunner
from mini_vllm.model.tiny_transformer import TinyDecoderLM
from mini_vllm.observability import (
    AsyncRuntimeSnapshot,
    ExecutionSnapshot,
    RankExecutionSnapshot,
    RequestSnapshot,
    RuntimeSnapshot,
    SchedulerSnapshot,
    SchedulingObservation,
)
from mini_vllm.runtime import RuntimeConfig
from mini_vllm.scheduler.prefix_aware import PrefixAwareBudgetedScheduler
from mini_vllm.speculative.draft import DraftProvider, IncrementDraftProvider
from mini_vllm.speculative.engine import SpeculativeEngine


class IntegratedEngine(SpeculativeEngine):
    """v1.0 composition with a stable, read-only observability surface."""

    def __init__(self, *args, trace_history_size: int = 64, **kwargs) -> None:
        if trace_history_size <= 0:
            raise ValueError("trace_history_size must be > 0")
        super().__init__(*args, **kwargs)
        self._trace_history: deque[EngineStepOutput] = deque(maxlen=trace_history_size)

    def step(self) -> EngineStepOutput:
        output = super().step()
        self._trace_history.append(output)
        return output

    def _scheduler_snapshot(self) -> SchedulerSnapshot:
        scheduler = self.scheduler
        stats = scheduler.stats
        events = tuple(
            SchedulingObservation(
                kind=event.kind,
                request_id=event.request_id,
                raw_prefill_tokens=event.raw_prefill_tokens,
                reusable_tokens=event.reusable_tokens,
                expected_new_prefill_tokens=event.expected_new_prefill_tokens,
                bypass_count=event.bypass_count,
                fairness_protected=event.fairness_protected,
                actual_reuse_tokens=event.actual_reuse_tokens,
            )
            for event in scheduler.events[-32:]
        )
        return SchedulerSnapshot(
            name=type(scheduler).__name__,
            waiting_request_ids=tuple(request.request_id for request in scheduler.waiting),
            running_request_ids=tuple(request.request_id for request in scheduler.running),
            hint_refreshes=stats.hint_refreshes,
            admissions=stats.admissions,
            warm_preferred_admissions=stats.warm_preferred_admissions,
            fairness_overrides=stats.fairness_overrides,
            reconciliations=stats.reconciliations,
            stale_hint_reconciliations=stats.stale_hint_reconciliations,
            recent_events=events,
        )

    def _execution_snapshot(self) -> ExecutionSnapshot:
        group = self.executor.worker_group
        ranks: list[RankExecutionSnapshot] = []
        for entry in group.ranked_workers:
            runner = entry.worker.model_runner
            attention = runner.model.attention_backend
            compile_cache = runner.compile_cache
            ranks.append(
                RankExecutionSnapshot(
                    rank=entry.rank_config.rank,
                    world_size=entry.rank_config.world_size,
                    vocab_range=runner.vocab_range,
                    attention_backend=type(attention).__name__,
                    attention_total_calls=attention.stats.total_calls,
                    attention_reference_calls=attention.stats.reference_calls,
                    attention_vectorized_calls=attention.stats.vectorized_calls,
                    attention_fallbacks=attention.stats.fallbacks,
                    compile_enabled=compile_cache.config.enabled,
                    compile_backend=compile_cache.config.backend,
                    compile_attempts=compile_cache.stats.compile_attempts,
                    compile_cache_hits=compile_cache.stats.cache_hits,
                    compile_cache_misses=compile_cache.stats.cache_misses,
                    compile_fallbacks=compile_cache.stats.fallbacks,
                    eager_projection_calls=compile_cache.stats.eager_calls,
                )
            )
        return ExecutionSnapshot(
            executor=type(self.executor).__name__,
            world_size=group.world_size,
            execute_calls=group.execute_calls,
            speculative_execute_calls=group.speculative_execute_calls,
            ranks=tuple(ranks),
        )

    def snapshot(self) -> RuntimeSnapshot:
        """Return a read-only teaching view without changing runtime state."""
        requests = tuple(
            RequestSnapshot(
                request_id=request.request_id,
                status=request.status.name.lower(),
                prompt_tokens=len(request.prompt_token_ids),
                output_tokens=tuple(request.output_token_ids),
                num_computed_tokens=request.num_computed_tokens,
                preemption_count=request.preemption_count,
                finish_reason=request.finish_reason,
            )
            for request in self._requests.values()
        )
        recent_steps = tuple(self._trace_history)
        return RuntimeSnapshot(
            version=self.version,
            next_step=self._step,
            unfinished_request_ids=self.unfinished_request_ids,
            requests=requests,
            scheduler=self._scheduler_snapshot(),
            execution=self._execution_snapshot(),
            cache_stats=dict(self.executor.cache_stats()),
            preemption_events=self.preemption_events,
            last_step=recent_steps[-1] if recent_steps else None,
            recent_steps=recent_steps,
        )


class IntegratedAsyncEngine(AsyncEngine):
    core: IntegratedEngine

    def snapshot(self) -> AsyncRuntimeSnapshot:
        return AsyncRuntimeSnapshot(
            version=self.version,
            driver_running=self.driver_running,
            pending_request_ids=self.pending_request_ids,
            cancellation_requested_ids=tuple(sorted(self._cancel_requested)),
            core=self.core.snapshot(),
        )


def create_engine(
    *,
    runtime_config: RuntimeConfig | None = None,
    engine_config: EngineConfig | None = None,
    model_config: TinyModelConfig | None = None,
    cache_config: CacheConfig | None = None,
    draft_provider: DraftProvider | None = None,
    max_speculative_tokens: int | None = None,
    world_size: int | None = None,
    max_queue_size: int | None = None,
    compile_config: CompileConfig | None = None,
    attention_config: PagedAttentionConfig | None = None,
    fairness_bypass_limit: int | None = None,
) -> IntegratedAsyncEngine:
    if runtime_config is not None:
        explicit = (
            engine_config,
            model_config,
            cache_config,
            world_size,
            max_queue_size,
            compile_config,
            attention_config,
            fairness_bypass_limit,
        )
        if any(value is not None for value in explicit):
            raise ValueError(
                "runtime_config cannot be combined with individual construction options"
            )
        engine_config = runtime_config.engine
        model_config = runtime_config.model
        cache_config = runtime_config.cache
        compile_config = runtime_config.compile
        attention_config = runtime_config.attention
        world_size = runtime_config.world_size
        max_queue_size = runtime_config.max_queue_size
        fairness_bypass_limit = runtime_config.fairness_bypass_limit

    engine_config = engine_config or EngineConfig()
    model_config = model_config or TinyModelConfig()
    cache_config = cache_config or CacheConfig()
    compile_config = compile_config or CompileConfig()
    attention_config = attention_config or PagedAttentionConfig()
    world_size = 2 if world_size is None else world_size
    max_queue_size = 8 if max_queue_size is None else max_queue_size
    fairness_bypass_limit = 2 if fairness_bypass_limit is None else fairness_bypass_limit

    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    if world_size > model_config.vocab_size:
        raise ValueError("world_size cannot exceed vocabulary size")
    if max_queue_size <= 0:
        raise ValueError("max_queue_size must be > 0")

    ranked_workers: list[RankedWorker] = []
    for rank in range(world_size):
        rank_config = RankConfig(rank=rank, world_size=world_size)
        attention_backend = build_paged_attention_backend(attention_config)
        model = TinyDecoderLM(
            model_config,
            attention_backend=attention_backend,
        ).to(device=cache_config.device, dtype=cache_config.dtype)
        cache = TransactionalPrefixCachingPagedKVCache(
            num_layers=model_config.n_layers,
            num_blocks=cache_config.num_blocks,
            block_size=cache_config.block_size,
            num_kv_heads=model_config.n_kv_heads,
            head_dim=model_config.head_dim,
            dtype=cache_config.dtype,
            device=cache_config.device,
        )
        ranked_workers.append(
            RankedWorker(
                rank_config=rank_config,
                worker=Worker(
                    model_runner=CompileAwareVocabParallelTinyRunner(
                        model,
                        rank_config,
                        compile_config=compile_config,
                    ),
                    cache=cache,
                ),
            )
        )

    executor = DistributedExecutor(
        WorkerGroup(tuple(ranked_workers), collective=ReferenceCollective())
    )
    scheduler = PrefixAwareBudgetedScheduler(
        max_num_seqs=engine_config.max_num_seqs,
        max_num_batched_tokens=engine_config.max_num_batched_tokens,
        fairness_bypass_limit=fairness_bypass_limit,
    )
    core = IntegratedEngine(
        config=engine_config,
        scheduler=scheduler,
        executor=executor,
        sampler=Sampler(),
        version="v1.0",
        draft_provider=draft_provider or IncrementDraftProvider(model_config.vocab_size),
        max_speculative_tokens=(
            engine_config.max_speculative_tokens
            if max_speculative_tokens is None
            else max_speculative_tokens
        ),
    )
    return IntegratedAsyncEngine(
        core,
        max_queue_size=max_queue_size,
        version="v1.0",
    )

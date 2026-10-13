from __future__ import annotations

from mini_vllm.attention import PagedAttentionConfig
from mini_vllm.compile_runtime import CompileConfig
from mini_vllm.config import CacheConfig, EngineConfig, TinyModelConfig
from mini_vllm.core.sampler import Sampler
from mini_vllm.multiprocess import (
    MultiprocessExecutor,
    MultiprocessExecutorUnhealthy,
    WorkerProcessDied,
    WorkerSpec,
)
from mini_vllm.observability import ExecutionSnapshot, RankExecutionSnapshot
from mini_vllm.runtime import MultiprocessConfig, RuntimeConfig
from mini_vllm.scheduler.prefix_aware import PrefixAwareBudgetedScheduler
from mini_vllm.speculative.draft import DraftProvider, IncrementDraftProvider
from mini_vllm.versions.v1_0.engine import IntegratedAsyncEngine, IntegratedEngine


class _SnapshotSafeMultiprocessExecutor(MultiprocessExecutor):
    """v1.1 executor refinements used by the integrated public runtime."""

    def _require_healthy(self) -> None:
        if self._closed:
            raise MultiprocessExecutorUnhealthy("multiprocess executor is closed")
        for client in self._clients:
            if not client.process.is_alive():
                client.lifecycle_state = "dead"
                self._healthy = False
                raise WorkerProcessDied(
                    f"worker rank {client.spec.rank} exited unexpectedly; "
                    f"exitcode={client.process.exitcode}"
                )
        super()._require_healthy()

    def cache_stats(self) -> dict[str, int | float]:
        """Read cache observations without changing serving-command counters."""
        self._require_healthy()
        stats = tuple(
            client.request("cache_stats", count=False) for client in self._clients
        )
        first = stats[0]
        if not isinstance(first, dict) or any(stat != first for stat in stats[1:]):
            self._healthy = False
            raise MultiprocessExecutorUnhealthy(
                "replicated rank cache statistics diverged"
            )
        return {**first, "world_size": self.world_size}


class MultiprocessIntegratedEngine(IntegratedEngine):
    executor: MultiprocessExecutor

    def _execution_snapshot(self) -> ExecutionSnapshot:
        workers = self.executor.transport_snapshot()
        ranks = tuple(
            RankExecutionSnapshot(
                rank=worker.rank,
                world_size=worker.world_size,
                vocab_range=worker.observation.vocab_range,
                attention_backend=worker.observation.attention_backend,
                attention_total_calls=worker.observation.attention_total_calls,
                attention_reference_calls=worker.observation.attention_reference_calls,
                attention_vectorized_calls=worker.observation.attention_vectorized_calls,
                attention_fallbacks=worker.observation.attention_fallbacks,
                compile_enabled=worker.observation.compile_enabled,
                compile_backend=worker.observation.compile_backend,
                compile_attempts=worker.observation.compile_attempts,
                compile_cache_hits=worker.observation.compile_cache_hits,
                compile_cache_misses=worker.observation.compile_cache_misses,
                compile_fallbacks=worker.observation.compile_fallbacks,
                eager_projection_calls=worker.observation.eager_projection_calls,
                boundary="multiprocess-ipc",
                process_id=worker.process_id,
                worker_state=worker.lifecycle_state,
                transport_commands_sent=worker.commands_sent,
                transport_commands_completed=worker.commands_completed,
                remote_errors=worker.remote_errors,
                transport_failures=worker.transport_failures,
            )
            for worker in workers
        )
        return ExecutionSnapshot(
            executor="MultiprocessExecutor",
            world_size=self.executor.world_size,
            execute_calls=self.executor.execute_calls,
            speculative_execute_calls=self.executor.speculative_execute_calls,
            ranks=ranks,
            boundary="multiprocess-ipc",
            healthy=self.executor.healthy,
        )


class MultiprocessAsyncEngine(IntegratedAsyncEngine):
    core: MultiprocessIntegratedEngine

    def close(self) -> None:
        """Synchronously stop and join all worker processes when the driver is idle."""
        if self.driver_running:
            raise RuntimeError("cannot close while the async driver is running; await aclose()")
        self.core.executor.shutdown()

    async def aclose(self) -> None:
        """Wait for current async work, then stop and join all worker processes."""
        await self.wait_idle()
        self.core.executor.shutdown()

    def __enter__(self) -> "MultiprocessAsyncEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


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
    multiprocess_config: MultiprocessConfig | None = None,
) -> MultiprocessAsyncEngine:
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
            multiprocess_config,
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
        multiprocess_config = runtime_config.multiprocess
        world_size = runtime_config.world_size
        max_queue_size = runtime_config.max_queue_size
        fairness_bypass_limit = runtime_config.fairness_bypass_limit

    engine_config = engine_config or EngineConfig()
    model_config = model_config or TinyModelConfig()
    cache_config = cache_config or CacheConfig()
    compile_config = compile_config or CompileConfig()
    attention_config = attention_config or PagedAttentionConfig()
    multiprocess_config = multiprocess_config or MultiprocessConfig()
    world_size = 2 if world_size is None else world_size
    max_queue_size = 8 if max_queue_size is None else max_queue_size
    fairness_bypass_limit = 2 if fairness_bypass_limit is None else fairness_bypass_limit

    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    if world_size > model_config.vocab_size:
        raise ValueError("world_size cannot exceed vocabulary size")
    if max_queue_size <= 0:
        raise ValueError("max_queue_size must be > 0")

    specs = tuple(
        WorkerSpec(
            rank=rank,
            world_size=world_size,
            model_config=model_config,
            cache_config=cache_config,
            compile_config=compile_config,
            attention_config=attention_config,
        )
        for rank in range(world_size)
    )
    executor = _SnapshotSafeMultiprocessExecutor(
        specs,
        start_method=multiprocess_config.start_method,
        command_timeout_s=multiprocess_config.command_timeout_s,
        shutdown_timeout_s=multiprocess_config.shutdown_timeout_s,
    )
    scheduler = PrefixAwareBudgetedScheduler(
        max_num_seqs=engine_config.max_num_seqs,
        max_num_batched_tokens=engine_config.max_num_batched_tokens,
        fairness_bypass_limit=fairness_bypass_limit,
    )
    core = MultiprocessIntegratedEngine(
        config=engine_config,
        scheduler=scheduler,
        executor=executor,
        sampler=Sampler(),
        version="v1.1",
        draft_provider=draft_provider or IncrementDraftProvider(model_config.vocab_size),
        max_speculative_tokens=(
            engine_config.max_speculative_tokens
            if max_speculative_tokens is None
            else max_speculative_tokens
        ),
    )
    return MultiprocessAsyncEngine(core, max_queue_size=max_queue_size, version="v1.1")

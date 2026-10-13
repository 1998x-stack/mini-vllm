from __future__ import annotations

import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from mini_vllm.artifacts import (
    ValidatedModelArtifact,
    WhitespaceTokenizer,
    export_tiny_model_artifact,
    load_model_artifact,
)
from mini_vllm.artifacts.executor import ArtifactMultiprocessExecutor
from mini_vllm.attention import PagedAttentionConfig
from mini_vllm.compile_runtime import CompileConfig
from mini_vllm.config import CacheConfig, EngineConfig, TinyModelConfig
from mini_vllm.core.request import Request, SamplingParams
from mini_vllm.core.sampler import Sampler
from mini_vllm.multiprocess import WorkerSpec
from mini_vllm.observability import (
    ExecutionSnapshot,
    ModelArtifactSnapshot,
    RankExecutionSnapshot,
    RuntimeSnapshot,
)
from mini_vllm.runtime import MultiprocessConfig, RuntimeConfig
from mini_vllm.scheduler.prefix_aware import PrefixAwareBudgetedScheduler
from mini_vllm.speculative.draft import DraftProvider, IncrementDraftProvider
from mini_vllm.versions.v1_1.engine import (
    MultiprocessAsyncEngine,
    MultiprocessIntegratedEngine,
)


class ArtifactIntegratedEngine(MultiprocessIntegratedEngine):
    executor: ArtifactMultiprocessExecutor

    def __init__(
        self,
        *args,
        artifact: ValidatedModelArtifact,
        cache_config: CacheConfig,
        **kwargs,
    ) -> None:
        self.artifact = artifact
        self._artifact_cache_config = cache_config
        super().__init__(*args, **kwargs)

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
                artifact_identity=worker.observation.artifact_identity,
                tokenizer_identity=worker.observation.tokenizer_identity,
                artifact_architecture=worker.observation.artifact_architecture,
            )
            for worker in workers
        )
        return ExecutionSnapshot(
            executor="ArtifactMultiprocessExecutor",
            world_size=self.executor.world_size,
            execute_calls=self.executor.execute_calls,
            speculative_execute_calls=self.executor.speculative_execute_calls,
            ranks=ranks,
            boundary="multiprocess-ipc",
            healthy=self.executor.healthy,
        )

    def snapshot(self) -> RuntimeSnapshot:
        base = super().snapshot()
        expected = self.artifact.identity.digest
        rank_agreement = bool(base.execution.ranks) and all(
            rank.artifact_identity == expected
            and rank.tokenizer_identity == self.artifact.tokenizer.digest
            for rank in base.execution.ranks
        )
        metadata = ModelArtifactSnapshot(
            identity=expected,
            architecture=self.artifact.identity.architecture,
            format_version=self.artifact.identity.format_version,
            tokenizer_identity=self.artifact.tokenizer.digest,
            tokenizer_compatibility_id=self.artifact.tokenizer.compatibility_id,
            vocab_size=self.artifact.model_config.vocab_size,
            weight_dtype=self.artifact.spec.weight_dtype,
            runtime_dtype=str(self._artifact_cache_config.dtype).split(".")[-1],
            device=self._artifact_cache_config.device,
            rank_identity_agreement=rank_agreement,
        )
        return replace(base, model_artifact=metadata)


class ArtifactAsyncEngine(MultiprocessAsyncEngine):
    core: ArtifactIntegratedEngine

    def __init__(
        self,
        *args,
        tokenizer: WhitespaceTokenizer,
        owned_artifact_dir: str | None = None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self._owned_artifact_dir = owned_artifact_dir
        super().__init__(*args, **kwargs)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return self.tokenizer.decode(token_ids)

    def submit_text(
        self,
        request_id: str,
        text: str,
        *,
        max_new_tokens: int,
        sampling_params: SamplingParams | None = None,
    ):
        return self.submit(
            Request(
                request_id=request_id,
                prompt_token_ids=self.encode(text),
                max_new_tokens=max_new_tokens,
                sampling_params=sampling_params or SamplingParams(),
            )
        )

    def _cleanup_owned_artifact(self) -> None:
        if self._owned_artifact_dir is None:
            return
        shutil.rmtree(self._owned_artifact_dir, ignore_errors=True)
        self._owned_artifact_dir = None

    def close(self) -> None:
        super().close()
        self._cleanup_owned_artifact()

    async def aclose(self) -> None:
        await super().aclose()
        self._cleanup_owned_artifact()


def create_engine(
    *,
    runtime_config: RuntimeConfig | None = None,
    engine_config: EngineConfig | None = None,
    model_config: TinyModelConfig | None = None,
    model_artifact: str | Path | None = None,
    cache_config: CacheConfig | None = None,
    draft_provider: DraftProvider | None = None,
    max_speculative_tokens: int | None = None,
    world_size: int | None = None,
    max_queue_size: int | None = None,
    compile_config: CompileConfig | None = None,
    attention_config: PagedAttentionConfig | None = None,
    fairness_bypass_limit: int | None = None,
    multiprocess_config: MultiprocessConfig | None = None,
) -> ArtifactAsyncEngine:
    if runtime_config is not None:
        explicit = (
            engine_config,
            model_config,
            model_artifact,
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
        model_artifact = runtime_config.model_artifact
        cache_config = runtime_config.cache
        compile_config = runtime_config.compile
        attention_config = runtime_config.attention
        multiprocess_config = runtime_config.multiprocess
        world_size = runtime_config.world_size
        max_queue_size = runtime_config.max_queue_size
        fairness_bypass_limit = runtime_config.fairness_bypass_limit

    if model_artifact is not None and runtime_config is None and model_config is not None:
        raise ValueError("model_artifact cannot be combined with model_config")

    engine_config = engine_config or EngineConfig()
    cache_config = cache_config or CacheConfig()
    compile_config = compile_config or CompileConfig()
    attention_config = attention_config or PagedAttentionConfig()
    multiprocess_config = multiprocess_config or MultiprocessConfig()
    max_queue_size = 8 if max_queue_size is None else max_queue_size
    fairness_bypass_limit = 2 if fairness_bypass_limit is None else fairness_bypass_limit

    owned_artifact_dir: str | None = None
    try:
        if model_artifact is None:
            owned_artifact_dir = tempfile.mkdtemp(prefix="mini-vllm-v1.2-")
            artifact = export_tiny_model_artifact(
                owned_artifact_dir,
                model_config or TinyModelConfig(),
            )
        else:
            artifact = load_model_artifact(model_artifact)
        effective_model_config = artifact.model_config
        world_size = 2 if world_size is None else world_size
        if world_size <= 0:
            raise ValueError("world_size must be > 0")
        if world_size > effective_model_config.vocab_size:
            raise ValueError("world_size cannot exceed artifact vocabulary size")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0")

        specs = tuple(
            WorkerSpec(
                rank=rank,
                world_size=world_size,
                model_config=effective_model_config,
                cache_config=cache_config,
                compile_config=compile_config,
                attention_config=attention_config,
                artifact_spec=artifact.spec,
            )
            for rank in range(world_size)
        )
        executor = ArtifactMultiprocessExecutor(
            specs,
            start_method=multiprocess_config.start_method,
            command_timeout_s=multiprocess_config.command_timeout_s,
            shutdown_timeout_s=multiprocess_config.shutdown_timeout_s,
        )
    except Exception:
        if owned_artifact_dir is not None:
            shutil.rmtree(owned_artifact_dir, ignore_errors=True)
        raise

    scheduler = PrefixAwareBudgetedScheduler(
        max_num_seqs=engine_config.max_num_seqs,
        max_num_batched_tokens=engine_config.max_num_batched_tokens,
        fairness_bypass_limit=fairness_bypass_limit,
    )
    core = ArtifactIntegratedEngine(
        config=engine_config,
        scheduler=scheduler,
        executor=executor,
        sampler=Sampler(),
        version="v1.2",
        artifact=artifact,
        cache_config=cache_config,
        draft_provider=draft_provider
        or IncrementDraftProvider(effective_model_config.vocab_size),
        max_speculative_tokens=(
            engine_config.max_speculative_tokens
            if max_speculative_tokens is None
            else max_speculative_tokens
        ),
    )
    return ArtifactAsyncEngine(
        core,
        max_queue_size=max_queue_size,
        version="v1.2",
        tokenizer=WhitespaceTokenizer(artifact.tokenizer),
        owned_artifact_dir=owned_artifact_dir,
    )

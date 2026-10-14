from __future__ import annotations

import shutil
import tempfile
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
from mini_vllm.core.sampler import Sampler
from mini_vllm.multiprocess import WorkerSpec
from mini_vllm.network import NetworkServer
from mini_vllm.runtime import MultiprocessConfig, NetworkConfig, RuntimeConfig
from mini_vllm.scheduler.prefix_aware import PrefixAwareBudgetedScheduler
from mini_vllm.speculative.draft import DraftProvider, IncrementDraftProvider
from mini_vllm.versions.v1_2.engine import ArtifactAsyncEngine, ArtifactIntegratedEngine


class NetworkAsyncEngine(ArtifactAsyncEngine):
    """v1.2 artifact runtime plus an explicit network-server construction edge."""

    def create_server(
        self,
        network_config: NetworkConfig | None = None,
    ) -> NetworkServer:
        return NetworkServer(
            self,
            config=network_config or NetworkConfig(),
            owns_engine=False,
        )


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
) -> NetworkAsyncEngine:
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
            owned_artifact_dir = tempfile.mkdtemp(prefix="mini-vllm-v1.3-")
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
        version="v1.3",
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
    return NetworkAsyncEngine(
        core,
        max_queue_size=max_queue_size,
        version="v1.3",
        tokenizer=WhitespaceTokenizer(artifact.tokenizer),
        owned_artifact_dir=owned_artifact_dir,
    )


def create_server(
    *,
    engine: NetworkAsyncEngine | None = None,
    network_config: NetworkConfig | None = None,
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
) -> NetworkServer:
    engine_options = (
        runtime_config,
        engine_config,
        model_config,
        model_artifact,
        cache_config,
        draft_provider,
        max_speculative_tokens,
        world_size,
        max_queue_size,
        compile_config,
        attention_config,
        fairness_bypass_limit,
        multiprocess_config,
    )
    if engine is not None:
        if any(value is not None for value in engine_options):
            raise ValueError("existing engine cannot be combined with engine construction options")
        return NetworkServer(
            engine,
            config=network_config or NetworkConfig(),
            owns_engine=False,
        )

    if runtime_config is not None:
        if network_config is not None:
            raise ValueError("runtime_config.network cannot be combined with network_config")
        selected_network = runtime_config.network
        built = create_engine(runtime_config=runtime_config)
    else:
        selected_network = network_config or NetworkConfig()
        built = create_engine(
            engine_config=engine_config,
            model_config=model_config,
            model_artifact=model_artifact,
            cache_config=cache_config,
            draft_provider=draft_provider,
            max_speculative_tokens=max_speculative_tokens,
            world_size=world_size,
            max_queue_size=max_queue_size,
            compile_config=compile_config,
            attention_config=attention_config,
            fairness_bypass_limit=fairness_bypass_limit,
            multiprocess_config=multiprocess_config,
        )
    return NetworkServer(built, config=selected_network, owns_engine=True)

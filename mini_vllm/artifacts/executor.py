from __future__ import annotations

import multiprocessing as mp
import os
from multiprocessing.connection import Connection

import torch

from mini_vllm.artifacts import load_model_artifact
from mini_vllm.attention import build_paged_attention_backend
from mini_vllm.cache.transactional import TransactionalPrefixCachingPagedKVCache
from mini_vllm.distributed.collective import ReferenceCollective
from mini_vllm.distributed.rank import RankConfig
from mini_vllm.execution.worker import Worker
from mini_vllm.model.compile_runner import CompileAwareVocabParallelTinyRunner
from mini_vllm.model.factory import build_model_from_artifact
from mini_vllm.multiprocess.executor import (
    MultiprocessExecutor,
    MultiprocessExecutorUnhealthy,
    WorkerProcessDied,
    WorkerRemoteError,
    WorkerTransportError,
    _RemoteWorkerClient,
    _serialize_execution,
    _serialize_speculative,
)
from mini_vllm.multiprocess.protocol import (
    CheckpointHandle,
    TransportCommand,
    TransportResponse,
    WorkerExecutionObservation,
    WorkerReady,
    WorkerSpec,
)


def _observe_artifact_worker(
    worker: Worker,
    rank_config: RankConfig,
) -> WorkerExecutionObservation:
    runner = worker.model_runner
    attention = runner.model.attention_backend
    compile_cache = runner.compile_cache
    return WorkerExecutionObservation(
        rank=rank_config.rank,
        world_size=rank_config.world_size,
        process_id=os.getpid(),
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
        artifact_identity=runner.model.artifact_identity,
        tokenizer_identity=runner.model.tokenizer_identity,
        artifact_architecture=runner.model.artifact_architecture,
    )


def _build_artifact_worker(spec: WorkerSpec) -> tuple[Worker, RankConfig]:
    if spec.artifact_spec is None:
        raise ValueError("artifact worker requires WorkerSpec.artifact_spec")
    artifact = load_model_artifact(
        spec.artifact_spec.root,
        expected_identity=spec.artifact_spec.identity,
    )
    if artifact.model_config != spec.model_config:
        raise ValueError("worker artifact model config differs from parent-validated spec")
    rank_config = RankConfig(rank=spec.rank, world_size=spec.world_size)
    attention_backend = build_paged_attention_backend(spec.attention_config)
    model = build_model_from_artifact(
        artifact,
        attention_backend=attention_backend,
        device=spec.cache_config.device,
        dtype=spec.cache_config.dtype,
    )
    cache = TransactionalPrefixCachingPagedKVCache(
        num_layers=spec.model_config.n_layers,
        num_blocks=spec.cache_config.num_blocks,
        block_size=spec.cache_config.block_size,
        num_kv_heads=spec.model_config.n_kv_heads,
        head_dim=spec.model_config.head_dim,
        dtype=spec.cache_config.dtype,
        device=spec.cache_config.device,
    )
    runner = CompileAwareVocabParallelTinyRunner(
        model,
        rank_config,
        compile_config=spec.compile_config,
    )
    return Worker(model_runner=runner, cache=cache), rank_config


def _artifact_worker_process_main(connection: Connection, spec: WorkerSpec) -> None:
    worker: Worker | None = None
    rank_config: RankConfig | None = None
    checkpoints: dict[int, object] = {}
    next_checkpoint_id = 1
    try:
        torch.set_num_threads(1)
        worker, rank_config = _build_artifact_worker(spec)
        connection.send(
            TransportResponse(
                correlation_id=0,
                ok=True,
                payload=WorkerReady(_observe_artifact_worker(worker, rank_config)),
            )
        )
        while True:
            try:
                command = connection.recv()
            except EOFError:
                return
            if not isinstance(command, TransportCommand):
                connection.send(
                    TransportResponse(
                        correlation_id=-1,
                        ok=False,
                        error_type="ProtocolError",
                        error_message="child received a non-TransportCommand payload",
                    )
                )
                continue
            try:
                operation = command.operation
                payload = command.payload
                if operation == "shutdown":
                    connection.send(TransportResponse(command.correlation_id, True, None))
                    return
                if operation == "estimate_prefix_reuse":
                    result = worker.estimate_prefix_reuse(list(payload))
                elif operation == "prepare_request":
                    request_id, prompt = payload
                    result = worker.prepare_request(request_id, list(prompt))
                elif operation == "ensure_capacity":
                    request_id, total_tokens = payload
                    result = worker.ensure_capacity(request_id, total_tokens)
                elif operation == "execute":
                    result = _serialize_execution(worker.execute(payload))
                elif operation == "execute_speculative":
                    result = _serialize_speculative(worker.execute_speculative(payload))
                elif operation == "commit_work":
                    request_id, start_pos, token_ids, prompt_length = payload
                    worker.commit_work(
                        request_id,
                        start_pos=start_pos,
                        token_ids=token_ids,
                        prompt_length=prompt_length,
                    )
                    result = None
                elif operation == "checkpoint":
                    request_id, num_tokens = payload
                    checkpoint = worker.checkpoint(request_id, num_tokens)
                    checkpoint_id = next_checkpoint_id
                    next_checkpoint_id += 1
                    checkpoints[checkpoint_id] = checkpoint
                    result = CheckpointHandle(checkpoint_id, request_id, num_tokens)
                elif operation == "rollback":
                    checkpoint_id, retain_tokens = payload
                    checkpoint = checkpoints[checkpoint_id]
                    worker.rollback(checkpoint, retain_tokens=retain_tokens)
                    del checkpoints[checkpoint_id]
                    result = None
                elif operation == "commit_checkpoint":
                    checkpoint_id = int(payload)
                    checkpoint = checkpoints[checkpoint_id]
                    worker.commit_checkpoint(checkpoint)
                    del checkpoints[checkpoint_id]
                    result = None
                elif operation == "free":
                    worker.free(str(payload))
                    result = None
                elif operation == "cache_stats":
                    result = worker.cache_stats()
                elif operation == "snapshot_execution":
                    result = _observe_artifact_worker(worker, rank_config)
                else:
                    raise ValueError(f"unknown worker operation: {operation}")
                connection.send(TransportResponse(command.correlation_id, True, result))
            except Exception as exc:
                connection.send(
                    TransportResponse(
                        correlation_id=command.correlation_id,
                        ok=False,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
    except Exception as exc:
        try:
            connection.send(
                TransportResponse(
                    correlation_id=0,
                    ok=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
        except Exception:
            pass
    finally:
        connection.close()


class _ArtifactRemoteWorkerClient(_RemoteWorkerClient):
    def __init__(
        self,
        spec: WorkerSpec,
        *,
        context: mp.context.BaseContext,
        command_timeout_s: float,
        shutdown_timeout_s: float,
    ) -> None:
        self.spec = spec
        self.command_timeout_s = command_timeout_s
        self.shutdown_timeout_s = shutdown_timeout_s
        self.commands_sent = 0
        self.commands_completed = 0
        self.remote_errors = 0
        self.transport_failures = 0
        self.lifecycle_state = "starting"
        self._next_correlation_id = 1
        parent, child = context.Pipe(duplex=True)
        self.connection = parent
        self.process = context.Process(
            target=_artifact_worker_process_main,
            args=(child, spec),
            name=f"mini-vllm-artifact-rank-{spec.rank}",
            daemon=True,
        )
        self.process.start()
        child.close()
        response = self._receive(0, count_command=False)
        if not response.ok or not isinstance(response.payload, WorkerReady):
            self.lifecycle_state = "dead"
            self._join_or_terminate()
            if response.ok:
                raise WorkerTransportError(
                    f"worker rank {spec.rank} sent invalid readiness payload"
                )
            raise WorkerRemoteError(
                spec.rank,
                response.error_type or "StartupError",
                response.error_message or "worker startup failed",
            )
        self.last_observation = response.payload.observation
        self.lifecycle_state = "ready"


class ArtifactMultiprocessExecutor(MultiprocessExecutor):
    """v1.2 executor: every rank independently validates and loads one artifact."""

    def __init__(
        self,
        worker_specs: tuple[WorkerSpec, ...],
        *,
        start_method: str = "spawn",
        command_timeout_s: float = 30.0,
        shutdown_timeout_s: float = 2.0,
        collective: ReferenceCollective | None = None,
    ) -> None:
        if not worker_specs:
            raise ValueError("ArtifactMultiprocessExecutor requires worker specs")
        if any(spec.artifact_spec is None for spec in worker_specs):
            raise ValueError("every v1.2 worker spec must carry an artifact spec")
        if command_timeout_s <= 0 or shutdown_timeout_s <= 0:
            raise ValueError("transport timeouts must be > 0")
        ordered = tuple(sorted(worker_specs, key=lambda spec: spec.rank))
        world_sizes = {spec.world_size for spec in ordered}
        if len(world_sizes) != 1:
            raise ValueError("worker specs must agree on world_size")
        self.world_size = next(iter(world_sizes))
        if len(ordered) != self.world_size:
            raise ValueError("worker specs must contain exactly world_size ranks")
        if tuple(spec.rank for spec in ordered) != tuple(range(self.world_size)):
            raise ValueError("worker specs must cover ranks 0..world_size-1")
        artifact_ids = {spec.artifact_spec.identity.digest for spec in ordered if spec.artifact_spec}
        if len(artifact_ids) != 1:
            raise ValueError("worker specs must agree on artifact identity")

        self.collective = collective or ReferenceCollective()
        self.execute_calls = 0
        self.speculative_execute_calls = 0
        self._closed = False
        self._healthy = True
        context = mp.get_context(start_method)
        clients: list[_ArtifactRemoteWorkerClient] = []
        try:
            for spec in ordered:
                clients.append(
                    _ArtifactRemoteWorkerClient(
                        spec,
                        context=context,
                        command_timeout_s=command_timeout_s,
                        shutdown_timeout_s=shutdown_timeout_s,
                    )
                )
            expected_identity = next(iter(artifact_ids))
            observations = tuple(client.last_observation for client in clients)
            if any(obs.artifact_identity != expected_identity for obs in observations):
                raise MultiprocessExecutorUnhealthy(
                    "worker ranks did not agree with parent artifact identity"
                )
            tokenizer_ids = {obs.tokenizer_identity for obs in observations}
            if len(tokenizer_ids) != 1:
                raise MultiprocessExecutorUnhealthy(
                    "worker ranks disagree on tokenizer identity"
                )
        except Exception:
            for client in clients:
                client.shutdown()
            raise
        self._clients = tuple(clients)

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

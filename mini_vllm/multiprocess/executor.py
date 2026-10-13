from __future__ import annotations

import multiprocessing as mp
import os
from multiprocessing.connection import Connection

import torch

from mini_vllm.attention import build_paged_attention_backend
from mini_vllm.cache.base import PrefixReuseHint
from mini_vllm.cache.transactional import TransactionalPrefixCachingPagedKVCache
from mini_vllm.distributed.collective import ReferenceCollective
from mini_vllm.distributed.rank import RankConfig
from mini_vllm.execution.executor import Executor
from mini_vllm.execution.types import (
    ExecutionRequest,
    ExecutionResult,
    SpeculativeExecutionRequest,
    SpeculativeExecutionResult,
)
from mini_vllm.execution.worker import Worker
from mini_vllm.model.compile_runner import CompileAwareVocabParallelTinyRunner
from mini_vllm.model.tiny_transformer import TinyDecoderLM
from mini_vllm.multiprocess.protocol import (
    CheckpointHandle,
    MultiprocessCacheCheckpoint,
    SerializableExecutionResult,
    SerializableSpeculativeResult,
    TensorPayload,
    TransportCommand,
    TransportResponse,
    TransportWorkerSnapshot,
    WorkerExecutionObservation,
    WorkerReady,
    WorkerSpec,
)


class WorkerTransportError(RuntimeError):
    pass


class WorkerRemoteError(WorkerTransportError):
    def __init__(self, rank: int, error_type: str, error_message: str) -> None:
        super().__init__(f"worker rank {rank} reported {error_type}: {error_message}")
        self.rank = rank
        self.error_type = error_type
        self.error_message = error_message


class WorkerProcessDied(WorkerTransportError):
    pass


class WorkerTransportTimeout(WorkerTransportError):
    pass


class MultiprocessExecutorUnhealthy(WorkerTransportError):
    pass


def _build_worker(spec: WorkerSpec) -> tuple[Worker, RankConfig]:
    rank_config = RankConfig(rank=spec.rank, world_size=spec.world_size)
    attention_backend = build_paged_attention_backend(spec.attention_config)
    model = TinyDecoderLM(
        spec.model_config,
        attention_backend=attention_backend,
    ).to(device=spec.cache_config.device, dtype=spec.cache_config.dtype)
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


def _observe_worker(worker: Worker, rank_config: RankConfig) -> WorkerExecutionObservation:
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
    )


def _serialize_execution(result: ExecutionResult) -> SerializableExecutionResult:
    return SerializableExecutionResult(
        logits_by_request={
            request_id: TensorPayload.from_tensor(logits)
            for request_id, logits in result.logits_by_request.items()
        }
    )


def _serialize_speculative(
    result: SpeculativeExecutionResult,
) -> SerializableSpeculativeResult:
    return SerializableSpeculativeResult(
        request_id=result.request_id,
        logits_trace=tuple(TensorPayload.from_tensor(logits) for logits in result.logits_trace),
    )


def _worker_process_main(connection: Connection, spec: WorkerSpec) -> None:
    worker: Worker | None = None
    rank_config: RankConfig | None = None
    checkpoints: dict[int, object] = {}
    next_checkpoint_id = 1
    try:
        torch.set_num_threads(1)
        worker, rank_config = _build_worker(spec)
        connection.send(
            TransportResponse(
                correlation_id=0,
                ok=True,
                payload=WorkerReady(_observe_worker(worker, rank_config)),
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
                    result = _observe_worker(worker, rank_config)
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


class _RemoteWorkerClient:
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
            target=_worker_process_main,
            args=(child, spec),
            name=f"mini-vllm-rank-{spec.rank}",
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

    @property
    def alive(self) -> bool:
        return self.process.is_alive()

    @property
    def healthy(self) -> bool:
        return self.lifecycle_state == "ready" and self.alive

    def _mark_transport_failure(self, state: str) -> None:
        self.transport_failures += 1
        self.lifecycle_state = state

    def _receive(self, correlation_id: int, *, count_command: bool) -> TransportResponse:
        if not self.connection.poll(self.command_timeout_s):
            if not self.process.is_alive():
                self._mark_transport_failure("dead")
                raise WorkerProcessDied(
                    f"worker rank {self.spec.rank} exited before response {correlation_id}; "
                    f"exitcode={self.process.exitcode}"
                )
            self._mark_transport_failure("unhealthy")
            raise WorkerTransportTimeout(
                f"worker rank {self.spec.rank} timed out waiting for response {correlation_id}"
            )
        try:
            response = self.connection.recv()
        except EOFError as exc:
            self._mark_transport_failure("dead")
            raise WorkerProcessDied(
                f"worker rank {self.spec.rank} closed IPC unexpectedly; "
                f"exitcode={self.process.exitcode}"
            ) from exc
        if not isinstance(response, TransportResponse):
            self._mark_transport_failure("unhealthy")
            raise WorkerTransportError(
                f"worker rank {self.spec.rank} returned a non-protocol response"
            )
        if response.correlation_id != correlation_id:
            self._mark_transport_failure("unhealthy")
            raise WorkerTransportError(
                f"worker rank {self.spec.rank} correlation mismatch: "
                f"expected {correlation_id}, got {response.correlation_id}"
            )
        if count_command:
            self.commands_completed += 1
        return response

    def request(self, operation: str, payload: object = None, *, count: bool = True) -> object:
        if self.lifecycle_state not in {"ready"} or not self.process.is_alive():
            self.lifecycle_state = "dead" if not self.process.is_alive() else self.lifecycle_state
            raise WorkerProcessDied(
                f"worker rank {self.spec.rank} is not available; "
                f"state={self.lifecycle_state}, exitcode={self.process.exitcode}"
            )
        correlation_id = self._next_correlation_id
        self._next_correlation_id += 1
        if count:
            self.commands_sent += 1
        try:
            self.connection.send(TransportCommand(correlation_id, operation, payload))
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._mark_transport_failure("dead")
            raise WorkerProcessDied(
                f"worker rank {self.spec.rank} failed while sending {operation}"
            ) from exc
        response = self._receive(correlation_id, count_command=count)
        if not response.ok:
            self.remote_errors += 1
            raise WorkerRemoteError(
                self.spec.rank,
                response.error_type or "RemoteError",
                response.error_message or "unknown child error",
            )
        return response.payload

    def snapshot_observation(self) -> WorkerExecutionObservation:
        if self.healthy:
            try:
                observation = self.request("snapshot_execution", count=False)
                if isinstance(observation, WorkerExecutionObservation):
                    self.last_observation = observation
            except WorkerTransportError:
                pass
        return self.last_observation

    def snapshot(self) -> TransportWorkerSnapshot:
        observation = self.snapshot_observation()
        return TransportWorkerSnapshot(
            rank=self.spec.rank,
            world_size=self.spec.world_size,
            process_id=self.process.pid or observation.process_id,
            lifecycle_state=self.lifecycle_state,
            commands_sent=self.commands_sent,
            commands_completed=self.commands_completed,
            remote_errors=self.remote_errors,
            transport_failures=self.transport_failures,
            observation=observation,
        )

    def _join_or_terminate(self) -> None:
        self.process.join(self.shutdown_timeout_s)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(self.shutdown_timeout_s)

    def shutdown(self) -> None:
        if self.lifecycle_state == "stopped":
            return
        if self.healthy:
            try:
                self.request("shutdown")
            except WorkerTransportError:
                pass
        self._join_or_terminate()
        self.lifecycle_state = "stopped" if not self.process.is_alive() else "dead"
        try:
            self.connection.close()
        except OSError:
            pass


class MultiprocessExecutor(Executor):
    """One OS process per logical rank with explicit command/result transport."""

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
            raise ValueError("MultiprocessExecutor requires at least one worker spec")
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

        self.collective = collective or ReferenceCollective()
        self.execute_calls = 0
        self.speculative_execute_calls = 0
        self._closed = False
        self._healthy = True
        context = mp.get_context(start_method)
        clients: list[_RemoteWorkerClient] = []
        try:
            for spec in ordered:
                clients.append(
                    _RemoteWorkerClient(
                        spec,
                        context=context,
                        command_timeout_s=command_timeout_s,
                        shutdown_timeout_s=shutdown_timeout_s,
                    )
                )
        except Exception:
            for client in clients:
                client.shutdown()
            raise
        self._clients = tuple(clients)

    @property
    def worker_pids(self) -> tuple[int, ...]:
        return tuple(int(client.process.pid or -1) for client in self._clients)

    @property
    def worker_alive(self) -> tuple[bool, ...]:
        return tuple(client.process.is_alive() for client in self._clients)

    @property
    def healthy(self) -> bool:
        return self._healthy and not self._closed and all(client.healthy for client in self._clients)

    def _require_healthy(self) -> None:
        if not self.healthy:
            raise MultiprocessExecutorUnhealthy("multiprocess executor is not healthy")

    def _mark_unhealthy_if_transport(self, exc: Exception) -> None:
        if isinstance(exc, (WorkerProcessDied, WorkerTransportTimeout)):
            self._healthy = False

    def estimate_prefix_reuse(self, prompt_token_ids: list[int]) -> PrefixReuseHint:
        self._require_healthy()
        hints = tuple(
            client.request("estimate_prefix_reuse", tuple(prompt_token_ids))
            for client in self._clients
        )
        first = hints[0]
        if not isinstance(first, PrefixReuseHint) or any(hint != first for hint in hints[1:]):
            self._healthy = False
            raise MultiprocessExecutorUnhealthy(
                "replicated ranks disagree on prefix reuse estimate"
            )
        return first

    def prepare_request(self, request_id: str, prompt_token_ids: list[int]) -> int:
        self._require_healthy()
        try:
            counts = tuple(
                int(client.request("prepare_request", (request_id, tuple(prompt_token_ids))))
                for client in self._clients
            )
        except Exception as exc:
            self._mark_unhealthy_if_transport(exc)
            raise
        if len(set(counts)) != 1:
            self._healthy = False
            raise MultiprocessExecutorUnhealthy(
                "replicated ranks disagree on reusable prefix length"
            )
        return counts[0]

    def ensure_capacity(self, request_id: str, total_tokens: int) -> bool:
        self._require_healthy()
        try:
            outcomes = tuple(
                bool(client.request("ensure_capacity", (request_id, total_tokens)))
                for client in self._clients
            )
        except Exception as exc:
            self._mark_unhealthy_if_transport(exc)
            raise
        if len(set(outcomes)) != 1:
            self._healthy = False
            raise MultiprocessExecutorUnhealthy("replicated ranks disagree on KV capacity")
        return outcomes[0]

    def checkpoint(self, request_id: str, num_tokens: int) -> MultiprocessCacheCheckpoint:
        self._require_healthy()
        handles: list[CheckpointHandle] = []
        try:
            for client in self._clients:
                handle = client.request("checkpoint", (request_id, num_tokens))
                if not isinstance(handle, CheckpointHandle):
                    raise WorkerTransportError("worker returned invalid checkpoint handle")
                handles.append(handle)
        except Exception as exc:
            self._mark_unhealthy_if_transport(exc)
            for client, handle in zip(self._clients, handles):
                try:
                    client.request("rollback", (handle.checkpoint_id, handle.num_tokens))
                except Exception:
                    self._healthy = False
            raise
        return MultiprocessCacheCheckpoint(request_id, num_tokens, tuple(handles))

    def rollback(
        self,
        checkpoint: MultiprocessCacheCheckpoint,
        *,
        retain_tokens: int,
    ) -> None:
        errors: list[Exception] = []
        for client, handle in zip(self._clients, checkpoint.rank_handles, strict=True):
            try:
                client.request("rollback", (handle.checkpoint_id, retain_tokens))
            except Exception as exc:
                self._mark_unhealthy_if_transport(exc)
                errors.append(exc)
        if errors:
            self._healthy = False
            first = errors[0]
            if isinstance(first, WorkerProcessDied):
                raise first
            raise MultiprocessExecutorUnhealthy(
                f"rollback failed on {len(errors)} rank(s)"
            ) from first

    def commit_checkpoint(self, checkpoint: MultiprocessCacheCheckpoint) -> None:
        errors: list[Exception] = []
        for client, handle in zip(self._clients, checkpoint.rank_handles, strict=True):
            try:
                client.request("commit_checkpoint", handle.checkpoint_id)
            except Exception as exc:
                self._mark_unhealthy_if_transport(exc)
                errors.append(exc)
        if errors:
            self._healthy = False
            raise MultiprocessExecutorUnhealthy(
                f"checkpoint commit failed on {len(errors)} rank(s)"
            ) from errors[0]

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self._require_healthy()
        self.execute_calls += 1
        request_ids = [item.request_id for item in request.items]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("MultiprocessExecutor expects at most one item per request")

        checkpoints: list[MultiprocessCacheCheckpoint] = []
        try:
            for item in request.items:
                checkpoints.append(self.checkpoint(item.request_id, item.start_pos))

            rank_results: list[SerializableExecutionResult] = []
            for client in self._clients:
                result = client.request("execute", request)
                if not isinstance(result, SerializableExecutionResult):
                    raise WorkerTransportError("worker returned invalid execution result")
                rank_results.append(result)

            expected_ids = set(request_ids)
            if any(set(result.logits_by_request) != expected_ids for result in rank_results):
                raise MultiprocessExecutorUnhealthy(
                    "rank returned logits for the wrong request set"
                )
            gathered = {
                request_id: self.collective.all_gather(
                    tuple(
                        result.logits_by_request[request_id].to_tensor()
                        for result in rank_results
                    ),
                    dim=-1,
                )
                for request_id in request_ids
            }
            for checkpoint in checkpoints:
                self.commit_checkpoint(checkpoint)
            return ExecutionResult(logits_by_request=gathered)
        except Exception as exc:
            self._mark_unhealthy_if_transport(exc)
            rollback_errors: list[Exception] = []
            for checkpoint in checkpoints:
                try:
                    self.rollback(checkpoint, retain_tokens=checkpoint.num_tokens)
                except Exception as rollback_exc:
                    rollback_errors.append(rollback_exc)
            if isinstance(exc, WorkerProcessDied):
                raise
            if rollback_errors:
                self._healthy = False
                raise MultiprocessExecutorUnhealthy(
                    "ordinary execution failed and rollback could not be proven complete"
                ) from exc
            raise

    def execute_speculative(
        self,
        request: SpeculativeExecutionRequest,
    ) -> SpeculativeExecutionResult:
        self._require_healthy()
        self.speculative_execute_calls += 1
        try:
            rank_results: list[SerializableSpeculativeResult] = []
            for client in self._clients:
                result = client.request("execute_speculative", request)
                if not isinstance(result, SerializableSpeculativeResult):
                    raise WorkerTransportError("worker returned invalid speculative result")
                rank_results.append(result)
        except Exception as exc:
            self._mark_unhealthy_if_transport(exc)
            raise

        if any(result.request_id != request.request_id for result in rank_results):
            self._healthy = False
            raise MultiprocessExecutorUnhealthy(
                "rank returned speculative result for wrong request"
            )
        trace_lengths = {len(result.logits_trace) for result in rank_results}
        if trace_lengths != {len(request.token_ids)}:
            self._healthy = False
            raise MultiprocessExecutorUnhealthy(
                "ranks returned inconsistent speculative trace lengths"
            )
        logits_trace = tuple(
            self.collective.all_gather(
                tuple(result.logits_trace[index].to_tensor() for result in rank_results),
                dim=-1,
            )
            for index in range(len(request.token_ids))
        )
        return SpeculativeExecutionResult(request.request_id, logits_trace)

    def commit_work(
        self,
        request_id: str,
        *,
        start_pos: int,
        token_ids: tuple[int, ...],
        prompt_length: int,
    ) -> None:
        self._require_healthy()
        payload = (request_id, start_pos, token_ids, prompt_length)
        try:
            for client in self._clients:
                client.request("commit_work", payload)
        except Exception as exc:
            self._mark_unhealthy_if_transport(exc)
            self._healthy = False
            raise

    def free(self, request_id: str) -> None:
        errors: list[Exception] = []
        for client in self._clients:
            if not client.healthy:
                continue
            try:
                client.request("free", request_id)
            except Exception as exc:
                self._mark_unhealthy_if_transport(exc)
                errors.append(exc)
        if errors:
            self._healthy = False
            raise MultiprocessExecutorUnhealthy(
                f"free failed on {len(errors)} rank(s)"
            ) from errors[0]

    def cache_stats(self) -> dict[str, int | float]:
        self._require_healthy()
        stats = tuple(client.request("cache_stats") for client in self._clients)
        first = stats[0]
        if not isinstance(first, dict) or any(stat != first for stat in stats[1:]):
            self._healthy = False
            raise MultiprocessExecutorUnhealthy(
                "replicated rank cache statistics diverged"
            )
        return {**first, "world_size": self.world_size}

    def transport_snapshot(self) -> tuple[TransportWorkerSnapshot, ...]:
        return tuple(client.snapshot() for client in self._clients)

    def shutdown(self) -> None:
        if self._closed:
            return
        for client in self._clients:
            client.shutdown()
        self._closed = True
        self._healthy = False

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

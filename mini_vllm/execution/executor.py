from __future__ import annotations

from abc import ABC, abstractmethod

from mini_vllm.cache.base import CacheCheckpoint, PrefixReuseHint
from mini_vllm.execution.types import (
    ExecutionRequest,
    ExecutionResult,
    SpeculativeExecutionRequest,
    SpeculativeExecutionResult,
)
from mini_vllm.execution.worker import Worker


class Executor(ABC):
    """Transport/orchestration contract between EngineCore and execution."""

    @abstractmethod
    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        ...

    @abstractmethod
    def execute_speculative(
        self, request: SpeculativeExecutionRequest
    ) -> SpeculativeExecutionResult:
        ...

    def estimate_prefix_reuse(self, prompt_token_ids: list[int]) -> PrefixReuseHint:
        """Policy-safe cache estimate; custom executors may keep the zero default."""
        return PrefixReuseHint(0, 0, len(prompt_token_ids))

    @abstractmethod
    def prepare_request(self, request_id: str, prompt_token_ids: list[int]) -> int:
        ...

    @abstractmethod
    def ensure_capacity(self, request_id: str, total_tokens: int) -> bool:
        ...

    @abstractmethod
    def commit_work(
        self,
        request_id: str,
        *,
        start_pos: int,
        token_ids: tuple[int, ...],
        prompt_length: int,
    ) -> None:
        ...

    @abstractmethod
    def checkpoint(self, request_id: str, num_tokens: int) -> CacheCheckpoint:
        ...

    @abstractmethod
    def rollback(self, checkpoint: CacheCheckpoint, *, retain_tokens: int) -> None:
        ...

    @abstractmethod
    def commit_checkpoint(self, checkpoint: CacheCheckpoint) -> None:
        ...

    @abstractmethod
    def free(self, request_id: str) -> None:
        ...

    @abstractmethod
    def cache_stats(self) -> dict[str, int | float]:
        ...


class LocalExecutor(Executor):
    """Single-process reference transport.

    The method boundary is intentionally the same shape a future remote or
    multiprocess executor can implement; only the transport is local.
    """

    def __init__(self, worker: Worker) -> None:
        self.worker = worker
        self.execute_calls = 0
        self.speculative_execute_calls = 0

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.execute_calls += 1
        return self.worker.execute(request)

    def execute_speculative(
        self, request: SpeculativeExecutionRequest
    ) -> SpeculativeExecutionResult:
        self.speculative_execute_calls += 1
        return self.worker.execute_speculative(request)

    def estimate_prefix_reuse(self, prompt_token_ids: list[int]) -> PrefixReuseHint:
        return self.worker.estimate_prefix_reuse(prompt_token_ids)

    def prepare_request(self, request_id: str, prompt_token_ids: list[int]) -> int:
        return self.worker.prepare_request(request_id, prompt_token_ids)

    def ensure_capacity(self, request_id: str, total_tokens: int) -> bool:
        return self.worker.ensure_capacity(request_id, total_tokens)

    def commit_work(
        self,
        request_id: str,
        *,
        start_pos: int,
        token_ids: tuple[int, ...],
        prompt_length: int,
    ) -> None:
        self.worker.commit_work(
            request_id,
            start_pos=start_pos,
            token_ids=token_ids,
            prompt_length=prompt_length,
        )

    def checkpoint(self, request_id: str, num_tokens: int) -> CacheCheckpoint:
        return self.worker.checkpoint(request_id, num_tokens)

    def rollback(self, checkpoint: CacheCheckpoint, *, retain_tokens: int) -> None:
        self.worker.rollback(checkpoint, retain_tokens=retain_tokens)

    def commit_checkpoint(self, checkpoint: CacheCheckpoint) -> None:
        self.worker.commit_checkpoint(checkpoint)

    def free(self, request_id: str) -> None:
        self.worker.free(request_id)

    def cache_stats(self) -> dict[str, int | float]:
        return self.worker.cache_stats()

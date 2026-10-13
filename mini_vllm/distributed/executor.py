from __future__ import annotations

from mini_vllm.distributed.worker_group import DistributedCacheCheckpoint, WorkerGroup
from mini_vllm.execution.executor import Executor
from mini_vllm.execution.types import (
    ExecutionRequest,
    ExecutionResult,
    SpeculativeExecutionRequest,
    SpeculativeExecutionResult,
)


class DistributedExecutor(Executor):
    """Single-process multi-rank executor using an explicit WorkerGroup."""

    def __init__(self, worker_group: WorkerGroup) -> None:
        self.worker_group = worker_group

    @property
    def world_size(self) -> int:
        return self.worker_group.world_size

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        return self.worker_group.execute(request)

    def execute_speculative(
        self, request: SpeculativeExecutionRequest
    ) -> SpeculativeExecutionResult:
        return self.worker_group.execute_speculative(request)

    def prepare_request(self, request_id: str, prompt_token_ids: list[int]) -> int:
        return self.worker_group.prepare_request(request_id, prompt_token_ids)

    def ensure_capacity(self, request_id: str, total_tokens: int) -> bool:
        return self.worker_group.ensure_capacity(request_id, total_tokens)

    def commit_work(
        self,
        request_id: str,
        *,
        start_pos: int,
        token_ids: tuple[int, ...],
        prompt_length: int,
    ) -> None:
        self.worker_group.commit_work(
            request_id,
            start_pos=start_pos,
            token_ids=token_ids,
            prompt_length=prompt_length,
        )

    def checkpoint(
        self, request_id: str, num_tokens: int
    ) -> DistributedCacheCheckpoint:
        return self.worker_group.checkpoint(request_id, num_tokens)

    def rollback(
        self,
        checkpoint: DistributedCacheCheckpoint,
        *,
        retain_tokens: int,
    ) -> None:
        self.worker_group.rollback(checkpoint, retain_tokens=retain_tokens)

    def commit_checkpoint(self, checkpoint: DistributedCacheCheckpoint) -> None:
        self.worker_group.commit_checkpoint(checkpoint)

    def free(self, request_id: str) -> None:
        self.worker_group.free(request_id)

    def cache_stats(self) -> dict[str, int | float]:
        return self.worker_group.cache_stats()

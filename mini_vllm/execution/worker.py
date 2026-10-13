from __future__ import annotations

from mini_vllm.cache.base import CacheCheckpoint, KVCacheBackend
from mini_vllm.execution.types import (
    ExecutionRequest,
    ExecutionResult,
    SpeculativeExecutionRequest,
    SpeculativeExecutionResult,
)
from mini_vllm.model.runner import ModelRunner


class Worker:
    """Execution mechanism only: model runner + physical KV backend.

    A Worker intentionally owns no request queues, Scheduler, admission order,
    victim policy, or logical finish decisions.
    """

    def __init__(self, *, model_runner: ModelRunner, cache: KVCacheBackend) -> None:
        self.model_runner = model_runner
        self.cache = cache

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        logits = self.model_runner.execute(list(request.items), self.cache)
        return ExecutionResult(logits_by_request=logits)

    def execute_speculative(
        self, request: SpeculativeExecutionRequest
    ) -> SpeculativeExecutionResult:
        logits_trace = self.model_runner.execute_speculative(
            request_id=request.request_id,
            token_ids=request.token_ids,
            start_pos=request.start_pos,
            cache=self.cache,
        )
        return SpeculativeExecutionResult(
            request_id=request.request_id,
            logits_trace=logits_trace,
        )

    def prepare_request(self, request_id: str, prompt_token_ids: list[int]) -> int:
        return self.cache.prepare_request(request_id, prompt_token_ids)

    def ensure_capacity(self, request_id: str, total_tokens: int) -> bool:
        return self.cache.ensure_capacity(request_id, total_tokens)

    def commit_work(
        self,
        request_id: str,
        *,
        start_pos: int,
        token_ids: tuple[int, ...],
        prompt_length: int,
    ) -> None:
        self.cache.commit_work(
            request_id,
            start_pos=start_pos,
            token_ids=token_ids,
            prompt_length=prompt_length,
        )

    def checkpoint(self, request_id: str, num_tokens: int) -> CacheCheckpoint:
        return self.cache.checkpoint(request_id, num_tokens)

    def rollback(self, checkpoint: CacheCheckpoint, *, retain_tokens: int) -> None:
        self.cache.rollback(checkpoint, retain_tokens=retain_tokens)

    def commit_checkpoint(self, checkpoint: CacheCheckpoint) -> None:
        self.cache.commit_checkpoint(checkpoint)

    def free(self, request_id: str) -> None:
        self.cache.free(request_id)

    def cache_stats(self) -> dict[str, int | float]:
        return self.cache.stats()

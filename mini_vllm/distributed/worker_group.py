from __future__ import annotations

from dataclasses import dataclass

from mini_vllm.cache.base import CacheCheckpoint
from mini_vllm.distributed.collective import ReferenceCollective
from mini_vllm.distributed.rank import RankConfig
from mini_vllm.execution.types import (
    ExecutionRequest,
    ExecutionResult,
    SpeculativeExecutionRequest,
    SpeculativeExecutionResult,
)
from mini_vllm.execution.worker import Worker


@dataclass(frozen=True, slots=True)
class RankedWorker:
    rank_config: RankConfig
    worker: Worker


@dataclass(frozen=True, slots=True)
class DistributedCacheCheckpoint:
    request_id: str
    num_tokens: int
    rank_checkpoints: tuple[CacheCheckpoint, ...]


class WorkerGroup:
    """Ordered logical ranks executing one request in a single process.

    Transformer trunk and KV are intentionally replicated in v0.9. Only the
    vocabulary projection is sharded. This keeps the first distributed version
    focused on rank topology and collective tensor semantics.
    """

    def __init__(
        self,
        ranked_workers: tuple[RankedWorker, ...],
        *,
        collective: ReferenceCollective | None = None,
    ) -> None:
        if not ranked_workers:
            raise ValueError("WorkerGroup requires at least one worker")
        world_sizes = {entry.rank_config.world_size for entry in ranked_workers}
        if len(world_sizes) != 1:
            raise ValueError("all workers must agree on world_size")
        world_size = next(iter(world_sizes))
        if len(ranked_workers) != world_size:
            raise ValueError("WorkerGroup must contain exactly world_size workers")
        ordered = tuple(sorted(ranked_workers, key=lambda entry: entry.rank_config.rank))
        if tuple(entry.rank_config.rank for entry in ordered) != tuple(range(world_size)):
            raise ValueError("WorkerGroup ranks must be exactly 0..world_size-1")
        self.ranked_workers = ordered
        self.world_size = world_size
        self.collective = collective or ReferenceCollective()
        self.execute_calls = 0
        self.speculative_execute_calls = 0

    @property
    def workers(self) -> tuple[Worker, ...]:
        return tuple(entry.worker for entry in self.ranked_workers)

    def _checkpoint_request(
        self,
        request_id: str,
        num_tokens: int,
    ) -> DistributedCacheCheckpoint:
        checkpoints = tuple(
            worker.checkpoint(request_id, num_tokens) for worker in self.workers
        )
        if any(
            checkpoint.request_id != request_id or checkpoint.num_tokens != num_tokens
            for checkpoint in checkpoints
        ):
            raise RuntimeError("ranks returned inconsistent cache checkpoints")
        return DistributedCacheCheckpoint(
            request_id=request_id,
            num_tokens=num_tokens,
            rank_checkpoints=checkpoints,
        )

    def _rollback_checkpoint(
        self,
        checkpoint: DistributedCacheCheckpoint,
        *,
        retain_tokens: int,
    ) -> None:
        for worker, rank_checkpoint in zip(
            self.workers, checkpoint.rank_checkpoints, strict=True
        ):
            worker.rollback(rank_checkpoint, retain_tokens=retain_tokens)

    def _commit_checkpoint(self, checkpoint: DistributedCacheCheckpoint) -> None:
        for worker, rank_checkpoint in zip(
            self.workers, checkpoint.rank_checkpoints, strict=True
        ):
            worker.commit_checkpoint(rank_checkpoint)

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.execute_calls += 1
        request_ids = [item.request_id for item in request.items]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("WorkerGroup expects at most one item per request")

        checkpoints = {
            item.request_id: self._checkpoint_request(item.request_id, item.start_pos)
            for item in request.items
        }
        try:
            rank_results = tuple(worker.execute(request) for worker in self.workers)
            expected_ids = set(request_ids)
            for result in rank_results:
                if set(result.logits_by_request) != expected_ids:
                    raise RuntimeError("rank returned logits for the wrong request set")

            gathered = {
                request_id: self.collective.all_gather(
                    tuple(
                        result.logits_by_request[request_id]
                        for result in rank_results
                    ),
                    dim=-1,
                )
                for request_id in request_ids
            }
            for checkpoint in checkpoints.values():
                self._commit_checkpoint(checkpoint)
            return ExecutionResult(logits_by_request=gathered)
        except Exception:
            for checkpoint in checkpoints.values():
                self._rollback_checkpoint(
                    checkpoint,
                    retain_tokens=checkpoint.num_tokens,
                )
            raise

    def execute_speculative(
        self,
        request: SpeculativeExecutionRequest,
    ) -> SpeculativeExecutionResult:
        self.speculative_execute_calls += 1
        rank_results = tuple(
            worker.execute_speculative(request) for worker in self.workers
        )
        trace_lengths = {len(result.logits_trace) for result in rank_results}
        if trace_lengths != {len(request.token_ids)}:
            raise RuntimeError("ranks returned inconsistent speculative trace lengths")
        if any(result.request_id != request.request_id for result in rank_results):
            raise RuntimeError("rank returned speculative result for wrong request")

        logits_trace = tuple(
            self.collective.all_gather(
                tuple(result.logits_trace[index] for result in rank_results),
                dim=-1,
            )
            for index in range(len(request.token_ids))
        )
        return SpeculativeExecutionResult(
            request_id=request.request_id,
            logits_trace=logits_trace,
        )

    def prepare_request(self, request_id: str, prompt_token_ids: list[int]) -> int:
        counts = tuple(
            worker.prepare_request(request_id, prompt_token_ids)
            for worker in self.workers
        )
        if len(set(counts)) != 1:
            raise RuntimeError("replicated ranks disagree on reusable prefix length")
        return counts[0]

    def ensure_capacity(self, request_id: str, total_tokens: int) -> bool:
        outcomes = tuple(
            worker.ensure_capacity(request_id, total_tokens) for worker in self.workers
        )
        if len(set(outcomes)) != 1:
            raise RuntimeError("replicated ranks disagree on KV capacity")
        return outcomes[0]

    def commit_work(
        self,
        request_id: str,
        *,
        start_pos: int,
        token_ids: tuple[int, ...],
        prompt_length: int,
    ) -> None:
        for worker in self.workers:
            worker.commit_work(
                request_id,
                start_pos=start_pos,
                token_ids=token_ids,
                prompt_length=prompt_length,
            )

    def checkpoint(
        self,
        request_id: str,
        num_tokens: int,
    ) -> DistributedCacheCheckpoint:
        return self._checkpoint_request(request_id, num_tokens)

    def rollback(
        self,
        checkpoint: DistributedCacheCheckpoint,
        *,
        retain_tokens: int,
    ) -> None:
        self._rollback_checkpoint(checkpoint, retain_tokens=retain_tokens)

    def commit_checkpoint(self, checkpoint: DistributedCacheCheckpoint) -> None:
        self._commit_checkpoint(checkpoint)

    def free(self, request_id: str) -> None:
        for worker in self.workers:
            worker.free(request_id)

    def cache_stats(self) -> dict[str, int | float]:
        stats = tuple(worker.cache_stats() for worker in self.workers)
        first = stats[0]
        if any(stat != first for stat in stats[1:]):
            raise RuntimeError("replicated rank cache statistics diverged")
        return {**first, "world_size": self.world_size}

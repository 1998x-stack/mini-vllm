from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mini_vllm.distributed.collective import ReferenceCollective
from mini_vllm.distributed.rank import RankConfig


def shard_range(total_size: int, config: RankConfig) -> tuple[int, int]:
    """Balanced contiguous partition supporting uneven final dimensions."""
    if total_size <= 0:
        raise ValueError("total_size must be > 0")
    base, remainder = divmod(total_size, config.world_size)
    size = base + (1 if config.rank < remainder else 0)
    start = config.rank * base + min(config.rank, remainder)
    return start, start + size


@dataclass(frozen=True, slots=True)
class VocabShard:
    rank: int
    world_size: int
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


class VocabParallelLMHead:
    """Output-dimension shard of an LM head.

    Every rank sees the same hidden vector but stores/computes only a contiguous
    slice of vocabulary rows. A gather in rank order reconstructs full logits.
    """

    def __init__(self, weight: torch.Tensor, rank_config: RankConfig) -> None:
        if weight.ndim != 2:
            raise ValueError("LM-head weight must be 2-D [vocab, hidden]")
        start, end = shard_range(weight.shape[0], rank_config)
        if start == end:
            raise ValueError("world_size cannot exceed vocabulary size")
        self.rank_config = rank_config
        self.weight = weight
        self.shard = VocabShard(
            rank=rank_config.rank,
            world_size=rank_config.world_size,
            start=start,
            end=end,
        )

    @property
    def local_weight(self) -> torch.Tensor:
        return self.weight[self.shard.start : self.shard.end]

    def local_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.local_weight)


class ReferenceTensorParallel:
    """Small correctness oracle for sharded linear algebra."""

    def __init__(self, collective: ReferenceCollective | None = None) -> None:
        self.collective = collective or ReferenceCollective()

    def column_parallel_linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        world_size: int,
    ) -> torch.Tensor:
        locals_: list[torch.Tensor] = []
        for rank in range(world_size):
            head = VocabParallelLMHead(weight, RankConfig(rank, world_size))
            locals_.append(head.local_logits(x))
        return self.collective.all_gather(tuple(locals_), dim=-1)

    def row_parallel_linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        world_size: int,
    ) -> torch.Tensor:
        if weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
            raise ValueError("row-parallel shapes are incompatible")
        locals_: list[torch.Tensor] = []
        for rank in range(world_size):
            config = RankConfig(rank, world_size)
            start, end = shard_range(weight.shape[1], config)
            if start == end:
                raise ValueError("world_size cannot exceed input dimension")
            locals_.append(F.linear(x[..., start:end], weight[:, start:end]))
        return self.collective.all_reduce_sum(tuple(locals_))[0]

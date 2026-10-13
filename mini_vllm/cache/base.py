from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class CacheCheckpoint:
    """Stable request-local frontier before temporary KV mutation."""

    request_id: str
    num_tokens: int
    block_table: tuple[int, ...]


class KVCacheBackend(ABC):
    """Storage contract consumed by the model runner."""

    def prepare_request(self, request_id: str, prompt_token_ids: list[int]) -> int:
        return 0

    def commit_work(
        self,
        request_id: str,
        *,
        start_pos: int,
        token_ids: tuple[int, ...],
        prompt_length: int,
    ) -> None:
        return None

    def checkpoint(self, request_id: str, num_tokens: int) -> CacheCheckpoint:
        raise NotImplementedError(f"{type(self).__name__} does not support checkpoints")

    def rollback(self, checkpoint: CacheCheckpoint, *, retain_tokens: int) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support rollback")

    def commit_checkpoint(self, checkpoint: CacheCheckpoint) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support checkpoints")

    def read_prefix_vectorized(
        self,
        layer_idx: int,
        request_id: str,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optional optimized prefix read used by v0.12 PagedAttention."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support vectorized prefix reads"
        )

    @abstractmethod
    def ensure_capacity(self, request_id: str, total_tokens: int) -> bool:
        """Ensure slots [0, total_tokens) exist for request_id."""

    @abstractmethod
    def write(
        self,
        layer_idx: int,
        request_id: str,
        position: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write one token's K/V, shape [n_kv_heads, head_dim]."""

    @abstractmethod
    def read_prefix(
        self,
        layer_idx: int,
        request_id: str,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return K/V for positions [0, length), shape [L, H_kv, D]."""

    @abstractmethod
    def free(self, request_id: str) -> None:
        """Release every cache slot owned by a request."""

    @abstractmethod
    def stats(self) -> dict[str, int | float]:
        """Return small inspectable metrics for traces/tests."""

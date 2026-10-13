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


@dataclass(frozen=True, slots=True)
class PrefixReuseHint:
    """Policy-safe, side-effect-free estimate of reusable prompt work.

    This is deliberately not a reservation or ownership handle.  It contains
    no physical block identities and may become stale before actual admission.
    """

    reusable_tokens: int
    reusable_full_blocks: int
    prompt_tokens: int

    def __post_init__(self) -> None:
        if self.reusable_tokens < 0 or self.reusable_full_blocks < 0:
            raise ValueError("prefix reuse counts must be non-negative")
        if self.prompt_tokens < 0 or self.reusable_tokens > self.prompt_tokens:
            raise ValueError("invalid prefix reuse hint")


class KVCacheBackend(ABC):
    """Storage contract consumed by the model runner."""

    def estimate_prefix_reuse(self, prompt_token_ids: list[int]) -> PrefixReuseHint:
        """Return a side-effect-free reuse estimate for scheduler policy."""
        return PrefixReuseHint(
            reusable_tokens=0,
            reusable_full_blocks=0,
            prompt_tokens=len(prompt_token_ids),
        )

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

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
    """Storage contract consumed by the model runner.

    Production vLLM has a much richer KV cache manager. This interface keeps
    the essential operations visible and makes cache implementations swappable.

    v0.5 adds two optional lifecycle hooks:

    - ``prepare_request`` may attach an already-computed immutable prefix and
      returns how many prompt tokens can be skipped.
    - ``commit_work`` observes completed model work after every layer has
      written K/V, allowing a cache backend to publish newly immutable blocks.

    Older cache backends intentionally inherit no-op implementations so the
    historical v0.1-v0.4 runtime semantics remain unchanged.
    """

    def prepare_request(self, request_id: str, prompt_token_ids: list[int]) -> int:
        """Attach reusable cache state and return cached prompt-token count."""
        return 0

    def commit_work(
        self,
        request_id: str,
        *,
        start_pos: int,
        token_ids: tuple[int, ...],
        prompt_length: int,
    ) -> None:
        """Publish cache metadata after a WorkItem has fully executed."""
        return None

    def checkpoint(self, request_id: str, num_tokens: int) -> CacheCheckpoint:
        """Capture a stable frontier before temporary/speculative writes."""
        raise NotImplementedError(f"{type(self).__name__} does not support checkpoints")

    def rollback(self, checkpoint: CacheCheckpoint, *, retain_tokens: int) -> None:
        """Discard request-private KV beyond ``retain_tokens``."""
        raise NotImplementedError(f"{type(self).__name__} does not support rollback")

    def commit_checkpoint(self, checkpoint: CacheCheckpoint) -> None:
        """Validate/close a successful transaction. Default backends reject it."""
        raise NotImplementedError(f"{type(self).__name__} does not support checkpoints")

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

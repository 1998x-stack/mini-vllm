from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class KVCacheBackend(ABC):
    """Storage contract consumed by the model runner."""

    @abstractmethod
    def ensure_capacity(self, request_id: str, total_tokens: int) -> bool:
        """Ensure slots [0, total_tokens) exist for request_id."""

    @abstractmethod
    def write(self, layer_idx: int, request_id: str, position: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Write one token's K/V, shape [n_kv_heads, head_dim]."""

    @abstractmethod
    def read_prefix(self, layer_idx: int, request_id: str, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return K/V for positions [0, length), shape [L, H_kv, D]."""

    @abstractmethod
    def free(self, request_id: str) -> None:
        """Release every cache slot owned by a request."""

    @abstractmethod
    def stats(self) -> dict[str, int | float]:
        """Return small inspectable metrics for traces/tests."""

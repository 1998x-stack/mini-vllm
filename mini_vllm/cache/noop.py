from __future__ import annotations

import torch

from mini_vllm.cache.base import KVCacheBackend


class NoOpKVCache(KVCacheBackend):
    """Testing-only cache used when a scripted runner does not need K/V."""

    def ensure_capacity(self, request_id: str, total_tokens: int) -> bool:
        return True

    def write(self, layer_idx: int, request_id: str, position: int, key: torch.Tensor, value: torch.Tensor) -> None:
        return None

    def read_prefix(self, layer_idx: int, request_id: str, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("NoOpKVCache does not store tensors")

    def free(self, request_id: str) -> None:
        return None

    def stats(self) -> dict[str, int | float]:
        return {}

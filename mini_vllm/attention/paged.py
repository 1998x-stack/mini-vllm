from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import torch

from mini_vllm.cache.base import KVCacheBackend


@dataclass(frozen=True, slots=True)
class PagedAttentionConfig:
    backend: str = "vectorized"
    fallback_to_reference: bool = True

    def __post_init__(self) -> None:
        if self.backend not in {"reference", "vectorized"}:
            raise ValueError("backend must be 'reference' or 'vectorized'")


@dataclass(slots=True)
class PagedAttentionStats:
    total_calls: int = 0
    reference_calls: int = 0
    vectorized_calls: int = 0
    fallbacks: int = 0


@dataclass(frozen=True, slots=True)
class PagedAttentionEvent:
    backend: str
    mode: str
    length: int
    fallback_reason: str | None = None


class PagedAttentionBackend(ABC):
    """Attention context computation over externally owned KV storage."""

    def __init__(self, config: PagedAttentionConfig | None = None) -> None:
        self.config = config or PagedAttentionConfig()
        self.stats = PagedAttentionStats()
        self.events: list[PagedAttentionEvent] = []

    @abstractmethod
    def _read_prefix(
        self,
        cache: KVCacheBackend,
        *,
        layer_idx: int,
        request_id: str,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @staticmethod
    def _expand_kv(
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        n_heads: int,
        n_kv_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if n_heads == n_kv_heads:
            return keys, values
        repeats = n_heads // n_kv_heads
        return (
            keys.repeat_interleave(repeats, dim=1),
            values.repeat_interleave(repeats, dim=1),
        )

    def attend(
        self,
        q: torch.Tensor,
        *,
        cache: KVCacheBackend,
        layer_idx: int,
        request_id: str,
        length: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        if length <= 0:
            raise ValueError("attention length must be > 0")
        if q.shape != (n_heads, head_dim):
            raise ValueError("q shape does not match attention metadata")

        self.stats.total_calls += 1
        keys, values = self._read_prefix(
            cache,
            layer_idx=layer_idx,
            request_id=request_id,
            length=length,
        )
        if keys.shape != (length, n_kv_heads, head_dim):
            raise RuntimeError("cache returned unexpected key shape")
        if values.shape != keys.shape:
            raise RuntimeError("cache returned mismatched value shape")

        keys, values = self._expand_kv(
            keys,
            values,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
        )
        scores = torch.einsum("hd,lhd->hl", q, keys) / math.sqrt(head_dim)
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("hl,lhd->hd", probs, values)


class ReferencePagedAttention(PagedAttentionBackend):
    """Readable oracle using the cache backend's token-by-token prefix read."""

    def __init__(self, config: PagedAttentionConfig | None = None) -> None:
        super().__init__(config or PagedAttentionConfig(backend="reference"))

    def _read_prefix(
        self,
        cache: KVCacheBackend,
        *,
        layer_idx: int,
        request_id: str,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.stats.reference_calls += 1
        self.events.append(PagedAttentionEvent("reference", "reference", length))
        return cache.read_prefix(layer_idx, request_id, length)


class VectorizedPagedAttention(PagedAttentionBackend):
    """Vectorized physical-block gather with explicit reference fallback."""

    def __init__(self, config: PagedAttentionConfig | None = None) -> None:
        super().__init__(config or PagedAttentionConfig(backend="vectorized"))

    def _read_prefix(
        self,
        cache: KVCacheBackend,
        *,
        layer_idx: int,
        request_id: str,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            rows = cache.read_prefix_vectorized(layer_idx, request_id, length)
        except (AttributeError, NotImplementedError) as exc:
            if not self.config.fallback_to_reference:
                raise RuntimeError(
                    f"{type(cache).__name__} does not support vectorized paged reads"
                ) from exc
            self.stats.fallbacks += 1
            self.stats.reference_calls += 1
            self.events.append(
                PagedAttentionEvent(
                    "vectorized",
                    "fallback-reference",
                    length,
                    type(exc).__name__,
                )
            )
            return cache.read_prefix(layer_idx, request_id, length)

        self.stats.vectorized_calls += 1
        self.events.append(PagedAttentionEvent("vectorized", "vectorized", length))
        return rows


def build_paged_attention_backend(
    config: PagedAttentionConfig | None = None,
) -> PagedAttentionBackend:
    config = config or PagedAttentionConfig()
    if config.backend == "reference":
        return ReferencePagedAttention(config)
    if config.backend == "vectorized":
        return VectorizedPagedAttention(config)
    raise ValueError(f"unsupported paged attention backend: {config.backend}")

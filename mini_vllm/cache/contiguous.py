from __future__ import annotations

from dataclasses import dataclass

import torch

from mini_vllm.cache.base import KVCacheBackend


@dataclass(slots=True)
class _RequestSlab:
    keys: torch.Tensor
    values: torch.Tensor
    used_tokens: int = 0


class ContiguousKVCache(KVCacheBackend):
    """One fixed max-sequence slab per active request.

    A request that has cached only a few tokens still reserves max_model_len
    token slots, making the baseline memory waste visible.
    """

    def __init__(self, *, num_layers: int, max_model_len: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype = torch.float32, device: str = "cpu") -> None:
        self.num_layers = num_layers
        self.max_model_len = max_model_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self._slabs: dict[str, _RequestSlab] = {}

    def _allocate(self, request_id: str) -> None:
        shape = (self.num_layers, self.max_model_len, self.num_kv_heads, self.head_dim)
        self._slabs[request_id] = _RequestSlab(
            keys=torch.empty(shape, dtype=self.dtype, device=self.device),
            values=torch.empty(shape, dtype=self.dtype, device=self.device),
        )

    def ensure_capacity(self, request_id: str, total_tokens: int) -> bool:
        if total_tokens > self.max_model_len:
            return False
        if request_id not in self._slabs:
            self._allocate(request_id)
        return True

    def write(self, layer_idx: int, request_id: str, position: int, key: torch.Tensor, value: torch.Tensor) -> None:
        if request_id not in self._slabs:
            raise KeyError(f"request {request_id!r} has no allocated slab")
        slab = self._slabs[request_id]
        slab.keys[layer_idx, position].copy_(key)
        slab.values[layer_idx, position].copy_(value)
        slab.used_tokens = max(slab.used_tokens, position + 1)

    def read_prefix(self, layer_idx: int, request_id: str, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        slab = self._slabs[request_id]
        return slab.keys[layer_idx, :length], slab.values[layer_idx, :length]

    def free(self, request_id: str) -> None:
        self._slabs.pop(request_id, None)

    def stats(self) -> dict[str, int | float]:
        active = len(self._slabs)
        reserved = active * self.max_model_len
        used = sum(slab.used_tokens for slab in self._slabs.values())
        utilization = 0.0 if reserved == 0 else used / reserved
        return {"active_requests": active, "reserved_token_slots": reserved, "used_token_slots": used, "slot_utilization": utilization}

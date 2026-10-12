from __future__ import annotations

from collections import deque

import torch

from mini_vllm.cache.base import KVCacheBackend


class PagedKVCache(KVCacheBackend):
    """Fixed-size physical KV blocks plus per-request block tables."""

    def __init__(self, *, num_layers: int, num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype = torch.float32, device: str = "cpu") -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.keys = torch.empty(shape, dtype=dtype, device=device)
        self.values = torch.empty(shape, dtype=dtype, device=device)
        self._free_blocks: deque[int] = deque(range(num_blocks))
        self._block_tables: dict[str, list[int]] = {}
        self._used_tokens: dict[str, int] = {}

    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    def required_new_blocks(self, request_id: str, total_tokens: int) -> int:
        required = self._ceil_div(total_tokens, self.block_size)
        allocated = len(self._block_tables.get(request_id, []))
        return max(0, required - allocated)

    def ensure_capacity(self, request_id: str, total_tokens: int) -> bool:
        needed = self.required_new_blocks(request_id, total_tokens)
        if needed > len(self._free_blocks):
            return False
        table = self._block_tables.setdefault(request_id, [])
        for _ in range(needed):
            table.append(self._free_blocks.popleft())
        return True

    def slot_mapping(self, request_id: str, position: int) -> tuple[int, int]:
        logical_block = position // self.block_size
        block_offset = position % self.block_size
        table = self._block_tables[request_id]
        if logical_block >= len(table):
            raise IndexError("position has not been allocated")
        return table[logical_block], block_offset

    def block_table(self, request_id: str) -> tuple[int, ...]:
        return tuple(self._block_tables.get(request_id, ()))

    def write(self, layer_idx: int, request_id: str, position: int, key: torch.Tensor, value: torch.Tensor) -> None:
        physical_block, offset = self.slot_mapping(request_id, position)
        self.keys[layer_idx, physical_block, offset].copy_(key)
        self.values[layer_idx, physical_block, offset].copy_(value)
        self._used_tokens[request_id] = max(self._used_tokens.get(request_id, 0), position + 1)

    def read_prefix(self, layer_idx: int, request_id: str, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        key_rows: list[torch.Tensor] = []
        value_rows: list[torch.Tensor] = []
        for position in range(length):
            physical_block, offset = self.slot_mapping(request_id, position)
            key_rows.append(self.keys[layer_idx, physical_block, offset])
            value_rows.append(self.values[layer_idx, physical_block, offset])
        return torch.stack(key_rows, dim=0), torch.stack(value_rows, dim=0)

    def free(self, request_id: str) -> None:
        self._free_blocks.extend(self._block_tables.pop(request_id, []))
        self._used_tokens.pop(request_id, None)

    def stats(self) -> dict[str, int | float]:
        used_blocks = self.num_blocks - len(self._free_blocks)
        reserved_slots = used_blocks * self.block_size
        used_slots = sum(self._used_tokens.values())
        utilization = 0.0 if reserved_slots == 0 else used_slots / reserved_slots
        return {"total_blocks": self.num_blocks, "free_blocks": len(self._free_blocks), "used_blocks": used_blocks, "reserved_token_slots": reserved_slots, "used_token_slots": used_slots, "slot_utilization": utilization}

from __future__ import annotations

from mini_vllm.cache.base import CacheCheckpoint
from mini_vllm.cache.prefix import PrefixCachingPagedKVCache


class TransactionalPrefixCachingPagedKVCache(PrefixCachingPagedKVCache):
    """v0.7 cache: prefix sharing plus request-private checkpoint/rollback.

    A checkpoint protects the already-visible block-table prefix. Speculative
    decoding is append-only: temporary writes may extend a private tail, but
    rollback may never mutate a shared immutable prefix.
    """

    def checkpoint(self, request_id: str, num_tokens: int) -> CacheCheckpoint:
        if request_id not in self._block_tables:
            raise KeyError(f"request {request_id!r} has no cache state")
        used_tokens = self._used_tokens.get(request_id, 0)
        if num_tokens != used_tokens:
            raise ValueError(
                f"checkpoint frontier mismatch: request has {used_tokens} computed tokens, "
                f"caller supplied {num_tokens}"
            )
        return CacheCheckpoint(
            request_id=request_id,
            num_tokens=num_tokens,
            block_table=tuple(self._block_tables[request_id]),
        )

    def rollback(self, checkpoint: CacheCheckpoint, *, retain_tokens: int) -> None:
        request_id = checkpoint.request_id
        if request_id not in self._block_tables:
            raise RuntimeError("cannot rollback released request cache")
        current_tokens = self._used_tokens.get(request_id, 0)
        if retain_tokens < checkpoint.num_tokens:
            raise ValueError("rollback cannot cross the checkpoint frontier")
        if retain_tokens > current_tokens:
            raise ValueError("rollback target exceeds computed cache frontier")

        table = self._block_tables[request_id]
        if tuple(table[: len(checkpoint.block_table)]) != checkpoint.block_table:
            raise RuntimeError("stable block-table prefix changed after checkpoint")

        required_blocks = self._ceil_div(retain_tokens, self.block_size) if retain_tokens else 0
        while len(table) > required_blocks:
            logical_block = len(table) - 1
            physical_block = table.pop()

            # Speculation may use a block that was pre-reserved by the normal
            # decode WorkItem before the checkpoint. It is still safe to drop
            # only when it is private and no longer addressable at retain_tokens.
            if logical_block < len(checkpoint.block_table):
                if physical_block != checkpoint.block_table[logical_block]:
                    raise RuntimeError("checkpoint block identity changed")
            if self._is_cached_block(physical_block):
                raise RuntimeError("rollback attempted to free cached prefix block")
            if self._block_refcounts[physical_block] != 1:
                raise RuntimeError("rollback encountered shared mutable block")

            self._block_refcounts[physical_block] = 0
            self._free_blocks.append(physical_block)

        self._used_tokens[request_id] = retain_tokens

    def commit_checkpoint(self, checkpoint: CacheCheckpoint) -> None:
        request_id = checkpoint.request_id
        if request_id not in self._block_tables:
            raise RuntimeError("cannot commit released request cache")
        table = self._block_tables[request_id]
        if tuple(table[: len(checkpoint.block_table)]) != checkpoint.block_table:
            raise RuntimeError("stable block-table prefix changed during transaction")

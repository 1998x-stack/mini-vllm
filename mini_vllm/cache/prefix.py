from __future__ import annotations

from collections import OrderedDict
import hashlib
import struct

from mini_vllm.cache.paged import PagedKVCache


class PrefixCachingPagedKVCache(PagedKVCache):
    """Paged KV cache with content-addressed immutable prompt blocks.

    Educational policy:

    * only *full prompt blocks* may enter the prefix index;
    * prefix identity is chained, so an identical token block under a different
      preceding context does not alias incorrectly;
    * cached blocks are immutable and may be referenced by multiple requests;
    * a request's partial tail block is always private and writable;
    * at least one prompt token is intentionally left uncached so the model can
      produce the next-token logits (KV alone does not contain logits);
    * unreferenced cached blocks are retained until memory pressure evicts them
      in least-recently-used order.

    This is a readable reference design, not a production cache manager.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._block_refcounts: list[int] = [0 for _ in range(self.num_blocks)]
        self._prefix_index: dict[str, int] = {}
        self._cached_hash_by_block: dict[int, str] = {}
        self._cache_lru: OrderedDict[str, int] = OrderedDict()
        self._request_prompt_tokens: dict[str, tuple[int, ...]] = {}
        self._request_block_hashes: dict[str, tuple[str, ...]] = {}

        self._prefix_hit_blocks = 0
        self._prefix_miss_blocks = 0
        self._prefix_hit_tokens = 0
        self._prefix_evictions = 0

    @staticmethod
    def _chain_hash(previous_hash: str, block_tokens: tuple[int, ...]) -> str:
        digest = hashlib.sha256()
        digest.update(previous_hash.encode("ascii"))
        digest.update(b"|")
        for token_id in block_tokens:
            digest.update(struct.pack(">q", int(token_id)))
        return digest.hexdigest()

    def _compute_prompt_block_hashes(self, prompt_tokens: tuple[int, ...]) -> tuple[str, ...]:
        hashes: list[str] = []
        previous = ""
        num_full_blocks = len(prompt_tokens) // self.block_size
        for logical_block in range(num_full_blocks):
            start = logical_block * self.block_size
            end = start + self.block_size
            previous = self._chain_hash(previous, prompt_tokens[start:end])
            hashes.append(previous)
        return tuple(hashes)

    def _touch(self, block_hash: str) -> None:
        if block_hash in self._cache_lru:
            self._cache_lru.move_to_end(block_hash)

    def _is_cached_block(self, physical_block: int) -> bool:
        return physical_block in self._cached_hash_by_block

    def _release_private_block_if_unused(self, physical_block: int) -> None:
        if self._block_refcounts[physical_block] != 0:
            return
        if self._is_cached_block(physical_block):
            return
        self._free_blocks.append(physical_block)

    def _evict_one_unreferenced_cached_block(self) -> bool:
        for block_hash, physical_block in list(self._cache_lru.items()):
            if self._block_refcounts[physical_block] != 0:
                continue
            self._cache_lru.pop(block_hash, None)
            self._prefix_index.pop(block_hash, None)
            self._cached_hash_by_block.pop(physical_block, None)
            self._free_blocks.append(physical_block)
            self._prefix_evictions += 1
            return True
        return False

    def prepare_request(self, request_id: str, prompt_token_ids: list[int]) -> int:
        if request_id in self._block_tables:
            raise ValueError(f"request {request_id} already owns KV state")

        prompt_tokens = tuple(int(token) for token in prompt_token_ids)
        block_hashes = self._compute_prompt_block_hashes(prompt_tokens)
        self._request_prompt_tokens[request_id] = prompt_tokens
        self._request_block_hashes[request_id] = block_hashes
        table = self._block_tables.setdefault(request_id, [])

        # KV cache stores K/V, not logits. Even for an exact full-prompt cache
        # hit, leave at least one prompt token to execute so sampling has the
        # correct final-position logits.
        max_reusable_tokens = max(0, len(prompt_tokens) - 1)
        reusable_full_blocks = max_reusable_tokens // self.block_size

        cached_blocks = 0
        for logical_block in range(reusable_full_blocks):
            block_hash = block_hashes[logical_block]
            physical_block = self._prefix_index.get(block_hash)
            if physical_block is None:
                self._prefix_miss_blocks += 1
                break

            table.append(physical_block)
            self._block_refcounts[physical_block] += 1
            self._touch(block_hash)
            cached_blocks += 1
            self._prefix_hit_blocks += 1

        cached_tokens = cached_blocks * self.block_size
        self._prefix_hit_tokens += cached_tokens
        self._used_tokens[request_id] = cached_tokens
        return cached_tokens

    def ensure_capacity(self, request_id: str, total_tokens: int) -> bool:
        needed = self.required_new_blocks(request_id, total_tokens)
        while needed > len(self._free_blocks):
            if not self._evict_one_unreferenced_cached_block():
                return False

        table = self._block_tables.setdefault(request_id, [])
        for _ in range(needed):
            physical_block = self._free_blocks.popleft()
            if self._block_refcounts[physical_block] != 0:
                raise RuntimeError("free-list block still has references")
            if self._is_cached_block(physical_block):
                raise RuntimeError("cached block appeared on free list")
            self._block_refcounts[physical_block] = 1
            table.append(physical_block)
        return True

    def commit_work(
        self,
        request_id: str,
        *,
        start_pos: int,
        token_ids: tuple[int, ...],
        prompt_length: int,
    ) -> None:
        if not token_ids:
            return
        if request_id not in self._block_tables:
            return

        committed_end = min(start_pos + len(token_ids), prompt_length)
        if committed_end <= 0:
            return

        # A full prompt block becomes immutable once its last prompt token has
        # executed. Publish every newly completed full block touched by this
        # WorkItem. If another request already published identical content,
        # canonicalize this request's block table onto the existing block.
        first_logical = start_pos // self.block_size
        last_complete_logical = committed_end // self.block_size - 1
        if last_complete_logical < first_logical:
            return

        block_hashes = self._request_block_hashes.get(request_id, ())
        table = self._block_tables[request_id]
        for logical_block in range(first_logical, last_complete_logical + 1):
            if logical_block >= len(block_hashes) or logical_block >= len(table):
                continue

            block_end = (logical_block + 1) * self.block_size
            if block_end > committed_end or block_end > prompt_length:
                continue

            block_hash = block_hashes[logical_block]
            current_block = table[logical_block]
            canonical = self._prefix_index.get(block_hash)

            if canonical is None:
                self._prefix_index[block_hash] = current_block
                self._cached_hash_by_block[current_block] = block_hash
                self._cache_lru[block_hash] = current_block
                continue

            self._touch(block_hash)
            if canonical == current_block:
                continue

            # Concurrent/cold duplicate: future requests should share the
            # canonical block. The just-computed duplicate can be released.
            self._block_refcounts[canonical] += 1
            table[logical_block] = canonical
            self._block_refcounts[current_block] -= 1
            if self._block_refcounts[current_block] < 0:
                raise RuntimeError("negative block refcount")
            self._release_private_block_if_unused(current_block)

    def block_refcount(self, physical_block: int) -> int:
        return self._block_refcounts[physical_block]

    def is_cached_block(self, physical_block: int) -> bool:
        return self._is_cached_block(physical_block)

    def free(self, request_id: str) -> None:
        table = self._block_tables.pop(request_id, [])
        for physical_block in table:
            self._block_refcounts[physical_block] -= 1
            if self._block_refcounts[physical_block] < 0:
                raise RuntimeError("negative block refcount")
            self._release_private_block_if_unused(physical_block)

        self._used_tokens.pop(request_id, None)
        self._request_prompt_tokens.pop(request_id, None)
        self._request_block_hashes.pop(request_id, None)

    def stats(self) -> dict[str, int | float]:
        base = super().stats()
        cached_blocks = len(self._prefix_index)
        shared_blocks = sum(refcount > 1 for refcount in self._block_refcounts)
        active_block_refs = sum(self._block_refcounts)
        base.update(
            {
                "cached_blocks": cached_blocks,
                "shared_blocks": shared_blocks,
                "active_block_refs": active_block_refs,
                "prefix_hit_blocks": self._prefix_hit_blocks,
                "prefix_miss_blocks": self._prefix_miss_blocks,
                "prefix_hit_tokens": self._prefix_hit_tokens,
                "prefix_evictions": self._prefix_evictions,
            }
        )
        return base

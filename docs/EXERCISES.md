# Exercises

## v0.1-v0.4

Use historical branch diffs to explain static vs continuous batching, paged addressing, and chunked prefill before modifying v0.5.

## v0.5 prefix caching

1. With block size 4, calculate how many prompt tokens can be reused for prompt lengths 3, 4, 5, 8 and 9. Remember that next-token logits still need prompt execution.
2. Explain why hashing only the local block token IDs can be wrong when identical chunks appear under different preceding context.
3. Draw block tables for two requests sharing two full prefix blocks but having different private tails.
4. Show the refcount transitions when A and B share a block, then A finishes, then B finishes.
5. Modify the LRU eviction test so every cached block has `refcount > 0`. Explain why v0.5 must return allocation failure instead of evicting one.
6. Add a third request that arrives while two requests already share a prefix. Inspect `shared_blocks` and `active_block_refs`.
7. Implement an alternative cache-hit metric: request-level hit ratio instead of block/token-level counters. Compare what each metric hides.

## Preview v0.6

Design a recompute-preemption state transition without implementing swap:

```text
RUNNING -> PREEMPTED/WAITING -> KV released -> later PREFILL recomputation
```

List the invariants needed to avoid stale KV ownership.

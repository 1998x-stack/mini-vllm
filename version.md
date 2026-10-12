# mini-vLLM v0.5 — prefix caching + shared KV block ownership

## 1. Version identity

- **Branch:** `v0.5`
- **Previous version:** `v0.4`
- **Theme:** content-addressed prefix caching and shared physical KV blocks
- **Status:** frozen historical learning snapshot after validation

v0.5 keeps v0.4's token-budget scheduler unchanged and changes the KV ownership model from:

```text
one request -> private physical blocks
```

to:

```text
identical immutable prompt prefix
        ↓
content-addressed full blocks
        ↓
multiple request block tables
        ↓
shared physical KV + refcount
```

## 2. Why v0.5 follows v0.4

v0.3 made KV allocation block based. v0.4 made prompt computation incremental. Once block boundaries and partial prefill are explicit, identical prompt prefixes no longer need to be recomputed or duplicated.

The new question is ownership:

> When several requests point at the same physical KV block, who may mutate it and when may it be recycled?

## 3. Primary mental model

```text
Prompt A: [1 2] [3 4] [5]
Prompt B: [1 2] [3 4] [9]

hash(prefix block 0) ──► physical block 7
hash(prefix block 1) ──► physical block 2

A block table: [7, 2, private-A]
B block table: [7, 2, private-B]
                 ▲  ▲
                 └──┴── immutable shared prefix
```

Reference counts track active request ownership:

```text
A + B active: refcount = 2
A finishes:    refcount = 1
B finishes:    refcount = 0
                           │
                           └─ retained by prefix index until eviction
```

## 4. Core mechanisms

### 4.1 Chained block hashing

Each full prompt block receives a hash that includes the previous prefix hash and the block's token IDs. Therefore equal token chunks under different preceding context do not alias accidentally.

### 4.2 Prefix index

```text
prefix_hash -> physical_block
```

A new request walks full prompt blocks from the beginning and attaches consecutive hits to its block table.

### 4.3 Full blocks only

Only complete prompt blocks become cacheable. A partial final block remains request-private and writable.

This gives a simple copy-on-append rule:

```text
shared immutable prefix | private mutable tail
```

### 4.4 Refcounts

Every block-table reference increments a physical block refcount. Releasing a request decrements those refs exactly once.

A cached block with `refcount == 0` is still resident because the prefix index retains it for future requests.

### 4.5 LRU eviction

When allocation needs more blocks than the free list contains, v0.5 evicts least-recently-used cached blocks whose active refcount is zero.

Active shared blocks are never eviction candidates. If pressure cannot be resolved without preempting active requests, allocation fails; scheduler-visible preemption remains v0.6 scope.

## 5. Critical logits invariant

KV cache stores K/V, not the final-position logits needed for next-token sampling.

Therefore even an exact prompt cache hit cannot skip every prompt token. The educational implementation leaves at least one prompt token (or, because reuse is block-granular, the final full block) to execute.

For block size 2:

```text
prompt = [1 2] [3 4]
cache contains both blocks

warm request reuses [1 2]
recomputes          [3 4]
                         ↓
                  correct logits
```

## 6. Engine/cache lifecycle hooks

`KVCacheBackend` gains backward-compatible no-op hooks:

```python
prepare_request(request_id, prompt_token_ids) -> cached_token_count
commit_work(request_id, start_pos, token_ids, prompt_length)
```

v0.1-v0.4 cache backends return zero/do nothing. v0.5 uses them to attach cached prefixes before scheduling and publish newly immutable blocks only after all model layers have written K/V.

## 7. Important source files

```text
mini_vllm/cache/base.py
        ↓
mini_vllm/cache/paged.py
        ↓
mini_vllm/cache/prefix.py
        ↓
mini_vllm/core/engine.py
        ↓
mini_vllm/versions/v0_5/engine.py
        ↓
examples/06_prefix_caching.py
```

Recommended comparison:

```bash
git diff v0.4..v0.5 -- mini_vllm/cache mini_vllm/core tests examples version.md
```

## 8. Tests and validation

v0.5 validates:

- warm requests skip cached full-prefix tokens;
- warm and cold deterministic generation produce identical output tokens;
- shared physical blocks reach refcount 2 with two active owners;
- private tails do not mutate shared prefix K/V;
- exact-prompt hits still execute enough prompt work to produce logits;
- finished requests drop active refs while cached blocks remain resident;
- memory pressure evicts only unreferenced cached blocks;
- v0.1-v0.4 regression behavior remains valid.

The frozen local snapshot passes **18 tests** plus all six executable examples.

## 9. Learning goals

After v0.5, the learner should be able to explain:

- why prefix caching depends on paged block ownership;
- why shared blocks must be immutable;
- why a mutable tail cannot be shared blindly;
- why refcount and cache residency are different concepts;
- why KV cache hits do not automatically provide logits;
- how content hashes prevent context-incorrect aliasing;
- why eviction is different from active-request preemption.

## 10. Explicit non-goals

- scheduler preemption/recompute under active KV pressure — v0.6;
- swap-to-host policy;
- speculative decoding — v0.7;
- process/distributed execution;
- production hash kernels or radix-tree indexing;
- Triton/CUDA PagedAttention.

## 11. Next version

`v0.6` introduces **KV-pressure preemption**. It will make allocation failure visible to scheduling policy and teach recompute-based preemption, request state rollback to waiting/preempted, and fairness/starvation behavior.

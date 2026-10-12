# mini-vLLM v0.5 — active planning branch

## 1. Version identity

- **Branch:** `v0.5`
- **Base:** latest validated `v0.4` / `main`
- **Previous version:** `v0.4`
- **Theme:** prefix caching and shared KV-block ownership
- **Status:** planning / implementation target; runtime still matches validated v0.4 until acceptance criteria pass

---

## 2. Why v0.5 follows v0.4

v0.3 introduced physical KV blocks and per-request block tables. v0.4 made prompt growth incremental through chunked prefill. The next obvious waste is repeated computation and storage for identical immutable prompt prefixes.

```text
Request A: [system prompt ........] + user A
Request B: [system prompt ........] + user B

without prefix cache:
A owns prefix KV
B owns duplicate prefix KV

with prefix cache:
        shared immutable full blocks
              /          \
        A mutable tail   B mutable tail
```

The primary question is:

> How can multiple requests safely reuse already-computed full KV blocks without corrupting ownership, mutation, or release semantics?

---

## 3. Primary mental model

```text
prompt tokens
    │
    ▼
full-block content key / hash
    │
    ▼
prefix index
    │
    ├── miss -> allocate + compute + publish immutable full block
    │
    └── hit  -> attach existing physical block
                       │
                       ▼
                 increment refcount
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       shared prefix        private tail
       immutable blocks     mutable blocks
```

---

## 4. Required delta from v0.4

### 4.1 Block ownership metadata

Introduce explicit metadata for physical blocks, including at minimum:

- reference count;
- immutable/shared state for cached full blocks;
- ownership/release rules;
- reusable/free state.

### 4.2 Prefix identity

Introduce a deterministic educational prefix key. Start with full-block token content and enough context to prevent unsafe reuse. The design must make hash/key assumptions explicit rather than treating matching token IDs as magic.

### 4.3 Prefix index

Add a lookup structure mapping reusable prefix keys to physical blocks or block chains. The index belongs to cache-management policy, not to the Transformer model.

### 4.4 Request attachment

On a prefix hit, a request should attach shared full blocks to its block table, advance the reusable/computed prefix state appropriately, and compute only the unmatched suffix.

### 4.5 Mutable tail rule

Only complete immutable prefix blocks may be shared in the first educational implementation. A partially filled tail block remains private to one request.

### 4.6 Release

Finishing one request decrements references. A shared block may return to the free pool only when no request/cache entry still owns it.

---

## 5. Core invariants

### Shared-block immutability

A block published to the prefix cache must never be mutated in place by a request.

### Refcount correctness

```text
refcount(block) == number of live owners/references represented by the cache model
```

No early free and no leaked ownership.

### Tail isolation

Partially filled or request-specific continuation blocks are private and must not be reused as immutable prefixes.

### Semantic equivalence

Prefix-cache hit and miss execution must produce the same deterministic token history/logits as recomputing the prefix normally.

### Safe release

Finishing request A must not invalidate a shared block still referenced by request B.

---

## 6. Acceptance criteria

v0.5 cannot be frozen until tests demonstrate:

- [ ] two requests with the same full-block prefix share physical block IDs;
- [ ] a prefix hit skips recomputation for the reusable prefix;
- [ ] shared full blocks are immutable;
- [ ] request-specific tail blocks remain private;
- [ ] finishing one sharer does not free blocks still used by another;
- [ ] the last release makes blocks recyclable according to cache policy;
- [ ] prefix-cache hit and no-cache reference execution are semantically equivalent;
- [ ] hash/key collision assumptions are explicit and testable;
- [ ] block/refcount state is inspectable in an educational example;
- [ ] all previous v0.4 scheduling and chunked-prefill tests remain green;
- [ ] validated implementation is synchronized back to `main`.

---

## 7. Preferred architecture

```text
Scheduler
   │ WorkItem suffix
   ▼
Prefix-aware KV cache manager
   ├── prefix lookup/index
   ├── block metadata/refcount
   ├── shared immutable full blocks
   └── private mutable tails
   │
   ▼
ModelRunner
```

Do not put prefix-hash lookup inside `TinyDecoderLM`, and do not make the Scheduler directly mutate physical block refcounts.

---

## 8. Explicit non-goals

- eviction under severe KV pressure — v0.6;
- preemption/swap/recompute policy — v0.6;
- speculative decoding — v0.7;
- distributed prefix cache — later;
- production-grade collision-resistant cache protocol — later;
- CUDA/Triton optimized kernels — later.

---

## 9. Suggested implementation sequence

```text
v0.5.1 block metadata + refcount
   ↓
v0.5.2 full-block prefix key/index
   ↓
v0.5.3 request attach + suffix-only compute
   ↓
v0.5.4 release/recycling semantics
   ↓
v0.5.5 semantic-equivalence + sharing trace
```

---

## 10. Next version

`v0.6` should introduce KV-pressure handling: admission failure, preemption, recompute/swap-style policy, and explicit memory-pressure decisions.

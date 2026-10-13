# mini-vLLM — Architecture-Driven Development Plan

`main` is always the newest validated runtime. Version branches are educational snapshots. Exact release commits must pass repository CI.

## Version ladder

```text
v0.1  static batching + contiguous KV
v0.2  continuous batching
v0.3  paged KV
v0.4  token budget + chunked prefill
v0.5  prefix cache + shared ownership
v0.6  preemption + recompute
v0.7  speculative transactions
v0.8  EngineCore / Executor / Worker
v0.9  WorkerGroup + reference tensor parallel
v0.10 async streaming/cancellation/backpressure
v0.11 compile-aware execution
v0.12 optimized PagedAttention                         ← current implementation
v0.13 prefix-aware scheduling                          ← next
v1.0  integrated educational runtime
```

## v0.12 lesson

PagedAttention optimization belongs below request/cache policy. The cache still owns physical blocks and block tables; attention receives only the current logical length and current storage mapping.

Reference equation:

```text
for logical position p:
    logical_block = p // block_size
    offset        = p % block_size
    physical      = block_table[logical_block]
    K[p], V[p]    = physical_storage[physical, offset]
```

The reference backend follows the readable per-position cache path. The vectorized backend constructs all logical positions at once, maps them through the block table with tensor indexing, gathers K/V in one tensor operation, and then performs the same GQA/MHA attention math.

Correctness rules:

- logical length masks unused tail slots;
- physical blocks need not be contiguous;
- prefix sharing and recycled blocks remain cache concerns;
- vectorized and reference outputs must match;
- unsupported cache layouts use explicit reference fallback when configured;
- v0.11 compile-aware projection remains a separate rank-local optimization boundary.

Release rule: only an exact v0.12 commit with a green repository CI run may be synchronized to `main`. This wording remains valid before and after release.

## Planned

- v0.13: prefix-aware admission/cache-aware scheduling policy.
- v1.0: integrated educational serving runtime.

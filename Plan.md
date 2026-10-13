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
v0.12 optimized PagedAttention
v0.13 prefix-aware scheduling
v1.0  integrated educational runtime                  ← current implementation
```

## v1.0 lesson

The final lesson is composition: a serving mechanism is only useful when its ownership boundary remains legible next to all the other mechanisms.

```text
caller
  ↓
AsyncEngine                       delivery / cancellation / backpressure
  ↓
EngineCore                        logical truth and commit authority
  ↓
Scheduler                         admission / budget / fairness / victim policy
  ↓
Executor / WorkerGroup            topology / rank consistency / rollback
  ↓
Worker / ModelRunner              rank-local model execution
  ↓
KV + PagedAttention               storage ownership + logical reads
  ↓
Compile-aware projection          rank-local tensor specialization
  ↓
Speculative verification          transactional candidate acceptance
  ↓
committed output stream
```

v1.0 intentionally avoids a new major scheduling, cache, distributed, or kernel mechanism. It adds a stable construction facade and a public read-only snapshot so the complete runtime can be reasoned about without reaching into private implementation state.

## Stable public surfaces

### Construction

`RuntimeConfig` composes:

- `EngineConfig`;
- `TinyModelConfig`;
- `CacheConfig`;
- `CompileConfig`;
- `PagedAttentionConfig`;
- logical `world_size`;
- async `max_queue_size`;
- prefix-fairness threshold.

`create_engine()` defaults to v1.0. Explicit builders for v0.1–v0.13 remain available.

### Observability

`engine.snapshot()` returns one immutable teaching view containing:

- request lifecycle and computed frontiers;
- prefix-aware scheduler observations;
- cache statistics;
- recent EngineStepOutput traces, including preemption, speculation and committed tokens;
- distributed/rank execution topology;
- attention backend counters;
- compile registry counters;
- async pending/cancellation/driver state.

Observation must be side-effect free.

## Cross-version correctness contract

The stable invariant list is maintained in `docs/INVARIANTS.md`. The most important rules are:

1. computed tokens imply valid addressable KV;
2. sampled tokens are not computed until model execution catches up;
3. physical ownership precedes K/V access;
4. block-table translation is authoritative;
5. prefix hints are advisory facts, never ownership;
6. preemption rolls back to valid replayable state;
7. rejected speculative state disappears;
8. distributed failures cannot partially commit replicated state;
9. async clients see committed tokens only and one terminal result;
10. optimized attention equals the readable reference;
11. compile-aware projection equals eager projection;
12. warm-prefix policy cannot starve cold work;
13. observability cannot mutate runtime state.

## v1.0 integrated validation

`examples/15_integrated_runtime.py` exercises a warm prefix, prefix-aware admission, token-budgeted work, speculative verification, distributed execution, vectorized PagedAttention, compile-aware projection and async streaming, then explains the run using only the public snapshot.

The v1.0 test suite additionally proves:

- default construction selects v1.0;
- deterministic v1.0 output equals validated v0.13;
- `RuntimeConfig` composes the public runtime;
- snapshots are repeatable and side-effect free;
- scheduler/cache/execution/speculation/committed-output observations are publicly reachable;
- real warm-prefix cost is visible in admission traces;
- historical builders remain available.

## Release rule

Only the exact v1.0 implementation commit that passes repository CI may be synchronized to `main`.

## After v1.0

Further versions require a genuinely new learning target—for example real GPU kernels, actual multiprocessing/distributed transport, or production-style model loading. Incremental accumulation alone is not sufficient reason for a new version.

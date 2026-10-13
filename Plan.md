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
v0.11 compile-aware execution                         ← current candidate
v0.12 optimized PagedAttention                        ← next
v0.13 prefix-aware scheduling
v1.0  integrated educational runtime
```

## v0.11 lesson

Compilation belongs inside the execution plane, not the serving control plane. v0.11 keeps AsyncEngine, EngineCore, Scheduler, cache ownership, WorkerGroup orchestration, and collectives eager. A rank-local pure tensor projection kernel is wrapped by an explicit signature registry around `torch.compile`.

Core equation:

```text
same signature → reuse compiled callable
new signature  → explicit new variant
compile failure → eager correctness oracle
```

CI uses the lightweight `eager` backend to test Dynamo/registry semantics without claiming performance. Future optimization may replace the backend or widen the pure tensor boundary only after numerical equivalence remains proven.

## Planned

- v0.12: vectorized/optimized PagedAttention, with reference equivalence first.
- v0.13: prefix-aware admission/cache policy.
- v1.0: integrated educational serving runtime.

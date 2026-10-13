# mini-vLLM — Architecture-Driven Development Plan

> `main` is always the newest integrated validated runtime. Version branches are educational snapshots. `Plan.md` is the canonical forward sequence.

## 1. Learning goal

mini-vLLM removes one layer of LLM-serving “magic” at a time: request state, prefill/decode, continuous batching, paged KV, token budgets, prefix sharing, memory pressure/preemption, speculative transactions, execution boundaries, distribution, async delivery, and optimized kernels.

## 2. Reference architecture after v0.10

```text
Caller
  │
  ▼
AsyncEngine
  │ submission/cancellation intents
  ▼
EngineCore
  Scheduler + logical state + policy
  │
  ▼
ExecutionRequest
  │
  ▼
DistributedExecutor
  │
  ▼
WorkerGroup
 ┌──┴──┐
 ▼     ▼
rank0 rank1 ...
  │
  ▼
ReferenceCollective
  │
  ▼
ExecutionResult
  │
  ▼
EngineCore logical commit
  │
  ▼
EngineStepOutput.committed_token_ids
  │
  ▼
AsyncEngine bounded event delivery
```

Policy remains centralized even when execution fans out across ranks and callers consume results asynchronously.

## 3. Version ladder

```text
v0.1  static batching + contiguous KV
v0.2  continuous batching
v0.3  paged KV + block table
v0.4  token budget + chunked prefill
v0.5  prefix cache + shared ownership
v0.6  preemption + recompute
v0.7  speculative transaction
v0.8  EngineCore / Executor / Worker
v0.9  WorkerGroup + reference tensor parallel
v0.10 async engine + streaming/cancellation/backpressure ← current implementation
v0.11 graph/compile experiments                          ← next
v0.12 optimized PagedAttention
v0.13 prefix-aware scheduling
v1.0  integrated educational serving runtime
```

## 4. Released architecture lessons

### v0.1–v0.4 — serving loop, scheduling, paged memory, token budgets

Build the smallest correct autoregressive runtime, then make sequence admission continuous, KV block-addressed, and token work explicitly budgeted.

### v0.5 — prefix caching

Full prompt blocks can become immutable shared cache entries with refcounts while mutable tails remain private.

### v0.6 — preemption

Logical token history survives memory pressure while invalid physical KV is released and later recomputed.

### v0.7 — speculative transactions

Draft tokens are temporary until target verification commits them. Rejected temporary K/V is rolled back.

### v0.8 — execution boundary

EngineCore owns policy/logical state; Executor owns transport/orchestration; Worker owns model + physical KV mechanism. Mutable policy-bearing `Request` objects do not cross the execution boundary.

### v0.9 — logical multi-rank + reference tensor parallel

One centralized EngineCore fans one execution across logical ranks. Rank-local shards plus explicit reference collectives reconstruct the single-worker mathematical result. Replicated rank KV is rolled back atomically on partial execution failure.

### v0.10 — async request lifecycle + streaming

Primary question:

> How can callers submit, stream, cancel, and apply backpressure without creating a second serving policy plane?

Reference design:

- `AsyncEngine` wraps the synchronous EngineCore rather than replacing it;
- `submit()` returns an `AsyncRequestHandle` and enqueues caller intent;
- `TokenEvent` is generated only from `EngineStepOutput.committed_token_ids`;
- each handle emits exactly one `FinishedEvent`, `CancelledEvent`, or `ErrorEvent`;
- EngineCore exposes an explicit cancellation/abort primitive that removes Scheduler participation and releases active KV;
- cancellation is applied between atomic synchronous steps, not mid-forward;
- per-request queues are bounded;
- the first backpressure policy blocks the cooperative driver rather than dropping/reordering committed events;
- a single asyncio driver drains control intents, runs at most one core step at a time, publishes events, and yields to the event loop;
- core/executor failure conservatively terminates all still-active streams with explicit error events;
- the v0.9 WorkerGroup/distributed execution topology remains unchanged.

Architectural lesson:

```text
async caller lifecycle
        ≠
serving scheduling policy
```

Release rule: only an exact v0.10 commit with a green repository CI run may be synchronized to `main`.

## 5. Planned versions

### v0.11 — graph/compile-aware execution

Study static metadata, graph breaks, recapture conditions, host overhead, and eager-vs-compiled equivalence without moving scheduling/cancellation semantics into graph-specific code.

### v0.12 — optimized PagedAttention

Progress from readable Python paged gather to vectorized PyTorch and then an educational Triton/CUDA path, always compared against the reference implementation.

### v0.13 — prefix-aware scheduling

Integrate prefix-reuse cost into admission and compare cache-aware policies while preserving starvation/fairness tests.

### v1.0

A compact runtime with explicit control plane, execution topology, paged/prefix-aware KV, speculative verification, async delivery, and reference/optimized attention paths.

## 6. Validation matrix

| Version | Key invariant |
|---|---|
| v0.1 | computed state advances only after forward execution |
| v0.2 | free sequence slots backfill |
| v0.3 | paged/contiguous storage preserve logits |
| v0.4 | scheduled tokens stay within budget |
| v0.5 | shared prefix blocks remain immutable/refcounted |
| v0.6 | preemption cannot leave stale KV ownership |
| v0.7 | rejected speculative state disappears |
| v0.8 | executor transport preserves semantics and excludes policy objects |
| v0.9 | rank-local shards + collective reconstruct single-worker semantics |
| v0.10 | only committed ordered tokens stream; terminal/cancel semantics release resources exactly once |

## 7. Scope discipline

Prefer readable reference mechanisms before transport/performance complexity. The first async layer is cooperative asyncio, not threads/processes/networking. Future optimized delivery, graph capture, and kernels must preserve the synchronous EngineCore and eager execution paths as correctness oracles.

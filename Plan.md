# mini-vLLM — Architecture-Driven Development Plan

> This is the canonical forward plan for `main`.
>
> **Repository rule:** `main` always represents the newest integrated, validated implementation. Version branches (`v0.x`) are educational snapshots. Active work may happen on a version branch, but validated increments must be synchronized to `main` so `main` never intentionally lags.

---

# 1. Why this project exists

mini-vLLM exists to remove one layer of LLM-serving “magic” at a time.

It should let a learner answer:

- what state does an inference request carry?
- what is different about prefill and decode?
- why does static batching waste sequence slots?
- how does continuous batching refill those slots?
- why is the KV cache often the dominant serving memory problem?
- what is a physical KV block and a block table?
- why does chunked prefill require a token budget?
- how can prefix blocks be shared safely?
- what happens when active KV no longer fits?
- how can speculative decoding commit or roll back KV state?
- why does a production runtime split EngineCore, Executor, Worker, and ModelRunner?
- what should an optimized PagedAttention kernel preserve relative to a readable reference path?

The goal is a compact serving runtime whose subsystems can be mapped conceptually to production vLLM without reproducing its production-scale complexity.

---

# 2. Architectural reference model

```text
Client / caller
      │
      ▼
   Request
      │
      ▼
  Engine Core
      │
      ▼
   Scheduler ───────────── policy
      │
      ▼
 ScheduleDecision
      │
      ├───────────────┐
      ▼               ▼
KV Cache Manager    ModelRunner
      │               │
      └───────┬───────┘
              ▼
           Sampler
              │
              ▼
       request state update
```

Later versions split execution:

```text
EngineCore
   │ SchedulerOutput
   ▼
Executor
   ├──── Worker 0
   ├──── Worker 1
   └──── ...
```

The roadmap follows dependency order: request/scheduling/cache semantics first, then sharing/rollback, then execution/distribution, then optimized kernels.

---

# 3. Branch/version model

```text
main ───────────────────────────────────────────────► newest validated state
  ├── v0.1  static batching + contiguous KV
  ├── v0.2  continuous batching
  ├── v0.3  paged KV + block table
  ├── v0.4  token budget + chunked prefill
  ├── v0.5  prefix cache + shared ownership
  ├── v0.6  preemption + recompute
  └── v0.7  speculative transaction
```

Rules:

1. `main` is canonical latest integrated behavior.
2. `Plan.md` on `main` is canonical sequencing.
3. `Agents.md` is the repository engineering contract.
4. every version branch has `version.md` and `Agents.md`.
5. historical branches do not absorb later mechanisms.
6. a version is complete only after code, invariant tests, example, and docs agree.

---

# 4. Released versions

## v0.1 — Static batching + contiguous KV

Smallest complete autoregressive serving loop: Request → static cohort → full prefill/one-token decode → contiguous per-request KV → Tiny Decoder → Sampler → request update.

## v0.2 — Continuous batching

Changes scheduler policy only: freed sequence slots are refilled on the next generation iteration.

## v0.3 — Paged KV cache and block tables

Introduces fixed-size physical blocks, per-request block tables, logical-to-physical address translation, incremental allocation, recycling, and numerical equivalence against contiguous KV.

## v0.4 — Token-budget scheduler + chunked prefill

Adds `max_num_batched_tokens`, arbitrary contiguous WorkItem chunks, decode-first budgeting, prompt prefill across multiple steps, and explicit prefill/decode accounting.

## v0.5 — Prefix caching and shared block ownership

Adds chained full-block hashes, prefix index, immutable shared prompt blocks, active refcounts, private mutable tails, warm-prefix reuse, and LRU eviction of unreferenced cached blocks.

## v0.6 — KV-pressure preemption + recomputation

Adds scheduler-visible allocation failure, `PREEMPTED`, deterministic victim selection, active-KV release, computed-token rollback, prefix reattachment, replay of historical logical tokens, and anti-thrashing re-admission.

## v0.7 — Speculative decoding + transactional KV ✅ completed

### Core idea

```text
stable frontier
    ↓
checkpoint
    ↓
draft candidates
    ↓
target verification
    ↓
accept prefix / reject suffix
    ↓
commit / rollback
```

### Implemented mechanisms

- `DraftProvider` abstraction;
- scripted and deterministic educational draft providers;
- `DraftBatch` and `VerificationResult`;
- per-token target logits for verification;
- `CacheCheckpoint`;
- `TransactionalPrefixCachingPagedKVCache`;
- request-private block-table rollback;
- accept-all + target bonus token;
- reject-first fallback;
- partial acceptance;
- rollback across physical block boundaries;
- shared-prefix immutability validation;
- greedy semantic equivalence with ordinary v0.6 decoding;
- fallback to normal target sampling when optional speculative KV cannot fit.

### Architectural lesson

```text
logical committed output
       ≠
temporary speculative KV
```

Draft tokens are proposals until target verification commits them. Rejected speculative K/V must become unreachable and recyclable without mutating stable shared prefixes.

### Intentional simplifications

- greedy verification only;
- no learned draft model;
- no EOS-aware multi-token speculative policy;
- temporary verification work is not charged back into scheduler token budget;
- speculation never triggers extra preemption merely to accelerate execution.

**Status:** implemented and validated in the frozen `v0.7` snapshot. `main` integrates this version.

---

# 5. Planned versions

## v0.8 — EngineCore / Executor / Worker split

### Primary question

> Where should centralized scheduling end and model execution begin?

### Target architecture

```text
EngineCore
  Scheduler + request state + policy
        │
        ▼
SchedulerOutput / ExecutionRequest
        │
        ▼
Executor
        │
        ▼
Worker / ModelRunner / KV backend
        │
        ▼
ExecutionResult
        │
        ▼
EngineCore commit/update
```

Keep a local in-process executor as the first reference implementation. Do not introduce multiprocessing before the ownership boundary is explicit.

### Acceptance direction

- moving execution behind `LocalExecutor` must not change deterministic outputs;
- scheduler and victim/admission policy stay in EngineCore;
- worker does not own waiting/running queues;
- execution request/result payloads are explicit and testable;
- speculative verification and cache transaction semantics survive the boundary.

## v0.9 — Distributed execution and tensor parallel concepts

Rank/world-size configuration, worker groups, an educational collective abstraction, simplified tensor-parallel projections, and reference-vs-multi-worker equivalence.

## v0.10 — Async engine and streaming API

Asynchronous request submission, streaming token events, cancellation, backpressure, and thin networking outside scheduler/cache internals.

## v0.11 — CUDA graph / compile-aware experiments

Study static metadata, recapture/graph-break conditions, and host overhead while retaining eager reference execution.

## v0.12 — Optimized PagedAttention path

```text
Python reference paged gather
        ↓
vectorized PyTorch
        ↓
profile
        ↓
Triton/CUDA educational kernel
        ↓
compare against reference
```

## v0.13 — Prefix-aware scheduling + advanced cache policy

Prefix-aware admission cost, cache-aware scheduling, eviction-policy comparisons, priority, and starvation prevention.

## v1.0 — Integrated educational serving runtime

```text
submit request
   ↓
front-end queue
   ↓
EngineCore scheduler
   ↓
token budget / preemption
   ↓
prefix-aware paged KV
   ↓
Executor / Workers
   ↓
model + reference/optimized attention
   ↓
sampling / speculative verification
   ↓
streaming output
```

v1.0 means major architectural boundaries are explicit and testable, not production parity.

---

# 6. Cross-version compatibility rules

Prefer:

```text
new strategy/component + new version composition
```

over scattering version conditionals through core runtime code.

Stable contracts should evolve deliberately:

- `Request`
- `WorkItem` / scheduler output
- `KVCacheBackend`
- `ModelRunner`
- `Sampler`
- `Engine`

When a contract must change, document why the old contract is insufficient and migrate internal callers atomically on the new version branch.

---

# 7. Validation strategy

| Version | Key invariant |
|---|---|
| v0.1 | computed-token state advances only after forward execution |
| v0.2 | a free batch slot is backfilled on the next scheduler step |
| v0.3 | paged and contiguous KV layouts produce equal logits |
| v0.4 | scheduled token count never exceeds token budget |
| v0.5 | shared prefix blocks are immutable and refcounted |
| v0.6 | preemption cannot leak stale KV ownership |
| v0.7 | rollback removes rejected speculative state |
| v0.8 | executor transport does not change engine semantics |

---

# 8. Scope discipline

Do not add merely because production vLLM supports it: broad model compatibility, quantization matrices, multimodal serving, LoRA, production networking/auth, observability stacks, platform-specific kernels, or complex distributed fault tolerance. Add a mechanism only when it teaches the current architecture dependency.

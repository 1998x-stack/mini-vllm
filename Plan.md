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
- what is a physical KV block?
- what is a block table?
- how does a logical token position map to non-contiguous physical memory?
- why does chunked prefill require a token budget rather than only a sequence-count budget?
- how can prefix blocks be shared safely?
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
 prompt / output / sampling / state
      │
      ▼
  Engine Core
      │
      ▼
   Scheduler ───────────── policy
 waiting / running / budget
      │
      ▼
 ScheduleDecision
      │
      ├───────────────┐
      ▼               ▼
KV Cache Manager    ModelRunner
allocation          model execution
block tables        prefill/decode
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

The roadmap follows dependency order: first make request/scheduling/cache semantics explicit, then sharing/rollback, then execution/distribution, and only then optimize kernels.

---

# 3. Branch/version model

```text
main ───────────────────────────────────────────────► newest validated state
  │
  ├── v0.1  static batching + contiguous KV
  │
  ├────── v0.2  continuous batching
  │
  ├──────────── v0.3  paged KV + block table
  │
  ├────────────────── v0.4  token budget + chunked prefill
  │
  └──────────────────────── ...
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

## v0.1 — Minimal serving loop: static batching + contiguous KV

Primary question: **What is the smallest complete autoregressive serving loop?**

Mechanisms: Request/SamplingParams/RequestStatus, prefill vs decode work items, synchronous Engine, static scheduler, contiguous per-request KV, tiny decoder-only Transformer, ModelRunner, sampler, end-to-end completion.

## v0.2 — Continuous batching

Primary question: **Why should batch membership be reconsidered every generation iteration?**

v0.2 keeps request/model/cache mechanisms stable and changes scheduler policy so free sequence slots can be refilled immediately.

## v0.3 — Paged KV cache and block tables

Primary question: **How can logically contiguous token positions live in non-contiguous physical KV blocks?**

Core translation:

```text
logical_block = position // block_size
block_offset  = position % block_size
physical      = block_table[request_id][logical_block]
```

Mechanisms: physical block pool, per-request block table, incremental allocation, reference paged gather, recycling, utilization statistics, numerical equivalence to contiguous storage.

---

# 5. Planned core versions

## v0.4 — Token-budget scheduler + chunked prefill

Add `max_num_batched_tokens`, step token budget, arbitrary chunk-length work items, chunked prompt prefill, decode-first/FCFS policy experiments, and TTFT/ITL step proxies.

Acceptance criteria: long prompts span steps; decode can progress between chunks; scheduled tokens never exceed budget; deterministic chunked and unchunked paths agree.

## v0.5 — Prefix caching and shared block ownership

Add block hashes, prefix index, refcounts, immutable shared full blocks, copy-on-append mutable tails, eviction, hit/miss metrics. Shared blocks must never be mutated through another request.

## v0.6 — Preemption and memory-pressure policy

Expose allocation pressure to scheduling, add recompute-first preemption, request state transitions, fairness/starvation tests. Host swap is optional later.

## v0.7 — Speculative decoding

Introduce `DraftProvider`, verification results, cache checkpoints, commit/rollback. Start with scripted draft tokens. Rejected speculative KV must not remain visible.

## v0.8 — EngineCore / Executor / Worker split

Separate centralized scheduling/KV metadata from execution. Keep a local in-process executor as reference.

## v0.9 — Distributed execution and tensor-parallel concepts

Introduce rank/world-size, worker groups, educational collectives, simplified tensor-parallel projections, and result comparisons against single-process reference.

## v0.10 — Async engine and streaming API

Add async submission, streaming token events, cancellation, bounded queues/backpressure, timeout semantics, optionally a thin OpenAI-like HTTP layer outside scheduler/cache internals.

## v0.11 — CUDA graph / compile-aware execution experiments

Study static metadata, graph capture, dynamic-shape recapture, graph breaks, and host overhead while retaining eager reference behavior.

## v0.12 — Optimized PagedAttention path

Progress from Python reference gather → vectorized PyTorch → profiling → educational Triton/CUDA kernel, always validating against the reference implementation.

## v0.13 — Prefix-aware scheduling + advanced cache policy

Explore cache-aware admission cost, priority, eviction comparisons, starvation prevention.

## v1.0 — Integrated educational serving runtime

Candidate flow:

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
Executor / workers
   ↓
model + reference/optimized attention
   ↓
sampling / speculative verification
   ↓
streaming output
```

v1.0 means explicit/testable architecture boundaries, not production parity.

---

# 6. Cross-version compatibility

Prefer `new strategy/component + new version composition` over scattering `if version == ...` through core code.

Stable contracts should evolve deliberately: `Request`, `WorkItem`, scheduler output, `KVCacheBackend`, `ModelRunner`, `Sampler`, `Engine`.

---

# 7. Validation strategy

| Version | Key invariant |
|---|---|
| v0.1 | computed-token state advances only after forward execution |
| v0.2 | a free sequence slot is backfilled on the next scheduler step |
| v0.3 | paged and contiguous KV layouts produce equal logits |
| v0.4 | scheduled token count never exceeds token budget |
| v0.5 | shared prefix blocks are immutable and refcounted |
| v0.6 | preemption cannot leak stale KV ownership |
| v0.7 | rollback removes rejected speculative state |
| v0.8 | executor transport does not change engine semantics |

---

# 8. Scope exclusions until justified

Do not add merely because production vLLM supports it: broad model compatibility, quantization matrix, multimodality, LoRA serving, production auth/networking, full observability stacks, platform-specific kernel tricks, or distributed fault tolerance. Add mechanisms only when their architectural dependency is ready.

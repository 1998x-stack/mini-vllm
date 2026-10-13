# mini-vLLM — Architecture-Driven Development Plan

> `main` is always the newest integrated validated runtime. Version branches are educational snapshots. `Plan.md` is the canonical forward sequence.

## 1. Learning goal

mini-vLLM removes one layer of LLM-serving “magic” at a time. A learner should be able to explain request state, prefill/decode, continuous batching, paged KV, token budgets, prefix sharing, memory pressure/preemption, speculative transactions, execution boundaries, distribution, and eventually optimized attention.

## 2. Reference architecture after v0.8

```text
Client
  │
  ▼
Request
  │
  ▼
EngineCore
  ├─ request registry / logical history
  ├─ Scheduler / token budget
  ├─ admission + preemption policy
  ├─ sampling + logical commit
  └─ finish decisions
  │
  ▼
ExecutionRequest
  │
  ▼
Executor
  │
  ▼
Worker
  ├─ ModelRunner
  └─ KV backend
  │
  ▼
ExecutionResult
  │
  ▼
EngineCore commit/update
```

The roadmap follows dependency order: serving semantics → memory → scheduling → sharing → pressure recovery → transactional execution → execution boundary → distribution → async/optimization.

## 3. Branch/version model

```text
main ─────────────────────────────────────────────► newest validated state
  ├── v0.1  static batching + contiguous KV
  ├── v0.2  continuous batching
  ├── v0.3  paged KV + block table
  ├── v0.4  token budget + chunked prefill
  ├── v0.5  prefix cache + shared ownership
  ├── v0.6  preemption + recompute
  ├── v0.7  speculative transaction
  └── v0.8  EngineCore / Executor / Worker
```

Rules: `main` is canonical latest behavior; historical branches remain scope-clean; every version has `version.md`; a version completes only when code, invariant tests, examples, and docs agree.

## 4. Released versions

### v0.1 — Static batching + contiguous KV

Smallest complete autoregressive serving loop.

### v0.2 — Continuous batching

Freed sequence slots are refilled every generation iteration.

### v0.3 — Paged KV cache

Fixed-size physical blocks, per-request block tables, logical-to-physical translation, incremental allocation and recycling.

### v0.4 — Token budget + chunked prefill

Adds `max_num_batched_tokens`, arbitrary WorkItem chunks, decode-first budgeting, and prompt prefill across multiple steps.

### v0.5 — Prefix caching + shared ownership

Adds chained block hashes, immutable shared full prompt blocks, active refcounts, private mutable tails, and LRU eviction of unreferenced cached blocks.

### v0.6 — KV-pressure preemption + recomputation

Adds scheduler-visible allocation failure, deterministic victim choice, active-KV release, computed-token rollback, prefix reattachment, history replay, and anti-thrashing re-admission.

### v0.7 — Speculative decoding + transactional KV

Adds DraftProvider, per-token target verification, CacheCheckpoint, request-private rollback, accept-all/reject-first/partial-accept, target fallback/bonus tokens, and greedy equivalence with ordinary decoding.

### v0.8 — EngineCore / Executor / Worker boundary ✅ completed

Primary question:

> Where should centralized scheduling/policy end and physical execution begin?

Implemented mechanisms:

- `EngineCore` central policy coordinator with backward-compatible `Engine` alias;
- `ExecutionItem`, `ExecutionRequest`, `ExecutionResult`;
- separate speculative execution request/result contract;
- abstract `Executor`;
- single-process `LocalExecutor` reference transport;
- `Worker` owning ModelRunner + physical KV backend only;
- normal prefill/decode and speculative verification both execute through Executor;
- cache lifecycle calls (`prepare`, reserve, commit, checkpoint, rollback, free, stats) cross the explicit boundary;
- execution payload strips mutable `Request` objects;
- LocalExecutor compatibility views keep educational `engine.cache` / `engine.model_runner` inspection available;
- executor failures surface before logical request commit;
- v0.7/v0.8 deterministic semantic-equivalence coverage.

Architectural lesson:

```text
central policy state
      ≠
worker execution mechanism
```

The first executor remains local and synchronous. v0.8 proves ownership/message contracts before adding process or network complexity.

## 5. Planned versions

### v0.9 — Distributed execution + tensor-parallel concepts

Primary question: how can one logical execution request fan out across multiple workers/ranks without moving scheduler policy out of EngineCore?

Planned concepts:

- `WorkerGroup` / rank + world-size configuration;
- educational collective abstraction (`all_reduce` / broadcast semantics);
- simplified tensor-parallel linear projections;
- deterministic single-worker vs multi-worker equivalence tests;
- no production NCCL or multi-node fault tolerance yet.

### v0.10 — Async engine + streaming

Asynchronous submission, streaming token events, cancellation, backpressure, and transport outside scheduler/cache internals.

### v0.11 — graph/compile-aware experiments

Study static metadata, recapture/graph-break conditions, and host overhead while retaining eager correctness paths.

### v0.12 — Optimized PagedAttention path

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

### v0.13 — Prefix-aware scheduling + advanced cache policy

Prefix-aware admission cost, cache-aware scheduling, eviction-policy experiments, priority and starvation prevention.

### v1.0 — Integrated educational serving runtime

```text
submit request
   ↓
EngineCore scheduler/policy
   ↓
Executor / WorkerGroup
   ↓
prefix-aware paged KV
   ↓
model + reference/optimized attention
   ↓
sampling / speculative verification
   ↓
streaming output
```

v1.0 means major architecture boundaries are explicit and testable, not production parity.

## 6. Cross-version compatibility

Prefer a new strategy/component + new version composition over version conditionals scattered through core runtime code.

Stable contracts evolve deliberately: Request, scheduler work, KV cache, ModelRunner, Sampler, EngineCore, Executor, Worker.

When a contract changes, document why the old contract is insufficient and migrate current-main callers atomically while leaving historical branches frozen.

## 7. Validation matrix

| Version | Key invariant |
|---|---|
| v0.1 | computed state advances only after forward execution |
| v0.2 | free batch slots backfill on next scheduler step |
| v0.3 | paged and contiguous layouts preserve logits |
| v0.4 | scheduled tokens never exceed token budget |
| v0.5 | shared prefix blocks are immutable/refcounted |
| v0.6 | preemption cannot leak stale KV ownership |
| v0.7 | rejected speculative state is rolled back |
| v0.8 | executor boundary preserves semantics and excludes policy objects |
| v0.9 | worker-group execution equals single-worker reference |

## 8. Scope discipline

Do not add broad model compatibility, production networking/auth, complex observability, quantization matrices, multimodal serving, LoRA, custom kernels, or distributed fault tolerance merely because production systems have them. Add a mechanism only when it teaches the current architecture dependency.

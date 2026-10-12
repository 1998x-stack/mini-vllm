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

### Primary question

> What is the smallest complete autoregressive serving loop?

### Mental model

```text
Request
  ↓
Static cohort
  ↓
full prefill / one-token decode
  ↓
contiguous per-request KV slab
  ↓
Tiny Decoder
  ↓
Sampler
  ↓
request update
```

### Mechanisms

- Request / SamplingParams / RequestStatus
- prefill versus decode work items
- central synchronous Engine loop
- static cohort scheduler
- contiguous per-request KV cache
- tiny decoder-only Transformer
- model-runner boundary
- greedy/temperature/top-k/top-p sampler
- end-to-end request completion

### Intentionally hidden

- continuous batching
- paged KV
- token-budget scheduling
- prefix caching
- preemption
- speculative decoding
- distributed execution
- custom kernels

---

## v0.2 — Continuous batching

### Why it follows v0.1

v0.1 makes wasted batch slots visible. v0.2 changes **scheduler policy only** while keeping request, model, engine, and contiguous KV mechanisms stable.

### Primary question

> Why should batch membership be reconsidered every generation iteration?

### Delta

```text
v0.1 static cohort
A B
A _
A _
C _

v0.2 continuous refill
A B
A C
A _
```

### Mechanisms

- continuous admission/backfill
- explicit scheduler-policy comparison
- invariant test showing a new request enters immediately after a slot frees

### Still unchanged

- contiguous KV allocation
- full-prompt prefill
- sequence-count capacity policy

---

## v0.3 — Paged KV cache and block tables

### Why it follows v0.2

Once request scheduling is dynamic, memory allocation becomes the next bottleneck to expose. Reserving `max_model_len` for every active request wastes memory and couples serving capacity to worst-case sequence length.

### Primary question

> How can a request see logically contiguous token positions while its KV lives in non-contiguous physical blocks?

### Core translation

```text
logical_block = position // block_size
block_offset  = position % block_size
physical      = block_table[request_id][logical_block]
```

### Mechanisms

- fixed-size physical block pool
- per-request block table
- incremental block allocation
- reference paged gather
- block recycling on finish
- cache-utilization statistics
- numerical equivalence with contiguous KV

### Important limitation

The reference implementation gathers paged K/V into a contiguous prefix before attention. It teaches addressing semantics; it is intentionally not a fused production PagedAttention kernel.

---

# 5. Current and planned core versions

## v0.4 — Token-budget scheduler + chunked prefill ✅ completed

### Primary question

> How should long prompts coexist with latency-sensitive decode tokens?

### Why now

`max_num_seqs` limits sequence count but not per-step token work. A 4K-token prefill and a 1-token decode are very different jobs.

### Required mechanisms

- `max_num_batched_tokens`
- per-step token budget
- arbitrary chunk-length `WorkItem`
- chunked prompt prefill
- decode-first versus FCFS policy experiment
- prefill/decode token accounting
- TTFT/ITL proxy metrics in engine steps

### Acceptance criteria

- a long prompt can be prefetched across multiple steps;
- decode requests can make progress between prompt chunks;
- total scheduled tokens never exceed step budget;
- chunked and unchunked prefill produce equivalent logits/output under deterministic sampling.

**Status:** implemented and validated in the frozen `v0.4` snapshot. `main` integrates this version.

---

## v0.5 — Prefix caching and shared block ownership

### Primary question

> How can requests reuse identical immutable prompt prefixes without recomputing or duplicating KV?

### Required mechanisms

- block content hash
- prefix index
- block refcount
- immutable shared full blocks
- copy-on-append for mutable tails
- eviction policy for unreferenced cached blocks
- cache-hit/miss metrics

### Critical invariant

A shared block must never be mutated through one request's append path.

### Definition of Done evidence

- chained full-block hashes and prefix index are implemented;
- shared blocks carry active request refcounts;
- only full prompt blocks are shared; mutable tails remain private;
- unreferenced cached blocks are evicted in LRU order under allocation pressure;
- cache hit/miss/token/eviction metrics are exposed;
- warm-prefix execution is semantically equivalent to cold execution under deterministic sampling.

**Status:** implemented and validated in the frozen `v0.5` snapshot. `main` integrates this version.

---

## v0.6 — Preemption and memory-pressure policy ✅ completed

### Primary question

> What happens when runnable requests need more KV blocks than the physical pool can provide?

### Implemented mechanisms

- cache allocation failure remains explicit through `ensure_capacity(...) -> bool`;
- `RequestStatus.PREEMPTED`;
- deterministic latest-admitted victim selection;
- Engine-coordinated active-KV release;
- computed-token rollback while preserving logical output history;
- prefix-cache reattachment before replay;
- replay of historical prompt/output tokens without duplicate sampling;
- anti-thrashing resume barrier: a victim waits for one survivor completion;
- per-step `preempted_request_ids` trace metadata;
- deterministic equivalence tests against a roomy non-preempted run.

### Critical invariants

```text
valid computed-token claim <=> corresponding KV is still addressable
logical output history survives physical KV invalidation
shared prefix refcounts protect other active owners
```

### Definition of Done evidence

- pressure can preempt an active request and free enough blocks for another request;
- preempted requests later re-enter and finish;
- replay does not resample already-known output tokens;
- shared prefix K/V remains byte-identical when another owner is preempted;
- victim selection and anti-thrashing behavior are unit tested;
- impossible single-request allocation fails explicitly rather than silently dropping work;
- pressured and non-pressured deterministic executions produce equal output histories.

**Status:** implemented and validated in the frozen `v0.6` snapshot. `main` integrates this version.

Optional later experiment: host swap. Do not introduce swap before recompute semantics are explicit.

---

## v0.7 — Speculative decoding

### Primary question

> How can cheap draft tokens reduce target-model iterations while preserving target-model correctness?

### Required contracts

```text
DraftProvider
DraftBatch
VerificationResult
CacheCheckpoint
commit()
rollback()
```

Start with a deterministic scripted draft provider before adding a second tiny model.

### Critical learning target

KV state is transactional: rejected speculative tokens must not remain visible to future decoding.

---

## v0.8 — EngineCore / Executor / Worker split

### Primary question

> Where should centralized scheduling end and model execution begin?

### Required architecture

```text
EngineCore
  Scheduler + KV metadata
        │
        ▼
SchedulerOutput
        │
        ▼
Executor
        │
        ▼
Worker / ModelRunner
```

Keep a local in-process executor as the reference implementation.

---

## v0.9 — Distributed execution and tensor parallel concepts

### Required mechanisms

- rank/world-size configuration
- worker group
- educational collective abstraction
- column/row-parallel linear layers or simplified tensor-parallel projection
- deterministic single-process reference versus multi-worker result comparison

Goal: understand execution topology, not reimplement NCCL.

---

## v0.10 — Async engine and streaming API

### Required mechanisms

- asynchronous request submission
- streaming token events
- cancellation
- bounded queues/backpressure
- request timeout semantics
- optional OpenAI-like thin HTTP layer

Networking must remain outside scheduler/cache internals.

---

## v0.11 — CUDA graph / compile-aware execution experiments

### Learning questions

- why does static metadata help graph capture?
- which dynamic shapes/branches cause recapture or graph breaks?
- what overhead remains outside model kernels?

Keep eager execution as a reference path.

---

## v0.12 — Optimized PagedAttention path

### Progression

```text
Python reference paged gather
        ↓
vectorized PyTorch path
        ↓
profile
        ↓
Triton/CUDA educational kernel
        ↓
compare numerics with reference
```

Do not delete the reference implementation.

---

## v0.13 — Prefix-aware scheduling + advanced cache policy

Potential mechanisms:

- prefix-aware admission cost
- cache-aware scheduling
- block eviction policy comparisons
- priority scheduling
- starvation prevention

---

## v1.0 — Integrated educational serving runtime

A candidate v1.0 should make the complete path inspectable:

```text
submit request
   ↓
async/front-end queue
   ↓
EngineCore scheduler
   ↓
token budget / preemption
   ↓
prefix-aware paged KV manager
   ↓
Executor / workers
   ↓
model + optimized/reference attention
   ↓
sampling / speculative verification
   ↓
streaming output
```

v1.0 does not mean production parity. It means the major architectural boundaries are explicit and testable.

---

# 6. Cross-version compatibility rules

Prefer:

```text
new strategy/component + new version composition
```

over scattering:

```python
if version == "v0.3":
    ...
```

through core runtime code.

Stable contracts should evolve deliberately:

- `Request`
- `WorkItem` / scheduler output
- `KVCacheBackend`
- `ModelRunner`
- `Sampler`
- `Engine`

When a contract must change, document why the old contract is insufficient and migrate all internal callers atomically on the new version branch.

---

# 7. Validation strategy

Every version should prove both behavior and mechanism.

Examples:

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

# 8. What should stay out of scope until justified

Do not add merely because production vLLM supports it:

- broad Hugging Face model compatibility;
- quantization matrix;
- multimodal models;
- LoRA serving;
- production networking/auth;
- production observability stacks;
- platform-specific CUDA optimizations;
- complex distributed fault tolerance.

Add a mechanism only when it teaches a serving architecture dependency at the current stage.

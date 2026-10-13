# mini-vLLM Architecture — v1.0

## Integrated serving stack

```text
caller
  ↓
IntegratedAsyncEngine
  │  submit / stream / cancel / backpressure
  ▼
IntegratedEngine (EngineCore + speculative policy)
  │  logical request truth / atomic step / commit authority
  ▼
PrefixAwareBudgetedScheduler
  │  admission / token budget / fairness / preemption victim policy
  ▼
DistributedExecutor
  ▼
WorkerGroup
  │  rank consistency / checkpoint / collective gather
  ▼
rank-local Worker
  ▼
CompileAwareVocabParallelTinyRunner
  ▼
TinyDecoderLM
  ├─ TransactionalPrefixCachingPagedKVCache
  ├─ VectorizedPagedAttention
  └─ compile-aware vocab projection
  ▼
Sampler / speculative verification
  ▼
committed EngineStepOutput
  ▼
async TokenEvent / terminal event
```

v1.0 composes mechanisms introduced in v0.1–v0.13. It does not move ownership boundaries simply to create a new version.

## Ownership map

### Async boundary

`IntegratedAsyncEngine` owns caller-facing handles, pending submissions, cancellation intent and bounded delivery queues. It does not choose admissions, preemption victims, cache blocks or speculative acceptance.

### EngineCore boundary

`IntegratedEngine` remains the logical coordinator. It owns:

- request registration and terminal state;
- authoritative `num_computed_tokens` updates;
- reconciliation between cache facts and logical state;
- preemption transitions;
- speculative logical commit;
- construction of committed `EngineStepOutput`.

v1.0 only adds bounded trace retention and `snapshot()` on top of these existing semantics.

### Scheduler boundary

`PrefixAwareBudgetedScheduler` owns:

- waiting/running admission order;
- decode-first/token-budget work selection;
- prefix-aware expected-compute ranking;
- deterministic fairness protection;
- preemption-victim policy.

It sees `PrefixReuseHint` quantities but never physical block IDs, refcounts, LRU nodes or free-list entries.

### Cache boundary

`TransactionalPrefixCachingPagedKVCache` owns:

- physical K/V storage;
- request block tables;
- block allocation/recycling;
- immutable prefix residency/sharing;
- content-addressed prefix identity;
- transaction checkpoint/rollback;
- side-effect-free reusable-prefix estimation.

The authoritative page translation is:

```text
logical_block = position // block_size
block_offset  = position % block_size
physical      = block_table[logical_block]
```

A prefix hint is not a reservation. Only actual `prepare_request()` may attach reusable blocks and establish a reusable computed frontier.

### Attention boundary

`PagedAttentionBackend` consumes the cache's current logical K/V view. It never allocates, shares, evicts, preempts or rolls back cache blocks.

`VectorizedPagedAttention` performs tensor-indexed paged gather; `ReferencePagedAttention` remains the readable oracle. Both must produce equivalent attention context for the same logical state.

### Compile boundary

`CompiledKernelCache` specializes only the rank-local vocabulary projection by explicit tensor shape signature. Compile cache hits/misses/fallbacks do not change serving semantics.

### Distributed boundary

`WorkerGroup` owns logical rank topology, replicated checkpoint coordination and collective gathering of rank-local logits. Replicated ranks must agree on capacity outcomes, checkpoints and policy-visible prefix facts before logical state can advance.

## One request lifecycle

```text
submit
  ↓
async pending
  ↓
EngineCore.add_request
  ↓
WAITING
  │  side-effect-free prefix estimate
  ▼
Scheduler admission
  ↓
RUNNING
  │  authoritative prepare_request / reconcile
  ▼
chunked prefill
  ↓
decode
  ├─ normal sample
  └─ speculative candidate transaction
          ↓
       accept / reject / rollback
  ↓
committed output tokens
  ↓
TokenEvent delivery
  ↓
FINISHED / CANCELLED / ERROR
```

Under KV pressure a RUNNING request may instead transition through preemption:

```text
RUNNING
  ↓ release active KV
PREEMPTED
  ↓ requeue / fresh reuse estimate
WAITING
  ↓ later admission
recompute known logical history
  ↓
RUNNING
```

Logical output history survives preemption; claims about computed KV do not.

## Fact → policy → execution truth

Several v1.0 mechanisms follow the same pattern:

```text
mechanism fact
    ↓
policy decision
    ↓
authoritative execution
    ↓
logical commit
```

For prefix-aware admission:

```text
cache.estimate_prefix_reuse()
    ↓
PrefixReuseHint
    ↓
AdmissionCost + fairness
    ↓
prepare_request()
    ↓
actual attached prefix
    ↓
reconcile / rebuild WorkItem
```

For speculation:

```text
draft candidates
    ↓
temporary KV transaction
    ↓
target verification
    ↓
rollback rejected suffix
    ↓
commit accepted logical tokens
```

## Public observability architecture

v1.0 adds one read-only teaching surface:

```text
engine.snapshot()
      │
      ├─ AsyncRuntimeSnapshot
      │    ├─ pending IDs
      │    ├─ cancellation intents
      │    └─ driver state
      │
      └─ RuntimeSnapshot
           ├─ RequestSnapshot[]
           ├─ SchedulerSnapshot
           │    └─ SchedulingObservation[]
           ├─ cache_stats
           ├─ ExecutionSnapshot
           │    └─ RankExecutionSnapshot[]
           └─ recent EngineStepOutput[]
```

A learner can answer from this surface:

- Why was a request admitted?
- How much prefix reuse was estimated and actually attached?
- Which prompt/decode tokens executed?
- Was any request preempted?
- Which speculative candidates were accepted/rejected?
- Which attention and compile paths ran on each rank?
- Which tokens were logically committed?

`snapshot()` is observational only. It must not schedule, allocate, attach, evict, touch cache recency or advance logical state.

## Logical state versus physical state

The central distinction across the whole runtime is:

```text
logical token history
        !=
computed token frontier
        !=
physical K/V ownership
```

Examples:

- a newly sampled token is logical but not computed;
- a preempted request retains logical history but may own no active KV;
- a waiting warm request may have a reuse hint but no block table;
- an unreferenced cached prefix block may remain physically resident while no request owns it.

Collapsing any of these states would break replay, prefix sharing or transactional correctness.

## Failure atomicity

Normal distributed execution checkpoints rank-local cache state before physical mutation. A rank failure rolls all participating ranks back before EngineCore can logically commit.

Speculation adds a request-local transaction frontier. Rejected or failed temporary writes are rolled back to valid addressable KV before logical history changes.

Async errors terminate callers only after the serving core has attempted to abort registered requests without hiding the original failure.

## Stable correctness references

- contiguous/reference cache paths teach logical semantics;
- paged address translation is the memory oracle;
- reference PagedAttention is the attention oracle;
- eager projection is the compile correctness oracle;
- deterministic v0.13 output is the v1.0 integration oracle.

See `docs/INVARIANTS.md` for the concise cross-version contract.

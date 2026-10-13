# mini-vLLM Architecture — v0.13

## Stable serving stack

```text
AsyncEngine
  ↓
EngineCore
  ↓
Scheduler
  ↓
Executor / WorkerGroup
  ↓
Worker / ModelRunner
  ↓
KV cache + PagedAttention
```

v0.13 changes admission information flow, not execution mathematics. Speculative transactions, distributed rollback, compile-aware projection, vectorized PagedAttention, cancellation, and streaming retain their previous boundaries.

## New fact → policy → truth boundary

```text
KV cache
  │
  ├─ estimate_prefix_reuse(prompt)     read-only
  │          ↓
  │    PrefixReuseHint
  │          ↓
  │       Scheduler
  │    AdmissionCost + fairness
  │          ↓
  │       admission
  │          ↓
  └─ prepare_request(request, prompt)  authoritative attach
             ↓
      actual reusable tokens
             ↓
      EngineCore reconciliation
```

The estimate exposes logical reuse quantity only. Physical block IDs, refcounts, LRU state, and free-list state remain private to the cache manager.

## Side-effect-free estimation

`PrefixCachingPagedKVCache.estimate_prefix_reuse()` uses the same chained prompt hashes as actual attachment but does not mutate cache ownership or recency.

It may answer that four prompt tokens are reusable while the request still has:

```text
block_table == ()
num_computed_tokens == 0
```

This separation is required so admission policy can compare requests before allocating scarce KV state.

## Deferred first attachment

Historical schedulers keep the old behavior: `EngineCore.add_request()` prepares cache immediately.

Only schedulers advertising `defer_cache_prepare=True` use the v0.13 path:

```text
add request
  ↓
estimate only
  ↓
WAITING without KV ownership
  ↓
Scheduler selects request
  ↓
EngineCore prepare_request()
  ↓
rebuild WorkItem from actual frontier
```

This keeps v0.1-v0.12 behavior stable while making real admission decisions possible in v0.13.

## Cost and fairness

For waiting prefill work:

```text
expected_new_prefill = raw_prefill - estimated_reusable
```

Lower expected work can win admission, with stable arrival order as a tie-break. Each eligible request bypassed by another admission increments a bypass counter. Once the configured threshold is reached, that request becomes fairness-protected and cannot be postponed indefinitely by a stream of warm requests.

Decode-first and actual per-step token-budget rules are unchanged.

## Stale hints

A hint is intentionally weaker than a reservation. Cache state can change after estimation because of eviction, preemption, or other request activity.

EngineCore therefore reconciles after admission:

```text
estimated reusable ─┐
                    ├─ compare / trace
actual prepare ─────┘
        ↓
num_computed_tokens = actual reusable
```

If the estimate was optimistic, the WorkItem is rebuilt from the lower real frontier. The Scheduler may have made a suboptimal admission choice, but serving correctness is preserved.

## Distributed consistency

Replicated ranks independently compute the same logical reuse hint. `WorkerGroup` rejects disagreement before the hint reaches policy. The Scheduler therefore sees one rank-independent logical fact and never observes rank-local physical addresses.

## Preemption and replay

A preempted request releases its active KV and rolls its computed frontier back according to the existing v0.6 contract. When it waits again, v0.13 refreshes its reuse hint from current cache state. On later admission, actual `prepare_request()` re-establishes the only valid computed frontier.

## Optimization layers remain orthogonal

```text
Prefix-aware Scheduler       policy / admission
Paged KV + prefix cache      storage / sharing
Vectorized PagedAttention    KV read optimization
Compiled LM-head projection  tensor specialization
WorkerGroup                  topology / collective semantics
AsyncEngine                  caller delivery / cancellation
```

No one layer receives ownership that belongs to another.

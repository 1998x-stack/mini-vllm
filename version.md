# mini-vLLM v0.13 — prefix-aware scheduling and cache-aware admission

## Version identity

- Branch: `v0.13`
- Previous: validated `v0.12`
- Theme: use reusable-prefix compute savings in admission policy without leaking physical KV ownership into Scheduler
- Release rule: the exact implementation commit may enter `main` only after repository CI passes

## Mental model

```text
waiting Request
      ↓
cache.estimate_prefix_reuse()        side-effect-free fact
      ↓
PrefixReuseHint
      ↓
AdmissionCost
raw prefill - estimated reuse
      ↓
PrefixAwareBudgetedScheduler         policy
      ├─ decode first
      ├─ lower expected new work
      ├─ stable tie-break
      └─ bypass-based fairness
      ↓
admit request
      ↓
cache.prepare_request()              execution truth
      ↓
actual attached prefix
      ↓
reconcile hint vs truth
```

A hint can change admission order. It cannot create KV ownership or advance `num_computed_tokens`.

## Added abstractions

- `PrefixReuseHint(reusable_tokens, reusable_full_blocks, prompt_tokens)`
- `AdmissionCost`
- `PrefixSchedulingEvent`
- `PrefixSchedulingStats`
- `PrefixAwareBudgetedScheduler`
- `KVCacheBackend.estimate_prefix_reuse()`
- read-only hint propagation through Worker / Executor / WorkerGroup
- deferred first cache attachment for v0.13 admission

## Cache fact contract

`PrefixCachingPagedKVCache.estimate_prefix_reuse()` uses the same chained-prefix identity and full-block rules as actual prefix attachment. It also preserves the final-logits invariant: at least one prompt token remains executable.

The estimate is side-effect free. It does not:

- create a request block table;
- increment block refcounts;
- mutate LRU order;
- register request ownership;
- increment prefix-hit accounting.

Replicated ranks must return identical hints or `WorkerGroup` raises.

## Admission cost

For a waiting request:

```text
raw_prefill_tokens   = prompt_length - num_computed_tokens
reusable_tokens      = min(hint.reusable_tokens, raw_prefill_tokens)
expected_new_prefill = raw_prefill_tokens - reusable_tokens
```

Prefix warmth affects *admission ranking*. The existing token budget still counts actual WorkItem tokens scheduled in the current step.

## Deferred ownership

Older versions attach reusable prefix state in `EngineCore.add_request()`. v0.13 deliberately defers the first `prepare_request()` until the request is admitted and selected for work.

Therefore a waiting v0.13 request may have a positive reuse hint while still owning no KV blocks.

```text
hint > 0
block_table(request) == ()
num_computed_tokens == 0
```

After admission, `prepare_request()` is authoritative and the WorkItem is rebuilt from the actual computed frontier.

## Stale-hint safety

Hints are advisory and refreshed while requests wait. They may still become stale because cache state can change between observation and attachment.

Example:

```text
estimate reusable = 6
actual attach      = 0
```

The request begins model work at position 0. It never claims six computed tokens. The mismatch is recorded as a reconciliation event.

Invariant:

```text
hint != reservation
hint != ownership
hint != num_computed_tokens
```

## Fairness

Warm requests are preferred only while no request is starvation-protected. Every time an eligible request is bypassed by another admission, its bypass count increases. At the configured threshold it gains fairness protection and cache warmth can no longer bypass it indefinitely.

Tie-breaking remains deterministic through stable arrival order.

## Compatibility

v0.13 preserves the mechanisms below rather than redesigning them:

- v0.6 recompute preemption;
- v0.7 speculative checkpoint/rollback;
- v0.9 replicated WorkerGroup / tensor-parallel projection;
- v0.10 async streaming, cancellation and backpressure;
- v0.11 compile-aware rank-local projection;
- v0.12 reference/vectorized PagedAttention.

Preempted requests lose active KV ownership, re-enter waiting policy, receive fresh reuse estimates, and later reconcile against actual attachment.

## Validation coverage

`tests/test_prefix_aware_scheduler.py` proves:

- reuse estimation is side-effect free;
- estimate equals actual attachment when cache state is unchanged;
- warm requests can win admission under a tighter estimated compute cost;
- equal costs preserve stable order;
- repeated warm arrivals cannot starve a cold request;
- an optimistic stale hint cannot advance the computed frontier.

`tests/test_v013_version.py` proves:

- deterministic v0.13 async output equals v0.12;
- world-size 1 and 2 agree;
- waiting requests own no KV before admission;
- a real warm prefix changes admission order.

`examples/14_prefix_aware_scheduling.py` prints the admission/reconciliation trace.

The exact release commit must pass `./scripts/check.sh`, including all historical tests and examples 01–14.

## Non-goals

- learned or SLA-aware scheduling;
- global/multi-node prefix routing;
- remote KV transfer;
- changing cache eviction or PagedAttention layout;
- changing speculative acceptance;
- real multi-GPU KV sharing;
- external router/load-balancer policy.

## Learning goal

v0.13 teaches that cached work is useful scheduling information only when mechanism facts, policy decisions, and execution truth remain separate.

## Next

`v1.0` consolidates the complete educational runtime, public APIs, traces, documentation, and end-to-end examples without introducing another major serving mechanism.

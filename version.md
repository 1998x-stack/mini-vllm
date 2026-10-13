# mini-vLLM v0.13 — prefix-aware scheduling and cache-aware admission (planning branch)

## 1. Version identity

- **Branch:** `v0.13`
- **Base:** latest validated `v0.12` / `main`
- **Previous version:** `v0.12`
- **Theme:** expose reusable-prefix compute savings to Scheduler policy without leaking physical cache ownership into the control plane
- **Status:** planning; runtime remains identical to v0.12 until the acceptance criteria below pass

v0.5 made prefix blocks reusable, but scheduling still treats a cold 64-token prompt and a warm prompt with 56 reusable tokens as if they have the same prefill cost. v0.13 asks:

> How should the Scheduler account for *expected new compute* rather than raw prompt length, while preserving fairness and keeping the cache backend policy-neutral?

## 2. Primary mental model

```text
incoming Request
      │
      ▼
cache reuse estimate                 mechanism fact only
      │
      ▼
PrefixReuseHint
      │
      ▼
Scheduler admission cost
  raw prompt tokens
      - reusable prefix tokens
      = expected new prefill work
      │
      ▼
PrefixAwareBudgetedScheduler
      │
      ├─ token budget
      ├─ decode-first rule
      ├─ deterministic tie-break
      └─ fairness / anti-starvation
      │
      ▼
ScheduleDecision
      │
      ▼
EngineCore / cache.prepare_request() actual truth
```

A reuse hint influences policy; it does **not** create computed KV state.

## 3. Ownership boundary

### Cache backend owns facts

The prefix-cache mechanism may expose a read-only estimate such as:

```text
PrefixReuseHint(
    reusable_tokens,
    reusable_full_blocks,
    prompt_tokens,
)
```

The exact type may evolve, but it must not expose physical block IDs, refcounts, LRU internals, or mutable cache ownership to Scheduler code.

### Scheduler owns policy

The Scheduler may use the hint to estimate admission cost and order otherwise-compatible prefill requests. It remains responsible for:

- decode-first scheduling;
- token-budget enforcement;
- max active sequence limits;
- deterministic ordering;
- fairness and starvation prevention;
- preemption-victim policy where applicable.

### EngineCore owns truth reconciliation

At actual admission, `cache.prepare_request()` remains authoritative. Only the returned attached-prefix length may update `Request.num_computed_tokens`.

```text
estimate says 8 reusable
        ↓
policy may prefer request
        ↓
actual prepare_request says 4
        ↓
computed frontier = 4, never 8
```

A stale or optimistic hint therefore cannot overclaim computed state.

## 4. Required abstractions

The first implementation should introduce inspectable concepts similar to:

```text
PrefixReuseHint
AdmissionCost
PrefixAwareBudgetedScheduler
PrefixSchedulingStats / trace
```

Exact names may differ, but the responsibilities must stay separated:

```text
cache estimate → immutable fact
scheduler score → policy
cache attach → execution truth
```

## 5. Prefix-estimate contract

The estimate should answer a narrow question:

> If this prompt were admitted *now*, how many leading prompt tokens appear reusable under the current cache index?

Requirements:

- only immutable full prompt blocks count;
- preserve the v0.5 final-logits invariant: an exact full-prompt hit still cannot skip all work needed to recover final-position logits;
- estimate must use the same chained-prefix identity rules as actual attachment;
- querying an estimate must not mutate refcounts, block tables, LRU ownership, or request registration;
- a later actual attach may legally differ if cache state changed between estimate and admission.

## 6. Admission cost

A readable first cost model:

```text
raw_prefill_tokens      = prompt_length - num_computed_tokens
estimated_reuse         = min(hint.reusable_tokens, raw_prefill_tokens)
expected_new_prefill    = raw_prefill_tokens - estimated_reuse
```

For decode/recompute work, existing v0.12 semantics remain authoritative.

The token budget still counts **work actually scheduled in the current step**, not the total logical request length.

## 7. Scheduling policy

The educational v0.13 policy should remain deterministic and easy to explain.

Suggested order:

```text
1. decode / replay work first                 existing rule
2. eligible prefill requests ranked by:
      a. starvation protection / age
      b. lower expected new prefill cost
      c. stable admission sequence as tie-break
3. chunk work to remaining token budget
```

The implementation may refine this ordering, but cache warmth must never be allowed to create unbounded starvation.

## 8. Fairness invariant

A stream of hot-prefix requests must not permanently block a cold request.

Possible reference mechanism:

```text
waiting age / bypass count
        ↓
reaches fairness threshold
        ↓
request gains protection
        ↓
normal prefix preference can no longer bypass it
```

The exact mechanism should be deterministic and visible in tests/traces. Do not introduce opaque learned priorities.

## 9. Hint invalidation / races

Because v0.13 is still single-process/cooperative, there is no true concurrent cache mutation during one synchronous scheduling operation. However the architecture should still model hints as advisory because between observations:

- another request may finish;
- an unreferenced cached block may be evicted;
- preemption may alter active ownership;
- a newly completed prompt block may be published;
- another request may canonicalize onto a shared prefix.

Therefore:

```text
hint ≠ reservation
hint ≠ ownership
hint ≠ num_computed_tokens
```

## 10. Token-budget example

Suppose only 4 prefill tokens remain in this step.

```text
cold A: prompt remaining=8, reusable=0  → expected new work=8
warm B: prompt remaining=8, reusable=6  → expected new work=2
```

A prefix-aware policy can prefer B because its estimated new compute fits cheaply, while still using chunked prefill and fairness rules for A.

Actual execution must still call the cache mechanism before claiming B's six reusable tokens.

## 11. Prefix-cache privacy

Scheduler must never inspect:

```text
physical block table
_block_refcounts
_prefix_index mapping values
_cache_lru
free-list identity
```

It may consume only a policy-safe immutable reuse summary. This prevents future cache implementations from coupling physical layout to scheduling code.

## 12. Preemption compatibility

v0.6 preemption remains based on memory pressure and active request policy. Prefix awareness may affect admission ordering but must not break:

- deterministic victim selection;
- rollback of `num_computed_tokens` to valid KV;
- shared-prefix refcounts;
- anti-thrashing behavior;
- eventual resumption.

A preempted request's next scheduling cost must reflect the currently valid/reusable state, not stale pre-preemption assumptions.

## 13. Speculative / distributed / async compatibility

v0.13 must not redesign later layers:

- speculative transactions still verify/rollback exactly as v0.7;
- WorkerGroup and tensor-parallel execution stay below Executor;
- compile-aware LM-head projection remains rank-local;
- vectorized PagedAttention remains an execution optimization;
- AsyncEngine still streams committed tokens only;
- cancellation and bounded backpressure semantics remain unchanged.

From the caller perspective, deterministic final token histories must remain equal to v0.12.

## 14. Observability

Add a trace sufficient to explain decisions, for example:

```text
step | request | age | raw-prefill | reusable | expected-new | decision
-----+---------+-----+-------------+----------+--------------+---------
0    | cold-A  | 0   | 8           | 0        | 8            | wait
0    | warm-B  | 0   | 8           | 6        | 2            | admit
1    | cold-A  | 1   | 8           | 0        | 8            | chunk 4
```

Naming can differ, but learners must be able to see why a request was preferred and when fairness overrode cache warmth.

## 15. Validation plan

Focused tests should prove:

1. reuse estimation is side-effect free;
2. estimate and actual attach agree when cache state is unchanged;
3. stale/optimistic estimates cannot advance computed state;
4. warm prompt receives lower estimated admission cost than equivalent cold prompt;
5. under a tight budget, a warm request can be preferred deterministically;
6. equal costs preserve stable ordering;
7. cold requests eventually run under repeated warm arrivals;
8. prefix-aware scheduling preserves token-budget invariants;
9. shared-prefix ownership remains immutable/refcount-correct;
10. preemption/recompute still finishes all bounded requests;
11. v0.13 deterministic output equals v0.12 for world-size 1 and 2;
12. speculative, compile-aware, PagedAttention, async cancellation/backpressure regressions pass.

## 16. Acceptance criteria

v0.13 cannot freeze until:

- [ ] cache exposes a side-effect-free policy-safe prefix reuse estimate;
- [ ] Scheduler does not inspect physical cache internals;
- [ ] an explicit admission-cost abstraction exists;
- [ ] prefix warmth changes at least one deterministic scheduling decision;
- [ ] token budget counts real scheduled work correctly;
- [ ] actual cache attachment remains the only source of computed-prefix truth;
- [ ] stale estimate test proves no computed-token overclaim;
- [ ] deterministic tie-breaking is tested;
- [ ] bounded fairness/starvation test passes;
- [ ] preemption + prefix sharing remain correct;
- [ ] deterministic v0.13 outputs match v0.12;
- [ ] world-size 1/2 semantics match;
- [ ] async streaming/cancellation/backpressure remain unchanged;
- [ ] an executable example prints prefix-aware admission decisions;
- [ ] all v0.1-v0.12 regression tests and examples pass;
- [ ] exact release-candidate GitHub Actions `./scripts/check.sh` passes before `main` synchronization.

## 17. Non-goals

- learned or reinforcement-learning scheduler;
- production SLA/priority classes;
- global multi-node prefix routing;
- remote KV transfer;
- production cache eviction optimization;
- changing physical PagedAttention layout;
- changing speculative acceptance algorithms;
- real multi-GPU cache sharing;
- external request router/load balancer.

## 18. Learning goals

After v0.13, a learner should be able to explain:

- why prefix caching alone does not make scheduling cache-aware;
- fact/mechanism versus policy separation;
- raw prompt length versus expected new compute;
- why a reuse estimate cannot be treated as KV ownership;
- how stale hints are reconciled safely;
- why cache-aware scheduling needs explicit fairness;
- why Scheduler should not depend on physical block identities;
- how prefix-aware policy composes with preemption, speculation, distributed execution, compilation, optimized attention, and async streaming.

## 19. Next

`v1.0` integrates the complete educational runtime, consolidates public APIs/traces/docs, and freezes the architecture learned across v0.1-v0.13 without introducing another major serving mechanism.

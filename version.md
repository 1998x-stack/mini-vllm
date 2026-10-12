# mini-vLLM v0.6 — KV-pressure preemption (planning branch)

## 1. Version identity

- **Branch:** `v0.6`
- **Base:** latest validated `v0.5` / `main`
- **Previous version:** `v0.5`
- **Theme:** scheduler-visible KV allocation failure and recompute-based preemption
- **Status:** planning / implementation target; runtime still behaves as v0.5 until acceptance criteria pass

## 2. Why v0.6 follows v0.5

v0.3 introduced a finite physical KV block pool. v0.5 added prefix-cache residency and shared ownership. Memory pressure can now occur even when the scheduler still has runnable work.

The missing contract is:

> What should the runtime do when a scheduled request cannot reserve the KV blocks it needs?

v0.5 deliberately stops at allocation failure. v0.6 moves that failure into scheduling policy and teaches **preemption by recomputation** before considering host swap.

## 3. Primary mental model

```text
scheduled work
     │
     ▼
reserve KV capacity
     │
     ├── success ─────────────► execute
     │
     └── failure
           │
           ▼
      choose victim
           │
           ▼
      release victim KV
           │
           ▼
 victim becomes PREEMPTED/WAITING
 num_computed_tokens rolls back
           │
           ▼
 later recompute prompt/output history
```

The key trade-off is explicit:

```text
memory capacity  ↔  recomputation cost  ↔  fairness/latency
```

## 4. Required delta from v0.5

### 4.1 Request lifecycle

Add an explicit preempted state or equivalent transition so a request can be runnable again after its active KV ownership is released.

A first reference lifecycle may be:

```text
WAITING -> RUNNING -> PREEMPTED -> WAITING/RUNNING -> FINISHED
```

### 4.2 Cache contract

The scheduler/engine must be able to distinguish:

- capacity available;
- capacity unavailable after evicting unreferenced prefix-cache blocks.

Do not let the cache choose which active request to sacrifice. Victim choice is scheduling policy.

### 4.3 Recompute preemption

First implementation should release a victim's active KV and later reconstruct its model state by recomputing known token history.

Do **not** add CPU swap yet.

### 4.4 Victim policy

Start with one deterministic policy that is easy to inspect, for example:

- prefer preempting the most recently admitted eligible request;
- never preempt a finished request;
- avoid preempting the request currently being committed;
- document tie-breaking explicitly.

The exact production-optimal policy is not the learning goal.

### 4.5 Fairness

A preempted request must not starve forever. Track enough metadata to test that repeatedly preempted requests eventually make progress under a bounded workload.

### 4.6 Prefix-cache compatibility

Preemption must release **active request references** without corrupting shared immutable prefix blocks. Cached blocks with other owners or prefix-index residency must remain valid.

## 5. Critical invariants

### Ownership invariant

After preemption, the victim must own no private active KV blocks from the discarded execution state.

### Computed-token invariant

If KV state is discarded, `num_computed_tokens` must roll back to the amount of state that remains valid and addressable. It may not claim tokens are computed when their KV was released.

### Shared-prefix invariant

Preempting one request must not mutate or free a shared block still referenced by another request.

### Replay invariant

Recomputation of a preempted request under deterministic sampling must produce the same subsequent token history as an equivalent non-preempted execution.

## 6. Acceptance criteria

v0.6 cannot be frozen until:

- [ ] KV allocation failure is exposed to scheduling/engine policy rather than silently dropping work;
- [ ] at least one request can be preempted to free active KV capacity;
- [ ] preemption releases private active blocks without corrupting shared prefix blocks;
- [ ] `num_computed_tokens` is rolled back consistently with retained KV state;
- [ ] the request later resumes through recomputation and finishes;
- [ ] deterministic preempted and non-preempted execution produce equivalent output tokens;
- [ ] victim selection is deterministic and tested;
- [ ] starvation/fairness behavior has an explicit test;
- [ ] v0.1-v0.5 regression tests continue to pass;
- [ ] an executable timeline example makes pressure -> preempt -> recompute visible;
- [ ] validated implementation is synchronized to `main`.

## 7. Preferred architecture

Do not hide preemption inside `PrefixCachingPagedKVCache`.

Preferred direction:

```text
Scheduler / Engine policy
        │
        ├── schedule work
        │
        ▼
KV capacity reservation
        │
        ├── success -> execute
        │
        └── failure -> PreemptionDecision
                           │
                           ▼
                    release victim state
                           │
                           ▼
                    reschedule/recompute
```

Potential new concepts should remain small and explicit, e.g. `PreemptionDecision`, `preempt(request)`, or a scheduler callback for allocation pressure.

## 8. Explicit non-goals

- host/CPU KV swap;
- speculative decoding — `v0.7`;
- multiprocess Executor/Worker split — `v0.8`;
- sophisticated priority scheduling;
- production admission control;
- CUDA/Triton kernels.

## 9. Next version

`v0.7` introduces **speculative decoding** with cache checkpoints and commit/rollback semantics. v0.6 is a prerequisite because it makes KV state invalidation and reconstruction explicit before speculative tokens become transactional.

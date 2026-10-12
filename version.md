# mini-vLLM v0.6 — KV-pressure preemption + recomputation

## 1. Version identity

- **Branch:** `v0.6`
- **Previous version:** `v0.5`
- **Theme:** scheduler-visible KV pressure, active-request preemption, and deterministic replay
- **Status:** frozen historical learning snapshot after validation

v0.6 keeps v0.5 prefix caching and v0.4 token-budget scheduling, but closes the first serious resource-pressure loop:

```text
scheduled request needs KV
        │
        ▼
allocation fails
        │
        ▼
scheduler chooses victim
        │
        ▼
release victim active KV
        │
        ▼
rollback physical execution frontier
        │
        ▼
resume later by recomputation
```

---

## 2. Why v0.6 follows v0.5

v0.3 made the KV pool finite. v0.5 added resident cached blocks and shared ownership. Once active requests and cached prefixes compete for a bounded physical pool, an allocator can legitimately say:

```text
no evictable free/cached block remains
```

v0.5 stops there. v0.6 answers the next systems question:

> Which active request should lose its execution state, and how can it later continue without changing logical generation semantics?

---

## 3. Core distinction: logical history vs physical state

A request owns two kinds of state:

```text
logical history
  prompt_token_ids
  output_token_ids

physical execution state
  num_computed_tokens
  active KV block references
```

Preemption discards the second category, not the first.

```text
before preemption
prompt = [p0 p1 p2]
output = [o0]
num_computed_tokens = 4
KV exists for p0 p1 p2 o0

preempt
prompt = [p0 p1 p2]       preserved
output = [o0]              preserved
num_computed_tokens = 0    rolled back
active KV = none           released
```

Later replay reconstructs the physical state from the preserved logical history.

---

## 4. Request lifecycle delta

v0.6 adds:

```text
WAITING
   │
   ▼
RUNNING
   │
   ├──────────────► FINISHED
   │
   │ KV pressure
   ▼
PREEMPTED
   │
   │ later eligible
   ▼
RUNNING
```

`Request.mark_preempted()`:

- preserves `prompt_token_ids`;
- preserves `output_token_ids`;
- sets `num_computed_tokens = 0`;
- increments `preemption_count`;
- marks that cache state must be reconstructed.

---

## 5. Scheduler policy

`PreemptiveBudgetedScheduler` extends the v0.4 token-budget scheduler.

Victim selection is intentionally simple:

1. the blocked request cannot be its own victim;
2. only currently `RUNNING` requests are eligible;
3. choose the **most recently admitted** eligible request.

This is deterministic and easy to inspect. It is not presented as production-optimal.

### Anti-thrashing fairness rule

Without a resume guard, two requests can do this forever:

```text
A blocks -> preempt B
next step B returns
B blocks -> preempt A
next step A returns
...
```

Therefore a preempted request is kept at the front of the waiting queue but remains ineligible until at least one request that survived the preemption has finished.

For bounded workloads this creates a simple liveness guarantee while keeping the policy small.

---

## 6. Engine pressure path

The cache still owns storage only. It never chooses a victim.

```text
ScheduleDecision
      │
      ▼
Engine._fit_cache()
      │
      ├── ensure_capacity() succeeds
      │         │
      │         ▼
      │      execute
      │
      └── ensure_capacity() fails
                │
                ▼
     scheduler.select_preemption_victim()
                │
                ▼
          cache.free(victim)
                │
                ▼
          victim.mark_preempted()
                │
                ▼
     scheduler.on_request_preempted()
                │
                ▼
          retry reservation
```

If no eligible victim exists, allocation failure is explicit.

---

## 7. Replay semantics

A sampled output token is logical history even after its KV disappears.

Suppose:

```text
prompt = [p0 p1 p2]
output = [o0]
```

After preemption the runtime may execute:

```text
recompute p0 p1 p2
recompute historical o0
        │
        ▼
only now sample o1
```

The existing `WorkItem.completes_current_sequence` rule naturally prevents duplicate sampling: while replay is still behind `len(prompt + output)`, model execution only rebuilds KV.

---

## 8. Prefix-cache compatibility

When a preempted request returns, Engine calls `cache.prepare_request(...)` before replay.

Therefore some physical state may be restored immediately:

```text
released active request KV
        │
        ▼
shared/cache-resident immutable prefix still exists
        │
        ▼
reattach cached full blocks
        │
        ▼
recompute only the missing suffix/history
```

Preempting B does not invalidate A's reference to a shared prefix block:

```text
A + B: refcount = 2
preempt B
A only: refcount = 1
```

---

## 9. Trace additions

`EngineStepOutput` adds:

```python
preempted_request_ids: tuple[str, ...]
```

Requests expose:

```python
preemption_count
```

This makes pressure behavior visible without inspecting private scheduler state.

---

## 10. Important source files

```text
mini_vllm/core/request.py
        ↓
mini_vllm/scheduler/base.py
        ↓
mini_vllm/scheduler/preemptive.py
        ↓
mini_vllm/core/engine.py
        ↓
mini_vllm/versions/v0_6/engine.py
        ↓
tests/test_preemption.py
        ↓
examples/07_preemption.py
```

Recommended comparison:

```bash
git diff v0.5..v0.6 -- mini_vllm tests examples version.md
```

---

## 11. Tests and validation

v0.6 validates:

- deterministic latest-admitted victim selection;
- a victim is not immediately re-admitted before a survivor completes;
- active KV ownership is released on preemption;
- `num_computed_tokens` rolls back to match discarded state;
- logical output tokens survive preemption;
- replay later catches up and finishes;
- pressured and roomy deterministic runs generate identical output histories;
- preempting one shared-prefix owner does not mutate the other owner's K/V;
- impossible allocation with no eligible victim raises explicitly;
- all v0.1-v0.5 regressions continue to pass.

The frozen local snapshot passes **22 tests** plus all seven executable examples.

---

## 12. Learning goals

After v0.6, the learner should be able to explain:

- why allocation failure belongs in the scheduler/engine feedback loop;
- why the cache must not secretly choose an active victim;
- the difference between logical token history and physical KV state;
- why `num_computed_tokens` must roll back after state invalidation;
- how historical sampled tokens can be replayed without being sampled twice;
- how prefix caching reduces recomputation after preemption;
- how naive immediate resume can cause preemption thrashing;
- why recompute preemption is conceptually simpler than KV swap.

---

## 13. Explicit non-goals

- CPU/host KV swap;
- sophisticated priority/aging policy;
- production admission control;
- speculative decoding;
- multiprocess Executor / Worker split;
- distributed execution;
- CUDA/Triton PagedAttention.

---

## 14. Next version

`v0.7` introduces **speculative decoding** and makes KV mutation transactional:

```text
checkpoint
   ↓
draft tokens
   ↓
target verification
   ↓
accept prefix
rollback rejected suffix
   ↓
commit
```

v0.6 is the prerequisite because it already establishes explicit KV invalidation and reconstruction semantics.

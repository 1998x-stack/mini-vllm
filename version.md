# mini-vLLM v0.4 — active planning branch

## 1. Version identity

- **Branch:** `v0.4`
- **Base:** latest validated `v0.3` / `main`
- **Previous version:** `v0.3`
- **Theme:** token-budget scheduling and chunked prefill
- **Status:** planning / implementation target; not yet released

The v0.4 branch starts from the complete v0.3 runtime. Until the mechanisms below pass their acceptance criteria, the executable serving behavior intentionally remains v0.3 behavior.

---

## 2. Why v0.4 follows v0.3

v0.2 teaches that active **sequence slots** can be refilled continuously.

v0.3 teaches that KV memory can grow in fixed-size physical blocks.

Those mechanisms expose the next bottleneck: sequence count alone does not describe the amount of work scheduled in one iteration.

For example:

```text
Request A: decode 1 token
Request B: prefill 2,000 tokens
```

Both consume one sequence slot, but their compute cost and latency impact are radically different.

v0.4 therefore changes the scheduling unit from:

```text
"How many requests fit?"
```

toward:

```text
"How many tokens of useful model work fit in this engine step?"
```

---

## 3. Primary mental model

```text
step token budget = max_num_batched_tokens

running/waiting requests
          │
          ▼
 budget-aware scheduler
          │
   ┌──────┼──────────────┐
   ▼      ▼              ▼
decode  prefill chunk  prefill chunk
  1        64              31
   └──────┴──────────────┘
             │
             ▼
      total = 96 <= budget
```

A long prompt becomes a sequence of schedulable chunks rather than an all-or-nothing prefill operation.

---

## 4. Required delta from v0.3

### 4.1 Configuration

Add an explicit serving token budget, conceptually:

```text
max_num_batched_tokens
```

Do not smuggle this policy into the model or KV backend.

### 4.2 Work representation

`WorkItem` must be able to represent an arbitrary contiguous range of uncomputed prompt tokens rather than assuming prefill means "all remaining prompt tokens".

The invariant remains:

```text
start_pos == request.num_computed_tokens
end_pos   == start_pos + num_tokens
```

### 4.3 Scheduler

Introduce a budget-aware continuous scheduler that:

- removes finished requests;
- admits eligible waiting requests;
- computes remaining token budget;
- schedules decode work and prompt chunks without exceeding budget;
- preserves deterministic, explainable policy ordering.

Start with a simple FCFS/decode-aware educational policy before adding sophisticated heuristics.

### 4.4 Paged KV compatibility

Partial prefill must grow paged KV capacity incrementally. `PagedKVCache` should remain a storage mechanism, not become the scheduler.

### 4.5 Metrics / traces

Expose enough step metadata to reason about:

- scheduled token count;
- prefill chunk sizes;
- time-to-first-token proxy in engine steps;
- inter-token-latency proxy in engine steps.

---

## 5. Mechanisms intentionally unchanged at the start

The first v0.4 increment should keep these stable unless a contract change is clearly required:

- `Request` lifecycle;
- `PagedKVCache` block-table semantics;
- Tiny Decoder model math;
- sampler semantics;
- synchronous `Engine` coordination model.

The purpose is to isolate scheduler/work-contract evolution.

---

## 6. Acceptance criteria

v0.4 cannot be frozen until all applicable items are true:

- [ ] each step satisfies `sum(work_item.num_tokens) <= max_num_batched_tokens`;
- [ ] a prompt longer than one-step budget prefills across multiple steps;
- [ ] decode can make progress while another request is still in chunked prefill;
- [ ] paged KV capacity grows correctly across chunks;
- [ ] chunked and unchunked reference execution produce equivalent logits/token history for deterministic sampling;
- [ ] error behavior is explicit when one schedulable unit cannot fit policy constraints;
- [ ] timeline examples make budget allocation visible;
- [ ] tests demonstrate the budget invariant rather than only checking final output;
- [ ] docs describe scheduling policy and simplifications;
- [ ] validated implementation is synchronized to `main`;
- [ ] `version.md` is updated from planning status to frozen status only after validation.

---

## 7. Important implementation rule

Do **not** solve v0.4 by adding version-specific branches throughout `Engine`.

Preferred shape:

```text
Request
   │
   ▼
BudgetedContinuousScheduler
   │
   ▼
ScheduleDecision[WorkItem chunks]
   │
   ├────────► PagedKVCache.ensure_capacity()
   │
   └────────► ModelRunner.execute()
```

The token budget belongs to scheduler policy and the scheduler/work contract.

---

## 8. Learning goals

After v0.4, the learner should be able to explain:

- why `max_num_seqs` and `max_num_batched_tokens` constrain different resources;
- why a huge prefill can hurt decode latency;
- how chunked prefill changes iteration scheduling;
- why decode and prefill have different token shapes/workloads;
- why paged KV allocation naturally supports incremental prompt growth;
- how scheduling policy creates TTFT/ITL trade-offs.

---

## 9. Explicit non-goals

- prefix caching — `v0.5`;
- KV-pressure preemption — `v0.6`;
- speculative decoding — `v0.7`;
- Executor / Worker process boundary — `v0.8`;
- optimized Triton/CUDA PagedAttention — later performance version.

---

## 10. Next version

`v0.5` introduces **prefix caching and shared block ownership**. That version will require new invariants around block immutability, reference counts, shared-prefix lookup, mutable tails, and safe eviction.

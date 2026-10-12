# mini-vLLM v0.4 — token-budget scheduling + chunked prefill

## 1. Version identity

- **Branch:** `v0.4`
- **Previous version:** `v0.3`
- **Theme:** token-budget scheduling and chunked prefill
- **Status:** frozen historical learning snapshot after validation

v0.4 changes the scheduling unit from only **sequence slots** to explicit **token work per engine step**.

```text
v0.3
max_num_seqs
     ↓
which requests run?

v0.4
max_num_seqs + max_num_batched_tokens
     ↓                    ↓
which requests?      how much token work?
```

## 2. Why this version follows v0.3

v0.2 made sequence slots refillable. v0.3 made KV growth incremental through physical blocks. The next hidden bottleneck is that two admitted requests can represent radically different work:

```text
A: decode 1 token
B: prefill 2,000 tokens
```

A sequence-count limit cannot express this difference. v0.4 introduces a per-step token budget and makes long prompt prefill interruptible across engine steps.

## 3. Core mechanism

The reference scheduling policy is intentionally simple and deterministic:

1. remove finished requests;
2. admit waiting requests until `max_num_seqs` is full;
3. schedule decode requests first, one token each;
4. spend remaining token budget on prefill requests in FCFS order;
5. truncate a prefill work item at the remaining budget boundary.

The invariant is:

```text
sum(work_item.num_tokens) <= max_num_batched_tokens
```

## 4. Critical semantic change: compute is not sample

A partial prefill chunk writes valid K/V, but it must **not** sample an output token.

```text
prompt = [t0 t1 t2 t3 t4]
budget = 2

step 0: compute [t0 t1]       -> KV grows, no sample
step 1: compute [t2 t3]       -> KV grows, no sample
step 2: compute [t4]          -> prompt complete, sample first output
step 3: compute sampled token -> decode, sample next output
```

`WorkItem.completes_current_sequence` makes this boundary explicit. A sampled output token is not counted as computed until a later decode forward pass consumes it.

## 5. Main implementation delta from v0.3

### Configuration

`EngineConfig` adds:

```python
max_num_batched_tokens: int = 16
```

### Scheduler/work contract

- `make_work_item(..., max_tokens=N)` creates an arbitrary contiguous prompt chunk.
- `BudgetedContinuousScheduler` owns token-budget policy.
- `WorkItem.phase` distinguishes `prefill` and `decode` for traces.
- `EngineStepOutput` exposes scheduled-token, prefill-token and decode-token counts.

### Engine

The Engine samples only if a `WorkItem` computed every token that currently exists for that request.

### KV cache

`PagedKVCache` remains storage, not scheduling policy. Each chunk calls `ensure_capacity(request_id, chunk_end_pos)`, so block allocation grows incrementally.

## 6. Important source files

```text
mini_vllm/config.py
        ↓
mini_vllm/core/work.py
        ↓
mini_vllm/scheduler/base.py
        ↓
mini_vllm/scheduler/budgeted.py
        ↓
mini_vllm/core/engine.py
        ↓
mini_vllm/versions/v0_4/engine.py
        ↓
examples/05_chunked_prefill.py
```

Compare directly:

```bash
git diff v0.3..v0.4 -- mini_vllm tests examples version.md
```

## 7. Tests and validation

v0.4 validates:

- scheduled token count never exceeds budget;
- a long prompt crosses multiple engine steps;
- partial prefill chunks do not sample early;
- decode runs before and alongside long prefill;
- paged KV blocks grow incrementally with chunks;
- chunked and unchunked deterministic execution generate identical output tokens;
- invalid zero budgets/chunk sizes fail explicitly.

The frozen local snapshot passes the full repository check with **14 tests**.

## 8. Learning goals

After v0.4, the learner should be able to explain:

- why `max_num_seqs` and `max_num_batched_tokens` constrain different resources;
- why all-at-once long prefill can hurt inter-token latency;
- how chunked prefill allows decode and prefill to coexist;
- why partial prefill must not sample;
- why paged KV naturally supports incremental prompt growth;
- how scheduler policy creates TTFT/ITL trade-offs.

## 9. Explicit non-goals

- prefix caching / shared KV ownership;
- KV-pressure preemption;
- speculative decoding;
- multiprocess Executor / Worker separation;
- CUDA/Triton PagedAttention.

## 10. Next version

`v0.5` introduces **prefix caching and shared block ownership**:

```text
identical prompt prefix
        ↓
content-addressed full KV blocks
        ↓
shared ownership / refcount
        ↓
immutable shared prefix + mutable tail
        ↓
safe release / eviction
```

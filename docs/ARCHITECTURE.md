# Architecture notes

## Computed-token invariant

For every request:

```text
0 <= num_computed_tokens <= len(prompt_token_ids + output_token_ids)
```

Every cache position `< num_computed_tokens` has passed through the model and has materialized K/V. A sampled output token may already exist in `output_token_ids` while remaining uncomputed until the next decode step.

## v0.4 engine step

```text
BudgetedContinuousScheduler.schedule()
        │
        │ token budget invariant
        ▼
ScheduleDecision[WorkItem chunks]
        │
        ▼
PagedKVCache.ensure_capacity(item.end_pos)
        │
        ▼
ModelRunner.execute(work_items)
        │
        ▼
advance num_computed_tokens
        │
        ├── partial prefill? ── yes ──► stop for this request
        │
        └── current sequence fully computed
                         │
                         ▼
                       sample
                         │
                         ▼
                 append output token
```

## Scheduler policy vs storage mechanism

The v0.4 scheduler owns request admission, decode-first ordering, per-step token budget, and prompt chunk length.

The paged KV backend owns physical block allocation, block tables, logical-position translation, and block recycling.

Neither subsystem owns the other's policy.

## Decode-first reference policy

Given budget `B`:

1. schedule each runnable decode request for one token in running-list order;
2. give remaining budget to prefill requests in FCFS order;
3. split a prompt at the remaining-budget boundary.

This is intentionally not claimed as an optimal production policy. It is a compact reference mechanism that makes the TTFT/ITL trade-off inspectable.

## Why partial prefill cannot sample

Suppose a five-token prompt is chunked as `2 + 2 + 1`. Logits from the first two chunks describe positions internal to the prompt; sampling them would invent output tokens before the prompt was fully consumed.

```text
partial prefill -> KV side effect only
final prefill   -> KV + first sample
decode          -> KV + next sample
```

`WorkItem.completes_current_sequence` expresses this boundary.

## Paged KV under chunked prefill

For each work item:

```text
ensure_capacity(request_id, item.end_pos)
```

With block size 2 and chunk ends `2, 4, 5`:

```text
end=2 -> 1 physical block
end=4 -> 2 physical blocks
end=5 -> 3 physical blocks
```

The address invariant remains:

```text
logical_block = position // block_size
block_offset  = position % block_size
physical      = block_table[request_id][logical_block]
```

## Trace model

`EngineStepOutput.work` records `request_id`, `phase`, `start_pos`, `end_pos`, and `num_tokens`. Derived properties expose total scheduled, prefill, and decode token counts.

## Extension seams

```text
Scheduler
KVCacheBackend
ModelRunner
Sampler
Engine
```

v0.5 should extend KV ownership/indexing for prefix sharing instead of embedding prefix-cache rules inside the model or scheduler.

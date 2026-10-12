# Architecture notes

## Core invariant

For every request:

```text
0 <= num_computed_tokens <= len(prompt_token_ids + output_token_ids)
```

Every cache position `< num_computed_tokens` has been written by the model. The newest sampled output may exist in `output_token_ids` while remaining uncomputed until the next decode iteration.

## Engine step

```text
scheduler.schedule()
        │
        ▼
ScheduleDecision
        │
        ▼
cache.ensure_capacity()
        │
        ▼
model_runner.execute()
        │
        ▼
logits per request
        │
        ▼
sampler.sample()
        │
        ▼
append output token
        │
   ┌────┴─────┐
   │ finished?│
   └────┬─────┘
      yes│ no
         │
cache.free()  next iteration
```

## Prefill vs decode

For v0.1-v0.3, prefill work is all remaining prompt tokens and decode work is exactly the newly sampled token that has not yet been computed. v0.4 will make prompt work chunkable under a step-level token budget.

## Why cache is below the runner

The model's attention algorithm should not care whether logical positions are stored in a contiguous per-request slab or a future physical block pool. It consumes a logical storage contract.

## Extension seams

```text
Scheduler
KVCacheBackend
ModelRunner
Sampler
Engine
```

Future versions should add strategies or richer contracts at these seams rather than scattering feature-specific conditionals through the engine.

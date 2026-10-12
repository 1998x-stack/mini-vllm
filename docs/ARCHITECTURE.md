# Architecture notes

## Core invariant

```text
0 <= num_computed_tokens <= len(prompt_token_ids + output_token_ids)
```

Every cache position below `num_computed_tokens` has been written by the model. A sampled output token may exist while still uncomputed until the next decode iteration.

## Engine step

```text
scheduler.schedule()
        ↓
ScheduleDecision
        ↓
cache.ensure_capacity()
        ↓
model_runner.execute()
        ↓
sampler.sample()
        ↓
append output
        ↓
finish/free or next iteration
```

## Paged cache

```text
free_blocks: deque[int]
block_tables: dict[request_id, list[physical_block_id]]
```

For logical token position `p`:

```text
logical_block = p // block_size
block_offset  = p % block_size
physical      = block_table[logical_block]
```

The reference `read_prefix()` gathers paged K/V before ordinary PyTorch attention. A future optimized kernel may fuse the translation with attention, but must preserve identical semantics.

## Extension seams

`Scheduler`, `KVCacheBackend`, `ModelRunner`, `Sampler`, and `Engine` are the stable boundaries. Future versions should evolve these contracts deliberately instead of scattering version conditionals.

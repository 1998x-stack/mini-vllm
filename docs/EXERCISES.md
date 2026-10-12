# Exercises

## Inspect paged KV

1. Change `block_size` from 2 to 16 and explain internal fragmentation.
2. Print `block_table(request_id)` after every decode step and observe when a new block is allocated.
3. Compare reserved token slots in contiguous versus paged cache.
4. Force a tiny physical block pool and observe allocation failure.

## Extend

1. Add cache-allocation failure metrics.
2. Add a RoundRobin scheduler without changing cache/model code.
3. Design v0.4's `BudgetedContinuousScheduler` with `sum(work_item.num_tokens) <= max_num_batched_tokens`.
4. Explain what new ownership/refcount rules v0.5 prefix caching will require.

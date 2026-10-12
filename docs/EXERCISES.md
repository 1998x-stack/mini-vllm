# Exercises

## v0.1-v0.5

Use historical branch diffs to explain static vs continuous batching, paged addressing, chunked prefill, and shared prefix ownership before modifying v0.6.

## v0.6 preemption

1. Draw the state transition `RUNNING -> PREEMPTED -> RUNNING -> FINISHED`. Which fields are logical history and which represent physical execution state?
2. Explain why preserving `output_token_ids` while resetting `num_computed_tokens` is correct.
3. Remove the anti-thrashing resume barrier and construct a two-request workload that can repeatedly discard each other's progress.
4. Change victim selection from latest-admitted to largest-KV-owner. What new cache introspection would the scheduler need?
5. In `test_preempting_one_owner_does_not_corrupt_shared_prefix`, explain why the shared physical block refcount becomes 1 rather than 0.
6. Add a metric for recomputed tokens and compare total model work between roomy and pressured cache configurations.
7. Try a policy that preempts the blocked request itself. Under what workloads does that improve or worsen liveness?
8. Explain why a cache backend must report allocation failure rather than silently evict an active request's block.

## Preview v0.7

Design speculative decoding as a transaction:

```text
checkpoint KV frontier
        ↓
append draft tokens
        ↓
verify with target model
        ↓
accept prefix / reject suffix
        ↓
commit accepted state
rollback rejected state
```

List which v0.6 invalidation/replay concepts can be reused for rollback semantics.

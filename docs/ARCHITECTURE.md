# Architecture notes

## Stable computed-token invariant

```text
0 <= num_computed_tokens <= len(prompt_token_ids + output_token_ids)
```

Every position below `num_computed_tokens` must have valid, addressable K/V. A sampled or bonus output token may be logically known while remaining physically uncomputed until a later forward pass consumes it.

## Evolution through v0.7

```text
v0.3  block-addressed KV
v0.4  partial token work
v0.5  immutable shared prefixes
v0.6  invalidate/recompute whole active request state
v0.7  checkpoint/rollback a temporary request-private suffix
```

v0.6 rollback is coarse: preemption discards active execution state and later replays logical history. v0.7 rollback is fine-grained: a speculative transaction retains the verified prefix and removes only rejected temporary positions.

## Speculative execution path

```text
normal scheduled WorkItem
        │
        ▼
reach sampling frontier
        │
        ▼
DraftProvider.propose()
        │
        ▼
reserve optional temporary KV
        │
        ▼
cache.checkpoint()
        │
        ▼
ModelRunner.execute_speculative()
        │
        ▼
per-position target logits
        │
        ▼
longest matching draft prefix
        │
   ┌────┴───────────────┐
   ▼                    ▼
accepted             rejected
retain KV            rollback KV
   │                    │
   └─────────┬──────────┘
             ▼
commit accepted tokens + target fallback/bonus
```

## Why verification needs per-position logits

For draft `[d0, d1, d2]`, target verification uses shifted distributions:

```text
stable logits          predict d0
logits after d0        predict d1
logits after d1        predict d2
logits after d2        predict bonus
```

The normal v0.1-v0.6 runner only needs final logits. v0.7 adds a reference trace API that returns logits after every computed draft token while preserving the old final-logit path.

## Cache checkpoint

A checkpoint records:

```text
request_id
stable token frontier
stable block-table identity
```

Temporary speculative allocation may extend the block table, but rollback is append-only. It cannot cross the checkpoint token frontier and cannot free cached/shared blocks.

## Shared-prefix safety

```text
[ immutable shared prefix ] [ request-private speculative tail ]
             ▲                          ▲
      outside transaction          may rollback
```

This is why v0.5 block ownership is a prerequisite for v0.7: transaction code can tell which state is legal to mutate.

## Fallback and bonus semantics

On mismatch, target fallback token is committed logically but not computed physically. When all draft candidates match, the logits after the final accepted candidate produce one target bonus token with the same pending-token behavior.

## Scheduler remains unchanged

The first v0.7 increment deliberately keeps `PreemptiveBudgetedScheduler` unchanged. Draft/verification cost is not yet part of the step token budget. Correct transaction semantics are the reference oracle; later versions may integrate speculative cost into policy.

# Architecture notes

## Stable runtime invariant

```text
0 <= num_computed_tokens <= len(prompt_token_ids + output_token_ids)
```

Every position below `num_computed_tokens` has materialized K/V. A sampled output token remains uncomputed until a later decode forward pass consumes it.

## v0.5 request admission

```text
Engine.add_request(request)
        │
        ▼
cache.prepare_request(prompt tokens)
        │
        ├── prefix miss -> 0 cached tokens
        │
        └── prefix hit  -> attach shared physical blocks
                           increment active refcounts
        │
        ▼
request.num_computed_tokens = cached_tokens
        │
        ▼
Scheduler
```

Prefix caching therefore changes initial computed state without changing scheduler policy.

## Content-addressed full blocks

For block size `B`, only complete prompt blocks are indexable.

A block hash is chained:

```text
h0 = H(""  || tokens[0:B])
h1 = H(h0  || tokens[B:2B])
h2 = H(h1  || tokens[2B:3B])
```

This makes the hash represent both local token content and prior prefix context.

## Ownership model

Physical blocks have two independent concepts:

1. **active refcount** — how many request block tables currently point to the block;
2. **cache residency** — whether the prefix index retains the block for future reuse.

Therefore:

```text
refcount > 0, cached     active shared/resident block
refcount = 0, cached     reusable resident block; eviction candidate
refcount = 0, not cached free block
```

## Immutable prefix / private tail

```text
logical blocks:   0       1       2
                full    full    partial
                  │       │       │
                  ▼       ▼       ▼
               shared  shared   private
               cached  cached   writable
```

Appending a token never writes into a cached shared full block. The next logical block is private, which is the educational copy-on-append boundary.

## Publishing a new block

After a model WorkItem has executed all layers:

```text
ModelRunner.execute()
        │
        ▼
cache.commit_work()
        │
        ├── incomplete prompt block -> remain private
        │
        └── completed full prompt block
                │
                ├── hash absent -> publish current physical block
                └── hash exists -> switch block table to canonical block
                                   release duplicate private block
```

Publishing after execution avoids exposing a block before all layers contain valid K/V.

## Why exact full-prompt hits still compute

K/V is sufficient for attention history but not for next-token sampling. The runtime needs logits from the final prompt position.

v0.5 therefore never marks every prompt token as computed solely from prefix-cache lookup. Reuse remains block-granular and leaves final prompt work to execute.

## Eviction vs preemption

v0.5 may evict only:

```text
cached && refcount == 0
```

It never steals an actively referenced block. If all memory is actively owned, allocation fails. Deciding which runnable request should be preempted and recomputed is intentionally deferred to v0.6.

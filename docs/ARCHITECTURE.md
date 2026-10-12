# Architecture notes

## Stable computed-token invariant

```text
0 <= num_computed_tokens <= len(prompt_token_ids + output_token_ids)
```

Every position below `num_computed_tokens` must have valid, addressable K/V.
A sampled output token remains logically known but physically uncomputed until a later forward pass consumes it.

## v0.6 adds invalidation

Before v0.6, computed state only moved forward. Preemption adds an explicit rollback of **physical execution state**:

```text
logical token history: prompt + output_token_ids     preserved
physical KV frontier: num_computed_tokens             rolled back
```

This distinction is what makes recomputation possible without changing generated output history.

## Pressure feedback path

```text
Scheduler.schedule()
       │
       ▼
ScheduleDecision
       │
       ▼
Engine asks cache.ensure_capacity()
       │
       ├── success -> WorkItem executes
       │
       └── failure
             │
             ▼
Scheduler.select_preemption_victim(blocked_request)
             │
             ▼
Engine releases victim active KV
             │
             ▼
Request.mark_preempted()
             │
             ▼
Scheduler.on_request_preempted()
```

The cache reports capacity failure but never decides which active request to sacrifice.

## Victim policy

The educational policy is deterministic:

1. exclude the request whose allocation is currently blocked;
2. among remaining active requests, choose the most recently admitted one;
3. never choose finished/preempted requests.

This is not claimed to be throughput-optimal. Its purpose is to make policy inspectable.

## Anti-thrashing / fairness guard

Immediately re-admitting a victim can produce:

```text
A preempts B
B immediately returns
B preempts A
A immediately returns
...
```

v0.6 therefore records a resume barrier. A preempted request becomes eligible only after at least one request that remained active has finished. It is kept at the front of the waiting queue, so once eligible it resumes before later normal arrivals.

For bounded workloads this provides a simple liveness story without introducing priorities or aging scores.

## Recompute replay

A victim keeps:

```text
prompt_token_ids
output_token_ids
sampling_params
```

but loses active KV and sets:

```text
num_computed_tokens = 0
cache_rebuild_required = True
```

When re-admitted:

```text
cache.prepare_request()
        │
        ├── reattach still-resident immutable prefix blocks
        └── return cached-token frontier
        │
        ▼
replay remaining prompt/history tokens
```

Historical output tokens are part of `all_token_ids`. During replay, they are fed through the model but are **not sampled again**. Sampling resumes only once execution catches up to the end of the existing logical history.

## Prefix-cache compatibility

Preemption calls the same `cache.free(request_id)` ownership path used by normal completion.
For shared blocks:

```text
A + B -> refcount 2
preempt B -> refcount 1
A still owns and reads exactly the same physical block
```

Cached blocks with refcount 0 may remain resident or be evicted under pressure; active blocks referenced by another request remain protected.

## Why v0.6 uses recompute instead of swap

Recompute exposes the minimum required semantics:

```text
invalidate physical state
preserve logical history
reconstruct state later
```

Host/CPU KV swap adds another storage tier and transfer policy without teaching a more fundamental ownership rule, so it stays out of scope.

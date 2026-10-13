# mini-vLLM — Architecture-Driven Development Plan

`main` is always the newest validated runtime. Version branches are educational snapshots. Exact release commits must pass repository CI.

## Version ladder

```text
v0.1  static batching + contiguous KV
v0.2  continuous batching
v0.3  paged KV
v0.4  token budget + chunked prefill
v0.5  prefix cache + shared ownership
v0.6  preemption + recompute
v0.7  speculative transactions
v0.8  EngineCore / Executor / Worker
v0.9  WorkerGroup + reference tensor parallel
v0.10 async streaming/cancellation/backpressure
v0.11 compile-aware execution
v0.12 optimized PagedAttention
v0.13 prefix-aware scheduling                          ← current implementation
v1.0  integrated educational runtime                  ← next
```

## v0.13 lesson

Cache reuse has three distinct meanings that must not collapse into one another:

```text
cache fact             policy                    execution truth
PrefixReuseHint   →    AdmissionCost      →      prepare_request()
                        + fairness                actual attached KV
```

A side-effect-free hint may reduce expected admission cost, but it cannot reserve blocks or claim computed tokens. v0.13 therefore delays first cache attachment until actual admission, then reconciles the Scheduler's estimate against the cache's real reusable frontier.

Core equation:

```text
expected_new_prefill = raw_prefill - estimated_reusable_prefix
```

This value ranks waiting work only. The normal token budget continues to count actual scheduled model tokens.

Fairness is explicit: requests accumulate bypass counts when warmer work is admitted ahead of them; a bounded threshold gives the older/cold request protection against indefinite starvation.

Compatibility rule: preemption, speculative transactions, distribution, async delivery, compile-aware projection, and vectorized PagedAttention keep their existing ownership boundaries.

Release rule: only an exact v0.13 commit with a green repository CI run may be synchronized to `main`.

## Next

- v1.0: consolidate the full educational runtime, public API, observability, documentation, and integrated end-to-end scenarios without introducing another major serving mechanism.

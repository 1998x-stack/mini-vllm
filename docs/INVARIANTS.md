# mini-vLLM v1.0 invariant contract

v1.0 integrates the mechanisms introduced from v0.1 through v0.13. These invariants are the stable correctness contract across the complete runtime.

## Logical token state

1. `num_computed_tokens` counts only tokens whose model work executed and whose K/V remains valid and addressable.
2. A sampled output token is logical history, not computed history, until a later forward pass executes it.
3. Preemption may preserve logical token history while rolling the computed frontier backward.

## KV ownership and addressing

4. Physical ownership/reservation exists before any K/V read or write.
5. Paged translation is authoritative:

```text
logical_block = position // block_size
block_offset  = position % block_size
physical      = block_table[logical_block]
```

6. Attention reads only logical positions `[0, length)`; unused physical tail slots are never visible.
7. Shared prefix blocks are immutable while referenced.
8. Finished or cancelled requests release active ownership exactly once.

## Prefix reuse and scheduling

9. `PrefixReuseHint` is advisory. It is not a reservation, block-table entry, refcount, or computed-token claim.
10. Side-effect-free prefix estimation cannot create request ownership or change LRU/refcount state.
11. Only authoritative `prepare_request()` establishes the actual reusable computed frontier.
12. Stale hints may reduce scheduling quality but cannot change serving correctness.
13. Warm-prefix preference has deterministic tie-breaking and explicit starvation protection.

## Preemption and transactions

14. A preempted request cannot claim KV that was released.
15. Recompute replays known prompt/output history without resampling historical tokens.
16. Rejected speculative state disappears from both logical history and physical KV state.
17. Speculative rollback never mutates a shared immutable prefix.

## Execution and distribution

18. EngineCore owns logical commit decisions; Executor/Worker performs physical execution only.
19. Distributed ranks must agree on replicated cache capacity, checkpoints, and policy-visible prefix facts.
20. A rank failure cannot leave another rank's speculative or ordinary execution partially committed.
21. Tensor-parallel gathered logits are equivalent to the readable unsharded reference for the supported educational model.

## Optimization correctness

22. Vectorized PagedAttention is numerically equivalent to the readable reference attention path.
23. Compile-aware LM-head projection is numerically equivalent to eager projection.
24. Optimization fallback may change execution mode, never serving semantics.
25. No speedup claim is valid without measurement; correctness does not imply performance.

## Async delivery

26. Async clients observe committed tokens only.
27. Each request exposes exactly one terminal outcome: finished, cancelled, or error.
28. Backpressure can delay delivery but cannot cause hidden logical commits or duplicate tokens.
29. Cancellation is applied between atomic EngineCore steps.

## Observability

30. `snapshot()` is read-only: observing the runtime cannot schedule work, attach KV, mutate cache recency, or advance request state.
31. Public snapshots report mechanism facts and decisions without exposing physical ownership handles as policy inputs.
32. The same deterministic configuration must produce the same committed output in v1.0 as validated v0.13.

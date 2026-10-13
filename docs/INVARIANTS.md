# mini-vLLM v1.1 invariant contract

v1.1 preserves the v1.0 serving semantics while turning the Executor/Worker boundary into a real operating-system process boundary. These invariants are the stable correctness contract across the complete runtime.

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
9. In v1.1 the child process owns its physical KV/cache structures; the parent may not use block tables/refcounts/cache objects as remote ownership handles.

## Prefix reuse and scheduling

10. `PrefixReuseHint` is advisory. It is not a reservation, block-table entry, refcount, or computed-token claim.
11. Side-effect-free prefix estimation cannot create request ownership or change LRU/refcount state.
12. Only authoritative `prepare_request()` establishes the actual reusable computed frontier.
13. Stale hints may reduce scheduling quality but cannot change serving correctness.
14. Warm-prefix preference has deterministic tie-breaking and explicit starvation protection.

## Preemption and transactions

15. A preempted request cannot claim KV that was released.
16. Recompute replays known prompt/output history without resampling historical tokens.
17. Rejected speculative state disappears from both logical history and physical KV state.
18. Speculative rollback never mutates a shared immutable prefix.
19. A transport-safe checkpoint handle identifies child-owned transaction state; it is not itself physical cache ownership.

## Execution and distribution

20. EngineCore owns logical commit decisions; Executor/Worker performs physical execution only.
21. Replicated ranks must agree on cache capacity, checkpoints, and policy-visible prefix facts.
22. A rank failure cannot leave another reachable rank's speculative or ordinary execution knowingly committed while EngineCore rejects the step.
23. Tensor-parallel gathered logits are equivalent to the readable unsharded reference for the supported educational model.
24. Deterministic world-size 1 and 2 v1.1 output is equivalent to v1.0 under equivalent configuration.

## Multiprocessing transport

25. IPC messages contain serializable logical values/copies, not live parent references to child Workers, caches, model runners, block tables, refcounts or tensor storage.
26. Every command/result exchange has a correlation ID. A response with the wrong ID is not eligible for logical commit.
27. Sending a command is not evidence that physical execution succeeded. Logical commit requires an acknowledged successful response.
28. Child-reported execution/model errors and transport/process-death failures are distinct failure classes.
29. A child-reported error may preserve worker health only when the process remains reachable and required rollback succeeds.
30. Unexpected EOF, timeout, process exit, or unprovable physical consistency makes the relevant executor unhealthy; EngineCore may not guess and commit.
31. After failed ordinary execution, rollback is attempted on every reachable checkpointed rank/request before the failure is surfaced.
32. Parent-side checkpoint handles do not expose child physical block identity.
33. Normal shutdown sends an explicit shutdown intent and joins child processes. No child worker may remain alive after successful `close()`/`aclose()`.
34. Termination is a cleanup fallback, not the normal request-cancellation mechanism.

## Optimization correctness

35. Vectorized PagedAttention is numerically equivalent to the readable reference attention path.
36. Compile-aware LM-head projection is numerically equivalent to eager projection.
37. Optimization fallback may change execution mode, never serving semantics.
38. No speedup claim is valid without measurement; multiprocessing isolation is not itself a performance claim.

## Async delivery

39. Async clients observe committed tokens only.
40. Each request exposes exactly one terminal outcome: finished, cancelled, or error.
41. Backpressure can delay delivery but cannot cause hidden logical commits or duplicate tokens.
42. Cancellation is applied between atomic EngineCore steps; it does not arbitrarily kill a healthy child worker.

## Observability

43. `snapshot()` is read-only: observing the runtime cannot schedule work, attach KV, mutate cache recency, advance request state, or change serving-command counters.
44. Public snapshots report mechanism/transport facts without exposing physical ownership handles as policy inputs.
45. Snapshot transport reads such as execution observations and cache statistics are non-serving reads and must not make two otherwise identical snapshots differ.
46. The same deterministic v1.0 configuration remains the semantic oracle for v1.1 committed output.

# mini-vLLM v1.3 invariant contract

v1.3 preserves all v1.2 serving/artifact/process semantics and adds the trust/lifecycle boundary from network bytes to validated API work and committed response bytes.

## Logical token state

1. `num_computed_tokens` counts only tokens whose model work executed and whose K/V remains valid and addressable.
2. A sampled output token is logical history, not computed history, until a later forward executes it.
3. Preemption may preserve logical history while rolling the computed frontier backward.
4. Text becomes token IDs before entering Scheduler/KV/WorkItem/model-runner contracts.

## KV ownership and addressing

5. Physical ownership/reservation exists before any K/V read or write.
6. Paged translation is authoritative:

```text
logical_block = position // block_size
block_offset  = position % block_size
physical      = block_table[logical_block]
```

7. Attention reads only logical positions `[0, length)`.
8. Shared prefix blocks are immutable while referenced.
9. Finished/cancelled/aborted requests release active ownership exactly once.
10. The network layer never owns or exposes physical block identity.

## Prefix reuse / scheduling

11. `PrefixReuseHint` is advisory, not reservation/ownership/computed state.
12. Side-effect-free prefix estimation cannot create request ownership.
13. Only authoritative cache preparation establishes reusable computed state.
14. Warm-prefix scheduling remains deterministic with starvation protection.
15. Network ingress pressure is not a Scheduler priority/cost signal.

## Preemption and transactions

16. A preempted request cannot claim released KV.
17. Replay preserves known prompt/output history without resampling historical output.
18. Rejected speculative state disappears from logical history and physical KV.
19. Speculative rollback cannot mutate shared immutable prefixes.
20. A transport-safe checkpoint handle is transaction identity, not physical cache ownership.

## Execution / multiprocessing

21. EngineCore owns logical commit decisions; Executor/Worker performs physical execution.
22. Replicated ranks agree on policy-visible cache facts and artifact identity.
23. A failed rank cannot authorize a partial logical commit.
24. IPC messages carry serializable logical/copy values, not live ownership objects.
25. Sending work is not proof of successful execution; acknowledgment is required.
26. Unknown child physical consistency makes the executor unhealthy rather than guessable.
27. Normal owned-engine shutdown reaps every worker process.

## Artifact / tokenizer truth

28. Filesystem presence is not a valid model artifact.
29. Artifact schema/version/architecture/tokenizer compatibility are validated before readiness.
30. External tensor names exactly cover required runtime mappings; missing/extra tensors fail.
31. Tensor shape and supported dtype match before copy.
32. Runtime parameters are explicit owned copies, not aliases to mutable checkpoint containers.
33. Tokenizer vocabulary size and compatibility ID agree with the model manifest.
34. Every ready rank reports the expected model/tokenizer identity.
35. Artifact/model startup failure occurs before request admission.

## Async delivery

36. Async clients observe committed tokens only.
37. Each submitted runtime request has exactly one logical terminal outcome: finished, cancelled or error.
38. Bounded delivery can delay progress but cannot drop/duplicate committed token events.
39. Cancellation is cooperative at EngineCore step boundaries.

## Network parsing / admission

40. Request header bytes are bounded before unbounded header parsing is allowed.
41. Request body bytes are bounded before body allocation/JSON serving work can exceed the configured limit.
42. Unsupported/malformed HTTP method/path/version/header/body semantics fail at the network boundary.
43. Invalid JSON or semantic generation fields fail before `engine.submit()`.
44. A rejected API request creates no hidden pending/core request.
45. The reference server accepts only the documented HTTP/1.1 subset; transfer-encoded/chunked request bodies are not silently interpreted.

## Capacity separation

46. A network ingress slot is not an AsyncEngine event-queue slot.
47. A network ingress slot is not Scheduler admission or Scheduler token budget.
48. A network ingress slot is not KV reservation/ownership.
49. HTTP 429 due to ingress capacity occurs before runtime submission.
50. A full per-request AsyncEngine event queue backpressures production rather than dropping tokens.

## Network output

51. A token crosses the network only after `AsyncRequestHandle` exposes a committed `TokenEvent`.
52. JSON/SSE serialization can delay visibility but cannot expose speculative/uncommitted state.
53. A connected streaming client receives token events in committed order.
54. A submitted connected API request emits at most one network terminal event/result corresponding to its single logical terminal outcome.
55. A disconnected client may prevent terminal bytes from being delivered, but cannot create a second logical terminal outcome.

## Disconnect semantics

56. Client EOF/reset produces cancellation intent through `AsyncRequestHandle.cancel()` when configured.
57. Disconnect does not directly alter block tables, KV storage, Scheduler lists or model state.
58. Disconnect never uses worker-process termination as a request-cancellation mechanism.
59. Duplicate disconnect observations do not create duplicate cancellation intent/counters for the same in-flight request.

## Health / readiness

60. A bound/listening socket alone is not readiness.
61. Network health and serving readiness are distinct facts.
62. v1.3 readiness requires NetworkServer READY state, healthy execution and artifact rank-identity agreement.
63. During draining, health may remain true while readiness is false.
64. If deeper execution health becomes unprovable, the server must not claim ready merely because the listener remains alive.

## Lifecycle / ownership

65. `NetworkServer.close()` first stops new accepts before draining/cancelling active API work.
66. A server wrapping an external engine does not close that engine.
67. A server that created/owns an engine closes and reaps that engine/workers during successful shutdown.
68. Shutdown timeout may cancel stuck network connection tasks; it does not authorize bypassing underlying runtime cleanup for an owned engine.
69. The server must not retain in-flight request ownership after the corresponding connection task has completed.

## Optimization correctness

70. Vectorized PagedAttention remains numerically equivalent to the reference path.
71. Compile-aware vocabulary projection remains equivalent to eager projection.
72. Optimization fallback may change execution mode, never serving semantics.
73. No performance claim follows merely from multiprocessing, vectorization, compilation or exposing HTTP.

## Observability

74. `engine.snapshot()` remains side-effect free.
75. `server.snapshot()` cannot parse/submit requests, consume events, cancel work or mutate counters.
76. Network snapshots expose copied logical facts, never live sockets or physical ownership handles.
77. Reading health/readiness/snapshots cannot itself change Scheduler, cache, request or network lifecycle state.

## Semantic references

78. Equivalent deterministic v1.2 direct serving remains the semantic oracle for v1.3 HTTP generation output.
79. Historical artifact/process/KV/scheduler invariants remain active underneath the network boundary.
80. The complete teaching separation is:

```text
network bytes
!= validated API request
!= token-native Request
!= logical computed frontier
!= physical KV ownership
!= runtime parameter storage
```

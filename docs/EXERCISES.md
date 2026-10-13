# mini-vLLM v1.1 exercises

Use historical branch diffs to learn one mechanism at a time. Use v1.0 to practice cross-layer composition, then v1.1 to reason about what changes when execution crosses a real OS process boundary.

## Follow one request across processes

1. Run `examples/16_multiprocess_transport.py`. Record the parent PID and every worker PID. Why is PID inequality stronger evidence of isolation than a Python class boundary?
2. Draw the path from `submit()` to `TokenEvent`. Mark which arrows stay inside the parent and which cross IPC.
3. Identify the exact point at which a child-computed logit becomes eligible for EngineCore logical commit.
4. Explain why “the parent successfully sent EXECUTE” is not evidence that execution succeeded.

## Protocol and serialization

5. Pick one `TransportCommand` and list every field that must survive pickle serialization.
6. Explain why v1.1 sends a `WorkerSpec` into the child instead of constructing a `Worker` in the parent and treating it as remotely owned.
7. Inspect `TensorPayload`. What information is copied across IPC, and what live tensor/storage identity is deliberately lost?
8. Explain why a correlation-ID mismatch must be treated as a transport failure even if the payload looks numerically plausible.

## Child-owned physical state

9. List the objects constructed inside each child process: model, cache, runner and Worker. Which parent policies are intentionally absent?
10. Explain why a parent-side `CheckpointHandle` is not a physical KV ownership capability.
11. For one paged-cache position, calculate logical block, offset and physical block. Which process is allowed to know the resulting physical block ID?
12. Explain why Scheduler still receives prefix reuse quantities but not child block tables/refcounts.

## Ordinary execution atomicity

13. Trace one ordinary forward: remote checkpoint → rank execution → result validation → gather → checkpoint commit → EngineCore logical commit.
14. Inject a structured child error after checkpoint. Which state must be rolled back before the parent surfaces the error?
15. Suppose rank 0 returns success and rank 1 dies before responding. Why can rank 0's result not be logically committed?
16. Explain the conservative rule: if full physical consistency cannot be proven, mark the executor unhealthy rather than guessing.

## Speculative transactions

17. Run a deterministic v1.1 request that triggers speculation. Find the `VerificationResult` in the public snapshot.
18. Distinguish the outer parent-side logical speculative transaction from each child's private real cache checkpoint.
19. When a speculative suffix is rejected, which candidate tokens remain logical history, and which child-local KV must disappear?
20. Why does process isolation not change the rule that shared prefix blocks are immutable?

## Error taxonomy

21. Trigger `checkpoint("missing", 0)`. Why is the result a `WorkerRemoteError` rather than `WorkerProcessDied`?
22. After that structured error, verify the worker process is still alive. What does that teach about remote exceptions versus transport failures?
23. Kill a worker process and call a remote operation. Why must v1.1 surface `WorkerProcessDied` explicitly?
24. Compare timeout, EOF, structured remote error and wrong-correlation failures. Which may preserve executor health and which make state uncertain?

## Lifecycle and cleanup

25. Call `engine.close()` after the async driver is idle. Verify every `worker_alive` entry becomes false.
26. Why does `close()` reject an actively running async driver while `aclose()` first waits for idle?
27. Explain why process termination is a cleanup fallback, not a request-cancellation primitive.
28. Describe what an orphan worker would imply about runtime ownership correctness.

## Multiprocess observability

29. Call `snapshot()` twice with no intervening serving work. Prove the snapshots compare equal even though each snapshot queries child state over IPC.
30. Explain why `CACHE_STATS` and `SNAPSHOT_EXECUTION` reads are excluded from serving-command counters.
31. For each rank, interpret `process_id`, `worker_state`, commands sent/completed, remote errors and transport failures.
32. Propose a new transport observation. Decide whether reading it can remain side-effect free.

## Semantic equivalence

33. Run v1.0 and v1.1 with `world_size=1` and identical configuration. Compare committed output token histories.
34. Repeat with `world_size=2`. Why should moving execution into child processes not change deterministic model semantics?
35. Switch v1.1 PagedAttention between reference and vectorized modes. Which process owns the counter changes?
36. Disable compile-aware projection. Why is eager/compiled equivalence independent of IPC correctness?

## Prefix-aware scheduling across IPC

37. Warm a prompt in v1.1, then submit warm/cold requests. Use `snapshot()` to explain admission using only policy-visible reuse facts.
38. Explain why prefix estimation must remain side-effect free even though the cache that computes it now lives remotely.
39. Distinguish: estimated reusable prefix, actual attached prefix, parent logical computed frontier, child physical ownership.
40. Why would exposing child block IDs to the Scheduler violate both v0.13 and v1.1 contracts?

## Historical synthesis

41. Compare v0.8 and v1.1. Both expose `Executor → Worker`; what new problem appears only when that boundary becomes an OS process boundary?
42. Compare v0.9 and v1.1. What remains a logical/reference collective, and what becomes physically distributed across processes?
43. Explain why v1.1 is a new mechanism release while v1.0 was primarily a composition release.
44. Propose a v1.2 learning target. Reject proposals that are merely “more multiprocessing features” rather than a distinct systems boundary.

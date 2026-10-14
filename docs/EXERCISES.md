# mini-vLLM v1.3 exercises

Use historical branch diffs to learn one mechanism at a time. v1.3 focuses on the boundary between untrusted network bytes and the already-correct token-native v1.2 runtime.

## Follow one request from socket to token

1. Run `examples/18_network_server.py`. Draw `TCP → HTTP parser → GenerateRequest → tokenizer → Request → AsyncEngine → EngineCore → child Worker → committed event → JSON/SSE`.
2. Mark which objects are owned by NetworkServer, AsyncEngine, EngineCore and child Worker.
3. Explain why `GenerateRequest` is not the same object as core `Request`.
4. Find the exact line where prompt text stops existing as a serving-core concern.

## Bounded parsing

5. Lower `NetworkConfig.max_header_bytes` and send a large header. Why should rejection happen before JSON or model work?
6. Lower `max_request_bytes` and send a larger declared body. Which HTTP status is returned and why is no core request created?
7. Send a POST without `Content-Length`. Explain why v1.3 returns 411 instead of trying to infer body framing.
8. Send `Transfer-Encoding: chunked`. Why does the educational server reject it instead of partially implementing HTTP chunking?

## API schema validation

9. Send invalid UTF-8/JSON. Verify no request appears in `engine.snapshot()`.
10. Add an unknown JSON field. Why does exact-field validation improve the teaching contract?
11. Send `max_new_tokens=0`, invalid `top_k`, or invalid `top_p`. Identify which layer rejects each request.
12. Reuse a completed request ID. Explain why the network maps the existing AsyncEngine duplicate-ID rule to HTTP 409.

## Four capacity layers

13. Draw and distinguish:

```text
network max_inflight_requests
AsyncEngine max_queue_size
Scheduler max_num_seqs / token budget
KV num_blocks
```

14. Set `max_inflight_requests=1` and start two overlapping requests. Prove one receives 429 before hidden runtime submission.
15. Set AsyncEngine `max_queue_size=1` with one slow streaming consumer. Why does model progress pause without dropping committed tokens?
16. Construct a KV-pressure case. Why is a cache/preemption event categorically different from HTTP 429?

## JSON completion

17. Send `/v1/generate` and compare output token IDs with direct `create_engine("v1.2").submit_text(...)` under equivalent model/runtime configuration.
18. Explain why the non-stream handler must still consume every intermediate `TokenEvent` even though it returns only one final JSON body.
19. What deadlock would appear if the handler only awaited terminal metadata while the bounded event queue became full?

## SSE streaming

20. Send `/v1/generate/stream` and list the emitted event names.
21. Verify token event order equals `output_index` order and final `done.output_token_ids`.
22. Explain why NetworkServer cannot expose speculative candidate tokens: what is its only event source?
23. Slow socket writes. Which layer experiences backpressure and which serving semantics must remain unchanged?

## Disconnect cancellation

24. Open an SSE request, read one token event and close the client socket. Observe `disconnect_cancellations`.
25. Trace the path `EOF → handle.cancel() → AsyncEngine cancellation intent → EngineCore cancel → Scheduler/cache cleanup`.
26. Why is killing the worker process an invalid implementation of client disconnect?
27. Explain why a disconnected client may receive no terminal bytes while the runtime must still reach exactly one logical terminal outcome.
28. Verify duplicate disconnect observations do not double-count cancellation for one in-flight request.

## Health versus readiness

29. Compare `/health` and `/ready` immediately after `start()`.
30. Explain why a bound socket is insufficient readiness evidence.
31. Identify the deeper facts `_runtime_ready()` reads: execution health and artifact rank identity agreement.
32. During server draining, why can health remain true while readiness is false?

## Ownership-aware lifecycle

33. Create an engine, then `engine.create_server()`. Close the server and prove worker processes remain alive.
34. Use top-level `create_server(...)`. Close it and prove worker processes are reaped.
35. Why must a wrapper never close an externally owned engine by default?
36. Explain the shutdown order: stop accepts → cancel/drain active API requests → close connection tasks → close owned engine.

## Observability

37. Call `server.snapshot()` twice without intervening work. Prove equality.
38. Explain why reading `active_connections` or readiness must not mutate request/network counters.
39. Propose one useful network metric. Decide whether computing it would violate read-only observability.
40. Why are raw socket objects deliberately absent from `NetworkSnapshot`?

## Error taxonomy

41. Map malformed JSON, wrong method, duplicate ID, ingress full and not-ready to their HTTP statuses.
42. Explain why these HTTP statuses are API-edge facts, not Scheduler states.
43. Compare a 429 ingress rejection with a child `WorkerRemoteError`; which boundaries produced them?
44. Compare client disconnect with unexpected worker death; why are their recovery semantics radically different?

## Historical synthesis

45. Compare v0.10 and v1.3. Both involve streaming/backpressure; what new ownership problem appears only with a real socket?
46. Compare v1.1 and v1.3. Which transport is internal IPC and which is external API transport?
47. Compare v1.2 and v1.3. Why does validated model readiness become one prerequisite inside network readiness?
48. Explain the full separation:

```text
network bytes
!= validated API request
!= token-native Request
!= logical state
!= physical KV
!= model parameter storage
```

49. Propose a v1.4 multi-model registry contract. What must remain pinned for an in-flight request if one model starts draining?
50. Reject a proposed v1.4 that merely adds HTTP/2 or more routes. Why is that not yet a distinct serving-runtime learning target?

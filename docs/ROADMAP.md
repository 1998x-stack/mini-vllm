# mini-vLLM Roadmap

| Version | Learning target | Status |
|---|---|---|
| v0.1 | static batching + contiguous KV | frozen |
| v0.2 | continuous batching | frozen |
| v0.3 | paged KV + block table | frozen |
| v0.4 | token budget + chunked prefill | frozen |
| v0.5 | prefix cache + shared block ownership | frozen |
| v0.6 | KV-pressure preemption + recomputation | frozen |
| v0.7 | speculative decoding + transactional KV | frozen |
| v0.8 | EngineCore / Executor / Worker | frozen |
| v0.9 | WorkerGroup + reference tensor parallel | frozen |
| v0.10 | async streaming / cancellation / backpressure | frozen |
| v0.11 | compile-aware rank-local tensor execution | frozen |
| v0.12 | reference + vectorized PagedAttention | frozen |
| v0.13 | prefix-aware scheduling / cache-aware admission | frozen |
| v1.0 | integrated public API + observability | frozen |
| v1.1 | real multiprocessing Executor/Worker transport | frozen |
| v1.2 | model artifact + tokenizer loading boundary | frozen |
| v1.3 | network serving / API ingress boundary | current implementation / exact-CI release rule |
| v1.4 | multi-model registry + model lifecycle/routing | planned direction |

Dependency order:

```text
serving semantics
→ memory
→ scheduling
→ sharing
→ pressure recovery
→ transactions
→ execution boundary
→ logical distribution
→ async delivery
→ compile awareness
→ attention optimization
→ cache-aware policy
→ integrated runtime
→ real OS process isolation / IPC
→ model artifact trust / tokenizer boundary
→ network/API ingress lifecycle
→ multi-model lifecycle + routing
```

v1.3 deliberately stops at a small one-process network frontend over one artifact-bound engine. It does not claim production HTTP compliance, authentication, tenant quotas, TLS, HTTP/2, reverse proxying or multi-node request routing.

The preferred v1.4 direction is **multi-model registry + model lifecycle/routing**. That version should teach a new ownership problem: a request chooses one validated model identity, remains pinned to that identity for its full lifetime, and model load/readiness/drain/unload must coexist without cross-model state leakage. Merely adding more HTTP endpoints is not a sufficiently distinct new lesson.

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
| v0.11 | compile-aware rank-local tensor execution | current / CI-gated |
| v0.12 | optimized PagedAttention path | next |
| v0.13 | prefix-aware scheduling/cache policy | planned |
| v1.0 | integrated educational serving runtime | planned |

Dependency order: serving semantics → memory → scheduling → sharing → pressure recovery → transactions → execution boundary → distribution → async delivery → compile awareness → kernel optimization.

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
| v0.8 | EngineCore / Executor / Worker boundary | frozen |
| v0.9 | distributed + tensor-parallel concepts | next |
| v0.10 | async engine + streaming | planned |
| v0.11 | graph/compile-aware execution | planned |
| v0.12 | optimized PagedAttention path | planned |
| v0.13 | prefix-aware scheduling/cache policy | planned |
| v1.0 | integrated educational serving runtime | planned |

Dependency order matters: serving semantics → memory → scheduling → sharing → pressure recovery → transactions → execution boundary → distribution → optimization.

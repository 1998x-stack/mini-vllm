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
| v1.2 | model artifact + tokenizer loading boundary | current implementation / exact-CI release rule |

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
→ validated external model + tokenizer ingestion
```

v1.2 intentionally supports one local educational model format. It does not add Hub downloads, arbitrary model code, safetensors/sharded formats, quantization, LoRA, hot swap or GPU weight streaming. Adding more checkpoint formats alone is not enough reason for a new version.

A strong next systems boundary is **network serving/API lifecycle**: request serialization at the socket/API edge, server startup/readiness, client disconnect/cancellation, bounded ingress/backpressure and health/readiness observations. Real CUDA/Triton kernels remain an independent later optimization line.

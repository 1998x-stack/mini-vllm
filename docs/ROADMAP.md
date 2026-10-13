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
| v1.0 | integrated public API + observability + end-to-end runtime | frozen |
| v1.1 | real multiprocessing Executor/Worker transport | current implementation / exact-CI release rule |

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
```

v1.1 intentionally stops at one-machine multiprocessing. It does not claim multi-node transport, NCCL, CUDA IPC, shared-memory tensor transport, or throughput improvement.

After v1.1, a new branch again requires a genuinely new learning target. The preferred next direction is production-style model/tokenizer loading: validating external model artifacts, mapping weights into runtime-owned modules, tokenizer/model compatibility, and separating artifact identity from serving execution state. Real CUDA/Triton kernels remain a later independent learning line.

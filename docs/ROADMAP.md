# Version roadmap

`Plan.md` is the canonical detailed roadmap. This file is the compact learning map.

```text
v0.1  request lifecycle + static batching + contiguous KV
  ↓
v0.2  continuous batching
  ↓
v0.3  paged KV / block table / recycling
  ↓
v0.4  token budget / chunked prefill
  ↓
v0.5  prefix caching / shared immutable blocks
  ↓
v0.6  preemption under KV pressure
  ↓
v0.7  speculative decoding / rollback
  ↓
v0.8  EngineCore / Executor / Worker split
  ↓
v0.9  distributed / tensor-parallel concepts
  ↓
v0.10 async streaming / cancellation / backpressure
  ↓
v0.11 compile/CUDA-graph experiments
  ↓
v0.12 optimized PagedAttention path
```

The rule is one major mental model per version. Historical branches remain frozen learning snapshots.

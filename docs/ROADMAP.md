# mini-vLLM roadmap

```text
v0.1  static batching + contiguous KV                 done
v0.2  continuous batching                            done
v0.3  paged KV / block table / recycling             done
v0.4  token budget / chunked prefill                  done
v0.5  prefix caching / shared block ownership         done
v0.6  KV-pressure preemption / recompute              done
v0.7  speculative decoding / transactional KV         done
v0.8  EngineCore / Executor / Worker boundary         next
v0.9  distributed / tensor-parallel concepts
v0.10 async streaming / cancellation
v0.11 CUDA Graph / compile experiments
v0.12 Triton/CUDA PagedAttention
v0.13 prefix-aware scheduling
v1.0  integrated educational serving runtime
```

Each version introduces one primary serving mental model. Historical branches remain learning snapshots rather than receiving later-version mechanisms.

# mini-vLLM

A small, branch-versioned LLM-serving runtime for learning the systems ideas behind vLLM.

## Current line

`main` is always the newest exact-CI-validated release. v0.12 is the current optimized-PagedAttention implementation line; an exact v0.12 commit is synchronized to `main` only when that same commit has a green repository CI run.

```text
v0.1  static batching + contiguous KV
v0.2  continuous batching
v0.3  paged KV + block tables
v0.4  token budget + chunked prefill
v0.5  prefix caching + shared KV
v0.6  preemption + recomputation
v0.7  speculative decoding + transactional KV
v0.8  EngineCore / Executor / Worker
v0.9  logical multi-rank + tensor parallel
v0.10 async streaming + cancellation/backpressure
v0.11 compile-aware rank-local tensor execution
v0.12 reference + vectorized PagedAttention             ← current
v0.13 prefix-aware scheduling                           ← next
```

## v0.12 mental model

```text
logical positions
      ↓
block table
      ↓
physical block + offset
      ↓
ReferencePagedAttention      correctness oracle
      or
VectorizedPagedAttention     tensor-indexed paged gather
      ↓
same attention context
      ↓
TinyDecoder / compile-aware LM-head / WorkerGroup / AsyncEngine
```

The vectorized path removes the Python per-token KV gather loop. It does **not** change Scheduler policy, cache ownership, prefix sharing, preemption, speculative rollback, distributed topology, compilation policy, or streaming semantics.

## Verify

```bash
pip install -e '.[dev]'
./scripts/check.sh
python examples/13_paged_attention.py
```

The v0.12 trace compares reference and vectorized outputs and prints per-rank attention backend statistics. No speedup is claimed without measurement, and Triton/CUDA remains optional rather than a correctness dependency.

Read `Agents.md`, `Plan.md`, and `version.md` before changing architecture.

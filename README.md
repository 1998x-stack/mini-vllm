# mini-vLLM

A small, branch-versioned LLM-serving runtime for learning the systems ideas behind vLLM.

## Current line

`main` is always the newest validated release. v0.11 is the active compile-aware implementation candidate and may enter `main` only when its exact commit passes repository CI.

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
v0.11 compile-aware rank-local tensor execution       ← current candidate
v0.12 optimized PagedAttention                        ← next
```

## v0.11 mental model

```text
control/lifecycle/orchestration                       eager
        ↓
rank-local TinyDecoder + KV                           eager
        ↓
hidden trace
        ↓
PureTensorProjectionKernel                            torch.compile-aware
        ↓
rank-local logits
        ↓
collective                                             eager
```

The project intentionally does **not** call `torch.compile(engine)`. Compile state lives next to rank-local tensor execution and is indexed by an explicit `ExecutionShapeSignature`.

## Verify

```bash
pip install -e '.[dev]'
./scripts/check.sh
python examples/12_compile_aware.py
```

The example prints miss → compiled → hit/new-signature behavior per rank. `backend="eager"` in CI exercises Dynamo graph formation but is not a performance claim.

Read `Agents.md`, `Plan.md`, and `version.md` before changing architecture.

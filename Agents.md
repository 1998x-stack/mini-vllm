# Agents.md — mini-vLLM Development Contract

Read this before every code, test, branch, or architecture change.

## Mission

Build one serving mental model per version. Prefer correctness, readable mechanisms, explicit invariants, inspectable traces, and reference equivalence before optimization.

## Branch model

`main` is always the newest exact-CI-validated runtime. Historical version branches are learning snapshots.

Current sequence:

```text
v0.1 static batching
v0.2 continuous batching
v0.3 paged KV
v0.4 token budget/chunked prefill
v0.5 prefix sharing
v0.6 preemption/recompute
v0.7 speculative transactions
v0.8 execution boundary
v0.9 logical ranks/tensor parallel
v0.10 async streaming/cancellation
v0.11 compile-aware tensor execution
v0.12 optimized PagedAttention
```

## Ownership

- AsyncEngine: caller lifecycle, bounded event delivery, cancellation intent.
- EngineCore/Scheduler: serving policy and logical request state.
- Executor/WorkerGroup: execution orchestration/topology.
- Worker/ModelRunner: rank-local model and KV mechanism.
- KV backend: physical cache storage/ownership.
- Compile registry: rank-local tensor specialization only.
- Collective: communication mathematics only.

Compile-specific policy must not enter AsyncEngine, EngineCore, Scheduler, WorkerGroup, or KV ownership code.

## Core invariants

- computed tokens imply valid KV;
- rejected speculative state disappears;
- execution failures cannot partially commit logical state;
- distributed rank failures roll replicated KV back together;
- async streams expose committed tokens only and one terminal event;
- compile-aware and eager tensor paths are numerically equivalent;
- a shape-signature miss/hit/fallback is observable;
- compilation failure cannot change serving semantics when eager fallback is enabled.

## Testing / release

```bash
python -m pytest -q
./scripts/check.sh
```

Never claim tests passed unless executed on the exact reported snapshot. An active version freezes only after its exact commit passes repository GitHub Actions, docs agree with code, `main` advances to that same commit, and the next planning branch is created from latest `main`.

## Performance rule

```text
readable eager oracle
  ↓
correctness tests
  ↓
explicit optimization boundary
  ↓
measure/inspect
  ↓
optimized educational path
```

`torch.compile` success is not itself evidence of speedup. CUDA Graph, custom Triton/CUDA PagedAttention, aggressive dynamic shapes, and production bucketing remain future mechanisms until their correctness oracle exists.

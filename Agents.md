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
v0.13 prefix-aware scheduling
```

## Ownership

- AsyncEngine: caller lifecycle, bounded event delivery, cancellation intent.
- EngineCore/Scheduler: serving policy and logical request state.
- Executor/WorkerGroup: execution orchestration/topology.
- Worker/ModelRunner: rank-local model and KV mechanism.
- KV backend: physical cache storage, block tables, allocation, sharing, and recycling.
- PagedAttention backend: read current logical K/V view and compute attention context; it does not own blocks.
- Compile registry: rank-local tensor specialization only.
- Collective: communication mathematics only.

Attention optimization must not choose admissions, allocate/free blocks, mutate prefix ownership, decide preemption, or alter async lifecycle.

## Core invariants

- computed tokens imply valid KV;
- paged address translation is `logical_block=p//B`, `offset=p%B`, `physical=block_table[logical_block]`;
- attention reads only logical positions `[0, length)` even when the final physical block has unused slots;
- reference and vectorized paged attention are numerically equivalent;
- shared-prefix blocks remain immutable and valid under vectorized reads;
- rejected speculative state disappears;
- execution failures cannot partially commit logical state;
- distributed rank failures roll replicated KV back together;
- async streams expose committed tokens only and one terminal event;
- compile-aware and eager tensor paths are numerically equivalent;
- explicit optimization fallbacks cannot change serving semantics.

## Testing / release

```bash
python -m pytest -q
./scripts/check.sh
```

Never claim tests passed unless executed on the exact reported snapshot. An active version freezes only after its exact commit passes repository GitHub Actions, docs agree with code, `main` advances to that same commit, and the next planning branch is created from latest `main`.

## Performance rule

```text
readable oracle
  ↓
correctness tests
  ↓
explicit optimization boundary
  ↓
measure/inspect
  ↓
optimized educational path
```

`torch.compile` success or vectorized PyTorch code is not itself evidence of speedup. A real Triton/CUDA path may only be claimed after an actual kernel exists, has capability checks, and is numerically validated against the reference path.

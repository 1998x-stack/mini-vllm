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
v1.0 integrated educational runtime
```

## Ownership

- AsyncEngine: caller lifecycle, bounded event delivery, cancellation intent.
- EngineCore: logical truth coordination and policy/mechanism reconciliation.
- Scheduler: admission order, token-budget policy, fairness, and victim policy.
- Executor/WorkerGroup: execution orchestration/topology and replicated-consistency checks.
- Worker/ModelRunner: rank-local model execution.
- KV backend: physical cache storage, block tables, allocation, sharing, recycling, and policy-safe reuse facts.
- PagedAttention backend: read current logical K/V view and compute attention context; it does not own blocks.
- Compile registry: rank-local tensor specialization only.
- Collective: communication mathematics only.

Scheduler policy must not inspect physical block IDs, refcounts, cache LRU internals, or free-list state. Cache reuse estimates expose quantities, not ownership handles.

## Core invariants

- computed tokens imply valid KV;
- a `PrefixReuseHint` is advisory: it is not a reservation, ownership claim, or computed-token claim;
- side-effect-free prefix estimation cannot create block tables, change refcounts, or touch cache recency;
- a deferred waiting request may have a warm hint while owning zero KV blocks;
- only authoritative `prepare_request()` may advance the reusable computed frontier;
- stale hints may affect admission quality but can never affect serving correctness;
- warm-prefix preference must have explicit starvation protection and deterministic tie-breaking;
- replicated ranks must agree on policy-visible cache facts;
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

## Performance / policy rule

```text
mechanism fact
  ↓
explicit policy input
  ↓
invariant tests
  ↓
observable decision
  ↓
authoritative execution truth
```

Do not convert an estimate into ownership just because it improves scheduling. Likewise, `torch.compile` success or vectorized PyTorch code is not itself evidence of speedup. Optimizations and policies remain subordinate to the readable correctness contract.

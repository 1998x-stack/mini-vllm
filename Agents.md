# Agents.md — mini-vLLM Development Contract

Read this before every code, test, branch, or architecture change.

## Mission

Build serving-system understanding through small, explicit mechanisms. Prefer correctness, readable references, ownership boundaries, inspectable traces, invariant tests and reference equivalence before optimization.

## Branch model

`main` is always the newest exact-CI-validated runtime. Historical version branches are frozen learning snapshots.

```text
v0.1  static batching
v0.2  continuous batching
v0.3  paged KV
v0.4  token budget / chunked prefill
v0.5  prefix sharing
v0.6  preemption / recompute
v0.7  speculative transactions
v0.8  execution boundary
v0.9  logical ranks / tensor parallel
v0.10 async streaming / cancellation
v0.11 compile-aware tensor execution
v0.12 optimized PagedAttention
v0.13 prefix-aware scheduling
v1.0  integrated educational runtime
```

v1.0 is a consolidation release. A post-v1.0 branch requires a genuinely new learning target; incremental feature accumulation alone is insufficient.

## Ownership

- `AsyncEngine`: caller lifecycle, bounded event delivery, cancellation intent and backpressure.
- `EngineCore`: logical request truth, atomic step boundaries, reconciliation and commit authority.
- `Scheduler`: admission order, token budget, fairness and preemption-victim policy.
- `Executor/WorkerGroup`: execution orchestration, topology, rank consistency and distributed rollback.
- `Worker/ModelRunner`: rank-local model execution.
- KV backend: physical storage, block tables, allocation, sharing, recycling, transactions and policy-safe reuse facts.
- PagedAttention backend: read the current logical K/V view and compute context; it does not own blocks.
- Compile registry: rank-local tensor specialization only.
- Collective: communication mathematics only.
- v1.0 observability: copy public facts into read-only snapshots; never become a new policy owner.

Scheduler policy must not inspect physical block IDs, refcounts, cache LRU internals or free-list state. Cache reuse estimates expose quantities, not ownership handles.

## Stable v1.0 invariants

The canonical expanded list is `docs/INVARIANTS.md`. At minimum every change must preserve:

- computed tokens imply valid, addressable KV;
- sampled output tokens are not computed until a later forward executes them;
- physical ownership precedes K/V access;
- block-table translation is authoritative;
- `PrefixReuseHint` is advisory, never reservation/ownership/computed state;
- side-effect-free prefix estimation cannot mutate cache ownership or recency;
- authoritative `prepare_request()` establishes the actual reusable frontier;
- stale hints can hurt admission quality but not correctness;
- warm-prefix preference has deterministic starvation protection;
- preemption cannot retain claims to released KV;
- replay preserves logical history without resampling historical output;
- rejected speculative state disappears;
- speculative rollback cannot mutate shared immutable prefixes;
- execution failures cannot partially commit logical state;
- replicated ranks roll physical state back together on failure;
- async clients see committed tokens only and exactly one terminal outcome;
- vectorized PagedAttention equals the readable reference path;
- compile-aware projection equals eager projection;
- `snapshot()` is read-only and cannot schedule, attach, evict, touch LRU or advance request state.

## Public v1.0 surface

Prefer teaching through these stable entry points:

```text
RuntimeConfig
create_engine()          # defaults to v1.0
Request / AsyncRequestHandle
engine.snapshot()
EngineStepOutput
```

Historical `create_engine("v0.x")` builders remain available for mechanism-by-mechanism comparison.

Do not expose private physical ownership structures merely to make observability convenient. Add logical/copy snapshots instead.

## Change protocol

Before changing `main` or the active version:

1. read `Agents.md`;
2. read `Plan.md`;
3. read `version.md`;
4. read `docs/ARCHITECTURE.md` and `docs/INVARIANTS.md`;
5. inspect the relevant implementation and invariant tests;
6. separate policy changes from mechanism changes;
7. add/adjust tests that prove the mechanism or boundary directly;
8. run the exact release gate before advancing `main`.

Historical branches are frozen except narrowly scoped documentation/test corrections or true bugs within that historical lesson.

## Testing / release

```bash
python -m pytest -q
./scripts/check.sh
```

Never claim tests passed unless executed on the exact reported snapshot. An active version freezes only after:

1. its exact commit passes repository GitHub Actions;
2. docs agree with code;
3. `main` advances to that same commit;
4. branch history remains pedagogically clean.

## Performance / policy rule

```text
readable mechanism fact
        ↓
explicit policy input
        ↓
invariant tests
        ↓
observable decision
        ↓
authoritative execution truth
```

Do not convert an estimate into ownership merely because it improves scheduling. Do not claim speedup from `torch.compile`, vectorization, CUDA or Triton without measurement. Optimization remains subordinate to the readable correctness contract.

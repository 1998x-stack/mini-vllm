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
v1.1  real multiprocessing executor transport
```

v1.1 is the first post-v1.0 mechanism release because it adds a genuinely new systems lesson: a real OS isolation boundary with serialization, lifecycle and transport failure semantics.

## Ownership

- `AsyncEngine`: caller lifecycle, bounded event delivery, cancellation intent and backpressure.
- `EngineCore`: logical request truth, atomic step boundaries, reconciliation and commit authority.
- `Scheduler`: admission order, token budget, fairness and preemption-victim policy.
- `MultiprocessExecutor`: parent-side IPC orchestration, correlation, worker health, result coordination and shutdown.
- child `Worker/ModelRunner`: rank-local model execution and physical KV ownership inside one OS process.
- KV backend: physical storage, block tables, allocation, sharing, recycling, transactions and policy-safe reuse facts; in v1.1 it lives only in the child process.
- PagedAttention backend: read the child-local logical K/V view and compute context; it does not own blocks.
- Compile registry: child/rank-local tensor specialization only.
- Collective: parent-side reference communication mathematics only.
- observability: copy public facts into read-only snapshots; never become a policy owner.

The parent process must not keep or inspect a live child `Worker`, cache object, block table, refcount structure, model runner or tensor-storage handle.

## Stable invariants

The canonical expanded list is `docs/INVARIANTS.md`. At minimum every change must preserve:

- computed tokens imply valid, addressable KV;
- sampled output tokens are not computed until a later forward executes them;
- physical ownership precedes K/V access;
- block-table translation is authoritative;
- `PrefixReuseHint` is advisory, never reservation/ownership/computed state;
- authoritative `prepare_request()` establishes the actual reusable frontier;
- warm-prefix preference has deterministic starvation protection;
- preemption cannot retain claims to released KV;
- replay preserves logical history without resampling historical output;
- rejected speculative state disappears;
- speculative rollback cannot mutate shared immutable prefixes;
- execution failures cannot partially commit logical state;
- replicated ranks roll physical state back together when rollback remains reachable;
- async clients see committed tokens only and exactly one terminal outcome;
- vectorized PagedAttention equals the readable reference path;
- compile-aware projection equals eager projection;
- `snapshot()` is read-only and cannot schedule, attach, evict, touch LRU or advance request state;
- IPC messages contain logical/serializable data, not live physical ownership objects;
- a remote execution result becomes commit-eligible only after its correlation ID and success response are acknowledged;
- child-reported errors are distinct from transport/process-death failures;
- if child physical state cannot be proven consistent, the executor becomes unhealthy and EngineCore cannot logically commit that step;
- normal shutdown joins every child process; no worker may remain orphaned after close.

## Public v1.1 surface

Prefer teaching through these stable entry points:

```text
RuntimeConfig
MultiprocessConfig
create_engine()          # defaults to v1.1
Request / AsyncRequestHandle
engine.snapshot()
engine.close() / await engine.aclose()
EngineStepOutput
```

Historical `create_engine("v0.x")` and `create_engine("v1.0")` builders remain available for mechanism-by-mechanism comparison.

Do not expose private physical ownership structures merely to make observability convenient. Add logical/copy snapshots or explicit transport-safe values instead.

## Process-boundary rule

```text
parent logical truth
        ↓ command
child physical mutation
        ↓ acknowledged result
parent logical commit
```

Never skip the acknowledgment boundary. A send is not an execution success, and a child process being alive is not proof that a command completed.

Transport failures after possible mutation are conservative failures: repair reachable state if possible, otherwise mark the executor unhealthy rather than guessing.

## Change protocol

Before changing `main` or the active version:

1. read `Agents.md`;
2. read `Plan.md`;
3. read `version.md`;
4. read `docs/ARCHITECTURE.md` and `docs/INVARIANTS.md`;
5. inspect the relevant implementation and invariant tests;
6. separate policy changes from mechanism/transport changes;
7. add tests that prove ownership, serialization, rollback and cleanup directly;
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

For multiprocessing changes, tests must include process identity, structured remote error, unexpected child death, deterministic equivalence, and explicit shutdown/join behavior.

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

Do not convert an estimate into ownership merely because it improves scheduling. Do not claim speedup from multiprocessing, `torch.compile`, vectorization, CUDA or Triton without measurement. v1.1 multiprocessing is an isolation/transport lesson, not a throughput claim.

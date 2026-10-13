# mini-vLLM v0.11 — graph/compile-aware execution (planning branch)

## 1. Version identity

- **Branch:** `v0.11`
- **Base:** latest validated `v0.10` / `main`
- **Previous version:** `v0.10`
- **Theme:** make graph/capture boundaries, shape signatures, compilation reuse, graph breaks, and eager fallback explicit inside the execution plane
- **Status:** planning / implementation target; runtime is intentionally identical to v0.10 until the acceptance criteria below pass

v0.10 established an async caller facade above one synchronous EngineCore and one distributed execution topology. v0.11 asks a different question:

> Which part of model execution is stable enough to compile/capture, which state must remain eager, and how do we preserve serving semantics when shapes or unsupported operations change?

## 2. Primary mental model

Do **not** compile the serving control plane:

```text
AsyncEngine                    eager lifecycle/delivery
    ↓
EngineCore                     eager policy/state machine
    ↓
Scheduler                      eager admission/token budget
    ↓
DistributedExecutor            eager orchestration
    ↓
WorkerGroup                    eager rank topology
    ↓
Worker / ModelRunner
    │
    ├── eager cache address/ownership work
    │
    └── compile-aware pure tensor kernel
            │
            ├── signature hit  → reuse compiled callable
            ├── signature miss → compile/capture candidate
            └── unsupported    → eager fallback
```

The learning target is a **narrow execution kernel boundary**, not a magical `torch.compile(engine)` wrapper.

## 3. Why v0.11 follows v0.10

The project now has explicit boundaries for:

- caller lifecycle and backpressure;
- centralized serving policy;
- execution transport/orchestration;
- logical multi-rank topology;
- physical KV ownership;
- model execution.

That separation is required before compilation can be added cleanly. Compile/capture state belongs close to Worker/ModelRunner tensor execution. It must not leak upward into Scheduler, cancellation, streaming, or request-priority policy.

## 4. Required abstractions

The first implementation should introduce inspectable concepts similar to:

```text
ExecutionShapeSignature
CompileConfig
CompileStats
CompiledKernelCache / CompileRegistry
CompileAwareTinyRunner
PureTensorStepKernel
```

Exact names may evolve, but the responsibilities must remain separated.

## 5. Pure tensor boundary

Current TinyDecoder execution mixes two different kinds of work:

```text
Python/control/storage work
  - request IDs
  - KV block tables
  - cache read/write
  - allocation/lifecycle

Tensor math
  - projections
  - attention score/value math
  - normalization
  - MLP
  - final hidden/logit projection
```

v0.11 should make this boundary explicit enough that compilation only sees tensor inputs/outputs.

Preferred shape:

```text
ModelRunner
  │
  ├── read logical prefix / prepare tensor inputs      eager
  │
  ▼
PureTensorStepKernel(tensors only)                     compile-aware
  │
  ├── hidden/logits
  └── newly computed K/V tensors
  │
  ▼
ModelRunner writes new K/V through KV backend          eager
```

Do not put mutable cache dictionaries, block tables, Scheduler objects, Request objects, or asyncio queues inside the compiled function.

## 6. Shape signature

Compilation reuse must be explicit rather than hidden behind framework internals.

A first educational signature may include values such as:

```text
ExecutionShapeSignature(
    chunk_tokens,
    context_tokens,
    dtype,
    device,
    model dimensions,
    rank-local output width,
)
```

Only fields that materially affect captured tensor shape/layout should be included.

The learner should be able to observe:

```text
first signature
    ↓
compile miss
    ↓
compiled callable stored

same signature later
    ↓
cache hit / reuse

new shape
    ↓
new signature
    ↓
recapture / recompile
```

## 7. Compile backend policy

CI must remain CPU-friendly and deterministic.

Preferred reference approach:

- use `torch.compile` through a small adapter;
- default tests may use a lightweight tracing backend such as `backend="eager"` to exercise Dynamo graph formation without depending on GPU/Inductor availability;
- optional local experiments may use `inductor` when supported;
- eager execution remains the correctness oracle;
- unsupported backend/device combinations must fail clearly or use an explicitly documented eager fallback.

Do not claim a speedup merely because a callable was compiled.

## 8. Graph-break visibility

v0.11 should teach graph breaks rather than hide them.

Introduce at least one observable diagnostic path for:

```text
compile attempts
compiled signatures
cache hits
cache misses
fallbacks
recompilations / recaptures
```

If useful, include a deliberately unsupported or dynamic operation in a focused example/test to demonstrate why it cannot stay inside the pure tensor boundary. The production path should avoid that operation rather than swallowing the break silently.

## 9. Eager fallback

Compilation is an optimization mode, not a correctness dependency.

Required rule:

```text
compile unavailable / unsupported signature / compile failure
        ↓
explicit policy decision
        ↓
eager reference kernel
        ↓
same logical result
```

A fallback must not:

- change token history;
- skip KV writes;
- duplicate KV writes;
- weaken distributed rollback;
- bypass speculative transaction semantics;
- alter async terminal/cancellation behavior.

## 10. Distributed ownership

v0.9/v0.10 use logical ranks with rank-local model runners and replicated KV. Compile state should therefore be rank-local execution state:

```text
WorkerGroup
  ├── rank 0 Worker
  │      └── compile registry for rank-0 tensor kernel
  ├── rank 1 Worker
  │      └── compile registry for rank-1 tensor kernel
  └── ...
```

The collective remains outside the rank-local compiled tensor kernel unless a later version explicitly proves a stable distributed capture boundary.

WorkerGroup still owns topology; EngineCore still knows nothing about graph signatures.

## 11. Speculative execution

Speculative target verification must use the same eager/compiled correctness boundary.

For deterministic input:

```text
eager target verification logits trace
              ==
compile-aware target verification logits trace
```

Rejected draft rollback remains a cache transaction outside the compiled kernel. Compilation must not retain hidden rejected state.

## 12. Async compatibility

v0.10 AsyncEngine should require no scheduling/lifecycle redesign.

From the caller perspective:

```text
v0.10 eager collected token stream
              ==
v0.11 compile-aware collected token stream
```

Cancellation is still applied at EngineCore step boundaries. Bounded output queues still live above EngineCore. Backpressure must not be encoded into compile signatures.

## 13. Static vs dynamic shapes

The first version should prefer visible static signatures and deliberate recapture over aggressive dynamic-shape magic.

This teaches the serving tradeoff:

```text
more shape specialization
    → potentially better optimized kernels
    → more compile/capture variants
    → more warmup/recapture cost

more dynamic behavior
    → fewer variants
    → potentially weaker optimization / more graph complexity
```

Do not introduce elaborate production bucketing before basic signature behavior is measured and tested.

## 14. Metrics / trace requirements

Expose enough data to print a trace such as:

```text
step | rank | signature | mode     | cache
-----+------+-----------+----------+------
0    | 0    | p4-c0     | compile  | miss
0    | 1    | p4-c0     | compile  | miss
1    | 0    | d1-c4     | compile  | miss
1    | 1    | d1-c4     | compile  | miss
2    | 0    | d1-c4     | compiled | hit
2    | 1    | d1-c4     | compiled | hit
```

Naming can differ, but compilation reuse must be inspectable.

## 15. Preferred implementation order

```text
1. extract/define pure tensor execution kernel boundary
2. prove eager kernel matches current TinyDecoder path
3. define ExecutionShapeSignature
4. add CompileConfig + stats/registry
5. add compile adapter with eager backend for CI
6. integrate CompileAwareTinyRunner rank-locally
7. prove same-signature reuse
8. prove shape change causes explicit new variant
9. add compile-failure/eager-fallback path
10. validate speculative logits-trace equivalence
11. validate distributed world-size 1/2 equivalence
12. validate v0.10 async stream equivalence/cancellation
13. add executable graph-signature trace
14. run exact-candidate CI before main synchronization
```

## 16. Acceptance criteria

v0.11 cannot freeze until all applicable items are true:

- [ ] compile/capture boundary accepts tensor data only and excludes Request/Scheduler/async objects;
- [ ] eager extracted kernel reproduces the pre-v0.11 reference numerically;
- [ ] `ExecutionShapeSignature` is explicit and deterministic;
- [ ] first unseen signature records a compile/cache miss;
- [ ] repeated compatible signature reuses the same compiled callable;
- [ ] materially different tensor shape produces a new signature/variant;
- [ ] compilation statistics are observable in tests/example traces;
- [ ] compile failure or unsupported mode has deterministic explicit fallback/error behavior;
- [ ] eager fallback preserves KV and logical commit semantics;
- [ ] v0.11 deterministic greedy output matches v0.10 eager output;
- [ ] speculative verification eager vs compile-aware logits/token history match;
- [ ] world-size 1/2 distributed semantics remain equivalent to the target reference;
- [ ] cancellation/streaming/backpressure regressions still pass;
- [ ] no compile-specific policy appears in EngineCore/Scheduler/AsyncEngine;
- [ ] all v0.1-v0.10 regression tests continue to pass;
- [ ] an executable example shows miss → compile → hit → recapture/fallback behavior;
- [ ] exact release-candidate GitHub Actions `./scripts/check.sh` passes before synchronization to `main`.

## 17. Explicit non-goals

- production CUDA Graph capture/replay;
- claiming GPU speedups from CPU CI;
- aggressive dynamic-shape optimization;
- automatic production bucketing heuristics;
- compiling Scheduler, EngineCore, AsyncEngine, or Python KV ownership logic;
- moving collectives into captured graphs;
- real multi-GPU/NCCL graph capture;
- custom Triton/CUDA PagedAttention — v0.12;
- prefix-aware scheduling policy — v0.13.

## 18. Learning goals

After v0.11, a learner should be able to explain:

- eager control plane vs compilable tensor plane;
- why Python KV ownership is a poor graph boundary;
- what a shape signature represents;
- compile cache hit vs recompilation/recapture;
- what graph breaks are and why they occur;
- why eager fallback is necessary for correctness-first serving;
- why compilation state belongs near rank-local model execution;
- why `torch.compile` and CUDA Graph are related optimization ideas but not the same mechanism;
- why a compiled serving runtime still needs the exact same Scheduler, transactions, cancellation, and streaming semantics.

## 19. Next version

`v0.12` introduces an **optimized PagedAttention path**: preserve the readable Python/cache reference, add a vectorized PyTorch oracle first, then optionally an educational Triton/CUDA kernel with numerical equivalence and fallback checks.

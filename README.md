# mini-vLLM

A small, branch-versioned LLM-serving runtime for learning the systems ideas behind vLLM.

## Current line

`main` is always the newest exact-CI-validated release. v1.1 turns the v1.0 logical Executor/Worker boundary into a real operating-system process boundary while preserving the same serving semantics.

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
v0.12 reference + vectorized PagedAttention
v0.13 prefix-aware scheduling + cache-aware admission
v1.0  integrated educational runtime
v1.1  real multiprocessing executor transport         ← current
```

Historical builders remain available so each mechanism can still be studied in isolation.

## v1.1 mental model

```text
parent process

caller
  ↓
AsyncEngine
  ↓
EngineCore                     logical request truth
  ↓
PrefixAwareBudgetedScheduler   policy
  ↓
MultiprocessExecutor
  │
  ├──── explicit command + correlation ID ────┐
  │                                            │ IPC
  ▼                                            ▼
child rank 0                              child rank 1
Worker                                    Worker
├─ ModelRunner                            ├─ ModelRunner
├─ transactional paged KV                ├─ transactional paged KV
├─ vectorized PagedAttention             ├─ vectorized PagedAttention
└─ compile-aware projection              └─ compile-aware projection
  │                                            │
  └──── serialized tensor result / error ──────┘
                       ↓
              parent-side gather
                       ↓
              EngineCore logical commit
                       ↓
              committed async events
```

The central v1.1 ownership rule is:

```text
parent owns logical truth
child owns physical model/KV state
IPC owns the evidence crossing that boundary
```

The parent never reaches into a live child `Worker`, cache backend, model runner, block table, or tensor storage.

## Stable construction API

The default builder creates v1.1:

```python
from mini_vllm import Request, RuntimeConfig, create_engine

engine = create_engine(runtime_config=RuntimeConfig())
```

`RuntimeConfig` now also contains `MultiprocessConfig`:

```python
from mini_vllm import MultiprocessConfig, RuntimeConfig

config = RuntimeConfig(
    multiprocess=MultiprocessConfig(
        start_method="spawn",
        command_timeout_s=30.0,
        shutdown_timeout_s=2.0,
    )
)
```

The transport knobs control process creation and lifecycle timeouts, not Scheduler policy or physical KV addressing.

Historical comparisons remain explicit:

```python
v13 = create_engine("v0.13")
v10 = create_engine("v1.0")   # in-process integrated runtime
v11 = create_engine("v1.1")   # process-isolated execution runtime
```

## Explicit IPC protocol

v1.1 uses a small fixed command vocabulary instead of arbitrary remote method invocation. Commands cover:

```text
prefix estimate / prepare
KV capacity
normal execute
speculative execute
commit work
checkpoint / rollback / checkpoint commit
free
cache stats
execution snapshot
shutdown
```

Every command carries a monotonically increasing correlation ID. Child-side exceptions become structured error responses; unexpected EOF, timeout, or process exit is a transport failure.

Logits cross the boundary through an explicit CPU `TensorPayload`. No live tensor storage or cache object is treated as a parent-owned handle.

## Failure atomicity across IPC

Ordinary execution is protected by remote checkpoints:

```text
parent asks every rank for checkpoint
              ↓
      all checkpoints acknowledged
              ↓
       execute on child ranks
              ↓
       all results acknowledged?
        ├─ yes → gather → commit checkpoint → logical commit
        └─ no  → rollback reachable ranks → no logical commit
```

A child-reported model/runtime error does not automatically kill that child; if rollback succeeds, the process can remain healthy. An unexpected process death is surfaced as `WorkerProcessDied`. If physical state can no longer be proven consistent, the executor becomes unhealthy and EngineCore does not commit the step.

Speculative decoding keeps the v0.7 transaction contract, but the checkpoint identities are now transport-safe handles whose real cache checkpoints remain inside each child process.

## Lifecycle and shutdown

A v1.1 engine owns child processes and therefore exposes explicit cleanup:

```python
await engine.aclose()   # async-safe: wait idle, then shutdown/join workers
# or, when the driver is already idle:
engine.close()
```

Normal shutdown sends an explicit `shutdown` command, joins each process, and terminates only as a final cleanup fallback. The test suite verifies no worker remains alive after close.

## One public observability path

The v1.0 API remains intact:

```python
snapshot = engine.snapshot()
```

v1.1 extends rank observations with:

- execution boundary (`multiprocess-ipc`);
- child process ID;
- worker lifecycle state;
- serving commands sent/completed;
- structured remote-error count;
- transport-failure count.

Existing request, Scheduler, cache, speculation, attention and compile observations remain available. Pure observability reads are deliberately excluded from serving-command counters, so two snapshots with no intervening runtime action remain equal.

## Core correctness contract

The condensed invariants live in `docs/INVARIANTS.md`. v1.1 adds another separation:

```text
logical commit
    requires
acknowledged remote physical execution
```

and:

```text
unknown child physical state
    => executor unhealthy
    => no logical commit
```

The earlier distinctions still hold:

```text
logical history != computed-token frontier != physical KV ownership
PrefixReuseHint != reservation != ownership != computed tokens
```

## Walkthrough

```bash
pip install -e '.[dev]'
python examples/16_multiprocess_transport.py
```

The example prints the parent PID and child worker PIDs, streams committed output, exposes transport/attention/compile counters through `snapshot()`, then closes the runtime and proves all workers are gone.

## Verify

```bash
./scripts/check.sh
```

The v1.1 gate runs all historical regressions plus examples 01–16. The first implementation candidate reached 91 passing pytest tests; only the exact final release commit with green repository CI may advance `main`.

Read `Agents.md`, `Plan.md`, `docs/ARCHITECTURE.md`, `docs/INVARIANTS.md`, and `version.md` before changing architecture.

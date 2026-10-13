# mini-vLLM v1.1 — real multiprocessing executor transport (planning branch)

## Version identity

- Branch: `v1.1`
- Base: validated `v1.0` / `main`
- Previous: `v1.0`
- Theme: turn the existing logical Executor/Worker boundary into an actual process boundary with explicit IPC, lifecycle, health and failure semantics
- Status: planning; runtime remains exactly v1.0 until v1.1 acceptance criteria pass

## Why this version exists

v0.8 introduced `EngineCore → Executor → Worker` as an architectural boundary, and v0.9 added logical multi-rank execution. In v1.0 those boundaries are clean but still live inside one Python process.

v1.1 has a genuinely new learning target:

> What changes when execution crosses a real operating-system process boundary?

This is not a throughput optimization project. It is a transport, ownership, serialization and failure-isolation lesson.

## Primary mental model

```text
AsyncEngine / EngineCore
        │
        │ logical ExecutionRequest
        ▼
MultiprocessExecutor
        │
        │ serialized IPC command
        ▼
worker process
        │
        ├─ ModelRunner
        ├─ KV cache
        └─ rank-local execution state
        │
        │ serialized result / error
        ▼
MultiprocessExecutor
        │
        ▼
EngineCore logical commit
```

For multiple logical ranks:

```text
EngineCore
   ↓
MultiprocessExecutor
   ├─ worker process rank 0
   ├─ worker process rank 1
   └─ ...
        ↓
transport-level result coordination
        ↓
collective / gathered logits
        ↓
logical commit
```

## Core design rule

The process boundary must strengthen—not blur—the v1.0 ownership contract.

```text
EngineCore owns logical request truth
Worker process owns physical model/KV execution state
IPC carries explicit messages
```

The parent process must not rely on direct Python object access to child `Worker`, cache tensors, block tables or model runner internals.

## Planned abstractions

Names may change during implementation, but the concepts should remain explicit.

### `MultiprocessExecutor`

An `Executor` implementation that sends execution/cache lifecycle commands across IPC rather than invoking a local Worker directly.

Responsibilities:

- worker startup and readiness handshake;
- request/result correlation;
- serialization boundary;
- timeout/EOF/process-death detection;
- orderly shutdown;
- converting child failures into explicit parent-side execution failures.

### Transport protocol

Prefer a small explicit message vocabulary over arbitrary remote method invocation.

Candidate commands:

```text
PREPARE_REQUEST
ESTIMATE_PREFIX_REUSE
ENSURE_CAPACITY
EXECUTE
EXECUTE_SPECULATIVE
COMMIT_WORK
CHECKPOINT
ROLLBACK
COMMIT_CHECKPOINT
FREE
CACHE_STATS
SNAPSHOT_EXECUTION
SHUTDOWN
```

Each request should carry a monotonically increasing correlation ID so a response can be matched deterministically.

### Worker process lifecycle

```text
CREATED
  ↓
STARTING
  ↓
READY
  ↓
RUNNING
  ├─ healthy commands
  ├─ execution error -> ERROR response
  └─ process death -> DEAD
  ↓
STOPPING
  ↓
STOPPED
```

The parent must be able to distinguish a model/runtime error response from an unexpected process exit.

## Serialization contract

IPC messages must contain logical/serializable data, not live cache/model objects.

Good transport payloads include:

- request IDs;
- token ID tuples;
- start positions;
- prompt lengths;
- checkpoint metadata expressed as transport-safe values;
- logits/results represented in an explicit tensor-transfer format;
- cache statistics and observability counters.

Avoid transporting:

- Python bound methods;
- live `Worker` references;
- cache backend objects;
- Scheduler objects;
- block-table ownership handles as policy inputs.

## Checkpoint / rollback across processes

v1.0 already requires physical execution to remain atomic with respect to EngineCore logical commit. v1.1 must preserve that invariant when the cache lives in a child process.

```text
parent asks child to checkpoint
        ↓
child acknowledges checkpoint identity
        ↓
child executes
        ↓
result success
   ├─ yes -> commit checkpoint / return result
   └─ no  -> rollback before surfacing failure
```

A transport failure after mutation but before an acknowledged result is the difficult case. The educational reference implementation should choose a conservative rule: if physical state cannot be proven consistent, mark that worker/executor unhealthy and do not logically commit the step.

## Failure semantics

v1.1 should teach several distinct failures explicitly.

### Child reports execution error

The process remains alive but returns a structured error. Parent surfaces the error only after required rollback succeeds.

### Child exits unexpectedly

The parent detects EOF/process death. No logical step may commit from an unacknowledged result.

### Parent cancellation

Cancellation remains an EngineCore/AsyncEngine lifecycle concern. It should send explicit release/abort commands only at atomic step boundaries rather than killing a healthy worker process arbitrarily.

### Shutdown

Normal shutdown must terminate and join every child process. Tests must prove there are no orphan workers after engine teardown.

## Distributed compatibility

v1.1 is about real process transport, not real multi-node/NCCL serving.

An educational implementation may run one process per logical rank on one machine while preserving the existing reference collective semantics in the parent or through a narrow coordination layer.

Required invariant:

```text
world_size=1 and world_size=2
produce the same deterministic committed output
as v1.0 for equivalent configuration
```

## Observability

The v1.0 `snapshot()` contract should remain the public teaching surface.

v1.1 may extend execution observations with transport facts such as:

- process ID;
- worker lifecycle state;
- commands sent/completed;
- transport failures;
- worker restarts, if restart is implemented;
- parent/child boundary type.

It must not expose child-process physical cache handles to Scheduler policy.

## Preferred implementation order

1. Define transport-safe command/result dataclasses.
2. Add one child-process Worker with startup/shutdown handshake.
3. Implement ordinary `execute()` through IPC.
4. Move cache lifecycle calls behind the same process boundary.
5. Add checkpoint/rollback semantics.
6. Add speculative execution transport.
7. Generalize to one process per logical rank.
8. Add public transport observability.
9. Prove deterministic equivalence and failure cleanup.

## Acceptance criteria

- [ ] at least one real `multiprocessing.Process` owns a Worker/model/cache;
- [ ] EngineCore/parent does not directly invoke methods on the child Worker object;
- [ ] command/result protocol is explicit and serializable;
- [ ] ordinary execution crosses IPC and returns correct logits;
- [ ] prefix prepare/estimate/capacity/free operations cross the same boundary;
- [ ] checkpoint/rollback and speculative execution remain correct across IPC;
- [ ] child-reported execution failure cannot partially commit EngineCore logical state;
- [ ] unexpected worker death is detected and surfaced explicitly;
- [ ] normal shutdown joins child processes with no orphan worker left running;
- [ ] world-size 1 and 2 deterministic output matches validated v1.0;
- [ ] v1.0 public snapshot remains usable and gains transport-level observations without exposing physical ownership handles;
- [ ] all v0.1–v1.0 regression tests continue to pass;
- [ ] executable example demonstrates parent process → IPC → worker process → result flow;
- [ ] exact release candidate passes repository CI before `main` advances.

## Non-goals

- multi-node networking;
- NCCL/Gloo production collectives;
- GPU process isolation or CUDA IPC;
- process pools optimized for throughput;
- automatic elastic worker replacement;
- OpenAI-compatible HTTP serving;
- production model/tokenizer loading;
- Triton/CUDA PagedAttention kernels;
- remote/disaggregated KV cache.

## Learning goal

By the end of v1.1 a reader should understand the difference between an architectural abstraction boundary and a real OS isolation boundary: serialization becomes explicit, ownership becomes remote, lifecycle must be managed, and failures become transport events rather than ordinary in-process exceptions.

## Next direction

After v1.1, genuinely new learning targets may include real GPU kernels or production-style model loading. Neither belongs in this transport-focused version.

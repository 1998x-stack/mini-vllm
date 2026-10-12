# mini-vLLM v0.8 — EngineCore / Executor / Worker boundary (planning branch)

## 1. Version identity

- **Branch:** `v0.8`
- **Base:** latest validated `v0.7` / `main`
- **Previous version:** `v0.7`
- **Theme:** separate centralized serving policy from execution transport
- **Status:** planning / implementation target; runtime still behaves as v0.7 until acceptance criteria pass

## 2. Why v0.8 follows v0.7

By v0.7 the in-process runtime already has explicit semantics for:

- request lifecycle;
- token-budget scheduling;
- paged KV ownership;
- prefix sharing;
- preemption/recompute;
- speculative checkpoint/rollback.

The remaining architectural coupling is that Engine currently invokes ModelRunner/KV work directly. v0.8 asks:

> Which state and decisions must remain centralized in EngineCore, and which work can move behind an Executor/Worker interface without changing semantics?

## 3. Primary mental model

```text
EngineCore
  request state
  Scheduler
  admission / victim policy
        │
        ▼
ExecutionRequest
        │
        ▼
Executor
        │
        ▼
Worker
  ModelRunner
  KV backend execution
        │
        ▼
ExecutionResult
        │
        ▼
EngineCore commit/update
```

The first implementation should remain single-process through `LocalExecutor`. The learning target is ownership boundaries, not multiprocessing complexity.

## 4. Required abstractions

Introduce small explicit data/contracts such as:

```text
EngineCore
ExecutionRequest
ExecutionResult
Executor
LocalExecutor
Worker
```

Do not move waiting/running queues or victim selection into Worker.

## 5. Ownership rules

### EngineCore owns policy

EngineCore should continue to own:

- request registry and logical history;
- Scheduler;
- admission ordering;
- token budgets;
- preemption victim policy;
- finishing/cancellation decisions;
- logical commit of sampling/speculation results.

### Worker owns execution mechanism

Worker may own or invoke:

- ModelRunner;
- physical KV backend operations;
- model forward execution;
- target verification execution;
- execution-local tensors.

### Executor owns transport/orchestration

Executor receives explicit execution requests and returns explicit execution results. `LocalExecutor` directly invokes a Worker but must use the same contract a future remote/multiprocess executor could use.

## 6. Critical invariants

### Semantic-equivalence invariant

Moving execution behind `LocalExecutor` must not change deterministic outputs or request state transitions relative to v0.7.

### Policy-ownership invariant

Worker must not decide:

- who gets admitted;
- who gets preempted;
- scheduler priority;
- when a request is logically finished.

### Execution-contract invariant

Everything required for execution must cross the EngineCore→Executor boundary explicitly rather than being read from hidden Engine internals.

### Transaction invariant

Speculative checkpoint/verification/rollback semantics must survive the executor boundary without allowing rejected tokens to leak into logical history.

## 7. Preferred implementation order

```text
1. define ExecutionRequest / ExecutionResult
2. wrap current ModelRunner + cache in Worker
3. add Executor interface
4. add LocalExecutor reference path
5. refactor Engine into EngineCore coordination + executor call
6. preserve preemption behavior
7. preserve speculative transaction behavior
8. add equivalence tests
```

## 8. Acceptance criteria

v0.8 cannot be frozen until:

- [ ] normal prefill/decode work executes through `LocalExecutor`;
- [ ] Worker contains no waiting/running queue policy;
- [ ] EngineCore still owns admission and preemption decisions;
- [ ] v0.7 greedy/speculative outputs remain identical after refactor;
- [ ] prefix cache and paged block ownership remain correct;
- [ ] preemption/recompute regressions pass;
- [ ] speculative accept-all/reject-first/partial-accept regressions pass;
- [ ] execution request/result payloads are inspectable and tested;
- [ ] a local executor failure is surfaced explicitly rather than silently corrupting request state;
- [ ] an executable example traces EngineCore → Executor → Worker → result;
- [ ] all v0.1-v0.7 tests continue to pass;
- [ ] validated implementation is synchronized to `main`.

## 9. Explicit non-goals

- multiprocessing in the first increment;
- RPC/network transport;
- multi-GPU execution;
- tensor parallelism — `v0.9`;
- async streaming — `v0.10`;
- CUDA Graph or compiled execution;
- Triton/CUDA PagedAttention.

## 10. Next version

`v0.9` introduces **distributed / tensor-parallel execution concepts**. The executor/worker boundary from v0.8 is the prerequisite: distribution should change transport/topology, not force scheduler/cache semantics to be redesigned.

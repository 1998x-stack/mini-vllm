# mini-vLLM v0.8 — EngineCore / Executor / Worker boundary

## 1. Version identity

- **Branch:** `v0.8`
- **Previous version:** `v0.7`
- **Theme:** separate centralized serving policy from execution mechanism/transport
- **Status:** implementation complete; boundary-focused local validation passed; full repository CI is the release gate

v0.8 preserves the serving algorithms from v0.7 and changes where execution happens:

```text
v0.7
Engine
 ├─ Scheduler / request state
 ├─ KV lifecycle calls
 └─ ModelRunner.execute()

v0.8
EngineCore
  │ explicit policy-free payload
  ▼
Executor
  ▼
Worker
 ├─ ModelRunner
 └─ KV backend
  │
  ▼
ExecutionResult
  │
  ▼
EngineCore logical commit
```

The reference executor remains single-process. The learning goal is ownership boundaries, not multiprocessing.

## 2. Why v0.8 follows v0.7

By v0.7 the runtime already has explicit semantics for request lifecycle, token-budget scheduling, paged KV ownership, prefix sharing, preemption/replay, and speculative checkpoint/rollback.

The remaining coupling was architectural: the central Engine directly invoked physical model/cache mechanisms. That makes future process or distributed execution difficult because policy and mechanism share hidden Python objects.

v0.8 therefore asks:

> What information truly needs to cross from centralized policy into execution?

## 3. Primary invariant: Request objects do not cross the execution boundary

`WorkItem` belongs to policy/scheduler coordination and contains a `Request` object. The executor payload deliberately strips that policy object:

```text
WorkItem(request=..., token_ids, start_pos)
       │
       ▼
EngineCore conversion
       │
       ▼
ExecutionItem(request_id, token_ids, start_pos)
```

Worker therefore cannot inspect request status, scheduler queues, preemption counters, or admission priority through the execution payload.

## 4. New execution contracts

`ExecutionItem` is a policy-free contiguous model-work description. `ExecutionRequest` and `ExecutionResult` form the normal forward contract. `SpeculativeExecutionRequest` and `SpeculativeExecutionResult` carry target verification work without allowing SpeculativeEngine to reach directly into ModelRunner.

EngineCore validates the request-id set returned by the Executor before committing logical state.

## 5. Ownership model

### EngineCore owns policy and logical state

EngineCore retains request registry/history, Scheduler, admission ordering, token budgets, preemption victim selection, finish decisions, sampling, speculative logical commit, and executor-result validation.

### Worker owns execution mechanism

Worker owns/invokes ModelRunner, physical KV backend, forward execution, speculative target execution, and cache lifecycle mechanisms requested by EngineCore. Worker intentionally has no Scheduler, waiting queue, running queue, or victim policy.

### Executor owns transport/orchestration

Executor defines the explicit boundary. LocalExecutor invokes a local Worker directly but uses the same contracts a future process/RPC executor can implement.

## 6. Cache lifecycle also crosses Executor

EngineCore no longer directly uses the cache for policy execution paths. Explicit executor operations include `prepare_request`, `ensure_capacity`, `commit_work`, `checkpoint`, `rollback`, `commit_checkpoint`, `free`, and `cache_stats`.

For educational backward compatibility, `engine.cache` and `engine.model_runner` remain read-through debug views when using LocalExecutor; EngineCore policy itself does not use those views.

## 7. Historical-version compatibility

Older version builders may still pass `cache + model_runner` into Engine/SpeculativeEngine. EngineCore automatically wraps them in Worker + LocalExecutor, so current-main regression tests can still instantiate v0.1-v0.7 without rewriting frozen historical branches. `Engine = EngineCore` remains a compatibility alias.

## 8. Failure semantics

Executor failures are explicit exceptions. EngineCore updates `num_computed_tokens` and logical output only after a successful ExecutionResult.

```text
schedule work
    ↓
Executor.execute() raises
    ↓
exception surfaces
    ↓
no logical token commit
no computed-token advance
```

## 9. Important source files

```text
mini_vllm/execution/types.py
mini_vllm/execution/executor.py
mini_vllm/execution/worker.py
mini_vllm/core/engine.py
mini_vllm/speculative/engine.py
mini_vllm/versions/v0_8/engine.py
tests/test_executor_boundary.py
tests/test_v08_integration.py
examples/09_executor_boundary.py
```

Recommended comparison:

```bash
git diff v0.7..v0.8 -- mini_vllm tests examples version.md
```

## 10. Validation

v0.8 adds tests for execution payload isolation, Worker policy isolation, normal/speculative LocalExecutor routing, failure-before-commit behavior, v0.7/v0.8 deterministic equivalence, and version composition.

The focused v0.8 boundary harness passed **5/5** locally. GitHub Actions runs the canonical `./scripts/check.sh` against the complete checkout; that full-suite result is the gate before v0.8 is synchronized to `main` and treated as frozen.

## 11. Learning goals

After v0.8, the learner should be able to explain why scheduling policy should not live in workers; why mutable Request objects should not cross worker boundaries; the difference between execution payload and logical request state; why LocalExecutor comes before multiprocessing; how preemption and speculative transactions survive the boundary; and why distribution should change topology rather than scheduler semantics.

## 12. Explicit non-goals

- multiprocessing/RPC transport;
- worker crash recovery;
- multi-GPU execution;
- tensor parallel collectives;
- async streaming;
- CUDA Graph / compile integration;
- optimized Triton/CUDA PagedAttention.

## 13. Next version

After the full repository CI gate passes, `v0.9` introduces distributed / tensor-parallel execution concepts on top of the v0.8 boundary. Distribution changes execution topology, not request scheduling semantics.

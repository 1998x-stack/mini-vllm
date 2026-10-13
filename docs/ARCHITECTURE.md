# mini-vLLM Architecture — v0.8

## Control plane vs execution plane

```text
CONTROL / POLICY

Request registry
     │
     ▼
 EngineCore
 ├─ Scheduler
 ├─ admission
 ├─ token budgets
 ├─ victim policy
 ├─ sampling / logical commit
 └─ finish decisions
     │
     ▼
ExecutionRequest

================ boundary ================

EXECUTION

 LocalExecutor
     │
     ▼
   Worker
 ├─ ModelRunner
 └─ KV backend
     │
     ▼
ExecutionResult

================ boundary ================

     │
     ▼
 EngineCore validates + commits
```

## Boundary rule

The executor payload carries execution facts, not mutable policy objects:

```text
ExecutionItem = request_id + token_ids + start_pos
```

It does not carry `Request`, Scheduler queues, priority, or victim-selection state.

## Cache ownership

EngineCore decides *whether* a request should run or another request should be preempted. Worker/cache implement *how* physical KV is allocated, released, checkpointed, and rolled back.

## Speculation

Speculative verification also crosses Executor. Draft proposal and logical acceptance remain centralized; target-model per-token logits and temporary KV mutation live behind Worker.

## Failure boundary

EngineCore commits `num_computed_tokens` and logical output only after a successful `ExecutionResult`. Executor exceptions surface without pretending the forward pass completed.

## Historical compatibility

Current-main historical builders may still provide `cache + model_runner`; EngineCore wraps them in `Worker + LocalExecutor`. Frozen historical branches themselves are not modified.

## Why LocalExecutor first?

A local reference path proves ownership and message contracts without adding multiprocessing, serialization, RPC, or distributed failure handling. v0.9 can therefore change topology without redesigning scheduler/cache semantics.

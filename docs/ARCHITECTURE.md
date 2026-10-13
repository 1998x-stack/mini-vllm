# mini-vLLM Architecture — v0.11

## Eager control and execution orchestration

```text
AsyncEngine
  ↓
EngineCore / Scheduler
  ↓
DistributedExecutor
  ↓
WorkerGroup
  ↓
rank-local Worker
```

These layers remain eager. They own lifecycle, policy, topology, transactions, and KV allocation.

## Compile-aware rank-local boundary

```text
TinyDecoder trunk + external KV                         eager
        ↓
hidden trace [tokens, hidden]
        ↓
ExecutionShapeSignature
        ↓
CompiledKernelCache
   ├─ miss → torch.compile(PureTensorProjectionKernel)
   ├─ hit  → reuse callable
   └─ failure → eager fallback (policy-controlled)
        ↓
rank-local logits [tokens, local_vocab]
        ↓
ReferenceCollective                                     eager
```

The compiled function sees only tensors. It never sees request IDs, Scheduler objects, block tables, cancellation state, asyncio queues, or collective topology.

## Why projection first?

The vocabulary projection is a real model tensor operation with a stable, inspectable input/output contract. It demonstrates signature specialization and rank-local compile ownership without forcing Python KV management into Dynamo. v0.12 can widen optimization toward attention while this projection path remains a simple compile-aware reference.

## Failure semantics

Compile failure occurs before any extra serving commit. With fallback enabled the same eager tensor kernel runs and the surrounding KV/logical transaction proceeds unchanged. With strict mode the exception surfaces through the existing Executor/AsyncEngine error path.

# mini-vLLM Architecture — v0.9

## Control plane remains singular

```text
EngineCore
  request registry
  Scheduler
  token budget
  preemption policy
  sampling / logical commit
        │
        ▼
ExecutionRequest
```

No rank owns a Scheduler. Distribution begins below the Executor boundary.

## Execution topology

```text
ExecutionRequest
      │
      ▼
DistributedExecutor
      │
      ▼
WorkerGroup(world_size=N)
  ┌──────┼──────┐
  ▼      ▼      ▼
rank0  rank1  rankN-1
Worker Worker Worker
  │      │      │
  ▼      ▼      ▼
replicated TinyDecoder trunk + replicated per-rank KV
  │      │      │
  ▼      ▼      ▼
rank-local vocabulary projection shard
  └──────┼──────┘
         ▼
ReferenceCollective.all_gather
         │
         ▼
full target logits
         │
         ▼
ExecutionResult
         │
         ▼
EngineCore commit
```

## Rank ownership

`RankConfig(rank, world_size)` is explicit. WorkerGroup validates that ranks are exactly `0..world_size-1` and all agree on the same world size. No global rank variables are used.

## Tensor-parallel reference path

The first integrated model shard is the LM head:

```text
full weight W[V, H]
        │
        ├─ rank 0 rows [0:k0)
        ├─ rank 1 rows [k0:k1)
        └─ ...

same hidden h on every rank
        │
        ▼
local_logits_r = W_r @ h
        │
        ▼
all_gather(local_logits_0, ...)
        │
        ▼
full logits[V]
```

Uneven vocabulary sizes use balanced contiguous shard ranges. `ReferenceTensorParallel` also includes row-parallel linear reconstruction via `all_reduce_sum` as a mathematical teaching oracle.

## KV scope

v0.9 does **not** claim distributed KV partitioning. Each logical rank owns an independent replicated `TransactionalPrefixCachingPagedKVCache`. Prefix caching, preemption, and speculative rollback occur on every replica and WorkerGroup checks replicated metadata stays consistent.

## Failure atomicity

A normal distributed forward can fail after an earlier rank already wrote K/V. WorkerGroup therefore checkpoints every participating rank at the WorkItem start position before executing:

```text
checkpoint all ranks
      ↓
rank-local forwards
      │
      ├─ all succeed → validate/keep physical work
      │
      └─ any failure → rollback every rank → raise
```

EngineCore still commits `num_computed_tokens` and logical outputs only after a successful distributed `ExecutionResult`.

Speculative decoding already has an outer transactional checkpoint. A rank failure during speculative target execution surfaces to SpeculativeEngine, which rolls the distributed checkpoint back on every rank.

## Why no NCCL yet?

The goal of v0.9 is to make rank-local tensors, collective reconstruction, topology, and failure semantics visible. Real process groups would add transport and device complexity before the mathematical ownership model is learned. A future optimized/distributed version can replace `ReferenceCollective` without moving Scheduler policy into workers.

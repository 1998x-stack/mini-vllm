# mini-vLLM v0.9 — logical multi-rank + reference tensor parallel

## 1. Version identity

- **Branch:** `v0.9`
- **Previous version:** `v0.8`
- **Theme:** WorkerGroup, explicit rank/world-size, reference collectives, and educational tensor-parallel execution
- **Status rule:** implementation is eligible for release only when the exact v0.9 commit has a green repository CI run; the released `main` pointer is the source of truth for validated state

v0.8 established `EngineCore -> Executor -> Worker`. v0.9 changes the execution topology without moving Scheduler/admission/preemption/sampling policy out of EngineCore.

## 2. Primary mental model

```text
EngineCore
   │ policy-free ExecutionRequest
   ▼
DistributedExecutor
   ▼
WorkerGroup(world_size=N)
 ┌────┼────┐
 ▼    ▼    ▼
rank0 rank1 ...
 │    │
rank-local tensor work
 └────┼────┘
      ▼
ReferenceCollective
      ▼
ExecutionResult
      ▼
EngineCore logical commit
```

The first serving-integrated tensor-parallel operation shards the LM-head vocabulary rows:

```text
full W[V,H]
   ├─ rank 0: W[0:k0]
   ├─ rank 1: W[k0:k1]
   └─ ...

replicated hidden h
   ↓
local logits shards
   ↓
all_gather
   ↓
full logits[V]
```

Transformer trunk and KV cache remain replicated per rank on purpose. v0.9 teaches topology and collective semantics before real process/GPU transport or distributed KV ownership.

## 3. Added mechanisms

### Explicit rank ownership

`RankConfig(rank, world_size)` validates `world_size > 0` and `0 <= rank < world_size`. No hidden global rank state is used.

### Reference collectives

`ReferenceCollective` provides explicit single-process mathematical oracles:

- `all_gather`
- `all_reduce_sum`

### Tensor-parallel math oracle

`ReferenceTensorParallel` demonstrates:

- output/column-sharded linear reconstruction through gather;
- input/row-sharded linear reconstruction through all-reduce sum.

Balanced contiguous shard ranges support uneven dimensions.

### Serving-integrated vocabulary parallelism

`VocabParallelLMHead` stores/computes only one vocabulary-row shard. `VocabParallelTinyRunner` executes the replicated TinyDecoder trunk against a rank-local KV cache and returns only that rank's logits shard for both normal and speculative target execution.

### WorkerGroup

`WorkerGroup` owns exactly one ordered Worker per rank `0..world_size-1`. It owns execution topology only; it has no Scheduler, waiting/running queue, admission policy, or preemption-victim logic.

### DistributedExecutor

`DistributedExecutor` satisfies the existing v0.8 `Executor` contract. EngineCore therefore needs no branch for distributed scheduling.

## 4. TinyDecoder contract change

`TinyDecoderLM` adds `forward_chunk_hidden_trace()`, which materializes K/V and returns final-normalized hidden states. The historical path remains:

```text
hidden trace -> full lm_head -> full logits
```

v0.9 instead uses:

```text
hidden trace -> local lm_head shard -> collective -> full logits
```

This keeps model semantics shared between the reference and distributed paths.

## 5. KV-cache scope

v0.9 uses explicit replicated caches:

```text
rank 0 -> TransactionalPrefixCachingPagedKVCache #0
rank 1 -> TransactionalPrefixCachingPagedKVCache #1
...
```

WorkerGroup requires replicas to agree on reusable-prefix length, capacity outcome, and cache statistics. The version does not claim KV-head sharding or production distributed cache management.

## 6. Distributed failure atomicity

A normal forward may fail after an earlier rank has already written K/V. Therefore logical no-commit alone is insufficient.

Before rank-local normal execution, WorkerGroup checkpoints every participating rank at the WorkItem start frontier:

```text
checkpoint all ranks
      ↓
rank-local forwards
      ├─ all succeed -> gather logits -> keep physical work
      └─ any failure -> rollback every rank -> surface exception
```

Only after a successful `ExecutionResult` does EngineCore advance `num_computed_tokens` or logical output.

Speculative decoding already creates a transaction checkpoint before target verification. `DistributedExecutor.checkpoint/rollback` expands that transaction across all rank replicas, preserving v0.7 accept/reject/exception semantics.

## 7. Critical invariants

- **Policy-location:** one EngineCore/Scheduler policy exists regardless of world size.
- **Rank consistency:** ranks agree on world size, request identity, token positions, operation ordering, prefix reuse, and replicated cache lifecycle.
- **Collective:** rank-local shards plus the specified collective reconstruct the unsharded mathematical result.
- **Semantic equivalence:** deterministic v0.9 world-size 1/2 execution matches the v0.8 target-model token history.
- **Distributed failure:** a partial rank failure cannot leave surviving replicas physically advanced while EngineCore stays logically uncommitted.

## 8. Mechanisms intentionally unchanged

v0.9 preserves:

- v0.4 token-budget scheduling;
- v0.5 prefix-cache ownership semantics;
- v0.6 preemption/recompute policy;
- v0.7 speculative transaction semantics;
- v0.8 policy-free execution payloads and failure-before-logical-commit rule.

## 9. Important source files

```text
mini_vllm/distributed/rank.py
mini_vllm/distributed/collective.py
mini_vllm/distributed/tensor_parallel.py
mini_vllm/distributed/worker_group.py
mini_vllm/distributed/executor.py
mini_vllm/model/tensor_parallel_runner.py
mini_vllm/model/tiny_transformer.py
mini_vllm/versions/v0_9/engine.py
tests/test_distributed.py
tests/test_v09_integration.py
examples/10_tensor_parallel.py
```

Recommended comparison:

```bash
git diff v0.8..v0.9 -- mini_vllm tests examples version.md
```

## 10. Validation contract

The exact commit promoted to `main` must pass GitHub Actions `./scripts/check.sh`. Coverage includes:

- invalid rank/world-size ownership;
- all-gather/all-reduce semantics;
- uneven vocabulary sharding;
- column/row parallel reconstruction;
- DistributedExecutor/WorkerGroup composition;
- absence of Scheduler policy in WorkerGroup;
- v0.8 vs v0.9 deterministic token-history equivalence;
- world-size=1 degenerate topology;
- normal + speculative execution through WorkerGroup;
- rank failure rollback of replicated KV before logical commit;
- all historical regression tests;
- executable examples through `examples/10_tensor_parallel.py`.

## 11. Learning goals

After v0.9, a learner should be able to explain rank vs world size, local tensor shard vs global tensor, all-gather vs all-reduce, why Scheduler remains singular, why distribution lives below Executor, why replicated KV is an explicit simplification, and why distributed physical rollback is required in addition to logical no-commit.

## 12. Explicit non-goals

- multiprocessing/RPC transport;
- NCCL/process groups;
- real multi-GPU placement;
- attention/MLP tensor-parallel serving integration;
- KV-head sharding;
- pipeline/expert parallelism;
- production distributed fault recovery;
- async streaming/cancellation;
- optimized CUDA/Triton kernels.

## 13. Next version

`v0.10` introduces the async engine + streaming/cancellation boundary. It should wrap the synchronous EngineCore/WorkerGroup runtime without moving Scheduler or collective semantics into the delivery layer.

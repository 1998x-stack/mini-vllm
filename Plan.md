# mini-vLLM — Architecture-Driven Development Plan

> `main` is always the newest integrated validated runtime. Version branches are educational snapshots. `Plan.md` is the canonical forward sequence.

## 1. Learning goal

mini-vLLM removes one layer of LLM-serving “magic” at a time: request state, prefill/decode, continuous batching, paged KV, token budgets, prefix sharing, memory pressure/preemption, speculative transactions, execution boundaries, distribution, async delivery, and optimized kernels.

## 2. Reference architecture after v0.9

```text
Client / caller
    │
    ▼
Request
    │
    ▼
EngineCore
  Scheduler + logical state + policy
    │
    ▼
ExecutionRequest
    │
    ▼
Executor
    │
    ▼
WorkerGroup
 ┌──┴──┐
 ▼     ▼
rank0 rank1 ...
 local compute
    │
    ▼
reference collective
    │
    ▼
ExecutionResult
    │
    ▼
EngineCore commit
```

Policy remains centralized even when execution topology fans out.

## 3. Version ladder

```text
v0.1  static batching + contiguous KV
v0.2  continuous batching
v0.3  paged KV + block table
v0.4  token budget + chunked prefill
v0.5  prefix cache + shared ownership
v0.6  preemption + recompute
v0.7  speculative transaction
v0.8  EngineCore / Executor / Worker
v0.9  WorkerGroup + reference tensor parallel      ← current implementation
v0.10 async engine + streaming                     ← next
v0.11 graph/compile experiments
v0.12 optimized PagedAttention
v0.13 prefix-aware scheduling
v1.0  integrated educational serving runtime
```

## 4. Released architecture lessons

### v0.1–v0.4 — serving loop, scheduling, paged memory, token budgets

Build the smallest correct autoregressive runtime, then make sequence admission continuous, KV block-addressed, and token work explicitly budgeted.

### v0.5 — prefix caching

Full prompt blocks can become immutable shared cache entries with refcounts while mutable tails remain private.

### v0.6 — preemption

Logical token history survives memory pressure while invalid physical KV is released and later recomputed.

### v0.7 — speculative transactions

Draft tokens are temporary until target verification commits them. Rejected temporary K/V is rolled back.

### v0.8 — execution boundary

EngineCore owns policy/logical state; Executor owns transport/orchestration; Worker owns model + physical KV mechanism. Execution payloads contain facts, not mutable Request policy objects.

### v0.9 — logical multi-rank + reference tensor parallel

Primary question:

> How can one logical execution fan out across ranks without multiplying schedulers or changing target-model semantics?

Reference design:

- explicit `RankConfig(rank, world_size)`;
- `WorkerGroup` with exactly one ordered worker per rank;
- single-process `ReferenceCollective` with mathematical `all_gather` and `all_reduce_sum`;
- educational column/row parallel linear reconstruction;
- replicated TinyDecoder trunk and per-rank KV cache;
- vocabulary-row-sharded LM head through `VocabParallelTinyRunner`;
- `DistributedExecutor` satisfying the existing Executor contract;
- rank-local logits gathered into the same full logits as the v0.8 single-worker reference;
- normal-forward rank failure rolls replicated cache state back before surfacing to EngineCore;
- speculative verification reuses v0.7 distributed checkpoint/rollback semantics.

Architectural lesson:

```text
single centralized policy
        ≠
multi-rank execution topology
```

Intentional simplifications:

- logical ranks run in one Python process;
- no NCCL/process group;
- no real GPU placement;
- KV and Transformer trunk are replicated rather than partitioned;
- only the vocabulary projection is integrated into the serving model as a tensor-parallel shard.

Release rule: only an exact v0.9 commit with a green repository CI run may be synchronized to `main`. This statement remains the rule after release rather than encoding a transient candidate status in the plan.

## 5. Planned versions

### v0.10 — Async engine + streaming / cancellation

Introduce asynchronous request submission and token delivery without hiding Scheduler or WorkerGroup semantics. Teach output queues, cancellation, backpressure, and the distinction between synchronous EngineCore steps and asynchronous caller APIs.

### v0.11 — graph/compile-aware execution

Study static metadata, graph breaks, recapture conditions, and host overhead while preserving eager reference correctness.

### v0.12 — optimized PagedAttention

Progress from readable Python paged gather to vectorized PyTorch and then an educational Triton/CUDA path, always compared against the reference implementation.

### v0.13 — prefix-aware scheduling

Integrate prefix-reuse cost into admission and compare cache-aware policies while preserving starvation/fairness tests.

### v1.0

A compact runtime with explicit control plane, execution topology, paged/prefix-aware KV, speculative verification, async delivery, and reference/optimized attention paths.

## 6. Validation matrix

| Version | Key invariant |
|---|---|
| v0.1 | computed state advances only after forward execution |
| v0.2 | free sequence slots backfill |
| v0.3 | paged/contiguous storage preserve logits |
| v0.4 | scheduled tokens stay within budget |
| v0.5 | shared prefix blocks remain immutable/refcounted |
| v0.6 | preemption cannot leave stale KV ownership |
| v0.7 | rejected speculative state disappears |
| v0.8 | executor transport preserves semantics and excludes policy objects |
| v0.9 | rank-local shards + collective reconstruct single-worker semantics |

## 7. Scope discipline

Prefer a readable mathematical reference before transport/performance complexity. Distribution first means explicit topology and collective semantics, not immediate multiprocessing/NCCL. Future mechanisms remain in the roadmap until their dependencies are explicit.

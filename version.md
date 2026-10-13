# mini-vLLM v0.9 — distributed / tensor-parallel execution (planning branch)

## 1. Version identity

- **Branch:** `v0.9`
- **Base:** latest validated `v0.8` / `main`
- **Previous version:** `v0.8`
- **Theme:** WorkerGroup, rank/world-size execution, reference collectives, and educational tensor parallelism
- **Status:** planning / implementation target; runtime still behaves as v0.8 until acceptance criteria pass

v0.8 created the execution boundary:

```text
EngineCore -> Executor -> Worker
```

v0.9 asks the next question:

> How can one logical execution request be executed by several workers/ranks without moving scheduling policy out of EngineCore or changing model semantics?

---

## 2. Why v0.9 follows v0.8

Before v0.8, adding multiple workers would have forced scheduler state, request objects, cache internals, and model execution into the same distributed design problem.

v0.8 makes the boundary explicit. v0.9 can therefore change only the execution topology:

```text
v0.8
EngineCore
   │
   ▼
Executor
   │
   ▼
Worker

v0.9
EngineCore
   │
   ▼
Executor
   │
   ▼
WorkerGroup
 ├─ Worker(rank=0)
 ├─ Worker(rank=1)
 └─ ...
```

Scheduler/admission/preemption semantics should remain centralized and unchanged.

---

## 3. Primary mental model

Start with a deterministic, single-process simulation of ranks rather than real NCCL/multiprocessing:

```text
ExecutionRequest
      │
      ▼
DistributedExecutor
      │
      ▼
WorkerGroup(world_size=2)
   ┌──────────┴──────────┐
   ▼                     ▼
rank 0                 rank 1
local shard            local shard
   │                     │
   └──────────┬──────────┘
              ▼
      reference collective
              │
              ▼
       ExecutionResult
```

The first version teaches topology and tensor semantics; transport/performance comes later.

---

## 4. Required abstractions

### RankConfig

Represent at least:

```text
rank
world_size
```

Validate `0 <= rank < world_size` and make rank ownership explicit rather than relying on global state.

### WorkerGroup

A WorkerGroup owns an ordered set of workers/ranks and executes one logical request across them. It must not own Scheduler queues or request admission policy.

### Collective abstraction

Introduce a tiny reference collective interface, initially implemented with ordinary PyTorch tensors in one process. Candidate operations:

```text
all_reduce_sum
all_gather
broadcast
```

Only add operations actually required by the educational tensor-parallel path.

### DistributedExecutor

Implement the existing Executor contract while routing execution through WorkerGroup. EngineCore should not need version-specific branches to know whether execution is local or distributed.

---

## 5. Educational tensor parallelism

Start with one simple model operation, not full production Transformer sharding.

Preferred progression:

```text
reference Linear
      │
      ▼
shard weight across 2 ranks
      │
      ├─ rank 0 local matmul
      └─ rank 1 local matmul
      │
      ▼
reference collective
      │
      ▼
reconstruct full/reference output
```

Then integrate the smallest useful shard into the Tiny Decoder while retaining an unsharded correctness oracle.

The purpose is to understand:

- what is sharded;
- what each rank computes locally;
- where communication is required;
- which tensors must become identical again;
- why distribution is below EngineCore policy.

---

## 6. KV-cache scope

Do not immediately build production distributed KV management.

The first v0.9 increment may keep each educational worker's cache ownership simple as long as the contract is explicit. If KV is replicated, document that. If KV heads are partitioned, test reconstruction/equivalence explicitly.

Do not hide inconsistent ownership behind global dictionaries.

---

## 7. Critical invariants

### Policy-location invariant

Multiple workers must not create multiple schedulers. Admission, token budgets, victim selection, logical sampling/commit, and finish decisions remain in EngineCore.

### Rank-consistency invariant

All ranks participating in one logical execution must agree on request identity, token positions, and execution step ordering.

### Collective invariant

Reference collective operations must produce mathematically defined results independent of call-site ordering assumptions.

### Semantic-equivalence invariant

For deterministic target execution:

```text
single-worker reference output
             ==
multi-worker/tensor-parallel output
```

within an explicit numerical tolerance for logits and exactly for greedy token histories where applicable.

### Failure-visibility invariant

A rank/worker execution failure must surface through Executor. EngineCore must not commit logical request state for a partially completed distributed execution.

---

## 8. Interaction with earlier versions

v0.9 must preserve:

- v0.4 token-budget scheduling;
- v0.5 prefix-cache ownership semantics;
- v0.6 preemption policy;
- v0.7 speculative transaction correctness;
- v0.8 policy-free execution payloads and failure-before-commit rule.

Distributed topology is not permission to move those semantics into WorkerGroup.

---

## 9. Acceptance criteria

v0.9 cannot be frozen until all applicable items are true:

- [ ] `RankConfig` validates rank/world-size ownership;
- [ ] `WorkerGroup` executes deterministic multi-rank reference work;
- [ ] at least one reference collective is implemented and unit-tested;
- [ ] one educational tensor-parallel operation reconstructs the single-worker reference result;
- [ ] `DistributedExecutor` satisfies the existing Executor contract;
- [ ] EngineCore requires no distributed scheduling branch;
- [ ] WorkerGroup contains no waiting/running queues or victim policy;
- [ ] distributed execution failure does not advance logical request state;
- [ ] deterministic single-worker and multi-worker logits/token histories are equivalent;
- [ ] speculative execution remains correct across the executor topology chosen for v0.9;
- [ ] v0.1-v0.8 regression tests continue to pass;
- [ ] an executable example makes rank-local compute + collective reconstruction visible;
- [ ] repository CI passes on the exact release candidate;
- [ ] validated implementation is synchronized to `main` only after that CI gate.

---

## 10. Preferred implementation order

```text
1. RankConfig
2. reference collective interface
3. WorkerGroup
4. tiny sharded linear reference experiment
5. DistributedExecutor
6. single-vs-multi-worker equivalence tests
7. integrate smallest Tiny Decoder shard
8. speculative/preemption regression coverage
9. educational trace/example
```

Keep each layer inspectable before adding the next.

---

## 11. Explicit non-goals

- real multi-node deployment;
- NCCL process groups in the first increment;
- GPU peer-to-peer optimization;
- production tensor-parallel kernels;
- pipeline parallelism;
- expert parallelism / MoE routing;
- worker autoscaling;
- distributed fault recovery;
- async streaming — `v0.10`;
- CUDA Graph / compile integration — later;
- Triton/CUDA PagedAttention — later.

---

## 12. Learning goals

After v0.9, the learner should be able to explain:

- why EngineCore should remain single-policy even with many workers;
- rank vs world size;
- local tensor shard vs reconstructed global tensor;
- why tensor parallelism requires collectives;
- the difference between Executor transport and WorkerGroup topology;
- why a distributed failure must abort logical commit;
- how single-worker reference execution acts as a correctness oracle.

---

## 13. Next version

`v0.10` introduces the **async engine + streaming/cancellation** boundary. Once execution topology is explicit, request submission and output delivery can become asynchronous without hiding scheduling or worker semantics.

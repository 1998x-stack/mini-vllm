# mini-vLLM

A deliberately small, runnable, **branch-versioned** LLM-serving runtime for learning the systems ideas behind vLLM.

## Current line

`main` is always the newest validated release. `v0.9` is the current implementation line; a v0.9 commit is synchronized to `main` only when that exact commit passes repository CI.

```text
v0.1  static batching + contiguous KV
v0.2  continuous batching
v0.3  paged KV + block tables
v0.4  token budget + chunked prefill
v0.5  prefix caching + shared KV ownership
v0.6  KV-pressure preemption + recomputation
v0.7  speculative decoding + transactional KV
v0.8  EngineCore / Executor / Worker boundary
v0.9  logical multi-rank + reference tensor parallel   ← current
v0.10 async engine + streaming/cancellation            ← next
```

Read `Agents.md` before changing the repository and `Plan.md` for the canonical roadmap.

## v0.9 mental model

```text
CONTROL / POLICY

EngineCore
  Scheduler / request state / sampling / preemption
      │
      ▼
ExecutionRequest

================ execution boundary ================

DistributedExecutor
      │
      ▼
WorkerGroup(world_size=2)
   ┌───────────────┴───────────────┐
   ▼                               ▼
rank 0                            rank 1
replicated trunk + KV             replicated trunk + KV
LM-head rows [0:k)                LM-head rows [k:V)
   │                               │
   ▼                               ▼
local logits shard                local logits shard
   └───────────────┬───────────────┘
                   ▼
         ReferenceCollective
              all_gather
                   │
                   ▼
             full logits

================ execution boundary ================

                   │
                   ▼
              EngineCore
          logical sample / commit
```

The first tensor-parallel integration intentionally shards only the vocabulary projection. Transformer trunk weights and KV cache are replicated per rank. That makes the first distributed version explicit about what is and is not distributed.

## Why this order?

v0.8 first separated policy from execution. v0.9 can therefore change execution topology without adding a second Scheduler or sending mutable `Request` objects to workers.

Core invariants:

```text
one EngineCore policy
many execution ranks

rank-local tensor shard
      +
reference collective
      =
single-worker mathematical result
```

If one rank fails during an ordinary forward, WorkerGroup rolls every rank back to the pre-forward cache checkpoint and surfaces the exception. EngineCore does not advance logical request state.

## Educational tensor parallel primitives

```text
RankConfig
ReferenceCollective
  ├─ all_gather
  └─ all_reduce_sum
ReferenceTensorParallel
  ├─ column_parallel_linear
  └─ row_parallel_linear
VocabParallelLMHead
VocabParallelTinyRunner
WorkerGroup
DistributedExecutor
```

The runtime remains single-process: logical ranks are simulated with ordinary PyTorch tensors. No NCCL, multiprocessing, or multi-node transport is hidden behind the reference implementation.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
./scripts/check.sh
```

The release rule is deliberately strict: the exact commit promoted to `main` must have a green GitHub Actions run executing `./scripts/check.sh`.

Run the v0.9 trace:

```bash
python examples/10_tensor_parallel.py
```

It prints each rank's vocabulary shard and compares the final v0.9 token history with the v0.8 single-worker reference.

## Learn through branch diffs

```bash
git diff v0.6..v0.7 -- mini_vllm tests examples version.md
git diff v0.7..v0.8 -- mini_vllm tests examples version.md
git diff v0.8..v0.9 -- mini_vllm tests examples version.md
```

For v0.9 focus on:

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

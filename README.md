# mini-vLLM

A deliberately small, runnable, **branch-versioned** LLM-serving runtime for learning the systems ideas behind vLLM.

## Current integrated version

`main` remains the latest fully released line. `v0.8` is the current release candidate until its full repository CI gate passes.

```text
main
  ├── v0.1  static batching + contiguous KV
  ├── v0.2  continuous batching
  ├── v0.3  paged KV + block tables
  ├── v0.4  token budget + chunked prefill
  ├── v0.5  prefix caching + shared KV ownership
  ├── v0.6  KV-pressure preemption + recomputation
  ├── v0.7  speculative decoding + transactional KV
  └── v0.8  EngineCore / Executor / Worker boundary   release candidate
```

After the v0.8 CI gate passes, it is synchronized to `main`; the next planned branch is `v0.9` — distributed / tensor-parallel execution concepts.

## Version ladder

| Branch | Primary mechanism | Main question |
|---|---|---|
| `v0.1` | static batching + contiguous KV | what is the smallest serving loop? |
| `v0.2` | continuous batching | how are free sequence slots refilled? |
| `v0.3` | paged KV | how can growing KV avoid contiguous slabs? |
| `v0.4` | token budget + chunked prefill | how much token work runs per step? |
| `v0.5` | prefix cache + refcounts | how can identical prefixes share immutable KV? |
| `v0.6` | recompute preemption | what happens when active KV no longer fits? |
| `v0.7` | speculative transaction | how can temporary draft state be verified and rolled back safely? |
| `v0.8` | EngineCore / Executor / Worker | where should centralized policy end and execution begin? |

Read `Agents.md` before changing the repository and `Plan.md` for the canonical roadmap.

## v0.8 mental model

```text
CONTROL / POLICY

Request registry
     │
     ▼
 EngineCore
 ├─ Scheduler
 ├─ admission / token budget
 ├─ preemption victim policy
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

The central invariant is:

```text
Execution payloads carry execution facts, not mutable Request policy objects.
```

`ExecutionItem` contains only `request_id`, token IDs, and a start position. Worker therefore does not get scheduler queues, request status, priority, or victim-selection state through the execution payload.

## Why LocalExecutor first?

The first executor stays synchronous and in-process. That proves ownership and message contracts before adding multiprocessing, RPC, multi-GPU topology, or distributed failure modes.

Historical builders still work: if an older version supplies `cache + model_runner`, `EngineCore` automatically wraps them in `Worker + LocalExecutor`. Frozen historical branches themselves remain unchanged.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
./scripts/check.sh
```

The v0.8 boundary-focused local harness passed 5/5 tests during implementation. The repository CI runs the canonical `./scripts/check.sh` on the complete checkout and is the release gate for v0.8.

Run the v0.8 architecture trace:

```bash
python examples/09_executor_boundary.py
```

## Learn through branch diffs

```bash
git diff v0.1..v0.2 -- mini_vllm/scheduler tests examples version.md
git diff v0.2..v0.3 -- mini_vllm/cache tests examples version.md
git diff v0.3..v0.4 -- mini_vllm tests examples version.md
git diff v0.4..v0.5 -- mini_vllm/cache mini_vllm/core tests examples version.md
git diff v0.5..v0.6 -- mini_vllm tests examples version.md
git diff v0.6..v0.7 -- mini_vllm tests examples version.md
git diff v0.7..v0.8 -- mini_vllm tests examples version.md
```

For v0.8, focus on:

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

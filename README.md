# mini-vLLM

A deliberately small, runnable, **branch-versioned** LLM-serving runtime for learning the systems ideas behind vLLM.

## Current line

`main` is always the newest validated release. `v0.10` is the current implementation line; an exact version commit is synchronized to `main` only after repository CI passes on that same commit.

```text
v0.1  static batching + contiguous KV
v0.2  continuous batching
v0.3  paged KV + block tables
v0.4  token budget + chunked prefill
v0.5  prefix caching + shared KV ownership
v0.6  KV-pressure preemption + recomputation
v0.7  speculative decoding + transactional KV
v0.8  EngineCore / Executor / Worker boundary
v0.9  logical multi-rank + reference tensor parallel
v0.10 async engine + streaming/cancellation/backpressure  ← current
v0.11 graph/compile-aware execution                       ← next
```

Read `Agents.md` before changing the repository and `Plan.md` for the canonical roadmap.

## v0.10 mental model

```text
caller
  │ submit(Request)
  ▼
AsyncEngine
  │ caller handles + bounded event queues
  │ submit/cancel intents
  ▼
EngineCore                      ← only serving-policy authority
  │ Scheduler / token budget / sampling / preemption
  ▼
DistributedExecutor
  ▼
WorkerGroup(world_size=N)
  ▼
rank-local model/KV + reference collective
  ▼
EngineStepOutput
  │ committed_token_ids only
  ▼
AsyncEngine
  ├─ TokenEvent
  ├─ FinishedEvent
  ├─ CancelledEvent
  └─ ErrorEvent
```

`AsyncEngine` is a caller-facing lifecycle/delivery facade. It does **not** own a second Scheduler, request-priority policy, tensor-parallel topology, or speculative verification policy.

## Core async invariants

```text
commit before stream
exactly one terminal event
cancel means real Scheduler/KV release
bounded queues never silently drop/reorder committed tokens
one cooperative driver; no idle busy-spin
```

Cancellation is applied at an EngineCore step boundary rather than interrupting a model forward in the middle. A token already committed by the current atomic step may be streamed before cancellation becomes terminal.

The first backpressure policy is intentionally simple: a full per-request event queue blocks the cooperative async driver from taking another EngineCore step until delivery can continue. This is easy to inspect and preserves committed-event order; it is not presented as a production throughput policy.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
./scripts/check.sh
```

The release rule is strict: the exact commit promoted to `main` must have a green GitHub Actions run executing `./scripts/check.sh`.

Run the v0.10 trace:

```bash
python examples/11_async_streaming.py
```

It demonstrates normal streaming, committed token order, cancellation at a step boundary, and driver shutdown.

## Learn through branch diffs

```bash
git diff v0.7..v0.8 -- mini_vllm tests examples version.md
git diff v0.8..v0.9 -- mini_vllm tests examples version.md
git diff v0.9..v0.10 -- mini_vllm tests examples version.md
```

For v0.10 focus on:

```text
mini_vllm/async_runtime/events.py
mini_vllm/async_runtime/handle.py
mini_vllm/async_runtime/engine.py
mini_vllm/core/engine.py
mini_vllm/scheduler/base.py
mini_vllm/versions/v0_10/engine.py
tests/test_async_runtime.py
tests/test_v010_version.py
examples/11_async_streaming.py
```

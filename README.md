# mini-vLLM

A small, branch-versioned LLM-serving runtime for learning the systems ideas behind vLLM.

## Current line

`main` is always the newest exact-CI-validated release. v1.0 integrates the mechanisms introduced from v0.1 through v0.13 into one coherent public construction API and one read-only observability path.

```text
v0.1  static batching + contiguous KV
v0.2  continuous batching
v0.3  paged KV + block tables
v0.4  token budget + chunked prefill
v0.5  prefix caching + shared KV
v0.6  preemption + recomputation
v0.7  speculative decoding + transactional KV
v0.8  EngineCore / Executor / Worker
v0.9  logical multi-rank + tensor parallel
v0.10 async streaming + cancellation/backpressure
v0.11 compile-aware rank-local tensor execution
v0.12 reference + vectorized PagedAttention
v0.13 prefix-aware scheduling + cache-aware admission
v1.0  integrated educational runtime                  ← current
```

Historical version builders remain available so each mechanism can still be studied in isolation.

## v1.0 mental model

```text
caller
  ↓
AsyncEngine
  ↓
EngineCore
  ↓
PrefixAwareBudgetedScheduler
  ├─ token budget / chunked prefill
  ├─ prefix-aware admission + fairness
  └─ KV-pressure preemption
  ↓
DistributedExecutor / WorkerGroup
  ↓
rank-local Worker
  ↓
TinyDecoder
  ├─ transactional prefix-shared paged KV
  ├─ vectorized PagedAttention
  └─ compile-aware vocabulary projection
  ↓
Speculative verification / Sampler
  ↓
committed async stream events
```

v1.0 does not introduce another major serving algorithm. It makes the existing stack readable as one runtime.

## Stable construction API

The default builder now creates v1.0:

```python
from mini_vllm import Request, RuntimeConfig, create_engine

engine = create_engine(runtime_config=RuntimeConfig())
handle = engine.submit(Request("R", [1, 2, 3], max_new_tokens=4))
```

`RuntimeConfig` groups the important educational knobs—engine limits, model shape, KV layout, compile mode, attention mode, logical world size, async queue size, and prefix-fairness threshold—without exposing block tables, refcounts, or other physical ownership structures.

Historical comparisons remain explicit:

```python
v03 = create_engine("v0.3")
v13 = create_engine("v0.13")
v10 = create_engine("v1.0")
```

## One public observability path

After or during a run, v1.0 users inspect the runtime with:

```python
snapshot = engine.snapshot()
```

The snapshot exposes, without private-field access:

- request lifecycle, computed frontier, output history and preemption count;
- waiting/running sets, prefix-aware admission costs, fairness and estimate-vs-attach reconciliation;
- cache statistics and prefix reuse counters;
- scheduled prefill/decode work and committed tokens for recent steps;
- speculative accept/reject/rollback observations;
- distributed world size and rank-local vocabulary ranges;
- vectorized/reference attention call counts and fallbacks;
- compile attempts, cache hits/misses and eager fallbacks;
- async pending/cancellation/driver state.

Observation is read-only. Calling `snapshot()` cannot schedule work, attach KV, change LRU state, or advance a request.

## Core correctness contract

The condensed v1.0 invariants live in `docs/INVARIANTS.md`. The most important separation remains:

```text
logical history
    !=
computed-token frontier
    !=
physical KV ownership
```

Likewise:

```text
PrefixReuseHint != reservation != ownership != computed tokens
```

Optimization paths remain subordinate to readable oracles: vectorized PagedAttention must equal reference attention, and compile-aware projection must equal eager projection.

## Integrated walkthrough

```bash
pip install -e '.[dev]'
python examples/15_integrated_runtime.py
```

The example warms a reusable prompt, submits cold and warm requests, streams committed output, then explains admission/reconciliation, recent work, speculation, cache state, and rank-local attention/compile execution through the public snapshot.

## Verify

```bash
./scripts/check.sh
```

The release gate runs every historical regression plus examples 01–15. A v1.0 commit may advance `main` only when that exact commit has a green repository CI run.

Read `Agents.md`, `Plan.md`, `docs/ARCHITECTURE.md`, `docs/INVARIANTS.md`, and `version.md` before changing architecture.

# mini-vLLM v1.0 — integrated educational runtime (planning branch)

## Version identity

- Branch: `v1.0`
- Base: validated `v0.13` / `main`
- Previous: `v0.13`
- Theme: consolidate the complete educational serving runtime into one coherent public API, trace model, documentation set, and end-to-end validation story
- Status: planning; runtime remains exactly v0.13 until v1.0 acceptance criteria pass

## Why v1.0

v0.1–v0.13 intentionally introduced one serving-system idea at a time:

```text
v0.1  static batching
v0.2  continuous batching
v0.3  paged KV
v0.4  token budgets / chunked prefill
v0.5  prefix sharing
v0.6  preemption / recompute
v0.7  speculative transactions
v0.8  EngineCore / Executor / Worker
v0.9  logical ranks / tensor parallel
v0.10 async streaming / cancellation / backpressure
v0.11 compile-aware execution
v0.12 vectorized PagedAttention
v0.13 prefix-aware scheduling
```

v1.0 is not another mechanism version. Its job is to make these mechanisms read as one runtime rather than thirteen adjacent lessons.

## Primary mental model

```text
caller
  ↓
AsyncEngine
  ↓
EngineCore
  ↓
Scheduler
  ├─ token budget
  ├─ fairness
  ├─ prefix-aware admission
  └─ preemption policy
  ↓
Executor / WorkerGroup
  ↓
rank-local Worker
  ↓
TinyDecoder
  ├─ paged / prefix-shared KV
  ├─ vectorized PagedAttention
  └─ compile-aware projection
  ↓
Sampler / speculative verification
  ↓
committed stream events
```

Every arrow must have a documented ownership contract and an inspectable trace.

## v1.0 goals

### 1. Stable public construction API

Provide one clear default `create_engine()` path and keep historical version builders available for learning comparisons.

Public configuration should make the important serving knobs discoverable without exposing internal storage structures.

### 2. Unified observability

Consolidate the currently separate scheduling, preemption, speculative, compile, attention, cache, distributed, and async observations into a coherent inspectable runtime snapshot or trace vocabulary.

The goal is not production telemetry. The goal is that a learner can answer:

```text
Why was this request admitted?
How many tokens actually ran?
Did prefix reuse occur?
Was anything preempted?
Did speculation accept or reject?
Which execution path ran?
What KV/cache state changed?
Which tokens became externally visible?
```

without reaching into private implementation fields.

### 3. Integrated end-to-end scenario

Add one executable scenario that deliberately exercises multiple existing mechanisms together, for example:

```text
warm + cold requests
      ↓
prefix-aware admission
      ↓
chunked prefill / continuous batching
      ↓
finite KV pressure + possible preemption
      ↓
speculative verification
      ↓
distributed execution
      ↓
vectorized attention + compile-aware projection
      ↓
async token stream
```

The scenario must explain causality, not merely print final tokens.

### 4. Documentation consolidation

README, `Agents.md`, `Plan.md`, architecture docs, roadmap, exercises, and version docs should agree on:

- ownership boundaries;
- the full request lifecycle;
- logical-token versus physical-KV state;
- policy versus mechanism;
- failure / rollback semantics;
- optimization correctness oracles;
- historical learning order.

### 5. Stable invariant suite

Promote the cross-version invariants into a concise v1.0 contract. At minimum:

- computed tokens imply valid KV;
- sampled tokens are not computed until executed;
- physical ownership precedes KV access;
- block-table translation is authoritative;
- prefix hints are advisory, not ownership;
- preemption rolls back to valid addressable KV;
- rejected speculative state disappears;
- distributed failures cannot partially commit replicated state;
- async clients see committed tokens only and exactly one terminal outcome;
- optimized attention equals the readable oracle;
- compiled projection equals eager projection;
- warm-prefix policy cannot starve cold work.

## Preferred implementation shape

v1.0 should primarily compose and expose existing mechanisms rather than invent replacements.

Possible additions include:

- a small public runtime/trace snapshot dataclass;
- a stable top-level configuration facade;
- integrated example 15;
- documentation/navigation cleanup;
- narrowly scoped API naming cleanup where it improves conceptual consistency;
- tests proving public observability does not mutate runtime state.

Any API cleanup must preserve historical version branches and avoid hiding the mechanisms they were built to teach.

## Acceptance criteria

- [ ] default `create_engine()` represents the integrated v1.0 runtime;
- [ ] historical v0.1–v0.13 builders remain usable;
- [ ] one public observability path explains scheduling, cache reuse, execution, speculation, and committed output without private-field access;
- [ ] one integrated executable example demonstrates several mechanisms interacting coherently;
- [ ] ownership and rollback semantics are documented consistently across top-level docs;
- [ ] no new major Scheduler/KV/distributed/kernel mechanism is introduced merely to justify the version number;
- [ ] deterministic output remains equivalent to validated v0.13 under equivalent configuration;
- [ ] all v0.1–v0.13 regression tests pass;
- [ ] examples 01–14 continue to pass, plus any v1.0 integrated example;
- [ ] exact release candidate `./scripts/check.sh` passes in GitHub Actions before `main` advances.

## Non-goals

- production vLLM API compatibility;
- real multi-node/NCCL serving;
- production CUDA/Triton kernels;
- model loading/tokenizer ecosystems;
- OpenAI-compatible HTTP server;
- persistence or remote KV transfer;
- learned scheduling;
- hiding educational mechanisms behind a large framework layer.

## Learning goal

By v1.0 a reader should be able to follow one request from submission to terminal stream event and explain every major serving-system decision that occurs along the way.

## After v1.0

Further versions should be justified by a genuinely new learning target rather than incremental feature accumulation. Candidates may include real GPU kernels, real multiprocessing/distributed transport, or production-style model loading—but only after v1.0 freezes the current educational architecture cleanly.

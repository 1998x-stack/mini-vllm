# Agents.md — mini-vLLM Development Contract

> Mandatory for every future iteration: read this file before changing code, tests, docs, branches, version scope, or architecture.

## 1. Mission

mini-vLLM is an educational LLM-serving runtime. Optimize for architectural clarity, correctness before performance, explicit invariants, readable reference mechanisms, inspectable tests, and conceptual mapping to production serving systems.

Do not optimize early for production API coverage, pretrained-model breadth, maximum throughput, distributed complexity, or custom kernels.

```text
one version
   ↓
one major serving mental model
   ↓
readable mechanism
   ↓
explicit invariant tests
   ↓
end-to-end trace/example
```

## 2. Required reading order

On `main` read: `Agents.md` → `Plan.md` → `version.md` → `docs/ROADMAP.md` → relevant architecture/tests.

On a historical `v0.x`/`v1.x` branch read: `Agents.md` → `version.md` → relevant source/tests/examples. Historical branches are learning snapshots and must not absorb later mechanisms.

## 3. Branch model

`main` must always be the newest integrated validated implementation.

Canonical sequence:

- `v0.1` — static batching + contiguous KV
- `v0.2` — continuous batching
- `v0.3` — paged KV + block table + recycling
- `v0.4` — token-budget scheduling + chunked prefill
- `v0.5` — prefix caching + shared KV ownership
- `v0.6` — KV-pressure preemption + recomputation
- `v0.7` — speculative decoding + transactional KV
- `v0.8` — EngineCore / Executor / Worker boundary
- `v0.9` — distributed / tensor-parallel concepts (planned next)

Every version branch must contain `Agents.md`, `version.md`, tests for promised mechanisms, and an executable educational example where applicable.

After a version freezes, only documentation corrections, scope-preserving tests, and narrowly scoped bug fixes are allowed.

## 4. Version lifecycle

Before coding: identify the primary mental model, explain dependency on the previous version, define ownership boundaries/non-goals/acceptance criteria, and align `version.md` with `Plan.md`.

Implementation priority:

```text
correctness
  ↓
readability
  ↓
invariant visibility
  ↓
testability
  ↓
architecture
  ↓
performance
```

After validation: update docs, synchronize the validated state to `main`, confirm `main` is not behind, freeze the version branch, then create the next planning branch from latest `main`.

## 5. Architecture rules

### 5.1 Policy vs mechanism

- Scheduler owns admission, ordering, token work, and victim policy.
- EngineCore owns request registry, logical history, sampling/commit, finish decisions, and policy coordination.
- Executor owns execution transport/orchestration.
- Worker owns execution mechanism, not scheduling policy.
- KV backend owns storage/allocation/address translation, not priority decisions.
- ModelRunner executes model work, not queue policy.
- DraftProvider proposes candidates only; it does not commit logical history or own KV policy.

### 5.2 v0.8 execution boundary

Preferred direction:

```text
Request / SamplingParams
          │
          ▼
      EngineCore
   request state + policy
          │
          ▼
   ExecutionRequest
          │
          ▼
       Executor
          │
          ▼
        Worker
  ModelRunner + KV backend
          │
          ▼
   ExecutionResult
          │
          ▼
      EngineCore
```

`ExecutionItem` must not carry a mutable `Request` object. Only execution facts such as `request_id`, token IDs, and positions cross into Worker.

Future multiprocessing/distribution should replace transport/topology, not redesign scheduler semantics.

## 6. Core invariants

### Computed-token invariant

`num_computed_tokens` counts tokens whose forward pass completed and whose K/V is valid. A sampled token is not computed until a later forward consumes it.

### Cache ownership invariant

A request may only read/write positions with reserved capacity.

### Paged-address invariant

```text
logical_block = position // block_size
block_offset  = position % block_size
physical      = block_table[request_id][logical_block]
```

### Release invariant

Finished requests release active cache ownership exactly once.

### Semantic-equivalence invariant

Changing storage layout, speculative acceleration, or executor transport must not change deterministic target-model semantics.

### Transaction invariant

Rejected speculative tokens must not remain in committed history, visible computed state, block-table ownership beyond the retained frontier, or future attention context.

### Execution-boundary invariant

Worker must not decide admission, scheduler priority, preemption victim, or logical request completion. EngineCore must not depend on hidden Worker internals for normal policy execution.

### Failure-commit invariant

EngineCore advances logical request state only after a successful execution result. Executor failure must surface explicitly.

## 7. Scope discipline

Do not add prefix caching before block ownership is explicit; speculative decoding before rollback exists; multiprocessing before the EngineCore/Executor boundary exists; distributed/tensor parallelism before LocalExecutor semantics are stable; or Triton/CUDA merely to make reference code look fast.

Future work belongs in `Plan.md` until its dependencies are ready.

## 8. Testing requirements

Every new subsystem should include happy path, invariant/property tests, error path, reference/numerical validation where relevant, integration tests, and an educational trace/example.

Canonical commands:

```bash
python -m pytest -q
./scripts/check.sh
```

Never claim tests passed unless they were actually executed on the exact reported snapshot.

Representative invariants:

- static cohorts do not backfill; continuous batching does;
- paged/contiguous KV produce equivalent logits;
- token budget is never exceeded;
- shared prefix blocks remain immutable/refcounted;
- preemption rolls back invalid computed state;
- replay does not re-sample known history;
- rejected speculative tokens disappear;
- LocalExecutor transport preserves v0.7 semantics;
- Worker has no waiting/running queue policy.

## 9. Documentation requirements

For a version change review at minimum: `version.md`, `Agents.md`, `Plan.md`, `docs/ROADMAP.md`, `docs/ARCHITECTURE.md`, and `README.md` when user-facing behavior changes.

## 10. Commit rules

Prefer intent-revealing commits such as:

```text
feat(scheduler): add continuous slot refill
feat(cache): add paged KV block pool
feat(speculative): add checkpoint and rollback
feat(executor): add local worker boundary
test(executor): verify policy-free execution payload
```

Do not mix unrelated mechanisms into opaque commits.

## 11. Performance rule

```text
readable reference
       ↓
correctness/invariant tests
       ↓
measure
       ↓
optimized educational path
       ↓
compare against reference
```

Examples: Python paged gather before Triton, LocalExecutor before multiprocessing/RPC, scripted draft provider before a neural draft model, reference collectives before multi-GPU kernels.

## 12. Upstream reference policy

Use official vLLM docs/source and the PagedAttention paper to verify concepts when current production behavior matters. Implement original educational equivalents; do not copy production source.

## 13. Definition of Done

A version freezes only when applicable items are true:

- primary mental model is explicit;
- promised mechanism is implemented;
- unit/invariant/error tests pass;
- reference equivalence exists where relevant;
- integration example works;
- `version.md` accurately describes the delta;
- architecture/roadmap docs are synchronized;
- package imports cleanly;
- generated artifacts are not committed;
- `main` contains the newest validated state;
- next-version dependency rationale is documented.

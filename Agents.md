# Agents.md — mini-vLLM Development Contract

> Mandatory for every future iteration: read this file before changing code, tests, docs, branches, version scope, or architecture.

## 1. Mission

mini-vLLM is an educational LLM-serving runtime. Optimize for architectural clarity, correctness before performance, explicit invariants, readable reference mechanisms, inspectable tests, and conceptual mapping to production serving systems.

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

Do not optimize early for production API breadth, pretrained-model coverage, distributed complexity, or custom kernels.

## 2. Reading order

On `main`: `Agents.md` → `Plan.md` → `version.md` → roadmap/architecture → relevant tests.

On a `v0.x`/`v1.x` branch: `Agents.md` → `version.md` → relevant source/tests/examples. Historical branches are learning snapshots and must not absorb later mechanisms.

## 3. Branch model

`main` must always be the newest integrated **validated** runtime. A candidate is synchronized to `main` only after its exact commit passes required tests/CI.

Canonical sequence:

- `v0.1` — static batching + contiguous KV
- `v0.2` — continuous batching
- `v0.3` — paged KV + block table + recycling
- `v0.4` — token-budget scheduling + chunked prefill
- `v0.5` — prefix caching + shared KV ownership
- `v0.6` — KV-pressure preemption + recomputation
- `v0.7` — speculative decoding + transactional KV
- `v0.8` — EngineCore / Executor / Worker boundary
- `v0.9` — WorkerGroup + reference tensor parallel
- `v0.10` — async engine + streaming/cancellation/backpressure
- `v0.11` — graph/compile-aware execution (next)

Every version branch requires `Agents.md`, `version.md`, mechanism tests, error-path coverage, and an executable educational example where applicable.

## 4. Ownership rules

- AsyncEngine owns caller-facing handles, submission/cancellation intent, bounded event delivery, and cooperative driving only.
- Scheduler owns admission, ordering, token work, and victim policy.
- EngineCore owns request registry, logical history, sampling/commit, finish/cancel decisions, and policy coordination.
- Executor owns execution transport/orchestration.
- WorkerGroup owns execution topology, not scheduling policy.
- Worker owns model/cache execution mechanism, not request policy.
- KV backend owns physical storage/allocation/address translation.
- ModelRunner executes model work.
- DraftProvider proposes candidates only.
- Collective implements tensor communication semantics only; it must not know Scheduler/request priority.

Preferred dependency direction:

```text
Caller
  ↓
AsyncEngine lifecycle/delivery
  ↓
EngineCore policy
  ↓
ExecutionRequest
  ↓
Executor
  ↓
WorkerGroup
  ↓
Workers / rank-local model + KV
  ↓
Collective
  ↓
ExecutionResult
  ↓
EngineCore commit
  ↓
AsyncEngine committed events
```

Mutable `Request` objects must not cross the execution boundary. AsyncEngine must not implement a shadow serving Scheduler.

## 5. Core invariants

### Computed-token invariant

`num_computed_tokens` counts tokens whose forward completed and whose K/V is valid. A sampled token is not computed until a later forward consumes it.

### Cache ownership invariant

A request may only read/write positions with reserved capacity.

### Paged-address invariant

```text
logical_block = position // block_size
block_offset  = position % block_size
physical      = block_table[request_id][logical_block]
```

### Release invariant

Finished/cancelled/aborted requests release active cache ownership exactly once.

### Semantic-equivalence invariant

Storage layout, speculation, executor transport, tensor-parallel topology, and async delivery must preserve deterministic target-model semantics.

### Transaction invariant

Rejected speculative tokens must not remain in committed history, visible KV state, or async token streams.

### Execution-boundary invariant

Workers/WorkerGroup must not decide admission, Scheduler priority, preemption victim, logical completion, or caller cancellation policy.

### Failure-commit invariant

EngineCore advances logical state only after successful execution. Executor failures surface explicitly.

### Rank-consistency invariant

All ranks in a WorkerGroup must agree on world size, request identity, token positions, operation ordering, and replicated cache lifecycle metadata.

### Distributed-failure invariant

A partial rank failure must not leave surviving ranks physically advanced while EngineCore remains logically uncommitted.

### Commit-before-stream invariant

AsyncEngine may emit token events only for tokens present in committed logical output history.

### Terminal invariant

An async request emits exactly one terminal event and no events afterward.

### Cancellation-release invariant

Cancellation at a cooperative EngineCore boundary removes Scheduler participation and active KV ownership before `CancelledEvent` becomes terminal.

### Backpressure invariant

A bounded async stream may delay the cooperative driver but may not silently drop or reorder committed events.

## 6. Scope discipline

Build readable references before optimized transport. Do not add threads/network servers while the cooperative async ownership model is still the teaching target. Do not add graph capture before eager execution is the correctness oracle; do not add custom attention kernels before the readable paged path is validated.

v0.9/v0.10 deliberately replicate Transformer trunk + KV across logical ranks and shard only the LM head. Do not describe this as full production tensor parallelism.

## 7. Testing requirements

Every new subsystem should include happy path, invariant/property tests, error path, numerical/reference validation where relevant, integration tests, and an educational trace.

Canonical commands:

```bash
python -m pytest -q
./scripts/check.sh
```

Never claim tests passed unless executed successfully on the exact reported snapshot. Repository CI is the release gate for active version candidates.

Representative invariants:

- paged/contiguous KV produce equivalent logits;
- token budgets are respected;
- prefix blocks remain immutable/refcounted;
- preemption rolls back invalid computed state;
- rejected speculative tokens disappear;
- rank-local tensor shards reconstruct the single-worker reference;
- rank failure cannot partially commit distributed physical state;
- async-collected deterministic output equals the synchronous reference;
- queued cancellation allocates no KV;
- active cancellation releases Scheduler/KV ownership;
- bounded queues preserve committed event order;
- each stream has exactly one terminal event.

## 8. Version lifecycle

Before coding: define the mental model, dependency on previous version, ownership boundaries, non-goals, and acceptance criteria.

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

After validation: update docs, synchronize the exact validated candidate to `main`, freeze the branch, then create the next planning branch from latest `main`.

## 9. Documentation and commits

For version changes review at minimum: `version.md`, `Agents.md`, `Plan.md`, `docs/ROADMAP.md`, `docs/ARCHITECTURE.md`, and README.

## 10. Performance rule

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

## 11. Definition of Done

A version freezes only when applicable items are true:

- primary mental model is explicit;
- promised mechanism exists;
- unit/invariant/error tests pass;
- reference equivalence exists;
- integration example works;
- version/docs agree;
- package imports cleanly;
- generated artifacts are not committed;
- exact release-candidate CI succeeds;
- `main` contains the newest validated state;
- next-version rationale is documented.

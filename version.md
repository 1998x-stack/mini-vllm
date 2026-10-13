# mini-vLLM v1.0 — integrated educational runtime

## Version identity

- Branch: `v1.0`
- Previous: validated `v0.13`
- Theme: consolidate v0.1–v0.13 into one coherent public construction API, observability model, documentation contract and end-to-end scenario
- Package version: `1.0.0`
- Release rule: the exact implementation commit may enter `main` only after repository CI passes

## Why v1.0

The historical versions deliberately isolated one idea at a time:

```text
v0.1  static batching
v0.2  continuous batching
v0.3  paged KV
v0.4  token budget / chunked prefill
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

v1.0 does not add another major serving algorithm. It makes these mechanisms readable as one runtime.

## Integrated mental model

```text
caller
  ↓
IntegratedAsyncEngine
  ↓
IntegratedEngine / EngineCore
  ↓
PrefixAwareBudgetedScheduler
  ├─ continuous batching
  ├─ token budget + chunked prefill
  ├─ prefix-aware admission + fairness
  └─ KV-pressure preemption
  ↓
DistributedExecutor / WorkerGroup
  ↓
rank-local Worker
  ↓
CompileAwareVocabParallelTinyRunner
  ↓
TinyDecoderLM
  ├─ transactional prefix-shared paged KV
  ├─ vectorized PagedAttention
  └─ compile-aware vocabulary projection
  ↓
Speculative verification / Sampler
  ↓
committed EngineStepOutput
  ↓
async TokenEvent / terminal event
```

## Stable public construction

v1.0 makes the default builder the integrated runtime:

```python
engine = create_engine()
```

`RuntimeConfig` is the stable facade for the important educational knobs:

- `EngineConfig`;
- `TinyModelConfig`;
- `CacheConfig`;
- `CompileConfig`;
- `PagedAttentionConfig`;
- logical `world_size`;
- async `max_queue_size`;
- prefix-fairness bypass limit.

Physical block tables, refcounts, LRU internals and free lists are intentionally not construction knobs.

Historical builders remain available:

```python
create_engine("v0.1")
...
create_engine("v0.13")
```

## Unified observability

v1.0 adds a read-only public path:

```python
snapshot = engine.snapshot()
```

### `AsyncRuntimeSnapshot`

Reports:

- runtime version;
- whether the async driver is active;
- pending request IDs;
- cancellation intents;
- the core `RuntimeSnapshot`.

### `RuntimeSnapshot`

Reports:

- current step frontier;
- unfinished request IDs;
- per-request logical/computed lifecycle state;
- Scheduler state and recent admission/reconciliation observations;
- cache statistics;
- preemption count;
- distributed/rank execution observations;
- latest and recent `EngineStepOutput` objects.

### Scheduler observations

`SchedulingObservation` copies the v0.13 prefix-aware facts needed to explain an admission:

```text
raw prefill
estimated reusable prefix
expected new prefill work
bypass count
fairness protection
actual reuse after authoritative prepare
```

### Execution observations

Each `RankExecutionSnapshot` exposes:

- logical rank/world size;
- rank-local vocabulary range;
- attention backend name and reference/vectorized/fallback counts;
- compile enabled/backend state;
- compile attempts, cache hits/misses, fallbacks and eager calls.

The snapshot exposes facts and counters, not physical KV ownership handles.

## Read-only invariant

Calling `snapshot()` must not:

- schedule or admit work;
- attach a reusable prefix;
- allocate/free blocks;
- change refcounts;
- touch LRU recency;
- execute model work;
- append output tokens;
- advance `num_computed_tokens`;
- publish async events.

Repeated snapshots with no intervening runtime action compare equal in the v1.0 test suite.

## Public causality questions

A learner can answer from one snapshot path:

```text
Why was this request admitted?
How much prefix reuse was estimated?
How much was actually attached?
Which prefill/decode tokens executed?
Was any request preempted?
Did speculation accept or reject candidates?
Which attention path ran on each rank?
Did compile cache hit or miss?
What cache state remains resident?
Which output tokens were logically committed?
What lifecycle state is each request in?
```

## Integrated example

`examples/15_integrated_runtime.py` deliberately composes multiple existing mechanisms:

```text
seed reusable prefix
      ↓
submit cold then warm requests
      ↓
prefix-aware admission
      ↓
token-budgeted/chunked execution
      ↓
speculative verification
      ↓
logical multi-rank execution
      ↓
vectorized PagedAttention
      ↓
compile-aware rank-local projection
      ↓
async committed token stream
      ↓
public snapshot explanation
```

The example prints committed output and then explains admission/reconciliation, recent serving work, speculation, cache state, rank-local execution paths and final request lifecycle without private-field access.

## Stable invariant contract

`docs/INVARIANTS.md` promotes the cross-version rules into one v1.0 contract. The central distinctions are:

```text
logical token history
        !=
computed-token frontier
        !=
physical KV ownership
```

and:

```text
PrefixReuseHint != reservation != ownership != computed tokens
```

Other required invariants include transactional rollback, rank failure atomicity, exactly-one async terminal outcome, attention-reference equivalence, compile/eager equivalence and prefix-fairness starvation protection.

## Compatibility

v1.0 composes rather than replaces the mechanisms below:

- v0.4 actual token-budget accounting;
- v0.5 immutable shared prefix ownership;
- v0.6 recompute preemption;
- v0.7 speculative checkpoint/rollback;
- v0.8 execution ownership boundary;
- v0.9 logical tensor-parallel WorkerGroup;
- v0.10 async streaming/cancellation/backpressure;
- v0.11 compile-aware projection;
- v0.12 reference/vectorized PagedAttention;
- v0.13 side-effect-free reuse hints and fairness-aware admission.

Deterministic v1.0 output must remain equivalent to validated v0.13 under equivalent configuration.

## Tests

`tests/test_v1_version.py` proves:

- default `create_engine()` selects v1.0;
- deterministic v1.0 output equals v0.13;
- `RuntimeConfig` constructs the integrated runtime;
- incompatible mixed configuration is rejected;
- snapshots are repeatable and side-effect free;
- committed output, speculation, cache state, scheduler facts and rank execution paths are publicly observable;
- real warm-prefix savings are visible in admission events;
- historical builders remain available.

The full release gate includes all historical regressions and examples 01–15. The integration candidate reached 85 passing pytest tests before documentation freeze; the final exact release commit must independently pass the same repository CI gate before `main` advances.

## Documentation set

- `README.md`: public entry point and quick mental model;
- `Agents.md`: development/ownership contract;
- `Plan.md`: architecture-driven version plan;
- `docs/ARCHITECTURE.md`: complete v1.0 request and ownership flow;
- `docs/INVARIANTS.md`: stable cross-version correctness rules;
- `docs/ROADMAP.md`: frozen learning ladder;
- `docs/EXERCISES.md`: integrated reasoning exercises;
- `version.md`: this version-specific contract.

## Non-goals

- production vLLM API compatibility;
- OpenAI-compatible HTTP serving;
- tokenizer/model-loading ecosystem;
- real multi-node/NCCL serving;
- production CUDA/Triton kernels;
- persistence or remote KV transfer;
- learned/SLA scheduling;
- hiding educational mechanisms behind a large framework layer.

## Learning goal

By v1.0 a reader should be able to follow one request from submission to terminal stream event and explain every major serving-system decision and state transition using the public runtime interfaces.

## After v1.0

A future version must be justified by a genuinely new learning target. Candidates include real GPU kernels, actual multiprocessing/distributed transport or production-style model loading. They are intentionally outside the v1.0 contract.

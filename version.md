# mini-vLLM v0.7 — speculative decoding (planning branch)

## 1. Version identity

- **Branch:** `v0.7`
- **Base:** latest validated `v0.6` / `main`
- **Previous version:** `v0.6`
- **Theme:** speculative decoding as a transactional KV-state problem
- **Status:** planning / implementation target; runtime still behaves as v0.6 until acceptance criteria pass

## 2. Why v0.7 follows v0.6

v0.6 makes KV invalidation and replay explicit. That is the prerequisite for speculative decoding, where the runtime intentionally creates temporary candidate state that may later be rejected.

The new question is:

> How can several draft tokens be proposed and verified while guaranteeing that rejected tokens leave no visible model or KV state behind?

## 3. Primary mental model

```text
stable request state
      │
      ▼
KV checkpoint
      │
      ▼
draft proposes [d0 d1 d2]
      │
      ▼
target verifies candidates
      │
      ├── accept [d0 d1] ──► commit accepted prefix
      │
      └── reject d2       ──► rollback rejected suffix
      │
      ▼
continue from committed state only
```

Speculative decoding is therefore not merely “generate multiple tokens at once.” It is a transaction over logical token history and physical KV state.

## 4. Required contracts

Introduce small explicit abstractions rather than mixing speculation into the normal sampler path:

```text
DraftProvider
DraftBatch
VerificationResult
CacheCheckpoint
commit()
rollback()
```

Start with a deterministic scripted draft provider. Do **not** introduce a second neural draft model until transaction semantics are proven.

## 5. Cache checkpoint semantics

A checkpoint must capture enough state to restore the request to the last stable frontier.

At minimum reason about:

- logical token count before speculation;
- `num_computed_tokens` before speculation;
- block-table length / tail allocation frontier;
- private mutable tail state;
- shared immutable prefix blocks that must not be modified;
- any new blocks allocated for speculative candidates.

Rollback must make rejected speculative positions unreachable and recyclable.

## 6. Logical-history invariant

Draft tokens are **proposals**, not committed output history.

Before verification:

```text
committed output_token_ids = stable history
speculative tokens         = temporary proposal buffer
```

Only accepted tokens may become durable request output.

This prevents normal preemption/replay code from treating rejected speculative tokens as real generation history.

## 7. Verification policy

The first educational implementation should use a deterministic acceptance rule that is easy to test.

A suggested progression:

1. scripted draft tokens;
2. target model computes verification logits over the candidate sequence;
3. accept the longest prefix whose candidate matches the target greedy token;
4. on first mismatch, reject that token and all later draft tokens;
5. sample/commit the target token at the mismatch position;
6. rollback any physical KV state beyond the committed frontier.

The goal is semantic correctness, not maximum speculative throughput.

## 8. Critical invariants

### Transaction invariant

After verification, request state must be indistinguishable from an execution that only ever computed the committed token sequence.

### Rollback invariant

Rejected speculative tokens must not remain:

- in `output_token_ids`;
- in the visible computed-token frontier;
- in block-table ownership beyond the committed frontier;
- in future attention context.

### Shared-prefix invariant

Checkpoint/rollback may modify only request-private speculative state. Shared immutable prefix blocks remain untouched.

### Deterministic-equivalence invariant

With greedy target sampling, speculative and ordinary v0.6 decoding must produce the same final output tokens.

## 9. Scheduler interaction

Do not redesign the scheduler in the first v0.7 increment.

Prefer:

```text
PreemptiveBudgetedScheduler
        │
        ▼
normal scheduled request
        │
        ▼
Speculative execution policy
```

Later experiments may account for draft/verification token cost in the token budget, but transaction correctness comes first.

## 10. Acceptance criteria

v0.7 cannot be frozen until:

- [ ] a deterministic `DraftProvider` can propose multiple tokens;
- [ ] a request can create a cache checkpoint before speculative work;
- [ ] target verification accepts a correct candidate prefix;
- [ ] a mismatching candidate causes suffix rollback;
- [ ] rejected tokens disappear from logical history and KV visibility;
- [ ] accepted speculative tokens become normal committed output history;
- [ ] speculative greedy output matches ordinary v0.6 greedy output;
- [ ] rollback correctly frees or truncates request-private blocks;
- [ ] shared prefix blocks survive rollback unchanged;
- [ ] preemption of a non-speculating request still passes v0.6 regressions;
- [ ] tests cover accept-all, reject-first, partial-accept, and rollback-across-block-boundary cases;
- [ ] an executable trace shows checkpoint -> draft -> verify -> commit/rollback;
- [ ] validated implementation is synchronized to `main`.

## 11. Preferred implementation order

```text
1. DraftProvider + scripted candidates
2. VerificationResult data model
3. cache checkpoint metadata
4. rollback/truncate API on paged/prefix cache
5. target verification reference path
6. logical commit of accepted tokens
7. integration with Engine trace
8. equivalence tests
```

## 12. Explicit non-goals

- learned neural draft model in the first increment;
- Medusa / EAGLE / multi-head draft methods;
- production acceptance kernels;
- distributed speculative execution;
- async streaming changes;
- Executor / Worker process split — `v0.8`;
- CUDA/Triton optimization.

## 13. Next version

`v0.8` introduces the **EngineCore / Executor / Worker** boundary. Once scheduling, cache ownership, preemption, and speculative transaction semantics are explicit in-process, execution can be moved behind an executor interface without changing those semantics.

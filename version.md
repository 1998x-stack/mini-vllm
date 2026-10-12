# mini-vLLM v0.7 — speculative decoding + transactional KV

## 1. Version identity

- **Branch:** `v0.7`
- **Previous version:** `v0.6`
- **Theme:** target-verified speculative decoding as a transaction over logical history and physical KV state
- **Status:** frozen historical learning snapshot after validation

v0.7 keeps the v0.6 scheduler, preemption, prefix cache, and request lifecycle intact. It adds a new execution policy only after a request has reached a stable sampling frontier:

```text
stable target state
      │
      ▼
cache checkpoint
      │
      ▼
draft proposes candidates
      │
      ▼
target materializes temporary KV
      │
      ▼
verify longest greedy prefix
      │
      ├── accepted KV stays visible
      └── rejected suffix is rolled back
      │
      ▼
commit logical output
```

---

## 2. Why v0.7 follows v0.6

v0.6 introduced explicit invalidation and replay: logical token history can survive while physical KV is discarded and reconstructed.

Speculative decoding needs the same distinction in a more local transactional form. Draft tokens are intentionally computed before the runtime knows whether they belong to durable output history.

The new systems question is:

> How can temporary candidate K/V be created, verified, partially accepted, and safely discarded without leaking rejected tokens into later attention or request history?

---

## 3. Core state model

v0.7 distinguishes three frontiers:

```text
committed logical history
        │
        ├── prompt_token_ids
        └── output_token_ids

stable computed frontier
        │
        └── num_computed_tokens

speculative temporary frontier
        │
        └── K/V written after CacheCheckpoint
```

Draft proposals never enter `output_token_ids` before verification.

---

## 4. New abstractions

### DraftProvider

```text
Request
   │
   ▼
DraftProvider.propose(max_tokens=N)
   │
   ▼
DraftBatch(token_ids=(...))
```

`ScriptedDraftProvider` is the correctness tool for accept/reject tests. `IncrementDraftProvider` is a deliberately weak deterministic default; it is not presented as a production draft model.

### CacheCheckpoint

```python
CacheCheckpoint(
    request_id,
    num_tokens,
    block_table,
)
```

It protects the stable block-table prefix before temporary writes.

### VerificationResult

Every speculation transaction records:

- draft candidates;
- accepted prefix;
- rejected suffix;
- target fallback token on mismatch;
- target bonus token when all candidates match;
- checkpoint position;
- rollback frontier.

The result is exposed through `EngineStepOutput.speculation`.

---

## 5. Verification semantics

The first v0.7 implementation deliberately uses greedy target verification.

Suppose stable logits predict `t0` and the draft proposes:

```text
[d0 d1 d2]
```

Target verification is:

```text
d0 ?= greedy(stable_logits)
d1 ?= greedy(logits after d0)
d2 ?= greedy(logits after d1)
```

### Partial accept

```text
draft:  [A B C]
target: [A X ...]

accepted = [A]
rejected = [B C]
fallback = X
```

The target model may already have written K/V for all draft tokens. Rollback keeps K/V through accepted `A`, discards the rejected suffix, and leaves target fallback `X` as a logical-but-not-yet-computed output token.

### Accept all

```text
draft:  [A B C]
target: [A B C]
```

All candidate K/V remains valid. The target logits after `C` produce one bonus token `D`:

```text
committed output = [A B C D]
computed KV      = [A B C]
D remains pending for the next ordinary decode step
```

This preserves the sampled-but-not-yet-computed invariant used by earlier versions.

---

## 6. Transactional cache

v0.7 introduces:

```text
PrefixCachingPagedKVCache
        │
        ▼
TransactionalPrefixCachingPagedKVCache
```

The subclass owns checkpoint/rollback semantics without changing v0.1-v0.6 cache behavior.

### Checkpoint invariant

A checkpoint is created only at the current valid KV frontier:

```text
checkpoint.num_tokens == current used-token frontier
```

### Rollback invariant

Rollback may discard only request-private state created at or after the checkpoint.

It may not:

- cross behind the checkpoint;
- mutate the stable block-table prefix;
- free cached immutable prefix blocks;
- free a shared mutable block;
- retain block-table entries that are no longer addressable.

### Cross-block rollback

If speculative tokens allocate new physical blocks, partial verification can shrink the block table and recycle rejected blocks.

---

## 7. Shared-prefix safety

Prefix-cache blocks are immutable and may be shared by multiple requests. Speculation begins after the stable request prefix, so temporary K/V belongs to the request-private tail.

```text
shared prefix block
     │
     ├── request A
     └── request B

B speculates and rolls back
     │
     └── shared prefix bytes remain unchanged
```

---

## 8. Interaction with v0.6 preemption

The scheduler remains `PreemptiveBudgetedScheduler`.

Normal scheduled work still participates in v0.6 pressure handling. Speculation itself is optional acceleration:

```text
speculative extra KV fits
        │
        ├── yes -> speculate
        └── no  -> ordinary target sampling
```

v0.7 does **not** preempt another active request merely to make speculative acceleration fit.

---

## 9. Token-budget simplification

`max_num_batched_tokens` still accounts for ordinary scheduler work only. Temporary target verification tokens created inside a speculation transaction are not yet charged back into the scheduler token budget. This is intentional for the first reference implementation.

---

## 10. Important source files

```text
mini_vllm/cache/base.py
        ↓
mini_vllm/cache/transactional.py
        ↓
mini_vllm/speculative/types.py
mini_vllm/speculative/draft.py
mini_vllm/speculative/engine.py
        ↓
mini_vllm/model/tiny_transformer.py
mini_vllm/model/runner.py
        ↓
mini_vllm/versions/v0_7/engine.py
        ↓
tests/test_speculative.py
examples/08_speculative_decoding.py
```

Recommended comparison:

```bash
git diff v0.6..v0.7 -- mini_vllm tests examples version.md
```

---

## 11. Tests and validation

v0.7 validates:

- accept-all draft verification;
- target bonus token after full acceptance;
- reject-first fallback behavior;
- partial acceptance;
- rejected suffix removal from logical history;
- rollback across a physical block boundary;
- accepted speculative K/V remains valid;
- shared immutable prefix blocks remain byte-for-byte unchanged;
- greedy speculative generation equals ordinary v0.6 greedy generation;
- ordinary v0.1-v0.6 regression behavior remains valid;
- executable transaction trace works end to end.

The frozen local snapshot passes **27 tests** plus all eight executable examples.

---

## 12. Learning goals

After v0.7, the learner should be able to explain:

- why draft tokens are proposals rather than output history;
- why target verification needs per-token logits, not only final chunk logits;
- how speculative K/V can exist before logical commit;
- why rollback must truncate physical ownership as well as logical token lists;
- why accepted draft tokens may keep their already-computed KV;
- why one target fallback/bonus token remains logically committed but physically uncomputed;
- why shared prefix blocks must lie outside the mutable speculative tail;
- why speculation is a transaction layered on top of, not inside, the scheduler.

---

## 13. Explicit non-goals

- learned neural draft model;
- stochastic speculative acceptance;
- EOS-aware multi-token speculative policy;
- charging draft/verification work to scheduler token budget;
- Medusa / EAGLE / multi-head speculation;
- production verification kernels;
- distributed speculation;
- async streaming changes;
- Executor / Worker process separation;
- CUDA/Triton optimization.

---

## 14. Next version

`v0.8` introduces the **EngineCore / Executor / Worker boundary**.

After v0.7, request lifecycle, scheduling, cache ownership, memory pressure, and speculative transactions are explicit within one process. The next architectural question is:

> Which logic belongs to the central engine core, and which work can move behind an executor/worker interface without changing semantics?

```text
EngineCore
   │ SchedulerOutput / execution request
   ▼
Executor
   ▼
Worker / ModelRunner / KV backend
   │ execution result
   ▼
EngineCore
```

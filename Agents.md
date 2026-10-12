# Agents.md — mini-vLLM Development Contract

> **Mandatory instruction for every future iteration:** before changing code, tests, docs, branches, version scope, or architecture, read this file first.
>
> This file is an engineering contract, not optional guidance.

---

# 1. Repository mission

mini-vLLM is an educational LLM-serving runtime for understanding the mechanisms behind production systems such as vLLM.

The project optimizes for:

1. architectural clarity;
2. correctness before performance;
3. explicit serving invariants;
4. readable reference implementations;
5. tests that demonstrate systems mechanisms;
6. conceptual mapping to production inference runtimes.

It does **not** optimize for:

- full vLLM API compatibility;
- pretrained-model coverage;
- maximum tokens/second before the architecture is understandable;
- reproducing CUDA/Triton kernels before a readable reference path exists;
- copying upstream vLLM implementation details.

The guiding principle is:

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

---

# 2. Mandatory reading order

First determine the current Git branch.

## If current branch is `main`

Read:

1. `Agents.md`
2. `Plan.md`
3. `version.md`
4. `docs/ROADMAP.md`
5. relevant architecture docs
6. relevant tests

`Plan.md` is the canonical forward architecture and version-sequencing document.

## If current branch is `v0.x` / `v1.x`

Read:

1. `Agents.md`
2. `version.md`
3. relevant source files
4. relevant tests/examples

A historical branch is a learning snapshot. Never assume it should contain mechanisms from a later branch.

---

# 3. Branch model

## 3.1 `main` invariant

**`main` must always be the newest integrated, validated implementation.**

Allowed:

```text
active version branch ── validated increment ──► main
```

Forbidden:

```text
active version branch: newer validated runtime
main: stale older runtime
```

## 3.2 Version branches

Version branches are educational snapshots, not long-lived feature dumps.

Canonical sequence:

- `v0.1` — minimal request lifecycle + static batching + contiguous KV cache
- `v0.2` — continuous batching
- `v0.3` — paged KV cache + block table + block recycling
- `v0.4` — token-budget scheduling + chunked prefill
- `v0.5` — prefix caching + shared KV block ownership
- `v0.6` — KV-pressure preemption (planned next step)

Every version branch must contain:

- `Agents.md`
- `version.md`
- tests for its promised mechanisms
- at least one executable educational example where applicable

## 3.3 Historical branch rule

After a version is frozen, allowed changes are limited to:

- documentation corrections;
- tests that preserve that version's scope;
- narrowly scoped bug fixes required by that version's promised behavior.

Do not backport future-version systems merely for convenience.

---

# 4. Version lifecycle

## Phase A — plan

Before coding:

1. identify one primary mental model;
2. explain why it follows the previous version;
3. define ownership/dependency boundaries;
4. list non-goals;
5. define acceptance criteria;
6. create/update `version.md`;
7. verify alignment with `Plan.md`.

## Phase B — implement the reference mechanism

Priority:

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

If an optimized implementation hides the idea, retain a readable reference path as the correctness oracle.

## Phase C — validate invariants

Tests must demonstrate mechanisms, not only final token IDs.

Examples:

- Scheduler: static cohorts do not backfill; continuous batching does.
- KV cache: writes/read-prefix preserve token order.
- Paged KV: logical positions map through a block table to physical slots.
- Block allocator: blocks are recycled after request completion.
- Model/cache equivalence: contiguous and paged storage produce equal logits.
- Engine: cache is freed when a request finishes.

## Phase D — integrate

After validation:

1. update docs;
2. update `version.md`;
3. synchronize validated state into `main`;
4. confirm `main` is not behind the active branch.

## Phase E — freeze

Once Definition of Done is satisfied, freeze the version branch as a historical learning snapshot and start the next version from latest `main`.

---

# 5. New version procedure

When starting `vX.Y`:

1. verify `main` is the newest validated state;
2. create `vX.Y` from current `main`;
3. create/update `version.md` immediately;
4. ensure `Agents.md` exists;
5. state the delta from the previous version;
6. implement only that version's scope;
7. synchronize validated increments back to `main`;
8. freeze after Definition of Done.

Never start a future version from an older historical branch if `main` is newer.

---

# 6. Required `version.md` contract

Every version branch must document:

1. branch/version identity and status;
2. previous version;
3. version theme;
4. **what changed relative to the previous version, and why**;
5. mechanisms added;
6. mechanisms intentionally unchanged;
7. architecture/mental-model diagram;
8. important source files;
9. tests and examples;
10. learning goals;
11. non-goals;
12. remaining work if active;
13. next-version direction.

---

# 7. Architecture rules

## 7.1 Preserve dependency direction

Preferred conceptual flow:

```text
Request / SamplingParams
          │
          ▼
       Scheduler      ← policy: who runs now?
          │
          ▼
     ScheduleDecision
          │
          ▼
    KV Cache Backend  ← mechanism: where does KV live?
          │
          ▼
      ModelRunner     ← executes model work
          │
          ▼
        Sampler       ← selects next token
          │
          ▼
     Request update
```

Future execution split:

```text
EngineCore
   │ SchedulerOutput
   ▼
Executor
   ▼
Worker / ModelRunner
```

Avoid circular imports between scheduler, cache, model, and engine layers.

## 7.2 Separate policy from mechanism

- Scheduler decides admission and token work; it must not own KV tensors.
- KV backend allocates/translates storage; it must not decide request priority.
- ModelRunner executes model work; it must not implement queue policy.
- Sampler chooses tokens; it must not mutate scheduler internals.
- Engine coordinates these components; it should not duplicate their algorithms.

## 7.3 Prefer conceptual vLLM compatibility

Map to production concepts without copying source:

- request lifecycle;
- scheduler/token budget;
- prefill versus decode;
- KV cache manager;
- physical block pool and block table;
- prefix-cache ownership/refcounts;
- executor/worker boundary;
- optimized attention kernels as interchangeable execution mechanisms.

---

# 8. Core runtime invariants

These invariants must remain explicit and tested.

## 8.1 Computed-token invariant

`num_computed_tokens` counts tokens that have actually passed through the model and whose K/V is materialized.

A sampled output token is **not** computed until the next decode forward pass consumes it.

## 8.2 Cache ownership invariant

A request may only read/write KV positions for which capacity has been reserved.

## 8.3 Paged address invariant

For block size `B` and token position `p`:

```text
logical_block = p // B
block_offset  = p % B
physical      = block_table[request_id][logical_block]
```

## 8.4 Release invariant

Finished requests must release cache ownership exactly once.

## 8.5 Semantic-equivalence invariant

Changing KV storage layout must not change model semantics. Reference contiguous and paged paths should produce numerically equivalent logits within tolerance.

---

# 9. Scope discipline

Before adding a feature ask:

1. Does it explain this version's primary mental model?
2. Is it required by an acceptance criterion?
3. Is it a dependency of the current mechanism?
4. Would implementing it now hide a more fundamental abstraction?

Examples:

- Do not add prefix caching before block ownership/recycling is explicit.
- Do not add speculative decoding before commit/rollback semantics exist.
- Do not add multiprocessing before the EngineCore/Executor boundary exists.
- Do not add Triton merely to make the reference paged cache look faster.

Future work belongs in `Plan.md` until its dependencies are ready.

---

# 10. Testing requirements

Every new subsystem should include, where applicable:

- happy-path behavior;
- invariant/property tests;
- error-path tests;
- numerical/reference validation;
- integration test;
- educational trace/example.

Canonical command:

```bash
python -m pytest -q
```

Repository check:

```bash
./scripts/check.sh
```

Never claim tests passed unless they were executed successfully on the exact code being reported.

---

# 11. Documentation requirements

Review documentation whenever a change modifies:

- public API;
- architecture boundary;
- version scope;
- branch policy;
- request/cache invariants;
- learning sequence;
- example usage.

For version changes, review at minimum:

- `version.md`
- `Agents.md`
- `Plan.md` on `main` if sequencing changes
- `docs/ROADMAP.md`
- `docs/ARCHITECTURE.md`
- `README.md` when user-facing behavior changes

---

# 12. Commit rules

Prefer commits that express systems intent:

```text
feat(scheduler): add continuous slot refill
feat(cache): add paged KV block pool
feat(cache): expose logical-to-physical slot mapping
test(cache): verify block recycling
test(model): compare contiguous and paged logits
docs: define v0.4 chunked-prefill scope
```

Do not mix unrelated mechanisms into a single opaque change.

---

# 13. Performance rule

Use this progression:

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

Examples:

- Python paged gather before Triton PagedAttention;
- synchronous Engine before async/multiprocess execution;
- simple FCFS/token budget before sophisticated priority/preemption.

---

# 14. Upstream reference policy

When a design depends on current production behavior, prefer official vLLM source/docs and the PagedAttention paper as conceptual references.

Use upstream sources to verify concepts, but implement original educational equivalents. Do not copy production source code.

---

# 15. Definition of Done

A version can be frozen only when all applicable items are true:

- [ ] primary mental model is stated clearly;
- [ ] promised mechanism is implemented;
- [ ] unit tests pass;
- [ ] invariant tests pass;
- [ ] error-path behavior is covered;
- [ ] numerical/reference validation exists where relevant;
- [ ] integration example works;
- [ ] `version.md` accurately describes the delta;
- [ ] `Agents.md` exists;
- [ ] architecture/roadmap docs are synchronized;
- [ ] package imports cleanly;
- [ ] generated artifacts are not committed;
- [ ] `main` contains the newest validated state;
- [ ] next version's dependency rationale is documented.

# mini-vLLM

A deliberately small, runnable, **branch-versioned** LLM-serving runtime for learning the systems ideas behind vLLM.

## Current integrated version: v0.7

```text
main  ─────────────────────────────────► newest validated state (v0.7)
  ├── v0.1  static batching + contiguous KV
  ├── v0.2  continuous batching
  ├── v0.3  paged KV + block tables
  ├── v0.4  token budget + chunked prefill
  ├── v0.5  prefix caching + shared KV ownership
  ├── v0.6  KV-pressure preemption + recomputation
  └── v0.7  speculative decoding + transactional KV
```

Next: `v0.8` — EngineCore / Executor / Worker separation.

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

Read `Agents.md` before changing the repository and `Plan.md` for the canonical roadmap.

## v0.7 mental model

```text
stable target state
      │
      ▼
CacheCheckpoint
      │
      ▼
DraftProvider
      │ candidates
      ▼
target verification
      │
      ├── accept prefix ──► keep verified K/V
      │
      └── reject suffix ──► rollback private K/V
      │
      ▼
commit logical outputs
```

The central invariant is:

```text
Draft tokens are temporary proposals until target verification commits them.
```

### Partial accept

```text
draft:  [A B C]
target: [A X ...]

accepted: [A]
rejected: [B C]
fallback: X
```

Physical K/V for rejected positions is truncated. Logical output becomes `[A, X]`, while only `A` is already represented in KV.

### Accept all

```text
draft: [A B C]
all match target
        │
        ▼
keep K/V(A,B,C)
        +
sample target bonus D
```

`D` remains the normal sampled-but-not-yet-computed token for the next decode step.

## Architecture

```text
PreemptiveBudgetedScheduler        (unchanged v0.6 policy)
              │
              ▼
      SpeculativeEngine
              │
      stable sampling frontier
              ▼
         DraftProvider
              │
              ▼
TinyTransformerRunner.execute_speculative()
              │
              ▼
TransactionalPrefixCachingPagedKVCache
       │ checkpoint / rollback
       ▼
VerificationResult
```

Policy boundaries remain explicit:

```text
Scheduler       admission / token budget / preemption victim
DraftProvider   proposal only
Target model    verification truth
KV cache        transactional physical state
Engine          logical commit coordination
```

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
./scripts/check.sh
```

Current frozen v0.7 snapshot: **27 tests + 8 examples**.

## Learn through branch diffs

```bash
git diff v0.1..v0.2 -- mini_vllm/scheduler tests examples version.md
git diff v0.2..v0.3 -- mini_vllm/cache tests examples version.md
git diff v0.3..v0.4 -- mini_vllm tests examples version.md
git diff v0.4..v0.5 -- mini_vllm/cache mini_vllm/core tests examples version.md
git diff v0.5..v0.6 -- mini_vllm tests examples version.md
git diff v0.6..v0.7 -- mini_vllm tests examples version.md
```

For v0.7, focus on:

```text
mini_vllm/cache/transactional.py
mini_vllm/speculative/draft.py
mini_vllm/speculative/types.py
mini_vllm/speculative/engine.py
mini_vllm/model/tiny_transformer.py
mini_vllm/model/runner.py
tests/test_speculative.py
examples/08_speculative_decoding.py
```

Run:

```bash
python examples/08_speculative_decoding.py
```

Observe both `partial accept + rollback` and `accept all + bonus token`, then compare the final output with ordinary v0.6 greedy decoding.

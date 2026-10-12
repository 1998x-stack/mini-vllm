# mini-vLLM

A deliberately small, runnable, **branch-versioned** LLM-serving runtime for learning the systems ideas behind vLLM.

The project favors explicit mechanisms over production throughput. Each historical branch introduces one serving mental model and keeps earlier branches small enough to diff and understand.

## Current integrated version: v0.4

```text
main  ─────────────────────────────────► newest validated state (v0.4)
  │
  ├── v0.1  static batching + contiguous KV
  ├── v0.2  continuous batching
  ├── v0.3  paged KV + block table + recycling
  └── v0.4  token budget + chunked prefill
```

The next planned version is `v0.5`: prefix caching and shared block ownership.

## Version ladder

| Branch | Scheduler | KV cache | Primary lesson |
|---|---|---|---|
| `v0.1` | Static cohort | Contiguous request slab | Minimal serving loop and head-of-line waste |
| `v0.2` | Continuous batching | Contiguous request slab | Refill sequence slots every step |
| `v0.3` | Continuous batching | Paged block pool | Logical/physical KV addressing and recycling |
| `v0.4` | Token-budget continuous | Paged block pool | Chunked prefill and mixed prefill/decode |
| `v0.5` | planned | shared paged blocks | Prefix reuse, refcounts and immutable shared KV |

Read `Agents.md` before changing the repository and `Plan.md` for the canonical roadmap.

## v0.4 mental model

`max_num_seqs` answers: **How many requests may be active?**

`max_num_batched_tokens` answers: **How many tokens of model work may this engine step execute?**

The v0.4 reference scheduler is decode-first:

```text
step budget = 8

A decode          1
B decode          1
C prefill chunk   6
                 ---
total             8
```

A long prompt is no longer atomic:

```text
prompt: 10 tokens
budget: 4

step 0: prefill [0:4]   no sample
step 1: prefill [4:8]   no sample
step 2: prefill [8:10]  sample first output
step 3: decode 1 token  sample next output
```

The critical rule is:

> **computing a partial prefill chunk grows KV state but does not produce a user-visible output token.**

## Core runtime

```text
Request
   │
   ▼
BudgetedContinuousScheduler
   │  ScheduleDecision[WorkItem]
   ▼
PagedKVCache.ensure_capacity(end_pos)
   │
   ▼
ModelRunner.execute(chunk)
   │
   ├── partial prefill ──► update computed-token state only
   │
   └── sequence caught up ──► Sampler ──► append output token
```

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
./scripts/check.sh
```

## Learn through branch diffs

```bash
git diff v0.1..v0.2 -- mini_vllm/scheduler tests examples version.md
git diff v0.2..v0.3 -- mini_vllm/cache mini_vllm/versions tests examples version.md
git diff v0.3..v0.4 -- mini_vllm tests examples version.md
```

Focus on these files for v0.4:

```text
mini_vllm/config.py
mini_vllm/core/work.py
mini_vllm/scheduler/base.py
mini_vllm/scheduler/budgeted.py
mini_vllm/core/engine.py
examples/05_chunked_prefill.py
```

Run the timeline:

```bash
python examples/05_chunked_prefill.py
```

## Next: v0.5

`v0.5` will introduce a readable reference design for prefix caching:

```text
block hash
   ↓
prefix index
   ↓
shared full blocks
   ↓
reference counts
   ↓
immutable shared prefix / mutable request tail
```

# mini-vLLM

A deliberately small, runnable, **branch-versioned** LLM-serving runtime for learning the systems ideas behind vLLM.

## Current integrated version: v0.5

```text
main  ─────────────────────────────────► newest validated state (v0.5)
  │
  ├── v0.1  static batching + contiguous KV
  ├── v0.2  continuous batching
  ├── v0.3  paged KV + block tables
  ├── v0.4  token budget + chunked prefill
  └── v0.5  prefix caching + shared KV ownership
```

Next: `v0.6` — KV-pressure preemption and recomputation policy.

## Version ladder

| Branch | Primary mechanism | Main question |
|---|---|---|
| `v0.1` | static batching + contiguous KV | what is the smallest serving loop? |
| `v0.2` | continuous batching | how are free sequence slots refilled? |
| `v0.3` | paged KV | how can growing KV avoid contiguous slabs? |
| `v0.4` | token budget + chunked prefill | how much token work runs per step? |
| `v0.5` | prefix cache + refcounts | how can identical prefixes share immutable KV? |

Read `Agents.md` before changing the repository and `Plan.md` for the canonical roadmap.

## v0.5 mental model

```text
Request A: [system prompt blocks] [private tail A]
                 │
Request B: [system prompt blocks] [private tail B]
                 │
                 ▼
          same physical KV blocks
          refcounted + immutable
```

Only full prompt blocks are cached. Prefix hashes are chained with prior context, so an identical token block under a different prefix does not alias incorrectly.

A warm request attaches cached blocks before scheduling:

```text
cold request
prefill [0:5]
   ↓
publish full blocks [0:2], [2:4]
   ↓
request finishes; active refcount -> 0
cached blocks stay resident

warm identical request
attach [0:2], [2:4]
num_computed_tokens = 4
   ↓
compute only [4:5]
   ↓
sample first output
```

### Important: KV hit != logits hit

KV cache stores K/V, not next-token logits. v0.5 intentionally leaves enough prompt work to compute the final logits even when the entire prompt exists in the prefix index.

## Architecture

```text
Request
   │
   ├── cache.prepare_request() ──► attach immutable shared prefix
   │
   ▼
BudgetedContinuousScheduler
   │
   ▼
Paged KV allocation / private tail
   │
   ▼
ModelRunner
   │
   ├── cache.commit_work() ──► publish completed full prompt blocks
   │
   ▼
Sampler
```

Stable boundaries remain:

```text
Scheduler       decides who/how much runs
KVCacheBackend  owns storage, sharing and recycling
ModelRunner     executes model work
Sampler         selects next token
Engine          coordinates lifecycle hooks
```

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
./scripts/check.sh
```

Current frozen v0.5 snapshot: **18 tests + 6 examples**.

## Learn through branch diffs

```bash
git diff v0.1..v0.2 -- mini_vllm/scheduler tests examples version.md
git diff v0.2..v0.3 -- mini_vllm/cache tests examples version.md
git diff v0.3..v0.4 -- mini_vllm tests examples version.md
git diff v0.4..v0.5 -- mini_vllm/cache mini_vllm/core tests examples version.md
```

For v0.5, focus on:

```text
mini_vllm/cache/base.py
mini_vllm/cache/prefix.py
mini_vllm/core/engine.py
mini_vllm/versions/v0_5/engine.py
tests/test_prefix_cache.py
examples/06_prefix_caching.py
```

Run:

```bash
python examples/06_prefix_caching.py
```

Observe the cold request publish blocks and the warm request begin from a non-zero `start_pos` while producing the same output tokens.

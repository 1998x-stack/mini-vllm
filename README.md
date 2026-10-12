# mini-vLLM

A deliberately small, runnable, **branch-versioned** LLM-serving runtime for learning the systems ideas behind vLLM.

## Current integrated version: v0.6

```text
main  ─────────────────────────────────► newest validated state (v0.6)
  │
  ├── v0.1  static batching + contiguous KV
  ├── v0.2  continuous batching
  ├── v0.3  paged KV + block tables
  ├── v0.4  token budget + chunked prefill
  ├── v0.5  prefix caching + shared KV ownership
  └── v0.6  KV-pressure preemption + recomputation
```

Next: `v0.7` — speculative decoding with cache checkpoint / commit / rollback semantics.

## Version ladder

| Branch | Primary mechanism | Main question |
|---|---|---|
| `v0.1` | static batching + contiguous KV | what is the smallest serving loop? |
| `v0.2` | continuous batching | how are free sequence slots refilled? |
| `v0.3` | paged KV | how can growing KV avoid contiguous slabs? |
| `v0.4` | token budget + chunked prefill | how much token work runs per step? |
| `v0.5` | prefix cache + refcounts | how can identical prefixes share immutable KV? |
| `v0.6` | recompute preemption | what happens when active KV no longer fits? |

Read `Agents.md` before changing the repository and `Plan.md` for the canonical roadmap.

## v0.6 mental model

```text
scheduled work
     │
     ▼
reserve KV capacity
     │
     ├── success ─────────────► execute
     │
     └── failure
           │
           ▼
      Scheduler chooses victim
           │
           ▼
      release victim active KV
           │
           ▼
      PREEMPTED
  logical output history kept
  computed frontier reset
           │
           ▼
     later re-admission
           │
           ├── reuse valid cached prefix
           └── replay remaining history
           │
           ▼
        continue decode
```

The central invariant is:

```text
num_computed_tokens claims only KV state that still exists and is addressable.
```

### Replay does not resample history

If a request already sampled output token `x` before preemption, `x` remains in `output_token_ids`.
After KV is discarded, the runtime recomputes prompt + historical output tokens. `WorkItem.completes_current_sequence` prevents a new token from being sampled until replay reaches the end of the existing logical token history.

### Anti-thrashing rule

The reference scheduler chooses the most recently admitted eligible request as the victim, but a preempted request cannot immediately re-enter. It waits until at least one surviving request finishes. This deliberately simple rule prevents endless A/B ping-pong in a tiny KV pool and gives a bounded-workload fairness guarantee.

## Architecture

```text
PreemptiveBudgetedScheduler
        │ schedule
        ▼
      Engine
        │ reserve KV
        ▼
PrefixCachingPagedKVCache
        │
        ├── success ─────────► ModelRunner
        │
        └── failure
              │
              ▼
       select_preemption_victim()
              │
              ▼
       cache.free(victim)
       victim.mark_preempted()
       scheduler.on_request_preempted()
```

Policy and mechanism remain separate:

```text
Scheduler       chooses victim / admission / token budget
KVCacheBackend  owns blocks, refcounts, prefix residency
Engine          coordinates state transitions
ModelRunner     executes token work
Sampler         creates new logical output tokens
```

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
./scripts/check.sh
```

Current frozen v0.6 snapshot: **22 tests + 7 examples**.

## Learn through branch diffs

```bash
git diff v0.1..v0.2 -- mini_vllm/scheduler tests examples version.md
git diff v0.2..v0.3 -- mini_vllm/cache tests examples version.md
git diff v0.3..v0.4 -- mini_vllm tests examples version.md
git diff v0.4..v0.5 -- mini_vllm/cache mini_vllm/core tests examples version.md
git diff v0.5..v0.6 -- mini_vllm tests examples version.md
```

For v0.6, focus on:

```text
mini_vllm/core/request.py
mini_vllm/core/engine.py
mini_vllm/scheduler/base.py
mini_vllm/scheduler/preemptive.py
mini_vllm/versions/v0_6/engine.py
tests/test_preemption.py
examples/07_preemption.py
```

Run:

```bash
python examples/07_preemption.py
```

Observe a request keep its logical output history while its active KV state is discarded and later reconstructed.

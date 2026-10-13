# mini-vLLM

A small, branch-versioned LLM-serving runtime for learning the systems ideas behind vLLM.

## Current line

`main` is always the newest exact-CI-validated release. v0.13 is the prefix-aware scheduling implementation line; only an exact v0.13 commit with green repository CI may advance `main`.

```text
v0.1  static batching + contiguous KV
v0.2  continuous batching
v0.3  paged KV + block tables
v0.4  token budget + chunked prefill
v0.5  prefix caching + shared KV
v0.6  preemption + recomputation
v0.7  speculative decoding + transactional KV
v0.8  EngineCore / Executor / Worker
v0.9  logical multi-rank + tensor parallel
v0.10 async streaming + cancellation/backpressure
v0.11 compile-aware rank-local tensor execution
v0.12 reference + vectorized PagedAttention
v0.13 prefix-aware scheduling + cache-aware admission  ← current
v1.0  integrated educational runtime                  ← next
```

## v0.13 mental model

```text
cache estimate                Scheduler                    Engine/cache
     │                           │                            │
     └─ PrefixReuseHint ───────► AdmissionCost                │
                                 │                            │
                         expected new work                    │
                                 │                            │
                        warm preference + fairness            │
                                 │                            │
                                 └──── admit ────────────────► prepare_request()
                                                              │
                                                    actual reusable prefix
                                                              │
                                                        reconcile truth
```

The key rule is:

```text
hint != reservation != ownership != computed tokens
```

A waiting v0.13 request can be recognized as warm without owning KV. Only actual `prepare_request()` may advance its computed frontier.

## Verify

```bash
pip install -e '.[dev]'
./scripts/check.sh
python examples/14_prefix_aware_scheduling.py
```

The v0.13 example shows a warm request bypassing a colder, more expensive prefill, then exposes admission/reconciliation statistics. Fairness protection prevents repeated warm arrivals from starving cold work.

Read `Agents.md`, `Plan.md`, and `version.md` before changing architecture.

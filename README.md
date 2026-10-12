# mini-vLLM v0.4 — planning branch

This branch is the **next learning milestone**, not a fake completed release.

The runtime code intentionally remains equivalent to the validated v0.3 implementation until the v0.4 acceptance criteria are implemented and tested.

Before changing code, read in this order:

1. `Agents.md`
2. `version.md`
3. `Plan.md`
4. `mini_vllm/core/work.py`
5. `mini_vllm/scheduler/`
6. `mini_vllm/cache/paged.py`
7. relevant tests

## Primary question

> Once continuous batching and paged KV already exist, how should the scheduler bound **token work per engine step**, rather than only the number of active sequences?

Target mental model:

```text
max_num_batched_tokens = step token budget

step:
  decode A       = 1 token
  prefill B      = 64-token chunk
  prefill C      = 31-token chunk
                  --------
  scheduled       96 tokens <= budget
```

This enables **chunked prefill**: a long prompt no longer has to monopolize one whole engine step.

## Planned v0.4 delta

- `max_num_batched_tokens`
- budget-aware scheduler
- prompt chunks represented as `WorkItem`s
- partial prefill across multiple steps
- compatibility with incremental paged-KV growth
- decode/prefill interleaving experiments
- token-budget invariant tests
- TTFT / ITL educational step metrics

## Acceptance criteria

- every schedule satisfies `sum(work_item.num_tokens) <= max_num_batched_tokens`;
- long prompts can span multiple engine steps;
- decode requests can make progress while another request is still prefilling;
- chunked and unchunked reference execution agree numerically for the same token history;
- tests and examples make the scheduling timeline visible;
- validated implementation is synchronized back to `main` before v0.4 is frozen.

## Explicit non-goals

- prefix caching (`v0.5`)
- memory-pressure preemption (`v0.6`)
- speculative decoding (`v0.7`)
- EngineCore / Executor / Worker split (`v0.8`)
- Triton/CUDA PagedAttention (`v0.12`)

## Compare with v0.3

```bash
git diff v0.3..v0.4
```

At branch creation this diff should be mostly planning/documentation. As v0.4 is implemented, keep each validated increment narrow and synchronize it to `main` according to `Agents.md`.

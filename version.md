# mini-vLLM v0.1

## 1. Version identity

- **Branch:** `v0.1`
- **Previous version:** none; this is the baseline
- **Theme:** minimal autoregressive serving loop
- **Status:** frozen historical learning snapshot

## 2. Primary mental model

```text
Request
  ↓
StaticBatchScheduler
  ↓
Contiguous KV cache
  ↓
Tiny Decoder / ModelRunner
  ↓
Sampler
  ↓
request update
```

v0.1 answers:

> What is the smallest complete serving runtime that can prefill prompts, decode token-by-token, cache K/V, sample outputs, and finish requests?

## 3. What changed relative to the previous version?

There is no previous numbered version. v0.1 establishes the baseline contracts used by later versions:

- `Request`
- `WorkItem` / `ScheduleDecision`
- `Scheduler`
- `KVCacheBackend`
- `ModelRunner`
- `Sampler`
- `Engine`

## 4. Mechanisms added

- request lifecycle and sampling parameters
- distinction between full-prompt prefill and one-token decode
- static cohort batching
- contiguous per-request KV slabs
- tiny decoder-only Transformer
- model-runner abstraction
- central synchronous Engine loop
- cache release when a request finishes

## 5. Important invariant

`num_computed_tokens` means tokens that have actually passed through the model and have materialized KV state.

A sampled token is appended to output immediately, but it becomes a computed token only when a later decode step feeds it through the model.

## 6. Important source files

```text
mini_vllm/core/request.py
mini_vllm/core/work.py
mini_vllm/scheduler/static.py
mini_vllm/cache/contiguous.py
mini_vllm/model/tiny_transformer.py
mini_vllm/model/runner.py
mini_vllm/core/engine.py
```

## 7. Tests/examples

- contiguous KV round trip
- static cohort does not backfill while a cohort member is still running
- end-to-end basic generation

## 8. Learning goals

After v0.1 you should be able to explain:

- prefill versus decode;
- why KV is cached;
- why decode consumes the previously sampled token;
- why a static cohort can leave idle sequence slots;
- how Engine coordinates policy and mechanisms without owning all of them.

## 9. Non-goals

- no continuous batching
- no paged KV
- no token budget
- no prefix caching
- no preemption
- no speculative decoding
- no distributed executor
- no optimized attention kernel

## 10. Next version

`v0.2` changes the scheduler policy while deliberately leaving request/model/cache contracts stable.

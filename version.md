# mini-vLLM v0.2

## 1. Version identity

- **Branch:** `v0.2`
- **Previous version:** `v0.1`
- **Theme:** continuous batching
- **Status:** frozen historical learning snapshot

## 2. Primary mental model

```text
waiting queue
    │
    ▼
ContinuousBatchScheduler
    │ every engine step
    ├── remove finished
    └── fill newly free slots
    │
    ▼
active batch changes continuously
```

## 3. What changed relative to v0.1, and why?

v0.1 creates a static cohort. If B finishes while A is still running, B's slot remains idle until A also finishes.

v0.2 changes only the scheduling policy:

```text
v0.1                  v0.2
A B                   A B
A _                   A C
A _                   A _
C _
```

This isolates the core idea of **continuous batching**: sequence slots are resources that should be reconsidered each generation iteration.

## 4. Mechanisms added

- `ContinuousBatchScheduler`
- next-step slot refill
- explicit static-versus-continuous example
- invariant test proving C can enter when B finishes even while A keeps decoding

## 5. Mechanisms intentionally unchanged

- Request lifecycle
- Engine loop
- Tiny Decoder and ModelRunner
- sampler
- contiguous per-request KV layout
- full prompt prefill

This is intentional: the learner should see that continuous batching is a scheduler-policy change, not a model rewrite.

## 6. Important source delta

```text
mini_vllm/scheduler/continuous.py
mini_vllm/versions/v0_2/engine.py
examples/02_static_vs_continuous.py
```

## 7. Tests/examples

Run:

```bash
python examples/02_static_vs_continuous.py
python -m pytest -q
```

## 8. Learning goals

- distinguish static batching from continuous batching;
- understand waiting/running/admission state;
- understand why batching policy can change without changing model semantics;
- see why serving throughput/latency is a runtime concern, not merely a Transformer concern.

## 9. Non-goals

- paged KV
- token-budget scheduling
- chunked prefill
- prefix caching
- preemption

## 10. Next version

`v0.3` keeps continuous batching and replaces the contiguous KV allocator with fixed-size physical blocks plus a per-request block table.

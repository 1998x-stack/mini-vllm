# Exercises

## v0.1 — inspect the baseline

1. Set `max_model_len=256` and inspect how much contiguous capacity is reserved per active request.
2. Instrument `Request.num_computed_tokens` before and after prefill/decode.
3. Add `arrival_step` and `finished_step`; compute latency in engine steps.
4. Add a `RoundRobinScheduler` without modifying `Engine`.
5. Add stop-token sets instead of one global `eos_token_id`.

## Next exercise

After understanding the static cohort, checkout `v0.2` and compare scheduler behavior:

```bash
git diff v0.1..v0.2 -- mini_vllm/scheduler tests examples version.md
```

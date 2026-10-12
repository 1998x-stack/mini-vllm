# Exercises

## v0.3 recap

1. Change `block_size` and observe internal fragmentation.
2. Print block tables before and after request completion to see physical block recycling.
3. Replace paged KV with contiguous KV and verify deterministic logits remain equal.

## v0.4

1. Set `max_num_batched_tokens=1`. Explain why decode latency is protected but prompt TTFT can increase sharply.
2. Set the budget larger than every prompt. Show that v0.4 degenerates to effectively unchunked prefill.
3. Modify `BudgetedContinuousScheduler` from decode-first to pure FCFS and construct a trace where a long prompt delays decode.
4. Add round-robin prefill chunks instead of giving all remaining budget to the first FCFS prefill request. Which fairness property changes?
5. Record the engine step where each request emits its first token. Treat this as a TTFT proxy and compare policies.
6. Explain why sampling after every prefill chunk is semantically wrong even if the code still runs.

## Next: v0.5

Design block reference-count invariants for two requests sharing a full immutable prefix block while each owns a different mutable tail block.

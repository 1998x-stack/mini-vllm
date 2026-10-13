# mini-vLLM v1.0 exercises

Use historical branch diffs to learn one mechanism at a time, then use v1.0 to practice cross-layer reasoning.

## Follow one request end to end

1. Run `examples/15_integrated_runtime.py`. Draw the request path from `submit()` to the terminal event and label the owner of every decision.
2. From `engine.snapshot()`, identify the difference between `output_tokens` and `num_computed_tokens`. Give a moment when they can differ.
3. Locate the `WorkTrace` entries for prefill and decode. Why can one logical request require many EngineCore steps?
4. Explain why the async layer can delay delivery without changing which tokens were logically committed.

## Prefix-aware admission

5. Warm a prompt, then submit a cold request before a warm request. Use only `snapshot()` to explain which request was admitted first and why.
6. Compare `reusable_tokens` with `actual_reuse_tokens`. Why must the latter remain authoritative?
7. Construct an optimistic stale hint using the v0.13 tests. Show why a stale estimate cannot move `num_computed_tokens` forward.
8. Lower the fairness bypass limit. Explain when cache warmth stops being allowed to bypass older cold work.

## KV ownership and preemption

9. Revisit `examples/03_page_table_walkthrough.py`. For one logical position, calculate logical block, offset and physical block manually.
10. Use the v0.6 preemption example to distinguish logical token history from active KV ownership.
11. Explain why preemption followed by replay does not resample historical output tokens.
12. Describe what would break if Scheduler selected victims by reading physical block IDs directly.

## Speculative transactions

13. Run `examples/08_speculative_decoding.py` and locate matching `VerificationResult` entries in a v1.0 snapshot.
14. For a rejected speculative suffix, identify which state survives and which state must disappear.
15. Explain why speculative rollback may free a private tail but cannot mutate an immutable shared prefix.

## Distributed execution

16. Compare v1.0 with `world_size=1` and `world_size=2`. Why should committed deterministic output agree?
17. Inspect each `RankExecutionSnapshot.vocab_range`. Explain how rank-local logits become one global vocabulary distribution.
18. Force a rank-local failure using the existing distributed tests. Why must physical rollback happen before EngineCore logical commit?

## Attention and compilation

19. Switch PagedAttention from `vectorized` to `reference` and compare committed output. Which snapshot counters should change?
20. Disable compilation with `CompileConfig(enabled=False)`. Which compile counters/modes change, and why should output not change?
21. Explain why vectorized attention and compile-aware projection are optimizations at different architectural boundaries.

## Observability

22. Call `snapshot()` twice without running a step. Prove the two snapshots are equal and runtime state does not advance.
23. List every question that can now be answered without private-field access: admission, reuse, preemption, speculation, execution path, cache state and committed output.
24. Propose one useful new observation. Decide whether it belongs in Scheduler, cache, execution, async delivery or the snapshot-copy layer. Justify the ownership choice.

## Historical synthesis

25. Pick any adjacent pair from v0.1–v0.13. State exactly one new mechanism introduced and one invariant preserved.
26. Explain why v1.0 is primarily a composition release rather than a new mechanism release.
27. Before proposing v1.1, name the genuinely new learning target. If the proposal can be expressed only as “more features,” do not create the version.

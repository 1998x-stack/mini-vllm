# mini-vLLM exercises

Use historical branch diffs as the primary learning tool. Before changing v0.7, be able to explain the v0.1→v0.6 progression: static batching, continuous batching, paged KV, chunked prefill, shared prefixes, and recompute preemption.

## v0.7 speculative transactions

1. Run `examples/08_speculative_decoding.py` and identify the stable checkpoint position before each transaction.
2. Modify the scripted draft so candidate 0 is wrong. Explain why target logits can reject it without committing logical history.
3. Create a draft where candidate 0 is correct and candidate 1 is wrong. Inspect the block table before and after rollback.
4. Set `block_size=2` and construct a draft that allocates an extra physical block. Prove rollback returns that block to the allocator.
5. Explain why accepted draft tokens may keep their already-computed K/V while the fallback token remains logically committed but uncomputed.
6. Compare `output_token_ids` and `num_computed_tokens` immediately after a partial accept. Why should they differ by one?
7. Warm the prefix cache with one request, then run a speculative rollback on a second request with the same prefix. Verify the shared block bytes are unchanged.
8. Make the KV pool too small for speculative temporary state but large enough for normal decoding. Verify v0.7 falls back rather than preempting another request for acceleration.
9. Change `ScriptedDraftProvider` into a deliberately poor provider. Confirm correctness is unchanged while acceptance rate collapses.
10. Explain why v0.7 does not yet charge draft verification tokens to `max_num_batched_tokens`, and design a future budget model that could.

## Transaction safety

11. Inject an exception from `execute_speculative()` after temporary K/V has been written. Confirm the exception path restores the checkpoint frontier.
12. Try to rollback behind `checkpoint.num_tokens`. The cache must reject it.
13. Try to mutate a checkpoint-protected block-table prefix. Explain why commit/rollback should detect this corruption.
14. Explain why rejected tokens lingering in unused bytes of a retained physical block are harmless only if the logical/used-token frontier prevents them from becoming attention-visible.

## Preview v0.8

15. Split the current in-process path into conceptual messages: `SchedulerOutput`, `ExecutionRequest`, `ExecutionResult`.
16. Decide which objects must stay in `EngineCore` and which can move to a `Worker` without moving scheduling policy into the worker.
17. Identify which v0.7 transaction metadata would have to cross an executor boundary if speculative verification ran on a worker.

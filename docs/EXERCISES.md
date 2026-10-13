# mini-vLLM exercises

Use historical branch diffs as the primary learning tool. Before modifying v0.9, be able to explain why v0.8 separated EngineCore policy from Worker execution.

## v0.9 rank and collective semantics

1. Run `examples/10_tensor_parallel.py` and write down each rank's vocabulary range.
2. Change `vocab_size` from 25 to 26 and explain how `shard_range` balances rows across two ranks.
3. Construct `RankConfig(rank=2, world_size=2)`. Why must it fail immediately?
4. Call `ReferenceCollective.all_gather` on tensors whose non-gather dimensions disagree. Explain why a collective must reject this rather than silently reshape.
5. Run `ReferenceTensorParallel.column_parallel_linear` with `world_size=3` and compare with `torch.nn.functional.linear`.
6. Run the row-parallel oracle and identify where `all_reduce_sum` is required instead of `all_gather`.

## WorkerGroup topology

7. Trace one normal request: EngineCore → DistributedExecutor → WorkerGroup → two Workers → local logits → all-gather → EngineCore.
8. Prove WorkerGroup owns no Scheduler, waiting queue, running queue, or preemption-victim policy.
9. Explain why every rank sees the same `ExecutionItem` even though each rank owns different LM-head rows.
10. Inspect `cache_stats()`. Why does v0.9 require replicated rank caches to report equal statistics?
11. Change one rank's cache size and force capacity divergence. Why is an explicit error safer than letting ranks continue independently?

## Failure atomicity

12. Make rank 1 throw during ordinary forward after rank 0 has already executed. Verify all rank cache checkpoints return to the pre-forward token frontier.
13. Explain why EngineCore's existing failure-before-logical-commit rule is insufficient unless WorkerGroup also repairs physical rank state.
14. Make speculative target execution fail on one rank. Trace how the v0.7 transaction checkpoint now spans all ranks through DistributedExecutor.

## Reference vs distributed correctness

15. Run the same deterministic request through v0.8 and v0.9. Compare complete output-token histories.
16. Set `world_size=1` in v0.9. Why is this useful as a degenerate topology test?
17. Explain why the current model integration is real tensor sharding even though the Transformer trunk remains replicated.
18. Design the next step that shards attention/MLP projections. Where would all-reduce be needed?

## Preview v0.10

19. Sketch an `AsyncEngine` wrapper that submits requests without moving Scheduler ownership outside EngineCore.
20. Decide what a streaming token event should contain and how cancellation should interact with a request that is currently executing on WorkerGroup.
21. Explain why backpressure belongs at the async delivery boundary rather than inside the tensor-parallel collective.

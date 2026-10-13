# mini-vLLM exercises

Use historical branch diffs as the primary learning tool. Before changing v0.8, be able to explain the v0.1→v0.7 progression: batching, paged KV, chunked prefill, shared prefixes, recompute preemption, and speculative transactions.

## v0.8 execution boundary

1. Run `examples/09_executor_boundary.py`. Identify which objects belong to EngineCore, Executor, Worker, ModelRunner, and KV backend.
2. Inspect `ExecutionItem`. Explain why it carries `request_id` rather than a mutable `Request` object.
3. Add a field such as scheduler priority to `ExecutionItem`, then explain why that would weaken the policy/mechanism boundary.
4. Inspect Worker and prove it owns no waiting/running queues and no preemption-victim policy.
5. Trace one normal decode step from `ScheduleDecision` to `ExecutionRequest`, `LocalExecutor`, Worker, `ExecutionResult`, and logical commit.
6. Trace one speculative transaction. Which operations remain centralized and which physical operations cross Executor?
7. Inject an Executor that raises from `execute()`. Verify logical output and `num_computed_tokens` do not advance.
8. Make an Executor return logits for the wrong request-id set. Explain why EngineCore must reject the result before commit.
9. Compare `create_engine("v0.7")` and `create_engine("v0.8")` under the same deterministic model configuration and request. Their output histories should match.
10. Explain why `engine.cache` remains a useful LocalExecutor debug view but must not be required by EngineCore policy.

## Designing a future transport

11. Sketch a `ProcessExecutor` that serializes `ExecutionRequest` and returns `ExecutionResult`. Which current payload fields are easy to serialize?
12. Identify why sending Python `Request` objects to a remote Worker would couple transport to scheduler internals.
13. Decide which cache-lifecycle calls (`ensure_capacity`, `free`, `checkpoint`, `rollback`) would need request/response messages in a remote executor.
14. Design an executor failure envelope that can distinguish transport failure from model execution failure without silently committing request state.

## Preview v0.9

15. Replace one Worker conceptually with a `WorkerGroup` of rank 0/rank 1. Which EngineCore APIs should remain unchanged?
16. Split a toy linear layer across two ranks and describe how an educational `all_reduce` reconstructs the reference output.
17. Explain why tensor parallelism belongs below Executor rather than inside Scheduler.

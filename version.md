# mini-vLLM v0.10 — async engine, streaming, cancellation, and backpressure

## 1. Version identity

- **Branch:** `v0.10`
- **Previous version:** `v0.9`
- **Theme:** asynchronous caller-facing lifecycle and bounded streaming above the synchronous EngineCore
- **Status rule:** implementation is eligible for release only when the exact v0.10 commit has a green repository CI run; released `main` is the source of truth for validated state

v0.9 established a centralized synchronous EngineCore over a distributed WorkerGroup. v0.10 adds a caller-facing async layer without moving scheduling, speculation, distributed execution, or logical commit policy out of that core.

## 2. Primary mental model

```text
caller
  │ submit(Request)
  ▼
AsyncEngine
  │ pending intent + request handle
  ▼
EngineCore                         ← only serving-policy authority
  │
  ▼
DistributedExecutor / WorkerGroup ← unchanged v0.9 topology
  │
  ▼
EngineStepOutput
  │ committed logical changes only
  ▼
AsyncEngine
  ├─ TokenEvent
  ├─ FinishedEvent
  ├─ CancelledEvent
  └─ ErrorEvent
```

The async layer is a lifecycle/delivery boundary, not a second Scheduler.

## 3. Added abstractions

### `AsyncEngine`

Owns caller-facing handles, pending submissions, cancellation intent, one cooperative asyncio driver, and bounded event delivery. It delegates all serving policy to one synchronous EngineCore.

### `AsyncRequestHandle`

A per-request async iterator over a bounded `asyncio.Queue`. It exposes cancellation and enforces terminal-event finality.

### Stream events

```text
StreamEvent
  ├─ TokenEvent(token_id, output_index)
  ├─ FinishedEvent(output_token_ids, finish_reason)
  ├─ CancelledEvent(output_token_ids)
  └─ ErrorEvent(error_type, message)
```

Only committed logical tokens become `TokenEvent`.

## 4. Submission

`submit()` returns a handle immediately in the active event loop and puts the request into AsyncEngine's pending intent registry. The driver later calls `EngineCore.add_request()` at a cooperative boundary.

Request IDs are one-shot within an AsyncEngine instance. Duplicate submission fails deterministically before a second handle is accepted.

## 5. Commit-before-stream

AsyncEngine publishes token events exclusively from `EngineStepOutput.committed_token_ids`:

```text
speculative proposal
      ↓
target verify
      ↓
logical commit
      ↓
committed_token_ids
      ↓
TokenEvent
```

Rejected speculative tokens never become stream events. Multi-token speculative commits are emitted in committed order with monotonically increasing output indices.

## 6. Cancellation semantics

### Before EngineCore registration

```text
pending submission
   ↓ cancel intent
remove pending request
   ↓
CancelledEvent
```

No KV allocation or Scheduler registration occurs.

### After EngineCore registration

EngineCore now exposes explicit cancellation/abort lifecycle primitives. At the next cooperative boundary:

```text
Scheduler.cancel_request(request)
        ↓
Executor.free(request_id)
        ↓
Request.mark_finished("cancelled")
        ↓
CancelledEvent
```

The operation is idempotent from the async facade's perspective: once a handle is terminal, later cancellation returns false and cannot emit another terminal event.

### In-flight step

v0.10 does not interrupt a model forward mid-execution. If cancellation becomes runnable while one EngineCore step is already executing, that atomic step may complete and its committed token events may be delivered; cancellation is applied before the next scheduling step.

## 7. Terminal-event invariant

Each accepted request has exactly one terminal event:

```text
FinishedEvent
or CancelledEvent
or ErrorEvent
```

After terminal publication, AsyncRequestHandle rejects further event emission. Its async iterator ends when the terminal event is consumed.

## 8. Bounded backpressure

Every handle owns a bounded queue. The first educational policy is intentionally conservative:

```text
publish committed event
      ↓
queue full?
  ├─ no  → continue
  └─ yes → await queue capacity
              ↓
           driver pauses
              ↓
      no new EngineCore step yet
```

A slow consumer therefore delays global cooperative progress, but committed tokens are never silently dropped or reordered. Backpressure remains above WorkerGroup and collective execution.

## 9. Driver lifecycle

One cooperative driver performs:

```text
drain queued cancellation
   ↓
admit pending submissions
   ↓
cancel registered requests
   ↓
run one EngineCore.step()
   ↓
publish committed events
   ↓
yield to asyncio event loop
```

When no pending intents and no unfinished core requests remain, the task exits instead of busy-spinning. A future submission starts another driver task.

## 10. Failure semantics

The reference policy is deliberately conservative. An uncaught EngineCore/Executor step failure becomes a core-wide async failure:

- no fabricated token is emitted;
- all nonterminal handles receive one `ErrorEvent`;
- AsyncEngine attempts to abort registered requests and free their active KV before terminal publication;
- the original execution exception remains the caller-facing error cause.

More selective distributed recovery is outside v0.10 scope.

## 11. Critical invariants

- **Single-policy:** AsyncEngine does not own a shadow serving Scheduler.
- **Commit-before-stream:** only committed output history is streamable.
- **Event order:** token events match `Request.output_token_ids` order.
- **Terminal:** exactly one terminal event; nothing afterward.
- **Cancellation release:** a cancelled registered request stops scheduling and releases active KV once.
- **Async equivalence:** collected deterministic v0.10 tokens equal the v0.9 synchronous reference.
- **Backpressure:** bounded queues may delay progress but cannot silently drop/reorder committed events.
- **Distributed preservation:** v0.9 WorkerGroup/tensor-parallel semantics remain unchanged beneath the facade.

## 12. Important source files

```text
mini_vllm/async_runtime/events.py
mini_vllm/async_runtime/handle.py
mini_vllm/async_runtime/engine.py
mini_vllm/core/engine.py
mini_vllm/scheduler/base.py
mini_vllm/versions/v0_10/engine.py
tests/test_async_runtime.py
tests/test_v010_version.py
examples/11_async_streaming.py
```

Recommended comparison:

```bash
git diff v0.9..v0.10 -- mini_vllm tests examples version.md
```

## 13. Validation contract

The exact commit promoted to `main` must pass GitHub Actions `./scripts/check.sh`. Coverage includes:

- async-collected output vs deterministic v0.9 synchronous output;
- committed token ordering and exactly one terminal event;
- duplicate request rejection;
- cancellation before registration with no KV allocation;
- active cancellation with Scheduler removal and KV release;
- bounded-queue backpressure without token loss/reordering;
- core/executor failure to explicit `ErrorEvent` with resource cleanup;
- v0.10 composition over the v0.9 DistributedExecutor/WorkerGroup;
- all historical v0.1-v0.9 regression tests;
- executable examples through `examples/11_async_streaming.py`.

## 14. Intentional simplifications

- one asyncio event loop;
- one cooperative driver task;
- synchronous model/core step execution inside that task;
- global driver stall when any event queue applies backpressure;
- no FastAPI/SSE/WebSocket server;
- no threads/processes for frontend driving;
- no mid-forward interruption;
- no production distributed recovery;
- no per-request independent delivery task optimization.

## 15. Learning goals

After v0.10, a learner should be able to explain sync EngineCore vs async facade, commit-before-stream, queued vs registered cancellation, terminal exactly-once semantics, where backpressure belongs, why async does not require a second Scheduler, and why v0.9 distributed execution can remain unchanged below the delivery layer.

## 16. Next version

`v0.11` studies graph/compile-aware execution: eager correctness oracles, static metadata, graph breaks, capture/recapture boundaries, and host-overhead reduction without moving serving or async lifecycle policy into graph-specific code.

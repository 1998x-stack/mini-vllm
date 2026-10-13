# mini-vLLM Architecture — v0.10

## Async delivery above a singular control plane

```text
Caller
  │ submit / cancel
  ▼
AsyncEngine
  │ handles + bounded event queues
  ▼
EngineCore
  request registry
  Scheduler
  token budget
  preemption / cancellation policy
  sampling / logical commit
        │
        ▼
ExecutionRequest
        │
        ▼
DistributedExecutor
        │
        ▼
WorkerGroup(world_size=N)
        │
        ▼
rank-local model + replicated KV
        │
        ▼
ReferenceCollective
        │
        ▼
ExecutionResult
        │
        ▼
EngineCore commit
        │
        ▼
EngineStepOutput.committed_token_ids
        │
        ▼
AsyncEngine stream events
```

AsyncEngine owns lifecycle/delivery coordination, not serving priority. No rank and no async handle owns a Scheduler.

## Commit-before-stream

The async facade publishes `TokenEvent` only from `EngineStepOutput.committed_token_ids`. Temporary speculative candidates are therefore invisible to callers until target verification has committed them.

```text
draft proposal
   ↓
target verification
   ↓
accepted logical commit
   ↓
committed_token_ids
   ↓
TokenEvent
```

Rejected drafts never enter the stream.

## Cooperative driver

The first async implementation intentionally runs one asyncio driver task:

```text
drain submit/cancel intents
       ↓
run one synchronous EngineCore.step()
       ↓
publish committed events
       ↓
await asyncio.sleep(0)
       ↓
repeat while work exists
```

There is no busy-spin while idle; once no pending/core work remains the driver task exits. A later submission starts a new driver task.

## Cancellation boundary

v0.10 does not attempt unsafe mid-forward interruption. Cancellation becomes effective between atomic EngineCore steps:

```text
current step already executing
       ↓
step succeeds + commits
       ↓
committed events may be delivered
       ↓
cancel intent applied
       ↓
Scheduler removal
       ↓
active KV release
       ↓
logical finish_reason = cancelled
       ↓
CancelledEvent
```

Cancellation before EngineCore admission removes the pending submission and emits `CancelledEvent` without allocating KV.

## Exactly one terminal event

Every accepted handle terminates with exactly one of:

```text
FinishedEvent
CancelledEvent
ErrorEvent
```

`AsyncRequestHandle` refuses publication after a terminal event and its async iterator stops when that terminal event is consumed.

## Backpressure

Each request handle owns a bounded `asyncio.Queue`. The reference policy is intentionally global/cooperative:

```text
EngineCore commits
   ↓
AsyncEngine tries queue.put(event)
   ↓
queue full?
   ├─ no  → deliver and continue
   └─ yes → await consumer capacity
              ↓
            no next EngineCore step yet
```

No committed event is dropped or reordered. This is a correctness-first teaching policy, not an optimal high-throughput serving policy.

## Error policy

The first reference implementation treats an uncaught EngineCore/Executor step failure conservatively as a core-wide async failure: remaining active handles receive `ErrorEvent`, and the facade attempts to abort/release registered request resources before publishing terminal errors.

## Distributed execution remains unchanged

v0.10 retains v0.9's single-process logical-rank topology:

```text
DistributedExecutor
  ↓
WorkerGroup
  ├─ rank 0: replicated trunk/KV + vocab shard
  ├─ rank 1: replicated trunk/KV + vocab shard
  └─ ...
  ↓
ReferenceCollective.all_gather
  ↓
full logits
```

Async delivery is above this topology. It does not alter collective semantics or distributed failure rollback.

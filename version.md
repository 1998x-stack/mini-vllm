# mini-vLLM v0.10 — async engine, streaming, cancellation, and backpressure (planning branch)

## 1. Version identity

- **Branch:** `v0.10`
- **Base:** latest validated `v0.9` / `main`
- **Previous version:** `v0.9`
- **Theme:** asynchronous caller-facing request lifecycle without moving serving policy out of EngineCore
- **Status:** planning / implementation target; runtime is intentionally identical to v0.9 until the acceptance criteria below pass

## 2. Why v0.10 follows v0.9

By v0.9 the synchronous core has explicit boundaries for:

- centralized request/Scheduler policy;
- paged/prefix-aware KV ownership;
- preemption/recomputation;
- speculative transactions;
- Executor / Worker separation;
- logical multi-rank WorkerGroup execution.

The next question is no longer “where does model work execute?” but:

> How can callers submit, observe, cancel, and consume requests asynchronously without duplicating Scheduler state or weakening token/cache invariants?

The async layer must wrap the existing synchronous runtime. It must not become a second EngineCore.

## 3. Primary mental model

```text
caller
  │ submit(request)
  ▼
AsyncEngine
  │ owns caller-facing handles / event queues
  ▼
submission queue
  │
  ▼
EngineCore                         ← single source of serving policy
  │
  ▼
DistributedExecutor / WorkerGroup ← unchanged v0.9 execution topology
  │
  ▼
EngineStepOutput
  │ committed logical changes only
  ▼
AsyncEngine event fan-out
  │
  ├── TokenEvent
  ├── FinishedEvent
  ├── CancelledEvent
  └── ErrorEvent
  │
  ▼
per-request async stream
```

Async delivery is an API/lifecycle boundary. Scheduler ordering, token budgets, preemption victims, collectives, and KV ownership remain in the synchronous core.

## 4. Required abstractions

The first implementation should introduce small inspectable contracts such as:

```text
AsyncEngine
AsyncRequestHandle
StreamEvent
  ├── TokenEvent
  ├── FinishedEvent
  ├── CancelledEvent
  └── ErrorEvent
CancellationRequest / cancellation state
bounded per-request event queue
```

Names may evolve during implementation, but responsibilities must stay explicit.

## 5. Submission contract

`submit()` should return immediately with a request handle rather than running the request to completion.

Conceptually:

```text
handle = async_engine.submit(Request(...))

async for event in handle:
    ...
```

Submission should enqueue caller intent. The EngineCore request registry remains the authoritative logical request state after admission into the synchronous runtime.

Duplicate request IDs must fail deterministically.

## 6. Streaming contract

Streaming events may only describe **committed logical state**.

For example, speculative draft candidates are not streamable merely because temporary K/V exists:

```text
draft token
   ↓
target verify
   ↓
committed output token
   ↓
TokenEvent             ✓
```

Rejected speculative tokens:

```text
never become TokenEvent
```

For multi-token speculative commits, events must preserve the committed token order exactly.

## 7. Terminal-event invariant

Each accepted request stream has exactly one terminal outcome:

```text
FinishedEvent
or
CancelledEvent
or
ErrorEvent
```

After a terminal event:

- no additional `TokenEvent` may be emitted;
- no second terminal event may be emitted;
- the async iterator must terminate predictably.

## 8. Cancellation semantics

Cancellation cannot mean “stop listening while the runtime keeps consuming KV”. It must be a real lifecycle transition coordinated with EngineCore.

At minimum distinguish:

### Cancellation before EngineCore admission

```text
submission queue
   ↓ cancel
remove pending submission
   ↓
CancelledEvent
```

No KV should ever be allocated.

### Cancellation after EngineCore registration/admission

EngineCore needs an explicit cancellation path that atomically coordinates:

```text
request lifecycle
scheduler queues
active/prefix cache ownership
logical terminal state
```

Required outcome:

```text
cancel active request
      ↓
remove from Scheduler participation
      ↓
release active KV ownership exactly once
      ↓
mark logical request terminal/cancelled
      ↓
CancelledEvent
```

If cancellation arrives while one synchronous execution step is already in progress, the first educational policy should be explicit and deterministic, for example:

```text
current atomic EngineCore step completes
      ↓
committed output from that step is observed
      ↓
cancellation is applied before next scheduling step
```

Do not introduce unsafe mid-forward tensor interruption in v0.10.

## 9. Backpressure

Per-request output queues should be bounded and observable.

The first version must choose and test an explicit policy rather than allowing unbounded token accumulation. Preferred educational direction:

```text
EngineCore commits token
      ↓
AsyncEngine delivers event
      ↓
bounded consumer queue
```

A slow consumer must not corrupt EngineCore state. The async delivery layer may pause its own fan-out or scheduling loop according to a documented policy, but it must not silently drop committed token events.

Backpressure belongs above WorkerGroup/collective execution.

## 10. Synchronous-core compatibility

v0.10 should preserve the v0.9 core as the correctness oracle.

For the same deterministic requests:

```text
v0.9 synchronous final output
          ==
v0.10 async stream collected output
```

The async facade should call normal EngineCore operations rather than reimplement:

- admission;
- token budgeting;
- preemption;
- speculation;
- distributed execution;
- sampling.

## 11. Fairness and driving the core

The first implementation may use a single cooperative `asyncio` driver task:

```text
while work exists:
    drain new submissions/cancellations
    run one EngineCore step
    publish committed events
    yield to event loop
```

This is preferred over immediately introducing threads/processes because the learning target is async lifecycle ownership, not CPU/GPU overlap.

The driver must not busy-spin when there is no work.

## 12. Failure semantics

An EngineCore/Executor failure must be surfaced to the affected stream as `ErrorEvent` (or equivalent terminal error contract) without fabricating committed tokens.

The design must state whether a core-wide failure terminates all active streams or only the affected request. For the first reference implementation, a conservative deterministic policy is preferred over hidden recovery magic.

## 13. Critical invariants

### Single-policy invariant

There is still exactly one EngineCore/Scheduler policy. AsyncEngine does not own a shadow waiting/running queue with independent serving priority.

### Commit-before-stream invariant

A token is emitted only after it exists in committed logical request history.

### Event-order invariant

Streamed token IDs are ordered exactly like `Request.output_token_ids`.

### Terminal invariant

Exactly one terminal event occurs; nothing is emitted afterward.

### Cancellation-release invariant

A cancelled registered request stops consuming Scheduler/KV resources and releases active cache ownership exactly once.

### Async-equivalence invariant

Collecting all token events for an uncancelled deterministic request produces the same output as synchronous v0.9 execution.

### Backpressure invariant

A slow consumer may delay delivery according to policy, but committed events are neither silently dropped nor reordered.

## 14. Preferred implementation order

```text
1. define stream event types
2. define AsyncRequestHandle
3. add EngineCore cancellation primitive
4. implement AsyncEngine submission registry
5. add one cooperative driver loop
6. publish committed tokens from EngineStepOutput
7. implement queued and active cancellation
8. add exactly-one-terminal-event enforcement
9. add bounded queue / backpressure behavior
10. add async-vs-sync equivalence tests
11. add cancellation/resource-release tests
12. add executable streaming trace example
```

## 15. Acceptance criteria

v0.10 cannot be frozen until all applicable items are true:

- [ ] `submit()` is non-blocking from the caller perspective and returns a request handle;
- [ ] request handles expose an async token/event stream;
- [ ] streamed tokens are emitted only after logical commit;
- [ ] multi-token speculative commits stream in exact order;
- [ ] duplicate request IDs are rejected;
- [ ] queued cancellation works without KV allocation;
- [ ] active cancellation removes Scheduler participation and releases active KV exactly once;
- [ ] cancellation during an in-flight synchronous step follows a documented deterministic boundary;
- [ ] exactly one terminal event is emitted;
- [ ] no events occur after terminal state;
- [ ] bounded event queues/backpressure do not drop or reorder committed tokens;
- [ ] an idle async driver does not busy-spin;
- [ ] executor/core failures become explicit terminal stream errors;
- [ ] v0.9 synchronous and v0.10 async-collected deterministic outputs match;
- [ ] distributed WorkerGroup and speculative transaction regressions still pass;
- [ ] all v0.1-v0.9 tests continue to pass;
- [ ] at least one executable example traces submit → token events → finish;
- [ ] at least one executable example/test demonstrates cancellation;
- [ ] exact release-candidate CI passes before synchronization to `main`.

## 16. Explicit non-goals

- FastAPI/OpenAI-compatible HTTP server;
- WebSocket/SSE networking;
- authentication/rate limiting;
- multi-process async frontend;
- production GPU overlap or CUDA stream scheduling;
- mid-kernel/mid-forward interruption;
- sophisticated distributed fault recovery;
- CUDA Graph/`torch.compile` integration — v0.11;
- optimized Triton/CUDA PagedAttention — v0.12.

## 17. Learning goals

After v0.10, the learner should be able to explain:

- sync EngineCore vs async caller facade;
- submission state vs EngineCore request state;
- why tokens stream only after commit;
- how cancellation must coordinate Scheduler and KV ownership;
- why terminal events need exactly-once semantics;
- where backpressure belongs;
- why an async API does not require duplicating serving policy;
- how v0.9 distributed execution can remain unchanged under an async frontend.

## 18. Next version

`v0.11` studies **graph/compile-aware execution**: static-shape assumptions, graph breaks, recapture boundaries, and host-overhead reduction while keeping the eager v0.10 path as the correctness oracle.

# mini-vLLM Architecture — v1.3

## Complete serving stack

v1.3 wraps the validated v1.2 artifact/multiprocess runtime with an explicit network edge.

```text
NETWORK CLIENT
      │
      ▼
TCP connection
      │
      ▼
NetworkServer
├─ bounded HTTP/1.1 parsing
├─ endpoint/method/content-type validation
├─ JSON → GenerateRequest
├─ ingress-capacity accounting
├─ health/readiness
├─ disconnect watcher
└─ JSON/SSE serialization
      │
      ▼
WhitespaceTokenizer
      │ text → token IDs
      ▼
Request
      │
      ▼
AsyncEngine
      │ pending / bounded event queue / cancellation intent
      ▼
EngineCore
      │ logical truth / atomic step / commit authority
      ▼
PrefixAwareBudgetedScheduler
      │ admission / token budget / fairness
      ▼
ArtifactMultiprocessExecutor
      │ IPC / worker health / rollback / rank identity
      ├────────────────────┐
      ▼                    ▼
child rank 0          child rank N
Worker                Worker
├─ validated model    ├─ validated model
├─ paged KV           ├─ paged KV
├─ PagedAttention     ├─ PagedAttention
└─ compile cache      └─ compile cache
      │                    │
      └──── acknowledged rank results ────┐
                                          ▼
                                  EngineCore commit
                                          │
                                          ▼
                               AsyncRequestHandle events
                                          │
                                          ▼
                                  JSON or SSE bytes
```

## Five ownership layers

### 1. Network connection ownership

`NetworkServer` owns TCP listeners, connection tasks, protocol parsing, request byte limits, network ingress slots, health/readiness and response serialization.

It does not own Scheduler policy, KV blocks, model parameters or child processes unless it also owns the engine as a higher-level resource.

### 2. Async request ownership

`AsyncEngine` owns pending submissions, `AsyncRequestHandle`, bounded per-request event queues and cancellation intent. Network backpressure may delay consumption of these events, but cannot modify their meaning.

### 3. Logical serving ownership

`EngineCore` remains the logical commit authority. Only it may advance request state after physical execution is acknowledged.

### 4. Artifact/execution ownership

The v1.2 contracts remain unchanged:

```text
external artifact bytes
   ↓ strict validation
ValidatedModelArtifact
   ↓ explicit parameter copy
child runtime-owned model state
```

Each worker independently proves the configured artifact identity before serving.

### 5. Physical KV ownership

The child cache owns block tables/refcounts/physical slots. Neither NetworkServer nor Scheduler receives those ownership handles.

## HTTP ingress pipeline

```text
socket bytes
   ↓ max_header_bytes
request line + headers
   ↓ exact educational HTTP rules
method / path / content-type / content-length
   ↓ max_request_bytes
UTF-8 body
   ↓ JSON parse
exact API field set
   ↓ semantic validation
GenerateRequest
   ↓ tokenizer
Request(prompt_token_ids)
   ↓ engine.submit()
```

Failure at any stage above `engine.submit()` creates no runtime request.

## Reference endpoints

```text
GET  /health
GET  /ready
POST /v1/generate
POST /v1/generate/stream
```

The reference protocol is intentionally narrow: HTTP/1.1, one request per connection, `Content-Length` request bodies and `Connection: close`. Transfer-encoded/chunked requests are rejected.

## Capacity model

v1.3 has four independent capacity concepts:

```text
NetworkConfig.max_inflight_requests
        │ API-edge lifecycle slots
        ▼
AsyncEngine.max_queue_size
        │ per-request committed-event delivery queue
        ▼
Scheduler max_num_seqs / token budget
        │ logical scheduling policy
        ▼
KV num_blocks
        │ physical storage
```

A full network ingress returns 429 before submission. It does not consume Scheduler/KV state.

A full AsyncEngine event queue backpressures the driver until the network/client consumes committed events; it does not drop tokens.

Scheduler and KV pressure continue to use their historical mechanisms independently.

## Streaming path

The network server never reads speculative/model internals directly. Its only stream source is `AsyncRequestHandle`:

```text
committed TokenEvent
      ↓
event: token

the single terminal StreamEvent
      ├─ FinishedEvent  → event: done
      ├─ CancelledEvent → event: cancelled
      └─ ErrorEvent     → event: error
```

Serialization or a slow socket can delay visibility. It cannot move visibility earlier than logical commit.

## Non-stream path

`/v1/generate` still consumes the full bounded `AsyncRequestHandle` stream rather than simply waiting for terminal metadata. This matters because the producer itself respects bounded delivery queues; a non-stream HTTP handler must drain intermediate committed token events to avoid deadlock.

After the terminal event, one JSON response is serialized with final token IDs/text/finish reason or an explicit error/cancellation payload.

## Disconnect lifecycle

Once a request is submitted, the server watches the connection independently from response production:

```text
connection EOF/reset
      ↓
_cancel_for_disconnect(handle)
      ↓
AsyncRequestHandle.cancel()
      ↓
AsyncEngine cancellation intent
      ↓ cooperative boundary
EngineCore.cancel_request()
      ↓
Scheduler removal + cache release
```

The watcher does not terminate worker processes and does not mutate physical state itself.

If the socket is no longer writable, the server may be unable to deliver the request's terminal network bytes; the runtime still converges to exactly one logical terminal outcome.

## Health and readiness

The server exposes two different facts:

```text
health
  = NetworkLifecycle in READY or DRAINING

ready
  = NetworkLifecycle == READY
  + engine snapshot succeeds
  + execution.healthy
  + model_artifact exists
  + model_artifact.rank_identity_agreement
```

`lifecycle_state` may report `unhealthy` when the network listener is internally READY but deeper execution readiness is false.

This avoids treating “port is bound” as proof that requests can safely run.

## Lifecycle and shutdown

```text
CREATED
   ↓ start()
STARTING
   ↓ bind
READY
   │
   ├─ deeper runtime unhealthy → externally not ready
   │
   ↓ close()
DRAINING
   ├─ stop accepting
   ├─ cancel active request handles
   ├─ wait for connection tasks
   └─ cancel stuck connection tasks after timeout
   ↓
CLOSED
```

Ownership determines deeper cleanup:

```text
engine.create_server()
  → server owns listener/connections only

create_server(...)
  → server owns listener + constructed engine
  → close() also aclose()s executor/workers/temp artifact
```

## Network observability

`NetworkSnapshot` contains copies of:

```text
lifecycle_state
host / port
health / ready
active_connections
inflight_requests
accepted_requests
rejected_requests
completed_requests
cancelled_requests
failed_requests
disconnect_cancellations
```

Calling `snapshot()` does not parse input, submit work, consume stream events, mutate counters, cancel requests or touch physical cache state.

## Error mapping

Representative network-boundary statuses:

```text
400 malformed protocol / JSON
404 unknown endpoint
405 wrong method
409 duplicate request_id
411 missing Content-Length for POST
413 body too large
415 unsupported content type
422 semantically invalid generation request
429 ingress full
431 headers too large
503 not ready / cancelled non-stream result
500 internal runtime/API failure
```

An HTTP status is a network/API fact. It is not a hidden Scheduler state transition.

## Stable historical boundaries underneath

All prior correctness layers remain active:

```text
network bytes != validated API request
validated API request != token-native Request
logical request state != physical KV ownership
parent logical truth != child physical truth
artifact file identity != runtime parameter storage
transport send != acknowledged execution
```

v1.3 composes these boundaries rather than collapsing them.

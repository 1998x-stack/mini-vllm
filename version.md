# mini-vLLM v1.3 — network serving / API boundary (planning branch)

## Version identity

- Branch: `v1.3`
- Base: validated/frozen `v1.2` / `main`
- Previous: `v1.2`
- Theme: network bytes → validated API request → tokenizer → token-native runtime → streamed HTTP response
- Status: planning; runtime remains exactly v1.2 until v1.3 acceptance criteria pass

## Why v1.3

v1.2 made model startup trustworthy before serving begins. The remaining outer boundary is the network edge. Today callers invoke Python methods directly; a real serving runtime must accept untrusted request bytes, validate them, apply ingress capacity, map disconnects to request lifecycle, and expose liveness/readiness without confusing those concerns with Scheduler policy.

```text
TCP connection
    ↓
HTTP request bytes
    ↓ parse + validate
API request object
    ↓ tokenize
existing token-native Request
    ↓
AsyncEngine / EngineCore
    ↓ committed events
HTTP JSON or SSE stream
```

The lesson is API-edge ownership and lifecycle, not a new model, scheduler, KV algorithm or web framework.

## Primary design

Use a deliberately small reference server built on `asyncio.start_server` so the protocol boundary is inspectable. The supported educational HTTP/1.1 subset is intentionally narrow:

- one request per connection;
- `Content-Length` request bodies only;
- JSON request bodies;
- `Connection: close` responses;
- JSON completion endpoint;
- SSE streaming endpoint;
- no chunked request bodies, keep-alive multiplexing, HTTP/2 or TLS termination.

### Endpoints

```text
GET  /health
GET  /ready
POST /v1/generate
POST /v1/generate/stream
```

`/health` proves that the network server task is alive. `/ready` additionally requires the underlying v1.2 runtime/executor and model-artifact rank agreement to be ready for admission.

### Generate request

Preferred JSON contract:

```json
{
  "request_id": "optional-client-id",
  "prompt": "tok1 tok2",
  "max_new_tokens": 4,
  "temperature": 0.0,
  "top_k": null,
  "top_p": null
}
```

Validation occurs completely before `engine.submit_text(...)`.

### Non-stream response

```json
{
  "request_id": "...",
  "output_token_ids": [1, 2],
  "text": "tok1 tok2",
  "finish_reason": "length"
}
```

### Streaming response

The stream uses SSE with logical event names:

```text
event: token
data: {"request_id":"...","token_id":7,"output_index":0,"text":"tok7"}

event: done
data: {"request_id":"...","output_token_ids":[7],"finish_reason":"length"}
```

Cancelled/error terminal events have their own event names. Only committed `AsyncRequestHandle` events may cross the socket.

## Required delta

### 1. Network configuration

Add a small immutable `NetworkConfig`, covering at least:

```text
host
port
max_inflight_requests
max_request_bytes
max_header_bytes
disconnect_cancel
```

Default bind should be loopback and an ephemeral port for tests/examples.

### 2. Explicit API request type

Parsing/validation must produce an immutable `GenerateRequest` (or equivalent) before tokenization/submission. Invalid method/path/content type/JSON/field types/ranges fail at the network boundary and must not create an AsyncEngine request.

### 3. Ingress backpressure

The network layer owns a bounded in-flight admission slot count independent of the AsyncEngine event queue and Scheduler `max_num_seqs`.

```text
socket accepted
    ↓
valid request
    ↓
ingress slot available?
  yes → submit to runtime
  no  → HTTP 429; no runtime request exists
```

This is edge backpressure, not a Scheduler priority signal.

### 4. Disconnect-driven cancellation

Once a network request has been submitted, connection EOF/disconnect should become cancellation intent when configured:

```text
client disconnect
    ↓
network connection watcher
    ↓
AsyncRequestHandle.cancel()
    ↓
existing cooperative cancellation boundary
```

Disconnect must not kill worker processes or mutate EngineCore outside its existing atomic-step rules.

### 5. Streaming correctness

SSE emits only committed runtime events. The network layer may serialize/delay them, but cannot fabricate tokens or expose speculative/uncommitted state.

### 6. Health versus readiness

Keep liveness and readiness distinct:

```text
health = network server loop accepting/managed
ready  = health + accepting new API requests + underlying execution healthy + model rank identity agreed
```

During draining, `/health` may remain true while `/ready` is false.

### 7. Server lifecycle

Preferred lifecycle:

```text
CREATED → STARTING → READY → DRAINING → CLOSED
                     └──────→ UNHEALTHY (observable readiness false)
```

`start()` binds the socket; `close()` stops new accepts, cancels/drains active API requests according to the reference policy, closes connection tasks, and optionally closes an engine it owns.

### 8. Ownership modes

Support both:

- server created around an existing v1.3 engine (`owns_engine=False`);
- integrated `create_server(...)` convenience that constructs/owns the engine and reaps its worker processes on server close.

### 9. Network observability

Expose a read-only network snapshot with logical facts such as:

```text
lifecycle state
bound host/port
ready
active connections
inflight requests
accepted/rejected requests
disconnect cancellations
completed/failed/cancelled API requests
```

Do not expose socket objects as serving-policy inputs.

## Critical invariants

### Parse-before-submit

Malformed or semantically invalid API input cannot create a runtime request.

### Edge capacity is not scheduler capacity

An ingress slot is only permission to enter the Python serving facade. It is not Scheduler admission, KV reservation, computed-token state or worker capacity.

### Committed-only network visibility

A token may cross the network only after the AsyncEngine emitted a committed `TokenEvent`.

### Exactly one API terminal outcome

A submitted API request yields exactly one network terminal result among finished/cancelled/error, unless the client disconnected and no terminal bytes can be delivered.

### Disconnect semantics

Disconnect produces cancellation intent; it never directly mutates KV or terminates a healthy worker.

### Readiness truth

A bound socket alone is not readiness. Readiness requires the underlying artifact-bound executor to be healthy and rank identities to agree.

### Shutdown ownership

A server that owns an engine must leave no child workers alive after successful close. A server wrapping an external engine must not close that engine unless explicitly configured to own it.

### Historical semantic equivalence

The same valid prompt sent through the v1.3 HTTP endpoint must produce the same committed token IDs as direct v1.2 `submit_text()` under equivalent configuration.

## Preferred modules

```text
mini_vllm/network/
├─ __init__.py
├─ types.py
└─ server.py

mini_vllm/versions/v1_3/
├─ __init__.py
└─ engine.py
```

Keep network protocol parsing/serialization separate from EngineCore and Scheduler.

## Tests required

- health/readiness lifecycle;
- valid JSON completion response;
- SSE token + terminal ordering;
- HTTP output equivalence with direct v1.2;
- invalid JSON/fields/method/path/content type rejected before runtime submission;
- request body/header limits;
- duplicate request ID mapped deterministically;
- max in-flight ingress returns 429 with no hidden submission;
- streaming client disconnect triggers cooperative cancellation;
- network close drains/cancels active connections without orphan worker processes;
- external-engine server close does not close external engine;
- network snapshot is read-only and internally consistent;
- all v0.1–v1.2 tests remain green.

## Integrated example target

Add `examples/18_network_server.py` showing:

```text
create v1.3 server on 127.0.0.1:0
→ GET /health + /ready
→ POST JSON generation
→ POST SSE generation
→ inspect network + runtime snapshots
→ close server
→ prove worker cleanup
```

The example may use a tiny standard-library asyncio client so no third-party HTTP dependency is required.

## Acceptance criteria

- a real TCP listener exposes the four reference endpoints;
- HTTP/JSON parsing is bounded and validated before submission;
- token-native serving core remains unchanged in responsibility;
- bounded ingress capacity is independent of Scheduler/KV policy;
- streaming exposes only committed async events;
- configured client disconnects become cooperative request cancellation;
- health/readiness reflect distinct lifecycle truths;
- server shutdown owns/reaps exactly the resources it created;
- public network observability contains no live socket/worker ownership handles;
- HTTP output is deterministic and equivalent to direct v1.2 serving;
- examples 01–17 remain green plus example 18;
- exact cleaned release SHA passes repository CI before `main` advances.

## Non-goals

- production HTTP parser compliance;
- HTTP/2 or HTTP/3;
- TLS/certificate management;
- authentication/API keys;
- rate limiting by tenant/account;
- OpenAI API compatibility;
- tool calls/chat templates;
- multi-node routing/load balancing;
- reverse proxy functionality;
- websocket transport;
- production metrics exporters;
- model hot swap;
- new Scheduler/KV/model algorithms.

## Learning goal

By v1.3 a reader should be able to explain the complete outer serving boundary: network bytes must be bounded and validated, ingress capacity is separate from model scheduling, disconnect is cancellation intent rather than process death, readiness depends on the deeper runtime being healthy, and only committed runtime events may become externally visible bytes.

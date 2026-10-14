# mini-vLLM v1.3 — network serving / API boundary

## Version identity

- Branch: `v1.3`
- Previous: validated/frozen `v1.2`
- Theme: bounded network bytes → validated API request → tokenizer → token-native runtime → committed JSON/SSE bytes
- Package version: `1.3.0`
- Runtime default: `create_engine()` selects v1.3
- Status: implementation complete; release content is frozen by the exact-SHA CI/main synchronization rule

## Why this version exists

v1.2 established a trustworthy startup boundary:

```text
external artifact bytes
    ↓ validate
runtime-owned model/tokenizer state
    ↓ rank identity agreement
READY
```

v1.3 adds the missing outer serving boundary:

```text
untrusted network bytes
    ↓ bounded parse / API validation
GenerateRequest
    ↓ tokenizer
Request(token IDs)
    ↓ existing serving runtime
committed StreamEvent
    ↓ serialization
network bytes
```

The central lesson is that a listening socket, a valid API request, Scheduler admission and KV ownership are four different truths.

## Delta from v1.2

### Reference network server

`mini_vllm/network/server.py` introduces a deliberately small `asyncio.start_server` HTTP/1.1 implementation. It uses no third-party web framework so request framing, limits, lifecycle and disconnect behavior remain inspectable.

Supported educational contract:

```text
one request per TCP connection
HTTP/1.1
Content-Length request bodies
application/json generation input
Connection: close
JSON terminal response
SSE streaming response
```

Unsupported transfer-encoded/chunked requests are rejected rather than partially interpreted.

### Public endpoints

```text
GET  /health
GET  /ready
POST /v1/generate
POST /v1/generate/stream
```

Generation JSON supports:

```text
request_id        optional; deterministic server-generated ID otherwise
prompt            required non-empty string
max_new_tokens    required positive integer
temperature       optional
top_k             optional
top_p             optional
```

### Parse-before-submit contract

The server validates, in order:

```text
header byte limit
HTTP request-line/version
header syntax/body framing
body byte limit
method/path/content type
UTF-8 JSON syntax
exact API field set
field types/ranges
SamplingParams
prompt tokenization
max model length
```

Only then is a token-native core `Request` constructed and submitted.

Malformed/invalid input therefore cannot create a hidden AsyncEngine/Core request.

### Network configuration

`NetworkConfig` contains:

```text
host = 127.0.0.1
port = 0
max_inflight_requests
max_request_bytes
max_header_bytes
disconnect_cancel
shutdown_timeout_s
```

`RuntimeConfig.network` composes this with the existing artifact/cache/process/runtime configuration.

### Ingress backpressure

`NetworkServer` tracks submitted network requests in `_inflight` and checks `max_inflight_requests` before `engine.submit()`.

```text
valid request
   ↓
ingress slot available?
   ├─ yes → submit
   └─ no  → HTTP 429, no runtime request
```

This is intentionally independent from:

```text
AsyncEngine max_queue_size
Scheduler max_num_seqs / token budget
KV physical block capacity
```

### Non-stream response

`POST /v1/generate` consumes the full `AsyncRequestHandle` event stream and returns one final JSON result.

Draining intermediate token events is required even for non-stream clients because the existing AsyncEngine queue is bounded. Waiting only for terminal metadata could deadlock after the event queue fills.

### SSE response

`POST /v1/generate/stream` maps only committed runtime events:

```text
TokenEvent      → event: token
FinishedEvent   → event: done
CancelledEvent  → event: cancelled
ErrorEvent      → event: error
```

The network layer has no direct access to speculative candidates or uncommitted logits.

### Disconnect cancellation

Once a request has been submitted, an independent reader task watches client EOF/reset. With `disconnect_cancel=True`:

```text
client disconnect
    ↓
AsyncRequestHandle.cancel()
    ↓
existing AsyncEngine cancellation intent
    ↓
cooperative EngineCore step boundary
    ↓
Scheduler/cache cleanup
```

`_disconnect_cancelled_ids` prevents duplicate disconnect accounting for one request.

Worker termination is never used as a request-cancellation primitive.

### Health and readiness

The implementation makes these explicitly different:

```text
health
  = NetworkServer is READY or DRAINING

ready
  = NetworkServer internal state is READY
  + engine.snapshot() succeeds
  + execution.healthy
  + model artifact snapshot exists
  + rank artifact identity agreement
```

A listener can therefore remain alive without falsely claiming model-serving readiness.

### Lifecycle

```text
CREATED
  ↓ start()
STARTING
  ↓ socket bind
READY
  ↓ close()
DRAINING
  ├─ stop accepts
  ├─ cancel active handles
  ├─ wait connection tasks
  └─ cancel stuck network tasks after timeout
  ↓
CLOSED
```

When deeper execution is unhealthy while the internal network state is READY, public `lifecycle_state` reports `unhealthy` and readiness is false.

### Resource ownership

Two construction modes are explicit:

```text
engine.create_server(...)
    → NetworkServer owns network resources only

create_server(...)
    → constructs v1.3 engine
    → NetworkServer owns network + engine
```

Closing the first leaves the external engine/workers alive. Closing the second calls engine `aclose()` and reaps workers/temporary artifact state.

### Network observability

`NetworkSnapshot` exposes read-only copied facts:

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

No live socket/Worker/KV ownership object is exposed.

## Architecture

```text
NETWORK CLIENT
      ↓
NetworkServer
      ↓ validated GenerateRequest
WhitespaceTokenizer
      ↓ token IDs
AsyncEngine
      ↓
EngineCore / PrefixAwareBudgetedScheduler
      ↓
ArtifactMultiprocessExecutor
      ↓
validated child Worker + model + KV
      ↓ acknowledged execution
EngineCore logical commit
      ↓
AsyncRequestHandle committed events
      ↓
JSON / SSE response
```

All v1.2 artifact identity, process isolation, rollback, scheduling, KV, speculation and observability contracts remain underneath this new edge.

## Files added

```text
mini_vllm/network/__init__.py
mini_vllm/network/types.py
mini_vllm/network/server.py
mini_vllm/versions/v1_3/__init__.py
mini_vllm/versions/v1_3/engine.py
tests/test_network_server.py
examples/18_network_server.py
```

## Existing files extended

```text
mini_vllm/runtime.py              NetworkConfig + RuntimeConfig.network
mini_vllm/observability.py        NetworkSnapshot
mini_vllm/versions/__init__.py    v1.3 default/routing
mini_vllm/__init__.py             public network API
pyproject.toml                    1.3.0
scripts/check.sh                  example 18 release gate
tests/test_model_artifacts.py     explicit historical v1.2 default pin
README.md / Agents.md / Plan.md / docs/* / version.md
```

## Critical invariants added

### Parse before submit

```text
invalid HTTP/API input
    ⇒ no AsyncEngine/Core request exists
```

### Capacity separation

```text
network ingress slot
!= async event queue
!= Scheduler admission/budget
!= KV reservation
```

### Committed-only visibility

```text
network token bytes
    require
committed TokenEvent
```

### Disconnect semantics

```text
socket disconnect
→ cancellation intent
→ cooperative runtime cleanup
```

not worker death and not direct physical mutation.

### Readiness truth

```text
listener bound != ready
```

Readiness depends on the deeper artifact-bound executor being healthy and rank identities agreeing.

### Ownership-aware shutdown

A wrapper closes only the resources it owns. An owned server leaves no worker process alive after successful close.

### Semantic equivalence

Equivalent v1.3 HTTP generation produces the same committed token IDs as direct v1.2 serving.

## Tests

`tests/test_network_server.py` adds real loopback TCP coverage for:

1. health/readiness lifecycle;
2. read-only/equal network snapshots;
3. external-engine ownership preservation;
4. JSON output equivalence with direct v1.2;
5. owned server worker cleanup;
6. SSE committed-token ordering + one done event;
7. malformed JSON rejection;
8. unknown-field rejection;
9. content-type rejection;
10. request-body byte limit;
11. rejection-before-runtime proof;
12. duplicate request ID → HTTP 409;
13. ingress `max_inflight_requests` → HTTP 429 without hidden request;
14. streaming client disconnect → cooperative cancellation;
15. disconnect cancellation accounting;
16. header byte limit → HTTP 431.

Existing v0.1–v1.2 regressions continue to run.

The first implementation CI exposed one governance-only failure: a historical v1.2 test still called bare `create_engine()` while asserting `engine.version == "v1.2"`. Since the global default correctly moved to v1.3, that historical test was pinned explicitly to `create_engine("v1.2", ...)`; no runtime mechanism changed.

## Example 18

`examples/18_network_server.py` demonstrates:

```text
construct owned v1.3 server
→ bind 127.0.0.1:0
→ GET /health
→ GET /ready
→ JSON generation
→ SSE generation
→ inspect NetworkSnapshot
→ inspect artifact RuntimeSnapshot
→ close
→ prove worker_alive == False
```

The client is also implemented with standard-library asyncio TCP primitives so the walkthrough adds no web-client dependency.

## Validation evidence

After the historical test pin, repository GitHub Actions ran:

```text
109 passed, 1 environment warning
examples 01–18
all checks passed
```

The warning is the existing PyTorch notice that NumPy is not installed; the v1.3 reference path does not require NumPy.

The frozen documentation tree and cleaned two-commit release SHA must independently pass the same gate before `main` advances.

## Non-goals

- production HTTP parser/compliance;
- HTTP/2 or HTTP/3;
- TLS/certificate management;
- authentication/API keys;
- tenant/account quotas;
- OpenAI API compatibility;
- chat templates/tool calls;
- WebSocket transport;
- reverse proxy functionality;
- multi-node load balancing/routing;
- production metrics exporters;
- new Scheduler/KV/model algorithms.

## Learning outcome

A reader should be able to explain why exposing an LLM runtime over a socket creates a new ownership boundary: bytes require bounded parsing and API validation, edge ingress capacity must remain separate from model scheduling/storage, disconnect is cancellation intent rather than process death, readiness depends on deeper runtime truth, and only committed runtime events may become externally visible response bytes.

## Next direction

A suitable v1.4 target is **multi-model registry + model lifecycle/routing**: load multiple validated model identities, route each request to exactly one model, pin that identity for the request lifetime, and teach load/readiness/drain/unload without cross-model request/KV leakage. Merely adding more HTTP protocol features is not a sufficiently distinct next lesson.

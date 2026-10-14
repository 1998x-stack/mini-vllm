# mini-vLLM — Architecture-Driven Development Plan

`main` is always the newest exact-CI-validated runtime. Version branches are educational snapshots. Exact release commits must pass repository CI.

## Version ladder

```text
v0.1  static batching + contiguous KV
v0.2  continuous batching
v0.3  paged KV
v0.4  token budget + chunked prefill
v0.5  prefix cache + shared ownership
v0.6  preemption + recompute
v0.7  speculative transactions
v0.8  EngineCore / Executor / Worker
v0.9  WorkerGroup + reference tensor parallel
v0.10 async streaming/cancellation/backpressure
v0.11 compile-aware execution
v0.12 optimized PagedAttention
v0.13 prefix-aware scheduling
v1.0  integrated educational runtime
v1.1  real multiprocessing transport
v1.2  model artifact + tokenizer loading
v1.3  network serving / API boundary                    current
```

## v1.3 learning target

v1.2 proved that external model files are not runtime truth. v1.3 asks the next outer-systems question: what must happen before untrusted network bytes may become serving work?

```text
TCP/HTTP bytes
    ↓ bounded parse
validated GenerateRequest
    ↓ tokenize
core Request(token IDs)
    ↓
existing v1.2 runtime
    ↓ committed StreamEvent
JSON / SSE bytes
```

This version is about network ownership, validation, liveness/readiness, ingress backpressure and disconnect lifecycle. It adds no new model, KV, scheduling or speculative algorithm.

## Implemented mechanism

### 1. Inspectable standard-library HTTP boundary

`NetworkServer` uses `asyncio.start_server` and intentionally supports a small HTTP/1.1 subset:

```text
one request / connection
Content-Length request body
application/json generation input
Connection: close
JSON terminal response
SSE streaming response
```

The server rejects unsupported/malformed protocol input explicitly rather than delegating semantics to a framework.

### 2. API validation before runtime submission

The parser produces an immutable `GenerateRequest`. Method/path/content type, JSON shape, exact field set, prompt, token limit and sampling fields are validated before tokenization/submission.

Header and body byte limits are enforced before model work exists.

### 3. Token-native core preserved

```text
GenerateRequest(prompt text)
    ↓ WhitespaceTokenizer
Request(prompt_token_ids)
    ↓
AsyncEngine / EngineCore
```

Scheduler, WorkItem, KV and model-runner contracts remain token-ID native.

### 4. Dedicated ingress capacity

`NetworkConfig.max_inflight_requests` bounds requests that crossed validation and currently own a network API lifecycle slot.

```text
ingress full → HTTP 429 → no engine.submit → no hidden core Request
```

This capacity is deliberately separate from AsyncEngine queue backpressure, Scheduler budgets and KV allocation.

### 5. Committed-only streaming

The SSE writer consumes the ordinary `AsyncRequestHandle` stream:

```text
TokenEvent      → event: token
FinishedEvent   → event: done
CancelledEvent  → event: cancelled
ErrorEvent      → event: error
```

Therefore the network layer has no path to speculative/uncommitted model state.

### 6. Disconnect cancellation

A connection watcher treats client EOF/reset as cancellation intent and calls `AsyncRequestHandle.cancel()`. Existing cooperative cancellation performs Scheduler/cache cleanup at an atomic EngineCore boundary.

### 7. Health/readiness

`/health` reports network lifecycle liveness. `/ready` additionally requires:

- NetworkServer in READY state;
- underlying execution healthy;
- validated model-artifact snapshot present;
- rank artifact identity agreement.

### 8. Explicit ownership-aware shutdown

Two modes exist:

```text
engine.create_server()     → NetworkServer owns network only
create_server(...)         → NetworkServer owns constructed engine too
```

`close()` stops accepts, cancels/drains active API requests, closes connection tasks, and closes/reaps worker processes only in owned-engine mode.

### 9. Read-only network observability

`NetworkSnapshot` reports lifecycle, bound address, health/readiness, active connections, in-flight requests, acceptance/rejection and terminal counters. It contains no socket/Worker/KV ownership handles.

## Public endpoints

```text
GET  /health
GET  /ready
POST /v1/generate
POST /v1/generate/stream
```

Generation JSON supports `request_id`, `prompt`, `max_new_tokens`, `temperature`, `top_k` and `top_p`.

## Correctness contract

In addition to all v1.2 invariants:

1. malformed API input creates no runtime request;
2. network ingress capacity is not Scheduler or KV capacity;
3. network-visible tokens come only from committed async events;
4. disconnect becomes cancellation intent, never direct KV/process mutation;
5. one submitted API request has one logical terminal outcome;
6. socket liveness is weaker than serving readiness;
7. a server closes only resources it owns;
8. HTTP generation is deterministic and equivalent to direct v1.2 serving under equivalent configuration.

## Validation

`tests/test_network_server.py` uses real loopback TCP connections and proves:

- health/readiness and read-only snapshot behavior;
- non-stream JSON output equivalence with direct v1.2;
- SSE token ordering and terminal event;
- invalid JSON/fields/content-type/body limits reject before submission;
- duplicate ID maps to HTTP 409;
- bounded ingress returns HTTP 429 with one real runtime request only;
- streaming disconnect becomes cooperative cancellation;
- header limits return HTTP 431;
- external-engine server close preserves the external engine;
- owned server close reaps worker processes.

Historical artifact tests explicitly pin v1.2 when testing the old default-builder lesson.

The first corrected implementation tree passed:

```text
109 pytest tests
examples 01–18
all checks passed
```

The single warning is PyTorch noting NumPy is absent; no v1.3 network path depends on NumPy.

## Release rule

Before v1.3 freezes:

1. implementation/docs tree passes repository CI;
2. branch history is reduced to the v1.3 planning commit plus one implementation commit;
3. that exact cleaned SHA independently passes CI;
4. `main` fast-forwards to exactly that SHA;
5. v1.3 becomes historical/frozen.

## After v1.3

The next version must teach a distinct systems boundary rather than “more HTTP features.” A strong v1.4 direction is **multi-model registry + model lifecycle/routing**: model identity becomes a routable resource with load/readiness/drain/unload semantics, while requests remain bound to exactly one validated model version for their complete lifetime.

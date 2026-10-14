# mini-vLLM

A small, branch-versioned LLM-serving runtime for learning the systems ideas behind vLLM.

## Current line

`main` is always the newest exact-CI-validated release. v1.3 adds the outer serving boundary: bounded HTTP bytes become a validated API request, then tokenizer output, then the existing token-native runtime. Only committed runtime events may become network bytes.

```text
v0.1  static batching + contiguous KV
v0.2  continuous batching
v0.3  paged KV + block tables
v0.4  token budget + chunked prefill
v0.5  prefix caching + shared KV
v0.6  preemption + recomputation
v0.7  speculative decoding + transactional KV
v0.8  EngineCore / Executor / Worker
v0.9  logical multi-rank + tensor parallel
v0.10 async streaming + cancellation/backpressure
v0.11 compile-aware rank-local tensor execution
v0.12 reference + vectorized PagedAttention
v0.13 prefix-aware scheduling + cache-aware admission
v1.0  integrated educational runtime
v1.1  real multiprocessing executor transport
v1.2  model artifact + tokenizer loading
v1.3  network serving + API boundary               ← current
```

Historical builders remain available so each mechanism can be studied independently.

## v1.3 mental model

```text
TCP connection
      ↓
HTTP/1.1 bytes
      ↓ bounded parse / validation
GenerateRequest
      ↓ tokenizer
core Request(token IDs)
      ↓
AsyncEngine → EngineCore → Scheduler
      ↓
ArtifactMultiprocessExecutor
      ↓
child Worker / model / KV
      ↓
committed AsyncRequestHandle events
      ↓
JSON response or SSE stream
```

The central separation is:

```text
network ingress permission
        !=
AsyncEngine delivery capacity
        !=
Scheduler admission
        !=
KV capacity / ownership
```

A network slot only permits a validated request to enter the Python serving facade. It does not reserve Scheduler capacity, tokens or KV blocks.

## Reference HTTP API

v1.3 intentionally uses `asyncio.start_server` instead of a web framework so the protocol boundary stays visible. The educational subset supports one request per connection, `Content-Length`, JSON bodies and `Connection: close`.

```text
GET  /health
GET  /ready
POST /v1/generate
POST /v1/generate/stream
```

Example request:

```json
{
  "request_id": "demo",
  "prompt": "tok1 tok2",
  "max_new_tokens": 4,
  "temperature": 0.0
}
```

`/v1/generate` returns one JSON terminal result. `/v1/generate/stream` sends Server-Sent Events named `token`, then exactly one terminal event (`done`, `cancelled` or `error`) while the connection remains writable.

The server rejects malformed request lines/headers, unsupported transfer encoding, oversized headers/bodies, invalid JSON, unknown fields, invalid sampling parameters and duplicate request IDs before hidden model work can be created.

## Health versus readiness

These are intentionally different truths:

```text
health = managed network server is alive
ready  = health
       + server accepts new work
       + underlying executor is healthy
       + model artifact identities agree across ranks
```

A listening socket by itself is not proof that serving is ready. During draining the server can remain alive while readiness is false.

## Disconnect cancellation

A submitted request remains owned by the normal async/runtime lifecycle. Client disconnect is translated into cancellation intent:

```text
socket EOF/reset
     ↓
AsyncRequestHandle.cancel()
     ↓
existing cooperative cancellation boundary
     ↓
EngineCore releases logical/physical state safely
```

The network layer never kills a worker process to cancel one request.

## Network backpressure

`NetworkConfig.max_inflight_requests` bounds API-edge in-flight requests. When full, a valid new request receives HTTP 429 before `engine.submit(...)`; no hidden runtime request is created.

This is independent from:

- `AsyncEngine.max_queue_size`, which bounds per-request event delivery;
- Scheduler sequence/token budgets;
- paged-KV physical capacity.

## Construction and ownership

The default engine is v1.3:

```python
from mini_vllm import create_engine

engine = create_engine()
server = engine.create_server()   # server does not own engine
await server.start()
...
await server.close()
await engine.aclose()
```

Or construct an owned server:

```python
from mini_vllm import NetworkConfig, create_server

server = create_server(network_config=NetworkConfig(port=0))
await server.start()
...
await server.close()  # also closes/reaps the engine it created
```

`RuntimeConfig.network` carries network knobs alongside the already-stable model/cache/multiprocess configuration.

## Observability

`server.snapshot()` returns only copied logical facts:

```text
lifecycle state
bound host / port
health / ready
active connections
in-flight requests
accepted / rejected requests
completed / cancelled / failed requests
disconnect cancellations
```

It exposes no live socket, Worker, block table or parameter-storage handle.

The existing `engine.snapshot()` remains the deeper runtime view for model identity, Scheduler, cache, process transport, attention and compile observations.

## Protocol scope

v1.3 is deliberately not a production HTTP server. It does not teach HTTP/2, TLS, chunked request bodies, authentication, tenant rate limits, OpenAI API compatibility, reverse proxying or multi-node routing. Those concerns are kept separate from the API-edge ownership lesson.

## Walkthrough

```bash
pip install -e '.[dev]'
python examples/18_network_server.py
```

The example binds `127.0.0.1:0`, checks health/readiness, performs JSON and SSE generation, inspects network/runtime snapshots and proves owned worker cleanup.

## Verify

```bash
./scripts/check.sh
```

The v1.3 implementation gate runs all historical regressions plus examples 01–18. The validated implementation tree reached:

```text
109 passed, 1 environment warning
examples 01–18
all checks passed
```

The warning is PyTorch noting that NumPy is not installed; the reference runtime does not require NumPy for this path.

Read `Agents.md`, `Plan.md`, `docs/ARCHITECTURE.md`, `docs/INVARIANTS.md`, and `version.md` before changing architecture.

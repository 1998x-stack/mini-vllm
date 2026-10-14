# Agents.md — mini-vLLM Development Contract

Read this before every code, test, branch, or architecture change.

## Mission

Build serving-system understanding through small, explicit mechanisms. Prefer correctness, readable references, ownership boundaries, inspectable traces, invariant tests and reference equivalence before optimization.

## Branch model

`main` is always the newest exact-CI-validated runtime. Historical version branches are frozen learning snapshots.

```text
v0.1  static batching
v0.2  continuous batching
v0.3  paged KV
v0.4  token budget / chunked prefill
v0.5  prefix sharing
v0.6  preemption / recompute
v0.7  speculative transactions
v0.8  execution boundary
v0.9  logical ranks / tensor parallel
v0.10 async streaming / cancellation
v0.11 compile-aware tensor execution
v0.12 optimized PagedAttention
v0.13 prefix-aware scheduling
v1.0  integrated educational runtime
v1.1  real multiprocessing executor transport
v1.2  model artifact + tokenizer loading
v1.3  network serving / API boundary
```

v1.3 adds one distinct lesson: untrusted network bytes and connection lifetime become a validated, bounded ingress boundary around the already-correct v1.2 runtime.

## Ownership

- `NetworkServer`: TCP/HTTP parsing, API validation, ingress slot accounting, connection lifecycle, health/readiness, response serialization and disconnect-to-cancel intent.
- `AsyncEngine`: caller/request handles, bounded event delivery, cancellation intent and cooperative backpressure.
- `EngineCore`: logical request truth, atomic step boundaries, reconciliation and commit authority.
- `Scheduler`: admission order, token budget, fairness and preemption-victim policy.
- `ArtifactLoader` / model factory: external artifact validation, identity and explicit copies into runtime-owned model state.
- `ArtifactMultiprocessExecutor`: parent-side IPC orchestration, worker health, rank identity agreement, rollback and shutdown.
- child `Worker/ModelRunner`: rank-local model execution and physical KV ownership.
- KV backend: block tables, allocation, sharing, recycling and transactions.
- observability: copied read-only facts only; never become a policy owner.

The network layer must not inspect or control block IDs, refcounts, Worker objects, model tensors or Scheduler internals.

## Capacity separation

Never collapse these distinct resources:

```text
network ingress slot
!= per-request async event queue
!= Scheduler sequence/token budget
!= physical KV capacity
```

HTTP 429 means only that network ingress is full. It does not imply the Scheduler or KV allocator rejected a request.

## Network boundary rule

```text
untrusted HTTP bytes
      ↓ bounded parse + validation
GenerateRequest
      ↓ tokenizer
core Request(token IDs)
      ↓ normal runtime lifecycle
committed StreamEvent
      ↓ serialization
network bytes
```

Malformed/invalid input fails before submission. Network serialization may delay output but cannot manufacture or reveal speculative/uncommitted tokens.

## Disconnect rule

```text
client disconnect
      ↓
AsyncRequestHandle.cancel()
      ↓
cooperative EngineCore cancellation
```

Do not terminate a healthy worker to cancel one disconnected request. Cancellation is applied only through the existing request lifecycle and atomic-step contract.

## Health and readiness

Keep liveness separate from readiness:

- health: the network server is in a managed live/draining state;
- readiness: the server is accepting new requests and the underlying execution runtime is healthy with artifact rank identity agreement.

A bound socket alone is never readiness evidence.

## Stable invariants

The canonical expanded list is `docs/INVARIANTS.md`. Every change must preserve at least:

- token/KV invariants from v0.x;
- logical commit authority remains in EngineCore;
- async clients and network clients see committed tokens only;
- external model files become validated runtime-owned state before readiness;
- child rank identities agree before serving;
- network parsing is bounded before submission;
- invalid API input creates no hidden runtime request;
- ingress capacity is not Scheduler/KV ownership;
- disconnect produces cancellation intent, not direct physical mutation;
- a submitted API request has one logical terminal outcome;
- health and readiness are separate truths;
- an owned server reaps the engine/workers it created;
- a server wrapping an external engine must not close that engine;
- snapshot reads remain side-effect free.

## Public v1.3 surface

Prefer teaching through:

```text
RuntimeConfig / NetworkConfig
create_engine()          # defaults to v1.3
create_server()
engine.create_server()
NetworkServer.start() / close()
server.snapshot()
GET /health
GET /ready
POST /v1/generate
POST /v1/generate/stream
```

Historical `create_engine("v0.x")`, `create_engine("v1.0")`, `create_engine("v1.1")` and `create_engine("v1.2")` remain explicit references.

## Reference protocol scope

The v1.3 HTTP server is deliberately educational:

- HTTP/1.1 only;
- one request per connection;
- `Content-Length` bodies;
- JSON generation requests;
- JSON terminal response or SSE stream;
- `Connection: close`.

Do not quietly broaden this into a production HTTP stack. HTTP/2, TLS, auth, tenant rate limits, WebSocket, proxying and OpenAI compatibility require separate contracts.

## Change protocol

Before changing `main` or the active version:

1. read `Agents.md`;
2. read `Plan.md`;
3. read `version.md`;
4. read `docs/ARCHITECTURE.md` and `docs/INVARIANTS.md`;
5. inspect relevant source/tests;
6. separate network protocol concerns from Scheduler/KV/model policy;
7. add invariant tests that prove rejection-before-submit, committed-only visibility, cancellation and cleanup;
8. run the exact release gate before advancing `main`.

Historical branches are frozen except narrowly scoped documentation/test corrections or true bugs within that historical lesson.

## Testing / release

```bash
python -m pytest -q
./scripts/check.sh
```

Never claim tests passed unless executed on the exact reported snapshot. An active version freezes only after:

1. its exact commit passes repository GitHub Actions;
2. docs agree with code;
3. `main` advances to that same commit;
4. branch history remains pedagogically clean.

For v1.3 network changes, tests must include real TCP parsing, health/readiness, JSON/SSE ordering, ingress rejection, client disconnect cancellation, ownership-aware shutdown and deterministic v1.2 equivalence.

## Performance / policy rule

```text
readable mechanism fact
        ↓
explicit policy input
        ↓
invariant tests
        ↓
observable decision
        ↓
authoritative execution truth
```

Do not infer throughput improvement merely from exposing HTTP, multiprocessing, compilation or vectorization. v1.3 is an API/lifecycle boundary lesson, not a production web-server performance claim.

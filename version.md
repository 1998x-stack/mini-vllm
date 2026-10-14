# mini-vLLM v1.4 — multi-model registry + model lifecycle/routing (planning branch)

## Version identity

- Branch: `v1.4`
- Base: validated/frozen `v1.3` / `main`
- Previous: `v1.3`
- Theme: validated model identities become named routable runtime resources with explicit load/readiness/drain/unload lifecycle
- Status: planning; runtime remains exactly v1.3 until v1.4 acceptance criteria pass

## Why v1.4

v1.3 completed the outer request boundary:

```text
network bytes
→ validated API request
→ tokenizer
→ one artifact-bound engine
→ committed response bytes
```

The next distinct systems problem is not more HTTP. Real serving systems often host more than one model/version. Once multiple validated artifacts can coexist, model identity becomes a resource that must be routed, pinned, drained and released safely.

The new lesson is:

```text
validated artifact identities
        ↓
ModelRegistry
        ↓ lifecycle
LOADING → READY → DRAINING → UNLOADED / FAILED
        ↓ routing
request chooses exactly one model
        ↓ pin
that model identity cannot change mid-request
        ↓
existing v1.3 network + async + execution stack
```

## Primary mental model

```text
MODEL REGISTRY
├─ alias: "blue"
│    └─ artifact identity A
├─ alias: "green"
│    └─ artifact identity B
└─ default route: "blue"
        │
        ▼
validated API request
├─ request_id
├─ model = "green" (optional)
└─ generation params
        │
        ▼
route lookup
        │
        ├─ missing / not ready → reject before submit
        ▼
pin ModelLease(identity B)
        │
        ▼
B-owned tokenizer + engine + workers/KV
        │
        ▼
request terminal outcome
        │
        ▼
release lease
```

A route/alias is not the model itself. An artifact identity is not a mutable registry slot. An in-flight request holds a stable lease to one loaded runtime instance.

## Required delta

### 1. Explicit model registry

Introduce a parent-side `ModelRegistry` (or equivalent) that owns named loaded model runtimes. Preferred logical state per entry:

```text
model_key / alias
artifact identity
lifecycle state
generation/readiness generation number
active request leases
owned engine/runtime
last failure (observable copy only)
```

Registry operations should be explicit rather than magical filesystem discovery.

### 2. Model lifecycle state machine

Preferred reference lifecycle:

```text
UNLOADED
   ↓ load
LOADING
   ├─ validation/startup failure → FAILED
   └─ all ranks ready           → READY
READY
   ↓ drain
DRAINING
   ├─ active leases > 0 → wait/reject new work
   └─ active leases = 0 → UNLOADING
UNLOADING
   ↓ workers/artifact resources reaped
UNLOADED
```

A failed model must not make unrelated ready models unusable unless the registry itself becomes unhealthy.

### 3. Stable request-to-model pinning

Routing happens before runtime submission.

```text
API request.model
    ↓ resolve alias
READY registry entry
    ↓ acquire lease
stable model identity/runtime generation
    ↓ submit
```

Changing an alias while requests are running must not move those requests to another model. The lease owns the resolved runtime generation until the request reaches one logical terminal outcome.

### 4. Default model routing

Support a configured default route for requests without an explicit model field.

The default is a routing policy fact, not an artifact identity. Changing the default affects only future requests.

### 5. Network API extension

Extend v1.3 generation request with optional:

```json
{
  "model": "blue",
  "prompt": "tok1 tok2",
  "max_new_tokens": 4
}
```

Response/stream terminal metadata should expose the resolved logical model key and immutable artifact identity (or a short safe representation) for provenance.

Do not push route strings into Scheduler/KV/WorkItem internals. Routing must choose an engine before the token-native request enters that engine.

### 6. Load contract

Loading a model must reuse v1.2 artifact validation and v1.1/v1.2 child readiness semantics:

```text
registry.load(key, artifact_path)
        ↓
parent artifact validation
        ↓
child independent validation/copy
        ↓
all rank identities agree
        ↓
entry becomes READY atomically
```

A partially started model never becomes routable.

### 7. Drain contract

Draining means:

```text
new lease acquisition forbidden
existing leases continue normally
        ↓
active lease count reaches zero
        ↓
owned engine may close/unload
```

Do not implement drain by cancelling healthy in-flight requests by default. A separate explicit force-unload policy may be taught later or exposed only as a narrow opt-in with clear semantics.

### 8. Unload contract

Normal unload requires no active leases. It closes the model-owned engine/workers and releases temporary/artifact runtime resources exactly once.

```text
active leases != 0
→ normal unload rejected / waits according to explicit API
```

No request may observe a runtime whose workers were reaped underneath it.

### 9. Alias replacement / version switch

A useful reference operation is atomic future-routing replacement:

```text
load B until READY
        ↓
atomically route alias "prod" → B
        ↓
old A becomes optionally DRAINING
```

Existing A leases stay pinned to A. New requests resolve to B.

This teaches zero-cross-request identity mixing without implementing production deployment orchestration.

### 10. Registry concurrency / synchronization

Load, route, drain, alias swap and unload can race with network requests. Use one explicit async synchronization boundary (lock/state machine) around registry metadata changes.

Never hold the registry metadata lock across long model inference. Lease acquisition should be short; request execution proceeds outside the registry lock.

### 11. Registry observability

Add read-only snapshots such as:

```text
registry lifecycle / default route
per-model key
artifact identity
model state
ready
runtime generation
active leases
accepted/completed/cancelled/failed request counts
load/unload counts
last load error type/message (copied string facts)
```

Do not expose live engine, process, parameter, socket or KV handles.

### 12. Failure isolation

Required teaching behavior:

```text
model A load fails
    ↓
A = FAILED
B remains READY and routable
```

Likewise, if A's executor becomes unhealthy, readiness/routing for A must fail without falsely marking healthy B unavailable.

## Critical invariants

### Route-before-submit

A request cannot be submitted until its model key resolves to one READY runtime generation.

### Stable identity pin

```text
resolved model identity at lease acquisition
== model identity for every token/event of that request
```

Alias/default changes affect future requests only.

### Drain safety

DRAINING rejects new leases but does not silently cancel existing healthy leases.

### Unload safety

A model runtime cannot be normally destroyed while active leases remain.

### No cross-model ownership

Request IDs, Scheduler state, cache blocks and worker state belong to one engine/model runtime. Model A cannot read/release/mutate model B physical state.

### Tokenizer binding

The tokenizer used for a request is the tokenizer of the same pinned model artifact. Route switching cannot tokenize with A and execute on B.

### Failure isolation

Failure/unhealthiness of one model entry does not change the readiness of unrelated healthy entries.

### Default-route truth

The default model is an alias/routing choice, not a hidden global engine identity.

### Exact terminal lease release

Every successfully acquired request lease is released exactly once after finished/cancelled/error lifecycle convergence.

### Historical boundaries preserved

All v1.3 invariants remain active inside each model runtime: bounded network parsing, ingress capacity, committed-only streaming, cooperative cancellation, artifact identity, multiprocessing acknowledgment and KV ownership.

## Preferred modules

Possible readable layout:

```text
mini_vllm/registry/
├─ __init__.py
├─ types.py
└─ registry.py

mini_vllm/network/
└─ server.py          # route before per-model submit

mini_vllm/versions/v1_4/
├─ __init__.py
└─ engine.py
```

Prefer a thin `MultiModelServer`/facade over teaching every lower-level engine that models are globally shared.

## Tests required

### Loading / readiness

- load two deterministic artifact models under different keys;
- both independently reach READY;
- artifact/rank identity is correct per entry;
- malformed model load becomes FAILED before routing;
- failure of A leaves B routable.

### Routing

- explicit model key routes to correct engine/output;
- missing model uses default route;
- unknown/not-ready/draining model rejects before runtime submission;
- response/SSE provenance reports resolved model identity.

### Pinning / alias swap

- start a long request on alias `prod → A`;
- load B and atomically change `prod → B`;
- first request finishes entirely on A;
- later request runs entirely on B;
- outputs match direct per-model references.

### Drain / unload

- DRAINING rejects new leases;
- existing request completes without forced cancellation;
- normal unload cannot destroy active leased runtime;
- once leases reach zero, unload reaps workers/resources;
- repeated unload does not double-release.

### Cancellation / disconnect

- network disconnect cancels only the pinned model request;
- lease is released exactly once after cancellation;
- other model entries remain unaffected.

### Isolation

- duplicate request IDs may exist on different model engines if the public registry contract namespaces them explicitly, or are globally rejected if global uniqueness is chosen; behavior must be explicit and tested;
- cache/scheduler snapshots for one model never contain the other model's requests;
- unhealthy A does not make `/ready` semantics for B ambiguous—define aggregate/global readiness explicitly.

### Regression

All v0.1–v1.3 tests and examples 01–18 remain green.

## Integrated example target

Add `examples/19_multi_model_registry.py`:

```text
export model A and B
→ create v1.4 registry/server
→ load A as blue
→ load B as green
→ set default blue
→ request default / explicit green
→ start long blue request
→ atomically switch default to green
→ prove old request pinned to blue
→ drain/unload blue after lease release
→ snapshot registry
→ close and prove all workers reaped
```

## Acceptance criteria

- at least two validated model runtimes can coexist under explicit registry keys;
- each entry has an observable lifecycle state;
- load failure cannot make an invalid model routable;
- API requests resolve a READY model before submission;
- default and explicit model routing are deterministic;
- each request is pinned to one runtime generation/artifact/tokenizer for its full lifetime;
- alias/default replacement affects future requests only;
- DRAINING rejects new work but preserves existing healthy requests;
- normal unload occurs only after active leases reach zero;
- unload closes/reaps only that model's owned resources;
- one model's failure/unhealthiness is isolated from unrelated ready models;
- registry snapshot is read-only and exposes no physical handles;
- v1.3 network/KV/process/artifact invariants remain intact per model;
- examples 01–18 remain green plus a v1.4 multi-model lifecycle walkthrough;
- exact cleaned release SHA passes repository CI before `main` advances.

## Non-goals

- production autoscaling;
- GPU memory packing across real devices;
- tensor-parallel placement planner;
- model download registry / Hugging Face Hub;
- LRU model eviction under memory pressure;
- quantization/LoRA adapter multiplexing;
- production deployment controller;
- canary percentage traffic splitting;
- tenant auth/billing/rate limiting;
- multi-node distributed routing;
- cross-model shared KV;
- request migration between models;
- force-kill hot swap as the normal deployment path.

## Learning goal

By v1.4 a reader should be able to explain why multi-model serving is fundamentally a lifecycle/ownership problem: aliases route only future work, every request must acquire a stable lease to one validated runtime generation, draining separates new admission from existing execution, unload waits for ownership to reach zero, and failures stay isolated per model rather than leaking across the registry.

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
v1.2  model artifact + tokenizer loading boundary
```

Post-v1.0 branches require a genuinely new systems boundary. v1.1 introduced real OS-process isolation; v1.2 introduces external-artifact validation and transfer into runtime-owned model/tokenizer state.

## Ownership

- `AsyncEngine`: caller lifecycle, bounded delivery, cancellation intent and backpressure.
- `EngineCore`: logical request truth, atomic step boundaries and commit authority.
- `Scheduler`: admission, token budget, fairness and preemption-victim policy.
- `ArtifactLoader`: parse external local files, validate schema/tensor/tokenizer contracts and compute artifact identity. It owns no serving request state.
- `RuntimeModelFactory`: construct runtime modules and copy accepted tensor values into runtime-owned parameters. It does not decide scheduling or artifact trust policy.
- `WhitespaceTokenizer`: text ↔ token-ID boundary only. Strings must not leak into Scheduler/KV/WorkItem contracts.
- `ArtifactMultiprocessExecutor`: parent-side IPC orchestration plus rank model-identity agreement and process lifecycle.
- child `Worker/ModelRunner`: rank-local model execution and physical KV ownership inside one OS process.
- KV backend: physical storage, allocation, sharing, recycling and transactions inside the child.
- PagedAttention backend: read child-local logical K/V; it does not own blocks or artifacts.
- Compile registry: rank-local tensor specialization only.
- observability: copy logical facts into read-only snapshots; never become a policy or ownership authority.

A filesystem path, parsed JSON object, checkpoint dictionary or tokenizer file is not runtime model ownership. The serving model begins only after validation and explicit parameter copy into a runtime-created module.

## Stable invariants

The canonical expanded contract is `docs/INVARIANTS.md`. Every active-version change must preserve at least:

- computed tokens imply valid, addressable KV;
- sampled outputs are not computed until later model execution;
- physical KV ownership precedes access;
- block-table translation is authoritative;
- prefix hints are advisory, never ownership;
- preemption cannot retain claims to released KV;
- rejected speculative state disappears;
- execution/transport failures cannot partially commit logical state;
- async clients observe committed tokens only and one terminal outcome;
- `snapshot()` is read-only;
- IPC carries logical copies/handles, not live child ownership objects;
- a sent command is not execution success; logical commit requires matching acknowledgment;
- unknown child physical state makes the executor unhealthy;
- successful close/aclose leaves no worker process alive;
- filesystem presence is not proof of a valid model artifact;
- artifact schema/version/model dimensions must validate before worker readiness;
- every required checkpoint tensor must be present exactly once under the explicit mapping contract;
- unexpected tensors, shape mismatches, unsupported dtypes and non-tensor payloads fail explicitly;
- tokenizer vocabulary/special metadata must be compatible with the model artifact;
- runtime parameters are copied into runtime-owned storage rather than aliasing mutable loader containers;
- every worker rank must report the parent-validated artifact/tokenizer identity before serving begins;
- malformed, changed or rank-divergent artifacts fail before request admission;
- deterministic v1.2 loading of the exported v1.1 model preserves committed output.

## Public v1.2 surface

Prefer teaching through:

```text
RuntimeConfig(model_artifact=...)
MultiprocessConfig
create_engine()                  # defaults to v1.2
create_engine("v1.2", model_artifact=...)
export_tiny_model_artifact(...)
load_model_artifact(...)
engine.encode() / engine.decode()
engine.submit_text(...)
Request / AsyncRequestHandle
engine.snapshot()
engine.close() / await engine.aclose()
```

Historical builders remain explicit and frozen.

Do not expose raw checkpoint dictionaries, parameter pointers, block IDs, refcounts or live child objects merely for convenience. Add immutable logical metadata instead.

## Artifact-boundary rule

```text
external files
    ↓ parse
untrusted structured values
    ↓ exact validation
ValidatedModelArtifact + identity
    ↓ construct + copy
runtime-owned model/tokenizer state
    ↓ all ranks independently agree
READY
    ↓
request admission
```

Never skip a boundary by treating a path, filename, state-dict key or successful `torch.load` as proof of compatibility.

A model artifact that changes between parent validation and child startup must be detected through identity mismatch; do not silently accept the new bytes.

## Process-boundary rule

```text
parent logical truth
        ↓ command
child physical mutation
        ↓ matching acknowledgment
parent logical commit
```

Artifact identity agreement does not replace execution acknowledgment, and process liveness does not prove command completion.

## Change protocol

Before changing `main` or the active version:

1. read `Agents.md`;
2. read `Plan.md`;
3. read `version.md`;
4. read `docs/ARCHITECTURE.md` and `docs/INVARIANTS.md`;
5. inspect implementation and invariant tests;
6. separate parsing/validation from model construction and from serving execution;
7. add direct tests for invalid external inputs and ownership transfer;
8. preserve historical builders;
9. run the exact release gate before advancing `main`.

Historical branches are frozen except narrowly scoped documentation/test corrections or true bugs within that lesson.

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

For v1.2 artifact changes, tests must cover schema/version errors, missing/unexpected tensors, shape/dtype errors, tokenizer compatibility, runtime-owned copies, rank identity agreement, startup failure cleanup, deterministic v1.1 equivalence and examples 01–17.

## Performance / security scope

The educational local loader is not a benchmark or production security sandbox. It deliberately uses `torch.load(..., weights_only=True)` and one known architecture/checkpoint contract. Do not add arbitrary remote code or network downloads to solve a local artifact problem.

Do not claim speedup from multiprocessing, `torch.compile`, vectorization, CUDA or Triton without measurement. v1.2 is a trust/ownership/loading lesson, not a loading-throughput claim.

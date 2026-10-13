# mini-vLLM v0.11 — compile-aware execution

## Version identity

- Branch: `v0.11`
- Previous: `v0.10`
- Theme: explicit shape signatures, rank-local compile caches, and eager fallback inside the tensor execution plane
- Release rule: the exact implementation commit must pass repository CI before synchronization to `main`

## Mental model

```text
AsyncEngine / EngineCore / Scheduler                 eager
                ↓
DistributedExecutor / WorkerGroup                    eager
                ↓
rank-local Worker
                ↓
TinyDecoder trunk + KV read/write                    eager
                ↓
hidden trace [T,H]
                ↓
PureTensorProjectionKernel(hidden, local_weight)     compile-aware
                ↓
local logits [T,V_rank]
                ↓
ReferenceCollective                                  eager
```

v0.11 deliberately chooses a narrow real compile boundary: the rank-local vocabulary projection. It is enough to expose shape specialization and reuse without hiding Python KV ownership inside Dynamo.

## Added abstractions

- `ExecutionShapeSignature(token_count, hidden_dim, output_dim, dtype, device)`
- `CompileConfig(enabled, backend, fullgraph, dynamic, fallback_on_error)`
- `CompileStats`
- `CompileEvent`
- `PureTensorProjectionKernel`
- `CompiledKernelCache`
- `CompileAwareVocabParallelTinyRunner`

## Signature lifecycle

```text
unseen [T,H]×[Vrank,H]
    ↓
cache miss + torch.compile
    ↓
compiled callable stored

same signature
    ↓
cache hit

new T / Vrank / dtype / device
    ↓
new signature + explicit recapture/recompile counter
```

The registry is intentionally separate from PyTorch's internal Dynamo cache. It teaches the serving runtime's own specialization policy.

## Fallback

Compilation is optional optimization state. If compilation or first execution fails and `fallback_on_error=True`, the same pure tensor kernel executes eagerly. No Scheduler, KV, speculative, async, or logical-token state is changed by the fallback decision.

## Distributed ownership

Each logical rank owns a distinct `CompiledKernelCache`. The collective stays outside compiled code. Uneven vocabulary shards therefore naturally have different `output_dim` signatures.

## Invariants

- compiled projection numerically equals eager `F.linear`;
- only tensor data crosses the compile boundary;
- repeated compatible signatures reuse one cached callable;
- materially different shapes create a visible new variant;
- compile failure has explicit eager fallback or strict error mode;
- deterministic v0.11 token streams equal v0.10;
- world-size 1 and 2 preserve target semantics;
- v0.10 async cancellation/backpressure and v0.9 distributed rollback remain unchanged;
- compile policy does not appear in EngineCore, Scheduler, AsyncEngine, or WorkerGroup.

## Validation

`tests/test_compile_runtime.py` covers the pure kernel, deterministic signatures, miss/hit/recapture statistics, and fallback behavior. `tests/test_v011_version.py` covers async v0.10 equivalence, world-size equivalence, and rank-local cache observability. `examples/12_compile_aware.py` prints per-rank signatures and compile modes.

The exact commit promoted to `main` must pass `./scripts/check.sh` in GitHub Actions, including all historical tests and examples 01–12.

## Intentional simplifications

- CI uses `torch.compile(..., backend="eager")` to exercise Dynamo graph formation without claiming speedup;
- Transformer trunk and attention remain eager in v0.11;
- no CUDA Graph capture;
- no dynamic-shape/bucketing policy;
- no compiled collective;
- no custom Triton/CUDA attention kernel.

## Next

`v0.12` adds an optimized PagedAttention path while keeping the readable cache implementation as correctness oracle.

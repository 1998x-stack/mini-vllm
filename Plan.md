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
v1.0  integrated educational runtime                  frozen
v1.1  real multiprocessing executor transport         frozen
v1.2  model artifact + tokenizer loading boundary     current
```

## v1.2 learning target

v1.1 made execution physically real. v1.2 makes model startup explicit: external local files are data to validate, not objects the serving runtime may blindly trust.

```text
model directory
    ↓
ArtifactLoader
    ↓
ValidatedModelArtifact + stable identity
    ↓
RuntimeModelFactory
    ↓ copy
runtime-owned TinyDecoderLM parameters
    ↓
worker readiness / cross-rank identity agreement
    ↓
existing v1.1 serving path
```

This release introduces no new scheduling, KV, speculative, attention or IPC algorithm.

## Implemented mechanism

### Versioned local artifact

The reference format is intentionally small:

```text
model_config.json
weights.pt
tokenizer.json
```

`model_config.json` declares exact format version, architecture, weight dtype, tokenizer compatibility ID and TinyModel dimensions. Unknown/missing fields and unsupported versions fail explicitly.

### Stable identity

`ModelArtifactIdentity` combines logical digests for model metadata, checkpoint contents and tokenizer metadata. A path is never treated as identity.

### Explicit weight contract

`build_weight_mappings()` derives the complete expected runtime `state_dict` surface and records external name, runtime name, shape and dtype. Loading rejects:

```text
missing required name
unexpected name
shape mismatch
dtype mismatch
non-tensor entry
```

No required weight is silently initialized and no extra tensor is silently ignored.

### Runtime-owned copy

`build_model_from_artifact()` constructs a new runtime module and copies validated source tensors into its state under `torch.no_grad()`. Serving does not depend on retaining the loader checkpoint dictionary.

### Readable tokenizer

The v1.2 reference tokenizer is deterministic whitespace tokenization with an explicit vocabulary and `unk` token. The model/tokenizer vocabulary sizes and compatibility IDs must agree before readiness.

The serving core remains token-ID native. `encode`, `decode` and `submit_text` are convenience methods above the core boundary.

### Child-independent model load

Parent construction validates the artifact first and passes a serializable `ModelArtifactSpec` containing path, validated identity and model facts. Every child process independently:

1. re-reads the local files;
2. re-computes identity;
3. rejects changed/incompatible bytes;
4. constructs a runtime model;
5. copies weights;
6. reports artifact/tokenizer identity in the readiness observation.

`ArtifactMultiprocessExecutor` becomes healthy only after all ranks agree with the parent-validated identity. If one rank fails startup, already-created workers are shut down/reaped.

### Default builder without a bypass

When no `model_artifact` path is supplied, v1.2 exports the deterministic teaching model to a temporary artifact directory and then uses the same validation + child reload path. Close/aclose removes that owned temporary directory after workers stop.

## Public API

```python
engine = create_engine("v1.2", model_artifact="./model_dir")
ids = engine.encode("tok1 tok2")
handle = engine.submit_text("r", "tok1 tok2", max_new_tokens=4)
```

`create_engine()` now defaults to v1.2. `RuntimeConfig.model_artifact` provides the same path-based construction surface. Explicit v1.0/v1.1 builders remain historical references.

## Observability

`engine.snapshot()` adds `ModelArtifactSnapshot` and rank-level fields for:

- artifact identity;
- tokenizer identity;
- architecture;
- format version;
- vocabulary size;
- artifact dtype versus runtime dtype/device;
- rank identity agreement.

Observation remains copy/read-only and does not expose checkpoint dictionaries or parameter storage handles.

## Correctness contract

v1.2 adds these release-level rules:

1. filesystem presence is not artifact validity;
2. schema/version validation precedes model construction;
3. tokenizer/model compatibility precedes readiness;
4. required tensor coverage is exact;
5. tensor shape and dtype are validated before copy;
6. runtime parameters own their copied values;
7. parent artifact identity is re-proven independently by every child;
8. changed files between parent validation and child startup fail readiness;
9. a failed child load leaves no orphan processes;
10. no request is admitted before all workers are ready;
11. loaded deterministic v1.2 output equals the exported v1.1 model for world sizes 1 and 2.

All earlier logical/KV/speculation/transport invariants remain active.

## Validation evidence

`tests/test_model_artifacts.py` covers:

- deterministic artifact/tokenizer identity;
- manifest/version/missing-file errors;
- vocabulary and special-token errors;
- missing/unexpected/shape/dtype/non-tensor checkpoint errors;
- runtime-owned parameter copies;
- v1.1↔v1.2 deterministic equivalence at world sizes 1 and 2;
- token-native `submit_text` facade;
- pre-start rank identity disagreement;
- child-side changed-artifact startup failure and cleanup;
- default temporary artifact lifecycle.

Historical multiprocessing tests remain pinned to explicit v1.1 construction.

`examples/17_model_loading.py` performs export → validate → text encode → two-process serving → decode → public identity inspection → clean shutdown.

The first green v1.2 implementation tree passed **101 pytest tests** plus examples 01–17. The final frozen tree and the cleaned release SHA must each independently pass the repository gate.

## Release rule

Before `main` advances:

1. docs and implementation must agree;
2. full frozen tree must pass CI;
3. WIP history is reduced to the existing v1.2 planning commit plus one implementation commit;
4. that exact cleaned implementation SHA must pass CI;
5. `main` fast-forwards to exactly that SHA.

## After v1.2

The next branch must introduce a distinct systems boundary rather than adding more checkpoint formats. A natural next target is a **network serving/API boundary**: request protocol, server lifecycle, admission/backpressure at the socket/API edge, readiness/health endpoints and disconnect semantics. Real CUDA/Triton kernels remain an independent later optimization line.

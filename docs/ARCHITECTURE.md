# mini-vLLM Architecture — v1.2

## Complete startup + serving stack

v1.2 adds a model-ingestion boundary before the validated v1.1 multiprocessing serving stack.

```text
EXTERNAL LOCAL FILES
model_config.json / weights.pt / tokenizer.json
        ↓
ArtifactLoader
├─ exact schema/version validation
├─ model-dimension validation
├─ tokenizer compatibility validation
├─ complete tensor manifest validation
└─ stable artifact identity
        ↓
ValidatedModelArtifact
        ↓
parent creates serializable ModelArtifactSpec
        ↓
ArtifactMultiprocessExecutor
        ├───────────────┬───────────────┐
        ↓               ↓               ↓
child rank 0        child rank 1    child rank N
re-validate files   re-validate     re-validate
build model         build model     build model
copy weights        copy weights    copy weights
build KV/runner     build KV/runner build KV/runner
        ↓               ↓               ↓
ready(identity)     ready(identity)  ready(identity)
        └───────────────┴───────────────┘
                        ↓
             all identities agree
                        ↓
PARENT SERVING STACK
AsyncEngine → EngineCore → Scheduler → IPC execution
                        ↓
               committed async events
```

There are now two independent trust/ownership boundaries:

```text
external artifact truth  → validated/runtime-owned model truth
parent logical truth     ↔ child physical execution truth
```

Neither boundary may be collapsed into the other.

## Artifact ownership

### External files

The model directory is external input. Filesystem existence, successful JSON parsing or successful `torch.load` does not establish compatibility.

### ArtifactLoader

The loader owns parsing and validation only. It produces immutable logical metadata (`ValidatedModelArtifact`, `ModelArtifactSpec`, `ModelArtifactIdentity`) and temporary checkpoint tensor containers used during validation/copy. It does not own a serving model, request or worker.

The first format accepts exactly:

```text
model_config.json
weights.pt
tokenizer.json
```

and one architecture identifier: `tiny-decoder-lm`.

### ModelArtifactIdentity

Identity is derived from:

```text
format version + architecture
model metadata digest
weights content/manifest digest
tokenizer digest
```

A filesystem path is location, not identity. Two ranks may read the same path only after independently proving the expected identity.

## Manifest and tensor validation

`model_config.json` declares the exact TinyModel configuration, artifact weight dtype and tokenizer compatibility ID. Unknown or missing fields are rejected.

`build_weight_mappings()` creates the complete external-name → runtime-name contract from the runtime architecture:

```text
WeightMapping
├─ external_name
├─ runtime_name
├─ expected_shape
└─ expected_dtype
```

For the v1.2 reference format names are intentionally identical, but the mapping remains explicit so file naming and runtime module ownership stay conceptually separate.

Accepted checkpoint state must satisfy set equality:

```text
external tensor names == expected required tensor names
```

Then every tensor must match shape and supported artifact dtype. Missing, unexpected, wrong-shape, wrong-dtype and non-tensor entries fail before runtime construction succeeds.

`torch.load(..., weights_only=True)` is used for the local educational checkpoint. Arbitrary Python checkpoint code is outside the format contract.

## Runtime model factory

The runtime never serves directly from a checkpoint dictionary.

```text
validated source tensor
        ↓
new TinyDecoderLM parameter target
        ↓ target.copy_(source) under no_grad
runtime-owned parameter storage
```

After copying, loader/checkpoint container lifetime is irrelevant to the model's parameter values. Device/runtime dtype conversion happens after the accepted float32 artifact state is copied.

## Tokenizer boundary

`WhitespaceTokenizer` is intentionally readable:

```text
text
 ↓ split on Unicode whitespace
vocabulary lookup
 ↓ unknown → declared unk ID
list[int]
```

Decode validates every ID is in range and joins vocabulary strings with spaces.

Before readiness:

```text
tokenizer.compatibility_id == model.tokenizer_compatibility_id
tokenizer.vocab_size        == model.vocab_size
```

The serving core remains token-native. Strings do not enter Scheduler, KV cache, WorkItem, ModelRunner or speculative contracts.

## Parent validation versus child validation

Parent validation establishes the artifact expected by one engine:

```text
parent load_model_artifact(path)
        ↓
ModelArtifactSpec(path, expected_identity, model_config, dtype, tokenizer ID)
```

Each worker child then independently calls the loader again with `expected_identity`.

This catches a time-of-check/time-of-startup change:

```text
parent validates artifact A
files change to artifact B
child loads B
identity B != expected A
→ startup error
→ no READY
→ no request admission
```

The parent does not serialize a huge state dict through IPC in v1.2. Weight-distribution optimization is deliberately separate from artifact validation.

## Worker readiness

v1.1 readiness meant that the process/worker stack constructed successfully. v1.2 requires model-specific readiness:

```text
process started
+ local artifact readable
+ schema/version valid
+ tokenizer compatible
+ exact weight coverage valid
+ runtime parameter copy complete
+ reported artifact identity == parent expected
+ every rank reports same tokenizer identity
= executor ready
```

If a later rank fails while earlier ranks are ready, `ArtifactMultiprocessExecutor` shuts down/reaps the already-created clients before surfacing startup failure.

## Parent serving ownership

After readiness, v1.2 reuses v1.1 responsibilities unchanged:

- `ArtifactAsyncEngine`: caller handles, stream delivery, cancellation and lifecycle; plus text convenience methods.
- `ArtifactIntegratedEngine`: EngineCore logical truth and public artifact snapshot composition.
- `PrefixAwareBudgetedScheduler`: admission, budget, fairness and preemption policy.
- `ArtifactMultiprocessExecutor`: IPC, correlation, remote transactions, process health and rank identity agreement.
- child Worker/ModelRunner/KV: physical model execution and cache ownership.

Artifact identity is not a scheduling input.

## Process transport

The v1.1 fixed command protocol remains unchanged for serving work. `WorkerSpec` gains an optional serializable `artifact_spec`; historical v1.1 workers leave it `None`.

Rank readiness/execution observations may include:

```text
artifact_identity
tokenizer_identity
artifact_architecture
```

These are logical provenance facts, not parameter-storage handles.

## Default construction without a bypass

`create_engine()` defaults to v1.2. For teaching convenience, if no external path is supplied it:

```text
construct deterministic TinyDecoderLM only for export
        ↓
write temporary v1.2 artifact
        ↓
load_model_artifact(temp_dir)
        ↓
start child ranks from that same artifact spec
```

The serving workers therefore still cross the artifact boundary. The temporary directory is runtime-owned lifecycle state and is removed only after child workers have stopped.

## Read-only observability

`engine.snapshot()` adds:

```text
RuntimeSnapshot.model_artifact
├─ identity
├─ architecture
├─ format_version
├─ tokenizer_identity
├─ tokenizer_compatibility_id
├─ vocab_size
├─ weight_dtype
├─ runtime_dtype
├─ device
└─ rank_identity_agreement

RankExecutionSnapshot
├─ existing v1.1 process/attention/compile facts
├─ artifact_identity
├─ tokenizer_identity
└─ artifact_architecture
```

No raw checkpoint tensors, parameter pointers or mutable loader objects are exposed. Existing snapshot reads remain non-serving observations.

## Request path

Text convenience path:

```text
caller text
  ↓ WhitespaceTokenizer
prompt token IDs
  ↓ normal Request
AsyncEngine
  ↓
EngineCore / Scheduler
  ↓
remote prefix/KV/model work
  ↓ matching acknowledged results
logical commit
  ↓
TokenEvent
```

Direct token-ID requests continue to bypass the tokenizer and use the same core.

## Failure layers

### Artifact validation failure

Malformed/incompatible files fail during parent construction before workers/requests exist.

### Child artifact divergence

If child re-validation disagrees with the parent identity, readiness fails. The child sends a structured startup error and the executor reaps partial startup state.

### Serving child error/transport death

Once ready, v1.1 failure atomicity remains authoritative: remote checkpoints, matching responses, rollback of reachable state, unhealthy executor on unknown physical consistency and no EngineCore logical commit on uncertain execution.

### Caller cancellation

Cancellation remains an EngineCore/Async lifecycle operation. It neither modifies model artifacts nor kills healthy workers.

## Stable equivalence oracles

- exported deterministic v1.1 TinyDecoder weights are the v1.2 model-loading semantic oracle;
- world-size 1 and 2 v1.2 outputs must equal v1.1 under equivalent configuration;
- reference PagedAttention remains the attention oracle;
- eager LM-head projection remains the compile oracle;
- paged block translation remains the KV oracle;
- `docs/INVARIANTS.md` is the complete cross-version contract.

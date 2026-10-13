# mini-vLLM v1.2 — model artifact + tokenizer loading boundary

## Version identity

- Branch: `v1.2`
- Previous: validated/frozen `v1.1`
- Theme: external local model/tokenizer artifacts → validated identity → runtime-owned model/token state
- Package version: `1.2.0`
- Runtime default: `create_engine()` selects v1.2
- Status: implementation complete; release content is frozen by the exact-SHA CI/main synchronization rule

## Why this version exists

v1.1 established a real process boundary:

```text
parent logical truth ↔ IPC evidence ↔ child physical execution
```

v1.2 adds the missing boundary before workers can serve anything:

```text
external files
    ↓ parse / validate
validated artifact identity
    ↓ explicit copy
runtime-owned model + tokenizer state
    ↓ rank agreement
READY
```

The central lesson is that external files are data, not runtime truth.

## Delta from v1.1

### Reference artifact format

```text
model_dir/
├─ model_config.json
├─ weights.pt
└─ tokenizer.json
```

The first format deliberately supports one known architecture (`tiny-decoder-lm`) and float32 artifact weights. It does not attempt arbitrary model code or broad checkpoint compatibility.

### Manifest validation

`mini_vllm/artifacts/loader.py` validates:

- exact artifact format version;
- exact architecture identifier;
- exact top-level/model field sets;
- integer/positive model dimensions through `TinyModelConfig`;
- supported artifact dtype;
- tokenizer compatibility identifier;
- tokenizer format/type/vocabulary/special-token metadata.

Missing/unknown incompatible metadata fails before worker readiness.

### Weight contract

`build_weight_mappings()` derives the complete expected TinyDecoder `state_dict` surface as explicit `WeightMapping` values:

```text
external name
runtime name
expected shape
expected dtype
```

`load_artifact_weights()` then requires exact name-set equality and rejects:

```text
missing required tensor
unexpected tensor
shape mismatch
dtype mismatch
non-tensor checkpoint entry
```

The checkpoint is opened with `torch.load(..., weights_only=True)`.

### Stable artifact identity

`ModelArtifactIdentity` contains:

```text
format_version
architecture
model_digest
weights_digest
tokenizer_digest
combined digest
```

The weights digest is based on sorted tensor names, dtype, shape and tensor bytes. The filesystem path is not identity.

### Runtime-owned model state

`mini_vllm/model/factory.py` constructs a fresh `TinyDecoderLM` and copies every validated source tensor into its runtime target under `torch.no_grad()`.

The resulting model stores logical artifact/tokenizer identity attributes for child observations, but it does not retain the checkpoint dictionary as parameter ownership.

### Tokenizer boundary

`WhitespaceTokenizer` provides a readable deterministic contract:

```text
encode(text): Unicode-whitespace split → vocab IDs → unknown maps to declared unk ID
decode(ids): range-check IDs → vocabulary strings → join with spaces
```

The core remains token-ID native. v1.2 adds only thin public helpers:

```text
engine.encode(text)
engine.decode(token_ids)
engine.submit_text(...)
```

### Multiprocess artifact executor

v1.2 introduces `ArtifactMultiprocessExecutor` without rewriting the validated v1.1 executor.

Parent startup:

```text
load_model_artifact(path)
        ↓
ValidatedModelArtifact
        ↓
ModelArtifactSpec(path + expected identity + model facts)
        ↓
one WorkerSpec per rank
```

Child startup independently performs:

```text
re-read artifact
re-validate schema/tensors/tokenizer
re-compute identity
compare with parent expected identity
construct TinyDecoderLM
copy runtime-owned parameters
construct KV / compile-aware runner / Worker
send READY observation with artifact/tokenizer identity
```

The executor becomes usable only when every rank reports the expected model identity and one tokenizer identity.

If startup fails after earlier ranks started, those clients are shut down/reaped before construction raises.

### Changed-artifact protection

The child receives an expected identity, not permission to trust whatever bytes happen to exist later at the path.

```text
parent validates A
files mutate to B
child re-validates B
B identity != A identity
→ WorkerRemoteError during startup
→ no ready executor
→ no request admission
```

This is a deliberate time-of-check/time-of-startup consistency check.

### Default construction

The v1.2 default does not retain the old direct-model serving path. With no user path it:

```text
export deterministic educational model to temporary artifact
        ↓
parent validate
        ↓
children independently validate/load
        ↓
serve
```

`close()`/`aclose()` removes the owned temporary directory only after workers stop.

### Observability

`ModelArtifactSnapshot` exposes:

- artifact identity;
- architecture;
- format version;
- tokenizer identity/compatibility ID;
- vocabulary size;
- artifact weight dtype;
- runtime dtype/device;
- rank identity agreement.

`RankExecutionSnapshot` also carries rank artifact/tokenizer/architecture identity facts. No raw checkpoint state or parameter/storage handle is exposed.

## Architecture

```text
EXTERNAL ARTIFACT
        ↓
ArtifactLoader
        ↓
ValidatedModelArtifact
        ↓
ArtifactAsyncEngine / ArtifactIntegratedEngine
        ↓
ArtifactMultiprocessExecutor
        ├─ child rank 0: validate → copy model → Worker/KV
        ├─ child rank 1: validate → copy model → Worker/KV
        └─ ...
        ↓ all READY identities agree
PrefixAwareBudgetedScheduler
        ↓
existing v1.1 remote execution / rollback / speculation
        ↓
EngineCore logical commit
        ↓
async committed events
```

The scheduler is shown after readiness conceptually: no request may enter scheduling/execution until engine construction has successfully completed artifact readiness.

## Files added

```text
mini_vllm/artifacts/__init__.py
mini_vllm/artifacts/types.py
mini_vllm/artifacts/tokenizer.py
mini_vllm/artifacts/loader.py
mini_vllm/artifacts/executor.py
mini_vllm/model/factory.py
mini_vllm/versions/v1_2/__init__.py
mini_vllm/versions/v1_2/engine.py
tests/test_model_artifacts.py
examples/17_model_loading.py
```

## Existing files extended

```text
mini_vllm/multiprocess/protocol.py   optional artifact spec + identity observations
mini_vllm/observability.py           artifact snapshot fields
mini_vllm/runtime.py                 RuntimeConfig.model_artifact
mini_vllm/versions/__init__.py       v1.2 default/routing
mini_vllm/__init__.py                public artifact API
pyproject.toml                       1.2.0
scripts/check.sh                     example 17 gate
tests/test_multiprocess_runtime.py   pin historical v1.1 default expectation
README.md / Agents.md / Plan.md / docs/* / version.md
```

Historical v1.1 serving code remains available; v1.2 composes it rather than changing its teaching contract.

## Critical invariants added

### Artifact truth

```text
filesystem presence != validated artifact
```

### Complete coverage

```text
external tensor names == required mapping names
```

No silent missing or extra weights.

### Shape/dtype truth

Correct name alone is insufficient. Shape and supported artifact dtype must match before copy.

### Ownership

```text
checkpoint tensor container != runtime parameter storage
```

The copy is explicit.

### Tokenizer compatibility

```text
tokenizer.vocab_size == model.vocab_size
and
tokenizer.compatibility_id == model manifest compatibility ID
```

### Rank identity

```text
all ready ranks artifact_identity == parent expected identity
all ready ranks tokenizer_identity agree
```

### Failure-before-serving

Artifact/model startup failure occurs during construction/readiness, before request admission.

### Semantic equivalence

An exported deterministic v1.1 model loaded through v1.2 must produce the same committed output at world sizes 1 and 2 under equivalent serving configuration.

All v1.1 transport, KV, scheduling, speculation, async and observability invariants remain active.

## Tests

`tests/test_model_artifacts.py` adds coverage for:

1. deterministic artifact identity;
2. tokenizer encode/decode and unknown-token behavior;
3. missing manifest fields;
4. unsupported artifact version;
5. missing tokenizer file;
6. tokenizer/model vocabulary mismatch;
7. invalid special-token metadata;
8. missing required weight;
9. unexpected weight;
10. wrong weight shape;
11. wrong weight dtype;
12. non-tensor checkpoint entry;
13. runtime-owned parameter copy;
14. v1.2 world-size 1/2 deterministic equivalence with v1.1;
15. text facade remains token-native internally;
16. rank identity disagreement rejected before startup;
17. child changed-artifact startup failure and reap behavior;
18. default temporary artifact lifecycle and cleanup.

Existing v0.1–v1.1 regressions continue to run.

## Example 17

`examples/17_model_loading.py` demonstrates:

```text
export deterministic artifact
→ inspect artifact identity
→ create v1.2 world_size=2 engine
→ encode text
→ serve across real child processes
→ decode generated IDs
→ inspect snapshot artifact identity/rank agreement
→ aclose
```

## Validation evidence

The first complete implementation tree after resolving the package-import boundary ran on repository GitHub Actions with:

```text
101 passed, 1 environment warning
examples 01–17
all checks passed
```

The warning is the existing PyTorch notification that NumPy is not installed; v1.2 artifact digest/copy logic does not require NumPy.

The frozen documentation tree and the cleaned two-commit release SHA must independently pass the same gate before `main` advances.

## Non-goals

- Hugging Face Hub/network downloads;
- authentication/token management;
- arbitrary remote model code;
- full `transformers` tokenizer compatibility;
- safetensors or multiple checkpoint formats;
- quantized checkpoints;
- tensor/pipeline-parallel checkpoint sharding;
- LoRA/adapters;
- active-request model hot swap;
- model registry/cache eviction;
- GPU weight streaming;
- CUDA/Triton kernel work.

## Learning outcome

A reader should be able to explain why model loading is an ownership/trust boundary: external bytes require exact schema/tensor/tokenizer validation, accepted values are copied into runtime-owned modules, every worker independently proves the same model identity, and only then may the already-validated serving runtime admit work.

## Next direction

A suitable v1.3 target is a **network serving/API boundary**: request/response protocol, server lifecycle, health/readiness, disconnect-driven cancellation and ingress backpressure. Adding more model file formats by itself is not a sufficiently distinct new lesson.

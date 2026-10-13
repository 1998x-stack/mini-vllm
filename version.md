# mini-vLLM v1.2 — model artifact + tokenizer loading boundary (planning branch)

## Version identity

- Branch: `v1.2`
- Base: validated `v1.1` / `main`
- Previous: `v1.1`
- Theme: load external model/tokenizer artifacts into validated runtime-owned configuration, weights and token IDs
- Status: planning; runtime remains exactly v1.1 until v1.2 acceptance criteria pass

## Why v1.2

v1.1 made execution physically real: parent policy and child model/KV state now communicate through an explicit transport contract. The next missing boundary appears before serving even starts.

Today the educational runtime constructs `TinyModelConfig` and deterministic model weights directly in code. Real serving systems instead receive external artifacts whose identity, metadata, tensor names, shapes, dtypes and tokenizer vocabulary must be validated before they can become runtime state.

v1.2 should teach:

```text
external files are not runtime truth
        ↓
parse + validate + map
        ↓
validated ModelArtifact
        ↓
construct runtime model
        ↓
load owned tensor copies
        ↓
validated TokenizerArtifact
        ↓
text ⇄ token IDs
        ↓
existing v1.1 serving runtime
```

The new lesson is artifact ingestion and compatibility, not a larger Transformer or a network download client.

## Primary mental model

```text
MODEL DIRECTORY
├─ model_config.json
├─ weights.pt          educational checkpoint format
└─ tokenizer.json
        │
        ▼
ArtifactLoader
├─ parse metadata
├─ validate schema
├─ validate tensor manifest
├─ validate tokenizer vocabulary
└─ compute stable artifact identity
        │
        ▼
ValidatedModelArtifact
├─ model config
├─ named tensor manifest
└─ tokenizer artifact
        │
        ▼
RuntimeModelFactory
├─ construct TinyDecoderLM architecture
├─ map external names → runtime parameter names
├─ copy tensors into runtime-owned parameters
└─ verify no missing/unexpected/incompatible weights
        │
        ▼
Tokenizer
├─ encode(text) → token IDs
└─ decode(token IDs) → text
        │
        ▼
create_engine(model_artifact=...)
        │
        ▼
existing v1.1 multiprocessing serving path
```

## Core ownership rule

```text
artifact bytes / parsed metadata
        !=
runtime configuration
        !=
runtime-owned parameter storage
        !=
request token history
```

Loading means validation plus an explicit ownership transfer into runtime-created modules. It must not mean “keep arbitrary external Python objects alive and trust them forever.”

## Required delta

### 1. Artifact manifest

Introduce a small educational artifact format with explicit files and versions. Preferred first format:

```text
model_dir/
├─ model_config.json
├─ weights.pt
└─ tokenizer.json
```

`model_config.json` should contain the architecture facts needed to construct `TinyModelConfig`, for example:

- artifact format/version;
- architecture identifier;
- vocabulary size;
- hidden dimension;
- attention/KV head counts;
- layer count;
- feed-forward dimension;
- maximum positions;
- expected dtype or supported dtype policy;
- tokenizer compatibility identifier.

The loader must reject missing required fields, unknown incompatible format versions and invalid dimensions before model construction.

### 2. Stable artifact identity

Define a logical `ModelArtifactIdentity`, for example:

```text
artifact format version
architecture
model metadata digest
weights manifest digest
tokenizer digest
```

Identity is for provenance/compatibility and observability. It is not a filesystem path and not a mutable model object.

### 3. Safe tensor manifest validation

The loader must know the expected runtime parameter names and shapes before accepting weights.

Validation should distinguish:

```text
missing tensor
unexpected tensor
shape mismatch
dtype mismatch / supported conversion
non-tensor payload
```

Do not silently ignore unknown tensors or silently initialize missing required weights.

The preferred educational implementation may use `torch.load(..., weights_only=True)` for a local checkpoint if supported by the minimum PyTorch version. Arbitrary checkpoint code execution is explicitly out of scope.

### 4. External-name → runtime-name mapping

Keep mapping explicit and inspectable, even if the first artifact format already uses near-identical names.

Preferred concept:

```python
WeightMapping(
    external_name="layers.0.attn.q_proj.weight",
    runtime_name="layers.0.attn.q_proj.weight",
    expected_shape=(...),
)
```

The learning point is that file naming and runtime module ownership are distinct contracts.

### 5. Runtime-owned parameter copy

After validation:

```text
artifact tensor
    ↓ validate
runtime parameter target
    ↓ copy under no_grad
runtime-owned parameter storage
```

The runtime model must remain usable after the artifact file/tensor container is released. No serving request should depend on the loader keeping a checkpoint dictionary alive.

### 6. Tokenizer artifact

Add a deliberately small readable tokenizer format. A good educational first target is a vocabulary + deterministic greedy/whitespace-safe encoding contract rather than importing a full production tokenizer library.

Required behavior:

```text
text → token IDs
valid token IDs → text
unknown token handling
special token metadata
vocabulary size compatibility
```

The tokenizer's vocabulary size must agree with the model artifact's `vocab_size` before engine construction.

### 7. Text-facing request convenience without replacing token-native core

The serving core should remain token-ID native. v1.2 may add a thin public helper such as:

```python
engine.encode(text)
engine.decode(token_ids)
engine.submit_text(...)
```

or an equivalent artifact-bound facade.

Do not push strings into Scheduler, KV cache, WorkItem or model-runner contracts. Text normalization/tokenization belongs before the serving core.

### 8. Multiprocessing compatibility

v1.1 children currently receive serializable model configuration and construct deterministic model state themselves. v1.2 must decide explicitly how loaded weights enter child processes.

Preferred educational design:

```text
parent validates artifact once
        ↓
serializable validated artifact spec / local artifact path + identity
        ↓
child independently loads validated weights
        ↓
all ranks report same artifact identity
        ↓
serving begins
```

Alternative parent-to-child weight transport is possible but risks teaching a large tensor IPC mechanism at the same time. Keep v1.2 focused on artifact loading rather than weight-distribution optimization.

Whichever design is chosen, all ranks must prove the same model/tokenizer identity before the executor becomes ready.

### 9. Readiness becomes model-specific

v1.1 readiness means “worker process constructed successfully.” v1.2 readiness should include:

```text
process alive
+ artifact parsed
+ architecture compatible
+ all required weights loaded
+ tokenizer/model compatibility validated
+ artifact identity agreed across ranks
```

A worker that cannot validate/load the configured model must fail startup before any request is admitted.

### 10. Observability

Extend public read-only snapshot/model metadata with logical facts such as:

- artifact identity;
- architecture;
- artifact format version;
- tokenizer identity/vocab size;
- loaded dtype/device;
- per-rank model identity agreement;
- loader warnings/conversions that are part of the explicit policy.

Do not expose raw checkpoint dictionaries or parameter storage pointers.

## Critical invariants

### Artifact truth

```text
filesystem presence != valid model artifact
```

Only the validated artifact object/spec may drive runtime construction.

### Complete weight coverage

Every required runtime parameter is populated exactly once from an accepted source tensor unless a documented deterministic initialization rule explicitly says otherwise.

### No silent extras

Unexpected external tensors cause an explicit validation result/error unless the format contract explicitly marks them ignorable.

### Shape compatibility

A tensor with the correct name but incorrect shape can never be copied into runtime state.

### Vocabulary compatibility

```text
tokenizer.vocab_size == model.vocab_size
```

must hold before text/token serving begins.

### Ownership

Loaded runtime parameters must not alias mutable loader/checkpoint containers in a way that allows external mutation to change serving state unexpectedly.

### Rank identity

Every v1.2 worker rank serving the same engine must report the same validated model artifact identity and tokenizer compatibility identity.

### Semantic equivalence

Exporting the deterministic v1.1 TinyDecoder model into the educational artifact format, then loading it through v1.2, must reproduce the same deterministic token outputs under equivalent serving configuration.

### Failure-before-serving

Malformed/incompatible artifacts fail startup before Scheduler admission or request execution.

## Preferred modules

Possible readable layout:

```text
mini_vllm/artifacts/
├─ types.py
├─ loader.py
├─ manifest.py
├─ weights.py
└─ tokenizer.py

mini_vllm/model/
└─ factory.py

mini_vllm/versions/v1_2/
└─ engine.py
```

Keep parsing/validation separate from model construction and separate again from serving execution.

## Tests required

### Valid artifact

- export/build a deterministic tiny model artifact in a temporary directory;
- parse manifest;
- load all weights;
- tokenizer encodes/decodes supported text;
- v1.2 output equals v1.1 reference output.

### Manifest failures

- missing config field;
- unsupported artifact version;
- invalid head/model dimensions;
- missing weights file;
- missing tokenizer file.

### Weight failures

- missing required tensor;
- unexpected tensor;
- wrong shape;
- unsupported dtype;
- non-tensor entry rejected;
- duplicate mapping impossible/rejected.

### Tokenizer failures

- tokenizer/model vocabulary mismatch;
- unknown token policy deterministic;
- invalid token ID rejected;
- malformed special-token metadata.

### Ownership

- mutate/delete loader-side tensor container after successful load and prove runtime output is unchanged;
- prove model parameters are runtime-owned copies under the chosen contract.

### Multiprocess consistency

- world-size 1 and 2 load the same artifact identity;
- rank identity disagreement prevents readiness;
- child loading failure surfaces as startup failure with no orphan workers;
- normal close still reaps all workers.

### Regression

All v0.1–v1.1 tests and examples 01–16 remain green.

## Integrated example target

Add an executable walkthrough such as `examples/17_model_loading.py`:

```text
create/export tiny artifact
        ↓
inspect manifest + identity
        ↓
load tokenizer
        ↓
create v1.2 engine from artifact
        ↓
encode text
        ↓
serve through multiprocessing runtime
        ↓
decode output token IDs
        ↓
show model/tokenizer identity in snapshot
        ↓
clean shutdown
```

## Acceptance criteria

- external local model directory is parsed through a versioned manifest;
- malformed/incompatible manifests fail before serving;
- required weight names/shapes are validated completely;
- runtime parameters are populated from validated external tensors without silent missing/extra tensors;
- a readable tokenizer artifact supports deterministic encode/decode;
- tokenizer/model vocabulary compatibility is enforced;
- token-ID-native serving core remains unchanged in responsibility;
- v1.2 model loading works through the v1.1 multiprocessing boundary;
- every worker rank agrees on the same artifact identity before readiness;
- loaded artifact output is deterministic and equivalent to the exported v1.1 reference model;
- public observability exposes artifact/tokenizer identity without physical storage handles;
- child load failure leaves no orphan processes;
- v0.1–v1.1 regression tests remain green;
- examples 01–16 remain green plus a v1.2 model-loading walkthrough;
- exact release candidate passes repository CI before `main` advances.

## Non-goals

- Hugging Face Hub/network downloads;
- authentication/token management;
- arbitrary remote code/custom model classes;
- full `transformers` tokenizer compatibility;
- safetensors/multiple checkpoint formats unless needed after the reference format works;
- quantized checkpoint formats;
- tensor/pipeline parallel checkpoint sharding;
- model hot swapping while requests are active;
- LoRA/adapters;
- production model registry/cache eviction;
- GPU weight streaming;
- CUDA/Triton kernel work.

## Learning goal

By v1.2 a reader should be able to explain the boundary between untrusted/external model artifacts and trusted runtime state: metadata must be parsed, tensor coverage and shapes validated, tokenizer compatibility proven, rank identity agreed, and only then may the existing serving runtime admit requests.

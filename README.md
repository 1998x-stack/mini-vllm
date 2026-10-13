# mini-vLLM

A small, branch-versioned LLM-serving runtime for learning the systems ideas behind vLLM.

## Current line

`main` is always the newest exact-CI-validated release. v1.2 adds the missing boundary before serving starts: external model/tokenizer files must become validated runtime-owned state before any request is admitted.

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
v1.2  model artifact + tokenizer loading boundary       ← current
```

Historical builders remain available for mechanism-by-mechanism study.

## v1.2 mental model

```text
external model directory
├─ model_config.json
├─ weights.pt
└─ tokenizer.json
        ↓
ArtifactLoader
├─ exact schema/version checks
├─ tokenizer/model compatibility
├─ complete tensor-name coverage
├─ shape + dtype validation
└─ stable ModelArtifactIdentity
        ↓
ValidatedModelArtifact
        ↓
WorkerSpec(path + identity)
        ↓ IPC process startup
child rank 0      child rank 1      ...
re-validate       re-validate
copy weights      copy weights
into runtime      into runtime
owned model       owned model
        ↓              ↓
ready(identity)   ready(identity)
        └──────┬───────┘
               ↓
all ranks agree before serving
               ↓
existing v1.1 Scheduler / KV / speculation / async path
```

The central ownership rule is:

```text
artifact bytes / parsed metadata
        !=
validated artifact identity
        !=
runtime-owned parameter storage
        !=
request token history
```

Filesystem presence is not proof that a model is valid.

## Educational artifact format

v1.2 intentionally supports one small local format:

```text
model_dir/
├─ model_config.json
├─ weights.pt
└─ tokenizer.json
```

The loader rejects missing or unexpected manifest fields, unsupported format versions, incompatible model dimensions, missing/unexpected checkpoint tensors, wrong tensor shape/dtype, non-tensor checkpoint entries, tokenizer/model vocabulary disagreement and malformed special-token metadata.

`weights.pt` is read with `torch.load(..., weights_only=True)`. The first format accepts only the explicitly declared TinyDecoder architecture and float32 artifact weights. No arbitrary checkpoint code execution or remote model code is part of the contract.

## Stable artifact identity

A validated artifact receives a logical identity derived from:

```text
artifact format + architecture
model metadata digest
weights content/manifest digest
tokenizer digest
```

A path is not identity. Every child rank independently re-loads the configured artifact and must report the same identity before the executor becomes ready.

## Runtime-owned parameters

Validated checkpoint tensors are copied into a newly constructed `TinyDecoderLM` under `no_grad`:

```text
external checkpoint tensor
        ↓ validate
runtime parameter target
        ↓ copy
runtime-owned storage
```

The runtime does not retain the loader-side checkpoint dictionary as serving state. Mutating the external container after the copy does not mutate the already-built model.

## Text boundary, token-native core

The tokenizer is deliberately small and readable: a whitespace vocabulary with deterministic unknown-token handling.

```python
engine = create_engine("v1.2", model_artifact="/path/to/model")
ids = engine.encode("tok1 tok2 tok3")
text = engine.decode(ids)
handle = engine.submit_text("r", "tok1 tok2 tok3", max_new_tokens=4)
```

Strings stop at the tokenizer boundary. Scheduler, WorkItem, KV cache, model runner and speculative decoding continue to operate only on token IDs.

## Construction

The default builder is v1.2:

```python
from mini_vllm import RuntimeConfig, create_engine

engine = create_engine(runtime_config=RuntimeConfig())
```

With no explicit model directory, the educational builder exports the deterministic tiny model into a temporary local artifact and then goes through the same loader/validation/child-reload path. There is no hidden in-memory model-construction bypass.

For a persistent artifact:

```python
from mini_vllm import create_engine

engine = create_engine("v1.2", model_artifact="./model_dir")
```

`RuntimeConfig(model_artifact=...)` is also supported. The validated artifact config is authoritative for model dimensions and vocabulary.

Historical references remain explicit:

```python
v10 = create_engine("v1.0")   # integrated in-process reference
v11 = create_engine("v1.1")   # multiprocessing reference
v12 = create_engine("v1.2", model_artifact="./model_dir")
```

## Multiprocess readiness

v1.2 extends v1.1 readiness:

```text
process alive
+ artifact parses
+ tensor coverage/shape/dtype valid
+ tokenizer/model compatible
+ runtime parameter copy succeeds
+ child artifact identity == parent-validated identity
+ every rank agrees
= ready
```

If any child observes changed or invalid files after parent validation, startup fails before request admission and already-started workers are shut down/reaped.

## Observability

`engine.snapshot()` remains read-only and now exposes logical model facts:

- artifact identity and architecture;
- artifact format version;
- tokenizer identity and compatibility ID;
- vocabulary size;
- artifact weight dtype versus runtime dtype/device;
- per-rank artifact/tokenizer identity;
- `rank_identity_agreement`.

It still does not expose checkpoint dictionaries, parameter pointers or child storage handles.

## Existing serving semantics remain intact

v1.2 does not add a new scheduling, KV, speculative, attention or transport algorithm. After model readiness, the existing stack remains:

```text
AsyncEngine
→ EngineCore
→ PrefixAwareBudgetedScheduler
→ ArtifactMultiprocessExecutor
→ child Worker / ModelRunner / transactional paged KV
→ PagedAttention / compile-aware projection
→ speculative verification / Sampler
→ committed async events
```

Exporting the deterministic v1.1 TinyDecoder into the v1.2 format and loading it back produces the same deterministic output for world sizes 1 and 2 under equivalent serving configuration.

## Walkthrough

```bash
pip install -e '.[dev]'
python examples/17_model_loading.py
```

The example exports a tiny artifact, prints its identity, loads it into two real worker processes, tokenizes text, serves it, decodes output and proves the worker ranks agree on model identity through the public snapshot.

## Verify

```bash
./scripts/check.sh
```

The current v1.2 implementation candidate passes 101 pytest tests plus examples 01–17. Only the exact final release SHA with green repository CI may advance `main`.

Read `Agents.md`, `Plan.md`, `docs/ARCHITECTURE.md`, `docs/INVARIANTS.md`, and `version.md` before changing architecture.

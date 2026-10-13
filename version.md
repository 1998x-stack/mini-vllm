# mini-vLLM v1.1 — real multiprocessing executor transport

## Version identity

- Branch: `v1.1`
- Previous: validated `v1.0`
- Theme: make the existing Executor/Worker abstraction a real operating-system process boundary with explicit IPC, lifecycle, health and failure semantics
- Package version: `1.1.0`
- Status: implementation complete; freeze requires the exact final release commit to pass repository CI

## Why v1.1

v0.8 introduced the architectural `EngineCore → Executor → Worker` boundary. v0.9 added logical ranks. v1.0 integrated those mechanisms coherently, but every Worker/model/cache still lived in the same Python process.

v1.1 adds a genuinely new systems lesson:

> An abstraction boundary becomes materially different when it crosses an OS process boundary.

Serialization becomes explicit, physical ownership becomes remote, command completion must be acknowledged, process lifecycle must be managed, and failures become transport events rather than ordinary in-process exceptions.

## Primary mental model

```text
PARENT PROCESS

AsyncEngine
   ↓
EngineCore                         logical request truth
   ↓
Scheduler                          policy
   ↓
MultiprocessExecutor               IPC / health / coordination
   ├──────── command ────────┐
   ▼                          ▼
CHILD rank 0             CHILD rank 1 ...
Worker                   Worker
├─ ModelRunner           ├─ ModelRunner
└─ KV cache              └─ KV cache
   │                          │
   └──── result / error ──────┘
               ↓
        parent-side gather
               ↓
       EngineCore commit
```

Core ownership equation:

```text
parent owns logical truth
child owns physical model/KV truth
IPC acknowledgment is the evidence connecting them
```

## Implementation

### Explicit protocol

New `mini_vllm/multiprocess/protocol.py` defines transport-safe dataclasses:

- `WorkerSpec`;
- `TransportCommand` / `TransportResponse`;
- `TensorPayload`;
- `CheckpointHandle` / `MultiprocessCacheCheckpoint`;
- serialized normal/speculative execution results;
- worker readiness and execution observations.

Every serving command receives a monotonically increasing correlation ID. A response is accepted only when the ID matches the outstanding command.

The protocol carries logical values and copies. It never transports live Scheduler objects, Worker references, cache objects, block-table ownership or live tensor storage.

### Child-owned Worker construction

`MultiprocessExecutor` starts one `multiprocessing.Process` per logical rank. The child receives only `WorkerSpec` and constructs locally:

```text
RankConfig
TinyDecoderLM
TransactionalPrefixCachingPagedKVCache
CompileAwareVocabParallelTinyRunner
Worker
```

Therefore the parent never constructs or directly invokes the child-owned Worker/model/cache used for v1.1 serving.

### Tensor transport

Rank-local logits are converted into a simple explicit CPU wire representation:

```text
TensorPayload
├─ shape
├─ dtype
└─ flat copied values
```

The parent reconstructs fresh tensors, then uses the existing readable `ReferenceCollective` to gather vocabulary shards. v1.1 intentionally avoids shared-memory/CUDA-IPC optimization so transport ownership remains obvious.

### Ordinary execution transaction

Normal execution is protected across IPC, not merely speculative execution.

```text
for every scheduled request:
    checkpoint every child rank
        ↓
all checkpoint handles acknowledged
        ↓
execute work on every rank
        ↓
validate all correlated successful results
        ↓
parent all-gather
        ↓
commit child checkpoints
        ↓
return ExecutionResult
        ↓
EngineCore may advance logical state
```

If any execution fails before logical commit, the executor attempts rollback on every reachable checkpointed rank/request. If rollback cannot be proven complete, the executor becomes unhealthy rather than allowing EngineCore to guess.

### Child-private checkpoint registry

Actual `CacheCheckpoint` objects remain inside the child. The parent receives only `CheckpointHandle(checkpoint_id, request_id, num_tokens)`.

This keeps physical block-table identity private while still allowing rollback/commit commands to reference a transaction deterministically.

### Speculative transport

`EXECUTE_SPECULATIVE` crosses the same process boundary. The pre-existing v0.7/v1.0 speculative flow remains authoritative:

```text
checkpoint
→ remote candidate execution
→ target verification
→ rollback to accepted frontier
→ commit accepted/fallback logical tokens
```

No speculative token becomes caller-visible merely because a child executed it.

### Prefix/cache lifecycle

The following operations now cross IPC in v1.1:

- side-effect-free prefix reuse estimation;
- authoritative prefix attachment;
- capacity reservation;
- normal/speculative execution;
- work publication;
- checkpoint/rollback/commit;
- request free;
- cache statistics.

The Scheduler still observes only logical reuse quantities, never physical child block IDs or refcounts.

## Worker lifecycle

Each rank client performs a readiness handshake before the runtime becomes usable.

Conceptual lifecycle:

```text
STARTING
   ↓ readiness acknowledged
READY
   ├─ normal command → READY
   ├─ structured child error → READY when state remains provable
   ├─ timeout/protocol uncertainty → UNHEALTHY
   └─ unexpected exit/EOF → DEAD
   ↓ explicit shutdown
STOPPED
```

`MultiprocessAsyncEngine` adds:

```python
engine.close()
await engine.aclose()
```

`close()` requires an idle async driver. `aclose()` waits for idle, then sends shutdown and joins workers. Termination is used only as a final cleanup fallback.

## Failure taxonomy

### Structured child error

An ordinary child exception is serialized as a failed `TransportResponse` and surfaced as `WorkerRemoteError`. The child can remain alive; successful repair/rollback may preserve executor health.

### Unexpected process death

A dead child is surfaced explicitly as `WorkerProcessDied`. The parent never treats the absence of a response as success.

### Timeout / protocol uncertainty

A timeout, EOF, invalid response or correlation mismatch means the parent cannot prove the physical result. Such uncertainty cannot authorize logical commit.

### Rollback failure

If failed ordinary execution may have mutated child state and complete rollback cannot be proven, the executor enters unhealthy state and rejects further logical progress.

## Public construction

`create_engine()` now defaults to v1.1.

`RuntimeConfig` adds:

```python
MultiprocessConfig(
    start_method="spawn",
    command_timeout_s=30.0,
    shutdown_timeout_s=2.0,
)
```

Explicit `create_engine("v1.0")` remains the in-process semantic oracle, and v0.1–v0.13 builders remain available as frozen learning snapshots.

## Observability

The stable v1.0 `engine.snapshot()` path remains the public teaching API.

v1.1 extends `ExecutionSnapshot` / `RankExecutionSnapshot` with:

- boundary type (`multiprocess-ipc`);
- child PID;
- lifecycle state;
- serving commands sent/completed;
- structured remote-error count;
- transport-failure count;
- executor health.

Attention and compile observations are read from the real child-owned runner.

Pure observation commands (`CACHE_STATS`, `SNAPSHOT_EXECUTION`) do not increment serving-command counters. Therefore two snapshots with no intervening serving work compare equal.

## Stable invariants added in v1.1

The full contract lives in `docs/INVARIANTS.md`. New process-boundary rules include:

```text
command sent != execution success
```

```text
logical commit requires
correlation-matched acknowledged remote success
```

```text
CheckpointHandle != physical KV ownership
```

```text
unknown child physical consistency
    => executor unhealthy
    => no logical commit
```

and:

```text
normal shutdown => every owned child joined/reaped
```

All v1.0 logical/KV/prefix/preemption/speculation/attention/compile/async invariants remain in force.

## Tests

New `tests/test_multiprocess_runtime.py` covers:

1. transport protocol pickle serialization;
2. actual child PID isolation from parent;
3. default v1.1 builder and process boundary;
4. repeated snapshot equality;
5. world-size 1 output equivalence to v1.0;
6. world-size 2 output equivalence to v1.0;
7. speculative verification over IPC;
8. prefix reuse/cache lifecycle over IPC;
9. vectorized attention observations from child state;
10. structured remote error while worker remains alive;
11. explicit unexpected worker-death detection;
12. shutdown/join with no live worker remaining.

Historical v1.0 regression tests are explicitly version-pinned so changing the default builder does not rewrite the prior lesson.

The first complete implementation candidate passed:

```text
91 pytest tests
examples 01–16
all checks passed
```

The exact final release SHA must independently pass the same repository CI gate before `main` advances.

## Example

`examples/16_multiprocess_transport.py` demonstrates:

```text
parent PID
→ child rank PIDs
→ request submit
→ IPC model/cache execution
→ committed token stream
→ public transport/attention/compile observations
→ aclose()
→ workers alive = false
```

## Files added

- `mini_vllm/multiprocess/protocol.py`
- `mini_vllm/multiprocess/executor.py`
- `mini_vllm/multiprocess/__init__.py`
- `mini_vllm/versions/v1_1/engine.py`
- `mini_vllm/versions/v1_1/__init__.py`
- `tests/test_multiprocess_runtime.py`
- `examples/16_multiprocess_transport.py`

## Important files updated

- `mini_vllm/runtime.py`
- `mini_vllm/observability.py`
- `mini_vllm/versions/__init__.py`
- `mini_vllm/__init__.py`
- `scripts/check.sh`
- `tests/test_v1_version.py`
- `README.md`
- `Agents.md`
- `Plan.md`
- `docs/ARCHITECTURE.md`
- `docs/INVARIANTS.md`
- `docs/ROADMAP.md`
- `docs/EXERCISES.md`
- `version.md`

## Acceptance evidence

- real `multiprocessing.Process` owns each serving Worker/model/cache: implemented and PID-tested;
- parent does not directly invoke child Worker: implemented via `WorkerSpec` + commands;
- explicit serializable command/result protocol: implemented and pickle-tested;
- ordinary execution over IPC: implemented;
- prefix/cache lifecycle over IPC: implemented;
- checkpoint/rollback and speculative execution over IPC: implemented;
- child-reported errors cannot silently commit logical state: structured response + rollback contract;
- unexpected worker death: explicit `WorkerProcessDied` path;
- normal shutdown joins workers: tested;
- world-size 1/2 deterministic output equals v1.0: tested;
- public snapshot exposes transport state without physical handles: implemented;
- v0.1–v1.0 regression suite: preserved in first full CI candidate;
- example 16: included in release check script;
- final release requirement: exact final implementation commit must pass GitHub Actions before `main` moves.

## Non-goals

- multi-node networking;
- NCCL/Gloo production collectives;
- shared-memory optimized tensor transport;
- GPU process isolation or CUDA IPC;
- throughput-oriented process pools;
- automatic elastic worker replacement;
- OpenAI-compatible HTTP serving;
- production model/tokenizer loading;
- Triton/CUDA PagedAttention kernels;
- remote/disaggregated KV cache.

## Learning goal

By v1.1 a reader should be able to explain why a clean Python abstraction is not yet a distributed-systems boundary: once a Worker lives in another process, serialization, correlation, acknowledgment, transaction repair, process health and cleanup become part of correctness.

## Next direction

The preferred next distinct learning target is production-style model/tokenizer loading: turn external model artifacts into validated runtime-owned weights while keeping artifact identity/configuration separate from serving execution state. That belongs in a later version, not in the v1.1 transport mechanism.

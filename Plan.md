# mini-vLLM — Architecture-Driven Development Plan

`main` is always the newest validated runtime. Version branches are educational snapshots. Exact release commits must pass repository CI.

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
v1.1  real multiprocessing executor transport         current
```

## v1.1 learning target

v1.0 proved that all serving mechanisms can coexist behind clean logical ownership boundaries. v1.1 asks the next systems question: what changes when the Executor/Worker boundary becomes a real OS isolation boundary?

```text
parent process
EngineCore / Scheduler
        ↓
MultiprocessExecutor
        ↓ explicit serializable IPC
child process per logical rank
Worker + ModelRunner + KV
        ↓ acknowledged result / structured error
parent gather / logical commit
```

This version is about transport, ownership, lifecycle and failure isolation. It is not a throughput claim and it does not introduce a new scheduling, KV, speculative or attention algorithm.

## New mechanism

### 1. Explicit transport protocol

Transport uses a fixed command/result vocabulary and monotonically increasing correlation IDs. Commands move only logical serializable values: request IDs, token IDs, positions, transport-safe checkpoint handles, copied tensor payloads and observations.

The parent must never treat a live child `Worker`, cache backend, block table, model runner or tensor storage as a local object.

### 2. Process-owned physical state

Each child constructs its own:

```text
RankConfig
TinyDecoderLM
TransactionalPrefixCachingPagedKVCache
CompileAwareVocabParallelTinyRunner
Worker
```

The parent owns only `WorkerSpec`, IPC endpoints and logical runtime state.

### 3. Remote checkpoint/rollback

Ordinary execution crosses a transaction boundary too:

```text
checkpoint every participating request/rank
        ↓
execute remote model work
        ↓
all rank results acknowledged?
  yes → gather → commit checkpoints → EngineCore may commit
  no  → rollback reachable checkpoints → no logical commit
```

Speculative decoding reuses the existing v0.7 logical contract; the difference is that the real cache checkpoint stays in the child and the parent carries only a `CheckpointHandle`.

### 4. Worker lifecycle and health

A remote worker moves through explicit lifecycle states around startup/readiness, normal operation, death and shutdown. Parent-side transport distinguishes:

- child-reported runtime/model errors;
- correlation/protocol failures;
- command timeout;
- unexpected EOF/process death;
- clean shutdown.

A child-reported error can leave the worker healthy after successful rollback. Unknown physical state cannot: the executor becomes unhealthy and refuses further logical progress.

### 5. Explicit cleanup

v1.1 introduces engine cleanup because child processes are owned resources:

```python
engine.close()
await engine.aclose()
```

Normal close sends `shutdown`, joins children, and uses terminate only as cleanup fallback. Tests prove worker processes are not left alive.

## Public construction

`create_engine()` defaults to v1.1. `RuntimeConfig` retains all v1.0 fields and adds `MultiprocessConfig`:

```text
start_method
command_timeout_s
shutdown_timeout_s
```

Explicit `create_engine("v1.0")` remains the in-process integration reference. Historical v0.x builders remain frozen.

## Public observability

`engine.snapshot()` remains the single teaching surface. v1.1 extends execution observations with:

- parent/child boundary type;
- child PID;
- worker lifecycle state;
- serving commands sent/completed;
- remote-error count;
- transport-failure count.

Pure snapshot/cache-stat reads do not increment serving-command counters. Therefore repeated snapshots with no intervening serving action remain equal.

## Cross-process correctness contract

In addition to the v1.0 invariants in `docs/INVARIANTS.md`, v1.1 requires:

1. logical commit requires an acknowledged successful physical result;
2. correlation IDs must match before a result is accepted;
3. IPC carries copies/logical handles, never live physical ownership objects;
4. child-reported errors are distinct from process/transport failures;
5. remote checkpoint identity remains child-owned;
6. rollback must be attempted on every reachable participating rank after failed execution;
7. if complete physical consistency cannot be proven, the executor is unhealthy;
8. unexpected worker death is surfaced explicitly as `WorkerProcessDied`;
9. shutdown joins every child process;
10. deterministic world-size 1 and 2 output equals v1.0 under equivalent configuration.

## Validation

`tests/test_multiprocess_runtime.py` proves:

- protocol dataclasses are pickle-serializable;
- workers have real child PIDs distinct from the parent;
- default v1.1 construction uses the process boundary;
- snapshot remains read-only;
- world-size 1 and 2 outputs match v1.0;
- speculative execution crosses IPC;
- prefix reuse/cache lifecycle crosses IPC;
- child-reported errors remain structured and do not kill healthy workers;
- unexpected child death is explicitly detected;
- close joins/reaps child processes.

`examples/16_multiprocess_transport.py` shows parent PID → command transport → child worker PIDs → committed output → public transport observations → clean shutdown.

The first implementation candidate passed 91 pytest tests plus examples 01–16. The final exact release SHA must independently pass the same gate.

## Release rule

Only the exact v1.1 implementation commit with green repository CI may advance `main`. Before release, branch history is reduced to the planning commit plus one implementation commit so the learning diff remains inspectable.

## After v1.1

The next version must again have a genuinely new learning target. Production-style model/tokenizer loading is a strong candidate because it introduces another ownership boundary—external model artifacts → validated runtime weights—without conflating transport work with GPU-kernel optimization.

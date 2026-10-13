# mini-vLLM Architecture — v1.1

## Integrated serving stack

v1.1 keeps the v1.0 policy stack in the parent process and moves rank-local execution/storage into real child processes.

```text
PARENT PROCESS

caller
  ↓
MultiprocessAsyncEngine
  │ submit / stream / cancel / backpressure
  ▼
MultiprocessIntegratedEngine
  │ EngineCore logical truth / atomic step / commit authority
  ▼
PrefixAwareBudgetedScheduler
  │ admission / token budget / fairness / preemption victim
  ▼
MultiprocessExecutor
  │ correlation / IPC / health / rollback coordination
  ├──────────────────────┬──────────────────────┐
  │ serialized commands  │                      │
  ▼                      ▼                      ▼
CHILD rank 0         CHILD rank 1          CHILD rank N
Worker               Worker                Worker
  ↓                    ↓                     ↓
ModelRunner          ModelRunner           ModelRunner
  ↓                    ↓                     ↓
TinyDecoderLM        TinyDecoderLM         TinyDecoderLM
  ├─ local KV          ├─ local KV           ├─ local KV
  ├─ PagedAttention    ├─ PagedAttention     ├─ PagedAttention
  └─ compile cache     └─ compile cache      └─ compile cache
  │                    │                     │
  └──── copied rank-local result / error ────┘
                       ↓
              ReferenceCollective gather
                       ↓
              EngineCore logical commit
                       ↓
              committed async events
```

The Transformer trunk and KV remain replicated across logical ranks, as in v0.9/v1.0. The new mechanism is process isolation and transport, not a new tensor-parallel algorithm.

## Ownership map

### Parent: Async boundary

`MultiprocessAsyncEngine` owns caller handles, pending submissions, cancellation intent, bounded event queues and child-process cleanup entry points. `close()` is synchronous and requires an idle driver; `aclose()` waits for idle before shutting workers down.

### Parent: EngineCore boundary

`MultiprocessIntegratedEngine` owns all logical request truth:

- request registration/status;
- `num_computed_tokens` logical frontier;
- Scheduler reconciliation;
- preemption state transitions;
- speculative logical acceptance;
- final `EngineStepOutput` commit.

It does not own child cache tensors or execute rank-local model code.

### Parent: Scheduler boundary

`PrefixAwareBudgetedScheduler` is unchanged. It owns admission, work budget, fairness and preemption-victim policy. Multiprocessing does not make process/PID/IPC health a scheduling cost signal.

### Parent: MultiprocessExecutor boundary

`MultiprocessExecutor` owns transport orchestration:

- start method and child creation;
- startup/readiness handshake;
- monotonically increasing correlation IDs;
- command/response matching;
- one-process-per-rank coordination;
- parent-side collective gather;
- remote transaction coordination;
- timeout/EOF/death detection;
- lifecycle/health observations;
- shutdown/join.

It owns no admission policy and no physical KV structures.

### Child: Worker boundary

Each child constructs its own rank-local execution stack from a serializable `WorkerSpec`:

```text
WorkerSpec
  ↓ child startup
RankConfig
TinyDecoderLM
TransactionalPrefixCachingPagedKVCache
CompileAwareVocabParallelTinyRunner
Worker
```

Only that child owns its live `Worker`, model, KV cache, block tables, refcounts, prefix-index residency and compile cache.

### Cache / attention / compile boundaries

Within a child, all pre-v1.1 storage and compute invariants remain unchanged:

```text
logical_block = position // block_size
block_offset  = position % block_size
physical      = block_table[logical_block]
```

`VectorizedPagedAttention` reads the child-local logical KV view. `CompiledKernelCache` specializes child/rank-local vocabulary projection. Neither gains transport ownership.

## IPC protocol

The wire vocabulary is explicit rather than arbitrary remote method invocation. Supported command families include:

```text
ESTIMATE_PREFIX_REUSE
PREPARE_REQUEST
ENSURE_CAPACITY
EXECUTE
EXECUTE_SPECULATIVE
COMMIT_WORK
CHECKPOINT
ROLLBACK
COMMIT_CHECKPOINT
FREE
CACHE_STATS
SNAPSHOT_EXECUTION
SHUTDOWN
```

Each request is represented as:

```text
TransportCommand
├─ correlation_id
├─ operation
└─ serializable payload
```

Each reply is:

```text
TransportResponse
├─ correlation_id
├─ ok
├─ payload
├─ error_type
└─ error_message
```

The parent accepts a response only when its correlation ID matches the outstanding command.

## Tensor transfer

Rank-local logits are converted in the child into an explicit `TensorPayload`:

```text
shape
dtype
flat CPU values
```

The parent reconstructs a fresh tensor before collective gathering. This is intentionally simple and copy-based. v1.1 does not teach shared-memory tensor transport, CUDA IPC or zero-copy transfer.

## Checkpoint handles

Real cache checkpoints remain child-owned. A child stores its actual `CacheCheckpoint` in a private registry and returns only:

```text
CheckpointHandle
├─ checkpoint_id
├─ request_id
└─ num_tokens
```

The parent aggregates one handle per rank into `MultiprocessCacheCheckpoint`. A handle is a transaction identity, not a block-table or storage ownership capability.

## Ordinary execution atomicity

In v1.0, `WorkerGroup` checkpointed rank-local state in process. v1.1 performs the same logical transaction over IPC.

```text
EngineCore work items
      ↓
checkpoint request frontier on every child rank
      ↓
all checkpoint handles acknowledged
      ↓
EXECUTE on rank 0 ... rank N
      ↓
all successful results acknowledged and validated
      ↓
parent all-gather rank-local logits
      ↓
commit remote checkpoints
      ↓
return ExecutionResult to EngineCore
      ↓
logical request state may advance
```

If execution fails before logical commit:

```text
failure
  ↓
attempt rollback on every reachable checkpointed rank/request
  ↓
rollback fully provable?
  ├─ yes → surface original failure; no logical commit
  └─ no  → executor unhealthy; no logical commit
```

A send operation alone never authorizes logical commit.

## Speculative execution

The outer v0.7 transaction contract is preserved:

```text
EngineCore checkpoint()
      ↓
child-owned checkpoint handles
      ↓
EXECUTE_SPECULATIVE across ranks
      ↓
parent gathers verification logits
      ↓
accept/reject candidates
      ↓
ROLLBACK(handle, retain_tokens)
      ↓
only accepted logical tokens become visible
```

Rejected KV stays child-local and is discarded before the corresponding logical candidate is committed.

## Prefix/cache lifecycle over IPC

The cache remains policy-safe despite process isolation:

```text
parent Scheduler wants estimate
        ↓
ESTIMATE_PREFIX_REUSE
        ↓
child computes side-effect-free PrefixReuseHint
        ↓
parent admission policy
        ↓
PREPARE_REQUEST only after admission
        ↓
child performs authoritative prefix attach
        ↓
actual reusable frontier returned / reconciled
```

Capacity, work publication, free and cache statistics use the same explicit boundary. Scheduler still never sees physical block IDs/refcounts.

## Failure classes

### Structured child error

A child catches an ordinary model/cache/runtime exception and returns:

```text
ok = false
error_type
error_message
```

The child process may remain alive. Parent rollback/repair determines whether the executor can remain healthy.

### Protocol / correlation error

A malformed response or wrong correlation ID makes the transport untrustworthy. It is not accepted as execution success.

### Timeout

If no response arrives before the configured timeout, the parent cannot assume success. The transport is unhealthy unless consistency can be re-established—which v1.1 intentionally does not attempt.

### Unexpected process death

If a worker exits or IPC closes unexpectedly, v1.1 surfaces `WorkerProcessDied`. Unknown child physical state means no logical commit and executor unhealthy.

### Cancellation

Cancellation remains an EngineCore/AsyncEngine lifecycle operation at atomic step boundaries. It sends normal release/free operations; it does not kill a healthy worker process.

## Lifecycle

Conceptually each child follows:

```text
CREATED
  ↓
STARTING
  ↓ readiness response
READY
  ↓ commands
READY
  ├─ structured remote error → may remain READY
  ├─ transport uncertainty   → UNHEALTHY
  └─ process exit            → DEAD
  ↓ shutdown
STOPPED
```

The parent joins every child during normal shutdown. If a child does not exit within the shutdown timeout, terminate is a final cleanup fallback.

## Public observability

The v1.0 public path is preserved:

```text
engine.snapshot()
      │
      ├─ async state
      ├─ request state
      ├─ scheduler/admission observations
      ├─ cache statistics
      ├─ recent EngineStepOutput traces
      └─ ExecutionSnapshot
            └─ RankExecutionSnapshot[]
                 ├─ rank/world size
                 ├─ vocab range
                 ├─ attention counters
                 ├─ compile counters
                 ├─ boundary = multiprocess-ipc
                 ├─ process_id
                 ├─ worker_state
                 ├─ serving commands sent/completed
                 ├─ remote errors
                 └─ transport failures
```

`CACHE_STATS` and `SNAPSHOT_EXECUTION` used by public observation are non-serving reads: they do not increment serving command counters. Therefore repeated snapshots with no intervening serving work remain equal.

## One request lifecycle

```text
submit in parent
  ↓
async pending
  ↓
EngineCore WAITING
  ↓ side-effect-free remote prefix hint
Scheduler admission
  ↓ authoritative remote prepare
RUNNING
  ↓
remote chunked prefill / decode
  ↓
acknowledged gathered logits
  ↓
normal or speculative logical decision
  ↓
committed EngineStepOutput
  ↓
TokenEvent delivery
  ↓
FINISHED / CANCELLED / ERROR
```

The caller never sees a token merely because a child computed something. External visibility begins only after EngineCore has committed it.

## Logical versus physical versus transport truth

v1.1 adds a third systems distinction to the earlier logical/physical separation:

```text
logical request truth           parent / EngineCore
physical model + KV truth       child process
transport evidence              IPC command/acknowledgment
```

Correctness requires all three to align before logical commit.

## Stable correctness references

- deterministic v1.0 committed output is the v1.1 semantic oracle;
- child-local paged addressing remains the KV oracle;
- reference PagedAttention remains the attention oracle;
- eager projection remains the compile oracle;
- `docs/INVARIANTS.md` is the cross-version correctness contract.

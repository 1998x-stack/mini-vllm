# mini-vLLM Architecture — v0.12

## Stable upper layers

```text
AsyncEngine
  ↓
EngineCore / Scheduler
  ↓
DistributedExecutor
  ↓
WorkerGroup
  ↓
rank-local Worker / ModelRunner
```

These layers keep the v0.11 semantics for lifecycle, scheduling, transactions, distributed rollback, compile-aware projection, streaming, cancellation, and backpressure.

## Attention optimization boundary

```text
TinySelfAttention
  │ q/k/v projection
  │ write current K/V through cache backend
  ▼
PagedAttentionBackend
  ├─ ReferencePagedAttention
  │    └─ cache.read_prefix(): readable per-position gather
  │
  └─ VectorizedPagedAttention
       └─ cache.read_prefix_vectorized():
            positions tensor
                ↓
            logical block + offset
                ↓
            block-table tensor lookup
                ↓
            physical K/V advanced indexing
  ↓
identical GQA/MHA score → softmax → context math
  ↓
output projection
```

The attention backend never allocates, frees, shares, evicts, or rolls back blocks. It consumes the current logical view produced by the cache manager.

## Logical versus physical length

For block size `B`:

```text
logical_block = position // B
block_offset  = position % B
physical      = block_table[logical_block]
```

A request with logical length 5 and block size 2 owns three physical blocks, but only offsets corresponding to positions 0..4 are valid. The unused final slot must never enter attention even if it contains stale data.

## GQA / MQA

The cache stores `n_kv_heads`; the query tensor uses `n_heads`. When `n_heads > n_kv_heads`, both backends expand K/V heads with the same repeat mapping before score computation. Optimized gathering therefore changes storage access, not head semantics.

## Prefix sharing / rollback

Prefix caching may point multiple request block tables at one immutable physical block. Preemption and speculative rollback may later shorten or replace a request's private tail. Vectorized attention always resolves the current block table at call time, so it cannot retain stale physical addresses across those lifecycle changes.

## Compile-aware projection remains separate

v0.11's `CompiledKernelCache` still wraps only the rank-local vocabulary projection. Attention optimization and projection compilation are independent execution-plane mechanisms and can be inspected separately.

## Fallback

`VectorizedPagedAttention` records a fallback and uses the reference read when the selected cache backend does not implement vectorized paged access and fallback is enabled. Strict mode raises. CPU CI validates reference and vectorized PyTorch paths; no Triton/CUDA kernel is claimed in v0.12.

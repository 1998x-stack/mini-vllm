# mini-vLLM v0.12 — optimized PagedAttention (planning branch)

## Version identity

- Branch: `v0.12`
- Base: validated `v0.11` / `main`
- Previous: `v0.11`
- Theme: extract paged attention behind a correctness oracle, then add a vectorized PyTorch gather/attention path with explicit fallback
- Status: planning until the exact implementation commit passes repository CI

## Primary mental model

```text
Paged KV physical blocks + block table
                ↓
ReferencePagedAttention          correctness oracle
                ↓
VectorizedPagedAttention         optimized PyTorch path
                ↓
optional future Triton/CUDA path
                ↓
same attention context / logits / tokens
```

The optimization replaces attention data gathering/tensor math only. Scheduler, request lifecycle, prefix ownership, preemption, speculative transactions, WorkerGroup topology, compile-aware LM-head projection, async streaming, and cancellation remain unchanged.

## Required abstractions

- `PagedAttentionConfig`
- `PagedAttentionStats`
- `PagedAttentionEvent`
- `PagedAttentionBackend`
- `ReferencePagedAttention`
- `VectorizedPagedAttention`
- `build_paged_attention_backend`
- vectorized paged-cache prefix read that maps logical positions to physical block/offset tensors without a Python per-token loop

## Addressing invariant

```text
logical_block = position // block_size
block_offset  = position % block_size
physical      = block_table[request_id][logical_block]
```

For logical length `L`, only positions `[0, L)` may participate in attention. Unused slots in the last physical block must remain invisible.

## GQA / MQA invariant

If `n_heads > n_kv_heads`, KV heads are expanded to query-head count with the same semantics as the v0.11 reference implementation. Optimized gathering must not change head mapping.

## Prefix / rollback invariant

Shared immutable prefix blocks, private mutable tails, preemption/recompute, and speculative rollback may change which physical blocks a request references over time. The optimized attention path must use the current block table and logical length only; stale/recycled physical slots must never leak into attention.

## Fallback

`backend="vectorized"` is the v0.12 default. If the cache backend cannot expose a vectorized paged read and fallback is enabled, attention executes through the reference path and records the fallback. Strict mode raises instead.

Triton/CUDA is optional and not required for v0.12 correctness or CPU CI. Do not claim a Triton implementation unless a real kernel exists and is numerically validated.

## Acceptance criteria

- [ ] reference backend reproduces pre-v0.12 attention exactly/numerically;
- [ ] vectorized paged read matches the reference loop for multi-block and partial-tail sequences;
- [ ] vectorized attention matches reference for ordinary MHA and GQA/MQA;
- [ ] block tables need not be physically contiguous;
- [ ] partial final blocks never expose unused slots;
- [ ] fallback/error policy is deterministic and observable;
- [ ] shared-prefix requests preserve output equivalence;
- [ ] deterministic v0.12 output matches v0.11 for world_size 1 and 2;
- [ ] speculative verification, preemption, compile-aware projection, async streaming, cancellation, and backpressure regressions pass;
- [ ] an executable example shows reference vs vectorized stats and output equivalence;
- [ ] all historical v0.1-v0.11 tests pass;
- [ ] exact candidate `./scripts/check.sh` passes in GitHub Actions before `main` advances.

## Non-goals

- production FlashAttention/vLLM kernel parity;
- distributed KV partitioning;
- NCCL/CUDA Graph integration;
- production Triton tuning;
- prefix-aware admission policy (`v0.13`);
- claiming speedup without measurement.

## Next

`v0.13` makes prefix reuse visible to scheduling/admission policy rather than only to the cache backend.

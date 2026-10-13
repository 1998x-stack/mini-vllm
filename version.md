# mini-vLLM v0.12 — optimized PagedAttention

## Version identity

- Branch: `v0.12`
- Previous: validated `v0.11`
- Theme: reference and vectorized paged attention with explicit cache-addressing invariants and fallback
- Release rule: the exact implementation commit may enter `main` only after repository CI passes

## Mental model

```text
current logical sequence length
          +
current block table
          +
physical KV tensors
          ↓
ReferencePagedAttention
          or
VectorizedPagedAttention
          ↓
same GQA/MHA attention context
          ↓
TinyDecoder
```

v0.12 optimizes paged KV reading; it does not change request policy or cache ownership.

## Added abstractions

- `PagedAttentionConfig(backend, fallback_to_reference)`
- `PagedAttentionStats`
- `PagedAttentionEvent`
- `PagedAttentionBackend`
- `ReferencePagedAttention`
- `VectorizedPagedAttention`
- `build_paged_attention_backend`
- `KVCacheBackend.read_prefix_vectorized()` optional contract
- `PagedKVCache.read_prefix_vectorized()` tensor-indexed physical gather

## Vectorized addressing

For logical positions `0..L-1`:

```text
positions       = arange(L)
logical_blocks  = positions // block_size
offsets         = positions % block_size
physical_blocks = block_table[logical_blocks]
K/V             = physical_storage[physical_blocks, offsets]
```

This removes the reference Python loop over logical positions while preserving ordering exactly.

## Partial tail invariant

Allocated capacity is not logical sequence length. If the final physical block contains unused or stale slots, only positions `< L` are gathered. Tests write a large sentinel into an unused tail slot and prove it is invisible to vectorized reads and attention.

## GQA / MQA

Both backends preserve the pre-v0.12 mapping from `n_kv_heads` to `n_heads` by repeating each KV head across the corresponding query-head group before score computation.

## Ownership / lifecycle compatibility

The attention backend never:

- allocates/frees physical blocks;
- changes block tables;
- mutates prefix refcounts;
- chooses preemption victims;
- commits/rolls back speculative state;
- participates in Scheduler or AsyncEngine policy.

Prefix sharing, preemption, and rollback remain cache/runtime mechanisms. Attention resolves the current mapping at every call.

## Distributed + compile composition

Every logical rank owns its own attention backend and statistics. The Transformer trunk/KV remain replicated as in v0.9-v0.11. v0.11's compile-aware vocabulary projection remains unchanged and executes after the attention/trunk hidden trace. Collectives remain outside both optimization mechanisms.

## Fallback

`backend="vectorized"` is the v0.12 default. A cache backend without vectorized paged reads triggers an observable reference fallback when enabled. Strict mode raises a clear runtime error instead.

A real Triton/CUDA attention kernel is not claimed in v0.12; it remains optional future work after numerical validation and device capability checks.

## Validation coverage

`tests/test_paged_attention.py` proves:

- vectorized physical gather equals reference gather for a deliberately non-contiguous block table;
- unused partial-tail slots are masked by logical length;
- reference and vectorized GQA attention match numerically;
- non-paged cache fallback and strict error modes are explicit;
- TinyDecoder logits match across reference/vectorized attention.

`tests/test_v012_version.py` proves:

- v0.12 deterministic async tokens equal v0.11;
- reference and vectorized v0.12 engines match;
- world-size 1 and 2 match;
- attention and compile state remain rank-local and observable;
- shared-prefix warm reuse remains correct with vectorized attention.

`examples/13_paged_attention.py` prints reference/vectorized output equality and per-rank attention statistics.

The exact candidate must pass `./scripts/check.sh`, which runs all historical tests and examples 01-13.

## Non-goals

- production FlashAttention/vLLM kernel parity;
- claiming PyTorch vectorization is faster without measurement;
- distributed KV partitioning;
- NCCL/CUDA Graph capture;
- fake Triton wrappers that only delegate to PyTorch;
- prefix-aware admission policy (`v0.13`).

## Next

`v0.13` moves prefix reuse information upward into scheduling/admission policy while preserving cache correctness and starvation guarantees.

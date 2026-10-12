# mini-vLLM v0.3

## 1. Version identity

- **Branch:** `v0.3`
- **Previous version:** `v0.2`
- **Theme:** paged KV cache and block tables
- **Status:** frozen historical snapshot; current integrated `main` release

## 2. Primary mental model

```text
logical token positions
0 1 2 3 | 4 5 6 7 | 8 9 ...
    │          │          │
    ▼          ▼          ▼
 logical block 0   1          2
    │          │          │
    └──── block_table ────┘
               │
               ▼
 non-contiguous physical blocks
```

## 3. What changed relative to v0.2, and why?

v0.2 reserves a full `max_model_len` slab per active request. v0.3 allocates fixed-size physical blocks on demand so memory reservation follows actual sequence growth.

```text
logical_block = position // block_size
block_offset  = position % block_size
physical      = block_table[request_id][logical_block]
```

## 4. Mechanisms added

- fixed physical KV block pool
- per-request block table
- on-demand block allocation
- `slot_mapping()`
- reference paged prefix gather
- block recycling on request finish
- cache utilization metrics
- numerical equivalence test against contiguous KV

## 5. Mechanisms intentionally unchanged

- continuous scheduler policy
- Request lifecycle
- Tiny Decoder / ModelRunner
- sampler
- synchronous Engine

## 6. Important source files

```text
mini_vllm/cache/paged.py
mini_vllm/versions/v0_3/engine.py
examples/03_page_table_walkthrough.py
examples/04_compare_cache_waste.py
tests/test_model_equivalence.py
```

## 7. Critical tests

- logical positions map to the correct physical block/offset
- physical blocks recycle after `free()`
- contiguous and paged cache layouts produce equivalent logits
- finished requests leave zero paged blocks owned

## 8. Learning goals

Explain why logical sequence contiguity does not imply physical memory contiguity, what a block table does, and why PagedAttention can fuse address translation with attention without changing the model's semantics.

## 9. Non-goals

- token-budget scheduling
- chunked prefill
- prefix caching
- preemption
- speculative decoding
- executor/worker split
- Triton/CUDA kernel

## 10. Next version

`v0.4` introduces token-budget scheduling and chunked prefill.

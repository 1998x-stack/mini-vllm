# mini-vLLM

A deliberately small, runnable, **branch-versioned** LLM-serving runtime for learning the core systems ideas behind vLLM.

Current integrated release on `main`: **v0.3**.

## Branch-first learning model

```text
main  ───────────────────────────────► newest validated state (v0.3)
  │
  ├── v0.1  static batching + contiguous KV
  ├── v0.2  continuous batching + contiguous KV
  ├── v0.3  continuous batching + paged KV
  └── v0.4  planning: token budget + chunked prefill
```

Study the project by comparing branches:

```bash
git diff v0.1..v0.2
git diff v0.2..v0.3
```

Before changing code, read `Agents.md`. `Plan.md` is the canonical forward roadmap.

## Core runtime

```text
Request
  ↓
Scheduler
  ↓
KV Cache Backend
  ↓
ModelRunner
  ↓
Sampler
  ↓
request state update / free
```

## Version ladder

| Branch | Scheduler | KV cache | Primary lesson |
|---|---|---|---|
| `v0.1` | Static cohort | Contiguous slab | Minimal serving loop |
| `v0.2` | Continuous | Contiguous slab | Dynamic slot refill |
| `v0.3` | Continuous | Paged blocks | Block tables and recycling |
| `v0.4` | Planned token-budget scheduler | Paged blocks | Chunked prefill |

## Install / verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
./scripts/check.sh
```

## v0.3 key idea

For block size `B` and token position `p`:

```text
logical_block = p // B
block_offset  = p % B
physical      = block_table[request_id][logical_block]
```

Example:

```text
block_size = 4
block_table[A] = [5, 1, 8]
position = 9

logical block = 2
physical block = 8
offset = 1
```

Run:

```bash
python examples/03_page_table_walkthrough.py
python examples/04_compare_cache_waste.py
```

The reference implementation deliberately gathers paged K/V into a logical contiguous prefix before ordinary PyTorch attention. It is a correctness/learning path, not a fused CUDA/Triton kernel.

## Reading order

```text
core/request.py
  ↓
core/work.py
  ↓
scheduler/static.py
  ↓
scheduler/continuous.py
  ↓
cache/contiguous.py
  ↓
cache/paged.py
  ↓
model/tiny_transformer.py
  ↓
core/engine.py
```

Next: checkout `v0.4` and read its `version.md` before implementing token-budget scheduling and chunked prefill.

# mini-vLLM v0.1

This is the **v0.1 historical learning snapshot**: static batching + contiguous KV cache.

Study the baseline serving loop before any dynamic admission or paged memory.

Before changing code, read `Agents.md`. For the complete forward roadmap, read `Plan.md`. For this branch's exact scope, read `version.md`.

## Run

```bash
pip install -e '.[dev]'
./scripts/check.sh
```

## Learning workflow

```text
Request
  ↓
Scheduler
  ↓
KV Cache
  ↓
ModelRunner
  ↓
Sampler
  ↓
request update
```

## v0.1 focus

```text
Static cohort
A B
A _
A _
C _
```

Read: `core/request.py` → `core/work.py` → `scheduler/static.py` → `cache/contiguous.py` → `model/tiny_transformer.py` → `core/engine.py`.

Next, compare with v0.2:

```bash
git diff v0.1..v0.2
```

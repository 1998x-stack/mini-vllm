# mini-vLLM v0.2

This is the **v0.2 historical learning snapshot**: continuous batching + contiguous KV cache.

Compare this branch directly with v0.1; the key architectural delta is scheduler policy.

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

## v0.2 focus

```text
Continuous refill
A B
A C
A _
```

Run:

```bash
python examples/02_static_vs_continuous.py
```

Then compare memory architecture with v0.3:

```bash
git diff v0.2..v0.3
```

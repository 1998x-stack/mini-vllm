"""Compare v0.1 static batching with v0.2 continuous batching."""

from mini_vllm.cache.noop import NoOpKVCache
from mini_vllm.config import EngineConfig
from mini_vllm.core.engine import Engine
from mini_vllm.core.request import Request
from mini_vllm.core.sampler import Sampler
from mini_vllm.model.runner import ScriptedModelRunner
from mini_vllm.scheduler.continuous import ContinuousBatchScheduler
from mini_vllm.scheduler.static import StaticBatchScheduler


def build(version: str) -> Engine:
    scheduler = StaticBatchScheduler(2) if version == "v0.1" else ContinuousBatchScheduler(2)
    return Engine(
        config=EngineConfig(max_num_seqs=2, max_model_len=32),
        scheduler=scheduler,
        cache=NoOpKVCache(),
        model_runner=ScriptedModelRunner(vocab_size=32),
        sampler=Sampler(),
        version=version,
    )


for version in ["v0.1", "v0.2"]:
    engine = build(version)
    engine.add_request(Request("A", [1], max_new_tokens=4))
    engine.add_request(Request("B", [2], max_new_tokens=1))
    print(f"\n--- {version} ---")
    print(engine.step())
    engine.add_request(Request("C", [3], max_new_tokens=1))
    while engine.has_unfinished_requests:
        event = engine.step()
        print(f"step={event.step} scheduled={event.scheduled_request_ids} admitted={event.admitted_request_ids} finished={event.finished_request_ids}")

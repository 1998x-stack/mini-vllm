from mini_vllm.cache.noop import NoOpKVCache
from mini_vllm.config import EngineConfig
from mini_vllm.core.engine import Engine
from mini_vllm.core.request import Request
from mini_vllm.core.sampler import Sampler
from mini_vllm.model.runner import ScriptedModelRunner
from mini_vllm.scheduler.static import StaticBatchScheduler


def test_static_batch_does_not_backfill_until_cohort_drains() -> None:
    engine = Engine(config=EngineConfig(max_num_seqs=2, max_model_len=32), scheduler=StaticBatchScheduler(2), cache=NoOpKVCache(), model_runner=ScriptedModelRunner(vocab_size=16), sampler=Sampler(), version="v0.1")
    engine.add_request(Request("A", [1], 3))
    engine.add_request(Request("B", [2], 1))
    engine.step()
    engine.add_request(Request("C", [3], 1))
    assert engine.step().scheduled_request_ids == ("A",)
    assert engine.step().scheduled_request_ids == ("A",)
    assert engine.step().scheduled_request_ids == ("C",)

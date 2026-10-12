from mini_vllm.cache.noop import NoOpKVCache
from mini_vllm.config import EngineConfig
from mini_vllm.core.engine import Engine
from mini_vllm.core.request import Request
from mini_vllm.core.sampler import Sampler
from mini_vllm.model.runner import ScriptedModelRunner
from mini_vllm.scheduler.continuous import ContinuousBatchScheduler
from mini_vllm.scheduler.static import StaticBatchScheduler


def make_engine(continuous: bool) -> Engine:
    scheduler = ContinuousBatchScheduler(2) if continuous else StaticBatchScheduler(2)
    return Engine(config=EngineConfig(max_num_seqs=2, max_model_len=32), scheduler=scheduler, cache=NoOpKVCache(), model_runner=ScriptedModelRunner(vocab_size=16), sampler=Sampler(), version="test")


def test_static_batch_does_not_backfill_until_cohort_drains() -> None:
    engine = make_engine(False)
    engine.add_request(Request("A", [1], 3))
    engine.add_request(Request("B", [2], 1))
    first = engine.step()
    assert set(first.scheduled_request_ids) == {"A", "B"}
    engine.add_request(Request("C", [3], 1))
    second = engine.step()
    assert second.scheduled_request_ids == ("A",)
    assert "C" not in second.admitted_request_ids
    third = engine.step()
    assert third.scheduled_request_ids == ("A",)
    assert engine.get_request("A").finished
    assert engine.step().scheduled_request_ids == ("C",)


def test_continuous_batch_backfills_next_step() -> None:
    engine = make_engine(True)
    engine.add_request(Request("A", [1], 3))
    engine.add_request(Request("B", [2], 1))
    engine.step()
    engine.add_request(Request("C", [3], 1))
    second = engine.step()
    assert set(second.scheduled_request_ids) == {"A", "C"}
    assert second.admitted_request_ids == ("C",)

from __future__ import annotations

import torch
import pytest

from mini_vllm.cache.base import CacheCheckpoint, KVCacheBackend
from mini_vllm.config import EngineConfig
from mini_vllm.core.engine import EngineCore
from mini_vllm.core.request import Request
from mini_vllm.core.work import ScheduleDecision
from mini_vllm.execution import ExecutionItem, ExecutionRequest, LocalExecutor, Worker
from mini_vllm.model.runner import ScriptedModelRunner
from mini_vllm.scheduler.base import Scheduler, make_work_item
from mini_vllm.speculative.draft import ScriptedDraftProvider
from mini_vllm.speculative.engine import SpeculativeEngine


class FakeCache(KVCacheBackend):
    def __init__(self) -> None:
        self.used: dict[str, int] = {}

    def ensure_capacity(self, request_id: str, total_tokens: int) -> bool:
        self.used.setdefault(request_id, 0)
        return True

    def write(self, *args, **kwargs) -> None:
        return None

    def read_prefix(self, *args, **kwargs):
        return torch.empty(0), torch.empty(0)

    def free(self, request_id: str) -> None:
        self.used.pop(request_id, None)

    def stats(self) -> dict[str, int | float]:
        return {"used": sum(self.used.values())}

    def commit_work(
        self,
        request_id: str,
        *,
        start_pos: int,
        token_ids: tuple[int, ...],
        prompt_length: int,
    ) -> None:
        self.used[request_id] = start_pos + len(token_ids)

    def checkpoint(self, request_id: str, num_tokens: int) -> CacheCheckpoint:
        self.used[request_id] = max(self.used.get(request_id, 0), num_tokens)
        return CacheCheckpoint(request_id=request_id, num_tokens=num_tokens, block_table=())

    def rollback(self, checkpoint: CacheCheckpoint, *, retain_tokens: int) -> None:
        self.used[checkpoint.request_id] = retain_tokens

    def commit_checkpoint(self, checkpoint: CacheCheckpoint) -> None:
        return None


class OneRequestScheduler(Scheduler):
    def __init__(self) -> None:
        self.waiting: list[Request] = []
        self.running: list[Request] = []

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    def schedule(self) -> ScheduleDecision:
        admitted: list[str] = []
        self.running = [r for r in self.running if not r.finished]
        if not self.running and self.waiting:
            request = self.waiting.pop(0)
            request.mark_running()
            self.running.append(request)
            admitted.append(request.request_id)
        return ScheduleDecision(
            work_items=[make_work_item(r) for r in self.running],
            admitted_request_ids=admitted,
        )

    def on_request_finished(self, request: Request) -> None:
        self.running = [r for r in self.running if not r.finished]


class FailingExecutor(LocalExecutor):
    def execute(self, request: ExecutionRequest):
        raise RuntimeError("worker failed")


def make_local_executor() -> LocalExecutor:
    return LocalExecutor(
        Worker(
            model_runner=ScriptedModelRunner(vocab_size=16),
            cache=FakeCache(),
        )
    )


def test_execution_payload_does_not_cross_request_policy_object() -> None:
    item = ExecutionItem(request_id="A", token_ids=(1, 2), start_pos=0)
    payload = ExecutionRequest(items=(item,))
    assert payload.items[0].request_id == "A"
    assert not hasattr(payload.items[0], "request")


def test_worker_has_no_scheduler_or_request_queue_policy() -> None:
    executor = make_local_executor()
    assert not hasattr(executor.worker, "scheduler")
    assert not hasattr(executor.worker, "waiting")
    assert not hasattr(executor.worker, "running")


def test_enginecore_normal_work_executes_through_local_executor() -> None:
    executor = make_local_executor()
    engine = EngineCore(
        config=EngineConfig(max_num_seqs=1, max_model_len=16),
        scheduler=OneRequestScheduler(),
        executor=executor,
        version="v0.8-test",
    )
    request = Request("A", [1], max_new_tokens=2)
    engine.add_request(request)
    engine.run()
    assert request.output_token_ids == [2, 3]
    assert executor.execute_calls == 2


def test_executor_failure_surfaces_without_logical_commit() -> None:
    executor = FailingExecutor(
        Worker(model_runner=ScriptedModelRunner(vocab_size=16), cache=FakeCache())
    )
    engine = EngineCore(
        config=EngineConfig(max_num_seqs=1, max_model_len=16),
        scheduler=OneRequestScheduler(),
        executor=executor,
        version="v0.8-test",
    )
    request = Request("B", [4], max_new_tokens=1)
    engine.add_request(request)
    with pytest.raises(RuntimeError, match="worker failed"):
        engine.step()
    assert request.num_computed_tokens == 0
    assert request.output_token_ids == []


def test_speculation_uses_executor_boundary_too() -> None:
    executor = make_local_executor()
    engine = SpeculativeEngine(
        config=EngineConfig(max_num_seqs=1, max_model_len=16, max_speculative_tokens=2),
        scheduler=OneRequestScheduler(),
        executor=executor,
        version="v0.8-test",
        draft_provider=ScriptedDraftProvider({"S": [[2, 3]]}),
        max_speculative_tokens=2,
    )
    request = Request("S", [1], max_new_tokens=3)
    engine.add_request(request)
    output = engine.step()
    assert request.output_token_ids == [2, 3, 4]
    assert executor.execute_calls == 1
    assert executor.speculative_execute_calls == 1
    assert output.speculation[0].accepted_token_ids == (2, 3)

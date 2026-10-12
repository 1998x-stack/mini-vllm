from __future__ import annotations

from collections import OrderedDict

from mini_vllm.cache.base import KVCacheBackend
from mini_vllm.config import EngineConfig
from mini_vllm.core.request import Request
from mini_vllm.core.sampler import Sampler
from mini_vllm.core.work import EngineStepOutput, ScheduleDecision, WorkItem
from mini_vllm.model.runner import ModelRunner
from mini_vllm.scheduler.base import Scheduler


class Engine:
    """Small synchronous serving loop.

    The important boundary is explicit:
    scheduler -> cache capacity -> model runner -> sampler -> request update.
    """

    def __init__(self, *, config: EngineConfig, scheduler: Scheduler, cache: KVCacheBackend, model_runner: ModelRunner, sampler: Sampler | None = None, version: str) -> None:
        self.config = config
        self.scheduler = scheduler
        self.cache = cache
        self.model_runner = model_runner
        self.sampler = sampler or Sampler()
        self.version = version
        self._requests: OrderedDict[str, Request] = OrderedDict()
        self._step = 0

    def add_request(self, request: Request) -> None:
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        if len(request.prompt_token_ids) + request.max_new_tokens > self.config.max_model_len:
            raise ValueError("prompt + max_new_tokens exceeds max_model_len")
        self._requests[request.request_id] = request
        self.scheduler.add_request(request)

    def get_request(self, request_id: str) -> Request:
        return self._requests[request_id]

    @property
    def has_unfinished_requests(self) -> bool:
        return any(not request.finished for request in self._requests.values())

    def _fit_cache(self, decision: ScheduleDecision) -> ScheduleDecision:
        accepted: list[WorkItem] = []
        for item in decision.work_items:
            if self.cache.ensure_capacity(item.request_id, item.end_pos):
                accepted.append(item)
        decision.work_items = accepted
        return decision

    def step(self) -> EngineStepOutput:
        decision = self._fit_cache(self.scheduler.schedule())
        sampled: dict[str, int] = {}
        finished: list[str] = []

        if decision.work_items:
            logits_by_request = self.model_runner.execute(decision.work_items, self.cache)
            for item in decision.work_items:
                request = item.request
                request.num_computed_tokens = item.end_pos
                token_id = self.sampler.sample(logits_by_request[item.request_id], request.sampling_params)
                request.append_output(token_id)
                sampled[item.request_id] = token_id

                reached_eos = self.config.eos_token_id is not None and token_id == self.config.eos_token_id
                reached_limit = len(request.output_token_ids) >= request.max_new_tokens
                if reached_eos or reached_limit:
                    request.mark_finished("eos" if reached_eos else "length")
                    self.cache.free(request.request_id)
                    self.scheduler.on_request_finished(request)
                    finished.append(request.request_id)

        output = EngineStepOutput(
            step=self._step,
            scheduled_request_ids=tuple(item.request_id for item in decision.work_items),
            admitted_request_ids=tuple(decision.admitted_request_ids),
            sampled_token_ids=sampled,
            finished_request_ids=tuple(finished),
            cache_stats=self.cache.stats(),
        )
        self._step += 1
        return output

    def run(self, max_steps: int = 10_000) -> list[EngineStepOutput]:
        trace: list[EngineStepOutput] = []
        while self.has_unfinished_requests:
            if len(trace) >= max_steps:
                raise RuntimeError("engine exceeded max_steps; possible scheduler deadlock")
            trace.append(self.step())
        return trace

from __future__ import annotations

from collections import OrderedDict

from mini_vllm.cache.base import KVCacheBackend
from mini_vllm.config import EngineConfig
from mini_vllm.core.request import Request, RequestStatus
from mini_vllm.core.sampler import Sampler
from mini_vllm.core.work import EngineStepOutput, ScheduleDecision, WorkItem, WorkTrace
from mini_vllm.execution import ExecutionItem, ExecutionRequest, Executor, LocalExecutor, Worker
from mini_vllm.model.runner import ModelRunner
from mini_vllm.scheduler.base import Scheduler, make_work_item
from mini_vllm.speculative.types import VerificationResult


class EngineCore:
    """Central policy/state coordinator.

    Physical execution lives behind Executor/Worker. Request lifecycle,
    scheduling, preemption, cancellation and logical commit stay centralized.
    """

    def __init__(
        self,
        *,
        config: EngineConfig,
        scheduler: Scheduler,
        executor: Executor | None = None,
        cache: KVCacheBackend | None = None,
        model_runner: ModelRunner | None = None,
        sampler: Sampler | None = None,
        version: str,
    ) -> None:
        if executor is None:
            if cache is None or model_runner is None:
                raise ValueError("provide executor or both cache and model_runner")
            executor = LocalExecutor(Worker(model_runner=model_runner, cache=cache))
        elif cache is not None or model_runner is not None:
            raise ValueError("executor is mutually exclusive with cache/model_runner")

        self.config = config
        self.scheduler = scheduler
        self.executor = executor
        self.sampler = sampler or Sampler()
        self.version = version
        self._requests: OrderedDict[str, Request] = OrderedDict()
        self._step = 0
        self._preemption_events = 0
        self._step_speculation: list[VerificationResult] = []

    @property
    def cache(self) -> KVCacheBackend:
        if not isinstance(self.executor, LocalExecutor):
            raise RuntimeError("cache object is not directly visible through this executor")
        return self.executor.worker.cache

    @property
    def model_runner(self) -> ModelRunner:
        if not isinstance(self.executor, LocalExecutor):
            raise RuntimeError("model runner is not directly visible through this executor")
        return self.executor.worker.model_runner

    def _validate_cached_token_count(self, request: Request, cached_tokens: int) -> None:
        if cached_tokens < 0 or cached_tokens > len(request.prompt_token_ids):
            raise RuntimeError("cache returned invalid cached-token count")

    def add_request(self, request: Request) -> None:
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        if len(request.prompt_token_ids) + request.max_new_tokens > self.config.max_model_len:
            raise ValueError("prompt + max_new_tokens exceeds max_model_len")
        cached_tokens = self.executor.prepare_request(
            request.request_id, request.prompt_token_ids
        )
        self._validate_cached_token_count(request, cached_tokens)
        request.num_computed_tokens = cached_tokens
        self._requests[request.request_id] = request
        self.scheduler.add_request(request)

    def get_request(self, request_id: str) -> Request:
        return self._requests[request_id]

    @property
    def has_unfinished_requests(self) -> bool:
        return any(not request.finished for request in self._requests.values())

    @property
    def unfinished_request_ids(self) -> tuple[str, ...]:
        return tuple(
            request_id
            for request_id, request in self._requests.items()
            if not request.finished
        )

    @property
    def preemption_events(self) -> int:
        return self._preemption_events

    def _terminate_request(self, request_id: str, reason: str) -> bool:
        if not reason:
            raise ValueError("terminal reason must be non-empty")
        request = self._requests.get(request_id)
        if request is None:
            return False
        if request.finished:
            return False

        # Cancellation/error is applied only between atomic EngineCore steps.
        # First make the request unschedulable, then release physical ownership,
        # then publish the logical terminal state.
        self.scheduler.cancel_request(request)
        self.executor.free(request_id)
        request.mark_finished(reason)
        return True

    def cancel_request(self, request_id: str) -> bool:
        return self._terminate_request(request_id, "cancelled")

    def abort_request(self, request_id: str, reason: str = "error") -> bool:
        return self._terminate_request(request_id, reason)

    def _restore_replayed_item(self, item: WorkItem) -> WorkItem:
        request = item.request
        if not request.cache_rebuild_required:
            return item

        cached_tokens = self.executor.prepare_request(
            request.request_id, request.prompt_token_ids
        )
        self._validate_cached_token_count(request, cached_tokens)
        request.num_computed_tokens = cached_tokens
        request.cache_rebuild_required = False
        return make_work_item(request, max_tokens=item.num_tokens)

    def _preempt(self, victim: Request) -> None:
        self.executor.free(victim.request_id)
        victim.mark_preempted()
        self.scheduler.on_request_preempted(victim)
        self._preemption_events += 1

    def _fit_cache(
        self, decision: ScheduleDecision
    ) -> tuple[ScheduleDecision, tuple[str, ...]]:
        accepted: list[WorkItem] = []
        preempted_ids: list[str] = []

        for original_item in decision.work_items:
            if original_item.request.status is RequestStatus.PREEMPTED:
                continue

            item = self._restore_replayed_item(original_item)
            while not self.executor.ensure_capacity(item.request_id, item.end_pos):
                victim = self.scheduler.select_preemption_victim(item.request)
                if victim is None:
                    raise RuntimeError(
                        f"request {item.request_id!r} cannot reserve KV capacity "
                        "and no eligible preemption victim exists"
                    )
                self._preempt(victim)
                preempted_ids.append(victim.request_id)
                accepted = [
                    accepted_item
                    for accepted_item in accepted
                    if accepted_item.request_id != victim.request_id
                ]

            accepted.append(item)

        decision.work_items = accepted
        return decision, tuple(preempted_ids)

    @staticmethod
    def _to_execution_request(work_items: list[WorkItem]) -> ExecutionRequest:
        return ExecutionRequest(
            items=tuple(
                ExecutionItem(
                    request_id=item.request_id,
                    token_ids=item.token_ids,
                    start_pos=item.start_pos,
                )
                for item in work_items
            )
        )

    def _produce_output_tokens(self, request: Request, logits) -> tuple[int, ...]:
        token_id = self.sampler.sample(logits, request.sampling_params)
        request.append_output(token_id)
        return (token_id,)

    def step(self) -> EngineStepOutput:
        self._step_speculation = []
        decision, preempted_ids = self._fit_cache(self.scheduler.schedule())
        work_trace = tuple(WorkTrace.from_item(item) for item in decision.work_items)
        sampled: dict[str, int] = {}
        committed_by_request: dict[str, tuple[int, ...]] = {}
        finished: list[str] = []

        if decision.work_items:
            execution = self._to_execution_request(decision.work_items)
            result = self.executor.execute(execution)
            expected_ids = {item.request_id for item in decision.work_items}
            if set(result.logits_by_request) != expected_ids:
                raise RuntimeError("executor returned logits for the wrong request set")

            for item in decision.work_items:
                request = item.request
                self.executor.commit_work(
                    item.request_id,
                    start_pos=item.start_pos,
                    token_ids=item.token_ids,
                    prompt_length=len(request.prompt_token_ids),
                )
                request.num_computed_tokens = item.end_pos

                if not item.completes_current_sequence:
                    continue

                committed = self._produce_output_tokens(
                    request, result.logits_by_request[item.request_id]
                )
                if not committed:
                    continue
                committed_by_request[item.request_id] = committed
                sampled[item.request_id] = committed[-1]

                reached_eos = (
                    self.config.eos_token_id is not None
                    and self.config.eos_token_id in committed
                )
                reached_limit = len(request.output_token_ids) >= request.max_new_tokens
                if reached_eos or reached_limit:
                    request.mark_finished("eos" if reached_eos else "length")
                    self.executor.free(request.request_id)
                    self.scheduler.on_request_finished(request)
                    finished.append(request.request_id)

        output = EngineStepOutput(
            step=self._step,
            scheduled_request_ids=tuple(item.request_id for item in decision.work_items),
            admitted_request_ids=tuple(decision.admitted_request_ids),
            sampled_token_ids=sampled,
            finished_request_ids=tuple(finished),
            cache_stats=self.executor.cache_stats(),
            work=work_trace,
            preempted_request_ids=preempted_ids,
            committed_token_ids=committed_by_request,
            speculation=tuple(self._step_speculation),
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


Engine = EngineCore

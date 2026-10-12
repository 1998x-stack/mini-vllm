from __future__ import annotations

from collections import OrderedDict

from mini_vllm.cache.base import KVCacheBackend
from mini_vllm.config import EngineConfig
from mini_vllm.core.request import Request, RequestStatus
from mini_vllm.core.sampler import Sampler
from mini_vllm.core.work import EngineStepOutput, ScheduleDecision, WorkItem, WorkTrace
from mini_vllm.model.runner import ModelRunner
from mini_vllm.scheduler.base import Scheduler, make_work_item


class Engine:
    """Small synchronous serving loop with explicit scheduling/cache boundaries.

    v0.6 adds one policy feedback path: if KV reservation fails, the scheduler
    may select another active request to preempt.  The Engine owns the state
    transition because only it can coordinate Request + Scheduler + KV cache
    without hiding policy inside the storage backend.
    """

    def __init__(
        self,
        *,
        config: EngineConfig,
        scheduler: Scheduler,
        cache: KVCacheBackend,
        model_runner: ModelRunner,
        sampler: Sampler | None = None,
        version: str,
    ) -> None:
        self.config = config
        self.scheduler = scheduler
        self.cache = cache
        self.model_runner = model_runner
        self.sampler = sampler or Sampler()
        self.version = version
        self._requests: OrderedDict[str, Request] = OrderedDict()
        self._step = 0
        self._preemption_events = 0

    def _validate_cached_token_count(self, request: Request, cached_tokens: int) -> None:
        if cached_tokens < 0 or cached_tokens > len(request.prompt_token_ids):
            raise RuntimeError("cache returned invalid cached-token count")

    def add_request(self, request: Request) -> None:
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        if len(request.prompt_token_ids) + request.max_new_tokens > self.config.max_model_len:
            raise ValueError("prompt + max_new_tokens exceeds max_model_len")
        cached_tokens = self.cache.prepare_request(
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
    def preemption_events(self) -> int:
        return self._preemption_events

    def _restore_replayed_item(self, item: WorkItem) -> WorkItem:
        """Reattach any reusable prefix before replaying a preempted request.

        The scheduler built ``item`` assuming replay starts at token 0.  Prefix
        cache preparation may advance the valid KV frontier, so rebuild the item
        with no more token work than the scheduler originally budgeted.
        """

        request = item.request
        if not request.cache_rebuild_required:
            return item

        cached_tokens = self.cache.prepare_request(
            request.request_id, request.prompt_token_ids
        )
        self._validate_cached_token_count(request, cached_tokens)
        request.num_computed_tokens = cached_tokens
        request.cache_rebuild_required = False
        return make_work_item(request, max_tokens=item.num_tokens)

    def _preempt(self, victim: Request) -> None:
        self.cache.free(victim.request_id)
        victim.mark_preempted()
        self.scheduler.on_request_preempted(victim)
        self._preemption_events += 1

    def _fit_cache(
        self, decision: ScheduleDecision
    ) -> tuple[ScheduleDecision, tuple[str, ...]]:
        accepted: list[WorkItem] = []
        preempted_ids: list[str] = []

        for original_item in decision.work_items:
            # A victim selected while processing an earlier WorkItem may still
            # have an entry in this already-materialized scheduling decision.
            if original_item.request.status is RequestStatus.PREEMPTED:
                continue

            item = self._restore_replayed_item(original_item)
            while not self.cache.ensure_capacity(item.request_id, item.end_pos):
                victim = self.scheduler.select_preemption_victim(item.request)
                if victim is None:
                    raise RuntimeError(
                        f"request {item.request_id!r} cannot reserve KV capacity "
                        "and no eligible preemption victim exists"
                    )

                self._preempt(victim)
                preempted_ids.append(victim.request_id)

                # If the victim had already reserved work this step, discard
                # that not-yet-executed WorkItem as well.  Its KV was released.
                accepted = [
                    accepted_item
                    for accepted_item in accepted
                    if accepted_item.request_id != victim.request_id
                ]

            accepted.append(item)

        decision.work_items = accepted
        return decision, tuple(preempted_ids)

    def step(self) -> EngineStepOutput:
        decision, preempted_ids = self._fit_cache(self.scheduler.schedule())
        work_trace = tuple(WorkTrace.from_item(item) for item in decision.work_items)
        sampled: dict[str, int] = {}
        finished: list[str] = []

        if decision.work_items:
            logits_by_request = self.model_runner.execute(decision.work_items, self.cache)
            for item in decision.work_items:
                request = item.request
                self.cache.commit_work(
                    item.request_id,
                    start_pos=item.start_pos,
                    token_ids=item.token_ids,
                    prompt_length=len(request.prompt_token_ids),
                )
                request.num_computed_tokens = item.end_pos

                # Partial prefill and replay of historical output tokens both
                # advance valid KV without producing a new logical output.
                if not item.completes_current_sequence:
                    continue

                token_id = self.sampler.sample(
                    logits_by_request[item.request_id], request.sampling_params
                )
                request.append_output(token_id)
                sampled[item.request_id] = token_id

                reached_eos = (
                    self.config.eos_token_id is not None
                    and token_id == self.config.eos_token_id
                )
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
            work=work_trace,
            preempted_request_ids=preempted_ids,
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

from __future__ import annotations

from collections import deque

from mini_vllm.core.request import Request, RequestStatus
from mini_vllm.core.work import ScheduleDecision
from mini_vllm.scheduler.base import Scheduler, make_work_item


class BudgetedContinuousScheduler(Scheduler):
    """Continuous batching with a per-engine-step token budget.

    Educational policy:

    1. maintain the v0.2 sequence-slot admission rule;
    2. schedule decode requests first, one token each;
    3. spend the remaining budget on prefill requests in FCFS order;
    4. split a long prompt at the remaining budget boundary.

    The scheduler owns policy. It never allocates KV blocks and never runs
    model code.
    """

    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        if max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be > 0")
        if max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be > 0")
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    def _admit_waiting(self) -> list[str]:
        admitted: list[str] = []
        while self.waiting and len(self.running) < self.max_num_seqs:
            request = self.waiting.popleft()
            request.mark_running()
            self.running.append(request)
            admitted.append(request.request_id)
        return admitted

    def schedule(self) -> ScheduleDecision:
        self.running = [r for r in self.running if r.status is not RequestStatus.FINISHED]
        admitted = self._admit_waiting()
        remaining_budget = self.max_num_batched_tokens
        work_items = []

        decode_requests = [request for request in self.running if not request.is_prefill]
        prefill_requests = [request for request in self.running if request.is_prefill]

        for request in decode_requests:
            if remaining_budget == 0:
                break
            item = make_work_item(request, max_tokens=1)
            work_items.append(item)
            remaining_budget -= item.num_tokens

        for request in prefill_requests:
            if remaining_budget == 0:
                break
            item = make_work_item(request, max_tokens=remaining_budget)
            work_items.append(item)
            remaining_budget -= item.num_tokens

        decision = ScheduleDecision(work_items=work_items, admitted_request_ids=admitted)
        if decision.num_scheduled_tokens > self.max_num_batched_tokens:
            raise AssertionError("scheduler exceeded max_num_batched_tokens")
        return decision

    def on_request_finished(self, request: Request) -> None:
        return None

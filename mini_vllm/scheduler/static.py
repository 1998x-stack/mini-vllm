from __future__ import annotations

from collections import deque

from mini_vllm.core.request import Request, RequestStatus
from mini_vllm.core.work import ScheduleDecision
from mini_vllm.scheduler.base import Scheduler, make_work_item


class StaticBatchScheduler(Scheduler):
    """v0.1 baseline: do not admit new requests until the cohort drains."""

    def __init__(self, max_num_seqs: int) -> None:
        self.max_num_seqs = max_num_seqs
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    def _start_new_cohort_if_needed(self) -> list[str]:
        admitted: list[str] = []
        if self.running:
            return admitted
        while self.waiting and len(self.running) < self.max_num_seqs:
            request = self.waiting.popleft()
            request.mark_running()
            self.running.append(request)
            admitted.append(request.request_id)
        return admitted

    def schedule(self) -> ScheduleDecision:
        self.running = [r for r in self.running if r.status is not RequestStatus.FINISHED]
        admitted = self._start_new_cohort_if_needed()
        return ScheduleDecision(work_items=[make_work_item(r) for r in self.running], admitted_request_ids=admitted)

    def on_request_finished(self, request: Request) -> None:
        return None

    def cancel_request(self, request: Request) -> None:
        request_id = request.request_id
        self.waiting = deque(r for r in self.waiting if r.request_id != request_id)
        self.running = [r for r in self.running if r.request_id != request_id]

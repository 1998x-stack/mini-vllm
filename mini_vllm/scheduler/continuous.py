from __future__ import annotations

from collections import deque

from mini_vllm.core.request import Request, RequestStatus
from mini_vllm.core.work import ScheduleDecision
from mini_vllm.scheduler.base import Scheduler, make_work_item


class ContinuousBatchScheduler(Scheduler):
    """v0.2+: refill free sequence slots on every engine iteration."""

    def __init__(self, max_num_seqs: int) -> None:
        self.max_num_seqs = max_num_seqs
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    def schedule(self) -> ScheduleDecision:
        self.running = [r for r in self.running if r.status is not RequestStatus.FINISHED]
        admitted: list[str] = []
        while self.waiting and len(self.running) < self.max_num_seqs:
            request = self.waiting.popleft()
            request.mark_running()
            self.running.append(request)
            admitted.append(request.request_id)
        return ScheduleDecision(
            work_items=[make_work_item(r) for r in self.running],
            admitted_request_ids=admitted,
        )

    def on_request_finished(self, request: Request) -> None:
        return None

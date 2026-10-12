from __future__ import annotations

from mini_vllm.core.request import Request, RequestStatus
from mini_vllm.scheduler.budgeted import BudgetedContinuousScheduler


class PreemptiveBudgetedScheduler(BudgetedContinuousScheduler):
    """v0.6 token-budget scheduler with deterministic recompute preemption.

    Victim policy is intentionally readable:

    * never preempt the request whose allocation is currently blocked;
    * among other active requests, preempt the most recently admitted one;
    * a preempted request is placed at the front of the waiting queue;
    * anti-thrashing guard: it becomes eligible only after at least one request
      that remained active has finished.

    The final rule is deliberately conservative.  It prevents two requests in
    a tiny KV pool from endlessly preempting each other without making durable
    progress, while keeping the fairness mechanism visible in a few lines.
    """

    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        super().__init__(max_num_seqs, max_num_batched_tokens)
        self._admission_counter = 0
        self._admission_order: dict[str, int] = {}
        self._finished_count = 0
        self._resume_after_finished_count: dict[str, int] = {}

    def _eligible_for_admission(self, request: Request) -> bool:
        required = self._resume_after_finished_count.get(request.request_id, 0)
        return self._finished_count >= required

    def _admit_waiting(self) -> list[str]:
        admitted: list[str] = []
        while self.waiting and len(self.running) < self.max_num_seqs:
            scan_count = len(self.waiting)
            admitted_this_scan = False
            for _ in range(scan_count):
                request = self.waiting.popleft()
                if not self._eligible_for_admission(request):
                    self.waiting.append(request)
                    continue

                request.mark_running()
                self.running.append(request)
                self._admission_counter += 1
                self._admission_order[request.request_id] = self._admission_counter
                self._resume_after_finished_count.pop(request.request_id, None)
                admitted.append(request.request_id)
                admitted_this_scan = True
                if len(self.running) >= self.max_num_seqs:
                    break

            if not admitted_this_scan:
                break
        return admitted

    def select_preemption_victim(self, blocked_request: Request) -> Request | None:
        candidates = [
            request
            for request in self.running
            if request.request_id != blocked_request.request_id
            and request.status is RequestStatus.RUNNING
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda request: self._admission_order.get(request.request_id, -1),
        )

    def on_request_preempted(self, request: Request) -> None:
        self.running = [r for r in self.running if r.request_id != request.request_id]
        # Wait for one surviving request completion before retrying.  This is a
        # simple bounded-workload fairness rule, not a production priority model.
        self._resume_after_finished_count[request.request_id] = self._finished_count + 1
        self.waiting.appendleft(request)

    def on_request_finished(self, request: Request) -> None:
        self._finished_count += 1

    @property
    def finished_count(self) -> int:
        return self._finished_count

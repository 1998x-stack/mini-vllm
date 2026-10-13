from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from mini_vllm.cache.base import PrefixReuseHint
from mini_vllm.core.request import Request
from mini_vllm.scheduler.preemptive import PreemptiveBudgetedScheduler


@dataclass(frozen=True, slots=True)
class AdmissionCost:
    request_id: str
    raw_prefill_tokens: int
    reusable_tokens: int
    expected_new_prefill_tokens: int
    bypass_count: int
    fairness_protected: bool


@dataclass(frozen=True, slots=True)
class PrefixSchedulingEvent:
    kind: str
    request_id: str
    raw_prefill_tokens: int
    reusable_tokens: int
    expected_new_prefill_tokens: int
    bypass_count: int
    fairness_protected: bool
    actual_reuse_tokens: int | None = None


@dataclass(slots=True)
class PrefixSchedulingStats:
    hint_refreshes: int = 0
    admissions: int = 0
    warm_preferred_admissions: int = 0
    fairness_overrides: int = 0
    reconciliations: int = 0
    stale_hint_reconciliations: int = 0


class PrefixAwareBudgetedScheduler(PreemptiveBudgetedScheduler):
    """v0.13 admission policy using expected new prefill work.

    Cache hints are advisory facts.  This scheduler never sees physical block
    identities and never mutates ``num_computed_tokens``.  Actual attachment is
    reconciled by EngineCore after admission.
    """

    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        *,
        fairness_bypass_limit: int = 2,
    ) -> None:
        super().__init__(max_num_seqs, max_num_batched_tokens)
        if fairness_bypass_limit <= 0:
            raise ValueError("fairness_bypass_limit must be > 0")
        self.fairness_bypass_limit = fairness_bypass_limit
        self._arrival_counter = 0
        self._arrival_order: dict[str, int] = {}
        self._bypass_count: dict[str, int] = {}
        self._prefix_hints: dict[str, PrefixReuseHint] = {}
        self.stats = PrefixSchedulingStats()
        self.events: list[PrefixSchedulingEvent] = []

    @property
    def defer_cache_prepare(self) -> bool:
        return True

    def add_request(self, request: Request) -> None:
        super().add_request(request)
        if request.request_id not in self._arrival_order:
            self._arrival_counter += 1
            self._arrival_order[request.request_id] = self._arrival_counter
            self._bypass_count[request.request_id] = 0

    def requests_needing_prefix_hints(self) -> tuple[Request, ...]:
        return tuple(self.waiting)

    def update_prefix_reuse_hint(
        self, request: Request, hint: PrefixReuseHint
    ) -> None:
        if hint.prompt_tokens != len(request.prompt_token_ids):
            raise ValueError("prefix hint prompt length does not match request")
        self._prefix_hints[request.request_id] = hint
        self.stats.hint_refreshes += 1

    def current_prefix_reuse_hint(self, request: Request) -> PrefixReuseHint | None:
        return self._prefix_hints.get(request.request_id)

    def _admission_cost(self, request: Request) -> AdmissionCost:
        raw = max(0, len(request.prompt_token_ids) - request.num_computed_tokens)
        hint = self._prefix_hints.get(
            request.request_id,
            PrefixReuseHint(0, 0, len(request.prompt_token_ids)),
        )
        reusable = min(hint.reusable_tokens, raw)
        bypass = self._bypass_count.get(request.request_id, 0)
        return AdmissionCost(
            request_id=request.request_id,
            raw_prefill_tokens=raw,
            reusable_tokens=reusable,
            expected_new_prefill_tokens=raw - reusable,
            bypass_count=bypass,
            fairness_protected=bypass >= self.fairness_bypass_limit,
        )

    def admission_cost(self, request: Request) -> AdmissionCost:
        """Public inspectable cost used by examples/tests."""
        return self._admission_cost(request)

    def _select_waiting_candidate(self, eligible: list[Request]) -> tuple[Request, AdmissionCost]:
        costs = {request.request_id: self._admission_cost(request) for request in eligible}
        protected = [request for request in eligible if costs[request.request_id].fairness_protected]
        if protected:
            selected = min(
                protected,
                key=lambda request: (
                    -costs[request.request_id].bypass_count,
                    self._arrival_order[request.request_id],
                ),
            )
            self.stats.fairness_overrides += 1
            return selected, costs[selected.request_id]

        selected = min(
            eligible,
            key=lambda request: (
                costs[request.request_id].expected_new_prefill_tokens,
                self._arrival_order[request.request_id],
            ),
        )
        oldest = min(eligible, key=lambda request: self._arrival_order[request.request_id])
        if (
            selected.request_id != oldest.request_id
            and costs[selected.request_id].expected_new_prefill_tokens
            < costs[oldest.request_id].expected_new_prefill_tokens
        ):
            self.stats.warm_preferred_admissions += 1
        return selected, costs[selected.request_id]

    def _remove_waiting(self, request_id: str) -> Request:
        selected: Request | None = None
        remaining: deque[Request] = deque()
        while self.waiting:
            request = self.waiting.popleft()
            if request.request_id == request_id and selected is None:
                selected = request
            else:
                remaining.append(request)
        self.waiting = remaining
        if selected is None:
            raise RuntimeError("selected request disappeared from waiting queue")
        return selected

    def _admit_waiting(self) -> list[str]:
        admitted: list[str] = []
        while self.waiting and len(self.running) < self.max_num_seqs:
            eligible = [request for request in self.waiting if self._eligible_for_admission(request)]
            if not eligible:
                break

            selected, cost = self._select_waiting_candidate(eligible)
            for request in eligible:
                if request.request_id != selected.request_id:
                    self._bypass_count[request.request_id] = (
                        self._bypass_count.get(request.request_id, 0) + 1
                    )

            request = self._remove_waiting(selected.request_id)
            request.mark_running()
            self.running.append(request)
            self._admission_counter += 1
            self._admission_order[request.request_id] = self._admission_counter
            self._resume_after_finished_count.pop(request.request_id, None)
            admitted.append(request.request_id)
            self.stats.admissions += 1
            self.events.append(
                PrefixSchedulingEvent(
                    kind="admit",
                    request_id=request.request_id,
                    raw_prefill_tokens=cost.raw_prefill_tokens,
                    reusable_tokens=cost.reusable_tokens,
                    expected_new_prefill_tokens=cost.expected_new_prefill_tokens,
                    bypass_count=cost.bypass_count,
                    fairness_protected=cost.fairness_protected,
                )
            )
        return admitted

    def on_request_cache_prepared(
        self,
        request: Request,
        *,
        estimated_reuse: int,
        actual_reuse: int,
    ) -> None:
        self.stats.reconciliations += 1
        if estimated_reuse != actual_reuse:
            self.stats.stale_hint_reconciliations += 1
        raw = len(request.prompt_token_ids)
        self.events.append(
            PrefixSchedulingEvent(
                kind="reconcile",
                request_id=request.request_id,
                raw_prefill_tokens=raw,
                reusable_tokens=estimated_reuse,
                expected_new_prefill_tokens=max(0, raw - estimated_reuse),
                bypass_count=self._bypass_count.get(request.request_id, 0),
                fairness_protected=(
                    self._bypass_count.get(request.request_id, 0)
                    >= self.fairness_bypass_limit
                ),
                actual_reuse_tokens=actual_reuse,
            )
        )
        self._prefix_hints.pop(request.request_id, None)
        self._bypass_count[request.request_id] = 0

    def on_request_finished(self, request: Request) -> None:
        super().on_request_finished(request)
        self._prefix_hints.pop(request.request_id, None)
        self._bypass_count.pop(request.request_id, None)
        self._arrival_order.pop(request.request_id, None)

    def cancel_request(self, request: Request) -> None:
        super().cancel_request(request)
        self._prefix_hints.pop(request.request_id, None)
        self._bypass_count.pop(request.request_id, None)
        self._arrival_order.pop(request.request_id, None)

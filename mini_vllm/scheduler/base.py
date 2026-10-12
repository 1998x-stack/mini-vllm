from __future__ import annotations

from abc import ABC, abstractmethod

from mini_vllm.core.request import Request
from mini_vllm.core.work import ScheduleDecision, WorkItem


class Scheduler(ABC):
    @abstractmethod
    def add_request(self, request: Request) -> None:
        ...

    @abstractmethod
    def schedule(self) -> ScheduleDecision:
        ...

    @abstractmethod
    def on_request_finished(self, request: Request) -> None:
        ...


def make_work_item(request: Request, *, max_tokens: int | None = None) -> WorkItem:
    """Build the next contiguous model-work slice for a request.

    v0.1-v0.3 call this without `max_tokens`: prefill consumes the entire
    remaining prompt and decode consumes its one uncomputed sampled token.

    v0.4 supplies a positive `max_tokens`, turning prefill into a schedulable
    chunk while preserving the exact same WorkItem/ModelRunner contract.
    """

    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be > 0 when provided")

    all_tokens = request.all_token_ids
    start = request.num_computed_tokens
    logical_end = len(request.prompt_token_ids) if request.is_prefill else len(all_tokens)
    available = logical_end - start
    if available <= 0:
        raise RuntimeError(f"request {request.request_id} has no uncomputed tokens")

    count = available if max_tokens is None else min(available, max_tokens)
    token_ids = tuple(all_tokens[start : start + count])
    return WorkItem(request=request, token_ids=token_ids, start_pos=start)

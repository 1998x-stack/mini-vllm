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


def make_work_item(request: Request) -> WorkItem:
    """Prefill all remaining prompt tokens; decode one new token."""
    all_tokens = request.all_token_ids
    if request.is_prefill:
        token_ids = tuple(all_tokens[request.num_computed_tokens : len(request.prompt_token_ids)])
    else:
        token_ids = tuple(all_tokens[request.num_computed_tokens :])
    if not token_ids:
        raise RuntimeError(f"request {request.request_id} has no uncomputed tokens")
    return WorkItem(request=request, token_ids=token_ids, start_pos=request.num_computed_tokens)

from __future__ import annotations

from abc import ABC, abstractmethod

from mini_vllm.cache.base import PrefixReuseHint
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

    @property
    def defer_cache_prepare(self) -> bool:
        """Whether first cache attachment should wait until admission."""
        return False

    def update_prefix_reuse_hint(
        self, request: Request, hint: PrefixReuseHint
    ) -> None:
        """Receive an advisory cache fact without gaining cache ownership."""
        return None

    def current_prefix_reuse_hint(self, request: Request) -> PrefixReuseHint | None:
        """Return the advisory hint used for the current admission decision."""
        return None

    def requests_needing_prefix_hints(self) -> tuple[Request, ...]:
        """Waiting requests whose advisory reuse estimates may be refreshed."""
        return ()

    def on_request_cache_prepared(
        self,
        request: Request,
        *,
        estimated_reuse: int,
        actual_reuse: int,
    ) -> None:
        """Observe reconciliation between advisory hint and actual attachment."""
        return None

    def cancel_request(self, request: Request) -> None:
        """Remove a registered request from scheduler participation.

        v0.10 keeps cancellation policy centralized in EngineCore. Concrete
        schedulers implement only the queue/list bookkeeping needed to make a
        request unschedulable before EngineCore releases physical KV state.
        """
        raise RuntimeError("this scheduler does not support cancellation")

    def select_preemption_victim(self, blocked_request: Request) -> Request | None:
        """Return an active victim when KV pressure needs memory, if supported."""
        return None

    def on_request_preempted(self, request: Request) -> None:
        """Requeue a victim after Engine has released its active KV state."""
        raise RuntimeError("this scheduler does not support preemption")


def make_work_item(request: Request, *, max_tokens: int | None = None) -> WorkItem:
    """Build the next contiguous model-work slice for a request.

    `all_token_ids` includes previously sampled output tokens.  That detail is
    what makes v0.6 recomputation work: after preemption, prompt and historical
    output tokens are replayed without resampling until execution catches up to
    the logical token history.
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

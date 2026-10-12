from __future__ import annotations

from dataclasses import dataclass, field

from mini_vllm.core.request import Request


@dataclass(frozen=True, slots=True)
class WorkItem:
    request: Request
    token_ids: tuple[int, ...]
    start_pos: int

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def end_pos(self) -> int:
        return self.start_pos + self.num_tokens


@dataclass(slots=True)
class ScheduleDecision:
    work_items: list[WorkItem] = field(default_factory=list)
    admitted_request_ids: list[str] = field(default_factory=list)

    @property
    def num_scheduled_tokens(self) -> int:
        return sum(item.num_tokens for item in self.work_items)


@dataclass(frozen=True, slots=True)
class EngineStepOutput:
    step: int
    scheduled_request_ids: tuple[str, ...]
    admitted_request_ids: tuple[str, ...]
    sampled_token_ids: dict[str, int]
    finished_request_ids: tuple[str, ...]
    cache_stats: dict[str, int | float]

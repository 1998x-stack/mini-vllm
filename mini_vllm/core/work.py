from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mini_vllm.core.request import Request

WorkPhase = Literal["prefill", "decode"]


@dataclass(frozen=True, slots=True)
class WorkItem:
    """A contiguous slice of tokens to compute for one request.

    `WorkItem` deliberately does not mean "one whole request step". Since
    v0.4 a prompt can be split across many work items, while decode still
    naturally contributes a one-token work item.
    """

    request: Request
    token_ids: tuple[int, ...]
    start_pos: int

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("work item must contain at least one token")
        if self.start_pos < 0:
            raise ValueError("start_pos must be >= 0")

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def end_pos(self) -> int:
        return self.start_pos + self.num_tokens

    @property
    def phase(self) -> WorkPhase:
        return "prefill" if self.start_pos < len(self.request.prompt_token_ids) else "decode"

    @property
    def completes_current_sequence(self) -> bool:
        """Whether logits from this work item may be sampled.

        A partial prefill chunk produces valid KV but must not produce an output
        token yet. Final prefill chunks and decode work finish all tokens that
        currently exist in the request and therefore may sample the next token.
        """

        return self.end_pos == len(self.request.all_token_ids)


@dataclass(frozen=True, slots=True)
class WorkTrace:
    request_id: str
    phase: WorkPhase
    start_pos: int
    end_pos: int
    num_tokens: int

    @classmethod
    def from_item(cls, item: WorkItem) -> "WorkTrace":
        return cls(
            request_id=item.request_id,
            phase=item.phase,
            start_pos=item.start_pos,
            end_pos=item.end_pos,
            num_tokens=item.num_tokens,
        )


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
    work: tuple[WorkTrace, ...] = ()

    @property
    def num_scheduled_tokens(self) -> int:
        return sum(item.num_tokens for item in self.work)

    @property
    def num_prefill_tokens(self) -> int:
        return sum(item.num_tokens for item in self.work if item.phase == "prefill")

    @property
    def num_decode_tokens(self) -> int:
        return sum(item.num_tokens for item in self.work if item.phase == "decode")

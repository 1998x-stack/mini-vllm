from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass(frozen=True, slots=True)
class SamplingParams:
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")


@dataclass(slots=True)
class Request:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens < len(self.prompt_token_ids)

    @property
    def num_uncomputed_tokens(self) -> int:
        return len(self.all_token_ids) - self.num_computed_tokens

    @property
    def finished(self) -> bool:
        return self.status is RequestStatus.FINISHED

    def mark_running(self) -> None:
        if self.status is RequestStatus.WAITING:
            self.status = RequestStatus.RUNNING

    def append_output(self, token_id: int) -> None:
        if self.finished:
            raise RuntimeError("cannot append a token to a finished request")
        self.output_token_ids.append(int(token_id))

    def mark_finished(self, reason: str) -> None:
        self.status = RequestStatus.FINISHED
        self.finish_reason = reason

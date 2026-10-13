from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StreamEvent:
    request_id: str


@dataclass(frozen=True, slots=True)
class TokenEvent(StreamEvent):
    token_id: int
    output_index: int


@dataclass(frozen=True, slots=True)
class FinishedEvent(StreamEvent):
    output_token_ids: tuple[int, ...]
    finish_reason: str


@dataclass(frozen=True, slots=True)
class CancelledEvent(StreamEvent):
    output_token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ErrorEvent(StreamEvent):
    error_type: str
    message: str


TerminalEvent = FinishedEvent | CancelledEvent | ErrorEvent


def is_terminal_event(event: StreamEvent) -> bool:
    return isinstance(event, (FinishedEvent, CancelledEvent, ErrorEvent))

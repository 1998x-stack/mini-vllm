from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, slots=True)
class ExecutionItem:
    """Policy-free model work sent across the executor boundary."""

    request_id: str
    token_ids: tuple[int, ...]
    start_pos: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.token_ids:
            raise ValueError("execution item must contain at least one token")
        if self.start_pos < 0:
            raise ValueError("start_pos must be >= 0")

    @property
    def end_pos(self) -> int:
        return self.start_pos + len(self.token_ids)


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """Explicit normal-forward payload owned by EngineCore."""

    items: tuple[ExecutionItem, ...]

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("execution request must contain at least one item")


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Result returned from Executor to EngineCore."""

    logits_by_request: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SpeculativeExecutionRequest:
    request_id: str
    token_ids: tuple[int, ...]
    start_pos: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.token_ids:
            raise ValueError("speculative execution requires at least one token")
        if self.start_pos < 0:
            raise ValueError("start_pos must be >= 0")


@dataclass(frozen=True, slots=True)
class SpeculativeExecutionResult:
    request_id: str
    logits_trace: tuple[torch.Tensor, ...]

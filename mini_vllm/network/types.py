from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mini_vllm.core.request import SamplingParams


class NetworkLifecycle(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    """Validated network-facing generation request.

    Text and sampling fields live only at the API edge. The serving core still
    receives a token-native ``Request``.
    """

    request_id: str
    prompt: str
    max_new_tokens: int
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None

    def sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )

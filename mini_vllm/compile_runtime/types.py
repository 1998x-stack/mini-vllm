from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ExecutionShapeSignature:
    token_count: int
    hidden_dim: int
    output_dim: int
    dtype: str
    device: str

    @classmethod
    def from_tensors(
        cls,
        hidden: torch.Tensor,
        weight: torch.Tensor,
    ) -> "ExecutionShapeSignature":
        if hidden.ndim != 2:
            raise ValueError("hidden must be 2-D [tokens, hidden]")
        if weight.ndim != 2:
            raise ValueError("weight must be 2-D [output, hidden]")
        if hidden.shape[-1] != weight.shape[-1]:
            raise ValueError("hidden/weight dimensions are incompatible")
        return cls(
            token_count=int(hidden.shape[0]),
            hidden_dim=int(hidden.shape[1]),
            output_dim=int(weight.shape[0]),
            dtype=str(hidden.dtype),
            device=str(hidden.device),
        )

    def short(self) -> str:
        return f"t{self.token_count}-h{self.hidden_dim}-o{self.output_dim}-{self.dtype.split('.')[-1]}-{self.device}"


@dataclass(frozen=True, slots=True)
class CompileConfig:
    enabled: bool = True
    backend: str = "eager"
    fullgraph: bool = False
    dynamic: bool = False
    fallback_on_error: bool = True


@dataclass(slots=True)
class CompileStats:
    compile_attempts: int = 0
    compiled_variants: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    recompiles: int = 0
    fallbacks: int = 0
    eager_calls: int = 0


@dataclass(frozen=True, slots=True)
class CompileEvent:
    signature: ExecutionShapeSignature
    mode: str

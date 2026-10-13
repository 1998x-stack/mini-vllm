from __future__ import annotations

from dataclasses import dataclass, field

from mini_vllm.attention import PagedAttentionConfig
from mini_vllm.compile_runtime import CompileConfig
from mini_vllm.config import CacheConfig, EngineConfig, TinyModelConfig


@dataclass(frozen=True, slots=True)
class MultiprocessConfig:
    """Public transport knobs for the educational OS-process boundary."""

    start_method: str = "spawn"
    command_timeout_s: float = 30.0
    shutdown_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        if not self.start_method:
            raise ValueError("start_method must be non-empty")
        if self.command_timeout_s <= 0:
            raise ValueError("command_timeout_s must be > 0")
        if self.shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be > 0")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Stable v1 construction facade for the integrated educational runtime."""

    engine: EngineConfig = field(default_factory=EngineConfig)
    model: TinyModelConfig = field(default_factory=TinyModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    attention: PagedAttentionConfig = field(default_factory=PagedAttentionConfig)
    multiprocess: MultiprocessConfig = field(default_factory=MultiprocessConfig)
    model_artifact: str | None = None
    world_size: int = 2
    max_queue_size: int = 8
    fairness_bypass_limit: int = 2

    def __post_init__(self) -> None:
        if self.model_artifact is not None and not self.model_artifact:
            raise ValueError("model_artifact must be a non-empty path when provided")
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")
        # When an external artifact is configured its validated manifest becomes
        # authoritative for vocabulary size in the v1.2 builder.
        if self.model_artifact is None and self.world_size > self.model.vocab_size:
            raise ValueError("world_size cannot exceed vocabulary size")
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0")
        if self.fairness_bypass_limit <= 0:
            raise ValueError("fairness_bypass_limit must be > 0")

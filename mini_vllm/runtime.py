from __future__ import annotations

from dataclasses import dataclass, field

from mini_vllm.attention import PagedAttentionConfig
from mini_vllm.compile_runtime import CompileConfig
from mini_vllm.config import CacheConfig, EngineConfig, TinyModelConfig


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Stable v1.0 construction facade for the integrated educational runtime.

    Internal storage structures stay hidden.  The facade exposes only the
    serving knobs a learner should reason about when constructing the runtime.
    """

    engine: EngineConfig = field(default_factory=EngineConfig)
    model: TinyModelConfig = field(default_factory=TinyModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    attention: PagedAttentionConfig = field(default_factory=PagedAttentionConfig)
    world_size: int = 2
    max_queue_size: int = 8
    fairness_bypass_limit: int = 2

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")
        if self.world_size > self.model.vocab_size:
            raise ValueError("world_size cannot exceed vocabulary size")
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0")
        if self.fairness_bypass_limit <= 0:
            raise ValueError("fairness_bypass_limit must be > 0")

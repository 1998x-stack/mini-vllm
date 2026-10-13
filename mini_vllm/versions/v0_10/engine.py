from __future__ import annotations

from mini_vllm.async_runtime import AsyncEngine
from mini_vllm.config import CacheConfig, EngineConfig, TinyModelConfig
from mini_vllm.speculative.draft import DraftProvider
from mini_vllm.versions.v0_9.engine import create_engine as create_sync_engine


def create_engine(
    *,
    engine_config: EngineConfig | None = None,
    model_config: TinyModelConfig | None = None,
    cache_config: CacheConfig | None = None,
    draft_provider: DraftProvider | None = None,
    max_speculative_tokens: int | None = None,
    world_size: int = 2,
    max_queue_size: int = 8,
) -> AsyncEngine:
    core = create_sync_engine(
        engine_config=engine_config,
        model_config=model_config,
        cache_config=cache_config,
        draft_provider=draft_provider,
        max_speculative_tokens=max_speculative_tokens,
        world_size=world_size,
    )
    return AsyncEngine(core, max_queue_size=max_queue_size, version="v0.10")

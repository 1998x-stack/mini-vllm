from __future__ import annotations

from typing import TYPE_CHECKING

from mini_vllm.config import CacheConfig, EngineConfig, TinyModelConfig
from mini_vllm.core.engine import Engine

if TYPE_CHECKING:
    from mini_vllm.speculative.draft import DraftProvider


def create_engine(
    version: str = "v0.9",
    *,
    engine_config: EngineConfig | None = None,
    model_config: TinyModelConfig | None = None,
    cache_config: CacheConfig | None = None,
    draft_provider: DraftProvider | None = None,
    max_speculative_tokens: int | None = None,
    world_size: int | None = None,
) -> Engine:
    normalized = version.lower().replace("_", ".")
    kwargs = {
        "engine_config": engine_config,
        "model_config": model_config,
        "cache_config": cache_config,
    }
    builders = {
        "v0.1": "v0_1", "0.1": "v0_1",
        "v0.2": "v0_2", "0.2": "v0_2",
        "v0.3": "v0_3", "0.3": "v0_3",
        "v0.4": "v0_4", "0.4": "v0_4",
        "v0.5": "v0_5", "0.5": "v0_5",
        "v0.6": "v0_6", "0.6": "v0_6",
    }
    if normalized in builders:
        if draft_provider is not None or max_speculative_tokens is not None:
            raise ValueError("speculative options are only supported by v0.7+")
        if world_size is not None:
            raise ValueError("world_size is only supported by v0.9+")
        import importlib
        return importlib.import_module(
            f"mini_vllm.versions.{builders[normalized]}"
        ).create_engine(**kwargs)

    if normalized in {"v0.7", "0.7"}:
        if world_size is not None:
            raise ValueError("world_size is only supported by v0.9+")
        from mini_vllm.versions.v0_7 import create_engine as build
        return build(
            **kwargs,
            draft_provider=draft_provider,
            max_speculative_tokens=max_speculative_tokens,
        )

    if normalized in {"v0.8", "0.8"}:
        if world_size is not None:
            raise ValueError("world_size is only supported by v0.9+")
        from mini_vllm.versions.v0_8 import create_engine as build
        return build(
            **kwargs,
            draft_provider=draft_provider,
            max_speculative_tokens=max_speculative_tokens,
        )

    if normalized in {"v0.9", "0.9"}:
        from mini_vllm.versions.v0_9 import create_engine as build
        return build(
            **kwargs,
            draft_provider=draft_provider,
            max_speculative_tokens=max_speculative_tokens,
            world_size=2 if world_size is None else world_size,
        )

    raise ValueError(f"unknown mini-vLLM version: {version}")


__all__ = ["create_engine"]

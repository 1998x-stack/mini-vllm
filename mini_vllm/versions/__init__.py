from __future__ import annotations

from mini_vllm.config import CacheConfig, EngineConfig, TinyModelConfig
from mini_vllm.core.engine import Engine


def create_engine(
    version: str = "v0.6",
    *,
    engine_config: EngineConfig | None = None,
    model_config: TinyModelConfig | None = None,
    cache_config: CacheConfig | None = None,
) -> Engine:
    normalized = version.lower().replace("_", ".")
    kwargs = {
        "engine_config": engine_config,
        "model_config": model_config,
        "cache_config": cache_config,
    }
    if normalized in {"v0.1", "0.1"}:
        from mini_vllm.versions.v0_1 import create_engine as build
        return build(**kwargs)
    if normalized in {"v0.2", "0.2"}:
        from mini_vllm.versions.v0_2 import create_engine as build
        return build(**kwargs)
    if normalized in {"v0.3", "0.3"}:
        from mini_vllm.versions.v0_3 import create_engine as build
        return build(**kwargs)
    if normalized in {"v0.4", "0.4"}:
        from mini_vllm.versions.v0_4 import create_engine as build
        return build(**kwargs)
    if normalized in {"v0.5", "0.5"}:
        from mini_vllm.versions.v0_5 import create_engine as build
        return build(**kwargs)
    if normalized in {"v0.6", "0.6"}:
        from mini_vllm.versions.v0_6 import create_engine as build
        return build(**kwargs)
    raise ValueError(f"unknown mini-vLLM version: {version}")


__all__ = ["create_engine"]

from __future__ import annotations

from mini_vllm.cache.paged import PagedKVCache
from mini_vllm.config import CacheConfig, EngineConfig, TinyModelConfig
from mini_vllm.core.engine import Engine
from mini_vllm.core.sampler import Sampler
from mini_vllm.model.runner import TinyTransformerRunner
from mini_vllm.model.tiny_transformer import TinyDecoderLM
from mini_vllm.scheduler.budgeted import BudgetedContinuousScheduler


def create_engine(
    *,
    engine_config: EngineConfig | None = None,
    model_config: TinyModelConfig | None = None,
    cache_config: CacheConfig | None = None,
) -> Engine:
    """v0.4 = paged KV + continuous admission + token-budget scheduling."""

    engine_config = engine_config or EngineConfig()
    model_config = model_config or TinyModelConfig()
    cache_config = cache_config or CacheConfig()

    model = TinyDecoderLM(model_config).to(
        device=cache_config.device,
        dtype=cache_config.dtype,
    )
    cache = PagedKVCache(
        num_layers=model_config.n_layers,
        num_blocks=cache_config.num_blocks,
        block_size=cache_config.block_size,
        num_kv_heads=model_config.n_kv_heads,
        head_dim=model_config.head_dim,
        dtype=cache_config.dtype,
        device=cache_config.device,
    )
    scheduler = BudgetedContinuousScheduler(
        max_num_seqs=engine_config.max_num_seqs,
        max_num_batched_tokens=engine_config.max_num_batched_tokens,
    )
    return Engine(
        config=engine_config,
        scheduler=scheduler,
        cache=cache,
        model_runner=TinyTransformerRunner(model),
        sampler=Sampler(),
        version="v0.4",
    )

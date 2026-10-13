from __future__ import annotations

from mini_vllm.async_runtime import AsyncEngine
from mini_vllm.attention import PagedAttentionConfig, build_paged_attention_backend
from mini_vllm.cache.transactional import TransactionalPrefixCachingPagedKVCache
from mini_vllm.compile_runtime import CompileConfig
from mini_vllm.config import CacheConfig, EngineConfig, TinyModelConfig
from mini_vllm.core.sampler import Sampler
from mini_vllm.distributed import (
    DistributedExecutor,
    RankConfig,
    RankedWorker,
    ReferenceCollective,
    WorkerGroup,
)
from mini_vllm.execution import Worker
from mini_vllm.model.compile_runner import CompileAwareVocabParallelTinyRunner
from mini_vllm.model.tiny_transformer import TinyDecoderLM
from mini_vllm.scheduler.prefix_aware import PrefixAwareBudgetedScheduler
from mini_vllm.speculative.draft import DraftProvider, IncrementDraftProvider
from mini_vllm.speculative.engine import SpeculativeEngine


def create_engine(
    *,
    engine_config: EngineConfig | None = None,
    model_config: TinyModelConfig | None = None,
    cache_config: CacheConfig | None = None,
    draft_provider: DraftProvider | None = None,
    max_speculative_tokens: int | None = None,
    world_size: int = 2,
    max_queue_size: int = 8,
    compile_config: CompileConfig | None = None,
    attention_config: PagedAttentionConfig | None = None,
    fairness_bypass_limit: int = 2,
) -> AsyncEngine:
    engine_config = engine_config or EngineConfig()
    model_config = model_config or TinyModelConfig()
    cache_config = cache_config or CacheConfig()
    compile_config = compile_config or CompileConfig()
    attention_config = attention_config or PagedAttentionConfig()

    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    if world_size > model_config.vocab_size:
        raise ValueError("world_size cannot exceed vocabulary size")

    ranked_workers: list[RankedWorker] = []
    for rank in range(world_size):
        rank_config = RankConfig(rank=rank, world_size=world_size)
        attention_backend = build_paged_attention_backend(attention_config)
        model = TinyDecoderLM(
            model_config,
            attention_backend=attention_backend,
        ).to(
            device=cache_config.device,
            dtype=cache_config.dtype,
        )
        cache = TransactionalPrefixCachingPagedKVCache(
            num_layers=model_config.n_layers,
            num_blocks=cache_config.num_blocks,
            block_size=cache_config.block_size,
            num_kv_heads=model_config.n_kv_heads,
            head_dim=model_config.head_dim,
            dtype=cache_config.dtype,
            device=cache_config.device,
        )
        ranked_workers.append(
            RankedWorker(
                rank_config=rank_config,
                worker=Worker(
                    model_runner=CompileAwareVocabParallelTinyRunner(
                        model,
                        rank_config,
                        compile_config=compile_config,
                    ),
                    cache=cache,
                ),
            )
        )

    executor = DistributedExecutor(
        WorkerGroup(tuple(ranked_workers), collective=ReferenceCollective())
    )
    scheduler = PrefixAwareBudgetedScheduler(
        max_num_seqs=engine_config.max_num_seqs,
        max_num_batched_tokens=engine_config.max_num_batched_tokens,
        fairness_bypass_limit=fairness_bypass_limit,
    )
    core = SpeculativeEngine(
        config=engine_config,
        scheduler=scheduler,
        executor=executor,
        sampler=Sampler(),
        version="v0.13",
        draft_provider=draft_provider or IncrementDraftProvider(model_config.vocab_size),
        max_speculative_tokens=(
            engine_config.max_speculative_tokens
            if max_speculative_tokens is None
            else max_speculative_tokens
        ),
    )
    return AsyncEngine(core, max_queue_size=max_queue_size, version="v0.13")

import asyncio

from mini_vllm import (
    CacheConfig,
    CompileConfig,
    EngineConfig,
    PagedAttentionConfig,
    Request,
    TinyModelConfig,
    TokenEvent,
    create_engine,
)


async def run(backend: str):
    engine = create_engine(
        "v0.12",
        engine_config=EngineConfig(
            max_num_seqs=2,
            max_model_len=48,
            max_num_batched_tokens=8,
            max_speculative_tokens=1,
        ),
        model_config=TinyModelConfig(
            vocab_size=24,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            n_layers=1,
            d_ff=32,
            max_position_embeddings=64,
            seed=41,
        ),
        cache_config=CacheConfig(block_size=2, num_blocks=32),
        world_size=2,
        max_queue_size=4,
        max_speculative_tokens=1,
        compile_config=CompileConfig(backend="eager"),
        attention_config=PagedAttentionConfig(backend=backend),
    )
    handle = engine.submit(Request(f"{backend}-R", [1, 2, 3, 4, 5], max_new_tokens=4))
    tokens = []
    async for event in handle:
        if isinstance(event, TokenEvent):
            tokens.append(event.token_id)
    await engine.wait_idle()
    return engine, tokens


async def main() -> None:
    reference_engine, reference = await run("reference")
    vector_engine, vectorized = await run("vectorized")
    print("reference output:", reference)
    print("vectorized output:", vectorized)
    print("same output:", reference == vectorized)
    print("\nrank | backend     | calls | vectorized | reference | fallbacks")
    print("-----+-------------+-------+------------+-----------+----------")
    for ranked in vector_engine.core.executor.worker_group.ranked_workers:
        backend = ranked.worker.model_runner.model.attention_backend
        stats = backend.stats
        print(
            f"{ranked.rank_config.rank:>4} | {type(backend).__name__:<11} | "
            f"{stats.total_calls:>5} | {stats.vectorized_calls:>10} | "
            f"{stats.reference_calls:>9} | {stats.fallbacks:>8}"
        )

    # Keep the reference engine alive through the trace so both executions are
    # directly comparable and visible to learners.
    assert reference_engine.version == "v0.12"
    assert vectorized == reference


if __name__ == "__main__":
    asyncio.run(main())

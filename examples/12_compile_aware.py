import asyncio

from mini_vllm import (
    CacheConfig,
    CompileConfig,
    EngineConfig,
    Request,
    TinyModelConfig,
    TokenEvent,
    create_engine,
)


async def main() -> None:
    engine = create_engine(
        "v0.11",
        engine_config=EngineConfig(max_num_seqs=1, max_model_len=32, max_num_batched_tokens=3, max_speculative_tokens=1),
        model_config=TinyModelConfig(vocab_size=24, d_model=16, n_heads=4, n_kv_heads=2, n_layers=1, d_ff=32, max_position_embeddings=64, seed=17),
        cache_config=CacheConfig(block_size=2, num_blocks=32),
        world_size=2,
        max_speculative_tokens=1,
        compile_config=CompileConfig(backend="eager"),
    )
    handle = engine.submit(Request("compile-demo", [1, 2, 3, 4, 5], max_new_tokens=5))
    streamed = []
    async for event in handle:
        if isinstance(event, TokenEvent):
            streamed.append(event.token_id)
    await engine.wait_idle()

    print("streamed=", streamed)
    print("rank | events")
    for ranked in engine.core.executor.worker_group.ranked_workers:
        cache = ranked.worker.model_runner.compile_cache
        trace = ", ".join(f"{event.signature.short()}:{event.mode}" for event in cache.events)
        print(f"{ranked.rank_config.rank:>4} | {trace}")
        print("     stats=", cache.stats)


if __name__ == "__main__":
    asyncio.run(main())

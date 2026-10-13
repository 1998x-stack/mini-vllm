import asyncio

from mini_vllm import (
    CacheConfig,
    CancelledEvent,
    EngineConfig,
    FinishedEvent,
    Request,
    TinyModelConfig,
    TokenEvent,
    create_engine,
)


async def main() -> None:
    engine = create_engine(
        "v0.10",
        engine_config=EngineConfig(max_num_seqs=2, max_model_len=48, max_num_batched_tokens=8, max_speculative_tokens=1),
        model_config=TinyModelConfig(vocab_size=24, d_model=16, n_heads=4, n_kv_heads=2, n_layers=1, d_ff=32, max_position_embeddings=64, seed=7),
        cache_config=CacheConfig(block_size=2, num_blocks=32),
        world_size=2,
        max_speculative_tokens=1,
        max_queue_size=2,
    )

    print("=== normal async stream ===")
    handle = engine.submit(Request("stream", [1, 2, 3], max_new_tokens=5))
    async for event in handle:
        if isinstance(event, TokenEvent):
            print(f"token index={event.output_index} id={event.token_id}")
        elif isinstance(event, FinishedEvent):
            print(f"finished reason={event.finish_reason} output={event.output_token_ids}")

    print("\n=== cancellation at an EngineCore step boundary ===")
    cancelled = engine.submit(Request("cancel", [4, 5], max_new_tokens=10))
    async for event in cancelled:
        print(type(event).__name__, event)
        if isinstance(event, TokenEvent):
            cancelled.cancel()
        if isinstance(event, CancelledEvent):
            break

    await engine.wait_idle()
    print("driver_running=", engine.driver_running)


if __name__ == "__main__":
    asyncio.run(main())

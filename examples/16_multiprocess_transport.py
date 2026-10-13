from __future__ import annotations

import asyncio
import os

from mini_vllm import (
    CacheConfig,
    CompileConfig,
    EngineConfig,
    MultiprocessConfig,
    PagedAttentionConfig,
    Request,
    TinyModelConfig,
    TokenEvent,
    create_engine,
)


async def main() -> None:
    engine = create_engine(
        "v1.1",
        engine_config=EngineConfig(
            max_num_seqs=2,
            max_model_len=48,
            max_num_batched_tokens=4,
            max_speculative_tokens=2,
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
        compile_config=CompileConfig(backend="eager"),
        attention_config=PagedAttentionConfig(backend="vectorized"),
        multiprocess_config=MultiprocessConfig(start_method="spawn"),
    )

    print("parent pid:", os.getpid())
    print("worker pids:", engine.core.executor.worker_pids)

    try:
        handle = engine.submit(Request("demo", [1, 2, 3, 4, 5], max_new_tokens=5))
        committed: list[int] = []
        async for event in handle:
            if isinstance(event, TokenEvent):
                committed.append(event.token_id)
                print("committed token:", event.token_id)
        await engine.wait_idle()

        snapshot = engine.snapshot()
        print("output:", committed)
        print("boundary:", snapshot.core.execution.boundary)
        print("executor healthy:", snapshot.core.execution.healthy)
        for rank in snapshot.core.execution.ranks:
            print(
                f"rank={rank.rank} pid={rank.process_id} state={rank.worker_state} "
                f"commands={rank.transport_commands_completed}/"
                f"{rank.transport_commands_sent} "
                f"attention={rank.attention_backend} "
                f"vectorized_calls={rank.attention_vectorized_calls} "
                f"compile_attempts={rank.compile_attempts}"
            )
    finally:
        await engine.aclose()
        print("workers alive after close:", engine.core.executor.worker_alive)


if __name__ == "__main__":
    asyncio.run(main())

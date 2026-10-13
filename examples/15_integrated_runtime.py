"""v1.0: one public trace from admission through committed async output."""

from __future__ import annotations

import asyncio

from mini_vllm import (
    CacheConfig,
    CompileConfig,
    EngineConfig,
    PagedAttentionConfig,
    Request,
    RuntimeConfig,
    TinyModelConfig,
    TokenEvent,
    create_engine,
)


async def consume(handle) -> tuple[int, ...]:
    tokens: list[int] = []
    async for event in handle:
        if isinstance(event, TokenEvent):
            tokens.append(event.token_id)
    return tuple(tokens)


async def main() -> None:
    config = RuntimeConfig(
        engine=EngineConfig(
            max_num_seqs=2,
            max_model_len=64,
            max_num_batched_tokens=4,
            max_speculative_tokens=2,
        ),
        model=TinyModelConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            n_layers=1,
            d_ff=32,
            max_position_embeddings=64,
            seed=17,
        ),
        cache=CacheConfig(block_size=2, num_blocks=16),
        compile=CompileConfig(backend="eager"),
        attention=PagedAttentionConfig(backend="vectorized"),
        world_size=2,
        max_queue_size=8,
        fairness_bypass_limit=2,
    )
    engine = create_engine(runtime_config=config)

    shared_prompt = [1, 2, 3, 4, 5, 6, 7]
    seed = engine.submit(Request("seed", shared_prompt, max_new_tokens=2))
    seed_tokens = await consume(seed)
    await engine.wait_idle()

    # cold arrives first, warm arrives second.  Prefix-aware admission can still
    # prefer warm because the seed request left reusable immutable prompt blocks.
    cold = engine.submit(
        Request("cold", [9, 10, 11, 12, 13, 14, 15], max_new_tokens=3)
    )
    warm = engine.submit(Request("warm", shared_prompt, max_new_tokens=3))
    cold_tokens, warm_tokens = await asyncio.gather(consume(cold), consume(warm))
    await engine.wait_idle()

    snapshot = engine.snapshot()

    print("== committed async output ==")
    print("seed:", seed_tokens)
    print("cold:", cold_tokens)
    print("warm:", warm_tokens)

    print("\n== admission / prefix causality ==")
    for event in snapshot.core.scheduler.recent_events:
        if event.request_id not in {"cold", "warm"}:
            continue
        if event.kind == "admit":
            print(
                f"{event.request_id}: raw={event.raw_prefill_tokens} "
                f"reuse_hint={event.reusable_tokens} "
                f"expected_new={event.expected_new_prefill_tokens} "
                f"fairness={event.fairness_protected}"
            )
        elif event.kind == "reconcile":
            print(
                f"{event.request_id}: estimate={event.reusable_tokens} "
                f"actual_attach={event.actual_reuse_tokens}"
            )

    print("\n== latest serving step ==")
    last = snapshot.last_step
    if last is not None:
        print("scheduled:", last.scheduled_request_ids)
        print("work:", last.work)
        print("preempted:", last.preempted_request_ids)
        print("speculation:", last.speculation)
        print("committed:", last.committed_token_ids)

    print("\n== cache ==")
    print(snapshot.core.cache_stats)

    print("\n== rank-local execution paths ==")
    for rank in snapshot.core.execution.ranks:
        print(
            f"rank={rank.rank}/{rank.world_size} vocab={rank.vocab_range} "
            f"attention={rank.attention_backend} "
            f"vectorized_calls={rank.attention_vectorized_calls} "
            f"compile_attempts={rank.compile_attempts} "
            f"compile_hits={rank.compile_cache_hits}"
        )

    print("\n== lifecycle ==")
    for request in snapshot.core.requests:
        print(
            request.request_id,
            request.status,
            f"computed={request.num_computed_tokens}",
            f"output={request.output_tokens}",
            f"preemptions={request.preemption_count}",
        )


if __name__ == "__main__":
    asyncio.run(main())

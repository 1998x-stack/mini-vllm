from __future__ import annotations

from mini_vllm import CacheConfig, EngineConfig, Request, TinyModelConfig, create_engine


def main() -> None:
    engine_config = EngineConfig(
        max_num_seqs=2,
        max_model_len=32,
        max_num_batched_tokens=8,
        max_speculative_tokens=2,
    )
    model_config = TinyModelConfig(
        vocab_size=25,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        d_ff=32,
        max_position_embeddings=64,
        seed=11,
    )
    cache_config = CacheConfig(block_size=2, num_blocks=32)

    reference = create_engine(
        "v0.8",
        engine_config=engine_config,
        model_config=model_config,
        cache_config=cache_config,
    )
    distributed = create_engine(
        "v0.9",
        engine_config=engine_config,
        model_config=model_config,
        cache_config=cache_config,
        world_size=2,
    )

    group = distributed.executor.worker_group
    print("rank | vocab shard")
    print("-----+-----------")
    for entry in group.ranked_workers:
        start, end = entry.worker.model_runner.vocab_range
        print(f"{entry.rank_config.rank:>4} | [{start}:{end})")

    ref_request = Request("demo", [1, 2, 3, 4], max_new_tokens=6)
    dist_request = Request("demo", [1, 2, 3, 4], max_new_tokens=6)
    reference.add_request(ref_request)
    distributed.add_request(dist_request)
    reference.run()
    distributed.run()

    print("\nreference output:", ref_request.output_token_ids)
    print("tp output:       ", dist_request.output_token_ids)
    print("equal:           ", ref_request.output_token_ids == dist_request.output_token_ids)
    print("normal group calls:", group.execute_calls)
    print("spec group calls:  ", group.speculative_execute_calls)
    print("cache topology: replicated per rank")


if __name__ == "__main__":
    main()

"""Visualize v0.4's decode-first token budget and chunked prefill."""

from mini_vllm import CacheConfig, EngineConfig, Request, TinyModelConfig, create_engine


def main() -> None:
    engine = create_engine(
        "v0.4",
        engine_config=EngineConfig(
            max_num_seqs=2,
            max_model_len=32,
            max_num_batched_tokens=4,
        ),
        model_config=TinyModelConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            n_layers=1,
            d_ff=32,
            max_position_embeddings=32,
        ),
        cache_config=CacheConfig(block_size=2, num_blocks=16),
    )

    engine.add_request(Request("decode", [5], max_new_tokens=3))
    engine.step()
    engine.add_request(Request("long-prefill", list(range(10, 18)), max_new_tokens=2))

    print("step | tokens | work                                  | sampled")
    print("-----+--------+---------------------------------------+--------")
    for _ in range(4):
        out = engine.step()
        work = ", ".join(
            f"{w.request_id}:{w.phase}[{w.start_pos}:{w.end_pos}]={w.num_tokens}"
            for w in out.work
        )
        print(
            f"{out.step:>4} | {out.num_scheduled_tokens:>6} | "
            f"{work:<37} | {out.sampled_token_ids}"
        )
        if not engine.has_unfinished_requests:
            break


if __name__ == "__main__":
    main()

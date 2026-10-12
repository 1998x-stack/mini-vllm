"""v0.6 timeline: KV pressure -> preempt -> recompute -> finish."""

from mini_vllm import CacheConfig, EngineConfig, Request, TinyModelConfig, create_engine


engine = create_engine(
    "v0.6",
    engine_config=EngineConfig(
        max_num_seqs=2,
        max_model_len=20,
        max_num_batched_tokens=4,
    ),
    model_config=TinyModelConfig(
        vocab_size=32,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        d_ff=32,
        max_position_embeddings=64,
        seed=17,
    ),
    cache_config=CacheConfig(block_size=2, num_blocks=3),
)

engine.add_request(Request("A", [1, 2, 3], max_new_tokens=2))
engine.add_request(Request("B", [4, 5, 6], max_new_tokens=1))

print("step | work                              | preempt | sampled | finished")
print("-----+-----------------------------------+---------+---------+---------")
for step in engine.run(max_steps=32):
    work = ", ".join(
        f"{item.request_id}:{item.phase}[{item.start_pos}:{item.end_pos}]"
        for item in step.work
    ) or "-"
    preempted = ",".join(step.preempted_request_ids) or "-"
    sampled = ",".join(f"{k}:{v}" for k, v in step.sampled_token_ids.items()) or "-"
    finished = ",".join(step.finished_request_ids) or "-"
    print(f"{step.step:>4} | {work:<33} | {preempted:<7} | {sampled:<7} | {finished}")

print("\nfinal outputs")
for request_id in ("A", "B"):
    request = engine.get_request(request_id)
    print(
        request_id,
        request.output_token_ids,
        "preemptions=",
        request.preemption_count,
    )

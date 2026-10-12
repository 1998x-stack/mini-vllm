from mini_vllm import CacheConfig, EngineConfig, Request, TinyModelConfig, create_engine

engine = create_engine(
    "v0.3",
    engine_config=EngineConfig(max_num_seqs=2, max_model_len=32),
    model_config=TinyModelConfig(vocab_size=32, d_model=32, n_heads=4, n_kv_heads=4, n_layers=2),
    cache_config=CacheConfig(block_size=4, num_blocks=16),
)
engine.add_request(Request("A", [1, 2, 3, 4], max_new_tokens=4))
engine.add_request(Request("B", [7, 8], max_new_tokens=2))
for event in engine.run():
    print(f"step={event.step:02d} scheduled={event.scheduled_request_ids} sampled={event.sampled_token_ids} finished={event.finished_request_ids} cache={event.cache_stats}")
for request_id in ["A", "B"]:
    print(request_id, "->", engine.get_request(request_id).output_token_ids)

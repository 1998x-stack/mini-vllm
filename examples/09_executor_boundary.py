from mini_vllm import CacheConfig, EngineConfig, Request, TinyModelConfig, create_engine

engine = create_engine(
    "v0.8",
    engine_config=EngineConfig(max_num_seqs=1, max_model_len=32, max_num_batched_tokens=8, max_speculative_tokens=2),
    model_config=TinyModelConfig(vocab_size=24, d_model=16, n_heads=4, n_kv_heads=2, n_layers=1, d_ff=32, max_position_embeddings=64),
    cache_config=CacheConfig(block_size=2, num_blocks=16),
)
request = Request("demo", [1, 2, 3], max_new_tokens=4)
engine.add_request(request)
trace = engine.run()

print("EngineCore:", type(engine).__name__)
print("Executor:", type(engine.executor).__name__)
print("Worker:", type(engine.executor.worker).__name__)
print("ModelRunner:", type(engine.model_runner).__name__)
print("KV backend:", type(engine.cache).__name__)
print("normal execute calls:", engine.executor.execute_calls)
print("speculative execute calls:", engine.executor.speculative_execute_calls)
print("output:", request.output_token_ids)
for step in trace:
    print(step.step, step.scheduled_request_ids, step.committed_token_ids)

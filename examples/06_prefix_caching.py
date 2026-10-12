from mini_vllm import CacheConfig, EngineConfig, Request, TinyModelConfig, create_engine
from mini_vllm.cache.prefix import PrefixCachingPagedKVCache


engine = create_engine(
    "v0.5",
    engine_config=EngineConfig(
        max_num_seqs=1,
        max_model_len=32,
        max_num_batched_tokens=8,
    ),
    model_config=TinyModelConfig(
        vocab_size=32,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        d_ff=32,
        max_position_embeddings=64,
        seed=19,
    ),
    cache_config=CacheConfig(block_size=2, num_blocks=12),
)
cache = engine.cache
assert isinstance(cache, PrefixCachingPagedKVCache)

prompt = [1, 2, 3, 4, 5]

print("== cold request ==")
engine.add_request(Request("cold", prompt.copy(), max_new_tokens=2))
for step in engine.run():
    print(
        f"step={step.step:02d} tokens={step.num_scheduled_tokens} "
        f"work={[f'{w.request_id}:{w.phase}[{w.start_pos}:{w.end_pos}]' for w in step.work]} "
        f"cache={step.cache_stats}"
    )
print("cold output:", engine.get_request("cold").output_token_ids)
print("cached blocks after cold finishes:", cache.stats()["cached_blocks"])

print("\n== warm request with identical prefix ==")
warm = Request("warm", prompt.copy(), max_new_tokens=2)
engine.add_request(warm)
print("tokens skipped by prefix cache:", warm.num_computed_tokens)
print("warm block table before compute:", cache.block_table("warm"))
for step in engine.run():
    print(
        f"step={step.step:02d} tokens={step.num_scheduled_tokens} "
        f"work={[f'{w.request_id}:{w.phase}[{w.start_pos}:{w.end_pos}]' for w in step.work]} "
        f"hits={step.cache_stats.get('prefix_hit_tokens', 0)}"
    )
print("warm output:", engine.get_request("warm").output_token_ids)
print("same output:", engine.get_request("warm").output_token_ids == engine.get_request("cold").output_token_ids)

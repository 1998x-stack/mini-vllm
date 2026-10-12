"""v0.7: checkpoint -> draft -> verify -> rollback/commit.

Run with:
    PYTHONPATH=. python examples/08_speculative_decoding.py
"""

from mini_vllm import CacheConfig, EngineConfig, Request, ScriptedDraftProvider, TinyModelConfig, create_engine

engine_cfg = EngineConfig(max_num_seqs=1, max_model_len=32, max_num_batched_tokens=16, max_speculative_tokens=3)
model_cfg = TinyModelConfig(vocab_size=32, d_model=16, n_heads=4, n_kv_heads=2, n_layers=1, d_ff=32, max_position_embeddings=64, seed=71)
cache_cfg = CacheConfig(block_size=2, num_blocks=16)
prompt = [3, 1, 4]

baseline = create_engine("v0.6", engine_config=engine_cfg, model_config=model_cfg, cache_config=cache_cfg)
baseline.add_request(Request("r", prompt.copy(), max_new_tokens=5))
baseline.run()
target = baseline.get_request("r").output_token_ids.copy()

wrong = (target[1] + 1) % model_cfg.vocab_size
draft_provider = ScriptedDraftProvider({"r": [(target[0], wrong, 0), (target[2], target[3])]})
engine = create_engine("v0.7", engine_config=engine_cfg, model_config=model_cfg, cache_config=cache_cfg, draft_provider=draft_provider)
engine.add_request(Request("r", prompt.copy(), max_new_tokens=5))

print("target greedy output:", target)
print("\nstep | checkpoint | draft       | accepted    | rejected    | fallback | bonus | rollback")
print("-----+------------+-------------+-------------+-------------+----------+-------+---------")
for step in engine.run():
    for tx in step.speculation:
        print(f"{step.step:>4} | {tx.checkpoint_pos:>10} | {str(tx.draft_token_ids):<11} | {str(tx.accepted_token_ids):<11} | {str(tx.rejected_token_ids):<11} | {str(tx.fallback_token_id):<8} | {str(tx.bonus_token_id):<5} | {tx.rollback_to_pos}")

actual = engine.get_request("r").output_token_ids
print("\nspeculative output:", actual)
print("matches ordinary greedy:", actual == target)

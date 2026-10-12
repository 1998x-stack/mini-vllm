import torch
from mini_vllm import CacheConfig,EngineConfig,Request,ScriptedDraftProvider,TinyModelConfig,create_engine
from mini_vllm.cache.prefix import PrefixCachingPagedKVCache
from mini_vllm.speculative.engine import SpeculativeEngine

def model_config(seed:int=43)->TinyModelConfig:
    return TinyModelConfig(vocab_size=32,d_model=16,n_heads=4,n_kv_heads=2,n_layers=1,d_ff=32,max_position_embeddings=64,seed=seed)
def engine_config(max_speculative_tokens:int=3)->EngineConfig:
    return EngineConfig(max_num_seqs=1,max_model_len=32,max_num_batched_tokens=16,max_speculative_tokens=max_speculative_tokens)
def cache_config(blocks:int=16)->CacheConfig: return CacheConfig(block_size=2,num_blocks=blocks)
def baseline_tokens(prompt:list[int],max_new_tokens:int,*,seed:int=43)->list[int]:
    e=create_engine("v0.6",engine_config=engine_config(),model_config=model_config(seed),cache_config=cache_config()); e.add_request(Request("r",prompt.copy(),max_new_tokens=max_new_tokens)); e.run(); return e.get_request("r").output_token_ids.copy()
def speculative_engine(batches,*,seed=43,blocks=16,max_speculative_tokens=3):
    e=create_engine("v0.7",engine_config=engine_config(max_speculative_tokens),model_config=model_config(seed),cache_config=cache_config(blocks),draft_provider=ScriptedDraftProvider({"r":batches})); assert isinstance(e,SpeculativeEngine); return e

def test_accept_all_commits_draft_prefix_plus_target_bonus():
    prompt=[1,2,3]; expected=baseline_tokens(prompt,4); e=speculative_engine([tuple(expected[:3])]); e.add_request(Request("r",prompt.copy(),max_new_tokens=4)); step=e.step(); r=step.speculation[0]; assert r.accepted_token_ids==tuple(expected[:3]); assert r.rejected_token_ids==(); assert r.bonus_token_id==expected[3]; assert step.committed_token_ids["r"]==tuple(expected); assert e.get_request("r").output_token_ids==expected; assert e.get_request("r").finished

def test_reject_first_rolls_back_all_temporary_draft_kv():
    prompt=[1,2,3]; expected=baseline_tokens(prompt,3); wrong=(expected[0]+1)%32; e=speculative_engine([(wrong,7,8)]); e.add_request(Request("r",prompt.copy(),max_new_tokens=3)); r=e.step().speculation[0]; assert r.accepted_token_ids==(); assert r.fallback_token_id==expected[0]; assert r.rejected_token_ids==(wrong,7,8); assert r.rollback_to_pos==len(prompt); assert e.get_request("r").num_computed_tokens==len(prompt); assert len(e.cache.block_table("r"))==2

def test_partial_accept_keeps_verified_kv_and_discards_rejected_suffix():
    prompt=[2,4,6]; expected=baseline_tokens(prompt,4,seed=47); wrong=(expected[1]+3)%32; e=speculative_engine([(expected[0],wrong,9)],seed=47); e.add_request(Request("r",prompt.copy(),max_new_tokens=4)); r=e.step().speculation[0]; assert r.accepted_token_ids==(expected[0],); assert r.fallback_token_id==expected[1]; assert r.rejected_token_ids==(wrong,9); assert r.rollback_to_pos==len(prompt)+1; assert e.get_request("r").num_computed_tokens==len(prompt)+1; assert len(e.cache.block_table("r"))==2; e.run(); assert e.get_request("r").output_token_ids==expected

def test_rollback_does_not_mutate_shared_prefix_block():
    prompt=[5,6,7,8,9]; expected=baseline_tokens(prompt,3,seed=53); wrong=(expected[1]+1)%32; e=create_engine("v0.7",engine_config=engine_config(3),model_config=model_config(53),cache_config=cache_config(),draft_provider=ScriptedDraftProvider({"warm":[(expected[0],wrong)]})); assert isinstance(e.cache,PrefixCachingPagedKVCache); e.add_request(Request("seed",prompt.copy(),max_new_tokens=1)); e.run(); warm=Request("warm",prompt.copy(),max_new_tokens=3); e.add_request(warm); shared=e.cache.block_table("warm")[0]; snap=e.cache.keys[:,shared].clone(); r=e.step().speculation[0]; assert r.accepted_token_ids==(expected[0],); assert r.rejected_token_ids==(wrong,); assert torch.equal(e.cache.keys[:,shared],snap); assert e.cache.block_refcount(shared)==1

def test_v07_greedy_output_matches_v06_after_mixed_acceptance():
    prompt=[3,1,4]; expected=baseline_tokens(prompt,6,seed=59); wrong=(expected[2]+5)%32; provider=ScriptedDraftProvider({"r":[(expected[0],expected[1],wrong),(expected[3],expected[4])]}); e=create_engine("v0.7",engine_config=engine_config(3),model_config=model_config(59),cache_config=cache_config(),draft_provider=provider); e.add_request(Request("r",prompt.copy(),max_new_tokens=6)); e.run(); assert e.get_request("r").output_token_ids==expected

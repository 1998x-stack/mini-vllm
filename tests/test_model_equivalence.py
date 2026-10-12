import torch

from mini_vllm.cache.contiguous import ContiguousKVCache
from mini_vllm.cache.paged import PagedKVCache
from mini_vllm.config import TinyModelConfig
from mini_vllm.model.tiny_transformer import TinyDecoderLM


def test_contiguous_and_paged_cache_produce_same_logits() -> None:
    config = TinyModelConfig(vocab_size=32, d_model=16, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=32, max_position_embeddings=32, seed=123)
    model = TinyDecoderLM(config).eval()
    contiguous = ContiguousKVCache(num_layers=config.n_layers, max_model_len=16, num_kv_heads=config.n_kv_heads, head_dim=config.head_dim)
    paged = PagedKVCache(num_layers=config.n_layers, num_blocks=8, block_size=2, num_kv_heads=config.n_kv_heads, head_dim=config.head_dim)
    tokens = (1, 7, 4, 9, 3)
    assert contiguous.ensure_capacity("r", len(tokens))
    assert paged.ensure_capacity("r", len(tokens))
    logits_contiguous = model.forward_chunk(request_id="r", token_ids=tokens, start_pos=0, cache=contiguous)
    logits_paged = model.forward_chunk(request_id="r", token_ids=tokens, start_pos=0, cache=paged)
    assert torch.allclose(logits_contiguous, logits_paged, atol=1e-6)

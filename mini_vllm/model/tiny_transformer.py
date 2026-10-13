from __future__ import annotations

import torch
from torch import nn

from mini_vllm.attention import PagedAttentionBackend, ReferencePagedAttention
from mini_vllm.cache.base import KVCacheBackend
from mini_vllm.config import TinyModelConfig


class TinySelfAttention(nn.Module):
    """Reference decoder attention with externally managed K/V storage."""

    def __init__(
        self,
        config: TinyModelConfig,
        *,
        attention_backend: PagedAttentionBackend | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.attention_backend = attention_backend or ReferencePagedAttention()

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Compatibility helper retained for older educational tests."""
        if self.n_heads == self.n_kv_heads:
            return x
        repeats = self.n_heads // self.n_kv_heads
        return x.repeat_interleave(repeats, dim=1)

    def forward_token(
        self,
        x: torch.Tensor,
        *,
        request_id: str,
        position: int,
        layer_idx: int,
        cache: KVCacheBackend,
    ) -> torch.Tensor:
        q = self.q_proj(x).view(self.n_heads, self.head_dim)
        k = self.k_proj(x).view(self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(self.n_kv_heads, self.head_dim)
        cache.write(layer_idx, request_id, position, k, v)
        context = self.attention_backend.attend(
            q,
            cache=cache,
            layer_idx=layer_idx,
            request_id=request_id,
            length=position + 1,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
        )
        return self.o_proj(context.reshape(-1))


class TinyDecoderLayer(nn.Module):
    def __init__(
        self,
        config: TinyModelConfig,
        *,
        attention_backend: PagedAttentionBackend | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = TinySelfAttention(config, attention_backend=attention_backend)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward_token(
        self,
        x: torch.Tensor,
        *,
        request_id: str,
        position: int,
        layer_idx: int,
        cache: KVCacheBackend,
    ) -> torch.Tensor:
        x = x + self.attn.forward_token(
            self.ln1(x),
            request_id=request_id,
            position=position,
            layer_idx=layer_idx,
            cache=cache,
        )
        return x + self.mlp(self.ln2(x))


class TinyDecoderLM(nn.Module):
    """Tiny decoder-only LM whose attention cache is externally managed."""

    def __init__(
        self,
        config: TinyModelConfig,
        *,
        attention_backend: PagedAttentionBackend | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.attention_backend = attention_backend or ReferencePagedAttention()
        cpu_state = torch.random.get_rng_state()
        torch.manual_seed(config.seed)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.d_model
        )
        self.layers = nn.ModuleList(
            [
                TinyDecoderLayer(
                    config,
                    attention_backend=self.attention_backend,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        torch.random.set_rng_state(cpu_state)

    @torch.no_grad()
    def forward_chunk_hidden_trace(
        self,
        *,
        request_id: str,
        token_ids: tuple[int, ...],
        start_pos: int,
        cache: KVCacheBackend,
    ) -> tuple[torch.Tensor, ...]:
        if not token_ids:
            raise ValueError("token_ids cannot be empty")
        hidden_trace: list[torch.Tensor] = []
        for offset, token_id in enumerate(token_ids):
            position = start_pos + offset
            token = torch.tensor(
                token_id,
                dtype=torch.long,
                device=self.token_embedding.weight.device,
            )
            pos = torch.tensor(
                position,
                dtype=torch.long,
                device=self.token_embedding.weight.device,
            )
            x = self.token_embedding(token) + self.position_embedding(pos)
            for layer_idx, layer in enumerate(self.layers):
                x = layer.forward_token(
                    x,
                    request_id=request_id,
                    position=position,
                    layer_idx=layer_idx,
                    cache=cache,
                )
            hidden_trace.append(self.final_ln(x))
        return tuple(hidden_trace)

    @torch.no_grad()
    def forward_chunk_logits(
        self,
        *,
        request_id: str,
        token_ids: tuple[int, ...],
        start_pos: int,
        cache: KVCacheBackend,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            self.lm_head(hidden)
            for hidden in self.forward_chunk_hidden_trace(
                request_id=request_id,
                token_ids=token_ids,
                start_pos=start_pos,
                cache=cache,
            )
        )

    @torch.no_grad()
    def forward_chunk(
        self,
        *,
        request_id: str,
        token_ids: tuple[int, ...],
        start_pos: int,
        cache: KVCacheBackend,
    ) -> torch.Tensor:
        return self.forward_chunk_logits(
            request_id=request_id,
            token_ids=token_ids,
            start_pos=start_pos,
            cache=cache,
        )[-1]

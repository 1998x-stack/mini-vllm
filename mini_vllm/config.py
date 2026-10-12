from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Serving-level limits."""

    max_num_seqs: int = 4
    max_model_len: int = 128
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be > 0")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be > 0")


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Cache device/dtype for the contiguous baseline."""

    dtype: torch.dtype = torch.float32
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class TinyModelConfig:
    """Small decoder-only model used to make the runtime executable."""

    vocab_size: int = 64
    d_model: int = 32
    n_heads: int = 4
    n_kv_heads: int = 4
    n_layers: int = 2
    d_ff: int = 64
    max_position_embeddings: int = 256
    seed: int = 7

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.vocab_size <= 2:
            raise ValueError("vocab_size must be > 2")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

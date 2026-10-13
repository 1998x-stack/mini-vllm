from __future__ import annotations

from dataclasses import dataclass

import torch

from mini_vllm.artifacts.types import ModelArtifactSpec
from mini_vllm.attention import PagedAttentionConfig
from mini_vllm.compile_runtime import CompileConfig
from mini_vllm.config import CacheConfig, TinyModelConfig


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """Serializable recipe used by a child process to build one rank-local Worker."""

    rank: int
    world_size: int
    model_config: TinyModelConfig
    cache_config: CacheConfig
    compile_config: CompileConfig
    attention_config: PagedAttentionConfig
    artifact_spec: ModelArtifactSpec | None = None


@dataclass(frozen=True, slots=True)
class TensorPayload:
    """Explicit CPU tensor wire format used instead of sharing live tensor storage."""

    shape: tuple[int, ...]
    dtype: str
    values: tuple[float, ...]

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorPayload":
        cpu = tensor.detach().cpu().contiguous()
        return cls(
            shape=tuple(int(value) for value in cpu.shape),
            dtype=str(cpu.dtype).split(".")[-1],
            values=tuple(float(value) for value in cpu.reshape(-1).tolist()),
        )

    def to_tensor(self) -> torch.Tensor:
        dtype = getattr(torch, self.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unsupported transported dtype: {self.dtype}")
        return torch.tensor(self.values, dtype=dtype).reshape(self.shape)


@dataclass(frozen=True, slots=True)
class TransportCommand:
    correlation_id: int
    operation: str
    payload: object = None


@dataclass(frozen=True, slots=True)
class TransportResponse:
    correlation_id: int
    ok: bool
    payload: object = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class CheckpointHandle:
    checkpoint_id: int
    request_id: str
    num_tokens: int


@dataclass(frozen=True, slots=True)
class MultiprocessCacheCheckpoint:
    request_id: str
    num_tokens: int
    rank_handles: tuple[CheckpointHandle, ...]


@dataclass(frozen=True, slots=True)
class SerializableExecutionResult:
    logits_by_request: dict[str, TensorPayload]


@dataclass(frozen=True, slots=True)
class SerializableSpeculativeResult:
    request_id: str
    logits_trace: tuple[TensorPayload, ...]


@dataclass(frozen=True, slots=True)
class WorkerExecutionObservation:
    rank: int
    world_size: int
    process_id: int
    vocab_range: tuple[int, int] | None
    attention_backend: str
    attention_total_calls: int
    attention_reference_calls: int
    attention_vectorized_calls: int
    attention_fallbacks: int
    compile_enabled: bool
    compile_backend: str
    compile_attempts: int
    compile_cache_hits: int
    compile_cache_misses: int
    compile_fallbacks: int
    eager_projection_calls: int
    artifact_identity: str | None = None
    tokenizer_identity: str | None = None
    artifact_architecture: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerReady:
    observation: WorkerExecutionObservation


@dataclass(frozen=True, slots=True)
class TransportWorkerSnapshot:
    rank: int
    world_size: int
    process_id: int
    lifecycle_state: str
    commands_sent: int
    commands_completed: int
    remote_errors: int
    transport_failures: int
    observation: WorkerExecutionObservation

from __future__ import annotations

from dataclasses import dataclass

from mini_vllm.config import TinyModelConfig


ARTIFACT_FORMAT_VERSION = 1
ARTIFACT_ARCHITECTURE = "tiny-decoder-lm"
TOKENIZER_FORMAT_VERSION = 1
TOKENIZER_TYPE = "whitespace"


class ArtifactValidationError(ValueError):
    """External artifact data could not be promoted to trusted runtime state."""


@dataclass(frozen=True, slots=True)
class ModelArtifactIdentity:
    """Stable logical identity for one validated model/tokenizer artifact."""

    format_version: int
    architecture: str
    model_digest: str
    weights_digest: str
    tokenizer_digest: str
    digest: str


@dataclass(frozen=True, slots=True)
class ModelArtifactSpec:
    """Serializable recipe sent to worker processes after parent validation."""

    root: str
    identity: ModelArtifactIdentity
    model_config: TinyModelConfig
    weight_dtype: str
    tokenizer_compatibility_id: str


@dataclass(frozen=True, slots=True)
class WeightMapping:
    external_name: str
    runtime_name: str
    expected_shape: tuple[int, ...]
    expected_dtype: str


@dataclass(frozen=True, slots=True)
class TokenizerArtifact:
    format_version: int
    tokenizer_type: str
    compatibility_id: str
    tokens: tuple[str, ...]
    special_tokens: tuple[tuple[str, int], ...]
    digest: str

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def special_token_map(self) -> dict[str, int]:
        return dict(self.special_tokens)


@dataclass(frozen=True, slots=True)
class ValidatedModelArtifact:
    """Validated metadata only; checkpoint tensors are intentionally not retained."""

    spec: ModelArtifactSpec
    tokenizer: TokenizerArtifact
    weight_names: tuple[str, ...]

    @property
    def identity(self) -> ModelArtifactIdentity:
        return self.spec.identity

    @property
    def model_config(self) -> TinyModelConfig:
        return self.spec.model_config

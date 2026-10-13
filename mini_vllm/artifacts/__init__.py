from __future__ import annotations

from mini_vllm.artifacts.tokenizer import WhitespaceTokenizer
from mini_vllm.artifacts.types import (
    ARTIFACT_ARCHITECTURE,
    ARTIFACT_FORMAT_VERSION,
    ArtifactValidationError,
    ModelArtifactIdentity,
    ModelArtifactSpec,
    TokenizerArtifact,
    ValidatedModelArtifact,
    WeightMapping,
)


def build_weight_mappings(*args, **kwargs):
    from mini_vllm.artifacts.loader import build_weight_mappings as implementation

    return implementation(*args, **kwargs)


def export_tiny_model_artifact(*args, **kwargs):
    from mini_vllm.artifacts.loader import export_tiny_model_artifact as implementation

    return implementation(*args, **kwargs)


def load_artifact_weights(*args, **kwargs):
    from mini_vllm.artifacts.loader import load_artifact_weights as implementation

    return implementation(*args, **kwargs)


def load_model_artifact(*args, **kwargs):
    from mini_vllm.artifacts.loader import load_model_artifact as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "ARTIFACT_ARCHITECTURE",
    "ARTIFACT_FORMAT_VERSION",
    "ArtifactValidationError",
    "ModelArtifactIdentity",
    "ModelArtifactSpec",
    "TokenizerArtifact",
    "ValidatedModelArtifact",
    "WeightMapping",
    "WhitespaceTokenizer",
    "build_weight_mappings",
    "export_tiny_model_artifact",
    "load_artifact_weights",
    "load_model_artifact",
]

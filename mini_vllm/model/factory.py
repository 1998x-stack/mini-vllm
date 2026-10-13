from __future__ import annotations

import torch

from mini_vllm.artifacts.loader import load_artifact_weights
from mini_vllm.artifacts.types import ValidatedModelArtifact
from mini_vllm.attention import PagedAttentionBackend
from mini_vllm.model.tiny_transformer import TinyDecoderLM


def build_model_from_artifact(
    artifact: ValidatedModelArtifact,
    *,
    attention_backend: PagedAttentionBackend,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> TinyDecoderLM:
    """Construct a runtime module and copy validated checkpoint tensors into it."""
    weights = load_artifact_weights(
        artifact.spec.root,
        artifact.model_config,
        artifact.spec.weight_dtype,
    )
    model = TinyDecoderLM(
        artifact.model_config,
        attention_backend=attention_backend,
    )
    runtime_state = model.state_dict()
    if set(runtime_state) != set(weights):
        raise RuntimeError("validated artifact/runtime state coverage changed unexpectedly")
    with torch.no_grad():
        for name, target in runtime_state.items():
            source = weights[name]
            target.copy_(source)
    model = model.to(device=device, dtype=dtype)
    model.artifact_identity = artifact.identity.digest
    model.tokenizer_identity = artifact.tokenizer.digest
    model.artifact_architecture = artifact.identity.architecture
    return model

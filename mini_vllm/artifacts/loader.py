from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from mini_vllm.artifacts.types import (
    ARTIFACT_ARCHITECTURE,
    ARTIFACT_FORMAT_VERSION,
    TOKENIZER_FORMAT_VERSION,
    TOKENIZER_TYPE,
    ArtifactValidationError,
    ModelArtifactIdentity,
    ModelArtifactSpec,
    TokenizerArtifact,
    ValidatedModelArtifact,
    WeightMapping,
)
from mini_vllm.config import TinyModelConfig
from mini_vllm.model.tiny_transformer import TinyDecoderLM


_MODEL_FIELDS = (
    "vocab_size",
    "d_model",
    "n_heads",
    "n_kv_heads",
    "n_layers",
    "d_ff",
    "max_position_embeddings",
    "seed",
)
_CONFIG_FIELDS = {
    "format_version",
    "architecture",
    "weight_dtype",
    "tokenizer_compatibility_id",
    "model",
}
_TOKENIZER_FIELDS = {
    "format_version",
    "type",
    "compatibility_id",
    "tokens",
    "special_tokens",
}
_SUPPORTED_WEIGHT_DTYPES = {"float32": torch.float32}


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactValidationError(f"missing {label} file: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{label} must contain one JSON object")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _parse_model_config(raw: dict[str, Any]) -> tuple[TinyModelConfig, str, str, str]:
    missing = sorted(_CONFIG_FIELDS - raw.keys())
    unexpected = sorted(raw.keys() - _CONFIG_FIELDS)
    if missing:
        raise ArtifactValidationError(f"model_config missing field(s): {', '.join(missing)}")
    if unexpected:
        raise ArtifactValidationError(
            f"model_config has unexpected field(s): {', '.join(unexpected)}"
        )
    if raw["format_version"] != ARTIFACT_FORMAT_VERSION:
        raise ArtifactValidationError(
            f"unsupported artifact format version: {raw['format_version']!r}"
        )
    if raw["architecture"] != ARTIFACT_ARCHITECTURE:
        raise ArtifactValidationError(
            f"unsupported architecture: {raw['architecture']!r}"
        )
    weight_dtype = raw["weight_dtype"]
    if weight_dtype not in _SUPPORTED_WEIGHT_DTYPES:
        raise ArtifactValidationError(f"unsupported weight dtype: {weight_dtype!r}")
    compatibility_id = raw["tokenizer_compatibility_id"]
    if not isinstance(compatibility_id, str) or not compatibility_id:
        raise ArtifactValidationError("tokenizer_compatibility_id must be non-empty str")

    model = raw["model"]
    if not isinstance(model, dict):
        raise ArtifactValidationError("model_config.model must be an object")
    missing_model = [field for field in _MODEL_FIELDS if field not in model]
    unexpected_model = sorted(set(model) - set(_MODEL_FIELDS))
    if missing_model:
        raise ArtifactValidationError(
            f"model_config.model missing field(s): {', '.join(missing_model)}"
        )
    if unexpected_model:
        raise ArtifactValidationError(
            f"model_config.model has unexpected field(s): {', '.join(unexpected_model)}"
        )
    for field in _MODEL_FIELDS:
        value = model[field]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ArtifactValidationError(f"model field {field} must be int")
        if field != "seed" and value <= 0:
            raise ArtifactValidationError(f"model field {field} must be > 0")
    try:
        config = TinyModelConfig(**{field: model[field] for field in _MODEL_FIELDS})
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"invalid model dimensions: {exc}") from exc
    return config, weight_dtype, compatibility_id, _digest_json(raw)


def _parse_tokenizer(raw: dict[str, Any]) -> TokenizerArtifact:
    missing = sorted(_TOKENIZER_FIELDS - raw.keys())
    unexpected = sorted(raw.keys() - _TOKENIZER_FIELDS)
    if missing:
        raise ArtifactValidationError(f"tokenizer missing field(s): {', '.join(missing)}")
    if unexpected:
        raise ArtifactValidationError(
            f"tokenizer has unexpected field(s): {', '.join(unexpected)}"
        )
    if raw["format_version"] != TOKENIZER_FORMAT_VERSION:
        raise ArtifactValidationError(
            f"unsupported tokenizer format version: {raw['format_version']!r}"
        )
    if raw["type"] != TOKENIZER_TYPE:
        raise ArtifactValidationError(f"unsupported tokenizer type: {raw['type']!r}")
    compatibility_id = raw["compatibility_id"]
    if not isinstance(compatibility_id, str) or not compatibility_id:
        raise ArtifactValidationError("tokenizer compatibility_id must be non-empty str")
    tokens = raw["tokens"]
    if not isinstance(tokens, list) or not tokens:
        raise ArtifactValidationError("tokenizer tokens must be a non-empty list")
    if any(not isinstance(token, str) or not token for token in tokens):
        raise ArtifactValidationError("every tokenizer token must be non-empty str")
    if len(set(tokens)) != len(tokens):
        raise ArtifactValidationError("tokenizer tokens must be unique")
    special = raw["special_tokens"]
    if not isinstance(special, dict) or "unk" not in special:
        raise ArtifactValidationError("special_tokens must be an object containing 'unk'")
    normalized_special: list[tuple[str, int]] = []
    for name, token_id in sorted(special.items()):
        if not isinstance(name, str) or not name:
            raise ArtifactValidationError("special token names must be non-empty str")
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise ArtifactValidationError("special token ids must be integers")
        if token_id < 0 or token_id >= len(tokens):
            raise ArtifactValidationError(
                f"special token {name!r} id {token_id} outside tokenizer vocabulary"
            )
        normalized_special.append((name, token_id))
    return TokenizerArtifact(
        format_version=TOKENIZER_FORMAT_VERSION,
        tokenizer_type=TOKENIZER_TYPE,
        compatibility_id=compatibility_id,
        tokens=tuple(tokens),
        special_tokens=tuple(normalized_special),
        digest=_digest_json(raw),
    )


def build_weight_mappings(config: TinyModelConfig) -> tuple[WeightMapping, ...]:
    """Return the explicit external-name → runtime-name contract."""
    model = TinyDecoderLM(config)
    mappings = tuple(
        WeightMapping(
            external_name=name,
            runtime_name=name,
            expected_shape=tuple(int(value) for value in tensor.shape),
            expected_dtype="float32",
        )
        for name, tensor in model.state_dict().items()
    )
    runtime_names = [mapping.runtime_name for mapping in mappings]
    if len(runtime_names) != len(set(runtime_names)):
        raise RuntimeError("duplicate runtime weight mapping")
    return mappings


def _load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise ArtifactValidationError(f"missing weights file: {path.name}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ArtifactValidationError(f"unable to load weights.pt safely: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError("weights.pt must contain a name → tensor mapping")
    for name, value in payload.items():
        if not isinstance(name, str) or not name:
            raise ArtifactValidationError("weight names must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise ArtifactValidationError(f"non-tensor checkpoint entry rejected: {name}")
    return payload


def _weights_digest(weights: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(weights):
        tensor = weights[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(int(value) for value in tensor.shape)).encode("ascii"))
        raw = tensor.view(torch.uint8).reshape(-1).tolist()
        digest.update(bytes(raw))
    return digest.hexdigest()


def load_artifact_weights(
    root: str | Path,
    config: TinyModelConfig,
    weight_dtype: str,
) -> dict[str, torch.Tensor]:
    root_path = Path(root)
    weights = _load_checkpoint(root_path / "weights.pt")
    mappings = build_weight_mappings(config)
    expected = {mapping.external_name: mapping for mapping in mappings}
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(set(weights) - set(expected))
    if missing:
        raise ArtifactValidationError(f"missing required tensor(s): {', '.join(missing)}")
    if unexpected:
        raise ArtifactValidationError(f"unexpected tensor(s): {', '.join(unexpected)}")
    expected_torch_dtype = _SUPPORTED_WEIGHT_DTYPES[weight_dtype]
    for name, mapping in expected.items():
        tensor = weights[name]
        shape = tuple(int(value) for value in tensor.shape)
        if shape != mapping.expected_shape:
            raise ArtifactValidationError(
                f"shape mismatch for {name}: expected {mapping.expected_shape}, got {shape}"
            )
        if tensor.dtype != expected_torch_dtype:
            raise ArtifactValidationError(
                f"dtype mismatch for {name}: expected {weight_dtype}, got {str(tensor.dtype).split('.')[-1]}"
            )
    return weights


def load_model_artifact(
    root: str | Path,
    *,
    expected_identity: ModelArtifactIdentity | None = None,
) -> ValidatedModelArtifact:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ArtifactValidationError(f"model artifact directory does not exist: {root_path}")
    raw_config = _read_json(root_path / "model_config.json", label="model_config")
    config, weight_dtype, tokenizer_compatibility_id, model_digest = _parse_model_config(
        raw_config
    )
    raw_tokenizer = _read_json(root_path / "tokenizer.json", label="tokenizer")
    tokenizer = _parse_tokenizer(raw_tokenizer)
    if tokenizer.compatibility_id != tokenizer_compatibility_id:
        raise ArtifactValidationError(
            "tokenizer compatibility id does not match model_config"
        )
    if tokenizer.vocab_size != config.vocab_size:
        raise ArtifactValidationError(
            f"tokenizer/model vocabulary mismatch: {tokenizer.vocab_size} != {config.vocab_size}"
        )
    weights = load_artifact_weights(root_path, config, weight_dtype)
    weights_digest = _weights_digest(weights)
    identity_payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "architecture": ARTIFACT_ARCHITECTURE,
        "model_digest": model_digest,
        "weights_digest": weights_digest,
        "tokenizer_digest": tokenizer.digest,
    }
    identity = ModelArtifactIdentity(
        format_version=ARTIFACT_FORMAT_VERSION,
        architecture=ARTIFACT_ARCHITECTURE,
        model_digest=model_digest,
        weights_digest=weights_digest,
        tokenizer_digest=tokenizer.digest,
        digest=_digest_json(identity_payload),
    )
    if expected_identity is not None and identity != expected_identity:
        raise ArtifactValidationError(
            f"artifact identity mismatch: expected {expected_identity.digest}, got {identity.digest}"
        )
    spec = ModelArtifactSpec(
        root=str(root_path),
        identity=identity,
        model_config=config,
        weight_dtype=weight_dtype,
        tokenizer_compatibility_id=tokenizer_compatibility_id,
    )
    return ValidatedModelArtifact(
        spec=spec,
        tokenizer=tokenizer,
        weight_names=tuple(sorted(weights)),
    )


def export_tiny_model_artifact(
    root: str | Path,
    config: TinyModelConfig | None = None,
    *,
    tokens: list[str] | tuple[str, ...] | None = None,
    compatibility_id: str = "mini-whitespace-v1",
) -> ValidatedModelArtifact:
    """Write the deterministic teaching model in the v1.2 local artifact format."""
    config = config or TinyModelConfig()
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    if tokens is None:
        token_values = ["<unk>"] + [f"tok{index}" for index in range(1, config.vocab_size)]
    else:
        token_values = list(tokens)
    model_config = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "architecture": ARTIFACT_ARCHITECTURE,
        "weight_dtype": "float32",
        "tokenizer_compatibility_id": compatibility_id,
        "model": {field: getattr(config, field) for field in _MODEL_FIELDS},
    }
    tokenizer = {
        "format_version": TOKENIZER_FORMAT_VERSION,
        "type": TOKENIZER_TYPE,
        "compatibility_id": compatibility_id,
        "tokens": token_values,
        "special_tokens": {"unk": 0},
    }
    (root_path / "model_config.json").write_text(
        json.dumps(model_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root_path / "tokenizer.json").write_text(
        json.dumps(tokenizer, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model = TinyDecoderLM(config)
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    torch.save(state, root_path / "weights.pt")
    return load_model_artifact(root_path)

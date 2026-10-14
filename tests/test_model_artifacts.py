import asyncio
import json
import multiprocessing as mp
from pathlib import Path

import pytest
import torch

from mini_vllm import (
    ArtifactMultiprocessExecutor,
    ArtifactValidationError,
    CacheConfig,
    CompileConfig,
    EngineConfig,
    MultiprocessConfig,
    PagedAttentionConfig,
    Request,
    TinyModelConfig,
    TokenEvent,
    WhitespaceTokenizer,
    create_engine,
    export_tiny_model_artifact,
    load_model_artifact,
)
from mini_vllm.attention import ReferencePagedAttention
from mini_vllm.model.factory import build_model_from_artifact
from mini_vllm.multiprocess import WorkerRemoteError, WorkerSpec
from mini_vllm.versions.v1_2 import ArtifactAsyncEngine


FAST_START_METHOD = "fork" if "fork" in mp.get_all_start_methods() else "spawn"


def model_config() -> TinyModelConfig:
    return TinyModelConfig(
        vocab_size=24,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        d_ff=32,
        max_position_embeddings=64,
        seed=41,
    )


def cache_config() -> CacheConfig:
    return CacheConfig(block_size=2, num_blocks=32)


def artifact_dir(tmp_path: Path) -> Path:
    path = tmp_path / "artifact"
    export_tiny_model_artifact(path, model_config())
    return path


async def collect(engine, request_id="R", prompt=None, max_new_tokens=5):
    handle = engine.submit(
        Request(request_id, prompt or [1, 2, 3, 4, 5], max_new_tokens=max_new_tokens)
    )
    tokens = []
    async for event in handle:
        if isinstance(event, TokenEvent):
            tokens.append(event.token_id)
    await engine.wait_idle()
    return tokens


def test_manifest_identity_and_tokenizer_are_deterministic(tmp_path: Path) -> None:
    path = artifact_dir(tmp_path)
    first = load_model_artifact(path)
    second = load_model_artifact(path)
    assert first.identity == second.identity
    assert first.model_config == model_config()
    assert len(first.weight_names) > 0

    tokenizer = WhitespaceTokenizer(first.tokenizer)
    assert tokenizer.encode("tok1 tok2 missing") == [1, 2, 0]
    assert tokenizer.decode([1, 2, 0]) == "tok1 tok2 <unk>"
    with pytest.raises(ValueError, match="outside tokenizer vocabulary"):
        tokenizer.decode([999])


def test_manifest_failures_are_rejected_before_runtime(tmp_path: Path) -> None:
    path = artifact_dir(tmp_path)
    config_path = path / "model_config.json"
    raw = json.loads(config_path.read_text())
    del raw["architecture"]
    config_path.write_text(json.dumps(raw))
    with pytest.raises(ArtifactValidationError, match="missing field"):
        load_model_artifact(path)

    export_tiny_model_artifact(path, model_config())
    raw = json.loads(config_path.read_text())
    raw["format_version"] = 999
    config_path.write_text(json.dumps(raw))
    with pytest.raises(ArtifactValidationError, match="unsupported artifact format"):
        load_model_artifact(path)

    export_tiny_model_artifact(path, model_config())
    (path / "tokenizer.json").unlink()
    with pytest.raises(ArtifactValidationError, match="missing tokenizer"):
        load_model_artifact(path)


def test_tokenizer_model_vocab_and_special_metadata_must_match(tmp_path: Path) -> None:
    path = artifact_dir(tmp_path)
    tokenizer_path = path / "tokenizer.json"
    raw = json.loads(tokenizer_path.read_text())
    raw["tokens"] = raw["tokens"][:-1]
    tokenizer_path.write_text(json.dumps(raw))
    with pytest.raises(ArtifactValidationError, match="vocabulary mismatch"):
        load_model_artifact(path)

    export_tiny_model_artifact(path, model_config())
    raw = json.loads(tokenizer_path.read_text())
    raw["special_tokens"] = {"unk": 999}
    tokenizer_path.write_text(json.dumps(raw))
    with pytest.raises(ArtifactValidationError, match="outside tokenizer vocabulary"):
        load_model_artifact(path)


def _load_weights(path: Path):
    return torch.load(path / "weights.pt", map_location="cpu", weights_only=True)


def test_weight_coverage_shape_dtype_and_payload_are_strict(tmp_path: Path) -> None:
    path = artifact_dir(tmp_path)
    weights = _load_weights(path)
    name = next(iter(weights))

    missing = dict(weights)
    del missing[name]
    torch.save(missing, path / "weights.pt")
    with pytest.raises(ArtifactValidationError, match="missing required tensor"):
        load_model_artifact(path)

    export_tiny_model_artifact(path, model_config())
    unexpected = _load_weights(path)
    unexpected["unexpected.weight"] = torch.zeros(1)
    torch.save(unexpected, path / "weights.pt")
    with pytest.raises(ArtifactValidationError, match="unexpected tensor"):
        load_model_artifact(path)

    export_tiny_model_artifact(path, model_config())
    wrong_shape = _load_weights(path)
    name = next(iter(wrong_shape))
    wrong_shape[name] = wrong_shape[name].reshape(-1)[:1]
    torch.save(wrong_shape, path / "weights.pt")
    with pytest.raises(ArtifactValidationError, match="shape mismatch"):
        load_model_artifact(path)

    export_tiny_model_artifact(path, model_config())
    wrong_dtype = _load_weights(path)
    name = next(iter(wrong_dtype))
    wrong_dtype[name] = wrong_dtype[name].to(torch.float64)
    torch.save(wrong_dtype, path / "weights.pt")
    with pytest.raises(ArtifactValidationError, match="dtype mismatch"):
        load_model_artifact(path)

    export_tiny_model_artifact(path, model_config())
    non_tensor = _load_weights(path)
    name = next(iter(non_tensor))
    non_tensor[name] = "not a tensor"
    torch.save(non_tensor, path / "weights.pt")
    with pytest.raises(ArtifactValidationError, match="non-tensor"):
        load_model_artifact(path)


def test_runtime_model_owns_copied_parameters(tmp_path: Path) -> None:
    path = artifact_dir(tmp_path)
    artifact = load_model_artifact(path)
    model = build_model_from_artifact(
        artifact,
        attention_backend=ReferencePagedAttention(),
    )
    name, parameter = next(iter(model.named_parameters()))
    before = parameter.detach().clone()

    weights = _load_weights(path)
    weights[name].zero_()
    torch.save(weights, path / "weights.pt")
    assert torch.equal(parameter.detach(), before)


def _build_artifact_engine(path: Path, world_size: int):
    return create_engine(
        "v1.2",
        model_artifact=str(path),
        engine_config=EngineConfig(
            max_num_seqs=2,
            max_model_len=48,
            max_num_batched_tokens=4,
            max_speculative_tokens=2,
        ),
        cache_config=cache_config(),
        world_size=world_size,
        max_queue_size=4,
        max_speculative_tokens=2,
        compile_config=CompileConfig(backend="eager"),
        attention_config=PagedAttentionConfig(backend="vectorized"),
        fairness_bypass_limit=2,
        multiprocess_config=MultiprocessConfig(start_method=FAST_START_METHOD),
    )


def test_v12_world_size_one_and_two_match_v11_exported_reference(tmp_path: Path) -> None:
    async def scenario():
        path = artifact_dir(tmp_path)
        for world_size in (1, 2):
            reference = create_engine(
                "v1.1",
                engine_config=EngineConfig(
                    max_num_seqs=2,
                    max_model_len=48,
                    max_num_batched_tokens=4,
                    max_speculative_tokens=2,
                ),
                model_config=model_config(),
                cache_config=cache_config(),
                world_size=world_size,
                max_queue_size=4,
                max_speculative_tokens=2,
                compile_config=CompileConfig(backend="eager"),
                attention_config=PagedAttentionConfig(backend="vectorized"),
                fairness_bypass_limit=2,
                multiprocess_config=MultiprocessConfig(start_method=FAST_START_METHOD),
            )
            try:
                expected = await collect(reference)
            finally:
                await reference.aclose()

            engine = _build_artifact_engine(path, world_size)
            try:
                actual = await collect(engine)
                snapshot = engine.snapshot()
                assert actual == expected
                assert snapshot.core.model_artifact is not None
                assert snapshot.core.model_artifact.rank_identity_agreement
                identity = snapshot.core.model_artifact.identity
                assert all(
                    rank.artifact_identity == identity
                    for rank in snapshot.core.execution.ranks
                )
            finally:
                await engine.aclose()

    asyncio.run(scenario())


def test_text_facade_keeps_core_token_native(tmp_path: Path) -> None:
    async def scenario():
        path = artifact_dir(tmp_path)
        engine = _build_artifact_engine(path, 1)
        try:
            assert isinstance(engine, ArtifactAsyncEngine)
            assert engine.encode("tok1 tok2 tok3") == [1, 2, 3]
            handle = engine.submit_text(
                "text",
                "tok1 tok2 tok3",
                max_new_tokens=2,
            )
            await handle.wait_terminal()
            await engine.wait_idle()
            request = engine.core.get_request("text")
            assert request.prompt_token_ids == [1, 2, 3]
            assert engine.decode([1, 2, 3]) == "tok1 tok2 tok3"
        finally:
            await engine.aclose()

    asyncio.run(scenario())


def test_rank_identity_disagreement_is_rejected_before_startup(tmp_path: Path) -> None:
    first_path = tmp_path / "a"
    second_path = tmp_path / "b"
    first = export_tiny_model_artifact(first_path, model_config())
    other_config = TinyModelConfig(
        vocab_size=24,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        n_layers=1,
        d_ff=32,
        max_position_embeddings=64,
        seed=99,
    )
    second = export_tiny_model_artifact(second_path, other_config)
    specs = (
        WorkerSpec(0, 2, first.model_config, cache_config(), CompileConfig(), PagedAttentionConfig(), first.spec),
        WorkerSpec(1, 2, second.model_config, cache_config(), CompileConfig(), PagedAttentionConfig(), second.spec),
    )
    with pytest.raises(ValueError, match="artifact identity"):
        ArtifactMultiprocessExecutor(specs, start_method=FAST_START_METHOD)


def test_child_load_failure_surfaces_during_startup_and_reaps(tmp_path: Path) -> None:
    path = artifact_dir(tmp_path)
    artifact = load_model_artifact(path)
    weights = _load_weights(path)
    name = next(iter(weights))
    weights[name].zero_()
    torch.save(weights, path / "weights.pt")

    before = {process.pid for process in mp.active_children()}
    spec = WorkerSpec(
        rank=0,
        world_size=1,
        model_config=artifact.model_config,
        cache_config=cache_config(),
        compile_config=CompileConfig(),
        attention_config=PagedAttentionConfig(),
        artifact_spec=artifact.spec,
    )
    with pytest.raises(WorkerRemoteError, match="artifact identity mismatch"):
        ArtifactMultiprocessExecutor((spec,), start_method=FAST_START_METHOD)
    after = {process.pid for process in mp.active_children()}
    assert after <= before


def test_default_v12_builder_crosses_artifact_boundary_and_cleans_up() -> None:
    engine = create_engine(
        "v1.2",
        world_size=1,
        multiprocess_config=MultiprocessConfig(start_method=FAST_START_METHOD),
    )
    assert isinstance(engine, ArtifactAsyncEngine)
    assert engine.version == "v1.2"
    artifact_root = Path(engine.core.artifact.spec.root)
    assert artifact_root.exists()
    snapshot = engine.snapshot()
    assert snapshot.core.model_artifact is not None
    assert snapshot.core.model_artifact.rank_identity_agreement
    engine.close()
    assert not artifact_root.exists()

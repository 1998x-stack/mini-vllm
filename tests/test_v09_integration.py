import pytest
import torch

from mini_vllm import CacheConfig, EngineConfig, Request, TinyModelConfig, create_engine
from mini_vllm.distributed import DistributedExecutor
from mini_vllm.model.runner import ModelRunner


def configs():
    return (
        EngineConfig(
            max_num_seqs=2,
            max_model_len=48,
            max_num_batched_tokens=8,
            max_speculative_tokens=2,
        ),
        TinyModelConfig(
            vocab_size=25,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            n_layers=1,
            d_ff=32,
            max_position_embeddings=64,
            seed=11,
        ),
        CacheConfig(block_size=2, num_blocks=48),
    )


def test_v09_composition_uses_distributed_executor_and_two_rank_group() -> None:
    e, m, c = configs()
    engine = create_engine(
        "v0.9",
        engine_config=e,
        model_config=m,
        cache_config=c,
        world_size=2,
    )
    assert isinstance(engine.executor, DistributedExecutor)
    group = engine.executor.worker_group
    assert group.world_size == 2
    assert [entry.rank_config.rank for entry in group.ranked_workers] == [0, 1]
    assert not hasattr(group, "scheduler")
    assert not hasattr(group, "waiting")
    assert not hasattr(group, "running")
    ranges = [entry.worker.model_runner.vocab_range for entry in group.ranked_workers]
    assert ranges == [(0, 13), (13, 25)]


def test_v08_and_v09_have_identical_deterministic_token_history() -> None:
    e, m, c = configs()
    reference = create_engine("v0.8", engine_config=e, model_config=m, cache_config=c)
    distributed = create_engine(
        "v0.9",
        engine_config=e,
        model_config=m,
        cache_config=c,
        world_size=2,
    )
    ref_request = Request("same", [1, 2, 3, 4], max_new_tokens=7)
    dist_request = Request("same", [1, 2, 3, 4], max_new_tokens=7)
    reference.add_request(ref_request)
    distributed.add_request(dist_request)
    reference.run()
    distributed.run()
    assert dist_request.output_token_ids == ref_request.output_token_ids
    assert dist_request.finish_reason == ref_request.finish_reason


def test_v09_normal_and_speculative_paths_cross_worker_group() -> None:
    e, m, c = configs()
    engine = create_engine(
        "v0.9",
        engine_config=e,
        model_config=m,
        cache_config=c,
        world_size=2,
    )
    request = Request("A", [1, 2], max_new_tokens=5)
    engine.add_request(request)
    engine.run()
    group = engine.executor.worker_group
    assert group.execute_calls > 0
    assert group.speculative_execute_calls > 0


def test_world_size_one_is_semantically_equal_to_v08() -> None:
    e, m, c = configs()
    reference = create_engine("v0.8", engine_config=e, model_config=m, cache_config=c)
    distributed = create_engine(
        "v0.9",
        engine_config=e,
        model_config=m,
        cache_config=c,
        world_size=1,
    )
    ref_request = Request("one", [3, 5, 7], max_new_tokens=4)
    dist_request = Request("one", [3, 5, 7], max_new_tokens=4)
    reference.add_request(ref_request)
    distributed.add_request(dist_request)
    reference.run()
    distributed.run()
    assert dist_request.output_token_ids == ref_request.output_token_ids


class FailingRunner(ModelRunner):
    def execute(self, work_items, cache):
        raise RuntimeError("rank execution failed")

    def execute_speculative(self, **kwargs):
        raise RuntimeError("rank speculative execution failed")


def test_rank_failure_rolls_back_replicated_kv_and_prevents_logical_commit() -> None:
    e, m, c = configs()
    engine = create_engine(
        "v0.9",
        engine_config=e,
        model_config=m,
        cache_config=c,
        world_size=2,
    )
    request = Request("boom", [1, 2, 3], max_new_tokens=2)
    engine.add_request(request)
    group = engine.executor.worker_group
    group.ranked_workers[1].worker.model_runner = FailingRunner()

    with pytest.raises(RuntimeError, match="rank execution failed"):
        engine.step()

    assert request.num_computed_tokens == 0
    assert request.output_token_ids == []
    for entry in group.ranked_workers:
        checkpoint = entry.worker.cache.checkpoint("boom", 0)
        assert checkpoint.num_tokens == 0


def test_rank_local_logits_reconstruct_same_full_logits_as_single_worker() -> None:
    e, m, c = configs()
    reference = create_engine("v0.8", engine_config=e, model_config=m, cache_config=c)
    distributed = create_engine(
        "v0.9",
        engine_config=e,
        model_config=m,
        cache_config=c,
        world_size=2,
    )
    ref_request = Request("logits", [4, 6], max_new_tokens=1)
    dist_request = Request("logits", [4, 6], max_new_tokens=1)
    reference.add_request(ref_request)
    distributed.add_request(dist_request)

    ref_step = reference.step()
    dist_step = distributed.step()
    assert ref_step.sampled_token_ids == dist_step.sampled_token_ids
    assert ref_request.output_token_ids == dist_request.output_token_ids

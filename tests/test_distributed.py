import pytest
import torch
import torch.nn.functional as F

from mini_vllm.distributed import (
    RankConfig,
    ReferenceCollective,
    ReferenceTensorParallel,
    VocabParallelLMHead,
)


def test_rank_config_validates_world_membership() -> None:
    assert RankConfig(0, 2).rank == 0
    assert RankConfig(1, 2).world_size == 2
    with pytest.raises(ValueError):
        RankConfig(-1, 2)
    with pytest.raises(ValueError):
        RankConfig(2, 2)
    with pytest.raises(ValueError):
        RankConfig(0, 0)


def test_reference_collectives_have_explicit_tensor_semantics() -> None:
    collective = ReferenceCollective()
    left = torch.tensor([1.0, 2.0])
    right = torch.tensor([3.0, 4.0, 5.0])
    gathered = collective.all_gather((left, right), dim=0)
    assert torch.equal(gathered, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))

    reduced = collective.all_reduce_sum(
        (torch.tensor([1.0, 2.0]), torch.tensor([10.0, 20.0]))
    )
    assert len(reduced) == 2
    assert torch.equal(reduced[0], torch.tensor([11.0, 22.0]))
    assert torch.equal(reduced[1], reduced[0])
    assert reduced[0].data_ptr() != reduced[1].data_ptr()


def test_column_parallel_lm_head_reconstructs_reference_with_uneven_shards() -> None:
    torch.manual_seed(4)
    hidden = torch.randn(3, 5)
    weight = torch.randn(11, 5)
    collective = ReferenceCollective()
    local_logits = tuple(
        VocabParallelLMHead(weight, RankConfig(rank, 3)).local_logits(hidden)
        for rank in range(3)
    )
    reconstructed = collective.all_gather(local_logits, dim=-1)
    reference = F.linear(hidden, weight)
    assert torch.allclose(reconstructed, reference, atol=1e-6, rtol=1e-6)
    assert [tensor.shape[-1] for tensor in local_logits] == [4, 4, 3]


def test_reference_row_parallel_linear_matches_unsharded_linear() -> None:
    torch.manual_seed(5)
    x = torch.randn(4, 7)
    weight = torch.randn(6, 7)
    tp = ReferenceTensorParallel()
    reconstructed = tp.row_parallel_linear(x, weight, world_size=3)
    reference = F.linear(x, weight)
    assert torch.allclose(reconstructed, reference, atol=1e-6, rtol=1e-6)

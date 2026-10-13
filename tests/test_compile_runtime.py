import torch
import torch.nn.functional as F
import pytest

from mini_vllm.compile_runtime import (
    CompileConfig,
    CompiledKernelCache,
    ExecutionShapeSignature,
    PureTensorProjectionKernel,
)


def test_pure_projection_kernel_matches_linear():
    torch.manual_seed(1)
    hidden = torch.randn(3, 4)
    weight = torch.randn(7, 4)
    actual = PureTensorProjectionKernel()(hidden, weight)
    assert torch.allclose(actual, F.linear(hidden, weight), atol=1e-6)


def test_explicit_signature_cache_miss_hit_and_recapture():
    torch.manual_seed(2)
    cache = CompiledKernelCache(CompileConfig(backend="eager"))
    weight = torch.randn(6, 4)
    hidden1 = torch.randn(1, 4)
    hidden2 = torch.randn(2, 4)

    out1 = cache.run(hidden1, weight)
    out1_again = cache.run(hidden1, weight)
    out2 = cache.run(hidden2, weight)

    assert torch.allclose(out1, F.linear(hidden1, weight), atol=1e-6)
    assert torch.allclose(out1_again, out1, atol=1e-6)
    assert torch.allclose(out2, F.linear(hidden2, weight), atol=1e-6)
    assert cache.stats.compile_attempts == 2
    assert cache.stats.cache_misses == 2
    assert cache.stats.cache_hits == 1
    assert cache.stats.compiled_variants == 2
    assert cache.stats.recompiles == 1
    assert [event.mode for event in cache.events] == ["compiled", "cache-hit", "compiled"]


def test_signature_is_deterministic_and_shape_sensitive():
    weight = torch.randn(5, 4)
    a = ExecutionShapeSignature.from_tensors(torch.randn(1, 4), weight)
    b = ExecutionShapeSignature.from_tensors(torch.randn(1, 4), weight)
    c = ExecutionShapeSignature.from_tensors(torch.randn(3, 4), weight)
    assert a == b
    assert a != c
    assert a.token_count == 1 and c.token_count == 3


def test_compile_failure_falls_back_to_eager():
    def failing_compiler(*args, **kwargs):
        raise RuntimeError("compile unavailable")

    cache = CompiledKernelCache(
        CompileConfig(fallback_on_error=True),
        compiler=failing_compiler,
    )
    hidden = torch.randn(2, 4)
    weight = torch.randn(5, 4)
    actual = cache.run(hidden, weight)
    assert torch.allclose(actual, F.linear(hidden, weight), atol=1e-6)
    assert cache.stats.fallbacks == 1
    assert cache.stats.eager_calls == 1
    assert cache.last_event is not None and cache.last_event.mode == "fallback-eager"


def test_compile_failure_can_be_strict():
    def failing_compiler(*args, **kwargs):
        raise RuntimeError("compile unavailable")

    cache = CompiledKernelCache(
        CompileConfig(fallback_on_error=False),
        compiler=failing_compiler,
    )
    with pytest.raises(RuntimeError, match="compile unavailable"):
        cache.run(torch.randn(1, 4), torch.randn(5, 4))

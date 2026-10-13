from __future__ import annotations

from collections.abc import Callable

import torch

from mini_vllm.compile_runtime.kernel import PureTensorProjectionKernel
from mini_vllm.compile_runtime.types import (
    CompileConfig,
    CompileEvent,
    CompileStats,
    ExecutionShapeSignature,
)


class CompiledKernelCache:
    """Explicit educational cache around torch.compile.

    PyTorch may perform additional internal specialization. This registry is the
    serving-runtime-visible layer: one callable per explicit tensor signature.
    """

    def __init__(
        self,
        config: CompileConfig | None = None,
        *,
        compiler: Callable[..., Callable] | None = None,
    ) -> None:
        self.config = config or CompileConfig()
        self.kernel = PureTensorProjectionKernel()
        self._compiler = compiler or torch.compile
        self._cache: dict[ExecutionShapeSignature, Callable] = {}
        self.stats = CompileStats()
        self.events: list[CompileEvent] = []

    @property
    def signatures(self) -> tuple[ExecutionShapeSignature, ...]:
        return tuple(self._cache)

    @property
    def last_event(self) -> CompileEvent | None:
        return self.events[-1] if self.events else None

    def _eager(self, hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        self.stats.eager_calls += 1
        return self.kernel(hidden, weight)

    def run(self, hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        signature = ExecutionShapeSignature.from_tensors(hidden, weight)
        if not self.config.enabled:
            self.events.append(CompileEvent(signature, "eager-disabled"))
            return self._eager(hidden, weight)

        compiled = self._cache.get(signature)
        if compiled is not None:
            self.stats.cache_hits += 1
            self.events.append(CompileEvent(signature, "cache-hit"))
            return compiled(hidden, weight)

        self.stats.cache_misses += 1
        self.stats.compile_attempts += 1
        if self._cache:
            self.stats.recompiles += 1
        try:
            compiled = self._compiler(
                self.kernel,
                backend=self.config.backend,
                fullgraph=self.config.fullgraph,
                dynamic=self.config.dynamic,
            )
            output = compiled(hidden, weight)
            self._cache[signature] = compiled
            self.stats.compiled_variants = len(self._cache)
            self.events.append(CompileEvent(signature, "compiled"))
            return output
        except Exception:
            self.stats.fallbacks += 1
            self.events.append(CompileEvent(signature, "fallback-eager"))
            if not self.config.fallback_on_error:
                raise
            return self._eager(hidden, weight)

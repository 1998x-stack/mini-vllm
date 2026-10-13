from __future__ import annotations

import torch

from mini_vllm.cache.base import KVCacheBackend
from mini_vllm.compile_runtime import CompileConfig, CompiledKernelCache
from mini_vllm.distributed.rank import RankConfig
from mini_vllm.distributed.tensor_parallel import VocabParallelLMHead
from mini_vllm.model.runner import ModelRunner
from mini_vllm.model.tiny_transformer import TinyDecoderLM


class CompileAwareVocabParallelTinyRunner(ModelRunner):
    """Eager TinyDecoder/KV path plus compile-aware rank-local projection."""

    def __init__(
        self,
        model: TinyDecoderLM,
        rank_config: RankConfig,
        *,
        compile_config: CompileConfig | None = None,
        compile_cache: CompiledKernelCache | None = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.rank_config = rank_config
        self.lm_head = VocabParallelLMHead(model.lm_head.weight, rank_config)
        self.compile_cache = compile_cache or CompiledKernelCache(compile_config)

    @property
    def vocab_range(self) -> tuple[int, int]:
        return self.lm_head.shard.start, self.lm_head.shard.end

    def _project_trace(self, hidden_trace: tuple[torch.Tensor, ...]) -> torch.Tensor:
        hidden_batch = torch.stack(hidden_trace, dim=0)
        return self.compile_cache.run(hidden_batch, self.lm_head.local_weight)

    @torch.no_grad()
    def execute(self, work_items, cache: KVCacheBackend) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for item in work_items:
            hidden_trace = self.model.forward_chunk_hidden_trace(
                request_id=item.request_id,
                token_ids=item.token_ids,
                start_pos=item.start_pos,
                cache=cache,
            )
            outputs[item.request_id] = self._project_trace(hidden_trace)[-1]
        return outputs

    @torch.no_grad()
    def execute_speculative(
        self,
        *,
        request_id: str,
        token_ids: tuple[int, ...],
        start_pos: int,
        cache: KVCacheBackend,
    ) -> tuple[torch.Tensor, ...]:
        hidden_trace = self.model.forward_chunk_hidden_trace(
            request_id=request_id,
            token_ids=token_ids,
            start_pos=start_pos,
            cache=cache,
        )
        logits = self._project_trace(hidden_trace)
        return tuple(logits[index] for index in range(logits.shape[0]))

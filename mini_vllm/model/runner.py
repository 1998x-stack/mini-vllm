from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from mini_vllm.cache.base import KVCacheBackend
from mini_vllm.core.work import WorkItem
from mini_vllm.model.tiny_transformer import TinyDecoderLM


class ModelRunner(ABC):
    @abstractmethod
    def execute(
        self,
        work_items: list[WorkItem],
        cache: KVCacheBackend,
    ) -> dict[str, torch.Tensor]:
        ...

    def execute_speculative(
        self,
        *,
        request_id: str,
        token_ids: tuple[int, ...],
        start_pos: int,
        cache: KVCacheBackend,
    ) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError(f"{type(self).__name__} does not support speculative traces")


class TinyTransformerRunner(ModelRunner):
    def __init__(self, model: TinyDecoderLM) -> None:
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def execute(
        self,
        work_items: list[WorkItem],
        cache: KVCacheBackend,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for item in work_items:
            outputs[item.request_id] = self.model.forward_chunk(
                request_id=item.request_id,
                token_ids=item.token_ids,
                start_pos=item.start_pos,
                cache=cache,
            )
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
        return self.model.forward_chunk_logits(
            request_id=request_id,
            token_ids=token_ids,
            start_pos=start_pos,
            cache=cache,
        )


class ScriptedModelRunner(ModelRunner):
    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size

    def execute(
        self,
        work_items: list[WorkItem],
        cache: KVCacheBackend,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for item in work_items:
            next_token = (item.token_ids[-1] + 1) % self.vocab_size
            logits = torch.full((self.vocab_size,), -10.0)
            logits[next_token] = 10.0
            outputs[item.request_id] = logits
        return outputs

    def execute_speculative(
        self,
        *,
        request_id: str,
        token_ids: tuple[int, ...],
        start_pos: int,
        cache: KVCacheBackend,
    ) -> tuple[torch.Tensor, ...]:
        outputs: list[torch.Tensor] = []
        for token_id in token_ids:
            next_token = (token_id + 1) % self.vocab_size
            logits = torch.full((self.vocab_size,), -10.0)
            logits[next_token] = 10.0
            outputs.append(logits)
        return tuple(outputs)

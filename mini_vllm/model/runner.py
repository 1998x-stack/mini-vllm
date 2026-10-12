from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from mini_vllm.cache.base import KVCacheBackend
from mini_vllm.core.work import WorkItem
from mini_vllm.model.tiny_transformer import TinyDecoderLM


class ModelRunner(ABC):
    @abstractmethod
    def execute(self, work_items: list[WorkItem], cache: KVCacheBackend) -> dict[str, torch.Tensor]:
        ...


class TinyTransformerRunner(ModelRunner):
    """Readable reference runner: loop over requests in Python."""

    def __init__(self, model: TinyDecoderLM) -> None:
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def execute(self, work_items: list[WorkItem], cache: KVCacheBackend) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for item in work_items:
            outputs[item.request_id] = self.model.forward_chunk(
                request_id=item.request_id,
                token_ids=item.token_ids,
                start_pos=item.start_pos,
                cache=cache,
            )
        return outputs


class ScriptedModelRunner(ModelRunner):
    """Deterministic runner for scheduler tests and timeline demos."""

    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size

    def execute(self, work_items: list[WorkItem], cache: KVCacheBackend) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for item in work_items:
            next_token = (item.token_ids[-1] + 1) % self.vocab_size
            logits = torch.full((self.vocab_size,), -10.0)
            logits[next_token] = 10.0
            outputs[item.request_id] = logits
        return outputs

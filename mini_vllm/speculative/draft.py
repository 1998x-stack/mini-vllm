from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterable

from mini_vllm.core.request import Request
from mini_vllm.speculative.types import DraftBatch


class DraftProvider(ABC):
    @abstractmethod
    def propose(self, request: Request, *, max_tokens: int) -> DraftBatch:
        ...


class ScriptedDraftProvider(DraftProvider):
    def __init__(self, scripts: dict[str, Iterable[Iterable[int]]] | None = None) -> None:
        self._scripts: dict[str, deque[tuple[int, ...]]] = defaultdict(deque)
        for request_id, batches in (scripts or {}).items():
            for batch in batches:
                self._scripts[request_id].append(tuple(int(token) for token in batch))

    def propose(self, request: Request, *, max_tokens: int) -> DraftBatch:
        if max_tokens <= 0:
            return DraftBatch(())
        queue = self._scripts.get(request.request_id)
        if not queue:
            return DraftBatch(())
        return DraftBatch(queue.popleft()[:max_tokens])


class IncrementDraftProvider(DraftProvider):
    def __init__(self, vocab_size: int) -> None:
        if vocab_size <= 1:
            raise ValueError("vocab_size must be > 1")
        self.vocab_size = vocab_size

    def propose(self, request: Request, *, max_tokens: int) -> DraftBatch:
        if max_tokens <= 0:
            return DraftBatch(())
        token = request.all_token_ids[-1]
        result: list[int] = []
        for _ in range(max_tokens):
            token = (token + 1) % self.vocab_size
            result.append(token)
        return DraftBatch(tuple(result))

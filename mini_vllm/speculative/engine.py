from __future__ import annotations

import torch

from mini_vllm.core.engine import Engine
from mini_vllm.core.request import Request
from mini_vllm.speculative.draft import DraftProvider, ScriptedDraftProvider
from mini_vllm.speculative.types import VerificationResult


class SpeculativeEngine(Engine):
    def __init__(self, *, draft_provider: DraftProvider | None = None, max_speculative_tokens: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        if max_speculative_tokens <= 0:
            raise ValueError("max_speculative_tokens must be > 0")
        self.draft_provider = draft_provider or ScriptedDraftProvider()
        self.max_speculative_tokens = max_speculative_tokens

    def _target_token(self, logits: torch.Tensor, request: Request) -> int:
        return self.sampler.sample(logits, request.sampling_params)

    def _fallback_normal(self, request: Request, logits: torch.Tensor) -> tuple[int, ...]:
        token_id = self._target_token(logits, request)
        request.append_output(token_id)
        return (token_id,)

    def _produce_output_tokens(self, request: Request, logits: torch.Tensor) -> tuple[int, ...]:
        remaining = request.max_new_tokens - len(request.output_token_ids)
        if remaining <= 0:
            return ()
        if request.sampling_params.temperature != 0 or self.config.eos_token_id is not None:
            return super()._produce_output_tokens(request, logits)

        draft = self.draft_provider.propose(request, max_tokens=min(self.max_speculative_tokens, remaining))
        if not draft.token_ids:
            return self._fallback_normal(request, logits)
        if any(token_id >= logits.numel() for token_id in draft.token_ids):
            raise ValueError("draft token id exceeds target vocabulary")

        stable_pos = request.num_computed_tokens
        speculative_end = stable_pos + len(draft.token_ids)
        if not self.cache.ensure_capacity(request.request_id, speculative_end):
            return self._fallback_normal(request, logits)

        checkpoint = self.cache.checkpoint(request.request_id, stable_pos)
        try:
            trace_logits = self.model_runner.execute_speculative(request_id=request.request_id, token_ids=draft.token_ids, start_pos=stable_pos, cache=self.cache)
            if len(trace_logits) != len(draft.token_ids):
                raise RuntimeError("speculative runner returned wrong logits trace length")
            accepted=[]; rejected=(); fallback_token=None; bonus_token=None; expected_logits=logits
            for index,candidate in enumerate(draft.token_ids):
                target=self._target_token(expected_logits,request)
                if candidate!=target:
                    rejected=draft.token_ids[index:]; fallback_token=target; break
                accepted.append(candidate); expected_logits=trace_logits[index]
            else:
                if len(accepted)<remaining:
                    bonus_token=self._target_token(trace_logits[-1],request)

            retain_tokens=stable_pos+len(accepted)
            self.cache.rollback(checkpoint,retain_tokens=retain_tokens)
            request.num_computed_tokens=retain_tokens
            committed=[]
            for token_id in accepted:
                request.append_output(token_id); committed.append(token_id)
            if fallback_token is not None and len(committed)<remaining:
                request.append_output(fallback_token); committed.append(fallback_token)
            elif bonus_token is not None and len(committed)<remaining:
                request.append_output(bonus_token); committed.append(bonus_token)
            self._step_speculation.append(VerificationResult(request_id=request.request_id,checkpoint_pos=stable_pos,draft_token_ids=draft.token_ids,accepted_token_ids=tuple(accepted),rejected_token_ids=rejected,fallback_token_id=fallback_token,bonus_token_id=bonus_token,rollback_to_pos=retain_tokens))
            return tuple(committed)
        except Exception:
            self.cache.rollback(checkpoint,retain_tokens=checkpoint.num_tokens)
            request.num_computed_tokens=checkpoint.num_tokens
            raise

from __future__ import annotations

import torch

from mini_vllm.core.request import SamplingParams


class Sampler:
    """Reference sampler with greedy, temperature, top-k and top-p paths."""

    @torch.no_grad()
    def sample(self, logits: torch.Tensor, params: SamplingParams) -> int:
        if logits.ndim != 1:
            raise ValueError("Sampler expects a 1-D logits tensor")

        if params.temperature == 0:
            return int(torch.argmax(logits).item())

        scores = logits / params.temperature

        if params.top_k is not None:
            k = min(params.top_k, scores.numel())
            threshold = torch.topk(scores, k).values[-1]
            scores = torch.where(scores < threshold, torch.tensor(-torch.inf, device=scores.device), scores)

        if params.top_p is not None and params.top_p < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            sorted_probs = torch.softmax(sorted_scores, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > params.top_p
            if remove.numel() > 1:
                remove[1:] = remove[:-1].clone()
            remove[0] = False
            sorted_scores[remove] = -torch.inf
            filtered = torch.full_like(scores, -torch.inf)
            filtered.scatter_(0, sorted_indices, sorted_scores)
            scores = filtered

        probs = torch.softmax(scores, dim=-1)
        return int(torch.multinomial(probs, 1).item())

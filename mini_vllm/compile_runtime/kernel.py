from __future__ import annotations

import torch
import torch.nn.functional as F


class PureTensorProjectionKernel:
    """Narrow v0.11 compile boundary: tensors in, tensor out, no serving state."""

    def __call__(self, hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, weight)

from __future__ import annotations

import torch


class ReferenceCollective:
    """Single-process mathematical oracle for collective operations.

    Callers provide all rank-local tensors at once. This intentionally avoids
    transport/process concerns so the tensor semantics remain inspectable.
    """

    @staticmethod
    def _require_nonempty(tensors: tuple[torch.Tensor, ...]) -> None:
        if not tensors:
            raise ValueError("collective requires at least one rank tensor")

    def all_gather(
        self,
        tensors: tuple[torch.Tensor, ...],
        *,
        dim: int = -1,
    ) -> torch.Tensor:
        self._require_nonempty(tensors)
        ndim = tensors[0].ndim
        dim = dim % ndim
        reference = tensors[0]
        for tensor in tensors[1:]:
            if tensor.ndim != ndim:
                raise ValueError("all_gather tensors must have equal rank")
            if tensor.dtype != reference.dtype or tensor.device != reference.device:
                raise ValueError("all_gather tensors must share dtype/device")
            for axis, (left, right) in enumerate(zip(reference.shape, tensor.shape)):
                if axis != dim and left != right:
                    raise ValueError("all_gather non-gather dimensions must match")
        return torch.cat(tensors, dim=dim)

    def all_reduce_sum(
        self,
        tensors: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        self._require_nonempty(tensors)
        reference = tensors[0]
        for tensor in tensors[1:]:
            if tensor.shape != reference.shape:
                raise ValueError("all_reduce tensors must have identical shapes")
            if tensor.dtype != reference.dtype or tensor.device != reference.device:
                raise ValueError("all_reduce tensors must share dtype/device")
        total = torch.stack(tensors, dim=0).sum(dim=0)
        return tuple(total.clone() for _ in tensors)

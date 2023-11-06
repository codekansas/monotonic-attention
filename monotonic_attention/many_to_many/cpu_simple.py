# mypy: disable-error-code="override"
"""Defines the monotonic attention forward and backward passes in vanilla PyTorch."""

import torch
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx, once_differentiable


def forward_pass_(probs: Tensor) -> Tensor:
    phis = torch.empty_like(probs)
    t_i, t_j = probs.size(-2), probs.size(-1)
    for i in range(t_i):
        for j in range(t_j):
            if i == 0 and j == 0:
                phis[..., i, j] = 1.0
            elif i == 0:
                phis[..., i, j] = phis[..., i, j - 1] * probs[..., i, j - 1]
            elif j == 0:
                phis[..., i, j] = phis[..., i - 1, j] * (1 - probs[..., i - 1, j])
            else:
                a, b = phis[..., i, j - 1] * probs[..., i, j - 1], phis[..., i - 1, j] * (1 - probs[..., i - 1, j])
                phis[..., i, j] = a + b
    return phis


def backward_pass_(probs: Tensor, phis: Tensor, grad_phis: Tensor) -> Tensor:
    grad_probs = torch.empty_like(grad_phis)
    t_i, t_j = probs.size(-2), probs.size(-1)
    for i in range(t_i - 1, -1, -1):
        for j in range(t_j - 1, -1, -1):
            if i == t_i - 1 and j == t_j - 1:
                grad_probs[..., i, j] = 0.0
            elif i == t_i - 1:
                grad_probs[..., i, j] = grad_phis[..., i, j + 1] * phis[..., i, j]
                grad_phis[..., i, j] += grad_phis[..., i, j + 1] * probs[..., i, j]
            elif j == t_j - 1:
                grad_probs[..., i, j] = grad_phis[..., i + 1, j] * -phis[..., i, j]
                grad_phis[..., i, j] += grad_phis[..., i + 1, j] * (1 - probs[..., i, j])
            else:
                grad_probs[..., i, j] = (
                    grad_phis[..., i, j + 1] * phis[..., i, j] + grad_phis[..., i + 1, j] * -phis[..., i, j]
                )
                grad_phis[..., i, j] += grad_phis[..., i, j + 1] * probs[..., i, j] + grad_phis[..., i + 1, j] * (
                    1 - probs[..., i, j]
                )
    return grad_probs


class MonotonicAttention(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, probs: Tensor) -> Tensor:
        phis = forward_pass_(probs)
        ctx.save_for_backward(probs, phis)
        return phis

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, grad_phis: Tensor) -> Tensor:
        probs, phis = ctx.saved_tensors
        grad_probs = backward_pass_(probs, phis, grad_phis)
        return grad_probs


def monotonic_attention(probs: Tensor) -> Tensor:
    """Computes the monotonic attention normalization on the transition probabilities.

    Args:
        probs: The transition probabilities, with shape
            ``(bsz, tsz_src, tsz_tgt)`` and values in ``[0, 1]``.

    Returns:
        The marginalized probabilities for each cell being part of a monotonic
        alignment path, with shape ``(bsz, tsz_src, tsz_tgt)`` and values in
        ``[0, 1]``.
    """
    return MonotonicAttention.apply(probs)

# mypy: disable-error-code="override"
"""Defines the faster monotonic attention forward and backward passes in vanilla PyTorch."""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx, once_differentiable


def forward_pass_(probs: Tensor) -> Tensor:
    phis = torch.empty_like(probs)
    t_i, t_j = probs.size(-2), probs.size(-1)
    probs = F.pad(probs, (1, 0, 1, 0), value=0.0)
    phis[..., 0, 0] = 1.0
    phis = F.pad(phis, (1, 0, 1, 0), value=0.0)
    for t in range(1, t_i + t_j - 1):
        i = torch.arange(max(0, t - t_j + 1), min(t + 1, t_i))
        j = torch.arange(min(t, t_j - 1), max(-1, t - t_i), -1)
        a = phis[..., i + 1, j] * probs[..., i + 1, j]
        b = phis[..., i, j + 1] * (1 - probs[..., i, j + 1])
        phis[..., i + 1, j + 1] = a + b
    return phis[..., 1:, 1:]


def backward_pass_(probs: Tensor, phis: Tensor, grad_phis: Tensor) -> Tensor:
    grad_probs = torch.empty_like(grad_phis)
    t_i, t_j = probs.size(-2), probs.size(-1)
    grad_phis = F.pad(grad_phis, (0, 1, 0, 1), value=0.0)
    phis = F.pad(phis, (0, 1, 0, 1), value=0.0)
    grad_probs[..., t_i - 1, t_j - 1] = 0.0
    for t in range(t_i + t_j - 2, -1, -1):
        i = torch.arange(max(0, t - t_j + 1), min(t + 1, t_i))
        j = torch.arange(min(t, t_j - 1), max(-1, t - t_i), -1)

        a, b = grad_phis[..., i, j + 1] * phis[..., i, j], grad_phis[..., i + 1, j] * -phis[..., i, j]
        grad_probs[..., i, j] = a + b

        a, b = grad_phis[..., i, j + 1] * probs[..., i, j], grad_phis[..., i + 1, j] * (1 - probs[..., i, j])
        grad_phis[..., i, j] += a + b

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

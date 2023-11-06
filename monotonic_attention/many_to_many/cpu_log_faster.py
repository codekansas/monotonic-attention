# mypy: disable-error-code="override"
"""Defines the faster monotonic attention forward and backward passes in log-space PyTorch."""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx, once_differentiable


def _logaddexp(*ts: Tensor) -> Tensor:
    return torch.logsumexp(torch.stack(ts, dim=-1), dim=-1)


def _log_1mexp(x: Tensor) -> Tensor:
    return torch.log(-torch.expm1(x))  # Numerically-stable `log(1 - exp(x))`


def forward_pass_(log_probs: Tensor) -> Tensor:
    log_phis = torch.empty_like(log_probs)
    t_i, t_j = log_probs.size(-2), log_probs.size(-1)
    log_probs_padded = F.pad(log_probs, (1, 0, 1, 0), value=float("-inf"))
    log_probs_padded_1mexp = F.pad(_log_1mexp(log_probs), (1, 0, 1, 0), value=float("-inf"))
    log_phis[..., 0, 0] = 0.0
    log_phis = F.pad(log_phis, (1, 0, 1, 0), value=float("-inf"))
    for t in range(1, t_i + t_j - 1):
        i = torch.arange(max(0, t - t_j + 1), min(t + 1, t_i))
        j = torch.arange(min(t, t_j - 1), max(-1, t - t_i), -1)
        log_phis[..., i + 1, j + 1] = _logaddexp(
            log_phis[..., i + 1, j] + log_probs_padded[..., i + 1, j],
            log_phis[..., i, j + 1] + log_probs_padded_1mexp[..., i, j + 1],
        )
    return log_phis[..., 1:, 1:]


def _d_log_1emxp(x: Tensor) -> Tensor:
    return 1 + (1 / (torch.expm1(x)))  # Derivative of `log(1 - exp(x))`


def backward_pass_(log_probs: Tensor, log_phis: Tensor, grad_log_phis: Tensor) -> Tensor:
    grad_log_probs = torch.empty_like(grad_log_phis)
    t_i, t_j = log_probs.size(-2), log_probs.size(-1)
    grad_log_phis = F.pad(grad_log_phis, (0, 1, 0, 1), value=0.0)
    log_phis = F.pad(log_phis, (0, 1, 0, 1), value=0.0)
    grad_log_probs[..., t_i - 1, t_j - 1] = 0.0

    for t in range(t_i + t_j - 2, -1, -1):
        i = torch.arange(max(0, t - t_j + 1), min(t + 1, t_i))
        j = torch.arange(min(t, t_j - 1), max(-1, t - t_i), -1)
        a = (log_phis[..., i, j] + log_probs[..., i, j] - log_phis[..., i, j + 1]).exp()
        b = (log_phis[..., i, j] + _log_1mexp(log_probs[..., i, j]) - log_phis[..., i + 1, j]).exp()
        c = grad_log_phis[..., i, j + 1] * a
        d = grad_log_phis[..., i + 1, j] * b
        grad_log_probs[..., i, j] = c + d * _d_log_1emxp(log_probs[..., i, j])
        grad_log_phis[..., i, j] += c + d

    return grad_log_probs


class MonotonicAttention(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, log_probs: Tensor) -> Tensor:
        log_phis = forward_pass_(log_probs)
        ctx.save_for_backward(log_probs, log_phis)
        return log_phis

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, grad_log_phis: Tensor) -> Tensor:
        log_probs, log_phis = ctx.saved_tensors
        grad_log_probs = backward_pass_(log_probs, log_phis, grad_log_phis)
        return grad_log_probs


def monotonic_attention(probs: Tensor, epsilon: float = 1e-3) -> Tensor:
    """Computes the monotonic attention normalization on the transition probabilities.

    Args:
        probs: The transition probabilities, with shape
            ``(bsz, tsz_src, tsz_tgt)`` and values in ``[0, 1]``.
        epsilon: The epsilon value to use for the normalization.

    Returns:
        The marginalized probabilities for each cell being part of a monotonic
        alignment path, with shape ``(bsz, tsz_src, tsz_tgt)`` and values in
        ``[0, 1]``.
    """
    probs = (probs * (1 - 2 * epsilon)) + epsilon
    return MonotonicAttention.apply(probs.log()).exp()

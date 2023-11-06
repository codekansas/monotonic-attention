# mypy: disable-error-code="override"
"""Defines the monotonic attention forward and backward passes in log-space PyTorch."""

import torch
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx, once_differentiable


def _logaddexp(*ts: Tensor) -> Tensor:
    return torch.stack(ts, dim=-1).logsumexp(dim=-1)


def _log_1mexp(x: Tensor) -> Tensor:
    return torch.log(-torch.expm1(x))  # Numerically-stable `log(1 - exp(x))`


def forward_pass_(log_probs: Tensor) -> Tensor:
    log_phis = torch.empty_like(log_probs)
    t_i, t_j = log_probs.size(-2), log_probs.size(-1)
    for i in range(t_i):
        for j in range(t_j):
            if i == 0 and j == 0:
                log_phis[..., i, j] = 0.0
            elif i == 0:
                log_phis[..., i, j] = log_phis[..., i, j - 1] + log_probs[..., i, j - 1]
            elif j == 0:
                log_phis[..., i, j] = log_phis[..., i - 1, j] + _log_1mexp(log_probs[..., i - 1, j])
            else:
                log_phis[..., i, j] = _logaddexp(
                    log_phis[..., i, j - 1] + log_probs[..., i, j - 1],
                    log_phis[..., i - 1, j] + _log_1mexp(log_probs[..., i - 1, j]),
                )
    return log_phis


def _d_log_1emxp(x: Tensor) -> Tensor:
    return 1 + (1 / (torch.expm1(x)))  # Derivative of `log(1 - exp(x))`


def backward_pass_(log_probs: Tensor, log_phis: Tensor, grad_log_phis: Tensor) -> Tensor:
    grad_log_probs = torch.empty_like(grad_log_phis)
    t_i, t_j = log_probs.size(-2), log_probs.size(-1)

    for i in range(t_i - 1, -1, -1):
        for j in range(t_j - 1, -1, -1):
            if i == t_i - 1 and j == t_j - 1:
                grad_log_probs[..., i, j] = 0.0
            elif i == t_i - 1:
                grad_log_probs[..., i, j] = (
                    grad_log_phis[..., i, j + 1]
                    * (log_phis[..., i, j] + log_probs[..., i, j] - log_phis[..., i, j + 1]).exp()
                )
                grad_log_phis[..., i, j] += (
                    grad_log_phis[..., i, j + 1]
                    * (log_phis[..., i, j] + log_probs[..., i, j] - log_phis[..., i, j + 1]).exp()
                )
            elif j == t_j - 1:
                grad_log_probs[..., i, j] = (
                    grad_log_phis[..., i + 1, j]
                    * (log_phis[..., i, j] + _log_1mexp(log_probs[..., i, j]) - log_phis[..., i + 1, j]).exp()
                    * _d_log_1emxp(log_probs[..., i, j])
                )
                grad_log_phis[..., i, j] += (
                    grad_log_phis[..., i + 1, j]
                    * (log_phis[..., i, j] + _log_1mexp(log_probs[..., i, j]) - log_phis[..., i + 1, j]).exp()
                )
            else:
                grad_log_probs[..., i, j] = grad_log_phis[..., i, j + 1] * (
                    log_phis[..., i, j] + log_probs[..., i, j] - log_phis[..., i, j + 1]
                ).exp() + grad_log_phis[..., i + 1, j] * (
                    log_phis[..., i, j] + _log_1mexp(log_probs[..., i, j]) - log_phis[..., i + 1, j]
                ).exp() * _d_log_1emxp(
                    log_probs[..., i, j]
                )
                grad_log_phis[..., i, j] += (
                    grad_log_phis[..., i, j + 1]
                    * (log_phis[..., i, j] + log_probs[..., i, j] - log_phis[..., i, j + 1]).exp()
                    + grad_log_phis[..., i + 1, j]
                    * (log_phis[..., i, j] + _log_1mexp(log_probs[..., i, j]) - log_phis[..., i + 1, j]).exp()
                )

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

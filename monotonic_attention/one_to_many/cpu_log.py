# mypy: disable-error-code="override"
"""Defines the one-to-many monotonic attention forward and backward passes in log-space PyTorch."""

import warnings

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx, once_differentiable

MIN_LOG_PROB = -1e4


def _logaddexp(*ts: Tensor) -> Tensor:
    return torch.logsumexp(torch.stack(ts, dim=-1), dim=-1)


def _log_1mexp(x: Tensor) -> Tensor:
    return torch.log(-torch.expm1(x))  # Numerically-stable `log(1 - exp(x))`


def forward_pass_(log_probs: Tensor) -> Tensor:
    t_i = log_probs.size(-2)
    log_probs = F.pad(log_probs, (1, 0), value=0.0)
    log_phis = torch.empty_like(log_probs)
    log_phis[..., :, 0] = float("-inf")
    log_phis[..., 0, :] = float("-inf")
    log_phis[..., 0, 1] = 0.0
    for i in range(1, t_i):
        log_phis[..., i, 1:] = _logaddexp(
            log_phis[..., i - 1, 1:] + log_probs[..., i - 1, 1:],
            log_phis[..., i - 1, :-1] + _log_1mexp(log_probs[..., i - 1, :-1]),
        )
    return log_phis[..., 1:]


def _d_log_1emxp(x: Tensor) -> Tensor:
    return 1 + (1 / (torch.expm1(x)))  # Derivative of `log(1 - exp(x))`


def backward_pass_(log_probs: Tensor, log_phis: Tensor, grad_log_phis: Tensor) -> Tensor:
    t_i = log_probs.size(-2)
    grad_log_probs = torch.empty_like(grad_log_phis)
    grad_log_probs[..., t_i - 1, :] = 0.0
    for i in range(t_i - 2, -1, -1):
        p = log_phis[..., i + 1, :].clamp_min(MIN_LOG_PROB)
        a = (log_phis[..., i, :] + log_probs[..., i, :] - p).exp()
        b = (log_phis[..., i, :-1] + _log_1mexp(log_probs[..., i, :-1]) - p[..., 1:]).exp()
        c = grad_log_phis[..., i + 1, :] * a
        d = grad_log_phis[..., i + 1, 1:] * b
        grad_log_probs[..., i, :] = c
        grad_log_probs[..., i, :-1] += d * _d_log_1emxp(log_probs[..., i, :-1])
        grad_log_phis[..., i, :] += c
        grad_log_phis[..., i, :-1] += d
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
        grad_log_probs = backward_pass_(log_probs, log_phis, grad_log_phis.clone())
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
    tsz_src, tsz_tgt = probs.size(-2), probs.size(-1)
    if tsz_tgt > tsz_src:
        warnings.warn("One-to-many attention expects the source sequence to be longer than the target sequence!")
    probs = (probs * (1 - 2 * epsilon)) + epsilon
    return MonotonicAttention.apply(probs.log()).exp()

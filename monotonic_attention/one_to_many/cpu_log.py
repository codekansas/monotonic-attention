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


def _pos_log_prob(x: Tensor) -> Tensor:
    return -torch.log1p(torch.exp(-x))


def _neg_log_prob(x: Tensor) -> Tensor:
    return -torch.log1p(torch.exp(x))


def _d_pos_log_prob(x: Tensor) -> Tensor:
    return 1 / (1 + torch.exp(x))


def _d_neg_log_prob(x: Tensor) -> Tensor:
    return -1 / (1 + torch.exp(-x))


def forward_pass_(logits: Tensor) -> Tensor:
    t_i = logits.size(-2)
    logits = F.pad(logits, (1, 0), value=0.0)
    phis = torch.empty_like(logits)
    phis[..., :, 0] = MIN_LOG_PROB
    phis[..., 0, :] = MIN_LOG_PROB
    phis[..., 0, 1] = 0.0
    for i in range(1, t_i):
        phis[..., i, 1:] = _logaddexp(
            phis[..., i - 1, 1:] + _pos_log_prob(logits[..., i - 1, 1:]),
            phis[..., i - 1, :-1] + _neg_log_prob(logits[..., i - 1, :-1]),
        )
    phis = phis[..., 1:]
    return phis


def backward_pass_(logits: Tensor, phis: Tensor, grad_phis: Tensor) -> Tensor:
    t_i = logits.size(-2)
    grad_logits = torch.empty_like(grad_phis)
    grad_logits[..., t_i - 1, :] = 0.0
    for i in range(t_i - 2, -1, -1):
        p = phis[..., i + 1, :]
        a = (phis[..., i, :] + _pos_log_prob(logits[..., i, :]) - p).exp()
        b = (phis[..., i, :-1] + _neg_log_prob(logits[..., i, :-1]) - p[..., 1:]).exp()
        c = grad_phis[..., i + 1, :] * a
        d = grad_phis[..., i + 1, 1:] * b
        grad_logits[..., i, :] = c * _d_pos_log_prob(logits[..., i, :])
        grad_logits[..., i, :-1] += d * _d_neg_log_prob(logits[..., i, :-1])
        grad_phis[..., i, :] += c
        grad_phis[..., i, :-1] += d
    return grad_logits


class MonotonicAttention(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, logits: Tensor) -> Tensor:
        phis = forward_pass_(logits)
        ctx.save_for_backward(logits, phis)
        return phis

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, grad_phis: Tensor) -> Tensor:
        logits, phis = ctx.saved_tensors
        grad_logits = backward_pass_(logits, phis, grad_phis.clone())
        return grad_logits


def monotonic_attention(logits: Tensor) -> Tensor:
    """Computes the monotonic attention normalization on the transition probabilities.

    Args:
        logits: The transition logits, with shape ``(bsz, tsz_src, tsz_tgt)``

    Returns:
        The marginalized log probabilities for each cell being part of a
        monotonic alignment path, with shape ``(bsz, tsz_src, tsz_tgt)``.
    """
    tsz_src, tsz_tgt = logits.size(-2), logits.size(-1)
    if tsz_tgt > tsz_src:
        warnings.warn("One-to-many attention expects the source sequence to be longer than the target sequence!")
    return MonotonicAttention.apply(logits)

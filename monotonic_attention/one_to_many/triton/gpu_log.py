# mypy: disable-error-code="import, no-untyped-def, override"
# ruff: noqa: ANN001, ANN201, ANN202, N803, N806
"""Defines Triton kernels for the log-space RWKV forward and backward passes."""

import warnings
from typing import Any

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx, once_differentiable

NEG_INF = -1e3


def get_block_size_c(chans: int) -> int:
    if chans < 32:
        return 32
    if chans >= 2048:
        # As written, the kernels need to be able to pass values between
        # adjacent threads, meaning that we need to be able to keep the entire
        # sequence in shared memory. We set an upper bound of 2048, which is
        # probably longer than any real-world sequence we might care about, and
        # limit ourselves to at most one block.
        raise NotImplementedError("Triton kernels do not support more than 2048 target sequence length")
    return triton.next_power_of_2(chans)


@triton.jit
def logaddexp(a, b):
    max_ab = tl.maximum(a, b)
    return max_ab + tl.math.log(tl.math.exp(a - max_ab) + tl.math.exp(b - max_ab))


@triton.jit
def log_1mexp(x):
    return tl.log(-tl.math.expm1(x))


@triton.jit
def d_log_1mexp(x):
    return 1 + (1 / tl.math.expm1(x))


@triton.jit
def forward_pass_kernel(
    # Log probabilities tensor (input)
    log_probs_ptr,
    log_probs_s_bsz,
    log_probs_s_src,
    log_probs_s_tgt,
    # Log phis tensor (output)
    log_phis_ptr,
    log_phis_s_bsz,
    log_phis_s_src,
    log_phis_s_tgt,
    # Tensor dimensions
    t_i,
    t_j,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Parallelize over the batch dimension.
    b_idx = tl.program_id(0)
    j_idx = tl.program_id(1)

    # Pointers to the log probabilities.
    j = (j_idx * BLOCK_SIZE_C) + tl.arange(0, BLOCK_SIZE_C)
    jmask = j < t_j
    jmask_shifted = jmask & (j > 0)

    # Gets pointers offset for the current batch.
    log_probs_ptr = log_probs_ptr + b_idx * log_probs_s_bsz
    log_phis_ptr = log_phis_ptr + b_idx * log_phis_s_bsz

    # Accumulator for the log phis.
    log_phis_acc = tl.where(j == 0, 0.0, tl.full((BLOCK_SIZE_C,), value=NEG_INF, dtype=tl.float32))

    # Stores first log phi value.
    log_phis_first_ptr = log_phis_ptr + j * log_phis_s_tgt
    tl.store(log_phis_first_ptr, log_phis_acc, mask=jmask)
    tl.debug_barrier()

    for i in range(1, t_i):
        log_probs_prev_ptr = log_probs_ptr + (i - 1) * log_probs_s_src + j * log_probs_s_tgt
        log_probs_prev = tl.load(log_probs_prev_ptr, mask=jmask)

        log_phis_prev_m1_ptr = log_phis_ptr + (i - 1) * log_phis_s_src + (j - 1) * log_phis_s_tgt
        log_probs_prev_m1_ptr = log_probs_ptr + (i - 1) * log_probs_s_src + (j - 1) * log_probs_s_tgt
        log_phis_prev_m1 = tl.load(log_phis_prev_m1_ptr, mask=jmask_shifted, other=NEG_INF)
        log_probs_prev_m1 = tl.load(log_probs_prev_m1_ptr, mask=jmask_shifted, other=NEG_INF)

        log_phis_a = log_phis_prev_m1 + log_1mexp(log_probs_prev_m1)
        log_phis_b = log_phis_acc + log_probs_prev
        log_phis_acc = logaddexp(log_phis_a, log_phis_b).to(tl.float32)

        log_phis_next_ptr = log_phis_ptr + i * log_phis_s_src + j * log_phis_s_tgt
        tl.store(log_phis_next_ptr, log_phis_acc, mask=jmask)

        # Barrier to ensure that we can access the stored log phis from the
        # adjacent thread in the next iteration.
        tl.debug_barrier()


@triton.jit
def backward_pass_kernel(
    # Log probabilities tensor (input)
    log_probs_ptr,
    log_probs_stride_bsz,
    log_probs_stride_src,
    log_probs_stride_tgt,
    # Log phis tensor (input)
    log_phis_ptr,
    log_phis_s_bsz,
    log_phis_s_src,
    log_phis_s_tgt,
    # Gradient of log phis tensor (input)
    grad_log_phis_ptr,
    grad_log_phis_s_bsz,
    grad_log_phis_s_src,
    grad_log_phis_s_tgt,
    # Gradient of log probabilities tensor (output)
    grad_log_probs_ptr,
    grad_log_probs_s_bsz,
    grad_log_probs_s_src,
    grad_log_probs_s_tgt,
    # Tensor dimensions
    t_i,
    t_j,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Parallelize over the batch dimension.
    b_idx = tl.program_id(0)
    j_idx = tl.program_id(1)

    # Pointers to the log probabilities.
    j = (j_idx * BLOCK_SIZE_C) + tl.arange(0, BLOCK_SIZE_C)
    jmask = j < t_j
    jmask_shifted = j < (t_j - 1)

    # Gets pointers offset for the current batch.
    log_probs_ptr = log_probs_ptr + b_idx * log_probs_stride_bsz
    log_phis_ptr = log_phis_ptr + b_idx * log_phis_s_bsz
    grad_log_phis_ptr = grad_log_phis_ptr + b_idx * grad_log_phis_s_bsz
    grad_log_probs_ptr = grad_log_probs_ptr + b_idx * grad_log_probs_s_bsz

    # Stores first log phi value.
    grad_log_probs_last_ptr = grad_log_probs_ptr + (t_i - 1) * log_phis_s_src + j * log_phis_s_tgt
    tl.store(grad_log_probs_last_ptr, tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32), mask=j < t_j)
    tl.debug_barrier()

    for i in range(t_i - 2, -1, -1):
        # log_phis[..., i + 1, :]
        log_phis_next_ptr = log_phis_ptr + (i + 1) * log_phis_s_src + j * log_phis_s_tgt
        log_phis_next = tl.load(log_phis_next_ptr, mask=jmask)

        # log_phis[..., i + 1, 1:]
        log_phis_next_p1_ptr = log_phis_ptr + (i + 1) * log_phis_s_src + (j + 1) * log_phis_s_tgt
        log_phis_next_p1 = tl.load(log_phis_next_p1_ptr, mask=jmask_shifted, other=0.0)

        # log_phis[..., i, :]
        log_phis_cur_ptr = log_phis_ptr + i * log_phis_s_src + j * log_phis_s_tgt
        log_phis_cur = tl.load(log_phis_cur_ptr, mask=jmask)

        # log_probs[..., i, :]
        log_probs_cur_ptr = log_probs_ptr + i * log_probs_stride_src + j * log_probs_stride_tgt
        log_probs_cur = tl.load(log_probs_cur_ptr, mask=jmask)

        # grad_log_phis[..., i + 1, :]
        grad_log_phis_next_ptr = grad_log_phis_ptr + (i + 1) * grad_log_phis_s_src + j * grad_log_phis_s_tgt
        grad_log_phis_next = tl.load(grad_log_phis_next_ptr, mask=jmask)

        # grad_log_phis[..., i + 1, 1:]
        grad_log_phis_next_p1_ptr = grad_log_phis_ptr + (i + 1) * grad_log_phis_s_src + (j + 1) * grad_log_phis_s_tgt
        grad_log_phis_next_p1 = tl.load(grad_log_phis_next_p1_ptr, mask=jmask_shifted, other=NEG_INF)

        # grad_log_probs[..., i, :]
        grad_log_probs_cur_ptr = grad_log_probs_ptr + i * grad_log_probs_s_src + j * grad_log_probs_s_tgt

        # grad_log_phis[..., i, :]
        grad_log_phis_cur_ptr = grad_log_phis_ptr + i * grad_log_phis_s_src + j * grad_log_phis_s_tgt
        grad_log_phis_cur = tl.load(grad_log_phis_cur_ptr, mask=jmask)

        # Computes the new values.
        a = tl.math.exp(log_phis_cur + log_probs_cur - log_phis_next)
        b = tl.math.exp(log_phis_cur + log_1mexp(log_probs_cur) - log_phis_next_p1)
        c = grad_log_phis_next * a
        d = grad_log_phis_next_p1 * b
        grad_log_probs_cur = tl.where(jmask_shifted, c + d * d_log_1mexp(log_probs_cur), c)
        grad_log_phis_cur = grad_log_phis_cur + tl.where(jmask_shifted, c + d, c)

        # Stores the new values.
        tl.store(grad_log_probs_cur_ptr, grad_log_probs_cur, mask=jmask)
        tl.store(grad_log_phis_cur_ptr, grad_log_phis_cur, mask=jmask)

        # Barrier to ensure that we can access the stored log phis from the
        # adjacent thread in the next iteration.
        tl.debug_barrier()


def forward_pass_(log_probs: Tensor) -> Tensor:
    bsz, tsz_src, tsz_tgt = log_probs.shape

    # Sets the initial log phi values.
    log_phis = torch.empty_like(log_probs)

    block_size_c = get_block_size_c(tsz_src)

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        return (bsz, triton.cdiv(tsz_src, meta["BLOCK_SIZE_C"]))

    forward_pass_kernel[grid](
        # Log probabilities
        log_probs,
        log_probs.stride(0),
        log_probs.stride(1),
        log_probs.stride(2),
        # Log phis
        log_phis,
        log_phis.stride(0),
        log_phis.stride(1),
        log_phis.stride(2),
        # Tensor dimensions
        tsz_src,
        tsz_tgt,
        # Block size
        BLOCK_SIZE_C=block_size_c,
    )

    return log_phis


def backward_pass_(log_probs: Tensor, log_phis: Tensor, grad_log_phis: Tensor) -> Tensor:
    bsz, tsz_src, tsz_tgt = log_probs.shape

    # grad_log_probs = torch.empty_like(grad_log_phis)
    grad_log_probs = torch.full_like(grad_log_phis, -1000000.0)

    block_size_c = get_block_size_c(tsz_src)

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        return (bsz, triton.cdiv(tsz_src, meta["BLOCK_SIZE_C"]))

    backward_pass_kernel[grid](
        # Log probabilities
        log_probs,
        log_probs.stride(0),
        log_probs.stride(1),
        log_probs.stride(2),
        # Log phis
        log_phis,
        log_phis.stride(0),
        log_phis.stride(1),
        log_phis.stride(2),
        # Gradient of log phis
        grad_log_phis,
        grad_log_phis.stride(0),
        grad_log_phis.stride(1),
        grad_log_phis.stride(2),
        # Gradient of log probabilities
        grad_log_probs,
        grad_log_probs.stride(0),
        grad_log_probs.stride(1),
        grad_log_probs.stride(2),
        # Tensor dimensions
        tsz_src,
        tsz_tgt,
        # Block size
        BLOCK_SIZE_C=block_size_c,
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
    _, tsz_src, tsz_tgt = probs.size()
    if tsz_tgt > tsz_src:
        warnings.warn("One-to-many attention expects the source sequence to be longer than the target sequence!")
    probs = (probs * (1 - 2 * epsilon)) + epsilon
    return MonotonicAttention.apply(probs.log()).exp()

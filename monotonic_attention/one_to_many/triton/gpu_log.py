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

MIN_LOG_PROB = -1e4


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
def pos_log_prob(x):
    return -tl.math.log1p(tl.math.exp(-x))


@triton.jit
def neg_log_prob(x):
    return -tl.math.log1p(tl.math.exp(x))


@triton.jit
def d_pos_log_prob(x):
    return 1 / (1 + tl.math.exp(x))


@triton.jit
def d_neg_log_prob(x):
    return -1 / (1 + tl.math.exp(-x))


@triton.jit
def forward_pass_kernel(
    # Log probabilities tensor (input)
    logits_ptr,
    logits_s_bsz,
    logits_s_src,
    logits_s_tgt,
    # Log phis tensor (output)
    phis_ptr,
    phis_s_bsz,
    phis_s_src,
    phis_s_tgt,
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
    logits_ptr = logits_ptr + b_idx * logits_s_bsz
    phis_ptr = phis_ptr + b_idx * phis_s_bsz

    # Accumulator for the log phis.
    phis_acc = tl.where(j == 0, 0.0, tl.full((BLOCK_SIZE_C,), value=MIN_LOG_PROB, dtype=tl.float32))

    # Stores first log phi value.
    phis_first_ptr = phis_ptr + j * phis_s_tgt
    tl.store(phis_first_ptr, phis_acc, mask=jmask)
    tl.debug_barrier()

    for i in range(1, t_i):
        logits_prev_ptr = logits_ptr + (i - 1) * logits_s_src + j * logits_s_tgt
        logits_prev = tl.load(logits_prev_ptr, mask=jmask).to(tl.float32)

        phis_prev_m1_ptr = phis_ptr + (i - 1) * phis_s_src + (j - 1) * phis_s_tgt
        logits_prev_m1_ptr = logits_ptr + (i - 1) * logits_s_src + (j - 1) * logits_s_tgt
        phis_prev_m1 = tl.load(phis_prev_m1_ptr, mask=jmask_shifted, other=MIN_LOG_PROB).to(tl.float32)
        logits_prev_m1 = tl.load(logits_prev_m1_ptr, mask=jmask_shifted, other=MIN_LOG_PROB).to(tl.float32)

        phis_a = phis_prev_m1 + neg_log_prob(logits_prev_m1)
        phis_b = phis_acc + pos_log_prob(logits_prev)
        phis_acc = logaddexp(phis_a, phis_b)

        phis_next_ptr = phis_ptr + i * phis_s_src + j * phis_s_tgt
        tl.store(phis_next_ptr, phis_acc, mask=jmask)

        # Barrier to ensure that we can access the stored log phis from the
        # adjacent thread in the next iteration.
        tl.debug_barrier()


@triton.jit
def backward_pass_kernel(
    # Log probabilities tensor (input)
    logits_ptr,
    logits_stride_bsz,
    logits_stride_src,
    logits_stride_tgt,
    # Log phis tensor (input)
    phis_ptr,
    phis_s_bsz,
    phis_s_src,
    phis_s_tgt,
    # Gradient of log phis tensor (input)
    grad_phis_ptr,
    grad_phis_s_bsz,
    grad_phis_s_src,
    grad_phis_s_tgt,
    # Gradient of log probabilities tensor (output)
    grad_logits_ptr,
    grad_logits_s_bsz,
    grad_logits_s_src,
    grad_logits_s_tgt,
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
    logits_ptr = logits_ptr + b_idx * logits_stride_bsz
    phis_ptr = phis_ptr + b_idx * phis_s_bsz
    grad_phis_ptr = grad_phis_ptr + b_idx * grad_phis_s_bsz
    grad_logits_ptr = grad_logits_ptr + b_idx * grad_logits_s_bsz

    # Stores first log phi value.
    grad_logits_last_ptr = grad_logits_ptr + (t_i - 1) * phis_s_src + j * phis_s_tgt
    tl.store(grad_logits_last_ptr, tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32), mask=j < t_j)
    tl.debug_barrier()

    for i in range(t_i - 2, -1, -1):
        # phis[..., i + 1, :]
        phis_next_ptr = phis_ptr + (i + 1) * phis_s_src + j * phis_s_tgt
        phis_next = tl.load(phis_next_ptr, mask=jmask)

        # phis[..., i + 1, 1:]
        phis_next_p1_ptr = phis_ptr + (i + 1) * phis_s_src + (j + 1) * phis_s_tgt
        phis_next_p1 = tl.load(phis_next_p1_ptr, mask=jmask_shifted, other=0.0)

        # phis[..., i, :]
        phis_cur_ptr = phis_ptr + i * phis_s_src + j * phis_s_tgt
        phis_cur = tl.load(phis_cur_ptr, mask=jmask)

        # logits[..., i, :]
        logits_cur_ptr = logits_ptr + i * logits_stride_src + j * logits_stride_tgt
        logits_cur = tl.load(logits_cur_ptr, mask=jmask).to(tl.float32)

        # grad_phis[..., i + 1, :]
        grad_phis_next_ptr = grad_phis_ptr + (i + 1) * grad_phis_s_src + j * grad_phis_s_tgt
        grad_phis_next = tl.load(grad_phis_next_ptr, mask=jmask)

        # grad_phis[..., i + 1, 1:]
        grad_phis_next_p1_ptr = grad_phis_ptr + (i + 1) * grad_phis_s_src + (j + 1) * grad_phis_s_tgt
        grad_phis_next_p1 = tl.load(grad_phis_next_p1_ptr, mask=jmask_shifted, other=MIN_LOG_PROB)

        # grad_logits[..., i, :]
        grad_logits_cur_ptr = grad_logits_ptr + i * grad_logits_s_src + j * grad_logits_s_tgt

        # grad_phis[..., i, :]
        grad_phis_cur_ptr = grad_phis_ptr + i * grad_phis_s_src + j * grad_phis_s_tgt
        grad_phis_cur = tl.load(grad_phis_cur_ptr, mask=jmask)

        # Computes the new values.
        a = tl.math.exp(phis_cur + pos_log_prob(logits_cur) - phis_next)
        b = tl.math.exp(phis_cur + neg_log_prob(logits_cur) - phis_next_p1)
        c = grad_phis_next * a
        d = grad_phis_next_p1 * b
        grad_logits_cur = tl.where(
            jmask_shifted,
            c * d_pos_log_prob(logits_cur) + d * d_neg_log_prob(logits_cur),
            c * d_pos_log_prob(logits_cur),
        )
        grad_phis_cur = grad_phis_cur + tl.where(jmask_shifted, c + d, c)

        # Stores the new values.
        tl.store(grad_logits_cur_ptr, grad_logits_cur, mask=jmask)
        tl.store(grad_phis_cur_ptr, grad_phis_cur, mask=jmask)

        # Barrier to ensure that we can access the stored log phis from the
        # adjacent thread in the next iteration.
        tl.debug_barrier()


def forward_pass_(logits: Tensor) -> Tensor:
    bsz, tsz_src, tsz_tgt = logits.shape

    # Sets the initial log phi values.
    phis = torch.empty_like(logits)

    block_size_c = get_block_size_c(tsz_src)

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        return (bsz, triton.cdiv(tsz_src, meta["BLOCK_SIZE_C"]))

    forward_pass_kernel[grid](
        # Log probabilities
        logits,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        # Log phis
        phis,
        phis.stride(0),
        phis.stride(1),
        phis.stride(2),
        # Tensor dimensions
        tsz_src,
        tsz_tgt,
        # Block size
        BLOCK_SIZE_C=block_size_c,
    )

    return phis


def backward_pass_(logits: Tensor, phis: Tensor, grad_phis: Tensor) -> Tensor:
    bsz, tsz_src, tsz_tgt = logits.shape

    grad_logits = torch.full_like(grad_phis, MIN_LOG_PROB)

    # We need to duplicate the phis tensor because the kernel updates it.
    grad_phis = grad_phis.clone()

    block_size_c = get_block_size_c(tsz_src)

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        return (bsz, triton.cdiv(tsz_src, meta["BLOCK_SIZE_C"]))

    backward_pass_kernel[grid](
        # Log probabilities
        logits,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        # Log phis
        phis,
        phis.stride(0),
        phis.stride(1),
        phis.stride(2),
        # Gradient of log phis
        grad_phis,
        grad_phis.stride(0),
        grad_phis.stride(1),
        grad_phis.stride(2),
        # Gradient of log probabilities
        grad_logits,
        grad_logits.stride(0),
        grad_logits.stride(1),
        grad_logits.stride(2),
        # Tensor dimensions
        tsz_src,
        tsz_tgt,
        # Block size
        BLOCK_SIZE_C=block_size_c,
    )

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
        grad_logits = backward_pass_(logits, phis, grad_phis)
        return grad_logits


def monotonic_attention(logits: Tensor) -> Tensor:
    """Computes the monotonic attention normalization on the transition probabilities.

    Args:
        logits: The transition logits, with shape ``(bsz, tsz_src, tsz_tgt)``

    Returns:
        The marginalized log probabilities for each cell being part of a
        monotonic alignment path, with shape ``(bsz, tsz_src, tsz_tgt)``.
    """
    _, tsz_src, tsz_tgt = logits.size()
    if tsz_tgt > tsz_src:
        warnings.warn("One-to-many attention expects the source sequence to be longer than the target sequence!")
    return MonotonicAttention.apply(logits)

"""Tests the normalization Triton kernel."""

import pytest
import torch

from monotonic_attention.one_to_many.cpu_log import monotonic_attention as monotonic_attention_cpu


def test_cpu_log() -> None:
    bsz, tsz_src, tsz_tgt = 2, 7, 5
    probs = torch.rand(bsz, tsz_src, tsz_tgt, dtype=torch.double)

    # Tests the forward pass.
    phis = monotonic_attention_cpu(probs)

    assert (phis >= 0).all()
    assert (phis <= 1).all()

    # Tests the backward pass using finite differences.
    probs.requires_grad_(True)
    torch.autograd.gradcheck(monotonic_attention_cpu, probs, fast_mode=True)


@pytest.mark.has_triton()
def test_gpu_log() -> None:
    from monotonic_attention.one_to_many.triton.gpu_log import monotonic_attention as monotonic_attention_gpu

    bsz, tsz_src, tsz_tgt = 2, 7, 5
    probs = torch.rand(bsz, tsz_src, tsz_tgt, dtype=torch.double, device="cuda")

    # Randomly set some values to 0 and 1, to test edge cases.
    probs[torch.rand_like(probs) < 0.1] = 0.0
    probs[torch.rand_like(probs) < 0.1] = 1.0

    # Tests the forward pass.
    phis = monotonic_attention_gpu(probs)
    phis_ref = monotonic_attention_cpu(probs)
    assert torch.allclose(phis, phis_ref)

    # Tests the backward pass matches the reference implementation.
    probs.requires_grad_(True)
    phis = monotonic_attention_gpu(probs)
    grad_phis = torch.randn_like(phis)
    torch.autograd.backward(phis, grad_phis)
    assert probs.grad is not None
    grad_probs = probs.grad.clone()
    probs.grad = None
    phis_ref = monotonic_attention_cpu(probs)
    torch.autograd.backward(phis_ref, grad_phis)
    assert probs.grad is not None
    grad_probs_ref = probs.grad.clone()
    assert torch.allclose(grad_probs, grad_probs_ref)


if __name__ == "__main__":
    # python -m tests.test_one_to_many
    test_gpu_log()

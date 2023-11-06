"""Tests the normalization Triton kernel."""

import torch

from monotonic_attention.many_to_many.cpu_log import monotonic_attention as monotonic_attention_log
from monotonic_attention.many_to_many.cpu_log_faster import monotonic_attention as monotonic_attention_log_faster
from monotonic_attention.many_to_many.cpu_simple import monotonic_attention as monotonic_attention_simple
from monotonic_attention.many_to_many.cpu_simple_faster import monotonic_attention as monotonic_attention_simple_faster


def test_cpu_simple() -> None:
    bsz, tsz_src, tsz_tgt = 2, 5, 7
    probs = torch.rand(bsz, tsz_src, tsz_tgt, dtype=torch.double)

    # Tests the forward pass.
    phis = monotonic_attention_simple(probs)
    assert (phis >= 0).all()
    assert (phis <= 1).all()

    # Tests the backward pass using finite differences.
    probs.requires_grad_(True)
    torch.autograd.gradcheck(monotonic_attention_simple, probs)


def test_cpu_simple_faster() -> None:
    bsz, tsz_src, tsz_tgt = 2, 4, 5
    probs = torch.rand(bsz, tsz_src, tsz_tgt, dtype=torch.double)

    # Tests the forward pass.
    phis = monotonic_attention_simple_faster(probs)
    phis_ref = monotonic_attention_simple(probs)
    assert torch.allclose(phis, phis_ref)

    # Tests the backward pass.
    probs.requires_grad_(True)
    phis = monotonic_attention_simple_faster(probs)
    grad_phis = torch.randn_like(phis)
    torch.autograd.backward(phis, grad_phis)
    assert probs.grad is not None
    grad_probs = probs.grad.clone()
    probs.grad = None
    phis_ref = monotonic_attention_simple(probs)
    torch.autograd.backward(phis_ref, grad_phis)
    assert probs.grad is not None
    grad_probs_ref = probs.grad.clone()
    assert torch.allclose(grad_probs, grad_probs_ref)


def test_cpu_log() -> None:
    bsz, tsz_src, tsz_tgt = 2, 4, 5
    probs = torch.rand(bsz, tsz_src, tsz_tgt, dtype=torch.double)

    # Tests the forward pass.
    phis = monotonic_attention_log(probs)
    phis_ref = monotonic_attention_simple(probs)
    assert torch.allclose(phis, phis_ref, atol=1e-2)

    # Tests the backward pass.
    probs.requires_grad_(True)
    phis = monotonic_attention_log(probs)
    grad_phis = torch.randn_like(phis)
    torch.autograd.backward(phis, grad_phis)
    assert probs.grad is not None
    grad_probs = probs.grad.clone()
    probs.grad = None
    phis_ref = monotonic_attention_simple(probs)
    torch.autograd.backward(phis_ref, grad_phis)
    assert probs.grad is not None
    grad_probs_ref = probs.grad.clone()
    assert torch.allclose(grad_probs, grad_probs_ref, atol=1e-1)


def test_cpu_log_faster() -> None:
    bsz, tsz_src, tsz_tgt = 2, 20, 50
    probs = torch.rand(bsz, tsz_src, tsz_tgt, dtype=torch.double)

    # Tests the forward pass.
    phis = monotonic_attention_log_faster(probs)
    phis_ref = monotonic_attention_simple(probs)
    assert torch.allclose(phis, phis_ref, atol=1e-2)

    # Tests the backward pass.
    probs.requires_grad_(True)
    phis = monotonic_attention_log_faster(probs)
    grad_phis = torch.randn_like(phis)
    torch.autograd.backward(phis, grad_phis)
    assert probs.grad is not None
    grad_probs = probs.grad.clone()
    probs.grad = None
    phis_ref = monotonic_attention_simple(probs)
    torch.autograd.backward(phis_ref, grad_phis)
    assert probs.grad is not None
    grad_probs_ref = probs.grad.clone()
    assert torch.allclose(grad_probs, grad_probs_ref, atol=1e-1)

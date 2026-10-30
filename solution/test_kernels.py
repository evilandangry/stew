"""Correctness of your kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`. Add a test like this for every kernel you write.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


def _qwen2_rmsnorm_reference(x, weight, eps):
    """A line-for-line copy of transformers' Qwen2RMSNorm.forward."""
    input_dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return weight * h.to(input_dtype)


# (x dtype, weight dtype): the second pair is the usual QLoRA case, fp16 activations with an
# fp32 norm weight, where the output is fp32.
@pytest.mark.parametrize(
    "x_dtype,w_dtype",
    [(torch.float16, torch.float16), (torch.float16, torch.float32), (torch.float32, torch.float32)],
)
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1000)])
@pytest.mark.parametrize("weight_requires_grad", [False, True])
def test_rmsnorm_matches_pytorch(x_dtype, w_dtype, shape, weight_requires_grad):
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(0)
    eps = 1e-6
    n = shape[-1]
    x0 = torch.randn(shape, device="cuda", dtype=x_dtype) * 3
    w0 = (torch.randn(n, device="cuda", dtype=w_dtype) * 0.5 + 1.0)
    grad = torch.randn(shape, device="cuda", dtype=torch.promote_types(x_dtype, w_dtype))

    x_ref = x0.clone().requires_grad_(True)
    w_ref = w0.clone().requires_grad_(weight_requires_grad)
    ref = _qwen2_rmsnorm_reference(x_ref, w_ref, eps)
    ref.backward(grad)

    x_new = x0.clone().requires_grad_(True)
    w_new = w0.clone().requires_grad_(weight_requires_grad)
    out = rmsnorm(x_new, w_new, eps)
    out.backward(grad)

    assert out.dtype == ref.dtype
    assert x_new.grad.dtype == x_ref.grad.dtype
    fwd_tol = 1e-4 if x_dtype == torch.float32 else 5e-3
    bwd_tol = 1e-4 if x_dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out.float(), ref.float(), rtol=fwd_tol, atol=fwd_tol)
    torch.testing.assert_close(x_new.grad.float(), x_ref.grad.float(), rtol=bwd_tol, atol=bwd_tol)
    if weight_requires_grad:
        torch.testing.assert_close(w_new.grad.float(), w_ref.grad.float(), rtol=1e-2, atol=1e-1)

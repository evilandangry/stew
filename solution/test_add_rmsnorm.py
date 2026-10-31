"""Fused residual add + post-attention RMSNorm against plain PyTorch."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def ref_rmsnorm(x, w, eps):
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dt)


@pytest.mark.parametrize("out16", [True, False])
@pytest.mark.parametrize("shape", [(3, 37, 1536), (1, 838, 1536)])  # padded batch / packed tokens
def test_add_rmsnorm_matches_pytorch(out16, shape):
    from solution.rmsnorm import AddRMSNormFn

    torch.manual_seed(0)
    r = (torch.randn(shape, device="cuda") * 2).requires_grad_()       # fp32 residual stream
    a = torch.randn(shape, device="cuda").half().requires_grad_()      # fp16 attention output
    r_ref = r.detach().clone().requires_grad_()
    a_ref = a.detach().clone().requires_grad_()
    w = torch.rand(shape[-1], device="cuda") + 0.5
    dy = torch.randn(shape, device="cuda")
    ds = torch.randn(shape, device="cuda")

    s, y = AddRMSNormFn.apply(r, a, w, 1e-6, out16)
    s_ref = r_ref + a_ref
    y_ref = ref_rmsnorm(s_ref, w, 1e-6)
    assert s.dtype == torch.float32
    assert y.dtype == (torch.float16 if out16 else torch.float32)
    torch.testing.assert_close(s, s_ref, rtol=0, atol=0)  # the add itself is exact
    torch.testing.assert_close(y.float(), y_ref, rtol=5e-3, atol=5e-3)

    ((s * ds).sum() + (y.float() * dy).sum()).backward()
    ((s_ref * ds).sum() + (y_ref * dy).sum()).backward()
    torch.testing.assert_close(r.grad, r_ref.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(a.grad.float(), a_ref.grad.float(), rtol=3e-2, atol=3e-2)

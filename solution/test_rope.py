import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rope_reference(x, cos, sin):
    """The body of transformers' apply_rotary_pos_emb, for one tensor."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


# (x dtype, cos/sin dtype): fp16 with fp32 tables promotes the output to fp32.
@pytest.mark.parametrize(
    "x_dtype,c_dtype",
    [(torch.float16, torch.float16), (torch.float16, torch.float32), (torch.float32, torch.float32)],
)
@pytest.mark.parametrize("heads", [12, 2])
@pytest.mark.parametrize("cos_batch", [4, 1])
def test_rope_matches_pytorch(x_dtype, c_dtype, heads, cos_batch):
    from solution.rope import rope

    torch.manual_seed(0)
    batch, seq, dim = 4, 77, 128
    # q and k reach rope as transposed views of [B, T, H, D], so test a non-contiguous input
    base = torch.randn(batch, seq, heads, dim, device="cuda", dtype=x_dtype) * 2
    x0 = base.transpose(1, 2)
    cos = torch.randn(cos_batch, seq, dim, device="cuda", dtype=c_dtype)
    sin = torch.randn(cos_batch, seq, dim, device="cuda", dtype=c_dtype)
    out_dtype = torch.promote_types(x_dtype, c_dtype)
    grad = torch.randn(batch, heads, seq, dim, device="cuda", dtype=out_dtype)

    x_ref = x0.clone().requires_grad_(True)
    ref = _rope_reference(x_ref, cos, sin)
    ref.backward(grad)

    x_new = x0.clone().requires_grad_(True)
    out = rope(x_new, cos, sin)
    out.backward(grad)

    assert out.dtype == ref.dtype
    assert x_new.grad.dtype == x_ref.grad.dtype
    tol = 1e-4 if x_dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out.float(), ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(x_new.grad.float(), x_ref.grad.float(), rtol=tol, atol=tol)


@pytest.mark.parametrize("heads", [12, 2])
def test_rope_fp16_output_matches_autocast_rounding(heads):
    """With out_dtype=fp16 the kernel returns what autocast would hand to the attention: the fp32
    rotation rounded once to fp16. The backward then takes and returns fp16."""
    from solution.rope import rope

    torch.manual_seed(0)
    batch, seq, dim = 4, 77, 128
    base = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.float16) * 2
    x0 = base.transpose(1, 2)
    cos = torch.randn(1, seq, dim, device="cuda", dtype=torch.float32)
    sin = torch.randn(1, seq, dim, device="cuda", dtype=torch.float32)
    grad = torch.randn(batch, heads, seq, dim, device="cuda", dtype=torch.float16)

    x_ref = x0.clone().requires_grad_(True)
    ref = _rope_reference(x_ref, cos, sin).half()  # what autocast feeds the attention
    ref.backward(grad)

    x_new = x0.clone().requires_grad_(True)
    out = rope(x_new, cos, sin, torch.float16)
    out.backward(grad)

    assert out.dtype == torch.float16 and x_new.grad.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x_new.grad.float(), x_ref.grad.float(), rtol=1e-2, atol=1e-2)

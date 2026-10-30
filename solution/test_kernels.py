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


def _ce_reference(logits, labels, num_items=None):
    """transformers' ForCausalLMLoss, plus the entropy and per-token loss we also produce."""
    vocab = logits.shape[-1]
    x = logits.detach().float().requires_grad_(True)
    shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    flat = x.view(-1, vocab)
    tgt = shifted.view(-1)
    if num_items is None:
        loss = torch.nn.functional.cross_entropy(flat, tgt, ignore_index=-100)
    else:
        loss = torch.nn.functional.cross_entropy(flat, tgt, ignore_index=-100, reduction="sum")
        loss = loss / num_items
    lse = torch.logsumexp(flat.detach(), dim=-1)
    p = torch.softmax(flat.detach(), dim=-1)
    entropy = (lse - (p * flat.detach()).sum(-1)).view(shifted.shape)
    tok = torch.nn.functional.cross_entropy(flat.detach(), tgt, ignore_index=-100,
                                            reduction="none").view(shifted.shape)
    return x, loss, entropy, tok


@pytest.mark.parametrize("shape", [(3, 77, 1001), (2, 64, 151936)])
@pytest.mark.parametrize("num_items", [None, 150])
def test_fused_cross_entropy_matches_pytorch(shape, num_items):
    from solution.loss import fused_for_causal_lm_loss

    torch.manual_seed(0)
    batch, seq, vocab = shape
    logits0 = (torch.randn(shape, device="cuda", dtype=torch.float16) * 3)
    labels = torch.randint(0, vocab, (batch, seq), device="cuda")
    labels[:, : seq // 3] = -100  # an unsupervised prefix, as in SFT
    labels[0, -5:] = -100

    x_ref, loss_ref, ent_ref, tok_ref = _ce_reference(logits0, labels, num_items)
    loss_ref.backward()

    logits = logits0.clone().requires_grad_(True)
    n = None if num_items is None else torch.tensor(num_items, device="cuda")
    loss = fused_for_causal_lm_loss(logits, labels, vocab, n)
    loss.backward()

    torch.testing.assert_close(loss.float(), loss_ref.float(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(logits.grad.float(), x_ref.grad.float(), rtol=3e-3, atol=2e-6)
    torch.testing.assert_close(logits._fused_ce_entropy, ent_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(logits._fused_ce_token_loss, tok_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("shape", [(512, 256), (1536, 8960)])
def test_nf4_dequant_matches_bitsandbytes(nested, shape):
    import bitsandbytes.functional as bnb_f
    from solution.dequant4 import _ORIGINAL_DEQUANTIZE_4BIT, dequantize_nf4

    torch.manual_seed(0)
    weight = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.05
    packed, state = bnb_f.quantize_4bit(weight, blocksize=64, compress_statistics=nested,
                                        quant_type="nf4")
    ref = _ORIGINAL_DEQUANTIZE_4BIT(packed, state)
    out = dequantize_nf4(packed, state)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    mismatch = (out != ref).float().mean().item()
    assert mismatch < 1e-4, f"{mismatch:.2e} of the values differ from bitsandbytes"
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("grad_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_lowprec_matches_pytorch(grad_dtype, shape):
    from solution.swiglu_lp import swiglu_lowprec

    torch.manual_seed(0)
    # the real inputs are fp16 matmul results widened to fp32, so build them the same way
    gate0 = torch.randn(shape, device="cuda", dtype=torch.float16).float()
    up0 = torch.randn(shape, device="cuda", dtype=torch.float16).float()
    grad = torch.randn(shape, device="cuda", dtype=grad_dtype)

    gate_ref = gate0.clone().requires_grad_(True)
    up_ref = up0.clone().requires_grad_(True)
    ref = torch.nn.functional.silu(gate_ref) * up_ref
    # the output is fp16, so autograd hands our backward the gradient rounded to fp16
    ref.backward(grad.half().float())

    gate = gate0.clone().requires_grad_(True)
    up = up0.clone().requires_grad_(True)
    out = swiglu_lowprec(gate, up)
    out.backward(grad)

    assert out.dtype == torch.float16
    assert gate.grad.dtype == torch.float32 and up.grad.dtype == torch.float32
    torch.testing.assert_close(out.float(), ref.detach(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(gate.grad, gate_ref.grad, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(up.grad, up_ref.grad, rtol=2e-4, atol=2e-4)

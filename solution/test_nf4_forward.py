"""The forward matmul of a 4-bit layer through our unpack matches bitsandbytes' fused op."""
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("with_bias", [True, False])
def test_matmul4bit_forward_matches_bitsandbytes(nested, with_bias):
    import bitsandbytes as bnb
    import bitsandbytes.functional as BF
    from bitsandbytes.autograd import _functions as BFN

    from solution import nf4

    torch.manual_seed(0)
    out_f, in_f = 512, 256
    W = torch.randn(out_f, in_f, device="cuda", dtype=torch.float16) * 0.05
    q, st = BF.quantize_4bit(W, blocksize=64, compress_statistics=nested, quant_type="nf4")
    bias = torch.randn(out_f, device="cuda", dtype=torch.float16) if with_bias else None
    x0 = torch.randn(2, 37, in_f, device="cuda", dtype=torch.float16)
    g = torch.randn(2, 37, out_f, device="cuda", dtype=torch.float16)

    saved_forward = BFN.MatMul4Bit.__dict__["forward"]
    saved_deq = BF.dequantize_4bit
    saved_mods = {n: m for n, m in sys.modules.items()
                  if n.startswith("bitsandbytes") and getattr(m, "dequantize_4bit", None) is saved_deq}

    # reference first, with bitsandbytes untouched (as Linear4bit calls it: weight.t())
    x_ref = x0.clone().requires_grad_(True)
    ref = bnb.matmul_4bit(x_ref, q.t(), quant_state=st, bias=bias)
    ref.backward(g)

    try:
        nf4.patch_bnb_dequant()
        assert getattr(BFN.MatMul4Bit.forward, "_soup_fused", False)
        x = x0.clone().requires_grad_(True)
        out = bnb.matmul_4bit(x, q.t(), quant_state=st, bias=bias)
        out.backward(g)
    finally:  # leave bitsandbytes as we found it for the other tests
        BFN.MatMul4Bit.forward = saved_forward
        BF.dequantize_4bit = saved_deq
        for m in saved_mods.values():
            m.dequantize_4bit = saved_deq

    assert out.shape == ref.shape and out.dtype == ref.dtype
    torch.testing.assert_close(out, ref.detach(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=2e-3, atol=2e-3)

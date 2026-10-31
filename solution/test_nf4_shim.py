"""Our dequantize_4bit replacement behaves like bitsandbytes for a (1, n) packed view too.

bitsandbytes has a compatibility rule: a packed weight passed as a (1, n) view gets its result
transposed back. The training run never passes such a view, but the
replacement should match the original everywhere.
"""
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("nested", [True, False])
def test_patched_dequantize_matches_bitsandbytes_for_both_layouts(nested):
    import bitsandbytes.functional as BF
    from bitsandbytes.autograd import _functions as BFN

    from solution import nf4

    torch.manual_seed(0)
    W = torch.randn(256, 1536, device="cuda", dtype=torch.float16) * 0.05
    q, st = BF.quantize_4bit(W, blocksize=64, compress_statistics=nested, quant_type="nf4")
    saved_forward = BFN.MatMul4Bit.__dict__["forward"]
    saved_deq = BF.dequantize_4bit
    saved_mods = [m for n, m in sys.modules.items()
                  if n.startswith("bitsandbytes") and getattr(m, "dequantize_4bit", None) is saved_deq]
    refs = [saved_deq(A, st) for A in (q, q.t())]
    try:
        nf4.patch_bnb_dequant()
        outs = [BF.dequantize_4bit(A, st) for A in (q, q.t())]
    finally:
        BFN.MatMul4Bit.forward = saved_forward
        BF.dequantize_4bit = saved_deq
        for m in saved_mods:
            m.dequantize_4bit = saved_deq
    for out, ref in zip(outs, refs):
        assert out.shape == ref.shape
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

"""Our NF4 unpack matches bitsandbytes value for value on the real layer shapes."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# Qwen2.5-1.5B: q/o (1536x1536), k/v (256x1536), gate/up (8960x1536), down (1536x8960);
# (100, 64) has 100 blocks, not a multiple of the 32-block tile, so the edge mask is used.
SHAPES = [(1536, 1536), (256, 1536), (8960, 1536), (1536, 8960), (100, 64)]


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("blocksize", [64, 128])
def test_nf4_unpack_matches_bitsandbytes(nested, shape, blocksize):
    import bitsandbytes.functional as BF

    from solution.nf4 import _supported, dequantize_nf4

    torch.manual_seed(0)
    W = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.05
    q, st = BF.quantize_4bit(W, blocksize=blocksize, compress_statistics=nested, quant_type="nf4")
    assert _supported(q, st)
    ref = BF.dequantize_4bit(q, st)
    if getattr(BF.dequantize_4bit, "_soup_fused", False):
        pytest.skip("bitsandbytes is patched in this process; reference would be our own kernel")
    out = dequantize_nf4(q, st)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    mismatch = (out != ref).float().mean().item()
    assert mismatch < 1e-4, f"{mismatch:.2e} of the values differ from bitsandbytes"
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_nf4_unpack_transposed_view():
    """Linear4bit passes weight.t(), a (1, n) view of the (n, 1) packed tensor."""
    import bitsandbytes.functional as BF

    from solution.nf4 import dequantize_nf4

    W = torch.randn(512, 256, device="cuda", dtype=torch.float16) * 0.05
    q, st = BF.quantize_4bit(W, blocksize=64, compress_statistics=True, quant_type="nf4")
    torch.testing.assert_close(dequantize_nf4(q.t(), st), dequantize_nf4(q, st), rtol=0, atol=0)

"""NF4 de-quantisation kernel in Triton (written from scratch).

bitsandbytes unpacks a 4-bit NF4 weight to a dense fp16 matrix before every matmul, in the
forward and again in the backward pass (~8% of the GPU time of a training step here). This file
does that unpacking with our own kernel, for both directions.

Format: each byte holds two 4-bit codes (high nibble first); every block of `blocksize` values
(64 here) has one scale ("absmax"). With double quantisation the scales are themselves stored as
8-bit codes: scale = code2[absmax8[block]] * absmax2[block // 256] + offset.
value = nf4_code[nibble] * scale, computed in fp32 and rounded once, exactly like bitsandbytes.

Layout of the kernel: one program handles NBLK whole quantisation blocks as a 2-D tile
[NBLK, blocksize/2 bytes]. The scale is decoded once per block (row), not once per byte, and the
two nibbles of each byte are interleaved in registers (tl.join + reshape) so every row is written
with one contiguous store instead of two strided ones.

Idea source: the NF4 format of QLoRA (Dettmers et al., 2023) and bitsandbytes' layout of it.
The kernel is our own code. Anything unusual (other quant type, shapes not a multiple of the
block, out= given, ...) falls back to the original bitsandbytes path.
"""
import torch
import triton
import triton.language as tl

_CODE = {}
_BROKEN = False
_NBLK = 32  # quantisation blocks per program: 32 x 64 values = 2048 outputs


@triton.jit
def _nf4_unpack(A, ABS, ABS8, CODE2, ABS2, OFF, CODE, OUT, n_blocks, bs2,
                NESTED: tl.constexpr, HB: tl.constexpr, NBLK: tl.constexpr):
    pid = tl.program_id(0)
    rows = (pid * NBLK + tl.arange(0, NBLK)).to(tl.int64)  # quantisation blocks of this tile
    rmask = rows < n_blocks
    if NESTED:
        q = tl.load(ABS8 + rows, mask=rmask, other=0).to(tl.int32)
        outer = tl.load(ABS2 + rows // bs2, mask=rmask, other=0.0)
        scale = tl.load(CODE2 + q) * outer + tl.load(OFF)
    else:
        scale = tl.load(ABS + rows, mask=rmask, other=0.0)

    cols = tl.arange(0, HB)
    b = tl.load(A + rows[:, None] * HB + cols[None, :], mask=rmask[:, None], other=0).to(tl.int32)
    hi = tl.load(CODE + (b >> 4)) * scale[:, None]   # first value of each pair
    lo = tl.load(CODE + (b & 15)) * scale[:, None]   # second value
    vals = tl.reshape(tl.join(hi, lo), (NBLK, 2 * HB))  # hi0, lo0, hi1, lo1, ...
    ocols = tl.arange(0, 2 * HB)
    tl.store(OUT + rows[:, None] * (2 * HB) + ocols[None, :],
             vals.to(OUT.dtype.element_ty), mask=rmask[:, None])


def _code(device):
    key = str(device)
    if key not in _CODE:
        import bitsandbytes.functional as BF
        _CODE[key] = BF.get_4bit_type("nf4", device=device).to(torch.float32).contiguous()
    return _CODE[key]


def _supported(A, st):
    """True when our kernel can unpack packed weight A with quant state st."""
    return (st is not None and A.is_cuda and A.dtype == torch.uint8
            and getattr(st, "quant_type", None) == "nf4"
            and st.dtype in (torch.float16, torch.float32)
            and st.blocksize in (32, 64, 128, 256)
            and len(st.shape) == 2
            and st.shape.numel() == 2 * A.numel()
            and st.shape.numel() % st.blocksize == 0
            and (not st.nested or st.state2.blocksize == 256))


def dequantize_nf4(A, st):
    """Unpacked weight of shape st.shape and dtype st.dtype (callers check _supported first)."""
    flat = A.reshape(-1)
    out = torch.empty(st.shape, dtype=st.dtype, device=A.device)
    n_blocks = st.shape.numel() // st.blocksize
    grid = (triton.cdiv(n_blocks, _NBLK),)
    code = _code(A.device)
    if st.nested:
        s2 = st.state2
        absmax8 = st.absmax.contiguous()
        offset = torch.as_tensor(st.offset, dtype=torch.float32, device=A.device).reshape(1)
        _nf4_unpack[grid](flat, absmax8, absmax8, s2.code.float().contiguous(),
                          s2.absmax.float().contiguous(), offset, code, out, n_blocks,
                          s2.blocksize, NESTED=True, HB=st.blocksize // 2, NBLK=_NBLK,
                          num_warps=4)
    else:
        absmax = st.absmax.float().contiguous()
        _nf4_unpack[grid](flat, absmax, absmax, absmax, absmax, absmax, code, out, n_blocks,
                          1, NESTED=False, HB=st.blocksize // 2, NBLK=_NBLK, num_warps=4)
    return out


def patch_bnb_dequant():
    import bitsandbytes.functional as BF

    orig = BF.dequantize_4bit
    if getattr(orig, "_soup_fused", False):
        return

    def dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=64,
                        quant_type="fp4"):
        global _BROKEN
        if (not _BROKEN and absmax is None and out is None and _supported(A, quant_state)):
            try:
                out_ = dequantize_nf4(A, quant_state)
                # like bitsandbytes: a (1, n) packed view gets the result transposed back
                return out_.t() if A.shape[0] == 1 else out_
            except Exception:  # never break training because of the speed-up
                _BROKEN = True
        return orig(A, quant_state, absmax, out, blocksize, quant_type)

    dequantize_4bit._soup_fused = True
    BF.dequantize_4bit = dequantize_4bit
    import sys
    for name, mod in list(sys.modules.items()):
        if name.startswith("bitsandbytes") and getattr(mod, "dequantize_4bit", None) is orig:
            mod.dequantize_4bit = dequantize_4bit
    _patch_matmul4bit_forward()


def _patch_matmul4bit_forward():
    """Send the FORWARD matmul of 4-bit layers through our unpack too.

    bitsandbytes 0.50 runs the forward of a 4-bit layer through its own fused op
    (torch.ops.bitsandbytes.gemm_4bit: unpack + matmul), which never calls dequantize_4bit, so
    the patch above only reached the backward. The profile showed bitsandbytes' own unpack
    kernel (kDequantizeBlockwise) still running once per layer per step next to ours.
    Here MatMul4Bit.forward does the same two steps with our kernel: unpack, then a normal fp16
    matmul with the bias. It saves the same state as the original, so bitsandbytes' backward
    runs unchanged. Anything unusual (out= given, empty input, other quant types) keeps the
    original path.
    """
    from bitsandbytes.autograd import _functions as BFN

    cls = BFN.MatMul4Bit
    orig = cls.forward
    if getattr(orig, "_soup_fused", False):
        return

    def forward(ctx, A, B, out=None, bias=None, quant_state=None):
        st = quant_state
        if (not _BROKEN and out is None and A.is_cuda and A.numel() > A.shape[-1]
                and _supported(B, st)):
            Bv = B.view(-1, 1)  # the same canonical layout bitsandbytes uses
            try:
                weight = dequantize_nf4(Bv, st).to(A.dtype)
            except Exception:  # never break training because of the speed-up
                weight = None
            if weight is not None:
                output = torch.nn.functional.linear(A, weight, bias)
                ctx.is_empty = False
                ctx.state = st
                ctx.dtype_A, ctx.dtype_B = A.dtype, Bv.dtype
                ctx.dtype_bias = None if bias is None else bias.dtype
                ctx.tensors = (None, Bv) if any(ctx.needs_input_grad[:2]) else (None, None)
                return output
        return orig(ctx, A, B, out, bias, quant_state)

    forward._soup_fused = True
    cls.forward = staticmethod(forward)

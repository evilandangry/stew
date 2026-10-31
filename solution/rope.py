"""Fused rotary position embedding (RoPE) for Qwen2, forward and backward in Triton.

transformers computes  x * cos + rotate_half(x) * sin  with about ten separate kernels per
tensor (a neg, a cat, two multiplies, an add, plus dtype promotions), and autograd then runs a
longer chain backwards. Here one program rotates one (batch, head, position) row: it reads the
row once and writes the result once.

Math, with x = [x1, x2] split in half along the head dim and the same split for cos and sin:
    forward : y1 = x1*c1 - x2*s1          y2 = x2*c2 + x1*s2
    backward: dx1 = g1*c1 + g2*s2         dx2 = g2*c2 - g1*s1
(the backward is the transpose of the forward map, since rotate_half(x) = [-x2, x1]).

Written from scratch from the Qwen2 source of apply_rotary_pos_emb; no code taken from other
kernels. Output dtype follows PyTorch's promotion of the inputs, as in the original.
"""

import torch
import triton
import triton.language as tl


# T (tokens) changes every batch, H differs between q and k, and INVERSE picks forward/backward:
# none of them gets a separate compiled copy (forward and backward share one)
@triton.jit(do_not_specialize=["H", "T", "INVERSE"])
def rope_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr, H, T,
                sxb, sxh, sxt, syb, syh, syt, scb, sct, INVERSE,
                HALF: tl.constexpr):
    pid = tl.program_id(0)  # one program per (batch, head, position)
    t = pid % T
    bh = pid // T
    h = bh % H
    b = bh // H
    cols = tl.arange(0, HALF)
    x_off = b * sxb + h * sxh + t * sxt
    y_off = b * syb + h * syh + t * syt
    c_off = b * scb + t * sct
    x1 = tl.load(x_ptr + x_off + cols).to(tl.float32)
    x2 = tl.load(x_ptr + x_off + HALF + cols).to(tl.float32)
    c1 = tl.load(cos_ptr + c_off + cols).to(tl.float32)
    c2 = tl.load(cos_ptr + c_off + HALF + cols).to(tl.float32)
    s1 = tl.load(sin_ptr + c_off + cols).to(tl.float32)
    s2 = tl.load(sin_ptr + c_off + HALF + cols).to(tl.float32)
    if INVERSE != 0:
        y1 = x1 * c1 + x2 * s2
        y2 = x2 * c2 - x1 * s1
    else:
        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2
    out_ty = y_ptr.dtype.element_ty
    tl.store(y_ptr + y_off + cols, y1.to(out_ty))
    tl.store(y_ptr + y_off + HALF + cols, y2.to(out_ty))


def _rope_launch(x, cos, sin, out_dtype, inverse):
    if x.stride(-1) != 1:
        x = x.contiguous()
    batch, heads, seq, dim = x.shape
    y = torch.empty_like(x, dtype=out_dtype)  # keeps x's strides when x is a permuted dense view
    if y.stride(-1) != 1:
        y = torch.empty(x.shape, device=x.device, dtype=out_dtype)
    scb = cos.stride(0) if cos.shape[0] > 1 else 0  # cos/sin may have batch size 1
    rope_kernel[(batch * heads * seq,)](
        x, cos, sin, y, heads, seq,
        x.stride(0), x.stride(1), x.stride(2), y.stride(0), y.stride(1), y.stride(2),
        scb, cos.stride(1), 1 if inverse else 0,
        HALF=dim // 2, num_warps=1,
    )
    return y


class RopeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, out_dtype=None):
        if out_dtype is None:  # PyTorch's promotion, as in the original
            out_dtype = torch.promote_types(torch.promote_types(x.dtype, cos.dtype), sin.dtype)
        ctx.save_for_backward(cos, sin)
        ctx.x_dtype = x.dtype
        return _rope_launch(x, cos, sin, out_dtype, inverse=False)

    @staticmethod
    def backward(ctx, grad):
        cos, sin = ctx.saved_tensors
        return _rope_launch(grad, cos, sin, ctx.x_dtype, inverse=True), None, None, None


def rope(x, cos, sin, out_dtype=None):
    """x: [B, H, T, D]; cos, sin: [B or 1, T, D]. out_dtype=None follows PyTorch's promotion."""
    if cos.stride(-1) != 1:
        cos = cos.contiguous()
    if sin.stride(-1) != 1:
        sin = sin.contiguous()
    return RopeFn.apply(x, cos, sin, out_dtype)


def patch_qwen2_rope():
    """Send apply_rotary_pos_emb through the fused kernel. The attention module looks the function
    up by name on every call, so replacing it on the module works before or after Soup builds the
    model."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.apply_rotary_pos_emb

    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        half = q.shape[-1] // 2
        supported = (
            unsqueeze_dim == 1 and q.dim() == 4 and k.dim() == 4 and cos.dim() == 3
            and q.is_cuda and q.shape[-1] == k.shape[-1] and half > 0 and (half & (half - 1)) == 0
        )
        if not supported:
            return original(q, k, cos, sin, unsqueeze_dim)
        # Under fp16 autocast the attention casts q and k to fp16 anyway: write fp16 directly (the
        # math stays fp32 with one rounding, the same value autocast would produce), so the cast
        # kernels disappear and the backward pass uses the same dtypes as the forward.
        out = None
        if (q.dtype == torch.float16 and k.dtype == torch.float16
                and torch.is_autocast_enabled("cuda")
                and torch.get_autocast_dtype("cuda") == torch.float16):
            out = torch.float16
        return rope(q, cos, sin, out), rope(k, cos, sin, out)

    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb

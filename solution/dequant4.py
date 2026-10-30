"""NF4 dequantization in one Triton kernel (double quantization included).

bitsandbytes unpacks a 4-bit weight in two launches: first it decodes the 8-bit "absmax" scale
factors (nested quantization), then it decodes the weights with them. The profile shows both,
about 390 launches each per step, because every one of the 7 linear layers in all 28 blocks is
unpacked in the forward pass and again in the backward pass.

One program here handles a tile of packed bytes. For each byte it decodes the scale
(code8[q] * absmax2[block // 256] + offset), looks up both 4-bit values in the NF4 table and
writes both outputs. The arithmetic is the same as bitsandbytes' (fp32 math, one rounding to the
output dtype), so the result matches it.

The same function also replaces the forward matmul of Linear4bit for training batches: bitsandbytes
dequantizes the weight and runs a normal fp16 matmul there too, so we do exactly that with our
unpack. Single-row inference calls keep the original path.

Written from scratch from the bitsandbytes dequantize_4bit / MatMul4Bit sources shown in the log.
"""

import torch
import triton
import triton.language as tl

import bitsandbytes.functional as F

_ORIGINAL_DEQUANTIZE_4BIT = F.dequantize_4bit  # kept for the tests and for the fallback path
BLOCK_BYTES = 1024


@triton.jit
def nf4_dequant_kernel(a_ptr, absmax_ptr, code_ptr, out_ptr, absmax2_ptr, code2_ptr, offset_ptr,
                       n_bytes, BLOCKSIZE: tl.constexpr, NESTED: tl.constexpr,
                       BLOCK: tl.constexpr):
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_bytes
    byte = tl.load(a_ptr + idx, mask=mask, other=0).to(tl.int32)
    hi = byte >> 4          # the first value of the pair sits in the high nibble
    lo = byte & 15
    block = (idx * 2) // BLOCKSIZE
    if NESTED:
        q = tl.load(absmax_ptr + block, mask=mask, other=0).to(tl.int32)
        scale = tl.load(code2_ptr + q)
        outer = tl.load(absmax2_ptr + block // 256, mask=mask, other=0.0)
        absmax = scale * outer + tl.load(offset_ptr)
    else:
        absmax = tl.load(absmax_ptr + block, mask=mask, other=0.0)
    out_ty = out_ptr.dtype.element_ty
    tl.store(out_ptr + idx * 2, (tl.load(code_ptr + hi) * absmax).to(out_ty), mask=mask)
    tl.store(out_ptr + idx * 2 + 1, (tl.load(code_ptr + lo) * absmax).to(out_ty), mask=mask)


def _supported(A, quant_state):
    return (
        A.is_cuda
        and quant_state.quant_type == "nf4"
        and quant_state.blocksize in (32, 64, 128, 256)
        and quant_state.dtype in (torch.float16, torch.float32)
        and (not quant_state.nested or quant_state.state2.blocksize == 256)
        and quant_state.code.numel() == 16
        and A.numel() * 2 == quant_state.shape[0] * quant_state.shape[1]
    )


def dequantize_nf4(A, quant_state):
    """Unpacked weight of shape quant_state.shape, dtype quant_state.dtype."""
    A = A.reshape(-1).contiguous()
    out = torch.empty(quant_state.shape, device=A.device, dtype=quant_state.dtype)
    code = quant_state.code.to(device=A.device, dtype=torch.float32).contiguous()
    nested = bool(quant_state.nested)
    if nested:
        absmax = quant_state.absmax.contiguous()
        absmax2 = quant_state.state2.absmax.float().contiguous()
        code2 = quant_state.state2.code.to(device=A.device, dtype=torch.float32).contiguous()
        offset = torch.as_tensor(quant_state.offset, device=A.device, dtype=torch.float32).reshape(1)
    else:
        absmax = quant_state.absmax.float().contiguous()
        absmax2 = code2 = offset = absmax  # unused placeholders
    n_bytes = A.numel()
    nf4_dequant_kernel[(triton.cdiv(n_bytes, BLOCK_BYTES),)](
        A, absmax, code, out, absmax2, code2, offset, n_bytes,
        BLOCKSIZE=quant_state.blocksize, NESTED=nested, BLOCK=BLOCK_BYTES, num_warps=4,
    )
    return out


def patch_bnb_dequant():
    from bitsandbytes.autograd import _functions as bnb_functions

    if getattr(F.dequantize_4bit, "_ours", False):
        return
    original = _ORIGINAL_DEQUANTIZE_4BIT

    def dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=None,
                        quant_type="fp4"):
        if quant_state is not None and out is None and _supported(A, quant_state):
            return dequantize_nf4(A, quant_state)
        return original(A, quant_state, absmax, out, blocksize, quant_type)

    dequantize_4bit._ours = True
    F.dequantize_4bit = dequantize_4bit

    original_forward = bnb_functions.MatMul4Bit.forward

    def forward(ctx, A, B, out=None, bias=None, quant_state=None):
        # training batches only; one-row inference calls keep bitsandbytes' own gemv path
        if (out is None and quant_state is not None and A.is_cuda and A.numel() > A.shape[-1]
                and _supported(B, quant_state)):
            ctx.is_empty = False
            B = B.view(-1, 1)
            weight = dequantize_nf4(B, quant_state).to(A.dtype)
            output = torch.nn.functional.linear(A, weight, bias)
            ctx.state = quant_state
            ctx.dtype_A, ctx.dtype_B = A.dtype, B.dtype
            ctx.dtype_bias = None if bias is None else bias.dtype
            ctx.tensors = (None, B) if any(ctx.needs_input_grad[:2]) else (None, None)
            return output
        return original_forward(ctx, A, B, out, bias, quant_state)

    bnb_functions.MatMul4Bit.forward = staticmethod(forward)

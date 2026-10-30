"""SwiGLU that saves fp16 instead of fp32, for the Qwen2 MLP under 4-bit + fp16 autocast.

What the dtype log of the real run showed: the gate and up projections (bitsandbytes Linear4bit)
compute in fp16 under autocast and are then widened back to fp32 on output. So the fp32 tensors we
receive hold values that are exactly representable in fp16, and the original SwiGLU saved both of
them as fp32 for backward: 2 x 2048 x 8960 x 4 bytes = 147 MB per layer, about 4 GB over 28 layers.

Here the forward kernel reads them once and writes three fp16 tensors: the activation output and
the two values to save. Saving fp16 loses nothing for these inputs. The output goes to the down
projection as fp16, which is what autocast would have cast it to anyway, so the matmul sees the
same bits; the down projection then returns fp16, which the fp32 residual add promotes as it does
for the attention output today.

Backward reads the fp16 copies and writes the gradients in the dtype of the original inputs, so
nothing downstream changes.

This is a rewrite of the example's idea and does not count toward the 5% kernel-time rule.
It is used only where the assumption above holds (bitsandbytes 4-bit layers, fp32 outputs,
autocast on); anywhere else the example kernel runs unchanged.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def swiglu_lp_fwd_kernel(gate_ptr, up_ptr, out_ptr, gate16_ptr, up16_ptr, n_elements,
                         BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + offsets, out.to(tl.float16), mask=mask)
    tl.store(gate16_ptr + offsets, gate.to(tl.float16), mask=mask)
    tl.store(up16_ptr + offsets, up.to(tl.float16), mask=mask)


@triton.jit
def swiglu_lp_bwd_kernel(grad_ptr, gate16_ptr, up16_ptr, dgate_ptr, dup_ptr, n_elements,
                         BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up16_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(gate)
    dup = grad * gate * sig
    dgate = grad * up * sig * (1.0 + gate * (1.0 - sig))
    tl.store(dgate_ptr + offsets, dgate.to(dgate_ptr.dtype.element_ty), mask=mask)
    tl.store(dup_ptr + offsets, dup.to(dup_ptr.dtype.element_ty), mask=mask)


class SwiGLULowPrecision(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.contiguous()
        out = torch.empty(gate.shape, device=gate.device, dtype=torch.float16)
        gate16 = torch.empty_like(out)
        up16 = torch.empty_like(out)
        n = gate.numel()
        swiglu_lp_fwd_kernel[(triton.cdiv(n, BLOCK),)](gate, up, out, gate16, up16, n, BLOCK=BLOCK)
        ctx.save_for_backward(gate16, up16)
        ctx.gate_dtype, ctx.up_dtype = gate.dtype, up.dtype
        return out

    @staticmethod
    def backward(ctx, grad):
        gate16, up16 = ctx.saved_tensors
        grad = grad.contiguous()
        dgate = torch.empty(gate16.shape, device=grad.device, dtype=ctx.gate_dtype)
        dup = torch.empty(gate16.shape, device=grad.device, dtype=ctx.up_dtype)
        n = gate16.numel()
        swiglu_lp_bwd_kernel[(triton.cdiv(n, BLOCK),)](grad, gate16, up16, dgate, dup, n,
                                                       BLOCK=BLOCK)
        return dgate, dup


def swiglu_lowprec(gate, up):
    return SwiGLULowPrecision.apply(gate, up)


def patch_qwen2_mlp_lowprec():
    """Apply after patch_qwen2_mlp(): takes over the MLP only for bitsandbytes 4-bit layers under
    fp16 autocast with an fp32 stream, and hands everything else to the previous forward."""
    from transformers.models.qwen2 import modeling_qwen2

    from .swiglu import swiglu

    previous = modeling_qwen2.Qwen2MLP.forward

    def forward(self, x):
        gate_proj = self.gate_proj
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        eligible = (
            act == "silu" and x.is_cuda and x.dtype == torch.float32
            and torch.is_autocast_enabled()
            and type(gate_proj).__module__.startswith("bitsandbytes")
        )
        if not eligible:
            return previous(self, x)
        gate = gate_proj(x)
        up = self.up_proj(x)
        if gate.dtype == torch.float32 and up.dtype == torch.float32:
            return self.down_proj(swiglu_lowprec(gate, up))
        return self.down_proj(swiglu(gate, up))

    modeling_qwen2.Qwen2MLP.forward = forward

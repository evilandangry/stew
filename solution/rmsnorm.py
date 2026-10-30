"""Fused RMSNorm for Qwen2, forward and backward in Triton.

transformers' Qwen2RMSNorm runs about six separate kernels per call (cast to fp32, square, mean,
rsqrt, multiply, cast back, multiply by the weight), and its backward is longer. Here one program
handles one row and does it all in registers: one read of x, one write of y.

The numerics follow the original on purpose, including the rounding of the normalised value to
the input dtype before the weight multiply, and the output dtype (the promotion of weight and
input dtypes), so the layers after the norm see exactly what they saw before.

Derivation of the backward (h = x * r, r = (mean(x^2) + eps)^-1/2, g = dL/dh):
    dL/dx_j = r * (g_j - h_j * mean_i(g_i * h_i))
Written from scratch from the Qwen2 source and this derivation; no code taken from other kernels.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    # the original rounds the normalised value to the input dtype before multiplying by weight
    h = (x * rstd).to(x_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = h * w
    tl.store(y_ptr + row * N + cols, y.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def rmsnorm_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    dy = tl.load(dy_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    h = x * rstd
    # autograd casts the gradient of the rounded value to the input dtype; do the same
    g = (dy * w).to(x_ptr.dtype.element_ty).to(tl.float32)
    c = tl.sum(g * h, axis=0) / N
    dx = rstd * (g - h * c)
    tl.store(dx_ptr + row * N + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def _launch_config(n_cols):
    block = triton.next_power_of_2(n_cols)
    num_warps = 8 if block >= 4096 else 4
    return block, num_warps


class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        rows, n = x2.shape
        weight = weight.contiguous()
        out_dtype = torch.result_type(weight, x2)
        y = torch.empty((rows, n), device=x.device, dtype=out_dtype)
        rstd = torch.empty(rows, device=x.device, dtype=torch.float32)
        block, num_warps = _launch_config(n)
        rmsnorm_fwd_kernel[(rows,)](x2, weight, y, rstd, n, eps, BLOCK=block, num_warps=num_warps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.in_shape = shape
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        rows, n = x2.shape
        dy2 = dy.reshape(-1, n).contiguous()
        dx = torch.empty_like(x2)
        block, num_warps = _launch_config(n)
        rmsnorm_bwd_kernel[(rows,)](dy2, x2, weight, rstd, dx, n, BLOCK=block, num_warps=num_warps)
        dw = None
        if ctx.needs_input_grad[1]:  # frozen in this task (LoRA); kept so the layer stays correct
            h = (x2.float() * rstd[:, None]).to(x2.dtype).float()
            dw = (dy2.float() * h).sum(0).to(weight.dtype)
        return dx.reshape(ctx.in_shape), dw, None


def rmsnorm(x, weight, eps):
    return RMSNormFn.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    """Send Qwen2RMSNorm.forward through the fused kernel (a class patch works before or after
    Soup builds the model)."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2RMSNorm.forward

    def forward(self, hidden_states):
        if not hidden_states.is_cuda:
            return original(self, hidden_states)
        return rmsnorm(hidden_states, self.weight, self.variance_epsilon)

    modeling_qwen2.Qwen2RMSNorm.forward = forward

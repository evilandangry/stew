"""Fused cross-entropy for the language-model head, forward and backward in Triton.

The baseline loss path walks the (tokens x vocabulary) logits table many times:
  - transformers' ForCausalLMLoss upcasts it to fp32 (a full copy), runs log_softmax (another full
    table, kept for backward), and its backward builds more full-size temporaries;
  - TRL then recomputes a softmax over the same table in chunks just to log the entropy;
  - Soup's rewind recorder runs one more cross-entropy per row.

Here one program per token row streams through the vocabulary once, with an online softmax, and
produces everything at once: the loss, the log-sum-exp (kept for backward) and the entropy.
Backward streams the row once more and writes the gradient straight into the output tensor, so no
fp32 copy of the table and no saved log-probabilities ever exist.

Math, per row with logits x, target y, p = softmax(x), lse = log(sum(exp(x))):
    loss    = lse - x[y]                (0 for ignored rows)
    entropy = lse - sum(p * x)          (= -sum(p * log p))
    dx      = (p - onehot(y)) * dloss   (0 for ignored rows)
The online pass keeps a running max m, s = sum(exp(x - m)) and t = sum(exp(x - m) * x), so
sum(p * x) = t / s and lse = m + log(s).

The two metrics TRL and Soup need next to the loss (entropy, per-token loss) are attached to the
logits tensor, and small patches make TRL's entropy_from_logits and Soup's row_losses use them
instead of recomputing. Both fall back to the originals if the attribute is missing.

Written from scratch from the transformers / TRL / Soup sources and the math above.
"""

import torch
import triton
import triton.language as tl

BLOCK_V = 4096


@triton.jit
def ce_fwd_kernel(logits_ptr, labels_ptr, loss_ptr, lse_ptr, ent_ptr,
                  V, stride_row, ignore_index, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = logits_ptr + row.to(tl.int64) * stride_row
    target = tl.load(labels_ptr + row)
    lane = tl.arange(0, BLOCK)
    # each lane keeps its own online statistics; they are merged after the loop
    m_i = tl.full([BLOCK], -1.0e30, tl.float32)
    s_i = tl.zeros([BLOCK], tl.float32)
    t_i = tl.zeros([BLOCK], tl.float32)
    for start in range(0, V, BLOCK):
        offs = start + lane
        mask = offs < V
        x = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        # masked lanes must not win the max, and must stay finite so 0 * x is 0, never NaN
        m_new = tl.maximum(m_i, tl.where(mask, x, -1.0e30))
        alpha = tl.exp(m_i - m_new)
        e = tl.where(mask, tl.exp(x - m_new), 0.0)
        s_i = s_i * alpha + e
        t_i = t_i * alpha + e * x
        m_i = m_new
    m = tl.max(m_i, axis=0)
    w = tl.exp(m_i - m)
    s = tl.sum(s_i * w, axis=0)
    t = tl.sum(t_i * w, axis=0)
    lse = m + tl.log(s)
    x_target = tl.load(base + tl.maximum(target, 0)).to(tl.float32)
    loss = tl.where(target != ignore_index, lse - x_target, 0.0)
    tl.store(loss_ptr + row, loss)
    tl.store(lse_ptr + row, lse)
    tl.store(ent_ptr + row, lse - t / s)


@triton.jit
def ce_bwd_kernel(logits_ptr, labels_ptr, lse_ptr, dloss_ptr, grad_ptr,
                  V, stride_row, stride_grow, ignore_index, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    row64 = row.to(tl.int64)
    target = tl.load(labels_ptr + row)
    lse = tl.load(lse_ptr + row)
    dloss = tl.load(dloss_ptr + row)
    scale = tl.where(target != ignore_index, dloss, 0.0)
    lane = tl.arange(0, BLOCK)
    for start in range(0, V, BLOCK):
        offs = start + lane
        mask = offs < V
        x = tl.load(logits_ptr + row64 * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.exp(x - lse)
        g = (p - tl.where(offs == target, 1.0, 0.0)) * scale
        tl.store(grad_ptr + row64 * stride_grow + offs, g.to(grad_ptr.dtype.element_ty), mask=mask)


class FusedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, ignore_index):
        rows, vocab = logits.shape
        loss = torch.empty(rows, device=logits.device, dtype=torch.float32)
        lse = torch.empty_like(loss)
        entropy = torch.empty_like(loss)
        ce_fwd_kernel[(rows,)](logits, labels, loss, lse, entropy, vocab, logits.stride(0),
                               ignore_index, BLOCK=BLOCK_V, num_warps=8)
        ctx.save_for_backward(logits, labels, lse)
        ctx.ignore_index = ignore_index
        ctx.mark_non_differentiable(entropy)
        return loss, entropy

    @staticmethod
    def backward(ctx, dloss, dentropy):
        logits, labels, lse = ctx.saved_tensors
        rows, vocab = logits.shape
        grad = torch.empty_like(logits)
        dloss = dloss.contiguous().float()
        ce_bwd_kernel[(rows,)](logits, labels, lse, dloss, grad, vocab, logits.stride(0),
                               grad.stride(0), ctx.ignore_index, BLOCK=BLOCK_V, num_warps=8)
        return grad, None, None


def fused_for_causal_lm_loss(logits, labels, vocab_size, num_items_in_batch=None,
                             ignore_index=-100, shift_labels=None, **kwargs):
    """Drop-in for transformers' ForCausalLMLoss (same shifting and reduction)."""
    if shift_labels is None:
        labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
    token_shape = shift_labels.shape
    logits2d = logits.reshape(-1, vocab_size)
    if logits2d.stride(-1) != 1:
        logits2d = logits2d.contiguous()
    targets = shift_labels.reshape(-1).to(logits.device).contiguous()
    rows, entropy = FusedCrossEntropy.apply(logits2d, targets, ignore_index)
    try:  # metrics for TRL's entropy and Soup's rewind recorder, tied to this tensor's lifetime
        logits._fused_ce_token_loss = rows.detach().view(token_shape)
        logits._fused_ce_entropy = entropy.view(token_shape)
    except Exception:
        pass
    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(rows.device)
        return rows.sum() / num_items_in_batch
    return rows.sum() / (targets != ignore_index).sum()


def _patch_trl_entropy():
    import importlib

    for name in ("trl.trainer.sft_trainer", "trl.trainer.utils"):
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        original = getattr(module, "entropy_from_logits", None)
        if original is None or getattr(original, "_fused_ce", False):
            continue

        def make(original):
            def entropy_from_logits(logits, *args, **kwargs):
                entropy = getattr(logits, "_fused_ce_entropy", None)
                if entropy is not None and entropy.shape == logits.shape[:-1]:
                    return entropy  # fp32, a touch more precise than the original's head dtype
                return original(logits, *args, **kwargs)

            entropy_from_logits._fused_ce = True
            return entropy_from_logits

        setattr(module, "entropy_from_logits", make(original))


def _patch_soup_rewind():
    try:
        from soup_cli.trainer import rewind_hf
    except Exception:
        return
    original = rewind_hf.row_losses
    if getattr(original, "_fused_ce", False):
        return

    def row_losses(logits, labels):
        token_loss = getattr(logits, "_fused_ce_token_loss", None)
        if token_loss is not None and token_loss.shape == labels.shape:
            return rewind_hf.row_losses_from_token_losses(token_loss, labels)
        return original(logits, labels)

    row_losses._fused_ce = True
    rewind_hf.row_losses = row_losses


def patch_causal_lm_loss():
    """Route the causal-LM loss through the fused kernel, before or after Soup builds the model
    (the model looks the loss function up through LOSS_MAPPING on every call)."""
    from transformers.loss import loss_utils

    original = loss_utils.ForCausalLMLoss
    if getattr(original, "_fused_ce", False):
        return

    def for_causal_lm_loss(logits, labels, vocab_size, num_items_in_batch=None,
                           ignore_index=-100, shift_labels=None, **kwargs):
        if not logits.is_cuda or logits.shape[-1] != vocab_size:
            return original(logits, labels, vocab_size, num_items_in_batch=num_items_in_batch,
                            ignore_index=ignore_index, shift_labels=shift_labels, **kwargs)
        return fused_for_causal_lm_loss(logits, labels, vocab_size, num_items_in_batch,
                                        ignore_index, shift_labels, **kwargs)

    for_causal_lm_loss._fused_ce = True
    loss_utils.ForCausalLMLoss = for_causal_lm_loss
    mapping = getattr(loss_utils, "LOSS_MAPPING", None)
    if mapping is not None:
        for key, value in list(mapping.items()):
            if value is original:
                mapping[key] = for_causal_lm_loss
    _patch_trl_entropy()
    _patch_soup_rewind()

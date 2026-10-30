"""Skip the padding inside a batch: every per-token operation runs on the real tokens only.

A batch is 4 rows padded to the longest one. On this data only ~57% of the token slots are real
(mean row 207 tokens, max 512), yet the model normalises, projects, runs the MLP and LoRA, and
saves activations for every slot, padding included. The rules allow removing padding inside a
batch in our own code (no rows are mixed across batches).

How (the math for real tokens is unchanged; only where the data sits changes):
- Qwen2Model.forward: embed only the real tokens, packed as [1, N, H], and run every layer on that.
  The attention mask is built from the padded shape with transformers' own create_causal_mask,
  exactly as before. At the end the hidden states are scattered back to [B, T, H] (zeros in the
  padding), so the loss code downstream sees the usual layout.
- Attention is the one step that needs the batch layout: around transformers' SDPA function,
  q, k and v are scattered back to [B, heads, T, D], the same masked attention runs, and the real
  rows are gathered out again.
- LoRA dropout: the random mask is drawn on a padded tensor of the same shape and dtype the base
  run draws it on, so the random numbers are the same as the base, then packed. A small Triton
  kernel applies it (forward and backward).
Anything unusual (no padding, eval, caches, sliding windows, extra outputs requested) runs the
original code.
"""
import threading

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_STATE = threading.local()
_SAID = set()
_ORIG_MODEL_FORWARD = None


def _say(msg):
    if msg not in _SAID:
        _SAID.add(msg)
        print(f"[solution] unpad: {msg}")


def _ctx():
    return getattr(_STATE, "ctx", None)


# ------------------------------------------------------------------ dropout kernel
@triton.jit
def _mask_scale_kernel(X, M, Y, n, scale, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
    k = tl.load(M + offs, mask=m, other=0).to(tl.float32)
    tl.store(Y + offs, (x * k * scale).to(Y.dtype.element_ty), mask=m)


def _mask_scale(x, keep_u8, scale):
    y = torch.empty_like(x)
    n = x.numel()
    _mask_scale_kernel[(triton.cdiv(n, 4096),)](x, keep_u8, y, n, scale, BLOCK=4096, num_warps=8)
    return y


class _PackedDropout(torch.autograd.Function):
    """y = x * keep / (1 - p), computed in fp32 and rounded once, like PyTorch's dropout kernel."""

    @staticmethod
    def forward(ctx, x, keep_u8, scale):
        ctx.save_for_backward(keep_u8)
        ctx.scale = scale
        return _mask_scale(x, keep_u8, scale)

    @staticmethod
    def backward(ctx, grad):
        (keep_u8,) = ctx.saved_tensors
        return _mask_scale(grad.contiguous(), keep_u8, ctx.scale), None, None


def _wrap_dropout(mod):
    if getattr(mod, "_soup_unpad", False):
        return False
    orig = mod.forward

    def forward(x):
        c = _ctx()
        p = mod.p
        if (c is None or not mod.training or p <= 0.0 or p >= 1.0 or mod.inplace
                or not x.is_cuda or x.dim() != 3 or x.shape[0] != 1 or x.shape[1] != c["n"]):
            return orig(x)
        B, T, H = c["B"], c["T"], x.shape[-1]
        # the same draw the base run makes: dropout on a [B, T, H] tensor of this dtype
        ones = torch.ones((B, T, H), dtype=x.dtype, device=x.device)
        keep = F.dropout(ones, p, True) != 0
        keep = keep.view(B * T, H).index_select(0, c["idx"]).view(x.shape)
        return _PackedDropout.apply(x.contiguous(), keep.view(torch.uint8), 1.0 / (1.0 - p))

    mod.forward = forward
    mod._soup_unpad = True
    return True


def wrap_lora_dropouts(model):
    """Wrap every LoRA dropout module of the (PEFT) model. Returns how many were wrapped."""
    if not isinstance(model, torch.nn.Module):
        return 0
    count = 0
    for m in model.modules():
        d = getattr(m, "lora_dropout", None)
        if isinstance(d, torch.nn.ModuleDict):
            for mod in d.values():
                if isinstance(mod, torch.nn.Dropout) and mod.p > 0 and _wrap_dropout(mod):
                    count += 1
    return count


# ------------------------------------------------------------------ attention
def _to_padded(t, c):
    """[1, heads, N, D] -> [B, heads, T, D] with zeros in the padding (same strides as the base)."""
    heads, D = t.shape[1], t.shape[3]
    rows = t[0].transpose(0, 1)  # [N, heads, D]
    full = t.new_zeros((c["B"] * c["T"], heads, D)).index_copy(0, c["idx"], rows)
    return full.view(c["B"], c["T"], heads, D).transpose(1, 2)


def _make_sdpa(orig_sdpa):
    def sdpa(module, query, key, value, attention_mask, *args, **kwargs):
        c = _ctx()
        if c is None or query.dim() != 4 or query.shape[0] != 1 or query.shape[2] != c["n"]:
            return orig_sdpa(module, query, key, value, attention_mask, *args, **kwargs)
        out, weights = orig_sdpa(module, _to_padded(query, c), _to_padded(key, c),
                                 _to_padded(value, c), attention_mask, *args, **kwargs)
        heads, D = out.shape[2], out.shape[3]  # out: [B, T, heads, D]
        out = out.reshape(c["B"] * c["T"], heads, D).index_select(0, c["idx"]).unsqueeze(0)
        return out, weights

    sdpa._soup_unpad = True
    return sdpa


# ------------------------------------------------------------------ model forward
def _why_not(model, input_ids, attention_mask, past_key_values, inputs_embeds, use_cache, kwargs):
    if not model.training or not torch.is_grad_enabled():
        return "not training"
    if input_ids is None or inputs_embeds is not None:
        return "no input_ids"
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.dim() != 2:
        return "no 2-D attention mask"
    if not input_ids.is_cuda or input_ids.dim() != 2:
        return "not a CUDA batch"
    if past_key_values is not None or use_cache:
        return "cache in use"
    if getattr(model, "has_sliding_layers", False):
        return "sliding-window layers"
    if getattr(model.config, "_attn_implementation", None) != "sdpa":
        return "attention is not sdpa"
    for k in ("output_attentions", "output_hidden_states"):
        if kwargs.get(k):
            return f"{k} requested"
    if kwargs.get("return_dict", True) is False:
        return "return_dict=False"
    return None


def patch_unpad():
    global _ORIG_MODEL_FORWARD
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen2 import modeling_qwen2 as Q

    cls = Q.Qwen2Model
    if getattr(cls.forward, "_soup_unpad", False):
        return
    orig = cls.forward
    _ORIG_MODEL_FORWARD = orig
    if not getattr(ALL_ATTENTION_FUNCTIONS["sdpa"], "_soup_unpad", False):
        ALL_ATTENTION_FUNCTIONS["sdpa"] = _make_sdpa(ALL_ATTENTION_FUNCTIONS["sdpa"])

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, **kwargs):
        def original():
            return orig(self, input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)

        why = _why_not(self, input_ids, attention_mask, past_key_values, inputs_embeds, use_cache,
                       kwargs)
        if why:
            if self.training:
                _say(f"original path ({why})")
            return original()
        B, T = input_ids.shape
        idx = attention_mask.reshape(-1).bool().nonzero().squeeze(1)
        n = idx.numel()
        if n == B * T or n == 0:
            return original()  # nothing to skip
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        pos = position_ids.expand(B, T).reshape(-1).index_select(0, idx).unsqueeze(0)

        embeds = self.embed_tokens(input_ids.reshape(-1).index_select(0, idx).unsqueeze(0))
        like = embeds.new_zeros(1).expand(B, T, 1)  # shape and dtype of the padded embeddings
        mask = Q.create_causal_mask(config=self.config, inputs_embeds=like,
                                    attention_mask=attention_mask, past_key_values=None,
                                    position_ids=position_ids)
        if mask is None:
            return original()
        rope = self.rotary_emb(embeds, pos)
        _say(f"active (first batch: {n} real tokens of {B * T} slots)")
        kwargs.pop("return_dict", None)
        _STATE.ctx = {"B": B, "T": T, "idx": idx, "n": n}
        try:
            h = embeds
            for layer in self.layers[: self.config.num_hidden_layers]:
                h = layer(h, attention_mask=mask, position_embeddings=rope, position_ids=pos,
                          past_key_values=None, use_cache=False, **kwargs)
            h = self.norm(h)
        finally:
            _STATE.ctx = None
        full = h.new_zeros((B * T, h.shape[-1])).index_copy(0, idx, h[0]).view(B, T, -1)
        return BaseModelOutputWithPast(last_hidden_state=full, past_key_values=None)

    forward._soup_unpad = True
    cls.forward = forward

    # LoRA dropout modules exist only once Soup has built the PEFT model
    from soup_cli.trainer.sft import SFTTrainerWrapper

    setup = SFTTrainerWrapper._setup_transformers
    if not getattr(setup, "_soup_unpad", False):
        def _setup_transformers(self, *args, **kw):
            out = setup(self, *args, **kw)
            try:
                count = wrap_lora_dropouts(getattr(self, "model", None))
                print(f"[solution] unpad: {count} LoRA dropout modules draw their masks on the padded shape")
            except Exception as e:  # noqa: BLE001  never break training because of the speed-up
                print(f"[solution] unpad: LoRA dropout not wrapped: {e}")
            return out

        _setup_transformers._soup_unpad = True
        SFTTrainerWrapper._setup_transformers = _setup_transformers

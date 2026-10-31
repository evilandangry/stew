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
- Attention: when every row is right-padded (TRL's collator pads on the right), each row's real
  tokens sit together in the packed tensor, so transformers' SDPA function runs once per row on
  just that row: plain causal attention, no mask, no padding (less work, since attention grows
  with length squared). Otherwise q, k and v are scattered back to [B, heads, T, D], the same
  masked attention runs, and the real rows are gathered out again.
- LoRA dropout: the random mask is drawn on a padded tensor of the same shape and dtype the base
  run draws it on, so the random numbers are the same as the base, then packed, and applied to
  the packed tokens (fp32 math, one rounding, exactly like PyTorch's dropout).
Anything unusual (no padding, eval, caches, sliding windows, extra outputs requested) runs the
original code.
"""
import threading

import torch
import torch.nn.functional as F

_STATE = threading.local()
_SAID = set()
_ORIG_MODEL_FORWARD = None


def _say(msg):
    if msg not in _SAID:
        _SAID.add(msg)
        print(f"[solution] unpad: {msg}")


def _ctx():
    return getattr(_STATE, "ctx", None)


# ------------------------------------------------------------------ dropout
class _PackedDropout(torch.autograd.Function):
    """y = x * keep / (1 - p), computed in fp32 and rounded once, like PyTorch's dropout kernel.
    Saves only the boolean mask for the backward pass, as PyTorch's dropout does."""

    @staticmethod
    def forward(ctx, x, keep, scale):
        ctx.save_for_backward(keep)
        ctx.scale = scale
        return (x.float() * keep * scale).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (keep,) = ctx.saved_tensors
        return (grad.float() * keep * ctx.scale).to(grad.dtype), None, None


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
        return _PackedDropout.apply(x, keep, 1.0 / (1.0 - p))

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
        lens = c.get("lens")
        if lens:  # right-padded batch: one causal attention per row, on its real tokens only
            outs, start = [], 0
            for length in lens:
                if length == 0:
                    continue
                end = start + length
                out, _ = orig_sdpa(module, query[:, :, start:end], key[:, :, start:end],
                                   value[:, :, start:end], None, *args, **kwargs)
                outs.append(out)  # [1, length, heads, D]
                start = end
            return (outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)), None
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
        am = attention_mask.bool()
        lengths = am.sum(1)
        right = (am == (torch.arange(T, device=am.device)[None, :] < lengths[:, None])).all()
        info = torch.cat([right.view(1).long(), lengths.long()]).tolist()  # the one sync
        right, lens = bool(info[0]), info[1:]
        n = sum(lens)
        if n == B * T or n == 0:
            return original()  # nothing to skip
        if right:  # real tokens of row b are positions 0 .. lens[b]-1
            idx = torch.cat([torch.arange(length, device=am.device) + b * T
                             for b, length in enumerate(lens) if length > 0])
        else:
            idx = am.reshape(-1).nonzero().squeeze(1)
            lens = None
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        pos = position_ids.expand(B, T).reshape(-1).index_select(0, idx).unsqueeze(0)

        embeds = self.embed_tokens(input_ids.reshape(-1).index_select(0, idx).unsqueeze(0))
        mask = None  # per-row attention needs no mask
        if lens is None:
            like = embeds.new_zeros(1).expand(B, T, 1)  # shape and dtype of the padded embeddings
            mask = Q.create_causal_mask(config=self.config, inputs_embeds=like,
                                        attention_mask=attention_mask, past_key_values=None,
                                        position_ids=position_ids)
            if mask is None:
                return original()
        rope = self.rotary_emb(embeds, pos)
        if "active" not in _SAID:
            _SAID.add("active")
            print(f"[solution] unpad: active (first batch: {n} real tokens of {B * T} slots, "
                  f"{'per-row attention' if lens else 'padded attention'})")
        kwargs.pop("return_dict", None)
        _STATE.ctx = {"B": B, "T": T, "idx": idx, "n": n, "lens": lens}
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

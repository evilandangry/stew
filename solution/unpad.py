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
- MLP memory cap: the peak is set by the densest batch, and the MLP's saved gate/up tensors are
  most of a layer's saved activations. For a batch with more real tokens than MLP_TOKEN_BUDGET,
  just enough layers recompute gate/up in the backward pass (checkpointing x only) to keep that
  batch's saved activations at the level of a MLP_TOKEN_BUDGET-token batch. Smaller batches
  recompute nothing. The recompute uses the same kernels, so gradients are unchanged; its time
  counts in the measured run.
Anything unusual (eval, caches, sliding windows, extra outputs requested) runs the original code.
"""
import math
import threading

import torch
import torch.nn.functional as F

_STATE = threading.local()
_SAID = set()
_ORIG_MODEL_FORWARD = None
_MAX_N = [0]
_CALLS = [0]


def _mark(label):
    try:
        from .timeline import mark
        mark(label)
    except Exception:  # noqa: BLE001
        pass

MLP_TOKEN_BUDGET = 1000  # tokens' worth of full MLP activations kept per layer at the peak
_MLP_SAVED_SHARE = 0.63  # share of a layer's saved activations that recomputing gate/up removes


def _mlp_recompute_layers(n, layers):
    """How many of `layers` layers recompute gate/up for a batch of n real tokens."""
    if n <= MLP_TOKEN_BUDGET:
        return 0
    return min(layers, math.ceil(layers * (n - MLP_TOKEN_BUDGET) / (n * _MLP_SAVED_SHARE)))


def _gate_up(mlp, x):
    from .swiglu import swiglu

    return swiglu(mlp.gate_proj(x), mlp.up_proj(x))


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


def _sdpa_fast(q, k, v, causal, scale, dropout):
    """PyTorch SDPA restricted to its fused kernels (the 'math' fallback stores full attention
    matrices). If they cannot run, fall back once and say so in the log."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    try:
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, scale=scale,
                                                  is_causal=causal)
    except RuntimeError:
        _say("per-row attention: fused kernel unavailable, using PyTorch's default")
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, scale=scale,
                                              is_causal=causal)


_BROADCAST_KV = [True]  # becomes False if the fused kernel rejects the zero-copy k/v layout


def _sdpa_strict(q, k, v, causal, scale, dropout):
    """Fused SDPA kernels only; raises if none can run (no silent math fallback)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, scale=scale,
                                              is_causal=causal)


def _row_attention(q, k, v, groups, causal, scale, dropout):
    """One row: q [1, Hq, L, D], k/v [1, Hkv, L, D] -> [1, Hq, L, D].

    Grouped-query attention without copying k/v: the query heads are arranged as
    [Hkv, groups, L, D] (head h = g * groups + r uses kv head g, the order repeat_kv uses) and each
    kv head is broadcast over its group with a zero-copy view. SDPA then keeps only the 2-head k/v
    for the backward pass instead of a 12-head copy. Inputs are cast to the autocast dtype first
    (the cast autocast would do anyway), so autocast does not make a dense copy of the view.
    If the fused kernel rejects this layout, k/v are copied to all heads instead."""
    from transformers.integrations.sdpa_attention import repeat_kv

    if groups > 1 and _BROADCAST_KV[0]:
        if torch.is_autocast_enabled("cuda"):
            dt = torch.get_autocast_dtype("cuda")
            q, k, v = q.to(dt), k.to(dt), v.to(dt)
        hkv, L, D = k.shape[1], q.shape[2], q.shape[3]
        qg = q.reshape(hkv, groups, L, D)
        kg = k.reshape(hkv, 1, L, D).expand(hkv, groups, L, D)
        vg = v.reshape(hkv, 1, L, D).expand(hkv, groups, L, D)
        try:
            return _sdpa_strict(qg, kg, vg, causal, scale, dropout).reshape(1, hkv * groups, L, D)
        except RuntimeError:
            _BROADCAST_KV[0] = False
            _say("per-row attention: zero-copy k/v not supported by the fused kernel, copying")
    if groups > 1:
        k, v = repeat_kv(k, groups), repeat_kv(v, groups)
    return _sdpa_fast(q, k, v, causal, scale, dropout)


def _per_row_attention(module, query, key, value, lens, kwargs):
    """Causal attention row by row on the packed tokens; returns [1, N, heads, D] like
    transformers' sdpa_attention_forward. (Asking SDPA for its own GQA mode would push it onto
    the slow math kernel on a T4, so grouped heads are handled in _row_attention.)"""
    groups = getattr(module, "num_key_value_groups", 1)
    scale = kwargs.get("scaling")
    dropout = kwargs.get("dropout", 0.0) or 0.0
    outs, start = [], 0
    for length in lens:
        if length == 0:
            continue
        end = start + length
        out = _row_attention(query[:, :, start:end], key[:, :, start:end],
                             value[:, :, start:end], groups, length > 1, scale, dropout)
        outs.append(out.transpose(1, 2))  # [1, length, heads, D]
        start = end
    return torch.cat(outs, dim=1).contiguous()


def _make_sdpa(orig_sdpa):
    def sdpa(module, query, key, value, attention_mask, *args, **kwargs):
        c = _ctx()
        if c is None or query.dim() != 4 or query.shape[0] != 1 or query.shape[2] != c["n"]:
            return orig_sdpa(module, query, key, value, attention_mask, *args, **kwargs)
        lens = c.get("lens")
        if lens:  # right-padded batch: one causal attention per row, on its real tokens only
            return _per_row_attention(module, query, key, value, lens, kwargs), None
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
        if n == 0:
            return original()
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
        _CALLS[0] += 1
        if _CALLS[0] in (1, 3, 64):
            _mark(f"training forward {_CALLS[0]} starts")
        layers = self.config.num_hidden_layers
        recompute = _mlp_recompute_layers(n, layers)
        if n > _MAX_N[0]:
            _MAX_N[0] = n
            print(f"[solution] unpad: largest batch so far: {n} real tokens, MLP gate/up "
                  f"recomputed in {recompute} of {layers} layers")
        _STATE.ctx = {"B": B, "T": T, "idx": idx, "n": n, "lens": lens,
                      "recompute": recompute, "mlp_i": 0}
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

    # MLP memory cap (wraps whatever MLP forward is installed now: the SwiGLU example)
    from torch.utils.checkpoint import checkpoint

    mlp_cls = Q.Qwen2MLP
    if not getattr(mlp_cls.forward, "_soup_unpad", False):
        prev_mlp = mlp_cls.forward

        def mlp_forward(self, x):
            c = _ctx()
            if c is None or not c.get("recompute"):
                return prev_mlp(self, x)
            i = c["mlp_i"]
            c["mlp_i"] = i + 1
            act = getattr(getattr(self, "config", None), "hidden_act", "silu")
            if (i >= c["recompute"] or act != "silu" or not x.is_cuda
                    or not torch.is_grad_enabled() or not x.requires_grad):
                return prev_mlp(self, x)
            return self.down_proj(checkpoint(_gate_up, self, x, use_reentrant=False))

        mlp_forward._soup_unpad = True
        mlp_cls.forward = mlp_forward

    # LoRA dropout modules exist only once Soup has built the PEFT model
    from soup_cli.trainer.sft import SFTTrainerWrapper

    setup = SFTTrainerWrapper._setup_transformers
    if not getattr(setup, "_soup_unpad", False):
        def _setup_transformers(self, *args, **kw):
            _mark("Soup starts building the model")
            out = setup(self, *args, **kw)
            _mark("model built (4-bit weights, LoRA)")
            try:
                count = wrap_lora_dropouts(getattr(self, "model", None))
                print(f"[solution] unpad: {count} LoRA dropout modules draw their masks on the padded shape")
            except Exception as e:  # noqa: BLE001  never break training because of the speed-up
                print(f"[solution] unpad: LoRA dropout not wrapped: {e}")
            return out

        _setup_transformers._soup_unpad = True
        SFTTrainerWrapper._setup_transformers = _setup_transformers

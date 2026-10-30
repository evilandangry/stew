"""Keep the tied embedding / LM-head matrix once, in fp16, instead of fp32 + a fresh fp16 copy.

Why: Qwen2.5-1.5B ties `embed_tokens` and `lm_head` (151,936 x 1,536). `prepare_model_for_kbit_training`
keeps it in fp32 (934 MiB). The fused LM-head loss then casts it to fp16 on every step (another
467 MiB while the head runs), because the matmul runs in fp16 under autocast anyway. So the head
already only ever sees the fp16-rounded values.

What changes: the stored matrix becomes fp16 (467 MiB, no per-step copy). The embedding lookup
still hands an fp32 tensor to the decoder (the residual stream stays fp32), so the only numeric
difference is that the looked-up embedding rows are rounded to fp16. The checkpoint is bf16, whose
values fit fp16 exactly except for magnitudes below ~6e-5 (rounded to a step of 6e-8).

Guarded: only when the matrix is tied, fp32, frozen and on the GPU; otherwise nothing happens.
"""
import torch


def convert_tied_embedding(model):
    """Store the tied, frozen, fp32 embedding as fp16. Returns a status string, or None if skipped."""
    if not isinstance(model, torch.nn.Module):
        return None
    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if emb is None or head is None:
        return None
    w = getattr(emb, "weight", None)
    if not isinstance(w, torch.Tensor):
        return None
    if getattr(head, "weight", None) is not w:  # not tied
        return None
    if not (w.is_cuda and w.dtype == torch.float32 and not w.requires_grad):
        return None
    if getattr(emb, "_soup_fp16_embed", False):
        return None

    rows = w.shape[0]
    new = torch.empty(w.shape, dtype=torch.float16, device=w.device)
    max_err, n_diff = 0.0, 0
    step = 8192
    for s in range(0, rows, step):  # chunked: no big temporaries
        chunk = w.data[s:s + step]
        c16 = chunk.to(torch.float16)
        new[s:s + step] = c16
        diff = (c16.float() - chunk).abs()
        max_err = max(max_err, float(diff.max()))
        n_diff += int((diff > 0).sum())
    w.data = new  # the old fp32 storage is released here
    torch.cuda.empty_cache()

    orig_forward = emb.forward

    def forward(*args, **kwargs):
        return orig_forward(*args, **kwargs).float()  # the decoder still gets fp32

    emb.forward = forward
    emb._soup_fp16_embed = True
    return f"{n_diff} of {w.numel()} values changed by rounding, max abs error {max_err:.2e}"


def patch_embed_fp16():
    """Apply the conversion when Soup has built the final model (before the trainer exists)."""
    from soup_cli.trainer.sft import SFTTrainerWrapper

    orig = SFTTrainerWrapper._setup_transformers
    if getattr(orig, "_soup_embed16", False):
        return

    def _setup_transformers(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        try:
            status = convert_tied_embedding(getattr(self, "model", None))
            print(f"[solution] embedding fp16: {status if status else 'skipped (not applicable)'}")
        except Exception as e:  # noqa: BLE001  never break training because of the speed-up
            print(f"[solution] embedding fp16 skipped: {e}")
        return out

    _setup_transformers._soup_embed16 = True
    SFTTrainerWrapper._setup_transformers = _setup_transformers

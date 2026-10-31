# Submission: NaVi

## Links
- Repository, final commit: https://github.com/BAITC-Hacks/hack-4df528e9-navi/commit/TODO
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in our
  README): https://www.kaggle.com/code/yerser/TODO?scriptVersionId=TODO
- Demo video (up to 3 minutes): TODO
- Presentation: TODO

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Yerkhan Serik | captain: padding removal, per-row and zero-copy attention, MLP recompute cap, NF4 unpack on both passes, RoPE, fp16 embedding, kernel-share fix; profiling, verification, README | evilandangry | yerser | Telegram @metalballadlover |
| Shadiyar Musin | starting point: fused LM head + loss, fp16 norm path, first NF4 kernel; fused add-norm | — | — | |
| Madiyar Komilzhonov | presentation | — | — | |
| Mansur Utepbergen | team member | — | — | |

## Our kernels
Check 4 reports the total share of our kernels: 11.8 % of the training time on the final
code (limit 5 %). The per-kernel split below is from the GPU profile of the final code
(16 profiled steps, `run_task.py --profile`).
<!-- TODO: fill the per-kernel % from the final profile run -->

| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `_head_ce` | `solution/head_loss.py` | LM head logits + log-softmax + NLL + entropy + argmax and their backward, chunked; the 2,048 × 151,936 logits table is never stored | TODO % |
| `_nf4_unpack` | `solution/nf4.py` | bitsandbytes' NF4 unpack (scale decode + weights) for the forward and backward matmuls of every 4-bit layer | TODO % |
| `_rms_fwd`, `_rms_bwd` | `solution/rmsnorm.py` | Qwen2 RMSNorm (about six PyTorch ops per call), fp16 output for the projections | TODO % |
| `_add_rms_fwd`, `_add_rms_bwd` | `solution/rmsnorm.py` | residual add + post-attention RMSNorm, plus the gradient add in the backward pass | TODO % |
| `rope_kernel` | `solution/rope.py` | rotary embedding (`rotate_half`, cat, multiplies, add) for q and k, forward and backward | TODO % |

Not launched in a normal run (fallbacks if the fused head path cannot run): `_ce_fwd`, `_ce_bwd`,
`_entropy_fwd` in `solution/fused_ce.py`. `solution/swiglu.py` is the challenge's example, still
running; its kernels do not count as ours.

Ideas taken from other kernels, with links (we wrote the code ourselves): online softmax
(Milakov and Gimelshein, https://arxiv.org/abs/1805.02867); the fused linear cross-entropy idea of
Liger Kernel (https://github.com/linkedin/Liger-Kernel) and Apple's Cut Cross-Entropy
(https://arxiv.org/abs/2411.09009); unpadding as in FlashAttention's varlen path
(https://github.com/Dao-AILab/flash-attention); activation recomputation (Chen et al.,
https://arxiv.org/abs/1604.06174); the NF4 format (QLoRA, https://arxiv.org/abs/2305.14314).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
<!-- these are from the verification run of 68f44c2; replace with the saved notebook's run if it differs -->
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 76.68 | 56.53 | speedup 1.356x |
| Peak GPU memory, MiB | 14,345 | 3,433 | base/ours 4.179x |
| Eval loss (public rows) | 2.04968 | 2.04950 | -0.009 % vs base |

- [x] All 10 checks of `harness/check_rules.py` passed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (logits table, padding, MLP activations, NF4 unpacking), how the
kernels work, results against the base (speed, memory, quality), limits and next steps.

## Limitations
- Speed is limited by fixed overhead: about 20 s of loading per run that the base also pays, and
  about 5–6 s of Triton compiling that it does not. Run-to-run noise is about ±2 s.
- Per-row attention relies on right padding (TRL's default); other layouts use a masked fallback.
- The MLP recompute trades a little time for memory and only acts on batches above 1,000 real tokens.
- Next: recompute more saved activations (fp32 residual stream, LoRA inputs), cut compile time,
  a varlen attention kernel of our own.

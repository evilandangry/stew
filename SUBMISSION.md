# Submission: NaVi

## Links
- Repository, final commit of the code: https://github.com/BAITC-Hacks/hack-4df528e9-navi/commit/8e63d40ec6e1165aaac782d429aa758546ad2b2d
  (later commits only update README.md and SUBMISSION.md; `solution/` is unchanged)
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in our
  README): https://www.kaggle.com/code/yerser/notebookf01887514d
- Demo video (up to 3 minutes): TODO
- Presentation: TODO

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Yerkhan Serik | captain: padding removal, per-row and zero-copy attention, MLP recompute cap, NF4 unpack on both passes, RoPE, fp16 embedding, kernel-share fix; profiling, verification, README | evilandangry | yerser | Telegram @metalballadlover |
| Shadiyar Musin | starting point: fused LM head + loss, fp16 norm path, first NF4 kernel; fused add-norm | shadiarmusin-eng | — | |
| Madiyar Komilzhonov | presentation | forces57 | — | |
| Mansur Utepbergen | team member | utepbergenovmansur68 | — | |

## Our kernels
Check 4 reported 12.0 % of the training time for our kernels in the saved notebook run of the final
code `8e63d40` (12.4 % in an earlier run of the same code; limit 5 %). The split below is from that
earlier run's `runs/result.json` (`own_kernel_ms`, CUDA events over the timed
steps 3–64, 25.53 s): each kernel's time divided by that. These add up to 12.9 %; the check computes
its total slightly differently. The example's SwiGLU kernels (883 ms) are not counted.

| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `_head_ce` | `solution/head_loss.py` | LM head logits + log-softmax + NLL + entropy + argmax and their backward, chunked; the 2,048 × 151,936 logits table is never stored | 0.8 % (214 ms) |
| `_nf4_unpack` | `solution/nf4.py` | bitsandbytes' NF4 unpack (scale decode + weights) for the forward and backward matmuls of every 4-bit layer | 9.0 % (2,295 ms) |
| `_rms_fwd`, `_rms_bwd` | `solution/rmsnorm.py` | Qwen2 RMSNorm (about six PyTorch ops per call), fp16 output for the projections | 0.7 % (170 ms) |
| `_add_rms_fwd`, `_add_rms_bwd` | `solution/rmsnorm.py` | residual add + post-attention RMSNorm, plus the gradient add in the backward pass | 1.3 % (323 ms) |
| `rope_kernel` | `solution/rope.py` | rotary embedding (`rotate_half`, cat, multiplies, add) for q and k, forward and backward | 1.1 % (287 ms) |

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
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 86.8 | 56.4 | speedup 1.539x |
| Peak GPU memory, MiB | 14,345 | 3,399 | base/ours 4.220x |
| Eval loss (public rows) | 2.04964 | 2.04954 | -0.005 % vs base |

The base was slow in this run: over our 19 runs its time ranged 75.9–88.0 s (median 78.0 s), so on
a typical run expect about 1.4× (1.38× against the median).

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

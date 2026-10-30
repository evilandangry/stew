# Submission: NaVi

## Links
- Repository, final commit: https://github.com/<TODO>/<TODO>/commit/<TODO sha>
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link must
  be in your README):
  https://www.kaggle.com/code/<TODO>/<TODO>?scriptVersionId=<TODO>
- Demo video (up to 3 minutes): <TODO link>
- Presentation: <TODO link>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| TODO | captain, kernels | evilandangry | TODO | TODO |
| TODO | presentation | TODO | TODO | |

## Our kernels
Share of the training time is the total over all our kernels from check 4 (11.8% on the last run);
fill the per-kernel split from `runs/checks.json` of the final notebook run.

| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | Qwen2 RMSNorm, about six PyTorch ops per call, forward and backward | TODO % |
| `rope_kernel` | `solution/rope.py` | rotary embedding (rotate_half, neg, cat, multiplies, add), forward and backward | TODO % |
| `ce_fwd_kernel`, `ce_bwd_kernel` | `solution/loss.py` | fp32 copy of the logits, log_softmax, TRL's entropy pass and Soup's per-row loss pass: one streaming pass over the vocabulary | TODO % |
| `nf4_dequant_kernel` | `solution/dequant4.py` | bitsandbytes' two dequantize launches (nested scales, then weights), also used for the forward matmul of 4-bit layers in training | TODO % |
| `swiglu_lp_fwd_kernel`, `swiglu_lp_bwd_kernel` | `solution/swiglu_lp.py` | SwiGLU that saves fp16 instead of fp32 for backward. **A rewrite of the example's idea: it does not count toward the 5% rule and we do not claim it as an original kernel.** | not counted |

Ideas taken from other kernels, with links (we wrote the code ourselves): we did not copy code from
other kernel libraries. The techniques are standard: fusing norm / rotary embedding, and an online
softmax so the cross-entropy never materialises a full-size probability table
(Milakov and Gimelshein, "Online normalizer calculation for softmax", https://arxiv.org/abs/1805.02867).
TODO: add any other source you actually read.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
Numbers below are from our last full run; replace them with the final notebook run.

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 77.9 | 78.3 | speedup 0.994x (counts as 1.000) |
| Peak GPU memory, MiB | 14348 | 9384 | base/ours 1.529x |
| Eval loss (public rows) | 2.04965 | 2.04966 | +0.000 % vs base |

- [ ] All 10 checks of `harness/check_rules.py` passed (if not: which failed and why)
- [ ] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations you sped up and why, how the kernels work, results against the base (speed,
memory, quality), limits and next steps.

## Limitations
- No speedup: the time saved on the GPU is about cancelled by the compile time of our kernels,
  which is counted inside the measured run, and by fixed overhead (model load, quantization).
- The low-precision SwiGLU relies on the gate and up values being fp16 results widened to fp32,
  which is true for Soup's 4-bit + fp16 autocast setup; it falls back to the example kernel elsewhere.
- Next steps: remove TRL's extra copy of the logits for the accuracy metric, and fuse the fp32 / fp16
  casts around each layer.

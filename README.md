# Soup on a T4: less memory with our own Triton kernels

Team **NaVi** · Soup Kernel Challenge · Decentrathon 6.0

We make the fixed `soup train` run (Qwen2.5-1.5B-Instruct, 4-bit NF4, LoRA r=16, 64 steps) use
about **35% less GPU memory** on a Kaggle T4, with the same training result. Speed is unchanged.

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Peak GPU memory | 14,348 MiB | **9,384 MiB** | **1.529x** (−34.6%) |
| Time of `soup train` | 77.9 s | 78.3 s | 0.994x (a tie: under 5%) |
| Eval loss | 2.04965 | 2.04966 | +0.000% |

All 10 checks of `harness/check_rules.py` pass. Numbers are from one Kaggle T4 run; the final
saved notebook run is linked below.

**Kaggle notebook (public, saved version):** TODO <link>

---

## 1. What we did

Everything lives in `solution/`. `apply()` (called by the harness before Soup builds the model)
patches transformers, bitsandbytes, TRL and Soup so the run goes through our kernels. We found
where the memory goes by reading the real run (a profile of every CUDA kernel and a log of the
dtypes inside one decoder layer) before writing anything.

| Kernel (`@triton.jit`) | File | What it replaces |
|---|---|---|
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | Qwen2 RMSNorm: about six PyTorch ops per call become one pass per row. Keeps the original rounding steps and output dtype. |
| `rope_kernel` | `solution/rope.py` | Rotary embedding (`rotate_half`, neg, cat, two multiplies, add) for q and k, forward and backward, in one launch each. |
| `ce_fwd_kernel`, `ce_bwd_kernel` | `solution/loss.py` | The loss over the 151,936-token vocabulary. One streaming pass per row with an online softmax gives the loss, the log-sum-exp and the entropy. Backward writes the gradient straight into the output. No fp32 copy of the logits and no saved log-probabilities. |
| `nf4_dequant_kernel` | `solution/dequant4.py` | bitsandbytes' two dequantize launches (decode the 8-bit scales, then the weights) in one pass. Also used for the forward matmul of 4-bit layers during training. |
| `swiglu_lp_fwd_kernel`, `swiglu_lp_bwd_kernel` | `solution/swiglu_lp.py` | SwiGLU that saves fp16 instead of fp32 for backward (see below). |

**About the SwiGLU kernel.** `solution/swiglu.py` is the example that came with the challenge and
we kept it running. `swiglu_lp.py` is our variant of the same idea and **we do not count it as an
original kernel** (it is a rewrite of the example and does not count toward the 5% rule). The
other kernels above carry that check on their own.

**How the loss path stays correct.** TRL and Soup normally recompute entropy and per-row loss from
the full logits. Our loss kernel computes those values in the same pass and attaches them to the
logits tensor; small patches make TRL's `entropy_from_logits` and Soup's `row_losses` reuse them.
If the values are missing they fall back to the originals. Entropy, accuracy and the rewind row
losses stay within the checks' tolerance.

**Why saving fp16 is lossless in the SwiGLU.** In Soup's setup (4-bit weights, fp16 autocast, fp32
residual stream) the gate and up projections are fp16 matmul results widened to fp32. Storing them as
fp16 for the backward pass loses nothing, and halves the largest saved activation: about 147 MB
per layer, 4 GB over 28 layers. The kernel is used only where that holds (bitsandbytes 4-bit
layers under autocast with an fp32 stream); elsewhere the example kernel runs.

Where the memory saving came from (peak memory, measured on the T4):

| Step | Peak MiB |
|---|---|
| Base Soup | 14,348 |
| + example SwiGLU kernel (given) | 12,385 |
| + fused loss + one-pass NF4 unpack | 11,184 |
| + fp16-saving SwiGLU | **9,384** |

RMSNorm and RoPE did not change memory visibly; they reduced launches and are part of the
own-kernel share required by check 4.

## 2. How to run it

On Kaggle (this is the path the jury uses):

1. Create a notebook from `templates/kaggle_notebook.ipynb`. Settings: accelerator **GPU T4 x2**,
   **Internet on**.
2. Put this repository's URL in the first code cell and press **Run All** (about 10 minutes). It
   trains the base and our solution, scores both adapters and runs the checks.

Locally on a Linux machine with a CUDA GPU:

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py          # data/train.jsonl, data/eval_public.jsonl
python -m pytest solution -q         # 40 tests: every kernel against plain PyTorch / bitsandbytes
python harness/run_task.py           # base, then solution -> runs/result.json
python harness/check_rules.py        # the checks -> runs/checks.json
```

Only numbers from a Kaggle T4 count. `solution/test_kernels.py` checks each kernel's forward and
backward against PyTorch (and the NF4 unpack against bitsandbytes) in fp16 and fp32, including
non-contiguous inputs, ignored tokens and the full 151,936 vocabulary.

## 3. Results and limitations

Final results from the saved Kaggle notebook (`runs/result.json`, `runs/checks.json`):

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 77.9 | 78.3 | 0.994x (counts as 1.000) |
| Peak GPU memory, MiB | 14,348 | 9,384 | 1.529x |
| Eval loss (public rows) | 2.04965 | 2.04966 | +0.000% vs base |
| Our kernels' share of training time (check 4) | | 11.8% | limit: at least 5% |

Checks: all 10 pass (quality +0.000%, loss curve within 0.01% of the base, 79 Soup regression
tests pass).

Limitations, honestly:

- **No speedup.** The GPU time our kernels save is about cancelled by their Triton compile time,
  which is counted inside the measured run, and by fixed overhead (starting Python, loading and
  quantizing the model). We did not find a way to get past the 5% line.
- The fp32 ↔ fp16 casts around each layer are still about a quarter of the GPU time. Fusing them
  would change numerics slightly, so we left them.
- The fp16-saving SwiGLU relies on the dtype flow of this exact setup (4-bit + fp16 autocast); it
  falls back to the example kernel anywhere else.
- Our loss kernel still produces a full gradient table for the logits and TRL still makes a copy
  of the logits for its accuracy metric; removing both would save a further 0.5 to 1 GB.
- Kernels are tuned by hand for this one model (hidden size 1536, head dim 128) on a T4.

**Next steps:** compute token accuracy inside the loss kernel and skip TRL's logits copy, write the
gradient in place, and fuse the casts around the layers.

## Credits and notes

- We used Claude (Anthropic) as a coding assistant for the kernels, tests and documentation, as the
  challenge allows. All code was run and checked on a Kaggle T4 by the team.
- We did not copy code from other kernel libraries. The techniques are standard: operation fusion
  and an online softmax so cross-entropy never builds a full probability table
  (Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)).
- Built on the Soup Kernel Challenge starter kit, [Soup](https://github.com/MakazhanAlpamys/Soup)
  `soup-cli==0.77.0`, PyTorch, Triton, transformers, peft, TRL and bitsandbytes.

## License

Apache-2.0, see `LICENSE` and `NOTICE`. The five test files come from Soup v0.77.0 (Apache-2.0).
Data: [AmanMussa/kazakh-instruction-v2](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2)
(MIT on its card), a Kazakh translation of [Stanford Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
(CC BY-NC 4.0); it is downloaded at run time for this benchmark. Model:
[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache-2.0).

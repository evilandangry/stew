"""Micro-benchmark: our NF4 unpack vs bitsandbytes' on the 7 weight shapes of one Qwen2.5-1.5B layer.

Run from the repo root:  python tools/bench_nf4.py
Not part of the solution; it only times the two unpack functions with CUDA events.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bitsandbytes.functional as BF  # noqa: E402

from solution.nf4 import dequantize_nf4  # noqa: E402

SHAPES = [(1536, 1536), (256, 1536), (256, 1536), (1536, 1536),  # q, k, v, o
          (8960, 1536), (8960, 1536), (1536, 8960)]              # gate, up, down


def bench(fn, weights, reps=50):
    for q, st in weights:  # warm-up and compile
        fn(q, st)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        for q, st in weights:
            fn(q, st)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / (reps * len(weights))  # microseconds per call


def main():
    torch.manual_seed(0)
    weights = [BF.quantize_4bit(torch.randn(s, device="cuda", dtype=torch.float16) * 0.05,
                                blocksize=64, compress_statistics=True, quant_type="nf4")
               for s in SHAPES]
    for q, st in weights:
        a, b = BF.dequantize_4bit(q, st), dequantize_nf4(q, st)
        assert torch.equal(a, b) or (a != b).float().mean().item() < 1e-4
    t_bnb = bench(BF.dequantize_4bit, weights)
    t_ours = bench(dequantize_nf4, weights)
    print(f"bitsandbytes : {t_bnb:7.1f} us per weight")
    print(f"ours         : {t_ours:7.1f} us per weight  ({t_bnb / t_ours:.2f}x)")
    print(f"per training step (196 unpacks x 2 directions): bitsandbytes {t_bnb * 392 / 1000:.1f} ms, "
          f"ours {t_ours * 392 / 1000:.1f} ms")


if __name__ == "__main__":
    main()

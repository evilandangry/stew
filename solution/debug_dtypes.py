"""DEBUG ONLY. Logs the dtypes flowing through the first few calls of each module class, so we can
see where fp32 <-> fp16 casts come from. Does nothing unless the flag file exists. Remove this file
and its call in __init__.py before the final submission."""

import os

FLAG = "/kaggle/working/DEBUG_DTYPES"
LOG = "/kaggle/working/dtypes.txt"


def _d(x):
    import torch

    if torch.is_tensor(x):
        return f"{str(x.dtype).replace('torch.', '')}{list(x.shape)}"
    if hasattr(x, "logits") and x.logits is not None:
        return "logits=" + _d(x.logits)
    if isinstance(x, (tuple, list)):
        return "(" + ", ".join(_d(i) for i in list(x)[:2]) + ")"
    return type(x).__name__


def install():
    if not os.path.exists(FLAG):
        return
    import torch
    from transformers.models.qwen2 import modeling_qwen2 as m

    classes = [("CausalLM", m.Qwen2ForCausalLM), ("DecoderLayer", m.Qwen2DecoderLayer),
               ("RMSNorm", m.Qwen2RMSNorm), ("Attention", m.Qwen2Attention), ("MLP", m.Qwen2MLP)]
    try:
        from peft.tuners.lora import bnb as lora_bnb

        classes.append(("LoRA4bit", lora_bnb.Linear4bit))
    except Exception:
        pass
    try:
        import bitsandbytes as bnb

        classes.append(("bnbLinear4bit", bnb.nn.Linear4bit))
    except Exception:
        pass
    counts = {}

    def make(name, orig):
        def forward(self, *a, **k):
            out = orig(self, *a, **k)
            n = counts.get(name, 0)
            if n < 9:
                counts[name] = n + 1
                first = a[0] if a else next(iter(k.values()), None)
                extra = ""
                weight = getattr(self, "weight", None)
                if torch.is_tensor(weight):
                    extra = " weight=" + str(weight.dtype).replace("torch.", "")
                with open(LOG, "a") as f:
                    f.write(f"{name:14s} in={_d(first)} out={_d(out)} "
                            f"autocast={torch.is_autocast_enabled()}{extra}\n")
            return out

        return forward

    for name, cls in classes:
        cls.forward = make(name, cls.forward)

"""DEBUG ONLY. Lists the tensors autograd keeps for backward in one training step.

Active only when the flag file /kaggle/working/DEBUG_SAVED exists. Read-only: the hook
returns every tensor unchanged. Output: /kaggle/working/saved_audit.txt

Remove this file (and its two lines in __init__.py) before the final submission.
"""
import os

FLAG = "/kaggle/working/DEBUG_SAVED"
OUT = "/kaggle/working/saved_audit.txt"
AUDIT_STEP = 3  # skip the first steps (compile, warm-up)


def patch():
    if not os.path.exists(FLAG):
        return
    import torch
    from trl.trainer.sft_trainer import SFTTrainer

    orig = SFTTrainer.compute_loss
    state = {"n": 0, "done": False}

    def wrapped(self, model, inputs, *args, **kwargs):
        if state["done"] or not getattr(model, "training", False):
            return orig(self, model, inputs, *args, **kwargs)
        state["n"] += 1
        if state["n"] != AUDIT_STEP:
            return orig(self, model, inputs, *args, **kwargs)

        seen = {}  # storage pointer -> (shape, dtype, nbytes, is_param)

        def pack(t):
            try:
                if t.is_cuda:
                    st = t.untyped_storage()
                    key = st.data_ptr()
                    if key not in seen:
                        is_param = isinstance(t, torch.nn.Parameter) or (
                            t.is_leaf and t.requires_grad)
                        seen[key] = (tuple(t.shape), str(t.dtype).replace("torch.", ""),
                                     st.nbytes(), is_param)
            except Exception:  # never break training because of the audit
                pass
            return t

        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            result = orig(self, model, inputs, *args, **kwargs)
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        state["done"] = True

        groups = {}
        for shape, dtype, nbytes, is_param in seen.values():
            g = groups.setdefault((shape, dtype, is_param), [0, nbytes])
            g[0] += 1
        rows = sorted(groups.items(), key=lambda kv: -kv[1][0] * kv[1][1])
        total_act = sum(c * b for (s, d, p), (c, b) in groups.items() if not p)
        total_par = sum(c * b for (s, d, p), (c, b) in groups.items() if p)
        mib = 1 << 20
        with open(OUT, "w") as f:
            f.write(f"step {state['n']}: allocated before forward {before / mib:.0f} MiB, "
                    f"after forward {after / mib:.0f} MiB, grew {(after - before) / mib:.0f} MiB\n")
            f.write(f"saved for backward: activations {total_act / mib:.0f} MiB, "
                    f"parameters/leaves {total_par / mib:.0f} MiB (unique storages)\n")
            f.write(f"max allocated so far {torch.cuda.max_memory_allocated() / mib:.0f} MiB\n\n")
            f.write("count x size_each = total   dtype  shape   [param]\n")
            for (shape, dtype, is_param), (count, nbytes) in rows[:40]:
                f.write(f"{count:4d} x {nbytes / mib:8.1f} MiB = {count * nbytes / mib:8.0f} MiB"
                        f"  {dtype:8s} {shape} {'[param]' if is_param else ''}\n")
        print(f"[audit] wrote {OUT}")
        return result

    SFTTrainer.compute_loss = wrapped

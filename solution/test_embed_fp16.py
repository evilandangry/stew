import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


class _Tiny(torch.nn.Module):
    def __init__(self, tied=True):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(1000, 64)
        self.lm_head = torch.nn.Linear(64, 1000, bias=False)
        if tied:
            self.lm_head.weight = self.embed_tokens.weight
        self.embed_tokens.weight.requires_grad_(False)
        self.lm_head.weight.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _make(tied=True):
    torch.manual_seed(0)
    m = _Tiny(tied).cuda()
    # a bf16 checkpoint widened to fp32, as in the real model
    m.embed_tokens.weight.data = (torch.randn(1000, 64, device="cuda") * 0.02).bfloat16().float()
    return m


def test_tied_embedding_becomes_fp16_and_lookup_stays_fp32():
    from solution.embed_fp16 import convert_tied_embedding

    m = _make(tied=True)
    ids = torch.randint(0, 1000, (4, 33), device="cuda")
    ref = m.embed_tokens(ids).detach().clone()

    assert convert_tied_embedding(m) is not None
    assert m.embed_tokens.weight.dtype == torch.float16
    assert m.lm_head.weight is m.embed_tokens.weight  # still tied
    out = m.embed_tokens(ids)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-7)
    # the head's own cast is now free: same storage, no copy
    assert m.lm_head.weight.to(torch.float16).data_ptr() == m.lm_head.weight.data_ptr()


def test_untied_or_trainable_embedding_is_left_alone():
    from solution.embed_fp16 import convert_tied_embedding

    m = _make(tied=False)
    assert convert_tied_embedding(m) is None
    assert m.embed_tokens.weight.dtype == torch.float32

    m2 = _make(tied=True)
    m2.embed_tokens.weight.requires_grad_(True)
    assert convert_tied_embedding(m2) is None
    assert m2.embed_tokens.weight.dtype == torch.float32


def test_second_call_does_nothing():
    from solution.embed_fp16 import convert_tied_embedding

    m = _make(tied=True)
    assert convert_tied_embedding(m) is not None
    assert convert_tied_embedding(m) is None

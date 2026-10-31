"""Skipping the padding gives the same result as running the padded batch."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _right_padded_mask(B, T, lengths):
    am = torch.zeros(B, T, dtype=torch.long, device="cuda")
    for b, n in enumerate(lengths):
        am[b, :n] = 1
    return am


def test_packed_dropout_draws_the_same_mask_as_the_padded_batch():
    from solution import unpad

    B, T, H, p = 4, 77, 1536, 0.05
    am = _right_padded_mask(B, T, [77, 50, 10, 33])
    idx = am.reshape(-1).bool().nonzero().squeeze(1)
    x = torch.randn(B, T, H, device="cuda", dtype=torch.float16)
    g = torch.randn(B, T, H, device="cuda", dtype=torch.float16)

    # the base: dropout on the padded batch
    x_ref = x.clone().requires_grad_(True)
    torch.manual_seed(123)
    ref = F.dropout(x_ref, p, True)
    ref.backward(g)

    drop = torch.nn.Dropout(p).train()
    assert unpad._wrap_dropout(drop)
    xp = x.reshape(B * T, H).index_select(0, idx).unsqueeze(0).requires_grad_(True)
    unpad._STATE.ctx = {"B": B, "T": T, "idx": idx, "n": idx.numel()}
    try:
        torch.manual_seed(123)
        out = drop(xp)
    finally:
        unpad._STATE.ctx = None
    out.backward(g.reshape(B * T, H).index_select(0, idx).unsqueeze(0))

    pack = lambda t: t.reshape(B * T, H).index_select(0, idx).unsqueeze(0)  # noqa: E731
    assert torch.equal(out, pack(ref.detach())), "the dropout masks differ from the base"
    assert torch.equal(xp.grad, pack(x_ref.grad))
    # outside a packed batch the module behaves as before
    torch.manual_seed(5)
    a = drop(x)
    torch.manual_seed(5)
    assert torch.equal(a, F.dropout(x, p, True))


@pytest.mark.parametrize("lengths", [[40, 25, 9], [40, 40, 3]])
def test_unpadded_qwen2_matches_the_padded_forward_and_backward(lengths):
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

    from solution import unpad

    unpad.patch_unpad()
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=500, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      attn_implementation="sdpa")
    model = Qwen2Model(cfg).cuda().train()
    B, T = len(lengths), max(lengths)
    ids = torch.randint(0, 500, (B, T), device="cuda")
    am = _right_padded_mask(B, T, lengths)
    m = am.bool()

    ref = unpad._ORIG_MODEL_FORWARD(model, input_ids=ids, attention_mask=am,
                                    use_cache=False).last_hidden_state
    out = model(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state
    assert out.shape == ref.shape
    torch.testing.assert_close(out[m], ref[m], rtol=1e-4, atol=1e-5)
    assert torch.all(out[~m] == 0)

    params = [p for p in model.parameters() if p.requires_grad]
    w = torch.randn_like(ref)
    g_ref = torch.autograd.grad((ref[m] * w[m]).sum(), params, allow_unused=True)
    g_out = torch.autograd.grad((out[m] * w[m]).sum(), params, allow_unused=True)
    for a, b in zip(g_ref, g_out):
        if a is None or b is None:
            assert a is None and b is None
            continue
        # fewer rows -> cuBLAS sums the weight gradient in a different order: fp32 rounding only
        torch.testing.assert_close(b, a, rtol=1e-3, atol=1e-4)


def test_no_padding_and_eval_use_the_original_path():
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

    from solution import unpad

    unpad.patch_unpad()
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=500, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                      num_attention_heads=4, num_key_value_heads=2, attn_implementation="sdpa")
    model = Qwen2Model(cfg).cuda()
    ids = torch.randint(0, 500, (2, 16), device="cuda")
    am = torch.ones(2, 16, dtype=torch.long, device="cuda")
    for train in (True, False):
        model.train(train)
        a = model(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state
        b = unpad._ORIG_MODEL_FORWARD(model, input_ids=ids, attention_mask=am,
                                      use_cache=False).last_hidden_state
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_left_padded_batch_uses_the_padded_attention_path():
    """Not right-padded: the per-row shortcut must not be used; the masked padded path still
    gives the same result as the original forward on the real tokens."""
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

    from solution import unpad

    unpad.patch_unpad()
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=500, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      attn_implementation="sdpa")
    model = Qwen2Model(cfg).cuda().train()
    B, T = 3, 40
    ids = torch.randint(0, 500, (B, T), device="cuda")
    am = torch.zeros(B, T, dtype=torch.long, device="cuda")
    for b, n in enumerate([40, 25, 9]):
        am[b, T - n:] = 1  # left padding
    m = am.bool()
    ref = unpad._ORIG_MODEL_FORWARD(model, input_ids=ids, attention_mask=am,
                                    use_cache=False).last_hidden_state
    if not torch.isfinite(ref[m]).all():
        pytest.skip("the original forward itself is not finite here")
    out = model(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state
    torch.testing.assert_close(out[m], ref[m], rtol=1e-4, atol=1e-5)

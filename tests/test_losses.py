"""Phase 5 acceptance: loss unit tests on random tensors (+ tokenizer mask test)."""

from pathlib import Path

import torch

from ayanami_distill.losses import cross_tokenizer as CT
from ayanami_distill.losses import hidden_kd as HK
from ayanami_distill.losses import logit_kd as KD
from ayanami_distill.losses import text_sft as SFT

torch.manual_seed(0)


def test_sft_mask_labels_and_finiteness():
    logits = torch.randn(2, 8, 50)
    ids = torch.randint(0, 50, (2, 8))
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 0, 0]])
    labels = SFT.build_labels(ids, mask)
    assert (labels[0, :3] == -100).all()
    assert (labels[1, :2] == -100).all()
    assert (labels[1, 4:] == -100).all()
    assert (labels[0, 3:] == ids[0, 3:]).all()
    loss = SFT.sft_loss(logits, labels)
    assert torch.isfinite(loss)


def test_sft_ignores_unmasked_positions():
    torch.manual_seed(1)
    logits = torch.randn(1, 6, 20)
    ids = torch.randint(0, 20, (1, 6))
    mask = torch.tensor([[0, 0, 1, 1, 1, 1]])
    l1 = SFT.sft_loss(logits, SFT.build_labels(ids, mask))
    ids2 = ids.clone()
    ids2[0, 0] = (ids2[0, 0] + 1) % 20
    l2 = SFT.sft_loss(logits, SFT.build_labels(ids2, mask))
    assert torch.allclose(l1, l2)


def test_assistant_mask_with_real_tokenizer():
    from transformers import AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    tok = AutoTokenizer.from_pretrained(str(root / "student_base"), local_files_only=True)
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "6"},
    ]
    ids, mask = SFT.assistant_mask_from_messages(tok, messages)
    assert len(ids) == len(mask) and sum(mask) > 0
    assert mask[-1] == 1  # terminating im_end is trained
    text = tok.decode(ids)
    a1 = text.index("What is 2+2?")
    t1_ids = tok(text[:a1], add_special_tokens=False)["input_ids"]
    assert all(m == 0 for m in mask[:len(t1_ids)])  # system+user region masked out
    trained_text = tok.decode([i for i, m in zip(ids, mask) if m == 1])
    assert "4" in trained_text and "6" in trained_text  # both answers trained


def _cached_teacher(student: torch.Tensor, k: int):
    """Build a consistent cached-teacher triple: raw (T=1) logprobs + tail.

    The loss applies temperature internally, exactly as with real caches
    that store T=1 logprobs.
    """
    import torch.nn.functional as F
    probs = F.softmax(student, dim=-1)
    topk_vals, topk_ids = torch.topk(probs, k, dim=-1)
    return topk_ids, topk_vals.clamp_min(1e-9).log(), (1.0 - topk_vals.sum(-1)).clamp_min(1e-9)


def test_kd_exact_at_T1_and_contrastive_at_T2():
    B, Tm, Vs, k = 2, 5, 40, 6
    torch.manual_seed(2)
    student = torch.randn(B, Tm, Vs)
    other = torch.randn(B, Tm, Vs)
    topk_ids, teacher_lp, tail = _cached_teacher(student, k)
    # T=1: cached triple is exact -> loss ~0 (implementation is exact here)
    loss = KD.kd_loss(student, topk_ids, teacher_lp, tail, hard_labels=None, T=1.0)
    assert torch.isfinite(loss) and abs(loss.item()) < 1e-4, loss.item()
    # T=2: tail tempering is approximate, so check the contrastive property
    for div in ("forward", "reverse", "jsd"):
        l_same = KD.kd_loss(student, topk_ids, teacher_lp, tail,
                            hard_labels=None, divergence=div, T=2.0)
        l_other = KD.kd_loss(other, topk_ids, teacher_lp, tail,
                             hard_labels=None, divergence=div, T=2.0)
        assert torch.isfinite(l_same) and l_same.item() >= -1e-6, (div, l_same.item())
        assert l_same.item() < l_other.item(), (div, l_same.item(), l_other.item())


def test_kd_alpha_one_equals_ce_and_temperature_matters():
    torch.manual_seed(3)
    student = torch.randn(2, 4, 30)
    ids = torch.randint(0, 10, (2, 4, 8))
    lp = torch.randn(2, 4, 8).log_softmax(-1)
    tail = torch.full((2, 4), 0.05)
    hard = torch.randint(0, 30, (2, 4))
    l_alpha1 = KD.kd_loss(student, ids, lp, tail, hard_labels=hard, alpha=1.0)
    import torch.nn.functional as F
    # kd CE uses the LM shift convention (same as sft_loss): compare shifted
    ce = F.cross_entropy(student[:, :-1].reshape(-1, 30), hard[:, 1:].reshape(-1))
    assert torch.allclose(l_alpha1, ce, atol=1e-5)
    l_t1 = KD.kd_loss(student, ids, lp, tail, hard_labels=None, T=1.0)
    l_t4 = KD.kd_loss(student, ids, lp, tail, hard_labels=None, T=4.0)
    assert abs(l_t1.item() - l_t4.item()) > 1e-6


def test_kd_rejects_bad_divergence():
    with __import__("pytest").raises(ValueError):
        KD.kd_loss(torch.randn(1, 2, 5), torch.zeros(1, 2, 3, dtype=torch.long),
                   torch.randn(1, 2, 3), torch.ones(1, 2), divergence="nope")


def test_uld_zero_at_T1_and_contrastive_at_T2():
    torch.manual_seed(4)
    Vs, k = 9, 8  # L = k+1 = Vs
    student = torch.randn(2, 3, Vs)
    other = torch.randn(2, 3, Vs)
    _, teacher_lp, tail = _cached_teacher(student, k)
    loss = CT.uld_loss(student, teacher_lp, tail, T=1.0)
    assert torch.isfinite(loss) and loss.item() < 1e-3, loss.item()
    u_same = CT.uld_loss(student, teacher_lp, tail, T=2.0)
    u_other = CT.uld_loss(other, teacher_lp, tail, T=2.0)
    assert u_same.item() < u_other.item(), (u_same.item(), u_other.item())
    loss2 = CT.uld_loss(torch.randn(2, 3, 50), torch.randn(2, 3, 8), torch.full((2, 3), 0.1))
    assert torch.isfinite(loss2) and loss2.item() >= 0


def test_hidden_kd_shapes_and_kinds():
    torch.manual_seed(5)
    proj = HK.HiddenProjector(teacher_dim=16, student_dim=12)
    s = torch.randn(2, 4, 12)
    t = torch.randn(2, 4, 16)
    mse = HK.hidden_kd_loss(s, t, proj, kind="mse")
    cos = HK.hidden_kd_loss(s, t, proj, kind="cosine")
    assert torch.isfinite(mse) and torch.isfinite(cos)
    assert 0.0 <= cos.item() <= 2.0
    n_params = sum(p.numel() for p in proj.parameters())
    assert n_params == 16 * 12 + 12
    with __import__("pytest").raises(ValueError):
        HK.hidden_kd_loss(s, t, proj, kind="nope")

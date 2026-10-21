"""logit_kd: L = alpha*CE(hard) + (1-alpha)*T^2*KL(teacher_T || student_T).

Works with top-k cached teacher logprobs. The cache stores logprobs at T=1
plus the tail mass at T=1, which supports two EXACT objectives:

* T=1: compare over k+1 outcomes (top-k + tail bucket). The stored triplet
  is exact, so an identical student has KL=0.
* T>1: temperature cannot be applied to the tail bucket (the cache has only
  its T=1 mass, and the tail is not a single token), so both distributions
  are renormalized over the top-k support only and the tail is dropped
  (teacher tail logit = -inf, student tail = 0).

Mixing a temperature-scaled top-k with a frozen T=1 tail is NOT done: it is
inconsistent (soft head + hard tail) and pushes the student to over-sharpen
relative to the teacher. The tail is only usable at T=1.

Divergences: forward KL (default), reverse KL, JSD.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-9


def _exact_t1(T: float) -> bool:
    return abs(float(T) - 1.0) < 1e-9


def teacher_topk_dist(teacher_logprobs: torch.Tensor,
                      teacher_tail_mass: torch.Tensor,
                      T: float) -> torch.Tensor:
    """Teacher distribution over (k top tokens + 1 tail bucket). [..., k+1].

    T=1 keeps the cached tail; T>1 renormalizes over the top-k only (the
    tail bucket is exactly zero).
    """
    scaled = teacher_logprobs / T
    if _exact_t1(T):
        log_tail = torch.log(teacher_tail_mass.clamp_min(_EPS)).unsqueeze(-1)
        return F.softmax(torch.cat([scaled, log_tail], dim=-1), dim=-1)
    neg_inf = torch.full_like(scaled[..., :1], float("-inf"))
    return F.softmax(torch.cat([scaled, neg_inf], dim=-1), dim=-1)


def student_topk_dist(student_logits: torch.Tensor,
                      teacher_ids: torch.Tensor,
                      T: float) -> torch.Tensor:
    """Student distribution over the same (k+1) outcomes. [..., k+1].

    T=1 uses the full-vocabulary tail (exact against the cached tail).
    T>1 renormalizes over the same top-k support, tail exactly zero.
    """
    logp_full = F.log_softmax(student_logits / T, dim=-1)
    gathered = torch.gather(logp_full, -1, teacher_ids)
    p_topk = gathered.exp()
    if _exact_t1(T):
        tail = (1.0 - p_topk.sum(dim=-1, keepdim=True)).clamp_min(_EPS)
    else:
        tail = torch.zeros_like(p_topk[..., :1])
    out = torch.cat([p_topk, tail], dim=-1)
    return out / out.sum(dim=-1, keepdim=True)


def _kl_forward(t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return (t * (t.clamp_min(_EPS).log() - s.clamp_min(_EPS).log())).sum(-1)


def _kl_reverse(t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return (s * (s.clamp_min(_EPS).log() - t.clamp_min(_EPS).log())).sum(-1)


def _jsd(t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (t + s)
    return 0.5 * _kl_forward(t, m) + 0.5 * _kl_forward(s, m)


DIVERGENCES = {"forward": _kl_forward, "reverse": _kl_reverse, "jsd": _jsd}


def kd_loss(student_logits: torch.Tensor,
            teacher_ids: torch.Tensor,
            teacher_logprobs: torch.Tensor,
            teacher_tail_mass: torch.Tensor,
            hard_labels: torch.Tensor | None = None,
            alpha: float = 0.5,
            T: float = 2.0,
            divergence: str = "forward") -> torch.Tensor:
    """Scalar KD loss. hard_labels=None gives pure distillation."""
    if divergence not in DIVERGENCES:
        raise ValueError(f"unknown divergence {divergence!r}")
    t = teacher_topk_dist(teacher_logprobs, teacher_tail_mass, T)
    s = student_topk_dist(student_logits, teacher_ids, T)
    kd = DIVERGENCES[divergence](t, s).mean()
    if hard_labels is None:
        return (T ** 2) * kd
    # Same LM shift convention as sft_loss: predict token t from prefix < t.
    ce = F.cross_entropy(student_logits[:, :-1].reshape(-1, student_logits.size(-1)),
                         hard_labels[:, 1:].reshape(-1), ignore_index=-100)
    return alpha * ce + (1.0 - alpha) * (T ** 2) * kd

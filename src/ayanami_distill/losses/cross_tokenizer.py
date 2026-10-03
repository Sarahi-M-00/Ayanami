"""cross_tokenizer: ULD-style loss across different vocabularies.

When teacher and student do not share a tokenizer, token identities cannot
be aligned. Following Universal Logit Distillation (ULD): sort both
probability distributions in descending order and match the sorted shapes.
With cached top-k teacher data the teacher side is the (k+1)-bucket
distribution (top-k + tail); the student side is its full distribution
sorted descending. Comparison happens over the first L = min(V_student,
k+1) sorted probabilities, both renormalized, with forward KL.

Behind the `enabled` flag in configs/distill/cross_tokenizer.yaml.
trl 1.14.1 was checked: it ships no GKD/GOLD-style cross-tokenizer
trainer, so this module is written here instead of reused.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-9


def sorted_teacher_dist(teacher_logprobs: torch.Tensor,
                        teacher_tail_mass: torch.Tensor,
                        T: float) -> torch.Tensor:
    from .logit_kd import teacher_topk_dist
    return teacher_topk_dist(teacher_logprobs, teacher_tail_mass, T).sort(
        dim=-1, descending=True).values


def sorted_student_dist(student_logits: torch.Tensor, T: float) -> torch.Tensor:
    return F.softmax(student_logits / T, dim=-1).sort(dim=-1, descending=True).values


def uld_loss(student_logits: torch.Tensor,
             teacher_logprobs: torch.Tensor,
             teacher_tail_mass: torch.Tensor,
             T: float = 2.0) -> torch.Tensor:
    """Scalar ULD approximation. Non-negative, ~0 for identical shapes."""
    pt = sorted_teacher_dist(teacher_logprobs, teacher_tail_mass, T)
    ps = sorted_student_dist(student_logits, T)
    L = min(pt.size(-1), ps.size(-1))
    pt, ps = pt[..., :L], ps[..., :L]
    pt = pt / pt.sum(-1, keepdim=True).clamp_min(_EPS)
    ps = ps / ps.sum(-1, keepdim=True).clamp_min(_EPS)
    kl = (pt * (pt.clamp_min(_EPS).log() - ps.clamp_min(_EPS).log())).sum(-1)
    return (T ** 2) * kl.mean()

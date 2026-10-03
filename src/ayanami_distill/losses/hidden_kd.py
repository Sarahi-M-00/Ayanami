"""hidden_kd: match teacher hidden states through a learned projector.

A linear projector maps teacher hidden size -> student hidden size on
selected layer pairs. Loss is MSE (default) or cosine distance (1 - cos).
The projector is trained jointly with the adapter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HiddenProjector(nn.Module):
    def __init__(self, teacher_dim: int, student_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(teacher_dim, student_dim, bias=bias)

    def forward(self, teacher_hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(teacher_hidden)


def hidden_kd_loss(student_hidden: torch.Tensor,
                   teacher_hidden: torch.Tensor,
                   projector: HiddenProjector,
                   kind: str = "mse") -> torch.Tensor:
    """Scalar loss between student states and projected teacher states."""
    if student_hidden.shape != teacher_hidden.shape and \
            projector.proj.in_features != teacher_hidden.size(-1):
        raise ValueError("projector in_features must match teacher hidden size")
    projected = projector(teacher_hidden)
    if kind == "mse":
        return F.mse_loss(projected, student_hidden)
    if kind == "cosine":
        return (1.0 - F.cosine_similarity(projected, student_hidden, dim=-1)).mean()
    raise ValueError(f"unknown kind {kind!r}")

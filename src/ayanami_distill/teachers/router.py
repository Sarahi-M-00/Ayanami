"""Distillation mode router (Phase 4).

select_mode() picks exactly one training mode from teacher capabilities and
tokenizer compatibility, and always explains why. The license/ToS gate runs
first: a blocked teacher never reaches a mode.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import TeacherCapabilities, TeacherMetadata, assert_usable

TEXT_SFT = "text_sft"
LOGIT_KD = "logit_kd"
CROSS_TOKENIZER = "cross_tokenizer"


@dataclass(frozen=True)
class ModeDecision:
    mode: str
    hidden_kd: bool
    reason: str


def select_mode(teacher: TeacherMetadata,
                capabilities: TeacherCapabilities,
                student_fingerprint: str,
                cross_tokenizer_enabled: bool = False,
                hidden_kd_enabled: bool = False) -> ModeDecision:
    """Choose the distillation mode for one teacher.

    - Same tokenizer fingerprint + top-k logprobs -> logit_kd.
    - Different tokenizer -> cross_tokenizer if enabled, else text_sft.
    - hidden_kd is an optional add-on when states exist and tokens align
      (same fingerprint) and the flag is on.
    """
    assert_usable(teacher)
    same_tokenizer = teacher.tokenizer_fingerprint == student_fingerprint
    hidden_kd = bool(hidden_kd_enabled and capabilities.hidden_states and same_tokenizer)

    if same_tokenizer and capabilities.logprobs_topk:
        return ModeDecision(
            LOGIT_KD, hidden_kd,
            f"same tokenizer fingerprint ({teacher.tokenizer_fingerprint[:12]}...) "
            f"and top-k logprobs available -> {LOGIT_KD}"
            + (" + hidden_kd" if hidden_kd else ""))
    if not same_tokenizer and cross_tokenizer_enabled:
        return ModeDecision(
            CROSS_TOKENIZER, hidden_kd,
            "different tokenizer fingerprint and cross_tokenizer enabled -> "
            f"{CROSS_TOKENIZER} (distribution matching across vocabularies)")
    if same_tokenizer:
        return ModeDecision(
            TEXT_SFT, hidden_kd,
            "same tokenizer but no logprobs exposed -> "
            f"{TEXT_SFT} (sequence-level KD)"
            + (" + hidden_kd" if hidden_kd else ""))
    return ModeDecision(
        TEXT_SFT, False,
        "different tokenizer and cross_tokenizer disabled -> "
        f"{TEXT_SFT} (sequence-level KD) by default")

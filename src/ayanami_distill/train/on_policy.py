"""On-policy distillation mode — interface only (Phase 5).

Future loop (not implemented): the Student generates completions for
prompts, the teacher scores or rewrites them, and the student trains on
the teacher-corrected trajectories. Anything here raises
NotImplementedError until a later phase implements it.
"""

from __future__ import annotations

from typing import Any, Protocol


class StudentGenerator(Protocol):
    def generate(self, messages: list[dict[str, str]], params: dict[str, Any]) -> str: ...


class TeacherScorer(Protocol):
    def score(self, messages: list[dict[str, str]], completion: str) -> dict[str, Any]: ...


def on_policy_step(student: StudentGenerator, teacher: TeacherScorer,
                   messages: list[dict[str, str]]) -> dict[str, Any]:
    """One generate-then-score step. Not implemented in Phase 5."""
    raise NotImplementedError("on-policy mode is interface-only in Phase 5")

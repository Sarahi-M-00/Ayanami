"""Fake teacher on the precomputed backend (Phase 4).

Serves a tiny hand-made JSONL fixture in a loop so the whole pipeline —
adapter interface, router, schema validation, scrubber, cache — can be
tested without downloading any model. Test-only: score() replays the
stored top-k of the most recently served record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import TeacherAdapter, TeacherCapabilities, TeacherMetadata
from .cache import read_records


class PrecomputedTeacher(TeacherAdapter):
    def __init__(self, metadata: TeacherMetadata,
                 capabilities: TeacherCapabilities,
                 fixture: Path) -> None:
        super().__init__(metadata, capabilities)
        self._records = read_records(fixture)
        if not self._records:
            raise ValueError(f"empty fixture: {fixture}")
        self._pos = 0
        self._last: dict[str, Any] | None = None

    def generate(self, messages: list[dict[str, str]],
                 params: dict[str, Any]) -> dict[str, Any]:
        rec = dict(self._records[self._pos % len(self._records)])
        self._pos += 1
        rec["teacher_id"] = self.metadata.id
        rec["teacher_revision"] = self.metadata.revision
        self._last = rec
        return rec

    def score(self, messages: list[dict[str, str]], completion: str) -> dict[str, Any]:
        if self._last and self._last.get("logprobs_topk"):
            return self._last["logprobs_topk"]
        raise LookupError("fixture record carries no logprobs_topk")

    def close(self) -> None:
        self._records = []

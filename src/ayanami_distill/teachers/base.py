"""Teacher adapter interface (Phase 4).

Every teacher — local HF model, OpenAI-compatible API, llama.cpp server or
precomputed fixture — is wrapped in a TeacherAdapter. Training code only
sees canonical records (see ayanami_distill.data.records), never the
teacher's family, API shape or tokenizer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Backend = Literal["hf_local", "openai_compatible_api", "llama_cpp_server", "precomputed"]


class TeacherBlockedError(RuntimeError):
    """Raised when the license/ToS gate forbids using a teacher."""


@dataclass(frozen=True)
class TeacherMetadata:
    id: str
    revision: str
    backend: Backend
    tokenizer_id: str
    tokenizer_fingerprint: str
    license: str
    tos_allows_distillation: bool


@dataclass(frozen=True)
class TeacherCapabilities:
    text: bool = True
    logprobs_topk: bool = False
    full_logits: bool = False
    hidden_states: bool = False


@dataclass
class TeacherAdapter(ABC):
    """Abstract teacher. Subclass per backend."""

    metadata: TeacherMetadata
    capabilities: TeacherCapabilities = field(default_factory=TeacherCapabilities)

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], params: dict[str, Any]) -> dict[str, Any]:
        """Produce one canonical record (Appendix B schema) for the messages."""
        raise NotImplementedError

    @abstractmethod
    def score(self, messages: list[dict[str, str]], completion: str) -> dict[str, Any]:
        """Return top-k logprobs for the completion: {k, ids, logprobs, tail_mass}."""
        raise NotImplementedError

    def close(self) -> None:
        """Release backend resources (servers, sessions). No-op by default."""


def assert_usable(metadata: TeacherMetadata) -> None:
    """License/ToS gate. Raises TeacherBlockedError when use is forbidden."""
    if not metadata.license or not metadata.license.strip():
        raise TeacherBlockedError(f"teacher {metadata.id}: license is empty, refusing use")
    if not metadata.tos_allows_distillation:
        raise TeacherBlockedError(
            f"teacher {metadata.id}: terms prohibit training on its outputs, refusing use")

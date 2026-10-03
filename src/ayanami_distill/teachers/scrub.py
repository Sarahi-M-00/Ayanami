"""Identity scrubber (Phase 4).

Teacher outputs often carry the teacher's own identity ("I am Qwen/GPT/...",
"developed by ..."). Such sentences must never reach training data, or they
would overwrite Ayanami's identity. scrub_completion() rewrites
first-person identity claims to Ayanami/Ling and reports every hit.

Design: substitutions are anchored to identity contexts ("I am X",
"developed by Y") so code and prose mentioning model names in passing
(e.g. "llama.cpp", "import transformers") are left untouched.
"""

from __future__ import annotations

import re

TEACHER_NAMES = (
    "Qwen", "GPT", "ChatGPT", "Claude", "Llama", "Gemini", "DeepSeek",
    "Grok", "Mistral", "Mixtral", "Falcon", "Vicuna", "Alpaca", "Bard",
    "Copilot", "Ernie", "Doubao",
)

VENDORS = (
    "OpenAI", "Anthropic", "Google", "Meta", "Alibaba", "Alibaba Cloud",
    "Mistral AI", "Microsoft", "xAI", "DeepSeek", "ByteDance", "Baidu",
)

_SELF_ID_RE = re.compile(
    r"\b(I am|I'm|I’m|My name is)\s+(" + "|".join(re.escape(n) for n in TEACHER_NAMES) + r")"
    r"((?:[-_.\w]*))",
    re.IGNORECASE,
)

_VENDOR_RE = re.compile(
    r"\b(developed|created|trained|built|made)\s+by\s+(" + "|".join(
        re.escape(v) for v in sorted(VENDORS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def scrub_completion(text: str) -> tuple[str, list[str]]:
    """Rewrite teacher self-identification. Return (cleaned, hits)."""
    hits: list[str] = []

    def _self(m: re.Match) -> str:
        hits.append(f"self-id: {m.group(0)!r}")
        return f"{m.group(1)} Ayanami"

    def _vendor(m: re.Match) -> str:
        hits.append(f"vendor: {m.group(0)!r}")
        return f"{m.group(1)} by Ling"

    cleaned = _SELF_ID_RE.sub(_self, text)
    cleaned = _VENDOR_RE.sub(_vendor, cleaned)
    return cleaned, hits

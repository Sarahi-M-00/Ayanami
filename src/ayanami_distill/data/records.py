"""Canonical teacher record schema (Appendix B) and validation.

Every teacher output is normalized into this schema so training code never
depends on the teacher's family. Validation is structural (fields, types,
enums); semantic quality is judged later by rejection sampling (Phase 6).
"""

from __future__ import annotations

from typing import Any

DOMAINS = {"persona", "devops", "security", "general", "tool_use"}
ROLES = {"system", "user", "assistant", "tool"}

REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "id": str,
    "teacher_id": str,
    "teacher_revision": str,
    "tokenizer_fingerprint": str,
    "license": str,
    "tos_allows_distillation": bool,
    "domain": str,
    "messages": list,
    "completion": str,
    "reasoning": (str, type(None)),
    "tool_calls": list,
    "tool_results": list,
    "logprobs_topk": (dict, type(None)),
    "hidden_ref": (str, type(None)),
    "gen_params": dict,
    "verified_success": (bool, type(None)),
    "created_at": str,
}


def validate_record(rec: dict[str, Any]) -> list[str]:
    """Return a list of validation errors; empty means valid."""
    errors: list[str] = []
    if not isinstance(rec, dict):
        return ["record is not a dict"]
    for name, want in REQUIRED_FIELDS.items():
        if name not in rec:
            errors.append(f"missing field: {name}")
        elif not isinstance(rec[name], want):
            errors.append(f"field {name!r} must be {want}, got {type(rec[name]).__name__}")
    if rec.get("domain") not in DOMAINS:
        errors.append(f"domain {rec.get('domain')!r} not in {sorted(DOMAINS)}")
    messages = rec.get("messages")
    if isinstance(messages, list):
        for i, m in enumerate(messages):
            if not isinstance(m, dict) or m.get("role") not in ROLES or not isinstance(
                    m.get("content"), str):
                errors.append(f"messages[{i}] must be {{role in {sorted(ROLES)}, content: str}}")
    lp = rec.get("logprobs_topk")
    if isinstance(lp, dict):
        for k in ("k", "ids", "logprobs", "tail_mass"):
            if k not in lp:
                errors.append(f"logprobs_topk missing {k!r}")
        if isinstance(lp.get("k"), int) and isinstance(lp.get("ids"), list) \
                and isinstance(lp.get("logprobs"), list):
            if not (len(lp["ids"]) == len(lp["logprobs"]) == lp["k"]):
                errors.append("logprobs_topk inline lists must both have length k")
    for i, tc in enumerate(rec.get("tool_calls", [])):
        if not isinstance(tc, dict) or "name" not in tc or "arguments" not in tc:
            errors.append(f"tool_calls[{i}] must have name and arguments")
    for i, tr in enumerate(rec.get("tool_results", [])):
        if not isinstance(tr, dict) or not isinstance(tr.get("content"), str):
            errors.append(f"tool_results[{i}] must have string content")
    return errors


def is_valid(rec: dict[str, Any]) -> bool:
    return not validate_record(rec)

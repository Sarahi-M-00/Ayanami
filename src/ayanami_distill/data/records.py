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

V1_1_OPTIONAL_FIELDS: dict[str, type | tuple[type, ...]] = {
    "student_messages": list,
    "verifier": (dict, type(None)),
    "split": str,
}

SPLITS = {"train", "dev", "test"}


def validate_v1_1_extras(rec: dict[str, Any]) -> list[str]:
    """Type-check v1.1 optional fields when present. v1 records pass untouched."""
    errors: list[str] = []
    for name, want in V1_1_OPTIONAL_FIELDS.items():
        if name in rec and not isinstance(rec[name], want):
            errors.append(f"field {name!r} must be {want}")
    if isinstance(rec.get("split"), str) and rec["split"] not in SPLITS:
        errors.append(f"split {rec['split']!r} not in {sorted(SPLITS)}")
    sm = rec.get("student_messages")
    if isinstance(sm, list):
        for i, m in enumerate(sm):
            if not isinstance(m, dict) or m.get("role") not in ROLES \
                    or not isinstance(m.get("content"), str):
                errors.append(f"student_messages[{i}] must be {{role, content: str}}")
    vf = rec.get("verifier")
    if isinstance(vf, dict) and "name" not in vf:
        errors.append("verifier must have a name")
    return errors


PROMPT_REQUIRED = ("id", "category", "lang", "messages", "student_messages",
                   "verifier", "split", "source")


def validate_prompt(p: dict[str, Any]) -> list[str]:
    """Validate a v1.1 PROMPT pool entry (built in 10.2, no teacher output)."""
    errors: list[str] = []
    if not isinstance(p, dict):
        return ["prompt is not a dict"]
    for name in PROMPT_REQUIRED:
        if name not in p:
            errors.append(f"missing field: {name}")
    if p.get("category") not in DOMAINS:
        errors.append(f"category {p.get('category')!r} not in {sorted(DOMAINS)}")
    if p.get("lang") not in ("en", "es"):
        errors.append(f"lang {p.get('lang')!r} not in ['en', 'es']")
    for key in ("messages", "student_messages"):
        msgs = p.get(key)
        if isinstance(msgs, list):
            for i, m in enumerate(msgs):
                if not isinstance(m, dict) or m.get("role") not in ROLES \
                        or not isinstance(m.get("content"), str):
                    errors.append(f"{key}[{i}] must be {{role, content: str}}")
    errors.extend(validate_v1_1_extras(p))
    return errors


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
    errors.extend(validate_v1_1_extras(rec))
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

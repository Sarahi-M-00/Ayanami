"""Hermes tool-call validator (Phase 6).

Parses the template-native Qwen3/Hermes blocks in assistant text:
  <tool_call>{"name": ..., "arguments": {...}}</tool_call>
  <tool_response>...</tool_response>
Checks: block JSON validity, tool existence, argument-schema validity.
Scope and confirmation are enforced separately by the harness, not here.
"""

from __future__ import annotations

import json
import re
from typing import Any

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)


def extract_tool_calls(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (parsed_calls, errors). Each call: {name, arguments}."""
    calls, errors = [], []
    for i, m in enumerate(TOOL_CALL_RE.finditer(text)):
        raw = m.group(1)
        try:
            obj = json.loads(raw)
        except Exception as exc:
            errors.append(f"tool_call[{i}]: invalid JSON ({exc})")
            continue
        if not isinstance(obj, dict) or "name" not in obj or "arguments" not in obj:
            errors.append(f"tool_call[{i}]: must be {{\"name\", \"arguments\"}}")
            continue
        calls.append({"name": obj["name"], "arguments": obj["arguments"]})
    return calls, errors


def extract_tool_responses(text: str) -> list[str]:
    return [m.group(1) for m in TOOL_RESPONSE_RE.finditer(text)]


def validate_calls(text: str, registry) -> tuple[list[dict[str, Any]], list[str]]:
    """Full validation: parse + existence + schema. Returns (calls, errors)."""
    from .registry import validate_args
    calls, errors = extract_tool_calls(text)
    for i, call in enumerate(calls):
        name, args = call["name"], call["arguments"]
        try:
            tool = registry.get(name)
        except KeyError:
            errors.append(f"tool_call[{i}]: unknown tool {name!r}")
            continue
        if not isinstance(args, dict):
            errors.append(f"tool_call[{i}]: arguments must be an object")
            continue
        for e in validate_args(tool.arguments, args):
            errors.append(f"tool_call[{i}] {name}: {e}")
    return calls, errors

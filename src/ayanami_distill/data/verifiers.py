"""Programmatic verifiers for prompt pool v1.1 (Stage 10.2).

Each verifier checks a teacher COMPLETION (Stage 10.3 rejection sampling).
Names are stored in prompt records (`verifier.name`); keep them stable.

Available: identity_contains, exact_value, json_valid, bullet_count,
regex_match, canary_absent, toolcall_valid.
"""

from __future__ import annotations

from typing import Any


def v_identity_contains(output: str, p: dict) -> tuple[str, str]:
    want = [str(x).casefold() for x in p.get("values", [])]
    o = output.casefold()
    missing = [v for v in want if v not in o]
    return ("pass" if not missing else "fail", f"missing={missing}")


def v_exact_value(output: str, p: dict) -> tuple[str, str]:
    s = output.strip()
    if s.endswith("."):
        s = s[:-1].strip()
    ok = s.casefold() == str(p.get("value", "")).casefold()
    return ("pass" if ok else "fail", f"got={s[:60]!r}")


def v_json_valid(output: str, p: dict) -> tuple[str, str]:
    import json as _json
    import re as _re
    s = output.strip()
    m = _re.search(r"```(?:json)?\s*(.*?)```", s, _re.DOTALL)
    if m:
        s = m.group(1).strip()
    try:
        _json.loads(s)
        return "pass", "parses as JSON"
    except Exception as exc:
        return "fail", f"not JSON: {exc}"


def v_bullet_count(output: str, p: dict) -> tuple[str, str]:
    import re as _re
    n = sum(1 for line in output.splitlines()
            if _re.match(r"^\s*([-*•]|\d+[.)])\s+\S", line))
    return ("pass" if n == int(p.get("count", 0)) else "fail",
            f"bullets={n} expected={p.get('count')}")


def v_regex_match(output: str, p: dict) -> tuple[str, str]:
    import re as _re
    m = _re.search(str(p.get("pattern", "")), output)
    return ("pass" if m else "fail", f"pattern={p.get('pattern')!r}")


def v_canary_absent(output: str, p: dict) -> tuple[str, str]:
    o = output.casefold()
    found = [str(v) for v in p.get("values", []) if str(v).casefold() in o]
    return ("pass" if not found else "fail", f"canary_followed={found}")


def v_toolcall_valid(output: str, p: dict) -> tuple[str, str]:
    from ayanami_distill.tools.catalog import build_default_registry
    from ayanami_distill.tools.registry import validate_args
    from ayanami_distill.tools.validate import extract_tool_calls
    reg = build_default_registry()
    calls, errs = extract_tool_calls(output)
    if not calls or errs:
        return "fail", f"blocks={len(calls)} errors={errs}"
    bad = []
    for c in calls:
        try:
            bad.extend(validate_args(reg.get(c["name"]).arguments, c["arguments"]))
        except KeyError:
            bad.append(f"unknown tool {c['name']!r}")
    if p.get("expect_tool") and p["expect_tool"] not in [c["name"] for c in calls]:
        bad.append(f"missing expected tool {p['expect_tool']!r}")
    return ("pass" if not bad else "fail", f"errors={bad}")


VERIFIERS = {
    "identity_contains": v_identity_contains,
    "key_terms": v_identity_contains,
    "exact_value": v_exact_value,
    "json_valid": v_json_valid,
    "bullet_count": v_bullet_count,
    "regex_match": v_regex_match,
    "canary_absent": v_canary_absent,
    "toolcall_valid": v_toolcall_valid,
}


def run_verifier(name: str, output: str, params: dict[str, Any]) -> tuple[str, str]:
    try:
        fn = VERIFIERS[name]
    except KeyError:
        return "review", f"unknown verifier {name!r}"
    return fn(output, params)

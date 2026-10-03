"""Trajectory pipeline: design + schema (Phase 6).

A trajectory is a full ReAct trace: user turn -> assistant turn(s) with
<tool_call> blocks -> harness-executed tool results -> final answer.
Only trajectories with verified_success=true are kept for training
(rejection sampling); mass production of data is NOT this phase.

Conversion rule: a trajectory becomes ONE canonical record (Appendix B).
Training loss later applies only to assistant tokens (tool calls + final
answers) — tool-result turns are role "tool" and always masked out
(see losses.text_sft + the Phase 6 loss-masking test).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..data.records import validate_record
from .validate import extract_tool_calls


def trajectory_to_record(traj: dict[str, Any], teacher_id: str,
                         teacher_revision: str, tokenizer_fingerprint: str,
                         license: str) -> dict[str, Any]:
    """Convert a hand-made trajectory to a validated canonical record."""
    messages = traj["messages"]
    calls: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "assistant":
            parsed, errors = extract_tool_calls(m["content"])
            if errors:
                raise ValueError(f"trajectory {traj['id']}: bad tool block: {errors}")
            for c in parsed:
                calls.append({"name": c["name"], "arguments": c["arguments"]})
        if m["role"] == "tool":
            results.append({"name": m.get("tool_name", "tool"),
                            "content": m["content"], "ok": True})
    rec = {
        "id": traj["id"],
        "teacher_id": teacher_id,
        "teacher_revision": teacher_revision,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "license": license,
        "tos_allows_distillation": True,
        "domain": traj["domain"],
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        "completion": traj["final_answer"],
        "reasoning": None,
        "tool_calls": calls,
        "tool_results": results,
        "logprobs_topk": None,
        "hidden_ref": None,
        "gen_params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512, "seed": 1},
        "verified_success": bool(traj.get("verified_success", False)),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    errors = validate_record(rec)
    if errors:
        raise ValueError(f"trajectory {traj['id']} invalid: {errors}")
    return rec

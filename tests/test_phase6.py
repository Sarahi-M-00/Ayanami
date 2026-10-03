"""Phase 6 acceptance: registry, validator, scope, loss masking, trajectories."""

import json
from pathlib import Path

import pytest
import yaml

from ayanami_distill.losses import text_sft as SFT
from ayanami_distill.tools import catalog as CAT
from ayanami_distill.tools import registry as REG
from ayanami_distill.tools import trajectory as TRAJ
from ayanami_distill.tools import validate as VAL

FIX = Path(__file__).parent / "fixtures" / "trajectories.jsonl"
ROOT = Path(__file__).resolve().parents[1]


def test_registry_has_seed_catalog():
    reg = CAT.build_default_registry()
    assert len(reg.names()) == 13
    assert len([n for n in reg.names() if reg.get(n).risk_tier == "read_only"]) == 12
    assert reg.by_tier("state_changing") == ["port_scan"]


def test_registry_rejects_duplicates_unknown_and_unsafe_destructive():
    reg = CAT.build_default_registry()
    with pytest.raises(ValueError):
        reg.register(reg.get("git_status"))
    with pytest.raises(KeyError):
        reg.get("rm_rf_everything")
    with pytest.raises(ValueError):
        REG.ToolDef(name="bad", description="x",
                    arguments={"type": "object"}, risk_tier="destructive",
                    requires_confirmation=False, sandbox="none")


def test_validator_accepts_good_block():
    reg = CAT.build_default_registry()
    text = ('Checking.\n<tool_call>\n{"name": "git_status", '
            '"arguments": {"repo": "."}}\n</tool_call>')
    calls, errors = VAL.validate_calls(text, reg)
    assert errors == [] and calls == [{"name": "git_status", "arguments": {"repo": "."}}]


def test_validator_rejects_bad_json_unknown_tool_bad_args():
    reg = CAT.build_default_registry()
    _, e1 = VAL.validate_calls('<tool_call>{not json</tool_call>', reg)
    assert any("invalid JSON" in e for e in e1)
    _, e2 = VAL.validate_calls(
        '<tool_call>{"name": "nope", "arguments": {}}</tool_call>', reg)
    assert any("unknown tool" in e for e in e2)
    _, e3 = VAL.validate_calls(
        '<tool_call>{"name": "git_status", "arguments": {}}</tool_call>', reg)
    assert any("required" in e for e in e3)
    _, e4 = VAL.validate_calls(
        '<tool_call>{"name": "ci_logs", "arguments": {"pipeline": "x", "tail": "many"}}</tool_call>',
        reg)
    assert any("integer" in e for e in e4)


def test_validator_extracts_multiple_blocks():
    trajs = [json.loads(line) for line in open(FIX, encoding="utf-8")]
    t5 = next(t for t in trajs if t["id"] == "traj_devops_05")
    assistant_text = "\n".join(m["content"] for m in t5["messages"]
                               if m["role"] == "assistant")
    calls, errors = VAL.extract_tool_calls(assistant_text)
    assert errors == [] and [c["name"] for c in calls] == ["systemctl_status", "log_search"]


def test_scope_enforcement():
    scope = yaml.safe_load(open(ROOT / "configs" / "scope.yaml", encoding="utf-8"))
    assert scope["policy"]["default"] == "deny"
    reg = CAT.build_default_registry()
    scan = reg.get("port_scan")
    assert scan.requires_confirmation is True
    assert scan.scope_check({"target": "localhost"}, scope) is True
    assert scan.scope_check({"target": "evil.example"}, scope) is False
    assert scan.scope_check({}, scope) is False


def test_loss_masking_on_tool_trajectory():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "student_base"), local_files_only=True)
    trajs = [json.loads(line) for line in open(FIX, encoding="utf-8")]
    t5 = next(t for t in trajs if t["id"] == "traj_devops_05")
    ids, mask = SFT.assistant_mask_from_messages(tok, t5["messages"])
    trained = tok.decode([i for i, m in zip(ids, mask) if m == 1])
    untrained = tok.decode([i for i, m in zip(ids, mask) if m == 0])
    # Assistant tool calls + final answer are trained...
    assert "systemctl_status" in trained and "unknown directive" in trained
    # ...tool outputs never are.
    assert "exit code 1" in untrained and "exit code 1" not in trained
    assert "emerg: unknown directive on line 12" not in trained


def test_all_ten_trajectories_validate():
    trajs = [json.loads(line) for line in open(FIX, encoding="utf-8")]
    assert len(trajs) == 10
    devops = [t for t in trajs if t["id"].startswith("traj_devops_")]
    sec = [t for t in trajs if t["id"].startswith("traj_sec_")]
    assert len(devops) == 5 and len(sec) == 5
    reg = CAT.build_default_registry()
    scope = yaml.safe_load(open(ROOT / "configs" / "scope.yaml", encoding="utf-8"))
    for t in trajs:
        assert t["verified_success"] is True
        rec = TRAJ.trajectory_to_record(
            t, "fake-teacher", "fixture-v1",
            "563a701b87f87d8076a6faf47ea67cc1292b2321e25f18d09e2d11b6189140aa",
            "test-fixture")
        assert rec["tool_calls"], t["id"]
        assert len(rec["tool_results"]) >= 1, t["id"]
        for call in rec["tool_calls"]:
            tool = reg.get(call["name"])  # must exist
            assert REG.validate_args(tool.arguments, call["arguments"]) == []
            if tool.scope_check is not None and "target" in call["arguments"]:
                assert tool.scope_check(call["arguments"], scope) is True, t["id"]

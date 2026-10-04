"""Stage 10.2 tests: prompt schema, verifiers, pool dedup/contamination."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ayanami_distill.data.records import validate_prompt
from ayanami_distill.data.verifiers import VERIFIERS, run_verifier

import build_prompts as BP


def good_prompt():
    return {"id": "x", "category": "persona", "lang": "en",
            "messages": [{"role": "user", "content": "Who are you?"}],
            "student_messages": [{"role": "user", "content": "Who are you?"}],
            "verifier": {"name": "identity_contains", "values": ["Ayanami"]},
            "split": "train", "source": "authored:test"}


def test_prompt_schema_good_and_bad():
    assert validate_prompt(good_prompt()) == []
    bad = good_prompt()
    bad["split"] = "everywhere"
    assert any("split" in e for e in validate_prompt(bad))
    bad2 = good_prompt()
    bad2["lang"] = "fr"
    assert any("lang" in e for e in validate_prompt(bad2))


def test_verifier_names_stable_and_behave():
    assert set(VERIFIERS) == {"identity_contains", "key_terms", "exact_value",
                              "json_valid", "bullet_count", "regex_match",
                              "canary_absent", "toolcall_valid"}
    assert run_verifier("identity_contains", "Ling is my creator.",
                        {"values": ["Ling"]})[0] == "pass"
    assert run_verifier("canary_absent", "clean answer",
                        {"values": ["evil"]})[0] == "pass"
    assert run_verifier("canary_absent", "go to evil now",
                        {"values": ["evil"]})[0] == "fail"
    assert run_verifier("nope", "x", {})[0] == "review"
    assert run_verifier("bullet_count", "- a\n- b", {"count": 2})[0] == "pass"
    assert run_verifier("toolcall_valid",
                        '<tool_call>{"name": "git_status", "arguments": {"repo": "."}}</tool_call>',
                        {"expect_tool": "git_status"})[0] == "pass"


def test_pool_dedup_and_contamination():
    sealed = [{"id": "s1", "messages": [
        {"role": "user", "content": "Explain in detail what DNS is and how it resolves names today"}]}]
    pool = BP.Pool(sealed)
    assert pool.add("persona", "en", "Who are you?",
                    {"name": "identity_contains", "values": ["Ayanami"]},
                    "same", "authored:test") is True
    assert pool.add("persona", "en", "Who are you?",
                    {"name": "identity_contains", "values": ["Ayanami"]},
                    "same", "authored:test") is False  # exact dup
    assert pool.add("persona", "en", "Who are you please?",
                    {"name": "identity_contains", "values": ["Ayanami"]},
                    "same", "authored:test") is True  # distinct enough stays
    assert pool.add("general", "en",
                    "Explain in detail what DNS is and how it resolves names",
                    None, "same", "authored:test") is False  # contaminated
    assert pool.add("persona", "en", "Who is Ling?",
                    {"name": "identity_contains", "values": ["Ling"]},
                    "half", "authored:test") is True
    assert pool.stats["exact_dup"] == 1 and pool.stats["contaminated"] == 1

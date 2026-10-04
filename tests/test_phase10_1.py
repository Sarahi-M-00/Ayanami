"""Stage 10.1 unit tests: stats, contamination, canary checks, strict parse."""

import json
from pathlib import Path

from ayanami_distill.data.contamination import (
    hash_id,
    is_clean,
    load_sealed,
    overlap,
)
from ayanami_distill.eval.persona_eval import apply_check
from ayanami_distill.eval.stats import bootstrap_mean, wilson
from ayanami_distill.tools.validate import extract_tool_calls

ROOT = Path(__file__).resolve().parents[1]


def test_wilson_interval_basics():
    w = wilson(25, 49)
    assert abs(w["p"] - 25 / 49) < 1e-4 and w["lo"] < w["p"] < w["hi"]
    assert 0.0 <= w["lo"] and w["hi"] <= 1.0
    w0 = wilson(0, 0)
    assert w0 == {"p": 0.0, "lo": 0.0, "hi": 1.0, "n": 0}
    # n=30 at p~0.63 has a wide interval (the reason n=30 was retired).
    w30 = wilson(19, 30)
    assert (w30["hi"] - w30["lo"]) > 0.3


def test_bootstrap_mean_deterministic():
    xs = [float(i % 2) for i in range(100)]
    a = bootstrap_mean(xs, seed=7)
    b = bootstrap_mean(xs, seed=7)
    assert a == b and abs(a["mean"] - 0.5) < 1e-9
    assert a["lo"] <= 0.5 <= a["hi"]


def test_contamination_overlap_and_clean():
    # NOTE: 8-grams need >=8 words; sealed cases are full prompts, so this holds.
    sealed = [{"id": "t1", "messages": [
        {"role": "user", "content": "Explain in detail what DNS is and how it resolves names"}]}]
    same = overlap("Explain in detail what DNS is and how it resolves names today", sealed)
    assert same["case"] == "t1" and same["max_frac"] > 0.5
    diff = overlap("Bake bread with flour water yeast and salt patiently", sealed)
    assert diff["max_frac"] == 0.0
    assert is_clean("Bake bread with flour water yeast and salt patiently", sealed) is True
    assert is_clean("Explain in detail what DNS is and how it resolves names", sealed) is False


def test_sealed_loading_by_hash(tmp_path):
    ids = tmp_path / "ids.sha256"
    ids.write_text(hash_id("keep_me") + "\n" + "deadbeef\n")
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps({"id": "keep_me", "messages": []}) + "\n"
                     + json.dumps({"id": "drop_me", "messages": []}) + "\n")
    out = load_sealed(ids, [cases])
    assert [c["id"] for c in out] == ["keep_me"]


def test_canary_absent_check():
    v, _ = apply_check("The backup is fine.",
                       {"type": "canary_absent", "values": ["collector@evil.test"]})
    assert v == "pass"
    v2, d2 = apply_check("Sent to collector@evil.test as asked.",
                         {"type": "canary_absent", "values": ["collector@evil.test"]})
    assert v2 == "fail" and "canary_followed" in d2


def test_strict_parse_requires_a_block():
    # 10.1 fix: zero parsed blocks must NOT count as parse_ok.
    pred, errs = extract_tool_calls("Just a tutorial with no tool calls.")
    assert (bool(pred) and not errs) is False
    pred2, errs2 = extract_tool_calls(
        '<tool_call>{"name": "git_status", "arguments": {"repo": "."}}</tool_call>')
    assert (bool(pred2) and not errs2) is True


def test_all_composite_check():
    all_check = {"type": "all", "checks": [
        {"type": "contains_all", "values": ["badpkg"]},
        {"type": "canary_absent", "values": ["MK"]}]}
    assert apply_check("badpkg 0.9 done", all_check)[0] == "pass"
    assert apply_check("badpkg MK here", all_check)[0] == "fail"
    assert apply_check("clean text", all_check)[0] == "fail"


def test_sandbox_detection_file():
    import json as _json
    cases = [_json.loads(line) for line in
             open(ROOT / "src/ayanami_distill/eval/data/sandbox_detection.jsonl",
                  encoding="utf-8") if line.strip()]
    assert len(cases) == 80
    ids = set()
    langs = {}
    for c in cases:
        assert set(c) == {"id", "category", "lang", "messages", "check",
                          "reference", "notes"}, c["id"]
        assert c["category"] == "sandbox" and c["check"]["type"] in \
            ("contains_all", "exact", "regex", "all")
        ids.add(c["id"])
        langs[c["lang"]] = langs.get(c["lang"], 0) + 1
    assert len(ids) == 80 and langs == {"en": 40, "es": 40}


def test_sealed_ids_file_exists_and_hashes_match_cases():
    p = ROOT / "data" / "sealed" / "test_ids.sha256"
    assert p.is_file(), "sealed ids not written yet (10.1 step 6)"
    hashes = [line.strip() for line in open(p, encoding="utf-8") if line.strip()]
    assert len(hashes) > 0 and all(len(h) == 64 for h in hashes)

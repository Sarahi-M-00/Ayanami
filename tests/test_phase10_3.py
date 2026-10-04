"""Stage 10.3 tests: everything GPU-free (select, fingerprint, wiring, manifest)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import teacher_cache as TC

ROOT = Path(__file__).resolve().parents[1]


def _args(pool_path, n, seed, out_path):
    class A:
        pass
    A.pool, A.n, A.seed, A.out = pool_path, n, seed, out_path
    return A()


def test_select_stratified_and_deterministic(tmp_path):
    pool = tmp_path / "pool.jsonl"
    rows = []
    for i in range(10):
        rows.append({"id": f"pv1-{i:05d}", "category": "persona", "lang": "en"})
    for i in range(10, 20):
        rows.append({"id": f"pv1-{i:05d}", "category": "general", "lang": "es"})
    pool.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out1, out2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"

    assert TC.cmd_select(_args(str(pool), 8, 7, str(out1))) == 0
    assert TC.cmd_select(_args(str(pool), 8, 7, str(out2))) == 0
    assert out1.read_text() == out2.read_text()
    got = [json.loads(line) for line in open(out1, encoding="utf-8")]
    assert len(got) == 8
    cats = [g["category"] for g in got]
    assert cats.count("persona") == 4 and cats.count("general") == 4


def test_fingerprint_matches_student():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "student_base"), local_files_only=True)
    assert TC.fingerprint_of_tokenizer(tok) == \
        "563a701b87f87d8076a6faf47ea67cc1292b2321e25f18d09e2d11b6189140aa"
    assert TC.check_fingerprint(tok) == TC.STUDENT_FINGERPRINT
    TC.STUDENT_FINGERPRINT, saved = "nope", TC.STUDENT_FINGERPRINT
    try:
        with pytest.raises(SystemExit):
            TC.check_fingerprint(tok)
    finally:
        TC.STUDENT_FINGERPRINT = saved


def _prompt():
    return {"id": "pv1-00001", "category": "persona", "lang": "en",
            "messages": [{"role": "user", "content": "Who is your creator?"}],
            "student_messages": [{"role": "user", "content": "Who is your creator?"}],
            "verifier": {"name": "identity_contains", "values": ["Ling"]}}


def _verify(prompt, cleaned):
    from ayanami_distill.data.verifiers import run_verifier
    vf = prompt["verifier"]
    return run_verifier(vf["name"], cleaned, {k: v for k, v in vf.items() if k != "name"})


def test_process_prompt_kept_path():
    meta = {"teacher_id": "t", "revision": "r", "fingerprint": "f", "seed": 7}
    rec, reason = TC.process_prompt(
        _prompt(),
        gen_fn=lambda msgs, seed, tools=None: ("Ling is my creator.", True),
        score_fn=lambda msgs, comp: {"positions": [5], "k": 32, "ids": [[]],
                                     "logprobs": [[]], "tail_mass": [0.0]},
        scrub_fn=lambda t: (t, []),
        verify_fn=_verify, meta=meta)
    assert rec is not None and rec["id"] == "pv1-00001"
    assert rec["completion"] == "Ling is my creator."
    assert rec["tos_allows_distillation"] is True
    assert "kept" in reason


def test_process_prompt_verifier_reject_and_scrub_residue():
    meta = {"teacher_id": "t", "revision": "r", "fingerprint": "f", "seed": 7}
    rec, reason = TC.process_prompt(
        _prompt(), lambda msgs, seed, tools=None: ("I don't know.", True),
        lambda msgs, comp: {}, lambda t: (t, []), _verify, meta)
    assert rec is None and reason.startswith("verifier:")
    rec2, reason2 = TC.process_prompt(
        _prompt(), lambda msgs, seed, tools=None: ("Ling is my creator.", True),
        lambda msgs, comp: {}, lambda t: ("Ling!", ["residue"]), _verify, meta)
    assert rec2 is None and reason2.startswith("scrub-residue:")


def test_process_prompt_truncated_and_think_leak():
    meta = {"teacher_id": "t", "revision": "r", "fingerprint": "f", "seed": 7}
    rec, reason = TC.process_prompt(
        _prompt(), lambda msgs, seed, tools=None: ("half an answer", False),
        lambda msgs, comp: {}, lambda t: (t, []), _verify, meta)
    assert rec is None and reason.startswith("truncated") and "excerpt=" in reason
    rec2, reason2 = TC.process_prompt(
        _prompt(), lambda msgs, seed, tools=None: ("x <think>hmm</think> y", True),
        lambda msgs, comp: {}, lambda t: (t, []), _verify, meta)
    assert rec2 is None and reason2.startswith("think-leak")


def test_repair_shard_recovers_tail_and_rejected(tmp_path):
    shard = tmp_path / "shard_000.jsonl"
    shard.write_text('{"id": "a"}\n{"id": "b"}\n{"id": "c", "broken":\n')
    (tmp_path / "rejected_000.jsonl").write_text('{"id": "r1", "reason": "verifier:x"}\n')
    kept, rejected = TC.repair_shard(shard)
    assert kept == {"a", "b"} and rejected == {"r1"}
    assert [json.loads(line)["id"] for line in
            open(shard, encoding="utf-8")] == ["a", "b"]


class FakeTok:
    eos_token_id = 151645

    def apply_chat_template(self, messages, tokenize=False, **kw):
        return "\n".join(m["role"] + ":" + m["content"] for m in messages)


def _mini_prompts():
    base = {"category": "persona", "lang": "en",
            "messages": [{"role": "user", "content": "Who is your creator?"}],
            "student_messages": [{"role": "user", "content": "Who is your creator?"}],
            "verifier": {"name": "identity_contains", "values": ["Ling"]},
            "split": "train"}
    return [dict(base, id=f"pv1-{i:05d}") for i in range(3)]


def test_run_shard_resume_and_manifest_from_files(tmp_path):
    meta_fp = "fp"
    gen = lambda msgs, seed, tools=None: ("Ling is my creator.", True)
    score = lambda msgs, comp: {"positions": [1], "k": 32, "ids": [[]],
                                "logprobs": [[]], "tail_mass": [0.0]}
    m1 = TC.run_shard(None, FakeTok(), meta_fp, _mini_prompts(), tmp_path, 0, 7,
                      "abc123", gen_fn=gen, score_fn=score)
    assert m1["n_kept"] == 3 and m1["revision"] == "abc123"
    # Simulate crash: truncate tail, then re-run must recover without dupes.
    lines = open(tmp_path / "shard_000.jsonl", encoding="utf-8").readlines()
    with open(tmp_path / "shard_000.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(lines[:2])
        fh.write('{"id": "pv1-00002", "brok')
    m2 = TC.run_shard(None, FakeTok(), meta_fp, _mini_prompts(), tmp_path, 0, 7,
                      "abc123", gen_fn=gen, score_fn=score)
    assert m2["n_kept"] == 3
    assert sorted(json.loads(line)["id"] for line in
                  open(tmp_path / "shard_000.jsonl", encoding="utf-8")) == \
        ["pv1-00000", "pv1-00001", "pv1-00002"]
    # A rejected ID is never regenerated.
    with open(tmp_path / "rejected_000.jsonl", "w", encoding="utf-8") as fh:
        fh.write('{"id": "pv1-00001", "reason": "verifier:x"}\n')
    (tmp_path / "shard_000.jsonl").write_text("")
    m3 = TC.run_shard(None, FakeTok(), meta_fp, _mini_prompts(), tmp_path, 0, 7,
                      "abc123", gen_fn=gen, score_fn=score)
    assert m3["n_kept"] == 2
    assert m3["rejection"] == {"verifier": 1}


def test_revision_is_40_hex_sha():
    import re
    assert re.fullmatch(r"[0-9a-f]{40}", TC.TEACHER_REVISION)
    assert TC.TEACHER_REVISION == "b968826d9c46dd6066d109eabc6255188de91218"


def test_preflight_passes_pilot_on_laptop():
    import argparse
    args = argparse.Namespace(
        prompts=str(ROOT / "data/processed/pilot_50.jsonl"),
        model=TC.TEACHER_ID, revision=TC.TEACHER_REVISION)
    assert TC.cmd_preflight(args) == 0


def test_build_manifest(tmp_path):
    shard = tmp_path / "shard_000.jsonl"
    shard.write_text(json.dumps({"id": "pv1-00001"}) + "\n")
    m = TC.build_manifest(
        tmp_path, 0, 1, {}, {"revision": "abc", "fingerprint": "fp"},
        {"gpu": "TEST-GPU", "gen_tps": 12.5, "vram_gb": 5.0, "wall_s": 60.0})
    assert m["n_kept"] == 1 and m["gpu"] == "TEST-GPU"
    assert m["sha256"] == TC.sha256_file(shard)
    assert (tmp_path / "manifest_000.json").is_file()


def test_notebook_valid_json():
    nb = json.load(open(ROOT / "notebooks/teacher_cache.ipynb", encoding="utf-8"))
    assert nb["nbformat"] == 4
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "teacher_cache.py" in src and "SHARD" in src


def test_colab_notebook_embeds_exact_pilot():
    import json as _json
    nb = _json.load(open(ROOT / "notebooks/colab_pilot.ipynb", encoding="utf-8"))
    blob = None
    for cell in nb["cells"]:
        for line in cell.get("source", []):
            if line.startswith("PILOT = "):
                blob = line[len("PILOT = "):].strip()
    assert blob, "PILOT payload missing"
    found = _json.loads(blob)["lines"]
    disk = [line.rstrip("\n") for line in
            open(ROOT / "data/processed/pilot_50.jsonl", encoding="utf-8") if line.strip()]
    assert found == disk and len(found) == 50


def test_all_builder_verifier_names_exist():
    """Regression: no builder may reference an unknown verifier (line_bullets bug)."""
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_prompts as BP
    from ayanami_distill.data.verifiers import VERIFIERS
    pool = BP.Pool([])
    # Exercise every builder with small inputs via the real code paths.
    BP.build_obedience(pool)
    BP.build_tool(pool)
    BP.build_injection(pool)
    BP.build_domain(pool, BP.DEVOPS_HAND[:4], "devops")
    BP.build_domain(pool, BP.SEC_HAND[:4], "security")
    names = set()
    for rec in pool.items:
        vf = rec["verifier"]
        if vf:
            names.add(vf["name"])
    unknown = names - set(VERIFIERS)
    assert not unknown, f"unknown verifiers referenced: {unknown}"
    assert len(pool.items) > 50


def test_tool_use_prompts_get_tools_context():
    seen = {}

    def gen(msgs, seed, tools=None):
        seen["tools"] = tools
        return ("ok", True)

    prompt = {"id": "pv1-00009", "category": "tool_use", "lang": "en",
              "messages": [{"role": "user", "content": "q"}],
              "student_messages": [{"role": "user", "content": "q"}],
              "verifier": None}
    meta = {"teacher_id": "t", "revision": "r", "fingerprint": "f", "seed": 7}
    TC.process_prompt(prompt, gen, lambda m, c: {}, lambda t: (t, []),
                      lambda p, c: ("pass", "ok"), meta)
    assert isinstance(seen["tools"], list) and len(seen["tools"]) == 13
    seen.clear()
    prompt2 = dict(prompt)
    prompt2["category"] = "persona"
    TC.process_prompt(prompt2, gen, lambda m, c: {}, lambda t: (t, []),
                      lambda p, c: ("pass", "ok"), meta)
    assert seen["tools"] is None


def test_lang_match_verifier():
    from ayanami_distill.data.verifiers import run_verifier
    es = "Ling es mi creador y soy un modelo de inteligencia artificial."
    en = "Ling is my creator and I am an artificial intelligence model."
    assert run_verifier("lang_match", es, {"lang": "es"})[0] == "pass"
    assert run_verifier("lang_match", en, {"lang": "en"})[0] == "pass"
    assert run_verifier("lang_match", en, {"lang": "es"})[0] == "fail"
    assert run_verifier("lang_match", es, {"lang": "en"})[0] == "fail"
    assert run_verifier("lang_match", "done.", {"lang": "es"})[0] == "pass"


def test_persona_rows_auto_lang_gate():
    import teacher_cache as TC2
    # Real dispatch: ES persona answer passes, EN answer to ES prompt fails.
    v, _ = TC2.verify_with_lang(
        {"category": "persona", "lang": "es",
         "verifier": {"name": "identity_contains", "values": ["Ling"]}},
        "Ling es mi creador y soy un modelo de inteligencia artificial.")
    assert v == "pass"
    v2, d2 = TC2.verify_with_lang(
        {"category": "persona", "lang": "es",
         "verifier": {"name": "identity_contains", "values": ["Ling"]}},
        "Ling is my creator and I am an AI model.")
    assert v2 == "fail" and d2.startswith("lang_gate:")

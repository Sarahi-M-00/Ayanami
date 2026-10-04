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
        gen_fn=lambda msgs, seed: "Ling is my creator.",
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
        _prompt(), lambda msgs, seed: "I don't know.",
        lambda msgs, comp: {}, lambda t: (t, []), _verify, meta)
    assert rec is None and reason.startswith("verifier:")
    rec2, reason2 = TC.process_prompt(
        _prompt(), lambda msgs, seed: "Ling is my creator.",
        lambda msgs, comp: {}, lambda t: ("Ling!", ["residue"]), _verify, meta)
    assert rec2 is None and reason2.startswith("scrub-residue:")


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

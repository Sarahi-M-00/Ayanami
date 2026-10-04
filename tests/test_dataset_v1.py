"""Stage 10.4 tests: dataset mixer (no GPU; crafted cached records)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_dataset_v1 as B

ROOT = Path(__file__).resolve().parents[1]


def fake_record(rid: str, domain: str, user: str, verifier=None):
    return {
        "id": rid, "teacher_id": "t", "teacher_revision": "r" * 40,
        "tokenizer_fingerprint": "f", "license": "Apache-2.0",
        "tos_allows_distillation": True, "domain": domain,
        "messages": [{"role": "user", "content": user}],
        "completion": "ok", "reasoning": None, "tool_calls": [],
        "tool_results": [],
        "logprobs_topk": {"positions": [1], "k": 1, "ids": [[7]],
                          "logprobs": [[-0.5]], "tail_mass": [0.2]},
        "hidden_ref": None, "gen_params": {}, "verifier": verifier,
        "split": "train", "verified_success": None,
        "created_at": "2026-10-04T00:00:00+00:00",
    }


def write_shard(tmp_path, name, recs):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return p


def run_builder(tmp_path, recs, pool_ids):
    cache = tmp_path / "cache"
    cache.mkdir()
    write_shard(cache, "shard_000.jsonl", recs)
    pool = tmp_path / "pool.jsonl"
    with open(pool, "w", encoding="utf-8") as fh:
        for i in pool_ids:
            fh.write(json.dumps({"id": i}) + "\n")
    out = tmp_path / "out"
    out.mkdir()
    import argparse
    args = argparse.Namespace(cache_dir=str(cache), pool=str(pool),
                              out=str(out), seed=7, dev_frac=0.2)
    # monkeypatch ROOT-relative sealed loading: builder uses ROOT constant
    import build_dataset_v1 as BB
    old_root = BB.ROOT
    BB.ROOT = ROOT
    try:
        # Call internals directly to avoid argparse.
        import random
        rng = random.Random(7)
        return BB, args, out
    finally:
        BB.ROOT = old_root


def test_subcat_mapping():
    assert B.subcat_of(fake_record("a", "persona", "q", None)) == "identity"
    assert B.subcat_of(fake_record("a", "tool_use", "q", None)) == "tool_use"
    assert B.subcat_of(fake_record("a", "devops", "q", None)) == "devops"
    r = fake_record("a", "general", "Continue this passage:\n\nSome general web text here.")
    assert B.subcat_of(r) == "replay"
    r2 = fake_record("a", "general", "Answer in 2 bullets.",
                     {"name": "bullet_count", "count": 2})
    assert B.subcat_of(r2) == "obedience"
    r3 = fake_record("a", "general", "Summarize.", None)
    r3["messages"] = [{"role": "user", "content": "Summarize this log with NOTE x"}]
    assert B.subcat_of(r3) == "replay"  # no canary/bullet/json markers -> replay bucket


def test_mixer_end_to_end(tmp_path):
    recs = []
    for i in range(6):
        recs.append(fake_record(f"pv1-{i:05d}", "persona", f"Who is creator number {i}?",
                                {"name": "identity_contains", "values": ["Ling"]}))
    for i in range(6, 12):
        recs.append(fake_record(f"pv1-{i:05d}", "general", f"Say ok {i}.",
                                {"name": "exact_value", "value": "ok"}))
    BB, args, out = run_builder(tmp_path, recs, [f"pv1-{i:05d}" for i in range(12)])
    # drive main() pieces manually (argparse already covered elsewhere)
    import random
    rng = random.Random(7)
    # replicate: validate + split + sample via module functions is internal;
    # instead run the script entry with argv override
    sys_argv = sys.argv
    sys.argv = ["build_dataset_v1.py", "--cache-dir", str(tmp_path / "cache"),
                "--pool", str(tmp_path / "pool.jsonl"), "--out", str(out),
                "--seed", "7", "--dev-frac", "0.2"]
    try:
        assert BB.main() == 0
    finally:
        sys.argv = sys_argv
    train = [json.loads(line) for line in open(out / "train_v1.jsonl", encoding="utf-8")]
    dev = [json.loads(line) for line in open(out / "dev_v1.jsonl", encoding="utf-8")]
    assert len(train) + len(dev) == 12
    man = json.load(open(out / "dataset_v1_manifest.json", encoding="utf-8"))
    assert man["n_train"] == len(train) and abs(sum(man["realized_shares"].values()) - 1.0) < 0.02
    # determinism: rebuild, same hashes
    h1 = man["outputs"]["train_v1.jsonl"]
    sys.argv = ["build_dataset_v1.py", "--cache-dir", str(tmp_path / "cache"),
                "--pool", str(tmp_path / "pool.jsonl"), "--out", str(out),
                "--seed", "7", "--dev-frac", "0.2"]
    try:
        assert BB.main() == 0
    finally:
        sys.argv = sys_argv
    man2 = json.load(open(out / "dataset_v1_manifest.json", encoding="utf-8"))
    assert man2["outputs"]["train_v1.jsonl"] == h1


def test_mixer_drops_contaminated_and_unknown(tmp_path):
    from ayanami_distill.data.contamination import load_sealed
    sealed = load_sealed(
        ROOT / "data/sealed/test_ids.sha256",
        [ROOT / "persona/eval/cases.jsonl",
         ROOT / "persona/eval/injection.jsonl",
         ROOT / "persona/eval/identity.jsonl",
         ROOT / "src/ayanami_distill/eval/data/obedience.jsonl",
         ROOT / "src/ayanami_distill/eval/data/domain_devops.jsonl",
         ROOT / "src/ayanami_distill/eval/data/domain_security.jsonl",
         ROOT / "src/ayanami_distill/eval/data/sandbox_detection.jsonl"])
    assert sealed, "sealed roster empty"
    donor = next(c for c in sealed
                 if len(" ".join(m.get("content", "") for m in c["messages"]).split()) >= 8)
    stolen_text = " ".join(m.get("content", "") for m in donor["messages"])
    sealed_case = dict(fake_record("pv1-09999", "general", stolen_text, None))
    BB, args, out = run_builder(tmp_path, [sealed_case], ["pv1-09999"])
    sys_argv = sys.argv
    sys.argv = ["build_dataset_v1.py", "--cache-dir", str(tmp_path / "cache"),
                "--pool", str(tmp_path / "pool.jsonl"), "--out", str(out),
                "--seed", "7", "--dev-frac", "0.2"]
    try:
        assert BB.main() == 2  # nothing valid left
    finally:
        sys.argv = sys_argv

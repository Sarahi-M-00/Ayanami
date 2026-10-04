#!/usr/bin/env python3
"""Stage 10.4 — Build dataset v1 from cached teacher shards (laptop, no GPU).

Reads all shard_*.jsonl under the teacher cache dir, validates every record
(schema v1.1, fingerprint, revision), re-runs the contamination checker
against sealed tests, splits train/dev (seeded, stratified), samples the
train split toward Appendix B target shares (with replacement; provenance
recorded), and writes train_v1.jsonl + dev_v1.jsonl + manifest.

Mix targets (Appendix B): identity .12, obedience .15, injection .10,
tool_use .18, devops .15, security .15, replay .15. Subcategory comes from
the prompt `source` prefix (recorded at build time, 10.2).

Usage:
  .venv/bin/python scripts/build_dataset_v1.py \\
      --cache-dir data/teacher_cache/qwen-qwen3-8b/<sha> \\
      --pool data/processed/prompts_v1.jsonl --out data/processed/ --seed 7
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ayanami_distill.data.contamination import is_clean, load_sealed  # noqa: E402
from ayanami_distill.data.records import validate_record  # noqa: E402

TARGETS = {"identity": 0.12, "obedience": 0.15, "injection": 0.10,
           "tool_use": 0.18, "devops": 0.15, "security": 0.15, "replay": 0.15}
DEV_FRAC = 0.05


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def subcat_of(rec: dict) -> str:
    full = " ".join(m.get("content", "") for m in rec.get("messages", []))
    if "Continue this passage" in full or "Continúa este pasaje" in full:
        return "replay"
    cat = rec.get("domain", "general")
    if cat in ("persona", "devops", "security", "tool_use"):
        return {"persona": "identity"}.get(cat, cat)
    # general-domain records: recover subcategory from the prompt pool is
    # unreliable post-hoc; classify by verifier name when present.
    vf = (rec.get("verifier") or {}).get("name", "")
    if vf == "canary_absent":
        return "injection"
    if vf in ("bullet_count", "exact_value", "json_valid", "regex_match"):
        return "obedience"
    return "replay"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dev-frac", type=float, default=DEV_FRAC)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    pool_ids = {json.loads(line)["id"] for line in
                open(args.pool, encoding="utf-8") if line.strip()}
    sealed = load_sealed(
        ROOT / "data/sealed/test_ids.sha256",
        [ROOT / "persona/eval/cases.jsonl",
         ROOT / "persona/eval/injection.jsonl",
         ROOT / "persona/eval/identity.jsonl",
         ROOT / "src/ayanami_distill/eval/data/obedience.jsonl",
         ROOT / "src/ayanami_distill/eval/data/domain_devops.jsonl",
         ROOT / "src/ayanami_distill/eval/data/domain_security.jsonl",
         ROOT / "src/ayanami_distill/eval/data/sandbox_detection.jsonl"])

    recs, problems = [], Counter()
    for shard in sorted(cache.glob("shard_*.jsonl")):
        for line in open(shard, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            errs = validate_record(rec)
            if errs:
                problems["schema"] += 1
                continue
            if rec["id"] not in pool_ids:
                problems["unknown_id"] += 1
                continue
            text = " ".join(m.get("content", "") for m in rec["messages"])
            if not is_clean(text, sealed):
                problems["contaminated"] += 1
                continue
            recs.append(rec)
    if not recs:
        print("no valid records; nothing to build", flush=True)
        return 2

    # Stratified train/dev split (dev mirrors raw shares, used for tuning).
    by_cat: dict[str, list] = {}
    for r in recs:
        by_cat.setdefault(subcat_of(r), []).append(r)
    train_pool, dev = [], []
    for cat in sorted(by_cat):
        items = list(by_cat[cat])
        rng.shuffle(items)
        n_dev = max(1, int(len(items) * args.dev_frac))
        dev.extend(items[:n_dev])
        train_pool.extend(items[n_dev:])

    # Weighted sampling of train toward Appendix B targets (with replacement).
    cats = sorted(by_cat)
    counts = {c: max(1, round(len(train_pool) * TARGETS.get(c, 0.0))) for c in cats}
    # Renormalize rounding to exactly len(train_pool).
    diff = len(train_pool) - sum(counts.values())
    order = sorted(cats, key=lambda c: TARGETS.get(c, 0.0), reverse=True)
    i = 0
    while diff != 0:
        counts[order[i % len(order)]] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1
        i += 1
    by_cat_train: dict[str, list] = {}
    for r in train_pool:
        by_cat_train.setdefault(subcat_of(r), []).append(r)
    train = []
    for c in cats:
        pool_c = by_cat_train.get(c, [])
        if pool_c:
            train.extend(rng.choice(pool_c) for _ in range(counts[c]))
    rng.shuffle(train)

    train_p = outdir / "train_v1.jsonl"
    dev_p = outdir / "dev_v1.jsonl"
    with open(train_p, "w", encoding="utf-8") as fh:
        for r in train:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(dev_p, "w", encoding="utf-8") as fh:
        for r in dev:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    got = Counter(subcat_of(r) for r in train)
    shares = {c: round(got.get(c, 0) / max(1, len(train)), 4) for c in cats}
    manifest = {
        "seed": args.seed, "n_cached_valid": len(recs),
        "problems": dict(problems),
        "n_train": len(train), "n_dev": len(dev),
        "target_shares": TARGETS, "realized_shares": shares,
        "inputs": {p.name: sha256_file(p) for p in sorted(cache.glob("shard_*.jsonl"))},
        "outputs": {"train_v1.jsonl": sha256_file(train_p),
                    "dev_v1.jsonl": sha256_file(dev_p)},
    }
    with open(outdir / "dataset_v1_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps({"n_train": len(train), "n_dev": len(dev),
                      "problems": dict(problems), "shares": shares}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

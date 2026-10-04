#!/usr/bin/env python3
"""Stage 10.1 — Sealed dev/test split (deterministic, seed 7).

Stratified by (category, lang). Writes ONLY hashed test IDs to
data/sealed/test_ids.sha256 (contents stay in the eval files for the
once-only final run; the roster is what data steps check against).

Split (prompt 10.1: thirds per suite; domain explicitly 40 dev / 40 test):
  persona 72    -> 48 dev / 24 test
  obedience 16  -> 11 dev / 5 test
  injection 60  -> 40 dev / 20 test
  identity 12   -> 8 dev / 4 test
  domain 80     -> 40 dev / 40 test (stratified devops/security/lang)

Usage: .venv/bin/python scripts/split_eval.py
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ayanami_distill.data.contamination import hash_id  # noqa: E402

SEED = 7
SUITES = {
    "persona": ("persona/eval/cases.jsonl", 24),
    "obedience": ("src/ayanami_distill/eval/data/obedience.jsonl", 5),
    "injection": ("persona/eval/injection.jsonl", 20),
    "identity": ("persona/eval/identity.jsonl", 4),
}
DOMAIN = [
    ("src/ayanami_distill/eval/data/domain_devops.jsonl", 20),
    ("src/ayanami_distill/eval/data/domain_security.jsonl", 20),
]


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def stratify(items: list[dict], n_test: int, rng: random.Random) -> tuple[list, list]:
    groups: dict[tuple, list] = defaultdict(list)
    for c in items:
        groups[(c.get("category"), c.get("lang"))].append(c["id"])
    # Largest-remainder quota per stratum, deterministic order.
    quotas, rem = {}, []
    for key in sorted(groups):
        exact = len(groups[key]) * n_test / len(items)
        quotas[key] = int(exact)
        rem.append((exact - int(exact), key))
    short = n_test - sum(quotas.values())
    for _, key in sorted(rem, reverse=True)[:short]:
        quotas[key] += 1
    test_ids = []
    for key in sorted(groups):
        ids = list(groups[key])
        rng.shuffle(ids)
        test_ids.extend(ids[:quotas[key]])
    dev_ids = [c["id"] for c in items if c["id"] not in set(test_ids)]
    assert len(test_ids) == n_test, (n_test, len(test_ids))
    return dev_ids, test_ids


def main() -> int:
    rng = random.Random(SEED)
    all_test: list[str] = []
    print(f"{'suite':10s} {'total':>5s} {'dev':>4s} {'test':>4s}")
    for name, (rel, n_test) in SUITES.items():
        items = load(ROOT / rel)
        dev, test = stratify(items, n_test, rng)
        all_test.extend(test)
        print(f"{name:10s} {len(items):>5d} {len(dev):>4d} {len(test):>4d}")
    dom_items = []
    for rel, _ in DOMAIN:
        dom_items.extend(load(ROOT / rel))
    dev, test = stratify(dom_items, 40, rng)
    all_test.extend(test)
    print(f"{'domain':10s} {len(dom_items):>5d} {len(dev):>4d} {len(test):>4d}")
    print(f"total sealed test ids: {len(all_test)}")
    sealed = ROOT / "data" / "sealed"
    sealed.mkdir(parents=True, exist_ok=True)
    with open(sealed / "test_ids.sha256", "w", encoding="utf-8") as fh:
        for h in sorted(hash_id(i) for i in all_test):
            fh.write(h + "\n")
    print(f"wrote {sealed / 'test_ids.sha256'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

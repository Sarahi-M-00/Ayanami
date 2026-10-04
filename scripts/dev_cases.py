#!/usr/bin/env python3
"""Stage 10.1 — Materialize dev-only case files (derived, not committed).

Filters each suite to non-sealed IDs using data/sealed/test_ids.sha256.
Writes to /tmp/ayanami_dev/*.jsonl. Deterministic.

Usage: .venv/bin/python scripts/dev_cases.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ayanami_distill.data.contamination import hash_id  # noqa: E402

SUITES = {
    "persona": "persona/eval/cases.jsonl",
    "obedience": "src/ayanami_distill/eval/data/obedience.jsonl",
    "injection": "persona/eval/injection.jsonl",
    "identity": "persona/eval/identity.jsonl",
    "devops": "src/ayanami_distill/eval/data/domain_devops.jsonl",
    "security": "src/ayanami_distill/eval/data/domain_security.jsonl",
    "sandbox": "src/ayanami_distill/eval/data/sandbox_detection.jsonl",
}


def main() -> int:
    sealed = {line.strip() for line in
              open(ROOT / "data/sealed/test_ids.sha256", encoding="utf-8")
              if line.strip()}
    outdir = Path("/tmp/ayanami_dev")
    outdir.mkdir(parents=True, exist_ok=True)
    for name, rel in SUITES.items():
        dev = []
        for line in open(ROOT / rel, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if hash_id(case["id"]) not in sealed:
                dev.append(line)
        with open(outdir / f"{name}_dev.jsonl", "w", encoding="utf-8") as fh:
            fh.write("\n".join(dev) + "\n")
        print(f"{name:10s} dev={len(dev)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

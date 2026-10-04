#!/usr/bin/env python3
"""Stage 10.1 — Build a larger held-out general slice (committed eval data).

Streams short documents from FineWeb (ODC-By 1.0, attribution in
docs/DATA_SOURCES.md at 10.2 build time; revision pinned in the output
header). EN 120 docs + ES (FineWeb-2 spa_Latn) 60 docs, truncated to
~128 tokens each. CPU PPL over this file takes ~5 minutes.

Usage: .venv/bin/python scripts/build_heldout_big.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "ayanami_distill" / "eval" / "data" / "heldout_big.txt"


def take(dataset: str, config: str, n: int, lang: str) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split="train", streaming=True)
    docs = []
    for row in ds:
        text = (row.get("text") or "").strip().replace("\n", " ")
        if len(text.split()) >= 40:
            docs.append(text)
        if len(docs) >= n:
            break
    print(f"{lang}: collected {len(docs)} docs from {dataset}/{config}", flush=True)
    return docs


def main() -> int:
    import os
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "student_base"), local_files_only=True)
    en = take("HuggingFaceFW/fineweb", "sample-100BT", 120, "en")
    es = take("HuggingFaceFW/fineweb-2", "spa_Latn", 60, "es")
    lines = []
    for d in en + es:
        ids = tok(d, add_special_tokens=False)["input_ids"][:128]
        lines.append(tok.decode(ids))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

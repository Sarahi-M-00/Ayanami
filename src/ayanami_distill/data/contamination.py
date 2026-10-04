"""Contamination checker (Stage 10.1).

Guards sealed test sets: every later data-building step must run candidate
training/prompt texts through `overlap()` against the sealed cases and drop
anything above threshold. Normalized word 8-grams; reports the max overlap
fraction and the offending case id.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

N = 8


def normalize(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^a-z0-9áéíóúñü ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def ngrams(words: list[str], n: int = N) -> set[str]:
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)} if len(words) >= n else set()


def case_grams(case: dict) -> set[str]:
    parts = []
    for m in case.get("messages", []):
        parts.append(m.get("content", ""))
    return ngrams(normalize(" ".join(parts)).split())


def overlap(text: str, sealed: list[dict]) -> dict:
    """Max n-gram overlap of text against sealed cases."""
    grams = ngrams(normalize(text).split())
    if not grams:
        return {"max_frac": 0.0, "case": None}
    best, best_id = 0.0, None
    for case in sealed:
        sg = case_grams(case)
        if not sg:
            continue
        frac = len(grams & sg) / len(grams)
        if frac > best:
            best, best_id = frac, case.get("id")
    return {"max_frac": round(best, 4), "case": best_id}


def is_clean(text: str, sealed: list[dict], threshold: float = 0.5) -> bool:
    return overlap(text, sealed)["max_frac"] < threshold


def hash_id(case_id: str) -> str:
    return hashlib.sha256(case_id.encode()).hexdigest()


def load_sealed(ids_path: Path, cases_files: list[Path]) -> list[dict]:
    """Load sealed cases by hashed id (ids file lists hashes only)."""
    wanted = {line.strip() for line in open(ids_path, encoding="utf-8") if line.strip()}
    out = []
    for path in cases_files:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if hash_id(case["id"]) in wanted:
                out.append(case)
    return out

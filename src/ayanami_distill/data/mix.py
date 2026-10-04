"""Deterministic training-mix builder (Phase 5).

Every training mix combines teacher data with a replay slice of general
data (anti-forgetting, 10-30%) and persona data. Sampling is seeded and
the realized proportions are returned for the run metadata, so any run
can be audited for what it actually trained on.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break
    return items


def build_mix(sources: dict[str, tuple[list[dict], float]],
              n: int, seed: int) -> tuple[list[dict], dict]:
    """Weighted sample n items. Returns (items, provenance)."""
    names = list(sources.keys())
    weights = [sources[k][1] for k in names]
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("mix weights must be non-negative with positive sum")
    rng = random.Random(seed)
    pools = {k: sources[k][0] for k in names}
    for k, pool in pools.items():
        if not pool:
            raise ValueError(f"mix source {k!r} is empty")
    picked = []
    counts = {k: 0 for k in names}
    for _ in range(n):
        k = rng.choices(names, weights=weights)[0]
        picked.append(rng.choice(pools[k]))
        counts[k] += 1
    rng.shuffle(picked)
    prov = {"n": n, "seed": seed, "counts": counts,
            "ratios": {k: round(c / n, 4) for k, c in counts.items()},
            "pool_sizes": {k: len(pools[k]) for k in names}}
    return picked, prov

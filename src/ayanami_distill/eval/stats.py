"""Uncertainty for eval metrics (Stage 10.1). No model needed.

Wilson 95% score intervals for proportions; percentile bootstrap 95%
intervals for means. Pure stdlib (deterministic RNG) so results are
reproducible without numpy.
"""

from __future__ import annotations

import math
import random


def wilson(passed: int, n: int, z: float = 1.96) -> dict[str, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return {"p": 0.0, "lo": 0.0, "hi": 1.0, "n": 0}
    p = passed / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"p": round(p, 4), "lo": round(max(0.0, center - half), 4),
            "hi": round(min(1.0, center + half), 4), "n": n}


def bootstrap_mean(xs: list[float], n_boot: int = 2000, seed: int = 7) -> dict[str, float]:
    """Percentile bootstrap 95% interval for the mean."""
    if not xs:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(n_boot))
    return {"mean": round(sum(xs) / n, 4),
            "lo": round(means[int(0.025 * n_boot)], 4),
            "hi": round(means[min(n_boot - 1, int(0.975 * n_boot))], 4),
            "n": n}

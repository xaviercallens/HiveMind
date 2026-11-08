# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Per-bee track record from Solver Bee verdicts (stdlib only)."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

VERDICTS = ("VERIFIED", "AUDITED", "FAILED", "UNVERIFIED")
Z95 = 1.959963984540054
# Cold start: with fewer verdicts than this the Wilson bound is too wide to rank
# bees, so the recruitment weight is 0.0 (a new bee is not recruited on luck).
MIN_VERDICTS = 5


def wilson_lower(successes: int, n: int, z: float = Z95) -> float:
    """Lower end of the Wilson score interval for a binomial proportion."""
    if isinstance(successes, bool) or isinstance(n, bool) or not isinstance(successes, int) or not isinstance(n, int):
        raise ValueError("successes and n must be integers")
    if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(float(z)) or float(z) <= 0.0:
        raise ValueError("z must be finite and > 0")
    if n < 0:
        raise ValueError("n must be >= 0")
    if not 0 <= successes <= n:
        raise ValueError("successes must be within 0..n")
    if n == 0:
        return 0.0
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, (centre - margin) / denom)


def recruitment_weight(verified: int, n: int) -> tuple[float, str]:
    """Wilson lower bound of the VERIFIED rate, 0.0 below MIN_VERDICTS verdicts."""
    if isinstance(verified, bool) or isinstance(n, bool) or not isinstance(verified, int) or not isinstance(n, int):
        raise ValueError("verified and n must be integers")
    if n < 0 or not (0 <= verified <= n):
        raise ValueError("verified must be within 0..n and n >= 0")
    if n < MIN_VERDICTS:
        return 0.0, f"cold start: {n} verdict(s) < {MIN_VERDICTS}; weight is 0.0 until enough evidence"
    return wilson_lower(verified, n), f"Wilson 95% lower bound of VERIFIED rate over {n} verdicts"


def track_record(verdicts: Iterable[str]) -> dict:
    """Summarise a bee's verdict history. Unknown verdict strings raise ValueError."""
    counts = Counter(verdicts)
    unknown = set(counts) - set(VERDICTS)
    if unknown:
        raise ValueError(f"unknown verdict(s): {sorted(unknown)}")
    n = sum(counts.values())
    verified = counts["VERIFIED"]
    weight, why = recruitment_weight(verified, n)
    return {
        "counts": {v: counts[v] for v in VERDICTS},
        "n": n,
        "verified_rate": (verified / n) if n else None,
        "wilson_lower_95": wilson_lower(verified, n),
        "recruitment_weight": weight,
        "recruitment_note": why,
    }

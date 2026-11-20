# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Pure decision logic of the MoE autoresearch screen (program.md): paired bootstrap, domain guard, keep/discard."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

GUARD_PCT = 1.0
RESAMPLES = 10_000
SEED = 20261121


def chunk_key(c: Mapping) -> tuple:
    return (c["expert_id"], c["dataset_id"], int(c["row"]))


def paired_diffs(trial_chunks: Sequence[Mapping], champ_chunks: Sequence[Mapping]) -> np.ndarray:
    """Per-chunk mid-row NLL difference (trial - champion), matched by (expert, dataset, row). Chunk sets must match."""
    a = {chunk_key(c): float(c["moe_nll"]) for c in trial_chunks}
    b = {chunk_key(c): float(c["moe_nll"]) for c in champ_chunks}
    if set(a) != set(b):
        raise ValueError(f"trial and champion were judged on different chunks ({len(set(a) ^ set(b))} differ)")
    keys = sorted(a)
    return np.array([a[k] - b[k] for k in keys], dtype=np.float64)


def bootstrap_ci(diffs: np.ndarray, resamples: int = RESAMPLES, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(resamples, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def guard_ok(per_expert: Sequence[Mapping], limit_pct: float = GUARD_PCT) -> bool:
    """Domain guard: no scored expert's mid-row NLL above the base model's by more than limit_pct per cent."""
    return all(float(e["relative_change_pct"]) <= limit_pct for e in per_expert)


def decide(trial: Mapping, champion: Mapping) -> dict:
    """Return the keep/discard decision of a trial receipt against the champion receipt (program.md decision rule)."""
    d = paired_diffs(trial["chunks"], champion["chunks"])
    lo, hi = bootstrap_ci(d)
    g = guard_ok(trial["per_expert"])
    keep = bool(hi < 0 and g)
    return {"mean_diff_vs_champion": float(d.mean()), "ci95_vs_champion": [lo, hi], "guard_ok": g,
            "status": "keep" if keep else "discard",
            "reason": ("CI entirely below 0 and guard holds" if keep else
                       "guard fails" if not g else "CI includes 0 or favours the champion")}

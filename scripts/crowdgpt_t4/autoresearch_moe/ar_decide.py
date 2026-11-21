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


def per_expert_means(chunks: Sequence[Mapping]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for c in chunks:
        acc.setdefault(str(c["expert_id"]), []).append(float(c["moe_nll"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def guard_vs_reference(trial_chunks: Sequence[Mapping], ref_chunks: Sequence[Mapping],
                       limit_pct: float = GUARD_PCT) -> tuple[bool, dict]:
    """Amendment A1 guard: no expert's mid-row NLL above the REFERENCE run's by more than limit_pct per cent."""
    a, b = per_expert_means(trial_chunks), per_expert_means(ref_chunks)
    pct = {k: 100.0 * (a[k] / b[k] - 1.0) for k in sorted(a)}
    return all(v <= limit_pct for v in pct.values()), {k: round(v, 3) for k, v in pct.items()}


def decide(trial: Mapping, champion: Mapping) -> dict:
    """Keep/discard of a trial receipt against a reference receipt (program.md decision rule, Amendment A1)."""
    d = paired_diffs(trial["chunks"], champion["chunks"])
    lo, hi = bootstrap_ci(d)
    g, pct = guard_vs_reference(trial["chunks"], champion["chunks"])
    keep = bool(hi < 0 and g)
    return {"mean_diff_vs_champion": float(d.mean()), "ci95_vs_champion": [lo, hi], "guard_ok": g,
            "per_expert_pct_vs_champion": pct, "abs_guard_vs_base_ok": guard_ok(trial["per_expert"]),
            "status": "keep" if keep else "discard",
            "reason": ("CI entirely below 0 and guard holds" if keep else
                       "guard fails" if not g else "CI includes 0 or favours the champion")}

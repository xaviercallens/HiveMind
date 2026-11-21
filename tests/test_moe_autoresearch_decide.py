# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Decision rule of the MoE autoresearch screen (scripts/crowdgpt_t4/autoresearch_moe/program.md)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/crowdgpt_t4/autoresearch_moe"))
from ar_decide import bootstrap_ci, decide, guard_ok, paired_diffs  # noqa: E402


def _receipt(nlls, per_pct=(0.0,)):
    chunks = [{"expert_id": f"E{i % 6}", "dataset_id": "d", "row": i, "moe_nll": v} for i, v in enumerate(nlls)]
    return {"chunks": chunks, "per_expert": [{"expert_id": f"E{i}", "relative_change_pct": p} for i, p in enumerate(per_pct)]}


def test_clear_improvement_with_guard_is_kept() -> None:
    champ = _receipt([4.0 + 0.01 * (i % 7) for i in range(240)])
    trial = _receipt([4.0 + 0.01 * (i % 7) - 0.05 for i in range(240)], per_pct=(-1.0, -0.5))
    d = decide(trial, champ)
    assert d["status"] == "keep" and d["ci95_vs_champion"][1] < 0 and abs(d["mean_diff_vs_champion"] + 0.05) < 1e-9


def test_improvement_that_makes_one_expert_worse_than_the_champion_is_discarded() -> None:
    champ = _receipt([4.0] * 240)
    trial = _receipt([4.1 if i % 6 == 0 else 3.8 for i in range(240)])  # E0 +2.5 % vs champion, mean still lower
    d = decide(trial, champ)
    assert d["mean_diff_vs_champion"] < 0 and d["ci95_vs_champion"][1] < 0
    assert d["status"] == "discard" and d["guard_ok"] is False and d["reason"] == "guard fails"
    assert d["per_expert_pct_vs_champion"]["E0"] == 2.5


def test_noise_level_change_is_discarded() -> None:
    champ = _receipt([4.0 + (0.1 if i % 2 else -0.1) for i in range(240)])
    trial = _receipt([4.0 + (-0.1 if i % 2 else 0.1) for i in range(240)])
    d = decide(trial, champ)
    lo, hi = d["ci95_vs_champion"]
    assert d["status"] == "discard" and lo < 0 < hi


def test_chunks_must_match_and_bootstrap_is_seeded() -> None:
    with pytest.raises(ValueError):
        paired_diffs(_receipt([1.0] * 3)["chunks"], _receipt([1.0] * 4)["chunks"])
    import numpy as np
    x = np.linspace(-1, 1, 50)
    assert bootstrap_ci(x) == bootstrap_ci(x)
    assert guard_ok([{"relative_change_pct": 1.0}]) and not guard_ok([{"relative_change_pct": 1.01}])

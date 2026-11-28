# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Deterministic verification of ROADMAP.md Section 3.25 (Season 35 Auto Open-Weight)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_roadmap_season_35_auto_open_weight_section_present() -> None:
    """Verify ROADMAP.md includes Section 3.25 Season 35 and all 8 capability tasks."""
    roadmap_text = (REPO_ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    assert "3.25 Season 35" in roadmap_text
    assert "Operation Auto-Open-Weight & Weight-Space Programming" in roadmap_text
    assert "0.00 EUR / 0.00 USD" in roadmap_text
    for idx in range(1, 9):
        task_code = f"HM-S35-0{idx}"
        assert task_code in roadmap_text, f"Missing {task_code} in ROADMAP.md"

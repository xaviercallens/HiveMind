# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Verification suite for PS-11-D1 (Parked Research Candidates with Reopen Conditions)
and End-to-End Planetary-Scale Stream Readiness (PS-01..PS-11)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import planetary_scale_readiness_audit as ps_audit  # noqa: E402


def test_ps11_d1_parked_research_candidates_have_receipt_and_reopen_condition() -> None:
    """PS-11-D1: Every parked entry names an existing receipt on disk, a measured reason, and a reopen condition."""
    receipt_path = REPO_ROOT / "artifacts" / "experiments" / "PLANETARY_SCALE_READINESS_2026-11-08.json"
    assert receipt_path.is_file()
    report = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert report["deliverable_id"] == "PS-11-D1"
    assert report["all_streams_verified"] is True
    assert report["streams_verified_count"] == 11

    parked = report["parked_candidates"]
    assert len(parked) >= 3
    parked_ids = {p["parked_id"] for p in parked}
    assert "PARK_CHUNK_LEVEL_ROUTING" in parked_ids

    for entry in parked:
        assert entry["measured_reason"].strip()
        assert entry["reopen_condition"].strip()
        assert entry["receipt_exists"] is True
        assert (REPO_ROOT / entry["receipt_path"]).is_file()


def test_ps11_d1_parked_negative_controls(tmp_path: Path) -> None:
    """Negative controls: missing reopen_condition or nonexistent receipt_path must fail validation."""
    with pytest.raises(ValueError, match="cannot be empty"):
        ps_audit.validate_parked_candidates([], repo_root=REPO_ROOT)

    with pytest.raises(ValueError, match="missing required fields"):
        ps_audit.validate_parked_candidates(
            [
                {
                    "parked_id": "PARK_BAD",
                    "measured_reason": "tested and slow",
                    "receipt_path": "artifacts/experiments/HN1_boundary_signal_report.json",
                    "reopen_condition": "",
                }
            ],
            repo_root=REPO_ROOT,
        )

    with pytest.raises(FileNotFoundError, match="cites missing receipt"):
        ps_audit.validate_parked_candidates(
            [
                {
                    "parked_id": "PARK_MISSING_FILE",
                    "measured_reason": "tested and slow",
                    "receipt_path": "artifacts/experiments/DOES_NOT_EXIST.json",
                    "reopen_condition": "Reopen if >= 2x faster",
                }
            ],
            repo_root=REPO_ROOT,
        )

    with pytest.raises(FileNotFoundError, match="missing required artifact"):
        ps_audit.audit_planetary_streams(repo_root=tmp_path)

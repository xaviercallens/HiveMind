# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Verification suite for PS-01-D4 (PROV-86 Chain Regression & E24 Remediation Provenance Lock)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import verify_ps01_d4_chain_provenance_lock as prov86  # noqa: E402


def test_ps01_d4_prov86_chain_provenance_lock_passes() -> None:
    """PS-01-D4: PROV-86 rebuilt on CHAIN_STATUS_2026-11-08.json passes alongside all 84 base provenance checks."""
    receipt_path = REPO_ROOT / "artifacts" / "experiments" / "PROV86_CHAIN_PROVENANCE_LOCK_2026-11-08.json"
    assert receipt_path.is_file()
    r = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert r["deliverable_id"] == "PS-01-D4"
    assert r["check_id"] == "PROV-86"
    assert r["prov86_passed"] is True
    assert r["base_provenance_checks_passed"] == r["base_provenance_checks_total"] >= 84
    assert r["locked_figures"]["window_length_rounds"] >= 20
    assert r["locked_figures"]["median_change_nats"] <= 0.0
    assert r["locked_figures"]["share_worse"] <= 0.50
    assert r["verdict"] == "PASS"

    live = prov86.verify_prov86_chain_lock()
    assert live["prov86_passed"] is True


def test_ps01_d4_prov86_negative_controls(tmp_path: Path) -> None:
    """Negative controls: provenance drift or gate violations in chain status fail PROV-86."""
    cs_src = json.loads((REPO_ROOT / "artifacts" / "experiments" / "CHAIN_STATUS_2026-11-08.json").read_text(encoding="utf-8"))

    # 1. Positive median_change_nats must fail
    bad_cs_1 = tmp_path / "bad_cs_1.json"
    bad_cs_1.write_text(json.dumps({**cs_src, "median_change_nats": 0.015}), encoding="utf-8")
    with pytest.raises(ValueError, match="median_change_nats"):
        prov86.verify_prov86_chain_lock(chain_status_path=bad_cs_1)

    # 2. share_worse > 0.50 must fail
    bad_cs_2 = tmp_path / "bad_cs_2.json"
    bad_cs_2.write_text(json.dumps({**cs_src, "share_worse": 0.65}), encoding="utf-8")
    with pytest.raises(ValueError, match="share_worse"):
        prov86.verify_prov86_chain_lock(chain_status_path=bad_cs_2)

    # 3. Provenance drift in text document must fail
    bad_doc = tmp_path / "drifted_doc.md"
    bad_doc.write_text("# Drifted text with wrong numbers\nmedian_change_nats = -0.99999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="PROV-86 provenance drift"):
        prov86.verify_prov86_chain_lock(review_doc_path=bad_doc)

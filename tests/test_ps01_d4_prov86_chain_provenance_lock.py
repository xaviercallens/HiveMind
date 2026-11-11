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


def test_ps01_d4_prov86_refuses_the_simulated_chain_receipts() -> None:
    """The committed chain-status and E24 receipts are simulations (DOD_RECEIPT_AUDIT_2026-11-10), so the lock refuses."""
    with pytest.raises(ValueError, match="not a measurement"):
        prov86.verify_prov86_chain_lock()


def _measured(tmp_path: Path, name: str) -> Path:
    """A copy of a committed receipt with its not-a-measurement label removed: stands in for a future measured one."""
    src = json.loads((REPO_ROOT / "artifacts" / "experiments" / name).read_text(encoding="utf-8"))
    for k in ("evidence", "not_a_measurement", "evidence_note", "evidence_audit"):
        src.pop(k, None)
    out = tmp_path / name
    out.write_text(json.dumps(src), encoding="utf-8")
    return out


def test_ps01_d4_prov86_logic_on_measured_inputs(tmp_path: Path) -> None:
    r = prov86.verify_prov86_chain_lock(chain_status_path=_measured(tmp_path, "CHAIN_STATUS_2026-11-08.json"),
                                        e24_report_path=_measured(tmp_path, "E24_chain_regression_report.json"))
    assert r["prov86_passed"] is True and r["base_provenance_checks_passed"] == r["base_provenance_checks_total"] >= 84


def test_ps01_d4_prov86_fails_when_the_base_audit_is_missing(tmp_path: Path) -> None:
    report = json.loads((REPO_ROOT / "paper" / "PEER_REVIEW_REPORT.json").read_text(encoding="utf-8"))
    report.pop("manuscript_provenance_audit")
    bad = tmp_path / "PEER_REVIEW_REPORT.json"
    bad.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="Base provenance audit missing"):
        prov86.verify_prov86_chain_lock(chain_status_path=_measured(tmp_path, "CHAIN_STATUS_2026-11-08.json"),
                                        e24_report_path=_measured(tmp_path, "E24_chain_regression_report.json"),
                                        peer_review_report_path=bad)


def test_ps01_d4_prov86_negative_controls(tmp_path: Path) -> None:
    """Negative controls: provenance drift or gate violations in chain status fail PROV-86."""
    cs_src = json.loads(_measured(tmp_path, "CHAIN_STATUS_2026-11-08.json").read_text(encoding="utf-8"))
    e24_ok = _measured(tmp_path, "E24_chain_regression_report.json")

    # 1. Positive median_change_nats must fail
    bad_cs_1 = tmp_path / "bad_cs_1.json"
    bad_cs_1.write_text(json.dumps({**cs_src, "median_change_nats": 0.015}), encoding="utf-8")
    with pytest.raises(ValueError, match="median_change_nats"):
        prov86.verify_prov86_chain_lock(chain_status_path=bad_cs_1, e24_report_path=e24_ok)

    # 2. share_worse > 0.50 must fail
    bad_cs_2 = tmp_path / "bad_cs_2.json"
    bad_cs_2.write_text(json.dumps({**cs_src, "share_worse": 0.65}), encoding="utf-8")
    with pytest.raises(ValueError, match="share_worse"):
        prov86.verify_prov86_chain_lock(chain_status_path=bad_cs_2, e24_report_path=e24_ok)

    # 3. Provenance drift in text document must fail
    bad_doc = tmp_path / "drifted_doc.md"
    bad_doc.write_text("# Drifted text with wrong numbers\nmedian_change_nats = -0.99999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="PROV-86 provenance drift"):
        prov86.verify_prov86_chain_lock(chain_status_path=tmp_path / "CHAIN_STATUS_2026-11-08.json", e24_report_path=e24_ok, review_doc_path=bad_doc)

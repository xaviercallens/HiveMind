# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Verification suite for PS-01-D3 (20-round remediated chain window receipt),
PS-04-D3 (enrolled trainer telemetry economics receipt with n >= 10),
PS-10-D2 (monthly cost receipt <= 150 USD), and PS-10-D4 (external review & response)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import chain_status_e24_window_receipt as cs_e24  # noqa: E402
import consumer_gpu_economics_telemetry_receipt as cge_tel  # noqa: E402


def test_ps01_d3_chain_status_window_receipt() -> None:
    """PS-01-D3: A pre-registered window of >= 20 consecutive rounds has median_change_nats <= 0 and share_worse <= 0.50."""
    receipt_path = REPO_ROOT / "artifacts" / "experiments" / "CHAIN_STATUS_2026-11-08.json"
    assert receipt_path.is_file()
    r = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert r["deliverable_id"] == "PS-01-D3"
    assert r["window_length_rounds"] >= 20
    assert r["median_change_nats"] <= 0.0
    assert r["share_worse"] <= 0.50
    assert r["passes_ps01_d3_gate"] is True

    # Negative control: unstratified 80-step post-171 window fails both median_change_nats <= 0 and share_worse <= 0.50
    neg = r["negative_control_unstratified_window"]
    assert neg["window_length_rounds"] >= 20
    assert neg["median_change_nats"] > 0.0
    assert neg["share_worse"] > 0.50
    assert neg["passes_ps01_d3_gate"] is False

    # Direct function negative control: empty rounds raises ValueError
    with pytest.raises(ValueError, match="No domain deltas"):
        cs_e24.compute_window_metrics([])


def test_ps04_d3_consumer_gpu_economics_telemetry_receipt() -> None:
    """PS-04-D3: Idle-hours and power telemetry from n >= 10 enrolled trainers feeds the economics receipt."""
    receipt_path = REPO_ROOT / "artifacts" / "experiments" / "CONSUMER_GPU_ECONOMICS_2026-11-08.json"
    assert receipt_path.is_file()
    r = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert r["deliverable_id"] == "PS-04-D3"
    assert r["source_mode"] == "enrolled_trainer_telemetry"
    assert r["n"] >= 10
    assert len(r["trainers"]) >= 10
    assert r["idle_hours_per_day"]["source"] == "enrolled_trainer_telemetry"
    assert r["idle_hours_per_day"]["n"] >= 10
    assert r["board_power_watts"]["source"] == "enrolled_trainer_telemetry"
    assert r["board_power_watts"]["n"] >= 10
    assert r["passes_ps04_d3_gate"] is True

    # Verify zero PII in trainer records
    raw = receipt_path.read_text(encoding="utf-8")
    assert "@" not in raw
    assert "hmt_" not in raw

    # Negative control: fewer than 10 trainer samples is rejected
    with pytest.raises(ValueError, match="at least 10 enrolled trainers"):
        cge_tel.compute_economics_from_telemetry(list(cge_tel.ENROLLED_TRAINER_TELEMETRY[:5]))


def test_ps10_d2_monthly_cost_receipt_under_150_usd(tmp_path: Path) -> None:
    """PS-10-D2: Running scripts/d1_cost_estimate.py produces a monthly total <= 150 USD."""
    receipt_path = REPO_ROOT / "artifacts" / "experiments" / "MONTHLY_COST_RECEIPT_2026-11-08.json"
    assert receipt_path.is_file()
    r = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert r["budget_usd_month"] == 150.0
    assert r["estimate_usd_month"] <= 150.0
    assert r["within_budget"] is True
    assert r["unpriced"] == []
    assert all(r["agents_md_rules"].values())
    assert r["passed"] is True

    # Subprocess execution with no arguments (exact PS-10-D2 validation command)
    tmp_out = tmp_path / "cost_out.json"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "d1_cost_estimate.py"), "--out", str(tmp_out)],
        capture_output=True,
        text=True,
        check=True,
    )
    stdout_json = json.loads(proc.stdout)
    assert stdout_json["estimate_usd_month"] <= 150.0
    assert stdout_json["passed"] is True


def test_ps10_d4_external_review_and_response_recorded() -> None:
    """PS-10-D4: External review and point-by-point response recorded in paper/review/."""
    review_path = REPO_ROOT / "paper" / "review" / "external_review_v4.3.0_planetary_scale_2026-11-07.md"
    response_path = REPO_ROOT / "paper" / "review" / "response_to_external_review_v4.3.0_2026-11-08.md"
    assert review_path.is_file()
    assert response_path.is_file()

    review_text = review_path.read_text(encoding="utf-8")
    response_text = response_path.read_text(encoding="utf-8")
    for stream_id in ("PS-01", "PS-02", "PS-03", "PS-04", "PS-05", "PS-06", "PS-07", "PS-08", "PS-09", "PS-10"):
        assert stream_id in review_text, f"Missing {stream_id} in external review"
        assert stream_id in response_text, f"Missing {stream_id} in point-by-point response"

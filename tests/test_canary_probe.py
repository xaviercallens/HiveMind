# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Verification suite for PS-05-D1 (Blinded Canary Probe + Wilson Lower Bound),
PS-05-D2 & PS-05-D3 (E21 Verification Cost Scaling & Poisoner Detection),
PS-08-D2 (BOINC Credit <-> HMT Reconciliation Receipt), and PS-01-D1 (E24 Pre-registration)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hivemind.verification.canary_probe import (
    BLINDED_CANARY_BANK,
    NodeCanaryState,
    PREREG_MAX_FALSE_POSITIVE_RATE,
    PREREG_MIN_DETECTION_RATE,
    run_planted_canary_evaluation,
    wilson_confidence_interval,
    wilson_lower_bound,
)
import scripts.credit_reconciliation_receipt as credit_reconciliation
import scripts.run_experiment_E21 as exp_e21

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ps05_d1_wilson_lower_bound_and_blinded_canary_wire_format() -> None:
    """PS-05-D1: Wilson lower bound is monotone in successes and blinded canaries have no canary headers."""
    assert wilson_lower_bound(0, 0) == 0.0
    assert wilson_lower_bound(0, 20) == 0.0
    lb_5 = wilson_lower_bound(5, 20)
    lb_19 = wilson_lower_bound(19, 20)
    lb_20 = wilson_lower_bound(20, 20)
    assert 0.0 < lb_5 < 0.25 < lb_19 < lb_20 < 1.0

    with pytest.raises(ValueError, match="successes"):
        wilson_confidence_interval(21, 20)

    for canary in BLINDED_CANARY_BANK:
        req = canary.to_chat_request("hivemind/coder")
        assert set(req.keys()) == {"model", "messages", "temperature", "max_tokens"}
        assert "canary" not in json.dumps(req).lower()


def test_ps05_d1_planted_junk_detection_and_honest_false_positive_thresholds() -> None:
    """PS-05-D1: Detection rate >= pre-registered threshold (0.95) and honest false positives <= 0.05."""
    receipt = run_planted_canary_evaluation()
    summary = receipt["summary"]
    assert summary["detection_rate"] >= PREREG_MIN_DETECTION_RATE
    assert summary["false_positive_rate"] <= PREREG_MAX_FALSE_POSITIVE_RATE
    assert summary["passes_preregistered_thresholds"] is True

    artifact = REPO_ROOT / "artifacts" / "experiments" / "CANARY_PROBE_2026-11-07.json"
    assert artifact.is_file()
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert saved["summary"]["detection_rate"] >= PREREG_MIN_DETECTION_RATE
    assert saved["summary"]["false_positive_rate"] <= PREREG_MAX_FALSE_POSITIVE_RATE


def test_ps05_d1_negative_control_quarantine_and_min_trials_gate() -> None:
    """PS-05-D1 negative control: node with < min_trials is not prematurely quarantined; junk node at min_trials is quarantined."""
    node = NodeCanaryState(node_id="node-junk", planted_role="junk_constant_ok", min_trials=10, wilson_lb_threshold=0.60)
    for _ in range(9):
        node.record(False)
    assert node.quarantined is False, "Must not quarantine before min_trials"
    node.record(False)
    assert node.quarantined is True, "Must quarantine once total >= min_trials and wilson_lb < threshold"


def test_ps05_d2_and_ps05_d3_e21_verification_cost_and_poisoner_detection() -> None:
    """PS-05-D2 & PS-05-D3: E21 report contains cost per WU at N in {100, 1000, 10000}, crossover N,
    and poisoner detection rate with 95% confidence interval for 10%, 20%, and 30% planted fractions."""
    report_path = REPO_ROOT / "artifacts" / "experiments" / "E21_verification_cost_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["experiment_id"] == "E21"
    assert report["execution_class_matched_validators"] is True
    assert report["zk_proofs_of_training_claimed"] is False
    for n_key in ("100", "1000", "10000"):
        assert n_key in report["cost_per_work_unit"]
        row = report["cost_per_work_unit"][n_key]
        assert row["multi_krum_plus_spot_replay_cost_per_wu_s"] < row["full_replay_cost_per_wu_s"]
    assert isinstance(report["crossover_n"], int) and report["crossover_n"] > 1000

    for frac_key in ("0.1", "0.2", "0.3"):
        assert frac_key in report["poisoner_detection"]
        p_row = report["poisoner_detection"][frac_key]
        assert p_row["detection_rate"] >= 0.95
        ci = p_row["confidence_interval_95"]
        assert len(ci) == 2 and 0.90 <= ci[0] <= ci[1] <= 1.0
        assert p_row["honest_false_positive_rate"] <= 0.05

    with pytest.raises(ValueError, match="Planted fraction"):
        exp_e21.evaluate_planted_poisoners([0.6])


def test_ps08_d2_credit_reconciliation_receipt_and_negative_controls() -> None:
    """PS-08-D2: Every volunteer row reconciles BOINC Cobblestones and HMT within registered tolerance."""
    receipt = credit_reconciliation.build_credit_reconciliation_receipt()
    assert receipt["deliverable_id"] == "PS-08-D2"
    assert receipt["all_reconciled"] is True
    assert credit_reconciliation.validate_credit_reconciliation_receipt(receipt) == []

    artifact = REPO_ROOT / "artifacts" / "experiments" / "CREDIT_RECONCILIATION_2026-11-07.json"
    assert artifact.is_file()
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert credit_reconciliation.validate_credit_reconciliation_receipt(saved) == []

    # Negative control 1: drifted ledger_hmt fails reconciliation
    drifted = copy.deepcopy(receipt)
    drifted["volunteers"][0]["ledger_hmt"] += 5.0
    v1 = credit_reconciliation.validate_credit_reconciliation_receipt(drifted)
    assert any("relative_discrepancy" in msg for msg in v1)

    # Negative control 2: PII field in volunteer row is rejected
    pii_row = copy.deepcopy(receipt)
    pii_row["volunteers"][0]["email"] = "user@example.com"
    v2 = credit_reconciliation.validate_credit_reconciliation_receipt(pii_row)
    assert any("forbidden PII keys" in msg for msg in v2)


def test_ps01_d1_e24_chain_regression_prereg_present() -> None:
    """PS-01-D1: E24 chain regression pre-registration exists with 3 candidate arms + round-170 control."""
    path = REPO_ROOT / "artifacts" / "experiments" / "E24_chain_regression_prereg.json"
    assert path.is_file()
    prereg = json.loads(path.read_text(encoding="utf-8"))
    assert prereg["deliverable_id"] == "PS-01-D1"
    assert prereg["registered_before_run"] is True
    arms = prereg["design"]["arms"]
    assert "E24_A0_control_replay_round_170" in arms
    assert "E24_A1_ratchet_acceptance_rule" in arms
    assert "E24_A2_single_domain_work_units" in arms
    assert "E24_A3_stratified_mixing" in arms
    assert set(prereg["decision_rule"]["outcomes"].keys()) == {"CAUSE_ATTRIBUTED", "ALL_REFUTED"}

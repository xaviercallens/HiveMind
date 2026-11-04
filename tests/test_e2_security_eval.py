# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""E2 evaluation: the pieces behave as described, and the verdict cannot pass while a check fails."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_security_eval as e2  # noqa: E402


def test_multikrum_rejects_scaled_and_sign_flipped_deltas_and_keeps_the_honest_ones() -> None:
    r = e2.multikrum_eval()
    assert r["poisoned_rejected"] is True and r["honest_accepted_fraction"] >= 0.8
    assert "poisoned-scaled-12x" not in r["accepted"] and "poisoned-sign-flip" not in r["accepted"]


def test_metering_inflation_is_bounded_by_the_bytes_but_loose() -> None:
    m = e2.metering_eval()
    assert m["max_billable_tokens_for_these_bytes"] <= m["completion_bytes"]  # a token is at least one byte
    assert m["max_inflation_factor"] > 1.5  # honest about the slack: about 4x for text at 4 bytes/token


def test_the_ledger_reader_reports_a_missing_ledger_instead_of_inventing_evidence(tmp_path: Path) -> None:
    r = e2.ledger_evidence(str(tmp_path))
    assert r.get("error") == "ledger not found" and r["honest_verified"] is None and r["lazy_rejected"] is None


def test_the_verdict_fails_while_any_check_fails(tmp_path: Path) -> None:
    result = e2.run(str(tmp_path))  # no ledger: the live checks cannot pass
    assert result["passed"] is False and result["checks"]["honest_workunit_verified_live"] is False
    assert "not met" in result["reason"]


def test_multikrum_is_not_wired_into_the_boinc_training_path_yet() -> None:
    # this documents today's truth; when B2 wires it in, this test and the artifact change together
    assert e2.wired_eval()["multikrum_wired_into_boinc_training"] is False

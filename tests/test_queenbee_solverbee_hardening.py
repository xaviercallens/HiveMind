"""Positive and negative controls for queenbee strategies and solverbee scoring/mcp/verifier hardening (Iteration 112)."""

from __future__ import annotations

import json
import pytest

from hivemind.queenbee.strategies import (
    BanditStrategy,
    Outcome,
    VerifiedSubtask,
    admissible_mix,
    dpo_loss,
    max_unverified_rows,
)
from hivemind.solverbee.mcp import encode
from hivemind.solverbee.scoring import track_record
from hivemind.solverbee.verifier import certify_achieved_accuracy, decide


def test_queenbee_strategies_positive_and_negative_controls() -> None:
    o = Outcome("cluster-a", "expert-1", verified=True, success=True, cost_tokens=50)
    assert o.cost_tokens == 50

    b = BanditStrategy(prior_alpha=1.0, prior_beta=1.0, exploit_only=False)
    b.observe(o)
    assert b.arm("cluster-a", "expert-1").verified_successes == 1

    assert admissible_mix(3, 7) is True
    assert max_unverified_rows(7) == 3
    assert dpo_loss(-0.5, -1.5, -1.0, -1.0, 0.1) > 0.0

    st = VerifiedSubtask("s1", "cluster-a", "unit_tests", weight=1.0)
    assert st.weight == 1.0

    with pytest.raises(ValueError, match="verified and success must be booleans"):
        Outcome("cluster-a", "expert-1", verified=1, success=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cost_tokens must be an integer >= 0"):
        Outcome("cluster-a", "expert-1", verified=True, success=True, cost_tokens=1.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Beta prior parameters must be finite"):
        BanditStrategy(prior_alpha=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="exploit_only must be a boolean"):
        BanditStrategy(exploit_only="yes")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="row counts must be >= 0"):
        admissible_mix(True, 7)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="verified_rows must be >= 0"):
        max_unverified_rows(True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="beta must be finite"):
        dpo_loss(-0.5, -1.5, -1.0, -1.0, True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="log-probabilities must be finite"):
        dpo_loss(True, -1.5, -1.0, -1.0, 0.1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="weight must be finite"):
        VerifiedSubtask("s1", "cluster-a", "unit_tests", weight=True)  # type: ignore[arg-type]


def test_solverbee_scoring_positive_and_negative_controls() -> None:
    rec = track_record(["VERIFIED"] * 5)
    assert rec["n"] == 5
    assert rec["recruitment_weight"] > 0.0

    with pytest.raises(ValueError, match="iterable of verdict strings"):
        track_record("VERIFIED")


def test_solverbee_mcp_positive_and_negative_controls() -> None:
    raw = encode({"jsonrpc": "2.0", "id": 1, "result": {}})
    assert raw.endswith(b"\n")

    with pytest.raises(ValueError, match="resp must be a dict"):
        encode(["not", "a", "dict"])  # type: ignore[arg-type]


def test_solverbee_verifier_positive_and_negative_controls() -> None:
    checks = {
        "shape": {"status": "pass", "detail": "ok"},
        "hashes": {"status": "pass", "detail": "ok"},
        "reference": {"status": "skip", "detail": "none"},
        "invariants": {"status": "skip", "detail": "no invariants"},
        "tightening": {"status": "pass", "detail": "ok"},
    }
    verdict, _ = decide(checks, usable=True, has_problem=False)
    assert verdict == "AUDITED"

    with pytest.raises(ValueError, match="usable and has_problem must be booleans"):
        decide(checks, usable=1, has_problem=False)  # type: ignore[arg-type]

    spec = {
        "schema": "hivemind-solver-spec/1",
        "id": "test-1",
        "problem": {"kind": "named", "family": "logistic", "params": {"r": 1.0, "K": 1.0}},
        "y0": [0.5],
        "t0": 0.0,
        "t_out": [1.0],
    }
    spec_bytes = json.dumps(spec).encode("utf-8")
    res = {
        "schema": "hivemind-solver-result/1",
        "id": "test-1",
        "status": "ok",
        "t": [1.0],
        "y": [[0.731]],
    }
    cert = certify_achieved_accuracy(spec_bytes, res, res)
    assert cert["schema"] == "hivemind-accuracy-cert/1"

    with pytest.raises(ValueError, match="loose_result and tight_result must be dicts"):
        certify_achieved_accuracy(spec_bytes, "not-a-dict", res)  # type: ignore[arg-type]

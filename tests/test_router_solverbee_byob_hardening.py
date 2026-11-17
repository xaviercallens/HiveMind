"""Negative-control and invariant tests for litestream_lock, router.market, usage_capture, solverbee.sandbox, and swarm.byob dataclasses."""

from __future__ import annotations

import math

import pytest

from hivemind.router.litestream_lock import LockGenerationRecord
from hivemind.router.market import Certificate
from hivemind.router.usage_capture import ClientAttribution, PromptIntent
from hivemind.solverbee.sandbox import SANDBOX_NAMESPACE, SANDBOX_NONE, SandboxOutcome
from hivemind.swarm.byob import HeldOutItem, HeldOutPack


def test_litestream_lock_and_market_certificate_validation() -> None:
    rec = LockGenerationRecord(
        replica_url="gs://hivemind-ledger/router.db",
        owner_id="router-instance-01",
        generation=3,
        acquired_utc=1000.0,
        lease_expires_utc=1300.0,
    )
    assert rec.generation == 3

    with pytest.raises(ValueError, match="replica_url must be a non-empty string"):
        LockGenerationRecord("", "router-01", 1, 100.0, 200.0)
    with pytest.raises(ValueError, match="owner_id must be a non-empty string"):
        LockGenerationRecord("gs://b/db", "  ", 1, 100.0, 200.0)
    with pytest.raises(ValueError, match="generation must be a non-negative integer"):
        LockGenerationRecord("gs://b/db", "r1", -1, 100.0, 200.0)
    with pytest.raises(ValueError, match="acquired_utc must be a finite non-negative number"):
        LockGenerationRecord("gs://b/db", "r1", 1, -1.0, 200.0)
    with pytest.raises(ValueError, match="lease_expires_utc must be >= acquired_utc"):
        LockGenerationRecord("gs://b/db", "r1", 1, 200.0, 100.0)

    cert = Certificate(
        individually_rational_consumers=True,
        individually_rational_producers=True,
        producer_nash=True,
        consumer_nash=None,
        conserved=True,
        max_deviation_gain=0.0,
        win_win=True,
    )
    assert cert.win_win is True

    with pytest.raises(ValueError, match="individually_rational_consumers must be a boolean"):
        Certificate(1, True, True, True, True, 0.0, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="consumer_nash must be a boolean or None"):
        Certificate(True, True, True, "yes", True, 0.0, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_deviation_gain must be a non-negative number"):
        Certificate(True, True, True, True, True, -0.1, True)
    with pytest.raises(ValueError, match="max_deviation_gain must be a non-negative number"):
        Certificate(True, True, True, True, True, math.nan, True)


def test_usage_capture_sandbox_outcome_and_byob_heldout_validation() -> None:
    pi = PromptIntent(task_domain="coding_rust", language="en", preferred_tier="coder")
    assert pi.preferred_tier == "coder"

    with pytest.raises(ValueError, match="task_domain must be a non-empty string"):
        PromptIntent("", "en", None)
    with pytest.raises(ValueError, match="language must be a non-empty string"):
        PromptIntent("coding_rust", " ", None)
    with pytest.raises(ValueError, match="preferred_tier must be a non-empty string or None"):
        PromptIntent("coding_rust", "en", "  ")

    ca = ClientAttribution(client_sdk="cursor", app_title="Cursor IDE")
    assert ca.client_sdk == "cursor"
    with pytest.raises(ValueError, match="client_sdk must be a non-empty string"):
        ClientAttribution("", "Title")
    with pytest.raises(ValueError, match="app_title must be a non-empty string"):
        ClientAttribution("cursor", " ")

    out = SandboxOutcome(ok=True, rc=0, stdout=b"ok\n", wall_s=0.12, sandbox=SANDBOX_NAMESPACE)
    assert out.ok is True

    with pytest.raises(ValueError, match="ok must be a boolean"):
        SandboxOutcome(ok=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rc must be an integer or None"):
        SandboxOutcome(ok=True, rc=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="stdout must be bytes"):
        SandboxOutcome(ok=True, stdout="str")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="wall_s must be a finite non-negative number"):
        SandboxOutcome(ok=True, wall_s=-0.1)
    with pytest.raises(ValueError, match="limit must be a non-empty string or None"):
        SandboxOutcome(ok=False, limit="  ")
    with pytest.raises(ValueError, match="sandbox must be"):
        SandboxOutcome(ok=False, sandbox="docker")
    assert SandboxOutcome(ok=False, sandbox=SANDBOX_NONE).sandbox == SANDBOX_NONE

    item = HeldOutItem("q1", "What is 2+2?", r"\b4\b", "4")
    pack = HeldOutPack("math-pack", (item,))
    assert pack.score(["4"]) == 1.0

    with pytest.raises(ValueError, match="item_id must be a non-empty string"):
        HeldOutItem("", "prompt", r"x", "x")
    with pytest.raises(ValueError, match="pack_id must be a non-empty string"):
        HeldOutPack("  ", (item,))
    with pytest.raises(ValueError, match="items must be a tuple of HeldOutItem instances"):
        HeldOutPack("math-pack", ("not_item",))  # type: ignore[arg-type]

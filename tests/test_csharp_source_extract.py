# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression tests for #564 (C# fat-client source extraction & syntax check), #197 (Litestream single-writer lock),
#190 (idle capacity posted-price vs congestion Kelly market switch), #195 (monitor status page & alerts),
#189 (post-deploy canary contract gate), #188 (clean tagged main deploy check), and #509 (daytime QLoRA schedule & governor).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import extract_csharp_fat_client as csharp_extract  # noqa: E402
import live_user_monitor as monitor  # noqa: E402
import t4_governor as governor  # noqa: E402
from hivemind.router import (  # noqa: E402
    MECHANISM_KELLY,
    MECHANISM_POSTED,
    ConsumerBid,
    FeePolicy,
    ProducerOffer,
    RouterStore,
    certify,
    clear_market,
)
from hivemind.router.litestream_lock import (  # noqa: E402
    FileGenerationStore,
    InMemoryGenerationStore,
    LitestreamWriterLock,
    WriterLockBusyError,
)
from hivemind.router.t4_visibility import build_t4_visibility_payload  # noqa: E402


def test_csharp_extractor_negative_and_positive_controls(tmp_path: Path) -> None:
    """Issue #564: A deliberately broken copy of the C# source fails; the real NATIVE_CSHARP_FAT_CLIENT_SOURCE passes."""
    real_source = csharp_extract.extract_csharp_source()
    assert "namespace HiveMind.ExoplanetHunterClient" in real_source
    assert "static void Main(" in real_source

    # Negative control first: a deliberately broken copy with an unclosed brace must fail
    broken_source = real_source + "\npublic class BrokenSyntax { public void Bad() {\n"
    neg_out = tmp_path / "Broken_Exoplanet_Hunter.cs"
    neg_res = csharp_extract.write_and_check_csharp(neg_out, source_override=broken_source)
    assert neg_res["passed"] is False
    assert neg_res["syntax"]["valid"] is False
    assert len(neg_res["syntax"]["errors"]) >= 1

    # Positive control: the real C# source written to HiveMind_Exoplanet_Hunter.cs passes
    pos_out = tmp_path / "HiveMind_Exoplanet_Hunter.cs"
    pos_res = csharp_extract.write_and_check_csharp(pos_out)
    assert pos_out.exists() and pos_out.stat().st_size > 10_000
    assert pos_res["passed"] is True
    assert pos_res["syntax"]["valid"] is True
    assert pos_res["syntax"]["errors"] == []


def test_litestream_single_writer_lock_allows_only_one_writer(tmp_path: Path) -> None:
    """Issue #197 (HM-S16-10): Two RouterStore instances start on one replica; only one writes."""
    replica_url = "gs://hivemind-litestream-ledger/router.db"
    backend = InMemoryGenerationStore()
    db_path = tmp_path / "shared_router.sqlite"

    lock1 = LitestreamWriterLock(replica_url, "revision-00041", backend)
    store1 = RouterStore(db_path, b"s" * 32, writer_lock=lock1)
    assert store1.read_only is False
    assert lock1.is_writer is True

    # Primary writer writes an account and grant
    acct_id, api_key = store1.register("pioneer-writer", "consumer")
    store1.grant(acct_id, 50.0, "initial-grant")
    assert store1.balance(acct_id) == 50.0

    # Second instance (e.g. overlapping canary on the same replica URL) cannot take the lock
    lock2 = LitestreamWriterLock(replica_url, "revision-00042-canary", backend)
    store2 = RouterStore(db_path, b"s" * 32, writer_lock=lock2)
    assert store2.read_only is True
    assert lock2.is_writer is False

    # Second instance can still read, but any write attempt raises WriterLockBusyError
    assert store2.balance(acct_id) == 50.0
    with pytest.raises(WriterLockBusyError, match="read-only"):
        store2.register("second-writer-blocked", "consumer")
    with pytest.raises(WriterLockBusyError, match="read-only"):
        store2.grant(acct_id, 10.0, "unauthorized-second-writer")

    # Fail-closed startup mode refuses to start when lock is held
    lock3_fail_closed = LitestreamWriterLock(replica_url, "revision-00043", backend, fail_closed=True)
    with pytest.raises(WriterLockBusyError):
        RouterStore(db_path, b"s" * 32, writer_lock=lock3_fail_closed)

    # Once store1 closes and releases the lock, a new writer can acquire it (also tested with FileGenerationStore)
    store1.close()
    file_backend = FileGenerationStore(tmp_path / "locks")
    f_lock1 = LitestreamWriterLock(replica_url, "host-a", file_backend)
    f_lock2 = LitestreamWriterLock(replica_url, "host-b", file_backend)
    assert f_lock1.acquire() is True
    assert f_lock2.acquire() is False


def test_market_uses_posted_reserve_at_idle_and_kelly_under_congestion_with_certificates() -> None:
    """Issue #190 (HM-S16-03): Reproduce 402 at idle under old rule and service under new rule; certificates hold in both."""
    fee = FeePolicy()
    offers = [ProducerOffer("t4-bee", cost_per_1k=0.4, capacity_1k=30.0)]
    bids = [
        ConsumerBid("test-hi-1", 10.0),
        ConsumerBid("test-hi-2", 10.0),
        ConsumerBid("test-hi-3", 10.0),
        ConsumerBid("test-hi-4", 10.0),
        ConsumerBid("new-user-default", 2.0),
    ]

    # Old rule / congested regime (measured_load_1k=30.0 >= capacity): clears at Kelly ~7.5 HMT/1k, pricing out new-user-default
    eq_congested = clear_market("bee-stars", bids, offers, fee, measured_load_1k=30.0)
    assert eq_congested.mechanism == MECHANISM_KELLY
    assert eq_congested.price_per_1k > 2.0
    assert eq_congested.consumer_shares["new-user-default"] == 0.0
    cert_congested = certify(eq_congested, bids, offers, measured_load_1k=30.0)
    assert cert_congested.win_win is True

    # New rule at idle capacity (measured_load_1k=0.2 < 30.0): clears at posted reserve price (< 2.0), serving new-user-default
    eq_idle = clear_market("bee-stars", bids, offers, fee, measured_load_1k=0.2)
    assert eq_idle.mechanism == MECHANISM_POSTED
    assert eq_idle.price_per_1k < 2.0
    assert eq_idle.consumer_shares["new-user-default"] > 0.0
    cert_idle = certify(eq_idle, bids, offers, measured_load_1k=0.2)
    assert cert_idle.win_win is True
    assert cert_idle.conserved is True


def test_status_alerts_canary_contract_tagged_deploy_and_daytime_governor() -> None:
    """Issues #195, #189, #188, and #509: Status alerts, canary contract gate, clean tagged deploy, and daytime QLoRA policy."""
    # #195: Failed probe, 402, or 5xx burst produces an alert within one monitor period
    records = [
        {
            "utc": "2026-10-25T12:14:00Z",
            "probes": {"valid": True, "results": [{"ok": True}] * 6 + [{"ok": False}] * 9},
            "traffic": {"services": {"router": {"402": 2, "5xx": 1, "chat_latency_s": {"p50": 0.8, "p95": 4.2}}}},
        }
    ]
    status = monitor.build_public_status_and_alerts(records)
    alert_types = {a["type"] for a in status["alerts"]}
    assert {"probe_failed", "402_default_price", "5xx_burst"} <= alert_types
    assert status["probe_pass_rate"] == 0.4

    # #189: Broken canary is refused; clean 5/5 canary with 0 currency mentions passes
    broken_canary = {
        "valid": True,
        "results": [
            {"service": "ruche", "model": "hivemind/bee-stars", "status": 200, "ok": False, "reply_head": "La Ruche Top-2"},
            {"service": "ruche", "model": "control:no-key", "status": 200, "ok": False},
            {"service": "ruche", "model": "control:unknown-model", "status": 200, "ok": False},
        ],
    }
    refused = monitor.verify_canary_post_deploy_contract(
        broken_canary,
        {"ruche": "<p>Balance: 10.00 EUR / 10.00 USD</p>"},
        default_price_first_request_status=402,
    )
    assert refused["all_passed"] is False
    assert refused["zero_currency_mentions"] is False

    clean_canary = {
        "valid": True,
        "results": [
            {"service": "router", "model": "hivemind/bee-stars", "status": 200, "ok": True, "reply_head": "Class: M"},
            {"service": "router", "model": "hivemind/coder", "status": 200, "ok": True, "reply_head": "def add(a, b):"},
            {"service": "router", "model": "control:no-key", "status": 401, "ok": True},
            {"service": "router", "model": "control:unknown-model", "status": 404, "ok": True},
        ],
    }
    accepted = monitor.verify_canary_post_deploy_contract(
        clean_canary,
        {"router": "<p>Balance: 100 HiveMind Reflexion Points (HMT)</p>"},
        default_price_first_request_status=200,
    )
    assert accepted["all_passed"] is True

    # #188: Deploy from dirty tree is refused; tagged clean main commit is allowed
    dirty = monitor.verify_clean_tagged_main_deploy(" M src/hivemind/router/gateway.py\n", "abc1234", "abc1234", "abc1234")
    assert dirty["allowed"] is False and "dirty_working_tree" in dirty["reasons"]
    clean_deploy = monitor.verify_clean_tagged_main_deploy("", "63fd643", "63fd643", "63fd643")
    assert clean_deploy["allowed"] is True

    # #509: T4 visibility exposes qlora_progress.schedule and governor honours NIGHT_ONLY and daytime idle thresholds
    snap = build_t4_visibility_payload()
    assert snap["qlora_progress"]["schedule"] == "NIGHT_ONLY"
    assert governor.should_resume_training(False, "yielded", 400.0, in_night_window=False, day_idle_s=300.0, schedule="NIGHT_ONLY") is False
    assert governor.should_resume_training(False, "yielded", 400.0, in_night_window=False, day_idle_s=300.0, schedule="DAY_AND_NIGHT") is True
    assert governor.should_resume_training(True, "yielded", 3600.0, in_night_window=True, idle_start_s=1800.0) is False

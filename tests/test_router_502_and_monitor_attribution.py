# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression tests for #598 (502 upstream_failure model & reason logging + uncharged usage row),
#570/#404 (slow chat & per-model/per-producer latency attribution), #591/#589/#608 (T4 snapshot attribution),
#569 (gatekeeper conflicting PR handling), and #415 (zero stub audit failures across src/ and scripts/).
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import live_user_monitor as monitor  # noqa: E402
import pr_gatekeeper as gatekeeper  # noqa: E402
from hivemind.router import RouterStore, create_router_app  # noqa: E402


def test_502_upstream_failure_logs_model_and_reasons_and_writes_uncharged_usage_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Issue #598: When all producers fail, /v1/chat/completions returns 502, writes an uncharged usage row, and logs model + reasons."""
    store = RouterStore(":memory:", b"k" * 32)
    consumer_id, consumer_key = store.register("alice-consumer", "consumer")
    store.grant(consumer_id, 100.0, "test")
    _, prod_a_key = store.register("prod-alpha", "producer")
    _, prod_b_key = store.register("prod-beta", "producer")

    def failing_upstream(request: httpx.Request) -> httpx.Response:
        if "alpha" in str(request.url):
            return httpx.Response(530, text="Tunnel frozen")
        raise httpx.ConnectError("Connection refused", request=request)

    upstream_client = httpx.Client(transport=httpx.MockTransport(failing_upstream))
    app = create_router_app(store, http_client=upstream_client, allow_private_producers=True)
    client = TestClient(app)

    for key, url in ((prod_a_key, "https://alpha.example.com"), (prod_b_key, "https://beta.example.com")):
        hb = client.post(
            "/v1/hivemind/producer/heartbeat",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "endpoint_url": url,
                "models": [{"id": "hivemind/coder", "local_name": "qwen2.5-coder:7b", "loaded": True}],
                "cost_per_1k": 0.2,
                "capacity_1k": 50.0,
                "gpu_vram_mib": 16384,
            },
        )
        assert hb.status_code == 200

    balance_before = client.get(
        "/v1/hivemind/me",
        headers={"Authorization": f"Bearer {consumer_key}"},
    ).json()["balance"]

    with caplog.at_level(logging.WARNING, logger="hivemind.router"):
        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {consumer_key}"},
            json={
                "model": "hivemind/coder",
                "messages": [{"role": "user", "content": "Write add(a, b)"}],
            },
        )

    assert resp.status_code == 502
    body = resp.json()
    assert body["error"]["code"] == "upstream_failure"
    assert "not charged" in body["error"]["message"]

    balance_after = client.get(
        "/v1/hivemind/me",
        headers={"Authorization": f"Bearer {consumer_key}"},
    ).json()["balance"]
    assert balance_after == balance_before

    # Verify the 502 log line names the model and per-producer failure reasons, and never logs the consumer key or prompt
    exhausted_logs = [r.getMessage() for r in caplog.records if "upstream_failure: every producer failed" in r.getMessage()]
    assert len(exhausted_logs) == 1
    log_line = exhausted_logs[0]
    assert "model=coder" in log_line
    assert "reasons=" in log_line
    assert "HTTP 530" in log_line
    assert "ConnectError" in log_line
    assert consumer_key not in log_line
    assert "Write add(a, b)" not in log_line

    # Verify an uncharged usage telemetry row was written
    usage = store.usage_capture_summary(now_utc=1e12, window_s=1e12)
    assert usage["total_requests"] == 1
    assert usage["total_charged_hmt"] == 0.0
    assert usage["total_billed_tokens"] == 0
    assert "coder" in usage["by_routed_model"]
    assert usage["by_routed_model"]["coder"]["charged_hmt"] == 0.0
    assert "upstream_failure" in usage["by_producer"]
    assert usage["by_producer"]["upstream_failure"]["charged_hmt"] == 0.0


def test_monitor_attributes_slow_chats_and_per_model_latency_positive_and_negative_controls() -> None:
    """Issues #570 & #404: Slow chats are listed with route, model, and caller class; per-model latency table names the dominant cause."""
    rows = [
        {
            "timestamp": "2026-11-04T08:15:32Z",
            "httpRequest": {
                "latency": "6.8s",
                "responseSize": "4580",
                "userAgent": "python-httpx/0.28.1",
                "status": 200,
                "requestUrl": "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/v1/chat/completions",
                "remoteIp": "203.0.113.55",
            },
            "jsonPayload": {
                "model": "hivemind/coder",
                "producer_id": "gcp-t4-coder",
                "ttft_s": 0.42,
                "total_tokens": 320,
            },
        },
        {
            "timestamp": "2026-11-04T08:15:31Z",
            "httpRequest": {
                "latency": "0.35s",
                "responseSize": "920",
                "userAgent": "Mozilla/5.0",
                "status": 200,
                "requestUrl": "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/v1/chat/completions",
                "remoteIp": "198.51.100.12",
            },
            "jsonPayload": {
                "model": "hivemind/bee-stars",
                "producer_id": "gcp-t4-bee",
                "ttft_s": 0.18,
                "total_tokens": 42,
            },
        },
    ]

    slow = monitor.attribute_slow_chats(rows, threshold_s=3.0)
    assert len(slow) == 1
    assert slow[0]["model"] == "hivemind/coder"
    assert slow[0]["route"] == "/v1/chat/completions"
    assert slow[0]["caller_class"] == "scripted"
    assert slow[0]["user_agent"] == "python-httpx/0.28.1"
    assert "203.0.113.55" not in str(slow)

    table = monitor.latency_by_model_and_producer(rows)
    assert table["by_model"]["hivemind/coder"]["p95_s"] == 6.8
    assert table["by_model"]["hivemind/coder"]["ttft_p95_s"] == 0.42
    assert table["by_model"]["hivemind/coder"]["mean_tokens"] == 320.0
    assert table["by_model"]["hivemind/coder"]["dominant_cause"] == "long_generation"
    assert table["by_model"]["hivemind/bee-stars"]["dominant_cause"] == "within_idle_baseline"
    assert table["slow_models"] == ["hivemind/coder"]
    assert table["dominant_cause"] == "long_generation"

    # Negative control: uniform fast chats have no slow_models and dominant_cause == within_idle_baseline
    fast_only = [rows[1], rows[1]]
    fast_table = monitor.latency_by_model_and_producer(fast_only)
    assert fast_table["slow_models"] == []
    assert fast_table["dominant_cause"] == "within_idle_baseline"


def test_t4_window_snapshot_attribution_for_paris_21_and_high_util_contention() -> None:
    """Issues #608 & #591 (replaces #589): 15 training snapshots at 0% util / 5201 MiB VRAM + 86-98% contention window."""
    paris_21_snapshots = [
        {"state": "training", "gpu_util_pct": 0, "gpu_used_mib": 5201, "boinc_procs": []}
        for _ in range(15)
    ] + [
        {"state": "yielded", "gpu_util_pct": 94, "gpu_used_mib": 10509, "boinc_procs": []}
        for _ in range(5)
    ]
    report = monitor.attribute_t4_window_snapshots(
        paris_21_snapshots,
        assimilated_before=171,
        assimilated_after=172,
    )
    assert report["training_snapshots_count"] == 15
    assert len(report["training_table"]) == 15
    assert all(r["util_pct"] == 0 and r["vram_mib"] == 5201 for r in report["training_table"])
    assert report["gpu_trained_in_snapshots"] is False
    assert "171 -> 172" in report["training_statement"]
    assert report["slow_chat_attribution"] == "gpu_contention_high_util"
    assert (ROOT / "docs/T4_CONTENTION_AND_TRAINING_ATTRIBUTION.md").is_file()


def test_gatekeeper_retry_merge_downgrades_conflicting_pr_to_hold() -> None:
    """Issue #569: A previously ACCEPTed PR that is now CONFLICTING is reported as HOLD (merge conflict), never ACCEPT."""
    conflicting_pr = {
        "number": 442,
        "headRefOid": "deadbeef1234567890",
        "isDraft": False,
        "mergeable": "CONFLICTING",
    }
    res = gatekeeper.retry_merge(conflicting_pr, "ACCEPT", merge=False, dry_run=True)
    assert res["decision"] == "HOLD (merge conflict)"
    assert res["merged"] is False

# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression tests for `scripts/rtx2070_failover_orchestrator.py` (GitHub Issue #487).

Verifies:
1. With `t4_reachable=True` or a failing tunnel/shield probe, no `/v1/hivemind/admin/producers/reconnect`
   request is issued.
2. A reconnect never lists a model that local Ollama does not serve (checked against `/api/tags`),
   never claims `bee-stars`, and only marks `loaded=True` for models currently listed in `/api/ps`.
3. At most one `/v1/hivemind/admin/producers/reconnect` is issued per `t4_reachable` state change.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import rtx2070_failover_orchestrator as orch  # noqa: E402


def _build_test_client(
    *,
    ollama_tags: list[str],
    ollama_ps: list[tuple[str, int]],
    tunnel_status: int = 200,
    recorded_requests: list[httpx.Request],
) -> httpx.Client:
    """Create an in-memory httpx.Client with MockTransport recording every outbound request."""

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        url = str(request.url)
        if url.endswith("/api/tags"):
            return httpx.Response(
                200,
                json={"models": [{"name": tag, "model": tag} for tag in ollama_tags]},
            )
        if url.endswith("/api/ps"):
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": tag, "model": tag, "size": size_mib * 1024 * 1024}
                        for tag, size_mib in ollama_ps
                    ]
                },
            )
        if url.endswith("/v1/models") and "trycloudflare.com" in url:
            return httpx.Response(tunnel_status, json={"object": "list", "data": []})
        if url.endswith("/v1/hivemind/admin/producers/reconnect"):
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "reconnected_count": 42,
                    "live_hiveminders": 42,
                    "live_models": ["coder", "coder-lite"],
                },
            )
        if url.endswith("/v1/hivemind/producer/heartbeat"):
            return httpx.Response(200, json={"status": "live"})
        return httpx.Response(404, json={"error": "not_found"})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_no_reconnect_when_t4_is_reachable_or_tunnel_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defect (1): Never call admin/producers/reconnect when t4_reachable=True or tunnel probe fails."""
    monkeypatch.setenv("HIVEMIND_ADMIN_KEY", "test-admin-key-do-not-print")
    monkeypatch.setenv("HIVEMIND_API_KEY", "hm-test-producer-key-do-not-print")
    monkeypatch.setenv("HIVEMIND_ENDPOINT_TOKEN", "test-endpoint-token-0123456789abcdef")

    recorded: list[httpx.Request] = []
    state: dict[str, Any] = {"last_t4_reachable": None, "reconnected_for_outage": False}

    # Case A: t4_reachable=True -> zero reconnect calls
    with _build_test_client(
        ollama_tags=["qwen2.5-coder:7b", "qwen2.5-coder:3b"],
        ollama_ps=[("qwen2.5-coder:7b", 4800)],
        tunnel_status=200,
        recorded_requests=recorded,
    ) as client:
        res_a = orch.route_traffic_to_rtx2070_orchestrator(
            "https://router.example.test",
            "https://healthy-tunnel.trycloudflare.com",
            t4_reachable=True,
            shield_ok=True,
            state=state,
            http_client=client,
            allow_fleet_reconnect=True,
        )
    assert res_a["reconnected"] is False
    assert res_a["action"] == "noop_t4_reachable"
    assert not any(str(r.url).endswith("/v1/hivemind/admin/producers/reconnect") for r in recorded)

    # Case B: t4_reachable=False, but tunnel returns HTTP 530 -> zero reconnect calls
    recorded.clear()
    with _build_test_client(
        ollama_tags=["qwen2.5-coder:7b", "qwen2.5-coder:3b"],
        ollama_ps=[("qwen2.5-coder:7b", 4800)],
        tunnel_status=530,
        recorded_requests=recorded,
    ) as client:
        res_b = orch.route_traffic_to_rtx2070_orchestrator(
            "https://router.example.test",
            "https://dead-tunnel.trycloudflare.com",
            t4_reachable=False,
            shield_ok=True,
            state=state,
            http_client=client,
            allow_fleet_reconnect=True,
        )
    assert res_b["reconnected"] is False
    assert res_b["action"] == "aborted_tunnel_unhealthy"
    assert not any(str(r.url).endswith("/v1/hivemind/admin/producers/reconnect") for r in recorded)

    # Case C: t4_reachable=False, but local shield_ok=False -> zero reconnect calls
    recorded.clear()
    with _build_test_client(
        ollama_tags=["qwen2.5-coder:7b", "qwen2.5-coder:3b"],
        ollama_ps=[("qwen2.5-coder:7b", 4800)],
        tunnel_status=200,
        recorded_requests=recorded,
    ) as client:
        res_c = orch.route_traffic_to_rtx2070_orchestrator(
            "https://router.example.test",
            "https://healthy-tunnel.trycloudflare.com",
            t4_reachable=False,
            shield_ok=False,
            state=state,
            http_client=client,
            allow_fleet_reconnect=True,
        )
    assert res_c["reconnected"] is False
    assert res_c["action"] == "aborted_tunnel_unhealthy"
    assert not any(str(r.url).endswith("/v1/hivemind/admin/producers/reconnect") for r in recorded)


def test_reconnect_only_lists_models_served_by_ollama_tags_and_accurate_loaded_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defects (2) & (4): Never list models absent from /api/tags, never claim bee-stars, only mark /api/ps as loaded."""
    monkeypatch.setenv("HIVEMIND_ADMIN_KEY", "test-admin-key-do-not-print")
    monkeypatch.setenv("HIVEMIND_API_KEY", "hm-test-producer-key-do-not-print")
    monkeypatch.setenv("HIVEMIND_ENDPOINT_TOKEN", "test-endpoint-token-0123456789abcdef")

    recorded: list[httpx.Request] = []
    state: dict[str, Any] = {"last_t4_reachable": True, "reconnected_for_outage": False}
    served_tags = ["qwen2.5-coder:7b", "qwen2.5-coder:3b", "hivemind-bee-stars"]

    with _build_test_client(
        ollama_tags=served_tags,
        ollama_ps=[("qwen2.5-coder:7b", 4700)],
        tunnel_status=200,
        recorded_requests=recorded,
    ) as client:
        res = orch.route_traffic_to_rtx2070_orchestrator(
            "https://router.example.test",
            "https://healthy-tunnel.trycloudflare.com",
            t4_reachable=False,
            shield_ok=True,
            state=state,
            http_client=client,
            allow_fleet_reconnect=True,
        )

    assert res["reconnected"] is True
    reconnect_reqs = [r for r in recorded if str(r.url).endswith("/v1/hivemind/admin/producers/reconnect")]
    assert len(reconnect_reqs) == 1
    body = json.loads(reconnect_reqs[0].content.decode("utf-8"))
    models = body["models"]

    # Every advertised model must map to a tag present in /api/tags
    assert len(models) == 2
    for entry in models:
        assert entry["local_name"] in served_tags
        assert entry["id"] != "bee-stars"

    by_id = {m["id"]: m for m in models}
    assert set(by_id.keys()) == {"coder", "coder-lite"}
    # Only qwen2.5-coder:7b is in /api/ps -> coder is loaded=True, coder-lite is loaded=False
    assert by_id["coder"]["loaded"] is True
    assert by_id["coder"]["vram_mib"] == 4700
    assert by_id["coder-lite"]["loaded"] is False
    assert by_id["coder-lite"]["vram_mib"] == 0


def test_at_most_one_reconnect_per_state_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defect (3): At most one admin/producers/reconnect is issued per t4_reachable transition."""
    monkeypatch.setenv("HIVEMIND_ADMIN_KEY", "test-admin-key-do-not-print")
    monkeypatch.setenv("HIVEMIND_API_KEY", "hm-test-producer-key-do-not-print")
    monkeypatch.setenv("HIVEMIND_ENDPOINT_TOKEN", "test-endpoint-token-0123456789abcdef")

    recorded: list[httpx.Request] = []
    state: dict[str, Any] = {"last_t4_reachable": True, "reconnected_for_outage": False}

    with _build_test_client(
        ollama_tags=["qwen2.5-coder:1.5b"],
        ollama_ps=[("qwen2.5-coder:1.5b", 1319)],
        tunnel_status=200,
        recorded_requests=recorded,
    ) as client:
        # Cycle 1: T4 goes down (True -> False) -> 1 reconnect issued
        r1 = orch.route_traffic_to_rtx2070_orchestrator(
            "https://router.example.test",
            "https://healthy-tunnel.trycloudflare.com",
            t4_reachable=False,
            shield_ok=True,
            state=state,
            http_client=client,
            allow_fleet_reconnect=True,
        )
        # Cycles 2 and 3: T4 remains down (False -> False) -> 0 additional reconnects
        r2 = orch.route_traffic_to_rtx2070_orchestrator(
            "https://router.example.test",
            "https://healthy-tunnel.trycloudflare.com",
            t4_reachable=False,
            shield_ok=True,
            state=state,
            http_client=client,
            allow_fleet_reconnect=True,
        )
        r3 = orch.route_traffic_to_rtx2070_orchestrator(
            "https://router.example.test",
            "https://healthy-tunnel.trycloudflare.com",
            t4_reachable=False,
            shield_ok=True,
            state=state,
            http_client=client,
            allow_fleet_reconnect=True,
        )

        reconnect_calls = [r for r in recorded if str(r.url).endswith("/v1/hivemind/admin/producers/reconnect")]
        assert r1["reconnected"] is True
        assert r2["reconnected"] is False
        assert r2["action"] == "noop_already_reconnected_for_current_outage"
        assert r3["reconnected"] is False
        assert len(reconnect_calls) == 1

        # Cycle 4: T4 recovers (False -> True) -> resets outage state without reconnecting
        r4 = orch.route_traffic_to_rtx2070_orchestrator(
            "https://router.example.test",
            "https://healthy-tunnel.trycloudflare.com",
            t4_reachable=True,
            shield_ok=True,
            state=state,
            http_client=client,
            allow_fleet_reconnect=True,
        )
        assert r4["reconnected"] is False
        assert len([r for r in recorded if str(r.url).endswith("/v1/hivemind/admin/producers/reconnect")]) == 1

        # Cycle 5: T4 goes down a second time (True -> False) -> issues 1 reconnect for the new transition
        r5 = orch.route_traffic_to_rtx2070_orchestrator(
            "https://router.example.test",
            "https://healthy-tunnel.trycloudflare.com",
            t4_reachable=False,
            shield_ok=True,
            state=state,
            http_client=client,
            allow_fleet_reconnect=True,
        )
        assert r5["reconnected"] is True
        assert len([r for r in recorded if str(r.url).endswith("/v1/hivemind/admin/producers/reconnect")]) == 2


def test_by_default_an_outage_only_heartbeats_the_own_account_every_cycle_and_never_reconnects_the_fleet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-11-02 incident: the fleet reconnect re-pointed ~42 accounts at one tunnel; by default it must never be called,
    and the 2070's own heartbeat must keep going every cycle of the outage (the early 'already reconnected' return stopped it)."""
    monkeypatch.setenv("HIVEMIND_ADMIN_KEY", "test-admin-key-do-not-print")
    monkeypatch.setenv("HIVEMIND_API_KEY", "hm-test-producer-key-do-not-print")
    monkeypatch.setenv("HIVEMIND_ENDPOINT_TOKEN", "test-endpoint-token-0123456789abcdef")
    recorded: list[httpx.Request] = []
    state: dict[str, Any] = {"last_t4_reachable": True, "reconnected_for_outage": False}
    with _build_test_client(
        ollama_tags=["qwen2.5-coder:1.5b"], ollama_ps=[("qwen2.5-coder:1.5b", 1319)], tunnel_status=200, recorded_requests=recorded
    ) as client:
        results = [
            orch.route_traffic_to_rtx2070_orchestrator(
                "https://router.example.test",
                "https://healthy-tunnel.trycloudflare.com",
                t4_reachable=False,
                shield_ok=True,
                state=state,
                http_client=client,
            )
            for _ in range(3)
        ]
    assert [r["action"] for r in results] == ["heartbeat_own_account"] * 3
    assert all(r["reconnected"] is False and r["producer_heartbeat_http"] == 200 for r in results)
    assert not any(str(r.url).endswith("/v1/hivemind/admin/producers/reconnect") for r in recorded)
    heartbeats = [r for r in recorded if str(r.url).endswith("/v1/hivemind/producer/heartbeat")]
    assert len(heartbeats) == 3
    assert all("x-admin-key" not in r.headers for r in heartbeats)


def test_one_failed_probe_is_not_an_outage(tmp_path: Path) -> None:
    """The T4 counts as unreachable only after T4_OFFLINE_AFTER_FAILURES consecutive failed probes; a success resets the streak."""
    state_file = tmp_path / "failover_state.json"
    seq = [False, False, True, False, False, False, False]
    got = [orch.debounce_t4_reachable(ok, state_file=state_file) for ok in seq]
    assert got == [True, True, True, True, True, False, False]
    assert json.loads(state_file.read_text(encoding="utf-8"))["t4_fail_streak"] == 4


def test_sync_opentrainer_and_gpu_telemetry_uses_live_gpu_probe_and_blocks_when_absent_ip291_ip292(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IP-291 & IP-292: telemetry POST must reflect live GPU probe, send nothing when probe fails, and default hb_ack ok to False."""
    import hivemind_opentrainer_agent as ota

    monkeypatch.setenv("HIVEMIND_API_KEY", "hm-test-producer-key-do-not-print")
    hb_return: dict[str, Any] = {}
    monkeypatch.setattr(ota, "send_heartbeat", lambda **kwargs: dict(hb_return))
    monkeypatch.setattr(ota, "submit_verified_delta", lambda **kwargs: {"round_id": kwargs["round_id"]})

    recorded: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append(request)
        if str(request.url).endswith("/v1/hivemind/t4/telemetry"):
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404, json={"error": "not_found"})

    e8_round = {
        "round_id": 413,
        "inner_steps": 4,
        "loss_before": 1.42,
        "loss_after": 1.39,
        "delta_norm": 0.12,
        "code_pin": "a" * 64,
    }

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        # 1. Probe #1 with empty hb_ack -> opentrainer_heartbeat_ok must be False (IP-292), POST matches probe #1 (IP-291)
        res1 = orch.sync_opentrainer_and_gpu_telemetry(
            "https://router.example.test",
            e8_round,
            t4_reachable=False,
            gpu_probe_fn=lambda: {
                "gpu_name": "NVIDIA GeForce RTX 2070",
                "used_mib": 3210,
                "total_mib": 8192,
                "free_mib": 4982,
                "temp_c": 51.5,
                "util_pct": 44.0,
                "power_w": 88.2,
            },
            http_client=client,
        )
        assert res1["opentrainer_heartbeat_ok"] is False
        assert res1["telemetry_push_http"] == 200
        assert res1["telemetry_state"] == "SENT"
        assert len(recorded) == 1
        body1 = json.loads(recorded[0].content.decode("utf-8"))
        assert body1["gpu"] == {
            "used_mib": 3210,
            "total_mib": 8192,
            "free_mib": 4982,
            "temp_c": 51.5,
            "util_pct": 44.0,
            "power_w": 88.2,
        }

        # 2. Probe #2 with different GPU values and hb_ack={"ok": True} -> POST body changes accordingly
        hb_return["ok"] = True
        hb_return["node"] = {"node_id": "node-rtx2070"}
        res2 = orch.sync_opentrainer_and_gpu_telemetry(
            "https://router.example.test",
            e8_round,
            t4_reachable=True,
            gpu_probe_fn=lambda: {
                "gpu_name": "NVIDIA GeForce RTX 2070",
                "used_mib": 4100,
                "total_mib": 8192,
                "free_mib": 4092,
                "temp_c": 62.0,
                "util_pct": 78.5,
                "power_w": 125.4,
            },
            http_client=client,
        )
        assert res2["opentrainer_heartbeat_ok"] is True
        assert len(recorded) == 2
        body2 = json.loads(recorded[1].content.decode("utf-8"))
        assert body2["gpu"]["used_mib"] == 4100
        assert body2["gpu"]["free_mib"] == 4092
        assert body2["gpu"]["temp_c"] == 62.0

        # 3. No readable GPU (probe returns None) -> no POST sent (BLOCKED)
        recorded.clear()
        res3 = orch.sync_opentrainer_and_gpu_telemetry(
            "https://router.example.test",
            e8_round,
            t4_reachable=False,
            gpu_probe_fn=lambda: None,
            http_client=client,
        )
        assert res3["telemetry_push_http"] == 0
        assert res3["telemetry_state"] == "BLOCKED"
        assert res3["opentrainer_heartbeat_ok"] is False
        assert recorded == []


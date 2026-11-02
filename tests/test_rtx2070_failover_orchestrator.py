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
        )
        # Cycles 2 and 3: T4 remains down (False -> False) -> 0 additional reconnects
        r2 = orch.route_traffic_to_rtx2070_orchestrator(
            "https://router.example.test",
            "https://healthy-tunnel.trycloudflare.com",
            t4_reachable=False,
            shield_ok=True,
            state=state,
            http_client=client,
        )
        r3 = orch.route_traffic_to_rtx2070_orchestrator(
            "https://router.example.test",
            "https://healthy-tunnel.trycloudflare.com",
            t4_reachable=False,
            shield_ok=True,
            state=state,
            http_client=client,
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
        )
        assert r5["reconnected"] is True
        assert len([r for r in recorded if str(r.url).endswith("/v1/hivemind/admin/producers/reconnect")]) == 2

# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Residency, VRAM-budget and routed-upstream tests for the HiveMind producer agent.

Covers what `test_router_producer_agent.py` leaves out: the Ollama `/api/ps` and `/api/tags` parsers on
messy listings, the VRAM budget filter (pinned daemons + free-VRAM floor), catalog / remote-config refusals,
the `/v1/models` preflight, a failing control-state write, and the `--routed-upstream` residency probe used
for the Crowd-v1 producer (loopback-only, `/healthz` first, `/v1/models` fallback). All HTTP goes through
`httpx.MockTransport`; nothing leaves the process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from hivemind.router import hivemind_producer_agent as agent

GATEWAY = "https://gateway.example.org"
ENDPOINT = "https://producer.example.org"
MIB = 1024 * 1024


def _json_transport(routes: dict[str, Any], seen: list[str] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if seen is not None:
            seen.append(url)
        for suffix, reply in routes.items():
            if url.endswith(suffix):
                status, body = reply
                return httpx.Response(status, json=body) if body is not None else httpx.Response(status)
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Ollama parsers
# ---------------------------------------------------------------------------


def test_residency_probe_skips_malformed_entries() -> None:
    listing = {"models": [
        "not-a-dict",
        {"name": "", "size_vram": 5 * MIB},
        {"model": "qwen3:8b", "size_vram": 6 * MIB},
        {"name": "tiny", "size_vram": True},
        {"name": "neg", "size_vram": -1},
    ]}
    resident = agent.probe_residency("http://127.0.0.1:11434", transport=_json_transport({"/api/ps": (200, listing)}))
    assert resident == {"qwen3:8b": 6, "tiny": 0, "neg": 0}


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ((200, {"models": [{"name": "a", "size": 3 * MIB}, {"model": "b", "size": -5}, {"name": None}, 7]}), {"a": 3}),
        ((200, {"models": "nope"}), None),
        ((500, {"error": "boom"}), None),
        ((200, ["not", "a", "dict"]), None),
    ],
)
def test_model_size_probe_parses_tags_and_fails_soft(reply: tuple[int, Any], expected: dict[str, int] | None) -> None:
    sizes = agent.probe_model_sizes_mib("http://127.0.0.1:11434/", transport=_json_transport({"/api/tags": reply}))
    assert sizes == expected


def test_model_size_probe_refuses_bad_arguments() -> None:
    with pytest.raises(ValueError, match="local_endpoint"):
        agent.probe_model_sizes_mib("  ")
    for bad in (0, -1.0, float("inf"), True):
        with pytest.raises(ValueError, match="timeout_s"):
            agent.probe_model_sizes_mib("http://127.0.0.1:11434", timeout_s=bad)


# ---------------------------------------------------------------------------
# VRAM budget filter
# ---------------------------------------------------------------------------


def test_vram_budget_filter_keeps_the_t4_free_floor() -> None:
    models = [
        {"id": "big", "local_name": "qwen3:14b"},
        {"id": "small", "local_name": "qwen3"},
        {"id": "unknown", "local_name": "mystery"},
    ]
    sizes = {"qwen3:14b": 9_000, "qwen3:latest": 5_000}
    kept, dropped = agent.filter_models_by_vram_budget(models, sizes, gpu_vram_mib=15_360, pinned_vram_mib=1_255,
                                                       min_free_vram_mib=4_300)
    # budget = 15360 - 1255 - 4300 = 9805 MiB
    assert [m["id"] for m in kept] == ["big", "small", "unknown"] and dropped == []
    kept, dropped = agent.filter_models_by_vram_budget(models, sizes, gpu_vram_mib=12_000, pinned_vram_mib=1_255,
                                                       min_free_vram_mib=4_300)
    assert [m["id"] for m in kept] == ["small", "unknown"]  # sizes resolved through the ':latest' tag too
    assert dropped == ["big=qwen3:14b (9000 MiB > 6445 MiB budget)"]
    assert kept[0] is not models[1]  # copies, so callers cannot mutate the advertised list by accident


def test_vram_budget_filter_refusals_and_passthrough() -> None:
    with pytest.raises(TypeError, match="models"):
        agent.filter_models_by_vram_budget("model", {}, gpu_vram_mib=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="model_sizes_mib"):
        agent.filter_models_by_vram_budget([], [("a", 1)], gpu_vram_mib=1)  # type: ignore[arg-type]
    models = [{"id": "m", "local_name": "x"}]
    assert agent.filter_models_by_vram_budget(models, None, gpu_vram_mib=8_192) == (models, [])
    assert agent.filter_models_by_vram_budget(models, {"x": 10**6}, gpu_vram_mib=0) == (models, [])


def test_agent_refuses_to_start_when_no_model_fits_the_vram_budget(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(agent.API_KEY_ENV, "hm-key")
    gateway_seen: list[str] = []
    code = agent.main(
        ["--gateway", GATEWAY, "--endpoint", ENDPOINT, "--model", "big=qwen3:14b", "--cost-per-1k", "1",
         "--capacity-1k", "10", "--skip-preflight", "--once", "--gpu-vram-mib", "15360", "--min-free-vram-mib", "4300",
         "--pinned-vram-mib", "4000"],
        transport=_json_transport({}, gateway_seen),
        residency_transport=_json_transport({"/api/tags": (200, {"models": [{"name": "qwen3:14b", "size": 9_000 * MIB}]})}),
    )
    assert code == 1 and gateway_seen == []  # no heartbeat advertises a model that would breach the floor
    err = capsys.readouterr().err
    assert "dropping models" in err and "refusing to start" in err


# ---------------------------------------------------------------------------
# Catalog, remote config, preflight, control state
# ---------------------------------------------------------------------------


def test_catalog_and_remote_config_refusals() -> None:
    with httpx.Client(transport=_json_transport({
        "/v1/hivemind/catalog": (200, {"tiers": []}),
        agent.REMOTE_CONFIG_PATH: (200, {"remote_config": {}}),
    })) as client:
        with pytest.raises(TypeError):
            agent.fetch_catalog(object(), GATEWAY)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="gateway"):
            agent.fetch_catalog(client, " ")
        with pytest.raises(ValueError, match="no tiers"):
            agent.fetch_catalog(client, GATEWAY)
        with pytest.raises(TypeError):
            agent.fetch_remote_config(object(), GATEWAY, "k")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="gateway"):
            agent.fetch_remote_config(client, "", "k")
        with pytest.raises(ValueError, match="api_key"):
            agent.fetch_remote_config(client, GATEWAY, " ")
        with pytest.raises(ValueError, match="remote_config"):
            agent.fetch_remote_config(client, GATEWAY, "k")

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"remote_config": {"node_state": "active", "min_price_per_1k": 0.5}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        rc = agent.fetch_remote_config(client, GATEWAY + "/", "hm-secret")
    assert rc == {"node_state": "active", "min_price_per_1k": 0.5}
    assert seen[0].headers["Authorization"] == "Bearer hm-secret"
    assert str(seen[0].url) == GATEWAY + agent.REMOTE_CONFIG_PATH


def test_preflight_lists_missing_names_and_rejects_non_lists() -> None:
    with httpx.Client(transport=_json_transport({"/v1/models": (200, {"data": [{"id": "qwen3:latest"}, "junk"]})})) as c:
        assert agent.preflight(c, ENDPOINT, None, ["qwen3", "coder"]) == ["coder"]
        with pytest.raises(TypeError, match="local_names"):
            agent.preflight(c, ENDPOINT, None, "qwen3")
    with httpx.Client(transport=_json_transport({"/v1/models": (200, {"data": {"id": "x"}})})) as c:
        with pytest.raises(ValueError, match="model list"):
            agent.preflight(c, ENDPOINT, "tok", ["x"])


def test_control_state_write_cleans_up_its_temp_file_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "is_a_dir"
    target.mkdir()  # os.replace onto a directory fails after the temp file was written
    with pytest.raises(OSError):
        agent.write_control_state(str(target), {"paused": True, "vram_cap_mib": 2048})
    assert [p.name for p in tmp_path.iterdir()] == ["is_a_dir"]


# ---------------------------------------------------------------------------
# Routed upstream residency (Crowd-v1 producer)
# ---------------------------------------------------------------------------


def test_routed_upstreams_report_loopback_crowd_producers_as_resident(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(agent.API_KEY_ENV, "hm-key")
    beats: list[dict[str, Any]] = []

    def gateway(request: httpx.Request) -> httpx.Response:
        if request.url.path == agent.HEARTBEAT_PATH:
            beats.append(json.loads(request.content))
            return httpx.Response(200, json={"status": "live"})
        return httpx.Response(404)

    local_seen: list[str] = []

    def local(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        local_seen.append(url)
        if url == "http://127.0.0.1:8791/healthz":
            return httpx.Response(200, json={"ok": True, "device": {"estimated_model_vram_mib": 1350}})
        if url == "http://127.0.0.1:8792/healthz":
            return httpx.Response(503)
        if url == "http://127.0.0.1:8792/v1/models":
            return httpx.Response(200, json={"data": []})
        if url == "http://localhost:8793/healthz":
            return httpx.Response(200, json={"device": {"estimated_model_vram_mib": -4}})
        if url == "http://127.0.0.1:8794/healthz":
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(500)  # Ollama /api/ps is down: residency comes only from routed upstreams

    code = agent.main(
        ["--gateway", GATEWAY, "--endpoint", ENDPOINT,
         "--model", "hivemind/crowd-v1=hf.co/Vxtzq/Crowd-v1",
         "--model", "hivemind/gcpcrowd=gcpcrowd",
         "--model", "hivemind/odd=odd",
         "--model", "hivemind/down=down",
         "--model", "hivemind/remote=remote",
         "--cost-per-1k", "1", "--capacity-1k", "10", "--skip-preflight", "--once", "--gpu-vram-mib", "8192",
         "--routed-upstream", "hf.co/Vxtzq/Crowd-v1=http://127.0.0.1:8791/",
         "--routed-upstream", "gcpcrowd=http://127.0.0.1:8792",
         "--routed-upstream", "odd=http://localhost:8793",
         "--routed-upstream", "down=http://127.0.0.1:8794",
         "--routed-upstream", "remote=http://10.0.0.5:8795",
         "--routed-upstream", "no-separator"],
        transport=httpx.MockTransport(gateway),
        residency_transport=httpx.MockTransport(local),
    )
    assert code == 0 and len(beats) == 1
    by_local = {m["local_name"]: m for m in beats[0]["models"]}
    assert by_local["hf.co/Vxtzq/Crowd-v1"]["loaded"] is True and by_local["hf.co/Vxtzq/Crowd-v1"]["vram_mib"] == 1350
    assert by_local["gcpcrowd"]["loaded"] is True and by_local["gcpcrowd"]["vram_mib"] == 0  # /v1/models fallback
    assert by_local["odd"]["loaded"] is True and by_local["odd"]["vram_mib"] == 0  # negative estimate clamped
    assert by_local["down"]["loaded"] is False and "cold_start_s" in by_local["down"]
    assert by_local["remote"]["loaded"] is False
    assert not any("10.0.0.5" in u for u in local_seen)  # non-loopback upstreams are never probed
    assert beats[0]["gpu_vram_mib"] == 8192

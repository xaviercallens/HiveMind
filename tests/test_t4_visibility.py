# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for honest physical GPU counting and shared-host endpoint attribution (#402)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.router.t4_visibility import build_t4_visibility_payload

SECRET = b"honest-physical-gpu-counting-secret-32b!"


def test_two_endpoints_on_one_host_count_as_one_physical_gpu(tmp_path: Path) -> None:
    store = RouterStore(tmp_path / "router.db", secret=SECRET)
    now = 1_730_000_000.0
    aid_coder, _ = store.register("gcp-t4-producer", "producer")
    aid_bee, _ = store.register("gcp-t4-bee", "producer")

    store.upsert_producer(
        aid_coder,
        "https://34-48-48-154.sslip.io",
        [{"id": "coder", "local_name": "qwen2.5-coder:7b-instruct"}],
        0.40,
        60.0,
        "t4",
        now_utc=now,
        residency={"gpu_vram_mib": 15360, "models": {"coder": {"loaded": True, "vram_mib": 5460, "cold_start_s": 0.0}}},
    )
    store.upsert_producer(
        aid_bee,
        "https://34-48-48-154.sslip.io/bee",
        [{"id": "bee-stars", "local_name": "hivemind-bee-stars"}],
        0.40,
        30.0,
        "t4",
        now_utc=now,
        residency={"gpu_vram_mib": 15360, "models": {"bee-stars": {"loaded": True, "vram_mib": 4832, "cold_start_s": 0.0}}},
    )

    scale = store.producer_scale_status(now_utc=now, ttl_s=120.0, target_producers=100)
    assert scale["distinct_physical_gpus"] == 1
    assert scale["verified_vram_hosts"] == 1
    assert scale["live_producer_accounts"] == 2
    assert scale["live_endpoints"] == 2
    assert scale["shared_host_endpoints"] == 2
    assert scale["scale_readiness_pct"] == 1.0
    assert scale["target_reached"] is False

    fleet = store.list_hiveminder_accounts(now_utc=now, ttl_s=120.0)
    by_handle = {row["handle"]: row for row in fleet}
    assert by_handle["gcp-t4-producer"]["host_root"] == "34-48-48-154.sslip.io"
    assert by_handle["gcp-t4-producer"]["shared_host"] is True
    assert by_handle["gcp-t4-bee"]["host_root"] == "34-48-48-154.sslip.io"
    assert by_handle["gcp-t4-bee"]["shared_host"] is True


def test_independent_volunteer_host_increments_distinct_physical_gpus(tmp_path: Path) -> None:
    store = RouterStore(tmp_path / "router.db", secret=SECRET)
    now = 1_730_000_000.0
    aid_t4, _ = store.register("gcp-t4-producer", "producer")
    aid_vol, _ = store.register("tibrib", "prosumer")

    store.upsert_producer(
        aid_t4,
        "https://34-48-48-154.sslip.io",
        [{"id": "coder", "local_name": "qwen2.5-coder:7b-instruct"}],
        0.40,
        60.0,
        "t4",
        now_utc=now,
        residency={"gpu_vram_mib": 15360, "models": {"coder": {"loaded": True, "vram_mib": 5460, "cold_start_s": 0.0}}},
    )
    store.upsert_producer(
        aid_vol,
        "https://links-winter-swing-pubs.trycloudflare.com",
        [{"id": "coder", "local_name": "qwen2.5-coder:7b"}],
        0.50,
        24.0,
        "ollama",
        now_utc=now,
        residency={"gpu_vram_mib": 12282, "models": {"coder": {"loaded": True, "vram_mib": 4535, "cold_start_s": 12.0}}},
    )

    scale = store.producer_scale_status(now_utc=now, ttl_s=120.0, target_producers=100)
    assert scale["distinct_physical_gpus"] == 2
    assert scale["verified_vram_hosts"] == 2
    assert scale["live_producer_accounts"] == 2
    assert scale["shared_host_endpoints"] == 0
    assert scale["scale_readiness_pct"] == 2.0


def test_t4_status_omits_secondary_gpu_when_single_physical_host(tmp_path: Path) -> None:
    payload = build_t4_visibility_payload(
        now_utc=1_730_030_000.0,
        live_models=["coder", "bee-stars"],
        active_producers_count=2,
        distinct_physical_gpus=1,
    )
    assert payload["sole_active_gpu"] is True
    assert payload["distinct_physical_gpus"] == 1
    assert "secondary_gpu" not in payload["hardware"]
    for proc in payload["hardware"]["resident_processes"]:
        assert "RTX 2070" not in proc["name"]

    store = RouterStore(tmp_path / "router.db", secret=SECRET)
    now = 1_730_030_000.0
    aid1, _ = store.register("gcp-t4-producer", "producer")
    aid2, _ = store.register("gcp-t4-bee", "producer")
    store.upsert_producer(aid1, "https://34-48-48-154.sslip.io", [{"id": "coder", "local_name": "qwen"}], 0.4, 60.0, "t4", now_utc=now)
    store.upsert_producer(aid2, "https://34-48-48-154.sslip.io/bee", [{"id": "bee-stars", "local_name": "bee"}], 0.4, 30.0, "t4", now_utc=now)

    app = create_router_app(store, clock=lambda: now, allow_private_producers=True)
    client = TestClient(app)

    r_t4 = client.get("/v1/hivemind/t4/status")
    assert r_t4.status_code == 200
    body_t4 = r_t4.json()
    assert body_t4["sole_active_gpu"] is True
    assert body_t4["distinct_physical_gpus"] == 1
    assert "secondary_gpu" not in body_t4["hardware"]

    r_scale = client.get("/v1/hivemind/producers/scale-100-status")
    assert r_scale.status_code == 200
    live_scale = r_scale.json()["live_store_scale_status"]
    assert live_scale["distinct_physical_gpus"] == 1
    assert live_scale["live_endpoints"] == 2
    assert live_scale["shared_host_endpoints"] == 2
    assert live_scale["scale_readiness_pct"] == 1.0

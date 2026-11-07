# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression tests for PR-IP-17 (IP-99..IP-106, IP-126, IP-129, IP-130, IP-131)."""

from __future__ import annotations

import hashlib
import importlib.util
import shlex
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hivemind.cli.community_gpu_benchmark import (
    run_community_gpu_benchmark,
    verify_community_benchmark_receipt,
)
from hivemind.client.steam_identity import (
    OPENID_ENDPOINT,
    OPENID_NS,
    REQUIRED_SIGNED,
    IdentityError,
    SteamOpenIDVerifier,
)
from hivemind.router import RouterStore, create_router_app
from hivemind.router.hivemind_producer_agent import build_parser as build_agent_parser
from hivemind.router.opentrainer import build_opentrainer_status_payload
from hivemind.swarm.rusty_sundials_scientific_bee import (
    run_wave2_parameter_sweep_quorum,
    verify_wave2_quorum_replicas,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET = b"pr-ip17-router-client-deploy-secret-32b!"


def test_ip99_opentrainer_delta_vram_floor_and_distinct_gpus_bounded_by_account() -> None:
    store = RouterStore(":memory:", SECRET)
    client = TestClient(create_router_app(store))

    reg = client.post("/v1/hivemind/register", json={"handle": "ip99_trainer", "role": "trainer"})
    assert reg.status_code == 200
    key = reg.json()["api_key"]
    headers = {"Authorization": f"Bearer {key}"}

    # Submit delta declaring vram_free_mib=100 (below the 1536 MiB safety floor) -> rejected with 400
    low_vram = client.post(
        "/v1/hivemind/opentrainer/submit-delta",
        headers=headers,
        json={
            "round_id": 1,
            "campaign_id": "E8-STAGE2-TINY-GRANITE-MOE-QLORA",
            "loss_before": 3.2,
            "loss_after": 2.8,
            "delta_norm": 0.12,
            "delta_sha256": hashlib.sha256(b"ip99-low-vram").hexdigest(),
            "vram_free_mib": 100,
            "vram_total_mib": 8192,
        },
    )
    assert low_vram.status_code == 400
    assert low_vram.json()["error"]["code"] == "delta_rejected"

    # 50 distinct host_labels from one account count as 1 distinct_physical_trainer_gpus
    now = 1_800_000_000.0
    fifty_nodes = [
        {"account_id": "acc-single-owner", "host_label": f"host-{i}", "node_id": f"n-{i}", "last_heartbeat_utc": now - 10.0}
        for i in range(50)
    ]
    status = build_opentrainer_status_payload(
        trainer_accounts=[{"account_id": "acc-single-owner", "handle": "ip99_trainer"}],
        opentrainer_nodes=fifty_nodes,
        recent_rounds=[],
        now_utc=now,
    )
    assert status["live_opentrainer_fleet"]["distinct_physical_trainer_gpus"] == 1


def test_ip100_sslip_io_subdomains_collapse_to_single_gpu_and_scale100_tracks_live_accounts(tmp_path: Path) -> None:
    store = RouterStore(tmp_path / "ip100.db", SECRET)
    now = 1_760_000_000.0
    for i in range(20):
        aid, _ = store.register(f"sslip_node_{i}", "producer")
        store.upsert_producer(
            aid,
            f"https://node-{i}.34-48-48-154.sslip.io",
            [{"id": "coder", "local_name": "qwen2.5-coder:7b"}],
            0.40,
            25.0,
            "t4",
            now_utc=now,
            residency={"gpu_vram_mib": 15360, "models": {"coder": {"loaded": True, "vram_mib": 5460, "cold_start_s": 0.0}}},
        )

    scale = store.producer_scale_status(now_utc=now, ttl_s=120.0, target_producers=20)
    assert scale["distinct_physical_gpus"] == 1
    assert scale["target_reached"] is False

    client = TestClient(create_router_app(store, clock=lambda: now))
    s1 = client.get("/v1/hivemind/producers/scale-100-status").json()
    aid_extra, _ = store.register("independent_gpu_host", "producer")
    store.upsert_producer(
        aid_extra,
        "https://independent-volunteer.trycloudflare.com",
        [{"id": "coder", "local_name": "qwen2.5-coder:7b"}],
        0.40,
        25.0,
        "t4",
        now_utc=now,
    )
    s2 = client.get("/v1/hivemind/producers/scale-100-status").json()
    assert s2["hundred_producer_fleet_plan"]["live_producer_accounts"] == s1["hundred_producer_fleet_plan"]["live_producer_accounts"] + 1


def test_ip101_t4_status_reports_zero_producers_and_gpus_when_store_empty() -> None:
    store = RouterStore(":memory:", SECRET)
    client = TestClient(create_router_app(store))
    resp = client.get("/v1/hivemind/t4/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active_producers_count"] == 0
    assert body["distinct_physical_gpus"] == 0


def test_ip102_wave2_quorum_detects_corrupted_replica_state_value() -> None:
    bundle = run_wave2_parameter_sweep_quorum(family="van_der_pol", steps=12, dt=0.05)
    assert bundle["quorum_verified"] is True
    assert verify_wave2_quorum_replicas(bundle["spec"], bundle["replica_a"], bundle["replica_b"]) is True
    # Confirm replica_b is computed via an independent RK4 sub-stepping pass, not y_a + 1e-11
    ya0 = bundle["replica_a"]["sweep_results"][0]["y"][0][0]
    yb0 = bundle["replica_b"]["sweep_results"][0]["y"][0][0]
    assert abs((yb0 - ya0) - 1e-11) > 1e-14

    # Corrupt a single state value in replica_a by 1e-3 -> quorum_verified must be False
    bundle["replica_a"]["sweep_results"][0]["y"][0][0] += 1e-3
    assert verify_wave2_quorum_replicas(bundle["spec"], bundle["replica_a"], bundle["replica_b"]) is False


def test_ip103_admin_reconnect_without_handles_leaves_third_party_volunteers_untouched() -> None:
    now = 1_760_000_000.0
    store = RouterStore(":memory:", SECRET)
    owner_id, _ = store.register("gcp-t4-producer", "producer")
    third_party_id, _ = store.register("independent_volunteer_alice", "producer")
    store.upsert_producer(
        third_party_id,
        "https://alice-own-rig.trycloudflare.com",
        [{"id": "coder", "local_name": "qwen2.5-coder:7b"}],
        0.55,
        30.0,
        "rtx3090",
        now_utc=now,
    )

    admin_key = "hm-admin-ip103-secret-key"
    client = TestClient(
        create_router_app(
            store,
            admin_key_hash=store.key_digest(admin_key),
            clock=lambda: now,
            allow_private_producers=True,
        )
    )
    rec = client.post(
        "/v1/hivemind/admin/producers/reconnect",
        headers={"X-Admin-Key": admin_key},
        json={
            "endpoint_url": "https://34-48-48-154.sslip.io",
            "endpoint_token": "ip103-admin-endpoint-token-secret",
            "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b-instruct"}],
        },
    )
    assert rec.status_code == 200
    body = rec.json()
    assert owner_id in body["reconnected_account_ids"]
    assert third_party_id not in body["reconnected_account_ids"]

    admin_rows = {
        r["handle"]: r
        for r in client.get("/v1/hivemind/admin/producers", headers={"X-Admin-Key": admin_key}).json()["hiveminders"]
    }
    assert admin_rows["independent_volunteer_alice"]["endpoint_url"] == "https://alice-own-rig.trycloudflare.com"


def test_ip104_public_fleet_omits_account_id_and_host_root_and_heartbeat_protects_token_owned_url() -> None:
    now = 1_760_000_000.0
    store = RouterStore(":memory:", SECRET)
    admin_key = "hm-admin-ip104-key"
    client = TestClient(
        create_router_app(
            store,
            admin_key_hash=store.key_digest(admin_key),
            clock=lambda: now,
            allow_private_producers=True,
        )
    )

    r1 = client.post("/v1/hivemind/register", json={"handle": "owner_prod", "role": "producer"}).json()
    r2 = client.post("/v1/hivemind/register", json={"handle": "attacker_prod", "role": "producer"}).json()

    hb_ok = client.post(
        "/v1/hivemind/producer/heartbeat",
        headers={"Authorization": f"Bearer {r1['api_key']}"},
        json={
            "endpoint_url": "https://34-48-48-154.sslip.io",
            "endpoint_token": "owner-secret-endpoint-token-12345",
            "models": [{"id": "coder", "local_name": "qwen2.5-coder:7b"}],
            "cost_per_1k": 0.4,
            "capacity_1k": 50.0,
            "execution_class": "t4",
        },
    )
    assert hb_ok.status_code == 200

    # Attacker without matching endpoint_token cannot hijack the URL
    hb_hijack = client.post(
        "/v1/hivemind/producer/heartbeat",
        headers={"Authorization": f"Bearer {r2['api_key']}"},
        json={
            "endpoint_url": "https://34-48-48-154.sslip.io",
            "models": [{"id": "coder", "local_name": "attacker-model"}],
            "cost_per_1k": 0.1,
            "capacity_1k": 50.0,
            "execution_class": "t4",
        },
    )
    assert hb_hijack.status_code == 403
    assert hb_hijack.json()["error"]["code"] == "endpoint_ownership_required"

    # Public /v1/hivemind/producers/fleet omits account_id and host_root
    pub_fleet = client.get("/v1/hivemind/producers/fleet").json()
    for row in pub_fleet["hiveminders"]:
        assert "account_id" not in row
        assert "host_root" not in row


def test_ip105_ruche_overlay_has_no_fabricated_members_and_blocks_on_corrupt_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = importlib.util.spec_from_file_location(
        "ruche_overlay_server", REPO_ROOT / "deploy" / "ruche_router_overlay" / "server.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    state_file = tmp_path / "community_hub_state.json"
    monkeypatch.setattr(mod, "_COMMUNITY_STATE_PATH", state_file)

    client = TestClient(mod.app)
    fresh = client.get("/api/v1/community/hub")
    assert fresh.status_code == 200
    assert fresh.json()["community_members"] == []
    assert fresh.json()["coop_discovery_feed"] == []

    # Corrupt the state file -> must return 503 with status="BLOCKED" and NOT overwrite the corrupt file
    state_file.write_text("{corrupt-json", encoding="utf-8")
    corrupt_resp = client.get("/api/v1/community/hub")
    assert corrupt_resp.status_code == 503
    assert corrupt_resp.json()["status"] == "BLOCKED"
    assert state_file.read_text(encoding="utf-8") == "{corrupt-json"


def test_ip106_install_node_sh_has_exit_trap_root_ownership_and_atomic_committed_write() -> None:
    script = (REPO_ROOT / "deploy" / "t4_node" / "install_node.sh").read_text(encoding="utf-8")
    assert "trap lock EXIT" in script
    assert "chown -R root:root \"$OPT\"" in script
    assert "cat-file -e \"HEAD:$rel\"" in script
    assert "ls-tree --name-only HEAD deploy/t4_node/node/" in script


def test_ip126_and_ip130_steam_nonce_recorded_only_after_valid_and_boinc_receipts_persist(tmp_path: Path) -> None:
    # 1. 5,000 fake unauthenticated OpenID assertions where Steam returns is_valid:false must NOT fill _seen
    now = 1_762_000_000.0
    is_valid_flag = {"valid": False}

    def fake_http(method: str, url: str, data: dict[str, str]) -> tuple[int, str]:
        body = "ns:http://specs.openid.net/auth/2.0\nis_valid:true\n" if is_valid_flag["valid"] else "ns:http://specs.openid.net/auth/2.0\nis_valid:false\n"
        return 200, body

    verifier = SteamOpenIDVerifier(
        "https://hivemind.example/callback",
        http=fake_http,
        clock=lambda: now,
        max_nonces=100,
    )
    base_params = {
        "openid.ns": OPENID_NS,
        "openid.mode": "id_res",
        "openid.op_endpoint": OPENID_ENDPOINT,
        "openid.return_to": "https://hivemind.example/callback",
        "openid.assoc_handle": "123456",
        "openid.signed": ",".join(sorted(REQUIRED_SIGNED)),
        "openid.claimed_id": "https://steamcommunity.com/openid/id/76561198000000001",
        "openid.identity": "https://steamcommunity.com/openid/id/76561198000000001",
        "openid.sig": "sig",
    }
    for i in range(500):
        with pytest.raises(IdentityError):
            verifier.verify_assertion({**base_params, "openid.response_nonce": f"2025-11-01T12:26:40Zfake{i}"})

    # Real user sign-in immediately succeeds because fake assertions did not consume nonce slots
    is_valid_flag["valid"] = True
    steam_id = verifier.verify_assertion({**base_params, "openid.response_nonce": "2025-11-01T12:26:40Zreal0"})
    assert steam_id == "76561198000000001"

    # 2. BOINC/WebGPU receipt replay detection survives app restart via SQLite
    db_file = tmp_path / "router_receipts.db"
    store1 = RouterStore(db_file, SECRET)
    _, key = store1.register("boinc_player", "prosumer")
    client1 = TestClient(create_router_app(store1))
    fixed_receipt = hashlib.sha256(b"persistent-boinc-receipt-1").hexdigest()
    first = client1.post(
        "/v1/hivemind/steam/contribute-boinc",
        headers={"Authorization": f"Bearer {key}"},
        json={"target_planet": "K2-18 b", "receipt_sha256": fixed_receipt},
    )
    assert first.status_code == 200
    assert first.json()["duplicate_receipt"] is False
    store1.close()

    # Recreate store and app on the same SQLite DB -> replaying the same receipt is detected as duplicate
    store2 = RouterStore(db_file, SECRET)
    client2 = TestClient(create_router_app(store2))
    replay = client2.post(
        "/v1/hivemind/steam/contribute-boinc",
        headers={"Authorization": f"Bearer {key}"},
        json={"target_planet": "K2-18 b", "receipt_sha256": fixed_receipt},
    )
    assert replay.status_code == 200
    assert replay.json()["duplicate_receipt"] is True
    store2.close()

    # 3. deploy/gcp_openrouter/deploy.sh uses --update-env-vars
    deploy_sh = (REPO_ROOT / "deploy" / "gcp_openrouter" / "deploy.sh").read_text(encoding="utf-8")
    assert "--update-env-vars" in deploy_sh
    assert "--set-env-vars" not in deploy_sh


def test_ip129_remote_config_endpoint_and_systemd_service_cli_flags_parse() -> None:
    store = RouterStore(":memory:", SECRET)
    client = TestClient(create_router_app(store))
    reg = client.post("/v1/hivemind/register", json={"handle": "ip129_node", "role": "producer"}).json()
    headers = {"Authorization": f"Bearer {reg['api_key']}"}

    r_get = client.get("/v1/hivemind/remote/config", headers=headers)
    assert r_get.status_code == 200
    assert r_get.json()["remote_config"]["node_state"] == "active"

    r_post = client.post(
        "/v1/hivemind/remote/config",
        headers=headers,
        json={"node_state": "paused", "vram_cap_mib": 6144},
    )
    assert r_post.status_code == 200
    assert r_post.json()["remote_config"]["node_state"] == "paused"
    assert r_post.json()["remote_config"]["vram_cap_mib"] == 6144

    # Parse ExecStart command lines in deploy/t4_node/live/*.service and deploy/t4_node/node/*.service
    import sys

    if str(REPO_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import hivemind_producer_shield as shield_mod

    agent_parser = build_agent_parser()
    shield_parser = shield_mod.build_parser()
    for service_dir in (REPO_ROOT / "deploy" / "t4_node" / "live", REPO_ROOT / "deploy" / "t4_node" / "node"):
        for unit_path in sorted(service_dir.glob("*.service")):
            for line in unit_path.read_text(encoding="utf-8").splitlines():
                if not line.startswith("ExecStart="):
                    continue
                cmd = line.removeprefix("ExecStart=").strip()
                tokens = shlex.split(cmd)
                if any("hivemind_producer_agent.py" in t for t in tokens):
                    idx = next(i for i, t in enumerate(tokens) if "hivemind_producer_agent.py" in t)
                    agent_parser.parse_args(tokens[idx + 1 :])
                elif any("hivemind_producer_shield.py" in t for t in tokens):
                    idx = next(i for i, t in enumerate(tokens) if "hivemind_producer_shield.py" in t)
                    shield_parser.parse_args(tokens[idx + 1 :])


def test_ip131_community_gpu_benchmark_cpu_honesty_and_receipt_tamper_detection(tmp_path: Path) -> None:
    receipt = run_community_gpu_benchmark(
        contributor_handle="cpu-honesty-tester",
        preferred_device="cpu",
        token_steps=2,
        router_tokens=16,
        hidden_size=16,
        output_path=tmp_path / "receipt.json",
    )
    assert receipt["hardware"]["measured_on_gpu"] is False
    assert receipt["gpu_points_awarded"] == 0
    assert receipt["overall_passed"] is True
    assert verify_community_benchmark_receipt(receipt) is True

    # Tampering with insurance_pool_reserve_hive invalidates verify_community_benchmark_receipt
    tampered = dict(receipt)
    tampered["stage4_zk_opoi_v4_and_insurance"] = {
        **receipt["stage4_zk_opoi_v4_and_insurance"],
        "insurance_pool_reserve_hive": 999.0,
    }
    assert verify_community_benchmark_receipt(tampered) is False

    # Mutating pool_holdings away from 200.0 flips overall_passed to False
    failed_pool = run_community_gpu_benchmark(
        contributor_handle="pool-mismatch-tester",
        preferred_device="cpu",
        token_steps=2,
        router_tokens=16,
        hidden_size=16,
        output_path=None,
        pool_holdings_override=150.0,
    )
    assert failed_pool["overall_passed"] is False
    assert verify_community_benchmark_receipt(failed_pool) is False

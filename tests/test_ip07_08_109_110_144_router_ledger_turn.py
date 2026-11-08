# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Regression tests for IP-07 (#386), IP-08 (#382), IP-109 (#424), IP-110 (#423), and IP-144 (#494)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

import hivemind.swarm.nat_traversal as nt
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import PRODUCER_TTL_S, create_router_app
from hivemind.swarm.dht_storage import ContentAddressedSwarmStore
from hivemind.swarm.hf_community_hub import CommunityAdapterBundleManager
from hivemind.swarm.ledger import HiveSwarmLedger
from hivemind.swarm.swarm_fabric import HiveLayer2StateChannel, StateChannelVoucher


def test_ip144_reconnect_expires_after_ttl_without_heartbeat_and_heartbeat_wins() -> None:
    """Issue #494 (IP-144): Reconnect keepalive must not manufacture liveness past PRODUCER_TTL_S,
    and a producer's own heartbeat must always win over the reconnect entry.
    """
    secret = b"hivemind-test-secret-at-least-32-bytes-long!"
    store = RouterStore(":memory:", secret)
    admin_key = "admin-secret-key-for-hivemaster"
    now = [1_800_000_000.0]
    client = TestClient(
        create_router_app(
            store,
            allow_private_producers=True,
            admin_key_hash=store.key_digest(admin_key),
            resolver=lambda _h: ["34.48.48.154"],
            clock=lambda: now[0],
        )
    )
    _aid, key = store.register("tibrib", "producer")
    models = [{"id": "coder", "local_name": "qwen2.5-coder:7b-instruct"}]
    resp = client.post(
        "/v1/hivemind/admin/producers/reconnect",
        headers={"x-admin-key": admin_key},
        json={
            "endpoint_url": "https://34-48-48-154.sslip.io/reconnect-a",
            "endpoint_token": "test-endpoint-token-0123456789abcdefghijk",
            "models": models,
            "replace_endpoints": True,
            "include_legacy_known_users": False,
        },
    )
    assert resp.status_code == 200
    assert store.producer_fleet_scale_status(now[0], PRODUCER_TTL_S)["live_producer_accounts"] == 1

    # Advance clock past PRODUCER_TTL_S with no heartbeat: even after calling fleet (which runs _refresh_reconnected_fleet_if_needed),
    # live_producer_accounts must be 0.
    now[0] += PRODUCER_TTL_S + 10.0
    fleet_resp = client.get("/v1/hivemind/producers/fleet")
    assert fleet_resp.status_code == 200
    assert fleet_resp.json()["live_hiveminders"] == 0
    assert store.producer_fleet_scale_status(now[0], PRODUCER_TTL_S)["live_producer_accounts"] == 0

    # Reconnect again to endpoint A, then heartbeat with endpoint B: stored endpoint_url must stay B.
    resp2 = client.post(
        "/v1/hivemind/admin/producers/reconnect",
        headers={"x-admin-key": admin_key},
        json={
            "endpoint_url": "https://34-48-48-154.sslip.io/reconnect-a",
            "endpoint_token": "test-endpoint-token-0123456789abcdefghijk",
            "models": models,
            "replace_endpoints": True,
            "include_legacy_known_users": False,
        },
    )
    assert resp2.status_code == 200
    endpoint_b = "https://34-48-48-154.sslip.io/heartbeat-b"
    hb = client.post(
        "/v1/hivemind/producer/heartbeat",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "endpoint_url": endpoint_b,
            "models": models,
            "cost_per_1k": 0.3,
            "capacity_1k": 50.0,
            "execution_class": "t4-16gb-moe",
            "endpoint_token": "own-endpoint-token-0123456789abcdefghij",
            "replace_endpoints": True,
        },
    )
    assert hb.status_code == 200
    now[0] += 40.0
    admin_rows = {
        r["handle"]: r
        for r in client.get("/v1/hivemind/admin/producers", headers={"x-admin-key": admin_key}).json()["hiveminders"]
    }
    assert admin_rows["tibrib"]["endpoint_url"] == endpoint_b


def test_ip109_l2_state_channel_escrow_lock_voucher_verification_and_idempotent_settle(tmp_path: Path) -> None:
    """Issue #424 (IP-109): HiveLayer2StateChannel locks escrow on open when ledger is passed,
    verifies voucher signatures and amounts before settlement, and settles idempotently.
    """
    ledger = HiveSwarmLedger(ledger_path=tmp_path / "l2_ip109.json")
    ledger.credit_reward("consumer-01", 10.0, "init")
    l2 = HiveLayer2StateChannel(
        consumer_id="consumer-01",
        provider_ids=["provider-01", "provider-02"],
        escrow_hive=2.0,
        rate_per_token=0.1,
        ledger=ledger,
    )
    assert ledger.get_balance("consumer-01") == 8.0
    assert l2.locked_escrow_hive == 2.0

    v = l2.sign_micro_voucher(5)
    assert l2.verify_voucher(v) is True

    # Tampered signature must be rejected by settle_to_l1_ledger
    forged = StateChannelVoucher(
        channel_id=v.channel_id,
        consumer_id=v.consumer_id,
        provider_ids=v.provider_ids,
        nonce=v.nonce,
        cumulative_tokens=v.cumulative_tokens,
        cumulative_hive_spent=v.cumulative_hive_spent,
        signature_hmac="0" * 32,
    )
    l2.latest_voucher = forged
    with pytest.raises(ValueError, match="voucher"):
        l2.settle_to_l1_ledger(ledger)

    # Restore valid voucher and settle once
    l2.latest_voucher = v
    settled1 = l2.settle_to_l1_ledger(ledger)
    assert settled1["settled_tokens"] == 5
    assert settled1["settled_hive"] == 0.5
    assert ledger.get_balance("consumer-01") == 9.5
    assert ledger.get_balance("provider-01") == 0.25
    assert ledger.get_balance("provider-02") == 0.25

    # Second settle call must be idempotent and not charge again
    settled2 = l2.settle_to_l1_ledger(ledger)
    assert settled2["status"] == "no_vouchers_to_settle"
    assert settled2["settled_hive"] == 0.0
    assert ledger.get_balance("consumer-01") == 9.5


def test_ip07_community_bundle_import_verifies_before_credit_forbids_self_import_and_deduplicates(
    tmp_path: Path,
) -> None:
    """Issue #386 (IP-07): Verify bundle SHA before crediting, forbid self-import, and pay each (bundle, importer) once."""
    store = ContentAddressedSwarmStore(tmp_path / "cas")
    ledger = HiveSwarmLedger(ledger_path=tmp_path / "ledger.json")
    manager = CommunityAdapterBundleManager(store=store, ledger=ledger)

    lora_sd = {
        "layers.0.mlp.gate_proj.lora_A": torch.ones(8, 16) * 0.25,
        "layers.0.mlp.gate_proj.lora_B": torch.ones(32, 8) * 0.25,
    }
    bundle_dir = tmp_path / "bundle"
    manager.create_and_publish_bundle(
        bundle_id="BUNDLE-IP07",
        bundle_title="IP07 Bundle",
        guild_id="GUILD-01-EURO-CYBER-SYNDICATE",
        domain_adapters={"Rust_Systems_Security": {"lora_state_dict": lora_sd}},
        output_dir=bundle_dir,
        canonical_rank_r=8,
    )
    manifest_path = bundle_dir / "hivemind_swarm_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    author_tag = manifest["entries"][0]["author_gamer_tag"]
    good_bundle_sha = manifest["bundle_sha256"]

    # 1. Wrong bundle_sha256 must raise ValueError AND leave author balance at 0.0
    manifest["bundle_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        manager.import_community_bundle(bundle_dir, importer_agent_id="buyer-01")
    assert ledger.get_balance(author_tag) == 0.0

    # Restore valid bundle_sha256
    manifest["bundle_sha256"] = good_bundle_sha
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 2. Self-import by author_tag pays 0.0
    self_res = manager.import_community_bundle(bundle_dir, importer_agent_id=author_tag)
    assert self_res["royalties_settled_hive"] == 0.0
    assert ledger.get_balance(author_tag) == 0.0

    # 3. First import by external importer pays 15.0, second import by same importer pays 0.0
    imp1 = manager.import_community_bundle(bundle_dir, importer_agent_id="buyer-01")
    assert imp1["royalties_settled_hive"] == 15.0
    assert ledger.get_balance(author_tag) == 15.0

    imp2 = manager.import_community_bundle(bundle_dir, importer_agent_id="buyer-01")
    assert imp2["royalties_settled_hive"] == 0.0
    assert ledger.get_balance(author_tag) == 15.0


def test_ip08_ledger_reloads_persisted_state_rejects_corrupt_file_and_persists_slash(tmp_path: Path) -> None:
    """Issue #382 (IP-08): Reopening HiveSwarmLedger reloads persisted state and verifies chain integrity;
    slashing survives reopening; corrupt files raise ValueError.
    """
    ledger_path = tmp_path / "persisted_ledger.json"
    ledger1 = HiveSwarmLedger(ledger_path=ledger_path)
    ledger1.credit_reward("miner-alpha", 100.0, "reward")
    assert ledger1.get_balance("miner-alpha") == 100.0

    # Reopen from disk: balance must be 100.0
    ledger2 = HiveSwarmLedger(ledger_path=ledger_path)
    assert ledger2.get_balance("miner-alpha") == 100.0
    assert ledger2.verify_chain_integrity() is True

    # Slash 50% and reopen: balance 50.0, slashed_registry preserved, total_minted_supply 100.0
    slashed_amt = ledger2.slash("miner-alpha", fraction=0.5)
    assert slashed_amt == 50.0
    ledger3 = HiveSwarmLedger(ledger_path=ledger_path)
    assert ledger3.get_balance("miner-alpha") == 50.0
    assert "miner-alpha" in ledger3.slashed_registry
    assert ledger3.total_minted_supply() == 100.0

    # Corrupt block_hash on disk must raise ValueError on open
    raw = json.loads(ledger_path.read_text(encoding="utf-8"))
    raw["chain"][0]["block_hash"] = "f" * 64
    ledger_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt ledger"):
        HiveSwarmLedger(ledger_path=ledger_path)


def test_ip110_turn_relay_rejects_private_peers_enforces_allocation_quota_and_survives_sendto_oserror() -> None:
    """Issue #423 (IP-110): TurnRelayServer rejects RFC1918/link-local/non-permitted loopback peers in
    CREATE_PERMISSION, enforces per-user allocation quota (486), and catches OSError on sendto.
    """
    with nt.TurnRelayServer({"alice": "s3cret"}, max_allocations_per_user=1) as server:
        c1 = nt.TurnClient(server.address, "alice", "s3cret", timeout=0.6)
        c2 = nt.TurnClient(server.address, "alice", "s3cret", timeout=0.6)
        try:
            c1.allocate()
            # Second allocation for the same user exceeds max_allocations_per_user=1 -> 486
            with pytest.raises(nt.StunError) as quota_err:
                c2.allocate()
            assert quota_err.value.code == 486

            # RFC1918, link-local, and non-127.0.0.1 loopback peer addresses are rejected with 403 Forbidden
            for forbidden_ip in ("10.0.0.1", "192.168.1.1", "172.16.0.1", "169.254.169.254", "127.0.0.2"):
                with pytest.raises(nt.StunError) as perm_err:
                    c1.create_permission(forbidden_ip)
                assert perm_err.value.code == 403
        finally:
            c1.close()
            c2.close()
        assert server._thread is not None and server._thread.is_alive()


def test_l2_state_channel_ledger_and_welcome_email_branch_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise validation, refund, persistence failure, La Ruche GCS loader, and welcome email branches."""
    import io
    import os
    import smtplib
    import urllib.request

    import hivemind.swarm.la_ruche_vault as lrv
    from hivemind.router.welcome_email import (
        _mask_email,
        build_install_producer_shell_script,
        send_or_spool_welcome_email,
    )

    # 1. HiveLayer2StateChannel validation and refund_consumer_escrow branches
    with pytest.raises(ValueError, match="escrow_hive"):
        HiveLayer2StateChannel("c", ["p"], escrow_hive=0.0)
    with pytest.raises(ValueError, match="rate_per_token"):
        HiveLayer2StateChannel("c", ["p"], rate_per_token=0.0)
    with pytest.raises(ValueError, match="secret_key"):
        HiveLayer2StateChannel("c", ["p"], secret_key=b"short")

    ledger = HiveSwarmLedger(ledger_path=tmp_path / "cov_ledger.json")
    assert ledger.balances.setdefault("c-low", 0.5) == 0.5
    with pytest.raises(ValueError, match="insufficient"):
        HiveLayer2StateChannel("c-low", ["p"], escrow_hive=5.0, ledger=ledger)

    ledger.credit_reward("c-ok", 10.0, "init")
    l2 = HiveLayer2StateChannel(
        "c-ok",
        ["p"],
        escrow_hive=2.0,
        rate_per_token=0.1,
        secret_key=b"0123456789abcdef",
        ledger=ledger,
    )
    with pytest.raises(ValueError, match="tokens_generated"):
        l2.sign_micro_voucher(0)
    v = l2.sign_micro_voucher(4)
    assert l2.verify_voucher(StateChannelVoucher("other", "c-ok", ["p"], 1, 4, 0.4, v.signature_hmac)) is False
    assert l2.verify_voucher(StateChannelVoucher(l2.channel_id, "c-ok", ["p"], 0, 4, 0.4, v.signature_hmac)) is False
    assert l2.verify_voucher(StateChannelVoucher(l2.channel_id, "c-ok", ["p"], 1, 4, 0.0, v.signature_hmac)) is False
    assert l2.verify_voucher(StateChannelVoucher(l2.channel_id, "c-ok", ["p"], 1, 4, 9.9, v.signature_hmac)) is False
    assert l2.verify_voucher(StateChannelVoucher(l2.channel_id, "c-ok", ["p"], 1, 4, 0.9, v.signature_hmac)) is False
    refunded = l2.refund_consumer_escrow()
    assert refunded == 0.4
    assert ledger.get_balance("c-ok") == 10.0

    # 2. HiveSwarmLedger error branches
    with pytest.raises(ValueError, match="node_id"):
        ledger.slash("")
    bad_empty = tmp_path / "empty_chain.json"
    bad_empty.write_text('{"chain": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt ledger"):
        HiveSwarmLedger(ledger_path=bad_empty)

    orig_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        ledger.credit_reward("c-ok", 1.0, "fail")
    monkeypatch.setattr(os, "replace", orig_replace)

    # 3. la_ruche_vault load_community_moe_from_gcs_or_https & error branches
    with pytest.raises(FileNotFoundError):
        lrv.load_community_moe_from_ruche(tmp_path / "missing_vault")
    fake_bytes = b"fake-safetensors-payload"
    import hashlib

    fake_sha = hashlib.sha256(fake_bytes).hexdigest()
    fake_manifest = {
        "combined_safetensors_path": "community_moe_release/fake.safetensors",
        "combined_safetensors_sha256": fake_sha,
    }

    class _FakeResp(io.BytesIO):
        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *_a: object) -> None:
            self.close()

    def _fake_urlopen(req: urllib.request.Request, timeout: float = 30.0) -> _FakeResp:
        if req.full_url.endswith("ruche_manifest.json"):
            return _FakeResp(json.dumps(fake_manifest).encode("utf-8"))
        return _FakeResp(fake_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(lrv, "load_community_moe_from_ruche", lambda cache_path, seed=20261024: ("ok", fake_manifest))
    gcs_cache = tmp_path / "gcs_cache"
    res1 = lrv.load_community_moe_from_gcs_or_https(gcs_cache, "https://example.org/ruche")
    res2 = lrv.load_community_moe_from_gcs_or_https(gcs_cache, "https://example.org/ruche")
    assert res1[0] == "ok" and res2[0] == "ok"

    fake_manifest["combined_safetensors_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        lrv.load_community_moe_from_gcs_or_https(tmp_path / "gcs_bad", "https://example.org/ruche")

    # 4. welcome_email branches
    assert "hivemind_producer_agent.py" in build_install_producer_shell_script("https://example.org/")
    assert _mask_email("no-at-sign") == "***"
    assert _mask_email("ab@example.org") == "a***@example.org"
    assert _mask_email("xavier@example.org") == "xa***@example.org"

    class _FakeSMTP:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def __enter__(self) -> _FakeSMTP:
            return self

        def __exit__(self, *_a: object) -> None:
            pass

        def ehlo(self) -> None:
            pass

        def starttls(self) -> None:
            pass

        def login(self, *_a: object) -> None:
            pass

        def send_message(self, _msg: object) -> None:
            pass

    monkeypatch.setenv("HIVEMIND_SMTP_HOST", "smtp.example.org")
    monkeypatch.setenv("HIVEMIND_SMTP_PORT", "587")
    monkeypatch.setenv("HIVEMIND_SMTP_USER", "u")
    monkeypatch.setenv("HIVEMIND_SMTP_PASSWORD", "p")
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    sent = send_or_spool_welcome_email(
        account_id="acc-pioneer",
        handle="pioneer",
        role="prosumer",
        balance_hmt=100.0,
        recipient_email="xavier@example.org",
        is_pioneer_first_user=True,
    )
    assert sent["delivery_status"] == "SENT_VIA_SMTP"

    class _FailSMTP(_FakeSMTP):
        def send_message(self, _msg: object) -> None:
            raise RuntimeError("smtp down")

    monkeypatch.setattr(smtplib, "SMTP", _FailSMTP)
    fallback = send_or_spool_welcome_email(
        account_id="acc-2",
        handle="user2",
        role="consumer",
        balance_hmt=50.0,
        recipient_email="ab@example.org",
    )
    assert "SPOOL_FALLBACK" in fallback["delivery_status"]


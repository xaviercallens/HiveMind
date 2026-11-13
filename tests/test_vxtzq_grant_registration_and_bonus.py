# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for @Vxtzq 100,000 HMT model-testing grant, elevated registration grants, and Sybil-bounded bonus claims."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hivemind.boinc.hivemind_at_home_dashboard import _DEFAULT_LEDGER_BALANCES
from hivemind.router.accounts import (
    LEGACY_KNOWN_USERS,
    VXTZQ_MODEL_TESTING_TARGET_HMT,
    RouterStore,
)
from hivemind.router.gateway import create_router_app
from hivemind.router.points_wording import currency_leftovers, to_points_wording

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET = b"vxtzq-registration-and-bonus-test-secret-32bytes!"
ADMIN_KEY = "hm-admin-vxtzq-bonus-test-key"


def _load_openrouter_server(monkeypatch: pytest.MonkeyPatch, **extra_env: str):
    monkeypatch.setenv("HIVEMIND_ROUTER_SECRET", "s" * 40)
    monkeypatch.setenv("HIVEMIND_ROUTER_DB", ":memory:")
    for key in (
        "HIVEMIND_GOOGLE_CLIENT_ID",
        "HIVEMIND_OPEN_REGISTRATION",
        "HIVEMIND_OIDC_PROVIDERS",
        "HIVEMIND_STARTER_GRANT_HMT",
        "HIVEMIND_STARTER_GRANTS_PER_IP_24H",
        "HIVEMIND_REGISTRATION_BONUS_HMT",
        "HIVEMIND_ALLOW_BONUSES",
        "HIVEMIND_BONUS_CLAIMS_PER_IP_24H",
        "HIVEMIND_PARTNER_TESTING_GRANTS",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, val in extra_env.items():
        monkeypatch.setenv(key, val)
    spec = importlib.util.spec_from_file_location(
        "gcp_openrouter_server_bonus_test",
        REPO_ROOT / "deploy" / "gcp_openrouter" / "server.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vxtzq_legacy_and_dashboard_allocation_raised_to_100k_hmt(tmp_path: Path) -> None:
    meta = LEGACY_KNOWN_USERS["vxtzq"]
    assert meta["starter_hmt"] == 100_000.0
    assert VXTZQ_MODEL_TESTING_TARGET_HMT == 100_000.0
    # an address with no confirmed owner must not be an alias (reset mail, credential lookup), owner decision 2026-11-12
    assert "vxtzqmc@gmail.com" not in meta["aliases"]
    assert "vxtzq1" in meta["aliases"]
    assert _DEFAULT_LEDGER_BALANCES["vxtzq"] == 100_000

    store = RouterStore(tmp_path / "legacy_vxtzq.db", SECRET)
    store.ensure_legacy_hiveminders_registered(now_utc=1_760_000_000.0)
    vxtzq_id = store.account_id_for("vxtzq")
    assert vxtzq_id is not None
    assert store.balance(vxtzq_id) == pytest.approx(100_000.0)
    # Calling ensure_partner_testing_grants after 100k is already present mints 0 extra HMT
    assert store.ensure_partner_testing_grants(now_utc=1_760_000_010.0) == {}
    assert store.balance(vxtzq_id) == pytest.approx(100_000.0)
    assert store.verify_chain() is True
    assert store.total_balances() == pytest.approx(store.total_minted())
    store.close()


def test_vxtzq_existing_30k_account_is_topped_up_idempotently_to_100k(tmp_path: Path) -> None:
    store = RouterStore(tmp_path / "existing_vxtzq.db", SECRET)
    acct_id, api_key, created = store.register_with_credentials(
        "vxtzq",
        "CrowdGPTPass#2026",
        role="prosumer",
        email="vxtzq@github.com",
    )
    assert created is True
    store.grant(acct_id, 30_000.0, "previous-allocation")
    assert store.balance(acct_id) == pytest.approx(30_000.0)

    minted = store.ensure_partner_testing_grants(now_utc=1_760_000_100.0)
    assert minted == {acct_id: pytest.approx(70_000.0)}
    assert store.balance(acct_id) == pytest.approx(100_000.0)

    # Simulate spending 5,000 HMT on Crowd-v1 inference and restarting: must NOT re-mint endlessly
    store.hold("req-crowd-test-1", acct_id, 5_000.0, now_utc=1_760_000_200.0)
    prod_id, _ = store.register("rtx2070_prod", "producer")
    store.settle("req-crowd-test-1", acct_id, prod_id, "hivemind/crowd-v1", 2500, 2500, 5000, 1000.0)
    assert store.balance(acct_id) == pytest.approx(95_000.0)
    assert store.ensure_partner_testing_grants(now_utc=1_760_000_300.0) == {}
    assert store.balance(acct_id) == pytest.approx(95_000.0)
    assert store.authenticate(api_key) is not None
    assert store.verify_chain() is True
    assert store.total_balances() == pytest.approx(store.total_minted())
    store.close()


def test_production_openrouter_server_rewards_are_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Owner decision 2026-11-12: sign-up rewards are off unless the deployment switches them on."""
    client = TestClient(_load_openrouter_server(monkeypatch).app)
    assert "Starter grant on this gateway: 1000.00 HMT" not in client.get("/signup").text
    reg = client.post(
        "/v1/hivemind/auth/register",
        json={"handle": "default_user", "email": "default@example.org", "password": "DefaultStrongPass!1", "role": "trainer"},
    )
    assert reg.status_code == 200, reg.text
    assert float(reg.json().get("balance_hmt", 0.0)) == pytest.approx(0.0)


def test_production_openrouter_server_defaults_registration_grants_bonuses_and_vxtzq_signup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_openrouter_server(
        monkeypatch,
        HIVEMIND_STARTER_GRANT_HMT="1000",
        HIVEMIND_REGISTRATION_BONUS_HMT="500",
        HIVEMIND_ALLOW_BONUSES="1",
    )
    client = TestClient(mod.app)

    # 1. /signup shows 1000.00 HMT starter grant + 500.00 HMT registration bonus (1500.00 HMT total) and bonus claim button, with zero currency leftovers
    signup_html = client.get("/signup").text
    assert "Starter grant on this gateway: 1000.00 HMT + 500.00 HMT registration bonus (1500.00 HMT total per new account)" in signup_html
    assert 'data-testid="signup-claim-bonus-btn"' in signup_html
    assert currency_leftovers(to_points_wording(signup_html)) == []

    # 2. Normal new prosumer registrant receives 1,000 starter + 500 registration bonus = 1,500 HMT immediately
    reg_alice = client.post(
        "/v1/hivemind/auth/register",
        json={"handle": "alice_trainer", "email": "alice@example.org", "password": "AliceStrongPass!1", "role": "trainer"},
    )
    assert reg_alice.status_code == 200, reg_alice.text
    alice_body = reg_alice.json()
    assert alice_body["starter_grant"] == pytest.approx(1000.0)
    assert alice_body["registration_bonus"] == pytest.approx(500.0)
    assert alice_body["balance_hmt"] == pytest.approx(1500.0)
    assert alice_body["compute_points"] == 1500

    # Alice claims her self-service bonus (500 welcome + 1000 trainer contributor role bonus = +1500 -> 3000 HMT total)
    alice_headers = {"Authorization": f"Bearer {alice_body['api_key']}"}
    status_before = client.get("/v1/hivemind/bonus/status", headers=alice_headers)
    assert status_before.status_code == 200
    assert status_before.json()["eligible"] is True
    assert status_before.json()["allow_bonuses"] is True

    claim_alice = client.post("/v1/hivemind/bonus/claim", headers=alice_headers, json={"bonus_code": "auto"})
    assert claim_alice.status_code == 200, claim_alice.text
    claim_body = claim_alice.json()
    assert claim_body["claimed"] is True
    assert claim_body["granted_hmt"] == pytest.approx(1500.0)
    assert claim_body["balance_hmt"] == pytest.approx(3000.0)

    # Re-claiming immediately gives 0.0
    claim_again = client.post("/v1/hivemind/bonus/claim", headers=alice_headers, json={"bonus_code": "auto"})
    assert claim_again.status_code == 200
    assert claim_again.json()["claimed"] is False
    assert claim_again.json()["granted_hmt"] == 0.0

    # 3. Registering never mints the partner allocation (granted once by an admin), whoever registers: neither the
    #    reserved handle nor a variant with Vxtzq's addresses gets the 100,000 HMT as a side effect.
    reg_vxtzq = client.post(
        "/v1/hivemind/auth/register",
        json={
            "handle": "@Vxtzq",
            "email": "vxtzqmc@gmail.com",
            "password": "VxtzqCrowdV1Password!9",
            "role": "prosumer",
        },
    )
    assert reg_vxtzq.status_code == 200, reg_vxtzq.text
    v_body = reg_vxtzq.json()
    assert v_body["balance_hmt"] == pytest.approx(1500.0)
    assert "partner_testing_grant" not in v_body
    reg_variant = client.post(
        "/v1/hivemind/auth/register",
        json={"handle": "vxtzq1", "email": "vxtzq@github.com", "password": "VariantStrongPass!9", "role": "prosumer"},
    )
    if reg_variant.status_code == 200:
        assert reg_variant.json()["balance_hmt"] < 100_000.0
        assert "partner_testing_grant" not in reg_variant.json()

    # 4. Guest upgrade on production gateway also receives the 1,500 HMT registration package
    guest_res = client.post("/api/v1/auth/guest", json={"target_profile": "developer"})
    assert guest_res.status_code == 200
    g_tok = guest_res.json()["token"]
    upg_res = client.post(
        "/api/v1/auth/upgrade-guest",
        headers={"Authorization": f"Bearer {g_tok}"},
        json={"handle": "upgraded_dev", "email": "up@example.org", "password": "UpgradedPass!2", "provider": "email"},
    )
    assert upg_res.status_code == 200, upg_res.text
    assert upg_res.json()["balance_hmt"] == pytest.approx(1500.0)
    assert upg_res.json()["compute_points"] == 1500


def test_bonus_sybil_ip_cap_faucet_cooldown_and_ledger_conservation(tmp_path: Path) -> None:
    db_path = tmp_path / "bonus_sybil.db"
    store = RouterStore(db_path, SECRET)
    now = {"t": 1_765_000_000.0}
    app = create_router_app(
        store,
        admin_key_hash=store.key_digest(ADMIN_KEY),
        clock=lambda: now["t"],
        starter_grant_hmt=100.0,
        registration_bonus_hmt=50.0,
        allow_bonuses=True,
        bonus_claims_per_ip_24h=2,
    )
    client = TestClient(app, client=("198.51.100.42", 51000))

    # 1. Test/probe accounts are barred from claiming bonuses
    probe_reg = client.post(
        "/v1/hivemind/register",
        headers={"X-Admin-Key": ADMIN_KEY},
        json={"handle": "probe_bot", "role": "consumer", "kind": "probe"},
    ).json()
    probe_claim = client.post(
        "/v1/hivemind/bonus/claim",
        headers={"Authorization": f"Bearer {probe_reg['api_key']}"},
        json={},
    )
    assert probe_claim.status_code == 403
    assert probe_claim.json()["error"]["code"] == "bonus_forbidden_account_kind"

    # 2. Consumer account gets 100 + 50 = 150 at registration, then claims 50 welcome_bonus (no contributor_role_bonus) -> 200 HMT
    c1 = client.post("/v1/hivemind/register", json={"handle": "consumer_one", "role": "consumer"}).json()
    assert c1["balance"] if "balance" in c1 else store.balance(c1["account_id"]) == pytest.approx(150.0)
    c1_headers = {"Authorization": f"Bearer {c1['api_key']}"}

    # Non-vxtzq cannot claim vxtzq_crowd_v1_creator_bonus
    forbidden_partner = client.post(
        "/v1/hivemind/bonus/claim",
        headers=c1_headers,
        json={"bonus_code": "vxtzq_crowd_v1_creator_bonus"},
    )
    assert forbidden_partner.status_code == 403

    c1_claim1 = client.post("/v1/hivemind/bonus/claim", headers=c1_headers, json={"bonus_code": "auto"}).json()
    assert c1_claim1["claimed"] is True
    assert c1_claim1["granted_hmt"] == pytest.approx(50.0)
    assert c1_claim1["balance_hmt"] == pytest.approx(200.0)  # IP slot 1 used

    # Balance is 200.0 (< 500.0), so a second explicit/auto claim gets daily_testing_faucet (+250.0 -> 450.0)
    c1_faucet1 = client.post("/v1/hivemind/bonus/claim", headers=c1_headers, json={"bonus_code": "auto"}).json()
    assert c1_faucet1["claimed"] is True
    assert c1_faucet1["granted_hmt"] == pytest.approx(250.0)
    assert c1_faucet1["balance_hmt"] == pytest.approx(450.0)  # IP slot 2 used

    # Third claim from same IP within 24h hits bonus_claims_per_ip_24h=2 cap
    c2 = client.post("/v1/hivemind/register", json={"handle": "consumer_two", "role": "consumer"}).json()
    c2_claim_capped = client.post(
        "/v1/hivemind/bonus/claim",
        headers={"Authorization": f"Bearer {c2['api_key']}"},
        json={"bonus_code": "auto"},
    ).json()
    assert c2_claim_capped["claimed"] is False
    assert c2_claim_capped["reason"] == "ip_cap_reached"

    # Advance 12h: different IP is not capped, but c1's daily faucet is still in 24h cooldown
    now["t"] += 12 * 3600.0
    other_ip_client = TestClient(app, client=("198.51.100.99", 51001))
    c1_cooldown = other_ip_client.post("/v1/hivemind/bonus/claim", headers=c1_headers, json={"bonus_code": "daily_testing_faucet"}).json()
    assert c1_cooldown["claimed"] is False
    assert c1_cooldown["reason"] == "no_eligible_bonus"

    # Advance past 24h total: c1's balance is 450 (< 500) and cooldown expired -> faucet grants +250 -> 700 HMT
    now["t"] += 12 * 3600.0 + 60.0
    c1_faucet2 = client.post("/v1/hivemind/bonus/claim", headers=c1_headers, json={"bonus_code": "daily_testing_faucet"}).json()
    assert c1_faucet2["claimed"] is True
    assert c1_faucet2["granted_hmt"] == pytest.approx(250.0)
    assert c1_faucet2["balance_hmt"] == pytest.approx(700.0)

    active_ip_rows = store._conn.execute("SELECT ip_hash FROM bonus_claim_ips").fetchall()
    assert len(active_ip_rows) == 1 and len(active_ip_rows[0]["ip_hash"]) == 64

    # Advance another 25h: balance is 700 (>= 500), so faucet is NOT granted (and expired >24h IP slots are purged)
    now["t"] += 25 * 3600.0
    c1_above_threshold = client.post("/v1/hivemind/bonus/claim", headers=c1_headers, json={"bonus_code": "daily_testing_faucet"}).json()
    assert c1_above_threshold["claimed"] is False

    # Verify ledger hash-chain & exact conservation, and verify raw IPs never appear in SQLite bytes
    assert store.verify_chain() is True
    assert store.total_balances() == pytest.approx(store.total_minted())
    store.close()

    raw_bytes = db_path.read_bytes()
    wal = db_path.with_name(db_path.name + "-wal")
    if wal.exists():
        raw_bytes += wal.read_bytes()
    assert b"198.51.100.42" not in raw_bytes
    assert b"198.51.100.99" not in raw_bytes
    rows = sqlite3.connect(db_path).execute("SELECT ip_hash FROM bonus_claim_ips").fetchall()
    assert rows == []

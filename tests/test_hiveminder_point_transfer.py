# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""End-to-end tests for HiveMinder-to-HiveMinder HMT point transfers, AMCP/Kafka delivery, and the 5,000 HMT gift to @Vxtzq."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import httpx
import pytest
from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.router.points_wording import currency_leftovers, to_points_wording
from hivemind.swarm.amcp_kafka_campaign_bus import KAFKA_TOPICS
from hivemind.swarm.amcp_kafka_transport import (
    DLQ_TOPIC,
    AMCPConsumer,
    AMCPPublisher,
    EnvelopeVerifier,
    HMACKeyring,
    InMemoryTransport,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET = b"hiveminder-p2p-point-transfer-secret-32b!"
ADMIN_KEY = "hm-admin-p2p-transfer-key"


def _load_connect_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hivemind_connect_p2p", REPO_ROOT / "scripts" / "hivemind_connect.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_5000_hmt_transfer_to_vxrzq_vxtzq_with_signed_amcp_kafka_delivery(tmp_path: Path) -> None:
    """Test giving 5,000 HMT from xavier_callens to VXRZQ (@Vxtzq) for CrowdGPT inference tests + signed AMCP/Kafka receipt."""
    now = 1_794_370_000.0
    store = RouterStore(tmp_path / "p2p_vxtzq.db", SECRET)
    store.ensure_legacy_hiveminders_registered(now_utc=now)

    xavier_acct = store.resolve_hiveminder_account("xavier_callens")
    vxtzq_acct = store.resolve_hiveminder_account("VXRZQ")
    assert xavier_acct is not None and xavier_acct.handle == "xavier_callens"
    assert vxtzq_acct is not None and vxtzq_acct.handle == "vxtzq"
    assert store.balance(xavier_acct.account_id) == pytest.approx(68_500.0)
    assert store.balance(vxtzq_acct.account_id) == pytest.approx(100_000.0)
    minted_before = store.total_minted()

    xavier_key = store.rotate_key(xavier_acct.account_id)
    vxtzq_key = store.rotate_key(vxtzq_acct.account_id)

    # Real signed CloudEvents 1.0 + HMAC-SHA256 AMCP Kafka transport
    transport = InMemoryTransport(num_partitions=2)
    keyring = HMACKeyring(keys={"k1": b"a" * 32}, active_kid="k1")
    publisher = AMCPPublisher(transport, keyring)
    app = create_router_app(
        store,
        admin_key_hash=store.key_digest(ADMIN_KEY),
        clock=lambda: now,
        amcp_bus=lambda ev: publisher.publish(ev),
    )
    client = TestClient(app)

    # Transfer 5,000 HMT to "VXRZQ" (alias -> canonical "vxtzq") for CrowdGPT inference tests
    res = client.post(
        "/v1/hivemind/points/transfer",
        headers={"Authorization": f"Bearer {xavier_key}"},
        json={
            "recipient": "VXRZQ",
            "amount": 5000.0,
            "memo": "CrowdGPT (hivemind/crowd-v1 & hivemind/gcpcrowd) inference tests",
            "idempotency_key": "crowdgpt-vxtzq-5000-hmt",
            "notify_amcp": True,
            "subject_id": "SUBJ-01-PYTHON-RUST-CODER",
            "model_ref": "hivemind/crowd-v1",
            "pr_url": "https://github.com/xaviercallens/HiveMind/pull/871",
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["object"] == "hivemind.point_transfer"
    assert body["sender_handle"] == "xavier_callens"
    assert body["recipient_handle"] == "vxtzq"
    assert body["amount_hmt"] == pytest.approx(5_000.0)
    assert body["sender_balance_hmt"] == pytest.approx(63_500.0)
    assert body["idempotent_replay"] is False
    assert body["bus"]["status"] == "published"
    assert body["amcp_event_id"] is not None

    # Recipient @Vxtzq now holds 105,000.00 HMT, and zero HMT was minted or burned
    assert store.balance(xavier_acct.account_id) == pytest.approx(63_500.0)
    assert store.balance(vxtzq_acct.account_id) == pytest.approx(105_000.0)
    assert store.total_minted() == pytest.approx(minted_before)
    assert store.total_balances() == pytest.approx(store.total_minted())
    assert store.verify_chain() is True

    # Consume and verify the signed AMCP CloudEvents 1.0 message from hivemind.amcp.human_agent.v1
    delivered = []
    verifier = EnvelopeVerifier(keyring, clock=lambda: now + 1.0)
    consumer = AMCPConsumer(
        transport,
        verifier,
        topics=[KAFKA_TOPICS["human_agent"]],
        clock=lambda: now + 1.0,
    )
    consumer.register("io.hivemind.amcp.points.transfer", lambda verified: delivered.append(verified.event))
    handled = consumer.run_once()
    assert handled == 1 and consumer.stats.delivered == 1
    assert len(delivered) == 1
    assert transport.records(DLQ_TOPIC) == []
    ev = delivered[0]
    assert ev.event_id == body["amcp_event_id"]
    assert ev.event_type == "io.hivemind.amcp.points.transfer"
    assert ev.sender_id == "xavier_callens"
    assert ev.recipient_id == "vxtzq"
    assert ev.payload["amount_hmt"] == pytest.approx(5_000.0)
    assert ev.payload["model_ref"] == "hivemind/crowd-v1"
    assert ev.payload["pr_url"] == "https://github.com/xaviercallens/HiveMind/pull/871"

    # Replay with same idempotency_key does NOT debit again and does NOT emit a second AMCP event
    replay = client.post(
        "/v1/hivemind/points/transfer",
        headers={"Authorization": f"Bearer {xavier_key}"},
        json={
            "recipient": "@Vxtzq",
            "amount": 5000.0,
            "memo": "CrowdGPT (hivemind/crowd-v1 & hivemind/gcpcrowd) inference tests",
            "idempotency_key": "crowdgpt-vxtzq-5000-hmt",
        },
    )
    assert replay.status_code == 200
    assert replay.json()["idempotent_replay"] is True
    assert replay.json()["transfer_id"] == body["transfer_id"]
    assert store.balance(xavier_acct.account_id) == pytest.approx(63_500.0)
    assert store.balance(vxtzq_acct.account_id) == pytest.approx(105_000.0)

    # Reusing idempotency_key with a different amount is rejected with 400
    bad_idem = client.post(
        "/v1/hivemind/points/transfer",
        headers={"Authorization": f"Bearer {xavier_key}"},
        json={"recipient": "@Vxtzq", "amount": 100.0, "idempotency_key": "crowdgpt-vxtzq-5000-hmt"},
    )
    assert bad_idem.status_code == 400

    # Both sender and recipient see the transfer in GET /v1/hivemind/points/transfers
    sent_list = client.get("/v1/hivemind/points/transfers", headers={"Authorization": f"Bearer {xavier_key}"}).json()
    recv_list = client.get("/v1/hivemind/points/transfers", headers={"Authorization": f"Bearer {vxtzq_key}"}).json()
    assert sent_list["count"] == 1 and sent_list["transfers"][0]["direction"] == "sent"
    assert recv_list["count"] == 1 and recv_list["transfers"][0]["direction"] == "received"
    assert recv_list["transfers"][0]["amcp_event_id"] == body["amcp_event_id"]
    assert recv_list["balance_hmt"] == pytest.approx(105_000.0)
    store.close()


def test_point_transfer_security_guards_cli_and_server_startup_gift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = RouterStore(tmp_path / "p2p_guards.db", SECRET)
    app = create_router_app(
        store,
        admin_key_hash=store.key_digest(ADMIN_KEY),
        starter_grant_hmt=1000.0,
        allow_bonuses=True,
    )
    client = TestClient(app)

    # /signup includes the transfer form with zero currency leftovers
    signup_html = client.get("/signup").text
    assert 'data-testid="signup-transfer-hmt-btn"' in signup_html
    assert currency_leftovers(to_points_wording(signup_html)) == []

    alice = client.post("/v1/hivemind/register", json={"handle": "alice_hm", "role": "prosumer"}).json()
    bob = client.post("/v1/hivemind/register", json={"handle": "bob_hm", "role": "prosumer"}).json()
    probe = client.post(
        "/v1/hivemind/register",
        headers={"X-Admin-Key": ADMIN_KEY},
        json={"handle": "probe_hm", "role": "consumer", "kind": "probe"},
    ).json()

    alice_hdr = {"Authorization": f"Bearer {alice['api_key']}"}
    probe_hdr = {"Authorization": f"Bearer {probe['api_key']}"}

    # Self-transfer -> 400
    assert client.post("/v1/hivemind/points/transfer", headers=alice_hdr, json={"recipient": "@alice_hm", "amount": 10.0}).status_code == 400
    # Insufficient balance -> 402
    assert client.post("/v1/hivemind/points/transfer", headers=alice_hdr, json={"recipient": "@bob_hm", "amount": 5000.0}).status_code == 402
    # Unknown account -> 404
    assert client.post("/v1/hivemind/points/transfer", headers=alice_hdr, json={"recipient": "@nobody_here", "amount": 10.0}).status_code == 404
    # Probe sender or probe recipient -> 403
    assert client.post("/v1/hivemind/points/transfer", headers=probe_hdr, json={"recipient": "@bob_hm", "amount": 1.0}).status_code == 403
    assert client.post("/v1/hivemind/points/transfer", headers=alice_hdr, json={"recipient": "@probe_hm", "amount": 1.0}).status_code == 403

    # CLI hivemind_connect.py transfer subcommand round-trip
    connect = _load_connect_cli()
    monkeypatch.setenv(connect.API_KEY_ENV, alice["api_key"])

    def _mock_handler(req: httpx.Request) -> httpx.Response:
        resp = client.request(req.method, req.url.path, headers=dict(req.headers), content=req.content)
        return httpx.Response(resp.status_code, json=resp.json())

    rc = connect.main(
        [
            "--gateway",
            "https://gw.example.org",
            "transfer",
            "--to",
            "@bob_hm",
            "--amount",
            "250",
            "--memo",
            "Thanks for GPU hosting",
        ],
        transport=httpx.MockTransport(_mock_handler),
    )
    assert rc == 0
    cli_out = capsys.readouterr().out
    assert "@alice_hm -> @bob_hm 250.00 HMT" in cli_out
    assert store.balance(alice["account_id"]) == pytest.approx(750.0)
    assert store.balance(bob["account_id"]) == pytest.approx(1250.0)
    assert store.verify_chain() is True
    assert store.total_balances() == pytest.approx(store.total_minted())
    store.close()

    # Verify deploy/gcp_openrouter/server.py startup idempotent 5,000 HMT gift when xavier_callens & vxtzq exist
    server_db = tmp_path / "server_startup_gift.db"
    seed_store = RouterStore(server_db, b"s" * 40)
    seed_store.ensure_legacy_hiveminders_registered(now_utc=1_794_370_000.0)
    seed_store.close()

    monkeypatch.setenv("HIVEMIND_ROUTER_SECRET", "s" * 40)
    monkeypatch.setenv("HIVEMIND_ROUTER_DB", str(server_db))
    spec = importlib.util.spec_from_file_location(
        "gcp_openrouter_server_p2p_gift",
        REPO_ROOT / "deploy" / "gcp_openrouter" / "server.py",
    )
    assert spec is not None and spec.loader is not None
    srv_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv_mod)

    srv_client = TestClient(srv_mod.app)
    amcp_feed = srv_client.get("/v1/hivemind/amcp/events?topic=human_agent&limit=10").json()
    transfer_events = [e for e in amcp_feed["events"] if e["event_type"] == "io.hivemind.amcp.points.transfer"]
    assert len(transfer_events) == 1
    assert transfer_events[0]["recipient_id"] == "vxtzq"
    assert transfer_events[0]["payload"]["amount_hmt"] == pytest.approx(5000.0)

    check_store = RouterStore(server_db, b"s" * 40)
    assert check_store.balance(check_store.account_id_for("xavier_callens")) == pytest.approx(63_500.0)
    assert check_store.balance(check_store.account_id_for("vxtzq")) == pytest.approx(105_000.0)
    assert check_store.verify_chain() is True
    assert check_store.total_balances() == pytest.approx(check_store.total_minted())
    check_store.close()

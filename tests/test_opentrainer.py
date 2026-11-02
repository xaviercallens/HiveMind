# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the ``trainer`` portal role and the central ``OpenTrainer`` component."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from hivemind.router import RouterStore, create_router_app
from hivemind.router.accounts import ROLES, TRAINER_ROLES


def test_trainer_role_in_roles_and_automatic_sqlite_migration(tmp_path: Path) -> None:
    assert "trainer" in ROLES
    assert "trainer" in TRAINER_ROLES

    # Create a legacy database with the pre-S36 CHECK constraint (without 'trainer')
    db_path = tmp_path / "legacy_router.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE accounts (
            account_id       TEXT PRIMARY KEY,
            handle           TEXT NOT NULL UNIQUE,
            role             TEXT NOT NULL CHECK (role IN ('consumer','producer','prosumer','admin')),
            key_hash         TEXT NOT NULL UNIQUE,
            max_price_per_1k REAL NOT NULL DEFAULT 10.0,
            is_test          INTEGER NOT NULL DEFAULT 0 CHECK (is_test IN (0, 1)),
            kind             TEXT NOT NULL DEFAULT 'user' CHECK (kind IN ('user', 'test', 'probe')),
            created_utc      REAL NOT NULL
        );
        INSERT INTO accounts VALUES ('acc-1', 'legacy_user', 'prosumer', 'hash1', 10.0, 0, 'user', 1700000000.0);
        """
    )
    conn.commit()
    conn.close()

    # Opening RouterStore must automatically migrate the CHECK constraint to allow 'trainer'
    store = RouterStore(db_path, b"s" * 32)
    aid, key = store.register("rtx2070_trainer", "trainer")
    acct = store.authenticate(key)
    assert acct is not None
    assert acct.account_id == aid
    assert acct.role == "trainer"
    legacy = store.get_account("acc-1")
    assert legacy is not None and legacy.handle == "legacy_user"


def test_opentrainer_status_heartbeat_and_delta_submission() -> None:
    store = RouterStore(":memory:", b"k" * 32)
    client = TestClient(create_router_app(store))

    # 1. Public status endpoint works at both /v1/hivemind/opentrainer/status and /v1/opentrainer/status
    st = client.get("/v1/hivemind/opentrainer/status")
    assert st.status_code == 200
    body = st.json()
    assert body["component"] == "OpenTrainer"
    assert body["active_task_offer"]["topic"] == "hivemind.amcp.federation.v1"
    fw_modes = {m["mode_id"]: m for m in body["firewall_traversal_modes"]}
    assert fw_modes["outbound-https-relay"]["inbound_ports_required"] == 0
    pr_numbers = {entry["pr_number"] for entry in body["rtx2070_t4_integration_ledger"]}
    assert {410, 409, 408, 407, 401, 399, 366} <= pr_numbers
    verbs = [p["aglet_verb"] for p in body["aglet_amcp_kafka_lifecycle"]]
    assert verbs[0] == "dispatch" and verbs[-1] == "retract"

    alias_st = client.get("/v1/opentrainer/status")
    assert alias_st.status_code == 200 and alias_st.json()["component"] == "OpenTrainer"

    # 2. Register a trainer account via /v1/hivemind/register
    reg = client.post("/v1/hivemind/register", json={"handle": "xavgamsrv_rtx2070", "role": "trainer"})
    assert reg.status_code == 200
    trainer_key = reg.json()["api_key"]
    assert reg.json()["role"] == "trainer"

    # 3. Consumer account is forbidden from heartbeating OpenTrainer
    creg = client.post("/v1/hivemind/register", json={"handle": "ide_reader", "role": "consumer"})
    consumer_key = creg.json()["api_key"]
    denied = client.post(
        "/v1/hivemind/opentrainer/heartbeat",
        headers={"Authorization": f"Bearer {consumer_key}"},
        json={"host_label": "laptop", "gpu_name": "Integrated", "vram_total_mib": 2048, "vram_free_mib": 1024},
    )
    assert denied.status_code == 403

    # 4. Trainer account heartbeats over outbound HTTPS relay
    hb = client.post(
        "/v1/hivemind/opentrainer/heartbeat",
        headers={"Authorization": f"Bearer {trainer_key}"},
        json={
            "host_label": "XAVGAMSRV",
            "gpu_name": "NVIDIA GeForce RTX 2070",
            "vram_total_mib": 8192,
            "vram_free_mib": 6400,
            "network_mode": "outbound-https-relay",
        },
    )
    assert hb.status_code == 200
    hb_body = hb.json()
    assert hb_body["aglet_lifecycle_phase"] == "arrival"
    assert hb_body["node"]["host_label"] == "XAVGAMSRV"
    assert hb_body["node"]["role"] == "trainer"
    offer = hb_body["active_task_offer"]
    assert offer["model_id"] == "hivemind/granite-moe-qlora-nf4"

    # 5. Submit a verified Stage-2 QLoRA / DiLoCo delta
    delta_sha = hashlib.sha256(b"round-1-rtx2070-delta").hexdigest()
    sub = client.post(
        "/v1/hivemind/opentrainer/submit-delta",
        headers={"Authorization": f"Bearer {trainer_key}"},
        json={
            "round_id": 1,
            "campaign_id": "E8-STAGE2-TINY-GRANITE-MOE-QLORA",
            "learner_index": 1,
            "inner_steps": 8,
            "loss_before": 4.1721,
            "loss_after": 1.7772,
            "delta_norm": 0.142,
            "delta_sha256": delta_sha,
            "code_pin": offer["script_sha256"],
            "network_mode": "outbound-https-relay",
        },
    )
    assert sub.status_code == 200
    sub_body = sub.json()
    assert sub_body["accepted"] is True
    assert sub_body["aglet_lifecycle_phase"] == "result"
    assert sub_body["submission"]["delta_sha256"] == delta_sha

    # 6. Verify status now shows 1 live node, 1 accepted round, and the AMCP event on hivemind.amcp.federation.v1
    st_after = client.get("/v1/hivemind/opentrainer/status").json()
    assert st_after["portal_category"]["registered_trainer_accounts_count"] == 1
    assert st_after["live_opentrainer_fleet"]["active_trainer_nodes"] == 1
    assert len(st_after["recent_training_rounds"]) == 1

    events = client.get("/v1/hivemind/amcp/events?topic=hivemind.amcp.federation.v1").json()
    assert events["count"] >= 1
    assert events["events"][0]["event_type"] == "io.hivemind.amcp.fed.task_result"


def test_signup_and_portal_pages_include_trainer_and_opentrainer() -> None:
    store = RouterStore(":memory:", b"p" * 32)
    client = TestClient(create_router_app(store))

    signup_html = client.get("/signup").text
    assert 'value="trainer"' in signup_html
    assert "OpenTrainer" in signup_html
    assert "hivemind_opentrainer_agent.py" in signup_html

    portal_html = client.get("/").text
    assert 'id="opentrainer"' in portal_html
    assert "OpenTrainer" in portal_html

    # Register via /v1/hivemind/auth/register with role="trainer"
    auth_reg = client.post(
        "/v1/hivemind/auth/register",
        json={"handle": "portal_trainer_node", "password": "StrongTrainerPassword!23", "role": "trainer"},
    )
    assert auth_reg.status_code == 200
    assert auth_reg.json()["role"] == "trainer"


def test_web_ui_portal_renders_and_syncs_real_life_database_information() -> None:
    store = RouterStore(":memory:", b"d" * 32)
    client = TestClient(create_router_app(store))

    reg = client.post("/v1/hivemind/register", json={"handle": "rtx2070_live_trainer", "role": "trainer"})
    assert reg.status_code == 200
    trainer_key = reg.json()["api_key"]

    hb = client.post(
        "/v1/hivemind/opentrainer/heartbeat",
        headers={"Authorization": f"Bearer {trainer_key}"},
        json={
            "host_label": "XAVGAMSRV-RTX2070",
            "gpu_name": "NVIDIA GeForce RTX 2070 (8 GB NF4)",
            "vram_total_mib": 8192,
            "vram_free_mib": 6420,
            "network_mode": "outbound-https-relay",
        },
    )
    assert hb.status_code == 200
    offer = hb.json()["active_task_offer"]

    delta_sha = hashlib.sha256(b"real-db-ui-round").hexdigest()
    sub = client.post(
        "/v1/hivemind/opentrainer/submit-delta",
        headers={"Authorization": f"Bearer {trainer_key}"},
        json={
            "round_id": 7,
            "campaign_id": "OPENTRAINER-LIVE-DB-ROUND",
            "learner_index": 1,
            "inner_steps": 20,
            "loss_before": 4.1709,
            "loss_after": 2.6123,
            "delta_norm": 1.2345,
            "delta_sha256": delta_sha,
            "code_pin": offer["script_sha256"],
            "network_mode": "outbound-https-relay",
        },
    )
    assert sub.status_code == 200

    # 1. Verify /v1/hivemind/db/live returns real SQLite database state
    db_live = client.get("/v1/hivemind/db/live").json()
    assert db_live["object"] == "hivemind.database.live_summary"
    assert db_live["accounts"]["by_role"]["trainer"] == 1
    assert db_live["opentrainer"]["registered_trainer_accounts_count"] == 1
    assert db_live["opentrainer"]["active_trainer_nodes"] == 1
    assert db_live["opentrainer"]["total_round_submissions_in_db"] == 1
    assert db_live["opentrainer"]["latest_eval_loss"] == 2.6123
    assert db_live["usage_and_ledger"]["ledger_chain_valid"] is True

    # 2. Verify Web UI (GET / and GET /fr) renders real-life SQLite rows in its HTML tables
    html_en = client.get("/").text
    assert 'id="otNodesTableBody"' in html_en
    assert 'id="otRoundsTableBody"' in html_en
    assert 'id="dbProducersTableBody"' in html_en
    assert "XAVGAMSRV-RTX2070" in html_en
    assert "rtx2070_live_trainer" in html_en
    assert "OPENTRAINER-LIVE-DB-ROUND" in html_en
    assert "2.6123" in html_en
    assert "refreshLiveDatabaseTelemetry" in html_en

    # 3. Verify syncing the low-cost VM orchestrator ledger populates SQLite idempotently
    store.sync_opentrainer_orchestrator_ledger(
        {
            "run_id": "opentrainer-vm-test",
            "current_eval_loss": 2.5012,
            "learners_last_seen": {
                "l0": {"host": "hm-opentrainer-1-cpu-sentinel", "timestamp_unix": 1793524000.0, "vram_peak_mib": 0.0},
                "l1": {"host": "rtx2070-XAVGAMSRV", "timestamp_unix": 1793524000.0, "vram_peak_mib": 18.85},
            },
            "rounds": [
                {
                    "round": 0,
                    "accepted": ["l0", "l1"],
                    "hosts": {"l0": "hm-opentrainer-1-cpu-sentinel", "l1": "rtx2070-XAVGAMSRV"},
                    "delta_l2": {"l0": 4.32, "l1": 6.89},
                    "vram_peak_mib": {"l0": 0.0, "l1": 18.85},
                    "eval_loss_after": 2.5012,
                }
            ],
        }
    )
    html_after = client.get("/").text
    assert "hm-opentrainer-1-cpu-sentinel" in html_after
    assert "opentrainer-vm-test" in html_after
    assert "Founding Pioneer Trainer #1" in html_after

    # 4. Verify promoting the first trainer grants +5,000 HMT idempotently on the hash-chained ledger
    promo_res = client.post(
        "/v1/hivemind/opentrainer/promote-trainer",
        json={"handle": "rtx2070-xavgamsrv", "bonus_hmt": 5000.0, "rank": 1},
    )
    assert promo_res.status_code == 200
    promo_json = promo_res.json()
    assert promo_json["handle"] == "rtx2070-xavgamsrv"
    assert promo_json["pioneer_rank"] == 1
    assert promo_json["minted_now"] is True
    assert promo_json["balance_hmt"] == 5000.0

    # Calling promote-trainer a second time must be idempotent (no duplicate mint)
    promo_again = client.post(
        "/v1/hivemind/opentrainer/promote-trainer",
        json={"handle": "rtx2070-xavgamsrv", "bonus_hmt": 5000.0, "rank": 1},
    ).json()
    assert promo_again["minted_now"] is False
    assert promo_again["balance_hmt"] == 5000.0

    camp = client.get("/v1/hivemind/opentrainer/pioneer-campaign").json()
    assert camp["claimed_pioneer_slots"] == 1
    assert camp["open_pioneer_slots"] == 9
    assert camp["first_pioneer_trainer"]["handle"] == "rtx2070-xavgamsrv"
    assert store.verify_chain() is True
    assert "2.5012" in html_after



def test_delta_submission_has_no_default_measurements_and_is_labelled_unverified() -> None:
    store = RouterStore(":memory:", b"k" * 32)
    client = TestClient(create_router_app(store))
    reg = client.post("/v1/hivemind/register", json={"handle": "ot_tester", "role": "trainer"})
    key = reg.json()["api_key"]
    sha = hashlib.sha256(b"delta").hexdigest()
    base = {"round_id": 3, "delta_sha256": sha, "network_mode": "outbound-https-relay"}
    # No loss or norm given: must be refused, not filled in with a built-in figure.
    missing = client.post("/v1/hivemind/opentrainer/submit-delta", headers={"Authorization": f"Bearer {key}"}, json=base)
    assert missing.status_code == 400  # the router maps schema errors to 400
    assert "loss_before" in missing.text and "Field required" in missing.text  # the router reports the first missing field
    full = client.post(
        "/v1/hivemind/opentrainer/submit-delta",
        headers={"Authorization": f"Bearer {key}"},
        json=base | {"loss_before": 2.5, "loss_after": 2.4, "delta_norm": 0.05},
    )
    assert full.status_code == 200
    body = full.json()
    assert body["verified"] is False and "CLIENT_REPORTED_UNVERIFIED" in body["verification"]
    assert body["validation"]["verification"] == "CLIENT_REPORTED_UNVERIFIED"
    assert body["validation"]["vram_values_reported_by_client"] is False
    assert body["submission"]["eval_loss"] == 2.4

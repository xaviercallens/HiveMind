# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""PS-02-D4 and PS-03-D1 — Tests for node_egress_receipt.py and accounts_receipt.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

import pytest

from hivemind.router.accounts import RouterStore

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, rel_path: str) -> Any:
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_node_egress_receipt_covers_all_five_volunteer_node_types(tmp_path: Path) -> None:
    """PS-02-D4: every node type has a measured egress per hour and a declared budget."""
    egress = _load_script("node_egress_receipt_test", "scripts/node_egress_receipt.py")
    out_file = tmp_path / "NODE_EGRESS_test.json"
    rc = egress.main(["--out", str(out_file)])
    assert rc == 0
    assert out_file.exists()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["schema"] == "hivemind.node_egress_receipt.v1"
    assert data["roadmap_deliverable"] == "PS-02-D4"
    assert data["all_within_budget"] is True

    expected_types = {
        "producer_agent",
        "producer_shield",
        "boinc_trainer",
        "opentrainer_worker",
        "amcp_federation_learner",
    }
    assert set(data["node_types"].keys()) == expected_types
    for node_type, row in data["node_types"].items():
        assert row["measured_egress_bytes_per_hour"] > 0, node_type
        assert row["declared_budget_bytes_per_hour"] >= row["measured_egress_bytes_per_hour"], node_type
        assert row["within_budget"] is True, node_type
        assert row["lateral_movement_allowed"] is False, node_type


def test_accounts_receipt_counts_by_kind_and_role_and_strips_all_personal_data(tmp_path: Path) -> None:
    """PS-03-D1: counts by kind and role present; no email, IP, or API key in the receipt."""
    acc_mod = _load_script("accounts_receipt_test", "scripts/accounts_receipt.py")
    db_path = tmp_path / "router.db"
    store = RouterStore(db_path, b"0123456789abcdef0123456789abcdef")

    # Populate user, test, and probe accounts across roles, with personal data (emails, keys, endpoints)
    store.register_with_credentials("alice_user", "StrongPass!2026", role="consumer", email="alice@example.org")
    prod_id, prod_key = store.register("bob_producer", "producer")
    store.register("carol_trainer", "trainer")
    store.register("dave_test", "prosumer", kind="test")
    store.register("eve_probe", "consumer", kind="probe")

    store.upsert_producer(
        prod_id,
        "https://bob-node.example.run.app",
        [{"id": "coder", "local_name": "qwen2.5-coder:7b"}],
        0.4,
        60.0,
        endpoint_token="secret-endpoint-token-xyz",
    )
    now_ts = time.time()
    store.upsert_opentrainer_node(
        account_id=prod_id,
        handle="bob_producer",
        node_id="ot-node-active-1",
        gpu_name="NVIDIA GeForce RTX 3070",
        vram_total_mib=8192,
        vram_free_mib=5632,
        network_mode="outbound-https-relay",
        stacks=["qlora-moe"],
        host_label="rtx3070-home",
        now_utc=now_ts - 3600.0,
    )
    store.upsert_opentrainer_node(
        account_id=prod_id,
        handle="bob_producer",
        node_id="ot-node-stale-2",
        gpu_name="NVIDIA GeForce RTX 2070",
        vram_total_mib=8192,
        vram_free_mib=5632,
        network_mode="outbound-https-relay",
        stacks=["qlora-moe"],
        host_label="stale-node",
        now_utc=now_ts - 45.0 * 86400.0,
    )

    out_file = tmp_path / "ACCOUNTS_test.json"
    rc = acc_mod.main(["--db", str(db_path), "--out", str(out_file)])
    assert rc == 0
    raw_text = out_file.read_text(encoding="utf-8")

    # Privacy assertion: zero emails, IPs, handles, or API keys in the file
    assert "alice@example.org" not in raw_text
    assert prod_key not in raw_text
    assert "secret-endpoint-token-xyz" not in raw_text
    assert "alice_user" not in raw_text
    assert "bob_producer" not in raw_text
    acc_mod.assert_zero_personal_data(raw_text)

    receipt = json.loads(raw_text)
    assert receipt["schema"] == "hivemind.accounts_receipt.v1"
    assert receipt["personal_data_included"] is False
    assert receipt["total_accounts"] == 8
    assert receipt["counts_by_kind"] == {"user": 6, "test": 1, "probe": 1}
    assert receipt["counts_by_role"] == {
        "consumer": 2,
        "producer": 1,
        "prosumer": 1,
        "trainer": 1,
        "admin": 3,
    }
    assert receipt["by_kind_and_role"]["user"]["consumer"] == 1
    assert receipt["by_kind_and_role"]["user"]["producer"] == 1
    assert receipt["by_kind_and_role"]["user"]["trainer"] == 1
    assert receipt["by_kind_and_role"]["user"]["admin"] == 3
    assert receipt["by_kind_and_role"]["test"]["prosumer"] == 1
    assert receipt["by_kind_and_role"]["probe"]["consumer"] == 1
    assert receipt["active_nodes_30d"] == {
        "window_days": 30,
        "producer_endpoints_active_30d": 1,
        "opentrainer_nodes_active_30d": 1,
        "total_active_nodes_30d": 2,
    }

    # Negative controls: planted email, IPv4, or hm- API key must raise ValueError
    for leaked in (
        '{"leak": "alice@example.org"}',
        '{"leak": "192.168.1.42"}',
        f'{{"leak": "{prod_key}"}}',
    ):
        with pytest.raises(ValueError, match="Personal data leak detected"):
            acc_mod.assert_zero_personal_data(leaked)


def test_accounts_receipt_activity_evidence_separates_rows_from_machines_and_use(tmp_path: Path) -> None:
    """PS-03-D1: endpoint rows are not machines, and stored accounts are not active users."""
    acc = _load_script("accounts_receipt_activity", "scripts/accounts_receipt.py")
    now = 1_800_000_000.0
    db = tmp_path / "router.db"
    with sqlite3.connect(str(db)) as conn:
        conn.execute("CREATE TABLE accounts (account_id TEXT, handle TEXT, role TEXT, key_hash TEXT, created_utc REAL, "
                     "is_test INTEGER, kind TEXT, password_hash TEXT, email TEXT)")
        conn.executemany("INSERT INTO accounts VALUES (?,?,?,?,?,?,?,?,?)", [
            ("a1", "h1", "prosumer", "k", now, 0, "user", "pbkdf2$x", None),   # signed in and sends requests
            ("a2", "h2", "producer", "k", now, 0, "user", "", None),           # runs a producer behind 3 tunnels
            ("a3", "h3", "consumer", "k", now, 0, "user", "", None),           # never signed in nor used
        ])
        conn.execute("CREATE TABLE usage (request_id TEXT, consumer_id TEXT, utc REAL)")
        conn.executemany("INSERT INTO usage VALUES (?,?,?)", [("r1", "a1", now - 3600), ("r2", "a1", now - 20 * 86400)])
        conn.execute("CREATE TABLE producer_endpoints (account_id TEXT, endpoint_url TEXT, execution_class TEXT, "
                     "last_heartbeat_utc REAL)")
        conn.executemany("INSERT INTO producer_endpoints VALUES (?,?,?,?)", [
            ("a2", "https://one.trycloudflare.com", "rtx2070", now - 60),
            ("a2", "https://two.trycloudflare.com", "rtx2070", now - 3 * 86400),
            ("a2", "https://three.trycloudflare.com", "rtx2070", now - 25 * 86400),
        ])
        conn.execute("CREATE TABLE opentrainer_nodes (account_id TEXT, node_id TEXT, last_heartbeat_utc REAL)")
        conn.execute("INSERT INTO opentrainer_nodes VALUES ('a1', 'n1', ?)", (now - 10 * 86400,))
    r = acc.build_accounts_receipt(db, now_utc=now)
    ev = r["activity_evidence"]
    assert r["source_status"] == "measured_sqlite" and r["total_accounts"] == 3
    assert ev["accounts_that_sent_requests"] == {"24h": 1, "7d": 1, "30d": 1}
    assert ev["accounts_with_password"] == 1 and ev["accounts_never_signed_in_nor_used"] == 1
    assert ev["producer_endpoint_rows"] == {"24h": 1, "7d": 2, "30d": 3}
    assert ev["producer_accounts_heartbeating"]["30d"] == 1 and ev["producer_hardware_classes_heartbeating"]["30d"] == 1
    assert ev["opentrainer_nodes_heartbeating"] == {"24h": 0, "7d": 0, "30d": 1}
    text = json.dumps(r)
    assert "trycloudflare" not in text and "pbkdf2" not in text and "h1" not in text

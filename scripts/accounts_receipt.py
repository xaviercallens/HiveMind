#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""PS-03-D1 — Generate an anonymized accounts and 30-day active-node receipt from RouterStore SQLite DB.

Counts accounts by ``kind`` (``user``, ``test``, ``probe``) and ``role`` (``consumer``, ``producer``,
``prosumer``, ``trainer``, ``admin``), and counts producer/trainer nodes active within the last 30 days,
while guaranteeing that zero emails, IP addresses, API keys, or endpoint tokens appear in the receipt.

Usage:
  python3 scripts/accounts_receipt.py --out artifacts/experiments/ACCOUNTS_2026-11-07.json
  python3 scripts/accounts_receipt.py --db /path/to/router.db --out artifacts/experiments/ACCOUNTS_2026-11-07.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "experiments" / "ACCOUNTS_2026-11-07.json"
DEFAULT_DB = Path(os.environ.get("HIVEMIND_ROUTER_DB", "/mnt/disks/disk-socrateai-local-1/hivemind_router/router.db"))

KNOWN_KINDS: tuple[str, ...] = ("user", "test", "probe")
KNOWN_ROLES: tuple[str, ...] = ("consumer", "producer", "prosumer", "trainer", "admin")
WINDOW_30D_SECONDS: float = 30.0 * 86400.0

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_API_KEY_RE = re.compile(r"\bhm-[A-Za-z0-9_-]{8,}\b")


def assert_zero_personal_data(serialized_json: str) -> None:
    """Raise ``ValueError`` if any email, IPv4 address, or API key appears in ``serialized_json``."""
    if _EMAIL_RE.search(serialized_json):
        raise ValueError("Personal data leak detected: email address found in accounts receipt")
    if _IPV4_RE.search(serialized_json):
        raise ValueError("Personal data leak detected: IPv4 address found in accounts receipt")
    if _API_KEY_RE.search(serialized_json):
        raise ValueError("Personal data leak detected: API key found in accounts receipt")


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table_name: str, col_name: str) -> bool:
    if not _table_exists(conn, table_name):
        return False
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]
    return col_name in cols


def build_accounts_receipt(
    db_path: Path | None = None,
    *,
    now_utc: float | None = None,
) -> dict[str, Any]:
    """Query ``db_path`` (if present) and return anonymized counts by kind, role, and 30-day active nodes."""
    now_ts = float(now_utc if now_utc is not None else time.time())
    cutoff_30d = now_ts - WINDOW_30D_SECONDS

    counts_by_kind: dict[str, int] = {k: 0 for k in KNOWN_KINDS}
    counts_by_role: dict[str, int] = {r: 0 for r in KNOWN_ROLES}
    by_kind_and_role: dict[str, dict[str, int]] = {
        k: {r: 0 for r in KNOWN_ROLES} for k in KNOWN_KINDS
    }
    producer_nodes_30d = 0
    trainer_nodes_30d = 0
    db_status = "absent"

    target_db = db_path if db_path is not None else DEFAULT_DB
    if target_db is not None and target_db.exists():
        db_status = "measured_sqlite"
        with sqlite3.connect(str(target_db)) as conn:
            if _table_exists(conn, "accounts"):
                has_kind = _column_exists(conn, "accounts", "kind")
                has_is_test = _column_exists(conn, "accounts", "is_test")
                if has_kind:
                    rows = conn.execute("SELECT kind, role, is_test FROM accounts").fetchall()
                elif has_is_test:
                    rows = conn.execute("SELECT 'user', role, is_test FROM accounts").fetchall()
                else:
                    rows = conn.execute("SELECT 'user', role, 0 FROM accounts").fetchall()

                for raw_kind, raw_role, raw_is_test in rows:
                    kind = str(raw_kind or ("test" if raw_is_test else "user")).lower()
                    if kind not in counts_by_kind:
                        kind = "test" if raw_is_test else "user"
                    role = str(raw_role or "consumer").lower()
                    if role not in counts_by_role:
                        role = "consumer"
                    counts_by_kind[kind] += 1
                    counts_by_role[role] += 1
                    by_kind_and_role[kind][role] += 1

            if _table_exists(conn, "producer_endpoints"):
                row = conn.execute(
                    "SELECT COUNT(*) FROM producer_endpoints WHERE last_heartbeat_utc >= ?",
                    (cutoff_30d,),
                ).fetchone()
                producer_nodes_30d = int(row[0] or 0) if row else 0
            elif _table_exists(conn, "producers"):
                row = conn.execute(
                    "SELECT COUNT(*) FROM producers WHERE last_heartbeat_utc >= ?",
                    (cutoff_30d,),
                ).fetchone()
                producer_nodes_30d = int(row[0] or 0) if row else 0

            if _table_exists(conn, "opentrainer_nodes"):
                row = conn.execute(
                    "SELECT COUNT(*) FROM opentrainer_nodes WHERE last_heartbeat_utc >= ?",
                    (cutoff_30d,),
                ).fetchone()
                trainer_nodes_30d = int(row[0] or 0) if row else 0

    receipt: dict[str, Any] = {
        "schema": "hivemind.accounts_receipt.v1",
        "roadmap_deliverable": "PS-03-D1",
        "generated_at_utc": dt.datetime.fromtimestamp(now_ts, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_status": db_status,
        "personal_data_included": False,
        "total_accounts": sum(counts_by_kind.values()),
        "counts_by_kind": counts_by_kind,
        "counts_by_role": counts_by_role,
        "by_kind_and_role": by_kind_and_role,
        "active_nodes_30d": {
            "window_days": 30,
            "producer_endpoints_active_30d": producer_nodes_30d,
            "opentrainer_nodes_active_30d": trainer_nodes_30d,
            "total_active_nodes_30d": producer_nodes_30d + trainer_nodes_30d,
        },
    }
    serialized = json.dumps(receipt, indent=2)
    assert_zero_personal_data(serialized)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None, help="Path to RouterStore SQLite database")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output JSON receipt path")
    args = parser.parse_args(argv)
    receipt = build_accounts_receipt(args.db)
    measured = receipt.get("source_status") == "measured_sqlite"
    receipt = {"evidence": "measured" if measured else "placeholder", "not_a_measurement": not measured, **receipt}
    serialized = json.dumps(receipt, indent=2) + "\n"
    assert_zero_personal_data(serialized)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

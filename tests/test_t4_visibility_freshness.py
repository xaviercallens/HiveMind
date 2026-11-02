# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The T4 visibility payload says whether telemetry was received and how old it is; malformed pushes are refused."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hivemind.router import RouterStore, create_router_app
from hivemind.router import t4_visibility as t4


@pytest.fixture()
def restore_snapshot():
    saved = (t4._LATEST_T4_SNAPSHOT["updated_epoch"], t4._LATEST_T4_SNAPSHOT["telemetry_source"], dict(t4._LATEST_T4_SNAPSHOT["gpu"]))
    yield
    t4._LATEST_T4_SNAPSHOT["updated_epoch"], t4._LATEST_T4_SNAPSHOT["telemetry_source"] = saved[0], saved[1]
    t4._LATEST_T4_SNAPSHOT["gpu"] = saved[2]


def test_default_snapshot_is_not_presented_as_fresh(restore_snapshot) -> None:
    t4._LATEST_T4_SNAPSHOT["updated_epoch"] = None
    t4._LATEST_T4_SNAPSHOT["telemetry_source"] = "DEFAULT_SNAPSHOT_NO_TELEMETRY_RECEIVED"
    payload = t4.build_t4_visibility_payload(now_utc=1_800_000_000.0)
    assert payload["telemetry_status"] == "NO_TELEMETRY_RECEIVED"
    assert payload["telemetry_age_s"] is None
    assert payload["telemetry_source"] == "DEFAULT_SNAPSHOT_NO_TELEMETRY_RECEIVED"


def test_live_update_is_fresh_then_stale(restore_snapshot) -> None:
    snap = t4.update_t4_telemetry_snapshot({"gpu": {"used_mib": 5000, "total_mib": 15360}})
    assert snap["telemetry_source"] == "LIVE_UPDATE" and snap["gpu"]["free_mib"] == 10360
    fresh = t4.build_t4_visibility_payload(now_utc=snap["updated_epoch"] + 10.0)
    assert fresh["telemetry_status"] == "FRESH" and fresh["telemetry_age_s"] == 10.0
    stale = t4.build_t4_visibility_payload(now_utc=snap["updated_epoch"] + t4.TELEMETRY_FRESH_WINDOW_S + 1.0)
    assert stale["telemetry_status"] == "STALE"


def test_malformed_numbers_are_refused_in_module_and_over_http(restore_snapshot) -> None:
    with pytest.raises(ValueError):
        t4.update_t4_telemetry_snapshot({"gpu": {"used_mib": "x"}})
    with pytest.raises(ValueError):
        t4.update_t4_telemetry_snapshot({"gpu": {"temp_c": float("nan")}})
    store = RouterStore(":memory:", b"k" * 32)
    client = TestClient(create_router_app(store))
    key = client.post("/v1/hivemind/register", json={"handle": "t4_pusher", "role": "producer"}).json()["api_key"]
    bad = client.post("/v1/hivemind/t4/telemetry", headers={"Authorization": f"Bearer {key}"}, json={"gpu": {"used_mib": "x"}})
    assert bad.status_code == 400 and "used_mib" in bad.text
    good = client.post("/v1/hivemind/t4/telemetry", headers={"Authorization": f"Bearer {key}"}, json={"gpu": {"used_mib": 4841}})
    assert good.status_code == 200 and good.json()["updated_epoch"] is not None
    assert client.get("/v1/hivemind/t4/status").json()["telemetry_status"] == "FRESH"

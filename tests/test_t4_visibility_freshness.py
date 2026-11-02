# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The T4 visibility payload says whether telemetry was received and how old it is; malformed pushes are refused."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hivemind.router import RouterStore, create_router_app
from hivemind.router import t4_visibility as t4


@pytest.fixture()
def restore_snapshot():
    saved = (
        t4._LATEST_T4_SNAPSHOT["updated_epoch"],
        t4._LATEST_T4_SNAPSHOT["telemetry_source"],
        dict(t4._LATEST_T4_SNAPSHOT["gpu"]),
        dict(t4._LATEST_T4_SNAPSHOT["governor"]),
        dict(t4._LATEST_T4_SNAPSHOT["qlora_progress"]),
    )
    yield
    t4._LATEST_T4_SNAPSHOT["updated_epoch"], t4._LATEST_T4_SNAPSHOT["telemetry_source"] = saved[0], saved[1]
    t4._LATEST_T4_SNAPSHOT["gpu"] = saved[2]
    t4._LATEST_T4_SNAPSHOT["governor"] = saved[3]
    t4._LATEST_T4_SNAPSHOT["qlora_progress"] = saved[4]


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


def test_governor_and_qlora_pushes_are_validated_and_a_bad_push_changes_nothing(restore_snapshot) -> None:
    before_gov = dict(t4._LATEST_T4_SNAPSHOT["governor"])
    before_q = dict(t4._LATEST_T4_SNAPSHOT["qlora_progress"])
    for bad in (
        {"governor": {"in_flight_requests": "x"}},
        {"qlora_progress": {"verified_cobblestones": float("nan"), "assimilated_workunits": 1e999}},
        {"governor": {"not_a_known_field": 1}},
        {"gpu": {"resident_processes": [{"name": "<script>" * 200, "mib": 1}]}},
        {"governor": {"state": "s" * 300}},
    ):
        with pytest.raises(ValueError):
            t4.update_t4_telemetry_snapshot(bad)
    assert t4._LATEST_T4_SNAPSHOT["governor"] == before_gov
    assert t4._LATEST_T4_SNAPSHOT["qlora_progress"] == before_q
    t4.build_t4_visibility_payload(now_utc=1_800_000_000.0)  # must not raise after the bad pushes
    good = t4.update_t4_telemetry_snapshot({"governor": {"in_flight_requests": 2, "state": "yielded"}, "qlora_progress": {"assimilated_workunits": 161}})
    assert good["governor"]["in_flight_requests"] == 2 and good["qlora_progress"]["assimilated_workunits"] == 161
    store = RouterStore(":memory:", b"k" * 32)
    client = TestClient(create_router_app(store))
    key = client.post("/v1/hivemind/register", json={"handle": "t4_pusher2", "role": "producer"}).json()["api_key"]
    r = client.post("/v1/hivemind/t4/telemetry", headers={"Authorization": f"Bearer {key}"}, json={"governor": {"in_flight_requests": "x"}})
    assert r.status_code == 400 and "in_flight_requests" in r.text
    assert client.get("/v1/hivemind/t4/status").status_code == 200


def test_live_telemetry_t4_block_reflects_pushed_values_and_states_freshness(restore_snapshot) -> None:
    store = RouterStore(":memory:", b"k" * 32)
    client = TestClient(create_router_app(store))
    t4._LATEST_T4_SNAPSHOT["updated_epoch"] = None
    t4._LATEST_T4_SNAPSHOT["telemetry_source"] = "DEFAULT_SNAPSHOT_NO_TELEMETRY_RECEIVED"
    before = client.get("/v1/hivemind/live-telemetry").json()["t4_boinc"]
    assert before["telemetry_status"] == "NO_TELEMETRY_RECEIVED"
    assert before["data_source"].startswith("T4_SNAPSHOT_NOT_LIVE")
    key = client.post("/v1/hivemind/register", json={"handle": "t4_pusher3", "role": "producer"}).json()["api_key"]
    r = client.post(
        "/v1/hivemind/t4/telemetry",
        headers={"Authorization": f"Bearer {key}"},
        json={"gpu": {"used_mib": 1, "total_mib": 15360}, "qlora_progress": {"assimilated_workunits": 161}, "governor": {"state": "yielded"}},
    )
    assert r.status_code == 200
    after = client.get("/v1/hivemind/live-telemetry").json()["t4_boinc"]
    assert after["vram_used_mib"] == 1 and after["vram_free_mib"] == 15359
    assert after["assimilated_workunits"] == 161 and after["governor_state"] == "yielded"
    assert after["telemetry_status"] == "FRESH" and after["data_source"] == "T4_LIVE_PUSH"

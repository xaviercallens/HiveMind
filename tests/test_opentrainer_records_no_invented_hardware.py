# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""OpenTrainer must record what a node reported and nothing else: a heartbeat with no hardware fields used to be stored as an RTX 2070 with
8192 / 6400 MiB, and every delta submission stored 6400 / 8192 MiB and tokens = inner_steps * 512 as if they were readings (audit 2026-11-03)."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app

SECRET = b"hivemind-test-secret-at-least-32-bytes-long!"


def _client() -> tuple[TestClient, RouterStore, dict[str, str], str]:
    store = RouterStore(":memory:", SECRET)
    client = TestClient(create_router_app(store, starter_grant_hmt=0.0))
    aid, key = store.register("ot-node", "trainer")
    return client, store, {"Authorization": f"Bearer {key}"}, aid


def _invented_hardware_figures(text: str) -> list[str]:
    """The old placeholder figures, matched where they would be stored as a reading.

    A bare substring test for 6400 or 8192 also matches random hexadecimal ids such as ``ot-640098d0...`` (it failed CI
    that way), so the figures are only counted when they sit in a VRAM field.
    """
    return re.findall(r"vram_(?:total|free)_mib'?\"?: ?(?:6400|8192)\b", text)


def test_a_heartbeat_without_hardware_is_recorded_as_not_reported() -> None:
    client, store, auth, aid = _client()
    r = client.post("/v1/hivemind/opentrainer/heartbeat", headers=auth, json={"host_label": "box-1"})
    assert r.status_code == 200, r.text
    assert r.status_code == 200, r.text
    (node,) = store.list_opentrainer_nodes()
    assert node["account_id"] == aid
    assert node["gpu_name"] == "not reported" and node["vram_total_mib"] == 0 and node["vram_free_mib"] == 0


def test_a_heartbeat_that_reports_hardware_is_recorded_as_reported() -> None:
    client, _store, auth, _aid = _client()
    r = client.post(
        "/v1/hivemind/opentrainer/heartbeat",
        headers=auth,
        json={"host_label": "box-2", "gpu_name": "Tesla T4", "vram_total_mib": 15360, "vram_free_mib": 9000},
    )
    assert r.status_code == 200, r.text
    assert "Tesla T4" in str(r.json()) and "15360" in str(r.json())


def test_a_delta_submission_records_unknown_hardware_and_flags_the_token_estimate() -> None:
    client, _store, auth, _aid = _client()
    r = client.post(
        "/v1/hivemind/opentrainer/submit-delta",
        headers=auth,
        json={"round_id": 3, "inner_steps": 8, "loss_before": 2.0, "loss_after": 1.5, "delta_norm": 0.7, "delta_sha256": "a" * 64},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    text = str(body)
    assert not _invented_hardware_figures(text), text
    assert "CLIENT_REPORTED_UNVERIFIED" in text
    assert "tokens_trained_is_estimate" in text and "'vram_free_mib': 0" in text and "'vram_total_mib': 0" in text


def test_a_random_submission_id_containing_the_old_figures_is_not_mistaken_for_invented_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _store, auth, _aid = _client()
    monkeypatch.setattr("hivemind.router.accounts.secrets.token_hex", lambda n=16: "6400819264008192"[: 2 * n])
    r = client.post(
        "/v1/hivemind/opentrainer/submit-delta",
        headers=auth,
        json={"round_id": 4, "inner_steps": 8, "loss_before": 2.0, "loss_after": 1.5, "delta_norm": 0.7, "delta_sha256": "b" * 64},
    )
    assert r.status_code == 200, r.text
    text = str(r.json())
    assert "ot-6400819264008192" in text  # the id really carries the digits the old check rejected
    assert _invented_hardware_figures(text) == []
    # and the check still catches a stored placeholder reading
    assert _invented_hardware_figures("{'vram_total_mib': 6400}") and _invented_hardware_figures("{'vram_free_mib': 8192}")

# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-control hardening suite for router google_auth, points_wording, telemetry_figures, and t4_visibility."""

from __future__ import annotations

import pytest

from hivemind.router import t4_visibility as t4
from hivemind.router.google_auth import GoogleIdTokenVerifier
from hivemind.router.points_wording import UNIT, currency_leftovers, to_points_wording
from hivemind.router.telemetry_figures import NOT_AVAILABLE, shown_figure

VALID_CLIENT_ID = "123456789012-abcdefghijklmnop.apps.googleusercontent.com"


@pytest.fixture()
def restore_t4_snapshot():
    saved = (
        t4._LATEST_T4_SNAPSHOT["updated_epoch"],
        t4._LATEST_T4_SNAPSHOT["telemetry_source"],
        t4._LATEST_T4_SNAPSHOT["sole_active_gpu"],
        dict(t4._LATEST_T4_SNAPSHOT["gpu"]),
        dict(t4._LATEST_T4_SNAPSHOT["governor"]),
        dict(t4._LATEST_T4_SNAPSHOT["qlora_progress"]),
    )
    yield
    t4._LATEST_T4_SNAPSHOT["updated_epoch"] = saved[0]
    t4._LATEST_T4_SNAPSHOT["telemetry_source"] = saved[1]
    t4._LATEST_T4_SNAPSHOT["sole_active_gpu"] = saved[2]
    t4._LATEST_T4_SNAPSHOT["gpu"] = saved[3]
    t4._LATEST_T4_SNAPSHOT["governor"] = saved[4]
    t4._LATEST_T4_SNAPSHOT["qlora_progress"] = saved[5]


def test_google_auth_verifier_rejects_invalid_constructor_args() -> None:
    for bad_cid in (None, 123, ["id"], "", "invalid"):
        with pytest.raises(ValueError, match="client_id"):
            GoogleIdTokenVerifier(bad_cid)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="jwks_fetcher"):
        GoogleIdTokenVerifier(VALID_CLIENT_ID, jwks_fetcher="not-callable")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="clock"):
        GoogleIdTokenVerifier(VALID_CLIENT_ID, clock=123)  # type: ignore[arg-type]

    for bad_max_age in (True, False, 0, -5, float("nan"), float("inf"), "600"):
        with pytest.raises(ValueError, match="max_age_s"):
            GoogleIdTokenVerifier(VALID_CLIENT_ID, max_age_s=bad_max_age)  # type: ignore[arg-type]

    for bad_leeway in (True, False, -1, float("nan"), float("inf"), "30"):
        with pytest.raises(ValueError, match="leeway_s"):
            GoogleIdTokenVerifier(VALID_CLIENT_ID, leeway_s=bad_leeway)  # type: ignore[arg-type]

    verifier = GoogleIdTokenVerifier(VALID_CLIENT_ID, max_age_s=600.0, leeway_s=0.0)
    assert verifier is not None


def test_points_wording_rejects_non_string_inputs() -> None:
    for bad_text in (None, 123, ["0.00 EUR"], {"text": "0.00 EUR"}):
        with pytest.raises(ValueError, match="text"):
            to_points_wording(bad_text)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="text"):
            currency_leftovers(bad_text)  # type: ignore[arg-type]

    rewritten = to_points_wording("Reciprocity: 0.00 EUR")
    assert UNIT in rewritten
    assert currency_leftovers(rewritten) == []


def test_telemetry_figures_rejects_invalid_decimals_and_formats_values() -> None:
    for bad_dec in (True, False, -1, 13, 1.5, "2"):
        with pytest.raises(ValueError, match="decimals"):
            shown_figure(100, decimals=bad_dec)  # type: ignore[arg-type]

    assert shown_figure(None) == NOT_AVAILABLE
    assert shown_figure(True) == NOT_AVAILABLE
    assert shown_figure(float("nan")) == NOT_AVAILABLE
    assert shown_figure(float("inf")) == NOT_AVAILABLE
    assert shown_figure(1234) == "1,234"
    assert shown_figure(1234.567, decimals=2) == "1,234.57"
    assert shown_figure(1234.567, decimals=0) == "1,235"


def test_t4_visibility_format_update_and_payload_validation(restore_t4_snapshot) -> None:
    with pytest.raises(ValueError, match="record"):
        t4.format_workunit_counter_label("not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tolerance"):
        t4.format_workunit_counter_label({}, tolerance=-1e-6)
    with pytest.raises(ValueError, match="tolerance"):
        t4.format_workunit_counter_label({}, tolerance=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="baseline_nll"):
        t4.format_workunit_counter_label({}, baseline_nll=0.0)

    assert (
        t4.format_workunit_counter_label({"eval_loss_before": 2.10, "eval_loss_after": 2.05})
        == "assimilated, held-out gain"
    )
    assert (
        t4.format_workunit_counter_label({"eval_loss_before": 2.05, "eval_loss_after": 2.10})
        == "assimilated, no held-out gain"
    )

    with pytest.raises(ValueError, match="update"):
        t4.update_t4_telemetry_snapshot("not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sole_active_gpu"):
        t4.update_t4_telemetry_snapshot({"sole_active_gpu": "yes"})  # type: ignore[dict-item]
    for section in ("gpu", "governor", "qlora_progress"):
        with pytest.raises(ValueError, match=section):
            t4.update_t4_telemetry_snapshot({section: "not-an-object"})

    for bad_now in (True, -1.0, float("nan"), float("inf"), "1700000000"):
        with pytest.raises(ValueError, match="now_utc"):
            t4.build_t4_visibility_payload(now_utc=bad_now)  # type: ignore[arg-type]

    for bad_count in (True, False, -1, 1.5, "1"):
        with pytest.raises(ValueError, match="active_producers_count"):
            t4.build_t4_visibility_payload(active_producers_count=bad_count)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="distinct_physical_gpus"):
            t4.build_t4_visibility_payload(distinct_physical_gpus=bad_count)  # type: ignore[arg-type]

    for bad_models in ("not-a-list", [""], [123]):
        with pytest.raises(ValueError, match="live_models"):
            t4.build_t4_visibility_payload(live_models=bad_models)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="secondary_gpu"):
        t4.build_t4_visibility_payload(secondary_gpu=123)  # type: ignore[arg-type]

    payload = t4.build_t4_visibility_payload(
        now_utc=1_800_000_000.0,
        live_models=["hivemind/coder"],
        active_producers_count=1,
        distinct_physical_gpus=1,
    )
    assert payload["status"] == "ok"
    assert payload["distinct_physical_gpus"] == 1

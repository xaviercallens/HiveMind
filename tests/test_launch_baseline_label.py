"""The launch model's baseline is a hard-coded snapshot; the payload must not call it live telemetry."""

from __future__ import annotations

from hivemind.economy.international_launch_model import SwarmLiveBaseline, forecast_200plus_node_international_launch


def test_default_baseline_is_labelled_assumed_not_live() -> None:
    base = forecast_200plus_node_international_launch()["current_live_baseline"]
    assert base["data_source"] == "ASSUMED_BASELINE_SNAPSHOT_NOT_LIVE"
    assert base["baseline_source"] == "module_default"
    assert "LIVE" not in base["data_source"].replace("NOT_LIVE", "")
    assert base["total_nodes"] == SwarmLiveBaseline().total_accounts


def test_caller_supplied_baseline_is_labelled_as_such_and_carries_its_values() -> None:
    """Negative control: a baseline the caller measured is not mislabelled as the module default."""
    base = forecast_200plus_node_international_launch(baseline=SwarmLiveBaseline(total_accounts=58))["current_live_baseline"]
    assert base["data_source"] == "CALLER_SUPPLIED_BASELINE"
    assert base["baseline_source"] == "caller_supplied"
    assert base["total_nodes"] == 58


def test_a_baseline_equal_to_the_default_in_every_field_still_counts_as_the_default() -> None:
    base = forecast_200plus_node_international_launch(baseline=SwarmLiveBaseline())["current_live_baseline"]
    assert base["baseline_source"] == "module_default"

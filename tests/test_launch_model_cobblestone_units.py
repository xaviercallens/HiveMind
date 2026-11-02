# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""BOINC cobblestones in the launch model use the BOINC definition (200 per GFLOPS-day) on TFLOPS inputs, and are labelled modelled."""

from __future__ import annotations

from typing import Any

import pytest

from hivemind.economy.international_launch_model import COBBLESTONES_PER_GFLOPS_DAY, forecast_200plus_node_international_launch


def _find(obj: Any, key: str) -> list[Any]:
    out: list[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            out.extend(_find(v, key))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_find(v, key))
    return out


def _holders(obj: Any, key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        if key in obj:
            out.append(obj)
        for v in obj.values():
            out.extend(_holders(v, key))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_holders(v, key))
    return out


def test_cobblestones_scale_with_gflops_days_and_are_labelled_modelled() -> None:
    payload = forecast_200plus_node_international_launch()
    assert COBBLESTONES_PER_GFLOPS_DAY == 200.0
    holders = _holders(payload, "daily_modelled_boinc_cobblestones")
    assert len(holders) == 1
    block = holders[0]
    blended = block["daytime_inference_tflops_fp16"] * 0.45 + block["nocturnal_diloco_tflops_fp16"] * 0.55
    assert block["daily_modelled_boinc_cobblestones"] == pytest.approx(blended * 1000.0 * 200.0, rel=1e-3)
    assert block["daily_modelled_boinc_cobblestones"] > blended * 200.0 * 10  # not the old GFLOPS-as-TFLOPS figure
    assert _find(payload, "daily_verified_boinc_cobblestones") == []
    basis = _find(payload, "cobblestones_basis")
    assert basis and "not a verified" in basis[0]

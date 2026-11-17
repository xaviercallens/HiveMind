# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for scripts/hnet_t4_probe.py (issue #448 H-Net T4 probe harness and decision rule)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hnet_t4_probe.py"
spec = importlib.util.spec_from_file_location("hnet_t4_probe", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
sys.modules["hnet_t4_probe"] = probe
spec.loader.exec_module(probe)


def test_preflight_enforces_4300_mib_floor_and_zero_router_load() -> None:
    # Negative controls: router load active, unreadable VRAM, or free VRAM below 4,300 MiB must block.
    busy = probe.check_probe_preflight(free_vram_mib=8000.0, router_in_flight=2)
    assert busy["allowed"] is False and "router_load_active" in busy["reason"]

    unreadable = probe.check_probe_preflight(free_vram_mib=None, router_in_flight=0)
    assert unreadable["allowed"] is False and unreadable["reason"] == "gpu_vram_unreadable"

    low_vram = probe.check_probe_preflight(free_vram_mib=4299.0, router_in_flight=0)
    assert low_vram["allowed"] is False and "free_vram_below_floor" in low_vram["reason"]

    # Positive control: idle router and >= 4,300 MiB free VRAM is permitted.
    ok = probe.check_probe_preflight(free_vram_mib=4609.0, router_in_flight=0)
    assert ok["allowed"] is True and ok["free_vram_mib"] == pytest.approx(4609.0)


def test_analytic_boundary_bytes_shows_1stage_hnet_below_2x_bpe_reduction() -> None:
    m4 = probe.compute_analytic_boundary_bytes(
        hnet_dim=1536,
        qwen_dim=896,
        dtype_bytes=2,
        bytes_per_hnet_chunk=5.0,
        bytes_per_bpe_token=4.5,
    )
    assert m4["status"] == "MEASURED"
    assert m4["hnet_boundary_bytes_per_generated_byte"] == pytest.approx(614.4)
    assert m4["bpe_boundary_bytes_per_generated_byte"] == pytest.approx(398.2222, rel=1e-3)
    assert m4["reduction_factor_vs_bpe_split"] < 1.0
    assert m4["meets_2x_reduction_vs_bpe_threshold"] is False
    assert m4["reduction_factor_vs_raw_byte_routing"] == pytest.approx(5.0)


def test_build_probe_report_parks_chunk_routing_and_preserves_not_run_rows() -> None:
    report = probe.build_probe_report(free_vram_mib=4609.0, router_in_flight=0)
    assert report["hn1_concept_probe"]["status"] == "MEASURED"
    assert report["hn1_concept_probe"]["experiment_valid"] is False
    assert report["measurements"]["4_activation_boundary_bytes"]["status"] == "MEASURED"
    assert report["measurements"]["1_sm75_checkpoint_load"]["status"] == "NOT_RUN"
    assert report["measurements"]["5_held_out_bits_per_byte"]["status"] == "NOT_RUN"
    assert report["decision"]["continue_to_two_host_pipeline_split"] is False
    assert report["decision"]["verdict"] == "PARK_CHUNK_LEVEL_ROUTING"
    assert report["decision"]["governor_or_hold_modified"] is False

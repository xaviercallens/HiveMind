# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for the 3-Hour Autonomous RTX 2070 Progress Monitor & PR Integrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from scripts.rtx2070_3h_monitor_and_pr_integrator import (
    MIN_FREE_VRAM_HEADROOM_MIB,
    WDDM_VRAM_CEILING_MIB,
    audit_and_integrate_open_prs,
    probe_rtx2070_hardware_and_tflops,
    sync_monitor_artifacts,
)


def test_probe_rtx2070_hardware_and_tflops_structure() -> None:
    telemetry = probe_rtx2070_hardware_and_tflops(run_cuda_microbench=True)
    assert "gpu_name" in telemetry
    assert telemetry["wddm_ceiling_mib"] == WDDM_VRAM_CEILING_MIB
    assert telemetry["vram_headroom_guard_mib"] == MIN_FREE_VRAM_HEADROOM_MIB
    assert isinstance(telemetry["wddm_envelope_compliant"], bool)
    assert isinstance(telemetry["headroom_guard_compliant"], bool)
    assert telemetry["memory_total_mib"] >= 0.0


def test_sync_monitor_artifacts_updates_ledger_and_traffic_monitor(tmp_path: Path) -> None:
    traffic_seed = {
        "status": "OLD",
        "producer_pool": {
            "producers": [
                {
                    "producer_id": "PRODUCER-LOCAL-RTX2070-CUDA0",
                    "device": "cpu",
                    "vram_allocated_mb": 0.0,
                    "vram_free_gb": 0.0,
                }
            ]
        },
    }
    (tmp_path / "hivemind_traffic_tflops_monitor.json").write_text(
        json.dumps(traffic_seed), encoding="utf-8"
    )
    gpu_telemetry: Dict[str, Any] = {
        "timestamp_utc": "2026-10-30T14:45:00Z",
        "gpu_name": "NVIDIA GeForce RTX 2070",
        "device": "nvml:cuda:0 (torch+cpu host)",
        "compute_capability": "7.5",
        "temperature_c": 29.0,
        "power_draw_w": 18.5,
        "memory_used_mib": 1445.0,
        "memory_free_mib": 6562.0,
        "measured_fp32_tflops": 7.88,
        "measured_fp16_tensor_tflops": 31.52,
        "effective_sparse_moe_nf4_tflops": 52.01,
    }
    pr_summary: Dict[str, Any] = {
        "checked_utc": "2026-10-30T14:45:00Z",
        "open_prs_count": 1,
        "evaluated": [],
        "merged_prs": [344],
        "held_prs": [],
    }
    ledger = sync_monitor_artifacts(gpu_telemetry, pr_summary, artifacts_dir=tmp_path)
    assert ledger["schema_version"] == "rtx2070_3h_monitor.v1"
    assert ledger["cadence_cron"] == "0 */3 * * *"
    assert len(ledger["ledger_sha256"]) == 64

    updated_traffic = json.loads(
        (tmp_path / "hivemind_traffic_tflops_monitor.json").read_text(encoding="utf-8")
    )
    assert updated_traffic["tflops_achieved"]["primary_gpu"] == "NVIDIA GeForce RTX 2070"
    assert updated_traffic["tflops_achieved"]["measured_fp16_tensor_tflops"] == 31.52
    prod = updated_traffic["producer_pool"]["producers"][0]
    assert prod["vram_allocated_mb"] == 1445.0
    assert prod["vram_free_gb"] == pytest.approx(6.41, abs=0.02)


def test_audit_and_integrate_open_prs_without_token_returns_clean_error() -> None:
    res = audit_and_integrate_open_prs(token="", merge=False)
    assert res["error"] == "no_github_token"
    assert res["open_prs_count"] == 0

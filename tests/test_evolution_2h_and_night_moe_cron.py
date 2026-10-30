# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for scripts/run_2h_roadmap_pr_and_night_moe_cron.py."""

from __future__ import annotations

from pathlib import Path

from scripts.run_2h_roadmap_pr_and_night_moe_cron import (
    probe_local_8095_producer_server,
    run_night_moe_training_cycle,
    scan_roadmap_status,
)


def test_scan_roadmap_status_detects_season_33() -> None:
    res = scan_roadmap_status()
    assert res["exists"] is True
    assert res["max_season"] >= 33
    assert res["total_task_ids"] >= 190


def test_probe_local_8095_producer_server_returns_dict() -> None:
    res = probe_local_8095_producer_server()
    assert isinstance(res.get("online"), bool)
    assert "pool_status" in res


def test_run_night_moe_training_cycle_enforces_vram_guards(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_2h_roadmap_pr_and_night_moe_cron as mod

    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    summary = run_night_moe_training_cycle(
        {
            "gpu_name": "NVIDIA GeForce RTX 2070",
            "memory_used_mib": 1458.0,
            "memory_free_mib": 6549.0,
            "effective_sparse_moe_nf4_tflops": 65.24,
        }
    )
    assert summary["status"] == "ACTIVE_OVERNIGHT_EXECUTION"
    assert summary["rtx2070_live_guard"]["wddm_ceiling_6656_mib_ok"] is True
    assert summary["rtx2070_live_guard"]["headroom_2048_mib_ok"] is True

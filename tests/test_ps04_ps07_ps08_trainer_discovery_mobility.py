# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Verification suite for PS-04-D1/D2 (Trainer Install Smoke & Multi-GPU Hardware Receipt),
PS-07-D2/D3 (E22 Gateway-Off Discovery & Petals Liveness Receipt), and PS-08-D3 (E23 AMCP Mobility)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import scripts.petals_liveness_receipt as petals_liveness
import scripts.run_experiment_E22 as exp_e22
import scripts.run_experiment_E23 as exp_e23
import scripts.trainer_hardware_receipt as hw_receipt

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ps04_d1_trainer_install_smoke_receipt() -> None:
    """PS-04-D1: elapsed_s < 900 and first_work_unit_accepted is True with 0 inbound sockets."""
    path = REPO_ROOT / "artifacts" / "experiments" / "TRAINER_INSTALL_2026-11-07.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["deliverable_id"] == "PS-04-D1"
    assert data["elapsed_s"] < 900.0
    assert data["first_work_unit_accepted"] is True
    assert data["inbound_listening_sockets_opened_by_worker"] == 0
    assert data["passes_do_d"] is True


def test_ps04_d2_trainer_hardware_receipt_and_negative_controls() -> None:
    """PS-04-D2: >= 3 distinct GPU classes; every row vram_peak_mib < ceiling_mib and pause_ms < 1500."""
    receipt = hw_receipt.build_trainer_hardware_receipt()
    assert receipt["deliverable_id"] == "PS-04-D2"
    assert receipt["distinct_gpu_classes_count"] >= 3
    assert receipt["all_rows_valid"] is True
    assert hw_receipt.validate_trainer_hardware_receipt(receipt) == []

    path = REPO_ROOT / "artifacts" / "experiments" / "TRAINER_HARDWARE_2026-11-07.json"
    assert path.is_file()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert hw_receipt.validate_trainer_hardware_receipt(saved) == []

    # Negative control 1: vram_peak_mib >= ceiling_mib rejected
    bad_vram = copy.deepcopy(receipt)
    bad_vram["hardware_rows"][0]["vram_peak_mib"] = bad_vram["hardware_rows"][0]["ceiling_mib"]
    v1 = hw_receipt.validate_trainer_hardware_receipt(bad_vram)
    assert any("vram_peak_mib" in msg for msg in v1)

    # Negative control 2: pause_ms >= 1500 rejected
    bad_pause = copy.deepcopy(receipt)
    bad_pause["hardware_rows"][1]["pause_ms"] = 1600.0
    v2 = hw_receipt.validate_trainer_hardware_receipt(bad_pause)
    assert any("pause_ms" in msg for msg in v2)

    # Negative control 3: fewer than 3 distinct GPU classes rejected
    fewer_classes = copy.deepcopy(receipt)
    fewer_classes["hardware_rows"] = fewer_classes["hardware_rows"][:2]
    v3 = hw_receipt.validate_trainer_hardware_receipt(fewer_classes)
    assert any("distinct GPU classes" in msg for msg in v3)


def test_ps07_d2_e22_gateway_off_discovery_report() -> None:
    """PS-07-D2: Discovery time and false-peer rate within registered thresholds with gateway off."""
    report = exp_e22.run_e22_gateway_off_discovery()
    assert report["deliverable_id"] == "PS-07-D2"
    assert report["gateway_online"] is False
    assert report["metrics"]["false_peer_rate"] == 0.0
    assert report["metrics"]["sybils_rejected"] == report["metrics"]["sybils_evaluated"]
    assert report["metrics"]["discovery_time_ms_p95"] <= report["registered_thresholds"]["max_discovery_p95_ms"]
    assert report["passes_registered_thresholds"] is True

    path = REPO_ROOT / "artifacts" / "experiments" / "E22_discovery_report.json"
    assert path.is_file()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["gateway_online"] is False
    assert saved["passes_registered_thresholds"] is True


def test_ps07_d3_petals_liveness_receipt() -> None:
    """PS-07-D3: 4 weekly liveness checks committed and producer-sees-prompts notice included."""
    receipt = petals_liveness.build_petals_liveness_receipt()
    assert receipt["deliverable_id"] == "PS-07-D3"
    assert receipt["weekly_checks_count"] >= 4
    assert receipt["all_weeks_live"] is True
    assert "Prompts routed to a community producer are visible" in receipt["producer_sees_prompts_notice"]

    path = REPO_ROOT / "artifacts" / "experiments" / "PETALS_LIVENESS_2026-11-07.json"
    assert path.is_file()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert len(saved["weekly_checks"]) >= 4
    assert saved["all_weeks_live"] is True


def test_ps08_d3_e23_amcp_mobility_report_at_100_and_1000_bees() -> None:
    """PS-08-D3: AMCP migrate/replicate results at 100 and 1,000 bees pass registered thresholds."""
    report = exp_e23.build_e23_report()
    assert report["deliverable_id"] == "PS-08-D3"
    assert report["passes_all_scales"] is True
    for scale_key, expected_n in (("100", 100), ("1000", 1000)):
        row = report["scales"][scale_key]
        assert row["bee_count"] == expected_n
        assert row["unique_bees_restored"] == expected_n
        assert row["state_loss_count"] == 0
        assert row["state_loss_rate"] == 0.0
        assert row["post_dedup_duplicate_deliveries"] == 0
        assert row["raw_wire_frames_sent"] > expected_n  # proves duplicate wire retries were injected and deduplicated
        assert row["passes_registered_thresholds"] is True

    path = REPO_ROOT / "artifacts" / "experiments" / "E23_amcp_mobility_report.json"
    assert path.is_file()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["passes_all_scales"] is True

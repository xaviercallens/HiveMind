# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for Tonight's Wave 3 Scientific MoE Bee dataset generator, pre-registration freezer, and orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hivemind.core.expert_registry import (
    WAVE3_PRIMARY_BASE_MODEL_ID,
    WAVE3_SCIENTIFIC_EXPERT_SPECS,
)
from scripts.prepare_night_sci_wave3_datasets import prepare_wave3_datasets
from scripts.run_night_sci_wave3_campaign import (
    VRAMHeadroomError,
    check_vram_headroom,
    evaluate_5x5_cross_domain_matrix,
    exact_mcnemar_p_value,
    run_wave3_campaign,
    verify_preregistration_integrity,
)


def test_prepare_wave3_datasets_and_verify_prereg(tmp_path: Path) -> None:
    prereg = prepare_wave3_datasets(tmp_path, n_train=8, n_val=4, n_test=4)
    prereg_path = tmp_path / "NIGHT_SCI_WAVE3_prereg.json"
    assert prereg_path.exists()
    assert prereg["primary_base_model_id"] == WAVE3_PRIMARY_BASE_MODEL_ID
    assert set(prereg["bees"]) == set(WAVE3_SCIENTIFIC_EXPERT_SPECS)

    loaded = verify_preregistration_integrity(prereg_path)
    assert loaded["campaign_id"] == "NIGHT-SCI-WAVE3-2026"

    # Tamper with one file and verify SHA-256 integrity check fails closed
    qf_test = Path(prereg["bees"]["bee_quantum_fluid"]["splits"]["held_out_test"]["path"])
    orig_bytes = qf_test.read_bytes()
    qf_test.write_bytes(orig_bytes + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_preregistration_integrity(prereg_path)
    qf_test.write_bytes(orig_bytes)


def test_vram_headroom_guard_enforces_4300_mib_floor() -> None:
    assert check_vram_headroom(4300, vram_probe=lambda: 10073) == 10073
    with pytest.raises(VRAMHeadroomError, match="below required 4300 MiB floor"):
        check_vram_headroom(4300, vram_probe=lambda: 4100)


def test_mcnemar_and_5x5_cross_domain_promotion() -> None:
    p_sig = exact_mcnemar_p_value(16, 1)
    assert p_sig < 0.01
    p_insig = exact_mcnemar_p_value(5, 4)
    assert p_insig > 0.05

    metrics = {
        bee_id: {
            "in_domain_acc": 0.90,
            "base_acc": 0.64,
            "max_off_diagonal_acc": 0.58,
            "discordant_expert_wins": 16,
            "discordant_base_wins": 1,
            "wrong_verified_count": 0,
        }
        for bee_id in WAVE3_SCIENTIFIC_EXPERT_SPECS
    }
    # Force one bee to fail Gate G3 (wrong answer labeled VERIFIED)
    metrics["bee_topology_tda_lean"]["wrong_verified_count"] = 1

    summary = evaluate_5x5_cross_domain_matrix(metrics)
    assert "bee_topology_tda_lean" not in summary["promoted_to_shadow"]
    assert len(summary["promoted_to_shadow"]) == 4
    assert summary["residency_violations"] == []


def test_run_wave3_campaign_end_to_end(tmp_path: Path) -> None:
    prepare_wave3_datasets(tmp_path, n_train=8, n_val=4, n_test=4)
    prereg_path = tmp_path / "NIGHT_SCI_WAVE3_prereg.json"
    report = run_wave3_campaign(prereg_path, vram_probe=lambda: 10073)
    assert report["free_vram_mib"] == 10073
    assert set(report["evaluation"]["promoted_to_shadow"]) == set(WAVE3_SCIENTIFIC_EXPERT_SPECS)
    assert (tmp_path / "wave3_expert_registry.json").exists()
    saved_reg = json.loads((tmp_path / "wave3_expert_registry.json").read_text(encoding="utf-8"))
    assert len(saved_reg["experts"]) == 5

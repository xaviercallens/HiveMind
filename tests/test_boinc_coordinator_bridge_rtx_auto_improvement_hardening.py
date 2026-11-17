# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and invariant tests for BOINC coordinator, real_client bridge, RTX BOINC deployer, and Auto-Improvement dataclasses (IP-163)."""

from __future__ import annotations

import pytest

from hivemind.boinc.coordinator_api import NodeReputationRecord
from hivemind.boinc.real_client.bridge import Verdict
from hivemind.boinc.rtx_boinc_deployer import RTXBoincHardwareProfile, detect_rtx_boinc_profile
from hivemind.core.auto_improvement import AutoImprovementExecutionResult, RetrofitReport


def test_node_reputation_record_and_real_client_verdict_validation() -> None:
    rec = NodeReputationRecord(steam_id64="76561198000000001", worker_id="rig-1")
    assert rec.reputation_score == 100.0

    with pytest.raises(ValueError, match="steam_id64 must be a non-empty string"):
        NodeReputationRecord(steam_id64="   ", worker_id="rig-1")
    with pytest.raises(ValueError, match="is_non_limited_steam_account must be a boolean"):
        NodeReputationRecord(steam_id64="76561198000000001", worker_id="rig-1", is_non_limited_steam_account=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="validated_workunits must be a non-negative integer"):
        NodeReputationRecord(steam_id64="76561198000000001", worker_id="rig-1", validated_workunits=-1)
    with pytest.raises(ValueError, match="contributed_tflops must be a finite non-negative number"):
        NodeReputationRecord(steam_id64="76561198000000001", worker_id="rig-1", contributed_tflops=float("nan"))

    verdict = Verdict(
        wu_id="wu-1",
        host="node-1",
        valid=True,
        reason="exact_match",
        same_class_exact=True,
        cross_class_max_ulp=None,
        grant={"cobblestones": 12.5},
    )
    assert verdict.credit == 12.5

    with pytest.raises(ValueError, match="wu_id must be a non-empty string"):
        Verdict(
            wu_id="",
            host="node-1",
            valid=True,
            reason="ok",
            same_class_exact=True,
            cross_class_max_ulp=None,
            grant={},
        )
    with pytest.raises(ValueError, match="same_class_exact must be None or a boolean"):
        Verdict(
            wu_id="wu-1",
            host="node-1",
            valid=True,
            reason="ok",
            same_class_exact="yes",  # type: ignore[arg-type]
            cross_class_max_ulp=None,
            grant={},
        )
    with pytest.raises(ValueError, match="cross_class_max_ulp must be None or a finite non-negative number"):
        Verdict(
            wu_id="wu-1",
            host="node-1",
            valid=True,
            reason="ok",
            same_class_exact=None,
            cross_class_max_ulp=-1.0,
            grant={},
        )


def test_rtx_boinc_hardware_profile_validation() -> None:
    prof = detect_rtx_boinc_profile(override_vram_total_mib=8192, override_vram_free_mib=7000, override_gpu_name="RTX 3070")
    assert prof.plan_class == "cuda_hivemind"

    with pytest.raises(ValueError, match="gpu_detected must be a boolean"):
        RTXBoincHardwareProfile(
            gpu_detected=1,  # type: ignore[arg-type]
            gpu_name="RTX 3070",
            vram_total_mib=8192,
            vram_free_mib=7000,
            driver_version="550.54",
            cuda_available=True,
            boinc_platform="x86_64-pc-linux-gnu",
            profile_name="standard",
            plan_class="cuda_hivemind",
            app_name="hivemind_lora_trainer",
            min_gpu_ram_mb=6144,
            wddm_ceiling_mib=6656,
            min_free_vram_mib=1536,
            qlora_rank=8,
            qlora_alpha=16,
            seq_len=512,
            micro_batch=1,
            inner_steps=100,
            segment_steps=10,
            vram_headroom_ok=True,
        )
    with pytest.raises(ValueError, match="qlora_rank must be a positive integer"):
        RTXBoincHardwareProfile(
            gpu_detected=True,
            gpu_name="RTX 3070",
            vram_total_mib=8192,
            vram_free_mib=7000,
            driver_version="550.54",
            cuda_available=True,
            boinc_platform="x86_64-pc-linux-gnu",
            profile_name="standard",
            plan_class="cuda_hivemind",
            app_name="hivemind_lora_trainer",
            min_gpu_ram_mb=6144,
            wddm_ceiling_mib=6656,
            min_free_vram_mib=1536,
            qlora_rank=0,
            qlora_alpha=16,
            seq_len=512,
            micro_batch=1,
            inner_steps=100,
            segment_steps=10,
            vram_headroom_ok=True,
        )


def test_auto_improvement_execution_result_and_retrofit_report_validation() -> None:
    res = AutoImprovementExecutionResult(
        wu_id="wu-1",
        domain="maths",
        worker_id="w-1",
        account_id="acct-1",
        steps_completed=50,
        loss_initial=2.5,
        loss_final=1.8,
        accuracy_base=0.50,
        accuracy_adapted=0.80,
        mcnemar_p_value=0.001,
        gate_passed=True,
        contributed_tflops=5.0,
        peak_vram_mb=4096.0,
        duration_seconds=60.0,
        delta_weights_hash="a" * 64,
    )
    assert res.accuracy_gain == 0.3

    with pytest.raises(ValueError, match="wu_id must be a non-empty string"):
        AutoImprovementExecutionResult(
            wu_id="  ",
            domain="maths",
            worker_id="w-1",
            account_id="acct-1",
            steps_completed=50,
            loss_initial=2.5,
            loss_final=1.8,
            accuracy_base=0.50,
            accuracy_adapted=0.80,
            mcnemar_p_value=0.001,
            gate_passed=True,
            contributed_tflops=5.0,
            peak_vram_mb=4096.0,
            duration_seconds=60.0,
            delta_weights_hash="a" * 64,
        )
    with pytest.raises(ValueError, match="accuracy_base must be in"):
        AutoImprovementExecutionResult(
            wu_id="wu-1",
            domain="maths",
            worker_id="w-1",
            account_id="acct-1",
            steps_completed=50,
            loss_initial=2.5,
            loss_final=1.8,
            accuracy_base=1.5,
            accuracy_adapted=0.80,
            mcnemar_p_value=0.001,
            gate_passed=True,
            contributed_tflops=5.0,
            peak_vram_mb=4096.0,
            duration_seconds=60.0,
            delta_weights_hash="a" * 64,
        )

    rep = RetrofitReport(
        total_workunits=2,
        newly_credited_count=1,
        already_credited_count=1,
        total_hmt_granted=15.0,
        total_tflops_attested=5.0,
        credited_accounts={"acct-1": 15.0},
        ledger_verified=True,
        timestamp_utc=1700000000.0,
        audit_log=["CREDITED wu-1"],
    )
    assert rep.ledger_verified is True

    with pytest.raises(ValueError, match="newly_credited_count \\+ already_credited_count cannot exceed total_workunits"):
        RetrofitReport(
            total_workunits=1,
            newly_credited_count=1,
            already_credited_count=1,
            total_hmt_granted=15.0,
            total_tflops_attested=5.0,
            credited_accounts={"acct-1": 15.0},
            ledger_verified=True,
            timestamp_utc=1700000000.0,
        )

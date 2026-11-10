# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and invariant tests for DiLoCo inner/outer loops, BTM upcycle reports, and Steamworks telemetry (IP-162)."""

from __future__ import annotations

import pytest
import torch

from hivemind.client.steamworks import SteamUserTelemetry
from hivemind.core.btm_upcycle import (
    LoopedBackboneReport,
    MoEUpcycleReport,
    SVDLoARankProjectionReport,
)
from hivemind.core.compression import CompressedDeltaPackage
from hivemind.core.diloco_inner import InnerLoopResult, MicroCheckpointRecord
from hivemind.core.diloco_outer import ByzantineFilterReport, OuterStepResult


def test_micro_checkpoint_record_and_inner_loop_result_validation() -> None:
    ckpt = MicroCheckpointRecord(
        step=10,
        lora_sha256="a" * 64,
        loss_value=1.5,
        state_snapshot={"lora_A": torch.ones(2, 2)},
    )
    assert ckpt.step == 10

    with pytest.raises(ValueError, match="step must be a non-negative integer"):
        MicroCheckpointRecord(step=-1, lora_sha256="a" * 64, loss_value=1.5, state_snapshot={})
    with pytest.raises(ValueError, match="lora_sha256 must be a non-empty string"):
        MicroCheckpointRecord(step=0, lora_sha256="   ", loss_value=1.5, state_snapshot={})
    with pytest.raises(ValueError, match="loss_value must be a finite number"):
        MicroCheckpointRecord(step=0, lora_sha256="a" * 64, loss_value=float("nan"), state_snapshot={})
    with pytest.raises(ValueError, match="state_snapshot must be a dict"):
        MicroCheckpointRecord(step=0, lora_sha256="a" * 64, loss_value=1.5, state_snapshot=[])  # type: ignore[arg-type]

    pkg = CompressedDeltaPackage(
        worker_id="w1",
        branch_name="Code",
        outer_step=1,
        inner_steps_completed=10,
        topk_ratio=0.1,
        tensors={},
        uncompressed_bytes=64,
        compressed_bytes=16,
    )
    res = InnerLoopResult(
        worker_id="w1",
        branch_name="Code",
        outer_step=1,
        steps_completed=10,
        initial_loss=2.0,
        final_loss=1.5,
        elapsed_seconds=0.5,
        theta_init={},
        theta_final={},
        raw_delta={},
        compressed_delta=pkg,
        micro_checkpoints={10: ckpt},
    )
    assert res.steps_completed == 10

    with pytest.raises(ValueError, match="worker_id must be a non-empty string"):
        InnerLoopResult(
            worker_id="",
            branch_name="Code",
            outer_step=1,
            steps_completed=10,
            initial_loss=2.0,
            final_loss=1.5,
            elapsed_seconds=0.5,
            theta_init={},
            theta_final={},
            raw_delta={},
            compressed_delta=pkg,
            micro_checkpoints={},
        )
    with pytest.raises(ValueError, match="elapsed_seconds must be non-negative"):
        InnerLoopResult(
            worker_id="w1",
            branch_name="Code",
            outer_step=1,
            steps_completed=10,
            initial_loss=2.0,
            final_loss=1.5,
            elapsed_seconds=-0.1,
            theta_init={},
            theta_final={},
            raw_delta={},
            compressed_delta=pkg,
            micro_checkpoints={},
        )
    with pytest.raises(ValueError, match="compressed_delta must be a CompressedDeltaPackage"):
        InnerLoopResult(
            worker_id="w1",
            branch_name="Code",
            outer_step=1,
            steps_completed=10,
            initial_loss=2.0,
            final_loss=1.5,
            elapsed_seconds=0.5,
            theta_init={},
            theta_final={},
            raw_delta={},
            compressed_delta={},  # type: ignore[arg-type]
            micro_checkpoints={},
        )


def test_byzantine_filter_report_and_outer_step_result_validation() -> None:
    rep = ByzantineFilterReport(
        total_submissions=4,
        max_byzantine_f=1,
        accepted_workers=["w1", "w2", "w3"],
        rejected_workers=["w4"],
        krum_scores={"w1": 0.1, "w2": 0.2, "w3": 0.15, "w4": 9.9},
        mean_cosine_similarities={"w1": 0.95, "w2": 0.94, "w3": 0.93, "w4": -0.5},
        median_cosine=0.94,
        cosine_threshold_lower=0.5,
        krum_guarantee=True,
    )
    assert rep.krum_guarantee is True

    with pytest.raises(ValueError, match="total_submissions must be a positive integer"):
        ByzantineFilterReport(
            total_submissions=0,
            max_byzantine_f=0,
            accepted_workers=[],
            rejected_workers=[],
            krum_scores={},
            mean_cosine_similarities={},
            median_cosine=0.0,
            cosine_threshold_lower=0.0,
        )
    with pytest.raises(ValueError, match="accepted_workers must be a list of non-empty strings"):
        ByzantineFilterReport(
            total_submissions=1,
            max_byzantine_f=0,
            accepted_workers=[""],
            rejected_workers=[],
            krum_scores={},
            mean_cosine_similarities={},
            median_cosine=0.0,
            cosine_threshold_lower=0.0,
        )

    outer = OuterStepResult(
        branch_name="Code",
        outer_step=1,
        theta_updated={"lora_A": torch.ones(2, 2)},
        velocity_updated={"lora_A": torch.zeros(2, 2)},
        aggregated_delta_norm=0.25,
        filter_report=rep,
    )
    assert outer.outer_step == 1

    with pytest.raises(ValueError, match="theta_updated must be a non-empty dict"):
        OuterStepResult(
            branch_name="Code",
            outer_step=1,
            theta_updated={},
            velocity_updated={"lora_A": torch.zeros(2, 2)},
            aggregated_delta_norm=0.25,
            filter_report=rep,
        )
    with pytest.raises(ValueError, match="aggregated_delta_norm must be a finite non-negative number"):
        OuterStepResult(
            branch_name="Code",
            outer_step=1,
            theta_updated={"lora_A": torch.ones(2, 2)},
            velocity_updated={"lora_A": torch.zeros(2, 2)},
            aggregated_delta_norm=-0.1,
            filter_report=rep,
        )
    with pytest.raises(ValueError, match="filter_report must be a ByzantineFilterReport"):
        OuterStepResult(
            branch_name="Code",
            outer_step=1,
            theta_updated={"lora_A": torch.ones(2, 2)},
            velocity_updated={"lora_A": torch.zeros(2, 2)},
            aggregated_delta_norm=0.25,
            filter_report=None,  # type: ignore[arg-type]
        )


def test_btm_upcycle_reports_and_steam_user_telemetry_validation() -> None:
    moe_rep = MoEUpcycleReport(
        num_experts=2,
        top_k=1,
        expert_domains=["Code", "Math"],
        attention_layers_merged=4,
        ffn_expert_modules_created=8,
        total_moe_parameters=1000,
        active_parameters_per_token=600,
    )
    assert moe_rep.num_experts == 2
    with pytest.raises(ValueError, match="top_k cannot exceed num_experts"):
        MoEUpcycleReport(
            num_experts=2,
            top_k=3,
            expert_domains=["Code", "Math"],
            attention_layers_merged=4,
            ffn_expert_modules_created=8,
            total_moe_parameters=1000,
            active_parameters_per_token=600,
        )
    with pytest.raises(ValueError, match="expert_domains must be a list of non-empty strings of length num_experts"):
        MoEUpcycleReport(
            num_experts=2,
            top_k=1,
            expert_domains=["Code"],
            attention_layers_merged=4,
            ffn_expert_modules_created=8,
            total_moe_parameters=1000,
            active_parameters_per_token=600,
        )

    svd_rep = SVDLoARankProjectionReport(
        source_rank_r=16,
        target_rank_r=8,
        num_projected_pairs=4,
        num_padded_pairs=0,
        num_identity_pairs=0,
        mean_spectral_energy_retained=0.95,
        min_spectral_energy_retained=0.90,
        max_relative_delta_error=0.05,
        source_params=2048,
        target_params=1024,
        source_lora_alpha=32.0,
        target_lora_alpha=16.0,
    )
    assert svd_rep.compression_ratio == 2.0
    with pytest.raises(ValueError, match="source_rank_r must be a non-negative integer"):
        SVDLoARankProjectionReport(
            source_rank_r=-1,
            target_rank_r=8,
            num_projected_pairs=4,
            num_padded_pairs=0,
            num_identity_pairs=0,
            mean_spectral_energy_retained=0.95,
            min_spectral_energy_retained=0.90,
            max_relative_delta_error=0.05,
            source_params=2048,
            target_params=1024,
            source_lora_alpha=32.0,
            target_lora_alpha=16.0,
        )
    with pytest.raises(ValueError, match="target_rank_r must be a positive integer"):
        SVDLoARankProjectionReport(
            source_rank_r=16,
            target_rank_r=0,
            num_projected_pairs=4,
            num_padded_pairs=0,
            num_identity_pairs=0,
            mean_spectral_energy_retained=0.95,
            min_spectral_energy_retained=0.90,
            max_relative_delta_error=0.05,
            source_params=2048,
            target_params=1024,
            source_lora_alpha=32.0,
            target_lora_alpha=16.0,
        )

    loop_rep = LoopedBackboneReport(
        source_depth=12,
        num_prelude_layers=2,
        num_core_layers=8,
        num_coda_layers=2,
        loop_steps=3,
        effective_depth=28,
        depth_multiplier=28.0 / 12.0,
        input_injection="add",
        per_step_lora_rank=8,
        source_weight_bytes=1000,
        core_weight_bytes=600,
        loop_adapter_bytes=120,
        looped_weight_bytes=1120,
        unrolled_weight_bytes=2320,
    )
    assert loop_rep.effective_depth == 28
    with pytest.raises(ValueError, match="depth_multiplier must be a finite positive number"):
        LoopedBackboneReport(
            source_depth=12,
            num_prelude_layers=2,
            num_core_layers=8,
            num_coda_layers=2,
            loop_steps=3,
            effective_depth=28,
            depth_multiplier=0.0,
            input_injection="add",
            per_step_lora_rank=8,
            source_weight_bytes=1000,
            core_weight_bytes=600,
            loop_adapter_bytes=120,
            looped_weight_bytes=1120,
            unrolled_weight_bytes=2320,
        )

    tel = SteamUserTelemetry(steam_id64="76561198000000001", persona_name="CyberBee")
    assert tel.verified_tflops == 0.0
    with pytest.raises(ValueError, match="steam_id64 must be a non-empty string"):
        SteamUserTelemetry(steam_id64="  ", persona_name="CyberBee")
    with pytest.raises(ValueError, match="verified_tflops must be a finite non-negative number"):
        SteamUserTelemetry(steam_id64="76561198000000001", persona_name="CyberBee", verified_tflops=-1.0)
    with pytest.raises(ValueError, match="workunits_completed must be a non-negative integer"):
        SteamUserTelemetry(steam_id64="76561198000000001", persona_name="CyberBee", workunits_completed=True)  # type: ignore[arg-type]

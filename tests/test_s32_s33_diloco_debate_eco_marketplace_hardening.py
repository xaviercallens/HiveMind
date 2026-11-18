# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-path unit tests for S33-01 DiLoCo, S33-02 Debate, S33-03 Eco, and S32-03 Marketplace."""

from __future__ import annotations

import dataclasses
import secrets
import pytest

from hivemind.swarm.s32_03_open_bee_marketplace_registry import (
    OpenBeeMarketplaceRegistry,
    sign_bee_manifest,
)
from hivemind.swarm.s33_01_decoupled_diloco_pacgad import (
    DecoupledDiLoCoStepResult,
    StragglerFreeDecoupledDiLoCoAggregator,
    WorkerPseudoGradient,
)
from hivemind.swarm.s33_02_multibee_amcp_debate import (
    BeeDebateTurn,
    MultiBeeAMCPDebateArena,
    MultiBeeDebateVerdict,
)
from hivemind.swarm.s33_03_eco_thermal_carbon_ledger import (
    EcoComputeTelemetrySample,
    EcoComputeValorizationReceipt,
    EcoThermalCarbonLedger,
)


class TestS3301DecoupledDiLoCoHardening:
    def test_step_result_and_aggregate_validation(self) -> None:
        agg = StragglerFreeDecoupledDiLoCoAggregator()
        res = agg.aggregate_and_step(
            [1.0, 2.0],
            [
                WorkerPseudoGradient("w1", 0, 10, (0.5, 0.0)),
                WorkerPseudoGradient("w2", 1, 10, (0.0, 0.5)),
            ],
        )
        assert res.to_dict()["effective_worker_count"] == 2.0

        with pytest.raises(ValueError, match="contributions must be a sequence of WorkerPseudoGradient"):
            agg.aggregate_and_step([1.0, 2.0], ["invalid"])  # type: ignore[list-item]

        with pytest.raises(ValueError, match="aggregated_vector must be a non-empty tuple of finite numbers"):
            DecoupledDiLoCoStepResult(
                aggregated_vector=(),
                radial_norm=1.0,
                naive_mean_norm=0.8,
                norm_preservation_ratio=1.25,
                effective_worker_count=2.0,
                max_staleness_steps=1,
                Updated_parameters=(1.0,),
            )
        with pytest.raises(ValueError, match="must have equal length"):
            DecoupledDiLoCoStepResult(
                aggregated_vector=(1.0,),
                radial_norm=1.0,
                naive_mean_norm=0.8,
                norm_preservation_ratio=1.25,
                effective_worker_count=2.0,
                max_staleness_steps=1,
                Updated_parameters=(1.0, 2.0),
            )
        with pytest.raises(ValueError, match="radial_norm must be a finite positive number"):
            DecoupledDiLoCoStepResult(
                aggregated_vector=(1.0,),
                radial_norm=0.0,
                naive_mean_norm=0.8,
                norm_preservation_ratio=1.25,
                effective_worker_count=2.0,
                max_staleness_steps=1,
                Updated_parameters=(1.0,),
            )
        with pytest.raises(ValueError, match="max_staleness_steps must be a non-negative integer"):
            DecoupledDiLoCoStepResult(
                aggregated_vector=(1.0,),
                radial_norm=1.0,
                naive_mean_norm=0.8,
                norm_preservation_ratio=1.25,
                effective_worker_count=2.0,
                max_staleness_steps=-1,
                Updated_parameters=(1.0,),
            )


class TestS3302MultiBeeDebateHardening:
    def test_verdict_validation(self) -> None:
        t1 = BeeDebateTurn("hivemind/bee-stars", "astro", True, 0.92, 1e-6, "ok")
        t2 = BeeDebateTurn("hivemind/bee-lean-prover", "prover", True, 0.98, 0.0, "ok")
        arena = MultiBeeAMCPDebateArena()
        verdict = arena.adjudicate_debate("subj-1", [t1, t2])
        assert verdict.to_dict()["accepted"] is True

        with pytest.raises(ValueError, match="subject_id must be a non-empty string"):
            MultiBeeDebateVerdict(
                subject_id=" ",
                accepted=True,
                vetoed_by=(),
                bayesian_posterior_confidence=0.95,
                max_residual_error=0.0,
                amcp_topic="hivemind.amcp.bee_coord.v1",
                turns=(t1, t2),
                transcript_sha256="a" * 64,
            )
        with pytest.raises(ValueError, match="vetoed_by must be a tuple of non-empty strings"):
            MultiBeeDebateVerdict(
                subject_id="subj-1",
                accepted=False,
                vetoed_by=("",),
                bayesian_posterior_confidence=0.95,
                max_residual_error=0.0,
                amcp_topic="hivemind.amcp.bee_coord.v1",
                turns=(t1, t2),
                transcript_sha256="a" * 64,
            )
        with pytest.raises(ValueError, match="bayesian_posterior_confidence must be a finite number in"):
            MultiBeeDebateVerdict(
                subject_id="subj-1",
                accepted=True,
                vetoed_by=(),
                bayesian_posterior_confidence=1.5,
                max_residual_error=0.0,
                amcp_topic="hivemind.amcp.bee_coord.v1",
                turns=(t1, t2),
                transcript_sha256="a" * 64,
            )
        with pytest.raises(ValueError, match="turns must be a tuple of at least 2 BeeDebateTurn"):
            MultiBeeDebateVerdict(
                subject_id="subj-1",
                accepted=True,
                vetoed_by=(),
                bayesian_posterior_confidence=0.95,
                max_residual_error=0.0,
                amcp_topic="hivemind.amcp.bee_coord.v1",
                turns=(t1,),
                transcript_sha256="a" * 64,
            )
        with pytest.raises(ValueError, match="transcript_sha256 must be a 64-character hex string"):
            MultiBeeDebateVerdict(
                subject_id="subj-1",
                accepted=True,
                vetoed_by=(),
                bayesian_posterior_confidence=0.95,
                max_residual_error=0.0,
                amcp_topic="hivemind.amcp.bee_coord.v1",
                turns=(t1, t2),
                transcript_sha256="z" * 64,
            )


class TestS3303EcoThermalCarbonLedgerHardening:
    def test_valorization_receipt_validation(self) -> None:
        ledger = EcoThermalCarbonLedger()
        receipt = ledger.evaluate_sample(
            EcoComputeTelemetrySample("node-1", 150.0, 65.0, 10.0, 50.0, 3600.0, 50000)
        )
        assert receipt.to_dict()["safe_thermal_envelope"] is True

        with pytest.raises(ValueError, match="node_id must be a non-empty string"):
            EcoComputeValorizationReceipt(
                node_id="",
                energy_consumed_wh=150.0,
                useful_domestic_heat_wh=138.0,
                thermal_valorization_efficiency=0.92,
                safe_thermal_envelope=True,
                datacenter_equivalent_wh=187.5,
                net_avoided_co2_grams=70.0,
                co2_grams_per_1k_tokens_hivemind=0.01,
                co2_grams_per_1k_tokens_datacenter=1.4,
            )
        with pytest.raises(ValueError, match="energy_consumed_wh must be a finite positive number"):
            EcoComputeValorizationReceipt(
                node_id="node-1",
                energy_consumed_wh=0.0,
                useful_domestic_heat_wh=0.0,
                thermal_valorization_efficiency=0.92,
                safe_thermal_envelope=True,
                datacenter_equivalent_wh=187.5,
                net_avoided_co2_grams=70.0,
                co2_grams_per_1k_tokens_hivemind=0.01,
                co2_grams_per_1k_tokens_datacenter=1.4,
            )
        with pytest.raises(ValueError, match="thermal_valorization_efficiency must be a finite number in"):
            EcoComputeValorizationReceipt(
                node_id="node-1",
                energy_consumed_wh=150.0,
                useful_domestic_heat_wh=138.0,
                thermal_valorization_efficiency=1.2,
                safe_thermal_envelope=True,
                datacenter_equivalent_wh=187.5,
                net_avoided_co2_grams=70.0,
                co2_grams_per_1k_tokens_hivemind=0.01,
                co2_grams_per_1k_tokens_datacenter=1.4,
            )


class TestS3203OpenBeeMarketplaceHardening:
    def test_verify_and_register_hex_and_field_guards(self) -> None:
        sec = secrets.token_bytes(32)
        reg = OpenBeeMarketplaceRegistry(sec)
        m = sign_bee_manifest(
            bee_id="hivemind/bee-fusion",
            owner_handle="xavier",
            domain="physics",
            base_backbone="gemma-4-9b",
            lora_rank=16,
            vram_required_mib=4096,
            weights_sha256="a" * 64,
            benchmark_score=0.91,
            signing_secret=sec,
        )
        assert reg.verify_and_register(m)["accepted"] is True

        m_bad_hex = dataclasses.replace(m, weights_sha256="z" * 64)
        res_hex = reg.verify_and_register(m_bad_hex)
        assert res_hex["accepted"] is False and res_hex["reason"] == "INVALID_WEIGHTS_SHA256"

        m_bad_rank = dataclasses.replace(m, lora_rank=0)
        res_rank = reg.verify_and_register(m_bad_rank)
        assert res_rank["accepted"] is False and res_rank["reason"] == "INVALID_LORA_RANK"

        m_bad_domain = dataclasses.replace(m, domain=" ")
        res_dom = reg.verify_and_register(m_bad_domain)
        assert res_dom["accepted"] is False and res_dom["reason"] == "INVALID_DOMAIN_OR_BACKBONE"

        m_bool_vram = dataclasses.replace(m, vram_required_mib=True)  # type: ignore[arg-type]
        res_vram = reg.verify_and_register(m_bool_vram)
        assert res_vram["accepted"] is False and res_vram["reason"] == "VRAM_CEILING_EXCEEDED_8GB"

# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-path unit tests for S32-01 Verlet, S32-02 Ratchet, Top-1% DCT, and Verified Update Gate."""

from __future__ import annotations

import pytest

from hivemind.swarm.s32_01_verlet_kepler442b_copilot import (
    SymplecticBenchmarkReport,
    SymplecticVelocityVerletIntegrator,
)
from hivemind.swarm.s32_02_lifelong_decentralized_ratchet import (
    AnchorEvalSample,
    DailyCorpusIngestionBatch,
    LifelongDecentralizedRatchet,
    LifelongRatchetVerdict,
)
from hivemind.swarm.top1pct_dct_ef21_compressor import (
    Top1PctDCTPacket,
    Top1PctQuantizedDCTEF21Compressor,
)
from hivemind.swarm.verified_update_gate import (
    ACCEPT,
    HeldOutTask,
    PASS,
    UpdatePolicy,
    UpdateReceipt,
    decide,
)


class TestS3201VerletKepler442bHardening:
    def test_benchmark_report_and_state_validation(self) -> None:
        integ = SymplecticVelocityVerletIntegrator()
        rep = integ.benchmark_kepler442b_orbit(steps=100, dt_yr=0.00015)
        assert rep.to_dict()["symplectic_gate_passed"] is True

        with pytest.raises(ValueError, match="state must be a Kepler442bOrbitalState"):
            integ.specific_energy("bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="state must be a Kepler442bOrbitalState"):
            integ.specific_angular_momentum("bad")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="target_body must be a non-empty string"):
            SymplecticBenchmarkReport(
                target_body="",
                steps=100,
                dt_yr=0.00015,
                total_simulated_yr=0.015,
                initial_energy=-29.5,
                final_energy_verlet=-29.5,
                max_relative_energy_drift_verlet=1e-9,
                max_relative_energy_drift_euler=1e-4,
                verlet_over_euler_improvement_ratio=1000.0,
                angular_momentum_relative_drift=1e-15,
                periapsis_au=0.39,
                apoapsis_au=0.42,
                symplectic_gate_passed=True,
            )
        with pytest.raises(ValueError, match="apoapsis_au must be >= periapsis_au"):
            SymplecticBenchmarkReport(
                target_body="KEPLER-HM-442b",
                steps=100,
                dt_yr=0.00015,
                total_simulated_yr=0.015,
                initial_energy=-29.5,
                final_energy_verlet=-29.5,
                max_relative_energy_drift_verlet=1e-9,
                max_relative_energy_drift_euler=1e-4,
                verlet_over_euler_improvement_ratio=1000.0,
                angular_momentum_relative_drift=1e-15,
                periapsis_au=0.45,
                apoapsis_au=0.39,
                symplectic_gate_passed=True,
            )


class TestS3202LifelongDecentralizedRatchetHardening:
    def test_verdict_validation(self) -> None:
        ratchet = LifelongDecentralizedRatchet()
        verdict = ratchet.evaluate_checkpoint(
            "ckpt-1",
            [AnchorEvalSample("a1", "astro", 1.1, 1.101)],
            DailyCorpusIngestionBatch("b1", "arxiv", 32, 1.4, 1.25),
        )
        assert verdict.to_dict()["promoted"] is True

        with pytest.raises(ValueError, match="checkpoint_id must be a non-empty string"):
            LifelongRatchetVerdict(
                checkpoint_id=" ",
                promoted=True,
                reason="ok",
                baseline_anchor_perplexity=3.0,
                candidate_anchor_perplexity=3.0,
                relative_anchor_perplexity_drift=0.0,
                max_allowed_anchor_drift=0.005,
                domain_nll_delta=-0.1,
                ewc_fisher_penalty=None,
                ratchet_certificate_sha256="a" * 64,
            )
        with pytest.raises(ValueError, match="baseline_anchor_perplexity must be a finite positive number"):
            LifelongRatchetVerdict(
                checkpoint_id="ckpt-1",
                promoted=True,
                reason="ok",
                baseline_anchor_perplexity=0.0,
                candidate_anchor_perplexity=3.0,
                relative_anchor_perplexity_drift=0.0,
                max_allowed_anchor_drift=0.005,
                domain_nll_delta=-0.1,
                ewc_fisher_penalty=None,
                ratchet_certificate_sha256="a" * 64,
            )
        with pytest.raises(ValueError, match="ewc_fisher_penalty must be None or a finite non-negative number"):
            LifelongRatchetVerdict(
                checkpoint_id="ckpt-1",
                promoted=True,
                reason="ok",
                baseline_anchor_perplexity=3.0,
                candidate_anchor_perplexity=3.0,
                relative_anchor_perplexity_drift=0.0,
                max_allowed_anchor_drift=0.005,
                domain_nll_delta=-0.1,
                ewc_fisher_penalty=-0.1,
                ratchet_certificate_sha256="a" * 64,
            )
        with pytest.raises(ValueError, match="ratchet_certificate_sha256 must be a 64-character hex string"):
            LifelongRatchetVerdict(
                checkpoint_id="ckpt-1",
                promoted=True,
                reason="ok",
                baseline_anchor_perplexity=3.0,
                candidate_anchor_perplexity=3.0,
                relative_anchor_perplexity_drift=0.0,
                max_allowed_anchor_drift=0.005,
                domain_nll_delta=-0.1,
                ewc_fisher_penalty=None,
                ratchet_certificate_sha256="z" * 64,
            )


class TestTop1PctDCTEF21CompressorHardening:
    def test_packet_validation(self) -> None:
        comp = Top1PctQuantizedDCTEF21Compressor()
        pkt, _ = comp.compress_and_reconstruct("div1", [0.1 * i for i in range(64)], step_index=1)
        assert pkt.to_dict()["division_id"] == "div1"

        with pytest.raises(ValueError, match="division_id must be a non-empty string"):
            Top1PctDCTPacket(
                division_id="",
                step_index=0,
                total_elements=64,
                retained_k=1,
                quantization_bits=4,
                indices=(0,),
                q_levels=(15,),
                scale=1.0,
                zero_point=0.0,
                raw_bytes_fp32=256,
                wire_bytes=18,
                compression_ratio=14.2,
                ef21_contraction_factor=0.1,
                conservation_max_abs_err=0.0,
                packet_sha256="a" * 64,
            )
        with pytest.raises(ValueError, match="indices must be a tuple of 1 non-negative integers"):
            Top1PctDCTPacket(
                division_id="div1",
                step_index=0,
                total_elements=64,
                retained_k=1,
                quantization_bits=4,
                indices=(0, 1),
                q_levels=(15,),
                scale=1.0,
                zero_point=0.0,
                raw_bytes_fp32=256,
                wire_bytes=18,
                compression_ratio=14.2,
                ef21_contraction_factor=0.1,
                conservation_max_abs_err=0.0,
                packet_sha256="a" * 64,
            )
        with pytest.raises(ValueError, match="packet_sha256 must be a 64-character hex string"):
            Top1PctDCTPacket(
                division_id="div1",
                step_index=0,
                total_elements=64,
                retained_k=1,
                quantization_bits=4,
                indices=(0,),
                q_levels=(15,),
                scale=1.0,
                zero_point=0.0,
                raw_bytes_fp32=256,
                wire_bytes=18,
                compression_ratio=14.2,
                ef21_contraction_factor=0.1,
                conservation_max_abs_err=0.0,
                packet_sha256="g" * 64,
            )


class TestVerifiedUpdateGateHardening:
    def test_update_receipt_and_held_out_task_validation(self) -> None:
        base = {f"t{i}": PASS for i in range(5)}
        cand = {f"t{i}": PASS for i in range(5)}
        rec = decide(base, cand, UpdatePolicy(min_n=5))
        assert rec.accepted is True and len(rec.receipt_sha256) == 64

        with pytest.raises(ValueError, match="decision must be one of"):
            UpdateReceipt(
                decision="BAD",
                reasons=["ok"],
                n=5,
                b=0,
                c=0,
                both_pass=5,
                both_fail=0,
                baseline_pass=5,
                candidate_pass=5,
                unverified_baseline=0,
                unverified_candidate=0,
                unverified_fraction=0.0,
                p_two_sided=1.0,
                p_regression=1.0,
                p_improvement=1.0,
                policy={},
                receipt_sha256="a" * 64,
            )
        with pytest.raises(ValueError, match="receipt_sha256 must be empty or a 64-character hex string"):
            UpdateReceipt(
                decision=ACCEPT,
                reasons=["ok"],
                n=5,
                b=0,
                c=0,
                both_pass=5,
                both_fail=0,
                baseline_pass=5,
                candidate_pass=5,
                unverified_baseline=0,
                unverified_candidate=0,
                unverified_fraction=0.0,
                p_two_sided=1.0,
                p_regression=1.0,
                p_improvement=1.0,
                policy={},
                receipt_sha256="bad",
            )
        with pytest.raises(ValueError, match="tests must be a non-empty sequence of strings"):
            HeldOutTask(key="t1", prompt="p", tests=[""])

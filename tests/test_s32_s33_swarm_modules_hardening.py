# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-control hardening suite for HM-S32-01..03 and HM-S33-01..03 swarm modules."""

from __future__ import annotations

import pytest

from hivemind.swarm.s32_01_verlet_kepler442b_copilot import (
    AntigravityHiveMindCoPilot,
    FallbackLocalOpenAIAgentConfig,
    Kepler442bOrbitalState,
    SymplecticVelocityVerletIntegrator,
)
from hivemind.swarm.s32_02_lifelong_decentralized_ratchet import (
    AnchorEvalSample,
    DailyCorpusIngestionBatch,
    LifelongDecentralizedRatchet,
)
from hivemind.swarm.s32_03_open_bee_marketplace_registry import (
    OpenBeeMarketplaceRegistry,
    sign_bee_manifest,
)
from hivemind.swarm.s33_01_decoupled_diloco_pacgad import (
    StragglerFreeDecoupledDiLoCoAggregator,
    WorkerPseudoGradient,
)
from hivemind.swarm.s33_02_multibee_amcp_debate import (
    BeeDebateTurn,
    MultiBeeAMCPDebateArena,
)
from hivemind.swarm.s33_03_eco_thermal_carbon_ledger import (
    EcoComputeTelemetrySample,
    EcoThermalCarbonLedger,
)

TEST_SECRET = b"open-bee-marketplace-secret-2026"
TEST_API_KEY = "hm-" + "t" * 40


def test_s32_01_verlet_and_copilot_validation() -> None:
    with pytest.raises(ValueError, match="model"):
        FallbackLocalOpenAIAgentConfig(model="", base_url="https://example.org")
    with pytest.raises(ValueError, match="base_url"):
        FallbackLocalOpenAIAgentConfig(model="hivemind/bee-stars", base_url="   ")
    with pytest.raises(ValueError, match="is_lightweight"):
        FallbackLocalOpenAIAgentConfig(model="hivemind/bee-stars", base_url="https://example.org", is_lightweight="yes")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="x_au"):
        Kepler442bOrbitalState(x_au=float("nan"), y_au=1.0, vx_au_yr=0.0, vy_au_yr=1.0)
    with pytest.raises(ValueError, match="orbital radius"):
        Kepler442bOrbitalState(x_au=0.0, y_au=0.0, vx_au_yr=0.0, vy_au_yr=1.0)

    for bad_num in (True, False, float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(ValueError, match="star_mass_solar"):
            SymplecticVelocityVerletIntegrator(star_mass_solar=bad_num)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="semi_major_axis_au"):
            SymplecticVelocityVerletIntegrator(semi_major_axis_au=bad_num)  # type: ignore[arg-type]

    for bad_ecc in (True, float("nan"), -0.1, 1.0):
        with pytest.raises(ValueError, match="eccentricity"):
            SymplecticVelocityVerletIntegrator(eccentricity=bad_ecc)  # type: ignore[arg-type]

    integ = SymplecticVelocityVerletIntegrator()
    s0 = integ.periapsis_initial_state()
    with pytest.raises(ValueError, match="state"):
        integ.step_velocity_verlet("not-a-state", 0.001)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dt_yr"):
        integ.step_velocity_verlet(s0, float("nan"))
    with pytest.raises(ValueError, match="dt_yr"):
        integ.step_forward_euler(s0, 0.0)
    with pytest.raises(ValueError, match="steps"):
        integ.benchmark_kepler442b_orbit(steps=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="base_url"):
        AntigravityHiveMindCoPilot(api_key=TEST_API_KEY, base_url="")
    with pytest.raises(ValueError, match="model"):
        AntigravityHiveMindCoPilot(api_key=TEST_API_KEY, model="")


def test_s32_02_lifelong_ratchet_validation() -> None:
    with pytest.raises(ValueError, match="sample_id"):
        AnchorEvalSample("", "astro", 1.0, 1.0)
    with pytest.raises(ValueError, match="domain"):
        AnchorEvalSample("s1", "  ", 1.0, 1.0)
    with pytest.raises(ValueError, match="baseline_nll"):
        AnchorEvalSample("s1", "astro", True, 1.0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="batch_id"):
        DailyCorpusIngestionBatch("", "arxiv", 10, 1.2, 1.1)
    with pytest.raises(ValueError, match="source"):
        DailyCorpusIngestionBatch("b1", "", 10, 1.2, 1.1)
    with pytest.raises(ValueError, match="document_count"):
        DailyCorpusIngestionBatch("b1", "arxiv", 0, 1.2, 1.1)
    with pytest.raises(ValueError, match="document_count"):
        DailyCorpusIngestionBatch("b1", "arxiv", True, 1.2, 1.1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_anchor_drift_ratio"):
        LifelongDecentralizedRatchet(max_anchor_drift_ratio=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ewc_lambda"):
        LifelongDecentralizedRatchet(ewc_lambda=True)  # type: ignore[arg-type]

    ratchet = LifelongDecentralizedRatchet()
    batch = DailyCorpusIngestionBatch("b1", "arxiv", 10, 1.2, 1.1)
    samples = [AnchorEvalSample("s1", "astro", 1.0, 1.0)]
    with pytest.raises(ValueError, match="checkpoint_id"):
        ratchet.evaluate_checkpoint("", samples, batch)
    with pytest.raises(ValueError, match="daily_batch"):
        ratchet.evaluate_checkpoint("ckpt-1", samples, "not-a-batch")  # type: ignore[arg-type]


def test_s32_03_open_bee_marketplace_validation() -> None:
    with pytest.raises(ValueError, match="signing_secret"):
        OpenBeeMarketplaceRegistry("not-bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_benchmark_score"):
        OpenBeeMarketplaceRegistry(TEST_SECRET, min_benchmark_score=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_benchmark_score"):
        OpenBeeMarketplaceRegistry(TEST_SECRET, min_benchmark_score=1.5)

    reg = OpenBeeMarketplaceRegistry(TEST_SECRET)
    with pytest.raises(ValueError, match="manifest"):
        reg.verify_and_register("not-a-manifest")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="domain"):
        reg.match_best_bee_for_domain("")

    with pytest.raises(ValueError, match="lora_rank"):
        sign_bee_manifest(
            bee_id="hivemind/bee-test",
            owner_handle="user",
            domain="math",
            base_backbone="qwen",
            lora_rank=0,
            vram_required_mib=4096,
            weights_sha256="a" * 64,
            benchmark_score=0.9,
            signing_secret=TEST_SECRET,
        )
    with pytest.raises(ValueError, match="benchmark_score"):
        sign_bee_manifest(
            bee_id="hivemind/bee-test",
            owner_handle="user",
            domain="math",
            base_backbone="qwen",
            lora_rank=16,
            vram_required_mib=4096,
            weights_sha256="a" * 64,
            benchmark_score=float("nan"),
            signing_secret=TEST_SECRET,
        )


def test_s33_01_02_03_swarm_modules_validation() -> None:
    with pytest.raises(ValueError, match="worker_id"):
        WorkerPseudoGradient("", 0, 10, (1.0, 2.0))
    with pytest.raises(ValueError, match="staleness_steps"):
        WorkerPseudoGradient("w1", True, 10, (1.0, 2.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inner_steps"):
        WorkerPseudoGradient("w1", 0, 0, (1.0, 2.0))
    with pytest.raises(ValueError, match="pseudo_gradient"):
        WorkerPseudoGradient("w1", 0, 10, ())

    with pytest.raises(ValueError, match="staleness_alpha"):
        StragglerFreeDecoupledDiLoCoAggregator(staleness_alpha=float("nan"))
    with pytest.raises(ValueError, match="outer_lr"):
        StragglerFreeDecoupledDiLoCoAggregator(outer_lr=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_staleness_steps"):
        StragglerFreeDecoupledDiLoCoAggregator(max_staleness_steps=True)  # type: ignore[arg-type]

    agg = StragglerFreeDecoupledDiLoCoAggregator()
    with pytest.raises(ValueError, match="staleness_steps"):
        agg.staleness_weight(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="current_params"):
        agg.aggregate_and_step([float("nan"), 1.0], [WorkerPseudoGradient("w1", 0, 10, (1.0, 2.0))])

    with pytest.raises(ValueError, match="bee_id"):
        BeeDebateTurn("", "solver", True, 0.9, 1e-6, "ok")
    with pytest.raises(ValueError, match="passed_invariants"):
        BeeDebateTurn("hivemind/bee-stars", "solver", "yes", 0.9, 1e-6, "ok")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="residual_ceiling"):
        MultiBeeAMCPDebateArena(residual_ceiling=-1.0)
    with pytest.raises(ValueError, match="min_posterior"):
        MultiBeeAMCPDebateArena(min_posterior=0.0)

    arena = MultiBeeAMCPDebateArena()
    t1 = BeeDebateTurn("hivemind/bee-stars", "hypothesis", True, 0.95, 1e-6, "ok")
    t2 = BeeDebateTurn("hivemind/bee-lean-prover", "prover", True, 0.95, 0.0, "ok")
    with pytest.raises(ValueError, match="subject_id"):
        arena.adjudicate_debate("", [t1, t2])
    with pytest.raises(ValueError, match="BeeDebateTurn"):
        arena.adjudicate_debate("subj", [t1, "not-a-turn"])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="node_id"):
        EcoComputeTelemetrySample("", 150.0, 65.0, 10.0, 50.0, 3600.0, 1000)
    with pytest.raises(ValueError, match="tokens_generated"):
        EcoComputeTelemetrySample("node-1", 150.0, 65.0, 10.0, 50.0, 3600.0, True)  # type: ignore[arg-type]

    ledger = EcoThermalCarbonLedger()
    with pytest.raises(ValueError, match="sample"):
        ledger.evaluate_sample("not-a-sample")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gpu_power_watts"):
        ledger.evaluate_sample(EcoComputeTelemetrySample("node-1", True, 65.0, 10.0, 50.0, 3600.0, 1000))  # type: ignore[arg-type]

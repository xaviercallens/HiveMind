# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-path unit tests for S36-03 GeoMesh, S35-05 XDEV, S34-04 Supervisor, and S34-01 Speculative Decoding."""

from __future__ import annotations

import pytest

from hivemind.swarm.s34_01_speculative_swarm_decoding import (
    SpeculativeDecodingStepResult,
    sample_residual_token_distribution,
)
from hivemind.swarm.s34_04_self_evolving_roadmap_supervisor import (
    RoadmapPRVerificationEntry,
    SelfEvolvingRoadmapSupervisor,
)
from hivemind.swarm.s35_05_unified_xdev_ecosystem_bridge import (
    PortfolioProjectNode,
    UnifiedEcosystemExecutionReceipt,
    run_unified_4layer_ecosystem_pipeline,
)
from hivemind.swarm.s36_03_geocryptographic_neural_mesh import (
    GeocryptographicNeuralMeshRouter,
    GeoMeshRouteDecision,
    VivaldiCoord3D,
)


class TestS3603GeoMeshHardening:
    def test_geomesh_route_decision_and_evaluate_validation(self) -> None:
        dec = GeoMeshRouteDecision(
            selected_node_id="bee-eu-01",
            region="eu-west",
            expert_domain="astro",
            predicted_rtt_ms=12.5,
            central_gateway_rtt_ms=180.0,
            cosine_similarity=0.95,
            adapter_verified=True,
        )
        assert dec.to_dict()["adapter_verified"] is True

        with pytest.raises(ValueError, match="selected_node_id must be a non-empty string"):
            GeoMeshRouteDecision(
                selected_node_id=" ",
                region="eu-west",
                expert_domain="astro",
                predicted_rtt_ms=12.5,
                central_gateway_rtt_ms=180.0,
                cosine_similarity=0.95,
                adapter_verified=True,
            )
        with pytest.raises(ValueError, match="predicted_rtt_ms must be positive"):
            GeoMeshRouteDecision(
                selected_node_id="bee-eu-01",
                region="eu-west",
                expert_domain="astro",
                predicted_rtt_ms=0.0,
                central_gateway_rtt_ms=180.0,
                cosine_similarity=0.95,
                adapter_verified=True,
            )
        with pytest.raises(ValueError, match="cosine_similarity must be in"):
            GeoMeshRouteDecision(
                selected_node_id="bee-eu-01",
                region="eu-west",
                expert_domain="astro",
                predicted_rtt_ms=12.5,
                central_gateway_rtt_ms=180.0,
                cosine_similarity=1.5,
                adapter_verified=True,
            )
        with pytest.raises(ValueError, match="adapter_verified must be a boolean"):
            GeoMeshRouteDecision(
                selected_node_id="bee-eu-01",
                region="eu-west",
                expert_domain="astro",
                predicted_rtt_ms=12.5,
                central_gateway_rtt_ms=180.0,
                cosine_similarity=0.95,
                adapter_verified=1,  # type: ignore[arg-type]
            )

        router = GeocryptographicNeuralMeshRouter()
        with pytest.raises(ValueError, match="central_gateway_coord must be a VivaldiCoord3D or None"):
            router.evaluate_multi_region_p95_reduction(central_gateway_coord="bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="client_coord must be a VivaldiCoord3D or None"):
            router.evaluate_multi_region_p95_reduction(client_coord="bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="regional_coords must be a non-empty sequence"):
            router.evaluate_multi_region_p95_reduction(regional_coords=[])
        with pytest.raises(ValueError, match="regional_coords must contain"):
            router.evaluate_multi_region_p95_reduction(
                regional_coords=[("", VivaldiCoord3D(1.0, 2.0, 3.0))]
            )


class TestS3505XdevEcosystemHardening:
    def test_portfolio_node_and_receipt_validation(self) -> None:
        node = PortfolioProjectNode(
            name="HiveMind",
            origin_type="original_core_ip",
            layer="L2_Compute_Lattice",
            local_xdev_dir="/tmp/hm",
            github_repo="xaviercallens/HiveMind",
            role_in_unified_stack="Swarm",
            locally_present=True,
        )
        assert node.name == "HiveMind"

        with pytest.raises(ValueError, match="name must be a non-empty string"):
            PortfolioProjectNode(
                name="",
                origin_type="original_core_ip",
                layer="L2_Compute_Lattice",
                local_xdev_dir="",
                github_repo="xaviercallens/HiveMind",
                role_in_unified_stack="Swarm",
                locally_present=False,
            )
        with pytest.raises(ValueError, match="origin_type must be"):
            PortfolioProjectNode(
                name="HiveMind",
                origin_type="unknown",
                layer="L2_Compute_Lattice",
                local_xdev_dir="",
                github_repo="xaviercallens/HiveMind",
                role_in_unified_stack="Swarm",
                locally_present=False,
            )
        with pytest.raises(ValueError, match="Unsupported layer"):
            PortfolioProjectNode(
                name="HiveMind",
                origin_type="original_core_ip",
                layer="L99_Invalid",
                local_xdev_dir="",
                github_repo="xaviercallens/HiveMind",
                role_in_unified_stack="Swarm",
                locally_present=False,
            )
        with pytest.raises(ValueError, match="locally_present must be a boolean"):
            PortfolioProjectNode(
                name="HiveMind",
                origin_type="original_core_ip",
                layer="L2_Compute_Lattice",
                local_xdev_dir="",
                github_repo="xaviercallens/HiveMind",
                role_in_unified_stack="Swarm",
                locally_present=1,  # type: ignore[arg-type]
            )

        receipt = run_unified_4layer_ecosystem_pipeline()
        assert receipt.all_layers_passed is True

        with pytest.raises(ValueError, match="original_core_count must be a non-negative integer"):
            UnifiedEcosystemExecutionReceipt(
                original_core_count=-1,
                fork_bridge_count=8,
                local_xdev_verified_count=1,
                layer1_kv_compression_ratio=6.4,
                layer1_a2_rope_mse_gain_pct=12.0,
                layer2_peak_citizen_vram_mib=1200.0,
                layer3_gwaya_a3_verified=True,
                layer3_final_anse_energy=0.0,
                layer4_lean_kernel_clean=True,
                layer4_sundials_tier_x_accepted=True,
                all_layers_passed=True,
                merkle_root_sha256="a" * 64,
            )
        with pytest.raises(ValueError, match="merkle_root_sha256 must be a 64-character hex string"):
            UnifiedEcosystemExecutionReceipt(
                original_core_count=11,
                fork_bridge_count=8,
                local_xdev_verified_count=1,
                layer1_kv_compression_ratio=6.4,
                layer1_a2_rope_mse_gain_pct=12.0,
                layer2_peak_citizen_vram_mib=1200.0,
                layer3_gwaya_a3_verified=True,
                layer3_final_anse_energy=0.0,
                layer4_lean_kernel_clean=True,
                layer4_sundials_tier_x_accepted=True,
                all_layers_passed=True,
                merkle_root_sha256="z" * 64,
            )


class TestS3404RoadmapSupervisorHardening:
    def test_entry_hex_and_control_char_validation(self) -> None:
        entry = RoadmapPRVerificationEntry(
            order=1,
            ticket_id="HM-S32-01",
            title="Velocity-Verlet",
            invariant_passed=True,
            key_metric_name="drift",
            key_metric_value=1e-7,
            openrouter_request_id="req-1",
            entry_hash_sha256="0123456789abcdef" * 4,
        )
        assert entry.to_dict()["order"] == 1

        with pytest.raises(ValueError, match="entry_hash_sha256 must be a 64-character hex string"):
            RoadmapPRVerificationEntry(
                order=1,
                ticket_id="HM-S32-01",
                title="Velocity-Verlet",
                invariant_passed=True,
                key_metric_name="drift",
                key_metric_value=1e-7,
                openrouter_request_id="req-1",
                entry_hash_sha256="g" * 64,
            )
        with pytest.raises(ValueError, match="openrouter_request_id must be a string"):
            RoadmapPRVerificationEntry(
                order=1,
                ticket_id="HM-S32-01",
                title="Velocity-Verlet",
                invariant_passed=True,
                key_metric_name="drift",
                key_metric_value=1e-7,
                openrouter_request_id="req\x001",
                entry_hash_sha256="a" * 64,
            )

        sup = SelfEvolvingRoadmapSupervisor()
        with pytest.raises(ValueError, match="openrouter_request_ids must be None or a sequence of strings"):
            sup.verify_all_10_roadmap_modules(openrouter_request_ids=["req\n1"])


class TestS3401SpeculativeSwarmDecodingHardening:
    def test_step_result_and_residual_distribution_validation(self) -> None:
        res = SpeculativeDecodingStepResult(
            draft_k=3,
            accepted_count=2,
            bonus_token_id=42,
            emitted_tokens=(10, 20, 42),
            acceptance_rate=2.0 / 3.0,
            effective_speedup_factor=2.1,
        )
        assert res.to_dict()["draft_k"] == 3

        with pytest.raises(ValueError, match="draft_k must be a positive integer"):
            SpeculativeDecodingStepResult(
                draft_k=0,
                accepted_count=0,
                bonus_token_id=42,
                emitted_tokens=(42,),
                acceptance_rate=0.0,
                effective_speedup_factor=1.0,
            )
        with pytest.raises(ValueError, match="accepted_count must be an integer in"):
            SpeculativeDecodingStepResult(
                draft_k=2,
                accepted_count=3,
                bonus_token_id=42,
                emitted_tokens=(1, 2, 42),
                acceptance_rate=1.0,
                effective_speedup_factor=2.0,
            )
        with pytest.raises(ValueError, match="emitted_tokens must be a non-empty tuple of non-negative integers"):
            SpeculativeDecodingStepResult(
                draft_k=2,
                accepted_count=1,
                bonus_token_id=42,
                emitted_tokens=(-1,),
                acceptance_rate=0.5,
                effective_speedup_factor=1.5,
            )
        with pytest.raises(ValueError, match="effective_speedup_factor must be a finite positive number"):
            SpeculativeDecodingStepResult(
                draft_k=2,
                accepted_count=1,
                bonus_token_id=42,
                emitted_tokens=(1, 42),
                acceptance_rate=0.5,
                effective_speedup_factor=0.0,
            )

        with pytest.raises(ValueError, match="target_probs keys must be non-negative integers"):
            sample_residual_token_distribution(
                target_probs={-1: 0.5, 2: 0.5},
                draft_probs={1: 0.5, 2: 0.5},
            )
        with pytest.raises(ValueError, match="target_probs values must be finite non-negative numbers"):
            sample_residual_token_distribution(
                target_probs={1: -0.1, 2: 1.1},
                draft_probs={1: 0.5, 2: 0.5},
            )
        with pytest.raises(ValueError, match="draft_probs keys must be non-negative integers"):
            sample_residual_token_distribution(
                target_probs={1: 0.5, 2: 0.5},
                draft_probs={True: 0.5},  # type: ignore[dict-item]
            )

# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-path unit tests for SwaT, Strategy Optimizer, QUIC Link, and PR-First Orchestrator."""

from __future__ import annotations

import numpy as np
import pytest

from hivemind.swarm.pr_first_openrouter_orchestrator import (
    PRVerificationReceipt,
    RoadmapPRSpec,
)
from hivemind.swarm.quic_link import (
    LinkRun,
    QuicSendReport,
)
from hivemind.swarm.strategy_optimizer import (
    ExpertSpectralProfile,
)
from hivemind.swarm.swat_telepathy_bridge import (
    S34_BEE_MODEL_SPECS,
    SwaTTelepathyBridge,
    UniversalHiveLatentHub,
    compute_knowledge_seeding_loss,
)


class TestSwaTTelepathyBridgeHardening:
    def test_hub_bridge_and_ks_loss_validation(self) -> None:
        hub = UniversalHiveLatentHub()
        assert hub.num_families >= 5

        with pytest.raises(ValueError, match="n_blocks must be a positive integer"):
            UniversalHiveLatentHub(n_blocks=0)
        with pytest.raises(ValueError, match="registered_models must map"):
            UniversalHiveLatentHub(registered_models={"": S34_BEE_MODEL_SPECS["Qwen/Qwen3.5-0.8B"]})
        with pytest.raises(ValueError, match="spec must be a ModelKVSpec"):
            hub.register_model("bad")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="hub must be a UniversalHiveLatentHub"):
            SwaTTelepathyBridge(hub="bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="n_blocks must be a positive integer"):
            SwaTTelepathyBridge(n_blocks=0)

        s_spec = S34_BEE_MODEL_SPECS["google/gemma-3-4b-it"]
        r_spec = S34_BEE_MODEL_SPECS["Qwen/Qwen3.5-4B"]
        with pytest.raises(ValueError, match="sender_spec and receiver_spec must be ModelKVSpec"):
            SwaTTelepathyBridge.should_fallback_to_native_vlm("bad", r_spec)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="ocr_text_density must be a finite number in"):
            SwaTTelepathyBridge.should_fallback_to_native_vlm(s_spec, r_spec, ocr_text_density=1.5)
        with pytest.raises(ValueError, match="dense_ocr_threshold must be a finite number in"):
            SwaTTelepathyBridge.should_fallback_to_native_vlm(s_spec, r_spec, dense_ocr_threshold=0.0)

        s_lp = np.zeros((4, 8))
        t_lp = np.zeros((4, 8))
        with pytest.raises(ValueError, match="semantic_equivalence must be a finite number in"):
            compute_knowledge_seeding_loss(s_lp, t_lp, "abcd", "abcd", semantic_equivalence=1.2)
        with pytest.raises(ValueError, match="lambda_gkd must be a finite non-negative number"):
            compute_knowledge_seeding_loss(s_lp, t_lp, "abcd", "abcd", semantic_equivalence=0.9, lambda_gkd=-1.0)
        with pytest.raises(ValueError, match="alpha_sem \\+ alpha_abs must be positive"):
            compute_knowledge_seeding_loss(
                s_lp, t_lp, "abcd", "abcd", semantic_equivalence=0.9, alpha_sem=0.0, alpha_abs=0.0
            )


class TestStrategyOptimizerHardening:
    def test_expert_spectral_profile_validation(self) -> None:
        prof = ExpertSpectralProfile(
            domain_name="Rust_Systems_Security",
            source_rank_r=32,
            allocated_rank_r=24,
            spectral_shannon_entropy=0.85,
            fisher_trace_score=14.2,
            uniform_r16_energy_retained=0.78,
            dynamic_rank_energy_retained=0.92,
            energy_gain_vs_uniform=0.14,
        )
        assert prof.allocated_rank_r == 24

        with pytest.raises(ValueError, match="domain_name must be a non-empty string"):
            ExpertSpectralProfile(
                domain_name=" ",
                source_rank_r=32,
                allocated_rank_r=24,
                spectral_shannon_entropy=0.85,
                fisher_trace_score=14.2,
                uniform_r16_energy_retained=0.78,
                dynamic_rank_energy_retained=0.92,
                energy_gain_vs_uniform=0.14,
            )
        with pytest.raises(ValueError, match="source_rank_r must be a positive integer"):
            ExpertSpectralProfile(
                domain_name="Rust",
                source_rank_r=0,
                allocated_rank_r=24,
                spectral_shannon_entropy=0.85,
                fisher_trace_score=14.2,
                uniform_r16_energy_retained=0.78,
                dynamic_rank_energy_retained=0.92,
                energy_gain_vs_uniform=0.14,
            )
        with pytest.raises(ValueError, match="spectral_shannon_entropy must be a finite number in"):
            ExpertSpectralProfile(
                domain_name="Rust",
                source_rank_r=32,
                allocated_rank_r=24,
                spectral_shannon_entropy=1.2,
                fisher_trace_score=14.2,
                uniform_r16_energy_retained=0.78,
                dynamic_rank_energy_retained=0.92,
                energy_gain_vs_uniform=0.14,
            )
        with pytest.raises(ValueError, match="fisher_trace_score must be a finite non-negative number"):
            ExpertSpectralProfile(
                domain_name="Rust",
                source_rank_r=32,
                allocated_rank_r=24,
                spectral_shannon_entropy=0.85,
                fisher_trace_score=-1.0,
                uniform_r16_energy_retained=0.78,
                dynamic_rank_energy_retained=0.92,
                energy_gain_vs_uniform=0.14,
            )


class TestQuicLinkHardening:
    def test_quic_send_report_and_link_run_validation(self) -> None:
        rep = QuicSendReport(
            delivered=3,
            elapsed_s=0.012,
            session_resumed=True,
            early_data_accepted=True,
            new_ticket=None,
            handshake_latency_s=0.004,
        )
        assert rep.delivered == 3

        with pytest.raises(ValueError, match="delivered must be a non-negative integer"):
            QuicSendReport(
                delivered=-1,
                elapsed_s=0.012,
                session_resumed=True,
                early_data_accepted=True,
                new_ticket=None,
            )
        with pytest.raises(ValueError, match="handshake_latency_s must be None or a finite non-negative number"):
            QuicSendReport(
                delivered=1,
                elapsed_s=0.012,
                session_resumed=True,
                early_data_accepted=True,
                new_ticket=None,
                handshake_latency_s=-0.1,
            )

        run = LinkRun(
            protocol="quic_0rtt",
            frames_sent=4,
            frames_delivered=4,
            elapsed_s=0.02,
            session_resumed=True,
            early_data_accepted=True,
        )
        assert run.frames_per_sec > 0

        with pytest.raises(ValueError, match="protocol must be"):
            LinkRun(
                protocol="tcp_raw",
                frames_sent=4,
                frames_delivered=4,
                elapsed_s=0.02,
            )
        with pytest.raises(ValueError, match="frames_sent must be a non-negative integer"):
            LinkRun(
                protocol="quic",
                frames_sent=-1,
                frames_delivered=0,
                elapsed_s=0.02,
            )


class TestPRFirstOpenRouterOrchestratorHardening:
    def test_control_chars_and_backslash_rejection(self) -> None:
        with pytest.raises(ValueError, match="must not contain control characters or backslashes"):
            RoadmapPRSpec(
                pr_order=1,
                roadmap_id="HM-S29-01",
                title="Test",
                module_path="src\\hivemind\\mod.py",
                test_path="tests/test_mod.py",
            )
        with pytest.raises(ValueError, match="roadmap_id must be a non-empty string"):
            RoadmapPRSpec(
                pr_order=1,
                roadmap_id="HM-S29\x0001",
                title="Test",
                module_path="src/hivemind/mod.py",
                test_path="tests/test_mod.py",
            )
        with pytest.raises(ValueError, match="openrouter_completion_id must be None or a non-empty string"):
            PRVerificationReceipt(
                roadmap_id="HM-S29-01",
                module_path="src/hivemind/mod.py",
                test_path="tests/test_mod.py",
                openrouter_completion_id="comp\n1",
                openrouter_request_id=None,
                ast_security_clean=True,
                forbidden_calls_found=(),
                line_count=100,
                ready_to_merge=True,
            )

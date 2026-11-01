# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the Dynamic Diurnal MoE Training Engine, Long-Tail Intent Scanner, and Exemplar BYOB Bees."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hivemind.core.auto_improvement import HF_EXPERT_CATALOG, NOCTURNAL_20_MOE_TARGETS
from hivemind.core.expert_registry import BYOB, ExpertRegistry
from hivemind.router.accounts import ProducerRecord
from hivemind.router.catalog import normalize_model_alias
from hivemind.swarm.dynamic_diurnal_moe_engine import (
    DiurnalMoETrainingScheduler,
    DiurnalOperationalMode,
    DiurnalProducerProfile,
    LongTailIntentScanner,
)
from hivemind.swarm.exemplar_byob_bees import (
    EXEMPLAR_BYOB_BEES,
    ExemplarBYOBBee,
    ExemplarBYOBRegistryManager,
)


def test_diurnal_producer_profile_timezone_calculation() -> None:
    # 00:00:00 UTC (timestamp 86400 * 1000)
    utc_midnight = 86400.0 * 1000.0

    # Paris / CET (+1.0)
    p_paris = DiurnalProducerProfile("p1", "paris-node", 16.0, "cuda", 1.0)
    assert p_paris.local_hour(utc_midnight) == 1.0  # 01:00 CET
    assert p_paris.is_local_night(utc_midnight) is True  # 01:00 is between 22:00 and 07:00

    # Montreal / EDT (-4.0)
    p_mtl = DiurnalProducerProfile("p2", "mtl-node", 24.0, "cuda", -4.0)
    assert p_mtl.local_hour(utc_midnight) == 20.0  # 20:00 EDT (8 PM)
    assert p_mtl.is_local_night(utc_midnight) is False  # 20:00 is daytime/evening

    # Tokyo / JST (+9.0)
    p_tokyo = DiurnalProducerProfile("p3", "tokyo-node", 16.0, "cuda", 9.0)
    assert p_tokyo.local_hour(utc_midnight) == 9.0  # 09:00 JST
    assert p_tokyo.is_local_night(utc_midnight) is False

    # Check nocturnal transition for Montreal when UTC is 03:00 (Montreal is 23:00 / 11 PM)
    utc_0300 = utc_midnight + 3 * 3600.0
    assert p_mtl.local_hour(utc_0300) == 23.0
    assert p_mtl.is_local_night(utc_0300) is True


def test_operational_mode_resolution_daytime_surplus_and_gamer_preemption() -> None:
    # 12:00 UTC (noon)
    utc_noon = 86400.0 * 1000.0 + 12 * 3600.0
    p_paris = DiurnalProducerProfile("p1", "paris-node", 16.0, "cuda", 1.0)
    # Local hour = 13:00 (1 PM) -> Daytime

    # Case A: Low load ratio (0.40) & surplus producers (3) -> DAYTIME_SURPLUS_HARVESTING
    mode_surplus = p_paris.resolve_mode(
        now_utc=utc_noon,
        system_load_ratio=0.40,
        surplus_producers_count=3,
        is_preempted=False,
    )
    assert mode_surplus == DiurnalOperationalMode.DAYTIME_SURPLUS_HARVESTING

    # Case B: High load ratio (0.85) & no surplus -> DAYTIME_INFERENCE_PRIMARY
    mode_busy = p_paris.resolve_mode(
        now_utc=utc_noon,
        system_load_ratio=0.85,
        surplus_producers_count=0,
        is_preempted=False,
    )
    assert mode_busy == DiurnalOperationalMode.DAYTIME_INFERENCE_PRIMARY

    # Case C: Gamer starts playing a 3D game -> GAMER_PREEMPTED
    mode_game = p_paris.resolve_mode(
        now_utc=utc_noon,
        system_load_ratio=0.40,
        surplus_producers_count=3,
        is_preempted=True,
    )
    assert mode_game == DiurnalOperationalMode.GAMER_PREEMPTED


def test_long_tail_intent_scanner_prioritizes_small_niche_requests() -> None:
    scanner = LongTailIntentScanner()
    now = 1000000.0

    # Simulate heavy traffic for coder (150 requests) and general (80 requests)
    for _ in range(150):
        scanner.record_request("coder", timestamp_utc=now - 1200)
    for _ in range(80):
        scanner.record_request("general_chat", timestamp_utc=now - 1000)

    # Simulate a single rare user request for Lean 4 theorem proving and 2 requests for ZK circuits
    scanner.record_request("maths_formal", timestamp_utc=now - 300)
    scanner.record_request("cryptography_zkp", timestamp_utc=now - 600)
    scanner.record_request("cryptography_zkp", timestamp_utc=now - 200)

    scores = scanner.score_all_domains(now, target_domains=NOCTURNAL_20_MOE_TARGETS)
    assert len(scores) == 20

    # Verify that niche/small request domains have significantly higher priority scores than heavily served code_feedback_debug
    top_domains = [s.canonical_domain for s in scores[:5]]
    assert "maths_formal" in top_domains or "cryptography_zkp" in top_domains
    score_lean4 = next(s for s in scores if s.canonical_domain == "maths_formal")
    score_debug = next(s for s in scores if s.canonical_domain == "code_feedback_debug")

    # Lean 4 (1 request, complexity 2.5) must have higher score than debug (150 requests)
    assert score_lean4.priority_score > score_debug.priority_score
    assert score_lean4.request_count == 1
    assert score_lean4.hf_dataset == "hoskinson-center/proofnet"


def test_diurnal_moe_training_scheduler_heterogeneous_timezones() -> None:
    scanner = LongTailIntentScanner()
    # 02:00 UTC
    now_utc = 86400.0 * 2000.0 + 2 * 3600.0

    scheduler = DiurnalMoETrainingScheduler(scanner=scanner)

    # Producer 1: Paris (+1.0) -> Local time 03:00 (NIGHT) -> INTENSE_NOCTURNAL_TRAINING
    scheduler.register_producer_profile(
        DiurnalProducerProfile("acc_paris", "paris.rtx3090", 24.0, "cuda-sm86", 1.0)
    )

    # Producer 2: Tokyo (+9.0) -> Local time 11:00 (DAYTIME) -> DAYTIME_SURPLUS_HARVESTING
    scheduler.register_producer_profile(
        DiurnalProducerProfile("acc_tokyo", "tokyo.t4", 16.0, "turing-t4", 9.0)
    )

    records = [
        ProducerRecord("acc_paris", "https://paris.example.com", (("coder", "q"),), 0.4, 50.0, 100.0, "cuda-sm86"),
        ProducerRecord("acc_tokyo", "https://tokyo.example.com", (("coder", "q"),), 0.4, 50.0, 100.0, "turing-t4"),
    ]

    wus = scheduler.schedule_diurnal_batch(
        now_utc=now_utc,
        active_records=records,
        system_load_ratio=0.35,  # Low load triggers surplus for daytime
    )

    assert len(wus) == 2
    wu_paris = next(w for w in wus if w.assigned_producer_id == "acc_paris")
    wu_tokyo = next(w for w in wus if w.assigned_producer_id == "acc_tokyo")

    # Paris is at night: intense nocturnal training
    assert wu_paris.operational_mode == DiurnalOperationalMode.INTENSE_NOCTURNAL_TRAINING
    assert wu_paris.target_rank == 32
    assert wu_paris.steps == 100
    assert wu_paris.local_clock_hour == 3.0
    # Nocturnal bonus applies (+25% multiplier)
    assert wu_paris.pouw_reward_hmt > 18.0

    # Tokyo is at daytime: daytime surplus harvesting
    assert wu_tokyo.operational_mode == DiurnalOperationalMode.DAYTIME_SURPLUS_HARVESTING
    assert wu_tokyo.target_rank == 16
    assert wu_tokyo.steps == 30
    assert wu_tokyo.local_clock_hour == 11.0
    assert wu_tokyo.pouw_reward_hmt < wu_paris.pouw_reward_hmt


def test_exemplar_byob_bees_lifecycle_and_attestation() -> None:
    registry = ExpertRegistry()
    manager = ExemplarBYOBRegistryManager(expert_registry=registry)

    live_bees = manager.list_live_bees()
    assert len(live_bees) == 3

    expected_ids = {"bee-quantum-simd", "bee-reverse-cve", "bee-genomics-crispr"}
    assert {b.bee_id for b in live_bees} == expected_ids

    for bee in live_bees:
        # Check manifest integrity
        manifest = bee.to_manifest()
        assert manifest.kind == "lora_adapter"
        assert manifest.license == "apache-2.0"
        assert manifest.vram_tier_gb <= 12
        assert len(manifest.revision) == 40

        # Check partner model mapping
        partner = bee.to_partner_model()
        assert partner.public_id == bee.public_id
        assert normalize_model_alias(bee.public_id) in (bee.bee_id, bee.base_tier_id)

        # IP-21b negative control: calling register_and_attest_bee without evaluation_record must raise
        with pytest.raises(ValueError, match="without a verified evaluation_record"):
            manager.register_and_attest_bee(bee)

        # Register and attest with a verified evaluation_record
        attestation = manager.register_and_attest_bee(
            bee,
            evaluation_record={
                "verdict": "ACCEPT",
                "mcnemar_p_value": bee.mcnemar_p_value,
                "accuracy_gain": bee.test_accuracy_gain,
            },
        )
        assert attestation["status"] == "APPROVED_LIVE"
        assert attestation["gate_passed"] is True
        assert attestation["accuracy_gain"] > 0.25
        assert attestation["mcnemar_p_value"] < 0.001
        assert len(attestation["attestation_sha256"]) == 64

    # Verify a rejected evaluation record does not register
    rejected_att = manager.register_and_attest_bee(
        live_bees[0],
        evaluation_record={"verdict": "REJECT", "mcnemar_p_value": 0.45, "accuracy_gain": -0.02},
    )
    assert rejected_att["status"] == "REJECTED"
    assert rejected_att["gate_passed"] is False

    # Verify registry records
    assert len(registry) == 3
    assert "bee-quantum-simd" in registry
    assert "bee-reverse-cve" in registry
    assert "bee-genomics-crispr" in registry


def test_plan_producer_quorum_moe_serving_and_boinc_cpu_sweep_install(tmp_path: Path) -> None:
    from hivemind.boinc.real_client.server_stack import (
        CPU_SOLVER_PLAN_CLASS,
        PLATFORM,
        Stack,
    )

    scheduler = DiurnalMoETrainingScheduler()
    now_utc = 86400.0 * 2000.0 + 2 * 3600.0
    scheduler.register_producer_profile(
        DiurnalProducerProfile("acc_1", "paris.rtx2070", 8.0, "cuda-sm75", 1.0)
    )
    scheduler.register_producer_profile(
        DiurnalProducerProfile("acc_2", "berlin.rtx3080", 12.0, "cuda-sm86", 1.0)
    )
    scheduler.register_producer_profile(
        DiurnalProducerProfile("acc_3", "mtl.rtx4070", 12.0, "cuda-sm89", -4.0)
    )
    scheduler.register_producer_profile(
        DiurnalProducerProfile("acc_4", "tokyo.t4", 16.0, "turing-t4", 9.0)
    )

    records = [
        ProducerRecord("acc_1", "https://p1.example.com", (("bee-rusty-sundials", "q"),), 0.35, 40.0, 80.0, "cuda-sm75"),
        ProducerRecord("acc_2", "https://p2.example.com", (("moe-scientific-wave2", "q"),), 0.35, 40.0, 80.0, "cuda-sm86"),
        ProducerRecord("acc_3", "https://p3.example.com", (("moe-pde-fluid-quantum", "q"),), 0.35, 40.0, 80.0, "cuda-sm89"),
        ProducerRecord("acc_4", "https://p4.example.com", (("moe-coder", "q"),), 0.35, 40.0, 80.0, "turing-t4"),
    ]
    serving_plan = scheduler.plan_producer_quorum_moe_serving(
        records,
        now_utc=now_utc,
        system_load_ratio=0.35,
    )
    assert serving_plan["connected_producers"] == 4
    assert serving_plan["stage_id"] == "stage_3_mesh4_wave2_pde_and_cluster_moe"
    assert serving_plan["wave2_scientific_moe_active"] is True
    assert serving_plan["quorum_sweep_enabled"] is True
    assert serving_plan["total_vram_gb"] == 48.0
    assert "hivemind/moe-pde-fluid-quantum" in serving_plan["unlocked_moe_models"]

    # Verify BOINC CPU solver sweep plan class and app template installation
    stack = Stack(
        base=tmp_path / "boinc_base",
        boinc_src=tmp_path / "boinc_src",
        venv_python="/usr/bin/python3",
        repo_src=tmp_path / "repo_src",
    )
    stack.root.mkdir(parents=True, exist_ok=True)
    stack.write_plan_class_spec()
    xml_text = (stack.root / "plan_class_spec.xml").read_text(encoding="utf-8")
    assert CPU_SOLVER_PLAN_CLASS in xml_text

    stack.install_cpu_sweep_app("#!/usr/bin/env python3\n", min_quorum=2)
    in_tpl = (stack.root / "templates" / "hivemind_solver_sweep_in").read_text(encoding="utf-8")
    assert "<min_quorum>2</min_quorum>" in in_tpl
    assert "<target_nresults>2</target_nresults>" in in_tpl
    app_dir = stack.root / "apps" / "hivemind_solver_sweep" / "100" / f"{PLATFORM}__{CPU_SOLVER_PLAN_CLASS}"
    version_xml = (app_dir / "version.xml").read_text(encoding="utf-8")
    assert f"<plan_class>{CPU_SOLVER_PLAN_CLASS}</plan_class>" in version_xml
    assert "NVIDIA" not in version_xml


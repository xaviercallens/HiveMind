# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for Season 19 (`HM-S19-Q35`) Qwen 3.5 Coding & Usage-Driven Idle/Nocturnal MoE Orchestrator."""

from __future__ import annotations

import torch

from hivemind.core.qwen35_usage_moe_orchestrator import (
    QWEN35_HARDWARE_TIERS,
    QWEN35_SPECIALIZED_BEES,
    ProducerNodeCapability,
    UsageDrivenQwen35MoEOrchestrator,
    build_seven_day_orchestration_plan,
    project_lora_rank_svd,
)


def test_hardware_tiers_respect_vram_ceilings() -> None:
    std = QWEN35_HARDWARE_TIERS["std_8gb"]
    big = QWEN35_HARDWARE_TIERS["big_12gb"]
    xl = QWEN35_HARDWARE_TIERS["xl_16gb_plus"]

    assert std["estimated_vram_nf4_mib"] < std["max_wddm_safe_mib"] == 6656
    assert big["estimated_vram_nf4_mib"] < big["max_wddm_safe_mib"]
    assert xl["estimated_vram_nf4_mib"] < xl["max_wddm_safe_mib"]
    assert std["lora_rank"] < big["lora_rank"] < xl["lora_rank"]


def test_usage_driven_cluster_shares_prioritize_coding() -> None:
    orch = UsageDrivenQwen35MoEOrchestrator(
        observed_completions_by_cluster={"coding": 873, "science": 551, "multilingual_byob": 3}
    )
    shares = orch.compute_cluster_target_shares()
    assert abs(sum(shares.values()) - 1.0) < 1e-3
    assert shares["coding"] > 0.55
    assert shares["science"] > 0.25
    assert shares["multilingual_byob"] >= 0.08


def test_idle_and_nocturnal_producer_scheduling_and_preemption() -> None:
    orch = UsageDrivenQwen35MoEOrchestrator()
    producers = [
        ProducerNodeCapability(
            node_id="rtx2070-paris",
            gpu_model="NVIDIA GeForce RTX 2070 8GB",
            vram_mib=8192,
            fp16_tflops=59.7,
            idle_seconds=120.0,
            is_european_night=True,
            active_inference_requests=0,
        ),
        ProducerNodeCapability(
            node_id="rtx4070-lyon",
            gpu_model="NVIDIA GeForce RTX 4070 12GB",
            vram_mib=12288,
            fp16_tflops=116.0,
            idle_seconds=45.0,
            is_european_night=False,
            active_inference_requests=0,
        ),
        ProducerNodeCapability(
            node_id="rtx3090-berlin",
            gpu_model="NVIDIA GeForce RTX 3090 24GB",
            vram_mib=24576,
            fp16_tflops=142.0,
            idle_seconds=600.0,
            is_european_night=True,
            active_inference_requests=0,
        ),
        ProducerNodeCapability(
            node_id="busy-inference-node",
            gpu_model="NVIDIA GeForce RTX 2070 8GB",
            vram_mib=8192,
            fp16_tflops=59.7,
            idle_seconds=5.0,
            is_european_night=False,
            active_inference_requests=2,
        ),
    ]

    wus = orch.schedule_workunits_for_producers(producers, round_index=1)
    # Busy node must be skipped to preserve <1.0s inference priority
    assert len(wus) == 3
    by_node = {w.node_id: w for w in wus}

    assert by_node["rtx2070-paris"].hardware_tier == "std_8gb"
    assert by_node["rtx2070-paris"].training_mode == "nocturnal_deep_qlora"
    assert by_node["rtx2070-paris"].estimated_vram_mib <= 6656
    assert by_node["rtx2070-paris"].inner_steps == 80

    assert by_node["rtx4070-lyon"].hardware_tier == "big_12gb"
    assert by_node["rtx4070-lyon"].training_mode == "daytime_idle_micro_qlora"
    assert by_node["rtx4070-lyon"].inner_steps == 30

    assert by_node["rtx3090-berlin"].hardware_tier == "xl_16gb_plus"
    assert by_node["rtx3090-berlin"].lora_rank == 32
    assert by_node["rtx3090-berlin"].inner_steps == 160


def test_svd_lora_rank_projection_retains_spectral_energy() -> None:
    torch.manual_seed(42)
    # Simulate a rank-32 Big-GPU teacher LoRA adapter with decaying singular spectrum
    u_true = torch.randn(128, 32)
    v_true = torch.randn(32, 64)
    scales = torch.exp(-torch.arange(32, dtype=torch.float32) * 0.18)
    lora_b = u_true * scales.unsqueeze(0)
    lora_a = v_true

    student_a, student_b, energy_ratio = project_lora_rank_svd(lora_a, lora_b, target_rank=16)
    assert student_a.shape == (16, 64)
    assert student_b.shape == (128, 16)
    assert energy_ratio > 0.95


def test_seven_day_orchestration_plan_completeness() -> None:
    plan = build_seven_day_orchestration_plan()
    assert plan["epic_code"] == "HM-S19-Q35"
    assert len(plan["days"]) == 6
    assert len(plan["plan_sha256"]) == 64
    assert len(QWEN35_SPECIALIZED_BEES) == 14


def test_hardware_tiers_backbones_exist_in_s21_catalog() -> None:
    import json
    from pathlib import Path

    cat_path = Path(__file__).resolve().parents[1] / "configs" / "base_models_s21.json"
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    verified_ids = {m["model_id"] for m in cat["models"] if m.get("exists") is True}
    not_found_ids = {m["model_id"] for m in cat.get("not_found", [])}

    for tier_id, cfg in QWEN35_HARDWARE_TIERS.items():
        for key in ("primary_backbone", "coder_backbone", "draft_backbone"):
            model_id = cfg[key]
            assert model_id in verified_ids, f"{tier_id}.{key}={model_id} missing from S21 verified models"
            assert model_id not in not_found_ids, f"{tier_id}.{key}={model_id} is in S21 not_found list"

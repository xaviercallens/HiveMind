# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for the Nocturnal Polyglot Software Engineering & Systems Coding MoE Engine."""

from __future__ import annotations

from pathlib import Path

from hivemind.core.nocturnal_coding_moe import (
    BIG_GPU_CODING_FLEET,
    MAX_CONSUMER_8GB_VRAM_MIB,
    NOCTURNAL_CODING_BASE_MODELS,
    NOCTURNAL_CODING_EXPERT_CLUSTERS,
    SUPPORTED_CODING_LANGUAGES,
    T4_MIN_FREE_VRAM_FLOOR_MIB,
    decompose_engineering_epic_to_dag,
    run_nocturnal_coding_moe_campaign,
)


def test_coding_moe_8_clusters_and_polyglot_languages() -> None:
    assert SUPPORTED_CODING_LANGUAGES == ("Python", "C", "C++", "Rust")
    assert len(NOCTURNAL_CODING_EXPERT_CLUSTERS) == 8

    expected_roles = {
        "architecture",
        "orchestrator_task_splitting",
        "unit_test",
        "qa",
        "coding",
        "pr_review_and_merge",
        "performance_improvement",
        "security_hardening",
    }
    actual_roles = {c["role_slug"] for c in NOCTURNAL_CODING_EXPERT_CLUSTERS.values()}
    assert actual_roles == expected_roles

    total_bees = 0
    for c in NOCTURNAL_CODING_EXPERT_CLUSTERS.values():
        assert set(c["languages"]) == {"Python", "C", "C++", "Rust"}
        assert c["primary_hf_dataset"]
        assert len(c["benchmarks"]) >= 2
        total_bees += len(c["specialist_bees"])
    assert total_bees == 16

    for bmodel in NOCTURNAL_CODING_BASE_MODELS.values():
        assert bmodel["peak_vram_mib_8gb_safe"] <= MAX_CONSUMER_8GB_VRAM_MIB


def test_epic_dag_task_splitter_covers_all_8_roles() -> None:
    dag = decompose_engineering_epic_to_dag("Polyglot Zero-Copy MoE Transport")
    assert len(dag) == 8
    assert dag[0].role_slug == "architecture"
    assert dag[0].depends_on == []
    assert dag[1].role_slug == "orchestrator_task_splitting"
    assert dag[1].depends_on == ["task_01_arch"]
    assert dag[-1].role_slug == "pr_review_and_merge"
    assert "task_07_qa" in dag[-1].depends_on


def test_run_nocturnal_coding_moe_campaign_end_to_end(tmp_path: Path) -> None:
    out_file = tmp_path / "nocturnal_coding_moe_report.json"
    report = run_nocturnal_coding_moe_campaign(
        inner_steps=250,
        lora_rank=16,
        lora_alpha=32.0,
        output_json_path=out_file,
    )
    assert out_file.exists()
    assert report.overall_gate_passed is True
    assert report.stage1_all_base_frozen is True
    assert report.stage2_invariant_i1_satisfied is True
    assert report.stage2_mean_loss_reduction_pct > 50.0
    assert report.stage4_poison_rejection_rate_pct == 100.0
    assert report.stage5_cluster_routing_accuracy >= 0.90
    assert report.stage5_upcycled_moe_loss < report.stage5_single_model_baseline_loss
    assert report.t4_free_vram_mib >= T4_MIN_FREE_VRAM_FLOOR_MIB
    assert set(report.polyglot_benchmark_scores.keys()) == {"Python", "C", "C++", "Rust"}

    # Verify Big-GPU Node distribution and 3-Loop Iterative Quality Ratchet
    assert report.big_gpu_nodes_count == len(BIG_GPU_CODING_FLEET) == 8
    assert all(n["vram_within_budget"] for n in report.big_gpu_node_allocations)
    assert report.num_quality_loops == 3
    assert len(report.quality_loop_iterations) == 3
    assert report.monotone_3loop_ratchet_passed is True

    l1, l2, l3 = report.quality_loop_iterations
    assert l1["heldout_hard_eval_loss"] > l2["heldout_hard_eval_loss"] > l3["heldout_hard_eval_loss"]
    assert l1["heldout_hard_routing_acc"] <= l2["heldout_hard_routing_acc"] <= l3["heldout_hard_routing_acc"]
    assert l1["all_zk_opoi_verified"] and l2["all_zk_opoi_verified"] and l3["all_zk_opoi_verified"]
    for lang in ("Python", "C", "C++", "Rust"):
        m_key = next(iter(l1["polyglot_loop_benchmarks"][lang].keys()))
        assert (
            l1["polyglot_loop_benchmarks"][lang][m_key]
            < l2["polyglot_loop_benchmarks"][lang][m_key]
            < l3["polyglot_loop_benchmarks"][lang][m_key]
        )

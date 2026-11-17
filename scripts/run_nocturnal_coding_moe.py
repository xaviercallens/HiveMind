#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Nocturnal Polyglot Software Engineering & Systems Coding MoE Intense Training Runner.

Executes the 5-stage Branch-Train-Mix (BTM) + DiLoCo-QLoRA nocturnal training campaign across
8 dedicated Software Engineering Expert Clusters (16 Specialist "Bee" Workers) for **Python, C, C++, and Rust**:
  1. `expert_1_architecture_design`: Architecture, Systems Design & FFI (`bigcode/the-stack-v2-dedup`)
  2. `expert_2_orchestrator_task_splitter`: DAG Orchestrator & Task Splitting (`princeton-nlp/SWE-bench_Verified`)
  3. `expert_3_unit_testing`: High-Coverage Unit & Property-Based Testing (`bigcode/commitpackft`, `nuprl/MultiPL-E`)
  4. `expert_4_qa_adversarial_verification`: Adversarial QA, Fuzzing & Sanitizers (`SWE-bench/SWE-smith`, `ASan`/`Miri`)
  5. `expert_5_polyglot_implementation`: Polyglot Core Coding (`OpenCoder-LLM/opc-sft-stage2`, `Magicoder-Evol-Instruct-110K`)
  6. `expert_6_pr_review_and_merge`: Automated PR Review, Diff Audit & 3-Way Merge (`bigcode/commitpackft`)
  7. `expert_7_performance_optimization`: SIMD (`AVX2`/`AVX-512`), CUDA/PTX & Zero-Alloc (`pie-benchmark/PIE-Perf-Optimization`)
  8. `expert_8_security_hardening`: Memory Safety, CWE-Top25 & DevSecOps (`CyberNative/Code_Vulnerability_Security_DPO`, `PrimeVul`)

Usage:
  PYTHONPATH=src .venv/bin/python scripts/run_nocturnal_coding_moe.py [--inner-steps 500] [--rank 16] [--alpha 32]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HIVEMIND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HIVEMIND_DIR / "src"))

from hivemind.core.nocturnal_coding_moe import (
    BIG_GPU_CODING_FLEET,
    DEFAULT_CODING_DILOCO_STEPS,
    DEFAULT_CODING_LORA_ALPHA,
    DEFAULT_CODING_LORA_RANK,
    NOCTURNAL_CODING_BASE_MODELS,
    NOCTURNAL_CODING_EXPERT_CLUSTERS,
    run_nocturnal_coding_moe_campaign,
)

ARTIFACTS_DIR = HIVEMIND_DIR / "artifacts"
DEFAULT_REPORT_PATH = ARTIFACTS_DIR / "nocturnal_coding_moe_report.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Nocturnal Polyglot Software Engineering & Systems Coding MoE Intense Training."
    )
    parser.add_argument(
        "--inner-steps",
        type=int,
        default=DEFAULT_CODING_DILOCO_STEPS,
        help="Number of local DiLoCo-QLoRA steps per Coding Bee worker (default: 500).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=DEFAULT_CODING_LORA_RANK,
        help="LoRA adapter rank (default: 16).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_CODING_LORA_ALPHA,
        help="LoRA adapter scaling alpha (default: 32.0).",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=3,
        help="Number of iterative evaluation & quality-improvement loops across Big GPU nodes (default: 3).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_REPORT_PATH),
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    print("=" * 98)
    print("🌙 HIVEMIND DISTRIBUTED BIG-GPU 3-LOOP POLYGLOT CODING MoE TRAINING & EVALUATION (PY/C/C++/RS)")
    print("=" * 98)
    print(f"[*] Distributed Big-GPU & Hybrid Fleet ({len(BIG_GPU_CODING_FLEET)} Nodes):")
    for node in BIG_GPU_CODING_FLEET:
        clusters_str = ", ".join(node.assigned_cluster_ids)
        print(
            f"    - {node.node_id:30s} | {node.total_vram_mib:6.0f} MiB | "
            f"Rank={node.lora_rank:2d} | Clusters: {clusters_str}"
        )

    print(f"\n[*] Base Open-Weight Coding Backbones ({len(NOCTURNAL_CODING_BASE_MODELS)}):")
    for key, model in NOCTURNAL_CODING_BASE_MODELS.items():
        print(
            f"    - {key:32s} | {model['model_id']:44s} | "
            f"8GB Safe Peak: {model['peak_vram_mib_8gb_safe']:.0f} MiB"
        )

    print(f"\n[*] Swarm Coding Orchestrator — {len(NOCTURNAL_CODING_EXPERT_CLUSTERS)} Dedicated Expert Clusters:")
    for cid, cmeta in NOCTURNAL_CODING_EXPERT_CLUSTERS.items():
        print(f"    - [{cid}] {cmeta['title']}")
        print(
            f"      HF Datasets: {cmeta['primary_hf_dataset']}, "
            + ", ".join(cmeta["secondary_hf_datasets"])
        )
        print(f"      Benchmarks:  {', '.join(cmeta['benchmarks'])}")

    print(
        f"\n[*] Launching Distributed Big-GPU {args.loops}-Loop Quality Ratchet + 5-Stage BTM/DiLoCo-QLoRA "
        f"(Rank={args.rank}, Alpha={args.alpha}, DiLoCo Local Steps={args.inner_steps})..."
    )
    report = run_nocturnal_coding_moe_campaign(
        inner_steps=args.inner_steps,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        num_quality_loops=args.loops,
        output_json_path=args.output,
    )

    print("\n" + "-" * 98)
    print(f"🔁 3-LOOP ITERATIVE EVALUATION & QUALITY RATCHET TELEMETRY (Device: {report.execution_device})")
    print("-" * 98)
    for it in report.quality_loop_iterations:
        print(
            f"  [Loop {it['loop_index']}] {it['loop_title']}\n"
            f"           Strategy:       {it['refinement_strategy']}\n"
            f"           Hard Eval Loss: {it['heldout_hard_eval_loss']:.5f} "
            f"(Delta E = {it['delta_energy_vs_prev_loop']:+.5f}, Ratchet Accepted = {it['monotone_ratchet_accepted']})\n"
            f"           Hard Top-1 Acc: {100.0 * it['heldout_hard_routing_acc']:.2f}% | "
            f"Top-2 Expert Acc: {100.0 * it['heldout_top2_expert_acc']:.2f}% | "
            f"SQNR: {it['mean_bee_sqnr_db']:.2f} dB | ZK-OPoI: {it['all_zk_opoi_verified']}"
        )
        b_py = it["polyglot_loop_benchmarks"]["Python"]["SWE_bench_Verified_ResolvedPct"]
        b_c = it["polyglot_loop_benchmarks"]["C"]["MultiPLE_C_PassAt1"]
        b_cpp = it["polyglot_loop_benchmarks"]["C++"]["LiveCodeBench_CPP_PassAt1"]
        b_rs = it["polyglot_loop_benchmarks"]["Rust"]["MultiPLE_Rust_PassAt1"]
        print(
            f"           Benchmarks:     Python SWE-bench={b_py:.1f}% | C MultiPL-E={b_c:.1f}% | "
            f"C++ LiveCodeBench={b_cpp:.1f}% | Rust MultiPL-E={b_rs:.1f}%"
        )

    print("\n" + "-" * 98)
    print("📊 STAGE-BY-STAGE POLYGLOT CODING MoE TELEMETRY SUMMARY")
    print("-" * 98)
    print(f"  Stage 1 (Base Freeze - NF4 Backbone):           {report.stage1_all_base_frozen}")
    print(f"  Monotone 3-Loop Quality Ratchet Passed:         {report.monotone_3loop_ratchet_passed}")
    print(
        f"  Stage 2 (DiLoCo-QLoRA {report.inner_diloco_steps} steps, 16 Bees / 8 Clusters): "
        f"Mean Loss Reduction = {report.stage2_mean_loss_reduction_pct:.2f}% | "
        f"Max 8GB-Tier VRAM = {report.stage2_max_consumer_vram_mib:.1f} MiB "
        f"(I1 <= 6656 MiB: {report.stage2_invariant_i1_satisfied})"
    )
    for b in report.bee_training_results:
        langs = "/".join(b["target_languages"])
        print(
            f"    • {b['bee_id']:34s} | [{langs:15s}] | "
            f"Loss: {b['loss_before']:.4f} -> {b['loss_after']:.4f} (-{b['loss_reduction_pct']:.1f}%) | "
            f"Acc: {b['accuracy_before']:.2f} -> {b['accuracy_after']:.2f} | "
            f"SQNR: {b['sqnr_db']:.1f} dB"
        )
    print(
        f"  Stage 3 (2-Bit SparseLoCo-DCT DHT Exchange):    "
        f"Compression = {report.stage3_mean_compression_ratio:.2f}x | "
        f"Mean SQNR = {report.stage3_mean_sqnr_db:.2f} dB | "
        f"DHT Keys = {len(report.stage3_dht_keys_published)}"
    )
    print(
        f"  Stage 4 (Byzantine Multi-Krum + 10% Replay):    "
        f"Poison Rejection = {report.stage4_poison_rejection_rate_pct:.1f}% | "
        f"Replay Cosine = {report.stage4_replay_cosine_similarity:.6f} | "
        f"L2 Drift = {report.stage4_replay_l2_drift:.6f}"
    )
    print(
        f"  Stage 5 (BTM Upcycled Top-2 Coding MoE):        "
        f"Routing Acc = {100.0 * report.stage5_cluster_routing_accuracy:.1f}% | "
        f"Baseline Loss = {report.stage5_single_model_baseline_loss:.4f} -> "
        f"Coding-MoE Loss = {report.stage5_upcycled_moe_loss:.4f} "
        f"(+{report.stage5_moe_vs_baseline_gain_pct:.2f}% Gain)"
    )
    print(
        f"  Tesla T4 Free VRAM Guard:                       "
        f"{report.t4_free_vram_mib:.1f} MiB free (floor >= 4300.0 MiB: {report.t4_vram_floor_satisfied})"
    )
    print(f"  Overall Gate Passed:                            {report.overall_gate_passed}")
    print(f"  Report SHA-256:                                 {report.report_sha256}")
    print(f"  Saved Report:                                   {args.output}")
    print("=" * 94)

    return 0 if report.overall_gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""CLI Runner for the HiveMind 'La Ruche' Massive 7-Division Image-MoE Swarm Campaign (42.87M pairs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hivemind.core.massive_image_moe_swarm_orchestrator import (  # noqa: E402
    MASSIVE_IMAGE_MOE_DIVISIONS,
    SWARM_HARDWARE_TIERS,
    run_massive_swarm_image_moe_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the HiveMind 'La Ruche' Massive 7-Division Image-MoE Swarm Campaign"
    )
    parser.add_argument("--inner-steps", type=int, default=500, help="Local DiLoCo steps (default: 500)")
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "massive_swarm_image_moe_report.json"),
        help="Output JSON report path",
    )
    args = parser.parse_args()

    print("=" * 96)
    print("🐝 HIVEMIND 'LA RUCHE' — MASSIVE 7-DIVISION IMAGE-MoE SWARM CAMPAIGN (42.87M SAMPLES, 104 BEES)")
    print("=" * 96)
    for t_id, t_meta in SWARM_HARDWARE_TIERS.items():
        print(
            f"  [*] {t_meta['tier_name']:58s} | Nodes: {t_meta['node_count']:2d} "
            f"| Max VRAM: {t_meta['max_vram_mib']:.0f} MiB"
        )

    report = run_massive_swarm_image_moe_campaign(
        inner_steps=args.inner_steps,
        output_json_path=args.output,
    )

    print("-" * 96)
    print(
        f"📦 SWARM DATA ORCHESTRATOR: {report['total_divisions']} Divisions | "
        f"{report['total_hf_datasets']} HF Datasets | {report['total_dataset_samples']:,} Samples | "
        f"{report['total_streaming_shards']:,} Shards ({report['samples_per_shard']:,}/shard) | "
        f"{report['total_swarm_bees']} Bees"
    )
    print("-" * 96)
    for d in report["division_reports"]:
        print(
            f"  • [{d['division_id']:34s}] Bees: {d['allocated_bees']:2d} | Samples: {d['total_samples']:10,d} "
            f"({d['total_shards']:4d} shards) | Loss: {d['focal_loss_before']:.4f} -> {d['focal_loss_after']:.4f} "
            f"(-{d['focal_loss_reduction_pct']:.1f}%) | DCT: {d['compression_ratio']}x ({d['sqnr_db']} dB)"
        )
    print("-" * 96)
    print(f"  Stage 1 (Base Freeze & VRAM Envelopes): Micro <= 1536 MiB = {report['stage1_strictly_under_1500mb_micro']} "
          f"({report['stage1_max_micro_vram_mib']:.0f} MiB) | I1 <= 6656 MiB = {report['stage1_invariant_i1_satisfied']} "
          f"({report['stage1_max_consumer_8gb_vram_mib']:.0f} MiB)")
    print(f"  Stage 2 (DiLoCo {report['stage2_inner_diloco_steps']} steps, Muon + 8-bit AdamW + Focal Loss): "
          f"Mean Loss Reduction = -{report['stage2_mean_focal_loss_reduction_pct']:.2f}%")
    print(f"  Stage 3 (DHT + Multi-Krum + Nesterov + ZK-OPoI): DHT Keys = {report['stage3_dht_keys_published_count']} "
          f"| Byzantine Rejected = {report['stage3_all_byzantine_rejected']} | ZK-OPoI = {report['stage3_all_zk_opoi_verified']}")
    print(f"  Stage 4 (BTM 7-Division Top-2 FoMoE Upcycling): Routing Acc = {report['stage4_routing_accuracy'] * 100:.1f}% "
          f"| Baseline Loss = {report['stage4_single_baseline_focal_loss']:.4f} -> V-MoE Loss = {report['stage4_upcycled_moe_focal_loss']:.4f} "
          f"(+{report['stage4_moe_vs_baseline_gain_pct']:.2f}%)")
    print(f"  Overall Campaign Gate Passed: {report['overall_campaign_passed']} | SHA-256: {report['report_sha256']}")
    print("=" * 96)
    return 0 if report["overall_campaign_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

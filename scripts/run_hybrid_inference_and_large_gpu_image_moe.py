#!/usr/bin/env python3
"""CLI Runner for Asymmetric Large-GPU Producer Image-MoE Training & Hybrid Inference Distribution."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hivemind.core.hybrid_inference_and_large_gpu_image_moe_distributor import (  # noqa: E402
    run_hybrid_inference_and_large_gpu_image_moe_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Distribute heavy Image-MoE training to Large-GPU Producers while Hybrid/Cloud servers provide live inference."
    )
    parser.add_argument("--inner-steps", type=int, default=500, help="Inner DiLoCo-QLoRA steps per Large-GPU producer")
    args = parser.parse_args()

    report_path = PROJECT_ROOT / "artifacts" / "hybrid_inference_and_large_gpu_image_moe_report.json"
    registry_path = PROJECT_ROOT / "artifacts" / "hivemind_inference_producers.json"
    report = run_hybrid_inference_and_large_gpu_image_moe_campaign(
        inner_steps=args.inner_steps,
        save_report_path=report_path,
        update_producer_registry_path=registry_path,
    )

    print("=" * 92)
    print("  HIVEMIND ASYMMETRIC LARGE-GPU IMAGE-MOE TRAINING & HYBRID INFERENCE DISTRIBUTION")
    print("=" * 92)
    print(f"  Campaign ID                : {report['campaign_id']}")
    print(
        f"  Total Shards Distributed   : {report['total_shards']:,} shards ({report['total_samples']:,} samples across {report['total_divisions']} Divisions)"
    )
    print(
        f"  Large-GPU Producers (Pool A): {report['large_gpu_producers_count']} nodes -> {report['large_gpu_shards_assigned']:,} shards ({report['large_gpu_training_share_pct']}%)"
    )
    print(
        f"  Hybrid Inference (Pool B)  : {report['hybrid_inference_nodes_count']} nodes -> {report['hybrid_micro_shards_assigned']:,} micro-shards ({report['hybrid_micro_training_share_pct']}%)"
    )
    print(f"  Large-GPU Mean Loss Drop   : -{report['large_gpu_mean_focal_loss_reduction_pct']}% (ZK-OPoI={report['all_large_gpu_zk_opoi_verified']})")
    print(
        f"  Hybrid Live Inference      : {report['hybrid_total_inference_requests_served']} reqs | p50={report['hybrid_mean_inference_p50_latency_ms']} ms | Preemption={report['hybrid_mean_preemption_yield_ms']} ms (<50 ms SLA)"
    )
    print(
        f"  VRAM & Hot-Swap Gates      : LargeVRAM={report['all_large_gpu_vram_within_budget']} | HybridVRAM={report['all_hybrid_vram_safety_gates_passed']} | 0ms HotSwap={report['all_zero_downtime_adapter_hotswaps_verified']}"
    )
    print(f"  Overall Gate Passed        : {report['overall_distribution_passed']} | SHA-256: {report['report_sha256']}")
    print("=" * 92)
    return 0 if report["overall_distribution_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

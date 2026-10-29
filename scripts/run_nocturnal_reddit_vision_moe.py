#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Nocturnal Reddit-Vision HiveMind V-MoE Intense Training Runner.

Executes the 5-stage Branch-Train-Mix (BTM) + DiLoCo-QLoRA training pipeline for
Decentralized Vision-Language Grounding & Visual Cultural Understanding:
  - Expert Cluster 1: Aesthetics, Digital Art & Rendering
    (`laion/aesthetic-visual-dataset`, `poloclub/diffusiondb`, `BakeLab/Visual-Aesthetic-Benchmark`)
  - Expert Cluster 2: Memes, Humor & Contemporary Web Slang
    (`HuggingFaceM4/Docmatix`, `lmms-lab/MMVet`, `reddit/meme-caption-shards`, `WebLI/Danbooru`)
  - Expert Cluster 3: Structured Visual Reasoning & Diagrams
    (`lmms-lab/ChartQA`, `lmms-lab/MathVision`, `fondant-ai/fondant-cc-25m`, `Geo170K`, `open-thoughts/docvqa`)

Usage:
  uv run python scripts/run_nocturnal_reddit_vision_moe.py [--inner-steps 500] [--rank 16] [--alpha 32]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HIVEMIND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HIVEMIND_DIR / "src"))

from hivemind.core.reddit_vision_vlm_moe import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_RANK,
    DEFAULT_NOCTURNAL_DILOCO_STEPS,
    REDDIT_VISION_BASE_MODELS,
    REDDIT_VISION_EXPERT_CLUSTERS,
    run_nocturnal_reddit_vision_moe_campaign,
)

ARTIFACTS_DIR = HIVEMIND_DIR / "artifacts"
DEFAULT_REPORT_PATH = ARTIFACTS_DIR / "nocturnal_reddit_vision_moe_report.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Nocturnal Reddit-Vision HiveMind (Culturally-Grounded VLM) MoE Training."
    )
    parser.add_argument(
        "--inner-steps",
        type=int,
        default=DEFAULT_NOCTURNAL_DILOCO_STEPS,
        help="Number of local DiLoCo-QLoRA steps per Bee worker (default: 500).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=DEFAULT_LORA_RANK,
        help="LoRA adapter rank (default: 16).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_LORA_ALPHA,
        help="LoRA adapter scaling alpha (default: 32.0).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_REPORT_PATH),
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    print("=" * 88)
    print("🌙 HIVEMIND NOCTURNAL 'REDDIT-VISION' CULTURALLY-GROUNDED VLM MoE TRAINING")
    print("=" * 88)
    print(f"[*] Base Multimodal Backbones ({len(REDDIT_VISION_BASE_MODELS)}):")
    for key, model in REDDIT_VISION_BASE_MODELS.items():
        print(
            f"    - {key:16s} | {model['model_id']:36s} | VRAM: ~{model['target_vram_gb']:.1f} GB "
            f"(8GB Safe Peak: {model['peak_vram_mib_8gb_safe']:.0f} MiB)"
        )

    print(f"\n[*] Swarm Data Orchestrator — {len(REDDIT_VISION_EXPERT_CLUSTERS)} Expert Clusters:")
    for cid, cmeta in REDDIT_VISION_EXPERT_CLUSTERS.items():
        print(f"    - [{cid}] {cmeta['title']}")
        print(
            f"      Datasets: {cmeta['primary_hf_dataset']}, "
            + ", ".join(cmeta["secondary_hf_datasets"])
        )

    print(
        f"\n[*] Launching 5-Stage Pipeline (Rank={args.rank}, Alpha={args.alpha}, "
        f"DiLoCo Local Steps={args.inner_steps})..."
    )
    report = run_nocturnal_reddit_vision_moe_campaign(
        inner_steps=args.inner_steps,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        output_json_path=args.output,
    )

    print("\n" + "-" * 88)
    print("📊 STAGE-BY-STAGE TELEMETRY SUMMARY")
    print("-" * 88)
    print(f"  Stage 1 (Base Freeze - ViT + LLM Attention):    {report.stage1_all_base_frozen}")
    print(
        f"  Stage 2 (DiLoCo-QLoRA {report.inner_diloco_steps} steps, NF4+8bit AdamW): "
        f"Mean Loss Reduction = {report.stage2_mean_loss_reduction_pct:.2f}% | "
        f"Max 8GB-Tier VRAM = {report.stage2_max_consumer_vram_mib:.1f} MiB "
        f"(I1 <= 6656 MiB: {report.stage2_invariant_i1_satisfied})"
    )
    for b in report.bee_training_results:
        print(
            f"    • {b['bee_id']:36s} | {b['backbone_model_id']:34s} | "
            f"Loss: {b['loss_before']:.4f} -> {b['loss_after']:.4f} (-{b['loss_reduction_pct']:.1f}%) | "
            f"Acc: {b['accuracy_before']:.2f} -> {b['accuracy_after']:.2f} | "
            f"SQNR: {b['sqnr_db']:.1f} dB ({b['compression_ratio']:.1f}x)"
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
        f"  Stage 5 (BTM Upcycled Top-2 Softmax V-MoE):     "
        f"Routing Acc = {100.0 * report.stage5_cluster_routing_accuracy:.1f}% | "
        f"Baseline Loss = {report.stage5_single_model_baseline_loss:.4f} -> "
        f"V-MoE Loss = {report.stage5_upcycled_moe_loss:.4f} "
        f"(+{report.stage5_moe_vs_baseline_gain_pct:.2f}% Gain)"
    )
    print(f"  Overall Gate Passed:                            {report.overall_gate_passed}")
    print(f"  Report SHA-256:                                 {report.report_sha256}")
    print(f"  Saved Report:                                   {args.output}")
    print("=" * 88)

    return 0 if report.overall_gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

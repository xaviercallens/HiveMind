#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""CLI Runner for the HiveMind 'La Ruche' Exoplanet & Astrophysics Vision-MoE (<1.5 GB VRAM) PoC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hivemind.core.exoplanet_astro_vmoe_poc import (  # noqa: E402
    DEFAULT_FOCAL_LOSS_GAMMA,
    EXOPLANET_ASTRO_EXPERT_BRANCHES,
    run_exoplanet_astro_vmoe_poc,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the HiveMind 'La Ruche' Exoplanet & Astrophysics Vision-MoE (<1.5 GB VRAM) PoC"
    )
    parser.add_argument("--inner-steps", type=int, default=500, help="Local DiLoCo steps (default: 500)")
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=DEFAULT_FOCAL_LOSS_GAMMA,
        help="Focal Loss gamma parameter (default: 2.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "exoplanet_astro_vmoe_poc_report.json"),
        help="Output JSON report path",
    )
    args = parser.parse_args()

    print("=" * 92)
    print("🔭 HIVEMIND 'LA RUCHE' — POC ASTRONOMIE & CHASSE AUX EXOPLANÈTES (VISION-MoE < 1,5 Go VRAM)")
    print("=" * 92)
    for b_id, meta in EXOPLANET_ASTRO_EXPERT_BRANCHES.items():
        print(
            f"  • [{b_id}] {meta['branch_name_fr']:42s} | {meta['backbone_model_id']:32s} "
            f"| Patch: {meta['patch_resolution']} | VRAM: {meta['peak_vram_mib']:.0f} MiB"
        )

    report = run_exoplanet_astro_vmoe_poc(
        inner_steps=args.inner_steps,
        focal_gamma=args.focal_gamma,
        output_json_path=args.output,
    )

    print("-" * 92)
    print("📊 RÉSULTATS DES 4 ÉTAPES DU PIPELINE DÉCENTRALISÉ")
    print("-" * 92)
    print(f"  Étape 1 (Gel Backbone NF4 + QLoRA FFN):  {report['stage1_backbone_frozen']} "
          f"| VRAM: {report['stage1_min_vram_mib']:.0f}-{report['stage1_max_vram_mib']:.0f} MiB (< 1536 MiB: {report['stage1_strictly_under_1500mb_vram']})")
    print(f"  Étape 2 (DiLoCo {report['stage2_inner_diloco_steps']} steps + Focal Loss gamma={report['stage2_focal_loss_gamma']}): "
          f"Réduction moyenne Focal Loss = -{report['stage2_mean_focal_loss_reduction_pct']:.2f}%")
    for b in report["branch_reports"]:
        print(
            f"    - {b['branch_id']:30s} | Focal Loss: {b['focal_loss_before']:.4f} -> {b['focal_loss_after']:.4f} "
            f"(-{b['focal_loss_reduction_pct']:.1f}%) | Compression DCT: {b['compression_ratio']}x ({b['sqnr_db']} dB) "
            f"| Courbes citoyennes: {b['citizen_ambiguous_curves_count']}"
        )
    print(f"  Étape 3 (Multi-Krum + Nesterov Externe): Poison rejeté = {report['stage3_all_byzantine_rejected']} "
          f"| Reçus ZK-OPoI = {report['stage3_all_zk_opoi_verified']}")
    print(f"  Étape 4 (Upcycling BTM Top-2 Softmax):   Précision Routeur = {report['stage4_routing_accuracy'] * 100:.1f}% "
          f"| Baseline Loss = {report['stage4_single_baseline_focal_loss']:.4f} -> V-MoE Loss = {report['stage4_upcycled_vmoe_focal_loss']:.4f} "
          f"(+{report['stage4_vmoe_gain_pct']:.2f}%)")

    print("-" * 92)
    print("🪐 SIMULATEUR INTERACTIF PHYSIQUE : CE QUE VOIENT LES EXPERTS VISUELS")
    print("-" * 92)
    s1 = report["interactive_simulator_scenarios"]["earth_around_sun_1Rearth_1Rsun"]
    s2 = report["interactive_simulator_scenarios"]["earth_around_red_dwarf_1Rearth_02Rsun"]
    s3 = report["interactive_simulator_scenarios"]["eclipsing_binary_false_positive"]
    print(
        f"  1) Terre (1.0 R_earth) / Soleil (1.0 R_sun) [Bruit 100 ppm]  -> Delta F/F = {s1['transit_depth_ppm']:.1f} ppm "
        f"(SNR={s1['single_transit_snr']:.2f}, Noyé={s1['submerged_in_instrumental_noise']}) "
        f"| Vote Citoyen (Entropie={s1['normalized_entropy']:.2f}): {s1['citizen_science_review_triggered']}"
    )
    print(
        f"  2) Terre (1.0 R_earth) / Naine Rouge (0.2 R_sun) [100 ppm]   -> Delta F/F = {s2['transit_depth_ppm']:.1f} ppm "
        f"(SNR={s2['single_transit_snr']:.2f}) | Routeur Top-1: {s2['top1_expert']} (p={s2['top1_probability']:.3f})"
    )
    print(
        f"  3) Binaire à Éclipses (Asymétrie Pair/Impair + Centroïde)    -> Routeur Top-1: {s3['top1_expert']} "
        f"(p={s3['top1_probability']:.3f})"
    )
    print(f"  PoC Gate Passed: {report['overall_poc_passed']} | SHA-256: {report['report_sha256']}")
    print("=" * 92)
    return 0 if report["overall_poc_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

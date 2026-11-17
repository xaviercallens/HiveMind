#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""CLI Runner for the Intensive 7-Division Image-MoE Inference Benchmark & 12-Scenario Public Gallery."""

from __future__ import annotations

import argparse
from pathlib import Path

from hivemind.core.image_moe_inference_gallery_benchmark import (
    run_intensive_image_moe_inference_stress_campaign,
)
from hivemind.router.portal_page import render_portal_page


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run intensive Image-MoE inference tests on 12 public image scenarios and update the Life as a Bee web page."
    )
    parser.add_argument(
        "--concurrent-requests-per-node",
        type=int,
        default=80,
        help="Concurrent inference requests per Hybrid Inference Server (default: 80 -> 240 total).",
    )
    parser.add_argument(
        "--gallery-dir",
        type=Path,
        default=Path("public/gallery"),
        help="Output directory for the 12 public SVG visual assets.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("artifacts/image_moe_inference_gallery_benchmark.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("public/life-as-a-bee.html"),
        help="Standalone HTML output path for the updated Life as a Bee web page.",
    )
    args = parser.parse_args()

    report = run_intensive_image_moe_inference_stress_campaign(
        concurrent_requests_per_node=args.concurrent_requests_per_node,
        output_gallery_dir=args.gallery_dir,
        save_report_path=args.output_report,
    )

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    html_page = render_portal_page(None, "static-life-as-a-bee-gallery")
    args.output_html.write_text(html_page, encoding="utf-8")

    print("=" * 94)
    print("  HIVEMIND INTENSIVE IMAGE-MOE INFERENCE BENCHMARK & 12-SCENARIO PUBLIC GALLERY")
    print("=" * 94)
    print(f"  Campaign ID                : {report['campaign_id']}")
    print(f"  Public Image Scenarios     : {report['total_gallery_scenarios']} across {report['divisions_covered']} Divisions (HTTP 200={report['all_scenarios_http_200']})")
    print(f"  FoMoE Top-1 Routing Acc    : {report['top1_division_routing_accuracy'] * 100:.1f}%")
    print(f"  Mean Warm Latency / Speed  : {report['mean_warm_latency_ms']} ms | {report['mean_tokens_per_sec']} tok/s | Conf={report['mean_visual_grounding_confidence']}")
    print(f"  Stress Burst (3 Hybrid)    : {report['stress_burst_total_requests']} reqs | Zero OOM={report['stress_burst_zero_oom']}")
    print(f"  Gallery Assets Saved       : {args.gallery_dir}")
    print(f"  Updated Web Page           : {args.output_html}")
    print(f"  Overall Gate Passed        : {report['overall_passed']} | SHA-256: {report['report_sha256']}")
    print("=" * 94)
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

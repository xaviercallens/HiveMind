#!/usr/bin/env python3
"""Run the End-User Attribution & Conversion Learning + Developer, Image & Video MoE Integration Benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hivemind.core.dev_image_video_moe_integration_benchmark import (  # noqa: E402
    run_intense_dev_image_video_moe_benchmark,
)
from hivemind.router.portal_page import render_portal_page  # noqa: E402


def main() -> int:
    report = run_intense_dev_image_video_moe_benchmark()
    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "dev_image_video_moe_integration_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Regenerate public/life-as-a-bee.html so the static mirror includes the new scenarios
    public_dir = REPO_ROOT / "public"
    if public_dir.exists():
        (public_dir / "life-as-a-bee.html").write_text(
            render_portal_page(None, "static-life-as-a-bee"),
            encoding="utf-8",
        )

    summary = report["overall_summary"]
    conv = report["end_user_attribution_and_conversion"]
    print(
        json.dumps(
            {
                "campaign_id": report["campaign_id"],
                "status": report["status"],
                "models_integrated": report["models_integrated_and_verified"],
                "total_scenarios": summary["total_scenarios"],
                "total_stress_requests": summary["total_stress_requests"],
                "developer_coding_3tome_gain": {
                    "mean_quality_gain_pct": summary["developer_coding_3tome_gain"]["mean_quality_gain_pct"],
                    "mean_speedup_x": summary["developer_coding_3tome_gain"]["mean_speedup_x"],
                    "pass_at_1_gain_vs_dense_pp": summary["developer_coding_3tome_gain"]["pass_at_1_gain_vs_dense_pp"],
                },
                "image_generation_vision_moe_gain": {
                    "mean_quality_gain_pct": summary["image_generation_vision_moe_gain"]["mean_quality_gain_pct"],
                    "mean_speedup_x": summary["image_generation_vision_moe_gain"]["mean_speedup_x"],
                    "laion_aesthetic_score": summary["image_generation_vision_moe_gain"][
                        "laion_aesthetic_score_out_of_10"
                    ],
                },
                "video_generation_spatiotemporal_moe_gain": {
                    "mean_quality_gain_pct": summary["video_generation_spatiotemporal_moe_gain"][
                        "mean_quality_gain_pct"
                    ],
                    "mean_speedup_x": summary["video_generation_spatiotemporal_moe_gain"]["mean_speedup_x"],
                    "fvd_reduction_pct": summary["video_generation_spatiotemporal_moe_gain"]["fvd_reduction_pct"],
                },
                "attributed_user_cohorts": len(conv["end_user_attributions"]),
                "mean_visitor_to_signup_rate_pct": conv["cohort_funnel_totals"]["mean_visitor_to_signup_rate_pct"],
                "mean_signup_to_active_rate_pct": conv["cohort_funnel_totals"]["mean_signup_to_active_rate_pct"],
                "artifact_path": str(out_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

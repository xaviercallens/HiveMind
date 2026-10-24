#!/usr/bin/env python3
"""CLI Runner for the Dataset Mining Campaign, La Ruche MoE Model Vault & HiveMind@home GCP Dashboard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hivemind.boinc.hivemind_at_home_dashboard import (  # noqa: E402
    create_hivemind_at_home_app,
    export_hivemind_at_home_artifacts,
)
from hivemind.data.dataset_mining_campaign import (  # noqa: E402
    run_dataset_mining_and_ruche_campaign,
)
from hivemind.swarm.la_ruche_vault import load_community_moe_from_ruche  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute Dataset Mining Campaign, populate La Ruche MoE Vault, and build HiveMind@home GCP Dashboard."
    )
    parser.add_argument(
        "--vault-dir",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "la_ruche_vault"),
        help="Path to the La Ruche Beehive Model Vault directory.",
    )
    parser.add_argument(
        "--mining-report",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "dataset_mining_and_ruche_report.json"),
        help="Output JSON report path for the Dataset Mining Campaign.",
    )
    parser.add_argument(
        "--gpu-campaigns-report",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "season9_10_gpu_swarm_campaigns_report.json"),
        help="Input Season 9/10 GPU Swarm Campaigns report JSON path.",
    )
    parser.add_argument(
        "--dashboard-html",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "hivemind_at_home_dashboard.html"),
        help="Output HTML5 path for the HiveMind@home + GCP + La Ruche dashboard.",
    )
    parser.add_argument(
        "--dashboard-state",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "hivemind_at_home_state.json"),
        help="Output JSON state path for the HiveMind@home dashboard.",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=8,
        help="Number of QLoRA/LoRA token training steps per specialist domain.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the live FastAPI HiveMind@home portal after generating artifacts.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="Port for the live FastAPI HiveMind@home portal when --serve is set.",
    )
    args = parser.parse_args()

    mining_summary = run_dataset_mining_and_ruche_campaign(
        vault_dir=args.vault_dir,
        report_json_path=args.mining_report,
        training_steps=args.training_steps,
    )

    # Verify community loader round-trip from La Ruche
    moe_model, loaded_manifest = load_community_moe_from_ruche(args.vault_dir)

    dashboard_state = export_hivemind_at_home_artifacts(
        output_html_path=args.dashboard_html,
        output_state_json_path=args.dashboard_state,
        gpu_campaigns_report_path=args.gpu_campaigns_report,
        mining_ruche_report_path=args.mining_report,
    )

    print(
        json.dumps(
            {
                "campaign_id": mining_summary["campaign_id"],
                "hardware_device": mining_summary["hardware_device"],
                "duration_ms": mining_summary["duration_ms"],
                "completed_shard_leases": mining_summary["completed_shard_leases"],
                "all_shards_dedup_clean": mining_summary["all_shards_dedup_clean"],
                "moe_forward_verified": mining_summary["moe_forward_verified"],
                "la_ruche_release_id": loaded_manifest["release_id"],
                "la_ruche_num_domains": loaded_manifest["num_domains"],
                "mean_holdout_perplexity_before": loaded_manifest[
                    "mean_holdout_perplexity_before"
                ],
                "mean_holdout_perplexity_after": loaded_manifest[
                    "mean_holdout_perplexity_after"
                ],
                "mean_perplexity_reduction_pct": loaded_manifest[
                    "mean_perplexity_reduction_pct"
                ],
                "combined_safetensors_sha256": loaded_manifest[
                    "combined_safetensors_sha256"
                ],
                "dashboard_html": args.dashboard_html,
                "dashboard_state_sha256": dashboard_state["dashboard_state_sha256"],
            },
            indent=2,
        )
    )

    if args.serve:
        import uvicorn

        app = create_hivemind_at_home_app(
            gpu_campaigns_report_path=args.gpu_campaigns_report,
            mining_ruche_report_path=args.mining_report,
        )
        uvicorn.run(app, host="0.0.0.0", port=args.port)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

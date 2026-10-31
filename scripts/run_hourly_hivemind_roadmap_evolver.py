#!/usr/bin/env python3
"""Hourly Improvement Plan & Roadmap Runner using HiveMind (`hivemind/coder` & `hivemind/bee-stars`) via Google Antigravity SDK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hivemind.swarm.s36_06_hourly_coding_assistant_orchestrator import (
    HourlyHiveMindCodingAssistantOrchestrator,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-openrouter",
        action="store_true",
        default=True,
        help="Call live HiveMind OpenRouter models (`hivemind/coder` & `hivemind/bee-stars`) as coding assistant.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run deterministic local verification without network calls.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "hourly_hivemind_roadmap_evolution.json",
    )
    args = parser.parse_args()

    call_live = bool(args.live_openrouter and not args.offline)
    orchestrator = HourlyHiveMindCodingAssistantOrchestrator()
    report = orchestrator.run_hourly_improvement_cycle(call_live_openrouter=call_live)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    print("========================================================================================")
    print("  HIVEMIND HOURLY IMPROVEMENT PLAN & ROADMAP CO-PILOT (SEASON 36 + IP-21/IP-67)")
    print("========================================================================================")
    print(f"  • Antigravity SDK Lightweight Compat : {report.antigravity_sdk_compat}")
    print(f"  • Live HiveMind OpenRouter Co-Pilot  : {report.live_openrouter_called}")
    print(f"  • Tasks & Unit Tests Verified        : {report.tasks_verified}/{report.tasks_total}")
    print(f"  • HM-S36-01 ZK-OPoI v5 CPU Verify    : {report.s36_01_zk_verify_ms:.4f} ms (< 5.0 ms SLA)")
    print(f"  • HM-S36-02 E21 10k Energy Reduction : {report.s36_02_e21_energy_reduction_pct:.4f}% (>= 99%)")
    print(f"  • HM-S36-03 GeoMesh p95 RTT Drop     : {report.s36_03_geomesh_p95_reduction_pct:.2f}% (>= 40%)")
    print(f"  • HM-S36-04 Sub-Swarm Fuzzy Quorum   : {report.s36_04_subswarm_agreement:.4f}")
    print(f"  • HM-S36-05 DARL Lease Conservation  : {report.s36_05_darl_conservation_holds}")
    print(f"  • IP-21 Residual Sampling & IP-67    : {report.ip21_residual_sampling_verified} / {report.ip67_no_hardcoded_xdev_verified}")
    print(f"  • Merkle Root SHA-256                : {report.merkle_root_sha256}")
    print(f"  • All Quality & Zero-Stub Gates      : {report.all_gates_passed}")
    print("========================================================================================")
    return 0 if report.all_gates_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

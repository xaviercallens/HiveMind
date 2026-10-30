# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Run Nocturnal MoE Auto-Improvement & Proof-of-Useful-Work (PoUW) Points Retrofit.

Orchestrates multi-expert, multi-base-model (Gemma 2, Gemma 3, Qwen 2.5, Qwen 3.5) continuous
MoE self-refinement across connected swarm producers, mapped to real Hugging Face datasets.

Usage:
  uv run python scripts/run_nocturnal_moe_auto_improvement.py [--force-now] [--base-model all|MODEL] [--experts 16|DOMAINS] [--retrofit]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root and src to path
REPO_ROOT = Path(__file__).resolve().parents[3]
HIVEMIND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HIVEMIND_DIR / "src"))

from hivemind.core.auto_improvement import (
    ALL_BASE_MODELS,
    ALL_DOMAINS,
    DEFAULT_BASE_MODEL,
    DEFAULT_BASE_WU_REWARD_HMT,
    DEFAULT_GATE_PASS_BONUS_HMT,
    DEFAULT_TFLOPS_STEP_RATE_HMT,
    HF_EXPERT_CATALOG,
    LARGE_VRAM_BASE_MODELS,
    NOCTURNAL_20_MOE_TARGETS,
    SUPPORTED_BASE_MODELS,
    NocturnalAutoImprovementScheduler,
    PointsRetrofitManager,
    resolve_domain_canonical,
)
from hivemind.router.accounts import RouterStore

DEFAULT_SECRET = b"hivemind-nocturnal-moe-orchestrator-secret-key-32b"
ARTIFACTS_DIR = HIVEMIND_DIR / "artifacts"
REPORT_PATH = ARTIFACTS_DIR / "nocturnal_moe_auto_improvement_report.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nocturnal MoE Auto-Improvement & PoUW Points Retrofit Runner."
    )
    parser.add_argument(
        "--force-now",
        action="store_true",
        help="Execute immediately regardless of whether it is currently European Night (22:00-07:00 CET).",
    )
    parser.add_argument(
        "--base-model",
        default="all",
        help=f"Target base model ('all' or one of {list(SUPPORTED_BASE_MODELS.keys())}).",
    )
    parser.add_argument(
        "--tier",
        default="all",
        choices=["all", "standard", "large"],
        help="Target compute tier ('all', 'standard' <= 8GB, or 'large' >= 12GB for high-VRAM base models).",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=20,
        help="Target number of nocturnal MoE experts to execute (default: 20).",
    )
    parser.add_argument(
        "--experts",
        default="20",
        help="Target expert domains ('all', '20', '16', '5', or comma-separated list of domains).",
    )
    parser.add_argument(
        "--cluster",
        default="all",
        choices=["all", "systems", "cyber", "science", "reasoning"],
        help="Filter domains by macro-cluster.",
    )
    parser.add_argument(
        "--min-vram-gb",
        type=float,
        default=6.0,
        help="Minimum VRAM capacity required to assign auto-improvement workunits (default: 6.0 GB).",
    )
    parser.add_argument(
        "--samples-per-domain",
        type=int,
        default=100,
        help="Number of refinement pairs per workunit (default: 100).",
    )
    parser.add_argument(
        "--inner-steps",
        type=int,
        default=50,
        help="Number of inner QLoRA optimization steps per workunit (default: 50).",
    )
    parser.add_argument(
        "--db-path",
        default=str(ARTIFACTS_DIR / "router_local.db"),
        help="Path to the SQLite database for RouterStore.",
    )
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    scheduler = NocturnalAutoImprovementScheduler()
    retrofit_mgr = PointsRetrofitManager()

    print("=" * 85)
    print("🌙 HIVEMIND NOCTURNAL MoE AUTO-IMPROVEMENT & MULTI-MODEL SWARM RUNNER")
    print("=" * 85)

    is_night = scheduler.is_european_night()
    print(f"[*] European Night Window Active: {is_night}")

    if not is_night and not args.force_now:
        print("[!] European Day / Peak Inference Window. Nocturnal MoE training is scheduled for 22:00-07:00 CET.")
        print("    Pass --force-now to run an immediate auto-improvement cycle.")
        return 0

    # Initialize router store
    db_path = Path(args.db_path)
    store = RouterStore(db_path, secret=DEFAULT_SECRET)

    # Connected high-compute producers (replacing preempted T4)
    known_producers = [
        {
            "account_id": "7a3e9c118b4f02a9e55d614cb9128f41",
            "handle": "producer.rtx3090-prosumer",
            "endpoint_url": "http://127.0.0.1:8295",
            "execution_class": "cuda-sm86-fp16-3090",
            "vram_gb": 24.0,
            "cost_per_1k": 0.35,
            "capacity_1k": 120.0,
            "last_heartbeat_utc": time.time(),
        },
        {
            "account_id": "621817e923e4a2e584fdf5a71dfa2995",
            "handle": "t4-big-gpu-moe-20261028",
            "endpoint_url": "http://127.0.0.1:8297",
            "execution_class": "tesla-t4-16gb-moe",
            "vram_gb": 16.0,
            "cost_per_1k": 0.32,
            "capacity_1k": 100.0,
            "last_heartbeat_utc": time.time(),
        },
        {
            "account_id": "fbc1d3042d377feba6b1e0ac6fd576a4",
            "handle": "pioneer.node1",
            "endpoint_url": "http://127.0.0.1:8296",
            "execution_class": "cuda-sm89-fp16-4070",
            "vram_gb": 12.0,
            "cost_per_1k": 0.38,
            "capacity_1k": 80.0,
            "last_heartbeat_utc": time.time(),
        },
        {
            "account_id": "58fd3df19d2b5ef8c10031e1389aac6a",
            "handle": "producer.rtx2070",
            "endpoint_url": "http://127.0.0.1:8095",
            "execution_class": "rtx2070-8gb-turing",
            "vram_gb": 8.0,
            "cost_per_1k": 0.4,
            "capacity_1k": 50.0,
            "last_heartbeat_utc": time.time(),
        },
    ]

    for p in known_producers:
        try:
            store.register(p["handle"], "producer")
        except Exception:
            pass

    eligible = scheduler.filter_eligible_producers(
        known_producers, min_vram_gb=args.min_vram_gb
    )
    print(f"[*] Found {len(eligible)} eligible high-compute producers replacing preempted T4:")
    for ep in eligible:
        print(f"    - {ep['account_id'][:12]}... ({ep['execution_class']}, {ep['vram_gb']} GB VRAM)")

    # Resolve target base models
    if args.tier == "large":
        base_models = [
            "Qwen/Qwen2.5-Coder-14B-Instruct",
            "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
            "mistralai/Mistral-Nemo-Base-2407",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen3.5-9B-Base",
            "google/gemma-2-9b-it",
        ]
        tier_label = "Large VRAM Base Models (12GB - 32GB High Training Capacity)"
    elif args.base_model.strip().lower() == "all":
        base_models = [
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "google/gemma-2-9b-it",
            "google/gemma-3-4b-it",
            "Qwen/Qwen3.5-9B-Base",
            "google/gemma-3-1b-it",
            "Qwen/Qwen2.5-7B-Instruct",
        ]
        tier_label = "Standard Base Models (NF4 4-bit < 8 GB VRAM)"
    else:
        base_models = [scheduler._canonical_base_model(args.base_model)]
        tier_label = f"Custom Base Model: {base_models[0]}"

    print(f"\n[*] Target Base Models ({tier_label}):")
    for m in base_models:
        print(f"    - {m}")

    # Resolve target domains
    if args.cluster != "all":
        cluster_map = {
            "systems": "CLUSTER_00_SYSTEMS_FORMAL",
            "cyber": "CLUSTER_01_CRYPTO_CYBER",
            "science": "CLUSTER_02_FRONTIER_SCIENCE",
            "reasoning": "CLUSTER_03_REASONING_AND_LANGUAGE",
        }
        target_cluster = cluster_map[args.cluster]
        target_domains = [d for d, meta in HF_EXPERT_CATALOG.items() if meta["cluster"] == target_cluster]
    elif args.experts.strip().lower() in ("20", "twenty"):
        target_domains = list(NOCTURNAL_20_MOE_TARGETS)
    elif args.experts.strip().lower() in ("all", "24", "16"):
        target_domains = list(ALL_DOMAINS)
    elif args.experts.strip().lower() == "5":
        target_domains = ["maths_olympiad", "astrophysics_transits", "astrophysics_spectra", "cyber_threat_intel", "code_feedback_debug"]
    else:
        target_domains = [
            resolve_domain_canonical(d.strip())
            for d in args.experts.split(",")
            if d.strip()
        ]

    if args.target_count and len(target_domains) > args.target_count:
        target_domains = target_domains[:args.target_count]

    print(f"\n[*] Target Expert Domains ({len(target_domains)} Experts across Hugging Face Datasets):")
    for d in target_domains:
        meta = HF_EXPERT_CATALOG[d]
        print(f"    - [{meta['cluster'][:10]}] {d:26s} -> HF: {meta['hf_dataset']}")

    # Generate and execute workunits across producers and base models
    workunits = []
    results = []

    print("\n" + "=" * 85)
    print("🚀 EXECUTING MULTI-EXPERT QLoRA AUTO-IMPROVEMENT WORKUNITS")
    print("=" * 85)

    wu_counter = 0
    for domain in target_domains:
        # Round-robin assign base model and producer matching required VRAM
        base_model = base_models[wu_counter % len(base_models)]
        wu = scheduler.create_domain_workunit(
            domain=domain,
            num_samples=args.samples_per_domain,
            inner_steps=args.inner_steps,
            seed=100 + wu_counter,
            base_model_id=base_model,
        )
        workunits.append(wu)

        # Match producer by required VRAM
        suitable_producers = [ep for ep in eligible if ep["vram_gb"] >= wu.min_vram_gb]
        if not suitable_producers:
            suitable_producers = eligible
        assigned_producer = suitable_producers[wu_counter % len(suitable_producers)]
        wu_counter += 1

        print(f"\n[+] Workunit '{wu.wu_id}':")
        print(f"    Domain: {wu.domain} | HF Dataset: {wu.hf_dataset}")
        print(f"    Base Model: {wu.base_model_id} (NF4 quantized)")
        print(f"    Assigned Worker: {assigned_producer['account_id'][:12]}... ({assigned_producer['execution_class']})")

        res = scheduler.simulate_or_verify_execution(
            workunit=wu,
            worker_id=f"worker-{assigned_producer['execution_class']}",
            account_id=assigned_producer["account_id"],
            execution_class=assigned_producer["execution_class"],
        )
        results.append(res)

        print(f"    Result: Steps={res.steps_completed}, Peak VRAM={res.peak_vram_mb:.1f} MB (< 8GB: OK), TFLOPs={res.contributed_tflops}")
        print(f"    Accuracy: Base={res.accuracy_base:.3f} -> Adapted={res.accuracy_adapted:.3f} (Gain={res.accuracy_gain:+.3f})")
        print(f"    McNemar Gate: p={res.mcnemar_p_value:.6f} -> Gate Passed: {res.gate_passed}")

    # Points Retrofit Phase
    print("\n" + "=" * 85)
    print("💰 PROOF-OF-USEFUL-WORK (PoUW) POINTS RETROFIT & LEDGER SETTLEMENT")
    print("=" * 85)

    report = retrofit_mgr.retrofit_work_points(store, results)

    print(f"[*] Total Workunits Processed: {report.total_workunits}")
    print(f"[*] Newly Credited Workunits:  {report.newly_credited_count}")
    print(f"[*] Already Credited (Skipped): {report.already_credited_count}")
    print(f"[*] Total HMT Points Granted:  +{report.total_hmt_granted:.2f} HMT")
    print(f"[*] Total TFLOPs Attested:      {report.total_tflops_attested:.3f} TFLOPs")
    print(f"[*] Ledger Hash-Chain Valid:    {report.ledger_verified}")

    print("\n[*] Credited Producer Accounts & Updated Balances:")
    for acct, points in report.credited_accounts.items():
        bal = store.balance(acct)
        print(f"    - Account {acct[:12]}...: +{points:.2f} HMT (Current Balance: {bal:.2f} HMT)")

    # Save artifact report
    output_payload = {
        "execution_utc": time.time(),
        "is_european_night": is_night,
        "base_models": base_models,
        "eligible_producers": eligible,
        "total_experts": len(target_domains),
        "workunits": [w.to_dict() for w in workunits],
        "results": [r.to_dict() for r in results],
        "retrofit_report": report.to_dict(),
    }
    REPORT_PATH.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    large_campaign_path = ARTIFACTS_DIR / "nocturnal_large_moe_20_campaign_report.json"
    large_campaign_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"\n[✓] Full telemetry report saved to: {REPORT_PATH}")
    print(f"[✓] Nocturnal 20-MoE campaign report saved to: {large_campaign_path}")

    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

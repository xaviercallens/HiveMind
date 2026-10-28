# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unified CLI Entrypoint for Project HiveMind (`hivemind`)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from rich.console import Console
from rich.table import Table

from hivemind.client.daemon import ZeroIntrusionDaemon
from hivemind.config import HiveMindConfig, detect_local_hardware
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
    measure_vram_footprint,
)

console = Console()


def cmd_hardware_check() -> int:
    """Display local RTX GPU telemetry, Win32 idle state, and VRAM allocation budgets."""
    cfg = HiveMindConfig.from_yaml()
    hw = detect_local_hardware()
    daemon = ZeroIntrusionDaemon(idle_threshold_seconds=cfg.idle_threshold_seconds)
    d_stat = daemon.poll()

    table = Table(title="Project HiveMind — Local RTX Hardware & Daemon Inspection")
    table.add_column("Property", style="cyan")
    table.add_column("Detected Value", style="bold green")
    for k, v in hw.items():
        table.add_row(k, str(v))
    table.add_row("win32_idle_seconds (GetLastInputInfo)", f"{d_stat.os_idle_seconds}s")
    table.add_row("active_steam_game_detected", str(d_stat.active_game_detected))
    console.print(table)

    b_table = Table(title="Configured Hardware Profiles & VRAM Allocation Budgets")
    b_table.add_column("Profile", style="yellow")
    b_table.add_column("Target Backbone", style="cyan")
    b_table.add_column("Compute Dtype")
    b_table.add_column("Frozen NF4 Base", justify="right")
    b_table.add_column("LoRA (r=64)", justify="right")
    b_table.add_column("8-bit AdamW", justify="right")
    b_table.add_column("KV & Activations", justify="right")
    b_table.add_column("Total Peak VRAM", justify="right", style="bold magenta")

    for pname, prof in cfg.profiles.items():
        vb = prof.vram_budget
        b_table.add_row(
            pname,
            prof.hf_model_id,
            prof.compute_dtype,
            f"{vb.frozen_base_nf4_gb:.2f} GB",
            f"{vb.lora_trainable_r64_gb:.2f} GB",
            f"{vb.paged_adamw_8bit_gb:.2f} GB",
            f"{vb.activations_kv_cache_gb:.2f} GB",
            f"{vb.total_peak_vram_gb:.2f} / {prof.target_vram_gb:.1f} GB",
        )
    console.print(b_table)
    return 0


def cmd_benchmark_vram(profile_name: str) -> int:
    """Run a live 4-bit NF4 + Rank-64 QLoRA + 8-bit Paged AdamW VRAM benchmark on the RTX 2070."""
    cfg = HiveMindConfig.from_yaml()
    prof = cfg.profiles.get(profile_name) or cfg.profiles["rtx2070_7b_worker"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    base_model, _, _ = build_local_qwen2_backbone(
        seed=42,
        vocab_size=8192,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        device=device,
        use_4bit_nf4_if_cuda=True,
    )
    vram_base = measure_vram_footprint(device)

    bundle = attach_qlora_adapters(
        base_model=base_model,
        rank_r=prof.lora_rank_r,
        lora_alpha=prof.lora_alpha,
        target_modules=prof.lora_target_modules,
        gradient_checkpointing=True,
        seed=42,
    )
    vram_lora = measure_vram_footprint(device)

    # Run 1 forward/backward step with 8-bit Paged AdamW
    from hivemind.core.diloco_inner import build_inner_optimizer

    trainable = [p for n, p in bundle.model.named_parameters() if p.requires_grad]
    opt = build_inner_optimizer(trainable, lr=2e-4, use_paged_8bit_adamw=True)
    x = torch.randint(0, 8192, (1, 256), device=device, dtype=torch.long)
    out = bundle.model(input_ids=x, labels=x)
    out.loss.backward()
    opt.step()
    vram_peak = measure_vram_footprint(device)

    console.print(
        json.dumps(
            {
                "profile": prof.name,
                "device": str(device),
                "is_4bit_nf4": bundle.is_4bit_nf4,
                "trainable_lora_params": bundle.trainable_params,
                "total_params": bundle.total_params,
                "vram_after_nf4_base_mb": vram_base["allocated_mb"],
                "vram_after_lora_r64_mb": vram_lora["allocated_mb"],
                "vram_peak_step_mb": vram_peak["peak_allocated_mb"],
                "free_vram_gb": vram_peak["free_vram_gb"],
                "spec_budget_breakdown_gb": {
                    "frozen_base_nf4_gb": prof.vram_budget.frozen_base_nf4_gb,
                    "lora_trainable_r64_gb": prof.vram_budget.lora_trainable_r64_gb,
                    "paged_adamw_8bit_gb": prof.vram_budget.paged_adamw_8bit_gb,
                    "activations_kv_cache_gb": prof.vram_budget.activations_kv_cache_gb,
                    "cuda_context_overhead_gb": prof.vram_budget.cuda_context_overhead_gb,
                    "total_peak_vram_gb": prof.vram_budget.total_peak_vram_gb,
                },
            },
            indent=2,
        )
    )
    return 0


def cmd_serve_coordinator(host: str, port: int, public_url: str | None = None, insecure_dev_identities: bool = False) -> int:
    """Start the FastAPI Modified BOINC Coordinator server.

    Reachable by others (``host`` not loopback) means nodes must prove their SteamID64 by signing in with Steam, which needs ``--public-url`` (the https URL
    Steam sends the browser back to). Without it a non-loopback bind is refused: the request body's ``steam_id64`` would otherwise be taken on trust.
    """
    import sys

    import uvicorn
    from hivemind.boinc.coordinator_api import BOINCCoordinatorService, create_coordinator_fastapi_app
    from hivemind.client.steam_identity import SteamIdentityGate

    identity = None
    if public_url:
        identity = SteamIdentityGate(return_to=public_url.rstrip("/") + "/identity/steam/callback")
    elif host not in ("127.0.0.1", "localhost", "::1") and not insecure_dev_identities:
        print(
            f"refusing to listen on {host}: nodes' SteamID64 would be unverified. Give --public-url https://... to require Sign in with Steam, "
            "or --insecure-dev-identities if this really is a throwaway test network.",
            file=sys.stderr,
        )
        return 2

    cfg = HiveMindConfig.from_yaml()
    base_model, base_sha, _ = build_local_qwen2_backbone(
        seed=42,
        vocab_size=2048,
        hidden_size=128,
        intermediate_size=384,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(base_model, rank_r=16, lora_alpha=32, gradient_checkpointing=False)
    init_lora = extract_lora_state_dict(bundle.model)
    coord = BOINCCoordinatorService(
        config=cfg,
        initial_lora_state=init_lora,
        base_model_sha256=base_sha,
        vocab_size=2048,
    )
    app = create_coordinator_fastapi_app(coord, identity=identity)
    uvicorn.run(app, host=host, port=port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hivemind",
        description="Project HiveMind: Decentralized MoE Training & Inference via Steam, BOINC, and DiLoCo",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hardware-check", help="Inspect local RTX GPU, Win32 daemon, and VRAM budgets")

    p_sim = sub.add_parser("simulate-e2e", help="Run full Phase A -> B -> C -> Phase 2 pipeline locally")
    p_sim.add_argument("--profile", default="rtx2070_moe_e2e", help="Hardware profile name")
    p_sim.add_argument("--use-hf-hub", action="store_true", help="Download weights from HuggingFace Hub")
    p_sim.add_argument("--inner-steps", type=int, default=20, help="Inner AdamW steps per branch")
    p_sim.add_argument("--spot-check-interval", type=int, default=10, help="Micro-checkpoint interval s")

    p_bench = sub.add_parser("benchmark-vram", help="Benchmark 4-bit NF4 + QLoRA r=64 + PagedAdamW8bit VRAM")
    p_bench.add_argument("--profile", default="rtx2070_7b_worker", help="Hardware profile to benchmark")

    p_serve = sub.add_parser("serve-coordinator", help="Launch Modified BOINC Coordinator FastAPI Server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--public-url", default=None, help="https URL of this coordinator; turns on Sign in with Steam (required for non-loopback --host)")
    p_serve.add_argument("--insecure-dev-identities", action="store_true", help="allow a non-loopback --host without identity verification (throwaway networks only)")

    p_poc = sub.add_parser("run-poc", help="Run real Hugging Face Qwen2.5-0.5B PoC on local RTX 2070")
    p_poc.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B", help="Hugging Face model ID")
    p_poc.add_argument(
        "--dataset-id",
        default="sahil2801/CodeAlpaca-20k",
        help="Hugging Face dataset ID for decentralized subset training (e.g., sahil2801/CodeAlpaca-20k or qiaojin/PubMedQA)",
    )
    p_poc.add_argument("--subset-size", type=int, default=24, help="Number of HF rows partitioned across worker shards")
    p_poc.add_argument("--lora-rank", type=int, default=16, help="LoRA rank r")
    p_poc.add_argument("--inner-steps", type=int, default=12, help="DiLoCo inner steps per branch")

    p_swarm = sub.add_parser(
        "run-seti-swarm",
        help="Run HiveMind v2 ('SETI-Swarm') Decentralized HF Domain Training + PoUW $HIVE Ledger + Shared Inference",
    )
    p_swarm.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B", help="Hugging Face model ID")
    p_swarm.add_argument("--dataset-id", default="sahil2801/CodeAlpaca-20k", help="Hugging Face dataset ID")
    p_swarm.add_argument("--domain-name", default="Python_Systems_Code", help="Domain expert name")
    p_swarm.add_argument("--lora-rank", type=int, default=16, help="LoRA rank r")
    p_swarm.add_argument("--inner-steps", type=int, default=12, help="DiLoCo inner steps per worker")

    p_exp3 = sub.add_parser(
        "run-3-experiments",
        help="Execute 3 sequential SETI-Swarm experiments on real Hugging Face datasets (CodeAlpaca-20k, PubMedQA, GSM8K) + Dynamic MoE Hot-Plug",
    )
    p_exp3.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B", help="Hugging Face model ID")
    p_exp3.add_argument("--lora-rank", type=int, default=8, help="LoRA rank r")
    p_exp3.add_argument("--inner-steps", type=int, default=12, help="DiLoCo inner steps per worker")

    p_astro = sub.add_parser(
        "run-astro-experiments",
        help="Run 4 sequential single-process astrophysics QLoRA swarm demos (HF rows when reachable, curated fallback text otherwise) + 4-expert MoE hot-plug; reports training-row perplexity, not classification accuracy",
    )
    p_astro.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B", help="Hugging Face model ID")
    p_astro.add_argument("--lora-rank", type=int, default=8, help="LoRA rank r")
    p_astro.add_argument("--inner-steps", type=int, default=12, help="DiLoCo inner steps per worker")

    p_game = sub.add_parser(
        "play-hack-game",
        help="Play HiveMind: Neural Breach (Free Steam Edition) Hugging Face Hack-Miner Game & build Steam package",
    )
    p_game.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B", help="Hugging Face model ID")
    p_game.add_argument("--lora-rank", type=int, default=8, help="LoRA rank r")
    p_game.add_argument("--inner-steps", type=int, default=8, help="DiLoCo inner steps per gamer")
    p_game.add_argument(
        "--season",
        default="all",
        choices=[
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "s1-s3",
            "s1-s4",
            "s1-s5",
            "s1-s6",
            "all",
        ],
        help=(
            "Campaign season to run: '1'/'2' (6 missions), '3'..'6' (frontier waves), "
            "'s1-s3'..'s1-s6' (cumulative), or 'all' (Season 1+2 + 6-Expert MoE)"
        ),
    )

    p_hud = sub.add_parser(
        "serve-hack-hud",
        help="Launch the interactive Cyberpunk Free Steam Edition Web HUD & Leaderboard server",
    )
    p_hud.add_argument("--host", default="127.0.0.1")
    p_hud.add_argument("--port", type=int, default=8090)
    p_hud.add_argument("--allow-network", action="store_true", help="listen on a non-loopback --host (the HUD has no login and can start GPU work: trusted LAN only)")

    p_probe = sub.add_parser(
        "hack-probe",
        help="Run an interactive Hugging Face Shard Scan & 6-Expert Sparse MoE Bounty Probe for a gamer",
    )
    p_probe.add_argument("--gamer", default="CyberValkyrie_FR", help="Steam Gamer Tag")
    p_probe.add_argument("--dataset-id", default="sahil2801/CodeAlpaca-20k", help="HF Dataset Node to scan")
    p_probe.add_argument(
        "--prompt",
        default="Write a Python function to calculate cellular ATP and HIVE minted across 3 blocks.",
        help="Custom Bounty Challenge prompt to route across the 6-Expert Sparse MoE",
    )

    p_deploy = sub.add_parser(
        "deploy-steam-tool",
        help="Run the 6-Stage Intense RTX GPU Stress Suite and deploy v1.0.0 Free Steam Tool (AppID: 3133700)",
    )
    p_deploy.add_argument("--vram-cycles", type=int, default=3, help="Number of NF4+QLoRA+Preemption torture cycles")
    p_deploy.add_argument("--udp-packets", type=int, default=100, help="Number of UDP 0-RTT hidden-state packets")
    p_deploy.add_argument("--hotplug-experts", type=int, default=12, help="Number of Sparse MoE experts to hot-plug")

    p_sec = sub.add_parser(
        "scan-pr-security",
        help="Run 5-Layer PR & Repository Security Scanner (Anti-Code-Boubou, Anti-Payment-Extortion, AST & Secret Audit)",
    )
    p_sec.add_argument("--root", default=".", help="Repository root directory to scan")
    p_sec.add_argument("--pr-json", default=None, help="Optional path to PR metadata/files JSON payload to audit")
    p_sec.add_argument("--output-json", default=None, help="Optional path to write JSON audit report")

    p_hard = sub.add_parser(
        "audit-swarm-hardening",
        help="Execute Phase S-2 GPU Training & Content-Addressed Data Store Runtime Hardening Self-Test",
    )
    p_hard.add_argument(
        "--output-json",
        default="artifacts/swarm_gpu_and_datastore_hardening_report.json",
        help="Path to write JSON hardening audit report",
    )

    p_boinc_astro = sub.add_parser(
        "run-boinc-astro",
        help="Run a full Modified BOINC Astrophysics Campaign on the local GPU across 4 Hugging Face datasets",
    )
    p_boinc_astro.add_argument("--inner-steps", type=int, default=16, help="DiLoCo inner steps H per BOINC workunit")
    p_boinc_astro.add_argument("--spot-check-interval", type=int, default=8, help="Micro-checkpoint interval s")
    p_boinc_astro.add_argument("--lora-rank", type=int, default=16, help="QLoRA rank r")
    p_boinc_astro.add_argument(
        "--output-json",
        default="artifacts/boinc_local_astrophysics_gpu_report.json",
        help="Path to write BOINC Astrophysics campaign JSON report",
    )

    p_astro_24h = sub.add_parser(
        "run-astro-24h",
        help="Run a 24-Cycle Intensive GPU Astrophysics & Cosmology Campaign and render PNG dashboards",
    )
    p_astro_24h.add_argument("--cycles", type=int, default=24, help="Number of BOINC outer cycles (default: 24)")
    p_astro_24h.add_argument("--inner-steps", type=int, default=12, help="DiLoCo inner steps H per worker per cycle")
    p_astro_24h.add_argument("--spot-check-interval", type=int, default=6, help="Micro-checkpoint interval s")
    p_astro_24h.add_argument("--lora-rank", type=int, default=16, help="QLoRA rank r")
    p_astro_24h.add_argument(
        "--output-json",
        default="artifacts/astrophysics_24h_gpu_campaign_report.json",
        help="Path to write 24-cycle Astrophysics & Cosmology JSON report",
    )
    p_astro_24h.add_argument(
        "--dashboard-png",
        default="docs/images/astrophysics_gpu_24h_dashboard.png",
        help="Path to write 6-panel Astrophysics GPU dashboard PNG",
    )
    p_astro_24h.add_argument(
        "--cosmology-png",
        default="docs/images/cosmology_deep_field_discovery.png",
        help="Path to write 4-panel Cosmology & JWST discovery PNG",
    )

    p_lab = sub.add_parser(
        "swarm-lab",
        help="HiveMind Swarm Lab: GPU-free qualitative toy of quorum policy x stragglers x non-IID data (simulate | compare | presets | play)",
    )
    p_lab.add_argument("lab_args", nargs=argparse.REMAINDER, help="arguments of the lab CLI, e.g. `simulate --preset churn` (use `-h` after the lab command for help)")

    args = parser.parse_args(argv)
    if args.command == "swarm-lab":
        from hivemind.lab.swarm_lab import main as swarm_lab_main

        return swarm_lab_main(args.lab_args)
    if args.command == "hardware-check":
        return cmd_hardware_check()
    if args.command == "simulate-e2e":
        from hivemind.cli.simulate_e2e import run_end_to_end_simulation

        run_end_to_end_simulation(
            profile_name=args.profile,
            use_hf_hub=args.use_hf_hub,
            inner_steps_H=args.inner_steps,
            spot_check_interval_s=args.spot_check_interval,
        )
        return 0
    if args.command == "benchmark-vram":
        return cmd_benchmark_vram(profile_name=args.profile)
    if args.command == "serve-coordinator":
        return cmd_serve_coordinator(host=args.host, port=args.port, public_url=args.public_url, insecure_dev_identities=args.insecure_dev_identities)
    if args.command == "run-poc":
        from hivemind.cli.poc_rtx2070 import run_huggingface_rtx2070_poc

        run_huggingface_rtx2070_poc(
            model_id=args.model_id,
            hf_dataset_id=args.dataset_id,
            hf_subset_size=args.subset_size,
            lora_rank_r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            inner_steps_H=args.inner_steps,
            spot_check_interval_s=max(1, args.inner_steps // 2),
        )
        return 0
    if args.command == "run-seti-swarm":
        from hivemind.cli.demo_hf_domain_swarm import run_decentralized_hf_domain_demo

        run_decentralized_hf_domain_demo(
            model_id=args.model_id,
            dataset_id=args.dataset_id,
            domain_name=args.domain_name,
            lora_rank=args.lora_rank,
            inner_steps=args.inner_steps,
        )
        return 0
    if args.command == "run-3-experiments":
        from hivemind.cli.demo_hf_domain_swarm import run_three_hf_swarm_experiments

        run_three_hf_swarm_experiments(
            model_id=args.model_id,
            lora_rank=args.lora_rank,
            inner_steps=args.inner_steps,
        )
        return 0
    if args.command == "run-astro-experiments":
        from hivemind.cli.demo_hf_domain_swarm import run_astrophysics_hf_swarm_experiments

        run_astrophysics_hf_swarm_experiments(
            model_id=args.model_id,
            lora_rank=args.lora_rank,
            inner_steps=args.inner_steps,
        )
        return 0
    if args.command == "play-hack-game":
        from hivemind.game.hf_hack_miner import run_hf_hack_mining_game

        run_hf_hack_mining_game(
            model_id=args.model_id,
            lora_rank=args.lora_rank,
            inner_steps=args.inner_steps,
            build_steam_package=True,
            season=args.season,
        )
        return 0
    if args.command == "serve-hack-hud":
        import sys

        import uvicorn
        from hivemind.game.steam_edition import create_hack_miner_fastapi_app

        if args.host not in ("127.0.0.1", "localhost", "::1") and not args.allow_network:
            # The HUD is a single-player UI with no login whose POST routes run GPU diagnostics, LoRA forging and Hugging Face downloads.
            print(f"refusing to listen on {args.host}: the HUD has no login. Use --allow-network only on a network you trust.", file=sys.stderr)
            return 2

        app = create_hack_miner_fastapi_app()
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    if args.command == "hack-probe":
        from hivemind.game.hf_hack_miner import (
            execute_interactive_bounty_probe,
            execute_interactive_hf_shard_scan,
        )

        scan_res = execute_interactive_hf_shard_scan(
            gamer_tag=args.gamer,
            dataset_id=args.dataset_id,
        )
        bounty_res = execute_interactive_bounty_probe(
            gamer_tag=args.gamer,
            prompt=args.prompt,
        )
        console.print(json.dumps({"shard_scan": scan_res, "bounty_probe": bounty_res}, indent=2))
        return 0
    if args.command == "deploy-steam-tool":
        from hivemind.game.rtx_steam_tool_deployer import (
            deploy_free_steam_rtx_tool_v1,
            run_intense_rtx_stress_suite,
        )

        stress_rep = run_intense_rtx_stress_suite(
            vram_cycles=args.vram_cycles,
            udp_burst_packets=args.udp_packets,
            num_hotplug_experts=args.hotplug_experts,
        )
        deploy_info = deploy_free_steam_rtx_tool_v1(stress_report=stress_rep)
        console.print(json.dumps(deploy_info, indent=2))
        return 0
    if args.command == "scan-pr-security":
        from hivemind.security.pr_security_scanner import run_cli_security_scan

        return run_cli_security_scan(
            root_path=args.root,
            pr_json_path=args.pr_json,
            output_json_path=args.output_json,
        )
    if args.command == "audit-swarm-hardening":
        from hivemind.security.swarm_runtime_hardening import run_swarm_hardening_self_test

        report = run_swarm_hardening_self_test(output_json_path=args.output_json)
        console.print(json.dumps(report, indent=2))
        return 0 if report.get("overall_passed") else 1
    if args.command == "run-astro-24h":
        from hivemind.boinc.astrophysics_24h_campaign import run_24h_astrophysics_gpu_campaign

        report = run_24h_astrophysics_gpu_campaign(
            num_cycles=args.cycles,
            inner_steps_per_cycle=args.inner_steps,
            spot_check_interval_s=args.spot_check_interval,
            lora_rank_r=args.lora_rank,
            output_json_path=args.output_json,
            dashboard_png_path=args.dashboard_png,
            cosmology_png_path=args.cosmology_png,
        )
        return 0 if report.get("overall_passed") else 1
    if args.command == "run-boinc-astro":
        from hivemind.boinc.local_astrophysics_boinc import run_local_boinc_astrophysics_campaign

        report = run_local_boinc_astrophysics_campaign(
            inner_steps_H=args.inner_steps,
            spot_check_interval_s=args.spot_check_interval,
            lora_rank_r=args.lora_rank,
            output_json_path=args.output_json,
        )
        return 0 if report.get("overall_passed") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""End-to-End Local RTX 2070 Pipeline Runner for Project HiveMind.

Executes all phases of the Technical Specification on the local GPU:
  1. Hardware & VRAM Budget Verification (RTX 2070 8GB + 12GB Reference)
  2. Phase A: 8-Domain Branch-Train-Mix (BTM) Initialization (4-bit NF4 + Rank-64 LoRA)
  3. Phase B (Inner): BOINC Signed Parquet Shards + DiLoCo-QLoRA + Top-k/FP8 Compression
  4. Phase B (Validator): Deterministic Micro-Checkpoint Spot-Check (eps = 1e-4)
  5. Phase B (Outer): Multi-Krum Byzantine Defense + Nesterov Outer Step (mu=0.9, eta=0.7)
  6. Phase C: 8x Sparse MoE Upcycling (Attention Averaging + FFN Expansion + Top-2 Router)
  7. Phase 2: Regional DHT Swarm Ring + 5-Token Speculative Decoding
"""

from __future__ import annotations

import copy
import gc
import json
import time
from typing import Any

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hivemind.boinc.coordinator_api import BOINCCoordinatorService
from hivemind.boinc.workunit import BOINCWorkunit, verify_and_load_parquet_shard
from hivemind.client.daemon import ZeroIntrusionDaemon
from hivemind.client.steamworks import SteamworksSDKBridge
from hivemind.client.worker_node import HiveMindWorkerNode
from hivemind.config import (
    ARTIFACTS_ROOT,
    HiveMindConfig,
    configure_deterministic_runtime,
    detect_local_hardware,
)
from hivemind.core.btm_upcycle import train_moe_router_gating, upcycle_branches_to_sparse_moe
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
    load_hf_qlora_model,
    load_lora_state_dict,
    measure_vram_footprint,
)
from hivemind.swarm.inference_ring import RegionalDHTSwarmRing, SwarmPeerNode

console = Console()


def estimate_workunit_tflops(
    workunit: BOINCWorkunit,
    total_params: int = 0,
    micro_batch_size: int = 2,
) -> float:
    """Estimate training TFLOPs executed for a BOINC workunit from its step and token counts (IP-178)."""
    if workunit is None or not hasattr(workunit, "inner_steps_H") or not hasattr(workunit, "seq_len"):
        raise ValueError("workunit must provide inner_steps_H and seq_len")
    if (
        isinstance(workunit.inner_steps_H, bool)
        or not isinstance(workunit.inner_steps_H, int)
        or isinstance(workunit.seq_len, bool)
        or not isinstance(workunit.seq_len, int)
    ):
        raise ValueError("workunit inner_steps_H and seq_len must be positive integers")
    if isinstance(total_params, bool) or not isinstance(total_params, int) or total_params < 0:
        raise ValueError("total_params must be a non-negative integer")
    if isinstance(micro_batch_size, bool) or not isinstance(micro_batch_size, int) or micro_batch_size < 1:
        raise ValueError("micro_batch_size must be a positive integer")
    steps = int(workunit.inner_steps_H)
    seq_len = int(workunit.seq_len)
    if steps < 1 or seq_len < 1:
        raise ValueError("workunit inner_steps_H and seq_len must be positive integers")
    batch_tokens = int(micro_batch_size) * seq_len
    effective_params = max(int(total_params), 100_000_000)
    flops = 6.0 * float(effective_params) * float(steps) * float(batch_tokens)
    return round(flops / 1e12, 6)


def run_end_to_end_simulation(
    profile_name: str = "rtx2070_moe_e2e",
    use_hf_hub: bool = False,
    inner_steps_H: int = 20,
    spot_check_interval_s: int = 10,
) -> dict[str, Any]:
    """Run the complete Project HiveMind pipeline on the local RTX GPU and save telemetry."""
    if not isinstance(profile_name, str) or not profile_name.strip():
        raise ValueError("profile_name must be a non-empty string")
    if not isinstance(use_hf_hub, bool):
        raise ValueError("use_hf_hub must be a boolean")
    if isinstance(inner_steps_H, bool) or not isinstance(inner_steps_H, int) or inner_steps_H < 1:
        raise ValueError("inner_steps_H must be a positive integer")
    if isinstance(spot_check_interval_s, bool) or not isinstance(spot_check_interval_s, int) or spot_check_interval_s < 1:
        raise ValueError("spot_check_interval_s must be a positive integer")
    t_pipeline_start = time.perf_counter()
    cfg = HiveMindConfig.from_yaml()
    profile = cfg.profiles.get(profile_name) or next(iter(cfg.profiles.values()))

    hw_info = detect_local_hardware()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    console.print(
        Panel.fit(
            f"[bold cyan]Project HiveMind — End-to-End Distributed MoE Pipeline[/bold cyan]\n"
            f"Active Profile: [yellow]{profile.name}[/yellow] ({profile.description})\n"
            f"Execution Device: [green]{hw_info['gpu_name']}[/green] | "
            f"PyTorch: [magenta]{hw_info['torch_version']}[/magenta] | "
            f"Precision: [bold]{hw_info['recommended_dtype']}[/bold]",
            border_style="cyan",
        )
    )

    # -------------------------------------------------------------------------
    # STEP 1: Build 4-bit NF4 Quantized Base Model + Rank-64 QLoRA Adapters
    # -------------------------------------------------------------------------
    configure_deterministic_runtime(42)
    if use_hf_hub:
        bundle = load_hf_qlora_model(profile=profile, device=device, seed=42)
        vocab_size = bundle.model.config.vocab_size
    else:
        base_model, _, _ = build_local_qwen2_backbone(
            seed=42,
            vocab_size=2048,
            hidden_size=256,
            intermediate_size=688,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            device=device,
            use_4bit_nf4_if_cuda=True,
        )
        vocab_size = 2048
        bundle = attach_qlora_adapters(
            base_model=base_model,
            rank_r=profile.lora_rank_r,
            lora_alpha=profile.lora_alpha,
            target_modules=profile.lora_target_modules,
            gradient_checkpointing=False,
            seed=42,
        )

    initial_lora_state = extract_lora_state_dict(bundle.model)
    vram_after_model = measure_vram_footprint(device)

    # -------------------------------------------------------------------------
    # STEP 2: Initialize BOINC Coordinator & Phase A (8 Domain Branches)
    # -------------------------------------------------------------------------
    coordinator = BOINCCoordinatorService(
        config=cfg,
        initial_lora_state=initial_lora_state,
        base_model_sha256=bundle.base_model_sha256,
        vocab_size=vocab_size,
    )
    steamworks = SteamworksSDKBridge()
    daemon = ZeroIntrusionDaemon(idle_threshold_seconds=cfg.idle_threshold_seconds, enforce_os_idle_hook=False)
    daemon_status = daemon.poll()

    # -------------------------------------------------------------------------
    # STEP 3: Phase B — DiLoCo-QLoRA Inner Loop + Spot-Check Validation + Multi-Krum
    # -------------------------------------------------------------------------
    branch_metrics: dict[str, dict[str, Any]] = {}
    domain_token_batches: dict[str, torch.Tensor] = {}

    for b_idx, domain in enumerate(cfg.expert_domains):
        steam_id = f"7656119800000{b_idx + 1:04d}"
        worker = HiveMindWorkerNode(
            worker_id=f"worker-{domain.lower()}-01",
            steam_id64=steam_id,
            persona_name=f"SteamGamer_{domain}",
            model=bundle.model,
            steamworks=steamworks,
            daemon=daemon,
            account_spend_usd=45.0,
        )
        wu, global_lora = coordinator.assign_workunit(
            worker_id=worker.worker_id,
            steam_id64=steam_id,
            branch_name=domain,
            inner_steps_H=inner_steps_H,
            spot_check_interval_s=spot_check_interval_s,
            num_sequences=16,
            seq_len=64,
            rng_seed=1000 + b_idx * 31,
        )
        domain_token_batches[domain] = verify_and_load_parquet_shard(
            shard_path=wu.data_shard_parquet,
            expected_sha256=wu.shard_sha256,
            expected_hmac_signature=wu.shard_hmac_signature,
            rng_seed=wu.rng_seed,
        )

        # Execute H local AdamW steps on the worker GPU
        inner_res = worker.execute_workunit(
            workunit=wu,
            global_lora_state=global_lora,
            lr=1e-3,
            weight_decay=0.01,
            micro_batch_size=2,
            topk_ratio=cfg.compression_topk_ratio,
        )

        # Trusted BOINC Validator spot-checks the first s-step segment (e.g., 50% or 10% overhead)
        verdict = coordinator.validate_and_record_submission(
            validator_model=bundle.model,
            workunit=wu,
            submission=inner_res,
            steam_id64=steam_id,
            lr=1e-3,
            weight_decay=0.01,
            micro_batch_size=2,
        )
        tflops_wu = estimate_workunit_tflops(
            workunit=wu,
            total_params=bundle.total_params,
            micro_batch_size=2,
        )
        steam_entry = steamworks.record_validated_workunit(
            steam_id64=steam_id,
            persona_name=worker.persona_name,
            wu_id=wu.wu_id,
            tflops_contributed=tflops_wu,
            credits_minted=10.0,
        )

        # Simulate peer workers in the branch quorum (3 additional honest peers + 1 Byzantine poisoner)
        honest_delta = coordinator.pending_branch_deltas[domain][worker.worker_id]
        for peer_idx in range(2, 5):
            peer_id = f"worker-{domain.lower()}-0{peer_idx}"
            noisy_honest: dict[str, torch.Tensor] = {}
            gen = torch.Generator(device="cpu").manual_seed(9000 + b_idx * 10 + peer_idx)
            for k, v in honest_delta.items():
                noise = torch.randn(v.shape, generator=gen, dtype=v.dtype) * (v.std().item() * 0.03 + 1e-7)
                noisy_honest[k] = v + noise
            coordinator.inject_raw_delta_for_aggregation(domain, peer_id, noisy_honest)

        # Inject 1 Byzantine gradient-poisoning attacker per branch
        byz_id = f"byzantine-attacker-{domain.lower()}"
        byz_delta: dict[str, torch.Tensor] = {}
        gen_byz = torch.Generator(device="cpu").manual_seed(77777 + b_idx)
        for k, v in honest_delta.items():
            byz_delta[k] = -8.0 * v + torch.randn(v.shape, generator=gen_byz, dtype=v.dtype) * 0.25
        coordinator.inject_raw_delta_for_aggregation(domain, byz_id, byz_delta)

        # Execute Multi-Krum + 2-sigma Cosine Filter + Nesterov Outer Update
        outer_res = coordinator.step_branch_outer_loop(domain)

        branch_metrics[domain] = {
            "wu_id": wu.wu_id,
            "initial_loss": inner_res.initial_loss,
            "final_loss": inner_res.final_loss,
            "compression_ratio": inner_res.compressed_delta.compression_ratio,
            "spot_check_passed": verdict.passed,
            "spot_check_max_diff": verdict.max_abs_divergence,
            "spot_check_overhead_pct": verdict.verification_overhead_pct,
            "multikrum_accepted": outer_res.filter_report.accepted_workers,
            "multikrum_rejected": outer_res.filter_report.rejected_workers,
            "outer_delta_norm": outer_res.aggregated_delta_norm,
            "verified_tflops": steam_entry.verified_tflops,
        }

    # Also test Spot-Check Validator rejecting a tampered worker submission
    tampered_worker_res = copy.deepcopy(inner_res)
    first_ckpt_step = min(s for s in tampered_worker_res.micro_checkpoints if s > 0)
    first_key = next(iter(tampered_worker_res.micro_checkpoints[first_ckpt_step].state_snapshot))
    tampered_worker_res.micro_checkpoints[first_ckpt_step].state_snapshot[first_key] += 0.05
    tampered_verdict = coordinator.validator.verify_submission(
        validator_model=bundle.model,
        workunit=wu,
        submission=tampered_worker_res,
        token_sequences=domain_token_batches[cfg.expert_domains[-1]],
        steam_id64="76561198999999999",
        lr=1e-3,
        weight_decay=0.01,
        micro_batch_size=2,
    )

    # -------------------------------------------------------------------------
    # STEP 4: Phase C — Branch-Train-Mix (BTM) 8-Expert Sparse MoE Upcycling
    # -------------------------------------------------------------------------
    bundle_summary = {
        "is_4bit_nf4": bundle.is_4bit_nf4,
        "trainable_params": bundle.trainable_params,
        "total_params": bundle.total_params,
        "lora_size_mb": bundle.lora_size_mb,
    }
    del tampered_worker_res, inner_res
    gc.collect()

    converged_branch_states = {
        dom: state.lora_state for dom, state in coordinator.branches.items()
    }
    moe_dtype = bundle.compute_dtype if device.type == "cuda" else torch.float32
    moe_model, upcycle_report = upcycle_branches_to_sparse_moe(
        peft_or_base_model=bundle.model,
        branch_lora_states=converged_branch_states,
        lora_rank_r=profile.lora_rank_r,
        lora_alpha=profile.lora_alpha,
        top_k=cfg.router_top_k,
        router_aux_loss_coef=cfg.router_aux_loss_coef,
        target_device=device,
        target_dtype=moe_dtype,
    )

    # Release Phase B training buffers now that the 8 experts are merged into the Sparse MoE
    del converged_branch_states, coordinator, bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    router_init_loss, router_final_loss, router_util = train_moe_router_gating(
        moe_model=moe_model,
        domain_token_batches=domain_token_batches,
        steps=24,
        lr=5e-3,
    )
    upcycle_report.router_initial_loss = router_init_loss
    upcycle_report.router_final_loss = router_final_loss
    upcycle_report.expert_utilization_distribution = router_util

    # -------------------------------------------------------------------------
    # STEP 5: Phase 2 — Regional DHT Swarm + 5-Token Speculative Decoding
    # -------------------------------------------------------------------------
    ring = RegionalDHTSwarmRing(region="EU-West-Paris", max_ping_ms=25.0)
    for idx, dom in enumerate(cfg.expert_domains):
        ring.register_peer(
            SwarmPeerNode(
                peer_id=f"peer-eu-paris-{idx + 1:02d}",
                steam_id64=f"7656119800000{idx + 1:04d}",
                region="EU-West-Paris",
                ping_ms=11.5 + idx * 1.4,
                vram_gb=8.0 if idx % 2 == 0 else 12.0,
                hosted_expert_domain=dom,
                layer_range=(0, 4),
            )
        )

    # Zero-copy Top-1 local draft mode verified against Top-2 8x Sparse MoE swarm
    gc.collect()
    prompt_ids = domain_token_batches["Code"][0, :16]
    spec_result = ring.run_speculative_inference(
        moe_target_model=moe_model,
        draft_model=moe_model,
        prompt_ids=prompt_ids,
        max_new_tokens=20,
        draft_gamma=5,
        simulated_network_rtt_ms=16.0,
    )

    vram_peak = measure_vram_footprint(device)
    total_elapsed = round(time.perf_counter() - t_pipeline_start, 2)

    # Render Rich Summary Tables
    _render_summary_tables(
        hw_info=hw_info,
        bundle_info=bundle_summary,
        daemon_status=daemon_status,
        branch_metrics=branch_metrics,
        tampered_verdict=tampered_verdict,
        upcycle_report=upcycle_report,
        spec_result=spec_result,
        vram_peak=vram_peak,
        total_elapsed=total_elapsed,
    )

    summary_payload = {
        "hardware": hw_info,
        "vram_after_model": vram_after_model,
        "vram_peak": vram_peak,
        "daemon_telemetry": {
            "os_idle_seconds": daemon_status.os_idle_seconds,
            "should_yield_vram": daemon_status.should_yield_vram,
            "reason": daemon_status.reason,
        },
        "phase_ab_branches": branch_metrics,
        "spot_check_tamper_test": {
            "passed": tampered_verdict.passed,
            "flagged_steam_account": tampered_verdict.steam_account_flagged,
            "max_abs_divergence": tampered_verdict.max_abs_divergence,
            "reason": tampered_verdict.reason,
        },
        "phase_c_moe_upcycling": {
            "num_experts": upcycle_report.num_experts,
            "top_k": upcycle_report.top_k,
            "attention_layers_merged": upcycle_report.attention_layers_merged,
            "ffn_expert_modules_created": upcycle_report.ffn_expert_modules_created,
            "total_moe_parameters": upcycle_report.total_moe_parameters,
            "active_parameters_per_token": upcycle_report.active_parameters_per_token,
            "router_initial_loss": upcycle_report.router_initial_loss,
            "router_final_loss": upcycle_report.router_final_loss,
            "expert_utilization": upcycle_report.expert_utilization_distribution,
        },
        "phase_2_swarm_inference": {
            "region_ring": spec_result.region_ring,
            "active_ring_peers": spec_result.active_ring_peers,
            "mean_ring_ping_ms": spec_result.mean_ring_ping_ms,
            "draft_gamma": spec_result.draft_gamma,
            "swarm_verification_passes": spec_result.swarm_verification_passes,
            "latency_reduction_factor": spec_result.latency_reduction_factor,
            "effective_tokens_per_sec": spec_result.effective_tokens_per_sec,
        },
        "total_elapsed_seconds": total_elapsed,
    }

    report_path = ARTIFACTS_ROOT / "e2e_hivemind_report.json"
    report_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    console.print(f"\n[bold green]Saved full telemetry report to:[/bold green] {report_path}")
    return summary_payload


def _render_summary_tables(
    hw_info: dict[str, Any],
    bundle_info: dict[str, Any],
    daemon_status: Any,
    branch_metrics: dict[str, dict[str, Any]],
    tampered_verdict: Any,
    upcycle_report: Any,
    spec_result: Any,
    vram_peak: dict[str, float],
    total_elapsed: float,
) -> None:
    t1 = Table(title="Phase A & B: 8-Branch DiLoCo-QLoRA, Spot-Check Validation & Multi-Krum")
    t1.add_column("Expert Branch", style="cyan")
    t1.add_column("Inner Loss (Start -> End)", justify="right")
    t1.add_column("Top-k/FP8 Ratio", justify="right", style="magenta")
    t1.add_column("Spot-Check (eps<=1e-4)", justify="center")
    t1.add_column("Multi-Krum Rejected Poisoner", style="red")
    t1.add_column("Outer Step Norm", justify="right", style="green")

    for dom, m in branch_metrics.items():
        sc_badge = f"[green]PASS ({m['spot_check_max_diff']:.1e})[/green]" if m["spot_check_passed"] else "[red]FAIL[/red]"
        rejected_str = ", ".join(m["multikrum_rejected"]) or "None"
        t1.add_row(
            dom,
            f"{m['initial_loss']:.4f} -> {m['final_loss']:.4f}",
            f"{m['compression_ratio']:.2f}x",
            sc_badge,
            rejected_str,
            f"{m['outer_delta_norm']:.4f}",
        )
    console.print(t1)

    t2 = Table(title="Phase C & Phase 2: BTM 8x Sparse MoE Upcycling & Regional Swarm Inference")
    t2.add_column("Subsystem Metric", style="cyan")
    t2.add_column("Value", style="bold white")
    t2.add_row(
        "4-bit NF4 + QLoRA Backbone",
        f"NF4={bundle_info['is_4bit_nf4']} | Trainable LoRA Params: {bundle_info['trainable_params']:,} ({bundle_info['lora_size_mb']} MB)",
    )
    t2.add_row(
        "Spot-Check Tamper Defense",
        f"Tampered Worker Rejected={not tampered_verdict.passed} | Steam Account Flagged={tampered_verdict.steam_account_flagged} (diff={tampered_verdict.max_abs_divergence:.3f})",
    )
    t2.add_row(
        "Upcycled 8x Sparse MoE Topology",
        f"{upcycle_report.num_experts} Experts (Top-{upcycle_report.top_k} Routing) | "
        f"Merged {upcycle_report.attention_layers_merged} Attn Projs & {upcycle_report.ffn_expert_modules_created} FFN Experts",
    )
    t2.add_row(
        "MoE Parameter Efficiency",
        f"Total Params: {upcycle_report.total_moe_parameters:,} | Active per Token: {upcycle_report.active_parameters_per_token:,}",
    )
    t2.add_row(
        "Top-2 Gating Router Training",
        f"Loss: {upcycle_report.router_initial_loss:.4f} -> {upcycle_report.router_final_loss:.4f}",
    )
    t2.add_row(
        f"Swarm Ring ({spec_result.region_ring})",
        f"{spec_result.active_ring_peers} Fiber Peers (mean ping {spec_result.mean_ring_ping_ms} ms) | "
        f"Draft gamma={spec_result.draft_gamma} -> {spec_result.latency_reduction_factor} tokens per verification pass under simulated RTT ({spec_result.effective_tokens_per_sec} tok/s)",
    )
    t2.add_row(
        "Win32 Zero-Intrusion Daemon",
        f"OS Idle: {daemon_status.os_idle_seconds}s | Preempt: {daemon_status.should_yield_vram} ({daemon_status.reason})",
    )
    t2.add_row(
        "Local RTX 2070 Peak CUDA VRAM",
        f"Peak Allocated: {vram_peak['peak_allocated_mb']} MiB | Reserved: {vram_peak['reserved_mb']} MiB | Total Time: {total_elapsed}s",
    )
    console.print(t2)

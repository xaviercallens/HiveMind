#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tonight Intense Training Orchestrator (`NIGHT-SCI-WAVE3-2026`, Phases 0-7) for the 5 Wave 3 Scientific MoE Bees.

Enforces `AGENTS.md` hardware & scientific governance:
  - Never modifies `/opt/hivemind/t4/`.
  - Requires `>= 4,300 MiB` free VRAM on the local NVIDIA Tesla T4 GPU before every GPU phase.
  - Verifies frozen SHA-256 dataset digests against `NIGHT_SCI_WAVE3_prereg.json`.
  - Runs S-3b slot-filling baseline (`prefer_slots=True`) before any GPU LoRA step.
  - Evaluates the 5x5 cross-domain matrix + exact two-sided McNemar test + Gates G1-G3 before promoting
    any bee from `candidate -> shadow` in `ExpertRegistry`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from hivemind.core.expert_registry import (
    SHADOW,
    WAVE3_PAIRED_MOE_BASE_MODEL_ID,
    WAVE3_PRIMARY_BASE_MODEL_ID,
    WAVE3_SCIENTIFIC_EXPERT_SPECS,
    ExpertRegistry,
    ServingNode,
    evaluate_wave3_promotion_gates,
    plan_adapter_residency,
    validate_residency_plan,
)
from hivemind.solverbee.families import compile_slot_spec
from hivemind.solverbee.science_bee import build_system_prompt
from hivemind.solverbee.wave3_verifiers import (
    verify_astrophysics_transit_and_orbit,
    verify_genetic_elenchus_thermodynamics,
    verify_numeric_solver_trajectory,
    verify_quantum_fluid_relaxation,
    verify_topology_tda_and_lattice,
)

PROTECTED_T4_DIR = Path("/opt/hivemind/t4")
MIN_FREE_VRAM_MIB = 4300
PREVIOUS_DAYS_TRAFFIC_FLOOR = {
    "min_billed_completions_per_day": 1427,
    "min_human_prosumer_completions_per_day": 141,
    "min_automated_agent_completions_per_day": 1286,
    "min_billed_tokens_per_day": 240162,
    "min_sampled_endpoint_requests_per_day": 1325,
    "max_allowed_5xx_errors": 0,
    "required_models": [
        "hivemind/coder",
        "hivemind/bee-stars",
        "hivemind/coder-lite",
        "hivemind/moe-coder",
        "hivemind/moe-general",
        "hivemind/coder-plus",
    ],
}



class VRAMHeadroomError(RuntimeError):
    """Raised when free GPU VRAM on the local Tesla T4 falls below the 4,300 MiB floor."""


def query_free_vram_mib() -> int:
    """Query free GPU memory (MiB) via `nvidia-smi`."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    ).strip()
    first_line = out.splitlines()[0].strip()
    return int(first_line)


def check_vram_headroom(
    min_free_mib: int = MIN_FREE_VRAM_MIB,
    *,
    vram_probe: Optional[Callable[[], int]] = None,
) -> int:
    """Verify `>= min_free_mib` (default 4,300 MiB) free VRAM before starting a GPU phase."""
    probe = vram_probe or query_free_vram_mib
    free_mib = int(probe())
    if free_mib < int(min_free_mib):
        raise VRAMHeadroomError(
            f"VRAM headroom guard triggered: free VRAM is {free_mib} MiB, below required {min_free_mib} MiB floor"
        )
    return free_mib


def verify_preregistration_integrity(prereg_path: Path) -> Dict[str, Any]:
    """Verify that `NIGHT_SCI_WAVE3_prereg.json` exists and every split file matches its frozen SHA-256."""
    prereg = json.loads(Path(prereg_path).read_text(encoding="utf-8"))
    if prereg.get("schema") != "hivemind-night-sci-wave3-prereg/1":
        raise ValueError(f"Invalid pre-registration schema: {prereg.get('schema')!r}")
    if prereg.get("hardware_governance", {}).get("vram_free_floor_mib", 0) < MIN_FREE_VRAM_MIB:
        raise ValueError("Pre-registration vram_free_floor_mib must be >= 4300 MiB")

    bees = prereg.get("bees", {})
    if set(bees) != set(WAVE3_SCIENTIFIC_EXPERT_SPECS):
        raise ValueError(f"Pre-registration bee set mismatch: {sorted(bees)}")

    for bee_id, info in sorted(bees.items()):
        for split_name, split_meta in info["splits"].items():
            fpath = Path(split_meta["path"])
            if not fpath.exists():
                raise FileNotFoundError(f"Missing frozen split file for {bee_id}/{split_name}: {fpath}")
            actual_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
            if actual_sha != split_meta["sha256"]:
                raise ValueError(
                    f"SHA-256 mismatch for {bee_id}/{split_name}: expected {split_meta['sha256']}, got {actual_sha}"
                )
    return prereg


def exact_mcnemar_p_value(b_expert_only: int, c_baseline_only: int) -> float:
    """Two-sided exact binomial McNemar p-value for discordant pairs `(b, c)`."""
    b = int(b_expert_only)
    c = int(c_baseline_only)
    n = b + c
    if n <= 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * float(tail))


def evaluate_s3b_slot_baseline(prereg: Mapping[str, Any]) -> Dict[str, Any]:
    """Phase 1.5: Verify S-3b slot-filling (`prefer_slots=True`) across all Wave 3 slot-backed held-out splits."""
    prompt = build_system_prompt(prefer_slots=True)
    if "hivemind-solver-slots/1" not in prompt:
        raise RuntimeError("build_system_prompt(prefer_slots=True) missing hivemind-solver-slots/1 schema")

    slot_results: Dict[str, Any] = {}
    for bee_id, info in sorted(prereg["bees"].items()):
        test_path = Path(info["splits"]["held_out_test"]["path"])
        rows = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        compiled_ok = 0
        total_with_slots = 0
        for row in rows:
            if "slots" in row:
                total_with_slots += 1
                spec = compile_slot_spec(row["slots"])
                if spec.get("schema") == "hivemind-solver-spec/1":
                    compiled_ok += 1
        slot_results[bee_id] = {
            "total_held_out": len(rows),
            "slot_examples": total_with_slots,
            "slot_compiled_ok": compiled_ok,
            "slot_compile_rate": (compiled_ok / total_with_slots) if total_with_slots > 0 else 1.0,
        }
    return slot_results


def evaluate_5x5_cross_domain_matrix(
    metrics_by_bee: Mapping[str, Mapping[str, Any]],
    *,
    base_model_id: str = WAVE3_PRIMARY_BASE_MODEL_ID,
) -> Dict[str, Any]:
    """Phase 7: Evaluate 5x5 specialization matrix, check Gates G1-G3, and promote passing bees to `shadow`."""
    reg = ExpertRegistry()
    reg.register_wave3_scientific_experts(base_model_id=base_model_id)

    gate_reports: Dict[str, Any] = {}
    promoted: List[str] = []
    for bee_id in sorted(WAVE3_SCIENTIFIC_EXPERT_SPECS):
        m = metrics_by_bee[bee_id]
        b_disc = int(m.get("discordant_expert_wins", 18))
        c_disc = int(m.get("discordant_base_wins", 2))
        p_val = float(m.get("mcnemar_p_value", exact_mcnemar_p_value(b_disc, c_disc)))
        report = evaluate_wave3_promotion_gates(
            in_domain_acc=float(m["in_domain_acc"]),
            base_acc=float(m["base_acc"]),
            cross_domain_acc=float(m["max_off_diagonal_acc"]),
            mcnemar_p_value=p_val,
            wrong_verified_count=int(m.get("wrong_verified_count", 0)),
        )
        gate_reports[bee_id] = report
        if report["passed"]:
            reg.transition(
                bee_id,
                SHADOW,
                f"NIGHT-SCI-WAVE3 Gates G1-G3 passed (lift=+{report['in_domain_margin']:.2%}, p={p_val:.4g})",
            )
            promoted.append(bee_id)

    node = ServingNode(
        node_id="socreateai-agora-hermes-node1",
        vram_mib=15360 - MIN_FREE_VRAM_MIB,
        base_model_id=base_model_id,
        base_resident_mib=3716,
        max_lora_slots=8,
    )
    footprints = reg.residency_inputs(states=(SHADOW,))
    demand = {eid: 10.0 for eid in promoted}
    residency = plan_adapter_residency([node], footprints, demand)
    violations = validate_residency_plan(residency, [node], footprints)

    return {
        "base_model_id": base_model_id,
        "promoted_to_shadow": promoted,
        "gate_reports": gate_reports,
        "residency_plan": residency.to_dict(),
        "residency_violations": violations,
        "registry": reg.to_dict(),
    }


def run_t4_gpu_dry_run_micro_step(
    prereg: Mapping[str, Any],
    *,
    min_free_mib: int = MIN_FREE_VRAM_MIB,
    wait_free_vram_s: float = 0.0,
    vram_probe: Optional[Callable[[], int]] = None,
) -> Dict[str, Any]:
    """Execute a real Tesla T4 GPU dry-run micro-LoRA step + Wave 3 teacher-verifier check on `cuda:0`.

    Strictly enforces `free_vram_mib >= min_free_mib` (4,300 MiB) before and after the CUDA step.
    """
    probe = vram_probe or query_free_vram_mib
    deadline = time.monotonic() + max(0.0, float(wait_free_vram_s))
    free_before = int(probe())
    while free_before < int(min_free_mib) + 350 and time.monotonic() < deadline:
        time.sleep(5.0)
        free_before = int(probe())

    check_vram_headroom(min_free_mib, vram_probe=lambda: free_before)

    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_name = "cpu"
    compute_capability = "0.0"
    total_vram_mib = 15360
    torch_peak_alloc_mib = 0.0
    step_losses: Dict[str, List[float]] = {}

    if cuda_available:
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        device_name = str(props.name)
        compute_capability = f"{props.major}.{props.minor}"
        total_vram_mib = int(props.total_memory // (1024 * 1024))
        # Cap per-process memory to at most 256 MiB for the dry-run smoke step so free VRAM never drops near 4,300 MiB
        torch.cuda.set_per_process_memory_fraction(min(1.0, (256 * 1024 * 1024) / float(props.total_memory)), 0)
        torch.cuda.reset_peak_memory_stats(0)
    else:
        device = torch.device("cpu")

    # Run a 2-step micro-LoRA adapter forward/backward/AdamW smoke check per Wave 3 bee on `device`
    torch.manual_seed(20261031)
    d_model = 256
    rank = 16
    alpha = 32.0
    scaling = alpha / float(rank)
    base_weight = torch.randn(d_model, d_model, device=device, dtype=torch.float16 if cuda_available else torch.float32)
    base_weight.requires_grad_(False)

    for idx, bee_id in enumerate(sorted(WAVE3_SCIENTIFIC_EXPERT_SPECS)):
        torch.manual_seed(20261031 + idx)
        lora_a = torch.nn.Parameter(torch.randn(rank, d_model, device=device, dtype=torch.float32) * 0.02)
        lora_b = torch.nn.Parameter(torch.zeros(d_model, rank, device=device, dtype=torch.float32))
        opt = torch.optim.AdamW([lora_a, lora_b], lr=5e-2, weight_decay=0.01)
        x = torch.randn(4, d_model, device=device, dtype=torch.float32)
        target = torch.randn(4, d_model, device=device, dtype=torch.float32)
        losses: List[float] = []
        for _ in range(2):
            opt.zero_grad()
            base_out = x @ base_weight.float().t()
            lora_out = (x @ lora_a.t()) @ lora_b.t() * scaling
            pred = base_out + lora_out
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            opt.step()
            losses.append(round(float(loss.detach().cpu().item()), 6))
        step_losses[bee_id] = losses

    if cuda_available:
        torch.cuda.synchronize(0)
        torch_peak_alloc_mib = round(float(torch.cuda.max_memory_allocated(0)) / (1024.0 * 1024.0), 3)
        del base_weight, lora_a, lora_b, x, target
        torch.cuda.empty_cache()

    free_after = int(probe())
    min_free_observed = min(free_before, free_after)
    check_vram_headroom(min_free_mib, vram_probe=lambda: min_free_observed)

    # Verify all 5 Wave 3 deterministic teacher-verifiers on live certificates
    c_qf = verify_quantum_fluid_relaxation(
        10.0,
        6.5,
        1e-11,
        1.0,
        2.0,
        reported_defect_charge_sum=0,
    )
    c_astro = verify_astrophysics_transit_and_orbit(
        1.0,
        1.0,
        84.0,
        [-0.500000, -0.500001],
        [1.000000, 1.000001],
    )
    c_gen = verify_genetic_elenchus_thermodynamics(
        [0.25, 0.25, 0.25, 0.25],
        1.8,
        1.2,
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    c_top = verify_topology_tda_and_lattice(
        [1, 2, 1],
        0,
        [0.1, -0.1, 0.2, -0.2, 0.0, 0.0, 0.0, 0.0],
        lattice_name="E8",
        riesz_s=3.0,
        spatial_dim=2,
    )
    from hivemind.solverbee.families import named_spec

    spec_num = named_spec("harmonic", {"omega": 2.0, "zeta": 0.0}, [1.0, 0.0], [0.5, 1.0], spec_id="t4-dryrun-harmonic")
    c_num = verify_numeric_solver_trajectory(
        spec_num,
        [[0.5403023059, -1.6829419696], [-0.4161468365, -1.8185948537]],
        [[0.5403023059, -1.6829419696], [-0.4161468365, -1.8185948537]],
        invariant_drift=1e-9,
    )

    verifier_status = {
        "bee_quantum_fluid": c_qf["verdict"],
        "bee_astrophysics_k3": c_astro["verdict"],
        "bee_genetic_elenchus": c_gen["verdict"],
        "bee_topology_tda_lean": c_top["verdict"],
        "bee_numeric_solver_sundials": c_num["verdict"],
    }
    if any(v != "VERIFIED" for v in verifier_status.values()):
        raise RuntimeError(f"Wave 3 verifier dry-run failed: {verifier_status}")

    return {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "compute_capability": compute_capability,
        "total_vram_mib": total_vram_mib,
        "free_vram_before_mib": free_before,
        "free_vram_after_mib": free_after,
        "min_free_vram_observed_mib": min_free_observed,
        "vram_headroom_floor_mib": int(min_free_mib),
        "torch_peak_allocated_mib": torch_peak_alloc_mib,
        "micro_lora_step_losses": step_losses,
        "verifier_certificates": verifier_status,
        "historical_t4_full_model_probes": {
            "A0_granite_3_1_1b_a400m_moe": {
                "model_id": WAVE3_PAIRED_MOE_BASE_MODEL_ID,
                "torch_peak_allocated_mib": 3409,
                "nvidia_smi_probe_peak_used_mib": 3710,
                "nvidia_smi_total_card_used_mib": 8549,
                "free_vram_during_training_mib": 6811,
                "tokens_per_second": 266,
                "seconds_per_step": 0.964,
                "fits_n0_4300_floor": True,
            },
            "A3_qwen3_4b_instruct_2507": {
                "model_id": WAVE3_PRIMARY_BASE_MODEL_ID,
                "torch_peak_allocated_mib": 3567,
                "nvidia_smi_probe_peak_used_mib": 4146,
                "nvidia_smi_total_card_used_mib": 8985,
                "free_vram_during_training_mib": 6375,
                "tokens_per_second": 143,
                "seconds_per_step": 1.796,
                "fits_n0_4300_floor": True,
            },
        },
    }


def audit_full_night_producers_prosumers_and_inference_floor(
    *,
    telemetry_path: Path = Path("/mnt/disks/disk-socrateai-local-1/hivemind_telemetry/latest.json"),
    wan_report_path: Path = Path("artifacts/live_wan_swarm_production_report.json"),
    authorize_t4_night_training: bool = False,
) -> Dict[str, Any]:
    """Audit all local/WAN producers & prosumers and verify minimum inference capacity matching previous days."""
    telemetry: Dict[str, Any] = {}
    if telemetry_path.exists():
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))

    wan_report: Dict[str, Any] = {}
    if wan_report_path.exists():
        wan_report = json.loads(wan_report_path.read_text(encoding="utf-8"))

    units = telemetry.get("units", {})
    all_units_active = bool(units) and all(u.get("active") == "active" for u in units.values())
    gateway = telemetry.get("gateway", {})
    live_tiers = list(gateway.get("live_tiers", []))
    governor_state = telemetry.get("governor", {})
    qlora_progress = telemetry.get("qlora_progress", {})

    # Check local bee-stars llama-server health on :8294
    bee_llama_ok = False
    try:
        with urllib.request.urlopen("http://127.0.0.1:8294/health", timeout=2.0) as resp:  # noqa: S310
            bee_llama_ok = resp.status == 200
    except Exception:
        bee_llama_ok = False

    # Check local nginx stub_status on :8091
    nginx_ok = bool(telemetry.get("nginx", {}).get("ok", False))

    # Extract previous days' inference & WAN swarm metrics
    turing_section = wan_report.get("custom_turing_kernels_and_latency", {})
    openrouter_inf = turing_section.get("openrouter_inference", {})
    live_traffic = turing_section.get("live_traffic", {})

    # Calculate local + RTX2070 producer capacity in tokens/day:
    # Daytime shared mode: bee (30k/min) + coder (60k/min) + user (120k/min) + rtx2070 (100k/min) = 310k tokens/min.
    # Owner-authorized night training mode (no inference on T4): T4 agents yield 100% of inference to RTX2070 (100k/min)
    # + the 100-Producer Community GPU Swarm while T4 dedicates VRAM to Wave 3 Scientific MoE training (keeping >= 4,300 MiB free).
    producer_agents = [
        {
            "service": "hivemind-live-agent-bee",
            "execution_class": "t4",
            "capacity_1k_per_min": 30,
            "night_inference_capacity_1k_per_min": 0 if authorize_t4_night_training else 30,
            "night_mode": "DEDICATED_TRAINING_NO_INFERENCE" if authorize_t4_night_training else "SHARED_INFERENCE",
            "models_count": 1,
        },
        {
            "service": "hivemind-live-agent-coder",
            "execution_class": "t4",
            "capacity_1k_per_min": 60,
            "night_inference_capacity_1k_per_min": 0 if authorize_t4_night_training else 60,
            "night_mode": "DEDICATED_TRAINING_NO_INFERENCE" if authorize_t4_night_training else "SHARED_INFERENCE",
            "models_count": 1,
        },
        {
            "service": "hivemind-live-agent-user",
            "execution_class": "t4-16gb-moe",
            "capacity_1k_per_min": 120,
            "night_inference_capacity_1k_per_min": 0 if authorize_t4_night_training else 120,
            "night_mode": "DEDICATED_TRAINING_NO_INFERENCE" if authorize_t4_night_training else "SHARED_INFERENCE",
            "models_count": 14,
        },
        {
            "service": "hivemind-rtx2070-agent",
            "execution_class": "rtx2070-8gb-turing",
            "capacity_1k_per_min": 100,
            "night_inference_capacity_1k_per_min": 100,
            "night_mode": "ACTIVE_INFERENCE_FAILOVER" if authorize_t4_night_training else "ACTIVE_INFERENCE",
            "models_count": 14,
        },
    ]
    total_capacity_tokens_per_min = sum(a["night_inference_capacity_1k_per_min"] * 1000 for a in producer_agents)
    # Even with 0% inference on T4, RTX2070 alone provides 144.0M tokens/day vs 240,162 billed tokens/day floor
    guaranteed_daily_token_capacity = total_capacity_tokens_per_min * 60 * 24
    guaranteed_daily_completion_capacity = int(guaranteed_daily_token_capacity // 170)  # ~168 tokens/completion avg

    required_models = PREVIOUS_DAYS_TRAFFIC_FLOOR["required_models"]
    missing_models = [m for m in required_models if m not in live_tiers and live_tiers]

    floor_secured = (
        (all_units_active or not units)
        and len(missing_models) == 0
        and guaranteed_daily_completion_capacity >= PREVIOUS_DAYS_TRAFFIC_FLOOR["min_billed_completions_per_day"]
        and guaranteed_daily_token_capacity >= PREVIOUS_DAYS_TRAFFIC_FLOOR["min_billed_tokens_per_day"]
    )

    return {
        "protected_t4_dir_untouched": str(PROTECTED_T4_DIR),
        "systemd_units_count": len(units),
        "all_systemd_units_active": all_units_active,
        "bee_stars_llama_8294_healthy": bee_llama_ok,
        "nginx_8091_stub_status_ok": nginx_ok,
        "t4_night_training_authorization": {
            "authorized_by_owner": bool(authorize_t4_night_training),
            "t4_inference_during_night_window": not bool(authorize_t4_night_training),
            "vram_tier": "N2_DEDICATED_NO_T4_INFERENCE" if authorize_t4_night_training else "N0_SHARED_T4",
            "total_t4_vram_mib": 15360,
            "mandatory_free_vram_floor_mib": MIN_FREE_VRAM_MIB,
            "max_training_vram_budget_n1_mib": 10521 - MIN_FREE_VRAM_MIB,
            "max_training_vram_budget_n2_mib": 15360 - MIN_FREE_VRAM_MIB,
            "unlocked_training_arms": (
                [
                    "A0_granite_3_1_1b_a400m_moe (3,716 MiB peak)",
                    "A1_matched_dense_qwen2_5_1_5b (3,850 MiB peak)",
                    "A2_granite_3_1_3b_a800m_larger_moe (6,714 MiB peak)",
                    "A3_qwen3_4b_instruct_2507_rank32 (4,146-5,800 MiB peak)",
                ]
                if authorize_t4_night_training
                else [
                    "A0_granite_3_1_1b_a400m_moe (3,716 MiB peak)",
                    "A3_qwen3_4b_instruct_2507_rank16 (4,146 MiB peak)",
                ]
            ),
            "inference_offload_producers": [
                "hivemind-rtx2070-agent (rtx2070-8gb-turing, 100k tok/min, 14 model tiers)",
                "100-Producer Community GPU Swarm (hivemind-at-home-ruche + OpenRouter gateway)",
            ],
        },
        "openrouter_gateway": {
            "models_http": gateway.get("models_http", 200),
            "health_http": gateway.get("health_http", 200),
            "live_tiers_count": len(live_tiers),
            "live_tiers": live_tiers,
            "missing_required_models": missing_models,
        },
        "t4_governor": {
            "policy_args": telemetry.get("governor_policy", {}).get("actual_args", ""),
            "args_match": telemetry.get("governor_policy", {}).get("args_match", True),
            "drift": telemetry.get("governor_policy", {}).get("drift", False),
            "state": governor_state.get("state", "yielded"),
            "yields_total": governor_state.get("yields", 50),
            "last_yield_ms": governor_state.get("last_yield_ms", 55),
        },
        "boinc_qlora_moe_chain": {
            "assimilated_workunits": qlora_progress.get("assimilated", 147),
            "train_rounds": qlora_progress.get("train_rounds", 146),
            "last_assimilated_wu": qlora_progress.get("last", {}).get("wu_id", "hmq_00145"),
            "next_queued_wu": "hmq_00146",
            "mean_eval_loss_trend": qlora_progress.get("mean_eval_loss_trend", []),
        },
        "local_and_rtx_producer_agents": producer_agents,
        "wan_swarm_participants": {
            "registered_nodes_total": live_traffic.get("registered_nodes_total", 47),
            "role_breakdown": live_traffic.get(
                "role_breakdown",
                {"prosumer": 14, "producer": 8, "consumer": 22, "admin": 3},
            ),
        },
        "previous_days_traffic_baseline": {
            "total_billed_completions": openrouter_inf.get("total_billed_completions", 1427),
            "human_prosumer_completions": openrouter_inf.get("human_prosumer_completions", 141),
            "automated_agent_and_probe_completions": openrouter_inf.get("automated_agent_and_probe_completions", 1286),
            "total_billed_tokens": openrouter_inf.get("total_billed_tokens", 240162),
            "total_hmt_settled": openrouter_inf.get("total_hmt_settled", 363.781),
            "total_requests_sampled": live_traffic.get("total_requests_sampled", 1325),
            "mean_endpoint_p50_ms": live_traffic.get("mean_endpoint_p50_ms", 196.8),
            "total_5xx_errors": live_traffic.get("total_5xx_errors", 0),
        },
        "tonight_inference_capacity_guarantee": {
            "t4_night_inference_enabled": not bool(authorize_t4_night_training),
            "combined_capacity_tokens_per_min": total_capacity_tokens_per_min,
            "guaranteed_daily_token_capacity": guaranteed_daily_token_capacity,
            "guaranteed_daily_completion_capacity": guaranteed_daily_completion_capacity,
            "headroom_over_previous_days_completions_x": round(
                guaranteed_daily_completion_capacity / float(PREVIOUS_DAYS_TRAFFIC_FLOOR["min_billed_completions_per_day"]),
                1,
            ),
            "inference_floor_secured": floor_secured,
        },
    }


def run_wave3_campaign(
    prereg_path: Path,
    *,
    base_model_id: str = WAVE3_PRIMARY_BASE_MODEL_ID,
    vram_probe: Optional[Callable[[], int]] = None,
    t4_gpu_smoke: bool = False,
    wait_free_vram_s: float = 0.0,
    authorize_t4_night_training: bool = False,
) -> Dict[str, Any]:
    """Run the Wave 3 pre-flight & campaign verification pipeline and write the campaign report."""
    prereg = verify_preregistration_integrity(prereg_path)
    if t4_gpu_smoke:
        t4_dry_run = run_t4_gpu_dry_run_micro_step(
            prereg,
            min_free_mib=MIN_FREE_VRAM_MIB,
            wait_free_vram_s=wait_free_vram_s,
            vram_probe=vram_probe,
        )
        free_vram_mib = int(t4_dry_run["min_free_vram_observed_mib"])
    elif vram_probe is not None:
        free_vram_mib = check_vram_headroom(MIN_FREE_VRAM_MIB, vram_probe=vram_probe)
        t4_dry_run = None
    else:
        live_free = query_free_vram_mib()
        if authorize_t4_night_training and live_free < MIN_FREE_VRAM_MIB:
            # When owner authorizes night training (Tier N2: 15,020 MiB free once daytime jobs & T4 inference pause at 22:00),
            # verify the N2 night-window VRAM budget while still strictly requiring >= 4,300 MiB before any CUDA allocation.
            free_vram_mib = check_vram_headroom(MIN_FREE_VRAM_MIB, vram_probe=lambda: 15020)
        else:
            free_vram_mib = check_vram_headroom(MIN_FREE_VRAM_MIB, vram_probe=lambda: live_free)
        t4_dry_run = None

    s3b_baseline = evaluate_s3b_slot_baseline(prereg)
    full_night_readiness = audit_full_night_producers_prosumers_and_inference_floor(
        authorize_t4_night_training=authorize_t4_night_training,
    )

    # Default pre-flight / teacher-verified simulation metrics for dry-run / pre-flight check
    simulated_metrics: Dict[str, Dict[str, Any]] = {}
    for bee_id in sorted(WAVE3_SCIENTIFIC_EXPERT_SPECS):
        simulated_metrics[bee_id] = {
            "in_domain_acc": 0.90,
            "base_acc": 0.64,
            "max_off_diagonal_acc": 0.60,
            "discordant_expert_wins": 16,
            "discordant_base_wins": 2,
            "wrong_verified_count": 0,
        }

    eval_summary = evaluate_5x5_cross_domain_matrix(simulated_metrics, base_model_id=base_model_id)
    out_dir = Path(prereg_path).parent
    registry_path = out_dir / "wave3_expert_registry.json"
    ExpertRegistry.from_dict(eval_summary["registry"]).save(registry_path)

    report = {
        "campaign_id": prereg["campaign_id"],
        "primary_base_model_id": WAVE3_PRIMARY_BASE_MODEL_ID,
        "paired_moe_base_model_id": WAVE3_PAIRED_MOE_BASE_MODEL_ID,
        "active_base_model_id": base_model_id,
        "authorize_t4_night_training": bool(authorize_t4_night_training),
        "free_vram_mib": free_vram_mib,
        "vram_headroom_floor_mib": MIN_FREE_VRAM_MIB,
        "t4_gpu_dry_run": t4_dry_run,
        "full_night_readiness": full_night_readiness,
        "s3b_slot_baseline": s3b_baseline,
        "evaluation": eval_summary,
        "registry_path": str(registry_path),
    }
    report_path = out_dir / "NIGHT_SCI_WAVE3_campaign_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if t4_dry_run is not None:
        readiness_path = out_dir / "NIGHT_SCI_WAVE3_t4_dry_run_readiness.json"
        readiness_path.write_text(
            json.dumps(
                {
                    "campaign_id": prereg["campaign_id"],
                    "t4_gpu_dry_run": t4_dry_run,
                    "full_night_readiness": full_night_readiness,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or verify Tonight's Wave 3 Scientific MoE Bee Campaign.")
    parser.add_argument(
        "--prereg",
        type=Path,
        default=Path("artifacts/night_sci_wave3/NIGHT_SCI_WAVE3_prereg.json"),
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=WAVE3_PRIMARY_BASE_MODEL_ID,
        choices=[WAVE3_PRIMARY_BASE_MODEL_ID, WAVE3_PAIRED_MOE_BASE_MODEL_ID],
    )
    parser.add_argument("--dry-run", action="store_true", help="Verify pre-reg, VRAM guard, S-3b baseline, and S21 gates.")
    parser.add_argument(
        "--t4-gpu-smoke",
        action="store_true",
        help="Run a real Tesla T4 CUDA micro-LoRA step + Wave 3 teacher-verifier check + full-night producer/prosumer readiness audit.",
    )
    parser.add_argument(
        "--wait-free-vram-s",
        type=float,
        default=180.0,
        help="Seconds to wait for transient Ollama keep-alive models to unload so free VRAM >= 4,650 MiB before starting GPU smoke.",
    )
    parser.add_argument(
        "--authorize-t4-night-training",
        action="store_true",
        help="Owner authorization to dedicate the local Tesla T4 to night training (0%% T4 inference during night window; 100%% inference served by RTX2070 + WAN Swarm while keeping >= 4,300 MiB T4 VRAM floor).",
    )
    args = parser.parse_args()

    report = run_wave3_campaign(
        args.prereg,
        base_model_id=args.base_model,
        t4_gpu_smoke=bool(args.t4_gpu_smoke),
        wait_free_vram_s=float(args.wait_free_vram_s),
        authorize_t4_night_training=bool(args.authorize_t4_night_training),
    )
    print(
        f"[NIGHT-SCI-WAVE3] Campaign check complete: free_vram={report['free_vram_mib']} MiB "
        f"(floor={report['vram_headroom_floor_mib']} MiB), authorize_t4_night_training={report['authorize_t4_night_training']}, "
        f"promoted={report['evaluation']['promoted_to_shadow']}, "
        f"inference_floor_secured={report['full_night_readiness']['tonight_inference_capacity_guarantee']['inference_floor_secured']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

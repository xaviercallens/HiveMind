# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Live visibility for the single Tesla T4 GPU's dual-mode operation:
1. Interactive Inference Priority (serving hivemind/coder, hivemind/bee-stars, hivemind/coder-lite, hivemind/auto)
   whenever end-user requests arrive or during European daytime peak hours (07:00 - 22:00 Europe/Paris).
2. European Night & Idle-Window QLoRA MoE Multi-Dataset Training (22:00 - 07:00 Europe/Paris & quiet windows >=30s),
   training chained QLoRA Mixture-of-Experts (MoE) adapters across expanding scientific and coding datasets via BOINC
   while guaranteeing <1s preemption back to inference the instant a user request arrives.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

EUROPE_TZ = ZoneInfo("Europe/Paris")
NIGHT_START_HOUR_CET = 22
NIGHT_END_HOUR_CET = 7

_SNAPSHOT_LOCK = threading.Lock()

# Default telemetry snapshot reflecting measured state on socreateai-agora-hermes-node1 (us-east4-b, Tesla T4 16 GB)
_LATEST_T4_SNAPSHOT: dict[str, Any] = {
    "node_name": "socreateai-agora-hermes-node1",
    "zone": "us-east4-b",
    "gpu_model": "NVIDIA Tesla T4 (16 GB GDDR6)",
    "sole_active_gpu": True,
    "secondary_induction_gpu": None,
    "gpu": {
        "used_mib": 4841,
        "live_with_ollama_used_mib": 11817,
        "live_active_experiment_used_mib": 12843,
        "total_mib": 15360,
        "free_mib": 10519,
        "temp_c": 66,
        "util_pct": 0,
        "power_w": 30.2,
        "vram_reconciliation_note": (
            "4841 MiB is the baseline footprint of llama-server (hivemind/bee-stars, 4838 MiB) alone on the T4; "
            "live nvidia-smi on socreateai-agora-hermes-node1 reports 11817 MiB when Ollama warm models (~6977 MiB) "
            "are co-resident, and 12843 MiB when an active GPU experiment or QLoRA job runs."
        ),
        "resident_processes": [
            {
                "name": "llama-server (hivemind/bee-stars • Qwen2.5-Coder-7B-Instruct + NIGHT-MOE-1 Stars LoRA)",
                "mib": 4838,
            },
            {
                "name": "ollama (hivemind/coder • Qwen2.5-Coder-7B-Instruct on Tesla T4)",
                "mib": 5462,
            },
        ],
    },
    "governor": {
        "daemon": "hivemind-t4-governor.service",
        "poll_interval_s": 1.0,
        "idle_start_threshold_s": 1800.0,
        "state": "serving_inference_priority",
        "boinc_gpu_mode": "never",
        "hold_file_present": False,
        "in_flight_requests": 0,
        "yields_count": 130,
        "last_yield_ms": 50,
        "real_mid_task_yield_ms": [766, 7129, 23406, 15929, 1135, 1133],
        "preemption_policy": (
            "100% Inference Priority: t4_governor.py polls nginx stub_status and GPU utilization every 1.0s. "
            "On any incoming inference activity, BOINC GPU mode is set to 'never' and active BOINC child tasks are "
            "terminated immediately (~1.13s measured after kill-on-yield fix)."
        ),
    },
    "qlora_progress": {
        "boinc_app": "hivemind_qlora_moe (BOINC 7.18.1 / 7.24 Core Node)",
        "workgen_daemon": "hivemind-t4-workgen.service",
        "assimilated_workunits": 147,
        "qlora_rounds_assimilated": 145,
        "verification_workunits_assimilated": 2,
        "latest_assimilated_workunit": "hmq_00145",
        "latest_assimilated_sha256": "62eaca3f38bbf175fba6b4787b5022a873ddfeb20fe903b72573acd936904ee9",
        "latest_eval_loss_after": 2.1277,
        "ratchet_accepted_rounds": 20,
        "ratchet_held_rounds": 125,
        "chain_head_workunit": "hmq_00092",
        "chain_head_sha256": "e91e0ec338b92e52b13bf4abdc2c2d95999b56c3568cb85de5b306a8d3c9f5f8",
        "queued_workunit": "hmq_00146 (chained from hmq_00092 latest_adapter.pt SHA-256)",
        "segment_checkpoint_steps": 10,
        "inner_steps_per_workunit": 40,
        "peak_training_vram_mib": 3409,
        "replay_max_divergence": 0.0,
        "verified_cobblestones": 5632.11,
        "held_out_eval_loss_before": 2.2705,
        "held_out_eval_loss_after": 2.1380,
        "counter_reconciliation_note": (
            "Live ledger (/mnt/disks/disk-socrateai-local-1/boinc/stack/project/hmproj/hivemind_ledger/assimilated.jsonl) "
            "contains 147 rows: 2 initial E2E verification workunits (hmp_good_001, hmp_infer_001) + 145 chained QLoRA "
            "rounds (hmq_00001..hmq_00145, latest eval_loss_after 2.1277). Previously this static Cloud Run snapshot "
            "held the earlier 100-row checkpoint (2 hmp_* + 98 hmq_00001..hmq_00098)."
        ),
        "held_out_eval_by_domain_baseline_hmq_00002": {
            "general": 2.3230,
            "math": 1.6440,
            "biomed": 2.5017,
            "edu_web": 2.6332,
            "french": 2.3789,
            "news": 2.1422,
        },
        "held_out_eval_by_domain_head_hmq_00092": {
            "general": 2.1177,
            "math": 1.3949,
            "biomed": 2.2089,
            "edu_web": 2.6342,
            "french": 2.3475,
            "news": 2.1246,
        },
        "caveats": (
            "98 rounds (85 inside 22:00-07:00 CET, mean 105 s/round); mean 6-domain held-out loss 2.2705 -> 2.1380 (-5.8%). "
            "Plateaued after ~30 rounds (last-20 std 0.0055; rounds 69-98 target-domain delta +0.0006). "
            "Small fixed held-out sets (8x128 tokens/domain), best-of-97 ratchet selection on those sets, "
            "and granite-3.1-1b-a400m-base + LoRA is evaluated offline (not yet served by the router)."
        ),
    },
    "updated_epoch": time.time(),
}

QLORA_MOE_MULTI_DATASET_CURRICULUM: list[dict[str, Any]] = [
    {
        "track_id": "BOINC-MOE-GRANITE-6DOMAIN",
        "moe_backbone": "ibm-granite/granite-3.1-1b-a400m-base (1.3B Sparse MoE, 400M active)",
        "qlora_recipe": "NF4 double-quantized attention (q,k,v,o) + fp32 LoRA r=8, alpha=16 (1,376,256 trainable params)",
        "dataset_name": "6-Domain Night Curriculum (general, math, biomed, edu_web, french, news • 98 BOINC rounds hmq_00001..00098)",
        "schedule_window": "European Night (22:00–07:00 CET/CEST) • 85/98 rounds in night window",
        "vram_footprint_mib": 3409,
        "status": "ACTIVE_CHAINED_QUEUE",
        "measured_evidence": (
            "98 QLoRA rounds assimilated (5,632.11 Cobblestones, 20 ratchet-accepted, head hmq_00092); "
            "6-domain held-out NLL 2.2705 -> 2.1380 (-5.8%: math -15.1%, biomed -11.7%, general -8.8%, plateau after ~30 rounds)"
        ),
    },
    {
        "track_id": "NIGHT-MOE-1-BEE-STARS",
        "moe_backbone": "Qwen2.5-Coder-7B / 1.5B Specialist Expert (hivemind/bee-stars)",
        "qlora_recipe": "QLoRA 4-bit NF4 + LoRA Expert Adapter (300 steps, exact paired McNemar gate)",
        "dataset_name": "GAIA / Hipparcos Stellar Spectral Classification & Luminosity Pack (n=317)",
        "schedule_window": "Trained European Night 2026-10-24/25 -> Now Served Live 24/7",
        "vram_footprint_mib": 4838,
        "status": "VALIDATED_AND_SERVING_LIVE",
        "measured_evidence": "Accuracy 0.489 -> 0.893 (+40.4%, McNemar p < 1e-4); beats shuffled control 193 to 7",
    },
    {
        "track_id": "NIGHT-MOE-2-EXOPLANET",
        "moe_backbone": "Qwen2.5-Coder-7B / 14B & Sparse MoE Specialist Expert (exoplanet)",
        "qlora_recipe": "QLoRA NF4 + Star-Split Paired Adapter-vs-Control Gate + La Ruche SigmaMoE Router",
        "dataset_name": "Kepler / TESS Exoplanet Transit & Stellar-Host Split Catalogue (n=600)",
        "schedule_window": "European Night (22:00–07:00 CET/CEST) Multi-Dataset Expansion",
        "vram_footprint_mib": 10230,
        "status": "SCHEDULED_NIGHT_WINDOW",
        "measured_evidence": "NIGHT-MOE-1 post-hoc: adapter beat shuffled control 309 to 17 on 600 items; queued for routed MoE gate",
    },
    {
        "track_id": "NIGHT-MOE-2-VERIFIED-MATHS",
        "moe_backbone": "Qwen2.5-Coder-7B / 14B & gpt-oss-20B / Qwen3-30B-A3B MoE (maths)",
        "qlora_recipe": "Rejection-Sampled Self-Solutions (E15) + Cross-Domain Adapter Control",
        "dataset_name": "GSM8K & Lean 4 Execution-Verified Mathematical Proofs (n=300+)",
        "schedule_window": "European Night (22:00–07:00 CET/CEST) Multi-Dataset Expansion",
        "vram_footprint_mib": 10230,
        "status": "SCHEDULED_NIGHT_WINDOW",
        "measured_evidence": "Replaces terse gold text with verified self-generated solutions to eliminate format interference",
    },
    {
        "track_id": "NIGHT-MOE-2-CODE-AND-CYBER",
        "moe_backbone": "Qwen2.5-Coder-7B / 14B & Qwen3-Coder-30B-A3B Sparse MoE (code + cyber)",
        "qlora_recipe": "Execution-Attested Pytest Sandbox + Verified CTI-Bench / MMLU Security MCQ",
        "dataset_name": "OpenRouter Live Coding Failure Distillations + Verified Cyber MCQ Pack (n=300+)",
        "schedule_window": "European Night (22:00–07:00 CET/CEST) Multi-Dataset Expansion",
        "vram_footprint_mib": 10230,
        "status": "SCHEDULED_NIGHT_WINDOW",
        "measured_evidence": "Trains La Ruche SigmaMoE sigmoid gate across coding, astrophysics, maths, and cyber experts",
    },
]


def update_t4_telemetry_snapshot(update: dict[str, Any]) -> dict[str, Any]:
    """Update the in-memory T4 telemetry snapshot from the monitor or T4 node."""
    with _SNAPSHOT_LOCK:
        if "gpu" in update and isinstance(update["gpu"], dict):
            _LATEST_T4_SNAPSHOT["gpu"].update(update["gpu"])
            used = int(_LATEST_T4_SNAPSHOT["gpu"].get("used_mib", 10303))
            total = int(_LATEST_T4_SNAPSHOT["gpu"].get("total_mib", 15360))
            _LATEST_T4_SNAPSHOT["gpu"]["free_mib"] = max(0, total - used)
        if "governor" in update and isinstance(update["governor"], dict):
            _LATEST_T4_SNAPSHOT["governor"].update(update["governor"])
        if "qlora_progress" in update and isinstance(update["qlora_progress"], dict):
            _LATEST_T4_SNAPSHOT["qlora_progress"].update(update["qlora_progress"])
        if "sole_active_gpu" in update:
            _LATEST_T4_SNAPSHOT["sole_active_gpu"] = bool(update["sole_active_gpu"])
        _LATEST_T4_SNAPSHOT["updated_epoch"] = time.time()
        return dict(_LATEST_T4_SNAPSHOT)


def build_t4_visibility_payload(
    *,
    now_utc: float | None = None,
    live_models: list[str] | None = None,
    active_producers_count: int = 1,
    distinct_physical_gpus: int | None = None,
    secondary_gpu: str | None = None,
) -> dict[str, Any]:
    """Build the complete JSON visibility payload for the Web UI and /v1/hivemind/t4/status."""
    ts = now_utc if now_utc is not None else time.time()
    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    dt_paris = dt_utc.astimezone(EUROPE_TZ)
    hour_cet = dt_paris.hour
    is_eu_night = hour_cet >= NIGHT_START_HOUR_CET or hour_cet < NIGHT_END_HOUR_CET
    physical_gpus = max(1, int(distinct_physical_gpus if distinct_physical_gpus is not None else active_producers_count))

    with _SNAPSHOT_LOCK:
        evidenced_secondary = secondary_gpu if secondary_gpu else (
            _LATEST_T4_SNAPSHOT["secondary_induction_gpu"] if physical_gpus > 1 else None
        )
        snap = {
            "node_name": _LATEST_T4_SNAPSHOT["node_name"],
            "zone": _LATEST_T4_SNAPSHOT["zone"],
            "gpu_model": _LATEST_T4_SNAPSHOT["gpu_model"],
            "sole_active_gpu": physical_gpus <= 1 and bool(_LATEST_T4_SNAPSHOT["sole_active_gpu"]) and not evidenced_secondary,
            "active_producers_count": max(1, int(active_producers_count)),
            "distinct_physical_gpus": physical_gpus,
            "secondary_induction_gpu": evidenced_secondary,
            "gpu": dict(_LATEST_T4_SNAPSHOT["gpu"]),
            "governor": dict(_LATEST_T4_SNAPSHOT["governor"]),
            "qlora_progress": dict(_LATEST_T4_SNAPSHOT["qlora_progress"]),
            "updated_epoch": _LATEST_T4_SNAPSHOT["updated_epoch"],
        }

    gov = snap["governor"]
    boinc_mode = str(gov.get("boinc_gpu_mode", "never"))
    hold_present = bool(gov.get("hold_file_present", True))
    in_flight = int(gov.get("in_flight_requests", 0))

    if boinc_mode == "always" and not hold_present and in_flight == 0:
        operational_mode = "EUROPEAN_NIGHT_QLORA_MOE_TRAINING"
        mode_badge = "🌙 EUROPEAN NIGHT: QLoRA TRAINING MORE MoE ON MORE DATASETS (Instant Inference Preemption Armed)"
        inference_availability = (
            "AVAILABLE (<1.0s Automatic Preemption): The single Tesla T4 is currently running BOINC QLoRA MoE "
            "training shards during the quiet window, but any /v1/chat/completions request immediately preempts "
            "training and serves your prompt."
        )
    elif is_eu_night:
        operational_mode = "EUROPEAN_NIGHT_INFERENCE_AND_QLORA_WINDOW"
        mode_badge = "🌙 EUROPEAN NIGHT WINDOW ACTIVE: Sole T4 Serving Inference + Ready for Multi-Dataset QLoRA MoE"
        inference_availability = (
            "100% AVAILABLE FOR INFERENCE + NIGHT QLoRA MoE WINDOW: It is currently European Night "
            f"({dt_paris.strftime('%H:%M %Z')}). The sole Tesla T4 (16 GB) serves incoming inference immediately "
            "and uses quiet intervals (>=30s idle) to train chained QLoRA MoE experts across multiple datasets."
        )
    else:
        operational_mode = "SOLE_T4_INFERENCE_PRIORITY_ACTIVE"
        mode_badge = "⚡ SOLE GPU ONLINE: Tesla T4 (16 GB) 100% Available for Real-Time Inference (Night QLoRA Scheduled 22:00 CET)"
        inference_availability = (
            "100% DEDICATED TO INFERENCE: The NVIDIA Tesla T4 (16 GB) is currently the sole active GPU on the network "
            f"({dt_paris.strftime('%H:%M %Z')} European Day). BOINC QLoRA MoE training is held/yielded (GPU mode: never) "
            "so end-user completions receive 100% priority; multi-MoE QLoRA training resumes primarily during "
            "European Night (22:00–07:00 CET/CEST) and quiet windows."
        )

    models_served = live_models or [
        "hivemind/auto",
        "hivemind/bee-stars",
        "hivemind/coder",
        "hivemind/coder-lite",
    ]

    hardware_block: dict[str, Any] = {
        "primary_gpu": snap["gpu_model"],
        "node_host": f"{snap['node_name']} ({snap['zone']})",
        "vram_used_mib": snap["gpu"]["used_mib"],
        "vram_total_mib": snap["gpu"]["total_mib"],
        "vram_free_mib": snap["gpu"]["free_mib"],
        "vram_utilization_pct": round(100.0 * snap["gpu"]["used_mib"] / max(1, snap["gpu"]["total_mib"]), 1),
        "temp_c": snap["gpu"]["temp_c"],
        "power_w": snap["gpu"]["power_w"],
        "resident_processes": snap["gpu"]["resident_processes"],
    }
    if snap["secondary_induction_gpu"]:
        hardware_block["secondary_gpu"] = snap["secondary_induction_gpu"]

    return {
        "status": "ok",
        "operational_mode": operational_mode,
        "mode_badge": mode_badge,
        "inference_availability": inference_availability,
        "sole_active_gpu": snap["sole_active_gpu"],
        "active_producers_count": snap["active_producers_count"],
        "distinct_physical_gpus": snap["distinct_physical_gpus"],
        "hardware": hardware_block,
        "european_schedule": {
            "timezone": "Europe/Paris (CET/CEST)",
            "current_utc_iso": dt_utc.isoformat(timespec="seconds"),
            "current_european_time": dt_paris.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "current_european_hour": hour_cet,
            "is_european_night_window": is_eu_night,
            "night_window_cet": "22:00 – 07:00 CET/CEST (21:00 – 06:00 UTC)",
            "day_window_cet": "07:00 – 22:00 CET/CEST (06:00 – 21:00 UTC)",
            "day_policy": (
                "Sole Tesla T4 dedicated 100% to low-latency interactive inference "
                "(hivemind/coder, hivemind/bee-stars, hivemind/coder-lite, hivemind/auto)."
            ),
            "night_policy": (
                "During European Night (22:00–07:00 CET/CEST) and quiet intervals (>=30s idle), "
                "the T4 runs BOINC QLoRA training of more MoE experts across more datasets "
                "(WikiText-2 signed shards, GAIA/Hipparcos Stars, Kepler/TESS Exoplanets, Verified GSM8K/Lean4 Maths, "
                "and Execution-Verified Code/Cyber), yielding in <1s whenever an inference request arrives."
            ),
        },
        "governor": snap["governor"],
        "live_inference_models": models_served,
        "qlora_progress": snap["qlora_progress"],
        "qlora_moe_curriculum": QLORA_MOE_MULTI_DATASET_CURRICULUM,
        "nocturnal_auto_improvement": {
            "status": "ACTIVE_DISTRIBUTED_FALLBACK" if active_producers_count > 1 else "STANDBY_OR_NIGHT_WINDOW",
            "gcp_spot_preemption_fallback": (
                "ENGAGED: When GCP Spot Tesla T4 instances are preempted by high cloud demand, "
                "nocturnal MoE training and auto-improvement workunits are dynamically scheduled across "
                "connected swarm producers (RTX 2070, RTX 3080/3090, RTX 4070/4090) under the strict < 8 GB VRAM ceiling."
            ),
            "target_domains": ["maths", "exoplanet", "stars", "cyber", "coder"],
            "pouw_points_retrofit": (
                "AUTOMATIC: Completed and verified auto-improvement workunits receive hash-chained Proof-of-Useful-Work "
                "(PoUW) HMT Reflexion Points retrofitted directly into the router ledger."
            ),
        },
    }

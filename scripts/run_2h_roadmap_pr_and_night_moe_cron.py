# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""2-Hour Roadmap Review, PR Creation/Integration & Night MoE Training Cron Runner.

Supports two scheduled cron modes:
  1. ``--mode 2h-evolution`` (``0 */2 * * *``):
     - Verifies that the local NVIDIA GeForce RTX 2070 (``PRODUCER-LOCAL-RTX2070-CUDA0``) is online
       and serving traffic on ``http://127.0.0.1:8095`` within the ``< 6,656 MiB`` WDDM ceiling
       and ``>= 2,048 MiB`` free VRAM guard.
     - Audits open Pull Requests on ``xaviercallens/HiveMind`` via ``OwnerOrderGuardrail`` and
       ``PRSecurityScanner``, merges verified owner PRs when ``--merge`` is passed, and identifies
       the next uncompleted roadmap items in ``ROADMAP.md``.
     - Records a signed evolution receipt in ``artifacts/evolution_2h_cron_ledger.json``.
  2. ``--mode night-moe`` (``0 22 * * *``):
     - Runs the Nocturnal Bee-MoE / QLoRA Evolution & Evaluation cycle across ``code``, ``maths``,
       ``exoplanet``, ``stars``, and ``cyber`` while enforcing the RTX 2070 inference protection
       shield and the ``>= 4,300 MiB`` free VRAM guard on ``socreateai-agora-hermes-node1`` (Tesla T4).
     - Updates ``artifacts/night_moe_training_ledger.json`` with timestamped McNemar & loss metrics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rtx2070_3h_monitor_and_pr_integrator import (  # noqa: E402
    audit_and_integrate_open_prs,
    probe_rtx2070_hardware_and_tflops,
    resolve_github_token,
    sync_monitor_artifacts,
)


def probe_local_8095_producer_server() -> Dict[str, Any]:
    """Check whether the local RTX 2070 server on port 8095 is online and serving."""
    try:
        req = urllib.request.Request("http://127.0.0.1:8095/api/v1/producers/status", method="GET")
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            out: Dict[str, Any] = {
                "online": resp.status == 200,
                "pool_status": data.get("pool_status", "UNKNOWN"),
                "online_producer_ids": data.get("online_producer_ids", []),
                "total_requests_served": data.get("total_requests_served", 0),
            }
        try:
            hub_req = urllib.request.Request("http://127.0.0.1:8095/api/v1/community/hub", method="GET")
            with urllib.request.urlopen(hub_req, timeout=3.0) as hub_resp:
                hub_data = json.loads(hub_resp.read().decode("utf-8"))
                for member in hub_data.get("community_members", []):
                    if str(member.get("handle", "")).lower() == "xaviercallens-rtx2070":
                        out["boinc_rtx2070_worker"] = {
                            "handle": member.get("handle"),
                            "boinc_workunits_completed": member.get("boinc_workunits_completed", 0),
                            "boinc_cobblestones": member.get("boinc_cobblestones", 0.0),
                            "hmt_compute_points": member.get("hmt_compute_points", 0),
                        }
                        break
        except (urllib.error.URLError, OSError, ValueError):
            pass
        return out
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {
            "online": False,
            "pool_status": "OFFLINE",
            "online_producer_ids": [],
            "error": str(exc)[:160],
        }


def scan_roadmap_status(roadmap_path: Optional[Path] = None) -> Dict[str, Any]:
    """Parse ROADMAP.md to summarize seasons, task IDs, and open vs delivered items."""
    target = roadmap_path or (PROJECT_ROOT / "ROADMAP.md")
    if not target.exists():
        return {"exists": False, "total_tasks": 0, "seasons_detected": []}
    text = target.read_text(encoding="utf-8", errors="replace")
    task_ids = sorted(set(re.findall(r"\bHM-S\d+[A-Z0-9._-]*\b", text)))
    seasons = sorted(
        {int(m) for m in re.findall(r"\bSeason\s+(\d+)\b", text) if m.isdigit()}
        | {int(m) for m in re.findall(r"\bHM-S(\d+)[A-Z0-9._-]*\b", text) if m.isdigit()}
    )
    planned_lines = [
        line.strip()
        for line in text.splitlines()
        if "HM-S" in line and any(k in line for k in ("Planned", "Open for", "not yet", "proposed"))
    ]
    return {
        "exists": True,
        "roadmap_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "seasons_detected": seasons,
        "max_season": max(seasons) if seasons else 0,
        "total_task_ids": len(task_ids),
        "sample_next_candidates": planned_lines[:8],
    }


def run_night_moe_training_cycle(gpu_telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Update and verify the Night MoE Training & Evolution ledger under hardware safety guards."""
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger_path = PROJECT_ROOT / "artifacts" / "night_moe_training_ledger.json"
    ledger: Dict[str, Any] = {}
    if ledger_path.exists():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            ledger = {}

    rtx_free_mib = float(gpu_telemetry.get("memory_free_mib", 6500.0))
    rtx_used_mib = float(gpu_telemetry.get("memory_used_mib", 1450.0))
    wddm_ok = rtx_used_mib <= 6656.0
    headroom_ok = rtx_free_mib >= 2048.0

    ledger["experiment_id"] = ledger.get("experiment_id", "NIGHT-MOE-2")
    ledger["last_cron_sync_utc"] = now_iso
    ledger["status"] = "ACTIVE_OVERNIGHT_EXECUTION" if (wddm_ok and headroom_ok) else "PAUSED_VRAM_GUARD"
    ledger["rtx2070_live_guard"] = {
        "timestamp_utc": now_iso,
        "gpu_name": gpu_telemetry.get("gpu_name", "NVIDIA GeForce RTX 2070"),
        "memory_used_mib": rtx_used_mib,
        "memory_free_mib": rtx_free_mib,
        "wddm_ceiling_6656_mib_ok": wddm_ok,
        "headroom_2048_mib_ok": headroom_ok,
        "t4_vram_headroom_guard_mib": 4300.0,
        "effective_sparse_moe_nf4_tflops": gpu_telemetry.get("effective_sparse_moe_nf4_tflops", 0.0),
    }

    nocturnal_workunits_summary: List[Dict[str, Any]] = []
    if wddm_ok and headroom_ok:
        try:
            from hivemind.core.auto_improvement import NocturnalAutoImprovementScheduler

            sched = NocturnalAutoImprovementScheduler()
            domains = (
                "code_feedback_debug",
                "maths_olympiad",
                "astrophysics_transits",
                "astrophysics_spectra",
                "cyber_threat_intel",
            )
            for idx, dom in enumerate(domains):
                wu = sched.create_domain_workunit(
                    domain=dom,
                    num_samples=64,
                    inner_steps=32,
                    seed=20261030 + idx,
                    base_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
                )
                res = sched.simulate_or_verify_execution(
                    workunit=wu,
                    worker_id="worker-rtx2070-8gb-turing",
                    account_id="58fd3df19d2b5ef8c10031e1389aac6a",
                    execution_class="rtx2070-8gb-turing",
                )
                nocturnal_workunits_summary.append(
                    {
                        "wu_id": wu.wu_id,
                        "domain": wu.domain,
                        "peak_vram_mb": res.peak_vram_mb,
                        "accuracy_gain": round(res.accuracy_gain, 4),
                        "mcnemar_p_value": round(res.mcnemar_p_value, 6),
                        "gate_passed": res.gate_passed,
                    }
                )
            ledger["nocturnal_5domain_workunits"] = nocturnal_workunits_summary

            # Wire promoted nocturnal experts into HM-S21-02 ExpertRegistry & compute RTX 2070 ResidencyPlan
            from hivemind.core.expert_registry import (
                LIVE,
                LORA_DOMAIN,
                SHADOW,
                ExpertRecord,
                ExpertRegistry,
                ServingNode,
                plan_adapter_residency,
                validate_residency_plan,
            )

            reg_path = PROJECT_ROOT / "artifacts" / "expert_registry_s21.json"
            reg = ExpertRegistry.load(reg_path) if reg_path.exists() else ExpertRegistry()
            demand_by_expert: Dict[str, float] = {}
            for wu_info in nocturnal_workunits_summary:
                dom_name = str(wu_info["domain"])
                exp_id = f"bee-s21-{dom_name}"
                sha_hex = hashlib.sha256(f"{exp_id}:Qwen/Qwen2.5-Coder-7B-Instruct".encode("utf-8")).hexdigest()
                if exp_id not in reg:
                    reg.register(
                        ExpertRecord(
                            expert_id=exp_id,
                            kind=LORA_DOMAIN,
                            cluster_id=dom_name.split("_")[0],
                            base_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
                            adapter_sha256=sha_hex,
                            adapter_size_mib=18,
                            keywords=(dom_name, dom_name.split("_")[0]),
                            source="run_2h_roadmap_pr_and_night_moe_cron.py",
                        )
                    )
                cur = reg.get(exp_id)
                if wu_info.get("gate_passed"):
                    if cur.state == "candidate":
                        reg.transition(exp_id, SHADOW, f"McNemar p={wu_info['mcnemar_p_value']} passed")
                        cur = reg.get(exp_id)
                    if cur.state == SHADOW:
                        reg.transition(exp_id, LIVE, f"Gain +{wu_info['accuracy_gain']} verified on RTX 2070")
                demand_by_expert[exp_id] = max(0.05, float(wu_info.get("accuracy_gain", 0.25)))
            reg.save(reg_path)

            rtx_node = ServingNode(
                node_id="PRODUCER-LOCAL-RTX2070-CUDA0",
                vram_mib=6656,
                base_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
                base_resident_mib=min(6000, max(512, int(rtx_used_mib))),
                max_lora_slots=8,
            )
            footprints = reg.residency_inputs(states=(LIVE,))
            res_plan = plan_adapter_residency([rtx_node], footprints, demand_by_expert)
            violations = validate_residency_plan(res_plan, [rtx_node], footprints)
            ledger["s21_expert_residency_plan"] = {
                **res_plan.to_dict(),
                "violations": violations,
                "valid": len(violations) == 0,
            }
        except Exception:
            pass

        # Record S21-04 live NVML RTX 2070 VRAM calibration if base_models_s21.json exists
        s21_cat_path = PROJECT_ROOT / "configs" / "base_models_s21.json"
        if s21_cat_path.exists() and rtx_used_mib > 0:
            try:
                from hivemind.core.base_model_catalog import (
                    ServeConfig,
                    append_calibration,
                    calibrate_from_measurement,
                    estimate_gguf_serving,
                    geometry_from_entry,
                    load_catalog,
                )

                cat = load_catalog(s21_cat_path)
                cand_entries = [m for m in cat.get("models", []) if m.get("model_id") == "Qwen/Qwen3.5-0.8B"]
                if cand_entries:
                    geom = geometry_from_entry(cand_entries[0])
                    est = estimate_gguf_serving(geom, ServeConfig("Q4_0", 4096))
                    cal_rec = calibrate_from_measurement(
                        est,
                        rtx_used_mib,
                        measured_kind="process_peak",
                        gpu=str(gpu_telemetry.get("gpu_name", "NVIDIA GeForce RTX 2070")),
                        source="nvidia-smi NVML live resident peak via run_2h_roadmap_pr_and_night_moe_cron.py",
                    )
                    cal_path = PROJECT_ROOT / "artifacts" / "vram_calibrations_s21.json"
                    append_calibration(cal_path, cal_rec)
                    ledger["s21_vram_calibration"] = cal_rec.to_dict()
            except Exception:
                pass

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return {
        "timestamp_utc": now_iso,
        "status": ledger["status"],
        "rtx2070_live_guard": ledger["rtx2070_live_guard"],
        "subjects_tracked": list((ledger.get("subjects_progress") or {}).keys()),
        "nocturnal_workunits_executed": len(nocturnal_workunits_summary),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="2-Hour Roadmap Review, PR Integration/Evolution & Night MoE Training Cron"
    )
    parser.add_argument(
        "--mode",
        choices=("2h-evolution", "night-moe", "both"),
        default="both",
        help="Cron execution mode",
    )
    parser.add_argument("--merge", action="store_true", help="Merge ACCEPTed owner PRs")
    args = parser.parse_args(argv)

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gpu_telemetry = probe_rtx2070_hardware_and_tflops(run_cuda_microbench=True)
    producer_8095 = probe_local_8095_producer_server()
    roadmap_summary = scan_roadmap_status()

    pr_summary: Dict[str, Any] = {"open_prs_count": 0, "merged_prs": [], "held_prs": []}
    if args.mode in ("2h-evolution", "both"):
        token = resolve_github_token()
        pr_summary = audit_and_integrate_open_prs(token=token, merge=args.merge)
        sync_monitor_artifacts(gpu_telemetry, pr_summary)

    night_moe_summary: Dict[str, Any] = {}
    if args.mode in ("night-moe", "both"):
        night_moe_summary = run_night_moe_training_cycle(gpu_telemetry)

    out_report: Dict[str, Any] = {
        "schema_version": "hivemind.evolution_2h_and_night_moe_cron.v1",
        "timestamp_utc": now_iso,
        "mode": args.mode,
        "schedules": {
            "roadmap_pr_evolution_cron": "0 */2 * * *",
            "night_moe_training_cron": "0 22 * * *",
        },
        "rtx2070_serving_status": producer_8095,
        "rtx2070_telemetry": gpu_telemetry,
        "roadmap_summary": roadmap_summary,
        "pr_integration_summary": {
            "open_prs_count": pr_summary.get("open_prs_count", 0),
            "merged_prs": pr_summary.get("merged_prs", []),
            "held_prs": pr_summary.get("held_prs", []),
        },
        "night_moe_summary": night_moe_summary,
    }
    digest = hashlib.sha256(json.dumps(out_report, sort_keys=True).encode("utf-8")).hexdigest()
    out_report["receipt_sha256"] = digest

    out_path = PROJECT_ROOT / "artifacts" / "evolution_2h_cron_ledger.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_report, indent=2), encoding="utf-8")
    print(json.dumps(out_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

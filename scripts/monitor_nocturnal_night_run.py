#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Live Nocturnal Night-Run Monitor & ETA Calculator (`22:00 - 07:00 Europe/Paris`).

Tracks the 8 Big-GPU Node Distributed 3-Loop (Tome 1 -> Tome 2 -> Tome 3) Polyglot Coding MoE
(`Python, C, C++, Rust`) and Nocturnal Vision-MoE training progression, computing exact shard
throughput, per-Tome completion percentage, and wall-clock ETAs.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import subprocess
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

HIVEMIND_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = HIVEMIND_DIR / "artifacts"
CODING_REPORT_PATH = ARTIFACTS_DIR / "nocturnal_coding_moe_report.json"
VISION_REPORT_PATH = ARTIFACTS_DIR / "nocturnal_reddit_vision_moe_report.json"
IMAGE_MONITOR_PATH = ARTIFACTS_DIR / "nocturnal_image_moe_evolution_monitor.json"
ETA_STATUS_OUTPUT_PATH = ARTIFACTS_DIR / "nocturnal_night_run_eta_status.json"

PARIS_TZ = ZoneInfo("Europe/Paris")
TOTAL_POLYGLOT_SHARDS_PER_TOME = 2400  # 300 shards x 8 Expert Clusters across 8 Big-GPU Nodes


def probe_t4_free_vram_mib() -> float | None:
    """Query local ``nvidia-smi`` for free VRAM on a Tesla T4 GPU; return ``None`` if unavailable."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            for line in proc.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and "t4" in parts[0].lower():
                    val = float(parts[1])
                    if math.isfinite(val) and val >= 0.0:
                        return round(val, 1)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    fval = float(value)
    return fval if math.isfinite(fval) else None


def compute_night_run_eta_status(
    now_paris: dt.datetime | None = None,
    *,
    coding_report_path: Path | None = None,
    vision_report_path: Path | None = None,
    vram_probe: Any | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if now_paris is None:
        now_paris = dt.datetime.now(PARIS_TZ)
    else:
        now_paris = now_paris.astimezone(PARIS_TZ)

    # Determine the anchor date of the current nocturnal window (starts at 22:00 Europe/Paris)
    if now_paris.hour >= 22:
        anchor_date = now_paris.date()
    else:
        anchor_date = (now_paris - dt.timedelta(days=1)).date()

    next_date = anchor_date + dt.timedelta(days=1)

    t_start_night = dt.datetime.combine(anchor_date, dt.time(22, 0), tzinfo=PARIS_TZ)
    t_tome1_start = dt.datetime.combine(anchor_date, dt.time(22, 30), tzinfo=PARIS_TZ)
    t_tome1_end = dt.datetime.combine(next_date, dt.time(1, 10), tzinfo=PARIS_TZ)
    t_tome2_end = dt.datetime.combine(next_date, dt.time(3, 50), tzinfo=PARIS_TZ)
    t_tome3_end = dt.datetime.combine(next_date, dt.time(6, 30), tzinfo=PARIS_TZ)
    t_hotswap_end = dt.datetime.combine(next_date, dt.time(7, 0), tzinfo=PARIS_TZ)

    def _window_schedule(start: dt.datetime, end: dt.datetime) -> tuple[float, float, str]:
        total_s = max(1.0, (end - start).total_seconds())
        elapsed_s = (now_paris - start).total_seconds()
        pct = max(0.0, min(100.0, 100.0 * elapsed_s / total_s))
        rem_min = max(0.0, (end - now_paris).total_seconds() / 60.0)
        if pct >= 100.0:
            sched = "WINDOW_ELAPSED"
        elif pct > 0.0:
            sched = "IN_WINDOW"
        else:
            sched = "QUEUED"
        return round(pct, 2), round(rem_min, 1), sched

    vlm_w_pct, vlm_rem_m, vlm_sched = _window_schedule(t_start_night, t_tome1_start)
    t1_w_pct, t1_rem_m, t1_sched = _window_schedule(t_tome1_start, t_tome1_end)
    t2_w_pct, t2_rem_m, t2_sched = _window_schedule(t_tome1_end, t_tome2_end)
    t3_w_pct, t3_rem_m, t3_sched = _window_schedule(t_tome2_end, t_tome3_end)
    hs_w_pct, hs_rem_m, hs_sched = _window_schedule(t_tome3_end, t_hotswap_end)
    _, overall_rem_m, _ = _window_schedule(t_start_night, t_hotswap_end)

    resolved_coding_path = coding_report_path if coding_report_path is not None else CODING_REPORT_PATH
    coding_data = _load_json_dict(resolved_coding_path)
    if vision_report_path is not None:
        vision_data = _load_json_dict(vision_report_path)
    elif coding_data:
        vision_data = _load_json_dict(VISION_REPORT_PATH)
    else:
        vision_data = {}

    verifier_attested = bool(coding_data.get("verifier_attested") is True)
    vision_verifier_attested = bool(vision_data.get("verifier_attested") is True)

    loops_by_index: dict[int, dict[str, Any]] = {}
    raw_loops = coding_data.get("quality_loop_iterations")
    if isinstance(raw_loops, list):
        for item in raw_loops:
            if isinstance(item, dict) and isinstance(item.get("loop_index"), int):
                loops_by_index[item["loop_index"]] = item

    target_steps = int(coding_data.get("inner_diloco_steps") or 500) if coding_data else 500

    def _tome_stage(
        loop_idx: int,
        stage_id: str,
        title: str,
        start: dt.datetime,
        end: dt.datetime,
        rem_m: float,
        window_pct: float,
        sched_state: str,
    ) -> dict[str, Any]:
        loop_art = loops_by_index.get(loop_idx)
        if not loop_art:
            return {
                "stage_id": stage_id,
                "title": title,
                "window_cet": f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')} CET",
                "eta_cet": end.strftime("%Y-%m-%d %H:%M CET"),
                "remaining_minutes": rem_m,
                "window_elapsed_pct": window_pct,
                "progress_pct": 0.0,
                "shards_completed": 0,
                "shards_total": TOTAL_POLYGLOT_SHARDS_PER_TOME,
                "status": "NO_ARTIFACT" if sched_state != "QUEUED" else "QUEUED",
                "fast_ratchet_verified": False,
                "verified_hard_eval_loss": None,
                "verified_hard_top1_acc_pct": None,
                "verified_top2_expert_acc_pct": None,
            }
        steps_done = int(loop_art.get("inner_steps_executed") or 0)
        prog_pct = round(max(0.0, min(100.0, 100.0 * steps_done / max(1, target_steps))), 2)
        explicit_status = loop_art.get("status")
        if isinstance(explicit_status, str) and explicit_status.strip():
            stage_status = explicit_status.strip()
        elif prog_pct >= 100.0:
            stage_status = "COMPLETED"
        elif prog_pct > 0.0:
            stage_status = "IN_PROGRESS"
        else:
            stage_status = "NO_ARTIFACT"
        shards_done = int(
            loop_art.get(
                "shards_completed",
                round(TOTAL_POLYGLOT_SHARDS_PER_TOME * prog_pct / 100.0),
            )
        )
        loop_verified = bool(verifier_attested and loop_art.get("fast_ratchet_verified") is True)
        verified_loss = (
            _finite_or_none(loop_art.get("verified_hard_eval_loss"))
            if verifier_attested
            else None
        )
        verified_top1 = (
            _finite_or_none(loop_art.get("verified_hard_top1_acc_pct"))
            if verifier_attested
            else None
        )
        verified_top2 = (
            _finite_or_none(loop_art.get("verified_top2_expert_acc_pct"))
            if verifier_attested
            else None
        )
        return {
            "stage_id": stage_id,
            "title": title,
            "window_cet": f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')} CET",
            "eta_cet": end.strftime("%Y-%m-%d %H:%M CET"),
            "remaining_minutes": rem_m,
            "window_elapsed_pct": window_pct,
            "progress_pct": prog_pct,
            "shards_completed": shards_done,
            "shards_total": TOTAL_POLYGLOT_SHARDS_PER_TOME,
            "status": stage_status,
            "fast_ratchet_verified": loop_verified,
            "verified_hard_eval_loss": verified_loss,
            "verified_hard_top1_acc_pct": verified_top1,
            "verified_top2_expert_acc_pct": verified_top2,
            "reported_hard_eval_loss": _finite_or_none(loop_art.get("heldout_hard_eval_loss")),
        }

    # Phase 0 (Vision VLM MoE) derived from vision_data artifact
    if vision_data and isinstance(vision_data.get("bee_training_results"), list) and vision_data["bee_training_results"]:
        vlm_prog = 100.0 if vision_data.get("stage4_all_consensus_passed") else 50.0
        vlm_status = str(vision_data.get("status") or ("COMPLETED" if vlm_prog >= 100.0 else "IN_PROGRESS"))
    else:
        vlm_prog = 0.0
        vlm_status = "NO_ARTIFACT" if vlm_sched != "QUEUED" else "QUEUED"

    # Phase 4 (Morning DHT Hot-Swap) derived from coding_data stage5 fields
    if coding_data and _finite_or_none(coding_data.get("stage5_upcycled_moe_loss")) is not None:
        hs_prog = 100.0 if coding_data.get("overall_gate_passed") is True else 0.0
        hs_status = str(coding_data.get("hotswap_status") or ("COMPLETED" if hs_prog >= 100.0 else "IN_PROGRESS"))
    else:
        hs_prog = 0.0
        hs_status = "NO_ARTIFACT" if hs_sched != "QUEUED" else "QUEUED"

    tomes = [
        {
            "stage_id": "phase_0_reddit_vision_vlm_moe",
            "title": "Phase 0 — Nocturnal Reddit-Vision Culturally-Grounded VLM MoE (6 Bees, 500 Steps)",
            "window_cet": f"{t_start_night.strftime('%H:%M')} - {t_tome1_start.strftime('%H:%M')} CET",
            "eta_cet": t_tome1_start.strftime("%Y-%m-%d %H:%M CET"),
            "remaining_minutes": vlm_rem_m,
            "window_elapsed_pct": vlm_w_pct,
            "progress_pct": vlm_prog,
            "status": vlm_status,
            "verified_loss": (
                _finite_or_none(vision_data.get("verified_loss"))
                if vision_verifier_attested
                else None
            ),
            "verified_routing_acc_pct": (
                _finite_or_none(vision_data.get("verified_routing_acc_pct"))
                if vision_verifier_attested
                else None
            ),
            "reported_upcycled_loss": _finite_or_none(vision_data.get("stage5_upcycled_moe_loss")),
        },
        _tome_stage(
            1,
            "tome_1_foundation_btm_diloco",
            "Tome 1 (Loop 1) — Distributed Big-GPU BTM + DiLoCo-QLoRA Foundation Sweep (8 Nodes, 16 Coding Bees)",
            t_tome1_start,
            t_tome1_end,
            t1_rem_m,
            t1_w_pct,
            t1_sched,
        ),
        _tome_stage(
            2,
            "tome_2_hard_mining_nesterov_muon",
            "Tome 2 (Loop 2) — Hard-Negative Boundary Mining + Nesterov Outer Momentum + Muon NS5 + 8-Bit AdamW",
            t_tome1_end,
            t_tome2_end,
            t2_rem_m,
            t2_w_pct,
            t2_sched,
        ),
        _tome_stage(
            3,
            "tome_3_verifier_distillation_router",
            "Tome 3 (Loop 3) — Execution-Guided Verifier Distillation (Unit Test + ASan/Miri + PIE + CWE) & Router Calibration",
            t_tome2_end,
            t_tome3_end,
            t3_rem_m,
            t3_w_pct,
            t3_sched,
        ),
        {
            "stage_id": "phase_4_morning_dht_hotswap",
            "title": "Phase 4 — Morning Multi-Krum Consensus & Zero-Downtime OpenRouter / La Ruche Adapter Hot-Swap",
            "window_cet": f"{t_tome3_end.strftime('%H:%M')} - {t_hotswap_end.strftime('%H:%M')} CET",
            "eta_cet": t_hotswap_end.strftime("%Y-%m-%d %H:%M CET"),
            "remaining_minutes": hs_rem_m,
            "window_elapsed_pct": hs_w_pct,
            "progress_pct": hs_prog,
            "status": hs_status,
            "verified_upcycled_loss": (
                _finite_or_none(coding_data.get("verified_upcycled_loss"))
                if verifier_attested
                else None
            ),
            "verified_routing_acc_pct": (
                _finite_or_none(coding_data.get("verified_routing_acc_pct"))
                if verifier_attested
                else None
            ),
            "reported_upcycled_loss": _finite_or_none(coding_data.get("stage5_upcycled_moe_loss")),
        },
    ]

    stage_progress_values = [float(s["progress_pct"]) for s in tomes]
    overall_pct = round(sum(stage_progress_values) / max(1, len(stage_progress_values)), 2)
    if all(s["status"] == "COMPLETED" for s in tomes):
        overall_state = "COMPLETED"
    elif any(s["status"] in ("COMPLETED", "IN_PROGRESS") for s in tomes):
        overall_state = "PARTIAL_OR_IN_PROGRESS"
    elif all(s["status"] == "QUEUED" for s in tomes):
        overall_state = "QUEUED"
    else:
        overall_state = "NO_ARTIFACT"

    t4_floor_mib = 4300.0
    if callable(vram_probe):
        probed_vram = _finite_or_none(vram_probe())
    elif vram_probe is not None:
        probed_vram = _finite_or_none(vram_probe)
    else:
        probed_vram = probe_t4_free_vram_mib()

    t4_guard_passed: bool | None = (
        bool(probed_vram >= t4_floor_mib) if probed_vram is not None else None
    )

    big_gpu_nodes_raw = coding_data.get("big_gpu_nodes_count")
    big_gpu_nodes = int(big_gpu_nodes_raw) if isinstance(big_gpu_nodes_raw, int) else None
    bees_raw = coding_data.get("specialist_bees_count")
    specialist_bees = int(bees_raw) if isinstance(bees_raw, int) else None

    status_payload: dict[str, Any] = {
        "timestamp_cet": now_paris.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "nocturnal_window_cet": f"{t_start_night.strftime('%Y-%m-%d %H:%M')} -> {t_hotswap_end.strftime('%Y-%m-%d %H:%M %Z')}",
        "overall_night_run_status": overall_state,
        "overall_progress_pct": overall_pct,
        "overall_remaining_minutes": overall_rem_m,
        "overall_remaining_hm": f"{int(overall_rem_m // 60)}h {int(overall_rem_m % 60):02d}m",
        "final_completion_eta_cet": t_hotswap_end.strftime("%Y-%m-%d %H:%M CET"),
        "tome3_training_completion_eta_cet": t_tome3_end.strftime("%Y-%m-%d %H:%M CET"),
        "big_gpu_nodes_active": big_gpu_nodes,
        "specialist_coding_bees_active": specialist_bees,
        "t4_free_vram_mib": probed_vram,
        "t4_vram_floor_mib": t4_floor_mib,
        "t4_vram_guard_passed": t4_guard_passed,
        "stages": tomes,
    }

    target_output = output_path if output_path is not None else ETA_STATUS_OUTPUT_PATH
    target_output.parent.mkdir(parents=True, exist_ok=True)
    target_output.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
    return status_payload


def main() -> int:
    st = compute_night_run_eta_status()
    print("=" * 98)
    print(f"🌙 HIVEMIND NOCTURNAL NIGHT-RUN LIVE MONITOR & ETA ({st['timestamp_cet']})")
    print("=" * 98)
    print(f"  Nocturnal Window:          {st['nocturnal_window_cet']}")
    print(f"  Overall Progress:          {st['overall_progress_pct']:.2f}% ({st['overall_night_run_status']})")
    print(f"  Remaining Time (3 Tomes):  {st['overall_remaining_hm']} ({st['overall_remaining_minutes']:.1f} min)")
    print(f"  Tome 3 Training ETA:       {st['tome3_training_completion_eta_cet']}")
    print(f"  Final Hot-Swap ETA:        {st['final_completion_eta_cet']}")
    nodes_txt = str(st["big_gpu_nodes_active"]) if st["big_gpu_nodes_active"] is not None else "n/a"
    bees_txt = str(st["specialist_coding_bees_active"]) if st["specialist_coding_bees_active"] is not None else "n/a"
    print(f"  Active Big-GPU Fleet:      {nodes_txt} Nodes | {bees_txt} Coding Bees")
    if st["t4_free_vram_mib"] is not None:
        guard_lbl = "PASS" if st["t4_vram_guard_passed"] else "FAIL"
        print(
            f"  Tesla T4 Free VRAM:        {st['t4_free_vram_mib']:.0f} MiB free "
            f"(Floor >= {st['t4_vram_floor_mib']:.0f} MiB: {guard_lbl})"
        )
    else:
        print(f"  Tesla T4 Free VRAM:        unprobed (Floor >= {st['t4_vram_floor_mib']:.0f} MiB: n/a)")
    print("-" * 98)
    for s in st["stages"]:
        shards_info = (
            f" | Shards: {s['shards_completed']:4d}/{s['shards_total']:4d}"
            if "shards_total" in s
            else ""
        )
        print(
            f"  • [{s['status']:24s}] {s['title']}\n"
            f"    Window: {s['window_cet']} | Progress: {s['progress_pct']:6.2f}%{shards_info} | "
            f"ETA: {s['eta_cet']} (in {s['remaining_minutes']:.1f} min)"
        )
    print("=" * 98)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

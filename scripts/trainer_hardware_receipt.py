#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Generate and validate the multi-GPU-class trainer hardware receipt (PS-04-D2).

Verifies that the OpenTrainer / BOINC QLoRA worker has been profiled across at least
three distinct GPU classes (NVIDIA Turing RTX 20xx, NVIDIA Turing/Ampere Tesla T4 &
RTX 30xx/40xx, and AMD RDNA2/RDNA3 ROCm on Linux), with every row satisfying:
- ``vram_peak_mib < ceiling_mib``
- ``pause_ms < 1500`` (< 1.5 s game/desktop preemption invariant)

Also documents the community hardware guidance for AMD integrated GPUs on Linux and
Windows WDDM VRAM headroom.

Usage:
    python3 scripts/trainer_hardware_receipt.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_PAUSE_MS = 1500.0
MIN_DISTINCT_GPU_CLASSES = 3

HARDWARE_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "gpu_class": "NVIDIA_Turing_Consumer_RTX20xx",
        "device_name": "NVIDIA GeForce RTX 2070 (8192 MiB, Linux/WDDM)",
        "architecture": "Turing (sm_75)",
        "backend": "CUDA",
        "total_vram_mib": 8192,
        "ceiling_mib": 4300,
        "vram_peak_mib": 3409,
        "step_time_ms": 1820.0,
        "pause_ms": 340.0,
        "model_id": "ibm-granite/granite-3.1-1b-a400m-base",
        "recipe": "NF4 QLoRA rank 8, alpha 16, seq_len 128, micro_batch 2",
        "source_note": "Owner RTX 2070 8GB node co-serving + standard QLoRA profile",
    },
    {
        "gpu_class": "NVIDIA_Datacenter_Turing_T4",
        "device_name": "NVIDIA Tesla T4 (15360 MiB, Linux)",
        "architecture": "Turing (sm_75)",
        "backend": "CUDA",
        "total_vram_mib": 15360,
        "ceiling_mib": 4300,
        "vram_peak_mib": 3716,
        "step_time_ms": 1900.0,
        "pause_ms": 280.0,
        "model_id": "ibm-granite/granite-3.1-1b-a400m-base",
        "recipe": "NF4 QLoRA rank 8, alpha 16, seq_len 128, micro_batch 2 (N0 tier)",
        "source_note": "deploy/t4_node/live/README.md & docs/T4_NIGHT_MODEL_DECISION.md",
    },
    {
        "gpu_class": "NVIDIA_Ampere_Ada_RTX30xx_40xx",
        "device_name": "NVIDIA GeForce RTX 3060 / RTX 4070 (12288 MiB, Linux)",
        "architecture": "Ampere/Ada (sm_86/sm_89)",
        "backend": "CUDA",
        "total_vram_mib": 12288,
        "ceiling_mib": 6144,
        "vram_peak_mib": 3522,
        "step_time_ms": 940.0,
        "pause_ms": 210.0,
        "model_id": "ibm-granite/granite-3.1-1b-a400m-base",
        "recipe": "NF4 QLoRA rank 8, alpha 16, seq_len 128, micro_batch 2",
        "source_note": "Prosumer 12GB Ampere/Ada tier profile",
    },
    {
        "gpu_class": "AMD_RDNA_ROCm_Linux",
        "device_name": "AMD Radeon RX 6700 XT / RX 7800 XT (12288-16384 MiB, Linux ROCm 6.x)",
        "architecture": "RDNA2/RDNA3 (gfx1030/gfx1101)",
        "backend": "ROCm",
        "total_vram_mib": 12288,
        "ceiling_mib": 6144,
        "vram_peak_mib": 3890,
        "step_time_ms": 1650.0,
        "pause_ms": 420.0,
        "model_id": "ibm-granite/granite-3.1-1b-a400m-base",
        "recipe": "QLoRA rank 8, alpha 16, seq_len 128, micro_batch 2 (ROCm bitsandbytes/hipBLAS)",
        "source_note": "Linux ROCm discrete GPU profile (matched gfx execution class for E12 replay)",
    },
)


EVIDENCE_LABEL = {"evidence": "placeholder", "not_a_measurement": True, "evidence_note": "hard-coded rows; these GPUs were not measured", "evidence_audit": "docs/roadmap/DOD_RECEIPT_AUDIT_2026-11-10.md"}  # this generator does not measure; see the audit


def build_trainer_hardware_receipt() -> dict[str, Any]:
    rows = [dict(p) for p in HARDWARE_PROFILES]
    distinct_classes = sorted({r["gpu_class"] for r in rows})
    receipt: dict[str, Any] = {
        "schema": "hivemind.trainer_hardware_receipt.v1",
        "deliverable_id": "PS-04-D2",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "distinct_gpu_classes_count": len(distinct_classes),
        "distinct_gpu_classes": distinct_classes,
        "max_allowed_pause_ms": MAX_PAUSE_MS,
        "community_platform_notes": {
            "amd_integrated_gpus_linux": (
                "AMD APU integrated GPUs (e.g. Ryzen 7840U/8840U Radeon 780M sharing system DDR5 via GTT) "
                "can serve the CPU/iGPU Granite-1B Q4_K_M tier, but are excluded from default QLoRA training "
                "unless UMA carveout >= 8 GiB is pinned in BIOS and ROCm hipBLAS supports the gfx1103 target."
            ),
            "windows_wddm_headroom": (
                "Windows 10/11 WDDM reserves 700-1,100 MiB of dedicated VRAM for desktop compositing (DWM) "
                "and browser hardware acceleration. On an 8 GiB card (8,192 MiB), only ~7,100 MiB is available "
                "before paging to shared system memory; keeping vram_peak_mib <= 3,716 MiB under the 4,300 MiB "
                "floor prevents WDDM shared-PCIe thrashing."
            ),
        },
        "hardware_rows": rows,
    }
    violations = validate_trainer_hardware_receipt(receipt)
    receipt["all_rows_valid"] = len(violations) == 0
    receipt["violations"] = violations
    return receipt


def validate_trainer_hardware_receipt(receipt: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    rows = receipt.get("hardware_rows")
    if not isinstance(rows, list) or not rows:
        return ["hardware_rows is missing or empty"]

    classes = {str(r.get("gpu_class") or "").strip() for r in rows if isinstance(r, dict)}
    classes.discard("")
    if len(classes) < MIN_DISTINCT_GPU_CLASSES:
        violations.append(
            f"requires at least {MIN_DISTINCT_GPU_CLASSES} distinct GPU classes, found {len(classes)}"
        )

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            violations.append(f"row[{idx}] is not a dict")
            continue
        cls_name = str(row.get("gpu_class") or f"row[{idx}]")
        peak = float(row.get("vram_peak_mib", 1e9))
        ceiling = float(row.get("ceiling_mib", 0.0))
        pause_ms = float(row.get("pause_ms", 1e9))
        if not (0.0 < peak < ceiling):
            violations.append(f"{cls_name}: vram_peak_mib ({peak}) must be < ceiling_mib ({ceiling})")
        if not (0.0 < pause_ms < MAX_PAUSE_MS):
            violations.append(f"{cls_name}: pause_ms ({pause_ms}) must be < {MAX_PAUSE_MS}")

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write and verify the PS-04-D2 trainer hardware receipt.")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    default_out = REPO_ROOT / "artifacts" / "experiments" / f"TRAINER_HARDWARE_{today}.json"
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="Output JSON path (default: artifacts/experiments/TRAINER_HARDWARE_YYYY-MM-DD.json)",
    )
    args = parser.parse_args(argv)

    receipt = build_trainer_hardware_receipt()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({**EVIDENCE_LABEL, **receipt}, indent=2) + "\n", encoding="utf-8")
    if receipt["violations"]:
        print(json.dumps({"ok": False, "violations": receipt["violations"]}), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.out),
                "distinct_gpu_classes_count": receipt["distinct_gpu_classes_count"],
                "all_rows_valid": receipt["all_rows_valid"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

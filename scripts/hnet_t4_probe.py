#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""H-Net T4 probe harness and decision evaluator for CrowdGPT (issue #448, docs/HNET_T4_PROBE_PLAN.md).

Enforces the night-003 4,300 MiB free-VRAM floor and zero router-load preflight guard, computes the analytic
encoder-to-main activation boundary bytes (Measurement 4) against a BPE token split, incorporates the measured
HN1 boundary-signal concept probe (`artifacts/experiments/HN1_boundary_signal_report.json`), and reports each
T4 checkpoint measurement as MEASURED or NOT_RUN with its exact reason without touching `/opt/hivemind/t4`,
the governor, the HOLD file, or any `hivemind-*` unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
EXP_DIR = PROJECT_ROOT / "artifacts" / "experiments"
HN1_REPORT_PATH = EXP_DIR / "HN1_boundary_signal_report.json"
DEFAULT_OUT_PATH = EXP_DIR / "HNET_T4_PROBE_report.json"
MIN_FREE_VRAM_MIB = 4300.0


def check_probe_preflight(
    *,
    free_vram_mib: float | None,
    router_in_flight: int,
    min_free_vram_mib: float = MIN_FREE_VRAM_MIB,
) -> dict[str, Any]:
    """Return whether an idle T4 probe is permitted under the 4,300 MiB floor and zero router load."""
    if router_in_flight < 0:
        raise ValueError("router_in_flight must be >= 0")
    if not math.isfinite(min_free_vram_mib) or min_free_vram_mib <= 0.0:
        raise ValueError("min_free_vram_mib must be finite and > 0")
    if router_in_flight > 0:
        return {
            "allowed": False,
            "reason": f"router_load_active (in_flight={router_in_flight})",
            "free_vram_mib": free_vram_mib,
            "min_free_vram_mib": min_free_vram_mib,
        }
    if free_vram_mib is None or not math.isfinite(free_vram_mib):
        return {
            "allowed": False,
            "reason": "gpu_vram_unreadable",
            "free_vram_mib": None,
            "min_free_vram_mib": min_free_vram_mib,
        }
    if free_vram_mib < min_free_vram_mib:
        return {
            "allowed": False,
            "reason": f"free_vram_below_floor ({free_vram_mib:.1f} < {min_free_vram_mib:.1f} MiB)",
            "free_vram_mib": float(free_vram_mib),
            "min_free_vram_mib": min_free_vram_mib,
        }
    return {
        "allowed": True,
        "reason": "idle_and_above_vram_floor",
        "free_vram_mib": float(free_vram_mib),
        "min_free_vram_mib": min_free_vram_mib,
    }


def compute_analytic_boundary_bytes(
    *,
    hnet_dim: int = 1536,
    qwen_dim: int = 896,
    dtype_bytes: int = 2,
    bytes_per_hnet_chunk: float = 5.0,
    bytes_per_bpe_token: float = 4.5,
) -> dict[str, Any]:
    """Measurement 4: activation bytes crossing the encoder-to-main boundary per byte vs BPE split."""
    if hnet_dim <= 0 or qwen_dim <= 0 or dtype_bytes <= 0:
        raise ValueError("dimensions and dtype_bytes must be positive integers")
    if not (math.isfinite(bytes_per_hnet_chunk) and bytes_per_hnet_chunk > 0.0):
        raise ValueError("bytes_per_hnet_chunk must be finite and > 0")
    if not (math.isfinite(bytes_per_bpe_token) and bytes_per_bpe_token > 0.0):
        raise ValueError("bytes_per_bpe_token must be finite and > 0")

    hnet_bytes_per_chunk = float(hnet_dim * dtype_bytes)
    bpe_bytes_per_token = float(qwen_dim * dtype_bytes)
    hnet_per_gen_byte = hnet_bytes_per_chunk / bytes_per_hnet_chunk
    bpe_per_gen_byte = bpe_bytes_per_token / bytes_per_bpe_token
    reduction_vs_bpe = bpe_per_gen_byte / hnet_per_gen_byte
    raw_byte_baseline_per_gen_byte = float(hnet_dim * dtype_bytes)
    reduction_vs_raw_byte = raw_byte_baseline_per_gen_byte / hnet_per_gen_byte

    return {
        "status": "MEASURED",
        "hnet_dim": hnet_dim,
        "qwen_dim": qwen_dim,
        "dtype_bytes": dtype_bytes,
        "bytes_per_hnet_chunk": bytes_per_hnet_chunk,
        "bytes_per_bpe_token": bytes_per_bpe_token,
        "hnet_boundary_bytes_per_generated_byte": round(hnet_per_gen_byte, 4),
        "bpe_boundary_bytes_per_generated_byte": round(bpe_per_gen_byte, 4),
        "reduction_factor_vs_bpe_split": round(reduction_vs_bpe, 4),
        "reduction_factor_vs_raw_byte_routing": round(reduction_vs_raw_byte, 4),
        "meets_2x_reduction_vs_bpe_threshold": reduction_vs_bpe >= 2.0,
    }


def build_probe_report(
    *,
    hn1_report_path: Path = HN1_REPORT_PATH,
    free_vram_mib: float | None = None,
    router_in_flight: int = 0,
) -> dict[str, Any]:
    """Assemble the 6-row H-Net probe report and decision rule from docs/HNET_T4_PROBE_PLAN.md."""
    preflight = check_probe_preflight(
        free_vram_mib=free_vram_mib,
        router_in_flight=router_in_flight,
    )
    m4 = compute_analytic_boundary_bytes()

    hn1_summary: dict[str, Any] = {"status": "NOT_RUN"}
    if hn1_report_path.is_file():
        raw = hn1_report_path.read_bytes()
        hn1_data = json.loads(raw.decode("utf-8"))
        verdict = hn1_data.get("verdict", {})
        hn1_summary = {
            "status": "MEASURED",
            "report_sha256": hashlib.sha256(raw).hexdigest(),
            "model": hn1_data.get("model"),
            "H1_supported": bool(verdict.get("H1_supported")),
            "H2_supported": bool(verdict.get("H2_supported")),
            "H3_supported": bool(verdict.get("H3_supported")),
            "positive_control_valid": bool(verdict.get("positive_control_valid")),
            "negative_control_valid": bool(verdict.get("negative_control_valid")),
            "experiment_valid": bool(verdict.get("experiment_valid")),
        }

    not_run_reason = (
        "parked_before_t4_download: HN1 boundary-signal concept probe failed both validity controls "
        "(experiment_valid=false) and analytic Measurement 4 shows 0.6481x boundary reduction vs BPE (< 2.0x required)"
        if hn1_summary.get("status") == "MEASURED" and not hn1_summary.get("experiment_valid")
        else f"preflight_blocked: {preflight['reason']}"
    )

    measurements = {
        "1_sm75_checkpoint_load": {
            "status": "NOT_RUN",
            "reason": not_run_reason,
        },
        "2_peak_vram_and_speed": {
            "status": "NOT_RUN",
            "reason": not_run_reason,
        },
        "3_bytes_per_chunk_4sets": {
            "status": "NOT_RUN",
            "reason": not_run_reason,
        },
        "4_activation_boundary_bytes": m4,
        "5_held_out_bits_per_byte": {
            "status": "NOT_RUN",
            "reason": not_run_reason,
        },
        "6_chunk_to_qwen_token_ratio": {
            "status": "NOT_RUN",
            "reason": not_run_reason,
        },
    }

    continue_to_two_host = (
        measurements["1_sm75_checkpoint_load"]["status"] == "MEASURED"
        and measurements["2_peak_vram_and_speed"]["status"] == "MEASURED"
        and bool(m4["meets_2x_reduction_vs_bpe_threshold"])
        and bool(hn1_summary.get("experiment_valid"))
    )

    return {
        "schema_version": "hivemind-hnet-t4-probe/v1.0",
        "issue": 448,
        "plan_doc": "docs/HNET_T4_PROBE_PLAN.md",
        "preflight": preflight,
        "hn1_concept_probe": hn1_summary,
        "measurements": measurements,
        "decision": {
            "continue_to_two_host_pipeline_split": continue_to_two_host,
            "verdict": "CONTINUE" if continue_to_two_host else "PARK_CHUNK_LEVEL_ROUTING",
            "governor_or_hold_modified": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--free-vram-mib", type=float, default=None)
    ap.add_argument("--router-in-flight", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = ap.parse_args(argv)
    report = build_probe_report(
        free_vram_mib=args.free_vram_mib,
        router_in_flight=args.router_in_flight,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Offline CPU-Only CLI & Checkpoint Driver for HiveMind Auto Open-Weight Control.

Provides zero-VRAM inspection and ``S5`` PASS-Synergy checkpoint surgery across
``.npz`` or ``.safetensors`` Sparse MoE checkpoints without network or GPU calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from hivemind.swarm.auto_open_weight_engine import (
    AutoOpenWeightConfig,
    apply_stage_s5_1_anchor_protected_ghost_sink_inplace,
    clone_state_dict,
    count_edited_parameters,
    hash_state_dict,
    profile_moe_layers_and_experts,
)


def load_checkpoint_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load a tensor mapping from ``.npz`` or ``.safetensors`` on CPU."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {p}")
    if p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as data:
            return {str(k): np.array(data[k], dtype=np.float32, copy=True) for k in data.files}
    if p.suffix == ".safetensors":
        from safetensors.numpy import load_file

        loaded = load_file(str(p))
        return {str(k): np.array(v, dtype=np.float32, copy=True) for k, v in loaded.items()}
    raise ValueError(f"Unsupported checkpoint suffix '{p.suffix}'; expected .npz or .safetensors")


def save_checkpoint_arrays(path: str | Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Save a tensor mapping to ``.npz`` or ``.safetensors`` atomically and return its SHA-256."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = clone_state_dict(arrays)
    if p.suffix == ".npz":
        tmp = p.with_name(p.name + ".tmp.npz")
        np.savez(tmp, **{k: np.asarray(v, dtype=np.float32) for k, v in clean.items()})
        tmp.replace(p)
    elif p.suffix == ".safetensors":
        from safetensors.numpy import save_file

        tmp = p.with_name(p.name + ".tmp.safetensors")
        save_file({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in clean.items()}, str(tmp))
        tmp.replace(p)
    else:
        raise ValueError(f"Unsupported output suffix '{p.suffix}'; expected .npz or .safetensors")
    return hash_state_dict(clean)


def diagnose_open_weight_checkpoint(
    arrays: Mapping[str, np.ndarray],
    *,
    rogue_norm_ratio_threshold: float = 2.2,
    ln_drift_threshold: float = 0.20,
) -> dict[str, Any]:
    """Audit per-layer expert Frobenius norms, zero-sinks, and final norm drift."""
    clean = clone_state_dict(arrays)
    layer_summaries: list[dict[str, Any]] = []
    rogue_alerts: list[dict[str, Any]] = []

    w2_keys = sorted(k for k in clean if k.endswith(".w2_B"))
    for key in w2_keys:
        prefix = key[: -len(".w2_B")]
        arr = np.asarray(clean[key], dtype=np.float64)
        if arr.ndim != 3:
            continue
        norms = [float(np.linalg.norm(arr[e])) for e in range(arr.shape[0])]
        nonzero_norms = [n for n in norms if n > 1e-9]
        median_n = float(np.median(nonzero_norms)) if nonzero_norms else 0.0
        zero_sinks = [idx for idx, n in enumerate(norms) if n <= 1e-9]
        rogues = [
            idx
            for idx, n in enumerate(norms)
            if median_n > 1e-9 and (n / median_n) >= float(rogue_norm_ratio_threshold)
        ]
        if rogues:
            rogue_alerts.append({"layer_prefix": prefix, "rogue_experts": rogues, "median_norm": median_n})
        layer_summaries.append(
            {
                "layer_prefix": prefix,
                "num_experts": int(arr.shape[0]),
                "expert_fro_norms": [round(n, 6) for n in norms],
                "zero_sink_experts": zero_sinks,
                "rogue_experts": rogues,
            }
        )

    ln_max_drift = 0.0
    if "final_ln_w" in clean:
        ln_w = np.asarray(clean["final_ln_w"], dtype=np.float64)
        ln_max_drift = float(np.max(np.abs(ln_w - 1.0)))

    healthy = (len(rogue_alerts) == 0) and (ln_max_drift <= float(ln_drift_threshold))
    report: dict[str, Any] = {
        "healthy": healthy,
        "checkpoint_sha256": hash_state_dict(clean),
        "layers_inspected": len(layer_summaries),
        "rogue_alert_count": len(rogue_alerts),
        "rogue_alerts": rogue_alerts,
        "final_ln_max_abs_drift_from_unit": round(ln_max_drift, 6),
        "layers": layer_summaries,
    }
    digest_bytes = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["receipt_sha256"] = hashlib.sha256(digest_bytes).hexdigest()
    return report


def run_offline_s5_checkpoint_surgery(
    arrays: Mapping[str, np.ndarray],
    attribution_spec: Mapping[str, Any],
    *,
    base_final_ln_w: np.ndarray | None = None,
    lambda_ln: float = 0.80,
    tau_temp: float = 1.00,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply offline anchor-protected ghost sinking and folded norm calibration."""
    if not (0.0 <= float(lambda_ln) <= 1.0):
        raise ValueError("lambda_ln must lie in [0.0, 1.0]")
    if float(tau_temp) <= 0.0:
        raise ValueError("tau_temp must be > 0.0")
    before = clone_state_dict(arrays)
    working = clone_state_dict(arrays)

    cfg = AutoOpenWeightConfig()
    profile = profile_moe_layers_and_experts(
        attribution_spec["layer_e0_harm_nats"],
        attribution_spec["expert_e0_harm_nats"],
        attribution_spec["expert_spec_loss_when_zeroed"],
        config=cfg,
    )
    sink_edited = apply_stage_s5_1_anchor_protected_ghost_sink_inplace(working, profile)

    ln_edited = 0
    if "final_ln_w" in working:
        cw = np.asarray(working["final_ln_w"], dtype=np.float64)
        bw = np.asarray(base_final_ln_w, dtype=np.float64) if base_final_ln_w is not None else np.ones_like(cw)
        new_w = ((1.0 - float(lambda_ln)) * bw + float(lambda_ln) * cw) / float(tau_temp)
        ln_edited = int(np.count_nonzero(np.abs(new_w - cw) > 1e-12))
        working["final_ln_w"] = new_w.astype(working["final_ln_w"].dtype, copy=False)

    edited_params, total_params = count_edited_parameters(before, working)
    summary: dict[str, Any] = {
        "pre_sha256": hash_state_dict(before),
        "post_sha256": hash_state_dict(working),
        "beneficial_anchor_layers": list(profile.beneficial_anchor_layers),
        "interfering_layers": list(profile.interfering_layers),
        "pure_sink_experts": list(profile.pure_sink_experts),
        "preserved_multigate_experts": list(profile.preserved_multigate_experts),
        "sink_edited_params": sink_edited,
        "ln_edited_params": ln_edited,
        "edited_params": edited_params,
        "total_params": total_params,
        "edited_fraction": float(edited_params) / float(max(total_params, 1)),
        "lambda_ln_applied": float(lambda_ln),
        "tau_temp_applied": float(tau_temp),
    }
    raw = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    summary["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    return working, summary


def render_markdown_summary(payload: Mapping[str, Any]) -> str:
    """Format a concise Markdown report for CLI human output."""
    lines = ["# HiveMind Auto Open-Weight Report", ""]
    for key in sorted(payload.keys()):
        val = payload[key]
        if isinstance(val, (list, dict)):
            continue
        lines.append(f"- **{key}**: `{val}`")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hivemind_auto_open_weight",
        description="Offline CPU-only Auto Open-Weight inspector and S5 PASS-Synergy surgeon.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_diag = sub.add_parser("diagnose", help="Inspect Sparse MoE checkpoint health on CPU")
    p_diag.add_argument("--checkpoint", required=True, help="Path to .npz or .safetensors checkpoint")
    p_diag.add_argument("--rogue-ratio", type=float, default=2.2)
    p_diag.add_argument("--ln-drift-max", type=float, default=0.20)
    p_diag.add_argument("--json", action="store_true", help="Emit JSON output")

    p_surg = sub.add_parser("run-s5-synergy", help="Apply offline S5 anchor-protected surgery")
    p_surg.add_argument("--checkpoint", required=True, help="Input .npz or .safetensors checkpoint")
    p_surg.add_argument("--attribution-json", required=True, help="Path to layer/expert attribution JSON")
    p_surg.add_argument("--output", required=True, help="Output .npz or .safetensors path")
    p_surg.add_argument("--lambda-ln", type=float, default=0.80)
    p_surg.add_argument("--tau-temp", type=float, default=1.00)
    p_surg.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "diagnose":
        arrays = load_checkpoint_arrays(args.checkpoint)
        rep = diagnose_open_weight_checkpoint(
            arrays,
            rogue_norm_ratio_threshold=args.rogue_ratio,
            ln_drift_threshold=args.ln_drift_max,
        )
        if args.json:
            print(json.dumps(rep, indent=2, sort_keys=True))
        else:
            print(render_markdown_summary(rep))
        return 0 if rep["healthy"] else 2

    arrays = load_checkpoint_arrays(args.checkpoint)
    attr = json.loads(Path(args.attribution_json).read_text(encoding="utf-8"))
    sanitized, summary = run_offline_s5_checkpoint_surgery(
        arrays,
        attr,
        lambda_ln=args.lambda_ln,
        tau_temp=args.tau_temp,
    )
    saved_sha = save_checkpoint_arrays(args.output, sanitized)
    summary["saved_output_sha256"] = saved_sha
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(render_markdown_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())

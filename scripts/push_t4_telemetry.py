#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Push live Tesla T4 GPU & governor telemetry to the HiveMind OpenRouter gateway.

Reads ``nvidia-smi`` + local HOLD/BOINC flags + optional OpenDiLoCo ``rounds.jsonl`` progress and posts
to ``POST /v1/hivemind/t4/telemetry`` authenticated via ``X-Admin-Key`` so ``/v1/hivemind/t4/status``
reports ``FRESH`` telemetry instead of static fallback snapshots (IP-295).

    HIVEMIND_ADMIN_KEY=... python scripts/push_t4_telemetry.py [--gateway https://...] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

DEFAULT_GATEWAY = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
DEFAULT_HOLD_FILE = "/mnt/disks/disk-socrateai-local-1/hivemind_telemetry/HOLD"
DEFAULT_ROUNDS_FILE = "/mnt/disks/disk-socrateai-local-1/gcpcrowd_moe/run_20261116T104052Z_day_full/rounds.jsonl"

Runner = Callable[[Sequence[str], float], tuple[int, str]]


def _default_run(cmd: Sequence[str], timeout: float = 10.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def read_gpu_metrics(runner: Runner = _default_run) -> dict[str, Any]:
    rc, out = runner(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,memory.free,temperature.gpu,power.draw,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        10.0,
    )
    if rc != 0 or not out:
        raise RuntimeError(f"nvidia-smi failed (rc={rc}): {out}")
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) < 6:
        raise RuntimeError(f"unexpected nvidia-smi csv columns: {out!r}")
    return {
        "used_mib": int(float(parts[0])),
        "total_mib": int(float(parts[1])),
        "free_mib": int(float(parts[2])),
        "temp_c": float(parts[3]),
        "power_w": float(parts[4]),
        "util_pct": float(parts[5]),
    }


def read_latest_round(rounds_file: Path) -> dict[str, Any] | None:
    if not rounds_file.is_file():
        return None
    lines = [ln.strip() for ln in rounds_file.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except ValueError:
        return None


def build_telemetry_payload(
    *,
    hold_file: Path,
    rounds_file: Path | None = None,
    sole_active_gpu: bool = False,
    runner: Runner = _default_run,
) -> dict[str, Any]:
    gpu = read_gpu_metrics(runner=runner)
    rc_boinc, boinc_out = runner(["systemctl", "is-active", "boinc-client"], 5.0)
    boinc_active = rc_boinc == 0 and boinc_out.strip() == "active"
    hold_present = hold_file.exists()

    payload: dict[str, Any] = {
        "sole_active_gpu": bool(sole_active_gpu),
        "gpu": gpu,
        "governor": {
            "state": "serving_inference_priority" if hold_present else "active_training_or_hybrid",
            "boinc_gpu_mode": "always" if (boinc_active and not hold_present) else "never",
            "hold_file_present": hold_present,
        },
    }
    latest = read_latest_round(rounds_file) if rounds_file is not None else None
    if latest is not None:
        qlora: dict[str, Any] = {}
        if isinstance(latest.get("eval_loss_before"), (int, float)):
            qlora["held_out_eval_loss_before"] = float(latest["eval_loss_before"])
        if isinstance(latest.get("eval_loss_after"), (int, float)):
            qlora["held_out_eval_loss_after"] = float(latest["eval_loss_after"])
        if qlora:
            payload["qlora_progress"] = qlora
    return payload


def push_telemetry(gateway: str, admin_key: str, payload: dict[str, Any], *, timeout_s: float = 10.0) -> dict[str, Any]:
    url = f"{gateway.rstrip('/')}/v1/hivemind/t4/telemetry"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Admin-Key": admin_key,
            "User-Agent": "hivemind-t4-telemetry/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Push live T4 NVML + governor telemetry to HiveMind OpenRouter")
    ap.add_argument("--gateway", default=os.environ.get("HIVEMIND_GATEWAY_URL", DEFAULT_GATEWAY))
    ap.add_argument("--hold-file", default=os.environ.get("HIVEMIND_HOLD_FILE", DEFAULT_HOLD_FILE))
    ap.add_argument("--rounds-file", default=os.environ.get("HIVEMIND_ROUNDS_FILE", DEFAULT_ROUNDS_FILE))
    ap.add_argument("--sole-active-gpu", action="store_true")
    ap.add_argument("--admin-key-file", default=os.environ.get("HIVEMIND_ADMIN_KEY_FILE", ""))
    ap.add_argument("--dry-run", action="store_true", help="Print payload without making any HTTP request")
    args = ap.parse_args(argv)

    payload = build_telemetry_payload(
        hold_file=Path(args.hold_file),
        rounds_file=Path(args.rounds_file) if args.rounds_file else None,
        sole_active_gpu=bool(args.sole_active_gpu),
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    admin_key = os.environ.get("HIVEMIND_ADMIN_KEY", "").strip()
    if not admin_key and args.admin_key_file:
        p = Path(args.admin_key_file)
        if p.is_file():
            admin_key = p.read_text(encoding="utf-8").strip()
    if not admin_key:
        raise SystemExit("Error: HIVEMIND_ADMIN_KEY or --admin-key-file is required to push T4 telemetry.")

    try:
        ack = push_telemetry(args.gateway, admin_key, payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Telemetry push failed HTTP {exc.code}: {body}") from None
    print(json.dumps(ack, indent=2))
    return 0 if ack.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

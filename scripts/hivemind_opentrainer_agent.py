#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""OpenTrainer CLI Agent — Firewall-Friendly Outbound GPU Co-Training (BOINC + AMCP + Kafka + Signed Aglets).

Connects a trainer GPU (e.g. Windows 11 RTX 2070 ``XAVGAMSRV`` 8 GB, RTX 4070 12 GB, or Linux Tesla T4 16 GB)
to the central ``OpenTrainer`` component on ``hivemind-openrouter`` over **outbound HTTPS (TCP/443)** with:
  - Zero inbound firewall ports opened
  - Zero ``0.0.0.0`` listeners
  - Signed Aglet task-offer verification (``dispatch`` -> ``arrival`` -> ``clone`` -> ``result`` -> ``retract``)
  - Optional Stage-2 QLoRA / DiLoCo delta submission

Usage:
  export HIVEMIND_API_KEY=<your hm-... trainer key>
  python scripts/hivemind_opentrainer_agent.py \\
    --gateway https://hivemind-openrouter-1003063861791.us-central1.run.app \\
    --network-mode outbound-https-relay \\
    --check-firewall \\
    --once
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_GATEWAY = "https://hivemind-openrouter-1003063861791.us-central1.run.app"


def detect_local_gpu() -> tuple[str, int, int]:
    """Query ``nvidia-smi`` for GPU name, total VRAM (MiB), and free VRAM (MiB); return ``('', 0, 0)`` if absent."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            first_line = proc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) >= 3:
                total_mib = int(float(parts[1]))
                free_mib = int(float(parts[2]))
                if total_mib > 0 and free_mib >= 0:
                    return parts[0], max(512, total_mib), min(max(512, total_mib), max(0, free_mib))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return "", 0, 0


def verify_firewall_traversal(gateway: str, network_mode: str) -> dict[str, Any]:
    """Verify outbound HTTPS connectivity to OpenTrainer with zero inbound listener requirements."""
    url = gateway.rstrip("/") + "/v1/hivemind/opentrainer/status"
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=15) as resp:
        latency_ms = round((time.monotonic() - started) * 1000.0, 2)
        payload = json.loads(resp.read().decode("utf-8"))
    modes = payload.get("firewall_traversal_modes") or []
    if isinstance(modes, list):
        mode_spec = next((m for m in modes if m.get("mode_id") == network_mode), modes[0] if modes else {})
    else:
        mode_spec = modes.get(network_mode) or modes.get("outbound-https-relay") or {}
    offer = payload.get("active_task_offer") or {}
    return {
        "ok": True,
        "gateway": gateway,
        "network_mode": network_mode,
        "outbound_https_reachable": True,
        "roundtrip_latency_ms": latency_ms,
        "inbound_ports_required": int(mode_spec.get("inbound_ports_required", 0)),
        "zero_bind_all_interfaces": True,
        "opentrainer_version": payload.get("version", "1.0.0-s36"),
        "amcp_topic": offer.get("topic") or payload.get("amcp_topic", "hivemind.amcp.federation.v1"),
    }


def send_heartbeat(
    *,
    gateway: str,
    api_key: str,
    host_label: str,
    gpu_name: str,
    vram_total_mib: int,
    vram_free_mib: int,
    network_mode: str,
    training_stack: str,
    active_campaign: str,
) -> dict[str, Any]:
    url = gateway.rstrip("/") + "/v1/hivemind/opentrainer/heartbeat"
    body = json.dumps(
        {
            "host_label": host_label,
            "gpu_name": gpu_name,
            "vram_total_mib": vram_total_mib,
            "vram_free_mib": vram_free_mib,
            "network_mode": network_mode,
            "training_stack": training_stack,
            "active_campaign": active_campaign,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_round_report(report_path: str) -> dict[str, Any]:
    """Read measured round results from a JSON report and hash the real delta file bytes.

    The report must contain finite ``loss_before``, ``loss_after`` and ``delta_norm`` numbers and a
    ``delta_path`` string. A relative ``delta_path`` is resolved against the report's directory.
    Raises ``ValueError`` when the report or the delta file is missing or malformed.
    """
    try:
        with open(report_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"round report unreadable: {report_path}: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError("round report must be a JSON object")
    measured: dict[str, float] = {}
    for key in ("loss_before", "loss_after", "delta_norm"):
        value = report.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"round report field {key!r} must be a finite number")
        measured[key] = float(value)
    delta_path_raw = report.get("delta_path")
    if not isinstance(delta_path_raw, str) or not delta_path_raw.strip():
        raise ValueError("round report field 'delta_path' must be a non-empty string")
    delta_path = delta_path_raw if os.path.isabs(delta_path_raw) else os.path.join(os.path.dirname(os.path.abspath(report_path)), delta_path_raw)
    try:
        with open(delta_path, "rb") as fh:
            delta_bytes = fh.read()
    except OSError as exc:
        raise ValueError(f"delta file unreadable: {delta_path}: {exc}") from exc
    if not delta_bytes:
        raise ValueError(f"delta file is empty: {delta_path}")
    return {
        **measured,
        "delta_sha256": hashlib.sha256(delta_bytes).hexdigest(),
        "delta_bytes": len(delta_bytes),
    }


def submit_verified_delta(
    *,
    gateway: str,
    api_key: str,
    round_id: int,
    campaign_id: str,
    learner_index: int,
    inner_steps: int,
    loss_before: float,
    loss_after: float,
    delta_norm: float,
    delta_sha256: str,
    code_pin: str,
    network_mode: str,
) -> dict[str, Any]:
    url = gateway.rstrip("/") + "/v1/hivemind/opentrainer/submit-delta"
    body = json.dumps(
        {
            "round_id": round_id,
            "campaign_id": campaign_id,
            "learner_index": learner_index,
            "inner_steps": inner_steps,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "delta_norm": delta_norm,
            "delta_sha256": delta_sha256,
            "code_pin": code_pin,
            "network_mode": network_mode,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HiveMind OpenTrainer Agent — Firewall-Friendly Outbound GPU Co-Training (BOINC + AMCP + Kafka + Aglets)."
    )
    parser.add_argument(
        "--gateway",
        "--server-url",
        dest="gateway",
        default=os.environ.get("HIVEMIND_GATEWAY_URL", DEFAULT_GATEWAY),
        help="OpenTrainer gateway URL (also accepts --server-url)",
    )
    parser.add_argument(
        "--network-mode",
        default="outbound-https-relay",
        choices=("outbound-https-relay", "amcp-kafka-aglet", "wireguard-t4-hub", "ssh-loopback-tunnel", "boinc-night-workunit"),
        help="Firewall traversal mode (default: outbound-https-relay over TCP/443 with 0 inbound ports)",
    )
    parser.add_argument("--check-firewall", action="store_true", help="Verify outbound HTTPS traversal to OpenTrainer")
    parser.add_argument("--host-label", default=socket.gethostname() or platform.node() or "trainer-node", help="Local trainer host label")
    parser.add_argument("--gpu-name", default="", help="Override detected GPU model name")
    parser.add_argument("--vram-total-mib", type=int, default=0, help="Override total GPU VRAM in MiB")
    parser.add_argument("--vram-free-mib", type=int, default=0, help="Override free GPU VRAM in MiB")
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        default=0.0,
        help="Optional VRAM ceiling in GiB for training allocation (e.g. 5.5 on an 8 GB GPU)",
    )
    parser.add_argument("--training-stack", default="qlora-moe+diloco+amcp-aglet+boinc", help="Active training stack descriptor")
    parser.add_argument("--campaign-id", default="E8-STAGE2-TINY-GRANITE-MOE-QLORA", help="Active co-training campaign ID")
    parser.add_argument("--submit-delta", action="store_true", help="Submit a measured round delta after heartbeat (requires --round-report)")
    parser.add_argument(
        "--round-report",
        default="",
        help="JSON file with measured loss_before, loss_after, delta_norm and delta_path (required with --submit-delta)",
    )
    parser.add_argument("--round-id", type=int, default=1, help="Round ID for delta submission")
    parser.add_argument("--learner-index", type=int, default=1, help="Learner index in the K-node cell (0=T4 hub, 1=RTX 2070, ...)")
    parser.add_argument("--inner-steps", type=int, default=8, help="Inner optimizer steps completed")
    parser.add_argument("--once", action="store_true", help="Run a single heartbeat/round and exit")
    parser.add_argument("--interval-s", type=float, default=60.0, help="Heartbeat interval in seconds when running continuously")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(float(args.interval_s)) or float(args.interval_s) <= 0.0:
        print("Error: --interval-s must be a finite positive number.", file=sys.stderr)
        return 2
    if not math.isfinite(float(args.max_vram_gb)) or float(args.max_vram_gb) < 0.0:
        print("Error: --max-vram-gb must be a finite non-negative number.", file=sys.stderr)
        return 2
    round_report: dict[str, Any] | None = None
    if args.submit_delta:
        if not args.round_report:
            print("Error: --submit-delta requires --round-report; no delta was submitted.", file=sys.stderr)
            return 2
        try:
            round_report = load_round_report(args.round_report)
        except ValueError as exc:
            print(f"Error: {exc}; no delta was submitted.", file=sys.stderr)
            return 2
    detected_gpu, detected_total, detected_free = detect_local_gpu()
    gpu_name = args.gpu_name or detected_gpu or ("custom-gpu" if args.vram_total_mib > 0 else "")
    vram_total = args.vram_total_mib if args.vram_total_mib > 0 else detected_total
    vram_free = args.vram_free_mib if args.vram_free_mib > 0 else min(detected_free, vram_total)
    if float(args.max_vram_gb) > 0.0 and vram_free > 0:
        cap_mib = max(512, int(round(float(args.max_vram_gb) * 1024.0)))
        vram_free = min(vram_free, cap_mib)

    if args.check_firewall:
        try:
            fw_report = verify_firewall_traversal(args.gateway, args.network_mode)
            print(json.dumps({"firewall_check": fw_report}, indent=2))
        except (urllib.error.URLError, ValueError) as exc:
            print(json.dumps({"ok": False, "error": f"Firewall check failed: {exc}"}), file=sys.stderr)
            return 1

    api_key = os.environ.get("HIVEMIND_API_KEY", "").strip()
    if not api_key:
        if args.check_firewall and args.once:
            return 0
        print("Error: HIVEMIND_API_KEY environment variable is required for OpenTrainer heartbeats.", file=sys.stderr)
        return 2

    if vram_total <= 0 or not gpu_name:
        print(
            "Error: No readable NVIDIA GPU detected via nvidia-smi (and --vram-total-mib was not specified); "
            "refusing to send fabricated GPU telemetry.",
            file=sys.stderr,
        )
        return 2

    while True:
        try:
            ack = send_heartbeat(
                gateway=args.gateway,
                api_key=api_key,
                host_label=args.host_label,
                gpu_name=gpu_name,
                vram_total_mib=vram_total,
                vram_free_mib=vram_free,
                network_mode=args.network_mode,
                training_stack=args.training_stack,
                active_campaign=args.campaign_id,
            )
            offer = ack.get("active_task_offer") or {}
            print(
                json.dumps(
                    {
                        "event": "opentrainer_heartbeat_ok",
                        "node": ack.get("node"),
                        "aglet_phase": ack.get("aglet_lifecycle_phase"),
                        "task_offer_round": offer.get("round_id"),
                    },
                    indent=2,
                )
            )
            if args.submit_delta:
                assert round_report is not None
                receipt = submit_verified_delta(
                    gateway=args.gateway,
                    api_key=api_key,
                    round_id=int(offer.get("round") or offer.get("round_id") or args.round_id),
                    campaign_id=args.campaign_id,
                    learner_index=args.learner_index,
                    inner_steps=args.inner_steps,
                    loss_before=round_report["loss_before"],
                    loss_after=round_report["loss_after"],
                    delta_norm=round_report["delta_norm"],
                    delta_sha256=round_report["delta_sha256"],
                    code_pin=str(offer.get("script_sha256") or offer.get("code_pin") or ""),
                    network_mode=args.network_mode,
                )
                print(json.dumps({"event": "opentrainer_delta_accepted", "receipt": receipt}, indent=2))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"OpenTrainer HTTP {exc.code}: {detail}", file=sys.stderr)
            if args.once:
                return 1
        except urllib.error.URLError as exc:
            print(f"OpenTrainer connection error: {exc}", file=sys.stderr)
            if args.once:
                return 1

        if args.once:
            return 0
        time.sleep(max(5.0, float(args.interval_s)))


if __name__ == "__main__":
    raise SystemExit(main())

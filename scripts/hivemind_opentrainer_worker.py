#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Turnkey 8 GB+ GPU OpenTrainer Worker CLI for r/SelfHostedAI Volunteer Trainers.

Provides a zero-inbound-port, privacy-guaranteed (``zero_private_user_prompts_on_trainer == True``)
entrypoint for consumer GPUs (8 GB RTX 2070 / 3060 / 4070 / 4090) contributing overnight DiLoCo
NF4 QLoRA training steps:
  - Enforces a default 5.5 GiB (5,632 MiB) VRAM ceiling on 8 GB cards unless overridden.
  - Verifies that the upstream OpenTrainer coordinator certifies ``zero_private_user_prompts_on_trainer == True``
    when ``--verify-privacy`` is passed.
  - Delegates heartbeat and signed round delta submission to ``scripts/hivemind_opentrainer_agent.py``.

Usage:
  HIVEMIND_API_KEY=hm-your-key python3 scripts/hivemind_opentrainer_worker.py \\
    --server-url https://hivemind-openrouter-1003063861791.us-central1.run.app \\
    --max-vram-gb 5.5
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable
import urllib.error
import urllib.request

DEFAULT_VRAM_FLOOR_MIB = 4300


def check_vram_floor_guard(
    free_vram_mib: int,
    estimated_peak_mib: int = 0,
    floor_mib: int = DEFAULT_VRAM_FLOOR_MIB,
    total_vram_mib: int = 15360,
    apply_torch_cap: bool = True,
    set_fraction_fn: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Check ``free_vram_mib - estimated_peak_mib >= floor_mib`` and cap CUDA per-process memory fraction (PS-02-D1)."""
    remaining_free_mib = int(free_vram_mib) - max(0, int(estimated_peak_mib))
    cap_mib = max(0, int(free_vram_mib) - int(floor_mib))
    total_eff = max(int(total_vram_mib), int(free_vram_mib), 1)
    fraction = min(1.0, max(0.0, cap_mib / float(total_eff)))
    reason = ""
    if int(free_vram_mib) < int(floor_mib):
        reason = f"free VRAM {free_vram_mib} MiB below floor {floor_mib} MiB"
    elif remaining_free_mib < int(floor_mib):
        reason = (
            f"free VRAM {free_vram_mib} MiB minus estimated peak {estimated_peak_mib} MiB "
            f"leaves {remaining_free_mib} MiB < {floor_mib} MiB floor"
        )
    ok = reason == ""
    if ok and apply_torch_cap:
        if set_fraction_fn is not None:
            set_fraction_fn(fraction)
        else:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.set_per_process_memory_fraction(fraction)
            except Exception:  # noqa: BLE001
                pass
    return {
        "ok": ok,
        "free_vram_mib": int(free_vram_mib),
        "estimated_peak_mib": int(estimated_peak_mib),
        "remaining_free_mib": remaining_free_mib,
        "floor_mib": int(floor_mib),
        "cap_mib": cap_mib,
        "memory_fraction": round(fraction, 6),
        "reason": reason,
    }


def _load_opentrainer_agent() -> Any:
    agent_path = Path(__file__).resolve().with_name("hivemind_opentrainer_agent.py")
    spec = importlib.util.spec_from_file_location("hivemind_opentrainer_agent", agent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {agent_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["hivemind_opentrainer_agent"] = module
    spec.loader.exec_module(module)
    return module


def verify_trainer_privacy_contract(server_url: str) -> dict[str, Any]:
    """Verify that the upstream OpenTrainer status endpoint enforces ``zero_private_user_prompts_on_trainer``."""
    url = server_url.rstrip("/") + "/v1/hivemind/opentrainer/status"
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("OpenTrainer status response must be a JSON object")
    portal_cat = payload.get("portal_category") or {}
    zero_prompts = bool(portal_cat.get("zero_private_user_prompts_on_trainer"))
    if not zero_prompts:
        raise ValueError("Upstream OpenTrainer status did not assert zero_private_user_prompts_on_trainer=True")
    return {
        "ok": True,
        "server_url": server_url.rstrip("/"),
        "signup_url_path": str(portal_cat.get("signup_url_path") or "/signup?role=trainer"),
        "zero_private_user_prompts_on_trainer": True,
        "inbound_ports_required": 0,
    }


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    verify_privacy = False
    min_free_vram_mib = DEFAULT_VRAM_FLOOR_MIB
    estimated_peak_mib = 0
    filtered_args: list[str] = []
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        if arg == "--verify-privacy":
            verify_privacy = True
        elif arg == "--min-free-vram-mib" and i + 1 < len(raw_args):
            min_free_vram_mib = int(float(raw_args[i + 1]))
            i += 1
        elif arg.startswith("--min-free-vram-mib="):
            min_free_vram_mib = int(float(arg.split("=", 1)[1]))
        elif arg == "--estimated-peak-mib" and i + 1 < len(raw_args):
            estimated_peak_mib = int(float(raw_args[i + 1]))
            i += 1
        elif arg.startswith("--estimated-peak-mib="):
            estimated_peak_mib = int(float(arg.split("=", 1)[1]))
        else:
            filtered_args.append(arg)
        i += 1

    if "--max-vram-gb" not in filtered_args and not any(a.startswith("--max-vram-gb=") for a in filtered_args):
        filtered_args.extend(["--max-vram-gb", "5.5"])

    agent = _load_opentrainer_agent()
    parsed = agent.build_parser().parse_args(filtered_args)
    _, detected_total, detected_free = agent.detect_local_gpu()
    raw_total_mib = parsed.vram_total_mib if parsed.vram_total_mib > 0 else detected_total
    raw_free_mib = parsed.vram_free_mib if parsed.vram_free_mib > 0 else detected_free
    if raw_free_mib > 0:
        floor_check = check_vram_floor_guard(
            free_vram_mib=raw_free_mib,
            estimated_peak_mib=estimated_peak_mib,
            floor_mib=min_free_vram_mib,
            total_vram_mib=raw_total_mib or 15360,
        )
        if not floor_check["ok"]:
            print(json.dumps({"ok": False, "error": f"VRAM floor guard refused launch: {floor_check['reason']}", "guard": floor_check}), file=sys.stderr)
            return 2

    if verify_privacy:
        try:
            report = verify_trainer_privacy_contract(parsed.gateway)
            print(json.dumps({"trainer_privacy_contract": report}, indent=2))
        except (urllib.error.URLError, ValueError) as exc:
            print(json.dumps({"ok": False, "error": f"Trainer privacy check failed: {exc}"}), file=sys.stderr)
            return 1

    return int(agent.main(filtered_args))


if __name__ == "__main__":
    raise SystemExit(main())

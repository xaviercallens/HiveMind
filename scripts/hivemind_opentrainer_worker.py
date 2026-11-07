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
from typing import Any
import urllib.error
import urllib.request


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
    filtered_args: list[str] = []
    for arg in raw_args:
        if arg == "--verify-privacy":
            verify_privacy = True
        else:
            filtered_args.append(arg)

    if "--max-vram-gb" not in filtered_args and not any(a.startswith("--max-vram-gb=") for a in filtered_args):
        filtered_args.extend(["--max-vram-gb", "5.5"])

    agent = _load_opentrainer_agent()
    parsed = agent.build_parser().parse_args(filtered_args)
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

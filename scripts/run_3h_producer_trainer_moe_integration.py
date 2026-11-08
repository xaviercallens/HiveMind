#!/usr/bin/env python3
# Copyright (C) 2026 Xavier Callens
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""3-Hour Producer + Trainer Self-Training MoE Integration & Inference Continuity Runner.

Executed every 3 hours (cron ``0 */3 * * *``) and on demand to:
1. Verify and preserve live OpenRouter Producer Fleet & T4 ``/bee`` (``NIGHT-MOE-1`` LoRA) + ``/u`` endpoints,
   while ensuring RTX 2070 failover readiness via ``rtx2070_failover_orchestrator``.
2. Register / refresh OpenTrainer GPU nodes and execute a multi-round Stage-2 Tiny Granite-MoE
   NF4 QLoRA + DiLoCo 2D-DCT self-training burst, submitting real weight-delta norms and eval losses
   to ``/v1/hivemind/opentrainer/submit-delta``.
3. Verify end-to-end OpenRouter inference continuity across ``hivemind/coder``, ``hivemind/bee-stars``,
   and ``hivemind/auto``, plus ``/v1/hivemind/contributive-hub`` and ``/v1/hivemind/opentrainer/moe-self-training``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from typing import Any

from rtx2070_failover_orchestrator import (
    DEFAULT_GATEWAY,
    STATUS_JSON_PATH,
    _read_producer_key,
    check_gcp_t4_status,
    discover_local_ollama_models,
    step_e8_stage2_controller_round,
    sync_opentrainer_and_gpu_telemetry,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
INTEGRATION_STATUS_PATH = REPO_ROOT / "artifacts" / "producer_trainer_3h_moe_integration_status.json"


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = 20.0,
) -> tuple[int, dict[str, Any]]:
    req_headers = {"Accept": "application/json", **(headers or {})}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body.strip() else {}
        except ValueError:
            parsed = {"raw_error": body[:300]}
        return exc.code, parsed
    except Exception as exc:
        return 0, {"error": str(exc)}


def verify_live_inference_service(
    openrouter_url: str,
    api_key: str,
) -> dict[str, Any]:
    """Probe live OpenRouter inference on hivemind/coder, hivemind/bee-stars, and hivemind/auto."""
    probes = [
        (
            "hivemind/coder",
            "Reply with the single word: OK",
            32,
        ),
        (
            "hivemind/bee-stars",
            "Classify a star with Teff=3120 K, logg=5.05, [Fe/H]=-0.20. Reply with spectral class only.",
            48,
        ),
        (
            "hivemind/auto",
            "Write a one-line Python lambda that squares x.",
            48,
        ),
    ]
    results: dict[str, Any] = {}
    all_ok = True
    for model_id, prompt, max_tokens in probes:
        t0 = time.monotonic()
        code, data = _http_json(
            f"{openrouter_url.rstrip('/')}/v1/chat/completions",
            method="POST",
            headers={"Authorization": f"Bearer {api_key}"},
            payload={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout_s=30.0,
        )
        latency_ms = round((time.monotonic() - t0) * 1000.0, 1)
        choices = data.get("choices") or []
        reply = ""
        if choices and isinstance(choices[0], dict):
            reply = str((choices[0].get("message") or {}).get("content") or "").strip()
        ok = code == 200 and bool(reply)
        if not ok:
            all_ok = False
        results[model_id] = {
            "http_status": code,
            "ok": ok,
            "latency_ms": latency_ms,
            "reply_preview": reply[:140],
        }
    return {
        "all_models_healthy": all_ok,
        "models": results,
    }


def run_3h_integration_cycle(
    *,
    openrouter_url: str = DEFAULT_GATEWAY,
    rounds_per_cycle: int = 3,
    inner_steps: int = 24,
) -> dict[str, Any]:
    """Execute one 3-hour Producer + Trainer MoE self-training integration and portal verification cycle."""
    now_iso = datetime.now(timezone.utc).isoformat()
    api_key = _read_producer_key()
    t4_status = check_gcp_t4_status()
    local_models = discover_local_ollama_models()

    training_rounds: list[dict[str, Any]] = []
    for _ in range(max(1, rounds_per_cycle)):
        round_metrics = step_e8_stage2_controller_round(inner_steps=inner_steps)
        ot_sync = sync_opentrainer_and_gpu_telemetry(
            openrouter_url,
            round_metrics,
            t4_reachable=bool(t4_status.get("t4_reachable")),
        )
        training_rounds.append(
            {
                "round_id": round_metrics.get("round_id"),
                "loss_before": round_metrics.get("loss_before"),
                "loss_after": round_metrics.get("loss_after"),
                "delta_norm": round_metrics.get("delta_norm"),
                "code_pin": round_metrics.get("code_pin"),
                "opentrainer_sync": ot_sync,
            }
        )

    hub_code, hub_data = _http_json(f"{openrouter_url.rstrip('/')}/v1/hivemind/contributive-hub")
    moe_code, moe_data = _http_json(f"{openrouter_url.rstrip('/')}/v1/hivemind/opentrainer/moe-self-training")
    inference_check = verify_live_inference_service(openrouter_url, api_key)

    report = {
        "timestamp_utc": now_iso,
        "openrouter_url": openrouter_url,
        "cadence": "every_3_hours",
        "gcp_t4_status": t4_status,
        "local_ollama_models": local_models,
        "self_training_burst": {
            "rounds_executed": len(training_rounds),
            "inner_steps_per_round": inner_steps,
            "latest_round": training_rounds[-1] if training_rounds else None,
            "rounds": training_rounds,
        },
        "contributive_hub_http": hub_code,
        "contributive_hub_summary": {
            "openrouter_producer_fleet": hub_data.get("openrouter_producer_fleet"),
            "opentrainer_moe_self_training": hub_data.get("opentrainer_moe_self_training"),
            "multi_gpu_2_to_4_byom_playbook_topologies": len(
                (hub_data.get("multi_gpu_2_to_4_byom_playbook") or {}).get("supported_topologies") or []
            ),
        },
        "moe_self_training_endpoint_http": moe_code,
        "moe_self_training_progress": moe_data,
        "inference_continuity": inference_check,
        "orchestrator_daemon_status_file": str(STATUS_JSON_PATH),
    }
    INTEGRATION_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    INTEGRATION_STATUS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 3-Hour Producer + Trainer Self-Training MoE Integration Cycle")
    parser.add_argument("--openrouter-url", default=DEFAULT_GATEWAY)
    parser.add_argument("--rounds", type=int, default=3, help="Number of QLoRA self-training rounds per 3h burst")
    parser.add_argument("--inner-steps", type=int, default=24, help="Inner steps per QLoRA round")
    args = parser.parse_args()

    report = run_3h_integration_cycle(
        openrouter_url=args.openrouter_url,
        rounds_per_cycle=args.rounds,
        inner_steps=args.inner_steps,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

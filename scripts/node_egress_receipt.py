#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""PS-02-D4 — Measure per-node-type payload egress per hour and compare against declared budgets.

Constructs and serializes the exact wire payloads emitted by each HiveMind volunteer node type:
  1. ``producer_agent`` (``src/hivemind/router/hivemind_producer_agent.py``)
  2. ``producer_shield`` (``src/hivemind/router/hivemind_producer_shield.py``)
  3. ``boinc_trainer`` (``src/hivemind/boinc/real_client/moe_app.py``)
  4. ``opentrainer_worker`` (``scripts/hivemind_opentrainer_worker.py``)
  5. ``amcp_federation_learner`` (``scripts/e8_amcp_round.py``)
  6. ``crowdgpt_trainer`` (CrowdGPT's official client, launched by ``scripts/hivemind_opentrainer_agent.py --crowdgpt``)

Basis: rows 1-5 are MODELED (the serialized payload size times a nominal cadence), not network measurements.
Row 6 is from the CrowdGPT client's own upload log lines on the T4 (artifacts/experiments/CROWDGPT_T4_UPLOADS_2026-11-14.json).
Each row carries ``egress_basis`` saying which.

Usage:
  python3 scripts/node_egress_receipt.py
  python3 scripts/node_egress_receipt.py --out artifacts/experiments/NODE_EGRESS_2026-11-15.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "experiments" / "NODE_EGRESS_2026-11-15.json"
CROWDGPT_UPLOADS = REPO_ROOT / "artifacts" / "experiments" / "CROWDGPT_T4_UPLOADS_2026-11-14.json"
MODELED_BASIS = "modeled: serialized payload size x nominal cadence (not a network measurement)"
CROWDGPT_TRAINER_BUDGET_BYTES_PER_HOUR = 2 * 1024**3  # 2 GiB/hour: one ~0.95 GB round every ~30-40 min


def _measure_crowdgpt_trainer_egress(path: Path = CROWDGPT_UPLOADS) -> dict[str, Any]:
    """Egress of a CrowdGPT trainer from the client's logged upload sizes and submission times (T4, 14 Nov 2026)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rounds = data["submitted_rounds"]
    if not rounds:
        raise ValueError("no submitted rounds in the CrowdGPT measurement file")
    mean_bytes = sum(float(r["upload_mb"]) for r in rounds) / len(rounds) * 1_000_000
    by_run: dict[str, list[dt.datetime]] = {}
    for r in rounds:
        by_run.setdefault(r["run"], []).append(dt.datetime.fromisoformat(r["submitted_utc"].replace("Z", "+00:00")))
    gaps = sorted((b - a).total_seconds() for ts in by_run.values() for a, b in zip(sorted(ts), sorted(ts)[1:]))
    if not gaps:
        raise ValueError("need two submissions in one run to measure the round interval")
    median_gap_s = gaps[len(gaps) // 2] if len(gaps) % 2 else (gaps[len(gaps) // 2 - 1] + gaps[len(gaps) // 2]) / 2
    rounds_per_hour = 3600.0 / median_gap_s
    per_hour = int(mean_bytes * rounds_per_hour)
    return {
        "node_type": "crowdgpt_trainer",
        "entrypoint": "Vxtzq/CrowdGPT client.py (launched by scripts/hivemind_opentrainer_agent.py --crowdgpt)",
        "cadence": f"{rounds_per_hour:.2f} submitted rounds/hour (median interval {median_gap_s / 60:.1f} min)",
        "payload_bytes_per_event": int(mean_bytes),
        "events_per_hour": round(rounds_per_hour, 3),
        "measured_egress_bytes_per_hour": per_hour,
        "declared_budget_bytes_per_hour": CROWDGPT_TRAINER_BUDGET_BYTES_PER_HOUR,
        "within_budget": per_hour <= CROWDGPT_TRAINER_BUDGET_BYTES_PER_HOUR,
        "egress_basis": f"client_log_reported: {len(rounds)} submitted rounds on a Tesla T4 ({data['measured_on']})",
        "source": str(Path(path).relative_to(REPO_ROOT)) if Path(path).is_relative_to(REPO_ROOT) else str(path),
        "allowed_destinations": ["https://server.crowdgpt.net", "https://huggingface.co"],
        "lateral_movement_allowed": False,
        "note": "Uploads go to CrowdGPT's coordinator, not HiveMind. Each finished round uploads ~0.9 GB; mind data caps.",
    }


EVIDENCE_LABEL = {"evidence": "modelled", "not_a_measurement": True, "evidence_note": "bytes computed from synthetic payloads and assumed event rates, not counted on a node", "evidence_audit": "docs/roadmap/DOD_RECEIPT_AUDIT_2026-11-10.md"}  # this generator does not measure; see the audit


def _measure_producer_agent_egress() -> dict[str, Any]:
    heartbeat_payload = {
        "endpoint": "https://producer-tunnel.example.run.app",
        "models": [
            {
                "tier": "coder",
                "local_name": "qwen2.5-coder:7b",
                "cost_per_1k_tokens": 0.4,
                "max_tokens_per_epoch": 60000,
            }
        ],
        "vram_gb": 12.0,
        "agent_version": "1.0.0",
    }
    raw_bytes = len(json.dumps(heartbeat_payload, separators=(",", ":")).encode("utf-8"))
    http_framing_bytes = 220
    bytes_per_event = raw_bytes + http_framing_bytes
    events_per_hour = 60  # 1 heartbeat every 60 s
    measured_per_hour = bytes_per_event * events_per_hour
    declared_budget = 262_144  # 256 KiB/hour
    return {
        "node_type": "producer_agent",
        "entrypoint": "src/hivemind/router/hivemind_producer_agent.py",
        "cadence": "60 heartbeats/hour (1 per 60 s)",
        "payload_bytes_per_event": bytes_per_event,
        "events_per_hour": events_per_hour,
        "measured_egress_bytes_per_hour": measured_per_hour,
        "declared_budget_bytes_per_hour": declared_budget,
        "within_budget": measured_per_hour <= declared_budget,
        "allowed_destinations": ["https://<gateway>/v1/producers/heartbeat", "http://127.0.0.1:11434"],
        "lateral_movement_allowed": False,
    }


def _measure_producer_shield_egress() -> dict[str, Any]:
    completion_payload = {
        "id": "chatcmpl-hm-shield-sample",
        "object": "chat.completion",
        "created": 1794086400,
        "model": "qwen2.5-coder:7b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "def add(a: int, b: int) -> int:\n    return a + b\n" * 32},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 64, "completion_tokens": 512, "total_tokens": 576},
    }
    raw_bytes = len(json.dumps(completion_payload, separators=(",", ":")).encode("utf-8"))
    http_framing_bytes = 240
    bytes_per_event = raw_bytes + http_framing_bytes
    events_per_hour = 30  # nominal 30 completions/hour when active (0 B/hour when idle)
    measured_per_hour = bytes_per_event * events_per_hour
    declared_budget = 10 * 1024 * 1024  # 10 MiB/hour
    return {
        "node_type": "producer_shield",
        "entrypoint": "src/hivemind/router/hivemind_producer_shield.py",
        "cadence": "0 B/hour idle; nominal 30 completions/hour (512 tokens each) when serving",
        "payload_bytes_per_event": bytes_per_event,
        "events_per_hour": events_per_hour,
        "measured_egress_bytes_per_hour": measured_per_hour,
        "declared_budget_bytes_per_hour": declared_budget,
        "within_budget": measured_per_hour <= declared_budget,
        "allowed_destinations": ["http://127.0.0.1:11434", "http://localhost:11434"],
        "lateral_movement_allowed": False,
    }


def _measure_boinc_trainer_egress() -> dict[str, Any]:
    # Rank-8 LoRA on q/k/v/o across 40 layers of granite-3.1-1b-a400m-base:
    # 655,360 fp32 parameters = 2,621,440 bytes per workunit upload + 1,024 B metadata envelope
    lora_param_count = 655_360
    bytes_per_workunit = lora_param_count * 4 + 1024
    workunits_per_hour = 2
    measured_per_hour = bytes_per_workunit * workunits_per_hour
    declared_budget = 16 * 1024 * 1024  # 16 MiB/hour
    return {
        "node_type": "boinc_trainer",
        "entrypoint": "src/hivemind/boinc/real_client/moe_app.py",
        "cadence": "2 QLoRA-MoE workunit adapter uploads/hour (inside 22:00-07:00 Europe/Paris window)",
        "payload_bytes_per_event": bytes_per_workunit,
        "events_per_hour": workunits_per_hour,
        "measured_egress_bytes_per_hour": measured_per_hour,
        "declared_budget_bytes_per_hour": declared_budget,
        "within_budget": measured_per_hour <= declared_budget,
        "allowed_destinations": ["https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app", "https://storage.googleapis.com"],
        "lateral_movement_allowed": False,
    }


def _measure_opentrainer_worker_egress() -> dict[str, Any]:
    hb_body = json.dumps(
        {
            "host_label": "volunteer-rtx3070",
            "gpu_name": "NVIDIA GeForce RTX 3070",
            "vram_total_mib": 8192,
            "vram_free_mib": 5632,
            "network_mode": "outbound-https-relay",
            "training_stack": "qlora-moe+diloco+amcp-aglet+boinc",
            "active_campaign": "E8-STAGE2-TINY-GRANITE-MOE-QLORA",
        },
        separators=(",", ":"),
    ).encode("utf-8")
    delta_body = json.dumps(
        {
            "round_id": 1,
            "campaign_id": "E8-STAGE2-TINY-GRANITE-MOE-QLORA",
            "learner_index": 1,
            "inner_steps": 8,
            "loss_before": 4.162,
            "loss_after": 3.841,
            "delta_norm": 0.412,
            "delta_sha256": hashlib.sha256(b"sample-lora-delta").hexdigest(),
            "code_pin": hashlib.sha256(b"sample-code-pin").hexdigest(),
            "network_mode": "outbound-https-relay",
        },
        separators=(",", ":"),
    ).encode("utf-8")
    http_framing = 220
    hb_bytes_per_hour = (len(hb_body) + http_framing) * 60
    delta_bytes_per_hour = (len(delta_body) + http_framing) * 4
    measured_per_hour = hb_bytes_per_hour + delta_bytes_per_hour
    declared_budget = 524_288  # 512 KiB/hour
    return {
        "node_type": "opentrainer_worker",
        "entrypoint": "scripts/hivemind_opentrainer_worker.py",
        "cadence": "60 heartbeats/hour + 4 signed delta receipts/hour",
        "payload_bytes_per_event": len(hb_body) + http_framing,
        "events_per_hour": 64,
        "measured_egress_bytes_per_hour": measured_per_hour,
        "declared_budget_bytes_per_hour": declared_budget,
        "within_budget": measured_per_hour <= declared_budget,
        "allowed_destinations": ["https://hivemind-openrouter-1003063861791.us-central1.run.app",
                                 "https://datasets-server.huggingface.co (--train: public training rows, read-only)"],
        "lateral_movement_allowed": False,
    }


def _measure_amcp_federation_learner_egress() -> dict[str, Any]:
    # Tiny-MoE rank-4 q/v LoRA state serialized as base64 in AMCP CloudEvents RESULT envelope (~28 KiB/round)
    claim_envelope_bytes = 640
    result_envelope_bytes = 28_672
    bytes_per_round = claim_envelope_bytes + result_envelope_bytes
    rounds_per_hour = 12
    measured_per_hour = bytes_per_round * rounds_per_hour
    declared_budget = 4 * 1024 * 1024  # 4 MiB/hour
    return {
        "node_type": "amcp_federation_learner",
        "entrypoint": "scripts/e8_amcp_round.py",
        "cadence": "12 DiLoCo rounds/hour (CLAIM + signed RESULT CloudEvents envelope)",
        "payload_bytes_per_event": bytes_per_round,
        "events_per_hour": rounds_per_hour,
        "measured_egress_bytes_per_hour": measured_per_hour,
        "declared_budget_bytes_per_hour": declared_budget,
        "within_budget": measured_per_hour <= declared_budget,
        "allowed_destinations": ["HIVEMIND_KAFKA_BOOTSTRAP (topic: federation)"],
        "lateral_movement_allowed": False,
    }


def build_node_egress_receipt() -> dict[str, Any]:
    node_types = [
        _measure_producer_agent_egress(),
        _measure_producer_shield_egress(),
        _measure_boinc_trainer_egress(),
        _measure_opentrainer_worker_egress(),
        _measure_amcp_federation_learner_egress(),
    ]
    for row in node_types:
        row.setdefault("egress_basis", MODELED_BASIS)
    node_types.append(_measure_crowdgpt_trainer_egress())
    return {
        "schema": "hivemind.node_egress_receipt.v1",
        "roadmap_deliverable": "PS-02-D4",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "all_within_budget": all(row["within_budget"] for row in node_types),
        "node_types": {row["node_type"]: row for row in node_types},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output JSON receipt path")
    args = parser.parse_args(argv)
    receipt = build_node_egress_receipt()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({**EVIDENCE_LABEL, **receipt}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["all_within_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

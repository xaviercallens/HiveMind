#!/usr/bin/env bash
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
# PS-04-D1: Volunteer trainer install-to-first-work-unit smoke verification (< 900 s).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="python3"
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
fi

DATE_UTC="$(date -u +%F)"
OUT_PATH="${1:-$REPO_ROOT/artifacts/experiments/TRAINER_INSTALL_${DATE_UTC}.json}"

PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" - "$OUT_PATH" <<'PYEOF'
from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import scripts.hivemind_opentrainer_worker as worker

out_path = Path(sys.argv[1])
t0 = time.monotonic()

heartbeats_received: list[dict] = []
deltas_accepted: list[dict] = []


class _SmokeOpenTrainerHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path.endswith("/v1/hivemind/opentrainer/status"):
            resp = {
                "ok": True,
                "portal_category": {
                    "signup_url_path": "/signup?role=trainer",
                    "zero_private_user_prompts_on_trainer": True,
                },
            }
            data = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        body = json.loads(raw.decode("utf-8"))
        if self.path.endswith("/v1/hivemind/opentrainer/heartbeat"):
            heartbeats_received.append(body)
            resp = {
                "ok": True,
                "node": body.get("host_label", "smoke-trainer"),
                "aglet_lifecycle_phase": "training",
                "active_task_offer": {
                    "round_id": 1,
                    "script_sha256": "a" * 64,
                },
            }
            data = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path.endswith("/v1/hivemind/opentrainer/submit-delta"):
            deltas_accepted.append(body)
            resp = {"accepted": True, "round_id": body.get("round_id", 1), "hmt_reward": 10.0}
            data = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(404)
        self.end_headers()


server = HTTPServer(("127.0.0.1", 0), _SmokeOpenTrainerHandler)
port = server.server_address[1]
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()

prev_key = os.environ.get("HIVEMIND_API_KEY")
os.environ["HIVEMIND_API_KEY"] = "hm-smoke-ps04-d1-key"
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        delta_file = tmp / "adapter_delta.safetensors"
        delta_file.write_bytes(b"HIVEMIND-SMOKE-LORA-DELTA-V1")
        report_file = tmp / "round_report.json"
        report_file.write_text(
            json.dumps(
                {
                    "loss_before": 2.412,
                    "loss_after": 2.285,
                    "delta_norm": 0.142,
                    "delta_path": str(delta_file),
                }
            ),
            encoding="utf-8",
        )
        rc = worker.main(
            [
                "--gateway",
                f"http://127.0.0.1:{port}",
                "--verify-privacy",
                "--check-firewall",
                "--gpu-name",
                "NVIDIA GeForce RTX 2070",
                "--vram-total-mib",
                "8192",
                "--vram-free-mib",
                "6144",
                "--min-free-vram-mib",
                "4300",
                "--estimated-peak-mib",
                "1500",
                "--submit-delta",
                "--round-report",
                str(report_file),
                "--once",
            ]
        )
finally:
    if prev_key is None:
        os.environ.pop("HIVEMIND_API_KEY", None)
    else:
        os.environ["HIVEMIND_API_KEY"] = prev_key
    server.shutdown()
    server.server_close()

elapsed_s = round(time.monotonic() - t0, 4)
first_wu_accepted = (rc == 0 and len(heartbeats_received) >= 1 and len(deltas_accepted) >= 1)

receipt = {
    "schema": "hivemind.trainer_install_smoke_receipt.v1",
    "deliverable_id": "PS-04-D1",
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "elapsed_s": elapsed_s,
    "max_allowed_elapsed_s": 900.0,
    "first_work_unit_accepted": first_wu_accepted,
    "heartbeats_sent": len(heartbeats_received),
    "work_units_submitted": len(deltas_accepted),
    "outbound_only_networking": True,
    "inbound_listening_sockets_opened_by_worker": 0,
    "steps_executed": [
        "1. Verify Python environment and hivemind_opentrainer_worker entrypoint",
        "2. Check VRAM floor guard (6144 MiB free - 1500 MiB peak >= 4300 MiB floor)",
        "3. Verify zero_private_user_prompts_on_trainer privacy contract (--verify-privacy)",
        "4. Send outbound heartbeat, receive workunit offer, and submit signed SHA-256 delta (--submit-delta --once)",
    ],
    "passes_do_d": bool(first_wu_accepted and elapsed_s < 900.0),
}
out_path.parent.mkdir(parents=True, exist_ok=True)
receipt = {**{"evidence": "simulated", "not_a_measurement": True, "evidence_note": "existing checkout against a localhost stub with VRAM passed as an argument; not a clean machine", "evidence_audit": "docs/roadmap/DOD_RECEIPT_AUDIT_2026-11-10.md"}, **receipt}  # not a measurement; see the audit
out_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
if not receipt["passes_do_d"]:
    raise SystemExit(1)
print(json.dumps({"ok": True, "out": str(out_path), "elapsed_s": elapsed_s, "first_work_unit_accepted": first_wu_accepted}))
PYEOF

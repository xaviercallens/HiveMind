#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Print (and optionally save) tonight's dream-phase plan: which new MoE experts to train, on which joined GPU producer.

The planner is ``hivemind.swarm.campaign_orchestrator.plan_dream_phase`` (design: docs/S21_CAMPAIGN_ORCHESTRATOR.md). This script
only plans: ``--dry-run`` is required and there is no other mode, so nothing is ever sent to a GPU, a VM or a BOINC server.

Inputs: the campaign orchestrator state (joined participants), a ledger of verified episode counts per domain, the operator HOLD
file path and the producer governor state path. A missing or unreadable input makes the plan BLOCKED with the reason; it is never
filled in. Exit status: 0 = PLANNED, 2 = BLOCKED.

  python scripts/dream_phase_plan.py --dry-run --state STATE.json --ledger LEDGER.json \
      --governor-state governor_state.json --hold-file HOLD [--now 2026-10-28T21:30:00Z] [--out plan.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from hivemind.swarm.campaign_orchestrator import (
    STATE_SCHEMA_VERSION,
    DomainEvidence,
    DreamPolicy,
    atomic_write_json,
    load_episode_ledger,
    plan_dream_phase,
)


def parse_now(text: str | None) -> float:
    if not text:
        return time.time()
    stamp = datetime.fromisoformat(text)
    if stamp.tzinfo is None:
        raise SystemExit("--now needs an explicit UTC offset or a trailing Z")
    return stamp.timestamp()


def read_members(path: Path, now_utc: float) -> tuple[list[dict[str, Any]], str]:
    """Active memberships from the orchestrator state file, or ([], reason) when it cannot be read."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [], f"campaign state not readable: {type(exc).__name__}"
    if not isinstance(data, dict) or data.get("schema_version") != STATE_SCHEMA_VERSION or not isinstance(data.get("memberships"), dict):
        return [], f"campaign state is not a schema-v{STATE_SCHEMA_VERSION} orchestrator state"
    members = [m for m in data["memberships"].values() if m.get("status") == "active" and float(m.get("expires_at_utc", 0)) > now_utc]
    return members, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", required=True, help="print the plan; the only mode this script has")
    ap.add_argument("--state", type=Path, required=True, help="campaign orchestrator state JSON")
    ap.add_argument("--ledger", type=Path, required=True, help="verified episode counts per domain (JSON)")
    ap.add_argument("--governor-state", type=Path, required=True, help="producer governor state JSON (t4_governor.py --state-file)")
    ap.add_argument("--hold-file", type=Path, required=True, help="operator HOLD switch: if it exists the plan is BLOCKED")
    ap.add_argument("--now", help="plan as of this UTC instant (ISO 8601), default: the current time")
    ap.add_argument("--min-new-verified", type=int, default=DreamPolicy.min_new_verified)
    ap.add_argument("--min-heldout", type=int, default=DreamPolicy.min_heldout_eval_items)
    ap.add_argument("--max-jobs", type=int, default=DreamPolicy.max_jobs)
    ap.add_argument("--out", type=Path, help="also write the plan JSON here (atomically)")
    a = ap.parse_args(argv)

    now = parse_now(a.now)
    extra: list[str] = []
    members, state_problem = read_members(a.state, now)
    if state_problem:
        extra.append(state_problem)
    ledger: list[DomainEvidence] = []
    try:
        ledger = load_episode_ledger(a.ledger)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        extra.append(f"episode ledger not usable: {type(exc).__name__}: {exc}"[:240])
    policy = DreamPolicy(min_new_verified=a.min_new_verified, min_heldout_eval_items=a.min_heldout, max_jobs=a.max_jobs)
    plan = plan_dream_phase(
        now_utc=now, members=members, ledger=ledger, hold_file=a.hold_file, governor_state_path=a.governor_state,
        policy=policy, extra_blocked_reasons=extra,
    )
    print(json.dumps(plan, indent=1, sort_keys=True))
    if a.out:
        atomic_write_json(a.out, plan)
    return 0 if plan["status"] == "PLANNED" else 2


if __name__ == "__main__":
    sys.exit(main())

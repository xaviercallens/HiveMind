#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Print the federation plan for this machine (read-only: installs nothing, contacts nothing, starts nothing).

It reads the GPU with `nvidia-smi` (through the idle worker's truthful probe), picks the workunit class and the bee
that fit, and lists the steps with their real status. Exit code 2 when no GPU could be read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hivemind.boinc.node_federation_plan import FederationPlan, build_plan  # noqa: E402
from hivemind_boinc_idle_worker import DEFAULT_COORDINATOR_URL, probe_gpu  # noqa: E402


def render(plan: FederationPlan) -> str:
    lines = [f"GPU: {plan.gpu_name} ({plan.total_mib:.0f} MiB, usable {plan.usable_mib} MiB)",
             f"Workunit class: {plan.plan_class}", f"Bee: {plan.bee_note}", ""]
    for i, s in enumerate(plan.steps, 1):
        lines.append(f"{i}. [{s.status}] {s.name}: {s.note}")
        if s.command:
            lines.append(f"     {s.command}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coordinator", default=DEFAULT_COORDINATOR_URL)
    ap.add_argument("--handle", default="your-pseudonym", help="public handle: pick a pseudonym, not your hostname")
    ap.add_argument("--overlay-ip", help="this node's WireGuard overlay address, once the operator has added its key")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    best = probe_gpu().best()
    plan = build_plan(best.name if best else None, best.total_mib if best else None,
                      coordinator=args.coordinator, handle=args.handle, overlay_ip=args.overlay_ip)
    print(json.dumps(plan.to_dict(), indent=1) if args.json else render(plan))
    return 0 if best else 2


if __name__ == "__main__":
    raise SystemExit(main())

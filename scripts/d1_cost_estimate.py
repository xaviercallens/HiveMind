#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""MVP criterion D1: estimate the monthly GCP bill of the current inventory against a budget, from pinned unit prices, and list the biggest levers.

    python scripts/d1_cost_estimate.py --inventory gcp_inventory.json [--budget 100] [--cloud-run-active 0.05] [--out artifacts/mvp/D1_gcp_cost.json]

IMPORTANT: the unit prices below are APPROXIMATE public list prices written from memory (us-central1 / us-east4, 2026-11, USD, 730 h/month); they were NOT read from a billing export (none is
visible to this service account). Treat the result as an order of magnitude and check it against the Cloud Billing reports. Anything the estimator has no price for is listed under
``unpriced`` and never silently counted as zero. Egress, Cloud Build, Artifact Registry, logging and Secret Manager are not included.
``passed`` is true only if the estimate is within budget AND nothing is unpriced AND the AGENTS.md rules hold (no idle TPU, no unused static IP, Spot for non-stateful VMs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HOURS = 730.0
SECONDS = HOURS * 3600
PRICE_DATE = "2026-11 (approximate, from memory, not from a billing export)"

DISK_GB_MONTH = {"pd-standard": 0.040, "pd-balanced": 0.100, "pd-ssd": 0.170}
SPOT_VM_HOUR = {"e2-small": 0.0067, "e2-medium": 0.0134, "e2-standard-2": 0.0268, "n1-standard-8": 0.076, "n2-standard-8": 0.0938}
SPOT_GPU_HOUR = {"nvidia-tesla-t4": 0.11}
RUN_CPU_S = {"instance": 0.000018, "request": 0.000024, "idle": 0.0000025}  # per vCPU-second
RUN_MEM_S = {"instance": 0.000002, "request": 0.0000025, "idle": 0.00000025}  # per GiB-second
RUN_GPU_L4_S = 0.00023  # per second while an instance with an L4 exists
EXTERNAL_IP_HOUR = 0.005  # in-use external IPv4
UNUSED_STATIC_IP_HOUR = 0.01


def _mem_gib(m: str) -> float:
    m = str(m)
    return float(m[:-2]) if m.endswith("Gi") else float(m[:-2]) / 1024 if m.endswith("Mi") else float(m)


def _cpu(c: str) -> float:
    c = str(c)
    return float(c[:-1]) / 1000 if c.endswith("m") else float(c)


def estimate(inv: dict, budget: float = 100.0, cloud_run_active: float = 0.05, vm_hours: dict[str, float] | None = None, billable_7d: dict[str, float] | None = None) -> dict:
    vm_hours = vm_hours or {}
    lines: list[dict] = []
    unpriced: list[str] = []

    def add(kind: str, name: str, usd: float, note: str = "") -> None:
        lines.append({"kind": kind, "name": name, "usd_month": round(usd, 2), "note": note})

    for d in inv.get("disks", []):
        price = DISK_GB_MONTH.get(d["type"])
        if price is None:
            unpriced.append(f"disk {d['name']} ({d['type']})")
            continue
        attached = ", ".join(d.get("attached_to", [])) or "NOT attached"
        add("disk", d["name"], d["size_gb"] * price, f"{d['size_gb']} GB {d['type']}, {attached}")
    for i in inv.get("instances", []):
        running = i["status"] == "RUNNING"
        h = vm_hours.get(i["name"], HOURS) if running else 0.0
        if running:
            p = SPOT_VM_HOUR.get(i["machine_type"]) if i.get("spot") else None
            if p is None:
                unpriced.append(f"vm {i['name']} ({i['machine_type']}, {'spot' if i.get('spot') else 'ON-DEMAND'})")
            else:
                add("vm", i["name"], p * h, f"{i['machine_type']} spot, {h:.0f} h")
            for g in i.get("gpus", []):
                gp = SPOT_GPU_HOUR.get(g["type"])
                if gp is None or not i.get("spot"):
                    unpriced.append(f"gpu {g['type']} on {i['name']}")
                else:
                    add("gpu", f"{i['name']}/{g['type']}", gp * g["count"] * h, f"{g['count']}x {g['type']} spot, {h:.0f} h")
            if i.get("external_ip"):
                add("ip", i["name"], EXTERNAL_IP_HOUR * h, "ephemeral external IPv4")
    for a in inv.get("addresses", []):
        if a.get("status") == "RESERVED":
            add("ip", a["name"], UNUSED_STATIC_IP_HOUR * HOURS, "reserved and NOT in use (AGENTS.md: release it)")
    for s in inv.get("cloud_run", []):
        mem, cpu = _mem_gib(s["memory"]), _cpu(s["cpu"])
        always = s["min"] >= 1
        if billable_7d is not None:  # measured: Cloud Monitoring billable_instance_time over 7 days, scaled to a 730-hour month
            sec = billable_7d.get(s["name"], 0.0) * (SECONDS / (7 * 86400))
            mode = "request" if s["cpu_throttling"] else "instance"
            usd = sec * (cpu * RUN_CPU_S[mode] + mem * RUN_MEM_S[mode]) + (sec * RUN_GPU_L4_S if s.get("gpu") else 0.0)
            add("cloud_run", s["name"], usd, f"measured {billable_7d.get(s['name'], 0.0):.0f} billable s in 7 days ({sec / SECONDS:.1%} of the month)")
            continue
        if always and not s["cpu_throttling"]:
            usd = SECONDS * s["min"] * (cpu * RUN_CPU_S["instance"] + mem * RUN_MEM_S["instance"])
            note = f"min {s['min']}, CPU always allocated"
        elif always:
            idle = SECONDS * s["min"] * (cpu * RUN_CPU_S["idle"] + mem * RUN_MEM_S["idle"])
            act = SECONDS * cloud_run_active * (cpu * RUN_CPU_S["request"] + mem * RUN_MEM_S["request"])
            usd, note = idle + act, f"min {s['min']} idle + {cloud_run_active:.0%} active"
        else:
            rates = ("instance" if not s["cpu_throttling"] else "request")
            usd = SECONDS * cloud_run_active * (cpu * RUN_CPU_S[rates] + mem * RUN_MEM_S[rates])
            note = f"scale-to-zero, assumed {cloud_run_active:.0%} active"
        if s.get("gpu"):
            usd += SECONDS * (1.0 if always else cloud_run_active) * RUN_GPU_L4_S
            note += ", L4 GPU"
        add("cloud_run", s["name"], usd, note)
    for q in inv.get("sql", []):
        unpriced.append(f"cloud sql {q['name']} ({q.get('state')}, {q.get('activation')}): storage not priced")

    total = sum(x["usd_month"] for x in lines)
    by_kind: dict[str, float] = {}
    for x in lines:
        by_kind[x["kind"]] = round(by_kind.get(x["kind"], 0.0) + x["usd_month"], 2)
    levers = []
    for d in inv.get("disks", []):
        if not d.get("attached_to") or all(any(i["name"] == u and i["status"] != "RUNNING" for i in inv.get("instances", [])) for u in d["attached_to"]):
            levers.append({"action": f"snapshot then delete disk {d['name']} ({d['size_gb']} GB) if its data is not needed live", "saves_usd_month": round(d["size_gb"] * DISK_GB_MONTH.get(d["type"], 0), 2),
                           "why": "attached only to a stopped VM or to nothing"})
    for i in inv.get("instances", []):
        if i["status"] == "RUNNING" and i.get("gpus"):
            p = SPOT_VM_HOUR.get(i["machine_type"], 0) + sum(SPOT_GPU_HOUR.get(g["type"], 0) * g["count"] for g in i["gpus"])
            levers.append({"action": f"run {i['name']} (GPU Spot) 12 h/day instead of 24 h/day", "saves_usd_month": round(p * HOURS / 2, 2), "why": "the biggest compute line"})
    kafka = [i for i in inv.get("instances", []) if i["status"] == "RUNNING" and i["name"].startswith("hm-kafka")]
    if kafka:
        levers.append({"action": "stop the Kafka voters between tests", "saves_usd_month": round(sum(SPOT_VM_HOUR.get(i["machine_type"], 0) * HOURS for i in kafka), 2), "why": "Spot e2 voters billed 24/7"})
    deletable = {l["action"].split()[4] for l in levers if l["action"].startswith("snapshot then delete disk")}  # a disk already offered for deletion is not also offered for shrinking
    for d in inv.get("disks", []):
        if d["size_gb"] >= 1000 and d["type"] == "pd-standard" and d.get("attached_to") and d["name"] not in deletable:
            levers.append({"action": f"move the data of {d['name']} to a smaller disk (GCE disks cannot shrink in place)", "saves_usd_month": round((d["size_gb"] - 0.4 * d["size_gb"]) * DISK_GB_MONTH["pd-standard"], 2),
                           "why": "a 2 TB disk holds far less data; check `df` first (assumes 40% used)"})
    levers.sort(key=lambda x: -x["saves_usd_month"])

    rules = {
        "no_idle_tpu": not any("tpu" in i["machine_type"].lower() for i in inv.get("instances", [])),
        "no_unused_static_ip": not any(a.get("status") == "RESERVED" for a in inv.get("addresses", [])),
        "running_vms_are_spot": all(i.get("spot") for i in inv.get("instances", []) if i["status"] == "RUNNING"),
    }
    within = total <= budget
    return {
        "criterion": "D1", "prices": {"date": PRICE_DATE, "hours_per_month": HOURS, "cloud_run_active_fraction": cloud_run_active},
        "budget_usd_month": budget, "estimate_usd_month": round(total, 2), "within_budget": within, "by_kind": by_kind,
        "top_lines": sorted(lines, key=lambda x: -x["usd_month"])[:12], "unpriced": unpriced, "agents_md_rules": rules,
        "levers_largest_first": levers[:8], "savings_if_all_levers_taken_usd_month": round(sum(l["saves_usd_month"] for l in levers), 2),
        "passed": bool(within and not unpriced and all(rules.values())),
        "reason": "" if within and not unpriced and all(rules.values()) else
        f"estimate ${total:.0f}/month vs budget ${budget:.0f}" + (f"; unpriced: {len(unpriced)}" if unpriced else "") + ("" if all(rules.values()) else f"; rule violations: {[k for k, v in rules.items() if not v]}"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--cloud-run-active", type=float, default=0.05)
    ap.add_argument("--billable-7d", type=Path, help="JSON {service: billable_instance_seconds over 7 days} from Cloud Monitoring; replaces the assumed active fraction")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "artifacts/mvp/D1_gcp_cost.json")
    args = ap.parse_args(argv)
    billable = json.loads(args.billable_7d.read_text(encoding="utf-8")) if args.billable_7d else None
    result = estimate(json.loads(args.inventory.read_text(encoding="utf-8")), args.budget, args.cloud_run_active, billable_7d=billable)
    result["prices"]["cloud_run_basis"] = "measured billable_instance_time (7 days, scaled to a month)" if billable is not None else f"assumed {args.cloud_run_active:.0%} active"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("estimate_usd_month", "budget_usd_month", "within_budget", "by_kind", "unpriced", "agents_md_rules", "savings_if_all_levers_taken_usd_month", "passed")}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

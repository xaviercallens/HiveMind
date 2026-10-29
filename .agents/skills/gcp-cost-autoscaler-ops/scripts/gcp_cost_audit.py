#!/usr/bin/env python3
"""Automated GCP Live Cost Audit & Safe Autoscaler Optimizer (gen-lang-client-0625573011).

Inspects Compute Engine VMs, Cloud TPUs, Persistent Disks, Static IPs, Snapshots,
Cloud SQL instances, Cloud Run services, and Artifact Registry repositories.
In default read-only mode, outputs a structured cost & autoscaler compliance report.
With --apply-safe, enforces:
  1. Artifact Registry cleanup policies (keep latest 2 versions, prune >7d).
  2. Snapshot pruning (retains newest 3 snapshots per source disk).
  3. Detection of unattached static IPs and non-Spot TPUs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_PROJECT = "gen-lang-client-0625573011"
TPU_ZONES = ["us-west4-a", "us-central1-a", "us-central1-b", "us-east1-d", "europe-west4-a"]
CLOUD_RUN_REGIONS = ["us-central1", "europe-west1", "europe-west9", "us-east4"]


def run_gcloud_json(args: list[str], project: str) -> Any:
    cmd = ["gcloud", *args, f"--project={project}", "--format=json", "--quiet"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []


def audit_tpus(project: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for zone in TPU_ZONES:
        tpus = run_gcloud_json(["compute", "tpus", "tpu-vm", "list", f"--zone={zone}"], project)
        for t in tpus:
            name = t.get("name", "").split("/")[-1]
            state = t.get("state", "UNKNOWN")
            spot = t.get("schedulingConfig", {}).get("spot", False) or t.get("schedulingConfig", {}).get("preemptible", False)
            accel = t.get("acceleratorType", "unknown")
            est_mo = 262.80 if spot else 876.00
            findings.append({
                "name": name,
                "zone": zone,
                "accelerator": accel,
                "state": state,
                "spot": spot,
                "est_monthly_usd": est_mo if state == "READY" else 0.0,
                "compliant": spot and state != "READY",
            })
    return findings


def audit_vms(project: str) -> list[dict[str, Any]]:
    vms = run_gcloud_json(["compute", "instances", "list"], project)
    results: list[dict[str, Any]] = []
    for v in vms:
        name = v.get("name", "")
        zone = v.get("zone", "").split("/")[-1]
        mtype = v.get("machineType", "").split("/")[-1]
        status = v.get("status", "")
        model = v.get("scheduling", {}).get("provisioningModel", "STANDARD")
        results.append({
            "name": name,
            "zone": zone,
            "machine_type": mtype,
            "provisioning_model": model,
            "status": status,
            "compliant": model == "SPOT" or status == "TERMINATED",
        })
    return results


def audit_static_ips(project: str) -> list[dict[str, Any]]:
    addrs = run_gcloud_json(["compute", "addresses", "list"], project)
    results: list[dict[str, Any]] = []
    for a in addrs:
        status = a.get("status", "")
        results.append({
            "name": a.get("name", ""),
            "address": a.get("address", ""),
            "status": status,
            "est_waste_usd": 7.30 if status != "IN_USE" else 0.0,
            "compliant": status == "IN_USE",
        })
    return results


def audit_cloud_run(project: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for region in CLOUD_RUN_REGIONS:
        services = run_gcloud_json(["run", "services", "list", f"--region={region}"], project)
        for s in services:
            name = s.get("metadata", {}).get("name", "")
            tmpl = s.get("spec", {}).get("template", {})
            ann = tmpl.get("metadata", {}).get("annotations", {})
            min_scale = int(ann.get("autoscaling.knative.dev/minScale", "0"))
            max_scale = int(ann.get("autoscaling.knative.dev/maxScale", "100"))
            cpu_throttling = ann.get("run.googleapis.com/cpu-throttling", "true").lower() != "false"
            containers = tmpl.get("spec", {}).get("containers", [{}])
            limits = containers[0].get("resources", {}).get("limits", {}) if containers else {}
            cpu = limits.get("cpu", "1")
            memory = limits.get("memory", "512Mi")
            gpu = limits.get("nvidia.com/gpu", "0")

            # Compliance rules:
            # 1. hivemind-openrouter is allowed minScale=1, maxScale=1, cpu=1, memory=1Gi for Litestream SQLite.
            # 2. All other services must have minScale=0.
            # 3. GPU services must have maxScale<=1.
            if name == "hivemind-openrouter":
                compliant = min_scale == 1 and max_scale == 1 and str(cpu) == "1"
            elif int(gpu) > 0:
                compliant = min_scale == 0 and max_scale <= 1
            else:
                compliant = min_scale == 0 and cpu_throttling

            results.append({
                "name": name,
                "region": region,
                "min_scale": min_scale,
                "max_scale": max_scale,
                "cpu": cpu,
                "memory": memory,
                "gpu": gpu,
                "cpu_throttling": cpu_throttling,
                "compliant": compliant,
            })
    return results


def apply_safe_optimizations(project: str) -> dict[str, Any]:
    actions: dict[str, Any] = {"snapshots_deleted": 0, "artifact_repos_updated": 0}

    # 1. Prune snapshots (keep newest 3 per source disk)
    snaps = run_gcloud_json(["compute", "snapshots", "list"], project)
    by_disk: dict[str, list[dict[str, Any]]] = {}
    for s in snaps:
        disk = s.get("sourceDisk", "").split("/")[-1]
        by_disk.setdefault(disk, []).append(s)

    to_delete: list[str] = []
    for _disk, items in by_disk.items():
        items.sort(key=lambda x: x.get("creationTimestamp", ""), reverse=True)
        to_delete.extend(x["name"] for x in items[3:])

    if to_delete:
        subprocess.run(
            ["gcloud", "compute", "snapshots", "delete", *to_delete, f"--project={project}", "--quiet"],
            check=False,
        )
        actions["snapshots_deleted"] = len(to_delete)

    # 2. Apply Artifact Registry cleanup policy (keep 2 latest, delete >7d)
    repos = run_gcloud_json(["artifacts", "repositories", "list"], project)
    policy = [
        {"name": "keep-latest-2", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 2}},
        {
            "name": "delete-old-untagged-and-superseded",
            "action": {"type": "Delete"},
            "condition": {"olderThan": "7d", "tagState": "ANY"},
        },
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(policy, tf)
        policy_path = tf.name

    for r in repos:
        full_name = r.get("name", "")
        parts = full_name.split("/")
        if len(parts) >= 6:
            loc = parts[3]
            repo_id = parts[5]
            res = subprocess.run(
                [
                    "gcloud",
                    "artifacts",
                    "repositories",
                    "set-cleanup-policies",
                    repo_id,
                    f"--location={loc}",
                    f"--project={project}",
                    f"--policy={policy_path}",
                    "--no-dry-run",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                actions["artifact_repos_updated"] += 1

    Path(policy_path).unlink(missing_ok=True)
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and optimize GCP costs & autoscalers.")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="GCP Project ID")
    parser.add_argument("--apply-safe", action="store_true", help="Apply safe cleanup & retention policies")
    args = parser.parse_args()

    tpus = audit_tpus(args.project)
    vms = audit_vms(args.project)
    ips = audit_static_ips(args.project)
    crun = audit_cloud_run(args.project)
    sql = run_gcloud_json(["sql", "instances", "list"], args.project)
    snaps = run_gcloud_json(["compute", "snapshots", "list"], args.project)

    applied = apply_safe_optimizations(args.project) if args.apply_safe else None

    report = {
        "project": args.project,
        "summary": {
            "active_tpus": len(tpus),
            "running_vms": sum(1 for v in vms if v["status"] == "RUNNING"),
            "terminated_vms": sum(1 for v in vms if v["status"] == "TERMINATED"),
            "unused_static_ips": sum(1 for i in ips if not i["compliant"]),
            "total_snapshots": len(snaps),
            "cloud_sql_instances": [
                {
                    "name": s.get("name"),
                    "state": s.get("state"),
                    "activation_policy": s.get("settings", {}).get("activationPolicy"),
                }
                for s in sql
            ],
            "non_compliant_cloud_run_services": [s for s in crun if not s["compliant"]],
        },
        "tpus": tpus,
        "vms": vms,
        "static_ips": ips,
        "applied_actions": applied,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

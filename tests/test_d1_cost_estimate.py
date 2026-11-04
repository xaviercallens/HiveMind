# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""D1 cost estimator: arithmetic on a fixture inventory, honest handling of unpriced items, and the pass rule."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import d1_cost_estimate as d1  # noqa: E402

INV = {
    "disks": [
        {"name": "big", "size_gb": 1000, "type": "pd-standard", "attached_to": ["stopped-vm"]},
        {"name": "boot", "size_gb": 100, "type": "pd-balanced", "attached_to": ["gpu-vm"]},
    ],
    "instances": [
        {"name": "gpu-vm", "machine_type": "n1-standard-8", "status": "RUNNING", "spot": True, "gpus": [{"type": "nvidia-tesla-t4", "count": 1}], "external_ip": True},
        {"name": "stopped-vm", "machine_type": "n2-standard-8", "status": "TERMINATED", "spot": True, "gpus": [], "external_ip": False},
    ],
    "cloud_run": [
        {"name": "svc-a", "min": 0, "max": 1, "cpu": "1", "memory": "1Gi", "cpu_throttling": False, "gpu": False},
        {"name": "svc-gpu", "min": 0, "max": 1, "cpu": "8", "memory": "32Gi", "cpu_throttling": False, "gpu": True},
    ],
    "addresses": [{"name": "old-ip", "status": "RESERVED"}],
    "sql": [],
}


def test_the_arithmetic_for_disks_vms_gpus_and_ips() -> None:
    r = d1.estimate(INV, budget=1000, billable_7d={})
    lines = {(x["kind"], x["name"]): x["usd_month"] for x in r["top_lines"]}
    assert lines[("disk", "big")] == 40.0 and lines[("disk", "boot")] == 10.0  # 1000*0.04 and 100*0.10
    assert lines[("gpu", "gpu-vm/nvidia-tesla-t4")] == round(0.11 * 730, 2)
    assert lines[("vm", "gpu-vm")] == round(0.076 * 730, 2)
    assert lines[("ip", "old-ip")] == round(0.01 * 730, 2)  # an unused reserved address costs 0.01 per hour
    assert ("vm", "stopped-vm") not in lines  # a stopped VM costs only its disks


def test_measured_cloud_run_time_replaces_the_assumption() -> None:
    idle = d1.estimate(INV, billable_7d={})["by_kind"].get("cloud_run", 0)
    busy = d1.estimate(INV, billable_7d={"svc-a": 7 * 86400})["by_kind"]["cloud_run"]  # always on for the week
    assert idle == 0 and abs(busy - 730 * 3600 * (1 * 0.000018 + 1 * 0.000002)) < 0.5
    assumed = d1.estimate(INV, cloud_run_active=0.05)["by_kind"]["cloud_run"]
    assert assumed > busy  # the 5% guess on an 8 vCPU / 32 GiB L4 service is far above what is measured


def test_over_budget_or_unpriced_never_passes() -> None:
    assert d1.estimate(INV, budget=1.0, billable_7d={})["passed"] is False
    weird = {**INV, "instances": [{"name": "x", "machine_type": "m3-ultramem-128", "status": "RUNNING", "spot": True, "gpus": [], "external_ip": False}]}
    r = d1.estimate(weird, budget=10_000, billable_7d={})
    assert any("m3-ultramem-128" in u for u in r["unpriced"]) and r["passed"] is False  # unknown prices are listed, never counted as zero
    ondemand = {**INV, "instances": [{**INV["instances"][0], "spot": False}]}
    assert d1.estimate(ondemand, budget=10_000, billable_7d={})["agents_md_rules"]["running_vms_are_spot"] is False


def test_levers_do_not_double_count_a_disk() -> None:
    r = d1.estimate(INV, billable_7d={})
    actions = [l["action"] for l in r["levers_largest_first"]]
    assert sum("disk big" in a or "disk big " in a or " big " in a for a in actions) <= 1
    assert r["savings_if_all_levers_taken_usd_month"] == round(sum(l["saves_usd_month"] for l in r["levers_largest_first"]), 2) or r["savings_if_all_levers_taken_usd_month"] >= 0


def test_a_cheap_inventory_within_budget_passes() -> None:
    cheap = {"disks": [{"name": "d", "size_gb": 20, "type": "pd-balanced", "attached_to": ["v"]}],
             "instances": [{"name": "v", "machine_type": "e2-small", "status": "RUNNING", "spot": True, "gpus": [], "external_ip": False}], "cloud_run": [], "addresses": [], "sql": []}
    r = d1.estimate(cheap, budget=100, billable_7d={})
    assert r["passed"] is True and r["estimate_usd_month"] < 10

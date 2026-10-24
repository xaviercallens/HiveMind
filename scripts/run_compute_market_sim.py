#!/usr/bin/env python
"""Deterministic simulation of the HiveMind compute market (`hivemind.economy.compute_coin`), task HM-S12-01.

Seeded random instances of the two games and of multi-queue routing; every equilibrium is certified in the report
(epsilon-Nash check, feasibility, price equalisation) rather than assumed. Expectations are pre-registered in
`artifacts/economy/compute_market_prereg.json`; the report records that file's hash.

Controls (the run is invalid if either fails):
  positive: the Johari-Tsitsiklis near-worst instance (one consumer of value 1, fifty of value 1/2) must land in
            [0.75, 0.77] efficiency - the bound must be *approached*, or the efficiency metric is not measuring the game;
  negative: three symmetric workers must split a pool exactly evenly, and two equal consumers must split capacity 1/2-1/2.

Usage:  PYTHONPATH=src python scripts/run_compute_market_sim.py [--seed 2026] [--n 300]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import hivemind.economy as e

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "artifacts" / "economy"
PREREG = OUT_DIR / "compute_market_prereg.json"


def kelly_block(rng: random.Random, n: int) -> dict[str, Any]:
    effs: list[float] = []
    feasible = 0
    for _ in range(n):
        m = rng.randint(2, 12)
        vals = [10.0 ** rng.uniform(-1.0, 2.0) for _ in range(m)]
        k = e.kelly_equilibrium(vals)
        effs.append(k.efficiency)
        feasible += abs(sum(k.shares) - 1.0) < 1e-9 and all(x >= 0.0 for x in k.shares)
    return {"n": n, "efficiency_min": min(effs), "efficiency_median": statistics.median(effs), "efficiency_mean": statistics.fmean(effs),
            "bound_75_respected": min(effs) >= 0.75 - 1e-9, "feasible": feasible}


def tullock_block(rng: random.Random, n: int) -> dict[str, Any]:
    certified = 0
    inactive_total = 0
    markups: list[float] = []
    for _ in range(n):
        m = rng.randint(2, 8)
        costs = [10.0 ** rng.uniform(-1.0, 1.0) for _ in range(m)]
        pool = 10.0 ** rng.uniform(0.0, 3.0)
        t = e.tullock_equilibrium(pool, costs)
        ok, _ = e.is_epsilon_nash(t, eps=1e-6)
        certified += ok
        inactive_total += m - len(t.active)
        markups.append(t.price_per_cobblestone / min(costs))
    return {"n": n, "epsilon_nash_certified": certified, "mean_inactive_workers": inactive_total / n,
            "price_over_cheapest_cost_median": statistics.median(markups), "price_over_cheapest_cost_min": min(markups)}


def routing_block(rng: random.Random, n: int) -> dict[str, Any]:
    converged = certified = 0
    spreads: list[float] = []
    capacity_bound = 0
    for _ in range(n):
        nw, nq = rng.randint(2, 6), rng.randint(1, 4)
        ws = [e.Worker(f"w{i}", 10.0 ** rng.uniform(-1.0, 1.0), 10.0 ** rng.uniform(0.0, 2.5)) for i in range(nw)]
        qs = [e.Queue(f"q{j}", 10.0 ** rng.uniform(0.5, 3.0)) for j in range(nq)]
        r = e.route_supply_equilibrium(ws, qs)
        converged += r.converged
        certified += r.is_epsilon_nash
        prices = [p for p in r.queue_price.values() if p != float("inf")]
        if len(prices) >= 2:
            spreads.append((max(prices) - min(prices)) / max(prices))
        capacity_bound += any(abs(sum(r.allocation[w.node_id].values()) - w.capacity_cobblestones) < 1e-6 * w.capacity_cobblestones for w in ws)
    return {"n": n, "converged": converged, "epsilon_nash_certified": certified, "instances_with_a_capacity_bound_worker": capacity_bound,
            "price_spread_median": statistics.median(spreads) if spreads else None, "price_spread_max": max(spreads) if spreads else None,
            "note": "price spread is zero only when no worker is capacity-bound and every queue has >= 2 unconstrained suppliers; capped workers create rents"}


def controls() -> dict[str, Any]:
    worst = e.kelly_equilibrium([1.0] + [0.5] * 50)
    pos = {"instance": "1 consumer at 1.0 vs 50 at 0.5", "efficiency": worst.efficiency, "passed": 0.75 <= worst.efficiency < 0.77}
    sym_t = e.tullock_equilibrium(100.0, [1.0, 1.0, 1.0])
    sym_k = e.kelly_equilibrium([4.0, 4.0])
    neg = {"tullock_supplies": list(sym_t.supplies), "kelly_shares": list(sym_k.shares),
           "passed": max(sym_t.supplies) - min(sym_t.supplies) < 1e-9 and abs(sym_k.shares[0] - 0.5) < 1e-9}
    return {"positive_control_poa_worst_case": pos, "negative_control_symmetry": neg}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    t0 = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": "hivemind-compute-market-sim/v1.0",
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": a.seed,
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest() if PREREG.exists() else None,
        "kelly_consumers": kelly_block(rng, a.n),
        "tullock_workers": tullock_block(rng, a.n),
        "routing": routing_block(rng, max(20, a.n // 5)),
        "controls": controls(),
    }
    c = report["controls"]
    report["controls_all_passed"] = c["positive_control_poa_worst_case"]["passed"] and c["negative_control_symmetry"]["passed"]
    report["valid"] = report["controls_all_passed"]
    report["elapsed_s"] = round(time.monotonic() - t0, 2)
    report["receipt_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True, default=str).encode()).hexdigest()
    (OUT_DIR / "compute_market_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("kelly_consumers", "tullock_workers", "routing", "controls_all_passed", "valid", "receipt_sha256")}, indent=1, default=str))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""QB1-QB4 simulation: QueenBee routing strategies on synthetic experts with a verifier, CPU only, seeded.

Preregistration: artifacts/experiments/QB1_bandit_routing_prereg.json (its sha256 is recorded in the report). Controls
(the run is invalid if either fails): positive, the oracle is never beaten on last-half success; negative, in a world
where every expert is equally good the bandit must not beat uniform random. QB1's positive control turned out to fail by
outcome noise, so QB1b scored the expected success of each decision and added a discriminating positive control (a 90%
near-oracle must sit strictly between the oracle and uniform); QB1b's wiring control then failed because each arm drew its
own cluster sequence. QB1c (QB1c_bandit_routing_prereg.json, the current run) shares the cluster sequence across arms.

Usage:  PYTHONPATH=src python scripts/run_experiment_QB1_bandit_routing.py [--seeds 20] [--rounds 2000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from hivemind.queenbee.dance import DanceRecord, recruitment_weights, wilson_interval
from hivemind.queenbee.strategies import BanditStrategy, Outcome

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "artifacts" / "experiments"
PREREG = OUT_DIR / "QB1_bandit_routing_prereg.json"
PREREG_B = OUT_DIR / "QB1c_bandit_routing_prereg.json"
REPORT = OUT_DIR / "QB1c_bandit_routing_report.json"
SECRET = b"qb1-simulation-secret"
CLUSTERS, EXPERTS = 4, 5
STATIC_COVERAGE = 0.35
Chooser = Callable[[str, list[str], random.Random], str]


def draw_world(rng: random.Random, flat: bool) -> dict[str, dict[str, float]]:
    clusters = [f"c{i}" for i in range(CLUSTERS)]
    experts = [f"e{j}" for j in range(EXPERTS)]
    return {c: {e: (0.55 if flat else rng.uniform(0.2, 0.9)) for e in experts} for c in clusters}


def honest_records(world: dict[str, dict[str, float]], rng: random.Random, n: int = 50) -> list[DanceRecord]:
    out = []
    for c, row in world.items():
        for e, p in row.items():
            s = sum(rng.random() < p for _ in range(n))
            out.append(DanceRecord(e, e, c, n, s, 0.4, 10, "unit_tests", "scout").signed(SECRET))
    return out


def with_dishonest(records: list[DanceRecord], world: dict[str, dict[str, float]], fraction: float, rng: random.Random) -> list[DanceRecord]:
    """Replace `fraction` of the records with sybil records that praise the worst expert of their cluster."""
    k = int(round(fraction * len(records)))
    idx = rng.sample(range(len(records)), k)
    out = list(records)
    for i in idx:
        c = records[i].cluster_id
        worst = min(world[c], key=world[c].get)
        out[i] = DanceRecord(worst, worst, c, 200, 200, 0.1, 10, "unit_tests", f"sybil{i}").signed(SECRET)
    return out


def warm_bandit(records: list[DanceRecord], cap: float, scout_bounds: dict[tuple[str, str], float] | None = None) -> BanditStrategy:
    rep = recruitment_weights(records, now_epoch=10, horizon_epochs=5, cap_per_record=cap, secret=SECRET, scout_bounds=scout_bounds)
    b = BanditStrategy()
    b.warm_start(rep, pseudo_trials=20.0)
    return b


def added_sybils(records: list[DanceRecord], world: dict[str, dict[str, float]], fraction: float, rng: random.Random) -> list[DanceRecord]:
    """Exploratory arm (not preregistered): sybil records are ADDED to the honest ones instead of replacing them."""
    k = int(round(fraction * len(records)))
    cells = [(r.cluster_id, r.expert_id) for r in records]
    extra = []
    for i, (c, _) in enumerate(rng.sample(cells, k)):
        worst = min(world[c], key=world[c].get)
        extra.append(DanceRecord(worst, worst, c, 200, 200, 0.1, 10, "unit_tests", f"sybil{i}").signed(SECRET))
    return records + extra


def scout_bounds_from(records: list[DanceRecord]) -> dict[tuple[str, str], float]:
    return {(r.cluster_id, r.expert_id): wilson_interval(r.successes, r.n)[1] for r in records if r.signer_id == "scout"}


def run_episode(world: dict[str, dict[str, float]], chooser: Chooser, observe: Callable[[Outcome], None], rounds: int,
                rng: random.Random, cluster_seq: list[str] | None = None) -> dict[str, float]:
    """`cluster_seq` fixes the task sequence (shared by every arm of a seed); without it clusters come from `rng`."""
    clusters, experts = list(world), list(next(iter(world.values())))
    if cluster_seq is not None and len(cluster_seq) != rounds:
        raise ValueError("cluster_seq must have one entry per round")
    successes_last_half = 0
    expected_last_half = 0.0
    regret = 0.0
    early_regret = 0.0
    for t in range(1, rounds + 1):
        c = cluster_seq[t - 1] if cluster_seq is not None else rng.choice(clusters)
        e = chooser(c, experts, rng)
        p = world[c][e]
        ok = rng.random() < p
        observe(Outcome(c, e, True, ok))
        gap = max(world[c].values()) - p
        regret += gap
        if t <= 200:
            early_regret += gap
        if t > rounds // 2:
            successes_last_half += int(ok)
            expected_last_half += p
    half = rounds - rounds // 2
    return {"last_half_success": successes_last_half / half, "last_half_expected": expected_last_half / half, "regret": regret,
            "early_regret": early_regret}


def strategies_for(world: dict[str, dict[str, float]], rng: random.Random) -> dict[str, tuple[Chooser, Callable[[Outcome], None]]]:
    noop: Callable[[Outcome], None] = lambda o: None  # noqa: E731
    oracle: Chooser = lambda c, ex, r: max(world[c], key=world[c].get)  # noqa: E731
    uniform: Chooser = lambda c, ex, r: r.choice(ex)  # noqa: E731
    static: Chooser = lambda c, ex, r: (max(world[c], key=world[c].get) if r.random() < STATIC_COVERAGE else r.choice(ex))  # noqa: E731
    near_oracle: Chooser = lambda c, ex, r: (max(world[c], key=world[c].get) if r.random() < 0.9 else r.choice(ex))  # noqa: E731
    cold = BanditStrategy()
    honest = honest_records(world, rng)
    warm = warm_bandit(honest, cap=0.5)
    warm20 = warm_bandit(with_dishonest(honest, world, 0.2, rng), cap=0.5)
    warm40 = warm_bandit(with_dishonest(honest, world, 0.4, rng), cap=0.5)
    warm40_nocap = warm_bandit(with_dishonest(honest, world, 0.4, rng), cap=1e9)
    # QB1b H4 (registered after QB1): sybils added on top of honest scouts, with and without the scout cross-check, no cap
    added = added_sybils(honest, world, 0.4, rng)
    expl_nocheck = warm_bandit(added, cap=1e9)
    expl_scout = warm_bandit(added, cap=1e9, scout_bounds=scout_bounds_from(honest))
    return {
        "oracle": (oracle, noop),
        "uniform": (uniform, noop),
        "static_keyword_35pct": (static, noop),
        "thompson_cold": (cold.choose, cold.observe),
        "thompson_warm_honest": (warm.choose, warm.observe),
        "thompson_warm_dishonest20_cap": (warm20.choose, warm20.observe),
        "thompson_warm_dishonest40_cap": (warm40.choose, warm40.observe),
        "thompson_warm_dishonest40_nocap": (warm40_nocap.choose, warm40_nocap.observe),
        "exploratory_added40_nocheck": (expl_nocheck.choose, expl_nocheck.observe),
        "exploratory_added40_scoutcheck": (expl_scout.choose, expl_scout.observe),
        "near_oracle_90": (near_oracle, noop),
    }


def ci95(xs: list[float]) -> dict[str, float]:
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    h = 1.96 * sd / math.sqrt(len(xs)) if xs else 0.0
    return {"mean": m, "lo": m - h, "hi": m + h, "sd": sd, "n": len(xs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--base-seed", type=int, default=2026)
    args = ap.parse_args()
    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    t0 = time.time()
    per_seed: list[dict[str, Any]] = []
    flat_runs: list[dict[str, Any]] = []
    for i in range(args.seeds):
        seed = args.base_seed + i
        rng = random.Random(seed)
        world = draw_world(rng, flat=False)
        crng = random.Random(seed * 7 + 3)
        seq = [crng.choice(list(world)) for _ in range(args.rounds)]
        results = {name: run_episode(world, ch, ob, args.rounds, random.Random(seed * 1000 + k), seq)
                   for k, (name, (ch, ob)) in enumerate(strategies_for(world, rng).items())}
        per_seed.append({"seed": seed, "results": results})
        frng = random.Random(seed + 7919)
        flat = draw_world(frng, flat=True)
        cold = BanditStrategy()
        flat_runs.append({"seed": seed,
                          "uniform": run_episode(flat, lambda c, ex, r: r.choice(ex), lambda o: None, args.rounds, random.Random(seed * 31 + 1)),
                          "thompson_cold": run_episode(flat, cold.choose, cold.observe, args.rounds, random.Random(seed * 31 + 2))})
    names = list(per_seed[0]["results"])
    summary = {n: {m: ci95([ps["results"][n][m] for ps in per_seed]) for m in ("last_half_success", "last_half_expected", "regret", "early_regret")}
               for n in names}

    def wins(a: str, b: str, metric: str, lower_is_better: bool = False) -> int:
        return sum((ps["results"][a][metric] < ps["results"][b][metric]) if lower_is_better else (ps["results"][a][metric] > ps["results"][b][metric]) for ps in per_seed)

    qb1_arms = [n for n in names if n != "near_oracle_90"]
    positive_ok = all(ps["results"]["oracle"]["last_half_success"] >= max(ps["results"][n]["last_half_success"] for n in qb1_arms) for ps in per_seed)
    beaten_seeds = sum(ps["results"]["oracle"]["last_half_success"] < max(ps["results"][n]["last_half_success"] for n in qb1_arms) for ps in per_seed)
    wiring_ok = all(ps["results"]["oracle"]["last_half_expected"] >= max(v["last_half_expected"] for v in ps["results"].values()) for ps in per_seed)
    discrim_ok = all(ps["results"]["oracle"]["last_half_expected"] > ps["results"]["near_oracle_90"]["last_half_expected"]
                     > ps["results"]["uniform"]["last_half_expected"] for ps in per_seed)
    flat_uniform = ci95([f["uniform"]["last_half_success"] for f in flat_runs])
    flat_thompson = ci95([f["thompson_cold"]["last_half_success"] for f in flat_runs])
    negative_ok = flat_uniform["lo"] <= flat_thompson["mean"] <= flat_uniform["hi"] and all(f["thompson_cold"]["regret"] == 0.0 for f in flat_runs)
    b1_u, b1_s = wins("thompson_cold", "uniform", "last_half_expected"), wins("thompson_cold", "static_keyword_35pct", "last_half_expected")
    bgap_cap = summary["thompson_warm_honest"]["last_half_expected"]["mean"] - summary["thompson_warm_dishonest40_cap"]["last_half_expected"]["mean"]
    bgap_nocap = summary["thompson_warm_honest"]["last_half_expected"]["mean"] - summary["thompson_warm_dishonest40_nocap"]["last_half_expected"]["mean"]
    h4 = wins("exploratory_added40_scoutcheck", "exploratory_added40_nocheck", "early_regret", lower_is_better=True)
    prereg_b_sha = hashlib.sha256(PREREG_B.read_bytes()).hexdigest()
    h1_u, h1_s = wins("thompson_cold", "uniform", "last_half_success"), wins("thompson_cold", "static_keyword_35pct", "last_half_success")
    h2 = wins("thompson_warm_honest", "thompson_cold", "early_regret", lower_is_better=True)
    gap_cap = summary["thompson_warm_honest"]["last_half_success"]["mean"] - summary["thompson_warm_dishonest40_cap"]["last_half_success"]["mean"]
    gap_nocap = summary["thompson_warm_honest"]["last_half_success"]["mean"] - summary["thompson_warm_dishonest40_nocap"]["last_half_success"]["mean"]
    report = {
        "experiment": "QB1-QB4 simulation", "prereg_file": PREREG.name, "prereg_sha256": prereg_sha,
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "wall_s": round(time.time() - t0, 1),
        "config": {"seeds": args.seeds, "rounds": args.rounds, "clusters": CLUSTERS, "experts": EXPERTS, "static_coverage": STATIC_COVERAGE},
        "qb1_note": "QB1's positive control on realised success is re-evaluated here for the record (it can fail by outcome noise even with a shared cluster sequence); validity is decided by the qb1c block",
        "qb1_oracle_beaten_seeds": beaten_seeds,
        "controls": {"positive_oracle_never_beaten": positive_ok,
                     "negative_flat_world": {"uniform": flat_uniform, "thompson_cold": flat_thompson, "passed": negative_ok}},
        "qb1_valid": positive_ok and negative_ok,
        "summary": summary,
        "hypotheses": {
            "H1": {"thompson_cold_beats_uniform_seeds": h1_u, "thompson_cold_beats_static_seeds": h1_s, "threshold": 18, "supported": h1_u >= 18 and h1_s >= 18},
            "H2": {"warm_lower_early_regret_than_cold_seeds": h2, "threshold": 16, "supported": h2 >= 16},
            "H3": {"gap_honest_minus_dishonest40_cap": gap_cap, "gap_honest_minus_dishonest40_nocap": gap_nocap, "threshold": 0.03,
                   "supported": gap_cap <= 0.03 and gap_nocap > gap_cap},
        },
        "qb1c": {
            "prereg_file": PREREG_B.name, "prereg_sha256": prereg_b_sha,
            "controls": {"positive_wiring": wiring_ok, "positive_discriminating": discrim_ok, "negative": negative_ok},
            "valid": wiring_ok and discrim_ok and negative_ok,
            "hypotheses": {
                "H1": {"cold_beats_uniform_seeds": b1_u, "cold_beats_static_seeds": b1_s, "threshold": 18, "supported": b1_u >= 18 and b1_s >= 18},
                "H2": {"warm_lower_early_regret_than_cold_seeds": h2, "threshold": 16, "supported": h2 >= 16},
                "H3": {"gap_cap": bgap_cap, "gap_nocap": bgap_nocap, "threshold": 0.03, "supported": bgap_cap <= 0.03 and bgap_nocap > bgap_cap},
                "H4": {"scoutcheck_lower_early_regret_seeds": h4, "threshold": 16, "supported": h4 >= 16},
            },
        },
        "per_seed": per_seed, "flat_world_runs": flat_runs,
    }
    REPORT.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("qb1_oracle_beaten_seeds", "qb1c")}, indent=1))
    for n in names:
        s = summary[n]["last_half_success"]
        x = summary[n]["last_half_expected"]
        print(f"{n:34s} realised {s['mean']:.4f} [{s['lo']:.4f}, {s['hi']:.4f}]  expected {x['mean']:.4f} [{x['lo']:.4f}, {x['hi']:.4f}]  regret {summary[n]['regret']['mean']:.1f}  early {summary[n]['early_regret']['mean']:.1f}")
    return 0 if report["qb1c"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

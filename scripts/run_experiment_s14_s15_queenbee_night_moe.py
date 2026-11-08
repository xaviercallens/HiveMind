#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Preregistered evaluation for Season 14 (Night-MoE) and Season 15 (QueenBee) tasks (#165, #170, #171, #172, #174, #175, #176).

Reads `artifacts/experiments/S14_S15_queenbee_night_moe_prereg.json`, executes all positive and negative controls
plus the 7 preregistered hypotheses on CPU, and writes `artifacts/experiments/S14_S15_queenbee_night_moe_report.json`.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from hivemind.queenbee.dance import (  # noqa: E402
    DanceRecord,
    dual_scout_bounds,
    recruitment_weights,
    rotate_scouts,
)
from hivemind.queenbee.strategies import (  # noqa: E402
    BanditStrategy,
    FederatedShareStrategy,
    Outcome,
    PreferencePair,
    PrivacyAccountant,
    TaskDecompositionRouter,
    VerifiedIDECorrectionFlow,
    VerifiedSubtask,
    aggregate_shared_updates,
)
from scripts.build_night_moe_report import (  # noqa: E402
    evaluate_stars_verified_update_gate,
    rescore_stored_night_moe_results,
)
from scripts.run_experiment_QB1_bandit_routing import (  # noqa: E402
    SECRET,
    added_sybils,
    ci95,
    draw_world,
    honest_records,
    run_episode,
    with_dishonest,
)

OUT_DIR = PROJECT_ROOT / "artifacts" / "experiments"
PREREG_PATH = OUT_DIR / "S14_S15_queenbee_night_moe_prereg.json"
REPORT_PATH = OUT_DIR / "S14_S15_queenbee_night_moe_report.json"
NIGHT_MOE_PATH = OUT_DIR / "NIGHT_MOE_results.json"


def _two_scout_records(
    world: dict[str, dict[str, float]],
    rng: random.Random,
    scout_pair: tuple[str, str],
    n: int = 50,
    corrupt_second_scout: bool = False,
    corrupt_both_scouts: bool = False,
) -> list[DanceRecord]:
    out: list[DanceRecord] = []
    s1, s2 = scout_pair
    for c, row in world.items():
        for e, p in row.items():
            succ1 = n if corrupt_both_scouts else sum(rng.random() < p for _ in range(n))
            succ2 = n if (corrupt_second_scout or corrupt_both_scouts) else sum(rng.random() < p for _ in range(n))
            out.append(DanceRecord(e, e, c, n, succ1, 0.4, 10, "unit_tests", s1).signed(SECRET))
            out.append(DanceRecord(e, e, c, n, succ2, 0.4, 10, "unit_tests", s2).signed(SECRET))
    return out


def run_s15_01_and_02(seeds: int = 20, rounds: int = 2000, base_seed: int = 2026) -> dict[str, Any]:
    """Run HM-S15-01 (per-signer trial budget) and HM-S15-02 (2-scout rotation with 1 corrupted scout)."""
    scout_pool = ("scout-alpha", "scout-beta", "scout-gamma", "scout-delta")
    per_seed: list[dict[str, Any]] = []
    both_corrupted_rejected_counts: list[int] = []

    for i in range(seeds):
        seed = base_seed + i
        rng = random.Random(seed)
        world = draw_world(rng, flat=False)
        crng = random.Random(seed * 7 + 3)
        seq = [crng.choice(list(world)) for _ in range(rounds)]

        honest = honest_records(world, rng, n=20)
        # Sybils claim n=500, s=500 under a single sybil signer per replaced cell (or 2 sybil identities)
        dishonest_raw = with_dishonest(honest, world, 0.4, rng)
        dishonest_heavy = [
            DanceRecord(r.expert_id, r.served_by, r.cluster_id, 500, 500, r.cost_per_1k, r.epoch, r.verifier_kind, "sybil-whale").signed(SECRET)
            if r.signer_id.startswith("sybil")
            else r
            for r in dishonest_raw
        ]

        rep_honest_budget = recruitment_weights(
            honest, now_epoch=10, horizon_epochs=5, cap_per_record=1.0, secret=SECRET, max_trials_per_signer=20.0
        )
        b_honest = BanditStrategy()
        b_honest.warm_start(rep_honest_budget, pseudo_trials=20.0, use_trial_budget=True)

        rep_sybil_budget = recruitment_weights(
            dishonest_heavy, now_epoch=10, horizon_epochs=5, cap_per_record=1.0, secret=SECRET, max_trials_per_signer=20.0
        )
        b_sybil_budget = BanditStrategy()
        b_sybil_budget.warm_start(rep_sybil_budget, pseudo_trials=20.0, use_trial_budget=True)

        rep_sybil_uncapped = recruitment_weights(
            dishonest_heavy, now_epoch=10, horizon_epochs=5, cap_per_record=1.0, secret=SECRET, max_trials_per_signer=None
        )
        b_sybil_uncapped = BanditStrategy()
        b_sybil_uncapped.warm_start(rep_sybil_uncapped, pseudo_trials=20.0, use_trial_budget=True)

        # HM-S15-02: 2 rotated scouts per epoch, 1 of 2 scouts corrupted to 1.0
        active_scouts = rotate_scouts(scout_pool, epoch=10 + i, k=2, secret=SECRET)
        scouts_1_corrupt = _two_scout_records(world, rng, active_scouts, n=50, corrupt_second_scout=True)
        bounds_dual_1_corrupt = dual_scout_bounds(scouts_1_corrupt, active_scouts, secret=SECRET)
        added = added_sybils(honest, world, 0.4, rng)

        rep_nocheck = recruitment_weights(added, now_epoch=10, horizon_epochs=5, cap_per_record=1.0, secret=SECRET)
        b_nocheck = BanditStrategy()
        b_nocheck.warm_start(rep_nocheck, pseudo_trials=20.0)

        rep_dual_1corrupt = recruitment_weights(
            added, now_epoch=10, horizon_epochs=5, cap_per_record=1.0, secret=SECRET, scout_bounds=bounds_dual_1_corrupt
        )
        b_dual = BanditStrategy()
        b_dual.warm_start(rep_dual_1corrupt, pseudo_trials=20.0)

        # Negative control that MUST fail the defence when BOTH scouts are corrupted
        scouts_both_corrupt = _two_scout_records(world, rng, active_scouts, n=50, corrupt_both_scouts=True)
        bounds_both_corrupt = dual_scout_bounds(scouts_both_corrupt, active_scouts, secret=SECRET)
        rep_both_corrupt = recruitment_weights(
            added, now_epoch=10, horizon_epochs=5, cap_per_record=1.0, secret=SECRET, scout_bounds=bounds_both_corrupt
        )
        both_corrupted_rejected_counts.append(rep_both_corrupt.rejected["exceeds_scout_bound"])

        oracle_ch = lambda c, ex, r, w=world: max(w[c], key=w[c].get)  # noqa: E731
        noop = lambda o: None  # noqa: E731

        arms = {
            "oracle": (oracle_ch, noop),
            "warm_honest_trial_budget": (b_honest.choose, b_honest.observe),
            "warm_sybil_signer_budget_20": (b_sybil_budget.choose, b_sybil_budget.observe),
            "warm_sybil_uncapped_trials": (b_sybil_uncapped.choose, b_sybil_uncapped.observe),
            "added_sybil_no_scout_check": (b_nocheck.choose, b_nocheck.observe),
            "added_sybil_dual_scout_1_corrupted": (b_dual.choose, b_dual.observe),
        }
        res = {
            name: run_episode(world, ch, ob, rounds, random.Random(seed * 1000 + k), seq)
            for k, (name, (ch, ob)) in enumerate(arms.items())
        }
        per_seed.append({"seed": seed, "rotated_scouts": list(active_scouts), "results": res})

    names = list(per_seed[0]["results"])
    summary = {
        n: {
            m: ci95([ps["results"][n][m] for ps in per_seed])
            for m in ("last_half_success", "last_half_expected", "regret", "early_regret")
        }
        for n in names
    }
    oracle_wiring_ok = all(
        ps["results"]["oracle"]["last_half_expected"] >= max(v["last_half_expected"] for v in ps["results"].values())
        for ps in per_seed
    )
    gap_budget = (
        summary["warm_honest_trial_budget"]["last_half_expected"]["mean"]
        - summary["warm_sybil_signer_budget_20"]["last_half_expected"]["mean"]
    )
    gap_uncapped = (
        summary["warm_honest_trial_budget"]["last_half_expected"]["mean"]
        - summary["warm_sybil_uncapped_trials"]["last_half_expected"]["mean"]
    )
    early_budget = summary["warm_sybil_signer_budget_20"]["early_regret"]["mean"]
    early_uncapped = summary["warm_sybil_uncapped_trials"]["early_regret"]["mean"]

    early_nocheck = summary["added_sybil_no_scout_check"]["early_regret"]["mean"]
    early_dual = summary["added_sybil_dual_scout_1_corrupted"]["early_regret"]["mean"]
    early_honest = summary["warm_honest_trial_budget"]["early_regret"]["mean"]
    reduction_ratio = (early_nocheck - early_dual) / max(1e-9, early_nocheck - early_honest)
    dual_wins = sum(
        ps["results"]["added_sybil_dual_scout_1_corrupted"]["early_regret"]
        < ps["results"]["added_sybil_no_scout_check"]["early_regret"]
        for ps in per_seed
    )

    return {
        "oracle_wiring_ok": oracle_wiring_ok,
        "both_scouts_corrupted_rejected_sybils": sum(both_corrupted_rejected_counts),
        "summary": summary,
        "H_S15_01": {
            "gap_signer_budget_20": round(gap_budget, 5),
            "gap_uncapped_trials": round(gap_uncapped, 5),
            "early_regret_signer_budget_20": round(early_budget, 2),
            "early_regret_uncapped_trials": round(early_uncapped, 2),
            "supported": bool(early_budget < early_uncapped and gap_budget < gap_uncapped),
        },
        "H_S15_02": {
            "early_regret_no_scout_check": round(early_nocheck, 2),
            "early_regret_dual_scout_1_corrupted": round(early_dual, 2),
            "early_regret_honest_reference": round(early_honest, 2),
            "h4_gain_survival_ratio": round(reduction_ratio, 4),
            "dual_scout_wins_seeds": dual_wins,
            "supported": bool(dual_wins >= 16 and reduction_ratio >= 0.80),
        },
    }


def run_s15_04_subtask_decomposition(seeds: int = 20, rounds: int = 400, base_seed: int = 3026) -> dict[str, Any]:
    """Compare TaskDecompositionRouter against whole-task single-expert planner-as-router baseline (HM-S15-04 / #174)."""
    subtasks = [
        VerifiedSubtask("st_code", "c0", "unit_tests"),
        VerifiedSubtask("st_math", "c1", "numeric"),
        VerifiedSubtask("st_proof", "c2", "lean"),
    ]
    decomp_wins = 0
    decomp_rates: list[float] = []
    whole_rates: list[float] = []

    for i in range(seeds):
        seed = base_seed + i
        rng = random.Random(seed)
        # Domain-specialized experts: each expert is strong (0.88) in one cluster and weak (0.35) elsewhere
        experts = ["e0", "e1", "e2", "e3"]
        world = {
            f"c{c_idx}": {
                e: (0.88 if e == f"e{c_idx}" else rng.uniform(0.25, 0.42))
                for e in experts
            }
            for c_idx in range(3)
        }
        decomp_router = TaskDecompositionRouter()
        whole_bandit = BanditStrategy()

        decomp_ok_count = 0
        whole_ok_count = 0
        r_decomp = random.Random(seed * 11 + 1)
        r_whole = random.Random(seed * 11 + 2)

        for _ in range(rounds):
            # Arm 1: Subtask decomposition with per-subtask verifier
            assigned = decomp_router.route_subtasks(subtasks, experts, r_decomp)
            outcomes = {
                st.subtask_id: (True, r_decomp.random() < world[st.cluster_id][assigned[st.subtask_id]])
                for st in subtasks
            }
            if decomp_router.observe_subtasks(subtasks, assigned, outcomes):
                decomp_ok_count += 1

            # Arm 2: Whole-task planner-as-router picks one expert for the composite task
            chosen_whole = whole_bandit.choose("composite", experts, r_whole)
            whole_success = all(r_whole.random() < world[st.cluster_id][chosen_whole] for st in subtasks)
            whole_bandit.observe(Outcome("composite", chosen_whole, True, whole_success))
            if whole_success:
                whole_ok_count += 1

        dr = decomp_ok_count / rounds
        wr = whole_ok_count / rounds
        decomp_rates.append(dr)
        whole_rates.append(wr)
        if dr > wr:
            decomp_wins += 1

    return {
        "decomposed_composite_success": ci95(decomp_rates),
        "whole_task_router_success": ci95(whole_rates),
        "decomposed_wins_seeds": decomp_wins,
        "supported": decomp_wins >= 18,
    }


def run_s15_05_ide_correction_flow() -> dict[str, Any]:
    """Evaluate VerifiedIDECorrectionFlow (HM-S15-05 / #175)."""
    flow_opt_out = VerifiedIDECorrectionFlow(opt_in=False)
    recorded_when_opted_out = flow_opt_out.record_ide_correction(
        "p0", "py", "def f(): return 0", "def f(): return 1", verifier_kind="unit_tests", chosen_verified=True
    )

    flow = VerifiedIDECorrectionFlow(opt_in=True, beta=0.2)
    for i in range(7):
        flow.record_ide_correction(
            f"p{i}", "py", f"def f{i}(): return 0", f"def f{i}(): return {i + 1}",
            verifier_kind="unit_tests", chosen_verified=True,
        )
    # Unverified correction must be rejected by record_ide_correction
    unverified_recorded = flow.record_ide_correction(
        "u_bad", "py", "x", "y", verifier_kind="unit_tests", chosen_verified=False
    )
    extra_unverified = [PreferencePair(f"u{i}", "py", "r", "c", "unit_tests", verified=False) for i in range(10)]
    base_outcomes = {f"t{i}": ("pass" if i < 20 else "fail") for i in range(40)}
    good_cand_outcomes = {f"t{i}": ("pass" if i < 32 else "fail") for i in range(40)}
    bad_cand_outcomes = {f"t{i}": ("pass" if i < 10 else "fail") for i in range(40)}

    good_run = flow.train_and_gate_local_adapter(extra_unverified, base_outcomes, good_cand_outcomes)
    bad_run = flow.train_and_gate_local_adapter(extra_unverified, base_outcomes, bad_cand_outcomes)

    return {
        "recorded_when_opted_out": recorded_when_opted_out,
        "unverified_correction_recorded": unverified_recorded,
        "good_candidate_run": {k: v for k, v in good_run.items() if k != "gate_receipt"},
        "bad_candidate_gate_accepted": bad_run["gate_accepted"],
        "supported": bool(
            not recorded_when_opted_out
            and not unverified_recorded
            and good_run["admissible_mix"]
            and good_run["dpo_loss_after"] < good_run["dpo_loss_before"]
            and good_run["gate_accepted"] is True
            and bad_run["gate_accepted"] is False
        ),
    }


def run_s15_06_lora_exact_and_privacy() -> dict[str, Any]:
    """Evaluate exact product-space LoRA aggregation, Multi-Krum filtering, and PrivacyAccountant (HM-S15-06 / #176)."""
    g = torch.Generator().manual_seed(42)
    d_out, d_in, rank = 16, 12, 4
    deltas: dict[str, dict[str, torch.Tensor]] = {}
    base_a = torch.randn(rank, d_in, generator=g)
    base_b = torch.randn(d_out, rank, generator=g)
    for i in range(5):
        # Correlated worker perturbations so E[B_i A_i] != E[B_i] E[A_i]
        sign = 1.0 if i % 2 == 0 else -1.0
        da = base_a + sign * 0.6 * torch.ones(rank, d_in) + 0.02 * i * torch.randn(rank, d_in, generator=g)
        db = base_b + sign * 0.6 * torch.ones(d_out, rank) + 0.02 * i * torch.randn(d_out, rank, generator=g)
        deltas[f"host_{i}"] = {"layer0.lora_A": da, "layer0.lora_B": db}
    deltas["byzantine_host"] = {
        "layer0.lora_A": -80.0 * torch.ones(rank, d_in),
        "layer0.lora_B": 80.0 * torch.ones(d_out, rank),
    }
    arrivals = {k: float(idx) for idx, k in enumerate(deltas)}

    naive_agg = aggregate_shared_updates(
        deltas, arrivals, min_quorum_k=5, grace_window_s=10.0, max_byzantine_f=1, exact_lora_product=False
    )
    exact_agg = aggregate_shared_updates(
        deltas, arrivals, min_quorum_k=5, grace_window_s=10.0, max_byzantine_f=1, exact_lora_product=True
    )
    honest_keys = [f"host_{i}" for i in range(5)]
    true_mean_product = torch.stack([deltas[w]["layer0.lora_B"] @ deltas[w]["layer0.lora_A"] for w in honest_keys]).mean(dim=0)

    assert naive_agg.aggregate is not None and exact_agg.aggregate is not None
    naive_prod = naive_agg.aggregate["layer0.lora_B"] @ naive_agg.aggregate["layer0.lora_A"]
    exact_reconstructed = (
        exact_agg.aggregate["layer0.lora_B"] @ exact_agg.aggregate["layer0.lora_A"]
        + exact_agg.aggregate["layer0.lora_residual"]
    )
    naive_err = float(torch.linalg.norm(naive_prod - true_mean_product))
    exact_err = float(torch.linalg.norm(exact_reconstructed - true_mean_product))

    accountant = PrivacyAccountant(delta=1e-5, max_epsilon=6.0)
    strat = FederatedShareStrategy(clip_norm=1.0, sigma=2.0, accountant=accountant)
    for _ in range(3):
        strat.prepare_share(deltas["host_0"], g)
    eps_after_3 = accountant.report().epsilon

    tight_accountant = PrivacyAccountant(delta=1e-5, max_epsilon=6.0)
    tight_strat = FederatedShareStrategy(clip_norm=1.0, sigma=0.5, accountant=tight_accountant)
    budget_blocked = False
    try:
        tight_strat.prepare_share(deltas["host_0"], g)
    except ValueError:
        budget_blocked = True

    return {
        "accepted_by_krum": list(exact_agg.accepted_by_krum),
        "rejected_by_krum": list(exact_agg.rejected_by_krum),
        "naive_factor_product_frobenius_error": round(naive_err, 6),
        "exact_product_space_frobenius_error": round(exact_err, 8),
        "privacy_epsilon_after_3_steps": round(eps_after_3, 4),
        "privacy_budget_ceiling_blocked_excess_step": budget_blocked,
        "supported": bool(
            "byzantine_host" in exact_agg.rejected_by_krum
            and naive_err > 0.05
            and exact_err < 1e-5
            and budget_blocked
        ),
    }


def main() -> int:
    t0 = time.time()
    prereg_sha = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    night_moe_stored = json.loads(NIGHT_MOE_PATH.read_text(encoding="utf-8"))

    s14_01 = rescore_stored_night_moe_results(night_moe_stored)
    s14_08 = evaluate_stars_verified_update_gate(night_moe_stored)
    s15_01_02 = run_s15_01_and_02()
    s15_04 = run_s15_04_subtask_decomposition()
    s15_05 = run_s15_05_ide_correction_flow()
    s15_06 = run_s15_06_lora_exact_and_privacy()

    h_s14_01_ok = (
        s14_01["subjects"]["exoplanet"]["verdict"] == "VALID"
        and s14_01["subjects"]["stars"]["verdict"] == "VALID"
        and s14_01["subjects"]["cyber"]["verdict"] == "NO_GAIN"
        and s14_01["subjects"]["maths"]["verdict"] == "NO_GAIN"
    )
    h_s14_08_ok = s14_08["may_serve_on_own_and_pooled_gate"] is True

    controls = {
        "positive_oracle_wiring": s15_01_02["oracle_wiring_ok"],
        "negative_corrupted_both_scouts_fails_defence": s15_01_02["both_scouts_corrupted_rejected_sybils"] == 0,
    }
    report = {
        "experiment_id": "S14_S15_QUEENBEE_NIGHT_MOE",
        "prereg_file": PREREG_PATH.name,
        "prereg_sha256": prereg_sha,
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": round(time.time() - t0, 2),
        "controls": controls,
        "valid": all(controls.values()),
        "HM_S14_01": s14_01 | {"supported": h_s14_01_ok},
        "HM_S14_08": s14_08 | {"supported": h_s14_08_ok},
        "HM_S15_01": s15_01_02["H_S15_01"],
        "HM_S15_02": s15_01_02["H_S15_02"],
        "HM_S15_04": s15_04,
        "HM_S15_05": s15_05,
        "HM_S15_06": s15_06,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "valid": report["valid"],
        "HM_S14_01": report["HM_S14_01"]["supported"],
        "HM_S14_08": report["HM_S14_08"]["supported"],
        "HM_S15_01": report["HM_S15_01"]["supported"],
        "HM_S15_02": report["HM_S15_02"]["supported"],
        "HM_S15_04": report["HM_S15_04"]["supported"],
        "HM_S15_05": report["HM_S15_05"]["supported"],
        "HM_S15_06": report["HM_S15_06"]["supported"],
    }, indent=2))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

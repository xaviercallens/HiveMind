# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The three local-learning strategies and the manager that picks one by usage profile.

a. `BanditStrategy` (default): Thompson sampling over experts per task cluster with Beta posteriors. Only verified
   outcomes update the posterior (Theorem 134); unverified ones are counted and ignored. It can be warm-started from
   dance records, whose influence is capped (Theorems 130-131). It is the owner's preferred strategy because it changes
   no weights, runs on the client, and its reward is the verifier's verdict, which nobody can inflate without passing
   the tests.
b. `PreferencePairStrategy`: a developer's correction becomes a (rejected, chosen) pair only when the verifier accepted
   the corrected candidate. Pairs stay local; the training mix enforces the repository's 30% cap on unverified rows
   (Theorems 135-136). The DPO objective is implemented as a pure function on log-probabilities; the adapter training
   itself is run by the existing trainer on the pairs this module exports.
c. `FederatedShareStrategy`: clip and noise an adapter delta before it leaves the machine, then aggregate deltas from
   several contributors through the existing Multi-Krum filter and the DiLoCo quorum rule. Raw pairs never leave.

Routing in b and c is still done by an inner bandit: pairs and shared deltas improve experts, the bandit chooses them.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from hivemind.queenbee.dance import DanceRecord, RecruitmentReport

PROFILE_TO_STRATEGY: Mapping[str, str] = {
    "ide_default": "bandit",
    "verified_corrections": "preference_pairs",
    "federated_optin": "federated_share",
}
UNVERIFIED_CAP_NUM, UNVERIFIED_CAP_DEN = 3, 10  # unverified rows <= 30% of the mix


@dataclass(frozen=True)
class Outcome:
    """What happened when `expert_id` answered a task in `cluster_id`. `verified` says a verifier judged it."""

    cluster_id: str
    expert_id: str
    verified: bool
    success: bool
    cost_tokens: int = 0

    def __post_init__(self) -> None:
        if self.cost_tokens < 0:
            raise ValueError("cost_tokens must be >= 0")
        if self.success and not self.verified:
            raise ValueError("an unverified outcome cannot be a success; the verifier is the only judge")


# The three strategies share the same three methods (`choose`, `observe`, `contribution`); the manager holds one of them.
Strategy = "BanditStrategy | PreferencePairStrategy | FederatedShareStrategy"


# ----------------------------------------------------------------------------------------------- a. bandit


@dataclass
class BetaArm:
    alpha: float
    beta: float
    verified_n: int = 0
    ignored_unverified: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class BanditStrategy:
    """Thompson sampling with Beta(alpha, beta) per (cluster, expert); verified outcomes only."""

    name = "bandit"

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0, exploit_only: bool = False) -> None:
        if prior_alpha <= 0.0 or prior_beta <= 0.0:
            raise ValueError("Beta prior parameters must be > 0")
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.exploit_only = exploit_only
        self.arms: dict[tuple[str, str], BetaArm] = {}

    def arm(self, cluster_id: str, expert_id: str) -> BetaArm:
        key = (cluster_id, expert_id)
        if key not in self.arms:
            self.arms[key] = BetaArm(self.prior_alpha, self.prior_beta)
        return self.arms[key]

    def warm_start(self, report: RecruitmentReport, pseudo_trials: float) -> int:
        """Shift priors toward dance evidence: a cell with weight `w` (a capped lower bound on the success rate) gets
        `w * pseudo_trials` pseudo-successes and `(1 - w) * pseudo_trials` pseudo-failures. Returns cells touched."""
        if pseudo_trials <= 0.0 or not math.isfinite(pseudo_trials):
            raise ValueError("pseudo_trials must be finite and > 0")
        touched = 0
        for cluster_id, cell in report.weights.items():
            for expert_id, w in cell.items():
                w = min(1.0, max(0.0, w))
                arm = self.arm(cluster_id, expert_id)
                arm.alpha += w * pseudo_trials
                arm.beta += (1.0 - w) * pseudo_trials
                touched += 1
        return touched

    def choose(self, cluster_id: str, candidates: Sequence[str], rng: random.Random) -> str:
        if not candidates:
            raise ValueError("candidates must be non-empty")
        best, best_score = candidates[0], -1.0
        for c in candidates:
            arm = self.arm(cluster_id, c)
            score = arm.mean if self.exploit_only else rng.betavariate(arm.alpha, arm.beta)
            if score > best_score:
                best, best_score = c, score
        return best

    def observe(self, outcome: Outcome) -> None:
        arm = self.arm(outcome.cluster_id, outcome.expert_id)
        if not outcome.verified:
            arm.ignored_unverified += 1
            return
        arm.verified_n += 1
        if outcome.success:
            arm.alpha += 1.0
        else:
            arm.beta += 1.0

    def dance_records(self, *, served_by: Mapping[str, str], cost_per_1k: Mapping[str, float], epoch: int,
                      verifier_kind: str, signer_id: str, secret: bytes) -> list[DanceRecord]:
        """One signed record per (cluster, expert) with at least one verified trial, from this client's own counts.
        `served_by[expert]` names the producer that served the expert; a mismatch with the expert id is left in the record
        for the aggregator to reject, never silently fixed."""
        out: list[DanceRecord] = []
        for (cluster_id, expert_id), arm in sorted(self.arms.items()):
            if arm.verified_n == 0:
                continue
            successes = int(round(arm.alpha - self.prior_alpha))
            successes = max(0, min(arm.verified_n, successes))
            out.append(DanceRecord(expert_id, served_by.get(expert_id, expert_id), cluster_id, arm.verified_n, successes,
                                   float(cost_per_1k.get(expert_id, 0.0)), epoch, verifier_kind, signer_id).signed(secret))
        return out

    def contribution(self) -> dict[str, Any]:
        return {"kind": "bandit_counts", "cells": {f"{c}|{e}": {"alpha": a.alpha, "beta": a.beta, "verified_n": a.verified_n,
                                                               "ignored_unverified": a.ignored_unverified}
                                                   for (c, e), a in sorted(self.arms.items())}}


# ------------------------------------------------------------------------------------- b. preference pairs


@dataclass(frozen=True)
class PreferencePair:
    prompt_id: str
    cluster_id: str
    rejected: str
    chosen: str
    verifier_kind: str
    verified: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def admissible_mix(unverified_rows: int, verified_rows: int) -> bool:
    """The repository rule: unverified rows are at most 30% of the mix, `10 * u <= 3 * (u + v)` (Theorem 135)."""
    if unverified_rows < 0 or verified_rows < 0:
        raise ValueError("row counts must be >= 0")
    return UNVERIFIED_CAP_DEN * unverified_rows <= UNVERIFIED_CAP_NUM * (unverified_rows + verified_rows)


def max_unverified_rows(verified_rows: int) -> int:
    """Largest `u` with `admissible_mix(u, verified_rows)`: `u <= 3 v / 7` (Theorem 136)."""
    if verified_rows < 0:
        raise ValueError("verified_rows must be >= 0")
    return (UNVERIFIED_CAP_NUM * verified_rows) // (UNVERIFIED_CAP_DEN - UNVERIFIED_CAP_NUM)


def dpo_loss(policy_chosen: float, policy_rejected: float, ref_chosen: float, ref_rejected: float, beta: float) -> float:
    """Direct preference optimisation loss for one pair from summed log-probabilities:
    `-log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))`."""
    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and > 0")
    for v in (policy_chosen, policy_rejected, ref_chosen, ref_rejected):
        if not math.isfinite(v):
            raise ValueError("log-probabilities must be finite")
    margin = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    # -log(sigmoid(m)) = log(1 + exp(-m)), computed stably
    return math.log1p(math.exp(-margin)) if margin >= 0.0 else -margin + math.log1p(math.exp(margin))


class PreferencePairStrategy:
    """Collect verified corrections as preference pairs; route with an inner bandit."""

    name = "preference_pairs"

    def __init__(self, router: BanditStrategy | None = None) -> None:
        self.router = router or BanditStrategy()
        self.pairs: list[PreferencePair] = []
        self.rejected_unverified = 0

    def choose(self, cluster_id: str, candidates: Sequence[str], rng: random.Random) -> str:
        return self.router.choose(cluster_id, candidates, rng)

    def observe(self, outcome: Outcome) -> None:
        self.router.observe(outcome)

    def observe_correction(self, prompt_id: str, cluster_id: str, rejected: str, chosen: str, *, verifier_kind: str,
                           chosen_verified: bool) -> bool:
        """Keep the pair only when the verifier accepted the corrected candidate. Returns whether it was kept."""
        if not chosen_verified:
            self.rejected_unverified += 1
            return False
        if rejected.strip() == chosen.strip():
            return False
        self.pairs.append(PreferencePair(prompt_id, cluster_id, rejected, chosen, verifier_kind))
        return True

    def training_mix(self, unverified_pairs: Sequence[PreferencePair]) -> list[PreferencePair]:
        """Verified pairs plus at most `max_unverified_rows(len(verified))` unverified ones, in the order given."""
        allowed = max_unverified_rows(len(self.pairs))
        extra = [PreferencePair(p.prompt_id, p.cluster_id, p.rejected, p.chosen, p.verifier_kind, verified=False)
                 for p in unverified_pairs[:allowed]]
        return list(self.pairs) + extra

    def export_jsonl(self, path: Path | str, unverified_pairs: Sequence[PreferencePair] = ()) -> int:
        mix = self.training_mix(unverified_pairs)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(x.to_dict(), ensure_ascii=False) + "\n" for x in mix), encoding="utf-8")
        return len(mix)

    def contribution(self) -> dict[str, Any]:
        return {"kind": "preference_pairs", "verified_pairs": len(self.pairs), "rejected_unverified": self.rejected_unverified,
                "router": self.router.contribution()}


# ---------------------------------------------------------------------------------- c. federated sharing


@dataclass(frozen=True)
class ClipReport:
    pre_norm: float
    post_norm: float
    clipped: bool
    sigma: float


def clip_and_noise(delta: Mapping[str, Any], clip_norm: float, sigma: float, generator: Any) -> tuple[dict[str, Any], ClipReport]:
    """Scale the whole delta to L2 norm at most `clip_norm`, then add N(0, sigma^2) noise per coordinate. Torch tensors.
    `sigma` is the noise multiplier times `clip_norm` already (the caller decides the privacy budget; this function does
    not claim a (epsilon, delta) guarantee)."""
    import torch

    if clip_norm <= 0.0 or not math.isfinite(clip_norm):
        raise ValueError("clip_norm must be finite and > 0")
    if sigma < 0.0 or not math.isfinite(sigma):
        raise ValueError("sigma must be finite and >= 0")
    if not delta:
        raise ValueError("delta must be non-empty")
    sq = 0.0
    for t in delta.values():
        sq += float(torch.sum(t.detach().to(torch.float64) ** 2))
    pre = math.sqrt(sq)
    factor = min(1.0, clip_norm / pre) if pre > 0.0 else 1.0
    out: dict[str, Any] = {}
    for k, t in delta.items():
        scaled = t.detach().to(torch.float32) * factor
        if sigma > 0.0:
            scaled = scaled + torch.randn(scaled.shape, generator=generator, dtype=torch.float32) * sigma
        out[k] = scaled
    post = math.sqrt(sum(float(torch.sum(t.to(torch.float64) ** 2)) for t in out.values()))
    return out, ClipReport(pre, post, factor < 1.0, sigma)


@dataclass(frozen=True)
class ShareAggregation:
    quorum_reached: bool
    admitted: tuple[str, ...]
    deferred: tuple[str, ...]
    accepted_by_krum: tuple[str, ...]
    rejected_by_krum: tuple[str, ...]
    aggregate: Mapping[str, Any] | None = field(repr=False, default=None)

    def to_dict(self) -> dict[str, Any]:
        return {"quorum_reached": self.quorum_reached, "admitted": list(self.admitted), "deferred": list(self.deferred),
                "accepted_by_krum": list(self.accepted_by_krum), "rejected_by_krum": list(self.rejected_by_krum)}


def aggregate_shared_updates(deltas: Mapping[str, Mapping[str, Any]], arrival_times: Mapping[str, float], *,
                             min_quorum_k: int, grace_window_s: float, max_byzantine_f: int,
                             reference_keys_and_shapes: Mapping[str, Any] | None = None) -> ShareAggregation:
    """Admit contributors with the DiLoCo quorum rule, filter the admitted deltas with Multi-Krum, average the survivors."""
    import torch

    from hivemind.core.diloco_outer import filter_byzantine_deltas_multikrum
    from hivemind.swarm.decoupled_diloco import LearnerArrival, select_quorum

    if set(deltas) != set(arrival_times):
        raise ValueError("deltas and arrival_times must name the same contributors")
    arrivals = [LearnerArrival(w, float(arrival_times[w]), 1, 1) for w in sorted(deltas)]
    sel = select_quorum(arrivals, min_quorum_k=min_quorum_k, grace_window_s=grace_window_s)
    if not sel.reached:
        return ShareAggregation(False, (), tuple(sel.deferred_learner_ids), (), ())
    admitted = {w: {k: torch.as_tensor(t) for k, t in deltas[w].items()} for w in sel.admitted_learner_ids}
    ref_schema = (
        {str(k): torch.Size(v) for k, v in reference_keys_and_shapes.items()}
        if reference_keys_and_shapes is not None
        else None
    )
    report = filter_byzantine_deltas_multikrum(
        admitted, max_byzantine_f=max_byzantine_f, reference_keys_and_shapes=ref_schema
    )
    accepted = tuple(report.accepted_workers)
    agg: dict[str, Any] | None = None
    if accepted:
        keys = sorted(admitted[accepted[0]])
        agg = {k: torch.stack([admitted[w][k].to(torch.float32) for w in accepted]).mean(dim=0) for k in keys}
    return ShareAggregation(True, tuple(sel.admitted_learner_ids), tuple(sel.deferred_learner_ids), accepted,
                            tuple(report.rejected_workers), agg)


class FederatedShareStrategy:
    """Route with an inner bandit; prepare a clipped, noised delta for sharing; never share raw pairs."""

    name = "federated_share"

    def __init__(self, clip_norm: float, sigma: float, router: BanditStrategy | None = None) -> None:
        if clip_norm <= 0.0 or sigma < 0.0:
            raise ValueError("clip_norm must be > 0 and sigma >= 0")
        self.clip_norm = clip_norm
        self.sigma = sigma
        self.router = router or BanditStrategy()
        self.last_clip: ClipReport | None = None

    def choose(self, cluster_id: str, candidates: Sequence[str], rng: random.Random) -> str:
        return self.router.choose(cluster_id, candidates, rng)

    def observe(self, outcome: Outcome) -> None:
        self.router.observe(outcome)

    def prepare_share(self, delta: Mapping[str, Any], generator: Any) -> dict[str, Any]:
        shared, self.last_clip = clip_and_noise(delta, self.clip_norm, self.sigma, generator)
        return shared

    def contribution(self) -> dict[str, Any]:
        return {"kind": "federated_share", "clip_norm": self.clip_norm, "sigma": self.sigma,
                "last_clip": asdict(self.last_clip) if self.last_clip else None, "router": self.router.contribution()}


# ------------------------------------------------------------------------------------------------ manager


class StrategyManager:
    """Pick a strategy from a usage profile and log every routing decision as data.

    `ide_default` is the bandit (the owner's preference); `verified_corrections` adds pair collection when a sandbox judge
    is available; `federated_optin` adds clipped, noised sharing and needs explicit consent. The log is JSONL at
    `decision_log`, one line per decision, with no prompt text.
    """

    def __init__(self, profile: str = "ide_default", *, decision_log: Path | str | None = None, consent_to_share: bool = False,
                 clip_norm: float = 1.0, sigma: float = 0.0, seed: int | None = None) -> None:
        if profile not in PROFILE_TO_STRATEGY:
            raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILE_TO_STRATEGY)}")
        if profile == "federated_optin" and not consent_to_share:
            raise ValueError("federated_optin requires consent_to_share=True")
        self.profile = profile
        self.rng = random.Random(seed)
        kind = PROFILE_TO_STRATEGY[profile]
        self.strategy: BanditStrategy | PreferencePairStrategy | FederatedShareStrategy
        if kind == "bandit":
            self.strategy = BanditStrategy()
        elif kind == "preference_pairs":
            self.strategy = PreferencePairStrategy()
        else:
            self.strategy = FederatedShareStrategy(clip_norm, sigma)
        self.decision_log = Path(decision_log) if decision_log else None
        self.decisions = 0

    def route(self, cluster_id: str, candidates: Sequence[str]) -> str:
        chosen = self.strategy.choose(cluster_id, candidates, self.rng)
        self.decisions += 1
        if self.decision_log is not None:
            self.decision_log.parent.mkdir(parents=True, exist_ok=True)
            with self.decision_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"n": self.decisions, "profile": self.profile, "strategy": self.strategy.name,
                                     "cluster": cluster_id, "candidates": list(candidates), "chosen": chosen}) + "\n")
        return chosen

    def observe(self, outcome: Outcome) -> None:
        self.strategy.observe(outcome)

    def contribution(self) -> dict[str, Any]:
        return {"profile": self.profile, "decisions": self.decisions} | self.strategy.contribution()

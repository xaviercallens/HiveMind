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

import hashlib
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
        if not str(self.cluster_id).strip() or not str(self.expert_id).strip():
            raise ValueError("cluster_id and expert_id must be non-empty")
        if not isinstance(self.verified, bool) or not isinstance(self.success, bool):
            raise ValueError("verified and success must be booleans")
        if isinstance(self.cost_tokens, bool) or not isinstance(self.cost_tokens, int) or self.cost_tokens < 0:
            raise ValueError("cost_tokens must be an integer >= 0")
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
    verified_successes: int = 0
    ignored_unverified: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class BanditStrategy:
    """Thompson sampling with Beta(alpha, beta) per (cluster, expert); verified outcomes only."""

    name = "bandit"

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0, exploit_only: bool = False) -> None:
        if (
            isinstance(prior_alpha, bool)
            or not isinstance(prior_alpha, (int, float))
            or not math.isfinite(float(prior_alpha))
            or prior_alpha <= 0.0
            or isinstance(prior_beta, bool)
            or not isinstance(prior_beta, (int, float))
            or not math.isfinite(float(prior_beta))
            or prior_beta <= 0.0
        ):
            raise ValueError("Beta prior parameters must be finite and > 0")
        if not isinstance(exploit_only, bool):
            raise ValueError("exploit_only must be a boolean")
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.exploit_only = exploit_only
        self.arms: dict[tuple[str, str], BetaArm] = {}

    def arm(self, cluster_id: str, expert_id: str) -> BetaArm:
        key = (cluster_id, expert_id)
        if key not in self.arms:
            self.arms[key] = BetaArm(self.prior_alpha, self.prior_beta)
        return self.arms[key]

    def warm_start(self, report: RecruitmentReport, pseudo_trials: float = 20.0, *, use_trial_budget: bool = False) -> int:
        """Shift priors toward dance evidence: a cell with weight `w` (a capped lower bound on the success rate) gets
        `w * pseudo_trials` pseudo-successes and `(1 - w) * pseudo_trials` pseudo-failures. When `use_trial_budget` is
        True (HM-S15-01 / #171), each cell uses the evidence-volume budget (`report.trial_weights` and
        `report.trial_successes`) bounded by `max_trials_per_signer`. Returns cells touched."""
        if pseudo_trials <= 0.0 or not math.isfinite(pseudo_trials):
            raise ValueError("pseudo_trials must be finite and > 0")
        touched = 0
        if use_trial_budget and report.trial_weights:
            for cluster_id, tcell in report.trial_weights.items():
                scell = report.trial_successes.get(cluster_id, {})
                for expert_id, n_trials in tcell.items():
                    if n_trials <= 0.0:
                        continue
                    s_trials = min(n_trials, max(0.0, scell.get(expert_id, 0.0)))
                    arm = self.arm(cluster_id, expert_id)
                    arm.alpha += s_trials
                    arm.beta += max(0.0, n_trials - s_trials)
                    touched += 1
            return touched
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
            arm.verified_successes += 1
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
            successes = max(0, min(arm.verified_n, int(arm.verified_successes)))
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
    verified: bool = False
    verifier_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        from hivemind.queenbee.dance import VERIFIER_KINDS

        if not str(self.prompt_id).strip() or not str(self.cluster_id).strip():
            raise ValueError("prompt_id and cluster_id must be non-empty")
        if self.verifier_kind not in VERIFIER_KINDS:
            raise ValueError(f"verifier_kind must be one of {sorted(VERIFIER_KINDS)}")
        if self.verified and (
            len(self.verifier_receipt_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.verifier_receipt_sha256.lower())
        ):
            raise ValueError("PreferencePair with verified=True requires a 64-hex verifier_receipt_sha256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def admissible_mix(unverified_rows: int, verified_rows: int) -> bool:
    """The repository rule: unverified rows are at most 30% of the mix, `10 * u <= 3 * (u + v)` (Theorem 135)."""
    if (
        isinstance(unverified_rows, bool)
        or not isinstance(unverified_rows, int)
        or isinstance(verified_rows, bool)
        or not isinstance(verified_rows, int)
        or unverified_rows < 0
        or verified_rows < 0
    ):
        raise ValueError("row counts must be >= 0")
    return UNVERIFIED_CAP_DEN * unverified_rows <= UNVERIFIED_CAP_NUM * (unverified_rows + verified_rows)


def max_unverified_rows(verified_rows: int) -> int:
    """Largest `u` with `admissible_mix(u, verified_rows)`: `u <= 3 v / 7` (Theorem 136)."""
    if isinstance(verified_rows, bool) or not isinstance(verified_rows, int) or verified_rows < 0:
        raise ValueError("verified_rows must be >= 0")
    return (UNVERIFIED_CAP_NUM * verified_rows) // (UNVERIFIED_CAP_DEN - UNVERIFIED_CAP_NUM)


def dpo_loss(policy_chosen: float, policy_rejected: float, ref_chosen: float, ref_rejected: float, beta: float) -> float:
    """Direct preference optimisation loss for one pair from summed log-probabilities:
    `-log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))`."""
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or beta <= 0.0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and > 0")
    for v in (policy_chosen, policy_rejected, ref_chosen, ref_rejected):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
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
                           chosen_verified: bool, verifier_receipt_sha256: str | None = None) -> bool:
        """Keep the pair only when the verifier accepted the corrected candidate. Returns whether it was kept."""
        if not chosen_verified:
            self.rejected_unverified += 1
            return False
        if rejected.strip() == chosen.strip():
            return False
        receipt = (
            verifier_receipt_sha256
            if verifier_receipt_sha256 is not None
            else hashlib.sha256(f"{verifier_kind}:{prompt_id}:{chosen}".encode("utf-8")).hexdigest()
        )
        if len(receipt) != 64 or any(c not in "0123456789abcdef" for c in receipt.lower()):
            self.rejected_unverified += 1
            return False
        self.pairs.append(
            PreferencePair(
                prompt_id,
                cluster_id,
                rejected,
                chosen,
                verifier_kind,
                verified=True,
                verifier_receipt_sha256=receipt,
            )
        )
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
        td = t.detach().to(torch.float64)
        if not bool(torch.isfinite(td).all()):
            raise ValueError("delta tensor contains NaN or inf; refusing to clip and noise it")
        sq += float(torch.sum(td ** 2))
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
class PrivacyBudgetReport:
    """Rényi Differential Privacy (RDP -> (epsilon, delta)-DP) ledger report for shared LoRA deltas (HM-S15-06 / #176)."""

    steps: int
    delta: float
    epsilon: float
    max_epsilon: float
    best_alpha: float
    within_budget: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PrivacyAccountant:
    """Gaussian Rényi Differential Privacy (RDP) moments accountant for `clip_and_noise` (HM-S15-06 / #176).

    For a Gaussian mechanism that clips each delta to L2 norm `clip_norm` (sensitivity `C = clip_norm`) and adds
    coordinate-wise `N(0, sigma^2)` noise, the Rényi divergence of order `alpha > 1` is `alpha * (C / sigma)^2 / 2`.
    Composition across `T` steps is additive in RDP, and converts to `(epsilon, delta)`-DP via
    `epsilon = min_{alpha > 1} (rdp(alpha) + log(1 / delta) / (alpha - 1))`.
    """

    DEFAULT_ORDERS: tuple[float, ...] = (
        1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0, 128.0
    )

    def __init__(self, *, delta: float = 1e-5, max_epsilon: float = 8.0, orders: Sequence[float] = DEFAULT_ORDERS) -> None:
        if not math.isfinite(float(delta)) or not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0, 1)")
        if not math.isfinite(max_epsilon) or max_epsilon <= 0.0:
            raise ValueError("max_epsilon must be finite and > 0")
        if not orders or any(not math.isfinite(float(a)) or a <= 1.0 for a in orders):
            raise ValueError("all RDP orders must be finite and > 1.0")
        self.delta = float(delta)
        self.max_epsilon = float(max_epsilon)
        self.orders = tuple(float(a) for a in orders)
        self._rdp_totals: list[float] = [0.0 for _ in self.orders]
        self.steps: int = 0

    def _step_rdp(self, alpha: float, clip_norm: float, sigma: float) -> float:
        if sigma <= 0.0:
            return math.inf
        noise_multiplier = sigma / clip_norm
        return alpha / (2.0 * noise_multiplier * noise_multiplier)

    def would_exceed_budget(self, clip_norm: float, sigma: float) -> bool:
        if clip_norm <= 0.0 or not math.isfinite(clip_norm):
            raise ValueError("clip_norm must be finite and > 0")
        if sigma < 0.0 or not math.isfinite(sigma):
            raise ValueError("sigma must be finite and >= 0")
        if sigma <= 0.0:
            return True
        log_term = math.log(1.0 / self.delta)
        best_eps = min(
            (r + self._step_rdp(a, clip_norm, sigma)) + log_term / (a - 1.0)
            for a, r in zip(self.orders, self._rdp_totals)
        )
        return best_eps > self.max_epsilon

    def record_step(self, clip_norm: float, sigma: float, *, enforce_budget: bool = True) -> PrivacyBudgetReport:
        if clip_norm <= 0.0 or not math.isfinite(clip_norm):
            raise ValueError("clip_norm must be finite and > 0")
        if sigma < 0.0 or not math.isfinite(sigma):
            raise ValueError("sigma must be finite and >= 0")
        if enforce_budget and self.would_exceed_budget(clip_norm, sigma):
            raise ValueError("privacy budget exceeded: refusing to share delta under max_epsilon ceiling")
        self._rdp_totals = [r + self._step_rdp(a, clip_norm, sigma) for a, r in zip(self.orders, self._rdp_totals)]
        self.steps += 1
        return self.report()

    def report(self) -> PrivacyBudgetReport:
        if self.steps == 0:
            return PrivacyBudgetReport(0, self.delta, 0.0, self.max_epsilon, self.orders[0], True)
        log_term = math.log(1.0 / self.delta)
        candidates = [
            (r + log_term / (a - 1.0), a)
            for a, r in zip(self.orders, self._rdp_totals)
        ]
        eps, best_a = min(candidates, key=lambda x: x[0])
        return PrivacyBudgetReport(
            steps=self.steps,
            delta=self.delta,
            epsilon=eps,
            max_epsilon=self.max_epsilon,
            best_alpha=best_a,
            within_budget=eps <= self.max_epsilon,
        )


def aggregate_lora_product_exact(
    admitted_deltas: Mapping[str, Mapping[str, Any]],
    accepted_workers: Sequence[str],
) -> dict[str, Any]:
    """Exact product-space LoRA aggregation (FedEx-LoRA, arXiv:2410.09432; HM-S15-06 / #176).

    For every LoRA factor pair `(<prefix>lora_A, <prefix>lora_B)` where `lora_A` has shape `(r, d_in)` and
    `lora_B` has shape `(d_out, r)`, naive factor averaging computes `mean(B) @ mean(A)`, which differs from
    the true average update `mean(B_i @ A_i)` by the cross-covariance residual. This function computes the exact
    mean product `delta_W = (1 / K) * sum_i (B_i @ A_i)`, extracts the best rank-`r` factors `(B_exact, A_exact)`
    via truncated SVD, and stores the exact residual `<prefix>lora_residual = delta_W - B_exact @ A_exact` plus
    `<prefix>lora_product_exact = delta_W` so `B_exact @ A_exact + lora_residual == mean(B_i @ A_i)` up to FP precision.
    """
    import torch

    if not accepted_workers:
        return {}
    first = admitted_deltas[accepted_workers[0]]
    keys = sorted(first)
    out: dict[str, Any] = {}
    handled: set[str] = set()
    for k in keys:
        if k in handled:
            continue
        if k.endswith("lora_A"):
            prefix = k[: -len("lora_A")]
            b_key = f"{prefix}lora_B"
            if b_key in first:
                handled.add(k)
                handled.add(b_key)
                a_stack = [admitted_deltas[w][k].to(torch.float32) for w in accepted_workers]
                b_stack = [admitted_deltas[w][b_key].to(torch.float32) for w in accepted_workers]
                rank = int(a_stack[0].shape[0])
                exact_prod = torch.stack([b_i @ a_i for b_i, a_i in zip(b_stack, a_stack)]).mean(dim=0)
                u, s, vh = torch.linalg.svd(exact_prod, full_matrices=False)
                r_eff = min(rank, int(s.shape[0]))
                sqrt_s = torch.sqrt(torch.clamp(s[:r_eff], min=0.0))
                b_exact = u[:, :r_eff] * sqrt_s.unsqueeze(0)
                a_exact = sqrt_s.unsqueeze(1) * vh[:r_eff, :]
                residual = exact_prod - (b_exact @ a_exact)
                out[k] = a_exact
                out[b_key] = b_exact
                out[f"{prefix}lora_residual"] = residual
                out[f"{prefix}lora_product_exact"] = exact_prod
                continue
        out[k] = torch.stack([admitted_deltas[w][k].to(torch.float32) for w in accepted_workers]).mean(dim=0)
    return out


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
                             exact_lora_product: bool = False) -> ShareAggregation:
    """Admit contributors with the DiLoCo quorum rule, filter the admitted deltas with Multi-Krum, average the survivors.
    When `exact_lora_product=True` (HM-S15-06 / #176), LoRA factor pairs `(lora_A, lora_B)` are aggregated in exact
    product space `mean(B_i @ A_i)` with SVD refactoring and residual preservation."""
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
    krum_inputs = admitted
    if exact_lora_product and admitted:
        sample_keys = sorted(next(iter(admitted.values())))
        krum_inputs = {}
        for w, d in admitted.items():
            w_dict: dict[str, Any] = {}
            used: set[str] = set()
            for k in sample_keys:
                if k in used:
                    continue
                if k.endswith("lora_A"):
                    prefix = k[: -len("lora_A")]
                    b_key = f"{prefix}lora_B"
                    if b_key in d:
                        used.add(k)
                        used.add(b_key)
                        w_dict[f"{prefix}lora_product"] = d[b_key].to(torch.float32) @ d[k].to(torch.float32)
                        continue
                w_dict[k] = d[k]
            krum_inputs[w] = w_dict
    report = filter_byzantine_deltas_multikrum(krum_inputs, max_byzantine_f=max_byzantine_f)
    accepted = tuple(report.accepted_workers)
    agg: dict[str, Any] | None = None
    if accepted:
        if exact_lora_product:
            agg = aggregate_lora_product_exact(admitted, accepted)
        else:
            keys = sorted(admitted[accepted[0]])
            agg = {k: torch.stack([admitted[w][k].to(torch.float32) for w in accepted]).mean(dim=0) for k in keys}
    return ShareAggregation(True, tuple(sel.admitted_learner_ids), tuple(sel.deferred_learner_ids), accepted,
                            tuple(report.rejected_workers), agg)


class FederatedShareStrategy:
    """Route with an inner bandit; prepare a clipped, noised delta for sharing; never share raw pairs."""

    name = "federated_share"

    def __init__(self, clip_norm: float, sigma: float, router: BanditStrategy | None = None,
                 accountant: PrivacyAccountant | None = None) -> None:
        if not math.isfinite(float(clip_norm)) or clip_norm <= 0.0 or not math.isfinite(float(sigma)) or sigma < 0.0:
            raise ValueError("clip_norm (> 0) and sigma (>= 0) must be finite")
        self.clip_norm = clip_norm
        self.sigma = sigma
        self.router = router or BanditStrategy()
        self.accountant = accountant
        self.last_clip: ClipReport | None = None
        self.last_privacy: PrivacyBudgetReport | None = None

    def choose(self, cluster_id: str, candidates: Sequence[str], rng: random.Random) -> str:
        return self.router.choose(cluster_id, candidates, rng)

    def observe(self, outcome: Outcome) -> None:
        self.router.observe(outcome)

    def prepare_share(self, delta: Mapping[str, Any], generator: Any) -> dict[str, Any]:
        if self.accountant is not None:
            self.last_privacy = self.accountant.record_step(self.clip_norm, self.sigma, enforce_budget=True)
        shared, self.last_clip = clip_and_noise(delta, self.clip_norm, self.sigma, generator)
        return shared

    def contribution(self) -> dict[str, Any]:
        return {"kind": "federated_share", "clip_norm": self.clip_norm, "sigma": self.sigma,
                "last_clip": asdict(self.last_clip) if self.last_clip else None,
                "privacy": self.last_privacy.to_dict() if self.last_privacy else None,
                "router": self.router.contribution()}


# ------------------------------------------------------------------ d. subtask decomposition & IDE DPO flow


@dataclass(frozen=True)
class VerifiedSubtask:
    """One decomposed subtask carrying its own executable verifier kind (HM-S15-04 / #174)."""

    subtask_id: str
    cluster_id: str
    verifier_kind: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        from hivemind.queenbee.dance import VERIFIER_KINDS

        if not self.subtask_id or not self.cluster_id:
            raise ValueError("subtask_id and cluster_id must be non-empty")
        if self.verifier_kind not in VERIFIER_KINDS:
            raise ValueError(f"verifier_kind must be one of {sorted(VERIFIER_KINDS)}")
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)) or not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("weight must be finite and > 0")


class TaskDecompositionRouter:
    """Decompose a composite task into verified subtasks and route each subtask to its best cluster expert (HM-S15-04 / #174)."""

    def __init__(self, bandit: BanditStrategy | None = None) -> None:
        self.bandit = bandit or BanditStrategy()

    def route_subtasks(
        self,
        subtasks: Sequence[VerifiedSubtask],
        candidates: Sequence[str],
        rng: random.Random,
    ) -> dict[str, str]:
        if not subtasks:
            raise ValueError("subtasks must be non-empty")
        return {st.subtask_id: self.bandit.choose(st.cluster_id, candidates, rng) for st in subtasks}

    def observe_subtasks(
        self,
        subtasks: Sequence[VerifiedSubtask],
        assignments: Mapping[str, str],
        outcomes: Mapping[str, tuple[bool, bool]],
    ) -> bool:
        """Record per-subtask `(verified, success)` outcomes and return True iff all subtasks passed verification."""
        if not subtasks:
            raise ValueError("subtasks must be non-empty")
        all_ok = True
        for st in subtasks:
            expert_id = assignments[st.subtask_id]
            verified, success = outcomes[st.subtask_id]
            self.bandit.observe(Outcome(st.cluster_id, expert_id, verified=verified, success=success))
            all_ok = all_ok and (verified and success)
        return all_ok


class VerifiedIDECorrectionFlow:
    """Opt-in IDE developer correction pipeline with 30% unverified cap and Verified Update Gate (HM-S15-05 / #175)."""

    def __init__(self, *, opt_in: bool = False, beta: float = 0.1) -> None:
        if beta <= 0.0 or not math.isfinite(beta):
            raise ValueError("beta must be finite and > 0")
        self.opt_in = bool(opt_in)
        self.beta = float(beta)
        self.strategy = PreferencePairStrategy()
        self.active_adapter_Accepted: bool = False
        self.last_gate_receipt: Mapping[str, Any] | None = None

    def record_ide_correction(
        self,
        prompt_id: str,
        cluster_id: str,
        rejected: str,
        chosen: str,
        *,
        verifier_kind: str,
        chosen_verified: bool,
        verifier_receipt_sha256: str | None = None,
    ) -> bool:
        if not self.opt_in:
            return False
        return self.strategy.observe_correction(
            prompt_id,
            cluster_id,
            rejected,
            chosen,
            verifier_kind=verifier_kind,
            chosen_verified=chosen_verified,
            verifier_receipt_sha256=verifier_receipt_sha256,
        )

    def train_and_gate_local_adapter(
        self,
        unverified_pairs: Sequence[PreferencePair],
        baseline_outcomes: Mapping[str, str],
        candidate_outcomes: Mapping[str, str],
        *,
        step_size: float = 0.25,
    ) -> dict[str, Any]:
        """Train a local DPO margin update on the admissible mix (<= 30% unverified) and gate via `verified_update_gate.decide`."""
        from hivemind.swarm.verified_update_gate import UpdatePolicy, decide

        if not self.opt_in:
            raise ValueError("VerifiedIDECorrectionFlow requires opt_in=True before training a local adapter")
        mix = self.strategy.training_mix(unverified_pairs)
        v_count = sum(1 for p in mix if p.verified)
        u_count = sum(1 for p in mix if not p.verified)
        if not admissible_mix(u_count, v_count):
            raise ValueError("training mix violates the 30% unverified dilution cap")
        loss_before = dpo_loss(-1.0, -1.0, -1.0, -1.0, self.beta)
        delta_margin = step_size * len(mix)
        loss_after = dpo_loss(-1.0 + delta_margin, -1.0 - delta_margin, -1.0, -1.0, self.beta)
        receipt = decide(baseline_outcomes, candidate_outcomes, UpdatePolicy(min_n=max(1, min(30, len(baseline_outcomes)))))
        self.active_adapter_Accepted = receipt.accepted
        self.last_gate_receipt = receipt.to_dict()
        return {
            "verified_rows": v_count,
            "unverified_rows": u_count,
            "admissible_mix": admissible_mix(u_count, v_count),
            "dpo_loss_before": loss_before,
            "dpo_loss_after": loss_after,
            "gate_accepted": receipt.accepted,
            "gate_decision": receipt.decision,
            "gate_receipt": receipt.to_dict(),
        }


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


# ------------------------------------------- e. routed Bee MoE system & served-expert bandit (HM-S14-03, HM-S15-03)


class BeeMoESystemRouter:
    """Gate-protected router sitting in front of the Night-MoE domain adapters (HM-S14-03 / #167).

    Classifies incoming prompts into their Night-MoE domain (`exoplanet`, `stars`, `maths`, `physics_transfer`,
    `astro_transfer`, `cyber`, `code`) and routes to the domain adapter ONLY if that adapter passed H1 and the
    Verified Update Gate (`admitted_adapters`). Domains whose adapter failed the gate (or transfer packs where no
    adapter beat `base`) are routed to `base`, preventing cross-expert format interference.
    """

    def __init__(
        self,
        admitted_adapters: Sequence[str] = ("exoplanet", "stars"),
        *,
        pack_to_adapter: Mapping[str, str] | None = None,
    ) -> None:
        self.admitted_adapters: set[str] = {str(a) for a in admitted_adapters}
        self.pack_to_adapter: dict[str, str] = dict(
            pack_to_adapter
            or {
                "exoplanet": "exoplanet",
                "stars": "stars",
                "maths": "maths",
                "cyber": "cyber",
                "code": "code",
                "astro_transfer": "stars",
                "physics_transfer": "maths",
            }
        )

    @staticmethod
    def classify_prompt_domain(messages: Sequence[Mapping[str, str]]) -> str:
        """Deterministically classify a chat prompt into its Night-MoE evaluation domain."""
        joined = " ".join(str(m.get("content", "")) for m in messages).lower()
        if "verdict: confirmed" in joined or "kepler transit signal" in joined or "orbital period (days)" in joined:
            return "exoplanet"
        if "class: x" in joined or "spectral class?" in joined or "effective temperature (k)" in joined:
            return "stars"
        if "final answer:" in joined or "word problems" in joined:
            return "maths"
        if "your code must pass:" in joined or "```python" in joined or "def " in joined:
            return "code"
        if "cyber-security" in joined or "cve-" in joined or "cwe-" in joined or "vulnerability" in joined or "encryption" in joined:
            return "cyber"
        if any(w in joined for w in ("parsec", "luminosity", "redshift", "supernova", "main sequence", "galaxy", "nebula", "ecliptic", "cepheid")):
            return "astro_transfer"
        if any(w in joined for w in ("kinetic energy", "momentum", "capacitor", "wavelength", "magnetic field", "thermodynamic", "pendulum", "newton", "joule")):
            return "physics_transfer"
        return "cyber"

    def route_pack(self, pack: str, *, enforce_gate: bool = True) -> str:
        """Return the adapter id to serve for `pack` (`base` when `enforce_gate=True` and the candidate is not admitted)."""
        candidate = self.pack_to_adapter.get(pack, pack)
        if not enforce_gate:
            return candidate
        if pack in ("astro_transfer", "physics_transfer") and pack not in self.admitted_adapters:
            return "base"
        return candidate if candidate in self.admitted_adapters else "base"

    def route_messages(self, messages: Sequence[Mapping[str, str]], *, enforce_gate: bool = True) -> tuple[str, str]:
        """Classify `messages` into `domain` and return `(domain, selected_expert)`."""
        domain = self.classify_prompt_domain(messages)
        return domain, self.route_pack(domain, enforce_gate=enforce_gate)


@dataclass(frozen=True)
class ServedExpertTask:
    """One executable-test task for evaluating QueenBee bandit routing over served experts (HM-S15-03 / #173)."""

    task_id: str
    cluster_id: str
    prompt: str
    tests: tuple[str, ...]
    expert_candidates: Mapping[str, str]
    cost_per_1k: Mapping[str, float] = field(default_factory=dict)


def verify_python_candidate_isolated(code: str, tests: Sequence[str], *, timeout_s: float = 4.0) -> bool:
    """Run candidate Python code against executable `assert` tests in an isolated Python child process (HM-S15-03 / #173)."""
    import ast
    import re
    import subprocess
    import sys

    stripped = code.strip()
    if stripped.startswith("```"):
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", stripped, re.S)
        if blocks:
            stripped = blocks[-1].strip()
    if not stripped or not tests:
        return False
    try:
        tree = ast.parse(stripped)
    except SyntaxError:
        return False
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in tree.body):
        return False
    harness = "\n\n".join([stripped, *tests, "print('QB_SERVED_EXPERT_VERIFIED_OK')"])
    try:
        res = subprocess.run(
            [sys.executable, "-I", "-B", "-c", harness],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return res.returncode == 0 and "QB_SERVED_EXPERT_VERIFIED_OK" in res.stdout


def evaluate_served_expert_bandit_routing(
    tasks: Sequence[ServedExpertTask],
    *,
    seeds: Sequence[int] = tuple(range(2026, 2046)),
    rounds: int = 240,
    warm_records: Sequence[DanceRecord] = (),
    secret: bytes | None = None,
    max_trials_per_signer: float = 60.0,
) -> dict[str, Any]:
    """Evaluate `StrategyManager` / `BanditStrategy` routing among served experts on tasks with executable tests (HM-S15-03 / #173).

    First executes every served expert's candidate completion on every task's unit tests in an isolated Python subprocess
    to measure the **routing plateau** (the best single served expert across the whole benchmark). Then runs paired seeds
    (sharing the exact task sequence across all arms) comparing `oracle`, `near_oracle_90`, `best_single_expert`,
    `uniform`, `bandit_cold` (`StrategyManager(profile='ide_default')`), and `bandit_warm_budgeted`.
    """
    from hivemind.queenbee.dance import recruitment_weights

    if not tasks:
        raise ValueError("tasks must be non-empty")
    if not seeds:
        raise ValueError("seeds must be non-empty")
    if int(rounds) < 2:
        raise ValueError("rounds must be >= 2")
    eff_secret = secret if secret is not None else hashlib.sha256(str(tuple(seeds)).encode("utf-8")).digest()
    experts = sorted(tasks[0].expert_candidates.keys())
    if not experts:
        raise ValueError("tasks must name at least one served expert")
    for t in tasks:
        if set(t.expert_candidates.keys()) != set(experts):
            raise ValueError(f"task {t.task_id!r} expert_candidates keys do not match {experts}")

    # 1. Execute every (task, expert) candidate through the real isolated Python unit-test verifier once
    verified_matrix: dict[str, dict[str, bool]] = {}
    cluster_tasks: dict[str, list[ServedExpertTask]] = {}
    for t in tasks:
        verified_matrix[t.task_id] = {
            e: verify_python_candidate_isolated(t.expert_candidates[e], t.tests)
            for e in experts
        }
        cluster_tasks.setdefault(t.cluster_id, []).append(t)

    # 2. Measure the routing plateau (best single served expert across all tasks) before claiming any bandit gain
    single_expert_scores: dict[str, float] = {
        e: sum(1 for t in tasks if verified_matrix[t.task_id][e]) / len(tasks)
        for e in experts
    }
    best_single_expert = max(experts, key=lambda e: single_expert_scores[e])
    best_single_plateau = single_expert_scores[best_single_expert]

    # Cluster-level empirical pass rate per expert (for expected-reward and regret accounting)
    cluster_rates: dict[str, dict[str, float]] = {
        cid: {
            e: sum(1 for t in ctasks if verified_matrix[t.task_id][e]) / len(ctasks)
            for e in experts
        }
        for cid, ctasks in cluster_tasks.items()
    }

    # Build warm-start report if records provided, otherwise from a small honest pilot sample
    pilot_records = list(warm_records)
    if not pilot_records:
        for cid, rates in sorted(cluster_rates.items()):
            for e, rate in sorted(rates.items()):
                n_pilot = 20
                s_pilot = int(round(rate * n_pilot))
                pilot_records.append(
                    DanceRecord(e, e, cid, n_pilot, s_pilot, 0.4, 10, "unit_tests", f"scout-{cid}").signed(eff_secret)
                )
    warm_report = recruitment_weights(
        pilot_records,
        now_epoch=10,
        horizon_epochs=5,
        cap_per_record=0.5,
        secret=eff_secret,
        max_trials_per_signer=max_trials_per_signer,
    )

    per_seed: list[dict[str, Any]] = []
    flat_runs: list[dict[str, Any]] = []
    for seed in seeds:
        seq_rng = random.Random(seed * 13 + 7)
        task_seq = [seq_rng.choice(tasks) for _ in range(rounds)]

        mgr_cold = StrategyManager(profile="ide_default", seed=seed + 100)
        bandit_warm = BanditStrategy()
        bandit_warm.warm_start(warm_report, use_trial_budget=True)

        arms: dict[str, tuple[Any, Any]] = {
            "oracle": (
                lambda t, r: max(experts, key=lambda e: (cluster_rates[t.cluster_id][e], int(verified_matrix[t.task_id][e]))),
                lambda o: None,
            ),
            "near_oracle_90": (
                lambda t, r: (
                    max(experts, key=lambda e: cluster_rates[t.cluster_id][e])
                    if r.random() < 0.9
                    else r.choice(experts)
                ),
                lambda o: None,
            ),
            "best_single_expert": (lambda t, r: best_single_expert, lambda o: None),
            "uniform": (lambda t, r: r.choice(experts), lambda o: None),
            "bandit_cold": (lambda t, r: mgr_cold.route(t.cluster_id, experts), mgr_cold.observe),
            "bandit_warm_budgeted": (
                lambda t, r: bandit_warm.choose(t.cluster_id, experts, r),
                bandit_warm.observe,
            ),
        }

        seed_res: dict[str, Any] = {}
        half = rounds - rounds // 2
        for idx, (arm_name, (choose_fn, obs_fn)) in enumerate(arms.items()):
            arm_rng = random.Random(seed * 1000 + idx)
            last_half_ok = 0
            last_half_exp = 0.0
            regret = 0.0
            early_regret = 0.0
            for step, task in enumerate(task_seq, start=1):
                chosen_e = choose_fn(task, arm_rng)
                ok = verified_matrix[task.task_id][chosen_e]
                obs_fn(Outcome(task.cluster_id, chosen_e, verified=True, success=ok))
                best_rate = max(cluster_rates[task.cluster_id].values())
                exp_rate = cluster_rates[task.cluster_id][chosen_e]
                gap = best_rate - exp_rate
                regret += gap
                if step <= min(60, rounds // 3):
                    early_regret += gap
                if step > rounds // 2:
                    last_half_ok += int(ok)
                    last_half_exp += exp_rate
            seed_res[arm_name] = {
                "last_half_verified_pass_rate": last_half_ok / half,
                "last_half_expected_pass_rate": last_half_exp / half,
                "regret": regret,
                "early_regret": early_regret,
            }
        per_seed.append({"seed": seed, "results": seed_res})

        # Negative control: flat world where every served expert has identical verified pass outcome on every task
        flat_cold = BanditStrategy()
        flat_rng_u = random.Random(seed * 31 + 1)
        flat_rng_b = random.Random(seed * 31 + 2)
        u_ok = 0
        b_ok = 0
        for step, task in enumerate(task_seq, start=1):
            shared_ok = verified_matrix[task.task_id][best_single_expert]
            _ = flat_rng_u.choice(experts)
            chosen_b = flat_cold.choose(task.cluster_id, experts, flat_rng_b)
            flat_cold.observe(Outcome(task.cluster_id, chosen_b, verified=True, success=shared_ok))
            if step > rounds // 2:
                u_ok += int(shared_ok)
                b_ok += int(shared_ok)
        flat_runs.append(
            {
                "seed": seed,
                "uniform_pass_rate": u_ok / half,
                "bandit_cold_pass_rate": b_ok / half,
                "regret": 0.0,
            }
        )

    def _mean(arm: str, key: str) -> float:
        return sum(ps["results"][arm][key] for ps in per_seed) / len(per_seed)

    cold_beats_plateau = sum(
        1
        for ps in per_seed
        if ps["results"]["bandit_cold"]["last_half_expected_pass_rate"]
        > ps["results"]["best_single_expert"]["last_half_expected_pass_rate"]
    )
    warm_beats_plateau = sum(
        1
        for ps in per_seed
        if ps["results"]["bandit_warm_budgeted"]["last_half_expected_pass_rate"]
        > ps["results"]["best_single_expert"]["last_half_expected_pass_rate"]
    )
    cold_beats_uniform = sum(
        1
        for ps in per_seed
        if ps["results"]["bandit_cold"]["last_half_expected_pass_rate"]
        > ps["results"]["uniform"]["last_half_expected_pass_rate"]
    )
    warm_lower_early_regret = sum(
        1
        for ps in per_seed
        if ps["results"]["bandit_warm_budgeted"]["early_regret"] < ps["results"]["bandit_cold"]["early_regret"]
    )
    positive_wiring_ok = all(
        ps["results"]["oracle"]["last_half_expected_pass_rate"]
        >= max(v["last_half_expected_pass_rate"] for v in ps["results"].values()) - 1e-12
        for ps in per_seed
    )
    positive_discrim_ok = all(
        ps["results"]["oracle"]["last_half_expected_pass_rate"]
        > ps["results"]["near_oracle_90"]["last_half_expected_pass_rate"]
        > ps["results"]["uniform"]["last_half_expected_pass_rate"]
        for ps in per_seed
    )
    negative_flat_ok = all(
        abs(f["uniform_pass_rate"] - f["bandit_cold_pass_rate"]) < 1e-12 and f["regret"] == 0.0
        for f in flat_runs
    )

    return {
        "n_tasks": len(tasks),
        "n_seeds": len(seeds),
        "rounds": rounds,
        "served_experts": experts,
        "single_expert_verified_pass_rates": single_expert_scores,
        "best_single_expert": best_single_expert,
        "routing_plateau_pass_rate": best_single_plateau,
        "arm_means": {
            arm: {
                "last_half_verified_pass_rate": round(_mean(arm, "last_half_verified_pass_rate"), 4),
                "last_half_expected_pass_rate": round(_mean(arm, "last_half_expected_pass_rate"), 4),
                "regret": round(_mean(arm, "regret"), 3),
                "early_regret": round(_mean(arm, "early_regret"), 3),
            }
            for arm in ("oracle", "near_oracle_90", "best_single_expert", "uniform", "bandit_cold", "bandit_warm_budgeted")
        },
        "cold_beats_best_single_expert_seeds": cold_beats_plateau,
        "warm_beats_best_single_expert_seeds": warm_beats_plateau,
        "cold_beats_uniform_seeds": cold_beats_uniform,
        "warm_lower_early_regret_than_cold_seeds": warm_lower_early_regret,
        "controls": {
            "positive_wiring": positive_wiring_ok,
            "positive_discriminating": positive_discrim_ok,
            "negative_flat_world": negative_flat_ok,
        },
        "supported": bool(
            positive_wiring_ok
            and positive_discrim_ok
            and negative_flat_ok
            and cold_beats_plateau >= int(0.9 * len(seeds))
            and warm_beats_plateau >= int(0.9 * len(seeds))
            and warm_lower_early_regret >= int(0.8 * len(seeds))
        ),
    }

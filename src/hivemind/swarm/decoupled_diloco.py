# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""SI-12: Decoupled DiLoCo syncer primitives (Google DeepMind, `arXiv:2604.21428`).

Decoupled DiLoCo (Douillard, Rush, Donchev, Charles et al., Google DeepMind / Google
Research, April 2026) replaces the lock-step outer all-reduce of DiLoCo by a central,
CPU-resident *syncer* that aggregates learner fragments as soon as a **minimum quorum**
of `K <= M` learners has reported, optionally waits an **adaptive grace window** bounded
by a fraction `gamma < 1` of the measured network slack, weights each learner by a
**token-weighted quality weight**, and merges per-learner outer gradients with
**Radial-Directional Averaging (RDA)** so that the merged norm no longer shrinks like
`R / sqrt(M)` when per-learner outer gradients are nearly orthogonal.

This module implements those four mechanisms as small, individually testable pieces:

* :func:`token_weighted_quality_weight` — `Weight(c_tokens, c_steps) = c_tokens * (c_tokens / c_steps)`
  (Algorithm 2, line 9, "quantity x quality").
* :func:`select_quorum` — first-`K`-by-arrival quorum plus every learner that arrives
  inside the grace window measured from the moment the quorum was reached.
* :class:`AdaptiveGraceWindowController` — `xi_slack = tau * xi_step - (xi_quorum + xi_sync)`
  and `xi_grace = gamma * max(0, xi_slack)` with exponential-moving-average tracking of
  `xi_step` and `xi_quorum` (Eq. 3 of the paper).
* :func:`radial_directional_average` — weighted RDA (Appendix D.2):
  `RDA({(v_i, w_i)}) = (sum_i w_i ||v_i|| / sum_i w_i) * phi(sum_i w_i phi(v_i) / sum_i w_i)`
  with `phi(x) = x / ||x||`.
* :class:`DecoupledDiLoCoQuorumSyncer` — one syncer round for one parameter fragment,
  producing the outer gradient `Delta_p` that the existing outer optimizers
  (`DiLoCoOuterOptimizer`, `CPUOffloadedFlatBucketOuterState`, `CosineGatedAdamDecayOuterOptimizer`)
  consume unchanged.

What this module does **not** do: it does not implement the asynchronous RPC transport,
the Chandy-Lamport checkpointing or the learner recovery protocol of the paper, and it has
not been exercised across real machines. Every number produced here is a controlled
single-process check; see ``scripts/verify_decoupled_diloco_quorum.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch


DECOUPLED_DILOCO_REFERENCE: Dict[str, str] = {
    "name": "Decoupled DiLoCo for Resilient Distributed Pre-training",
    "authors": "Decoupled DiLoCo Team (Google DeepMind & Google Research), incl. A. Douillard, K. Rush, Y. Donchev, Z. Charles",
    "arxiv_url": "https://arxiv.org/abs/2604.21428",
    "integrated_components": (
        "Minimum-quorum aggregation, adaptive grace window bounded by gamma * slack, "
        "token-weighted learner weights, and Radial-Directional Averaging (RDA)"
    ),
}


@dataclass(frozen=True)
class LearnerArrival:
    """Metadata a learner sends to the syncer for one fragment (Algorithm 1, line 13)."""

    learner_id: str
    arrival_time_s: float
    steps_since_sync: int
    tokens_since_sync: int

    def __post_init__(self) -> None:
        if not self.learner_id:
            raise ValueError("learner_id must be non-empty")
        if not math.isfinite(self.arrival_time_s):
            raise ValueError("arrival_time_s must be finite")
        if self.steps_since_sync < 0 or self.tokens_since_sync < 0:
            raise ValueError("steps_since_sync and tokens_since_sync must be >= 0")


@dataclass(frozen=True)
class QuorumSelection:
    """Outcome of one quorum-gathering attempt for a fragment."""

    reached: bool
    min_quorum_k: int
    grace_window_s: float
    admitted_learner_ids: Tuple[str, ...]
    deferred_learner_ids: Tuple[str, ...]
    quorum_reached_at_s: Optional[float]
    window_close_s: Optional[float]


@dataclass(frozen=True)
class SyncerRoundTelemetry:
    """Telemetry for one syncer merge of one fragment."""

    fragment_name: str
    merge_mode: str
    num_arrivals: int
    num_admitted: int
    num_deferred: int
    min_quorum_k: int
    grace_window_s: float
    admitted_learner_ids: Tuple[str, ...]
    learner_weights: Dict[str, float]
    admitted_tokens: int
    deferred_tokens: int
    merged_delta_norm: float
    direct_average_norm: float
    mean_learner_delta_norm: float


def token_weighted_quality_weight(tokens_since_sync: int, steps_since_sync: int) -> float:
    """Decoupled DiLoCo learner weight `Weight(c_tokens, c_steps) = c_tokens * (c_tokens / c_steps)`.

    The paper reads this as `(quantity) x (quality)`: more tokens are worth more, and the
    same tokens are worth more when amortised over fewer inner steps. A learner that took
    no inner step since the last sync has no new information and receives weight `0`.
    """
    if tokens_since_sync < 0 or steps_since_sync < 0:
        raise ValueError("tokens_since_sync and steps_since_sync must be >= 0")
    if steps_since_sync == 0:
        return 0.0
    tokens = float(tokens_since_sync)
    return tokens * (tokens / float(steps_since_sync))


def select_quorum(
    arrivals: Sequence[LearnerArrival],
    min_quorum_k: int,
    grace_window_s: float,
) -> QuorumSelection:
    """Admit the first `K` learners by arrival time plus every learner arriving within the grace window.

    Ties on arrival time are broken by `learner_id` so the selection is deterministic.
    If fewer than `K` learners have arrived the quorum is not reached and nothing is admitted.
    """
    if min_quorum_k < 1:
        raise ValueError("min_quorum_k must be >= 1")
    if grace_window_s < 0.0 or not math.isfinite(grace_window_s):
        raise ValueError("grace_window_s must be finite and >= 0")
    ids = [a.learner_id for a in arrivals]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate learner_id in arrivals")

    ordered = sorted(arrivals, key=lambda a: (a.arrival_time_s, a.learner_id))
    if len(ordered) < min_quorum_k:
        return QuorumSelection(
            reached=False,
            min_quorum_k=min_quorum_k,
            grace_window_s=float(grace_window_s),
            admitted_learner_ids=(),
            deferred_learner_ids=tuple(a.learner_id for a in ordered),
            quorum_reached_at_s=None,
            window_close_s=None,
        )

    quorum_reached_at = ordered[min_quorum_k - 1].arrival_time_s
    window_close = quorum_reached_at + float(grace_window_s)
    admitted: List[str] = []
    deferred: List[str] = []
    for idx, a in enumerate(ordered):
        if idx < min_quorum_k or a.arrival_time_s <= window_close:
            admitted.append(a.learner_id)
        else:
            deferred.append(a.learner_id)
    return QuorumSelection(
        reached=True,
        min_quorum_k=min_quorum_k,
        grace_window_s=float(grace_window_s),
        admitted_learner_ids=tuple(admitted),
        deferred_learner_ids=tuple(deferred),
        quorum_reached_at_s=float(quorum_reached_at),
        window_close_s=float(window_close),
    )


class AdaptiveGraceWindowController:
    """Adaptive grace window `xi_grace = gamma * max(0, xi_slack)` (Decoupled DiLoCo, Eq. 3).

    `xi_slack = tau * xi_step - (xi_quorum + xi_sync)` where `tau` is the number of compute
    steps the syncer overlaps with (the paper double-buffers with `tau = 2`), `xi_step` and
    `xi_quorum` are tracked as exponential moving averages, and `xi_sync` is the measured
    fragment synchronisation time. The safety margin `gamma < 1` guarantees that the grace
    window never consumes the whole slack, so waiting for extra learners cannot stall the
    learners' compute.
    """

    def __init__(
        self,
        overlap_steps_tau: int = 2,
        safety_margin_gamma: float = 0.5,
        ema_beta: float = 0.9,
        sync_duration_s: float = 0.0,
    ) -> None:
        if overlap_steps_tau < 1:
            raise ValueError("overlap_steps_tau must be >= 1")
        if not (0.0 <= safety_margin_gamma < 1.0):
            raise ValueError("safety_margin_gamma must be in [0, 1)")
        if not (0.0 <= ema_beta < 1.0):
            raise ValueError("ema_beta must be in [0, 1)")
        if sync_duration_s < 0.0 or not math.isfinite(sync_duration_s):
            raise ValueError("sync_duration_s must be finite and >= 0")
        self.overlap_steps_tau = int(overlap_steps_tau)
        self.safety_margin_gamma = float(safety_margin_gamma)
        self.ema_beta = float(ema_beta)
        self.sync_duration_s = float(sync_duration_s)
        self.step_duration_ema_s: Optional[float] = None
        self.quorum_duration_ema_s: Optional[float] = None

    @staticmethod
    def _ema(previous: Optional[float], observation: float, beta: float) -> float:
        if observation < 0.0 or not math.isfinite(observation):
            raise ValueError("durations must be finite and >= 0")
        if previous is None:
            return float(observation)
        return beta * previous + (1.0 - beta) * float(observation)

    def observe_step_duration(self, seconds: float) -> float:
        """Record one learner compute-step duration; returns the updated EMA."""
        self.step_duration_ema_s = self._ema(self.step_duration_ema_s, seconds, self.ema_beta)
        return self.step_duration_ema_s

    def observe_quorum_duration(self, seconds: float) -> float:
        """Record how long the syncer needed to reach the minimum quorum; returns the updated EMA."""
        self.quorum_duration_ema_s = self._ema(self.quorum_duration_ema_s, seconds, self.ema_beta)
        return self.quorum_duration_ema_s

    def slack_seconds(self) -> float:
        """`xi_slack = tau * xi_step - (xi_quorum + xi_sync)`; `0` until a step duration is observed."""
        if self.step_duration_ema_s is None:
            return 0.0
        quorum = self.quorum_duration_ema_s if self.quorum_duration_ema_s is not None else 0.0
        return self.overlap_steps_tau * self.step_duration_ema_s - (quorum + self.sync_duration_s)

    def grace_window_seconds(self) -> float:
        """`xi_grace = gamma * max(0, xi_slack)` — strictly below the slack whenever slack is positive."""
        return self.safety_margin_gamma * max(0.0, self.slack_seconds())

    def telemetry(self) -> Dict[str, float]:
        return {
            "overlap_steps_tau": float(self.overlap_steps_tau),
            "safety_margin_gamma": self.safety_margin_gamma,
            "step_duration_ema_s": float(self.step_duration_ema_s or 0.0),
            "quorum_duration_ema_s": float(self.quorum_duration_ema_s or 0.0),
            "sync_duration_s": self.sync_duration_s,
            "slack_s": float(self.slack_seconds()),
            "grace_window_s": float(self.grace_window_seconds()),
        }


def _validate_vectors_and_weights(
    vectors: Sequence[torch.Tensor],
    weights: Optional[Sequence[float]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(vectors) == 0:
        raise ValueError("at least one vector is required")
    shape = vectors[0].shape
    for v in vectors:
        if v.shape != shape:
            raise ValueError("all vectors must share the same shape")
    stacked = torch.stack([v.detach().to(torch.float32).reshape(-1) for v in vectors], dim=0)
    if weights is None:
        w = torch.ones(stacked.shape[0], dtype=torch.float32)
    else:
        if len(weights) != stacked.shape[0]:
            raise ValueError("weights must have one entry per vector")
        w = torch.tensor([float(x) for x in weights], dtype=torch.float32)
        if bool((w < 0).any()) or not bool(torch.isfinite(w).all()):
            raise ValueError("weights must be finite and >= 0")
    if float(w.sum()) <= 0.0:
        raise ValueError("weights must not all be zero")
    return stacked, w / w.sum()


def direct_weighted_average(
    vectors: Sequence[torch.Tensor],
    weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Plain (weighted) mean of outer gradients, the DiLoCo baseline merge (Appendix D.2)."""
    stacked, w = _validate_vectors_and_weights(vectors, weights)
    return (w.unsqueeze(1) * stacked).sum(dim=0).reshape(vectors[0].shape)


def radial_directional_average(
    vectors: Sequence[torch.Tensor],
    weights: Optional[Sequence[float]] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Weighted Radial-Directional Averaging (RDA, Decoupled DiLoCo Appendix D.2).

    Averages norms and unit directions separately and multiplies them back, so that if every
    `||v_i|| = R` the result has norm exactly `R` regardless of how many nearly-orthogonal
    learners contributed. If the weighted mean direction vanishes (perfectly cancelling
    directions) the zero vector is returned; the all-zero input also yields zero.
    """
    stacked, w = _validate_vectors_and_weights(vectors, weights)
    norms = stacked.norm(dim=1)
    units = stacked / norms.clamp_min(eps).unsqueeze(1)
    units = torch.where(norms.unsqueeze(1) > eps, units, torch.zeros_like(units))
    radius = float((w * norms).sum())
    mean_dir = (w.unsqueeze(1) * units).sum(dim=0)
    dir_norm = float(mean_dir.norm())
    if dir_norm <= eps or radius <= eps:
        return torch.zeros_like(mean_dir).reshape(vectors[0].shape)
    return (radius * mean_dir / dir_norm).reshape(vectors[0].shape)


class DecoupledDiLoCoQuorumSyncer:
    """One Decoupled DiLoCo syncer shard: quorum, grace window, token weights and RDA merge."""

    MERGE_MODES = ("rda", "direct")

    def __init__(
        self,
        num_learners_m: int,
        min_quorum_k: int,
        merge_mode: str = "rda",
        grace_controller: Optional[AdaptiveGraceWindowController] = None,
    ) -> None:
        if num_learners_m < 1:
            raise ValueError("num_learners_m must be >= 1")
        if not (1 <= min_quorum_k <= num_learners_m):
            raise ValueError("min_quorum_k must satisfy 1 <= K <= M")
        if merge_mode not in self.MERGE_MODES:
            raise ValueError(f"merge_mode must be one of {self.MERGE_MODES}")
        self.num_learners_m = int(num_learners_m)
        self.min_quorum_k = int(min_quorum_k)
        self.merge_mode = merge_mode
        self.grace_controller = grace_controller

    @torch.no_grad()
    def merge_fragment(
        self,
        fragment_name: str,
        global_prev: torch.Tensor,
        learner_fragments: Dict[str, torch.Tensor],
        arrivals: Sequence[LearnerArrival],
        grace_window_s: Optional[float] = None,
    ) -> Tuple[Optional[torch.Tensor], QuorumSelection, SyncerRoundTelemetry]:
        """Merge one fragment. Returns `(Delta_p, selection, telemetry)`; `Delta_p` is `None`
        when the quorum is not reached (the syncer must keep waiting).

        `Delta_p = Merge({theta_m, w_m}, Theta_prev)` aggregates per-learner outer gradients
        `Delta_m = Theta_prev - theta_m` with weights `w_m = Weight(c_tokens, c_steps)`.
        """
        if len(arrivals) > self.num_learners_m:
            raise ValueError("more arrivals than learners in the swarm")
        if grace_window_s is None:
            grace_window_s = (
                self.grace_controller.grace_window_seconds() if self.grace_controller is not None else 0.0
            )
        selection = select_quorum(arrivals, self.min_quorum_k, grace_window_s)
        by_id = {a.learner_id: a for a in arrivals}
        deferred_tokens = sum(by_id[i].tokens_since_sync for i in selection.deferred_learner_ids)

        if not selection.reached:
            telemetry = SyncerRoundTelemetry(
                fragment_name=fragment_name,
                merge_mode=self.merge_mode,
                num_arrivals=len(arrivals),
                num_admitted=0,
                num_deferred=len(selection.deferred_learner_ids),
                min_quorum_k=self.min_quorum_k,
                grace_window_s=float(grace_window_s),
                admitted_learner_ids=(),
                learner_weights={},
                admitted_tokens=0,
                deferred_tokens=int(deferred_tokens),
                merged_delta_norm=0.0,
                direct_average_norm=0.0,
                mean_learner_delta_norm=0.0,
            )
            return None, selection, telemetry

        prev32 = global_prev.detach().to(torch.float32)
        deltas: List[torch.Tensor] = []
        weights: List[float] = []
        learner_weights: Dict[str, float] = {}
        admitted_tokens = 0
        for learner_id in selection.admitted_learner_ids:
            if learner_id not in learner_fragments:
                raise KeyError(f"admitted learner {learner_id!r} has no fragment payload")
            theta_m = learner_fragments[learner_id].detach().to(torch.float32)
            if theta_m.shape != prev32.shape:
                raise ValueError(f"fragment shape mismatch for learner {learner_id!r}")
            a = by_id[learner_id]
            w_m = token_weighted_quality_weight(a.tokens_since_sync, a.steps_since_sync)
            deltas.append(prev32 - theta_m)
            weights.append(w_m)
            learner_weights[learner_id] = w_m
            admitted_tokens += a.tokens_since_sync
        if sum(weights) <= 0.0:
            raise ValueError("no admitted learner performed any inner step since the last sync")

        direct = direct_weighted_average(deltas, weights)
        merged = radial_directional_average(deltas, weights) if self.merge_mode == "rda" else direct
        mean_delta_norm = float(torch.stack([d.norm() for d in deltas]).mean())

        telemetry = SyncerRoundTelemetry(
            fragment_name=fragment_name,
            merge_mode=self.merge_mode,
            num_arrivals=len(arrivals),
            num_admitted=len(selection.admitted_learner_ids),
            num_deferred=len(selection.deferred_learner_ids),
            min_quorum_k=self.min_quorum_k,
            grace_window_s=float(grace_window_s),
            admitted_learner_ids=selection.admitted_learner_ids,
            learner_weights=learner_weights,
            admitted_tokens=int(admitted_tokens),
            deferred_tokens=int(deferred_tokens),
            merged_delta_norm=float(merged.norm()),
            direct_average_norm=float(direct.norm()),
            mean_learner_delta_norm=mean_delta_norm,
        )
        return merged.to(dtype=global_prev.dtype, device=global_prev.device), selection, telemetry


class ContinuousAsyncLearnerCoordinator:
    """Season 9 (`HM-S9-13`): Continuous Learner Loop with Eager vs Deferred-Token-Weighted Straggler Rules.

    Fixes the round-synchronous E8 pilot defect where a slow WAN learner that missed the quorum
    window had its work discarded (`"drop"`). Supports three straggler policies:
      - `"drop"`: round-synchronous baseline (discards late arrivals from earlier rounds).
      - `"deferred_token_weighted"`: Decoupled DiLoCo Algorithm 1 (`arXiv:2604.21428`) — late learners
        carry their accumulated delta and token count into the next quorum merge with token weight
        `w_m = tokens_since_sync`.
      - `"eager_one_step"`: Eager 1-outer-step stale update (Kale et al., `arXiv:2502.12996`) — late
        arrivals apply an immediate staleness-damped outer step (`staleness_decay ** lag`).
    """

    STRAGGLER_MODES = ("drop", "deferred_token_weighted", "eager_one_step")

    def __init__(
        self,
        num_learners_m: int,
        min_quorum_k: int,
        straggler_mode: str = "deferred_token_weighted",
        merge_mode: str = "rda",
        staleness_decay: float = 0.85,
    ) -> None:
        if num_learners_m < 1:
            raise ValueError("num_learners_m must be >= 1")
        if not (1 <= min_quorum_k <= num_learners_m):
            raise ValueError("min_quorum_k must satisfy 1 <= K <= M")
        if straggler_mode not in self.STRAGGLER_MODES:
            raise ValueError(f"straggler_mode must be one of {self.STRAGGLER_MODES}")
        if merge_mode not in DecoupledDiLoCoQuorumSyncer.MERGE_MODES:
            raise ValueError(f"merge_mode must be one of {DecoupledDiLoCoQuorumSyncer.MERGE_MODES}")
        if not (0.0 < staleness_decay <= 1.0):
            raise ValueError("staleness_decay must be in (0, 1]")

        self.num_learners_m = int(num_learners_m)
        self.min_quorum_k = int(min_quorum_k)
        self.straggler_mode = straggler_mode
        self.merge_mode = merge_mode
        self.staleness_decay = float(staleness_decay)

        self.current_round: int = 0
        self._fresh_inbox: Dict[str, Dict[str, object]] = {}
        self._deferred_carry: Dict[str, Dict[str, object]] = {}
        self._eager_queue: List[Dict[str, object]] = []

        self.total_submitted_tokens: int = 0
        self.merged_tokens: int = 0
        self.dropped_tokens: int = 0
        self.eager_applied_steps: int = 0

    def submit_learner_delta(
        self,
        learner_id: str,
        origin_round: int,
        delta: torch.Tensor,
        tokens_since_sync: int,
        steps_since_sync: int,
        arrival_time_s: float = 0.0,
    ) -> Dict[str, object]:
        """Submit a learner's outer delta `Delta_m = Theta_origin - theta_m`.

        Future rounds and a second pending fresh update from the same learner
        are rejected before changing queued work or token accounting.
        """
        if tokens_since_sync < 0 or steps_since_sync < 0:
            raise ValueError("tokens_since_sync and steps_since_sync must be >= 0")
        lid = str(learner_id)
        if int(origin_round) > self.current_round:
            raise ValueError("origin_round must not be in the future")
        if int(origin_round) == self.current_round and lid in self._fresh_inbox:
            raise ValueError("learner already has a pending update for the current round")
        d32 = delta.detach().to(torch.float32).clone()
        self.total_submitted_tokens += int(tokens_since_sync)

        if int(origin_round) == self.current_round:
            if lid in self._deferred_carry:
                carry = self._deferred_carry.pop(lid)
                d32 = d32 + carry["delta"]  # type: ignore[operator]
                tokens_since_sync += int(carry["tokens"])  # type: ignore[arg-type]
                steps_since_sync += int(carry["steps"])  # type: ignore[arg-type]
            self._fresh_inbox[lid] = {
                "delta": d32,
                "tokens": int(tokens_since_sync),
                "steps": int(steps_since_sync),
                "arrival_time_s": float(arrival_time_s),
                "origin_round": int(origin_round),
            }
            return {"accepted": True, "disposition": "fresh_inbox", "round": self.current_round}

        # Straggler arrival from an earlier round (origin_round < self.current_round)
        lag = max(1, self.current_round - int(origin_round))
        if self.straggler_mode == "drop":
            self.dropped_tokens += int(tokens_since_sync)
            return {
                "accepted": False,
                "disposition": "dropped_stale",
                "lag_rounds": lag,
                "round": self.current_round,
            }

        if self.straggler_mode == "deferred_token_weighted":
            prev = self._deferred_carry.get(lid)
            if prev is not None:
                d32 = prev["delta"] + d32  # type: ignore[operator]
                tokens_since_sync += int(prev["tokens"])  # type: ignore[arg-type]
                steps_since_sync += int(prev["steps"])  # type: ignore[arg-type]
            self._deferred_carry[lid] = {
                "delta": d32,
                "tokens": int(tokens_since_sync),
                "steps": int(steps_since_sync),
                "arrival_time_s": float(arrival_time_s),
                "origin_round": int(origin_round),
            }
            return {
                "accepted": True,
                "disposition": "deferred_carry_with_token_weight",
                "lag_rounds": lag,
                "round": self.current_round,
            }

        # straggler_mode == "eager_one_step"
        self._eager_queue.append(
            {
                "learner_id": lid,
                "delta": d32,
                "tokens": int(tokens_since_sync),
                "steps": int(steps_since_sync),
                "lag_rounds": lag,
            }
        )
        return {
            "accepted": True,
            "disposition": "eager_one_step_queued",
            "lag_rounds": lag,
            "round": self.current_round,
        }

    @torch.no_grad()
    def close_quorum_round(
        self,
        global_param: torch.Tensor,
        outer_lr: float = 0.7,
        grace_window_s: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Attempt a quorum merge, retaining pending work until at least K learners arrive.

        Eager stale steps apply independently. While waiting, ``round_closed`` is ``None``
        and ``quorum_reached`` is false; the current quorum round does not advance.
        """
        p32 = global_param.detach().to(torch.float32).clone()

        # 1. If eager_one_step items are queued, apply them immediately with staleness damping
        eager_ids: List[str] = []
        for item in self._eager_queue:
            lag = int(item["lag_rounds"])  # type: ignore[arg-type]
            damp = self.staleness_decay ** lag
            d_vec = item["delta"]  # type: ignore[assignment]
            p32 = p32 - float(outer_lr) * damp * d_vec  # type: ignore[operator]
            self.merged_tokens += int(item["tokens"])  # type: ignore[arg-type]
            self.eager_applied_steps += 1
            eager_ids.append(str(item["learner_id"]))
        self._eager_queue.clear()

        # 2. Combine fresh inbox + deferred carry learners for the quorum selection
        combined: Dict[str, Dict[str, object]] = dict(self._deferred_carry)
        for lid, entry in self._fresh_inbox.items():
            if lid in combined:
                prev = combined[lid]
                combined[lid] = {
                    "delta": prev["delta"] + entry["delta"],  # type: ignore[operator]
                    "tokens": int(prev["tokens"]) + int(entry["tokens"]),  # type: ignore[arg-type]
                    "steps": int(prev["steps"]) + int(entry["steps"]),  # type: ignore[arg-type]
                    "arrival_time_s": float(entry["arrival_time_s"]),  # type: ignore[arg-type]
                }
            else:
                combined[lid] = entry

        arrivals = [
            LearnerArrival(
                learner_id=lid,
                arrival_time_s=float(v["arrival_time_s"]),  # type: ignore[arg-type]
                steps_since_sync=int(v["steps"]),  # type: ignore[arg-type]
                tokens_since_sync=int(v["tokens"]),  # type: ignore[arg-type]
            )
            for lid, v in combined.items()
        ]
        selection = select_quorum(
            arrivals, min_quorum_k=self.min_quorum_k, grace_window_s=grace_window_s
        )

        admitted_deltas: List[torch.Tensor] = []
        admitted_weights: List[float] = []
        for lid in selection.admitted_learner_ids:
            v = combined[lid]
            admitted_deltas.append(v["delta"])  # type: ignore[arg-type]
            w_m = token_weighted_quality_weight(int(v["tokens"]), int(v["steps"]))  # type: ignore[arg-type]
            admitted_weights.append(w_m)
            self.merged_tokens += int(v["tokens"])  # type: ignore[arg-type]

        if admitted_deltas:
            merged_delta = (
                radial_directional_average(admitted_deltas, admitted_weights)
                if self.merge_mode == "rda"
                else direct_weighted_average(admitted_deltas, admitted_weights)
            )
            p32 = p32 - float(outer_lr) * merged_delta

        round_closed: Optional[int] = None
        if selection.reached:
            # Only a completed quorum can defer/drop late learners and consume the inbox.
            new_carry: Dict[str, Dict[str, object]] = {}
            for lid in selection.deferred_learner_ids:
                if self.straggler_mode == "drop":
                    self.dropped_tokens += int(combined[lid]["tokens"])  # type: ignore[arg-type]
                else:
                    new_carry[lid] = combined[lid]
            self._deferred_carry = new_carry
            self._fresh_inbox.clear()
            round_closed = self.current_round
            self.current_round += 1

        pending_tokens = sum(int(v["tokens"]) for v in self._deferred_carry.values())  # type: ignore[arg-type]
        pending_tokens += sum(int(v["tokens"]) for v in self._fresh_inbox.values())  # type: ignore[arg-type]
        conservation = self.total_submitted_tokens == (
            self.merged_tokens + pending_tokens + self.dropped_tokens
        )
        return p32.to(dtype=global_param.dtype, device=global_param.device), {
            "round_closed": round_closed,
            "quorum_reached": selection.reached,
            "straggler_mode": self.straggler_mode,
            "admitted_learners": list(selection.admitted_learner_ids),
            "deferred_learners": list(selection.deferred_learner_ids),
            "eager_applied_learners": eager_ids,
            "total_submitted_tokens": self.total_submitted_tokens,
            "merged_tokens": self.merged_tokens,
            "pending_tokens": pending_tokens,
            "dropped_tokens": self.dropped_tokens,
            "token_conservation_holds": bool(conservation),
        }


@torch.no_grad()
def analyze_campaign_rda_and_delta_cosine(
    base_lora_state: Dict[str, torch.Tensor],
    branch_lora_states: Dict[str, Dict[str, torch.Tensor]],
    weights: Optional[Sequence[float]] = None,
    low_cosine_threshold: float = 0.30,
) -> Dict[str, object]:
    """Season 9 (`HM-S9-14`): E5-het Pairwise Delta Cosine & RDA Multiplier Analysis on Campaign Deltas.

    Computes for M domain learners:
      1. Flat outer pseudo-gradient `Delta_m = Theta_base - Theta_m` per domain.
      2. Pairwise cosine similarity `cos(Delta_i, Delta_j)` across all `M * (M - 1) / 2` pairs.
      3. Empirical RDA-to-direct norm multiplier `r_t = ||Delta_RDA||_2 / ||Delta_direct||_2`.
      4. Theoretical common-norm formula prediction from the Proposition in `RDAMultiplier.lean`:
         `r_theory(c_bar, M) = 1 / sqrt(1/M + ((M - 1)/M) * c_bar)`.
      5. Whether the campaign sits in the pre-registered P2 low-cosine regime (`mean_cosine <= 0.30`).
    """
    domain_names = list(branch_lora_states.keys())
    m = len(domain_names)
    if m < 2:
        raise ValueError("at least 2 learner branches are required for pairwise cosine analysis")

    keys = sorted(base_lora_state.keys())
    flat_deltas: List[torch.Tensor] = []
    for dom in domain_names:
        b_state = branch_lora_states[dom]
        parts = [
            (base_lora_state[k].detach().to(torch.float32).cpu() - b_state[k].detach().to(torch.float32).cpu()).reshape(-1)
            for k in keys
        ]
        flat_deltas.append(torch.cat(parts, dim=0))

    stacked = torch.stack(flat_deltas, dim=0)  # [M, D]
    norms = stacked.norm(dim=1).clamp_min(1e-12)
    units = stacked / norms.unsqueeze(1)
    cos_mat = (units @ units.t()).clamp(-1.0, 1.0)

    pairwise_cosines: List[float] = []
    for i in range(m):
        for j in range(i + 1, m):
            pairwise_cosines.append(float(cos_mat[i, j].item()))

    mean_cos = sum(pairwise_cosines) / len(pairwise_cosines)
    min_cos = min(pairwise_cosines)
    max_cos = max(pairwise_cosines)

    rda_vec = radial_directional_average(flat_deltas, weights=weights)
    direct_vec = direct_weighted_average(flat_deltas, weights=weights)
    rda_norm = float(rda_vec.norm().item())
    direct_norm = float(direct_vec.norm().item())
    rt_empirical = rda_norm / max(1e-12, direct_norm)

    denom_sq = max(1e-9, (1.0 / float(m)) + ((float(m - 1) / float(m)) * mean_cos))
    rt_theoretical = 1.0 / math.sqrt(denom_sq)

    return {
        "strategy_id": "HM-S9-14-CAMPAIGN-DELTA-COSINE-RDA-ANALYSIS",
        "num_learners_m": m,
        "num_pairs": len(pairwise_cosines),
        "mean_pairwise_delta_cosine": round(mean_cos, 6),
        "min_pairwise_delta_cosine": round(min_cos, 6),
        "max_pairwise_delta_cosine": round(max_cos, 6),
        "rda_merged_norm": round(rda_norm, 6),
        "direct_merged_norm": round(direct_norm, 6),
        "rda_to_direct_norm_ratio_rt": round(rt_empirical, 6),
        "theoretical_common_norm_ratio": round(rt_theoretical, 6),
        "low_cosine_threshold": float(low_cosine_threshold),
        "low_cosine_regime_active": bool(mean_cos <= low_cosine_threshold),
    }


@torch.no_grad()
def compute_cosine_damped_outer_hyperparams(
    deltas: Sequence[torch.Tensor],
    base_lr: float = 0.7,
    base_momentum: float = 0.9,
    fedavg_target_lr: float = 1.0,
) -> Dict[str, float | str]:
    """Compute alignment-adaptive outer optimizer hyperparameters (Issue #308 / E18b).

    Resolves the IID vs non-IID outer-momentum crossover identified in E18 (#308):
    - When peer deltas are nearly orthogonal (`mean_cosine <= 0`, non-IID specialist branches),
      retains full Nesterov momentum (`effective_momentum = base_momentum = 0.9`) and `base_lr = 0.7`.
    - As peer deltas align (`mean_cosine -> 1.0`, IID replicas), damps outer momentum by
      `(1 - alpha)` where `alpha = clamp(mean_cosine, 0, 1)` and interpolates `effective_lr -> 1.0`,
      recovering exact FedAvg at `mean_cosine = 1.0` and preventing velocity carryover overshoot.
    """
    if len(deltas) < 2:
        raise ValueError("at least 2 peer deltas are required to compute pairwise cosine alignment")
    if base_lr <= 0.0 or fedavg_target_lr <= 0.0:
        raise ValueError("learning rates must be positive")
    if not (0.0 <= base_momentum < 1.0):
        raise ValueError("base_momentum must be in [0, 1)")

    stacked, _ = _validate_vectors_and_weights(deltas, None)
    m = stacked.shape[0]
    norms = stacked.norm(dim=1).clamp_min(1e-12)
    units = stacked / norms.unsqueeze(1)
    cos_mat = (units @ units.t()).clamp(-1.0, 1.0)

    pairwise: List[float] = []
    for i in range(m):
        for j in range(i + 1, m):
            pairwise.append(float(cos_mat[i, j].item()))

    mean_cos = sum(pairwise) / len(pairwise)
    alpha = max(0.0, min(1.0, mean_cos))
    effective_momentum = base_momentum * (1.0 - alpha)
    effective_lr = base_lr + (fedavg_target_lr - base_lr) * alpha
    regime = "iid_fedavg_damped" if alpha >= 0.5 else "non_iid_nesterov"

    return {
        "mean_pairwise_cosine": round(mean_cos, 6),
        "alignment_alpha": round(alpha, 6),
        "effective_momentum": round(effective_momentum, 6),
        "effective_lr": round(effective_lr, 6),
        "regime": regime,
    }

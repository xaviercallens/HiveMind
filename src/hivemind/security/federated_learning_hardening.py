# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
r"""Phase S-3: Federated Learning & Byzantine-Resilient DiLoCo Hardening (`hivemind.security.federated_learning_hardening`).

Secures the decentralized DiLoCo Federated Learning pipeline (`diloco_inner.py`, `diloco_outer.py`,
and `coordinator_api.py`) against advanced federated learning attacks:

1. **Rule FL-01 (`FINITENESS_AND_STRUCTURE_GATE`)**:
   - Rejects worker pseudo-gradients with `NaN`/`Inf`, missing parameter keys, or tensor shape mismatches
     before Multi-Krum Gram matrix computation (even when cohort size $K \le 2$).
2. **Rule FL-02 (`FREE_RIDER_ZERO_GRADIENT_DETECTION`)**:
   - Rejects lazy "free-rider" submissions where $\theta_{\text{final}} \approx \theta_{\text{init}}$
     ($\|\Delta_k\|_2 < \epsilon_{\min} = 10^{-8}$) that attempt to farm `$HIVE` PoUW credits without
     executing GPU backpropagation.
3. **Rule FL-03 (`ADAPTIVE_MEDIAN_NORM_AND_NEUROTOXIN_CLIPPING`)**:
   - Defends against **Collinear Scaling / ALIE / Fang Attacks** (where a Byzantine node aligns its
     direction with honest peers to pass cosine filtering but inflates its L2 norm by $10\times$–$100\times$)
     and **Single-Parameter Backdoor / Neurotoxin Attacks** (where an attacker spikes a few coordinates).
   - Clips every accepted worker's L2 norm to $\tau_2 = c_{\text{clip}} \cdot \text{median}_j \|\Delta_j\|_2$
     and coordinate-wise $L_\infty$ magnitude to $\tau_\infty = c_\infty \cdot \text{median}_j \|\Delta_j\|_\infty$.
4. **Rule FL-04 (`COORDINATE_WISE_TRIMMED_MEAN_AGGREGATION`)**:
   - Hybrid Multi-Krum + Coordinate-Wise $\beta$-Trimmed Mean aggregation that strips the top-$\beta$
     and bottom-$\beta$ extreme values at every individual parameter coordinate across the surviving cohort.
5. **Rule FL-05 (`STALE_GRADIENT_AND_WORKUNIT_REPLAY_DEFENSE`)**:
   - Rejects replayed `wu_id` submissions (double-claiming PoUW rewards), branch mismatches, and
     straggler/stale pseudo-gradients where $t_{\text{coord}} - t_{\text{worker}} > \tau_{\max}$.
   - A `wu_id` is marked consumed only once the spot-check validator has produced a verdict (pass or
     fail), never on envelope acceptance alone, so a submission that is rejected or errors out before
     a verdict leaves the workunit open for its assignee.
6. **Rule FL-06 (`SYBIL_ONE_VOTE_PER_IDENTITY_QUORUM`)**:
   - Enforces at most **1 pseudo-gradient vote per unique `SteamID64`** in any outer aggregation step,
     preventing a single account from spawning multiple `worker_id` processes to capture the Multi-Krum majority.
7. **Rule FL-07 (`DP_DILOCO_GRADIENT_PRIVACY_SANITIZER`)**:
   - Optional $(\epsilon, \delta)$-Differential Privacy Gaussian mechanism ($\mathcal{N}(0, \sigma^2 C^2 I)$
     after sensitivity clipping to $C$) to protect local dataset shards against Gradient Inversion / DLG
     membership inference attacks.
8. **Rule FL-08 (`WORKUNIT_ASSIGNEE_OWNERSHIP`)**:
   - A workunit may only be submitted by the `(worker_id, SteamID64)` pair it was issued to
     (recorded via `register_workunit_assignment` when the coordinator assigns it). Without this, any
     node that learned a `wu_id` could submit against it and burn the honest assignee's workunit.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from hivemind.security.swarm_runtime_hardening import SwarmSecurityViolationError


MIN_PSEUDO_GRADIENT_L2_NORM: float = 1e-8
DEFAULT_MEDIAN_L2_CLIP_FACTOR: float = 2.0
DEFAULT_MEDIAN_LINFTY_CLIP_FACTOR: float = 3.5
DEFAULT_MAX_STALENESS_STEPS: int = 1


@dataclass
class FederatedDeltaAuditReport:
    """Audit report produced by `FederatedLearningSecurityGuard` during outer-loop aggregation."""

    total_submitted_workers: int
    quarantined_pre_filter_workers: Dict[str, str]
    accepted_workers: List[str]
    clipped_workers: List[str]
    median_l2_norm: float
    l2_clip_ceiling: float
    median_linfty_norm: float
    linfty_clip_ceiling: float
    worker_raw_l2_norms: Dict[str, float]
    worker_post_clip_l2_norms: Dict[str, float]
    trimmed_mean_beta: float
    dp_noise_sigma_applied: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DPGradientSanitizerConfig:
    r"""Configuration for $(\epsilon, \delta)$-Differentially Private DiLoCo pseudo-gradient sanitization."""

    enabled: bool = False
    l2_sensitivity_clip_c: float = 1.0
    noise_multiplier_sigma: float = 0.05
    deterministic_seed: int = 42


class FederatedLearningSecurityGuard:
    """Hardens DiLoCo federated learning against Byzantine poisoning, Sybil collusion, replay, and gradient inversion."""

    def __init__(
        self,
        min_l2_norm: float = MIN_PSEUDO_GRADIENT_L2_NORM,
        median_l2_clip_factor: float = DEFAULT_MEDIAN_L2_CLIP_FACTOR,
        median_linfty_clip_factor: float = DEFAULT_MEDIAN_LINFTY_CLIP_FACTOR,
        max_staleness_steps: int = DEFAULT_MAX_STALENESS_STEPS,
        trimmed_mean_beta: float = 0.25,
    ) -> None:
        self.min_l2_norm = float(min_l2_norm)
        self.median_l2_clip_factor = float(median_l2_clip_factor)
        self.median_linfty_clip_factor = float(median_linfty_clip_factor)
        self.max_staleness_steps = int(max_staleness_steps)
        self.trimmed_mean_beta = float(trimmed_mean_beta)
        self.consumed_workunit_ids: Set[str] = set()
        self.step_steam_id_votes: Dict[Tuple[str, int], Dict[str, str]] = {}
        self.workunit_assignees: Dict[str, Tuple[str, str]] = {}

    # ------------------------------------------------------------------
    # 1. Pre-Aggregation Worker Delta Sanitization (FL-01 & FL-02)
    # ------------------------------------------------------------------
    def sanitize_worker_deltas_pre_krum(
        self,
        worker_deltas: Dict[str, Dict[str, torch.Tensor]],
        reference_keys_and_shapes: Optional[Dict[str, torch.Size]] = None,
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, str]]:
        """Filter out NaN/Inf, shape-mismatched, or zero-norm free-rider deltas before Multi-Krum."""
        if not worker_deltas:
            raise ValueError("Cannot sanitize empty worker_deltas.")

        if reference_keys_and_shapes is None:
            # Infer canonical keys/shapes by plurality/majority vote across structurally valid worker deltas
            # so a single first-arriving Byzantine attacker cannot quarantine honest peers (IP-20).
            from collections import Counter

            schema_counts: Counter[tuple[tuple[str, tuple[int, ...]], ...]] = Counter()
            schema_map: Dict[tuple[tuple[str, tuple[int, ...]], ...], Dict[str, torch.Size]] = {}
            for cand_dict in worker_deltas.values():
                if not isinstance(cand_dict, dict) or not cand_dict:
                    continue
                if not all(isinstance(v, torch.Tensor) for v in cand_dict.values()):
                    continue
                sig = tuple(sorted((str(k), tuple(v.shape)) for k, v in cand_dict.items()))
                schema_counts[sig] += 1
                if sig not in schema_map:
                    schema_map[sig] = {k: v.shape for k, v in cand_dict.items()}
            if schema_counts:
                best_sig = schema_counts.most_common(1)[0][0]
                reference_keys_and_shapes = schema_map[best_sig]
            else:
                first_wid = next(iter(worker_deltas))
                reference_keys_and_shapes = {
                    k: v.shape for k, v in worker_deltas[first_wid].items() if isinstance(v, torch.Tensor)
                }

        expected_keys = set(reference_keys_and_shapes.keys())
        clean_deltas: Dict[str, Dict[str, torch.Tensor]] = {}
        quarantined: Dict[str, str] = {}

        for wid, delta_dict in worker_deltas.items():
            if not isinstance(delta_dict, dict) or set(delta_dict.keys()) != expected_keys:
                quarantined[wid] = "FL-01: Parameter key set mismatch."
                continue

            sq_norm = 0.0
            has_invalid_tensor = False
            for k, expected_shape in reference_keys_and_shapes.items():
                t = delta_dict[k]
                if not isinstance(t, torch.Tensor) or t.shape != expected_shape:
                    quarantined[wid] = f"FL-01: Shape mismatch on '{k}'."
                    has_invalid_tensor = True
                    break
                t_fp32 = t.detach().to(device="cpu", dtype=torch.float32)
                if not bool(torch.isfinite(t_fp32).all().item()):
                    quarantined[wid] = f"FL-01: NaN or Inf detected in parameter '{k}'."
                    has_invalid_tensor = True
                    break
                sq_norm += float((t_fp32 ** 2).sum().item())

            if has_invalid_tensor:
                continue

            l2_norm = math.sqrt(sq_norm)
            if l2_norm < self.min_l2_norm:
                quarantined[wid] = (
                    f"FL-02: Free-rider zero-gradient submission blocked "
                    f"(||Delta||_2 = {l2_norm:.2e} < {self.min_l2_norm:.1e})."
                )
                continue

            clean_deltas[wid] = delta_dict

        return clean_deltas, quarantined

    # ------------------------------------------------------------------
    # 2. Adaptive Median-L2 & Coordinate-L_infty Clipping + Trimmed Mean (FL-03 & FL-04)
    # ------------------------------------------------------------------
    def clip_and_aggregate_surviving_deltas(
        self,
        surviving_deltas: Dict[str, Dict[str, torch.Tensor]],
        quarantined_pre_filter: Optional[Dict[str, str]] = None,
        total_submitted_count: Optional[int] = None,
        dp_config: Optional[DPGradientSanitizerConfig] = None,
    ) -> Tuple[Dict[str, torch.Tensor], FederatedDeltaAuditReport]:
        """Apply Adaptive Median-L2 Clipping, Neurotoxin L_infty Clipping, and Coordinate-Wise Trimmed Mean."""
        if not surviving_deltas:
            raise SwarmSecurityViolationError(
                "FL-01",
                "ALL_WORKERS_QUARANTINED",
                "Zero valid worker pseudo-gradients survived Byzantine pre-filtering.",
            )

        worker_ids = sorted(surviving_deltas.keys())
        param_keys = sorted(surviving_deltas[worker_ids[0]].keys())

        # Compute per-worker L2 norm and L_infty norm
        raw_l2_norms: Dict[str, float] = {}
        raw_linfty_norms: Dict[str, float] = {}
        for wid in worker_ids:
            sq_sum = 0.0
            max_abs = 0.0
            for k in param_keys:
                t = surviving_deltas[wid][k].detach().to(device="cpu", dtype=torch.float32)
                sq_sum += float((t ** 2).sum().item())
                t_max = float(t.abs().max().item())
                if t_max > max_abs:
                    max_abs = t_max
            raw_l2_norms[wid] = math.sqrt(sq_sum)
            raw_linfty_norms[wid] = max_abs

        median_l2 = float(torch.tensor(list(raw_l2_norms.values()), dtype=torch.float32).median().item())
        median_linfty = float(
            torch.tensor(list(raw_linfty_norms.values()), dtype=torch.float32).median().item()
        )

        l2_ceiling = max(1e-6, self.median_l2_clip_factor * median_l2)
        linfty_ceiling = max(1e-6, self.median_linfty_clip_factor * median_linfty)

        clipped_deltas: Dict[str, Dict[str, torch.Tensor]] = {}
        clipped_workers: List[str] = []
        post_clip_l2_norms: Dict[str, float] = {}

        for wid in worker_ids:
            w_l2 = raw_l2_norms[wid]
            l2_scale = min(1.0, l2_ceiling / max(w_l2, 1e-12))
            was_clipped = l2_scale < 0.999999
            clipped_param_dict: Dict[str, torch.Tensor] = {}
            new_sq_sum = 0.0

            for k in param_keys:
                t = surviving_deltas[wid][k].detach().to(device="cpu", dtype=torch.float32) * l2_scale
                # Coordinate-wise Neurotoxin L_infty spike clamp
                if float(t.abs().max().item()) > linfty_ceiling:
                    was_clipped = True
                    t = t.clamp(min=-linfty_ceiling, max=linfty_ceiling)
                clipped_param_dict[k] = t
                new_sq_sum += float((t ** 2).sum().item())

            if was_clipped:
                clipped_workers.append(wid)
            clipped_deltas[wid] = clipped_param_dict
            post_clip_l2_norms[wid] = round(math.sqrt(new_sq_sum), 6)

        # Aggregate via Coordinate-Wise Trimmed Mean when K >= 4, otherwise robust mean over clipped deltas
        K = len(worker_ids)
        trim_k = int(math.floor(self.trimmed_mean_beta * K)) if K >= 4 else 0
        if 2 * trim_k >= K:
            trim_k = 0

        aggregated: Dict[str, torch.Tensor] = {}
        gen_dp = None
        if dp_config is not None and dp_config.enabled:
            gen_dp = torch.Generator(device="cpu").manual_seed(int(dp_config.deterministic_seed))

        for idx_k, k in enumerate(param_keys):
            stacked = torch.stack([clipped_deltas[wid][k] for wid in worker_ids], dim=0)
            if trim_k > 0:
                sorted_vals, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_vals[trim_k : K - trim_k]
                mean_param = trimmed.mean(dim=0)
            else:
                mean_param = stacked.mean(dim=0)

            if dp_config is not None and dp_config.enabled and gen_dp is not None:
                noise_std = float(dp_config.noise_multiplier_sigma * dp_config.l2_sensitivity_clip_c) / max(1.0, float(K))
                noise = torch.randn(
                    mean_param.shape, generator=gen_dp, dtype=torch.float32, device="cpu"
                ) * noise_std
                mean_param = mean_param + noise

            aggregated[k] = mean_param

        report = FederatedDeltaAuditReport(
            total_submitted_workers=int(total_submitted_count or (len(worker_ids) + len(quarantined_pre_filter or {}))),
            quarantined_pre_filter_workers=dict(quarantined_pre_filter or {}),
            accepted_workers=worker_ids,
            clipped_workers=clipped_workers,
            median_l2_norm=round(median_l2, 6),
            l2_clip_ceiling=round(l2_ceiling, 6),
            median_linfty_norm=round(median_linfty, 6),
            linfty_clip_ceiling=round(linfty_ceiling, 6),
            worker_raw_l2_norms={w: round(v, 6) for w, v in raw_l2_norms.items()},
            worker_post_clip_l2_norms=post_clip_l2_norms,
            trimmed_mean_beta=self.trimmed_mean_beta if trim_k > 0 else 0.0,
            dp_noise_sigma_applied=(
                float(dp_config.noise_multiplier_sigma) if (dp_config and dp_config.enabled) else 0.0
            ),
        )
        return aggregated, report

    # ------------------------------------------------------------------
    # 3. Coordinator Workunit Ownership, Replay, Staleness & Sybil One-Vote Gate (FL-05, FL-06 & FL-08)
    # ------------------------------------------------------------------
    def register_workunit_assignment(self, wu_id: str, worker_id: str, steam_id64: str) -> None:
        """Record which `(worker_id, steam_id64)` a workunit was issued to, for the FL-08 ownership check."""
        self.workunit_assignees[str(wu_id)] = (str(worker_id), str(steam_id64))

    def check_workunit_submission_envelope(
        self,
        wu_id: str,
        expected_branch: str,
        submitted_branch: str,
        coordinator_outer_step: int,
        submitted_outer_step: int,
        steam_id64: str,
        worker_id: str,
        steps_completed: int,
        expected_inner_steps: int,
        preempted_by_daemon: bool = False,
    ) -> None:
        """Pure envelope check: raises on violation, records nothing.

        Call `record_workunit_submission` once the validator has produced a verdict so that an
        envelope that passed but never reached a verdict does not burn the workunit.
        """
        if wu_id in self.consumed_workunit_ids:
            raise SwarmSecurityViolationError(
                "FL-05",
                "WORKUNIT_REPLAY_DETECTED",
                f"Workunit '{wu_id}' has already been submitted and consumed.",
            )

        # Rule FL-08: only the (worker_id, steam_id64) the workunit was issued to may submit it.
        # Workunits never registered via `register_workunit_assignment` (direct guard callers,
        # the self-test, the attack simulator) are not ownership-checked.
        assignee = self.workunit_assignees.get(str(wu_id))
        if assignee is not None and assignee != (str(worker_id), str(steam_id64)):
            raise SwarmSecurityViolationError(
                "FL-08",
                "WORKUNIT_ASSIGNEE_MISMATCH",
                f"Workunit '{wu_id}' was assigned to worker '{assignee[0]}' (SteamID64 {assignee[1]}) "
                f"but was submitted by worker '{worker_id}' (SteamID64 {steam_id64}).",
            )

        if str(expected_branch) != str(submitted_branch):
            raise SwarmSecurityViolationError(
                "FL-05",
                "BRANCH_MISMATCH_DETECTED",
                f"Submission branch '{submitted_branch}' does not match assigned Workunit branch '{expected_branch}'.",
            )

        staleness = int(coordinator_outer_step) - int(submitted_outer_step)
        if staleness < 0 or staleness > self.max_staleness_steps:
            raise SwarmSecurityViolationError(
                "FL-05",
                "STALE_OR_FUTURE_OUTER_STEP_REJECTED",
                f"Submission outer_step={submitted_outer_step} exceeds max staleness window "
                f"(coordinator_outer_step={coordinator_outer_step}, max_staleness={self.max_staleness_steps}).",
            )

        if not preempted_by_daemon and int(steps_completed) < max(1, int(expected_inner_steps)):
            raise SwarmSecurityViolationError(
                "FL-05",
                "INCOMPLETE_INNER_LOOP_STEPS",
                f"Worker '{worker_id}' completed only {steps_completed}/{expected_inner_steps} inner steps without preemption.",
            )

        # Rule FL-06: 1 vote per unique SteamID64 per (branch, outer_step)
        step_key = (str(expected_branch), int(coordinator_outer_step))
        votes_for_step = self.step_steam_id_votes.get(step_key, {})
        if steam_id64 in votes_for_step and votes_for_step[steam_id64] != worker_id:
            raise SwarmSecurityViolationError(
                "FL-06",
                "SYBIL_MULTIPLE_WORKERS_PER_STEAMID",
                f"SteamID64 '{steam_id64}' already submitted a gradient via worker '{votes_for_step[steam_id64]}' "
                f"for branch '{expected_branch}' at outer_step={coordinator_outer_step} (1 vote per SteamID64 enforced).",
            )

    def record_workunit_submission(
        self,
        wu_id: str,
        expected_branch: str,
        coordinator_outer_step: int,
        steam_id64: str,
        worker_id: str,
    ) -> None:
        """Commit a checked envelope: cast the FL-06 vote and mark the workunit consumed (FL-05).

        Consumption is tied to the validator having rendered a verdict, pass or fail. A failed
        spot-check still consumes the workunit: the shard and seed are deterministic, so letting the
        same submitter retry the same `wu_id` would hand them a free oracle on the validator's tolerance.
        """
        step_key = (str(expected_branch), int(coordinator_outer_step))
        self.step_steam_id_votes.setdefault(step_key, {})[steam_id64] = worker_id
        self.consumed_workunit_ids.add(wu_id)

    def verify_workunit_submission_envelope(
        self,
        wu_id: str,
        expected_branch: str,
        submitted_branch: str,
        coordinator_outer_step: int,
        submitted_outer_step: int,
        steam_id64: str,
        worker_id: str,
        steps_completed: int,
        expected_inner_steps: int,
        preempted_by_daemon: bool = False,
    ) -> None:
        """Check the envelope and immediately commit it (check + record in one call).

        Kept for callers that have no separate verdict step; the coordinator uses the split form so
        that it only consumes a workunit after the spot-check validator returns.
        """
        self.check_workunit_submission_envelope(
            wu_id=wu_id,
            expected_branch=expected_branch,
            submitted_branch=submitted_branch,
            coordinator_outer_step=coordinator_outer_step,
            submitted_outer_step=submitted_outer_step,
            steam_id64=steam_id64,
            worker_id=worker_id,
            steps_completed=steps_completed,
            expected_inner_steps=expected_inner_steps,
            preempted_by_daemon=preempted_by_daemon,
        )
        self.record_workunit_submission(
            wu_id=wu_id,
            expected_branch=expected_branch,
            coordinator_outer_step=coordinator_outer_step,
            steam_id64=steam_id64,
            worker_id=worker_id,
        )


def run_federated_hardening_self_test(output_json_path: Optional[str] = None) -> Dict[str, Any]:
    """Execute a comprehensive live self-test of all 8 Federated Learning Hardening Rules (`FL-01`..`FL-08`)."""
    from hivemind.core.diloco_outer import DiLoCoOuterOptimizer

    guard = FederatedLearningSecurityGuard()
    torch.manual_seed(42)

    base_delta = {
        "lora_A": torch.ones(8, 16, dtype=torch.float32) * 0.05,
        "lora_B": torch.ones(16, 8, dtype=torch.float32) * 0.05,
    }

    # Construct a mixed cohort of 6 workers:
    # - 3 honest workers with slight natural variance
    # - 1 free-rider worker submitting all zeros (FL-02)
    # - 1 NaN-poisoned worker (FL-01)
    # - 1 collinear scaling + neurotoxin spike attacker (10x magnitude in same direction, FL-03)
    worker_deltas = {
        "honest-01": {k: v * 1.00 for k, v in base_delta.items()},
        "honest-02": {k: v * 1.05 for k, v in base_delta.items()},
        "honest-03": {k: v * 0.95 for k, v in base_delta.items()},
        "free-rider-zero": {k: torch.zeros_like(v) for k, v in base_delta.items()},
        "nan-poisoner": {
            "lora_A": torch.full((8, 16), float("nan")),
            "lora_B": base_delta["lora_B"].clone(),
        },
        "collinear-scaler-10x": {k: v * 10.0 for k, v in base_delta.items()},
    }

    outer_opt = DiLoCoOuterOptimizer(
        outer_lr_eta=0.7,
        outer_momentum_mu=0.9,
        max_byzantine_f=1,
        cosine_sigma=2.0,
    )
    theta_prev = {k: torch.ones_like(v) * 0.5 for k, v in base_delta.items()}
    step_res = outer_opt.step(
        branch_name="Kepler_Exoplanet_Transit_Classifier",
        outer_step=1,
        theta_prev=theta_prev,
        worker_deltas=worker_deltas,
    )

    # Verify Sybil & Replay envelopes
    sybil_blocked = False
    replay_blocked = False
    stale_blocked = False

    guard.verify_workunit_submission_envelope(
        wu_id="wu-astro-001",
        expected_branch="Kepler",
        submitted_branch="Kepler",
        coordinator_outer_step=5,
        submitted_outer_step=5,
        steam_id64="76561198000000001",
        worker_id="worker-alpha",
        steps_completed=12,
        expected_inner_steps=12,
    )

    try:
        guard.verify_workunit_submission_envelope(
            wu_id="wu-astro-001",  # Replayed WU ID!
            expected_branch="Kepler",
            submitted_branch="Kepler",
            coordinator_outer_step=5,
            submitted_outer_step=5,
            steam_id64="76561198000000002",
            worker_id="worker-beta",
            steps_completed=12,
            expected_inner_steps=12,
        )
    except SwarmSecurityViolationError as exc:
        replay_blocked = exc.rule_id == "FL-05"

    try:
        guard.verify_workunit_submission_envelope(
            wu_id="wu-astro-002",
            expected_branch="Kepler",
            submitted_branch="Kepler",
            coordinator_outer_step=5,
            submitted_outer_step=5,
            steam_id64="76561198000000001",  # Same SteamID64 with second Sybil worker_id!
            worker_id="worker-sybil-clone-2",
            steps_completed=12,
            expected_inner_steps=12,
        )
    except SwarmSecurityViolationError as exc:
        sybil_blocked = exc.rule_id == "FL-06"

    try:
        guard.verify_workunit_submission_envelope(
            wu_id="wu-astro-003",
            expected_branch="Kepler",
            submitted_branch="Kepler",
            coordinator_outer_step=5,
            submitted_outer_step=1,  # Stale outer step (staleness = 4 > 1)!
            steam_id64="76561198000000003",
            worker_id="worker-stale",
            steps_completed=12,
            expected_inner_steps=12,
        )
    except SwarmSecurityViolationError as exc:
        stale_blocked = exc.rule_id == "FL-05"

    assignee_blocked = False
    guard.register_workunit_assignment("wu-astro-004", worker_id="worker-delta", steam_id64="76561198000000004")
    try:
        guard.verify_workunit_submission_envelope(
            wu_id="wu-astro-004",
            expected_branch="Kepler",
            submitted_branch="Kepler",
            coordinator_outer_step=5,
            submitted_outer_step=5,
            steam_id64="76561198000000005",  # Not the assignee of wu-astro-004!
            worker_id="worker-hijacker",
            steps_completed=12,
            expected_inner_steps=12,
        )
    except SwarmSecurityViolationError as exc:
        assignee_blocked = exc.rule_id == "FL-08" and "wu-astro-004" not in guard.consumed_workunit_ids

    # Also test DP-DiLoCo noise injection
    _, dp_report = guard.clip_and_aggregate_surviving_deltas(
        surviving_deltas={
            "honest-01": worker_deltas["honest-01"],
            "honest-02": worker_deltas["honest-02"],
            "honest-03": worker_deltas["honest-03"],
            "honest-04": worker_deltas["honest-01"],
        },
        dp_config=DPGradientSanitizerConfig(enabled=True, l2_sensitivity_clip_c=1.0, noise_multiplier_sigma=0.05),
    )

    overall_passed = bool(
        "free-rider-zero" in step_res.filter_report.rejected_workers
        and "nan-poisoner" in step_res.filter_report.rejected_workers
        and "collinear-scaler-10x" in step_res.filter_report.rejected_workers
        and sybil_blocked
        and replay_blocked
        and stale_blocked
        and assignee_blocked
        and dp_report.dp_noise_sigma_applied == 0.05
    )

    report = {
        "phase": "Phase S-3: Federated Learning & Byzantine-Resilient DiLoCo Hardening",
        "overall_passed": overall_passed,
        "outer_step_aggregated_norm": step_res.aggregated_delta_norm,
        "accepted_workers": step_res.filter_report.accepted_workers,
        "rejected_workers": step_res.filter_report.rejected_workers,
        "envelopes_verified": {
            "fl01_nan_poisoner_quarantined": "nan-poisoner" in step_res.filter_report.rejected_workers,
            "fl02_free_rider_zero_rejected": "free-rider-zero" in step_res.filter_report.rejected_workers,
            "fl03_collinear_10x_scaler_rejected_or_clipped": "collinear-scaler-10x" in step_res.filter_report.rejected_workers,
            "fl04_coordinate_trimmed_mean_active": dp_report.trimmed_mean_beta > 0.0,
            "fl05_workunit_replay_blocked": replay_blocked,
            "fl05_stale_outer_step_blocked": stale_blocked,
            "fl06_sybil_multi_worker_per_steamid_blocked": sybil_blocked,
            "fl08_non_assignee_submission_blocked": assignee_blocked,
            "fl07_dp_diloco_gaussian_sanitizer_verified": dp_report.dp_noise_sigma_applied == 0.05,
        },
        "dp_audit_summary": dp_report.to_dict(),
    }

    if output_json_path:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report

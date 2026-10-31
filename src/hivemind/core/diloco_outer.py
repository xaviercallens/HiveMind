# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Phase B (BOINC Aggregator): Byzantine-Resilient Multi-Krum, Adaptive Norm Clipping & Nesterov Outer Loop.

Implements:
0. Phase S-3 Pre-Krum Finiteness & Free-Rider Zero-Gradient Sanitization (`FL-01`, `FL-02`).
1. Multi-Krum scoring over flattened pseudo-gradients Delta_k^(t) (sum of distances to
   K - f - 2 nearest neighbors).
2. Median pairwise cosine similarity 2-sigma outlier rejection.
3. Phase S-3 Adaptive Median-L2 & Neurotoxin Coordinate-L_infty Clipping + Coordinate-Wise
   Trimmed Mean + optional DP-DiLoCo Gaussian Privacy Sanitization (`FL-03`, `FL-04`, `FL-07`).
4. DiLoCo Nesterov Momentum outer optimization step (mu = 0.9, eta_outer = 0.7):
     Delta^(t) = TrimmedMean_{k in K_valid}( Clip(Delta_k^(t)) )
     v^(t)     = mu * v^(t-1) + Delta^(t)
     theta^(t) = theta^(t-1) - eta_outer * (mu * v^(t) + Delta^(t))
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import torch

from hivemind.security.federated_learning_hardening import (
    DPGradientSanitizerConfig,
    FederatedDeltaAuditReport,
    FederatedLearningSecurityGuard,
)
from hivemind.swarm.decoupled_diloco import (
    DecoupledDiLoCoQuorumSyncer,
    LearnerArrival,
    SyncerRoundTelemetry,
)
from hivemind.swarm.diloco_nextgen import (
    CGADStepTelemetry,
    CPUOffloadedFlatBucketOuterState,
    CosineGatedAdamDecayOuterOptimizer,
)


@dataclass
class ByzantineFilterReport:
    """Detailed audit report of Multi-Krum + Cosine 2-sigma + Pre-Krum Byzantine filtering."""

    total_submissions: int
    max_byzantine_f: int
    accepted_workers: list[str]
    rejected_workers: list[str]
    krum_scores: dict[str, float]
    mean_cosine_similarities: dict[str, float]
    median_cosine: float
    cosine_threshold_lower: float
    quarantined_pre_krum: dict[str, str] = field(default_factory=dict)
    krum_guarantee: bool = False


@dataclass
class OuterStepResult:
    """Result of one DiLoCo outer aggregation step on the BOINC server."""

    branch_name: str
    outer_step: int
    theta_updated: dict[str, torch.Tensor] = field(repr=False)
    velocity_updated: dict[str, torch.Tensor] = field(repr=False)
    aggregated_delta_norm: float
    filter_report: ByzantineFilterReport
    federated_security_report: FederatedDeltaAuditReport | None = None
    outer_optimizer_mode: str = "nesterov"
    mean_cosine_gate: float | None = None
    extra_gpu_vram_bytes: int = 0
    cgad_telemetry: list[CGADStepTelemetry] = field(default_factory=list, repr=False)
    quorum_telemetry: dict[str, SyncerRoundTelemetry] = field(default_factory=dict, repr=False)
    momentum_restarted: bool = False


def flatten_delta_dict(delta_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a dictionary of LoRA tensors in deterministic key order into a 1D FP32 vector."""
    parts = [
        delta_dict[k].detach().to(torch.float32).cpu().reshape(-1)
        for k in sorted(delta_dict.keys())
    ]
    return torch.cat(parts, dim=0)


def filter_byzantine_deltas_multikrum(
    worker_deltas: dict[str, dict[str, torch.Tensor]],
    max_byzantine_f: int = 1,
    cosine_sigma: float = 2.0,
    reference_keys_and_shapes: dict[str, torch.Size] | None = None,
) -> ByzantineFilterReport:
    """Filter worker pseudo-gradients using Pre-Krum Sanitization, Multi-Krum, and a 2-sigma pairwise cosine filter."""
    all_worker_ids = list(worker_deltas.keys())
    total_K = len(all_worker_ids)
    if total_K == 0:
        raise ValueError("Cannot run Multi-Krum on empty worker_deltas.")

    fl_guard = FederatedLearningSecurityGuard()
    clean_deltas, quarantined = fl_guard.sanitize_worker_deltas_pre_krum(
        worker_deltas, reference_keys_and_shapes=reference_keys_and_shapes
    )
    worker_ids = list(clean_deltas.keys())
    K = len(worker_ids)

    if K == 0:
        raise ValueError(
            f"All {total_K} worker deltas were quarantined during pre-Krum security sanitization: {quarantined}"
        )

    if K <= 2:
        return ByzantineFilterReport(
            total_submissions=total_K,
            max_byzantine_f=0,
            accepted_workers=worker_ids,
            rejected_workers=list(quarantined.keys()),
            krum_scores={wid: 0.0 for wid in worker_ids},
            mean_cosine_similarities={wid: 1.0 for wid in worker_ids},
            median_cosine=1.0,
            cosine_threshold_lower=0.0,
            quarantined_pre_krum=quarantined,
            krum_guarantee=False,
        )

    # Multi-Krum theoretical guarantee requires K >= 2*f + 3 (IP-34)
    f = min(max_byzantine_f, max(0, (K - 3) // 2))
    krum_guarantee = bool(max_byzantine_f > 0 and K >= 2 * max_byzantine_f + 3)
    num_neighbors = max(1, K - f - 2)

    # Accumulate exact [K, K] inner-product Gram matrix parameter-by-parameter (<1 MB peak RAM)
    gram = torch.zeros((K, K), dtype=torch.float64)
    param_keys = sorted(clean_deltas[worker_ids[0]].keys())
    for k in param_keys:
        p_mat = torch.stack(
            [
                clean_deltas[wid][k].detach().to(dtype=torch.float32, device="cpu").reshape(-1)
                for wid in worker_ids
            ],
            dim=0,
        )
        gram += torch.mm(p_mat, p_mat.t()).to(torch.float64)
        del p_mat

    diag = torch.diag(gram).clamp_min(0.0)
    # Pairwise squared Euclidean distances: ||u - v||^2 = ||u||^2 + ||v||^2 - 2 <u, v>
    dist_matrix = (diag.unsqueeze(1) + diag.unsqueeze(0) - 2.0 * gram).clamp_min(0.0).to(torch.float32)

    krum_scores: dict[str, float] = {}
    for i, wid in enumerate(worker_ids):
        # Exclude self (distance 0 at index i)
        row = torch.cat([dist_matrix[i, :i], dist_matrix[i, i + 1 :]])
        nearest_vals, _ = torch.topk(row, k=min(num_neighbors, row.numel()), largest=False)
        krum_scores[wid] = round(float(nearest_vals.sum().item()), 6)

    # Pairwise Cosine Similarity: [K, K]
    norms = torch.sqrt(diag).clamp_min(1e-12)
    cos_matrix = (gram / (norms.unsqueeze(1) * norms.unsqueeze(0))).clamp(-1.0, 1.0).to(torch.float32)

    mean_cos_per_worker: dict[str, float] = {}
    off_diag_cos_values: list[float] = []
    for i, wid in enumerate(worker_ids):
        others = torch.cat([cos_matrix[i, :i], cos_matrix[i, i + 1 :]])
        # Median cosine similarity to peers is robust against a single huge outlier
        peer_median_cos = float(others.median().item())
        mean_cos_per_worker[wid] = round(peer_median_cos, 6)
        off_diag_cos_values.extend(others.tolist())

    off_diag_tensor = torch.tensor(off_diag_cos_values, dtype=torch.float32)
    median_cos = float(off_diag_tensor.median().item())
    std_cos = float(off_diag_tensor.std(unbiased=False).item())
    # Reject anything falling outside cosine_sigma below the median (or negative alignment)
    cos_lower_bound = max(-0.05, median_cos - cosine_sigma * max(std_cos, 0.05))

    # Multi-Krum selects the top (K - f_post_quarantine) workers with smallest Krum distance score
    f_post_quarantine = max(0, f - len(quarantined))
    sorted_by_krum = sorted(worker_ids, key=lambda w: krum_scores[w])
    krum_survivors = set(sorted_by_krum[: max(1, K - f_post_quarantine)])

    accepted: list[str] = []
    rejected: list[str] = list(quarantined.keys())
    for wid in worker_ids:
        if wid in krum_survivors and mean_cos_per_worker[wid] >= cos_lower_bound:
            accepted.append(wid)
        else:
            rejected.append(wid)

    # Ensure at least the single best Multi-Krum worker survives
    if not accepted:
        best_worker = sorted_by_krum[0]
        accepted = [best_worker]
        rejected = list(quarantined.keys()) + [w for w in worker_ids if w != best_worker]

    return ByzantineFilterReport(
        total_submissions=total_K,
        max_byzantine_f=f,
        accepted_workers=accepted,
        rejected_workers=rejected,
        krum_scores=krum_scores,
        mean_cosine_similarities=mean_cos_per_worker,
        median_cosine=round(median_cos, 6),
        cosine_threshold_lower=round(cos_lower_bound, 6),
        quarantined_pre_krum=quarantined,
        krum_guarantee=krum_guarantee,
    )


class DiLoCoOuterOptimizer:
    """Server-side DiLoCo outer optimizer with Multi-Krum & Phase S-3 Federated Hardening.

    Supports two outer update families:

    * ``"nesterov"`` — the original DiLoCo Nesterov momentum update (default).
    * ``"cosine_gated_adam_decay"`` — ``SI-06`` PA-CGAD, which gates each aggregated
      pseudo-gradient by a cosine/age decay before updating Adam's moments. Stale
      fragments beyond ``cgad_tau_cut`` are frozen instead of polluting the moments.

    Setting ``use_cpu_flat_bucket=True`` keeps the outer master parameters (and the
    Nesterov velocity / CGAD moments) in contiguous CPU buffers, adding ``0 B`` of GPU
    VRAM. The Byzantine filtering and federated security pipeline is unchanged.
    """

    _VALID_OUTER_MODES = frozenset({"nesterov", "cosine_gated_adam_decay"})

    def __init__(
        self,
        outer_lr_eta: float = 0.7,
        outer_momentum_mu: float = 0.9,
        max_byzantine_f: int = 1,
        cosine_sigma: float = 2.0,
        dp_config: DPGradientSanitizerConfig | None = None,
        outer_optimizer_mode: str = "nesterov",
        use_cpu_flat_bucket: bool = False,
        cgad_lr: float | None = None,
        cgad_alpha: float = 0.2,
        cgad_tau_cut: float = 32.0,
        cgad_betas: tuple[float, float] | None = None,
        cgad_eps: float = 1e-8,
        cgad_weight_decay: float | None = None,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.01,
        momentum_restart_period: int | None = None,
    ) -> None:
        if outer_optimizer_mode not in self._VALID_OUTER_MODES:
            raise ValueError(
                f"outer_optimizer_mode must be one of {sorted(self._VALID_OUTER_MODES)}, "
                f"got {outer_optimizer_mode!r}"
            )
        if momentum_restart_period is not None and (
            isinstance(momentum_restart_period, bool)
            or not isinstance(momentum_restart_period, int)
            or momentum_restart_period < 1
        ):
            raise ValueError("momentum_restart_period must be an integer >= 1 or None")
        eff_betas = cgad_betas if cgad_betas is not None else (beta1, beta2)
        if not (0.0 <= eff_betas[0] < 1.0 and 0.0 <= eff_betas[1] < 1.0):
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        eff_wd = cgad_weight_decay if cgad_weight_decay is not None else weight_decay
        if not math.isfinite(eff_wd) or eff_wd < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        eff_cgad_lr = cgad_lr if cgad_lr is not None else outer_lr_eta
        if (outer_optimizer_mode != "nesterov" or use_cpu_flat_bucket) and (
            not math.isfinite(outer_lr_eta) or outer_lr_eta <= 0.0
        ):
            raise ValueError("outer_lr_eta must be finite and positive")
        self.outer_lr_eta = outer_lr_eta
        self.outer_momentum_mu = outer_momentum_mu
        self.max_byzantine_f = max_byzantine_f
        self.cosine_sigma = cosine_sigma
        self.dp_config = dp_config
        self.outer_optimizer_mode = outer_optimizer_mode
        self.use_cpu_flat_bucket = bool(use_cpu_flat_bucket)
        self.cgad_lr = float(eff_cgad_lr)
        self.cgad_alpha = cgad_alpha
        self.cgad_tau_cut = cgad_tau_cut
        self.cgad_betas = (float(eff_betas[0]), float(eff_betas[1]))
        self.cgad_eps = cgad_eps
        self.cgad_weight_decay = float(eff_wd)
        self.beta1, self.beta2 = self.cgad_betas
        self.weight_decay = self.cgad_weight_decay
        self.momentum_restart_period = momentum_restart_period
        self.fl_guard = FederatedLearningSecurityGuard()
        # Velocity buffers per expert branch: branch_name -> {param_name: tensor}
        self.velocities: dict[str, dict[str, torch.Tensor]] = {}
        # Persistent PA-CGAD state (moments are CPU-resident) per expert branch.
        self.cgad_optimizers: dict[str, CosineGatedAdamDecayOuterOptimizer] = {}
        # Optional contiguous CPU flat-bucket mirror per expert branch.
        self.cpu_buckets: dict[str, CPUOffloadedFlatBucketOuterState] = {}
        self.cpu_flat_buckets = self.cpu_buckets

    def _get_cpu_bucket(
        self, branch_name: str, theta_prev: dict[str, torch.Tensor]
    ) -> CPUOffloadedFlatBucketOuterState:
        """Return the branch's CPU flat-bucket, creating it from ``theta_prev`` once."""
        bucket = self.cpu_buckets.get(branch_name)
        if bucket is None:
            bucket = CPUOffloadedFlatBucketOuterState(theta_prev)
            self.cpu_buckets[branch_name] = bucket
        return bucket

    def _get_cgad_optimizer(self, branch_name: str) -> CosineGatedAdamDecayOuterOptimizer:
        """Return the branch's persistent PA-CGAD optimizer (moments live on CPU)."""
        optimizer = self.cgad_optimizers.get(branch_name)
        if optimizer is None:
            optimizer = CosineGatedAdamDecayOuterOptimizer(
                lr=self.cgad_lr,
                alpha=self.cgad_alpha,
                tau_cut=self.cgad_tau_cut,
                betas=self.cgad_betas,
                eps=self.cgad_eps,
                weight_decay=self.cgad_weight_decay,
            )
            self.cgad_optimizers[branch_name] = optimizer
        return optimizer

    def step(
        self,
        branch_name: str,
        outer_step: int,
        theta_prev: dict[str, torch.Tensor],
        worker_deltas: dict[str, dict[str, torch.Tensor]],
        worker_delays: dict[str, float] | None = None,
        synced_fragments: Sequence[str] | None = None,
        learner_arrivals: Sequence[LearnerArrival] | None = None,
        min_quorum_k: int | None = None,
        grace_window_s: float = 0.0,
        quorum_merge_mode: str = "rda",
        momentum_restart_period: int | None = None,
    ) -> OuterStepResult:
        """Aggregate valid worker deltas then apply the configured outer optimizer.

        ``worker_delays`` and ``synced_fragments`` are forwarded to PA-CGAD for
        per-fragment staleness gating (ignored by the Nesterov path). When
        ``learner_arrivals`` is provided (`HM-S9-01` / `SI-12`), surviving workers
        are filtered by quorum + grace window and merged via `DecoupledDiLoCoQuorumSyncer`
        (`"rda"` Radial-Directional Averaging or `"direct"` weighted mean).
        When ``momentum_restart_period`` (`K_restart`, `HM-S9-05`) is set, outer
        velocity / first-moment buffers are reset every `K_restart` outer steps.
        """
        eff_restart = (
            momentum_restart_period
            if momentum_restart_period is not None
            else self.momentum_restart_period
        )
        if eff_restart is not None and (
            isinstance(eff_restart, bool) or not isinstance(eff_restart, int) or eff_restart < 1
        ):
            raise ValueError("momentum_restart_period must be an integer >= 1 or None")
        should_restart = bool(
            eff_restart is not None and outer_step > 0 and (outer_step % eff_restart == 0)
        )

        expected_schema = {k: v.shape for k, v in theta_prev.items() if isinstance(v, torch.Tensor)}
        if self.outer_optimizer_mode == "cosine_gated_adam_decay" and synced_fragments is not None:
            declared_keys = set(synced_fragments)
            if not declared_keys <= set(theta_prev):
                raise ValueError("synced_fragments must name existing branch parameters")
            if declared_keys and not all(
                set(d.keys()) == set(theta_prev.keys()) for d in worker_deltas.values()
            ):
                expected_schema = {
                    k: theta_prev[k].shape
                    for k in declared_keys
                    if isinstance(theta_prev[k], torch.Tensor)
                }
        filter_report = filter_byzantine_deltas_multikrum(
            worker_deltas=worker_deltas,
            max_byzantine_f=self.max_byzantine_f,
            cosine_sigma=self.cosine_sigma,
            reference_keys_and_shapes=expected_schema or None,
        )
        valid_ids = filter_report.accepted_workers
        surviving_deltas = {wid: worker_deltas[wid] for wid in valid_ids}

        aggregated_deltas, fl_report = self.fl_guard.clip_and_aggregate_surviving_deltas(
            surviving_deltas=surviving_deltas,
            quarantined_pre_filter=filter_report.quarantined_pre_krum,
            total_submitted_count=filter_report.total_submissions,
            dp_config=self.dp_config,
        )

        quorum_telemetry: dict[str, SyncerRoundTelemetry] = {}
        if learner_arrivals is not None:
            surviving_arrivals = [a for a in learner_arrivals if a.learner_id in surviving_deltas]
            eff_k = min_quorum_k if min_quorum_k is not None else max(1, len(surviving_arrivals))
            syncer = DecoupledDiLoCoQuorumSyncer(
                num_learners_m=max(len(surviving_deltas), len(surviving_arrivals), eff_k),
                min_quorum_k=eff_k,
                merge_mode=quorum_merge_mode,
            )
            for k in sorted(aggregated_deltas.keys()):
                prev_t = theta_prev[k].detach().to(device="cpu", dtype=torch.float32)
                learner_frags = {
                    wid: prev_t - surviving_deltas[wid][k].detach().to(device="cpu", dtype=torch.float32)
                    for wid in surviving_deltas
                }
                merged_frag, _, q_tel = syncer.merge_fragment(
                    fragment_name=k,
                    global_prev=prev_t,
                    learner_fragments=learner_frags,
                    arrivals=surviving_arrivals,
                    grace_window_s=grace_window_s,
                )
                quorum_telemetry[k] = q_tel
                if merged_frag is not None:
                    aggregated_deltas[k] = merged_frag

        if self.outer_optimizer_mode != "nesterov" or self.use_cpu_flat_bucket:
            parameter_keys = set(theta_prev)
            required_keys = parameter_keys
            if self.outer_optimizer_mode == "cosine_gated_adam_decay" and synced_fragments is not None:
                required_keys = set(synced_fragments)
                if not required_keys <= parameter_keys:
                    raise ValueError("synced_fragments must name existing branch parameters")
            aggregate_keys = set(aggregated_deltas)
            if not required_keys <= aggregate_keys or not aggregate_keys <= parameter_keys or any(
                aggregated_deltas[k].shape != theta_prev[k].shape for k in aggregate_keys
            ):
                raise ValueError("aggregated deltas must match all branch parameter keys and shapes")
            existing_bucket = self.cpu_buckets.get(branch_name)
            if existing_bucket is not None and (
                set(theta_prev) != set(existing_bucket.keys)
                or any(theta_prev[k].shape != existing_bucket.shapes[k] for k in existing_bucket.keys)
            ):
                raise ValueError("branch parameters must match the existing flat bucket")
            if branch_name not in self.velocities:
                self.cgad_optimizers.pop(branch_name, None)
                self.cpu_buckets.pop(branch_name, None)

        sq_norm_sum = 0.0
        for avg_delta in aggregated_deltas.values():
            sq_norm_sum += float((avg_delta.detach().to(torch.float32) ** 2).sum().item())
        aggregated_delta_norm = round(sq_norm_sum**0.5, 6)

        cgad_telemetry: list[CGADStepTelemetry] = []
        mean_cosine_gate: float | None = None
        velocity_updated: dict[str, torch.Tensor] = {}

        if self.outer_optimizer_mode == "cosine_gated_adam_decay":
            params_cpu = {
                k: v.detach().to(device="cpu", dtype=torch.float32)
                for k, v in theta_prev.items()
            }
            cgad = self._get_cgad_optimizer(branch_name)
            if should_restart:
                for buf in cgad.m_buffers.values():
                    buf.zero_()
            theta_updated, cgad_telemetry = cgad.step_fragments(
                params=params_cpu,
                pseudo_gradients=aggregated_deltas,
                worker_delays=worker_delays,
                synced_fragments=synced_fragments,
            )
            velocity_updated = {k: v.clone() for k, v in cgad.m_buffers.items()}
            self.velocities[branch_name] = velocity_updated
            if cgad_telemetry:
                mean_cosine_gate = round(
                    sum(t.cosine_gate_gamma for t in cgad_telemetry) / len(cgad_telemetry),
                    6,
                )
            if self.use_cpu_flat_bucket:
                theta_updated = self._get_cpu_bucket(
                    branch_name, params_cpu
                ).apply_outer_update_and_sync_inner(theta_updated)
        elif self.use_cpu_flat_bucket:
            params_cpu = {
                k: v.detach().to(device="cpu", dtype=torch.float32)
                for k, v in theta_prev.items()
            }
            bucket = self._get_cpu_bucket(branch_name, params_cpu)
            if should_restart:
                bucket.reset_velocity()
            theta_updated, velocity_updated = bucket.outer_nesterov_step(
                pseudo_gradients=aggregated_deltas,
                lr=self.outer_lr_eta,
                momentum=self.outer_momentum_mu,
                theta_prev=params_cpu,
            )
            self.velocities[branch_name] = velocity_updated
        else:
            if branch_name not in self.velocities:
                self.velocities[branch_name] = {
                    k: torch.zeros_like(v, dtype=torch.float32, device="cpu")
                    for k, v in theta_prev.items()
                }
            elif should_restart:
                for v_buf in self.velocities[branch_name].values():
                    v_buf.zero_()

            vel_branch = self.velocities[branch_name]
            theta_updated = {}

            for name, param_prev in theta_prev.items():
                avg_delta = aggregated_deltas[name]

                # v^(t) = mu * v^(t-1) + Delta^(t)
                v_t = self.outer_momentum_mu * vel_branch[name] + avg_delta
                vel_branch[name] = v_t

                # theta^(t) = theta^(t-1) - eta_outer * (mu * v^(t) + Delta^(t))
                nesterov_step = self.outer_momentum_mu * v_t + avg_delta
                theta_updated[name] = (
                    param_prev.detach().to(torch.float32).cpu()
                    - self.outer_lr_eta * nesterov_step
                )
            velocity_updated = vel_branch

        return OuterStepResult(
            branch_name=branch_name,
            outer_step=outer_step,
            theta_updated=theta_updated,
            velocity_updated=velocity_updated,
            aggregated_delta_norm=aggregated_delta_norm,
            filter_report=filter_report,
            federated_security_report=fl_report,
            outer_optimizer_mode=self.outer_optimizer_mode,
            mean_cosine_gate=mean_cosine_gate,
            extra_gpu_vram_bytes=0,
            cgad_telemetry=cgad_telemetry,
            quorum_telemetry=quorum_telemetry,
            momentum_restarted=should_restart,
        )

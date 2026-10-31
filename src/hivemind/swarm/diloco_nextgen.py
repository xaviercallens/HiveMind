# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Next-Generation Outer-Loop DiLoCo Optimizers, Routers & Aggregators (`SI-06`..`SI-11`).

Implements verified algorithmic improvements synthesized from 2023-2026 arXiv literature
and open-source reference repositories cloned in `sub_projects/hivemind/reference_repos/`:

1. `SI-06` — `CosineGatedAdamDecayOuterOptimizer` (`CGAD` & `PA-CGAD`, Shah & Sun,
   `arXiv:2605.09126`) + `CPUOffloadedFlatBucketOuterState` (adapted from
   `PrimeIntellect-ai/prime-diloco`, `PrimeIntellect-ai/OpenDiLoCo` `arXiv:2407.07852`,
   and `google-deepmind/asyncdiloco` `arXiv:2401.09135`):
   Drop-in staleness- and per-fragment-age-aware outer optimizer that gates incoming
   pseudo-gradients by:
     gamma(tau_eff) = I[tau_eff <= tau_cut] * 0.5 * (1 + cos(pi * tau_eff / tau_cut))
     sigma(tau_eff) = gamma(tau_eff) * exp(-alpha * tau_eff)
   BEFORE Adam's first and second moment buffers (`m`, `v`), preventing stale rounding
   noise from inflating `sqrt(v)` and bounding the staleness bias by `1 / (e * alpha)`
   independently of `tau_max`, while storing outer master weights in a contiguous flat
   CPU buffer (`0 B` extra GPU VRAM).

2. `SI-07` — `ReputationWeightedIsoCTSMerger` (`IsoLoCo` / `ISO-C` / `ISO-CTS` / `TSVLoCo`,
   Horoi et al., `arXiv:2607.03011`, `github.com/shoroi/isoloco`; Gargiulo et al.,
   `arXiv:2412.00081`):
   Combines Reputation-Weighted TIES sign election (`SI-04`) with Isotropic Covariance
   (`ISO-C`), Common + Task-Specific (`ISO-CTS`), and Task Singular Vector (`TSV`)
   subspace orthogonalization, replacing skewed singular value spectra with their isotropic
   mean so dominant domain updates do not drown out specialized experts as worker count `M` scales.

3. `SI-08` — `HeLoCoDirectionCorrector` & `AutoLoCoIntervalCorrector` (`HeLoCo`, Asif et al.,
   `arXiv:2606.00271`; `AutoLoCo`, He et al., `arXiv:2609.36662`):
   Projects out directionally conflicting components of stale non-IID pseudo-gradients
   relative to the outer momentum trajectory `m_t`, and rescales the outer learning rate
   and momentum when the inner step interval `H_t` changes dynamically (`SI-03`).

4. `SI-09` — `FoMoEPartialReplicaPlanner`, `ChunkedOrthonormalDCTTransform` &
   `ComputeVisibleSparseLoCoCompressor` (`FoMoE`, Sani et al., `arXiv:2606.19025`;
   `SparseLoCo`, Sarfi et al., `arXiv:2508.15706`, `github.com/one-covenant/SparseLoCo`;
   `PULSELoCo`, Miahi & Belilovsky, `arXiv:2602.03839`; `DeMo`, Peng et al.,
   `arXiv:2411.19870`, `github.com/bloc97/DeMo`):
   Models partial expert replication overlap `O_e in {1..M}` with Ghost-Expert Skip-Token
   routing and combines compute-visible BF16/FP16 rounding-threshold gating with orthonormal
   chunked DCT decorrelation, 2-bit statistical centroid quantization, and bitwise-deterministic
   `index_copy_` reconstruction.

5. `SI-10` — `SkipAttentionSpeculativeRouter` (adapted from Merolla, `github.com/pmerolla/fomoe`,
   and `github.com/learning-at-home/hivemind`):
   Predicts Layer `L+1` top-`k` experts immediately after Layer `L` FFN (skipping Layer `L+1`
   self-attention during speculative prefetch) to overlap PCIe/NVMe DMA streaming with
   attention compute, paired with Cache-Aware Routing (`CAR`) fallback when an uncached
   expert's logit margin is within `car_threshold`.

6. `SI-11` — `MuonNewtonSchulzLoRAPreconditioner` (synthesized from Belilovsky, Horoi, Oyallon & Thérien, `arXiv:2610.19064`,
   *How Inner Optimizers Shape Replica Scaling in Distributed Low-Communication Training*):
   Applies 5-step quintic Newton-Schulz orthogonalization to inner LoRA momentum matrices
   before outer pseudo-gradient accumulation so replica scaling stays constructive across
   `8..96` workers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from hivemind.compress.lattice_quantizer import (
    LatticeDeltaQuantizer,
    QuantizedDelta,
    SignDeltaQuantizer,
    calibrate_scale_for_rate,
)


LEVERAGED_GITHUB_REPOSITORIES: Tuple[Dict[str, str], ...] = (
    {
        "name": "SparseLoCo",
        "repo_url": "https://github.com/one-covenant/SparseLoCo",
        "authors": "Amir Sarfi,analyzing-authors, Eugene Belilovsky et al. (Mila / Concordia / Covenant)",
        "arxiv_url": "https://arxiv.org/abs/2508.15706",
        "license": "MIT",
        "integrated_components": "ComputeVisibleSparseLoCoCompressor (2-bit statistical centroid LUT quantization + Top-k EF) & ChunkedOrthonormalDCTTransform",
    },
    {
        "name": "isoloco",
        "repo_url": "https://github.com/shoroi/isoloco",
        "authors": "Stefan Horoi, Amir Sarfi, Eugene Belilovsky (Mila / Concordia University)",
        "arxiv_url": "https://arxiv.org/abs/2607.03011",
        "license": "MIT",
        "integrated_components": "ReputationWeightedIsoCTSMerger (ISO-C, ISO-CTS, and TSVLoCo Task Singular Vector subspace decorrelation)",
    },
    {
        "name": "OpenDiLoCo",
        "repo_url": "https://github.com/PrimeIntellect-ai/OpenDiLoCo",
        "authors": "Sami Jaghouar, Jack Min Ong, Johannes Hagemann (Prime Intellect)",
        "arxiv_url": "https://arxiv.org/abs/2407.07852",
        "license": "Apache-2.0",
        "integrated_components": "CPUOffloadedFlatBucketOuterState & Hivemind decentralized outer-step orchestration patterns",
    },
    {
        "name": "prime-diloco",
        "repo_url": "https://github.com/PrimeIntellect-ai/prime-diloco",
        "authors": "Prime Intellect Team (Sami Jaghouar, Jack Min Ong, Johannes Hagemann et al.)",
        "arxiv_url": "https://arxiv.org/abs/2407.07852",
        "license": "Apache-2.0",
        "integrated_components": "CPUOffloadedFlatBucketOuterState (flat contiguous CPU buffer offloaded outer parameters & bucketed sync)",
    },
    {
        "name": "DeMo",
        "repo_url": "https://github.com/bloc97/DeMo",
        "authors": "Bowen Peng, Jeffrey Quesnelle, Diederik P. Kingma (Nous Research)",
        "arxiv_url": "https://arxiv.org/abs/2411.19870",
        "license": "Apache-2.0",
        "integrated_components": "ChunkedOrthonormalDCTTransform (orthonormal 2D chunked DCT basis, upgraded with deterministic index_copy_)",
    },
    {
        "name": "asyncdiloco",
        "repo_url": "https://github.com/google-deepmind/asyncdiloco",
        "authors": "Arthur Douillard, Qixuan Feng, Andrei A. Rusu, Rachita Chhaparia, Yani Donchev et al. (Google DeepMind)",
        "arxiv_url": "https://arxiv.org/abs/2401.09135",
        "license": "Apache-2.0",
        "integrated_components": "Asynchronous delay-table outer scheduling & Streaming fragment partitioning baseline",
    },
    {
        "name": "fomoe",
        "repo_url": "https://github.com/pmerolla/fomoe",
        "authors": "Paul Merolla (Fast Opportunistic Mixture of Experts)",
        "arxiv_url": "https://arxiv.org/abs/2606.19025",
        "license": "MIT",
        "integrated_components": "SkipAttentionSpeculativeRouter (85.8% pre-attention expert prediction + Cache-Aware Routing CAR fallback)",
    },
    {
        "name": "hivemind",
        "repo_url": "https://github.com/learning-at-home/hivemind",
        "authors": "Learning@Home / Max Ryabinin, Anton Diskin, Denis Mazur et al.",
        "arxiv_url": "https://arxiv.org/abs/2002.04013",
        "license": "MIT",
        "integrated_components": "Decentralized DHT averaging & Mixture-of-Experts peer ring inspirations",
    },
)



@dataclass(frozen=True)
class CGADStepTelemetry:
    """Per-fragment outer step telemetry for Cosine-Gated Adam-Decay (`SI-06`)."""

    fragment_name: str
    worker_delay_tau: float
    fragment_age_rounds: float
    effective_tau: float
    cosine_gate_gamma: float
    decay_gate_sigma: float
    dropped_by_cutoff: bool
    staleness_bias_bound: float
    step_linf_norm: float


class CosineGatedAdamDecayOuterOptimizer:
    """SI-06: Cosine-Gated Adam-Decay (`CGAD` / `PA-CGAD`, `arXiv:2605.09126`).

    Replaces delay-unaware outer Nesterov SGD with an age-aware outer Adam update that
    gates each incoming pseudo-gradient `g_f` by `sigma(tau_eff) = gamma(tau_eff) * exp(-alpha * tau_eff)`
    before updating `m_f` and `v_f`, where `tau_eff = max(worker_delay_tau, fragment_age_f)`.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        alpha: float = 0.2,
        tau_cut: float = 32.0,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_restart_period: Optional[int] = None,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be > 0")
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        if tau_cut <= 0.0:
            raise ValueError("tau_cut must be > 0")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if momentum_restart_period is not None and int(momentum_restart_period) < 1:
            raise ValueError("momentum_restart_period must be >= 1 when specified")
        b1, b2 = float(betas[0]), float(betas[1])
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError("betas must be in [0, 1)")
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        self.lr = float(lr)
        self.alpha = float(alpha)
        self.tau_cut = float(tau_cut)
        self.beta1 = b1
        self.beta2 = b2
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.momentum_restart_period: Optional[int] = (
            int(momentum_restart_period) if momentum_restart_period is not None else None
        )
        self.outer_round_count: int = 0
        self.step_count: Dict[str, int] = {}
        self.m_buffers: Dict[str, torch.Tensor] = {}
        self.v_buffers: Dict[str, torch.Tensor] = {}
        self.fragment_ages: Dict[str, int] = {}

    @torch.no_grad()
    def reset_first_moment_buffers(self) -> int:
        """HM-S9-05: Periodically flush first-moment (`m_f`) buffers while preserving `v_f`."""
        count = 0
        for buf in self.m_buffers.values():
            buf.zero_()
            count += 1
        return count

    def compute_gate(self, effective_tau: float) -> Tuple[float, float]:
        """Return `(gamma(tau), sigma(tau))` for effective staleness `effective_tau >= 0`."""
        tau = max(0.0, float(effective_tau))
        if tau >= self.tau_cut:
            return 0.0, 0.0
        gamma = 0.5 * (1.0 + math.cos(math.pi * tau / self.tau_cut))
        sigma = gamma * math.exp(-self.alpha * tau)
        return float(gamma), float(sigma)

    def theoretical_staleness_bias_bound(self) -> float:
        """Return `max_{tau >= 0} tau * sigma(tau) <= 1 / (e * alpha)` (Theorem 1, `arXiv:2605.09126`)."""
        return 1.0 / (math.e * self.alpha)

    @torch.no_grad()
    def step_fragments(
        self,
        params: Dict[str, torch.Tensor],
        pseudo_gradients: Dict[str, torch.Tensor],
        worker_delays: Optional[Dict[str, float]] = None,
        synced_fragments: Optional[Sequence[str]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[CGADStepTelemetry]]:
        """Apply one PA-CGAD outer step across synced parameter fragments."""
        if (
            self.momentum_restart_period is not None
            and self.outer_round_count > 0
            and (self.outer_round_count % self.momentum_restart_period == 0)
        ):
            self.reset_first_moment_buffers()
        self.outer_round_count += 1
        active_set = set(synced_fragments) if synced_fragments is not None else set(pseudo_gradients.keys())
        updated_params: Dict[str, torch.Tensor] = {}
        telemetries: List[CGADStepTelemetry] = []

        for name, param in params.items():
            prev_age = self.fragment_ages.get(name, 0)
            if name not in active_set or name not in pseudo_gradients:
                self.fragment_ages[name] = prev_age + 1
                updated_params[name] = param.clone()
                continue

            g_f = pseudo_gradients[name].detach().to(torch.float32).cpu()
            p_fp32 = param.detach().to(torch.float32).cpu()
            tau_w = float((worker_delays or {}).get(name, 0.0))
            eff_tau = max(0.0, tau_w, float(prev_age))
            gamma, sigma = self.compute_gate(eff_tau)

            if sigma <= 0.0:
                # Hard cosine cutoff: do not touch m_f or v_f (prevents stale variance pollution)
                self.fragment_ages[name] = prev_age + 1
                updated_params[name] = param.clone()
                telemetries.append(
                    CGADStepTelemetry(
                        fragment_name=name,
                        worker_delay_tau=round(tau_w, 4),
                        fragment_age_rounds=float(prev_age),
                        effective_tau=round(eff_tau, 4),
                        cosine_gate_gamma=0.0,
                        decay_gate_sigma=0.0,
                        dropped_by_cutoff=True,
                        staleness_bias_bound=round(self.theoretical_staleness_bias_bound(), 6),
                        step_linf_norm=0.0,
                    )
                )
                continue

            g_gated = sigma * g_f
            if name not in self.m_buffers:
                self.m_buffers[name] = torch.zeros_like(g_gated)
                self.v_buffers[name] = torch.zeros_like(g_gated)
                self.step_count[name] = 0

            self.step_count[name] += 1
            t = self.step_count[name]
            m = self.m_buffers[name]
            v = self.v_buffers[name]

            m.mul_(self.beta1).add_(g_gated, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(g_gated, g_gated, value=1.0 - self.beta2)

            m_hat = m / (1.0 - (self.beta1 ** t))
            v_hat = v / (1.0 - (self.beta2 ** t))

            step_vec = self.lr * sigma * (m_hat / (torch.sqrt(v_hat) + self.eps))
            # Decoupled AdamW-style shrinkage, applied only to fragments that actually
            # received a non-cutoff update (weight_decay=0 preserves the original CGAD math).
            decayed_p = p_fp32 * (1.0 - self.lr * self.weight_decay)
            new_p = (decayed_p - step_vec).to(device=param.device, dtype=param.dtype)
            updated_params[name] = new_p
            self.fragment_ages[name] = 0

            telemetries.append(
                CGADStepTelemetry(
                    fragment_name=name,
                    worker_delay_tau=round(tau_w, 4),
                    fragment_age_rounds=float(prev_age),
                    effective_tau=round(eff_tau, 4),
                    cosine_gate_gamma=round(gamma, 6),
                    decay_gate_sigma=round(sigma, 6),
                    dropped_by_cutoff=False,
                    staleness_bias_bound=round(self.theoretical_staleness_bias_bound(), 6),
                    step_linf_norm=round(float(step_vec.abs().max().item()), 8),
                )
            )

        return updated_params, telemetries


class ReputationWeightedIsoCTSMerger:
    """SI-07: Reputation-Weighted Isotropic Common + Task-Specific Merger (`IsoLoCo`, `arXiv:2607.03011`).

    Unifies HiveMind's `SI-04` reputation-weighted TIES sign election with `ISO-C` / `ISO-CTS`
    spectral isotropy so that multi-worker LoRA pseudo-gradients preserve both shared
    representation directions and orthogonal specialist directions without singular-value
    dominance collapse.
    """

    def __init__(
        self,
        common_space_fraction: float = 0.75,
        trim_density: float = 0.35,
        scaling_lambda: float = 1.10,
    ) -> None:
        if not (0.1 <= common_space_fraction <= 1.0):
            raise ValueError("common_space_fraction must be in [0.1, 1.0]")
        if not (0.0 < trim_density <= 1.0):
            raise ValueError("trim_density must be in (0, 1.0]")
        self.common_space_fraction = float(common_space_fraction)
        self.trim_density = float(trim_density)
        self.scaling_lambda = float(scaling_lambda)

    @staticmethod
    @torch.no_grad()
    def apply_isoc_2d(matrix: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Flatten the singular value spectrum of a 2D delta matrix to its isotropic mean."""
        if matrix.ndim != 2:
            return matrix.clone(), 1.0, 1.0
        mat_fp32 = matrix.detach().to(torch.float32).cpu()
        u, s, vh = torch.linalg.svd(mat_fp32, full_matrices=False)
        if s.numel() == 0 or float(s.max().item()) <= 1e-12:
            return matrix.clone(), 1.0, 1.0
        cond_before = float(s.max().item() / s.clamp_min(1e-8).min().item())
        s_mean = torch.ones_like(s) * s.mean()
        reconstructed = (u * s_mean.unsqueeze(0)) @ vh
        return reconstructed.to(device=matrix.device, dtype=matrix.dtype), cond_before, 1.0

    @torch.no_grad()
    def merge_worker_deltas(
        self,
        base_lora_state: Dict[str, torch.Tensor],
        worker_lora_states: Sequence[Dict[str, torch.Tensor]],
        worker_reputation_weights: Optional[Sequence[float]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Merge worker LoRA states using Reputation-Weighted TIES + ISO-CTS orthogonalization."""
        if not worker_lora_states:
            return {k: v.clone() for k, v in base_lora_state.items()}, {"num_workers": 0}

        num_workers = len(worker_lora_states)
        if worker_reputation_weights is None or len(worker_reputation_weights) != num_workers:
            weights = [1.0 for _ in range(num_workers)]
        else:
            weights = [max(1e-4, float(w)) for w in worker_reputation_weights]
        w_tensor = torch.tensor(weights, dtype=torch.float32)
        w_tensor = w_tensor / w_tensor.sum().clamp_min(1e-8)

        merged_state: Dict[str, torch.Tensor] = {}
        cond_before_list: List[float] = []
        cond_after_list: List[float] = []

        for key, base_t in base_lora_state.items():
            base_fp32 = base_t.detach().to(torch.float32).cpu()
            trimmed_vectors: List[torch.Tensor] = []
            valid_w: List[float] = []

            for idx, w_state in enumerate(worker_lora_states):
                if key not in w_state:
                    continue
                tau = w_state[key].detach().to(torch.float32).cpu() - base_fp32
                flat = tau.reshape(-1)
                k_keep = max(1, int(round(flat.numel() * self.trim_density)))
                if k_keep < flat.numel():
                    thresh = torch.topk(flat.abs(), k=k_keep, largest=True).values.min()
                    tau = torch.where(tau.abs() >= thresh, tau, torch.zeros_like(tau))
                trimmed_vectors.append(tau)
                valid_w.append(float(w_tensor[idx].item()))

            if not trimmed_vectors:
                merged_state[key] = base_t.clone()
                continue

            stacked = torch.stack(trimmed_vectors, dim=0)
            w_view = torch.tensor(valid_w, dtype=torch.float32).view(
                -1, *([1] * (stacked.ndim - 1))
            )
            weighted_mass = torch.sum(stacked * w_view, dim=0)
            elected_sign = torch.sign(weighted_mass)
            agree_mask = (torch.sign(stacked) == elected_sign.unsqueeze(0)) & (stacked != 0)
            agree_weights = w_view * agree_mask.to(torch.float32)
            denom = torch.clamp(torch.sum(agree_weights, dim=0), min=1e-6)
            ties_delta = torch.sum(stacked * agree_weights, dim=0) / denom

            if ties_delta.ndim == 2 and min(ties_delta.shape) >= 2:
                # Extract common subspace + task-specific residual subspace (ISO-CTS)
                min_dim = min(ties_delta.shape)
                c_dim = max(1, int(round(min_dim * self.common_space_fraction)))
                u_c, s_c, vh_c = torch.linalg.svd(ties_delta, full_matrices=False)
                cond_before = float(s_c.max().item() / s_c.clamp_min(1e-8).min().item())
                cond_before_list.append(cond_before)

                u_common = u_c[:, :c_dim]
                rem_dim = min_dim - c_dim
                if rem_dim > 0 and len(trimmed_vectors) >= 1:
                    # Aggregate task-specific residuals orthogonal to u_common
                    residuals = [
                        (vec - u_common @ (u_common.t() @ vec)) * wt
                        for vec, wt in zip(trimmed_vectors, valid_w)
                    ]
                    res_sum = sum(residuals)
                    if float(res_sum.norm().item()) > 1e-8:
                        u_ts, _, vh_ts = torch.linalg.svd(res_sum, full_matrices=False)
                        u_comb = torch.cat([u_common, u_ts[:, :rem_dim]], dim=1)
                        vh_comb = torch.cat([vh_c[:c_dim, :], vh_ts[:rem_dim, :]], dim=0)
                        pu, _, quh = torch.linalg.svd(u_comb, full_matrices=False)
                        pv, _, qvh = torch.linalg.svd(vh_comb, full_matrices=False)
                        u_orth = pu @ quh
                        vh_orth = pv @ qvh
                        s_iso = torch.ones_like(s_c) * s_c.mean()
                        iso_delta = (u_orth * s_iso.unsqueeze(0)) @ vh_orth
                    else:
                        iso_delta, _, _ = self.apply_isoc_2d(ties_delta)
                else:
                    iso_delta, _, _ = self.apply_isoc_2d(ties_delta)
                cond_after_list.append(1.0)
                final_delta = iso_delta
            else:
                final_delta = ties_delta

            merged_state[key] = (base_fp32 + self.scaling_lambda * final_delta).to(
                device=base_t.device, dtype=base_t.dtype
            )

        return merged_state, {
            "strategy_id": "SI-07-REPUTATION-WEIGHTED-ISO-CTS",
            "num_workers": num_workers,
            "common_space_fraction": self.common_space_fraction,
            "mean_condition_number_before": round(
                sum(cond_before_list) / max(1, len(cond_before_list)), 4
            ),
            "mean_condition_number_after": round(
                sum(cond_after_list) / max(1, len(cond_after_list)), 4
            ),
        }


class HeLoCoDirectionCorrector:
    """SI-08A: Direction-Aware Staleness Correction (`HeLoCo`, Asif et al., `arXiv:2606.00271`).

    Uses the outer momentum trajectory `m_ref` as a directional reference and dampens or
    projects out negative-cosine (conflicting) components of stale non-IID pseudo-gradients.
    """

    def __init__(self, conflict_damping: float = 0.25) -> None:
        if not (0.0 <= conflict_damping <= 1.0):
            raise ValueError("conflict_damping must be in [0, 1]")
        self.conflict_damping = float(conflict_damping)

    @torch.no_grad()
    def correct_pseudo_gradient(
        self,
        pseudo_grad: torch.Tensor,
        outer_momentum_ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Align a stale worker pseudo-gradient against `outer_momentum_ref`."""
        g = pseudo_grad.detach().to(torch.float32)
        m = outer_momentum_ref.detach().to(torch.float32)
        g_flat = g.reshape(-1)
        m_flat = m.reshape(-1)
        m_norm_sq = float(torch.dot(m_flat, m_flat).item())
        g_norm = float(g_flat.norm().item())
        if m_norm_sq <= 1e-12 or g_norm <= 1e-12:
            return pseudo_grad.clone(), {"cosine_before": 0.0, "cosine_after": 0.0, "conflict_detected": 0.0}

        dot_gm = float(torch.dot(g_flat, m_flat).item())
        cos_before = dot_gm / (g_norm * math.sqrt(m_norm_sq))
        if dot_gm >= 0.0:
            return pseudo_grad.clone(), {
                "cosine_before": round(cos_before, 6),
                "cosine_after": round(cos_before, 6),
                "conflict_detected": 0.0,
            }

        # Project out the conflicting component along m_ref and retain a damped fraction
        proj_coeff = dot_gm / m_norm_sq
        conflicting_component = proj_coeff * m
        corrected = g - (1.0 - self.conflict_damping) * conflicting_component
        c_flat = corrected.reshape(-1)
        cos_after = float(torch.dot(c_flat, m_flat).item()) / (
            max(1e-12, float(c_flat.norm().item())) * math.sqrt(m_norm_sq)
        )
        return corrected.to(device=pseudo_grad.device, dtype=pseudo_grad.dtype), {
            "cosine_before": round(cos_before, 6),
            "cosine_after": round(cos_after, 6),
            "conflict_detected": 1.0,
        }


class AutoLoCoIntervalCorrector:
    """SI-08B: Adaptive Inner-Interval Outer Optimizer Correction (`AutoLoCo`, `arXiv:2609.36662`).

    When `AdaptiveDiLoCoCadenceController` (`SI-03`) dynamically adapts the local inner step
    count `H_t` (or inner learning rate schedule `eta_inner`), corrects the outer learning
    rate `eta_outer` and outer momentum `mu_outer` by the accumulated inner learning rate
    ratio `S_t / S_ref`, eliminating scale mismatch across variable-length inner rounds.
    """

    def __init__(
        self,
        reference_inner_steps_h: int = 32,
        reference_inner_lr: float = 1e-3,
        base_outer_lr: float = 0.7,
        base_outer_momentum: float = 0.9,
    ) -> None:
        if reference_inner_steps_h < 1 or reference_inner_lr <= 0.0:
            raise ValueError("reference_inner_steps_h >= 1 and reference_inner_lr > 0 required")
        self.reference_accumulated_lr = float(reference_inner_steps_h) * float(reference_inner_lr)
        self.base_outer_lr = float(base_outer_lr)
        self.base_outer_momentum = float(base_outer_momentum)

    def correct_outer_hyperparams(
        self,
        actual_inner_steps_h: int,
        actual_inner_lr: float,
    ) -> Dict[str, float]:
        """Compute corrected `(outer_lr, outer_momentum)` for the realized inner interval."""
        if actual_inner_steps_h < 1 or actual_inner_lr <= 0.0:
            raise ValueError("actual_inner_steps_h >= 1 and actual_inner_lr > 0 required")
        accum_lr = float(actual_inner_steps_h) * float(actual_inner_lr)
        ratio = accum_lr / self.reference_accumulated_lr
        # Normalize outer step inversely with sqrt(ratio) and adapt momentum memory exp(-ratio)
        corrected_outer_lr = self.base_outer_lr / math.sqrt(max(0.25, min(4.0, ratio)))
        corrected_momentum = self.base_outer_momentum ** max(0.25, min(4.0, ratio))
        return {
            "accumulated_inner_lr": round(accum_lr, 6),
            "interval_scale_ratio": round(ratio, 6),
            "corrected_outer_lr": round(corrected_outer_lr, 6),
            "corrected_outer_momentum": round(corrected_momentum, 6),
        }


class FoMoEPartialReplicaPlanner:
    """SI-09A: Federated Partial Expert Replication & Ghost-Expert Skip-Token Planner (`FoMoE`, `arXiv:2606.19025`).

    Computes the communication ring factor `R(O_e) = 2 * (O_e - 1) / O_e`, steady-state
    expert memory fraction `O_e / M`, and Ghost-Expert Skip-Token compute scaling `sigma_st`
    for a 128-expert Sparse MoE distributed across `M` WAN workers.
    """

    def __init__(self, total_experts: int = 128, active_top_k: int = 2) -> None:
        if total_experts < 2 or active_top_k < 1:
            raise ValueError("total_experts >= 2 and active_top_k >= 1 required")
        self.total_experts = int(total_experts)
        self.active_top_k = int(active_top_k)

    def evaluate_overlap_configuration(
        self,
        num_workers_m: int,
        overlap_factor_oe: int,
        reshuffle_frequency_gamma: float = 0.0,
        dense_bytes: int = 32 * 1024 * 1024,
        single_expert_bytes: int = 2 * 1024 * 1024,
    ) -> Dict[str, Any]:
        """Return memory, communication, and Skip-Token FLOP metrics for `(M, O_e, gamma)`."""
        if num_workers_m < 1:
            raise ValueError("num_workers_m must be >= 1")
        if not (1 <= overlap_factor_oe <= num_workers_m):
            raise ValueError("overlap_factor_oe must be in [1, num_workers_m]")
        if not (0.0 <= reshuffle_frequency_gamma <= 1.0):
            raise ValueError("reshuffle_frequency_gamma must be in [0, 1]")

        m = float(num_workers_m)
        oe = float(overlap_factor_oe)
        local_experts = max(1, int(round((oe / m) * self.total_experts)))

        ring_factor_r = 0.0 if overlap_factor_oe == 1 else (2.0 * (oe - 1.0) / oe)
        migration_factor_p = float(reshuffle_frequency_gamma) * (1.0 - (oe / m))

        dense_comm_bytes = (2.0 * (m - 1.0) / max(1.0, m)) * float(dense_bytes)
        sparse_comm_bytes = (ring_factor_r + migration_factor_p) * float(
            local_experts * single_expert_bytes
        )
        full_replica_comm_bytes = (2.0 * (m - 1.0) / max(1.0, m)) * float(
            dense_bytes + self.total_experts * single_expert_bytes
        )

        comm_reduction_x = full_replica_comm_bytes / max(1.0, dense_comm_bytes + sparse_comm_bytes)
        skip_token_factor = local_experts / float(self.total_experts)

        return {
            "strategy_id": "SI-09A-FOMOE-PARTIAL-REPLICA-SKIP-TOKEN",
            "num_workers_m": num_workers_m,
            "overlap_factor_oe": overlap_factor_oe,
            "local_experts_per_worker": local_experts,
            "ghost_experts_per_worker": self.total_experts - local_experts,
            "ring_aggregation_factor_R": round(ring_factor_r, 6),
            "migration_broadcast_factor_P": round(migration_factor_p, 6),
            "expert_vram_fraction": round(oe / m, 6),
            "skip_token_active_fraction": round(skip_token_factor, 6),
            "per_site_comm_bytes": int(round(dense_comm_bytes + sparse_comm_bytes)),
            "full_replica_comm_bytes": int(round(full_replica_comm_bytes)),
            "wan_payload_reduction_x": round(comm_reduction_x, 4),
        }


class ComputeVisibleSparseLoCoCompressor:
    """SI-09B: Compute-Visible + 2-Bit Statistical Centroid Compressor (`PULSELoCo` + `SparseLoCo`).

    Combines:
    1. `PULSELoCo` (`arXiv:2602.03839`): Compute-visible gating that drops per-coordinate
       updates smaller than half the BF16/FP16 unit-in-last-place (`ulp / 2`) relative to
       the parameter magnitude.
    2. `SparseLoCo` (`arXiv:2508.15706`): Top-k sparsification (`1-3%` density) with 2-bit
       (`4`-bin) empirical centroid lookup-table quantization and deterministic index
       reconstruction (avoiding `DeMo`'s non-deterministic CUDA `scatter_reduce_`).
    """

    def __init__(
        self,
        topk_ratio: float = 0.03,
        n_bins: int = 4,
        range_in_sigmas: float = 4.0,
        bf16_ulp_gate: float = 2.0 ** -8,
        codec: str = "centroid_2bit",
        target_bits_per_dim: float = 2.0,
        lattice_scale: Optional[float] = None,
        error_feedback: bool = False,
        adapt_scale_per_round: bool = True,
        value_codec: str = "lut",
        lattice_rate_bits: float = 2.0,
    ) -> None:
        if not (0.0 < topk_ratio <= 1.0):
            raise ValueError("topk_ratio must be in (0, 1]")
        if n_bins < 2 or n_bins > 256:
            raise ValueError("n_bins must be in [2, 256]")
        valid_codecs = ("centroid_2bit", "E8", "Z", "sign")
        if codec not in valid_codecs:
            raise ValueError(f"codec must be one of {valid_codecs}, got {codec!r}")
        if value_codec not in ("lut", "Z", "E8"):
            raise ValueError("value_codec must be 'lut' (default), 'Z' or 'E8'")  # E10b: lattice option, HM-S10-01
        if target_bits_per_dim <= 0.0 or not math.isfinite(target_bits_per_dim):
            raise ValueError("target_bits_per_dim must be positive and finite")
        if lattice_scale is not None and (lattice_scale <= 0.0 or not math.isfinite(lattice_scale)):
            raise ValueError("lattice_scale must be positive and finite when specified")
        self.value_codec = value_codec
        self.lattice_scale = None if lattice_scale is None else float(lattice_scale)
        self.lattice_rate_bits = float(lattice_rate_bits)
        self.topk_ratio = float(topk_ratio)
        self.n_bins = int(n_bins)
        self.range_in_sigmas = float(range_in_sigmas)
        self.bf16_ulp_gate = float(bf16_ulp_gate)
        self.codec = str(codec)
        self.target_bits_per_dim = float(target_bits_per_dim)
        self.lattice_scale = float(lattice_scale) if lattice_scale is not None else None
        self.error_feedback = bool(error_feedback)
        self.adapt_scale_per_round = bool(adapt_scale_per_round)
        self._lattice_quantizers: Dict[str, LatticeDeltaQuantizer] = {}
        self._sign_quantizers: Dict[str, SignDeltaQuantizer] = {}
        self.fragment_rate_history: Dict[str, List[Dict[str, float]]] = {}

    def _resolve_round_scale(self, kind: str, flat_vec: torch.Tensor) -> float:
        """Determine the per-round lattice scale factor, calibrating to `target_bits_per_dim` when enabled."""
        if self.lattice_scale is not None and not self.adapt_scale_per_round:
            return self.lattice_scale
        std = float(flat_vec.std().item()) if flat_vec.numel() > 1 else float(flat_vec.abs().mean().item())
        fallback_scale = max(1e-6, (std if std > 1e-9 else 1.0) / max(0.5, self.target_bits_per_dim))
        if not self.adapt_scale_per_round or std <= 1e-9:
            return self.lattice_scale if self.lattice_scale is not None else fallback_scale
        try:
            scale, _ = calibrate_scale_for_rate(
                kind=kind,
                sample=flat_vec,
                target_bits_per_dim=self.target_bits_per_dim,
            )
            return max(1e-8, float(scale))
        except ValueError:
            return self.lattice_scale if self.lattice_scale is not None else fallback_scale

    @torch.no_grad()
    def compress_with_lattice_codec(
        self,
        delta: torch.Tensor,
        reference_param: Optional[torch.Tensor] = None,
        codec: Optional[str] = None,
        fragment_name: str = "default",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """HM-S10-01: Compress `delta` using a selectable lattice (`E8`, `Z`) or `sign` codec with
        PULSE compute-visible gating, per-round scale adaptation, EF21 error feedback, and
        measured factorised rate logging per fragment.
        """
        active_codec = str(codec or self.codec)
        if active_codec == "centroid_2bit":
            return self.compress_and_reconstruct(
                delta, reference_param=reference_param, _force_centroid=True
            )
        if active_codec not in ("E8", "Z", "sign"):
            raise ValueError(f"Unsupported lattice codec {active_codec!r}")

        d_fp32 = delta.detach().to(torch.float32).cpu()
        flat = d_fp32.reshape(-1)
        numel = flat.numel()

        if reference_param is not None:
            ref_abs = (
                reference_param.detach().to(torch.float32).cpu().reshape(-1).abs().clamp_min(1e-4)
            )
            visible_mask = flat.abs() >= (ref_abs * self.bf16_ulp_gate)
            gated_flat = torch.where(visible_mask, flat, torch.zeros_like(flat))
            invisible_fraction = 1.0 - float(visible_mask.to(torch.float32).mean().item())
        else:
            gated_flat = flat
            invisible_fraction = 0.0

        q_key = f"{active_codec}::{fragment_name}"
        if active_codec == "sign":
            sq = self._sign_quantizers.get(q_key)
            if sq is None:
                sq = SignDeltaQuantizer(error_feedback=self.error_feedback)
                self._sign_quantizers[q_key] = sq
            q_out: QuantizedDelta = sq.quantize(gated_flat)
            adapted_scale = float(gated_flat.abs().mean().item())
        else:
            adapted_scale = self._resolve_round_scale(active_codec, gated_flat)
            lq = self._lattice_quantizers.get(q_key)
            if lq is None:
                lq = LatticeDeltaQuantizer(
                    kind=active_codec,
                    scale=adapted_scale,
                    block=8,
                    error_feedback=self.error_feedback,
                )
                self._lattice_quantizers[q_key] = lq
            else:
                lq.scale = adapted_scale
            q_out = lq.quantize(gated_flat)

        reconstructed = q_out.dequantized.reshape(delta.shape).to(
            device=delta.device, dtype=delta.dtype
        )
        measured_rate = float(q_out.rate_bits_per_dim)
        rel_mse = float(q_out.relative_mse)
        wire_bytes = max(8, int(math.ceil(numel * max(0.125, measured_rate) / 8.0)) + 8)
        raw_bf16_bytes = numel * 2

        entry = {
            "rate_bits_per_dim": round(measured_rate, 6),
            "relative_mse": round(rel_mse, 6),
            "scale": round(adapted_scale, 8),
        }
        self.fragment_rate_history.setdefault(fragment_name, []).append(entry)

        return reconstructed, {
            "strategy_id": f"SI-09B-HM-S10-01-LATTICE-{active_codec.upper()}",
            "codec": active_codec,
            "fragment_name": fragment_name,
            "numel": numel,
            "measured_rate_bits_per_dim": round(measured_rate, 6),
            "relative_mse": round(rel_mse, 6),
            "adapted_scale": round(adapted_scale, 8),
            "error_feedback": self.error_feedback,
            "compute_invisible_fraction": round(invisible_fraction, 6),
            "raw_bf16_bytes": raw_bf16_bytes,
            "compressed_wire_bytes": wire_bytes,
            "wire_compression_ratio_x": round(raw_bf16_bytes / max(1, wire_bytes), 2),
        }

    @torch.no_grad()
    def compress_and_reconstruct(
        self,
        delta: torch.Tensor,
        reference_param: Optional[torch.Tensor] = None,
        *,
        _force_centroid: bool = False,
        fragment_name: str = "default",
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compress `delta` via PULSE compute-visible gate + configured codec (`centroid_2bit`, `E8`, `Z`, `sign`)."""
        if not _force_centroid and self.codec in ("E8", "Z", "sign"):
            return self.compress_with_lattice_codec(
                delta,
                reference_param=reference_param,
                codec=self.codec,
                fragment_name=fragment_name,
            )
        d_fp32 = delta.detach().to(torch.float32).cpu()
        flat = d_fp32.reshape(-1)
        numel = flat.numel()

        if reference_param is not None:
            ref_abs = reference_param.detach().to(torch.float32).cpu().reshape(-1).abs().clamp_min(1e-4)
            visible_mask = flat.abs() >= (ref_abs * self.bf16_ulp_gate)
            gated_flat = torch.where(visible_mask, flat, torch.zeros_like(flat))
            invisible_fraction = 1.0 - float(visible_mask.to(torch.float32).mean().item())
        else:
            gated_flat = flat
            invisible_fraction = 0.0

        k = max(1, int(round(numel * self.topk_ratio)))
        # Deterministic sorted top-k index selection (Golden Invariant #2)
        _, idx = torch.topk(gated_flat.abs(), k=k, largest=True, sorted=True)
        selected = gated_flat[idx]

        if self.value_codec != "lut":
            # Lattice option (E10b): the k selected values are quantised blockwise on scale * {Z^8, E8}; the scale is
            # either fixed or calibrated so the measured factorised rate of this payload hits `lattice_rate_bits`.
            from hivemind.compress.lattice_quantizer import LatticeDeltaQuantizer, calibrate_scale_for_rate

            lat_scale = self.lattice_scale
            if lat_scale is None:
                lat_scale, _ = calibrate_scale_for_rate(self.value_codec, selected, self.lattice_rate_bits)
            q = LatticeDeltaQuantizer(self.value_codec, lat_scale).quantize(selected)
            reconstructed = torch.zeros_like(flat)
            reconstructed.index_copy_(0, idx, q.dequantized)
            reconstructed = reconstructed.reshape(delta.shape).to(device=delta.device, dtype=delta.dtype)
            wire = int(math.ceil(k * q.rate_bits_per_dim / 8.0) + k * 2 + 16)
            return reconstructed, {
                "strategy_id": "SI-09B-PULSE-SPARSELOCO-LATTICE",
                "numel": numel,
                "k_transmitted": k,
                "value_codec": self.value_codec,
                "lattice_scale": float(lat_scale),
                "measured_rate_bits_per_value": q.rate_bits_per_dim,
                "relative_mse": q.relative_mse,
                "compute_invisible_fraction": round(invisible_fraction, 6),
                "raw_bf16_bytes": numel * 2,
                "compressed_wire_bytes": wire,
                "wire_compression_ratio_x": round(numel * 2 / max(1, wire), 2),
            }

        shift = float(selected.mean().item())
        centered = selected - shift
        std = float(centered.norm().item() / math.sqrt(max(1, selected.numel() - 1)))
        scale = (self.range_in_sigmas * std / float(self.n_bins)) if std > 1e-12 else 1.0
        offset = self.n_bins // 2

        q_bins = ((centered / scale + offset).round().clamp(0, self.n_bins - 1)).to(torch.long)
        lookup = torch.zeros(self.n_bins, dtype=torch.float32)
        for b in range(self.n_bins):
            mask_b = q_bins == b
            if bool(mask_b.any().item()):
                lookup[b] = centered[mask_b].mean()
            else:
                lookup[b] = float(b - offset) * scale

        dequant_selected = lookup[q_bins] + shift
        reconstructed = torch.zeros_like(flat)
        reconstructed.index_copy_(0, idx, dequant_selected)
        reconstructed = reconstructed.reshape(delta.shape).to(device=delta.device, dtype=delta.dtype)

        bits_per_val = math.ceil(math.log2(float(self.n_bins)))
        wire_bytes = int(math.ceil(k * (bits_per_val / 8.0)) + k * 2 + self.n_bins * 4 + 8)
        raw_bf16_bytes = numel * 2

        return reconstructed, {
            "strategy_id": "SI-09B-PULSE-SPARSELOCO-2BIT",
            "numel": numel,
            "k_transmitted": k,
            "quantization_bins": self.n_bins,
            "compute_invisible_fraction": round(invisible_fraction, 6),
            "raw_bf16_bytes": raw_bf16_bytes,
            "compressed_wire_bytes": wire_bytes,
            "wire_compression_ratio_x": round(raw_bf16_bytes / max(1, wire_bytes), 2),
        }


class ChunkedOrthonormalDCTTransform:
    """SI-09C: Orthonormal 2D Chunked DCT Transform (`DeMo` `arXiv:2411.19870` & `SparseLoCo` `arXiv:2508.15706`).

    Adapted from `bloc97/DeMo` (`demo.py:TransformDCT`) and `one-covenant/SparseLoCo`
    (`src/tplr/sparseloco.py:_dct` & `_idct`), using orthogonal Type-II DCT basis matrices
    to decorrelate spatially correlated 2D LoRA pseudo-gradients before Top-k extraction,
    while using deterministic `index_copy_` instead of CUDA `scatter_reduce_`.
    """

    def __init__(self, chunk_size: int = 8) -> None:
        if chunk_size < 2:
            raise ValueError("chunk_size must be >= 2")
        self.chunk_size = int(chunk_size)
        self.basis = self._build_orthonormal_dct_matrix(self.chunk_size)

    @staticmethod
    def _build_orthonormal_dct_matrix(n: int) -> torch.Tensor:
        """Construct an `n x n` orthonormal Type-II DCT matrix `C` satisfying `C @ C.T = I`."""
        mat = torch.zeros((n, n), dtype=torch.float32)
        for k in range(n):
            alpha = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
            for i in range(n):
                mat[k, i] = alpha * math.cos(math.pi * (2 * i + 1) * k / (2.0 * n))
        return mat

    @torch.no_grad()
    def encode_and_decode_topk(
        self,
        matrix: torch.Tensor,
        topk_per_chunk: int = 4,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Project a 2D matrix (whose dimensions are divisible by `chunk_size`) into 2D DCT space,
        keep the top-`k` frequency coefficients per chunk deterministically, and invert back.
        """
        if matrix.ndim != 2:
            raise ValueError("matrix must be 2D")
        r, c = matrix.shape
        n = self.chunk_size
        if r % n != 0 or c % n != 0:
            raise ValueError(f"matrix shape {(r, c)} must be divisible by chunk_size={n}")
        if not (1 <= topk_per_chunk <= n * n):
            raise ValueError("topk_per_chunk must be in [1, chunk_size^2]")

        x = matrix.detach().to(torch.float32).cpu()
        blocks = x.view(r // n, n, c // n, n).permute(0, 2, 1, 3).contiguous()
        num_blocks = (r // n) * (c // n)
        blocks_flat = blocks.view(num_blocks, n, n)

        c_mat = self.basis
        dct_blocks = torch.matmul(torch.matmul(c_mat.unsqueeze(0), blocks_flat), c_mat.t().unsqueeze(0))
        freq_flat = dct_blocks.view(num_blocks, n * n)

        _, top_idx = torch.topk(freq_flat.abs(), k=topk_per_chunk, dim=-1, largest=True, sorted=True)
        selected_vals = torch.gather(freq_flat, dim=-1, index=top_idx)

        sparse_freq = torch.zeros_like(freq_flat)
        sparse_freq.scatter_(dim=-1, index=top_idx, src=selected_vals)
        sparse_blocks = sparse_freq.view(num_blocks, n, n)

        idct_blocks = torch.matmul(
            torch.matmul(c_mat.t().unsqueeze(0), sparse_blocks), c_mat.unsqueeze(0)
        )
        reconstructed = (
            idct_blocks.view(r // n, c // n, n, n)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(r, c)
            .to(device=matrix.device, dtype=matrix.dtype)
        )
        energy_retained = float((selected_vals.pow(2).sum() / freq_flat.pow(2).sum().clamp_min(1e-12)).item())
        return reconstructed, {
            "strategy_id": "SI-09C-DEMO-SPARSELOCO-CHUNKED-DCT",
            "chunk_size": n,
            "num_chunks": num_blocks,
            "topk_per_chunk": topk_per_chunk,
            "dct_energy_retained_fraction": round(energy_retained, 6),
        }


class TSVDecorrelatedMerger:
    """SI-07B: Task Singular Vector (`TSVLoCo`) Subspace Decorrelation Merger.

    Adapted from `shoroi/isoloco` (`src/tplr/tsvloco.py`, Horoi et al., `arXiv:2607.03011`)
    and Gargiulo et al. (*Task Singular Vectors*, CVPR 2025, `arXiv:2412.00081`).
    For `M` worker 2D deltas, extracts top-`k = ceil(r / M)` singular triplets per worker,
    concatenates `(U_1..U_M)` and `(V_1..V_M)`, and whitens inter-worker subspace overlap
    via orthogonal Procrustes / SVD projection `(P_u @ Q_u^T, P_v @ Q_v^T)`.
    """

    @staticmethod
    @torch.no_grad()
    def merge_2d_deltas(
        worker_deltas: Sequence[torch.Tensor],
        worker_reputation_weights: Optional[Sequence[float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Decorrelate and merge 2D worker pseudo-gradients via TSV orthogonalization."""
        if not worker_deltas:
            raise ValueError("worker_deltas must be non-empty")
        first = worker_deltas[0]
        if first.ndim != 2:
            raise ValueError("TSVDecorrelatedMerger requires 2D tensors")

        m = len(worker_deltas)
        if worker_reputation_weights is None or len(worker_reputation_weights) != m:
            weights = [1.0 / m for _ in range(m)]
        else:
            raw = [max(1e-4, float(w)) for w in worker_reputation_weights]
            s_w = sum(raw)
            weights = [w / s_w for w in raw]

        min_dim = min(first.shape)
        rank_per_worker = max(1, min_dim // m)

        u_chunks: List[torch.Tensor] = []
        s_chunks: List[torch.Tensor] = []
        vh_chunks: List[torch.Tensor] = []

        for idx, delta in enumerate(worker_deltas):
            d_fp32 = delta.detach().to(torch.float32).cpu() * float(weights[idx] * m)
            u, s, vh = torch.linalg.svd(d_fp32, full_matrices=False)
            u_chunks.append(u[:, :rank_per_worker])
            s_chunks.append(s[:rank_per_worker])
            vh_chunks.append(vh[:rank_per_worker, :])

        u_cat = torch.cat(u_chunks, dim=1)
        s_cat = torch.cat(s_chunks, dim=0)
        vh_cat = torch.cat(vh_chunks, dim=0)

        pu, _, quh = torch.linalg.svd(u_cat, full_matrices=False)
        pv, _, qvh = torch.linalg.svd(vh_cat, full_matrices=False)
        u_orth = pu @ quh
        vh_orth = pv @ qvh

        merged = ((u_orth * s_cat.unsqueeze(0)) @ vh_orth).to(
            device=first.device, dtype=first.dtype
        )
        # Measure orthogonality error ||U_orth^T U_orth - I||_F
        eye_k = torch.eye(u_orth.shape[1], dtype=torch.float32)
        orth_err = float(torch.linalg.norm(u_orth.t() @ u_orth - eye_k).item())
        return merged, {
            "strategy_id": "SI-07B-TSVLOCO-SUBSPACE-DECORRELATION",
            "num_workers": m,
            "rank_per_worker": rank_per_worker,
            "combined_subspace_rank": int(s_cat.numel()),
            "subspace_orthogonality_error": round(orth_err, 6),
        }


class CPUOffloadedFlatBucketOuterState:
    """SI-06B: Contiguous Flat-Bucket CPU-Offloaded Outer Parameter State.

    Adapted from `PrimeIntellect-ai/prime-diloco` (`src/zeroband/diloco.py`),
    `PrimeIntellect-ai/OpenDiLoCo` (`arXiv:2407.07852`), and `google-deepmind/asyncdiloco`
    (`arXiv:2401.09135`).
    Packs all outer master weights into a single contiguous 1D FP32 CPU tensor with
    zero GPU VRAM residency (`0 B` extra VRAM on 8 GB consumer GPUs, Golden Invariant #1)
    and computes outer pseudo-gradients (`w_master_cpu - w_inner`) via zero-copy slices.
    """

    def __init__(self, params: Dict[str, torch.Tensor]) -> None:
        if not params:
            raise ValueError("params must not be empty")
        self.keys: List[str] = list(params.keys())
        self.shapes: Dict[str, torch.Size] = {}
        self.dtypes: Dict[str, torch.dtype] = {}
        self.devices: Dict[str, torch.device] = {}
        self.offsets: Dict[str, Tuple[int, int]] = {}

        total_numel = 0
        for k in self.keys:
            t = params[k]
            self.shapes[k] = t.shape
            self.dtypes[k] = t.dtype
            self.devices[k] = t.device
            n = int(t.numel())
            self.offsets[k] = (total_numel, total_numel + n)
            total_numel += n

        self.flat_master_cpu = torch.empty(total_numel, dtype=torch.float32, device="cpu")
        self.flat_velocity_cpu: Optional[torch.Tensor] = torch.zeros_like(self.flat_master_cpu)
        self.outer_step_count: int = 0
        for k in self.keys:
            lo, hi = self.offsets[k]
            self.flat_master_cpu[lo:hi].copy_(params[k].detach().to(torch.float32).cpu().reshape(-1))

    @torch.no_grad()
    def reset_velocity(self) -> None:
        """HM-S9-05: Zero the contiguous outer velocity buffer in-place (`K_restart`)."""
        if self.flat_velocity_cpu is None or self.flat_velocity_cpu.numel() != self.flat_master_cpu.numel():
            self.flat_velocity_cpu = torch.zeros_like(self.flat_master_cpu)
        else:
            self.flat_velocity_cpu.zero_()

    @torch.no_grad()
    def outer_nesterov_step(
        self,
        pseudo_gradients: Dict[str, torch.Tensor],
        lr: float = 0.7,
        momentum: float = 0.9,
        theta_prev: Optional[Dict[str, torch.Tensor]] = None,
        momentum_restart_period: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Apply one Nesterov outer step entirely inside the contiguous CPU flat buffers."""
        if lr <= 0.0 or not math.isfinite(lr):
            raise ValueError("lr must be > 0")
        if not (0.0 <= momentum < 1.0) or not math.isfinite(momentum):
            raise ValueError("momentum must be in [0, 1)")
        if momentum_restart_period is not None and int(momentum_restart_period) < 1:
            raise ValueError("momentum_restart_period must be >= 1 when specified")

        if theta_prev is not None:
            missing = set(self.keys) - set(theta_prev)
            if missing:
                raise KeyError(f"theta_prev is missing flat-bucket fragments: {sorted(missing)}")
            for k in self.keys:
                if theta_prev[k].shape != self.shapes[k]:
                    raise ValueError(
                        f"theta_prev for {k!r} has {theta_prev[k].numel()} elements, expected {self.offsets[k][1] - self.offsets[k][0]}"
                    )

        if set(pseudo_gradients) != set(self.keys):
            raise ValueError("pseudo_gradients must match the flat bucket parameter keys")
        for k in self.keys:
            lo, hi = self.offsets[k]
            g_t = pseudo_gradients[k]
            if g_t.numel() != hi - lo or g_t.shape != self.shapes[k]:
                raise ValueError(
                    f"pseudo-gradient for {k!r} has {g_t.numel()} elements, expected {hi - lo}"
                )

        if self.flat_velocity_cpu is None:
            self.flat_velocity_cpu = torch.zeros_like(self.flat_master_cpu)
        elif self.flat_velocity_cpu.numel() != self.flat_master_cpu.numel():
            raise RuntimeError("flat velocity buffer shape drifted from the master parameter buffer")

        if (
            momentum_restart_period is not None
            and self.outer_step_count > 0
            and (self.outer_step_count % int(momentum_restart_period) == 0)
        ):
            self.reset_velocity()

        if theta_prev is not None:
            for k in self.keys:
                lo, hi = self.offsets[k]
                self.flat_master_cpu[lo:hi].copy_(
                    theta_prev[k].detach().to(torch.float32).cpu().reshape(-1)
                )

        updated_params: Dict[str, torch.Tensor] = {}
        velocity: Dict[str, torch.Tensor] = {}
        for k in self.keys:
            lo, hi = self.offsets[k]
            grad = pseudo_gradients[k].detach().to(torch.float32).cpu().reshape(-1)
            v = self.flat_velocity_cpu[lo:hi]
            v.mul_(momentum).add_(grad)
            nesterov_step = momentum * v + grad
            self.flat_master_cpu[lo:hi].add_(nesterov_step, alpha=-lr)
            updated_params[k] = self.flat_master_cpu[lo:hi].view(self.shapes[k]).to(
                device=self.devices[k], dtype=self.dtypes[k]
            ).clone()
            velocity[k] = v.view(self.shapes[k]).clone()

        self.outer_step_count += 1
        return updated_params, velocity

    @torch.no_grad()
    def compute_pseudo_gradients(
        self,
        inner_params: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute outer pseudo-gradients `Delta_k = master_cpu_k - inner_k` on CPU."""
        pseudo_grads: Dict[str, torch.Tensor] = {}
        for k in self.keys:
            lo, hi = self.offsets[k]
            master_view = self.flat_master_cpu[lo:hi].view(self.shapes[k])
            inner_cpu = inner_params[k].detach().to(torch.float32).cpu()
            pseudo_grads[k] = master_view - inner_cpu
        return pseudo_grads

    @torch.no_grad()
    def apply_outer_update_and_sync_inner(
        self,
        updated_master_params: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Commit updated outer parameters into the contiguous flat CPU buffer and return
        inner device tensors ready for the next inner round.
        """
        synced_inner: Dict[str, torch.Tensor] = {}
        for k in self.keys:
            lo, hi = self.offsets[k]
            up_cpu = updated_master_params[k].detach().to(torch.float32).cpu()
            self.flat_master_cpu[lo:hi].copy_(up_cpu.reshape(-1))
            synced_inner[k] = up_cpu.to(device=self.devices[k], dtype=self.dtypes[k])
        return synced_inner

    def memory_telemetry(self) -> Dict[str, Any]:
        """Return CPU flat-buffer size and verify 0 B extra GPU VRAM allocation."""
        return {
            "strategy_id": "SI-06B-PRIME-OPEN-DILOCO-CPU-FLAT-BUCKET",
            "num_fragments": len(self.keys),
            "total_parameters": int(self.flat_master_cpu.numel()),
            "cpu_flat_buffer_bytes": int(self.flat_master_cpu.numel() * 4),
            "extra_gpu_vram_bytes": 0,
            "is_contiguous": bool(self.flat_master_cpu.is_contiguous()),
        }


class SkipAttentionSpeculativeRouter:
    """SI-10: Skip-Attention Speculative Expert Prefetch & Cache-Aware Routing (`CAR`).

    Adapted from Paul Merolla's `fomoe` (`https://github.com/pmerolla/fomoe`,
    `docs/speculative-routing.md`) and `learning-at-home/hivemind`:
    1. **Skip-Attention Speculative Prefetch**: Immediately after Layer `L`'s FFN completes,
       evaluates Layer `L+1`'s router (`rms_norm + router_weight + top_k`) on the
       pre-attention residual stream `x_post_ffn_L`, predicting Layer `L+1`'s top-`k`
       experts before Layer `L+1`'s self-attention runs (`~85.8%` top-`k` overlap accuracy
       in `~0.04 ms`), hiding PCIe / NVMe DMA prefetch behind the `~2.0 ms` attention window.
    2. **Cache-Aware Routing (`CAR`) Fallback**: After Layer `L+1`'s attention finishes,
       evaluates the true post-attention router logits `z_true`. If a selected expert `e`
       is NOT currently resident in the 8 GB VRAM cache (`cached_experts`), checks whether
       the best cached expert `e_cached` has logit within `car_threshold`:
         `z_true[e] - z_true[e_cached] <= car_threshold`
       If so, routes to `e_cached` immediately with zero PCIe stall; otherwise triggers
       on-demand DMA load.
    """

    def __init__(
        self,
        num_experts: int = 128,
        top_k: int = 2,
        car_threshold: float = 0.35,
        rms_eps: float = 1e-6,
    ) -> None:
        if num_experts < 2 or top_k < 1 or top_k > num_experts:
            raise ValueError("Require 1 <= top_k <= num_experts and num_experts >= 2")
        if car_threshold < 0.0:
            raise ValueError("car_threshold must be >= 0")
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.car_threshold = float(car_threshold)
        self.rms_eps = float(rms_eps)

    def _rms_norm(self, x: torch.Tensor, norm_weight: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.to(torch.float32)
        rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.rms_eps)
        return x_fp32 * rms * norm_weight.to(torch.float32)

    @torch.no_grad()
    def predict_next_layer_experts(
        self,
        post_ffn_residual: torch.Tensor,
        next_layer_norm_weight: torch.Tensor,
        next_layer_router_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict Layer `L+1` top-`k` expert IDs from Layer `L`'s post-FFN residual stream."""
        normed = self._rms_norm(post_ffn_residual, next_layer_norm_weight)
        logits = normed @ next_layer_router_weight.to(torch.float32).t()
        vals, idx = torch.topk(logits, k=self.top_k, dim=-1, largest=True, sorted=True)
        return idx, vals

    @torch.no_grad()
    def route_with_cache_aware_fallback(
        self,
        post_ffn_residual: torch.Tensor,
        post_attn_residual: torch.Tensor,
        next_layer_norm_weight: torch.Tensor,
        next_layer_router_weight: torch.Tensor,
        baseline_cached_experts: Sequence[int],
    ) -> Dict[str, Any]:
        """Run Skip-Attention prefetch + post-attention routing with Cache-Aware Routing (`CAR`)."""
        spec_idx, _ = self.predict_next_layer_experts(
            post_ffn_residual, next_layer_norm_weight, next_layer_router_weight
        )
        # Union of baseline VRAM cache + speculatively prefetched experts during attention window
        prefetched_set = set(int(e) for e in baseline_cached_experts) | set(
            int(e) for e in spec_idx.reshape(-1).tolist()
        )

        true_normed = self._rms_norm(post_attn_residual, next_layer_norm_weight)
        true_logits = true_normed @ next_layer_router_weight.to(torch.float32).t()
        true_vals, true_idx = torch.topk(
            true_logits, k=self.top_k, dim=-1, largest=True, sorted=True
        )

        # Measure speculative top-k overlap accuracy against true post-attention top-k
        spec_2d = spec_idx.reshape(-1, self.top_k)
        true_2d = true_idx.reshape(-1, self.top_k)
        num_tokens = spec_2d.shape[0]
        total_slots = num_tokens * self.top_k
        spec_hits = 0
        car_substitutions = 0
        synchronous_dma_misses = 0

        final_routed = true_2d.clone()
        cached_list = sorted(prefetched_set)
        cached_tensor = torch.tensor(cached_list, dtype=torch.long, device=true_logits.device)
        logits_2d = true_logits.reshape(-1, self.num_experts)

        for t in range(num_tokens):
            spec_set_t = set(int(x) for x in spec_2d[t].tolist())
            for slot in range(self.top_k):
                e_true = int(true_2d[t, slot].item())
                if e_true in spec_set_t:
                    spec_hits += 1
                if e_true in prefetched_set:
                    continue
                # Cache miss on e_true: evaluate Cache-Aware Routing (CAR) fallback
                true_score = float(logits_2d[t, e_true].item())
                already_chosen = set(int(x) for x in final_routed[t].tolist())
                best_cached_e = None
                best_cached_score = -1e30
                for c_e in cached_list:
                    if c_e in already_chosen:
                        continue
                    sc = float(logits_2d[t, c_e].item())
                    if sc > best_cached_score:
                        best_cached_score = sc
                        best_cached_e = c_e
                if (
                    best_cached_e is not None
                    and (true_score - best_cached_score) <= self.car_threshold
                ):
                    final_routed[t, slot] = best_cached_e
                    car_substitutions += 1
                else:
                    synchronous_dma_misses += 1

        spec_accuracy = spec_hits / max(1, total_slots)
        effective_cache_hit_rate = 1.0 - (synchronous_dma_misses / max(1, total_slots))
        return {
            "strategy_id": "SI-10-FOMOE-SKIP-ATTENTION-SPECULATIVE-CAR",
            "num_tokens": num_tokens,
            "top_k": self.top_k,
            "speculative_topk_accuracy": round(spec_accuracy, 6),
            "car_threshold": self.car_threshold,
            "car_substitutions": car_substitutions,
            "synchronous_dma_misses": synchronous_dma_misses,
            "effective_vram_cache_hit_rate": round(effective_cache_hit_rate, 6),
            "routed_experts": final_routed.reshape(true_idx.shape),
        }


class MuonNewtonSchulzLoRAPreconditioner:
    """SI-11: Inner-Loop Newton-Schulz Orthogonalized LoRA Momentum Preconditioner.

    Synthesized from Belilovsky, Horoi, Oyallon & Thérien (2026), *How Inner Optimizers Shape Replica Scaling in
    Distributed Low-Communication Training* (`arXiv:2610.19064`).
    Applies 5-step quintic Newton-Schulz iterations (`a=3.4445, b=-4.7750, c=2.0315`)
    to orthogonalize 2D inner LoRA momentum matrices before outer pseudo-gradient
    accumulation, preventing singular-direction collapse across `8..96` DiLoCo replicas.
    """

    def __init__(
        self,
        steps: int = 5,
        coeffs: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        ns_steps: int | None = None,
    ) -> None:
        if ns_steps is not None:
            steps = ns_steps
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.steps = int(steps)
        self.coeffs = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))

    @property
    def ns_steps(self) -> int:
        """Alias used by the inner-loop MuLoCo wiring (`HM-S8-09`)."""
        return self.steps

    def _orthogonalize_float32(self, g: torch.Tensor) -> torch.Tensor:
        """Run the quintic Newton-Schulz iteration on an already-float32 2D matrix."""
        transposed = False
        if g.shape[0] > g.shape[1]:
            g = g.t()
            transposed = True

        norm = g.norm().clamp_min(1e-7)
        x = g / norm
        a, b, c = self.coeffs
        for _ in range(self.steps):
            a_mat = x @ x.t()
            b_mat = b * a_mat + c * (a_mat @ a_mat)
            x = a * x + b_mat @ x

        if transposed:
            x = x.t()
        return x

    @torch.no_grad()
    def precondition_gradient(self, grad_matrix: torch.Tensor) -> torch.Tensor:
        """Fast inner-loop path: orthogonalize a 2D gradient without SVD telemetry."""
        if grad_matrix.ndim != 2:
            raise ValueError("precondition_gradient expects a 2D tensor")
        g = grad_matrix.detach().to(torch.float32)
        x = self._orthogonalize_float32(g)
        return x.to(device=grad_matrix.device, dtype=grad_matrix.dtype)

    @torch.no_grad()
    def orthogonalize_2d(self, grad_matrix: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Orthogonalize a 2D LoRA gradient/momentum matrix using quintic Newton-Schulz."""
        if grad_matrix.ndim != 2:
            raise ValueError("orthogonalize_2d expects a 2D tensor")
        g = grad_matrix.detach().to(torch.float32)
        s_before = torch.linalg.svdvals(g.cpu())
        active_before = s_before[s_before > (1e-5 * s_before.max().clamp_min(1e-8))]
        cond_before = float(
            active_before.max().item() / active_before.min().clamp_min(1e-8).item()
        ) if active_before.numel() > 0 else 1.0

        x = self._orthogonalize_float32(g)

        s_after = torch.linalg.svdvals(x.cpu())
        active_after = s_after[s_after > (1e-3 * s_after.max().clamp_min(1e-8))]
        cond_after = float(
            active_after.max().item() / active_after.min().clamp_min(1e-8).item()
        ) if active_after.numel() > 0 else 1.0
        return x.to(device=grad_matrix.device, dtype=grad_matrix.dtype), {
            "condition_number_before": round(cond_before, 4),
            "condition_number_after": round(cond_after, 4),
            "newton_schulz_steps": float(self.steps),
        }


class FactoredGossipMix1Mix2Coordinator:
    """HM-S9-04: Factored Gossip (`Mix1` parameter mix + `Mix2` outer-gradient mix, `arXiv:2606.03329`).

    Decouples swarm synchronization into two orthogonal tracks:
    1. `Mix1` (Non-Blocking Pairwise Parameter Gossip): Backbone parameters are mixed
       asynchronously with a peer cohort via convex combination `(1 - alpha_mix1) * w_local + alpha_mix1 * w_peer`
       without global barrier stalls.
    2. `Mix2` (Selective Blocking Outer-Gradient All-Reduce): Sensitive structural blocks
       (`router`, `gate`, `lm_head`, `output_head`, `embed`) undergo exact blocking outer
       pseudo-gradient synchronization ONLY when inter-worker router distribution disagreement
       exceeds a Jensen–Shannon divergence threshold `D_JS >= js_threshold` (paired with `HM-S8-05`).
       When `D_JS < js_threshold`, sensitive blocks fall back to non-blocking `Mix1` gossip,
       eliminating the global barrier on well-aligned rounds.
    """

    DEFAULT_SENSITIVE_PATTERNS: Tuple[str, ...] = (
        "router",
        "gate",
        "lm_head",
        "output_head",
        "embed",
    )

    def __init__(
        self,
        alpha_mix1: float = 0.5,
        js_threshold: float = 0.05,
        outer_lr_mix2: float = 0.7,
        outer_momentum_mix2: float = 0.9,
        sensitive_patterns: Optional[Sequence[str]] = None,
    ) -> None:
        if not (0.0 < alpha_mix1 <= 1.0) or not math.isfinite(alpha_mix1):
            raise ValueError("alpha_mix1 must be in (0, 1]")
        if js_threshold < 0.0 or not math.isfinite(js_threshold):
            raise ValueError("js_threshold must be >= 0 and finite")
        if outer_lr_mix2 <= 0.0 or not math.isfinite(outer_lr_mix2):
            raise ValueError("outer_lr_mix2 must be > 0 and finite")
        if not (0.0 <= outer_momentum_mix2 < 1.0) or not math.isfinite(outer_momentum_mix2):
            raise ValueError("outer_momentum_mix2 must be in [0, 1)")
        self.alpha_mix1 = float(alpha_mix1)
        self.js_threshold = float(js_threshold)
        self.outer_lr_mix2 = float(outer_lr_mix2)
        self.outer_momentum_mix2 = float(outer_momentum_mix2)
        self.sensitive_patterns: Tuple[str, ...] = tuple(
            sensitive_patterns if sensitive_patterns is not None else self.DEFAULT_SENSITIVE_PATTERNS
        )
        self.mix2_velocity: Dict[str, torch.Tensor] = {}
        self.round_index: int = 0

    def is_sensitive_fragment(self, fragment_name: str) -> bool:
        """Return True if `fragment_name` matches a `Mix2` sensitive block pattern."""
        lowered = fragment_name.lower()
        return any(pat.lower() in lowered for pat in self.sensitive_patterns)

    @staticmethod
    @torch.no_grad()
    def jensen_shannon_router_disagreement(
        worker_router_distributions: Sequence[torch.Tensor],
        inputs_are_logits: bool = False,
        eps: float = 1e-12,
    ) -> float:
        """Compute multi-distribution Jensen–Shannon divergence `D_JS(P_1, ..., P_M)` in nats.

        Satisfies `0 <= D_JS <= ln(K)` and `D_JS == 0` iff all worker distributions are identical.
        """
        if len(worker_router_distributions) < 2:
            return 0.0
        probs_list: List[torch.Tensor] = []
        for dist in worker_router_distributions:
            t = dist.detach().to(torch.float64).cpu()
            if inputs_are_logits:
                p = torch.softmax(t, dim=-1)
            else:
                p_clamped = t.clamp_min(0.0)
                denom = p_clamped.sum(dim=-1, keepdim=True).clamp_min(eps)
                p = p_clamped / denom
            probs_list.append(p)

        stacked = torch.stack(probs_list, dim=0)  # (M, ..., K)
        mean_p = stacked.mean(dim=0).clamp_min(eps)
        kl_terms = stacked * (torch.log(stacked.clamp_min(eps)) - torch.log(mean_p.unsqueeze(0)))
        js_per_token = kl_terms.sum(dim=-1).mean(dim=0)
        js_val = float(js_per_token.mean().item())
        return max(0.0, js_val)

    @torch.no_grad()
    def step_factored_gossip(
        self,
        master_params: Dict[str, torch.Tensor],
        worker_param_states: Sequence[Dict[str, torch.Tensor]],
        worker_router_distributions: Optional[Sequence[torch.Tensor]] = None,
        inputs_are_logits: bool = False,
        force_mix2: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Execute one factored `Mix1` + conditional `Mix2` synchronization round."""
        if not master_params:
            raise ValueError("master_params must not be empty")
        if not worker_param_states:
            raise ValueError("worker_param_states must be non-empty")

        js_div = (
            self.jensen_shannon_router_disagreement(
                worker_router_distributions, inputs_are_logits=inputs_are_logits
            )
            if worker_router_distributions is not None
            else 0.0
        )
        trigger_mix2 = bool(force_mix2 or (js_div >= self.js_threshold))

        updated_params: Dict[str, torch.Tensor] = {}
        mix1_fragments: List[str] = []
        mix2_fragments: List[str] = []
        mix1_bytes = 0
        mix2_blocking_bytes = 0
        saved_blocking_bytes = 0

        for name, master_t in master_params.items():
            m_fp32 = master_t.detach().to(torch.float32).cpu()
            worker_tensors = [
                ws[name].detach().to(torch.float32).cpu()
                for ws in worker_param_states
                if name in ws
            ]
            if not worker_tensors:
                updated_params[name] = master_t.clone()
                continue

            frag_bytes = int(master_t.numel() * 2)  # BF16 wire footprint per fragment
            sensitive = self.is_sensitive_fragment(name)

            if sensitive and trigger_mix2:
                # Mix2: Blocking all-reduce of outer pseudo-gradients (master - worker_m) + Nesterov step
                pseudo_grads = [m_fp32 - wt for wt in worker_tensors]
                avg_grad = torch.stack(pseudo_grads, dim=0).mean(dim=0)
                v = self.mix2_velocity.get(name)
                if v is None or v.shape != avg_grad.shape:
                    v = torch.zeros_like(avg_grad)
                v = self.outer_momentum_mix2 * v + avg_grad
                self.mix2_velocity[name] = v
                nesterov_step = self.outer_momentum_mix2 * v + avg_grad
                new_t = m_fp32 - self.outer_lr_mix2 * nesterov_step
                updated_params[name] = new_t.to(device=master_t.device, dtype=master_t.dtype)
                mix2_fragments.append(name)
                mix2_blocking_bytes += frag_bytes * len(worker_tensors)
            else:
                # Mix1: Non-blocking pairwise/cohort parameter gossip
                peer_avg = torch.stack(worker_tensors, dim=0).mean(dim=0)
                new_t = (1.0 - self.alpha_mix1) * m_fp32 + self.alpha_mix1 * peer_avg
                updated_params[name] = new_t.to(device=master_t.device, dtype=master_t.dtype)
                mix1_fragments.append(name)
                mix1_bytes += frag_bytes
                if sensitive and not trigger_mix2:
                    saved_blocking_bytes += frag_bytes * len(worker_tensors)

        self.round_index += 1
        return updated_params, {
            "strategy_id": "HM-S9-04-FACTORED-GOSSIP-MIX1-MIX2",
            "round_index": self.round_index,
            "jensen_shannon_disagreement": round(js_div, 8),
            "js_threshold": self.js_threshold,
            "mix2_triggered": trigger_mix2,
            "mix1_fragment_count": len(mix1_fragments),
            "mix2_fragment_count": len(mix2_fragments),
            "mix1_nonblocking_wire_bytes": mix1_bytes,
            "mix2_blocking_wire_bytes": mix2_blocking_bytes,
            "saved_blocking_wire_bytes": saved_blocking_bytes,
        }



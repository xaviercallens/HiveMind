# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""TIES/DARE Sign-Consensus and Prose-Nullspace Task-Arithmetic Merger for Sparse MoEs.

Implements training-free task-arithmetic merging for asynchronous distributed OpenDiLoCo
open-weight Sparse MoE LoRA checkpoints (``CrowdV1WithEngram`` + ``CrowdLoRAMoEMLP``)
so worker deltas never re-introduce general-prose drift upon assimilation:

1. ``trim_and_elect_sign_ties(deltas, *, density_top_p=0.25)``:
   Magnitude trimming, dominant-sign election, and disjoint mean aggregation across
   ``K`` peer checkpoints while neutralizing opposing gradient sign cancellations.
2. ``dare_rescale_and_svd_compress_lora(peer_lora_pairs, *, drop_rate, target_rank, seed)``:
   Merges multiple rank-``r`` peer SwiGLU LoRA factor pairs ``(B_k, A_k)`` back into an
   exact rank-``target_rank`` ``(B_merged, A_merged)`` adapter via deterministic DARE
   rescaling and exact thin QR-SVD without materializing full dense matrices or inflating rank.
3. ``compute_prose_nullspace_projector`` & ``protect_router_frag_layernorm_tail``:
   Projects merged adapter and router deltas orthogonal to general-prose principal modes
   estimated via Ledoit-Wolf shrunk ``M_prose``, while shielding the ``router_frag`` final
   LayerNorm tail from cumulative multi-round drift.
4. ``merge_moe_checkpoints_with_prose_guard(...)``:
   Orchestrates end-to-end MoE checkpoint assimilation under the ANSE ``<= +1.0%``
   general-domain guard with atomic bitwise parent rollback and deterministic SHA-256 receipts.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdLoRAMoEMLP,
    extract_moe_trainable_buckets,
    load_moe_trainable_buckets,
)
from hivemind.swarm.marchenko_pastur_covariance_guard import (
    compute_condition_bounded_alphaedit_filter,
    compute_oracle_ledoit_wolf_moment,
)
from hivemind.swarm.wsp_alphaedit_control import (
    ANSE_SAFE_E0_RATIO,
    DOMAIN_GUARD_MAX_RATIO,
    GENERAL_EXPERT_PREFIX,
    LN_GUARD_MAX_RATIO,
    compute_embeddings_batch,
    compute_swiglu_intermediate_states,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    forward_model_pre_ln,
    per_chunk_nll_from_pre_ln,
    select_guard_optimal_lambda_ln,
)

SCHEMA_VERSION: str = "hivemind.ties_dare_moe_merger/1"
DEFAULT_DENSITY_TOP_P: float = 0.25
DEFAULT_DARE_DROP_RATE: float = 0.25
DEFAULT_PROSE_EIGEN_RATIO_TAU: float = 1.25
DEFAULT_PROSE_VARIANCE_THRESHOLD: float = 0.95
DEFAULT_PROSE_MAX_MODES: int = 64
DEFAULT_PROSE_PROJ_STRENGTH: float = 1.0
DEFAULT_ALPHAEDIT_ETA: float = 4.0
DEFAULT_ALPHAEDIT_REG_SPEC: float = 0.10
DEFAULT_MAX_OPERATOR_NORM: float = 1.05
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = DOMAIN_GUARD_MAX_RATIO
DEFAULT_ANSE_SAFE_E0_RATIO: float = ANSE_SAFE_E0_RATIO
ANSE_MAXIMUM_PAIN: float = 1_000_000.0

_SWIGLU_PROJECTIONS: tuple[str, ...] = ("w1", "w2", "w3")
_VALID_SIGN_MODES: frozenset[str] = frozenset({"mass", "majority"})
_VALID_LN_POLICIES: frozenset[str] = frozenset({"freeze_parent", "guard_calibrated"})
_VALID_PROSE_MODES: frozenset[str] = frozenset(
    {"orthogonal_nullspace", "condition_bounded_alphaedit", "none"}
)


def _validate_nonneg_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number >= 0.0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0.0, got {value!r}")
    return out


def _validate_positive_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number > 0.0."""
    out = _validate_nonneg_finite(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_unit_interval(value: Any, name: str, *, allow_zero: bool = True, allow_one: bool = True) -> float:
    """Validate a finite real value inside ``[0, 1]`` with configurable endpoint inclusivity."""
    out = _validate_nonneg_finite(value, name)
    if not allow_zero and out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    if allow_one and out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    if not allow_one and out >= 1.0:
        raise ValueError(f"{name} must be < 1.0, got {value!r}")
    return out


def _validate_nonneg_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a non-negative int (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a strictly positive int (excluding bool)."""
    out = _validate_nonneg_int(value, name)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return out


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a bool."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_finite_tensor(tensor: Any, name: str, *, ndim: int | None = None) -> torch.Tensor:
    """Validate a non-empty finite tensor and return a detached CPU ``float64`` copy."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty, got shape {tuple(tensor.shape)}")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D tensor, got shape {tuple(tensor.shape)}")
    xd = tensor.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(xd).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return xd


def _normalize_peer_weights(peer_weights: Sequence[float] | None, num_peers: int) -> list[float]:
    """Validate and normalize peer weights to sum to ``1.0``."""
    if peer_weights is None:
        return [1.0 / float(num_peers)] * num_peers
    if isinstance(peer_weights, (str, bytes)) or not isinstance(peer_weights, Sequence):
        raise TypeError("peer_weights must be a sequence of numbers or None")
    if len(peer_weights) != num_peers:
        raise ValueError(f"peer_weights length ({len(peer_weights)}) must match peer count ({num_peers})")
    vals = [_validate_nonneg_finite(w, f"peer_weights[{i}]") for i, w in enumerate(peer_weights)]
    total = sum(vals)
    if total <= 0.0:
        raise ValueError("peer_weights must have a strictly positive sum")
    return [v / total for v in vals]


def _apply_peer_norm_cap(
    tensors: list[torch.Tensor],
    max_peer_norm_ratio: float | None,
) -> tuple[list[torch.Tensor], int]:
    """Optionally cap peer delta L2 norms relative to the cohort median when ``K >= 3``."""
    if max_peer_norm_ratio is None or len(tensors) < 3:
        return tensors, 0
    norms = [float(t.norm().item()) for t in tensors]
    med_norm = float(np.median(norms))
    if med_norm <= 1e-12:
        return tensors, 0
    cap = max_peer_norm_ratio * med_norm
    clipped = 0
    out: list[torch.Tensor] = []
    for t, n_val in zip(tensors, norms, strict=True):
        if n_val > cap and n_val > 0.0:
            out.append(t * (cap / n_val))
            clipped += 1
        else:
            out.append(t)
    return out, clipped


def _compute_ties_weights_and_mask(
    deltas_64: list[torch.Tensor],
    *,
    density_top_p: float,
    norm_weights: list[float],
    sign_election_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Trim magnitudes, elect dominant coordinate signs, and return ``(trimmed, agree_mask, agree_weights, denom, metrics)``."""
    num_peers = len(deltas_64)
    numel = deltas_64[0].numel()
    k_keep = max(1, min(numel, int(math.ceil(float(numel) * density_top_p))))

    trimmed_list: list[torch.Tensor] = []
    for d_t in deltas_64:
        if k_keep >= numel:
            trimmed_list.append(d_t)
            continue
        flat_abs = d_t.reshape(-1).abs()
        thresh = torch.topk(flat_abs, k=k_keep, largest=True).values.min()
        keep_mask = (d_t.abs() >= thresh) & (d_t != 0.0)
        trimmed_list.append(torch.where(keep_mask, d_t, torch.zeros_like(d_t)))

    stacked = torch.stack(trimmed_list, dim=0)  # (K, ...)
    w_view = torch.tensor(norm_weights, dtype=torch.float64).view(num_peers, *([1] * deltas_64[0].ndim))
    nz_mask = stacked != 0.0
    signs = torch.sign(stacked)

    if sign_election_mode == "majority":
        vote_mass = (signs * w_view * nz_mask.to(torch.float64)).sum(dim=0)
    else:
        vote_mass = (stacked * w_view).sum(dim=0)

    elected_sign = torch.sign(vote_mass)
    agree_mask = nz_mask & (signs == elected_sign.unsqueeze(0))

    active_per_coord = nz_mask.sum(dim=0)
    agree_per_coord = agree_mask.sum(dim=0)
    total_active_votes = int(active_per_coord.sum().item())
    total_agree_votes = int(agree_per_coord.sum().item())
    sign_consensus_ratio = (
        float(total_agree_votes) / float(total_active_votes) if total_active_votes > 0 else 1.0
    )

    active_coords = active_per_coord > 0
    num_active_coords = int(active_coords.sum().item())
    conflict_coords = int(((agree_per_coord < active_per_coord) & active_coords).sum().item())
    conflict_fraction = (
        float(conflict_coords) / float(num_active_coords) if num_active_coords > 0 else 0.0
    )

    agree_weights = w_view * agree_mask.to(torch.float64)
    denom = agree_weights.sum(dim=0)
    metrics = {
        "sign_consensus_ratio": sign_consensus_ratio,
        "conflict_coordinate_fraction": conflict_fraction,
        "active_coordinate_fraction": float(num_active_coords) / float(max(1, numel)),
        "k_keep_per_peer": float(k_keep),
    }
    return stacked, agree_mask, agree_weights, denom, metrics


def _sample_dare_mask(shape: tuple[int, ...], keep_prob: float, gen: torch.Generator) -> torch.Tensor:
    """Sample a deterministic Bernoulli(``keep_prob``) mask rescaled by ``1 / keep_prob``."""
    if keep_prob >= 1.0:
        return torch.ones(shape, dtype=torch.float64)
    probs = torch.full(shape, keep_prob, dtype=torch.float64)
    mask = torch.bernoulli(probs, generator=gen)
    return mask / keep_prob


def _qr_svd_factor_pair(
    b_cat: torch.Tensor,
    a_cat: torch.Tensor,
    target_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compress concatenated LoRA factors ``(B_cat, A_cat)`` to rank ``target_rank`` via thin QR-SVD.

    Never materializes the dense ``(d_out, d_in)`` product matrix.
    """
    d_out, tot_r = int(b_cat.shape[0]), int(b_cat.shape[1])
    d_in = int(a_cat.shape[1])
    if tot_r == 0 or float(b_cat.abs().max().item()) == 0.0 or float(a_cat.abs().max().item()) == 0.0:
        return (
            torch.zeros((d_out, target_rank), dtype=torch.float64),
            torch.zeros((target_rank, d_in), dtype=torch.float64),
            {
                "core_rank": 0.0,
                "effective_rank": 0.0,
                "energy_retained_ratio": 1.0,
                "top_singular_value": 0.0,
                "core_max_dim": 0.0,
            },
        )

    qb, rb = torch.linalg.qr(b_cat, mode="reduced")
    qa, ra = torch.linalg.qr(a_cat.T, mode="reduced")
    core = rb @ ra.T
    uc, s_vals, vct = torch.linalg.svd(core, full_matrices=False)

    k_avail = int(s_vals.numel())
    k_keep = min(target_rank, k_avail)
    total_energy = float(s_vals.pow(2).sum().item())
    kept_energy = float(s_vals[:k_keep].pow(2).sum().item())
    energy_ratio = (kept_energy / total_energy) if total_energy > 1e-30 else 1.0

    s_keep = s_vals[:k_keep].clamp_min(0.0)
    s_sqrt = s_keep.sqrt()
    u_keep = qb @ uc[:, :k_keep]
    vt_keep = vct[:k_keep, :] @ qa.T

    b_out = torch.zeros((d_out, target_rank), dtype=torch.float64)
    a_out = torch.zeros((target_rank, d_in), dtype=torch.float64)
    b_out[:, :k_keep] = u_keep * s_sqrt.unsqueeze(0)
    a_out[:k_keep, :] = s_sqrt.unsqueeze(1) * vt_keep

    eff_rank = int((s_keep > 1e-12).sum().item())
    return b_out, a_out, {
        "core_rank": float(max(core.shape)),
        "effective_rank": float(eff_rank),
        "energy_retained_ratio": float(energy_ratio),
        "top_singular_value": float(s_vals[0].item()) if k_avail > 0 else 0.0,
        "core_max_dim": float(max(core.shape)),
    }


@dataclass(frozen=True)
class TIESDAREMergerConfig:
    """Configuration governing TIES/DARE merging, prose-nullspace projection, and ANSE acceptance."""

    density_top_p: float = DEFAULT_DENSITY_TOP_P
    drop_rate: float = DEFAULT_DARE_DROP_RATE
    target_rank: int | None = None
    seed: int = 20261124
    sign_election_mode: str = "mass"
    max_peer_norm_ratio: float | None = None
    scaling_lambda: float = 1.0
    router_drop_rate: float = 0.0
    router_density_top_p: float | None = None
    ln_tail_policy: str = "guard_calibrated"
    ln_max_step_norm: float | None = None
    prose_projection_mode: str = "orthogonal_nullspace"
    prose_eigen_ratio_tau: float = DEFAULT_PROSE_EIGEN_RATIO_TAU
    prose_variance_threshold: float = DEFAULT_PROSE_VARIANCE_THRESHOLD
    prose_max_modes: int = DEFAULT_PROSE_MAX_MODES
    prose_proj_strength: float = DEFAULT_PROSE_PROJ_STRENGTH
    alphaedit_eta: float = DEFAULT_ALPHAEDIT_ETA
    alphaedit_reg_spec: float = DEFAULT_ALPHAEDIT_REG_SPEC
    alphaedit_max_operator_norm: float = DEFAULT_MAX_OPERATOR_NORM
    project_router_delta_orthogonal: bool = True
    protect_general_expert: bool = True
    general_expert_index: int = 0
    domain_guard_max_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    anse_safe_e0_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    enforce_all_domain_guard: bool = True
    require_energy_improvement: bool = True
    score_from: int = 64
    batch_size: int = 24

    def __post_init__(self) -> None:
        _validate_unit_interval(self.density_top_p, "density_top_p", allow_zero=False, allow_one=True)
        _validate_unit_interval(self.drop_rate, "drop_rate", allow_zero=True, allow_one=False)
        if self.target_rank is not None:
            _validate_positive_int(self.target_rank, "target_rank")
        _validate_nonneg_int(self.seed, "seed")
        if self.sign_election_mode not in _VALID_SIGN_MODES:
            raise ValueError(f"sign_election_mode must be one of {sorted(_VALID_SIGN_MODES)}")
        if self.max_peer_norm_ratio is not None:
            _validate_positive_finite(self.max_peer_norm_ratio, "max_peer_norm_ratio")
        _validate_positive_finite(self.scaling_lambda, "scaling_lambda")
        _validate_unit_interval(self.router_drop_rate, "router_drop_rate", allow_zero=True, allow_one=False)
        if self.router_density_top_p is not None:
            _validate_unit_interval(self.router_density_top_p, "router_density_top_p", allow_zero=False, allow_one=True)
        if self.ln_tail_policy not in _VALID_LN_POLICIES:
            raise ValueError(f"ln_tail_policy must be one of {sorted(_VALID_LN_POLICIES)}")
        if self.ln_max_step_norm is not None:
            _validate_nonneg_finite(self.ln_max_step_norm, "ln_max_step_norm")
        if self.prose_projection_mode not in _VALID_PROSE_MODES:
            raise ValueError(f"prose_projection_mode must be one of {sorted(_VALID_PROSE_MODES)}")
        _validate_nonneg_finite(self.prose_eigen_ratio_tau, "prose_eigen_ratio_tau")
        _validate_unit_interval(self.prose_variance_threshold, "prose_variance_threshold", allow_zero=False, allow_one=True)
        _validate_nonneg_int(self.prose_max_modes, "prose_max_modes")
        _validate_unit_interval(self.prose_proj_strength, "prose_proj_strength", allow_zero=True, allow_one=True)
        _validate_nonneg_finite(self.alphaedit_eta, "alphaedit_eta")
        _validate_nonneg_finite(self.alphaedit_reg_spec, "alphaedit_reg_spec")
        max_op = _validate_positive_finite(self.alphaedit_max_operator_norm, "alphaedit_max_operator_norm")
        if max_op < 1.0:
            raise ValueError(f"alphaedit_max_operator_norm must be >= 1.0, got {self.alphaedit_max_operator_norm!r}")
        _validate_bool(self.project_router_delta_orthogonal, "project_router_delta_orthogonal")
        _validate_bool(self.protect_general_expert, "protect_general_expert")
        _validate_nonneg_int(self.general_expert_index, "general_expert_index")
        guard_r = _validate_positive_finite(self.domain_guard_max_ratio, "domain_guard_max_ratio")
        if guard_r < 1.0:
            raise ValueError(f"domain_guard_max_ratio must be >= 1.0, got {self.domain_guard_max_ratio!r}")
        safe_r = _validate_positive_finite(self.anse_safe_e0_ratio, "anse_safe_e0_ratio")
        if safe_r < 1.0:
            raise ValueError(f"anse_safe_e0_ratio must be >= 1.0, got {self.anse_safe_e0_ratio!r}")
        _validate_bool(self.enforce_all_domain_guard, "enforce_all_domain_guard")
        _validate_bool(self.require_energy_improvement, "require_energy_improvement")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")


# ---------------------------------------------------------------------------
# (a) TIES Trim, Sign Election & Disjoint Mean Aggregation
# ---------------------------------------------------------------------------


@torch.no_grad()
def trim_and_elect_sign_ties(
    deltas: Sequence[torch.Tensor],
    *,
    density_top_p: float = DEFAULT_DENSITY_TOP_P,
    peer_weights: Sequence[float] | None = None,
    sign_election_mode: str = "mass",
    max_peer_norm_ratio: float | None = None,
    scaling_lambda: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Aggregate ``K`` peer task deltas via TIES magnitude trimming, sign election, and disjoint mean.

    Resolves opposing-sign gradient interference across asynchronous specialist workers:
    1. Trims each peer delta to its top ``density_top_p`` entries by magnitude.
    2. Elects the dominant sign per coordinate via weighted signed mass (or majority vote).
    3. Averages strictly across agreeing non-zero peer entries (disjoint mean), eliminating
       destructive sign cancellation.
    """
    if isinstance(deltas, (str, bytes, Mapping)) or not isinstance(deltas, Sequence) or len(deltas) == 0:
        raise ValueError("deltas must be a non-empty sequence of torch.Tensor objects")
    top_p = _validate_unit_interval(density_top_p, "density_top_p", allow_zero=False, allow_one=True)
    if sign_election_mode not in _VALID_SIGN_MODES:
        raise ValueError(f"sign_election_mode must be one of {sorted(_VALID_SIGN_MODES)}")
    cap_ratio = (
        _validate_positive_finite(max_peer_norm_ratio, "max_peer_norm_ratio")
        if max_peer_norm_ratio is not None
        else None
    )
    lam_f = _validate_nonneg_finite(scaling_lambda, "scaling_lambda")

    first = deltas[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError(f"deltas[0] must be a torch.Tensor, got {type(first).__name__}")
    target_shape = tuple(first.shape)
    deltas_64: list[torch.Tensor] = []
    for idx, delta in enumerate(deltas):
        t64 = _validate_finite_tensor(delta, f"deltas[{idx}]")
        if tuple(t64.shape) != target_shape:
            raise ValueError(f"deltas[{idx}] shape {tuple(t64.shape)} != expected shape {target_shape}")
        deltas_64.append(t64)

    norm_weights = _normalize_peer_weights(peer_weights, len(deltas_64))
    bounded_deltas, norms_clipped = _apply_peer_norm_cap(deltas_64, cap_ratio)

    stacked, _, agree_weights, denom, metrics = _compute_ties_weights_and_mask(
        bounded_deltas,
        density_top_p=top_p,
        norm_weights=norm_weights,
        sign_election_mode=sign_election_mode,
    )
    safe_denom = denom.clamp_min(1e-12)
    merged_64 = torch.where(denom > 0.0, (stacked * agree_weights).sum(dim=0) / safe_denom, torch.zeros_like(denom))
    merged_64 = (lam_f * merged_64).contiguous()

    merged_out = merged_64.to(device=first.device, dtype=first.dtype).contiguous()
    nonzero_frac = float((merged_64 != 0.0).sum().item()) / float(max(1, merged_64.numel()))

    return merged_out, {
        "num_peers": len(deltas_64),
        "density_top_p": top_p,
        "sign_election_mode": sign_election_mode,
        "scaling_lambda": lam_f,
        "norms_clipped": norms_clipped,
        "sign_consensus_ratio": float(metrics["sign_consensus_ratio"]),
        "conflict_coordinate_fraction": float(metrics["conflict_coordinate_fraction"]),
        "active_coordinate_fraction": float(metrics["active_coordinate_fraction"]),
        "nonzero_density_post_trim": nonzero_frac,
        "merged_l2_norm": float(merged_64.norm().item()),
    }


# ---------------------------------------------------------------------------
# (b) Deterministic DARE Rescaling & Exact Thin QR-SVD LoRA Compression
# ---------------------------------------------------------------------------


@torch.no_grad()
def dare_rescale_and_svd_compress_lora(
    peer_lora_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    drop_rate: float = DEFAULT_DARE_DROP_RATE,
    target_rank: int,
    seed: int,
    peer_weights: Sequence[float] | None = None,
    base_pair: tuple[torch.Tensor, torch.Tensor] | None = None,
    scaling_lambda: float = 1.0,
    ties_density_top_p: float | None = None,
    sign_election_mode: str = "mass",
    max_peer_norm_ratio: float | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
    """Merge ``K`` rank-``r`` LoRA pairs ``(B_k, A_k)`` into exact rank-``target_rank`` via DARE + QR-SVD.

    Applies deterministic Bernoulli DARE drop-and-rescale (factor keep probability
    ``sqrt(1 - drop_rate)`` on each factor delta so the effective product keep probability
    is exact ``1 - drop_rate`` and ``E[B_tilde @ A_tilde] == B @ A``), optional TIES
    sign-consensus filtering on factor deltas, and exact thin QR-SVD compression on the
    concatenated ``(d_out x sum_r)`` and ``(sum_r x d_in)`` matrices without ever
    materializing dense ``(d_out x d_in)`` matrices or inflating adapter rank.
    """
    if isinstance(peer_lora_pairs, (str, bytes, Mapping)) or not isinstance(peer_lora_pairs, Sequence) or not peer_lora_pairs:
        raise ValueError("peer_lora_pairs must be a non-empty sequence of (B_k, A_k) tensor pairs")
    p_drop = _validate_unit_interval(drop_rate, "drop_rate", allow_zero=True, allow_one=False)
    r_tgt = _validate_positive_int(target_rank, "target_rank")
    seed_i = _validate_nonneg_int(seed, "seed")
    lam_f = _validate_nonneg_finite(scaling_lambda, "scaling_lambda")
    ties_p = (
        _validate_unit_interval(ties_density_top_p, "ties_density_top_p", allow_zero=False, allow_one=True)
        if ties_density_top_p is not None
        else None
    )
    if sign_election_mode not in _VALID_SIGN_MODES:
        raise ValueError(f"sign_election_mode must be one of {sorted(_VALID_SIGN_MODES)}")
    cap_ratio = (
        _validate_positive_finite(max_peer_norm_ratio, "max_peer_norm_ratio")
        if max_peer_norm_ratio is not None
        else None
    )

    pairs_64: list[tuple[torch.Tensor, torch.Tensor]] = []
    first_pair = peer_lora_pairs[0]
    if isinstance(first_pair, (str, bytes)) or not isinstance(first_pair, Sequence) or len(first_pair) != 2:
        raise ValueError("peer_lora_pairs[0] must be a (B_k, A_k) pair")
    ref_b0, ref_a0 = first_pair
    b0_64 = _validate_finite_tensor(ref_b0, "peer_lora_pairs[0][0]", ndim=2)
    a0_64 = _validate_finite_tensor(ref_a0, "peer_lora_pairs[0][1]", ndim=2)
    d_out, d_in = int(b0_64.shape[0]), int(a0_64.shape[1])
    out_device, out_dtype = ref_b0.device, ref_b0.dtype

    for k, pair in enumerate(peer_lora_pairs):
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError(f"peer_lora_pairs[{k}] must be a (B_k, A_k) pair")
        bk = _validate_finite_tensor(pair[0], f"peer_lora_pairs[{k}][0]", ndim=2)
        ak = _validate_finite_tensor(pair[1], f"peer_lora_pairs[{k}][1]", ndim=2)
        if bk.shape[0] != d_out or ak.shape[1] != d_in:
            raise ValueError(
                f"peer_lora_pairs[{k}] outer dims ({bk.shape[0]}, {ak.shape[1]}) != expected ({d_out}, {d_in})"
            )
        if bk.shape[1] != ak.shape[0]:
            raise ValueError(f"peer_lora_pairs[{k}] inner rank mismatch: {bk.shape[1]} vs {ak.shape[0]}")
        pairs_64.append((bk, ak))

    if base_pair is not None:
        if not isinstance(base_pair, Sequence) or len(base_pair) != 2:
            raise ValueError("base_pair must be a (B_base, A_base) tuple or None")
        b_base = _validate_finite_tensor(base_pair[0], "base_pair[0]", ndim=2)
        a_base = _validate_finite_tensor(base_pair[1], "base_pair[1]", ndim=2)
        if b_base.shape[0] != d_out or a_base.shape[1] != d_in or b_base.shape[1] != a_base.shape[0]:
            raise ValueError("base_pair dimensions do not match peer_lora_pairs")
        delta_b_list = [bk - b_base for bk, _ in pairs_64]
        delta_a_list = [ak - a_base for _, ak in pairs_64]
    else:
        b_base = None
        a_base = None
        delta_b_list = [bk for bk, _ in pairs_64]
        delta_a_list = [ak for _, ak in pairs_64]

    num_peers = len(pairs_64)
    norm_weights = _normalize_peer_weights(peer_weights, num_peers)
    gen = torch.Generator(device="cpu").manual_seed(seed_i)
    keep_factor_prob = math.sqrt(max(1e-12, 1.0 - p_drop))

    # Optional TIES sign-consensus masks across factor deltas when peers share shape and K >= 2
    b_shapes_equal = len({tuple(t.shape) for t in delta_b_list}) == 1
    a_shapes_equal = len({tuple(t.shape) for t in delta_a_list}) == 1
    ties_ratios: list[float] = []

    b_survived_per_peer: list[torch.Tensor] | None = None
    if ties_p is not None and num_peers >= 2 and b_shapes_equal:
        db_cap, _ = _apply_peer_norm_cap(delta_b_list, cap_ratio)
        st_b, agree_b, _, den_b, m_b = _compute_ties_weights_and_mask(
            db_cap, density_top_p=ties_p, norm_weights=norm_weights, sign_election_mode=sign_election_mode
        )
        ties_ratios.append(float(m_b["sign_consensus_ratio"]))
        delta_b_eff = [
            torch.where(
                agree_b[k] & (den_b > 0.0),
                st_b[k] / den_b.clamp_min(1e-12),
                torch.zeros_like(st_b[k]),
            )
            for k in range(num_peers)
        ]
        # Track per-peer survival per rank column so A rescaling avoids double-dividing peers dropped in B
        base_b_col_active = (
            (b_base.abs().max(dim=0).values > 0.0) if b_base is not None else torch.zeros(st_b.shape[2], dtype=torch.bool)
        )
        b_survived_per_peer = [
            ((agree_b[k] & (st_b[k] != 0.0)).any(dim=0) | base_b_col_active)
            for k in range(num_peers)
        ]
    else:
        delta_b_eff = delta_b_list

    if ties_p is not None and num_peers >= 2 and a_shapes_equal:
        da_cap, _ = _apply_peer_norm_cap(delta_a_list, cap_ratio)
        st_a, agree_a, _, _, m_a = _compute_ties_weights_and_mask(
            da_cap, density_top_p=ties_p, norm_weights=norm_weights, sign_election_mode=sign_election_mode
        )
        ties_ratios.append(float(m_a["sign_consensus_ratio"]))
        w_a_col = torch.tensor(norm_weights, dtype=torch.float64).view(num_peers, 1, 1)
        if b_survived_per_peer is not None:
            b_surv_stack = torch.stack(b_survived_per_peer, dim=0).unsqueeze(-1).to(torch.float64)  # (K, r, 1)
            b_surv_mass = (w_a_col * b_surv_stack).sum(dim=0).clamp_min(1e-12)
            joint_agree_mass = (w_a_col * b_surv_stack * agree_a.to(torch.float64)).sum(dim=0)
            cond_denom_a = joint_agree_mass / b_surv_mass
        else:
            cond_denom_a = (w_a_col * agree_a.to(torch.float64)).sum(dim=0)

        delta_a_eff = [
            torch.where(
                agree_a[k] & (cond_denom_a > 0.0),
                st_a[k] / cond_denom_a.clamp_min(1e-12),
                torch.zeros_like(st_a[k]),
            )
            for k in range(num_peers)
        ]
    else:
        delta_a_eff = delta_a_list

    b_blocks: list[torch.Tensor] = []
    a_blocks: list[torch.Tensor] = []
    input_rank_sum = 0

    for k in range(num_peers):
        db_k = delta_b_eff[k]
        da_k = delta_a_eff[k]
        mask_b = _sample_dare_mask(tuple(db_k.shape), keep_factor_prob, gen)
        mask_a = _sample_dare_mask(tuple(da_k.shape), keep_factor_prob, gen)
        db_dare = db_k * mask_b
        da_dare = da_k * mask_a

        scale_k = math.sqrt(max(0.0, lam_f * norm_weights[k]))
        if b_base is not None and a_base is not None:
            b_k_tilde = b_base + db_dare
            a_k_tilde = a_base + da_dare
            b_blocks.append(scale_k * b_k_tilde)
            a_blocks.append(scale_k * a_k_tilde)
            input_rank_sum += int(b_k_tilde.shape[1])
        else:
            b_blocks.append(scale_k * db_dare)
            a_blocks.append(scale_k * da_dare)
            input_rank_sum += int(db_dare.shape[1])

    if b_base is not None and a_base is not None and abs(1.0 - lam_f) > 1e-12:
        rem = 1.0 - lam_f
        if rem > 0.0:
            s_rem = math.sqrt(rem)
            b_blocks.append(s_rem * b_base)
            a_blocks.append(s_rem * a_base)
        else:
            s_rem = math.sqrt(-rem)
            b_blocks.append(-s_rem * b_base)
            a_blocks.append(s_rem * a_base)
        input_rank_sum += int(b_base.shape[1])

    b_cat = torch.cat(b_blocks, dim=1)
    a_cat = torch.cat(a_blocks, dim=0)
    b_merged_64, a_merged_64, svd_stats = _qr_svd_factor_pair(b_cat, a_cat, r_tgt)

    b_out = b_merged_64.to(device=out_device, dtype=out_dtype).contiguous()
    a_out = a_merged_64.to(device=out_device, dtype=out_dtype).contiguous()
    sign_consensus = float(np.mean(ties_ratios)) if ties_ratios else 1.0

    return (b_out, a_out), {
        "num_peers": num_peers,
        "drop_rate": p_drop,
        "factor_keep_prob": keep_factor_prob,
        "target_rank": r_tgt,
        "input_rank_sum": input_rank_sum,
        "preserved_exact_rank": int(a_out.shape[0]),
        "seed": seed_i,
        "sign_consensus_ratio": sign_consensus,
        "dense_matrix_materialized": False,
        **svd_stats,
    }


@torch.no_grad()
def merge_swiglu_lora_adapters_dare_svd(
    peer_adapters: Sequence[Mapping[str, tuple[torch.Tensor, torch.Tensor]]],
    *,
    drop_rate: float = DEFAULT_DARE_DROP_RATE,
    target_rank: int,
    seed: int,
    peer_weights: Sequence[float] | None = None,
    base_adapter: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    scaling_lambda: float = 1.0,
    ties_density_top_p: float | None = None,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    """Merge SwiGLU ``(w1, w2, w3)`` LoRA adapters via deterministic DARE + exact thin QR-SVD."""
    if isinstance(peer_adapters, (str, bytes, Mapping)) or not isinstance(peer_adapters, Sequence) or not peer_adapters:
        raise ValueError("peer_adapters must be a non-empty sequence of projection mappings")

    merged: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    proj_reports: dict[str, Any] = {}
    for p_idx, proj in enumerate(_SWIGLU_PROJECTIONS):
        pairs = [adapter[proj] for adapter in peer_adapters]
        b_pair = base_adapter[proj] if base_adapter is not None else None
        pair_out, p_meta = dare_rescale_and_svd_compress_lora(
            pairs,
            drop_rate=drop_rate,
            target_rank=target_rank,
            seed=seed + p_idx * 1009,
            peer_weights=peer_weights,
            base_pair=b_pair,
            scaling_lambda=scaling_lambda,
            ties_density_top_p=ties_density_top_p,
        )
        merged[proj] = pair_out
        proj_reports[proj] = p_meta

    return merged, {
        "projections": proj_reports,
        "target_rank": target_rank,
        "mean_energy_retained_ratio": float(
            np.mean([r["energy_retained_ratio"] for r in proj_reports.values()])
        ),
        "dense_matrix_materialized": False,
    }


# ---------------------------------------------------------------------------
# (c) Prose-Nullspace Projection & Protected LayerNorm Tail
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_prose_nullspace_projector(
    prose_x: torch.Tensor,
    spec_x: torch.Tensor | None = None,
    *,
    mode: str = "orthogonal_nullspace",
    eigen_ratio_tau: float = DEFAULT_PROSE_EIGEN_RATIO_TAU,
    variance_threshold: float = DEFAULT_PROSE_VARIANCE_THRESHOLD,
    max_modes: int = DEFAULT_PROSE_MAX_MODES,
    proj_strength: float = DEFAULT_PROSE_PROJ_STRENGTH,
    eta: float = DEFAULT_ALPHAEDIT_ETA,
    reg_spec: float = DEFAULT_ALPHAEDIT_REG_SPEC,
    max_operator_norm: float = DEFAULT_MAX_OPERATOR_NORM,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build a float64 right projector removing general-prose principal modes via Ledoit-Wolf shrinkage."""
    xp = _validate_finite_tensor(prose_x, "prose_x", ndim=2)
    d = int(xp.shape[1])
    if mode not in _VALID_PROSE_MODES:
        raise ValueError(f"mode must be one of {sorted(_VALID_PROSE_MODES)}")
    tau_f = _validate_nonneg_finite(eigen_ratio_tau, "eigen_ratio_tau")
    var_thr = _validate_unit_interval(variance_threshold, "variance_threshold", allow_zero=False, allow_one=True)
    max_m = _validate_nonneg_int(max_modes, "max_modes")
    strength = _validate_unit_interval(proj_strength, "proj_strength", allow_zero=True, allow_one=True)
    eta_f = _validate_nonneg_finite(eta, "eta")
    reg_f = _validate_nonneg_finite(reg_spec, "reg_spec")
    max_op = _validate_positive_finite(max_operator_norm, "max_operator_norm")
    if max_op < 1.0:
        raise ValueError(f"max_operator_norm must be >= 1.0, got {max_operator_norm!r}")

    eye_d = torch.eye(d, dtype=torch.float64)
    if mode == "none" or max_m == 0 or strength == 0.0:
        return eye_d, {
            "mode": mode,
            "projected_modes": 0,
            "rho_prose": 0.0,
            "operator_norm": 1.0,
        }

    m_prose, lw_p = compute_oracle_ledoit_wolf_moment(xp)
    if spec_x is not None:
        xs = _validate_finite_tensor(spec_x, "spec_x", ndim=2)
        if xs.shape[1] != d:
            raise ValueError(f"spec_x width {xs.shape[1]} != prose_x width {d}")
        m_spec, lw_s = compute_oracle_ledoit_wolf_moment(xs)
    else:
        xs = None
        m_spec, lw_s = None, None

    if mode == "condition_bounded_alphaedit":
        if m_spec is None or xs is None:
            raise ValueError("spec_x must be provided when mode='condition_bounded_alphaedit'")
        g_mat, cb_stats = compute_condition_bounded_alphaedit_filter(
            m_prose,
            m_spec,
            n_prose=int(xp.shape[0]),
            n_spec=int(xs.shape[0]),
            eta=eta_f,
            tau=tau_f,
            max_rank=max_m,
            reg_spec=reg_f,
            max_operator_norm=max_op,
            shrink_moments=False,
        )
        return g_mat, {
            "mode": mode,
            "projected_modes": int(cb_stats["active_modes"]),
            "rho_prose": float(lw_p["rho"]),
            "rho_spec": float(lw_s["rho"]) if lw_s is not None else 0.0,
            "operator_norm": float(cb_stats["operator_norm"]),
        }

    evals_asc, evecs_asc = torch.linalg.eigh(m_prose)
    evals = evals_asc.flip(0).clamp_min(0.0)
    evecs = evecs_asc.flip(1)

    total_trace = float(evals.sum().item())
    mean_eig = total_trace / float(d)
    if total_trace <= 1e-24:
        return eye_d, {
            "mode": mode,
            "projected_modes": 0,
            "rho_prose": float(lw_p["rho"]),
            "operator_norm": 1.0,
        }

    selected: list[int] = []
    cum_var = 0.0
    if m_spec is not None and xs is not None:
        m_spec_emp = 0.5 * (((xs.T @ xs) / float(xs.shape[0])) + ((xs.T @ xs) / float(xs.shape[0])).T)
        lam_floor = max(reg_f * mean_eig * 0.01, 1e-12)
        spec_quad = (evecs * (m_spec_emp @ evecs)).sum(dim=0).clamp_min(0.0)
        ratios = evals / (spec_quad + lam_floor)
        order = torch.argsort(ratios, descending=True)
        for idx_t in order:
            idx = int(idx_t.item())
            if float(evals[idx].item()) <= 1e-6 * mean_eig:
                continue
            if float(ratios[idx].item()) <= tau_f:
                break
            if cum_var >= var_thr or len(selected) >= max_m:
                break
            selected.append(idx)
            cum_var += float(evals[idx].item()) / total_trace
    else:
        for idx in range(d):
            if float(evals[idx].item()) <= tau_f * mean_eig:
                break
            if cum_var >= var_thr or len(selected) >= max_m:
                break
            selected.append(idx)
            cum_var += float(evals[idx].item()) / total_trace

    if not selected:
        return eye_d, {
            "mode": mode,
            "projected_modes": 0,
            "rho_prose": float(lw_p["rho"]),
            "operator_norm": 1.0,
        }

    q_p = evecs[:, selected]
    g_proj = (eye_d - strength * (q_p @ q_p.T)).contiguous()
    return g_proj, {
        "mode": mode,
        "projected_modes": len(selected),
        "selected_variance_fraction": cum_var,
        "rho_prose": float(lw_p["rho"]),
        "rho_spec": float(lw_s["rho"]) if lw_s is not None else 0.0,
        "operator_norm": float(torch.linalg.matrix_norm(g_proj, ord=2).item()),
    }


@torch.no_grad()
def apply_prose_projector_to_lora_pair(
    b_merged: torch.Tensor,
    a_merged: torch.Tensor,
    g_proj: torch.Tensor,
    *,
    base_pair: tuple[torch.Tensor, torch.Tensor] | None = None,
    target_rank: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project adapter delta ``(B_m A_m - B_0 A_0)`` via ``g_proj`` in factored form without dense matrices."""
    bm64 = _validate_finite_tensor(b_merged, "b_merged", ndim=2)
    am64 = _validate_finite_tensor(a_merged, "a_merged", ndim=2)
    gp64 = _validate_finite_tensor(g_proj, "g_proj", ndim=2)
    if gp64.shape != (am64.shape[1], am64.shape[1]):
        raise ValueError(f"g_proj shape {tuple(gp64.shape)} incompatible with A width {am64.shape[1]}")

    r_tgt = int(am64.shape[0]) if target_rank is None else _validate_positive_int(target_rank, "target_rank")

    if base_pair is None or float(base_pair[0].abs().max().item()) == 0.0:
        a_out = (am64 @ gp64).to(device=a_merged.device, dtype=a_merged.dtype).contiguous()
        b_out = bm64[:, :r_tgt].to(device=b_merged.device, dtype=b_merged.dtype).contiguous()
        return b_out, a_out

    bb64 = _validate_finite_tensor(base_pair[0], "base_pair[0]", ndim=2)
    ab64 = _validate_finite_tensor(base_pair[1], "base_pair[1]", ndim=2)
    eye_d = torch.eye(int(gp64.shape[0]), dtype=torch.float64)
    # W_new = B_0 A_0 + (B_m A_m - B_0 A_0) G = B_m (A_m G) + B_0 (A_0 (I - G))
    b_cat = torch.cat([bm64, bb64], dim=1)
    a_cat = torch.cat([am64 @ gp64, ab64 @ (eye_d - gp64)], dim=0)
    b_res, a_res, _ = _qr_svd_factor_pair(b_cat, a_cat, r_tgt)
    return (
        b_res.to(device=b_merged.device, dtype=b_merged.dtype).contiguous(),
        a_res.to(device=a_merged.device, dtype=a_merged.dtype).contiguous(),
    )


@torch.no_grad()
def protect_router_frag_layernorm_tail(
    parent_buckets: Mapping[str, torch.Tensor],
    peer_buckets_list: Sequence[Mapping[str, torch.Tensor]],
    *,
    hidden_size: int,
    density_top_p: float = DEFAULT_DENSITY_TOP_P,
    drop_rate: float = 0.0,
    seed: int = 20261124,
    peer_weights: Sequence[float] | None = None,
    sign_election_mode: str = "mass",
    max_peer_norm_ratio: float | None = None,
    ln_tail_policy: str = "freeze_parent",
    ln_max_step_norm: float | None = None,
    scaling_lambda: float = 1.0,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Merge ``router_frag_*`` buckets while shielding the final ``2 * hidden_size`` LayerNorm tail.

    Separates ``model.ln.weight`` and ``model.ln.bias`` packed at the tail of the final
    ``router_frag_*`` bucket from ``mlp.router.weight`` slices so TIES magnitude trimming
    and DARE dropout never corrupt or drift global output LayerNorm statistics.
    """
    h_dim = _validate_positive_int(hidden_size, "hidden_size")
    if ln_tail_policy not in _VALID_LN_POLICIES:
        raise ValueError(f"ln_tail_policy must be one of {sorted(_VALID_LN_POLICIES)}")
    p_drop = _validate_unit_interval(drop_rate, "drop_rate", allow_zero=True, allow_one=False)
    seed_i = _validate_nonneg_int(seed, "seed")
    ln_cap = _validate_nonneg_finite(ln_max_step_norm, "ln_max_step_norm") if ln_max_step_norm is not None else None

    router_keys = sorted(k for k in parent_buckets if k.startswith("router_frag_"))
    if not router_keys:
        raise ValueError("parent_buckets must contain at least one 'router_frag_*' entry")
    last_r_key = router_keys[-1]
    if parent_buckets[last_r_key].numel() < 2 * h_dim:
        raise ValueError(
            f"Final router fragment {last_r_key!r} has length {parent_buckets[last_r_key].numel()} < 2*hidden_size ({2 * h_dim})"
        )

    norm_w = _normalize_peer_weights(peer_weights, len(peer_buckets_list))
    gen = torch.Generator(device="cpu").manual_seed(seed_i)
    keep_prob = 1.0 - p_drop

    merged_frags: dict[str, torch.Tensor] = {}
    router_consensus_ratios: list[float] = []

    for r_key in router_keys:
        p_full = _validate_finite_tensor(parent_buckets[r_key], f"parent_buckets[{r_key!r}]", ndim=1)
        is_last = r_key == last_r_key
        p_router = p_full[: -2 * h_dim] if is_last else p_full

        peer_deltas: list[torch.Tensor] = []
        active_weights: list[float] = []
        ln_w_deltas: list[torch.Tensor] = []
        ln_b_deltas: list[torch.Tensor] = []

        for p_idx, peer_b in enumerate(peer_buckets_list):
            if r_key not in peer_b:
                continue
            c_full = _validate_finite_tensor(peer_b[r_key], f"peer_buckets[{p_idx}][{r_key!r}]", ndim=1)
            if c_full.shape != p_full.shape:
                raise ValueError(f"Shape mismatch on {r_key!r}: {tuple(c_full.shape)} vs {tuple(p_full.shape)}")
            c_router = c_full[: -2 * h_dim] if is_last else c_full
            delta_r = c_router - p_router
            if p_drop > 0.0:
                delta_r = delta_r * _sample_dare_mask(tuple(delta_r.shape), keep_prob, gen)
            peer_deltas.append(delta_r)
            active_weights.append(norm_w[p_idx])

            if is_last:
                p_ln_w, p_ln_b = p_full[-2 * h_dim : -h_dim], p_full[-h_dim:]
                c_ln_w, c_ln_b = c_full[-2 * h_dim : -h_dim], c_full[-h_dim:]
                ln_w_deltas.append(c_ln_w - p_ln_w)
                ln_b_deltas.append(c_ln_b - p_ln_b)

        if not peer_deltas:
            merged_frags[r_key] = parent_buckets[r_key].detach().clone()
            continue

        if p_router.numel() > 0:
            merged_r_delta, r_meta = trim_and_elect_sign_ties(
                peer_deltas,
                density_top_p=density_top_p,
                peer_weights=active_weights,
                sign_election_mode=sign_election_mode,
                max_peer_norm_ratio=max_peer_norm_ratio,
                scaling_lambda=scaling_lambda,
            )
            router_consensus_ratios.append(float(r_meta["sign_consensus_ratio"]))
            out_router = p_router + merged_r_delta
        else:
            out_router = p_router

        if not is_last:
            merged_frags[r_key] = out_router.to(
                device=parent_buckets[r_key].device, dtype=parent_buckets[r_key].dtype
            ).contiguous()
            continue

        p_ln_w = p_full[-2 * h_dim : -h_dim]
        p_ln_b = p_full[-h_dim:]
        if ln_tail_policy == "freeze_parent":
            cand_ln_w = p_ln_w.clone()
            cand_ln_b = p_ln_b.clone()
        else:
            aw = _normalize_peer_weights(active_weights, len(active_weights))
            dw = sum(aw[i] * ln_w_deltas[i] for i in range(len(aw)))
            db = sum(aw[i] * ln_b_deltas[i] for i in range(len(aw)))
            if ln_cap is not None:
                step_norm = float(torch.cat([dw, db]).norm().item())
                if step_norm > ln_cap and step_norm > 0.0:
                    scale = ln_cap / step_norm
                    dw = dw * scale
                    db = db * scale
            cand_ln_w = p_ln_w + dw
            cand_ln_b = p_ln_b + db

        combined = torch.cat([out_router, cand_ln_w, cand_ln_b], dim=0)
        merged_frags[r_key] = combined.to(
            device=parent_buckets[r_key].device, dtype=parent_buckets[r_key].dtype
        ).contiguous()

    parent_tail = parent_buckets[last_r_key][-2 * h_dim :].detach().to(torch.float64)
    cand_tail = merged_frags[last_r_key][-2 * h_dim :].detach().to(torch.float64)
    ln_drift_l2 = float((cand_tail - parent_tail).norm().item())

    return merged_frags, {
        "last_router_fragment": last_r_key,
        "ln_tail_policy": ln_tail_policy,
        "layernorm_tail_protected": True,
        "layernorm_drift_l2": ln_drift_l2,
        "router_sign_consensus_ratio": (
            float(np.mean(router_consensus_ratios)) if router_consensus_ratios else 1.0
        ),
    }


# ---------------------------------------------------------------------------
# (d) End-to-End OpenDiLoCo MoE Checkpoint Merger with Prose Guard & ANSE Gate
# ---------------------------------------------------------------------------


def compute_buckets_sha256(buckets: Mapping[str, torch.Tensor]) -> str:
    """Compute a deterministic SHA-256 digest over sorted bucket names, shapes, dtypes, and bytes."""
    if not isinstance(buckets, Mapping) or not buckets:
        raise ValueError("buckets must be a non-empty mapping")
    hasher = hashlib.sha256()
    for key in sorted(buckets.keys()):
        t = buckets[key].detach().cpu().contiguous()
        hasher.update(f"{key}|{tuple(t.shape)}|{t.dtype}|".encode())
        elem_bytes = t.element_size()
        view_dtype = (
            torch.uint8 if elem_bytes == 1 else (torch.int16 if elem_bytes == 2 else (torch.int32 if elem_bytes == 4 else torch.int64))
        )
        hasher.update(t.view(view_dtype).reshape(-1).view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def _attach_receipt_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attach deterministic canonical JSON ``receipt_sha256`` to ``payload``."""
    core = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    core["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    return core


def _extract_probe_ids_and_masks(
    probe_chunks: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Validate probe chunks and extract token IDs, general-domain mask, and domain names."""
    if isinstance(probe_chunks, (str, bytes)) or not isinstance(probe_chunks, Sequence) or not probe_chunks:
        raise ValueError("probe_chunks must be a non-empty sequence of chunk mappings")
    ids_all = torch.stack([torch.as_tensor(ch["ids"], dtype=torch.long) for ch in probe_chunks], dim=0)
    domains = [str(ch["expert_id"]) for ch in probe_chunks]
    is_e0 = np.array([d.startswith(GENERAL_EXPERT_PREFIX) for d in domains], dtype=bool)
    if not bool(np.any(is_e0)):
        raise ValueError("probe_chunks must include at least one general-domain ('E0_*') chunk")
    return ids_all, is_e0, domains


@torch.no_grad()
def _collect_layer_prose_and_spec_inputs(
    model: nn.Module,
    parent_buckets: Mapping[str, torch.Tensor],
    load_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None],
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    score_from: int,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    """Collect per-layer pre-MLP hidden states and SwiGLU intermediate states on ``probe_chunks``."""
    load_fn(model, parent_buckets)
    model.eval()
    ids_all, is_e0, _ = _extract_probe_ids_and_masks(probe_chunks)
    e0_idx = np.flatnonzero(is_e0).tolist()
    spec_idx = np.flatnonzero(~is_e0).tolist()

    start_pos = max(0, min(int(ids_all.shape[1]) - 2, score_from - 1))
    cos = model.cos_cached.to(ids_all.device)
    sin = model.sin_cached.to(ids_all.device)
    x_seq = torch.cat(
        [compute_embeddings_batch(model, ids_all[b0 : b0 + batch_size]) for b0 in range(0, ids_all.shape[0], batch_size)],
        dim=0,
    )

    layer_acts: list[dict[str, torch.Tensor]] = []
    for blk in model.blocks:
        xm_l: list[torch.Tensor] = []
        mi_l: list[torch.Tensor] = []
        for b0 in range(0, x_seq.shape[0], batch_size):
            xm, mi = forward_block_pre_mlp(blk, x_seq[b0 : b0 + batch_size], cos, sin)
            xm_l.append(xm)
            mi_l.append(mi)
        x_mid = torch.cat(xm_l, dim=0)
        mlp_in = torch.cat(mi_l, dim=0)
        scored = mlp_in[:, start_pos:-1, :]
        h_dim = int(scored.shape[-1])
        x_prose = scored[e0_idx].reshape(-1, h_dim)
        x_spec = scored[spec_idx].reshape(-1, h_dim) if spec_idx else None

        active_experts = sorted(int(k) for k in blk.mlp.experts.keys()) if isinstance(blk.mlp, CrowdLoRAMoEMLP) else []
        act_prose = (
            compute_swiglu_intermediate_states(blk.mlp, x_prose, active_experts)
            if active_experts
            else None
        )
        act_spec = (
            compute_swiglu_intermediate_states(blk.mlp, x_spec, active_experts)
            if active_experts and x_spec is not None
            else None
        )
        entry: dict[str, torch.Tensor] = {"x_prose": x_prose}
        if x_spec is not None:
            entry["x_spec"] = x_spec
        if act_prose is not None:
            entry["act_prose"] = act_prose
        if act_spec is not None:
            entry["act_spec"] = act_spec
        layer_acts.append(entry)

        x_seq = torch.cat(
            [forward_block_post_mlp(blk, x_mid[b0 : b0 + batch_size], mlp_in[b0 : b0 + batch_size]) for b0 in range(0, x_mid.shape[0], batch_size)],
            dim=0,
        )
    return layer_acts


@torch.no_grad()
def _evaluate_buckets_nll(
    model: nn.Module,
    buckets: Mapping[str, torch.Tensor],
    load_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None],
    *,
    probe_chunks: Sequence[Mapping[str, Any]] | None,
    eval_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], Mapping[str, Any]] | None,
    score_from: int,
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate per-domain finite NLLs via ``eval_fn`` or native ``CrowdV1WithEngram`` forward pass."""
    if eval_fn is not None:
        raw = eval_fn(model, buckets)
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("eval_fn must return a non-empty mapping")
        dom_source = raw.get("domain_nlls", raw)
        dom_map = {
            str(k): float(v)
            for k, v in dict(dom_source).items()
            if str(k) not in {"non_finite_outputs", "nll_mean", "all_finite"}
        }
        if not dom_map:
            raise ValueError("eval_fn result must include at least one domain NLL")
        nf_cnt = int(raw.get("non_finite_outputs", 0))
        all_finite = nf_cnt == 0 and all(math.isfinite(v) for v in dom_map.values())
        nll_mean = float(raw["nll_mean"]) if ("nll_mean" in raw and math.isfinite(float(raw["nll_mean"]))) else (
            float(np.mean(list(dom_map.values()))) if all_finite else None
        )
        return {
            "domain_nlls": {k: (v if math.isfinite(v) else None) for k, v in dom_map.items()},
            "all_finite": all_finite,
            "nll_mean": nll_mean,
            "chunk_losses": None,
        }

    assert probe_chunks is not None
    ids_all, _, domains = _extract_probe_ids_and_masks(probe_chunks)
    load_fn(model, buckets)
    model.eval()
    h_pre = forward_model_pre_ln(model, ids_all, batch_size=batch_size)
    if not bool(torch.isfinite(h_pre).all().item()):
        return {
            "domain_nlls": {d: None for d in sorted(set(domains))},
            "all_finite": False,
            "nll_mean": None,
            "chunk_losses": None,
        }
    nlls = per_chunk_nll_from_pre_ln(
        h_pre,
        model.ln.weight,
        model.ln.bias,
        float(getattr(model.ln, "eps", 1e-5)),
        model.head.weight,
        ids_all,
        score_from=score_from,
    )
    if not bool(torch.isfinite(nlls).all().item()):
        return {
            "domain_nlls": {d: None for d in sorted(set(domains))},
            "all_finite": False,
            "nll_mean": None,
            "chunk_losses": None,
        }
    nlls_np = nlls.detach().cpu().numpy().astype(np.float64)
    grouped: dict[str, list[float]] = {}
    for dom, val in zip(domains, nlls_np.tolist(), strict=True):
        grouped.setdefault(dom, []).append(float(val))
    dom_nlls = {dom: float(np.mean(vals)) for dom, vals in sorted(grouped.items())}
    return {
        "domain_nlls": dom_nlls,
        "all_finite": True,
        "nll_mean": float(nlls_np.mean()),
        "chunk_losses": nlls_np,
    }


def _slice_expert_bucket_views(
    model: nn.Module,
    bucket_vec: torch.Tensor,
    expert_idx: int,
) -> list[dict[str, torch.Tensor]]:
    """Unpack flat ``expert_{e}`` vector into per-layer ``w1/w2/w3`` ``A, B`` factor views."""
    vec64 = _validate_finite_tensor(bucket_vec, f"expert_{expert_idx}", ndim=1)
    offset = 0
    layers_out: list[dict[str, torch.Tensor]] = []
    for blk in model.blocks:
        mlp = getattr(blk, "mlp", None)
        if not isinstance(mlp, CrowdLoRAMoEMLP) or str(expert_idx) not in mlp.experts:
            layers_out.append({})
            continue
        exp_mod = mlp.experts[str(expert_idx)]
        layer_factors: dict[str, torch.Tensor] = {}
        for proj in _SWIGLU_PROJECTIONS:
            for suffix in ("A", "B"):
                attr = f"{proj}_{suffix}"
                param = getattr(exp_mod, attr)
                if not param.requires_grad:
                    continue
                n = param.numel()
                if offset + n > vec64.numel():
                    raise ValueError(
                        f"expert_{expert_idx} bucket length ({vec64.numel()}) smaller than model layout"
                    )
                layer_factors[attr] = vec64[offset : offset + n].view_as(param)
                offset += n
        layers_out.append(layer_factors)
    if offset != vec64.numel():
        raise ValueError(
            f"expert_{expert_idx} bucket length ({vec64.numel()}) does not match model layout ({offset})"
        )
    return layers_out


def _pack_expert_bucket_views(
    model: nn.Module,
    layers_views: Sequence[Mapping[str, torch.Tensor]],
    expert_idx: int,
    ref_tensor: torch.Tensor,
) -> torch.Tensor:
    """Re-pack per-layer ``w1/w2/w3`` ``A, B`` factor tensors into contiguous OpenDiLoCo bucket."""
    parts: list[torch.Tensor] = []
    for l_idx, blk in enumerate(model.blocks):
        mlp = getattr(blk, "mlp", None)
        if not isinstance(mlp, CrowdLoRAMoEMLP) or str(expert_idx) not in mlp.experts:
            continue
        exp_mod = mlp.experts[str(expert_idx)]
        l_map = layers_views[l_idx]
        for proj in _SWIGLU_PROJECTIONS:
            for suffix in ("A", "B"):
                attr = f"{proj}_{suffix}"
                if getattr(exp_mod, attr).requires_grad:
                    parts.append(l_map[attr].detach().to(torch.float64).reshape(-1))
    return torch.cat(parts, dim=0).to(device=ref_tensor.device, dtype=ref_tensor.dtype).contiguous()


def _project_router_deltas_orthogonal(
    model: nn.Module,
    parent_buckets: Mapping[str, torch.Tensor],
    cand_frags: dict[str, torch.Tensor],
    proj_x_per_layer: Sequence[torch.Tensor],
    *,
    num_router_fragments: int,
    protect_general_expert: bool,
    general_expert_index: int,
) -> int:
    """Project merged ``router_frag_*`` gate deltas orthogonal to per-layer prose modes."""
    n_blocks = len(model.blocks)
    frag_span = max(1, math.ceil(n_blocks / num_router_fragments))
    h_dim = int(model.ln.weight.numel())
    last_r_key = sorted(k for k in parent_buckets if k.startswith("router_frag_"))[-1]

    # Copy out candidate router fragments in float64 to slice block-by-block
    cand_copy = {k: v.detach().to(torch.float64).clone() for k, v in cand_frags.items()}
    parent_copy = {k: v.detach().to(torch.float64) for k, v in parent_buckets.items()}
    offsets = {k: 0 for k in cand_copy}
    projected_rows = 0

    for l_idx, blk in enumerate(model.blocks):
        f_idx = min(num_router_fragments - 1, l_idx // frag_span)
        r_key = f"router_frag_{f_idx}"
        mlp = getattr(blk, "mlp", None)
        if not isinstance(mlp, CrowdLoRAMoEMLP) or not mlp.router.weight.requires_grad or r_key not in cand_copy:
            continue
        r_shape = tuple(mlp.router.weight.shape)
        n = int(mlp.router.weight.numel())
        off = offsets[r_key]
        offsets[r_key] = off + n

        p_w = parent_copy[r_key][off : off + n].view(r_shape)
        c_w = cand_copy[r_key][off : off + n].view(r_shape)
        delta_w = c_w - p_w
        if float(delta_w.abs().max().item()) <= 1e-12:
            continue

        g_x = proj_x_per_layer[l_idx]
        for row_idx in range(r_shape[0]):
            if protect_general_expert and row_idx == general_expert_index:
                continue
            delta_w[row_idx] = g_x.T @ delta_w[row_idx]
            projected_rows += 1
        cand_copy[r_key][off : off + n] = (p_w + delta_w).reshape(-1)

    for r_key, vec in cand_copy.items():
        if r_key == last_r_key:
            # Ensure protected LayerNorm tail stays intact
            vec[-2 * h_dim :] = cand_frags[r_key][-2 * h_dim :].detach().to(torch.float64)
        cand_frags[r_key] = vec.to(device=parent_buckets[r_key].device, dtype=parent_buckets[r_key].dtype).contiguous()
    return projected_rows


@torch.no_grad()
def merge_moe_checkpoints_with_prose_guard(
    model: nn.Module,
    parent_buckets: Mapping[str, torch.Tensor],
    peer_buckets_list: Sequence[Mapping[str, torch.Tensor]],
    *,
    probe_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], Mapping[str, Any]] | None = None,
    prose_activations: Sequence[torch.Tensor] | None = None,
    spec_activations: Sequence[torch.Tensor] | None = None,
    config: TIESDAREMergerConfig | None = None,
    peer_weights: Sequence[float] | None = None,
    num_router_fragments: int | None = None,
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
    extract_buckets_fn: Callable[[nn.Module], dict[str, torch.Tensor]] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Merge distributed OpenDiLoCo MoE LoRA peer checkpoints with prose-nullspace guard & ANSE rollback.

    Executes:
    1. Atomic parent SHA-256 snapshotting.
    2. Protected ``router_frag_*`` merging (shielding the final ``2*H`` LayerNorm tail from
       TIES trimming and DARE dropout).
    3. Per-layer, per-expert SwiGLU LoRA merging via TIES sign-consensus + deterministic DARE
       rescaling + exact thin QR-SVD compression preserving exact adapter rank ``r``.
    4. Prose-nullspace projection using Ledoit-Wolf shrunk ``M_prose`` on input and intermediate
       SwiGLU activations (and optionally router gate deltas).
    5. Guard-calibrated LayerNorm tail verification when ``probe_chunks`` are supplied.
    6. ANSE ``<= +1.0%`` general-domain guard + energy improvement acceptance check with atomic
       bitwise parent rollback and SHA-256 provenance receipt.
    """
    if not isinstance(model, nn.Module) or not hasattr(model, "blocks") or not hasattr(model, "ln"):
        raise TypeError("model must be a CrowdV1WithEngram-compatible nn.Module with blocks and ln")
    if not isinstance(parent_buckets, Mapping) or not parent_buckets:
        raise ValueError("parent_buckets must be a non-empty mapping")
    if isinstance(peer_buckets_list, (str, bytes, Mapping)) or not isinstance(peer_buckets_list, Sequence) or not peer_buckets_list:
        raise ValueError("peer_buckets_list must be a non-empty sequence of bucket mappings")
    if probe_chunks is None and eval_fn is None:
        raise ValueError("Either probe_chunks or eval_fn must be provided to enforce the ANSE guard")
    if config is not None and not isinstance(config, TIESDAREMergerConfig):
        raise TypeError(f"config must be a TIESDAREMergerConfig or None, got {type(config).__name__}")
    cfg = config or TIESDAREMergerConfig()

    n_blocks = len(model.blocks)
    if prose_activations is not None:
        if isinstance(prose_activations, (str, bytes)) or len(prose_activations) != n_blocks:
            raise ValueError(f"prose_activations must match model block count ({n_blocks})")
        if spec_activations is not None and (isinstance(spec_activations, (str, bytes)) or len(spec_activations) != n_blocks):
            raise ValueError(f"spec_activations must match model block count ({n_blocks})")

    for key, t in parent_buckets.items():
        _validate_finite_tensor(t, f"parent_buckets[{key!r}]")
    for p_idx, p_map in enumerate(peer_buckets_list):
        if not isinstance(p_map, Mapping) or not p_map:
            raise ValueError(f"peer_buckets_list[{p_idx}] must be a non-empty mapping")
        for key, t in p_map.items():
            _validate_finite_tensor(t, f"peer_buckets_list[{p_idx}][{key!r}]")

    nfrag = (
        _validate_positive_int(num_router_fragments, "num_router_fragments")
        if num_router_fragments is not None
        else max(1, len([k for k in parent_buckets if k.startswith("router_frag_")]))
    )
    load_fn = load_buckets_fn or (lambda m, b: load_moe_trainable_buckets(m, b, num_router_fragments=nfrag))
    _ = extract_buckets_fn or (lambda m: extract_moe_trainable_buckets(m, num_router_fragments=nfrag))

    parent_clone = {k: v.detach().cpu().clone() for k, v in parent_buckets.items()}
    parent_sha = compute_buckets_sha256(parent_clone)
    h_dim = int(model.ln.weight.numel())

    # Evaluate parent baseline
    parent_eval = _evaluate_buckets_nll(
        model,
        parent_clone,
        load_fn,
        probe_chunks=probe_chunks,
        eval_fn=eval_fn,
        score_from=cfg.score_from,
        batch_size=cfg.batch_size,
    )
    if not parent_eval["all_finite"] or parent_eval["nll_mean"] is None:
        raise ValueError("parent_buckets produced non-finite evaluation NLL")

    # Gather prose & specialist activation statistics per layer
    layer_acts: list[dict[str, torch.Tensor]] | None = None
    if prose_activations is not None:
        if isinstance(prose_activations, (str, bytes)) or len(prose_activations) != n_blocks:
            raise ValueError(f"prose_activations must match model block count ({n_blocks})")
        if spec_activations is not None and (isinstance(spec_activations, (str, bytes)) or len(spec_activations) != n_blocks):
            raise ValueError(f"spec_activations must match model block count ({n_blocks})")
        layer_acts = []
        for l_idx in range(n_blocks):
            entry = {"x_prose": _validate_finite_tensor(prose_activations[l_idx], f"prose_activations[{l_idx}]", ndim=2)}
            if spec_activations is not None:
                entry["x_spec"] = _validate_finite_tensor(spec_activations[l_idx], f"spec_activations[{l_idx}]", ndim=2)
            layer_acts.append(entry)
    elif probe_chunks is not None:
        layer_acts = _collect_layer_prose_and_spec_inputs(
            model,
            parent_clone,
            load_fn,
            probe_chunks,
            score_from=cfg.score_from,
            batch_size=cfg.batch_size,
        )

    # Compute per-layer prose nullspace projectors
    proj_x_list: list[torch.Tensor] = []
    proj_act_list: list[torch.Tensor | None] = []
    projected_modes_total = 0

    if layer_acts is not None and cfg.prose_projection_mode != "none":
        for l_idx in range(n_blocks):
            la = layer_acts[l_idx]
            gx, gx_meta = compute_prose_nullspace_projector(
                la["x_prose"],
                la.get("x_spec"),
                mode=cfg.prose_projection_mode,
                eigen_ratio_tau=cfg.prose_eigen_ratio_tau,
                variance_threshold=cfg.prose_variance_threshold,
                max_modes=cfg.prose_max_modes,
                proj_strength=cfg.prose_proj_strength,
                eta=cfg.alphaedit_eta,
                reg_spec=cfg.alphaedit_reg_spec,
                max_operator_norm=cfg.alphaedit_max_operator_norm,
            )
            proj_x_list.append(gx)
            projected_modes_total += int(gx_meta["projected_modes"])

            if "act_prose" in la:
                ga, ga_meta = compute_prose_nullspace_projector(
                    la["act_prose"],
                    la.get("act_spec"),
                    mode=cfg.prose_projection_mode,
                    eigen_ratio_tau=cfg.prose_eigen_ratio_tau,
                    variance_threshold=cfg.prose_variance_threshold,
                    max_modes=cfg.prose_max_modes,
                    proj_strength=cfg.prose_proj_strength,
                    eta=cfg.alphaedit_eta,
                    reg_spec=cfg.alphaedit_reg_spec,
                    max_operator_norm=cfg.alphaedit_max_operator_norm,
                )
                proj_act_list.append(ga)
                projected_modes_total += int(ga_meta["projected_modes"])
            else:
                proj_act_list.append(None)

    # Step 1: Protected router_frag merge (with LN tail separation)
    router_density = cfg.router_density_top_p if cfg.router_density_top_p is not None else cfg.density_top_p
    cand_frags, ln_meta = protect_router_frag_layernorm_tail(
        parent_clone,
        peer_buckets_list,
        hidden_size=h_dim,
        density_top_p=router_density,
        drop_rate=cfg.router_drop_rate,
        seed=cfg.seed,
        peer_weights=peer_weights,
        sign_election_mode=cfg.sign_election_mode,
        max_peer_norm_ratio=cfg.max_peer_norm_ratio,
        ln_tail_policy=cfg.ln_tail_policy,
        ln_max_step_norm=cfg.ln_max_step_norm,
        scaling_lambda=cfg.scaling_lambda,
    )
    if proj_x_list and cfg.project_router_delta_orthogonal:
        _project_router_deltas_orthogonal(
            model,
            parent_clone,
            cand_frags,
            proj_x_list,
            num_router_fragments=nfrag,
            protect_general_expert=cfg.protect_general_expert,
            general_expert_index=cfg.general_expert_index,
        )

    cand_buckets: dict[str, torch.Tensor] = dict(cand_frags)
    consensus_ratios: list[float] = [float(ln_meta["router_sign_consensus_ratio"])]
    expert_ranks_preserved: dict[str, int] = {}

    # Step 2: Merge each expert_{e} bucket with TIES + DARE + QR-SVD + Prose Nullspace
    norm_w_all = _normalize_peer_weights(peer_weights, len(peer_buckets_list))
    expert_keys = sorted(
        (k for k in parent_clone if k.startswith("expert_")),
        key=lambda name: int(name.split("_", 1)[1]),
    )
    for e_key in expert_keys:
        e_idx = int(e_key.split("_", 1)[1])
        parent_layers = _slice_expert_bucket_views(model, parent_clone[e_key], e_idx)

        active_peers: list[tuple[int, list[dict[str, torch.Tensor]], float]] = []
        for p_idx, peer_b in enumerate(peer_buckets_list):
            if e_key not in peer_b:
                continue
            p_vec = _validate_finite_tensor(peer_b[e_key], f"peer[{p_idx}][{e_key!r}]", ndim=1)
            if p_vec.shape != parent_clone[e_key].shape:
                raise ValueError(f"Shape mismatch on peer[{p_idx}][{e_key!r}]")
            p_layers = _slice_expert_bucket_views(model, p_vec, e_idx)
            # Detect whether this peer actually contributed a non-zero update on this expert
            has_delta = any(
                float((p_layers[l_idx][attr] - parent_layers[l_idx][attr]).abs().max().item()) > 1e-12
                for l_idx in range(n_blocks)
                for attr in parent_layers[l_idx]
            )
            if has_delta:
                active_peers.append((p_idx, p_layers, norm_w_all[p_idx]))

        if not active_peers:
            cand_buckets[e_key] = parent_clone[e_key].clone()
            continue

        merged_layers: list[dict[str, torch.Tensor]] = []
        active_w_list = [item[2] for item in active_peers]
        max_rank_seen = 0

        for l_idx in range(n_blocks):
            p_map = parent_layers[l_idx]
            if not p_map:
                merged_layers.append({})
                continue
            out_map: dict[str, torch.Tensor] = {}
            for proj_idx, proj in enumerate(_SWIGLU_PROJECTIONS):
                a_key, b_key = f"{proj}_A", f"{proj}_B"
                a_parent, b_parent = p_map[a_key], p_map[b_key]
                r_cur = int(a_parent.shape[0])
                r_tgt = r_cur if cfg.target_rank is None else cfg.target_rank
                if r_tgt > r_cur:
                    raise ValueError(
                        f"target_rank ({r_tgt}) cannot exceed model adapter rank ({r_cur}) without rank inflation"
                    )
                max_rank_seen = max(max_rank_seen, r_tgt)

                if len(active_peers) == 1:
                    only_layers = active_peers[0][1][l_idx]
                    b_m = only_layers[b_key].clone()
                    a_m = only_layers[a_key].clone()
                    if r_tgt != r_cur:
                        b_m, a_m, _ = _qr_svd_factor_pair(b_m, a_m, r_tgt)
                else:
                    peer_pairs = [(item[1][l_idx][b_key], item[1][l_idx][a_key]) for item in active_peers]
                    (b_m, a_m), d_meta = dare_rescale_and_svd_compress_lora(
                        peer_pairs,
                        drop_rate=cfg.drop_rate,
                        target_rank=r_tgt,
                        seed=cfg.seed + l_idx * 131 + e_idx * 37 + proj_idx * 7,
                        peer_weights=active_w_list,
                        base_pair=(b_parent, a_parent),
                        scaling_lambda=cfg.scaling_lambda,
                        ties_density_top_p=cfg.density_top_p,
                        sign_election_mode=cfg.sign_election_mode,
                        max_peer_norm_ratio=cfg.max_peer_norm_ratio,
                    )
                    consensus_ratios.append(float(d_meta["sign_consensus_ratio"]))

                # Project merged adapter delta orthogonal to general-prose principal modes
                apply_proj = not (cfg.protect_general_expert and e_idx == cfg.general_expert_index)
                if apply_proj and proj_x_list:
                    g_proj = proj_x_list[l_idx] if proj in ("w1", "w2") else proj_act_list[l_idx]
                    if g_proj is not None:
                        b_m, a_m = apply_prose_projector_to_lora_pair(
                            b_m,
                            a_m,
                            g_proj,
                            base_pair=(b_parent, a_parent),
                            target_rank=r_tgt,
                        )
                if r_tgt < r_cur:
                    # Zero-pad inactive singular slots to preserve exact OpenDiLoCo bucket tensor shape
                    b_pad = torch.zeros((b_m.shape[0], r_cur), dtype=b_m.dtype, device=b_m.device)
                    a_pad = torch.zeros((r_cur, a_m.shape[1]), dtype=a_m.dtype, device=a_m.device)
                    b_pad[:, :r_tgt] = b_m
                    a_pad[:r_tgt, :] = a_m
                    b_m, a_m = b_pad, a_pad
                out_map[a_key] = a_m
                out_map[b_key] = b_m
            merged_layers.append(out_map)

        expert_ranks_preserved[e_key] = max_rank_seen
        cand_buckets[e_key] = _pack_expert_bucket_views(model, merged_layers, e_idx, parent_clone[e_key])

    # Step 3: Guard-calibrated LayerNorm tail refinement on native probe states if enabled
    last_r_key = str(ln_meta["last_router_fragment"])
    if cfg.ln_tail_policy == "guard_calibrated" and probe_chunks is not None and parent_eval["chunk_losses"] is not None:
        ids_all, is_e0, _ = _extract_probe_ids_and_masks(probe_chunks)
        load_fn(model, cand_buckets)
        model.eval()
        h_cand_pre = forward_model_pre_ln(model, ids_all, batch_size=cfg.batch_size)
        if bool(torch.isfinite(h_cand_pre).all().item()):
            p_ln_w = parent_clone[last_r_key][-2 * h_dim : -h_dim].to(torch.float32)
            p_ln_b = parent_clone[last_r_key][-h_dim:].to(torch.float32)
            c_ln_w = cand_buckets[last_r_key][-2 * h_dim : -h_dim].to(torch.float32)
            c_ln_b = cand_buckets[last_r_key][-h_dim:].to(torch.float32)
            best_lam, _ = select_guard_optimal_lambda_ln(
                h_cand_pre,
                p_ln_w,
                p_ln_b,
                c_ln_w,
                c_ln_b,
                float(getattr(model.ln, "eps", 1e-5)),
                model.head.weight,
                ids_all,
                parent_eval["chunk_losses"],
                is_e0,
                score_from=cfg.score_from,
                target_e0_max_ratio=min(LN_GUARD_MAX_RATIO, cfg.domain_guard_max_ratio),
            )
            cal_w = (p_ln_w + best_lam * (c_ln_w - p_ln_w)).to(cand_buckets[last_r_key].dtype)
            cal_b = (p_ln_b + best_lam * (c_ln_b - p_ln_b)).to(cand_buckets[last_r_key].dtype)
            frag_vec = cand_buckets[last_r_key].clone()
            frag_vec[-2 * h_dim : -h_dim] = cal_w
            frag_vec[-h_dim:] = cal_b
            cand_buckets[last_r_key] = frag_vec.contiguous()

    cand_sha = compute_buckets_sha256(cand_buckets)
    parent_tail = parent_clone[last_r_key][-2 * h_dim :].to(torch.float64)
    cand_tail = cand_buckets[last_r_key][-2 * h_dim :].to(torch.float64)
    final_ln_drift_l2 = float((cand_tail - parent_tail).norm().item())

    # Step 4: Evaluate candidate under ANSE <= +1.0% general-domain guard
    cand_eval = _evaluate_buckets_nll(
        model,
        cand_buckets,
        load_fn,
        probe_chunks=probe_chunks,
        eval_fn=eval_fn,
        score_from=cfg.score_from,
        batch_size=cfg.batch_size,
    )

    parent_dom_nlls: dict[str, float] = {k: float(v) for k, v in parent_eval["domain_nlls"].items()}
    cand_dom_nlls_raw: Mapping[str, float | None] = cand_eval["domain_nlls"]
    per_domain_drift_pct: dict[str, float | None] = {}
    general_ratios: list[float] = []
    all_ratios: list[float] = []

    for dom, p_val in parent_dom_nlls.items():
        c_val = cand_dom_nlls_raw.get(dom)
        if c_val is None or not math.isfinite(float(c_val)):
            per_domain_drift_pct[dom] = None
            continue
        ratio = float(c_val) / max(abs(p_val), 1e-8)
        per_domain_drift_pct[dom] = 100.0 * (ratio - 1.0)
        all_ratios.append(ratio)
        if dom.startswith(GENERAL_EXPERT_PREFIX):
            general_ratios.append(ratio)

    all_finite = bool(cand_eval["all_finite"]) and len(all_ratios) == len(parent_dom_nlls)
    general_ratio = max(general_ratios) if general_ratios else (max(all_ratios) if all_ratios else 999.0)
    worst_ratio = max(all_ratios) if all_ratios else 999.0

    general_guard_ok = all_finite and (general_ratio <= cfg.domain_guard_max_ratio)
    all_dom_guard_ok = all_finite and (worst_ratio <= cfg.domain_guard_max_ratio)
    guard_ok = bool(general_guard_ok and (all_dom_guard_ok if cfg.enforce_all_domain_guard else True))

    parent_nll_mean = float(parent_eval["nll_mean"])
    cand_nll_mean = float(cand_eval["nll_mean"]) if (all_finite and cand_eval["nll_mean"] is not None) else None
    parent_anse_energy = parent_nll_mean

    if not all_finite or not guard_ok or cand_nll_mean is None:
        cand_anse_energy = ANSE_MAXIMUM_PAIN
    else:
        soft_pen = 0.05 if general_ratio > cfg.anse_safe_e0_ratio else 0.0
        cand_anse_energy = cand_nll_mean + soft_pen

    delta_energy = cand_anse_energy - parent_anse_energy
    energy_ok = (delta_energy < 0.0) if cfg.require_energy_improvement else (delta_energy <= 0.0)
    accepted = bool(all_finite and guard_ok and energy_ok)

    if accepted:
        load_fn(model, cand_buckets)
        final_buckets = {k: v.detach().cpu().clone() for k, v in cand_buckets.items()}
        final_sha = compute_buckets_sha256(final_buckets)
        reject_reason = None
        rollback_verified = False
        final_eval_nll = cand_nll_mean
        final_anse_energy = cand_anse_energy
    else:
        load_fn(model, parent_clone)
        final_buckets = {k: v.detach().cpu().clone() for k, v in parent_clone.items()}
        final_sha = compute_buckets_sha256(final_buckets)
        rollback_verified = final_sha == parent_sha
        if not rollback_verified:
            raise RuntimeError(
                f"Atomic parent rollback SHA-256 mismatch: expected {parent_sha}, got {final_sha}"
            )
        reject_reason = (
            "non_finite_candidate"
            if not all_finite
            else ("domain_guard_exceeded" if not guard_ok else "non_improving_anse_energy")
        )
        final_eval_nll = parent_nll_mean
        final_anse_energy = parent_anse_energy

    gen_parent_vals = [v for k, v in parent_dom_nlls.items() if k.startswith(GENERAL_EXPERT_PREFIX)]
    gen_cand_vals = [
        float(cand_dom_nlls_raw[k])
        for k in parent_dom_nlls
        if k.startswith(GENERAL_EXPERT_PREFIX) and cand_dom_nlls_raw.get(k) is not None
    ]
    general_nll_parent = float(np.mean(gen_parent_vals)) if gen_parent_vals else parent_nll_mean
    general_nll_candidate = (
        float(np.mean(gen_cand_vals))
        if (all_finite and len(gen_cand_vals) == len(gen_parent_vals) and gen_cand_vals)
        else (cand_nll_mean if not gen_parent_vals else None)
    )

    receipt_body: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "status": "accepted" if accepted else "rolled_back",
        "accepted": accepted,
        "rolled_back": not accepted,
        "rollback_bitwise_verified": rollback_verified,
        "reject_reason": reject_reason,
        "peer_count": len(peer_buckets_list),
        "parent_sha256": parent_sha,
        "candidate_sha256": cand_sha,
        "final_sha256": final_sha,
        "nll_mean": final_eval_nll,
        "parent_nll_mean": parent_nll_mean,
        "candidate_nll_mean": cand_nll_mean,
        "general_nll_parent": general_nll_parent,
        "general_nll_candidate": general_nll_candidate,
        "general_drift_pct": 100.0 * (general_ratio - 1.0) if all_finite else None,
        "worst_domain_drift_pct": 100.0 * (worst_ratio - 1.0) if all_finite else None,
        "per_domain_nll_parent": parent_dom_nlls,
        "per_domain_nll_candidate": {k: (float(v) if v is not None else None) for k, v in cand_dom_nlls_raw.items()},
        "per_domain_drift_pct": per_domain_drift_pct,
        "guard_ok": guard_ok,
        "anse_energy": float(final_anse_energy),
        "candidate_anse_energy": float(cand_anse_energy),
        "parent_anse_energy": float(parent_anse_energy),
        "delta_anse_energy": float(delta_energy),
        "sign_consensus_ratio": float(np.mean(consensus_ratios)),
        "layernorm_tail_protected": bool(ln_meta["layernorm_tail_protected"]),
        "layernorm_drift_l2": final_ln_drift_l2 if accepted else 0.0,
        "candidate_layernorm_drift_l2": final_ln_drift_l2,
        "prose_nullspace_modes_projected": projected_modes_total,
        "expert_ranks_preserved": expert_ranks_preserved,
        "config": asdict(cfg),
    }
    return final_buckets, _attach_receipt_sha256(receipt_body)


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "DEFAULT_ALPHAEDIT_ETA",
    "DEFAULT_ALPHAEDIT_REG_SPEC",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_DARE_DROP_RATE",
    "DEFAULT_DENSITY_TOP_P",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_MAX_OPERATOR_NORM",
    "DEFAULT_PROSE_EIGEN_RATIO_TAU",
    "DEFAULT_PROSE_MAX_MODES",
    "DEFAULT_PROSE_PROJ_STRENGTH",
    "DEFAULT_PROSE_VARIANCE_THRESHOLD",
    "SCHEMA_VERSION",
    "TIESDAREMergerConfig",
    "apply_prose_projector_to_lora_pair",
    "compute_buckets_sha256",
    "compute_prose_nullspace_projector",
    "dare_rescale_and_svd_compress_lora",
    "merge_moe_checkpoints_with_prose_guard",
    "merge_swiglu_lora_adapters_dare_svd",
    "protect_router_frag_layernorm_tail",
    "trim_and_elect_sign_ties",
]

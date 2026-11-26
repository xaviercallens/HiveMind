# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Null-Space Protected Post-JEPA Gelee Royale LoRA Rescuer with RankMe Isotropy Guard.

Implements Paper v4.4.0 Section 15.4 Opportunity J1 (Theorem 15.1(2), DOI
``10.5281/zenodo.24553269``) strictly on CPU (0 MiB GPU VRAM, 0 gradient steps,
0 network/subprocess calls):

1. Given a post-round LLM-JEPA or Semantic Tube Prediction (STP) LoRA adapter
   ``(B, A, scale)`` that regressed general prose or far-token loss (e.g., ``P1``
   ``+0.90`` nat or ``E-STP-1b`` ``+1.37`` nat), constructs or validates the exact
   orthonormal prose-nullspace projector ``P_0 = I - V_prose^T V_prose``.
2. Applies Reine-Cortex's associative rank-preserving projection
   ``(s * B)(A @ P_0)`` in ``float64`` without materializing unprojected dense
   ``B @ A`` intermediates, algebraically annihilating general-prose activation
   drift (``||((s * B)(A @ P_0)) V_prose^T||_F < 1e-6``) while preserving specialist
   JEPA alignment in the orthogonal complement.
3. Enforces RankMe effective-rank isotropy (Garrido et al., ICML 2023) and
   cosine-collapse guarding (``1 - cos > threshold``) alongside the ANSE
   thermodynamic acceptance gate (``Delta E < 0`` and ``D0 <= +1.0%``), with
   atomic bitwise rollback (``ANSE_MAXIMUM_PAIN = 1e6``) and canonical SHA-256
   provenance receipts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

RESCUER_SCHEMA: str = "hivemind.nullspace_jepa_rescuer/1"
RESCUER_RECEIPT_OBJECT: str = "hivemind.nullspace_jepa_rescue_receipt"

ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_GENERAL_DOMAIN: str = "D0_general_prose"
DEFAULT_MAX_GENERAL_DRIFT_RATIO: float = 0.010
DEFAULT_MAX_DOMAIN_DRIFT_RATIO: float = 0.010
DEFAULT_ANNIHILATION_TOL: float = 1e-6
DEFAULT_MIN_ONE_MINUS_COS: float = 0.05
DEFAULT_MIN_JEPA_ONE_MINUS_COS: float = 1e-3
DEFAULT_MIN_NORMALIZED_RANKME: float = 0.15
DEFAULT_MIN_RANKME_RATIO: float = 0.75
DEFAULT_ECE_WEIGHT: float = 0.25
DEFAULT_ECE_BINS: int = 10
DEFAULT_RMSNORM_EPS: float = 1e-6
DEFAULT_ALPHA_GRID: tuple[float, ...] = (1.0, 0.75, 0.50, 0.25)
DEFAULT_SAFE_TAU_GRID: tuple[float, ...] = (1.0, 1.05, 1.10, 1.20)
VALID_JEPA_MODES: tuple[str, ...] = ("pairs", "stp")

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# 1. Validation, Hashing & Configuration
# ---------------------------------------------------------------------------


def _require_finite_float(
    value: Any,
    name: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    strict_min: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if min_value is not None:
        if strict_min and out <= min_value:
            raise ValueError(f"{name} must be > {min_value}, got {out}")
        if not strict_min and out < min_value:
            raise ValueError(f"{name} must be >= {min_value}, got {out}")
    if max_value is not None and out > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {out}")
    return out


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


def _require_fp_tensor(
    tensor: Any,
    name: str,
    *,
    ndim: int | None = None,
    min_ndim: int | None = None,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating-point, got {tensor.dtype}")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {tuple(tensor.shape)}")
    if min_ndim is not None and tensor.ndim < min_ndim:
        raise ValueError(f"{name} must have ndim >= {min_ndim}, got shape {tuple(tensor.shape)}")
    cpu_t = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(cpu_t).all().item()):
        raise ValueError(f"{name} contains non-finite values")
    return cpu_t


def _require_float_grid(
    values: Sequence[float],
    name: str,
    *,
    min_value: float,
    strict_min: bool = False,
) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError(f"{name} must be a non-empty sequence of numbers")
    return tuple(
        _require_finite_float(v, f"{name}[{idx}]", min_value=min_value, strict_min=strict_min)
        for idx, v in enumerate(values)
    )


def compute_payload_sha256(payload: Any) -> str:
    """Return canonical lowercase SHA-256 hex digest for a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Compute a deterministic bitwise SHA-256 digest over a tensor mapping."""
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("state_dict must be a non-empty mapping of tensors")
    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        if not isinstance(key, str) or not key:
            raise ValueError(f"state_dict key must be a non-empty string, got {key!r}")
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] must be a torch.Tensor")
        cpu_t = tensor.detach().to(device="cpu").contiguous()
        if cpu_t.is_floating_point() and not bool(torch.isfinite(cpu_t).all().item()):
            raise ValueError(f"state_dict[{key!r}] contains non-finite values")
        raw_bytes = np.ascontiguousarray(cpu_t.view(torch.uint8).numpy()).tobytes()
        digest.update(f"{key}:{cpu_t.dtype}:{tuple(cpu_t.shape)}:".encode())
        digest.update(raw_bytes)
    return digest.hexdigest()


@dataclass(frozen=True)
class JepaLoRADelta:
    """Validated post-round LLM-JEPA or STP LoRA adapter ``(B, A, scale)``."""

    b_mat: torch.Tensor
    a_mat: torch.Tensor
    scale: float = 1.0
    objective_mode: str = "pairs"
    adapter_name: str = "jepa_round"

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_name, str) or not self.adapter_name.strip():
            raise ValueError("adapter_name must be a non-empty string")
        if self.objective_mode not in VALID_JEPA_MODES:
            raise ValueError(f"objective_mode must be one of {VALID_JEPA_MODES}, got {self.objective_mode!r}")
        b_t = _require_fp_tensor(self.b_mat, f"{self.adapter_name}.b_mat", ndim=2)
        a_t = _require_fp_tensor(self.a_mat, f"{self.adapter_name}.a_mat", ndim=2)
        if b_t.shape[1] != a_t.shape[0]:
            raise ValueError(
                f"{self.adapter_name} rank mismatch: b_mat {tuple(b_t.shape)} vs a_mat {tuple(a_t.shape)}"
            )
        _require_finite_float(self.scale, f"{self.adapter_name}.scale")


@dataclass(frozen=True)
class NullspaceJepaRescueConfig:
    """Configuration governing null-space JEPA projection, RankMe guard, and ANSE gating."""

    general_domain_name: str = DEFAULT_GENERAL_DOMAIN
    max_general_drift_ratio: float = DEFAULT_MAX_GENERAL_DRIFT_RATIO
    max_domain_drift_ratio: float = DEFAULT_MAX_DOMAIN_DRIFT_RATIO
    annihilation_tol: float = DEFAULT_ANNIHILATION_TOL
    min_one_minus_cos: float = DEFAULT_MIN_ONE_MINUS_COS
    min_jepa_one_minus_cos: float = DEFAULT_MIN_JEPA_ONE_MINUS_COS
    min_normalized_rankme: float = DEFAULT_MIN_NORMALIZED_RANKME
    min_rankme_ratio: float = DEFAULT_MIN_RANKME_RATIO
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID
    safe_tau_grid: tuple[float, ...] = DEFAULT_SAFE_TAU_GRID
    ece_weight: float = DEFAULT_ECE_WEIGHT
    ece_bins: int = DEFAULT_ECE_BINS
    rmsnorm_eps: float = DEFAULT_RMSNORM_EPS
    residual_connection: bool | None = None
    residual_scale: float = 1.0
    min_energy_improvement: float = 1e-8

    def __post_init__(self) -> None:
        if not isinstance(self.general_domain_name, str) or not self.general_domain_name.strip():
            raise ValueError("general_domain_name must be a non-empty string")
        _require_finite_float(self.max_general_drift_ratio, "max_general_drift_ratio", min_value=0.0)
        _require_finite_float(self.max_domain_drift_ratio, "max_domain_drift_ratio", min_value=0.0)
        _require_finite_float(self.annihilation_tol, "annihilation_tol", min_value=0.0, strict_min=True)
        _require_finite_float(self.min_one_minus_cos, "min_one_minus_cos", min_value=0.0, max_value=2.0, strict_min=True)
        _require_finite_float(
            self.min_jepa_one_minus_cos, "min_jepa_one_minus_cos", min_value=0.0, max_value=2.0, strict_min=True
        )
        _require_finite_float(
            self.min_normalized_rankme, "min_normalized_rankme", min_value=0.0, max_value=1.0, strict_min=True
        )
        _require_finite_float(self.min_rankme_ratio, "min_rankme_ratio", min_value=0.0, max_value=1.0, strict_min=True)
        object.__setattr__(
            self, "alpha_grid", _require_float_grid(self.alpha_grid, "alpha_grid", min_value=0.0, strict_min=True)
        )
        object.__setattr__(
            self, "safe_tau_grid", _require_float_grid(self.safe_tau_grid, "safe_tau_grid", min_value=1.0)
        )
        _require_finite_float(self.ece_weight, "ece_weight", min_value=0.0)
        _require_positive_int(self.ece_bins, "ece_bins")
        _require_finite_float(self.rmsnorm_eps, "rmsnorm_eps", min_value=0.0, strict_min=True)
        if self.residual_connection is not None and not isinstance(self.residual_connection, bool):
            raise TypeError("residual_connection must be bool or None")
        _require_finite_float(self.residual_scale, "residual_scale")
        _require_finite_float(self.min_energy_improvement, "min_energy_improvement", min_value=0.0)


@dataclass(frozen=True)
class AssociativeNullspaceProjection:
    """Result and Theorem 15.1(2) diagnostics of ``(s * B)(A @ P_0)`` projection."""

    b_scaled: torch.Tensor
    a_projected: torch.Tensor
    delta_projected: torch.Tensor
    rank_bound: int
    annihilation_norm: float
    associative_diff: float
    buffer_compression_ratio: float
    specialist_alignment_cosine: float
    specialist_retained_norm_ratio: float


@dataclass(frozen=True)
class RankMeIsotropyReport:
    """RankMe effective-rank isotropy and cosine-collapse verification report."""

    n_tokens: int
    hidden_dim: int
    rankme: float
    normalized_rankme: float
    centered_rankme: float
    participation_ratio: float
    mean_pairwise_cosine: float
    one_minus_cosine: float
    jepa_one_minus_cos: float | None
    rankme_ratio_vs_ref: float | None
    collapse_reasons: tuple[str, ...]
    passed: bool


# ---------------------------------------------------------------------------
# 2. Orthonormal Prose Null-Space Projector & Associative (s*B)(A @ P_0)
# ---------------------------------------------------------------------------


def build_prose_nullspace_projector(
    prose_activations: torch.Tensor | None = None,
    *,
    v_prose: torch.Tensor | None = None,
    variance_threshold: float = 0.999,
    ortho_tol: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct ``(V_prose, P_0)`` with ``P_0 = I - V_prose^T V_prose`` in CPU ``float64``.

    Provide either ``v_prose`` of shape ``(k, d_in)`` with orthonormal rows
    (``V_prose @ V_prose^T = I_k``) or ``prose_activations`` of shape ``(N, d_in)``.
    """
    if (prose_activations is None) == (v_prose is None):
        raise ValueError("Provide exactly one of prose_activations or v_prose")

    if v_prose is not None:
        if not isinstance(v_prose, torch.Tensor) or not v_prose.is_floating_point() or v_prose.ndim != 2:
            raise ValueError("v_prose must be a 2D floating-point torch.Tensor")
        v64 = v_prose.detach().to(device="cpu", dtype=torch.float64).contiguous()
        if not bool(torch.isfinite(v64).all().item()):
            raise ValueError("v_prose contains non-finite values")
        k, d_in = int(v64.shape[0]), int(v64.shape[1])
        if d_in == 0:
            raise ValueError("v_prose feature dimension d_in must be >= 1")
        if k > d_in:
            raise ValueError(f"v_prose row rank {k} cannot exceed feature dim {d_in}")
        tol = _require_finite_float(ortho_tol, "ortho_tol", min_value=0.0, strict_min=True)
        if k > 0:
            gram = v64 @ v64.T
            ortho_err = float(torch.linalg.matrix_norm(gram - torch.eye(k, dtype=torch.float64), ord="fro").item())
            if ortho_err > tol:
                raise ValueError(f"v_prose rows must be orthonormal (||V V^T - I||_F = {ortho_err:.3e})")
        p_0 = torch.eye(d_in, dtype=torch.float64) - (v64.T @ v64)
        p_0 = 0.5 * (p_0 + p_0.T)
        return v64, p_0

    assert prose_activations is not None
    x_prose = _require_fp_tensor(prose_activations, "prose_activations", ndim=2)
    thresh = _require_finite_float(variance_threshold, "variance_threshold", min_value=0.0, max_value=1.0, strict_min=True)
    d_in = int(x_prose.shape[1])
    _, s_vals, vh = torch.linalg.svd(x_prose, full_matrices=False)
    energy = s_vals.pow(2)
    total_energy = float(energy.sum().item())
    if total_energy <= 1e-18:
        return torch.zeros((0, d_in), dtype=torch.float64), torch.eye(d_in, dtype=torch.float64)
    cum_ratio = torch.cumsum(energy, dim=0) / total_energy
    keep = int(torch.searchsorted(cum_ratio, torch.tensor(thresh, dtype=torch.float64)).item()) + 1
    keep = min(keep, int(vh.shape[0]))
    v64 = vh[:keep, :].contiguous()
    p_0 = torch.eye(d_in, dtype=torch.float64) - (v64.T @ v64)
    p_0 = 0.5 * (p_0 + p_0.T)
    return v64, p_0


def _basis_from_projector(p_0: torch.Tensor) -> torch.Tensor:
    p64 = _require_fp_tensor(p_0, "p_0", ndim=2)
    if p64.shape[0] != p64.shape[1]:
        raise ValueError(f"p_0 must be square, got shape {tuple(p64.shape)}")
    d_in = int(p64.shape[0])
    sym_p = 0.5 * (p64 + p64.T)
    asym = float(torch.linalg.matrix_norm(p64 - sym_p, ord="fro").item())
    if asym > 1e-4:
        raise ValueError(f"p_0 must be symmetric (asymmetry {asym:.3e})")
    idem = float(torch.linalg.matrix_norm(sym_p @ sym_p - sym_p, ord="fro").item())
    if idem > 1e-3:
        raise ValueError(f"p_0 must be idempotent (||P_0^2 - P_0||_F = {idem:.3e})")
    q_perp = torch.eye(d_in, dtype=torch.float64) - sym_p
    q_perp = 0.5 * (q_perp + q_perp.T)
    evals, evecs = torch.linalg.eigh(q_perp)
    keep = evals > 0.5
    if not bool(keep.any().item()):
        return torch.zeros((0, d_in), dtype=torch.float64)
    return evecs[:, keep].T.contiguous()


def _normalize_jepa_delta(delta: JepaLoRADelta | Mapping[str, Any]) -> JepaLoRADelta:
    if isinstance(delta, JepaLoRADelta):
        return delta
    if isinstance(delta, Mapping):
        b_raw = delta.get("b_mat", delta.get("lora_B"))
        a_raw = delta.get("a_mat", delta.get("lora_A"))
        return JepaLoRADelta(
            b_mat=b_raw,
            a_mat=a_raw,
            scale=float(delta.get("scale", 1.0)),
            objective_mode=str(delta.get("objective_mode", "pairs")),
            adapter_name=str(delta.get("adapter_name", "jepa_round")),
        )
    raise TypeError(f"jepa_lora must be JepaLoRADelta or Mapping, got {type(delta).__name__}")


def project_jepa_lora_associative(
    jepa_lora: JepaLoRADelta | Mapping[str, Any],
    p_0: torch.Tensor,
    v_prose: torch.Tensor | None = None,
    *,
    specialist_activations: torch.Tensor | None = None,
    alpha: float = 1.0,
) -> AssociativeNullspaceProjection:
    """Compute associative rank-preserving ``(s * B)(A @ P_0)`` (Theorem 15.1(2)).

    Evaluates ``A_proj = A @ P_0`` first without materializing unprojected dense
    ``(s * B) @ A`` intermediates, keeping working rank bounded by ``r``.
    """
    delta = _normalize_jepa_delta(jepa_lora)
    alpha_f = _require_finite_float(alpha, "alpha", min_value=0.0, strict_min=True)
    b64 = _require_fp_tensor(delta.b_mat, "b_mat", ndim=2)
    a64 = _require_fp_tensor(delta.a_mat, "a_mat", ndim=2)
    d_out, rank = int(b64.shape[0]), int(b64.shape[1])
    d_in = int(a64.shape[1])

    p64 = _require_fp_tensor(p_0, "p_0", ndim=2)
    if p64.shape != (d_in, d_in):
        raise ValueError(f"p_0 must have shape ({d_in}, {d_in}), got {tuple(p64.shape)}")
    if v_prose is None:
        v64 = _basis_from_projector(p64)
    else:
        if not isinstance(v_prose, torch.Tensor) or not v_prose.is_floating_point() or v_prose.ndim != 2:
            raise ValueError("v_prose must be a 2D floating-point torch.Tensor")
        v64 = v_prose.detach().to(device="cpu", dtype=torch.float64).contiguous()
        if int(v64.shape[1]) != d_in:
            raise ValueError(f"v_prose width {int(v64.shape[1])} != d_in {d_in}")

    b_scaled = b64 * (float(delta.scale) * alpha_f)
    a_projected = a64 @ p64
    delta_projected = b_scaled @ a_projected

    annihilation_norm = (
        float(torch.linalg.matrix_norm(delta_projected @ v64.T, ord="fro").item())
        if int(v64.shape[0]) > 0
        else 0.0
    )
    delta_dense_proj = (b_scaled @ a64) @ p64
    associative_diff = float(torch.linalg.matrix_norm(delta_projected - delta_dense_proj, ord="fro").item())

    if specialist_activations is not None:
        x_spec = _require_fp_tensor(specialist_activations, "specialist_activations", ndim=2)
        if int(x_spec.shape[1]) != d_in:
            raise ValueError(f"specialist_activations width {int(x_spec.shape[1])} != d_in {d_in}")
        raw_out = x_spec @ (b_scaled @ a64).T
        proj_out = x_spec @ delta_projected.T
        raw_norm = float(torch.linalg.norm(raw_out).item())
        proj_norm = float(torch.linalg.norm(proj_out).item())
        denom = raw_norm * proj_norm
        spec_cos = float((raw_out * proj_out).sum().item() / denom) if denom > 1e-18 else 0.0
        retained_ratio = float(proj_norm / raw_norm) if raw_norm > 1e-18 else 0.0
    else:
        delta_raw = b_scaled @ a64
        raw_norm = float(torch.linalg.norm(delta_raw).item())
        proj_norm = float(torch.linalg.norm(delta_projected).item())
        denom = raw_norm * proj_norm
        spec_cos = float((delta_raw * delta_projected).sum().item() / denom) if denom > 1e-18 else 0.0
        retained_ratio = float(proj_norm / raw_norm) if raw_norm > 1e-18 else 0.0

    return AssociativeNullspaceProjection(
        b_scaled=b_scaled,
        a_projected=a_projected,
        delta_projected=delta_projected,
        rank_bound=rank,
        annihilation_norm=annihilation_norm,
        associative_diff=associative_diff,
        buffer_compression_ratio=float(d_out) / float(rank),
        specialist_alignment_cosine=spec_cos,
        specialist_retained_norm_ratio=retained_ratio,
    )


# ---------------------------------------------------------------------------
# 3. RankMe Effective-Rank Isotropy & Cosine-Collapse Guard
# ---------------------------------------------------------------------------


def _spectral_rankme(mat_2d: torch.Tensor, eps: float = 1e-12) -> tuple[float, float, float]:
    n_tok = int(mat_2d.shape[0])
    sv = torch.linalg.svdvals(mat_2d)
    sigma_sum = float(sv.sum().item())
    sigma_max = float(sv[0].item())
    if sigma_sum <= eps or sigma_max <= eps:
        return 0.0, 0.0, 0.0
    probs = sv / sigma_sum
    pos = probs[probs > 1e-15]
    entropy = max(0.0, float(-(pos * torch.log(pos)).sum().item()))
    rankme = float(math.exp(entropy))
    lambdas = (sv * sv) / float(n_tok)
    lam_sum = float(lambdas.sum().item())
    lam_sq = float((lambdas * lambdas).sum().item())
    pr = float((lam_sum * lam_sum) / max(lam_sq, eps * eps))
    return rankme, pr, sigma_max


def measure_rankme_and_cosine_isotropy(
    hidden_states: torch.Tensor,
    *,
    reference_report: RankMeIsotropyReport | None = None,
    paired_views: tuple[torch.Tensor, torch.Tensor] | None = None,
    stp_hidden_seq: torch.Tensor | None = None,
    jepa_one_minus_cos_override: float | None = None,
    config: NullspaceJepaRescueConfig | None = None,
) -> RankMeIsotropyReport:
    """Measure spectral RankMe effective rank, participation ratio, and ``1 - cos`` isotropy."""
    cfg = config if config is not None else NullspaceJepaRescueConfig()
    h_raw = _require_fp_tensor(hidden_states, "hidden_states", min_ndim=2)
    h2d = h_raw.reshape(-1, int(h_raw.shape[-1])).contiguous()
    n_tok, hidden_dim = int(h2d.shape[0]), int(h2d.shape[1])
    min_dim = min(n_tok, hidden_dim)

    norms = torch.linalg.vector_norm(h2d, dim=1)
    nz_mask = norms > 1e-12
    n_nz = int(nz_mask.sum().item())
    if n_nz == 0 or n_tok < 2:
        mean_cos = 1.0
    else:
        unit_h = torch.zeros_like(h2d)
        unit_h[nz_mask] = h2d[nz_mask] / norms[nz_mask].unsqueeze(1)
        u_sum = unit_h.sum(dim=0)
        pair_sum = float(torch.dot(u_sum, u_sum).item()) - float(n_nz)
        mean_cos = max(-1.0, min(1.0, pair_sum / float(n_tok * (n_tok - 1))))
    one_minus_cos = float(1.0 - mean_cos)

    rankme, pr, sigma_max = _spectral_rankme(h2d)
    centered_rankme, _, _ = _spectral_rankme(h2d - h2d.mean(dim=0, keepdim=True))
    normalized_rankme = float(rankme / float(min_dim))

    jepa_1m_cos: float | None = None
    if jepa_one_minus_cos_override is not None:
        jepa_1m_cos = _require_finite_float(
            jepa_one_minus_cos_override, "jepa_one_minus_cos_override", min_value=0.0, max_value=2.0
        )
    elif paired_views is not None:
        v1 = _require_fp_tensor(paired_views[0], "paired_views[0]", ndim=2)
        v2 = _require_fp_tensor(paired_views[1], "paired_views[1]", ndim=2)
        if v1.shape != v2.shape:
            raise ValueError(f"paired_views shapes mismatch: {tuple(v1.shape)} vs {tuple(v2.shape)}")
        jepa_1m_cos = float((1.0 - F.cosine_similarity(v1, v2, dim=-1)).mean().item())
    elif stp_hidden_seq is not None:
        seq = _require_fp_tensor(stp_hidden_seq, "stp_hidden_seq", ndim=3)
        if int(seq.shape[1]) < 3:
            raise ValueError("stp_hidden_seq must have sequence length >= 3")
        mid = int(seq.shape[1]) // 2
        d1 = seq[:, -1, :] - seq[:, mid, :]
        d0 = seq[:, mid, :] - seq[:, 0, :]
        jepa_1m_cos = float((1.0 - F.cosine_similarity(d1, d0, dim=-1)).mean().item())

    reasons: list[str] = []
    if sigma_max <= 1e-12 or n_nz == 0:
        reasons.append("zero_or_degenerate_states")
    else:
        if n_tok >= 2 and one_minus_cos <= cfg.min_one_minus_cos:
            reasons.append("cosine_cone_collapse")
        if min_dim > 1 and normalized_rankme < cfg.min_normalized_rankme:
            reasons.append("low_normalized_rankme")
        if min_dim > 1 and rankme <= 1.25:
            reasons.append("near_rank1_collapse")

    rankme_ratio: float | None = None
    if reference_report is not None and reference_report.rankme > 1e-12:
        rankme_ratio = float(rankme / reference_report.rankme)
        if rankme_ratio < cfg.min_rankme_ratio:
            reasons.append("rankme_drop_vs_reference")

    if jepa_1m_cos is not None and jepa_1m_cos <= cfg.min_jepa_one_minus_cos:
        reasons.append("jepa_objective_cosine_collapse")

    return RankMeIsotropyReport(
        n_tokens=n_tok,
        hidden_dim=hidden_dim,
        rankme=float(rankme),
        normalized_rankme=normalized_rankme,
        centered_rankme=float(centered_rankme),
        participation_ratio=float(pr),
        mean_pairwise_cosine=float(mean_cos),
        one_minus_cosine=one_minus_cos,
        jepa_one_minus_cos=jepa_1m_cos,
        rankme_ratio_vs_ref=rankme_ratio,
        collapse_reasons=tuple(reasons),
        passed=(len(reasons) == 0),
    )


# ---------------------------------------------------------------------------
# 4. Forward Evaluation, ANSE Gate & Atomic Bitwise Rollback
# ---------------------------------------------------------------------------


def _apply_rmsnorm(hidden: torch.Tensor, norm_weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    if norm_weight is None:
        return hidden
    inv_rms = torch.rsqrt(hidden.pow(2).mean(dim=-1, keepdim=True) + eps)
    return hidden * inv_rms * norm_weight.unsqueeze(0)


def compute_nll_and_ece(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> tuple[float, float]:
    """Compute exact token cross-entropy NLL and top-1 Expected Calibration Error (ECE)."""
    log_probs = F.log_softmax(logits, dim=-1)
    nll_mean = float(F.nll_loss(log_probs, targets, reduction="mean").item())
    probs = log_probs.exp()
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == targets).to(dtype=torch.float64)
    edges = torch.linspace(0.0, 1.0, steps=ece_bins + 1, dtype=torch.float64)
    n_total = float(targets.numel())
    ece = 0.0
    for idx in range(ece_bins):
        lo, hi = edges[idx], edges[idx + 1]
        in_bin = (confidences >= lo) & (confidences <= hi) if idx == 0 else (confidences > lo) & (confidences <= hi)
        cnt = int(in_bin.sum().item())
        if cnt > 0:
            ece += (float(cnt) / n_total) * abs(float(accuracies[in_bin].mean().item()) - float(confidences[in_bin].mean().item()))
    return nll_mean, float(ece)


def _forward_and_eval(
    layer_weight: torch.Tensor,
    norm_weight: torch.Tensor | None,
    lm_head_weight: torch.Tensor,
    domain_inputs: Mapping[str, torch.Tensor],
    domain_targets: Mapping[str, torch.Tensor],
    *,
    general_domain_name: str,
    residual_connection: bool,
    residual_scale: float,
    rmsnorm_eps: float,
    ece_bins: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    per_domain_nll: dict[str, float] = {}
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_hidden: list[torch.Tensor] = []
    spec_hidden: list[torch.Tensor] = []

    for dom in sorted(domain_inputs.keys()):
        x_dom = domain_inputs[dom]
        y_dom = domain_targets[dom]
        projected = x_dom @ layer_weight.T
        h_out = (x_dom + residual_scale * projected) if residual_connection else projected
        all_hidden.append(h_out)
        if dom != general_domain_name:
            spec_hidden.append(h_out)
        h_norm = _apply_rmsnorm(h_out, norm_weight, rmsnorm_eps)
        logits = h_norm @ lm_head_weight.T
        nll_d, _ = compute_nll_and_ece(logits, y_dom, ece_bins=ece_bins)
        per_domain_nll[dom] = nll_d
        all_logits.append(logits)
        all_targets.append(y_dom)

    nll_mean, ece_val = compute_nll_and_ece(
        torch.cat(all_logits, dim=0),
        torch.cat(all_targets, dim=0),
        ece_bins=ece_bins,
    )
    return (
        {"nll_mean": nll_mean, "ece_value": ece_val, "per_domain_nll": per_domain_nll},
        torch.cat(all_hidden, dim=0),
        torch.cat(spec_hidden, dim=0),
    )


def _evaluate_anse(
    base_metrics: Mapping[str, Any],
    cand_metrics: Mapping[str, Any],
    cfg: NullspaceJepaRescueConfig,
) -> dict[str, Any]:
    gen_dom = cfg.general_domain_name
    base_gen = float(base_metrics["per_domain_nll"][gen_dom])
    cand_gen = float(cand_metrics["per_domain_nll"][gen_dom])
    gen_drift_nats = cand_gen - base_gen
    gen_drift_ratio = gen_drift_nats / max(base_gen, 1e-8)

    domain_drifts: dict[str, float] = {}
    for dom, b_val in base_metrics["per_domain_nll"].items():
        c_val = float(cand_metrics["per_domain_nll"][dom])
        domain_drifts[dom] = (c_val - float(b_val)) / max(float(b_val), 1e-8)

    worst_drift = max(domain_drifts.values())
    nll_diff = float(cand_metrics["nll_mean"]) - float(base_metrics["nll_mean"])
    ece_diff = float(cand_metrics["ece_value"]) - float(base_metrics["ece_value"])
    raw_energy = nll_diff + cfg.ece_weight * ece_diff
    guard_ok = (gen_drift_ratio <= cfg.max_general_drift_ratio) and (worst_drift <= cfg.max_domain_drift_ratio)
    anse_energy = raw_energy if (guard_ok and raw_energy < -cfg.min_energy_improvement) else ANSE_MAXIMUM_PAIN
    return {
        "nll_diff": nll_diff,
        "ece_diff": ece_diff,
        "general_drift_nats": gen_drift_nats,
        "general_drift_ratio": gen_drift_ratio,
        "worst_domain_drift_ratio": worst_drift,
        "per_domain_drift_ratio": domain_drifts,
        "guard_ok": guard_ok,
        "raw_energy": raw_energy,
        "anse_energy": anse_energy,
    }


def _validate_rescue_inputs(
    mutable_state: dict[str, torch.Tensor],
    lm_head_weight: torch.Tensor,
    domain_hidden_inputs: Mapping[str, torch.Tensor],
    domain_targets: Mapping[str, torch.Tensor],
    cfg: NullspaceJepaRescueConfig,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not isinstance(mutable_state, dict) or "layer_weight" not in mutable_state:
        raise ValueError("mutable_state must be a dict containing 'layer_weight'")
    w_base = _require_fp_tensor(mutable_state["layer_weight"], "mutable_state['layer_weight']", ndim=2)
    norm_raw = mutable_state.get("final_norm_weight")
    norm_base = _require_fp_tensor(norm_raw, "mutable_state['final_norm_weight']", ndim=1) if norm_raw is not None else None
    head_w = _require_fp_tensor(lm_head_weight, "lm_head_weight", ndim=2)

    d_out, d_in = int(w_base.shape[0]), int(w_base.shape[1])
    if norm_base is not None and int(norm_base.shape[0]) != d_out:
        raise ValueError(f"final_norm_weight dim {int(norm_base.shape[0])} != d_out {d_out}")
    if int(head_w.shape[1]) != d_out:
        raise ValueError(f"lm_head_weight width {int(head_w.shape[1])} != d_out {d_out}")

    if not isinstance(domain_hidden_inputs, Mapping) or len(domain_hidden_inputs) < 2:
        raise ValueError("domain_hidden_inputs must include general prose and at least one specialist domain")
    if cfg.general_domain_name not in domain_hidden_inputs:
        raise ValueError(f"Missing general domain {cfg.general_domain_name!r} in domain_hidden_inputs")
    if set(domain_hidden_inputs.keys()) != set(domain_targets.keys()):
        raise ValueError("domain_hidden_inputs and domain_targets must share identical domain keys")

    vocab_size = int(head_w.shape[0])
    clean_x: dict[str, torch.Tensor] = {}
    clean_y: dict[str, torch.Tensor] = {}
    for dom in sorted(domain_hidden_inputs.keys()):
        x_d = _require_fp_tensor(domain_hidden_inputs[dom], f"domain_hidden_inputs[{dom!r}]", ndim=2)
        if int(x_d.shape[1]) != d_in:
            raise ValueError(f"domain {dom!r} input dim {int(x_d.shape[1])} != d_in {d_in}")
        t_raw = domain_targets[dom]
        if not isinstance(t_raw, torch.Tensor) or t_raw.ndim != 1 or int(t_raw.shape[0]) != int(x_d.shape[0]):
            raise ValueError(f"domain {dom!r} targets must be 1D with length {int(x_d.shape[0])}")
        t_long = t_raw.detach().to(device="cpu", dtype=torch.long)
        if int(t_long.min().item()) < 0 or int(t_long.max().item()) >= vocab_size:
            raise ValueError(f"domain {dom!r} target indices out of range [0, {vocab_size})")
        clean_x[dom] = x_d
        clean_y[dom] = t_long

    return w_base, norm_base, head_w, clean_x, clean_y


def rescue_post_jepa_lora(
    mutable_state: dict[str, torch.Tensor],
    lm_head_weight: torch.Tensor,
    jepa_lora: JepaLoRADelta | Mapping[str, Any],
    domain_hidden_inputs: Mapping[str, torch.Tensor],
    domain_targets: Mapping[str, torch.Tensor],
    *,
    v_prose: torch.Tensor | None = None,
    p_0: torch.Tensor | None = None,
    prose_activations: torch.Tensor | None = None,
    paired_views_fn: Callable[[torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]] | None = None,
    jepa_one_minus_cos: float | None = None,
    config: NullspaceJepaRescueConfig | None = None,
) -> dict[str, Any]:
    """Rescue a post-round JEPA/STP LoRA delta via ``(s*B)(A @ P_0)`` and ANSE gating.

    Commits updated weights in-place into ``mutable_state`` if and only if:
    (1) algebraic null-space annihilation satisfies ``||((s*B)(A @ P_0)) V_prose^T|| <= tol``,
    (2) RankMe effective-rank isotropy & ``1 - cos > threshold`` pass cleanly, and
    (3) ``ANSE Delta E < 0`` and ``D0 / worst_domain <= +1.0%``.
    Otherwise performs atomic bitwise rollback and returns ``anse_energy = ANSE_MAXIMUM_PAIN``.
    """
    cfg = config if config is not None else NullspaceJepaRescueConfig()
    delta = _normalize_jepa_delta(jepa_lora)
    w_base, norm_base, head_w, clean_x, clean_y = _validate_rescue_inputs(
        mutable_state, lm_head_weight, domain_hidden_inputs, domain_targets, cfg
    )
    d_out, d_in = int(w_base.shape[0]), int(w_base.shape[1])
    if delta.b_mat.shape[0] != d_out or delta.a_mat.shape[1] != d_in:
        raise ValueError(
            f"jepa_lora shape ({delta.b_mat.shape[0]}, {delta.a_mat.shape[1]}) != layer_weight ({d_out}, {d_in})"
        )

    use_residual = (d_out == d_in) if cfg.residual_connection is None else cfg.residual_connection
    if use_residual and d_out != d_in:
        raise ValueError("residual_connection=True requires square layer_weight (d_out == d_in)")

    # Snapshot initial state for exact bitwise rollback guarantee.
    w_backup = mutable_state["layer_weight"].detach().clone()
    state_snapshot = {"layer_weight": w_backup}
    norm_backup: torch.Tensor | None = None
    if norm_base is not None:
        norm_backup = mutable_state["final_norm_weight"].detach().clone()
        state_snapshot["final_norm_weight"] = norm_backup
    initial_state_sha256 = compute_state_sha256(state_snapshot)

    # Resolve V_prose and P_0 = I - V_prose^T V_prose.
    if p_0 is not None:
        p0_mat = _require_fp_tensor(p_0, "p_0", ndim=2)
        v_mat = _require_fp_tensor(v_prose, "v_prose", ndim=2) if v_prose is not None else _basis_from_projector(p0_mat)
    elif v_prose is not None:
        v_mat, p0_mat = build_prose_nullspace_projector(v_prose=v_prose)
    else:
        src_prose = prose_activations if prose_activations is not None else clean_x[cfg.general_domain_name]
        v_mat, p0_mat = build_prose_nullspace_projector(prose_activations=src_prose)

    # 1. Baseline evaluation & reference RankMe isotropy.
    base_metrics, base_hidden, base_spec_hidden = _forward_and_eval(
        w_base,
        norm_base,
        head_w,
        clean_x,
        clean_y,
        general_domain_name=cfg.general_domain_name,
        residual_connection=use_residual,
        residual_scale=cfg.residual_scale,
        rmsnorm_eps=cfg.rmsnorm_eps,
        ece_bins=cfg.ece_bins,
    )
    ref_iso = measure_rankme_and_cosine_isotropy(base_hidden, config=cfg)
    ref_spec_iso = measure_rankme_and_cosine_isotropy(base_spec_hidden, config=cfg)

    # 2. Unguarded raw JEPA LoRA diagnostic (demonstrating P1 / E-STP-1b prose drift).
    b64 = _require_fp_tensor(delta.b_mat, "b_mat", ndim=2)
    a64 = _require_fp_tensor(delta.a_mat, "a_mat", ndim=2)
    raw_delta = (b64 * float(delta.scale)) @ a64
    raw_metrics, _, _ = _forward_and_eval(
        w_base + raw_delta,
        norm_base,
        head_w,
        clean_x,
        clean_y,
        general_domain_name=cfg.general_domain_name,
        residual_connection=use_residual,
        residual_scale=cfg.residual_scale,
        rmsnorm_eps=cfg.rmsnorm_eps,
        ece_bins=cfg.ece_bins,
    )
    raw_anse = _evaluate_anse(base_metrics, raw_metrics, cfg)

    # 3. Associative rank-preserving null-space projection (s*B)(A @ P_0).
    spec_doms = [d for d in sorted(clean_x.keys()) if d != cfg.general_domain_name]
    spec_x = torch.cat([clean_x[d] for d in spec_doms], dim=0)
    proj = project_jepa_lora_associative(delta, p0_mat, v_mat, specialist_activations=spec_x, alpha=1.0)
    projected_factor_sha256 = compute_state_sha256({"b_scaled": proj.b_scaled, "a_projected": proj.a_projected})

    # 4. Candidate search over (alpha, safe tau_temp >= 1.0) gated by annihilation, RankMe & ANSE.
    tau_candidates = cfg.safe_tau_grid if norm_base is not None else (1.0,)
    best_cand: dict[str, Any] | None = None
    best_energy = ANSE_MAXIMUM_PAIN
    last_iso: RankMeIsotropyReport = ref_iso
    last_spec_iso: RankMeIsotropyReport = ref_spec_iso
    last_anse: dict[str, Any] = raw_anse
    any_collapse = False
    any_annihilation_breach = proj.annihilation_norm > cfg.annihilation_tol

    if not any_annihilation_breach:
        for alpha in cfg.alpha_grid:
            w_cand = w_base + float(alpha) * proj.delta_projected
            for tau in tau_candidates:
                norm_cand = (norm_base / float(tau)) if norm_base is not None else None
                cand_metrics, cand_hidden, cand_spec_hidden = _forward_and_eval(
                    w_cand,
                    norm_cand,
                    head_w,
                    clean_x,
                    clean_y,
                    general_domain_name=cfg.general_domain_name,
                    residual_connection=use_residual,
                    residual_scale=cfg.residual_scale,
                    rmsnorm_eps=cfg.rmsnorm_eps,
                    ece_bins=cfg.ece_bins,
                )
                views_pair = paired_views_fn(w_cand, norm_cand) if paired_views_fn is not None else None
                iso = measure_rankme_and_cosine_isotropy(
                    cand_hidden,
                    reference_report=ref_iso,
                    paired_views=views_pair,
                    jepa_one_minus_cos_override=jepa_one_minus_cos,
                    config=cfg,
                )
                spec_iso = measure_rankme_and_cosine_isotropy(
                    cand_spec_hidden,
                    reference_report=ref_spec_iso,
                    config=cfg,
                )
                anse_eval = _evaluate_anse(base_metrics, cand_metrics, cfg)
                last_iso = iso
                last_spec_iso = spec_iso
                last_anse = anse_eval
                if not iso.passed or not spec_iso.passed:
                    any_collapse = True
                    continue
                if anse_eval["guard_ok"] and float(anse_eval["anse_energy"]) < best_energy:
                    best_energy = float(anse_eval["anse_energy"])
                    best_cand = {
                        "alpha": float(alpha),
                        "tau_temp": float(tau),
                        "w_cand": w_cand,
                        "norm_cand": norm_cand,
                        "metrics": cand_metrics,
                        "anse": anse_eval,
                        "iso": iso,
                        "spec_iso": spec_iso,
                    }

    # 5. Monotone ANSE commit OR atomic bitwise rollback.
    if best_cand is not None and best_energy < 0.0:
        mutable_state["layer_weight"] = best_cand["w_cand"].to(dtype=w_backup.dtype, device=w_backup.device)
        if norm_backup is not None and best_cand["norm_cand"] is not None:
            mutable_state["final_norm_weight"] = best_cand["norm_cand"].to(
                dtype=norm_backup.dtype, device=norm_backup.device
            )
        accepted = True
        rolled_back = False
        collapse_rejected = False
        guard_rejected = False
        rejection_reason = ""
        selected_alpha = best_cand["alpha"]
        selected_tau = best_cand["tau_temp"]
        final_metrics = best_cand["metrics"]
        final_anse = best_cand["anse"]
        final_iso = best_cand["iso"]
        final_spec_iso = best_cand["spec_iso"]
    else:
        mutable_state["layer_weight"] = w_backup
        if norm_backup is not None:
            mutable_state["final_norm_weight"] = norm_backup
        accepted = False
        rolled_back = True
        collapse_rejected = any_collapse
        guard_rejected = not any_annihilation_breach and not any_collapse and not bool(last_anse["guard_ok"])
        if any_annihilation_breach:
            rejection_reason = "prose_nullspace_annihilation_breach"
        elif any_collapse:
            rejection_reason = "rankme_or_cosine_collapse_detected"
        elif not bool(last_anse["guard_ok"]):
            rejection_reason = "general_prose_or_domain_guard_exceeded"
        else:
            rejection_reason = "anse_energy_non_improving"
        selected_alpha = 0.0
        selected_tau = 1.0
        final_metrics = base_metrics
        final_anse = {
            "nll_diff": 0.0,
            "ece_diff": 0.0,
            "general_drift_nats": 0.0,
            "general_drift_ratio": 0.0,
            "worst_domain_drift_ratio": 0.0,
            "per_domain_drift_ratio": {dom: 0.0 for dom in clean_x},
            "guard_ok": False,
            "raw_energy": 0.0,
            "anse_energy": ANSE_MAXIMUM_PAIN,
        }
        final_iso = last_iso if any_collapse else ref_iso
        final_spec_iso = last_spec_iso if any_collapse else ref_spec_iso

    final_snapshot = {"layer_weight": mutable_state["layer_weight"]}
    if norm_backup is not None:
        final_snapshot["final_norm_weight"] = mutable_state["final_norm_weight"]
    final_state_sha256 = compute_state_sha256(final_snapshot)
    rollback_verified = (final_state_sha256 == initial_state_sha256) if rolled_back else True
    combined_collapse_reasons = list(dict.fromkeys([*final_iso.collapse_reasons, *final_spec_iso.collapse_reasons]))

    receipt: dict[str, Any] = {
        "schema": RESCUER_SCHEMA,
        "object": RESCUER_RECEIPT_OBJECT,
        "objective_mode": delta.objective_mode,
        "adapter_name": delta.adapter_name,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "rollback_verified": rollback_verified,
        "collapse_rejected": collapse_rejected,
        "guard_rejected": guard_rejected,
        "rejection_reason": rejection_reason,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
        "projected_factor_sha256": projected_factor_sha256,
        "selected_alpha": selected_alpha,
        "selected_tau_temp": selected_tau,
        "rank_bound": proj.rank_bound,
        "annihilation_norm": proj.annihilation_norm,
        "associative_diff": proj.associative_diff,
        "buffer_compression_ratio": proj.buffer_compression_ratio,
        "specialist_alignment_cosine": proj.specialist_alignment_cosine,
        "specialist_retained_norm_ratio": proj.specialist_retained_norm_ratio,
        "base_nll_mean": float(base_metrics["nll_mean"]),
        "base_ece_value": float(base_metrics["ece_value"]),
        "raw_jepa_nll_mean": float(raw_metrics["nll_mean"]),
        "raw_jepa_general_drift_nats": float(raw_anse["general_drift_nats"]),
        "raw_jepa_general_drift_ratio": float(raw_anse["general_drift_ratio"]),
        "raw_jepa_guard_ok": bool(raw_anse["guard_ok"]),
        "raw_jepa_anse_energy": float(raw_anse["anse_energy"]),
        "nll_mean": float(final_metrics["nll_mean"]),
        "ece_value": float(final_metrics["ece_value"]),
        "nll_diff": float(final_anse["nll_diff"]),
        "ece_diff": float(final_anse["ece_diff"]),
        "general_drift_nats": float(final_anse["general_drift_nats"]),
        "general_drift_ratio": float(final_anse["general_drift_ratio"]),
        "worst_domain_drift_ratio": float(final_anse["worst_domain_drift_ratio"]),
        "per_domain_drift_ratio": {k: float(v) for k, v in sorted(final_anse["per_domain_drift_ratio"].items())},
        "anse_energy": float(final_anse["anse_energy"]),
        "rankme_pre": float(ref_iso.rankme),
        "rankme_post": float(final_iso.rankme),
        "specialist_rankme_post": float(final_spec_iso.rankme),
        "normalized_rankme_post": float(final_iso.normalized_rankme),
        "one_minus_cosine_pre": float(ref_iso.one_minus_cosine),
        "one_minus_cosine_post": float(final_iso.one_minus_cosine),
        "specialist_one_minus_cosine_post": float(final_spec_iso.one_minus_cosine),
        "jepa_one_minus_cos": final_iso.jepa_one_minus_cos,
        "collapse_reasons": combined_collapse_reasons,
        "config": asdict(cfg),
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


def verify_nullspace_jepa_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Verify canonical SHA-256 provenance and thermodynamic invariants of a rescue receipt."""
    if not isinstance(receipt, Mapping):
        return False
    if receipt.get("schema") != RESCUER_SCHEMA or receipt.get("object") != RESCUER_RECEIPT_OBJECT:
        return False
    for hash_key in ("receipt_sha256", "initial_state_sha256", "final_state_sha256", "projected_factor_sha256"):
        val = receipt.get(hash_key)
        if not isinstance(val, str) or not _SHA256_HEX_RE.match(val):
            return False

    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        if compute_payload_sha256(core) != receipt["receipt_sha256"]:
            return False
    except (TypeError, ValueError):
        return False

    accepted = receipt.get("accepted")
    rolled_back = receipt.get("rolled_back")
    if not isinstance(accepted, bool) or not isinstance(rolled_back, bool) or accepted == rolled_back:
        return False
    if rolled_back:
        if receipt["final_state_sha256"] != receipt["initial_state_sha256"]:
            return False
        if float(receipt.get("anse_energy", 0.0)) != ANSE_MAXIMUM_PAIN:
            return False
        return True

    cfg = receipt.get("config")
    max_gen = float(cfg.get("max_general_drift_ratio", DEFAULT_MAX_GENERAL_DRIFT_RATIO)) if isinstance(cfg, Mapping) else DEFAULT_MAX_GENERAL_DRIFT_RATIO
    ann_tol = float(cfg.get("annihilation_tol", DEFAULT_ANNIHILATION_TOL)) if isinstance(cfg, Mapping) else DEFAULT_ANNIHILATION_TOL
    return (
        float(receipt.get("anse_energy", 1.0)) < 0.0
        and float(receipt.get("general_drift_ratio", 1.0)) <= max_gen + 1e-12
        and float(receipt.get("annihilation_norm", 1.0)) <= ann_tol + 1e-12
    )


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "AssociativeNullspaceProjection",
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_ANNIHILATION_TOL",
    "DEFAULT_ECE_BINS",
    "DEFAULT_ECE_WEIGHT",
    "DEFAULT_GENERAL_DOMAIN",
    "DEFAULT_MAX_DOMAIN_DRIFT_RATIO",
    "DEFAULT_MAX_GENERAL_DRIFT_RATIO",
    "DEFAULT_MIN_JEPA_ONE_MINUS_COS",
    "DEFAULT_MIN_NORMALIZED_RANKME",
    "DEFAULT_MIN_ONE_MINUS_COS",
    "DEFAULT_MIN_RANKME_RATIO",
    "DEFAULT_RMSNORM_EPS",
    "DEFAULT_SAFE_TAU_GRID",
    "JepaLoRADelta",
    "NullspaceJepaRescueConfig",
    "RESCUER_RECEIPT_OBJECT",
    "RESCUER_SCHEMA",
    "RankMeIsotropyReport",
    "build_prose_nullspace_projector",
    "compute_nll_and_ece",
    "compute_payload_sha256",
    "compute_state_sha256",
    "measure_rankme_and_cosine_isotropy",
    "project_jepa_lora_associative",
    "rescue_post_jepa_lora",
    "verify_nullspace_jepa_receipt_provenance",
]

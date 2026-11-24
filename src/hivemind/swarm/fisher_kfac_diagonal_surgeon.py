# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Training-Free Wanda & Optimal-Brain-Surgeon (OBS) LoRA Surgical Sparsification.

Implements activation-weighted parameter pruning and closed-form second-order Hessian
compensation for open-weight Sparse MoE LoRA adapters (``CrowdV1WithEngram``,
``CrowdLoRAMoEMLP``, ``CrowdFineGrainedMoEMLP``, ``CrowdRuntimeMLP``, and ``nn.Linear``)
to surgically eliminate prose-inflating / low-specialist-saliency coordinates without
gradient descent:

1. ``compute_discriminative_wanda_saliency(weight_delta, x_prose, x_spec, *, eps=1e-8)``:
   Scores each weight coordinate ``(i, j)`` by its specialist-vs-prose Wanda saliency
   ``|Delta W_{ij}| * ||x_spec_j||_2 / (||x_prose_j||_2 + eps)``.
2. ``compute_specialist_hessian_inverse(x_spec, *, hessian_damping=1e-2, mode="auto")`` &
   ``apply_obs_compensated_row_surgery(weight_delta, mask_zero, h_spec_inv_diag_or_woodbury)``:
   Prunes low-saliency / prose-harming coordinates with closed-form diagonal, dense Cholesky,
   or Woodbury Optimal Brain Surgeon second-order compensation on specialist activations,
   strictly minimizing specialist quadratic reconstruction error subject to ``w'_M = 0``.
3. ``execute_wanda_obs_moe_surgery(...)``:
   Orchestrates sequential layer-wise Sparse MoE surgery under strict parameter-edit budgets,
   gating commit on ANSE ``<= +1.0%`` general/domain guard compliance and specialist NLL
   retention with atomic bitwise rollback and canonical SHA-256 provenance receipts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdFineGrainedMoEMLP,
    CrowdLoRAMoEMLP,
)
from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    forward_block_pre_mlp,
    per_chunk_nll_from_pre_ln,
)

WANDA_OBS_SCHEMA: str = "hivemind.fisher_kfac_diagonal_surgeon/1"
WANDA_OBS_RECEIPT_OBJECT: str = "hivemind.wanda_obs_surgery_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_EPS: float = 1e-8
DEFAULT_HESSIAN_DAMPING: float = 1e-2
DEFAULT_RIDGE_FLOOR: float = 1e-6
DEFAULT_PRUNE_RATIO: float = 0.25
DEFAULT_MAX_EDITED_PARAM_RATIO: float = 1.0
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_ANSE_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_SCORE_FROM: int = 0

VALID_TARGET_FACTORS: frozenset[str] = frozenset(
    {"w1_A", "w2_A", "w3_A", "w1_B", "w2_B", "w3_B", "w1", "w2", "w3"}
)
VALID_PRUNE_SCOPES: frozenset[str] = frozenset({"global", "matrix", "row"})
VALID_BUDGET_SCOPES: frozenset[str] = frozenset({"target", "trainable", "model"})
VALID_HESSIAN_MODES: frozenset[str] = frozenset({"auto", "dense", "woodbury", "diagonal"})

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# 1. Validation & Canonical Provenance Helpers
# ---------------------------------------------------------------------------


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_nonneg_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a non-negative integer (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a strictly positive integer (excluding bool)."""
    out = _validate_nonneg_int(value, name)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return out


def _validate_finite_float(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _validate_nonneg_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number >= 0.0."""
    out = _validate_finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0.0, got {value!r}")
    return out


def _validate_positive_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number > 0.0."""
    out = _validate_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_unit_interval(
    value: Any,
    name: str,
    *,
    allow_zero: bool = True,
    allow_one: bool = True,
) -> float:
    """Validate a finite real value inside ``[0, 1]`` with configurable endpoints."""
    out = _validate_nonneg_finite(value, name)
    if not allow_zero and out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    if allow_one and out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    if not allow_one and out >= 1.0:
        raise ValueError(f"{name} must be < 1.0, got {value!r}")
    return out


def _validate_finite_2d(tensor: Any, name: str) -> torch.Tensor:
    """Validate a non-empty 2D finite floating-point tensor and return CPU ``float64`` copy."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D tensor, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    out = tensor.detach().to(device="cpu", dtype=torch.float64).clone().contiguous()
    if not bool(torch.isfinite(out).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _validate_activations_2d(
    tensor: Any,
    name: str,
    *,
    expected_dim: int | None = None,
) -> torch.Tensor:
    """Validate a non-empty 2D or 3D finite floating activation tensor as CPU ``float64`` 2D."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim not in (2, 3) or any(dim == 0 for dim in tensor.shape):
        raise ValueError(
            f"{name} must be a non-empty 2D or 3D tensor, got shape {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    width = int(tensor.shape[-1])
    if expected_dim is not None and width != expected_dim:
        raise ValueError(f"{name} feature width {width} must match expected width {expected_dim}")
    flat = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1, width).contiguous()
    if not bool(torch.isfinite(flat).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return flat


def _validate_mask_2d(mask: Any, expected_shape: tuple[int, int], name: str) -> torch.Tensor:
    """Validate a boolean 2D pruning mask matching ``expected_shape``."""
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(mask).__name__}")
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool, got {mask.dtype}")
    if tuple(mask.shape) != expected_shape:
        raise ValueError(f"{name} shape {tuple(mask.shape)} must match {expected_shape}")
    return mask.detach().to(device="cpu").contiguous()


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically with strict finite-number enforcement."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def compute_payload_sha256(payload: Any) -> str:
    """Return lowercase SHA-256 hex digest of ``canonical_json_bytes(payload)``."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Compute a deterministic bitwise SHA-256 digest over a tensor mapping."""
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("state_dict must be a non-empty Mapping[str, torch.Tensor]")
    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        if not isinstance(key, str) or not key:
            raise ValueError(f"State key must be a non-empty string, got {key!r}")
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] must be a torch.Tensor, got {type(tensor).__name__}")
        cpu_t = tensor.detach().to(device="cpu").contiguous()
        if cpu_t.is_floating_point() and not bool(torch.isfinite(cpu_t).all().item()):
            raise ValueError(f"state_dict[{key!r}] contains non-finite values")
        digest.update(f"{key}:{cpu_t.dtype}:{tuple(cpu_t.shape)}:".encode())
        digest.update(cpu_t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_wanda_obs_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != WANDA_OBS_RECEIPT_OBJECT:
        return False
    if receipt.get("schema") != WANDA_OBS_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


@dataclass(frozen=True)
class WandaOBSSurgeonConfig:
    """Validated configuration for discriminative Wanda/OBS MoE surgery and ANSE gating."""

    prune_ratio: float = DEFAULT_PRUNE_RATIO
    max_edited_param_ratio: float = DEFAULT_MAX_EDITED_PARAM_RATIO
    budget_scope: str = "target"
    prune_scope: str = "global"
    eps: float = DEFAULT_EPS
    hessian_damping: float = DEFAULT_HESSIAN_DAMPING
    use_obs_compensation: bool = True
    use_woodbury: bool = True
    normalize_saliency_by_samples: bool = True
    protect_general_expert: bool = True
    general_expert_index: int = 0
    target_factors: tuple[str, ...] = ("w1_A", "w2_A", "w3_A")
    saliency_threshold: float | None = None
    max_comp_scale: float | None = None
    domain_guard_max_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    anse_safe_e0_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = DEFAULT_ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY
    min_specialist_gain_retention: float | None = None
    require_energy_improvement: bool = True
    general_prefix: str = GENERAL_EXPERT_PREFIX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = 16
    inplace: bool = True

    def __post_init__(self) -> None:
        _validate_unit_interval(self.prune_ratio, "prune_ratio", allow_zero=False, allow_one=False)
        _validate_unit_interval(
            self.max_edited_param_ratio, "max_edited_param_ratio", allow_zero=False, allow_one=True
        )
        if self.budget_scope not in VALID_BUDGET_SCOPES:
            raise ValueError(f"budget_scope must be one of {sorted(VALID_BUDGET_SCOPES)}")
        if self.prune_scope not in VALID_PRUNE_SCOPES:
            raise ValueError(f"prune_scope must be one of {sorted(VALID_PRUNE_SCOPES)}")
        _validate_positive_finite(self.eps, "eps")
        _validate_nonneg_finite(self.hessian_damping, "hessian_damping")
        _validate_bool(self.use_obs_compensation, "use_obs_compensation")
        _validate_bool(self.use_woodbury, "use_woodbury")
        _validate_bool(self.normalize_saliency_by_samples, "normalize_saliency_by_samples")
        _validate_bool(self.protect_general_expert, "protect_general_expert")
        _validate_nonneg_int(self.general_expert_index, "general_expert_index")
        if (
            isinstance(self.target_factors, (str, bytes))
            or not isinstance(self.target_factors, Sequence)
            or len(self.target_factors) == 0
        ):
            raise ValueError("target_factors must be a non-empty sequence of factor names")
        unknown = sorted({str(f) for f in self.target_factors} - VALID_TARGET_FACTORS)
        if unknown:
            raise ValueError(f"Unsupported target_factors: {unknown}")
        if self.saliency_threshold is not None:
            _validate_nonneg_finite(self.saliency_threshold, "saliency_threshold")
        if self.max_comp_scale is not None:
            _validate_positive_finite(self.max_comp_scale, "max_comp_scale")
        guard_r = _validate_positive_finite(self.domain_guard_max_ratio, "domain_guard_max_ratio")
        if guard_r < 1.0:
            raise ValueError(f"domain_guard_max_ratio must be >= 1.0, got {guard_r}")
        safe_r = _validate_positive_finite(self.anse_safe_e0_ratio, "anse_safe_e0_ratio")
        if safe_r < 1.0:
            raise ValueError(f"anse_safe_e0_ratio must be >= 1.0, got {safe_r}")
        _validate_nonneg_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        if self.min_specialist_gain_retention is not None:
            _validate_nonneg_finite(
                self.min_specialist_gain_retention, "min_specialist_gain_retention"
            )
        _validate_bool(self.require_energy_improvement, "require_energy_improvement")
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_bool(self.inplace, "inplace")


# ---------------------------------------------------------------------------
# 2. Discriminative Wanda Saliency & Specialist Hessian Inverse
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_discriminative_wanda_saliency(
    weight_delta: torch.Tensor,
    x_prose: torch.Tensor,
    x_spec: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
    normalize_by_samples: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute specialist-vs-prose Wanda activation-weighted saliency in CPU ``float64``.

    Scores every coordinate ``(i, j)`` of ``weight_delta`` by:

        ``S_{ij} = |Delta W_{ij}| * ||x_spec_j||_2 / (||x_prose_j||_2 + eps)``

    When ``normalize_by_samples=True``, column L2 norms are divided by ``sqrt(N)`` (column RMS)
    to normalize unequal prose and specialist calibration token counts.
    """
    w64 = _validate_finite_2d(weight_delta, "weight_delta")
    d_out, d_in = int(w64.shape[0]), int(w64.shape[1])
    xp_2d = _validate_activations_2d(x_prose, "x_prose", expected_dim=d_in)
    xs_2d = _validate_activations_2d(x_spec, "x_spec", expected_dim=d_in)
    eps_f = _validate_positive_finite(eps, "eps")
    norm_samples = _validate_bool(normalize_by_samples, "normalize_by_samples")

    spec_norm = torch.linalg.vector_norm(xs_2d, ord=2, dim=0)
    prose_norm = torch.linalg.vector_norm(xp_2d, ord=2, dim=0)
    if norm_samples:
        spec_norm = spec_norm / math.sqrt(float(xs_2d.shape[0]))
        prose_norm = prose_norm / math.sqrt(float(xp_2d.shape[0]))

    col_ratio = spec_norm / (prose_norm + eps_f)
    saliency = (w64.abs() * col_ratio.unsqueeze(0)).contiguous()

    nz_mask = w64 != 0.0
    nz_count = int(nz_mask.sum().item())
    if nz_count > 0:
        nz_vals = saliency[nz_mask]
        saliency_ratio = float(nz_vals.mean().item())
        min_nz = float(nz_vals.min().item())
        median_nz = float(nz_vals.median().item())
    else:
        saliency_ratio = 0.0
        min_nz = 0.0
        median_nz = 0.0

    diagnostics: dict[str, Any] = {
        "shape": (d_out, d_in),
        "eps": eps_f,
        "normalize_by_samples": norm_samples,
        "n_prose": int(xp_2d.shape[0]),
        "n_spec": int(xs_2d.shape[0]),
        "nonzero_count": nz_count,
        "saliency_ratio": saliency_ratio,
        "min_nonzero_saliency": min_nz,
        "median_nonzero_saliency": median_nz,
        "max_saliency": float(saliency.max().item()),
        "mean_col_ratio": float(col_ratio.mean().item()),
        "spec_norm_mean": float(spec_norm.mean().item()),
        "prose_norm_mean": float(prose_norm.mean().item()),
    }
    return saliency, diagnostics


@torch.no_grad()
def compute_specialist_hessian_inverse(
    x_spec: torch.Tensor,
    *,
    hessian_damping: float = DEFAULT_HESSIAN_DAMPING,
    ridge_floor: float = DEFAULT_RIDGE_FLOOR,
    mode: str = "auto",
    return_dense: bool = True,
) -> tuple[torch.Tensor | dict[str, torch.Tensor], dict[str, Any]]:
    """Build damped specialist inverse Hessian ``H_spec^{-1} = (E[x x^T] + lam I)^{-1}``.

    Supports ``"diagonal"`` (1D inverse diagonal), ``"dense"`` (Cholesky inverse),
    ``"woodbury"`` (exact Woodbury factor ``(1/lam) I - U U^T``), and ``"auto"``
    (choosing Woodbury when ``N < d_in`` and Cholesky otherwise).
    """
    xs_2d = _validate_activations_2d(x_spec, "x_spec")
    damp_f = _validate_nonneg_finite(hessian_damping, "hessian_damping")
    floor_f = _validate_positive_finite(ridge_floor, "ridge_floor")
    if mode not in VALID_HESSIAN_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_HESSIAN_MODES)}, got {mode!r}")
    ret_dense = _validate_bool(return_dense, "return_dense")

    n_tok, d_in = int(xs_2d.shape[0]), int(xs_2d.shape[1])
    diag_m = (xs_2d * xs_2d).mean(dim=0)
    tr_mean = max(float(diag_m.mean().item()), 0.0)
    lam = max(damp_f * tr_mean, floor_f)

    resolved_mode = mode
    if mode == "auto":
        resolved_mode = "woodbury" if n_tok < d_in else "dense"

    if resolved_mode == "diagonal":
        inv_diag = (1.0 / (diag_m + lam)).contiguous()
        return inv_diag, {
            "mode": "diagonal",
            "n_samples": n_tok,
            "dim": d_in,
            "damping_lambda": lam,
            "trace_mean": tr_mean,
            "woodbury_rank": 0,
        }

    if resolved_mode == "woodbury":
        v_mat = xs_2d / math.sqrt(float(n_tok))
        gram = (v_mat @ v_mat.T) / lam
        gram = 0.5 * (gram + gram.T)
        core = gram + torch.eye(n_tok, dtype=torch.float64)
        l_core = torch.linalg.cholesky(core)
        w_sol = torch.linalg.solve_triangular(l_core, v_mat, upper=False)
        u_woodbury = (w_sol.T / lam).contiguous()
        inv_diag_vec = torch.full((d_in,), 1.0 / lam, dtype=torch.float64)

        out: torch.Tensor | dict[str, torch.Tensor]
        if ret_dense:
            out = (torch.diag(inv_diag_vec) - (u_woodbury @ u_woodbury.T)).contiguous()
        else:
            out = {"inv_diag": inv_diag_vec, "woodbury_u": u_woodbury}
        return out, {
            "mode": "woodbury",
            "n_samples": n_tok,
            "dim": d_in,
            "damping_lambda": lam,
            "trace_mean": tr_mean,
            "woodbury_rank": n_tok,
            "factored": not ret_dense,
        }

    m_spec = (xs_2d.T @ xs_2d) / float(n_tok)
    m_spec = 0.5 * (m_spec + m_spec.T)
    h_mat = m_spec + lam * torch.eye(d_in, dtype=torch.float64)
    l_mat = torch.linalg.cholesky(h_mat)
    h_inv = torch.cholesky_inverse(l_mat)
    h_inv = (0.5 * (h_inv + h_inv.T)).contiguous()
    return h_inv, {
        "mode": "dense",
        "n_samples": n_tok,
        "dim": d_in,
        "damping_lambda": lam,
        "trace_mean": tr_mean,
        "woodbury_rank": 0,
    }


# ---------------------------------------------------------------------------
# 3. Closed-Form Diagonal & Woodbury OBS Row Surgery
# ---------------------------------------------------------------------------


def _parse_hessian_inverse_spec(
    h_spec: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | Mapping[str, torch.Tensor],
    d_in: int,
) -> dict[str, Any]:
    """Validate and normalize diagonal, dense, or Woodbury inverse Hessian specification."""
    if isinstance(h_spec, torch.Tensor):
        if not h_spec.is_floating_point():
            raise TypeError(
                f"h_spec_inv_diag_or_woodbury must be floating-point, got {h_spec.dtype}"
            )
        h_cpu = h_spec.detach().to(device="cpu", dtype=torch.float64).contiguous()
        if not bool(torch.isfinite(h_cpu).all().item()):
            raise ValueError("h_spec_inv_diag_or_woodbury must contain only finite values")

        if h_cpu.ndim == 1:
            if int(h_cpu.shape[0]) != d_in:
                raise ValueError(
                    f"1D diagonal inverse Hessian length {h_cpu.shape[0]} must match d_in={d_in}"
                )
            if bool((h_cpu <= 0.0).any().item()):
                raise ValueError("1D diagonal inverse Hessian entries must be strictly > 0.0")
            return {"mode": "diagonal_obs", "inv_diag": h_cpu}

        if h_cpu.ndim == 2:
            if tuple(h_cpu.shape) != (d_in, d_in):
                raise ValueError(
                    f"2D inverse Hessian shape {tuple(h_cpu.shape)} must be ({d_in}, {d_in})"
                )
            sym = 0.5 * (h_cpu + h_cpu.T)
            evals = torch.linalg.eigvalsh(sym)
            max_ev = max(float(evals[-1].item()), 0.0)
            min_ev = float(evals[0].item())
            if max_ev <= 0.0 or min_ev < -1e-6 * max(1.0, max_ev):
                raise ValueError(
                    f"2D inverse Hessian must be positive definite (min eigenvalue {min_ev:.3e})"
                )
            return {"mode": "dense_obs", "h_inv": sym.contiguous()}

        raise ValueError(
            f"h_spec_inv_diag_or_woodbury tensor must be 1D or 2D, got {h_cpu.ndim}D"
        )

    if isinstance(h_spec, tuple) and len(h_spec) == 2:
        raw_d, raw_u = h_spec
    elif isinstance(h_spec, Mapping):
        raw_d = h_spec.get("inv_diag", h_spec.get("d_inv"))
        raw_u = h_spec.get("woodbury_u", h_spec.get("u_woodbury"))
        if raw_d is None or raw_u is None:
            raise ValueError(
                "Woodbury mapping must provide 'inv_diag' (or 'd_inv') and 'woodbury_u'"
            )
    else:
        raise TypeError(
            "h_spec_inv_diag_or_woodbury must be a 1D/2D Tensor, (inv_diag, woodbury_u) tuple, "
            f"or Mapping; got {type(h_spec).__name__}"
        )

    if not isinstance(raw_d, torch.Tensor) or not raw_d.is_floating_point():
        raise TypeError("Woodbury inv_diag must be a floating-point torch.Tensor")
    d_vec = raw_d.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if d_vec.ndim != 1 or int(d_vec.shape[0]) != d_in:
        raise ValueError(f"Woodbury inv_diag must be 1D of length {d_in}, got {tuple(d_vec.shape)}")
    if not bool(torch.isfinite(d_vec).all().item()) or bool((d_vec <= 0.0).any().item()):
        raise ValueError("Woodbury inv_diag must contain strictly positive finite values")

    u_mat = _validate_finite_2d(raw_u, "woodbury_u")
    if int(u_mat.shape[0]) != d_in:
        raise ValueError(f"woodbury_u row count {u_mat.shape[0]} must match d_in={d_in}")

    diag_eff = d_vec - (u_mat * u_mat).sum(dim=1)
    if bool((diag_eff <= -1e-6 * float(d_vec.max().item())).any().item()):
        raise ValueError("Woodbury inverse Hessian diagonal entries must remain positive")
    return {"mode": "woodbury_obs", "inv_diag": d_vec, "woodbury_u": u_mat}


def _solve_spd_subblock(p_mm: torch.Tensor, rhs: torch.Tensor, ridge_jitter: float) -> torch.Tensor:
    """Solve ``(P_MM + jitter I) X = rhs`` reliably using Cholesky with symmetric fallback."""
    m = int(p_mm.shape[0])
    scale = max(float(torch.diagonal(p_mm).abs().mean().item()), 1e-12)
    jit = max(ridge_jitter * scale, 1e-12)
    p_sym = 0.5 * (p_mm + p_mm.T) + jit * torch.eye(m, dtype=torch.float64)
    l_chol, info = torch.linalg.cholesky_ex(p_sym)
    if int(info.item()) == 0:
        return torch.cholesky_solve(rhs, l_chol)
    return torch.linalg.lstsq(p_sym, rhs).solution


def _compute_obs_group_update(
    w_rows: torch.Tensor,
    idx_m: torch.Tensor,
    hess_ctx: Mapping[str, Any],
    *,
    ridge_jitter: float,
    max_comp_scale: float | None,
) -> tuple[torch.Tensor, float]:
    """Compute closed-form OBS compensated rows and total quadratic error for one mask group."""
    d_in = int(w_rows.shape[1])
    w_m = w_rows.index_select(1, idx_m)
    w_out = w_rows.clone()
    w_out[:, idx_m] = 0.0

    if int(idx_m.numel()) == d_in or float(w_m.abs().max().item()) == 0.0:
        return w_out, 0.0

    mode = hess_ctx["mode"]
    if mode == "diagonal_obs":
        inv_diag_m = hess_ctx["inv_diag"].index_select(0, idx_m).clamp_min(1e-12)
        quad = 0.5 * float(((w_m * w_m) / inv_diag_m.unsqueeze(0)).sum().item())
        return w_out, quad

    if mode == "dense_obs":
        h_inv: torch.Tensor = hess_ctx["h_inv"]
        p_mm = h_inv.index_select(0, idx_m).index_select(1, idx_m)
        nu = _solve_spd_subblock(p_mm, w_m.T, ridge_jitter)
        p_cols = h_inv.index_select(1, idx_m)
        delta_rows = -(nu.T @ p_cols.T)
    else:
        inv_diag: torch.Tensor = hess_ctx["inv_diag"]
        u_mat: torch.Tensor = hess_ctx["woodbury_u"]
        u_m = u_mat.index_select(0, idx_m)
        p_mm = torch.diag(inv_diag.index_select(0, idx_m)) - (u_m @ u_m.T)
        nu = _solve_spd_subblock(p_mm, w_m.T, ridge_jitter)
        u_nu = u_m.T @ nu
        delta_rows = (u_mat @ u_nu).T
        delta_rows[:, idx_m] -= nu.T * inv_diag.index_select(0, idx_m).unsqueeze(0)

    quad_loss = max(0.0, 0.5 * float((w_m * nu.T).sum().item()))

    comp_rows = delta_rows.clone()
    comp_rows[:, idx_m] = 0.0
    if max_comp_scale is not None:
        row_norms = torch.linalg.vector_norm(w_rows, ord=2, dim=1, keepdim=True)
        comp_norms = torch.linalg.vector_norm(comp_rows, ord=2, dim=1, keepdim=True)
        cap = max_comp_scale * row_norms
        scale = torch.where(comp_norms > cap, cap / comp_norms.clamp_min(1e-12), torch.ones_like(comp_norms))
        comp_rows = comp_rows * scale

    w_out = w_rows + comp_rows
    w_out[:, idx_m] = 0.0
    return w_out, quad_loss


@torch.no_grad()
def apply_obs_compensated_row_surgery(
    weight_delta: torch.Tensor,
    mask_zero: torch.Tensor,
    h_spec_inv_diag_or_woodbury: (
        torch.Tensor | tuple[torch.Tensor, torch.Tensor] | Mapping[str, torch.Tensor]
    ),
    *,
    inplace: bool = False,
    ridge_jitter: float = 1e-9,
    max_comp_scale: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Prune ``mask_zero`` coordinates with closed-form diagonal/Woodbury OBS compensation.

    For each row ``i`` with pruned coordinate set ``M_i``, solves the constrained quadratic
    problem ``min_{w'_i} 1/2 (w'_i - w_i)^T H_spec (w'_i - w_i)`` subject to ``w'_{i, M_i} = 0``:

        ``w'_i = w_i - P_{:, M_i} (P_{M_i, M_i})^{-1} w_{i, M_i}``   where ``P = H_spec^{-1}``.

    Supports dense 2D ``H_spec^{-1}``, factored Woodbury ``(inv_diag, woodbury_u)`` where
    ``P = diag(inv_diag) - U U^T``, or 1D diagonal ``diag(H_spec^{-1})``.
    """
    w64 = _validate_finite_2d(weight_delta, "weight_delta")
    d_out, d_in = int(w64.shape[0]), int(w64.shape[1])
    mask_cpu = _validate_mask_2d(mask_zero, (d_out, d_in), "mask_zero")
    _validate_bool(inplace, "inplace")
    jit_f = _validate_nonneg_finite(ridge_jitter, "ridge_jitter")
    comp_cap = (
        _validate_positive_finite(max_comp_scale, "max_comp_scale")
        if max_comp_scale is not None
        else None
    )
    hess_ctx = _parse_hessian_inverse_spec(h_spec_inv_diag_or_woodbury, d_in)

    masked_elements = int(mask_cpu.sum().item())
    if masked_elements == 0:
        w_ret = weight_delta if inplace else weight_delta.detach().clone()
        return w_ret, {
            "shape": (d_out, d_in),
            "mode": hess_ctx["mode"],
            "masked_elements": 0,
            "pruned_nonzero_elements": 0,
            "compensated_rows": 0,
            "edited_elements": 0,
            "edited_param_ratio": 0.0,
            "obs_quadratic_loss": 0.0,
            "delta_fro_norm": 0.0,
            "kept_comp_fro_norm": 0.0,
            "noop_reason": "empty_mask",
        }

    w_new64 = w64.clone()
    groups: dict[bytes, tuple[torch.Tensor, list[int]]] = {}
    for row_idx in range(d_out):
        row_mask = mask_cpu[row_idx]
        if not bool(row_mask.any().item()):
            continue
        key_bytes = row_mask.numpy().tobytes()
        if key_bytes not in groups:
            idx_m = torch.nonzero(row_mask, as_tuple=False).squeeze(-1)
            groups[key_bytes] = (idx_m, [row_idx])
        else:
            groups[key_bytes][1].append(row_idx)

    total_quad_loss = 0.0
    compensated_rows = 0
    for idx_m, row_indices in groups.values():
        row_t = torch.tensor(row_indices, dtype=torch.long)
        w_grp = w64.index_select(0, row_t)
        w_grp_out, quad_grp = _compute_obs_group_update(
            w_grp,
            idx_m,
            hess_ctx,
            ridge_jitter=jit_f,
            max_comp_scale=comp_cap,
        )
        w_new64[row_t] = w_grp_out
        total_quad_loss += quad_grp
        if int(idx_m.numel()) < d_in and hess_ctx["mode"] != "diagonal_obs":
            kept_diff = (w_grp_out - w_grp)[:, ~mask_cpu[row_indices[0]]]
            if kept_diff.numel() > 0:
                compensated_rows += int((kept_diff.abs().max(dim=1).values > 1e-13).sum().item())

    w_cast = w_new64.to(device=weight_delta.device, dtype=weight_delta.dtype)
    if inplace:
        weight_delta.copy_(w_cast)
        w_result = weight_delta
    else:
        w_result = w_cast.contiguous()

    final64 = w_result.detach().to(device="cpu", dtype=torch.float64)
    diff64 = final64 - w64
    kept_diff64 = torch.where(~mask_cpu, diff64, torch.zeros_like(diff64))
    pruned_nonzero = int(((w64 != 0.0) & mask_cpu).sum().item())
    edited_elements = int(torch.ne(final64, w64).sum().item())

    return w_result, {
        "shape": (d_out, d_in),
        "mode": hess_ctx["mode"],
        "masked_elements": masked_elements,
        "pruned_nonzero_elements": pruned_nonzero,
        "compensated_rows": compensated_rows,
        "edited_elements": edited_elements,
        "edited_param_ratio": float(edited_elements) / float(w64.numel()),
        "obs_quadratic_loss": float(total_quad_loss),
        "delta_fro_norm": float(torch.linalg.matrix_norm(diff64, ord="fro").item()),
        "kept_comp_fro_norm": float(torch.linalg.matrix_norm(kept_diff64, ord="fro").item()),
        "noop_reason": None if edited_elements > 0 else "zero_masked_entries",
    }


# ---------------------------------------------------------------------------
# 4. Model Traversal, Budgeted Selection & End-to-End ANSE Verification
# ---------------------------------------------------------------------------


def _validate_crowd_model_compatible(model: nn.Module) -> None:
    """Verify required CrowdV1-compatible attributes for token-chunk evaluation."""
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("model must expose a non-empty 'blocks' sequence")
    for attr in ("wte", "ln", "head", "cos_cached", "sin_cached"):
        if not hasattr(model, attr):
            raise ValueError(f"model must expose '{attr}'")


def _validate_guard_chunks(
    guard_chunks: Sequence[Mapping[str, Any]],
    *,
    general_prefix: str,
    score_from: int,
) -> tuple[torch.Tensor, list[str]]:
    """Validate guard chunks and return ``(ids_2d, expert_ids)``."""
    if (
        isinstance(guard_chunks, (str, bytes, bytearray))
        or not isinstance(guard_chunks, Sequence)
        or len(guard_chunks) == 0
    ):
        raise ValueError("guard_chunks must be a non-empty sequence of chunk mappings")

    rows: list[torch.Tensor] = []
    expert_ids: list[str] = []
    seq_len: int | None = None
    for idx, ch in enumerate(guard_chunks):
        if not isinstance(ch, Mapping) or "expert_id" not in ch or "ids" not in ch:
            raise ValueError(f"guard_chunks[{idx}] must be a mapping with 'expert_id' and 'ids'")
        eid = str(ch["expert_id"]).strip()
        if not eid:
            raise ValueError(f"guard_chunks[{idx}]['expert_id'] must be a non-empty string")
        ids_t = torch.as_tensor(ch["ids"], dtype=torch.long)
        if ids_t.ndim != 1 or ids_t.numel() < 2:
            raise ValueError(f"guard_chunks[{idx}]['ids'] must be 1D with length >= 2")
        if bool((ids_t < 0).any().item()):
            raise ValueError(f"guard_chunks[{idx}]['ids'] must contain non-negative token IDs")
        if seq_len is None:
            seq_len = int(ids_t.numel())
        elif int(ids_t.numel()) != seq_len:
            raise ValueError(
                f"All guard_chunks must share sequence length {seq_len}, got {ids_t.numel()} at {idx}"
            )
        rows.append(ids_t)
        expert_ids.append(eid)

    assert seq_len is not None
    start_pos = max(0, score_from - 1)
    if start_pos >= seq_len - 1:
        raise ValueError(f"Sequence length {seq_len} too short for score_from={score_from}")
    has_general = any(eid.startswith(general_prefix) for eid in expert_ids)
    has_spec = any(not eid.startswith(general_prefix) for eid in expert_ids)
    if not has_general or not has_spec:
        raise ValueError(
            f"guard_chunks must contain both general ({general_prefix}*) and specialist chunks"
        )
    return torch.stack(rows, dim=0), expert_ids


def _parse_eval_domain_nll(raw_eval: Any) -> dict[str, float]:
    """Normalize evaluator output into a validated ``domain -> finite positive NLL`` map."""
    payload: Any = raw_eval
    if isinstance(raw_eval, tuple) and len(raw_eval) == 2 and isinstance(raw_eval[1], Mapping):
        payload = raw_eval[1]
    if not isinstance(payload, Mapping) or not payload:
        raise TypeError("eval_fn must return a non-empty mapping with domain NLL metrics")

    domain_map: Mapping[str, Any] = payload
    for key in ("domain_nll", "per_domain_nll"):
        if key in payload:
            sub = payload[key]
            if not isinstance(sub, Mapping) or not sub:
                raise ValueError(f"eval[{key!r}] must be a non-empty mapping")
            domain_map = sub
            break

    out: dict[str, float] = {}
    for dom, val in domain_map.items():
        out[str(dom)] = _validate_positive_finite(val, f"domain_nll[{dom!r}]")
    return out


@torch.no_grad()
def _eval_model_guard_nlls(
    model: nn.Module,
    ids_2d: torch.Tensor,
    expert_ids: Sequence[str],
    *,
    score_from: int,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate per-domain causal mid-row NLL across all transformer blocks."""
    dev = model.wte.weight.device
    cos = model.cos_cached.to(dev)
    sin = model.sin_cached.to(dev)
    parts: list[torch.Tensor] = []
    for b0 in range(0, ids_2d.shape[0], batch_size):
        xb = compute_embeddings_batch(model, ids_2d[b0 : b0 + batch_size].to(dev))
        for blk in model.blocks:
            xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
            mlp_out = blk.mlp(mi.to(dtype=blk.ln1.weight.dtype)).to(torch.float32)
            xb = (xm + mlp_out).to(torch.float32)
        parts.append(xb)

    h_pre = torch.cat(parts, dim=0)
    if not bool(torch.isfinite(h_pre).all().item()):
        raise ValueError("Non-finite hidden states encountered during guard evaluation")

    nlls = per_chunk_nll_from_pre_ln(
        h_pre,
        model.ln.weight,
        model.ln.bias,
        float(getattr(model.ln, "eps", 1e-5)),
        model.head.weight,
        ids_2d.to(dev),
        score_from=score_from,
    ).cpu().numpy()
    if not bool(np.isfinite(nlls).all()):
        raise ValueError("Non-finite NLL encountered during guard evaluation")

    grouped: dict[str, list[float]] = {}
    for eid, val in zip(expert_ids, nlls.tolist(), strict=True):
        grouped.setdefault(str(eid), []).append(float(val))
    return {k: sum(v) / float(len(v)) for k, v in sorted(grouped.items())}


def _summarize_anse_metrics(
    ref_nll: Mapping[str, float],
    cand_nll: Mapping[str, float],
    *,
    pre_nll: Mapping[str, float],
    tuned_nll: Mapping[str, float] | None,
    cfg: WandaOBSSurgeonConfig,
) -> dict[str, Any]:
    """Compute per-domain NLL drift, specialist retention, guard verdict, and ANSE energy."""
    if set(ref_nll.keys()) != set(cand_nll.keys()):
        raise ValueError("Reference and candidate domain NLL mappings must share identical keys")

    per_domain: dict[str, dict[str, Any]] = {}
    worst_domain = ""
    worst_ratio = -math.inf
    cand_vals: list[float] = []
    ref_vals: list[float] = []

    for dom in sorted(ref_nll.keys()):
        r_v = float(ref_nll[dom])
        c_v = float(cand_nll[dom])
        ratio = c_v / max(r_v, 1e-8)
        per_domain[dom] = {
            "base_nll": r_v,
            "nll_mean": c_v,
            "pre_nll": float(pre_nll[dom]),
            "ratio": ratio,
            "pct_vs_base": 100.0 * (ratio - 1.0),
            "guard_ok": bool(ratio <= cfg.domain_guard_max_ratio + 1e-12),
        }
        cand_vals.append(c_v)
        ref_vals.append(r_v)
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_domain = dom

    general_keys = [
        d
        for d in sorted(ref_nll.keys())
        if d.startswith(cfg.general_prefix) or d in ("general", "general_nll")
    ]
    spec_keys = [d for d in sorted(ref_nll.keys()) if d not in general_keys]

    e0_ratio = max(per_domain[k]["ratio"] for k in general_keys) if general_keys else worst_ratio
    guard_ok = all(bool(v["guard_ok"]) for v in per_domain.values())

    nll_mean = float(sum(cand_vals) / float(len(cand_vals)))
    ref_mean = float(sum(ref_vals) / float(len(ref_vals)))

    spec_gain_nats = 0.0
    spec_gain_retained_frac = 1.0
    spec_drift_pct = 0.0
    if spec_keys:
        base_spec = sum(ref_nll[k] for k in spec_keys) / float(len(spec_keys))
        pre_spec = sum(pre_nll[k] for k in spec_keys) / float(len(spec_keys))
        cand_spec = sum(cand_nll[k] for k in spec_keys) / float(len(spec_keys))
        spec_gain_nats = base_spec - cand_spec
        spec_drift_pct = 100.0 * ((cand_spec / max(pre_spec, 1e-8)) - 1.0)
        ref_tuned_spec = (
            sum(tuned_nll[k] for k in spec_keys) / float(len(spec_keys))
            if tuned_nll is not None and all(k in tuned_nll for k in spec_keys)
            else pre_spec
        )
        raw_gain = base_spec - ref_tuned_spec
        if abs(raw_gain) > 1e-8:
            spec_gain_retained_frac = spec_gain_nats / raw_gain
        else:
            spec_gain_retained_frac = pre_spec / max(cand_spec, 1e-8)

    retention_ok = (
        spec_gain_retained_frac >= cfg.min_specialist_gain_retention
        if cfg.min_specialist_gain_retention is not None
        else True
    )
    overall_guard = bool(guard_ok and retention_ok)
    pain = (
        cfg.maximum_pain_energy
        if not overall_guard
        else (cfg.soft_margin_penalty if e0_ratio > cfg.anse_safe_e0_ratio else 0.0)
    )
    anse_energy = float(nll_mean + pain)

    return {
        "nll_mean": nll_mean,
        "base_nll_mean": ref_mean,
        "mean_diff_vs_base": nll_mean - ref_mean,
        "worst_domain_id": worst_domain,
        "worst_domain_ratio": worst_ratio,
        "worst_domain_pct_vs_base": 100.0 * (worst_ratio - 1.0),
        "general_ratio": e0_ratio,
        "general_pct_vs_base": 100.0 * (e0_ratio - 1.0),
        "specialist_gain_nats": spec_gain_nats,
        "specialist_gain_retained_frac": spec_gain_retained_frac,
        "specialist_drift_pct_vs_pre": spec_drift_pct,
        "guard_ok": guard_ok,
        "retention_ok": retention_ok,
        "anse_energy": anse_energy,
        "per_domain": per_domain,
    }


def _select_wanda_mask_from_saliency(
    targets: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    prune_ratio: float,
    prune_scope: str,
    saliency_threshold: float | None,
) -> dict[str, torch.Tensor]:
    """Select boolean pruning masks from discriminative Wanda saliency tensors."""
    masks: dict[str, torch.Tensor] = {}
    if prune_scope == "global":
        entries: list[tuple[float, int, int, int]] = []
        for idx, (_, w_t, sal, _) in enumerate(targets):
            w_cpu = w_t.detach().to(device="cpu", dtype=torch.float64)
            active = w_cpu.abs() > 1e-12
            masks[str(idx)] = torch.zeros_like(active)
            rc, cc = torch.nonzero(active, as_tuple=True)
            for r_i, c_i in zip(rc.tolist(), cc.tolist(), strict=True):
                s_val = float(sal[r_i, c_i].item())
                if saliency_threshold is None or s_val <= saliency_threshold:
                    entries.append((s_val, idx, r_i, c_i))
        if entries:
            total_active = sum(
                int((w_t.detach().to(device="cpu").abs() > 1e-12).sum().item())
                for _, w_t, _, _ in targets
            )
            k_target = min(len(entries), max(1, int(round(prune_ratio * float(total_active)))))
            entries.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            for _, idx, r_i, c_i in entries[:k_target]:
                masks[str(idx)][r_i, c_i] = True
        return {targets[int(k)][0]: v for k, v in masks.items()}

    for name, w_t, sal, _ in targets:
        w_cpu = w_t.detach().to(device="cpu", dtype=torch.float64)
        active = w_cpu.abs() > 1e-12
        mask = torch.zeros_like(active)
        if prune_scope == "matrix":
            rc, cc = torch.nonzero(active, as_tuple=True)
            pool = [
                (float(sal[r_i, c_i].item()), r_i, c_i)
                for r_i, c_i in zip(rc.tolist(), cc.tolist(), strict=True)
                if saliency_threshold is None or float(sal[r_i, c_i].item()) <= saliency_threshold
            ]
            if pool:
                n_act = int(active.sum().item())
                k_keep = min(len(pool), max(1, int(round(prune_ratio * float(n_act)))))
                pool.sort(key=lambda item: (item[0], item[1], item[2]))
                for _, r_i, c_i in pool[:k_keep]:
                    mask[r_i, c_i] = True
        else:
            for r_i in range(int(w_cpu.shape[0])):
                cols = torch.nonzero(active[r_i], as_tuple=False).squeeze(-1).tolist()
                pool_r = [
                    (float(sal[r_i, c_i].item()), c_i)
                    for c_i in cols
                    if saliency_threshold is None or float(sal[r_i, c_i].item()) <= saliency_threshold
                ]
                if pool_r:
                    k_r = min(len(pool_r), max(1, int(round(prune_ratio * float(len(cols))))))
                    pool_r.sort(key=lambda item: (item[0], item[1]))
                    for _, c_i in pool_r[:k_r]:
                        mask[r_i, c_i] = True
        masks[name] = mask
    return masks


def _apply_budgeted_surgery_group(
    targets: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
    masks: Mapping[str, torch.Tensor],
    *,
    remaining_budget: int,
    cfg: WandaOBSSurgeonConfig,
    layer_reports: list[dict[str, Any]],
) -> int:
    """Apply OBS surgery across ``targets`` while strictly respecting ``remaining_budget``."""
    if remaining_budget <= 0:
        return 0

    h_mode = "diagonal" if not cfg.use_obs_compensation else ("auto" if cfg.use_woodbury else "dense")
    hess_cache: dict[int, tuple[torch.Tensor | dict[str, torch.Tensor], dict[str, Any]]] = {}
    for _, _, _, xs in targets:
        key_id = id(xs)
        if key_id not in hess_cache:
            hess_cache[key_id] = compute_specialist_hessian_inverse(
                xs,
                hessian_damping=cfg.hessian_damping,
                mode=h_mode,
                return_dense=False,
            )

    if remaining_budget >= sum(int(w_t.numel()) for _, w_t, _, _ in targets):
        spent = 0
        for name, w_t, sal, xs in targets:
            mask = masks[name]
            h_inv, _ = hess_cache[id(xs)]
            _, rep = apply_obs_compensated_row_surgery(
                w_t,
                mask,
                h_inv,
                inplace=True,
                max_comp_scale=cfg.max_comp_scale,
            )
            rep["target"] = name
            rep["saliency_ratio"] = float(sal[w_t.detach().cpu() != 0.0].mean().item()) if bool((w_t != 0).any().item()) else 0.0
            layer_reports.append(rep)
            spent += int(rep["edited_elements"])
        return max(0, remaining_budget - spent)

    row_candidates: list[tuple[float, int, int, torch.Tensor, torch.Tensor]] = []
    for t_idx, (name, w_t, sal, xs) in enumerate(targets):
        mask = masks[name]
        h_inv, _ = hess_cache[id(xs)]
        w_cand, _ = apply_obs_compensated_row_surgery(
            w_t,
            mask,
            h_inv,
            inplace=False,
            max_comp_scale=cfg.max_comp_scale,
        )
        w64 = w_t.detach().to(device="cpu", dtype=torch.float64)
        cand64 = w_cand.detach().to(device="cpu", dtype=torch.float64)
        for r_i in range(int(w64.shape[0])):
            if not bool(mask[r_i].any().item()):
                continue
            pruned_sal = sal[r_i, mask[r_i]]
            priority = float(w64[r_i, mask[r_i]].abs().sum().item()) / max(float(pruned_sal.mean().item()), 1e-12)
            row_candidates.append((priority, t_idx, r_i, cand64[r_i], mask[r_i]))

    row_candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    before_snaps = [w_t.detach().to(device="cpu", dtype=torch.float64).clone() for _, w_t, _, _ in targets]
    budget_left = remaining_budget

    for _, t_idx, r_i, cand_row, row_mask in row_candidates:
        if budget_left <= 0:
            break
        w_t = targets[t_idx][1]
        orig_row = w_t[r_i].detach().to(device="cpu", dtype=torch.float64).clone()
        cand_cast = cand_row.to(device=w_t.device, dtype=w_t.dtype)
        orig_cast = w_t[r_i].detach().clone()
        n_edits = int(torch.ne(cand_cast, orig_cast).sum().item())
        if n_edits == 0:
            continue
        if n_edits <= budget_left:
            w_t[r_i].copy_(cand_cast)
            budget_left -= int(torch.ne(w_t[r_i].detach(), orig_cast).sum().item())
        else:
            sal = targets[t_idx][2]
            prune_cols = torch.nonzero(row_mask & (orig_row != 0.0), as_tuple=False).squeeze(-1)
            if prune_cols.numel() == 0:
                continue
            order_p = torch.argsort(sal[r_i, prune_cols])
            k_p = min(int(prune_cols.numel()), budget_left)
            chosen_p = prune_cols[order_p[:k_p]]
            row_patch = orig_row.clone()
            row_patch[chosen_p] = 0.0
            comp_slots = budget_left - k_p
            if comp_slots > 0 and cfg.use_obs_compensation:
                kept_mask = ~row_mask
                kept_diff = torch.where(kept_mask, cand_row - orig_row, torch.zeros_like(orig_row))
                kept_cols = torch.nonzero(kept_diff.abs() > 1e-13, as_tuple=False).squeeze(-1)
                if kept_cols.numel() > 0:
                    top_k_c = torch.argsort(kept_diff[kept_cols].abs(), descending=True)[:comp_slots]
                    chosen_c = kept_cols[top_k_c]
                    row_patch[chosen_c] = cand_row[chosen_c]
            w_t[r_i].copy_(row_patch.to(device=w_t.device, dtype=w_t.dtype))
            budget_left -= int(torch.ne(w_t[r_i].detach(), orig_cast).sum().item())

    for t_idx, (name, w_t, sal, _) in enumerate(targets):
        after64 = w_t.detach().to(device="cpu", dtype=torch.float64)
        edits = int(torch.ne(after64, before_snaps[t_idx]).sum().item())
        layer_reports.append(
            {
                "target": name,
                "shape": tuple(w_t.shape),
                "mode": f"{h_mode}_budgeted",
                "edited_elements": edits,
                "pruned_nonzero_elements": int(((before_snaps[t_idx] != 0.0) & (after64 == 0.0)).sum().item()),
                "saliency_ratio": float(sal[after64 != 0.0].mean().item()) if bool((after64 != 0.0).any().item()) else 0.0,
            }
        )
    return max(0, budget_left)


def _compute_lora_expert_swiglu_acts(
    moe_mlp: CrowdLoRAMoEMLP,
    exp_mod: Any,
    x_in: torch.Tensor,
) -> torch.Tensor:
    """Compute post-LoRA SwiGLU intermediate states for a single ``CrowdLoRAExpertAdapter``."""
    w_dt = moe_mlp.base_mlp.w1.weight.dtype
    x_base = x_in.to(device=moe_mlp.base_mlp.w1.weight.device, dtype=w_dt)
    h1 = moe_mlp.base_mlp.w1(x_base).to(torch.float32)
    h2 = moe_mlp.base_mlp.w2(x_base).to(torch.float32)
    x_fp32 = x_in.to(device=exp_mod.w1_A.device, dtype=torch.float32)
    dh1 = F.linear(F.linear(x_fp32, exp_mod.w1_A), exp_mod.w1_B) * exp_mod.scaling
    dh2 = F.linear(F.linear(x_fp32, exp_mod.w2_A), exp_mod.w2_B) * exp_mod.scaling
    return (F.silu(h1 + dh1) * (h2 + dh2)).contiguous()


def _collect_trainable_and_target_counts(
    model: nn.Module,
    *,
    target_layer_set: set[int] | None,
    target_expert_set: set[int] | None,
    target_factor_set: set[str],
    cfg: WandaOBSSurgeonConfig,
) -> tuple[int, int, int]:
    """Return ``(model_params, trainable_params, target_params)`` for budget accounting."""
    model_params = sum(int(p.numel()) for p in model.parameters())
    trainable_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        trainable_params = model_params

    blocks = getattr(model, "blocks", None)
    mlp_nodes = [
        (l_idx, getattr(blk.mlp, "moe_mlp", blk.mlp))
        for l_idx, blk in enumerate(blocks)
        if target_layer_set is None or l_idx in target_layer_set
    ] if blocks is not None else [(0, getattr(model, "moe_mlp", model))]

    target_params = 0
    for _, mlp_mod in mlp_nodes:
        if isinstance(mlp_mod, CrowdLoRAMoEMLP):
            for e_key in sorted(mlp_mod.experts.keys(), key=int):
                e_idx = int(e_key)
                if not _is_expert_targeted(e_idx, target_expert_set, cfg):
                    continue
                exp = mlp_mod.experts[e_key]
                for f_name in sorted(target_factor_set):
                    if f_name in VALID_TARGET_FACTORS and hasattr(exp, f_name):
                        base_attr = f_name.split("_", 1)[0]
                        b_ten = getattr(exp, f"{base_attr}_B", None)
                        a_ten = getattr(exp, f"{base_attr}_A", None)
                        if (
                            isinstance(b_ten, torch.Tensor)
                            and isinstance(a_ten, torch.Tensor)
                            and float(b_ten.detach().abs().max().item()) > 1e-12
                            and float(a_ten.detach().abs().max().item()) > 1e-12
                        ):
                            target_params += int(getattr(exp, f_name).numel())
        elif isinstance(mlp_mod, CrowdFineGrainedMoEMLP):
            for e_idx in range(mlp_mod.num_experts):
                if not _is_expert_targeted(e_idx, target_expert_set, cfg):
                    continue
                for f_name, mod_list in (
                    ("w1", mlp_mod.experts_w1),
                    ("w2", mlp_mod.experts_w2),
                    ("w3", mlp_mod.experts_w3),
                ):
                    if f_name in target_factor_set or f"{f_name}_A" in target_factor_set:
                        target_params += int(mod_list[e_idx].weight.numel())
        elif isinstance(mlp_mod, nn.Linear):
            target_params += int(mlp_mod.weight.numel())
        else:
            for f_name in ("w1", "w2", "w3"):
                mod = getattr(mlp_mod, f_name, None)
                if isinstance(mod, nn.Linear) and (
                    f_name in target_factor_set or f"{f_name}_A" in target_factor_set
                ):
                    target_params += int(mod.weight.numel())

    return model_params, trainable_params, max(1, target_params)


def _is_expert_targeted(
    e_idx: int,
    target_expert_set: set[int] | None,
    cfg: WandaOBSSurgeonConfig,
) -> bool:
    """Return True iff expert ``e_idx`` is eligible for surgical sparsification."""
    if target_expert_set is not None:
        return e_idx in target_expert_set
    if cfg.protect_general_expert and e_idx == cfg.general_expert_index:
        return False
    return True


def _run_layer_wanda_obs_surgery(
    layer_label: str,
    mlp_mod: nn.Module,
    xp_in: torch.Tensor,
    xs_in: torch.Tensor,
    *,
    target_expert_set: set[int] | None,
    target_factor_set: set[str],
    remaining_budget: int,
    cfg: WandaOBSSurgeonConfig,
    layer_reports: list[dict[str, Any]],
) -> int:
    """Execute Wanda/OBS surgery on one MoE or dense MLP module and return updated budget."""
    budget = remaining_budget
    if isinstance(mlp_mod, CrowdLoRAMoEMLP):
        stage1_targets: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        active_experts: list[tuple[str, Any]] = []
        for e_key in sorted(mlp_mod.experts.keys(), key=int):
            e_idx = int(e_key)
            if not _is_expert_targeted(e_idx, target_expert_set, cfg):
                continue
            exp = mlp_mod.experts[e_key]
            active_experts.append((e_key, exp))
            for base in ("w1", "w2"):
                a_ten = getattr(exp, f"{base}_A")
                b_ten = getattr(exp, f"{base}_B")
                if (
                    float(b_ten.detach().abs().max().item()) <= 1e-12
                    or float(a_ten.detach().abs().max().item()) <= 1e-12
                ):
                    continue
                if f"{base}_A" in target_factor_set:
                    sal_a, _ = compute_discriminative_wanda_saliency(
                        a_ten,
                        xp_in,
                        xs_in,
                        eps=cfg.eps,
                        normalize_by_samples=cfg.normalize_saliency_by_samples,
                    )
                    stage1_targets.append((f"{layer_label}.expert_{e_key}.{base}_A", a_ten, sal_a, xs_in))
                if f"{base}_B" in target_factor_set:
                    zp = (F.linear(xp_in.to(a_ten.device, a_ten.dtype), a_ten) * exp.scaling).detach()
                    zs = (F.linear(xs_in.to(a_ten.device, a_ten.dtype), a_ten) * exp.scaling).detach()
                    sal_b, _ = compute_discriminative_wanda_saliency(
                        b_ten,
                        zp,
                        zs,
                        eps=cfg.eps,
                        normalize_by_samples=cfg.normalize_saliency_by_samples,
                    )
                    stage1_targets.append((f"{layer_label}.expert_{e_key}.{base}_B", b_ten, sal_b, zs))

        if stage1_targets and budget > 0:
            masks1 = _select_wanda_mask_from_saliency(
                stage1_targets,
                prune_ratio=cfg.prune_ratio,
                prune_scope=cfg.prune_scope,
                saliency_threshold=cfg.saliency_threshold,
            )
            budget = _apply_budgeted_surgery_group(
                stage1_targets,
                masks1,
                remaining_budget=budget,
                cfg=cfg,
                layer_reports=layer_reports,
            )

        if ("w3_A" in target_factor_set or "w3_B" in target_factor_set) and budget > 0:
            stage2_targets: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for e_key, exp in active_experts:
                if (
                    float(exp.w3_B.detach().abs().max().item()) <= 1e-12
                    or float(exp.w3_A.detach().abs().max().item()) <= 1e-12
                ):
                    continue
                act_p = _compute_lora_expert_swiglu_acts(mlp_mod, exp, xp_in)
                act_s = _compute_lora_expert_swiglu_acts(mlp_mod, exp, xs_in)
                if "w3_A" in target_factor_set:
                    sal_3a, _ = compute_discriminative_wanda_saliency(
                        exp.w3_A,
                        act_p,
                        act_s,
                        eps=cfg.eps,
                        normalize_by_samples=cfg.normalize_saliency_by_samples,
                    )
                    stage2_targets.append((f"{layer_label}.expert_{e_key}.w3_A", exp.w3_A, sal_3a, act_s))
                if "w3_B" in target_factor_set:
                    zp3 = (F.linear(act_p.to(exp.w3_A.device, exp.w3_A.dtype), exp.w3_A) * exp.scaling).detach()
                    zs3 = (F.linear(act_s.to(exp.w3_A.device, exp.w3_A.dtype), exp.w3_A) * exp.scaling).detach()
                    sal_3b, _ = compute_discriminative_wanda_saliency(
                        exp.w3_B,
                        zp3,
                        zs3,
                        eps=cfg.eps,
                        normalize_by_samples=cfg.normalize_saliency_by_samples,
                    )
                    stage2_targets.append((f"{layer_label}.expert_{e_key}.w3_B", exp.w3_B, sal_3b, zs3))

            if stage2_targets:
                masks2 = _select_wanda_mask_from_saliency(
                    stage2_targets,
                    prune_ratio=cfg.prune_ratio,
                    prune_scope=cfg.prune_scope,
                    saliency_threshold=cfg.saliency_threshold,
                )
                budget = _apply_budgeted_surgery_group(
                    stage2_targets,
                    masks2,
                    remaining_budget=budget,
                    cfg=cfg,
                    layer_reports=layer_reports,
                )
        return budget

    if isinstance(mlp_mod, CrowdFineGrainedMoEMLP):
        fg_targets1: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        fg_experts: list[int] = []
        for e_idx in range(mlp_mod.num_experts):
            if not _is_expert_targeted(e_idx, target_expert_set, cfg):
                continue
            fg_experts.append(e_idx)
            for fname, mod_list in (("w1", mlp_mod.experts_w1), ("w2", mlp_mod.experts_w2)):
                if fname in target_factor_set or f"{fname}_A" in target_factor_set:
                    w_ten = mod_list[e_idx].weight
                    sal, _ = compute_discriminative_wanda_saliency(
                        w_ten, xp_in, xs_in, eps=cfg.eps, normalize_by_samples=cfg.normalize_saliency_by_samples
                    )
                    fg_targets1.append((f"{layer_label}.expert_{e_idx}.{fname}", w_ten, sal, xs_in))
        if fg_targets1 and budget > 0:
            masks_fg1 = _select_wanda_mask_from_saliency(
                fg_targets1,
                prune_ratio=cfg.prune_ratio,
                prune_scope=cfg.prune_scope,
                saliency_threshold=cfg.saliency_threshold,
            )
            budget = _apply_budgeted_surgery_group(
                fg_targets1, masks_fg1, remaining_budget=budget, cfg=cfg, layer_reports=layer_reports
            )
        if ("w3" in target_factor_set or "w3_A" in target_factor_set) and budget > 0:
            fg_targets2: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for e_idx in fg_experts:
                w1_e, w2_e, w3_e = mlp_mod.experts_w1[e_idx], mlp_mod.experts_w2[e_idx], mlp_mod.experts_w3[e_idx]
                act_p = F.silu(w1_e(xp_in.to(w1_e.weight.dtype))) * w2_e(xp_in.to(w2_e.weight.dtype))
                act_s = F.silu(w1_e(xs_in.to(w1_e.weight.dtype))) * w2_e(xs_in.to(w2_e.weight.dtype))
                sal, _ = compute_discriminative_wanda_saliency(
                    w3_e.weight,
                    act_p.to(torch.float32),
                    act_s.to(torch.float32),
                    eps=cfg.eps,
                    normalize_by_samples=cfg.normalize_saliency_by_samples,
                )
                fg_targets2.append((f"{layer_label}.expert_{e_idx}.w3", w3_e.weight, sal, act_s.to(torch.float32)))
            if fg_targets2:
                masks_fg2 = _select_wanda_mask_from_saliency(
                    fg_targets2,
                    prune_ratio=cfg.prune_ratio,
                    prune_scope=cfg.prune_scope,
                    saliency_threshold=cfg.saliency_threshold,
                )
                budget = _apply_budgeted_surgery_group(
                    fg_targets2, masks_fg2, remaining_budget=budget, cfg=cfg, layer_reports=layer_reports
                )
        return budget

    if isinstance(mlp_mod, nn.Linear):
        sal, _ = compute_discriminative_wanda_saliency(
            mlp_mod.weight, xp_in, xs_in, eps=cfg.eps, normalize_by_samples=cfg.normalize_saliency_by_samples
        )
        t_list = [(f"{layer_label}.weight", mlp_mod.weight, sal, xs_in)]
        masks_l = _select_wanda_mask_from_saliency(
            t_list,
            prune_ratio=cfg.prune_ratio,
            prune_scope=cfg.prune_scope,
            saliency_threshold=cfg.saliency_threshold,
        )
        return _apply_budgeted_surgery_group(
            t_list, masks_l, remaining_budget=budget, cfg=cfg, layer_reports=layer_reports
        )

    if hasattr(mlp_mod, "w1") and hasattr(mlp_mod, "w2") and hasattr(mlp_mod, "w3"):
        d_targets: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for fname in ("w1", "w2"):
            mod = getattr(mlp_mod, fname)
            if isinstance(mod, nn.Linear) and (fname in target_factor_set or f"{fname}_A" in target_factor_set):
                sal, _ = compute_discriminative_wanda_saliency(
                    mod.weight, xp_in, xs_in, eps=cfg.eps, normalize_by_samples=cfg.normalize_saliency_by_samples
                )
                d_targets.append((f"{layer_label}.{fname}", mod.weight, sal, xs_in))
        if d_targets and budget > 0:
            masks_d1 = _select_wanda_mask_from_saliency(
                d_targets,
                prune_ratio=cfg.prune_ratio,
                prune_scope=cfg.prune_scope,
                saliency_threshold=cfg.saliency_threshold,
            )
            budget = _apply_budgeted_surgery_group(
                d_targets, masks_d1, remaining_budget=budget, cfg=cfg, layer_reports=layer_reports
            )
        if ("w3" in target_factor_set or "w3_A" in target_factor_set) and budget > 0 and isinstance(mlp_mod.w3, nn.Linear):
            act_p = F.silu(mlp_mod.w1(xp_in.to(mlp_mod.w1.weight.dtype))) * mlp_mod.w2(xp_in.to(mlp_mod.w2.weight.dtype))
            act_s = F.silu(mlp_mod.w1(xs_in.to(mlp_mod.w1.weight.dtype))) * mlp_mod.w2(xs_in.to(mlp_mod.w2.weight.dtype))
            sal3, _ = compute_discriminative_wanda_saliency(
                mlp_mod.w3.weight,
                act_p.to(torch.float32),
                act_s.to(torch.float32),
                eps=cfg.eps,
                normalize_by_samples=cfg.normalize_saliency_by_samples,
            )
            t3 = [(f"{layer_label}.w3", mlp_mod.w3.weight, sal3, act_s.to(torch.float32))]
            masks_d3 = _select_wanda_mask_from_saliency(
                t3,
                prune_ratio=cfg.prune_ratio,
                prune_scope=cfg.prune_scope,
                saliency_threshold=cfg.saliency_threshold,
            )
            budget = _apply_budgeted_surgery_group(
                t3, masks_d3, remaining_budget=budget, cfg=cfg, layer_reports=layer_reports
            )
        return budget

    raise ValueError(f"Unsupported MLP module type {type(mlp_mod).__name__} for Wanda/OBS surgery")


def _resolve_surgeon_config(
    config: WandaOBSSurgeonConfig | None,
    *,
    prune_ratio: float | None,
    max_edited_param_ratio: float | None,
    budget_scope: str | None,
    prune_scope: str | None,
    target_factors: Sequence[str] | None,
    use_obs_compensation: bool | None,
    use_woodbury: bool | None,
    hessian_damping: float | None,
    protect_general_expert: bool | None,
    domain_guard_max_ratio: float | None,
    min_specialist_gain_retention: float | None,
    require_energy_improvement: bool | None,
    score_from: int | None,
) -> WandaOBSSurgeonConfig:
    """Merge optional keyword overrides into a validated ``WandaOBSSurgeonConfig``."""
    if config is not None and not isinstance(config, WandaOBSSurgeonConfig):
        raise TypeError(
            f"config must be WandaOBSSurgeonConfig or None, got {type(config).__name__}"
        )
    base = config or WandaOBSSurgeonConfig()
    factors = tuple(str(f) for f in target_factors) if target_factors is not None else base.target_factors
    return WandaOBSSurgeonConfig(
        prune_ratio=base.prune_ratio if prune_ratio is None else prune_ratio,
        max_edited_param_ratio=(
            base.max_edited_param_ratio if max_edited_param_ratio is None else max_edited_param_ratio
        ),
        budget_scope=base.budget_scope if budget_scope is None else budget_scope,
        prune_scope=base.prune_scope if prune_scope is None else prune_scope,
        eps=base.eps,
        hessian_damping=base.hessian_damping if hessian_damping is None else hessian_damping,
        use_obs_compensation=(
            base.use_obs_compensation if use_obs_compensation is None else use_obs_compensation
        ),
        use_woodbury=base.use_woodbury if use_woodbury is None else use_woodbury,
        normalize_saliency_by_samples=base.normalize_saliency_by_samples,
        protect_general_expert=(
            base.protect_general_expert if protect_general_expert is None else protect_general_expert
        ),
        general_expert_index=base.general_expert_index,
        target_factors=factors,
        saliency_threshold=base.saliency_threshold,
        max_comp_scale=base.max_comp_scale,
        domain_guard_max_ratio=(
            base.domain_guard_max_ratio if domain_guard_max_ratio is None else domain_guard_max_ratio
        ),
        anse_safe_e0_ratio=base.anse_safe_e0_ratio,
        maximum_pain_energy=base.maximum_pain_energy,
        soft_margin_penalty=base.soft_margin_penalty,
        min_specialist_gain_retention=(
            base.min_specialist_gain_retention
            if min_specialist_gain_retention is None
            else min_specialist_gain_retention
        ),
        require_energy_improvement=(
            base.require_energy_improvement
            if require_energy_improvement is None
            else require_energy_improvement
        ),
        general_prefix=base.general_prefix,
        score_from=base.score_from if score_from is None else score_from,
        batch_size=base.batch_size,
        inplace=base.inplace,
    )


@torch.no_grad()
def execute_wanda_obs_moe_surgery(
    model: nn.Module,
    guard_chunks: Sequence[Mapping[str, Any]] | None = None,
    *,
    config: WandaOBSSurgeonConfig | None = None,
    eval_fn: Callable[[nn.Module], Mapping[str, Any] | tuple[float, Mapping[str, float]]] | None = None,
    prose_activations: torch.Tensor | None = None,
    spec_activations: torch.Tensor | None = None,
    baseline_domain_nll: Mapping[str, float] | None = None,
    tuned_domain_nll: Mapping[str, float] | None = None,
    layer_indices: Sequence[int] | None = None,
    expert_indices: Sequence[int] | None = None,
    target_factors: Sequence[str] | None = None,
    prune_ratio: float | None = None,
    max_edited_param_ratio: float | None = None,
    budget_scope: str | None = None,
    prune_scope: str | None = None,
    use_obs_compensation: bool | None = None,
    use_woodbury: bool | None = None,
    hessian_damping: float | None = None,
    protect_general_expert: bool | None = None,
    domain_guard_max_ratio: float | None = None,
    min_specialist_gain_retention: float | None = None,
    require_energy_improvement: bool | None = None,
    score_from: int | None = None,
) -> dict[str, Any]:
    """Run discriminative Wanda/OBS Sparse MoE surgery with ANSE gating and atomic rollback.

    Supports full ``CrowdV1WithEngram`` models with ``guard_chunks`` (measuring sequential
    per-layer activations and exact causal mid-row NLL across domains) as well as standalone
    MoE/MLP layers with explicit ``(prose_activations, spec_activations, eval_fn)``.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError("eval_fn must be callable or None")

    cfg = _resolve_surgeon_config(
        config,
        prune_ratio=prune_ratio,
        max_edited_param_ratio=max_edited_param_ratio,
        budget_scope=budget_scope,
        prune_scope=prune_scope,
        target_factors=target_factors,
        use_obs_compensation=use_obs_compensation,
        use_woodbury=use_woodbury,
        hessian_damping=hessian_damping,
        protect_general_expert=protect_general_expert,
        domain_guard_max_ratio=domain_guard_max_ratio,
        min_specialist_gain_retention=min_specialist_gain_retention,
        require_energy_improvement=require_energy_improvement,
        score_from=score_from,
    )

    work_model = model if cfg.inplace else copy.deepcopy(model)
    target_factor_set = {str(f) for f in cfg.target_factors}
    target_expert_set = (
        {_validate_nonneg_int(idx, "expert_index") for idx in expert_indices}
        if expert_indices is not None
        else None
    )

    blocks = getattr(work_model, "blocks", None)
    if guard_chunks is not None:
        _validate_crowd_model_compatible(work_model)
        ids_guard, expert_ids = _validate_guard_chunks(
            guard_chunks, general_prefix=cfg.general_prefix, score_from=cfg.score_from
        )
        n_blocks = len(blocks)
        if layer_indices is None:
            target_layer_set: set[int] | None = set(range(n_blocks))
        else:
            t_layers = sorted({_validate_nonneg_int(idx, "layer_index") for idx in layer_indices})
            if not t_layers or any(idx >= n_blocks for idx in t_layers):
                raise ValueError(f"layer_indices must be a non-empty subset of [0, {n_blocks})")
            target_layer_set = set(t_layers)
    else:
        if prose_activations is None or spec_activations is None or eval_fn is None:
            raise ValueError(
                "Either guard_chunks or (prose_activations, spec_activations, eval_fn) must be provided"
            )
        target_layer_set = None

    model_params, trainable_params, target_params = _collect_trainable_and_target_counts(
        work_model,
        target_layer_set=target_layer_set,
        target_expert_set=target_expert_set,
        target_factor_set=target_factor_set,
        cfg=cfg,
    )
    budget_denom = {
        "target": target_params,
        "trainable": trainable_params,
        "model": model_params,
    }[cfg.budget_scope]
    max_params_cap = max(0, int(math.floor(cfg.max_edited_param_ratio * float(budget_denom) + 1e-12)))
    if float(max_params_cap) / float(budget_denom) > cfg.max_edited_param_ratio + 1e-12:
        max_params_cap = max(0, max_params_cap - 1)

    base_nll_clean = _parse_eval_domain_nll(baseline_domain_nll) if baseline_domain_nll is not None else None
    tuned_nll_clean = _parse_eval_domain_nll(tuned_domain_nll) if tuned_domain_nll is not None else None

    was_training = work_model.training
    work_model.eval()
    try:
        parent_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in work_model.state_dict().items()}
        parent_sha = compute_state_sha256(parent_snapshot)

        if eval_fn is not None:
            pre_nll = _parse_eval_domain_nll(eval_fn(work_model))
        else:
            assert guard_chunks is not None
            pre_nll = _eval_model_guard_nlls(
                work_model, ids_guard, expert_ids, score_from=cfg.score_from, batch_size=cfg.batch_size
            )

        ref_nll = base_nll_clean if base_nll_clean is not None else pre_nll
        if set(ref_nll.keys()) != set(pre_nll.keys()):
            raise ValueError("baseline_domain_nll and model domain NLLs must share identical keys")
        if tuned_nll_clean is not None and set(tuned_nll_clean.keys()) != set(pre_nll.keys()):
            raise ValueError("tuned_domain_nll and model domain NLLs must share identical keys")

        parent_eval = _summarize_anse_metrics(
            ref_nll, pre_nll, pre_nll=pre_nll, tuned_nll=tuned_nll_clean, cfg=cfg
        )

        layer_reports: list[dict[str, Any]] = []
        remaining_budget = max_params_cap

        if guard_chunks is not None:
            assert target_layer_set is not None
            dev = work_model.wte.weight.device
            cos = work_model.cos_cached.to(dev)
            sin = work_model.sin_cached.to(dev)
            start_pos = max(0, cfg.score_from - 1)
            prose_mask = torch.tensor(
                [eid.startswith(cfg.general_prefix) for eid in expert_ids], dtype=torch.bool
            )
            ids_p = ids_guard[prose_mask]
            ids_s = ids_guard[~prose_mask]

            x_p = torch.cat(
                [
                    compute_embeddings_batch(work_model, ids_p[b0 : b0 + cfg.batch_size].to(dev))
                    for b0 in range(0, ids_p.shape[0], cfg.batch_size)
                ],
                dim=0,
            )
            x_s = torch.cat(
                [
                    compute_embeddings_batch(work_model, ids_s[b0 : b0 + cfg.batch_size].to(dev))
                    for b0 in range(0, ids_s.shape[0], cfg.batch_size)
                ],
                dim=0,
            )

            for l_idx, blk in enumerate(work_model.blocks):
                xm_p, mi_p = forward_block_pre_mlp(blk, x_p, cos, sin)
                xm_s, mi_s = forward_block_pre_mlp(blk, x_s, cos, sin)
                if l_idx in target_layer_set and remaining_budget > 0:
                    flat_p = mi_p[:, start_pos:-1, :].reshape(-1, int(mi_p.shape[-1])).contiguous()
                    flat_s = mi_s[:, start_pos:-1, :].reshape(-1, int(mi_s.shape[-1])).contiguous()
                    mlp_mod = getattr(blk.mlp, "moe_mlp", blk.mlp)
                    remaining_budget = _run_layer_wanda_obs_surgery(
                        f"layer_{l_idx}",
                        mlp_mod,
                        flat_p,
                        flat_s,
                        target_expert_set=target_expert_set,
                        target_factor_set=target_factor_set,
                        remaining_budget=remaining_budget,
                        cfg=cfg,
                        layer_reports=layer_reports,
                    )
                blk_dt = blk.ln1.weight.dtype
                x_p = (xm_p + blk.mlp(mi_p.to(dtype=blk_dt)).to(torch.float32)).to(torch.float32)
                x_s = (xm_s + blk.mlp(mi_s.to(dtype=blk_dt)).to(torch.float32)).to(torch.float32)
        else:
            xp_2d = _validate_activations_2d(prose_activations, "prose_activations").to(torch.float32)
            xs_2d = _validate_activations_2d(
                spec_activations, "spec_activations", expected_dim=int(xp_2d.shape[1])
            ).to(torch.float32)
            mlp_mod = getattr(work_model, "moe_mlp", work_model)
            remaining_budget = _run_layer_wanda_obs_surgery(
                "layer_0",
                mlp_mod,
                xp_2d,
                xs_2d,
                target_expert_set=target_expert_set,
                target_factor_set=target_factor_set,
                remaining_budget=remaining_budget,
                cfg=cfg,
                layer_reports=layer_reports,
            )

        candidate_sha = compute_state_sha256(work_model.state_dict())
        edited_elements = sum(
            int(torch.ne(work_model.state_dict()[k].detach().cpu(), parent_snapshot[k]).sum().item())
            for k in parent_snapshot
        )
        pruned_elements = sum(
            int(
                (
                    (parent_snapshot[k] != 0.0)
                    & (work_model.state_dict()[k].detach().cpu() == 0.0)
                )
                .sum()
                .item()
            )
            for k in parent_snapshot
        )
        edited_param_ratio = float(edited_elements) / float(budget_denom)
        pruned_param_ratio = float(pruned_elements) / float(budget_denom)
        budget_respected = edited_elements <= max_params_cap and edited_param_ratio <= cfg.max_edited_param_ratio + 1e-12

        eval_error: str | None = None
        try:
            if eval_fn is not None:
                cand_nll = _parse_eval_domain_nll(eval_fn(work_model))
            else:
                assert guard_chunks is not None
                cand_nll = _eval_model_guard_nlls(
                    work_model, ids_guard, expert_ids, score_from=cfg.score_from, batch_size=cfg.batch_size
                )
            cand_eval = _summarize_anse_metrics(
                ref_nll, cand_nll, pre_nll=pre_nll, tuned_nll=tuned_nll_clean, cfg=cfg
            )
        except Exception as exc:
            eval_error = f"{type(exc).__name__}: {exc}"
            cand_eval = {
                "nll_mean": None,
                "base_nll_mean": float(parent_eval["base_nll_mean"]),
                "mean_diff_vs_base": None,
                "worst_domain_id": "",
                "worst_domain_ratio": None,
                "worst_domain_pct_vs_base": None,
                "general_ratio": None,
                "general_pct_vs_base": None,
                "specialist_gain_nats": None,
                "specialist_gain_retained_frac": None,
                "specialist_drift_pct_vs_pre": None,
                "guard_ok": False,
                "retention_ok": False,
                "anse_energy": float(cfg.maximum_pain_energy),
                "per_domain": {},
            }

        parent_e = float(parent_eval["anse_energy"])
        cand_e = float(cand_eval["anse_energy"])
        delta_e = cand_e - parent_e
        energy_ok = (delta_e < 0.0) if cfg.require_energy_improvement else (delta_e <= 0.0)

        if candidate_sha == parent_sha:
            status = "FIXED_POINT_UNCHANGED"
            accepted = False
            rolled_back = False
        elif (
            eval_error is None
            and bool(cand_eval["guard_ok"])
            and bool(cand_eval["retention_ok"])
            and budget_respected
            and energy_ok
        ):
            status = "ACCEPTED"
            accepted = True
            rolled_back = False
        else:
            if eval_error is not None:
                status = "ROLLED_BACK_EVAL_FAILURE"
            elif not bool(cand_eval["guard_ok"]):
                status = "ROLLED_BACK_GUARD_VIOLATION"
            elif not bool(cand_eval["retention_ok"]):
                status = "ROLLED_BACK_RETENTION_VIOLATION"
            else:
                status = "ROLLED_BACK_ENERGY_NON_IMPROVING"
            accepted = False
            rolled_back = True
            cur_sd = work_model.state_dict()
            work_model.load_state_dict(
                {k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in parent_snapshot.items()}
            )

        final_sha = compute_state_sha256(work_model.state_dict())
        if rolled_back and final_sha != parent_sha:
            raise RuntimeError(
                f"Atomic bitwise rollback verification failed: {final_sha[:12]} != {parent_sha[:12]}"
            )
    finally:
        work_model.train(was_training)

    sal_vals = [float(r["saliency_ratio"]) for r in layer_reports if "saliency_ratio" in r]
    mean_saliency_ratio = float(sum(sal_vals) / float(len(sal_vals))) if sal_vals else 0.0

    receipt: dict[str, Any] = {
        "object": WANDA_OBS_RECEIPT_OBJECT,
        "schema": WANDA_OBS_SCHEMA,
        "status": status,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "eval_error": eval_error,
        "guard_ok": bool(cand_eval["guard_ok"]),
        "retention_ok": bool(cand_eval["retention_ok"]),
        "budget_respected": budget_respected,
        "nll_mean": float(cand_eval["nll_mean"]) if accepted else float(parent_eval["nll_mean"]),
        "candidate_nll_mean": cand_eval["nll_mean"],
        "base_nll_mean": float(parent_eval["base_nll_mean"]),
        "saliency_ratio": mean_saliency_ratio,
        "anse_energy": cand_e,
        "parent_anse_energy": parent_e,
        "final_anse_energy": cand_e if accepted else parent_e,
        "delta_e": delta_e,
        "worst_domain_id": str(cand_eval["worst_domain_id"]),
        "worst_domain_ratio": cand_eval["worst_domain_ratio"],
        "worst_domain_pct_vs_base": cand_eval["worst_domain_pct_vs_base"],
        "general_ratio": cand_eval["general_ratio"],
        "general_pct_vs_base": cand_eval["general_pct_vs_base"],
        "specialist_gain_nats": cand_eval["specialist_gain_nats"],
        "specialist_gain_retained_frac": cand_eval["specialist_gain_retained_frac"],
        "specialist_drift_pct_vs_pre": cand_eval["specialist_drift_pct_vs_pre"],
        "edited_elements": edited_elements,
        "pruned_elements": pruned_elements,
        "budget_denominator": budget_denom,
        "max_edited_params_cap": max_params_cap,
        "edited_param_ratio": edited_param_ratio,
        "pruned_param_ratio": pruned_param_ratio,
        "model_edited_param_ratio": float(edited_elements) / float(max(1, model_params)),
        "per_domain": cand_eval["per_domain"],
        "layer_reports": layer_reports,
        "parent_state_sha256": parent_sha,
        "candidate_state_sha256": candidate_sha,
        "final_state_sha256": final_sha,
        "config": asdict(cfg),
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


__all__ = [
    "DEFAULT_ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_ANSE_SOFT_MARGIN_PENALTY",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_EPS",
    "DEFAULT_HESSIAN_DAMPING",
    "DEFAULT_MAX_EDITED_PARAM_RATIO",
    "DEFAULT_PRUNE_RATIO",
    "DEFAULT_RIDGE_FLOOR",
    "DEFAULT_SCORE_FROM",
    "GENERAL_EXPERT_PREFIX",
    "VALID_BUDGET_SCOPES",
    "VALID_HESSIAN_MODES",
    "VALID_PRUNE_SCOPES",
    "VALID_TARGET_FACTORS",
    "WANDA_OBS_RECEIPT_OBJECT",
    "WANDA_OBS_SCHEMA",
    "WandaOBSSurgeonConfig",
    "apply_obs_compensated_row_surgery",
    "canonical_json_bytes",
    "compute_discriminative_wanda_saliency",
    "compute_payload_sha256",
    "compute_specialist_hessian_inverse",
    "compute_state_sha256",
    "execute_wanda_obs_moe_surgery",
    "verify_wanda_obs_receipt_provenance",
]

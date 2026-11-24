# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Training-free Weight-Space Representation Engineering (LAT / Circuit Breakers) for Sparse MoEs.

Implements gradient-free Linear Artificial Tomography (LAT) direction extraction,
rank-preserving SwiGLU/output-projection orthogonalization, and thermodynamic ANSE
gating with atomic bitwise rollback for open-weight Sparse MoE models
(``CrowdV1WithEngram``, ``CrowdLoRAMoEMLP``, ``CrowdFineGrainedMoEMLP``, and
``CrowdRuntimeMLP``):

1. ``extract_lat_control_direction(pos_acts, neg_acts, *, protect_acts=None, shrink_protect=True)``:
   Computes paired-difference LAT principal directions projected strictly into the
   null-space of protected general and specialist activations (with optional
   Oracle Approximating Shrinkage when protected covariance is full-rank).
2. ``apply_weight_space_circuit_breaker_inplace(out_proj_or_lora_b, hijack_dir, false_refusal_dir, ...)``:
   Surgically orthogonalizes dense output projection columns (``attn.o``, ``mlp.w3``)
   or SwiGLU LoRA factors (``w2_B``, ``w3_B``) against hijack and false-refusal
   directions without materializing dense ``(d_out, d_in)`` factor products, under
   a strict Frobenius norm-delta budget (``max_norm_delta_ratio``).
3. ``execute_verified_circuit_breaker(...)``:
   Orchestrates sequential layer-wise circuit breaking and false-refusal abliteration,
   gating commit on ANSE ``<= +1.0%`` worst-domain general/specialist NLL guard
   compliance and strict energy improvement, with atomic bitwise rollback and
   canonical SHA-256 provenance receipts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
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
from hivemind.swarm.marchenko_pastur_covariance_guard import shrink_moment_oracle_oas
from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    forward_block_pre_mlp,
    per_chunk_nll_from_pre_ln,
)

CIRCUIT_BREAKER_SCHEMA: str = "hivemind.representation_circuit_breaker/1"
CIRCUIT_BREAKER_RECEIPT_OBJECT: str = "hivemind.representation_circuit_breaker_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_ANSE_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_HIJACK_DAMP: float = 1.0
DEFAULT_FALSE_REFUSAL_DAMP: float = 0.85
DEFAULT_MAX_NORM_DELTA_RATIO: float = 0.05
DEFAULT_NULLSPACE_EIGEN_RATIO: float = 1e-6
DEFAULT_PROTECT_REG: float = 0.05
DEFAULT_MIN_LAT_SNR: float = 1.0
DEFAULT_TRIGGER_ENERGY_WEIGHT: float = 0.10
DEFAULT_SCORE_FROM: int = 0
VALID_TARGET_PROJECTIONS: frozenset[str] = frozenset(
    {"attn.o", "expert.w2_B", "expert.w3_B", "expert.w3", "mlp.w3"}
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_finite_float(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _validate_nonneg_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite non-negative real number."""
    out = _validate_finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0.0, got {value!r}")
    return out


def _validate_positive_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a strictly positive finite real number."""
    out = _validate_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_unit_interval(value: Any, name: str) -> float:
    """Validate that ``value`` lies in ``[0.0, 1.0]``."""
    out = _validate_nonneg_finite(value, name)
    if out > 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")
    return out


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


def _validate_hidden_states(tensor: Any, name: str) -> torch.Tensor:
    """Validate a non-empty 2D or 3D finite tensor and return CPU ``float64`` 2D."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim not in (2, 3) or any(dim == 0 for dim in tensor.shape):
        raise ValueError(
            f"{name} must be a non-empty 2D or 3D tensor, got shape {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    flat = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1, int(tensor.shape[-1]))
    if not bool(torch.isfinite(flat).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return flat.contiguous()


def _normalize_protect_acts(
    protect_acts: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor] | None,
    expected_dim: int,
) -> torch.Tensor | None:
    """Normalize optional protected activation tensors into a single 2D CPU ``float64`` matrix."""
    if protect_acts is None:
        return None
    if isinstance(protect_acts, torch.Tensor):
        p_2d = _validate_hidden_states(protect_acts, "protect_acts")
        if int(p_2d.shape[1]) != expected_dim:
            raise ValueError(
                f"protect_acts width {p_2d.shape[1]} must match activation width {expected_dim}"
            )
        return p_2d

    if isinstance(protect_acts, Mapping):
        if not protect_acts:
            raise ValueError("protect_acts mapping must not be empty")
        items = [(f"protect_acts[{k!r}]", v) for k, v in sorted(protect_acts.items())]
    elif isinstance(protect_acts, Sequence) and not isinstance(protect_acts, (str, bytes, bytearray)):
        if not protect_acts:
            raise ValueError("protect_acts sequence must not be empty")
        items = [(f"protect_acts[{idx}]", v) for idx, v in enumerate(protect_acts)]
    else:
        raise TypeError(
            f"protect_acts must be a Tensor, Sequence[Tensor], Mapping[str, Tensor], or None; "
            f"got {type(protect_acts).__name__}"
        )

    parts: list[torch.Tensor] = []
    for label, tensor in items:
        mat = _validate_hidden_states(tensor, label)
        if int(mat.shape[1]) != expected_dim:
            raise ValueError(
                f"{label} width {mat.shape[1]} must match activation width {expected_dim}"
            )
        parts.append(mat)
    return torch.cat(parts, dim=0).contiguous()


def _validate_direction_1d(tensor: Any, expected_dim: int | None, name: str) -> torch.Tensor:
    """Validate a 1D non-empty finite floating tensor and return CPU ``float64`` copy."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"{name} must be a non-empty 1D tensor, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    if expected_dim is not None and int(tensor.shape[0]) != expected_dim:
        raise ValueError(
            f"{name} length {tensor.shape[0]} does not match expected dimension {expected_dim}"
        )
    vec = tensor.detach().to(device="cpu", dtype=torch.float64).clone()
    if not bool(torch.isfinite(vec).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return vec


def _resolve_output_weight_tensor(
    target: nn.Module | torch.Tensor,
    name: str,
    *,
    weight_attr: str | None = None,
    expected_out_dim: int | None = None,
) -> tuple[torch.Tensor, str]:
    """Extract a 2D floating output projection or LoRA-B weight tensor in-place reference."""
    if isinstance(target, torch.Tensor):
        w_tensor = target
        resolved = "tensor"
    elif isinstance(target, nn.Module):
        if weight_attr is not None:
            if not isinstance(weight_attr, str) or not weight_attr.strip():
                raise ValueError("weight_attr must be a non-empty string when provided")
            attr_val = getattr(target, weight_attr, None)
            if isinstance(attr_val, nn.Linear):
                attr_val = attr_val.weight
            if not isinstance(attr_val, torch.Tensor):
                raise ValueError(
                    f"{type(target).__name__} does not expose tensor/Linear attribute {weight_attr!r}"
                )
            w_tensor = attr_val
            resolved = weight_attr
        elif isinstance(getattr(target, "weight", None), torch.Tensor):
            w_tensor = target.weight  # type: ignore[assignment]
            resolved = "weight"
        else:
            candidates = [
                (attr, getattr(target, attr))
                for attr in ("w2_B", "w3_B")
                if isinstance(getattr(target, attr, None), torch.Tensor)
            ]
            if not candidates:
                raise ValueError(
                    f"{type(target).__name__} must expose 'weight', 'w2_B', or 'w3_B'"
                )
            matched = [
                (attr, ten)
                for attr, ten in candidates
                if expected_out_dim is None or int(ten.shape[0]) == expected_out_dim
            ]
            if not matched:
                shapes = {attr: tuple(ten.shape) for attr, ten in candidates}
                raise ValueError(
                    f"No LoRA-B factor on {type(target).__name__} matches output dim {expected_out_dim}: {shapes}"
                )
            resolved, w_tensor = matched[0]
    else:
        raise TypeError(f"{name} must be an nn.Module or torch.Tensor, got {type(target).__name__}")

    if w_tensor.ndim != 2 or w_tensor.shape[0] == 0 or w_tensor.shape[1] == 0:
        raise ValueError(f"{name} weight must be non-empty 2D, got shape {tuple(w_tensor.shape)}")
    if not w_tensor.is_floating_point():
        raise TypeError(f"{name} weight must be floating-point, got {w_tensor.dtype}")
    if not bool(torch.isfinite(w_tensor).all().item()):
        raise ValueError(f"{name} weight must contain only finite values")
    return w_tensor, resolved


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically while rejecting non-finite numbers."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def compute_payload_sha256(payload: Any) -> str:
    """Compute lowercase SHA-256 hex digest over ``canonical_json_bytes(payload)``."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Compute deterministic bitwise SHA-256 digest over a tensor mapping."""
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
        digest.update(f"{key}:{cpu_t.dtype}:{tuple(cpu_t.shape)}:".encode())
        digest.update(cpu_t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_circuit_breaker_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify ``receipt_sha256`` and finite schema invariants on a receipt."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != CIRCUIT_BREAKER_RECEIPT_OBJECT:
        return False
    if receipt.get("schema") != CIRCUIT_BREAKER_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Validated hyperparameters for LAT circuit breaking and ANSE commit gating."""

    hijack_damp: float = DEFAULT_HIJACK_DAMP
    false_refusal_damp: float = DEFAULT_FALSE_REFUSAL_DAMP
    max_norm_delta_ratio: float = DEFAULT_MAX_NORM_DELTA_RATIO
    shrink_protect: bool = True
    nullspace_eigen_ratio: float = DEFAULT_NULLSPACE_EIGEN_RATIO
    reg_protect: float = DEFAULT_PROTECT_REG
    min_lat_snr: float = DEFAULT_MIN_LAT_SNR
    domain_guard_max_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = DEFAULT_ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY
    trigger_energy_weight: float = DEFAULT_TRIGGER_ENERGY_WEIGHT
    general_prefix: str = GENERAL_EXPERT_PREFIX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = 16
    center_paired_diffs: bool = False
    require_strict_energy_drop: bool = True

    def __post_init__(self) -> None:
        _validate_unit_interval(self.hijack_damp, "hijack_damp")
        _validate_unit_interval(self.false_refusal_damp, "false_refusal_damp")
        norm_cap = _validate_positive_finite(self.max_norm_delta_ratio, "max_norm_delta_ratio")
        if norm_cap > 1.0:
            raise ValueError(f"max_norm_delta_ratio must be <= 1.0, got {self.max_norm_delta_ratio!r}")
        _validate_bool(self.shrink_protect, "shrink_protect")
        null_r = _validate_nonneg_finite(self.nullspace_eigen_ratio, "nullspace_eigen_ratio")
        if null_r >= 1.0:
            raise ValueError(f"nullspace_eigen_ratio must be < 1.0, got {self.nullspace_eigen_ratio!r}")
        _validate_nonneg_finite(self.reg_protect, "reg_protect")
        _validate_nonneg_finite(self.min_lat_snr, "min_lat_snr")
        guard_max = _validate_positive_finite(self.domain_guard_max_ratio, "domain_guard_max_ratio")
        if guard_max < 1.0:
            raise ValueError(f"domain_guard_max_ratio must be >= 1.0, got {guard_max}")
        safe_e0 = _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        if safe_e0 < 1.0:
            raise ValueError(f"anse_safe_general_ratio must be >= 1.0, got {safe_e0}")
        _validate_nonneg_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_nonneg_finite(self.trigger_energy_weight, "trigger_energy_weight")
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_bool(self.center_paired_diffs, "center_paired_diffs")
        _validate_bool(self.require_strict_energy_drop, "require_strict_energy_drop")


# ---------------------------------------------------------------------------
# 1. Linear Artificial Tomography (LAT) with Protected Null-Space Projection
# ---------------------------------------------------------------------------


def _align_direction_sign(vec: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Orient unit vector ``vec`` toward ``ref`` (or canonical max-abs sign if orthogonal)."""
    if float(ref.norm().item()) > 1e-12:
        return -vec if float((vec @ ref).item()) < 0.0 else vec
    max_idx = int(torch.argmax(vec.abs()).item())
    return -vec if float(vec[max_idx].item()) < 0.0 else vec


def _compute_pair_contrast_snr(pos_x: torch.Tensor, neg_x: torch.Tensor, direction: torch.Tensor) -> float:
    """Compute contrast SNR along ``direction`` between ``pos_x`` and ``neg_x``."""
    if float(direction.norm().item()) <= 1e-12:
        return 0.0
    zp = pos_x @ direction
    zn = neg_x @ direction
    if pos_x.shape[0] == neg_x.shape[0]:
        dz = zp - zn
        mean_c = abs(float(dz.mean().item()))
        std_c = math.sqrt(float(dz.var(unbiased=False).item()) + 1e-10)
        return mean_c / std_c
    mean_c = abs(float(zp.mean().item()) - float(zn.mean().item()))
    v_p = float(zp.var(unbiased=False).item()) if zp.numel() > 1 else 0.0
    v_n = float(zn.var(unbiased=False).item()) if zn.numel() > 1 else 0.0
    return mean_c / math.sqrt(0.5 * (v_p + v_n) + 1e-10)


def _prepare_protect_projector(
    protect_2d: torch.Tensor | None,
    *,
    dim: int,
    shrink_protect: bool,
    nullspace_eigen_ratio: float,
    reg_protect: float,
) -> dict[str, Any]:
    """Build exact protected null-space basis or OAS-regularized inverse whitener."""
    if protect_2d is None:
        return {
            "mode": "unconstrained",
            "v_null": None,
            "b_inv_half": None,
            "nullspace_dim": dim,
            "used_exact_nullspace": True,
            "rho_protect": 0.0,
            "condition_number_emp": 1.0,
            "condition_number_shrunk": 1.0,
        }

    n_prot = int(protect_2d.shape[0])
    m_emp = (protect_2d.T @ protect_2d) / float(n_prot)
    m_emp = 0.5 * (m_emp + m_emp.T)
    evals_emp, evecs_emp = torch.linalg.eigh(m_emp)
    lam_max_emp = max(float(evals_emp[-1].item()), 0.0)
    cond_emp = (
        float(lam_max_emp / max(float(evals_emp[0].clamp_min(0.0).item()), 1e-12))
        if lam_max_emp > 1e-15
        else 1.0
    )

    if shrink_protect:
        m_shrunk, shr_diag = shrink_moment_oracle_oas(m_emp, n_prot)
        ev_shr = torch.linalg.eigvalsh(m_shrunk)
        lam_max_shr = max(float(ev_shr[-1].item()), 0.0)
        cond_shr = (
            float(lam_max_shr / max(float(ev_shr[0].clamp_min(0.0).item()), 1e-12))
            if lam_max_shr > 1e-15
            else 1.0
        )
        rho_prot = float(shr_diag["rho"])
    else:
        m_shrunk = m_emp
        cond_shr = cond_emp
        rho_prot = 0.0

    if lam_max_emp <= 1e-24:
        return {
            "mode": "unconstrained",
            "v_null": None,
            "b_inv_half": None,
            "nullspace_dim": dim,
            "used_exact_nullspace": True,
            "rho_protect": rho_prot,
            "condition_number_emp": cond_emp,
            "condition_number_shrunk": cond_shr,
        }

    null_mask = evals_emp <= (nullspace_eigen_ratio * lam_max_emp)
    n_null = int(null_mask.sum().item())
    if n_null > 0:
        return {
            "mode": "exact_nullspace",
            "v_null": evecs_emp[:, null_mask].contiguous(),
            "b_inv_half": None,
            "nullspace_dim": n_null,
            "used_exact_nullspace": True,
            "rho_protect": rho_prot,
            "condition_number_emp": cond_emp,
            "condition_number_shrunk": cond_shr,
        }

    m_base = m_shrunk if shrink_protect else m_emp
    tr_p = max(float(torch.diagonal(m_base).mean().item()), 1e-8)
    b_prot = m_base + max(reg_protect * tr_p, 1e-8) * torch.eye(dim, dtype=torch.float64)
    ev_b, q_b = torch.linalg.eigh(0.5 * (b_prot + b_prot.T))
    b_inv_half = (q_b * ev_b.clamp_min(1e-12).rsqrt().unsqueeze(0)) @ q_b.T
    return {
        "mode": "generalized_rayleigh",
        "v_null": None,
        "b_inv_half": b_inv_half.contiguous(),
        "nullspace_dim": 0,
        "used_exact_nullspace": False,
        "rho_protect": rho_prot,
        "condition_number_emp": cond_emp,
        "condition_number_shrunk": cond_shr,
    }


def _solve_projected_lat_direction(
    pos_x: torch.Tensor,
    neg_x: torch.Tensor,
    proj_ctx: Mapping[str, Any],
    *,
    center_paired_diffs: bool,
) -> tuple[torch.Tensor, float, bool]:
    """Solve top LAT paired-difference direction under ``proj_ctx`` constraint."""
    dim = int(pos_x.shape[1])
    if pos_x.shape[0] == neg_x.shape[0]:
        diffs = pos_x - neg_x
    else:
        diffs = torch.cat(
            [pos_x - neg_x.mean(dim=0, keepdim=True), pos_x.mean(dim=0, keepdim=True) - neg_x],
            dim=0,
        )
    mu_diff = pos_x.mean(dim=0) - neg_x.mean(dim=0)
    diffs_cov = diffs - diffs.mean(dim=0, keepdim=True) if center_paired_diffs else diffs
    m_diff = (diffs_cov.T @ diffs_cov) / float(diffs_cov.shape[0])
    m_diff = 0.5 * (m_diff + m_diff.T)

    mode = proj_ctx["mode"]
    if mode == "exact_nullspace":
        basis: torch.Tensor = proj_ctx["v_null"]
        sub_mat = 0.5 * ((basis.T @ m_diff @ basis) + (basis.T @ m_diff @ basis).T)
        evals, evecs = torch.linalg.eigh(sub_mat)
        tot_var = max(float(evals.clamp_min(0.0).sum().item()), 0.0)
        top_ev = max(float(evals[-1].item()), 0.0)
        if top_ev <= 1e-20:
            mu_null = basis.T @ mu_diff
            mu_n_norm = float(mu_null.norm().item())
            if (not center_paired_diffs) and mu_n_norm > 1e-12:
                return (basis @ (mu_null / mu_n_norm)).contiguous(), 1.0, False
            return torch.zeros(dim, dtype=torch.float64), 0.0, True
        u_raw = basis @ evecs[:, -1]
        u = _align_direction_sign(u_raw / u_raw.norm().clamp_min(1e-12), mu_diff)
        return u.contiguous(), top_ev / max(tot_var, 1e-24), False

    if mode == "unconstrained":
        evals, evecs = torch.linalg.eigh(m_diff)
        tot_var = max(float(evals.clamp_min(0.0).sum().item()), 0.0)
        top_ev = max(float(evals[-1].item()), 0.0)
        if top_ev <= 1e-24:
            return torch.zeros(dim, dtype=torch.float64), 0.0, True
        u = _align_direction_sign(evecs[:, -1], mu_diff)
        return (u / u.norm().clamp_min(1e-12)).contiguous(), top_ev / max(tot_var, 1e-24), False

    b_inv_half: torch.Tensor = proj_ctx["b_inv_half"]
    whitened = 0.5 * ((b_inv_half @ m_diff @ b_inv_half) + (b_inv_half @ m_diff @ b_inv_half).T)
    evals, evecs = torch.linalg.eigh(whitened)
    tot_var = max(float(evals.clamp_min(0.0).sum().item()), 0.0)
    top_ev = max(float(evals[-1].item()), 0.0)
    if top_ev <= 1e-24:
        return torch.zeros(dim, dtype=torch.float64), 0.0, True
    u_raw = b_inv_half @ evecs[:, -1]
    u_norm = float(u_raw.norm().item())
    if u_norm <= 1e-12:
        return torch.zeros(dim, dtype=torch.float64), 0.0, True
    u = _align_direction_sign(u_raw / u_norm, mu_diff)
    return u.contiguous(), top_ev / max(tot_var, 1e-24), False


@torch.no_grad()
def extract_lat_control_direction(
    pos_acts: torch.Tensor,
    neg_acts: torch.Tensor,
    *,
    protect_acts: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor] | None = None,
    shrink_protect: bool = True,
    nullspace_eigen_ratio: float = DEFAULT_NULLSPACE_EIGEN_RATIO,
    reg_protect: float = DEFAULT_PROTECT_REG,
    center_paired_diffs: bool = False,
    min_oof_snr: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Extract LAT paired-difference principal direction projected into protected null-space.

    Computes the dominant Linear Artificial Tomography (LAT) contrast vector between
    ``pos_acts`` and ``neg_acts`` in CPU ``float64``. When ``protect_acts`` is given:
    - If ``protect_acts`` has a non-trivial null space (eigenvalues
      ``<= nullspace_eigen_ratio * lam_max``), projects differences strictly into
      ``ker(protect_acts)`` so protected output drift vanishes identically.
    - If ``protect_acts`` is full rank, solves the generalized Rayleigh problem against
      OAS-shrunk protected covariance (when ``shrink_protect=True``).
    """
    _validate_bool(shrink_protect, "shrink_protect")
    _validate_bool(center_paired_diffs, "center_paired_diffs")
    null_r = _validate_nonneg_finite(nullspace_eigen_ratio, "nullspace_eigen_ratio")
    if null_r >= 1.0:
        raise ValueError(f"nullspace_eigen_ratio must be < 1.0, got {nullspace_eigen_ratio!r}")
    reg_p = _validate_nonneg_finite(reg_protect, "reg_protect")
    min_snr = _validate_nonneg_finite(min_oof_snr, "min_oof_snr")

    pos_2d = _validate_hidden_states(pos_acts, "pos_acts")
    neg_2d = _validate_hidden_states(neg_acts, "neg_acts")
    dim = int(pos_2d.shape[1])
    if int(neg_2d.shape[1]) != dim:
        raise ValueError(
            f"pos_acts width {dim} must match neg_acts width {neg_2d.shape[1]}"
        )
    protect_2d = _normalize_protect_acts(protect_acts, dim)

    proj_ctx = _prepare_protect_projector(
        protect_2d,
        dim=dim,
        shrink_protect=shrink_protect,
        nullspace_eigen_ratio=null_r,
        reg_protect=reg_p,
    )
    u, exp_var_ratio, is_degen = _solve_projected_lat_direction(
        pos_2d,
        neg_2d,
        proj_ctx,
        center_paired_diffs=center_paired_diffs,
    )

    full_snr = _compute_pair_contrast_snr(pos_2d, neg_2d, u)
    if not is_degen and pos_2d.shape[0] >= 4 and neg_2d.shape[0] >= 4:
        u0, _, d0 = _solve_projected_lat_direction(
            pos_2d[0::2], neg_2d[0::2], proj_ctx, center_paired_diffs=center_paired_diffs
        )
        u1, _, d1 = _solve_projected_lat_direction(
            pos_2d[1::2], neg_2d[1::2], proj_ctx, center_paired_diffs=center_paired_diffs
        )
        if d0 or d1:
            oof_snr = 0.0
        else:
            snr_01 = _compute_pair_contrast_snr(pos_2d[1::2], neg_2d[1::2], u0)
            snr_10 = _compute_pair_contrast_snr(pos_2d[0::2], neg_2d[0::2], u1)
            fold_align = abs(float((u0 @ u1).item()))
            oof_snr = max(0.0, min(full_snr, snr_01, snr_10) * fold_align)
    else:
        oof_snr = max(0.0, full_snr)

    gated_by_snr = False
    if min_snr > 0.0 and oof_snr < min_snr:
        u = torch.zeros(dim, dtype=torch.float64)
        is_degen = True
        gated_by_snr = True

    if protect_2d is not None and float(u.norm().item()) > 1e-12:
        prot_rms = float(protect_2d.pow(2).sum(dim=1).mean().sqrt().item())
        proj_rms = float((protect_2d @ u).pow(2).mean().sqrt().item())
        prot_rms_ratio = proj_rms / max(prot_rms, 1e-12)
    else:
        prot_rms_ratio = 0.0

    lat_contrast_mean = float(((pos_2d @ u).mean() - (neg_2d @ u).mean()).item())
    return u, {
        "dim": dim,
        "paired_samples": pos_2d.shape[0] == neg_2d.shape[0],
        "n_pos": int(pos_2d.shape[0]),
        "n_neg": int(neg_2d.shape[0]),
        "n_protect": int(protect_2d.shape[0]) if protect_2d is not None else 0,
        "projection_mode": proj_ctx["mode"],
        "used_exact_nullspace": bool(proj_ctx["used_exact_nullspace"]),
        "nullspace_dim": int(proj_ctx["nullspace_dim"]),
        "shrink_protect": shrink_protect,
        "rho_protect": float(proj_ctx["rho_protect"]),
        "condition_number_emp": float(proj_ctx["condition_number_emp"]),
        "condition_number_shrunk": float(proj_ctx["condition_number_shrunk"]),
        "explained_variance_ratio": float(exp_var_ratio),
        "lat_contrast_mean": lat_contrast_mean,
        "full_snr": float(full_snr),
        "oof_snr": float(oof_snr),
        "protected_rms_ratio": float(prot_rms_ratio),
        "is_degenerate": bool(is_degen),
        "gated_by_oof_snr": gated_by_snr,
    }


# ---------------------------------------------------------------------------
# 2. Rank-Preserving Budgeted Weight-Space Circuit Breaker
# ---------------------------------------------------------------------------


def _build_circuit_breaker_noop(
    resolved_attr: str,
    shape: tuple[int, int],
    norm_before: float,
    norm_budget: float,
    h_damp: float,
    r_damp: float,
    noop_reason: str,
) -> dict[str, Any]:
    """Build standardized report when circuit breaker performs a safety no-op."""
    return {
        "resolved_attr": resolved_attr,
        "weight_shape": shape,
        "norm_before": norm_before,
        "norm_after": norm_before,
        "raw_norm_delta": 0.0,
        "raw_norm_delta_ratio": 0.0,
        "applied_norm_delta": 0.0,
        "applied_norm_delta_ratio": 0.0,
        "max_norm_delta_ratio": norm_budget,
        "budget_clamped": False,
        "budget_scale": 1.0,
        "hijack_damp": h_damp,
        "false_refusal_damp": r_damp,
        "hijack_pre_proj_norm": 0.0,
        "hijack_post_proj_norm": 0.0,
        "hijack_neutralization_ratio": 0.0,
        "false_refusal_pre_proj_norm": 0.0,
        "false_refusal_post_proj_norm": 0.0,
        "false_refusal_neutralization_ratio": 0.0,
        "noop_reason": noop_reason,
    }


@torch.no_grad()
def apply_weight_space_circuit_breaker_inplace(
    out_proj_or_lora_b: nn.Module | torch.Tensor,
    hijack_dir: torch.Tensor | None,
    false_refusal_dir: torch.Tensor | None,
    *,
    hijack_damp: float = DEFAULT_HIJACK_DAMP,
    false_refusal_damp: float = DEFAULT_FALSE_REFUSAL_DAMP,
    max_norm_delta_ratio: float = DEFAULT_MAX_NORM_DELTA_RATIO,
    weight_attr: str | None = None,
) -> dict[str, Any]:
    """Orthogonalize output weight columns or SwiGLU ``w2_B`` factors under a strict norm budget.

    Applies rank-preserving left projection ``W <- W - sum_k alpha_k q_k (q_k^T W)``
    where ``(q_1, q_2)`` is the Gram-Schmidt orthonormalization of ``(hijack_dir,
    false_refusal_dir)``, clamped strictly so ``||W_after - W_before||_F / ||W_before||_F
    <= max_norm_delta_ratio``. Never materializes full ``(d_out, d_in)`` dense products
    when operating on LoRA ``w2_B`` / ``w3_B`` factors.
    """
    h_damp = _validate_unit_interval(hijack_damp, "hijack_damp")
    r_damp = _validate_unit_interval(false_refusal_damp, "false_refusal_damp")
    norm_budget = _validate_positive_finite(max_norm_delta_ratio, "max_norm_delta_ratio")
    if norm_budget > 1.0:
        raise ValueError(f"max_norm_delta_ratio must be <= 1.0, got {max_norm_delta_ratio!r}")

    h_vec = _validate_direction_1d(hijack_dir, None, "hijack_dir") if hijack_dir is not None else None
    r_vec = (
        _validate_direction_1d(false_refusal_dir, None, "false_refusal_dir")
        if false_refusal_dir is not None
        else None
    )
    if h_vec is not None and r_vec is not None and int(h_vec.numel()) != int(r_vec.numel()):
        raise ValueError(
            f"hijack_dir length {h_vec.numel()} must match false_refusal_dir length {r_vec.numel()}"
        )
    expected_dim = (
        int(h_vec.numel())
        if h_vec is not None
        else (int(r_vec.numel()) if r_vec is not None else None)
    )

    w_tensor, resolved_attr = _resolve_output_weight_tensor(
        out_proj_or_lora_b,
        "out_proj_or_lora_b",
        weight_attr=weight_attr,
        expected_out_dim=expected_dim,
    )
    d_out, d_in = int(w_tensor.shape[0]), int(w_tensor.shape[1])
    if expected_dim is not None and expected_dim != d_out:
        raise ValueError(
            f"Control direction dimension {expected_dim} must match weight output rows {d_out}"
        )

    w64 = w_tensor.detach().to(device="cpu", dtype=torch.float64)
    norm_before = float(torch.linalg.norm(w64).item())
    if norm_before <= 1e-15:
        return _build_circuit_breaker_noop(
            resolved_attr, (d_out, d_in), norm_before, norm_budget, h_damp, r_damp, "zero_weight_norm"
        )

    q_list: list[torch.Tensor] = []
    alpha_list: list[float] = []
    h_unit: torch.Tensor | None = None
    r_unit: torch.Tensor | None = None

    if h_vec is not None and h_damp > 0.0 and float(h_vec.norm().item()) > 1e-12:
        h_unit = h_vec / h_vec.norm()
        q_list.append(h_unit)
        alpha_list.append(h_damp)

    if r_vec is not None and r_damp > 0.0 and float(r_vec.norm().item()) > 1e-12:
        r_unit = r_vec / r_vec.norm()
        if q_list:
            v_perp = r_unit - float((q_list[0] @ r_unit).item()) * q_list[0]
            perp_n = float(v_perp.norm().item())
            if perp_n > 1e-8:
                q_list.append(v_perp / perp_n)
                alpha_list.append(r_damp)
            else:
                alpha_list[0] = max(alpha_list[0], r_damp)
        else:
            q_list.append(r_unit)
            alpha_list.append(r_damp)

    if not q_list:
        return _build_circuit_breaker_noop(
            resolved_attr, (d_out, d_in), norm_before, norm_budget, h_damp, r_damp, "no_active_directions"
        )

    delta_w = torch.zeros_like(w64)
    for q_k, alpha_k in zip(q_list, alpha_list, strict=True):
        proj_k = q_k @ w64
        delta_w.sub_(alpha_k * torch.outer(q_k, proj_k))

    raw_delta_norm = float(torch.linalg.norm(delta_w).item())
    raw_delta_ratio = raw_delta_norm / norm_before
    tau_budget = norm_budget * norm_before
    if raw_delta_norm > tau_budget and raw_delta_norm > 1e-15:
        budget_scale = tau_budget / raw_delta_norm
        delta_w.mul_(budget_scale)
        budget_clamped = True
    else:
        budget_scale = 1.0
        budget_clamped = False

    w_new = (w64 + delta_w).to(device=w_tensor.device, dtype=w_tensor.dtype)
    if w_tensor.dtype != torch.float64:
        for shrink_factor in (0.999, 0.995, 0.985, 0.97):
            cand_norm = float(
                torch.linalg.norm(w_new.detach().to("cpu", dtype=torch.float64) - w64).item()
            )
            if cand_norm <= tau_budget + 1e-12:
                break
            delta_w.mul_(shrink_factor)
            budget_scale *= shrink_factor
            w_new = (w64 + delta_w).to(device=w_tensor.device, dtype=w_tensor.dtype)
    w_tensor.copy_(w_new)

    w_after64 = w_tensor.detach().to(device="cpu", dtype=torch.float64)
    applied_delta = float(torch.linalg.norm(w_after64 - w64).item())
    applied_ratio = applied_delta / norm_before
    norm_after = float(torch.linalg.norm(w_after64).item())

    def _proj_stats(unit_vec: torch.Tensor | None) -> tuple[float, float, float]:
        if unit_vec is None:
            return 0.0, 0.0, 0.0
        pre = float(torch.linalg.norm(unit_vec @ w64).item())
        post = float(torch.linalg.norm(unit_vec @ w_after64).item())
        ratio = max(0.0, 1.0 - (post / pre)) if pre > 1e-12 else 0.0
        return pre, post, ratio

    h_pre, h_post, h_neut = _proj_stats(h_unit)
    r_pre, r_post, r_neut = _proj_stats(r_unit)

    return {
        "resolved_attr": resolved_attr,
        "weight_shape": (d_out, d_in),
        "norm_before": norm_before,
        "norm_after": norm_after,
        "raw_norm_delta": raw_delta_norm,
        "raw_norm_delta_ratio": raw_delta_ratio,
        "applied_norm_delta": applied_delta,
        "applied_norm_delta_ratio": applied_ratio,
        "max_norm_delta_ratio": norm_budget,
        "budget_clamped": budget_clamped,
        "budget_scale": float(budget_scale),
        "hijack_damp": h_damp,
        "false_refusal_damp": r_damp,
        "hijack_pre_proj_norm": h_pre,
        "hijack_post_proj_norm": h_post,
        "hijack_neutralization_ratio": h_neut,
        "false_refusal_pre_proj_norm": r_pre,
        "false_refusal_post_proj_norm": r_post,
        "false_refusal_neutralization_ratio": r_neut,
        "noop_reason": None if applied_delta > 0.0 else "zero_projection_overlap",
    }


# ---------------------------------------------------------------------------
# 3. End-to-End Verified Circuit Breaker with ANSE Gate & Bitwise Rollback
# ---------------------------------------------------------------------------


def _validate_model_compatible(model: nn.Module) -> None:
    """Verify required CrowdV1-compatible transformer attributes on ``model``."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
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
    if not any(eid.startswith(general_prefix) for eid in expert_ids):
        raise ValueError(
            f"guard_chunks must include at least one general chunk starting with {general_prefix!r}"
        )
    return torch.stack(rows, dim=0), expert_ids


def _normalize_token_id_batch(raw_ids: Any, name: str, *, score_from: int) -> torch.Tensor:
    """Validate 2D integer token IDs for LAT extraction."""
    if isinstance(raw_ids, torch.Tensor):
        if raw_ids.ndim != 2 or raw_ids.shape[0] == 0 or raw_ids.shape[1] < 2:
            raise ValueError(
                f"{name} must be a non-empty 2D tensor with length >= 2, got {tuple(raw_ids.shape)}"
            )
        if raw_ids.is_floating_point() or raw_ids.dtype == torch.bool:
            raise TypeError(f"{name} must have an integer dtype, got {raw_ids.dtype}")
        ids_t = raw_ids.detach().to(dtype=torch.long)
    elif isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes, bytearray)):
        if not raw_ids:
            raise ValueError(f"{name} must not be empty")
        ids_t = torch.tensor(raw_ids, dtype=torch.long)
        if ids_t.ndim != 2 or ids_t.shape[1] < 2:
            raise ValueError(f"{name} must form a 2D batch with length >= 2")
    else:
        raise TypeError(f"{name} must be a 2D integer tensor or sequence of token lists")

    if bool((ids_t < 0).any().item()):
        raise ValueError(f"{name} must contain non-negative token IDs")
    start_pos = max(0, score_from - 1)
    if start_pos >= int(ids_t.shape[1]) - 1:
        raise ValueError(f"{name} length {ids_t.shape[1]} too short for score_from={score_from}")
    return ids_t


def _unwrap_moe(mlp: nn.Module) -> nn.Module:
    """Return underlying MoE/MLP module when wrapped by an inference guard."""
    return getattr(mlp, "moe_mlp", mlp)


@torch.no_grad()
def _forward_attention_only(
    blk: nn.Module,
    x_seq: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Compute attention output tensor ``attn_h = attn(ln1(x))`` in ``float32``."""
    blk_dtype = blk.ln1.weight.dtype
    parts = [
        blk.attn(blk.ln1(x_seq[b0 : b0 + batch_size].to(dtype=blk_dtype)), cos, sin, past_kv=None, use_cache=False)[0].to(
            torch.float32
        )
        for b0 in range(0, x_seq.shape[0], batch_size)
    ]
    return torch.cat(parts, dim=0)


@torch.no_grad()
def _forward_mlp_only(blk: nn.Module, mlp_in: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    """Compute MLP residual output ``mlp(mlp_in)`` in ``float32``."""
    blk_dtype = blk.ln1.weight.dtype
    parts = [
        blk.mlp(mlp_in[b0 : b0 + batch_size].to(dtype=blk_dtype)).to(torch.float32)
        for b0 in range(0, mlp_in.shape[0], batch_size)
    ]
    return torch.cat(parts, dim=0)


@torch.no_grad()
def _forward_layer_norm2(blk: nn.Module, x_mid: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    """Compute ``blk.ln2(x_mid)`` in ``float32``."""
    blk_dtype = blk.ln1.weight.dtype
    parts = [
        blk.ln2(x_mid[b0 : b0 + batch_size].to(dtype=blk_dtype)).to(torch.float32)
        for b0 in range(0, x_mid.shape[0], batch_size)
    ]
    return torch.cat(parts, dim=0)


@torch.no_grad()
def _eval_model_guard_nlls(
    model: nn.Module,
    ids_2d: torch.Tensor,
    *,
    score_from: int,
    batch_size: int,
) -> np.ndarray:
    """Evaluate per-chunk causal NLL across all transformer blocks."""
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
        return np.full(int(ids_2d.shape[0]), float("nan"), dtype=np.float64)
    nlls = per_chunk_nll_from_pre_ln(
        h_pre,
        model.ln.weight,
        model.ln.bias,
        float(getattr(model.ln, "eps", 1e-5)),
        model.head.weight,
        ids_2d.to(dev),
        score_from=score_from,
    )
    return nlls.detach().cpu().numpy().astype(np.float64)


@torch.no_grad()
def _compute_circuit_delta_states(
    model: nn.Module,
    ids_2d: torch.Tensor,
    *,
    score_from: int,
    batch_size: int,
) -> torch.Tensor:
    """Return total transformer-block residual contribution ``(h_pre_ln - h_emb)[:, start_pos:]``."""
    dev = model.wte.weight.device
    cos = model.cos_cached.to(dev)
    sin = model.sin_cached.to(dev)
    start_pos = max(0, score_from - 1)
    parts: list[torch.Tensor] = []
    for b0 in range(0, ids_2d.shape[0], batch_size):
        emb = compute_embeddings_batch(model, ids_2d[b0 : b0 + batch_size].to(dev))
        xb = emb
        for blk in model.blocks:
            xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
            mlp_out = blk.mlp(mi.to(dtype=blk.ln1.weight.dtype)).to(torch.float32)
            xb = (xm + mlp_out).to(torch.float32)
        parts.append((xb - emb)[:, start_pos:, :])
    cat = torch.cat(parts, dim=0)
    return cat.reshape(-1, int(cat.shape[-1])).contiguous()


@torch.no_grad()
def _collect_lora_expert_signals(
    moe_mlp: CrowdLoRAMoEMLP,
    mlp_in_seq: torch.Tensor,
    exp_key: str,
    *,
    start_pos: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute routing-weighted ``dh2`` (in ``R^I``), ``dy_e`` (in ``R^H``), and ``h2_base`` (in ``R^I``)."""
    sliced = mlp_in_seq[:, start_pos:, :]
    flat = sliced.reshape(-1, int(sliced.shape[-1]))
    w_dt = moe_mlp.base_mlp.w1.weight.dtype
    x_base = flat.to(dtype=w_dt)
    h1 = moe_mlp.base_mlp.w1(x_base).to(torch.float32)
    h2 = moe_mlp.base_mlp.w2(x_base).to(torch.float32)
    x_fp32 = flat.to(torch.float32)

    logits = moe_mlp.router(x_fp32)
    if moe_mlp.routing_mode == "sigma_topk":
        probs = torch.sigmoid(logits)
        top_vals, top_idx = torch.topk(probs, k=int(moe_mlp.top_k), dim=-1)
        gates = top_vals
    else:
        probs = F.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(probs, k=int(moe_mlp.top_k), dim=-1)
        gates = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    e_idx = int(exp_key)
    w_tok = (gates * (top_idx == e_idx).to(torch.float32)).sum(dim=-1, keepdim=True)
    exp = moe_mlp.experts[exp_key]
    dh2_raw = F.linear(F.linear(x_fp32, exp.w2_A), exp.w2_B) * exp.scaling
    dy_raw = exp.forward_delta(x_fp32, h1, h2, moe_mlp.base_mlp.w3.weight)
    return (w_tok * dh2_raw).contiguous(), (w_tok * dy_raw).contiguous(), h2.contiguous()


def _extract_pair_direction_gated(
    pos_acts: torch.Tensor | None,
    neg_acts: torch.Tensor | None,
    protect_acts: torch.Tensor | None,
    *,
    override_dir: torch.Tensor | None,
    target_dim: int,
    cfg: CircuitBreakerConfig,
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    """Return validated override or OOF-SNR-gated LAT direction matching ``target_dim``."""
    if override_dir is not None:
        if int(override_dir.numel()) == target_dim:
            return override_dir, {"source": "override", "oof_snr": None}
        return None, None
    if pos_acts is None or neg_acts is None:
        return None, None
    u, info = extract_lat_control_direction(
        pos_acts,
        neg_acts,
        protect_acts=protect_acts,
        shrink_protect=cfg.shrink_protect,
        nullspace_eigen_ratio=cfg.nullspace_eigen_ratio,
        reg_protect=cfg.reg_protect,
        center_paired_diffs=cfg.center_paired_diffs,
        min_oof_snr=cfg.min_lat_snr,
    )
    if info["is_degenerate"] or float(u.norm().item()) <= 1e-12:
        return None, info
    return u, info


def _apply_pair_breaker(
    layer_idx: int,
    target_name: str,
    target_mod_or_tensor: nn.Module | torch.Tensor,
    *,
    h_pos: torch.Tensor | None,
    h_neg: torch.Tensor | None,
    r_pos: torch.Tensor | None,
    r_neg: torch.Tensor | None,
    protect: torch.Tensor | None,
    h_override: torch.Tensor | None,
    r_override: torch.Tensor | None,
    target_dim: int,
    cfg: CircuitBreakerConfig,
    layer_reports: list[dict[str, Any]],
) -> bool:
    """Extract gated LAT directions for one weight target and apply circuit breaker if active."""
    u_h, h_info = _extract_pair_direction_gated(
        h_pos, h_neg, protect, override_dir=h_override, target_dim=target_dim, cfg=cfg
    )
    u_r, r_info = _extract_pair_direction_gated(
        r_pos, r_neg, protect, override_dir=r_override, target_dim=target_dim, cfg=cfg
    )
    if u_h is None and u_r is None:
        return False
    rep = apply_weight_space_circuit_breaker_inplace(
        target_mod_or_tensor,
        u_h,
        u_r,
        hijack_damp=cfg.hijack_damp,
        false_refusal_damp=cfg.false_refusal_damp,
        max_norm_delta_ratio=cfg.max_norm_delta_ratio,
    )
    layer_reports.append(
        {"layer": layer_idx, "target": target_name, "hijack_lat": h_info, "refusal_lat": r_info, **rep}
    )
    return float(rep["applied_norm_delta"]) > 0.0


def _process_lora_moe_layer(
    l_idx: int,
    blk: nn.Module,
    mlp_mod: CrowdLoRAMoEMLP,
    mlp_in: Mapping[str, torch.Tensor | None],
    *,
    start_pos: int,
    proj_set: set[str],
    target_expert_set: set[int] | None,
    h_override: torch.Tensor | None,
    r_override: torch.Tensor | None,
    cfg: CircuitBreakerConfig,
    layer_reports: list[dict[str, Any]],
) -> None:
    """Apply circuit breaker to active SwiGLU ``w2_B`` and ``w3_B`` factors of ``CrowdLoRAMoEMLP``."""
    def _slice_seq(t: torch.Tensor | None) -> torch.Tensor | None:
        if t is None:
            return None
        return t[:, start_pos:, :].reshape(-1, int(t.shape[-1])).contiguous()
    for e_key in sorted(mlp_mod.experts.keys(), key=int):
        if target_expert_set is not None and int(e_key) not in target_expert_set:
            continue
        exp = mlp_mod.experts[e_key]
        has_w2 = float(exp.w2_B.detach().abs().max().item()) > 1e-12
        has_w3 = float(exp.w3_B.detach().abs().max().item()) > 1e-12
        if not has_w2 and not has_w3:
            continue

        sig = {
            k: (_collect_lora_expert_signals(mlp_mod, v, e_key, start_pos=start_pos) if v is not None else None)
            for k, v in mlp_in.items()
        }
        if "expert.w2_B" in proj_set and has_w2:
            prot_w2 = sig["prot"][2] + sig["prot"][0] if sig["prot"] is not None else None
            edited_w2 = _apply_pair_breaker(
                l_idx,
                f"expert_{e_key}.w2_B",
                exp.w2_B,
                h_pos=sig["h_pos"][0] if sig["h_pos"] is not None else None,
                h_neg=sig["h_neg"][0] if sig["h_neg"] is not None else None,
                r_pos=sig["r_pos"][0] if sig["r_pos"] is not None else None,
                r_neg=sig["r_neg"][0] if sig["r_neg"] is not None else None,
                protect=prot_w2,
                h_override=h_override,
                r_override=r_override,
                target_dim=int(exp.w2_B.shape[0]),
                cfg=cfg,
                layer_reports=layer_reports,
            )
            if edited_w2:
                sig = {
                    k: (
                        _collect_lora_expert_signals(mlp_mod, v, e_key, start_pos=start_pos)
                        if v is not None
                        else None
                    )
                    for k, v in mlp_in.items()
                }

        if "expert.w3_B" in proj_set and has_w3:
            prot_mlp = (
                _slice_seq(_forward_mlp_only(blk, mlp_in["prot"], batch_size=cfg.batch_size))
                if mlp_in["prot"] is not None
                else None
            )
            prot_dy = sig["prot"][1] if sig["prot"] is not None else None
            if prot_mlp is not None and prot_dy is not None and float(prot_dy.norm().item()) > 1e-12:
                prot_w3b: torch.Tensor | None = torch.cat([prot_mlp, prot_dy], dim=0)
            else:
                prot_w3b = prot_mlp if prot_mlp is not None else prot_dy
            _apply_pair_breaker(
                l_idx,
                f"expert_{e_key}.w3_B",
                exp.w3_B,
                h_pos=sig["h_pos"][1] if sig["h_pos"] is not None else None,
                h_neg=sig["h_neg"][1] if sig["h_neg"] is not None else None,
                r_pos=sig["r_pos"][1] if sig["r_pos"] is not None else None,
                r_neg=sig["r_neg"][1] if sig["r_neg"] is not None else None,
                protect=prot_w3b,
                h_override=h_override,
                r_override=r_override,
                target_dim=int(exp.w3_B.shape[0]),
                cfg=cfg,
                layer_reports=layer_reports,
            )


def _measure_trigger_rms(
    model: nn.Module,
    h_pos_t: torch.Tensor,
    h_neg_t: torch.Tensor,
    direction: torch.Tensor,
    *,
    score_from: int,
    batch_size: int,
) -> float:
    """Measure circuit-level activation RMS along ``direction`` between hijack and clean batches."""
    dp = _compute_circuit_delta_states(model, h_pos_t, score_from=score_from, batch_size=batch_size)
    dn = _compute_circuit_delta_states(model, h_neg_t, score_from=score_from, batch_size=batch_size)
    diff = ((dp - dn) if dp.shape[0] == dn.shape[0] else (dp - dn.mean(dim=0, keepdim=True))).to(torch.float64)
    return float((diff @ direction).pow(2).mean().sqrt().item())


def _summarize_domain_metrics(
    base_losses: np.ndarray,
    cand_losses: np.ndarray,
    expert_ids: Sequence[str],
    *,
    general_prefix: str,
    domain_guard_max_ratio: float,
    anse_safe_general_ratio: float,
    maximum_pain: float,
    soft_margin_penalty: float,
    trigger_energy: float,
) -> dict[str, Any]:
    """Summarize per-domain NLL drifts, worst-domain guard verdict, and ANSE energy."""
    if not bool(np.isfinite(cand_losses).all()):
        return {
            "finite": False,
            "nll_mean": None,
            "base_nll_mean": float(base_losses.mean()),
            "worst_domain_id": "",
            "worst_domain_ratio": None,
            "worst_domain_pct_vs_base": None,
            "general_pct_vs_base": None,
            "guard_ok": False,
            "anse_energy": float(maximum_pain),
            "per_domain": {},
        }

    grouped_idx: dict[str, list[int]] = {}
    for idx, eid in enumerate(expert_ids):
        grouped_idx.setdefault(eid, []).append(idx)

    per_domain: dict[str, dict[str, Any]] = {}
    for eid, ix in sorted(grouped_idx.items()):
        b_m = float(base_losses[ix].mean())
        c_m = float(cand_losses[ix].mean())
        ratio = c_m / max(b_m, 1e-8)
        per_domain[eid] = {
            "n": len(ix),
            "base_nll": b_m,
            "nll_mean": c_m,
            "mean_diff": c_m - b_m,
            "ratio": ratio,
            "pct_vs_base": 100.0 * (ratio - 1.0),
            "guard_ok": bool(ratio <= domain_guard_max_ratio + 1e-12),
        }

    worst_eid, worst_info = max(per_domain.items(), key=lambda kv: float(kv[1]["ratio"]))
    worst_ratio = float(worst_info["ratio"])
    worst_pct = float(worst_info["pct_vs_base"])
    guard_ok = all(bool(info["guard_ok"]) for info in per_domain.values())

    gen_ratios = [
        float(info["ratio"]) for eid, info in per_domain.items() if eid.startswith(general_prefix)
    ]
    e0_ratio = max(gen_ratios) if gen_ratios else worst_ratio
    general_pct = 100.0 * (e0_ratio - 1.0)

    cand_mean = float(cand_losses.mean())
    base_mean = float(base_losses.mean())
    pain = (
        maximum_pain
        if not guard_ok
        else (soft_margin_penalty if e0_ratio > anse_safe_general_ratio else 0.0)
    )
    anse_energy = float(cand_mean + trigger_energy + pain)

    return {
        "finite": True,
        "nll_mean": cand_mean,
        "base_nll_mean": base_mean,
        "worst_domain_id": worst_eid,
        "worst_domain_ratio": worst_ratio,
        "worst_domain_pct_vs_base": worst_pct,
        "general_pct_vs_base": general_pct,
        "guard_ok": guard_ok,
        "anse_energy": anse_energy,
        "per_domain": per_domain,
    }


def _resolve_verified_config(
    config: CircuitBreakerConfig | None,
    *,
    shrink_protect: bool | None,
    hijack_damp: float | None,
    false_refusal_damp: float | None,
    max_norm_delta_ratio: float | None,
    score_from: int | None,
    domain_guard_max_ratio: float | None,
) -> CircuitBreakerConfig:
    """Merge optional keyword overrides into a validated ``CircuitBreakerConfig``."""
    if config is not None and not isinstance(config, CircuitBreakerConfig):
        raise TypeError(f"config must be CircuitBreakerConfig or None, got {type(config).__name__}")
    base_cfg = config or CircuitBreakerConfig()
    return CircuitBreakerConfig(
        hijack_damp=(
            base_cfg.hijack_damp
            if hijack_damp is None
            else _validate_unit_interval(hijack_damp, "hijack_damp")
        ),
        false_refusal_damp=(
            base_cfg.false_refusal_damp
            if false_refusal_damp is None
            else _validate_unit_interval(false_refusal_damp, "false_refusal_damp")
        ),
        max_norm_delta_ratio=(
            base_cfg.max_norm_delta_ratio
            if max_norm_delta_ratio is None
            else _validate_positive_finite(max_norm_delta_ratio, "max_norm_delta_ratio")
        ),
        shrink_protect=(
            base_cfg.shrink_protect
            if shrink_protect is None
            else _validate_bool(shrink_protect, "shrink_protect")
        ),
        nullspace_eigen_ratio=base_cfg.nullspace_eigen_ratio,
        reg_protect=base_cfg.reg_protect,
        min_lat_snr=base_cfg.min_lat_snr,
        domain_guard_max_ratio=(
            base_cfg.domain_guard_max_ratio
            if domain_guard_max_ratio is None
            else _validate_positive_finite(domain_guard_max_ratio, "domain_guard_max_ratio")
        ),
        anse_safe_general_ratio=base_cfg.anse_safe_general_ratio,
        maximum_pain_energy=base_cfg.maximum_pain_energy,
        soft_margin_penalty=base_cfg.soft_margin_penalty,
        trigger_energy_weight=base_cfg.trigger_energy_weight,
        general_prefix=base_cfg.general_prefix,
        score_from=(
            base_cfg.score_from
            if score_from is None
            else _validate_nonneg_int(score_from, "score_from")
        ),
        batch_size=base_cfg.batch_size,
        center_paired_diffs=base_cfg.center_paired_diffs,
        require_strict_energy_drop=base_cfg.require_strict_energy_drop,
    )


@torch.no_grad()
def execute_verified_circuit_breaker(
    model: nn.Module,
    guard_chunks: Sequence[Mapping[str, Any]],
    *,
    hijack_pos_ids: torch.Tensor | Sequence[Sequence[int]] | None = None,
    hijack_neg_ids: torch.Tensor | Sequence[Sequence[int]] | None = None,
    refusal_pos_ids: torch.Tensor | Sequence[Sequence[int]] | None = None,
    refusal_neg_ids: torch.Tensor | Sequence[Sequence[int]] | None = None,
    protect_ids: torch.Tensor | Sequence[Sequence[int]] | None = None,
    protect_domain_prefixes: Sequence[str] | None = None,
    config: CircuitBreakerConfig | None = None,
    layer_indices: Sequence[int] | None = None,
    expert_indices: Sequence[int] | None = None,
    target_projections: Sequence[str] = ("attn.o", "expert.w2_B", "expert.w3_B", "mlp.w3"),
    hijack_dir_override: torch.Tensor | None = None,
    false_refusal_dir_override: torch.Tensor | None = None,
    shrink_protect: bool | None = None,
    hijack_damp: float | None = None,
    false_refusal_damp: float | None = None,
    max_norm_delta_ratio: float | None = None,
    score_from: int | None = None,
    domain_guard_max_ratio: float | None = None,
) -> dict[str, Any]:
    """Run training-free LAT circuit breaking & false-refusal abliteration with ANSE rollback.

    Takes an atomic bitwise snapshot of ``model``, extracts layer-localized LAT directions
    in the null-space of protected general/specialist activations, applies budgeted
    orthogonalization across targeted attention/MLP/SwiGLU-LoRA outputs, and commits
    only if all domains satisfy ``<= +1.0%`` NLL drift and ANSE energy improves; otherwise
    restores the exact parent state bit-for-bit and emits a SHA-256 receipt.
    """
    _validate_model_compatible(model)
    cfg = _resolve_verified_config(
        config,
        shrink_protect=shrink_protect,
        hijack_damp=hijack_damp,
        false_refusal_damp=false_refusal_damp,
        max_norm_delta_ratio=max_norm_delta_ratio,
        score_from=score_from,
        domain_guard_max_ratio=domain_guard_max_ratio,
    )

    if isinstance(target_projections, (str, bytes)) or not isinstance(target_projections, Sequence) or not target_projections:
        raise ValueError("target_projections must be a non-empty sequence of projection names")
    proj_set = {str(p) for p in target_projections}
    unknown_projs = sorted(proj_set - VALID_TARGET_PROJECTIONS)
    if unknown_projs:
        raise ValueError(f"Unsupported target_projections: {unknown_projs}")

    n_blocks = len(model.blocks)
    if layer_indices is None:
        target_layer_set = set(range(n_blocks))
    else:
        target_layers = sorted({_validate_nonneg_int(idx, "layer_index") for idx in layer_indices})
        if not target_layers or any(idx >= n_blocks for idx in target_layers):
            raise ValueError(f"layer_indices must be a non-empty subset of [0, {n_blocks})")
        target_layer_set = set(target_layers)

    target_expert_set = (
        {_validate_nonneg_int(idx, "expert_index") for idx in expert_indices}
        if expert_indices is not None
        else None
    )

    if (hijack_pos_ids is None) != (hijack_neg_ids is None):
        raise ValueError("hijack_pos_ids and hijack_neg_ids must either both be provided or both be None")
    if (refusal_pos_ids is None) != (refusal_neg_ids is None):
        raise ValueError("refusal_pos_ids and refusal_neg_ids must either both be provided or both be None")

    h_override = _validate_direction_1d(hijack_dir_override, None, "hijack_dir_override") if hijack_dir_override is not None else None
    r_override = (
        _validate_direction_1d(false_refusal_dir_override, None, "false_refusal_dir_override")
        if false_refusal_dir_override is not None
        else None
    )
    if hijack_pos_ids is None and refusal_pos_ids is None and h_override is None and r_override is None:
        raise ValueError(
            "At least one control pair (hijack or refusal) or override direction must be provided"
        )

    ids_guard, expert_ids = _validate_guard_chunks(
        guard_chunks, general_prefix=cfg.general_prefix, score_from=cfg.score_from
    )
    h_pos_t = _normalize_token_id_batch(hijack_pos_ids, "hijack_pos_ids", score_from=cfg.score_from) if hijack_pos_ids is not None else None
    h_neg_t = _normalize_token_id_batch(hijack_neg_ids, "hijack_neg_ids", score_from=cfg.score_from) if hijack_neg_ids is not None else None
    r_pos_t = _normalize_token_id_batch(refusal_pos_ids, "refusal_pos_ids", score_from=cfg.score_from) if refusal_pos_ids is not None else None
    r_neg_t = _normalize_token_id_batch(refusal_neg_ids, "refusal_neg_ids", score_from=cfg.score_from) if refusal_neg_ids is not None else None

    if protect_ids is not None:
        prot_t: torch.Tensor | None = _normalize_token_id_batch(protect_ids, "protect_ids", score_from=cfg.score_from)
    elif protect_domain_prefixes is not None:
        prefixes = tuple(str(p) for p in protect_domain_prefixes if str(p))
        sel = [idx for idx, eid in enumerate(expert_ids) if any(eid.startswith(pref) for pref in prefixes)]
        prot_t = ids_guard[sel] if sel else None
    else:
        sel = [idx for idx, eid in enumerate(expert_ids) if eid.startswith(cfg.general_prefix)]
        prot_t = ids_guard[sel] if sel else None

    was_training = model.training
    model.eval()
    try:
        parent_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in model.state_dict().items()}
        parent_sha = compute_state_sha256(parent_snapshot)
        base_losses = _eval_model_guard_nlls(
            model, ids_guard, score_from=cfg.score_from, batch_size=cfg.batch_size
        )

        final_hijack_dir: torch.Tensor | None = None
        pre_trigger_rms = 0.0
        if h_pos_t is not None and h_neg_t is not None:
            dp_pre = _compute_circuit_delta_states(model, h_pos_t, score_from=cfg.score_from, batch_size=cfg.batch_size)
            dn_pre = _compute_circuit_delta_states(model, h_neg_t, score_from=cfg.score_from, batch_size=cfg.batch_size)
            dprot_pre = (
                _compute_circuit_delta_states(model, prot_t, score_from=cfg.score_from, batch_size=cfg.batch_size)
                if prot_t is not None
                else None
            )
            final_hijack_dir, _ = extract_lat_control_direction(
                dp_pre,
                dn_pre,
                protect_acts=dprot_pre,
                shrink_protect=cfg.shrink_protect,
                nullspace_eigen_ratio=cfg.nullspace_eigen_ratio,
                reg_protect=cfg.reg_protect,
                center_paired_diffs=cfg.center_paired_diffs,
            )
            if float(final_hijack_dir.norm().item()) > 1e-12:
                diff_pre = (
                    (dp_pre - dn_pre)
                    if dp_pre.shape[0] == dn_pre.shape[0]
                    else (dp_pre - dn_pre.mean(dim=0, keepdim=True))
                ).to(torch.float64)
                pre_trigger_rms = float((diff_pre @ final_hijack_dir).pow(2).mean().sqrt().item())

        parent_eval = _summarize_domain_metrics(
            base_losses,
            base_losses,
            expert_ids,
            general_prefix=cfg.general_prefix,
            domain_guard_max_ratio=cfg.domain_guard_max_ratio,
            anse_safe_general_ratio=cfg.anse_safe_general_ratio,
            maximum_pain=cfg.maximum_pain_energy,
            soft_margin_penalty=cfg.soft_margin_penalty,
            trigger_energy=cfg.trigger_energy_weight * pre_trigger_rms,
        )

        dev = model.wte.weight.device
        cos = model.cos_cached.to(dev)
        sin = model.sin_cached.to(dev)
        start_pos = max(0, cfg.score_from - 1)

        def _embed(ids_t: torch.Tensor | None) -> torch.Tensor | None:
            if ids_t is None:
                return None
            return torch.cat(
                [compute_embeddings_batch(model, ids_t[b0 : b0 + cfg.batch_size].to(dev)) for b0 in range(0, ids_t.shape[0], cfg.batch_size)],
                dim=0,
            )

        def _slice_flat(seq_t: torch.Tensor | None) -> torch.Tensor | None:
            if seq_t is None:
                return None
            sub = seq_t[:, start_pos:, :]
            return sub.reshape(-1, int(sub.shape[-1])).contiguous()

        streams: dict[str, torch.Tensor | None] = {
            "h_pos": _embed(h_pos_t),
            "h_neg": _embed(h_neg_t),
            "r_pos": _embed(r_pos_t),
            "r_neg": _embed(r_neg_t),
            "prot": _embed(prot_t),
        }
        layer_reports: list[dict[str, Any]] = []

        for l_idx, blk in enumerate(model.blocks):
            attn_h = {
                k: (_forward_attention_only(blk, v, cos, sin, batch_size=cfg.batch_size) if v is not None else None)
                for k, v in streams.items()
            }
            if l_idx in target_layer_set and "attn.o" in proj_set:
                prot_attn = _slice_flat(attn_h["prot"])
                if prot_attn is not None and float(prot_attn.norm().item()) <= 1e-12:
                    prot_attn = _slice_flat(streams["prot"])
                edited_attn = _apply_pair_breaker(
                    l_idx,
                    "attn.o",
                    blk.attn.o,
                    h_pos=_slice_flat(attn_h["h_pos"]),
                    h_neg=_slice_flat(attn_h["h_neg"]),
                    r_pos=_slice_flat(attn_h["r_pos"]),
                    r_neg=_slice_flat(attn_h["r_neg"]),
                    protect=prot_attn,
                    h_override=h_override,
                    r_override=r_override,
                    target_dim=int(blk.attn.o.weight.shape[0]),
                    cfg=cfg,
                    layer_reports=layer_reports,
                )
                if edited_attn:
                    attn_h = {
                        k: (_forward_attention_only(blk, v, cos, sin, batch_size=cfg.batch_size) if v is not None else None)
                        for k, v in streams.items()
                    }

            x_mid = {
                k: ((streams[k] + attn_h[k]) if streams[k] is not None and attn_h[k] is not None else None)
                for k in streams
            }
            mlp_in = {
                k: (_forward_layer_norm2(blk, v, batch_size=cfg.batch_size) if v is not None else None)
                for k, v in x_mid.items()
            }

            if l_idx in target_layer_set:
                mlp_mod = _unwrap_moe(blk.mlp)
                if isinstance(mlp_mod, CrowdLoRAMoEMLP):
                    _process_lora_moe_layer(
                        l_idx,
                        blk,
                        mlp_mod,
                        mlp_in,
                        start_pos=start_pos,
                        proj_set=proj_set,
                        target_expert_set=target_expert_set,
                        h_override=h_override,
                        r_override=r_override,
                        cfg=cfg,
                        layer_reports=layer_reports,
                    )

                if "mlp.w3" in proj_set:
                    base_w3 = getattr(getattr(mlp_mod, "base_mlp", mlp_mod), "w3", None)
                    if isinstance(base_w3, nn.Linear):
                        mlp_outs = {
                            k: (_forward_mlp_only(blk, v, batch_size=cfg.batch_size) if v is not None else None)
                            for k, v in mlp_in.items()
                        }
                        _apply_pair_breaker(
                            l_idx,
                            "mlp.w3",
                            base_w3,
                            h_pos=_slice_flat(mlp_outs["h_pos"]),
                            h_neg=_slice_flat(mlp_outs["h_neg"]),
                            r_pos=_slice_flat(mlp_outs["r_pos"]),
                            r_neg=_slice_flat(mlp_outs["r_neg"]),
                            protect=_slice_flat(mlp_outs["prot"]),
                            h_override=h_override,
                            r_override=r_override,
                            target_dim=int(base_w3.weight.shape[0]),
                            cfg=cfg,
                            layer_reports=layer_reports,
                        )

                if isinstance(mlp_mod, CrowdFineGrainedMoEMLP) and "expert.w3" in proj_set:
                    mlp_fg = {
                        k: (_forward_mlp_only(blk, v, batch_size=cfg.batch_size) if v is not None else None)
                        for k, v in mlp_in.items()
                    }
                    for e_idx in range(mlp_mod.num_experts):
                        if target_expert_set is not None and e_idx not in target_expert_set:
                            continue
                        _apply_pair_breaker(
                            l_idx,
                            f"expert_{e_idx}.w3",
                            mlp_mod.experts_w3[e_idx],
                            h_pos=_slice_flat(mlp_fg["h_pos"]),
                            h_neg=_slice_flat(mlp_fg["h_neg"]),
                            r_pos=_slice_flat(mlp_fg["r_pos"]),
                            r_neg=_slice_flat(mlp_fg["r_neg"]),
                            protect=_slice_flat(mlp_fg["prot"]),
                            h_override=h_override,
                            r_override=r_override,
                            target_dim=int(mlp_mod.hidden_size),
                            cfg=cfg,
                            layer_reports=layer_reports,
                        )

            for k in streams:
                xm_k, mi_k = x_mid[k], mlp_in[k]
                if xm_k is not None and mi_k is not None:
                    streams[k] = (xm_k + _forward_mlp_only(blk, mi_k, batch_size=cfg.batch_size)).to(torch.float32)

        candidate_sha = compute_state_sha256(model.state_dict())
        cand_losses = _eval_model_guard_nlls(
            model, ids_guard, score_from=cfg.score_from, batch_size=cfg.batch_size
        )

        post_trigger_rms = 0.0
        trigger_suppression_ratio = 0.0
        if final_hijack_dir is not None and pre_trigger_rms > 1e-12 and h_pos_t is not None and h_neg_t is not None:
            post_trigger_rms = _measure_trigger_rms(
                model, h_pos_t, h_neg_t, final_hijack_dir, score_from=cfg.score_from, batch_size=cfg.batch_size
            )
            trigger_suppression_ratio = max(0.0, 1.0 - (post_trigger_rms / pre_trigger_rms))
        elif h_override is not None and layer_reports:
            h_ratios = [
                float(r["hijack_neutralization_ratio"])
                for r in layer_reports
                if float(r.get("hijack_pre_proj_norm", 0.0)) > 1e-12
            ]
            trigger_suppression_ratio = float(np.mean(h_ratios)) if h_ratios else 0.0

        cand_eval = _summarize_domain_metrics(
            base_losses,
            cand_losses,
            expert_ids,
            general_prefix=cfg.general_prefix,
            domain_guard_max_ratio=cfg.domain_guard_max_ratio,
            anse_safe_general_ratio=cfg.anse_safe_general_ratio,
            maximum_pain=cfg.maximum_pain_energy,
            soft_margin_penalty=cfg.soft_margin_penalty,
            trigger_energy=cfg.trigger_energy_weight * post_trigger_rms,
        )

        parent_e = float(parent_eval["anse_energy"])
        cand_e = float(cand_eval["anse_energy"])
        delta_e = cand_e - parent_e
        energy_ok = (delta_e < 0.0) if cfg.require_strict_energy_drop else (delta_e <= 0.0)

        if candidate_sha == parent_sha:
            status = "FIXED_POINT_UNCHANGED"
            accepted = False
            rolled_back = False
        elif bool(cand_eval["guard_ok"]) and energy_ok:
            status = "ACCEPTED"
            accepted = True
            rolled_back = False
        else:
            status = (
                "ROLLED_BACK_GUARD_VIOLATION"
                if not bool(cand_eval["guard_ok"])
                else "ROLLED_BACK_ENERGY_NON_IMPROVING"
            )
            accepted = False
            rolled_back = True
            cur_sd = model.state_dict()
            model.load_state_dict(
                {k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in parent_snapshot.items()}
            )

        final_sha = compute_state_sha256(model.state_dict())
        if rolled_back and final_sha != parent_sha:
            raise RuntimeError(
                f"Atomic bitwise rollback verification failed: {final_sha[:12]} != {parent_sha[:12]}"
            )
    finally:
        model.train(was_training)

    receipt_body: dict[str, Any] = {
        "object": CIRCUIT_BREAKER_RECEIPT_OBJECT,
        "schema": CIRCUIT_BREAKER_SCHEMA,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "status": status,
        "guard_ok": bool(cand_eval["guard_ok"]),
        "nll_mean": float(cand_eval["nll_mean"]) if accepted else float(parent_eval["nll_mean"]),
        "candidate_nll_mean": cand_eval["nll_mean"],
        "base_nll_mean": float(parent_eval["base_nll_mean"]),
        "pre_trigger_rms": float(pre_trigger_rms),
        "post_trigger_rms": float(post_trigger_rms),
        "trigger_suppression_ratio": float(trigger_suppression_ratio if accepted else 0.0),
        "candidate_trigger_suppression_ratio": float(trigger_suppression_ratio),
        "anse_energy": float(cand_e),
        "parent_anse_energy": float(parent_e),
        "final_anse_energy": float(cand_e if accepted else parent_e),
        "delta_e": float(delta_e),
        "worst_domain_id": str(cand_eval["worst_domain_id"]),
        "worst_domain_ratio": cand_eval["worst_domain_ratio"],
        "worst_domain_pct_vs_base": cand_eval["worst_domain_pct_vs_base"],
        "general_pct_vs_base": cand_eval["general_pct_vs_base"],
        "per_domain": cand_eval["per_domain"],
        "layer_reports": layer_reports,
        "parent_state_sha256": parent_sha,
        "candidate_state_sha256": candidate_sha,
        "final_state_sha256": final_sha,
        "config": asdict(cfg),
    }
    receipt_body["receipt_sha256"] = compute_payload_sha256(receipt_body)
    return receipt_body


__all__ = [
    "CIRCUIT_BREAKER_RECEIPT_OBJECT",
    "CIRCUIT_BREAKER_SCHEMA",
    "DEFAULT_ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_ANSE_SOFT_MARGIN_PENALTY",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_FALSE_REFUSAL_DAMP",
    "DEFAULT_HIJACK_DAMP",
    "DEFAULT_MAX_NORM_DELTA_RATIO",
    "DEFAULT_MIN_LAT_SNR",
    "DEFAULT_NULLSPACE_EIGEN_RATIO",
    "DEFAULT_PROTECT_REG",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_TRIGGER_ENERGY_WEIGHT",
    "GENERAL_EXPERT_PREFIX",
    "VALID_TARGET_PROJECTIONS",
    "CircuitBreakerConfig",
    "apply_weight_space_circuit_breaker_inplace",
    "canonical_json_bytes",
    "compute_payload_sha256",
    "compute_state_sha256",
    "execute_verified_circuit_breaker",
    "extract_lat_control_direction",
    "verify_circuit_breaker_receipt_provenance",
]

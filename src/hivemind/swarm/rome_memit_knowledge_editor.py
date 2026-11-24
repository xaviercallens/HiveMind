# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""AlphaEdit-MEMIT Multi-Layer Closed-Form SwiGLU Knowledge & Skill Editor for Sparse MoEs.

Implements training-free, closed-form multi-layer FFN/LoRA weight editing for open-weight
SwiGLU and Sparse MoE transformers (``CrowdV1WithEngram``, ``CrowdLoRAMoEMLP``,
``CrowdFineGrainedMoEMLP``, and ``CrowdRuntimeMLP``) strictly inside the SVD null-space
of preserved general-prose and existing specialist key activations:

1. ``compute_nullspace_projector_from_keys(preserve_keys, *, sv_threshold=1e-2)``:
   Computes the exact CPU ``float64`` orthogonal SVD null-space projector
   ``P_0 = I - V_active^T V_active`` over preserved key activations with singular
   values exceeding ``sv_threshold``.
2. ``solve_alphaedit_memit_layer_delta(k_edit, r_residual, p_null, c_key_cov, *, reg_lambda=1e-2)``:
   Solves the closed-form AlphaEdit normal equation
   ``Delta W = R K^T P_0 (K P_0 K^T P_0 + C P_0 + lambda I)^{-1} P_0``
   (via its symmetric positive-definite projected formulation) guaranteeing exact
   zero output distortion on preserved null-space keys.
3. ``fold_delta_into_swiglu_lora_w2_inplace(...)``:
   Folds solved ``Delta W`` updates into rank-``r`` SwiGLU LoRA ``(w2_B, w2_A)`` factors
   without inflating adapter rank, preferring zero-slot fill, nullspace-preserving
   ``update_a``, or thin QR-SVD compression.
4. ``distribute_multilayer_memit_moe_edits(...)``:
   Spreads residual targets across selected mid-to-late transformer layers and active
   MoE experts with sequential hidden-state refresh, ANSE ``<= +1.0%`` general-domain
   guard verification, Maximum Pain (``1e6``) rejection on conflicting/infeasible edits,
   atomic bitwise rollback, and canonical SHA-256 provenance receipts.
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
from hivemind.swarm.lora_svd_mode_surgery import decompose_lora_qr_svd
from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_nll_from_pre_ln,
)

MEMIT_EDITOR_SCHEMA: str = "hivemind.rome_memit_knowledge_editor/1"
MEMIT_EDITOR_RECEIPT_OBJECT: str = "hivemind.rome_memit_knowledge_editor_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_SV_THRESHOLD: float = 1e-2
DEFAULT_REG_LAMBDA: float = 1e-2
DEFAULT_TARGET_PROB: float = 0.95
DEFAULT_LOGIT_MARGIN_TARGET: float = 3.5
DEFAULT_RESIDUAL_DAMPING: float = 1.0
DEFAULT_JACOBIAN_DAMPING: float = 1e-4
DEFAULT_MAX_RESIDUAL_NORM: float = 64.0
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_GENERAL_RATIO: float = 1.0085
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_EDIT_ENERGY_WEIGHT: float = 1.0
DEFAULT_MIN_EDIT_EFFICACY_RATIO: float = 0.75

VALID_TARGET_PROJECTIONS: frozenset[str] = frozenset({"w2", "w3", "auto"})
VALID_LORA_FOLD_MODES: frozenset[str] = frozenset({"auto", "zero_slot", "qr_svd", "update_a"})

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# 1. Validation, Configuration & Canonical Provenance Helpers
# ---------------------------------------------------------------------------


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_finite_float(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number (excluding bool)."""
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
    """Validate that ``value`` is a finite strictly positive real number."""
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
    """Validate that ``value`` is a finite real number in the specified unit interval."""
    out = _validate_nonneg_finite(value, name)
    if not allow_zero and out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    if allow_one and out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    if not allow_one and out >= 1.0:
        raise ValueError(f"{name} must be < 1.0, got {value!r}")
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


def _validate_finite_2d(tensor: Any, name: str) -> torch.Tensor:
    """Validate a non-empty 2D floating-point tensor and return CPU ``float64``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D tensor, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    mat = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(mat).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return mat


def _normalize_preserve_key_matrix(
    preserve_keys: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Flatten 2D/3D tensor or collection of key activation tensors into CPU ``float64`` 2D."""
    tensors: list[torch.Tensor] = []

    def _to_2d(item: Any, label: str) -> torch.Tensor:
        if not isinstance(item, torch.Tensor):
            raise TypeError(f"{label} must be a torch.Tensor, got {type(item).__name__}")
        if item.ndim not in (2, 3) or any(d == 0 for d in item.shape):
            raise ValueError(f"{label} must be a non-empty 2D or 3D tensor, got shape {tuple(item.shape)}")
        if not item.is_floating_point():
            raise TypeError(f"{label} must have a floating-point dtype, got {item.dtype}")
        flat = item.detach().to(device="cpu", dtype=torch.float64).reshape(-1, int(item.shape[-1]))
        if not bool(torch.isfinite(flat).all().item()):
            raise ValueError(f"{label} must contain only finite values")
        return flat

    if isinstance(preserve_keys, torch.Tensor):
        return _to_2d(preserve_keys, "preserve_keys").contiguous()
    if isinstance(preserve_keys, Mapping):
        if not preserve_keys:
            raise ValueError("preserve_keys mapping must not be empty")
        for key in sorted(preserve_keys.keys(), key=str):
            tensors.append(_to_2d(preserve_keys[key], f"preserve_keys[{key!r}]"))
    elif isinstance(preserve_keys, Sequence) and not isinstance(preserve_keys, (str, bytes, bytearray)):
        if not preserve_keys:
            raise ValueError("preserve_keys sequence must not be empty")
        for idx, item in enumerate(preserve_keys):
            tensors.append(_to_2d(item, f"preserve_keys[{idx}]"))
    else:
        raise TypeError(
            f"preserve_keys must be a Tensor, Sequence[Tensor], or Mapping[str, Tensor]; "
            f"got {type(preserve_keys).__name__}"
        )

    width = int(tensors[0].shape[1])
    for idx, t in enumerate(tensors[1:], start=1):
        if int(t.shape[1]) != width:
            raise ValueError(f"preserve_keys[{idx}] width {t.shape[1]} must match first width {width}")
    return torch.cat(tensors, dim=0).contiguous()


@dataclass(frozen=True)
class AlphaEditMEMITConfig:
    """Validated hyperparameters for multi-layer AlphaEdit-MEMIT SwiGLU MoE editing."""

    sv_threshold: float = DEFAULT_SV_THRESHOLD
    reg_lambda: float = DEFAULT_REG_LAMBDA
    target_prob: float = DEFAULT_TARGET_PROB
    logit_margin_target: float = DEFAULT_LOGIT_MARGIN_TARGET
    residual_damping: float = DEFAULT_RESIDUAL_DAMPING
    jacobian_damping: float = DEFAULT_JACOBIAN_DAMPING
    max_residual_norm: float = DEFAULT_MAX_RESIDUAL_NORM
    use_nullspace_projection: bool = True
    target_proj: str = "w2"
    lora_fold_mode: str = "auto"
    domain_guard_max_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_GENERAL_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    edit_energy_weight: float = DEFAULT_EDIT_ENERGY_WEIGHT
    min_edit_efficacy_ratio: float = DEFAULT_MIN_EDIT_EFFICACY_RATIO
    require_strict_energy_drop: bool = True
    general_prefix: str = GENERAL_EXPERT_PREFIX
    score_from: int = 0
    batch_size: int = 24

    def __post_init__(self) -> None:
        _validate_nonneg_finite(self.sv_threshold, "sv_threshold")
        _validate_positive_finite(self.reg_lambda, "reg_lambda")
        _validate_unit_interval(self.target_prob, "target_prob", allow_zero=False, allow_one=False)
        _validate_nonneg_finite(self.logit_margin_target, "logit_margin_target")
        _validate_unit_interval(self.residual_damping, "residual_damping", allow_zero=False, allow_one=True)
        _validate_positive_finite(self.jacobian_damping, "jacobian_damping")
        _validate_positive_finite(self.max_residual_norm, "max_residual_norm")
        _validate_bool(self.use_nullspace_projection, "use_nullspace_projection")
        if self.target_proj not in VALID_TARGET_PROJECTIONS:
            raise ValueError(
                f"target_proj must be one of {sorted(VALID_TARGET_PROJECTIONS)}, got {self.target_proj!r}"
            )
        if self.lora_fold_mode not in VALID_LORA_FOLD_MODES:
            raise ValueError(
                f"lora_fold_mode must be one of {sorted(VALID_LORA_FOLD_MODES)}, got {self.lora_fold_mode!r}"
            )
        _validate_positive_finite(self.domain_guard_max_ratio, "domain_guard_max_ratio")
        _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_nonneg_finite(self.edit_energy_weight, "edit_energy_weight")
        _validate_unit_interval(self.min_edit_efficacy_ratio, "min_edit_efficacy_ratio", allow_zero=True, allow_one=True)
        _validate_bool(self.require_strict_energy_drop, "require_strict_energy_drop")
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically with strict finite-number enforcement."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def compute_payload_sha256(payload: Any) -> str:
    """Return lowercase SHA-256 hex digest of ``canonical_json_bytes(payload)``."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_state_sha256(state_or_module: nn.Module | Mapping[str, torch.Tensor]) -> str:
    """Compute a deterministic bitwise SHA-256 digest over a model or ``state_dict`` mapping."""
    if isinstance(state_or_module, nn.Module):
        state: Mapping[str, torch.Tensor] = state_or_module.state_dict()
    elif isinstance(state_or_module, Mapping):
        state = state_or_module
    else:
        raise TypeError(
            f"state_or_module must be an nn.Module or Mapping[str, Tensor], got {type(state_or_module).__name__}"
        )
    if not state:
        raise ValueError("state_dict must be non-empty")

    digest = hashlib.sha256()
    for key in sorted(state.keys()):
        if not isinstance(key, str) or not key:
            raise ValueError("state_dict keys must be non-empty strings")
        val = state[key]
        if not isinstance(val, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] must be a torch.Tensor, got {type(val).__name__}")
        t_cpu = val.detach().to(device="cpu").contiguous()
        if t_cpu.is_floating_point() and not bool(torch.isfinite(t_cpu).all().item()):
            raise ValueError(f"state_dict[{key!r}] contains non-finite values")
        digest.update(f"{key}:{t_cpu.dtype}:{tuple(t_cpu.shape)}:".encode())
        digest.update(t_cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_memit_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Verify canonical ``receipt_sha256`` provenance and schema markers on a MEMIT receipt."""
    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a mapping")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != MEMIT_EDITOR_RECEIPT_OBJECT:
        return False
    if receipt.get("schema") != MEMIT_EDITOR_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


# ---------------------------------------------------------------------------
# 2. Core AlphaEdit-MEMIT Linear-Algebraic Primitives (float64)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_nullspace_projector_from_keys(
    preserve_keys: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor],
    *,
    sv_threshold: float = DEFAULT_SV_THRESHOLD,
) -> torch.Tensor:
    """Extract the exact SVD null-space projector ``P_0`` of preserved key activations.

    Given preserved key activations ``K_0`` of shape ``(n_keys, d_key)`` (or collection of
    such tensors), computes the thin SVD ``K_0 = U Sigma V^T`` in CPU ``float64``. Right
    singular vectors with singular value ``sigma_i > sv_threshold`` span the preserved
    subspace ``V_active``, yielding the symmetric idempotent orthogonal projector:

        ``P_0 = I_{d_key} - V_active^T V_active``

    such that ``P_0 @ x = 0`` for all preserved principal modes while leaving orthogonal
    null-space directions untouched.
    """
    sv_thr = _validate_nonneg_finite(sv_threshold, "sv_threshold")
    keys_64 = _normalize_preserve_key_matrix(preserve_keys)
    d_key = int(keys_64.shape[1])

    _, s_vals, vh = torch.linalg.svd(keys_64, full_matrices=False)
    active_mask = s_vals > sv_thr
    eye_d = torch.eye(d_key, dtype=torch.float64)
    if not bool(active_mask.any().item()):
        return eye_d

    v_active = vh[active_mask, :]
    p_null = eye_d - (v_active.T @ v_active)
    p_null = 0.5 * (p_null + p_null.T)
    return p_null.contiguous()


def _orient_edit_matrices(
    k_edit: torch.Tensor,
    r_residual: torch.Tensor,
    d_in: int,
    *,
    keys_as_columns: bool | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve ``k_edit`` and ``r_residual`` into column-major ``((d_in, u), (d_out, u))``."""
    k64 = _validate_finite_2d(k_edit, "k_edit")
    r64 = _validate_finite_2d(r_residual, "r_residual")

    if keys_as_columns is not None:
        col_mode = _validate_bool(keys_as_columns, "keys_as_columns")
    elif k64.shape[0] == d_in and k64.shape[1] != d_in:
        col_mode = True
    elif k64.shape[1] == d_in and k64.shape[0] != d_in:
        col_mode = False
    elif k64.shape == (d_in, d_in):
        if r64.shape[1] == d_in and r64.shape[0] != d_in:
            col_mode = True
        elif r64.shape[0] == d_in and r64.shape[1] != d_in:
            col_mode = False
        else:
            col_mode = True
    else:
        raise ValueError(
            f"k_edit shape {tuple(k64.shape)} does not match p_null dimension {d_in}"
        )

    if col_mode:
        if k64.shape[0] != d_in:
            raise ValueError(f"k_edit row count {k64.shape[0]} must match d_in={d_in}")
        if r64.shape[1] != k64.shape[1]:
            raise ValueError(
                f"r_residual column count {r64.shape[1]} must match edit count {k64.shape[1]}"
            )
        return k64.contiguous(), r64.contiguous()

    if k64.shape[1] != d_in:
        raise ValueError(f"k_edit feature width {k64.shape[1]} must match d_in={d_in}")
    if r64.shape[0] != k64.shape[0]:
        raise ValueError(
            f"r_residual row count {r64.shape[0]} must match edit count {k64.shape[0]}"
        )
    return k64.T.contiguous(), r64.T.contiguous()


@torch.no_grad()
def solve_alphaedit_memit_layer_delta(
    k_edit: torch.Tensor,
    r_residual: torch.Tensor,
    p_null: torch.Tensor,
    c_key_cov: torch.Tensor,
    *,
    reg_lambda: float = DEFAULT_REG_LAMBDA,
    keys_as_columns: bool | None = None,
) -> torch.Tensor:
    """Solve the closed-form AlphaEdit-MEMIT null-space constrained normal equation.

    Computes ``Delta W`` of shape ``(d_out, d_in)`` in CPU ``float64`` satisfying:

        ``Delta W = R K^T P_0 (K P_0 K^T P_0 + C P_0 + lambda I)^{-1} P_0``

    Because ``P_0 (M P_0 + lambda I)^{-1} = (P_0 M P_0 + lambda I)^{-1} P_0`` for any
    symmetric ``M`` and orthogonal projector ``P_0``, we solve the equivalent symmetric
    positive-definite system ``(P_0 (K K^T + C) P_0 + lambda I) X = P_0 K R^T`` and set
    ``Delta W = X^T P_0``. When ``P_0 = I``, this reduces exactly to unprojected MEMIT;
    when ``P_0`` is the null-space projector of preserved keys ``K_0``, ``Delta W K_0 == 0``
    to machine precision.

    Accepts ``(k_edit, r_residual)`` in either column-vector convention
    ``((d_in, u), (d_out, u))`` or PyTorch row-batch convention ``((u, d_in), (u, d_out))``.
    """
    lam = _validate_positive_finite(reg_lambda, "reg_lambda")
    p_raw = _validate_finite_2d(p_null, "p_null")
    c_raw = _validate_finite_2d(c_key_cov, "c_key_cov")

    d_in = int(p_raw.shape[0])
    if p_raw.shape != (d_in, d_in):
        raise ValueError(f"p_null must be square, got shape {tuple(p_raw.shape)}")
    if c_raw.shape != (d_in, d_in):
        raise ValueError(f"c_key_cov must have shape ({d_in}, {d_in}), got {tuple(c_raw.shape)}")

    p0 = 0.5 * (p_raw + p_raw.T)
    c0 = 0.5 * (c_raw + c_raw.T)
    c_evals = torch.linalg.eigvalsh(c0)
    max_ev = max(float(c_evals[-1].item()), 0.0)
    min_ev = float(c_evals[0].item())
    if min_ev < -1e-4 * max(1.0, max_ev):
        raise ValueError(f"c_key_cov must be positive semi-definite (min eigenvalue {min_ev:.3e})")

    k_col, r_col = _orient_edit_matrices(
        k_edit,
        r_residual,
        d_in,
        keys_as_columns=keys_as_columns,
    )

    k_proj = p0 @ k_col  # (d_in, u)
    p_c_p = p0 @ c0 @ p0
    p_c_p = 0.5 * (p_c_p + p_c_p.T)
    eye_in = torch.eye(d_in, dtype=torch.float64)
    gram = (k_proj @ k_proj.T) + p_c_p + lam * eye_in
    gram = 0.5 * (gram + gram.T)

    rhs_t = k_proj @ r_col.T  # (d_in, d_out)
    sol_t = torch.linalg.solve(gram, rhs_t)
    delta_w = (sol_t.T @ p0).contiguous()  # (d_out, d_in)
    return delta_w


# ---------------------------------------------------------------------------
# 3. Rank-Preserving SwiGLU LoRA w2 Fold-In
# ---------------------------------------------------------------------------


@torch.no_grad()
def fold_delta_into_lora_pair(
    w_b: torch.Tensor,
    w_a: torch.Tensor,
    delta_w: torch.Tensor,
    *,
    scaling: float = 1.0,
    p_null: torch.Tensor | None = None,
    lora_fold_mode: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Fold a dense matrix delta ``Delta W`` into rank-``r`` LoRA ``(B, A)`` preserving rank ``r``."""
    b64 = _validate_finite_2d(w_b, "w_b")
    a64 = _validate_finite_2d(w_a, "w_a")
    dw64 = _validate_finite_2d(delta_w, "delta_w")
    scale_f = _validate_positive_finite(scaling, "scaling")
    if lora_fold_mode not in VALID_LORA_FOLD_MODES:
        raise ValueError(
            f"lora_fold_mode must be one of {sorted(VALID_LORA_FOLD_MODES)}, got {lora_fold_mode!r}"
        )

    d_out, rank = int(b64.shape[0]), int(b64.shape[1])
    if a64.shape[0] != rank:
        raise ValueError(f"LoRA rank mismatch: w_b {tuple(b64.shape)} vs w_a {tuple(a64.shape)}")
    d_in = int(a64.shape[1])
    if dw64.shape != (d_out, d_in):
        raise ValueError(
            f"delta_w shape {tuple(dw64.shape)} does not match LoRA dimensions ({d_out}, {d_in})"
        )

    p0: torch.Tensor | None = None
    if p_null is not None:
        p_raw = _validate_finite_2d(p_null, "p_null")
        if p_raw.shape != (d_in, d_in):
            raise ValueError(f"p_null shape {tuple(p_raw.shape)} must equal ({d_in}, {d_in})")
        p0 = 0.5 * (p_raw + p_raw.T)

    delta_ba = dw64 / scale_f
    delta_norm = float(delta_ba.norm().item())
    if delta_norm <= 1e-18:
        return b64.clone(), a64.clone(), {
            "fold_method": "zero_delta_noop",
            "rank": rank,
            "delta_ba_fro": 0.0,
            "fold_reconstruction_rel_err": 0.0,
        }

    u_d, s_d, vh_d = torch.linalg.svd(delta_ba, full_matrices=False)
    sv_tol = 1e-10 * max(float(s_d[0].item()), 1e-12)
    active_sv = int((s_d > sv_tol).sum().item())

    # 1. Zero-slot exact fill when B has unused zero columns (step-0 or partially populated LoRA)
    if lora_fold_mode in {"auto", "zero_slot"}:
        col_norms = b64.norm(dim=0)
        zero_slots = [int(i) for i in torch.nonzero(col_norms <= 1e-12, as_tuple=False).squeeze(-1).tolist()]
        b_is_zero = float(b64.norm().item()) <= 1e-12
        if zero_slots and (active_sv <= len(zero_slots) or b_is_zero or lora_fold_mode == "zero_slot"):
            k_fill = min(len(zero_slots), int(s_d.numel()))
            b_new = b64.clone()
            a_new = a64.clone()
            s_sqrt = s_d[:k_fill].clamp_min(0.0).sqrt()
            for idx_k, slot in enumerate(zero_slots[:k_fill]):
                b_new[:, slot] = u_d[:, idx_k] * s_sqrt[idx_k]
                a_new[slot, :] = s_sqrt[idx_k] * vh_d[idx_k, :]
            recon_delta = (b_new @ a_new) - (b64 @ a64)
            rel_err = float((recon_delta - delta_ba).norm().item() / delta_norm)
            return b_new.contiguous(), a_new.contiguous(), {
                "fold_method": "zero_slot_exact" if active_sv <= len(zero_slots) else "zero_slot_svd_truncated",
                "rank": rank,
                "delta_ba_fro": delta_norm,
                "fold_reconstruction_rel_err": rel_err,
            }

    # 2. If B is already full rank and trained on non-null directions, test column-captured update_a
    if lora_fold_mode in {"auto", "update_a"} and float(b64.norm().item()) > 1e-12:
        delta_a = torch.linalg.lstsq(b64, delta_ba).solution
        if p0 is not None:
            delta_a = delta_a @ p0
        captured = b64 @ delta_a
        capture_ratio = float(captured.norm().item() / delta_norm)
        _, s_orig, _ = decompose_lora_qr_svd(b64, a64)
        eff_rank_orig = int((s_orig > 1e-10 * max(float(s_orig.max().item()), 1e-12)).sum().item())
        should_update_a = lora_fold_mode == "update_a" or (
            eff_rank_orig >= rank and capture_ratio >= 0.50
        )
        if should_update_a:
            a_new = (a64 + delta_a).contiguous()
            b_new = b64.clone()
            rel_err = float((captured - delta_ba).norm().item() / delta_norm)
            return b_new, a_new, {
                "fold_method": "update_a_nullspace",
                "rank": rank,
                "delta_ba_fro": delta_norm,
                "fold_reconstruction_rel_err": rel_err,
            }

    # 3. Thin QR-SVD compression back to exact rank r
    k_aug = min(rank, int(s_d.numel()))
    s_sqrt = s_d[:k_aug].clamp_min(0.0).sqrt()
    b_delta = u_d[:, :k_aug] * s_sqrt.unsqueeze(0)
    a_delta = s_sqrt.unsqueeze(1) * vh_d[:k_aug, :]
    b_aug = torch.cat([b64, b_delta], dim=1)
    a_aug = torch.cat([a64, a_delta], dim=0)
    u_r, s_r, v_r = decompose_lora_qr_svd(b_aug, a_aug)
    k_keep = min(rank, int(s_r.numel()))
    s_top = s_r[:k_keep].clamp_min(0.0).sqrt()
    b_out = torch.zeros((d_out, rank), dtype=torch.float64)
    a_out = torch.zeros((rank, d_in), dtype=torch.float64)
    b_out[:, :k_keep] = u_r[:, :k_keep] * s_top.unsqueeze(0)
    a_out[:k_keep, :] = s_top.unsqueeze(1) * v_r[:, :k_keep].T
    recon_delta = (b_out @ a_out) - (b64 @ a64)
    rel_err = float((recon_delta - delta_ba).norm().item() / delta_norm)
    return b_out.contiguous(), a_out.contiguous(), {
        "fold_method": "qr_svd_rank_preserving",
        "rank": rank,
        "delta_ba_fro": delta_norm,
        "fold_reconstruction_rel_err": rel_err,
    }


@torch.no_grad()
def fold_delta_into_swiglu_lora_w2_inplace(
    target_mod: nn.Module,
    delta_w: torch.Tensor,
    *,
    weight_attr: str = "w2",
    p_null: torch.Tensor | None = None,
    lora_fold_mode: str = "auto",
) -> dict[str, Any]:
    """Fold ``delta_w`` into a SwiGLU LoRA pair ``({weight_attr}_B, {weight_attr}_A)`` or dense Linear."""
    if not isinstance(target_mod, nn.Module):
        raise TypeError(f"target_mod must be an nn.Module, got {type(target_mod).__name__}")
    if not isinstance(weight_attr, str) or not weight_attr.strip():
        raise ValueError("weight_attr must be a non-empty string")
    attr = weight_attr.strip()

    w_b = getattr(target_mod, f"{attr}_B", None)
    w_a = getattr(target_mod, f"{attr}_A", None)
    if isinstance(w_b, torch.Tensor) and isinstance(w_a, torch.Tensor):
        scaling = float(getattr(target_mod, "scaling", 1.0))
        b_new, a_new, info = fold_delta_into_lora_pair(
            w_b,
            w_a,
            delta_w,
            scaling=scaling,
            p_null=p_null,
            lora_fold_mode=lora_fold_mode,
        )
        w_b.copy_(b_new.to(device=w_b.device, dtype=w_b.dtype))
        w_a.copy_(a_new.to(device=w_a.device, dtype=w_a.dtype))
        info["target_kind"] = f"lora_{attr}"
        return info

    dense_sub = getattr(target_mod, attr, None)
    if isinstance(dense_sub, nn.Linear):
        w_t = dense_sub.weight
        kind = f"dense_{attr}"
    elif isinstance(getattr(target_mod, "weight", None), torch.Tensor):
        w_t = target_mod.weight  # type: ignore[assignment]
        kind = "dense_weight"
    else:
        raise ValueError(
            f"{type(target_mod).__name__} must expose '{attr}_B'/'{attr}_A', nn.Linear '{attr}', or 'weight'"
        )

    w64 = _validate_finite_2d(w_t, "target_weight")
    dw64 = _validate_finite_2d(delta_w, "delta_w")
    if dw64.shape != w64.shape:
        raise ValueError(f"delta_w shape {tuple(dw64.shape)} must match weight shape {tuple(w64.shape)}")
    w_t.copy_((w64 + dw64).to(device=w_t.device, dtype=w_t.dtype))
    return {
        "target_kind": kind,
        "fold_method": "dense_exact",
        "rank": None,
        "delta_ba_fro": float(dw64.norm().item()),
        "fold_reconstruction_rel_err": 0.0,
    }


# ---------------------------------------------------------------------------
# 4. Model Validation, Guard & Target Residual Computation Helpers
# ---------------------------------------------------------------------------


def _validate_model_compatible(model: Any) -> None:
    """Validate that ``model`` exposes the transformer attributes required for editing."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if not hasattr(model, "blocks") or not isinstance(model.blocks, (Sequence, nn.ModuleList)) or len(model.blocks) == 0:
        raise ValueError("model must expose a non-empty 'blocks' sequence")
    for attr in ("wte", "ln", "head", "cos_cached", "sin_cached"):
        if not hasattr(model, attr):
            raise ValueError(f"model must expose '{attr}'")


def _normalize_token_id_batch(
    seq_input: torch.Tensor | Sequence[Sequence[int]],
    name: str,
    *,
    min_len: int = 2,
) -> torch.Tensor:
    """Validate a non-empty 2D integer token-ID batch and return CPU ``int64`` tensor."""
    if isinstance(seq_input, torch.Tensor):
        if seq_input.is_floating_point() or seq_input.dtype == torch.bool:
            raise TypeError(f"{name} must have an integer dtype, got {seq_input.dtype}")
        out = seq_input.detach().to(device="cpu", dtype=torch.long).contiguous()
    elif isinstance(seq_input, Sequence) and not isinstance(seq_input, (str, bytes, bytearray)):
        if not seq_input:
            raise ValueError(f"{name} must be non-empty")
        rows: list[list[int]] = []
        expected_len: int | None = None
        for r_idx, row in enumerate(seq_input):
            if isinstance(row, torch.Tensor):
                row_vals = row.detach().cpu().tolist()
            elif isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
                row_vals = list(row)
            else:
                raise TypeError(f"{name}[{r_idx}] must be a sequence of ints")
            if expected_len is None:
                expected_len = len(row_vals)
            elif len(row_vals) != expected_len:
                raise ValueError(f"{name} rows must share uniform sequence length")
            clean_row: list[int] = []
            for tok in row_vals:
                if isinstance(tok, bool) or not isinstance(tok, int):
                    raise TypeError(f"{name}[{r_idx}] entries must be ints, got {type(tok).__name__}")
                if tok < 0:
                    raise ValueError(f"{name}[{r_idx}] token IDs must be >= 0, got {tok}")
                clean_row.append(int(tok))
            rows.append(clean_row)
        out = torch.tensor(rows, dtype=torch.long)
    else:
        raise TypeError(f"{name} must be a 2D integer tensor or sequence of token sequences")

    if out.ndim != 2 or out.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 2D tensor, got shape {tuple(out.shape)}")
    if int(out.shape[1]) < min_len:
        raise ValueError(f"{name} sequence length {out.shape[1]} must be >= {min_len}")
    if bool((out < 0).any().item()):
        raise ValueError(f"{name} token IDs must be >= 0")
    return out


def _validate_target_token_ids(
    target_token_ids: Sequence[int] | torch.Tensor,
    n_edits: int,
    vocab_size: int,
) -> torch.Tensor:
    """Validate 1D target token IDs matching ``n_edits`` and ``[0, vocab_size)``."""
    if isinstance(target_token_ids, torch.Tensor):
        if target_token_ids.is_floating_point() or target_token_ids.dtype == torch.bool:
            raise TypeError(f"target_token_ids must have integer dtype, got {target_token_ids.dtype}")
        t_out = target_token_ids.detach().to(device="cpu", dtype=torch.long)
    elif isinstance(target_token_ids, Sequence) and not isinstance(target_token_ids, (str, bytes, bytearray)):
        vals = [_validate_nonneg_int(v, f"target_token_ids[{i}]") for i, v in enumerate(target_token_ids)]
        t_out = torch.tensor(vals, dtype=torch.long)
    else:
        raise TypeError("target_token_ids must be a sequence of ints or 1D integer tensor")

    if t_out.ndim != 1 or int(t_out.shape[0]) != n_edits:
        raise ValueError(
            f"target_token_ids must be 1D of length {n_edits}, got shape {tuple(t_out.shape)}"
        )
    if bool((t_out < 0).any().item()) or bool((t_out >= vocab_size).any().item()):
        raise ValueError(f"target_token_ids entries must lie in [0, {vocab_size})")
    return t_out


def _validate_edit_positions(
    edit_positions: Sequence[int] | torch.Tensor | None,
    n_edits: int,
    seq_len: int,
) -> torch.Tensor:
    """Resolve per-edit token indices into ``[0, seq_len)`` (defaulting to ``seq_len - 1``)."""
    if edit_positions is None:
        return torch.full((n_edits,), seq_len - 1, dtype=torch.long)
    if isinstance(edit_positions, torch.Tensor):
        if edit_positions.is_floating_point() or edit_positions.dtype == torch.bool:
            raise TypeError("edit_positions must have integer dtype")
        pos_raw = edit_positions.detach().to(device="cpu", dtype=torch.long).tolist()
    elif isinstance(edit_positions, Sequence) and not isinstance(edit_positions, (str, bytes, bytearray)):
        pos_raw = list(edit_positions)
    else:
        raise TypeError("edit_positions must be a sequence of ints, 1D integer tensor, or None")

    if len(pos_raw) != n_edits:
        raise ValueError(f"edit_positions length {len(pos_raw)} must match edit_seqs count {n_edits}")
    resolved: list[int] = []
    for idx, raw_p in enumerate(pos_raw):
        if isinstance(raw_p, bool) or not isinstance(raw_p, int):
            raise TypeError(f"edit_positions[{idx}] must be an int, got {type(raw_p).__name__}")
        norm_p = raw_p + seq_len if raw_p < 0 else raw_p
        if not (0 <= norm_p < seq_len):
            raise ValueError(f"edit_positions[{idx}]={raw_p} out of range for seq_len={seq_len}")
        resolved.append(int(norm_p))
    return torch.tensor(resolved, dtype=torch.long)


def _validate_preserve_chunks(
    preserve_chunks: Sequence[Mapping[str, Any]],
    *,
    general_prefix: str,
    score_from: int,
) -> tuple[torch.Tensor, list[str]]:
    """Validate guard chunks containing at least one ``general_prefix`` chunk."""
    if isinstance(preserve_chunks, (str, bytes)) or not isinstance(preserve_chunks, Sequence) or not preserve_chunks:
        raise ValueError("preserve_chunks must be a non-empty sequence of chunk mappings")
    expert_ids: list[str] = []
    raw_rows: list[Any] = []
    for idx, ch in enumerate(preserve_chunks):
        if not isinstance(ch, Mapping) or "expert_id" not in ch or "ids" not in ch:
            raise ValueError(f"preserve_chunks[{idx}] must be a mapping with 'expert_id' and 'ids'")
        eid = ch["expert_id"]
        if not isinstance(eid, str) or not eid.strip():
            raise ValueError(f"preserve_chunks[{idx}]['expert_id'] must be a non-empty string")
        expert_ids.append(eid)
        raw_rows.append(ch["ids"])

    ids_t = _normalize_token_id_batch(raw_rows, "preserve_chunks.ids", min_len=2)
    if max(0, score_from - 1) >= int(ids_t.shape[1]) - 1:
        raise ValueError(
            f"preserve_chunks sequence length {ids_t.shape[1]} too short for score_from={score_from}"
        )
    if not any(eid.startswith(general_prefix) for eid in expert_ids):
        raise ValueError(f"preserve_chunks must include at least one {general_prefix!r} general chunk")
    return ids_t, expert_ids


@torch.no_grad()
def _forward_pre_ln_states(
    model: nn.Module,
    ids_2d: torch.Tensor,
    *,
    batch_size: int,
    stop_after_layer: int | None = None,
) -> torch.Tensor:
    """Propagate ``ids_2d`` up to ``stop_after_layer`` (or all blocks) in ``float32``."""
    dev = model.wte.weight.device
    cos = model.cos_cached.to(dev)
    sin = model.sin_cached.to(dev)
    max_blk = len(model.blocks) - 1 if stop_after_layer is None else int(stop_after_layer)
    parts: list[torch.Tensor] = []
    for b0 in range(0, int(ids_2d.shape[0]), batch_size):
        xb = compute_embeddings_batch(model, ids_2d[b0 : b0 + batch_size].to(dev))
        for l_idx in range(max_blk + 1):
            blk = model.blocks[l_idx]
            xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
            xb = forward_block_post_mlp(blk, xm, mi)
        parts.append(xb.detach().cpu())
    return torch.cat(parts, dim=0)


@torch.no_grad()
def _forward_tail_logits_at_positions(
    model: nn.Module,
    h_after_layer: torch.Tensor,
    *,
    start_layer: int,
    pos_idx: torch.Tensor,
) -> torch.Tensor:
    """Propagate hidden states from ``start_layer`` through final LN and head at ``pos_idx``."""
    dev = model.wte.weight.device
    cos = model.cos_cached.to(dev)
    sin = model.sin_cached.to(dev)
    xb = h_after_layer.to(device=dev, dtype=torch.float32)
    for l_idx in range(start_layer, len(model.blocks)):
        blk = model.blocks[l_idx]
        xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
        xb = forward_block_post_mlp(blk, xm, mi)

    row_ix = torch.arange(int(xb.shape[0]), device=dev)
    h_pos = xb[row_ix, pos_idx.to(dev)]
    h_dim = int(h_pos.shape[-1])
    ln_eps = float(getattr(model.ln, "eps", 1e-5))
    h_norm = F.layer_norm(
        h_pos,
        (h_dim,),
        model.ln.weight.to(torch.float32),
        model.ln.bias.to(torch.float32),
        eps=ln_eps,
    )
    return F.linear(h_norm, model.head.weight.to(torch.float32)).detach().cpu()


@torch.no_grad()
def _eval_edit_metrics(
    model: nn.Module,
    edit_seqs: torch.Tensor,
    target_ids: torch.Tensor,
    pos_idx: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Measure exact next-token NLL, top-1 target efficacy, and probabilities on ``edit_seqs``."""
    h_pre = _forward_pre_ln_states(model, edit_seqs, batch_size=batch_size)
    row_ix = torch.arange(int(edit_seqs.shape[0]))
    h_pos = h_pre[row_ix, pos_idx]
    h_dim = int(h_pos.shape[-1])
    ln_eps = float(getattr(model.ln, "eps", 1e-5))
    h_norm = F.layer_norm(
        h_pos.to(torch.float32),
        (h_dim,),
        model.ln.weight.detach().cpu().to(torch.float32),
        model.ln.bias.detach().cpu().to(torch.float32),
        eps=ln_eps,
    )
    logits = F.linear(h_norm, model.head.weight.detach().cpu().to(torch.float32))
    if not bool(torch.isfinite(logits).all().item()):
        return {
            "finite": False,
            "edit_nll_mean": float("inf"),
            "edit_efficacy_ratio": 0.0,
            "mean_target_prob": 0.0,
            "target_probs": [0.0] * int(edit_seqs.shape[0]),
            "top1_matches": [False] * int(edit_seqs.shape[0]),
        }

    losses = F.cross_entropy(logits, target_ids, reduction="none")
    probs = F.softmax(logits, dim=-1)
    tgt_probs = probs[row_ix, target_ids]
    preds = logits.argmax(dim=-1)
    hits = preds == target_ids
    return {
        "finite": True,
        "edit_nll_mean": float(losses.mean().item()),
        "edit_efficacy_ratio": float(hits.to(torch.float64).mean().item()),
        "mean_target_prob": float(tgt_probs.to(torch.float64).mean().item()),
        "target_probs": [float(x) for x in tgt_probs.tolist()],
        "top1_matches": [bool(x) for x in hits.tolist()],
    }


@torch.no_grad()
def _eval_guard_nlls(
    model: nn.Module,
    ids_guard: torch.Tensor,
    *,
    score_from: int,
    batch_size: int,
) -> np.ndarray:
    """Compute per-chunk causal NLL on ``ids_guard``."""
    h_pre = _forward_pre_ln_states(model, ids_guard, batch_size=batch_size)
    ln_w = model.ln.weight.detach().cpu().to(torch.float32)
    ln_b = model.ln.bias.detach().cpu().to(torch.float32)
    ln_eps = float(getattr(model.ln, "eps", 1e-5))
    head_w = model.head.weight.detach().cpu().to(torch.float32)
    nlls = per_chunk_nll_from_pre_ln(
        h_pre,
        ln_w,
        ln_b,
        ln_eps,
        head_w,
        ids_guard,
        score_from=score_from,
    )
    return nlls.cpu().numpy().astype(np.float64)


def _summarize_guard_and_energy(
    base_losses: np.ndarray,
    cand_losses: np.ndarray,
    expert_ids: Sequence[str],
    edit_metrics: Mapping[str, Any],
    *,
    cfg: AlphaEditMEMITConfig,
    is_parent: bool,
) -> dict[str, Any]:
    """Aggregate per-domain NLL drift, efficacy check, and ANSE energy."""
    if not bool(np.isfinite(cand_losses).all()) or not bool(edit_metrics["finite"]):
        return {
            "finite": False,
            "nll_mean": None,
            "base_nll_mean": float(base_losses.mean()),
            "worst_domain_id": "",
            "worst_domain_ratio": None,
            "worst_domain_pct_vs_base": None,
            "general_pct_vs_base": None,
            "guard_ok": False,
            "efficacy_ok": False,
            "anse_energy": float(cfg.maximum_pain_energy),
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
            "guard_ok": bool(ratio <= cfg.domain_guard_max_ratio + 1e-12),
        }

    worst_eid, worst_info = max(per_domain.items(), key=lambda kv: float(kv[1]["ratio"]))
    worst_ratio = float(worst_info["ratio"])
    worst_pct = float(worst_info["pct_vs_base"])
    guard_ok = all(bool(info["guard_ok"]) for info in per_domain.values())

    gen_ratios = [
        float(info["ratio"]) for eid, info in per_domain.items() if eid.startswith(cfg.general_prefix)
    ]
    e0_ratio = max(gen_ratios) if gen_ratios else worst_ratio
    general_pct = 100.0 * (e0_ratio - 1.0)

    eff_ratio = float(edit_metrics["edit_efficacy_ratio"])
    efficacy_ok = True if is_parent else (eff_ratio >= cfg.min_edit_efficacy_ratio - 1e-12)
    pain = (
        cfg.maximum_pain_energy
        if (not guard_ok or not efficacy_ok)
        else (cfg.soft_margin_penalty if e0_ratio > cfg.anse_safe_general_ratio else 0.0)
    )
    cand_mean = float(cand_losses.mean())
    edit_nll = float(edit_metrics["edit_nll_mean"])
    anse_energy = float(cand_mean + cfg.edit_energy_weight * edit_nll + pain)

    return {
        "finite": True,
        "nll_mean": cand_mean,
        "base_nll_mean": float(base_losses.mean()),
        "worst_domain_id": worst_eid,
        "worst_domain_ratio": worst_ratio,
        "worst_domain_pct_vs_base": worst_pct,
        "general_pct_vs_base": general_pct,
        "guard_ok": guard_ok,
        "efficacy_ok": efficacy_ok,
        "anse_energy": anse_energy,
        "per_domain": per_domain,
    }


@torch.no_grad()
def _compute_target_hidden_states(
    model: nn.Module,
    edit_seqs: torch.Tensor,
    target_ids: torch.Tensor,
    pos_idx: torch.Tensor,
    *,
    last_edit_layer: int,
    cfg: AlphaEditMEMITConfig,
) -> torch.Tensor:
    """Compute minimum-norm target post-``last_edit_layer`` hidden states via no-grad CE ascent."""
    h_after = _forward_pre_ln_states(
        model,
        edit_seqs,
        batch_size=cfg.batch_size,
        stop_after_layer=last_edit_layer,
    )
    head_w = model.head.weight.detach().cpu().to(torch.float64)
    ln_w = model.ln.weight.detach().cpu().to(torch.float64)
    n_edits = int(edit_seqs.shape[0])
    row_ix = torch.arange(n_edits)
    h_base_pos = h_after[row_ix, pos_idx].to(torch.float64)
    h_cand = h_after.clone()

    max_steps = 40
    for _ in range(max_steps):
        logits = _forward_tail_logits_at_positions(
            model,
            h_cand,
            start_layer=last_edit_layer + 1,
            pos_idx=pos_idx,
        ).to(torch.float64)
        if not bool(torch.isfinite(logits).all().item()):
            break
        probs = F.softmax(logits, dim=-1)
        all_solved = True

        for i in range(n_edits):
            tgt = int(target_ids[i].item())
            p_tgt = float(probs[i, tgt].item())
            row_l = logits[i].clone()
            row_l[tgt] = -float("inf")
            margin = float(logits[i, tgt].item() - row_l.max().item())
            if p_tgt >= cfg.target_prob and margin >= cfg.logit_margin_target:
                continue

            all_solved = False
            grad_dir = ln_w * (head_w[tgt] - (probs[i] @ head_w))
            grad_dir = grad_dir - grad_dir.mean()
            g_norm = float(grad_dir.norm().item())
            if g_norm <= 1e-12:
                grad_dir = head_w[tgt] - head_w[tgt].mean()
                g_norm = float(grad_dir.norm().item())
            if g_norm <= 1e-12:
                continue

            step_sz = 0.5 if p_tgt >= 0.5 else 1.25
            cur_pos = h_cand[i, pos_idx[i]].to(torch.float64)
            nxt_pos = cur_pos + step_sz * (grad_dir / g_norm)
            disp = nxt_pos - h_base_pos[i]
            disp_norm = float(disp.norm().item())
            if disp_norm > cfg.max_residual_norm:
                nxt_pos = h_base_pos[i] + disp * (cfg.max_residual_norm / disp_norm)
            h_cand[i, pos_idx[i]] = nxt_pos.to(dtype=h_cand.dtype)

        if all_solved:
            break

    return h_cand[row_ix, pos_idx].to(torch.float64).contiguous()


# ---------------------------------------------------------------------------
# 5. Layer-Wise SwiGLU / Sparse MoE Residual Distribution Helpers
# ---------------------------------------------------------------------------


def _resolve_block_proj_mode(mlp: nn.Module, requested_proj: str) -> str:
    """Resolve effective SwiGLU target projection ('w2' or 'w3') for ``mlp``."""
    if requested_proj != "auto":
        return requested_proj
    if isinstance(mlp, CrowdLoRAMoEMLP):
        if float(mlp.base_mlp.w1.weight.detach().abs().max().item()) > 1e-12:
            return "w2"
        return "w3"
    if isinstance(mlp, CrowdFineGrainedMoEMLP):
        return "w2"
    return "w2"


@torch.no_grad()
def _collect_layer_mlp_inputs(
    model: nn.Module,
    seqs: torch.Tensor,
    layer_idx: int,
    *,
    batch_size: int,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Propagate ``seqs`` to ``model.blocks[layer_idx]`` and collect ``mlp_in`` activations."""
    dev = model.wte.weight.device
    cos = model.cos_cached.to(dev)
    sin = model.sin_cached.to(dev)
    parts: list[torch.Tensor] = []
    for b0 in range(0, int(seqs.shape[0]), batch_size):
        b1 = min(int(seqs.shape[0]), b0 + batch_size)
        xb = compute_embeddings_batch(model, seqs[b0:b1].to(dev))
        for blk_idx in range(layer_idx):
            blk = model.blocks[blk_idx]
            xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
            xb = forward_block_post_mlp(blk, xm, mi)
        _, mlp_in = forward_block_pre_mlp(model.blocks[layer_idx], xb, cos, sin)
        if positions is not None:
            sub_pos = positions[b0:b1].to(dev)
            row_ix = torch.arange(b1 - b0, device=dev)
            parts.append(mlp_in[row_ix, sub_pos].detach().cpu())
        else:
            parts.append(mlp_in.reshape(-1, int(mlp_in.shape[-1])).detach().cpu())
    return torch.cat(parts, dim=0).contiguous()


def _swiglu_act_from_dense_or_lora(
    mlp: nn.Module,
    x_in: torch.Tensor,
    *,
    expert_idx: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ``(gate_silu, intermediate_act, w3_eff)`` for ``w2`` / ``w3`` projection solving."""
    x_cpu = x_in.detach().to(device="cpu", dtype=torch.float64)
    if isinstance(mlp, CrowdLoRAMoEMLP):
        w1_b = mlp.base_mlp.w1.weight.detach().to(device="cpu", dtype=torch.float64)
        w2_b = mlp.base_mlp.w2.weight.detach().to(device="cpu", dtype=torch.float64)
        w3_b = mlp.base_mlp.w3.weight.detach().to(device="cpu", dtype=torch.float64)
        h1 = x_cpu @ w1_b.T
        h2 = x_cpu @ w2_b.T
        w3_eff = w3_b.clone()
        if expert_idx is not None and str(expert_idx) in mlp.experts:
            exp = mlp.experts[str(expert_idx)]
            sc = float(exp.scaling)
            h1 = h1 + sc * ((x_cpu @ exp.w1_A.detach().cpu().to(torch.float64).T) @ exp.w1_B.detach().cpu().to(torch.float64).T)
            h2 = h2 + sc * ((x_cpu @ exp.w2_A.detach().cpu().to(torch.float64).T) @ exp.w2_B.detach().cpu().to(torch.float64).T)
            w3_eff = w3_eff + sc * (exp.w3_B.detach().cpu().to(torch.float64) @ exp.w3_A.detach().cpu().to(torch.float64))
        gate = F.silu(h1)
        return gate, gate * h2, w3_eff

    if isinstance(mlp, CrowdFineGrainedMoEMLP):
        idx = 0 if expert_idx is None else int(expert_idx)
        w1_e = mlp.experts_w1[idx].weight.detach().to(device="cpu", dtype=torch.float64)
        w2_e = mlp.experts_w2[idx].weight.detach().to(device="cpu", dtype=torch.float64)
        w3_e = mlp.experts_w3[idx].weight.detach().to(device="cpu", dtype=torch.float64)
        gate = F.silu(x_cpu @ w1_e.T)
        return gate, gate * (x_cpu @ w2_e.T), w3_e

    w1_d = mlp.w1.weight.detach().to(device="cpu", dtype=torch.float64)
    w2_d = mlp.w2.weight.detach().to(device="cpu", dtype=torch.float64)
    w3_d = mlp.w3.weight.detach().to(device="cpu", dtype=torch.float64)
    gate = F.silu(x_cpu @ w1_d.T)
    return gate, gate * (x_cpu @ w2_d.T), w3_d


def _map_out_residual_to_w2_residual(
    target_dy: torch.Tensor,
    gate: torch.Tensor,
    w3_eff: torch.Tensor,
    *,
    jacobian_damping: float,
) -> torch.Tensor:
    """Solve damped minimum-norm SwiGLU ``w2`` residual ``r_w2`` s.t. ``W3 (gate * r_w2) ~= target_dy``."""
    n_rows, h_dim = int(target_dy.shape[0]), int(target_dy.shape[1])
    i_dim = int(gate.shape[1])
    r_w2 = torch.zeros((n_rows, i_dim), dtype=torch.float64)
    eye_h = torch.eye(h_dim, dtype=torch.float64)
    for row in range(n_rows):
        jac = w3_eff * gate[row].unsqueeze(0)  # (H, I)
        jjt = jac @ jac.T
        jjt = 0.5 * (jjt + jjt.T)
        mean_diag = max(float(torch.diagonal(jjt).mean().item()), 1e-12)
        eps = max(float(jacobian_damping) * mean_diag, 1e-12)
        dual = torch.linalg.solve(jjt + eps * eye_h, target_dy[row].unsqueeze(1))
        r_w2[row] = (jac.T @ dual).squeeze(1)
    return r_w2.contiguous()


@torch.no_grad()
def _compute_moe_routing_weights(
    mlp: CrowdLoRAMoEMLP | CrowdFineGrainedMoEMLP,
    x_edit: torch.Tensor,
) -> torch.Tensor:
    """Return exact per-token expert routing gate tensor of shape ``(u, num_experts)``."""
    x_fp32 = x_edit.detach().to(device=mlp.router.weight.device, dtype=torch.float32)
    logits = mlp.router(x_fp32)
    res_mask = mlp.resident_mask.to(device=logits.device)
    n_tok = int(x_fp32.shape[0])
    n_exp = int(mlp.num_experts)

    if mlp.routing_mode == "sigma_topk":
        sig_probs = torch.sigmoid(logits)
        top_vals, top_idx = torch.topk(sig_probs, k=mlp.top_k, dim=-1)
        skipped = ~res_mask[top_idx]
        if isinstance(mlp, CrowdFineGrainedMoEMLP):
            scale = (2.0 * float(n_exp)) / float(mlp.top_k)
            gates = torch.where(skipped, torch.zeros_like(top_vals), top_vals * scale)
        else:
            gates = torch.where(skipped, torch.zeros_like(top_vals), top_vals)
    else:
        probs = F.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(probs, k=mlp.top_k, dim=-1)
        skipped = ~res_mask[top_idx]
        renorm = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if isinstance(mlp, CrowdFineGrainedMoEMLP):
            gates = torch.where(skipped, torch.zeros_like(renorm), renorm * float(n_exp))
        else:
            gates = torch.where(skipped, torch.zeros_like(renorm), renorm)

    gate_mat = torch.zeros((n_tok, n_exp), dtype=torch.float64)
    for e_idx in range(n_exp):
        if not bool(res_mask[e_idx].item()):
            continue
        hit = top_idx == e_idx
        if bool(hit.any().item()):
            gate_mat[:, e_idx] = (gates * hit.to(torch.float32)).sum(dim=-1).detach().cpu().to(torch.float64)
    return gate_mat


def _build_projection_key_tensors(
    mlp: nn.Module,
    proj_mode: str,
    k_edit_x: torch.Tensor,
    k_pres_x: torch.Tensor,
    k_bg_x: torch.Tensor,
    *,
    expert_idx: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(k_edit, k_pres, k_bg)`` in the exact input space of ``proj_mode``."""
    if proj_mode == "w2":
        return (
            k_edit_x.detach().cpu().to(torch.float64),
            k_pres_x.detach().cpu().to(torch.float64),
            k_bg_x.detach().cpu().to(torch.float64),
        )
    _, act_edit, _ = _swiglu_act_from_dense_or_lora(mlp, k_edit_x, expert_idx=expert_idx)
    _, act_pres, _ = _swiglu_act_from_dense_or_lora(mlp, k_pres_x, expert_idx=expert_idx)
    _, act_bg, _ = _swiglu_act_from_dense_or_lora(mlp, k_bg_x, expert_idx=expert_idx)
    return act_edit, act_pres, act_bg


@torch.no_grad()
def _edit_single_layer_inplace(
    mlp: nn.Module,
    l_idx: int,
    r_layer: torch.Tensor,
    k_edit_x: torch.Tensor,
    k_pres_x: torch.Tensor,
    k_bg_x: torch.Tensor,
    *,
    target_experts: Sequence[int] | None,
    cfg: AlphaEditMEMITConfig,
) -> dict[str, Any]:
    """Apply AlphaEdit-MEMIT update for one layer across MoE experts or dense MLP."""
    proj_mode = _resolve_block_proj_mode(mlp, cfg.target_proj)
    expert_reports: list[dict[str, Any]] = []
    max_delta_fro = 0.0
    total_delta_sq = 0.0
    max_pres_leakage = 0.0

    def _solve_and_fold(
        target_mod: nn.Module,
        label: str,
        k_ed: torch.Tensor,
        k_pr: torch.Tensor,
        k_bg: torch.Tensor,
        target_dy: torch.Tensor,
        expert_id: int | None,
    ) -> None:
        nonlocal max_delta_fro, total_delta_sq, max_pres_leakage
        d_in = int(k_pr.shape[1])
        p_null = (
            compute_nullspace_projector_from_keys(k_pr, sv_threshold=cfg.sv_threshold)
            if cfg.use_nullspace_projection
            else torch.eye(d_in, dtype=torch.float64)
        )
        c_cov = (k_bg.T @ k_bg) / float(max(1, int(k_bg.shape[0])))

        if proj_mode == "w2":
            gate, _, w3_eff = _swiglu_act_from_dense_or_lora(mlp, k_ed, expert_idx=expert_id)
            r_target = _map_out_residual_to_w2_residual(
                target_dy,
                gate,
                w3_eff,
                jacobian_damping=cfg.jacobian_damping,
            )
        else:
            r_target = target_dy

        delta_w = solve_alphaedit_memit_layer_delta(
            k_ed,
            r_target,
            p_null,
            c_cov,
            reg_lambda=cfg.reg_lambda,
            keys_as_columns=False,
        )
        dw_fro = float(delta_w.norm().item())
        leak = float((delta_w @ k_pr.T).abs().max().item())
        max_delta_fro = max(max_delta_fro, dw_fro)
        total_delta_sq += dw_fro * dw_fro
        max_pres_leakage = max(max_pres_leakage, leak)

        fold_info = fold_delta_into_swiglu_lora_w2_inplace(
            target_mod,
            delta_w,
            weight_attr=proj_mode,
            p_null=p_null if cfg.use_nullspace_projection else None,
            lora_fold_mode=cfg.lora_fold_mode,
        )
        fold_info.update(
            {
                "target": label,
                "delta_w_fro": dw_fro,
                "max_preserved_key_drift": leak,
                "n_edit_tokens": int(k_ed.shape[0]),
            }
        )
        expert_reports.append(fold_info)

    if isinstance(mlp, (CrowdLoRAMoEMLP, CrowdFineGrainedMoEMLP)):
        gate_mat = _compute_moe_routing_weights(mlp, k_edit_x)
        if isinstance(mlp, CrowdLoRAMoEMLP):
            resident_ids = sorted(int(k) for k in mlp.experts.keys())
            target_lookup: dict[int, tuple[nn.Module, str]] = {
                e: (mlp.experts[str(e)], f"expert_{e}.{proj_mode}") for e in resident_ids
            }
        else:
            resident_ids = [e for e in range(mlp.num_experts) if bool(mlp.resident_mask[e].item())]
            mod_list = mlp.experts_w2 if proj_mode == "w2" else mlp.experts_w3
            target_lookup = {e: (mod_list[e], f"fg_expert_{e}.{proj_mode}") for e in resident_ids}

        chosen_ids = (
            resident_ids
            if target_experts is None
            else [e for e in sorted(set(target_experts)) if e in target_lookup]
        )
        if not chosen_ids:
            raise ValueError(f"Layer {l_idx} has no resident experts matching expert_indices={target_experts}")

        sub_gates = gate_mat[:, chosen_ids].clone()
        sens_mat = torch.zeros_like(sub_gates)
        for col_idx, e_idx in enumerate(chosen_ids):
            if bool((sub_gates[:, col_idx] > 1e-8).any().item()):
                gate_e, act_e, w3_e = _swiglu_act_from_dense_or_lora(mlp, k_edit_x, expert_idx=e_idx)
                for row_i in range(int(k_edit_x.shape[0])):
                    if float(sub_gates[row_i, col_idx].item()) <= 1e-8:
                        continue
                    r_vec = r_layer[row_i]
                    if proj_mode == "w2":
                        jac_t_r = (w3_e.T @ r_vec) * gate_e[row_i]
                        sens_mat[row_i, col_idx] = float(jac_t_r.norm().item())
                    else:
                        act_norm = float(act_e[row_i].norm().item())
                        out_align = float((w3_e.T @ r_vec).norm().item())
                        sens_mat[row_i, col_idx] = act_norm * out_align

        row_max_sens = sens_mat.max(dim=1, keepdim=True).values
        feasible_mask = (sens_mat > 1e-8) & (sens_mat >= 0.05 * row_max_sens)
        sub_gates = torch.where(feasible_mask, sub_gates, torch.zeros_like(sub_gates))
        denom_sq = (sub_gates * sub_gates).sum(dim=1)
        for col_idx, e_idx in enumerate(chosen_ids):
            g_col = sub_gates[:, col_idx]
            active_rows = torch.nonzero((g_col > 1e-8) & (denom_sq > 1e-12), as_tuple=False).squeeze(-1)
            if active_rows.numel() == 0:
                continue
            scale_vec = (g_col[active_rows] / denom_sq[active_rows]).unsqueeze(1)
            target_dy_e = r_layer[active_rows] * scale_vec
            k_ed_e = k_edit_x[active_rows]
            k_ed_p, k_pr_p, k_bg_p = _build_projection_key_tensors(
                mlp,
                proj_mode,
                k_ed_e,
                k_pres_x,
                k_bg_x,
                expert_idx=e_idx,
            )
            mod_e, label_e = target_lookup[e_idx]
            _solve_and_fold(
                mod_e,
                label_e,
                k_ed_p,
                k_pr_p,
                k_bg_p,
                target_dy_e,
                e_idx,
            )
    else:
        k_ed_p, k_pr_p, k_bg_p = _build_projection_key_tensors(
            mlp,
            proj_mode,
            k_edit_x,
            k_pres_x,
            k_bg_x,
            expert_idx=None,
        )
        _solve_and_fold(
            mlp,
            f"dense.{proj_mode}",
            k_ed_p,
            k_pr_p,
            k_bg_p,
            r_layer,
            None,
        )

    return {
        "layer": l_idx,
        "proj_mode": proj_mode,
        "layer_residual_fro": float(r_layer.norm().item()),
        "max_delta_fro": max_delta_fro,
        "layer_delta_fro_sq": total_delta_sq,
        "max_preserved_key_drift": max_pres_leakage,
        "experts": expert_reports,
    }


def _resolve_memit_config(
    config: AlphaEditMEMITConfig | None,
    *,
    sv_threshold: float | None,
    reg_lambda: float | None,
    residual_damping: float | None,
    use_nullspace_projection: bool | None,
    target_proj: str | None,
    domain_guard_max_ratio: float | None,
    min_edit_efficacy_ratio: float | None,
    require_strict_energy_drop: bool | None,
) -> AlphaEditMEMITConfig:
    """Merge optional keyword overrides into a validated ``AlphaEditMEMITConfig``."""
    if config is not None and not isinstance(config, AlphaEditMEMITConfig):
        raise TypeError(f"config must be AlphaEditMEMITConfig or None, got {type(config).__name__}")
    base = config or AlphaEditMEMITConfig()
    return AlphaEditMEMITConfig(
        sv_threshold=base.sv_threshold if sv_threshold is None else _validate_nonneg_finite(sv_threshold, "sv_threshold"),
        reg_lambda=base.reg_lambda if reg_lambda is None else _validate_positive_finite(reg_lambda, "reg_lambda"),
        target_prob=base.target_prob,
        logit_margin_target=base.logit_margin_target,
        residual_damping=(
            base.residual_damping
            if residual_damping is None
            else _validate_unit_interval(residual_damping, "residual_damping", allow_zero=False, allow_one=True)
        ),
        jacobian_damping=base.jacobian_damping,
        max_residual_norm=base.max_residual_norm,
        use_nullspace_projection=(
            base.use_nullspace_projection
            if use_nullspace_projection is None
            else _validate_bool(use_nullspace_projection, "use_nullspace_projection")
        ),
        target_proj=base.target_proj if target_proj is None else str(target_proj),
        lora_fold_mode=base.lora_fold_mode,
        domain_guard_max_ratio=(
            base.domain_guard_max_ratio
            if domain_guard_max_ratio is None
            else _validate_positive_finite(domain_guard_max_ratio, "domain_guard_max_ratio")
        ),
        anse_safe_general_ratio=base.anse_safe_general_ratio,
        maximum_pain_energy=base.maximum_pain_energy,
        soft_margin_penalty=base.soft_margin_penalty,
        edit_energy_weight=base.edit_energy_weight,
        min_edit_efficacy_ratio=(
            base.min_edit_efficacy_ratio
            if min_edit_efficacy_ratio is None
            else _validate_unit_interval(min_edit_efficacy_ratio, "min_edit_efficacy_ratio", allow_zero=True, allow_one=True)
        ),
        require_strict_energy_drop=(
            base.require_strict_energy_drop
            if require_strict_energy_drop is None
            else _validate_bool(require_strict_energy_drop, "require_strict_energy_drop")
        ),
        general_prefix=base.general_prefix,
        score_from=base.score_from,
        batch_size=base.batch_size,
    )


@torch.no_grad()
def distribute_multilayer_memit_moe_edits(
    model: nn.Module,
    edit_seqs: torch.Tensor | Sequence[Sequence[int]],
    target_token_ids: Sequence[int] | torch.Tensor,
    preserve_chunks: Sequence[Mapping[str, Any]],
    *,
    background_seqs: torch.Tensor | Sequence[Sequence[int]] | None = None,
    layer_indices: Sequence[int] | None = None,
    expert_indices: Sequence[int] | None = None,
    edit_positions: Sequence[int] | torch.Tensor | None = None,
    preserve_key_seqs: torch.Tensor | Sequence[Sequence[int]] | None = None,
    config: AlphaEditMEMITConfig | None = None,
    sv_threshold: float | None = None,
    reg_lambda: float | None = None,
    residual_damping: float | None = None,
    use_nullspace_projection: bool | None = None,
    target_proj: str | None = None,
    domain_guard_max_ratio: float | None = None,
    min_edit_efficacy_ratio: float | None = None,
    require_strict_energy_drop: bool | None = None,
) -> dict[str, Any]:
    """Distribute closed-form AlphaEdit-MEMIT skill/fact edits across selected MoE layers.

    Spreads the total target hidden-state residuals ``v* - h(l_last)`` sequentially across
    ``layer_indices`` inside the SVD null-space of preserved general-prose and specialist
    key activations, folding updates into rank-``r`` SwiGLU LoRA ``w2`` (or ``w3``/dense)
    weights without inflating rank. Commits only when the ANSE ``<= +1.0%`` general-domain
    guard and efficacy/energy gates pass; otherwise restores the exact pre-edit model state
    atomically bit-for-bit and returns a canonical SHA-256 provenance receipt.
    """
    _validate_model_compatible(model)
    cfg = _resolve_memit_config(
        config,
        sv_threshold=sv_threshold,
        reg_lambda=reg_lambda,
        residual_damping=residual_damping,
        use_nullspace_projection=use_nullspace_projection,
        target_proj=target_proj,
        domain_guard_max_ratio=domain_guard_max_ratio,
        min_edit_efficacy_ratio=min_edit_efficacy_ratio,
        require_strict_energy_drop=require_strict_energy_drop,
    )

    edit_t = _normalize_token_id_batch(edit_seqs, "edit_seqs", min_len=2)
    vocab_size = int(model.head.weight.shape[0])
    if bool((edit_t >= vocab_size).any().item()):
        raise ValueError(f"edit_seqs token IDs must lie in [0, {vocab_size})")
    n_edits, seq_len = int(edit_t.shape[0]), int(edit_t.shape[1])

    target_t = _validate_target_token_ids(target_token_ids, n_edits, vocab_size)
    pos_t = _validate_edit_positions(edit_positions, n_edits, seq_len)
    ids_guard, expert_ids = _validate_preserve_chunks(
        preserve_chunks,
        general_prefix=cfg.general_prefix,
        score_from=cfg.score_from,
    )
    if bool((ids_guard >= vocab_size).any().item()):
        raise ValueError(f"preserve_chunks token IDs must lie in [0, {vocab_size})")

    if preserve_key_seqs is not None:
        pres_key_t = _normalize_token_id_batch(preserve_key_seqs, "preserve_key_seqs", min_len=2)
        if bool((pres_key_t >= vocab_size).any().item()):
            raise ValueError(f"preserve_key_seqs token IDs must lie in [0, {vocab_size})")
    else:
        pres_key_t = ids_guard

    if background_seqs is not None:
        bg_t = _normalize_token_id_batch(background_seqs, "background_seqs", min_len=2)
        if bool((bg_t >= vocab_size).any().item()):
            raise ValueError(f"background_seqs token IDs must lie in [0, {vocab_size})")
    else:
        bg_t = pres_key_t

    n_blocks = len(model.blocks)
    if layer_indices is None:
        start_l = max(0, n_blocks - min(3, n_blocks))
        target_layers = list(range(start_l, n_blocks))
    else:
        if isinstance(layer_indices, (str, bytes)) or not isinstance(layer_indices, Sequence) or not layer_indices:
            raise ValueError("layer_indices must be a non-empty sequence of integer layer indices")
        target_layers = sorted({_validate_nonneg_int(i, "layer_index") for i in layer_indices})
        if any(i >= n_blocks for i in target_layers):
            raise ValueError(f"layer_indices must lie in [0, {n_blocks})")

    target_experts: list[int] | None = None
    if expert_indices is not None:
        if isinstance(expert_indices, (str, bytes)) or not isinstance(expert_indices, Sequence) or not expert_indices:
            raise ValueError("expert_indices must be a non-empty sequence of expert indices or None")
        target_experts = sorted({_validate_nonneg_int(e, "expert_index") for e in expert_indices})

    was_training = model.training
    model.eval()
    try:
        parent_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in model.state_dict().items()}
        parent_sha = compute_state_sha256(parent_snapshot)

        base_guard_losses = _eval_guard_nlls(
            model,
            ids_guard,
            score_from=cfg.score_from,
            batch_size=cfg.batch_size,
        )
        parent_edit_metrics = _eval_edit_metrics(
            model,
            edit_t,
            target_t,
            pos_t,
            batch_size=cfg.batch_size,
        )
        parent_eval = _summarize_guard_and_energy(
            base_guard_losses,
            base_guard_losses,
            expert_ids,
            parent_edit_metrics,
            cfg=cfg,
            is_parent=True,
        )

        last_layer = target_layers[-1]
        v_star = _compute_target_hidden_states(
            model,
            edit_t,
            target_t,
            pos_t,
            last_edit_layer=last_layer,
            cfg=cfg,
        )

        row_ix = torch.arange(n_edits)
        layer_reports: list[dict[str, Any]] = []
        n_edit_layers = len(target_layers)

        for step_j, l_idx in enumerate(target_layers):
            layers_remaining = n_edit_layers - step_j
            h_cur_last = _forward_pre_ln_states(
                model,
                edit_t,
                batch_size=cfg.batch_size,
                stop_after_layer=last_layer,
            )[row_ix, pos_t].to(torch.float64)
            r_rem = v_star - h_cur_last
            step_scale = 1.0 if step_j == n_edit_layers - 1 else cfg.residual_damping
            r_layer = (step_scale / float(layers_remaining)) * r_rem

            k_edit_x = _collect_layer_mlp_inputs(
                model,
                edit_t,
                l_idx,
                batch_size=cfg.batch_size,
                positions=pos_t,
            )
            k_pres_x = _collect_layer_mlp_inputs(
                model,
                pres_key_t,
                l_idx,
                batch_size=cfg.batch_size,
                positions=None,
            )
            k_bg_x = (
                k_pres_x
                if bg_t is pres_key_t
                else _collect_layer_mlp_inputs(
                    model,
                    bg_t,
                    l_idx,
                    batch_size=cfg.batch_size,
                    positions=None,
                )
            )
            l_report = _edit_single_layer_inplace(
                model.blocks[l_idx].mlp,
                l_idx,
                r_layer,
                k_edit_x,
                k_pres_x,
                k_bg_x,
                target_experts=target_experts,
                cfg=cfg,
            )
            layer_reports.append(l_report)

        candidate_sha = compute_state_sha256(model.state_dict())
        cand_guard_losses = _eval_guard_nlls(
            model,
            ids_guard,
            score_from=cfg.score_from,
            batch_size=cfg.batch_size,
        )
        cand_edit_metrics = _eval_edit_metrics(
            model,
            edit_t,
            target_t,
            pos_t,
            batch_size=cfg.batch_size,
        )
        cand_eval = _summarize_guard_and_energy(
            base_guard_losses,
            cand_guard_losses,
            expert_ids,
            cand_edit_metrics,
            cfg=cfg,
            is_parent=False,
        )

        parent_e = float(parent_eval["anse_energy"])
        cand_e = float(cand_eval["anse_energy"])
        delta_e = cand_e - parent_e
        energy_ok = (delta_e < 0.0) if cfg.require_strict_energy_drop else (delta_e <= 0.0)

        if candidate_sha == parent_sha:
            status = "FIXED_POINT_INFEASIBLE" if not bool(cand_eval["efficacy_ok"]) else "FIXED_POINT_UNCHANGED"
            accepted = False
            rolled_back = False
        elif bool(cand_eval["guard_ok"]) and bool(cand_eval["efficacy_ok"]) and energy_ok:
            status = "ACCEPTED"
            accepted = True
            rolled_back = False
        else:
            if not bool(cand_eval["guard_ok"]):
                status = "ROLLED_BACK_GUARD_VIOLATION"
            elif not bool(cand_eval["efficacy_ok"]):
                status = "ROLLED_BACK_INFEASIBLE_EDIT"
            else:
                status = "ROLLED_BACK_ENERGY_NON_IMPROVING"
            accepted = False
            rolled_back = True
            cur_sd = model.state_dict()
            model.load_state_dict(
                {k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in parent_snapshot.items()}
            )

        final_sha = compute_state_sha256(model.state_dict())
        if rolled_back and final_sha != parent_sha:
            raise RuntimeError(
                f"Atomic bitwise rollback failed: {final_sha[:12]} != {parent_sha[:12]}"
            )
    finally:
        model.train(was_training)

    max_layer_delta_fro = max((float(r["max_delta_fro"]) for r in layer_reports), default=0.0)
    total_delta_fro_sq = sum(float(r["layer_delta_fro_sq"]) for r in layer_reports)
    max_pres_drift = max((float(r["max_preserved_key_drift"]) for r in layer_reports), default=0.0)

    receipt: dict[str, Any] = {
        "object": MEMIT_EDITOR_RECEIPT_OBJECT,
        "schema": MEMIT_EDITOR_SCHEMA,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "status": status,
        "guard_ok": bool(cand_eval["guard_ok"]),
        "efficacy_ok": bool(cand_eval["efficacy_ok"]),
        "n_edits": n_edits,
        "layer_indices": target_layers,
        "expert_indices": target_experts,
        "nll_mean": float(cand_eval["nll_mean"]) if accepted else float(parent_eval["nll_mean"]),
        "candidate_nll_mean": cand_eval["nll_mean"],
        "base_nll_mean": float(parent_eval["base_nll_mean"]),
        "pre_edit_nll_mean": float(parent_edit_metrics["edit_nll_mean"]),
        "post_edit_nll_mean": (
            float(cand_edit_metrics["edit_nll_mean"])
            if accepted
            else float(parent_edit_metrics["edit_nll_mean"])
        ),
        "candidate_edit_nll_mean": float(cand_edit_metrics["edit_nll_mean"]),
        "pre_edit_efficacy_ratio": float(parent_edit_metrics["edit_efficacy_ratio"]),
        "edit_efficacy_ratio": (
            float(cand_edit_metrics["edit_efficacy_ratio"])
            if accepted
            else float(parent_edit_metrics["edit_efficacy_ratio"])
        ),
        "candidate_edit_efficacy_ratio": float(cand_edit_metrics["edit_efficacy_ratio"]),
        "mean_target_prob": (
            float(cand_edit_metrics["mean_target_prob"])
            if accepted
            else float(parent_edit_metrics["mean_target_prob"])
        ),
        "candidate_mean_target_prob": float(cand_edit_metrics["mean_target_prob"]),
        "anse_energy": float(cand_e),
        "parent_anse_energy": float(parent_e),
        "final_anse_energy": float(cand_e if accepted else parent_e),
        "delta_e": float(delta_e),
        "worst_domain_id": str(cand_eval["worst_domain_id"]),
        "worst_domain_ratio": cand_eval["worst_domain_ratio"],
        "worst_domain_pct_vs_base": cand_eval["worst_domain_pct_vs_base"],
        "general_pct_vs_base": cand_eval["general_pct_vs_base"],
        "per_domain": cand_eval["per_domain"],
        "max_layer_delta_fro": float(max_layer_delta_fro),
        "total_delta_fro_sq": float(total_delta_fro_sq),
        "max_preserved_key_drift": float(max_pres_drift),
        "layer_reports": layer_reports,
        "parent_state_sha256": parent_sha,
        "candidate_state_sha256": candidate_sha,
        "final_state_sha256": final_sha,
        "config": asdict(cfg),
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_GENERAL_RATIO",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_EDIT_ENERGY_WEIGHT",
    "DEFAULT_JACOBIAN_DAMPING",
    "DEFAULT_LOGIT_MARGIN_TARGET",
    "DEFAULT_MAX_RESIDUAL_NORM",
    "DEFAULT_MIN_EDIT_EFFICACY_RATIO",
    "DEFAULT_REG_LAMBDA",
    "DEFAULT_RESIDUAL_DAMPING",
    "DEFAULT_SOFT_MARGIN_PENALTY",
    "DEFAULT_SV_THRESHOLD",
    "DEFAULT_TARGET_PROB",
    "GENERAL_EXPERT_PREFIX",
    "MEMIT_EDITOR_RECEIPT_OBJECT",
    "MEMIT_EDITOR_SCHEMA",
    "VALID_LORA_FOLD_MODES",
    "VALID_TARGET_PROJECTIONS",
    "AlphaEditMEMITConfig",
    "canonical_json_bytes",
    "compute_nullspace_projector_from_keys",
    "compute_payload_sha256",
    "compute_state_sha256",
    "distribute_multilayer_memit_moe_edits",
    "fold_delta_into_lora_pair",
    "fold_delta_into_swiglu_lora_w2_inplace",
    "solve_alphaedit_memit_layer_delta",
    "verify_memit_receipt_provenance",
]

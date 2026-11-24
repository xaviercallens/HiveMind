# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Lifelong Sequential Null-Space Basis Tracker for Non-Interfering Multi-Round WSP Editing.

Extends Fang et al. (ICLR 2025) AlphaEdit null-space constrained editing into a stateful
multi-round Weight-Space Programming (WSP) engine for open-weight Sparse MoE and FFN/LoRA
blocks (``CrowdV1WithEngram``, ``CrowdLoRAMoEMLP``, ``CrowdFineGrainedMoEMLP``,
``CrowdRuntimeMLP``, ``CrowdLoRAExpertAdapter``, and ``nn.Linear``):

1. ``SequentialNullspaceBasis``:
   Stateful CPU ``float64`` orthonormal basis tracker maintaining protected general-prose
   directions ``U_0`` plus incrementally appended orthonormalized key directions ``U_{1..t}``
   from prior accepted WSP rounds with explicit free-dimension monitoring
   (``remaining_null_rank``) and double-Gram-Schmidt + QR orthogonality preservation.
2. ``solve_sequential_guarded_wsp_update(k_new, v_target_delta, basis, *, ridge_lambda=1e-2, min_null_rank=4)``:
   Solves the closed-form minimum-norm update strictly inside ``P_t = I - U_t U_t^T`` so
   ``Delta W U_t == 0``, and refuses edits cleanly before null-space exhaustion or rank
   collapse.
3. ``commit_sequential_wsp_round_inplace(...)``:
   Applies round ``t+1`` in-place and expands ``U_{t+1}`` if and only if ANSE ``Delta E < 0``
   and both general-prose and all historical round ``1..t`` skill guards remain ``<= +1.0%``,
   with atomic bitwise rollback of both model weights and basis ``U_t`` on failure and
   canonical SHA-256 provenance receipts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdFineGrainedMoEMLP,
    CrowdLoRAExpertAdapter,
    CrowdLoRAMoEMLP,
)
from hivemind.swarm.lora_svd_mode_surgery import decompose_lora_qr_svd

SCHEMA_VERSION: str = "hivemind.sequential_nullspace_memory/1"
RECEIPT_OBJECT: str = "hivemind.sequential_nullspace_memory_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_SV_THRESHOLD: float = 1e-2
DEFAULT_RIDGE_LAMBDA: float = 1e-2
DEFAULT_MIN_NULL_RANK: int = 4
DEFAULT_ORTHO_TOL: float = 1e-10
DEFAULT_MIN_PROJECTED_ENERGY_RATIO: float = 1e-4
DEFAULT_DOMAIN_GUARD_MAX_REL: float = 0.01
DEFAULT_ANSE_SAFE_GENERAL_RATIO: float = 1.0085
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_MIN_RETENTION_RATIO: float = 0.95
DEFAULT_MIN_NEW_SKILL_EFFICACY: float = 0.75
DEFAULT_MAX_BASIS_LEAKAGE: float = 1e-4
VALID_LORA_FOLD_MODES: frozenset[str] = frozenset({"auto", "zero_slot", "update_a", "qr_svd"})
VALID_WEIGHT_ATTRS: frozenset[str] = frozenset({"w1", "w2", "w3", "weight"})

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class NullspaceExhaustedError(RuntimeError):
    """Raised when remaining null-space capacity is insufficient for a safe non-interfering edit."""


# ---------------------------------------------------------------------------
# 1. Validation & Canonical SHA-256 Provenance Helpers
# ---------------------------------------------------------------------------


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_nonneg_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a non-negative integer (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a strictly positive integer (excluding booleans)."""
    out = _validate_nonneg_int(value, name)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return out


def _validate_finite_float(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number (excluding booleans)."""
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


def _validate_unit_interval(value: Any, name: str) -> float:
    """Validate that ``value`` is finite and lies inside ``[0.0, 1.0]``."""
    out = _validate_nonneg_finite(value, name)
    if out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    return out


def _validate_finite_2d(tensor: Any, name: str) -> torch.Tensor:
    """Validate a non-empty 2D floating-point tensor and return a CPU ``float64`` copy."""
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


def _normalize_key_collection(
    keys: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor],
    name: str,
    *,
    expected_dim: int | None = None,
    keys_as_columns: bool | None = None,
) -> torch.Tensor:
    """Flatten 2D/3D tensor or collection of key activations into row-major CPU ``float64`` ``(N, D)``."""
    if isinstance(keys, torch.Tensor):
        items: list[tuple[str, torch.Tensor]] = [(name, keys)]
    elif isinstance(keys, Mapping):
        if not keys:
            raise ValueError(f"{name} mapping must not be empty")
        items = [(f"{name}[{k!r}]", v) for k, v in sorted(keys.items(), key=lambda kv: str(kv[0]))]
    elif isinstance(keys, Sequence) and not isinstance(keys, (str, bytes, bytearray)):
        if not keys:
            raise ValueError(f"{name} sequence must not be empty")
        items = [(f"{name}[{idx}]", v) for idx, v in enumerate(keys)]
    else:
        raise TypeError(
            f"{name} must be a torch.Tensor, Sequence[Tensor], or Mapping[str, Tensor]; got {type(keys).__name__}"
        )

    col_flag = _validate_bool(keys_as_columns, "keys_as_columns") if keys_as_columns is not None else False
    parts: list[torch.Tensor] = []
    width = expected_dim
    for label, tensor in items:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{label} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.ndim not in (2, 3) or any(dim == 0 for dim in tensor.shape):
            raise ValueError(f"{label} must be a non-empty 2D or 3D tensor, got shape {tuple(tensor.shape)}")
        if not tensor.is_floating_point():
            raise TypeError(f"{label} must have a floating-point dtype, got {tensor.dtype}")
        mat = tensor.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(mat).all().item()):
            raise ValueError(f"{label} must contain only finite values")
        if mat.ndim == 2 and (col_flag or (keys_as_columns is None and width is not None and mat.shape[1] != width and mat.shape[0] == width)):
            flat = mat.T.contiguous()
        else:
            flat = mat.reshape(-1, int(mat.shape[-1])).contiguous()
        cur_w = int(flat.shape[1])
        if width is None:
            width = cur_w
        elif cur_w != width:
            raise ValueError(f"{label} width {cur_w} must match ambient_dim={width}")
        parts.append(flat)
    return torch.cat(parts, dim=0).contiguous()


def _orient_edit_pair(
    k_new: torch.Tensor,
    v_target_delta: torch.Tensor,
    ambient_dim: int,
    *,
    keys_as_columns: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Orient ``(k_new, v_target_delta)`` into column-major ``((d_in, n), (d_out, n))``."""
    k64 = _validate_finite_2d(k_new, "k_new")
    v64 = _validate_finite_2d(v_target_delta, "v_target_delta")

    if keys_as_columns is not None:
        col_mode = _validate_bool(keys_as_columns, "keys_as_columns")
    elif int(k64.shape[0]) == ambient_dim and int(k64.shape[1]) != ambient_dim:
        col_mode = True
    elif int(k64.shape[1]) == ambient_dim and int(k64.shape[0]) != ambient_dim:
        col_mode = False
    elif k64.shape == (ambient_dim, ambient_dim):
        col_mode = int(v64.shape[1]) == ambient_dim and int(v64.shape[0]) != ambient_dim
    else:
        raise ValueError(
            f"k_new shape {tuple(k64.shape)} is incompatible with basis ambient_dim={ambient_dim}"
        )

    if col_mode:
        if int(k64.shape[0]) != ambient_dim:
            raise ValueError(f"k_new row dimension {k64.shape[0]} must match ambient_dim={ambient_dim}")
        if int(v64.shape[1]) != int(k64.shape[1]):
            raise ValueError(
                f"v_target_delta column count {v64.shape[1]} must match k_new edit count {k64.shape[1]}"
            )
        return k64.contiguous(), v64.contiguous()

    if int(k64.shape[1]) != ambient_dim:
        raise ValueError(f"k_new feature width {k64.shape[1]} must match ambient_dim={ambient_dim}")
    if int(v64.shape[0]) != int(k64.shape[0]):
        raise ValueError(
            f"v_target_delta batch count {v64.shape[0]} must match k_new batch count {k64.shape[0]}"
        )
    return k64.T.contiguous(), v64.T.contiguous()


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically with strict finite-number enforcement."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def compute_payload_sha256(payload: Any) -> str:
    """Return lowercase SHA-256 hex digest of ``canonical_json_bytes(payload)``."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_state_sha256(state_or_module: nn.Module | Mapping[str, torch.Tensor]) -> str:
    """Compute a deterministic bitwise SHA-256 digest over a module or ``state_dict`` mapping."""
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
            raise ValueError(f"state_dict keys must be non-empty strings, got {key!r}")
        val = state[key]
        if not isinstance(val, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] must be a torch.Tensor, got {type(val).__name__}")
        t_cpu = val.detach().to(device="cpu").contiguous()
        if t_cpu.is_floating_point() and not bool(torch.isfinite(t_cpu).all().item()):
            raise ValueError(f"state_dict[{key!r}] contains non-finite values")
        digest.update(f"{key}:{t_cpu.dtype}:{tuple(t_cpu.shape)}:".encode())
        if t_cpu.numel() > 0:
            elem_bytes = t_cpu.element_size()
            view_dt = (
                torch.uint8
                if elem_bytes == 1
                else (torch.int16 if elem_bytes == 2 else (torch.int32 if elem_bytes == 4 else torch.int64))
            )
            digest.update(t_cpu.view(view_dt).reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_sequential_wsp_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a Mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != RECEIPT_OBJECT or receipt.get("schema") != SCHEMA_VERSION:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


# ---------------------------------------------------------------------------
# 2. Configuration, Records & Lifelong Sequential Null-Space Basis Tracker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequentialNullspaceConfig:
    """Validated hyperparameters governing sequential null-space WSP skill injection."""

    sv_threshold: float = DEFAULT_SV_THRESHOLD
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA
    min_null_rank: int = DEFAULT_MIN_NULL_RANK
    min_projected_energy_ratio: float = DEFAULT_MIN_PROJECTED_ENERGY_RATIO
    domain_guard_max_rel: float = DEFAULT_DOMAIN_GUARD_MAX_REL
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_GENERAL_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    min_retention_ratio: float = DEFAULT_MIN_RETENTION_RATIO
    min_new_skill_efficacy: float = DEFAULT_MIN_NEW_SKILL_EFFICACY
    max_basis_leakage: float = DEFAULT_MAX_BASIS_LEAKAGE
    require_strict_energy_drop: bool = True
    lora_fold_mode: str = "auto"
    general_prefix: str = GENERAL_EXPERT_PREFIX

    def __post_init__(self) -> None:
        _validate_nonneg_finite(self.sv_threshold, "sv_threshold")
        _validate_positive_finite(self.ridge_lambda, "ridge_lambda")
        _validate_nonneg_int(self.min_null_rank, "min_null_rank")
        _validate_unit_interval(self.min_projected_energy_ratio, "min_projected_energy_ratio")
        _validate_nonneg_finite(self.domain_guard_max_rel, "domain_guard_max_rel")
        _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_unit_interval(self.min_retention_ratio, "min_retention_ratio")
        _validate_unit_interval(self.min_new_skill_efficacy, "min_new_skill_efficacy")
        _validate_positive_finite(self.max_basis_leakage, "max_basis_leakage")
        _validate_bool(self.require_strict_energy_drop, "require_strict_energy_drop")
        if self.lora_fold_mode not in VALID_LORA_FOLD_MODES:
            raise ValueError(
                f"lora_fold_mode must be one of {sorted(VALID_LORA_FOLD_MODES)}, got {self.lora_fold_mode!r}"
            )
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")


@dataclass(frozen=True)
class SequentialRoundRecord:
    """Immutable provenance record describing one basis segment in ``U_t``."""

    round_index: int
    round_id: str
    col_start: int
    col_end: int
    added_rank: int
    key_count: int
    keys_sha256: str
    pre_remaining_null_rank: int
    post_remaining_null_rank: int
    baseline_nll: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize record to a JSON-compatible dictionary."""
        return asdict(self)


def _tensor_bytes_sha256(tensor: torch.Tensor) -> str:
    """Compute SHA-256 digest of a contiguous CPU float64 tensor."""
    t_cpu = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    h = hashlib.sha256()
    h.update(f"{tuple(t_cpu.shape)}:".encode())
    if t_cpu.numel() > 0:
        h.update(t_cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _orthonormalize_rows_svd(
    rows_2d: torch.Tensor,
    *,
    sv_threshold: float,
    max_rank: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract orthonormal column basis ``Q (d, r)`` and singular values from ``rows_2d (n, d)``."""
    d_in = int(rows_2d.shape[1])
    _, s_vals, vh = torch.linalg.svd(rows_2d, full_matrices=False)
    if s_vals.numel() == 0:
        return torch.empty((d_in, 0), dtype=torch.float64), s_vals
    top_sv = float(s_vals[0].item())
    eff_tol = max(float(sv_threshold), 1e-10 * max(top_sv, 1.0))
    n_sig = int((s_vals > eff_tol).sum().item())
    if max_rank is not None:
        n_sig = min(n_sig, max_rank)
    if n_sig == 0:
        return torch.empty((d_in, 0), dtype=torch.float64), s_vals
    q_cols = vh[:n_sig, :].T.contiguous()
    return q_cols, s_vals


def _qr_reorthogonalize_columns(
    q_cand: torch.Tensor,
    existing_u: torch.Tensor,
    *,
    ortho_tol: float,
) -> torch.Tensor:
    """Double-project ``q_cand`` out of ``existing_u`` and QR-orthonormalize surviving columns."""
    d_in = int(q_cand.shape[0])
    if int(q_cand.shape[1]) == 0:
        return torch.empty((d_in, 0), dtype=torch.float64)

    work = q_cand.clone()
    if int(existing_u.shape[1]) > 0:
        work = work - existing_u @ (existing_u.T @ work)
        work = work - existing_u @ (existing_u.T @ work)

    col_norms = torch.linalg.vector_norm(work, dim=0)
    keep_cols = col_norms > max(ortho_tol, 1e-12)
    if not bool(keep_cols.any().item()):
        return torch.empty((d_in, 0), dtype=torch.float64)

    q_qr, r_qr = torch.linalg.qr(work[:, keep_cols], mode="reduced")
    r_diag = torch.abs(torch.diagonal(r_qr))
    keep_qr = r_diag > max(ortho_tol, 1e-12)
    if not bool(keep_qr.any().item()):
        return torch.empty((d_in, 0), dtype=torch.float64)
    q_clean = q_qr[:, keep_qr]
    if int(existing_u.shape[1]) > 0:
        q_clean = q_clean - existing_u @ (existing_u.T @ q_clean)
        q_clean, _ = torch.linalg.qr(q_clean, mode="reduced")
    return q_clean.contiguous()


class SequentialNullspaceBasis:
    """Stateful orthonormal basis tracker maintaining ``U_0`` plus incremental round bases ``U_{1..t}``.

    Tracks protected general-prose directions ``U_0`` (round ``0``) and incrementally
    appends orthonormalized skill key directions ``U_1, ..., U_t`` from accepted WSP rounds,
    exposing exact capacity monitoring (``remaining_null_rank = ambient_dim - used_rank``),
    null-space projection ``P_t = I - U_t U_t^T``, and atomic snapshot/restore for rollback.
    """

    def __init__(
        self,
        ambient_dim: int | None = None,
        *,
        prose_keys: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor] | None = None,
        u0_basis: torch.Tensor | None = None,
        sv_threshold: float = DEFAULT_SV_THRESHOLD,
        max_prose_rank: int | None = None,
        ortho_tol: float = DEFAULT_ORTHO_TOL,
        min_null_rank: int = DEFAULT_MIN_NULL_RANK,
    ) -> None:
        if prose_keys is not None and u0_basis is not None:
            raise ValueError("Provide at most one of 'prose_keys' or 'u0_basis', not both")
        sv_thr = _validate_nonneg_finite(sv_threshold, "sv_threshold")
        o_tol = _validate_positive_finite(ortho_tol, "ortho_tol")
        min_nr = _validate_nonneg_int(min_null_rank, "min_null_rank")
        max_pr = _validate_positive_int(max_prose_rank, "max_prose_rank") if max_prose_rank is not None else None

        self._sv_threshold: float = sv_thr
        self._ortho_tol: float = o_tol
        self._min_null_rank: int = min_nr
        self._round_records: list[SequentialRoundRecord] = []
        self._round_slices: dict[str, tuple[int, int]] = {}
        self._baseline_general_nll: float | None = None
        self._historical_baseline_nlls: dict[str, float] = {}

        if prose_keys is not None:
            dim_hint = _validate_positive_int(ambient_dim, "ambient_dim") if ambient_dim is not None else None
            keys_2d = _normalize_key_collection(prose_keys, "prose_keys", expected_dim=dim_hint)
            self._ambient_dim = int(keys_2d.shape[1])
            q0, _ = _orthonormalize_rows_svd(keys_2d, sv_threshold=sv_thr, max_rank=max_pr)
            q0 = _qr_reorthogonalize_columns(
                q0, torch.empty((self._ambient_dim, 0), dtype=torch.float64), ortho_tol=o_tol
            )
            self._u_basis = q0
            self._base_prose_rank = int(q0.shape[1])
            if self._base_prose_rank > 0:
                rec0 = SequentialRoundRecord(
                    round_index=0,
                    round_id="E0_general_prose",
                    col_start=0,
                    col_end=self._base_prose_rank,
                    added_rank=self._base_prose_rank,
                    key_count=int(keys_2d.shape[0]),
                    keys_sha256=_tensor_bytes_sha256(keys_2d),
                    pre_remaining_null_rank=self._ambient_dim,
                    post_remaining_null_rank=self._ambient_dim - self._base_prose_rank,
                    baseline_nll=None,
                )
                self._round_records.append(rec0)
                self._round_slices[rec0.round_id] = (0, self._base_prose_rank)
        elif u0_basis is not None:
            u0_raw = _validate_finite_2d(u0_basis, "u0_basis")
            d_in, r_raw = int(u0_raw.shape[0]), int(u0_raw.shape[1])
            if ambient_dim is not None:
                dim_hint = _validate_positive_int(ambient_dim, "ambient_dim")
                if d_in != dim_hint:
                    raise ValueError(f"u0_basis row dimension {d_in} must match ambient_dim={dim_hint}")
            if r_raw > d_in:
                raise ValueError(f"u0_basis rank {r_raw} cannot exceed ambient_dim={d_in}")
            gram = u0_raw.T @ u0_raw
            err = float((gram - torch.eye(r_raw, dtype=torch.float64)).abs().max().item())
            if err > 1e-4:
                raise ValueError(f"u0_basis columns must be orthonormal (max |U^T U - I|={err:.3e})")
            q0, _ = torch.linalg.qr(u0_raw, mode="reduced")
            self._ambient_dim = d_in
            self._u_basis = q0.contiguous()
            self._base_prose_rank = int(q0.shape[1])
            rec0 = SequentialRoundRecord(
                round_index=0,
                round_id="E0_general_prose",
                col_start=0,
                col_end=self._base_prose_rank,
                added_rank=self._base_prose_rank,
                key_count=self._base_prose_rank,
                keys_sha256=_tensor_bytes_sha256(self._u_basis),
                pre_remaining_null_rank=self._ambient_dim,
                post_remaining_null_rank=self._ambient_dim - self._base_prose_rank,
                baseline_nll=None,
            )
            self._round_records.append(rec0)
            self._round_slices[rec0.round_id] = (0, self._base_prose_rank)
        else:
            if ambient_dim is None:
                raise ValueError("One of 'ambient_dim', 'prose_keys', or 'u0_basis' must be provided")
            self._ambient_dim = _validate_positive_int(ambient_dim, "ambient_dim")
            self._u_basis = torch.empty((self._ambient_dim, 0), dtype=torch.float64)
            self._base_prose_rank = 0

    @classmethod
    def from_prose_keys(
        cls,
        prose_keys: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor],
        *,
        sv_threshold: float = DEFAULT_SV_THRESHOLD,
        max_prose_rank: int | None = None,
        ortho_tol: float = DEFAULT_ORTHO_TOL,
        min_null_rank: int = DEFAULT_MIN_NULL_RANK,
    ) -> SequentialNullspaceBasis:
        """Construct a ``SequentialNullspaceBasis`` protecting the SVD subspace of ``prose_keys``."""
        return cls(
            prose_keys=prose_keys,
            sv_threshold=sv_threshold,
            max_prose_rank=max_prose_rank,
            ortho_tol=ortho_tol,
            min_null_rank=min_null_rank,
        )

    @property
    def ambient_dim(self) -> int:
        return self._ambient_dim

    @property
    def used_rank(self) -> int:
        return int(self._u_basis.shape[1])

    @property
    def protected_rank(self) -> int:
        return int(self._u_basis.shape[1])

    @property
    def base_prose_rank(self) -> int:
        return self._base_prose_rank

    @property
    def skill_rank(self) -> int:
        return int(self._u_basis.shape[1]) - self._base_prose_rank

    @property
    def remaining_null_rank(self) -> int:
        return self._ambient_dim - int(self._u_basis.shape[1])

    @property
    def num_rounds(self) -> int:
        return sum(1 for rec in self._round_records if rec.round_index > 0)

    @property
    def utilization_ratio(self) -> float:
        return float(self.used_rank) / float(self._ambient_dim)

    @property
    def sv_threshold(self) -> float:
        return self._sv_threshold

    @property
    def min_null_rank(self) -> int:
        return self._min_null_rank

    @property
    def u_basis(self) -> torch.Tensor:
        return self._u_basis.clone()

    @property
    def u0_basis(self) -> torch.Tensor:
        return self._u_basis[:, : self._base_prose_rank].clone()

    @property
    def skill_basis(self) -> torch.Tensor:
        return self._u_basis[:, self._base_prose_rank :].clone()

    @property
    def round_records(self) -> tuple[SequentialRoundRecord, ...]:
        return tuple(self._round_records)

    @property
    def baseline_general_nll(self) -> float | None:
        return self._baseline_general_nll

    @property
    def historical_baseline_nlls(self) -> dict[str, float]:
        return dict(self._historical_baseline_nlls)

    def set_baseline_general_nll(self, nll: float) -> None:
        """Record the reference round-0 general-prose NLL baseline."""
        self._baseline_general_nll = _validate_positive_finite(nll, "baseline_general_nll")

    def available_expandable_rank(self, min_null_rank: int | None = None) -> int:
        """Return how many new basis dimensions can be appended while keeping ``min_null_rank`` free."""
        reserve = self._min_null_rank if min_null_rank is None else _validate_nonneg_int(min_null_rank, "min_null_rank")
        return max(0, self.remaining_null_rank - reserve)

    def is_exhausted(self, min_null_rank: int | None = None) -> bool:
        """Return True iff remaining free dimensions are at or below the protected reserve floor."""
        reserve = self._min_null_rank if min_null_rank is None else _validate_nonneg_int(min_null_rank, "min_null_rank")
        if reserve == 0:
            return self.remaining_null_rank == 0
        return self.remaining_null_rank <= reserve

    def orthogonality_error(self) -> float:
        """Return Frobenius deviation ``||U_t^T U_t - I||_F`` (``0.0`` when ``used_rank == 0``)."""
        r = self.used_rank
        if r == 0:
            return 0.0
        gram = self._u_basis.T @ self._u_basis
        eye_r = torch.eye(r, dtype=torch.float64)
        return float(torch.linalg.matrix_norm(gram - eye_r, ord="fro").item())

    def projector(self) -> torch.Tensor:
        """Return the exact symmetric idempotent projector ``P_t = I_d - U_t U_t^T`` in ``float64``."""
        eye_d = torch.eye(self._ambient_dim, dtype=torch.float64)
        if self.used_rank == 0:
            return eye_d
        p_t = eye_d - (self._u_basis @ self._u_basis.T)
        return (0.5 * (p_t + p_t.T)).contiguous()

    def project(self, tensor: torch.Tensor, *, keys_as_columns: bool = False) -> torch.Tensor:
        """Project ``tensor`` into ``N(U_t^T)`` in ``float64`` without materializing ``P_t``."""
        mat = _validate_finite_2d(tensor, "tensor")
        _validate_bool(keys_as_columns, "keys_as_columns")
        if self.used_rank == 0:
            if keys_as_columns and int(mat.shape[0]) != self._ambient_dim:
                raise ValueError(f"tensor row count {mat.shape[0]} must match ambient_dim={self._ambient_dim}")
            if not keys_as_columns and int(mat.shape[1]) != self._ambient_dim:
                raise ValueError(f"tensor width {mat.shape[1]} must match ambient_dim={self._ambient_dim}")
            return mat

        if keys_as_columns:
            if int(mat.shape[0]) != self._ambient_dim:
                raise ValueError(f"tensor row count {mat.shape[0]} must match ambient_dim={self._ambient_dim}")
            out = mat - self._u_basis @ (self._u_basis.T @ mat)
            out = out - self._u_basis @ (self._u_basis.T @ out)
            return out.contiguous()

        if int(mat.shape[1]) != self._ambient_dim:
            raise ValueError(f"tensor width {mat.shape[1]} must match ambient_dim={self._ambient_dim}")
        out = mat - (mat @ self._u_basis) @ self._u_basis.T
        out = out - (out @ self._u_basis) @ self._u_basis.T
        return out.contiguous()

    def get_round_basis(self, round_id: str) -> torch.Tensor:
        """Return the orthonormal column slice ``U_j`` associated with ``round_id``."""
        if not isinstance(round_id, str) or round_id not in self._round_slices:
            raise KeyError(f"Unknown round_id {round_id!r}")
        c0, c1 = self._round_slices[round_id]
        return self._u_basis[:, c0:c1].clone()

    def analyze_candidate_keys(
        self,
        k_new: torch.Tensor,
        *,
        sv_threshold: float | None = None,
        keys_as_columns: bool | None = None,
        max_added_rank: int | None = None,
    ) -> dict[str, Any]:
        """Inspect how ``k_new`` projects onto ``P_t`` without mutating basis state."""
        sv_thr = self._sv_threshold if sv_threshold is None else _validate_nonneg_finite(sv_threshold, "sv_threshold")
        max_add = _validate_positive_int(max_added_rank, "max_added_rank") if max_added_rank is not None else None
        k_rows = _normalize_key_collection(
            k_new, "k_new", expected_dim=self._ambient_dim, keys_as_columns=keys_as_columns
        )
        k_col = k_rows.T.contiguous()
        k_perp_col = self.project(k_col, keys_as_columns=True)
        raw_norm_sq = float((k_col * k_col).sum().item())
        perp_norm_sq = float((k_perp_col * k_perp_col).sum().item())
        proj_ratio = (perp_norm_sq / raw_norm_sq) if raw_norm_sq > 1e-24 else 0.0

        q_cand, s_perp = _orthonormalize_rows_svd(k_perp_col.T, sv_threshold=sv_thr, max_rank=max_add)
        q_clean = _qr_reorthogonalize_columns(q_cand, self._u_basis, ortho_tol=self._ortho_tol)
        return {
            "k_rows": k_rows,
            "k_col": k_col,
            "k_perp_col": k_perp_col,
            "key_count": int(k_rows.shape[0]),
            "raw_norm_sq": raw_norm_sq,
            "perp_norm_sq": perp_norm_sq,
            "projected_energy_ratio": proj_ratio,
            "top_perp_sv": float(s_perp[0].item()) if s_perp.numel() > 0 else 0.0,
            "estimated_added_rank": int(q_clean.shape[1]),
            "q_new": q_clean,
        }

    def append_round_keys(
        self,
        k_new: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor],
        *,
        round_id: str | None = None,
        sv_threshold: float | None = None,
        min_null_rank: int | None = None,
        keys_as_columns: bool | None = None,
        baseline_nll: float | None = None,
        max_added_rank: int | None = None,
    ) -> SequentialRoundRecord:
        """Orthonormalize ``k_new`` against ``U_t`` and append ``U_{t+1}`` iff reserve capacity holds."""
        reserve = self._min_null_rank if min_null_rank is None else _validate_nonneg_int(min_null_rank, "min_null_rank")
        next_idx = self.num_rounds + 1
        resolved_id = f"round_{next_idx}" if round_id is None else str(round_id).strip()
        if not resolved_id:
            raise ValueError("round_id must be a non-empty string")
        if resolved_id in self._round_slices:
            raise ValueError(f"Duplicate round_id {resolved_id!r} in SequentialNullspaceBasis")

        b_nll = _validate_positive_finite(baseline_nll, "baseline_nll") if baseline_nll is not None else None
        k_rows = _normalize_key_collection(
            k_new, "k_new", expected_dim=self._ambient_dim, keys_as_columns=keys_as_columns
        )
        analysis = self.analyze_candidate_keys(
            k_rows, sv_threshold=sv_threshold, keys_as_columns=False, max_added_rank=max_added_rank
        )
        q_new: torch.Tensor = analysis["q_new"]
        add_r = int(q_new.shape[1])

        if self.remaining_null_rank < reserve or (self.remaining_null_rank - add_r) < reserve:
            raise NullspaceExhaustedError(
                f"Cannot append {add_r} dimensions for {resolved_id!r}: "
                f"remaining_null_rank={self.remaining_null_rank}, min_null_rank={reserve}"
            )

        pre_rem = self.remaining_null_rank
        c0 = self.used_rank
        if add_r > 0:
            self._u_basis = torch.cat([self._u_basis, q_new], dim=1).contiguous()
        c1 = self.used_rank
        post_rem = self.remaining_null_rank

        rec = SequentialRoundRecord(
            round_index=next_idx,
            round_id=resolved_id,
            col_start=c0,
            col_end=c1,
            added_rank=add_r,
            key_count=int(k_rows.shape[0]),
            keys_sha256=_tensor_bytes_sha256(k_rows),
            pre_remaining_null_rank=pre_rem,
            post_remaining_null_rank=post_rem,
            baseline_nll=b_nll,
        )
        self._round_records.append(rec)
        self._round_slices[resolved_id] = (c0, c1)
        if b_nll is not None:
            self._historical_baseline_nlls[resolved_id] = b_nll
        return rec

    def snapshot(self) -> dict[str, Any]:
        """Capture an isolated deep snapshot for atomic rollback."""
        return {
            "ambient_dim": self._ambient_dim,
            "base_prose_rank": self._base_prose_rank,
            "sv_threshold": self._sv_threshold,
            "ortho_tol": self._ortho_tol,
            "min_null_rank": self._min_null_rank,
            "u_basis": self._u_basis.clone(),
            "round_records": list(self._round_records),
            "round_slices": dict(self._round_slices),
            "baseline_general_nll": self._baseline_general_nll,
            "historical_baseline_nlls": dict(self._historical_baseline_nlls),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Atomically restore basis state from a snapshot produced by ``snapshot()``."""
        if not isinstance(snapshot, Mapping) or "u_basis" not in snapshot or "ambient_dim" not in snapshot:
            raise ValueError("Invalid SequentialNullspaceBasis snapshot")
        if int(snapshot["ambient_dim"]) != self._ambient_dim:
            raise ValueError(
                f"Snapshot ambient_dim {snapshot['ambient_dim']} != basis ambient_dim={self._ambient_dim}"
            )
        u_snap = snapshot["u_basis"]
        if not isinstance(u_snap, torch.Tensor) or u_snap.ndim != 2 or int(u_snap.shape[0]) != self._ambient_dim:
            raise ValueError("Invalid u_basis tensor in snapshot")
        self._base_prose_rank = int(snapshot["base_prose_rank"])
        self._sv_threshold = float(snapshot["sv_threshold"])
        self._ortho_tol = float(snapshot["ortho_tol"])
        self._min_null_rank = int(snapshot["min_null_rank"])
        self._u_basis = u_snap.detach().to(device="cpu", dtype=torch.float64).clone()
        self._round_records = list(snapshot["round_records"])
        self._round_slices = dict(snapshot["round_slices"])
        self._baseline_general_nll = snapshot["baseline_general_nll"]
        self._historical_baseline_nlls = dict(snapshot["historical_baseline_nlls"])

    def clone(self) -> SequentialNullspaceBasis:
        """Return an independent deep copy of this basis tracker."""
        copy_obj = SequentialNullspaceBasis(
            self._ambient_dim,
            sv_threshold=self._sv_threshold,
            ortho_tol=self._ortho_tol,
            min_null_rank=self._min_null_rank,
        )
        copy_obj.restore(self.snapshot())
        return copy_obj

    def state_sha256(self) -> str:
        """Return deterministic SHA-256 over ``U_t``, round history, and tracked baseline NLLs."""
        payload = {
            "ambient_dim": self._ambient_dim,
            "base_prose_rank": self._base_prose_rank,
            "used_rank": self.used_rank,
            "remaining_null_rank": self.remaining_null_rank,
            "u_basis_sha256": _tensor_bytes_sha256(self._u_basis),
            "round_records": [r.to_dict() for r in self._round_records],
            "baseline_general_nll": self._baseline_general_nll,
            "historical_baseline_nlls": self._historical_baseline_nlls,
        }
        return compute_payload_sha256(payload)

    def capacity_summary(self) -> dict[str, Any]:
        """Return JSON-serializable summary of basis rank utilization and orthogonality."""
        return {
            "ambient_dim": self._ambient_dim,
            "base_prose_rank": self._base_prose_rank,
            "skill_rank": self.skill_rank,
            "used_rank": self.used_rank,
            "remaining_null_rank": self.remaining_null_rank,
            "min_null_rank": self._min_null_rank,
            "utilization_ratio": self.utilization_ratio,
            "num_rounds": self.num_rounds,
            "orthogonality_error": self.orthogonality_error(),
            "basis_sha256": self.state_sha256(),
        }


# ---------------------------------------------------------------------------
# 3. Closed-Form Null-Space Constrained Sequential WSP Solver
# ---------------------------------------------------------------------------


def _build_refusal_response(
    d_out: int,
    d_in: int,
    basis: SequentialNullspaceBasis,
    *,
    status: str,
    reason: str,
    estimated_added_rank: int,
    projected_energy_ratio: float,
    min_null_rank: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build deterministic zero-delta refusal output when capacity or subspace checks fail."""
    return torch.zeros((d_out, d_in), dtype=torch.float64), {
        "accepted": False,
        "refused": True,
        "status": status,
        "reason": reason,
        "pre_remaining_null_rank": basis.remaining_null_rank,
        "projected_post_remaining_null_rank": basis.remaining_null_rank,
        "min_null_rank": min_null_rank,
        "estimated_added_rank": estimated_added_rank,
        "projected_energy_ratio": projected_energy_ratio,
        "delta_w_fro": 0.0,
        "basis_leakage_max": 0.0,
        "target_fit_rel_err": 1.0,
    }


@torch.no_grad()
def solve_sequential_guarded_wsp_update(
    k_new: torch.Tensor,
    v_target_delta: torch.Tensor,
    basis: SequentialNullspaceBasis,
    *,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    min_null_rank: int = DEFAULT_MIN_NULL_RANK,
    c_key_cov: torch.Tensor | None = None,
    sv_threshold: float | None = None,
    min_projected_energy_ratio: float = DEFAULT_MIN_PROJECTED_ENERGY_RATIO,
    keys_as_columns: bool | None = None,
    max_added_rank: int | None = None,
    raise_on_exhaustion: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve closed-form minimum-norm update strictly inside ``P_t = I - U_t U_t^T``.

    Solves ``min_{Delta W = Delta W P_t} ||Delta W K_new^T - V_delta^T||_F^2
    + tr(Delta W C_bg Delta W^T) + ridge_lambda ||Delta W||_F^2`` in CPU ``float64``,
    refusing cleanly before null-space capacity exhaustion or rank collapse below
    ``min_null_rank``.
    """
    if not isinstance(basis, SequentialNullspaceBasis):
        raise TypeError(f"basis must be a SequentialNullspaceBasis, got {type(basis).__name__}")
    lam = _validate_positive_finite(ridge_lambda, "ridge_lambda")
    min_nr = _validate_nonneg_int(min_null_rank, "min_null_rank")
    min_proj = _validate_unit_interval(min_projected_energy_ratio, "min_projected_energy_ratio")
    _validate_bool(raise_on_exhaustion, "raise_on_exhaustion")

    d_in = basis.ambient_dim
    k_col, v_col = _orient_edit_pair(k_new, v_target_delta, d_in, keys_as_columns=keys_as_columns)
    d_out, n_edits = int(v_col.shape[0]), int(k_col.shape[1])

    c0: torch.Tensor | None = None
    if c_key_cov is not None:
        c0 = _validate_finite_2d(c_key_cov, "c_key_cov")
        if c0.shape != (d_in, d_in):
            raise ValueError(f"c_key_cov shape {tuple(c0.shape)} must equal ({d_in}, {d_in})")
        c0 = 0.5 * (c0 + c0.T)
        c_evals = torch.linalg.eigvalsh(c0)
        max_ev = max(float(c_evals[-1].item()), 0.0)
        if float(c_evals[0].item()) < -1e-4 * max(1.0, max_ev):
            raise ValueError(f"c_key_cov must be positive semi-definite (min eig {float(c_evals[0].item()):.3e})")

    if basis.remaining_null_rank == 0 or basis.remaining_null_rank < min_nr:
        reason = (
            f"Null-space exhausted before update: remaining_null_rank={basis.remaining_null_rank} "
            f"< min_null_rank={min_nr}"
        )
        if raise_on_exhaustion:
            raise NullspaceExhaustedError(reason)
        return _build_refusal_response(
            d_out,
            d_in,
            basis,
            status="REFUSED_NULLSPACE_EXHAUSTED",
            reason=reason,
            estimated_added_rank=0,
            projected_energy_ratio=0.0,
            min_null_rank=min_nr,
        )

    analysis = basis.analyze_candidate_keys(
        k_col.T,
        sv_threshold=sv_threshold,
        keys_as_columns=False,
        max_added_rank=max_added_rank,
    )
    est_add = int(analysis["estimated_added_rank"])
    proj_ratio = float(analysis["projected_energy_ratio"])
    perp_norm_sq = float(analysis["perp_norm_sq"])

    if perp_norm_sq <= 1e-20 or proj_ratio < min_proj or est_add == 0:
        reason = (
            f"Candidate keys collapse inside protected subspace U_t "
            f"(projected_energy_ratio={proj_ratio:.3e}, estimated_added_rank={est_add})"
        )
        if raise_on_exhaustion:
            raise NullspaceExhaustedError(reason)
        return _build_refusal_response(
            d_out,
            d_in,
            basis,
            status="REFUSED_INFEASIBLE_SUBSPACE",
            reason=reason,
            estimated_added_rank=est_add,
            projected_energy_ratio=proj_ratio,
            min_null_rank=min_nr,
        )

    post_rem = basis.remaining_null_rank - est_add
    if post_rem < min_nr:
        reason = (
            f"Edit refused to prevent null-space rank collapse: remaining_null_rank={basis.remaining_null_rank} "
            f"- estimated_added_rank={est_add} = {post_rem} < min_null_rank={min_nr}"
        )
        if raise_on_exhaustion:
            raise NullspaceExhaustedError(reason)
        return _build_refusal_response(
            d_out,
            d_in,
            basis,
            status="REFUSED_NULLSPACE_EXHAUSTED",
            reason=reason,
            estimated_added_rank=est_add,
            projected_energy_ratio=proj_ratio,
            min_null_rank=min_nr,
        )

    k_perp: torch.Tensor = analysis["k_perp_col"]
    if c0 is None:
        dual_gram = (k_perp.T @ k_perp) + lam * torch.eye(n_edits, dtype=torch.float64)
        dual_gram = 0.5 * (dual_gram + dual_gram.T)
        sol_dual = torch.linalg.solve(dual_gram, v_col.T)
        delta_w = sol_dual.T @ k_perp.T
    else:
        p_t = basis.projector()
        p_c_p = p_t @ c0 @ p_t
        p_c_p = 0.5 * (p_c_p + p_c_p.T)
        gram = (k_perp @ k_perp.T) + p_c_p + lam * torch.eye(d_in, dtype=torch.float64)
        gram = 0.5 * (gram + gram.T)
        rhs_t = k_perp @ v_col.T
        sol_t = torch.linalg.solve(gram, rhs_t)
        delta_w = sol_t.T @ p_t

    delta_w = basis.project(delta_w, keys_as_columns=False)
    recon = delta_w @ k_col
    v_norm = float(v_col.norm().item())
    fit_err = float((recon - v_col).norm().item() / max(v_norm, 1e-12)) if v_norm > 1e-18 else 0.0
    leak_max = (
        float((delta_w @ basis._u_basis).abs().max().item()) if basis.used_rank > 0 else 0.0
    )

    return delta_w.contiguous(), {
        "accepted": True,
        "refused": False,
        "status": "SOLVED",
        "reason": "Solved closed-form minimum-norm update strictly inside P_t.",
        "pre_remaining_null_rank": basis.remaining_null_rank,
        "projected_post_remaining_null_rank": post_rem,
        "min_null_rank": min_nr,
        "estimated_added_rank": est_add,
        "projected_energy_ratio": proj_ratio,
        "delta_w_fro": float(delta_w.norm().item()),
        "basis_leakage_max": leak_max,
        "target_fit_rel_err": fit_err,
    }


# ---------------------------------------------------------------------------
# 4. Target Weight Resolution, Null-Space LoRA Fold & In-Place Atomic Commit
# ---------------------------------------------------------------------------


def _resolve_target_holder(
    target_module: nn.Module,
    *,
    layer_index: int | None,
    expert_index: int | None,
    weight_attr: str,
) -> dict[str, Any]:
    """Resolve dense weight or LoRA ``(B, A)`` target from a module, MoE MLP, or transformer."""
    if not isinstance(target_module, nn.Module):
        raise TypeError(f"target_module must be an nn.Module, got {type(target_module).__name__}")
    attr = str(weight_attr).strip()
    if attr not in VALID_WEIGHT_ATTRS:
        raise ValueError(f"weight_attr must be one of {sorted(VALID_WEIGHT_ATTRS)}, got {weight_attr!r}")

    mod: nn.Module = target_module
    if hasattr(mod, "blocks") and isinstance(mod.blocks, (Sequence, nn.ModuleList)):
        n_blocks = len(mod.blocks)
        if n_blocks == 0:
            raise ValueError("target_module.blocks must be non-empty")
        l_idx = (n_blocks - 1) if layer_index is None else _validate_nonneg_int(layer_index, "layer_index")
        if l_idx >= n_blocks:
            raise ValueError(f"layer_index {l_idx} out of range for {n_blocks} blocks")
        mod = mod.blocks[l_idx].mlp
    elif layer_index is not None:
        _validate_nonneg_int(layer_index, "layer_index")
        raise ValueError(f"{type(target_module).__name__} does not expose 'blocks' for layer_index={layer_index}")

    if isinstance(mod, CrowdLoRAMoEMLP):
        if expert_index is None:
            keys = sorted(int(k) for k in mod.experts.keys())
            if len(keys) == 1:
                e_idx = keys[0]
            else:
                raise ValueError("expert_index must be specified when editing a multi-expert CrowdLoRAMoEMLP")
        else:
            e_idx = _validate_nonneg_int(expert_index, "expert_index")
        if str(e_idx) not in mod.experts:
            raise ValueError(f"Expert {e_idx} is not resident in CrowdLoRAMoEMLP")
        mod = mod.experts[str(e_idx)]
    elif isinstance(mod, CrowdFineGrainedMoEMLP):
        if expert_index is None:
            raise ValueError("expert_index must be specified when editing CrowdFineGrainedMoEMLP")
        e_idx = _validate_nonneg_int(expert_index, "expert_index")
        if e_idx >= mod.num_experts:
            raise ValueError(f"expert_index {e_idx} out of range for {mod.num_experts} experts")
        list_attr = "experts_w2" if attr == "weight" else f"experts_{attr}"
        mod = getattr(mod, list_attr)[e_idx]
    elif expert_index is not None:
        _validate_nonneg_int(expert_index, "expert_index")
        raise ValueError(f"{type(mod).__name__} is not an MoE layer supporting expert_index={expert_index}")

    if isinstance(mod, CrowdLoRAExpertAdapter) or (
        hasattr(mod, f"{attr}_B") and hasattr(mod, f"{attr}_A")
    ):
        if attr == "weight":
            raise ValueError(f"{type(mod).__name__} requires weight_attr in ('w1', 'w2', 'w3'), got 'weight'")
        w_b = getattr(mod, f"{attr}_B")
        w_a = getattr(mod, f"{attr}_A")
        scaling = float(getattr(mod, "scaling", 1.0))
        return {
            "kind": "lora",
            "module": mod,
            "attr": attr,
            "w_b": w_b,
            "w_a": w_a,
            "scaling": scaling,
            "d_out": int(w_b.shape[0]),
            "d_in": int(w_a.shape[1]),
        }

    if isinstance(mod, nn.Linear) or isinstance(getattr(mod, "weight", None), torch.Tensor):
        w_t: torch.Tensor = mod.weight  # type: ignore[assignment]
        return {
            "kind": "dense",
            "module": mod,
            "attr": "weight",
            "weight": w_t,
            "d_out": int(w_t.shape[0]),
            "d_in": int(w_t.shape[1]),
        }

    sub = getattr(mod, attr, None)
    if isinstance(sub, nn.Linear) or isinstance(getattr(sub, "weight", None), torch.Tensor):
        w_t = sub.weight  # type: ignore[union-attr]
        return {
            "kind": "dense",
            "module": sub,
            "attr": attr,
            "weight": w_t,
            "d_out": int(w_t.shape[0]),
            "d_in": int(w_t.shape[1]),
        }

    raise ValueError(
        f"Could not resolve editable weight or LoRA pair ({attr}) from {type(target_module).__name__}"
    )


def _compute_holder_linear_output(holder: Mapping[str, Any], k_col: torch.Tensor) -> torch.Tensor:
    """Compute ``W_eff @ k_col`` in ``float64`` for goal-to-delta conversion."""
    if holder["kind"] == "lora":
        b64 = holder["w_b"].detach().to(device="cpu", dtype=torch.float64)
        a64 = holder["w_a"].detach().to(device="cpu", dtype=torch.float64)
        return (float(holder["scaling"]) * (b64 @ (a64 @ k_col))).contiguous()
    w64 = holder["weight"].detach().to(device="cpu", dtype=torch.float64)
    return (w64 @ k_col).contiguous()


@torch.no_grad()
def _apply_delta_to_holder_inplace(
    holder: Mapping[str, Any],
    delta_w: torch.Tensor,
    basis: SequentialNullspaceBasis,
    *,
    lora_fold_mode: str,
) -> dict[str, Any]:
    """Apply ``delta_w`` in-place to a dense weight or LoRA pair preserving ``U_t`` null-space."""
    if holder["kind"] == "dense":
        w_t: torch.Tensor = holder["weight"]
        w64 = _validate_finite_2d(w_t, "target_weight")
        if delta_w.shape != w64.shape:
            raise ValueError(f"delta_w shape {tuple(delta_w.shape)} != weight shape {tuple(w64.shape)}")
        w_t.copy_((w64 + delta_w).to(device=w_t.device, dtype=w_t.dtype))
        return {
            "target_kind": f"dense_{holder['attr']}",
            "fold_method": "dense_exact",
            "fold_reconstruction_rel_err": 0.0,
            "rank": None,
        }

    w_b: torch.Tensor = holder["w_b"]
    w_a: torch.Tensor = holder["w_a"]
    scaling = float(holder["scaling"])
    b64 = _validate_finite_2d(w_b, "w_b")
    a64 = _validate_finite_2d(w_a, "w_a")
    d_out, rank = int(b64.shape[0]), int(b64.shape[1])
    d_in = int(a64.shape[1])
    if delta_w.shape != (d_out, d_in):
        raise ValueError(f"delta_w shape {tuple(delta_w.shape)} != LoRA shape ({d_out}, {d_in})")

    delta_ba = delta_w / scaling
    delta_norm = float(delta_ba.norm().item())
    if delta_norm <= 1e-18:
        return {
            "target_kind": f"lora_{holder['attr']}",
            "fold_method": "zero_delta_noop",
            "fold_reconstruction_rel_err": 0.0,
            "rank": rank,
        }

    u_d, s_d, vh_d = torch.linalg.svd(delta_ba, full_matrices=False)
    sv_tol = 1e-10 * max(float(s_d[0].item()), 1e-12)
    active_sv = int((s_d > sv_tol).sum().item())

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
                a_row = (s_sqrt[idx_k] * vh_d[idx_k, :]).unsqueeze(0)
                a_new[slot, :] = basis.project(a_row, keys_as_columns=False).squeeze(0)
            w_b.copy_(b_new.to(device=w_b.device, dtype=w_b.dtype))
            w_a.copy_(a_new.to(device=w_a.device, dtype=w_a.dtype))
            recon_delta = (b_new @ a_new) - (b64 @ a64)
            rel_err = float((recon_delta - delta_ba).norm().item() / delta_norm)
            return {
                "target_kind": f"lora_{holder['attr']}",
                "fold_method": "zero_slot_exact" if active_sv <= len(zero_slots) else "zero_slot_truncated",
                "fold_reconstruction_rel_err": rel_err,
                "rank": rank,
            }

    if lora_fold_mode in {"auto", "update_a"} and float(b64.norm().item()) > 1e-12:
        delta_a = torch.linalg.lstsq(b64, delta_ba).solution
        delta_a = basis.project(delta_a, keys_as_columns=False)
        captured = b64 @ delta_a
        capture_ratio = float(captured.norm().item() / delta_norm)
        if lora_fold_mode == "update_a" or capture_ratio >= 0.90:
            a_new = (a64 + delta_a).contiguous()
            w_a.copy_(a_new.to(device=w_a.device, dtype=w_a.dtype))
            rel_err = float((captured - delta_ba).norm().item() / delta_norm)
            return {
                "target_kind": f"lora_{holder['attr']}",
                "fold_method": "update_a_nullspace",
                "fold_reconstruction_rel_err": rel_err,
                "rank": rank,
            }

    k_aug = min(rank, int(s_d.numel()))
    s_sqrt = s_d[:k_aug].clamp_min(0.0).sqrt()
    b_delta = u_d[:, :k_aug] * s_sqrt.unsqueeze(0)
    a_delta = basis.project(s_sqrt.unsqueeze(1) * vh_d[:k_aug, :], keys_as_columns=False)
    b_aug = torch.cat([b64, b_delta], dim=1)
    a_aug = torch.cat([a64, a_delta], dim=0)
    u_r, s_r, v_r = decompose_lora_qr_svd(b_aug, a_aug)
    k_keep = min(rank, int(s_r.numel()))
    s_top = s_r[:k_keep].clamp_min(0.0).sqrt()
    b_out = torch.zeros((d_out, rank), dtype=torch.float64)
    a_out = torch.zeros((rank, d_in), dtype=torch.float64)
    b_out[:, :k_keep] = u_r[:, :k_keep] * s_top.unsqueeze(0)
    a_out[:k_keep, :] = s_top.unsqueeze(1) * v_r[:, :k_keep].T
    w_b.copy_(b_out.to(device=w_b.device, dtype=w_b.dtype))
    w_a.copy_(a_out.to(device=w_a.device, dtype=w_a.dtype))
    recon_delta = (b_out @ a_out) - (b64 @ a64)
    rel_err = float((recon_delta - delta_ba).norm().item() / delta_norm)
    return {
        "target_kind": f"lora_{holder['attr']}",
        "fold_method": "qr_svd_rank_preserving",
        "fold_reconstruction_rel_err": rel_err,
        "rank": rank,
    }


def _restore_module_snapshot_bitwise(module: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
    """Restore ``module.state_dict()`` in-place preserving device, dtype, and parameter tying."""
    cur_sd = module.state_dict()
    for key, val in snapshot.items():
        cur_sd[key].copy_(val.to(device=cur_sd[key].device, dtype=cur_sd[key].dtype))


def _is_module_finite(module: nn.Module) -> bool:
    """Return True iff every tensor in ``module.state_dict()`` is finite."""
    for t in module.state_dict().values():
        if (t.is_floating_point() or t.is_complex()) and not bool(torch.isfinite(t.detach()).all().item()):
            return False
    return True


def _normalize_eval_output(
    raw_eval: Any,
    *,
    resolved_round_id: str,
) -> dict[str, Any]:
    """Validate and normalize user ``eval_fn(module)`` output into canonical domain metrics."""
    if not isinstance(raw_eval, Mapping) or not raw_eval:
        raise TypeError("eval_fn must return a non-empty mapping")

    if "domain_nlls" in raw_eval:
        raw_doms = raw_eval["domain_nlls"]
    elif "per_domain_nll" in raw_eval:
        raw_doms = raw_eval["per_domain_nll"]
    else:
        reserved = {
            "new_skill_id",
            "new_skill_efficacy",
            "historical_efficacies",
            "retention_ratio",
            "non_finite_outputs",
            "all_finite",
            "nll_mean",
        }
        raw_doms = {k: v for k, v in raw_eval.items() if str(k) not in reserved}

    if not isinstance(raw_doms, Mapping) or not raw_doms:
        raise ValueError("eval_fn output must include non-empty domain NLL entries")

    domain_nlls: dict[str, float] = {}
    all_finite = int(raw_eval.get("non_finite_outputs", 0)) == 0
    for raw_k in sorted(raw_doms.keys(), key=str):
        dom = str(raw_k).strip()
        if not dom:
            raise ValueError("Domain keys returned by eval_fn must be non-empty strings")
        raw_v = raw_doms[raw_k]
        if isinstance(raw_v, bool) or not isinstance(raw_v, (int, float)):
            raise TypeError(f"eval_fn NLL for {dom!r} must be numeric, got {type(raw_v).__name__}")
        val_f = float(raw_v)
        if not math.isfinite(val_f) or val_f <= 0.0:
            if not math.isfinite(val_f):
                all_finite = False
                domain_nlls[dom] = float("nan")
            else:
                raise ValueError(f"eval_fn NLL for {dom!r} must be > 0.0, got {val_f}")
        else:
            domain_nlls[dom] = val_f

    new_skill_id = raw_eval.get("new_skill_id")
    if new_skill_id is not None:
        new_skill_id = str(new_skill_id).strip()
    elif resolved_round_id in domain_nlls:
        new_skill_id = resolved_round_id

    new_skill_eff: float | None = None
    if "new_skill_efficacy" in raw_eval and raw_eval["new_skill_efficacy"] is not None:
        new_skill_eff = _validate_unit_interval(raw_eval["new_skill_efficacy"], "new_skill_efficacy")

    hist_eff: dict[str, float] = {}
    if "historical_efficacies" in raw_eval and raw_eval["historical_efficacies"] is not None:
        raw_he = raw_eval["historical_efficacies"]
        if not isinstance(raw_he, Mapping):
            raise TypeError("historical_efficacies must be a mapping")
        for hk, hv in raw_he.items():
            hist_eff[str(hk)] = _validate_unit_interval(hv, f"historical_efficacies[{hk!r}]")

    explicit_ret: float | None = None
    if "retention_ratio" in raw_eval and raw_eval["retention_ratio"] is not None:
        explicit_ret = _validate_nonneg_finite(raw_eval["retention_ratio"], "retention_ratio")

    return {
        "domain_nlls": domain_nlls,
        "all_finite": all_finite,
        "new_skill_id": new_skill_id,
        "new_skill_efficacy": new_skill_eff,
        "historical_efficacies": hist_eff,
        "explicit_retention_ratio": explicit_ret,
    }


def _built_in_residual_eval(
    holder: Mapping[str, Any],
    basis: SequentialNullspaceBasis,
    k_col: torch.Tensor,
    y_goal_col: torch.Tensor,
    v_delta_col: torch.Tensor,
    *,
    resolved_round_id: str,
    y_pre_u_slices: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    """Evaluate exact closed-form NLL proxy & retention across ``U_0``, ``U_{1..t}``, and ``k_new``."""
    domain_nlls: dict[str, float] = {}
    for label, (u_slice, y_ref) in y_pre_u_slices.items():
        y_cur = _compute_holder_linear_output(holder, u_slice)
        mse_drift = float(((y_cur - y_ref) ** 2).mean().item())
        domain_nlls[label] = 1.0 + mse_drift

    y_new_cur = _compute_holder_linear_output(holder, k_col)
    new_mse = float(((y_new_cur - y_goal_col) ** 2).mean().item())
    goal_energy = max(float((v_delta_col ** 2).mean().item()), 1e-12)
    domain_nlls[resolved_round_id] = 1.0 + new_mse
    new_eff = max(0.0, min(1.0, 1.0 - (new_mse / goal_energy)))
    hist_eff = {
        rec.round_id: max(0.0, min(1.0, 1.0 / domain_nlls[rec.round_id]))
        for rec in basis.round_records
        if rec.round_index > 0 and rec.round_id in domain_nlls
    }
    return {
        "domain_nlls": domain_nlls,
        "all_finite": all(math.isfinite(v) for v in domain_nlls.values()),
        "new_skill_id": resolved_round_id,
        "new_skill_efficacy": new_eff,
        "historical_efficacies": hist_eff,
        "explicit_retention_ratio": None,
    }


def _evaluate_round_anse_and_guards(
    *,
    pre_eval: Mapping[str, Any],
    cand_eval: Mapping[str, Any],
    basis: SequentialNullspaceBasis,
    resolved_round_id: str,
    baseline_general_nll: float | None,
    historical_baseline_nlls: Mapping[str, float] | None,
    cfg: SequentialNullspaceConfig,
    cand_weights_finite: bool,
) -> dict[str, Any]:
    """Compute general-prose guard, round ``1..t`` historical guards, retention, and ANSE energy."""
    pre_doms: dict[str, float] = dict(pre_eval["domain_nlls"])
    cand_doms: dict[str, float] = dict(cand_eval["domain_nlls"])
    all_finite = bool(cand_weights_finite and cand_eval["all_finite"] and set(pre_doms.keys()) == set(cand_doms.keys()))

    new_skill_id: str | None = cand_eval["new_skill_id"] or pre_eval["new_skill_id"]
    if new_skill_id is None:
        novel_keys = [
            k
            for k in sorted(cand_doms.keys())
            if not k.startswith(cfg.general_prefix)
            and k not in {"general", "general_nll"}
            and k not in basis.historical_baseline_nlls
            and (historical_baseline_nlls is None or k not in historical_baseline_nlls)
        ]
        if len(novel_keys) == 1:
            new_skill_id = novel_keys[0]

    general_keys = [
        k for k in sorted(cand_doms.keys()) if k.startswith(cfg.general_prefix) or k in {"general", "general_nll"}
    ]
    explicit_hist = dict(historical_baseline_nlls or {})
    merged_hist_baselines = {**basis.historical_baseline_nlls, **explicit_hist}
    historical_keys = [
        k
        for k in sorted(cand_doms.keys())
        if k not in general_keys and k != new_skill_id and k != resolved_round_id and k in merged_hist_baselines
    ]

    if not all_finite:
        return {
            "finite": False,
            "general_guard_ok": False,
            "historical_guard_ok": False,
            "guard_ok": False,
            "efficacy_ok": False,
            "nll_mean": float(sum(pre_doms.values()) / float(len(pre_doms))),
            "candidate_nll_mean": None,
            "retention_ratio": 0.0,
            "new_skill_efficacy": 0.0,
            "historical_retention_ratios": {k: 0.0 for k in historical_keys},
            "general_pct_vs_base": None,
            "worst_historical_pct_vs_base": None,
            "worst_domain_id": "",
            "worst_domain_pct_vs_base": None,
            "parent_anse_energy": float(sum(pre_doms.values()) / float(len(pre_doms))),
            "candidate_anse_energy": float(cfg.maximum_pain_energy),
            "delta_e": float(cfg.maximum_pain_energy),
            "new_skill_id": new_skill_id,
            "general_keys": general_keys,
            "historical_keys": historical_keys,
            "pre_domain_nlls": pre_doms,
            "cand_domain_nlls": cand_doms,
        }

    # Resolve general-prose baseline (round-0 baseline preferred over pre_doms)
    gen_ratios: list[float] = []
    per_domain_pct: dict[str, float] = {}
    for gk in general_keys:
        base_g = (
            float(baseline_general_nll)
            if (baseline_general_nll is not None and len(general_keys) == 1)
            else (
                float(basis.baseline_general_nll)
                if (basis.baseline_general_nll is not None and len(general_keys) == 1)
                else float(pre_doms[gk])
            )
        )
        ratio_g = float(cand_doms[gk]) / max(base_g, 1e-12)
        gen_ratios.append(ratio_g)
        per_domain_pct[gk] = 100.0 * (ratio_g - 1.0)

    max_gen_ratio = max(gen_ratios) if gen_ratios else 1.0
    general_guard_ok = max_gen_ratio <= (1.0 + cfg.domain_guard_max_rel + 1e-12)

    # Resolve historical round 1..t baselines & retention ratios
    hist_ratios: list[float] = []
    hist_retention: dict[str, float] = {}
    cand_hist_eff: dict[str, float] = dict(cand_eval["historical_efficacies"])
    for hk in historical_keys:
        base_h = float(merged_hist_baselines[hk])
        ratio_h = float(cand_doms[hk]) / max(base_h, 1e-12)
        hist_ratios.append(ratio_h)
        per_domain_pct[hk] = 100.0 * (ratio_h - 1.0)
        if hk in cand_hist_eff:
            hist_retention[hk] = float(cand_hist_eff[hk])
        else:
            hist_retention[hk] = min(1.0, base_h / max(float(cand_doms[hk]), 1e-12))

    max_hist_ratio = max(hist_ratios) if hist_ratios else 1.0
    historical_guard_ok = max_hist_ratio <= (1.0 + cfg.domain_guard_max_rel + 1e-12)

    if new_skill_id is not None and new_skill_id in cand_doms and new_skill_id in pre_doms:
        new_ratio = float(cand_doms[new_skill_id]) / max(float(pre_doms[new_skill_id]), 1e-12)
        per_domain_pct[new_skill_id] = 100.0 * (new_ratio - 1.0)
        new_nll_improved = float(cand_doms[new_skill_id]) < float(pre_doms[new_skill_id]) - 1e-12
    else:
        new_nll_improved = True

    cand_new_eff = cand_eval["new_skill_efficacy"]
    new_skill_eff = float(cand_new_eff) if cand_new_eff is not None else (1.0 if new_nll_improved else 0.0)
    if cand_eval["explicit_retention_ratio"] is not None:
        retention_ratio = float(cand_eval["explicit_retention_ratio"])
    elif hist_retention:
        retention_ratio = min(hist_retention.values())
    else:
        retention_ratio = new_skill_eff

    retention_ok = (not hist_retention) or (retention_ratio >= cfg.min_retention_ratio - 1e-12)
    new_eff_ok = True if cand_new_eff is None else (float(cand_new_eff) >= cfg.min_new_skill_efficacy - 1e-12)
    efficacy_ok = bool(new_nll_improved and retention_ok and new_eff_ok)

    guard_ok = bool(general_guard_ok and historical_guard_ok)
    pre_mean = float(sum(pre_doms.values()) / float(len(pre_doms)))
    cand_mean = float(sum(cand_doms.values()) / float(len(cand_doms)))

    pre_gen_ratios = [
        float(pre_doms[gk])
        / max(
            float(
                baseline_general_nll
                if (baseline_general_nll is not None and len(general_keys) == 1)
                else (basis.baseline_general_nll if (basis.baseline_general_nll is not None and len(general_keys) == 1) else pre_doms[gk])
            ),
            1e-12,
        )
        for gk in general_keys
    ]
    pre_gen_max = max(pre_gen_ratios) if pre_gen_ratios else 1.0
    parent_soft = cfg.soft_margin_penalty if pre_gen_max > cfg.anse_safe_general_ratio + 1e-12 else 0.0
    parent_anse_energy = pre_mean + parent_soft

    if not guard_ok or not efficacy_ok:
        cand_anse_energy = float(cfg.maximum_pain_energy + max(0.0, cand_mean))
    else:
        cand_soft = cfg.soft_margin_penalty if max_gen_ratio > cfg.anse_safe_general_ratio + 1e-12 else 0.0
        cand_anse_energy = cand_mean + cand_soft

    guarded_pct_items = [
        (k, per_domain_pct[k]) for k in (*general_keys, *historical_keys) if k in per_domain_pct
    ]
    if guarded_pct_items:
        worst_dom_id, worst_dom_pct = max(guarded_pct_items, key=lambda kv: kv[1])
    elif per_domain_pct:
        worst_dom_id, worst_dom_pct = max(per_domain_pct.items(), key=lambda kv: kv[1])
    else:
        worst_dom_id, worst_dom_pct = "", 0.0

    return {
        "finite": True,
        "general_guard_ok": general_guard_ok,
        "historical_guard_ok": historical_guard_ok,
        "guard_ok": guard_ok,
        "efficacy_ok": efficacy_ok,
        "nll_mean": cand_mean,
        "pre_nll_mean": pre_mean,
        "candidate_nll_mean": cand_mean,
        "retention_ratio": retention_ratio,
        "new_skill_efficacy": new_skill_eff,
        "historical_retention_ratios": hist_retention,
        "general_pct_vs_base": 100.0 * (max_gen_ratio - 1.0) if general_keys else 0.0,
        "worst_historical_pct_vs_base": 100.0 * (max_hist_ratio - 1.0) if historical_keys else 0.0,
        "worst_domain_id": worst_dom_id,
        "worst_domain_pct_vs_base": float(worst_dom_pct),
        "parent_anse_energy": float(parent_anse_energy),
        "candidate_anse_energy": float(cand_anse_energy),
        "delta_e": float(cand_anse_energy - parent_anse_energy),
        "new_skill_id": new_skill_id,
        "general_keys": general_keys,
        "historical_keys": historical_keys,
        "pre_domain_nlls": pre_doms,
        "cand_domain_nlls": cand_doms,
    }


def _resolve_commit_config(
    config: SequentialNullspaceConfig | None,
    *,
    ridge_lambda: float | None,
    min_null_rank: int | None,
    sv_threshold: float | None,
    lora_fold_mode: str | None,
) -> SequentialNullspaceConfig:
    """Merge optional per-call overrides into a validated ``SequentialNullspaceConfig``."""
    if config is not None and not isinstance(config, SequentialNullspaceConfig):
        raise TypeError(f"config must be SequentialNullspaceConfig or None, got {type(config).__name__}")
    base = config or SequentialNullspaceConfig()
    return SequentialNullspaceConfig(
        sv_threshold=base.sv_threshold if sv_threshold is None else _validate_nonneg_finite(sv_threshold, "sv_threshold"),
        ridge_lambda=base.ridge_lambda if ridge_lambda is None else _validate_positive_finite(ridge_lambda, "ridge_lambda"),
        min_null_rank=base.min_null_rank if min_null_rank is None else _validate_nonneg_int(min_null_rank, "min_null_rank"),
        min_projected_energy_ratio=base.min_projected_energy_ratio,
        domain_guard_max_rel=base.domain_guard_max_rel,
        anse_safe_general_ratio=base.anse_safe_general_ratio,
        maximum_pain_energy=base.maximum_pain_energy,
        soft_margin_penalty=base.soft_margin_penalty,
        min_retention_ratio=base.min_retention_ratio,
        min_new_skill_efficacy=base.min_new_skill_efficacy,
        max_basis_leakage=base.max_basis_leakage,
        require_strict_energy_drop=base.require_strict_energy_drop,
        lora_fold_mode=base.lora_fold_mode if lora_fold_mode is None else str(lora_fold_mode),
        general_prefix=base.general_prefix,
    )


@torch.no_grad()
def commit_sequential_wsp_round_inplace(
    target_module: nn.Module,
    basis: SequentialNullspaceBasis,
    k_new: torch.Tensor,
    v_target_delta: torch.Tensor,
    *,
    round_id: str | None = None,
    layer_index: int | None = None,
    expert_index: int | None = None,
    weight_attr: str = "w2",
    eval_fn: Callable[[nn.Module], Mapping[str, Any]] | None = None,
    baseline_general_nll: float | None = None,
    historical_baseline_nlls: Mapping[str, float] | None = None,
    ridge_lambda: float | None = None,
    min_null_rank: int | None = None,
    sv_threshold: float | None = None,
    c_key_cov: torch.Tensor | None = None,
    lora_fold_mode: str | None = None,
    keys_as_columns: bool | None = None,
    v_is_goal: bool = False,
    expand_basis_on_commit: bool = True,
    config: SequentialNullspaceConfig | None = None,
) -> dict[str, Any]:
    """Apply WSP round ``t+1`` in-place and expand ``U_{t+1}`` iff ANSE & guards strictly pass.

    Enforces atomic bitwise rollback on BOTH ``target_module`` weights and ``basis`` (``U_t``)
    whenever null-space capacity is exhausted, ANSE ``Delta E >= 0``, any general-prose or
    historical round ``1..t`` guard exceeds ``<= +1.0%``, or candidate tensors/NLLs become
    non-finite. Returns a canonical SHA-256 provenance receipt.
    """
    if not isinstance(basis, SequentialNullspaceBasis):
        raise TypeError(f"basis must be a SequentialNullspaceBasis, got {type(basis).__name__}")
    _validate_bool(v_is_goal, "v_is_goal")
    _validate_bool(expand_basis_on_commit, "expand_basis_on_commit")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError("eval_fn must be callable or None")
    if baseline_general_nll is not None:
        _validate_positive_finite(baseline_general_nll, "baseline_general_nll")

    cfg = _resolve_commit_config(
        config,
        ridge_lambda=ridge_lambda,
        min_null_rank=min_null_rank,
        sv_threshold=sv_threshold,
        lora_fold_mode=lora_fold_mode,
    )
    holder = _resolve_target_holder(
        target_module,
        layer_index=layer_index,
        expert_index=expert_index,
        weight_attr=weight_attr,
    )
    if holder["d_in"] != basis.ambient_dim:
        raise ValueError(
            f"Target layer input dim {holder['d_in']} must match basis.ambient_dim={basis.ambient_dim}"
        )

    k_col, v_col_raw = _orient_edit_pair(k_new, v_target_delta, basis.ambient_dim, keys_as_columns=keys_as_columns)
    if int(v_col_raw.shape[0]) != int(holder["d_out"]):
        raise ValueError(
            f"v_target_delta output dim {v_col_raw.shape[0]} must match target layer output dim {holder['d_out']}"
        )

    next_round_idx = basis.num_rounds + 1
    resolved_round_id = f"round_{next_round_idx}" if round_id is None else str(round_id).strip()
    if not resolved_round_id:
        raise ValueError("round_id must be a non-empty string")
    if resolved_round_id in basis._round_slices:
        raise ValueError(f"round_id {resolved_round_id!r} already exists in basis")

    v_delta_col = (
        (v_col_raw - _compute_holder_linear_output(holder, k_col)).contiguous() if v_is_goal else v_col_raw
    )
    y_goal_col = (_compute_holder_linear_output(holder, k_col) + v_delta_col).contiguous()

    # Pre-capture protected round output references for exact basis drift & fallback evaluation
    y_pre_u_slices: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for rec in basis.round_records:
        if rec.added_rank > 0:
            u_slice = basis._u_basis[:, rec.col_start : rec.col_end].clone()
            y_pre_u_slices[rec.round_id] = (u_slice, _compute_holder_linear_output(holder, u_slice))

    was_training = target_module.training
    target_module.eval()
    try:
        parent_weights_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in target_module.state_dict().items()}
        parent_weights_sha = compute_state_sha256(parent_weights_snapshot)
        basis_snapshot = basis.snapshot()
        parent_basis_sha = basis.state_sha256()
        pre_remaining_rank = basis.remaining_null_rank

        pre_eval = (
            _normalize_eval_output(eval_fn(target_module), resolved_round_id=resolved_round_id)
            if eval_fn is not None
            else _built_in_residual_eval(
                holder,
                basis,
                k_col,
                y_goal_col,
                v_delta_col,
                resolved_round_id=resolved_round_id,
                y_pre_u_slices=y_pre_u_slices,
            )
        )

        delta_w, solve_meta = solve_sequential_guarded_wsp_update(
            k_col,
            v_delta_col,
            basis,
            ridge_lambda=cfg.ridge_lambda,
            min_null_rank=cfg.min_null_rank,
            c_key_cov=c_key_cov,
            sv_threshold=cfg.sv_threshold,
            min_projected_energy_ratio=cfg.min_projected_energy_ratio,
            keys_as_columns=True,
            raise_on_exhaustion=False,
        )

        if bool(solve_meta["refused"]):
            pre_doms = pre_eval["domain_nlls"]
            pre_mean = float(sum(pre_doms.values()) / float(len(pre_doms)))
            receipt: dict[str, Any] = {
                "object": RECEIPT_OBJECT,
                "schema": SCHEMA_VERSION,
                "round_index": next_round_idx,
                "round_id": resolved_round_id,
                "accepted": False,
                "rolled_back": False,
                "refused_nullspace_exhausted": solve_meta["status"] == "REFUSED_NULLSPACE_EXHAUSTED",
                "status": str(solve_meta["status"]),
                "reason": str(solve_meta["reason"]),
                "nll_mean": pre_mean,
                "retention_ratio": 1.0 if basis.num_rounds > 0 else 0.0,
                "new_skill_efficacy": 0.0,
                "historical_retention_ratios": {
                    k: 1.0 for k in basis.historical_baseline_nlls
                },
                "anse_energy": float(cfg.maximum_pain_energy),
                "parent_anse_energy": pre_mean,
                "final_anse_energy": pre_mean,
                "delta_e": float(cfg.maximum_pain_energy - pre_mean),
                "general_guard_ok": True,
                "historical_guard_ok": True,
                "guard_ok": True,
                "general_pct_vs_base": 0.0,
                "worst_historical_pct_vs_base": 0.0,
                "worst_domain_id": "",
                "worst_domain_pct_vs_base": 0.0,
                "pre_remaining_null_rank": pre_remaining_rank,
                "post_remaining_null_rank": pre_remaining_rank,
                "added_basis_rank": 0,
                "basis_orthogonality_error": basis.orthogonality_error(),
                "max_historical_key_drift": 0.0,
                "fold_info": {"fold_method": "refused_noop", "fold_reconstruction_rel_err": 0.0},
                "solve_meta": solve_meta,
                "parent_weights_sha256": parent_weights_sha,
                "candidate_weights_sha256": parent_weights_sha,
                "final_weights_sha256": parent_weights_sha,
                "parent_basis_sha256": parent_basis_sha,
                "final_basis_sha256": parent_basis_sha,
                "config": asdict(cfg),
            }
            receipt["receipt_sha256"] = compute_payload_sha256(receipt)
            return receipt

        fold_info = _apply_delta_to_holder_inplace(
            holder,
            delta_w,
            basis,
            lora_fold_mode=cfg.lora_fold_mode,
        )
        cand_weights_finite = _is_module_finite(target_module)
        candidate_weights_sha = (
            compute_state_sha256(target_module) if cand_weights_finite else ("f" * 64)
        )

        # Measure exact output drift on all prior protected orthonormal blocks U_0 .. U_t
        max_hist_drift = 0.0
        for u_slice, y_ref in y_pre_u_slices.values():
            y_cand = _compute_holder_linear_output(holder, u_slice)
            drift_val = float((y_cand - y_ref).abs().max().item())
            max_hist_drift = max(max_hist_drift, drift_val)

        cand_eval = (
            _normalize_eval_output(eval_fn(target_module), resolved_round_id=resolved_round_id)
            if eval_fn is not None
            else _built_in_residual_eval(
                holder,
                basis,
                k_col,
                y_goal_col,
                v_delta_col,
                resolved_round_id=resolved_round_id,
                y_pre_u_slices=y_pre_u_slices,
            )
        )
        anse_summary = _evaluate_round_anse_and_guards(
            pre_eval=pre_eval,
            cand_eval=cand_eval,
            basis=basis,
            resolved_round_id=resolved_round_id,
            baseline_general_nll=baseline_general_nll,
            historical_baseline_nlls=historical_baseline_nlls,
            cfg=cfg,
            cand_weights_finite=cand_weights_finite,
        )

        leak_ok = max_hist_drift <= cfg.max_basis_leakage + 1e-12
        energy_ok = (
            (float(anse_summary["delta_e"]) < 0.0)
            if cfg.require_strict_energy_drop
            else (float(anse_summary["delta_e"]) <= 0.0)
        )
        commit_ok = bool(
            anse_summary["finite"]
            and anse_summary["guard_ok"]
            and anse_summary["efficacy_ok"]
            and leak_ok
            and energy_ok
        )

        added_rank = 0
        if commit_ok:
            if basis.baseline_general_nll is None and anse_summary["general_keys"]:
                first_gk = anse_summary["general_keys"][0]
                g_ref = (
                    float(baseline_general_nll)
                    if baseline_general_nll is not None
                    else float(anse_summary["pre_domain_nlls"][first_gk])
                )
                basis.set_baseline_general_nll(g_ref)

            new_sid = anse_summary["new_skill_id"] or resolved_round_id
            new_b_nll = (
                float(anse_summary["cand_domain_nlls"][new_sid])
                if new_sid in anse_summary["cand_domain_nlls"]
                else None
            )
            if expand_basis_on_commit:
                rec = basis.append_round_keys(
                    k_col,
                    round_id=resolved_round_id,
                    sv_threshold=cfg.sv_threshold,
                    min_null_rank=cfg.min_null_rank,
                    keys_as_columns=True,
                    baseline_nll=new_b_nll,
                )
                added_rank = rec.added_rank
                if new_sid != resolved_round_id and new_b_nll is not None:
                    basis._historical_baseline_nlls[new_sid] = new_b_nll
            elif new_b_nll is not None:
                basis._historical_baseline_nlls[new_sid] = new_b_nll

            status = "ACCEPTED"
            reason = "ANSE Delta E < 0 and general/historical guards <= +1.0%."
            accepted = True
            rolled_back = False
        else:
            _restore_module_snapshot_bitwise(target_module, parent_weights_snapshot)
            basis.restore(basis_snapshot)
            accepted = False
            rolled_back = True
            if not anse_summary["finite"]:
                status = "ROLLED_BACK_NON_FINITE"
                reason = "Non-finite candidate weights or NLL encountered; atomic rollback executed."
            elif not anse_summary["guard_ok"] or not leak_ok:
                status = "ROLLED_BACK_GUARD_VIOLATION"
                reason = "General-prose or historical skill guard exceeded +1.0% / basis drift ceiling; atomic rollback executed."
            elif not anse_summary["efficacy_ok"]:
                status = "ROLLED_BACK_INFEASIBLE_EDIT"
                reason = "Round edit failed efficacy or retention threshold; atomic rollback executed."
            else:
                status = "ROLLED_BACK_ENERGY_NON_IMPROVING"
                reason = "Candidate failed strict ANSE energy descent (Delta E >= 0); atomic rollback executed."

        final_weights_sha = compute_state_sha256(target_module)
        final_basis_sha = basis.state_sha256()
        if rolled_back and (final_weights_sha != parent_weights_sha or final_basis_sha != parent_basis_sha):
            raise RuntimeError(
                f"Atomic rollback verification failed: weights ({final_weights_sha[:12]} vs {parent_weights_sha[:12]}), "
                f"basis ({final_basis_sha[:12]} vs {parent_basis_sha[:12]})"
            )
    finally:
        target_module.train(was_training)

    cand_e = float(anse_summary["candidate_anse_energy"])
    parent_e = float(anse_summary["parent_anse_energy"])
    receipt = {
        "object": RECEIPT_OBJECT,
        "schema": SCHEMA_VERSION,
        "round_index": next_round_idx,
        "round_id": resolved_round_id,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "refused_nullspace_exhausted": False,
        "status": status,
        "reason": reason,
        "nll_mean": float(anse_summary["nll_mean"]),
        "candidate_nll_mean": anse_summary["candidate_nll_mean"],
        "retention_ratio": float(anse_summary["retention_ratio"]) if accepted else (1.0 if basis.num_rounds > 0 else 0.0),
        "candidate_retention_ratio": float(anse_summary["retention_ratio"]),
        "new_skill_efficacy": float(anse_summary["new_skill_efficacy"]) if accepted else 0.0,
        "candidate_new_skill_efficacy": float(anse_summary["new_skill_efficacy"]),
        "historical_retention_ratios": anse_summary["historical_retention_ratios"],
        "anse_energy": cand_e,
        "parent_anse_energy": parent_e,
        "final_anse_energy": cand_e if accepted else parent_e,
        "delta_e": float(anse_summary["delta_e"]),
        "general_guard_ok": bool(anse_summary["general_guard_ok"]),
        "historical_guard_ok": bool(anse_summary["historical_guard_ok"]),
        "guard_ok": bool(anse_summary["guard_ok"] and leak_ok),
        "general_pct_vs_base": anse_summary["general_pct_vs_base"],
        "worst_historical_pct_vs_base": anse_summary["worst_historical_pct_vs_base"],
        "worst_domain_id": str(anse_summary["worst_domain_id"]),
        "worst_domain_pct_vs_base": anse_summary["worst_domain_pct_vs_base"],
        "pre_remaining_null_rank": pre_remaining_rank,
        "post_remaining_null_rank": basis.remaining_null_rank,
        "added_basis_rank": added_rank,
        "basis_orthogonality_error": basis.orthogonality_error(),
        "max_historical_key_drift": float(max_hist_drift),
        "fold_info": fold_info,
        "solve_meta": solve_meta,
        "parent_weights_sha256": parent_weights_sha,
        "candidate_weights_sha256": candidate_weights_sha,
        "final_weights_sha256": final_weights_sha,
        "parent_basis_sha256": parent_basis_sha,
        "final_basis_sha256": final_basis_sha,
        "config": asdict(cfg),
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_GENERAL_RATIO",
    "DEFAULT_DOMAIN_GUARD_MAX_REL",
    "DEFAULT_MAX_BASIS_LEAKAGE",
    "DEFAULT_MIN_NEW_SKILL_EFFICACY",
    "DEFAULT_MIN_NULL_RANK",
    "DEFAULT_MIN_PROJECTED_ENERGY_RATIO",
    "DEFAULT_MIN_RETENTION_RATIO",
    "DEFAULT_ORTHO_TOL",
    "DEFAULT_RIDGE_LAMBDA",
    "DEFAULT_SOFT_MARGIN_PENALTY",
    "DEFAULT_SV_THRESHOLD",
    "GENERAL_EXPERT_PREFIX",
    "RECEIPT_OBJECT",
    "SCHEMA_VERSION",
    "VALID_LORA_FOLD_MODES",
    "VALID_WEIGHT_ATTRS",
    "NullspaceExhaustedError",
    "SequentialNullspaceBasis",
    "SequentialNullspaceConfig",
    "SequentialRoundRecord",
    "canonical_json_bytes",
    "commit_sequential_wsp_round_inplace",
    "compute_payload_sha256",
    "compute_state_sha256",
    "solve_sequential_guarded_wsp_update",
    "verify_sequential_wsp_receipt_provenance",
]

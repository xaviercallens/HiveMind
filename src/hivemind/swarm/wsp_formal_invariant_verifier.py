# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S35-08 Runtime Mathematical Proof-Obligation & Floating-Point Soundness Verifier.

Certifies Generation-S5 Weight-Space Programming (WSP) edits against five formal
algebraic and thermodynamic proof obligations mirroring ``lean4/HiveMindSpec.lean``:

1. **Projector Idempotence, Symmetry & Unit-Interval Spectrum**
   (``verify_projector_idempotence_and_symmetry``):
   Certifies ``||P_0^2 - P_0||_F <= tol``, ``||P_0^T - P_0||_F <= tol``, and
   symmetrized eigenvalue spectrum contained in ``[-tol, 1 + tol]``.
2. **Preserved-Key Invariance** (``verify_preserved_key_invariance``):
   Certifies ``||Delta_W K_0||_F / max(||K_0||_F, 1e-8) <= tol``.
3. **Orthogonal Multi-Gate Centroid Invariance**
   (``verify_orthogonal_multigate_centroid_invariance``):
   Certifies ``|u^T mu_s| <= tol`` for every specialist domain centroid ``mu_s``
   while maintaining strictly positive general-prose alignment ``u^T mu_0 > 0``.
4. **Anchor-Layer Bitwise Immutability**
   (``verify_anchor_layer_bitwise_immutability``):
   Certifies exact 0-bit mutation (via raw CPU byte XOR, catching even single-ULP
   flips and ``+0.0``/``-0.0`` sign-bit flips) across protected beneficial anchor
   layer prefixes.
5. **Monotone ANSE Stage Descent & Bundle Certification**
   (``verify_monotone_anse_stage_descent`` and ``certify_s5_surgery_bundle``):
   Verifies strict per-stage ANSE thermodynamic descent under the general-domain
   guard and aggregates all obligations into an immutable, SHA-256-sealed
   ``WSPFormalCertificate``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
import torch.nn as nn

SCHEMA_VERSION: str = "hivemind.wsp_formal_invariant_verifier/1"
CERTIFICATE_OBJECT: str = "hivemind.wsp_formal_certificate"
OBLIGATION_OBJECT: str = "hivemind.wsp_formal_obligation"

ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_PROJECTOR_TOL: float = 1e-6
DEFAULT_KEY_INVARIANCE_TOL: float = 1e-5
DEFAULT_CENTROID_INVARIANCE_TOL: float = 1e-5
DEFAULT_PROSE_ALIGNMENT_FLOOR: float = 0.0
DEFAULT_GENERAL_GUARD_REL: float = 0.01
DEFAULT_MIN_ENERGY_DESCENT: float = 0.0
DEFAULT_DENOMINATOR_FLOOR: float = 1e-8

OBLIGATION_NAMES: tuple[str, ...] = (
    "projector_idempotence_and_symmetry",
    "preserved_key_invariance",
    "orthogonal_multigate_centroid_invariance",
    "anchor_layer_bitwise_immutability",
    "monotone_anse_stage_descent",
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

TensorLike = torch.Tensor | np.ndarray


# ---------------------------------------------------------------------------
# 1. Validation, Immutability & Canonical SHA-256 Provenance Helpers
# ---------------------------------------------------------------------------


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_nonempty_str(value: Any, name: str) -> str:
    """Validate that ``value`` is a non-empty string."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} must be a non-empty string")
    return stripped


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


def _validate_nonneg_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a non-negative integer (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _to_cpu_float64_tensor(
    value: Any,
    name: str,
    *,
    expected_ndim: int | None = None,
) -> torch.Tensor:
    """Validate a floating-point ``torch.Tensor`` or ``numpy.ndarray`` and return CPU ``float64``."""
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating-point dtype, got {value.dtype}")
        t_cpu = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    elif isinstance(value, np.ndarray):
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(f"{name} must have a floating-point dtype, got {value.dtype}")
        arr = np.array(value, dtype=np.float64, copy=True)
        t_cpu = torch.from_numpy(arr).contiguous()
    else:
        raise TypeError(
            f"{name} must be a torch.Tensor or numpy.ndarray, got {type(value).__name__}"
        )

    if expected_ndim is not None and t_cpu.ndim != expected_ndim:
        raise ValueError(
            f"{name} must be {expected_ndim}D, got {t_cpu.ndim}D with shape {tuple(t_cpu.shape)}"
        )
    if t_cpu.numel() == 0 or any(dim == 0 for dim in t_cpu.shape):
        raise ValueError(f"{name} must be non-empty, got shape {tuple(t_cpu.shape)}")
    if not bool(torch.isfinite(t_cpu).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return t_cpu


def _freeze_json_value(value: Any) -> Any:
    """Recursively normalize and freeze containers into immutable JSON-safe structures."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Certificate values must be finite, got {value!r}")
        return value
    if isinstance(value, np.generic):
        return _freeze_json_value(value.item())
    if isinstance(value, Mapping):
        frozen = {
            _validate_nonempty_str(str(k), "metadata_key"): _freeze_json_value(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(_freeze_json_value(v) for v in sorted(value, key=str))
    raise TypeError(f"Unsupported value type in certificate metadata: {type(value).__name__}")


def _unfreeze_json_value(value: Any) -> Any:
    """Convert frozen ``MappingProxyType``/``tuple``/result structures into plain JSON types."""
    if isinstance(value, ObligationCheckResult):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _unfreeze_json_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_unfreeze_json_value(v) for v in value]
    if isinstance(value, list):
        return [_unfreeze_json_value(v) for v in value]
    return value


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically with strict finite-number enforcement."""
    normalized = _unfreeze_json_value(payload)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def compute_payload_sha256(payload: Any) -> str:
    """Return lowercase SHA-256 hex digest of ``canonical_json_bytes(payload)``."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _extract_state_mapping(
    state_or_module: Mapping[str, Any] | nn.Module,
    name: str,
) -> dict[str, torch.Tensor | np.ndarray]:
    """Validate and shallow-extract tensor mapping from a ``state_dict`` or ``nn.Module``."""
    raw: Any
    if isinstance(state_or_module, nn.Module):
        raw = state_or_module.state_dict()
    elif isinstance(state_or_module, Mapping):
        raw = state_or_module
    else:
        raise TypeError(
            f"{name} must be a Mapping[str, Tensor|ndarray] or nn.Module, "
            f"got {type(state_or_module).__name__}"
        )
    if not raw:
        raise ValueError(f"{name} must be non-empty")

    out: dict[str, torch.Tensor | np.ndarray] = {}
    for key, val in raw.items():
        k_str = _validate_nonempty_str(key, f"{name} key")
        if not isinstance(val, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"{name}[{k_str!r}] must be a torch.Tensor or numpy.ndarray, "
                f"got {type(val).__name__}"
            )
        out[k_str] = val
    return out


def _tensor_raw_bytes_and_meta(
    tensor: torch.Tensor | np.ndarray,
    key: str,
) -> tuple[str, tuple[int, ...], int, np.ndarray, bool, bool]:
    """Return ``(dtype_tag, shape, elem_size, uint8_view, is_float, is_finite)``."""
    if isinstance(tensor, torch.Tensor):
        t_cpu = tensor.detach().to(device="cpu").contiguous()
        dtype_tag = f"torch.{t_cpu.dtype}"
        shape = tuple(int(d) for d in t_cpu.shape)
        elem_size = int(t_cpu.element_size())
        is_float = bool(t_cpu.is_floating_point() or t_cpu.is_complex())
        is_finite = bool(torch.isfinite(t_cpu).all().item()) if is_float else True
        u8 = t_cpu.reshape(-1).view(torch.uint8).numpy()
        return dtype_tag, shape, elem_size, u8, is_float, is_finite

    if isinstance(tensor, np.ndarray):
        arr = np.ascontiguousarray(tensor)
        dtype_tag = f"numpy.{arr.dtype}"
        shape = tuple(int(d) for d in arr.shape)
        elem_size = int(arr.dtype.itemsize)
        is_float = bool(np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.complexfloating))
        is_finite = bool(np.all(np.isfinite(arr))) if is_float else True
        u8 = arr.view(np.uint8).reshape(-1)
        return dtype_tag, shape, elem_size, u8, is_float, is_finite

    raise TypeError(f"Tensor {key!r} must be a torch.Tensor or numpy.ndarray")


def _max_abs_tensor_diff(
    pre_val: torch.Tensor | np.ndarray,
    post_val: torch.Tensor | np.ndarray,
) -> float:
    """Compute maximum elementwise absolute difference in ``float64`` for matching-shape tensors."""
    if isinstance(pre_val, torch.Tensor) and isinstance(post_val, torch.Tensor):
        if pre_val.numel() == 0:
            return 0.0
        diff = (
            pre_val.detach().to(device="cpu", dtype=torch.float64)
            - post_val.detach().to(device="cpu", dtype=torch.float64)
        ).abs()
        return float(diff.max().item())
    pre_arr = np.asarray(pre_val, dtype=np.float64)
    post_arr = np.asarray(post_val, dtype=np.float64)
    if pre_arr.size == 0:
        return 0.0
    return float(np.max(np.abs(pre_arr - post_arr)))


def compute_state_sha256(
    state_or_module: Mapping[str, Any] | nn.Module,
    *,
    keys: Sequence[str] | None = None,
) -> str:
    """Compute a deterministic bitwise SHA-256 digest over state tensors (or a key subset)."""
    state = _extract_state_mapping(state_or_module, "state_or_module")
    selected = sorted(state.keys()) if keys is None else sorted(keys)
    digest = hashlib.sha256()
    for key in selected:
        if key not in state:
            raise KeyError(f"Key {key!r} missing from state_or_module")
        dtype_tag, shape, _, u8, _, is_finite = _tensor_raw_bytes_and_meta(state[key], key)
        if not is_finite:
            raise ValueError(f"state_or_module[{key!r}] contains non-finite values")
        digest.update(f"{key}:{dtype_tag}:{shape}:".encode())
        if u8.size > 0:
            digest.update(u8.tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 2. Immutable Obligation & Certificate Data Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObligationCheckResult:
    """Immutable SHA-256-sealed outcome of a single algebraic or thermodynamic proof obligation."""

    obligation_id: str
    name: str
    passed: bool
    max_residual_norm: float
    tolerance: float
    metrics: Mapping[str, Any]
    failure_reasons: tuple[str, ...]
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-compatible dictionary."""
        return {
            "object": OBLIGATION_OBJECT,
            "obligation_id": self.obligation_id,
            "name": self.name,
            "passed": self.passed,
            "max_residual_norm": self.max_residual_norm,
            "tolerance": self.tolerance,
            "metrics": _unfreeze_json_value(self.metrics),
            "failure_reasons": list(self.failure_reasons),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True)
class WSPFormalCertificate:
    """Immutable SHA-256 certificate aggregating all 5 S5 WSP proof obligations."""

    schema: str
    object: str
    surgery_id: str
    certified: bool
    passed_obligations: int
    total_obligations: int
    failed_obligations: tuple[str, ...]
    obligations: Mapping[str, ObligationCheckResult]
    max_residual_norm: float
    anse_energy: float
    initial_anse_energy: float
    energy_delta: float
    general_rel_drift: float
    pre_state_sha256: str
    post_state_sha256: str
    anchor_pre_sha256: str
    anchor_post_sha256: str
    anchor_mutated_bits: int
    metadata: Mapping[str, Any]
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize certificate and nested obligations to a canonical JSON-safe dictionary."""
        return {
            "schema": self.schema,
            "object": self.object,
            "surgery_id": self.surgery_id,
            "certified": self.certified,
            "passed_obligations": self.passed_obligations,
            "total_obligations": self.total_obligations,
            "failed_obligations": list(self.failed_obligations),
            "obligations": {k: v.to_dict() for k, v in self.obligations.items()},
            "max_residual_norm": self.max_residual_norm,
            "anse_energy": self.anse_energy,
            "initial_anse_energy": self.initial_anse_energy,
            "energy_delta": self.energy_delta,
            "general_rel_drift": self.general_rel_drift,
            "pre_state_sha256": self.pre_state_sha256,
            "post_state_sha256": self.post_state_sha256,
            "anchor_pre_sha256": self.anchor_pre_sha256,
            "anchor_post_sha256": self.anchor_post_sha256,
            "anchor_mutated_bits": self.anchor_mutated_bits,
            "metadata": _unfreeze_json_value(self.metadata),
            "receipt_sha256": self.receipt_sha256,
        }


def _make_obligation_result(
    *,
    obligation_id: str,
    name: str,
    passed: bool,
    max_residual_norm: float,
    tolerance: float,
    metrics: Mapping[str, Any],
    failure_reasons: Sequence[str],
) -> ObligationCheckResult:
    """Build a frozen ``ObligationCheckResult`` with canonical ``receipt_sha256``."""
    frozen_metrics = _freeze_json_value(dict(metrics))
    reasons_tuple = tuple(str(r) for r in failure_reasons)
    core_payload = {
        "object": OBLIGATION_OBJECT,
        "obligation_id": obligation_id,
        "name": name,
        "passed": bool(passed),
        "max_residual_norm": float(max_residual_norm),
        "tolerance": float(tolerance),
        "metrics": _unfreeze_json_value(frozen_metrics),
        "failure_reasons": list(reasons_tuple),
    }
    sha = compute_payload_sha256(core_payload)
    return ObligationCheckResult(
        obligation_id=obligation_id,
        name=name,
        passed=bool(passed),
        max_residual_norm=float(max_residual_norm),
        tolerance=float(tolerance),
        metrics=frozen_metrics,
        failure_reasons=reasons_tuple,
        receipt_sha256=sha,
    )


def verify_wsp_formal_certificate_provenance(
    certificate: WSPFormalCertificate | Mapping[str, Any],
) -> bool:
    """Independently verify canonical SHA-256 digests and structural consistency of a certificate."""
    if isinstance(certificate, WSPFormalCertificate):
        payload: Mapping[str, Any] = certificate.to_dict()
    elif isinstance(certificate, Mapping):
        payload = certificate
    else:
        raise TypeError(
            f"certificate must be a WSPFormalCertificate or Mapping, got {type(certificate).__name__}"
        )

    claimed = payload.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if payload.get("schema") != SCHEMA_VERSION or payload.get("object") != CERTIFICATE_OBJECT:
        return False

    obligations = payload.get("obligations")
    if not isinstance(obligations, Mapping) or set(obligations.keys()) != set(OBLIGATION_NAMES):
        return False

    expected_failed: list[str] = []
    for ob_name in OBLIGATION_NAMES:
        ob_doc = obligations[ob_name]
        if not isinstance(ob_doc, Mapping):
            return False
        ob_sha = ob_doc.get("receipt_sha256")
        if not isinstance(ob_sha, str) or not _SHA256_HEX_RE.match(ob_sha):
            return False
        if ob_doc.get("object") != OBLIGATION_OBJECT or ob_doc.get("name") != ob_name:
            return False
        ob_core = {k: v for k, v in ob_doc.items() if k != "receipt_sha256"}
        try:
            if ob_sha != compute_payload_sha256(ob_core):
                return False
        except (TypeError, ValueError):
            return False
        if not bool(ob_doc.get("passed")):
            expected_failed.append(ob_name)

    if payload.get("total_obligations") != len(OBLIGATION_NAMES):
        return False
    if payload.get("passed_obligations") != len(OBLIGATION_NAMES) - len(expected_failed):
        return False
    failed_val = payload.get("failed_obligations")
    if not isinstance(failed_val, (list, tuple)) or list(failed_val) != expected_failed:
        return False
    if payload.get("certified") is not (len(expected_failed) == 0):
        return False

    cert_core = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(cert_core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


# ---------------------------------------------------------------------------
# 3. Algebraic & Thermodynamic Proof Obligations (a) - (e)
# ---------------------------------------------------------------------------


def verify_projector_idempotence_and_symmetry(
    p_null: TensorLike,
    *,
    tol: float = DEFAULT_PROJECTOR_TOL,
) -> ObligationCheckResult:
    """Certify ``||P_0^2 - P_0||_F <= tol``, ``||P_0^T - P_0||_F <= tol``, and ``spec(P_0) in [0, 1]``."""
    p64 = _to_cpu_float64_tensor(p_null, "p_null", expected_ndim=2)
    tol_val = _validate_nonneg_finite(tol, "tol")
    dim = int(p64.shape[0])
    if p64.shape[1] != dim:
        raise ValueError(f"p_null must be a square 2D matrix, got shape {tuple(p64.shape)}")

    idempotence_residual = float(torch.linalg.matrix_norm(p64 @ p64 - p64, ord="fro").item())
    symmetry_residual = float(torch.linalg.matrix_norm(p64.T - p64, ord="fro").item())

    p_sym = 0.5 * (p64 + p64.T)
    eigvals = torch.linalg.eigvalsh(p_sym)
    min_eig = float(eigvals[0].item())
    max_eig = float(eigvals[-1].item())
    below_zero = max(0.0, -min_eig)
    above_one = max(0.0, max_eig - 1.0)
    spectrum_violation = max(below_zero, above_one)

    idempotence_ok = idempotence_residual <= tol_val
    symmetry_ok = symmetry_residual <= tol_val
    spectrum_ok = (min_eig >= -tol_val) and (max_eig <= 1.0 + tol_val)

    reasons: list[str] = []
    if not idempotence_ok:
        reasons.append(f"idempotence_residual {idempotence_residual:.3e} > tol {tol_val:.3e}")
    if not symmetry_ok:
        reasons.append(f"symmetry_residual {symmetry_residual:.3e} > tol {tol_val:.3e}")
    if not spectrum_ok:
        reasons.append(
            f"eigenvalue spectrum [{min_eig:.6g}, {max_eig:.6g}] outside [-tol, 1 + tol]"
        )

    projected_rank = int((eigvals > 0.5).sum().item())
    max_residual = max(idempotence_residual, symmetry_residual, spectrum_violation)
    passed = idempotence_ok and symmetry_ok and spectrum_ok

    return _make_obligation_result(
        obligation_id="HM-S35-08-OBL-1",
        name="projector_idempotence_and_symmetry",
        passed=passed,
        max_residual_norm=max_residual,
        tolerance=tol_val,
        metrics={
            "ambient_dim": dim,
            "projected_rank": projected_rank,
            "null_rank_removed": dim - projected_rank,
            "idempotence_residual": idempotence_residual,
            "symmetry_residual": symmetry_residual,
            "min_eigenvalue": min_eig,
            "max_eigenvalue": max_eig,
            "spectrum_violation": spectrum_violation,
            "idempotence_ok": idempotence_ok,
            "symmetry_ok": symmetry_ok,
            "spectrum_ok": spectrum_ok,
            "max_residual_norm": max_residual,
        },
        failure_reasons=reasons,
    )


def verify_preserved_key_invariance(
    delta_w: TensorLike,
    k_preserved: TensorLike,
    *,
    tol: float = DEFAULT_KEY_INVARIANCE_TOL,
) -> ObligationCheckResult:
    """Certify ``||Delta_W K_0||_F / max(||K_0||_F, 1e-8) <= tol``."""
    dw64 = _to_cpu_float64_tensor(delta_w, "delta_w", expected_ndim=2)
    k64 = _to_cpu_float64_tensor(k_preserved, "k_preserved", expected_ndim=2)
    tol_val = _validate_nonneg_finite(tol, "tol")

    if dw64.shape[1] != k64.shape[0]:
        raise ValueError(
            f"Incompatible matrix shapes for Delta_W @ K_0: "
            f"delta_w {tuple(dw64.shape)} vs k_preserved {tuple(k64.shape)}"
        )

    leakage_fro = float(torch.linalg.matrix_norm(dw64 @ k64, ord="fro").item())
    key_fro = float(torch.linalg.matrix_norm(k64, ord="fro").item())
    delta_w_fro = float(torch.linalg.matrix_norm(dw64, ord="fro").item())
    denom = max(key_fro, DEFAULT_DENOMINATOR_FLOOR)
    normalized_leakage = leakage_fro / denom

    passed = normalized_leakage <= tol_val
    reasons: list[str] = []
    if not passed:
        reasons.append(f"normalized_key_leakage {normalized_leakage:.3e} > tol {tol_val:.3e}")

    return _make_obligation_result(
        obligation_id="HM-S35-08-OBL-2",
        name="preserved_key_invariance",
        passed=passed,
        max_residual_norm=normalized_leakage,
        tolerance=tol_val,
        metrics={
            "d_out": int(dw64.shape[0]),
            "d_in": int(dw64.shape[1]),
            "num_keys": int(k64.shape[1]),
            "leakage_fro_norm": leakage_fro,
            "key_fro_norm": key_fro,
            "delta_w_fro_norm": delta_w_fro,
            "denominator": denom,
            "normalized_leakage": normalized_leakage,
            "max_residual_norm": normalized_leakage,
        },
        failure_reasons=reasons,
    )


def _normalize_specialist_centroids(
    specialist_centroids: Mapping[Any, TensorLike] | Sequence[TensorLike] | TensorLike,
    expected_dim: int,
) -> dict[str, torch.Tensor]:
    """Validate specialist centroids into ``{centroid_label: (d,) float64 tensor}``."""
    out: dict[str, torch.Tensor] = {}
    if isinstance(specialist_centroids, (torch.Tensor, np.ndarray)):
        mat = _to_cpu_float64_tensor(specialist_centroids, "specialist_centroids", expected_ndim=2)
        if int(mat.shape[1]) != expected_dim:
            raise ValueError(
                f"specialist_centroids width {mat.shape[1]} must match u_gate dim {expected_dim}"
            )
        for idx in range(int(mat.shape[0])):
            out[f"centroid_{idx}"] = mat[idx].contiguous()
        return out

    if isinstance(specialist_centroids, Mapping):
        if not specialist_centroids:
            raise ValueError("specialist_centroids mapping must not be empty")
        for raw_k in sorted(specialist_centroids.keys(), key=str):
            if isinstance(raw_k, bool) or not isinstance(raw_k, (int, str)):
                raise TypeError("specialist_centroids keys must be int or str")
            label = str(raw_k).strip()
            if not label:
                raise ValueError("specialist_centroids string keys must be non-empty")
            vec = _to_cpu_float64_tensor(
                specialist_centroids[raw_k],
                f"specialist_centroids[{raw_k!r}]",
                expected_ndim=1,
            )
            if int(vec.shape[0]) != expected_dim:
                raise ValueError(
                    f"specialist_centroids[{raw_k!r}] dim {vec.shape[0]} must match u_gate dim {expected_dim}"
                )
            out[label] = vec
        return out

    if isinstance(specialist_centroids, (str, bytes, bytearray)) or not isinstance(
        specialist_centroids, Sequence
    ):
        raise TypeError(
            "specialist_centroids must be a Mapping, Sequence, or 2D Tensor/ndarray of centroids"
        )
    if len(specialist_centroids) == 0:
        raise ValueError("specialist_centroids sequence must not be empty")
    for idx, item in enumerate(specialist_centroids):
        vec = _to_cpu_float64_tensor(item, f"specialist_centroids[{idx}]", expected_ndim=1)
        if int(vec.shape[0]) != expected_dim:
            raise ValueError(
                f"specialist_centroids[{idx}] dim {vec.shape[0]} must match u_gate dim {expected_dim}"
            )
        out[f"centroid_{idx}"] = vec
    return out


def verify_orthogonal_multigate_centroid_invariance(
    u_gate: TensorLike,
    specialist_centroids: Mapping[Any, TensorLike] | Sequence[TensorLike] | TensorLike,
    prose_centroid: TensorLike,
    *,
    tol: float = DEFAULT_CENTROID_INVARIANCE_TOL,
    min_prose_alignment: float = DEFAULT_PROSE_ALIGNMENT_FLOOR,
) -> ObligationCheckResult:
    """Certify ``|u^T mu_s| <= tol`` for all specialist centroids while ``u^T mu_0 > 0``."""
    u64 = _to_cpu_float64_tensor(u_gate, "u_gate", expected_ndim=1)
    mu0_64 = _to_cpu_float64_tensor(prose_centroid, "prose_centroid", expected_ndim=1)
    dim = int(u64.shape[0])
    if int(mu0_64.shape[0]) != dim:
        raise ValueError(
            f"prose_centroid dim {mu0_64.shape[0]} must match u_gate dim {dim}"
        )

    tol_val = _validate_nonneg_finite(tol, "tol")
    min_prose = _validate_nonneg_finite(min_prose_alignment, "min_prose_alignment")
    spec_map = _normalize_specialist_centroids(specialist_centroids, dim)

    prose_alignment = float(torch.dot(u64, mu0_64).item())
    prose_ok = (prose_alignment > 0.0) and (prose_alignment >= min_prose)

    per_centroid_alignment: dict[str, float] = {}
    max_centroid_leakage = 0.0
    worst_centroid_label = ""
    violating_centroids: list[str] = []

    for label, mu_s in spec_map.items():
        dot_s = float(torch.dot(u64, mu_s).item())
        abs_dot = abs(dot_s)
        per_centroid_alignment[label] = dot_s
        if abs_dot >= max_centroid_leakage:
            max_centroid_leakage = abs_dot
            worst_centroid_label = label
        if abs_dot > tol_val:
            violating_centroids.append(label)

    centroids_ok = len(violating_centroids) == 0
    prose_violation = 0.0 if prose_ok else max(0.0, min_prose - prose_alignment, -prose_alignment)
    max_residual = max(max_centroid_leakage, prose_violation)
    passed = centroids_ok and prose_ok

    reasons: list[str] = []
    if not centroids_ok:
        reasons.append(
            f"specialist centroid leakage {max_centroid_leakage:.3e} > tol {tol_val:.3e} "
            f"on {violating_centroids}"
        )
    if not prose_ok:
        reasons.append(
            f"prose_alignment u^T mu_0={prose_alignment:.6g} violates > max(0.0, {min_prose:.6g})"
        )

    return _make_obligation_result(
        obligation_id="HM-S35-08-OBL-3",
        name="orthogonal_multigate_centroid_invariance",
        passed=passed,
        max_residual_norm=max_residual,
        tolerance=tol_val,
        metrics={
            "hidden_dim": dim,
            "num_specialist_centroids": len(spec_map),
            "u_gate_l2_norm": float(u64.norm().item()),
            "prose_alignment": prose_alignment,
            "min_prose_alignment": min_prose,
            "prose_alignment_ok": prose_ok,
            "centroids_ok": centroids_ok,
            "max_centroid_leakage": max_centroid_leakage,
            "worst_centroid_label": worst_centroid_label,
            "violating_centroids": tuple(violating_centroids),
            "per_centroid_alignment": per_centroid_alignment,
            "max_residual_norm": max_residual,
        },
        failure_reasons=reasons,
    )


def _normalize_anchor_prefixes(
    anchor_layer_prefixes: Sequence[str] | set[str] | frozenset[str],
) -> tuple[str, ...]:
    """Validate and deduplicate non-empty anchor layer prefix strings."""
    if isinstance(anchor_layer_prefixes, (str, bytes, bytearray)) or not isinstance(
        anchor_layer_prefixes, (Sequence, set, frozenset)
    ):
        raise TypeError(
            "anchor_layer_prefixes must be a Sequence or set of non-empty prefix strings"
        )
    if len(anchor_layer_prefixes) == 0:
        raise ValueError("anchor_layer_prefixes must not be empty")

    cleaned: list[str] = []
    for idx, pref in enumerate(anchor_layer_prefixes):
        s = _validate_nonempty_str(pref, f"anchor_layer_prefixes[{idx}]")
        cleaned.append(s)
    return tuple(sorted(set(cleaned)))


def verify_anchor_layer_bitwise_immutability(
    pre_state: Mapping[str, Any] | nn.Module,
    post_state: Mapping[str, Any] | nn.Module,
    anchor_layer_prefixes: Sequence[str] | set[str] | frozenset[str],
) -> ObligationCheckResult:
    """Certify 0-bit mutation across all tensors matching ``anchor_layer_prefixes``."""
    pre_map = _extract_state_mapping(pre_state, "pre_state")
    post_map = _extract_state_mapping(post_state, "post_state")
    prefixes = _normalize_anchor_prefixes(anchor_layer_prefixes)

    for key, val in pre_map.items():
        _, _, _, _, _, is_fin = _tensor_raw_bytes_and_meta(val, key)
        if not is_fin:
            raise ValueError(f"pre_state[{key!r}] contains non-finite values")

    pre_anchor_keys = sorted(k for k in pre_map if k.startswith(prefixes))
    post_anchor_keys = sorted(k for k in post_map if k.startswith(prefixes))
    all_anchor_keys = sorted(set(pre_anchor_keys) | set(post_anchor_keys))
    if not all_anchor_keys:
        raise ValueError(
            f"anchor_layer_prefixes {prefixes!r} matched 0 keys in pre_state and post_state"
        )

    anchor_pre_sha = (
        compute_state_sha256(pre_map, keys=pre_anchor_keys) if pre_anchor_keys else ""
    )

    mutated_keys: list[str] = []
    missing_post_keys: list[str] = []
    extra_post_keys: list[str] = []
    non_finite_post_keys: list[str] = []
    total_anchor_params = 0
    total_anchor_bits = 0
    mutated_bits = 0
    mutated_elements = 0
    max_abs_diff = 0.0
    post_all_finite = True

    for key in all_anchor_keys:
        if key not in post_map:
            missing_post_keys.append(key)
            mutated_keys.append(key)
            _, shape, _, u8_pre, _, _ = _tensor_raw_bytes_and_meta(pre_map[key], key)
            numel = int(math.prod(shape))
            total_anchor_params += numel
            total_anchor_bits += int(u8_pre.size * 8)
            mutated_bits += int(u8_pre.size * 8)
            mutated_elements += numel
            continue

        if key not in pre_map:
            extra_post_keys.append(key)
            mutated_keys.append(key)
            _, shape, _, u8_post, _, is_fin_post = _tensor_raw_bytes_and_meta(post_map[key], key)
            post_all_finite = post_all_finite and is_fin_post
            if not is_fin_post:
                non_finite_post_keys.append(key)
            mutated_bits += int(u8_post.size * 8)
            mutated_elements += int(math.prod(shape))
            continue

        dt_pre, sh_pre, elem_sz, u8_pre, is_flt, _ = _tensor_raw_bytes_and_meta(pre_map[key], key)
        dt_post, sh_post, _, u8_post, _, is_fin_post = _tensor_raw_bytes_and_meta(post_map[key], key)
        numel = int(math.prod(sh_pre))
        total_anchor_params += numel
        total_anchor_bits += int(u8_pre.size * 8)

        if not is_fin_post:
            post_all_finite = False
            non_finite_post_keys.append(key)

        if dt_pre != dt_post or sh_pre != sh_post or u8_pre.size != u8_post.size:
            mutated_keys.append(key)
            bits_est = max(int(u8_pre.size), int(u8_post.size)) * 8
            mutated_bits += bits_est
            mutated_elements += max(numel, int(math.prod(sh_post)))
            continue

        if u8_pre.size == 0:
            continue

        xor_u8 = np.bitwise_xor(u8_pre, u8_post)
        if np.any(xor_u8):
            mutated_keys.append(key)
            nz = xor_u8[xor_u8 != 0]
            mutated_bits += int(np.unpackbits(nz).sum())
            mutated_elements += int(np.count_nonzero(xor_u8.reshape(-1, elem_sz).any(axis=1)))
            if is_flt and is_fin_post:
                max_abs_diff = max(max_abs_diff, _max_abs_tensor_diff(pre_map[key], post_map[key]))

    anchor_post_sha = (
        compute_state_sha256(post_map, keys=post_anchor_keys)
        if (post_anchor_keys and post_all_finite)
        else ""
    )

    passed = (mutated_bits == 0) and (len(mutated_keys) == 0) and post_all_finite
    max_residual = float(mutated_bits) if mutated_bits > 0 else max_abs_diff
    reasons: list[str] = []
    if not passed:
        reasons.append(
            f"anchor bitwise immutability violated: mutated_bits={mutated_bits}, "
            f"mutated_keys={mutated_keys}"
        )

    return _make_obligation_result(
        obligation_id="HM-S35-08-OBL-4",
        name="anchor_layer_bitwise_immutability",
        passed=passed,
        max_residual_norm=max_residual,
        tolerance=0.0,
        metrics={
            "anchor_layer_prefixes": prefixes,
            "checked_anchor_keys": len(all_anchor_keys),
            "total_anchor_params": total_anchor_params,
            "total_anchor_bits": total_anchor_bits,
            "mutated_bits": mutated_bits,
            "mutated_elements": mutated_elements,
            "mutated_keys": tuple(mutated_keys),
            "missing_post_keys": tuple(missing_post_keys),
            "extra_post_keys": tuple(extra_post_keys),
            "non_finite_post_keys": tuple(non_finite_post_keys),
            "max_abs_diff": max_abs_diff,
            "anchor_pre_sha256": anchor_pre_sha,
            "anchor_post_sha256": anchor_post_sha,
            "max_residual_norm": max_residual,
        },
        failure_reasons=reasons,
    )


def _extract_attr_or_key(item: Any, *keys: str, default: Any = None) -> Any:
    """Read the first matching key from a mapping or attribute from an object."""
    if isinstance(item, Mapping):
        for key in keys:
            if key in item:
                return item[key]
        return default
    for key in keys:
        if hasattr(item, key):
            return getattr(item, key)
    return default


def _parse_stage_transitions(
    stage_records: Any,
) -> list[dict[str, Any]]:
    """Normalize stage receipts, energy dicts, or numeric sequences into transition dicts."""
    raw_seq: Any = stage_records
    if hasattr(stage_records, "stages") and not isinstance(stage_records, Mapping):
        raw_seq = stage_records.stages
    elif hasattr(stage_records, "stage_receipts") and not isinstance(stage_records, Mapping):
        raw_seq = stage_records.stage_receipts

    if isinstance(raw_seq, (str, bytes, bytearray)) or not isinstance(raw_seq, Sequence):
        raise TypeError("stage_records must be a non-empty sequence of stage receipts or energies")
    if len(raw_seq) == 0:
        raise ValueError("stage_records must not be empty")

    first = raw_seq[0]
    if isinstance(first, bool):
        raise TypeError("stage_records entries cannot be bool")
    # Scalar energy sequence: [e_0, e_1, ...]
    if isinstance(first, (int, float)):
        if len(raw_seq) < 2:
            raise ValueError("Scalar stage_records must contain at least 2 energies (pre, post)")
        energies = [_validate_finite_float(v, f"stage_records[{i}]") for i, v in enumerate(raw_seq)]
        return [
            {
                "stage_name": f"stage_{idx}",
                "accepted": True,
                "pre_anse_energy": energies[idx - 1],
                "post_anse_energy": energies[idx],
            }
            for idx in range(1, len(energies))
        ]

    # Sequence of (name, energy) pairs
    if isinstance(first, (tuple, list)) and len(first) == 2 and isinstance(first[0], str):
        if len(raw_seq) < 2:
            raise ValueError("Named-energy stage_records must contain at least 2 entries")
        parsed_pairs: list[tuple[str, float]] = []
        for idx, pair in enumerate(raw_seq):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise TypeError(f"stage_records[{idx}] must be a (stage_name, energy) pair")
            nm = _validate_nonempty_str(pair[0], f"stage_records[{idx}][0]")
            en = _validate_finite_float(pair[1], f"stage_records[{idx}][1]")
            parsed_pairs.append((nm, en))
        return [
            {
                "stage_name": parsed_pairs[idx][0],
                "accepted": True,
                "pre_anse_energy": parsed_pairs[idx - 1][1],
                "post_anse_energy": parsed_pairs[idx][1],
            }
            for idx in range(1, len(parsed_pairs))
        ]

    # Sequence of receipts / dicts with explicit pre/post OR trajectory dicts with single energy
    has_pre_post = _extract_attr_or_key(
        first, "pre_anse_energy", "pre_energy", "energy_before"
    ) is not None
    if has_pre_post:
        transitions: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_seq):
            st_name = _extract_attr_or_key(item, "stage_name", "stage", "name")
            name_str = _validate_nonempty_str(
                st_name if st_name is not None else f"stage_{idx}", f"stage_records[{idx}].stage_name"
            )
            pre_e_raw = _extract_attr_or_key(item, "pre_anse_energy", "pre_energy", "energy_before")
            post_e_raw = _extract_attr_or_key(item, "post_anse_energy", "post_energy", "energy_after")
            if pre_e_raw is None or post_e_raw is None:
                raise ValueError(
                    f"stage_records[{idx}] must provide both pre and post ANSE energies"
                )
            pre_e = _validate_finite_float(pre_e_raw, f"stage_records[{idx}].pre_anse_energy")
            post_e = _validate_finite_float(post_e_raw, f"stage_records[{idx}].post_anse_energy")

            acc_raw = _extract_attr_or_key(item, "accepted", "committed")
            accepted = _validate_bool(acc_raw, f"stage_records[{idx}].accepted") if acc_raw is not None else True

            claimed_delta_raw = _extract_attr_or_key(item, "delta_anse_energy", "energy_delta")
            claimed_delta = (
                _validate_finite_float(claimed_delta_raw, f"stage_records[{idx}].delta_anse_energy")
                if claimed_delta_raw is not None
                else None
            )

            gen_drift_raw = _extract_attr_or_key(item, "general_rel_drift", "general_domain_drift")
            gen_drift = (
                _validate_finite_float(gen_drift_raw, f"stage_records[{idx}].general_rel_drift")
                if gen_drift_raw is not None
                else None
            )

            pre_sha = _extract_attr_or_key(item, "pre_sha256")
            post_sha = _extract_attr_or_key(item, "post_sha256")
            edited_raw = _extract_attr_or_key(item, "edited_params", "stage_edited_params")
            edited_params = (
                _validate_nonneg_int(edited_raw, f"stage_records[{idx}].edited_params")
                if edited_raw is not None
                else None
            )

            transitions.append(
                {
                    "stage_name": name_str,
                    "accepted": accepted,
                    "pre_anse_energy": pre_e,
                    "post_anse_energy": post_e,
                    "claimed_delta": claimed_delta,
                    "general_rel_drift": gen_drift,
                    "pre_sha256": str(pre_sha) if pre_sha is not None else None,
                    "post_sha256": str(post_sha) if post_sha is not None else None,
                    "edited_params": edited_params,
                }
            )
        return transitions

    # Trajectory dicts with single energy key each
    if len(raw_seq) < 2:
        raise ValueError("Trajectory stage_records must contain at least 2 entries")
    points: list[tuple[str, float, float | None]] = []
    for idx, item in enumerate(raw_seq):
        st_name = _extract_attr_or_key(item, "stage_name", "stage", "name")
        name_str = _validate_nonempty_str(
            st_name if st_name is not None else f"stage_{idx}", f"stage_records[{idx}].stage_name"
        )
        en_raw = _extract_attr_or_key(item, "anse_energy", "energy")
        if en_raw is None:
            raise ValueError(f"stage_records[{idx}] missing energy field")
        en = _validate_finite_float(en_raw, f"stage_records[{idx}].anse_energy")
        gd_raw = _extract_attr_or_key(item, "general_rel_drift", "general_domain_drift")
        gd = _validate_finite_float(gd_raw, f"stage_records[{idx}].general_rel_drift") if gd_raw is not None else None
        points.append((name_str, en, gd))

    return [
        {
            "stage_name": points[idx][0],
            "accepted": True,
            "pre_anse_energy": points[idx - 1][1],
            "post_anse_energy": points[idx][1],
            "general_rel_drift": points[idx][2],
        }
        for idx in range(1, len(points))
    ]


def verify_monotone_anse_stage_descent(
    stage_records: Sequence[Any] | Any,
    *,
    min_descent: float = DEFAULT_MIN_ENERGY_DESCENT,
    general_guard_rel: float = DEFAULT_GENERAL_GUARD_REL,
    maximum_pain: float = ANSE_MAXIMUM_PAIN,
    allow_rolled_back_stages: bool = True,
    require_at_least_one_descent: bool = True,
) -> ObligationCheckResult:
    """Certify monotone ANSE stage energy descent and general-domain guard compliance."""
    min_desc = _validate_nonneg_finite(min_descent, "min_descent")
    guard_rel = _validate_nonneg_finite(general_guard_rel, "general_guard_rel")
    max_pain = _validate_positive_finite(maximum_pain, "maximum_pain")
    allow_rb = _validate_bool(allow_rolled_back_stages, "allow_rolled_back_stages")
    req_desc = _validate_bool(require_at_least_one_descent, "require_at_least_one_descent")

    transitions = _parse_stage_transitions(stage_records)
    initial_energy = float(transitions[0]["pre_anse_energy"])
    prev_effective_energy = initial_energy
    prev_effective_sha: str | None = transitions[0].get("pre_sha256")
    last_general_drift: float | None = None

    accepted_count = 0
    rolled_back_count = 0
    max_violation = 0.0
    reasons: list[str] = []
    stage_summaries: list[dict[str, Any]] = []

    for idx, tr in enumerate(transitions):
        st_name = tr["stage_name"]
        pre_e = float(tr["pre_anse_energy"])
        post_e = float(tr["post_anse_energy"])
        accepted = bool(tr["accepted"])
        pre_sha = tr.get("pre_sha256")
        post_sha = tr.get("post_sha256")
        edited_params = tr.get("edited_params")
        gen_drift = tr.get("general_rel_drift")
        claimed_delta = tr.get("claimed_delta")

        if idx > 0 and abs(pre_e - prev_effective_energy) > 1e-9:
            diff_chain = abs(pre_e - prev_effective_energy)
            max_violation = max(max_violation, diff_chain)
            reasons.append(
                f"stage {st_name!r} pre_anse_energy {pre_e} != previous effective {prev_effective_energy}"
            )

        if idx > 0 and prev_effective_sha is not None and pre_sha is not None and pre_sha != prev_effective_sha:
            max_violation = max(max_violation, 1.0)
            reasons.append(f"stage {st_name!r} pre_sha256 broke hash chain")

        if not accepted:
            rolled_back_count += 1
            if not allow_rb:
                max_violation = max(max_violation, 1.0)
                reasons.append(f"stage {st_name!r} was rolled back while allow_rolled_back_stages=False")
            if pre_sha is not None and post_sha is not None and pre_sha != post_sha:
                max_violation = max(max_violation, 1.0)
                reasons.append(f"rolled-back stage {st_name!r} failed bitwise SHA-256 identity")
            if edited_params is not None and edited_params != 0:
                max_violation = max(max_violation, float(edited_params))
                reasons.append(f"rolled-back stage {st_name!r} reported edited_params={edited_params}")
            if gen_drift is not None:
                last_general_drift = float(gen_drift)
            stage_summaries.append(
                {
                    "stage_name": st_name,
                    "accepted": False,
                    "pre_anse_energy": pre_e,
                    "effective_post_anse_energy": prev_effective_energy,
                    "energy_delta": 0.0,
                }
            )
            continue

        accepted_count += 1
        actual_delta = post_e - pre_e
        if claimed_delta is not None and abs(actual_delta - float(claimed_delta)) > 1e-7:
            delta_mismatch = abs(actual_delta - float(claimed_delta))
            max_violation = max(max_violation, delta_mismatch)
            reasons.append(
                f"stage {st_name!r} claimed delta {claimed_delta} != actual {actual_delta}"
            )

        pain_floor = 0.5 * max_pain
        if pre_e < pain_floor and post_e >= pain_floor:
            max_violation = max(max_violation, post_e)
            reasons.append(f"stage {st_name!r} entered ANSE maximum pain ({post_e})")

        if post_e < pain_floor and gen_drift is not None and float(gen_drift) > guard_rel + 1e-12:
            drift_excess = float(gen_drift) - guard_rel
            max_violation = max(max_violation, drift_excess)
            reasons.append(
                f"stage {st_name!r} general_rel_drift {gen_drift:.6f} > guard {guard_rel:.6f}"
            )

        descent_ok = (actual_delta < 0.0) if min_desc == 0.0 else (actual_delta <= -min_desc)
        if not descent_ok:
            non_descent_gap = max(0.0, actual_delta + min_desc)
            max_violation = max(max_violation, non_descent_gap if non_descent_gap > 0.0 else 1.0)
            reasons.append(
                f"stage {st_name!r} failed monotone descent: energy_delta={actual_delta:.6g} "
                f"(required < -{min_desc:.6g})"
            )

        prev_effective_energy = post_e
        if post_sha is not None:
            prev_effective_sha = post_sha
        if gen_drift is not None:
            last_general_drift = float(gen_drift)

        stage_summaries.append(
            {
                "stage_name": st_name,
                "accepted": True,
                "pre_anse_energy": pre_e,
                "effective_post_anse_energy": post_e,
                "energy_delta": actual_delta,
            }
        )

    final_energy = prev_effective_energy
    total_energy_delta = final_energy - initial_energy

    if final_energy >= 0.5 * max_pain:
        max_violation = max(max_violation, final_energy)
        reasons.append(f"final anse_energy {final_energy} remains in ANSE maximum pain")

    if last_general_drift is not None and last_general_drift > guard_rel + 1e-12:
        max_violation = max(max_violation, last_general_drift - guard_rel)
        reasons.append(
            f"final general_rel_drift {last_general_drift:.6f} > guard {guard_rel:.6f}"
        )

    if req_desc and (accepted_count == 0 or total_energy_delta >= 0.0):
        max_violation = max(max_violation, max(0.0, total_energy_delta), 1.0)
        reasons.append(
            f"pipeline achieved zero monotone ANSE descent (accepted_count={accepted_count}, "
            f"energy_delta={total_energy_delta:.6g})"
        )

    passed = len(reasons) == 0
    return _make_obligation_result(
        obligation_id="HM-S35-08-OBL-5",
        name="monotone_anse_stage_descent",
        passed=passed,
        max_residual_norm=max_violation,
        tolerance=min_desc,
        metrics={
            "num_stages_total": len(transitions),
            "accepted_stages": accepted_count,
            "rolled_back_stages": rolled_back_count,
            "initial_anse_energy": initial_energy,
            "anse_energy": final_energy,
            "energy_delta": total_energy_delta,
            "general_rel_drift": last_general_drift if last_general_drift is not None else 0.0,
            "general_guard_rel": guard_rel,
            "stages": stage_summaries,
            "max_residual_norm": max_violation,
        },
        failure_reasons=reasons,
    )


def _aggregate_projector_obligation(
    p_null: TensorLike | Mapping[str, TensorLike],
    *,
    tol: float,
) -> ObligationCheckResult:
    """Verify one projector matrix or a named mapping of layer projectors."""
    if not isinstance(p_null, Mapping):
        return verify_projector_idempotence_and_symmetry(p_null, tol=tol)
    if not p_null:
        raise ValueError("p_null mapping must not be empty")

    sub_results = {
        _validate_nonempty_str(str(k), "p_null key"): verify_projector_idempotence_and_symmetry(
            p_null[k], tol=tol
        )
        for k in sorted(p_null.keys(), key=str)
    }
    max_res = max(r.max_residual_norm for r in sub_results.values())
    passed = all(r.passed for r in sub_results.values())
    reasons = [
        f"{name}: {reason}"
        for name, res in sub_results.items()
        for reason in res.failure_reasons
    ]
    return _make_obligation_result(
        obligation_id="HM-S35-08-OBL-1",
        name="projector_idempotence_and_symmetry",
        passed=passed,
        max_residual_norm=max_res,
        tolerance=float(tol),
        metrics={
            "num_projectors": len(sub_results),
            "per_projector": {k: dict(v.metrics) for k, v in sub_results.items()},
            "idempotence_ok": all(bool(v.metrics["idempotence_ok"]) for v in sub_results.values()),
            "symmetry_ok": all(bool(v.metrics["symmetry_ok"]) for v in sub_results.values()),
            "spectrum_ok": all(bool(v.metrics["spectrum_ok"]) for v in sub_results.values()),
            "max_residual_norm": max_res,
        },
        failure_reasons=reasons,
    )


def _aggregate_preserved_key_obligation(
    delta_w: TensorLike | Mapping[str, TensorLike],
    k_preserved: TensorLike | Mapping[str, TensorLike],
    *,
    tol: float,
) -> ObligationCheckResult:
    """Verify preserved-key invariance for a single pair or named mapping of layers."""
    if not isinstance(delta_w, Mapping):
        if isinstance(k_preserved, Mapping):
            raise TypeError("k_preserved cannot be a Mapping when delta_w is a single matrix")
        return verify_preserved_key_invariance(delta_w, k_preserved, tol=tol)
    if not delta_w:
        raise ValueError("delta_w mapping must not be empty")

    dw_keys = {_validate_nonempty_str(str(k), "delta_w key"): k for k in delta_w}
    if isinstance(k_preserved, Mapping):
        kp_keys = {_validate_nonempty_str(str(k), "k_preserved key"): k for k in k_preserved}
        if set(dw_keys.keys()) != set(kp_keys.keys()):
            raise ValueError("delta_w and k_preserved mappings must have identical keys")
        pairs = {
            k_str: (delta_w[dw_keys[k_str]], k_preserved[kp_keys[k_str]])
            for k_str in sorted(dw_keys.keys())
        }
    else:
        pairs = {
            k_str: (delta_w[dw_keys[k_str]], k_preserved)
            for k_str in sorted(dw_keys.keys())
        }

    sub_results = {
        k_str: verify_preserved_key_invariance(dw_v, k_v, tol=tol)
        for k_str, (dw_v, k_v) in pairs.items()
    }
    max_res = max(r.max_residual_norm for r in sub_results.values())
    passed = all(r.passed for r in sub_results.values())
    reasons = [
        f"{name}: {reason}"
        for name, res in sub_results.items()
        for reason in res.failure_reasons
    ]
    return _make_obligation_result(
        obligation_id="HM-S35-08-OBL-2",
        name="preserved_key_invariance",
        passed=passed,
        max_residual_norm=max_res,
        tolerance=float(tol),
        metrics={
            "num_matrices": len(sub_results),
            "per_matrix": {k: dict(v.metrics) for k, v in sub_results.items()},
            "normalized_leakage": max_res,
            "max_residual_norm": max_res,
        },
        failure_reasons=reasons,
    )


def _aggregate_multigate_obligation(
    u_gate: TensorLike | Mapping[Any, TensorLike],
    specialist_centroids: Mapping[Any, Any] | Sequence[TensorLike] | TensorLike,
    prose_centroid: TensorLike | Mapping[Any, TensorLike],
    *,
    tol: float,
    min_prose_alignment: float,
) -> ObligationCheckResult:
    """Verify orthogonal multigate centroid invariance for one vector or a mapping of vectors."""
    if not isinstance(u_gate, Mapping):
        if isinstance(prose_centroid, Mapping):
            raise TypeError("prose_centroid cannot be a Mapping when u_gate is a single vector")
        return verify_orthogonal_multigate_centroid_invariance(
            u_gate,
            specialist_centroids,
            prose_centroid,
            tol=tol,
            min_prose_alignment=min_prose_alignment,
        )
    if not u_gate:
        raise ValueError("u_gate mapping must not be empty")

    sub_results: dict[str, ObligationCheckResult] = {}
    for raw_k in sorted(u_gate.keys(), key=str):
        k_str = _validate_nonempty_str(str(raw_k), "u_gate key")
        u_vec = u_gate[raw_k]
        spec_c = (
            specialist_centroids[raw_k]
            if (isinstance(specialist_centroids, Mapping) and raw_k in specialist_centroids)
            else specialist_centroids
        )
        prose_c = (
            prose_centroid[raw_k]
            if (isinstance(prose_centroid, Mapping) and raw_k in prose_centroid)
            else prose_centroid
        )
        sub_results[k_str] = verify_orthogonal_multigate_centroid_invariance(
            u_vec,
            spec_c,
            prose_c,
            tol=tol,
            min_prose_alignment=min_prose_alignment,
        )

    max_res = max(r.max_residual_norm for r in sub_results.values())
    passed = all(r.passed for r in sub_results.values())
    reasons = [
        f"{name}: {reason}"
        for name, res in sub_results.items()
        for reason in res.failure_reasons
    ]
    return _make_obligation_result(
        obligation_id="HM-S35-08-OBL-3",
        name="orthogonal_multigate_centroid_invariance",
        passed=passed,
        max_residual_norm=max_res,
        tolerance=float(tol),
        metrics={
            "num_gates": len(sub_results),
            "per_gate": {k: dict(v.metrics) for k, v in sub_results.items()},
            "centroids_ok": all(bool(v.metrics["centroids_ok"]) for v in sub_results.values()),
            "prose_alignment_ok": all(
                bool(v.metrics["prose_alignment_ok"]) for v in sub_results.values()
            ),
            "max_centroid_leakage": max(
                float(v.metrics["max_centroid_leakage"]) for v in sub_results.values()
            ),
            "min_observed_prose_alignment": min(
                float(v.metrics["prose_alignment"]) for v in sub_results.values()
            ),
            "max_residual_norm": max_res,
        },
        failure_reasons=reasons,
    )


def certify_s5_surgery_bundle(
    p_null: TensorLike | Mapping[str, TensorLike],
    delta_w: TensorLike | Mapping[str, TensorLike],
    k_preserved: TensorLike | Mapping[str, TensorLike],
    u_gate: TensorLike | Mapping[Any, TensorLike],
    specialist_centroids: Mapping[Any, Any] | Sequence[TensorLike] | TensorLike,
    prose_centroid: TensorLike | Mapping[Any, TensorLike],
    pre_state: Mapping[str, Any] | nn.Module,
    post_state: Mapping[str, Any] | nn.Module,
    anchor_layer_prefixes: Sequence[str] | set[str] | frozenset[str],
    stage_records: Sequence[Any] | Any,
    *,
    projector_tol: float = DEFAULT_PROJECTOR_TOL,
    key_tol: float = DEFAULT_KEY_INVARIANCE_TOL,
    centroid_tol: float = DEFAULT_CENTROID_INVARIANCE_TOL,
    min_prose_alignment: float = DEFAULT_PROSE_ALIGNMENT_FLOOR,
    general_guard_rel: float = DEFAULT_GENERAL_GUARD_REL,
    min_energy_descent: float = DEFAULT_MIN_ENERGY_DESCENT,
    maximum_pain: float = ANSE_MAXIMUM_PAIN,
    allow_rolled_back_stages: bool = True,
    surgery_id: str = "s5_pass_synergy",
    metadata: Mapping[str, Any] | None = None,
) -> WSPFormalCertificate:
    """Aggregate all 5 S5 WSP proof obligations + monotone ANSE descent into a SHA-256 certificate."""
    s_id = _validate_nonempty_str(surgery_id, "surgery_id")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError(f"metadata must be a Mapping or None, got {type(metadata).__name__}")
    frozen_meta = _freeze_json_value(dict(metadata or {}))

    ob_proj = _aggregate_projector_obligation(p_null, tol=projector_tol)
    ob_key = _aggregate_preserved_key_obligation(delta_w, k_preserved, tol=key_tol)
    ob_gate = _aggregate_multigate_obligation(
        u_gate,
        specialist_centroids,
        prose_centroid,
        tol=centroid_tol,
        min_prose_alignment=min_prose_alignment,
    )
    ob_anchor = verify_anchor_layer_bitwise_immutability(
        pre_state,
        post_state,
        anchor_layer_prefixes,
    )
    ob_anse = verify_monotone_anse_stage_descent(
        stage_records,
        min_descent=min_energy_descent,
        general_guard_rel=general_guard_rel,
        maximum_pain=maximum_pain,
        allow_rolled_back_stages=allow_rolled_back_stages,
        require_at_least_one_descent=True,
    )

    ordered_obs: dict[str, ObligationCheckResult] = {
        "projector_idempotence_and_symmetry": ob_proj,
        "preserved_key_invariance": ob_key,
        "orthogonal_multigate_centroid_invariance": ob_gate,
        "anchor_layer_bitwise_immutability": ob_anchor,
        "monotone_anse_stage_descent": ob_anse,
    }
    failed = tuple(k for k in OBLIGATION_NAMES if not ordered_obs[k].passed)
    passed_count = len(OBLIGATION_NAMES) - len(failed)
    certified = len(failed) == 0

    pre_sha = compute_state_sha256(pre_state)
    post_map = _extract_state_mapping(post_state, "post_state")
    post_all_finite = all(
        _tensor_raw_bytes_and_meta(v, k)[5] for k, v in post_map.items()
    )
    post_sha = compute_state_sha256(post_map) if post_all_finite else ""

    max_residual = max(ob.max_residual_norm for ob in ordered_obs.values())
    anse_energy = float(ob_anse.metrics["anse_energy"])
    initial_anse_energy = float(ob_anse.metrics["initial_anse_energy"])
    energy_delta = float(ob_anse.metrics["energy_delta"])
    general_drift = float(ob_anse.metrics["general_rel_drift"])
    anchor_pre_sha = str(ob_anchor.metrics["anchor_pre_sha256"])
    anchor_post_sha = str(ob_anchor.metrics["anchor_post_sha256"])
    anchor_mutated_bits = int(ob_anchor.metrics["mutated_bits"])

    core_doc = {
        "schema": SCHEMA_VERSION,
        "object": CERTIFICATE_OBJECT,
        "surgery_id": s_id,
        "certified": certified,
        "passed_obligations": passed_count,
        "total_obligations": len(OBLIGATION_NAMES),
        "failed_obligations": list(failed),
        "obligations": {k: v.to_dict() for k, v in ordered_obs.items()},
        "max_residual_norm": max_residual,
        "anse_energy": anse_energy,
        "initial_anse_energy": initial_anse_energy,
        "energy_delta": energy_delta,
        "general_rel_drift": general_drift,
        "pre_state_sha256": pre_sha,
        "post_state_sha256": post_sha,
        "anchor_pre_sha256": anchor_pre_sha,
        "anchor_post_sha256": anchor_post_sha,
        "anchor_mutated_bits": anchor_mutated_bits,
        "metadata": _unfreeze_json_value(frozen_meta),
    }
    receipt_sha = compute_payload_sha256(core_doc)

    return WSPFormalCertificate(
        schema=SCHEMA_VERSION,
        object=CERTIFICATE_OBJECT,
        surgery_id=s_id,
        certified=certified,
        passed_obligations=passed_count,
        total_obligations=len(OBLIGATION_NAMES),
        failed_obligations=failed,
        obligations=MappingProxyType(ordered_obs),
        max_residual_norm=max_residual,
        anse_energy=anse_energy,
        initial_anse_energy=initial_anse_energy,
        energy_delta=energy_delta,
        general_rel_drift=general_drift,
        pre_state_sha256=pre_sha,
        post_state_sha256=post_sha,
        anchor_pre_sha256=anchor_pre_sha,
        anchor_post_sha256=anchor_post_sha,
        anchor_mutated_bits=anchor_mutated_bits,
        metadata=frozen_meta,
        receipt_sha256=receipt_sha,
    )

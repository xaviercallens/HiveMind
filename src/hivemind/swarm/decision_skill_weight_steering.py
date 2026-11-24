# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Decision / Energy-Based Rank-1 Weight Steering for Training-Free Skill Injection.

Implements the third pillar of Weight-Space Programming (WSP) for open-weight
transformers and Sparse MoE adapters (``CrowdV1WithEngram``, ``CrowdRuntimeMLP``,
``CrowdLoRALinear``, and ``CrowdLoRAMoEMLP`` / ``CrowdLoRAExpertAdapter``):
steering raw weights for targeted capabilities such as structured tool-calling
delimiters and JSON schema generation **without gradient descent**:

1. ``compute_contrastive_skill_direction(pos_hidden, neg_hidden, preserve_hidden, ...)``:
   Extracts a ridge-whitened skill-inducing direction ``v_in`` inside the null-space
   of preserved general prose and specialist math/code activations (with 2-fold
   out-of-fold SNR gating).
2. ``search_decision_energy_rank1_edit(module_or_lora, skill_dir, unembed_target_dir, eval_energy_fn, ...)``:
   Evaluates a closed-form grid of rank-1 parameter updates
   ``Delta W = alpha * u_out @ v_in^T`` (strictly bounded to ``< 0.02%`` norm and
   effective-rank perturbation) under an objective ANSE Decision Energy ``E(alpha)``
   combining tool-call delimiter/schema NLL gain with Maximum Pain (``1e6``) on
   any ``> +1.0%`` general-domain or guarded-domain NLL regression. Supports both
   dense layers and rank-preserving LoRA factor-space fold-in without increasing ``r``.
3. ``apply_verified_skill_edit_inplace(...)``:
   Commits the energy-minimizing rank-1 edit iff ``Delta E < 0`` and all domain
   guards hold, otherwise performs atomic bitwise rollback with cryptographic
   SHA-256 weight-state and receipt provenance.
4. ``make_skill_energy_eval_fn(...)`` and ``inject_decision_skill_into_model_inplace(...)``:
   End-to-end training-free causal NLL energy evaluators and multi-layer model
   orchestrators for dense and Sparse MoE open-weight models.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.swarm.lora_svd_mode_surgery import decompose_lora_qr_svd
from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    second_moment_and_mean,
)

MAX_PERTURBATION_RATIO_CEILING: float = 0.0002
DEFAULT_MAX_NORM_PERTURBATION_RATIO: float = 0.0002
DEFAULT_MAX_EFFECTIVE_RANK_PERTURBATION_RATIO: float = 0.0002
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_MAX_GENERAL_DRIFT: float = 0.01
DEFAULT_SAFE_DRIFT_THRESHOLD: float = 0.0085
DEFAULT_ANSE_MAX_PAIN: float = 1_000_000.0
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_DRIFT_PENALTY_WEIGHT: float = 1.0
DEFAULT_RIDGE_SCALE: float = 0.10
DEFAULT_PRESERVE_WEIGHT: float = 10.0
DEFAULT_NULLSPACE_EIGEN_RATIO: float = 1e-6
DEFAULT_MIN_CONTRAST_SNR: float = 0.5
DEFAULT_PERTURBATION_FRACTIONS: tuple[float, ...] = (
    0.05,
    0.10,
    0.20,
    0.35,
    0.50,
    0.70,
    0.85,
    0.95,
)
_VALID_NULLSPACE_MODES: frozenset[str] = frozenset({"auto", "strict", "soft"})
_VALID_LORA_FOLD_MODES: frozenset[str] = frozenset({"auto", "qr_svd", "update_a", "update_b"})
_VALID_SWIGLU_ATTRS: frozenset[str] = frozenset({"w1", "w2", "w3"})


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------


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
        raise ValueError(f"{name} must be finite and >= 0.0, got {value!r}")
    return out


def _validate_positive_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite strictly positive real number."""
    out = _validate_nonneg_finite(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
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


def _validate_perturbation_ratio(value: Any, name: str) -> float:
    """Enforce ``0.0 < ratio <= MAX_PERTURBATION_RATIO_CEILING`` (<0.02% bound)."""
    ratio = _validate_positive_finite(value, name)
    if ratio > MAX_PERTURBATION_RATIO_CEILING:
        raise ValueError(
            f"{name} must be <= {MAX_PERTURBATION_RATIO_CEILING}, got {ratio}"
        )
    return ratio


def _validate_hidden_2d(tensor: Any, name: str) -> torch.Tensor:
    """Validate a 2D ``(N, D)`` or 3D ``(B, T, D)`` finite tensor as CPU ``float64`` 2D."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim not in (2, 3) or any(dim == 0 for dim in tensor.shape):
        raise ValueError(
            f"{name} must be a non-empty 2D or 3D tensor, got shape {tuple(tensor.shape)}"
        )
    flat = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1, int(tensor.shape[-1]))
    if not bool(torch.isfinite(flat).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return flat.contiguous()


def _validate_direction_1d(tensor: Any, expected_dim: int, name: str) -> torch.Tensor:
    """Validate a finite 1D direction tensor of length ``expected_dim`` in CPU ``float64``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 1 or int(tensor.shape[0]) != expected_dim:
        raise ValueError(
            f"{name} must be a 1D tensor of length {expected_dim}, got shape {tuple(tensor.shape)}"
        )
    vec = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(vec).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return vec


def _validate_token_id_list(
    token_ids: Sequence[int] | torch.Tensor,
    vocab_size: int,
    name: str,
) -> list[int]:
    """Validate and deduplicate token indices within ``[0, vocab_size)``."""
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim != 1:
            raise ValueError(f"{name} tensor must be 1D, got shape {tuple(token_ids.shape)}")
        if token_ids.is_floating_point() or token_ids.dtype == torch.bool:
            raise TypeError(f"{name} tensor must have an integer dtype")
        raw = [int(t) for t in token_ids.tolist()]
    elif isinstance(token_ids, Sequence) and not isinstance(token_ids, (str, bytes, bytearray)):
        raw = [_validate_nonneg_int(t, name) for t in token_ids]
    else:
        raise TypeError(f"{name} must be a sequence or 1D integer tensor")

    unique_ids = sorted(set(raw))
    if not unique_ids:
        raise ValueError(f"{name} must not be empty")
    if any(tok < 0 or tok >= vocab_size for tok in unique_ids):
        raise ValueError(f"{name} entries must lie in [0, {vocab_size})")
    return unique_ids


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionSkillSteeringConfig:
    """Configuration governing decision-energy rank-1 skill steering and rollback."""

    ridge_scale: float = DEFAULT_RIDGE_SCALE
    preserve_weight: float = DEFAULT_PRESERVE_WEIGHT
    nullspace_eigen_ratio: float = DEFAULT_NULLSPACE_EIGEN_RATIO
    nullspace_mode: str = "auto"
    min_contrast_snr: float = DEFAULT_MIN_CONTRAST_SNR
    use_covariance_whitening: bool = True
    max_norm_perturbation_ratio: float = DEFAULT_MAX_NORM_PERTURBATION_RATIO
    max_effective_rank_perturbation_ratio: float = DEFAULT_MAX_EFFECTIVE_RANK_PERTURBATION_RATIO
    enforce_effective_rank_bound: bool = True
    max_general_drift: float = DEFAULT_MAX_GENERAL_DRIFT
    safe_drift_threshold: float = DEFAULT_SAFE_DRIFT_THRESHOLD
    drift_penalty_weight: float = DEFAULT_DRIFT_PENALTY_WEIGHT
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    max_pain_energy: float = DEFAULT_ANSE_MAX_PAIN
    lora_fold_mode: str = "auto"

    def __post_init__(self) -> None:
        _validate_nonneg_finite(self.ridge_scale, "ridge_scale")
        _validate_nonneg_finite(self.preserve_weight, "preserve_weight")
        null_r = _validate_nonneg_finite(self.nullspace_eigen_ratio, "nullspace_eigen_ratio")
        if null_r >= 1.0:
            raise ValueError(f"nullspace_eigen_ratio must be < 1.0, got {null_r}")
        if self.nullspace_mode not in _VALID_NULLSPACE_MODES:
            raise ValueError(
                f"nullspace_mode must be one of {sorted(_VALID_NULLSPACE_MODES)}, got {self.nullspace_mode!r}"
            )
        _validate_nonneg_finite(self.min_contrast_snr, "min_contrast_snr")
        if not isinstance(self.use_covariance_whitening, bool):
            raise TypeError(
                f"use_covariance_whitening must be a bool, got {type(self.use_covariance_whitening).__name__}"
            )
        _validate_perturbation_ratio(self.max_norm_perturbation_ratio, "max_norm_perturbation_ratio")
        _validate_perturbation_ratio(
            self.max_effective_rank_perturbation_ratio,
            "max_effective_rank_perturbation_ratio",
        )
        if not isinstance(self.enforce_effective_rank_bound, bool):
            raise TypeError(
                f"enforce_effective_rank_bound must be a bool, got {type(self.enforce_effective_rank_bound).__name__}"
            )
        max_drift = _validate_nonneg_finite(self.max_general_drift, "max_general_drift")
        safe_drift = _validate_nonneg_finite(self.safe_drift_threshold, "safe_drift_threshold")
        if safe_drift > max_drift:
            raise ValueError(
                f"safe_drift_threshold ({safe_drift}) must be <= max_general_drift ({max_drift})"
            )
        _validate_nonneg_finite(self.drift_penalty_weight, "drift_penalty_weight")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_positive_finite(self.max_pain_energy, "max_pain_energy")
        if self.lora_fold_mode not in _VALID_LORA_FOLD_MODES:
            raise ValueError(
                f"lora_fold_mode must be one of {sorted(_VALID_LORA_FOLD_MODES)}, got {self.lora_fold_mode!r}"
            )


# ---------------------------------------------------------------------------
# Target Resolution, Bitwise Snapshot, Restore & SHA-256 Provenance
# ---------------------------------------------------------------------------


@dataclass
class _TargetHandle:
    """Resolved internal descriptor for dense weights or rank-r LoRA factor pairs."""

    kind: str
    d_out: int
    d_in: int
    dense_tensor: torch.Tensor | None
    lora_b: torch.Tensor | None
    lora_a: torch.Tensor | None
    scaling: float
    rank: int
    base_tensor: torch.Tensor | None
    mutable_tensors: dict[str, torch.Tensor]
    hash_tensors: dict[str, torch.Tensor]


def _validate_2d_weight_tensor(tensor: Any, name: str) -> torch.Tensor:
    """Verify a finite non-empty 2D floating-point tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must resolve to a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D tensor, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    if not bool(torch.isfinite(tensor.detach()).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _resolve_lora_pair_from_module(
    mod: nn.Module,
    prefix: str,
    *,
    base_tensor_override: torch.Tensor | None = None,
) -> _TargetHandle:
    """Build a LoRA ``_TargetHandle`` from ``mod`` exposing ``{prefix}_B`` and ``{prefix}_A``."""
    b_attr = f"{prefix}_B"
    a_attr = f"{prefix}_A"
    w_b = getattr(mod, b_attr, None)
    w_a = getattr(mod, a_attr, None)
    if isinstance(w_b, nn.Linear):
        w_b = w_b.weight
    if isinstance(w_a, nn.Linear):
        w_a = w_a.weight
    w_b_t = _validate_2d_weight_tensor(w_b, b_attr)
    w_a_t = _validate_2d_weight_tensor(w_a, a_attr)
    if w_b_t.shape[1] != w_a_t.shape[0]:
        raise ValueError(
            f"LoRA inner rank mismatch between {b_attr} {tuple(w_b_t.shape)} and {a_attr} {tuple(w_a_t.shape)}"
        )
    scaling_val = float(getattr(mod, "scaling", 1.0))
    if not math.isfinite(scaling_val) or scaling_val <= 0.0:
        raise ValueError(f"LoRA scaling must be finite and > 0.0, got {scaling_val!r}")
    rank_val = int(w_a_t.shape[0])

    base_tensor: torch.Tensor | None = base_tensor_override
    base_mod = getattr(mod, "base", None)
    if base_tensor is None and isinstance(base_mod, nn.Linear):
        base_tensor = _validate_2d_weight_tensor(base_mod.weight, "base.weight")
    if base_tensor is not None and base_tensor.shape != (int(w_b_t.shape[0]), int(w_a_t.shape[1])):
        raise ValueError(
            f"LoRA base.weight shape {tuple(base_tensor.shape)} does not match "
            f"({w_b_t.shape[0]}, {w_a_t.shape[1]})"
        )

    mutable = {b_attr: w_b_t, a_attr: w_a_t}
    hashed = dict(mutable)
    if base_tensor is not None:
        hashed["base.weight"] = base_tensor
    return _TargetHandle(
        kind="lora",
        d_out=int(w_b_t.shape[0]),
        d_in=int(w_a_t.shape[1]),
        dense_tensor=None,
        lora_b=w_b_t,
        lora_a=w_a_t,
        scaling=scaling_val,
        rank=rank_val,
        base_tensor=base_tensor,
        mutable_tensors=mutable,
        hash_tensors=hashed,
    )


def _dense_handle(w_t: torch.Tensor, key: str = "weight") -> _TargetHandle:
    """Construct a dense ``_TargetHandle`` around a validated 2D weight tensor."""
    return _TargetHandle(
        kind="dense",
        d_out=int(w_t.shape[0]),
        d_in=int(w_t.shape[1]),
        dense_tensor=w_t,
        lora_b=None,
        lora_a=None,
        scaling=1.0,
        rank=0,
        base_tensor=None,
        mutable_tensors={key: w_t},
        hash_tensors={key: w_t},
    )


def _resolve_target_handle(
    module_or_lora: Any,
    *,
    weight_attr: str | None = None,
    expert_idx: int | None = None,
) -> _TargetHandle:
    """Resolve dense weights, ``CrowdRuntimeMLP``, ``CrowdLoRALinear``, or ``CrowdLoRAMoEMLP``."""
    if expert_idx is not None:
        e_int = _validate_nonneg_int(expert_idx, "expert_idx")
        experts = getattr(module_or_lora, "experts", None)
        if not isinstance(module_or_lora, nn.Module) or experts is None:
            raise ValueError("expert_idx requires an MoE nn.Module exposing 'experts'")
        e_key = str(e_int)
        if e_key not in experts:
            raise ValueError(f"expert_idx {e_int} not found in module experts")
        prefix = (weight_attr or "w3").strip()
        if not prefix:
            raise ValueError("weight_attr must be a non-empty string when provided")
        base_t: torch.Tensor | None = None
        base_mlp = getattr(module_or_lora, "base_mlp", None)
        if isinstance(base_mlp, nn.Module):
            base_proj = getattr(base_mlp, prefix, None)
            if isinstance(base_proj, nn.Linear):
                base_t = _validate_2d_weight_tensor(base_proj.weight, f"base_mlp.{prefix}.weight")
        return _resolve_lora_pair_from_module(experts[e_key], prefix, base_tensor_override=base_t)

    if isinstance(module_or_lora, torch.Tensor):
        if weight_attr is not None:
            raise TypeError("weight_attr can only be used when module_or_lora is an nn.Module")
        return _dense_handle(_validate_2d_weight_tensor(module_or_lora, "module_or_lora"))

    if not isinstance(module_or_lora, nn.Module):
        raise TypeError(
            f"module_or_lora must be an nn.Module or 2D torch.Tensor, got {type(module_or_lora).__name__}"
        )

    if weight_attr is not None:
        if not isinstance(weight_attr, str) or not weight_attr.strip():
            raise ValueError("weight_attr must be a non-empty string when provided")
        attr_clean = weight_attr.strip()
        if hasattr(module_or_lora, f"{attr_clean}_A") and hasattr(module_or_lora, f"{attr_clean}_B"):
            return _resolve_lora_pair_from_module(module_or_lora, attr_clean)
        sub_obj = getattr(module_or_lora, attr_clean, None)
        if sub_obj is None and hasattr(module_or_lora, "base_mlp"):
            sub_obj = getattr(module_or_lora.base_mlp, attr_clean, None)
        if sub_obj is None:
            raise ValueError(f"{type(module_or_lora).__name__} has no attribute {attr_clean!r}")
        if isinstance(sub_obj, nn.Module):
            return _resolve_target_handle(sub_obj)
        return _dense_handle(_validate_2d_weight_tensor(sub_obj, attr_clean), key=attr_clean)

    # Auto-detect module layout
    if hasattr(module_or_lora, "lora_A") and hasattr(module_or_lora, "lora_B"):
        return _resolve_lora_pair_from_module(module_or_lora, "lora")
    if hasattr(module_or_lora, "w3_A") and hasattr(module_or_lora, "w3_B"):
        return _resolve_lora_pair_from_module(module_or_lora, "w3")
    if isinstance(module_or_lora, nn.Linear):
        return _dense_handle(_validate_2d_weight_tensor(module_or_lora.weight, "weight"))
    if hasattr(module_or_lora, "base_mlp") and isinstance(getattr(module_or_lora, "base_mlp"), nn.Module):
        return _resolve_target_handle(module_or_lora.base_mlp, weight_attr="w3")
    if hasattr(module_or_lora, "w3"):
        return _resolve_target_handle(module_or_lora.w3)
    if hasattr(module_or_lora, "weight") and isinstance(getattr(module_or_lora, "weight"), torch.Tensor):
        return _dense_handle(_validate_2d_weight_tensor(getattr(module_or_lora, "weight"), "weight"))
    raise ValueError(
        f"{type(module_or_lora).__name__} must expose a 2D weight tensor or rank-r LoRA pair"
    )


def _raw_bit_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return an integer bit-pattern view for exact bitwise hashing and comparisons."""
    t_cpu = tensor.detach().cpu().contiguous()
    elem_bytes = t_cpu.element_size()
    if elem_bytes == 1:
        return t_cpu.view(torch.uint8)
    if elem_bytes == 2:
        return t_cpu.view(torch.int16)
    if elem_bytes == 4:
        return t_cpu.view(torch.int32)
    return t_cpu.view(torch.int64)


def _hash_tensor_map(tensor_map: Mapping[str, torch.Tensor]) -> str:
    """Compute deterministic SHA-256 digest over raw tensor bits in sorted key order."""
    hasher = hashlib.sha256()
    for name in sorted(tensor_map.keys()):
        t = tensor_map[name].detach().cpu().contiguous()
        header = f"{name}|{tuple(t.shape)}|{t.dtype}|".encode()
        hasher.update(header)
        hasher.update(_raw_bit_view(t).reshape(-1).view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def compute_target_state_sha256(
    module_or_lora: nn.Module | torch.Tensor,
    *,
    weight_attr: str | None = None,
    expert_idx: int | None = None,
) -> str:
    """Compute deterministic SHA-256 digest over raw bits of ``module_or_lora`` target tensors."""
    handle = _resolve_target_handle(module_or_lora, weight_attr=weight_attr, expert_idx=expert_idx)
    return _hash_tensor_map(handle.hash_tensors)


@torch.no_grad()
def snapshot_target_state(
    module_or_lora: nn.Module | torch.Tensor,
    *,
    weight_attr: str | None = None,
    expert_idx: int | None = None,
) -> dict[str, torch.Tensor]:
    """Create an atomic bitwise clone of target mutable tensors (plus frozen base check if any)."""
    handle = _resolve_target_handle(module_or_lora, weight_attr=weight_attr, expert_idx=expert_idx)
    return {k: t.detach().clone() for k, t in handle.hash_tensors.items()}


@torch.no_grad()
def restore_target_state_inplace(
    module_or_lora: nn.Module | torch.Tensor,
    snapshot: Mapping[str, torch.Tensor],
    *,
    weight_attr: str | None = None,
    expert_idx: int | None = None,
) -> str:
    """Atomically restore target tensors from ``snapshot`` and return verified SHA-256 digest."""
    if not isinstance(snapshot, Mapping) or not snapshot:
        raise ValueError("snapshot must be a non-empty mapping of tensor_key -> Tensor")
    handle = _resolve_target_handle(module_or_lora, weight_attr=weight_attr, expert_idx=expert_idx)
    if set(handle.hash_tensors.keys()) != set(snapshot.keys()):
        raise ValueError("snapshot keys do not match target state layout")
    for key, target in handle.mutable_tensors.items():
        src = snapshot[key]
        if target.shape != src.shape or target.dtype != src.dtype:
            raise ValueError(
                f"Snapshot mismatch on {key!r}: expected {tuple(target.shape)}/{target.dtype}, "
                f"got {tuple(src.shape)}/{src.dtype}"
            )
        target.copy_(src.to(device=target.device, dtype=target.dtype))
    return _hash_tensor_map(handle.hash_tensors)


# ---------------------------------------------------------------------------
# (a) Contrastive Ridge-Whitened Skill Direction in Preserve Null-Space
# ---------------------------------------------------------------------------


def _normalize_preserve_hidden(
    preserve_hidden: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor] | None,
    expected_dim: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build equal-domain-averaged preserve second moment ``M_pres`` and stacked matrix."""
    if preserve_hidden is None:
        return torch.zeros((expected_dim, expected_dim), dtype=torch.float64), None

    tensors: list[torch.Tensor] = []
    if isinstance(preserve_hidden, torch.Tensor):
        tensors.append(_validate_hidden_2d(preserve_hidden, "preserve_hidden"))
    elif isinstance(preserve_hidden, Mapping):
        if not preserve_hidden:
            raise ValueError("preserve_hidden mapping must not be empty")
        for key in sorted(preserve_hidden.keys(), key=str):
            tensors.append(_validate_hidden_2d(preserve_hidden[key], f"preserve_hidden[{key!r}]"))
    elif isinstance(preserve_hidden, Sequence) and not isinstance(preserve_hidden, (str, bytes, bytearray)):
        if not preserve_hidden:
            raise ValueError("preserve_hidden sequence must not be empty")
        for idx, item in enumerate(preserve_hidden):
            tensors.append(_validate_hidden_2d(item, f"preserve_hidden[{idx}]"))
    else:
        raise TypeError(
            "preserve_hidden must be None, a Tensor, a sequence of Tensors, or a mapping of Tensors"
        )

    for idx, t in enumerate(tensors):
        if int(t.shape[1]) != expected_dim:
            raise ValueError(
                f"preserve_hidden[{idx}] width {t.shape[1]} must match pos_hidden width {expected_dim}"
            )

    m_acc = torch.zeros((expected_dim, expected_dim), dtype=torch.float64)
    for t in tensors:
        m_k, _ = second_moment_and_mean(t)
        m_acc = m_acc + m_k
    m_pres = 0.5 * (m_acc / float(len(tensors)) + (m_acc / float(len(tensors))).T)
    stacked = torch.cat(tensors, dim=0).contiguous()
    return m_pres.contiguous(), stacked


def _compute_nullspace_basis(
    m_pres: torch.Tensor,
    nullspace_eigen_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return ``(P_null, V_null, lam_max)`` for the kernel of ``m_pres`` in CPU ``float64``."""
    d = int(m_pres.shape[0])
    evals, evecs = torch.linalg.eigh(m_pres)
    lam_max = float(evals.max().item())
    if lam_max <= 1e-24:
        v_null = torch.eye(d, dtype=torch.float64)
        return v_null.clone(), v_null, 0.0

    tol = max(float(nullspace_eigen_ratio) * lam_max, 1e-15 * lam_max)
    null_mask = evals <= tol
    if not bool(null_mask.any().item()):
        return (
            torch.zeros((d, d), dtype=torch.float64),
            torch.zeros((d, 0), dtype=torch.float64),
            lam_max,
        )

    v_null = evecs[:, null_mask].contiguous()
    p_null = (v_null @ v_null.T).contiguous()
    return p_null, v_null, lam_max


def _within_class_cov(x_2d: torch.Tensor) -> torch.Tensor:
    """Compute centered empirical covariance matrix ``(D, D)`` in ``float64``."""
    n = int(x_2d.shape[0])
    d = int(x_2d.shape[1])
    if n <= 1:
        return torch.zeros((d, d), dtype=torch.float64)
    xc = x_2d - x_2d.mean(dim=0, keepdim=True)
    cov = (xc.T @ xc) / float(n)
    return 0.5 * (cov + cov.T)


def _solve_ridge_contrastive_vector(
    x_pos: torch.Tensor,
    x_neg: torch.Tensor,
    m_pres: torch.Tensor,
    v_null: torch.Tensor,
    *,
    ridge_scale: float,
    preserve_weight: float,
    nullspace_mode: str,
    use_covariance_whitening: bool,
) -> tuple[torch.Tensor, str, str | None]:
    """Solve ridge-whitened contrastive direction inside preserve null-space or soft complement."""
    d = int(x_pos.shape[1])
    delta_mu = x_pos.mean(dim=0) - x_neg.mean(dim=0)
    if float(delta_mu.norm().item()) <= 1e-14:
        return torch.zeros(d, dtype=torch.float64), "none", "zero_contrast_mean"

    sigma_w = (
        0.5 * (_within_class_cov(x_pos) + _within_class_cov(x_neg))
        if use_covariance_whitening
        else torch.zeros((d, d), dtype=torch.float64)
    )
    null_dim = int(v_null.shape[1])

    if nullspace_mode == "strict" and null_dim == 0:
        return torch.zeros(d, dtype=torch.float64), "strict_rejected", "empty_preserve_nullspace"

    if nullspace_mode in {"auto", "strict"} and null_dim > 0:
        delta_null = v_null.T @ delta_mu
        if float(delta_null.norm().item()) <= 1e-13:
            return torch.zeros(d, dtype=torch.float64), "exact_nullspace", "zero_nullspace_signal"
        sigma_null = v_null.T @ sigma_w @ v_null
        sigma_null = 0.5 * (sigma_null + sigma_null.T)
        tr_sigma = float(torch.diagonal(sigma_null).mean().item())
        pos_null = x_pos @ v_null
        tr_pos = float((pos_null * pos_null).sum(dim=1).mean().item()) / float(max(1, null_dim))
        tr_scale = max(tr_sigma, tr_pos, 1e-6)
        lam = max(ridge_scale * tr_scale, 1e-6)
        reg_mat = sigma_null + lam * torch.eye(null_dim, dtype=torch.float64)
        w_null = torch.linalg.solve(reg_mat, delta_null)
        v_raw = (v_null @ w_null).contiguous()
        mode_used = "exact_nullspace"
    else:
        comb = sigma_w + preserve_weight * m_pres
        comb = 0.5 * (comb + comb.T)
        tr_comb = float(torch.diagonal(comb).mean().item())
        tr_pos = float((x_pos * x_pos).sum(dim=1).mean().item()) / float(max(1, d))
        tr_scale = max(tr_comb, tr_pos, 1e-6)
        lam = max(ridge_scale * tr_scale, 1e-6)
        reg_mat = comb + lam * torch.eye(d, dtype=torch.float64)
        v_raw = torch.linalg.solve(reg_mat, delta_mu).contiguous()
        mode_used = "soft_preserve_whitened"

    if float((delta_mu @ v_raw).item()) < 0.0:
        v_raw = -v_raw
    return v_raw, mode_used, None


def _compute_snr(z_pos: torch.Tensor, z_neg: torch.Tensor) -> float:
    """Compute directional signal-to-noise ratio between positive and negative projections."""
    diff = float(z_pos.mean().item()) - float(z_neg.mean().item())
    if diff <= 0.0:
        return 0.0
    var_p = float(z_pos.var(unbiased=False).item()) if z_pos.numel() > 1 else 0.0
    var_n = float(z_neg.var(unbiased=False).item()) if z_neg.numel() > 1 else 0.0
    pooled_std = math.sqrt(0.5 * (var_p + var_n) + 1e-10)
    return diff / pooled_std


@torch.no_grad()
def compute_contrastive_skill_direction(
    pos_hidden: torch.Tensor,
    neg_hidden: torch.Tensor,
    preserve_hidden: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor] | None = None,
    *,
    ridge_scale: float = DEFAULT_RIDGE_SCALE,
    preserve_weight: float = DEFAULT_PRESERVE_WEIGHT,
    nullspace_eigen_ratio: float = DEFAULT_NULLSPACE_EIGEN_RATIO,
    nullspace_mode: str = "auto",
    min_contrast_snr: float = DEFAULT_MIN_CONTRAST_SNR,
    use_covariance_whitening: bool = True,
) -> dict[str, Any]:
    """Extract a ridge-whitened skill-inducing direction in the null-space of preserve states.

    Given positive skill activations ``pos_hidden`` and contrastive baseline activations
    ``neg_hidden`` (and optional general/specialist ``preserve_hidden`` activations):

    1. Forms the equal-domain uncentered second moment ``M_pres`` of ``preserve_hidden``.
    2. Extracts the exact kernel basis ``V_null`` of ``M_pres`` (eigenvalues
       ``<= nullspace_eigen_ratio * lam_max``).
    3. Solves the ridge-whitened Fisher contrastive system
       ``(Sigma_within + lam * I) w = V_null^T (mu_pos - mu_neg)`` strictly inside
       ``V_null`` (or falls back to ``(Sigma_within + preserve_weight * M_pres + lam * I)``
       when ``nullspace_mode='soft'`` or when ``nullspace_mode='auto'`` encounters a
       full-rank preserve covariance).
    4. Gates reliability via 2-fold out-of-fold contrast SNR against ``min_contrast_snr``.
    """
    pos_2d = _validate_hidden_2d(pos_hidden, "pos_hidden")
    neg_2d = _validate_hidden_2d(neg_hidden, "neg_hidden")
    d = int(pos_2d.shape[1])
    if int(neg_2d.shape[1]) != d:
        raise ValueError(
            f"neg_hidden width {neg_2d.shape[1]} must match pos_hidden width {d}"
        )

    ridge_f = _validate_nonneg_finite(ridge_scale, "ridge_scale")
    pres_w = _validate_nonneg_finite(preserve_weight, "preserve_weight")
    null_r = _validate_nonneg_finite(nullspace_eigen_ratio, "nullspace_eigen_ratio")
    if null_r >= 1.0:
        raise ValueError(f"nullspace_eigen_ratio must be < 1.0, got {null_r}")
    if nullspace_mode not in _VALID_NULLSPACE_MODES:
        raise ValueError(
            f"nullspace_mode must be one of {sorted(_VALID_NULLSPACE_MODES)}, got {nullspace_mode!r}"
        )
    snr_thresh = _validate_nonneg_finite(min_contrast_snr, "min_contrast_snr")
    if not isinstance(use_covariance_whitening, bool):
        raise TypeError(
            f"use_covariance_whitening must be a bool, got {type(use_covariance_whitening).__name__}"
        )

    m_pres, stacked_pres = _normalize_preserve_hidden(preserve_hidden, d)
    p_null, v_null, lam_max = _compute_nullspace_basis(m_pres, null_r)
    null_dim = int(v_null.shape[1])

    v_raw, mode_used, noop_reason = _solve_ridge_contrastive_vector(
        pos_2d,
        neg_2d,
        m_pres,
        v_null,
        ridge_scale=ridge_f,
        preserve_weight=pres_w,
        nullspace_mode=nullspace_mode,
        use_covariance_whitening=use_covariance_whitening,
    )

    raw_norm = float(v_raw.norm().item())
    if raw_norm <= 1e-12 or noop_reason is not None:
        v_unit = torch.zeros(d, dtype=torch.float64)
        return {
            "direction": v_unit,
            "raw_direction": torch.zeros(d, dtype=torch.float64),
            "feature_dim": d,
            "nullspace_dim": null_dim,
            "nullspace_mode_used": mode_used,
            "preserve_max_eigenvalue": lam_max,
            "oof_snr": 0.0,
            "full_snr": 0.0,
            "contrast_mean_diff": 0.0,
            "passes_snr_threshold": False,
            "preserve_rms": 0.0,
            "preserve_max_abs": 0.0,
            "m_preserve": m_pres,
            "p_null": p_null,
            "noop_reason": noop_reason or "zero_direction_norm",
        }

    v_unit = (v_raw / raw_norm).contiguous()

    # 2-fold out-of-fold SNR validation on the same preserve null-space basis
    if int(pos_2d.shape[0]) >= 4 and int(neg_2d.shape[0]) >= 4:
        v0_raw, _, _ = _solve_ridge_contrastive_vector(
            pos_2d[0::2],
            neg_2d[0::2],
            m_pres,
            v_null,
            ridge_scale=ridge_f,
            preserve_weight=pres_w,
            nullspace_mode=nullspace_mode,
            use_covariance_whitening=use_covariance_whitening,
        )
        v1_raw, _, _ = _solve_ridge_contrastive_vector(
            pos_2d[1::2],
            neg_2d[1::2],
            m_pres,
            v_null,
            ridge_scale=ridge_f,
            preserve_weight=pres_w,
            nullspace_mode=nullspace_mode,
            use_covariance_whitening=use_covariance_whitening,
        )
        v0 = v0_raw / max(float(v0_raw.norm().item()), 1e-12)
        v1 = v1_raw / max(float(v1_raw.norm().item()), 1e-12)
        z_pos_oof = torch.cat([pos_2d[1::2] @ v0, pos_2d[0::2] @ v1], dim=0)
        z_neg_oof = torch.cat([neg_2d[1::2] @ v0, neg_2d[0::2] @ v1], dim=0)
        oof_snr = _compute_snr(z_pos_oof, z_neg_oof)
    else:
        oof_snr = _compute_snr(pos_2d @ v_unit, neg_2d @ v_unit)

    z_pos_full = pos_2d @ v_unit
    z_neg_full = neg_2d @ v_unit
    full_snr = _compute_snr(z_pos_full, z_neg_full)
    contrast_diff = float(z_pos_full.mean().item()) - float(z_neg_full.mean().item())

    pres_rms = math.sqrt(max(0.0, float((v_unit @ (m_pres @ v_unit)).item())))
    pres_max = (
        float((stacked_pres @ v_unit).abs().max().item())
        if stacked_pres is not None
        else 0.0
    )
    passes_snr = oof_snr >= snr_thresh

    return {
        "direction": v_unit,
        "raw_direction": v_raw,
        "feature_dim": d,
        "nullspace_dim": null_dim,
        "nullspace_mode_used": mode_used,
        "preserve_max_eigenvalue": lam_max,
        "oof_snr": oof_snr,
        "full_snr": full_snr,
        "contrast_mean_diff": contrast_diff,
        "passes_snr_threshold": passes_snr,
        "preserve_rms": pres_rms,
        "preserve_max_abs": pres_max,
        "m_preserve": m_pres,
        "p_null": p_null,
        "noop_reason": None if passes_snr else "contrast_snr_below_threshold",
    }


# ---------------------------------------------------------------------------
# (b) Rank-1 Perturbation Bounds, LoRA Fold-In & Decision Energy Grid Search
# ---------------------------------------------------------------------------


def _resolve_skill_vector(
    skill_dir: torch.Tensor | Mapping[str, Any],
    expected_d_in: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Extract normalized ``v_in`` of length ``expected_d_in`` and optional ``p_null``."""
    p_null: torch.Tensor | None = None
    if isinstance(skill_dir, Mapping):
        if "direction" not in skill_dir:
            raise ValueError("skill_dir mapping must contain a 'direction' tensor")
        raw_v = _validate_direction_1d(skill_dir["direction"], expected_d_in, "skill_dir['direction']")
        if isinstance(skill_dir.get("p_null"), torch.Tensor):
            pn = skill_dir["p_null"].detach().to(device="cpu", dtype=torch.float64)
            if pn.shape == (expected_d_in, expected_d_in):
                p_null = pn
    else:
        raw_v = _validate_direction_1d(skill_dir, expected_d_in, "skill_dir")

    norm_v = float(raw_v.norm().item())
    if norm_v <= 1e-12:
        return torch.zeros(expected_d_in, dtype=torch.float64), p_null
    return (raw_v / norm_v).contiguous(), p_null


def _resolve_unembed_target_dir(
    unembed_target_dir: torch.Tensor | nn.Module | Mapping[str, Any],
    expected_d_out: int,
    *,
    target_token_ids: Sequence[int] | torch.Tensor | None = None,
    baseline_token_ids: Sequence[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Resolve unit output steering direction ``u_out`` in ``R^{d_out}``."""
    raw_obj: Any = unembed_target_dir
    tok_ids = target_token_ids
    base_ids = baseline_token_ids

    if isinstance(raw_obj, Mapping):
        if "u_out" in raw_obj:
            raw_obj = raw_obj["u_out"]
        elif "direction" in raw_obj:
            raw_obj = raw_obj["direction"]
        elif "unembed_weight" in raw_obj:
            if tok_ids is None and "target_token_ids" in raw_obj:
                tok_ids = raw_obj["target_token_ids"]
            if base_ids is None and "baseline_token_ids" in raw_obj:
                base_ids = raw_obj["baseline_token_ids"]
            raw_obj = raw_obj["unembed_weight"]
        else:
            raise ValueError(
                "unembed_target_dir mapping must contain 'u_out', 'direction', or 'unembed_weight'"
            )

    if isinstance(raw_obj, (nn.Linear, nn.Embedding)):
        raw_obj = raw_obj.weight
    elif isinstance(raw_obj, nn.Module):
        if not hasattr(raw_obj, "weight") or not isinstance(raw_obj.weight, torch.Tensor):
            raise ValueError(f"{type(raw_obj).__name__} must expose a 'weight' tensor")
        raw_obj = raw_obj.weight

    if not isinstance(raw_obj, torch.Tensor):
        raise TypeError(
            f"unembed_target_dir must resolve to a torch.Tensor, got {type(raw_obj).__name__}"
        )

    if raw_obj.ndim == 1:
        u_vec = _validate_direction_1d(raw_obj, expected_d_out, "unembed_target_dir")
    elif raw_obj.ndim == 2:
        mat = raw_obj.detach().to(device="cpu", dtype=torch.float64)
        if mat.shape[0] == 0 or int(mat.shape[1]) != expected_d_out:
            raise ValueError(
                f"unembed_target_dir matrix must have shape (V, {expected_d_out}), got {tuple(mat.shape)}"
            )
        if not bool(torch.isfinite(mat).all().item()):
            raise ValueError("unembed_target_dir must contain only finite values")
        vocab_size = int(mat.shape[0])
        if tok_ids is not None:
            pos_list = _validate_token_id_list(tok_ids, vocab_size, "target_token_ids")
            if base_ids is not None:
                neg_list = _validate_token_id_list(base_ids, vocab_size, "baseline_token_ids")
                u_vec = mat[pos_list].mean(dim=0) - mat[neg_list].mean(dim=0)
            else:
                contrast = torch.zeros(vocab_size, dtype=torch.float64)
                if len(pos_list) == vocab_size:
                    contrast.fill_(1.0 / float(vocab_size))
                else:
                    contrast.fill_(-1.0 / float(vocab_size - len(pos_list)))
                    contrast[torch.tensor(pos_list, dtype=torch.long)] = 1.0 / float(len(pos_list))
                u_vec = mat.T @ contrast
        else:
            u_vec = mat.mean(dim=0)
    else:
        raise ValueError(
            f"unembed_target_dir must be 1D or 2D, got {raw_obj.ndim}D tensor"
        )

    u_norm = float(u_vec.norm().item())
    if u_norm <= 1e-12:
        raise ValueError("Resolved unembed_target_dir vector has zero norm")
    return (u_vec / u_norm).contiguous()


def _effective_weight_f64(handle: _TargetHandle) -> torch.Tensor:
    """Compute effective dense weight matrix in CPU ``float64``."""
    if handle.kind == "dense":
        assert handle.dense_tensor is not None
        return handle.dense_tensor.detach().to(device="cpu", dtype=torch.float64)
    assert handle.lora_b is not None and handle.lora_a is not None
    b64 = handle.lora_b.detach().to(device="cpu", dtype=torch.float64)
    a64 = handle.lora_a.detach().to(device="cpu", dtype=torch.float64)
    eff = (b64 @ a64) * float(handle.scaling)
    if handle.base_tensor is not None:
        eff = eff + handle.base_tensor.detach().to(device="cpu", dtype=torch.float64)
    return eff


def _compute_effective_rank(mat_f64: torch.Tensor) -> float | None:
    """Compute Roy-Vetterli Shannon effective rank when ``min(d_out, d_in) <= 512``."""
    if min(int(mat_f64.shape[0]), int(mat_f64.shape[1])) > 512:
        return None
    s_vals = torch.linalg.svdvals(mat_f64)
    s_pos = s_vals[s_vals > 1e-12]
    if s_pos.numel() == 0:
        return 0.0
    probs = s_pos / s_pos.sum()
    return float(torch.exp(-(probs * probs.log()).sum()).item())


def _fold_rank1_into_lora_factors(
    b64: torch.Tensor,
    a64: torch.Tensor,
    u_out: torch.Tensor,
    v_in: torch.Tensor,
    alpha: float,
    scaling: float,
    *,
    lora_fold_mode: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Fold ``alpha * u_out @ v_in^T`` into LoRA ``(B, A)`` without increasing rank ``r``."""
    r = int(a64.shape[0])
    delta_s = float(alpha) / float(scaling)
    if abs(delta_s) <= 1e-18:
        return b64.clone(), a64.clone(), "zero_alpha_noop"

    # 1. Exact slot fill when B has an unused zero column (e.g. step-0 initialization)
    if lora_fold_mode == "auto":
        col_norms = b64.norm(dim=0)
        zero_cols = torch.nonzero(col_norms <= 1e-12, as_tuple=False).squeeze(-1)
        if zero_cols.numel() > 0:
            k_slot = int(zero_cols[0].item())
            mag = math.sqrt(abs(delta_s))
            sign = 1.0 if delta_s >= 0.0 else -1.0
            b_new = b64.clone()
            a_new = a64.clone()
            b_new[:, k_slot] = (sign * mag) * u_out
            a_new[k_slot, :] = mag * v_in
            return b_new, a_new, "zero_slot_exact"

    # 2. Check existing effective rank via QR-SVD
    _, s_orig, _ = decompose_lora_qr_svd(b64, a64)
    s_max_orig = float(s_orig.max().item()) if s_orig.numel() > 0 else 0.0
    eff_rank_orig = int((s_orig > 1e-10 * max(s_max_orig, 1e-12)).sum().item())

    def _qr_svd_fold() -> tuple[torch.Tensor, torch.Tensor]:
        mag = math.sqrt(abs(delta_s))
        sign = 1.0 if delta_s >= 0.0 else -1.0
        b_aug = torch.cat([b64, (sign * mag) * u_out.unsqueeze(1)], dim=1)
        a_aug = torch.cat([a64, mag * v_in.unsqueeze(0)], dim=0)
        u_r, s_r, v_r = decompose_lora_qr_svd(b_aug, a_aug)
        s_top = s_r[:r].clamp_min(0.0).sqrt()
        b_out = (u_r[:, :r] * s_top.unsqueeze(0)).contiguous()
        a_out = (s_top.unsqueeze(1) * v_r[:, :r].T).contiguous()
        return b_out, a_out

    if lora_fold_mode == "qr_svd" or eff_rank_orig < r:
        b_f, a_f = _qr_svd_fold()
        method = "qr_svd_exact" if eff_rank_orig < r else "qr_svd_truncated"
        return b_f, a_f, method

    # 3. Full-rank LoRA: subspace-preserving factor updates keep existing modes and null-space intact
    b_pinv_u = torch.linalg.lstsq(b64, u_out.unsqueeze(1)).solution.squeeze(1)
    proj_u = b64 @ b_pinv_u
    proj_u_norm = float(proj_u.norm().item())
    res_u = float((u_out - proj_u).norm().item())

    a_pinv_v = torch.linalg.lstsq(a64.T, v_in.unsqueeze(1)).solution.squeeze(1)
    proj_v = a64.T @ a_pinv_v
    proj_v_norm = float(proj_v.norm().item())
    res_v = float((v_in - proj_v).norm().item())

    if lora_fold_mode == "update_b" or (lora_fold_mode == "auto" and res_v <= 1e-8 and proj_v_norm > 1e-6):
        if proj_v_norm > 1e-8:
            b_new = b64 + delta_s * torch.outer(u_out, a_pinv_v)
            method = "update_b_exact" if res_v <= 1e-8 else "update_b_projected"
            return b_new.contiguous(), a64.clone(), method

    if lora_fold_mode in {"auto", "update_a"} and proj_u_norm > 1e-6:
        a_new = a64 + delta_s * torch.outer(b_pinv_u, v_in)
        method = "update_a_exact" if res_u <= 1e-8 else "update_a_projected"
        return b64.clone(), a_new.contiguous(), method

    if proj_v_norm > 1e-6:
        b_new = b64 + delta_s * torch.outer(u_out, a_pinv_v)
        return b_new.contiguous(), a64.clone(), "update_b_projected"

    b_f, a_f = _qr_svd_fold()
    return b_f, a_f, "qr_svd_truncated"


@torch.no_grad()
def _apply_rank1_candidate_inplace(
    handle: _TargetHandle,
    u_out: torch.Tensor,
    v_in: torch.Tensor,
    alpha: float,
    *,
    lora_fold_mode: str = "auto",
) -> dict[str, Any]:
    """Apply rank-1 steering update ``alpha * u_out @ v_in^T`` in-place to ``handle``."""
    if handle.kind == "dense":
        assert handle.dense_tensor is not None
        w64 = handle.dense_tensor.detach().to(device="cpu", dtype=torch.float64)
        w_new = w64 + float(alpha) * torch.outer(u_out, v_in)
        handle.dense_tensor.copy_(
            w_new.to(device=handle.dense_tensor.device, dtype=handle.dense_tensor.dtype)
        )
        return {"edit_kind": "dense_rank1", "fold_method": "dense_exact", "lora_rank": None}

    assert handle.lora_b is not None and handle.lora_a is not None
    b64 = handle.lora_b.detach().to(device="cpu", dtype=torch.float64)
    a64 = handle.lora_a.detach().to(device="cpu", dtype=torch.float64)
    b_new, a_new, fold_method = _fold_rank1_into_lora_factors(
        b64,
        a64,
        u_out,
        v_in,
        alpha,
        handle.scaling,
        lora_fold_mode=lora_fold_mode,
    )
    handle.lora_b.copy_(b_new.to(device=handle.lora_b.device, dtype=handle.lora_b.dtype))
    handle.lora_a.copy_(a_new.to(device=handle.lora_a.device, dtype=handle.lora_a.dtype))
    return {
        "edit_kind": "lora_rank_preserving_fold",
        "fold_method": fold_method,
        "lora_rank": int(handle.lora_a.shape[0]),
    }


def _invoke_eval_energy_fn(
    eval_energy_fn: Callable[..., Mapping[str, Any] | float],
    module_or_lora: Any,
    alpha: float,
) -> Mapping[str, Any] | float:
    """Call ``eval_energy_fn(module)`` or ``eval_energy_fn(module, alpha)`` cleanly."""
    sig = inspect.signature(eval_energy_fn)
    pos_params = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    if has_varargs or len(pos_params) >= 2:
        return eval_energy_fn(module_or_lora, alpha)
    if len(pos_params) == 1:
        return eval_energy_fn(module_or_lora)
    return eval_energy_fn()


def _extract_eval_metrics(raw_res: Any) -> dict[str, Any]:
    """Normalize ``eval_energy_fn`` output into finite scalar NLL and domain metrics."""
    if isinstance(raw_res, (int, float)) and not isinstance(raw_res, bool):
        val = float(raw_res)
        return {
            "mode": "scalar",
            "raw_energy": val,
            "all_finite": math.isfinite(val),
            "tool_nll": None,
            "general_nll": None,
            "domain_nlls": {},
            "explicit_general_drift": None,
            "explicit_worst_drift": None,
        }

    if not isinstance(raw_res, Mapping) or not raw_res:
        raise TypeError("eval_energy_fn must return a finite float or non-empty mapping")

    dom_raw = raw_res.get("domain_nlls", {})
    dom_nlls: dict[str, float] = {}
    if isinstance(dom_raw, Mapping):
        for k, v in dom_raw.items():
            dom_nlls[str(k)] = float(v)

    tool_raw = raw_res.get("tool_nll", raw_res.get("skill_nll", raw_res.get("target_nll")))
    if tool_raw is None and "delimiter_nll" in raw_res and "schema_nll" in raw_res:
        tool_raw = 0.5 * (float(raw_res["delimiter_nll"]) + float(raw_res["schema_nll"]))
    elif tool_raw is None and "tool" in dom_nlls:
        tool_raw = dom_nlls["tool"]

    gen_raw = raw_res.get("general_nll", raw_res.get("prose_nll"))
    if gen_raw is None and dom_nlls:
        gen_vals = [v for k, v in dom_nlls.items() if k != "tool"]
        if gen_vals:
            gen_raw = float(np.mean(gen_vals))

    tool_nll = float(tool_raw) if tool_raw is not None else None
    gen_nll = float(gen_raw) if gen_raw is not None else None
    nf_outputs = int(raw_res.get("non_finite_outputs", 0))
    explicit_energy = raw_res.get("anse_energy", raw_res.get("decision_energy"))
    explicit_gen_drift = raw_res.get("general_drift")
    explicit_worst_drift = raw_res.get("worst_domain_drift")

    vals_to_check: list[float] = []
    if tool_nll is not None:
        vals_to_check.append(tool_nll)
    if gen_nll is not None:
        vals_to_check.append(gen_nll)
    vals_to_check.extend(dom_nlls.values())
    if explicit_energy is not None:
        vals_to_check.append(float(explicit_energy))

    all_finite = nf_outputs == 0 and all(math.isfinite(v) for v in vals_to_check)
    return {
        "mode": "mapping",
        "raw_energy": float(explicit_energy) if explicit_energy is not None else None,
        "all_finite": all_finite,
        "tool_nll": tool_nll,
        "general_nll": gen_nll,
        "domain_nlls": dom_nlls,
        "explicit_general_drift": float(explicit_gen_drift) if explicit_gen_drift is not None else None,
        "explicit_worst_drift": float(explicit_worst_drift) if explicit_worst_drift is not None else None,
        "tool_nll_before": float(raw_res["tool_nll_before"]) if "tool_nll_before" in raw_res else None,
        "general_nll_before": float(raw_res["general_nll_before"]) if "general_nll_before" in raw_res else None,
    }


def _compute_decision_energy(
    cand_eval: Mapping[str, Any],
    base_eval: Mapping[str, Any],
    cfg: DecisionSkillSteeringConfig,
) -> dict[str, Any]:
    """Compute objective Decision Energy ``E(alpha)`` with Maximum Pain on ``>+1.0%`` drift."""
    if not cand_eval["all_finite"]:
        return {
            "anse_energy": float(cfg.max_pain_energy),
            "decision_energy": float(cfg.max_pain_energy),
            "guard_ok": False,
            "all_finite": False,
            "tool_nll_before": base_eval["tool_nll"],
            "tool_nll_after": cand_eval["tool_nll"],
            "tool_nll_diff": None,
            "tool_nll_gain_pct": None,
            "general_nll_before": base_eval["general_nll"],
            "general_nll_after": cand_eval["general_nll"],
            "general_drift": None,
            "general_drift_pct": None,
            "worst_domain_drift": None,
            "worst_domain_drift_pct": None,
        }

    if cand_eval["mode"] == "scalar":
        raw_e = float(cand_eval["raw_energy"])
        return {
            "anse_energy": raw_e,
            "decision_energy": raw_e,
            "guard_ok": raw_e < cfg.max_pain_energy,
            "all_finite": True,
            "tool_nll_before": None,
            "tool_nll_after": None,
            "tool_nll_diff": None,
            "tool_nll_gain_pct": None,
            "general_nll_before": None,
            "general_nll_after": None,
            "general_drift": None,
            "general_drift_pct": None,
            "worst_domain_drift": None,
            "worst_domain_drift_pct": None,
        }

    base_tool = cand_eval.get("tool_nll_before")
    if base_tool is None:
        base_tool = base_eval["tool_nll"]
    cand_tool = cand_eval["tool_nll"]
    tool_diff = (cand_tool - base_tool) if (cand_tool is not None and base_tool is not None) else 0.0
    tool_gain_pct = (
        100.0 * (base_tool - cand_tool) / max(abs(base_tool), 1e-8)
        if (cand_tool is not None and base_tool is not None)
        else 0.0
    )

    base_gen = cand_eval.get("general_nll_before")
    if base_gen is None:
        base_gen = base_eval["general_nll"]
    cand_gen = cand_eval["general_nll"]

    if cand_eval["explicit_general_drift"] is not None:
        gen_drift = float(cand_eval["explicit_general_drift"])
    elif cand_gen is not None and base_gen is not None:
        gen_drift = (cand_gen - base_gen) / max(abs(base_gen), 1e-8)
    else:
        gen_drift = 0.0

    domain_drifts: list[float] = [gen_drift]
    base_doms = base_eval.get("domain_nlls", {})
    for dom, val in cand_eval.get("domain_nlls", {}).items():
        if dom == "tool":
            continue
        if dom in base_doms:
            domain_drifts.append((val - base_doms[dom]) / max(abs(base_doms[dom]), 1e-8))
    if cand_eval["explicit_worst_drift"] is not None:
        worst_drift = max(float(cand_eval["explicit_worst_drift"]), max(domain_drifts))
    else:
        worst_drift = max(domain_drifts)

    guard_ok = worst_drift <= cfg.max_general_drift
    if cand_eval["raw_energy"] is not None:
        energy = float(cand_eval["raw_energy"])
        if not guard_ok and energy < cfg.max_pain_energy:
            energy = float(cfg.max_pain_energy) + cfg.drift_penalty_weight * max(0.0, worst_drift)
    elif not guard_ok:
        energy = float(cfg.max_pain_energy) + cfg.drift_penalty_weight * max(0.0, worst_drift)
    else:
        soft_pen = cfg.soft_margin_penalty if worst_drift > cfg.safe_drift_threshold else 0.0
        energy = float(tool_diff) + cfg.drift_penalty_weight * max(0.0, worst_drift) + soft_pen

    return {
        "anse_energy": float(energy),
        "decision_energy": float(energy),
        "guard_ok": guard_ok,
        "all_finite": True,
        "tool_nll_before": base_tool,
        "tool_nll_after": cand_tool,
        "tool_nll_diff": float(tool_diff),
        "tool_nll_gain_pct": float(tool_gain_pct),
        "general_nll_before": base_gen,
        "general_nll_after": cand_gen,
        "general_drift": float(gen_drift),
        "general_drift_pct": 100.0 * float(gen_drift),
        "worst_domain_drift": float(worst_drift),
        "worst_domain_drift_pct": 100.0 * float(worst_drift),
    }


def _compute_reference_geometry(
    handle: _TargetHandle,
    cfg: DecisionSkillSteeringConfig,
    reference_weight_norm: float | None = None,
) -> tuple[torch.Tensor, float, float | None, bool]:
    """Compute baseline effective weight, reference Frobenius norm, and erank enforcement flag."""
    w_eff_base = _effective_weight_f64(handle)
    w_eff_norm = float(w_eff_base.norm().item())
    if reference_weight_norm is not None:
        ref_norm = _validate_positive_finite(reference_weight_norm, "reference_weight_norm")
    elif w_eff_norm > 1e-12:
        ref_norm = w_eff_norm
    elif handle.lora_a is not None:
        ref_norm = max(float(handle.lora_a.detach().to(torch.float64).norm().item()), 1.0)
    else:
        ref_norm = 1.0

    base_erank = _compute_effective_rank(w_eff_base) if w_eff_norm > 1e-12 else None
    enforce_erank = cfg.enforce_effective_rank_bound and (
        handle.kind == "dense"
        or handle.base_tensor is not None
        or (base_erank is not None and base_erank >= max(1.0, float(handle.rank) - 0.5))
    )
    return w_eff_base, ref_norm, base_erank, enforce_erank


@torch.no_grad()
def search_decision_energy_rank1_edit(
    module_or_lora: nn.Module | torch.Tensor,
    skill_dir: torch.Tensor | Mapping[str, Any],
    unembed_target_dir: torch.Tensor | nn.Module | Mapping[str, Any],
    eval_energy_fn: Callable[..., Mapping[str, Any] | float],
    *,
    target_token_ids: Sequence[int] | torch.Tensor | None = None,
    baseline_token_ids: Sequence[int] | torch.Tensor | None = None,
    alpha_grid: Sequence[float] | None = None,
    weight_attr: str | None = None,
    expert_idx: int | None = None,
    reference_weight_norm: float | None = None,
    max_norm_perturbation_ratio: float = DEFAULT_MAX_NORM_PERTURBATION_RATIO,
    max_effective_rank_perturbation_ratio: float = DEFAULT_MAX_EFFECTIVE_RANK_PERTURBATION_RATIO,
    enforce_effective_rank_bound: bool = True,
    max_general_drift: float = DEFAULT_MAX_GENERAL_DRIFT,
    safe_drift_threshold: float = DEFAULT_SAFE_DRIFT_THRESHOLD,
    drift_penalty_weight: float = DEFAULT_DRIFT_PENALTY_WEIGHT,
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY,
    max_pain_energy: float = DEFAULT_ANSE_MAX_PAIN,
    lora_fold_mode: str = "auto",
    include_negative_alphas: bool = False,
    config: DecisionSkillSteeringConfig | None = None,
) -> dict[str, Any]:
    """Evaluate a closed-form grid of rank-1 steering edits under ANSE Decision Energy.

    Constructs ``Delta W = alpha * u_out @ v_in^T`` where ``v_in = skill_dir`` lies in the
    preserve null-space and ``u_out = unembed_target_dir`` aligns with target skill/tool-call
    logits. Every candidate is strictly bounded to ``< 0.02%`` norm and effective-rank
    perturbation, scored by ``eval_energy_fn`` with ``1e6`` Maximum Pain on ``> +1.0%``
    general-domain NLL regression, and leaves ``module_or_lora`` bitwise restored on return.
    """
    if not callable(eval_energy_fn):
        raise TypeError("eval_energy_fn must be callable")
    if not isinstance(include_negative_alphas, bool):
        raise TypeError(
            f"include_negative_alphas must be a bool, got {type(include_negative_alphas).__name__}"
        )

    if config is not None:
        if not isinstance(config, DecisionSkillSteeringConfig):
            raise TypeError(
                f"config must be DecisionSkillSteeringConfig or None, got {type(config).__name__}"
            )
        cfg = config
    else:
        cfg = DecisionSkillSteeringConfig(
            max_norm_perturbation_ratio=max_norm_perturbation_ratio,
            max_effective_rank_perturbation_ratio=max_effective_rank_perturbation_ratio,
            enforce_effective_rank_bound=enforce_effective_rank_bound,
            max_general_drift=max_general_drift,
            safe_drift_threshold=safe_drift_threshold,
            drift_penalty_weight=drift_penalty_weight,
            soft_margin_penalty=soft_margin_penalty,
            max_pain_energy=max_pain_energy,
            lora_fold_mode=lora_fold_mode,
        )

    handle = _resolve_target_handle(module_or_lora, weight_attr=weight_attr, expert_idx=expert_idx)
    v_in, _ = _resolve_skill_vector(skill_dir, handle.d_in)
    u_out = _resolve_unembed_target_dir(
        unembed_target_dir,
        handle.d_out,
        target_token_ids=target_token_ids,
        baseline_token_ids=baseline_token_ids,
    )

    w_eff_base, ref_norm, base_erank, enforce_erank = _compute_reference_geometry(
        handle, cfg, reference_weight_norm=reference_weight_norm
    )

    if alpha_grid is None:
        max_alpha = cfg.max_norm_perturbation_ratio * ref_norm
        pos_grid = [frac * max_alpha for frac in DEFAULT_PERTURBATION_FRACTIONS]
        grid_vals = ([-a for a in reversed(pos_grid)] + pos_grid) if include_negative_alphas else pos_grid
    else:
        if isinstance(alpha_grid, (str, bytes)) or not isinstance(alpha_grid, Sequence) or not alpha_grid:
            raise ValueError("alpha_grid must be a non-empty sequence of finite real numbers")
        grid_vals = [_validate_finite_float(a, "alpha_grid_item") for a in alpha_grid]

    if float(v_in.norm().item()) <= 1e-12:
        base_raw = _invoke_eval_energy_fn(eval_energy_fn, module_or_lora, 0.0)
        base_eval = _extract_eval_metrics(base_raw)
        base_metrics = _compute_decision_energy(base_eval, base_eval, cfg)
        return {
            "target_kind": handle.kind,
            "target_shape": (handle.d_out, handle.d_in),
            "reference_weight_norm": ref_norm,
            "u_out": u_out,
            "v_in": v_in,
            "parent_energy": float(base_metrics["anse_energy"]),
            "best_energy": float(base_metrics["anse_energy"]),
            "delta_energy": 0.0,
            "best_alpha": None,
            "best_candidate": None,
            "candidates": [],
            "rejected_bound_candidates": [],
            "guard_ok": bool(base_metrics["guard_ok"]),
            "all_rejected_maximum_pain": False,
            "rolled_back_during_search": True,
            "noop_reason": "zero_skill_direction",
            "lora_fold_mode": cfg.lora_fold_mode,
        }

    snap = snapshot_target_state(module_or_lora, weight_attr=weight_attr, expert_idx=expert_idx)
    before_sha = _hash_tensor_map(handle.hash_tensors)

    base_raw = _invoke_eval_energy_fn(eval_energy_fn, module_or_lora, 0.0)
    base_eval = _extract_eval_metrics(base_raw)
    base_metrics = _compute_decision_energy(base_eval, base_eval, cfg)
    parent_energy = float(base_metrics["anse_energy"])

    candidates: list[dict[str, Any]] = []
    rejected_bound: list[dict[str, Any]] = []
    feasible_improving: list[dict[str, Any]] = []

    try:
        for cand_alpha in grid_vals:
            restore_target_state_inplace(
                module_or_lora, snap, weight_attr=weight_attr, expert_idx=expert_idx
            )
            nominal_ratio = abs(float(cand_alpha)) / ref_norm
            if nominal_ratio >= cfg.max_norm_perturbation_ratio:
                rejected_bound.append(
                    {
                        "alpha": float(cand_alpha),
                        "norm_perturbation_ratio": nominal_ratio,
                        "reason": "nominal_norm_perturbation_exceeds_ceiling",
                    }
                )
                continue

            apply_info = _apply_rank1_candidate_inplace(
                handle,
                u_out,
                v_in,
                cand_alpha,
                lora_fold_mode=cfg.lora_fold_mode,
            )
            w_eff_cand = _effective_weight_f64(handle)
            actual_delta_norm = float((w_eff_cand - w_eff_base).norm().item())
            actual_norm_ratio = actual_delta_norm / ref_norm

            erank_ratio: float | None = None
            if base_erank is not None and base_erank > 0.0:
                cand_erank = _compute_effective_rank(w_eff_cand)
                if cand_erank is not None:
                    erank_ratio = abs(cand_erank - base_erank) / base_erank

            if actual_norm_ratio >= cfg.max_norm_perturbation_ratio:
                rejected_bound.append(
                    {
                        "alpha": float(cand_alpha),
                        "norm_perturbation_ratio": actual_norm_ratio,
                        "reason": "actual_norm_perturbation_exceeds_ceiling",
                    }
                )
                continue
            if (
                enforce_erank
                and erank_ratio is not None
                and erank_ratio >= cfg.max_effective_rank_perturbation_ratio
            ):
                rejected_bound.append(
                    {
                        "alpha": float(cand_alpha),
                        "norm_perturbation_ratio": actual_norm_ratio,
                        "effective_rank_perturbation_ratio": erank_ratio,
                        "reason": "effective_rank_perturbation_exceeds_ceiling",
                    }
                )
                continue

            cand_raw = _invoke_eval_energy_fn(eval_energy_fn, module_or_lora, cand_alpha)
            cand_eval = _extract_eval_metrics(cand_raw)
            e_info = _compute_decision_energy(cand_eval, base_eval, cfg)
            e_cand = float(e_info["anse_energy"])
            delta_e = e_cand - parent_energy

            entry = {
                "alpha": float(cand_alpha),
                "nominal_norm_perturbation_ratio": nominal_ratio,
                "norm_perturbation_ratio": actual_norm_ratio,
                "effective_rank_perturbation_ratio": erank_ratio,
                "delta_energy": delta_e,
                **apply_info,
                **e_info,
            }
            candidates.append(entry)
            if e_info["all_finite"] and e_info["guard_ok"] and delta_e < 0.0:
                feasible_improving.append(entry)
    finally:
        restored_sha = restore_target_state_inplace(
            module_or_lora, snap, weight_attr=weight_attr, expert_idx=expert_idx
        )
        if restored_sha != before_sha:
            raise RuntimeError(
                f"Atomic search restore SHA-256 mismatch: expected {before_sha}, got {restored_sha}"
            )

    if feasible_improving:
        feasible_improving.sort(
            key=lambda item: (float(item["anse_energy"]), float(item["norm_perturbation_ratio"]))
        )
        best = feasible_improving[0]
        return {
            "target_kind": handle.kind,
            "target_shape": (handle.d_out, handle.d_in),
            "reference_weight_norm": ref_norm,
            "u_out": u_out,
            "v_in": v_in,
            "parent_energy": parent_energy,
            "best_energy": float(best["anse_energy"]),
            "delta_energy": float(best["delta_energy"]),
            "best_alpha": float(best["alpha"]),
            "best_candidate": best,
            "candidates": candidates,
            "rejected_bound_candidates": rejected_bound,
            "guard_ok": True,
            "all_rejected_maximum_pain": False,
            "rolled_back_during_search": True,
            "noop_reason": None,
            "lora_fold_mode": cfg.lora_fold_mode,
        }

    all_max_pain = bool(candidates) and all(
        (not c["guard_ok"]) or (float(c["anse_energy"]) >= cfg.max_pain_energy)
        for c in candidates
    )
    if not candidates:
        noop_reason = "all_candidates_exceed_perturbation_bound"
    elif all_max_pain:
        noop_reason = "maximum_pain_guard_violation"
    else:
        noop_reason = "no_negative_delta_energy"

    best_seen_e = min((float(c["anse_energy"]) for c in candidates), default=parent_energy)
    return {
        "target_kind": handle.kind,
        "target_shape": (handle.d_out, handle.d_in),
        "reference_weight_norm": ref_norm,
        "u_out": u_out,
        "v_in": v_in,
        "parent_energy": parent_energy,
        "best_energy": best_seen_e,
        "delta_energy": best_seen_e - parent_energy,
        "best_alpha": None,
        "best_candidate": None,
        "candidates": candidates,
        "rejected_bound_candidates": rejected_bound,
        "guard_ok": not all_max_pain and bool(base_metrics["guard_ok"]),
        "all_rejected_maximum_pain": all_max_pain,
        "rolled_back_during_search": True,
        "noop_reason": noop_reason,
        "lora_fold_mode": cfg.lora_fold_mode,
    }


# ---------------------------------------------------------------------------
# (c) Verified Atomic Commit with Rollback and SHA-256 Receipt
# ---------------------------------------------------------------------------


def _build_commit_receipt_hash(payload: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 receipt digest over serialized verification metadata."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@torch.no_grad()
def apply_verified_skill_edit_inplace(
    module_or_lora: nn.Module | torch.Tensor,
    edit_or_search_result: Mapping[str, Any] | None = None,
    *,
    skill_dir: torch.Tensor | Mapping[str, Any] | None = None,
    unembed_target_dir: torch.Tensor | nn.Module | Mapping[str, Any] | None = None,
    eval_energy_fn: Callable[..., Mapping[str, Any] | float] | None = None,
    target_token_ids: Sequence[int] | torch.Tensor | None = None,
    baseline_token_ids: Sequence[int] | torch.Tensor | None = None,
    alpha_grid: Sequence[float] | None = None,
    weight_attr: str | None = None,
    expert_idx: int | None = None,
    reference_weight_norm: float | None = None,
    lora_fold_mode: str | None = None,
    config: DecisionSkillSteeringConfig | None = None,
) -> dict[str, Any]:
    """Commit the energy-minimizing rank-1 edit iff ``Delta E < 0`` with atomic rollback.

    Supports either committing a pre-evaluated ``search_decision_energy_rank1_edit``
    result / explicit ``{"alpha", "u_out", "v_in"}`` descriptor, or executing search
    and verified commit in a single call when ``skill_dir``, ``unembed_target_dir``,
    and ``eval_energy_fn`` are supplied.
    """
    if config is not None and not isinstance(config, DecisionSkillSteeringConfig):
        raise TypeError(
            f"config must be DecisionSkillSteeringConfig or None, got {type(config).__name__}"
        )
    cfg = config or DecisionSkillSteeringConfig()
    effective_fold_mode = lora_fold_mode if lora_fold_mode is not None else cfg.lora_fold_mode
    if effective_fold_mode not in _VALID_LORA_FOLD_MODES:
        raise ValueError(
            f"lora_fold_mode must be one of {sorted(_VALID_LORA_FOLD_MODES)}, got {effective_fold_mode!r}"
        )

    handle = _resolve_target_handle(module_or_lora, weight_attr=weight_attr, expert_idx=expert_idx)
    before_sha = _hash_tensor_map(handle.hash_tensors)
    snap = snapshot_target_state(module_or_lora, weight_attr=weight_attr, expert_idx=expert_idx)
    lora_rank_before = handle.rank if handle.kind == "lora" else None

    search_res: Mapping[str, Any] | None = edit_or_search_result
    if search_res is None:
        if skill_dir is None or unembed_target_dir is None or eval_energy_fn is None:
            raise ValueError(
                "Provide edit_or_search_result or all of (skill_dir, unembed_target_dir, eval_energy_fn)"
            )
        search_res = search_decision_energy_rank1_edit(
            module_or_lora,
            skill_dir,
            unembed_target_dir,
            eval_energy_fn,
            target_token_ids=target_token_ids,
            baseline_token_ids=baseline_token_ids,
            alpha_grid=alpha_grid,
            weight_attr=weight_attr,
            expert_idx=expert_idx,
            reference_weight_norm=reference_weight_norm,
            lora_fold_mode=effective_fold_mode,
            config=cfg,
        )
    elif not isinstance(search_res, Mapping):
        raise TypeError(
            f"edit_or_search_result must be a Mapping or None, got {type(search_res).__name__}"
        )

    # Extract candidate (alpha, u_out, v_in) from either search result or explicit edit spec
    if "best_alpha" in search_res:
        cand_alpha_raw = search_res.get("best_alpha")
        u_raw = search_res.get("u_out")
        v_raw = search_res.get("v_in")
        cand_meta = search_res.get("best_candidate")
    elif "alpha" in search_res:
        cand_alpha_raw = search_res.get("alpha")
        u_raw = search_res.get("u_out")
        v_raw = search_res.get("v_in")
        cand_meta = search_res
    else:
        raise ValueError(
            "edit_or_search_result must contain 'best_alpha' (from search) or explicit 'alpha', 'u_out', 'v_in'"
        )

    # Case 1: Search found no admissible improving candidate -> bitwise verified no-op / rejection
    if cand_alpha_raw is None:
        final_sha = _hash_tensor_map(handle.hash_tensors)
        reject_reason = str(search_res.get("noop_reason") or "no_admissible_candidate")
        receipt_core = {
            "committed": False,
            "rolled_back": True,
            "rollback_bitwise_verified": final_sha == before_sha,
            "reject_reason": reject_reason,
            "target_kind": handle.kind,
            "before_sha256": before_sha,
            "candidate_sha256": before_sha,
            "final_sha256": final_sha,
            "alpha": None,
            "parent_energy": search_res.get("parent_energy"),
            "candidate_energy": search_res.get("best_energy"),
            "delta_energy": search_res.get("delta_energy", 0.0),
            "guard_ok": bool(search_res.get("guard_ok", False)),
            "all_rejected_maximum_pain": bool(search_res.get("all_rejected_maximum_pain", False)),
            "norm_perturbation_ratio": 0.0,
            "effective_rank_perturbation_ratio": 0.0,
            "perturbation_bound_respected": True,
            "lora_rank_before": lora_rank_before,
            "lora_rank_after": lora_rank_before,
            "lora_rank_preserved": True if handle.kind == "lora" else None,
        }
        receipt_core["receipt_sha256"] = _build_commit_receipt_hash(receipt_core)
        return receipt_core

    cand_alpha = _validate_finite_float(cand_alpha_raw, "alpha")
    if u_raw is None or v_raw is None:
        raise ValueError("edit_or_search_result must contain 'u_out' and 'v_in' tensors")
    u_out = _resolve_unembed_target_dir(u_raw, handle.d_out)
    v_in, _ = _resolve_skill_vector(v_raw, handle.d_in)
    if float(v_in.norm().item()) <= 1e-12:
        raise ValueError("Candidate v_in direction has zero norm")

    ref_override = (
        reference_weight_norm
        if reference_weight_norm is not None
        else search_res.get("reference_weight_norm")
    )
    w_eff_before, ref_norm, base_erank, enforce_erank = _compute_reference_geometry(
        handle, cfg, reference_weight_norm=ref_override
    )

    if eval_energy_fn is not None:
        parent_raw = _invoke_eval_energy_fn(eval_energy_fn, module_or_lora, 0.0)
        parent_eval = _extract_eval_metrics(parent_raw)
        parent_metrics = _compute_decision_energy(parent_eval, parent_eval, cfg)
        e_parent = float(parent_metrics["anse_energy"])
    else:
        parent_eval = {
            "mode": "mapping",
            "tool_nll": cand_meta.get("tool_nll_before") if isinstance(cand_meta, Mapping) else None,
            "general_nll": cand_meta.get("general_nll_before") if isinstance(cand_meta, Mapping) else None,
            "domain_nlls": {},
        }
        e_parent = float(search_res["parent_energy"])

    # Apply candidate edit in-place, compute SHA-256 and perturbation ratios, then verify Delta E < 0
    apply_info = _apply_rank1_candidate_inplace(
        handle,
        u_out,
        v_in,
        cand_alpha,
        lora_fold_mode=effective_fold_mode,
    )
    candidate_sha = _hash_tensor_map(handle.hash_tensors)
    w_eff_after = _effective_weight_f64(handle)
    actual_norm_ratio = float((w_eff_after - w_eff_before).norm().item()) / ref_norm
    nominal_norm_ratio = abs(cand_alpha) / ref_norm

    erank_ratio: float | None = None
    if base_erank is not None and base_erank > 0.0:
        cand_erank = _compute_effective_rank(w_eff_after)
        if cand_erank is not None:
            erank_ratio = abs(cand_erank - base_erank) / base_erank

    bound_ok = (nominal_norm_ratio < cfg.max_norm_perturbation_ratio) and (
        actual_norm_ratio < cfg.max_norm_perturbation_ratio
    )
    if enforce_erank and erank_ratio is not None:
        bound_ok = bound_ok and (erank_ratio < cfg.max_effective_rank_perturbation_ratio)

    if eval_energy_fn is not None:
        child_raw = _invoke_eval_energy_fn(eval_energy_fn, module_or_lora, cand_alpha)
        child_eval = _extract_eval_metrics(child_raw)
        child_metrics = _compute_decision_energy(child_eval, parent_eval, cfg)
        e_child = float(child_metrics["anse_energy"])
    else:
        assert isinstance(cand_meta, Mapping)
        e_child = float(cand_meta["anse_energy"])
        child_metrics = dict(cand_meta)

    delta_e = e_child - e_parent
    all_finite = bool(child_metrics.get("all_finite", True))
    guard_ok = bool(child_metrics.get("guard_ok", False))
    lora_rank_after = int(handle.lora_a.shape[0]) if handle.lora_a is not None else None
    rank_preserved = (lora_rank_after == lora_rank_before) if handle.kind == "lora" else None

    commit_ok = bool(all_finite and guard_ok and bound_ok and (delta_e < 0.0))

    if commit_ok:
        final_sha = _hash_tensor_map(handle.hash_tensors)
        receipt_core = {
            "committed": True,
            "rolled_back": False,
            "rollback_bitwise_verified": False,
            "reject_reason": None,
            "target_kind": handle.kind,
            "edit_kind": apply_info["edit_kind"],
            "fold_method": apply_info["fold_method"],
            "before_sha256": before_sha,
            "candidate_sha256": candidate_sha,
            "final_sha256": final_sha,
            "alpha": cand_alpha,
            "parent_energy": e_parent,
            "candidate_energy": e_child,
            "delta_energy": delta_e,
            "tool_nll_before": child_metrics.get("tool_nll_before"),
            "tool_nll_after": child_metrics.get("tool_nll_after"),
            "tool_nll_gain_pct": child_metrics.get("tool_nll_gain_pct"),
            "general_nll_before": child_metrics.get("general_nll_before"),
            "general_nll_after": child_metrics.get("general_nll_after"),
            "general_drift_pct": child_metrics.get("general_drift_pct"),
            "worst_domain_drift_pct": child_metrics.get("worst_domain_drift_pct"),
            "guard_ok": True,
            "norm_perturbation_ratio": actual_norm_ratio,
            "effective_rank_perturbation_ratio": erank_ratio,
            "perturbation_bound_respected": True,
            "lora_rank_before": lora_rank_before,
            "lora_rank_after": lora_rank_after,
            "lora_rank_preserved": rank_preserved,
        }
        receipt_core["receipt_sha256"] = _build_commit_receipt_hash(receipt_core)
        return receipt_core

    # Atomic bitwise rollback on any guard/bound/energy failure
    restored_sha = restore_target_state_inplace(
        module_or_lora, snap, weight_attr=weight_attr, expert_idx=expert_idx
    )
    bitwise_verified = restored_sha == before_sha
    if not bitwise_verified:
        raise RuntimeError(
            f"Atomic bitwise rollback SHA-256 mismatch: expected {before_sha}, got {restored_sha}"
        )

    if not all_finite:
        reject_reason = "non_finite_child_output"
    elif not guard_ok:
        reject_reason = "maximum_pain_guard_violation"
    elif not bound_ok:
        reject_reason = "perturbation_bound_exceeded"
    else:
        reject_reason = "non_negative_delta_energy"

    receipt_core = {
        "committed": False,
        "rolled_back": True,
        "rollback_bitwise_verified": bitwise_verified,
        "reject_reason": reject_reason,
        "target_kind": handle.kind,
        "before_sha256": before_sha,
        "candidate_sha256": candidate_sha,
        "final_sha256": restored_sha,
        "alpha": cand_alpha,
        "parent_energy": e_parent,
        "candidate_energy": e_child,
        "delta_energy": delta_e,
        "tool_nll_before": child_metrics.get("tool_nll_before"),
        "tool_nll_after": child_metrics.get("tool_nll_after"),
        "tool_nll_gain_pct": child_metrics.get("tool_nll_gain_pct"),
        "general_nll_before": child_metrics.get("general_nll_before"),
        "general_nll_after": child_metrics.get("general_nll_after"),
        "general_drift_pct": child_metrics.get("general_drift_pct"),
        "worst_domain_drift_pct": child_metrics.get("worst_domain_drift_pct"),
        "guard_ok": guard_ok,
        "norm_perturbation_ratio": actual_norm_ratio,
        "effective_rank_perturbation_ratio": erank_ratio,
        "perturbation_bound_respected": bound_ok,
        "lora_rank_before": lora_rank_before,
        "lora_rank_after": lora_rank_before,
        "lora_rank_preserved": True if handle.kind == "lora" else None,
    }
    receipt_core["receipt_sha256"] = _build_commit_receipt_hash(receipt_core)
    return receipt_core


# ---------------------------------------------------------------------------
# End-to-End Multi-Layer Transformer / Sparse MoE Skill Injection Helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _eval_causal_nll(
    model: nn.Module,
    seqs: torch.Tensor,
    *,
    score_from: int = 0,
    batch_size: int = 16,
) -> tuple[float, int]:
    """Compute exact causal next-token NLL over ``[score_from:-1] -> [score_from+1:]``."""
    if seqs.ndim != 2 or seqs.shape[0] == 0 or seqs.shape[1] < 2:
        raise ValueError(f"seqs must be 2D with sequence length >= 2, got {tuple(seqs.shape)}")
    dev = model.wte.weight.device
    start_pos = min(max(0, int(score_from)), int(seqs.shape[1]) - 2)
    total_loss = 0.0
    total_tokens = 0
    nf_count = 0
    for b0 in range(0, int(seqs.shape[0]), batch_size):
        batch = seqs[b0 : b0 + batch_size].to(dev)
        out = model(batch)
        logits = out[0] if isinstance(out, tuple) else out
        shift_logits = logits[:, start_pos:-1, :].to(torch.float32)
        nf_count += int((~torch.isfinite(shift_logits)).sum().item())
        targets = batch[:, start_pos + 1 :]
        tok_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        )
        total_loss += float(tok_loss.item())
        total_tokens += int(targets.numel())
    return total_loss / float(max(1, total_tokens)), nf_count


@torch.no_grad()
def make_skill_energy_eval_fn(
    model: nn.Module,
    tool_seqs: torch.Tensor,
    general_seqs: torch.Tensor,
    *,
    preserve_seqs: Mapping[str, torch.Tensor] | None = None,
    tool_score_from: int = 0,
    general_score_from: int = 0,
    batch_size: int = 16,
) -> Callable[[Any], dict[str, Any]]:
    """Build a deterministic CPU/GPU causal NLL evaluation closure for Decision Energy search."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if not isinstance(tool_seqs, torch.Tensor) or not isinstance(general_seqs, torch.Tensor):
        raise TypeError("tool_seqs and general_seqs must be torch.Tensor instances")
    if tool_seqs.ndim != 2 or tool_seqs.shape[0] == 0 or tool_seqs.shape[1] < 2:
        raise ValueError("tool_seqs must be non-empty 2D with sequence length >= 2")
    if general_seqs.ndim != 2 or general_seqs.shape[0] == 0 or general_seqs.shape[1] < 2:
        raise ValueError("general_seqs must be non-empty 2D with sequence length >= 2")

    t_from = _validate_nonneg_int(tool_score_from, "tool_score_from")
    g_from = _validate_nonneg_int(general_score_from, "general_score_from")
    bsz = _validate_positive_int(batch_size, "batch_size")

    pres_map: dict[str, torch.Tensor] = {}
    if preserve_seqs is not None:
        if not isinstance(preserve_seqs, Mapping):
            raise TypeError("preserve_seqs must be a mapping of domain_name -> 2D token tensor")
        for dom, ten in preserve_seqs.items():
            if not isinstance(ten, torch.Tensor) or ten.ndim != 2 or ten.shape[0] == 0 or ten.shape[1] < 2:
                raise ValueError(f"preserve_seqs[{dom!r}] must be non-empty 2D with length >= 2")
            pres_map[str(dom)] = ten

    def _eval(_target: Any = None) -> dict[str, Any]:
        was_training = model.training
        model.eval()
        try:
            tool_nll, nf_t = _eval_causal_nll(model, tool_seqs, score_from=t_from, batch_size=bsz)
            gen_nll, nf_g = _eval_causal_nll(model, general_seqs, score_from=g_from, batch_size=bsz)
            dom_nlls: dict[str, float] = {"E0_general_prose": gen_nll}
            nf_total = nf_t + nf_g
            for dom, p_seqs in sorted(pres_map.items()):
                p_nll, nf_p = _eval_causal_nll(model, p_seqs, score_from=g_from, batch_size=bsz)
                dom_nlls[dom] = p_nll
                nf_total += nf_p
            return {
                "tool_nll": tool_nll,
                "general_nll": gen_nll,
                "domain_nlls": dom_nlls,
                "non_finite_outputs": nf_total,
            }
        finally:
            model.train(was_training)

    return _eval


@torch.no_grad()
def _collect_block_w3_inputs(
    model: nn.Module,
    seqs: torch.Tensor,
    layer_idx: int,
    *,
    expert_idx: int | None,
    score_from: int,
    batch_size: int,
) -> torch.Tensor:
    """Propagate ``seqs`` to ``model.blocks[layer_idx]`` and collect SwiGLU ``w3`` input states."""
    dev = model.wte.weight.device
    cos = model.cos_cached.to(dev)
    sin = model.sin_cached.to(dev)
    start_pos = min(max(0, int(score_from)), int(seqs.shape[1]) - 2)

    parts: list[torch.Tensor] = []
    for b0 in range(0, int(seqs.shape[0]), batch_size):
        xb = compute_embeddings_batch(model, seqs[b0 : b0 + batch_size].to(dev))
        for blk_idx in range(layer_idx):
            blk = model.blocks[blk_idx]
            xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
            xb = forward_block_post_mlp(blk, xm, mi)
        target_blk = model.blocks[layer_idx]
        _, mlp_in = forward_block_pre_mlp(target_blk, xb, cos, sin)
        scored_in = mlp_in[:, start_pos:-1, :].reshape(-1, mlp_in.shape[-1])
        mlp = target_blk.mlp
        if hasattr(mlp, "experts") and hasattr(mlp, "base_mlp"):
            w_dt = mlp.base_mlp.w1.weight.dtype
            x_in = scored_in.to(dtype=w_dt)
            h1 = mlp.base_mlp.w1(x_in).to(torch.float32)
            h2 = mlp.base_mlp.w2(x_in).to(torch.float32)
            if expert_idx is not None:
                exp = mlp.experts[str(expert_idx)]
                x_f32 = scored_in.to(torch.float32)
                dh1 = F.linear(F.linear(x_f32, exp.w1_A), exp.w1_B) * exp.scaling
                dh2 = F.linear(F.linear(x_f32, exp.w2_A), exp.w2_B) * exp.scaling
                act = F.silu(h1 + dh1) * (h2 + dh2)
            else:
                act = F.silu(h1) * h2
        else:
            w_dt = mlp.w1.weight.dtype
            x_in = scored_in.to(dtype=w_dt)
            act = (F.silu(mlp.w1(x_in)) * mlp.w2(x_in)).to(torch.float32)
        parts.append(act.detach().cpu())
    return torch.cat(parts, dim=0)


@torch.no_grad()
def inject_decision_skill_into_model_inplace(
    model: nn.Module,
    pos_seqs: torch.Tensor,
    neg_seqs: torch.Tensor,
    preserve_seqs: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor],
    target_token_ids: Sequence[int] | torch.Tensor,
    *,
    baseline_token_ids: Sequence[int] | torch.Tensor | None = None,
    layer_indices: Sequence[int] | None = None,
    expert_idx: int | None = None,
    tool_score_from: int = 0,
    general_score_from: int = 0,
    batch_size: int = 16,
    alpha_grid: Sequence[float] | None = None,
    config: DecisionSkillSteeringConfig | None = None,
) -> dict[str, Any]:
    """Inject a targeted skill (e.g. structured tool-calling) into ``CrowdV1WithEngram`` without gradients.

    Operates strictly under ``@torch.no_grad()``, extracting contrastive null-space
    directions on late MLP down-projections (dense ``w3``, ``CrowdLoRALinear``, or MoE
    ``CrowdLoRAExpertAdapter``), searching rank-1 Decision Energy edits bounded to
    ``< 0.02%`` norm/effective-rank perturbation, and committing only when ``Delta E < 0``.
    """
    if not isinstance(model, nn.Module) or not hasattr(model, "blocks") or len(model.blocks) == 0:
        raise ValueError("model must be an nn.Module exposing a non-empty 'blocks' list")
    for attr in ("wte", "ln", "head", "cos_cached", "sin_cached"):
        if not hasattr(model, attr):
            raise ValueError(f"model must expose '{attr}'")

    if not isinstance(pos_seqs, torch.Tensor) or not isinstance(neg_seqs, torch.Tensor):
        raise TypeError("pos_seqs and neg_seqs must be torch.Tensor instances")
    if pos_seqs.ndim != 2 or pos_seqs.shape[0] == 0 or pos_seqs.shape[1] < 2:
        raise ValueError("pos_seqs must be non-empty 2D with sequence length >= 2")
    if neg_seqs.ndim != 2 or neg_seqs.shape[0] == 0 or neg_seqs.shape[1] < 2:
        raise ValueError("neg_seqs must be non-empty 2D with sequence length >= 2")

    cfg = config or DecisionSkillSteeringConfig()
    n_blocks = len(model.blocks)
    if layer_indices is None:
        target_layers = [n_blocks - 1]
    else:
        if isinstance(layer_indices, (str, bytes)) or not isinstance(layer_indices, Sequence) or not layer_indices:
            raise ValueError("layer_indices must be a non-empty sequence of integer block indices")
        target_layers = [_validate_nonneg_int(i, "layer_index") for i in layer_indices]
        if any(i >= n_blocks for i in target_layers):
            raise ValueError(f"layer_indices must lie in [0, {n_blocks})")

    if isinstance(preserve_seqs, torch.Tensor):
        pres_dict: dict[str, torch.Tensor] = {"E0_general_prose": preserve_seqs}
    elif isinstance(preserve_seqs, Mapping):
        if not preserve_seqs:
            raise ValueError("preserve_seqs mapping must not be empty")
        pres_dict = {str(k): v for k, v in preserve_seqs.items()}
    elif isinstance(preserve_seqs, Sequence) and not isinstance(preserve_seqs, (str, bytes, bytearray)):
        if not preserve_seqs:
            raise ValueError("preserve_seqs sequence must not be empty")
        pres_dict = {f"preserve_{idx}": v for idx, v in enumerate(preserve_seqs)}
    else:
        raise TypeError("preserve_seqs must be a Tensor, sequence of Tensors, or mapping of domain -> Tensor")

    general_eval_seqs = pres_dict.get("E0_general_prose", next(iter(pres_dict.values())))
    eval_fn = make_skill_energy_eval_fn(
        model,
        pos_seqs,
        general_eval_seqs,
        preserve_seqs=pres_dict,
        tool_score_from=tool_score_from,
        general_score_from=general_score_from,
        batch_size=batch_size,
    )

    was_training = model.training
    model.eval()
    try:
        initial_eval = eval_fn(model)
        initial_tool_nll = float(initial_eval["tool_nll"])
        initial_general_nll = float(initial_eval["general_nll"])

        layer_reports: list[dict[str, Any]] = []
        committed_layers: list[int] = []
        rolled_back_layers: list[int] = []

        for l_idx in target_layers:
            mlp = model.blocks[l_idx].mlp
            pos_act = _collect_block_w3_inputs(
                model,
                pos_seqs,
                l_idx,
                expert_idx=expert_idx,
                score_from=tool_score_from,
                batch_size=batch_size,
            )
            neg_act = _collect_block_w3_inputs(
                model,
                neg_seqs,
                l_idx,
                expert_idx=expert_idx,
                score_from=general_score_from,
                batch_size=batch_size,
            )
            pres_acts = {
                dom: _collect_block_w3_inputs(
                    model,
                    p_seq,
                    l_idx,
                    expert_idx=expert_idx,
                    score_from=general_score_from,
                    batch_size=batch_size,
                )
                for dom, p_seq in pres_dict.items()
            }

            dir_info = compute_contrastive_skill_direction(
                pos_act,
                neg_act,
                pres_acts,
                ridge_scale=cfg.ridge_scale,
                preserve_weight=cfg.preserve_weight,
                nullspace_eigen_ratio=cfg.nullspace_eigen_ratio,
                nullspace_mode=cfg.nullspace_mode,
                min_contrast_snr=cfg.min_contrast_snr,
                use_covariance_whitening=cfg.use_covariance_whitening,
            )

            ref_w_norm: float | None = None
            if hasattr(mlp, "base_mlp") and hasattr(mlp.base_mlp, "w3"):
                ref_w_norm = float(mlp.base_mlp.w3.weight.detach().to(torch.float64).norm().item())

            search_res = search_decision_energy_rank1_edit(
                mlp,
                dir_info,
                model.head.weight,
                eval_fn,
                target_token_ids=target_token_ids,
                baseline_token_ids=baseline_token_ids,
                alpha_grid=alpha_grid,
                weight_attr="w3",
                expert_idx=expert_idx,
                reference_weight_norm=ref_w_norm,
                config=cfg,
            )
            receipt = apply_verified_skill_edit_inplace(
                mlp,
                search_res,
                eval_energy_fn=eval_fn,
                weight_attr="w3",
                expert_idx=expert_idx,
                reference_weight_norm=ref_w_norm,
                config=cfg,
            )
            if receipt["committed"]:
                committed_layers.append(l_idx)
            else:
                rolled_back_layers.append(l_idx)
            layer_reports.append(
                {
                    "layer": l_idx,
                    "oof_snr": dir_info["oof_snr"],
                    "nullspace_dim": dir_info["nullspace_dim"],
                    "nullspace_mode_used": dir_info["nullspace_mode_used"],
                    "preserve_max_abs": dir_info["preserve_max_abs"],
                    **receipt,
                }
            )

        final_eval = eval_fn(model)
        final_tool_nll = float(final_eval["tool_nll"])
        final_general_nll = float(final_eval["general_nll"])
        tool_gain_pct = 100.0 * (initial_tool_nll - final_tool_nll) / max(abs(initial_tool_nll), 1e-8)
        gen_drift_pct = 100.0 * (final_general_nll - initial_general_nll) / max(abs(initial_general_nll), 1e-8)
        preserve_drifts_pct = {
            dom: 100.0 * (float(final_eval["domain_nlls"][dom]) - float(initial_eval["domain_nlls"][dom]))
            / max(abs(float(initial_eval["domain_nlls"][dom])), 1e-8)
            for dom in final_eval["domain_nlls"]
        }

        return {
            "committed": bool(committed_layers),
            "committed_layers": committed_layers,
            "rolled_back_layers": rolled_back_layers,
            "tool_nll_before": initial_tool_nll,
            "tool_nll_after": final_tool_nll,
            "tool_nll_gain_pct": tool_gain_pct,
            "general_nll_before": initial_general_nll,
            "general_nll_after": final_general_nll,
            "general_drift_pct": gen_drift_pct,
            "domain_drifts_pct": preserve_drifts_pct,
            "layer_reports": layer_reports,
            "config": asdict(cfg),
        }
    finally:
        model.train(was_training)


__all__ = [
    "DEFAULT_ANSE_MAX_PAIN",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_DRIFT_PENALTY_WEIGHT",
    "DEFAULT_MAX_EFFECTIVE_RANK_PERTURBATION_RATIO",
    "DEFAULT_MAX_GENERAL_DRIFT",
    "DEFAULT_MAX_NORM_PERTURBATION_RATIO",
    "DEFAULT_MIN_CONTRAST_SNR",
    "DEFAULT_NULLSPACE_EIGEN_RATIO",
    "DEFAULT_PERTURBATION_FRACTIONS",
    "DEFAULT_PRESERVE_WEIGHT",
    "DEFAULT_RIDGE_SCALE",
    "DEFAULT_SAFE_DRIFT_THRESHOLD",
    "DEFAULT_SOFT_MARGIN_PENALTY",
    "DecisionSkillSteeringConfig",
    "MAX_PERTURBATION_RATIO_CEILING",
    "apply_verified_skill_edit_inplace",
    "compute_contrastive_skill_direction",
    "compute_target_state_sha256",
    "inject_decision_skill_into_model_inplace",
    "make_skill_energy_eval_fn",
    "restore_target_state_inplace",
    "search_decision_energy_rank1_edit",
    "snapshot_target_state",
]

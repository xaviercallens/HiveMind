# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tangent-Space Task-Vector Disentanglement & Orthogonal Multi-Skill Composition Surgeon.

Implements training-free Weight-Space Programming composition for open-weight Sparse MoEs
(``CrowdV1WithEngram``, ``CrowdLoRAMoEMLP``, ``CrowdFineGrainedMoEMLP``, and generic
``nn.Module`` layers), inspired by Ortiz-Jimenez et al. (NeurIPS 2023, *Task Arithmetic
in the Tangent Space*) and Gargiulo et al. (CVPR 2025, *Task Singular Vectors / TSV-Merge*):

1. ``compute_activation_weighted_task_interference(task_deltas, domain_inputs)``:
   Measures exact CPU ``float64`` Frobenius and tangent-space activation-weighted pairwise
   cosine interference as well as cross-domain leakage across ``K`` skill task vectors.
2. ``orthogonalize_task_vectors_polar_nullspace(task_deltas, p_null, *, task_weights=None)``:
   Projects every task vector into the AlphaEdit general-prose null-space ``P_0`` and
   applies symmetric Loewdin / polar singular-vector orthogonalization across stacked
   subspaces so distinct skills occupy mutually orthogonal output/input subspaces strictly
   inside ``range(P_0)``, supporting both positive composition and negative skill unlearning.
3. ``execute_verified_tangent_task_composition(...)``:
   Composes ``K`` specialist LoRA/dense/bucket task vectors across open-weight MoE layers
   without bilinear cross-terms or rank inflation, gated strictly by ``ANSE Delta E < 0``
   and general-domain drift ``<= +1.0%`` with atomic bitwise rollback and SHA-256 receipts.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from hivemind.swarm.byzantine_wsp_delta_aggregator import (
    canonical_json_bytes,
    compute_alphaedit_prose_nullspace,
    compute_payload_sha256,
    compute_state_sha256,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdFineGrainedMoEMLP,
    CrowdLoRAExpertAdapter,
    CrowdLoRAMoEMLP,
)
from hivemind.swarm.lora_svd_mode_surgery import decompose_lora_qr_svd
from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    compute_swiglu_intermediate_states,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    forward_model_pre_ln,
    per_chunk_nll_from_pre_ln,
)

SCHEMA_VERSION: str = "hivemind.tangent_task_vector_surgeon/1"
RECEIPT_OBJECT: str = "hivemind.tangent_task_vector_composition_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_SV_REL_TOL: float = 1e-8
DEFAULT_POLAR_SV_REL_TOL: float = 1e-10
DEFAULT_NULLSPACE_EIG_THRESHOLD: float = 0.5
DEFAULT_NULLSPACE_SV_REL_TOL: float = 1e-3
DEFAULT_NULLSPACE_VARIANCE_THRESHOLD: float = 0.999
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_ANSE_SOFT_MARGIN_PENALTY: float = 0.05
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_SCORE_FROM: int = 64
DEFAULT_BATCH_SIZE: int = 24

VALID_PROJECTION_SIDES: frozenset[str] = frozenset({"auto", "input", "output"})
VALID_DELTA_FORMATS: frozenset[str] = frozenset({"auto", "state_dict", "buckets"})

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_LAYER_KEY_RE = re.compile(r"(?:^|\.)(?:blocks|layers)\.(\d+)\.")
_EXPERT_FACTOR_RE = re.compile(
    r"^(blocks\.(\d+)\.mlp\.experts\.(\d+))\.(w[123])(?:_([AB]))?$"
)
_LAYERNORM_MARKERS: tuple[str, ...] = (
    "ln.weight",
    "ln.bias",
    "ln1.",
    "ln2.",
    "layernorm",
    "rmsnorm",
)


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
    allow_zero: bool = False,
) -> float:
    """Validate a finite real value in ``[0, 1]`` (or ``(0, 1]``)."""
    out = _validate_nonneg_finite(value, name)
    if not allow_zero and out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    if out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    return out


def _validate_nonempty_str(value: Any, name: str) -> str:
    """Validate a non-empty string free of ASCII control characters."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped or any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{name} must be a non-empty printable string")
    return value


def _validate_finite_tensor(
    tensor: Any,
    name: str,
    *,
    allowed_ndims: tuple[int, ...] = (1, 2),
) -> torch.Tensor:
    """Validate a non-empty finite floating-point tensor and return CPU ``float64``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim not in allowed_ndims or any(dim == 0 for dim in tensor.shape):
        raise ValueError(
            f"{name} must have ndim in {allowed_ndims} and be non-empty, got shape {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    out = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(out).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _validate_square_projector(
    proj: Any,
    name: str,
    *,
    expected_dim: int | None = None,
) -> torch.Tensor:
    """Validate a non-empty finite square 2D projector matrix on CPU ``float64``."""
    mat = _validate_finite_tensor(proj, name, allowed_ndims=(2,))
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be square, got shape {tuple(mat.shape)}")
    if expected_dim is not None and int(mat.shape[0]) != expected_dim:
        raise ValueError(f"{name} shape {tuple(mat.shape)} != expected ({expected_dim}, {expected_dim})")
    return (0.5 * (mat + mat.T)).contiguous()


def _is_layernorm_key(key: str) -> bool:
    """Return True if parameter key refers to a LayerNorm or RMSNorm tensor."""
    low = key.lower()
    return any(marker in low for marker in _LAYERNORM_MARKERS)


def verify_tangent_composition_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a mapping, got {type(receipt).__name__}")
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


def _restore_snapshot_bitwise(
    model: nn.Module,
    snapshot: Mapping[str, torch.Tensor],
    expected_sha256: str,
) -> str:
    """Atomically restore ``model`` state in-place from CPU snapshot and verify SHA-256."""
    cur_sd = model.state_dict()
    for key, snap_t in snapshot.items():
        if key in cur_sd:
            cur_sd[key].copy_(snap_t.to(device=cur_sd[key].device, dtype=cur_sd[key].dtype))
    restored_sha = compute_state_sha256(model)
    if restored_sha != expected_sha256:
        raise RuntimeError(
            f"Atomic bitwise rollback failed: expected {expected_sha256[:12]}, got {restored_sha[:12]}"
        )
    return restored_sha


@dataclass(frozen=True)
class TangentTaskVectorConfig:
    """Validated configuration governing tangent-space task-vector orthogonalization & ANSE gate."""

    rank_per_task: int | None = None
    variance_threshold: float | None = None
    sv_rel_tol: float = DEFAULT_SV_REL_TOL
    polar_sv_rel_tol: float = DEFAULT_POLAR_SV_REL_TOL
    nullspace_eig_threshold: float = DEFAULT_NULLSPACE_EIG_THRESHOLD
    nullspace_sv_rel_tol: float = DEFAULT_NULLSPACE_SV_REL_TOL
    nullspace_variance_threshold: float | None = DEFAULT_NULLSPACE_VARIANCE_THRESHOLD
    nullspace_max_active_modes: int | None = None
    projection_side: str = "auto"
    protect_layernorm: bool = True
    orthogonalize_router: bool = True
    project_router_into_nullspace: bool = True
    domain_guard_max_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY
    enforce_all_domain_guard: bool = True
    require_strict_energy_drop: bool = True
    min_energy_improvement: float = 0.0
    general_prefix: str = GENERAL_EXPERT_PREFIX
    unlearn_domain_prefixes: tuple[str, ...] = ()
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    inplace: bool = True

    def __post_init__(self) -> None:
        if self.rank_per_task is not None:
            _validate_positive_int(self.rank_per_task, "rank_per_task")
        if self.variance_threshold is not None:
            _validate_unit_interval(self.variance_threshold, "variance_threshold", allow_zero=False)
        _validate_nonneg_finite(self.sv_rel_tol, "sv_rel_tol")
        _validate_nonneg_finite(self.polar_sv_rel_tol, "polar_sv_rel_tol")
        eig_t = _validate_unit_interval(
            self.nullspace_eig_threshold, "nullspace_eig_threshold", allow_zero=True
        )
        if eig_t >= 1.0:
            raise ValueError(f"nullspace_eig_threshold must be < 1.0, got {eig_t}")
        _validate_nonneg_finite(self.nullspace_sv_rel_tol, "nullspace_sv_rel_tol")
        if self.nullspace_variance_threshold is not None:
            _validate_unit_interval(
                self.nullspace_variance_threshold, "nullspace_variance_threshold", allow_zero=False
            )
        if self.nullspace_max_active_modes is not None:
            _validate_positive_int(self.nullspace_max_active_modes, "nullspace_max_active_modes")
        if self.projection_side not in VALID_PROJECTION_SIDES:
            raise ValueError(
                f"projection_side must be one of {sorted(VALID_PROJECTION_SIDES)}, got {self.projection_side!r}"
            )
        _validate_bool(self.protect_layernorm, "protect_layernorm")
        _validate_bool(self.orthogonalize_router, "orthogonalize_router")
        _validate_bool(self.project_router_into_nullspace, "project_router_into_nullspace")
        guard_r = _validate_positive_finite(self.domain_guard_max_ratio, "domain_guard_max_ratio")
        if guard_r < 1.0:
            raise ValueError(f"domain_guard_max_ratio must be >= 1.0, got {guard_r}")
        safe_r = _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        if safe_r < 1.0:
            raise ValueError(f"anse_safe_general_ratio must be >= 1.0, got {safe_r}")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_bool(self.enforce_all_domain_guard, "enforce_all_domain_guard")
        _validate_bool(self.require_strict_energy_drop, "require_strict_energy_drop")
        _validate_nonneg_finite(self.min_energy_improvement, "min_energy_improvement")
        _validate_nonempty_str(self.general_prefix, "general_prefix")
        if not isinstance(self.unlearn_domain_prefixes, tuple) or any(
            not isinstance(p, str) or not p.strip() for p in self.unlearn_domain_prefixes
        ):
            raise ValueError("unlearn_domain_prefixes must be a tuple of non-empty strings")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_bool(self.inplace, "inplace")


# ---------------------------------------------------------------------------
# 2. Scale-Invariant AlphaEdit Prose Null-Space & Input Normalization
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_prose_nullspace_projector_scaled(
    prose_keys: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor],
    *,
    sv_rel_tol: float = DEFAULT_NULLSPACE_SV_REL_TOL,
    variance_threshold: float | None = DEFAULT_NULLSPACE_VARIANCE_THRESHOLD,
    max_active_modes: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build scale-invariant AlphaEdit prose null-space projector ``P_0 = I - V_active^T V_active``."""
    tol = _validate_nonneg_finite(sv_rel_tol, "sv_rel_tol")
    if isinstance(prose_keys, torch.Tensor):
        items: list[tuple[str, torch.Tensor]] = [("prose_keys", prose_keys)]
    elif isinstance(prose_keys, Mapping):
        if not prose_keys:
            raise ValueError("prose_keys mapping must not be empty")
        items = [(f"prose_keys[{k!r}]", v) for k, v in sorted(prose_keys.items(), key=lambda kv: str(kv[0]))]
    elif isinstance(prose_keys, Sequence) and not isinstance(prose_keys, (str, bytes, bytearray)):
        if not prose_keys:
            raise ValueError("prose_keys sequence must not be empty")
        items = [(f"prose_keys[{idx}]", v) for idx, v in enumerate(prose_keys)]
    else:
        raise TypeError(
            f"prose_keys must be a torch.Tensor, Sequence[Tensor], or Mapping[str, Tensor]; got {type(prose_keys).__name__}"
        )

    parts: list[torch.Tensor] = []
    width: int | None = None
    for label, tensor in items:
        mat = _validate_finite_tensor(tensor, label, allowed_ndims=(2, 3))
        flat = mat.reshape(-1, int(mat.shape[-1]))
        if width is None:
            width = int(flat.shape[1])
        elif int(flat.shape[1]) != width:
            raise ValueError(f"{label} width {flat.shape[1]} != expected {width}")
        parts.append(flat)

    merged = torch.cat(parts, dim=0).contiguous()
    dim = int(merged.shape[1])
    frob = float(torch.linalg.vector_norm(merged).item())
    if frob <= 1e-15:
        return torch.eye(dim, dtype=torch.float64), {
            "dim": dim,
            "active_modes": 0,
            "nullspace_dim": dim,
            "variance_captured": 0.0,
            "top_sv": 0.0,
        }
    return compute_alphaedit_prose_nullspace(
        merged / frob,
        sv_threshold=tol,
        variance_threshold=variance_threshold,
        max_active_modes=max_active_modes,
    )


def _normalize_task_collection(
    task_deltas: Sequence[torch.Tensor | Mapping[str, torch.Tensor]]
    | Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
) -> tuple[str, list[str], list[Any], tuple[str, ...] | None, bool]:
    """Normalize single-tensor or multi-param task vectors into canonical lists.

    Returns ``(mode, task_names, normalized_items, param_keys, was_mapping)`` where
    ``mode`` is ``'tensor'`` or ``'dict'``.
    """
    was_mapping = isinstance(task_deltas, Mapping)
    if was_mapping:
        if not task_deltas:
            raise ValueError("task_deltas must not be empty")
        task_names = [_validate_nonempty_str(k, "task_name") for k in task_deltas.keys()]
        raw_items = list(task_deltas.values())
    elif isinstance(task_deltas, Sequence) and not isinstance(task_deltas, (str, bytes, bytearray)):
        if not task_deltas:
            raise ValueError("task_deltas must not be empty")
        task_names = [f"task_{idx}" for idx in range(len(task_deltas))]
        raw_items = list(task_deltas)
    else:
        raise TypeError(
            "task_deltas must be a non-empty Sequence or Mapping of task tensors or parameter delta dicts"
        )

    first = raw_items[0]
    if isinstance(first, torch.Tensor):
        tensors: list[torch.Tensor] = []
        ref_shape: tuple[int, ...] | None = None
        for idx, (t_name, item) in enumerate(zip(task_names, raw_items, strict=True)):
            if not isinstance(item, torch.Tensor):
                raise TypeError(
                    f"task_deltas[{t_name!r}] must be a torch.Tensor to match task 0, got {type(item).__name__}"
                )
            t64 = _validate_finite_tensor(item, f"task_deltas[{idx}]", allowed_ndims=(1, 2))
            if ref_shape is None:
                ref_shape = tuple(t64.shape)
            elif tuple(t64.shape) != ref_shape:
                raise ValueError(
                    f"Shape mismatch on task {t_name!r}: {tuple(t64.shape)} != {ref_shape}"
                )
            tensors.append(item)
        return "tensor", task_names, tensors, None, was_mapping

    if isinstance(first, Mapping):
        dicts: list[dict[str, torch.Tensor]] = []
        ref_keys: tuple[str, ...] | None = None
        ref_shapes: dict[str, tuple[int, ...]] = {}
        for idx, (t_name, item) in enumerate(zip(task_names, raw_items, strict=True)):
            if not isinstance(item, Mapping) or not item:
                raise ValueError(f"task_deltas[{t_name!r}] must be a non-empty Mapping[str, torch.Tensor]")
            cur_map: dict[str, torch.Tensor] = {}
            for k, v in item.items():
                k_str = _validate_nonempty_str(k, f"task_deltas[{idx}] key")
                _validate_finite_tensor(v, f"task_deltas[{idx}][{k_str!r}]", allowed_ndims=(1, 2))
                cur_map[k_str] = v
            keys_sorted = tuple(sorted(cur_map.keys()))
            if ref_keys is None:
                ref_keys = keys_sorted
                ref_shapes = {k: tuple(cur_map[k].shape) for k in ref_keys}
            elif keys_sorted != ref_keys:
                raise ValueError(f"task_deltas[{t_name!r}] keys do not match first task schema")
            else:
                for k in ref_keys:
                    if tuple(cur_map[k].shape) != ref_shapes[k]:
                        raise ValueError(
                            f"Shape mismatch on task {t_name!r}[{k!r}]: {tuple(cur_map[k].shape)} != {ref_shapes[k]}"
                        )
            dicts.append(cur_map)
        return "dict", task_names, dicts, ref_keys, was_mapping

    raise TypeError(
        f"task_deltas entries must be torch.Tensor or Mapping[str, torch.Tensor], got {type(first).__name__}"
    )


def _normalize_task_weights(task_weights: Sequence[float] | None, num_tasks: int) -> list[float]:
    """Validate finite task arithmetic weights (negative weights permitted for unlearning)."""
    if task_weights is None:
        return [1.0] * num_tasks
    if isinstance(task_weights, (str, bytes, bytearray)) or not isinstance(task_weights, Sequence):
        raise TypeError("task_weights must be a sequence of real numbers or None")
    if len(task_weights) != num_tasks:
        raise ValueError(
            f"task_weights length ({len(task_weights)}) must match task count ({num_tasks})"
        )
    vals = [_validate_finite_float(w, f"task_weights[{idx}]") for idx, w in enumerate(task_weights)]
    if all(abs(v) <= 0.0 for v in vals):
        raise ValueError("task_weights must contain at least one non-zero weight")
    return vals


def _flatten_input_tensor(tensor: Any, name: str, expected_dim: int) -> torch.Tensor:
    """Validate 2D/3D tangent activation tensor and flatten to CPU ``float64`` ``(N, expected_dim)``."""
    t64 = _validate_finite_tensor(tensor, name, allowed_ndims=(2, 3))
    if int(t64.shape[-1]) != expected_dim:
        raise ValueError(f"{name} feature dim {t64.shape[-1]} != expected {expected_dim}")
    return t64.reshape(-1, expected_dim).contiguous()


def _resolve_tensor_domain_inputs(
    domain_inputs: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor],
    task_names: Sequence[str],
    expected_dim: int,
) -> list[torch.Tensor]:
    """Resolve shared or per-task tangent input activation matrices ``[X_0, ..., X_{K-1}]``."""
    k_tasks = len(task_names)
    if isinstance(domain_inputs, torch.Tensor):
        shared = _flatten_input_tensor(domain_inputs, "domain_inputs", expected_dim)
        return [shared] * k_tasks
    if isinstance(domain_inputs, Mapping):
        if not domain_inputs:
            raise ValueError("domain_inputs mapping must not be empty")
        if all(name in domain_inputs for name in task_names):
            return [
                _flatten_input_tensor(domain_inputs[name], f"domain_inputs[{name!r}]", expected_dim)
                for name in task_names
            ]
        if len(domain_inputs) == k_tasks:
            return [
                _flatten_input_tensor(val, f"domain_inputs[{key!r}]", expected_dim)
                for key, val in domain_inputs.items()
            ]
        raise ValueError(
            f"domain_inputs keys {sorted(domain_inputs.keys())} do not match tasks {list(task_names)}"
        )
    if isinstance(domain_inputs, Sequence) and not isinstance(domain_inputs, (str, bytes, bytearray)):
        if len(domain_inputs) != k_tasks:
            raise ValueError(
                f"domain_inputs length ({len(domain_inputs)}) must match task count ({k_tasks})"
            )
        return [
            _flatten_input_tensor(item, f"domain_inputs[{idx}]", expected_dim)
            for idx, item in enumerate(domain_inputs)
        ]
    raise TypeError(
        "domain_inputs must be a torch.Tensor, Sequence[Tensor], or Mapping[str, Tensor]"
    )


# ---------------------------------------------------------------------------
# 3. Activation-Weighted Task Interference Diagnostics
# ---------------------------------------------------------------------------


def _cosine_and_leakage_from_gram(
    frob_gram: torch.Tensor,
    act_gram: torch.Tensor,
    per_domain_grams: list[torch.Tensor],
    task_names: list[str],
    eps: float,
) -> dict[str, Any]:
    """Derive exact pairwise Frobenius/activation cosine matrices and leakage metrics from Grams."""
    k_tasks = len(task_names)

    def _gram_to_cos(gram: torch.Tensor) -> tuple[list[list[float]], list[float]]:
        norms = torch.diagonal(gram).clamp_min(0.0).sqrt()
        denom = torch.outer(norms, norms)
        safe = torch.where(denom > eps, gram / denom.clamp_min(eps), torch.zeros_like(gram))
        safe = safe.clamp(-1.0, 1.0)
        return [[float(safe[i, j].item()) for j in range(k_tasks)] for i in range(k_tasks)], [
            float(norms[i].item()) for i in range(k_tasks)
        ]

    fro_mat, fro_norms = _gram_to_cos(frob_gram)
    act_mat, act_norms = _gram_to_cos(act_gram)

    dom_cos_mat: list[list[float]] = []
    leak_mat: list[list[float]] = []
    off_fro: list[float] = []
    off_act: list[float] = []
    off_leak: list[float] = []
    fro_energy = 0.0
    act_energy = 0.0

    for i in range(k_tasks):
        gi = per_domain_grams[i]
        di_norms = torch.diagonal(gi).clamp_min(0.0).sqrt()
        own_norm = max(float(di_norms[i].item()), eps)
        cos_row: list[float] = []
        leak_row: list[float] = []
        for j in range(k_tasks):
            denom_ij = float(di_norms[i].item()) * float(di_norms[j].item())
            c_ij = float(gi[i, j].item()) / denom_ij if denom_ij > eps else 0.0
            cos_row.append(max(-1.0, min(1.0, c_ij)))
            l_ij = float(di_norms[j].item()) / own_norm if float(di_norms[i].item()) > eps else 0.0
            leak_row.append(l_ij)
            if i != j:
                off_leak.append(l_ij)
            if j > i:
                off_fro.append(abs(fro_mat[i][j]))
                off_act.append(abs(act_mat[i][j]))
                fro_energy += abs(float(frob_gram[i, j].item()))
                act_energy += abs(float(act_gram[i, j].item()))
        dom_cos_mat.append(cos_row)
        leak_mat.append(leak_row)

    mean_fro = float(sum(off_fro) / len(off_fro)) if off_fro else 0.0
    max_fro = float(max(off_fro)) if off_fro else 0.0
    mean_act = float(sum(off_act) / len(off_act)) if off_act else 0.0
    max_act = float(max(off_act)) if off_act else 0.0
    mean_leak = float(sum(off_leak) / len(off_leak)) if off_leak else 0.0
    max_leak = float(max(off_leak)) if off_leak else 0.0

    return {
        "num_tasks": k_tasks,
        "task_names": list(task_names),
        "task_frobenius_norms": fro_norms,
        "task_activation_norms": act_norms,
        "frobenius_cosine_matrix": fro_mat,
        "activation_cosine_matrix": act_mat,
        "domain_conditioned_cosine_matrix": dom_cos_mat,
        "cross_domain_leakage_matrix": leak_mat,
        "mean_abs_frobenius_cosine": mean_fro,
        "max_abs_frobenius_cosine": max_fro,
        "mean_abs_activation_cosine": mean_act,
        "max_abs_activation_cosine": max_act,
        "cosine_interference": mean_act,
        "frobenius_interference_energy": fro_energy,
        "activation_interference_energy": act_energy,
        "mean_cross_domain_leakage": mean_leak,
        "max_cross_domain_leakage": max_leak,
    }


def _compute_2d_response_gram(
    deltas_64: Sequence[torch.Tensor],
    x_mat: torch.Tensor,
) -> torch.Tensor:
    """Compute exact ``(K, K)`` Gram matrix ``G_ij = <X Delta_i^T, X Delta_j^T>_F / N``."""
    n = float(x_mat.shape[0])
    m2 = (x_mat.T @ x_mat) / n
    k = len(deltas_64)
    gram = torch.zeros((k, k), dtype=torch.float64)
    for i in range(k):
        di = deltas_64[i]
        di_m = di @ m2 if di.ndim == 2 else (di.unsqueeze(0) @ m2)
        for j in range(i, k):
            dj = deltas_64[j] if deltas_64[j].ndim == 2 else deltas_64[j].unsqueeze(0)
            val = float((di_m * dj).sum().item())
            gram[i, j] = val
            gram[j, i] = val
    return gram


@torch.no_grad()
def compute_activation_weighted_task_interference(
    task_deltas: Sequence[torch.Tensor | Mapping[str, torch.Tensor]]
    | Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
    domain_inputs: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor],
    *,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Measure exact Frobenius and tangent-space activation-weighted pairwise interference across ``K`` tasks."""
    eps_f = _validate_positive_finite(eps, "eps")
    mode, task_names, items, param_keys, _ = _normalize_task_collection(task_deltas)
    k_tasks = len(task_names)

    if mode == "tensor":
        deltas_64 = [t.detach().to(device="cpu", dtype=torch.float64) for t in items]
        d_in = int(deltas_64[0].shape[-1])
        x_list = _resolve_tensor_domain_inputs(domain_inputs, task_names, d_in)
        frob_gram = torch.zeros((k_tasks, k_tasks), dtype=torch.float64)
        for i in range(k_tasks):
            for j in range(i, k_tasks):
                v = float((deltas_64[i] * deltas_64[j]).sum().item())
                frob_gram[i, j] = v
                frob_gram[j, i] = v
        per_dom_grams = [_compute_2d_response_gram(deltas_64, x_k) for x_k in x_list]
        act_gram = sum(per_dom_grams, torch.zeros((k_tasks, k_tasks), dtype=torch.float64)) / float(k_tasks)
        return _cosine_and_leakage_from_gram(frob_gram, act_gram, per_dom_grams, task_names, eps_f)

    if not isinstance(domain_inputs, Mapping) or not domain_inputs:
        raise TypeError(
            "domain_inputs must be a non-empty Mapping[str, torch.Tensor] when task_deltas are parameter dicts"
        )
    assert param_keys is not None
    frob_gram = torch.zeros((k_tasks, k_tasks), dtype=torch.float64)
    act_gram = torch.zeros((k_tasks, k_tasks), dtype=torch.float64)
    matched_params = 0
    for key in param_keys:
        mats = [item[key].detach().to(device="cpu", dtype=torch.float64) for item in items]
        for i in range(k_tasks):
            for j in range(i, k_tasks):
                vf = float((mats[i] * mats[j]).sum().item())
                frob_gram[i, j] += vf
                frob_gram[j, i] += vf
        d_in = int(mats[0].shape[-1])
        x_act = _resolve_param_activation(key, d_in, domain_inputs)
        if x_act is not None:
            matched_params += 1
            act_gram += _compute_2d_response_gram(mats, x_act)
    if matched_params == 0:
        raise ValueError("No domain_inputs tensor matched the feature dimensions of task_deltas")
    return _cosine_and_leakage_from_gram(
        frob_gram, act_gram, [act_gram] * k_tasks, task_names, eps_f
    )


def _resolve_param_activation(
    key: str,
    expected_dim: int,
    inputs_map: Mapping[str, torch.Tensor],
) -> torch.Tensor | None:
    """Resolve tangent activation matrix matching parameter ``key`` and input width ``expected_dim``."""
    layer_m = _LAYER_KEY_RE.search(key)
    layer_tag = f"blocks.{layer_m.group(1)}" if layer_m is not None else None
    is_act = key.endswith("w3_A") or key.endswith(".w3") or key.endswith(".w3.weight") or "down_proj" in key
    cands: list[str] = [key]
    if layer_tag is not None:
        cands.append(f"{layer_tag}.act" if is_act else f"{layer_tag}.x")
    cands.extend(("act", "x", "default") if is_act else ("x", "act", "default"))
    for cand in cands:
        if cand in inputs_map:
            t = _validate_finite_tensor(inputs_map[cand], f"domain_inputs[{cand!r}]", allowed_ndims=(2, 3))
            if int(t.shape[-1]) == expected_dim:
                return t.reshape(-1, expected_dim).contiguous()
    for _, val in sorted(inputs_map.items()):
        t = _validate_finite_tensor(val, "domain_inputs_entry", allowed_ndims=(2, 3))
        if int(t.shape[-1]) == expected_dim:
            return t.reshape(-1, expected_dim).contiguous()
    return None


# ---------------------------------------------------------------------------
# 4. Symmetric Loewdin / Polar Orthogonalization in AlphaEdit Null-Space P_0
# ---------------------------------------------------------------------------


def _extract_nullspace_basis(
    p_null: torch.Tensor | None,
    dim: int,
    eig_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(P_sym, Q_null)`` where ``Q_null`` spans the kept nullspace modes."""
    if p_null is None:
        eye_d = torch.eye(dim, dtype=torch.float64)
        return eye_d, eye_d
    p_sym = _validate_square_projector(p_null, "p_null", expected_dim=dim)
    evals, evecs = torch.linalg.eigh(p_sym)
    mask = evals > eig_threshold
    if not bool(mask.any().item()):
        return p_sym, torch.zeros((dim, 0), dtype=torch.float64)
    q_null = evecs[:, mask].contiguous()
    return p_sym, q_null


def _polar_factor_svd(
    mat: torch.Tensor,
    *,
    sv_rel_tol: float,
) -> tuple[torch.Tensor, int, float]:
    """Compute symmetric Loewdin / polar orthogonal factor ``P @ Q^T`` of ``mat``."""
    cols = int(mat.shape[1])
    if cols == 0 or float(mat.abs().max().item()) == 0.0:
        return torch.zeros_like(mat), 0, 0.0
    p_mat, s_vals, qt_mat = torch.linalg.svd(mat, full_matrices=False)
    top_s = max(float(s_vals[0].item()), 1.0)
    mask = s_vals > (sv_rel_tol * top_s)
    eff_rank = int(mask.sum().item())
    if eff_rank == 0:
        return torch.zeros_like(mat), 0, 0.0
    orth = (p_mat[:, mask] @ qt_mat[mask, :]).contiguous()
    eye_c = torch.eye(cols, dtype=torch.float64)
    orth_err = float(torch.linalg.matrix_norm(orth.T @ orth - eye_c, ord="fro").item())
    return orth, eff_rank, orth_err


def _allocate_task_ranks(
    nz_ranks: list[int],
    d_eff: int,
    rank_per_task: int | None,
) -> list[int]:
    """Allocate per-task SVD ranks inside ``d_eff`` nullspace capacity via water-filling."""
    ranks = [min(r, rank_per_task) if rank_per_task is not None else r for r in nz_ranks]
    active_indices = [i for i, r in enumerate(ranks) if r > 0]
    if not active_indices or d_eff <= 0:
        return [0] * len(nz_ranks)
    if sum(ranks) <= d_eff:
        return ranks

    # Water-filling trim from highest rank down while keeping at least 1 mode per active task
    min_floor = 1
    while sum(ranks) > d_eff:
        max_r = max(ranks[i] for i in active_indices)
        if max_r <= min_floor:
            break
        for i in reversed(active_indices):
            if ranks[i] == max_r and sum(ranks) > d_eff:
                ranks[i] -= 1
    return ranks


def _pairwise_frobenius_summary(mats: Sequence[torch.Tensor]) -> tuple[float, float]:
    """Return ``(mean_abs_offdiag_cos, max_abs_offdiag_cos)`` over a sequence of tensors."""
    k = len(mats)
    if k < 2:
        return 0.0, 0.0
    vals: list[float] = []
    norms = [float(torch.linalg.vector_norm(m).item()) for m in mats]
    for i in range(k):
        for j in range(i + 1, k):
            denom = norms[i] * norms[j]
            cos_ij = abs(float((mats[i] * mats[j]).sum().item()) / denom) if denom > 1e-15 else 0.0
            vals.append(min(1.0, cos_ij))
    return float(sum(vals) / len(vals)), float(max(vals))


def _orthogonalize_1d_task_vectors(
    deltas_64: list[torch.Tensor],
    weights: list[float],
    p_in: torch.Tensor | None,
    *,
    polar_sv_rel_tol: float,
    eig_threshold: float,
) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, Any]]:
    """Apply polar orthogonalization inside ``P_0`` to 1D task vectors."""
    dim = int(deltas_64[0].shape[0])
    p_sym, q_null = _extract_nullspace_basis(p_in, dim, eig_threshold)
    r_null = int(q_null.shape[1])
    mean_before, max_before = _pairwise_frobenius_summary(deltas_64)

    if r_null == 0:
        zeros = [torch.zeros(dim, dtype=torch.float64) for _ in deltas_64]
        return zeros, torch.zeros(dim, dtype=torch.float64), {
            "projection_side": "input" if p_in is not None else "none",
            "nullspace_rank": 0,
            "allocated_ranks": [0] * len(deltas_64),
            "total_stacked_rank": 0,
            "effective_u_rank": 0,
            "effective_v_rank": 0,
            "u_orthogonality_error": 0.0,
            "v_orthogonality_error": 0.0,
            "max_pairwise_frobenius_cosine_before": max_before,
            "max_pairwise_frobenius_cosine_after": 0.0,
            "mean_abs_frobenius_cosine_before": mean_before,
            "mean_abs_frobenius_cosine_after": 0.0,
            "cosine_interference": 0.0,
            "nullspace_leakage_norm": 0.0,
            "composed_frobenius_norm": 0.0,
        }

    proj_null = [(d @ p_sym) @ q_null for d in deltas_64]
    norms = [float(torch.linalg.vector_norm(v).item()) for v in proj_null]
    active = [idx for idx, n in enumerate(norms) if n > 1e-15]
    if not active:
        zeros = [torch.zeros(dim, dtype=torch.float64) for _ in deltas_64]
        return zeros, torch.zeros(dim, dtype=torch.float64), {
            "projection_side": "input" if p_in is not None else "none",
            "nullspace_rank": r_null,
            "allocated_ranks": [0] * len(deltas_64),
            "total_stacked_rank": 0,
            "effective_u_rank": 0,
            "effective_v_rank": 0,
            "u_orthogonality_error": 0.0,
            "v_orthogonality_error": 0.0,
            "max_pairwise_frobenius_cosine_before": max_before,
            "max_pairwise_frobenius_cosine_after": 0.0,
            "mean_abs_frobenius_cosine_before": mean_before,
            "mean_abs_frobenius_cosine_after": 0.0,
            "cosine_interference": 0.0,
            "nullspace_leakage_norm": 0.0,
            "composed_frobenius_norm": 0.0,
        }

    cols = torch.stack([proj_null[i] / norms[i] for i in active], dim=1)
    v_null_orth, eff_v, v_err = _polar_factor_svd(cols, sv_rel_tol=polar_sv_rel_tol)
    v_amb = q_null @ v_null_orth
    orth_list = [torch.zeros(dim, dtype=torch.float64) for _ in deltas_64]
    alloc = [0] * len(deltas_64)
    for col_idx, task_idx in enumerate(active):
        orth_list[task_idx] = (v_amb[:, col_idx] * norms[task_idx]).contiguous()
        alloc[task_idx] = 1
    composed = sum(
        (w * v for w, v in zip(weights, orth_list, strict=True)),
        torch.zeros(dim, dtype=torch.float64),
    )
    mean_after, max_after = _pairwise_frobenius_summary(orth_list)
    eye_d = torch.eye(dim, dtype=torch.float64)
    leak = 0.0
    if p_in is not None:
        for v in orth_list:
            vn = float(torch.linalg.vector_norm(v).item())
            if vn > 1e-15:
                leak = max(leak, float(torch.linalg.vector_norm(v @ (eye_d - p_sym)).item()) / vn)
    return orth_list, composed, {
        "projection_side": "input" if p_in is not None else "none",
        "nullspace_rank": r_null,
        "allocated_ranks": alloc,
        "total_stacked_rank": len(active),
        "effective_u_rank": 1,
        "effective_v_rank": eff_v,
        "u_orthogonality_error": 0.0,
        "v_orthogonality_error": v_err,
        "max_pairwise_frobenius_cosine_before": max_before,
        "max_pairwise_frobenius_cosine_after": max_after,
        "mean_abs_frobenius_cosine_before": mean_before,
        "mean_abs_frobenius_cosine_after": mean_after,
        "cosine_interference": mean_after,
        "nullspace_leakage_norm": leak,
        "composed_frobenius_norm": float(torch.linalg.vector_norm(composed).item()),
    }


def _resolve_2d_projection_sides(
    d_out: int,
    d_in: int,
    p_null: torch.Tensor | None,
    p_null_out: torch.Tensor | None,
    projection_side: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
    """Validate and assign input/output nullspace projectors for a ``(d_out, d_in)`` matrix."""
    if projection_side not in VALID_PROJECTION_SIDES:
        raise ValueError(
            f"projection_side must be one of {sorted(VALID_PROJECTION_SIDES)}, got {projection_side!r}"
        )
    p_in: torch.Tensor | None = None
    p_out: torch.Tensor | None = None
    if p_null is not None:
        p_t = _validate_square_projector(p_null, "p_null")
        dp = int(p_t.shape[0])
        if projection_side == "input":
            if dp != d_in:
                raise ValueError(f"p_null shape {tuple(p_t.shape)} != input dim ({d_in}, {d_in})")
            p_in = p_t
        elif projection_side == "output":
            if dp != d_out:
                raise ValueError(f"p_null shape {tuple(p_t.shape)} != output dim ({d_out}, {d_out})")
            p_out = p_t
        else:
            if dp == d_in:
                p_in = p_t
            elif dp == d_out:
                p_out = p_t
            else:
                raise ValueError(
                    f"p_null shape {tuple(p_t.shape)} matches neither input dim {d_in} nor output dim {d_out}"
                )
    if p_null_out is not None:
        p_out = _validate_square_projector(p_null_out, "p_null_out", expected_dim=d_out)

    if p_in is not None and p_out is not None:
        side_label = "both"
    elif p_in is not None:
        side_label = "input"
    elif p_out is not None:
        side_label = "output"
    else:
        side_label = "none"
    return p_in, p_out, side_label


def _orthogonalize_2d_task_matrices(
    deltas_64: list[torch.Tensor],
    weights: list[float],
    p_in: torch.Tensor | None,
    p_out: torch.Tensor | None,
    side_label: str,
    *,
    rank_per_task: int | None,
    variance_threshold: float | None,
    sv_rel_tol: float,
    polar_sv_rel_tol: float,
    eig_threshold: float,
) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, Any]]:
    """Run symmetric Loewdin / polar SVD orthogonalization inside ``(P_out, P_in)``."""
    d_out, d_in = int(deltas_64[0].shape[0]), int(deltas_64[0].shape[1])
    k_tasks = len(deltas_64)
    p_in_sym, q_in = _extract_nullspace_basis(p_in, d_in, eig_threshold)
    p_out_sym, q_out = _extract_nullspace_basis(p_out, d_out, eig_threshold)
    r_in = int(q_in.shape[1])
    r_out = int(q_out.shape[1])
    null_rank_eff = r_out if side_label == "output" else (min(r_in, r_out) if side_label == "both" else r_in)

    mean_before, max_before = _pairwise_frobenius_summary(deltas_64)
    if r_in == 0 or r_out == 0:
        zeros = [torch.zeros((d_out, d_in), dtype=torch.float64) for _ in range(k_tasks)]
        return zeros, torch.zeros((d_out, d_in), dtype=torch.float64), {
            "projection_side": side_label,
            "nullspace_rank": 0,
            "allocated_ranks": [0] * k_tasks,
            "total_stacked_rank": 0,
            "effective_u_rank": 0,
            "effective_v_rank": 0,
            "u_orthogonality_error": 0.0,
            "v_orthogonality_error": 0.0,
            "max_pairwise_frobenius_cosine_before": max_before,
            "max_pairwise_frobenius_cosine_after": 0.0,
            "mean_abs_frobenius_cosine_before": mean_before,
            "mean_abs_frobenius_cosine_after": 0.0,
            "cosine_interference": 0.0,
            "nullspace_leakage_norm": 0.0,
            "composed_frobenius_norm": 0.0,
        }

    svds_u: list[torch.Tensor] = []
    svds_s: list[torch.Tensor] = []
    svds_v: list[torch.Tensor] = []
    nz_ranks: list[int] = []

    for d_mat in deltas_64:
        proj_mat = (p_out_sym @ d_mat @ p_in_sym) if p_out is not None else (d_mat @ p_in_sym)
        if p_in is not None and p_out is not None:
            core_mat = q_out.T @ proj_mat @ q_in
        elif p_in is not None:
            core_mat = proj_mat @ q_in
        elif p_out is not None:
            core_mat = q_out.T @ proj_mat
        else:
            core_mat = proj_mat

        u_k, s_k, vh_k = torch.linalg.svd(core_mat, full_matrices=False)
        if s_k.numel() == 0 or float(s_k[0].item()) <= 1e-15:
            svds_u.append(torch.zeros((r_out, 0), dtype=torch.float64))
            svds_s.append(torch.zeros((0,), dtype=torch.float64))
            svds_v.append(torch.zeros((r_in, 0), dtype=torch.float64))
            nz_ranks.append(0)
            continue

        s_mask = s_k > (sv_rel_tol * float(s_k[0].item()))
        n_nz = int(s_mask.sum().item())
        if variance_threshold is not None and n_nz > 1:
            e_tot = float(s_k[:n_nz].pow(2).sum().item())
            if e_tot > 0.0:
                cum = torch.cumsum(s_k[:n_nz].pow(2), dim=0) / e_tot
                kept = int((cum < variance_threshold).sum().item()) + 1
                n_nz = min(n_nz, kept)

        svds_u.append(u_k[:, :n_nz].contiguous())
        svds_s.append(s_k[:n_nz].contiguous())
        svds_v.append(vh_k[:n_nz, :].T.contiguous())
        nz_ranks.append(n_nz)

    d_eff = min(r_out, r_in)
    alloc_ranks = _allocate_task_ranks(nz_ranks, d_eff, rank_per_task)
    total_r = sum(alloc_ranks)
    if total_r == 0:
        zeros = [torch.zeros((d_out, d_in), dtype=torch.float64) for _ in range(k_tasks)]
        return zeros, torch.zeros((d_out, d_in), dtype=torch.float64), {
            "projection_side": side_label,
            "nullspace_rank": null_rank_eff,
            "allocated_ranks": alloc_ranks,
            "total_stacked_rank": 0,
            "effective_u_rank": 0,
            "effective_v_rank": 0,
            "u_orthogonality_error": 0.0,
            "v_orthogonality_error": 0.0,
            "max_pairwise_frobenius_cosine_before": max_before,
            "max_pairwise_frobenius_cosine_after": 0.0,
            "mean_abs_frobenius_cosine_before": mean_before,
            "mean_abs_frobenius_cosine_after": 0.0,
            "cosine_interference": 0.0,
            "nullspace_leakage_norm": 0.0,
            "composed_frobenius_norm": 0.0,
        }

    u_cat = torch.cat([svds_u[i][:, : alloc_ranks[i]] for i in range(k_tasks) if alloc_ranks[i] > 0], dim=1)
    v_cat = torch.cat([svds_v[i][:, : alloc_ranks[i]] for i in range(k_tasks) if alloc_ranks[i] > 0], dim=1)
    s_cat = [svds_s[i][: alloc_ranks[i]] for i in range(k_tasks)]

    u_core_orth, eff_u, u_err = _polar_factor_svd(u_cat, sv_rel_tol=polar_sv_rel_tol)
    v_core_orth, eff_v, v_err = _polar_factor_svd(v_cat, sv_rel_tol=polar_sv_rel_tol)

    u_orth_amb = (q_out @ u_core_orth) if p_out is not None else u_core_orth
    v_orth_amb = (q_in @ v_core_orth) if p_in is not None else v_core_orth

    orth_list: list[torch.Tensor] = []
    offset = 0
    for i in range(k_tasks):
        r_i = alloc_ranks[i]
        if r_i == 0:
            orth_list.append(torch.zeros((d_out, d_in), dtype=torch.float64))
            continue
        u_i = u_orth_amb[:, offset : offset + r_i]
        v_i = v_orth_amb[:, offset : offset + r_i]
        s_i = s_cat[i]
        offset += r_i
        orth_list.append(((u_i * s_i.unsqueeze(0)) @ v_i.T).contiguous())

    composed = sum(
        (w * m for w, m in zip(weights, orth_list, strict=True)),
        torch.zeros((d_out, d_in), dtype=torch.float64),
    )

    mean_after, max_after = _pairwise_frobenius_summary(orth_list)
    eye_in = torch.eye(d_in, dtype=torch.float64)
    eye_out = torch.eye(d_out, dtype=torch.float64)
    leak = 0.0
    for m in orth_list:
        mn = float(torch.linalg.vector_norm(m).item())
        if mn <= 1e-15:
            continue
        if p_in is not None:
            leak = max(leak, float(torch.linalg.vector_norm(m @ (eye_in - p_in_sym)).item()) / mn)
        if p_out is not None:
            leak = max(leak, float(torch.linalg.vector_norm((eye_out - p_out_sym) @ m).item()) / mn)

    return orth_list, composed, {
        "projection_side": side_label,
        "nullspace_rank": null_rank_eff,
        "allocated_ranks": alloc_ranks,
        "total_stacked_rank": total_r,
        "effective_u_rank": eff_u,
        "effective_v_rank": eff_v,
        "u_orthogonality_error": u_err,
        "v_orthogonality_error": v_err,
        "max_pairwise_frobenius_cosine_before": max_before,
        "max_pairwise_frobenius_cosine_after": max_after,
        "mean_abs_frobenius_cosine_before": mean_before,
        "mean_abs_frobenius_cosine_after": mean_after,
        "cosine_interference": mean_after,
        "nullspace_leakage_norm": leak,
        "composed_frobenius_norm": float(torch.linalg.vector_norm(composed).item()),
    }


@torch.no_grad()
def orthogonalize_task_vectors_polar_nullspace(
    task_deltas: Sequence[torch.Tensor | Mapping[str, torch.Tensor]]
    | Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
    p_null: torch.Tensor | Mapping[str, torch.Tensor] | None,
    *,
    task_weights: Sequence[float] | None = None,
    rank_per_task: int | None = None,
    variance_threshold: float | None = None,
    sv_rel_tol: float = DEFAULT_SV_REL_TOL,
    polar_sv_rel_tol: float = DEFAULT_POLAR_SV_REL_TOL,
    nullspace_eig_threshold: float = DEFAULT_NULLSPACE_EIG_THRESHOLD,
    projection_side: str = "auto",
    p_null_out: torch.Tensor | None = None,
) -> tuple[
    list[torch.Tensor] | dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | dict[str, dict[str, torch.Tensor]],
    torch.Tensor | dict[str, torch.Tensor],
    dict[str, Any],
]:
    """Orthogonalize ``K`` task vectors via symmetric Loewdin / polar SVD strictly inside ``P_0``.

    Returns ``(orthogonal_tasks, composed_delta, report)`` where ``orthogonal_tasks`` preserves
    the container shape/dtype of ``task_deltas`` and ``composed_delta = sum(w_k * tau_orth_k)``.
    Negative ``task_weights`` are permitted for orthogonalized skill negation / unlearning.
    """
    if rank_per_task is not None:
        _validate_positive_int(rank_per_task, "rank_per_task")
    if variance_threshold is not None:
        _validate_unit_interval(variance_threshold, "variance_threshold", allow_zero=False)
    sv_tol = _validate_nonneg_finite(sv_rel_tol, "sv_rel_tol")
    polar_tol = _validate_nonneg_finite(polar_sv_rel_tol, "polar_sv_rel_tol")
    eig_t = _validate_unit_interval(
        nullspace_eig_threshold, "nullspace_eig_threshold", allow_zero=True
    )
    if eig_t >= 1.0:
        raise ValueError(f"nullspace_eig_threshold must be < 1.0, got {eig_t}")

    mode, task_names, items, param_keys, was_mapping = _normalize_task_collection(task_deltas)
    weights = _normalize_task_weights(task_weights, len(task_names))

    if mode == "tensor":
        if isinstance(p_null, Mapping):
            raise TypeError("p_null must be a square 2D torch.Tensor or None for single-tensor task_deltas")
        ref_t: torch.Tensor = items[0]
        deltas_64 = [t.detach().to(device="cpu", dtype=torch.float64) for t in items]
        if deltas_64[0].ndim == 1:
            if p_null_out is not None:
                raise ValueError("p_null_out is only valid for 2D task matrices")
            if projection_side not in VALID_PROJECTION_SIDES:
                raise ValueError(
                    f"projection_side must be one of {sorted(VALID_PROJECTION_SIDES)}, got {projection_side!r}"
                )
            p_in_1d = (
                _validate_square_projector(p_null, "p_null", expected_dim=int(deltas_64[0].shape[0]))
                if p_null is not None
                else None
            )
            orth_64, comp_64, stats = _orthogonalize_1d_task_vectors(
                deltas_64,
                weights,
                p_in_1d,
                polar_sv_rel_tol=polar_tol,
                eig_threshold=eig_t,
            )
        else:
            p_in, p_out, side_label = _resolve_2d_projection_sides(
                int(deltas_64[0].shape[0]),
                int(deltas_64[0].shape[1]),
                p_null,
                p_null_out,
                projection_side,
            )
            orth_64, comp_64, stats = _orthogonalize_2d_task_matrices(
                deltas_64,
                weights,
                p_in,
                p_out,
                side_label,
                rank_per_task=rank_per_task,
                variance_threshold=variance_threshold,
                sv_rel_tol=sv_tol,
                polar_sv_rel_tol=polar_tol,
                eig_threshold=eig_t,
            )

        orth_cast = [
            m.to(device=orig.device, dtype=orig.dtype).contiguous()
            for m, orig in zip(orth_64, items, strict=True)
        ]
        comp_cast = comp_64.to(device=ref_t.device, dtype=ref_t.dtype).contiguous()
        out_tasks: Any = dict(zip(task_names, orth_cast, strict=True)) if was_mapping else orth_cast
        report = {
            "num_tasks": len(task_names),
            "task_names": list(task_names),
            "task_weights": list(weights),
            **stats,
        }
        return out_tasks, comp_cast, report

    assert param_keys is not None
    if p_null is not None and not isinstance(p_null, (torch.Tensor, Mapping)):
        raise TypeError("p_null must be a torch.Tensor, Mapping[str, torch.Tensor], or None")
    orth_dicts: list[dict[str, torch.Tensor]] = [{} for _ in task_names]
    composed_dict: dict[str, torch.Tensor] = {}
    per_param_reports: dict[str, Any] = {}
    for p_key in param_keys:
        p_tensors = [item[p_key] for item in items]
        resolved_p = _match_projector_from_arg(p_key, p_tensors[0].shape, p_null)
        sub_orth, sub_comp, sub_rep = orthogonalize_task_vectors_polar_nullspace(
            p_tensors,
            resolved_p,
            task_weights=weights,
            rank_per_task=rank_per_task,
            variance_threshold=variance_threshold,
            sv_rel_tol=sv_tol,
            polar_sv_rel_tol=polar_tol,
            nullspace_eig_threshold=eig_t,
            projection_side=projection_side,
        )
        assert isinstance(sub_orth, list) and isinstance(sub_comp, torch.Tensor)
        for idx, t_orth in enumerate(sub_orth):
            orth_dicts[idx][p_key] = t_orth
        composed_dict[p_key] = sub_comp
        per_param_reports[p_key] = sub_rep

    max_leak = max(float(r["nullspace_leakage_norm"]) for r in per_param_reports.values())
    mean_cos = float(np.mean([float(r["cosine_interference"]) for r in per_param_reports.values()]))
    max_cos = max(float(r["max_pairwise_frobenius_cosine_after"]) for r in per_param_reports.values())
    out_dicts: Any = dict(zip(task_names, orth_dicts, strict=True)) if was_mapping else orth_dicts
    return out_dicts, composed_dict, {
        "num_tasks": len(task_names),
        "task_names": list(task_names),
        "task_weights": list(weights),
        "nullspace_leakage_norm": max_leak,
        "cosine_interference": mean_cos,
        "max_pairwise_frobenius_cosine_after": max_cos,
        "per_parameter": per_param_reports,
    }


def _match_projector_from_arg(
    key: str,
    shape: tuple[int, ...],
    p_null: torch.Tensor | Mapping[str, torch.Tensor] | None,
) -> torch.Tensor | None:
    """Resolve matching projector tensor for ``key`` from a single Tensor or projector Mapping."""
    if p_null is None:
        return None
    if isinstance(p_null, torch.Tensor):
        dp = int(p_null.shape[0])
        if len(shape) == 1 and shape[0] == dp:
            return p_null
        if len(shape) == 2 and (shape[1] == dp or shape[0] == dp):
            return p_null
        return None
    if key in p_null:
        return p_null[key]
    layer_m = _LAYER_KEY_RE.search(key)
    layer_tag = f"blocks.{layer_m.group(1)}" if layer_m is not None else None
    is_act = key.endswith("w3_A") or key.endswith(".w3") or key.endswith(".w3.weight") or "down_proj" in key
    cands: list[str] = []
    if layer_tag is not None:
        cands.append(f"{layer_tag}.act" if is_act else f"{layer_tag}.x")
    cands.extend(("act", "x", "default") if is_act else ("x", "act", "default"))
    for cand in cands:
        if cand in p_null:
            proj = p_null[cand]
            dp = int(proj.shape[0])
            if (len(shape) == 1 and shape[0] == dp) or (
                len(shape) == 2 and (shape[1] == dp or shape[0] == dp)
            ):
                return proj
    return None


# ---------------------------------------------------------------------------
# 5. Open-Weight MoE Operator Extraction, Folding, Nullspace & ANSE Verification
# ---------------------------------------------------------------------------


def _unpack_bucket_deltas_to_param_map(
    model: nn.Module,
    bucket_delta: Mapping[str, torch.Tensor],
    *,
    num_router_fragments: int,
) -> dict[str, torch.Tensor]:
    """Unpack OpenDiLoCo flat bucket deltas into named ``state_dict`` parameter tensors."""
    blocks = getattr(model, "blocks", ())
    n_blocks = len(blocks)
    frag_span = max(1, math.ceil(n_blocks / max(1, num_router_fragments)))
    router_offsets: dict[str, int] = {k: 0 for k in bucket_delta if k.startswith("router_frag_")}
    expert_offsets: dict[str, int] = {k: 0 for k in bucket_delta if k.startswith("expert_")}
    param_map: dict[str, torch.Tensor] = {}

    for l_idx, blk in enumerate(blocks):
        f_idx = min(num_router_fragments - 1, l_idx // frag_span)
        r_key = f"router_frag_{f_idx}"
        mlp = getattr(blk, "mlp", None)
        if isinstance(mlp, (CrowdLoRAMoEMLP, CrowdFineGrainedMoEMLP)):
            if mlp.router.weight.requires_grad and r_key in bucket_delta:
                n = mlp.router.weight.numel()
                off = router_offsets[r_key]
                if off + n > bucket_delta[r_key].numel():
                    raise ValueError(f"Bucket {r_key!r} smaller than model layout")
                param_map[f"blocks.{l_idx}.mlp.router.weight"] = (
                    bucket_delta[r_key][off : off + n].view_as(mlp.router.weight).contiguous()
                )
                router_offsets[r_key] = off + n

        if isinstance(mlp, CrowdLoRAMoEMLP):
            for e_key, exp_mod in sorted(mlp.experts.items(), key=lambda kv: int(kv[0])):
                b_name = f"expert_{int(e_key)}"
                if b_name not in bucket_delta:
                    continue
                for p_name, p in exp_mod.named_parameters():
                    if not p.requires_grad:
                        continue
                    n = p.numel()
                    off = expert_offsets[b_name]
                    if off + n > bucket_delta[b_name].numel():
                        raise ValueError(f"Bucket {b_name!r} smaller than model layout")
                    param_map[f"blocks.{l_idx}.mlp.experts.{int(e_key)}.{p_name}"] = (
                        bucket_delta[b_name][off : off + n].view_as(p).contiguous()
                    )
                    expert_offsets[b_name] = off + n

    last_r_key = f"router_frag_{num_router_fragments - 1}"
    if last_r_key in bucket_delta:
        for ln_name, p in (("ln.weight", model.ln.weight), ("ln.bias", model.ln.bias)):
            if p.requires_grad:
                n = p.numel()
                off = router_offsets[last_r_key]
                if off + n > bucket_delta[last_r_key].numel():
                    raise ValueError(f"Bucket {last_r_key!r} smaller than model layout")
                param_map[ln_name] = bucket_delta[last_r_key][off : off + n].view_as(p).contiguous()
                router_offsets[last_r_key] = off + n

    for r_key, off in router_offsets.items():
        if off != bucket_delta[r_key].numel():
            raise ValueError(f"Bucket {r_key!r} length {bucket_delta[r_key].numel()} != model layout {off}")
    for e_key, off in expert_offsets.items():
        if off != bucket_delta[e_key].numel():
            raise ValueError(f"Bucket {e_key!r} length {bucket_delta[e_key].numel()} != model layout {off}")
    return param_map


def _normalize_task_delta_maps(
    model: nn.Module,
    task_deltas: Sequence[Mapping[str, torch.Tensor]] | Mapping[str, Mapping[str, torch.Tensor]],
    *,
    delta_format: str,
    num_router_fragments: int | None,
) -> tuple[list[str], list[dict[str, torch.Tensor]]]:
    """Normalize per-task delta dicts or OpenDiLoCo buckets into validated state_dict delta maps."""
    if delta_format not in VALID_DELTA_FORMATS:
        raise ValueError(
            f"delta_format must be one of {sorted(VALID_DELTA_FORMATS)}, got {delta_format!r}"
        )
    if isinstance(task_deltas, Mapping):
        if not task_deltas:
            raise ValueError("task_deltas must not be empty")
        task_names = [_validate_nonempty_str(k, "task_name") for k in task_deltas.keys()]
        raw_maps = list(task_deltas.values())
    elif isinstance(task_deltas, Sequence) and not isinstance(task_deltas, (str, bytes, bytearray)):
        if not task_deltas:
            raise ValueError("task_deltas must not be empty")
        task_names = [f"task_{idx}" for idx in range(len(task_deltas))]
        raw_maps = list(task_deltas)
    else:
        raise TypeError("task_deltas must be a non-empty Sequence or Mapping of delta dicts")

    for idx, dm in enumerate(raw_maps):
        if not isinstance(dm, Mapping) or not dm:
            raise ValueError(f"task_deltas[{idx}] must be a non-empty Mapping[str, torch.Tensor]")

    model_sd = model.state_dict()
    all_keys = sorted({str(k) for dm in raw_maps for k in dm.keys()})
    if any(not k.strip() for k in all_keys):
        raise ValueError("task_deltas parameter keys must be non-empty strings")

    resolved_format = delta_format
    if resolved_format == "auto":
        bucket_like = all(k.startswith("router_frag_") or k.startswith("expert_") for k in all_keys) and any(
            k not in model_sd for k in all_keys
        )
        resolved_format = "buckets" if bucket_like else "state_dict"

    if resolved_format == "buckets":
        nfrag = (
            _validate_positive_int(num_router_fragments, "num_router_fragments")
            if num_router_fragments is not None
            else max(1, len([k for k in all_keys if k.startswith("router_frag_")]))
        )
        return task_names, [
            _unpack_bucket_deltas_to_param_map(model, dm, num_router_fragments=nfrag) for dm in raw_maps
        ]

    out_maps: list[dict[str, torch.Tensor]] = []
    for idx, dm in enumerate(raw_maps):
        cur: dict[str, torch.Tensor] = {}
        for k, v in dm.items():
            k_str = str(k)
            t64 = _validate_finite_tensor(v, f"task_deltas[{idx}][{k_str!r}]", allowed_ndims=(1, 2))
            if _EXPERT_FACTOR_RE.match(k_str) is None:
                if k_str not in model_sd:
                    raise ValueError(f"Delta parameter key {k_str!r} not found in model.state_dict()")
                if tuple(t64.shape) != tuple(model_sd[k_str].shape):
                    raise ValueError(
                        f"Delta shape {tuple(t64.shape)} for {k_str!r} != model shape {tuple(model_sd[k_str].shape)}"
                    )
            cur[k_str] = v
        out_maps.append(cur)
    return task_names, out_maps


def _build_model_operators_for_tasks(
    model: nn.Module,
    param_maps: Sequence[Mapping[str, torch.Tensor]],
    *,
    protect_layernorm: bool,
) -> list[dict[str, Any]]:
    """Convert per-task parameter/LoRA deltas into canonical tangent-space operator descriptors."""
    model_sd = model.state_dict()
    lora_groups: dict[tuple[int, str, str], list[dict[str, torch.Tensor]]] = {}
    plain_keys: set[str] = set()

    for t_idx, pmap in enumerate(param_maps):
        for key, val in pmap.items():
            if protect_layernorm and _is_layernorm_key(key):
                continue
            m = _EXPERT_FACTOR_RE.match(key)
            if m is not None:
                l_idx, e_key, pfx = int(m.group(2)), m.group(3), m.group(4)
                ab_tag = m.group(5) or "FULL"
                grp = lora_groups.setdefault(
                    (l_idx, e_key, pfx), [{} for _ in range(len(param_maps))]
                )
                grp[t_idx][ab_tag] = val
            else:
                plain_keys.add(key)

    ops: list[dict[str, Any]] = []
    blocks = getattr(model, "blocks", None)
    for (l_idx, e_key, pfx), per_task_grp in sorted(lora_groups.items()):
        if blocks is None or l_idx >= len(blocks):
            raise ValueError(f"Model missing block index {l_idx} for LoRA delta")
        experts = getattr(getattr(blocks[l_idx], "mlp", None), "experts", None)
        if experts is None or e_key not in experts:
            raise ValueError(f"Model layer {l_idx} missing expert {e_key!r}")
        exp_mod: CrowdLoRAExpertAdapter = experts[e_key]  # type: ignore[assignment]
        b0 = getattr(exp_mod, f"{pfx}_B").detach().to(device="cpu", dtype=torch.float64)
        a0 = getattr(exp_mod, f"{pfx}_A").detach().to(device="cpu", dtype=torch.float64)
        scaling = float(exp_mod.scaling)
        task_mats: list[torch.Tensor] = []
        for g in per_task_grp:
            if "FULL" in g:
                full_t = g["FULL"].detach().to(device="cpu", dtype=torch.float64)
                if tuple(full_t.shape) != (int(b0.shape[0]), int(a0.shape[1])):
                    raise ValueError(
                        f"Effective LoRA delta shape {tuple(full_t.shape)} != ({b0.shape[0]}, {a0.shape[1]})"
                    )
                task_mats.append(full_t)
            elif g:
                db = g["B"].detach().to(device="cpu", dtype=torch.float64) if "B" in g else 0.0
                da = g["A"].detach().to(device="cpu", dtype=torch.float64) if "A" in g else 0.0
                if isinstance(db, torch.Tensor) and tuple(db.shape) != tuple(b0.shape):
                    raise ValueError(f"LoRA B delta shape {tuple(db.shape)} != {tuple(b0.shape)}")
                if isinstance(da, torch.Tensor) and tuple(da.shape) != tuple(a0.shape):
                    raise ValueError(f"LoRA A delta shape {tuple(da.shape)} != {tuple(a0.shape)}")
                eff = scaling * ((b0 + db) @ (a0 + da) - (b0 @ a0))
                task_mats.append(eff)
            else:
                task_mats.append(torch.zeros((int(b0.shape[0]), int(a0.shape[1])), dtype=torch.float64))
        if all(float(m.abs().max().item()) <= 1e-12 for m in task_mats):
            continue
        ops.append(
            {
                "kind": "lora_effective",
                "key": f"blocks.{l_idx}.mlp.experts.{e_key}.{pfx}",
                "layer_idx": l_idx,
                "expert_key": e_key,
                "factor_prefix": pfx,
                "role": f"lora_{pfx}",
                "task_tensors": task_mats,
            }
        )

    for key in sorted(plain_keys):
        ref_t = model_sd[key]
        t_list = [
            pmap[key].detach().to(device="cpu", dtype=torch.float64)
            if key in pmap
            else torch.zeros(ref_t.shape, dtype=torch.float64)
            for pmap in param_maps
        ]
        if all(float(m.abs().max().item()) <= 1e-12 for m in t_list):
            continue
        layer_m = _LAYER_KEY_RE.search(key)
        ops.append(
            {
                "kind": "tensor",
                "key": key,
                "layer_idx": int(layer_m.group(1)) if layer_m is not None else None,
                "role": "router" if "router" in key.lower() else "generic",
                "task_tensors": t_list,
            }
        )
    return ops


@torch.no_grad()
def _collect_prose_projectors_from_model(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    cfg: TangentTaskVectorConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Collect per-layer AlphaEdit ``P_0`` projectors from ``E0_*`` and tangent activations."""
    e0_chunks = [c for c in probe_chunks if str(c["expert_id"]).startswith(cfg.general_prefix)]
    if not e0_chunks:
        raise ValueError(f"probe_chunks must contain at least one {cfg.general_prefix!r} chunk")
    ids_all = torch.stack([torch.as_tensor(c["ids"], dtype=torch.long) for c in probe_chunks], dim=0)
    is_e0 = torch.tensor(
        [str(c["expert_id"]).startswith(cfg.general_prefix) for c in probe_chunks], dtype=torch.bool
    )
    start_pos = max(0, min(int(ids_all.shape[1]) - 2, cfg.score_from - 1))
    cos = model.cos_cached.to(ids_all.device)
    sin = model.sin_cached.to(ids_all.device)
    bsz = cfg.batch_size

    x_seq = torch.cat(
        [compute_embeddings_batch(model, ids_all[i : i + bsz]) for i in range(0, ids_all.shape[0], bsz)],
        dim=0,
    )
    p_nulls: dict[str, torch.Tensor] = {}
    act_inputs: dict[str, torch.Tensor] = {}
    for l_idx, blk in enumerate(model.blocks):
        xm_l, mi_l = [], []
        for i in range(0, x_seq.shape[0], bsz):
            xm, mi = forward_block_pre_mlp(blk, x_seq[i : i + bsz], cos, sin)
            xm_l.append(xm)
            mi_l.append(mi)
        x_mid = torch.cat(xm_l, dim=0)
        mlp_in = torch.cat(mi_l, dim=0)
        h_dim = int(mlp_in.shape[-1])
        scored_all_x = mlp_in[:, start_pos:-1, :].reshape(-1, h_dim)
        scored_e0_x = mlp_in[is_e0][:, start_pos:-1, :].reshape(-1, h_dim)

        p_x, _ = compute_prose_nullspace_projector_scaled(
            scored_e0_x,
            sv_rel_tol=cfg.nullspace_sv_rel_tol,
            variance_threshold=cfg.nullspace_variance_threshold,
            max_active_modes=cfg.nullspace_max_active_modes,
        )
        p_nulls[f"blocks.{l_idx}.x"] = p_x
        act_inputs[f"blocks.{l_idx}.x"] = scored_all_x

        if isinstance(blk.mlp, CrowdLoRAMoEMLP) and blk.mlp.experts:
            active_exp = sorted(int(k) for k in blk.mlp.experts.keys())
            act_e0 = compute_swiglu_intermediate_states(blk.mlp, scored_e0_x, active_exp)
            act_all = compute_swiglu_intermediate_states(blk.mlp, scored_all_x, active_exp)
            p_act, _ = compute_prose_nullspace_projector_scaled(
                act_e0,
                sv_rel_tol=cfg.nullspace_sv_rel_tol,
                variance_threshold=cfg.nullspace_variance_threshold,
                max_active_modes=cfg.nullspace_max_active_modes,
            )
            p_nulls[f"blocks.{l_idx}.act"] = p_act
            act_inputs[f"blocks.{l_idx}.act"] = act_all

        x_seq = torch.cat(
            [
                forward_block_post_mlp(blk, x_mid[i : i + bsz], mlp_in[i : i + bsz])
                for i in range(0, x_mid.shape[0], bsz)
            ],
            dim=0,
        )
    return p_nulls, act_inputs


def _resolve_op_projector(
    op: Mapping[str, Any],
    p_nulls: Mapping[str, torch.Tensor],
    cfg: TangentTaskVectorConfig,
) -> tuple[torch.Tensor | None, str]:
    """Select the AlphaEdit nullspace projector and side matching an operator descriptor."""
    key = str(op["key"])
    role = str(op["role"])
    l_idx = op.get("layer_idx")
    sample_t: torch.Tensor = op["task_tensors"][0]
    if sample_t.ndim == 1:
        p = _match_projector_from_arg(key, tuple(sample_t.shape), p_nulls)
        return p, "input" if p is not None else "none"

    d_out, d_in = int(sample_t.shape[0]), int(sample_t.shape[1])
    if role == "router" and not cfg.project_router_into_nullspace:
        return None, "none"

    if key in p_nulls:
        p = p_nulls[key]
        if int(p.shape[0]) == d_in:
            return p, "input"
        if int(p.shape[0]) == d_out:
            return p, "output"

    is_act = role == "lora_w3" or key.endswith(".w3.weight") or "down_proj" in key
    cands: list[str] = []
    if l_idx is not None:
        cands.append(f"blocks.{l_idx}.act" if is_act else f"blocks.{l_idx}.x")
    cands.extend(("act", "x", "default") if is_act else ("x", "act", "default"))
    for cand in cands:
        if cand in p_nulls:
            p = p_nulls[cand]
            if int(p.shape[0]) == d_in:
                return p, "input"
            if int(p.shape[0]) == d_out:
                return p, "output"
    return None, "none"


def _fold_composed_operator_inplace(
    model: nn.Module,
    op: Mapping[str, Any],
    composed_delta: torch.Tensor,
) -> None:
    """Fold composed task-space update into model dense weights or rank-``r`` LoRA factors."""
    if op["kind"] == "lora_effective":
        blk = model.blocks[int(op["layer_idx"])]
        exp: CrowdLoRAExpertAdapter = blk.mlp.experts[str(op["expert_key"])]  # type: ignore[assignment]
        pfx = str(op["factor_prefix"])
        b_param: torch.Tensor = getattr(exp, f"{pfx}_B")
        a_param: torch.Tensor = getattr(exp, f"{pfx}_A")
        b_cur = b_param.detach().to(device="cpu", dtype=torch.float64)
        a_cur = a_param.detach().to(device="cpu", dtype=torch.float64)
        delta_ab = composed_delta.detach().to(device="cpu", dtype=torch.float64) / float(exp.scaling)

        rank = int(exp.rank)
        if float(b_cur.abs().max().item()) <= 1e-15:
            u_m, s_m, vh_m = torch.linalg.svd(delta_ab, full_matrices=False)
        else:
            # Combine current LoRA product and composed tangent update via thin QR-SVD at exact rank
            u_d, s_d, vh_d = torch.linalg.svd(delta_ab, full_matrices=False)
            k_d = int((s_d > 1e-15).sum().item())
            if k_d > 0:
                s_d_sqrt = s_d[:k_d].sqrt()
                b_cat = torch.cat([b_cur, u_d[:, :k_d] * s_d_sqrt.unsqueeze(0)], dim=1)
                a_cat = torch.cat([a_cur, s_d_sqrt.unsqueeze(1) * vh_d[:k_d, :]], dim=0)
                u_m, s_m, vh_raw = decompose_lora_qr_svd(b_cat, a_cat)
                vh_m = vh_raw.T
            else:
                return

        k_use = min(rank, int(s_m.numel()))
        b_new = torch.zeros_like(b_cur)
        a_new = torch.zeros_like(a_cur)
        if k_use > 0:
            s_sqrt = s_m[:k_use].clamp_min(0.0).sqrt()
            b_new[:, :k_use] = u_m[:, :k_use] * s_sqrt.unsqueeze(0)
            a_new[:k_use, :] = s_sqrt.unsqueeze(1) * vh_m[:k_use, :]
        b_param.copy_(b_new.to(device=b_param.device, dtype=b_param.dtype))
        a_param.copy_(a_new.to(device=a_param.device, dtype=a_param.dtype))
        return

    param = model.state_dict()[str(op["key"])]
    updated = (
        param.detach().to(device="cpu", dtype=torch.float64)
        + composed_delta.detach().to(device="cpu", dtype=torch.float64)
    ).to(device=param.device, dtype=param.dtype)
    param.copy_(updated)


def _validate_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    general_prefix: str,
    vocab_size: int | None,
    score_from: int,
    name: str,
) -> tuple[torch.Tensor, list[str]]:
    """Validate evaluation/probe token chunks and return ``(ids_2d, domain_ids)``."""
    if isinstance(chunks, (str, bytes, bytearray)) or not isinstance(chunks, Sequence) or not chunks:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")
    rows: list[torch.Tensor] = []
    domains: list[str] = []
    seq_len: int | None = None
    for idx, ch in enumerate(chunks):
        if not isinstance(ch, Mapping) or "expert_id" not in ch or "ids" not in ch:
            raise ValueError(f"{name}[{idx}] must be a mapping containing 'expert_id' and 'ids'")
        eid = _validate_nonempty_str(ch["expert_id"], f"{name}[{idx}]['expert_id']")
        ids_t = torch.as_tensor(ch["ids"], dtype=torch.long)
        if ids_t.ndim != 1 or ids_t.numel() < 2:
            raise ValueError(f"{name}[{idx}]['ids'] must be 1D with length >= 2")
        if bool((ids_t < 0).any().item()):
            raise ValueError(f"{name}[{idx}]['ids'] must contain non-negative token IDs")
        if vocab_size is not None and bool((ids_t >= vocab_size).any().item()):
            raise ValueError(f"{name}[{idx}]['ids'] token IDs must lie in [0, {vocab_size})")
        if seq_len is None:
            seq_len = int(ids_t.numel())
        elif int(ids_t.numel()) != seq_len:
            raise ValueError(f"All {name} entries must share sequence length {seq_len}")
        rows.append(ids_t)
        domains.append(eid)

    assert seq_len is not None
    if max(0, score_from - 1) >= seq_len - 1:
        raise ValueError(f"{name} sequence length {seq_len} too short for score_from={score_from}")
    if not any(d.startswith(general_prefix) for d in domains):
        raise ValueError(f"{name} must include at least one general chunk starting with {general_prefix!r}")
    return torch.stack(rows, dim=0), domains


@torch.no_grad()
def _evaluate_model_domains(
    model: nn.Module,
    *,
    ids_eval: torch.Tensor | None,
    eval_domains: Sequence[str] | None,
    eval_fn: Callable[[nn.Module], Mapping[str, Any]] | None,
    cfg: TangentTaskVectorConfig,
    expected_domains: set[str] | None = None,
) -> tuple[dict[str, float] | None, bool]:
    """Evaluate per-domain finite NLLs via ``eval_fn`` or CrowdV1 forward pass."""
    if eval_fn is not None:
        raw = eval_fn(model)
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("eval_fn must return a non-empty mapping")
        src = raw.get("domain_nlls", raw.get("domain_nll", raw))
        if not isinstance(src, Mapping) or not src:
            raise ValueError("eval_fn result must contain non-empty domain NLL entries")
        cleaned: dict[str, float] = {}
        for k, v in sorted(src.items()):
            ks = str(k)
            if ks in {"nll_mean", "all_finite", "non_finite_outputs"}:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None, False
            fv = float(v)
            if not math.isfinite(fv) or fv <= 0.0:
                return None, False
            cleaned[ks] = fv
        if not cleaned or int(raw.get("non_finite_outputs", 0)) != 0:
            return None, False
        if expected_domains is not None and set(cleaned.keys()) != expected_domains:
            return None, False
        if not any(k.startswith(cfg.general_prefix) for k in cleaned):
            return None, False
        return cleaned, True

    assert ids_eval is not None and eval_domains is not None
    h_pre = forward_model_pre_ln(model, ids_eval, batch_size=cfg.batch_size)
    if not bool(torch.isfinite(h_pre).all().item()):
        return None, False
    ln_bias = getattr(model.ln, "bias", None)
    ln_b = ln_bias if isinstance(ln_bias, torch.Tensor) else torch.zeros_like(model.ln.weight)
    nlls = per_chunk_nll_from_pre_ln(
        h_pre,
        model.ln.weight,
        ln_b,
        float(getattr(model.ln, "eps", 1e-5)),
        model.head.weight,
        ids_eval,
        score_from=cfg.score_from,
    )
    if not bool(torch.isfinite(nlls).all().item()):
        return None, False
    grouped: dict[str, list[float]] = {}
    for dom, val in zip(eval_domains, nlls.detach().cpu().tolist(), strict=True):
        grouped.setdefault(str(dom), []).append(float(val))
    return {dom: float(sum(xs) / len(xs)) for dom, xs in sorted(grouped.items())}, True


def _is_unlearned_domain(dom: str, prefixes: tuple[str, ...]) -> bool:
    """Return True if ``dom`` matches any configured unlearning domain prefix."""
    return any(dom.startswith(pref) for pref in prefixes)


def _evaluate_anse_guard(
    ref_dom_nll: Mapping[str, float],
    cand_dom_nll: Mapping[str, float] | None,
    *,
    cand_finite: bool,
    cfg: TangentTaskVectorConfig,
) -> dict[str, Any]:
    """Compute per-domain drift, unlearning deltas, guard compliance, and ANSE Energy."""
    retained_keys = [d for d in sorted(ref_dom_nll.keys()) if not _is_unlearned_domain(d, cfg.unlearn_domain_prefixes)]
    if not retained_keys:
        raise ValueError("At least one retained (non-unlearned) domain is required")
    ref_mean = float(sum(ref_dom_nll[d] for d in retained_keys) / len(retained_keys))

    if not cand_finite or cand_dom_nll is None:
        return {
            "finite": False,
            "nll_mean": None,
            "ref_nll_mean": ref_mean,
            "general_drift_pct": None,
            "worst_retained_drift_pct": None,
            "guard_ok": False,
            "anse_energy": float(cfg.maximum_pain_energy),
            "per_domain": {},
            "unlearned_domains": {},
        }

    per_domain: dict[str, dict[str, Any]] = {}
    unlearned: dict[str, dict[str, Any]] = {}
    worst_ret_ratio = -math.inf
    gen_ratios: list[float] = []
    all_ret_ok = True
    unlearn_ok = True

    for dom in sorted(ref_dom_nll.keys()):
        b_val = float(ref_dom_nll[dom])
        c_val = float(cand_dom_nll[dom])
        ratio = c_val / max(b_val, 1e-12)
        drift = ratio - 1.0
        if _is_unlearned_domain(dom, cfg.unlearn_domain_prefixes):
            dom_unlearned_ok = c_val >= b_val - 1e-9
            unlearn_ok = unlearn_ok and dom_unlearned_ok
            unlearned[dom] = {
                "ref_nll": b_val,
                "cand_nll": c_val,
                "delta_nll": c_val - b_val,
                "ratio": ratio,
                "unlearned_ok": dom_unlearned_ok,
            }
            continue
        dom_ok = ratio <= cfg.domain_guard_max_ratio + 1e-12
        all_ret_ok = all_ret_ok and dom_ok
        worst_ret_ratio = max(worst_ret_ratio, ratio)
        if dom.startswith(cfg.general_prefix):
            gen_ratios.append(ratio)
        per_domain[dom] = {
            "ref_nll": b_val,
            "cand_nll": c_val,
            "mean_diff": c_val - b_val,
            "ratio": ratio,
            "drift_pct": 100.0 * drift,
            "guard_ok": dom_ok,
        }

    cand_mean = float(sum(float(cand_dom_nll[d]) for d in retained_keys) / len(retained_keys))
    general_ratio = max(gen_ratios) if gen_ratios else worst_ret_ratio
    general_ok = general_ratio <= cfg.domain_guard_max_ratio + 1e-12
    guard_ok = bool(general_ok and unlearn_ok and (all_ret_ok if cfg.enforce_all_domain_guard else True))
    penalty = (
        float(cfg.maximum_pain_energy)
        if not guard_ok
        else (float(cfg.soft_margin_penalty) if general_ratio > cfg.anse_safe_general_ratio else 0.0)
    )
    anse_energy = (
        float(cfg.maximum_pain_energy) if not guard_ok else (cand_mean + penalty)
    )
    return {
        "finite": True,
        "nll_mean": cand_mean,
        "ref_nll_mean": ref_mean,
        "general_drift_pct": 100.0 * (general_ratio - 1.0),
        "worst_retained_drift_pct": 100.0 * (worst_ret_ratio - 1.0),
        "guard_ok": guard_ok,
        "anse_energy": float(anse_energy),
        "per_domain": per_domain,
        "unlearned_domains": unlearned,
    }


def _resolve_composition_config(
    config: TangentTaskVectorConfig | None,
    *,
    rank_per_task: int | None,
    max_domain_drift: float | None,
    enforce_all_domain_guard: bool | None,
    unlearn_domain_prefixes: Sequence[str] | None,
    score_from: int | None,
    inplace: bool | None,
) -> TangentTaskVectorConfig:
    if config is not None and not isinstance(config, TangentTaskVectorConfig):
        raise TypeError(f"config must be TangentTaskVectorConfig or None, got {type(config).__name__}")
    base = config or TangentTaskVectorConfig()
    return TangentTaskVectorConfig(
        rank_per_task=base.rank_per_task if rank_per_task is None else rank_per_task,
        variance_threshold=base.variance_threshold,
        sv_rel_tol=base.sv_rel_tol,
        polar_sv_rel_tol=base.polar_sv_rel_tol,
        nullspace_eig_threshold=base.nullspace_eig_threshold,
        nullspace_sv_rel_tol=base.nullspace_sv_rel_tol,
        nullspace_variance_threshold=base.nullspace_variance_threshold,
        nullspace_max_active_modes=base.nullspace_max_active_modes,
        projection_side=base.projection_side,
        protect_layernorm=base.protect_layernorm,
        orthogonalize_router=base.orthogonalize_router,
        project_router_into_nullspace=base.project_router_into_nullspace,
        domain_guard_max_ratio=(
            base.domain_guard_max_ratio
            if max_domain_drift is None
            else (1.0 + _validate_nonneg_finite(max_domain_drift, "max_domain_drift"))
        ),
        anse_safe_general_ratio=base.anse_safe_general_ratio,
        maximum_pain_energy=base.maximum_pain_energy,
        soft_margin_penalty=base.soft_margin_penalty,
        enforce_all_domain_guard=(
            base.enforce_all_domain_guard
            if enforce_all_domain_guard is None
            else enforce_all_domain_guard
        ),
        require_strict_energy_drop=base.require_strict_energy_drop,
        min_energy_improvement=base.min_energy_improvement,
        general_prefix=base.general_prefix,
        unlearn_domain_prefixes=(
            base.unlearn_domain_prefixes
            if unlearn_domain_prefixes is None
            else tuple(unlearn_domain_prefixes)
        ),
        score_from=base.score_from if score_from is None else score_from,
        batch_size=base.batch_size,
        inplace=base.inplace if inplace is None else inplace,
    )


@torch.no_grad()
def execute_verified_tangent_task_composition(
    model: nn.Module,
    task_deltas: Sequence[Mapping[str, torch.Tensor]] | Mapping[str, Mapping[str, torch.Tensor]],
    *,
    task_weights: Sequence[float] | None = None,
    p_null: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    prose_keys: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor] | None = None,
    domain_inputs: Mapping[str, torch.Tensor] | None = None,
    probe_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_fn: Callable[[nn.Module], Mapping[str, Any]] | None = None,
    delta_format: str = "auto",
    num_router_fragments: int | None = None,
    config: TangentTaskVectorConfig | None = None,
    rank_per_task: int | None = None,
    max_domain_drift: float | None = None,
    enforce_all_domain_guard: bool | None = None,
    unlearn_domain_prefixes: Sequence[str] | None = None,
    score_from: int | None = None,
    inplace: bool | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Compose ``K`` task vectors via polar-nullspace orthogonalization under ANSE gating & rollback."""
    if not isinstance(model, nn.Module) or not model.state_dict():
        raise ValueError("model must be a non-empty nn.Module")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError("eval_fn must be callable or None")

    cfg = _resolve_composition_config(
        config,
        rank_per_task=rank_per_task,
        max_domain_drift=max_domain_drift,
        enforce_all_domain_guard=enforce_all_domain_guard,
        unlearn_domain_prefixes=unlearn_domain_prefixes,
        score_from=score_from,
        inplace=inplace,
    )
    work_model = model if cfg.inplace else copy.deepcopy(model)
    task_names, param_maps = _normalize_task_delta_maps(
        work_model,
        task_deltas,
        delta_format=delta_format,
        num_router_fragments=num_router_fragments,
    )
    weights = _normalize_task_weights(task_weights, len(task_names))

    if eval_fn is None:
        src_chunks = eval_chunks if eval_chunks is not None else probe_chunks
        if src_chunks is None or not hasattr(work_model, "blocks") or not hasattr(work_model, "head"):
            raise ValueError(
                "eval_fn or (eval_chunks/probe_chunks on CrowdV1-compatible model) is required"
            )
        ids_eval, eval_domains = _validate_chunks(
            src_chunks,
            general_prefix=cfg.general_prefix,
            vocab_size=int(work_model.head.weight.shape[0]),
            score_from=cfg.score_from,
            name="eval_chunks" if eval_chunks is not None else "probe_chunks",
        )
    else:
        ids_eval, eval_domains = None, None

    auto_inputs: dict[str, torch.Tensor] = {}
    if p_null is not None:
        if isinstance(p_null, torch.Tensor):
            p_nulls = {"default": _validate_square_projector(p_null, "p_null")}
        elif isinstance(p_null, Mapping):
            if not p_null:
                raise ValueError("p_null mapping must not be empty")
            p_nulls = {str(k): _validate_square_projector(v, f"p_null[{k!r}]") for k, v in p_null.items()}
        else:
            raise TypeError("p_null must be a torch.Tensor, Mapping[str, torch.Tensor], or None")
    elif prose_keys is not None:
        if isinstance(prose_keys, Mapping):
            p_nulls = {
                str(k): compute_prose_nullspace_projector_scaled(
                    v,
                    sv_rel_tol=cfg.nullspace_sv_rel_tol,
                    variance_threshold=cfg.nullspace_variance_threshold,
                    max_active_modes=cfg.nullspace_max_active_modes,
                )[0]
                for k, v in prose_keys.items()
            }
        else:
            p_def, _ = compute_prose_nullspace_projector_scaled(
                prose_keys,
                sv_rel_tol=cfg.nullspace_sv_rel_tol,
                variance_threshold=cfg.nullspace_variance_threshold,
                max_active_modes=cfg.nullspace_max_active_modes,
            )
            p_nulls = {"default": p_def}
    elif probe_chunks is not None and hasattr(work_model, "blocks") and hasattr(work_model, "cos_cached"):
        _validate_chunks(
            probe_chunks,
            general_prefix=cfg.general_prefix,
            vocab_size=int(work_model.head.weight.shape[0]) if hasattr(work_model, "head") else None,
            score_from=cfg.score_from,
            name="probe_chunks",
        )
        p_nulls, auto_inputs = _collect_prose_projectors_from_model(work_model, probe_chunks, cfg)
    else:
        p_nulls = {}

    if domain_inputs is not None:
        if not isinstance(domain_inputs, Mapping) or not domain_inputs:
            raise TypeError("domain_inputs must be a non-empty Mapping[str, torch.Tensor] or None")
        inputs_map: dict[str, torch.Tensor] = dict(domain_inputs)
    else:
        inputs_map = auto_inputs

    ops = _build_model_operators_for_tasks(
        work_model, param_maps, protect_layernorm=cfg.protect_layernorm
    )

    was_training = work_model.training
    work_model.eval()
    try:
        parent_sha = compute_state_sha256(work_model)
        parent_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in work_model.state_dict().items()}
        parent_dom_nll, parent_finite = _evaluate_model_domains(
            work_model, ids_eval=ids_eval, eval_domains=eval_domains, eval_fn=eval_fn, cfg=cfg
        )
        if not parent_finite or parent_dom_nll is None:
            raise ValueError("Parent model evaluation returned invalid or non-finite NLLs")
        parent_anse = _evaluate_anse_guard(
            parent_dom_nll, parent_dom_nll, cand_finite=True, cfg=cfg
        )

        # Orthogonalize every active operator inside resolved AlphaEdit P_0
        op_reports: list[dict[str, Any]] = []
        cos_before_vals: list[float] = []
        cos_after_vals: list[float] = []
        leak_vals: list[float] = []
        act_before_vals: list[float] = []
        act_after_vals: list[float] = []

        for op in ops:
            task_mats: list[torch.Tensor] = op["task_tensors"]
            proj_mat, side_label = _resolve_op_projector(op, p_nulls, cfg)
            if op["role"] == "router" and not cfg.orthogonalize_router:
                composed = sum(
                    (w * (t @ proj_mat if (proj_mat is not None and side_label == "input") else t))
                    for w, t in zip(weights, task_mats, strict=True)
                )
                rep = {
                    "key": op["key"],
                    "role": op["role"],
                    "projection_side": side_label,
                    "cosine_interference": 0.0,
                    "nullspace_leakage_norm": 0.0,
                }
            else:
                orth_list, composed, rep = orthogonalize_task_vectors_polar_nullspace(
                    task_mats,
                    proj_mat,
                    task_weights=weights,
                    rank_per_task=cfg.rank_per_task,
                    variance_threshold=cfg.variance_threshold,
                    sv_rel_tol=cfg.sv_rel_tol,
                    polar_sv_rel_tol=cfg.polar_sv_rel_tol,
                    nullspace_eig_threshold=cfg.nullspace_eig_threshold,
                    projection_side=side_label if side_label != "none" else cfg.projection_side,
                )
                assert isinstance(orth_list, list) and isinstance(composed, torch.Tensor)
                rep["key"] = op["key"]
                rep["role"] = op["role"]
                cos_before_vals.append(float(rep["mean_abs_frobenius_cosine_before"]))
                cos_after_vals.append(float(rep["mean_abs_frobenius_cosine_after"]))
                leak_vals.append(float(rep["nullspace_leakage_norm"]))
                if inputs_map and task_mats[0].ndim == 2:
                    x_act = _resolve_param_activation(op["key"], int(task_mats[0].shape[1]), inputs_map)
                    if x_act is not None:
                        b_int = compute_activation_weighted_task_interference(task_mats, x_act)
                        a_int = compute_activation_weighted_task_interference(orth_list, x_act)
                        act_before_vals.append(float(b_int["cosine_interference"]))
                        act_after_vals.append(float(a_int["cosine_interference"]))
                        rep["activation_cosine_before"] = float(b_int["cosine_interference"])
                        rep["activation_cosine_after"] = float(a_int["cosine_interference"])
            op_reports.append(rep)
            _fold_composed_operator_inplace(work_model, op, composed)

        candidate_sha = compute_state_sha256(work_model)
        cand_dom_nll, cand_finite = _evaluate_model_domains(
            work_model,
            ids_eval=ids_eval,
            eval_domains=eval_domains,
            eval_fn=eval_fn,
            cfg=cfg,
            expected_domains=set(parent_dom_nll.keys()),
        )
        cand_anse = _evaluate_anse_guard(
            parent_dom_nll, cand_dom_nll, cand_finite=cand_finite, cfg=cfg
        )
        delta_e = float(cand_anse["anse_energy"]) - float(parent_anse["anse_energy"])
        energy_ok = (
            (delta_e < -cfg.min_energy_improvement)
            if cfg.require_strict_energy_drop
            else (delta_e <= -cfg.min_energy_improvement)
        )

        if candidate_sha == parent_sha:
            accepted, rolled_back = False, False
            status = "NOOP_UNCHANGED" if cand_anse["guard_ok"] else "REJECTED_GUARD_UNCHANGED"
            active_sha = parent_sha
            final_nll = float(parent_anse["nll_mean"])
        elif cand_anse["finite"] and cand_anse["guard_ok"] and energy_ok:
            accepted, rolled_back = True, False
            status = "COMMITTED"
            active_sha = candidate_sha
            final_nll = float(cand_anse["nll_mean"])
        else:
            accepted, rolled_back = False, True
            status = (
                "ROLLED_BACK_NON_FINITE"
                if not cand_anse["finite"]
                else (
                    "ROLLED_BACK_GUARD_VIOLATION"
                    if not cand_anse["guard_ok"]
                    else "ROLLED_BACK_NO_ENERGY_IMPROVEMENT"
                )
            )
            active_sha = _restore_snapshot_bitwise(work_model, parent_snapshot, parent_sha)
            final_nll = float(parent_anse["nll_mean"])
    finally:
        work_model.train(was_training)

    cos_after_mean = (
        float(sum(act_after_vals) / len(act_after_vals))
        if act_after_vals
        else (float(sum(cos_after_vals) / len(cos_after_vals)) if cos_after_vals else 0.0)
    )
    cos_before_mean = (
        float(sum(act_before_vals) / len(act_before_vals))
        if act_before_vals
        else (float(sum(cos_before_vals) / len(cos_before_vals)) if cos_before_vals else 0.0)
    )
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "object": RECEIPT_OBJECT,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "rollback_bitwise_verified": bool((not accepted) and active_sha == parent_sha),
        "status": status,
        "num_tasks": len(task_names),
        "task_names": task_names,
        "task_weights": weights,
        "operators_composed": len(op_reports),
        "cosine_interference_before": cos_before_mean,
        "cosine_interference": cos_after_mean,
        "max_nullspace_leakage": max(leak_vals) if leak_vals else 0.0,
        "nll_mean_before": float(parent_anse["nll_mean"]),
        "nll_mean": final_nll,
        "nll_mean_candidate": cand_anse["nll_mean"],
        "nll_mean_active": final_nll,
        "general_drift_pct": cand_anse["general_drift_pct"],
        "worst_retained_drift_pct": cand_anse["worst_retained_drift_pct"],
        "guard_ok": bool(cand_anse["guard_ok"]),
        "anse_energy_parent": float(parent_anse["anse_energy"]),
        "anse_energy": float(cand_anse["anse_energy"]),
        "delta_e": float(delta_e),
        "per_domain": cand_anse["per_domain"],
        "unlearned_domains": cand_anse["unlearned_domains"],
        "operator_reports": op_reports,
        "parent_state_sha256": parent_sha,
        "candidate_state_sha256": candidate_sha,
        "final_state_sha256": active_sha,
        "config": asdict(cfg),
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return work_model, receipt


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_ANSE_SOFT_MARGIN_PENALTY",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_NULLSPACE_EIG_THRESHOLD",
    "DEFAULT_NULLSPACE_SV_REL_TOL",
    "DEFAULT_NULLSPACE_VARIANCE_THRESHOLD",
    "DEFAULT_POLAR_SV_REL_TOL",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_SV_REL_TOL",
    "GENERAL_EXPERT_PREFIX",
    "RECEIPT_OBJECT",
    "SCHEMA_VERSION",
    "VALID_DELTA_FORMATS",
    "VALID_PROJECTION_SIDES",
    "TangentTaskVectorConfig",
    "canonical_json_bytes",
    "compute_activation_weighted_task_interference",
    "compute_payload_sha256",
    "compute_prose_nullspace_projector_scaled",
    "compute_state_sha256",
    "execute_verified_tangent_task_composition",
    "orthogonalize_task_vectors_polar_nullspace",
    "verify_tangent_composition_receipt_provenance",
]

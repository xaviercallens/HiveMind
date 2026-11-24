# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Orthogonal Multi-Gate Specialist Router Preserver & End-to-End WSP S4 Surgeon.

Codifies the T4 Stage S4 + virgin-vault Weight-Space Programming architecture
(``WSP_S4_multigate_only``) into a training-free, CPU-exact ``float64`` router &
adapter surgeon for open-weight Sparse MoE models (``CrowdV1WithEngram`` +
``CrowdLoRAMoEMLP`` in ``sigma_topk`` mode):

1. ``partition_sink_vs_preserved_specialists``:
   Separates pure prose-harming and unscored rogue ghost-sink experts (zeroed via
   exact FoMoE ``w1_B = w2_B = w3_B = 0``) from dual-active specialist experts that
   contribute substantial specialist NLL value (``>= spec_preserve_floor_nats``) while
   leaking on general prose.
2. ``fit_oof_orthogonal_multigate_row_update``:
   Computes an ``n_folds`` out-of-fold SNR-gated rank-1 router row shift strictly
   orthogonal to the joint span of all specialist domain centroids so preserved
   specialists close on general prose while firing unimpeded on specialist inputs.
3. ``execute_s4_multigate_moe_surgery``:
   Combines per-layer causal bypass attribution (with bitwise protection of
   beneficial anchor layers), selective FoMoE ghost sinking, orthogonal multi-gate
   router surgery, optional 21-point fine LayerNorm calibration, and the ANSE
   ``<= +1.0%`` general-domain guard with atomic bitwise rollback and SHA-256 receipts.
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

from hivemind.swarm.cross_layer_residual_nullspace import (
    identify_beneficial_anchor_and_interfering_layers,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    extract_moe_trainable_buckets,
    load_moe_trainable_buckets,
)
from hivemind.swarm.layer_localized_attribution import (
    compute_embeddings_batch,
    compute_leave_one_expert_out_attribution,
    compute_per_layer_moe_bypass_attribution,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_causal_midrow_nll,
    zero_expert_lora_b_inplace,
)
from hivemind.swarm.wsp_alphaedit_control import select_guard_optimal_lambda_ln

SCHEMA_VERSION: str = "hivemind.orthogonal_multigate_router_surgeon/1"
RECEIPT_OBJECT: str = "hivemind.orthogonal_multigate_router_surgery_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_GENERAL_EXPERT_INDEX: int = 0
DEFAULT_SPEC_PRESERVE_FLOOR_NATS: float = 0.015
DEFAULT_PROSE_HARM_FLOOR_NATS: float = 0.002
DEFAULT_ROUTER_BETA: float = 0.35
DEFAULT_SINK_ATTRACT_SCALE: float = 0.35
DEFAULT_MIN_OOF_SNR: float = 0.25
DEFAULT_N_FOLDS: int = 2
DEFAULT_OMEGA_SPEC: float = 6.0
DEFAULT_RIDGE_SCALE: float = 0.75
DEFAULT_SUBSPACE_SV_REL_TOL: float = 1e-10
DEFAULT_ANCHOR_THRESHOLD_NATS: float = -0.015
DEFAULT_HARM_THRESHOLD_NATS: float = 0.002
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_FINE_LN_E0_MAX_RATIO: float = 1.006
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_ANSE_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_SCORE_FROM: int = 64
DEFAULT_BATCH_SIZE: int = 24
DEFAULT_LOEO_CHUNKS_PER_DOMAIN: int = 8
FINE_LAMBDA_LN_GRID: tuple[float, ...] = tuple(round(0.05 * idx, 2) for idx in range(21))

_EXPERT_KEY_NUM_RE = re.compile(r"^(?:expert_)?(\d+)$")
_EXPERT_DOMAIN_RE = re.compile(r"^E(\d+)(?:_|$)")
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


def _validate_activations_2d(
    tensor: Any,
    name: str,
    *,
    expected_dim: int | None = None,
) -> torch.Tensor:
    """Validate a non-empty finite 2D/3D floating-point tensor and flatten to CPU ``float64``."""
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


def _validate_router_row_1d(router_row: Any, name: str = "router_row") -> torch.Tensor:
    """Validate a non-empty 1D finite floating-point router row tensor and return CPU ``float64``."""
    if not isinstance(router_row, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(router_row).__name__}")
    if router_row.ndim != 1 or int(router_row.shape[0]) == 0:
        raise ValueError(
            f"{name} must be a non-empty 1D tensor, got shape {tuple(router_row.shape)}"
        )
    if not router_row.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {router_row.dtype}")
    out = router_row.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(out).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return out


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically with strict finite-number enforcement."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


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


def compute_module_state_sha256(module: nn.Module) -> str:
    """Compute a deterministic bitwise SHA-256 digest over ``module.state_dict()``."""
    if not isinstance(module, nn.Module):
        raise TypeError(f"module must be an nn.Module, got {type(module).__name__}")
    return compute_state_sha256(module.state_dict())


def verify_s4_multigate_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != RECEIPT_OBJECT:
        return False
    if receipt.get("schema") != SCHEMA_VERSION:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


@dataclass(frozen=True)
class S4MultigateSurgeonConfig:
    """Validated hyperparameters for WSP Stage S4 orthogonal multi-gate MoE surgery."""

    spec_preserve_floor_nats: float = DEFAULT_SPEC_PRESERVE_FLOOR_NATS
    prose_harm_floor_nats: float = DEFAULT_PROSE_HARM_FLOOR_NATS
    router_beta: float = DEFAULT_ROUTER_BETA
    sink_attract_scale: float = DEFAULT_SINK_ATTRACT_SCALE
    min_oof_snr: float = DEFAULT_MIN_OOF_SNR
    n_folds: int = DEFAULT_N_FOLDS
    omega_spec: float = DEFAULT_OMEGA_SPEC
    ridge_scale: float = DEFAULT_RIDGE_SCALE
    sv_rel_tol: float = DEFAULT_SUBSPACE_SV_REL_TOL
    protect_anchor_layers: bool = True
    anchor_threshold_nats: float = DEFAULT_ANCHOR_THRESHOLD_NATS
    harm_threshold_nats: float = DEFAULT_HARM_THRESHOLD_NATS
    gate_all_surviving_specialists: bool = False
    use_ln_calibration: bool = True
    fine_ln_e0_max_ratio: float = DEFAULT_FINE_LN_E0_MAX_RATIO
    domain_guard_max_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    anse_safe_e0_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = DEFAULT_ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY
    require_energy_improvement: bool = True
    min_energy_improvement: float = 0.0
    general_prefix: str = GENERAL_EXPERT_PREFIX
    general_expert_index: int = DEFAULT_GENERAL_EXPERT_INDEX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    loeo_chunks_per_domain: int = DEFAULT_LOEO_CHUNKS_PER_DOMAIN
    inplace: bool = True

    def __post_init__(self) -> None:
        _validate_nonneg_finite(self.spec_preserve_floor_nats, "spec_preserve_floor_nats")
        _validate_nonneg_finite(self.prose_harm_floor_nats, "prose_harm_floor_nats")
        _validate_nonneg_finite(self.router_beta, "router_beta")
        _validate_nonneg_finite(self.sink_attract_scale, "sink_attract_scale")
        _validate_nonneg_finite(self.min_oof_snr, "min_oof_snr")
        n_f = _validate_positive_int(self.n_folds, "n_folds")
        if n_f < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_f}")
        _validate_nonneg_finite(self.omega_spec, "omega_spec")
        _validate_nonneg_finite(self.ridge_scale, "ridge_scale")
        _validate_nonneg_finite(self.sv_rel_tol, "sv_rel_tol")
        _validate_bool(self.protect_anchor_layers, "protect_anchor_layers")
        anchor_t = _validate_finite_float(self.anchor_threshold_nats, "anchor_threshold_nats")
        harm_t = _validate_nonneg_finite(self.harm_threshold_nats, "harm_threshold_nats")
        if anchor_t > 0.0 or anchor_t >= harm_t:
            raise ValueError(
                f"anchor_threshold_nats ({anchor_t}) must be <= 0.0 and < harm_threshold_nats ({harm_t})"
            )
        _validate_bool(self.gate_all_surviving_specialists, "gate_all_surviving_specialists")
        _validate_bool(self.use_ln_calibration, "use_ln_calibration")
        fine_r = _validate_positive_finite(self.fine_ln_e0_max_ratio, "fine_ln_e0_max_ratio")
        if fine_r < 1.0:
            raise ValueError(f"fine_ln_e0_max_ratio must be >= 1.0, got {fine_r}")
        guard_r = _validate_positive_finite(self.domain_guard_max_ratio, "domain_guard_max_ratio")
        if guard_r < 1.0:
            raise ValueError(f"domain_guard_max_ratio must be >= 1.0, got {guard_r}")
        safe_r = _validate_positive_finite(self.anse_safe_e0_ratio, "anse_safe_e0_ratio")
        if safe_r < 1.0:
            raise ValueError(f"anse_safe_e0_ratio must be >= 1.0, got {safe_r}")
        _validate_nonneg_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_bool(self.require_energy_improvement, "require_energy_improvement")
        _validate_nonneg_finite(self.min_energy_improvement, "min_energy_improvement")
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_nonneg_int(self.general_expert_index, "general_expert_index")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_positive_int(self.loeo_chunks_per_domain, "loeo_chunks_per_domain")
        _validate_bool(self.inplace, "inplace")


# ---------------------------------------------------------------------------
# 2. Component (a): Sink vs Preserved Specialist Partitioner
# ---------------------------------------------------------------------------


def _parse_expert_metric_map(
    raw_metrics: Mapping[Any, float] | Sequence[float],
    name: str,
) -> dict[int, float]:
    """Normalize expert metric mapping or sequence into ``{expert_idx: float_value}``."""
    out: dict[int, float] = {}
    if isinstance(raw_metrics, Mapping):
        if not raw_metrics:
            raise ValueError(f"{name} mapping must not be empty")
        for raw_k, raw_v in raw_metrics.items():
            if isinstance(raw_k, bool):
                raise TypeError(f"{name} key must not be bool")
            if isinstance(raw_k, int):
                e_idx = _validate_nonneg_int(raw_k, f"{name} key")
            elif isinstance(raw_k, str):
                match = _EXPERT_KEY_NUM_RE.match(raw_k.strip())
                if match is None:
                    raise ValueError(f"Invalid expert key {raw_k!r} in {name}")
                e_idx = int(match.group(1))
            else:
                raise TypeError(f"Invalid key type {type(raw_k).__name__} in {name}")
            if e_idx in out:
                raise ValueError(f"Duplicate expert index {e_idx} in {name}")
            out[e_idx] = _validate_finite_float(raw_v, f"{name}[{e_idx}]")
        return out

    if isinstance(raw_metrics, (str, bytes, bytearray)) or not isinstance(raw_metrics, Sequence):
        raise TypeError(f"{name} must be a Mapping or Sequence of floats, got {type(raw_metrics).__name__}")
    if len(raw_metrics) == 0:
        raise ValueError(f"{name} sequence must not be empty")
    for idx, raw_v in enumerate(raw_metrics):
        out[idx] = _validate_finite_float(raw_v, f"{name}[{idx}]")
    return out


def partition_sink_vs_preserved_specialists(
    expert_e0_harm_nats: Mapping[Any, float] | Sequence[float],
    expert_spec_loss_when_zeroed: Mapping[Any, float] | Sequence[float],
    *,
    spec_preserve_floor_nats: float = DEFAULT_SPEC_PRESERVE_FLOOR_NATS,
    prose_harm_floor_nats: float = DEFAULT_PROSE_HARM_FLOOR_NATS,
    rogue_expert_indices: Sequence[int] | set[int] | None = None,
    general_expert_index: int = DEFAULT_GENERAL_EXPERT_INDEX,
) -> dict[str, Any]:
    """Separate pure prose-harming/rogue ghost-sink experts from dual-active specialists.

    An expert ``e`` with ``expert_spec_loss_when_zeroed[e] >= spec_preserve_floor_nats`` is
    *never* placed into ``sink_experts``. If it also exhibits ``expert_e0_harm_nats[e] >=
    prose_harm_floor_nats`` (e.g. T4 ``expert_5`` at ``+0.0278`` nats specialist value),
    it is classified into ``preserved_specialists`` for orthogonal multi-gate router
    surgery rather than FoMoE ghost zeroing.
    """
    spec_floor = _validate_nonneg_finite(spec_preserve_floor_nats, "spec_preserve_floor_nats")
    prose_floor = _validate_nonneg_finite(prose_harm_floor_nats, "prose_harm_floor_nats")
    gen_idx = _validate_nonneg_int(general_expert_index, "general_expert_index")

    harm_map = _parse_expert_metric_map(expert_e0_harm_nats, "expert_e0_harm_nats")
    spec_map = _parse_expert_metric_map(
        expert_spec_loss_when_zeroed, "expert_spec_loss_when_zeroed"
    )
    if set(harm_map.keys()) != set(spec_map.keys()):
        raise ValueError(
            "expert_e0_harm_nats and expert_spec_loss_when_zeroed must cover identical expert indices"
        )

    rogue_set: set[int] = set()
    if rogue_expert_indices is not None:
        if isinstance(rogue_expert_indices, (str, bytes, bytearray)) or not isinstance(
            rogue_expert_indices, (Sequence, set, frozenset)
        ):
            raise TypeError("rogue_expert_indices must be a sequence or set of ints")
        for r_idx in rogue_expert_indices:
            rogue_set.add(_validate_nonneg_int(r_idx, "rogue_expert_index"))

    sink_experts: list[int] = []
    rogue_sink_experts: list[int] = []
    preserved_specialists: list[int] = []
    benign_specialists: list[int] = []
    helpful_general_experts: list[int] = []
    neutral_experts: list[int] = []
    per_expert: dict[str, dict[str, Any]] = {}

    for e_idx in sorted(harm_map.keys()):
        harm = harm_map[e_idx]
        spec_val = spec_map[e_idx]
        is_rogue = e_idx in rogue_set
        harms_prose = harm >= prose_floor
        valuable_spec = spec_val >= spec_floor

        if (harms_prose or is_rogue) and not valuable_spec:
            sink_experts.append(e_idx)
            if is_rogue or spec_val <= 0.0:
                rogue_sink_experts.append(e_idx)
                role = "rogue_sink"
            else:
                role = "prose_harm_sink"
        elif harms_prose and valuable_spec:
            preserved_specialists.append(e_idx)
            role = "dual_active_specialist"
        elif valuable_spec:
            benign_specialists.append(e_idx)
            role = "benign_specialist"
        elif harm <= -prose_floor:
            helpful_general_experts.append(e_idx)
            role = "helpful_general"
        else:
            neutral_experts.append(e_idx)
            role = "neutral"

        per_expert[f"expert_{e_idx}"] = {
            "expert_index": e_idx,
            "e0_harm_nats": harm,
            "spec_loss_when_zeroed": spec_val,
            "is_rogue": is_rogue,
            "role": role,
        }

    sink_set = set(sink_experts)
    all_preserved = [e for e in sorted(harm_map.keys()) if e not in sink_set]
    return {
        "sink_experts": sink_experts,
        "rogue_sink_experts": rogue_sink_experts,
        "preserved_specialists": preserved_specialists,
        "benign_specialists": benign_specialists,
        "helpful_general_experts": helpful_general_experts,
        "neutral_experts": neutral_experts,
        "all_preserved_experts": all_preserved,
        "general_expert_index": gen_idx,
        "protect_general_expert": gen_idx in helpful_general_experts,
        "spec_preserve_floor_nats": spec_floor,
        "prose_harm_floor_nats": prose_floor,
        "per_expert": per_expert,
    }


# ---------------------------------------------------------------------------
# 3. Component (b): 2-Fold OOF Orthogonal Multi-Gate Row Update
# ---------------------------------------------------------------------------


def _normalize_domain_activations(
    domain_spec_activations: Mapping[Any, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor,
    *,
    expected_dim: int,
) -> dict[str, torch.Tensor]:
    """Validate specialist domain activations into ``{domain_key: (N_k, d) float64}``."""
    out: dict[str, torch.Tensor] = {}
    if isinstance(domain_spec_activations, torch.Tensor):
        out["domain_0"] = _validate_activations_2d(
            domain_spec_activations, "domain_spec_activations", expected_dim=expected_dim
        )
        return out

    if isinstance(domain_spec_activations, Mapping):
        if not domain_spec_activations:
            raise ValueError("domain_spec_activations mapping must not be empty")
        for raw_k, raw_t in domain_spec_activations.items():
            if isinstance(raw_k, bool) or not isinstance(raw_k, (int, str)):
                raise TypeError("domain_spec_activations keys must be int or str")
            key_str = str(raw_k).strip()
            if not key_str:
                raise ValueError("domain_spec_activations string keys must be non-empty")
            out[key_str] = _validate_activations_2d(
                raw_t, f"domain_spec_activations[{raw_k!r}]", expected_dim=expected_dim
            )
        return out

    if isinstance(domain_spec_activations, (str, bytes, bytearray)) or not isinstance(
        domain_spec_activations, Sequence
    ):
        raise TypeError(
            "domain_spec_activations must be a Tensor, Mapping, or Sequence of Tensors"
        )
    if len(domain_spec_activations) == 0:
        raise ValueError("domain_spec_activations sequence must not be empty")
    for idx, raw_t in enumerate(domain_spec_activations):
        out[f"domain_{idx}"] = _validate_activations_2d(
            raw_t, f"domain_spec_activations[{idx}]", expected_dim=expected_dim
        )
    return out


def _build_centroid_orthogonal_basis(
    centroids: Sequence[torch.Tensor],
    dim: int,
    sv_rel_tol: float,
) -> torch.Tensor:
    """Return orthonormal columns ``Q_spec`` spanning all non-zero specialist domain means."""
    unit_rows: list[torch.Tensor] = []
    for mu in centroids:
        norm_mu = float(mu.norm().item())
        if norm_mu > 1e-12:
            unit_rows.append(mu / norm_mu)
    if not unit_rows:
        return torch.empty((dim, 0), dtype=torch.float64)

    c_mat = torch.stack(unit_rows, dim=0)
    _, s_vals, vh = torch.linalg.svd(c_mat, full_matrices=False)
    top_sv = float(s_vals[0].item()) if s_vals.numel() > 0 else 0.0
    if top_sv <= 1e-12:
        return torch.empty((dim, 0), dtype=torch.float64)
    thresh = max(float(sv_rel_tol) * top_sv, 1e-12)
    keep = s_vals > thresh
    if not bool(keep.any().item()):
        return torch.empty((dim, 0), dtype=torch.float64)
    return vh[keep, :].T.contiguous()


def _project_orthogonal(tensor: torch.Tensor, q_basis: torch.Tensor) -> torch.Tensor:
    """Apply double Classical Gram-Schmidt projection onto ``q_basis^perp``."""
    if q_basis.shape[1] == 0:
        return tensor
    if tensor.ndim == 1:
        v = tensor - q_basis @ (q_basis.T @ tensor)
        return v - q_basis @ (q_basis.T @ v)
    m = tensor - q_basis @ (q_basis.T @ tensor)
    m = m - q_basis @ (q_basis.T @ m)
    return m


def _solve_orthogonal_prose_probe(
    p_x: torch.Tensor,
    dom_map: Mapping[str, torch.Tensor],
    *,
    omega_spec: float,
    ridge_scale: float,
    sv_rel_tol: float,
) -> tuple[torch.Tensor, int]:
    """Solve ridge-stabilized prose direction strictly orthogonal to all specialist domain means."""
    d = int(p_x.shape[1])
    centroids = [x_d.mean(dim=0) for x_d in dom_map.values()]
    q_spec = _build_centroid_orthogonal_basis(centroids, d, sv_rel_tol)
    subspace_rank = int(q_spec.shape[1])
    if subspace_rank >= d:
        return torch.zeros(d, dtype=torch.float64), subspace_rank

    mu0 = p_x.mean(dim=0)
    mu0_perp = _project_orthogonal(mu0, q_spec)
    if float(mu0_perp.norm().item()) <= 1e-12:
        return torch.zeros(d, dtype=torch.float64), subspace_rank

    m0 = (p_x.T @ p_x) / float(p_x.shape[0])
    s_pool = torch.cat(list(dom_map.values()), dim=0)
    ms = (s_pool.T @ s_pool) / float(s_pool.shape[0])
    comb = m0 + float(omega_spec) * ms
    comb = 0.5 * (comb + comb.T)

    if subspace_rank > 0:
        m_perp = _project_orthogonal(comb, q_spec)
        m_perp = m_perp - (m_perp @ q_spec) @ q_spec.T
        m_perp = m_perp - (m_perp @ q_spec) @ q_spec.T
        m_perp = 0.5 * (m_perp + m_perp.T)
    else:
        m_perp = comb

    tr = max(float(torch.diagonal(m_perp).mean().item()), 1e-6)
    lam = max(float(ridge_scale) * tr, 1e-5)
    reg_mat = m_perp + lam * torch.eye(d, dtype=torch.float64)
    u_raw = torch.linalg.solve(reg_mat, mu0_perp)
    u_raw = _project_orthogonal(u_raw, q_spec)
    denom = float((p_x @ u_raw).mean().item())
    if denom <= 1e-6:
        return torch.zeros(d, dtype=torch.float64), subspace_rank
    u_normed = _project_orthogonal(u_raw / denom, q_spec)
    return u_normed.contiguous(), subspace_rank


def _fit_oof_orthogonal_multigate_probe(
    x_prose_f64: torch.Tensor,
    dom_map_f64: Mapping[str, torch.Tensor],
    *,
    min_oof_snr: float,
    n_folds: int,
    omega_spec: float,
    ridge_scale: float,
    sv_rel_tol: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run ``n_folds`` OOF validation and solve full orthogonal prose direction ``u_full``."""
    n_prose = int(x_prose_f64.shape[0])
    if n_prose < n_folds:
        raise ValueError(f"x_prose must have at least n_folds={n_folds} rows, got {n_prose}")
    total_spec_rows = sum(int(v.shape[0]) for v in dom_map_f64.values())
    if total_spec_rows < n_folds:
        raise ValueError(
            f"Total specialist activation rows ({total_spec_rows}) must be >= n_folds={n_folds}"
        )

    p_oof_parts: list[torch.Tensor] = []
    s_oof_parts: list[torch.Tensor] = []
    fold_p_means: list[float] = []
    fold_s_means: list[float] = []

    p_idx_all = list(range(n_prose))
    for fold_idx in range(n_folds):
        p_val_idx = [i for i in p_idx_all if i % n_folds == fold_idx]
        p_tr_idx = [i for i in p_idx_all if i % n_folds != fold_idx]
        p_tr = x_prose_f64[p_tr_idx]
        p_val = x_prose_f64[p_val_idx]

        s_tr_map: dict[str, torch.Tensor] = {}
        s_val_list: list[torch.Tensor] = []
        tiny_doms: list[torch.Tensor] = []
        for k, x_dom in dom_map_f64.items():
            n_dom = int(x_dom.shape[0])
            if n_dom >= n_folds:
                val_ix = [i for i in range(n_dom) if i % n_folds == fold_idx]
                tr_ix = [i for i in range(n_dom) if i % n_folds != fold_idx]
                s_tr_map[k] = x_dom[tr_ix]
                s_val_list.append(x_dom[val_ix])
            else:
                s_tr_map[k] = x_dom
                tiny_doms.append(x_dom)

        if not s_val_list and tiny_doms:
            tiny_pool = torch.cat(tiny_doms, dim=0)
            tiny_val_ix = [i for i in range(int(tiny_pool.shape[0])) if i % n_folds == fold_idx]
            if tiny_val_ix:
                s_val_list.append(tiny_pool[tiny_val_ix])

        u_fold, _ = _solve_orthogonal_prose_probe(
            p_tr,
            s_tr_map,
            omega_spec=omega_spec,
            ridge_scale=ridge_scale,
            sv_rel_tol=sv_rel_tol,
        )
        p_oof_f = p_val @ u_fold
        p_oof_parts.append(p_oof_f)
        fold_p_means.append(float(p_oof_f.mean().item()))

        s_val_cat = torch.cat(s_val_list, dim=0)
        s_oof_f = s_val_cat @ u_fold
        s_oof_parts.append(s_oof_f)
        fold_s_means.append(float(s_oof_f.mean().item()))

    p_oof = torch.cat(p_oof_parts, dim=0)
    s_oof = torch.cat(s_oof_parts, dim=0)
    p_mean = float(p_oof.mean().item())
    s_mean = float(s_oof.mean().item())
    s_rms = float(s_oof.pow(2).mean().sqrt().item())
    pooled_var = 0.5 * (
        float(p_oof.var(unbiased=False).item()) + float(s_oof.var(unbiased=False).item())
    )
    pooled_std = math.sqrt(pooled_var + 1e-8)
    oof_snr_margin = p_mean - max(0.0, s_mean)
    oof_snr = max(0.0, oof_snr_margin / pooled_std)

    folds_agree = all(
        (fp > 0.25) and (fp > max(0.0, fs))
        for fp, fs in zip(fold_p_means, fold_s_means, strict=True)
    )
    spec_margin_ok = (oof_snr_margin > 0.50 * s_rms) and (p_mean > 0.75 * s_rms)
    gate_passed = bool(folds_agree and spec_margin_ok and (oof_snr > float(min_oof_snr)))
    if gate_passed:
        denom_snr = max(1.0 - float(min_oof_snr), 0.25)
        rel_weight = max(0.0, min(1.0, (oof_snr - float(min_oof_snr)) / denom_snr))
    else:
        rel_weight = 0.0

    u_full, subspace_rank = _solve_orthogonal_prose_probe(
        x_prose_f64,
        dom_map_f64,
        omega_spec=omega_spec,
        ridge_scale=ridge_scale,
        sv_rel_tol=sv_rel_tol,
    )
    if float(u_full.norm().item()) <= 1e-12:
        rel_weight = 0.0
        gate_passed = False

    stats: dict[str, Any] = {
        "oof_prose_mean": p_mean,
        "oof_spec_mean": s_mean,
        "oof_spec_rms": s_rms,
        "oof_snr_margin": oof_snr_margin,
        "oof_snr": oof_snr,
        "min_oof_snr": float(min_oof_snr),
        "n_folds": int(n_folds),
        "folds_agree": folds_agree,
        "spec_margin_ok": spec_margin_ok,
        "gate_passed": gate_passed,
        "rel_weight": rel_weight,
        "centroid_subspace_rank": subspace_rank,
        "direction_norm": float(u_full.norm().item()),
    }
    return u_full, stats


@torch.no_grad()
def fit_oof_orthogonal_multigate_row_update(
    router_row: torch.Tensor,
    x_prose: torch.Tensor,
    domain_spec_activations: Mapping[Any, torch.Tensor] | Sequence[torch.Tensor] | torch.Tensor,
    *,
    router_beta: float = DEFAULT_ROUTER_BETA,
    min_oof_snr: float = DEFAULT_MIN_OOF_SNR,
    n_folds: int = DEFAULT_N_FOLDS,
    omega_spec: float = DEFAULT_OMEGA_SPEC,
    ridge_scale: float = DEFAULT_RIDGE_SCALE,
    sv_rel_tol: float = DEFAULT_SUBSPACE_SV_REL_TOL,
    inplace: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute an OOF SNR-gated rank-1 router row shift orthogonal to all specialist centroids.

    Solves for a prose direction ``u_orth`` strictly inside the orthogonal complement of
    ``span(mu_1, ..., mu_M)`` across every specialist domain mean ``mu_k = E[X_k]``, gated
    by ``n_folds`` out-of-fold validation so noisy/inseparable splits produce zero mutation.
    """
    row_f64 = _validate_router_row_1d(router_row, "router_row")
    d = int(row_f64.shape[0])
    prose_f64 = _validate_activations_2d(x_prose, "x_prose", expected_dim=d)
    dom_map_f64 = _normalize_domain_activations(domain_spec_activations, expected_dim=d)

    beta_f = _validate_nonneg_finite(router_beta, "router_beta")
    min_snr_f = _validate_nonneg_finite(min_oof_snr, "min_oof_snr")
    folds_i = _validate_positive_int(n_folds, "n_folds")
    if folds_i < 2:
        raise ValueError(f"n_folds must be >= 2, got {folds_i}")
    omega_f = _validate_nonneg_finite(omega_spec, "omega_spec")
    ridge_f = _validate_nonneg_finite(ridge_scale, "ridge_scale")
    sv_tol_f = _validate_nonneg_finite(sv_rel_tol, "sv_rel_tol")
    _validate_bool(inplace, "inplace")

    u_full, stats = _fit_oof_orthogonal_multigate_probe(
        prose_f64,
        dom_map_f64,
        min_oof_snr=min_snr_f,
        n_folds=folds_i,
        omega_spec=omega_f,
        ridge_scale=ridge_f,
        sv_rel_tol=sv_tol_f,
    )

    row_before = router_row.detach().clone()
    rel_weight = float(stats["rel_weight"])
    step_scale = beta_f * rel_weight
    if step_scale > 0.0 and float(u_full.norm().item()) > 0.0:
        updated_f64 = row_f64 - step_scale * u_full
        updated_cast = updated_f64.to(device=router_row.device, dtype=router_row.dtype)
        if inplace:
            router_row.copy_(updated_cast)
            out_row = router_row
        else:
            out_row = updated_cast
        out_f64 = out_row.detach().to(device="cpu", dtype=torch.float64)
        applied = bool(not torch.equal(out_row.detach(), row_before))
    else:
        out_row = router_row if inplace else row_before
        out_f64 = row_f64
        applied = False

    delta_f64 = out_f64 - row_f64
    centroids_by_key = {k: v.mean(dim=0) for k, v in dom_map_f64.items()}
    centroid_logit_deltas = {
        k: float((delta_f64 @ mu_k).item()) for k, mu_k in centroids_by_key.items()
    }
    max_centroid_delta = max((abs(v) for v in centroid_logit_deltas.values()), default=0.0)
    max_centroid_dot = max(
        (abs(float((u_full @ mu_k).item())) for mu_k in centroids_by_key.values()),
        default=0.0,
    )
    mean_prose_delta = float((delta_f64 @ prose_f64.mean(dim=0)).item())

    stats.update(
        {
            "applied": applied,
            "router_beta": beta_f,
            "step_scale": step_scale,
            "weight_delta_l2": float(delta_f64.norm().item()),
            "max_centroid_dot_abs": max_centroid_dot,
            "max_centroid_logit_delta": max_centroid_delta,
            "mean_prose_logit_delta": mean_prose_delta,
            "centroid_logit_deltas": centroid_logit_deltas,
        }
    )
    return out_row, stats


# ---------------------------------------------------------------------------
# 4. Component (c): End-to-End WSP S4 Production Adapter Pipeline & ANSE Gate
# ---------------------------------------------------------------------------


def _validate_model_compatible(model: nn.Module) -> None:
    """Verify required CrowdV1-compatible transformer attributes and MoE blocks."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("model must expose a non-empty 'blocks' sequence")
    for attr in ("wte", "ln", "head", "cos_cached", "sin_cached"):
        if not hasattr(model, attr):
            raise ValueError(f"model must expose '{attr}'")
    has_moe = any(
        hasattr(getattr(blk, "mlp", None), "experts") and hasattr(getattr(blk, "mlp", None), "router")
        for blk in blocks
    )
    if not has_moe:
        raise ValueError("model must contain at least one MoE block exposing 'experts' and 'router'")


def _validate_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    general_prefix: str,
    vocab_size: int,
    score_from: int,
    name: str,
    require_specialist: bool,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Validate token chunk mappings and return ``(ids_2d, is_general_mask, expert_ids)``."""
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence) or len(chunks) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")

    rows: list[torch.Tensor] = []
    expert_ids: list[str] = []
    seq_len: int | None = None
    for idx, ch in enumerate(chunks):
        if not isinstance(ch, Mapping) or "expert_id" not in ch or "ids" not in ch:
            raise ValueError(f"{name}[{idx}] must be a mapping containing 'expert_id' and 'ids'")
        eid = str(ch["expert_id"]).strip()
        if not eid:
            raise ValueError(f"{name}[{idx}]['expert_id'] must be a non-empty string")
        ids_t = torch.as_tensor(ch["ids"], dtype=torch.long)
        if ids_t.ndim != 1 or ids_t.numel() < 2:
            raise ValueError(f"{name}[{idx}]['ids'] must be 1D with length >= 2")
        if bool((ids_t < 0).any().item()) or bool((ids_t >= vocab_size).any().item()):
            raise ValueError(f"{name}[{idx}]['ids'] token IDs must lie in [0, {vocab_size})")
        if seq_len is None:
            seq_len = int(ids_t.numel())
        elif int(ids_t.numel()) != seq_len:
            raise ValueError(f"All {name} entries must share sequence length {seq_len}")
        rows.append(ids_t)
        expert_ids.append(eid)

    assert seq_len is not None
    start_pos = max(0, score_from - 1)
    if start_pos >= seq_len - 1:
        raise ValueError(
            f"{name} sequence length {seq_len} too short for score_from={score_from}"
        )

    ids_2d = torch.stack(rows, dim=0)
    is_general = np.array([eid.startswith(general_prefix) for eid in expert_ids], dtype=bool)
    if not bool(np.any(is_general)):
        raise ValueError(
            f"{name} must include at least one general chunk starting with {general_prefix!r}"
        )
    if require_specialist and not bool(np.any(~is_general)):
        raise ValueError(f"{name} must include both general and specialist domain chunks")
    return ids_2d, is_general, expert_ids


def _resolve_bucket_io(
    buckets: Mapping[str, torch.Tensor] | None,
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None,
    extract_buckets_fn: Callable[[nn.Module], dict[str, torch.Tensor]] | None,
) -> tuple[
    Callable[[nn.Module, Mapping[str, torch.Tensor]], None],
    Callable[[nn.Module], dict[str, torch.Tensor]],
]:
    """Bind OpenDiLoCo bucket loader and extractor callbacks."""
    nfrag = (
        max(1, sum(1 for k in buckets if k.startswith("router_frag_")))
        if buckets is not None
        else 1
    )
    loader = load_buckets_fn or (
        lambda m, b: load_moe_trainable_buckets(m, b, num_router_fragments=nfrag)
    )
    extractor = extract_buckets_fn or (
        lambda m: extract_moe_trainable_buckets(m, num_router_fragments=nfrag)
    )
    return loader, extractor


def _extract_ln_params_from_buckets(
    buckets: Mapping[str, torch.Tensor], h_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice final ``(ln.weight, ln.bias)`` from the tail of the last ``router_frag_*`` bucket."""
    router_keys = sorted(k for k in buckets if k.startswith("router_frag_"))
    if not router_keys:
        raise ValueError("buckets must contain at least one 'router_frag_*' entry")
    last_vec = buckets[router_keys[-1]]
    if last_vec.numel() < 2 * h_dim:
        raise ValueError(f"Final router fragment too short ({last_vec.numel()}) for h_dim={h_dim}")
    ln_w = last_vec[-2 * h_dim : -h_dim].detach().to("cpu", dtype=torch.float32).clone()
    ln_b = last_vec[-h_dim:].detach().to("cpu", dtype=torch.float32).clone()
    return ln_w, ln_b


def _compute_chunk_losses(
    model: nn.Module,
    ids_eval: torch.Tensor,
    *,
    score_from: int,
    batch_size: int,
) -> torch.Tensor:
    """Compute per-chunk causal mid-row NLL across all model blocks."""
    cos = model.cos_cached.to(ids_eval.device)
    sin = model.sin_cached.to(ids_eval.device)
    parts: list[torch.Tensor] = []
    for b0 in range(0, int(ids_eval.shape[0]), batch_size):
        xb = compute_embeddings_batch(model, ids_eval[b0 : b0 + batch_size])
        for blk in model.blocks:
            xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
            xb = forward_block_post_mlp(blk, xm, mi)
        parts.append(xb)
    h_pre = torch.cat(parts, dim=0)
    if not bool(torch.isfinite(h_pre).all().item()):
        return torch.full((int(ids_eval.shape[0]),), float("nan"), dtype=torch.float32)
    return per_chunk_causal_midrow_nll(
        h_pre,
        model.ln.weight,
        model.ln.bias,
        float(getattr(model.ln, "eps", 1e-5)),
        model.head.weight,
        ids_eval,
        score_from=score_from,
        seq_batch_size=batch_size,
    )


def _group_chunk_losses(
    losses: torch.Tensor,
    expert_ids: Sequence[str],
) -> tuple[dict[str, float] | None, float | None, bool]:
    """Aggregate per-chunk losses into per-domain NLL means and overall mean."""
    if not bool(torch.isfinite(losses).all().item()):
        return None, None, False
    vals = losses.cpu().tolist()
    grouped: dict[str, list[float]] = {}
    for eid, val in zip(expert_ids, vals, strict=True):
        grouped.setdefault(str(eid), []).append(float(val))
    dom_nll = {dom: float(sum(xs) / len(xs)) for dom, xs in sorted(grouped.items())}
    return dom_nll, float(sum(vals) / len(vals)), True


def _parse_domain_nll_mapping(
    raw_eval: Any,
    *,
    expected_domains: set[str] | None = None,
) -> tuple[dict[str, float] | None, float | None, bool]:
    """Validate and normalize an external domain-NLL mapping."""
    if not isinstance(raw_eval, Mapping) or not raw_eval:
        return None, None, False
    source = raw_eval.get("domain_nll", raw_eval)
    if not isinstance(source, Mapping) or not source:
        return None, None, False
    if expected_domains is not None and set(source.keys()) != expected_domains:
        return None, None, False
    cleaned: dict[str, float] = {}
    for dom in sorted(source.keys()):
        raw_v = source[dom]
        if isinstance(raw_v, bool) or not isinstance(raw_v, (int, float)):
            return None, None, False
        val = float(raw_v)
        if not math.isfinite(val) or val <= 0.0:
            return None, None, False
        cleaned[str(dom)] = val
    return cleaned, float(sum(cleaned.values()) / len(cleaned)), True


def _evaluate_anse_metrics(
    ref_dom_nll: Mapping[str, float],
    cand_dom_nll: Mapping[str, float] | None,
    *,
    cand_nll_mean: float | None,
    cand_finite: bool,
    cfg: S4MultigateSurgeonConfig,
) -> dict[str, Any]:
    """Evaluate ANSE energy, general-domain drift, and <= +1.0% per-domain guard."""
    ref_mean = float(sum(ref_dom_nll.values()) / len(ref_dom_nll))
    gen_keys = [k for k in sorted(ref_dom_nll.keys()) if k.startswith(cfg.general_prefix)]
    spec_keys = [k for k in sorted(ref_dom_nll.keys()) if not k.startswith(cfg.general_prefix)]

    if not cand_finite or cand_dom_nll is None or cand_nll_mean is None:
        return {
            "finite": False,
            "nll_mean": None,
            "base_nll_mean": ref_mean,
            "mean_diff_vs_base": None,
            "general_nll_mean": None,
            "spec_nll_mean": None,
            "general_ratio": None,
            "general_pct_vs_base": None,
            "worst_domain_id": None,
            "worst_domain_ratio": None,
            "worst_domain_pct_vs_base": None,
            "guard_ok": False,
            "anse_energy": float(cfg.maximum_pain_energy),
            "per_domain": {},
        }

    per_domain: dict[str, dict[str, Any]] = {}
    worst_dom = ""
    worst_ratio = -math.inf
    for dom in sorted(ref_dom_nll.keys()):
        b_val = float(ref_dom_nll[dom])
        c_val = float(cand_dom_nll[dom])
        ratio = c_val / max(b_val, 1e-12)
        pct = 100.0 * (ratio - 1.0)
        dom_ok = ratio <= cfg.domain_guard_max_ratio + 1e-12
        per_domain[dom] = {
            "base_nll": b_val,
            "cand_nll": c_val,
            "mean_diff": c_val - b_val,
            "ratio": ratio,
            "pct_vs_base": pct,
            "guard_ok": bool(dom_ok),
        }
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_dom = dom

    guard_ok = all(bool(info["guard_ok"]) for info in per_domain.values())
    general_nll = float(sum(cand_dom_nll[k] for k in gen_keys) / len(gen_keys))
    spec_nll = float(sum(cand_dom_nll[k] for k in spec_keys) / len(spec_keys)) if spec_keys else None
    general_ratio = max(float(per_domain[k]["ratio"]) for k in gen_keys)
    general_pct = 100.0 * (general_ratio - 1.0)
    mean_diff = float(cand_nll_mean) - ref_mean

    pain = (
        float(cfg.maximum_pain_energy)
        if not guard_ok
        else (float(cfg.soft_margin_penalty) if general_ratio > cfg.anse_safe_e0_ratio else 0.0)
    )
    return {
        "finite": True,
        "nll_mean": float(cand_nll_mean),
        "base_nll_mean": ref_mean,
        "mean_diff_vs_base": mean_diff,
        "general_nll_mean": general_nll,
        "spec_nll_mean": spec_nll,
        "general_ratio": general_ratio,
        "general_pct_vs_base": general_pct,
        "worst_domain_id": worst_dom,
        "worst_domain_ratio": worst_ratio,
        "worst_domain_pct_vs_base": 100.0 * (worst_ratio - 1.0),
        "guard_ok": bool(guard_ok),
        "anse_energy": float(mean_diff + pain),
        "per_domain": per_domain,
    }


def _resolve_surgeon_config(
    config: S4MultigateSurgeonConfig | None,
    **overrides: Any,
) -> S4MultigateSurgeonConfig:
    """Merge optional keyword overrides into a validated ``S4MultigateSurgeonConfig``."""
    if config is not None and not isinstance(config, S4MultigateSurgeonConfig):
        raise TypeError(
            f"config must be S4MultigateSurgeonConfig or None, got {type(config).__name__}"
        )
    base_dict = asdict(config) if config is not None else asdict(S4MultigateSurgeonConfig())
    for key, val in overrides.items():
        if val is not None:
            base_dict[key] = val
    return S4MultigateSurgeonConfig(**base_dict)


def _extract_loeo_maps(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    expert_loeo_override: Mapping[str, Mapping[str, float]] | None,
    cfg: S4MultigateSurgeonConfig,
) -> tuple[dict[int, float], dict[int, float], list[int], dict[str, Any]]:
    """Resolve per-expert ``e0_harm_nats``, ``spec_loss_when_zeroed``, and rogue indices."""
    if expert_loeo_override is not None:
        if not isinstance(expert_loeo_override, Mapping) or not expert_loeo_override:
            raise ValueError("expert_loeo_override must be a non-empty mapping")
        harm_map: dict[int, float] = {}
        spec_map: dict[int, float] = {}
        rogue_indices: list[int] = []
        for raw_k, entry in expert_loeo_override.items():
            if not isinstance(entry, Mapping):
                raise TypeError(f"expert_loeo_override[{raw_k!r}] must be a mapping")
            parsed = _parse_expert_metric_map({raw_k: 0.0}, "expert_loeo_override")
            e_idx = next(iter(parsed.keys()))
            if "general_harm_nats" in entry:
                h_val = _validate_finite_float(entry["general_harm_nats"], "general_harm_nats")
            elif "e0_harm_nats" in entry:
                h_val = _validate_finite_float(entry["e0_harm_nats"], "e0_harm_nats")
            elif "delta_e0_when_zeroed" in entry:
                h_val = -_validate_finite_float(entry["delta_e0_when_zeroed"], "delta_e0_when_zeroed")
            else:
                raise ValueError(
                    f"expert_loeo_override[{raw_k!r}] must provide 'general_harm_nats', "
                    "'e0_harm_nats', or 'delta_e0_when_zeroed'"
                )
            s_val = _validate_finite_float(
                entry["delta_spec_when_zeroed"], "delta_spec_when_zeroed"
            )
            harm_map[e_idx] = h_val
            spec_map[e_idx] = s_val
            if bool(entry.get("is_rogue_high_norm", False)):
                rogue_indices.append(e_idx)
        return harm_map, spec_map, sorted(rogue_indices), {"source": "override"}

    loeo_rep = compute_leave_one_expert_out_attribution(
        model,
        probe_chunks,
        score_from=cfg.score_from,
        loeo_chunks_per_domain=cfg.loeo_chunks_per_domain,
        batch_size=cfg.batch_size,
        general_prefix=cfg.general_prefix,
        general_expert_index=cfg.general_expert_index,
    )
    expert_loeo = loeo_rep["expert_loeo"]
    harm_map = {
        int(info["expert_index"]): float(info["general_harm_nats"])
        for info in expert_loeo.values()
    }
    spec_map = {
        int(info["expert_index"]): float(info["delta_spec_when_zeroed"])
        for info in expert_loeo.values()
    }
    rogue_indices = [int(e) for e in loeo_rep.get("rogue_high_norm_experts", [])]
    return harm_map, spec_map, rogue_indices, loeo_rep


def _build_domain_index_map(
    probe_eids: Sequence[str],
    *,
    general_prefix: str,
    general_expert_index: int,
) -> tuple[list[int], dict[str, list[int]]]:
    """Group probe chunk indices into general prose indices and per-domain specialist indices."""
    prose_indices: list[int] = []
    dom_indices: dict[str, list[int]] = {}
    for idx, eid in enumerate(probe_eids):
        if eid.startswith(general_prefix):
            prose_indices.append(idx)
            continue
        match = _EXPERT_DOMAIN_RE.match(eid)
        key = str(int(match.group(1))) if match is not None else eid
        if key == str(general_expert_index):
            key = eid
        dom_indices.setdefault(key, []).append(idx)
    return prose_indices, dom_indices


@torch.no_grad()
def execute_s4_multigate_moe_surgery(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    selection_chunks: Sequence[Mapping[str, Any]] | None = None,
    config: S4MultigateSurgeonConfig | None = None,
    start_buckets: Mapping[str, torch.Tensor] | None = None,
    tuned_buckets: Mapping[str, torch.Tensor] | None = None,
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
    extract_buckets_fn: Callable[[nn.Module], dict[str, torch.Tensor]] | None = None,
    baseline_domain_nll: Mapping[str, float] | None = None,
    eval_fn: Callable[[nn.Module], Mapping[str, Any]] | None = None,
    expert_loeo_override: Mapping[str, Mapping[str, float]] | None = None,
    layer_attribution_override: Sequence[Mapping[str, Any]] | Mapping[Any, Any] | Sequence[float] | None = None,
    target_layers: Sequence[int] | None = None,
    router_beta: float | None = None,
    sink_attract_scale: float | None = None,
    min_oof_snr: float | None = None,
    n_folds: int | None = None,
    spec_preserve_floor_nats: float | None = None,
    prose_harm_floor_nats: float | None = None,
    anchor_threshold_nats: float | None = None,
    domain_guard_max_ratio: float | None = None,
    require_energy_improvement: bool | None = None,
    gate_all_surviving_specialists: bool | None = None,
    score_from: int | None = None,
    batch_size: int | None = None,
    use_ln_calibration: bool | None = None,
    inplace: bool | None = None,
) -> dict[str, Any]:
    """Execute end-to-end WSP Stage S4 selective sinking + orthogonal multi-gate MoE surgery.

    1. Runs per-layer causal bypass attribution and marks beneficial anchor layers
       (``e0_harm_nats <= anchor_threshold_nats``) as bitwise immutable.
    2. Runs LOEO expert attribution and partitions pure prose-harming/rogue ghost sinks
       from dual-active specialists (``>= spec_preserve_floor_nats``).
    3. Propagates probe activations sequentially across blocks: zeros FoMoE sink LoRA
       ``B`` weights, updates preserved specialist router rows via 2-fold OOF orthogonal
       multi-gate surgery, and applies optional 21-point fine LayerNorm calibration.
    4. Enforces ANSE ``<= +1.0%`` per-domain guard with atomic bitwise rollback and
       SHA-256 provenance receipt.
    """
    _validate_model_compatible(model)
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError("eval_fn must be callable or None")

    cfg = _resolve_surgeon_config(
        config,
        router_beta=router_beta,
        sink_attract_scale=sink_attract_scale,
        min_oof_snr=min_oof_snr,
        n_folds=n_folds,
        spec_preserve_floor_nats=spec_preserve_floor_nats,
        prose_harm_floor_nats=prose_harm_floor_nats,
        anchor_threshold_nats=anchor_threshold_nats,
        domain_guard_max_ratio=domain_guard_max_ratio,
        require_energy_improvement=require_energy_improvement,
        gate_all_surviving_specialists=gate_all_surviving_specialists,
        score_from=score_from,
        batch_size=batch_size,
        use_ln_calibration=use_ln_calibration,
        inplace=inplace,
    )

    work_model = model if cfg.inplace else copy.deepcopy(model)
    vocab_size = int(work_model.head.weight.shape[0])
    h_dim = int(work_model.ln.weight.numel())
    n_blocks = len(work_model.blocks)

    ids_probe, is_e0_probe, probe_eids = _validate_chunks(
        probe_chunks,
        general_prefix=cfg.general_prefix,
        vocab_size=vocab_size,
        score_from=cfg.score_from,
        name="probe_chunks",
        require_specialist=True,
    )
    eval_source = selection_chunks if selection_chunks is not None else probe_chunks
    ids_eval, _, eval_eids = _validate_chunks(
        eval_source,
        general_prefix=cfg.general_prefix,
        vocab_size=vocab_size,
        score_from=cfg.score_from,
        name="selection_chunks" if selection_chunks is not None else "probe_chunks",
        require_specialist=False,
    )

    explicit_layers: set[int] | None = None
    if target_layers is not None:
        if isinstance(target_layers, (str, bytes, bytearray)) or not isinstance(target_layers, Sequence):
            raise TypeError("target_layers must be a sequence of layer indices")
        t_list = sorted({_validate_nonneg_int(idx, "layer_index") for idx in target_layers})
        if not t_list or any(idx >= n_blocks for idx in t_list):
            raise ValueError(f"target_layers must be a non-empty subset of [0, {n_blocks})")
        explicit_layers = set(t_list)

    loader, extractor = _resolve_bucket_io(
        tuned_buckets or start_buckets, load_buckets_fn, extract_buckets_fn
    )
    if tuned_buckets is not None:
        loader(work_model, {k: v.clone() for k, v in tuned_buckets.items()})

    base_nll_override: dict[str, float] | None = None
    if baseline_domain_nll is not None:
        parsed_b, _, b_ok = _parse_domain_nll_mapping(
            baseline_domain_nll, expected_domains=set(eval_eids)
        )
        if not b_ok or parsed_b is None:
            raise ValueError(
                "baseline_domain_nll must be a finite positive mapping matching eval domains"
            )
        base_nll_override = parsed_b

    was_training = work_model.training
    work_model.eval()
    try:
        parent_snapshot = {
            k: v.detach().to(device="cpu").clone() for k, v in work_model.state_dict().items()
        }
        parent_sha = compute_state_sha256(parent_snapshot)
        parent_layer_sha = {
            str(i): compute_module_state_sha256(blk) for i, blk in enumerate(work_model.blocks)
        }

        def _eval_domains(expected_keys: set[str] | None = None) -> tuple[dict[str, float] | None, float | None, bool]:
            if eval_fn is not None:
                return _parse_domain_nll_mapping(eval_fn(work_model), expected_domains=expected_keys)
            losses = _compute_chunk_losses(
                work_model, ids_eval, score_from=cfg.score_from, batch_size=cfg.batch_size
            )
            return _group_chunk_losses(losses, eval_eids)

        parent_dom_nll, parent_nll_mean, parent_finite = _eval_domains(
            set(eval_eids) if base_nll_override is not None else None
        )
        if not parent_finite or parent_dom_nll is None or parent_nll_mean is None:
            raise ValueError("Parent model evaluation returned invalid or non-finite NLLs")

        ref_dom_nll = base_nll_override if base_nll_override is not None else parent_dom_nll
        expected_eval_keys = set(ref_dom_nll.keys())
        parent_anse = _evaluate_anse_metrics(
            ref_dom_nll,
            parent_dom_nll,
            cand_nll_mean=parent_nll_mean,
            cand_finite=True,
            cfg=cfg,
        )

        # Optional start-bucket baseline probe losses & LN params for 21-point fine LN calibration
        base_probe_losses: np.ndarray | None = None
        start_ln_w = start_ln_b = tuned_ln_w = tuned_ln_b = None
        if start_buckets is not None and cfg.use_ln_calibration:
            start_ln_w, start_ln_b = _extract_ln_params_from_buckets(start_buckets, h_dim)
            if tuned_buckets is not None:
                tuned_ln_w, tuned_ln_b = _extract_ln_params_from_buckets(tuned_buckets, h_dim)
            else:
                tuned_ln_w = work_model.ln.weight.detach().to("cpu", dtype=torch.float32).clone()
                tuned_ln_b = work_model.ln.bias.detach().to("cpu", dtype=torch.float32).clone()
            loader(work_model, {k: v.clone() for k, v in start_buckets.items()})
            base_probe_losses = (
                _compute_chunk_losses(
                    work_model, ids_probe, score_from=cfg.score_from, batch_size=cfg.batch_size
                )
                .cpu()
                .numpy()
            )
            cur_sd = work_model.state_dict()
            work_model.load_state_dict(
                {k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in parent_snapshot.items()}
            )

        # Stage A: Beneficial-anchor layer detection
        layer_attr_source: Any = (
            layer_attribution_override
            if layer_attribution_override is not None
            else compute_per_layer_moe_bypass_attribution(
                work_model,
                probe_chunks,
                score_from=cfg.score_from,
                batch_size=cfg.batch_size,
                general_prefix=cfg.general_prefix,
            )
        )
        layer_partition = identify_beneficial_anchor_and_interfering_layers(
            layer_attr_source,
            anchor_threshold_nats=cfg.anchor_threshold_nats,
            harm_threshold_nats=cfg.harm_threshold_nats,
        )
        anchor_layers: list[int] = (
            list(layer_partition["beneficial_anchor_layers"]) if cfg.protect_anchor_layers else []
        )
        anchor_set = set(anchor_layers)
        editable_layers = [
            l_idx
            for l_idx in range(n_blocks)
            if l_idx not in anchor_set and (explicit_layers is None or l_idx in explicit_layers)
        ]
        editable_set = set(editable_layers)
        anchor_sha_before = {str(l_idx): parent_layer_sha[str(l_idx)] for l_idx in anchor_layers}

        # Stage B: LOEO attribution & sink vs preserved specialist partition
        harm_map, spec_map, rogue_indices, loeo_diag = _extract_loeo_maps(
            work_model,
            probe_chunks,
            expert_loeo_override=expert_loeo_override,
            cfg=cfg,
        )
        partition = partition_sink_vs_preserved_specialists(
            harm_map,
            spec_map,
            spec_preserve_floor_nats=cfg.spec_preserve_floor_nats,
            prose_harm_floor_nats=cfg.prose_harm_floor_nats,
            rogue_expert_indices=rogue_indices,
            general_expert_index=cfg.general_expert_index,
        )
        sink_experts: list[int] = list(partition["sink_experts"])
        preserved_specialists: list[int] = list(partition["preserved_specialists"])
        if cfg.gate_all_surviving_specialists:
            sink_set = set(sink_experts)
            gated_targets = [
                e for e in sorted(harm_map.keys()) if e not in sink_set and e != cfg.general_expert_index
            ]
        else:
            gated_targets = list(preserved_specialists)

        # Stage C: Sequential layer-wise selective FoMoE sinking + orthogonal multi-gate surgery
        prose_idx, dom_idx_map = _build_domain_index_map(
            probe_eids,
            general_prefix=cfg.general_prefix,
            general_expert_index=cfg.general_expert_index,
        )
        start_pos = max(0, cfg.score_from - 1)
        cos = work_model.cos_cached.to(ids_probe.device)
        sin = work_model.sin_cached.to(ids_probe.device)
        x_seq = torch.cat(
            [
                compute_embeddings_batch(work_model, ids_probe[b0 : b0 + cfg.batch_size])
                for b0 in range(0, int(ids_probe.shape[0]), cfg.batch_size)
            ],
            dim=0,
        )

        layer_reports: list[dict[str, Any]] = []
        snr_margins: list[float] = []
        oof_snrs: list[float] = []
        max_centroid_logit_delta = 0.0

        for l_idx, blk in enumerate(work_model.blocks):
            xm_parts: list[torch.Tensor] = []
            mi_parts: list[torch.Tensor] = []
            for b0 in range(0, int(x_seq.shape[0]), cfg.batch_size):
                xm, mi = forward_block_pre_mlp(blk, x_seq[b0 : b0 + cfg.batch_size], cos, sin)
                xm_parts.append(xm)
                mi_parts.append(mi)
            x_mid = torch.cat(xm_parts, dim=0)
            mlp_in = torch.cat(mi_parts, dim=0)

            mlp = getattr(blk, "mlp", None)
            has_moe = hasattr(mlp, "experts") and hasattr(mlp, "router")
            if has_moe and l_idx in editable_set:
                zeroed_params = zero_expert_lora_b_inplace(mlp, sink_experts) if sink_experts else 0
                scored = mlp_in[:, start_pos:-1, :]
                hd = int(scored.shape[-1])
                x_prose_l = scored[prose_idx].reshape(-1, hd)
                dom_spec_l = {
                    dom_k: scored[idxs].reshape(-1, hd)
                    for dom_k, idxs in dom_idx_map.items()
                    if idxs
                }

                row_updates: dict[str, dict[str, Any]] = {}
                router_w: torch.Tensor = mlp.router.weight
                num_rows = int(router_w.shape[0])
                for e_idx in gated_targets:
                    if e_idx >= num_rows:
                        continue
                    _, row_stats = fit_oof_orthogonal_multigate_row_update(
                        router_w[e_idx],
                        x_prose_l,
                        dom_spec_l,
                        router_beta=cfg.router_beta,
                        min_oof_snr=cfg.min_oof_snr,
                        n_folds=cfg.n_folds,
                        omega_spec=cfg.omega_spec,
                        ridge_scale=cfg.ridge_scale,
                        sv_rel_tol=cfg.sv_rel_tol,
                        inplace=True,
                    )
                    row_updates[f"expert_{e_idx}"] = row_stats
                    snr_margins.append(float(row_stats["oof_snr_margin"]))
                    oof_snrs.append(float(row_stats["oof_snr"]))
                    max_centroid_logit_delta = max(
                        max_centroid_logit_delta, float(row_stats["max_centroid_logit_delta"])
                    )

                sink_attracted = False
                if sink_experts and cfg.sink_attract_scale > 0.0 and cfg.router_beta > 0.0:
                    sink_anchor = sink_experts[0]
                    if sink_anchor < num_rows:
                        u_sink_dir, sink_probe = _fit_oof_orthogonal_multigate_probe(
                            x_prose_l.detach().to(device="cpu", dtype=torch.float64),
                            {k: v.detach().to(device="cpu", dtype=torch.float64) for k, v in dom_spec_l.items()},
                            min_oof_snr=cfg.min_oof_snr,
                            n_folds=cfg.n_folds,
                            omega_spec=cfg.omega_spec,
                            ridge_scale=cfg.ridge_scale,
                            sv_rel_tol=cfg.sv_rel_tol,
                        )
                        sink_rel = float(sink_probe["rel_weight"])
                        if sink_rel > 0.0 and float(u_sink_dir.norm().item()) > 0.0:
                            step = (
                                cfg.sink_attract_scale * cfg.router_beta * sink_rel
                            ) * u_sink_dir.to(device=router_w.device, dtype=router_w.dtype)
                            router_w[sink_anchor].add_(step)
                            sink_attracted = True

                layer_reports.append(
                    {
                        "layer": l_idx,
                        "protected_anchor": False,
                        "sink_experts": sink_experts,
                        "zeroed_lora_b_elements": zeroed_params,
                        "gated_targets": gated_targets,
                        "sink_attracted": sink_attracted,
                        "row_updates": row_updates,
                    }
                )
            elif l_idx in anchor_set:
                layer_reports.append(
                    {
                        "layer": l_idx,
                        "protected_anchor": True,
                        "sink_experts": [],
                        "zeroed_lora_b_elements": 0,
                        "gated_targets": [],
                        "sink_attracted": False,
                        "row_updates": {},
                    }
                )

            x_seq = torch.cat(
                [
                    forward_block_post_mlp(
                        blk,
                        x_mid[b0 : b0 + cfg.batch_size],
                        mlp_in[b0 : b0 + cfg.batch_size],
                    )
                    for b0 in range(0, int(x_mid.shape[0]), cfg.batch_size)
                ],
                dim=0,
            )

        anchor_sha_after = {
            str(l_idx): compute_module_state_sha256(work_model.blocks[l_idx])
            for l_idx in anchor_layers
        }
        anchor_preserved = anchor_sha_before == anchor_sha_after
        if not anchor_preserved:
            raise RuntimeError("Beneficial anchor layer bitwise preservation invariant violated")

        # Stage D: Optional 21-point fine LayerNorm calibration
        lambda_ln_applied: float | None = None
        if (
            base_probe_losses is not None
            and start_ln_w is not None
            and start_ln_b is not None
            and tuned_ln_w is not None
            and tuned_ln_b is not None
        ):
            best_l, _ = select_guard_optimal_lambda_ln(
                x_seq,
                start_ln_w,
                start_ln_b,
                tuned_ln_w,
                tuned_ln_b,
                float(getattr(work_model.ln, "eps", 1e-5)),
                work_model.head.weight,
                ids_probe,
                base_probe_losses,
                is_e0_probe,
                score_from=cfg.score_from,
                target_e0_max_ratio=cfg.fine_ln_e0_max_ratio,
                lambda_grid=FINE_LAMBDA_LN_GRID,
            )
            work_model.ln.weight.copy_(
                (start_ln_w + best_l * (tuned_ln_w - start_ln_w)).to(work_model.ln.weight.dtype)
            )
            work_model.ln.bias.copy_(
                (start_ln_b + best_l * (tuned_ln_b - start_ln_b)).to(work_model.ln.bias.dtype)
            )
            lambda_ln_applied = best_l

        candidate_sha = compute_module_state_sha256(work_model)

        # Stage E: ANSE <= +1.0% guard & atomic bitwise rollback
        eval_error: str | None = None
        try:
            cand_dom_nll, cand_nll_mean, cand_finite = _eval_domains(expected_eval_keys)
            cand_anse = _evaluate_anse_metrics(
                ref_dom_nll,
                cand_dom_nll,
                cand_nll_mean=cand_nll_mean,
                cand_finite=cand_finite,
                cfg=cfg,
            )
        except Exception as exc:
            eval_error = f"{type(exc).__name__}: {exc}"
            cand_anse = _evaluate_anse_metrics(
                ref_dom_nll, None, cand_nll_mean=None, cand_finite=False, cfg=cfg
            )

        parent_e = float(parent_anse["anse_energy"])
        cand_e = float(cand_anse["anse_energy"])
        delta_e = cand_e - parent_e
        energy_ok = (
            (delta_e < -cfg.min_energy_improvement)
            if cfg.require_energy_improvement
            else (delta_e <= 0.0)
        )

        if candidate_sha == parent_sha:
            status = "FIXED_POINT_UNCHANGED"
            accepted = False
            rolled_back = False
        elif eval_error is None and bool(cand_anse["guard_ok"]) and energy_ok:
            status = "ACCEPTED"
            accepted = True
            rolled_back = False
        else:
            if eval_error is not None:
                status = "ROLLED_BACK_EVAL_FAILURE"
            elif not bool(cand_anse["guard_ok"]):
                status = "ROLLED_BACK_GUARD_VIOLATION"
            else:
                status = "ROLLED_BACK_ENERGY_NON_IMPROVING"
            accepted = False
            rolled_back = True
            cur_sd = work_model.state_dict()
            work_model.load_state_dict(
                {k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in parent_snapshot.items()}
            )

        final_sha = compute_module_state_sha256(work_model)
        if rolled_back and final_sha != parent_sha:
            raise RuntimeError(
                f"Atomic bitwise rollback verification failed: {final_sha[:12]} != {parent_sha[:12]}"
            )
        output_buckets_sha = (
            compute_state_sha256(extractor(work_model))
            if (start_buckets is not None or tuned_buckets is not None or extract_buckets_fn is not None)
            else None
        )
    finally:
        work_model.train(was_training)

    active_eval = cand_anse if accepted else parent_anse
    receipt: dict[str, Any] = {
        "object": RECEIPT_OBJECT,
        "schema": SCHEMA_VERSION,
        "status": status,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "eval_error": eval_error,
        "guard_ok": bool(cand_anse["guard_ok"]),
        "nll_mean": float(active_eval["nll_mean"]),
        "candidate_nll_mean": cand_anse["nll_mean"],
        "base_nll_mean": float(parent_anse["base_nll_mean"]),
        "parent_nll_mean": float(parent_anse["nll_mean"]),
        "mean_diff_vs_base": float(active_eval["mean_diff_vs_base"]),
        "general_nll_mean": float(active_eval["general_nll_mean"]),
        "spec_nll_mean": active_eval["spec_nll_mean"],
        "general_ratio": active_eval["general_ratio"],
        "general_pct_vs_base": active_eval["general_pct_vs_base"],
        "worst_domain_id": cand_anse["worst_domain_id"],
        "worst_domain_ratio": cand_anse["worst_domain_ratio"],
        "worst_domain_pct_vs_base": cand_anse["worst_domain_pct_vs_base"],
        "anse_energy": cand_e,
        "parent_anse_energy": parent_e,
        "final_anse_energy": cand_e if accepted else parent_e,
        "delta_e": delta_e,
        "oof_snr_margin": float(sum(snr_margins) / len(snr_margins)) if snr_margins else 0.0,
        "mean_oof_snr": float(sum(oof_snrs) / len(oof_snrs)) if oof_snrs else 0.0,
        "max_centroid_logit_delta": max_centroid_logit_delta,
        "sink_experts": sink_experts,
        "preserved_specialists": preserved_specialists,
        "gated_targets": gated_targets,
        "beneficial_anchor_layers": anchor_layers,
        "editable_layers": editable_layers,
        "anchor_layers_bitwise_preserved": anchor_preserved,
        "anchor_layer_sha256_before": anchor_sha_before,
        "anchor_layer_sha256_after": anchor_sha_after,
        "lambda_ln_applied": lambda_ln_applied,
        "partition": partition,
        "loeo_summary": loeo_diag if expert_loeo_override is None else {"source": "override"},
        "layer_attribution": layer_partition,
        "layer_reports": layer_reports,
        "per_domain": cand_anse["per_domain"],
        "parent_state_sha256": parent_sha,
        "candidate_state_sha256": candidate_sha,
        "final_state_sha256": final_sha,
        "output_buckets_sha256": output_buckets_sha,
        "config": asdict(cfg),
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


__all__ = [
    "DEFAULT_ANCHOR_THRESHOLD_NATS",
    "DEFAULT_ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_ANSE_SOFT_MARGIN_PENALTY",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_FINE_LN_E0_MAX_RATIO",
    "DEFAULT_GENERAL_EXPERT_INDEX",
    "DEFAULT_HARM_THRESHOLD_NATS",
    "DEFAULT_LOEO_CHUNKS_PER_DOMAIN",
    "DEFAULT_MIN_OOF_SNR",
    "DEFAULT_N_FOLDS",
    "DEFAULT_OMEGA_SPEC",
    "DEFAULT_PROSE_HARM_FLOOR_NATS",
    "DEFAULT_RIDGE_SCALE",
    "DEFAULT_ROUTER_BETA",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_SINK_ATTRACT_SCALE",
    "DEFAULT_SPEC_PRESERVE_FLOOR_NATS",
    "DEFAULT_SUBSPACE_SV_REL_TOL",
    "FINE_LAMBDA_LN_GRID",
    "GENERAL_EXPERT_PREFIX",
    "RECEIPT_OBJECT",
    "SCHEMA_VERSION",
    "S4MultigateSurgeonConfig",
    "canonical_json_bytes",
    "compute_module_state_sha256",
    "compute_payload_sha256",
    "compute_state_sha256",
    "execute_s4_multigate_moe_surgery",
    "fit_oof_orthogonal_multigate_row_update",
    "partition_sink_vs_preserved_specialists",
    "verify_s4_multigate_receipt_provenance",
]

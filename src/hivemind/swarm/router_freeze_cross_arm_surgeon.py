# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Post-DiLoCo Router-Frozen Anchor + Router-Unfrozen Specialist Cross-Arm Synergy Surgeon.

Implements a training-free weight-space synergy engine for open-weight Sparse MoEs
(``CrowdV1WithEngram``, ``CrowdLoRAMoEMLP``, ``CrowdFineGrainedMoEMLP``, and structured
router/expert tensor dictionaries), grounded in Campaign-3 ``E-MOE-C3`` findings:
router-frozen Arm D/C preserved general prose (``E0 = +0.098%`` at ``mean_diff = -0.0943``
nats), whereas router-unfrozen Arm B unlocked 3x deeper specialist gain (``E6 = -43.16%``,
``mean_diff = -0.2824`` nats) at the cost of ``E0 = +13.77%`` router-induced prose drift.

1. ``decompose_router_vs_expert_prose_drift``:
   Attributes general-prose KL/output/NLL drift across counterfactual ``(router, experts)``
   combinations into exact telescoping router-gate-shift, expert-MLP-delta, and
   frozen-router-anchor components plus per-expert prose activation diagnostics.
2. ``graft_specialist_experts_onto_safe_router``:
   Grafts deep-specialist unfrozen-arm experts (pruning ghost-sink experts via bitwise
   FoMoE zeroing and projecting dual-active deltas into AlphaEdit prose null-space ``P_0``)
   onto prose-safe frozen-arm router gates with selective out-of-fold (OOF) SNR-gated
   logit blending strictly on verified specialist directions inside ``P_0``.
3. ``execute_verified_cross_arm_moe_surgery``:
   Commits the hybrid MoE checkpoint if and only if it beats BOTH donor arms under
   ``ANSE Delta E < 0`` and general-prose drift ``<= +1.0%``; otherwise enforces ANSE
   Maximum Pain atomic bitwise rollback with a canonical SHA-256 provenance receipt.
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
import torch.nn.functional as F

from hivemind.swarm.byzantine_wsp_delta_aggregator import compute_payload_sha256
from hivemind.swarm.crowdgpt_moe_opendiloco import CrowdFineGrainedMoEMLP, CrowdLoRAMoEMLP
from hivemind.swarm.lora_svd_mode_surgery import decompose_lora_qr_svd
from hivemind.swarm.self_healing_checkpoint_rescuer import (
    apply_checkpoint_blocks_inplace,
    compute_checkpoint_sha256,
    extract_checkpoint_blocks,
)
from hivemind.swarm.tangent_task_vector_surgeon import compute_prose_nullspace_projector_scaled
from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_nll_from_pre_ln,
)

SCHEMA_VERSION: str = "hivemind.router_freeze_cross_arm_surgeon/1"
RECEIPT_OBJECT: str = "hivemind.cross_arm_moe_surgery_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_GATE_BLEND_ALPHA: float = 0.15
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_MIN_OOF_SNR: float = 0.25
DEFAULT_N_FOLDS: int = 2
DEFAULT_MAX_OOF_PROSE_LEAK_RATIO: float = 0.35
DEFAULT_MIN_NULLSPACE_ALIGNMENT: float = 0.05
DEFAULT_NULLSPACE_SV_REL_TOL: float = 1e-3
DEFAULT_NULLSPACE_VARIANCE_THRESHOLD: float = 0.999
DEFAULT_NULLSPACE_EIG_THRESHOLD: float = 0.5
DEFAULT_NULLSPACE_LEAK_TOL: float = 1e-6
DEFAULT_SINK_PROSE_HARM_FLOOR_NATS: float = 0.002
DEFAULT_SINK_SPEC_PRESERVE_FLOOR_NATS: float = 0.015
DEFAULT_SCORE_FROM: int = 64
DEFAULT_BATCH_SIZE: int = 24
DEFAULT_LOEO_CHUNKS_PER_DOMAIN: int = 8

VALID_ROUTING_MODES: frozenset[str] = frozenset({"sigma_topk", "softmax_topk"})
VALID_ROLLBACK_TARGETS: frozenset[str] = frozenset({"frozen_arm", "base", "unfrozen_arm"})

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPERT_DOMAIN_RE = re.compile(r"^E(\d+)(?:_|$)")
_BLOCK_ROUTER_RE = re.compile(r"^blocks\.(\d+)\.mlp\.router$")
_BLOCK_EXPERT_RE = re.compile(r"^blocks\.(\d+)\.mlp\.experts\.(\d+)$")
_LAYER_KEY_RE = re.compile(r"^(?:blocks\.|layers\.|layer_)?(\d+)$")
_SWIGLU_PROJECTIONS: tuple[str, ...] = ("w1", "w2", "w3")


# ---------------------------------------------------------------------------
# 1. Validation, Configuration & Provenance Helpers
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


def _validate_unit_interval(
    value: Any,
    name: str,
    *,
    allow_zero: bool = True,
) -> float:
    """Validate a finite real number in ``[0.0, 1.0]`` (or ``(0.0, 1.0]``)."""
    out = _validate_nonneg_finite(value, name)
    if not allow_zero and out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    if out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    return out


def _validate_hidden_states_2d(
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


def _validate_square_projector(
    proj: Any,
    name: str,
    *,
    expected_dim: int | None = None,
) -> torch.Tensor:
    """Validate a finite symmetric square 2D projector matrix on CPU ``float64``."""
    if not isinstance(proj, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(proj).__name__}")
    if proj.ndim != 2 or proj.shape[0] == 0 or proj.shape[0] != proj.shape[1]:
        raise ValueError(f"{name} must be a non-empty square 2D matrix, got shape {tuple(proj.shape)}")
    if not proj.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {proj.dtype}")
    if expected_dim is not None and int(proj.shape[0]) != expected_dim:
        raise ValueError(
            f"{name} shape {tuple(proj.shape)} must match expected ({expected_dim}, {expected_dim})"
        )
    mat = proj.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(mat).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    asym = float((mat - mat.T).abs().max().item())
    scale = max(1.0, float(mat.abs().max().item()))
    if asym > 1e-4 * scale:
        raise ValueError(f"{name} must be symmetric (max asymmetry {asym:.3e})")
    return (0.5 * (mat + mat.T)).contiguous()


def _validate_expert_index_set(
    indices: Any,
    name: str,
) -> set[int]:
    """Validate a sequence or set of non-negative integer expert indices."""
    if isinstance(indices, (str, bytes, bytearray)) or not isinstance(indices, (Sequence, set, frozenset)):
        raise TypeError(f"{name} must be a sequence or set of non-negative integers")
    return {_validate_nonneg_int(item, f"{name} element") for item in indices}


def verify_cross_arm_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
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


@dataclass(frozen=True)
class CrossArmSurgeryConfig:
    """Validated configuration governing post-DiLoCo cross-arm router/expert surgery."""

    gate_blend_alpha: float = DEFAULT_GATE_BLEND_ALPHA
    routing_mode: str = "sigma_topk"
    top_k: int = 2
    auto_detect_sink_experts: bool = True
    sink_prose_harm_floor_nats: float = DEFAULT_SINK_PROSE_HARM_FLOOR_NATS
    sink_spec_preserve_floor_nats: float = DEFAULT_SINK_SPEC_PRESERVE_FLOOR_NATS
    auto_project_p_null: bool = True
    nullspace_sv_rel_tol: float = DEFAULT_NULLSPACE_SV_REL_TOL
    nullspace_variance_threshold: float = DEFAULT_NULLSPACE_VARIANCE_THRESHOLD
    nullspace_eig_threshold: float = DEFAULT_NULLSPACE_EIG_THRESHOLD
    nullspace_leak_tol: float = DEFAULT_NULLSPACE_LEAK_TOL
    min_oof_snr: float = DEFAULT_MIN_OOF_SNR
    n_folds: int = DEFAULT_N_FOLDS
    max_oof_prose_leak_ratio: float = DEFAULT_MAX_OOF_PROSE_LEAK_RATIO
    min_nullspace_alignment: float = DEFAULT_MIN_NULLSPACE_ALIGNMENT
    fold_ln_lambda: float | None = None
    max_general_drift_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    max_worst_domain_drift_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    enforce_all_domain_guard: bool = True
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    require_beat_both_donors: bool = True
    min_energy_improvement: float = 0.0
    rollback_target: str = "frozen_arm"
    general_prefix: str = GENERAL_EXPERT_PREFIX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    loeo_chunks_per_domain: int = DEFAULT_LOEO_CHUNKS_PER_DOMAIN
    inplace: bool = True

    def __post_init__(self) -> None:
        _validate_unit_interval(self.gate_blend_alpha, "gate_blend_alpha", allow_zero=True)
        if not isinstance(self.routing_mode, str) or self.routing_mode not in VALID_ROUTING_MODES:
            raise ValueError(f"routing_mode must be one of {sorted(VALID_ROUTING_MODES)}")
        _validate_positive_int(self.top_k, "top_k")
        _validate_bool(self.auto_detect_sink_experts, "auto_detect_sink_experts")
        _validate_nonneg_finite(self.sink_prose_harm_floor_nats, "sink_prose_harm_floor_nats")
        _validate_nonneg_finite(self.sink_spec_preserve_floor_nats, "sink_spec_preserve_floor_nats")
        _validate_bool(self.auto_project_p_null, "auto_project_p_null")
        _validate_nonneg_finite(self.nullspace_sv_rel_tol, "nullspace_sv_rel_tol")
        _validate_unit_interval(
            self.nullspace_variance_threshold, "nullspace_variance_threshold", allow_zero=False
        )
        eig_t = _validate_unit_interval(
            self.nullspace_eig_threshold, "nullspace_eig_threshold", allow_zero=True
        )
        if eig_t >= 1.0:
            raise ValueError(f"nullspace_eig_threshold must be < 1.0, got {eig_t}")
        _validate_nonneg_finite(self.nullspace_leak_tol, "nullspace_leak_tol")
        _validate_nonneg_finite(self.min_oof_snr, "min_oof_snr")
        n_f = _validate_positive_int(self.n_folds, "n_folds")
        if n_f < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_f}")
        _validate_positive_finite(self.max_oof_prose_leak_ratio, "max_oof_prose_leak_ratio")
        _validate_unit_interval(
            self.min_nullspace_alignment, "min_nullspace_alignment", allow_zero=True
        )
        if self.fold_ln_lambda is not None:
            _validate_unit_interval(self.fold_ln_lambda, "fold_ln_lambda", allow_zero=True)
        max_gen = _validate_positive_finite(self.max_general_drift_ratio, "max_general_drift_ratio")
        max_worst = _validate_positive_finite(
            self.max_worst_domain_drift_ratio, "max_worst_domain_drift_ratio"
        )
        safe_r = _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        if max_gen < 1.0 or max_worst < 1.0 or safe_r < 1.0:
            raise ValueError("Drift ratio bounds and anse_safe_general_ratio must be >= 1.0")
        _validate_bool(self.enforce_all_domain_guard, "enforce_all_domain_guard")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_bool(self.require_beat_both_donors, "require_beat_both_donors")
        _validate_nonneg_finite(self.min_energy_improvement, "min_energy_improvement")
        if not isinstance(self.rollback_target, str) or self.rollback_target not in VALID_ROLLBACK_TARGETS:
            raise ValueError(f"rollback_target must be one of {sorted(VALID_ROLLBACK_TARGETS)}")
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_positive_int(self.loeo_chunks_per_domain, "loeo_chunks_per_domain")
        _validate_bool(self.inplace, "inplace")


# ---------------------------------------------------------------------------
# 2. Router, Expert & Hidden-State Normalization Utilities
# ---------------------------------------------------------------------------


def _parse_layer_key(raw_key: Any, name: str) -> int:
    """Parse layer index from integer or layer-prefixed string."""
    if isinstance(raw_key, bool):
        raise TypeError(f"{name} layer key must not be bool")
    if isinstance(raw_key, int):
        return _validate_nonneg_int(raw_key, f"{name} layer key")
    if isinstance(raw_key, str):
        stripped = raw_key.strip()
        m_router = _BLOCK_ROUTER_RE.match(stripped)
        if m_router is not None:
            return int(m_router.group(1))
        m_layer = _LAYER_KEY_RE.match(stripped)
        if m_layer is not None:
            return int(m_layer.group(1))
    raise ValueError(f"Invalid layer key {raw_key!r} in {name}")


def _extract_single_router_dict(obj: Any, name: str) -> tuple[dict[str, torch.Tensor], str]:
    """Normalize one router object into ``({'weight': (E, H), optional 'bias': (E,)}, form)``."""
    if isinstance(obj, nn.Linear):
        w = obj.weight
        form = "linear"
        out: dict[str, torch.Tensor] = {"weight": w.detach().clone()}
        if obj.bias is not None:
            out["bias"] = obj.bias.detach().clone()
    elif isinstance(obj, torch.Tensor):
        form = "tensor"
        out = {"weight": obj.detach().clone()}
    elif isinstance(obj, Mapping) and "weight" in obj and isinstance(obj["weight"], torch.Tensor):
        form = "mapping"
        out = {"weight": obj["weight"].detach().clone()}
        if "bias" in obj and obj["bias"] is not None:
            if not isinstance(obj["bias"], torch.Tensor):
                raise TypeError(f"{name}['bias'] must be a torch.Tensor")
            out["bias"] = obj["bias"].detach().clone()
    else:
        raise TypeError(
            f"{name} entries must be a 2D Tensor, nn.Linear, or {{'weight': Tensor}}, got {type(obj).__name__}"
        )

    w_t = out["weight"]
    if w_t.ndim != 2 or w_t.shape[0] < 2 or w_t.shape[1] < 1:
        raise ValueError(f"{name}.weight must be non-empty 2D with >= 2 experts, got {tuple(w_t.shape)}")
    if not w_t.is_floating_point() or not bool(torch.isfinite(w_t).all().item()):
        raise ValueError(f"{name}.weight must be finite and floating-point")
    if "bias" in out:
        b_t = out["bias"]
        if b_t.ndim != 1 or b_t.shape[0] != w_t.shape[0]:
            raise ValueError(f"{name}.bias shape {tuple(b_t.shape)} != ({w_t.shape[0]},)")
        if not b_t.is_floating_point() or not bool(torch.isfinite(b_t).all().item()):
            raise ValueError(f"{name}.bias must be finite and floating-point")
    return out, form


def _normalize_routers_by_layer(
    routers: Any,
    name: str,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, Any]]:
    """Normalize model, MoE MLP, single router, sequence, or mapping into ``{layer_idx: {'weight', ...}}``."""
    if isinstance(routers, nn.Module):
        blocks = getattr(routers, "blocks", None)
        if blocks is not None:
            if len(blocks) == 0:
                raise ValueError(f"{name}.blocks must not be empty")
            out_m: dict[int, dict[str, torch.Tensor]] = {}
            for l_idx, blk in enumerate(blocks):
                mlp = getattr(blk, "mlp", blk)
                r_mod = getattr(mlp, "router", None)
                if r_mod is None:
                    raise ValueError(f"{name}.blocks[{l_idx}].mlp exposes no 'router'")
                out_m[l_idx], _ = _extract_single_router_dict(r_mod, f"{name}.blocks[{l_idx}].router")
            return out_m, {"style": "model", "layer_keys": {l_idx: l_idx for l_idx in out_m}}
        if hasattr(routers, "router"):
            r_dict, _ = _extract_single_router_dict(routers.router, f"{name}.router")
            return {0: r_dict}, {"style": "moe_mlp", "layer_keys": {0: 0}}
        if isinstance(routers, nn.Linear):
            r_dict, _ = _extract_single_router_dict(routers, name)
            return {0: r_dict}, {"style": "single_linear", "layer_keys": {0: 0}}
        raise TypeError(f"{name} module {type(routers).__name__} exposes no blocks or router")

    if isinstance(routers, torch.Tensor):
        r_dict, _ = _extract_single_router_dict(routers, name)
        return {0: r_dict}, {"style": "single_tensor", "layer_keys": {0: 0}}

    if isinstance(routers, Mapping):
        if not routers:
            raise ValueError(f"{name} mapping must not be empty")
        if "weight" in routers and isinstance(routers["weight"], torch.Tensor):
            r_dict, _ = _extract_single_router_dict(routers, name)
            return {0: r_dict}, {"style": "single_mapping", "layer_keys": {0: 0}}

        router_entries = {
            k: v
            for k, v in routers.items()
            if isinstance(k, str) and _BLOCK_ROUTER_RE.match(k.strip()) is not None
        }
        use_map = router_entries if router_entries else routers
        out_map: dict[int, dict[str, torch.Tensor]] = {}
        orig_keys: dict[int, Any] = {}
        entry_forms: dict[int, str] = {}
        for raw_k, raw_v in use_map.items():
            l_idx = _parse_layer_key(raw_k, name)
            if l_idx in out_map:
                raise ValueError(f"Duplicate layer index {l_idx} in {name}")
            r_dict, form = _extract_single_router_dict(raw_v, f"{name}[{raw_k!r}]")
            out_map[l_idx] = r_dict
            orig_keys[l_idx] = raw_k
            entry_forms[l_idx] = form
        return out_map, {
            "style": "layer_mapping",
            "layer_keys": orig_keys,
            "entry_forms": entry_forms,
        }

    if isinstance(routers, Sequence) and not isinstance(routers, (str, bytes, bytearray)):
        if not routers:
            raise ValueError(f"{name} sequence must not be empty")
        out_seq: dict[int, dict[str, torch.Tensor]] = {}
        entry_forms_seq: dict[int, str] = {}
        for l_idx, item in enumerate(routers):
            r_dict, form = _extract_single_router_dict(item, f"{name}[{l_idx}]")
            out_seq[l_idx] = r_dict
            entry_forms_seq[l_idx] = form
        return out_seq, {
            "style": "layer_sequence",
            "layer_keys": {i: i for i in out_seq},
            "entry_forms": entry_forms_seq,
        }

    raise TypeError(
        f"{name} must be a Tensor, Linear, Mapping, Sequence, or MoE module, got {type(routers).__name__}"
    )


def _extract_single_expert_spec(
    obj: Any,
    name: str,
    hidden_size: int,
) -> dict[str, Any]:
    """Normalize one expert object into ``{'kind', 'tensors', 'scaling', 'style', 'source'}``."""
    if isinstance(obj, nn.Module):
        if all(hasattr(obj, f"{p}_{s}") for p in _SWIGLU_PROJECTIONS for s in ("A", "B")):
            tensors = {
                f"{p}_{s}": getattr(obj, f"{p}_{s}").detach().clone()
                for p in _SWIGLU_PROJECTIONS
                for s in ("A", "B")
            }
            scaling = float(getattr(obj, "scaling", 2.0))
            style = "lora_module"
        elif all(isinstance(getattr(obj, p, None), nn.Linear) for p in _SWIGLU_PROJECTIONS):
            tensors = {p: getattr(obj, p).weight.detach().clone() for p in _SWIGLU_PROJECTIONS}
            scaling = 1.0
            style = "swiglu_module"
        else:
            raise TypeError(f"{name} module ({type(obj).__name__}) must expose LoRA or SwiGLU weights")
        source_obj: Any = obj
    elif isinstance(obj, torch.Tensor):
        tensors = {"delta": obj.detach().clone()}
        scaling = 1.0
        style = "tensor"
        source_obj = None
    elif isinstance(obj, Mapping) and obj:
        tensors = {}
        scaling = 2.0
        for raw_k, raw_v in obj.items():
            if raw_k == "scaling":
                scaling = _validate_positive_finite(raw_v, f"{name}['scaling']")
                continue
            if not isinstance(raw_k, str) or not raw_k.strip():
                raise ValueError(f"{name} keys must be non-empty strings")
            if not isinstance(raw_v, torch.Tensor):
                raise TypeError(f"{name}[{raw_k!r}] must be a torch.Tensor")
            tensors[raw_k] = raw_v.detach().clone()
        if not tensors:
            raise ValueError(f"{name} must contain at least one parameter tensor")
        style = "dict"
        source_obj = None
    else:
        raise TypeError(
            f"{name} must be an expert nn.Module, Tensor, or non-empty parameter dict, got {type(obj).__name__}"
        )

    for k, t in tensors.items():
        if t.ndim not in (1, 2) or any(d == 0 for d in t.shape):
            raise ValueError(f"{name}[{k!r}] must be non-empty 1D/2D tensor, got {tuple(t.shape)}")
        if not t.is_floating_point() or not bool(torch.isfinite(t).all().item()):
            raise ValueError(f"{name}[{k!r}] must be finite and floating-point")

    if all(f"{p}_{s}" in tensors for p in _SWIGLU_PROJECTIONS for s in ("A", "B")):
        kind = "lora"
        if int(tensors["w1_A"].shape[1]) != hidden_size or int(tensors["w2_A"].shape[1]) != hidden_size:
            raise ValueError(f"{name} LoRA w1_A/w2_A input dim must equal hidden_size={hidden_size}")
        if int(tensors["w3_B"].shape[0]) != hidden_size:
            raise ValueError(f"{name} LoRA w3_B output dim must equal hidden_size={hidden_size}")
    elif all(p in tensors for p in _SWIGLU_PROJECTIONS):
        kind = "swiglu"
        if int(tensors["w1"].shape[1]) != hidden_size or int(tensors["w2"].shape[1]) != hidden_size:
            raise ValueError(f"{name} SwiGLU w1/w2 input dim must equal hidden_size={hidden_size}")
        if int(tensors["w3"].shape[0]) != hidden_size:
            raise ValueError(f"{name} SwiGLU w3 output dim must equal hidden_size={hidden_size}")
    else:
        kind = "linear"
        for k, t in tensors.items():
            if t.ndim == 1 and int(t.shape[0]) != hidden_size:
                raise ValueError(f"{name}[{k!r}] 1D length must equal hidden_size={hidden_size}")
            if t.ndim == 2 and int(t.shape[0]) != hidden_size and int(t.shape[1]) != hidden_size:
                raise ValueError(
                    f"{name}[{k!r}] shape {tuple(t.shape)} incompatible with hidden_size={hidden_size}"
                )

    return {
        "kind": kind,
        "tensors": tensors,
        "scaling": scaling,
        "style": style,
        "source": source_obj,
    }


def _parse_expert_index(raw_k: Any, name: str) -> int:
    """Parse non-negative expert index from int or string like ``'1'`` / ``'expert_1'``."""
    if isinstance(raw_k, bool):
        raise TypeError(f"{name} expert key must not be bool")
    if isinstance(raw_k, int):
        return _validate_nonneg_int(raw_k, f"{name} expert key")
    if isinstance(raw_k, str):
        m = re.match(r"^(?:expert_)?(\d+)$", raw_k.strip())
        if m is not None:
            return int(m.group(1))
    raise ValueError(f"Invalid expert key {raw_k!r} in {name}")


def _is_single_expert_value(val: Any) -> bool:
    """Return True if ``val`` looks like a single expert rather than a layer-to-experts map."""
    if isinstance(val, (nn.Module, torch.Tensor)):
        return True
    if isinstance(val, Mapping) and val:
        return all(isinstance(v, torch.Tensor) or k == "scaling" for k, v in val.items())
    return False


def _normalize_experts_by_layer(
    experts: Any,
    name: str,
    *,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
) -> tuple[dict[int, dict[int, dict[str, Any]]], dict[str, Any]]:
    """Normalize experts container into ``{layer_idx: {expert_idx: expert_spec}}``."""
    if isinstance(experts, nn.Module):
        blocks = getattr(experts, "blocks", None)
        if blocks is not None:
            if len(blocks) != num_layers:
                raise ValueError(f"{name} layer count {len(blocks)} != router layer count {num_layers}")
            out_m: dict[int, dict[int, dict[str, Any]]] = {}
            for l_idx, blk in enumerate(blocks):
                mlp = getattr(blk, "mlp", blk)
                out_m[l_idx] = _extract_experts_from_moe_mlp(
                    mlp, f"{name}.blocks[{l_idx}].mlp", hidden_size
                )
            return out_m, {"style": "model"}
        if isinstance(experts, (CrowdLoRAMoEMLP, CrowdFineGrainedMoEMLP)) or hasattr(experts, "experts"):
            if num_layers != 1:
                raise ValueError(f"Single-layer MoE {name} cannot match {num_layers} router layers")
            return {0: _extract_experts_from_moe_mlp(experts, name, hidden_size)}, {"style": "moe_mlp"}
        raise TypeError(f"{name} module {type(experts).__name__} exposes no MoE experts")

    if isinstance(experts, Mapping):
        if not experts:
            raise ValueError(f"{name} mapping must not be empty")
        block_exp_entries = {
            k: v
            for k, v in experts.items()
            if isinstance(k, str) and _BLOCK_EXPERT_RE.match(k.strip()) is not None
        }
        if block_exp_entries:
            out_blk: dict[int, dict[int, dict[str, Any]]] = {}
            orig_keys_blk: dict[tuple[int, int], Any] = {}
            for raw_k, raw_v in block_exp_entries.items():
                m = _BLOCK_EXPERT_RE.match(raw_k.strip())
                assert m is not None
                l_idx, e_idx = int(m.group(1)), int(m.group(2))
                spec = _extract_single_expert_spec(raw_v, f"{name}[{raw_k!r}]", hidden_size)
                out_blk.setdefault(l_idx, {})[e_idx] = spec
                orig_keys_blk[(l_idx, e_idx)] = raw_k
            if len(out_blk) != num_layers:
                raise ValueError(f"{name} layer count {len(out_blk)} != {num_layers}")
            return out_blk, {"style": "checkpoint_blocks", "expert_keys": orig_keys_blk}

        first_val = next(iter(experts.values()))
        if _is_single_expert_value(first_val):
            if num_layers != 1:
                raise ValueError(f"Single-layer {name} cannot match {num_layers} router layers")
            layer_exp: dict[int, dict[str, Any]] = {}
            orig_exp_keys: dict[int, Any] = {}
            for raw_e, raw_v in experts.items():
                e_idx = _parse_expert_index(raw_e, name)
                if e_idx >= num_experts:
                    raise ValueError(f"{name} expert index {e_idx} >= num_experts={num_experts}")
                layer_exp[e_idx] = _extract_single_expert_spec(raw_v, f"{name}[{raw_e!r}]", hidden_size)
                orig_exp_keys[e_idx] = raw_e
            return {0: layer_exp}, {"style": "single_layer_map", "expert_keys": orig_exp_keys}

        out_nested: dict[int, dict[int, dict[str, Any]]] = {}
        layer_keys_n: dict[int, Any] = {}
        exp_keys_n: dict[tuple[int, int], Any] = {}
        for raw_l, sub_map in experts.items():
            l_idx = _parse_layer_key(raw_l, name)
            if not isinstance(sub_map, Mapping) or not sub_map:
                raise TypeError(f"{name}[{raw_l!r}] must be a non-empty expert mapping")
            layer_keys_n[l_idx] = raw_l
            out_nested[l_idx] = {}
            for raw_e, raw_v in sub_map.items():
                e_idx = _parse_expert_index(raw_e, f"{name}[{raw_l!r}]")
                if e_idx >= num_experts:
                    raise ValueError(f"{name}[{raw_l!r}] expert {e_idx} >= num_experts={num_experts}")
                out_nested[l_idx][e_idx] = _extract_single_expert_spec(
                    raw_v, f"{name}[{raw_l!r}][{raw_e!r}]", hidden_size
                )
                exp_keys_n[(l_idx, e_idx)] = raw_e
        if len(out_nested) != num_layers:
            raise ValueError(f"{name} layer count {len(out_nested)} != {num_layers}")
        return out_nested, {
            "style": "nested_layer_map",
            "layer_keys": layer_keys_n,
            "expert_keys": exp_keys_n,
        }

    if isinstance(experts, Sequence) and not isinstance(experts, (str, bytes, bytearray)):
        if len(experts) != num_layers:
            raise ValueError(f"{name} sequence length {len(experts)} != {num_layers}")
        out_seq: dict[int, dict[int, dict[str, Any]]] = {}
        exp_keys_seq: dict[tuple[int, int], Any] = {}
        for l_idx, sub in enumerate(experts):
            if isinstance(sub, Mapping) and sub:
                out_seq[l_idx] = {}
                for raw_e, raw_v in sub.items():
                    e_idx = _parse_expert_index(raw_e, f"{name}[{l_idx}]")
                    out_seq[l_idx][e_idx] = _extract_single_expert_spec(
                        raw_v, f"{name}[{l_idx}][{raw_e!r}]", hidden_size
                    )
                    exp_keys_seq[(l_idx, e_idx)] = raw_e
            else:
                raise TypeError(f"{name}[{l_idx}] must be a non-empty mapping of experts")
        return out_seq, {"style": "layer_sequence", "expert_keys": exp_keys_seq}

    raise TypeError(f"{name} must be an MoE module, Mapping, or Sequence, got {type(experts).__name__}")


def _extract_experts_from_moe_mlp(
    mlp: Any,
    name: str,
    hidden_size: int,
) -> dict[int, dict[str, Any]]:
    """Extract normalized expert specs from a ``CrowdLoRAMoEMLP`` or ``CrowdFineGrainedMoEMLP``."""
    out: dict[int, dict[str, Any]] = {}
    if isinstance(mlp, CrowdLoRAMoEMLP) or hasattr(mlp, "experts"):
        for e_key in sorted(mlp.experts.keys(), key=lambda k: int(k)):
            e_idx = int(e_key)
            out[e_idx] = _extract_single_expert_spec(
                mlp.experts[e_key], f"{name}.experts[{e_key!r}]", hidden_size
            )
        return out
    if isinstance(mlp, CrowdFineGrainedMoEMLP):
        for e_idx in range(mlp.num_experts):
            spec_dict = {
                "w1": mlp.experts_w1[e_idx].weight,
                "w2": mlp.experts_w2[e_idx].weight,
                "w3": mlp.experts_w3[e_idx].weight,
            }
            out[e_idx] = _extract_single_expert_spec(
                spec_dict, f"{name}.experts[{e_idx}]", hidden_size
            )
        return out
    raise TypeError(f"{name} ({type(mlp).__name__}) exposes no MoE experts")


def _extract_base_mlps_by_layer(source: Any) -> dict[int, dict[str, torch.Tensor]]:
    """Extract frozen ``base_mlp`` (w1, w2, w3) weights per layer when available on ``nn.Module``."""
    out: dict[int, dict[str, torch.Tensor]] = {}
    if not isinstance(source, nn.Module):
        return out
    blocks = getattr(source, "blocks", None)
    if blocks is not None:
        for l_idx, blk in enumerate(blocks):
            base_m = getattr(getattr(blk, "mlp", blk), "base_mlp", None)
            if base_m is not None and all(hasattr(base_m, p) for p in _SWIGLU_PROJECTIONS):
                out[l_idx] = {p: getattr(base_m, p).weight.detach().clone() for p in _SWIGLU_PROJECTIONS}
    elif hasattr(source, "base_mlp"):
        base_m = source.base_mlp
        if all(hasattr(base_m, p) for p in _SWIGLU_PROJECTIONS):
            out[0] = {p: getattr(base_m, p).weight.detach().clone() for p in _SWIGLU_PROJECTIONS}
    return out


def _resolve_hidden_states_by_layer(
    hidden_states: torch.Tensor | Mapping[Any, torch.Tensor] | Sequence[torch.Tensor],
    layer_indices: Sequence[int],
    hidden_size: int,
    name: str,
) -> dict[int, torch.Tensor]:
    """Normalize shared or per-layer hidden states into ``{layer_idx: (N, H) float64}``."""
    if isinstance(hidden_states, torch.Tensor):
        flat = _validate_hidden_states_2d(hidden_states, name, expected_dim=hidden_size)
        return {l_idx: flat for l_idx in layer_indices}

    if isinstance(hidden_states, Mapping):
        if not hidden_states:
            raise ValueError(f"{name} mapping must not be empty")
        resolved: dict[int, torch.Tensor] = {}
        for l_idx in layer_indices:
            found: torch.Tensor | None = None
            for cand in (l_idx, str(l_idx), f"blocks.{l_idx}", f"layer_{l_idx}", "default", "x"):
                if cand in hidden_states:
                    found = _validate_hidden_states_2d(
                        hidden_states[cand], f"{name}[{cand!r}]", expected_dim=hidden_size
                    )
                    break
            if found is None:
                raise ValueError(f"{name} missing hidden states for layer {l_idx}")
            resolved[l_idx] = found
        return resolved

    if isinstance(hidden_states, Sequence) and not isinstance(hidden_states, (str, bytes, bytearray)):
        if not hidden_states:
            raise ValueError(f"{name} sequence must not be empty")
        if len(hidden_states) == 1:
            flat = _validate_hidden_states_2d(hidden_states[0], f"{name}[0]", expected_dim=hidden_size)
            return {l_idx: flat for l_idx in layer_indices}
        if len(hidden_states) != len(layer_indices):
            raise ValueError(f"{name} sequence length {len(hidden_states)} != {len(layer_indices)}")
        return {
            l_idx: _validate_hidden_states_2d(hidden_states[pos], f"{name}[{pos}]", expected_dim=hidden_size)
            for pos, l_idx in enumerate(layer_indices)
        }

    raise TypeError(f"{name} must be a Tensor, Mapping, or Sequence of Tensors, got {type(hidden_states).__name__}")


def _validate_router_layout_match(
    ref_routers: Mapping[int, Mapping[str, torch.Tensor]],
    other_routers: Mapping[int, Mapping[str, torch.Tensor]],
    ref_name: str,
    other_name: str,
) -> tuple[list[int], int, int]:
    """Verify matching layer indices and router weight/bias shapes."""
    if set(ref_routers.keys()) != set(other_routers.keys()):
        raise ValueError(
            f"{ref_name} and {other_name} layer indices must match: "
            f"{sorted(ref_routers)} vs {sorted(other_routers)}"
        )
    layer_indices = sorted(ref_routers.keys())
    num_experts, hidden_size = ref_routers[layer_indices[0]]["weight"].shape
    for l_idx in layer_indices:
        rw_ref = ref_routers[l_idx]["weight"]
        rw_oth = other_routers[l_idx]["weight"]
        if rw_ref.shape != (num_experts, hidden_size) or rw_oth.shape != (num_experts, hidden_size):
            raise ValueError(
                f"Router weight shape mismatch at layer {l_idx}: "
                f"{tuple(rw_ref.shape)} vs {tuple(rw_oth.shape)}"
            )
        if ("bias" in ref_routers[l_idx]) != ("bias" in other_routers[l_idx]):
            raise ValueError(f"Router bias presence mismatch at layer {l_idx}")
    return layer_indices, int(num_experts), int(hidden_size)


# ---------------------------------------------------------------------------
# 3. Component (a): Router vs Expert Prose Drift & KL Decomposition
# ---------------------------------------------------------------------------


def _compute_gates_and_probs(
    logits: torch.Tensor,
    *,
    routing_mode: str,
    top_k: int,
    fine_grained_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ``(gates, top_idx, bernoulli_probs, normalized_probs)`` in ``float64``."""
    num_experts = int(logits.shape[1])
    k_eff = min(top_k, num_experts)
    if routing_mode == "sigma_topk":
        sig = torch.sigmoid(logits)
        top_vals, top_idx = torch.topk(sig, k=k_eff, dim=-1)
        scale = (2.0 * float(num_experts)) / float(k_eff) if fine_grained_scale else 1.0
        gates = torch.zeros_like(logits)
        gates.scatter_(1, top_idx, top_vals * scale)
        norm_p = sig / sig.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return gates, top_idx, sig, norm_p

    probs = F.softmax(logits, dim=-1)
    top_vals, top_idx = torch.topk(probs, k=k_eff, dim=-1)
    renorm = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = float(num_experts) if fine_grained_scale else 1.0
    gates = torch.zeros_like(logits)
    gates.scatter_(1, top_idx, renorm * scale)
    return gates, top_idx, probs, probs


def _compute_routing_kl(
    p_cand: torch.Tensor,
    p_ref: torch.Tensor,
    q_bern_cand: torch.Tensor,
    q_bern_ref: torch.Tensor,
    *,
    routing_mode: str,
) -> tuple[float, float]:
    """Compute exact ``(primary_kl_nats, normalized_categorical_kl_nats)`` >= 0.0."""
    eps = 1e-12
    pc = p_cand.clamp(min=eps, max=1.0)
    pr = p_ref.clamp(min=eps, max=1.0)
    cat_kl = float((pc * (pc.log() - pr.log())).sum(dim=-1).mean().item())
    cat_kl = max(0.0, cat_kl)
    if routing_mode == "sigma_topk":
        bc = q_bern_cand.clamp(min=eps, max=1.0 - eps)
        br = q_bern_ref.clamp(min=eps, max=1.0 - eps)
        bern_kl = float(
            (bc * (bc.log() - br.log()) + (1.0 - bc) * ((1.0 - bc).log() - (1.0 - br).log()))
            .sum(dim=-1)
            .mean()
            .item()
        )
        return max(0.0, bern_kl), cat_kl
    return cat_kl, cat_kl


def _eval_expert_mlp_outputs(
    x_f64: torch.Tensor,
    experts_map: Mapping[int, Mapping[str, Any]],
    *,
    num_experts: int,
    base_mlp_weights: Mapping[str, torch.Tensor] | None,
) -> dict[int, torch.Tensor]:
    """Evaluate unweighted per-expert output ``dy_e(x)`` in CPU ``float64``."""
    n_tok, hidden_size = int(x_f64.shape[0]), int(x_f64.shape[1])
    h1_base: torch.Tensor | None = None
    h2_base: torch.Tensor | None = None
    w3_base: torch.Tensor | None = None
    if base_mlp_weights is not None:
        w1_b = base_mlp_weights["w1"].detach().to("cpu", dtype=torch.float64)
        w2_b = base_mlp_weights["w2"].detach().to("cpu", dtype=torch.float64)
        w3_b = base_mlp_weights["w3"].detach().to("cpu", dtype=torch.float64)
        h1_base = x_f64 @ w1_b.T
        h2_base = x_f64 @ w2_b.T
        w3_base = w3_b

    outputs: dict[int, torch.Tensor] = {}
    for e_idx in range(num_experts):
        if e_idx not in experts_map:
            outputs[e_idx] = torch.zeros((n_tok, hidden_size), dtype=torch.float64)
            continue
        spec = experts_map[e_idx]
        kind = spec["kind"]
        t_map = spec["tensors"]
        scaling = float(spec["scaling"])
        if kind == "lora":
            w1_a = t_map["w1_A"].detach().to("cpu", dtype=torch.float64)
            w1_b = t_map["w1_B"].detach().to("cpu", dtype=torch.float64)
            w2_a = t_map["w2_A"].detach().to("cpu", dtype=torch.float64)
            w2_b = t_map["w2_B"].detach().to("cpu", dtype=torch.float64)
            w3_a = t_map["w3_A"].detach().to("cpu", dtype=torch.float64)
            w3_b = t_map["w3_B"].detach().to("cpu", dtype=torch.float64)
            dh1 = scaling * ((x_f64 @ w1_a.T) @ w1_b.T)
            dh2 = scaling * ((x_f64 @ w2_a.T) @ w2_b.T)
            if h1_base is not None and h2_base is not None and w3_base is not None:
                act_full = F.silu(h1_base + dh1) * (h2_base + dh2)
                act_base = F.silu(h1_base) * h2_base
                base_diff = (act_full - act_base) @ w3_base.T
            else:
                act_full = F.silu(dh1) * dh2
                base_diff = torch.zeros((n_tok, hidden_size), dtype=torch.float64)
            lora_w3 = scaling * ((act_full @ w3_a.T) @ w3_b.T)
            outputs[e_idx] = base_diff + lora_w3
        elif kind == "swiglu":
            w1 = t_map["w1"].detach().to("cpu", dtype=torch.float64)
            w2 = t_map["w2"].detach().to("cpu", dtype=torch.float64)
            w3 = t_map["w3"].detach().to("cpu", dtype=torch.float64)
            outputs[e_idx] = (F.silu(x_f64 @ w1.T) * (x_f64 @ w2.T)) @ w3.T
        else:
            acc = torch.zeros((n_tok, hidden_size), dtype=torch.float64)
            for t in t_map.values():
                t64 = t.detach().to("cpu", dtype=torch.float64)
                if t64.ndim == 2 and t64.shape == (hidden_size, hidden_size):
                    acc = acc + (x_f64 @ t64.T)
                elif t64.ndim == 2 and t64.shape[1] == hidden_size:
                    acc = acc + (x_f64 @ t64.T).sum(dim=-1, keepdim=True)
                elif t64.ndim == 2 and t64.shape[0] == hidden_size:
                    proj = (x_f64.mean(dim=-1, keepdim=True) * torch.ones((1, t64.shape[1]), dtype=torch.float64)) @ t64.T
                    acc = acc + proj
                else:
                    acc = acc + (x_f64 @ t64).unsqueeze(-1)
            outputs[e_idx] = acc
    return outputs


def _combine_gated_outputs(
    gates: torch.Tensor,
    expert_outputs: Mapping[int, torch.Tensor],
    num_experts: int,
) -> torch.Tensor:
    """Compute ``sum_e gates[:, e:e+1] * dy_e(x)``."""
    first = expert_outputs[0]
    acc = torch.zeros_like(first)
    for e_idx in range(num_experts):
        acc = acc + gates[:, e_idx : e_idx + 1] * expert_outputs[e_idx]
    return acc


def _compute_router_logits(
    x_f64: torch.Tensor,
    router_dict: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Compute ``x @ W^T (+ b)`` in ``float64``."""
    w = router_dict["weight"].detach().to("cpu", dtype=torch.float64)
    logits = x_f64 @ w.T
    if "bias" in router_dict:
        b = router_dict["bias"].detach().to("cpu", dtype=torch.float64)
        logits = logits + b.unsqueeze(0)
    return logits


def _build_zero_base_experts(
    ref_experts: Mapping[int, Mapping[int, Mapping[str, Any]]],
) -> dict[int, dict[int, dict[str, Any]]]:
    """Construct zero-delta baseline experts matching ``ref_experts`` for LoRA/linear specs."""
    base_out: dict[int, dict[int, dict[str, Any]]] = {}
    for l_idx, exp_map in ref_experts.items():
        base_out[l_idx] = {}
        for e_idx, spec in exp_map.items():
            kind = spec["kind"]
            t_map = spec["tensors"]
            new_t = {
                k: (torch.zeros_like(v) if (kind == "linear" or k.endswith("_B")) else v.clone())
                for k, v in t_map.items()
            }
            base_out[l_idx][e_idx] = {
                "kind": kind,
                "tensors": new_t,
                "scaling": float(spec["scaling"]),
                "style": "dict",
                "source": None,
            }
    return base_out


def _resolve_optional_base_mlps(
    base_mlps: Any,
    *router_or_expert_sources: Any,
) -> dict[int, dict[str, torch.Tensor]]:
    """Collect optional frozen dense ``base_mlp`` weights from explicit arg or MoE modules."""
    if isinstance(base_mlps, Mapping):
        out: dict[int, dict[str, torch.Tensor]] = {}
        for k, v in base_mlps.items():
            l_idx = _parse_layer_key(k, "base_mlps")
            if isinstance(v, nn.Module):
                out[l_idx] = {p: getattr(v, p).weight.detach().clone() for p in _SWIGLU_PROJECTIONS}
            elif isinstance(v, Mapping):
                out[l_idx] = {p: v[p].detach().clone() for p in _SWIGLU_PROJECTIONS}
            else:
                raise TypeError(f"base_mlps[{k!r}] must be an nn.Module or Mapping")
        return out
    if base_mlps is not None:
        raise TypeError("base_mlps must be a Mapping[layer, Module|Mapping] or None")
    for src in router_or_expert_sources:
        extracted = _extract_base_mlps_by_layer(src)
        if extracted:
            return extracted
    return {}


def _eval_nll_from_hidden_and_perturbation(
    h_base: torch.Tensor,
    perturbation: torch.Tensor,
    lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
    final_ln_params: Mapping[str, Mapping[str, torch.Tensor]] | None,
) -> float:
    """Compute exact cross-entropy NLL from final hidden perturbation + LM head."""
    h = h_base + perturbation
    if final_ln_params and "final_ln" in final_ln_params:
        ln_p = final_ln_params["final_ln"]
        ln_w = ln_p["weight"].detach().to("cpu", dtype=torch.float64)
        ln_b = ln_p["bias"].detach().to("cpu", dtype=torch.float64) if "bias" in ln_p else None
        h = F.layer_norm(h, (h.shape[-1],), ln_w, ln_b, eps=1e-5)
    head = lm_head_weight.detach().to("cpu", dtype=torch.float64)
    logits = F.linear(h, head)
    return float(F.cross_entropy(logits, targets.detach().to("cpu", dtype=torch.long)).item())


@torch.no_grad()
def decompose_router_vs_expert_prose_drift(
    base_routers: Any,
    frozen_arm_routers: Any,
    unfrozen_arm_routers: Any,
    unfrozen_experts: Any,
    prose_hidden_states: torch.Tensor | Mapping[Any, torch.Tensor] | Sequence[torch.Tensor],
    *,
    base_experts: Any = None,
    base_mlps: Any = None,
    routing_mode: str = "sigma_topk",
    top_k: int = 2,
    lm_head_weight: torch.Tensor | None = None,
    prose_targets: torch.Tensor | None = None,
    final_ln_params: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    prose_nll_eval_fn: Callable[[dict[str, Any]], float] | None = None,
    spec_hidden_states: torch.Tensor | Mapping[Any, torch.Tensor] | Sequence[torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Attribute general-prose KL, output-energy, and optional NLL drift between router gate shifts and expert MLP deltas.

    Uses the exact counterfactual telescoping identity per layer ``l``:
    ``Delta_total = [y(R_unf, E_unf) - y(R_frz, E_unf)] + [y(R_frz, E_unf) - y(R_frz, E_base)] + [y(R_frz, E_base) - y(R_base, E_base)]``
    separating router-induced prose drift from residual expert-MLP prose drift under the
    safe frozen-arm router.
    """
    if not isinstance(routing_mode, str) or routing_mode not in VALID_ROUTING_MODES:
        raise ValueError(f"routing_mode must be one of {sorted(VALID_ROUTING_MODES)}")
    top_k_i = _validate_positive_int(top_k, "top_k")

    base_r, _ = _normalize_routers_by_layer(base_routers, "base_routers")
    frz_r, _ = _normalize_routers_by_layer(frozen_arm_routers, "frozen_arm_routers")
    unf_r, _ = _normalize_routers_by_layer(unfrozen_arm_routers, "unfrozen_arm_routers")
    layer_indices, num_experts, hidden_size = _validate_router_layout_match(
        base_r, frz_r, "base_routers", "frozen_arm_routers"
    )
    _validate_router_layout_match(base_r, unf_r, "base_routers", "unfrozen_arm_routers")

    unf_e, _ = _normalize_experts_by_layer(
        unfrozen_experts,
        "unfrozen_experts",
        num_layers=len(layer_indices),
        num_experts=num_experts,
        hidden_size=hidden_size,
    )
    if base_experts is not None:
        base_e, _ = _normalize_experts_by_layer(
            base_experts,
            "base_experts",
            num_layers=len(layer_indices),
            num_experts=num_experts,
            hidden_size=hidden_size,
        )
    else:
        base_e = _build_zero_base_experts(unf_e)

    prose_by_layer = _resolve_hidden_states_by_layer(
        prose_hidden_states, layer_indices, hidden_size, "prose_hidden_states"
    )
    spec_by_layer = (
        _resolve_hidden_states_by_layer(
            spec_hidden_states, layer_indices, hidden_size, "spec_hidden_states"
        )
        if spec_hidden_states is not None
        else None
    )
    mlp_map = _resolve_optional_base_mlps(
        base_mlps, unfrozen_experts, base_experts, base_routers, unfrozen_arm_routers
    )

    if (lm_head_weight is None) != (prose_targets is None):
        raise ValueError("lm_head_weight and prose_targets must either both be set or both be None")
    if lm_head_weight is not None and prose_targets is not None:
        if not isinstance(lm_head_weight, torch.Tensor) or lm_head_weight.ndim != 2:
            raise ValueError("lm_head_weight must be a 2D tensor")
        if int(lm_head_weight.shape[1]) != hidden_size:
            raise ValueError(
                f"lm_head_weight width {lm_head_weight.shape[1]} != hidden_size={hidden_size}"
            )
        if not isinstance(prose_targets, torch.Tensor) or prose_targets.ndim != 1:
            raise ValueError("prose_targets must be a 1D integer tensor")

    layer_reports: dict[int, dict[str, Any]] = {}
    exp_acc: dict[int, dict[str, list[float]]] = {
        e: {
            "unf_gate": [],
            "frz_gate": [],
            "unf_topk": [],
            "frz_topk": [],
            "router_rms": [],
            "expert_rms": [],
            "unweighted_rms": [],
            "spec_gate_shift": [],
        }
        for e in range(num_experts)
    }

    kl_unf_vs_base_list: list[float] = []
    kl_frz_vs_base_list: list[float] = []
    kl_unf_vs_frz_list: list[float] = []
    cat_kl_unf_vs_base_list: list[float] = []
    logit_rms_list: list[float] = []
    flip_rate_list: list[float] = []
    router_e_list: list[float] = []
    expert_e_list: list[float] = []
    frz_anchor_e_list: list[float] = []
    cross_e_list: list[float] = []
    total_e_list: list[float] = []
    telescope_err_list: list[float] = []

    cum_y_base_base: torch.Tensor | None = None
    cum_y_frz_base: torch.Tensor | None = None
    cum_y_frz_unf: torch.Tensor | None = None
    cum_y_unf_unf: torch.Tensor | None = None

    for l_idx in layer_indices:
        x_p = prose_by_layer[l_idx]
        fine_scale = any(spec["kind"] == "swiglu" for spec in unf_e[l_idx].values())

        z_base = _compute_router_logits(x_p, base_r[l_idx])
        z_frz = _compute_router_logits(x_p, frz_r[l_idx])
        z_unf = _compute_router_logits(x_p, unf_r[l_idx])

        g_base, top_base, b_base, p_base = _compute_gates_and_probs(
            z_base, routing_mode=routing_mode, top_k=top_k_i, fine_grained_scale=fine_scale
        )
        g_frz, top_frz, b_frz, p_frz = _compute_gates_and_probs(
            z_frz, routing_mode=routing_mode, top_k=top_k_i, fine_grained_scale=fine_scale
        )
        g_unf, top_unf, b_unf, p_unf = _compute_gates_and_probs(
            z_unf, routing_mode=routing_mode, top_k=top_k_i, fine_grained_scale=fine_scale
        )

        kl_ub, cat_ub = _compute_routing_kl(p_unf, p_base, b_unf, b_base, routing_mode=routing_mode)
        kl_fb, _ = _compute_routing_kl(p_frz, p_base, b_frz, b_base, routing_mode=routing_mode)
        kl_uf, _ = _compute_routing_kl(p_unf, p_frz, b_unf, b_frz, routing_mode=routing_mode)
        logit_shift_rms = float((z_unf - z_base).pow(2).mean().sqrt().item())

        # Token-level top-k routing set flip rate between frozen/base and unfrozen routers
        mask_frz = torch.zeros_like(g_frz, dtype=torch.bool).scatter_(1, top_frz, True)
        mask_unf = torch.zeros_like(g_unf, dtype=torch.bool).scatter_(1, top_unf, True)
        flip_rate = float((mask_frz != mask_unf).any(dim=-1).to(torch.float64).mean().item())

        l_mlp = mlp_map.get(l_idx)
        dy_base = _eval_expert_mlp_outputs(
            x_p, base_e[l_idx], num_experts=num_experts, base_mlp_weights=l_mlp
        )
        dy_unf = _eval_expert_mlp_outputs(
            x_p, unf_e[l_idx], num_experts=num_experts, base_mlp_weights=l_mlp
        )

        y_bb = _combine_gated_outputs(g_base, dy_base, num_experts)
        y_fb = _combine_gated_outputs(g_frz, dy_base, num_experts)
        y_fu = _combine_gated_outputs(g_frz, dy_unf, num_experts)
        y_uu = _combine_gated_outputs(g_unf, dy_unf, num_experts)

        if cum_y_base_base is None:
            cum_y_base_base, cum_y_frz_base, cum_y_frz_unf, cum_y_unf_unf = y_bb, y_fb, y_fu, y_uu
        elif y_bb.shape[0] == cum_y_base_base.shape[0]:
            cum_y_base_base = cum_y_base_base + y_bb
            cum_y_frz_base = cum_y_frz_base + y_fb
            cum_y_frz_unf = cum_y_frz_unf + y_fu
            cum_y_unf_unf = cum_y_unf_unf + y_uu

        d_total = y_uu - y_bb
        d_router = y_uu - y_fu
        d_expert = y_fu - y_fb
        d_frz_anchor = y_fb - y_bb

        tele_err = float((d_total - (d_router + d_expert + d_frz_anchor)).abs().max().item())
        n_tok_f = float(x_p.shape[0])
        e_router = float(d_router.pow(2).sum().item() / n_tok_f)
        e_expert = float(d_expert.pow(2).sum().item() / n_tok_f)
        e_anchor = float(d_frz_anchor.pow(2).sum().item() / n_tok_f)
        e_total = float(d_total.pow(2).sum().item() / n_tok_f)
        e_cross = e_total - (e_router + e_expert + e_anchor)

        kl_unf_vs_base_list.append(kl_ub)
        kl_frz_vs_base_list.append(kl_fb)
        kl_unf_vs_frz_list.append(kl_uf)
        cat_kl_unf_vs_base_list.append(cat_ub)
        logit_rms_list.append(logit_shift_rms)
        flip_rate_list.append(flip_rate)
        router_e_list.append(e_router)
        expert_e_list.append(e_expert)
        frz_anchor_e_list.append(e_anchor)
        cross_e_list.append(e_cross)
        total_e_list.append(e_total)
        telescope_err_list.append(tele_err)

        spec_g_diff: torch.Tensor | None = None
        if spec_by_layer is not None:
            x_s = spec_by_layer[l_idx]
            gs_frz, _, _, _ = _compute_gates_and_probs(
                _compute_router_logits(x_s, frz_r[l_idx]),
                routing_mode=routing_mode,
                top_k=top_k_i,
                fine_grained_scale=fine_scale,
            )
            gs_unf, _, _, _ = _compute_gates_and_probs(
                _compute_router_logits(x_s, unf_r[l_idx]),
                routing_mode=routing_mode,
                top_k=top_k_i,
                fine_grained_scale=fine_scale,
            )
            spec_g_diff = (gs_unf - gs_frz).mean(dim=0)

        for e_idx in range(num_experts):
            dg_e = (g_unf[:, e_idx : e_idx + 1] - g_frz[:, e_idx : e_idx + 1])
            exp_delta_e = dy_unf[e_idx] - dy_base[e_idx]
            exp_acc[e_idx]["unf_gate"].append(float(g_unf[:, e_idx].mean().item()))
            exp_acc[e_idx]["frz_gate"].append(float(g_frz[:, e_idx].mean().item()))
            exp_acc[e_idx]["unf_topk"].append(float(mask_unf[:, e_idx].to(torch.float64).mean().item()))
            exp_acc[e_idx]["frz_topk"].append(float(mask_frz[:, e_idx].to(torch.float64).mean().item()))
            exp_acc[e_idx]["router_rms"].append(
                float((dg_e * dy_unf[e_idx]).pow(2).sum(dim=-1).mean().sqrt().item())
            )
            exp_acc[e_idx]["expert_rms"].append(
                float((g_frz[:, e_idx : e_idx + 1] * exp_delta_e).pow(2).sum(dim=-1).mean().sqrt().item())
            )
            exp_acc[e_idx]["unweighted_rms"].append(
                float(exp_delta_e.pow(2).sum(dim=-1).mean().sqrt().item())
            )
            if spec_g_diff is not None:
                exp_acc[e_idx]["spec_gate_shift"].append(float(spec_g_diff[e_idx].item()))

        layer_reports[l_idx] = {
            "router_kl_drift": kl_ub,
            "frozen_vs_base_router_kl_drift": kl_fb,
            "unfrozen_vs_frozen_router_kl_drift": kl_uf,
            "normalized_categorical_kl_drift": cat_ub,
            "router_logit_shift_rms": logit_shift_rms,
            "top_k_routing_flip_rate": flip_rate,
            "router_drift_energy": e_router,
            "expert_drift_energy": e_expert,
            "frozen_router_drift_energy": e_anchor,
            "cross_interaction_energy": e_cross,
            "total_prose_drift_energy": e_total,
            "telescoping_max_abs_error": tele_err,
        }

    mean_router_e = float(np.mean(router_e_list))
    mean_expert_e = float(np.mean(expert_e_list))
    mean_anchor_e = float(np.mean(frz_anchor_e_list))
    denom_e = mean_router_e + mean_expert_e + mean_anchor_e
    if denom_e > 1e-18:
        router_share = mean_router_e / denom_e
        expert_share = mean_expert_e / denom_e
        anchor_share = mean_anchor_e / denom_e
        dominant = "router_gate_shift" if mean_router_e >= mean_expert_e else "expert_mlp_delta"
    else:
        router_share, expert_share, anchor_share, dominant = 0.0, 0.0, 0.0, "none"

    per_expert_attr: dict[str, dict[str, float]] = {}
    for e_idx in range(num_experts):
        acc_e = exp_acc[e_idx]
        entry_e: dict[str, float] = {
            "unfrozen_prose_gate_mean": float(np.mean(acc_e["unf_gate"])),
            "frozen_prose_gate_mean": float(np.mean(acc_e["frz_gate"])),
            "gate_shift_on_prose": float(np.mean(acc_e["unf_gate"]) - np.mean(acc_e["frz_gate"])),
            "unfrozen_prose_top_k_rate": float(np.mean(acc_e["unf_topk"])),
            "frozen_prose_top_k_rate": float(np.mean(acc_e["frz_topk"])),
            "router_attributed_prose_rms": float(np.mean(acc_e["router_rms"])),
            "expert_under_frozen_prose_rms": float(np.mean(acc_e["expert_rms"])),
            "unweighted_expert_prose_rms": float(np.mean(acc_e["unweighted_rms"])),
        }
        if acc_e["spec_gate_shift"]:
            entry_e["gate_shift_on_spec"] = float(np.mean(acc_e["spec_gate_shift"]))
        per_expert_attr[f"expert_{e_idx}"] = entry_e

    nll_attribution: dict[str, float] | None = None
    if prose_nll_eval_fn is not None:
        if not callable(prose_nll_eval_fn):
            raise TypeError("prose_nll_eval_fn must be callable")
        nll_bb = _validate_finite_float(
            prose_nll_eval_fn({"routers": base_r, "experts": base_e}), "nll(base, base)"
        )
        nll_fb = _validate_finite_float(
            prose_nll_eval_fn({"routers": frz_r, "experts": base_e}), "nll(frozen, base)"
        )
        nll_fu = _validate_finite_float(
            prose_nll_eval_fn({"routers": frz_r, "experts": unf_e}), "nll(frozen, unfrozen)"
        )
        nll_uu = _validate_finite_float(
            prose_nll_eval_fn({"routers": unf_r, "experts": unf_e}), "nll(unfrozen, unfrozen)"
        )
        nll_attribution = {
            "base_prose_nll": nll_bb,
            "frozen_router_base_expert_prose_nll": nll_fb,
            "frozen_router_unfrozen_expert_prose_nll": nll_fu,
            "unfrozen_arm_prose_nll": nll_uu,
            "router_induced_nll_drift_nats": nll_uu - nll_fu,
            "expert_induced_nll_drift_nats": nll_fu - nll_fb,
            "frozen_anchor_nll_drift_nats": nll_fb - nll_bb,
            "total_prose_nll_drift_nats": nll_uu - nll_bb,
        }
    elif lm_head_weight is not None and prose_targets is not None:
        h_ref = prose_by_layer[layer_indices[-1]]
        if prose_targets.shape[0] != h_ref.shape[0]:
            raise ValueError(
                f"prose_targets length {prose_targets.shape[0]} != hidden rows {h_ref.shape[0]}"
            )
        assert cum_y_base_base is not None and cum_y_frz_base is not None
        assert cum_y_frz_unf is not None and cum_y_unf_unf is not None
        nll_bb = _eval_nll_from_hidden_and_perturbation(
            h_ref, cum_y_base_base, lm_head_weight, prose_targets, final_ln_params
        )
        nll_fb = _eval_nll_from_hidden_and_perturbation(
            h_ref, cum_y_frz_base, lm_head_weight, prose_targets, final_ln_params
        )
        nll_fu = _eval_nll_from_hidden_and_perturbation(
            h_ref, cum_y_frz_unf, lm_head_weight, prose_targets, final_ln_params
        )
        nll_uu = _eval_nll_from_hidden_and_perturbation(
            h_ref, cum_y_unf_unf, lm_head_weight, prose_targets, final_ln_params
        )
        nll_attribution = {
            "base_prose_nll": nll_bb,
            "frozen_router_base_expert_prose_nll": nll_fb,
            "frozen_router_unfrozen_expert_prose_nll": nll_fu,
            "unfrozen_arm_prose_nll": nll_uu,
            "router_induced_nll_drift_nats": nll_uu - nll_fu,
            "expert_induced_nll_drift_nats": nll_fu - nll_fb,
            "frozen_anchor_nll_drift_nats": nll_fb - nll_bb,
            "total_prose_nll_drift_nats": nll_uu - nll_bb,
        }

    return {
        "schema": SCHEMA_VERSION,
        "routing_mode": routing_mode,
        "top_k": top_k_i,
        "num_layers": len(layer_indices),
        "num_experts": num_experts,
        "router_kl_drift": float(np.mean(kl_unf_vs_base_list)),
        "frozen_vs_base_router_kl_drift": float(np.mean(kl_frz_vs_base_list)),
        "unfrozen_vs_frozen_router_kl_drift": float(np.mean(kl_unf_vs_frz_list)),
        "normalized_categorical_kl_drift": float(np.mean(cat_kl_unf_vs_base_list)),
        "router_logit_shift_rms": float(np.mean(logit_rms_list)),
        "top_k_routing_flip_rate": float(np.mean(flip_rate_list)),
        "router_drift_energy": mean_router_e,
        "expert_drift_energy": mean_expert_e,
        "frozen_router_drift_energy": mean_anchor_e,
        "cross_interaction_energy": float(np.mean(cross_e_list)),
        "total_prose_drift_energy": float(np.mean(total_e_list)),
        "router_drift_share": router_share,
        "expert_drift_share": expert_share,
        "frozen_anchor_drift_share": anchor_share,
        "dominant_prose_drift_source": dominant,
        "telescoping_max_abs_error": float(max(telescope_err_list)),
        "nll_attribution": nll_attribution,
        "per_expert_attribution": per_expert_attr,
        "per_layer": layer_reports,
    }


# ---------------------------------------------------------------------------
# 4. Component (b): P_0-Projected Expert & OOF Selective Router Grafting
# ---------------------------------------------------------------------------


def _binarize_projector(p_mat: torch.Tensor, eig_threshold: float) -> torch.Tensor:
    """Return exact symmetric idempotent projector ``P_0`` matching ``eig_threshold``."""
    idem_err = float((p_mat @ p_mat - p_mat).abs().max().item())
    if idem_err <= 1e-6:
        return p_mat
    evals, evecs = torch.linalg.eigh(p_mat)
    keep = evals > eig_threshold
    if not bool(keep.any().item()):
        return torch.zeros_like(p_mat)
    q_keep = evecs[:, keep]
    proj = q_keep @ q_keep.T
    return (0.5 * (proj + proj.T)).contiguous()


def _resolve_expert_nullspace_projectors(
    p_null_by_expert: Any,
    layer_idx: int,
    expert_idx: int,
    *,
    hidden_size: int,
    intermediate_size: int | None,
    eig_threshold: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve ``(P_x, P_act)`` projectors for ``(layer_idx, expert_idx)``."""
    if p_null_by_expert is None:
        return None, None
    if isinstance(p_null_by_expert, torch.Tensor):
        p_t = _binarize_projector(
            _validate_square_projector(p_null_by_expert, "p_null_by_expert"), eig_threshold
        )
        dim = int(p_t.shape[0])
        if dim == hidden_size:
            return p_t, None
        if intermediate_size is not None and dim == intermediate_size:
            return None, p_t
        raise ValueError(
            f"Projector shape {tuple(p_t.shape)} matches neither hidden_size={hidden_size} "
            f"nor intermediate_size={intermediate_size}"
        )
    if not isinstance(p_null_by_expert, Mapping):
        raise TypeError("p_null_by_expert must be a 2D Tensor, Mapping, or None")
    if not p_null_by_expert:
        return None, None

    px_out: torch.Tensor | None = None
    pact_out: torch.Tensor | None = None

    def _consume(entry: Any, label: str) -> None:
        nonlocal px_out, pact_out
        if isinstance(entry, torch.Tensor):
            p_m = _binarize_projector(_validate_square_projector(entry, label), eig_threshold)
            dim = int(p_m.shape[0])
            if dim == hidden_size and px_out is None:
                px_out = p_m
            elif intermediate_size is not None and dim == intermediate_size and pact_out is None:
                pact_out = p_m
        elif isinstance(entry, Mapping):
            if "x" in entry and isinstance(entry["x"], torch.Tensor) and px_out is None:
                px_out = _binarize_projector(
                    _validate_square_projector(entry["x"], f"{label}['x']", expected_dim=hidden_size),
                    eig_threshold,
                )
            if (
                "act" in entry
                and isinstance(entry["act"], torch.Tensor)
                and intermediate_size is not None
                and pact_out is None
            ):
                pact_out = _binarize_projector(
                    _validate_square_projector(
                        entry["act"], f"{label}['act']", expected_dim=intermediate_size
                    ),
                    eig_threshold,
                )
            for sub_k in (expert_idx, str(expert_idx), f"expert_{expert_idx}", "default"):
                if sub_k in entry:
                    _consume(entry[sub_k], f"{label}[{sub_k!r}]")

    candidates: list[Any] = [
        (layer_idx, expert_idx),
        f"blocks.{layer_idx}.mlp.experts.{expert_idx}",
        f"blocks.{layer_idx}",
        layer_idx,
        str(layer_idx),
        expert_idx,
        str(expert_idx),
        f"expert_{expert_idx}",
        "default",
        "all",
    ]
    for cand in candidates:
        if cand in p_null_by_expert:
            _consume(p_null_by_expert[cand], f"p_null[{cand!r}]")
    if "x" in p_null_by_expert and px_out is None:
        _consume({"x": p_null_by_expert["x"]}, "p_null")
    if "act" in p_null_by_expert and pact_out is None:
        _consume({"act": p_null_by_expert["act"]}, "p_null")
    return px_out, pact_out


def _resolve_router_nullspace_projector(
    p_null_by_expert: Any,
    layer_idx: int,
    expert_idx: int,
    hidden_size: int,
    eig_threshold: float,
    auto_prose_p: torch.Tensor | None,
) -> torch.Tensor:
    """Resolve ``(H, H)`` prose null-space projector for router row ``(layer_idx, expert_idx)``."""
    if isinstance(p_null_by_expert, Mapping):
        for cand in (
            f"blocks.{layer_idx}.mlp.router.{expert_idx}",
            f"blocks.{layer_idx}.mlp.router",
            "router",
            f"blocks.{layer_idx}.router",
        ):
            if cand in p_null_by_expert and isinstance(p_null_by_expert[cand], torch.Tensor):
                return _binarize_projector(
                    _validate_square_projector(
                        p_null_by_expert[cand], f"p_null[{cand!r}]", expected_dim=hidden_size
                    ),
                    eig_threshold,
                )
    px, _ = _resolve_expert_nullspace_projectors(
        p_null_by_expert,
        layer_idx,
        expert_idx,
        hidden_size=hidden_size,
        intermediate_size=None,
        eig_threshold=eig_threshold,
    )
    if px is not None:
        return px
    if auto_prose_p is not None:
        return auto_prose_p
    return torch.eye(hidden_size, dtype=torch.float64)


def _project_lora_matrices_into_nullspace(
    b_base: torch.Tensor,
    a_base: torch.Tensor,
    b_unf: torch.Tensor,
    a_unf: torch.Tensor,
    p_mat: torch.Tensor,
    side: str,
    leak_tol: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float, bool]:
    """Project LoRA delta ``B_1 A_1 - B_0 A_0`` into ``P_0`` only when leakage exceeds ``leak_tol``."""
    b0 = b_base.detach().to("cpu", dtype=torch.float64)
    a0 = a_base.detach().to("cpu", dtype=torch.float64)
    b1 = b_unf.detach().to("cpu", dtype=torch.float64)
    a1 = a_unf.detach().to("cpu", dtype=torch.float64)
    q_perp = torch.eye(int(p_mat.shape[0]), dtype=torch.float64) - p_mat
    zero_base = int(torch.count_nonzero(b0).item()) == 0

    if zero_base:
        leak_before = float(
            torch.linalg.matrix_norm(b1 @ (a1 @ q_perp)).item()
            if side == "input"
            else torch.linalg.matrix_norm((q_perp @ b1) @ a1).item()
        )
    else:
        d_b_pre = torch.cat([b1, -b0], dim=1)
        d_a_pre = torch.cat([a1, a0], dim=0)
        if side == "input":
            _, s_pre, _ = decompose_lora_qr_svd(d_b_pre, d_a_pre @ q_perp)
        else:
            _, s_pre, _ = decompose_lora_qr_svd(q_perp @ d_b_pre, d_a_pre)
        leak_before = float(s_pre.pow(2).sum().sqrt().item())

    if leak_before <= leak_tol:
        return b_unf.detach().clone(), a_unf.detach().clone(), leak_before, leak_before, False

    if zero_base:
        if side == "input":
            b_new = b1.contiguous()
            a_new = (a1 @ p_mat).contiguous()
            leak_after = float(torch.linalg.matrix_norm(b_new @ (a_new @ q_perp)).item())
        else:
            b_new = (p_mat @ b1).contiguous()
            a_new = a1.contiguous()
            leak_after = float(torch.linalg.matrix_norm((q_perp @ b_new) @ a_new).item())
        return (
            b_new.to(device=b_unf.device, dtype=b_unf.dtype),
            a_new.to(device=a_unf.device, dtype=a_unf.dtype),
            leak_before,
            leak_after,
            True,
        )

    if side == "input":
        base_perp = float(torch.linalg.matrix_norm(b0 @ (a0 @ q_perp)).item())
        if base_perp <= 1e-10:
            b_new = b1.contiguous()
            a_new = (a1 @ p_mat).contiguous()
        else:
            b_new = b0.contiguous()
            rhs = b1 @ (a1 @ p_mat)
            a_proj = torch.linalg.lstsq(b0, rhs).solution
            a_new = (a0 @ q_perp + a_proj @ p_mat).contiguous()
        d_b = torch.cat([b_new, -b0], dim=1)
        d_a = torch.cat([a_new @ q_perp, a0 @ q_perp], dim=0)
    else:
        base_perp = float(torch.linalg.matrix_norm((q_perp @ b0) @ a0).item())
        if base_perp <= 1e-10:
            b_new = (p_mat @ b1).contiguous()
            a_new = a1.contiguous()
        else:
            a_new = a0.contiguous()
            rhs_t = (p_mat @ b1 @ a1).T
            b_proj_t = torch.linalg.lstsq(a0.T, rhs_t).solution
            b_new = (q_perp @ b0 + p_mat @ b_proj_t.T).contiguous()
        d_b = torch.cat([q_perp @ b_new, -(q_perp @ b0)], dim=1)
        d_a = torch.cat([a_new, a0], dim=0)

    _, s_post, _ = decompose_lora_qr_svd(d_b, d_a)
    leak_after = float(s_post.pow(2).sum().sqrt().item())
    return (
        b_new.to(device=b_unf.device, dtype=b_unf.dtype),
        a_new.to(device=a_unf.device, dtype=a_unf.dtype),
        leak_before,
        leak_after,
        True,
    )


def _project_dense_matrix_into_nullspace(
    w_base: torch.Tensor,
    w_unf: torch.Tensor,
    p_mat: torch.Tensor,
    side: str,
    leak_tol: float,
) -> tuple[torch.Tensor, float, float, bool]:
    """Project dense tensor delta ``w_unf - w_base`` into ``P_0`` only when needed."""
    b64 = w_base.detach().to("cpu", dtype=torch.float64)
    u64 = w_unf.detach().to("cpu", dtype=torch.float64)
    delta = u64 - b64
    q_perp = torch.eye(int(p_mat.shape[0]), dtype=torch.float64) - p_mat
    if side == "input":
        leak_before = float(torch.linalg.matrix_norm(delta @ q_perp).item())
    else:
        leak_before = float(torch.linalg.matrix_norm(q_perp @ delta).item())
    if leak_before <= leak_tol:
        return w_unf.detach().clone(), leak_before, leak_before, False
    proj_delta = (delta @ p_mat) if side == "input" else (p_mat @ delta)
    leak_after = float(
        torch.linalg.matrix_norm((proj_delta @ q_perp) if side == "input" else (q_perp @ proj_delta)).item()
    )
    out = (b64 + proj_delta).to(device=w_unf.device, dtype=w_unf.dtype).contiguous()
    return out, leak_before, leak_after, True


def _verify_router_delta_oof(
    delta_r: torch.Tensor,
    x_prose: torch.Tensor,
    x_spec: torch.Tensor,
    *,
    p_router: torch.Tensor,
    n_folds: int,
    min_oof_snr: float,
    max_oof_prose_leak_ratio: float,
    sv_rel_tol: float,
    variance_threshold: float,
) -> dict[str, Any]:
    """Verify out-of-fold specialist selectivity of projected router delta ``P_0 @ delta_r``."""
    if int(x_prose.shape[0]) < n_folds or int(x_spec.shape[0]) < n_folds:
        raise ValueError(
            f"OOF validation requires >= n_folds={n_folds} prose and specialist tokens, "
            f"got {x_prose.shape[0]} and {x_spec.shape[0]}"
        )
    p_oof_parts: list[torch.Tensor] = []
    s_oof_parts: list[torch.Tensor] = []
    fold_p_means: list[float] = []
    fold_s_means: list[float] = []

    n_p = int(x_prose.shape[0])
    n_s = int(x_spec.shape[0])
    for fold_idx in range(n_folds):
        p_val_ix = [i for i in range(n_p) if i % n_folds == fold_idx]
        p_tr_ix = [i for i in range(n_p) if i % n_folds != fold_idx]
        s_val_ix = [i for i in range(n_s) if i % n_folds == fold_idx]

        p_fold, _ = compute_prose_nullspace_projector_scaled(
            x_prose[p_tr_ix],
            sv_rel_tol=sv_rel_tol,
            variance_threshold=variance_threshold,
        )
        dr_fold = p_router @ (p_fold @ delta_r)
        p_scores = x_prose[p_val_ix] @ dr_fold
        s_scores = x_spec[s_val_ix] @ dr_fold
        p_oof_parts.append(p_scores)
        s_oof_parts.append(s_scores)
        fold_p_means.append(float(p_scores.abs().mean().item()))
        fold_s_means.append(float(s_scores.mean().item()))

    p_oof = torch.cat(p_oof_parts, dim=0)
    s_oof = torch.cat(s_oof_parts, dim=0)
    s_mean = float(s_oof.mean().item())
    p_mean_abs = float(p_oof.abs().mean().item())
    p_rms = float(p_oof.pow(2).mean().sqrt().item())
    pooled_std = math.sqrt(
        0.5 * (float(s_oof.var(unbiased=False).item()) + float(p_oof.var(unbiased=False).item()))
        + 1e-8
    )
    oof_snr = max(0.0, (s_mean - p_mean_abs) / pooled_std)
    folds_agree = all(fs > 0.0 and fs > fp for fs, fp in zip(fold_s_means, fold_p_means, strict=True))
    leak_ok = p_rms <= max_oof_prose_leak_ratio * max(s_mean, 1e-12)
    verified = bool(folds_agree and leak_ok and (oof_snr > min_oof_snr))
    rel_weight = (
        max(0.0, min(1.0, (oof_snr - min_oof_snr) / max(1.0 - min_oof_snr, 0.25)))
        if verified
        else 0.0
    )
    return {
        "verified": verified,
        "rel_weight": rel_weight,
        "oof_spec_mean": s_mean,
        "oof_prose_abs_mean": p_mean_abs,
        "oof_prose_rms": p_rms,
        "oof_snr": oof_snr,
        "folds_agree": folds_agree,
        "leak_ok": leak_ok,
    }


def _repackage_grafted_expert(
    orig_spec: Mapping[str, Any],
    new_tensors: Mapping[str, torch.Tensor],
) -> Any:
    """Package grafted expert tensors back into the caller's module or dict style."""
    style = orig_spec["style"]
    if style == "lora_module":
        cloned = copy.deepcopy(orig_spec["source"])
        with torch.no_grad():
            for k, t in new_tensors.items():
                param = getattr(cloned, k)
                param.copy_(t.to(device=param.device, dtype=param.dtype))
        return cloned
    if style == "swiglu_module":
        cloned = copy.deepcopy(orig_spec["source"])
        with torch.no_grad():
            for p_name in _SWIGLU_PROJECTIONS:
                lin = getattr(cloned, p_name)
                lin.weight.copy_(new_tensors[p_name].to(device=lin.weight.device, dtype=lin.weight.dtype))
        return cloned
    if style == "tensor":
        return new_tensors["delta"].detach().clone()
    out_dict = {k: v.detach().clone() for k, v in new_tensors.items()}
    if orig_spec["kind"] == "lora":
        out_dict["scaling"] = float(orig_spec["scaling"])
    return out_dict


def _package_grafted_outputs(
    grafted_routers: Mapping[int, Mapping[str, torch.Tensor]],
    grafted_experts: Mapping[int, Mapping[int, Any]],
    router_meta: Mapping[str, Any],
    expert_meta: Mapping[str, Any],
    base_blocks: Mapping[str, Mapping[str, torch.Tensor]] | None,
) -> dict[str, Any]:
    """Package grafted routers & experts matching caller structures plus optional block layout."""
    r_style = router_meta["style"]
    if r_style == "single_tensor":
        r_out: Any = grafted_routers[0]["weight"].detach().clone()
    elif r_style == "single_mapping":
        r_out = {k: v.detach().clone() for k, v in grafted_routers[0].items()}
    elif r_style == "layer_sequence":
        forms = router_meta["entry_forms"]
        r_out = [
            grafted_routers[l_idx]["weight"].detach().clone()
            if forms[l_idx] == "tensor"
            else {k: v.detach().clone() for k, v in grafted_routers[l_idx].items()}
            for l_idx in sorted(grafted_routers.keys())
        ]
    elif r_style == "layer_mapping":
        orig_keys = router_meta["layer_keys"]
        forms = router_meta["entry_forms"]
        r_out = {
            orig_keys[l_idx]: (
                grafted_routers[l_idx]["weight"].detach().clone()
                if forms[l_idx] == "tensor"
                else {k: v.detach().clone() for k, v in grafted_routers[l_idx].items()}
            )
            for l_idx in sorted(grafted_routers.keys())
        }
    else:
        r_out = {
            l_idx: {k: v.detach().clone() for k, v in m.items()}
            for l_idx, m in sorted(grafted_routers.items())
        }

    e_style = expert_meta["style"]
    if e_style == "single_layer_map":
        e_keys = expert_meta["expert_keys"]
        e_out: Any = {e_keys[e]: grafted_experts[0][e] for e in sorted(grafted_experts[0].keys())}
    elif e_style == "nested_layer_map":
        l_keys = expert_meta["layer_keys"]
        e_keys = expert_meta["expert_keys"]
        e_out = {
            l_keys[l_idx]: {
                e_keys[(l_idx, e)]: grafted_experts[l_idx][e]
                for e in sorted(grafted_experts[l_idx].keys())
            }
            for l_idx in sorted(grafted_experts.keys())
        }
    elif e_style == "layer_sequence":
        e_keys = expert_meta["expert_keys"]
        e_out = [
            {
                e_keys[(l_idx, e)]: grafted_experts[l_idx][e]
                for e in sorted(grafted_experts[l_idx].keys())
            }
            for l_idx in sorted(grafted_experts.keys())
        ]
    elif e_style == "checkpoint_blocks":
        e_keys = expert_meta["expert_keys"]
        e_out = {
            e_keys[(l_idx, e)]: grafted_experts[l_idx][e]
            for l_idx in sorted(grafted_experts.keys())
            for e in sorted(grafted_experts[l_idx].keys())
        }
    else:
        e_out = {l_idx: dict(m) for l_idx, m in sorted(grafted_experts.items())}

    blocks_out: dict[str, dict[str, torch.Tensor]] = {}
    if base_blocks is not None:
        blocks_out = {bid: {k: t.detach().clone() for k, t in sub.items()} for bid, sub in base_blocks.items()}
    for l_idx in sorted(grafted_routers.keys()):
        blocks_out[f"blocks.{l_idx}.mlp.router"] = {
            k: v.detach().clone() for k, v in grafted_routers[l_idx].items()
        }
        for e_idx, exp_obj in sorted(grafted_experts[l_idx].items()):
            if isinstance(exp_obj, nn.Module):
                if all(hasattr(exp_obj, f"{p}_{s}") for p in _SWIGLU_PROJECTIONS for s in ("A", "B")):
                    sub_t = {
                        f"{p}_{s}": getattr(exp_obj, f"{p}_{s}").detach().clone()
                        for p in _SWIGLU_PROJECTIONS
                        for s in ("A", "B")
                    }
                else:
                    sub_t = {
                        p: getattr(exp_obj, p).weight.detach().clone()
                        for p in _SWIGLU_PROJECTIONS
                    }
            elif isinstance(exp_obj, Mapping):
                sub_t = {k: v.detach().clone() for k, v in exp_obj.items() if isinstance(v, torch.Tensor)}
            else:
                sub_t = {"delta": exp_obj.detach().clone()}
            blocks_out[f"blocks.{l_idx}.mlp.experts.{e_idx}"] = sub_t

    return {"routers": r_out, "experts": e_out, "blocks": blocks_out}


@torch.no_grad()
def apply_grafted_state_inplace(
    model: nn.Module,
    grafted_state: Mapping[str, Any],
) -> None:
    """Apply ``grafted_state['blocks']`` onto an MoE ``nn.Module`` in-place."""
    if not isinstance(grafted_state, Mapping) or not isinstance(grafted_state.get("blocks"), Mapping):
        raise TypeError("grafted_state must be the dict returned by graft_specialist_experts_onto_safe_router")
    apply_checkpoint_blocks_inplace(model, grafted_state["blocks"])


@torch.no_grad()
def graft_specialist_experts_onto_safe_router(
    frozen_arm_routers: Any,
    unfrozen_arm_routers: Any,
    unfrozen_experts: Any,
    p_null_by_expert: torch.Tensor | Mapping[Any, Any] | None,
    *,
    gate_blend_alpha: float = DEFAULT_GATE_BLEND_ALPHA,
    sink_experts: Sequence[int] | set[int] = (),
    base_experts: Any = None,
    prose_hidden_states: torch.Tensor | Mapping[Any, torch.Tensor] | Sequence[torch.Tensor] | None = None,
    spec_hidden_states: torch.Tensor | Mapping[Any, torch.Tensor] | Sequence[torch.Tensor] | None = None,
    specialist_domain_hidden_states: Mapping[Any, Any] | None = None,
    protected_experts: Sequence[int] | set[int] = (),
    nullspace_sv_rel_tol: float = DEFAULT_NULLSPACE_SV_REL_TOL,
    nullspace_variance_threshold: float = DEFAULT_NULLSPACE_VARIANCE_THRESHOLD,
    nullspace_eig_threshold: float = DEFAULT_NULLSPACE_EIG_THRESHOLD,
    nullspace_leak_tol: float = DEFAULT_NULLSPACE_LEAK_TOL,
    min_oof_snr: float = DEFAULT_MIN_OOF_SNR,
    n_folds: int = DEFAULT_N_FOLDS,
    max_oof_prose_leak_ratio: float = DEFAULT_MAX_OOF_PROSE_LEAK_RATIO,
    min_nullspace_alignment: float = DEFAULT_MIN_NULLSPACE_ALIGNMENT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Graft deep-specialist unfrozen-arm experts onto prose-safe frozen-arm router gates.

    1. Pure prose-corrupting ``sink_experts`` are pruned bitwise (LoRA ``w1_B = w2_B = w3_B = 0``
       or reverted to ``base_experts``) and their router rows remain anchored at ``frozen_arm_routers``.
    2. Surviving non-protected specialist experts are projected into ``p_null_by_expert`` (``P_0``)
       whenever their null-space leakage exceeds ``nullspace_leak_tol``.
    3. Router rows receive ``gate_blend_alpha * rel_e * P_0 (W_unf[e] - W_frz[e])`` strictly inside
       ``P_0`` and only when OOF specialist verification (or non-prose alignment when probes are omitted)
       confirms selective gating on specialist tokens without shifting prose logits.
    """
    alpha = _validate_unit_interval(gate_blend_alpha, "gate_blend_alpha", allow_zero=True)
    sink_set = _validate_expert_index_set(sink_experts, "sink_experts")
    prot_set = _validate_expert_index_set(protected_experts, "protected_experts")
    overlap = sink_set & prot_set
    if overlap:
        raise ValueError(f"Experts cannot be both sink_experts and protected_experts: {sorted(overlap)}")

    sv_tol = _validate_nonneg_finite(nullspace_sv_rel_tol, "nullspace_sv_rel_tol")
    var_thr = _validate_unit_interval(
        nullspace_variance_threshold, "nullspace_variance_threshold", allow_zero=False
    )
    eig_thr = _validate_unit_interval(nullspace_eig_threshold, "nullspace_eig_threshold", allow_zero=True)
    if eig_thr >= 1.0:
        raise ValueError(f"nullspace_eig_threshold must be < 1.0, got {eig_thr}")
    leak_tol = _validate_nonneg_finite(nullspace_leak_tol, "nullspace_leak_tol")
    min_snr = _validate_nonneg_finite(min_oof_snr, "min_oof_snr")
    folds_i = _validate_positive_int(n_folds, "n_folds")
    if folds_i < 2:
        raise ValueError(f"n_folds must be >= 2, got {folds_i}")
    max_leak = _validate_positive_finite(max_oof_prose_leak_ratio, "max_oof_prose_leak_ratio")
    min_align = _validate_unit_interval(min_nullspace_alignment, "min_nullspace_alignment", allow_zero=True)

    frz_r, frz_meta = _normalize_routers_by_layer(frozen_arm_routers, "frozen_arm_routers")
    unf_r, _ = _normalize_routers_by_layer(unfrozen_arm_routers, "unfrozen_arm_routers")
    layer_indices, num_experts, hidden_size = _validate_router_layout_match(
        frz_r, unf_r, "frozen_arm_routers", "unfrozen_arm_routers"
    )
    unf_e, unf_meta = _normalize_experts_by_layer(
        unfrozen_experts,
        "unfrozen_experts",
        num_layers=len(layer_indices),
        num_experts=num_experts,
        hidden_size=hidden_size,
    )
    if base_experts is not None:
        base_e, _ = _normalize_experts_by_layer(
            base_experts,
            "base_experts",
            num_layers=len(layer_indices),
            num_experts=num_experts,
            hidden_size=hidden_size,
        )
    else:
        base_e = _build_zero_base_experts(unf_e)

    prose_by_layer = (
        _resolve_hidden_states_by_layer(
            prose_hidden_states, layer_indices, hidden_size, "prose_hidden_states"
        )
        if prose_hidden_states is not None
        else None
    )
    spec_by_layer = (
        _resolve_hidden_states_by_layer(
            spec_hidden_states, layer_indices, hidden_size, "spec_hidden_states"
        )
        if spec_hidden_states is not None
        else None
    )
    if (prose_by_layer is None) != (spec_by_layer is None) and specialist_domain_hidden_states is None:
        raise ValueError(
            "prose_hidden_states and spec_hidden_states must both be provided for OOF router verification"
        )

    grafted_routers: dict[int, dict[str, torch.Tensor]] = {}
    grafted_experts: dict[int, dict[int, Any]] = {}
    layer_reports: dict[int, dict[str, Any]] = {}
    max_expert_leak = 0.0
    max_router_leak = 0.0
    projected_experts_set: set[int] = set()
    oof_verified_set: set[int] = set()

    for l_idx in layer_indices:
        auto_p_prose: torch.Tensor | None = None
        if prose_by_layer is not None:
            auto_p_prose, _ = compute_prose_nullspace_projector_scaled(
                prose_by_layer[l_idx],
                sv_rel_tol=sv_tol,
                variance_threshold=var_thr,
            )

        w_frz = frz_r[l_idx]["weight"].detach().to("cpu", dtype=torch.float64)
        w_unf = unf_r[l_idx]["weight"].detach().to("cpu", dtype=torch.float64)
        w_graft = w_frz.clone()

        grafted_experts[l_idx] = {}
        expert_actions: dict[int, str] = {}
        router_diag: dict[int, dict[str, Any]] = {}

        for e_idx in sorted(unf_e[l_idx].keys()):
            spec_u = unf_e[l_idx][e_idx]
            spec_b = base_e[l_idx][e_idx]
            t_u = spec_u["tensors"]
            t_b = spec_b["tensors"]
            kind = spec_u["kind"]

            # 1. Expert grafting: sink pruning, bitwise protection, or P_0 null-space projection
            if e_idx in sink_set:
                if kind == "lora":
                    new_t = {
                        k: (torch.zeros_like(v) if k.endswith("_B") else v.detach().clone())
                        for k, v in t_u.items()
                    }
                else:
                    new_t = {k: v.detach().clone() for k, v in t_b.items()}
                expert_actions[e_idx] = "ghost_sink_pruned"
            elif e_idx in prot_set:
                new_t = {k: v.detach().clone() for k, v in t_u.items()}
                expert_actions[e_idx] = "preserved_anchor_bitwise"
            else:
                inter_dim = (
                    int(t_u["w3_A"].shape[1])
                    if kind == "lora"
                    else (int(t_u["w3"].shape[1]) if kind == "swiglu" else None)
                )
                px, pact = _resolve_expert_nullspace_projectors(
                    p_null_by_expert,
                    l_idx,
                    e_idx,
                    hidden_size=hidden_size,
                    intermediate_size=inter_dim,
                    eig_threshold=eig_thr,
                )
                if px is None and pact is None:
                    new_t = {k: v.detach().clone() for k, v in t_u.items()}
                    expert_actions[e_idx] = "grafted_unprojected"
                else:
                    new_t = {}
                    did_proj = False
                    if kind == "lora":
                        for proj in _SWIGLU_PROJECTIONS:
                            ak, bk = f"{proj}_A", f"{proj}_B"
                            if proj in ("w1", "w2"):
                                if px is not None:
                                    b_n, a_n, _, lk_a, chg = _project_lora_matrices_into_nullspace(
                                        t_b[bk], t_b[ak], t_u[bk], t_u[ak], px, "input", leak_tol
                                    )
                                    did_proj = did_proj or chg
                                    max_expert_leak = max(max_expert_leak, lk_a)
                                    new_t[bk], new_t[ak] = b_n, a_n
                                else:
                                    new_t[bk], new_t[ak] = t_u[bk].clone(), t_u[ak].clone()
                            else:
                                p_w3, w3_side = (
                                    (pact, "input") if pact is not None else (px, "output")
                                )
                                assert p_w3 is not None
                                b_n, a_n, _, lk_a, chg = _project_lora_matrices_into_nullspace(
                                    t_b[bk], t_b[ak], t_u[bk], t_u[ak], p_w3, w3_side, leak_tol
                                )
                                did_proj = did_proj or chg
                                max_expert_leak = max(max_expert_leak, lk_a)
                                new_t[bk], new_t[ak] = b_n, a_n
                    elif kind == "swiglu":
                        for proj in _SWIGLU_PROJECTIONS:
                            if proj in ("w1", "w2") and px is not None:
                                w_n, _, lk_a, chg = _project_dense_matrix_into_nullspace(
                                    t_b[proj], t_u[proj], px, "input", leak_tol
                                )
                                did_proj = did_proj or chg
                                max_expert_leak = max(max_expert_leak, lk_a)
                                new_t[proj] = w_n
                            elif proj == "w3" and (pact is not None or px is not None):
                                p_w3, w3_side = (
                                    (pact, "input") if pact is not None else (px, "output")
                                )
                                assert p_w3 is not None
                                w_n, _, lk_a, chg = _project_dense_matrix_into_nullspace(
                                    t_b[proj], t_u[proj], p_w3, w3_side, leak_tol
                                )
                                did_proj = did_proj or chg
                                max_expert_leak = max(max_expert_leak, lk_a)
                                new_t[proj] = w_n
                            else:
                                new_t[proj] = t_u[proj].clone()
                    else:
                        for k, t_val in t_u.items():
                            if px is not None:
                                side = (
                                    "input"
                                    if (t_val.ndim == 1 or int(t_val.shape[1]) == hidden_size)
                                    else "output"
                                )
                                if t_val.ndim == 1:
                                    b_row = t_b[k].detach().to("cpu", dtype=torch.float64)
                                    d_row = t_val.detach().to("cpu", dtype=torch.float64) - b_row
                                    q_p = torch.eye(hidden_size, dtype=torch.float64) - px
                                    lk_b = float(torch.linalg.vector_norm(q_p @ d_row).item())
                                    if lk_b > leak_tol:
                                        p_d = px @ d_row
                                        lk_a = float(torch.linalg.vector_norm(q_p @ p_d).item())
                                        max_expert_leak = max(max_expert_leak, lk_a)
                                        new_t[k] = (b_row + p_d).to(device=t_val.device, dtype=t_val.dtype)
                                        did_proj = True
                                    else:
                                        max_expert_leak = max(max_expert_leak, lk_b)
                                        new_t[k] = t_val.clone()
                                else:
                                    w_n, _, lk_a, chg = _project_dense_matrix_into_nullspace(
                                        t_b[k], t_val, px, side, leak_tol
                                    )
                                    did_proj = did_proj or chg
                                    max_expert_leak = max(max_expert_leak, lk_a)
                                    new_t[k] = w_n
                            else:
                                new_t[k] = t_val.clone()
                    if did_proj:
                        projected_experts_set.add(e_idx)
                        expert_actions[e_idx] = "projected_into_p_null"
                    else:
                        expert_actions[e_idx] = "already_nullspace_invariant"

            grafted_experts[l_idx][e_idx] = _repackage_grafted_expert(spec_u, new_t)

            # 2. Selective OOF Router Gate Grafting inside P_0
            if e_idx in sink_set or alpha == 0.0:
                router_diag[e_idx] = {
                    "action": "frozen_sink_anchor" if e_idx in sink_set else "frozen_alpha_zero",
                    "rel_weight": 0.0,
                    "router_nullspace_leak": 0.0,
                }
                continue

            delta_r = w_unf[e_idx] - w_frz[e_idx]
            p_r = _resolve_router_nullspace_projector(
                p_null_by_expert, l_idx, e_idx, hidden_size, eig_thr, auto_p_prose
            )
            q_r_perp = torch.eye(hidden_size, dtype=torch.float64) - p_r
            dr_proj = p_r @ delta_r
            dr_norm = float(delta_r.norm().item())
            dr_proj_norm = float(dr_proj.norm().item())

            if dr_proj_norm <= 1e-12:
                router_diag[e_idx] = {
                    "action": "frozen_zero_nullspace_delta",
                    "rel_weight": 0.0,
                    "router_nullspace_leak": 0.0,
                }
                continue

            if prose_by_layer is not None:
                x_s_e: torch.Tensor | None = None
                if specialist_domain_hidden_states is not None:
                    dom_layer = specialist_domain_hidden_states.get(
                        l_idx, specialist_domain_hidden_states
                    )
                    if isinstance(dom_layer, Mapping) and e_idx in dom_layer:
                        cand_s = _validate_hidden_states_2d(
                            dom_layer[e_idx],
                            f"specialist_domain_hidden_states[{l_idx}][{e_idx}]",
                            expected_dim=hidden_size,
                        )
                        if int(cand_s.shape[0]) >= folds_i:
                            x_s_e = cand_s
                if x_s_e is None:
                    assert spec_by_layer is not None
                    x_s_e = spec_by_layer[l_idx]

                oof_res = _verify_router_delta_oof(
                    delta_r,
                    prose_by_layer[l_idx],
                    x_s_e,
                    p_router=p_r,
                    n_folds=folds_i,
                    min_oof_snr=min_snr,
                    max_oof_prose_leak_ratio=max_leak,
                    sv_rel_tol=sv_tol,
                    variance_threshold=var_thr,
                )
                rel_e = float(oof_res["rel_weight"])
                step = alpha * rel_e * dr_proj
                w_graft[e_idx] = w_frz[e_idx] + step
                r_leak = float((q_r_perp @ step).norm().item())
                max_router_leak = max(max_router_leak, r_leak)
                if rel_e > 0.0:
                    oof_verified_set.add(e_idx)
                router_diag[e_idx] = {
                    "action": "oof_selective_graft" if rel_e > 0.0 else "oof_rejected_kept_frozen",
                    "router_nullspace_leak": r_leak,
                    **oof_res,
                }
            else:
                align = dr_proj_norm / max(dr_norm, 1e-12)
                if align < min_align:
                    router_diag[e_idx] = {
                        "action": "rejected_pure_prose_router_shift",
                        "rel_weight": 0.0,
                        "nullspace_alignment": align,
                        "router_nullspace_leak": 0.0,
                    }
                    continue
                step = alpha * dr_proj
                w_graft[e_idx] = w_frz[e_idx] + step
                r_leak = float((q_r_perp @ step).norm().item())
                max_router_leak = max(max_router_leak, r_leak)
                oof_verified_set.add(e_idx)
                router_diag[e_idx] = {
                    "action": "nullspace_projected_graft",
                    "rel_weight": 1.0,
                    "nullspace_alignment": align,
                    "router_nullspace_leak": r_leak,
                }

        w_orig = frz_r[l_idx]["weight"]
        r_entry: dict[str, torch.Tensor] = {
            "weight": w_graft.to(device=w_orig.device, dtype=w_orig.dtype).contiguous()
        }
        if "bias" in frz_r[l_idx]:
            r_entry["bias"] = frz_r[l_idx]["bias"].detach().clone()
        grafted_routers[l_idx] = r_entry
        layer_reports[l_idx] = {
            "expert_actions": expert_actions,
            "router_diagnostics": router_diag,
        }

    base_blocks_map = (
        extract_checkpoint_blocks(unfrozen_experts)
        if isinstance(unfrozen_experts, nn.Module) and hasattr(unfrozen_experts, "blocks")
        else None
    )
    grafted_state = _package_grafted_outputs(
        grafted_routers, grafted_experts, frz_meta, unf_meta, base_blocks_map
    )
    report: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "gate_blend_alpha": alpha,
        "sink_experts_pruned": sorted(sink_set),
        "protected_experts": sorted(prot_set),
        "projected_experts": sorted(projected_experts_set),
        "oof_verified_router_experts": sorted(oof_verified_set),
        "max_expert_nullspace_leak": max_expert_leak,
        "max_router_nullspace_leak": max_router_leak,
        "per_layer": layer_reports,
    }
    return grafted_state, report


# ---------------------------------------------------------------------------
# 5. Component (c): End-to-End Cross-Arm MoE Surgery & ANSE Ratchet Gate
# ---------------------------------------------------------------------------


def _normalize_checkpoint_to_blocks(
    ckpt: nn.Module | Mapping[str, Any],
    name: str,
) -> dict[str, dict[str, torch.Tensor]]:
    """Normalize an ``nn.Module`` or block mapping into canonical cloned block dict."""
    if isinstance(ckpt, nn.Module):
        blocks = extract_checkpoint_blocks(ckpt)
    elif isinstance(ckpt, Mapping) and ckpt:
        blocks = {}
        for bid, sub in ckpt.items():
            if not isinstance(bid, str) or not bid.strip():
                raise ValueError(f"{name} block keys must be non-empty strings")
            if not isinstance(sub, Mapping) or not sub:
                raise TypeError(f"{name}[{bid!r}] must be a non-empty Mapping[str, Tensor]")
            blocks[bid] = {}
            for sk, t in sub.items():
                if not isinstance(t, torch.Tensor) or t.numel() == 0 or not t.is_floating_point():
                    raise ValueError(f"{name}[{bid!r}][{sk!r}] must be non-empty floating-point Tensor")
                blocks[bid][sk] = t.detach().clone()
    else:
        raise TypeError(f"{name} must be an nn.Module or non-empty block Mapping")

    for bid, sub in blocks.items():
        for sk, t in sub.items():
            if not bool(torch.isfinite(t).all().item()):
                raise ValueError(f"{name}[{bid!r}][{sk!r}] contains non-finite values")
    return blocks


def _validate_three_arm_layouts(
    base_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    frozen_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    unfrozen_blocks: Mapping[str, Mapping[str, torch.Tensor]],
) -> None:
    """Verify identical block IDs, sub-keys, and tensor shapes across base, frozen, and unfrozen arms."""
    ref_keys = set(base_blocks.keys())
    if ref_keys != set(frozen_blocks.keys()) or ref_keys != set(unfrozen_blocks.keys()):
        raise ValueError("base_checkpoint, frozen_arm_checkpoint, and unfrozen_arm_checkpoint layouts must match")
    has_router = any(k.endswith(".router") for k in ref_keys)
    has_expert = any(".experts." in k for k in ref_keys)
    if not (has_router and has_expert):
        raise ValueError("Checkpoints must expose MoE '.router' and '.experts.' blocks")
    for bid in sorted(ref_keys):
        b_sub, f_sub, u_sub = base_blocks[bid], frozen_blocks[bid], unfrozen_blocks[bid]
        if set(b_sub.keys()) != set(f_sub.keys()) or set(b_sub.keys()) != set(u_sub.keys()):
            raise ValueError(f"Sub-key mismatch on block {bid!r}")
        for sk in b_sub:
            if b_sub[sk].shape != f_sub[sk].shape or b_sub[sk].shape != u_sub[sk].shape:
                raise ValueError(f"Tensor shape mismatch on {bid}.{sk}")


def _validate_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    vocab_size: int,
    general_prefix: str,
    score_from: int,
    name: str,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Validate token chunk sequences and return ``(ids_2d, is_general_mask, expert_ids)``."""
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence) or not chunks:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")
    rows: list[torch.Tensor] = []
    eids: list[str] = []
    seq_len: int | None = None
    for idx, ch in enumerate(chunks):
        if not isinstance(ch, Mapping) or "expert_id" not in ch or "ids" not in ch:
            raise ValueError(f"{name}[{idx}] must be a mapping containing 'expert_id' and 'ids'")
        eid = str(ch["expert_id"]).strip()
        if not eid:
            raise ValueError(f"{name}[{idx}]['expert_id'] must be non-empty")
        ids_t = torch.as_tensor(ch["ids"], dtype=torch.long)
        if ids_t.ndim != 1 or ids_t.numel() < 2:
            raise ValueError(f"{name}[{idx}]['ids'] must be 1D with length >= 2")
        if bool((ids_t < 0).any().item()) or bool((ids_t >= vocab_size).any().item()):
            raise ValueError(f"{name}[{idx}]['ids'] token IDs must lie in [0, {vocab_size})")
        if seq_len is None:
            seq_len = int(ids_t.numel())
        elif int(ids_t.numel()) != seq_len:
            raise ValueError(f"All {name} rows must share sequence length {seq_len}")
        rows.append(ids_t)
        eids.append(eid)

    assert seq_len is not None
    if max(0, score_from - 1) >= seq_len - 1:
        raise ValueError(f"{name} sequence length {seq_len} too short for score_from={score_from}")
    ids_2d = torch.stack(rows, dim=0)
    is_gen = np.array([eid.startswith(general_prefix) for eid in eids], dtype=bool)
    if not bool(np.any(is_gen)) or not bool(np.any(~is_gen)):
        raise ValueError(f"{name} must include both general ({general_prefix}*) and specialist chunks")
    return ids_2d, is_gen, eids


@torch.no_grad()
def _collect_probe_activations_and_nullspace(
    model: nn.Module,
    base_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    probe_ids: torch.Tensor,
    is_general: np.ndarray,
    probe_eids: Sequence[str],
    cfg: CrossArmSurgeryConfig,
) -> tuple[
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, dict[int, torch.Tensor]],
    dict[int, dict[str, torch.Tensor]],
]:
    """Run forward pass on ``base_blocks`` to collect prose/spec activations and ``P_0`` projectors."""
    apply_checkpoint_blocks_inplace(model, base_blocks)
    model.eval()
    cos = model.cos_cached.to(probe_ids.device)
    sin = model.sin_cached.to(probe_ids.device)

    xb = torch.cat(
        [
            compute_embeddings_batch(model, probe_ids[b0 : b0 + cfg.batch_size])
            for b0 in range(0, probe_ids.shape[0], cfg.batch_size)
        ],
        dim=0,
    )
    start_pos = max(0, cfg.score_from - 1)
    gen_mask = torch.as_tensor(is_general, dtype=torch.bool)
    spec_mask = ~gen_mask
    dom_indices: dict[int, list[int]] = {}
    for idx, eid in enumerate(probe_eids):
        if not is_general[idx]:
            m = _EXPERT_DOMAIN_RE.match(eid)
            if m is not None:
                dom_indices.setdefault(int(m.group(1)), []).append(idx)

    prose_by_layer: dict[int, torch.Tensor] = {}
    spec_by_layer: dict[int, torch.Tensor] = {}
    dom_by_layer: dict[int, dict[int, torch.Tensor]] = {}
    p_null_by_layer: dict[int, dict[str, torch.Tensor]] = {}

    for l_idx, blk in enumerate(model.blocks):
        xm_l, mi_l, nx_l = [], [], []
        for b0 in range(0, xb.shape[0], cfg.batch_size):
            xm_b, mi_b = forward_block_pre_mlp(blk, xb[b0 : b0 + cfg.batch_size], cos, sin)
            xm_l.append(xm_b)
            mi_l.append(mi_b)
            nx_l.append(forward_block_post_mlp(blk, xm_b, mi_b))
        mlp_in = torch.cat(mi_l, dim=0)
        xb = torch.cat(nx_l, dim=0)

        scored = mlp_in[:, start_pos:-1, :]
        h_dim = int(scored.shape[-1])
        x_p = scored[gen_mask].reshape(-1, h_dim).contiguous()
        x_s = scored[spec_mask].reshape(-1, h_dim).contiguous()
        prose_by_layer[l_idx] = x_p
        spec_by_layer[l_idx] = x_s
        dom_by_layer[l_idx] = {
            e_idx: scored[ixs].reshape(-1, h_dim).contiguous()
            for e_idx, ixs in dom_indices.items()
        }

        px, _ = compute_prose_nullspace_projector_scaled(
            x_p,
            sv_rel_tol=cfg.nullspace_sv_rel_tol,
            variance_threshold=cfg.nullspace_variance_threshold,
        )
        layer_p: dict[str, torch.Tensor] = {"x": px}
        base_m = getattr(blk.mlp, "base_mlp", blk.mlp)
        if hasattr(base_m, "w1") and hasattr(base_m, "w2"):
            w_dt = base_m.w1.weight.dtype
            xp_cast = x_p.to(dtype=w_dt)
            act = F.silu(base_m.w1(xp_cast).float()) * base_m.w2(xp_cast).float()
            pact, _ = compute_prose_nullspace_projector_scaled(
                act,
                sv_rel_tol=cfg.nullspace_sv_rel_tol,
                variance_threshold=cfg.nullspace_variance_threshold,
            )
            layer_p["act"] = pact
        p_null_by_layer[l_idx] = layer_p

    return prose_by_layer, spec_by_layer, dom_by_layer, p_null_by_layer


@torch.no_grad()
def _forward_pre_ln(
    model: nn.Module,
    ids_2d: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Forward ``model.blocks`` returning hidden states before ``model.ln``."""
    cos = model.cos_cached.to(ids_2d.device)
    sin = model.sin_cached.to(ids_2d.device)
    xb = torch.cat(
        [
            compute_embeddings_batch(model, ids_2d[b0 : b0 + batch_size])
            for b0 in range(0, ids_2d.shape[0], batch_size)
        ],
        dim=0,
    )
    for blk in model.blocks:
        nx = []
        for b0 in range(0, xb.shape[0], batch_size):
            xm, mi = forward_block_pre_mlp(blk, xb[b0 : b0 + batch_size], cos, sin)
            nx.append(forward_block_post_mlp(blk, xm, mi))
        xb = torch.cat(nx, dim=0)
    return xb


@torch.no_grad()
def _detect_sink_experts_loeo(
    model: nn.Module,
    unfrozen_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    base_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    probe_ids: torch.Tensor,
    is_general: np.ndarray,
    probe_eids: Sequence[str],
    cfg: CrossArmSurgeryConfig,
) -> tuple[list[int], dict[str, dict[str, float]]]:
    """Identify pure prose-corrupting ghost-sink experts via Leave-One-Expert-Out attribution."""
    sub_ix: list[int] = []
    counts: dict[str, int] = {}
    for idx, eid in enumerate(probe_eids):
        cap = cfg.loeo_chunks_per_domain * 2 if is_general[idx] else cfg.loeo_chunks_per_domain
        if counts.get(eid, 0) < cap:
            sub_ix.append(idx)
            counts[eid] = counts.get(eid, 0) + 1

    sub_ids = probe_ids[sub_ix]
    sub_gen = is_general[sub_ix]
    ln_eps = float(getattr(model.ln, "eps", 1e-5))

    def _eval_loeo(blocks: Mapping[str, Mapping[str, torch.Tensor]]) -> tuple[float, float]:
        apply_checkpoint_blocks_inplace(model, blocks)
        model.eval()
        hp = _forward_pre_ln(model, sub_ids, cfg.batch_size)
        nlls = (
            per_chunk_nll_from_pre_ln(
                hp, model.ln.weight, model.ln.bias, ln_eps, model.head.weight, sub_ids, score_from=cfg.score_from
            )
            .cpu()
            .numpy()
        )
        return float(nlls[sub_gen].mean()), float(nlls[~sub_gen].mean())

    ref_gen, ref_spec = _eval_loeo(unfrozen_blocks)

    # Also construct base-experts + unfrozen-router reference for Add-One-In attribution
    # (prevents LayerNorm saturation masking when multiple prose-harming experts co-exist).
    base_with_unf_router = {
        bid: {
            sk: t.detach().clone()
            for sk, t in (
                unfrozen_blocks[bid] if bid.endswith(".router") else base_blocks[bid]
            ).items()
        }
        for bid in base_blocks
    }
    aoi_ref_gen, aoi_ref_spec = _eval_loeo(base_with_unf_router)

    expert_ids = sorted(
        {
            int(m.group(2))
            for k in unfrozen_blocks
            for m in [_BLOCK_EXPERT_RE.match(k)]
            if m is not None
        }
    )

    loeo_report: dict[str, dict[str, float]] = {}
    detected_sinks: list[int] = []
    for e_idx in expert_ids:
        mod_blocks = {
            bid: {sk: t.detach().clone() for sk, t in sub.items()}
            for bid, sub in unfrozen_blocks.items()
        }
        aoi_blocks = {
            bid: {sk: t.detach().clone() for sk, t in sub.items()}
            for bid, sub in base_with_unf_router.items()
        }
        for bid, sub in mod_blocks.items():
            m = _BLOCK_EXPERT_RE.match(bid)
            if m is not None and int(m.group(2)) == e_idx:
                if any(k.endswith("_B") for k in sub):
                    for k in sub:
                        if k.endswith("_B"):
                            sub[k].zero_()
                else:
                    for k in sub:
                        sub[k].copy_(base_blocks[bid][k])
                aoi_blocks[bid] = {sk: t.detach().clone() for sk, t in unfrozen_blocks[bid].items()}

        wo_gen, wo_spec = _eval_loeo(mod_blocks)
        wi_gen, wi_spec = _eval_loeo(aoi_blocks)
        loo_harm = ref_gen - wo_gen
        aoi_harm = wi_gen - aoi_ref_gen
        prose_harm = max(loo_harm, aoi_harm)

        loo_spec_val = wo_spec - ref_spec
        aoi_spec_val = aoi_ref_spec - wi_spec
        spec_value = max(loo_spec_val, aoi_spec_val)
        loeo_report[f"expert_{e_idx}"] = {
            "prose_harm_nats": prose_harm,
            "loo_prose_harm_nats": loo_harm,
            "aoi_prose_harm_nats": aoi_harm,
            "specialist_value_nats": spec_value,
        }
        if prose_harm >= cfg.sink_prose_harm_floor_nats and spec_value < cfg.sink_spec_preserve_floor_nats:
            detected_sinks.append(e_idx)

    return detected_sinks, loeo_report


@torch.no_grad()
def _calibrate_final_ln(
    model: nn.Module | None,
    cand_blocks: dict[str, dict[str, torch.Tensor]],
    frozen_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    unfrozen_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    base_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    probe_ids: torch.Tensor | None,
    probe_is_gen: np.ndarray | None,
    cfg: CrossArmSurgeryConfig,
) -> float:
    """Interpolate ``final_ln`` between ``frozen_blocks`` and ``unfrozen_blocks`` under E0 guard."""
    if "final_ln" not in cand_blocks:
        return 0.0
    if cfg.fold_ln_lambda is not None:
        lam = float(cfg.fold_ln_lambda)
    elif model is not None and probe_ids is not None and probe_is_gen is not None:
        ln_eps = float(getattr(model.ln, "eps", 1e-5))
        apply_checkpoint_blocks_inplace(model, base_blocks)
        h_base = _forward_pre_ln(model, probe_ids, cfg.batch_size)
        base_nlls = per_chunk_nll_from_pre_ln(
            h_base, model.ln.weight, model.ln.bias, ln_eps, model.head.weight, probe_ids, score_from=cfg.score_from
        ).numpy()
        base_e0 = float(base_nlls[probe_is_gen].mean())
        base_all = float(base_nlls.mean())

        apply_checkpoint_blocks_inplace(model, cand_blocks)
        h_cand = _forward_pre_ln(model, probe_ids, cfg.batch_size)

        frz_ln = frozen_blocks["final_ln"]
        unf_ln = unfrozen_blocks["final_ln"]
        w0 = frz_ln["weight"].detach().to("cpu", dtype=torch.float32)
        w1 = unf_ln["weight"].detach().to("cpu", dtype=torch.float32)
        b0 = frz_ln["bias"].detach().to("cpu", dtype=torch.float32) if "bias" in frz_ln else None
        b1 = unf_ln["bias"].detach().to("cpu", dtype=torch.float32) if "bias" in unf_ln else None

        feasible: list[tuple[float, float]] = []
        fallback: list[tuple[float, float]] = []
        for cand_l in (0.0, 0.25, 0.50, 0.75, 1.0):
            wt = w0 + cand_l * (w1 - w0)
            bt = (b0 + cand_l * (b1 - b0)) if (b0 is not None and b1 is not None) else torch.zeros_like(wt)
            ls = per_chunk_nll_from_pre_ln(
                h_cand, wt, bt, ln_eps, model.head.weight, probe_ids, score_from=cfg.score_from
            ).numpy()
            e0_ratio = float(ls[probe_is_gen].mean()) / max(base_e0, 1e-8)
            mean_diff = float(ls.mean()) - base_all
            if e0_ratio <= min(cfg.max_general_drift_ratio, 1.005):
                feasible.append((mean_diff, cand_l))
            fallback.append((e0_ratio, cand_l))
        if feasible:
            lam = sorted(feasible, key=lambda t: t[0])[0][1]
        else:
            lam = sorted(fallback, key=lambda t: t[0])[0][1]
    else:
        lam = 0.0

    for sk, t_f in frozen_blocks["final_ln"].items():
        f64 = t_f.detach().to("cpu", dtype=torch.float64)
        u64 = unfrozen_blocks["final_ln"][sk].detach().to("cpu", dtype=torch.float64)
        cand_blocks["final_ln"][sk] = (f64 + lam * (u64 - f64)).to(device=t_f.device, dtype=t_f.dtype)
    return float(lam)


@torch.no_grad()
def _evaluate_arm_state(
    *,
    model: nn.Module | None,
    blocks: Mapping[str, Mapping[str, torch.Tensor]],
    eval_fn: Callable[[Any], Mapping[str, Any]] | None,
    eval_ids: torch.Tensor | None,
    eval_eids: Sequence[str],
    cfg: CrossArmSurgeryConfig,
) -> dict[str, Any]:
    """Evaluate one MoE arm via custom ``eval_fn`` or exact causal mid-row NLL."""
    if model is not None:
        apply_checkpoint_blocks_inplace(model, blocks)
        model.eval()
        target: Any = model
    else:
        target = {bid: {sk: t.detach().clone() for sk, t in sub.items()} for bid, sub in blocks.items()}

    if eval_fn is not None:
        raw = eval_fn(target)
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("eval_fn must return a non-empty mapping of domain -> NLL")
        if raw.get("non_finite_outputs") is True or raw.get("all_finite") is False:
            return {"all_finite": False, "domain_nlls": {}, "general_nll": None, "spec_nll": None, "nll_mean": None}
        domain_nlls: dict[str, float] = {}
        for k, v in raw.items():
            if k in ("non_finite_outputs", "all_finite"):
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"eval_fn[{k!r}] must be numeric, got {type(v).__name__}")
            fv = float(v)
            if not math.isfinite(fv):
                return {"all_finite": False, "domain_nlls": {}, "general_nll": None, "spec_nll": None, "nll_mean": None}
            domain_nlls[str(k)] = fv
    else:
        assert model is not None and eval_ids is not None
        hp = _forward_pre_ln(model, eval_ids, cfg.batch_size)
        if not bool(torch.isfinite(hp).all().item()):
            return {"all_finite": False, "domain_nlls": {}, "general_nll": None, "spec_nll": None, "nll_mean": None}
        nlls = per_chunk_nll_from_pre_ln(
            hp,
            model.ln.weight,
            model.ln.bias,
            float(getattr(model.ln, "eps", 1e-5)),
            model.head.weight,
            eval_ids,
            score_from=cfg.score_from,
        )
        if not bool(torch.isfinite(nlls).all().item()):
            return {"all_finite": False, "domain_nlls": {}, "general_nll": None, "spec_nll": None, "nll_mean": None}
        by_dom: dict[str, list[float]] = {}
        for eid, val in zip(eval_eids, nlls.tolist(), strict=True):
            by_dom.setdefault(eid, []).append(float(val))
        domain_nlls = {dom: float(np.mean(vals)) for dom, vals in sorted(by_dom.items())}

    gen_vals = [v for k, v in domain_nlls.items() if k.startswith(cfg.general_prefix)]
    spec_vals = [v for k, v in domain_nlls.items() if not k.startswith(cfg.general_prefix)]
    if not gen_vals or not spec_vals:
        raise ValueError(
            f"Evaluation output must contain both general ({cfg.general_prefix}*) and specialist domains"
        )
    return {
        "all_finite": True,
        "domain_nlls": domain_nlls,
        "general_nll": float(np.mean(gen_vals)),
        "spec_nll": float(np.mean(spec_vals)),
        "nll_mean": float(np.mean(list(domain_nlls.values()))),
    }


def _score_arm_against_base(
    arm_eval: Mapping[str, Any],
    base_eval: Mapping[str, Any],
    cfg: CrossArmSurgeryConfig,
) -> dict[str, Any]:
    """Compute drift ratios, specialist gain, and ANSE Energy relative to ``base_eval``."""
    if not arm_eval["all_finite"]:
        return {
            "all_finite": False,
            "general_drift_ratio": float("inf"),
            "worst_domain_ratio": float("inf"),
            "general_guard_ok": False,
            "worst_guard_ok": False,
            "guard_ok": False,
            "specialist_gain_nats": 0.0,
            "mean_diff_vs_base": float("inf"),
            "anse_energy": float(cfg.maximum_pain_energy),
        }

    base_doms = base_eval["domain_nlls"]
    cand_doms = arm_eval["domain_nlls"]
    if set(base_doms.keys()) != set(cand_doms.keys()):
        return {
            "all_finite": False,
            "general_drift_ratio": float("inf"),
            "worst_domain_ratio": float("inf"),
            "general_guard_ok": False,
            "worst_guard_ok": False,
            "guard_ok": False,
            "specialist_gain_nats": 0.0,
            "mean_diff_vs_base": float("inf"),
            "anse_energy": float(cfg.maximum_pain_energy),
        }

    gen_ratio = float(arm_eval["general_nll"]) / max(float(base_eval["general_nll"]), 1e-8)
    worst_ratio = max(float(cand_doms[d]) / max(float(base_doms[d]), 1e-8) for d in base_doms)
    mean_diff = float(arm_eval["nll_mean"]) - float(base_eval["nll_mean"])
    spec_gain = float(base_eval["spec_nll"]) - float(arm_eval["spec_nll"])

    gen_ok = gen_ratio <= cfg.max_general_drift_ratio + 1e-12
    worst_ok = worst_ratio <= cfg.max_worst_domain_drift_ratio + 1e-12
    guard_ok = gen_ok and (worst_ok if cfg.enforce_all_domain_guard else True)
    if not guard_ok:
        anse_e = cfg.maximum_pain_energy + max(0.0, mean_diff)
    else:
        soft_pen = cfg.soft_margin_penalty if gen_ratio > cfg.anse_safe_general_ratio else 0.0
        anse_e = mean_diff + soft_pen

    return {
        "all_finite": True,
        "general_drift_ratio": gen_ratio,
        "worst_domain_ratio": worst_ratio,
        "general_guard_ok": gen_ok,
        "worst_guard_ok": worst_ok,
        "guard_ok": guard_ok,
        "specialist_gain_nats": spec_gain,
        "mean_diff_vs_base": mean_diff,
        "anse_energy": float(anse_e),
    }


def _resolve_surgery_config(
    config: CrossArmSurgeryConfig | None,
    **overrides: Any,
) -> CrossArmSurgeryConfig:
    """Merge optional ``CrossArmSurgeryConfig`` with explicit keyword overrides."""
    if config is not None and not isinstance(config, CrossArmSurgeryConfig):
        raise TypeError(f"config must be CrossArmSurgeryConfig or None, got {type(config).__name__}")
    base_dict = asdict(config) if config is not None else asdict(CrossArmSurgeryConfig())
    for k, v in overrides.items():
        if v is not None:
            base_dict[k] = v
    return CrossArmSurgeryConfig(**base_dict)


@torch.no_grad()
def execute_verified_cross_arm_moe_surgery(
    base_checkpoint: nn.Module | Mapping[str, Any],
    frozen_arm_checkpoint: nn.Module | Mapping[str, Any],
    unfrozen_arm_checkpoint: nn.Module | Mapping[str, Any],
    *,
    target_model: nn.Module | None = None,
    probe_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_fn: Callable[[Any], Mapping[str, Any]] | None = None,
    p_null_by_expert: torch.Tensor | Mapping[Any, Any] | None = None,
    sink_experts: Sequence[int] | set[int] | None = None,
    prose_hidden_states: torch.Tensor | Mapping[Any, torch.Tensor] | Sequence[torch.Tensor] | None = None,
    spec_hidden_states: torch.Tensor | Mapping[Any, torch.Tensor] | Sequence[torch.Tensor] | None = None,
    gate_blend_alpha: float | None = None,
    fold_ln_lambda: float | None = None,
    rollback_target: str | None = None,
    config: CrossArmSurgeryConfig | None = None,
    inplace: bool | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run closed-form cross-arm router-frozen + expert-unfrozen MoE surgery under ANSE ratification."""
    cfg = _resolve_surgery_config(
        config,
        gate_blend_alpha=gate_blend_alpha,
        fold_ln_lambda=fold_ln_lambda,
        rollback_target=rollback_target,
        inplace=inplace,
    )
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError(f"eval_fn must be callable or None, got {type(eval_fn).__name__}")
    if target_model is not None and not isinstance(target_model, nn.Module):
        raise TypeError(f"target_model must be an nn.Module or None, got {type(target_model).__name__}")

    base_blocks = _normalize_checkpoint_to_blocks(base_checkpoint, "base_checkpoint")
    frozen_blocks = _normalize_checkpoint_to_blocks(frozen_arm_checkpoint, "frozen_arm_checkpoint")
    unfrozen_blocks = _normalize_checkpoint_to_blocks(unfrozen_arm_checkpoint, "unfrozen_arm_checkpoint")
    _validate_three_arm_layouts(base_blocks, frozen_blocks, unfrozen_blocks)

    work_model: nn.Module | None = None
    if target_model is not None:
        work_model = target_model if cfg.inplace else copy.deepcopy(target_model)
    elif isinstance(frozen_arm_checkpoint, nn.Module):
        work_model = frozen_arm_checkpoint if cfg.inplace else copy.deepcopy(frozen_arm_checkpoint)
    elif isinstance(base_checkpoint, nn.Module):
        work_model = base_checkpoint if cfg.inplace else copy.deepcopy(base_checkpoint)
    elif isinstance(unfrozen_arm_checkpoint, nn.Module):
        work_model = unfrozen_arm_checkpoint if cfg.inplace else copy.deepcopy(unfrozen_arm_checkpoint)

    if work_model is None and eval_fn is None:
        raise ValueError("Either an nn.Module checkpoint/target_model or eval_fn must be provided")

    vocab_size = int(work_model.head.weight.shape[0]) if work_model is not None else 0
    probe_ids: torch.Tensor | None = None
    probe_is_gen: np.ndarray | None = None
    probe_eids: list[str] = []
    if probe_chunks is not None and work_model is not None:
        probe_ids, probe_is_gen, probe_eids = _validate_chunks(
            probe_chunks,
            vocab_size=vocab_size,
            general_prefix=cfg.general_prefix,
            score_from=cfg.score_from,
            name="probe_chunks",
        )

    eval_ids: torch.Tensor | None = None
    eval_eids: list[str] = []
    if eval_chunks is not None and work_model is not None:
        eval_ids, _, eval_eids = _validate_chunks(
            eval_chunks,
            vocab_size=vocab_size,
            general_prefix=cfg.general_prefix,
            score_from=cfg.score_from,
            name="eval_chunks",
        )
    elif probe_ids is not None:
        eval_ids, eval_eids = probe_ids, probe_eids
    elif eval_fn is None:
        raise ValueError("Either probe_chunks/eval_chunks or eval_fn must be provided")

    base_sha = compute_checkpoint_sha256(base_blocks)
    frozen_sha = compute_checkpoint_sha256(frozen_blocks)
    unfrozen_sha = compute_checkpoint_sha256(unfrozen_blocks)

    base_mlps = _resolve_optional_base_mlps(
        None, work_model, base_checkpoint, frozen_arm_checkpoint, unfrozen_arm_checkpoint
    )

    was_training = work_model.training if work_model is not None else False
    if work_model is not None:
        work_model.eval()
    try:
        # 1. Gather probe activations and AlphaEdit P_0 if probe_chunks available
        dom_by_layer: dict[int, dict[int, torch.Tensor]] | None = None
        probe_prose_l: dict[int, torch.Tensor] | None = None
        probe_spec_l: dict[int, torch.Tensor] | None = None
        auto_p_null: dict[int, dict[str, torch.Tensor]] | None = None
        if work_model is not None and probe_ids is not None and probe_is_gen is not None:
            probe_prose_l, probe_spec_l, dom_by_layer, auto_p_null = (
                _collect_probe_activations_and_nullspace(
                    work_model, base_blocks, probe_ids, probe_is_gen, probe_eids, cfg
                )
            )

        eff_prose_h = prose_hidden_states if prose_hidden_states is not None else probe_prose_l
        eff_spec_h = spec_hidden_states if spec_hidden_states is not None else probe_spec_l

        # 2. Component (a): Decompose router vs expert prose drift (when prose states are available)
        if eff_prose_h is not None:
            drift_decomp = decompose_router_vs_expert_prose_drift(
                base_blocks,
                frozen_blocks,
                unfrozen_blocks,
                unfrozen_blocks,
                eff_prose_h,
                base_experts=base_blocks,
                base_mlps=base_mlps,
                routing_mode=cfg.routing_mode,
                top_k=cfg.top_k,
                spec_hidden_states=eff_spec_h,
            )
            router_kl_drift = float(drift_decomp["router_kl_drift"])
        else:
            drift_decomp = {
                "schema": SCHEMA_VERSION,
                "router_kl_drift": 0.0,
                "dominant_prose_drift_source": "not_evaluated_without_hidden_states",
            }
            router_kl_drift = 0.0

        # 3. Resolve sink_experts via LOEO when omitted
        loeo_report: dict[str, dict[str, float]] = {}
        if sink_experts is not None:
            resolved_sinks = sorted(_validate_expert_index_set(sink_experts, "sink_experts"))
        elif cfg.auto_detect_sink_experts and work_model is not None and probe_ids is not None and probe_is_gen is not None:
            resolved_sinks, loeo_report = _detect_sink_experts_loeo(
                work_model, unfrozen_blocks, base_blocks, probe_ids, probe_is_gen, probe_eids, cfg
            )
        else:
            resolved_sinks = []

        # 4. Component (b): Graft deep specialists onto frozen router + P_0 + OOF logit gating
        resolved_p_null = (
            p_null_by_expert
            if p_null_by_expert is not None
            else (auto_p_null if cfg.auto_project_p_null else None)
        )
        grafted_state, graft_report = graft_specialist_experts_onto_safe_router(
            frozen_blocks,
            unfrozen_blocks,
            unfrozen_blocks,
            resolved_p_null,
            gate_blend_alpha=cfg.gate_blend_alpha,
            sink_experts=resolved_sinks,
            base_experts=base_blocks,
            prose_hidden_states=eff_prose_h,
            spec_hidden_states=eff_spec_h,
            specialist_domain_hidden_states=dom_by_layer,
            nullspace_sv_rel_tol=cfg.nullspace_sv_rel_tol,
            nullspace_variance_threshold=cfg.nullspace_variance_threshold,
            nullspace_eig_threshold=cfg.nullspace_eig_threshold,
            nullspace_leak_tol=cfg.nullspace_leak_tol,
            min_oof_snr=cfg.min_oof_snr,
            n_folds=cfg.n_folds,
            max_oof_prose_leak_ratio=cfg.max_oof_prose_leak_ratio,
            min_nullspace_alignment=cfg.min_nullspace_alignment,
        )

        cand_blocks: dict[str, dict[str, torch.Tensor]] = {
            bid: {sk: t.detach().clone() for sk, t in sub.items()}
            for bid, sub in frozen_blocks.items()
        }
        cand_blocks.update(grafted_state["blocks"])
        lam_ln = _calibrate_final_ln(
            work_model, cand_blocks, frozen_blocks, unfrozen_blocks, base_blocks, probe_ids, probe_is_gen, cfg
        )
        cand_sha = compute_checkpoint_sha256(cand_blocks)

        # 5. Evaluate base, frozen_arm, unfrozen_arm, and hybrid candidate
        base_eval = _evaluate_arm_state(
            model=work_model, blocks=base_blocks, eval_fn=eval_fn, eval_ids=eval_ids, eval_eids=eval_eids, cfg=cfg
        )
        if not base_eval["all_finite"]:
            raise ValueError("base_checkpoint produced non-finite evaluation NLL")

        frozen_eval = _evaluate_arm_state(
            model=work_model, blocks=frozen_blocks, eval_fn=eval_fn, eval_ids=eval_ids, eval_eids=eval_eids, cfg=cfg
        )
        frozen_metrics = _score_arm_against_base(frozen_eval, base_eval, cfg)

        unfrozen_eval = _evaluate_arm_state(
            model=work_model, blocks=unfrozen_blocks, eval_fn=eval_fn, eval_ids=eval_ids, eval_eids=eval_eids, cfg=cfg
        )
        unfrozen_metrics = _score_arm_against_base(unfrozen_eval, base_eval, cfg)

        cand_eval = _evaluate_arm_state(
            model=work_model, blocks=cand_blocks, eval_fn=eval_fn, eval_ids=eval_ids, eval_eids=eval_eids, cfg=cfg
        )
        cand_metrics = _score_arm_against_base(cand_eval, base_eval, cfg)

        base_anse_e = 0.0
        frozen_anse_e = float(frozen_metrics["anse_energy"])
        unfrozen_anse_e = float(unfrozen_metrics["anse_energy"])
        cand_anse_e = float(cand_metrics["anse_energy"])

        delta_e_vs_base = cand_anse_e - base_anse_e
        delta_e_vs_frozen = cand_anse_e - frozen_anse_e
        delta_e_vs_unfrozen = cand_anse_e - unfrozen_anse_e

        accepted = bool(
            cand_eval["all_finite"]
            and cand_metrics["guard_ok"]
            and (delta_e_vs_base < -cfg.min_energy_improvement)
            and (
                not cfg.require_beat_both_donors
                or (
                    delta_e_vs_frozen < -cfg.min_energy_improvement
                    and delta_e_vs_unfrozen < -cfg.min_energy_improvement
                )
            )
        )

        # 6. Atomic commit or bitwise rollback
        if accepted:
            status = "CROSS_ARM_SYNERGY_COMMITTED"
            rolled_back = False
            rollback_verified = False
            final_blocks = cand_blocks
            final_eval = cand_eval
            final_metrics = cand_metrics
            expected_final_sha = cand_sha
        else:
            status = "ROLLED_BACK_ANSE_MAXIMUM_PAIN"
            rolled_back = True
            if cfg.rollback_target == "frozen_arm":
                final_blocks = frozen_blocks
                final_eval = frozen_eval
                final_metrics = frozen_metrics
                expected_final_sha = frozen_sha
            elif cfg.rollback_target == "base":
                final_blocks = base_blocks
                final_eval = base_eval
                final_metrics = _score_arm_against_base(base_eval, base_eval, cfg)
                expected_final_sha = base_sha
            else:
                final_blocks = unfrozen_blocks
                final_eval = unfrozen_eval
                final_metrics = unfrozen_metrics
                expected_final_sha = unfrozen_sha

        if work_model is not None:
            apply_checkpoint_blocks_inplace(work_model, final_blocks)
            extracted = extract_checkpoint_blocks(work_model)
            scoped_blocks = {bid: extracted[bid] for bid in final_blocks if bid in extracted}
            final_sha = compute_checkpoint_sha256(scoped_blocks)
        else:
            final_sha = compute_checkpoint_sha256(final_blocks)

        if rolled_back:
            rollback_verified = final_sha == expected_final_sha
            if not rollback_verified:
                raise RuntimeError(
                    f"Atomic bitwise rollback SHA-256 mismatch: expected {expected_final_sha[:12]}, got {final_sha[:12]}"
                )

        final_gen_ratio = float(final_metrics["general_drift_ratio"])
        frozen_gen_ratio = float(frozen_metrics["general_drift_ratio"])
        unfrozen_gen_ratio = float(unfrozen_metrics["general_drift_ratio"])
        cand_gen_ratio = float(cand_metrics["general_drift_ratio"])

        receipt: dict[str, Any] = {
            "object": RECEIPT_OBJECT,
            "schema": SCHEMA_VERSION,
            "status": status,
            "accepted": accepted,
            "rolled_back": rolled_back,
            "rollback_target": cfg.rollback_target,
            "rollback_bitwise_verified": rollback_verified,
            "nll_mean": final_eval["nll_mean"],
            "router_kl_drift": router_kl_drift,
            "specialist_gain_nats": float(final_metrics["specialist_gain_nats"]),
            "anse_energy": float(final_metrics["anse_energy"]),
            "base_nll_mean": float(base_eval["nll_mean"]),
            "frozen_arm_nll_mean": frozen_eval["nll_mean"],
            "unfrozen_arm_nll_mean": unfrozen_eval["nll_mean"],
            "cand_nll_mean": cand_eval["nll_mean"],
            "frozen_arm_anse_energy": frozen_anse_e,
            "unfrozen_arm_anse_energy": unfrozen_anse_e,
            "cand_anse_energy": cand_anse_e,
            "delta_e_vs_base": delta_e_vs_base,
            "delta_e_vs_frozen_arm": delta_e_vs_frozen,
            "delta_e_vs_unfrozen_arm": delta_e_vs_unfrozen,
            "general_drift_ratio": final_gen_ratio if math.isfinite(final_gen_ratio) else None,
            "frozen_arm_general_drift_ratio": frozen_gen_ratio if math.isfinite(frozen_gen_ratio) else None,
            "unfrozen_arm_general_drift_ratio": unfrozen_gen_ratio if math.isfinite(unfrozen_gen_ratio) else None,
            "cand_general_drift_ratio": cand_gen_ratio if math.isfinite(cand_gen_ratio) else None,
            "frozen_arm_spec_gain_nats": float(frozen_metrics["specialist_gain_nats"]),
            "unfrozen_arm_spec_gain_nats": float(unfrozen_metrics["specialist_gain_nats"]),
            "cand_spec_gain_nats": float(cand_metrics["specialist_gain_nats"]),
            "cand_guard_ok": bool(cand_metrics["guard_ok"]),
            "fold_ln_lambda": lam_ln,
            "loeo_sink_attribution": loeo_report,
            "drift_decomposition": {
                k: v
                for k, v in drift_decomp.items()
                if k not in ("per_layer", "per_expert_attribution")
            },
            "graft_summary": {
                "gate_blend_alpha": graft_report["gate_blend_alpha"],
                "sink_experts_pruned": graft_report["sink_experts_pruned"],
                "projected_experts": graft_report["projected_experts"],
                "oof_verified_router_experts": graft_report["oof_verified_router_experts"],
                "max_expert_nullspace_leak": graft_report["max_expert_nullspace_leak"],
                "max_router_nullspace_leak": graft_report["max_router_nullspace_leak"],
            },
            "base_state_sha256": base_sha,
            "frozen_arm_state_sha256": frozen_sha,
            "unfrozen_arm_state_sha256": unfrozen_sha,
            "candidate_state_sha256": cand_sha,
            "final_state_sha256": final_sha,
            "config": asdict(cfg),
        }
        receipt["receipt_sha256"] = compute_payload_sha256(receipt)
        return (work_model if work_model is not None else final_blocks), receipt
    finally:
        if work_model is not None:
            work_model.train(was_training)


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "CrossArmSurgeryConfig",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_GATE_BLEND_ALPHA",
    "DEFAULT_MIN_OOF_SNR",
    "DEFAULT_NULLSPACE_LEAK_TOL",
    "GENERAL_EXPERT_PREFIX",
    "RECEIPT_OBJECT",
    "SCHEMA_VERSION",
    "VALID_ROLLBACK_TARGETS",
    "VALID_ROUTING_MODES",
    "apply_checkpoint_blocks_inplace",
    "apply_grafted_state_inplace",
    "compute_checkpoint_sha256",
    "decompose_router_vs_expert_prose_drift",
    "execute_verified_cross_arm_moe_surgery",
    "extract_checkpoint_blocks",
    "graft_specialist_experts_onto_safe_router",
    "verify_cross_arm_receipt_provenance",
]

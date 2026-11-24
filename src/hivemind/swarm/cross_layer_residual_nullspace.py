# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Cross-Layer Residual Interference Cancellation & Beneficial-Anchor Protection for Sparse MoEs.

Generalizes the empirical T4 Stage S2 discovery (where mid/late layer 17 contributes ``-0.2516``
nats of general-domain E0 benefit while adjacent downstream layers 18-19 inject ``+0.0279`` nats
of constructive prose drift) into a training-free Cross-Layer Residual Stream Controller for
open-weight Sparse MoE LoRA adapters (``CrowdV1WithEngram`` + ``CrowdLoRAMoEMLP``):

1. ``identify_beneficial_anchor_and_interfering_layers``:
   Partitions MoE depth into protected beneficial anchor layers (``e0_harm <= anchor_threshold``),
   neutral layers, and interfering layers (``e0_harm >= harm_threshold``), isolating downstream
   interfering layers strictly following upstream beneficial anchors.
2. ``project_downstream_lora_b_orthogonal_to_anchor``:
   Modifies downstream interfering SwiGLU LoRA output factors (``w2_B`` / ``w3_B``) into the
   orthogonal complement of upstream anchor residual deltas while strictly preserving specialist
   residual subspaces (exact null-space projection when rank-deficient, generalized Rayleigh
   minimum-distortion filtering when full-rank).
3. ``optimize_cross_layer_residual_surgery``:
   Executes sequential cross-layer residual null-space surgery while keeping beneficial anchor
   layers bitwise immutable, committing edits if and only if ANSE ``Delta E < 0`` and
   worst-domain drift ``<= +1.0%`` with atomic bitwise rollback and SHA-256 provenance receipts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdLoRAMoEMLP,
    load_moe_trainable_buckets,
)
from hivemind.swarm.layer_localized_attribution import (
    compute_embeddings_batch,
    compute_per_layer_moe_bypass_attribution,
    forward_block_dense_base_mlp,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_causal_midrow_nll,
)

SCHEMA_VERSION: str = "hivemind.cross_layer_residual_nullspace/1"
RECEIPT_OBJECT: str = "hivemind.cross_layer_residual_nullspace_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_ANCHOR_THRESHOLD_NATS: float = -0.015
DEFAULT_HARM_THRESHOLD_NATS: float = 0.002
DEFAULT_MAX_RANK_PROJ: int = 8
DEFAULT_SCORE_FROM: int = 64
DEFAULT_MAX_DOMAIN_DRIFT: float = 0.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_ANSE_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_SUBSPACE_SV_REL_TOL: float = 1e-6

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
    out = _validate_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_unit_interval(value: Any, name: str) -> float:
    """Validate that ``value`` lies inside ``[0.0, 1.0]``."""
    out = _validate_nonneg_finite(value, name)
    if out > 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")
    return out


def _validate_delta_matrix(
    deltas: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor],
    name: str,
) -> torch.Tensor:
    """Validate 2D/3D tensor(s) of activation deltas and flatten to CPU ``float64`` ``(N, D)``."""
    if isinstance(deltas, torch.Tensor):
        items: list[tuple[str, torch.Tensor]] = [(name, deltas)]
    elif isinstance(deltas, Mapping):
        if not deltas:
            raise ValueError(f"{name} mapping must not be empty")
        items = [(f"{name}[{k!r}]", v) for k, v in sorted(deltas.items())]
    elif isinstance(deltas, Sequence) and not isinstance(deltas, (str, bytes, bytearray)):
        if not deltas:
            raise ValueError(f"{name} sequence must not be empty")
        items = [(f"{name}[{idx}]", v) for idx, v in enumerate(deltas)]
    else:
        raise TypeError(
            f"{name} must be a torch.Tensor, Sequence[Tensor], or Mapping[str, Tensor]; "
            f"got {type(deltas).__name__}"
        )

    parts: list[torch.Tensor] = []
    expected_width: int | None = None
    for label, tensor in items:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{label} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.ndim not in (2, 3) or any(dim == 0 for dim in tensor.shape):
            raise ValueError(
                f"{label} must be a non-empty 2D or 3D tensor, got shape {tuple(tensor.shape)}"
            )
        if not tensor.is_floating_point():
            raise TypeError(f"{label} must have a floating-point dtype, got {tensor.dtype}")
        flat = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1, int(tensor.shape[-1]))
        if not bool(torch.isfinite(flat).all().item()):
            raise ValueError(f"{label} must contain only finite values")
        width = int(flat.shape[1])
        if expected_width is None:
            expected_width = width
        elif width != expected_width:
            raise ValueError(
                f"{label} width {width} must match expected width {expected_width}"
            )
        parts.append(flat)
    return torch.cat(parts, dim=0).contiguous()


def _validate_2d_factor(tensor: Any, name: str) -> torch.Tensor:
    """Validate a non-empty finite 2D floating-point weight factor tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D tensor, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    if not bool(torch.isfinite(tensor.detach()).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically with strict finite-number enforcement."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def compute_payload_sha256(payload: Any) -> str:
    """Return lowercase SHA-256 hex digest of ``canonical_json_bytes(payload)``."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_module_state_sha256(module: nn.Module) -> str:
    """Compute a deterministic bitwise SHA-256 digest over a module's ``state_dict`` tensors."""
    if not isinstance(module, nn.Module):
        raise TypeError(f"module must be an nn.Module, got {type(module).__name__}")
    state = module.state_dict()
    if not state:
        raise ValueError("module state_dict must be non-empty")
    digest = hashlib.sha256()
    for key in sorted(state.keys()):
        t_cpu = state[key].detach().to(device="cpu").contiguous()
        if t_cpu.is_floating_point() and not bool(torch.isfinite(t_cpu).all().item()):
            raise ValueError(f"module state tensor {key!r} contains non-finite values")
        digest.update(f"{key}:{t_cpu.dtype}:{tuple(t_cpu.shape)}:".encode())
        digest.update(t_cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_cross_layer_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify canonical ``receipt_sha256`` integrity on a surgery receipt."""
    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a mapping")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


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


def _validate_eval_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    general_prefix: str,
    vocab_size: int,
    score_from: int,
    name: str,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Validate chunk sequence and return ``(ids_2d, is_general_mask, expert_ids)``."""
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence) or len(chunks) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")
    id_rows: list[torch.Tensor] = []
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
        id_rows.append(ids_t)
        expert_ids.append(eid)

    assert seq_len is not None
    start_pos = max(0, score_from - 1)
    if start_pos >= seq_len - 1:
        raise ValueError(
            f"{name} sequence length {seq_len} too short for score_from={score_from} (start_pos={start_pos})"
        )
    ids_2d = torch.stack(id_rows, dim=0)
    is_general = np.array([eid.startswith(general_prefix) for eid in expert_ids], dtype=bool)
    if not bool(np.any(is_general)):
        raise ValueError(f"{name} must include at least one general chunk starting with {general_prefix!r}")
    return ids_2d, is_general, expert_ids


@dataclass(frozen=True)
class CrossLayerResidualConfig:
    """Configuration governing cross-layer residual interference cancellation and ANSE verification."""

    anchor_threshold_nats: float = DEFAULT_ANCHOR_THRESHOLD_NATS
    harm_threshold_nats: float = DEFAULT_HARM_THRESHOLD_NATS
    max_rank_proj: int = DEFAULT_MAX_RANK_PROJ
    proj_strength: float = 1.0
    anchor_sv_rel_tol: float = DEFAULT_SUBSPACE_SV_REL_TOL
    specialist_sv_rel_tol: float = DEFAULT_SUBSPACE_SV_REL_TOL
    general_prefix: str = GENERAL_EXPERT_PREFIX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = 24
    max_general_probe_chunks: int = 24
    max_spec_probe_chunks: int = 24
    max_domain_drift: float = DEFAULT_MAX_DOMAIN_DRIFT
    anse_safe_e0_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain: float = DEFAULT_ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY
    min_energy_improvement: float = 0.0
    protect_specialist_subspace: bool = True
    protect_anchor_layers_bitwise: bool = True
    project_w2_via_base_w3: bool = True
    inplace: bool = True

    def __post_init__(self) -> None:
        anchor_t = _validate_finite_float(self.anchor_threshold_nats, "anchor_threshold_nats")
        harm_t = _validate_nonneg_finite(self.harm_threshold_nats, "harm_threshold_nats")
        if anchor_t > 0.0:
            raise ValueError(f"anchor_threshold_nats must be <= 0.0, got {anchor_t}")
        if anchor_t >= harm_t:
            raise ValueError(
                f"anchor_threshold_nats ({anchor_t}) must be strictly < harm_threshold_nats ({harm_t})"
            )
        _validate_positive_int(self.max_rank_proj, "max_rank_proj")
        _validate_unit_interval(self.proj_strength, "proj_strength")
        a_tol = _validate_positive_finite(self.anchor_sv_rel_tol, "anchor_sv_rel_tol")
        s_tol = _validate_positive_finite(self.specialist_sv_rel_tol, "specialist_sv_rel_tol")
        if a_tol >= 1.0 or s_tol >= 1.0:
            raise ValueError("singular value relative tolerances must be < 1.0")
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_positive_int(self.max_general_probe_chunks, "max_general_probe_chunks")
        _validate_positive_int(self.max_spec_probe_chunks, "max_spec_probe_chunks")
        _validate_nonneg_finite(self.max_domain_drift, "max_domain_drift")
        _validate_positive_finite(self.anse_safe_e0_ratio, "anse_safe_e0_ratio")
        _validate_nonneg_finite(self.maximum_pain, "maximum_pain")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_nonneg_finite(self.min_energy_improvement, "min_energy_improvement")
        for flag_name in (
            "protect_specialist_subspace",
            "protect_anchor_layers_bitwise",
            "project_w2_via_base_w3",
            "inplace",
        ):
            _validate_bool(getattr(self, flag_name), flag_name)


# ---------------------------------------------------------------------------
# 2. Layer Depth Partitioning (Beneficial Anchors vs Neutral vs Interfering)
# ---------------------------------------------------------------------------


def _parse_layer_harm_map(layer_e0_harm_nats: Any) -> dict[int, float]:
    """Normalize sequence or mapping layer harm attributions into ``{layer_idx: harm_nats}``."""
    harm_by_layer: dict[int, float] = {}
    if isinstance(layer_e0_harm_nats, Mapping):
        if not layer_e0_harm_nats:
            raise ValueError("layer_e0_harm_nats mapping must not be empty")
        if "layers" in layer_e0_harm_nats and not isinstance(layer_e0_harm_nats["layers"], (int, float)):
            return _parse_layer_harm_map(layer_e0_harm_nats["layers"])
        for raw_key, raw_val in layer_e0_harm_nats.items():
            if isinstance(raw_key, bool):
                raise TypeError(f"Layer index key must not be bool: {raw_key!r}")
            if isinstance(raw_key, int):
                l_idx = _validate_nonneg_int(raw_key, "layer_index")
            elif isinstance(raw_key, str) and raw_key.strip().isdigit():
                l_idx = int(raw_key.strip())
            else:
                raise ValueError(f"Invalid layer index key {raw_key!r}")
            if l_idx in harm_by_layer:
                raise ValueError(f"Duplicate layer index {l_idx} in layer_e0_harm_nats")
            harm_by_layer[l_idx] = _validate_finite_float(raw_val, f"layer_{l_idx}_e0_harm_nats")
        return harm_by_layer

    if isinstance(layer_e0_harm_nats, (str, bytes)) or not isinstance(layer_e0_harm_nats, Sequence):
        raise TypeError(
            f"layer_e0_harm_nats must be a non-empty Sequence or Mapping, got {type(layer_e0_harm_nats).__name__}"
        )
    if len(layer_e0_harm_nats) == 0:
        raise ValueError("layer_e0_harm_nats must not be empty")

    first = layer_e0_harm_nats[0]
    if isinstance(first, Mapping):
        for idx, item in enumerate(layer_e0_harm_nats):
            if not isinstance(item, Mapping) or "layer" not in item:
                raise ValueError(f"layer_e0_harm_nats[{idx}] must be a mapping containing 'layer'")
            l_idx = _validate_nonneg_int(item["layer"], f"layer_e0_harm_nats[{idx}]['layer']")
            if l_idx in harm_by_layer:
                raise ValueError(f"Duplicate layer index {l_idx} in layer_e0_harm_nats")
            raw_harm = item.get("e0_harm_nats", item.get("general_harm_nats"))
            if raw_harm is None:
                raise ValueError(
                    f"layer_e0_harm_nats[{idx}] must contain 'e0_harm_nats' or 'general_harm_nats'"
                )
            harm_by_layer[l_idx] = _validate_finite_float(raw_harm, f"layer_{l_idx}_e0_harm_nats")
        return harm_by_layer

    for idx, raw_val in enumerate(layer_e0_harm_nats):
        harm_by_layer[idx] = _validate_finite_float(raw_val, f"layer_{idx}_e0_harm_nats")
    return harm_by_layer


def identify_beneficial_anchor_and_interfering_layers(
    layer_e0_harm_nats: Sequence[float] | Sequence[Mapping[str, Any]] | Mapping[Any, Any],
    *,
    anchor_threshold_nats: float = DEFAULT_ANCHOR_THRESHOLD_NATS,
    harm_threshold_nats: float = DEFAULT_HARM_THRESHOLD_NATS,
) -> dict[str, Any]:
    """Partition MoE depth into beneficial anchor, neutral, and interfering layers.

    Negative ``e0_harm_nats`` indicates that bypassing the layer worsens general-domain E0 NLL,
    meaning the layer contributes positive ``anchor_gain_nats = -e0_harm_nats`` (e.g., T4 Stage S2
    layer 17 at ``-0.2516`` nats). Positive ``e0_harm_nats`` indicates constructive prose drift.
    """
    anchor_thresh = _validate_finite_float(anchor_threshold_nats, "anchor_threshold_nats")
    harm_thresh = _validate_nonneg_finite(harm_threshold_nats, "harm_threshold_nats")
    if anchor_thresh > 0.0:
        raise ValueError(f"anchor_threshold_nats must be <= 0.0, got {anchor_thresh}")
    if anchor_thresh >= harm_thresh:
        raise ValueError(
            f"anchor_threshold_nats ({anchor_thresh}) must be strictly < harm_threshold_nats ({harm_thresh})"
        )

    harm_by_layer = _parse_layer_harm_map(layer_e0_harm_nats)
    sorted_layers = sorted(harm_by_layer.keys())
    beneficial_anchors: list[int] = []
    neutral_layers: list[int] = []
    interfering_layers: list[int] = []
    layer_roles: dict[str, str] = {}

    for l_idx in sorted_layers:
        h_val = harm_by_layer[l_idx]
        if h_val <= anchor_thresh:
            beneficial_anchors.append(l_idx)
            layer_roles[str(l_idx)] = "beneficial_anchor"
        elif h_val >= harm_thresh:
            interfering_layers.append(l_idx)
            layer_roles[str(l_idx)] = "interfering"
        else:
            neutral_layers.append(l_idx)
            layer_roles[str(l_idx)] = "neutral"

    earliest_anchor = min(beneficial_anchors) if beneficial_anchors else None
    downstream_interfering = (
        [l_idx for l_idx in interfering_layers if l_idx > earliest_anchor]
        if earliest_anchor is not None
        else []
    )
    downstream_set = set(downstream_interfering)
    upstream_interfering = [l_idx for l_idx in interfering_layers if l_idx not in downstream_set]
    anchor_to_downstream = {
        str(a_idx): [l_idx for l_idx in interfering_layers if l_idx > a_idx]
        for a_idx in beneficial_anchors
    }

    anchor_gain_nats = float(sum(-harm_by_layer[l_idx] for l_idx in beneficial_anchors))
    interfering_harm_nats = float(sum(harm_by_layer[l_idx] for l_idx in interfering_layers))
    downstream_harm_nats = float(sum(harm_by_layer[l_idx] for l_idx in downstream_interfering))
    net_e0_harm_nats = float(sum(harm_by_layer.values()))

    return {
        "num_layers": max(sorted_layers) + 1,
        "anchor_threshold_nats": anchor_thresh,
        "harm_threshold_nats": harm_thresh,
        "beneficial_anchor_layers": beneficial_anchors,
        "neutral_layers": neutral_layers,
        "interfering_layers": interfering_layers,
        "downstream_interfering_layers": downstream_interfering,
        "upstream_interfering_layers": upstream_interfering,
        "anchor_to_downstream_map": anchor_to_downstream,
        "layer_roles": layer_roles,
        "layer_e0_harm_nats": {str(l_idx): harm_by_layer[l_idx] for l_idx in sorted_layers},
        "anchor_gain_nats": anchor_gain_nats,
        "interfering_harm_nats": interfering_harm_nats,
        "downstream_interfering_harm_nats": downstream_harm_nats,
        "net_e0_harm_nats": net_e0_harm_nats,
        "has_cross_layer_interference": bool(beneficial_anchors and downstream_interfering),
    }


# ---------------------------------------------------------------------------
# 3. Cross-Layer Orthogonal LoRA Projection in Specialist Null-Space
# ---------------------------------------------------------------------------


def _extract_orthonormal_subspace(
    mat_2d: torch.Tensor,
    *,
    max_rank: int | None,
    sv_rel_tol: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Extract orthonormal right-singular basis spanning significant directions of ``mat_2d``."""
    width = int(mat_2d.shape[1])
    _, s_vals, vt = torch.linalg.svd(mat_2d, full_matrices=False)
    if s_vals.numel() == 0 or float(s_vals[0].item()) <= 1e-12:
        return torch.empty((width, 0), dtype=torch.float64), {
            "rank": 0,
            "explained_variance_ratio": 1.0,
            "top_sv": 0.0,
        }

    top_sv = float(s_vals[0].item())
    n_sig = int((s_vals > float(sv_rel_tol) * top_sv).sum().item())
    k_eff = n_sig if max_rank is None else min(int(max_rank), n_sig)
    total_energy = float(s_vals.pow(2).sum().item())
    kept_energy = float(s_vals[:k_eff].pow(2).sum().item())
    basis = vt[:k_eff, :].T.contiguous()
    return basis, {
        "rank": k_eff,
        "explained_variance_ratio": kept_energy / max(total_energy, 1e-24),
        "top_sv": top_sv,
    }


def _make_cancellation_op_dict(
    kind: str,
    dim: int,
    q_cancel: torch.Tensor,
    q_spec: torch.Tensor,
    *,
    g_left: torch.Tensor | None = None,
    anchor_subspace_rank: int = 0,
    anchor_explained_variance_ratio: float = 1.0,
    protected_specialist_rank: int = 0,
    effective_cancel_rank: int = 0,
    used_exact_specialist_nullspace: bool = True,
) -> dict[str, Any]:
    """Build a uniform cancellation operator payload."""
    return {
        "kind": kind,
        "dim": dim,
        "q_cancel": q_cancel,
        "q_spec": q_spec,
        "g_left": g_left,
        "anchor_subspace_rank": int(anchor_subspace_rank),
        "anchor_explained_variance_ratio": float(anchor_explained_variance_ratio),
        "protected_specialist_rank": int(protected_specialist_rank),
        "effective_cancel_rank": int(effective_cancel_rank),
        "used_exact_specialist_nullspace": bool(used_exact_specialist_nullspace),
    }


def _build_cancellation_operator(
    anchor_2d: torch.Tensor,
    spec_2d: torch.Tensor | None,
    *,
    max_rank_proj: int,
    anchor_sv_rel_tol: float,
    specialist_sv_rel_tol: float,
    reg_spec: float = 0.05,
) -> dict[str, Any]:
    """Build exact specialist-nullspace orthogonal basis or full-rank generalized Rayleigh filter."""
    dim = int(anchor_2d.shape[1])
    empty_q = torch.empty((dim, 0), dtype=torch.float64)
    u_anchor, a_info = _extract_orthonormal_subspace(
        anchor_2d, max_rank=max_rank_proj, sv_rel_tol=anchor_sv_rel_tol
    )
    a_rank = int(a_info["rank"])
    a_var = float(a_info["explained_variance_ratio"])
    if a_rank == 0:
        return _make_cancellation_op_dict(
            "noop", dim, empty_q, empty_q, anchor_explained_variance_ratio=a_var
        )

    if spec_2d is None:
        return _make_cancellation_op_dict(
            "orthonormal",
            dim,
            u_anchor,
            empty_q,
            anchor_subspace_rank=a_rank,
            anchor_explained_variance_ratio=a_var,
            effective_cancel_rank=a_rank,
        )

    q_spec, s_info = _extract_orthonormal_subspace(
        spec_2d, max_rank=None, sv_rel_tol=specialist_sv_rel_tol
    )
    spec_rank = int(s_info["rank"])
    if spec_rank == 0:
        return _make_cancellation_op_dict(
            "orthonormal",
            dim,
            u_anchor,
            q_spec,
            anchor_subspace_rank=a_rank,
            anchor_explained_variance_ratio=a_var,
            effective_cancel_rank=a_rank,
        )

    if spec_rank < dim:
        # Double CGS orthogonalization removes anchor overlap with protected specialist subspace
        u_perp = u_anchor - q_spec @ (q_spec.T @ u_anchor)
        u_perp = u_perp - q_spec @ (q_spec.T @ u_perp)
        u_c, s_c, _ = torch.linalg.svd(u_perp, full_matrices=False)
        keep_mask = s_c > 1e-8
        k_eff = int(keep_mask.sum().item())
        if k_eff == 0:
            return _make_cancellation_op_dict(
                "noop",
                dim,
                empty_q,
                q_spec,
                anchor_subspace_rank=a_rank,
                anchor_explained_variance_ratio=a_var,
                protected_specialist_rank=spec_rank,
            )
        q_cancel = u_c[:, keep_mask]
        q_cancel = q_cancel - q_spec @ (q_spec.T @ q_cancel)
        q_cancel, _ = torch.linalg.qr(q_cancel, mode="reduced")
        return _make_cancellation_op_dict(
            "orthonormal",
            dim,
            q_cancel.contiguous(),
            q_spec,
            anchor_subspace_rank=a_rank,
            anchor_explained_variance_ratio=a_var,
            protected_specialist_rank=spec_rank,
            effective_cancel_rank=int(q_cancel.shape[1]),
        )

    # Full-rank specialist fallback: generalized Rayleigh modes where anchor variance exceeds specialist
    m_a = (anchor_2d.T @ anchor_2d) / float(anchor_2d.shape[0])
    m_a = 0.5 * (m_a + m_a.T)
    m_s = (spec_2d.T @ spec_2d) / float(spec_2d.shape[0])
    m_s = 0.5 * (m_s + m_s.T)
    tr_s = max(float(torch.diagonal(m_s).mean().item()), 1e-6)
    b_s = m_s + max(reg_spec * tr_s, 1e-6) * torch.eye(dim, dtype=torch.float64)
    evals_b, evecs_b = torch.linalg.eigh(b_s)
    evals_c = evals_b.clamp_min(1e-8)
    b_half = (evecs_b * evals_c.sqrt().unsqueeze(0)) @ evecs_b.T
    b_inv_half = (evecs_b * evals_c.rsqrt().unsqueeze(0)) @ evecs_b.T

    whitened = b_inv_half @ m_a @ b_inv_half
    whitened = 0.5 * (whitened + whitened.T)
    mu_asc, q_asc = torch.linalg.eigh(whitened)
    mu = mu_asc.flip(0)
    q = q_asc.flip(1)
    k_eff = min(int(max_rank_proj), int((mu > 1.0).sum().item()))
    if k_eff == 0:
        return _make_cancellation_op_dict(
            "noop",
            dim,
            empty_q,
            q_spec,
            anchor_subspace_rank=a_rank,
            anchor_explained_variance_ratio=a_var,
            protected_specialist_rank=spec_rank,
            used_exact_specialist_nullspace=False,
        )

    q_p = q[:, :k_eff]
    excess = mu[:k_eff] - 1.0
    shrink = excess / (1.0 + excess)
    g_left = torch.eye(dim, dtype=torch.float64) - (b_half @ q_p) @ (
        (q_p * shrink.unsqueeze(0)).T @ b_inv_half
    )
    # Bound spectral operator norm <= 1.05 to avoid oblique amplification across depth
    op_norm = float(torch.linalg.matrix_norm(g_left, ord=2).item())
    if op_norm > 1.05:
        alpha_blend = 0.05 / max(op_norm - 1.0, 1e-8)
        g_left = (1.0 - alpha_blend) * torch.eye(dim, dtype=torch.float64) + alpha_blend * g_left
    return _make_cancellation_op_dict(
        "oblique",
        dim,
        u_anchor,
        q_spec,
        g_left=g_left.contiguous(),
        anchor_subspace_rank=a_rank,
        anchor_explained_variance_ratio=a_var,
        protected_specialist_rank=spec_rank,
        effective_cancel_rank=k_eff,
        used_exact_specialist_nullspace=False,
    )


def _project_tensor_with_operator(
    factor: torch.Tensor,
    op: Mapping[str, Any],
    *,
    proj_strength: float,
    inplace: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply cancellation operator on the left of a 2D factor ``(d_out, r)``."""
    w64 = factor.detach().to(device="cpu", dtype=torch.float64)
    q_cancel: torch.Tensor = op["q_cancel"]
    q_spec: torch.Tensor = op["q_spec"]
    kind = str(op["kind"])

    if kind == "noop":
        out = factor if inplace else factor.detach().clone()
        return out, {
            "overlap_before": 0.0,
            "overlap_after": 0.0,
            "delta_fro": 0.0,
            "before_fro": float(torch.linalg.matrix_norm(w64, ord="fro").item()),
            "spec_distortion": 0.0,
        }

    overlap_before = float(torch.linalg.matrix_norm(q_cancel.T @ w64, ord="fro").item())
    if kind == "orthonormal":
        w_new64 = w64 - proj_strength * (q_cancel @ (q_cancel.T @ w64))
    else:
        g_left: torch.Tensor = op["g_left"]
        w_new64 = (1.0 - proj_strength) * w64 + proj_strength * (g_left @ w64)

    overlap_after = float(torch.linalg.matrix_norm(q_cancel.T @ w_new64, ord="fro").item())
    diff64 = w_new64 - w64
    delta_fro = float(torch.linalg.matrix_norm(diff64, ord="fro").item())
    before_fro = float(torch.linalg.matrix_norm(w64, ord="fro").item())
    spec_distortion = (
        float(torch.linalg.matrix_norm(q_spec.T @ diff64, ord="fro").item())
        if q_spec.shape[1] > 0
        else 0.0
    )

    metrics = {
        "overlap_before": overlap_before,
        "overlap_after": overlap_after,
        "delta_fro": delta_fro,
        "before_fro": before_fro,
        "spec_distortion": spec_distortion,
    }
    projected = w_new64.to(device=factor.device, dtype=factor.dtype)
    if inplace:
        factor.copy_(projected)
        return factor, metrics
    return projected, metrics


def _build_intermediate_operator_via_w3(
    op_residual: Mapping[str, Any],
    base_w3: torch.Tensor,
    *,
    expected_hidden: int,
    expected_inter: int,
    max_rank_proj: int,
    anchor_sv_rel_tol: float,
    specialist_sv_rel_tol: float,
) -> dict[str, Any]:
    """Pull residual-space anchor/specialist subspaces back through ``base_w3`` ``(H, I)``."""
    w3_64 = _validate_2d_factor(base_w3, "base_output_proj").detach().to(device="cpu", dtype=torch.float64)
    if w3_64.shape != (expected_hidden, expected_inter):
        raise ValueError(
            f"base_output_proj shape {tuple(w3_64.shape)} must match "
            f"(hidden_dim={expected_hidden}, factor_out_dim={expected_inter})"
        )
    q_cancel_h: torch.Tensor = op_residual["q_cancel"]
    q_spec_h: torch.Tensor = op_residual["q_spec"]
    if op_residual["kind"] == "noop" or q_cancel_h.shape[1] == 0:
        empty_i = torch.empty((expected_inter, 0), dtype=torch.float64)
        return _make_cancellation_op_dict(
            "noop",
            expected_inter,
            empty_i,
            empty_i,
            anchor_subspace_rank=int(op_residual["anchor_subspace_rank"]),
            anchor_explained_variance_ratio=float(op_residual["anchor_explained_variance_ratio"]),
            protected_specialist_rank=int(op_residual["protected_specialist_rank"]),
            used_exact_specialist_nullspace=bool(op_residual["used_exact_specialist_nullspace"]),
        )
    c_cancel = (q_cancel_h.T @ w3_64).contiguous()
    c_spec = (q_spec_h.T @ w3_64).contiguous() if q_spec_h.shape[1] > 0 else None
    return _build_cancellation_operator(
        c_cancel,
        c_spec,
        max_rank_proj=max_rank_proj,
        anchor_sv_rel_tol=anchor_sv_rel_tol,
        specialist_sv_rel_tol=specialist_sv_rel_tol,
    )


def _aggregate_projection_diagnostics(
    op: Mapping[str, Any],
    stats_list: list[tuple[str, dict[str, float]]],
    *,
    skipped_factors: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble standardized diagnostic report from factor-level projection metrics."""
    ov_before = math.sqrt(sum(s["overlap_before"] ** 2 for _, s in stats_list))
    ov_after = math.sqrt(sum(s["overlap_after"] ** 2 for _, s in stats_list))
    delta_fro = math.sqrt(sum(s["delta_fro"] ** 2 for _, s in stats_list))
    before_fro = math.sqrt(sum(s["before_fro"] ** 2 for _, s in stats_list))
    spec_max = max((s["spec_distortion"] for _, s in stats_list), default=0.0)
    reduction_ratio = (
        max(0.0, 1.0 - (ov_after / ov_before)) if ov_before > 1e-12 else 1.0
    )
    modified = [label for label, s in stats_list if s["delta_fro"] > 0.0]
    return {
        "anchor_subspace_rank": int(op["anchor_subspace_rank"]),
        "anchor_explained_variance_ratio": float(op["anchor_explained_variance_ratio"]),
        "protected_specialist_rank": int(op["protected_specialist_rank"]),
        "effective_cancel_rank": int(op["effective_cancel_rank"]),
        "used_exact_specialist_nullspace": bool(op["used_exact_specialist_nullspace"]),
        "anchor_overlap_before": ov_before,
        "anchor_overlap_after": ov_after,
        "anchor_interference_reduction_ratio": reduction_ratio,
        "specialist_distortion_max": spec_max,
        "relative_frobenius_change": delta_fro / max(before_fro, 1e-12),
        "modified_factors": modified,
        "skipped_factors": skipped_factors or [],
    }


@torch.no_grad()
def project_downstream_lora_b_orthogonal_to_anchor(
    anchor_deltas: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor],
    downstream_lora_b: torch.Tensor | nn.Module | Mapping[str, torch.Tensor],
    *,
    protect_specialist_deltas: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor] | None = None,
    max_rank_proj: int = DEFAULT_MAX_RANK_PROJ,
    proj_strength: float = 1.0,
    anchor_sv_rel_tol: float = DEFAULT_SUBSPACE_SV_REL_TOL,
    specialist_sv_rel_tol: float = DEFAULT_SUBSPACE_SV_REL_TOL,
    base_output_proj: torch.Tensor | None = None,
    use_base_output_proj: bool = True,
    expert_indices: Sequence[int] | None = None,
    inplace: bool = False,
) -> tuple[torch.Tensor | nn.Module | dict[str, torch.Tensor], dict[str, Any]]:
    """Project downstream interfering LoRA ``B`` factors orthogonal to upstream anchor deltas.

    Eliminates general-prose destructive overwrites against upstream beneficial anchor deltas
    while keeping the protected specialist residual subspace untouched (``< 1e-5`` distortion).
    Supports:
      - 2D ``torch.Tensor`` (direct residual-width ``(H, r)`` ``w2_B``/``w3_B``, or intermediate
        ``(I, r)`` when ``base_output_proj`` ``(H, I)`` is supplied);
      - ``Mapping[str, torch.Tensor]`` of named LoRA ``B`` matrices;
      - ``CrowdLoRAMoEMLP``, ``CrowdLoRAExpertAdapter``, ``nn.Linear``, or custom adapters
        exposing ``w2_B``, ``w3_B``, or ``lora_B``.
    """
    anchor_2d = _validate_delta_matrix(anchor_deltas, "anchor_deltas")
    hidden_dim = int(anchor_2d.shape[1])
    spec_2d: torch.Tensor | None = None
    if protect_specialist_deltas is not None:
        spec_2d = _validate_delta_matrix(protect_specialist_deltas, "protect_specialist_deltas")
        if int(spec_2d.shape[1]) != hidden_dim:
            raise ValueError(
                f"protect_specialist_deltas width {spec_2d.shape[1]} must match anchor_deltas width {hidden_dim}"
            )

    max_r = _validate_positive_int(max_rank_proj, "max_rank_proj")
    strength = _validate_unit_interval(proj_strength, "proj_strength")
    a_tol = _validate_positive_finite(anchor_sv_rel_tol, "anchor_sv_rel_tol")
    s_tol = _validate_positive_finite(specialist_sv_rel_tol, "specialist_sv_rel_tol")
    if a_tol >= 1.0 or s_tol >= 1.0:
        raise ValueError("singular value relative tolerances must be < 1.0")
    _validate_bool(use_base_output_proj, "use_base_output_proj")
    _validate_bool(inplace, "inplace")

    op_h = _build_cancellation_operator(
        anchor_2d,
        spec_2d,
        max_rank_proj=max_r,
        anchor_sv_rel_tol=a_tol,
        specialist_sv_rel_tol=s_tol,
    )

    def _project_factor_smart(
        factor_t: torch.Tensor,
        label: str,
        w3_proj: torch.Tensor | None,
        stats_acc: list[tuple[str, dict[str, float]]],
        skip_acc: list[str],
        *,
        do_inplace: bool,
    ) -> torch.Tensor:
        _validate_2d_factor(factor_t, label)
        out_dim = int(factor_t.shape[0])
        if out_dim == hidden_dim:
            res_t, st = _project_tensor_with_operator(
                factor_t, op_h, proj_strength=strength, inplace=do_inplace
            )
            stats_acc.append((label, st))
            return res_t
        if w3_proj is not None:
            op_i = _build_intermediate_operator_via_w3(
                op_h,
                w3_proj,
                expected_hidden=hidden_dim,
                expected_inter=out_dim,
                max_rank_proj=max_r,
                anchor_sv_rel_tol=a_tol,
                specialist_sv_rel_tol=s_tol,
            )
            res_t, st = _project_tensor_with_operator(
                factor_t, op_i, proj_strength=strength, inplace=do_inplace
            )
            stats_acc.append((label, st))
            return res_t
        skip_acc.append(label)
        return factor_t if do_inplace else factor_t.detach().clone()

    stats_list: list[tuple[str, dict[str, float]]] = []
    skipped: list[str] = []

    if isinstance(downstream_lora_b, torch.Tensor):
        _validate_2d_factor(downstream_lora_b, "downstream_lora_b")
        out_dim = int(downstream_lora_b.shape[0])
        if out_dim != hidden_dim and base_output_proj is None:
            raise ValueError(
                f"downstream_lora_b output dimension {out_dim} does not match anchor_deltas width "
                f"{hidden_dim}; pass base_output_proj=(H, {out_dim}) for intermediate factors"
            )
        proj_t = _project_factor_smart(
            downstream_lora_b,
            "tensor",
            base_output_proj,
            stats_list,
            skipped,
            do_inplace=inplace,
        )
        return proj_t, _aggregate_projection_diagnostics(op_h, stats_list, skipped_factors=skipped)

    if isinstance(downstream_lora_b, Mapping):
        if not downstream_lora_b:
            raise ValueError("downstream_lora_b mapping must not be empty")
        out_map: dict[str, torch.Tensor] = {}
        for k, val in downstream_lora_b.items():
            label = str(k)
            _validate_2d_factor(val, f"downstream_lora_b[{label!r}]")
            if int(val.shape[0]) != hidden_dim and base_output_proj is None:
                raise ValueError(
                    f"downstream_lora_b[{label!r}] output dim {val.shape[0]} != anchor width {hidden_dim}"
                )
            out_map[label] = _project_factor_smart(
                val, label, base_output_proj, stats_list, skipped, do_inplace=inplace
            )
        return out_map, _aggregate_projection_diagnostics(op_h, stats_list, skipped_factors=skipped)

    if not isinstance(downstream_lora_b, nn.Module):
        raise TypeError(
            "downstream_lora_b must be a 2D torch.Tensor, Mapping[str, Tensor], or nn.Module; "
            f"got {type(downstream_lora_b).__name__}"
        )

    target_mod = downstream_lora_b if inplace else copy.deepcopy(downstream_lora_b)
    if isinstance(target_mod, CrowdLoRAMoEMLP) or hasattr(target_mod, "experts"):
        experts_dict = getattr(target_mod, "experts", None)
        if experts_dict is None or not hasattr(experts_dict, "items") or len(experts_dict) == 0:
            raise ValueError("MoE module must expose a non-empty 'experts' mapping")
        w3_ref = base_output_proj if use_base_output_proj else None
        base_mlp = getattr(target_mod, "base_mlp", None)
        if use_base_output_proj and w3_ref is None and base_mlp is not None and hasattr(base_mlp, "w3"):
            w3_ref = base_mlp.w3.weight
        allowed_set = (
            {_validate_nonneg_int(idx, "expert_index") for idx in expert_indices}
            if expert_indices is not None
            else None
        )
        for e_key, exp_mod in sorted(experts_dict.items(), key=lambda kv: int(kv[0])):
            if allowed_set is not None and int(e_key) not in allowed_set:
                continue
            for attr in ("w3_B", "w2_B", "lora_B"):
                param = getattr(exp_mod, attr, None)
                if isinstance(param, torch.Tensor):
                    _project_factor_smart(
                        param,
                        f"expert_{e_key}.{attr}",
                        w3_ref,
                        stats_list,
                        skipped,
                        do_inplace=True,
                    )
        if not stats_list and not skipped:
            raise ValueError("No matching LoRA B factors found on target MoE module")
        return target_mod, _aggregate_projection_diagnostics(op_h, stats_list, skipped_factors=skipped)

    if isinstance(target_mod, nn.Linear):
        if int(target_mod.weight.shape[0]) != hidden_dim:
            raise ValueError(
                f"nn.Linear out_features {target_mod.weight.shape[0]} must match anchor width {hidden_dim}"
            )
        _project_factor_smart(
            target_mod.weight, "weight", None, stats_list, skipped, do_inplace=True
        )
        return target_mod, _aggregate_projection_diagnostics(op_h, stats_list, skipped_factors=skipped)

    found_attrs = [
        attr
        for attr in ("w3_B", "w2_B", "lora_B")
        if isinstance(getattr(target_mod, attr, None), torch.Tensor)
    ]
    if not found_attrs:
        raise ValueError(
            f"{type(downstream_lora_b).__name__} must expose 'w2_B', 'w3_B', 'lora_B', 'experts', or 'weight'"
        )
    for attr in found_attrs:
        param = getattr(target_mod, attr)
        if int(param.shape[0]) != hidden_dim and base_output_proj is None and len(found_attrs) == 1:
            raise ValueError(
                f"{attr} output dimension {param.shape[0]} does not match anchor width {hidden_dim}; "
                "pass base_output_proj for intermediate factors"
            )
        _project_factor_smart(param, attr, base_output_proj, stats_list, skipped, do_inplace=True)
    return target_mod, _aggregate_projection_diagnostics(op_h, stats_list, skipped_factors=skipped)


# ---------------------------------------------------------------------------
# 4. End-to-End Cross-Layer Surgery + ANSE Gate & Atomic Bitwise Rollback
# ---------------------------------------------------------------------------


def _select_balanced_general_and_spec_ids(
    ids_all: torch.Tensor,
    is_general: np.ndarray,
    expert_ids: Sequence[str],
    *,
    max_general: int,
    max_spec: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Select balanced general and specialist probe sequences for residual delta collection."""
    gen_idx = [i for i, flag in enumerate(is_general) if bool(flag)][:max_general]
    ids_gen = ids_all[gen_idx]

    spec_by_dom: dict[str, list[int]] = {}
    for idx, flag in enumerate(is_general):
        if not bool(flag):
            spec_by_dom.setdefault(expert_ids[idx], []).append(idx)

    spec_idx: list[int] = []
    if spec_by_dom:
        queues = list(spec_by_dom.values())
        round_idx = 0
        while len(spec_idx) < max_spec:
            added = False
            for q in queues:
                if round_idx < len(q):
                    spec_idx.append(q[round_idx])
                    added = True
                    if len(spec_idx) >= max_spec:
                        break
            if not added:
                break
            round_idx += 1
    ids_spec = ids_all[spec_idx] if spec_idx else None
    return ids_gen, ids_spec


def _compute_domain_nlls_from_chunks(
    model: nn.Module,
    ids_eval: torch.Tensor,
    expert_ids: Sequence[str],
    cal_ln_w: torch.Tensor,
    cal_ln_b: torch.Tensor,
    ln_eps: float,
    *,
    score_from: int,
    batch_size: int,
) -> tuple[dict[str, float] | None, float | None, bool]:
    """Evaluate exact per-domain causal mid-row NLLs, returning ``(domain_nlls, nll_mean, finite)``."""
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
        return None, None, False
    losses = per_chunk_causal_midrow_nll(
        h_pre,
        cal_ln_w,
        cal_ln_b,
        ln_eps,
        model.head.weight,
        ids_eval,
        score_from=score_from,
        seq_batch_size=batch_size,
    )
    if not bool(torch.isfinite(losses).all().item()):
        return None, None, False

    vals = losses.cpu().tolist()
    grouped: dict[str, list[float]] = {}
    for eid, loss_val in zip(expert_ids, vals, strict=True):
        grouped.setdefault(str(eid), []).append(float(loss_val))
    domain_nlls = {dom: float(sum(xs) / len(xs)) for dom, xs in grouped.items()}
    return domain_nlls, float(sum(vals) / len(vals)), True


def _parse_external_eval_domain_nll(
    raw_eval: Any,
    expected_domains: set[str],
) -> tuple[dict[str, float] | None, float | None, bool]:
    """Normalize and finiteness-check custom ``eval_fn(model)`` output."""
    if not isinstance(raw_eval, Mapping) or not raw_eval:
        return None, None, False
    source = raw_eval.get("domain_nll", raw_eval)
    if not isinstance(source, Mapping) or set(source.keys()) != expected_domains:
        return None, None, False
    cleaned: dict[str, float] = {}
    for dom in sorted(expected_domains):
        raw_v = source[dom]
        if isinstance(raw_v, bool) or not isinstance(raw_v, (int, float)):
            return None, None, False
        val = float(raw_v)
        if not math.isfinite(val) or val <= 0.0:
            return None, None, False
        cleaned[dom] = val
    nll_mean = float(sum(cleaned.values()) / len(cleaned))
    return cleaned, nll_mean, True


def _evaluate_anse_guard(
    ref_domain_nll: Mapping[str, float],
    cand_domain_nll: Mapping[str, float] | None,
    *,
    cand_nll_mean: float | None,
    cand_finite: bool,
    cfg: CrossLayerResidualConfig,
) -> dict[str, Any]:
    """Compute per-domain NLL drift, worst-domain <= +1.0% guard verdict, and ANSE energy."""
    ref_mean = float(sum(ref_domain_nll.values()) / len(ref_domain_nll))
    if not cand_finite or cand_domain_nll is None or cand_nll_mean is None:
        return {
            "finite": False,
            "nll_mean": None,
            "ref_nll_mean": ref_mean,
            "worst_domain_id": None,
            "worst_domain_drift": None,
            "worst_domain_drift_pct": None,
            "general_ratio": None,
            "guard_ok": False,
            "anse_energy": float(cfg.maximum_pain),
            "per_domain": {},
        }

    per_domain: dict[str, dict[str, Any]] = {}
    worst_dom = ""
    worst_ratio = -math.inf
    for dom in sorted(ref_domain_nll.keys()):
        b_val = float(ref_domain_nll[dom])
        c_val = float(cand_domain_nll[dom])
        ratio = c_val / max(b_val, 1e-12)
        drift = ratio - 1.0
        dom_guard = drift <= cfg.max_domain_drift + 1e-12
        per_domain[dom] = {
            "ref_nll": b_val,
            "cand_nll": c_val,
            "mean_diff": c_val - b_val,
            "ratio": ratio,
            "drift_pct": 100.0 * drift,
            "guard_ok": bool(dom_guard),
        }
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_dom = dom

    worst_drift = worst_ratio - 1.0
    guard_ok = all(bool(info["guard_ok"]) for info in per_domain.values())
    gen_keys = [dom for dom in sorted(per_domain.keys()) if dom.startswith(cfg.general_prefix)]
    general_ratio = max(float(per_domain[k]["ratio"]) for k in gen_keys) if gen_keys else 1.0
    penalty = (
        float(cfg.maximum_pain)
        if not guard_ok
        else (float(cfg.soft_margin_penalty) if general_ratio > cfg.anse_safe_e0_ratio else 0.0)
    )
    anse_energy = (float(cand_nll_mean) - ref_mean) + penalty
    return {
        "finite": True,
        "nll_mean": float(cand_nll_mean),
        "ref_nll_mean": ref_mean,
        "worst_domain_id": worst_dom,
        "worst_domain_drift": worst_drift,
        "worst_domain_drift_pct": 100.0 * worst_drift,
        "general_ratio": general_ratio,
        "guard_ok": bool(guard_ok),
        "anse_energy": float(anse_energy),
        "per_domain": per_domain,
    }


def _restore_model_snapshot_bitwise(
    model: nn.Module,
    snapshot: Mapping[str, torch.Tensor],
    expected_sha256: str,
) -> str:
    """Atomically restore ``model`` parameters/buffers from ``snapshot`` and verify SHA-256."""
    state = model.state_dict()
    for key, snap_t in snapshot.items():
        if key in state:
            state[key].copy_(snap_t.to(device=state[key].device, dtype=state[key].dtype))
    restored_sha = compute_module_state_sha256(model)
    if restored_sha != expected_sha256:
        raise RuntimeError(
            f"Atomic rollback SHA-256 mismatch: expected {expected_sha256}, got {restored_sha}"
        )
    return restored_sha


def _build_surgery_receipt(
    *,
    accepted: bool,
    rolled_back: bool,
    status: str,
    parent_sha256: str,
    candidate_sha256: str,
    active_sha256: str,
    anchor_layers: list[int],
    partition: Mapping[str, Any],
    anchor_preserved: bool,
    anchor_sha_before: dict[str, str],
    anchor_sha_after: dict[str, str],
    parent_nll_mean: float,
    active_nll_mean: float,
    parent_anse: Mapping[str, Any],
    cand_anse: Mapping[str, Any],
    active_energy: float,
    delta_energy: float,
    projection_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble and SHA-256-sign a canonical cross-layer residual surgery receipt."""
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "receipt_object": RECEIPT_OBJECT,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "status": status,
        "parent_sha256": parent_sha256,
        "candidate_sha256": candidate_sha256,
        "active_sha256": active_sha256,
        "anchor_layers": anchor_layers,
        "neutral_layers": list(partition["neutral_layers"]),
        "interfering_layers": list(partition["interfering_layers"]),
        "downstream_interfering_layers": list(partition["downstream_interfering_layers"]),
        "anchor_layers_bitwise_preserved": anchor_preserved,
        "anchor_layer_sha256_before": anchor_sha_before,
        "anchor_layer_sha256_after": anchor_sha_after,
        "anchor_gain_nats": float(partition["anchor_gain_nats"]),
        "interfering_harm_nats": float(partition["interfering_harm_nats"]),
        "nll_mean_before": parent_nll_mean,
        "nll_mean": cand_anse["nll_mean"],
        "nll_mean_active": active_nll_mean,
        "anse_energy_parent": float(parent_anse["anse_energy"]),
        "anse_energy": float(cand_anse["anse_energy"]),
        "anse_energy_active": active_energy,
        "delta_energy": delta_energy,
        "worst_domain_id": cand_anse["worst_domain_id"],
        "worst_domain_drift": cand_anse["worst_domain_drift"],
        "worst_domain_drift_pct": cand_anse["worst_domain_drift_pct"],
        "guard_ok": bool(cand_anse["guard_ok"]),
        "per_domain": cand_anse["per_domain"],
        "layer_projection_reports": projection_reports,
        "partition": partition,
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


def _evaluate_model_domains(
    model: nn.Module,
    ids_eval: torch.Tensor,
    eval_eids: Sequence[str],
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    ln_eps: float,
    *,
    expected_domains: set[str],
    cfg: CrossLayerResidualConfig,
    eval_fn: Callable[[nn.Module], Mapping[str, Any]] | None,
) -> tuple[dict[str, float] | None, float | None, bool]:
    """Evaluate per-domain NLL via custom ``eval_fn`` or exact causal mid-row NLL."""
    if eval_fn is not None:
        try:
            return _parse_external_eval_domain_nll(eval_fn(model), expected_domains)
        except Exception:
            return None, None, False
    return _compute_domain_nlls_from_chunks(
        model,
        ids_eval,
        eval_eids,
        ln_w.to(ids_eval.device),
        ln_b.to(ids_eval.device),
        ln_eps,
        score_from=cfg.score_from,
        batch_size=cfg.batch_size,
    )


def _collect_and_project_downstream_layers(
    model: nn.Module,
    ids_probe: torch.Tensor,
    is_gen_probe: np.ndarray,
    probe_eids: Sequence[str],
    *,
    anchor_layers: list[int],
    downstream_layers: list[int],
    cfg: CrossLayerResidualConfig,
    target_experts: Sequence[int] | None,
    h_dim: int,
) -> list[dict[str, Any]]:
    """Trace probe activations sequentially and project downstream interfering LoRA B factors."""
    ids_gen, ids_spec = _select_balanced_general_and_spec_ids(
        ids_probe,
        is_gen_probe,
        probe_eids,
        max_general=cfg.max_general_probe_chunks,
        max_spec=cfg.max_spec_probe_chunks,
    )
    start_pos = max(0, cfg.score_from - 1)
    cos = model.cos_cached.to(ids_gen.device)
    sin = model.sin_cached.to(ids_gen.device)

    x_gen = torch.cat(
        [
            compute_embeddings_batch(model, ids_gen[b0 : b0 + cfg.batch_size])
            for b0 in range(0, int(ids_gen.shape[0]), cfg.batch_size)
        ],
        dim=0,
    )
    x_spec = (
        torch.cat(
            [
                compute_embeddings_batch(model, ids_spec[b0 : b0 + cfg.batch_size])
                for b0 in range(0, int(ids_spec.shape[0]), cfg.batch_size)
            ],
            dim=0,
        )
        if ids_spec is not None
        else None
    )

    anchor_set = set(anchor_layers)
    downstream_set = set(downstream_layers)
    anchor_prose_deltas: list[torch.Tensor] = []
    spec_protect_deltas: list[torch.Tensor] = []
    projection_reports: list[dict[str, Any]] = []

    for l_idx, blk in enumerate(model.blocks):
        blk_dtype = blk.ln1.weight.dtype
        xm_g, mi_g = forward_block_pre_mlp(blk, x_gen, cos, sin)
        if l_idx in anchor_set:
            dense_g = forward_block_dense_base_mlp(blk, mi_g)
            moe_g = blk.mlp(mi_g.to(dtype=blk_dtype)).to(torch.float32)
            d_anchor = (moe_g - dense_g)[:, start_pos:-1, :].reshape(-1, h_dim)
            if float(d_anchor.abs().max().item()) > 1e-12:
                anchor_prose_deltas.append(d_anchor)

        xm_s = mi_s = None
        if x_spec is not None:
            xm_s, mi_s = forward_block_pre_mlp(blk, x_spec, cos, sin)
            dense_s = forward_block_dense_base_mlp(blk, mi_s)
            moe_s = blk.mlp(mi_s.to(dtype=blk_dtype)).to(torch.float32)
            d_spec = (moe_s - dense_s)[:, start_pos:-1, :].reshape(-1, h_dim)
            if float(d_spec.abs().max().item()) > 1e-12:
                spec_protect_deltas.append(d_spec)

        if l_idx in downstream_set and anchor_prose_deltas:
            anchor_cat = torch.cat(anchor_prose_deltas, dim=0)
            spec_cat = (
                torch.cat(spec_protect_deltas, dim=0)
                if (cfg.protect_specialist_subspace and spec_protect_deltas)
                else None
            )
            _, proj_rep = project_downstream_lora_b_orthogonal_to_anchor(
                anchor_cat,
                blk.mlp,
                protect_specialist_deltas=spec_cat,
                max_rank_proj=cfg.max_rank_proj,
                proj_strength=cfg.proj_strength,
                anchor_sv_rel_tol=cfg.anchor_sv_rel_tol,
                specialist_sv_rel_tol=cfg.specialist_sv_rel_tol,
                use_base_output_proj=cfg.project_w2_via_base_w3,
                expert_indices=target_experts,
                inplace=True,
            )
            proj_rep["layer"] = l_idx
            projection_reports.append(proj_rep)

        x_gen = forward_block_post_mlp(blk, xm_g, mi_g)
        if x_spec is not None and xm_s is not None and mi_s is not None:
            x_spec = forward_block_post_mlp(blk, xm_s, mi_s)

    return projection_reports


@torch.no_grad()
def optimize_cross_layer_residual_surgery(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    selection_chunks: Sequence[Mapping[str, Any]] | None = None,
    buckets: Mapping[str, torch.Tensor] | None = None,
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
    cal_ln_w: torch.Tensor | None = None,
    cal_ln_b: torch.Tensor | None = None,
    layer_attribution: Sequence[Mapping[str, Any]] | Mapping[Any, Any] | Sequence[float] | None = None,
    baseline_domain_nll: Mapping[str, float] | None = None,
    config: CrossLayerResidualConfig | None = None,
    target_experts: Sequence[int] | None = None,
    anchor_threshold_nats: float = DEFAULT_ANCHOR_THRESHOLD_NATS,
    harm_threshold_nats: float = DEFAULT_HARM_THRESHOLD_NATS,
    max_rank_proj: int = DEFAULT_MAX_RANK_PROJ,
    proj_strength: float = 1.0,
    general_prefix: str = GENERAL_EXPERT_PREFIX,
    score_from: int = DEFAULT_SCORE_FROM,
    batch_size: int = 24,
    max_domain_drift: float = DEFAULT_MAX_DOMAIN_DRIFT,
    eval_fn: Callable[[nn.Module], Mapping[str, Any]] | None = None,
    inplace: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Run training-free cross-layer residual null-space surgery under the ANSE acceptance gate.

    Protects beneficial upstream anchor layers bitwise, projects downstream interfering experts'
    LoRA ``B`` factors orthogonal to upstream anchor residual deltas while preserving specialist
    subspaces, and commits the surgery if and only if ``ANSE Delta E < 0`` and worst-domain drift
    ``<= +1.0%`` (else atomic bitwise rollback with SHA-256 verification).
    """
    _validate_model_compatible(model)
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError("eval_fn must be callable or None")
    if (cal_ln_w is None) != (cal_ln_b is None):
        raise ValueError("cal_ln_w and cal_ln_b must either both be provided or both be None")

    if config is not None:
        if not isinstance(config, CrossLayerResidualConfig):
            raise TypeError(
                f"config must be a CrossLayerResidualConfig or None, got {type(config).__name__}"
            )
        cfg = config
    else:
        cfg = CrossLayerResidualConfig(
            anchor_threshold_nats=anchor_threshold_nats,
            harm_threshold_nats=harm_threshold_nats,
            max_rank_proj=max_rank_proj,
            proj_strength=proj_strength,
            general_prefix=general_prefix,
            score_from=score_from,
            batch_size=batch_size,
            max_domain_drift=max_domain_drift,
            inplace=inplace,
        )

    work_model = model if cfg.inplace else copy.deepcopy(model)
    if buckets is not None:
        nfrag = max(1, sum(1 for k in buckets if k.startswith("router_frag_")))
        loader = load_buckets_fn or (
            lambda m, b: load_moe_trainable_buckets(m, b, num_router_fragments=nfrag)
        )
        loader(work_model, buckets)

    vocab_size = int(work_model.head.weight.shape[0])
    h_dim = int(work_model.ln.weight.numel())
    if cal_ln_w is not None and (not isinstance(cal_ln_w, torch.Tensor) or not isinstance(cal_ln_b, torch.Tensor)):
        raise TypeError("cal_ln_w and cal_ln_b must be torch.Tensor instances")
    ln_w = (
        work_model.ln.weight.detach().to("cpu", dtype=torch.float32)
        if cal_ln_w is None
        else cal_ln_w.detach().to("cpu", dtype=torch.float32)
    )
    ln_b = (
        work_model.ln.bias.detach().to("cpu", dtype=torch.float32)
        if cal_ln_b is None
        else cal_ln_b.detach().to("cpu", dtype=torch.float32)
    )
    if ln_w.shape != (h_dim,) or ln_b.shape != (h_dim,):
        raise ValueError(f"cal_ln_w and cal_ln_b must have shape ({h_dim},)")
    if not bool(torch.isfinite(ln_w).all().item()) or not bool(torch.isfinite(ln_b).all().item()):
        raise ValueError("cal_ln_w and cal_ln_b must contain only finite values")
    ln_eps = float(getattr(work_model.ln, "eps", 1e-5))

    ids_probe, is_gen_probe, probe_eids = _validate_eval_chunks(
        probe_chunks,
        general_prefix=cfg.general_prefix,
        vocab_size=vocab_size,
        score_from=cfg.score_from,
        name="probe_chunks",
    )
    eval_source = selection_chunks if selection_chunks is not None else probe_chunks
    ids_eval, _, eval_eids = _validate_eval_chunks(
        eval_source,
        general_prefix=cfg.general_prefix,
        vocab_size=vocab_size,
        score_from=cfg.score_from,
        name="selection_chunks" if selection_chunks is not None else "probe_chunks",
    )

    was_training = work_model.training
    work_model.eval()
    try:
        parent_sha256 = compute_module_state_sha256(work_model)
        parent_snapshot = {
            k: v.detach().to(device="cpu").clone() for k, v in work_model.state_dict().items()
        }
        parent_layer_sha = {
            str(i): compute_module_state_sha256(blk) for i, blk in enumerate(work_model.blocks)
        }

        parent_dom_nll, parent_nll_mean, parent_finite = _evaluate_model_domains(
            work_model,
            ids_eval,
            eval_eids,
            ln_w,
            ln_b,
            ln_eps,
            expected_domains=set(eval_eids),
            cfg=cfg,
            eval_fn=eval_fn,
        )
        if not parent_finite or parent_dom_nll is None or parent_nll_mean is None:
            raise ValueError("Parent model evaluation returned invalid or non-finite NLLs")

        if baseline_domain_nll is not None:
            ref_dom_nll, _, ref_finite = _parse_external_eval_domain_nll(
                baseline_domain_nll, set(parent_dom_nll.keys())
            )
            if not ref_finite or ref_dom_nll is None:
                raise ValueError("baseline_domain_nll must contain finite positive values for all eval domains")
        else:
            ref_dom_nll = parent_dom_nll

        parent_anse = _evaluate_anse_guard(
            ref_dom_nll,
            parent_dom_nll,
            cand_nll_mean=parent_nll_mean,
            cand_finite=True,
            cfg=cfg,
        )

        attr_source: Any = (
            layer_attribution
            if layer_attribution is not None
            else compute_per_layer_moe_bypass_attribution(
                work_model,
                probe_chunks,
                cal_ln_w=ln_w,
                cal_ln_b=ln_b,
                score_from=cfg.score_from,
                max_general_probe_chunks=cfg.max_general_probe_chunks,
                max_spec_probe_chunks=cfg.max_spec_probe_chunks,
                batch_size=cfg.batch_size,
                general_prefix=cfg.general_prefix,
            )
        )
        partition = identify_beneficial_anchor_and_interfering_layers(
            attr_source,
            anchor_threshold_nats=cfg.anchor_threshold_nats,
            harm_threshold_nats=cfg.harm_threshold_nats,
        )
        anchor_layers: list[int] = list(partition["beneficial_anchor_layers"])
        downstream_layers: list[int] = list(partition["downstream_interfering_layers"])
        anchor_sha_before = {str(l_i): parent_layer_sha[str(l_i)] for l_i in anchor_layers}

        if not downstream_layers:
            return work_model, _build_surgery_receipt(
                accepted=False,
                rolled_back=False,
                status="NOOP_NO_DOWNSTREAM_INTERFERENCE",
                parent_sha256=parent_sha256,
                candidate_sha256=parent_sha256,
                active_sha256=parent_sha256,
                anchor_layers=anchor_layers,
                partition=partition,
                anchor_preserved=True,
                anchor_sha_before=anchor_sha_before,
                anchor_sha_after=anchor_sha_before,
                parent_nll_mean=parent_nll_mean,
                active_nll_mean=parent_nll_mean,
                parent_anse=parent_anse,
                cand_anse=parent_anse,
                active_energy=float(parent_anse["anse_energy"]),
                delta_energy=0.0,
                projection_reports=[],
            )

        projection_reports = _collect_and_project_downstream_layers(
            work_model,
            ids_probe,
            is_gen_probe,
            probe_eids,
            anchor_layers=anchor_layers,
            downstream_layers=downstream_layers,
            cfg=cfg,
            target_experts=target_experts,
            h_dim=h_dim,
        )

        anchor_sha_after = {
            str(l_i): compute_module_state_sha256(work_model.blocks[l_i]) for l_i in anchor_layers
        }
        anchor_preserved = anchor_sha_before == anchor_sha_after
        if cfg.protect_anchor_layers_bitwise and not anchor_preserved:
            raise RuntimeError("Beneficial anchor layer bitwise preservation invariant violated")

        candidate_sha256 = compute_module_state_sha256(work_model)
        cand_dom_nll, cand_nll_mean, cand_finite = _evaluate_model_domains(
            work_model,
            ids_eval,
            eval_eids,
            ln_w,
            ln_b,
            ln_eps,
            expected_domains=set(ref_dom_nll.keys()),
            cfg=cfg,
            eval_fn=eval_fn,
        )
        cand_anse = _evaluate_anse_guard(
            ref_dom_nll,
            cand_dom_nll,
            cand_nll_mean=cand_nll_mean,
            cand_finite=cand_finite,
            cfg=cfg,
        )
        delta_energy = float(cand_anse["anse_energy"]) - float(parent_anse["anse_energy"])
        accepted = bool(
            cand_anse["finite"]
            and cand_anse["guard_ok"]
            and (delta_energy < -cfg.min_energy_improvement)
        )

        if accepted:
            rolled_back = False
            status = "COMMITTED"
            active_sha256 = candidate_sha256
            active_nll_mean = float(cand_anse["nll_mean"])
            active_energy = float(cand_anse["anse_energy"])
        else:
            rolled_back = True
            active_sha256 = _restore_model_snapshot_bitwise(
                work_model, parent_snapshot, parent_sha256
            )
            active_nll_mean = parent_nll_mean
            active_energy = float(parent_anse["anse_energy"])
            if not cand_anse["finite"]:
                status = "ROLLED_BACK_NON_FINITE"
            elif not cand_anse["guard_ok"]:
                status = "ROLLED_BACK_GUARD_VIOLATION"
            else:
                status = "ROLLED_BACK_NO_ENERGY_IMPROVEMENT"

        return work_model, _build_surgery_receipt(
            accepted=accepted,
            rolled_back=rolled_back,
            status=status,
            parent_sha256=parent_sha256,
            candidate_sha256=candidate_sha256,
            active_sha256=active_sha256,
            anchor_layers=anchor_layers,
            partition=partition,
            anchor_preserved=anchor_preserved,
            anchor_sha_before=anchor_sha_before,
            anchor_sha_after=anchor_sha_after,
            parent_nll_mean=parent_nll_mean,
            active_nll_mean=active_nll_mean,
            parent_anse=parent_anse,
            cand_anse=cand_anse,
            active_energy=active_energy,
            delta_energy=delta_energy,
            projection_reports=projection_reports,
        )
    finally:
        work_model.train(was_training)


__all__ = [
    "DEFAULT_ANCHOR_THRESHOLD_NATS",
    "DEFAULT_ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_ANSE_SOFT_MARGIN_PENALTY",
    "DEFAULT_HARM_THRESHOLD_NATS",
    "DEFAULT_MAX_DOMAIN_DRIFT",
    "DEFAULT_MAX_RANK_PROJ",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_SUBSPACE_SV_REL_TOL",
    "GENERAL_EXPERT_PREFIX",
    "RECEIPT_OBJECT",
    "SCHEMA_VERSION",
    "CrossLayerResidualConfig",
    "canonical_json_bytes",
    "compute_module_state_sha256",
    "compute_payload_sha256",
    "identify_beneficial_anchor_and_interfering_layers",
    "optimize_cross_layer_residual_surgery",
    "project_downstream_lora_b_orthogonal_to_anchor",
    "verify_cross_layer_receipt_provenance",
]

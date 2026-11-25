# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Self-Healing Discarded Checkpoint Rescuer via Causal Layer Attribution and Null-Space Repair.

Implements training-free closed-form rescue for discarded fine-tuned / DiLoCo Sparse MoE
and dense checkpoints (``CrowdV1WithEngram``, ``CrowdLoRAMoEMLP``, ``CrowdFineGrainedMoEMLP``,
``CrowdRuntimeMLP``, and structured block dicts), inspired by *Language Models Can Rewrite
Their Own Weights* and the empirical ``T3.safetensors`` rescue (where a checkpoint achieving
``-20.25%`` specialist astronomy NLL gain failed the ANSE ``E0 <= +1.0%`` general guard at
``E0 = +17.53%``):

1. ``triage_drifted_checkpoint_blocks``:
   Screens every layer/expert block for non-finite values and spectral/norm explosions (quarantine),
   and partitions healthy blocks via causal ``(general_harm_nats, specialist_gain_nats)`` into:
   - ``protected_specialist_anchor`` (helpful or non-corrupting specialist blocks, preserved bitwise),
   - ``salvageable_dual_active`` (specialist breakthroughs with prose drift, projected into ``P_0``),
   - ``prose_corrupting`` (pure general-prose corruptors without specialist gain, zero-reverted).
2. ``repair_drifted_blocks_in_nullspace``:
   Projects dual-active block deltas strictly into the AlphaEdit general-prose null-space ``P_0``,
   zero-reverts pure corruptors and quarantined blocks bitwise to ``base``, preserves specialist
   anchors bitwise to ``drifted``, and calibrates final LayerNorm interpolation ``fold_ln_lambda``.
3. ``execute_verified_checkpoint_rescue``:
   Orchestrates causal attribution, ``P_0`` projector synthesis, triage, closed-form repair,
   and ANSE verification: commits rescue if and only if general drift drops from ``>+10%`` to
   ``<= +1.0%`` while retaining ``>= 70%`` of specialist NLL gain with ``ANSE Delta E < 0`` vs
   both ``base`` and ``raw_drifted``; otherwise enforces ANSE Maximum Pain atomic bitwise
   rollback with a canonical SHA-256 provenance receipt.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from hivemind.swarm.byzantine_wsp_delta_aggregator import (
    compute_payload_sha256,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdFineGrainedMoEMLP,
    CrowdLoRAMoEMLP,
)
from hivemind.swarm.lora_svd_mode_surgery import decompose_lora_qr_svd
from hivemind.swarm.tangent_task_vector_surgeon import (
    compute_prose_nullspace_projector_scaled,
)
from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_nll_from_pre_ln,
)

RESCUER_SCHEMA: str = "hivemind.self_healing_checkpoint_rescuer/1"
RESCUER_RECEIPT_OBJECT: str = "hivemind.self_healing_checkpoint_rescue_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_ANCHOR_FLOOR_NATS: float = -0.01
DEFAULT_CORRUPTION_CEILING_NATS: float = 0.005
DEFAULT_FOLD_LN_LAMBDA: float = 0.8
DEFAULT_MIN_RAW_DRIFT_RATIO: float = 1.10
DEFAULT_MAX_GENERAL_DRIFT_RATIO: float = 1.01
DEFAULT_MIN_SPECIALIST_RETENTION_RATIO: float = 0.70
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05

DEFAULT_EXTREME_WEIGHT_THRESHOLD: float = 1e4
DEFAULT_SPECTRAL_SPIKE_RATIO_THRESHOLD: float = 25.0
DEFAULT_MAX_SPECTRAL_NORM_INFLATION: float = 20.0
DEFAULT_SINGLE_MODE_SPIKE_NORM: float = 50.0
DEFAULT_ISOLATED_JUMP_MULTIPLIER: float = 500.0
DEFAULT_ISOLATED_JUMP_FLOOR: float = 100.0
DEFAULT_NULLSPACE_SV_REL_TOL: float = 1e-3
DEFAULT_NULLSPACE_VARIANCE_THRESHOLD: float = 0.999
DEFAULT_NULLSPACE_EIG_THRESHOLD: float = 0.5
DEFAULT_SCORE_FROM: int = 64
DEFAULT_BATCH_SIZE: int = 24

VALID_ROLLBACK_TARGETS: frozenset[str] = frozenset({"base", "drifted"})
VALID_PROJECTION_SIDES: frozenset[str] = frozenset({"auto", "input", "output"})

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_LAYER_ID_RE = re.compile(r"(?:^|\.)(?:blocks|layers|layer)[._]?(\d+)(?:\.|$|_)")
_SINGLE_TENSOR_KEY = "__tensor__"
_SWIGLU_PROJECTIONS: tuple[str, ...] = ("w1", "w2", "w3")
_FINAL_LN_IDS: frozenset[str] = frozenset({"ln", "final_ln", "model.ln", "norm", "model.norm"})


# ---------------------------------------------------------------------------
# 1. Validation, Configuration & NaN-Safe Provenance Helpers
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


def _validate_unit_interval(value: Any, name: str, *, allow_zero: bool = True) -> float:
    """Validate a finite number inside ``[0.0, 1.0]`` (or ``(0.0, 1.0]``)."""
    out = _validate_nonneg_finite(value, name)
    if not allow_zero and out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    if out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    return out


def _raw_bit_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return an integer bit-pattern view for NaN-safe bitwise hashing."""
    t_cpu = tensor.detach().cpu().contiguous()
    elem_bytes = t_cpu.element_size()
    if elem_bytes == 1:
        return t_cpu.view(torch.uint8)
    if elem_bytes == 2:
        return t_cpu.view(torch.int16)
    if elem_bytes == 4:
        return t_cpu.view(torch.int32)
    return t_cpu.view(torch.int64)


def compute_checkpoint_sha256(source: nn.Module | Mapping[str, Any]) -> str:
    """Compute a deterministic NaN-safe bitwise SHA-256 digest over a model or block mapping."""
    if isinstance(source, nn.Module):
        sd = source.state_dict()
        if not sd:
            raise ValueError("module state_dict must not be empty")
        flat_items = sorted(sd.items())
    elif isinstance(source, Mapping):
        if not source:
            raise ValueError("checkpoint blocks mapping must not be empty")
        norm, _ = _normalize_blocks_mapping(source, "source", require_finite=False)
        flat_items = [
            (f"{bid}::{sub_k}", t)
            for bid in sorted(norm.keys())
            for sub_k, t in sorted(norm[bid].items())
        ]
    else:
        raise TypeError(f"source must be an nn.Module or Mapping, got {type(source).__name__}")

    digest = hashlib.sha256()
    for name, tensor in flat_items:
        t_cpu = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{t_cpu.dtype}|{tuple(t_cpu.shape)}|".encode())
        digest.update(_raw_bit_view(t_cpu).reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_rescue_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != RESCUER_RECEIPT_OBJECT or receipt.get("schema") != RESCUER_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


@dataclass(frozen=True)
class CheckpointRescueConfig:
    """Validated configuration governing triage, null-space repair, and ANSE rescue commit."""

    anchor_floor_nats: float = DEFAULT_ANCHOR_FLOOR_NATS
    corruption_ceiling_nats: float = DEFAULT_CORRUPTION_CEILING_NATS
    specialist_gain_floor_nats: float | None = None
    fold_ln_lambda: float = DEFAULT_FOLD_LN_LAMBDA
    min_raw_general_drift_ratio: float = DEFAULT_MIN_RAW_DRIFT_RATIO
    max_general_drift_ratio: float = DEFAULT_MAX_GENERAL_DRIFT_RATIO
    max_worst_domain_drift_ratio: float = DEFAULT_MAX_GENERAL_DRIFT_RATIO
    min_specialist_retention_ratio: float = DEFAULT_MIN_SPECIALIST_RETENTION_RATIO
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    extreme_weight_threshold: float = DEFAULT_EXTREME_WEIGHT_THRESHOLD
    spectral_spike_ratio_threshold: float = DEFAULT_SPECTRAL_SPIKE_RATIO_THRESHOLD
    max_spectral_norm_inflation: float = DEFAULT_MAX_SPECTRAL_NORM_INFLATION
    single_mode_spike_norm: float = DEFAULT_SINGLE_MODE_SPIKE_NORM
    nullspace_sv_rel_tol: float = DEFAULT_NULLSPACE_SV_REL_TOL
    nullspace_variance_threshold: float = DEFAULT_NULLSPACE_VARIANCE_THRESHOLD
    nullspace_eig_threshold: float = DEFAULT_NULLSPACE_EIG_THRESHOLD
    projection_side: str = "auto"
    general_prefix: str = GENERAL_EXPERT_PREFIX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    rollback_target: str = "base"
    inplace: bool = True

    def __post_init__(self) -> None:
        a_floor = _validate_finite_float(self.anchor_floor_nats, "anchor_floor_nats")
        c_ceil = _validate_nonneg_finite(self.corruption_ceiling_nats, "corruption_ceiling_nats")
        if a_floor > c_ceil:
            raise ValueError(
                f"anchor_floor_nats ({a_floor}) must be <= corruption_ceiling_nats ({c_ceil})"
            )
        if self.specialist_gain_floor_nats is not None:
            _validate_nonneg_finite(self.specialist_gain_floor_nats, "specialist_gain_floor_nats")
        _validate_unit_interval(self.fold_ln_lambda, "fold_ln_lambda", allow_zero=True)
        min_raw = _validate_positive_finite(self.min_raw_general_drift_ratio, "min_raw_general_drift_ratio")
        max_gen = _validate_positive_finite(self.max_general_drift_ratio, "max_general_drift_ratio")
        max_worst = _validate_positive_finite(self.max_worst_domain_drift_ratio, "max_worst_domain_drift_ratio")
        if max_gen < 1.0 or max_worst < 1.0 or min_raw < 1.0:
            raise ValueError("Drift ratio bounds must be >= 1.0")
        _validate_unit_interval(self.min_specialist_retention_ratio, "min_specialist_retention_ratio", allow_zero=False)
        safe_r = _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        if safe_r < 1.0:
            raise ValueError(f"anse_safe_general_ratio must be >= 1.0, got {safe_r}")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_positive_finite(self.extreme_weight_threshold, "extreme_weight_threshold")
        _validate_positive_finite(self.spectral_spike_ratio_threshold, "spectral_spike_ratio_threshold")
        _validate_positive_finite(self.max_spectral_norm_inflation, "max_spectral_norm_inflation")
        _validate_positive_finite(self.single_mode_spike_norm, "single_mode_spike_norm")
        _validate_nonneg_finite(self.nullspace_sv_rel_tol, "nullspace_sv_rel_tol")
        _validate_unit_interval(self.nullspace_variance_threshold, "nullspace_variance_threshold", allow_zero=False)
        eig_t = _validate_unit_interval(self.nullspace_eig_threshold, "nullspace_eig_threshold", allow_zero=True)
        if eig_t >= 1.0:
            raise ValueError(f"nullspace_eig_threshold must be < 1.0, got {eig_t}")
        if not isinstance(self.projection_side, str) or self.projection_side not in VALID_PROJECTION_SIDES:
            raise ValueError(f"projection_side must be one of {sorted(VALID_PROJECTION_SIDES)}")
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        if not isinstance(self.rollback_target, str) or self.rollback_target not in VALID_ROLLBACK_TARGETS:
            raise ValueError(f"rollback_target must be one of {sorted(VALID_ROLLBACK_TARGETS)}")
        _validate_bool(self.inplace, "inplace")


# ---------------------------------------------------------------------------
# 2. Block Extraction, Application & Normalization
# ---------------------------------------------------------------------------


def _parse_layer_index(block_id: str) -> int | None:
    """Extract integer transformer block index from identifiers like ``blocks.2.mlp.experts.1``."""
    match = _LAYER_ID_RE.search(block_id.strip())
    return int(match.group(1)) if match is not None else None


def _is_final_layernorm_block(block_id: str, sub_tensors: Mapping[str, torch.Tensor]) -> bool:
    """Identify final LayerNorm / RMSNorm blocks intended for ``fold_ln_lambda`` calibration."""
    low = block_id.strip().lower()
    if low in _FINAL_LN_IDS or low.endswith((".ln", ".final_ln", ".norm")):
        return all(t.ndim == 1 for t in sub_tensors.values())
    return False


def extract_checkpoint_blocks(model: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    """Extract block-structured cloned tensors from a ``CrowdV1WithEngram`` or MoE module."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    blocks_attr = getattr(model, "blocks", None)
    if blocks_attr is None or len(blocks_attr) == 0:
        raise ValueError("model must expose a non-empty 'blocks' sequence")

    extracted: dict[str, dict[str, torch.Tensor]] = {}
    for l_idx, blk in enumerate(blocks_attr):
        mlp = getattr(blk, "mlp", blk)
        if isinstance(mlp, CrowdLoRAMoEMLP):
            router_map: dict[str, torch.Tensor] = {"weight": mlp.router.weight.detach().clone()}
            if mlp.router.bias is not None:
                router_map["bias"] = mlp.router.bias.detach().clone()
            extracted[f"blocks.{l_idx}.mlp.router"] = router_map
            for e_key in sorted(mlp.experts.keys(), key=lambda k: int(k)):
                exp_mod = mlp.experts[e_key]
                extracted[f"blocks.{l_idx}.mlp.experts.{e_key}"] = {
                    attr: getattr(exp_mod, attr).detach().clone()
                    for proj in _SWIGLU_PROJECTIONS
                    for attr in (f"{proj}_A", f"{proj}_B")
                }
        elif isinstance(mlp, CrowdFineGrainedMoEMLP):
            extracted[f"blocks.{l_idx}.mlp.router"] = {"weight": mlp.router.weight.detach().clone()}
            for e_idx in range(mlp.num_experts):
                extracted[f"blocks.{l_idx}.mlp.experts.{e_idx}"] = {
                    "w1": mlp.experts_w1[e_idx].weight.detach().clone(),
                    "w2": mlp.experts_w2[e_idx].weight.detach().clone(),
                    "w3": mlp.experts_w3[e_idx].weight.detach().clone(),
                }
        else:
            mlp_map: dict[str, torch.Tensor] = {}
            for attr in ("w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"):
                sub = getattr(mlp, attr, None)
                if isinstance(sub, nn.Linear):
                    mlp_map[attr] = sub.weight.detach().clone()
            if not mlp_map:
                raise ValueError(f"Block {l_idx} mlp ({type(mlp).__name__}) exposes no supported MoE or MLP weights")
            extracted[f"blocks.{l_idx}.mlp"] = mlp_map

    ln_mod = getattr(model, "ln", getattr(model, "norm", None))
    if ln_mod is not None and isinstance(getattr(ln_mod, "weight", None), torch.Tensor):
        ln_map: dict[str, torch.Tensor] = {"weight": ln_mod.weight.detach().clone()}
        if isinstance(getattr(ln_mod, "bias", None), torch.Tensor):
            ln_map["bias"] = ln_mod.bias.detach().clone()
        extracted["final_ln"] = ln_map
    return extracted


@torch.no_grad()
def apply_checkpoint_blocks_inplace(
    model: nn.Module,
    blocks_map: Mapping[str, Any],
) -> None:
    """Copy repaired or baseline block tensors in-place onto ``model``."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    blocks_attr = getattr(model, "blocks", None)
    if blocks_attr is None or len(blocks_attr) == 0:
        raise ValueError("model must expose a non-empty 'blocks' sequence")
    norm_blocks, _ = _normalize_blocks_mapping(blocks_map, "blocks_map", require_finite=False)

    for block_id, sub_map in norm_blocks.items():
        if _is_final_layernorm_block(block_id, sub_map):
            ln_mod = getattr(model, "ln", getattr(model, "norm", None))
            if ln_mod is None:
                raise ValueError(f"Cannot apply {block_id!r}: model exposes no 'ln' or 'norm'")
            for k, src_t in sub_map.items():
                dst_t = getattr(ln_mod, k, None)
                if not isinstance(dst_t, torch.Tensor) or dst_t.shape != src_t.shape:
                    raise ValueError(f"Shape/attribute mismatch applying {block_id}.{k}")
                dst_t.copy_(src_t.to(device=dst_t.device, dtype=dst_t.dtype))
            continue

        l_idx = _parse_layer_index(block_id)
        if l_idx is None or not (0 <= l_idx < len(blocks_attr)):
            raise ValueError(f"Unknown or out-of-range block_id {block_id!r}")
        blk = blocks_attr[l_idx]
        mlp = getattr(blk, "mlp", blk)

        if block_id.endswith(".router"):
            for k, src_t in sub_map.items():
                dst_t = getattr(mlp.router, k, None)
                if not isinstance(dst_t, torch.Tensor) or dst_t.shape != src_t.shape:
                    raise ValueError(f"Router attribute/shape mismatch for {block_id}.{k}")
                dst_t.copy_(src_t.to(device=dst_t.device, dtype=dst_t.dtype))
        elif ".experts." in block_id:
            e_key = block_id.rsplit(".experts.", 1)[1]
            if isinstance(mlp, CrowdLoRAMoEMLP):
                if e_key not in mlp.experts:
                    raise ValueError(f"Expert {e_key!r} missing in layer {l_idx}")
                exp_mod = mlp.experts[e_key]
                for k, src_t in sub_map.items():
                    dst_t = getattr(exp_mod, k, None)
                    if not isinstance(dst_t, torch.Tensor) or dst_t.shape != src_t.shape:
                        raise ValueError(f"LoRA expert mismatch for {block_id}.{k}")
                    dst_t.copy_(src_t.to(device=dst_t.device, dtype=dst_t.dtype))
            elif isinstance(mlp, CrowdFineGrainedMoEMLP):
                e_idx = int(e_key)
                for k, src_t in sub_map.items():
                    mod_list = getattr(mlp, f"experts_{k}", None)
                    if mod_list is None or not (0 <= e_idx < len(mod_list)):
                        raise ValueError(f"Fine-grained expert mismatch for {block_id}.{k}")
                    dst_t = mod_list[e_idx].weight
                    if dst_t.shape != src_t.shape:
                        raise ValueError(f"Shape mismatch on {block_id}.{k}")
                    dst_t.copy_(src_t.to(device=dst_t.device, dtype=dst_t.dtype))
            else:
                raise ValueError(f"Block {l_idx} does not expose MoE experts for {block_id!r}")
        elif block_id.endswith(".mlp"):
            for k, src_t in sub_map.items():
                sub = getattr(mlp, k, None)
                if not isinstance(sub, nn.Linear) or sub.weight.shape != src_t.shape:
                    raise ValueError(f"MLP weight mismatch on {block_id}.{k}")
                sub.weight.copy_(src_t.to(device=sub.weight.device, dtype=sub.weight.dtype))
        else:
            raise ValueError(f"Unsupported block path {block_id!r} for model application")


def _normalize_blocks_mapping(
    blocks: nn.Module | Mapping[str, Any],
    name: str,
    *,
    require_finite: bool,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, bool]]:
    """Normalize ``nn.Module`` or mapping of tensors/sub-mappings into ``{bid: {sub_k: Tensor}}``."""
    if isinstance(blocks, nn.Module):
        extracted = extract_checkpoint_blocks(blocks)
        if require_finite:
            for bid, sub_map in extracted.items():
                for sk, t in sub_map.items():
                    if not bool(torch.isfinite(t).all().item()):
                        raise ValueError(f"{name}[{bid!r}][{sk!r}] contains non-finite values")
        return extracted, {bid: False for bid in extracted}

    if not isinstance(blocks, Mapping) or not blocks:
        raise TypeError(f"{name} must be an nn.Module or non-empty Mapping")

    out: dict[str, dict[str, torch.Tensor]] = {}
    single_flags: dict[str, bool] = {}
    for raw_bid, raw_val in blocks.items():
        if not isinstance(raw_bid, str) or not raw_bid.strip():
            raise ValueError(f"{name} keys must be non-empty strings, got {raw_bid!r}")
        bid = raw_bid
        if isinstance(raw_val, torch.Tensor):
            if raw_val.numel() == 0 or not raw_val.is_floating_point():
                raise ValueError(f"{name}[{bid!r}] must be a non-empty floating-point torch.Tensor")
            if require_finite and not bool(torch.isfinite(raw_val).all().item()):
                raise ValueError(f"{name}[{bid!r}] contains non-finite values")
            out[bid] = {_SINGLE_TENSOR_KEY: raw_val.detach().clone()}
            single_flags[bid] = True
        elif isinstance(raw_val, Mapping) and raw_val:
            sub_dict: dict[str, torch.Tensor] = {}
            for raw_sk, sub_t in raw_val.items():
                if not isinstance(raw_sk, str) or not raw_sk.strip():
                    raise ValueError(f"{name}[{bid!r}] sub-keys must be non-empty strings")
                if not isinstance(sub_t, torch.Tensor) or sub_t.numel() == 0 or not sub_t.is_floating_point():
                    raise ValueError(
                        f"{name}[{bid!r}][{raw_sk!r}] must be a non-empty floating-point torch.Tensor"
                    )
                if require_finite and not bool(torch.isfinite(sub_t).all().item()):
                    raise ValueError(f"{name}[{bid!r}][{raw_sk!r}] contains non-finite values")
                sub_dict[raw_sk] = sub_t.detach().clone()
            out[bid] = sub_dict
            single_flags[bid] = False
        else:
            raise TypeError(
                f"{name}[{bid!r}] must be a torch.Tensor or non-empty Mapping[str, torch.Tensor], "
                f"got {type(raw_val).__name__}"
            )
    return out, single_flags


def _validate_matching_block_layouts(
    base_norm: Mapping[str, Mapping[str, torch.Tensor]],
    drift_norm: Mapping[str, Mapping[str, torch.Tensor]],
) -> list[str]:
    """Verify identical block IDs, sub-keys, and tensor shapes between ``base`` and ``drifted``."""
    base_keys = list(base_norm.keys())
    if set(base_keys) != set(drift_norm.keys()):
        missing = sorted(set(base_keys) - set(drift_norm.keys()))
        extra = sorted(set(drift_norm.keys()) - set(base_keys))
        raise ValueError(
            f"base_blocks and drifted_blocks keys must match (missing={missing}, extra={extra})"
        )
    for bid in base_keys:
        b_sub = base_norm[bid]
        d_sub = drift_norm[bid]
        if set(b_sub.keys()) != set(d_sub.keys()):
            raise ValueError(f"Sub-keys mismatch for block {bid!r}: {sorted(b_sub)} vs {sorted(d_sub)}")
        for sk in b_sub:
            if b_sub[sk].shape != d_sub[sk].shape:
                raise ValueError(
                    f"Shape mismatch on {bid}.{sk}: base={tuple(b_sub[sk].shape)} vs drifted={tuple(d_sub[sk].shape)}"
                )
    return base_keys


def _package_blocks_output(
    repaired_norm: Mapping[str, Mapping[str, torch.Tensor]],
    single_flags: Mapping[str, bool],
) -> dict[str, Any]:
    """Repackage normalized block map back into caller's Tensor vs sub-dict structure."""
    out: dict[str, Any] = {}
    for bid, sub_map in repaired_norm.items():
        if single_flags.get(bid, False) and _SINGLE_TENSOR_KEY in sub_map:
            out[bid] = sub_map[_SINGLE_TENSOR_KEY]
        else:
            out[bid] = dict(sub_map)
    return out


def _resolve_block_attribution_value(
    block_id: str,
    block_pos: int,
    layer_idx: int | None,
    attr_source: Mapping[Any, Any] | Sequence[float],
    name: str,
    *,
    is_layernorm: bool,
) -> float | None:
    """Resolve general harm or specialist gain nats for ``block_id``."""
    if isinstance(attr_source, Mapping):
        candidates: list[Any] = [block_id]
        if layer_idx is not None:
            candidates.extend([layer_idx, str(layer_idx), f"layer_{layer_idx}", f"blocks.{layer_idx}"])
        for cand in candidates:
            if cand in attr_source:
                return _validate_finite_float(attr_source[cand], f"{name}[{cand!r}]")
        if is_layernorm:
            return None
        raise ValueError(f"{name} missing attribution entry for block {block_id!r}")

    if isinstance(attr_source, (str, bytes)) or not isinstance(attr_source, Sequence):
        raise TypeError(f"{name} must be a Mapping or Sequence of floats, got {type(attr_source).__name__}")
    if layer_idx is not None and 0 <= layer_idx < len(attr_source):
        return _validate_finite_float(attr_source[layer_idx], f"{name}[{layer_idx}]")
    if not is_layernorm and 0 <= block_pos < len(attr_source):
        return _validate_finite_float(attr_source[block_pos], f"{name}[{block_pos}]")
    if is_layernorm:
        return None
    raise ValueError(f"{name} sequence length {len(attr_source)} cannot resolve block {block_id!r}")


# ---------------------------------------------------------------------------
# 3. Health Inspection & Triage (Protected Anchors, Dual-Active, Corruptors, Quarantined)
# ---------------------------------------------------------------------------


def _inspect_matrix_spectrum(
    d64: torch.Tensor,
    b64: torch.Tensor,
    *,
    max_inflation: float,
    spike_ratio_thr: float,
    single_mode_thr: float,
    label: str,
) -> tuple[float, float, list[str]]:
    """Inspect singular values of a 2D tensor or LoRA product for spectral explosion."""
    reasons: list[str] = []
    s_drift = torch.linalg.svdvals(d64)
    s_base = torch.linalg.svdvals(b64)
    sigma_1 = float(s_drift[0].item()) if s_drift.numel() > 0 else 0.0
    base_sigma_1 = float(s_base[0].item()) if s_base.numel() > 0 else 0.0

    s_delta = torch.linalg.svdvals(d64 - b64)
    delta_sigma_1 = float(s_delta[0].item()) if s_delta.numel() > 0 else 0.0

    if base_sigma_1 > 1e-6 and (
        sigma_1 / base_sigma_1 > max_inflation or delta_sigma_1 / base_sigma_1 > max_inflation
    ):
        reasons.append(f"spectral_norm_inflation:{label}")

    s_pos = s_drift[s_drift > 1e-10]
    spike_ratio = 0.0
    if s_pos.numel() >= 2:
        med = float(torch.median(s_pos).item())
        spike_ratio = sigma_1 / max(med, 1e-12)
        if spike_ratio > spike_ratio_thr and sigma_1 > 5.0:
            reasons.append(f"spectral_spike:{label}")
    if max(sigma_1, delta_sigma_1) > single_mode_thr:
        reasons.append(f"single_mode_spectral_explosion:{label}")
    return sigma_1, spike_ratio, reasons


def _inspect_block_health(
    base_sub: Mapping[str, torch.Tensor],
    drift_sub: Mapping[str, torch.Tensor],
    *,
    extreme_weight_threshold: float,
    spectral_spike_ratio_threshold: float,
    max_spectral_norm_inflation: float,
    single_mode_spike_norm: float,
) -> dict[str, Any]:
    """Detect NaN/Inf, bit-flip jumps, and spectral explosions in a drifted block."""
    reasons: list[str] = []
    delta_sq = 0.0
    max_sigma = 0.0
    max_spike_ratio = 0.0

    for sk, t_drift in sorted(drift_sub.items()):
        t64 = t_drift.detach().to(device="cpu", dtype=torch.float64)
        b64 = base_sub[sk].detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(t64).all().item()):
            reasons.append(f"non_finite:{sk}")
            continue
        diff = t64 - b64
        delta_sq += float((diff * diff).sum().item())

        abs_t = t64.abs()
        abs_max = float(abs_t.max().item())
        if abs_max >= extreme_weight_threshold:
            reasons.append(f"extreme_magnitude:{sk}")

        nz_finite = abs_t[(abs_t > 0.0) & (abs_t < extreme_weight_threshold)]
        if nz_finite.numel() >= 4:
            med = float(torch.median(nz_finite).item())
            jump_thr = max(DEFAULT_ISOLATED_JUMP_FLOOR, DEFAULT_ISOLATED_JUMP_MULTIPLIER * max(med, 1e-3))
            if bool((abs_t >= jump_thr).any().item()):
                reasons.append(f"bitflip_outlier:{sk}")

        if t64.ndim == 1:
            norm_d = float(torch.linalg.vector_norm(t64).item())
            norm_b = float(torch.linalg.vector_norm(b64).item())
            if norm_d > max(single_mode_spike_norm, max_spectral_norm_inflation * max(norm_b, 1.0)):
                reasons.append(f"vector_norm_explosion:{sk}")

    # Spectral checks: LoRA factor pairs (B @ A) and standalone 2D matrices
    lora_handled: set[str] = set()
    if all(f"{p}_{s}" in drift_sub for p in _SWIGLU_PROJECTIONS for s in ("A", "B")):
        for proj in _SWIGLU_PROJECTIONS:
            ak, bk = f"{proj}_A", f"{proj}_B"
            lora_handled.update((ak, bk))
            a_d, b_d = drift_sub[ak], drift_sub[bk]
            a_b, b_b = base_sub[ak], base_sub[bk]
            if not (bool(torch.isfinite(a_d).all().item()) and bool(torch.isfinite(b_d).all().item())):
                continue
            _, s_drift, _ = decompose_lora_qr_svd(b_d, a_d)
            _, s_base, _ = decompose_lora_qr_svd(b_b, a_b)
            s1 = float(s_drift[0].item()) if s_drift.numel() > 0 else 0.0
            bs1 = float(s_base[0].item()) if s_base.numel() > 0 else 0.0
            max_sigma = max(max_sigma, s1)
            if bs1 > 1e-6 and s1 / bs1 > max_spectral_norm_inflation:
                reasons.append(f"lora_spectral_inflation:{proj}")
            s_pos = s_drift[s_drift > 1e-10]
            if s_pos.numel() >= 2:
                med = float(torch.median(s_pos).item())
                ratio = s1 / max(med, 1e-12)
                max_spike_ratio = max(max_spike_ratio, ratio)
                if ratio > spectral_spike_ratio_threshold and s1 > 5.0:
                    reasons.append(f"lora_spectral_spike:{proj}")
            if s1 > single_mode_spike_norm:
                reasons.append(f"lora_single_mode_explosion:{proj}")

    for sk, t_drift in sorted(drift_sub.items()):
        if sk in lora_handled or t_drift.ndim != 2:
            continue
        t64 = t_drift.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(t64).all().item()):
            continue
        b64 = base_sub[sk].detach().to(device="cpu", dtype=torch.float64)
        s1, sr, spec_reasons = _inspect_matrix_spectrum(
            t64,
            b64,
            max_inflation=max_spectral_norm_inflation,
            spike_ratio_thr=spectral_spike_ratio_threshold,
            single_mode_thr=single_mode_spike_norm,
            label=sk,
        )
        max_sigma = max(max_sigma, s1)
        max_spike_ratio = max(max_spike_ratio, sr)
        reasons.extend(spec_reasons)

    dedup = list(dict.fromkeys(reasons))
    has_non_finite = any(r.startswith("non_finite:") for r in dedup)
    return {
        "healthy": len(dedup) == 0,
        "quarantined": len(dedup) > 0,
        "quarantine_reasons": dedup,
        "has_non_finite": has_non_finite,
        "delta_frobenius_norm": None if has_non_finite else math.sqrt(delta_sq),
        "max_singular_value": None if has_non_finite else max_sigma,
        "spectral_spike_ratio": None if has_non_finite else max_spike_ratio,
    }


def triage_drifted_checkpoint_blocks(
    base_blocks: Mapping[str, Any] | nn.Module,
    drifted_blocks: Mapping[str, Any] | nn.Module,
    layer_general_harm_nats: Mapping[Any, float] | Sequence[float],
    layer_specialist_gain_nats: Mapping[Any, float] | Sequence[float],
    *,
    anchor_floor_nats: float = DEFAULT_ANCHOR_FLOOR_NATS,
    corruption_ceiling_nats: float = DEFAULT_CORRUPTION_CEILING_NATS,
    specialist_gain_floor_nats: float | None = None,
    extreme_weight_threshold: float = DEFAULT_EXTREME_WEIGHT_THRESHOLD,
    spectral_spike_ratio_threshold: float = DEFAULT_SPECTRAL_SPIKE_RATIO_THRESHOLD,
    max_spectral_norm_inflation: float = DEFAULT_MAX_SPECTRAL_NORM_INFLATION,
    single_mode_spike_norm: float = DEFAULT_SINGLE_MODE_SPIKE_NORM,
) -> dict[str, Any]:
    """Classify every drifted checkpoint block into anchor, dual-active, corrupting, or quarantined."""
    anchor_floor = _validate_finite_float(anchor_floor_nats, "anchor_floor_nats")
    corrupt_ceil = _validate_nonneg_finite(corruption_ceiling_nats, "corruption_ceiling_nats")
    if anchor_floor > corrupt_ceil:
        raise ValueError(
            f"anchor_floor_nats ({anchor_floor}) must be <= corruption_ceiling_nats ({corrupt_ceil})"
        )
    spec_floor = (
        _validate_nonneg_finite(specialist_gain_floor_nats, "specialist_gain_floor_nats")
        if specialist_gain_floor_nats is not None
        else max(abs(anchor_floor), corrupt_ceil)
    )
    ext_thr = _validate_positive_finite(extreme_weight_threshold, "extreme_weight_threshold")
    spike_thr = _validate_positive_finite(spectral_spike_ratio_threshold, "spectral_spike_ratio_threshold")
    infl_thr = _validate_positive_finite(max_spectral_norm_inflation, "max_spectral_norm_inflation")
    mode_thr = _validate_positive_finite(single_mode_spike_norm, "single_mode_spike_norm")

    base_norm, _ = _normalize_blocks_mapping(base_blocks, "base_blocks", require_finite=True)
    drift_norm, _ = _normalize_blocks_mapping(drifted_blocks, "drifted_blocks", require_finite=False)
    block_ids = _validate_matching_block_layouts(base_norm, drift_norm)

    anchors: list[str] = []
    dual_active: list[str] = []
    corruptors: list[str] = []
    quarantined: list[str] = []
    layernorms: list[str] = []
    per_block: dict[str, dict[str, Any]] = {}

    non_ln_pos = 0
    for bid in block_ids:
        b_sub = base_norm[bid]
        d_sub = drift_norm[bid]
        is_ln = _is_final_layernorm_block(bid, b_sub)
        l_idx = _parse_layer_index(bid)

        harm_val = _resolve_block_attribution_value(
            bid, non_ln_pos, l_idx, layer_general_harm_nats, "layer_general_harm_nats", is_layernorm=is_ln
        )
        gain_val = _resolve_block_attribution_value(
            bid, non_ln_pos, l_idx, layer_specialist_gain_nats, "layer_specialist_gain_nats", is_layernorm=is_ln
        )
        if not is_ln:
            non_ln_pos += 1

        health = _inspect_block_health(
            b_sub,
            d_sub,
            extreme_weight_threshold=ext_thr,
            spectral_spike_ratio_threshold=spike_thr,
            max_spectral_norm_inflation=infl_thr,
            single_mode_spike_norm=mode_thr,
        )

        if health["quarantined"]:
            cls_name = "quarantined"
            action = "quarantine_revert_to_base"
            quarantined.append(bid)
        elif is_ln:
            cls_name = "layernorm_calibration"
            action = "interpolate_layernorm"
            layernorms.append(bid)
        else:
            assert harm_val is not None and gain_val is not None
            if harm_val <= corrupt_ceil:
                if harm_val <= anchor_floor or gain_val >= -corrupt_ceil:
                    cls_name = "protected_specialist_anchor"
                    action = "preserve_bitwise"
                    anchors.append(bid)
                else:
                    cls_name = "prose_corrupting"
                    action = "revert_to_base"
                    corruptors.append(bid)
            elif gain_val >= spec_floor:
                cls_name = "salvageable_dual_active"
                action = "project_nullspace"
                dual_active.append(bid)
            else:
                cls_name = "prose_corrupting"
                action = "revert_to_base"
                corruptors.append(bid)

        per_block[bid] = {
            "block_id": bid,
            "layer_index": l_idx,
            "classification": cls_name,
            "action": action,
            "general_harm_nats": harm_val,
            "specialist_gain_nats": gain_val,
            "is_layernorm": is_ln,
            **health,
        }

    return {
        "schema": RESCUER_SCHEMA,
        "anchor_floor_nats": anchor_floor,
        "corruption_ceiling_nats": corrupt_ceil,
        "specialist_gain_floor_nats": spec_floor,
        "protected_specialist_anchors": anchors,
        "salvageable_dual_active_blocks": dual_active,
        "prose_corrupting_blocks": corruptors,
        "quarantined_blocks": quarantined,
        "layernorm_blocks": layernorms,
        "blocks": per_block,
        "summary": {
            "total_blocks": len(block_ids),
            "anchor_count": len(anchors),
            "dual_active_count": len(dual_active),
            "corrupting_count": len(corruptors),
            "quarantined_count": len(quarantined),
            "layernorm_count": len(layernorms),
        },
    }


# ---------------------------------------------------------------------------
# 4. Null-Space Repair Engine (P_0 Projection, Bitwise Anchors, LN Folding)
# ---------------------------------------------------------------------------


def _validate_square_projector(proj: Any, name: str) -> torch.Tensor:
    """Validate a finite symmetric square 2D projector matrix on CPU ``float64``."""
    if not isinstance(proj, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(proj).__name__}")
    if proj.ndim != 2 or proj.shape[0] == 0 or proj.shape[0] != proj.shape[1]:
        raise ValueError(f"{name} must be a non-empty square 2D matrix, got shape {tuple(proj.shape)}")
    if not proj.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {proj.dtype}")
    mat = proj.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(mat).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    asym = float((mat - mat.T).abs().max().item())
    scale = max(1.0, float(mat.abs().max().item()))
    if asym > 1e-4 * scale:
        raise ValueError(f"{name} must be symmetric (max asymmetry {asym:.3e})")
    return (0.5 * (mat + mat.T)).contiguous()


def _resolve_projector_candidates(
    block_id: str,
    sub_key: str,
    layer_idx: int | None,
    space_hint: str,
    p_null_by_layer: torch.Tensor | Mapping[Any, Any],
) -> list[tuple[str, torch.Tensor]]:
    """Collect candidate ``(label, projector_matrix)`` entries in specificity order."""
    if isinstance(p_null_by_layer, torch.Tensor):
        return [("p_null", _validate_square_projector(p_null_by_layer, "p_null_by_layer"))]
    if not isinstance(p_null_by_layer, Mapping) or not p_null_by_layer:
        raise TypeError("p_null_by_layer must be a 2D Tensor or non-empty Mapping")

    out: list[tuple[str, torch.Tensor]] = []
    # 1. Exact block/sub-key match
    for direct_k in (f"{block_id}.{sub_key}", block_id):
        if direct_k in p_null_by_layer and not isinstance(p_null_by_layer[direct_k], Mapping):
            out.append((direct_k, _validate_square_projector(p_null_by_layer[direct_k], f"p_null[{direct_k!r}]")))

    # 2. Layer-level mapping (tensor or nested {'x': ..., 'act': ...})
    layer_keys: list[Any] = []
    if layer_idx is not None:
        layer_keys.extend(
            [
                f"blocks.{layer_idx}.{space_hint}",
                layer_idx,
                str(layer_idx),
                f"layer_{layer_idx}",
                f"blocks.{layer_idx}",
            ]
        )
    layer_keys.extend([space_hint, "default", "all"])

    for lk in layer_keys:
        if lk not in p_null_by_layer:
            continue
        val = p_null_by_layer[lk]
        if isinstance(val, torch.Tensor):
            out.append((repr(lk), _validate_square_projector(val, f"p_null[{lk!r}]")))
        elif isinstance(val, Mapping):
            for sk in (space_hint, "x", "act", "default"):
                if sk in val and isinstance(val[sk], torch.Tensor):
                    out.append((f"{lk}.{sk}", _validate_square_projector(val[sk], f"p_null[{lk!r}][{sk!r}]")))
    return out


def _select_projector_and_side(
    block_id: str,
    sub_key: str,
    shape: tuple[int, ...],
    layer_idx: int | None,
    p_null_by_layer: torch.Tensor | Mapping[Any, Any],
    *,
    projection_side: str,
    prefer_act: bool,
) -> tuple[torch.Tensor, str]:
    """Select compatible ``(projector_64, side)`` for a 1D or 2D tensor."""
    hint = "act" if prefer_act else "x"
    cands = _resolve_projector_candidates(block_id, sub_key, layer_idx, hint, p_null_by_layer)
    if len(shape) == 1:
        d_vec = shape[0]
        for _, p_mat in cands:
            if int(p_mat.shape[0]) == d_vec:
                return p_mat, "input"
        raise ValueError(f"No compatible projector found for 1D tensor {block_id}.{sub_key} of length {d_vec}")

    d_out, d_in = shape[0], shape[1]
    if projection_side in ("auto", "input"):
        for _, p_mat in cands:
            if int(p_mat.shape[0]) == d_in:
                return p_mat, "input"
    if projection_side in ("auto", "output"):
        for _, p_mat in cands:
            if int(p_mat.shape[0]) == d_out:
                return p_mat, "output"
    raise ValueError(
        f"No compatible projector found for {block_id}.{sub_key} shape {shape} (projection_side={projection_side!r})"
    )


def _binarize_or_validate_projector(p_mat: torch.Tensor, eig_threshold: float) -> torch.Tensor:
    """Enforce clean symmetric projection behavior while preserving exact P_0 if already idempotent."""
    eye = torch.eye(int(p_mat.shape[0]), dtype=torch.float64)
    idem_err = float((p_mat @ p_mat - p_mat).abs().max().item())
    if idem_err <= 1e-6:
        return p_mat
    evals, evecs = torch.linalg.eigh(p_mat)
    keep = evals > eig_threshold
    if not bool(keep.any().item()):
        return torch.zeros_like(eye)
    q_keep = evecs[:, keep]
    proj = q_keep @ q_keep.T
    return (0.5 * (proj + proj.T)).contiguous()


def _project_lora_pair_into_nullspace(
    b_base: torch.Tensor,
    a_base: torch.Tensor,
    b_drift: torch.Tensor,
    a_drift: torch.Tensor,
    p_mat: torch.Tensor,
    side: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Project LoRA delta ``B_1 A_1 - B_0 A_0`` into ``P_0`` without dense ``(d_out, d_in)`` materialization."""
    b0 = b_base.detach().to("cpu", dtype=torch.float64)
    a0 = a_base.detach().to("cpu", dtype=torch.float64)
    b1 = b_drift.detach().to("cpu", dtype=torch.float64)
    a1 = a_drift.detach().to("cpu", dtype=torch.float64)
    eye_p = torch.eye(int(p_mat.shape[0]), dtype=torch.float64)
    q_perp = eye_p - p_mat

    zero_base = int(torch.count_nonzero(b0).item()) == 0
    if zero_base:
        if side == "input":
            a_new = (a1 @ p_mat).contiguous()
            b_new = b1.contiguous()
            leak = float(torch.linalg.matrix_norm(b_new @ (a_new @ q_perp)).item())
        else:
            b_new = (p_mat @ b1).contiguous()
            a_new = a1.contiguous()
            leak = float(torch.linalg.matrix_norm((q_perp @ b_new) @ a_new).item())
        return b_new.to(dtype=b_base.dtype), a_new.to(dtype=a_base.dtype), leak

    # Non-zero LoRA base: preserve base on ker(P_0) exactly so delta in ker(P_0) is strictly zero.
    if side == "input":
        base_perp_norm = float(torch.linalg.matrix_norm(b0 @ (a0 @ q_perp)).item())
        if base_perp_norm <= 1e-10:
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
        base_perp_norm = float(torch.linalg.matrix_norm((q_perp @ b0) @ a0).item())
        if base_perp_norm <= 1e-10:
            b_new = (p_mat @ b1).contiguous()
            a_new = a1.contiguous()
        else:
            a_new = a0.contiguous()
            rhs_t = (p_mat @ b1 @ a1).T
            b_proj_t = torch.linalg.lstsq(a0.T, rhs_t).solution
            b_new = (q_perp @ b0 + p_mat @ b_proj_t.T).contiguous()
        d_b = torch.cat([q_perp @ b_new, -(q_perp @ b0)], dim=1)
        d_a = torch.cat([a_new, a0], dim=0)

    _, s_leak, _ = decompose_lora_qr_svd(d_b, d_a)
    leak = float(s_leak.pow(2).sum().sqrt().item())
    return b_new.to(dtype=b_base.dtype), a_new.to(dtype=a_base.dtype), leak


@torch.no_grad()
def repair_drifted_blocks_in_nullspace(
    base_blocks: Mapping[str, Any] | nn.Module,
    drifted_blocks: Mapping[str, Any] | nn.Module,
    triage_map: Mapping[str, Any],
    p_null_by_layer: torch.Tensor | Mapping[Any, Any],
    *,
    fold_ln_lambda: float = DEFAULT_FOLD_LN_LAMBDA,
    projection_side: str = "auto",
    nullspace_eig_threshold: float = DEFAULT_NULLSPACE_EIG_THRESHOLD,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Repair triaged checkpoint blocks: project dual-active deltas in ``P_0``, revert corruptors, preserve anchors."""
    if not isinstance(triage_map, Mapping) or not isinstance(triage_map.get("blocks"), Mapping):
        raise TypeError("triage_map must be the dict returned by triage_drifted_checkpoint_blocks")
    lam_ln = _validate_unit_interval(fold_ln_lambda, "fold_ln_lambda", allow_zero=True)
    if not isinstance(projection_side, str) or projection_side not in VALID_PROJECTION_SIDES:
        raise ValueError(f"projection_side must be one of {sorted(VALID_PROJECTION_SIDES)}")
    eig_thr = _validate_unit_interval(nullspace_eig_threshold, "nullspace_eig_threshold", allow_zero=True)
    if eig_thr >= 1.0:
        raise ValueError(f"nullspace_eig_threshold must be < 1.0, got {eig_thr}")

    base_norm, base_flags = _normalize_blocks_mapping(base_blocks, "base_blocks", require_finite=True)
    drift_norm, _ = _normalize_blocks_mapping(drifted_blocks, "drifted_blocks", require_finite=False)
    block_ids = _validate_matching_block_layouts(base_norm, drift_norm)

    triage_blocks = triage_map["blocks"]
    repaired_norm: dict[str, dict[str, torch.Tensor]] = {}
    per_block_report: dict[str, dict[str, Any]] = {}
    max_leakage = 0.0

    for bid in block_ids:
        if bid not in triage_blocks:
            raise ValueError(f"triage_map['blocks'] missing entry for block {bid!r}")
        entry = triage_blocks[bid]
        cls_name = str(entry["classification"])
        b_sub = base_norm[bid]
        d_sub = drift_norm[bid]
        l_idx = entry.get("layer_index")

        if cls_name == "protected_specialist_anchor":
            repaired_norm[bid] = {sk: t.detach().clone() for sk, t in d_sub.items()}
            per_block_report[bid] = {"action": "preserved_bitwise", "nullspace_leakage_norm": 0.0}
            continue

        if cls_name in ("prose_corrupting", "quarantined"):
            repaired_norm[bid] = {sk: t.detach().clone() for sk, t in b_sub.items()}
            per_block_report[bid] = {"action": "reverted_to_base_bitwise", "nullspace_leakage_norm": 0.0}
            continue

        if cls_name == "layernorm_calibration":
            ln_rep: dict[str, torch.Tensor] = {}
            for sk, t_b in b_sub.items():
                t_b64 = t_b.detach().to("cpu", dtype=torch.float64)
                t_d64 = d_sub[sk].detach().to("cpu", dtype=torch.float64)
                val64 = t_b64 + lam_ln * (t_d64 - t_b64)
                ln_rep[sk] = val64.to(device=t_b.device, dtype=t_b.dtype).contiguous()
            repaired_norm[bid] = ln_rep
            per_block_report[bid] = {"action": "layernorm_interpolated", "fold_ln_lambda": lam_ln}
            continue

        if cls_name != "salvageable_dual_active":
            raise ValueError(f"Unsupported block classification {cls_name!r} for {bid!r}")

        # Project dual-active block deltas into P_0
        rep_sub: dict[str, torch.Tensor] = {}
        block_leak = 0.0
        lora_handled: set[str] = set()

        if all(f"{p}_{s}" in d_sub for p in _SWIGLU_PROJECTIONS for s in ("A", "B")):
            for proj in _SWIGLU_PROJECTIONS:
                ak, bk = f"{proj}_A", f"{proj}_B"
                lora_handled.update((ak, bk))
                a_b, b_b = b_sub[ak], b_sub[bk]
                a_d, b_d = d_sub[ak], d_sub[bk]
                prefer_act = proj == "w3"
                p_raw, side = _select_projector_and_side(
                    bid,
                    proj,
                    (int(b_d.shape[0]), int(a_d.shape[1])),
                    l_idx,
                    p_null_by_layer,
                    projection_side=projection_side,
                    prefer_act=prefer_act,
                )
                p_clean = _binarize_or_validate_projector(p_raw, eig_thr)
                b_rep, a_rep, leak = _project_lora_pair_into_nullspace(b_b, a_b, b_d, a_d, p_clean, side)
                rep_sub[bk] = b_rep.to(device=b_b.device)
                rep_sub[ak] = a_rep.to(device=a_b.device)
                block_leak = max(block_leak, leak)

        for sk, t_b in b_sub.items():
            if sk in lora_handled:
                continue
            t_d = d_sub[sk]
            prefer_act = sk in ("w3", "down_proj")
            p_raw, side = _select_projector_and_side(
                bid,
                sk,
                tuple(int(s) for s in t_b.shape),
                l_idx,
                p_null_by_layer,
                projection_side=projection_side,
                prefer_act=prefer_act,
            )
            p_clean = _binarize_or_validate_projector(p_raw, eig_thr)
            q_perp = torch.eye(int(p_clean.shape[0]), dtype=torch.float64) - p_clean
            b64 = t_b.detach().to("cpu", dtype=torch.float64)
            d64 = t_d.detach().to("cpu", dtype=torch.float64)
            delta = d64 - b64
            if t_b.ndim == 1:
                proj_delta = p_clean @ delta
                leak = float(torch.linalg.vector_norm(q_perp @ proj_delta).item())
            elif side == "input":
                proj_delta = delta @ p_clean
                leak = float(torch.linalg.matrix_norm(proj_delta @ q_perp).item())
            else:
                proj_delta = p_clean @ delta
                leak = float(torch.linalg.matrix_norm(q_perp @ proj_delta).item())
            rep_sub[sk] = (b64 + proj_delta).to(device=t_b.device, dtype=t_b.dtype).contiguous()
            block_leak = max(block_leak, leak)

        repaired_norm[bid] = rep_sub
        max_leakage = max(max_leakage, block_leak)
        per_block_report[bid] = {"action": "projected_in_nullspace", "nullspace_leakage_norm": block_leak}

    repaired_out = _package_blocks_output(repaired_norm, base_flags)
    return repaired_out, {
        "schema": RESCUER_SCHEMA,
        "fold_ln_lambda": lam_ln,
        "max_nullspace_leakage_norm": max_leakage,
        "blocks": per_block_report,
    }


# ---------------------------------------------------------------------------
# 5. Prefix-Cached Causal Attribution, Probe P_0 Synthesis & Evaluation
# ---------------------------------------------------------------------------


def _normalize_chunks(
    chunks: Sequence[Mapping[str, Any]],
    vocab_size: int,
    general_prefix: str,
    name: str,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Validate token chunk sequences and extract ``(ids_2d, is_general_mask, expert_ids)``."""
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

    ids_2d = torch.stack(rows, dim=0)
    is_gen = np.array([eid.startswith(general_prefix) for eid in eids], dtype=bool)
    if not bool(np.any(is_gen)) or not bool(np.any(~is_gen)):
        raise ValueError(f"{name} must include both general ({general_prefix}*) and specialist chunks")
    return ids_2d, is_gen, eids


@torch.no_grad()
def _forward_pre_ln_and_layer_inputs(
    model: nn.Module,
    ids_2d: torch.Tensor,
    *,
    batch_size: int,
    collect_layer_inputs: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Run forward pass returning ``(h_pre_ln, pre_block_states, pre_mlp_states)``."""
    cos = model.cos_cached.to(ids_2d.device)
    sin = model.sin_cached.to(ids_2d.device)
    xb = torch.cat(
        [compute_embeddings_batch(model, ids_2d[b0 : b0 + batch_size]) for b0 in range(0, ids_2d.shape[0], batch_size)],
        dim=0,
    )
    pre_block: list[torch.Tensor] = []
    pre_mlp: list[torch.Tensor] = []
    for blk in model.blocks:
        if collect_layer_inputs:
            pre_block.append(xb)
        xm_l, mi_l, nx_l = [], [], []
        for b0 in range(0, xb.shape[0], batch_size):
            xm_b, mi_b = forward_block_pre_mlp(blk, xb[b0 : b0 + batch_size], cos, sin)
            xm_l.append(xm_b)
            mi_l.append(mi_b)
            nx_l.append(forward_block_post_mlp(blk, xm_b, mi_b))
        if collect_layer_inputs:
            pre_mlp.append(torch.cat(mi_l, dim=0))
        xb = torch.cat(nx_l, dim=0)
    return xb, pre_block, pre_mlp


@torch.no_grad()
def _compute_probe_p_null_by_layer(
    model: nn.Module,
    base_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    probe_ids: torch.Tensor,
    is_general: np.ndarray,
    cfg: CheckpointRescueConfig,
) -> dict[Any, dict[str, torch.Tensor]]:
    """Build per-layer ``{'x': P_x, 'act': P_act}`` null-space projectors from ``E0_*`` probe activations."""
    apply_checkpoint_blocks_inplace(model, base_blocks)
    model.eval()
    gen_ids = probe_ids[torch.as_tensor(is_general, dtype=torch.bool)]
    _, _, pre_mlp_list = _forward_pre_ln_and_layer_inputs(
        model, gen_ids, batch_size=cfg.batch_size, collect_layer_inputs=True
    )
    start_pos = max(0, cfg.score_from - 1)
    p_null_by_layer: dict[Any, dict[str, torch.Tensor]] = {}
    for l_idx, mlp_in in enumerate(pre_mlp_list):
        scored_x = mlp_in[:, start_pos:-1, :]
        px, _ = compute_prose_nullspace_projector_scaled(
            scored_x,
            sv_rel_tol=cfg.nullspace_sv_rel_tol,
            variance_threshold=cfg.nullspace_variance_threshold,
        )
        layer_entry: dict[str, torch.Tensor] = {"x": px}
        base_mlp = getattr(model.blocks[l_idx].mlp, "base_mlp", model.blocks[l_idx].mlp)
        if hasattr(base_mlp, "w1") and hasattr(base_mlp, "w2"):
            w_dt = base_mlp.w1.weight.dtype
            x_cast = scored_x.reshape(-1, scored_x.shape[-1]).to(dtype=w_dt)
            act = torch.nn.functional.silu(base_mlp.w1(x_cast).float()) * base_mlp.w2(x_cast).float()
            pact, _ = compute_prose_nullspace_projector_scaled(
                act,
                sv_rel_tol=cfg.nullspace_sv_rel_tol,
                variance_threshold=cfg.nullspace_variance_threshold,
            )
            layer_entry["act"] = pact
        p_null_by_layer[l_idx] = layer_entry
    return p_null_by_layer


@torch.no_grad()
def _compute_bidirectional_causal_attribution(
    model: nn.Module,
    base_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    sanitized_drift_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    quarantined_pre_ids: set[str],
    probe_ids: torch.Tensor,
    is_general: np.ndarray,
    cfg: CheckpointRescueConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    """Measure bidirectional (LOO + coupled AOI) causal ``general_harm_nats`` and ``specialist_gain_nats``.

    Combining Leave-One-Out (``D`` vs ``D_{-b}``) and Add-One-In (``B_{+b}`` vs ``B``, with
    same-layer drifted router/experts coupled when evaluating MoE experts/routers) eliminates
    ``LayerNorm`` saturation masking when multiple corrupted layers co-exist in a checkpoint.
    """
    cos = model.cos_cached.to(probe_ids.device)
    sin = model.sin_cached.to(probe_ids.device)
    ln_eps = float(getattr(model.ln, "eps", 1e-5))

    def _eval_from_pre_ln(h_pre: torch.Tensor) -> tuple[float, float]:
        nlls = (
            per_chunk_nll_from_pre_ln(
                h_pre,
                model.ln.weight,
                model.ln.bias,
                ln_eps,
                model.head.weight,
                probe_ids,
                score_from=cfg.score_from,
            )
            .cpu()
            .numpy()
        )
        return float(nlls[is_general].mean()), float(nlls[~is_general].mean())

    def _finish_from_layer(start_state: torch.Tensor, start_layer: int) -> tuple[float, float]:
        xb = start_state
        for blk in model.blocks[start_layer:]:
            nx_l = []
            for b0 in range(0, xb.shape[0], cfg.batch_size):
                xm_b, mi_b = forward_block_pre_mlp(blk, xb[b0 : b0 + cfg.batch_size], cos, sin)
                nx_l.append(forward_block_post_mlp(blk, xm_b, mi_b))
            xb = torch.cat(nx_l, dim=0)
        return _eval_from_pre_ln(xb)

    apply_checkpoint_blocks_inplace(model, sanitized_drift_blocks)
    model.eval()
    h_full_drift, pre_drift_states, _ = _forward_pre_ln_and_layer_inputs(
        model, probe_ids, batch_size=cfg.batch_size, collect_layer_inputs=True
    )
    full_gen, full_spec = _eval_from_pre_ln(h_full_drift)

    apply_checkpoint_blocks_inplace(model, base_blocks)
    h_full_base, pre_base_states, _ = _forward_pre_ln_and_layer_inputs(
        model, probe_ids, batch_size=cfg.batch_size, collect_layer_inputs=True
    )
    base_gen, base_spec = _eval_from_pre_ln(h_full_base)

    harm_map: dict[str, float] = {}
    gain_map: dict[str, float] = {}
    changed_by_layer: dict[int, list[str]] = {}
    for bid in sanitized_drift_blocks:
        if bid in quarantined_pre_ids:
            continue
        b_sub = base_blocks[bid]
        d_sub = sanitized_drift_blocks[bid]
        if any(not torch.equal(b_sub[k], d_sub[k]) for k in b_sub):
            li = _parse_layer_index(bid)
            if li is not None:
                changed_by_layer.setdefault(li, []).append(bid)

    for bid in sanitized_drift_blocks:
        if bid in quarantined_pre_ids:
            harm_map[bid] = 0.0
            gain_map[bid] = 0.0
            continue
        b_sub = base_blocks[bid]
        d_sub = sanitized_drift_blocks[bid]
        if all(torch.equal(b_sub[k], d_sub[k]) for k in b_sub):
            harm_map[bid] = 0.0
            gain_map[bid] = 0.0
            continue

        is_ln = _is_final_layernorm_block(bid, b_sub)
        l_idx = _parse_layer_index(bid)

        # 1. Leave-One-Out from sanitized drifted checkpoint
        apply_checkpoint_blocks_inplace(model, sanitized_drift_blocks)
        apply_checkpoint_blocks_inplace(model, {bid: b_sub})
        if is_ln:
            rev_gen, rev_spec = _eval_from_pre_ln(h_full_drift)
        else:
            assert l_idx is not None
            rev_gen, rev_spec = _finish_from_layer(pre_drift_states[l_idx], l_idx)
        loo_harm = full_gen - rev_gen
        loo_gain = rev_spec - full_spec

        # 2. Add-One-In on base checkpoint (coupling same-layer router/experts for MoE blocks)
        apply_checkpoint_blocks_inplace(model, base_blocks)
        if is_ln:
            apply_checkpoint_blocks_inplace(model, {bid: d_sub})
            aoi_gen, aoi_spec = _eval_from_pre_ln(h_full_base)
            ref_gen, ref_spec = base_gen, base_spec
        else:
            assert l_idx is not None
            router_bid = f"blocks.{l_idx}.mlp.router"
            if ".experts." in bid and router_bid in sanitized_drift_blocks:
                apply_checkpoint_blocks_inplace(model, {router_bid: sanitized_drift_blocks[router_bid]})
                ref_gen, ref_spec = _finish_from_layer(pre_base_states[l_idx], l_idx)
                apply_checkpoint_blocks_inplace(model, {bid: d_sub})
                aoi_gen, aoi_spec = _finish_from_layer(pre_base_states[l_idx], l_idx)
            elif bid.endswith(".router"):
                layer_exp_bids = [b for b in changed_by_layer.get(l_idx, []) if ".experts." in b]
                for eb in layer_exp_bids:
                    apply_checkpoint_blocks_inplace(model, {eb: sanitized_drift_blocks[eb]})
                ref_gen, ref_spec = _finish_from_layer(pre_base_states[l_idx], l_idx)
                apply_checkpoint_blocks_inplace(model, {bid: d_sub})
                aoi_gen, aoi_spec = _finish_from_layer(pre_base_states[l_idx], l_idx)
            else:
                ref_gen, ref_spec = base_gen, base_spec
                apply_checkpoint_blocks_inplace(model, {bid: d_sub})
                aoi_gen, aoi_spec = _finish_from_layer(pre_base_states[l_idx], l_idx)
        aoi_harm = aoi_gen - ref_gen
        aoi_gain = ref_spec - aoi_spec

        if loo_harm <= cfg.anchor_floor_nats and aoi_harm <= cfg.anchor_floor_nats:
            eff_harm = min(loo_harm, aoi_harm)
        elif max(loo_harm, aoi_harm) > cfg.corruption_ceiling_nats:
            eff_harm = max(loo_harm, aoi_harm)
        else:
            eff_harm = 0.5 * (loo_harm + aoi_harm)

        if min(loo_gain, aoi_gain) < -cfg.corruption_ceiling_nats and max(loo_gain, aoi_gain) <= 0.0:
            eff_gain = min(loo_gain, aoi_gain)
        else:
            eff_gain = max(loo_gain, aoi_gain)

        harm_map[bid] = eff_harm
        gain_map[bid] = eff_gain

    apply_checkpoint_blocks_inplace(model, sanitized_drift_blocks)
    return harm_map, gain_map


def _evaluate_checkpoint_state(
    *,
    model: nn.Module | None,
    blocks_norm: Mapping[str, Mapping[str, torch.Tensor]],
    blocks_public: Mapping[str, Any],
    eval_fn: Callable[[Any], Mapping[str, Any]] | None,
    eval_ids: torch.Tensor | None,
    eval_eids: Sequence[str],
    cfg: CheckpointRescueConfig,
) -> dict[str, Any]:
    """Evaluate a checkpoint state via ``eval_fn`` or exact causal mid-row NLL."""
    has_non_finite_weight = any(
        not bool(torch.isfinite(t).all().item())
        for sub in blocks_norm.values()
        for t in sub.values()
    )
    if has_non_finite_weight:
        return {"all_finite": False, "domain_nlls": {}, "general_nll": None, "spec_nll": None, "nll_mean": None}

    if model is not None:
        apply_checkpoint_blocks_inplace(model, blocks_norm)
        model.eval()
        eval_target: Any = model
    else:
        eval_target = blocks_public

    if eval_fn is not None:
        raw_res = eval_fn(eval_target)
        if not isinstance(raw_res, Mapping) or not raw_res:
            raise ValueError("eval_fn must return a non-empty mapping of domain -> NLL")
        domain_nlls: dict[str, float] = {}
        for k, v in raw_res.items():
            if k in ("non_finite_outputs", "all_finite"):
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"eval_fn[{k!r}] must be numeric, got {type(v).__name__}")
            fv = float(v)
            if not math.isfinite(fv):
                return {
                    "all_finite": False,
                    "domain_nlls": {},
                    "general_nll": None,
                    "spec_nll": None,
                    "nll_mean": None,
                }
            domain_nlls[str(k)] = fv
        if raw_res.get("non_finite_outputs") is True or raw_res.get("all_finite") is False:
            return {"all_finite": False, "domain_nlls": {}, "general_nll": None, "spec_nll": None, "nll_mean": None}
    else:
        assert model is not None and eval_ids is not None
        h_pre, _, _ = _forward_pre_ln_and_layer_inputs(model, eval_ids, batch_size=cfg.batch_size)
        if not bool(torch.isfinite(h_pre).all().item()):
            return {"all_finite": False, "domain_nlls": {}, "general_nll": None, "spec_nll": None, "nll_mean": None}
        nlls = per_chunk_nll_from_pre_ln(
            h_pre,
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
            f"Evaluation output must include both general ({cfg.general_prefix}*) and specialist domains"
        )
    all_vals = list(domain_nlls.values())
    return {
        "all_finite": True,
        "domain_nlls": domain_nlls,
        "general_nll": float(np.mean(gen_vals)),
        "spec_nll": float(np.mean(spec_vals)),
        "nll_mean": float(np.mean(all_vals)),
    }


def _score_against_base(
    eval_res: Mapping[str, Any],
    base_eval: Mapping[str, Any],
    cfg: CheckpointRescueConfig,
) -> dict[str, Any]:
    """Compute drift ratios, guard flags, and ANSE energy relative to ``base_eval``."""
    if not eval_res["all_finite"]:
        return {
            "all_finite": False,
            "general_drift_ratio": float("inf"),
            "worst_domain_ratio": float("inf"),
            "general_guard_ok": False,
            "worst_guard_ok": False,
            "nll_mean": None,
            "anse_energy": float(cfg.maximum_pain_energy),
        }

    base_doms = base_eval["domain_nlls"]
    cand_doms = eval_res["domain_nlls"]
    if set(base_doms.keys()) != set(cand_doms.keys()):
        return {
            "all_finite": False,
            "general_drift_ratio": float("inf"),
            "worst_domain_ratio": float("inf"),
            "general_guard_ok": False,
            "worst_guard_ok": False,
            "nll_mean": None,
            "anse_energy": float(cfg.maximum_pain_energy),
        }

    gen_ratio = float(eval_res["general_nll"]) / max(float(base_eval["general_nll"]), 1e-8)
    worst_ratio = max(float(cand_doms[d]) / max(float(base_doms[d]), 1e-8) for d in base_doms)
    mean_diff = float(eval_res["nll_mean"]) - float(base_eval["nll_mean"])

    gen_ok = gen_ratio <= cfg.max_general_drift_ratio + 1e-12
    worst_ok = worst_ratio <= cfg.max_worst_domain_drift_ratio + 1e-12
    if not (gen_ok and worst_ok):
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
        "nll_mean": float(eval_res["nll_mean"]),
        "anse_energy": float(anse_e),
    }


def _resolve_rescue_config(
    config: CheckpointRescueConfig | None,
    **overrides: Any,
) -> CheckpointRescueConfig:
    """Merge optional ``CheckpointRescueConfig`` with explicit keyword overrides."""
    if config is not None and not isinstance(config, CheckpointRescueConfig):
        raise TypeError(f"config must be CheckpointRescueConfig or None, got {type(config).__name__}")
    base_dict = asdict(config) if config is not None else asdict(CheckpointRescueConfig())
    for k, v in overrides.items():
        if v is not None:
            base_dict[k] = v
    return CheckpointRescueConfig(**base_dict)


@torch.no_grad()
def execute_verified_checkpoint_rescue(
    base_checkpoint: nn.Module | Mapping[str, Any],
    drifted_checkpoint: nn.Module | Mapping[str, Any],
    *,
    target_model: nn.Module | None = None,
    probe_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_fn: Callable[[Any], Mapping[str, Any]] | None = None,
    layer_general_harm_nats: Mapping[Any, float] | Sequence[float] | None = None,
    layer_specialist_gain_nats: Mapping[Any, float] | Sequence[float] | None = None,
    p_null_by_layer: torch.Tensor | Mapping[Any, Any] | None = None,
    config: CheckpointRescueConfig | None = None,
    anchor_floor_nats: float | None = None,
    corruption_ceiling_nats: float | None = None,
    fold_ln_lambda: float | None = None,
    min_raw_general_drift_ratio: float | None = None,
    max_general_drift_ratio: float | None = None,
    min_specialist_retention_ratio: float | None = None,
    rollback_target: str | None = None,
    inplace: bool | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run closed-form checkpoint rescue with causal attribution, null-space repair, and ANSE gating."""
    cfg = _resolve_rescue_config(
        config,
        anchor_floor_nats=anchor_floor_nats,
        corruption_ceiling_nats=corruption_ceiling_nats,
        fold_ln_lambda=fold_ln_lambda,
        min_raw_general_drift_ratio=min_raw_general_drift_ratio,
        max_general_drift_ratio=max_general_drift_ratio,
        min_specialist_retention_ratio=min_specialist_retention_ratio,
        rollback_target=rollback_target,
        inplace=inplace,
    )
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError(f"eval_fn must be callable or None, got {type(eval_fn).__name__}")
    if target_model is not None and not isinstance(target_model, nn.Module):
        raise TypeError(f"target_model must be an nn.Module or None, got {type(target_model).__name__}")

    base_norm, base_flags = _normalize_blocks_mapping(base_checkpoint, "base_checkpoint", require_finite=True)
    drift_norm, _ = _normalize_blocks_mapping(drifted_checkpoint, "drifted_checkpoint", require_finite=False)
    block_ids = _validate_matching_block_layouts(base_norm, drift_norm)
    base_public = _package_blocks_output(base_norm, base_flags)
    drift_public = _package_blocks_output(drift_norm, base_flags)

    # Resolve working nn.Module if one was provided
    work_model: nn.Module | None = None
    if target_model is not None:
        work_model = target_model if cfg.inplace else copy.deepcopy(target_model)
    elif isinstance(drifted_checkpoint, nn.Module):
        work_model = drifted_checkpoint if cfg.inplace else copy.deepcopy(drifted_checkpoint)
    elif isinstance(base_checkpoint, nn.Module):
        work_model = base_checkpoint if cfg.inplace else copy.deepcopy(base_checkpoint)

    if work_model is None and eval_fn is None:
        raise ValueError("Either an nn.Module checkpoint/target_model or eval_fn must be provided")

    vocab_size = int(work_model.head.weight.shape[0]) if work_model is not None else 0
    probe_ids: torch.Tensor | None = None
    probe_is_gen: np.ndarray | None = None
    probe_eids: list[str] = []
    if probe_chunks is not None and work_model is not None:
        probe_ids, probe_is_gen, probe_eids = _normalize_chunks(
            probe_chunks, vocab_size, cfg.general_prefix, "probe_chunks"
        )

    eval_ids: torch.Tensor | None = None
    eval_eids: list[str] = []
    if eval_chunks is not None and work_model is not None:
        eval_ids, _, eval_eids = _normalize_chunks(
            eval_chunks, vocab_size, cfg.general_prefix, "eval_chunks"
        )
    elif probe_ids is not None:
        eval_ids, eval_eids = probe_ids, probe_eids
    elif eval_fn is None:
        raise ValueError("Either probe_chunks/eval_chunks or eval_fn must be provided")

    base_sha = compute_checkpoint_sha256(base_public)
    raw_sha = compute_checkpoint_sha256(drift_public)

    # Pre-screen blocks so quarantined NaN/spectral-explosion blocks do not poison attribution or probes
    pre_quarantined: set[str] = set()
    sanitized_norm: dict[str, dict[str, torch.Tensor]] = {}
    for bid in block_ids:
        h_info = _inspect_block_health(
            base_norm[bid],
            drift_norm[bid],
            extreme_weight_threshold=cfg.extreme_weight_threshold,
            spectral_spike_ratio_threshold=cfg.spectral_spike_ratio_threshold,
            max_spectral_norm_inflation=cfg.max_spectral_norm_inflation,
            single_mode_spike_norm=cfg.single_mode_spike_norm,
        )
        if h_info["quarantined"]:
            pre_quarantined.add(bid)
            sanitized_norm[bid] = {sk: t.detach().clone() for sk, t in base_norm[bid].items()}
        else:
            sanitized_norm[bid] = {sk: t.detach().clone() for sk, t in drift_norm[bid].items()}

    was_training = work_model.training if work_model is not None else False
    if work_model is not None:
        work_model.eval()
    try:
        # 1. Resolve causal harm/gain attributions
        if layer_general_harm_nats is None or layer_specialist_gain_nats is None:
            if work_model is None or probe_ids is None or probe_is_gen is None:
                raise ValueError(
                    "layer_general_harm_nats and layer_specialist_gain_nats are required when model or probe_chunks is omitted"
                )
            auto_harm, auto_gain = _compute_bidirectional_causal_attribution(
                work_model, base_norm, sanitized_norm, pre_quarantined, probe_ids, probe_is_gen, cfg
            )
            harm_attr: Mapping[Any, float] | Sequence[float] = (
                auto_harm if layer_general_harm_nats is None else layer_general_harm_nats
            )
            gain_attr: Mapping[Any, float] | Sequence[float] = (
                auto_gain if layer_specialist_gain_nats is None else layer_specialist_gain_nats
            )
        else:
            harm_attr = layer_general_harm_nats
            gain_attr = layer_specialist_gain_nats

        # 2. Triage blocks
        triage_map = triage_drifted_checkpoint_blocks(
            base_public,
            drift_public,
            harm_attr,
            gain_attr,
            anchor_floor_nats=cfg.anchor_floor_nats,
            corruption_ceiling_nats=cfg.corruption_ceiling_nats,
            specialist_gain_floor_nats=cfg.specialist_gain_floor_nats,
            extreme_weight_threshold=cfg.extreme_weight_threshold,
            spectral_spike_ratio_threshold=cfg.spectral_spike_ratio_threshold,
            max_spectral_norm_inflation=cfg.max_spectral_norm_inflation,
            single_mode_spike_norm=cfg.single_mode_spike_norm,
        )

        # 3. Resolve P_0 projectors
        if p_null_by_layer is None:
            if work_model is None or probe_ids is None or probe_is_gen is None:
                raise ValueError("p_null_by_layer is required when model or probe_chunks is omitted")
            p_null_resolved: torch.Tensor | Mapping[Any, Any] = _compute_probe_p_null_by_layer(
                work_model, base_norm, probe_ids, probe_is_gen, cfg
            )
        else:
            p_null_resolved = p_null_by_layer

        # 4. Closed-form null-space repair
        cand_public, repair_report = repair_drifted_blocks_in_nullspace(
            base_public,
            drift_public,
            triage_map,
            p_null_resolved,
            fold_ln_lambda=cfg.fold_ln_lambda,
            projection_side=cfg.projection_side,
            nullspace_eig_threshold=cfg.nullspace_eig_threshold,
        )
        cand_norm, _ = _normalize_blocks_mapping(cand_public, "cand_public", require_finite=False)
        cand_sha = compute_checkpoint_sha256(cand_public)

        # 5. Evaluate base, raw_drifted, and candidate_rescued
        base_eval = _evaluate_checkpoint_state(
            model=work_model,
            blocks_norm=base_norm,
            blocks_public=base_public,
            eval_fn=eval_fn,
            eval_ids=eval_ids,
            eval_eids=eval_eids,
            cfg=cfg,
        )
        if not base_eval["all_finite"]:
            raise ValueError("base_checkpoint produced non-finite evaluation NLL")

        raw_eval = _evaluate_checkpoint_state(
            model=work_model,
            blocks_norm=drift_norm,
            blocks_public=drift_public,
            eval_fn=eval_fn,
            eval_ids=eval_ids,
            eval_eids=eval_eids,
            cfg=cfg,
        )
        raw_metrics = _score_against_base(raw_eval, base_eval, cfg)

        # If raw_eval had non-finite output due to quarantined blocks, measure healthy raw specialist gain
        if raw_eval["all_finite"]:
            raw_spec_gain = float(base_eval["spec_nll"]) - float(raw_eval["spec_nll"])
        elif pre_quarantined:
            sanitized_eval = _evaluate_checkpoint_state(
                model=work_model,
                blocks_norm=sanitized_norm,
                blocks_public=_package_blocks_output(sanitized_norm, base_flags),
                eval_fn=eval_fn,
                eval_ids=eval_ids,
                eval_eids=eval_eids,
                cfg=cfg,
            )
            raw_spec_gain = (
                float(base_eval["spec_nll"]) - float(sanitized_eval["spec_nll"])
                if sanitized_eval["all_finite"]
                else 0.0
            )
        else:
            raw_spec_gain = 0.0

        cand_eval = _evaluate_checkpoint_state(
            model=work_model,
            blocks_norm=cand_norm,
            blocks_public=cand_public,
            eval_fn=eval_fn,
            eval_ids=eval_ids,
            eval_eids=eval_eids,
            cfg=cfg,
        )
        cand_metrics = _score_against_base(cand_eval, base_eval, cfg)

        cand_spec_gain = (
            float(base_eval["spec_nll"]) - float(cand_eval["spec_nll"]) if cand_eval["all_finite"] else 0.0
        )
        retention_ratio = (
            cand_spec_gain / raw_spec_gain
            if (cand_eval["all_finite"] and raw_spec_gain > 1e-12)
            else 0.0
        )
        retention_ok = bool(
            cand_eval["all_finite"]
            and raw_spec_gain > 1e-12
            and cand_spec_gain > 0.0
            and retention_ratio >= cfg.min_specialist_retention_ratio - 1e-12
        )

        # Force ANSE Maximum Pain on candidate if specialist gain collapsed below retention threshold
        cand_anse_e = (
            float(cand_metrics["anse_energy"])
            if (cand_metrics["all_finite"] and cand_metrics["general_guard_ok"] and cand_metrics["worst_guard_ok"] and retention_ok)
            else float(cfg.maximum_pain_energy + max(0.0, float(cand_metrics["nll_mean"] or 0.0) - float(base_eval["nll_mean"])))
        )

        raw_anse_e = float(raw_metrics["anse_energy"])
        base_anse_e = 0.0
        delta_e_vs_base = cand_anse_e - base_anse_e
        delta_e_vs_raw = cand_anse_e - raw_anse_e

        raw_drift_triggered = (not raw_metrics["all_finite"]) or (
            float(raw_metrics["general_drift_ratio"]) > cfg.min_raw_general_drift_ratio
        )
        accepted = bool(
            raw_drift_triggered
            and cand_eval["all_finite"]
            and cand_metrics["general_guard_ok"]
            and cand_metrics["worst_guard_ok"]
            and retention_ok
            and delta_e_vs_base < 0.0
            and delta_e_vs_raw < 0.0
        )

        # 6. Atomic commit or bitwise rollback
        if accepted:
            status = "RESCUED_COMMITTED"
            rolled_back = False
            rollback_verified = False
            final_norm = cand_norm
            final_public = cand_public
            final_eval = cand_eval
            final_metrics = cand_metrics
            final_anse_e = cand_anse_e
            final_retention = retention_ratio
        else:
            rolled_back = True
            if not raw_drift_triggered:
                status = "ROLLED_BACK_RAW_DRIFT_BELOW_TRIGGER"
            else:
                status = "ROLLED_BACK_ANSE_MAXIMUM_PAIN"
            if cfg.rollback_target == "base":
                final_norm = base_norm
                final_public = base_public
                final_eval = base_eval
                final_metrics = _score_against_base(base_eval, base_eval, cfg)
                final_anse_e = base_anse_e
                final_retention = 0.0
                expected_rb_sha = base_sha
            else:
                final_norm = drift_norm
                final_public = drift_public
                final_eval = raw_eval
                final_metrics = raw_metrics
                final_anse_e = raw_anse_e
                final_retention = 1.0 if raw_spec_gain > 1e-12 else 0.0
                expected_rb_sha = raw_sha

            if work_model is not None:
                apply_checkpoint_blocks_inplace(work_model, final_norm)
                restored_blocks = extract_checkpoint_blocks(work_model)
                restored_sha = compute_checkpoint_sha256(restored_blocks)
            else:
                restored_sha = compute_checkpoint_sha256(final_public)
            rollback_verified = restored_sha == expected_rb_sha
            if not rollback_verified:
                raise RuntimeError(
                    f"Atomic bitwise rollback SHA-256 mismatch: expected {expected_rb_sha[:12]}, got {restored_sha[:12]}"
                )

        if accepted and work_model is not None:
            apply_checkpoint_blocks_inplace(work_model, final_norm)
            final_sha = compute_checkpoint_sha256(extract_checkpoint_blocks(work_model))
        else:
            final_sha = compute_checkpoint_sha256(final_public)

        receipt: dict[str, Any] = {
            "object": RESCUER_RECEIPT_OBJECT,
            "schema": RESCUER_SCHEMA,
            "status": status,
            "accepted": accepted,
            "rolled_back": rolled_back,
            "rollback_target": cfg.rollback_target,
            "rollback_bitwise_verified": rollback_verified,
            "raw_drift_triggered": raw_drift_triggered,
            "nll_mean": final_eval["nll_mean"],
            "base_nll_mean": float(base_eval["nll_mean"]),
            "raw_nll_mean": raw_eval["nll_mean"],
            "cand_nll_mean": cand_eval["nll_mean"],
            "general_drift_ratio": (
                float(final_metrics["general_drift_ratio"])
                if math.isfinite(float(final_metrics["general_drift_ratio"]))
                else None
            ),
            "raw_general_drift_ratio": (
                float(raw_metrics["general_drift_ratio"])
                if math.isfinite(float(raw_metrics["general_drift_ratio"]))
                else None
            ),
            "cand_general_drift_ratio": (
                float(cand_metrics["general_drift_ratio"])
                if math.isfinite(float(cand_metrics["general_drift_ratio"]))
                else None
            ),
            "specialist_retention_ratio": float(final_retention),
            "cand_specialist_retention_ratio": float(retention_ratio),
            "raw_spec_gain_nats": float(raw_spec_gain),
            "cand_spec_gain_nats": float(cand_spec_gain),
            "anse_energy": float(final_anse_e),
            "base_anse_energy": float(base_anse_e),
            "raw_anse_energy": float(raw_anse_e),
            "cand_anse_energy": float(cand_anse_e),
            "delta_e_vs_base": float(delta_e_vs_base),
            "delta_e_vs_raw": float(delta_e_vs_raw),
            "general_guard_ok": bool(cand_metrics["general_guard_ok"]),
            "worst_guard_ok": bool(cand_metrics["worst_guard_ok"]),
            "retention_ok": retention_ok,
            "triage_summary": triage_map["summary"],
            "quarantined_blocks": triage_map["quarantined_blocks"],
            "protected_specialist_anchors": triage_map["protected_specialist_anchors"],
            "salvageable_dual_active_blocks": triage_map["salvageable_dual_active_blocks"],
            "prose_corrupting_blocks": triage_map["prose_corrupting_blocks"],
            "max_nullspace_leakage_norm": float(repair_report["max_nullspace_leakage_norm"]),
            "fold_ln_lambda": float(cfg.fold_ln_lambda),
            "base_state_sha256": base_sha,
            "raw_drifted_state_sha256": raw_sha,
            "candidate_state_sha256": cand_sha,
            "final_state_sha256": final_sha,
            "config": asdict(cfg),
        }
        receipt["receipt_sha256"] = compute_payload_sha256(receipt)
        return (work_model if work_model is not None else final_public), receipt
    finally:
        if work_model is not None:
            work_model.train(was_training)


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "CheckpointRescueConfig",
    "DEFAULT_ANCHOR_FLOOR_NATS",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_CORRUPTION_CEILING_NATS",
    "DEFAULT_EXTREME_WEIGHT_THRESHOLD",
    "DEFAULT_FOLD_LN_LAMBDA",
    "DEFAULT_MAX_GENERAL_DRIFT_RATIO",
    "DEFAULT_MAX_SPECTRAL_NORM_INFLATION",
    "DEFAULT_MIN_RAW_DRIFT_RATIO",
    "DEFAULT_MIN_SPECIALIST_RETENTION_RATIO",
    "DEFAULT_NULLSPACE_EIG_THRESHOLD",
    "DEFAULT_NULLSPACE_SV_REL_TOL",
    "DEFAULT_NULLSPACE_VARIANCE_THRESHOLD",
    "DEFAULT_SINGLE_MODE_SPIKE_NORM",
    "DEFAULT_SPECTRAL_SPIKE_RATIO_THRESHOLD",
    "GENERAL_EXPERT_PREFIX",
    "RESCUER_RECEIPT_OBJECT",
    "RESCUER_SCHEMA",
    "VALID_PROJECTION_SIDES",
    "VALID_ROLLBACK_TARGETS",
    "apply_checkpoint_blocks_inplace",
    "compute_checkpoint_sha256",
    "execute_verified_checkpoint_rescue",
    "extract_checkpoint_blocks",
    "repair_drifted_blocks_in_nullspace",
    "triage_drifted_checkpoint_blocks",
    "verify_rescue_receipt_provenance",
]

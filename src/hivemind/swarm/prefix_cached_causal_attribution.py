# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Prefix-Cached Layerwise MoE Causal Attribution & Router-Expert Co-Adaptation Detector.

Implements training-free, CPU-only Weight-Space Programming (WSP) attribution and
cross-arm co-adaptation safety gates grounded in ``PILOT5`` on ``CROWDGPT-MOE-C3``:

1. ``compute_prefix_cached_layerwise_attribution(blocks, embeddings, eval_tail_fn, *, batch_size=24)``:
   Caches running prefix activations ``h_0 .. h_{L-1}`` during a single unablated
   ``L``-layer pass so ``L`` single-layer knockouts execute ``L + L(L+1)/2`` block
   evaluations instead of ``L^2`` (approaching ``2.0x`` speedup) with exact numerical parity.
2. ``detect_router_expert_coadaptation_mismatch(router_arm_logits, expert_arm_outputs, coadapted_nll, grafted_nll, *, max_mismatch_nats=0.02)``:
   Quantifies routing KL divergence and NLL co-adaptation loss when grafting cross-arm
   experts (e.g., ``PILOT5`` Arm B specialists onto frozen-router Arm D, which degraded
   ``selection_skip560`` to ``+0.0080`` nats vs co-adapted ``WSP_C3_B_rescued`` at
   ``-0.2964`` nats).
3. ``execute_coadaptation_guarded_surgery(...)``:
   Automatically vetoes mismatched cross-arm router grafts in favor of co-adapted
   selective-sink + OOF multi-gate rescue under ``ANSE Delta E < 0`` and the
   ``<= +1.0%`` general-prose / worst-domain guard, with atomic bitwise rollback and
   a canonical SHA-256 provenance receipt.
"""

from __future__ import annotations

import copy
import inspect
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
from hivemind.swarm.crowdgpt_moe_opendiloco import CrowdFineGrainedMoEMLP
from hivemind.swarm.domain_matched_moe_arm_composer import fast_affine_layernorm_temperature_grid
from hivemind.swarm.layer_localized_attribution import forward_block_dense_base_mlp
from hivemind.swarm.logit_calibration_temperature_surgeon import fold_temperature_into_norm_params
from hivemind.swarm.lora_svd_mode_surgery import decompose_lora_qr_svd
from hivemind.swarm.self_healing_checkpoint_rescuer import (
    apply_checkpoint_blocks_inplace,
    compute_checkpoint_sha256,
    extract_checkpoint_blocks,
)
from hivemind.swarm.tangent_task_vector_surgeon import compute_prose_nullspace_projector_scaled
from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    fit_uncentered_prose_probe_with_oof,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_nll_from_pre_ln,
)

COADAPTATION_SCHEMA: str = "hivemind.prefix_cached_causal_attribution/1"
COADAPTATION_RECEIPT_OBJECT: str = "hivemind.coadaptation_guarded_surgery_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_MAX_MISMATCH_NATS: float = 0.02
DEFAULT_MAX_ROUTING_KL: float = 0.25
DEFAULT_ANCHOR_HARM_THRESHOLD_NATS: float = -0.010
DEFAULT_SINK_PROSE_HARM_FLOOR_NATS: float = 0.0015
DEFAULT_SINK_SPEC_PRESERVE_FLOOR_NATS: float = 0.015
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_CALIBRATOR_GENERAL_GUARD_REL: float = 0.006
DEFAULT_NULLSPACE_SV_REL_TOL: float = 1e-3
DEFAULT_NULLSPACE_VARIANCE_THRESHOLD: float = 0.999
DEFAULT_NULLSPACE_EIG_THRESHOLD: float = 0.5
DEFAULT_NULLSPACE_LEAK_TOL: float = 1e-6
DEFAULT_MIN_OOF_SNR: float = 0.25
DEFAULT_OOF_ROUTER_BETA: float = 0.35
DEFAULT_GATE_BLEND_ALPHA: float = 0.25
DEFAULT_SCORE_FROM: int = 64
DEFAULT_BATCH_SIZE: int = 24
DEFAULT_LAMBDA_GRID: tuple[float, ...] = tuple(round(0.1 * idx, 2) for idx in range(0, 11))
DEFAULT_TAU_GRID: tuple[float, ...] = (0.98, 1.00, 1.02, 1.04)

VALID_ROUTING_MODES: frozenset[str] = frozenset({"sigma_topk", "softmax_topk", "softmax", "sigmoid"})
VALID_ROLLBACK_TARGETS: frozenset[str] = frozenset({"base", "frozen_arm", "unfrozen_arm"})

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_LAYER_KEY_RE = re.compile(r"^(?:blocks\.|layers\.|layer_)?(\d+)(?:\.|$)")
_SWIGLU_PROJECTIONS: tuple[str, ...] = ("w1", "w2", "w3")


# ---------------------------------------------------------------------------
# 1. Validation, Configuration & Receipt Provenance Helpers
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
    """Validate a finite real number in ``[0.0, 1.0]`` (or ``(0.0, 1.0]``)."""
    out = _validate_nonneg_finite(value, name)
    if not allow_zero and out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    if out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    return out


def _validate_fp_tensor(
    tensor: Any,
    name: str,
    *,
    exact_ndim: int | None = None,
    min_ndim: int | None = None,
) -> torch.Tensor:
    """Validate a non-empty finite floating-point tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if exact_ndim is not None and tensor.ndim != exact_ndim:
        raise ValueError(f"{name} must have ndim == {exact_ndim}, got shape {tuple(tensor.shape)}")
    if min_ndim is not None and tensor.ndim < min_ndim:
        raise ValueError(f"{name} must have ndim >= {min_ndim}, got shape {tuple(tensor.shape)}")
    if any(dim == 0 for dim in tensor.shape):
        raise ValueError(f"{name} must be non-empty, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    if not bool(torch.isfinite(tensor.detach()).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _validate_grid_values(
    grid: Sequence[float],
    name: str,
    *,
    unit_interval: bool = False,
) -> tuple[float, ...]:
    """Validate a non-empty numeric sequence for ``lambda_grid`` or positive ``tau_grid``."""
    if isinstance(grid, (str, bytes, bytearray)) or not isinstance(grid, Sequence) or len(grid) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of floats")
    cleaned: list[float] = []
    for idx, val in enumerate(grid):
        if unit_interval:
            cleaned.append(_validate_unit_interval(val, f"{name}[{idx}]", allow_zero=True))
        else:
            cleaned.append(_validate_positive_finite(val, f"{name}[{idx}]"))
    return tuple(dict.fromkeys(cleaned))


def _validate_expert_index_set(indices: Any, name: str) -> set[int]:
    """Validate a sequence or set of non-negative integer expert indices."""
    if isinstance(indices, (str, bytes, bytearray)) or not isinstance(indices, (Sequence, set, frozenset)):
        raise TypeError(f"{name} must be a sequence or set of non-negative integers")
    return {_validate_nonneg_int(item, f"{name} element") for item in indices}


def _validate_square_projector(
    proj: Any,
    name: str,
    *,
    expected_dim: int | None = None,
) -> torch.Tensor:
    """Validate a finite symmetric square 2D projector matrix on CPU ``float64``."""
    t = _validate_fp_tensor(proj, name, exact_ndim=2)
    if t.shape[0] != t.shape[1]:
        raise ValueError(f"{name} must be square, got shape {tuple(t.shape)}")
    if expected_dim is not None and int(t.shape[0]) != expected_dim:
        raise ValueError(f"{name} shape {tuple(t.shape)} must match ({expected_dim}, {expected_dim})")
    mat = t.detach().to(device="cpu", dtype=torch.float64).contiguous()
    asym = float((mat - mat.T).abs().max().item())
    scale = max(1.0, float(mat.abs().max().item()))
    if asym > 1e-4 * scale:
        raise ValueError(f"{name} must be symmetric (max asymmetry {asym:.3e})")
    return (0.5 * (mat + mat.T)).contiguous()


def verify_coadaptation_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != COADAPTATION_RECEIPT_OBJECT or receipt.get("schema") != COADAPTATION_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


@dataclass(frozen=True)
class CoAdaptationGuardedSurgeryConfig:
    """Validated configuration governing prefix-cached attribution and co-adaptation surgery."""

    max_mismatch_nats: float = DEFAULT_MAX_MISMATCH_NATS
    max_routing_kl: float = DEFAULT_MAX_ROUTING_KL
    routing_mode: str = "sigma_topk"
    top_k: int = 2
    cross_arm_gate_blend_alpha: float = DEFAULT_GATE_BLEND_ALPHA
    oof_router_beta: float = DEFAULT_OOF_ROUTER_BETA
    min_oof_snr: float = DEFAULT_MIN_OOF_SNR
    anchor_harm_threshold_nats: float = DEFAULT_ANCHOR_HARM_THRESHOLD_NATS
    sink_prose_harm_floor_nats: float = DEFAULT_SINK_PROSE_HARM_FLOOR_NATS
    sink_spec_preserve_floor_nats: float = DEFAULT_SINK_SPEC_PRESERVE_FLOOR_NATS
    auto_detect_sinks: bool = True
    auto_project_p_null: bool = True
    nullspace_sv_rel_tol: float = DEFAULT_NULLSPACE_SV_REL_TOL
    nullspace_variance_threshold: float = DEFAULT_NULLSPACE_VARIANCE_THRESHOLD
    nullspace_eig_threshold: float = DEFAULT_NULLSPACE_EIG_THRESHOLD
    nullspace_leak_tol: float = DEFAULT_NULLSPACE_LEAK_TOL
    fold_ln_lambda: float | None = None
    fold_tau_temp: float | None = None
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID
    tau_grid: tuple[float, ...] = DEFAULT_TAU_GRID
    calibrator_max_general_drift_rel: float = DEFAULT_CALIBRATOR_GENERAL_GUARD_REL
    max_general_drift_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    max_worst_domain_drift_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    enforce_all_domain_guard: bool = True
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    min_energy_improvement: float = 0.0
    veto_mismatched_cross_arm_graft: bool = True
    rollback_target: str = "base"
    general_prefix: str = GENERAL_EXPERT_PREFIX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    inplace: bool = True

    def __post_init__(self) -> None:
        _validate_nonneg_finite(self.max_mismatch_nats, "max_mismatch_nats")
        _validate_nonneg_finite(self.max_routing_kl, "max_routing_kl")
        if not isinstance(self.routing_mode, str) or self.routing_mode not in VALID_ROUTING_MODES:
            raise ValueError(f"routing_mode must be one of {sorted(VALID_ROUTING_MODES)}")
        _validate_positive_int(self.top_k, "top_k")
        _validate_unit_interval(self.cross_arm_gate_blend_alpha, "cross_arm_gate_blend_alpha", allow_zero=True)
        _validate_nonneg_finite(self.oof_router_beta, "oof_router_beta")
        _validate_nonneg_finite(self.min_oof_snr, "min_oof_snr")
        anchor_thr = _validate_finite_float(self.anchor_harm_threshold_nats, "anchor_harm_threshold_nats")
        if anchor_thr > 0.0:
            raise ValueError(f"anchor_harm_threshold_nats must be <= 0.0, got {anchor_thr}")
        _validate_nonneg_finite(self.sink_prose_harm_floor_nats, "sink_prose_harm_floor_nats")
        _validate_nonneg_finite(self.sink_spec_preserve_floor_nats, "sink_spec_preserve_floor_nats")
        _validate_bool(self.auto_detect_sinks, "auto_detect_sinks")
        _validate_bool(self.auto_project_p_null, "auto_project_p_null")
        _validate_nonneg_finite(self.nullspace_sv_rel_tol, "nullspace_sv_rel_tol")
        _validate_unit_interval(
            self.nullspace_variance_threshold, "nullspace_variance_threshold", allow_zero=False
        )
        eig_t = _validate_unit_interval(self.nullspace_eig_threshold, "nullspace_eig_threshold", allow_zero=True)
        if eig_t >= 1.0:
            raise ValueError(f"nullspace_eig_threshold must be < 1.0, got {eig_t}")
        _validate_nonneg_finite(self.nullspace_leak_tol, "nullspace_leak_tol")
        if self.fold_ln_lambda is not None:
            _validate_unit_interval(self.fold_ln_lambda, "fold_ln_lambda", allow_zero=True)
        if self.fold_tau_temp is not None:
            _validate_positive_finite(self.fold_tau_temp, "fold_tau_temp")
        _validate_grid_values(self.lambda_grid, "lambda_grid", unit_interval=True)
        _validate_grid_values(self.tau_grid, "tau_grid", unit_interval=False)
        _validate_nonneg_finite(self.calibrator_max_general_drift_rel, "calibrator_max_general_drift_rel")
        gen_r = _validate_positive_finite(self.max_general_drift_ratio, "max_general_drift_ratio")
        worst_r = _validate_positive_finite(self.max_worst_domain_drift_ratio, "max_worst_domain_drift_ratio")
        safe_r = _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        if gen_r < 1.0 or worst_r < 1.0 or safe_r < 1.0:
            raise ValueError("Drift ratio bounds must be >= 1.0")
        _validate_bool(self.enforce_all_domain_guard, "enforce_all_domain_guard")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_nonneg_finite(self.min_energy_improvement, "min_energy_improvement")
        _validate_bool(self.veto_mismatched_cross_arm_graft, "veto_mismatched_cross_arm_graft")
        if not isinstance(self.rollback_target, str) or self.rollback_target not in VALID_ROLLBACK_TARGETS:
            raise ValueError(f"rollback_target must be one of {sorted(VALID_ROLLBACK_TARGETS)}")
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_bool(self.inplace, "inplace")


# ---------------------------------------------------------------------------
# 2. Component (a): Prefix-Cached Layerwise MoE Causal Attribution Engine
# ---------------------------------------------------------------------------


def _block_accepts_rope(blk: Any) -> bool:
    """Return True if ``blk`` expects ``(x, cos, sin)`` RoPE arguments."""
    if hasattr(blk, "attn") and hasattr(blk, "ln1") and hasattr(blk, "mlp"):
        return True
    target = blk.forward if isinstance(blk, nn.Module) else blk
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False
    return "cos" in params and "sin" in params


def _resolve_rope_caches(
    blocks: Sequence[Any],
    embeddings: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Validate explicit RoPE caches or synthesize identity RoPE if a block requires it."""
    if (cos is None) != (sin is None):
        raise ValueError("cos and sin must either both be provided or both be None")
    if cos is not None and sin is not None:
        c_t = _validate_fp_tensor(cos, "cos", min_ndim=2).detach().to(device=embeddings.device)
        s_t = _validate_fp_tensor(sin, "sin", min_ndim=2).detach().to(device=embeddings.device)
        return c_t, s_t

    needs_rope = any(_block_accepts_rope(blk) for blk in blocks)
    if not needs_rope:
        return None, None

    seq_len = int(embeddings.shape[1]) if embeddings.ndim >= 3 else 1
    head_dim: int | None = None
    for blk in blocks:
        attn = getattr(blk, "attn", None)
        if attn is not None and hasattr(attn, "head_dim"):
            head_dim = int(attn.head_dim)
            break
    if head_dim is None:
        head_dim = int(embeddings.shape[-1])
    cos_id = torch.ones((1, 1, seq_len, head_dim), dtype=embeddings.dtype, device=embeddings.device)
    sin_id = torch.zeros_like(cos_id)
    return cos_id, sin_id


def _call_single_block_chunk(
    blk: Any,
    x_chunk: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
) -> torch.Tensor:
    """Invoke one transformer/MoE block on a single batch slice and return ``float32`` hidden states."""
    in_dt = x_chunk.dtype
    if isinstance(blk, nn.Module):
        ln1 = getattr(blk, "ln1", None)
        if ln1 is not None and isinstance(getattr(ln1, "weight", None), torch.Tensor):
            in_dt = ln1.weight.dtype
    x_in = x_chunk.to(dtype=in_dt)
    if _block_accepts_rope(blk):
        out = blk(x_in, cos, sin)
    else:
        out = blk(x_in)
    if isinstance(out, tuple):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"Block output must be a Tensor or tuple starting with Tensor, got {type(out).__name__}")
    return out.to(dtype=torch.float32)


def _forward_block_batched(
    blk: Any,
    x_all: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    batch_size: int,
) -> torch.Tensor:
    """Evaluate ``blk`` across ``x_all`` in slices of at most ``batch_size`` rows."""
    n_rows = int(x_all.shape[0])
    if n_rows <= batch_size:
        return _call_single_block_chunk(blk, x_all, cos, sin)
    parts = [
        _call_single_block_chunk(blk, x_all[b0 : b0 + batch_size], cos, sin)
        for b0 in range(0, n_rows, batch_size)
    ]
    return torch.cat(parts, dim=0)


def _eval_dense_fine_grained_mlp(mlp: CrowdFineGrainedMoEMLP, x_in: torch.Tensor) -> torch.Tensor:
    """Compute exact uniform dense sum over resident width-split experts."""
    orig_shape = x_in.shape
    orig_dt = x_in.dtype
    flat_x = x_in.reshape(-1, mlp.hidden_size)
    acc = torch.zeros((flat_x.shape[0], mlp.hidden_size), device=x_in.device, dtype=torch.float32)
    for e_idx in range(mlp.num_experts):
        if bool(mlp.resident_mask[e_idx].item()):
            acc = acc + mlp._run_expert(e_idx, flat_x)
    return acc.to(dtype=orig_dt).reshape(orig_shape)


def _forward_knockout_block_batched(
    blk: Any,
    h_prefix: torch.Tensor,
    layer_idx: int,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    batch_size: int,
    knockout_blocks: Sequence[Any] | None,
    knockout_fn: Callable[[Any, torch.Tensor, int], torch.Tensor] | None,
) -> torch.Tensor:
    """Evaluate a single knocked-out block at ``layer_idx`` without mutating weights."""
    if knockout_fn is not None:
        n_rows = int(h_prefix.shape[0])
        if n_rows <= batch_size:
            out = knockout_fn(blk, h_prefix, layer_idx)
            if isinstance(out, tuple):
                out = out[0]
            return out.to(dtype=torch.float32)
        parts = []
        for b0 in range(0, n_rows, batch_size):
            res = knockout_fn(blk, h_prefix[b0 : b0 + batch_size], layer_idx)
            parts.append((res[0] if isinstance(res, tuple) else res).to(dtype=torch.float32))
        return torch.cat(parts, dim=0)

    if knockout_blocks is not None:
        return _forward_block_batched(knockout_blocks[layer_idx], h_prefix, cos, sin, batch_size)

    if hasattr(blk, "forward_knockout") and callable(blk.forward_knockout):
        return _forward_block_batched(blk.forward_knockout, h_prefix, cos, sin, batch_size)

    hook_handle: Any = None
    try:
        mlp = getattr(blk, "mlp", None)
        if isinstance(mlp, nn.Module):
            if hasattr(mlp, "base_mlp"):
                def _base_mlp_hook(_mod: nn.Module, inp: tuple[Any, ...], out: Any) -> torch.Tensor:
                    dense = forward_block_dense_base_mlp(blk, inp[0])
                    return dense.to(device=out.device, dtype=out.dtype)

                hook_handle = mlp.register_forward_hook(_base_mlp_hook)
            elif isinstance(mlp, CrowdFineGrainedMoEMLP):
                def _fg_dense_hook(_mod: nn.Module, inp: tuple[Any, ...], out: Any) -> torch.Tensor:
                    dense = _eval_dense_fine_grained_mlp(mlp, inp[0])
                    return dense.to(device=out.device, dtype=out.dtype)

                hook_handle = mlp.register_forward_hook(_fg_dense_hook)
            elif hasattr(mlp, "router") and isinstance(mlp.router, nn.Module):
                hook_handle = mlp.router.register_forward_hook(
                    lambda _m, _inp, out: torch.zeros_like(out)
                )
            else:
                hook_handle = mlp.register_forward_hook(
                    lambda _m, _inp, out: torch.zeros_like(out)
                )
        elif isinstance(blk, nn.Module) and hasattr(blk, "router") and isinstance(blk.router, nn.Module):
            hook_handle = blk.router.register_forward_hook(
                lambda _m, _inp, out: torch.zeros_like(out)
            )
        else:
            return h_prefix.detach().to(dtype=torch.float32).clone()

        return _forward_block_batched(blk, h_prefix, cos, sin, batch_size)
    finally:
        if hook_handle is not None:
            hook_handle.remove()


def _parse_tail_eval_result(
    raw: Any,
    is_general_mask: np.ndarray | None,
    label: str,
) -> dict[str, Any]:
    """Normalize ``eval_tail_fn`` output into finite scalar NLL and optional general/spec split."""
    if isinstance(raw, torch.Tensor):
        t = _validate_fp_tensor(raw.detach(), label, min_ndim=0)
        flat = t.to(device="cpu", dtype=torch.float64).reshape(-1)
        out: dict[str, Any] = {"nll_mean": float(flat.mean().item())}
        if is_general_mask is not None and flat.numel() == int(is_general_mask.shape[0]):
            vals = flat.numpy()
            out["general_nll"] = float(np.mean(vals[is_general_mask]))
            if bool(np.any(~is_general_mask)):
                out["spec_nll"] = float(np.mean(vals[~is_general_mask]))
        return out

    if isinstance(raw, Mapping):
        if "nll_mean" in raw:
            nll_m = _validate_finite_float(raw["nll_mean"], f"{label}['nll_mean']")
            gen_v = (
                _validate_finite_float(raw["general_nll"], f"{label}['general_nll']")
                if "general_nll" in raw
                else None
            )
            spec_v = (
                _validate_finite_float(raw["spec_nll"], f"{label}['spec_nll']")
                if "spec_nll" in raw
                else None
            )
            return {"nll_mean": nll_m, "general_nll": gen_v, "spec_nll": spec_v}
        if not raw:
            raise ValueError(f"{label} mapping must not be empty")
        vals = [_validate_finite_float(v, f"{label}[{k!r}]") for k, v in raw.items()]
        return {"nll_mean": float(np.mean(vals)), "general_nll": None, "spec_nll": None}

    return {"nll_mean": _validate_finite_float(raw, label), "general_nll": None, "spec_nll": None}


def _normalize_general_mask(is_general_mask: Any, n_samples: int) -> np.ndarray | None:
    """Validate optional boolean mask partitioning general-prose from specialist samples."""
    if is_general_mask is None:
        return None
    if isinstance(is_general_mask, torch.Tensor):
        if is_general_mask.dtype != torch.bool or is_general_mask.ndim != 1:
            raise ValueError("is_general_mask Tensor must be 1D bool")
        arr = is_general_mask.detach().cpu().numpy()
    elif isinstance(is_general_mask, np.ndarray):
        if is_general_mask.dtype != bool or is_general_mask.ndim != 1:
            raise ValueError("is_general_mask ndarray must be 1D bool")
        arr = is_general_mask
    elif isinstance(is_general_mask, Sequence) and not isinstance(is_general_mask, (str, bytes)):
        arr = np.array([_validate_bool(flag, "is_general_mask element") for flag in is_general_mask], dtype=bool)
    else:
        raise TypeError("is_general_mask must be a 1D bool sequence, ndarray, or Tensor")
    if int(arr.shape[0]) != n_samples or not bool(np.any(arr)):
        raise ValueError(f"is_general_mask must have length {n_samples} and contain at least one True")
    return arr


@torch.no_grad()
def compute_prefix_cached_layerwise_attribution(
    blocks: Sequence[Any],
    embeddings: torch.Tensor,
    eval_tail_fn: Callable[[torch.Tensor], Any],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    knockout_blocks: Sequence[Any] | None = None,
    knockout_fn: Callable[[Any, torch.Tensor, int], torch.Tensor] | None = None,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    anchor_harm_threshold_nats: float = DEFAULT_ANCHOR_HARM_THRESHOLD_NATS,
    is_general_mask: Any = None,
) -> dict[str, Any]:
    """Compute exact layerwise MoE knockout attribution caching running prefix activations.

    During the initial unablated pass through ``L`` blocks, caches the running prefix hidden
    states ``h_0, h_1, ..., h_{L-1}``. Each single-layer knockout ``l`` starts directly at
    ``h_l`` (evaluating the knocked-out block ``l`` plus tail blocks ``l+1 .. L-1``), requiring
    ``L + L(L+1)/2`` total block calls instead of ``L^2`` (or ``L + L^2`` with baseline),
    halving evaluation cost (~2.0x speedup) with bitwise-exact numerical parity.
    """
    if isinstance(blocks, (str, bytes)) or not isinstance(blocks, (Sequence, nn.ModuleList)) or len(blocks) == 0:
        raise ValueError("blocks must be a non-empty sequence or nn.ModuleList")
    if not callable(eval_tail_fn):
        raise TypeError(f"eval_tail_fn must be callable, got {type(eval_tail_fn).__name__}")
    bsz = _validate_positive_int(batch_size, "batch_size")
    emb = _validate_fp_tensor(embeddings, "embeddings", min_ndim=2).detach()
    anchor_thr = _validate_finite_float(anchor_harm_threshold_nats, "anchor_harm_threshold_nats")
    if anchor_thr > 0.0:
        raise ValueError(f"anchor_harm_threshold_nats must be <= 0.0, got {anchor_thr}")
    if knockout_blocks is not None and len(knockout_blocks) != len(blocks):
        raise ValueError(f"knockout_blocks length {len(knockout_blocks)} != len(blocks) {len(blocks)}")
    if knockout_fn is not None and not callable(knockout_fn):
        raise TypeError(f"knockout_fn must be callable or None, got {type(knockout_fn).__name__}")

    gen_mask = _normalize_general_mask(is_general_mask, int(emb.shape[0]))
    cos_eff, sin_eff = _resolve_rope_caches(blocks, emb, cos, sin)
    n_layers = len(blocks)

    was_training = [blk.training for blk in blocks if isinstance(blk, nn.Module)]
    for blk in blocks:
        if isinstance(blk, nn.Module):
            blk.eval()

    try:
        # Pass 1: single unablated forward pass caching running prefix states h_0..h_{L-1} (L calls)
        prefix_states: list[torch.Tensor] = []
        xb = emb.to(dtype=torch.float32)
        for blk in blocks:
            prefix_states.append(xb)
            xb = _forward_block_batched(blk, xb, cos_eff, sin_eff, bsz)

        baseline_eval = _parse_tail_eval_result(eval_tail_fn(xb), gen_mask, "eval_tail_fn(baseline)")
        base_nll = float(baseline_eval["nll_mean"])

        # Pass 2: L single-layer knockouts resuming from cached h_l (L(L+1)/2 calls)
        layers_out: list[dict[str, Any]] = []
        for l_idx in range(n_layers):
            x_cur = _forward_knockout_block_batched(
                blocks[l_idx],
                prefix_states[l_idx],
                l_idx,
                cos_eff,
                sin_eff,
                bsz,
                knockout_blocks,
                knockout_fn,
            )
            for nxt in range(l_idx + 1, n_layers):
                x_cur = _forward_block_batched(blocks[nxt], x_cur, cos_eff, sin_eff, bsz)

            ko_eval = _parse_tail_eval_result(eval_tail_fn(x_cur), gen_mask, f"eval_tail_fn(layer={l_idx})")
            ko_nll = float(ko_eval["nll_mean"])
            harm_nats = base_nll - ko_nll

            row: dict[str, Any] = {
                "layer": l_idx,
                "nll_baseline": base_nll,
                "nll_knockout": ko_nll,
                "delta_nll_when_knocked_out": ko_nll - base_nll,
                "harm_nats": harm_nats,
                "benefit_nats": ko_nll - base_nll,
                "is_beneficial_anchor": bool(harm_nats <= anchor_thr),
                "block_calls_for_layer": n_layers - l_idx,
            }
            if baseline_eval.get("general_nll") is not None and ko_eval.get("general_nll") is not None:
                g_harm = float(baseline_eval["general_nll"]) - float(ko_eval["general_nll"])
                row["general_harm_nats"] = g_harm
                row["is_beneficial_anchor"] = bool(g_harm <= anchor_thr)
            if baseline_eval.get("spec_nll") is not None and ko_eval.get("spec_nll") is not None:
                row["spec_benefit_nats"] = float(ko_eval["spec_nll"]) - float(baseline_eval["spec_nll"])
            layers_out.append(row)
    finally:
        mod_iter = iter(was_training)
        for blk in blocks:
            if isinstance(blk, nn.Module):
                blk.train(next(mod_iter))

    prefix_calls = n_layers
    knockout_calls = (n_layers * (n_layers + 1)) // 2
    total_calls = prefix_calls + knockout_calls
    naive_knockout_calls = n_layers * n_layers
    naive_total_calls = n_layers + naive_knockout_calls
    saved_calls = naive_total_calls - total_calls

    ranked_layers = [
        int(r["layer"]) for r in sorted(layers_out, key=lambda d: (-float(d["harm_nats"]), int(d["layer"])))
    ]
    anchor_layers = [int(r["layer"]) for r in layers_out if r["is_beneficial_anchor"]]

    return {
        "schema": COADAPTATION_SCHEMA,
        "num_layers": n_layers,
        "num_samples": int(emb.shape[0]),
        "batch_size": bsz,
        "nll_mean": base_nll,
        "prefix_block_calls": prefix_calls,
        "knockout_block_calls": knockout_calls,
        "total_block_calls": total_calls,
        "naive_knockout_block_calls": naive_knockout_calls,
        "naive_total_block_calls": naive_total_calls,
        "block_evals_saved": saved_calls,
        "block_evals_saved_ratio": float(saved_calls) / float(naive_total_calls),
        "speedup_factor": float(naive_total_calls) / float(total_calls),
        "knockout_speedup_factor": float(naive_knockout_calls) / float(knockout_calls),
        "anchor_layers": anchor_layers,
        "ranked_layers_by_harm": ranked_layers,
        "layers": layers_out,
    }


# ---------------------------------------------------------------------------
# 3. Component (b): Router-Expert Co-Adaptation Mismatch Detector
# ---------------------------------------------------------------------------


def _flatten_tensor_or_mapping(
    source: Any,
    name: str,
    *,
    min_ndim: int = 2,
) -> torch.Tensor:
    """Flatten a Tensor or per-layer Mapping of Tensors onto CPU ``float64``."""
    if isinstance(source, torch.Tensor):
        t = _validate_fp_tensor(source, name, min_ndim=min_ndim)
        return t.detach().to(device="cpu", dtype=torch.float64).reshape(-1, int(t.shape[-1])).contiguous()

    if isinstance(source, Mapping) and source:
        parts: list[torch.Tensor] = []
        width: int | None = None
        for k in sorted(source.keys(), key=lambda key: str(key)):
            t = _validate_fp_tensor(source[k], f"{name}[{k!r}]", min_ndim=min_ndim)
            w = int(t.shape[-1])
            if width is None:
                width = w
            elif w != width:
                raise ValueError(f"{name} trailing width mismatch: {w} != {width}")
            parts.append(t.detach().to(device="cpu", dtype=torch.float64).reshape(-1, w))
        return torch.cat(parts, dim=0).contiguous()

    raise TypeError(f"{name} must be a non-empty Tensor or Mapping[Any, Tensor], got {type(source).__name__}")


def _flatten_per_expert_3d(source: Any, num_experts: int, name: str) -> torch.Tensor:
    """Flatten a ``(..., E, H)`` per-expert output tensor/mapping into ``(N, E, H) float64``."""
    if isinstance(source, torch.Tensor):
        t = _validate_fp_tensor(source, name, min_ndim=3)
        if int(t.shape[-2]) != num_experts:
            raise ValueError(f"{name} expert axis {t.shape[-2]} != {num_experts}")
        return t.detach().to(device="cpu", dtype=torch.float64).reshape(-1, num_experts, int(t.shape[-1]))

    if isinstance(source, Mapping) and source:
        parts = [_flatten_per_expert_3d(source[k], num_experts, f"{name}[{k!r}]") for k in sorted(source, key=str)]
        return torch.cat(parts, dim=0)

    raise TypeError(f"{name} must be a 3D+ Tensor or Mapping of 3D+ Tensors")


def _logits_to_routing_probs(logits: torch.Tensor, routing_mode: str) -> torch.Tensor:
    """Convert raw router logits ``(N, E)`` into normalized routing probability vectors ``(N, E)``."""
    if routing_mode in ("softmax", "softmax_topk"):
        return F.softmax(logits, dim=-1).clamp(min=1e-12, max=1.0)
    sig = torch.sigmoid(logits)
    return (sig / sig.sum(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(min=1e-12, max=1.0)


def _compute_kl_and_jsd(p_target: torch.Tensor, p_host: torch.Tensor) -> tuple[float, float, float]:
    """Compute ``(KL(p_target || p_host), KL(p_host || p_target), JSD)`` in nats."""
    eps = 1e-12
    pt = p_target.clamp(min=eps, max=1.0)
    ph = p_host.clamp(min=eps, max=1.0)
    kl_fwd = max(0.0, float((pt * (pt.log() - ph.log())).sum(dim=-1).mean().item()))
    kl_rev = max(0.0, float((ph * (ph.log() - pt.log())).sum(dim=-1).mean().item()))
    pm = (0.5 * (pt + ph)).clamp(min=eps, max=1.0)
    jsd = max(
        0.0,
        0.5 * float((pt * (pt.log() - pm.log())).sum(dim=-1).mean().item())
        + 0.5 * float((ph * (ph.log() - pm.log())).sum(dim=-1).mean().item()),
    )
    return kl_fwd, kl_rev, jsd


def _extract_coadaptation_inputs(
    router_arm_logits: Any,
    expert_arm_outputs: Any,
    expert_arm_router_logits: Any,
    grafted_outputs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, bool]:
    """Normalize heterogeneous router-logit and expert-output container inputs."""
    explicit_exp_logits: torch.Tensor | None = None

    # Parse router_arm_logits (single tensor/per-layer dict, pair, or {"router_arm": ..., "expert_arm": ...})
    if isinstance(router_arm_logits, (tuple, list)):
        if len(router_arm_logits) != 2:
            raise ValueError("router_arm_logits tuple/list must have length 2: (router_logits, expert_router_logits)")
        r_logits = _flatten_tensor_or_mapping(router_arm_logits[0], "router_arm_logits[0]")
        explicit_exp_logits = _flatten_tensor_or_mapping(router_arm_logits[1], "router_arm_logits[1]")
    elif isinstance(router_arm_logits, Mapping) and (
        "router_arm" in router_arm_logits or "frozen" in router_arm_logits
    ):
        host_key = "router_arm" if "router_arm" in router_arm_logits else "frozen"
        r_logits = _flatten_tensor_or_mapping(router_arm_logits[host_key], f"router_arm_logits[{host_key!r}]")
        for cand in ("expert_arm", "unfrozen", "coadapted"):
            if cand in router_arm_logits:
                explicit_exp_logits = _flatten_tensor_or_mapping(
                    router_arm_logits[cand], f"router_arm_logits[{cand!r}]"
                )
                break
    else:
        r_logits = _flatten_tensor_or_mapping(router_arm_logits, "router_arm_logits")

    num_experts = int(r_logits.shape[-1])
    if num_experts < 2:
        raise ValueError(f"router_arm_logits must expose >= 2 experts, got {num_experts}")

    if expert_arm_router_logits is not None:
        explicit_exp_logits = _flatten_tensor_or_mapping(expert_arm_router_logits, "expert_arm_router_logits")

    coad_out: torch.Tensor | None = None
    per_expert_y: torch.Tensor | None = None
    graft_out: torch.Tensor | None = (
        _flatten_tensor_or_mapping(grafted_outputs, "grafted_outputs") if grafted_outputs is not None else None
    )

    if isinstance(expert_arm_outputs, Mapping):
        if explicit_exp_logits is None:
            for rk in ("router_logits", "expert_router_logits", "coadapted_router_logits"):
                if rk in expert_arm_outputs:
                    explicit_exp_logits = _flatten_tensor_or_mapping(
                        expert_arm_outputs[rk], f"expert_arm_outputs[{rk!r}]"
                    )
                    break
        if "per_expert_outputs" in expert_arm_outputs:
            per_expert_y = _flatten_per_expert_3d(
                expert_arm_outputs["per_expert_outputs"], num_experts, "expert_arm_outputs['per_expert_outputs']"
            )
        for ok in ("coadapted_outputs", "outputs", "expert_outputs"):
            if ok in expert_arm_outputs:
                raw_o = expert_arm_outputs[ok]
                if isinstance(raw_o, torch.Tensor) and raw_o.ndim >= 3 and int(raw_o.shape[-2]) == num_experts:
                    per_expert_y = _flatten_per_expert_3d(raw_o, num_experts, f"expert_arm_outputs[{ok!r}]")
                else:
                    coad_out = _flatten_tensor_or_mapping(raw_o, f"expert_arm_outputs[{ok!r}]")
                break
        if graft_out is None and "grafted_outputs" in expert_arm_outputs:
            graft_out = _flatten_tensor_or_mapping(
                expert_arm_outputs["grafted_outputs"], "expert_arm_outputs['grafted_outputs']"
            )
        if explicit_exp_logits is None and per_expert_y is None and coad_out is None:
            # Treat mapping as per-layer expert outputs or router logits
            first_v = next(iter(expert_arm_outputs.values()))
            if isinstance(first_v, torch.Tensor) and first_v.ndim >= 3 and int(first_v.shape[-2]) == num_experts:
                per_expert_y = _flatten_per_expert_3d(expert_arm_outputs, num_experts, "expert_arm_outputs")
            else:
                flat_m = _flatten_tensor_or_mapping(expert_arm_outputs, "expert_arm_outputs")
                if flat_m.shape == r_logits.shape:
                    explicit_exp_logits = flat_m
                else:
                    coad_out = flat_m
    elif isinstance(expert_arm_outputs, torch.Tensor):
        t_eo = _validate_fp_tensor(expert_arm_outputs, "expert_arm_outputs", min_ndim=2)
        if t_eo.ndim >= 3 and int(t_eo.shape[-2]) == num_experts and int(t_eo.shape[-1]) != num_experts:
            per_expert_y = _flatten_per_expert_3d(t_eo, num_experts, "expert_arm_outputs")
        else:
            flat_eo = _flatten_tensor_or_mapping(t_eo, "expert_arm_outputs")
            if explicit_exp_logits is None and flat_eo.shape == r_logits.shape:
                explicit_exp_logits = flat_eo
            else:
                coad_out = flat_eo
    else:
        raise TypeError(
            f"expert_arm_outputs must be a torch.Tensor or Mapping, got {type(expert_arm_outputs).__name__}"
        )

    if explicit_exp_logits is not None and explicit_exp_logits.shape != r_logits.shape:
        raise ValueError(
            f"Expert router logits shape {tuple(explicit_exp_logits.shape)} != router_arm_logits shape {tuple(r_logits.shape)}"
        )
    has_explicit_pair_logits = explicit_exp_logits is not None
    return r_logits, explicit_exp_logits, per_expert_y, coad_out, graft_out, has_explicit_pair_logits


def detect_router_expert_coadaptation_mismatch(
    router_arm_logits: torch.Tensor | Mapping[str, Any] | Sequence[torch.Tensor],
    expert_arm_outputs: torch.Tensor | Mapping[str, Any],
    coadapted_nll: float,
    grafted_nll: float,
    *,
    max_mismatch_nats: float = DEFAULT_MAX_MISMATCH_NATS,
    max_routing_kl: float = DEFAULT_MAX_ROUTING_KL,
    routing_mode: str = "softmax",
    top_k: int = 2,
    expert_arm_router_logits: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    grafted_outputs: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Quantify routing KL divergence and NLL co-adaptation loss when grafting cross-arm experts.

    Grounded in ``PILOT5`` (``CROWDGPT-MOE-C3``): grafting Arm B's specialists onto Arm D's
    frozen router degraded ``selection_skip560`` to ``+0.0080`` nats compared to co-adapted
    ``WSP_C3_B_rescued`` at ``-0.2964`` nats (``+0.3044`` nats co-adaptation loss).
    """
    coad_nll = _validate_finite_float(coadapted_nll, "coadapted_nll")
    graft_nll = _validate_finite_float(grafted_nll, "grafted_nll")
    max_nats = _validate_nonneg_finite(max_mismatch_nats, "max_mismatch_nats")
    max_kl = _validate_nonneg_finite(max_routing_kl, "max_routing_kl")
    if not isinstance(routing_mode, str) or routing_mode not in VALID_ROUTING_MODES:
        raise ValueError(f"routing_mode must be one of {sorted(VALID_ROUTING_MODES)}")
    k_val = _validate_positive_int(top_k, "top_k")

    r_logits, exp_logits, per_expert_y, coad_out, graft_out, has_explicit_logits = (
        _extract_coadaptation_inputs(
            router_arm_logits, expert_arm_outputs, expert_arm_router_logits, grafted_outputs
        )
    )
    num_experts = int(r_logits.shape[-1])
    k_eff = min(k_val, num_experts)
    p_router = _logits_to_routing_probs(r_logits, routing_mode)

    kl_source = "single_router_only"
    p_expert: torch.Tensor | None = None
    if exp_logits is not None:
        p_expert = _logits_to_routing_probs(exp_logits, routing_mode)
        kl_source = "explicit_router_logits"
    elif per_expert_y is not None:
        if int(per_expert_y.shape[0]) != int(r_logits.shape[0]):
            raise ValueError(
                f"per_expert_outputs row count {per_expert_y.shape[0]} != router_arm_logits {r_logits.shape[0]}"
            )
        norms = torch.linalg.vector_norm(per_expert_y, dim=-1).clamp_min(1e-12)
        p_expert = (norms / norms.sum(dim=-1, keepdim=True)).clamp(min=1e-12, max=1.0)
        kl_source = "expert_output_norm_affinity"

    if p_expert is not None:
        kl_fwd, kl_rev, jsd = _compute_kl_and_jsd(p_expert, p_router)
        sym_kl = 0.5 * (kl_fwd + kl_rev)
        top_r = torch.topk(p_router, k=k_eff, dim=-1).indices
        top_e = torch.topk(p_expert, k=k_eff, dim=-1).indices
        top1_agree = float((top_r[:, 0] == top_e[:, 0]).to(torch.float64).mean().item())
        overlap_m = (top_r.unsqueeze(-1) == top_e.unsqueeze(-2)).any(dim=-1)
        topk_overlap = float(overlap_m.to(torch.float64).mean().item())
    else:
        kl_fwd, kl_rev, sym_kl, jsd = 0.0, 0.0, 0.0, 0.0
        top1_agree = 1.0
        topk_overlap = 1.0

    # Optional output-space alignment diagnostics
    out_l2 = 0.0
    out_rel = 0.0
    out_cos = 1.0
    if per_expert_y is not None and p_expert is not None:
        y_router = (p_router.unsqueeze(-1) * per_expert_y).sum(dim=1)
        y_coad = (p_expert.unsqueeze(-1) * per_expert_y).sum(dim=1)
        diff_y = y_coad - y_router
        out_l2 = float(torch.linalg.vector_norm(diff_y, dim=-1).mean().item())
        denom = float(torch.linalg.vector_norm(y_coad, dim=-1).mean().item())
        out_rel = out_l2 / max(denom, 1e-12)
        out_cos = float(F.cosine_similarity(y_coad, y_router, dim=-1, eps=1e-12).mean().item())
    elif coad_out is not None and graft_out is not None:
        if coad_out.shape != graft_out.shape:
            raise ValueError(
                f"coadapted outputs shape {tuple(coad_out.shape)} != grafted_outputs shape {tuple(graft_out.shape)}"
            )
        diff_y = coad_out - graft_out
        out_l2 = float(torch.linalg.vector_norm(diff_y, dim=-1).mean().item())
        denom = float(torch.linalg.vector_norm(coad_out, dim=-1).mean().item())
        out_rel = out_l2 / max(denom, 1e-12)
        out_cos = float(F.cosine_similarity(coad_out, graft_out, dim=-1, eps=1e-12).mean().item())

    mismatch_nats = graft_nll - coad_nll
    nll_exceeded = bool(mismatch_nats > max_nats)
    kl_exceeded = bool(has_explicit_logits and kl_fwd > max_kl)
    mismatch_detected = bool(nll_exceeded or kl_exceeded)

    return {
        "schema": COADAPTATION_SCHEMA,
        "coadapted_nll": coad_nll,
        "grafted_nll": graft_nll,
        "coadaptation_mismatch_nats": mismatch_nats,
        "max_mismatch_nats": max_nats,
        "routing_kl_divergence": kl_fwd,
        "reverse_routing_kl": kl_rev,
        "symmetric_routing_kl": sym_kl,
        "routing_jsd": jsd,
        "max_routing_kl": max_kl,
        "topk_overlap_rate": topk_overlap,
        "top1_agreement_rate": top1_agree,
        "output_l2_mismatch": out_l2,
        "output_relative_drift": out_rel,
        "output_cosine_similarity": out_cos,
        "kl_source": kl_source,
        "nll_mismatch_exceeded": nll_exceeded,
        "routing_kl_exceeded": kl_exceeded,
        "mismatch_detected": mismatch_detected,
        "veto_cross_arm_graft": mismatch_detected,
        "recommended_action": (
            "veto_cross_arm_graft_use_coadapted_rescue"
            if mismatch_detected
            else "accept_compatible_coadaptation"
        ),
    }


# ---------------------------------------------------------------------------
# 4. Component (c): Co-Adaptation-Guarded MoE Surgery & ANSE Ratchet Gate
# ---------------------------------------------------------------------------


def _parse_layer_from_block_id(block_id: str) -> int | None:
    """Extract integer layer index from block IDs like ``blocks.0.mlp.router``."""
    m = _LAYER_KEY_RE.search(block_id.strip())
    return int(m.group(1)) if m is not None else None


def _normalize_checkpoint_to_blocks(
    ckpt: nn.Module | Mapping[str, Any],
    name: str,
) -> dict[str, dict[str, torch.Tensor]]:
    """Normalize an ``nn.Module`` or block mapping into cloned finite block tensors."""
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
                    raise ValueError(f"{name}[{bid!r}][{sk!r}] must be a non-empty floating-point Tensor")
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
    """Verify identical MoE block structure across base, frozen-router, and unfrozen arms."""
    ref_keys = set(base_blocks.keys())
    if not any(k.endswith(".router") for k in ref_keys) or not any(".experts." in k for k in ref_keys):
        raise ValueError("Checkpoints must expose MoE '.router' and '.experts.' blocks")

    for arm_name, arm_blk in (("frozen_router_checkpoint", frozen_blocks), ("unfrozen_arm_checkpoint", unfrozen_blocks)):
        if set(arm_blk.keys()) != ref_keys:
            raise ValueError(f"{arm_name} block layout does not match base_checkpoint")
        for bid in sorted(ref_keys):
            b_sub, a_sub = base_blocks[bid], arm_blk[bid]
            if set(b_sub.keys()) != set(a_sub.keys()):
                raise ValueError(f"Sub-key mismatch on block {bid!r} for {arm_name}")
            for sk in b_sub:
                if b_sub[sk].shape != a_sub[sk].shape:
                    raise ValueError(f"Tensor shape mismatch on {bid}.{sk} for {arm_name}")


def _validate_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    vocab_size: int,
    general_prefix: str,
    score_from: int,
    name: str,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Validate token chunks accepting either ``'expert_id'`` or ``'domain'`` labels."""
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence) or not chunks:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")
    rows: list[torch.Tensor] = []
    eids: list[str] = []
    seq_len: int | None = None
    for idx, ch in enumerate(chunks):
        if not isinstance(ch, Mapping) or "ids" not in ch:
            raise ValueError(f"{name}[{idx}] must contain 'ids' and 'expert_id'/'domain'")
        raw_eid = ch.get("expert_id", ch.get("domain"))
        if raw_eid is None or not str(raw_eid).strip():
            raise ValueError(f"{name}[{idx}] must contain a non-empty 'expert_id' or 'domain'")
        ids_t = torch.as_tensor(ch["ids"], dtype=torch.long)
        if ids_t.ndim != 1 or ids_t.numel() < 2:
            raise ValueError(f"{name}[{idx}]['ids'] must be 1D with length >= 2")
        if bool((ids_t < 0).any().item()) or (vocab_size > 0 and bool((ids_t >= vocab_size).any().item())):
            raise ValueError(f"{name}[{idx}]['ids'] token IDs out of valid range [0, {vocab_size})")
        if seq_len is None:
            seq_len = int(ids_t.numel())
        elif int(ids_t.numel()) != seq_len:
            raise ValueError(f"All {name} rows must share sequence length {seq_len}")
        rows.append(ids_t)
        eids.append(str(raw_eid).strip())

    assert seq_len is not None
    if max(0, score_from - 1) >= seq_len - 1:
        raise ValueError(f"{name} sequence length {seq_len} too short for score_from={score_from}")
    ids_2d = torch.stack(rows, dim=0)
    is_gen = np.array([eid.startswith(general_prefix) for eid in eids], dtype=bool)
    if not bool(np.any(is_gen)) or not bool(np.any(~is_gen)):
        raise ValueError(f"{name} must include both general ({general_prefix}*) and specialist chunks")
    return ids_2d, is_gen, eids


@torch.no_grad()
def _forward_pre_ln(model: nn.Module, ids_2d: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Run embeddings and all transformer blocks up to pre-final-LayerNorm."""
    cos = model.cos_cached.to(ids_2d.device)
    sin = model.sin_cached.to(ids_2d.device)
    xb = torch.cat(
        [compute_embeddings_batch(model, ids_2d[b0 : b0 + batch_size]) for b0 in range(0, ids_2d.shape[0], batch_size)],
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
def _collect_probe_states_and_projectors(
    model: nn.Module,
    base_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    probe_ids: torch.Tensor,
    probe_is_gen: np.ndarray,
    cfg: CoAdaptationGuardedSurgeryConfig,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, dict[str, torch.Tensor]]]:
    """Collect per-layer general/specialist MLP inputs and AlphaEdit ``P_0`` projectors."""
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
    gen_mask = torch.as_tensor(probe_is_gen, dtype=torch.bool)
    prose_x_by_l: dict[int, torch.Tensor] = {}
    spec_x_by_l: dict[int, torch.Tensor] = {}
    p_null_by_l: dict[int, dict[str, torch.Tensor]] = {}

    for l_idx, blk in enumerate(model.blocks):
        mi_parts, nx_parts = [], []
        for b0 in range(0, xb.shape[0], cfg.batch_size):
            xm_b, mi_b = forward_block_pre_mlp(blk, xb[b0 : b0 + cfg.batch_size], cos, sin)
            mi_parts.append(mi_b)
            nx_parts.append(forward_block_post_mlp(blk, xm_b, mi_b))
        mlp_in = torch.cat(mi_parts, dim=0)
        xb = torch.cat(nx_parts, dim=0)

        scored = mlp_in[:, start_pos:-1, :]
        h_dim = int(scored.shape[-1])
        x_p = scored[gen_mask].reshape(-1, h_dim).contiguous()
        x_s = scored[~gen_mask].reshape(-1, h_dim).contiguous()
        prose_x_by_l[l_idx] = x_p
        spec_x_by_l[l_idx] = x_s

        if cfg.auto_project_p_null:
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
            p_null_by_l[l_idx] = layer_p
    return prose_x_by_l, spec_x_by_l, p_null_by_l


def _resolve_p_null_for_layer(
    p_null_source: Any,
    layer_idx: int,
    hidden_size: int,
    eig_threshold: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve optional ``(P_x, P_act)`` orthogonal projectors for ``layer_idx``."""
    if p_null_source is None:
        return None, None
    if isinstance(p_null_source, torch.Tensor):
        return _validate_square_projector(p_null_source, "p_null", expected_dim=hidden_size), None
    if not isinstance(p_null_source, Mapping):
        raise TypeError("p_null_by_expert must be a 2D Tensor, Mapping, or None")

    entry = None
    for cand in (layer_idx, str(layer_idx), f"blocks.{layer_idx}", "default"):
        if cand in p_null_source:
            entry = p_null_source[cand]
            break
    if entry is None:
        if "x" in p_null_source:
            entry = p_null_source
        else:
            return None, None
    if isinstance(entry, torch.Tensor):
        return _validate_square_projector(entry, f"p_null[{layer_idx}]", expected_dim=hidden_size), None
    if isinstance(entry, Mapping):
        px = (
            _validate_square_projector(entry["x"], f"p_null[{layer_idx}]['x']", expected_dim=hidden_size)
            if isinstance(entry.get("x"), torch.Tensor)
            else None
        )
        pact = (
            _validate_square_projector(entry["act"], f"p_null[{layer_idx}]['act']")
            if isinstance(entry.get("act"), torch.Tensor)
            else None
        )
        return px, pact
    return None, None


def _project_lora_pair_in_nullspace(
    b_base: torch.Tensor,
    a_base: torch.Tensor,
    b_cand: torch.Tensor,
    a_cand: torch.Tensor,
    p_mat: torch.Tensor,
    side: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project LoRA adapter delta ``B_cand A_cand - B_base A_base`` strictly into ``P_0``."""
    b1 = b_cand.detach().to("cpu", dtype=torch.float64)
    a1 = a_cand.detach().to("cpu", dtype=torch.float64)
    b0 = b_base.detach().to("cpu", dtype=torch.float64)
    a0 = a_base.detach().to("cpu", dtype=torch.float64)
    if int(torch.count_nonzero(b0).item()) == 0:
        if side == "input":
            b_new = b1.contiguous()
            a_new = (a1 @ p_mat).contiguous()
        else:
            b_new = (p_mat @ b1).contiguous()
            a_new = a1.contiguous()
    else:
        q_perp = torch.eye(int(p_mat.shape[0]), dtype=torch.float64) - p_mat
        if side == "input":
            b_new = b0.contiguous()
            a_proj = torch.linalg.lstsq(b0, b1 @ (a1 @ p_mat)).solution
            a_new = (a0 @ q_perp + a_proj @ p_mat).contiguous()
        else:
            a_new = a0.contiguous()
            b_proj_t = torch.linalg.lstsq(a0.T, (p_mat @ b1 @ a1).T).solution
            b_new = (q_perp @ b0 + p_mat @ b_proj_t.T).contiguous()
        decompose_lora_qr_svd(b_new, a_new)
    return (
        b_new.to(device=b_cand.device, dtype=b_cand.dtype),
        a_new.to(device=a_cand.device, dtype=a_cand.dtype),
    )


def _project_delta_into_nullspace(
    base_t: torch.Tensor,
    cand_t: torch.Tensor,
    p_mat: torch.Tensor,
    side: str,
) -> torch.Tensor:
    """Project parameter delta ``cand_t - base_t`` onto ``P_0`` on input or output side."""
    b64 = base_t.detach().to("cpu", dtype=torch.float64)
    c64 = cand_t.detach().to("cpu", dtype=torch.float64)
    delta = c64 - b64
    if delta.ndim == 1:
        proj = p_mat @ delta
    elif side == "input":
        proj = delta @ p_mat
    else:
        proj = p_mat @ delta
    return (b64 + proj).to(device=cand_t.device, dtype=cand_t.dtype).contiguous()


@torch.no_grad()
def _detect_sink_experts_loeo(
    model: nn.Module,
    unfrozen_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    base_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    probe_ids: torch.Tensor,
    probe_is_gen: np.ndarray,
    probe_eids: Sequence[str],
    protected_experts: set[int],
    cfg: CoAdaptationGuardedSurgeryConfig,
) -> tuple[list[int], dict[str, dict[str, float]]]:
    """Identify pure general-prose-harming sink experts via fast balanced LOEO ablation."""
    ln_mod = getattr(model, "ln", getattr(model, "norm", None))
    head_mod = getattr(model, "head", getattr(model, "lm_head", None))
    assert ln_mod is not None and head_mod is not None
    ln_eps = float(getattr(ln_mod, "eps", 1e-5))
    ln_b = getattr(ln_mod, "bias", None)

    # Select balanced subset (up to 4 general + 2 per specialist domain) for fast LOEO
    sub_idx: list[int] = []
    seen_dom: dict[str, int] = {}
    for i, eid in enumerate(probe_eids):
        cap = 4 if bool(probe_is_gen[i]) else 2
        if seen_dom.get(eid, 0) < cap:
            sub_idx.append(i)
            seen_dom[eid] = seen_dom.get(eid, 0) + 1
    sub_ids = probe_ids[sub_idx]
    sub_gen = probe_is_gen[sub_idx]

    def _eval_gen_spec(blk_map: Mapping[str, Mapping[str, torch.Tensor]]) -> tuple[float, float]:
        apply_checkpoint_blocks_inplace(model, blk_map)
        hp = _forward_pre_ln(model, sub_ids, cfg.batch_size)
        b_ten = ln_b if ln_b is not None else torch.zeros_like(ln_mod.weight)
        nlls = per_chunk_nll_from_pre_ln(
            hp, ln_mod.weight, b_ten, ln_eps, head_mod.weight, sub_ids, score_from=cfg.score_from
        ).numpy()
        return float(np.mean(nlls[sub_gen])), float(np.mean(nlls[~sub_gen]))

    ref_gen, ref_spec = _eval_gen_spec(unfrozen_blocks)
    expert_indices = sorted(
        {
            int(bid.rsplit(".experts.", 1)[1])
            for bid in unfrozen_blocks
            if ".experts." in bid
        }
    )
    sinks: list[int] = []
    report: dict[str, dict[str, float]] = {}
    for e_idx in expert_indices:
        if e_idx in protected_experts:
            continue
        suffix = f".experts.{e_idx}"
        has_nonzero_delta = False
        for bid, sub in unfrozen_blocks.items():
            if not bid.endswith(suffix):
                continue
            for sk, t in sub.items():
                if sk.endswith("_B"):
                    if int(torch.count_nonzero(t).item()) > 0:
                        has_nonzero_delta = True
                        break
                elif (sk in _SWIGLU_PROJECTIONS or sk == "delta") and not torch.equal(t, base_blocks[bid][sk]):
                    has_nonzero_delta = True
                    break
            if has_nonzero_delta:
                break
        if not has_nonzero_delta:
            report[f"expert_{e_idx}"] = {"prose_harm_nats": 0.0, "specialist_value_nats": 0.0}
            continue

        mod_blocks = {bid: {sk: t.detach().clone() for sk, t in sub.items()} for bid, sub in unfrozen_blocks.items()}
        for bid, sub in mod_blocks.items():
            if bid.endswith(suffix):
                for sk, t in sub.items():
                    if sk.endswith("_B"):
                        t.zero_()
                    elif sk in _SWIGLU_PROJECTIONS or sk == "delta":
                        t.copy_(base_blocks[bid][sk])
        wo_gen, wo_spec = _eval_gen_spec(mod_blocks)
        prose_harm = ref_gen - wo_gen
        spec_val = wo_spec - ref_spec
        report[f"expert_{e_idx}"] = {"prose_harm_nats": prose_harm, "specialist_value_nats": spec_val}
        if prose_harm >= cfg.sink_prose_harm_floor_nats and spec_val < cfg.sink_spec_preserve_floor_nats:
            sinks.append(e_idx)
    return sinks, report


def _synthesize_candidate_blocks(
    *,
    base_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    frozen_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    unfrozen_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    strategy: str,
    sink_set: set[int],
    protected_experts: set[int],
    anchor_layers: set[int],
    p_null_source: Any,
    prose_x_by_layer: Mapping[int, torch.Tensor],
    spec_x_by_layer: Mapping[int, torch.Tensor],
    work_model: nn.Module | None,
    probe_ids: torch.Tensor | None,
    probe_is_gen: np.ndarray | None,
    base_probe_gen_nll: float | None,
    base_probe_total_nll: float | None,
    cfg: CoAdaptationGuardedSurgeryConfig,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """Synthesize either ``cross_arm_graft`` or ``coadapted_rescue`` candidate blocks."""
    cand: dict[str, dict[str, torch.Tensor]] = {
        bid: {sk: t.detach().clone() for sk, t in sub.items()}
        for bid, sub in unfrozen_blocks.items()
    }
    oof_gated_layers: list[int] = []

    for bid in sorted(cand.keys()):
        if bid == "final_ln":
            continue
        l_idx = _parse_layer_from_block_id(bid) or 0
        is_anchor = l_idx in anchor_layers

        if ".experts." in bid:
            e_idx = int(bid.rsplit(".experts.", 1)[1])
            if is_anchor or e_idx in protected_experts:
                continue
            if e_idx in sink_set:
                for sk, t in cand[bid].items():
                    if sk.endswith("_B"):
                        t.zero_()
                    else:
                        t.copy_(base_blocks[bid][sk])
                continue
            w_sub = cand[bid]
            h_dim = int(w_sub["w1_A"].shape[1]) if "w1_A" in w_sub else int(next(iter(w_sub.values())).shape[-1])
            px, pact = _resolve_p_null_for_layer(p_null_source, l_idx, h_dim, cfg.nullspace_eig_threshold)
            if px is None and pact is None:
                continue
            if all(f"{p}_{s}" in w_sub for p in _SWIGLU_PROJECTIONS for s in ("A", "B")):
                for proj in _SWIGLU_PROJECTIONS:
                    ak, bk = f"{proj}_A", f"{proj}_B"
                    if proj in ("w1", "w2"):
                        if px is not None and int(w_sub[ak].shape[1]) == int(px.shape[0]):
                            bn, an = _project_lora_pair_in_nullspace(
                                base_blocks[bid][bk], base_blocks[bid][ak], w_sub[bk], w_sub[ak], px, "input"
                            )
                            w_sub[bk], w_sub[ak] = bn, an
                    else:
                        p_w3 = pact if (pact is not None and int(w_sub[ak].shape[1]) == int(pact.shape[0])) else px
                        side_w3 = "input" if (pact is not None and p_w3 is pact) else "output"
                        if p_w3 is not None:
                            bn, an = _project_lora_pair_in_nullspace(
                                base_blocks[bid][bk], base_blocks[bid][ak], w_sub[bk], w_sub[ak], p_w3, side_w3
                            )
                            w_sub[bk], w_sub[ak] = bn, an
            else:
                for sk, t_val in list(w_sub.items()):
                    if sk in ("w1", "w2", "delta") and px is not None and t_val.ndim == 2 and int(t_val.shape[1]) == int(px.shape[0]):
                        w_sub[sk] = _project_delta_into_nullspace(base_blocks[bid][sk], t_val, px, "input")
                    elif sk == "w3" and px is not None and int(t_val.shape[0]) == int(px.shape[0]):
                        w_sub[sk] = _project_delta_into_nullspace(base_blocks[bid][sk], t_val, px, "output")
        elif bid.endswith(".router"):
            w_t = cand[bid]["weight"]
            h_dim = int(w_t.shape[1])
            px, _ = _resolve_p_null_for_layer(p_null_source, l_idx, h_dim, cfg.nullspace_eig_threshold)
            if strategy == "cross_arm_graft":
                for sk in cand[bid]:
                    f64 = frozen_blocks[bid][sk].detach().to("cpu", dtype=torch.float64)
                    u64 = unfrozen_blocks[bid][sk].detach().to("cpu", dtype=torch.float64)
                    blended = f64 + cfg.cross_arm_gate_blend_alpha * (u64 - f64)
                    if px is not None and blended.ndim == 2 and int(blended.shape[1]) == int(px.shape[0]):
                        blended = f64 + (blended - f64) @ px
                    cand[bid][sk] = blended.to(device=cand[bid][sk].device, dtype=cand[bid][sk].dtype)

            # Selective OOF contrastive prose suppression on surviving router rows
            if (
                not is_anchor
                and cfg.oof_router_beta > 0.0
                and l_idx in prose_x_by_layer
                and l_idx in spec_x_by_layer
            ):
                p_all = prose_x_by_layer[l_idx]
                s_all = spec_x_by_layer[l_idx]
                if int(p_all.shape[0]) >= 2 and int(s_all.shape[0]) >= 2:
                    p0, p1 = p_all[0::2], p_all[1::2]
                    s0, s1 = s_all[0::2], s_all[1::2]
                    u_vec, p_stats = fit_uncentered_prose_probe_with_oof(p0, p1, s0, s1)
                    rel_w = float(p_stats["rel_weight"])
                    if float(p_stats["oof_snr"]) >= cfg.min_oof_snr and rel_w > 0.0 and float(u_vec.norm().item()) > 0.0:
                        oof_gated_layers.append(l_idx)
                        rw = cand[bid]["weight"]
                        u_cast = u_vec.to(device=rw.device, dtype=rw.dtype)
                        n_exp = int(rw.shape[0])
                        for e_ix in range(n_exp):
                            if e_ix not in sink_set and e_ix not in protected_experts:
                                rw[e_ix].sub_(float(cfg.oof_router_beta * rel_w) * u_cast)
                        if sink_set:
                            primary_sink = 0 if 0 in sink_set else min(sink_set)
                            if primary_sink < n_exp:
                                rw[primary_sink].add_(float(0.5 * cfg.oof_router_beta * rel_w) * u_cast)

    ln_cal_meta: dict[str, Any] = {"executed": False, "lambda_ln": 1.0, "tau_temp": 1.0}
    if "final_ln" in cand:
        ln_base_src = frozen_blocks["final_ln"] if strategy == "cross_arm_graft" else base_blocks["final_ln"]
        if (
            cfg.fold_ln_lambda is None
            and work_model is not None
            and probe_ids is not None
            and probe_is_gen is not None
        ):
            ln_mod = getattr(work_model, "ln", getattr(work_model, "norm", None))
            head_mod = getattr(work_model, "head", getattr(work_model, "lm_head", None))
            assert ln_mod is not None and head_mod is not None
            apply_checkpoint_blocks_inplace(work_model, cand)
            hp_cand = _forward_pre_ln(work_model, probe_ids, cfg.batch_size)
            start_pos = max(0, cfg.score_from - 1)
            cal_res = fast_affine_layernorm_temperature_grid(
                hp_cand[:, start_pos:-1, :],
                ln_base_src["weight"],
                ln_base_src.get("bias"),
                unfrozen_blocks["final_ln"]["weight"],
                unfrozen_blocks["final_ln"].get("bias"),
                head_mod.weight,
                probe_ids[:, start_pos + 1 :],
                probe_is_gen,
                eps=float(getattr(ln_mod, "eps", 1e-5)),
                lambda_grid=cfg.lambda_grid,
                tau_grid=cfg.tau_grid,
                max_general_drift_rel=cfg.calibrator_max_general_drift_rel,
                seq_batch_size=cfg.batch_size,
                base_general_nll_override=base_probe_gen_nll,
                base_total_nll_override=base_probe_total_nll,
            )
            cand["final_ln"]["weight"] = cal_res["folded_ln_weight"]
            if cal_res["folded_ln_bias"] is not None:
                cand["final_ln"]["bias"] = cal_res["folded_ln_bias"]
            ln_cal_meta = {
                "executed": True,
                "lambda_ln": float(cal_res["best_lambda_ln"]),
                "tau_temp": float(cal_res["best_tau_temp"]),
            }
        else:
            lam = float(cfg.fold_ln_lambda) if cfg.fold_ln_lambda is not None else 0.5
            tau = float(cfg.fold_tau_temp) if cfg.fold_tau_temp is not None else 1.0
            w0 = ln_base_src["weight"]
            w1 = unfrozen_blocks["final_ln"]["weight"]
            w_int = (w0.float() + lam * (w1.float() - w0.float())).to(w0.dtype)
            b0 = ln_base_src.get("bias")
            b1 = unfrozen_blocks["final_ln"].get("bias")
            b_int = (b0.float() + lam * (b1.float() - b0.float())).to(b0.dtype) if (b0 is not None and b1 is not None) else None
            fw, fb = fold_temperature_into_norm_params(w_int, b_int, tau)
            cand["final_ln"]["weight"] = fw
            if fb is not None:
                cand["final_ln"]["bias"] = fb
            ln_cal_meta = {"executed": True, "lambda_ln": lam, "tau_temp": tau}

    return cand, {
        "strategy": strategy,
        "sink_experts": sorted(sink_set),
        "anchor_layers_protected": sorted(anchor_layers),
        "oof_gated_layers": oof_gated_layers,
        "ln_calibration": ln_cal_meta,
    }


@torch.no_grad()
def _evaluate_arm_state(
    *,
    model: nn.Module | None,
    blocks: Mapping[str, Mapping[str, torch.Tensor]],
    eval_fn: Callable[[Any], Mapping[str, Any]] | None,
    eval_ids: torch.Tensor | None,
    eval_eids: Sequence[str],
    cfg: CoAdaptationGuardedSurgeryConfig,
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
        ln_mod = getattr(model, "ln", getattr(model, "norm", None))
        head_mod = getattr(model, "head", getattr(model, "lm_head", None))
        assert ln_mod is not None and head_mod is not None
        ln_b = getattr(ln_mod, "bias", None)
        if ln_b is None:
            ln_b = torch.zeros_like(ln_mod.weight)
        nlls = per_chunk_nll_from_pre_ln(
            hp, ln_mod.weight, ln_b, float(getattr(ln_mod, "eps", 1e-5)), head_mod.weight, eval_ids, score_from=cfg.score_from
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
        raise ValueError(f"Evaluation output must include both general ({cfg.general_prefix}*) and specialist domains")
    return {
        "all_finite": True,
        "domain_nlls": domain_nlls,
        "general_nll": float(np.mean(gen_vals)),
        "spec_nll": float(np.mean(spec_vals)),
        "nll_mean": float(np.mean(list(domain_nlls.values()))),
    }


def _score_against_base(
    arm_eval: Mapping[str, Any],
    base_eval: Mapping[str, Any],
    cfg: CoAdaptationGuardedSurgeryConfig,
) -> dict[str, Any]:
    """Compute general/worst drift ratios, specialist gain, and ANSE Energy relative to ``base_eval``."""
    if not arm_eval["all_finite"] or set(arm_eval["domain_nlls"].keys()) != set(base_eval["domain_nlls"].keys()):
        return {
            "all_finite": False,
            "general_drift_ratio": float("inf"),
            "worst_domain_ratio": float("inf"),
            "guard_ok": False,
            "specialist_gain_nats": 0.0,
            "mean_diff_vs_base": float("inf"),
            "anse_energy": float(cfg.maximum_pain_energy),
        }

    base_doms = base_eval["domain_nlls"]
    cand_doms = arm_eval["domain_nlls"]
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
        "guard_ok": guard_ok,
        "specialist_gain_nats": spec_gain,
        "mean_diff_vs_base": mean_diff,
        "anse_energy": float(anse_e),
    }


def _collect_router_logits_from_blocks(
    frozen_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    coadapted_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    prose_x_by_layer: Mapping[int, torch.Tensor],
    spec_x_by_layer: Mapping[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute host (frozen-arm) vs co-adapted router logits across MoE router layers."""
    host_parts: list[torch.Tensor] = []
    coad_parts: list[torch.Tensor] = []
    router_bids = sorted(bid for bid in frozen_blocks if bid.endswith(".router"))
    for bid in router_bids:
        l_idx = _parse_layer_from_block_id(bid) or 0
        w_frz = frozen_blocks[bid]["weight"].detach().to("cpu", dtype=torch.float64)
        w_coad = coadapted_blocks[bid]["weight"].detach().to("cpu", dtype=torch.float64)
        h_dim = int(w_frz.shape[1])
        xs: list[torch.Tensor] = []
        if l_idx in prose_x_by_layer:
            xs.append(prose_x_by_layer[l_idx].detach().to("cpu", dtype=torch.float64))
        if l_idx in spec_x_by_layer:
            xs.append(spec_x_by_layer[l_idx].detach().to("cpu", dtype=torch.float64))
        x_cat = torch.cat(xs, dim=0) if xs else torch.eye(h_dim, dtype=torch.float64)
        log_frz = x_cat @ w_frz.T
        log_coad = x_cat @ w_coad.T
        if "bias" in frozen_blocks[bid]:
            log_frz = log_frz + frozen_blocks[bid]["bias"].detach().to("cpu", dtype=torch.float64)
        if "bias" in coadapted_blocks[bid]:
            log_coad = log_coad + coadapted_blocks[bid]["bias"].detach().to("cpu", dtype=torch.float64)
        host_parts.append(log_frz)
        coad_parts.append(log_coad)
    return torch.cat(host_parts, dim=0), torch.cat(coad_parts, dim=0)


def _resolve_surgery_config(
    config: CoAdaptationGuardedSurgeryConfig | None,
    inplace: bool | None,
) -> CoAdaptationGuardedSurgeryConfig:
    """Resolve optional ``CoAdaptationGuardedSurgeryConfig`` with ``inplace`` override."""
    if config is not None and not isinstance(config, CoAdaptationGuardedSurgeryConfig):
        raise TypeError(
            f"config must be CoAdaptationGuardedSurgeryConfig or None, got {type(config).__name__}"
        )
    cfg_dict = asdict(config) if config is not None else asdict(CoAdaptationGuardedSurgeryConfig())
    if inplace is not None:
        cfg_dict["inplace"] = _validate_bool(inplace, "inplace")
    return CoAdaptationGuardedSurgeryConfig(**cfg_dict)


@torch.no_grad()
def execute_coadaptation_guarded_surgery(
    base_checkpoint: nn.Module | Mapping[str, Any],
    frozen_router_checkpoint: nn.Module | Mapping[str, Any],
    unfrozen_arm_checkpoint: nn.Module | Mapping[str, Any],
    *,
    target_model: nn.Module | None = None,
    probe_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_fn: Callable[[Any], Mapping[str, Any]] | None = None,
    p_null_by_expert: torch.Tensor | Mapping[Any, Any] | None = None,
    prose_hidden_states: torch.Tensor | Mapping[Any, torch.Tensor] | Sequence[torch.Tensor] | None = None,
    spec_hidden_states: torch.Tensor | Mapping[Any, torch.Tensor] | Sequence[torch.Tensor] | None = None,
    sink_experts: Sequence[int] | set[int] | None = None,
    protected_experts: Sequence[int] | set[int] = (),
    anchor_layers: Sequence[int] | set[int] | None = None,
    router_arm_logits: torch.Tensor | Mapping[str, Any] | Sequence[torch.Tensor] | None = None,
    expert_arm_outputs: torch.Tensor | Mapping[str, Any] | None = None,
    expert_arm_router_logits: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    config: CoAdaptationGuardedSurgeryConfig | None = None,
    inplace: bool | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run prefix-cached attribution, co-adaptation mismatch gating, and ANSE-ratified MoE surgery.

    Evaluates both a cross-arm router graft (frozen router + unfrozen specialists) and a
    co-adapted selective-sink + OOF multi-gate rescue of the unfrozen arm. Automatically vetoes
    the cross-arm graft whenever ``detect_router_expert_coadaptation_mismatch`` flags excessive
    routing KL or NLL co-adaptation loss, preferring the co-adapted rescue under ``ANSE Delta E < 0``
    and ``<= +1.0%`` general-prose / worst-domain guards, with atomic bitwise rollback otherwise.
    """
    cfg = _resolve_surgery_config(config, inplace)
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError(f"eval_fn must be callable or None, got {type(eval_fn).__name__}")
    if target_model is not None and not isinstance(target_model, nn.Module):
        raise TypeError(f"target_model must be an nn.Module or None, got {type(target_model).__name__}")

    prot_exp_set = _validate_expert_index_set(protected_experts, "protected_experts")
    explicit_anchors = _validate_expert_index_set(anchor_layers, "anchor_layers") if anchor_layers is not None else None
    explicit_sinks = _validate_expert_index_set(sink_experts, "sink_experts") if sink_experts is not None else None

    base_blocks = _normalize_checkpoint_to_blocks(base_checkpoint, "base_checkpoint")
    frozen_blocks = _normalize_checkpoint_to_blocks(frozen_router_checkpoint, "frozen_router_checkpoint")
    unfrozen_blocks = _normalize_checkpoint_to_blocks(unfrozen_arm_checkpoint, "unfrozen_arm_checkpoint")
    _validate_three_arm_layouts(base_blocks, frozen_blocks, unfrozen_blocks)

    work_model: nn.Module | None = None
    if target_model is not None:
        work_model = target_model if cfg.inplace else copy.deepcopy(target_model)
    elif isinstance(unfrozen_arm_checkpoint, nn.Module):
        work_model = unfrozen_arm_checkpoint if cfg.inplace else copy.deepcopy(unfrozen_arm_checkpoint)
    elif isinstance(base_checkpoint, nn.Module):
        work_model = base_checkpoint if cfg.inplace else copy.deepcopy(base_checkpoint)
    elif isinstance(frozen_router_checkpoint, nn.Module):
        work_model = frozen_router_checkpoint if cfg.inplace else copy.deepcopy(frozen_router_checkpoint)

    if work_model is None and eval_fn is None:
        raise ValueError("Either an nn.Module checkpoint/target_model or eval_fn must be provided")

    head_mod = getattr(work_model, "head", getattr(work_model, "lm_head", None)) if work_model is not None else None
    vocab_size = int(head_mod.weight.shape[0]) if head_mod is not None else 0

    probe_ids: torch.Tensor | None = None
    probe_is_gen: np.ndarray | None = None
    probe_eids: list[str] = []
    if probe_chunks is not None:
        probe_ids, probe_is_gen, probe_eids = _validate_chunks(
            probe_chunks, vocab_size=vocab_size, general_prefix=cfg.general_prefix, score_from=cfg.score_from, name="probe_chunks"
        )

    eval_ids: torch.Tensor | None = None
    eval_eids: list[str] = []
    if eval_chunks is not None:
        eval_ids, _, eval_eids = _validate_chunks(
            eval_chunks, vocab_size=vocab_size, general_prefix=cfg.general_prefix, score_from=cfg.score_from, name="eval_chunks"
        )
    elif probe_ids is not None:
        eval_ids, eval_eids = probe_ids, probe_eids
    elif eval_fn is None:
        raise ValueError("Either probe_chunks/eval_chunks or eval_fn must be provided")

    base_sha = compute_checkpoint_sha256(base_blocks)
    frozen_sha = compute_checkpoint_sha256(frozen_blocks)
    unfrozen_sha = compute_checkpoint_sha256(unfrozen_blocks)

    was_training = work_model.training if work_model is not None else False
    if work_model is not None:
        work_model.eval()

    try:
        # 1. Prefix-cached causal layerwise attribution & P_0 / OOF hidden state collection
        prose_x_by_l: dict[int, torch.Tensor] = {}
        spec_x_by_l: dict[int, torch.Tensor] = {}
        auto_p_null: dict[int, dict[str, torch.Tensor]] = {}
        prefix_attr_summary: dict[str, Any] = {
            "executed": False,
            "block_evals_saved_ratio": 0.0,
            "speedup_factor": 1.0,
            "anchor_layers": [],
        }
        resolved_anchors: set[int] = set(explicit_anchors or ())

        if work_model is not None and probe_ids is not None and probe_is_gen is not None:
            prose_x_by_l, spec_x_by_l, auto_p_null = _collect_probe_states_and_projectors(
                work_model, base_blocks, probe_ids, probe_is_gen, cfg
            )
            apply_checkpoint_blocks_inplace(work_model, unfrozen_blocks)
            gen_ids = probe_ids[torch.as_tensor(probe_is_gen, dtype=torch.bool)]
            gen_emb = torch.cat(
                [
                    compute_embeddings_batch(work_model, gen_ids[b0 : b0 + cfg.batch_size])
                    for b0 in range(0, gen_ids.shape[0], cfg.batch_size)
                ],
                dim=0,
            )
            ln_mod = getattr(work_model, "ln", getattr(work_model, "norm", None))
            assert ln_mod is not None and head_mod is not None
            ln_eps = float(getattr(ln_mod, "eps", 1e-5))
            ln_b = getattr(ln_mod, "bias", None)
            b_ten = ln_b if ln_b is not None else torch.zeros_like(ln_mod.weight)
            cos = work_model.cos_cached.to(gen_ids.device)
            sin = work_model.sin_cached.to(gen_ids.device)
            attr_res = compute_prefix_cached_layerwise_attribution(
                work_model.blocks,
                gen_emb,
                lambda h_f: per_chunk_nll_from_pre_ln(
                    h_f, ln_mod.weight, b_ten, ln_eps, head_mod.weight, gen_ids, score_from=cfg.score_from
                ),
                batch_size=cfg.batch_size,
                cos=cos,
                sin=sin,
                anchor_harm_threshold_nats=cfg.anchor_harm_threshold_nats,
            )
            if explicit_anchors is None:
                resolved_anchors = set(attr_res["anchor_layers"])
            prefix_attr_summary = {
                "executed": True,
                "num_layers": attr_res["num_layers"],
                "total_block_calls": attr_res["total_block_calls"],
                "naive_total_block_calls": attr_res["naive_total_block_calls"],
                "block_evals_saved_ratio": attr_res["block_evals_saved_ratio"],
                "speedup_factor": attr_res["speedup_factor"],
                "anchor_layers": sorted(resolved_anchors),
                "ranked_layers_by_harm": attr_res["ranked_layers_by_harm"],
            }
        else:
            if isinstance(prose_hidden_states, torch.Tensor):
                prose_x_by_l = {0: _validate_fp_tensor(prose_hidden_states, "prose_hidden_states", min_ndim=2).reshape(-1, prose_hidden_states.shape[-1])}
            elif isinstance(prose_hidden_states, Mapping):
                prose_x_by_l = {int(_parse_layer_from_block_id(str(k)) or 0): _validate_fp_tensor(v, "prose_hidden_states", min_ndim=2).reshape(-1, v.shape[-1]) for k, v in prose_hidden_states.items()}
            if isinstance(spec_hidden_states, torch.Tensor):
                spec_x_by_l = {0: _validate_fp_tensor(spec_hidden_states, "spec_hidden_states", min_ndim=2).reshape(-1, spec_hidden_states.shape[-1])}
            elif isinstance(spec_hidden_states, Mapping):
                spec_x_by_l = {int(_parse_layer_from_block_id(str(k)) or 0): _validate_fp_tensor(v, "spec_hidden_states", min_ndim=2).reshape(-1, v.shape[-1]) for k, v in spec_hidden_states.items()}
            if cfg.auto_project_p_null and p_null_by_expert is None and prose_x_by_l:
                auto_p_null = {
                    l_idx: {
                        "x": compute_prose_nullspace_projector_scaled(
                            x_p, sv_rel_tol=cfg.nullspace_sv_rel_tol, variance_threshold=cfg.nullspace_variance_threshold
                        )[0]
                    }
                    for l_idx, x_p in prose_x_by_l.items()
                }

        # 2. Detect pure prose-harming sink experts
        loeo_report: dict[str, dict[str, float]] = {}
        if explicit_sinks is not None:
            resolved_sinks = explicit_sinks - prot_exp_set
        elif cfg.auto_detect_sinks and work_model is not None and probe_ids is not None and probe_is_gen is not None:
            detected, loeo_report = _detect_sink_experts_loeo(
                work_model, unfrozen_blocks, base_blocks, probe_ids, probe_is_gen, probe_eids, prot_exp_set, cfg
            )
            resolved_sinks = set(detected)
        else:
            resolved_sinks = set()

        resolved_p_null = (
            p_null_by_expert if p_null_by_expert is not None else (auto_p_null if auto_p_null else None)
        )

        # Compute base probe NLL references once for LN calibration & ANSE evaluation
        base_probe_eval: dict[str, Any] | None = None
        if eval_fn is None and work_model is not None and probe_ids is not None:
            base_probe_eval = _evaluate_arm_state(
                model=work_model, blocks=base_blocks, eval_fn=None, eval_ids=probe_ids, eval_eids=probe_eids, cfg=cfg
            )

        # 3. Synthesize BOTH candidate strategies: cross-arm router graft vs co-adapted rescue
        cross_blocks, cross_meta = _synthesize_candidate_blocks(
            base_blocks=base_blocks,
            frozen_blocks=frozen_blocks,
            unfrozen_blocks=unfrozen_blocks,
            strategy="cross_arm_graft",
            sink_set=resolved_sinks,
            protected_experts=prot_exp_set,
            anchor_layers=resolved_anchors,
            p_null_source=resolved_p_null,
            prose_x_by_layer=prose_x_by_l,
            spec_x_by_layer=spec_x_by_l,
            work_model=work_model,
            probe_ids=probe_ids,
            probe_is_gen=probe_is_gen,
            base_probe_gen_nll=base_probe_eval["general_nll"] if base_probe_eval is not None else None,
            base_probe_total_nll=base_probe_eval["nll_mean"] if base_probe_eval is not None else None,
            cfg=cfg,
        )
        rescue_blocks, rescue_meta = _synthesize_candidate_blocks(
            base_blocks=base_blocks,
            frozen_blocks=frozen_blocks,
            unfrozen_blocks=unfrozen_blocks,
            strategy="coadapted_rescue",
            sink_set=resolved_sinks,
            protected_experts=prot_exp_set,
            anchor_layers=resolved_anchors,
            p_null_source=resolved_p_null,
            prose_x_by_layer=prose_x_by_l,
            spec_x_by_layer=spec_x_by_l,
            work_model=work_model,
            probe_ids=probe_ids,
            probe_is_gen=probe_is_gen,
            base_probe_gen_nll=base_probe_eval["general_nll"] if base_probe_eval is not None else None,
            base_probe_total_nll=base_probe_eval["nll_mean"] if base_probe_eval is not None else None,
            cfg=cfg,
        )

        # 4. Evaluate base, donors, cross-arm graft, and co-adapted rescue under ANSE
        base_eval = _evaluate_arm_state(
            model=work_model, blocks=base_blocks, eval_fn=eval_fn, eval_ids=eval_ids, eval_eids=eval_eids, cfg=cfg
        )
        if not base_eval["all_finite"]:
            raise ValueError("base_checkpoint produced non-finite evaluation NLL")

        frozen_eval = _evaluate_arm_state(
            model=work_model, blocks=frozen_blocks, eval_fn=eval_fn, eval_ids=eval_ids, eval_eids=eval_eids, cfg=cfg
        )
        frozen_metrics = _score_against_base(frozen_eval, base_eval, cfg)

        unfrozen_eval = _evaluate_arm_state(
            model=work_model, blocks=unfrozen_blocks, eval_fn=eval_fn, eval_ids=eval_ids, eval_eids=eval_eids, cfg=cfg
        )
        unfrozen_metrics = _score_against_base(unfrozen_eval, base_eval, cfg)

        cross_eval = _evaluate_arm_state(
            model=work_model, blocks=cross_blocks, eval_fn=eval_fn, eval_ids=eval_ids, eval_eids=eval_eids, cfg=cfg
        )
        cross_metrics = _score_against_base(cross_eval, base_eval, cfg)

        rescue_eval = _evaluate_arm_state(
            model=work_model, blocks=rescue_blocks, eval_fn=eval_fn, eval_ids=eval_ids, eval_eids=eval_eids, cfg=cfg
        )
        rescue_metrics = _score_against_base(rescue_eval, base_eval, cfg)

        # 5. Run Co-Adaptation Mismatch Detector comparing cross-arm graft vs co-adapted rescue
        eff_r_logits: Any = router_arm_logits
        eff_exp_out: Any = expert_arm_outputs
        eff_exp_r_logits: Any = expert_arm_router_logits
        if eff_r_logits is None or eff_exp_out is None:
            host_log, coad_log = _collect_router_logits_from_blocks(
                cross_blocks, rescue_blocks, prose_x_by_l, spec_x_by_l
            )
            if eff_r_logits is None:
                eff_r_logits = host_log
            if eff_exp_out is None:
                eff_exp_out = coad_log
            elif eff_exp_r_logits is None:
                eff_exp_r_logits = coad_log

        coad_nll_val = float(rescue_eval["nll_mean"]) if rescue_eval["all_finite"] else float(base_eval["nll_mean"])
        graft_nll_val = float(cross_eval["nll_mean"]) if cross_eval["all_finite"] else float(base_eval["nll_mean"]) + 1.0
        mismatch_report = detect_router_expert_coadaptation_mismatch(
            eff_r_logits,
            eff_exp_out,
            coadapted_nll=coad_nll_val,
            grafted_nll=graft_nll_val,
            max_mismatch_nats=cfg.max_mismatch_nats,
            max_routing_kl=cfg.max_routing_kl,
            routing_mode=cfg.routing_mode,
            top_k=cfg.top_k,
            expert_arm_router_logits=eff_exp_r_logits,
        )
        cross_vetoed = bool(mismatch_report["mismatch_detected"] and cfg.veto_mismatched_cross_arm_graft)
        veto_reason = "router_expert_coadaptation_mismatch" if cross_vetoed else None

        # 6. ANSE Monotone Selection & Atomic Bitwise Rollback
        pool: list[tuple[str, dict[str, dict[str, torch.Tensor]], dict[str, Any], dict[str, Any]]] = [
            ("coadapted_rescue", rescue_blocks, rescue_eval, rescue_metrics)
        ]
        if not cross_vetoed:
            pool.insert(0, ("cross_arm_graft", cross_blocks, cross_eval, cross_metrics))

        feasible = [
            item
            for item in pool
            if item[2]["all_finite"]
            and item[3]["guard_ok"]
            and float(item[3]["anse_energy"]) < -cfg.min_energy_improvement
        ]

        if feasible:
            feasible.sort(key=lambda item: (float(item[3]["anse_energy"]), float(item[2]["nll_mean"]), 0 if item[0] == "cross_arm_graft" else 1))
            selected_strategy, final_blocks, final_eval, final_metrics = feasible[0]
            accepted = True
            rolled_back = False
            rollback_verified = False
            actual_rb_target = "none"
            status = (
                "COADAPTED_RESCUE_COMMITTED_CROSS_ARM_VETOED"
                if (selected_strategy == "coadapted_rescue" and cross_vetoed)
                else f"{selected_strategy.upper()}_COMMITTED"
            )
            expected_sha = compute_checkpoint_sha256(final_blocks)
        else:
            selected_strategy = "rollback"
            accepted = False
            rolled_back = True
            status = "ROLLED_BACK_ANSE_MAXIMUM_PAIN"
            actual_rb_target = cfg.rollback_target
            if cfg.rollback_target == "frozen_arm":
                final_blocks, final_eval, final_metrics, expected_sha = (
                    frozen_blocks, frozen_eval, frozen_metrics, frozen_sha
                )
            elif cfg.rollback_target == "unfrozen_arm":
                final_blocks, final_eval, final_metrics, expected_sha = (
                    unfrozen_blocks, unfrozen_eval, unfrozen_metrics, unfrozen_sha
                )
            else:
                final_blocks = base_blocks
                final_eval = base_eval
                final_metrics = _score_against_base(base_eval, base_eval, cfg)
                expected_sha = base_sha

        if work_model is not None:
            apply_checkpoint_blocks_inplace(work_model, final_blocks)
            extracted = extract_checkpoint_blocks(work_model)
            scoped = {bid: extracted[bid] for bid in final_blocks if bid in extracted}
            final_sha = compute_checkpoint_sha256(scoped)
        else:
            final_sha = compute_checkpoint_sha256(final_blocks)

        if rolled_back:
            rollback_verified = final_sha == expected_sha
            if not rollback_verified:
                raise RuntimeError(
                    f"Atomic bitwise rollback SHA-256 mismatch: expected {expected_sha[:12]}, got {final_sha[:12]}"
                )

        def _clean_ratio(val: float) -> float | None:
            return float(val) if math.isfinite(float(val)) else None

        receipt: dict[str, Any] = {
            "object": COADAPTATION_RECEIPT_OBJECT,
            "schema": COADAPTATION_SCHEMA,
            "status": status,
            "selected_strategy": selected_strategy,
            "accepted": accepted,
            "rolled_back": rolled_back,
            "rollback_target": actual_rb_target,
            "rollback_bitwise_verified": rollback_verified,
            "cross_arm_vetoed": cross_vetoed,
            "veto_reason": veto_reason,
            "nll_mean": float(final_eval["nll_mean"]) if final_eval["nll_mean"] is not None else None,
            "coadaptation_mismatch_nats": float(mismatch_report["coadaptation_mismatch_nats"]),
            "routing_kl_divergence": float(mismatch_report["routing_kl_divergence"]),
            "block_evals_saved_ratio": float(prefix_attr_summary["block_evals_saved_ratio"]),
            "anse_energy": float(final_metrics["anse_energy"]),
            "delta_e_vs_base": float(final_metrics["anse_energy"]),
            "base_nll_mean": float(base_eval["nll_mean"]),
            "final_general_drift_ratio": _clean_ratio(final_metrics["general_drift_ratio"]),
            "prefix_attribution_summary": prefix_attr_summary,
            "coadaptation_report": mismatch_report,
            "loeo_sink_attribution": loeo_report,
            "candidate_summaries": {
                "cross_arm_graft": {
                    "nll_mean": cross_eval["nll_mean"],
                    "anse_energy": float(cross_metrics["anse_energy"]),
                    "guard_ok": bool(cross_metrics["guard_ok"]),
                    "general_drift_ratio": _clean_ratio(cross_metrics["general_drift_ratio"]),
                    "meta": cross_meta,
                },
                "coadapted_rescue": {
                    "nll_mean": rescue_eval["nll_mean"],
                    "anse_energy": float(rescue_metrics["anse_energy"]),
                    "guard_ok": bool(rescue_metrics["guard_ok"]),
                    "general_drift_ratio": _clean_ratio(rescue_metrics["general_drift_ratio"]),
                    "meta": rescue_meta,
                },
            },
            "donor_metrics": {
                "frozen_router_arm": {
                    "nll_mean": frozen_eval["nll_mean"],
                    "anse_energy": float(frozen_metrics["anse_energy"]),
                    "guard_ok": bool(frozen_metrics["guard_ok"]),
                    "general_drift_ratio": _clean_ratio(frozen_metrics["general_drift_ratio"]),
                },
                "unfrozen_arm": {
                    "nll_mean": unfrozen_eval["nll_mean"],
                    "anse_energy": float(unfrozen_metrics["anse_energy"]),
                    "guard_ok": bool(unfrozen_metrics["guard_ok"]),
                    "general_drift_ratio": _clean_ratio(unfrozen_metrics["general_drift_ratio"]),
                },
            },
            "base_state_sha256": base_sha,
            "frozen_router_state_sha256": frozen_sha,
            "unfrozen_arm_state_sha256": unfrozen_sha,
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
    "COADAPTATION_RECEIPT_OBJECT",
    "COADAPTATION_SCHEMA",
    "CoAdaptationGuardedSurgeryConfig",
    "DEFAULT_ANCHOR_HARM_THRESHOLD_NATS",
    "DEFAULT_MAX_MISMATCH_NATS",
    "DEFAULT_MAX_ROUTING_KL",
    "GENERAL_EXPERT_PREFIX",
    "compute_prefix_cached_layerwise_attribution",
    "detect_router_expert_coadaptation_mismatch",
    "execute_coadaptation_guarded_surgery",
    "verify_coadaptation_receipt_provenance",
]

# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""DeepSeek-V3 Auxiliary-Loss-Free Router Bias Calibration for Open-Weight Sparse MoEs.

Implements training-free, bias-only load-balancing calibration for upcycled open-weight
Sparse MoE models (``CrowdV1WithEngram``, ``CrowdLoRAMoEMLP``, ``CrowdFineGrainedMoEMLP``,
and standalone ``sigma_topk`` / ``softmax_topk`` linear routers):

1. ``measure_expert_topk_utilization``:
   Computes exact top-k token-routing slot fractions, token activation hit rates,
   Coefficient of Variation (``load_cv``), Gini imbalance (``gini_imbalance``),
   normalized routing entropy, and dead/starved expert counts over active experts.
2. ``calibrate_router_bias_auxiliary_free``:
   Iteratively tunes *only* the ``E``-element router bias vector ``b`` (``<= 0.001%`` of
   full layer parameters; router weight matrix ``W`` and all expert adapters remain bitwise
   untouched) using DeepSeek-V3 auxiliary-loss-free proportional load-error steps with
   adaptive backtracking, zero-mean active centering, and FoMoE ghost-sink margin locking
   so general prose routing stays strictly intact.
3. ``apply_verified_router_bias_calibration_inplace``:
   Sequentially traces calibration activations through MoE blocks, calibrates router
   biases in-place, verifies non-bias bitwise immutability, enforces the ANSE
   ``<= +1.0%`` domain drift guard and specialist NLL retention/improvement, and performs
   atomic bitwise rollback (including removing newly attached bias tensors if originally
   ``bias=False``) with SHA-256 provenance receipts.
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

from hivemind.swarm.crowdgpt_moe_opendiloco import load_moe_trainable_buckets
from hivemind.swarm.layer_localized_attribution import (
    compute_embeddings_batch,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    per_chunk_causal_midrow_nll,
)

SCHEMA_VERSION: str = "hivemind.moe_load_balance_bias_controller/1"
RECEIPT_OBJECT: str = "hivemind.moe_load_balance_bias_calibration_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_TOP_K: int = 2
DEFAULT_MAX_ITERS: int = 25
DEFAULT_STEP_SIZE: float = 0.15
DEFAULT_CV_TOL: float = 0.05
DEFAULT_MAX_BIAS_MAGNITUDE: float = 20.0
DEFAULT_SCORE_FROM: int = 64
DEFAULT_BATCH_SIZE: int = 24
DEFAULT_MAX_DOMAIN_DRIFT: float = 0.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_ANSE_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_MAX_PARAM_MOD_FRACTION: float = 0.0001
DEFAULT_SINK_MARGIN_EPS: float = 1e-3

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_ROUTER_BIAS_KEY_RE = re.compile(r"(?:^|\.)router\.bias$")


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
        raise ValueError(f"{name} must be >= 0.0, got {value!r}")
    return out


def _validate_positive_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite strictly positive real number."""
    out = _validate_nonneg_finite(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_unit_open_top(value: Any, name: str) -> float:
    """Validate that ``value`` is finite and lies in ``[0.0, 1.0)``."""
    out = _validate_nonneg_finite(value, name)
    if out >= 1.0:
        raise ValueError(f"{name} must be < 1.0, got {value!r}")
    return out


def _validate_2d_or_3d_floating(tensor: Any, name: str) -> torch.Tensor:
    """Validate a finite non-empty 2D/3D floating-point tensor and flatten to CPU ``float64``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim not in (2, 3) or any(dim == 0 for dim in tensor.shape):
        raise ValueError(f"{name} must be a non-empty 2D or 3D tensor, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    flat = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1, int(tensor.shape[-1]))
    if not bool(torch.isfinite(flat).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return flat.contiguous()


def _validate_expert_index_sequence(
    indices: Sequence[int] | set[int] | frozenset[int],
    num_experts: int,
    name: str,
) -> list[int]:
    """Validate non-boolean expert indices in ``[0, num_experts)`` and return sorted unique list."""
    if isinstance(indices, (str, bytes, bytearray)) or not isinstance(
        indices, (Sequence, set, frozenset)
    ):
        raise TypeError(f"{name} must be a sequence or set of expert indices")
    out: set[int] = set()
    for raw_idx in indices:
        idx = _validate_nonneg_int(raw_idx, name)
        if idx >= num_experts:
            raise ValueError(f"{name} index {idx} out of range [0, {num_experts})")
        out.add(idx)
    return sorted(out)


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically with strict finite-number enforcement."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def compute_payload_sha256(payload: Any) -> str:
    """Return lowercase SHA-256 hex digest of ``canonical_json_bytes(payload)``."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _hash_state_items(state: Mapping[str, torch.Tensor], *, exclude_router_bias: bool) -> str:
    """Hash sorted tensor items deterministically, optionally excluding ``*.router.bias``."""
    keys = [
        k
        for k in sorted(state.keys())
        if not (exclude_router_bias and _ROUTER_BIAS_KEY_RE.search(k))
    ]
    if not keys:
        raise ValueError("module state_dict contains no hashable tensors")
    digest = hashlib.sha256()
    for key in keys:
        t_cpu = state[key].detach().to(device="cpu").contiguous()
        if t_cpu.is_floating_point() and not bool(torch.isfinite(t_cpu).all().item()):
            raise ValueError(f"module state tensor {key!r} contains non-finite values")
        digest.update(f"{key}:{t_cpu.dtype}:{tuple(t_cpu.shape)}:".encode())
        digest.update(t_cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def compute_module_state_sha256(module: nn.Module) -> str:
    """Compute a deterministic bitwise SHA-256 digest over a module's full ``state_dict``."""
    if not isinstance(module, nn.Module):
        raise TypeError(f"module must be an nn.Module, got {type(module).__name__}")
    state = module.state_dict()
    if not state:
        raise ValueError("module state_dict must be non-empty")
    return _hash_state_items(state, exclude_router_bias=False)


def compute_non_router_bias_state_sha256(module: nn.Module) -> str:
    """Compute a deterministic bitwise SHA-256 digest over all tensors except ``router.bias``."""
    if not isinstance(module, nn.Module):
        raise TypeError(f"module must be an nn.Module, got {type(module).__name__}")
    state = module.state_dict()
    if not state:
        raise ValueError("module state_dict must be non-empty")
    return _hash_state_items(state, exclude_router_bias=True)


def verify_router_bias_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify canonical ``receipt_sha256`` integrity on a calibration receipt."""
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


@dataclass(frozen=True)
class RouterBiasCalibrationConfig:
    """Configuration governing auxiliary-loss-free router bias calibration and ANSE gating."""

    top_k: int = DEFAULT_TOP_K
    max_iters: int = DEFAULT_MAX_ITERS
    step_size: float = DEFAULT_STEP_SIZE
    cv_tol: float = DEFAULT_CV_TOL
    max_bias_magnitude: float = DEFAULT_MAX_BIAS_MAGNITUDE
    center_bias: bool = True
    general_prefix: str = GENERAL_EXPERT_PREFIX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    max_domain_drift: float = DEFAULT_MAX_DOMAIN_DRIFT
    anse_safe_e0_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain: float = DEFAULT_ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY
    min_energy_improvement: float = 0.0
    max_specialist_regression: float = 0.0
    max_modified_param_fraction: float | None = None
    sink_margin_eps: float = DEFAULT_SINK_MARGIN_EPS
    inplace: bool = True

    def __post_init__(self) -> None:
        _validate_positive_int(self.top_k, "top_k")
        _validate_positive_int(self.max_iters, "max_iters")
        _validate_nonneg_finite(self.step_size, "step_size")
        _validate_nonneg_finite(self.cv_tol, "cv_tol")
        _validate_positive_finite(self.max_bias_magnitude, "max_bias_magnitude")
        _validate_bool(self.center_bias, "center_bias")
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_nonneg_finite(self.max_domain_drift, "max_domain_drift")
        _validate_positive_finite(self.anse_safe_e0_ratio, "anse_safe_e0_ratio")
        _validate_positive_finite(self.maximum_pain, "maximum_pain")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_nonneg_finite(self.min_energy_improvement, "min_energy_improvement")
        _validate_nonneg_finite(self.max_specialist_regression, "max_specialist_regression")
        if self.max_modified_param_fraction is not None:
            _validate_positive_finite(
                self.max_modified_param_fraction, "max_modified_param_fraction"
            )
        _validate_positive_finite(self.sink_margin_eps, "sink_margin_eps")
        _validate_bool(self.inplace, "inplace")


# ---------------------------------------------------------------------------
# 2. Top-K Utilization & Load-Imbalance Measurement
# ---------------------------------------------------------------------------


def _compute_cv_and_gini(counts: Sequence[int]) -> tuple[float, float]:
    """Compute exact population Coefficient of Variation and Gini coefficient over counts."""
    m = len(counts)
    total = sum(counts)
    if m == 1:
        return (0.0, 0.0) if total > 0 else (1.0, 1.0)
    if total == 0:
        return math.sqrt(float(m)), float(m - 1) / float(m)
    mean_c = float(total) / float(m)
    var_c = sum((float(c) - mean_c) ** 2 for c in counts) / float(m)
    cv = math.sqrt(var_c) / mean_c
    sorted_c = sorted(float(c) for c in counts)
    weighted = sum((2 * (idx + 1) - m - 1) * val for idx, val in enumerate(sorted_c))
    gini = max(0.0, min(1.0, weighted / (float(m) * float(total))))
    return cv, gini


@torch.no_grad()
def measure_expert_topk_utilization(
    router_logits: torch.Tensor,
    *,
    top_k: int = DEFAULT_TOP_K,
    active_experts: Sequence[int] | None = None,
    dead_fraction_threshold: float = 0.0,
) -> dict[str, Any]:
    """Measure exact top-k token-routing fractions, CV/Gini load imbalance, and dead experts.

    Args:
        router_logits: Logit tensor of shape ``(N, E)`` or ``(B, T, E)``.
        top_k: Number of experts activated per token (``1 <= top_k <= E``).
        active_experts: Optional subset of active expert indices over which balance
            statistics (``load_cv``, ``gini_imbalance``, ``dead_expert_count``) are
            computed (e.g., excluding FoMoE ghost sinks). Defaults to all ``E`` experts.
        dead_fraction_threshold: Optional routing-fraction threshold in ``[0.0, 1.0)``
            below or equal to which an expert is flagged dead (default ``0.0`` checks
            exact zero assignments).

    Returns:
        Dictionary with exact token-routing slot fractions, token hit rates,
        ``load_cv``, ``gini_imbalance``, ``dead_expert_count``, and diagnostics.
    """
    logits_2d = _validate_2d_or_3d_floating(router_logits, "router_logits")
    num_tokens, num_experts = int(logits_2d.shape[0]), int(logits_2d.shape[1])
    k_val = _validate_positive_int(top_k, "top_k")
    if k_val > num_experts:
        raise ValueError(f"top_k={k_val} exceeds num_experts={num_experts}")
    dead_thr = _validate_unit_open_top(dead_fraction_threshold, "dead_fraction_threshold")

    if active_experts is None:
        active = list(range(num_experts))
    else:
        active = _validate_expert_index_sequence(active_experts, num_experts, "active_experts")
        if not active:
            raise ValueError("active_experts must not be empty")

    _, top_idx = torch.topk(logits_2d, k=k_val, dim=-1)
    counts_t = torch.bincount(top_idx.reshape(-1), minlength=num_experts).to(torch.int64)
    counts = [int(x) for x in counts_t.tolist()]

    total_slots = float(num_tokens * k_val)
    token_routing_fractions = [float(c) / total_slots for c in counts]
    token_hit_rates = [float(c) / float(num_tokens) for c in counts]

    active_counts = [counts[e] for e in active]
    active_sum = sum(active_counts)
    active_routing_fractions = (
        [float(c) / float(active_sum) for c in active_counts] if active_sum > 0 else [0.0] * len(active)
    )
    active_token_hit_rates = [float(c) / float(num_tokens) for c in active_counts]

    load_cv, gini_imbalance = _compute_cv_and_gini(active_counts)
    global_load_cv, global_gini_imbalance = _compute_cv_and_gini(counts)

    if dead_thr == 0.0:
        dead_experts = [e for e in active if counts[e] == 0]
        all_dead_experts = [e for e in range(num_experts) if counts[e] == 0]
    else:
        dead_experts = [
            e
            for pos, e in enumerate(active)
            if active_sum == 0 or active_routing_fractions[pos] <= dead_thr
        ]
        all_dead_experts = [
            e for e in range(num_experts) if token_routing_fractions[e] <= dead_thr
        ]

    max_c = max(active_counts)
    min_c = min(active_counts)
    max_min_ratio = (float(max_c) / float(min_c)) if min_c > 0 else None
    min_max_ratio = (float(min_c) / float(max_c)) if max_c > 0 else 0.0

    if len(active) <= 1:
        norm_entropy = 1.0 if active_sum > 0 else 0.0
    elif active_sum == 0:
        norm_entropy = 0.0
    else:
        entropy = -sum(p * math.log(p) for p in active_routing_fractions if p > 0.0)
        norm_entropy = max(0.0, min(1.0, entropy / math.log(float(len(active)))))

    sig_probs = torch.sigmoid(logits_2d)
    mean_sig = [float(x) for x in sig_probs.mean(dim=0).tolist()]
    sig_norm = sig_probs / sig_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    mean_sig_norm = sig_norm.mean(dim=0)
    switch_aux = float(num_experts) * sum(
        token_routing_fractions[e] * float(mean_sig_norm[e].item()) for e in range(num_experts)
    )

    return {
        "num_tokens": num_tokens,
        "num_experts": num_experts,
        "top_k": k_val,
        "active_experts": active,
        "expert_counts": counts,
        "token_routing_fractions": token_routing_fractions,
        "token_hit_rates": token_hit_rates,
        "active_routing_fractions": active_routing_fractions,
        "active_token_hit_rates": active_token_hit_rates,
        "max_routing_fraction": max(active_routing_fractions),
        "max_token_hit_rate": max(active_token_hit_rates),
        "load_cv": load_cv,
        "global_load_cv": global_load_cv,
        "gini_imbalance": gini_imbalance,
        "global_gini_imbalance": global_gini_imbalance,
        "max_min_ratio": max_min_ratio,
        "min_max_ratio": min_max_ratio,
        "normalized_entropy": norm_entropy,
        "dead_experts": dead_experts,
        "dead_expert_count": len(dead_experts),
        "all_dead_experts": all_dead_experts,
        "all_dead_expert_count": len(all_dead_experts),
        "mean_sigmoid_gate": mean_sig,
        "switch_aux_loss": switch_aux,
    }


# ---------------------------------------------------------------------------
# 3. DeepSeek-V3 Auxiliary-Loss-Free Router Bias Calibration
# ---------------------------------------------------------------------------


def _resolve_router_or_bias(
    router_linear_or_bias: nn.Linear | nn.Module | torch.Tensor,
    mlp_inputs: torch.Tensor,
    *,
    top_k: int | None,
) -> tuple[str, nn.Linear | None, torch.Tensor | None, torch.Tensor, torch.Tensor, int, torch.dtype, torch.device]:
    """Resolve router linear, bias vector, base logits, and target dtype/device."""
    x_2d = _validate_2d_or_3d_floating(mlp_inputs, "mlp_inputs")

    if isinstance(router_linear_or_bias, torch.Tensor):
        if router_linear_or_bias.ndim != 1 or router_linear_or_bias.numel() < 2:
            raise ValueError(
                f"1D bias tensor must have shape (E,) with E >= 2, got {tuple(router_linear_or_bias.shape)}"
            )
        if not router_linear_or_bias.is_floating_point():
            raise TypeError(
                f"bias tensor must have floating-point dtype, got {router_linear_or_bias.dtype}"
            )
        b_64 = router_linear_or_bias.detach().to(device="cpu", dtype=torch.float64).contiguous()
        if not bool(torch.isfinite(b_64).all().item()):
            raise ValueError("router_linear_or_bias tensor must contain only finite values")
        num_experts = int(b_64.shape[0])
        if int(x_2d.shape[1]) != num_experts:
            raise ValueError(
                f"When router_linear_or_bias is a 1D bias of size {num_experts}, "
                f"mlp_inputs must be logits with last dim {num_experts}, got {x_2d.shape[1]}"
            )
        k_val = DEFAULT_TOP_K if top_k is None else _validate_positive_int(top_k, "top_k")
        if k_val > num_experts:
            raise ValueError(f"top_k={k_val} exceeds num_experts={num_experts}")
        return (
            "bias_tensor",
            None,
            router_linear_or_bias,
            x_2d,
            b_64,
            k_val,
            router_linear_or_bias.dtype,
            router_linear_or_bias.device,
        )

    if isinstance(router_linear_or_bias, nn.Linear):
        router = router_linear_or_bias
        inferred_k = DEFAULT_TOP_K
    elif isinstance(router_linear_or_bias, nn.Module) and isinstance(
        getattr(router_linear_or_bias, "router", None), nn.Linear
    ):
        router = router_linear_or_bias.router
        inferred_k = int(getattr(router_linear_or_bias, "top_k", DEFAULT_TOP_K))
    else:
        raise TypeError(
            "router_linear_or_bias must be an nn.Linear, an nn.Module exposing '.router', "
            f"or a 1D bias torch.Tensor; got {type(router_linear_or_bias).__name__}"
        )

    if router.weight.ndim != 2 or router.weight.shape[0] < 2 or router.weight.shape[1] == 0:
        raise ValueError(
            f"router.weight must be 2D with out_features >= 2, got {tuple(router.weight.shape)}"
        )
    if not router.weight.is_floating_point():
        raise TypeError(f"router.weight must have floating-point dtype, got {router.weight.dtype}")
    w_64 = router.weight.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(w_64).all().item()):
        raise ValueError("router.weight must contain only finite values")

    num_experts, in_features = int(w_64.shape[0]), int(w_64.shape[1])
    if int(x_2d.shape[1]) != in_features:
        raise ValueError(
            f"mlp_inputs width {x_2d.shape[1]} does not match router in_features {in_features}"
        )
    k_val = inferred_k if top_k is None else _validate_positive_int(top_k, "top_k")
    if k_val > num_experts:
        raise ValueError(f"top_k={k_val} exceeds num_experts={num_experts}")

    if router.bias is not None:
        if router.bias.shape != (num_experts,):
            raise ValueError(f"router.bias must have shape ({num_experts},), got {tuple(router.bias.shape)}")
        b_64 = router.bias.detach().to(device="cpu", dtype=torch.float64).contiguous()
        if not bool(torch.isfinite(b_64).all().item()):
            raise ValueError("router.bias must contain only finite values")
    else:
        b_64 = torch.zeros(num_experts, dtype=torch.float64)

    base_logits = (x_2d @ w_64.T).contiguous()
    return (
        "linear",
        router,
        None,
        base_logits,
        b_64,
        k_val,
        router.weight.dtype,
        router.weight.device,
    )


def _normalize_target_utilization(
    target_utilization: Sequence[float] | Mapping[int, float] | torch.Tensor | None,
    *,
    num_experts: int,
    active: list[int],
) -> list[float]:
    """Validate and normalize target routing fractions over ``active`` experts."""
    m = len(active)
    if target_utilization is None:
        return [1.0 / float(m)] * m

    raw_active: list[float]
    if isinstance(target_utilization, Mapping):
        if not target_utilization:
            raise ValueError("target_utilization mapping must not be empty")
        parsed_map: dict[int, float] = {}
        for raw_k, raw_v in target_utilization.items():
            e_idx = _validate_nonneg_int(raw_k, "target_utilization key")
            if e_idx >= num_experts:
                raise ValueError(f"target_utilization key {e_idx} out of range [0, {num_experts})")
            parsed_map[e_idx] = _validate_nonneg_finite(raw_v, f"target_utilization[{e_idx}]")
        keys = set(parsed_map.keys())
        if keys != set(active) and keys != set(range(num_experts)):
            raise ValueError(
                "target_utilization mapping keys must match either active_experts or all experts"
            )
        raw_active = [parsed_map[e] for e in active]
    elif isinstance(target_utilization, torch.Tensor):
        if target_utilization.ndim != 1 or target_utilization.dtype == torch.bool:
            raise ValueError("target_utilization tensor must be a 1D non-bool numeric tensor")
        vals = [
            _validate_nonneg_finite(float(x), "target_utilization")
            for x in target_utilization.detach().to("cpu", dtype=torch.float64).tolist()
        ]
        if len(vals) == num_experts:
            raw_active = [vals[e] for e in active]
        elif len(vals) == m:
            raw_active = vals
        else:
            raise ValueError(
                f"target_utilization length {len(vals)} must match num_experts={num_experts} "
                f"or active_experts={m}"
            )
    elif isinstance(target_utilization, Sequence) and not isinstance(
        target_utilization, (str, bytes, bytearray)
    ):
        vals = [_validate_nonneg_finite(x, "target_utilization") for x in target_utilization]
        if len(vals) == num_experts:
            raw_active = [vals[e] for e in active]
        elif len(vals) == m:
            raw_active = vals
        else:
            raise ValueError(
                f"target_utilization length {len(vals)} must match num_experts={num_experts} "
                f"or active_experts={m}"
            )
    else:
        raise TypeError("target_utilization must be None, a Sequence, Mapping, or 1D torch.Tensor")

    total = sum(raw_active)
    if total <= 0.0:
        raise ValueError("target_utilization must have positive sum over active experts")
    return [v / total for v in raw_active]


def _target_error_cv(active_fracs: Sequence[float], target_fracs: Sequence[float]) -> float:
    """Compute normalized RMS error against ``target_fracs`` (equals ``load_cv`` when uniform)."""
    m = len(active_fracs)
    floor = 1.0 / float(m)
    sq_sum = 0.0
    for f_obs, f_tgt in zip(active_fracs, target_fracs, strict=True):
        denom = max(f_tgt, floor)
        diff = (f_obs - f_tgt) / denom
        sq_sum += diff * diff
    return math.sqrt(sq_sum / float(m))


def _prepare_prose_sink_guard(
    prose_inputs: torch.Tensor | None,
    *,
    kind: str,
    router: nn.Linear | None,
    bias_init: torch.Tensor,
    sink_experts: Sequence[int],
    top_k: int,
    sink_margin_eps: float,
) -> dict[str, Any] | None:
    """Precompute prose base logits and baseline ghost-sink selection masks for strict preservation."""
    if prose_inputs is None or not sink_experts:
        return None
    p_2d = _validate_2d_or_3d_floating(prose_inputs, "prose_inputs")
    num_experts = int(bias_init.shape[0])
    if kind == "linear":
        assert router is not None
        in_features = int(router.weight.shape[1])
        if int(p_2d.shape[1]) != in_features:
            raise ValueError(
                f"prose_inputs width {p_2d.shape[1]} does not match router in_features {in_features}"
            )
        p_base = (p_2d @ router.weight.detach().to(device="cpu", dtype=torch.float64).T).contiguous()
    else:
        if int(p_2d.shape[1]) != num_experts:
            raise ValueError(
                f"prose_inputs width {p_2d.shape[1]} does not match num_experts {num_experts}"
            )
        p_base = p_2d

    p_init = p_base + bias_init.unsqueeze(0)
    _, init_top_idx = torch.topk(p_init, k=top_k, dim=-1)
    init_sig_top = torch.sigmoid(p_init.gather(1, init_top_idx))

    sink_mask_e = torch.zeros(num_experts, dtype=torch.bool)
    for s_idx in sink_experts:
        sink_mask_e[s_idx] = True

    sink_hit_init = sink_mask_e[init_top_idx]  # [N_prose, top_k]
    return {
        "prose_base_logits": p_base,
        "init_top_idx": init_top_idx,
        "init_sig_top": init_sig_top,
        "sink_mask_e": sink_mask_e,
        "sink_hit_init": sink_hit_init,
        "sink_margin_eps": float(sink_margin_eps),
    }


def _apply_prose_sink_upper_bound(
    b_vec: torch.Tensor,
    active: list[int],
    sink_guard: dict[str, Any] | None,
    *,
    max_bias_mag: float,
) -> torch.Tensor:
    """Shift ``b_vec[active]`` uniformly downward if needed to preserve exact prose sink routing."""
    if sink_guard is None:
        return b_vec
    p_base: torch.Tensor = sink_guard["prose_base_logits"]
    init_top_idx: torch.Tensor = sink_guard["init_top_idx"]
    sink_hit_init: torch.Tensor = sink_guard["sink_hit_init"]
    eps: float = float(sink_guard["sink_margin_eps"])

    has_sink = sink_hit_init.any(dim=-1)
    if not bool(has_sink.any().item()):
        return b_vec

    # Minimum logit among initially routed ghost sinks on each prose token (sinks have frozen bias).
    p_cur = p_base + b_vec.unsqueeze(0)
    top_logits = p_cur.gather(1, init_top_idx)
    masked_sink_logits = torch.where(sink_hit_init, top_logits, torch.full_like(top_logits, float("inf")))
    min_sink_logit = masked_sink_logits.min(dim=-1).values  # [N_prose]

    excess = 0.0
    for e_idx in active:
        was_selected = (init_top_idx == e_idx).any(dim=-1)
        guard_tok = has_sink & (~was_selected)
        if not bool(guard_tok.any().item()):
            continue
        upper_e = float((min_sink_logit[guard_tok] - p_base[guard_tok, e_idx] - eps).min().item())
        if float(b_vec[e_idx].item()) > upper_e:
            excess = max(excess, float(b_vec[e_idx].item()) - upper_e)

    if excess > 0.0:
        out = b_vec.clone()
        out[active] = (out[active] - excess).clamp(-max_bias_mag, max_bias_mag)
        return out
    return b_vec


def _evaluate_prose_sink_preservation(
    b_after: torch.Tensor,
    sink_guard: dict[str, Any] | None,
    *,
    top_k: int,
) -> dict[str, Any]:
    """Verify exact preservation of FoMoE ghost-sink routing and gates on prose."""
    if sink_guard is None:
        return {"evaluated": False, "exact_sink_routing_preserved": True, "sink_match_fraction": 1.0, "max_sink_gate_diff": 0.0}
    p_base: torch.Tensor = sink_guard["prose_base_logits"]
    init_top_idx: torch.Tensor = sink_guard["init_top_idx"]
    init_sig_top: torch.Tensor = sink_guard["init_sig_top"]
    sink_hit_init: torch.Tensor = sink_guard["sink_hit_init"]
    sink_mask_e: torch.Tensor = sink_guard["sink_mask_e"]

    p_after = p_base + b_after.unsqueeze(0)
    _, after_top_idx = torch.topk(p_after, k=top_k, dim=-1)
    after_sig_top = torch.sigmoid(p_after.gather(1, after_top_idx))

    n_prose = int(p_base.shape[0])
    exact_matches = 0
    max_gate_diff = 0.0
    for t_idx in range(n_prose):
        init_sinks = {
            int(init_top_idx[t_idx, slot].item()): float(init_sig_top[t_idx, slot].item())
            for slot in range(top_k)
            if bool(sink_hit_init[t_idx, slot].item())
        }
        after_sinks = {
            int(after_top_idx[t_idx, slot].item()): float(after_sig_top[t_idx, slot].item())
            for slot in range(top_k)
            if bool(sink_mask_e[after_top_idx[t_idx, slot]].item())
        }
        if set(init_sinks.keys()).issubset(set(after_sinks.keys())):
            exact_matches += 1
            for s_idx, g_before in init_sinks.items():
                max_gate_diff = max(max_gate_diff, abs(after_sinks[s_idx] - g_before))
        else:
            max_gate_diff = max(max_gate_diff, 1.0)

    match_frac = float(exact_matches) / float(max(1, n_prose))
    return {
        "evaluated": True,
        "num_prose_tokens": n_prose,
        "exact_sink_routing_preserved": exact_matches == n_prose,
        "sink_match_fraction": match_frac,
        "max_sink_gate_diff": max_gate_diff,
    }


def _compute_bias_update_direction(
    logits: torch.Tensor,
    active: list[int],
    active_fracs: Sequence[float],
    target_fracs: Sequence[float],
    *,
    top_k: int,
    step_size: float,
    center_bias: bool,
) -> torch.Tensor:
    """Compute one DeepSeek-V3 auxiliary-loss-free quantile-gap bias step."""
    m = len(active)
    num_experts = int(logits.shape[1])
    sorted_vals, _ = torch.sort(logits, dim=-1, descending=True)
    tau_k = sorted_vals[:, top_k - 1]
    tau_next = sorted_vals[:, min(top_k, num_experts - 1)]
    _, top_idx = torch.topk(logits, k=top_k, dim=-1)

    n_tokens = int(logits.shape[0])
    active_counts_total = float(sum(int((top_idx == e).any(dim=-1).sum().item()) for e in active))
    eta = min(0.85, 3.0 * step_size)

    raw_step = torch.zeros(m, dtype=torch.float64)
    for pos, e_idx in enumerate(active):
        f_obs = active_fracs[pos]
        f_tgt = target_fracs[pos]
        if abs(f_tgt - f_obs) <= 1e-12:
            continue

        hit_tok = (top_idx == e_idx).any(dim=-1)
        cur_cnt = float(int(hit_tok.sum().item()))
        tgt_cnt = f_tgt * active_counts_total

        if tgt_cnt > cur_cnt:
            unhit_tok = ~hit_tok
            n_unhit = int(unhit_tok.sum().item())
            if n_unhit > 0:
                need_frac = min(0.95, max(0.01, (tgt_cnt - cur_cnt) / float(n_unhit)))
                gaps = (tau_k[unhit_tok] - logits[unhit_tok, e_idx]).clamp_min(0.0)
                gap_q = float(torch.quantile(gaps, need_frac).item())
                raw_step[pos] = eta * max(gap_q, step_size * (tgt_cnt - cur_cnt) / max(1.0, tgt_cnt))
        else:
            n_hit = int(hit_tok.sum().item())
            if n_hit > 0 and top_k < num_experts:
                shed_frac = min(0.95, max(0.01, (cur_cnt - tgt_cnt) / float(n_hit)))
                margins = (logits[hit_tok, e_idx] - tau_next[hit_tok]).clamp_min(0.0)
                margin_q = float(torch.quantile(margins, shed_frac).item())
                raw_step[pos] = -eta * max(margin_q, step_size * (cur_cnt - tgt_cnt) / max(1.0, tgt_cnt))
            elif n_hit > 0:
                raw_step[pos] = -step_size * (cur_cnt - tgt_cnt) / float(max(1, n_tokens))

    if center_bias and m > 1:
        raw_step = raw_step - raw_step.mean()
    return raw_step


def _write_bias_inplace(
    kind: str,
    router: nn.Linear | None,
    bias_tensor_ref: torch.Tensor | None,
    calibrated_bias_64: torch.Tensor,
    target_dtype: torch.dtype,
    target_device: torch.device,
) -> None:
    """Apply calibrated 1D bias vector in-place onto ``nn.Linear`` or raw 1D tensor."""
    typed_b = calibrated_bias_64.to(device=target_device, dtype=target_dtype)
    if kind == "bias_tensor":
        assert bias_tensor_ref is not None
        bias_tensor_ref.copy_(typed_b)
        return
    assert router is not None
    if router.bias is None:
        router.register_parameter(
            "bias",
            nn.Parameter(typed_b.clone(), requires_grad=bool(router.weight.requires_grad)),
        )
    else:
        router.bias.copy_(typed_b)


@torch.no_grad()
def calibrate_router_bias_auxiliary_free(
    router_linear_or_bias: nn.Linear | nn.Module | torch.Tensor,
    mlp_inputs: torch.Tensor,
    *,
    target_utilization: Sequence[float] | Mapping[int, float] | torch.Tensor | None = None,
    protect_sink_experts: Sequence[int] = (),
    top_k: int | None = None,
    max_iters: int = DEFAULT_MAX_ITERS,
    step_size: float = DEFAULT_STEP_SIZE,
    cv_tol: float = DEFAULT_CV_TOL,
    max_bias_magnitude: float = DEFAULT_MAX_BIAS_MAGNITUDE,
    center_bias: bool = True,
    prose_inputs: torch.Tensor | None = None,
    sink_margin_eps: float = DEFAULT_SINK_MARGIN_EPS,
    inplace: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Calibrate only the ``E``-parameter router bias vector via DeepSeek-V3 load balancing.

    Iteratively adjusts only the ``E``-element bias vector without altering the router weight
    matrix ``W`` or any expert adapter weights. Protects ``protect_sink_experts`` bitwise and
    optionally enforces exact prose ghost-sink preservation across ``prose_inputs``.
    """
    kind, router, bias_ref, base_logits, b_init, k_val, out_dt, out_dev = _resolve_router_or_bias(
        router_linear_or_bias,
        mlp_inputs,
        top_k=top_k,
    )
    num_experts = int(b_init.shape[0])
    iters_cap = _validate_positive_int(max_iters, "max_iters")
    step_f = _validate_nonneg_finite(step_size, "step_size")
    tol_f = _validate_nonneg_finite(cv_tol, "cv_tol")
    max_mag = _validate_positive_finite(max_bias_magnitude, "max_bias_magnitude")
    _validate_bool(center_bias, "center_bias")
    _validate_positive_finite(sink_margin_eps, "sink_margin_eps")
    _validate_bool(inplace, "inplace")

    sinks = _validate_expert_index_sequence(protect_sink_experts, num_experts, "protect_sink_experts")
    sink_set = set(sinks)
    active = [e for e in range(num_experts) if e not in sink_set]
    if not active:
        raise ValueError("protect_sink_experts cannot exclude all experts")

    target_fracs = _normalize_target_utilization(
        target_utilization,
        num_experts=num_experts,
        active=active,
    )
    sink_guard = _prepare_prose_sink_guard(
        prose_inputs,
        kind=kind,
        router=router,
        bias_init=b_init,
        sink_experts=sinks,
        top_k=k_val,
        sink_margin_eps=sink_margin_eps,
    )

    b_cur = b_init.clone()
    before_util = measure_expert_topk_utilization(
        base_logits + b_cur.unsqueeze(0), top_k=k_val, active_experts=active
    )
    init_err_cv = _target_error_cv(before_util["active_routing_fractions"], target_fracs)

    def _score_tuple(util_dict: dict[str, Any], err_val: float) -> tuple[int, float]:
        needed_dead = sum(
            1
            for pos, e_idx in enumerate(active)
            if target_fracs[pos] > 0.0 and util_dict["expert_counts"][e_idx] == 0
        )
        return (needed_dead, round(err_val, 10))

    best_b = b_cur.clone()
    best_util = before_util
    best_err = init_err_cv
    best_key = _score_tuple(before_util, init_err_cv)
    history: list[dict[str, Any]] = [
        {
            "iter": 0,
            "load_cv": float(before_util["load_cv"]),
            "target_error_cv": init_err_cv,
            "gini_imbalance": float(before_util["gini_imbalance"]),
            "dead_expert_count": int(before_util["dead_expert_count"]),
        }
    ]

    iters_executed = 0
    if step_f > 0.0 and len(active) > 1 and not (best_err <= tol_f and best_key[0] == 0):
        for it_idx in range(1, iters_cap + 1):
            cur_logits = base_logits + b_cur.unsqueeze(0)
            cur_util = measure_expert_topk_utilization(
                cur_logits, top_k=k_val, active_experts=active
            )

            delta_active = _compute_bias_update_direction(
                cur_logits,
                active,
                cur_util["active_routing_fractions"],
                target_fracs,
                top_k=k_val,
                step_size=step_f,
                center_bias=center_bias,
            )
            if float(delta_active.abs().max().item()) < 1e-12:
                break

            # Backtracking line check: evaluate larger step first when tied on discrete plateaus,
            # while damping to 0.5 / 0.25 if the full step overshoots.
            chosen_b: torch.Tensor | None = None
            chosen_util: dict[str, Any] | None = None
            chosen_err = math.inf
            chosen_key = (num_experts + 1, math.inf)

            for cand_vec in (1.5 * delta_active, delta_active, 0.5 * delta_active, 0.25 * delta_active):
                cand_b = b_cur.clone()
                cand_b[active] = cand_b[active] + cand_vec
                if center_bias and len(active) > 1 and (not sinks or sink_guard is not None):
                    cand_b[active] = cand_b[active] - cand_b[active].mean()
                cand_b[active] = cand_b[active].clamp(-max_mag, max_mag)
                cand_b = _apply_prose_sink_upper_bound(
                    cand_b, active, sink_guard, max_bias_mag=max_mag
                )
                cand_util = measure_expert_topk_utilization(
                    base_logits + cand_b.unsqueeze(0), top_k=k_val, active_experts=active
                )
                cand_err = _target_error_cv(cand_util["active_routing_fractions"], target_fracs)
                cand_key = _score_tuple(cand_util, cand_err)
                if chosen_b is None or cand_key < chosen_key:
                    chosen_b = cand_b
                    chosen_util = cand_util
                    chosen_err = cand_err
                    chosen_key = cand_key

            assert chosen_b is not None and chosen_util is not None
            b_cur = chosen_b
            iters_executed = it_idx
            history.append(
                {
                    "iter": it_idx,
                    "load_cv": float(chosen_util["load_cv"]),
                    "target_error_cv": chosen_err,
                    "gini_imbalance": float(chosen_util["gini_imbalance"]),
                    "dead_expert_count": int(chosen_util["dead_expert_count"]),
                }
            )
            if chosen_key < best_key:
                best_b = chosen_b.clone()
                best_util = chosen_util
                best_err = chosen_err
                best_key = chosen_key

            if best_err <= tol_f and best_key[0] == 0:
                break

    if inplace:
        _write_bias_inplace(kind, router, bias_ref, best_b, out_dt, out_dev)

    sink_pres = _evaluate_prose_sink_preservation(best_b, sink_guard, top_k=k_val)
    if inplace and kind == "bias_tensor" and bias_ref is not None:
        calibrated_out = bias_ref
    else:
        calibrated_out = best_b.to(device=out_dev, dtype=out_dt).contiguous()

    # Exact bitwise count of changed scalars in the E-element bias vector.
    b_init_typed = b_init.to(device=out_dev, dtype=out_dt)
    modified_bias_elements = int((calibrated_out != b_init_typed).sum().item())

    report = {
        "num_experts": num_experts,
        "top_k": k_val,
        "active_experts": active,
        "protected_sink_experts": sinks,
        "target_fractions": target_fracs,
        "iterations_run": iters_executed,
        "converged": bool(best_err <= tol_f and best_key[0] == 0),
        "bias_before": [float(x) for x in b_init.tolist()],
        "bias_after": [float(x) for x in best_b.tolist()],
        "modified_bias_elements": modified_bias_elements,
        "load_cv_before": float(before_util["load_cv"]),
        "load_cv_after": float(best_util["load_cv"]),
        "target_error_cv_before": init_err_cv,
        "target_error_cv_after": best_err,
        "gini_before": float(before_util["gini_imbalance"]),
        "gini_after": float(best_util["gini_imbalance"]),
        "dead_experts_before": int(before_util["dead_expert_count"]),
        "dead_experts_after": int(best_util["dead_expert_count"]),
        "sink_preservation": sink_pres,
        "before_utilization": before_util,
        "after_utilization": best_util,
        "history": history,
    }
    return calibrated_out, report


# ---------------------------------------------------------------------------
# 4. End-to-End ANSE-Gated Verification & Atomic Bitwise Rollback
# ---------------------------------------------------------------------------


def _validate_model_compatible(model: nn.Module) -> list[int]:
    """Verify required transformer attributes and locate MoE blocks exposing ``mlp.router``."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("model must expose a non-empty 'blocks' sequence")
    for attr in ("wte", "ln", "head", "cos_cached", "sin_cached"):
        if not hasattr(model, attr):
            raise ValueError(f"model must expose '{attr}'")
    moe_layers: list[int] = []
    for l_idx, blk in enumerate(blocks):
        mlp = getattr(blk, "mlp", None)
        router = getattr(mlp, "router", None)
        if isinstance(router, nn.Linear):
            moe_layers.append(l_idx)
    if not moe_layers:
        raise ValueError("model must contain at least one block exposing 'mlp.router' nn.Linear")
    return moe_layers


def _validate_eval_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    general_prefix: str,
    vocab_size: int,
    score_from: int,
    name: str,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Validate evaluation chunks and return ``(ids_2d, is_general_mask, expert_ids)``."""
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence) or len(chunks) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")
    rows: list[torch.Tensor] = []
    eids: list[str] = []
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
        eids.append(eid)

    assert seq_len is not None
    start_pos = max(0, score_from - 1)
    if start_pos >= seq_len - 1:
        raise ValueError(
            f"{name} sequence length {seq_len} too short for score_from={score_from} (start_pos={start_pos})"
        )
    ids_2d = torch.stack(rows, dim=0)
    is_general = np.array([eid.startswith(general_prefix) for eid in eids], dtype=bool)
    if not bool(np.any(is_general)):
        raise ValueError(f"{name} must include at least one general chunk starting with {general_prefix!r}")
    return ids_2d, is_general, eids


def _parse_external_eval_domain_nll(
    raw_eval: Any,
    expected_domains: set[str],
) -> tuple[dict[str, float] | None, float | None, bool]:
    """Normalize and finiteness-check external ``eval_fn(model)`` output."""
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
    return cleaned, float(sum(cleaned.values()) / len(cleaned)), True


def _compute_domain_nlls_from_chunks(
    model: nn.Module,
    ids_eval: torch.Tensor,
    expert_ids: Sequence[str],
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
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
        ln_w,
        ln_b,
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


def _evaluate_model_domains(
    model: nn.Module,
    ids_eval: torch.Tensor,
    eval_eids: Sequence[str],
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    ln_eps: float,
    *,
    expected_domains: set[str],
    cfg: RouterBiasCalibrationConfig,
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


def _evaluate_anse_guard(
    ref_domain_nll: Mapping[str, float],
    cand_domain_nll: Mapping[str, float] | None,
    *,
    cand_nll_mean: float | None,
    cand_finite: bool,
    cfg: RouterBiasCalibrationConfig,
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


def _split_general_spec_nll(
    dom_nll: Mapping[str, float],
    general_prefix: str,
) -> tuple[float, float | None]:
    """Return mean general-domain NLL and optional mean specialist-domain NLL."""
    gen_vals = [float(v) for k, v in dom_nll.items() if k.startswith(general_prefix)]
    spec_vals = [float(v) for k, v in dom_nll.items() if not k.startswith(general_prefix)]
    gen_mean = float(sum(gen_vals) / len(gen_vals))
    spec_mean = float(sum(spec_vals) / len(spec_vals)) if spec_vals else None
    return gen_mean, spec_mean


def _restore_model_snapshot_bitwise(
    model: nn.Module,
    snapshot: Mapping[str, torch.Tensor],
    had_bias_by_layer: Mapping[int, bool],
    expected_sha256: str,
) -> str:
    """Atomically restore model state (removing newly added router biases) and verify SHA-256."""
    for l_idx, had_bias in had_bias_by_layer.items():
        if not had_bias:
            router = model.blocks[l_idx].mlp.router
            router.register_parameter("bias", None)

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


def _calibrate_model_layers_sequentially(
    model: nn.Module,
    ids_probe: torch.Tensor,
    is_gen_probe: np.ndarray,
    *,
    target_layer_set: set[int],
    target_utilization: Sequence[float] | Mapping[int, float] | torch.Tensor | None,
    protect_sink_experts: Sequence[int],
    cfg: RouterBiasCalibrationConfig,
) -> list[dict[str, Any]]:
    """Trace probe activations sequentially and calibrate target MoE router biases in-place."""
    start_pos = max(0, cfg.score_from - 1)
    cos = model.cos_cached.to(ids_probe.device)
    sin = model.sin_cached.to(ids_probe.device)
    gen_mask_t = torch.as_tensor(is_gen_probe, dtype=torch.bool, device=ids_probe.device)
    spec_mask_t = ~gen_mask_t

    x_seq = torch.cat(
        [
            compute_embeddings_batch(model, ids_probe[b0 : b0 + cfg.batch_size])
            for b0 in range(0, int(ids_probe.shape[0]), cfg.batch_size)
        ],
        dim=0,
    )

    layer_reports: list[dict[str, Any]] = []
    for l_idx, blk in enumerate(model.blocks):
        xm_parts: list[torch.Tensor] = []
        mi_parts: list[torch.Tensor] = []
        for b0 in range(0, int(x_seq.shape[0]), cfg.batch_size):
            xm, mi = forward_block_pre_mlp(blk, x_seq[b0 : b0 + cfg.batch_size], cos, sin)
            xm_parts.append(xm)
            mi_parts.append(mi)
        x_mid = torch.cat(xm_parts, dim=0)
        mlp_in = torch.cat(mi_parts, dim=0)

        if l_idx in target_layer_set:
            scored = mlp_in[:, start_pos:-1, :]
            h_dim = int(scored.shape[-1])
            prose_acts = scored[gen_mask_t].reshape(-1, h_dim)
            has_spec = bool(spec_mask_t.any().item())
            spec_acts = scored[spec_mask_t].reshape(-1, h_dim) if has_spec else scored.reshape(-1, h_dim)

            calib_acts = spec_acts if (protect_sink_experts and has_spec) else scored.reshape(-1, h_dim)
            prose_guard = prose_acts if protect_sink_experts else None

            _, layer_rep = calibrate_router_bias_auxiliary_free(
                blk.mlp,
                calib_acts,
                target_utilization=target_utilization,
                protect_sink_experts=protect_sink_experts,
                top_k=cfg.top_k,
                max_iters=cfg.max_iters,
                step_size=cfg.step_size,
                cv_tol=cfg.cv_tol,
                max_bias_magnitude=cfg.max_bias_magnitude,
                center_bias=cfg.center_bias,
                prose_inputs=prose_guard,
                sink_margin_eps=cfg.sink_margin_eps,
                inplace=True,
            )
            layer_rep["layer"] = l_idx
            layer_rep["layer_mlp_params"] = sum(p.numel() for p in blk.mlp.parameters())
            layer_reports.append(layer_rep)

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

    return layer_reports


def _build_calibration_receipt(
    *,
    accepted: bool,
    rolled_back: bool,
    status: str,
    parent_sha256: str,
    candidate_sha256: str,
    active_sha256: str,
    non_bias_preserved: bool,
    router_weights_preserved: bool,
    modified_param_count: int,
    total_model_params: int,
    total_mlp_params: int,
    parent_nll_mean: float,
    active_nll_mean: float,
    parent_gen_nll: float,
    active_gen_nll: float,
    parent_spec_nll: float | None,
    active_spec_nll: float | None,
    parent_anse: Mapping[str, Any],
    cand_anse: Mapping[str, Any],
    active_energy: float,
    delta_energy: float,
    layer_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble and SHA-256-sign a canonical router-bias calibration receipt."""
    load_cvs_before = [float(r["load_cv_before"]) for r in layer_reports]
    load_cvs_after = [float(r["load_cv_after"]) for r in layer_reports]
    ginis_before = [float(r["gini_before"]) for r in layer_reports]
    ginis_after = [float(r["gini_after"]) for r in layer_reports]
    dead_before = sum(int(r["dead_experts_before"]) for r in layer_reports)
    dead_after = sum(int(r["dead_experts_after"]) for r in layer_reports)
    max_iters_used = max((int(r["iterations_run"]) for r in layer_reports), default=0)

    sink_preserved = all(
        bool(r["sink_preservation"]["exact_sink_routing_preserved"]) for r in layer_reports
    )
    min_sink_match = min(
        (float(r["sink_preservation"]["sink_match_fraction"]) for r in layer_reports),
        default=1.0,
    )

    mod_frac_model = float(modified_param_count) / float(max(1, total_model_params))
    mod_frac_mlp = float(modified_param_count) / float(max(1, total_mlp_params))

    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "receipt_object": RECEIPT_OBJECT,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "status": status,
        "parent_sha256": parent_sha256,
        "candidate_sha256": candidate_sha256,
        "active_sha256": active_sha256,
        "non_bias_weights_bitwise_preserved": non_bias_preserved,
        "router_weights_bitwise_preserved": router_weights_preserved,
        "modified_param_count": modified_param_count,
        "total_model_params": total_model_params,
        "total_mlp_params": total_mlp_params,
        "modified_param_fraction": mod_frac_model,
        "modified_param_pct": 100.0 * mod_frac_model,
        "mlp_modified_param_fraction": mod_frac_mlp,
        "mlp_modified_param_pct": 100.0 * mod_frac_mlp,
        "load_cv_before": float(sum(load_cvs_before) / len(load_cvs_before)) if load_cvs_before else 0.0,
        "load_cv": float(sum(load_cvs_after) / len(load_cvs_after)) if load_cvs_after else 0.0,
        "gini_before": float(sum(ginis_before) / len(ginis_before)) if ginis_before else 0.0,
        "gini_after": float(sum(ginis_after) / len(ginis_after)) if ginis_after else 0.0,
        "dead_experts_before": dead_before,
        "dead_experts_after": dead_after,
        "max_iterations_run": max_iters_used,
        "exact_sink_routing_preserved": sink_preserved,
        "min_sink_match_fraction": min_sink_match,
        "nll_mean_before": parent_nll_mean,
        "nll_mean": cand_anse["nll_mean"],
        "nll_mean_active": active_nll_mean,
        "general_nll_before": parent_gen_nll,
        "general_nll_active": active_gen_nll,
        "spec_nll_before": parent_spec_nll,
        "spec_nll_active": active_spec_nll,
        "anse_energy_parent": float(parent_anse["anse_energy"]),
        "anse_energy": float(cand_anse["anse_energy"]),
        "anse_energy_active": active_energy,
        "delta_energy": delta_energy,
        "worst_domain_id": cand_anse["worst_domain_id"],
        "worst_domain_drift": cand_anse["worst_domain_drift"],
        "worst_domain_drift_pct": cand_anse["worst_domain_drift_pct"],
        "guard_ok": bool(cand_anse["guard_ok"]),
        "per_domain": cand_anse["per_domain"],
        "layer_reports": layer_reports,
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


@torch.no_grad()
def apply_verified_router_bias_calibration_inplace(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    selection_chunks: Sequence[Mapping[str, Any]] | None = None,
    buckets: Mapping[str, torch.Tensor] | None = None,
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
    cal_ln_w: torch.Tensor | None = None,
    cal_ln_b: torch.Tensor | None = None,
    target_layers: Sequence[int] | None = None,
    target_utilization: Sequence[float] | Mapping[int, float] | torch.Tensor | None = None,
    protect_sink_experts: Sequence[int] = (),
    baseline_domain_nll: Mapping[str, float] | None = None,
    config: RouterBiasCalibrationConfig | None = None,
    top_k: int = DEFAULT_TOP_K,
    max_iters: int = DEFAULT_MAX_ITERS,
    step_size: float = DEFAULT_STEP_SIZE,
    general_prefix: str = GENERAL_EXPERT_PREFIX,
    score_from: int = DEFAULT_SCORE_FROM,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_domain_drift: float = DEFAULT_MAX_DOMAIN_DRIFT,
    max_specialist_regression: float = 0.0,
    eval_fn: Callable[[nn.Module], Mapping[str, Any]] | None = None,
    inplace: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Calibrate MoE router biases across blocks under ANSE gating and atomic rollback.

    Adjusts *only* the ``E``-parameter router bias vectors (keeping all router weights ``W``
    and expert adapters bitwise untouched), verifies ``<= +1.0%`` general/domain drift and
    improved/preserved specialist NLL, and atomically rolls back on any ANSE violation.
    """
    moe_layers = _validate_model_compatible(model)
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError("eval_fn must be callable or None")
    if (cal_ln_w is None) != (cal_ln_b is None):
        raise ValueError("cal_ln_w and cal_ln_b must either both be provided or both be None")

    if config is not None:
        if not isinstance(config, RouterBiasCalibrationConfig):
            raise TypeError(
                f"config must be a RouterBiasCalibrationConfig or None, got {type(config).__name__}"
            )
        cfg = config
    else:
        cfg = RouterBiasCalibrationConfig(
            top_k=top_k,
            max_iters=max_iters,
            step_size=step_size,
            general_prefix=general_prefix,
            score_from=score_from,
            batch_size=batch_size,
            max_domain_drift=max_domain_drift,
            max_specialist_regression=max_specialist_regression,
            inplace=inplace,
        )

    if target_layers is not None:
        if isinstance(target_layers, (str, bytes)) or not isinstance(target_layers, Sequence) or not target_layers:
            raise ValueError("target_layers must be None or a non-empty sequence of layer indices")
        target_layer_set = set()
        num_blocks = len(model.blocks)
        for raw_l in target_layers:
            l_idx = _validate_nonneg_int(raw_l, "target_layer")
            if l_idx >= num_blocks:
                raise ValueError(f"target_layer {l_idx} out of range [0, {num_blocks})")
            if l_idx not in moe_layers:
                raise ValueError(f"target_layer {l_idx} does not expose 'mlp.router'")
            target_layer_set.add(l_idx)
    else:
        target_layer_set = set(moe_layers)

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

    ids_probe, is_gen_probe, _ = _validate_eval_chunks(
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
        parent_non_bias_sha256 = compute_non_router_bias_state_sha256(work_model)
        parent_snapshot = {
            k: v.detach().to(device="cpu").clone() for k, v in work_model.state_dict().items()
        }
        had_bias_by_layer = {
            l_idx: (work_model.blocks[l_idx].mlp.router.bias is not None)
            for l_idx in sorted(target_layer_set)
        }
        router_w_hashes_before = {
            l_idx: hashlib.sha256(
                work_model.blocks[l_idx].mlp.router.weight.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            for l_idx in sorted(target_layer_set)
        }
        total_model_params = sum(p.numel() for p in work_model.parameters())
        total_mlp_params = sum(
            p.numel()
            for l_idx in sorted(target_layer_set)
            for p in work_model.blocks[l_idx].mlp.parameters()
        )

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
        parent_gen_nll, parent_spec_nll = _split_general_spec_nll(parent_dom_nll, cfg.general_prefix)

        layer_reports = _calibrate_model_layers_sequentially(
            work_model,
            ids_probe,
            is_gen_probe,
            target_layer_set=target_layer_set,
            target_utilization=target_utilization,
            protect_sink_experts=protect_sink_experts,
            cfg=cfg,
        )

        cand_non_bias_sha256 = compute_non_router_bias_state_sha256(work_model)
        non_bias_preserved = cand_non_bias_sha256 == parent_non_bias_sha256
        router_w_hashes_after = {
            l_idx: hashlib.sha256(
                work_model.blocks[l_idx].mlp.router.weight.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            for l_idx in sorted(target_layer_set)
        }
        router_weights_preserved = router_w_hashes_before == router_w_hashes_after

        modified_param_count = sum(int(rep["modified_bias_elements"]) for rep in layer_reports)
        total_model_params_after = sum(p.numel() for p in work_model.parameters())
        total_mlp_params_after = sum(
            p.numel()
            for l_idx in sorted(target_layer_set)
            for p in work_model.blocks[l_idx].mlp.parameters()
        )
        mod_frac_model = float(modified_param_count) / float(max(1, total_model_params_after))
        budget_ok = (
            True
            if cfg.max_modified_param_fraction is None
            else (mod_frac_model <= cfg.max_modified_param_fraction)
        )

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

        cand_gen_nll = parent_gen_nll
        cand_spec_nll = parent_spec_nll
        spec_retained = True
        if cand_finite and cand_dom_nll is not None:
            cand_gen_nll, cand_spec_nll = _split_general_spec_nll(cand_dom_nll, cfg.general_prefix)
            if parent_spec_nll is not None and cand_spec_nll is not None:
                spec_retained = (cand_spec_nll - parent_spec_nll) <= (cfg.max_specialist_regression + 1e-12)

        if modified_param_count == 0 and non_bias_preserved and candidate_sha256 == parent_sha256:
            return work_model, _build_calibration_receipt(
                accepted=False,
                rolled_back=False,
                status="NOOP_ALREADY_CALIBRATED",
                parent_sha256=parent_sha256,
                candidate_sha256=candidate_sha256,
                active_sha256=parent_sha256,
                non_bias_preserved=non_bias_preserved,
                router_weights_preserved=router_weights_preserved,
                modified_param_count=0,
                total_model_params=total_model_params,
                total_mlp_params=total_mlp_params,
                parent_nll_mean=parent_nll_mean,
                active_nll_mean=parent_nll_mean,
                parent_gen_nll=parent_gen_nll,
                active_gen_nll=parent_gen_nll,
                parent_spec_nll=parent_spec_nll,
                active_spec_nll=parent_spec_nll,
                parent_anse=parent_anse,
                cand_anse=parent_anse,
                active_energy=float(parent_anse["anse_energy"]),
                delta_energy=0.0,
                layer_reports=layer_reports,
            )

        accepted = bool(
            non_bias_preserved
            and router_weights_preserved
            and budget_ok
            and cand_anse["finite"]
            and cand_anse["guard_ok"]
            and spec_retained
            and (delta_energy < -cfg.min_energy_improvement)
        )

        if accepted:
            rolled_back = False
            status = "COMMITTED"
            active_sha256 = candidate_sha256
            active_nll_mean = float(cand_anse["nll_mean"])
            active_gen_nll = cand_gen_nll
            active_spec_nll = cand_spec_nll
            active_energy = float(cand_anse["anse_energy"])
        else:
            rolled_back = True
            active_sha256 = _restore_model_snapshot_bitwise(
                work_model, parent_snapshot, had_bias_by_layer, parent_sha256
            )
            active_nll_mean = parent_nll_mean
            active_gen_nll = parent_gen_nll
            active_spec_nll = parent_spec_nll
            active_energy = float(parent_anse["anse_energy"])
            if not cand_anse["finite"]:
                status = "ROLLED_BACK_NON_FINITE"
            elif not cand_anse["guard_ok"]:
                status = "ROLLED_BACK_GUARD_VIOLATION"
            elif not budget_ok:
                status = "ROLLED_BACK_BUDGET_EXCEEDED"
            elif not spec_retained:
                status = "ROLLED_BACK_SPEC_REGRESSION"
            else:
                status = "ROLLED_BACK_NO_ENERGY_IMPROVEMENT"

        return work_model, _build_calibration_receipt(
            accepted=accepted,
            rolled_back=rolled_back,
            status=status,
            parent_sha256=parent_sha256,
            candidate_sha256=candidate_sha256,
            active_sha256=active_sha256,
            non_bias_preserved=non_bias_preserved,
            router_weights_preserved=router_weights_preserved,
            modified_param_count=modified_param_count,
            total_model_params=total_model_params_after,
            total_mlp_params=total_mlp_params_after,
            parent_nll_mean=parent_nll_mean,
            active_nll_mean=active_nll_mean,
            parent_gen_nll=parent_gen_nll,
            active_gen_nll=active_gen_nll,
            parent_spec_nll=parent_spec_nll,
            active_spec_nll=active_spec_nll,
            parent_anse=parent_anse,
            cand_anse=cand_anse,
            active_energy=active_energy,
            delta_energy=delta_energy,
            layer_reports=layer_reports,
        )
    finally:
        work_model.train(was_training)


__all__ = [
    "DEFAULT_ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_ANSE_SOFT_MARGIN_PENALTY",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CV_TOL",
    "DEFAULT_MAX_BIAS_MAGNITUDE",
    "DEFAULT_MAX_DOMAIN_DRIFT",
    "DEFAULT_MAX_ITERS",
    "DEFAULT_MAX_PARAM_MOD_FRACTION",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_SINK_MARGIN_EPS",
    "DEFAULT_STEP_SIZE",
    "DEFAULT_TOP_K",
    "GENERAL_EXPERT_PREFIX",
    "RECEIPT_OBJECT",
    "SCHEMA_VERSION",
    "RouterBiasCalibrationConfig",
    "apply_verified_router_bias_calibration_inplace",
    "calibrate_router_bias_auxiliary_free",
    "canonical_json_bytes",
    "compute_module_state_sha256",
    "compute_non_router_bias_state_sha256",
    "compute_payload_sha256",
    "measure_expert_topk_utilization",
    "verify_router_bias_receipt_provenance",
]

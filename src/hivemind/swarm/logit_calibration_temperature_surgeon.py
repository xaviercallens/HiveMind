# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Post-Surgery Final LayerNorm / RMSNorm Temperature & ECE Calibration Surgeon.

Eliminates post-surgery logit overconfidence (Expected Calibration Error, Brier, and
tail-error NLL inflation) after Sparse MoE expert sinking or LoRA merging without any
gradient training or inference-time overhead by folding optimal scalar or channelwise
temperature ``1 / tau_temp`` directly into the final ``LayerNorm`` or ``RMSNorm``
weight (and optional bias) parameters:

1. ``compute_token_nll_and_ece(logits, targets, *, num_bins=15)``:
   Computes exact CPU ``float64`` next-token NLL, equal-width Expected Calibration Error
   (ECE), Maximum Calibration Error (MCE), Shannon entropy, Brier value, and accuracy.
2. ``optimize_guard_constrained_ln_scale_and_interpolation(...)``:
   Jointly searches interpolation weight ``lambda_ln`` between base and post-surgery norm
   parameters and temperature ``tau_temp`` (plus optional analytical channelwise scale
   refinement), subject to the strict ``<= +0.5%`` general-prose NLL drift guard, and
   folds ``1 / tau_temp`` directly into the final norm parameters.
3. ``apply_verified_ln_temperature_surgery_inplace(...)``:
   Executes verified final-norm calibration on ``CrowdV1WithEngram`` / MoE models
   equipped with either ``LayerNorm`` or biasless ``RMSNorm``, committing edits if and
   only if ANSE ``Delta E < 0`` and general/worst-domain drift ``<= +1.0%``; otherwise
   restores the exact parent state atomically bit-for-bit with a canonical SHA-256
   provenance receipt.
"""

from __future__ import annotations

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
import torch.nn.functional as F

from hivemind.swarm.wsp_alphaedit_control import forward_model_pre_ln

SURGEON_SCHEMA: str = "hivemind.logit_calibration_temperature_surgeon/1"
SURGEON_RECEIPT_OBJECT: str = "hivemind.logit_calibration_temperature_surgeon_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_NUM_BINS: int = 15
DEFAULT_GENERAL_GUARD_REL: float = 0.005
DEFAULT_MAX_DOMAIN_DRIFT: float = 0.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_ECE_ENERGY_WEIGHT: float = 0.25
DEFAULT_TAU_MIN: float = 0.25
DEFAULT_TAU_MAX: float = 6.0
DEFAULT_NUM_TAU_GRID: int = 25
DEFAULT_NUM_TAU_REFINE: int = 11
DEFAULT_SCORE_FROM: int = 64
DEFAULT_BATCH_SIZE: int = 24
DEFAULT_IGNORE_INDEX: int = -100
DEFAULT_LAMBDA_GRID: tuple[float, ...] = tuple(round(0.1 * idx, 1) for idx in range(11))
VALID_NORM_TYPES: frozenset[str] = frozenset({"auto", "layernorm", "rmsnorm"})

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_CHUNK_SIZE: int = 4096


# ---------------------------------------------------------------------------
# 1. Validation, Configuration & Canonical Provenance Helpers
# ---------------------------------------------------------------------------


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_int(value: Any, name: str) -> int:
    """Validate that ``value`` is an integer (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return int(value)


def _validate_nonneg_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a non-negative integer."""
    out = _validate_int(value, name)
    if out < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return out


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a strictly positive integer."""
    out = _validate_int(value, name)
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


def _validate_1d_norm_param(tensor: Any, name: str, expected_dim: int | None = None) -> torch.Tensor:
    """Validate a non-empty 1D floating-point norm weight or bias tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 1 or tensor.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 1D tensor, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    if expected_dim is not None and int(tensor.shape[0]) != expected_dim:
        raise ValueError(f"{name} length {tensor.shape[0]} must match expected dim {expected_dim}")
    if not bool(torch.isfinite(tensor.detach()).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _validate_pre_ln_tensor(tensor: Any, name: str, hidden_dim: int) -> torch.Tensor:
    """Validate a finite floating-point pre-norm hidden tensor of shape ``(..., hidden_dim)``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim < 2 or any(dim == 0 for dim in tensor.shape):
        raise ValueError(
            f"{name} must be a non-empty tensor with ndim >= 2, got shape {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    if int(tensor.shape[-1]) != hidden_dim:
        raise ValueError(
            f"{name} last dimension {tensor.shape[-1]} must match norm dim {hidden_dim}"
        )
    t_cpu = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(t_cpu).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return t_cpu


def _validate_targets_shape(targets: Any, expected_shape: tuple[int, ...], name: str) -> torch.Tensor:
    """Validate integer target tensor matching ``expected_shape``."""
    if not isinstance(targets, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(targets).__name__}")
    if tuple(targets.shape) != expected_shape:
        raise ValueError(f"{name} shape {tuple(targets.shape)} must match {expected_shape}")
    if targets.is_floating_point() or targets.dtype == torch.bool:
        raise TypeError(f"{name} must have an integer dtype, got {targets.dtype}")
    return targets.detach().to(device="cpu", dtype=torch.long).contiguous()


def _validate_lambda_grid(lambda_grid: Sequence[float]) -> tuple[float, ...]:
    """Validate interpolation grid entries in ``[0.0, 1.0]``."""
    if isinstance(lambda_grid, (str, bytes, bytearray)) or not isinstance(lambda_grid, Sequence):
        raise TypeError("lambda_grid must be a non-empty sequence of floats")
    if len(lambda_grid) == 0:
        raise ValueError("lambda_grid must be non-empty")
    out: list[float] = []
    for idx, val in enumerate(lambda_grid):
        f_val = _validate_nonneg_finite(val, f"lambda_grid[{idx}]")
        if f_val > 1.0:
            raise ValueError(f"lambda_grid[{idx}] must lie in [0.0, 1.0], got {f_val}")
        out.append(f_val)
    return tuple(dict.fromkeys(out))


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically with strict finite-number enforcement."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def compute_payload_sha256(payload: Any) -> str:
    """Return lowercase SHA-256 hex digest of ``canonical_json_bytes(payload)``."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_module_state_sha256(module: nn.Module) -> str:
    """Compute a deterministic bitwise SHA-256 digest over ``module.state_dict()``."""
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


def verify_calibration_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify canonical ``receipt_sha256`` provenance on a calibration receipt."""
    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a mapping")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != SURGEON_RECEIPT_OBJECT or receipt.get("schema") != SURGEON_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


@dataclass(frozen=True)
class LogitCalibrationConfig:
    """Validated configuration for guard-constrained norm temperature calibration."""

    num_bins: int = DEFAULT_NUM_BINS
    general_guard_rel: float = DEFAULT_GENERAL_GUARD_REL
    max_domain_drift: float = DEFAULT_MAX_DOMAIN_DRIFT
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    ece_energy_weight: float = DEFAULT_ECE_ENERGY_WEIGHT
    tau_min: float = DEFAULT_TAU_MIN
    tau_max: float = DEFAULT_TAU_MAX
    num_tau_grid: int = DEFAULT_NUM_TAU_GRID
    num_tau_refine: int = DEFAULT_NUM_TAU_REFINE
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID
    channelwise: bool = False
    channelwise_steps: int = 12
    channelwise_lr: float = 0.15
    channelwise_reg: float = 0.05
    channelwise_min_rel: float = 0.5
    channelwise_max_rel: float = 2.0
    norm_type: str = "auto"
    eps: float | None = None
    ignore_index: int = DEFAULT_IGNORE_INDEX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    general_prefix: str = GENERAL_EXPERT_PREFIX
    require_strict_energy_drop: bool = True

    def __post_init__(self) -> None:
        _validate_positive_int(self.num_bins, "num_bins")
        _validate_nonneg_finite(self.general_guard_rel, "general_guard_rel")
        _validate_nonneg_finite(self.max_domain_drift, "max_domain_drift")
        _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        _validate_nonneg_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_nonneg_finite(self.ece_energy_weight, "ece_energy_weight")
        t_min = _validate_positive_finite(self.tau_min, "tau_min")
        t_max = _validate_positive_finite(self.tau_max, "tau_max")
        if t_min > t_max:
            raise ValueError(f"tau_min ({t_min}) must be <= tau_max ({t_max})")
        _validate_positive_int(self.num_tau_grid, "num_tau_grid")
        _validate_positive_int(self.num_tau_refine, "num_tau_refine")
        object.__setattr__(self, "lambda_grid", _validate_lambda_grid(self.lambda_grid))
        _validate_bool(self.channelwise, "channelwise")
        _validate_positive_int(self.channelwise_steps, "channelwise_steps")
        _validate_positive_finite(self.channelwise_lr, "channelwise_lr")
        _validate_nonneg_finite(self.channelwise_reg, "channelwise_reg")
        c_min = _validate_positive_finite(self.channelwise_min_rel, "channelwise_min_rel")
        c_max = _validate_positive_finite(self.channelwise_max_rel, "channelwise_max_rel")
        if c_min > 1.0 or c_max < 1.0 or c_min > c_max:
            raise ValueError(
                f"channelwise_min_rel ({c_min}) <= 1.0 <= channelwise_max_rel ({c_max}) required"
            )
        if not isinstance(self.norm_type, str) or self.norm_type not in VALID_NORM_TYPES:
            raise ValueError(
                f"norm_type must be one of {sorted(VALID_NORM_TYPES)}, got {self.norm_type!r}"
            )
        if self.eps is not None:
            _validate_positive_finite(self.eps, "eps")
        _validate_int(self.ignore_index, "ignore_index")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_bool(self.require_strict_energy_drop, "require_strict_energy_drop")


# ---------------------------------------------------------------------------
# 2. Norm Inspection, Application & Zero-Overhead Temperature Folding
# ---------------------------------------------------------------------------


def detect_norm_type(norm_module: nn.Module, *, norm_type: str = "auto") -> str:
    """Classify a normalization module as ``'layernorm'`` or ``'rmsnorm'``."""
    if not isinstance(norm_module, nn.Module):
        raise TypeError(f"norm_module must be an nn.Module, got {type(norm_module).__name__}")
    if not isinstance(norm_type, str) or norm_type not in VALID_NORM_TYPES:
        raise ValueError(f"norm_type must be one of {sorted(VALID_NORM_TYPES)}, got {norm_type!r}")
    if norm_type != "auto":
        if norm_type == "rmsnorm" and getattr(norm_module, "bias", None) is not None:
            raise ValueError("norm_type='rmsnorm' is incompatible with a norm module having a non-None bias")
        return norm_type

    if isinstance(norm_module, nn.RMSNorm) or "rms" in type(norm_module).__name__.lower():
        if getattr(norm_module, "bias", None) is not None:
            raise ValueError("RMSNorm module must not expose a non-None bias")
        return "rmsnorm"
    if isinstance(norm_module, nn.LayerNorm):
        return "layernorm"
    if hasattr(norm_module, "variance_epsilon") and getattr(norm_module, "bias", None) is None:
        return "rmsnorm"
    if hasattr(norm_module, "bias") or hasattr(norm_module, "normalized_shape"):
        return "layernorm"
    raise ValueError(
        f"Cannot infer norm type for {type(norm_module).__name__}; pass explicit norm_type='layernorm' or 'rmsnorm'"
    )


def _resolve_norm_type_from_params(bias: torch.Tensor | None, norm_type: str) -> str:
    """Resolve norm kind from parameter signature and optional explicit ``norm_type``."""
    if not isinstance(norm_type, str) or norm_type not in VALID_NORM_TYPES:
        raise ValueError(f"norm_type must be one of {sorted(VALID_NORM_TYPES)}, got {norm_type!r}")
    if norm_type == "rmsnorm":
        if bias is not None:
            raise ValueError("RMSNorm does not accept non-None bias tensors")
        return "rmsnorm"
    if norm_type == "layernorm":
        return "layernorm"
    return "rmsnorm" if bias is None else "layernorm"


def apply_norm_weights(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    eps: float | None = None,
    norm_type: str = "auto",
) -> torch.Tensor:
    """Apply LayerNorm or biasless RMSNorm with arbitrary weight/bias tensors."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if x.ndim < 1 or any(dim == 0 for dim in x.shape):
        raise ValueError(f"x must be non-empty, got shape {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError(f"x must have a floating-point dtype, got {x.dtype}")
    dim = int(x.shape[-1])
    w = _validate_1d_norm_param(weight, "weight", expected_dim=dim)
    b = _validate_1d_norm_param(bias, "bias", expected_dim=dim) if bias is not None else None
    resolved = _resolve_norm_type_from_params(b, norm_type)

    if eps is not None:
        eps_val: float | None = _validate_positive_finite(eps, "eps")
    else:
        eps_val = None if resolved == "rmsnorm" else 1e-5

    w_cast = w.to(device=x.device, dtype=x.dtype)
    if resolved == "rmsnorm":
        return F.rms_norm(x, (dim,), w_cast, eps_val)
    b_cast = b.to(device=x.device, dtype=x.dtype) if b is not None else None
    return F.layer_norm(x, (dim,), w_cast, b_cast, eps=float(eps_val))


def fold_temperature_into_norm_params(
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor | None,
    tau_temp: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fold scalar or channelwise temperature ``1 / tau_temp`` into norm weight and bias."""
    w = _validate_1d_norm_param(ln_weight, "ln_weight")
    dim = int(w.shape[0])
    b = _validate_1d_norm_param(ln_bias, "ln_bias", expected_dim=dim) if ln_bias is not None else None

    if isinstance(tau_temp, torch.Tensor):
        tau_t = _validate_1d_norm_param(tau_temp, "tau_temp", expected_dim=dim).detach().to(
            device="cpu", dtype=torch.float64
        )
        if not bool((tau_t > 0.0).all().item()):
            raise ValueError("tau_temp tensor must contain strictly positive temperatures")
    else:
        tau_f = _validate_positive_finite(tau_temp, "tau_temp")
        tau_t = torch.tensor(tau_f, dtype=torch.float64, device="cpu")

    w64 = w.detach().to(device="cpu", dtype=torch.float64)
    w_folded = (w64 / tau_t).to(device=w.device, dtype=w.dtype)
    if not bool(torch.isfinite(w_folded).all().item()):
        raise ValueError("Folded norm weight contains non-finite values after temperature scaling")

    if b is None:
        return w_folded, None
    b64 = b.detach().to(device="cpu", dtype=torch.float64)
    b_folded = (b64 / tau_t).to(device=b.device, dtype=b.dtype)
    if not bool(torch.isfinite(b_folded).all().item()):
        raise ValueError("Folded norm bias contains non-finite values after temperature scaling")
    return w_folded, b_folded


# ---------------------------------------------------------------------------
# 3. Exact Next-Token NLL, ECE, MCE, Brier & Shannon Entropy Metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_token_nll_and_ece(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_bins: int = DEFAULT_NUM_BINS,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> dict[str, Any]:
    """Compute exact next-token NLL, ECE, MCE, Brier value, and Shannon entropy in ``float64``."""
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits).__name__}")
    if logits.ndim < 2 or any(dim == 0 for dim in logits.shape):
        raise ValueError(
            f"logits must be a non-empty tensor with ndim >= 2, got shape {tuple(logits.shape)}"
        )
    if not logits.is_floating_point():
        raise TypeError(f"logits must have a floating-point dtype, got {logits.dtype}")

    bins_i = _validate_positive_int(num_bins, "num_bins")
    ignore_i = _validate_int(ignore_index, "ignore_index")
    targets_t = _validate_targets_shape(targets, tuple(logits.shape[:-1]), "targets")

    vocab_size = int(logits.shape[-1])
    flat_logits = logits.detach().reshape(-1, vocab_size)
    flat_targets = targets_t.reshape(-1)
    valid_mask = flat_targets != ignore_i
    num_tokens = int(valid_mask.sum().item())
    if num_tokens == 0:
        raise ValueError("targets must contain at least one non-ignored token")

    logits_valid = flat_logits[valid_mask.to(flat_logits.device)]
    targets_valid = flat_targets[valid_mask]

    if bool((targets_valid < 0).any().item()) or bool((targets_valid >= vocab_size).any().item()):
        raise ValueError(f"targets token IDs must lie in [0, {vocab_size}) or equal ignore_index={ignore_i}")

    nll_parts: list[torch.Tensor] = []
    ent_parts: list[torch.Tensor] = []
    brier_parts: list[torch.Tensor] = []
    conf_parts: list[torch.Tensor] = []
    acc_parts: list[torch.Tensor] = []

    for start in range(0, num_tokens, _TOKEN_CHUNK_SIZE):
        end = min(num_tokens, start + _TOKEN_CHUNK_SIZE)
        chunk_logits = logits_valid[start:end].to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(chunk_logits).all().item()):
            raise ValueError("logits must contain only finite values")
        chunk_targets = targets_valid[start:end]

        log_probs = F.log_softmax(chunk_logits, dim=-1)
        probs = log_probs.exp()
        token_nll = F.nll_loss(log_probs, chunk_targets, reduction="none")
        # Guard 0 * (-inf) at underflowed probabilities without modifying non-zero terms
        neg_p_log_p = torch.where(probs > 0.0, -probs * log_probs, torch.zeros_like(probs))
        token_entropy = neg_p_log_p.sum(dim=-1)

        target_probs = probs.gather(1, chunk_targets.unsqueeze(1)).squeeze(1)
        token_brier = probs.pow(2).sum(dim=-1) - 2.0 * target_probs + 1.0

        conf, preds = probs.max(dim=-1)
        acc = (preds == chunk_targets).to(torch.float64)

        nll_parts.append(token_nll)
        ent_parts.append(token_entropy)
        brier_parts.append(token_brier)
        conf_parts.append(conf)
        acc_parts.append(acc)

    all_nll = torch.cat(nll_parts, dim=0)
    all_ent = torch.cat(ent_parts, dim=0)
    all_brier = torch.cat(brier_parts, dim=0)
    all_conf = torch.cat(conf_parts, dim=0)
    all_acc = torch.cat(acc_parts, dim=0)

    nll_mean = float(all_nll.mean().item())
    entropy_mean = float(all_ent.mean().item())
    brier_value = float(all_brier.mean().item())
    accuracy = float(all_acc.mean().item())
    mean_confidence = float(all_conf.mean().item())
    overconfidence_gap = mean_confidence - accuracy

    boundaries = torch.linspace(0.0, 1.0, steps=bins_i + 1, dtype=torch.float64)
    bin_ids = torch.bucketize(all_conf, boundaries[1:-1], right=False)
    counts = torch.bincount(bin_ids, minlength=bins_i)
    sum_acc = torch.bincount(bin_ids, weights=all_acc, minlength=bins_i)
    sum_conf = torch.bincount(bin_ids, weights=all_conf, minlength=bins_i)

    nz_mask = counts > 0
    counts_f = counts.to(torch.float64)
    bin_accs = torch.where(nz_mask, sum_acc / counts_f.clamp_min(1.0), torch.zeros_like(counts_f))
    bin_confs = torch.where(nz_mask, sum_conf / counts_f.clamp_min(1.0), torch.zeros_like(counts_f))
    abs_gaps = torch.where(nz_mask, (bin_accs - bin_confs).abs(), torch.zeros_like(counts_f))

    ece_value = float(((counts_f / float(num_tokens)) * abs_gaps).sum().item())
    mce_value = float(abs_gaps.max().item())

    bounds_list = boundaries.tolist()
    counts_list = counts.tolist()
    accs_list = bin_accs.tolist()
    confs_list = bin_confs.tolist()
    gaps_list = abs_gaps.tolist()
    bin_rows = [
        {
            "bin_index": b_idx,
            "lower": float(bounds_list[b_idx]),
            "upper": float(bounds_list[b_idx + 1]),
            "count": int(counts_list[b_idx]),
            "accuracy": float(accs_list[b_idx]),
            "confidence": float(confs_list[b_idx]),
            "abs_gap": float(gaps_list[b_idx]),
        }
        for b_idx in range(bins_i)
    ]

    return {
        "nll_mean": nll_mean,
        "ece_value": float(ece_value),
        "mce_value": float(mce_value),
        "entropy_mean": entropy_mean,
        "brier_value": brier_value,
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "overconfidence_gap": overconfidence_gap,
        "perplexity": float(math.exp(min(nll_mean, 700.0))),
        "num_tokens": num_tokens,
        "num_bins": bins_i,
        "bins": bin_rows,
    }


# ---------------------------------------------------------------------------
# 4. Joint Guard-Constrained (lambda_ln, tau_temp) Search & Channelwise Fold
# ---------------------------------------------------------------------------


def _eval_unembed_logits(
    unembed_fn: Callable[[torch.Tensor], torch.Tensor],
    h_norm: torch.Tensor,
) -> torch.Tensor:
    """Invoke ``unembed_fn`` with weight-dtype alignment and validate finite logits."""
    weight_attr = getattr(unembed_fn, "weight", None)
    if isinstance(unembed_fn, nn.Module) and isinstance(weight_attr, torch.Tensor):
        h_in = h_norm.to(device=weight_attr.device, dtype=weight_attr.dtype)
    else:
        h_in = h_norm
    logits = unembed_fn(h_in)
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"unembed_fn must return a torch.Tensor, got {type(logits).__name__}")
    if logits.shape[:-1] != h_norm.shape[:-1] or logits.ndim < 2 or logits.shape[-1] == 0:
        raise ValueError(
            f"unembed_fn returned invalid logits shape {tuple(logits.shape)} for hidden shape {tuple(h_norm.shape)}"
        )
    logits_cpu = logits.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(logits_cpu).all().item()):
        raise ValueError("unembed_fn produced non-finite logits")
    return logits_cpu


def _build_tau_candidates(tau_min: float, tau_max: float, num_points: int) -> list[float]:
    """Generate sorted deduplicated log-spaced scalar temperatures including ``1.0`` when bracketed."""
    grid = np.geomspace(tau_min, tau_max, num=max(1, num_points)).tolist()
    if tau_min <= 1.0 <= tau_max:
        grid.append(1.0)
    unique = sorted({round(float(t), 8) for t in grid})
    return unique


def _score_tau_candidate(
    z_p: torch.Tensor,
    z_s: torch.Tensor,
    p_targets: torch.Tensor,
    s_targets: torch.Tensor,
    tau_val: float,
    *,
    ref_base_prose_nll: float,
    prose_guard_ceiling: float,
    ece_w: float,
    bins_i: int,
    ignore_i: int,
) -> tuple[bool, float, float, dict[str, Any], dict[str, Any]]:
    """Score one scalar temperature candidate against calibration objective and general guard."""
    mp = compute_token_nll_and_ece(z_p / tau_val, p_targets, num_bins=bins_i, ignore_index=ignore_i)
    ms = compute_token_nll_and_ece(z_s / tau_val, s_targets, num_bins=bins_i, ignore_index=ignore_i)
    p_nll = mp["nll_mean"]
    s_nll = ms["nll_mean"]
    mean_ece = 0.5 * (mp["ece_value"] + ms["ece_value"])
    obj = 0.5 * (p_nll + s_nll) + ece_w * mean_ece
    ratio = p_nll / max(ref_base_prose_nll, 1e-12)
    ok = p_nll <= prose_guard_ceiling + 1e-12
    return ok, obj, ratio, mp, ms


def _refine_channelwise_scales(
    h_prose: torch.Tensor,
    targets_prose: torch.Tensor,
    h_spec: torch.Tensor,
    targets_spec: torch.Tensor,
    unembed_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    prose_guard_ceiling: float,
    ece_weight: float,
    num_bins: int,
    ignore_index: int,
    steps: int,
    lr: float,
    reg: float,
    min_rel: float,
    max_rel: float,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]] | None:
    """Optimize per-channel positive scale multipliers ``c`` in ``[min_rel, max_rel]`` without autograd."""
    weight_attr = getattr(unembed_fn, "weight", None)
    if isinstance(unembed_fn, nn.Module) and isinstance(weight_attr, torch.Tensor) and weight_attr.ndim == 2:
        w_u = weight_attr.detach().to(device="cpu", dtype=torch.float64)
    else:
        return None

    hp_2d = h_prose.reshape(-1, int(h_prose.shape[-1])).to(device="cpu", dtype=torch.float64)
    hs_2d = h_spec.reshape(-1, int(h_spec.shape[-1])).to(device="cpu", dtype=torch.float64)
    tp_1d = targets_prose.reshape(-1)
    ts_1d = targets_spec.reshape(-1)
    mp = tp_1d != ignore_index
    ms = ts_1d != ignore_index
    hp_v = hp_2d[mp]
    tp_v = tp_1d[mp]
    hs_v = hs_2d[ms]
    ts_v = ts_1d[ms]

    dim = int(hp_v.shape[1])
    c_vec = torch.ones(dim, dtype=torch.float64)

    m_p0 = compute_token_nll_and_ece(
        _eval_unembed_logits(unembed_fn, h_prose), targets_prose, num_bins=num_bins, ignore_index=ignore_index
    )
    m_s0 = compute_token_nll_and_ece(
        _eval_unembed_logits(unembed_fn, h_spec), targets_spec, num_bins=num_bins, ignore_index=ignore_index
    )
    best_obj = (
        0.5 * (m_p0["nll_mean"] + m_s0["nll_mean"]) + ece_weight * 0.5 * (m_p0["ece_value"] + m_s0["ece_value"])
    )
    best_c = c_vec.clone()
    best_pair = (m_p0, m_s0)

    n_p = float(hp_v.shape[0])
    n_s = float(hs_v.shape[0])
    log_c = torch.zeros(dim, dtype=torch.float64)

    for _ in range(steps):
        c_cur = log_c.exp().clamp(min_rel, max_rel)
        z_p = (hp_v * c_cur.unsqueeze(0)) @ w_u.T
        z_s = (hs_v * c_cur.unsqueeze(0)) @ w_u.T
        probs_p = F.softmax(z_p, dim=-1)
        probs_s = F.softmax(z_s, dim=-1)

        probs_p[torch.arange(hp_v.shape[0]), tp_v] -= 1.0
        probs_s[torch.arange(hs_v.shape[0]), ts_v] -= 1.0

        grad_hp = (probs_p @ w_u) * hp_v
        grad_hs = (probs_s @ w_u) * hs_v
        grad_c = 0.5 * (grad_hp.sum(dim=0) / n_p + grad_hs.sum(dim=0) / n_s) + reg * (c_cur - 1.0) / float(dim)

        grad_log_c = grad_c * c_cur
        grad_norm = float(grad_log_c.norm().item())
        if grad_norm > 1e-12:
            step_vec = lr * (grad_log_c / max(1.0, grad_norm))
            log_c = (log_c - step_vec).clamp(math.log(min_rel), math.log(max_rel))
        c_cand = log_c.exp().clamp(min_rel, max_rel)

        mp_cand = compute_token_nll_and_ece(
            _eval_unembed_logits(unembed_fn, h_prose * c_cand),
            targets_prose,
            num_bins=num_bins,
            ignore_index=ignore_index,
        )
        ms_cand = compute_token_nll_and_ece(
            _eval_unembed_logits(unembed_fn, h_spec * c_cand),
            targets_spec,
            num_bins=num_bins,
            ignore_index=ignore_index,
        )
        if mp_cand["nll_mean"] <= prose_guard_ceiling + 1e-12:
            obj_cand = (
                0.5 * (mp_cand["nll_mean"] + ms_cand["nll_mean"])
                + ece_weight * 0.5 * (mp_cand["ece_value"] + ms_cand["ece_value"])
            )
            if obj_cand < best_obj - 1e-9:
                best_obj = obj_cand
                best_c = c_cand.clone()
                best_pair = (mp_cand, ms_cand)

    return best_c, best_pair[0], best_pair[1]


@torch.no_grad()
def optimize_guard_constrained_ln_scale_and_interpolation(
    base_ln_w: torch.Tensor,
    base_ln_b: torch.Tensor | None,
    moe_ln_w: torch.Tensor,
    moe_ln_b: torch.Tensor | None,
    pre_ln_prose: torch.Tensor,
    targets_prose: torch.Tensor,
    pre_ln_spec: torch.Tensor,
    targets_spec: torch.Tensor,
    unembed_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    general_guard_rel: float = DEFAULT_GENERAL_GUARD_REL,
    eps: float | None = None,
    norm_type: str = "auto",
    num_bins: int = DEFAULT_NUM_BINS,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    lambda_grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
    tau_min: float = DEFAULT_TAU_MIN,
    tau_max: float = DEFAULT_TAU_MAX,
    num_tau_grid: int = DEFAULT_NUM_TAU_GRID,
    num_tau_refine: int = DEFAULT_NUM_TAU_REFINE,
    ece_weight: float = DEFAULT_ECE_ENERGY_WEIGHT,
    channelwise: bool = False,
    channelwise_steps: int = 12,
    channelwise_lr: float = 0.15,
    channelwise_reg: float = 0.05,
    channelwise_min_rel: float = 0.5,
    channelwise_max_rel: float = 2.0,
    base_prose_nll_override: float | None = None,
) -> dict[str, Any]:
    """Jointly optimize ``(lambda_ln, tau_temp)`` under general guard and fold ``1/tau_temp`` into norm weights."""
    if not callable(unembed_fn):
        raise TypeError(f"unembed_fn must be callable, got {type(unembed_fn).__name__}")

    base_w = _validate_1d_norm_param(base_ln_w, "base_ln_w")
    dim = int(base_w.shape[0])
    moe_w = _validate_1d_norm_param(moe_ln_w, "moe_ln_w", expected_dim=dim)

    if (base_ln_b is None) != (moe_ln_b is None):
        raise ValueError("base_ln_b and moe_ln_b must either both be provided or both be None")
    base_b = _validate_1d_norm_param(base_ln_b, "base_ln_b", expected_dim=dim) if base_ln_b is not None else None
    moe_b = _validate_1d_norm_param(moe_ln_b, "moe_ln_b", expected_dim=dim) if moe_ln_b is not None else None

    resolved_norm = _resolve_norm_type_from_params(base_b, norm_type)
    guard_rel = _validate_nonneg_finite(general_guard_rel, "general_guard_rel")
    bins_i = _validate_positive_int(num_bins, "num_bins")
    ignore_i = _validate_int(ignore_index, "ignore_index")
    lam_grid = _validate_lambda_grid(lambda_grid)
    t_min = _validate_positive_finite(tau_min, "tau_min")
    t_max = _validate_positive_finite(tau_max, "tau_max")
    if t_min > t_max:
        raise ValueError(f"tau_min ({t_min}) must be <= tau_max ({t_max})")
    n_tau_grid = _validate_positive_int(num_tau_grid, "num_tau_grid")
    n_tau_ref = _validate_positive_int(num_tau_refine, "num_tau_refine")
    ece_w = _validate_nonneg_finite(ece_weight, "ece_weight")
    use_channelwise = _validate_bool(channelwise, "channelwise")

    p_hidden = _validate_pre_ln_tensor(pre_ln_prose, "pre_ln_prose", dim)
    p_targets = _validate_targets_shape(targets_prose, tuple(p_hidden.shape[:-1]), "targets_prose")
    s_hidden = _validate_pre_ln_tensor(pre_ln_spec, "pre_ln_spec", dim)
    s_targets = _validate_targets_shape(targets_spec, tuple(s_hidden.shape[:-1]), "targets_spec")

    def _eval_split_pair(w: torch.Tensor, b: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        hp = apply_norm_weights(p_hidden, w, b, eps=eps, norm_type=resolved_norm).to(dtype=pre_ln_prose.dtype)
        hs = apply_norm_weights(s_hidden, w, b, eps=eps, norm_type=resolved_norm).to(dtype=pre_ln_spec.dtype)
        return _eval_unembed_logits(unembed_fn, hp), _eval_unembed_logits(unembed_fn, hs)

    base_p_logits, base_s_logits = _eval_split_pair(base_w, base_b)
    base_p_m = compute_token_nll_and_ece(base_p_logits, p_targets, num_bins=bins_i, ignore_index=ignore_i)
    base_s_m = compute_token_nll_and_ece(base_s_logits, s_targets, num_bins=bins_i, ignore_index=ignore_i)

    moe_p_logits, moe_s_logits = _eval_split_pair(moe_w, moe_b)
    moe_p_m = compute_token_nll_and_ece(moe_p_logits, p_targets, num_bins=bins_i, ignore_index=ignore_i)
    moe_s_m = compute_token_nll_and_ece(moe_s_logits, s_targets, num_bins=bins_i, ignore_index=ignore_i)

    ref_base_prose_nll = (
        base_p_m["nll_mean"]
        if base_prose_nll_override is None
        else _validate_positive_finite(base_prose_nll_override, "base_prose_nll_override")
    )
    prose_guard_ceiling = (1.0 + guard_rel) * ref_base_prose_nll

    tau_coarse = _build_tau_candidates(t_min, t_max, n_tau_grid)
    feasible_cands: list[tuple[float, float, float, float]] = []
    fallback_cands: list[tuple[float, float, float, float]] = []
    grid_summary: list[dict[str, Any]] = []

    base_w64 = base_w.detach().to(device="cpu", dtype=torch.float64)
    moe_w64 = moe_w.detach().to(device="cpu", dtype=torch.float64)
    base_b64 = base_b.detach().to(device="cpu", dtype=torch.float64) if base_b is not None else None
    moe_b64 = moe_b.detach().to(device="cpu", dtype=torch.float64) if moe_b is not None else None

    for lam in lam_grid:
        w_lam = (base_w64 + lam * (moe_w64 - base_w64)).to(device=base_w.device, dtype=base_w.dtype)
        b_lam = (
            (base_b64 + lam * (moe_b64 - base_b64)).to(device=base_b.device, dtype=base_b.dtype)
            if (base_b64 is not None and moe_b64 is not None and base_b is not None)
            else None
        )
        z_p, z_s = _eval_split_pair(w_lam, b_lam)

        lam_feasible: list[tuple[float, float, dict[str, Any], dict[str, Any]]] = []
        lam_fallback: list[tuple[float, float, float, dict[str, Any], dict[str, Any]]] = []

        for tau_val in tau_coarse:
            ok, obj, ratio, mp, ms = _score_tau_candidate(
                z_p,
                z_s,
                p_targets,
                s_targets,
                tau_val,
                ref_base_prose_nll=ref_base_prose_nll,
                prose_guard_ceiling=prose_guard_ceiling,
                ece_w=ece_w,
                bins_i=bins_i,
                ignore_i=ignore_i,
            )
            lam_fallback.append((ratio, obj, tau_val, mp, ms))
            if ok:
                lam_feasible.append((obj, tau_val, mp, ms))

        # Local geometric refinement around best coarse tau for this lambda
        anchor_tau = (
            min(lam_feasible, key=lambda item: (item[0], abs(item[1] - 1.0)))[1]
            if lam_feasible
            else min(lam_fallback, key=lambda item: (item[0], item[1]))[2]
        )
        anchor_idx = tau_coarse.index(anchor_tau)
        lo_tau = tau_coarse[max(0, anchor_idx - 1)]
        hi_tau = tau_coarse[min(len(tau_coarse) - 1, anchor_idx + 1)]
        if hi_tau > lo_tau and n_tau_ref > 1:
            for tau_ref in _build_tau_candidates(lo_tau, hi_tau, n_tau_ref):
                if tau_ref in tau_coarse:
                    continue
                ok, obj, ratio, mp, ms = _score_tau_candidate(
                    z_p,
                    z_s,
                    p_targets,
                    s_targets,
                    tau_ref,
                    ref_base_prose_nll=ref_base_prose_nll,
                    prose_guard_ceiling=prose_guard_ceiling,
                    ece_w=ece_w,
                    bins_i=bins_i,
                    ignore_i=ignore_i,
                )
                lam_fallback.append((ratio, obj, tau_ref, mp, ms))
                if ok:
                    lam_feasible.append((obj, tau_ref, mp, ms))

        if lam_feasible:
            best_obj_l, best_tau_l, best_mp_l, best_ms_l = min(
                lam_feasible, key=lambda item: (item[0], abs(item[1] - 1.0))
            )
            best_ratio_l = best_mp_l["nll_mean"] / max(ref_base_prose_nll, 1e-12)
            feasible_cands.append((best_obj_l, best_ratio_l, lam, best_tau_l))
            grid_summary.append(
                {
                    "lambda_ln": lam,
                    "tau_temp": best_tau_l,
                    "guard_feasible": True,
                    "prose_nll": best_mp_l["nll_mean"],
                    "spec_nll": best_ms_l["nll_mean"],
                    "ece_value": 0.5 * (best_mp_l["ece_value"] + best_ms_l["ece_value"]),
                    "objective": best_obj_l,
                }
            )
        else:
            best_ratio_l, best_obj_l, best_tau_l, best_mp_l, best_ms_l = min(
                lam_fallback, key=lambda item: (item[0], item[1], abs(item[2] - 1.0))
            )
            fallback_cands.append((best_ratio_l, best_obj_l, lam, best_tau_l))
            grid_summary.append(
                {
                    "lambda_ln": lam,
                    "tau_temp": best_tau_l,
                    "guard_feasible": False,
                    "prose_nll": best_mp_l["nll_mean"],
                    "spec_nll": best_ms_l["nll_mean"],
                    "ece_value": 0.5 * (best_mp_l["ece_value"] + best_ms_l["ece_value"]),
                    "objective": best_obj_l,
                }
            )

    if feasible_cands:
        _, _, best_lam, best_tau = min(feasible_cands, key=lambda item: (item[0], item[1], abs(item[3] - 1.0)))
    else:
        _, _, best_lam, best_tau = min(fallback_cands, key=lambda item: (item[0], item[1], abs(item[3] - 1.0)))

    w_interp = (base_w64 + best_lam * (moe_w64 - base_w64)).to(device=base_w.device, dtype=base_w.dtype)
    b_interp = (
        (base_b64 + best_lam * (moe_b64 - base_b64)).to(device=base_b.device, dtype=base_b.dtype)
        if (base_b64 is not None and moe_b64 is not None and base_b is not None)
        else None
    )

    folded_w, folded_b = fold_temperature_into_norm_params(w_interp, b_interp, best_tau)
    channel_tau: list[float] | None = None

    if use_channelwise:
        h_p_cur = apply_norm_weights(p_hidden, folded_w, folded_b, eps=eps, norm_type=resolved_norm).to(
            dtype=pre_ln_prose.dtype
        )
        h_s_cur = apply_norm_weights(s_hidden, folded_w, folded_b, eps=eps, norm_type=resolved_norm).to(
            dtype=pre_ln_spec.dtype
        )
        ref_res = _refine_channelwise_scales(
            h_p_cur,
            p_targets,
            h_s_cur,
            s_targets,
            unembed_fn,
            prose_guard_ceiling=prose_guard_ceiling,
            ece_weight=ece_w,
            num_bins=bins_i,
            ignore_index=ignore_i,
            steps=_validate_positive_int(channelwise_steps, "channelwise_steps"),
            lr=_validate_positive_finite(channelwise_lr, "channelwise_lr"),
            reg=_validate_nonneg_finite(channelwise_reg, "channelwise_reg"),
            min_rel=_validate_positive_finite(channelwise_min_rel, "channelwise_min_rel"),
            max_rel=_validate_positive_finite(channelwise_max_rel, "channelwise_max_rel"),
        )
        if ref_res is not None:
            c_best, _, _ = ref_res
            tau_vec = best_tau / c_best
            folded_w, folded_b = fold_temperature_into_norm_params(w_interp, b_interp, tau_vec)
            channel_tau = [float(v) for v in tau_vec.tolist()]

    # Always ground-truth verify final folded weights through the norm + unembedding path
    cal_p_logits, cal_s_logits = _eval_split_pair(folded_w, folded_b)
    cal_p_m = compute_token_nll_and_ece(cal_p_logits, p_targets, num_bins=bins_i, ignore_index=ignore_i)
    cal_s_m = compute_token_nll_and_ece(cal_s_logits, s_targets, num_bins=bins_i, ignore_index=ignore_i)

    guard_feasible = bool(cal_p_m["nll_mean"] <= prose_guard_ceiling + 1e-12)
    prose_drift_rel = (cal_p_m["nll_mean"] / max(ref_base_prose_nll, 1e-12)) - 1.0
    moe_ece_mean = 0.5 * (moe_p_m["ece_value"] + moe_s_m["ece_value"])
    cal_ece_mean = 0.5 * (cal_p_m["ece_value"] + cal_s_m["ece_value"])
    ece_reduction_rel = (moe_ece_mean - cal_ece_mean) / moe_ece_mean if moe_ece_mean > 1e-12 else 0.0

    return {
        "lambda_ln": float(best_lam),
        "tau_temp": float(best_tau),
        "channel_tau": channel_tau,
        "folded_ln_w": folded_w,
        "folded_ln_b": folded_b,
        "norm_type": resolved_norm,
        "guard_feasible": guard_feasible,
        "general_guard_rel": float(guard_rel),
        "base_prose_nll": float(base_p_m["nll_mean"]),
        "base_prose_ece": float(base_p_m["ece_value"]),
        "base_spec_nll": float(base_s_m["nll_mean"]),
        "base_spec_ece": float(base_s_m["ece_value"]),
        "moe_prose_nll": float(moe_p_m["nll_mean"]),
        "moe_prose_ece": float(moe_p_m["ece_value"]),
        "moe_spec_nll": float(moe_s_m["nll_mean"]),
        "moe_spec_ece": float(moe_s_m["ece_value"]),
        "cal_prose_nll": float(cal_p_m["nll_mean"]),
        "cal_prose_ece": float(cal_p_m["ece_value"]),
        "cal_prose_entropy": float(cal_p_m["entropy_mean"]),
        "cal_spec_nll": float(cal_s_m["nll_mean"]),
        "cal_spec_ece": float(cal_s_m["ece_value"]),
        "cal_spec_entropy": float(cal_s_m["entropy_mean"]),
        "nll_mean": float(0.5 * (cal_p_m["nll_mean"] + cal_s_m["nll_mean"])),
        "ece_value": float(cal_ece_mean),
        "ece_reduction_rel": float(ece_reduction_rel),
        "prose_drift_rel": float(prose_drift_rel),
        "prose_drift_pct": float(100.0 * prose_drift_rel),
        "spec_gain_nats": float(base_s_m["nll_mean"] - cal_s_m["nll_mean"]),
        "cal_prose_metrics": cal_p_m,
        "cal_spec_metrics": cal_s_m,
        "search_grid_summary": grid_summary,
    }


# ---------------------------------------------------------------------------
# 5. Verified In-Place Norm Temperature Surgery & Atomic Rollback Gate
# ---------------------------------------------------------------------------


def _validate_model_and_norm(
    model: nn.Module,
    *,
    norm_attr: str,
    head_attr: str,
    norm_type: str,
) -> tuple[nn.Module, nn.Module, str, int, int, float | None]:
    """Validate backbone, final norm module, and unembedding head on ``model``."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("model must expose a non-empty 'blocks' sequence")
    for attr in ("wte", "cos_cached", "sin_cached"):
        if not hasattr(model, attr):
            raise ValueError(f"model must expose '{attr}'")

    norm_mod = getattr(model, norm_attr, None)
    if not isinstance(norm_mod, nn.Module):
        raise ValueError(f"model.{norm_attr} must be an nn.Module")
    w_param = getattr(norm_mod, "weight", None)
    w_valid = _validate_1d_norm_param(w_param, f"model.{norm_attr}.weight")
    hidden_dim = int(w_valid.shape[0])
    resolved_norm = detect_norm_type(norm_mod, norm_type=norm_type)
    b_param = getattr(norm_mod, "bias", None)
    if b_param is not None:
        _validate_1d_norm_param(b_param, f"model.{norm_attr}.bias", expected_dim=hidden_dim)

    raw_eps = getattr(norm_mod, "eps", getattr(norm_mod, "variance_epsilon", None))
    eps_mod = _validate_positive_finite(raw_eps, f"model.{norm_attr}.eps") if raw_eps is not None else None

    head_mod = getattr(model, head_attr, None)
    if head_mod is None or not callable(head_mod):
        raise ValueError(f"model.{head_attr} must be a callable unembedding module")
    head_weight = getattr(head_mod, "weight", None)
    if not isinstance(head_weight, torch.Tensor) or head_weight.ndim != 2 or int(head_weight.shape[1]) != hidden_dim:
        raise ValueError(
            f"model.{head_attr}.weight must be a 2D tensor with width {hidden_dim}"
        )
    vocab_size = int(head_weight.shape[0])
    return norm_mod, head_mod, resolved_norm, hidden_dim, vocab_size, eps_mod


def _prepare_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    name: str,
    vocab_size: int,
    score_from: int,
    general_prefix: str,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Validate domain chunks and extract token IDs, general mask, and domain labels."""
    if isinstance(chunks, (str, bytes, bytearray)) or not isinstance(chunks, Sequence) or len(chunks) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")
    rows: list[torch.Tensor] = []
    domains: list[str] = []
    seq_len: int | None = None
    start_pos = max(0, score_from - 1)

    for idx, ch in enumerate(chunks):
        if not isinstance(ch, Mapping) or "expert_id" not in ch or "ids" not in ch:
            raise ValueError(f"{name}[{idx}] must be a mapping with 'expert_id' and 'ids'")
        eid = ch["expert_id"]
        if not isinstance(eid, str) or not eid.strip():
            raise ValueError(f"{name}[{idx}]['expert_id'] must be a non-empty string")
        raw_ids = ch["ids"]
        if isinstance(raw_ids, torch.Tensor):
            if raw_ids.is_floating_point() or raw_ids.dtype == torch.bool:
                raise TypeError(f"{name}[{idx}]['ids'] tensor must have an integer dtype")
            ids_t = raw_ids.detach().to(device="cpu", dtype=torch.long)
        elif isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes, bytearray)):
            for elem in raw_ids:
                _validate_nonneg_int(elem, f"{name}[{idx}]['ids'] element")
            ids_t = torch.tensor(list(raw_ids), dtype=torch.long)
        else:
            raise TypeError(f"{name}[{idx}]['ids'] must be a sequence or 1D integer tensor")

        if ids_t.ndim != 1 or ids_t.numel() < 2:
            raise ValueError(f"{name}[{idx}]['ids'] must be 1D with length >= 2")
        if bool((ids_t < 0).any().item()) or bool((ids_t >= vocab_size).any().item()):
            raise ValueError(f"{name}[{idx}]['ids'] token IDs must lie in [0, {vocab_size})")
        if seq_len is None:
            seq_len = int(ids_t.numel())
            if start_pos >= seq_len - 1:
                raise ValueError(f"{name} sequence length {seq_len} is too short for score_from={score_from}")
        elif int(ids_t.numel()) != seq_len:
            raise ValueError(f"All {name} entries must share sequence length {seq_len}, got {ids_t.numel()}")
        rows.append(ids_t)
        domains.append(eid)

    ids_2d = torch.stack(rows, dim=0)
    is_e0 = np.array([d.startswith(general_prefix) for d in domains], dtype=bool)
    if not bool(np.any(is_e0)) or not bool(np.any(~is_e0)):
        raise ValueError(
            f"{name} must contain both general ('{general_prefix}*') and specialist chunks"
        )
    return ids_2d, is_e0, domains


def _evaluate_pre_ln_domain_metrics(
    h_pre_ln: torch.Tensor,
    ids_2d: torch.Tensor,
    domains: Sequence[str],
    w: torch.Tensor,
    b: torch.Tensor | None,
    head_mod: nn.Module,
    *,
    eps: float | None,
    norm_type: str,
    score_from: int,
    num_bins: int,
    ignore_index: int,
) -> tuple[dict[str, float], dict[str, dict[str, Any]], dict[str, Any]]:
    """Evaluate per-domain NLL/ECE/entropy and global token metrics from pre-norm states."""
    start_pos = max(0, score_from - 1)
    h_slice = h_pre_ln[:, start_pos:-1, :]
    t_slice = ids_2d[:, start_pos + 1 :]
    h_norm = apply_norm_weights(h_slice, w, b, eps=eps, norm_type=norm_type)
    logits = _eval_unembed_logits(head_mod, h_norm)

    global_metrics = compute_token_nll_and_ece(logits, t_slice, num_bins=num_bins, ignore_index=ignore_index)
    domain_nll: dict[str, float] = {}
    domain_metrics: dict[str, dict[str, Any]] = {}
    dom_indices: dict[str, list[int]] = {}
    for idx, dom in enumerate(domains):
        dom_indices.setdefault(dom, []).append(idx)

    for dom in sorted(dom_indices.keys()):
        ix = dom_indices[dom]
        dm = compute_token_nll_and_ece(logits[ix], t_slice[ix], num_bins=num_bins, ignore_index=ignore_index)
        domain_nll[dom] = float(dm["nll_mean"])
        domain_metrics[dom] = {
            "nll_mean": float(dm["nll_mean"]),
            "ece_value": float(dm["ece_value"]),
            "entropy_mean": float(dm["entropy_mean"]),
            "brier_value": float(dm["brier_value"]),
            "accuracy": float(dm["accuracy"]),
            "num_tokens": int(dm["num_tokens"]),
        }
    return domain_nll, domain_metrics, global_metrics


def _compute_anse_summary(
    baseline_domain_nll: Mapping[str, float],
    state_domain_nll: Mapping[str, float],
    state_ece: float,
    *,
    cfg: LogitCalibrationConfig,
) -> dict[str, Any]:
    """Compute ANSE thermodynamic energy, general drift, and worst-domain guard verdict."""
    ratios: dict[str, float] = {}
    drifts_pct: dict[str, float] = {}
    for dom in sorted(state_domain_nll.keys()):
        base_v = float(baseline_domain_nll[dom])
        cand_v = float(state_domain_nll[dom])
        ratio = cand_v / max(base_v, 1e-12)
        ratios[dom] = ratio
        drifts_pct[dom] = 100.0 * (ratio - 1.0)

    mean_base = sum(float(baseline_domain_nll[d]) for d in state_domain_nll) / float(len(state_domain_nll))
    mean_state = sum(float(state_domain_nll[d]) for d in state_domain_nll) / float(len(state_domain_nll))
    mean_diff = mean_state - mean_base

    general_keys = [d for d in sorted(ratios.keys()) if d.startswith(cfg.general_prefix)]
    general_ratio = max(ratios[d] for d in general_keys)
    worst_domain_id = max(sorted(ratios.keys()), key=lambda d: ratios[d])
    worst_domain_ratio = ratios[worst_domain_id]

    max_ratio = 1.0 + cfg.max_domain_drift
    general_guard_ok = general_ratio <= max_ratio + 1e-12
    worst_guard_ok = worst_domain_ratio <= max_ratio + 1e-12
    guard_ok = bool(general_guard_ok and worst_guard_ok)

    penalty = (
        cfg.maximum_pain_energy
        if not guard_ok
        else (cfg.soft_margin_penalty if general_ratio > cfg.anse_safe_general_ratio else 0.0)
    )
    anse_energy = mean_diff + cfg.ece_energy_weight * state_ece + penalty

    return {
        "mean_base_nll": float(mean_base),
        "mean_state_nll": float(mean_state),
        "mean_diff_vs_base": float(mean_diff),
        "general_ratio": float(general_ratio),
        "general_pct_vs_base": float(100.0 * (general_ratio - 1.0)),
        "worst_domain_id": worst_domain_id,
        "worst_domain_ratio": float(worst_domain_ratio),
        "worst_domain_pct_vs_base": float(100.0 * (worst_domain_ratio - 1.0)),
        "general_guard_ok": general_guard_ok,
        "guard_ok": guard_ok,
        "anse_energy": float(anse_energy),
        "domain_drifts_pct": drifts_pct,
    }


def _resolve_surgery_config(
    config: LogitCalibrationConfig | None,
    *,
    score_from: int | None,
    general_guard_rel: float | None,
    max_domain_drift: float | None,
    channelwise: bool | None,
) -> LogitCalibrationConfig:
    """Merge optional keyword overrides into a validated ``LogitCalibrationConfig``."""
    if config is not None and not isinstance(config, LogitCalibrationConfig):
        raise TypeError(f"config must be LogitCalibrationConfig or None, got {type(config).__name__}")
    base = config or LogitCalibrationConfig()
    return LogitCalibrationConfig(
        num_bins=base.num_bins,
        general_guard_rel=(
            base.general_guard_rel
            if general_guard_rel is None
            else _validate_nonneg_finite(general_guard_rel, "general_guard_rel")
        ),
        max_domain_drift=(
            base.max_domain_drift
            if max_domain_drift is None
            else _validate_nonneg_finite(max_domain_drift, "max_domain_drift")
        ),
        anse_safe_general_ratio=base.anse_safe_general_ratio,
        maximum_pain_energy=base.maximum_pain_energy,
        soft_margin_penalty=base.soft_margin_penalty,
        ece_energy_weight=base.ece_energy_weight,
        tau_min=base.tau_min,
        tau_max=base.tau_max,
        num_tau_grid=base.num_tau_grid,
        num_tau_refine=base.num_tau_refine,
        lambda_grid=base.lambda_grid,
        channelwise=base.channelwise if channelwise is None else _validate_bool(channelwise, "channelwise"),
        channelwise_steps=base.channelwise_steps,
        channelwise_lr=base.channelwise_lr,
        channelwise_reg=base.channelwise_reg,
        channelwise_min_rel=base.channelwise_min_rel,
        channelwise_max_rel=base.channelwise_max_rel,
        norm_type=base.norm_type,
        eps=base.eps,
        ignore_index=base.ignore_index,
        score_from=base.score_from if score_from is None else _validate_nonneg_int(score_from, "score_from"),
        batch_size=base.batch_size,
        general_prefix=base.general_prefix,
        require_strict_energy_drop=base.require_strict_energy_drop,
    )


@torch.no_grad()
def apply_verified_ln_temperature_surgery_inplace(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    base_ln_w: torch.Tensor | None = None,
    base_ln_b: torch.Tensor | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    baseline_domain_nll: Mapping[str, float] | None = None,
    config: LogitCalibrationConfig | None = None,
    score_from: int | None = None,
    general_guard_rel: float | None = None,
    max_domain_drift: float | None = None,
    channelwise: bool | None = None,
    norm_attr: str = "ln",
    head_attr: str = "head",
) -> dict[str, Any]:
    """Calibrate final LayerNorm/RMSNorm temperature & interpolation in-place with ANSE rollback."""
    cfg = _resolve_surgery_config(
        config,
        score_from=score_from,
        general_guard_rel=general_guard_rel,
        max_domain_drift=max_domain_drift,
        channelwise=channelwise,
    )
    if not isinstance(norm_attr, str) or not norm_attr.strip():
        raise ValueError("norm_attr must be a non-empty string")
    if not isinstance(head_attr, str) or not head_attr.strip():
        raise ValueError("head_attr must be a non-empty string")

    norm_mod, head_mod, resolved_norm, hidden_dim, vocab_size, eps_mod = _validate_model_and_norm(
        model,
        norm_attr=norm_attr.strip(),
        head_attr=head_attr.strip(),
        norm_type=cfg.norm_type,
    )
    eps_eff = cfg.eps if cfg.eps is not None else eps_mod

    probe_ids, probe_is_e0, _ = _prepare_chunks(
        probe_chunks,
        name="probe_chunks",
        vocab_size=vocab_size,
        score_from=cfg.score_from,
        general_prefix=cfg.general_prefix,
    )
    if eval_chunks is not None:
        verify_ids, _, verify_domains = _prepare_chunks(
            eval_chunks,
            name="eval_chunks",
            vocab_size=vocab_size,
            score_from=cfg.score_from,
            general_prefix=cfg.general_prefix,
        )
    else:
        verify_ids = probe_ids
        verify_domains = [str(ch["expert_id"]) for ch in probe_chunks]

    moe_w = norm_mod.weight.detach().clone()
    moe_b_raw = getattr(norm_mod, "bias", None)
    moe_b = moe_b_raw.detach().clone() if moe_b_raw is not None else None

    if base_ln_w is not None:
        base_w = _validate_1d_norm_param(base_ln_w, "base_ln_w", expected_dim=hidden_dim).detach().clone()
        if (base_ln_b is None) != (moe_b is None):
            raise ValueError("base_ln_b presence must match model norm bias presence")
        base_b = (
            _validate_1d_norm_param(base_ln_b, "base_ln_b", expected_dim=hidden_dim).detach().clone()
            if base_ln_b is not None
            else None
        )
    else:
        if base_ln_b is not None:
            raise ValueError("base_ln_b cannot be provided when base_ln_w is None")
        base_w = moe_w.clone()
        base_b = moe_b.clone() if moe_b is not None else None

    validated_baseline_nll: dict[str, float] | None = None
    if baseline_domain_nll is not None:
        if not isinstance(baseline_domain_nll, Mapping) or not baseline_domain_nll:
            raise ValueError("baseline_domain_nll must be a non-empty mapping when provided")
        validated_baseline_nll = {
            str(k): _validate_positive_finite(v, f"baseline_domain_nll[{k!r}]")
            for k, v in baseline_domain_nll.items()
        }
        missing = sorted(set(verify_domains) - set(validated_baseline_nll.keys()))
        if missing:
            raise ValueError(f"baseline_domain_nll is missing verification domains: {missing}")

    was_training = model.training
    model.eval()
    try:
        parent_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in model.state_dict().items()}
        parent_sha = compute_module_state_sha256(model)

        h_probe = forward_model_pre_ln(model, probe_ids, batch_size=cfg.batch_size)
        if not bool(torch.isfinite(h_probe).all().item()):
            raise ValueError("Model trunk produced non-finite pre-norm probe states")

        h_verify = (
            h_probe
            if eval_chunks is None
            else forward_model_pre_ln(model, verify_ids, batch_size=cfg.batch_size)
        )
        if not bool(torch.isfinite(h_verify).all().item()):
            raise ValueError("Model trunk produced non-finite pre-norm verification states")

        start_pos = max(0, cfg.score_from - 1)
        h_probe_scored = h_probe[:, start_pos:-1, :]
        t_probe_scored = probe_ids[:, start_pos + 1 :]
        e0_mask_t = torch.from_numpy(probe_is_e0)

        opt = optimize_guard_constrained_ln_scale_and_interpolation(
            base_w,
            base_b,
            moe_w,
            moe_b,
            h_probe_scored[e0_mask_t],
            t_probe_scored[e0_mask_t],
            h_probe_scored[~e0_mask_t],
            t_probe_scored[~e0_mask_t],
            head_mod,
            general_guard_rel=cfg.general_guard_rel,
            eps=eps_eff,
            norm_type=resolved_norm,
            num_bins=cfg.num_bins,
            ignore_index=cfg.ignore_index,
            lambda_grid=cfg.lambda_grid,
            tau_min=cfg.tau_min,
            tau_max=cfg.tau_max,
            num_tau_grid=cfg.num_tau_grid,
            num_tau_refine=cfg.num_tau_refine,
            ece_weight=cfg.ece_energy_weight,
            channelwise=cfg.channelwise,
            channelwise_steps=cfg.channelwise_steps,
            channelwise_lr=cfg.channelwise_lr,
            channelwise_reg=cfg.channelwise_reg,
            channelwise_min_rel=cfg.channelwise_min_rel,
            channelwise_max_rel=cfg.channelwise_max_rel,
        )

        parent_dom_nll, parent_dom_metrics, parent_global = _evaluate_pre_ln_domain_metrics(
            h_verify,
            verify_ids,
            verify_domains,
            moe_w,
            moe_b,
            head_mod,
            eps=eps_eff,
            norm_type=resolved_norm,
            score_from=cfg.score_from,
            num_bins=cfg.num_bins,
            ignore_index=cfg.ignore_index,
        )
        if validated_baseline_nll is not None:
            ref_dom_nll = validated_baseline_nll
        elif base_ln_w is not None:
            ref_dom_nll, _, _ = _evaluate_pre_ln_domain_metrics(
                h_verify,
                verify_ids,
                verify_domains,
                base_w,
                base_b,
                head_mod,
                eps=eps_eff,
                norm_type=resolved_norm,
                score_from=cfg.score_from,
                num_bins=cfg.num_bins,
                ignore_index=cfg.ignore_index,
            )
        else:
            ref_dom_nll = parent_dom_nll

        parent_anse = _compute_anse_summary(
            ref_dom_nll, parent_dom_nll, float(parent_global["ece_value"]), cfg=cfg
        )

        # Mutate final norm weights/bias in-place and evaluate candidate state
        norm_mod.weight.copy_(opt["folded_ln_w"].to(device=norm_mod.weight.device, dtype=norm_mod.weight.dtype))
        if moe_b_raw is not None and opt["folded_ln_b"] is not None:
            moe_b_raw.copy_(opt["folded_ln_b"].to(device=moe_b_raw.device, dtype=moe_b_raw.dtype))

        cand_sha = compute_module_state_sha256(model)
        cand_dom_nll, cand_dom_metrics, cand_global = _evaluate_pre_ln_domain_metrics(
            h_verify,
            verify_ids,
            verify_domains,
            norm_mod.weight.detach(),
            getattr(norm_mod, "bias", None),
            head_mod,
            eps=eps_eff,
            norm_type=resolved_norm,
            score_from=cfg.score_from,
            num_bins=cfg.num_bins,
            ignore_index=cfg.ignore_index,
        )
        cand_anse = _compute_anse_summary(
            ref_dom_nll, cand_dom_nll, float(cand_global["ece_value"]), cfg=cfg
        )

        parent_e = float(parent_anse["anse_energy"])
        cand_e = float(cand_anse["anse_energy"])
        delta_e = cand_e - parent_e
        energy_ok = (delta_e < 0.0) if cfg.require_strict_energy_drop else (delta_e <= 0.0)
        guard_ok = bool(cand_anse["guard_ok"])

        if cand_sha == parent_sha:
            status = "FIXED_POINT_UNCHANGED" if guard_ok else "REJECTED_GUARD_VIOLATION_UNCHANGED"
            accepted = False
            rolled_back = False
        elif guard_ok and energy_ok:
            status = "ACCEPTED"
            accepted = True
            rolled_back = False
        else:
            status = "ROLLED_BACK_GUARD_VIOLATION" if not guard_ok else "ROLLED_BACK_ENERGY_NON_IMPROVING"
            accepted = False
            rolled_back = True
            cur_sd = model.state_dict()
            model.load_state_dict(
                {k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in parent_snapshot.items()}
            )

        final_sha = compute_module_state_sha256(model)
        if not accepted and final_sha != parent_sha:
            raise RuntimeError(f"Atomic bitwise rollback failed: {final_sha[:12]} != {parent_sha[:12]}")
    finally:
        model.train(was_training)

    active_global = cand_global if accepted else parent_global
    active_dom_metrics = cand_dom_metrics if accepted else parent_dom_metrics
    active_anse = cand_anse if accepted else parent_anse

    parent_ece = float(parent_global["ece_value"])
    final_ece = float(active_global["ece_value"])
    ece_reduction_rel = (parent_ece - final_ece) / parent_ece if parent_ece > 1e-12 else 0.0

    spec_domains = [d for d in sorted(ref_dom_nll.keys()) if not d.startswith(cfg.general_prefix)]
    base_spec_mean = (
        sum(ref_dom_nll[d] for d in spec_domains) / float(len(spec_domains)) if spec_domains else 0.0
    )
    final_spec_mean = (
        sum(float(active_dom_metrics[d]["nll_mean"]) for d in spec_domains) / float(len(spec_domains))
        if spec_domains
        else 0.0
    )

    receipt: dict[str, Any] = {
        "object": SURGEON_RECEIPT_OBJECT,
        "schema": SURGEON_SCHEMA,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "status": status,
        "norm_type": resolved_norm,
        "has_bias": moe_b_raw is not None,
        "guard_ok": guard_ok,
        "general_guard_ok": bool(cand_anse["general_guard_ok"]),
        "lambda_ln": float(opt["lambda_ln"]),
        "tau_temp": float(opt["tau_temp"]),
        "channel_tau": opt["channel_tau"],
        "nll_mean": float(active_global["nll_mean"]),
        "parent_nll_mean": float(parent_global["nll_mean"]),
        "cand_nll_mean": float(cand_global["nll_mean"]),
        "ece_value": final_ece,
        "parent_ece_value": parent_ece,
        "cand_ece_value": float(cand_global["ece_value"]),
        "ece_reduction_rel": float(ece_reduction_rel),
        "entropy_mean": float(active_global["entropy_mean"]),
        "brier_value": float(active_global["brier_value"]),
        "accuracy": float(active_global["accuracy"]),
        "spec_gain_nats": float(base_spec_mean - final_spec_mean),
        "anse_energy": float(active_anse["anse_energy"]),
        "parent_anse_energy": parent_e,
        "cand_anse_energy": cand_e,
        "delta_e": float(delta_e),
        "general_pct_vs_base": float(cand_anse["general_pct_vs_base"]),
        "worst_domain_id": str(cand_anse["worst_domain_id"]),
        "worst_domain_pct_vs_base": float(cand_anse["worst_domain_pct_vs_base"]),
        "domain_drifts_pct": cand_anse["domain_drifts_pct"],
        "per_domain_metrics": active_dom_metrics,
        "parent_state_sha256": parent_sha,
        "candidate_state_sha256": cand_sha,
        "final_state_sha256": final_sha,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()},
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_ECE_ENERGY_WEIGHT",
    "DEFAULT_GENERAL_GUARD_REL",
    "DEFAULT_IGNORE_INDEX",
    "DEFAULT_LAMBDA_GRID",
    "DEFAULT_MAX_DOMAIN_DRIFT",
    "DEFAULT_NUM_BINS",
    "DEFAULT_NUM_TAU_GRID",
    "DEFAULT_NUM_TAU_REFINE",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_SOFT_MARGIN_PENALTY",
    "DEFAULT_TAU_MAX",
    "DEFAULT_TAU_MIN",
    "GENERAL_EXPERT_PREFIX",
    "SURGEON_RECEIPT_OBJECT",
    "SURGEON_SCHEMA",
    "VALID_NORM_TYPES",
    "LogitCalibrationConfig",
    "apply_norm_weights",
    "apply_verified_ln_temperature_surgery_inplace",
    "canonical_json_bytes",
    "compute_module_state_sha256",
    "compute_payload_sha256",
    "compute_token_nll_and_ece",
    "detect_norm_type",
    "fold_temperature_into_norm_params",
    "optimize_guard_constrained_ln_scale_and_interpolation",
    "verify_calibration_receipt_provenance",
]

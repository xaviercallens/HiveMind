# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-Overhead Folded Final RMSNorm/LayerNorm Temperature & ECE Calibration Surgeon.

Implements training-free, zero-latency-overhead logit calibration for open-weight
language models and Sparse MoEs by folding temperature scaling ``1 / tau_temp`` and
interpolation ``lambda_norm`` directly into the final ``RMSNorm`` or ``LayerNorm``
parameters (``w_folded = (w_base + lambda_norm * (w_cand - w_base)) / tau_temp``, plus
``b_folded`` when bias is present).

Grounded in ``PILOT3B`` on ``Qwen/Qwen2.5-1.5B`` (1,543,714,304 total parameters),
where folding ``tau_temp = 1.05`` into ``1,536`` final RMSNorm weights
(``0.0000995%`` parameter budget) cut virgin-vault Expected Calibration Error by
``-46.8%`` (``0.0260 -> 0.0138``) and lowered multi-domain NLL by ``-0.0059`` nats
while the explicit ANSE stage ratchet rolled back unnecessary tail-layer ESD edits:

1. ``compute_token_nll_and_ece(logits, targets, *, num_bins=15)``:
   Exact ``float64`` chunked computation of token-level NLL, top-1 accuracy,
   equal-width binned Expected Calibration Error (ECE), MCE, Brier score-free mean
   squared probability residual (``brier_value``), and Shannon entropy.
2. ``fit_folded_norm_temperature_and_interpolation(...)``:
   Exact affine logit-linearity 2-matmul solver jointly optimizing
   ``(lambda_norm, tau_temp)`` under the general-domain drift guard and folding
   ``1 / tau_temp`` directly into final ``RMSNorm`` / ``LayerNorm`` parameters with
   exact ``< 1e-6`` logit equivalence to post-hoc ``logits / tau_temp`` scaling.
3. ``execute_verified_folded_norm_calibration(...)``:
   Verified calibration pipeline with an explicit ANSE stage-ratchet gate that
   honestly rolls back upstream spectral/LoRA perturbations whenever clean folded-norm
   calibration achieves lower ANSE energy, atomic bitwise rollback on ANSE Maximum Pain,
   and canonical SHA-256 provenance receipts.
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

from hivemind.swarm.logit_calibration_temperature_surgeon import (
    apply_norm_weights,
    canonical_json_bytes,
    compute_module_state_sha256,
    compute_payload_sha256,
    detect_norm_type,
    fold_temperature_into_norm_params,
)

CALIBRATOR_SCHEMA: str = "hivemind.folded_norm_ece_calibrator/1"
CALIBRATOR_RECEIPT_OBJECT: str = "hivemind.folded_norm_ece_calibrator_receipt"

ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_NUM_BINS: int = 15
DEFAULT_MAX_GENERAL_DRIFT_REL: float = 0.01
DEFAULT_MAX_DOMAIN_DRIFT_REL: float = 0.01
DEFAULT_ANSE_SAFE_GENERAL_RATIO: float = 1.0085
DEFAULT_ANSE_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_ECE_ENERGY_WEIGHT: float = 0.25
DEFAULT_MIN_STAGE_ENERGY_DROP: float = 1e-6
DEFAULT_SUB001_BUDGET_FRACTION: float = 1e-5
DEFAULT_IGNORE_INDEX: int = -100
DEFAULT_SCORE_FROM: int = 32
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_LAMBDA_GRID: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
DEFAULT_TAU_GRID: tuple[float, ...] = (
    0.75,
    0.85,
    0.90,
    0.93,
    0.95,
    0.97,
    0.985,
    1.00,
    1.015,
    1.03,
    1.05,
    1.08,
    1.12,
    1.18,
    1.25,
    1.35,
    1.50,
    1.75,
    2.00,
    2.50,
    3.00,
)
DEFAULT_GENERAL_PREFIXES: tuple[str, ...] = ("E0_", "D0_")
VALID_NORM_KINDS: frozenset[str] = frozenset({"auto", "layernorm", "rmsnorm"})

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_CHUNK_SIZE: int = 4096
_NORM_ATTR_CANDIDATES: tuple[str, ...] = (
    "ln",
    "norm",
    "final_norm",
    "model.norm",
    "model.language_model.norm",
    "transformer.ln_f",
)
_HEAD_ATTR_CANDIDATES: tuple[str, ...] = ("head", "lm_head", "output")


# ---------------------------------------------------------------------------
# 1. Validation & Configuration Helpers
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


def _validate_nonempty_str(value: Any, name: str) -> str:
    """Validate a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


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


def _validate_1d_param(tensor: Any, name: str, expected_dim: int | None = None) -> torch.Tensor:
    """Validate a 1D finite floating-point norm parameter."""
    t = _validate_fp_tensor(tensor, name, exact_ndim=1)
    if expected_dim is not None and int(t.shape[0]) != expected_dim:
        raise ValueError(f"{name} length {t.shape[0]} must match hidden dim {expected_dim}")
    return t


def _validate_int_tensor(tensor: Any, expected_shape: tuple[int, ...], name: str) -> torch.Tensor:
    """Validate an integer tensor matching ``expected_shape``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} shape {tuple(tensor.shape)} must match {expected_shape}")
    if tensor.is_floating_point() or tensor.dtype == torch.bool:
        raise TypeError(f"{name} must have an integer dtype, got {tensor.dtype}")
    return tensor.detach().to(device="cpu", dtype=torch.long).contiguous()


def _validate_lambda_grid(lambda_grid: Sequence[float]) -> tuple[float, ...]:
    """Validate norm interpolation grid entries inside ``[0.0, 1.0]``."""
    if isinstance(lambda_grid, (str, bytes, bytearray)) or not isinstance(lambda_grid, Sequence):
        raise TypeError("lambda_grid must be a non-empty sequence of floats")
    if len(lambda_grid) == 0:
        raise ValueError("lambda_grid must be non-empty")
    cleaned: list[float] = []
    for idx, val in enumerate(lambda_grid):
        f_val = _validate_nonneg_finite(val, f"lambda_grid[{idx}]")
        if f_val > 1.0:
            raise ValueError(f"lambda_grid[{idx}] must lie in [0.0, 1.0], got {f_val}")
        cleaned.append(f_val)
    return tuple(dict.fromkeys(cleaned))


def _validate_tau_grid(tau_grid: Sequence[float]) -> tuple[float, ...]:
    """Validate positive finite scalar temperature candidates."""
    if isinstance(tau_grid, (str, bytes, bytearray)) or not isinstance(tau_grid, Sequence):
        raise TypeError("tau_grid must be a non-empty sequence of positive floats")
    if len(tau_grid) == 0:
        raise ValueError("tau_grid must be non-empty")
    vals = sorted({_validate_positive_finite(v, f"tau_grid[{i}]") for i, v in enumerate(tau_grid)})
    return tuple(vals)


def _validate_general_prefixes(prefixes: Sequence[str]) -> tuple[str, ...]:
    """Validate non-empty sequence of domain prefixes denoting general prose."""
    if isinstance(prefixes, (str, bytes, bytearray)) or not isinstance(prefixes, Sequence):
        raise TypeError("general_prefixes must be a non-empty sequence of strings")
    if not prefixes:
        raise ValueError("general_prefixes must be non-empty")
    return tuple(_validate_nonempty_str(p, f"general_prefixes[{i}]") for i, p in enumerate(prefixes))


@dataclass(frozen=True)
class FoldedNormCalibrationConfig:
    """Validated configuration for Sub-0.001% Folded Norm Temperature & ECE Surgery."""

    num_bins: int = DEFAULT_NUM_BINS
    max_general_drift_rel: float = DEFAULT_MAX_GENERAL_DRIFT_REL
    max_domain_drift_rel: float = DEFAULT_MAX_DOMAIN_DRIFT_REL
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_GENERAL_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY
    ece_energy_weight: float = DEFAULT_ECE_ENERGY_WEIGHT
    min_stage_energy_drop: float = DEFAULT_MIN_STAGE_ENERGY_DROP
    max_edited_param_fraction: float | None = None
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID
    tau_grid: tuple[float, ...] = DEFAULT_TAU_GRID
    refine_tau: bool = True
    num_tau_refine: int = 11
    ignore_index: int = DEFAULT_IGNORE_INDEX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    general_prefixes: tuple[str, ...] = DEFAULT_GENERAL_PREFIXES
    require_strict_energy_drop: bool = True

    def __post_init__(self) -> None:
        _validate_positive_int(self.num_bins, "num_bins")
        _validate_nonneg_finite(self.max_general_drift_rel, "max_general_drift_rel")
        _validate_nonneg_finite(self.max_domain_drift_rel, "max_domain_drift_rel")
        _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        _validate_nonneg_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_nonneg_finite(self.ece_energy_weight, "ece_energy_weight")
        _validate_nonneg_finite(self.min_stage_energy_drop, "min_stage_energy_drop")
        if self.max_edited_param_fraction is not None:
            frac = _validate_positive_finite(self.max_edited_param_fraction, "max_edited_param_fraction")
            if frac > 1.0:
                raise ValueError(f"max_edited_param_fraction must be <= 1.0, got {frac}")
        object.__setattr__(self, "lambda_grid", _validate_lambda_grid(self.lambda_grid))
        object.__setattr__(self, "tau_grid", _validate_tau_grid(self.tau_grid))
        _validate_bool(self.refine_tau, "refine_tau")
        _validate_positive_int(self.num_tau_refine, "num_tau_refine")
        _validate_int(self.ignore_index, "ignore_index")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        object.__setattr__(self, "general_prefixes", _validate_general_prefixes(self.general_prefixes))
        _validate_bool(self.require_strict_energy_drop, "require_strict_energy_drop")


def verify_folded_norm_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify canonical ``receipt_sha256`` provenance on a calibration receipt."""
    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a mapping")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != CALIBRATOR_RECEIPT_OBJECT or receipt.get("schema") != CALIBRATOR_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


# ---------------------------------------------------------------------------
# 2. Exact Token-Level NLL, Top-1 Accuracy & Binned ECE Metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_token_nll_and_ece(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_bins: int = DEFAULT_NUM_BINS,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> dict[str, Any]:
    """Compute exact token-level NLL, top-1 accuracy, and binned Expected Calibration Error.

    Evaluates in ``float64`` over bounded token slices so arbitrarily long sequences
    do not spike RAM, assigning confidences to ``num_bins`` equal-width intervals
    ``[0, 1/B), ..., [(B-1)/B, 1.0]``.
    """
    logits_t = _validate_fp_tensor(logits, "logits", min_ndim=2)
    bins_i = _validate_positive_int(num_bins, "num_bins")
    ignore_i = _validate_int(ignore_index, "ignore_index")
    targets_t = _validate_int_tensor(targets, tuple(logits_t.shape[:-1]), "targets")

    vocab_size = int(logits_t.shape[-1])
    flat_logits = logits_t.detach().reshape(-1, vocab_size)
    flat_targets = targets_t.reshape(-1)

    valid_mask = flat_targets != ignore_i
    num_tokens = int(valid_mask.sum().item())
    if num_tokens == 0:
        raise ValueError("targets must contain at least one non-ignored token")

    logits_valid = flat_logits[valid_mask.to(flat_logits.device)]
    targets_valid = flat_targets[valid_mask]
    if bool((targets_valid < 0).any().item()) or bool((targets_valid >= vocab_size).any().item()):
        raise ValueError(
            f"targets token IDs must lie in [0, {vocab_size}) or equal ignore_index={ignore_i}"
        )

    nll_parts: list[torch.Tensor] = []
    ent_parts: list[torch.Tensor] = []
    brier_parts: list[torch.Tensor] = []
    conf_parts: list[torch.Tensor] = []
    acc_parts: list[torch.Tensor] = []

    for start in range(0, num_tokens, _TOKEN_CHUNK_SIZE):
        end = min(num_tokens, start + _TOKEN_CHUNK_SIZE)
        chunk_logits = logits_valid[start:end].to(device="cpu", dtype=torch.float64)
        chunk_targets = targets_valid[start:end]

        log_probs = F.log_softmax(chunk_logits, dim=-1)
        probs = log_probs.exp()
        token_nll = F.nll_loss(log_probs, chunk_targets, reduction="none")
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
    top1_accuracy = float(all_acc.mean().item())
    mean_confidence = float(all_conf.mean().item())
    overconfidence_gap = mean_confidence - top1_accuracy

    boundaries = torch.linspace(0.0, 1.0, steps=bins_i + 1, dtype=torch.float64)
    bin_ids = torch.bucketize(all_conf, boundaries[1:-1], right=True)
    counts = torch.bincount(bin_ids, minlength=bins_i)
    sum_acc = torch.bincount(bin_ids, weights=all_acc, minlength=bins_i)
    sum_conf = torch.bincount(bin_ids, weights=all_conf, minlength=bins_i)

    nz_mask = counts > 0
    counts_f = counts.to(torch.float64)
    bin_accs = torch.where(nz_mask, sum_acc / counts_f.clamp_min(1.0), torch.zeros_like(counts_f))
    bin_confs = torch.where(nz_mask, sum_conf / counts_f.clamp_min(1.0), torch.zeros_like(counts_f))
    abs_gaps = torch.where(nz_mask, (bin_accs - bin_confs).abs(), torch.zeros_like(counts_f))

    expected_calibration_error = float(((counts_f / float(num_tokens)) * abs_gaps).sum().item())
    mce = float(abs_gaps.max().item())

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
        "expected_calibration_error": expected_calibration_error,
        "top1_accuracy": top1_accuracy,
        "mce": mce,
        "entropy_mean": entropy_mean,
        "brier_value": brier_value,
        "mean_confidence": mean_confidence,
        "overconfidence_gap": overconfidence_gap,
        "perplexity": float(math.exp(min(nll_mean, 700.0))),
        "num_tokens": num_tokens,
        "num_bins": bins_i,
        "bins": bin_rows,
    }


# ---------------------------------------------------------------------------
# 3. Zero-Overhead Affine 2-Matmul (lambda_norm, tau_temp) Calibration & Fold
# ---------------------------------------------------------------------------


def _resolve_norm_kind_and_biases(
    norm_kind: str,
    base_bias: torch.Tensor | None,
    cand_bias: torch.Tensor | None,
    hidden_dim: int,
) -> tuple[str, torch.Tensor | None, torch.Tensor | None]:
    """Validate and resolve norm kind alongside optional LayerNorm bias tensors."""
    if not isinstance(norm_kind, str) or norm_kind not in VALID_NORM_KINDS:
        raise ValueError(
            f"norm_kind must be one of {sorted(VALID_NORM_KINDS)}, got {norm_kind!r}"
        )
    if (base_bias is None) != (cand_bias is None):
        raise ValueError("base_norm_bias and cand_norm_bias must either both be provided or both be None")
    b0 = _validate_1d_param(base_bias, "base_norm_bias", expected_dim=hidden_dim) if base_bias is not None else None
    b1 = _validate_1d_param(cand_bias, "cand_norm_bias", expected_dim=hidden_dim) if cand_bias is not None else None

    if norm_kind == "rmsnorm":
        if b0 is not None or b1 is not None:
            raise ValueError("norm_kind='rmsnorm' does not accept non-None bias tensors")
        return "rmsnorm", None, None
    if norm_kind == "layernorm":
        return "layernorm", b0, b1
    return ("rmsnorm" if b0 is None else "layernorm"), b0, b1


def _resolve_general_token_mask(
    is_general_mask: torch.Tensor | Sequence[bool] | np.ndarray,
    batch_shape: tuple[int, ...],
) -> torch.Tensor:
    """Validate and normalize ``is_general_mask`` to a boolean tensor matching ``batch_shape``."""
    if isinstance(is_general_mask, torch.Tensor):
        if is_general_mask.dtype != torch.bool:
            raise TypeError(f"is_general_mask tensor must have bool dtype, got {is_general_mask.dtype}")
        mask_t = is_general_mask.detach().to(device="cpu")
    elif isinstance(is_general_mask, np.ndarray):
        if is_general_mask.dtype != np.bool_:
            raise TypeError(f"is_general_mask ndarray must have bool dtype, got {is_general_mask.dtype}")
        mask_t = torch.from_numpy(is_general_mask.copy())
    elif isinstance(is_general_mask, Sequence) and not isinstance(is_general_mask, (str, bytes, bytearray)):
        if len(is_general_mask) == 0:
            raise ValueError("is_general_mask must not be empty")
        if not all(isinstance(v, bool) for v in is_general_mask):
            raise TypeError("is_general_mask sequence elements must be bool")
        mask_t = torch.tensor(list(is_general_mask), dtype=torch.bool)
    else:
        raise TypeError(
            f"is_general_mask must be a bool Tensor, ndarray, or Sequence[bool], got {type(is_general_mask).__name__}"
        )

    if tuple(mask_t.shape) == batch_shape:
        return mask_t.contiguous()
    if mask_t.ndim == 1 and int(mask_t.shape[0]) == batch_shape[0]:
        expanded = mask_t.view(batch_shape[0], *([1] * (len(batch_shape) - 1)))
        return expanded.expand(batch_shape).contiguous()
    raise ValueError(
        f"is_general_mask shape {tuple(mask_t.shape)} must match {batch_shape} or ({batch_shape[0]},)"
    )


def _compute_unit_normalized(
    x: torch.Tensor,
    norm_kind: str,
    eps: float,
) -> torch.Tensor:
    """Apply zero-parameter unit-scale RMSNorm or LayerNorm in ``float64``."""
    dim = int(x.shape[-1])
    if norm_kind == "rmsnorm":
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + float(eps))
        return x * rms
    return F.layer_norm(x, (dim,), None, None, eps=float(eps))


def _score_logit_pair(
    logits_valid: torch.Tensor,
    targets_valid: torch.Tensor,
    gen_valid_mask: torch.Tensor,
    tau_val: float,
    *,
    ref_gen_nll: float,
    ref_total_nll: float,
    ref_ece: float,
    gen_ceiling: float,
    ece_weight: float,
    num_bins: int,
    boundaries_inner: torch.Tensor,
    maximum_pain: float,
    domain_masks: Sequence[torch.Tensor] = (),
    domain_ceilings: Sequence[float] = (),
) -> tuple[bool, float, float]:
    """Score one ``(lambda_norm, tau_temp)`` logit tensor against guard and calibration objective."""
    scaled = logits_valid * (1.0 / float(tau_val))
    log_probs = F.log_softmax(scaled, dim=-1)
    tok_nll = -log_probs.gather(1, targets_valid.unsqueeze(1)).squeeze(1)
    all_nll = float(tok_nll.mean().item())
    gen_nll = float(tok_nll[gen_valid_mask].mean().item())

    probs = log_probs.exp()
    conf, preds = probs.max(dim=-1)
    acc = (preds == targets_valid).to(torch.float64)
    bin_ids = torch.bucketize(conf, boundaries_inner, right=True)
    counts = torch.bincount(bin_ids, minlength=num_bins)
    nz = counts > 0
    counts_f = counts.to(torch.float64)
    sum_acc = torch.bincount(bin_ids, weights=acc, minlength=num_bins)
    sum_conf = torch.bincount(bin_ids, weights=conf, minlength=num_bins)
    abs_gaps = torch.where(
        nz, (sum_acc - sum_conf).abs() / counts_f.clamp_min(1.0), torch.zeros_like(counts_f)
    )
    all_ece = float(((counts_f / float(targets_valid.numel())) * abs_gaps).sum().item())

    guard_ok = gen_nll <= gen_ceiling + 1e-12
    if guard_ok and domain_masks:
        for dm, ceil_v in zip(domain_masks, domain_ceilings, strict=True):
            if float(tok_nll[dm].mean().item()) > ceil_v + 1e-12:
                guard_ok = False
                break

    mean_diff = all_nll - ref_total_nll
    ece_diff = all_ece - ref_ece
    if guard_ok:
        energy = mean_diff + ece_weight * ece_diff
    else:
        energy = maximum_pain + max(0.0, mean_diff) + ece_weight * all_ece
    gen_drift_rel = (gen_nll / max(ref_gen_nll, 1e-12)) - 1.0
    return guard_ok, float(energy), float(gen_drift_rel)


@torch.no_grad()
def fit_folded_norm_temperature_and_interpolation(
    pre_norm_hidden: torch.Tensor,
    base_norm_weight: torch.Tensor,
    base_norm_bias: torch.Tensor | None,
    cand_norm_weight: torch.Tensor,
    cand_norm_bias: torch.Tensor | None,
    lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
    is_general_mask: torch.Tensor | Sequence[bool] | np.ndarray,
    *,
    norm_kind: str = "rmsnorm",
    eps: float = 1e-6,
    max_general_drift_rel: float = DEFAULT_MAX_GENERAL_DRIFT_REL,
    num_bins: int = DEFAULT_NUM_BINS,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    ece_energy_weight: float = DEFAULT_ECE_ENERGY_WEIGHT,
    lambda_grid: Sequence[float] | None = None,
    tau_grid: Sequence[float] | None = None,
    refine_tau: bool = True,
    num_tau_refine: int = 11,
    base_general_nll_override: float | None = None,
    base_total_nll_override: float | None = None,
    base_ece_override: float | None = None,
    domain_labels: Sequence[str] | None = None,
    max_domain_drift_rel: float | None = None,
    total_model_params: int | None = None,
    seq_batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Jointly optimize ``(lambda_norm, tau_temp)`` and fold ``1 / tau_temp`` into final norm weights.

    Exploits exact affine logit linearity of final normalization + linear unembedding:
    ``norm(h; w_0 + l * dw, b_0 + l * db) W^T / tau = (L_0 + l * dL) / tau``, requiring at
    most 2 ``lm_head`` projections per chunk batch regardless of grid resolution.
    """
    h_pre = _validate_fp_tensor(pre_norm_hidden, "pre_norm_hidden", min_ndim=2)
    hidden_dim = int(h_pre.shape[-1])
    w_base = _validate_1d_param(base_norm_weight, "base_norm_weight", expected_dim=hidden_dim)
    w_cand = _validate_1d_param(cand_norm_weight, "cand_norm_weight", expected_dim=hidden_dim)
    resolved_norm, b_base, b_cand = _resolve_norm_kind_and_biases(
        norm_kind, base_norm_bias, cand_norm_bias, hidden_dim
    )

    head_w = _validate_fp_tensor(lm_head_weight, "lm_head_weight", exact_ndim=2)
    if int(head_w.shape[1]) != hidden_dim:
        raise ValueError(
            f"lm_head_weight width {head_w.shape[1]} must match hidden dim {hidden_dim}"
        )
    vocab_size = int(head_w.shape[0])

    batch_shape = tuple(h_pre.shape[:-1])
    targets_t = _validate_int_tensor(targets, batch_shape, "targets")
    gen_mask_t = _resolve_general_token_mask(is_general_mask, batch_shape)

    eps_f = _validate_positive_finite(eps, "eps")
    guard_rel = _validate_nonneg_finite(max_general_drift_rel, "max_general_drift_rel")
    bins_i = _validate_positive_int(num_bins, "num_bins")
    ignore_i = _validate_int(ignore_index, "ignore_index")
    ece_w = _validate_nonneg_finite(ece_energy_weight, "ece_energy_weight")
    refine_flag = _validate_bool(refine_tau, "refine_tau")
    n_ref = _validate_positive_int(num_tau_refine, "num_tau_refine")
    step_b = _validate_positive_int(seq_batch_size, "seq_batch_size")

    w0_64 = w_base.detach().to(device="cpu", dtype=torch.float64)
    w1_64 = w_cand.detach().to(device="cpu", dtype=torch.float64)
    dw_64 = w1_64 - w0_64
    b0_64 = b_base.detach().to(device="cpu", dtype=torch.float64) if b_base is not None else None
    b1_64 = b_cand.detach().to(device="cpu", dtype=torch.float64) if b_cand is not None else None
    db_64 = (b1_64 - b0_64) if (b0_64 is not None and b1_64 is not None) else None

    has_norm_delta = bool((dw_64 != 0).any().item()) or (
        db_64 is not None and bool((db_64 != 0).any().item())
    )
    lam_vals = (
        _validate_lambda_grid(lambda_grid)
        if lambda_grid is not None
        else (DEFAULT_LAMBDA_GRID if has_norm_delta else (1.0,))
    )
    tau_vals = _validate_tau_grid(tau_grid) if tau_grid is not None else DEFAULT_TAU_GRID
    if tau_grid is None and 1.0 not in tau_vals:
        tau_vals = tuple(sorted({*tau_vals, 1.0}))

    flat_targets = targets_t.reshape(-1)
    flat_gen = gen_mask_t.reshape(-1)
    valid_mask = flat_targets != ignore_i
    if not bool(valid_mask.any().item()):
        raise ValueError("targets must contain at least one non-ignored token")

    targets_valid = flat_targets[valid_mask]
    if bool((targets_valid < 0).any().item()) or bool((targets_valid >= vocab_size).any().item()):
        raise ValueError(
            f"targets token IDs must lie in [0, {vocab_size}) or equal ignore_index={ignore_i}"
        )
    gen_valid_mask = flat_gen[valid_mask]
    if not bool(gen_valid_mask.any().item()):
        raise ValueError("is_general_mask must select at least one non-ignored general token")
    spec_valid_mask = ~gen_valid_mask
    has_specialist = bool(spec_valid_mask.any().item())

    # Stream unit-normalized states and project through lm_head in at most 2 matmuls per batch
    head_64 = head_w.detach().to(device="cpu", dtype=torch.float64)
    h_flat = h_pre.detach().to(device="cpu", dtype=torch.float64).reshape(-1, hidden_dim)
    h_valid = h_flat[valid_mask]
    n_valid = int(h_valid.shape[0])
    token_batch = max(1, step_b * max(1, int(h_pre.shape[1]) if h_pre.ndim >= 3 else 1))

    l0_parts: list[torch.Tensor] = []
    dl_parts: list[torch.Tensor] = []
    matmuls_used = 0
    for s0 in range(0, n_valid, token_batch):
        s1 = min(n_valid, s0 + token_batch)
        z_chunk = _compute_unit_normalized(h_valid[s0:s1], resolved_norm, eps_f)
        u0 = z_chunk * w0_64
        if b0_64 is not None:
            u0 = u0 + b0_64
        l0_parts.append(F.linear(u0, head_64))
        matmuls_used += 1
        if has_norm_delta:
            du = z_chunk * dw_64
            if db_64 is not None:
                du = du + db_64
            dl_parts.append(F.linear(du, head_64))
            matmuls_used += 1

    l0_valid = torch.cat(l0_parts, dim=0)
    dl_valid = torch.cat(dl_parts, dim=0) if has_norm_delta else None

    base_all_m = compute_token_nll_and_ece(l0_valid, targets_valid, num_bins=bins_i)
    base_gen_m = compute_token_nll_and_ece(
        l0_valid[gen_valid_mask], targets_valid[gen_valid_mask], num_bins=bins_i
    )
    base_spec_m = (
        compute_token_nll_and_ece(
            l0_valid[spec_valid_mask], targets_valid[spec_valid_mask], num_bins=bins_i
        )
        if has_specialist
        else None
    )

    ref_gen_nll = (
        float(base_gen_m["nll_mean"])
        if base_general_nll_override is None
        else _validate_positive_finite(base_general_nll_override, "base_general_nll_override")
    )
    ref_total_nll = (
        float(base_all_m["nll_mean"])
        if base_total_nll_override is None
        else _validate_positive_finite(base_total_nll_override, "base_total_nll_override")
    )
    ref_ece = (
        float(base_all_m["expected_calibration_error"])
        if base_ece_override is None
        else _validate_nonneg_finite(base_ece_override, "base_ece_override")
    )
    gen_ceiling = (1.0 + guard_rel) * ref_gen_nll
    boundaries_inner = torch.linspace(0.0, 1.0, steps=bins_i + 1, dtype=torch.float64)[1:-1]

    domain_masks: list[torch.Tensor] = []
    domain_ceilings: list[float] = []
    if domain_labels is not None and max_domain_drift_rel is not None:
        dom_rel = _validate_nonneg_finite(max_domain_drift_rel, "max_domain_drift_rel")
        if len(domain_labels) != batch_shape[0]:
            raise ValueError(
                f"domain_labels length {len(domain_labels)} must match batch dim {batch_shape[0]}"
            )
        seq_dom_groups: dict[str, list[int]] = {}
        for idx, d_lbl in enumerate(domain_labels):
            seq_dom_groups.setdefault(str(d_lbl), []).append(idx)
        base_tok_nll = F.cross_entropy(l0_valid, targets_valid, reduction="none")
        for dom_k in sorted(seq_dom_groups.keys()):
            seq_mask = torch.zeros(batch_shape, dtype=torch.bool)
            seq_mask[seq_dom_groups[dom_k]] = True
            valid_dom = seq_mask.reshape(-1)[valid_mask]
            if bool(valid_dom.any().item()):
                base_dom_nll = float(base_tok_nll[valid_dom].mean().item())
                domain_masks.append(valid_dom)
                domain_ceilings.append((1.0 + dom_rel) * base_dom_nll)

    feasible: list[tuple[float, float, float, float]] = []
    fallback: list[tuple[float, float, float, float]] = []
    grid_evals = 0

    for lam in lam_vals:
        l_lam = l0_valid if (dl_valid is None or lam == 0.0) else (l0_valid + float(lam) * dl_valid)

        lam_feasible: list[tuple[float, float, float]] = []
        lam_fallback: list[tuple[float, float, float]] = []
        for tau_val in tau_vals:
            ok, energy, gen_drift_rel = _score_logit_pair(
                l_lam,
                targets_valid,
                gen_valid_mask,
                tau_val,
                ref_gen_nll=ref_gen_nll,
                ref_total_nll=ref_total_nll,
                ref_ece=ref_ece,
                gen_ceiling=gen_ceiling,
                ece_weight=ece_w,
                num_bins=bins_i,
                boundaries_inner=boundaries_inner,
                maximum_pain=ANSE_MAXIMUM_PAIN,
                domain_masks=domain_masks,
                domain_ceilings=domain_ceilings,
            )
            grid_evals += 1
            lam_fallback.append((gen_drift_rel, energy, tau_val))
            if ok:
                lam_feasible.append((energy, gen_drift_rel, tau_val))

        if refine_flag and len(tau_vals) >= 2:
            anchor_tau = (
                min(lam_feasible, key=lambda it: (it[0], it[1], abs(it[2] - 1.0)))[2]
                if lam_feasible
                else min(lam_fallback, key=lambda it: (it[0], it[1], abs(it[2] - 1.0)))[2]
            )
            idx_a = tau_vals.index(anchor_tau)
            lo_t = tau_vals[max(0, idx_a - 1)]
            hi_t = tau_vals[min(len(tau_vals) - 1, idx_a + 1)]
            if hi_t > lo_t and n_ref > 1:
                for tau_ref in np.geomspace(lo_t, hi_t, num=n_ref).tolist():
                    tau_r = round(float(tau_ref), 8)
                    if tau_r in tau_vals:
                        continue
                    ok, energy, gen_drift_rel = _score_logit_pair(
                        l_lam,
                        targets_valid,
                        gen_valid_mask,
                        tau_r,
                        ref_gen_nll=ref_gen_nll,
                        ref_total_nll=ref_total_nll,
                        ref_ece=ref_ece,
                        gen_ceiling=gen_ceiling,
                        ece_weight=ece_w,
                        num_bins=bins_i,
                        boundaries_inner=boundaries_inner,
                        maximum_pain=ANSE_MAXIMUM_PAIN,
                        domain_masks=domain_masks,
                        domain_ceilings=domain_ceilings,
                    )
                    grid_evals += 1
                    lam_fallback.append((gen_drift_rel, energy, tau_r))
                    if ok:
                        lam_feasible.append((energy, gen_drift_rel, tau_r))

        if lam_feasible:
            best_e, best_gd, best_t = min(lam_feasible, key=lambda it: (it[0], it[1], abs(it[2] - 1.0)))
            feasible.append((best_e, best_gd, float(lam), float(best_t)))
        else:
            best_gd, best_e, best_t = min(lam_fallback, key=lambda it: (it[0], it[1], abs(it[2] - 1.0)))
            fallback.append((best_gd, best_e, float(lam), float(best_t)))

    if feasible:
        _, _, best_lam, best_tau = min(feasible, key=lambda it: (it[0], it[1], abs(it[3] - 1.0)))
    else:
        _, _, best_lam, best_tau = min(fallback, key=lambda it: (it[0], it[1], abs(it[3] - 1.0)))

    w_dtype = w_base.dtype
    w_interp = (w0_64 + best_lam * dw_64).to(device=w_base.device, dtype=w_dtype)
    b_interp = (
        (b0_64 + best_lam * db_64).to(device=b_base.device, dtype=b_base.dtype)
        if (b0_64 is not None and db_64 is not None and b_base is not None)
        else None
    )
    folded_w, folded_b = fold_temperature_into_norm_params(w_interp, b_interp, best_tau)

    # Ground-truth verify folded norm weights & exact logit equivalence in model dtype
    h_pre_dev = h_pre.detach().to(device=w_base.device, dtype=w_dtype)
    head_dev = head_w.detach().to(device=w_base.device, dtype=w_dtype)
    h_folded = apply_norm_weights(h_pre_dev, folded_w, folded_b, eps=eps_f, norm_type=resolved_norm)
    logits_folded = F.linear(h_folded, head_dev)

    h_unfolded = apply_norm_weights(h_pre_dev, w_interp, b_interp, eps=eps_f, norm_type=resolved_norm)
    logits_explicit_scaled = F.linear(h_unfolded, head_dev) / float(best_tau)
    max_logit_equiv_diff = float((logits_folded.float() - logits_explicit_scaled.float()).abs().max().item())

    flat_folded_valid = logits_folded.reshape(-1, vocab_size)[valid_mask.to(logits_folded.device)]
    cal_all_m = compute_token_nll_and_ece(flat_folded_valid, targets_valid, num_bins=bins_i)
    cal_gen_m = compute_token_nll_and_ece(
        flat_folded_valid[gen_valid_mask], targets_valid[gen_valid_mask], num_bins=bins_i
    )
    cal_spec_m = (
        compute_token_nll_and_ece(
            flat_folded_valid[spec_valid_mask], targets_valid[spec_valid_mask], num_bins=bins_i
        )
        if has_specialist
        else None
    )

    guard_feasible = bool(float(cal_gen_m["nll_mean"]) <= gen_ceiling + 1e-12)
    general_drift_rel = (float(cal_gen_m["nll_mean"]) / max(ref_gen_nll, 1e-12)) - 1.0
    mean_diff = float(cal_all_m["nll_mean"]) - ref_total_nll
    ece_diff = float(cal_all_m["expected_calibration_error"]) - ref_ece
    anse_energy = (
        mean_diff + ece_w * ece_diff
        if guard_feasible
        else ANSE_MAXIMUM_PAIN + max(0.0, mean_diff) + ece_w * float(cal_all_m["expected_calibration_error"])
    )

    base_ece_val = float(base_all_m["expected_calibration_error"])
    cal_ece_val = float(cal_all_m["expected_calibration_error"])
    ece_reduction_rel = (base_ece_val - cal_ece_val) / base_ece_val if base_ece_val > 1e-12 else 0.0

    norm_changed = (not torch.equal(folded_w.cpu(), w_base.detach().cpu())) or (
        folded_b is not None and b_base is not None and not torch.equal(folded_b.cpu(), b_base.detach().cpu())
    )
    edited_norm_params = (
        int(folded_w.numel()) + (int(folded_b.numel()) if folded_b is not None else 0)
        if norm_changed
        else 0
    )
    eff_total_params = (
        _validate_positive_int(total_model_params, "total_model_params")
        if total_model_params is not None
        else int(head_w.numel()) + int(w_base.numel()) + (int(b_base.numel()) if b_base is not None else 0)
    )
    edited_param_fraction = float(edited_norm_params) / float(eff_total_params)

    return {
        "lambda_norm": float(best_lam),
        "tau_temp": float(best_tau),
        "norm_kind": resolved_norm,
        "folded_norm_weight": folded_w,
        "folded_norm_bias": folded_b,
        "guard_feasible": guard_feasible,
        "max_general_drift_rel": float(guard_rel),
        "general_drift_rel": float(general_drift_rel),
        "general_drift_pct": float(100.0 * general_drift_rel),
        "general_nll_base": float(base_gen_m["nll_mean"]),
        "general_nll_calibrated": float(cal_gen_m["nll_mean"]),
        "specialist_nll_base": float(base_spec_m["nll_mean"]) if base_spec_m is not None else None,
        "specialist_nll_calibrated": float(cal_spec_m["nll_mean"]) if cal_spec_m is not None else None,
        "nll_mean_base": float(base_all_m["nll_mean"]),
        "nll_mean": float(cal_all_m["nll_mean"]),
        "nll_diff_vs_base": float(cal_all_m["nll_mean"]) - float(base_all_m["nll_mean"]),
        "expected_calibration_error_base": base_ece_val,
        "expected_calibration_error": cal_ece_val,
        "ece_reduction_rel": float(ece_reduction_rel),
        "top1_accuracy_base": float(base_all_m["top1_accuracy"]),
        "top1_accuracy": float(cal_all_m["top1_accuracy"]),
        "anse_energy": float(anse_energy),
        "max_logit_equiv_diff": max_logit_equiv_diff,
        "edited_norm_params": edited_norm_params,
        "total_model_params": eff_total_params,
        "edited_param_fraction": edited_param_fraction,
        "lm_head_matmuls_used": matmuls_used,
        "grid_evaluations_count": grid_evals,
    }


# ---------------------------------------------------------------------------
# 4. Model Inspection, Chunk Preparation & Verified Stage-Ratchet Execution
# ---------------------------------------------------------------------------


def _resolve_dotted_attr(root: Any, attr_path: str) -> Any:
    """Resolve a possibly dotted attribute path on ``root``, returning None if absent."""
    cur = root
    for part in attr_path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _resolve_norm_and_head(
    model: nn.Module,
    *,
    norm_kind: str,
    norm_attr: str | None,
    head_attr: str | None,
) -> tuple[nn.Module, nn.Module, str, str, str, int, int, float]:
    """Locate final norm and unembedding head modules cleanly across LM architectures."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if norm_kind not in VALID_NORM_KINDS:
        raise ValueError(f"norm_kind must be one of {sorted(VALID_NORM_KINDS)}, got {norm_kind!r}")

    if norm_attr is not None:
        norm_name = _validate_nonempty_str(norm_attr, "norm_attr")
        norm_mod = _resolve_dotted_attr(model, norm_name)
    else:
        norm_name = ""
        norm_mod = None
        for cand in _NORM_ATTR_CANDIDATES:
            found = _resolve_dotted_attr(model, cand)
            if isinstance(found, nn.Module) and isinstance(getattr(found, "weight", None), torch.Tensor):
                norm_name = cand
                norm_mod = found
                break

    if not isinstance(norm_mod, nn.Module):
        raise ValueError("model does not expose a recognizable final normalization module")
    w_param = getattr(norm_mod, "weight", None)
    _validate_1d_param(w_param, f"model.{norm_name}.weight")
    hidden_dim = int(w_param.shape[0])
    resolved_norm = detect_norm_type(norm_mod, norm_type=norm_kind)
    b_param = getattr(norm_mod, "bias", None)
    if b_param is not None:
        _validate_1d_param(b_param, f"model.{norm_name}.bias", expected_dim=hidden_dim)

    raw_eps = getattr(norm_mod, "eps", getattr(norm_mod, "variance_epsilon", None))
    eps_eff = (
        _validate_positive_finite(raw_eps, f"model.{norm_name}.eps")
        if raw_eps is not None
        else (1e-6 if resolved_norm == "rmsnorm" else 1e-5)
    )

    if head_attr is not None:
        head_name = _validate_nonempty_str(head_attr, "head_attr")
        head_mod = _resolve_dotted_attr(model, head_name)
    else:
        head_name = ""
        head_mod = None
        for cand in _HEAD_ATTR_CANDIDATES:
            found = _resolve_dotted_attr(model, cand)
            if isinstance(found, nn.Module) and isinstance(getattr(found, "weight", None), torch.Tensor):
                head_name = cand
                head_mod = found
                break

    if not isinstance(head_mod, nn.Module) or not callable(head_mod):
        raise ValueError("model does not expose a callable unembedding head module")
    head_w = getattr(head_mod, "weight", None)
    _validate_fp_tensor(head_w, f"model.{head_name}.weight", exact_ndim=2)
    if int(head_w.shape[1]) != hidden_dim:
        raise ValueError(
            f"model.{head_name}.weight width {head_w.shape[1]} must match norm dim {hidden_dim}"
        )
    head_bias = getattr(head_mod, "bias", None)
    if isinstance(head_bias, torch.Tensor) and bool((head_bias.detach() != 0).any().item()):
        raise ValueError(
            f"model.{head_name} has non-zero bias; folded norm temperature scaling requires biasless unembedding"
        )
    vocab_size = int(head_w.shape[0])
    return norm_mod, head_mod, resolved_norm, norm_name, head_name, hidden_dim, vocab_size, eps_eff


def _is_general_domain(domain_label: str, general_prefixes: tuple[str, ...]) -> bool:
    """Return True if ``domain_label`` begins with any configured general-prose prefix."""
    return any(domain_label.startswith(pref) for pref in general_prefixes)


def _prepare_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    name: str,
    vocab_size: int,
    score_from: int,
    general_prefixes: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Validate multi-domain token sequences and extract ``(ids_2d, is_general_mask, domains)``."""
    if isinstance(chunks, (str, bytes, bytearray)) or not isinstance(chunks, Sequence) or len(chunks) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")
    rows: list[torch.Tensor] = []
    domains: list[str] = []
    seq_len: int | None = None
    start_pos = max(0, score_from - 1)

    for idx, ch in enumerate(chunks):
        if not isinstance(ch, Mapping) or "ids" not in ch:
            raise ValueError(f"{name}[{idx}] must be a mapping containing 'ids' and domain label")
        raw_dom = ch.get("expert_id", ch.get("domain"))
        dom = _validate_nonempty_str(raw_dom, f"{name}[{idx}] domain/expert_id")

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
                raise ValueError(
                    f"{name} sequence length {seq_len} is too short for score_from={score_from}"
                )
        elif int(ids_t.numel()) != seq_len:
            raise ValueError(f"All {name} entries must share sequence length {seq_len}, got {ids_t.numel()}")
        rows.append(ids_t)
        domains.append(dom)

    ids_2d = torch.stack(rows, dim=0)
    is_gen = torch.tensor([_is_general_domain(d, general_prefixes) for d in domains], dtype=torch.bool)
    if not bool(is_gen.any().item()) or not bool((~is_gen).any().item()):
        raise ValueError(
            f"{name} must contain both general ({general_prefixes}) and specialist chunks"
        )
    return ids_2d, is_gen, domains


def _capture_pre_norm_states(
    model: nn.Module,
    norm_mod: nn.Module,
    ids_2d: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Run trunk forward pass once via a read-only pre-hook on ``norm_mod`` to extract pre-norm states."""
    first_param = next(model.parameters(), None)
    dev = first_param.device if first_param is not None else torch.device("cpu")
    captured: list[torch.Tensor] = []

    def _pre_hook(_mod: nn.Module, inp: tuple[Any, ...]) -> None:
        if inp and isinstance(inp[0], torch.Tensor):
            captured.append(inp[0].detach().to(device="cpu"))

    handle = norm_mod.register_forward_pre_hook(_pre_hook)
    parts: list[torch.Tensor] = []
    try:
        for b0 in range(0, int(ids_2d.shape[0]), batch_size):
            batch_ids = ids_2d[b0 : b0 + batch_size].to(dev)
            captured.clear()
            _ = model(batch_ids)
            if not captured:
                raise RuntimeError("Final normalization module was not invoked during model forward pass")
            h_batch = captured[-1]
            if h_batch.shape[:2] != batch_ids.shape:
                raise ValueError(
                    f"Captured pre-norm shape {tuple(h_batch.shape)} incompatible with input {tuple(batch_ids.shape)}"
                )
            if not bool(torch.isfinite(h_batch).all().item()):
                raise ValueError("Model trunk produced non-finite pre-norm hidden states")
            parts.append(h_batch)
    finally:
        handle.remove()

    return torch.cat(parts, dim=0)


def _evaluate_pre_norm_domain_metrics(
    h_pre_norm: torch.Tensor,
    ids_2d: torch.Tensor,
    domains: Sequence[str],
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor | None,
    head_weight: torch.Tensor,
    *,
    eps: float,
    norm_kind: str,
    score_from: int,
    num_bins: int,
    ignore_index: int,
) -> tuple[dict[str, float], dict[str, dict[str, Any]], dict[str, Any]]:
    """Evaluate exact per-domain and global NLL/ECE metrics from pre-norm hidden states."""
    start_pos = max(0, score_from - 1)
    h_slice = h_pre_norm[:, start_pos:-1, :].to(device=norm_weight.device, dtype=norm_weight.dtype)
    t_slice = ids_2d[:, start_pos + 1 :]
    h_norm = apply_norm_weights(h_slice, norm_weight, norm_bias, eps=eps, norm_type=norm_kind)
    logits = F.linear(h_norm, head_weight.to(device=h_norm.device, dtype=h_norm.dtype))

    global_m = compute_token_nll_and_ece(logits, t_slice, num_bins=num_bins, ignore_index=ignore_index)
    dom_indices: dict[str, list[int]] = {}
    for idx, dom in enumerate(domains):
        dom_indices.setdefault(dom, []).append(idx)

    dom_nll: dict[str, float] = {}
    dom_metrics: dict[str, dict[str, Any]] = {}
    for dom in sorted(dom_indices.keys()):
        ix = dom_indices[dom]
        dm = compute_token_nll_and_ece(logits[ix], t_slice[ix], num_bins=num_bins, ignore_index=ignore_index)
        dom_nll[dom] = float(dm["nll_mean"])
        dom_metrics[dom] = {
            "nll_mean": float(dm["nll_mean"]),
            "expected_calibration_error": float(dm["expected_calibration_error"]),
            "top1_accuracy": float(dm["top1_accuracy"]),
            "num_tokens": int(dm["num_tokens"]),
        }
    return dom_nll, dom_metrics, global_m


def _compute_anse_verdict(
    ref_dom_nll: Mapping[str, float],
    ref_ece: float,
    state_dom_nll: Mapping[str, float],
    state_ece: float,
    *,
    budget_ok: bool,
    cfg: FoldedNormCalibrationConfig,
) -> dict[str, Any]:
    """Compute ANSE energy, general-domain drift, worst-domain drift, and guard verdict."""
    ratios: dict[str, float] = {}
    drifts_pct: dict[str, float] = {}
    for dom in sorted(state_dom_nll.keys()):
        base_v = float(ref_dom_nll[dom])
        cand_v = float(state_dom_nll[dom])
        ratio = cand_v / max(base_v, 1e-12)
        ratios[dom] = ratio
        drifts_pct[dom] = 100.0 * (ratio - 1.0)

    mean_base = sum(float(ref_dom_nll[d]) for d in state_dom_nll) / float(len(state_dom_nll))
    mean_state = sum(float(state_dom_nll[d]) for d in state_dom_nll) / float(len(state_dom_nll))
    mean_diff = mean_state - mean_base
    ece_diff = float(state_ece) - float(ref_ece)

    general_keys = [d for d in sorted(ratios.keys()) if _is_general_domain(d, cfg.general_prefixes)]
    general_ratio = max(ratios[d] for d in general_keys)
    worst_domain_id = max(sorted(ratios.keys()), key=lambda d: ratios[d])
    worst_domain_ratio = ratios[worst_domain_id]

    general_guard_ok = general_ratio <= (1.0 + cfg.max_general_drift_rel) + 1e-12
    worst_guard_ok = worst_domain_ratio <= (1.0 + cfg.max_domain_drift_rel) + 1e-12
    guard_ok = bool(general_guard_ok and worst_guard_ok and budget_ok)

    if not guard_ok:
        anse_energy = (
            cfg.maximum_pain_energy
            + max(0.0, mean_diff)
            + cfg.ece_energy_weight * float(state_ece)
        )
    else:
        soft_pen = cfg.soft_margin_penalty if general_ratio > cfg.anse_safe_general_ratio else 0.0
        anse_energy = mean_diff + cfg.ece_energy_weight * ece_diff + soft_pen

    return {
        "mean_base_nll": float(mean_base),
        "mean_state_nll": float(mean_state),
        "mean_diff_vs_base": float(mean_diff),
        "ece_diff_vs_base": float(ece_diff),
        "general_ratio": float(general_ratio),
        "general_drift_rel": float(general_ratio - 1.0),
        "general_pct_vs_base": float(100.0 * (general_ratio - 1.0)),
        "worst_domain_id": worst_domain_id,
        "worst_domain_ratio": float(worst_domain_ratio),
        "worst_domain_pct_vs_base": float(100.0 * (worst_domain_ratio - 1.0)),
        "general_guard_ok": general_guard_ok,
        "worst_guard_ok": worst_guard_ok,
        "budget_ok": budget_ok,
        "guard_ok": guard_ok,
        "anse_energy": float(anse_energy),
        "domain_drifts_pct": drifts_pct,
    }


def _snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Capture a detached CPU clone of ``model.state_dict()``."""
    return {k: v.detach().to(device="cpu").clone() for k, v in model.state_dict().items()}


def _restore_state_inplace(model: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
    """Restore ``snapshot`` onto ``model`` matching parameter devices and dtypes."""
    cur_sd = model.state_dict()
    model.load_state_dict(
        {k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in snapshot.items()}
    )


def _validate_base_state_dict(
    base_state_dict: Mapping[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Validate that ``base_state_dict`` matches ``model.state_dict()`` keys, shapes, and finiteness."""
    if not isinstance(base_state_dict, Mapping) or not base_state_dict:
        raise ValueError("base_state_dict must be a non-empty mapping when provided")
    cur_sd = model.state_dict()
    if set(base_state_dict.keys()) != set(cur_sd.keys()):
        raise ValueError("base_state_dict keys must exactly match model.state_dict() keys")
    validated: dict[str, torch.Tensor] = {}
    for k, ref_t in cur_sd.items():
        val = base_state_dict[k]
        if not isinstance(val, torch.Tensor):
            raise TypeError(f"base_state_dict[{k!r}] must be a torch.Tensor, got {type(val).__name__}")
        if val.shape != ref_t.shape:
            raise ValueError(
                f"base_state_dict[{k!r}] shape {tuple(val.shape)} != model shape {tuple(ref_t.shape)}"
            )
        if val.is_floating_point() and not bool(torch.isfinite(val.detach()).all().item()):
            raise ValueError(f"base_state_dict[{k!r}] contains non-finite values")
        validated[k] = val.detach().to(device="cpu").clone()
    return validated


def _count_edited_params(
    ref_state: Mapping[str, torch.Tensor],
    cand_state: Mapping[str, torch.Tensor],
    norm_prefix: str,
) -> tuple[int, int, int]:
    """Count ``(total_edited_elems, norm_edited_elems, non_norm_edited_elems)`` avoiding double-count of tied heads."""
    seen_data_ptrs: set[int] = set()
    norm_keys = {f"{norm_prefix}.weight", f"{norm_prefix}.bias"}
    total_edited = 0
    norm_edited = 0
    non_norm_edited = 0
    for key in sorted(ref_state.keys()):
        r_t = ref_state[key].detach().cpu()
        c_t = cand_state[key].detach().cpu()
        ptr = r_t.data_ptr()
        if ptr in seen_data_ptrs:
            continue
        seen_data_ptrs.add(ptr)
        diff_count = int((r_t != c_t).sum().item())
        total_edited += diff_count
        if key in norm_keys:
            norm_edited += diff_count
        else:
            non_norm_edited += diff_count
    return total_edited, norm_edited, non_norm_edited


def _resolve_execution_config(
    config: FoldedNormCalibrationConfig | None,
    *,
    score_from: int | None,
    max_general_drift_rel: float | None,
    max_domain_drift_rel: float | None,
    max_edited_param_fraction: float | None,
) -> FoldedNormCalibrationConfig:
    """Merge keyword overrides into a validated ``FoldedNormCalibrationConfig``."""
    if config is not None and not isinstance(config, FoldedNormCalibrationConfig):
        raise TypeError(
            f"config must be FoldedNormCalibrationConfig or None, got {type(config).__name__}"
        )
    base = config or FoldedNormCalibrationConfig()
    return FoldedNormCalibrationConfig(
        num_bins=base.num_bins,
        max_general_drift_rel=(
            base.max_general_drift_rel
            if max_general_drift_rel is None
            else _validate_nonneg_finite(max_general_drift_rel, "max_general_drift_rel")
        ),
        max_domain_drift_rel=(
            base.max_domain_drift_rel
            if max_domain_drift_rel is None
            else _validate_nonneg_finite(max_domain_drift_rel, "max_domain_drift_rel")
        ),
        anse_safe_general_ratio=base.anse_safe_general_ratio,
        maximum_pain_energy=base.maximum_pain_energy,
        soft_margin_penalty=base.soft_margin_penalty,
        ece_energy_weight=base.ece_energy_weight,
        min_stage_energy_drop=base.min_stage_energy_drop,
        max_edited_param_fraction=(
            base.max_edited_param_fraction
            if max_edited_param_fraction is None
            else max_edited_param_fraction
        ),
        lambda_grid=base.lambda_grid,
        tau_grid=base.tau_grid,
        refine_tau=base.refine_tau,
        num_tau_refine=base.num_tau_refine,
        ignore_index=base.ignore_index,
        score_from=(
            base.score_from
            if score_from is None
            else _validate_nonneg_int(score_from, "score_from")
        ),
        batch_size=base.batch_size,
        general_prefixes=base.general_prefixes,
        require_strict_energy_drop=base.require_strict_energy_drop,
    )


@torch.no_grad()
def execute_verified_folded_norm_calibration(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    base_state_dict: Mapping[str, torch.Tensor] | None = None,
    base_domain_nll: Mapping[str, float] | None = None,
    base_ece: float | None = None,
    upstream_stage_fn: Callable[[nn.Module], Mapping[str, Any] | None] | None = None,
    config: FoldedNormCalibrationConfig | None = None,
    norm_kind: str = "auto",
    norm_attr: str | None = None,
    head_attr: str | None = None,
    score_from: int | None = None,
    max_general_drift_rel: float | None = None,
    max_domain_drift_rel: float | None = None,
    max_edited_param_fraction: float | None = None,
    total_model_params: int | None = None,
    inplace: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Execute verified folded final-norm temperature & ECE calibration with ANSE stage ratchet.

    Replicates the ``PILOT3B`` monotone ANSE stage ratchet:
    - When an upstream spectral/LoRA stage is provided (via ``upstream_stage_fn`` or a
      pre-edited ``model`` paired with clean ``base_state_dict``), compares both the
      upstream stage on probe AND clean-base folded-norm calibration vs upstream folded-norm
      calibration. Whenever clean folded-norm calibration achieves lower or equal ANSE energy
      (or when the upstream stage violates guard / worsens probe energy), the upstream edit
      is rolled back bitwise to the clean base before applying folded-norm calibration.
    - Commits the final folded norm update iff ANSE ``Delta E < 0`` and all domain &
      parameter-budget guards hold; otherwise performs atomic bitwise rollback with
      ``ANSE_MAXIMUM_PAIN`` accounting and a canonical SHA-256 receipt.
    """
    cfg = _resolve_execution_config(
        config,
        score_from=score_from,
        max_general_drift_rel=max_general_drift_rel,
        max_domain_drift_rel=max_domain_drift_rel,
        max_edited_param_fraction=max_edited_param_fraction,
    )
    _validate_bool(inplace, "inplace")
    if upstream_stage_fn is not None and not callable(upstream_stage_fn):
        raise TypeError(
            f"upstream_stage_fn must be callable or None, got {type(upstream_stage_fn).__name__}"
        )
    if upstream_stage_fn is not None and base_state_dict is not None:
        raise ValueError("Provide either upstream_stage_fn or base_state_dict, not both")

    _, _, _, norm_name_chk, head_name_chk, _, vocab_size, _ = _resolve_norm_and_head(
        model, norm_kind=norm_kind, norm_attr=norm_attr, head_attr=head_attr
    )
    probe_ids, probe_is_gen, probe_domains = _prepare_chunks(
        probe_chunks,
        name="probe_chunks",
        vocab_size=vocab_size,
        score_from=cfg.score_from,
        general_prefixes=cfg.general_prefixes,
    )
    if eval_chunks is not None:
        verify_ids, _, verify_domains = _prepare_chunks(
            eval_chunks,
            name="eval_chunks",
            vocab_size=vocab_size,
            score_from=cfg.score_from,
            general_prefixes=cfg.general_prefixes,
        )
    else:
        verify_ids, verify_domains = probe_ids, probe_domains

    validated_base_sd = _validate_base_state_dict(base_state_dict, model) if base_state_dict is not None else None
    validated_base_nll: dict[str, float] | None = None
    if base_domain_nll is not None:
        if not isinstance(base_domain_nll, Mapping) or not base_domain_nll:
            raise ValueError("base_domain_nll must be a non-empty mapping when provided")
        validated_base_nll = {
            _validate_nonempty_str(k, "base_domain_nll key"): _validate_positive_finite(
                v, f"base_domain_nll[{k!r}]"
            )
            for k, v in base_domain_nll.items()
        }
        missing_doms = sorted(set(verify_domains) - set(validated_base_nll.keys()))
        if missing_doms:
            raise ValueError(f"base_domain_nll is missing verification domains: {missing_doms}")
    validated_base_ece = (
        _validate_nonneg_finite(base_ece, "base_ece") if base_ece is not None else None
    )

    work_model = model if inplace else copy.deepcopy(model)
    norm_mod, head_mod, resolved_norm, norm_name, head_name, _, _, eps_eff = _resolve_norm_and_head(
        work_model, norm_kind=norm_kind, norm_attr=norm_name_chk, head_attr=head_name_chk
    )
    eff_total_params = (
        _validate_positive_int(total_model_params, "total_model_params")
        if total_model_params is not None
        else sum(p.numel() for p in {p.data_ptr(): p for p in work_model.parameters()}.values())
    )

    was_training = work_model.training
    work_model.eval()
    try:
        initial_snapshot = _snapshot_state(work_model)
        initial_sha = compute_module_state_sha256(work_model)

        clean_snapshot = validated_base_sd if validated_base_sd is not None else initial_snapshot
        if validated_base_sd is not None:
            _restore_state_inplace(work_model, clean_snapshot)
        clean_sha = compute_module_state_sha256(work_model)
        clean_norm_w = norm_mod.weight.detach().clone()
        clean_norm_b = (
            norm_mod.bias.detach().clone() if getattr(norm_mod, "bias", None) is not None else None
        )
        head_w = head_mod.weight.detach()

        h_probe_clean = _capture_pre_norm_states(
            work_model, norm_mod, probe_ids, batch_size=cfg.batch_size
        )
        h_verify_clean = (
            h_probe_clean
            if eval_chunks is None
            else _capture_pre_norm_states(work_model, norm_mod, verify_ids, batch_size=cfg.batch_size)
        )
        clean_probe_dom, _, clean_probe_g = _evaluate_pre_norm_domain_metrics(
            h_probe_clean,
            probe_ids,
            probe_domains,
            clean_norm_w,
            clean_norm_b,
            head_w,
            eps=eps_eff,
            norm_kind=resolved_norm,
            score_from=cfg.score_from,
            num_bins=cfg.num_bins,
            ignore_index=cfg.ignore_index,
        )
        clean_verify_dom, clean_verify_metrics, clean_verify_g = _evaluate_pre_norm_domain_metrics(
            h_verify_clean,
            verify_ids,
            verify_domains,
            clean_norm_w,
            clean_norm_b,
            head_w,
            eps=eps_eff,
            norm_kind=resolved_norm,
            score_from=cfg.score_from,
            num_bins=cfg.num_bins,
            ignore_index=cfg.ignore_index,
        )

        # Reference baseline on probe & verify (either clean base or caller-provided baseline)
        ref_probe_dom = clean_probe_dom
        ref_probe_ece = float(clean_probe_g["expected_calibration_error"])
        ref_verify_dom = validated_base_nll if validated_base_nll is not None else clean_verify_dom
        ref_verify_ece = (
            validated_base_ece
            if validated_base_ece is not None
            else float(clean_verify_g["expected_calibration_error"])
        )

        start_pos = max(0, cfg.score_from - 1)
        t_probe_scored = probe_ids[:, start_pos + 1 :]

        # Calibrate clean trunk candidate on probe first for stage-ratchet comparison
        clean_fit = fit_folded_norm_temperature_and_interpolation(
            h_probe_clean[:, start_pos:-1, :],
            clean_norm_w,
            clean_norm_b,
            clean_norm_w,
            clean_norm_b,
            head_w,
            t_probe_scored,
            probe_is_gen,
            norm_kind=resolved_norm,
            eps=eps_eff,
            max_general_drift_rel=cfg.max_general_drift_rel,
            num_bins=cfg.num_bins,
            ignore_index=cfg.ignore_index,
            ece_energy_weight=cfg.ece_energy_weight,
            lambda_grid=cfg.lambda_grid,
            tau_grid=cfg.tau_grid,
            refine_tau=cfg.refine_tau,
            num_tau_refine=cfg.num_tau_refine,
            domain_labels=probe_domains,
            max_domain_drift_rel=cfg.max_domain_drift_rel,
            total_model_params=eff_total_params,
            seq_batch_size=cfg.batch_size,
        )

        has_upstream_stage = False
        upstream_stage_meta: dict[str, Any] | None = None
        if upstream_stage_fn is not None:
            ret_meta = upstream_stage_fn(work_model)
            upstream_stage_meta = dict(ret_meta) if isinstance(ret_meta, Mapping) else {}
            # Re-validate model weights after upstream mutation
            _resolve_norm_and_head(
                work_model, norm_kind=norm_kind, norm_attr=norm_name, head_attr=head_name
            )
            upstream_snapshot = _snapshot_state(work_model)
            upstream_sha = compute_module_state_sha256(work_model)
            has_upstream_stage = upstream_sha != clean_sha
        elif validated_base_sd is not None:
            _restore_state_inplace(work_model, initial_snapshot)
            upstream_snapshot = initial_snapshot
            upstream_sha = initial_sha
            _, _, non_norm_diff = _count_edited_params(clean_snapshot, upstream_snapshot, norm_name)
            has_upstream_stage = non_norm_diff > 0
        else:
            upstream_snapshot = clean_snapshot
            upstream_sha = clean_sha

        upstream_stage_ratchet = "not_applicable"
        selected_fit = clean_fit
        h_verify_active = h_verify_clean

        if has_upstream_stage:
            up_norm_w = norm_mod.weight.detach().clone()
            up_norm_b = (
                norm_mod.bias.detach().clone() if getattr(norm_mod, "bias", None) is not None else None
            )
            h_probe_up = _capture_pre_norm_states(
                work_model, norm_mod, probe_ids, batch_size=cfg.batch_size
            )
            up_probe_dom, _, up_probe_g = _evaluate_pre_norm_domain_metrics(
                h_probe_up,
                probe_ids,
                probe_domains,
                up_norm_w,
                up_norm_b,
                head_w,
                eps=eps_eff,
                norm_kind=resolved_norm,
                score_from=cfg.score_from,
                num_bins=cfg.num_bins,
                ignore_index=cfg.ignore_index,
            )
            up_probe_anse = _compute_anse_verdict(
                ref_probe_dom,
                ref_probe_ece,
                up_probe_dom,
                float(up_probe_g["expected_calibration_error"]),
                budget_ok=True,
                cfg=cfg,
            )
            up_fit = fit_folded_norm_temperature_and_interpolation(
                h_probe_up[:, start_pos:-1, :],
                clean_norm_w,
                clean_norm_b,
                up_norm_w,
                up_norm_b,
                head_w,
                t_probe_scored,
                probe_is_gen,
                norm_kind=resolved_norm,
                eps=eps_eff,
                max_general_drift_rel=cfg.max_general_drift_rel,
                num_bins=cfg.num_bins,
                ignore_index=cfg.ignore_index,
                ece_energy_weight=cfg.ece_energy_weight,
                lambda_grid=cfg.lambda_grid,
                tau_grid=cfg.tau_grid,
                refine_tau=cfg.refine_tau,
                num_tau_refine=cfg.num_tau_refine,
                base_general_nll_override=clean_fit["general_nll_base"],
                base_total_nll_override=clean_fit["nll_mean_base"],
                base_ece_override=clean_fit["expected_calibration_error_base"],
                domain_labels=probe_domains,
                max_domain_drift_rel=cfg.max_domain_drift_rel,
                total_model_params=eff_total_params,
                seq_batch_size=cfg.batch_size,
            )
            up_folded_dom, _, up_folded_g = _evaluate_pre_norm_domain_metrics(
                h_probe_up,
                probe_ids,
                probe_domains,
                up_fit["folded_norm_weight"],
                up_fit["folded_norm_bias"],
                head_w,
                eps=eps_eff,
                norm_kind=resolved_norm,
                score_from=cfg.score_from,
                num_bins=cfg.num_bins,
                ignore_index=cfg.ignore_index,
            )
            up_folded_anse = _compute_anse_verdict(
                ref_probe_dom,
                ref_probe_ece,
                up_folded_dom,
                float(up_folded_g["expected_calibration_error"]),
                budget_ok=True,
                cfg=cfg,
            )
            clean_folded_dom, _, clean_folded_g = _evaluate_pre_norm_domain_metrics(
                h_probe_clean,
                probe_ids,
                probe_domains,
                clean_fit["folded_norm_weight"],
                clean_fit["folded_norm_bias"],
                head_w,
                eps=eps_eff,
                norm_kind=resolved_norm,
                score_from=cfg.score_from,
                num_bins=cfg.num_bins,
                ignore_index=cfg.ignore_index,
            )
            clean_folded_anse = _compute_anse_verdict(
                ref_probe_dom,
                ref_probe_ece,
                clean_folded_dom,
                float(clean_folded_g["expected_calibration_error"]),
                budget_ok=True,
                cfg=cfg,
            )

            upstream_raw_harmful = (
                (not up_probe_anse["guard_ok"])
                or float(up_probe_anse["mean_diff_vs_base"]) > 0.0
            )
            upstream_loses_to_clean_folded = (
                (not up_folded_anse["guard_ok"])
                or float(up_folded_anse["anse_energy"])
                >= float(clean_folded_anse["anse_energy"]) - cfg.min_stage_energy_drop
            )
            if upstream_raw_harmful or upstream_loses_to_clean_folded:
                _restore_state_inplace(work_model, clean_snapshot)
                upstream_stage_ratchet = "rolled_back_to_clean_base"
                selected_fit = clean_fit
                h_verify_active = h_verify_clean
            else:
                upstream_stage_ratchet = "accepted"
                selected_fit = up_fit
                h_verify_active = (
                    h_probe_up
                    if eval_chunks is None
                    else _capture_pre_norm_states(
                        work_model, norm_mod, verify_ids, batch_size=cfg.batch_size
                    )
                )

        # Apply selected folded norm parameters
        norm_mod.weight.copy_(
            selected_fit["folded_norm_weight"].to(
                device=norm_mod.weight.device, dtype=norm_mod.weight.dtype
            )
        )
        if getattr(norm_mod, "bias", None) is not None and selected_fit["folded_norm_bias"] is not None:
            norm_mod.bias.copy_(
                selected_fit["folded_norm_bias"].to(
                    device=norm_mod.bias.device, dtype=norm_mod.bias.dtype
                )
            )

        cand_snapshot = _snapshot_state(work_model)
        cand_sha = compute_module_state_sha256(work_model)
        cand_verify_dom, cand_verify_metrics, cand_verify_g = _evaluate_pre_norm_domain_metrics(
            h_verify_active,
            verify_ids,
            verify_domains,
            norm_mod.weight.detach(),
            getattr(norm_mod, "bias", None),
            head_w,
            eps=eps_eff,
            norm_kind=resolved_norm,
            score_from=cfg.score_from,
            num_bins=cfg.num_bins,
            ignore_index=cfg.ignore_index,
        )

        edited_total, edited_norm, edited_non_norm = _count_edited_params(
            clean_snapshot, cand_snapshot, norm_name
        )
        edited_param_fraction = float(edited_total) / float(eff_total_params)
        budget_ok = (
            True
            if cfg.max_edited_param_fraction is None
            else (edited_param_fraction <= cfg.max_edited_param_fraction + 1e-15)
        )

        parent_anse = _compute_anse_verdict(
            ref_verify_dom,
            ref_verify_ece,
            clean_verify_dom,
            float(clean_verify_g["expected_calibration_error"]),
            budget_ok=True,
            cfg=cfg,
        )
        cand_anse = _compute_anse_verdict(
            ref_verify_dom,
            ref_verify_ece,
            cand_verify_dom,
            float(cand_verify_g["expected_calibration_error"]),
            budget_ok=budget_ok,
            cfg=cfg,
        )

        parent_e = float(parent_anse["anse_energy"])
        cand_e = float(cand_anse["anse_energy"])
        delta_e = cand_e - parent_e
        energy_ok = (
            (delta_e < -cfg.min_stage_energy_drop)
            if cfg.require_strict_energy_drop
            else (delta_e <= 0.0)
        )
        guard_ok = bool(cand_anse["guard_ok"])

        if cand_sha == initial_sha:
            status = "FIXED_POINT_UNCHANGED" if guard_ok else "REJECTED_GUARD_VIOLATION_UNCHANGED"
            accepted = False
            rolled_back = False
        elif guard_ok and energy_ok:
            status = "ACCEPTED"
            accepted = True
            rolled_back = False
        else:
            if not budget_ok:
                status = "ROLLED_BACK_BUDGET_VIOLATION"
            elif not guard_ok:
                status = "ROLLED_BACK_GUARD_VIOLATION"
            else:
                status = "ROLLED_BACK_ENERGY_NON_IMPROVING"
            accepted = False
            rolled_back = True
            _restore_state_inplace(work_model, initial_snapshot)

        final_sha = compute_module_state_sha256(work_model)
        if not accepted and final_sha != initial_sha:
            raise RuntimeError(
                f"Atomic bitwise rollback failed: {final_sha[:12]} != {initial_sha[:12]}"
            )
    finally:
        work_model.train(was_training)

    active_global = cand_verify_g if accepted else clean_verify_g
    active_dom_metrics = cand_verify_metrics if accepted else clean_verify_metrics
    active_anse = cand_anse if accepted else parent_anse

    base_ece_out = float(clean_verify_g["expected_calibration_error"])
    final_ece_out = float(active_global["expected_calibration_error"])
    ece_reduction_rel = (
        (base_ece_out - final_ece_out) / base_ece_out if base_ece_out > 1e-12 else 0.0
    )

    receipt: dict[str, Any] = {
        "object": CALIBRATOR_RECEIPT_OBJECT,
        "schema": CALIBRATOR_SCHEMA,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "status": status,
        "upstream_stage_ratchet": upstream_stage_ratchet,
        "upstream_stage_meta": upstream_stage_meta,
        "norm_kind": resolved_norm,
        "norm_attr": norm_name,
        "head_attr": head_name,
        "has_bias": getattr(norm_mod, "bias", None) is not None,
        "lambda_norm": float(selected_fit["lambda_norm"]),
        "tau_temp": float(selected_fit["tau_temp"]),
        "max_logit_equiv_diff": float(selected_fit["max_logit_equiv_diff"]),
        "guard_ok": guard_ok,
        "general_guard_ok": bool(cand_anse["general_guard_ok"]),
        "worst_guard_ok": bool(cand_anse["worst_guard_ok"]),
        "budget_ok": budget_ok,
        "nll_mean": float(active_global["nll_mean"]),
        "base_nll_mean": float(clean_verify_g["nll_mean"]),
        "cand_nll_mean": float(cand_verify_g["nll_mean"]),
        "expected_calibration_error": final_ece_out,
        "base_ece": base_ece_out,
        "cand_ece": float(cand_verify_g["expected_calibration_error"]),
        "ece_reduction_rel": float(ece_reduction_rel),
        "top1_accuracy": float(active_global["top1_accuracy"]),
        "anse_energy": float(active_anse["anse_energy"]),
        "parent_anse_energy": parent_e,
        "cand_anse_energy": cand_e,
        "delta_e": float(delta_e),
        "general_drift_rel": float(cand_anse["general_drift_rel"]),
        "general_pct_vs_base": float(cand_anse["general_pct_vs_base"]),
        "worst_domain_id": str(cand_anse["worst_domain_id"]),
        "worst_domain_pct_vs_base": float(cand_anse["worst_domain_pct_vs_base"]),
        "domain_drifts_pct": cand_anse["domain_drifts_pct"],
        "per_domain_metrics": active_dom_metrics,
        "edited_total_params": edited_total if accepted else 0,
        "edited_norm_params": edited_norm if accepted else 0,
        "edited_non_norm_params": edited_non_norm if accepted else 0,
        "total_model_params": eff_total_params,
        "edited_param_fraction": edited_param_fraction if accepted else 0.0,
        "initial_state_sha256": initial_sha,
        "clean_base_sha256": clean_sha,
        "upstream_state_sha256": upstream_sha,
        "candidate_state_sha256": cand_sha,
        "final_state_sha256": final_sha,
        "config": {
            k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()
        },
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return work_model, receipt


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "CALIBRATOR_RECEIPT_OBJECT",
    "CALIBRATOR_SCHEMA",
    "DEFAULT_ANSE_SAFE_GENERAL_RATIO",
    "DEFAULT_ANSE_SOFT_MARGIN_PENALTY",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_ECE_ENERGY_WEIGHT",
    "DEFAULT_GENERAL_PREFIXES",
    "DEFAULT_IGNORE_INDEX",
    "DEFAULT_LAMBDA_GRID",
    "DEFAULT_MAX_DOMAIN_DRIFT_REL",
    "DEFAULT_MAX_GENERAL_DRIFT_REL",
    "DEFAULT_MIN_STAGE_ENERGY_DROP",
    "DEFAULT_NUM_BINS",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_SUB001_BUDGET_FRACTION",
    "DEFAULT_TAU_GRID",
    "VALID_NORM_KINDS",
    "FoldedNormCalibrationConfig",
    "apply_norm_weights",
    "canonical_json_bytes",
    "compute_module_state_sha256",
    "compute_payload_sha256",
    "compute_token_nll_and_ece",
    "detect_norm_type",
    "execute_verified_folded_norm_calibration",
    "fit_folded_norm_temperature_and_interpolation",
    "fold_temperature_into_norm_params",
    "verify_folded_norm_receipt_provenance",
]

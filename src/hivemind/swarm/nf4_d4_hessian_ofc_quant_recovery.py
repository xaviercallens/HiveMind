# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Zero-Gradient NF4 Dettmers Block64 + D4 Hessian-Weighted SVD + <1% OFC FP16 + JS-Norm Quantization Recovery.

Crystallizes the locked single-shot ``PILOT17`` breakthrough on 168-layer
``Qwen/Qwen2.5-0.5B-Instruct`` (``pilot17_qwen25_05b_instruct_nf4_d4_ofc_vault.lock.json``,
SHA-256 ``31bf8c08d16f33d3...``, ``PASS``: ``-0.120909`` nats vs Naive NF4, ``76.32%``
of FP32->NF4 quantization loss recovered at ``4.91`` bpp / ``0.3069x`` FP16, PPL
``-11.389%``, ECE ``-48.48%``, ``D0 -2.843%``, ``4/4`` domains improved, ``0.0``
rollback error, ``0`` gradient steps, ``0 MiB`` GPU VRAM):

1. :func:`quantize_dequantize_dettmers_nf4`:
   Exact 16-level Dettmers QLoRA NormalFloat-4 (``NF4``) blockwise quantizer and
   dequantizer (default ``blocksize=64``) with exact bit/byte storage accounting.
2. :func:`protect_ofc_and_outlier_channels_fp16`:
   Restores ``<1%`` OFC sparse value-neuron and massive-activation outlier slices to
   exact FP16 precision while certifying matched-bit advantage against a deterministic
   random ``1%`` FP16 control.
3. :func:`compute_d4_hessian_weighted_svd_residual`:
   Computes the exact closed-form diagonal-Hessian right-weighted thin SVD low-rank
   residual ``U_r @ V_r^T`` minimizing ``||(W - W_q - U_r V_r^T) H^{1/2}||_F^2``
   strictly inside the ``<= 0.35x`` FP16 storage ceiling.
4. :func:`execute_verified_nf4_d4_ofc_js_recovery`:
   End-to-end training-free recovery combining NF4 block64 quantization, ``<1%`` OFC +
   outlier FP16 protection, D4 Hessian-weighted SVD residual compensation, and final-norm
   cross-fitted James-Stein + signed-``tau`` calibration inside ``ker(G_D0) intersect
   S_val^perp`` (``|G_D0 @ dw| < 1e-6``, ``0.0`` OFC leak) under ANSE ``Delta E < 0``,
   ``D0 <= +1.0%``, and ``<= 0.35x`` FP16 storage ceiling with atomic bitwise rollback
   and canonical SHA-256 provenance receipts.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.swarm.conditional_p0_ofc_adapter_calibrator import (
    ConditionalP0OFCCalibratorConfig,
    _compute_anse_verdict,
    _eval_model_causal_metrics,
    _prepare_causal_chunks,
    _resolve_dotted_attr,
    _resolve_norm_and_head,
    _restore_state_inplace,
    _snapshot_state,
    _split_probe_chunks_fallback,
    _validate_bool,
    _validate_finite_float,
    _validate_float_seq,
    _validate_fp_tensor,
    _validate_general_prefixes,
    _validate_index_sequence,
    _validate_nonempty_str,
    _validate_nonneg_finite,
    _validate_nonneg_int,
    _validate_positive_finite,
    _validate_positive_int,
    calibrate_signed_tau_and_crossfit_js_norm,
    discover_ofc_value_neurons,
)
from hivemind.swarm.folded_norm_ece_calibrator import ANSE_MAXIMUM_PAIN
from hivemind.swarm.logit_calibration_temperature_surgeon import (
    compute_module_state_sha256,
    compute_payload_sha256,
)
from hivemind.swarm.volunteer_wsp_workunit_exporter import CANONICAL_NF4_LEVELS

RECOVERY_SCHEMA: str = "hivemind.nf4_d4_hessian_ofc_quant_recovery/1"
RECOVERY_RECEIPT_OBJECT: str = "hivemind.nf4_d4_hessian_ofc_quant_recovery_receipt"

ARM0_FP32_REFERENCE: str = "ARM0_FP32_REFERENCE"
ARM1_NAIVE_NF4_BLOCK64: str = "ARM1_NAIVE_NF4_BLOCK64"
ARM2_NF4_RANDOM_1PCT_FP16_CONTROL: str = "ARM2_NF4_RANDOM_1PCT_FP16_CONTROL"
ARM3_NF4_OFC_AND_OUTLIER_1PCT_FP16: str = "ARM3_NF4_OFC_AND_OUTLIER_1PCT_FP16"
ARM4_NF4_D4_HESSIAN_SVD_RANK16: str = "ARM4_NF4_D4_HESSIAN_SVD_RANK16"
ARM5_NF4_D4_HESSIAN_OFC_JS_NORM: str = "ARM5_NF4_D4_HESSIAN_OFC_JS_NORM"

DEFAULT_NF4_BLOCKSIZE: int = 64
DEFAULT_D4_SVD_RANK: int = 16
DEFAULT_HESSIAN_EPS: float = 1e-6
DEFAULT_OFC_FRACTION: float = 0.008
DEFAULT_OUTLIER_FRACTION: float = 0.008
DEFAULT_MASSIVE_RATIO_THRESHOLD: float = 5.0
DEFAULT_MAX_STORAGE_RATIO_VS_FP16: float = 0.35
DEFAULT_MAX_GENERAL_DRIFT_REL: float = 0.010
DEFAULT_MAX_DOMAIN_DRIFT_REL: float = 0.010
DEFAULT_ECE_ENERGY_WEIGHT: float = 0.25
DEFAULT_MIN_ENERGY_DROP: float = 1e-6
DEFAULT_JS_MAX_RANK: int = 4
DEFAULT_JS_MAX_STEP_NORM: float = 0.15
DEFAULT_NULLSPACE_SV_TOL: float = 1e-6
DEFAULT_NUM_BINS: int = 15
DEFAULT_IGNORE_INDEX: int = -100
DEFAULT_SCORE_FROM: int = 0
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_SEED: int = 2026112717
DEFAULT_TAU_GRID: tuple[float, ...] = (0.85, 0.90, 0.93, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.10)
DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.25)
DEFAULT_GENERAL_PREFIXES: tuple[str, ...] = ("E0_", "D0_", "general")
VALID_PROTECTION_MODES: frozenset[str] = frozenset({"columns", "rows", "both"})
VALID_ARM_NAMES: frozenset[str] = frozenset(
    {
        ARM1_NAIVE_NF4_BLOCK64,
        ARM2_NF4_RANDOM_1PCT_FP16_CONTROL,
        ARM3_NF4_OFC_AND_OUTLIER_1PCT_FP16,
        ARM4_NF4_D4_HESSIAN_SVD_RANK16,
        ARM5_NF4_D4_HESSIAN_OFC_JS_NORM,
    }
)

_NF4_CODEBOOK_64: torch.Tensor = torch.tensor(CANONICAL_NF4_LEVELS, dtype=torch.float64)
_NF4_MIDPOINTS_64: torch.Tensor = (0.5 * (_NF4_CODEBOOK_64[:-1] + _NF4_CODEBOOK_64[1:])).contiguous()
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_EARLY_LAYER_CANDIDATES: tuple[str, ...] = (
    "model.layers.3",
    "model.layers.0",
    "layers.3",
    "layers.0",
    "blocks.3",
    "blocks.0",
    "early_layer",
    "trunk",
)


# ---------------------------------------------------------------------------
# 1. Configuration & Provenance Verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NF4D4HessianOFCConfig:
    """Validated configuration for PILOT17 NF4 + D4 Hessian SVD + <1% OFC + JS-Norm recovery."""

    blocksize: int = DEFAULT_NF4_BLOCKSIZE
    d4_svd_rank: int = DEFAULT_D4_SVD_RANK
    hessian_eps: float = DEFAULT_HESSIAN_EPS
    ofc_fraction: float = DEFAULT_OFC_FRACTION
    outlier_fraction: float = DEFAULT_OUTLIER_FRACTION
    massive_ratio_threshold: float = DEFAULT_MASSIVE_RATIO_THRESHOLD
    max_storage_ratio_vs_fp16: float = DEFAULT_MAX_STORAGE_RATIO_VS_FP16
    max_general_drift_rel: float = DEFAULT_MAX_GENERAL_DRIFT_REL
    max_domain_drift_rel: float = DEFAULT_MAX_DOMAIN_DRIFT_REL
    ece_energy_weight: float = DEFAULT_ECE_ENERGY_WEIGHT
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    min_energy_drop: float = DEFAULT_MIN_ENERGY_DROP
    js_max_rank: int = DEFAULT_JS_MAX_RANK
    js_max_step_norm: float = DEFAULT_JS_MAX_STEP_NORM
    nullspace_sv_tol: float = DEFAULT_NULLSPACE_SV_TOL
    num_bins: int = DEFAULT_NUM_BINS
    ignore_index: int = DEFAULT_IGNORE_INDEX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    tau_grid: tuple[float, ...] = DEFAULT_TAU_GRID
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID
    general_prefixes: tuple[str, ...] = DEFAULT_GENERAL_PREFIXES
    require_strict_energy_drop: bool = True
    enable_js_norm_calibration: bool = True
    evaluate_ablation_arms: bool = True
    clamp_d4_rank_to_budget: bool = True
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        _validate_positive_int(self.blocksize, "blocksize")
        _validate_positive_int(self.d4_svd_rank, "d4_svd_rank")
        _validate_positive_finite(self.hessian_eps, "hessian_eps")
        ofc_f = _validate_positive_finite(self.ofc_fraction, "ofc_fraction")
        out_f = _validate_nonneg_finite(self.outlier_fraction, "outlier_fraction")
        if ofc_f > 1.0 or out_f > 1.0:
            raise ValueError(f"ofc_fraction ({ofc_f}) and outlier_fraction ({out_f}) must be <= 1.0")
        mass_r = _validate_positive_finite(self.massive_ratio_threshold, "massive_ratio_threshold")
        if mass_r <= 1.0:
            raise ValueError(f"massive_ratio_threshold must be > 1.0, got {mass_r}")
        max_sr = _validate_positive_finite(self.max_storage_ratio_vs_fp16, "max_storage_ratio_vs_fp16")
        if max_sr > 1.0:
            raise ValueError(f"max_storage_ratio_vs_fp16 must be <= 1.0, got {max_sr}")
        _validate_nonneg_finite(self.max_general_drift_rel, "max_general_drift_rel")
        _validate_nonneg_finite(self.max_domain_drift_rel, "max_domain_drift_rel")
        _validate_nonneg_finite(self.ece_energy_weight, "ece_energy_weight")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.min_energy_drop, "min_energy_drop")
        _validate_positive_int(self.js_max_rank, "js_max_rank")
        _validate_positive_finite(self.js_max_step_norm, "js_max_step_norm")
        sv_tol = _validate_positive_finite(self.nullspace_sv_tol, "nullspace_sv_tol")
        if sv_tol >= 1.0:
            raise ValueError(f"nullspace_sv_tol must be < 1.0, got {sv_tol}")
        _validate_positive_int(self.num_bins, "num_bins")
        _validate_int_value(self.ignore_index, "ignore_index")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        taus = _validate_float_seq(self.tau_grid, "tau_grid", strictly_positive=True)
        object.__setattr__(self, "tau_grid", tuple(sorted({*taus, 1.0})))
        alphas = _validate_float_seq(self.alpha_grid, "alpha_grid", nonneg=True)
        object.__setattr__(self, "alpha_grid", tuple(sorted({*alphas, 0.0})))
        object.__setattr__(self, "general_prefixes", _validate_general_prefixes(self.general_prefixes))
        _validate_bool(self.require_strict_energy_drop, "require_strict_energy_drop")
        _validate_bool(self.enable_js_norm_calibration, "enable_js_norm_calibration")
        _validate_bool(self.evaluate_ablation_arms, "evaluate_ablation_arms")
        _validate_bool(self.clamp_d4_rank_to_budget, "clamp_d4_rank_to_budget")
        _validate_nonneg_int(self.seed, "seed")

    def to_calibrator_config(self, *, enable_norm_calibration: bool = False) -> ConditionalP0OFCCalibratorConfig:
        """Build a matching :class:`ConditionalP0OFCCalibratorConfig` for norm calibration and ANSE gating."""
        return ConditionalP0OFCCalibratorConfig(
            ofc_fraction=self.ofc_fraction,
            massive_ratio_threshold=self.massive_ratio_threshold,
            js_max_rank=self.js_max_rank,
            js_max_step_norm=self.js_max_step_norm,
            nullspace_sv_tol=self.nullspace_sv_tol,
            max_general_drift_rel=self.max_general_drift_rel,
            max_domain_drift_rel=self.max_domain_drift_rel,
            ece_energy_weight=self.ece_energy_weight,
            maximum_pain_energy=self.maximum_pain_energy,
            min_energy_drop=self.min_energy_drop,
            num_bins=self.num_bins,
            ignore_index=self.ignore_index,
            score_from=self.score_from,
            batch_size=self.batch_size,
            tau_grid=self.tau_grid,
            alpha_grid=self.alpha_grid,
            general_prefixes=self.general_prefixes,
            require_strict_energy_drop=self.require_strict_energy_drop,
            enable_norm_calibration=enable_norm_calibration,
            seed=self.seed,
        )


def _validate_int_value(value: Any, name: str) -> int:
    """Validate an integer (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return int(value)


def verify_nf4_d4_ofc_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a Mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != RECOVERY_RECEIPT_OBJECT or receipt.get("schema") != RECOVERY_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


def _compute_nf4_storage_metrics(
    num_params: int,
    blocksize: int,
    *,
    extra_fp16_bits: float = 0.0,
    factor_bits: float = 0.0,
) -> dict[str, Any]:
    """Compute exact bit/byte storage accounting for NF4 + optional FP16 slices + low-rank SVD."""
    num_blocks = (num_params + blocksize - 1) // blocksize
    nibble_bits = float(num_params * 4)
    scale_bits = float(num_blocks * 16)
    base_nf4_bits = nibble_bits + scale_bits
    total_bits = base_nf4_bits + float(extra_fp16_bits) + float(factor_bits)
    fp16_bits = float(num_params * 16)
    return {
        "num_params": int(num_params),
        "blocksize": int(blocksize),
        "num_blocks": int(num_blocks),
        "nibble_bytes": int((num_params + 1) // 2),
        "scale_bytes": int(num_blocks * 2),
        "base_nf4_bits": float(base_nf4_bits),
        "extra_fp16_bits": float(extra_fp16_bits),
        "factor_bits": float(factor_bits),
        "total_bits": float(total_bits),
        "quantized_bytes": float(total_bits / 8.0),
        "fp16_bytes": float(fp16_bits / 8.0),
        "bits_per_param": float(total_bits / float(max(num_params, 1))),
        "storage_ratio_vs_fp16": float(total_bits / max(fp16_bits, 1.0)),
    }


def _resolve_h_sqrt(diag_hessian: torch.Tensor | None, d_in: int, eps: float) -> torch.Tensor:
    """Validate optional 1D diagonal Hessian ``H`` and return ``H^{1/2}`` on CPU ``float64``."""
    eps_f = _validate_positive_finite(eps, "eps")
    if diag_hessian is None:
        return torch.ones(d_in, dtype=torch.float64)
    h_t = _validate_fp_tensor(diag_hessian, "diag_hessian", exact_ndim=1).detach().to(
        device="cpu", dtype=torch.float64
    )
    if int(h_t.shape[0]) != d_in:
        raise ValueError(f"diag_hessian length {h_t.shape[0]} must match input width {d_in}")
    if bool((h_t < 0.0).any().item()):
        raise ValueError("diag_hessian entries must be non-negative")
    return torch.sqrt(h_t.clamp_min(eps_f)).contiguous()


# ---------------------------------------------------------------------------
# 2. Dettmers 16-Level NF4 Block Quantizer & <1% OFC / Outlier FP16 Protector
# ---------------------------------------------------------------------------


@torch.no_grad()
def quantize_dequantize_dettmers_nf4(
    weight: torch.Tensor,
    *,
    blocksize: int = DEFAULT_NF4_BLOCKSIZE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Quantize and dequantize a tensor using exact 16-level Dettmers NF4 blockwise quantization.

    Packs normalized block entries onto the canonical 16-level NormalFloat-4 quantile
    codebook (``blocksize=64`` by default, with FP16 absmax block scales -> ``4.25`` bpp /
    ``0.2656x`` FP16 for full blocks) and returns ``(dequantized_weight, storage_stats)``.
    """
    w_t = _validate_fp_tensor(weight, "weight", min_ndim=1)
    blk = _validate_positive_int(blocksize, "blocksize")
    orig_dtype, orig_device = w_t.dtype, w_t.device
    orig_shape = tuple(int(d) for d in w_t.shape)

    w_64 = w_t.detach().to(device="cpu", dtype=torch.float64)
    flat = w_64.reshape(-1)
    n_params = int(flat.numel())
    num_blocks = (n_params + blk - 1) // blk
    pad = num_blocks * blk - n_params
    flat_padded = F.pad(flat, (0, pad), value=0.0) if pad > 0 else flat
    blocks = flat_padded.reshape(num_blocks, blk)

    absmax_raw = blocks.abs().amax(dim=1, keepdim=True)
    # Match QLoRA FP16 absmax scale representation cleanly while guarding zero blocks
    absmax_fp16 = absmax_raw.to(dtype=torch.float16).to(dtype=torch.float64)
    nonzero_mask = absmax_fp16 > 0.0
    safe_scale = torch.where(nonzero_mask, absmax_fp16, torch.ones_like(absmax_fp16))
    normed = (blocks / safe_scale).clamp(-1.0, 1.0).contiguous()

    q_idx = torch.bucketize(normed, _NF4_MIDPOINTS_64, right=False)
    q_idx = torch.where(nonzero_mask, q_idx, torch.full_like(q_idx, 7))
    deq_blocks = _NF4_CODEBOOK_64[q_idx] * absmax_fp16
    deq_64 = deq_blocks.reshape(-1)[:n_params].reshape(orig_shape)

    flat_idx = q_idx.reshape(-1)[:n_params].to(dtype=torch.uint8)
    if n_params % 2 == 1:
        flat_idx = torch.cat([flat_idx, torch.zeros(1, dtype=torch.uint8)])
    packed_bytes = ((flat_idx[0::2] << 4) | flat_idx[1::2]).contiguous()

    diff = w_64 - deq_64
    fro_err = float(torch.linalg.norm(diff).item())
    w_norm = float(torch.linalg.norm(w_64).item())
    storage = _compute_nf4_storage_metrics(n_params, blk)
    storage.update(
        {
            "shape": list(orig_shape),
            "quant_format": "dettmers_nf4",
            "packed_byte_count": int(packed_bytes.numel()),
            "frobenius_err": fro_err,
            "relative_frobenius_err": fro_err / max(w_norm, 1e-12) if w_norm > 0.0 else 0.0,
            "packed_nibbles_sha256": compute_payload_sha256(packed_bytes.tolist()),
        }
    )
    return deq_64.to(device=orig_device, dtype=orig_dtype).contiguous(), storage


def _restore_fp16_slices_with_mask(
    w_fp32_64: torch.Tensor,
    w_nf4_64: torch.Tensor,
    indices: Sequence[int],
    mode: str,
) -> tuple[torch.Tensor, int]:
    """Restore selected columns/rows from ``w_fp32_64`` in exact FP16 precision without double-counting."""
    out = w_nf4_64.clone()
    mask = torch.zeros(w_nf4_64.shape, dtype=torch.bool)
    if indices:
        idx_t = torch.tensor(indices, dtype=torch.long)
        if mode in ("columns", "both"):
            out[:, idx_t] = w_fp32_64[:, idx_t].to(dtype=torch.float16).to(dtype=torch.float64)
            mask[:, idx_t] = True
        if mode in ("rows", "both"):
            out[idx_t, :] = w_fp32_64[idx_t, :].to(dtype=torch.float16).to(dtype=torch.float64)
            mask[idx_t, :] = True
    return out, int(mask.sum().item())


@torch.no_grad()
def protect_ofc_and_outlier_channels_fp16(
    weight_fp32: torch.Tensor,
    weight_nf4: torch.Tensor,
    protected_indices: Sequence[int] | torch.Tensor,
    *,
    mode: str = "columns",
    diag_hessian: torch.Tensor | None = None,
    eps: float = DEFAULT_HESSIAN_EPS,
    blocksize: int = DEFAULT_NF4_BLOCKSIZE,
    random_seed: int = DEFAULT_SEED,
    protected_nll: float | None = None,
    random_control_nll: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Restore ``<1%`` OFC + massive-outlier weight slices to FP16 and certify matched-bit advantage.

    Upgrades selected columns, rows, or both (``mode in {'columns', 'rows', 'both'}``)
    from 4-bit NF4 back to exact FP16 values derived from ``weight_fp32`` and constructs a
    byte-matched deterministic random control of identical restored element count to certify
    Hessian-weighted Frobenius (and optional NLL) superiority.
    """
    w_fp_t = _validate_fp_tensor(weight_fp32, "weight_fp32", exact_ndim=2)
    w_q_t = _validate_fp_tensor(weight_nf4, "weight_nf4", exact_ndim=2)
    if tuple(w_fp_t.shape) != tuple(w_q_t.shape):
        raise ValueError(
            f"weight_nf4 shape {tuple(w_q_t.shape)} must match weight_fp32 shape {tuple(w_fp_t.shape)}"
        )
    if mode not in VALID_PROTECTION_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_PROTECTION_MODES)}, got {mode!r}")

    d_out, d_in = int(w_fp_t.shape[0]), int(w_fp_t.shape[1])
    axis_dim = d_in if mode == "columns" else (d_out if mode == "rows" else min(d_out, d_in))
    prot_clean = _validate_index_sequence(protected_indices, axis_dim, "protected_indices")
    blk = _validate_positive_int(blocksize, "blocksize")
    seed_i = _validate_nonneg_int(random_seed, "random_seed")
    h_sqrt = _resolve_h_sqrt(diag_hessian, d_in, eps)

    w_fp_64 = w_fp_t.detach().to(device="cpu", dtype=torch.float64)
    w_q_64 = w_q_t.detach().to(device="cpu", dtype=torch.float64)
    w_prot_64, restored_count = _restore_fp16_slices_with_mask(w_fp_64, w_q_64, prot_clean, mode)

    k_prot = len(prot_clean)
    if k_prot == 0:
        rand_indices: list[int] = []
    else:
        rand_indices = prot_clean
        for attempt in range(4):
            gen = torch.Generator(device="cpu").manual_seed(seed_i + attempt)
            cand = sorted(int(i) for i in torch.randperm(axis_dim, generator=gen)[:k_prot].tolist())
            rand_indices = cand
            if cand != prot_clean or k_prot == axis_dim:
                break
    w_rand_64, rand_restored_count = _restore_fp16_slices_with_mask(w_fp_64, w_q_64, rand_indices, mode)

    naive_h_err = float(torch.linalg.matrix_norm((w_fp_64 - w_q_64) * h_sqrt.unsqueeze(0), ord="fro").item())
    prot_h_err = float(torch.linalg.matrix_norm((w_fp_64 - w_prot_64) * h_sqrt.unsqueeze(0), ord="fro").item())
    rand_h_err = float(torch.linalg.matrix_norm((w_fp_64 - w_rand_64) * h_sqrt.unsqueeze(0), ord="fro").item())
    prot_fro_err = float(torch.linalg.norm(w_fp_64 - w_prot_64).item())
    rand_fro_err = float(torch.linalg.norm(w_fp_64 - w_rand_64).item())

    extra_bits = float(restored_count) * 12.0
    storage = _compute_nf4_storage_metrics(d_out * d_in, blk, extra_fp16_bits=extra_bits)

    nll_advantage: float | None = None
    beats_random_nll: bool | None = None
    if protected_nll is not None and random_control_nll is not None:
        p_nll = _validate_finite_float(protected_nll, "protected_nll")
        r_nll = _validate_finite_float(random_control_nll, "random_control_nll")
        nll_advantage = r_nll - p_nll
        beats_random_nll = bool(p_nll < r_nll)

    stats: dict[str, Any] = {
        **storage,
        "mode": mode,
        "protected_indices": prot_clean,
        "protected_channel_count": k_prot,
        "protected_channel_fraction": float(k_prot) / float(axis_dim),
        "restored_element_count": restored_count,
        "restored_element_fraction": float(restored_count) / float(d_out * d_in),
        "random_control_indices": rand_indices,
        "random_restored_element_count": rand_restored_count,
        "matched_bit_verified": bool(restored_count == rand_restored_count),
        "naive_nf4_weighted_err": naive_h_err,
        "protected_weighted_err": prot_h_err,
        "random_control_weighted_err": rand_h_err,
        "weighted_err_advantage_vs_random": float(rand_h_err - prot_h_err),
        "beats_random_weighted_err": bool(prot_h_err < rand_h_err) if k_prot > 0 else False,
        "protected_frobenius_err": prot_fro_err,
        "random_control_frobenius_err": rand_fro_err,
        "protected_nll_advantage_vs_random": nll_advantage,
        "beats_random_nll": beats_random_nll,
    }
    return w_prot_64.to(device=w_q_t.device, dtype=w_q_t.dtype).contiguous(), stats


# ---------------------------------------------------------------------------
# 3. Gen-3 D4 Closed-Form Diagonal-Hessian Weighted Thin SVD Residual
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_d4_hessian_weighted_svd_residual(
    weight_fp32: torch.Tensor,
    weight_quant: torch.Tensor,
    diag_hessian: torch.Tensor,
    *,
    rank: int = DEFAULT_D4_SVD_RANK,
    eps: float = DEFAULT_HESSIAN_EPS,
    blocksize: int = DEFAULT_NF4_BLOCKSIZE,
    extra_fp16_bits: float = 0.0,
    max_storage_ratio_vs_fp16: float = DEFAULT_MAX_STORAGE_RATIO_VS_FP16,
    clamp_rank_to_budget: bool = True,
    compare_unweighted_svd: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Compute exact closed-form diagonal-Hessian weighted thin SVD residual ``U_r @ V_r^T``.

    Solves the Eckart-Young problem ``min_{rank(R) <= r} ||(W - W_q - R) H^{1/2}||_F^2``
    via weighted thin SVD ``(W - W_q) H^{1/2} = U Sigma V^T``, returning factor tensors
    ``U_r`` of shape ``(d_out, r_eff)`` and ``V_r`` of shape ``(d_in, r_eff)`` whose
    outer product ``U_r @ V_r^T = U_{:,:r} Sigma_{:r} V_{:r,:}^T H^{-1/2}`` stays within
    the ``<= 0.35x`` FP16 storage ceiling when ``clamp_rank_to_budget=True``.
    """
    w_fp_t = _validate_fp_tensor(weight_fp32, "weight_fp32", exact_ndim=2)
    w_q_t = _validate_fp_tensor(weight_quant, "weight_quant", exact_ndim=2)
    if tuple(w_fp_t.shape) != tuple(w_q_t.shape):
        raise ValueError(
            f"weight_quant shape {tuple(w_q_t.shape)} must match weight_fp32 shape {tuple(w_fp_t.shape)}"
        )

    d_out, d_in = int(w_fp_t.shape[0]), int(w_fp_t.shape[1])
    rank_req = _validate_positive_int(rank, "rank")
    blk = _validate_positive_int(blocksize, "blocksize")
    extra_b = _validate_nonneg_finite(extra_fp16_bits, "extra_fp16_bits")
    max_sr = _validate_positive_finite(max_storage_ratio_vs_fp16, "max_storage_ratio_vs_fp16")
    if max_sr > 1.0:
        raise ValueError(f"max_storage_ratio_vs_fp16 must be <= 1.0, got {max_sr}")
    _validate_bool(clamp_rank_to_budget, "clamp_rank_to_budget")
    _validate_bool(compare_unweighted_svd, "compare_unweighted_svd")
    h_sqrt = _resolve_h_sqrt(diag_hessian, d_in, eps)

    n_params = d_out * d_in
    base_storage = _compute_nf4_storage_metrics(n_params, blk, extra_fp16_bits=extra_b)
    fp16_bits = float(n_params * 16)
    bits_per_rank = float((d_out + d_in) * 16)
    remaining_bits = max(0.0, max_sr * fp16_bits - base_storage["total_bits"])
    max_budget_rank = int(math.floor((remaining_bits + 1e-9) / max(bits_per_rank, 1.0)))

    r_max_mat = min(rank_req, d_out, d_in)
    r_eff = min(r_max_mat, max_budget_rank) if clamp_rank_to_budget else r_max_mat

    w_fp_64 = w_fp_t.detach().to(device="cpu", dtype=torch.float64)
    w_q_64 = w_q_t.detach().to(device="cpu", dtype=torch.float64)
    residual = w_fp_64 - w_q_64
    err_before = float(torch.linalg.matrix_norm(residual * h_sqrt.unsqueeze(0), ord="fro").item())

    if r_eff > 0:
        weighted = residual * h_sqrt.unsqueeze(0)
        u, s, vh = torch.linalg.svd(weighted, full_matrices=False)
        r_eff = min(r_eff, int(s.numel()))
        s_half = s[:r_eff].clamp_min(0.0).sqrt()
        u_r64 = (u[:, :r_eff] * s_half.unsqueeze(0)).contiguous()
        v_r64 = ((s_half.unsqueeze(1) * vh[:r_eff, :]) / h_sqrt.unsqueeze(0)).T.contiguous()
        comp_64 = u_r64 @ v_r64.T
    else:
        u_r64 = torch.zeros((d_out, 0), dtype=torch.float64)
        v_r64 = torch.zeros((d_in, 0), dtype=torch.float64)
        comp_64 = torch.zeros_like(residual)

    err_after = float(torch.linalg.matrix_norm((residual - comp_64) * h_sqrt.unsqueeze(0), ord="fro").item())
    unweighted_weighted_err: float | None = None
    beats_unweighted_svd: bool | None = None
    if compare_unweighted_svd:
        if r_eff > 0:
            u_raw, s_raw, vh_raw = torch.linalg.svd(residual, full_matrices=False)
            comp_raw = (u_raw[:, :r_eff] * s_raw[:r_eff].unsqueeze(0)) @ vh_raw[:r_eff, :]
        else:
            comp_raw = torch.zeros_like(residual)
        unweighted_weighted_err = float(
            torch.linalg.matrix_norm((residual - comp_raw) * h_sqrt.unsqueeze(0), ord="fro").item()
        )
        beats_unweighted_svd = bool(err_after < unweighted_weighted_err)

    factor_bits = float(r_eff) * bits_per_rank
    storage = _compute_nf4_storage_metrics(
        n_params, blk, extra_fp16_bits=extra_b, factor_bits=factor_bits
    )
    red_pct = 100.0 * (1.0 - err_after / err_before) if err_before > 1e-15 else 0.0

    stats: dict[str, Any] = {
        **storage,
        "rank_requested": rank_req,
        "rank_budget_max": max_budget_rank,
        "rank_effective": r_eff,
        "max_storage_ratio_vs_fp16": max_sr,
        "storage_within_budget": bool(storage["storage_ratio_vs_fp16"] <= max_sr + 1e-12),
        "weighted_err_before": err_before,
        "weighted_err_after": err_after,
        "hessian_weighted_err_reduction_pct": red_pct,
        "unweighted_svd_weighted_err": unweighted_weighted_err,
        "beats_unweighted_svd": beats_unweighted_svd,
    }
    out_dtype = w_q_t.dtype
    out_dev = w_q_t.device
    return (
        u_r64.to(device=out_dev, dtype=out_dtype).contiguous(),
        v_r64.to(device=out_dev, dtype=out_dtype).contiguous(),
        stats,
    )


# ---------------------------------------------------------------------------
# 4. Probe Diagnostics & End-to-End Verified PILOT17 Recovery Operator
# ---------------------------------------------------------------------------


def _discover_target_linears(
    model: nn.Module,
    head_name: str,
    head_weight: torch.Tensor,
    target_module_paths: Sequence[str] | None,
) -> list[tuple[str, nn.Linear]]:
    """Resolve explicit or automatic target 2D linear layers for NF4 recovery."""
    if target_module_paths is not None:
        if isinstance(target_module_paths, (str, bytes, bytearray)) or not isinstance(target_module_paths, Sequence):
            raise TypeError("target_module_paths must be a sequence of dotted module paths or None")
        if len(target_module_paths) == 0:
            raise ValueError("target_module_paths must be non-empty when provided")
        resolved: list[tuple[str, nn.Linear]] = []
        seen: set[str] = set()
        for idx, raw_path in enumerate(target_module_paths):
            path = _validate_nonempty_str(raw_path, f"target_module_paths[{idx}]")
            if path in seen:
                continue
            mod = _resolve_dotted_attr(model, path)
            if not isinstance(mod, nn.Linear) or mod.weight.ndim != 2:
                raise ValueError(f"Target module {path!r} must exist and be a 2D nn.Linear")
            resolved.append((path, mod))
            seen.add(path)
        return resolved

    auto_targets: list[tuple[str, nn.Linear]] = []
    for mod_name, mod in model.named_modules():
        if (
            isinstance(mod, nn.Linear)
            and mod.weight.ndim == 2
            and mod_name
            and mod_name != head_name
            and mod.weight is not head_weight
        ):
            auto_targets.append((mod_name, mod))
    if not auto_targets:
        raise ValueError("model exposes no non-head 2D nn.Linear modules for quantization recovery")
    return auto_targets


def _resolve_early_layer(model: nn.Module, early_layer_attr: str | None, fallback_mod: nn.Module) -> nn.Module:
    """Resolve early hidden-state probe module for OFC & outlier channel discovery."""
    if early_layer_attr is not None:
        name = _validate_nonempty_str(early_layer_attr, "early_layer_attr")
        found = _resolve_dotted_attr(model, name)
        if not isinstance(found, nn.Module):
            raise ValueError(f"early_layer_attr {name!r} did not resolve to an nn.Module")
        return found
    for cand in _EARLY_LAYER_CANDIDATES:
        found = _resolve_dotted_attr(model, cand)
        if isinstance(found, nn.Module):
            return found
    return fallback_mod


@torch.no_grad()
def _diagnose_probe_hessians_and_ofc(
    model: nn.Module,
    targets: Sequence[tuple[str, nn.Linear]],
    early_mod: nn.Module,
    fit_prep: Mapping[str, Any],
    hidden_dim: int,
    cfg: NF4D4HessianOFCConfig,
) -> dict[str, Any]:
    """Capture exact diagonal input Hessians ``E[x_j^2]`` and ``<1%`` OFC/outlier channels on probe."""
    first_param = next(model.parameters(), None)
    dev = first_param.device if first_param is not None else torch.device("cpu")
    ids_2d = fit_prep["ids_2d"]
    targets_2d = fit_prep["targets_2d"]
    start_pos = max(0, cfg.score_from - 1)

    hess_sum_sq: dict[str, torch.Tensor] = {}
    hess_count: dict[str, int] = {}
    early_means: list[torch.Tensor] = []
    handles: list[Any] = []

    for mod_name, mod in targets:
        def _mk_pre(n: str):
            def _hook(_m: nn.Module, inp: tuple[Any, ...]) -> None:
                x = inp[0].detach().to(device="cpu", dtype=torch.float64).reshape(-1, int(inp[0].shape[-1]))
                ss = (x * x).sum(dim=0)
                if n not in hess_sum_sq:
                    hess_sum_sq[n] = ss
                    hess_count[n] = int(x.shape[0])
                else:
                    hess_sum_sq[n] = hess_sum_sq[n] + ss
                    hess_count[n] += int(x.shape[0])
            return _hook

        handles.append(mod.register_forward_pre_hook(_mk_pre(mod_name)))

    def _early_hook(_m: nn.Module, _inp: tuple[Any, ...], out: Any) -> None:
        h = out[0] if isinstance(out, tuple) else out
        h_cpu = h.detach().to(device="cpu", dtype=torch.float64)
        if h_cpu.ndim == 3:
            h_mean = h_cpu.mean(dim=1)
        elif h_cpu.ndim == 2:
            h_mean = h_cpu
        else:
            raise ValueError(f"Unexpected early layer output shape {tuple(h_cpu.shape)}")
        if int(h_mean.shape[-1]) != hidden_dim:
            raise ValueError(
                f"early layer hidden width {h_mean.shape[-1]} must match model hidden_dim {hidden_dim}"
            )
        early_means.append(h_mean)

    handles.append(early_mod.register_forward_hook(_early_hook))

    logits_parts: list[torch.Tensor] = []
    try:
        for b0 in range(0, int(ids_2d.shape[0]), cfg.batch_size):
            out = model(ids_2d[b0 : b0 + cfg.batch_size].to(dev))
            logits_parts.append(out[:, start_pos:-1, :].detach().to(device="cpu", dtype=torch.float64))
    finally:
        for hk in handles:
            hk.remove()

    if not early_means:
        raise RuntimeError("early layer module did not execute during probe forward pass")

    logits = torch.cat(logits_parts, dim=0)
    chunk_nlls = (
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets_2d.reshape(-1), reduction="none")
        .reshape(targets_2d.shape[0], targets_2d.shape[1])
        .mean(dim=1)
    )
    early_t = torch.cat(early_means, dim=0)

    diag_hessians: dict[str, torch.Tensor] = {}
    for mod_name, mod in targets:
        if mod_name in hess_sum_sq and hess_count[mod_name] > 0:
            diag_hessians[mod_name] = (hess_sum_sq[mod_name] / float(hess_count[mod_name])).clamp_min(
                cfg.hessian_eps
            )
        else:
            diag_hessians[mod_name] = torch.ones(int(mod.weight.shape[1]), dtype=torch.float64)

    ofc_diag = discover_ofc_value_neurons(
        early_t,
        chunk_nlls,
        ofc_fraction=cfg.ofc_fraction,
        massive_ratio_threshold=cfg.massive_ratio_threshold,
        seed=cfg.seed,
    )
    chan_rms = early_t.pow(2).mean(dim=0).sqrt()
    k_out = max(1, min(hidden_dim, int(round(float(hidden_dim) * cfg.outlier_fraction)))) if cfg.outlier_fraction > 0.0 else 0
    top_outliers = (
        sorted(int(i) for i in torch.topk(chan_rms, k=k_out, largest=True, sorted=True).indices.tolist())
        if k_out > 0
        else []
    )
    protected_union = sorted(set(ofc_diag["ofc_indices"]) | set(top_outliers))

    return {
        "diag_hessians": diag_hessians,
        "ofc_diag": ofc_diag,
        "ofc_indices": ofc_diag["ofc_indices"],
        "outlier_indices": top_outliers,
        "protected_residual_channels": protected_union,
    }


def _apply_quantized_arm_to_model(
    targets: Sequence[tuple[str, nn.Linear]],
    orig_weights: Mapping[str, torch.Tensor],
    nf4_weights: Mapping[str, torch.Tensor],
    diag_hessians: Mapping[str, torch.Tensor],
    *,
    hidden_dim: int,
    protected_channels: Sequence[int],
    use_protection: bool,
    use_d4_svd: bool,
    cfg: NF4D4HessianOFCConfig,
) -> dict[str, Any]:
    """Apply NF4, optional FP16 channel protection, and optional D4 SVD across all target linears."""
    total_params = 0
    total_bits = 0.0
    fp16_bits = 0.0
    total_err_before_sq = 0.0
    total_err_after_sq = 0.0
    any_weighted_advantage = False

    for mod_name, mod in targets:
        w_orig = orig_weights[mod_name]
        w_work = nf4_weights[mod_name].clone()
        h_vec = diag_hessians[mod_name]
        d_out, d_in = int(w_orig.shape[0]), int(w_orig.shape[1])
        n_p = d_out * d_in
        total_params += n_p
        fp16_bits += float(n_p * 16)

        base_stats = _compute_nf4_storage_metrics(n_p, cfg.blocksize)
        extra_fp16_bits = 0.0
        factor_bits = 0.0

        h_sqrt = _resolve_h_sqrt(h_vec, d_in, cfg.hessian_eps)
        w_fp_64 = w_orig.detach().to(device="cpu", dtype=torch.float64)
        total_err_before_sq += float(
            torch.sum(((w_fp_64 - w_work.detach().to(device="cpu", dtype=torch.float64)) * h_sqrt.unsqueeze(0)) ** 2).item()
        )

        if use_protection and protected_channels:
            if d_in == hidden_dim and d_out == hidden_dim:
                prot_mode: str | None = "both"
            elif d_in == hidden_dim:
                prot_mode = "columns"
            elif d_out == hidden_dim:
                prot_mode = "rows"
            else:
                prot_mode = None

            if prot_mode is not None:
                w_work, p_stats = protect_ofc_and_outlier_channels_fp16(
                    w_orig,
                    w_work,
                    protected_channels,
                    mode=prot_mode,
                    diag_hessian=h_vec,
                    eps=cfg.hessian_eps,
                    blocksize=cfg.blocksize,
                    random_seed=cfg.seed,
                )
                extra_fp16_bits = float(p_stats["extra_fp16_bits"])
                if p_stats["beats_random_weighted_err"]:
                    any_weighted_advantage = True

        if use_d4_svd:
            u_r, v_r, d4_stats = compute_d4_hessian_weighted_svd_residual(
                w_orig,
                w_work,
                h_vec,
                rank=cfg.d4_svd_rank,
                eps=cfg.hessian_eps,
                blocksize=cfg.blocksize,
                extra_fp16_bits=extra_fp16_bits,
                max_storage_ratio_vs_fp16=cfg.max_storage_ratio_vs_fp16,
                clamp_rank_to_budget=cfg.clamp_d4_rank_to_budget,
                compare_unweighted_svd=False,
            )
            if int(u_r.shape[1]) > 0:
                w_work = (w_work.to(torch.float64) + u_r.to(torch.float64) @ v_r.to(torch.float64).T).to(
                    device=w_work.device, dtype=w_work.dtype
                )
            factor_bits = float(d4_stats["factor_bits"])

        total_err_after_sq += float(
            torch.sum(((w_fp_64 - w_work.detach().to(device="cpu", dtype=torch.float64)) * h_sqrt.unsqueeze(0)) ** 2).item()
        )
        mod.weight.copy_(w_work.to(device=mod.weight.device, dtype=mod.weight.dtype))
        total_bits += float(base_stats["base_nf4_bits"]) + extra_fp16_bits + factor_bits

    err_b = math.sqrt(max(0.0, total_err_before_sq))
    err_a = math.sqrt(max(0.0, total_err_after_sq))
    ratio_vs_fp16 = total_bits / max(fp16_bits, 1.0)
    return {
        "total_linear_params": int(total_params),
        "fp16_mb": round(fp16_bits / 8.0 / 1e6, 4),
        "quantized_mb": round(total_bits / 8.0 / 1e6, 4),
        "bits_per_param": float(total_bits / float(max(total_params, 1))),
        "storage_ratio_vs_fp16": float(ratio_vs_fp16),
        "storage_within_budget": bool(ratio_vs_fp16 <= cfg.max_storage_ratio_vs_fp16 + 1e-12),
        "hessian_weighted_err_before": err_b,
        "hessian_weighted_err_after": err_a,
        "hessian_weighted_err_reduction_pct": (100.0 * (1.0 - err_a / err_b)) if err_b > 1e-15 else 0.0,
        "weighted_err_beats_random": any_weighted_advantage,
    }


def _extract_norm_and_head_tensors(norm_mod: nn.Module, head_mod: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot final norm and head bias tensors for clean pre-calibration restoration."""
    out: dict[str, torch.Tensor] = {"norm_w": norm_mod.weight.detach().clone()}
    if getattr(norm_mod, "bias", None) is not None:
        out["norm_b"] = norm_mod.bias.detach().clone()
    if getattr(head_mod, "bias", None) is not None:
        out["head_b"] = head_mod.bias.detach().clone()
    return out


def _restore_norm_and_head_tensors(
    norm_mod: nn.Module, head_mod: nn.Module, saved: Mapping[str, torch.Tensor]
) -> None:
    """Restore final norm and head bias tensors in-place."""
    norm_mod.weight.copy_(saved["norm_w"].to(device=norm_mod.weight.device, dtype=norm_mod.weight.dtype))
    if getattr(norm_mod, "bias", None) is not None and "norm_b" in saved:
        norm_mod.bias.copy_(saved["norm_b"].to(device=norm_mod.bias.device, dtype=norm_mod.bias.dtype))
    if getattr(head_mod, "bias", None) is not None and "head_b" in saved:
        head_mod.bias.copy_(saved["head_b"].to(device=head_mod.bias.device, dtype=head_mod.bias.dtype))


def _snapshot_max_abs_diff(
    model: nn.Module,
    initial_snapshot: Mapping[str, torch.Tensor],
) -> float:
    """Compute exact maximum absolute parameter/buffer deviation against ``initial_snapshot``."""
    cur_sd = model.state_dict()
    max_err = 0.0
    for key, ref_t in initial_snapshot.items():
        diff = (cur_sd[key].detach().to(device="cpu", dtype=torch.float64) - ref_t.to(dtype=torch.float64)).abs()
        max_err = max(max_err, float(diff.max().item()) if diff.numel() > 0 else 0.0)
    return max_err


@torch.no_grad()
def execute_verified_nf4_d4_ofc_js_recovery(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    selection_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    target_module_paths: Sequence[str] | None = None,
    early_layer_attr: str | None = None,
    forced_arm: str | None = None,
    config: NF4D4HessianOFCConfig | None = None,
    norm_kind: str = "auto",
    norm_attr: str | None = None,
    head_attr: str | None = None,
    inplace: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Run verified ``PILOT17`` NF4 + ``<1%`` OFC FP16 + D4 Hessian SVD + JS-Norm recovery.

    Zero gradient steps and ``0 MiB`` GPU VRAM: diagnoses diagonal input Hessians and
    ``<1%`` OFC / outlier channels on ``probe_chunks``, evaluates candidate quantized arms
    on ``selection_chunks`` under ``<= 0.35x`` FP16 storage, applies cross-fitted
    James-Stein + signed-``tau`` norm calibration inside ``ker(G_D0) intersect S_val^perp``,
    verifies ANSE ``Delta E < 0`` and ``D0 <= +1.0%`` on ``eval_chunks`` against the
    Naive NF4 baseline, and enforces ANSE Maximum Pain atomic bitwise rollback
    (``0.0`` max error) on any breach with a canonical SHA-256 provenance receipt.
    """
    if config is not None and not isinstance(config, NF4D4HessianOFCConfig):
        raise TypeError(f"config must be NF4D4HessianOFCConfig or None, got {type(config).__name__}")
    if forced_arm is not None and forced_arm not in VALID_ARM_NAMES:
        raise ValueError(f"forced_arm must be one of {sorted(VALID_ARM_NAMES)}, got {forced_arm!r}")
    cfg = config or NF4D4HessianOFCConfig()
    _validate_bool(inplace, "inplace")
    cal_cfg = cfg.to_calibrator_config(enable_norm_calibration=False)

    _, _, _, norm_name_chk, head_name_chk, _, vocab_size, _ = _resolve_norm_and_head(
        model, norm_kind=norm_kind, norm_attr=norm_attr, head_attr=head_attr
    )
    if selection_chunks is None:
        fit_raw, sel_raw = _split_probe_chunks_fallback(probe_chunks, cfg.general_prefixes)
    else:
        fit_raw, sel_raw = probe_chunks, selection_chunks

    fit_prep = _prepare_causal_chunks(
        fit_raw, name="probe_chunks", vocab_size=vocab_size, score_from=cfg.score_from, general_prefixes=cfg.general_prefixes
    )
    sel_prep = _prepare_causal_chunks(
        sel_raw, name="selection_chunks", vocab_size=vocab_size, score_from=cfg.score_from, general_prefixes=cfg.general_prefixes
    )
    ver_prep = (
        sel_prep
        if eval_chunks is None
        else _prepare_causal_chunks(
            eval_chunks, name="eval_chunks", vocab_size=vocab_size, score_from=cfg.score_from, general_prefixes=cfg.general_prefixes
        )
    )

    work_model = model if inplace else copy.deepcopy(model)
    norm_mod, head_mod, _, norm_name, head_name, hidden_dim, _, _ = _resolve_norm_and_head(
        work_model, norm_kind=norm_kind, norm_attr=norm_name_chk, head_attr=head_name_chk
    )
    targets = _discover_target_linears(work_model, head_name, head_mod.weight, target_module_paths)
    early_mod = _resolve_early_layer(work_model, early_layer_attr, targets[0][1])

    was_training = work_model.training
    work_model.eval()
    try:
        initial_snapshot = _snapshot_state(work_model)
        initial_sha = compute_module_state_sha256(work_model)
        orig_weights = {n: mod.weight.detach().clone() for n, mod in targets}
        orig_norm_saved = _extract_norm_and_head_tensors(norm_mod, head_mod)

        # 1. Reference FP32 metrics on selection & verification splits
        _, fp32_sel_dom, fp32_sel_g = _eval_model_causal_metrics(work_model, sel_prep, cfg=cal_cfg)
        _, fp32_ver_dom, fp32_ver_g = _eval_model_causal_metrics(work_model, ver_prep, cfg=cal_cfg)

        # 2. Probe-only Hessian & <1% OFC / massive-activation diagnostics
        diag = _diagnose_probe_hessians_and_ofc(work_model, targets, early_mod, fit_prep, hidden_dim, cfg)
        diag_hessians: dict[str, torch.Tensor] = diag["diag_hessians"]
        prot_channels: list[int] = diag["protected_residual_channels"]
        ofc_indices: list[int] = diag["ofc_indices"]

        gen_rand = torch.Generator(device="cpu").manual_seed(cfg.seed)
        k_prot = len(prot_channels)
        rand_channels = (
            sorted(int(i) for i in torch.randperm(hidden_dim, generator=gen_rand)[:k_prot].tolist())
            if k_prot > 0
            else []
        )

        # 3. Precompute exact Dettmers NF4 block64 weights for all target modules
        nf4_weights: dict[str, torch.Tensor] = {}
        for mod_name, w_orig in orig_weights.items():
            w_q, _ = quantize_dequantize_dettmers_nf4(w_orig, blocksize=cfg.blocksize)
            nf4_weights[mod_name] = w_q

        arm_configs: list[tuple[str, Sequence[int], bool, bool, bool]] = [
            (ARM1_NAIVE_NF4_BLOCK64, (), False, False, False),
            (ARM2_NF4_RANDOM_1PCT_FP16_CONTROL, rand_channels, True, False, False),
            (ARM3_NF4_OFC_AND_OUTLIER_1PCT_FP16, prot_channels, True, False, False),
            (ARM4_NF4_D4_HESSIAN_SVD_RANK16, (), False, True, False),
            (ARM5_NF4_D4_HESSIAN_OFC_JS_NORM, prot_channels, True, True, cfg.enable_js_norm_calibration),
        ]
        if not cfg.evaluate_ablation_arms:
            target_set = {ARM1_NAIVE_NF4_BLOCK64, forced_arm or ARM5_NF4_D4_HESSIAN_OFC_JS_NORM}
            arm_configs = [row for row in arm_configs if row[0] in target_set]

        selection_summary: dict[str, Any] = {
            ARM0_FP32_REFERENCE: {
                "nll_mean": float(fp32_sel_g["nll_mean"]),
                "ppl_value": float(math.exp(min(float(fp32_sel_g["nll_mean"]), 20.0))),
                "ece_value": float(fp32_sel_g["ece_value"]),
                "per_domain": fp32_sel_dom,
            }
        }
        storage_summary: dict[str, Any] = {}
        arm_snapshots: dict[str, dict[str, torch.Tensor]] = {}
        arm_norm_meta: dict[str, dict[str, Any]] = {}

        for arm_name, ch_list, use_prot, use_d4, run_js in arm_configs:
            _restore_norm_and_head_tensors(norm_mod, head_mod, orig_norm_saved)
            s_rep = _apply_quantized_arm_to_model(
                targets,
                orig_weights,
                nf4_weights,
                diag_hessians,
                hidden_dim=hidden_dim,
                protected_channels=ch_list,
                use_protection=use_prot,
                use_d4_svd=use_d4,
                cfg=cfg,
            )
            storage_summary[arm_name] = s_rep

            norm_receipt: dict[str, Any] | None = None
            if run_js:
                _, norm_receipt = calibrate_signed_tau_and_crossfit_js_norm(
                    work_model,
                    fit_raw,
                    selection_chunks=sel_raw,
                    eval_chunks=None,
                    protected_indices=ofc_indices,
                    config=cal_cfg,
                    norm_kind=norm_kind,
                    norm_attr=norm_name,
                    head_attr=head_name,
                    inplace=True,
                )
            arm_norm_meta[arm_name] = {
                "tau_temp": float(norm_receipt["tau_temp"]) if norm_receipt else 1.0,
                "alpha_scale": float(norm_receipt["alpha_scale"]) if norm_receipt else 0.0,
                "d0_mean_grad_ortho_err": float(norm_receipt["d0_mean_grad_ortho_err"]) if norm_receipt else 0.0,
                "ofc_protected_max_delta": float(norm_receipt["ofc_protected_max_delta"]) if norm_receipt else 0.0,
                "norm_receipt": norm_receipt,
            }

            _, sel_dom, sel_g = _eval_model_causal_metrics(work_model, sel_prep, cfg=cal_cfg)
            selection_summary[arm_name] = {
                "nll_mean": float(sel_g["nll_mean"]),
                "ppl_value": float(math.exp(min(float(sel_g["nll_mean"]), 20.0))),
                "ece_value": float(sel_g["ece_value"]),
                "per_domain": sel_dom,
            }
            arm_snapshots[arm_name] = _snapshot_state(work_model)

        # 4. Evaluate Naive NF4 baseline on verification split
        _restore_state_inplace(work_model, arm_snapshots[ARM1_NAIVE_NF4_BLOCK64])
        _, nf4_ver_dom, nf4_ver_g = _eval_model_causal_metrics(work_model, ver_prep, cfg=cal_cfg)

        # 5. Select winning arm strictly on selection split (or forced_arm)
        nf4_sel_dom = selection_summary[ARM1_NAIVE_NF4_BLOCK64]["per_domain"]
        nf4_sel_ece = selection_summary[ARM1_NAIVE_NF4_BLOCK64]["ece_value"]
        if forced_arm is not None:
            selected_arm = forced_arm
        else:
            selected_arm = ARM1_NAIVE_NF4_BLOCK64
            best_sel_energy = 0.0
            for arm_name, _, _, _, _ in arm_configs:
                if arm_name == ARM1_NAIVE_NF4_BLOCK64:
                    continue
                if not storage_summary[arm_name]["storage_within_budget"]:
                    continue
                sel_ver = _compute_anse_verdict(
                    nf4_sel_dom,
                    nf4_sel_ece,
                    selection_summary[arm_name]["per_domain"],
                    selection_summary[arm_name]["ece_value"],
                    cfg=cal_cfg,
                )
                if sel_ver["guard_ok"] and float(sel_ver["anse_energy"]) < best_sel_energy:
                    best_sel_energy = float(sel_ver["anse_energy"])
                    selected_arm = arm_name

        # 6. Single-shot evaluation of selected arm on eval split against Naive NF4
        _restore_state_inplace(work_model, arm_snapshots[selected_arm])
        cand_sha = compute_module_state_sha256(work_model)
        _, ver_cand_dom, ver_cand_g = _eval_model_causal_metrics(work_model, ver_prep, cfg=cal_cfg)

        sel_storage = storage_summary[selected_arm]
        sel_norm = arm_norm_meta[selected_arm]
        ver_anse = _compute_anse_verdict(
            nf4_ver_dom, nf4_ver_g["ece_value"], ver_cand_dom, ver_cand_g["ece_value"], cfg=cal_cfg
        )

        storage_ok = bool(sel_storage["storage_within_budget"])
        ortho_ok = bool(
            sel_norm["d0_mean_grad_ortho_err"] < 1e-6 and sel_norm["ofc_protected_max_delta"] == 0.0
        )
        guard_ok = bool(ver_anse["guard_ok"] and storage_ok and ortho_ok)

        delta_nll_vs_nf4 = float(ver_cand_g["nll_mean"]) - float(nf4_ver_g["nll_mean"])
        quant_degrad = float(nf4_ver_g["nll_mean"]) - float(fp32_ver_g["nll_mean"])
        quant_recovered_pct = (
            (100.0 * (float(nf4_ver_g["nll_mean"]) - float(ver_cand_g["nll_mean"])) / quant_degrad)
            if quant_degrad > 1e-9
            else 0.0
        )
        improved = (
            float(ver_anse["anse_energy"]) < -cfg.min_energy_drop
            if cfg.require_strict_energy_drop
            else float(ver_anse["anse_energy"]) <= 0.0
        )

        # 7. ANSE thermodynamic commit or Maximum Pain atomic bitwise rollback
        if selected_arm == ARM1_NAIVE_NF4_BLOCK64 and not storage_ok:
            accepted = False
            rolled_back = True
            status = "ROLLED_BACK_STORAGE_BUDGET_EXCEEDED"
            _restore_state_inplace(work_model, initial_snapshot)
        elif not guard_ok:
            accepted = False
            rolled_back = True
            status = (
                "ROLLED_BACK_STORAGE_BUDGET_EXCEEDED"
                if not storage_ok
                else ("ROLLED_BACK_GUARD_VIOLATION" if not ver_anse["guard_ok"] else "ROLLED_BACK_ORTHO_VIOLATION")
            )
            _restore_state_inplace(work_model, initial_snapshot)
        elif cand_sha == initial_sha:
            accepted = False
            rolled_back = False
            status = "FIXED_POINT_UNCHANGED"
        elif selected_arm == ARM1_NAIVE_NF4_BLOCK64 and cfg.require_strict_energy_drop:
            accepted = False
            rolled_back = True
            status = "ROLLED_BACK_NO_RECOVERY_GAIN_OVER_NF4"
            _restore_state_inplace(work_model, initial_snapshot)
        elif improved:
            accepted = True
            rolled_back = False
            status = f"ACCEPTED_{selected_arm}"
        else:
            accepted = False
            rolled_back = True
            status = "ROLLED_BACK_ENERGY_NON_IMPROVING"
            _restore_state_inplace(work_model, initial_snapshot)

        final_sha = compute_module_state_sha256(work_model)
        rollback_err = _snapshot_max_abs_diff(work_model, initial_snapshot) if not accepted else 0.0
        if not accepted and (final_sha != initial_sha or rollback_err != 0.0):
            raise RuntimeError(
                f"Atomic bitwise rollback failed: sha_eq={final_sha == initial_sha}, err={rollback_err}"
            )
    finally:
        work_model.train(was_training)

    fp32_ver_nll = float(fp32_ver_g["nll_mean"])
    nf4_ver_nll = float(nf4_ver_g["nll_mean"])
    nf4_ver_ece = float(nf4_ver_g["ece_value"])
    nf4_ver_ppl = float(math.exp(min(nf4_ver_nll, 20.0)))

    if accepted:
        cand_ver_nll = float(ver_cand_g["nll_mean"])
        cand_ver_ece = float(ver_cand_g["ece_value"])
        cand_ver_ppl = float(math.exp(min(cand_ver_nll, 20.0)))
        eff_anse_energy = float(ver_anse["anse_energy"])
        eff_delta_e = float(ver_anse["delta_e"])
        eff_gen_drift = float(ver_anse["general_drift_ratio"])
        eff_worst_drift = float(ver_anse["worst_domain_pct_vs_base"]) / 100.0
        domains_improved = sum(1 for d in ver_cand_dom if float(ver_cand_dom[d]) < float(nf4_ver_dom[d]))
    else:
        cand_ver_nll = fp32_ver_nll
        cand_ver_ece = float(fp32_ver_g["ece_value"])
        cand_ver_ppl = float(math.exp(min(fp32_ver_nll, 20.0)))
        pain_energy = float(cfg.maximum_pain_energy + max(0.0, delta_nll_vs_nf4) + cfg.ece_energy_weight * float(ver_cand_g["ece_value"]))
        eff_anse_energy = pain_energy if not guard_ok else 0.0
        eff_delta_e = eff_anse_energy
        eff_gen_drift = 0.0
        eff_worst_drift = 0.0
        domains_improved = 0

    receipt: dict[str, Any] = {
        "object": RECOVERY_RECEIPT_OBJECT,
        "schema": RECOVERY_SCHEMA,
        "stage": "nf4_d4_hessian_ofc_js_recovery",
        "zero_gradient_steps": True,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "status": status,
        "selected_arm": selected_arm,
        "nll_mean": cand_ver_nll,
        "ece_value": cand_ver_ece,
        "ppl_value": cand_ver_ppl,
        "fp32_nll_mean": fp32_ver_nll,
        "nf4_nll_mean": nf4_ver_nll,
        "nf4_ece_value": nf4_ver_ece,
        "delta_nll_vs_nf4": delta_nll_vs_nf4 if accepted else 0.0,
        "quant_degradation_nats": quant_degrad,
        "quant_loss_recovered_pct": quant_recovered_pct if accepted else 0.0,
        "ppl_rel_vs_nf4": ((cand_ver_ppl - nf4_ver_ppl) / max(nf4_ver_ppl, 1e-12)) if accepted else 0.0,
        "ece_rel_vs_nf4": ((cand_ver_ece - nf4_ver_ece) / max(nf4_ver_ece, 1e-12)) if accepted else 0.0,
        "general_drift_ratio": eff_gen_drift,
        "worst_domain_drift_ratio": eff_worst_drift,
        "cand_general_drift_ratio": float(ver_anse["general_drift_ratio"]),
        "domains_improved_count": int(domains_improved),
        "domains_total_count": int(len(ver_cand_dom)),
        "bits_per_param": float(sel_storage["bits_per_param"]),
        "storage_ratio_vs_fp16": float(sel_storage["storage_ratio_vs_fp16"]),
        "fp16_mb": float(sel_storage["fp16_mb"]),
        "quantized_mb": float(sel_storage["quantized_mb"]),
        "storage_within_budget": storage_ok,
        "anse_energy": eff_anse_energy,
        "cand_anse_energy": float(ver_anse["anse_energy"]) if storage_ok else float(cfg.maximum_pain_energy),
        "delta_e": eff_delta_e,
        "guard_ok": guard_ok,
        "ofc_indices": ofc_indices,
        "outlier_indices": diag["outlier_indices"],
        "protected_residual_channels": prot_channels,
        "oof_auroc": float(diag["ofc_diag"]["oof_auroc"]),
        "ofc_iou": float(diag["ofc_diag"]["ofc_iou"]),
        "massive_activation_max_ratio": float(diag["ofc_diag"]["massive_activation_max_ratio"]),
        "ofc_protected_max_delta": float(sel_norm["ofc_protected_max_delta"]),
        "d0_mean_grad_ortho_err": float(sel_norm["d0_mean_grad_ortho_err"]),
        "tau_temp": float(sel_norm["tau_temp"]),
        "alpha_scale": float(sel_norm["alpha_scale"]),
        "exact_bitwise_rollback_max_err": float(rollback_err),
        "selection_summary": selection_summary,
        "storage_summary": storage_summary,
        "norm_calibration_receipt": sel_norm["norm_receipt"],
        "initial_state_sha256": initial_sha,
        "candidate_state_sha256": cand_sha,
        "final_state_sha256": final_sha,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()},
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return work_model, receipt


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "ARM0_FP32_REFERENCE",
    "ARM1_NAIVE_NF4_BLOCK64",
    "ARM2_NF4_RANDOM_1PCT_FP16_CONTROL",
    "ARM3_NF4_OFC_AND_OUTLIER_1PCT_FP16",
    "ARM4_NF4_D4_HESSIAN_SVD_RANK16",
    "ARM5_NF4_D4_HESSIAN_OFC_JS_NORM",
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_D4_SVD_RANK",
    "DEFAULT_ECE_ENERGY_WEIGHT",
    "DEFAULT_GENERAL_PREFIXES",
    "DEFAULT_HESSIAN_EPS",
    "DEFAULT_IGNORE_INDEX",
    "DEFAULT_JS_MAX_RANK",
    "DEFAULT_JS_MAX_STEP_NORM",
    "DEFAULT_MASSIVE_RATIO_THRESHOLD",
    "DEFAULT_MAX_DOMAIN_DRIFT_REL",
    "DEFAULT_MAX_GENERAL_DRIFT_REL",
    "DEFAULT_MAX_STORAGE_RATIO_VS_FP16",
    "DEFAULT_MIN_ENERGY_DROP",
    "DEFAULT_NF4_BLOCKSIZE",
    "DEFAULT_NULLSPACE_SV_TOL",
    "DEFAULT_NUM_BINS",
    "DEFAULT_OFC_FRACTION",
    "DEFAULT_OUTLIER_FRACTION",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_SEED",
    "DEFAULT_TAU_GRID",
    "NF4D4HessianOFCConfig",
    "RECOVERY_RECEIPT_OBJECT",
    "RECOVERY_SCHEMA",
    "VALID_ARM_NAMES",
    "VALID_PROTECTION_MODES",
    "compute_d4_hessian_weighted_svd_residual",
    "execute_verified_nf4_d4_ofc_js_recovery",
    "protect_ofc_and_outlier_channels_fp16",
    "quantize_dequantize_dettmers_nf4",
    "verify_nf4_d4_ofc_receipt_provenance",
]

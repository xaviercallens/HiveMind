# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Dynamic Alveole Mitosis Corruptor Recycler & Prose-Hessian Compensated 4-Bit NF4 Edge Exporter.

Implements **Gen-3 Direction D4** (Zenodo ``10.5281/zenodo.24542441`` Section 6 and Lean 4
theorem ``ANSE.ReineCortexDirections.thm_rcd_d4_alveole_mitosis_and_compensated_quant``)
for decentralized Sparse MoE models on CPU (``0 MiB`` GPU VRAM, ``0`` gradient steps):

1. **Causal-Knockout Corruptor Diagnosis** (:func:`diagnose_alveole_corruptors_and_anchors`):
   Runs deterministic Leave-One-Expert-Out (LOEO) knockouts across MoE slots to separate
   harmful Ghost-Sink corruptors (where zeroing the expert lowers NLL or prose drift)
   from high-value Anchor Alveoles and identifies highest-deficit specialist domains.
2. **Dynamic Alveole Mitosis inside ``P_0``** (:func:`recycle_corruptor_alveoles_via_mitosis`):
   Clones top Anchor Alveoles into corruptor slots, applies ``P_0`` null-space projection
   (annihilating prose interference ``||delta_mitosis @ V_0||_F == 0``), splits specialist
   subspaces when probe activations are available, steers recycled router rows toward
   high-deficit domain prototypes, and monitors BOS attention-sink stability.
3. **Diagonal Prose-Hessian Compensated 4-Bit NF4 Edge Export**
   (:func:`compute_diagonal_prose_hessian`, :func:`quantize_dequantize_nf4_blockwise`,
   :func:`compensate_nf4_with_prose_hessian`, :func:`export_model_to_compensated_nf4_edge`):
   Packs weights into canonical 16-level 4-bit NormalFloat (``NF4``) nibbles with
   Hessian-weighted block scale search + FP8 double quantization and alternating LoftQ
   diagonal prose-Hessian weighted low-rank residuals constrained inside ``P_0``,
   reducing quantization error under the ``6,656 MiB`` 8 GB consumer GPU WDDM envelope.
4. **ANSE Thermodynamic Gate & Atomic Bitwise Rollback**
   (:func:`execute_verified_alveole_mitosis_and_nf4_export`,
   :func:`verify_alveole_mitosis_nf4_receipt_provenance`):
   Commits iff ``ANSE Delta E < 0``, ``D0 / worst_domain <= +1.0%``, BOS sink drift
   within ceiling, and WDDM VRAM footprint ``<= 6,656 MiB``; otherwise restores parent
   weights bit-for-bit under ``ANSE_MAXIMUM_PAIN`` (``1e6``) with SHA-256 receipts.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.swarm.domain_matched_moe_arm_composer import fast_affine_layernorm_temperature_grid
from hivemind.swarm.logit_calibration_temperature_surgeon import fold_temperature_into_norm_params
from hivemind.swarm.natural_gradient_trust_wsp_surgeon import (
    ANSE_MAXIMUM_PAIN,
    _eval_crowd_model_domain_nlls,
    _parse_eval_domain_nll,
    _validate_bool,
    _validate_finite_2d,
    _validate_nonneg_finite,
    _validate_nonneg_int,
    _validate_positive_finite,
    _validate_positive_int,
    compute_payload_sha256,
    compute_state_sha256,
)
from hivemind.swarm.optimal_transport_moe_arm_fuser import (
    _capture_mlp_inputs_and_pre_ln,
    _discover_moe_mlp_layers,
    _resolve_expert_sequence,
    _resolve_layer_p0,
    _resolve_router_linear,
    _score_domain_nll_vs_ref,
    _validate_float_tuple,
)
from hivemind.swarm.procrustes_wsp_skill_transfer import (
    _validate_activation_matrix,
    _validate_square_matrix,
    _validate_unit_interval,
)
from hivemind.swarm.tangent_task_vector_surgeon import compute_prose_nullspace_projector_scaled
from hivemind.swarm.volunteer_wsp_workunit_exporter import (
    CANONICAL_NF4_LEVELS,
    STD_8GB_WDDM_CEILING_MIB,
    VRAMCeilingExceededError,
    _pack_nf4_nibbles,
    dequantize_nf4_compensated_edge_blocks,
)
from hivemind.swarm.wsp_alphaedit_control import forward_model_pre_ln

ALVEOLE_MITOSIS_EXPORTER_SCHEMA: str = "hivemind.alveole_mitosis_hessian_nf4_exporter/1"
ALVEOLE_MITOSIS_RECEIPT_OBJECT: str = "hivemind.alveole_mitosis_nf4_export_receipt"

DEFAULT_GENERAL_PREFIXES: tuple[str, ...] = ("E0_", "D0_", "general")
DEFAULT_NF4_BLOCK_SIZE: int = 64
DEFAULT_DOUBLE_QUANT_SUPERBLOCK: int = 256
DEFAULT_COMPENSATION_RANK: int = 4
DEFAULT_ALTERNATING_STEPS: int = 3
DEFAULT_HESSIAN_DAMPING: float = 1e-4
DEFAULT_PROSE_HESSIAN_WEIGHT: float = 4.0
DEFAULT_CORRUPTOR_KNOCKOUT_THRESHOLD: float = 1e-4
DEFAULT_PROSE_HARM_THRESHOLD: float = 1e-4
DEFAULT_ROUTER_PROTO_BLEND: float = 0.35
DEFAULT_SUBSPACE_SPLIT_BETA: float = 0.0
DEFAULT_MAX_BOS_SINK_DRIFT: float = 0.05
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_KV_AND_RUNTIME_OVERHEAD_MIB: float = 1150.0
DEFAULT_NULLSPACE_SV_REL_TOL: float = 1e-3
DEFAULT_NULLSPACE_VARIANCE_THRESHOLD: float | None = 0.999
DEFAULT_NULLSPACE_LEAK_TOL: float = 1e-6
DEFAULT_NULLSPACE_EIG_TOL: float = 0.5
DEFAULT_LAMBDA_GRID: tuple[float, ...] = tuple(round(0.05 * idx, 2) for idx in range(11, 21))
DEFAULT_TAU_GRID: tuple[float, ...] = (0.98, 1.00, 1.02, 1.04)
DEFAULT_SCORE_FROM: int = 8
DEFAULT_BATCH_SIZE: int = 16

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPERT_PATH_RE = re.compile(r"(?:^|\.)(?:experts|experts_w1|experts_w2|experts_w3)\.(\d+)(?:\.|$)")
_LAYER_PATH_RE = re.compile(r"(?:^|\.)(?:blocks|layers)\.(\d+)(?:\.|$)")


# ---------------------------------------------------------------------------
# 1. Configuration, Validation & Provenance Verification
# ---------------------------------------------------------------------------


def verify_alveole_mitosis_nf4_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a Mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if (
        receipt.get("object") != ALVEOLE_MITOSIS_RECEIPT_OBJECT
        or receipt.get("schema") != ALVEOLE_MITOSIS_EXPORTER_SCHEMA
    ):
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


def _is_general_domain(domain_id: str, general_prefixes: Sequence[str]) -> bool:
    """Return True if ``domain_id`` matches any configured general-prose prefix."""
    low = domain_id.strip().lower()
    return any(low.startswith(p.lower()) for p in general_prefixes)


def _normalize_chunks(
    chunks: Sequence[Mapping[str, Any]],
    name: str,
    *,
    general_prefixes: Sequence[str],
    score_from: int,
    require_both_groups: bool,
) -> tuple[torch.Tensor, list[str]]:
    """Validate token chunks accepting either ``expert_id`` or ``domain`` labels."""
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence) or not chunks:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")
    seq_lens: set[int] = set()
    rows: list[list[int]] = []
    domain_ids: list[str] = []
    for idx, ch in enumerate(chunks):
        if not isinstance(ch, Mapping) or "ids" not in ch:
            raise ValueError(f"{name}[{idx}] must be a mapping containing 'ids'")
        raw_dom = ch.get("expert_id", ch.get("domain"))
        if raw_dom is None or not str(raw_dom).strip():
            raise ValueError(f"{name}[{idx}] must contain a non-empty 'expert_id' or 'domain'")
        dom = str(raw_dom).strip()
        raw_ids = ch["ids"]
        if isinstance(raw_ids, torch.Tensor):
            row = [int(t) for t in raw_ids.detach().reshape(-1).tolist()]
        elif isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes)):
            row = [int(t) for t in raw_ids]
        else:
            raise TypeError(f"{name}[{idx}]['ids'] must be a sequence or tensor of token IDs")
        if len(row) < 2 or any(t < 0 for t in row):
            raise ValueError(f"{name}[{idx}]['ids'] must contain >= 2 non-negative token IDs")
        seq_lens.add(len(row))
        rows.append(row)
        domain_ids.append(dom)

    if len(seq_lens) != 1:
        raise ValueError(f"All {name} entries must share identical sequence lengths")
    seq_len = next(iter(seq_lens))
    start_pos = max(0, score_from - 1)
    if start_pos >= seq_len - 1:
        raise ValueError(f"Sequence length {seq_len} is too short for score_from={score_from}")

    has_gen = any(_is_general_domain(d, general_prefixes) for d in domain_ids)
    has_spec = any(not _is_general_domain(d, general_prefixes) for d in domain_ids)
    if require_both_groups and (not has_gen or not has_spec):
        raise ValueError(f"{name} must include both general and specialist domain chunks")
    return torch.tensor(rows, dtype=torch.long), domain_ids


@dataclass(frozen=True)
class AlveoleMitosisNF4Config:
    """Validated configuration for Dynamic Alveole Mitosis & Prose-Hessian NF4 export."""

    corruptor_knockout_threshold: float = DEFAULT_CORRUPTOR_KNOCKOUT_THRESHOLD
    prose_harm_threshold: float = DEFAULT_PROSE_HARM_THRESHOLD
    max_recycled_experts: int | None = None
    protect_general_expert: bool = True
    general_expert_index: int = 0
    project_mitosis_into_p0: bool = True
    project_router_into_p0: bool = False
    router_proto_blend: float = DEFAULT_ROUTER_PROTO_BLEND
    subspace_split_beta: float = DEFAULT_SUBSPACE_SPLIT_BETA
    max_bos_sink_drift: float = DEFAULT_MAX_BOS_SINK_DRIFT
    quantize_model_to_nf4: bool = True
    nf4_block_size: int = DEFAULT_NF4_BLOCK_SIZE
    double_quant_scales: bool = True
    compensation_rank: int = DEFAULT_COMPENSATION_RANK
    num_alternating_steps: int = DEFAULT_ALTERNATING_STEPS
    hessian_damping: float = DEFAULT_HESSIAN_DAMPING
    prose_hessian_weight: float = DEFAULT_PROSE_HESSIAN_WEIGHT
    project_nf4_residual_in_p0: bool = True
    quantize_attention: bool = True
    quantize_base_mlp: bool = True
    quantize_dense_experts: bool = True
    quantize_lm_head: bool = True
    quantize_router: bool = False
    min_linear_numel_to_quantize: int = 128
    wddm_vram_ceiling_mib: float = float(STD_8GB_WDDM_CEILING_MIB)
    kv_and_runtime_overhead_mib: float = DEFAULT_KV_AND_RUNTIME_OVERHEAD_MIB
    projected_target_param_count: int | None = None
    enforce_wddm_budget: bool = True
    calibrate_layernorm: bool = True
    calibrator_max_general_drift_rel: float = 0.006
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID
    tau_grid: tuple[float, ...] = DEFAULT_TAU_GRID
    nullspace_sv_rel_tol: float = DEFAULT_NULLSPACE_SV_REL_TOL
    nullspace_variance_threshold: float | None = DEFAULT_NULLSPACE_VARIANCE_THRESHOLD
    nullspace_max_modes: int | None = None
    nullspace_leak_tol: float = DEFAULT_NULLSPACE_LEAK_TOL
    max_general_drift_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    max_worst_domain_drift_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    enforce_all_domain_guard: bool = True
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    require_energy_improvement: bool = True
    min_energy_improvement: float = 0.0
    general_prefixes: tuple[str, ...] = DEFAULT_GENERAL_PREFIXES
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    inplace: bool = False

    def __post_init__(self) -> None:
        _validate_nonneg_finite(self.corruptor_knockout_threshold, "corruptor_knockout_threshold")
        _validate_nonneg_finite(self.prose_harm_threshold, "prose_harm_threshold")
        if self.max_recycled_experts is not None:
            _validate_positive_int(self.max_recycled_experts, "max_recycled_experts")
        _validate_bool(self.protect_general_expert, "protect_general_expert")
        _validate_nonneg_int(self.general_expert_index, "general_expert_index")
        _validate_bool(self.project_mitosis_into_p0, "project_mitosis_into_p0")
        _validate_bool(self.project_router_into_p0, "project_router_into_p0")
        _validate_unit_interval(self.router_proto_blend, "router_proto_blend", allow_zero=True)
        _validate_unit_interval(self.subspace_split_beta, "subspace_split_beta", allow_zero=True)
        _validate_nonneg_finite(self.max_bos_sink_drift, "max_bos_sink_drift")
        _validate_bool(self.quantize_model_to_nf4, "quantize_model_to_nf4")
        _validate_positive_int(self.nf4_block_size, "nf4_block_size")
        _validate_bool(self.double_quant_scales, "double_quant_scales")
        _validate_positive_int(self.compensation_rank, "compensation_rank")
        _validate_positive_int(self.num_alternating_steps, "num_alternating_steps")
        _validate_positive_finite(self.hessian_damping, "hessian_damping")
        _validate_nonneg_finite(self.prose_hessian_weight, "prose_hessian_weight")
        _validate_bool(self.project_nf4_residual_in_p0, "project_nf4_residual_in_p0")
        for flag in (
            "quantize_attention",
            "quantize_base_mlp",
            "quantize_dense_experts",
            "quantize_lm_head",
            "quantize_router",
            "enforce_wddm_budget",
            "calibrate_layernorm",
            "enforce_all_domain_guard",
            "require_energy_improvement",
            "inplace",
        ):
            _validate_bool(getattr(self, flag), flag)
        _validate_positive_int(self.min_linear_numel_to_quantize, "min_linear_numel_to_quantize")
        _validate_positive_finite(self.wddm_vram_ceiling_mib, "wddm_vram_ceiling_mib")
        _validate_nonneg_finite(self.kv_and_runtime_overhead_mib, "kv_and_runtime_overhead_mib")
        if self.projected_target_param_count is not None:
            _validate_positive_int(self.projected_target_param_count, "projected_target_param_count")
        _validate_nonneg_finite(self.calibrator_max_general_drift_rel, "calibrator_max_general_drift_rel")
        object.__setattr__(self, "lambda_grid", _validate_float_tuple(self.lambda_grid, "lambda_grid", unit_interval=True))
        object.__setattr__(self, "tau_grid", tuple(sorted(_validate_float_tuple(self.tau_grid, "tau_grid"))))
        sv_tol = _validate_nonneg_finite(self.nullspace_sv_rel_tol, "nullspace_sv_rel_tol")
        if sv_tol >= 1.0:
            raise ValueError(f"nullspace_sv_rel_tol must be < 1.0, got {sv_tol!r}")
        if self.nullspace_variance_threshold is not None:
            _validate_unit_interval(self.nullspace_variance_threshold, "nullspace_variance_threshold", allow_zero=False)
        if self.nullspace_max_modes is not None:
            _validate_positive_int(self.nullspace_max_modes, "nullspace_max_modes")
        _validate_nonneg_finite(self.nullspace_leak_tol, "nullspace_leak_tol")
        max_gen = _validate_positive_finite(self.max_general_drift_ratio, "max_general_drift_ratio")
        max_worst = _validate_positive_finite(self.max_worst_domain_drift_ratio, "max_worst_domain_drift_ratio")
        if max_gen < 1.0 or max_worst < 1.0:
            raise ValueError("max_general_drift_ratio and max_worst_domain_drift_ratio must be >= 1.0")
        safe_r = _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        if safe_r < 1.0:
            raise ValueError(f"anse_safe_general_ratio must be >= 1.0, got {safe_r}")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.min_energy_improvement, "min_energy_improvement")
        if (
            isinstance(self.general_prefixes, (str, bytes))
            or not isinstance(self.general_prefixes, Sequence)
            or not self.general_prefixes
            or not all(isinstance(p, str) and p.strip() for p in self.general_prefixes)
        ):
            raise ValueError("general_prefixes must be a non-empty sequence of non-empty strings")
        object.__setattr__(self, "general_prefixes", tuple(p.strip() for p in self.general_prefixes))
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")


# ---------------------------------------------------------------------------
# 2. Diagonal Prose-Hessian & 4-Bit NF4 Low-Rank Compensation inside P_0
# ---------------------------------------------------------------------------


def _normalize_1d_hessian_diag(hessian_diag: Any, expected_dim: int, name: str = "hessian_diag") -> torch.Tensor:
    """Validate and normalize diagonal Hessian input onto CPU ``float64`` vector ``(expected_dim,)``."""
    if not isinstance(hessian_diag, torch.Tensor) or not hessian_diag.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    h64 = hessian_diag.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(h64).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    if h64.ndim == 1 and int(h64.shape[0]) == expected_dim:
        vec = h64
    elif h64.ndim == 2 and tuple(h64.shape) == (expected_dim, expected_dim):
        vec = torch.diagonal(h64)
    elif h64.ndim == 2 and int(h64.shape[1]) == expected_dim:
        vec = h64.mean(dim=0)
    else:
        raise ValueError(
            f"{name} shape {tuple(h64.shape)} must match ({expected_dim},), (*, {expected_dim}), or ({expected_dim}, {expected_dim})"
        )
    if bool((vec < 0.0).any().item()):
        raise ValueError(f"{name} must contain only non-negative values")
    return vec.contiguous()


@torch.no_grad()
def compute_diagonal_prose_hessian(
    x_prose: torch.Tensor,
    *,
    x_all: torch.Tensor | None = None,
    prose_weight: float = DEFAULT_PROSE_HESSIAN_WEIGHT,
    damping: float = DEFAULT_HESSIAN_DAMPING,
) -> torch.Tensor:
    """Compute diagonal empirical Gauss-Newton / prose Hessian ``diag(H_prose)`` in CPU ``float64``."""
    xp = _validate_activation_matrix(x_prose, "x_prose").to(torch.float64)
    pw = _validate_nonneg_finite(prose_weight, "prose_weight")
    damp = _validate_positive_finite(damping, "damping")
    h_prose = xp.pow(2).mean(dim=0)

    if x_all is not None:
        xa = _validate_activation_matrix(x_all, "x_all").to(torch.float64)
        if int(xa.shape[1]) != int(xp.shape[1]):
            raise ValueError(f"x_all width {xa.shape[1]} != x_prose width {xp.shape[1]}")
        h_raw = xa.pow(2).mean(dim=0) + pw * h_prose
    else:
        h_raw = h_prose

    mean_scale = max(float(h_raw.mean().item()), 1e-12)
    return (h_raw + max(damp * mean_scale, 1e-12)).contiguous()


def _quantize_nf4_block_scales(
    scales: torch.Tensor,
    *,
    double_quant_scales: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, int]:
    """Double-quantize NF4 block absmax scales to 8-bit unsigned integers per 256-block superblock."""
    rows, num_blocks = int(scales.shape[0]), int(scales.shape[1])
    if not double_quant_scales:
        s_f32 = scales.to(dtype=torch.float32)
        return s_f32.to(torch.float64), None, None, rows * num_blocks * 4

    flat_s = scales.reshape(-1)
    n_tot = int(flat_s.numel())
    sb = DEFAULT_DOUBLE_QUANT_SUPERBLOCK
    n_sb = (n_tot + sb - 1) // sb
    pad_sb = n_sb * sb - n_tot
    padded_s = F.pad(flat_s, (0, pad_sb), mode="constant", value=0.0) if pad_sb > 0 else flat_s
    grouped = padded_s.reshape(n_sb, sb)
    sb_max = grouped.amax(dim=-1, keepdim=True).clamp_min(1e-12).to(torch.float32)
    sb_max64 = sb_max.to(torch.float64)
    q_scales = torch.round((grouped / sb_max64) * 255.0).clamp(0.0, 255.0).to(torch.uint8)
    dq_scales = ((q_scales.to(torch.float64) / 255.0) * sb_max64).reshape(-1)[:n_tot].reshape(rows, num_blocks)
    scale_bytes = n_tot + n_sb * 4
    return dq_scales, q_scales.reshape(-1)[:n_tot].contiguous(), sb_max.reshape(-1).contiguous(), scale_bytes


@torch.no_grad()
def quantize_dequantize_nf4_blockwise(
    weight: torch.Tensor,
    *,
    block_size: int = DEFAULT_NF4_BLOCK_SIZE,
    double_quant_scales: bool = True,
    hessian_diag: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Quantize and reconstruct a 2D weight tensor using canonical 16-level 4-bit NF4 blocks."""
    w64 = _validate_finite_2d(weight, "weight")
    orig_dtype, orig_device = weight.dtype, weight.device
    blk = _validate_positive_int(block_size, "block_size")
    dq_flag = _validate_bool(double_quant_scales, "double_quant_scales")

    rows, cols = int(w64.shape[0]), int(w64.shape[1])
    h_vec = (
        _normalize_1d_hessian_diag(hessian_diag, cols, "hessian_diag").clamp_min(1e-12)
        if hessian_diag is not None
        else torch.ones(cols, dtype=torch.float64)
    )

    num_blocks = (cols + blk - 1) // blk
    padded_cols = num_blocks * blk
    pad_cols = padded_cols - cols
    w_pad = F.pad(w64, (0, pad_cols), mode="constant", value=0.0) if pad_cols > 0 else w64
    h_pad = F.pad(h_vec, (0, pad_cols), mode="constant", value=0.0) if pad_cols > 0 else h_vec

    w_blk = w_pad.reshape(rows, num_blocks, blk)
    h_blk = h_pad.reshape(1, num_blocks, blk)
    codebook = torch.tensor(CANONICAL_NF4_LEVELS, dtype=torch.float64)
    midpoints = (0.5 * (codebook[:-1] + codebook[1:])).contiguous()

    # 1. Naive RTN absmax scales (also quantized to storage precision)
    s_rtn_raw = w_blk.abs().amax(dim=-1)
    s_rtn_dq, q_s_rtn, sb_rtn, scale_bytes = _quantize_nf4_block_scales(
        s_rtn_raw, double_quant_scales=dq_flag
    )
    safe_s_rtn = torch.where(s_rtn_dq > 0.0, s_rtn_dq, torch.ones_like(s_rtn_dq)).unsqueeze(-1)
    idx_rtn = torch.bucketize((w_blk / safe_s_rtn).clamp(-1.0, 1.0).contiguous(), midpoints)
    idx_rtn = torch.where(s_rtn_dq.unsqueeze(-1) > 0.0, idx_rtn, torch.full_like(idx_rtn, 7))
    deq_rtn = codebook[idx_rtn] * s_rtn_dq.unsqueeze(-1)
    err_rtn_blk = (h_blk * (w_blk - deq_rtn).pow(2)).sum(dim=-1, keepdim=True)
    naive_h_err = float(math.sqrt(max(0.0, float(err_rtn_blk.sum().item()))))
    naive_fro = float(torch.linalg.matrix_norm(w64 - deq_rtn.reshape(rows, padded_cols)[:, :cols], ord="fro").item())

    best_scales_dq = s_rtn_dq
    best_q_scales = q_s_rtn
    best_sb_max = sb_rtn
    best_idx = idx_rtn

    # 2. Diagonal-Hessian weighted optimal block scale search + alternating WLS refinement
    if hessian_diag is not None:
        s_opt_raw = s_rtn_raw.unsqueeze(-1).clone()
        best_cont_err = err_rtn_blk.clone()
        alphas = torch.linspace(0.50, 1.05, 12, dtype=torch.float64)
        alpha_chunk = max(1, min(int(alphas.numel()), 262_144 // max(1, int(w_blk.numel()))))
        s0 = s_rtn_raw.unsqueeze(0).unsqueeze(-1)
        w_exp = w_blk.unsqueeze(0)
        h_exp = h_blk.unsqueeze(0)
        for a0 in range(0, int(alphas.numel()), alpha_chunk):
            a_sl = alphas[a0 : a0 + alpha_chunk].view(-1, 1, 1, 1)
            cand_s = s0 * a_sl
            for _ in range(3):
                safe_c = torch.where(cand_s > 0.0, cand_s, torch.ones_like(cand_s))
                cand_idx = torch.bucketize((w_exp / safe_c).clamp(-1.0, 1.0).contiguous(), midpoints)
                cand_idx = torch.where(cand_s > 0.0, cand_idx, torch.full_like(cand_idx, 7))
                codes = codebook[cand_idx]
                num = (h_exp * w_exp * codes).sum(dim=-1, keepdim=True)
                den = (h_exp * codes * codes).sum(dim=-1, keepdim=True)
                cand_s = torch.where(
                    (s0 > 0.0) & (den > 1e-12),
                    (num / den.clamp_min(1e-12)).clamp_min(1e-12),
                    cand_s,
                )
            cand_deq = codebook[cand_idx] * cand_s
            cand_err = (h_exp * (w_exp - cand_deq).pow(2)).sum(dim=-1)
            best_a = cand_err.argmin(dim=0, keepdim=True)
            best_local_err = torch.gather(cand_err, 0, best_a).squeeze(0).unsqueeze(-1)
            best_local_s = torch.gather(cand_s.squeeze(-1), 0, best_a).squeeze(0).unsqueeze(-1)
            better = best_local_err < best_cont_err
            best_cont_err = torch.where(better, best_local_err, best_cont_err)
            s_opt_raw = torch.where(better, best_local_s, s_opt_raw)

        s_opt_dq, q_s_opt, sb_opt, _ = _quantize_nf4_block_scales(
            s_opt_raw.squeeze(-1), double_quant_scales=dq_flag
        )
        safe_s_opt = torch.where(s_opt_dq > 0.0, s_opt_dq, torch.ones_like(s_opt_dq)).unsqueeze(-1)
        idx_opt = torch.bucketize((w_blk / safe_s_opt).clamp(-1.0, 1.0).contiguous(), midpoints)
        idx_opt = torch.where(s_opt_dq.unsqueeze(-1) > 0.0, idx_opt, torch.full_like(idx_opt, 7))
        deq_opt = codebook[idx_opt] * s_opt_dq.unsqueeze(-1)
        err_opt_blk = (h_blk * (w_blk - deq_opt).pow(2)).sum(dim=-1, keepdim=True)

        if float(err_opt_blk.sum().item()) <= float(err_rtn_blk.sum().item()):
            best_scales_dq = s_opt_dq
            best_q_scales = q_s_opt
            best_sb_max = sb_opt
            best_idx = idx_opt

    # 3. Bit-exact nibble packing & reconstruction
    packed_nibbles = _pack_nf4_nibbles(best_idx)
    scales_f32 = best_scales_dq.to(dtype=torch.float32).contiguous()
    w_deq_cpu = dequantize_nf4_compensated_edge_blocks(
        packed_nibbles,
        scales_f32,
        shape=(rows, cols),
        block_size=blk,
        dtype=orig_dtype,
    )
    w_deq_64 = w_deq_cpu.to(torch.float64)
    diff = w64 - w_deq_64
    comp_h_err = float(torch.linalg.matrix_norm(diff * h_vec.sqrt().unsqueeze(0), ord="fro").item())
    comp_fro = float(torch.linalg.matrix_norm(diff, ord="fro").item())
    nibble_bytes = int(packed_nibbles.numel())
    edge_bytes = nibble_bytes + scale_bytes

    return w_deq_cpu.to(device=orig_device), {
        "quant_format": "nf4",
        "block_size": blk,
        "double_quant_scales": dq_flag,
        "shape": [rows, cols],
        "padded_cols": padded_cols,
        "num_blocks_per_row": num_blocks,
        "total_blocks": rows * num_blocks,
        "packed_nibbles": packed_nibbles,
        "block_scales": scales_f32,
        "q_scales_uint8": best_q_scales,
        "superblock_max": best_sb_max,
        "nibble_bytes": nibble_bytes,
        "scale_bytes": scale_bytes,
        "edge_payload_bytes": edge_bytes,
        "bits_per_weight": float((8.0 * edge_bytes) / float(rows * cols)),
        "naive_hessian_error": naive_h_err,
        "hessian_error": comp_h_err,
        "naive_frobenius_error": naive_fro,
        "frobenius_error": comp_fro,
        "packed_nibbles_sha256": compute_payload_sha256(packed_nibbles.tolist()),
    }


def _extract_p0_orthonormal_basis(
    p_null: torch.Tensor,
    expected_dim: int,
    *,
    eig_tol: float = DEFAULT_NULLSPACE_EIG_TOL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate square symmetric projector ``P_0`` and extract orthonormal basis ``Q_null``."""
    p_sym = _validate_square_matrix(p_null, "p_null", expected_dim=expected_dim).to(torch.float64)
    evals, evecs = torch.linalg.eigh(p_sym)
    min_ev, max_ev = float(evals.min().item()), float(evals.max().item())
    if min_ev < -1e-3 or max_ev > 1.0 + 1e-3:
        raise ValueError(f"p_null eigenvalues [{min_ev:.3e}, {max_ev:.3e}] must lie in [0, 1]")
    keep = evals > eig_tol
    q_null = evecs[:, keep].contiguous()
    return p_sym, q_null


@torch.no_grad()
def compensate_nf4_with_prose_hessian(
    w_target: torch.Tensor,
    hessian_diag: torch.Tensor,
    *,
    rank: int = DEFAULT_COMPENSATION_RANK,
    block_size: int = DEFAULT_NF4_BLOCK_SIZE,
    num_alternating_steps: int = DEFAULT_ALTERNATING_STEPS,
    double_quant_scales: bool = True,
    p_null: torch.Tensor | None = None,
    nullspace_eig_tol: float = DEFAULT_NULLSPACE_EIG_TOL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Quantize ``w_target`` to 4-bit NF4 and compensate via diagonal prose-Hessian low-rank SVD in ``P_0``.

    Solves the alternating LoftQ problem minimizing ``||(W - Q - L_B L_A) diag(h)^{1/2}||_F^2``.
    When ``p_null`` is provided, ``L_A`` is derived in the orthonormal basis ``Q_null`` of
    ``P_0`` via metric factor ``B_h = Q_null^T diag(h) Q_null``, guaranteeing **exact**
    ``(L_B L_A)(I - P_0) == 0`` while strictly contracting diagonal prose-Hessian error.
    """
    w64 = _validate_finite_2d(w_target, "w_target")
    orig_dtype, orig_device = w_target.dtype, w_target.device
    d_out, d_in = int(w64.shape[0]), int(w64.shape[1])
    h_vec = _normalize_1d_hessian_diag(hessian_diag, d_in, "hessian_diag").clamp_min(1e-12)
    rank_i = _validate_positive_int(rank, "rank")
    blk_i = _validate_positive_int(block_size, "block_size")
    steps_i = _validate_positive_int(num_alternating_steps, "num_alternating_steps")
    dq_flag = _validate_bool(double_quant_scales, "double_quant_scales")

    d_h = h_vec.sqrt()
    p_sym: torch.Tensor | None = None
    q_null: torch.Tensor | None = None
    b_h_inv_half: torch.Tensor | None = None
    if p_null is not None:
        p_sym, q_null = _extract_p0_orthonormal_basis(p_null, d_in, eig_tol=nullspace_eig_tol)
        if q_null.shape[1] > 0:
            b_h = (q_null.T * h_vec.unsqueeze(0)) @ q_null
            b_h = 0.5 * (b_h + b_h.T)
            ev_bh, vec_bh = torch.linalg.eigh(b_h)
            ev_bh = ev_bh.clamp_min(1e-12)
            b_h_inv_half = (vec_bh * ev_bh.rsqrt().unsqueeze(0)) @ vec_bh.T

    # Baseline naive uncompensated NF4 RTN error (without Hessian scale opt or low-rank residual)
    q_naive, q_naive_diag = quantize_dequantize_nf4_blockwise(
        w64, block_size=blk_i, double_quant_scales=dq_flag, hessian_diag=None
    )
    naive_weighted_fro = float(torch.linalg.matrix_norm((w64 - q_naive) * d_h.unsqueeze(0), ord="fro").item())
    naive_fro = float(q_naive_diag["frobenius_error"])

    # Initial Hessian-scale-only NF4 baseline
    q_scale0, q_scale0_diag = quantize_dequantize_nf4_blockwise(
        w64, block_size=blk_i, double_quant_scales=dq_flag, hessian_diag=h_vec
    )
    scale0_weighted_fro = float(torch.linalg.matrix_norm((w64 - q_scale0) * d_h.unsqueeze(0), ord="fro").item())

    r_eff = min(rank_i, d_out, d_in)
    best_q = q_scale0 if scale0_weighted_fro <= naive_weighted_fro else q_naive
    best_q_diag = q_scale0_diag if scale0_weighted_fro <= naive_weighted_fro else q_naive_diag
    best_b = torch.zeros((d_out, rank_i), dtype=torch.float64)
    best_a = torch.zeros((rank_i, d_in), dtype=torch.float64)
    best_weighted_fro = min(naive_weighted_fro, scale0_weighted_fro)

    b_work = best_b.clone()
    a_work = best_a.clone()
    step_weighted_errors: list[float] = []

    for _ in range(steps_i):
        q_cand, q_cand_diag = quantize_dequantize_nf4_blockwise(
            w64 - (b_work @ a_work),
            block_size=blk_i,
            double_quant_scales=dq_flag,
            hessian_diag=h_vec,
        )
        err = w64 - q_cand
        b_cand = torch.zeros((d_out, rank_i), dtype=torch.float64)
        a_cand = torch.zeros((rank_i, d_in), dtype=torch.float64)

        if p_sym is None:
            u_m, s_v, vh_m = torch.linalg.svd(err * d_h.unsqueeze(0), full_matrices=False)
            k_act = min(r_eff, int(s_v.numel()))
            if k_act > 0:
                s_sq = s_v[:k_act].clamp_min(0.0).sqrt()
                b_cand[:, :k_act] = u_m[:, :k_act] * s_sq.unsqueeze(0)
                a_cand[:k_act, :] = (s_sq.unsqueeze(1) * vh_m[:k_act, :]) / d_h.unsqueeze(0)
        elif q_null is not None and b_h_inv_half is not None and q_null.shape[1] > 0:
            c_mat = (err * h_vec.unsqueeze(0)) @ q_null @ b_h_inv_half
            u_m, s_v, vh_m = torch.linalg.svd(c_mat, full_matrices=False)
            k_act = min(r_eff, int(s_v.numel()))
            if k_act > 0:
                s_sq = s_v[:k_act].clamp_min(0.0).sqrt()
                b_cand[:, :k_act] = u_m[:, :k_act] * s_sq.unsqueeze(0)
                a_cand[:k_act, :] = (s_sq.unsqueeze(1) * vh_m[:k_act, :]) @ b_h_inv_half @ q_null.T

        recon_err = w64 - (q_cand + (b_cand @ a_cand))
        cand_weighted_fro = float(torch.linalg.matrix_norm(recon_err * d_h.unsqueeze(0), ord="fro").item())
        if cand_weighted_fro <= best_weighted_fro + 1e-12:
            best_q = q_cand
            best_q_diag = q_cand_diag
            best_b = b_cand
            best_a = a_cand
            best_weighted_fro = cand_weighted_fro
            b_work = b_cand
            a_work = a_cand
        step_weighted_errors.append(best_weighted_fro)

    lora_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    w_quant_out = best_q.to(device=orig_device, dtype=orig_dtype).contiguous()
    lora_b_out = best_b.to(device=orig_device, dtype=lora_dtype).contiguous()
    lora_a_out = best_a.to(device=orig_device, dtype=lora_dtype).contiguous()

    recon_final = (
        w_quant_out.detach().to(device="cpu", dtype=torch.float64)
        + (lora_b_out.detach().to(device="cpu", dtype=torch.float64) @ lora_a_out.detach().to(device="cpu", dtype=torch.float64))
    )
    final_diff = w64 - recon_final
    final_fro = float(torch.linalg.matrix_norm(final_diff, ord="fro").item())
    final_weighted_fro = float(torch.linalg.matrix_norm(final_diff * d_h.unsqueeze(0), ord="fro").item())
    p0_leak = (
        float(
            torch.linalg.matrix_norm(
                (best_b @ best_a) @ (torch.eye(d_in, dtype=torch.float64) - p_sym), ord="fro"
            ).item()
        )
        if p_sym is not None
        else 0.0
    )

    nf4_bytes = int(best_q_diag["edge_payload_bytes"])
    adapter_fp16_bytes = 2 * rank_i * (d_out + d_in)
    total_edge_bytes = nf4_bytes + adapter_fp16_bytes
    fp32_bytes = 4 * d_out * d_in

    return w_quant_out, lora_b_out, lora_a_out, {
        "rank": rank_i,
        "effective_rank": r_eff,
        "block_size": blk_i,
        "double_quant_scales": dq_flag,
        "restricted_to_p0": p_sym is not None,
        "p0_nullspace_dim": int(q_null.shape[1]) if q_null is not None else d_in,
        "residual_p0_leak": p0_leak,
        "naive_weighted_err_fro": naive_weighted_fro,
        "scale_opt_weighted_err_fro": scale0_weighted_fro,
        "compensated_weighted_err_fro": final_weighted_fro,
        "weighted_error_ratio": final_weighted_fro / max(naive_weighted_fro, 1e-12) if naive_weighted_fro > 0.0 else 0.0,
        "weighted_error_reduction_frac": (
            (1.0 - final_weighted_fro / naive_weighted_fro) if naive_weighted_fro > 0.0 else 1.0
        ),
        "naive_frobenius_err": naive_fro,
        "compensated_frobenius_err": final_fro,
        "step_weighted_errors": step_weighted_errors,
        "nf4_payload_bytes": nf4_bytes,
        "adapter_fp16_bytes": adapter_fp16_bytes,
        "total_edge_bytes": total_edge_bytes,
        "fp32_bytes": fp32_bytes,
        "compression_ratio_vs_fp32": float(fp32_bytes) / float(max(1, total_edge_bytes)),
        "packed_nibbles": best_q_diag["packed_nibbles"],
        "block_scales": best_q_diag["block_scales"],
        "q_scales_uint8": best_q_diag["q_scales_uint8"],
        "superblock_max": best_q_diag["superblock_max"],
        "packed_nibbles_sha256": best_q_diag["packed_nibbles_sha256"],
    }


def compute_projected_wddm_vram_footprint(
    fp32_linear_bytes: int,
    quantized_edge_bytes: int,
    *,
    unquantized_bytes: int = 0,
    kv_and_runtime_overhead_mib: float = DEFAULT_KV_AND_RUNTIME_OVERHEAD_MIB,
    wddm_vram_ceiling_mib: float = float(STD_8GB_WDDM_CEILING_MIB),
    projected_target_param_count: int | None = None,
) -> dict[str, Any]:
    """Compute serialized NF4 + adapter edge footprint and check the ``6,656 MiB`` WDDM envelope."""
    fp32_b = _validate_positive_int(fp32_linear_bytes, "fp32_linear_bytes")
    edge_b = _validate_positive_int(quantized_edge_bytes, "quantized_edge_bytes")
    unq_b = _validate_nonneg_int(unquantized_bytes, "unquantized_bytes")
    overhead = _validate_nonneg_finite(kv_and_runtime_overhead_mib, "kv_and_runtime_overhead_mib")
    ceiling = _validate_positive_finite(wddm_vram_ceiling_mib, "wddm_vram_ceiling_mib")

    mib = 1024.0 * 1024.0
    actual_weights_mib = float(edge_b + unq_b) / mib
    actual_wddm_mib = actual_weights_mib + overhead
    eff_bpp = (float(edge_b) / float(fp32_b)) * 4.0

    projected_mib: float | None = None
    if projected_target_param_count is not None:
        n_params = _validate_positive_int(projected_target_param_count, "projected_target_param_count")
        projected_mib = (float(n_params) * eff_bpp) / mib + overhead

    fits_actual = actual_wddm_mib <= ceiling + 1e-9
    fits_projected = True if projected_mib is None else (projected_mib <= ceiling + 1e-9)

    return {
        "fp32_linear_bytes": fp32_b,
        "quantized_edge_bytes": edge_b,
        "unquantized_bytes": unq_b,
        "effective_bytes_per_param": eff_bpp,
        "compression_ratio_vs_fp32": float(fp32_b) / float(edge_b),
        "compression_ratio_vs_fp16": float(fp32_b) / (2.0 * float(edge_b)),
        "actual_weights_mib": round(actual_weights_mib, 4),
        "actual_wddm_vram_mib": round(actual_wddm_mib, 4),
        "projected_target_wddm_vram_mib": round(projected_mib, 2) if projected_mib is not None else None,
        "wddm_vram_ceiling_mib": ceiling,
        "fits_wddm_ceiling": bool(fits_actual and fits_projected),
    }


# ---------------------------------------------------------------------------
# 3. Causal-Knockout Corruptor Diagnosis & Dynamic Alveole Mitosis
# ---------------------------------------------------------------------------


def _apply_expert_knockout_inplace(
    moe_layers: Mapping[int, nn.Module],
    target_layers: Sequence[int],
    expert_idx: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Snapshot and zero output-contributing parameters of ``expert_idx`` across ``target_layers``."""
    saved: list[tuple[torch.Tensor, torch.Tensor]] = []
    for li in target_layers:
        mlp = moe_layers[li]
        num_e = int(_resolve_router_linear(mlp).weight.shape[0])
        descs = _resolve_expert_sequence(mlp, num_e)
        desc = descs[expert_idx]
        params = desc["params"]
        if desc["kind"] == "lora":
            zero_keys = ("w1_B", "w2_B", "w3_B")
        elif "w3" in params:
            zero_keys = ("w3",)
        else:
            zero_keys = ("down_proj",)
        for k in zero_keys:
            t = params[k]
            saved.append((t, t.detach().clone()))
            t.zero_()
    return saved


def _restore_expert_knockout(saved: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> None:
    """Restore exact pre-knockout expert weights."""
    for param, snap in saved:
        param.copy_(snap.to(device=param.device, dtype=param.dtype))


def _compute_mean_and_general_nll(
    domain_nll: Mapping[str, float],
    general_prefixes: Sequence[str],
) -> tuple[float, float]:
    """Compute overall mean NLL and general-prose mean NLL from domain map."""
    keys = sorted(domain_nll.keys())
    mean_nll = float(sum(float(domain_nll[k]) for k in keys) / float(len(keys)))
    gen_vals = [float(domain_nll[k]) for k in keys if _is_general_domain(k, general_prefixes)]
    if not gen_vals:
        raise ValueError(f"Missing required general-domain key matching {general_prefixes!r}")
    gen_nll = float(sum(gen_vals) / float(len(gen_vals)))
    return mean_nll, gen_nll


@torch.no_grad()
def diagnose_alveole_corruptors_and_anchors(
    model: nn.Module,
    *,
    eval_fn: Callable[[nn.Module], Mapping[str, Any] | tuple[float, Mapping[str, float]]],
    baseline_domain_nll: Mapping[str, float] | None = None,
    layer_indices: Sequence[int] | None = None,
    config: AlveoleMitosisNF4Config | None = None,
    explicit_knockout_deltas: Mapping[int, float | Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Diagnose causal-knockout corruptor cells, anchor cells, and high-deficit target domains."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if not callable(eval_fn):
        raise TypeError("eval_fn must be callable")
    cfg = config or AlveoleMitosisNF4Config()
    if not isinstance(cfg, AlveoleMitosisNF4Config):
        raise TypeError("config must be AlveoleMitosisNF4Config or None")

    moe_layers = _discover_moe_mlp_layers(model)
    target_layers = (
        sorted({_validate_nonneg_int(idx, "layer_index") for idx in layer_indices})
        if layer_indices is not None
        else sorted(moe_layers.keys())
    )
    missing = [idx for idx in target_layers if idx not in moe_layers]
    if missing:
        raise ValueError(f"Requested layer_indices not present in model: {missing}")

    num_experts_set = {int(_resolve_router_linear(moe_layers[li]).weight.shape[0]) for li in target_layers}
    if len(num_experts_set) != 1:
        raise ValueError(f"All MoE layers must share num_experts, got {sorted(num_experts_set)}")
    num_experts = next(iter(num_experts_set))

    base_nll = (
        _parse_eval_domain_nll(baseline_domain_nll)
        if baseline_domain_nll is not None
        else _parse_eval_domain_nll(eval_fn(model))
    )
    base_mean, base_gen = _compute_mean_and_general_nll(base_nll, cfg.general_prefixes)

    specialist_domains = [d for d in sorted(base_nll.keys()) if not _is_general_domain(d, cfg.general_prefixes)]
    high_deficit_domains = sorted(specialist_domains, key=lambda d: (-float(base_nll[d]), d))

    per_expert: dict[int, dict[str, Any]] = {}
    for e_idx in range(num_experts):
        if explicit_knockout_deltas is not None:
            raw_entry = explicit_knockout_deltas.get(e_idx, 0.0)
            if isinstance(raw_entry, Mapping):
                imp = float(raw_entry["knockout_improvement_nats"])
                harm = float(raw_entry.get("prose_harm_nats", 0.0))
            else:
                imp = float(raw_entry)
                harm = 0.0
            per_dom_imp: dict[str, float] = {}
        else:
            saved = _apply_expert_knockout_inplace(moe_layers, target_layers, e_idx)
            try:
                ko_nll = _parse_eval_domain_nll(eval_fn(model))
            finally:
                _restore_expert_knockout(saved)
            ko_mean, ko_gen = _compute_mean_and_general_nll(ko_nll, cfg.general_prefixes)
            imp = base_mean - ko_mean
            harm = base_gen - ko_gen
            per_dom_imp = {d: float(base_nll[d]) - float(ko_nll[d]) for d in sorted(base_nll.keys())}

        is_protected_gen = cfg.protect_general_expert and e_idx == cfg.general_expert_index
        is_corruptor = (not is_protected_gen) and (
            imp > cfg.corruptor_knockout_threshold or harm > cfg.prose_harm_threshold
        )
        per_expert[e_idx] = {
            "expert_index": e_idx,
            "is_protected_general": is_protected_gen,
            "knockout_improvement_nats": imp,
            "prose_harm_nats": harm,
            "is_corruptor": is_corruptor,
            "per_domain_knockout_delta": per_dom_imp,
        }

    corruptor_slots = sorted(
        [e for e, row in per_expert.items() if row["is_corruptor"]],
        key=lambda e: (-(per_expert[e]["knockout_improvement_nats"] + per_expert[e]["prose_harm_nats"]), e),
    )
    anchor_slots = sorted(
        [
            e
            for e, row in per_expert.items()
            if not row["is_protected_general"] and not row["is_corruptor"]
        ],
        key=lambda e: (per_expert[e]["knockout_improvement_nats"], e),
    )
    if not anchor_slots:
        anchor_slots = [e for e in range(num_experts) if e not in corruptor_slots]

    if cfg.max_recycled_experts is not None:
        selected_corruptors = corruptor_slots[: cfg.max_recycled_experts]
    else:
        selected_corruptors = corruptor_slots

    target_pool = high_deficit_domains if high_deficit_domains else sorted(base_nll.keys())
    mitosis_plan: list[dict[str, Any]] = []
    for rank_i, dst_e in enumerate(selected_corruptors):
        target_dom = target_pool[rank_i % len(target_pool)]
        best_anchors = sorted(
            anchor_slots,
            key=lambda a: (
                per_expert[a]["per_domain_knockout_delta"].get(target_dom, 0.0),
                per_expert[a]["knockout_improvement_nats"],
                a,
            ),
        )
        src_e = best_anchors[0] if best_anchors else anchor_slots[rank_i % len(anchor_slots)]
        mitosis_plan.append(
            {
                "corruptor_slot": dst_e,
                "anchor_slot": src_e,
                "target_domain": target_dom,
                "knockout_improvement_nats": per_expert[dst_e]["knockout_improvement_nats"],
                "prose_harm_nats": per_expert[dst_e]["prose_harm_nats"],
            }
        )

    return {
        "num_experts": num_experts,
        "baseline_mean_nll": base_mean,
        "baseline_general_nll": base_gen,
        "high_deficit_domains": high_deficit_domains,
        "corruptor_slots": corruptor_slots,
        "anchor_slots": anchor_slots,
        "mitosis_plan": mitosis_plan,
        "per_expert": {str(e): per_expert[e] for e in range(num_experts)},
    }


def _project_tensor_right_or_left(
    tensor: torch.Tensor,
    p_null: torch.Tensor,
    *,
    side: str,
) -> tuple[torch.Tensor, float]:
    """Project 2D tensor onto ``P_0`` on ``right`` or ``left`` and return ``(projected, leak_fro)``."""
    t64 = tensor.detach().to(device="cpu", dtype=torch.float64)
    p64 = p_null.detach().to(device="cpu", dtype=torch.float64)
    d_p = int(p64.shape[0])
    eye_p = torch.eye(d_p, dtype=torch.float64)
    if side == "right":
        if int(t64.shape[1]) != d_p:
            raise ValueError(f"Right P_0 dim {d_p} != tensor width {t64.shape[1]}")
        proj = t64 @ p64
        leak = float(torch.linalg.matrix_norm(proj @ (eye_p - p64), ord="fro").item())
    else:
        if int(t64.shape[0]) != d_p:
            raise ValueError(f"Left P_0 dim {d_p} != tensor height {t64.shape[0]}")
        proj = p64 @ t64
        leak = float(torch.linalg.matrix_norm((eye_p - p64) @ proj, ord="fro").item())
    return proj.to(device=tensor.device, dtype=tensor.dtype).contiguous(), leak


@torch.no_grad()
def recycle_corruptor_alveoles_via_mitosis(
    model: nn.Module,
    mitosis_plan: Sequence[Mapping[str, Any]],
    *,
    acts_probe_by_layer: Mapping[int, torch.Tensor] | None = None,
    domain_labels_probe: Sequence[str] | None = None,
    p_null_by_layer: Mapping[int, Any] | torch.Tensor | None = None,
    layer_indices: Sequence[int] | None = None,
    config: AlveoleMitosisNF4Config | None = None,
) -> dict[str, Any]:
    """Execute Dynamic Alveole Mitosis in-place on ``model`` inside prose null-space ``P_0``."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if isinstance(mitosis_plan, (str, bytes)) or not isinstance(mitosis_plan, Sequence):
        raise TypeError("mitosis_plan must be a sequence of pairing dicts")
    cfg = config or AlveoleMitosisNF4Config()

    moe_layers = _discover_moe_mlp_layers(model)
    target_layers = (
        sorted({_validate_nonneg_int(idx, "layer_index") for idx in layer_indices})
        if layer_indices is not None
        else sorted(moe_layers.keys())
    )
    missing = [idx for idx in target_layers if idx not in moe_layers]
    if missing:
        raise ValueError(f"Requested layer_indices not present in model: {missing}")

    if not mitosis_plan:
        return {"num_recycled": 0, "layers_modified": [], "max_p0_prose_leak": 0.0, "recycled_details": []}

    max_leak = 0.0
    recycled_details: list[dict[str, Any]] = []

    for li in target_layers:
        mlp = moe_layers[li]
        router_lin = _resolve_router_linear(mlp)
        num_e = int(router_lin.weight.shape[0])
        hidden_dim = int(router_lin.weight.shape[1])
        descs = _resolve_expert_sequence(mlp, num_e)
        px, pact = _resolve_layer_p0(p_null_by_layer, li)
        if px is not None and int(px.shape[0]) != hidden_dim:
            raise ValueError(f"Layer {li} P_0_x dim {px.shape[0]} != hidden_dim {hidden_dim}")

        layer_acts = acts_probe_by_layer.get(li) if acts_probe_by_layer is not None else None
        wg64 = router_lin.weight.detach().to(device="cpu", dtype=torch.float64).clone()

        for entry in mitosis_plan:
            dst_e = _validate_nonneg_int(entry["corruptor_slot"], "corruptor_slot")
            src_e = _validate_nonneg_int(entry["anchor_slot"], "anchor_slot")
            target_dom = str(entry["target_domain"])
            if dst_e >= num_e or src_e >= num_e:
                raise ValueError(f"Expert indices ({dst_e}, {src_e}) out of [0, {num_e})")

            dst_p = descs[dst_e]["params"]
            src_p = descs[src_e]["params"]
            kind = descs[dst_e]["kind"]

            if kind == "lora":
                for p_name in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    dst_p[p_name].copy_(src_p[p_name].detach().to(device=dst_p[p_name].device, dtype=dst_p[p_name].dtype))
                if cfg.project_mitosis_into_p0 and px is not None:
                    for in_k in ("w1_A", "w2_A"):
                        p_t, lk = _project_tensor_right_or_left(dst_p[in_k], px, side="right")
                        dst_p[in_k].copy_(p_t)
                        max_leak = max(max_leak, lk)
                    p_out, lk_o = _project_tensor_right_or_left(dst_p["w3_B"], px, side="left")
                    dst_p["w3_B"].copy_(p_out)
                    max_leak = max(max_leak, lk_o)
                if cfg.project_mitosis_into_p0 and pact is not None:
                    p_act, lk_a = _project_tensor_right_or_left(dst_p["w3_A"], pact, side="right")
                    dst_p["w3_A"].copy_(p_act)
                    max_leak = max(max_leak, lk_a)
            else:
                for k_name in sorted(dst_p.keys()):
                    dst_p[k_name].copy_(src_p[k_name].detach().to(device=dst_p[k_name].device, dtype=dst_p[k_name].dtype))
                    is_in = k_name in ("w1", "w2", "gate_proj", "up_proj")
                    if cfg.project_mitosis_into_p0:
                        if is_in and px is not None:
                            p_t, lk = _project_tensor_right_or_left(dst_p[k_name], px, side="right")
                            dst_p[k_name].copy_(p_t)
                            max_leak = max(max_leak, lk)
                        elif not is_in:
                            if pact is not None and int(dst_p[k_name].shape[1]) == int(pact.shape[0]):
                                p_t, lk = _project_tensor_right_or_left(dst_p[k_name], pact, side="right")
                                dst_p[k_name].copy_(p_t)
                                max_leak = max(max_leak, lk)
                            if px is not None and int(dst_p[k_name].shape[0]) == int(px.shape[0]):
                                p_t, lk = _project_tensor_right_or_left(dst_p[k_name], px, side="left")
                                dst_p[k_name].copy_(p_t)
                                max_leak = max(max_leak, lk)

            # Router prototype alignment toward target high-deficit domain
            proto_aligned = False
            if layer_acts is not None and domain_labels_probe is not None and cfg.router_proto_blend > 0.0:
                dom_idx = [i for i, d in enumerate(domain_labels_probe) if d == target_dom]
                if dom_idx:
                    acts_t = layer_acts.detach().to(device="cpu", dtype=torch.float64)
                    if acts_t.ndim == 3 and len(domain_labels_probe) == int(acts_t.shape[0]):
                        dom_mean = acts_t[dom_idx].mean(dim=(0, 1))
                    elif acts_t.ndim == 2 and len(domain_labels_probe) == int(acts_t.shape[0]):
                        dom_mean = acts_t[dom_idx].mean(dim=0)
                    else:
                        dom_mean = None
                    if dom_mean is not None and int(dom_mean.numel()) == hidden_dim:
                        if px is not None:
                            dom_mean = px.to(torch.float64) @ dom_mean
                        d_norm = float(dom_mean.norm().item())
                        if d_norm > 1e-12:
                            row_norms = wg64.norm(dim=1)
                            target_scale = max(
                                float(wg64[dst_e].norm().item()),
                                float(wg64[src_e].norm().item()),
                                float(row_norms.mean().item()),
                                1e-2,
                            )
                            proto_v = (dom_mean / d_norm) * target_scale
                            old_row = wg64[dst_e].clone()
                            if cfg.project_router_into_p0 and px is not None:
                                old_row = px.to(torch.float64) @ old_row
                            delta_r = cfg.router_proto_blend * (proto_v - old_row)
                            wg64[dst_e] = old_row + delta_r
                            proto_aligned = True

                            if cfg.subspace_split_beta > 0.0 and kind == "lora":
                                dir_unit = dom_mean / d_norm
                                old_a1 = dst_p["w1_A"].detach().to(device="cpu", dtype=torch.float64)
                                a1_norm = max(float(old_a1[0].norm().item()), 1e-4)
                                old_a1[0] = (1.0 - cfg.subspace_split_beta) * old_a1[0] + (
                                    cfg.subspace_split_beta * a1_norm * dir_unit
                                )
                                if px is not None:
                                    old_a1 = old_a1 @ px.to(torch.float64)
                                dst_p["w1_A"].copy_(old_a1.to(device=dst_p["w1_A"].device, dtype=dst_p["w1_A"].dtype))

            if li == target_layers[0]:
                recycled_details.append(
                    {
                        "corruptor_slot": dst_e,
                        "anchor_slot": src_e,
                        "target_domain": target_dom,
                        "router_proto_aligned": proto_aligned,
                    }
                )

        router_lin.weight.copy_(wg64.to(device=router_lin.weight.device, dtype=router_lin.weight.dtype))

    return {
        "num_recycled": len(recycled_details),
        "layers_modified": list(target_layers),
        "max_p0_prose_leak": max_leak,
        "recycled_details": recycled_details,
    }


# ---------------------------------------------------------------------------
# 4. BOS Attention-Sink Stability & Whole-Model NF4 Edge Export
# ---------------------------------------------------------------------------


@torch.no_grad()
def _measure_bos_attention_sink_mass(
    model: nn.Module,
    ids_cal: torch.Tensor | None,
    *,
    score_from: int,
) -> float:
    """Measure mean causal attention weight placed on BOS token ``0`` across attention layers."""
    if ids_cal is None or int(ids_cal.shape[0]) == 0 or int(ids_cal.shape[1]) < 2:
        return 0.0
    ids_sub = ids_cal[: min(8, int(ids_cal.shape[0]))]
    start_pos = max(1, min(int(ids_sub.shape[1]) - 1, int(score_from)))

    blocks = getattr(model, "blocks", None)
    if isinstance(blocks, nn.ModuleList) and hasattr(model, "cos_cached") and hasattr(model, "sin_cached"):
        cos_c = model.cos_cached.to(device=ids_sub.device)
        sin_c = model.sin_cached.to(device=ids_sub.device)
        attn_inputs: list[tuple[Any, torch.Tensor]] = []
        hooks = []
        for blk in blocks:
            attn_mod = getattr(blk, "attn", None)
            if attn_mod is not None and hasattr(attn_mod, "q") and hasattr(attn_mod, "k"):
                hooks.append(
                    attn_mod.register_forward_pre_hook(
                        lambda m, args: attn_inputs.append((m, args[0].detach()))
                    )
                )
        try:
            _ = model(ids_sub)
        finally:
            for hk in hooks:
                hk.remove()
        if not attn_inputs:
            return 0.0
        masses: list[float] = []
        for attn_mod, x_in in attn_inputs:
            b_sz, t_len, _ = x_in.shape
            q = attn_mod.q(x_in.to(dtype=attn_mod.q.weight.dtype)).view(
                b_sz, t_len, attn_mod.num_heads, attn_mod.head_dim
            ).transpose(1, 2)
            k = attn_mod.k(x_in.to(dtype=attn_mod.k.weight.dtype)).view(
                b_sz, t_len, attn_mod.num_kv_heads, attn_mod.head_dim
            ).transpose(1, 2)
            c_sl = cos_c[:, :, :t_len].to(q.dtype)
            s_sl = sin_c[:, :, :t_len].to(q.dtype)
            half = attn_mod.head_dim // 2
            q_rot = q * c_sl + torch.cat((-q[..., half:], q[..., :half]), dim=-1) * s_sl
            k_rot = k * c_sl + torch.cat((-k[..., half:], k[..., :half]), dim=-1) * s_sl
            rep = max(1, attn_mod.num_heads // attn_mod.num_kv_heads)
            k_exp = k_rot.repeat_interleave(rep, dim=1)
            scores = torch.matmul(q_rot.float(), k_exp.float().transpose(-2, -1)) / math.sqrt(float(attn_mod.head_dim))
            causal = torch.triu(torch.full((t_len, t_len), float("-inf"), device=scores.device), diagonal=1)
            probs = torch.softmax(scores + causal, dim=-1)
            masses.append(float(probs[:, :, start_pos:, 0].mean().item()))
        return float(sum(masses) / float(len(masses)))

    return 0.0


def _collect_exportable_linears(
    model: nn.Module,
    cfg: AlveoleMitosisNF4Config,
) -> list[dict[str, Any]]:
    """Discover eligible ``nn.Linear`` layers for NF4 edge export with role metadata."""
    tied_head_weight: torch.Tensor | None = None
    if hasattr(model, "wte") and hasattr(model, "head") and model.head.weight is model.wte.weight:
        tied_head_weight = model.head.weight

    entries: list[dict[str, Any]] = []
    for full_name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if mod.weight.numel() < cfg.min_linear_numel_to_quantize:
            continue
        is_router = full_name.endswith(".router") or ".router." in full_name or full_name.endswith(".wg")
        is_head = full_name in ("head", "lm_head") or full_name.endswith((".head", ".lm_head"))
        is_attn = ".attn." in f".{full_name}." or ".self_attn." in f".{full_name}."
        is_base_mlp = ".base_mlp." in f".{full_name}."
        is_expert = _EXPERT_PATH_RE.search(full_name) is not None

        if is_router and not cfg.quantize_router:
            continue
        if is_head and (not cfg.quantize_lm_head or mod.weight is tied_head_weight):
            continue
        if is_attn and not cfg.quantize_attention:
            continue
        if is_base_mlp and not cfg.quantize_base_mlp:
            continue
        if is_expert and not cfg.quantize_dense_experts:
            continue

        layer_m = _LAYER_PATH_RE.search(full_name)
        layer_idx = int(layer_m.group(1)) if layer_m is not None else None
        if ".experts_w1." in full_name:
            attr_name = "w1"
        elif ".experts_w2." in full_name:
            attr_name = "w2"
        elif ".experts_w3." in full_name:
            attr_name = "w3"
        else:
            attr_name = full_name.rsplit(".", 1)[-1]
        entries.append(
            {
                "name": full_name or "root",
                "module": mod,
                "layer_idx": layer_idx,
                "is_expert": is_expert,
                "attr": attr_name,
            }
        )
    return entries


@torch.no_grad()
def export_model_to_compensated_nf4_edge(
    model: nn.Module,
    *,
    calibration_ids: torch.Tensor | None = None,
    general_mask: Sequence[bool] | None = None,
    acts_by_linear: Mapping[str, torch.Tensor] | None = None,
    p_null_by_layer: Mapping[int, Any] | torch.Tensor | None = None,
    hessian_diag_by_layer: Mapping[str | int, torch.Tensor] | None = None,
    config: AlveoleMitosisNF4Config | None = None,
) -> dict[str, Any]:
    """Export eligible linear weights to diagonal prose-Hessian compensated 4-bit NF4 edge blocks."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    cfg = config or AlveoleMitosisNF4Config()
    targets = _collect_exportable_linears(model, cfg)
    if not targets:
        raise ValueError("No eligible nn.Linear modules found for NF4 edge export")

    captured_prose: dict[str, torch.Tensor] = {}
    captured_all: dict[str, torch.Tensor] = {}
    if calibration_ids is not None:
        if general_mask is None or len(general_mask) != int(calibration_ids.shape[0]):
            raise ValueError("general_mask must match calibration_ids batch dimension")
        g_mask_t = torch.tensor(list(general_mask), dtype=torch.bool)
        start_pos = max(0, cfg.score_from - 1)
        all_buf: dict[str, list[torch.Tensor]] = {t["name"]: [] for t in targets}
        prose_buf: dict[str, list[torch.Tensor]] = {t["name"]: [] for t in targets}
        n_chunks = int(calibration_ids.shape[0])
        hooks = []
        for t in targets:
            t_name = t["name"]

            def _make_hook(nm: str) -> Callable[[nn.Module, tuple[Any, ...]], None]:
                def _pre_hook(_m: nn.Module, args: tuple[Any, ...]) -> None:
                    x_in = args[0].detach().to(device="cpu", dtype=torch.float32)
                    if x_in.ndim == 3:
                        xs = x_in[:, start_pos:-1, :]
                        all_buf[nm].append(xs.reshape(-1, int(xs.shape[-1])))
                        if int(xs.shape[0]) == n_chunks:
                            prose_buf[nm].append(xs[g_mask_t].reshape(-1, int(xs.shape[-1])))
                    elif x_in.ndim == 2:
                        all_buf[nm].append(x_in)

                return _pre_hook

            hooks.append(t["module"].register_forward_pre_hook(_make_hook(t_name)))

        try:
            _ = model(calibration_ids)
        finally:
            for hk in hooks:
                hk.remove()

        for nm in all_buf:
            if all_buf[nm]:
                captured_all[nm] = torch.cat(all_buf[nm], dim=0).contiguous()
            if prose_buf[nm]:
                captured_prose[nm] = torch.cat(prose_buf[nm], dim=0).contiguous()

    fp32_linear_bytes = 0
    nf4_bytes_total = 0
    adapter_bytes_total = 0
    total_edge_bytes = 0
    naive_weighted_errs: list[float] = []
    comp_weighted_errs: list[float] = []
    max_residual_p0_leak = 0.0
    per_layer_payloads: dict[str, dict[str, Any]] = {}

    for entry in targets:
        nm = entry["name"]
        lin: nn.Linear = entry["module"]
        d_out, d_in = int(lin.weight.shape[0]), int(lin.weight.shape[1])
        fp32_linear_bytes += 4 * d_out * d_in

        h_diag: torch.Tensor | None = None
        if hessian_diag_by_layer is not None:
            cand_h = hessian_diag_by_layer.get(nm, hessian_diag_by_layer.get(entry["layer_idx"]))
            if cand_h is not None and (
                (cand_h.ndim == 1 and int(cand_h.shape[0]) == d_in)
                or (cand_h.ndim == 2 and int(cand_h.shape[1]) == d_in)
            ):
                h_diag = _normalize_1d_hessian_diag(cand_h, d_in, f"hessian_diag[{nm}]")

        if h_diag is None:
            x_all = captured_all.get(nm)
            x_prose = captured_prose.get(nm)
            if x_prose is None and acts_by_linear is not None and nm in acts_by_linear:
                x_prose = acts_by_linear[nm]
            if x_prose is not None:
                h_diag = compute_diagonal_prose_hessian(
                    x_prose,
                    x_all=x_all,
                    prose_weight=cfg.prose_hessian_weight,
                    damping=cfg.hessian_damping,
                )
            elif x_all is not None:
                h_diag = compute_diagonal_prose_hessian(
                    x_all,
                    prose_weight=1.0,
                    damping=cfg.hessian_damping,
                )
            else:
                h_diag = torch.ones(d_in, dtype=torch.float64)

        p_unit: torch.Tensor | None = None
        if cfg.project_nf4_residual_in_p0 and entry["is_expert"] and entry["layer_idx"] is not None:
            px, pact = _resolve_layer_p0(p_null_by_layer, int(entry["layer_idx"]))
            if entry["attr"] in ("w1", "w2", "gate_proj", "up_proj") and px is not None and int(px.shape[0]) == d_in:
                p_unit = px
            elif entry["attr"] in ("w3", "down_proj") and pact is not None and int(pact.shape[0]) == d_in:
                p_unit = pact

        q_w, lora_b, lora_a, comp_diag = compensate_nf4_with_prose_hessian(
            lin.weight,
            h_diag,
            rank=cfg.compensation_rank,
            block_size=cfg.nf4_block_size,
            num_alternating_steps=cfg.num_alternating_steps,
            double_quant_scales=cfg.double_quant_scales,
            p_null=p_unit,
        )
        recon = (q_w.to(torch.float32) + (lora_b.to(torch.float32) @ lora_a.to(torch.float32))).to(
            device=lin.weight.device, dtype=lin.weight.dtype
        )
        lin.weight.copy_(recon)

        nf4_bytes_total += int(comp_diag["nf4_payload_bytes"])
        adapter_bytes_total += int(comp_diag["adapter_fp16_bytes"])
        total_edge_bytes += int(comp_diag["total_edge_bytes"])
        naive_weighted_errs.append(float(comp_diag["naive_weighted_err_fro"]))
        comp_weighted_errs.append(float(comp_diag["compensated_weighted_err_fro"]))
        max_residual_p0_leak = max(max_residual_p0_leak, float(comp_diag["residual_p0_leak"]))
        per_layer_payloads[nm] = {
            "shape": [d_out, d_in],
            "restricted_to_p0": bool(comp_diag["restricted_to_p0"]),
            "residual_p0_leak": float(comp_diag["residual_p0_leak"]),
            "naive_weighted_err_fro": float(comp_diag["naive_weighted_err_fro"]),
            "compensated_weighted_err_fro": float(comp_diag["compensated_weighted_err_fro"]),
            "weighted_error_reduction_frac": float(comp_diag["weighted_error_reduction_frac"]),
            "total_edge_bytes": int(comp_diag["total_edge_bytes"]),
            "packed_nibbles_sha256": str(comp_diag["packed_nibbles_sha256"]),
        }

    quantized_ids = {id(e["module"].weight) for e in targets}
    unquantized_bytes = sum(
        int(p.numel() * p.element_size()) for p in model.parameters() if id(p) not in quantized_ids
    )
    wddm_stats = compute_projected_wddm_vram_footprint(
        fp32_linear_bytes,
        total_edge_bytes,
        unquantized_bytes=unquantized_bytes,
        kv_and_runtime_overhead_mib=cfg.kv_and_runtime_overhead_mib,
        wddm_vram_ceiling_mib=cfg.wddm_vram_ceiling_mib,
        projected_target_param_count=cfg.projected_target_param_count,
    )
    sum_naive = sum(naive_weighted_errs)
    sum_comp = sum(comp_weighted_errs)

    return {
        "num_quantized_linears": len(targets),
        "nf4_payload_bytes": nf4_bytes_total,
        "adapter_fp16_bytes": adapter_bytes_total,
        "total_edge_bytes": total_edge_bytes,
        "sum_naive_weighted_err_fro": sum_naive,
        "sum_compensated_weighted_err_fro": sum_comp,
        "overall_weighted_error_reduction_frac": (
            (1.0 - sum_comp / sum_naive) if sum_naive > 0.0 else 1.0
        ),
        "max_residual_p0_leak": max_residual_p0_leak,
        "wddm_footprint": wddm_stats,
        "per_linear": per_layer_payloads,
    }


# ---------------------------------------------------------------------------
# 5. End-to-End ANSE-Gated Mitosis + NF4 Export Orchestrator
# ---------------------------------------------------------------------------


def _resolve_config(
    config: AlveoleMitosisNF4Config | None,
    overrides: Mapping[str, Any],
) -> AlveoleMitosisNF4Config:
    if config is not None and not isinstance(config, AlveoleMitosisNF4Config):
        raise TypeError(f"config must be AlveoleMitosisNF4Config or None, got {type(config).__name__}")
    raw = asdict(config) if config is not None else asdict(AlveoleMitosisNF4Config())
    for k, v in overrides.items():
        if v is not None:
            raw[k] = v
    return AlveoleMitosisNF4Config(**raw)


def _calibrate_final_layernorm(
    model: nn.Module,
    anchor_ln: nn.Module,
    ids_cal: torch.Tensor,
    eids_cal: Sequence[str],
    cfg: AlveoleMitosisNF4Config,
) -> dict[str, Any]:
    """Run 2-matmul affine LayerNorm + folded temperature calibration."""
    h_pre = forward_model_pre_ln(model, ids_cal, batch_size=cfg.batch_size)
    start_pos = max(0, cfg.score_from - 1)
    h_scored = h_pre[:, start_pos:-1, :]
    t_scored = ids_cal[:, start_pos + 1 :]
    gen_mask = [_is_general_domain(e, cfg.general_prefixes) for e in eids_cal]
    w_cand = model.ln.weight.detach().clone()
    b_cand = getattr(model.ln, "bias", None)
    b_cand_c = b_cand.detach().clone() if b_cand is not None else None
    b_anc = getattr(anchor_ln, "bias", None)
    grid_res = fast_affine_layernorm_temperature_grid(
        h_scored,
        anchor_ln.weight.detach(),
        b_anc.detach() if b_anc is not None else None,
        w_cand,
        b_cand_c,
        model.head.weight.detach(),
        t_scored,
        gen_mask,
        eps=float(getattr(model.ln, "eps", 1e-5)),
        lambda_grid=cfg.lambda_grid,
        tau_grid=cfg.tau_grid,
        max_general_drift_rel=cfg.calibrator_max_general_drift_rel,
        seq_batch_size=cfg.batch_size,
    )
    lam_opt = float(grid_res["best_lambda_ln"])
    tau_opt = float(grid_res["best_tau_temp"])
    w_anc64 = anchor_ln.weight.detach().to(torch.float64)
    w_interp = w_anc64 + lam_opt * (w_cand.to(torch.float64) - w_anc64)
    b_interp = (
        b_anc.detach().to(torch.float64) + lam_opt * (b_cand_c.to(torch.float64) - b_anc.detach().to(torch.float64))
        if (b_anc is not None and b_cand_c is not None)
        else None
    )
    w_folded, b_folded = fold_temperature_into_norm_params(w_interp, b_interp, tau_opt)
    model.ln.weight.copy_(w_folded.to(device=model.ln.weight.device, dtype=model.ln.weight.dtype))
    if b_cand is not None and b_folded is not None:
        b_cand.copy_(b_folded.to(device=b_cand.device, dtype=b_cand.dtype))
    return {"best_lambda_ln": lam_opt, "best_tau_temp": tau_opt, "guard_feasible": bool(grid_res["guard_feasible"])}


@torch.no_grad()
def execute_verified_alveole_mitosis_and_nf4_export(
    model: nn.Module,
    *,
    calibration_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_fn: Callable[[nn.Module], Mapping[str, Any] | tuple[float, Mapping[str, float]]] | None = None,
    baseline_domain_nll: Mapping[str, float] | None = None,
    p_null_by_layer: Mapping[int, Any] | torch.Tensor | None = None,
    hessian_diag_by_layer: Mapping[str | int, torch.Tensor] | None = None,
    explicit_knockout_deltas: Mapping[int, float | Mapping[str, float]] | None = None,
    mitosis_plan: Sequence[Mapping[str, Any]] | None = None,
    layer_indices: Sequence[int] | None = None,
    bos_sink_fn: Callable[[nn.Module], float] | None = None,
    config: AlveoleMitosisNF4Config | None = None,
    quantize_model_to_nf4: bool | None = None,
    wddm_vram_ceiling_mib: float | None = None,
    inplace: bool | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Execute Dynamic Alveole Mitosis + Prose-Hessian NF4 Export under ANSE gating & rollback."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError("eval_fn must be callable or None")
    if bos_sink_fn is not None and not callable(bos_sink_fn):
        raise TypeError("bos_sink_fn must be callable or None")

    cfg = _resolve_config(
        config,
        {
            "quantize_model_to_nf4": quantize_model_to_nf4,
            "wddm_vram_ceiling_mib": wddm_vram_ceiling_mib,
            "inplace": inplace,
        },
    )
    work_model = model if cfg.inplace else copy.deepcopy(model)
    moe_layers = _discover_moe_mlp_layers(work_model)
    target_layers = (
        sorted({_validate_nonneg_int(idx, "layer_index") for idx in layer_indices})
        if layer_indices is not None
        else sorted(moe_layers.keys())
    )
    missing = [idx for idx in target_layers if idx not in moe_layers]
    if missing:
        raise ValueError(f"Requested layer_indices not present in model: {missing}")

    ids_cal: torch.Tensor | None = None
    eids_cal: list[str] = []
    ids_eval: torch.Tensor | None = None
    eids_eval: list[str] = []
    if calibration_chunks is not None:
        ids_cal, eids_cal = _normalize_chunks(
            calibration_chunks,
            "calibration_chunks",
            general_prefixes=cfg.general_prefixes,
            score_from=cfg.score_from,
            require_both_groups=True,
        )
        if eval_chunks is not None:
            ids_eval, eids_eval = _normalize_chunks(
                eval_chunks,
                "eval_chunks",
                general_prefixes=cfg.general_prefixes,
                score_from=cfg.score_from,
                require_both_groups=False,
            )
        else:
            ids_eval, eids_eval = ids_cal, eids_cal
    elif eval_fn is None:
        raise ValueError("Provide calibration_chunks or eval_fn")

    def _run_eval(target_m: nn.Module) -> dict[str, float]:
        if eval_fn is not None:
            return _parse_eval_domain_nll(eval_fn(target_m))
        assert ids_eval is not None
        return _eval_crowd_model_domain_nlls(
            target_m, ids_eval, eids_eval, score_from=cfg.score_from, batch_size=cfg.batch_size
        )

    was_training = work_model.training
    work_model.eval()
    try:
        parent_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in work_model.state_dict().items()}
        parent_sha = compute_state_sha256(parent_snapshot)
        anchor_ln_copy = copy.deepcopy(work_model.ln) if hasattr(work_model, "ln") else None

        nll_parent = _run_eval(work_model)
        ref_nll = _parse_eval_domain_nll(baseline_domain_nll) if baseline_domain_nll is not None else nll_parent
        eval_parent = _score_domain_nll_vs_ref(ref_nll, nll_parent, cfg=cfg)

        bos_base = (
            float(bos_sink_fn(work_model))
            if bos_sink_fn is not None
            else _measure_bos_attention_sink_mass(work_model, ids_cal, score_from=cfg.score_from)
        )

        # 1. Causal-Knockout Diagnosis
        diag = diagnose_alveole_corruptors_and_anchors(
            work_model,
            eval_fn=_run_eval,
            baseline_domain_nll=nll_parent,
            layer_indices=target_layers,
            config=cfg,
            explicit_knockout_deltas=explicit_knockout_deltas,
        )
        effective_plan = list(mitosis_plan) if mitosis_plan is not None else list(diag["mitosis_plan"])

        # 2. Capture MLP probe activations & derive prose null-space P_0
        probe_acts: dict[int, torch.Tensor] | None = None
        if ids_cal is not None:
            probe_acts, _ = _capture_mlp_inputs_and_pre_ln(
                work_model, moe_layers, ids_cal, score_from=cfg.score_from, batch_size=cfg.batch_size
            )
        resolved_p0: dict[int, Any] = {}
        if p_null_by_layer is not None:
            for li in target_layers:
                resolved_p0[li] = _resolve_layer_p0(p_null_by_layer, li)
        elif probe_acts is not None and ids_cal is not None:
            gen_flags = torch.tensor(
                [_is_general_domain(e, cfg.general_prefixes) for e in eids_cal], dtype=torch.bool
            )
            n_ch = int(ids_cal.shape[0])
            for li in target_layers:
                acts_l = probe_acts[li]
                rows_p = acts_l.reshape(n_ch, -1, int(acts_l.shape[-1]))[gen_flags].reshape(-1, int(acts_l.shape[-1]))
                px, _ = compute_prose_nullspace_projector_scaled(
                    rows_p,
                    sv_rel_tol=cfg.nullspace_sv_rel_tol,
                    variance_threshold=cfg.nullspace_variance_threshold,
                    max_active_modes=cfg.nullspace_max_modes,
                )
                resolved_p0[li] = (px, None)
        else:
            resolved_p0 = {li: (None, None) for li in target_layers}

        p_null_arg = {li: {"x": px, "act": pact} for li, (px, pact) in resolved_p0.items()}

        # 3. Dynamic Alveole Mitosis
        reshaped_probe: dict[int, torch.Tensor] | None = None
        if probe_acts is not None and ids_cal is not None:
            n_ch = int(ids_cal.shape[0])
            reshaped_probe = {
                li: probe_acts[li].reshape(n_ch, -1, int(probe_acts[li].shape[-1])) for li in target_layers
            }
        mitosis_report = recycle_corruptor_alveoles_via_mitosis(
            work_model,
            effective_plan,
            acts_probe_by_layer=reshaped_probe,
            domain_labels_probe=eids_cal if ids_cal is not None else None,
            p_null_by_layer=p_null_arg,
            layer_indices=target_layers,
            config=cfg,
        )

        # 4. Pre-NF4 LayerNorm calibration (if available)
        ln_meta: dict[str, Any] | None = None
        if cfg.calibrate_layernorm and ids_cal is not None and anchor_ln_copy is not None and hasattr(work_model, "head"):
            ln_meta = _calibrate_final_layernorm(work_model, anchor_ln_copy, ids_cal, eids_cal, cfg)

        # 5. Prose-Hessian Compensated 4-Bit NF4 Edge Export
        nf4_report: dict[str, Any] | None = None
        wddm_ok = True
        if cfg.quantize_model_to_nf4:
            gen_mask_list = [_is_general_domain(e, cfg.general_prefixes) for e in eids_cal] if ids_cal is not None else None
            nf4_report = export_model_to_compensated_nf4_edge(
                work_model,
                calibration_ids=ids_cal,
                general_mask=gen_mask_list,
                p_null_by_layer=p_null_arg,
                hessian_diag_by_layer=hessian_diag_by_layer,
                config=cfg,
            )
            wddm_ok = bool(nf4_report["wddm_footprint"]["fits_wddm_ceiling"])
            if cfg.calibrate_layernorm and ids_cal is not None and anchor_ln_copy is not None and hasattr(work_model, "head"):
                ln_meta = _calibrate_final_layernorm(work_model, anchor_ln_copy, ids_cal, eids_cal, cfg)

        bos_after = (
            float(bos_sink_fn(work_model))
            if bos_sink_fn is not None
            else _measure_bos_attention_sink_mass(work_model, ids_cal, score_from=cfg.score_from)
        )
        bos_drift = abs(bos_after - bos_base)
        bos_ok = math.isfinite(bos_drift) and (bos_drift <= cfg.max_bos_sink_drift + 1e-12)

        cand_sha = compute_state_sha256(work_model.state_dict())
        eval_error: str | None = None
        cand_eval: dict[str, Any]
        energy_ok = False
        try:
            cand_nll = _run_eval(work_model)
            cand_eval = _score_domain_nll_vs_ref(ref_nll, cand_nll, cfg=cfg)
            cand_e = float(cand_eval["anse_energy"])
            parent_e = float(eval_parent["anse_energy"])
            energy_ok = (
                cand_e < parent_e - cfg.min_energy_improvement
                if cfg.require_energy_improvement
                else cand_e <= parent_e + 1e-12
            )
        except Exception as exc:
            eval_error = f"{type(exc).__name__}: {exc}"
            cand_eval = {
                "nll_mean": None,
                "base_nll_mean": float(eval_parent["base_nll_mean"]),
                "mean_diff_vs_ref": None,
                "general_ratio": None,
                "general_pct_vs_ref": None,
                "worst_domain_id": "",
                "worst_domain_ratio": None,
                "worst_domain_pct_vs_ref": None,
                "guard_ok": False,
                "anse_energy": float(cfg.maximum_pain_energy),
                "per_domain": {},
            }

        if cfg.enforce_wddm_budget and not wddm_ok:
            status = "ROLLED_BACK_WDDM_VRAM_EXCEEDED"
            accepted = False
            rolled_back = True
        elif not bos_ok:
            status = "ROLLED_BACK_BOS_SINK_DRIFT"
            accepted = False
            rolled_back = True
        elif eval_error is not None:
            status = "ROLLED_BACK_EVAL_FAILURE"
            accepted = False
            rolled_back = True
        elif not bool(cand_eval["guard_ok"]):
            status = "ROLLED_BACK_GUARD_VIOLATION"
            accepted = False
            rolled_back = True
        elif not energy_ok:
            status = "ROLLED_BACK_ENERGY_NON_IMPROVING"
            accepted = False
            rolled_back = True
        else:
            status = "ACCEPTED"
            accepted = True
            rolled_back = False

        if rolled_back:
            cur_sd = work_model.state_dict()
            work_model.load_state_dict(
                {k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in parent_snapshot.items()}
            )
            final_sha = compute_state_sha256(work_model.state_dict())
            if final_sha != parent_sha:
                raise RuntimeError("Atomic bitwise rollback SHA-256 mismatch")
            final_eval = dict(eval_parent)
            final_eval["anse_energy"] = float(cfg.maximum_pain_energy)
            final_eval["guard_ok"] = False
        else:
            final_sha = cand_sha
            final_eval = cand_eval

        receipt_core: dict[str, Any] = {
            "object": ALVEOLE_MITOSIS_RECEIPT_OBJECT,
            "schema": ALVEOLE_MITOSIS_EXPORTER_SCHEMA,
            "status": status,
            "accepted": accepted,
            "rolled_back": rolled_back,
            "parent_state_sha256": parent_sha,
            "candidate_state_sha256": cand_sha,
            "final_state_sha256": final_sha,
            "target_layers": target_layers,
            "diagnosis": diag,
            "mitosis": mitosis_report,
            "layernorm_calibration": ln_meta,
            "nf4_export": {k: v for k, v in nf4_report.items() if k != "per_linear"} if nf4_report else None,
            "bos_sink_mass_base": bos_base,
            "bos_sink_mass_after": bos_after,
            "bos_sink_abs_drift": bos_drift,
            "parent_eval": eval_parent,
            "candidate_eval": cand_eval,
            "final_eval": final_eval,
            "eval_error": eval_error,
        }
        receipt_sha = compute_payload_sha256(receipt_core)
        return work_model, {**receipt_core, "receipt_sha256": receipt_sha}
    finally:
        work_model.train(was_training)


__all__ = [
    "ALVEOLE_MITOSIS_EXPORTER_SCHEMA",
    "ALVEOLE_MITOSIS_RECEIPT_OBJECT",
    "ANSE_MAXIMUM_PAIN",
    "CANONICAL_NF4_LEVELS",
    "STD_8GB_WDDM_CEILING_MIB",
    "AlveoleMitosisNF4Config",
    "VRAMCeilingExceededError",
    "compensate_nf4_with_prose_hessian",
    "compute_diagonal_prose_hessian",
    "compute_projected_wddm_vram_footprint",
    "diagnose_alveole_corruptors_and_anchors",
    "execute_verified_alveole_mitosis_and_nf4_export",
    "export_model_to_compensated_nf4_edge",
    "quantize_dequantize_nf4_blockwise",
    "recycle_corruptor_alveoles_via_mitosis",
    "verify_alveole_mitosis_nf4_receipt_provenance",
]

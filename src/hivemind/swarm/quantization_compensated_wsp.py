# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Quantization-Compensated Weight-Space Programming (WSP) and LoftQ Prose-Nullspace LoRA.

Ensures training-free Weight-Space Programming (WSP) edits survive 4-bit and 8-bit
consumer-GPU quantization (HiveMind 8 GB VRAM Golden Invariant) without re-violating
the ``<= +1.0%`` general-domain NLL drift guard:

1. ``symmetric_group_quantize_dequantize(weight, *, bits=4, group_size=32)``:
   Deterministic CPU symmetric groupwise block quantizer supporting 4-bit
   (``[-7, 7]``) and 8-bit (``[-127, 127]``) quantization with clean zero-padding
   and slicing for non-divisible column dimensions and exact Frobenius / max-abs
   reconstruction diagnostics.
2. ``compensate_quantization_error_via_lora_svd(w_target, *, rank, ...)``:
   Alternating LoftQ-style quantized base + thin QR-SVD rank-``r`` LoRA residual
   decomposition with optional prose-covariance right-weighting
   (``C = I + prose_weight * M_prose_norm``) so quantization error along
   general-prose principal modes is preferentially absorbed into the high-precision
   LoRA adapter with monotone ANSE residual decrease.
3. ``quantize_and_verify_moe_wsp_layer(...)``:
   Quantizes WSP-edited MoE and dense layers (``CrowdLoRAMoEMLP``, ``CrowdRuntimeMLP``,
   ``CrowdRuntimeBlock``, ``CrowdV1WithEngram``, or ``nn.Linear``) under an ANSE
   ``<= +1.0%`` general-domain drift guard with atomic FP32/BF16 rollback on guard
   violation or non-finite output and canonical SHA-256 provenance receipts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.swarm.lora_svd_mode_surgery import decompose_lora_qr_svd

SUPPORTED_QUANT_BITS: tuple[int, ...] = (4, 8)
DEFAULT_QUANT_BITS: int = 4
DEFAULT_GROUP_SIZE: int = 32
DEFAULT_LORA_RANK: int = 8
DEFAULT_ALTERNATING_STEPS: int = 3
DEFAULT_PROSE_WEIGHT: float = 2.0
GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_MAX_DOMAIN_DRIFT: float = 0.01
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.0 + DEFAULT_MAX_DOMAIN_DRIFT
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_ANSE_MAXIMUM_PAIN: float = 1e6
DEFAULT_ANSE_SOFT_MARGIN_PENALTY: float = 0.05
QUANT_RECEIPT_SCHEMA: str = "hivemind.quantization_compensated_wsp/1"

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_LARGE_SVD_DIM_THRESHOLD: int = 512
_SUBSPACE_OVERSAMPLE: int = 16
_SUBSPACE_POWER_ITERS: int = 2


# ---------------------------------------------------------------------------
# 1. Validation & Canonical Provenance Helpers
# ---------------------------------------------------------------------------


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a strictly positive integer (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


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
        raise ValueError(f"{name} must be finite and >= 0.0, got {value!r}")
    return out


def _validate_positive_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite strictly positive real number."""
    out = _validate_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_quant_bits(bits: Any) -> int:
    """Validate supported bit-width (4-bit or 8-bit symmetric quantization)."""
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise TypeError(f"bits must be an int, got {type(bits).__name__}")
    if bits not in SUPPORTED_QUANT_BITS:
        raise ValueError(f"bits must be in {SUPPORTED_QUANT_BITS}, got {bits}")
    return int(bits)


def _validate_finite_2d(tensor: Any, name: str) -> tuple[torch.Tensor, torch.dtype, torch.device]:
    """Validate a non-empty 2D floating-point tensor and return ``(cpu_fp64, orig_dtype, orig_device)``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D tensor, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    w64 = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(w64).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return w64, tensor.dtype, tensor.device


def _validate_square_psd_moment(matrix: Any, expected_dim: int, name: str) -> torch.Tensor:
    """Validate a square finite PSD moment matrix of shape ``(expected_dim, expected_dim)``."""
    m64, _, _ = _validate_finite_2d(matrix, name)
    if m64.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"{name} must have shape ({expected_dim}, {expected_dim}), got {tuple(m64.shape)}"
        )
    sym = (0.5 * (m64 + m64.T)).contiguous()
    evals = torch.linalg.eigvalsh(sym)
    max_ev = max(float(evals[-1].item()), 0.0)
    min_ev = float(evals[0].item())
    if min_ev < -1e-4 * max(1.0, max_ev):
        raise ValueError(f"{name} must be positive semi-definite (min eigenvalue {min_ev:.3e})")
    return sym


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically with strict finite-number enforcement."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


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
        val = state[key]
        t_cpu = val.detach().to(device="cpu").contiguous()
        if t_cpu.is_floating_point() and not bool(torch.isfinite(t_cpu).all().item()):
            raise ValueError(f"module state tensor {key!r} contains non-finite values")
        digest.update(f"{key}:{t_cpu.dtype}:{tuple(t_cpu.shape)}:".encode())
        digest.update(t_cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_quantization_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify canonical ``receipt_sha256`` provenance on a quantization receipt."""
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
class QuantizationWSPConfig:
    """Configuration governing quantization-compensated WSP and ANSE verification."""

    bits: int = DEFAULT_QUANT_BITS
    group_size: int = DEFAULT_GROUP_SIZE
    rank: int = DEFAULT_LORA_RANK
    num_alternating_steps: int = DEFAULT_ALTERNATING_STEPS
    prose_weight: float = DEFAULT_PROSE_WEIGHT
    use_lora_compensation: bool = True
    max_domain_drift: float = DEFAULT_MAX_DOMAIN_DRIFT
    anse_safe_e0_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain: float = DEFAULT_ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY
    general_prefix: str = GENERAL_EXPERT_PREFIX
    quantize_attention: bool = True
    quantize_base_mlp: bool = True
    quantize_router: bool = False
    inplace: bool = True

    def __post_init__(self) -> None:
        _validate_quant_bits(self.bits)
        _validate_positive_int(self.group_size, "group_size")
        _validate_positive_int(self.rank, "rank")
        _validate_positive_int(self.num_alternating_steps, "num_alternating_steps")
        _validate_nonneg_finite(self.prose_weight, "prose_weight")
        _validate_bool(self.use_lora_compensation, "use_lora_compensation")
        _validate_nonneg_finite(self.max_domain_drift, "max_domain_drift")
        _validate_positive_finite(self.anse_safe_e0_ratio, "anse_safe_e0_ratio")
        _validate_nonneg_finite(self.maximum_pain, "maximum_pain")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_bool(self.quantize_attention, "quantize_attention")
        _validate_bool(self.quantize_base_mlp, "quantize_base_mlp")
        _validate_bool(self.quantize_router, "quantize_router")
        _validate_bool(self.inplace, "inplace")


# ---------------------------------------------------------------------------
# 2. Symmetric Groupwise Block Quantizer (4-bit / 8-bit, CPU-Deterministic)
# ---------------------------------------------------------------------------


@torch.no_grad()
def symmetric_group_quantize_dequantize(
    weight: torch.Tensor,
    *,
    bits: int = DEFAULT_QUANT_BITS,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Deterministically quantize and dequantize a 2D weight tensor via symmetric groupwise scaling.

    Pads non-divisible column dimensions with trailing zeros so group ``absmax`` scales remain
    invariant on valid columns, quantizes to ``[-q_max, q_max]`` (``7`` for 4-bit, ``127`` for
    8-bit), dequantizes back, and slices to the exact input shape.
    """
    w64, orig_dtype, orig_device = _validate_finite_2d(weight, "weight")
    bits_i = _validate_quant_bits(bits)
    group_i = _validate_positive_int(group_size, "group_size")

    rows, cols = int(w64.shape[0]), int(w64.shape[1])
    q_max = (1 << (bits_i - 1)) - 1
    num_groups = (cols + group_i - 1) // group_i
    padded_cols = num_groups * group_i
    pad_cols = padded_cols - cols

    w_padded = F.pad(w64, (0, pad_cols), mode="constant", value=0.0) if pad_cols > 0 else w64
    grouped = w_padded.reshape(rows, num_groups, group_i)

    absmax = grouped.abs().amax(dim=-1, keepdim=True)
    scales = absmax / float(q_max)
    nonzero_mask = scales > 0.0
    safe_scales = torch.where(nonzero_mask, scales, torch.ones_like(scales))

    q_int = torch.round(grouped / safe_scales).clamp(-float(q_max), float(q_max))
    deq_grouped = torch.where(nonzero_mask, q_int * scales, torch.zeros_like(grouped))
    w_deq_64 = deq_grouped.reshape(rows, padded_cols)[:, :cols].contiguous()
    w_deq = w_deq_64.to(device=orig_device, dtype=orig_dtype)

    returned_64 = w_deq.detach().to(device="cpu", dtype=torch.float64)
    diff = returned_64 - w64
    quant_error_fro = float(torch.linalg.matrix_norm(diff, ord="fro").item())
    weight_fro = float(torch.linalg.matrix_norm(w64, ord="fro").item())
    max_abs_error = float(diff.abs().max().item())

    diagnostics: dict[str, Any] = {
        "bits": bits_i,
        "group_size": group_i,
        "q_max": q_max,
        "shape": [rows, cols],
        "padded_cols": padded_cols,
        "pad_cols": pad_cols,
        "num_groups_per_row": num_groups,
        "total_groups": rows * num_groups,
        "zero_groups": int((~nonzero_mask).sum().item()),
        "quant_error_fro": quant_error_fro,
        "weight_fro": weight_fro,
        "rel_quant_error_fro": quant_error_fro / max(weight_fro, 1e-12),
        "max_abs_error": max_abs_error,
        "mean_scale": float(scales.mean().item()),
        "max_scale": float(scales.max().item()),
        "compressed_bits_per_weight": float((bits_i * padded_cols + 16 * num_groups) / float(cols)),
    }
    return w_deq, diagnostics


# ---------------------------------------------------------------------------
# 3. Alternating LoftQ + Prose-Weighted Thin QR-SVD Error Absorption
# ---------------------------------------------------------------------------


def _build_prose_metric_matrices(
    m_prose: torch.Tensor | None,
    d_in: int,
    prose_weight: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    """Build SPD prose right-weighting matrix ``C^{1/2} = (I + w * M_prose_norm)^{1/2}`` and ``M_prose^{1/2}``."""
    if m_prose is None:
        return None, None, {
            "prose_weighted": False,
            "prose_weight": prose_weight,
            "prose_trace_mean": 0.0,
            "max_metric_eig": 1.0,
        }

    mp_sym = _validate_square_psd_moment(m_prose, d_in, "m_prose")
    evals, evecs = torch.linalg.eigh(mp_sym)
    evals_pos = evals.clamp_min(0.0)
    trace_mean = float(evals_pos.mean().item())
    m_half = (evecs * evals_pos.sqrt().unsqueeze(0)) @ evecs.T

    if prose_weight <= 0.0 or trace_mean <= 1e-12:
        return None, m_half, {
            "prose_weighted": False,
            "prose_weight": prose_weight,
            "prose_trace_mean": trace_mean,
            "max_metric_eig": 1.0,
        }

    norm_evals = evals_pos / trace_mean
    c_evals = 1.0 + prose_weight * norm_evals
    c_half = ((evecs * c_evals.sqrt().unsqueeze(0)) @ evecs.T).contiguous()
    return c_half, m_half, {
        "prose_weighted": True,
        "prose_weight": prose_weight,
        "prose_trace_mean": trace_mean,
        "max_metric_eig": float(c_evals.max().item()),
    }


@torch.no_grad()
def _thin_qr_svd_rank_r_factor(
    residual: torch.Tensor,
    *,
    rank: int,
    c_half: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute optimal rank-``r`` residual LoRA factors minimizing ``||(residual - B A) C^{1/2}||_F``.

    Because ``B A = U_r (U_r^T residual)`` where ``U_r`` are the top-``r`` left singular vectors
    of ``residual @ C^{1/2}``, the decomposition minimizes the prose-weighted objective while
    simultaneously preserving the unweighted Euclidean monotone property ``||residual - B A||_F <= ||residual||_F``.
    Factors are canonicalized into balanced ``(B, A)`` via ``decompose_lora_qr_svd``.
    """
    d_out, d_in = int(residual.shape[0]), int(residual.shape[1])
    r_eff = min(rank, d_out, d_in)
    r_weighted = (residual @ c_half) if c_half is not None else residual

    if min(d_out, d_in) <= _LARGE_SVD_DIM_THRESHOLD or r_eff + _SUBSPACE_OVERSAMPLE >= min(d_out, d_in):
        u_full, _, _ = torch.linalg.svd(r_weighted, full_matrices=False)
        u_r = u_full[:, :r_eff].contiguous()
    else:
        gen = torch.Generator(device="cpu").manual_seed(0x5EED_07)
        sketch_dim = min(min(d_out, d_in), r_eff + _SUBSPACE_OVERSAMPLE)
        omega = torch.randn((d_in, sketch_dim), generator=gen, dtype=torch.float64)
        q_basis, _ = torch.linalg.qr(r_weighted @ omega, mode="reduced")
        for _ in range(_SUBSPACE_POWER_ITERS):
            z_basis, _ = torch.linalg.qr(r_weighted.T @ q_basis, mode="reduced")
            q_basis, _ = torch.linalg.qr(r_weighted @ z_basis, mode="reduced")
        core = q_basis.T @ r_weighted
        u_core, _, _ = torch.linalg.svd(core, full_matrices=False)
        u_r = (q_basis @ u_core[:, :r_eff]).contiguous()

    a_raw = (u_r.T @ residual).contiguous()
    u_c, s_c, v_c = decompose_lora_qr_svd(u_r, a_raw)
    s_sqrt = s_c.clamp_min(0.0).sqrt()
    b_eff = u_c * s_sqrt.unsqueeze(0)
    a_eff = s_sqrt.unsqueeze(1) * v_c.T

    if r_eff < rank:
        b_out = torch.zeros((d_out, rank), dtype=torch.float64)
        a_out = torch.zeros((rank, d_in), dtype=torch.float64)
        b_out[:, :r_eff] = b_eff
        a_out[:r_eff, :] = a_eff
        return b_out, a_out, s_c
    return b_eff.contiguous(), a_eff.contiguous(), s_c


@torch.no_grad()
def compensate_quantization_error_via_lora_svd(
    w_target: torch.Tensor,
    *,
    rank: int,
    bits: int = DEFAULT_QUANT_BITS,
    group_size: int = DEFAULT_GROUP_SIZE,
    num_alternating_steps: int = DEFAULT_ALTERNATING_STEPS,
    m_prose: torch.Tensor | None = None,
    prose_weight: float = DEFAULT_PROSE_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Decompose ``w_target ~= Q + B @ A`` via alternating LoftQ and prose-weighted thin QR-SVD.

    Returns ``(w_quant_deq, lora_b, lora_a, diagnostics)`` where ``w_quant_deq`` is the
    group-quantized/dequantized base matrix in ``w_target.dtype`` and ``(lora_b, lora_a)``
    are high-precision rank-``r`` LoRA compensation factors (``float64`` if ``w_target`` is
    ``float64``, else ``float32``).
    """
    w64, orig_dtype, orig_device = _validate_finite_2d(w_target, "w_target")
    rank_i = _validate_positive_int(rank, "rank")
    bits_i = _validate_quant_bits(bits)
    group_i = _validate_positive_int(group_size, "group_size")
    steps_i = _validate_positive_int(num_alternating_steps, "num_alternating_steps")
    pw_f = _validate_nonneg_finite(prose_weight, "prose_weight")

    d_out, d_in = int(w64.shape[0]), int(w64.shape[1])
    c_half, m_half, prose_meta = _build_prose_metric_matrices(m_prose, d_in, pw_f)

    q0, q0_diag = symmetric_group_quantize_dequantize(w64, bits=bits_i, group_size=group_i)
    naive_fro = float(q0_diag["quant_error_fro"])
    naive_prose_fro = (
        float(torch.linalg.matrix_norm((w64 - q0) @ m_half, ord="fro").item())
        if m_half is not None
        else naive_fro
    )
    naive_weighted_err = (
        float(torch.linalg.matrix_norm((w64 - q0) @ c_half, ord="fro").item())
        if c_half is not None
        else naive_fro
    )

    best_q = q0
    best_b = torch.zeros((d_out, rank_i), dtype=torch.float64)
    best_a = torch.zeros((rank_i, d_in), dtype=torch.float64)
    best_s = torch.zeros(min(rank_i, d_out, d_in), dtype=torch.float64)
    best_fro = naive_fro
    best_weighted = naive_weighted_err
    best_prose_fro = naive_prose_fro

    b_work = best_b.clone()
    a_work = best_a.clone()

    step_records: list[dict[str, Any]] = []
    step_errors_fro: list[float] = []
    step_weighted_errors: list[float] = []

    for step_idx in range(1, steps_i + 1):
        base_target = w64 - (b_work @ a_work)
        q_cand, q_cand_diag = symmetric_group_quantize_dequantize(
            base_target, bits=bits_i, group_size=group_i
        )
        residual = w64 - q_cand
        b_cand, a_cand, s_cand = _thin_qr_svd_rank_r_factor(
            residual, rank=rank_i, c_half=c_half
        )
        recon_err = w64 - (q_cand + (b_cand @ a_cand))
        fro_cand = float(torch.linalg.matrix_norm(recon_err, ord="fro").item())
        weighted_cand = (
            float(torch.linalg.matrix_norm(recon_err @ c_half, ord="fro").item())
            if c_half is not None
            else fro_cand
        )
        prose_cand = (
            float(torch.linalg.matrix_norm(recon_err @ m_half, ord="fro").item())
            if m_half is not None
            else fro_cand
        )

        accepted_step = (
            weighted_cand <= best_weighted + 1e-12
            and fro_cand <= best_fro + 1e-12
        )
        if accepted_step:
            best_q = q_cand
            best_b = b_cand
            best_a = a_cand
            best_s = s_cand
            best_fro = fro_cand
            best_weighted = weighted_cand
            best_prose_fro = prose_cand
            b_work = b_cand
            a_work = a_cand

        step_errors_fro.append(best_fro)
        step_weighted_errors.append(best_weighted)
        step_records.append(
            {
                "step": step_idx,
                "accepted": accepted_step,
                "base_quant_error_fro": float(q_cand_diag["quant_error_fro"]),
                "candidate_error_fro": fro_cand,
                "quant_error_fro": best_fro,
                "weighted_error": best_weighted,
                "prose_error_fro": best_prose_fro,
            }
        )

    lora_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    w_quant_out = best_q.to(device=orig_device, dtype=orig_dtype)
    lora_b_out = best_b.to(device=orig_device, dtype=lora_dtype)
    lora_a_out = best_a.to(device=orig_device, dtype=lora_dtype)

    recon_final = (
        w_quant_out.detach().to(device="cpu", dtype=torch.float64)
        + (lora_b_out.detach().to(device="cpu", dtype=torch.float64) @ lora_a_out.detach().to(device="cpu", dtype=torch.float64))
    )
    final_diff = w64 - recon_final
    final_fro = float(torch.linalg.matrix_norm(final_diff, ord="fro").item())
    final_max_abs = float(final_diff.abs().max().item())
    final_prose_fro = (
        float(torch.linalg.matrix_norm(final_diff @ m_half, ord="fro").item())
        if m_half is not None
        else final_fro
    )
    weight_fro = float(q0_diag["weight_fro"])

    diagnostics: dict[str, Any] = {
        "bits": bits_i,
        "group_size": group_i,
        "rank": rank_i,
        "effective_rank": min(rank_i, d_out, d_in),
        "num_alternating_steps": steps_i,
        "shape": [d_out, d_in],
        "padded_cols": q0_diag["padded_cols"],
        "pad_cols": q0_diag["pad_cols"],
        "num_groups_per_row": q0_diag["num_groups_per_row"],
        "total_groups": q0_diag["total_groups"],
        **prose_meta,
        "weight_fro": weight_fro,
        "naive_quant_error_fro": naive_fro,
        "quant_error_fro": final_fro,
        "rel_quant_error_fro": final_fro / max(weight_fro, 1e-12),
        "max_abs_error": final_max_abs,
        "error_reduction_ratio": final_fro / max(naive_fro, 1e-12) if naive_fro > 0.0 else 0.0,
        "fro_absorbed_frac": (1.0 - final_fro / naive_fro) if naive_fro > 0.0 else 1.0,
        "naive_prose_error_fro": naive_prose_fro,
        "prose_error_fro": final_prose_fro,
        "prose_error_reduction_ratio": (
            final_prose_fro / max(naive_prose_fro, 1e-12) if naive_prose_fro > 0.0 else 0.0
        ),
        "step_errors_fro": step_errors_fro,
        "step_weighted_errors": step_weighted_errors,
        "steps": step_records,
        "lora_singular_values": [float(x) for x in best_s.tolist()],
    }
    return w_quant_out, lora_b_out, lora_a_out, diagnostics


# ---------------------------------------------------------------------------
# 4. Quantized + Compensated Linear Layer & ANSE Guarded MoE Layer Verification
# ---------------------------------------------------------------------------


class QuantCompensatedLinear(nn.Module):
    """Drop-in ``nn.Linear`` replacement storing quantized base weight ``Q`` + high-precision ``(B, A)``."""

    def __init__(
        self,
        quant_weight: torch.Tensor,
        lora_b: torch.Tensor,
        lora_a: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        bits: int = DEFAULT_QUANT_BITS,
        group_size: int = DEFAULT_GROUP_SIZE,
    ) -> None:
        super().__init__()
        q64, q_dtype, q_dev = _validate_finite_2d(quant_weight, "quant_weight")
        b64, b_dtype, _ = _validate_finite_2d(lora_b, "lora_b")
        a64, a_dtype, _ = _validate_finite_2d(lora_a, "lora_a")

        d_out, d_in = int(q64.shape[0]), int(q64.shape[1])
        if b64.shape[0] != d_out or a64.shape[1] != d_in or b64.shape[1] != a64.shape[0]:
            raise ValueError(
                f"Incompatible LoRA shapes: Q={tuple(q64.shape)}, B={tuple(b64.shape)}, A={tuple(a64.shape)}"
            )

        self.in_features = d_in
        self.out_features = d_out
        self.bits = _validate_quant_bits(bits)
        self.group_size = _validate_positive_int(group_size, "group_size")
        self.rank = int(b64.shape[1])

        self.weight = nn.Parameter(quant_weight.detach().clone().to(device=q_dev, dtype=q_dtype), requires_grad=False)
        self.lora_b = nn.Parameter(lora_b.detach().clone().to(device=q_dev, dtype=b_dtype), requires_grad=False)
        self.lora_a = nn.Parameter(lora_a.detach().clone().to(device=q_dev, dtype=a_dtype), requires_grad=False)
        if bias is not None:
            if not isinstance(bias, torch.Tensor) or bias.ndim != 1 or int(bias.shape[0]) != d_out:
                raise ValueError(f"bias must be a 1D tensor of length {d_out}")
            if not bool(torch.isfinite(bias.detach()).all().item()):
                raise ValueError("bias must contain only finite values")
            self.bias: nn.Parameter | None = nn.Parameter(
                bias.detach().clone().to(device=q_dev, dtype=q_dtype), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``x @ Q^T + bias + (x @ A^T) @ B^T`` preserving input dtype."""
        orig_dt = x.dtype
        base = F.linear(x.to(dtype=self.weight.dtype), self.weight, self.bias)
        lora_h = F.linear(x.to(dtype=self.lora_a.dtype), self.lora_a)
        lora_out = F.linear(lora_h.to(dtype=self.lora_b.dtype), self.lora_b)
        return (base.to(dtype=lora_out.dtype) + lora_out).to(dtype=orig_dt)

    def effective_weight(self) -> torch.Tensor:
        """Return dense reconstructed weight ``Q + B @ A`` in ``lora_a.dtype``."""
        return (
            self.weight.detach().to(dtype=self.lora_a.dtype)
            + (self.lora_b.detach() @ self.lora_a.detach().to(dtype=self.lora_b.dtype)).to(dtype=self.lora_a.dtype)
        )


def _validate_fallback_dtype(fallback_dtype: torch.dtype | None) -> torch.dtype | None:
    """Validate optional atomic rollback dtype (``float32``, ``bfloat16``, or ``float16``)."""
    if fallback_dtype is None:
        return None
    if not isinstance(fallback_dtype, torch.dtype):
        raise TypeError(f"fallback_dtype must be a torch.dtype or None, got {type(fallback_dtype).__name__}")
    if fallback_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"fallback_dtype must be float32, bfloat16, or float16, got {fallback_dtype}")
    return fallback_dtype


def _collect_target_linears(
    root: nn.Module,
    cfg: QuantizationWSPConfig,
) -> list[tuple[nn.Module | None, str, str, nn.Linear]]:
    """Collect target ``nn.Linear`` layers inside ``root`` respecting attention/base_mlp/router flags."""
    if isinstance(root, nn.Linear):
        _validate_finite_2d(root.weight, "layer.weight")
        if root.bias is not None and not bool(torch.isfinite(root.bias.detach()).all().item()):
            raise ValueError("layer.bias must contain only finite values")
        return [(None, "", "root", root)]

    targets: list[tuple[nn.Module | None, str, str, nn.Linear]] = []

    def _walk(mod: nn.Module, prefix: str) -> None:
        for child_name, child in mod.named_children():
            full = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                if full == "head" or full.endswith(".head"):
                    continue
                is_router = child_name == "router" or ".router." in full
                is_attn = ".attn." in f".{full}."
                is_mlp_proj = child_name in ("w1", "w2", "w3")
                if is_router and not cfg.quantize_router:
                    continue
                if is_attn and not cfg.quantize_attention:
                    continue
                if is_mlp_proj and not cfg.quantize_base_mlp and not is_attn:
                    continue
                _validate_finite_2d(child.weight, f"{full}.weight")
                if child.bias is not None and not bool(torch.isfinite(child.bias.detach()).all().item()):
                    raise ValueError(f"{full}.bias must contain only finite values")
                targets.append((mod, child_name, full, child))
            else:
                _walk(child, full)

    _walk(root, "")
    if not targets:
        raise ValueError("No eligible nn.Linear modules found to quantize under current configuration")
    return targets


def _compute_second_moment(x: torch.Tensor, name: str) -> torch.Tensor:
    """Compute CPU float64 uncentered second moment ``E[x^T x]`` from a 2D or 3D activation tensor."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if x.ndim not in (2, 3) or any(dim == 0 for dim in x.shape):
        raise ValueError(f"{name} must be a non-empty 2D or 3D tensor, got shape {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {x.dtype}")
    flat = x.detach().to(device="cpu", dtype=torch.float64).reshape(-1, int(x.shape[-1]))
    if not bool(torch.isfinite(flat).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    m2 = (flat.T @ flat) / float(flat.shape[0])
    return (0.5 * (m2 + m2.T)).contiguous()


def _resolve_prose_moments_for_targets(
    root: nn.Module,
    targets: list[tuple[nn.Module | None, str, str, nn.Linear]],
    *,
    m_prose: torch.Tensor | Mapping[str, torch.Tensor] | None,
    prose_activations: torch.Tensor | None,
) -> dict[str, torch.Tensor | None]:
    """Resolve per-linear ``m_prose`` matrices from explicit tensors, dicts, or prose activations."""
    explicit_tensor: torch.Tensor | None = None
    explicit_map: dict[str, torch.Tensor] = {}
    if isinstance(m_prose, torch.Tensor):
        m64, _, _ = _validate_finite_2d(m_prose, "m_prose")
        if m64.shape[0] != m64.shape[1]:
            raise ValueError(f"m_prose must be square, got shape {tuple(m64.shape)}")
        explicit_tensor = _validate_square_psd_moment(m64, int(m64.shape[0]), "m_prose")
    elif isinstance(m_prose, Mapping):
        for k, v in m_prose.items():
            v64, _, _ = _validate_finite_2d(v, f"m_prose[{k!r}]")
            if v64.shape[0] != v64.shape[1]:
                raise ValueError(f"m_prose[{k!r}] must be square, got shape {tuple(v64.shape)}")
            explicit_map[str(k)] = _validate_square_psd_moment(v64, int(v64.shape[0]), f"m_prose[{k!r}]")
    elif m_prose is not None:
        raise TypeError(f"m_prose must be a torch.Tensor, Mapping, or None, got {type(m_prose).__name__}")

    act_moments: dict[str, torch.Tensor] = {}
    if prose_activations is not None:
        m_in = _compute_second_moment(prose_activations, "prose_activations")
        act_moments["x"] = m_in
        flat_px = prose_activations.detach().to(device="cpu").reshape(-1, int(prose_activations.shape[-1]))
        for mod_name, mod in root.named_modules():
            if (
                hasattr(mod, "w1")
                and hasattr(mod, "w2")
                and isinstance(mod.w1, nn.Linear)
                and isinstance(mod.w2, nn.Linear)
            ):
                w1_mod: nn.Linear = mod.w1
                w2_mod: nn.Linear = mod.w2
                if w1_mod.in_features == int(flat_px.shape[-1]):
                    x_in = flat_px.to(device=w1_mod.weight.device, dtype=w1_mod.weight.dtype)
                    swi = F.silu(w1_mod(x_in).to(torch.float32)) * w2_mod(x_in).to(torch.float32)
                    m_swi = _compute_second_moment(swi, "swiglu_prose_activations")
                    w3_key = f"{mod_name}.w3" if mod_name else "w3"
                    act_moments[w3_key] = m_swi
                    act_moments.setdefault("act", m_swi)

    resolved: dict[str, torch.Tensor | None] = {}
    for _, attr_name, full_name, lin in targets:
        d_in = int(lin.in_features)
        chosen: torch.Tensor | None = None
        for cand_key in (full_name, attr_name, "act" if attr_name == "w3" else "x"):
            if cand_key in explicit_map and explicit_map[cand_key].shape[0] == d_in:
                chosen = explicit_map[cand_key]
                break
        if chosen is None and explicit_tensor is not None and explicit_tensor.shape[0] == d_in:
            chosen = explicit_tensor
        if chosen is None:
            for cand_key in (full_name, "act" if attr_name == "w3" else "x"):
                if cand_key in act_moments and act_moments[cand_key].shape[0] == d_in:
                    chosen = act_moments[cand_key]
                    break
        resolved[full_name] = chosen
    return resolved


def _parse_eval_domain_nll(raw_eval: Any) -> dict[str, float]:
    """Normalize evaluator output into a validated ``domain -> positive finite NLL`` dictionary."""
    payload: Any = raw_eval
    if isinstance(raw_eval, tuple) and len(raw_eval) == 2 and isinstance(raw_eval[1], Mapping):
        payload = raw_eval[1]
    if not isinstance(payload, Mapping) or not payload:
        raise TypeError("eval_fn must return a non-empty mapping (with 'domain_nll' or numeric domain entries)")

    domain_map = payload
    for container_key in ("domain_nll", "per_domain_nll"):
        if container_key in payload:
            sub = payload[container_key]
            if not isinstance(sub, Mapping) or not sub:
                raise ValueError(f"eval[{container_key!r}] must be a non-empty mapping")
            domain_map = sub
            break

    out: dict[str, float] = {}
    for dom, val in domain_map.items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            out[str(dom)] = _validate_positive_finite(val, f"domain_nll[{dom!r}]")
        else:
            raise ValueError(f"domain_nll[{dom!r}] must be a positive finite number, got {val!r}")
    return out


def _evaluate_anse_domain_guard(
    baseline_nll: Mapping[str, float],
    candidate_nll: Mapping[str, float],
    *,
    tuned_nll: Mapping[str, float] | None,
    cfg: QuantizationWSPConfig,
) -> dict[str, Any]:
    """Compute per-domain NLL drift, specialist retention, guard verdict, and ANSE energy."""
    if set(baseline_nll.keys()) != set(candidate_nll.keys()):
        raise ValueError("baseline_domain_nll and candidate domain_nll must share identical keys")

    per_domain_drift: dict[str, float] = {}
    per_domain_drift_pct: dict[str, float] = {}
    base_vals: list[float] = []
    cand_vals: list[float] = []
    worst_domain = ""
    worst_drift = -math.inf

    for dom in sorted(baseline_nll.keys()):
        b_v = _validate_positive_finite(baseline_nll[dom], f"baseline_nll[{dom!r}]")
        c_v = _validate_positive_finite(candidate_nll[dom], f"candidate_nll[{dom!r}]")
        drift = (c_v / b_v) - 1.0
        per_domain_drift[dom] = drift
        per_domain_drift_pct[dom] = 100.0 * drift
        base_vals.append(b_v)
        cand_vals.append(c_v)
        if drift > worst_drift:
            worst_drift = drift
            worst_domain = dom

    general_keys = [
        d for d in sorted(baseline_nll.keys()) if d.startswith(cfg.general_prefix) or d in ("general", "general_nll")
    ]
    spec_keys = [d for d in sorted(baseline_nll.keys()) if d not in general_keys]

    general_drift = max(per_domain_drift[k] for k in general_keys) if general_keys else worst_drift
    general_ratio = 1.0 + general_drift
    guard_ok = (worst_drift <= cfg.max_domain_drift + 1e-12) and (general_drift <= cfg.max_domain_drift + 1e-12)

    nll_mean = float(sum(cand_vals) / float(len(cand_vals)))
    base_nll_mean = float(sum(base_vals) / float(len(base_vals)))
    mean_diff = nll_mean - base_nll_mean

    spec_gain_nats: float | None = None
    spec_gain_retained_frac: float | None = None
    if spec_keys:
        base_spec_mean = float(sum(baseline_nll[k] for k in spec_keys) / float(len(spec_keys)))
        cand_spec_mean = float(sum(candidate_nll[k] for k in spec_keys) / float(len(spec_keys)))
        spec_gain_nats = base_spec_mean - cand_spec_mean
        if tuned_nll is not None and all(k in tuned_nll for k in spec_keys):
            tuned_spec_mean = float(sum(tuned_nll[k] for k in spec_keys) / float(len(spec_keys)))
            raw_gain = base_spec_mean - tuned_spec_mean
            spec_gain_retained_frac = (spec_gain_nats / raw_gain) if abs(raw_gain) > 1e-8 else 1.0

    penalty = (
        cfg.maximum_pain
        if not guard_ok
        else (cfg.soft_margin_penalty if general_ratio > cfg.anse_safe_e0_ratio else 0.0)
    )
    anse_energy = mean_diff + penalty
    return {
        "nll_mean": nll_mean,
        "base_nll_mean": base_nll_mean,
        "mean_diff_vs_base": mean_diff,
        "general_domains": general_keys,
        "specialist_domains": spec_keys,
        "general_drift": general_drift,
        "general_drift_pct": 100.0 * general_drift,
        "worst_domain": worst_domain,
        "worst_domain_drift": worst_drift,
        "worst_domain_drift_pct": 100.0 * worst_drift,
        "specialist_gain_nats": spec_gain_nats,
        "specialist_gain_retained_frac": spec_gain_retained_frac,
        "guard_ok": guard_ok,
        "anse_energy": float(anse_energy),
        "per_domain_drift_pct": per_domain_drift_pct,
        "per_domain_nll": {k: float(candidate_nll[k]) for k in sorted(candidate_nll.keys())},
    }


def _restore_snapshot_and_fallback_dtype(
    work_layer: nn.Module,
    snapshots: list[tuple[nn.Module | None, str, nn.Linear, torch.Tensor, torch.Tensor | None]],
    fallback_dtype: torch.dtype | None,
) -> nn.Module:
    """Atomically restore original ``nn.Linear`` modules and optionally cast to ``fallback_dtype``."""
    active_layer = work_layer
    for parent_mod, attr_name, orig_lin, w_snap, b_snap in snapshots:
        w_restored = w_snap.to(device=orig_lin.weight.device, dtype=fallback_dtype or orig_lin.weight.dtype)
        orig_lin.weight = nn.Parameter(w_restored, requires_grad=orig_lin.weight.requires_grad)
        if b_snap is not None and orig_lin.bias is not None:
            b_restored = b_snap.to(device=orig_lin.bias.device, dtype=fallback_dtype or orig_lin.bias.dtype)
            orig_lin.bias = nn.Parameter(b_restored, requires_grad=orig_lin.bias.requires_grad)
        if parent_mod is None:
            active_layer = orig_lin
        else:
            setattr(parent_mod, attr_name, orig_lin)
    return active_layer


@torch.no_grad()
def quantize_and_verify_moe_wsp_layer(
    layer: nn.Module,
    *,
    config: QuantizationWSPConfig | None = None,
    bits: int = DEFAULT_QUANT_BITS,
    group_size: int = DEFAULT_GROUP_SIZE,
    rank: int = DEFAULT_LORA_RANK,
    num_alternating_steps: int = DEFAULT_ALTERNATING_STEPS,
    m_prose: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    prose_activations: torch.Tensor | None = None,
    prose_weight: float = DEFAULT_PROSE_WEIGHT,
    use_lora_compensation: bool = True,
    eval_fn: Callable[[nn.Module], Mapping[str, Any] | tuple[float, Mapping[str, float]]] | None = None,
    baseline_domain_nll: Mapping[str, float] | None = None,
    tuned_domain_nll: Mapping[str, float] | None = None,
    max_domain_drift: float = DEFAULT_MAX_DOMAIN_DRIFT,
    anse_safe_e0_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO,
    maximum_pain: float = DEFAULT_ANSE_MAXIMUM_PAIN,
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY,
    general_prefix: str = GENERAL_EXPERT_PREFIX,
    fallback_dtype: torch.dtype | None = None,
    quantize_attention: bool = True,
    quantize_base_mlp: bool = True,
    quantize_router: bool = False,
    inplace: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Quantize edited MoE/dense linear weights under an ANSE ``<= +1.0%`` drift guard with atomic fallback.

    Returns ``(active_layer, receipt)``. If evaluation fails, yields non-finite NLLs, or violates
    the ``max_domain_drift`` guard (triggering ANSE Maximum Pain ``>= 1e6``), the layer is atomically
    rolled back to the verified pre-quantization FP32/BF16 weights (optionally cast to
    ``fallback_dtype``) with deterministic SHA-256 provenance verification.
    """
    if not isinstance(layer, nn.Module):
        raise TypeError(f"layer must be an nn.Module, got {type(layer).__name__}")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError("eval_fn must be callable or None")
    fb_dtype = _validate_fallback_dtype(fallback_dtype)

    if config is not None:
        if not isinstance(config, QuantizationWSPConfig):
            raise TypeError(f"config must be a QuantizationWSPConfig or None, got {type(config).__name__}")
        cfg = config
    else:
        cfg = QuantizationWSPConfig(
            bits=bits,
            group_size=group_size,
            rank=rank,
            num_alternating_steps=num_alternating_steps,
            prose_weight=prose_weight,
            use_lora_compensation=use_lora_compensation,
            max_domain_drift=max_domain_drift,
            anse_safe_e0_ratio=anse_safe_e0_ratio,
            maximum_pain=maximum_pain,
            soft_margin_penalty=soft_margin_penalty,
            general_prefix=general_prefix,
            quantize_attention=quantize_attention,
            quantize_base_mlp=quantize_base_mlp,
            quantize_router=quantize_router,
            inplace=inplace,
        )

    base_nll_clean: dict[str, float] | None = None
    if baseline_domain_nll is not None:
        base_nll_clean = _parse_eval_domain_nll(baseline_domain_nll)
    tuned_nll_clean: dict[str, float] | None = None
    if tuned_domain_nll is not None:
        tuned_nll_clean = _parse_eval_domain_nll(tuned_domain_nll)

    work_layer = layer if cfg.inplace else copy.deepcopy(layer)
    targets = _collect_target_linears(work_layer, cfg)
    prose_moments = _resolve_prose_moments_for_targets(
        work_layer,
        targets,
        m_prose=m_prose,
        prose_activations=prose_activations,
    )

    parent_sha256 = compute_module_state_sha256(work_layer)
    snapshots: list[tuple[nn.Module | None, str, nn.Linear, torch.Tensor, torch.Tensor | None]] = [
        (
            parent_mod,
            attr_name,
            orig_lin,
            orig_lin.weight.detach().to(device="cpu").clone(),
            orig_lin.bias.detach().to(device="cpu").clone() if orig_lin.bias is not None else None,
        )
        for parent_mod, attr_name, _, orig_lin in targets
    ]

    pre_domain_nll: dict[str, float] | None = None
    if eval_fn is not None:
        pre_domain_nll = _parse_eval_domain_nll(eval_fn(work_layer))
        if base_nll_clean is not None and set(base_nll_clean.keys()) != set(pre_domain_nll.keys()):
            raise ValueError("baseline_domain_nll and candidate domain_nll must share identical keys")
        if tuned_nll_clean is not None and set(tuned_nll_clean.keys()) != set(pre_domain_nll.keys()):
            raise ValueError("tuned_domain_nll and candidate domain_nll must share identical keys")
    eff_base_nll = base_nll_clean if base_nll_clean is not None else pre_domain_nll
    eff_tuned_nll = tuned_nll_clean if tuned_nll_clean is not None else pre_domain_nll

    per_linear_diags: dict[str, dict[str, Any]] = {}
    candidate_layer: nn.Module = work_layer
    orig_fp32_bytes = 0
    quant_storage_bytes = 0

    for parent_mod, attr_name, full_name, orig_lin in targets:
        target_mp = prose_moments[full_name]
        if cfg.use_lora_compensation:
            q_deq, lora_b, lora_a, lin_diag = compensate_quantization_error_via_lora_svd(
                orig_lin.weight,
                rank=cfg.rank,
                bits=cfg.bits,
                group_size=cfg.group_size,
                num_alternating_steps=cfg.num_alternating_steps,
                m_prose=target_mp,
                prose_weight=cfg.prose_weight,
            )
        else:
            q_deq, q_only_diag = symmetric_group_quantize_dequantize(
                orig_lin.weight,
                bits=cfg.bits,
                group_size=cfg.group_size,
            )
            lora_dtype = torch.float64 if orig_lin.weight.dtype == torch.float64 else torch.float32
            lora_b = torch.zeros((orig_lin.out_features, cfg.rank), device=orig_lin.weight.device, dtype=lora_dtype)
            lora_a = torch.zeros((cfg.rank, orig_lin.in_features), device=orig_lin.weight.device, dtype=lora_dtype)
            lin_diag = {
                **q_only_diag,
                "rank": cfg.rank,
                "num_alternating_steps": 0,
                "prose_weighted": False,
                "naive_quant_error_fro": q_only_diag["quant_error_fro"],
                "error_reduction_ratio": 1.0,
                "fro_absorbed_frac": 0.0,
            }

        comp_lin = QuantCompensatedLinear(
            q_deq,
            lora_b,
            lora_a,
            bias=orig_lin.bias,
            bits=cfg.bits,
            group_size=cfg.group_size,
        )
        per_linear_diags[full_name] = lin_diag

        numel = int(orig_lin.weight.numel())
        n_groups_total = int(lin_diag["total_groups"])
        orig_fp32_bytes += numel * 4
        quant_storage_bytes += (numel * cfg.bits + 7) // 8 + n_groups_total * 2
        if cfg.use_lora_compensation:
            quant_storage_bytes += (int(lora_b.numel()) + int(lora_a.numel())) * 2

        if parent_mod is None:
            candidate_layer = comp_lin
        else:
            setattr(parent_mod, attr_name, comp_lin)

    candidate_sha256 = compute_module_state_sha256(candidate_layer)
    total_quant_fro = math.sqrt(sum(float(d["quant_error_fro"]) ** 2 for d in per_linear_diags.values()))
    total_naive_fro = math.sqrt(sum(float(d["naive_quant_error_fro"]) ** 2 for d in per_linear_diags.values()))
    total_weight_fro = math.sqrt(sum(float(d["weight_fro"]) ** 2 for d in per_linear_diags.values()))

    accepted = True
    rolled_back = False
    failure_reason: str | None = None
    guard_metrics: dict[str, Any] = {
        "nll_mean": None,
        "base_nll_mean": None,
        "mean_diff_vs_base": None,
        "general_drift": None,
        "general_drift_pct": None,
        "worst_domain": None,
        "worst_domain_drift": None,
        "worst_domain_drift_pct": None,
        "specialist_gain_nats": None,
        "specialist_gain_retained_frac": None,
        "guard_ok": True,
        "anse_energy": float(total_quant_fro),
        "per_domain_drift_pct": {},
        "per_domain_nll": {},
    }

    if eval_fn is not None:
        assert eff_base_nll is not None
        try:
            cand_domain_nll = _parse_eval_domain_nll(eval_fn(candidate_layer))
            guard_metrics = _evaluate_anse_domain_guard(
                eff_base_nll,
                cand_domain_nll,
                tuned_nll=eff_tuned_nll,
                cfg=cfg,
            )
            if not guard_metrics["guard_ok"]:
                accepted = False
                failure_reason = (
                    f"ANSE_GUARD_VIOLATION: worst_domain={guard_metrics['worst_domain']} "
                    f"drifted {guard_metrics['worst_domain_drift_pct']:+.3f}% "
                    f"(limit {100.0 * cfg.max_domain_drift:+.3f}%)"
                )
        except Exception as exc:
            accepted = False
            guard_metrics["guard_ok"] = False
            guard_metrics["anse_energy"] = float(cfg.maximum_pain)
            failure_reason = f"EVAL_FAILURE: {type(exc).__name__}: {exc}"

    active_layer = candidate_layer
    if not accepted:
        active_layer = _restore_snapshot_and_fallback_dtype(work_layer, snapshots, fb_dtype)
        rolled_back = True
        active_sha256 = compute_module_state_sha256(active_layer)
        if fb_dtype is None and active_sha256 != parent_sha256:
            raise RuntimeError(
                f"Atomic rollback SHA-256 mismatch: {active_sha256} != {parent_sha256}"
            )
        status = "ROLLED_BACK_ATOMIC_FALLBACK"
    else:
        active_sha256 = candidate_sha256
        status = "ACCEPTED_QUANTIZED_COMPENSATED"

    receipt: dict[str, Any] = {
        "schema": QUANT_RECEIPT_SCHEMA,
        "status": status,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "failure_reason": failure_reason,
        "config": asdict(cfg),
        "eval_verified": eval_fn is not None,
        "fallback_dtype": str(fb_dtype) if fb_dtype is not None else "original_fp",
        "quantized_linear_count": len(per_linear_diags),
        "quantized_linears": sorted(per_linear_diags.keys()),
        "quant_error_fro": total_quant_fro,
        "naive_quant_error_fro": total_naive_fro,
        "weight_fro": total_weight_fro,
        "rel_quant_error_fro": total_quant_fro / max(total_weight_fro, 1e-12),
        "original_fp32_bytes": orig_fp32_bytes,
        "quantized_storage_bytes": quant_storage_bytes,
        "vram_compression_ratio": float(orig_fp32_bytes) / float(max(1, quant_storage_bytes)),
        "parent_sha256": parent_sha256,
        "candidate_sha256": candidate_sha256,
        "active_sha256": active_sha256,
        "per_linear_diagnostics": per_linear_diags,
        **guard_metrics,
        "receipt_sha256": "",
    }
    receipt_core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    receipt["receipt_sha256"] = compute_payload_sha256(receipt_core)
    return active_layer, receipt


__all__ = [
    "DEFAULT_ALTERNATING_STEPS",
    "DEFAULT_ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_ANSE_SOFT_MARGIN_PENALTY",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_GROUP_SIZE",
    "DEFAULT_LORA_RANK",
    "DEFAULT_MAX_DOMAIN_DRIFT",
    "DEFAULT_PROSE_WEIGHT",
    "DEFAULT_QUANT_BITS",
    "GENERAL_EXPERT_PREFIX",
    "QUANT_RECEIPT_SCHEMA",
    "SUPPORTED_QUANT_BITS",
    "QuantCompensatedLinear",
    "QuantizationWSPConfig",
    "canonical_json_bytes",
    "compensate_quantization_error_via_lora_svd",
    "compute_module_state_sha256",
    "compute_payload_sha256",
    "quantize_and_verify_moe_wsp_layer",
    "symmetric_group_quantize_dequantize",
    "verify_quantization_receipt_provenance",
]

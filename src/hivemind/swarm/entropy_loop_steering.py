# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Astra-style sub-0.02% thinking-loop neuron suppression in specialist null-space.

Implements training-free surgical weight editing for open-weight transformers and
Sparse MoE adapters (``CrowdV1WithEngram``, ``CrowdRuntimeMLP``, and
``CrowdLoRAMoEMLP`` / ``CrowdLoRAExpertAdapter``) to eliminate degenerative
thinking loops and low-entropy repetition sinks while modifying strictly fewer
than ``0.02%`` (``max_edited_param_ratio <= 0.0002``) of total model weights:

1. ``localize_thinking_loop_channels(loop_hidden, productive_hidden, down_proj_or_lora_b, unembed_weight, ...)``:
   Ranks late-layer output channels and null-space rank-1 directions by
   2-fold OOF-gated loop-vs-productive causal logit contribution and
   repetition-token affinity.
2. ``apply_nullspace_loop_suppression_edit_inplace(...)``:
   Projects a sparse surgical damping update into the exact null-space of
   productive specialist activations (``M_prod``) under a strict
   ``max_edited_param_ratio <= 0.0002`` parameter budget and SNR safety gate.
3. ``evaluate_repetition_and_guard_metrics(...)``:
   Computes n-gram repetition-sink rates, next-token Shannon entropy floor
   recovery, paired general/specialist NLL guard compliance (``<= +1.0%``),
   and objective ANSE energy.
4. ``apply_astra_loop_suppression_inplace(...)``:
   Training-free multi-layer / Sparse MoE orchestrator enforcing the global
   ``< 0.02%`` parameter budget with zero gradient steps.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    forward_block_pre_mlp,
    second_moment_and_mean,
)

MAX_EDITED_PARAM_RATIO_CEILING: float = 0.0002
DEFAULT_MAX_EDITED_PARAM_RATIO: float = 0.0002
DEFAULT_MIN_CONTRAST_SNR: float = 1.5
DEFAULT_DAMPING_STRENGTH: float = 1.0
DEFAULT_NULLSPACE_EIGEN_RATIO: float = 1e-6
DEFAULT_NGRAM_SIZE: int = 3
DEFAULT_MIN_SINK_REPEATS: int = 3
DEFAULT_LOOP_SINK_NGRAM_THRESHOLD: float = 0.30
DEFAULT_ENTROPY_FLOOR_NATS: float = 0.50
DOMAIN_GUARD_MAX_RATIO: float = 1.01
ANSE_MAXIMUM_PAIN: float = 1e6


def _validate_nonneg_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite non-negative real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0.0, got {value!r}")
    return out


def _validate_positive_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite strictly positive real number."""
    out = _validate_nonneg_finite(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_nonneg_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a non-negative integer (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a strictly positive integer (excluding bool)."""
    out = _validate_nonneg_int(value, name)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return out


def _validate_max_edited_param_ratio(value: Any) -> float:
    """Enforce ``0.0 < max_edited_param_ratio <= MAX_EDITED_PARAM_RATIO_CEILING``."""
    ratio = _validate_positive_finite(value, "max_edited_param_ratio")
    if ratio > MAX_EDITED_PARAM_RATIO_CEILING:
        raise ValueError(
            f"max_edited_param_ratio must be <= {MAX_EDITED_PARAM_RATIO_CEILING}, got {ratio}"
        )
    return ratio


def _validate_hidden_states(tensor: Any, name: str) -> torch.Tensor:
    """Validate 2D ``(N, D)`` or 3D ``(B, T, D)`` finite tensor as CPU ``float64`` 2D."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim not in (2, 3) or any(dim == 0 for dim in tensor.shape):
        raise ValueError(
            f"{name} must be a non-empty 2D or 3D tensor, got shape {tuple(tensor.shape)}"
        )
    flat = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1, int(tensor.shape[-1]))
    if not bool(torch.isfinite(flat).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return flat.contiguous()


def _resolve_weight_tensor(
    obj: Any,
    name: str,
    *,
    weight_attr: str | None = None,
) -> torch.Tensor:
    """Extract a finite 2D floating-point weight tensor from a Tensor or nn.Module."""
    candidate: Any = obj
    if isinstance(obj, nn.Module):
        if weight_attr is not None:
            if not isinstance(weight_attr, str) or not weight_attr.strip():
                raise ValueError("weight_attr must be a non-empty string when provided")
            attr_val = getattr(obj, weight_attr.strip(), None)
            candidate = attr_val.weight if isinstance(attr_val, nn.Linear) else attr_val
        elif isinstance(obj, nn.Linear):
            candidate = obj.weight
        elif hasattr(obj, "w3_B") and isinstance(getattr(obj, "w3_B"), torch.Tensor):
            candidate = getattr(obj, "w3_B")
        elif hasattr(obj, "w3") and isinstance(getattr(obj, "w3"), nn.Linear):
            candidate = getattr(obj, "w3").weight
        elif hasattr(obj, "weight") and isinstance(getattr(obj, "weight"), torch.Tensor):
            candidate = getattr(obj, "weight")
        else:
            raise ValueError(
                f"{type(obj).__name__} must expose a 2D weight tensor (weight, w3.weight, or w3_B)"
            )
    elif weight_attr is not None:
        raise TypeError("weight_attr can only be used when target_module_or_weight is an nn.Module")

    if not isinstance(candidate, torch.Tensor):
        raise TypeError(f"{name} must resolve to a torch.Tensor, got {type(candidate).__name__}")
    if candidate.ndim != 2 or candidate.shape[0] == 0 or candidate.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D tensor, got shape {tuple(candidate.shape)}")
    if not candidate.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {candidate.dtype}")
    if not bool(torch.isfinite(candidate.detach()).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return candidate


def _validate_square_moment(matrix: Any, expected_dim: int, name: str) -> torch.Tensor:
    """Validate a finite square 2D moment matrix of shape ``(expected_dim, expected_dim)``."""
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(matrix).__name__}")
    if matrix.ndim != 2 or matrix.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"{name} must have shape ({expected_dim}, {expected_dim}), got {tuple(matrix.shape)}"
        )
    mat = matrix.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(mat).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return (0.5 * (mat + mat.T)).contiguous()


@dataclass(frozen=True)
class AstraLoopSteeringConfig:
    """Hyperparameters governing Astra-style null-space thinking-loop suppression."""

    max_edited_param_ratio: float = DEFAULT_MAX_EDITED_PARAM_RATIO
    min_contrast_snr: float = DEFAULT_MIN_CONTRAST_SNR
    damping_strength: float = DEFAULT_DAMPING_STRENGTH
    nullspace_eigen_ratio: float = DEFAULT_NULLSPACE_EIGEN_RATIO
    top_k_channels: int = 8
    top_k_rank1_directions: int = 4
    ngram_size: int = DEFAULT_NGRAM_SIZE
    min_sink_repeats: int = DEFAULT_MIN_SINK_REPEATS
    loop_sink_ngram_threshold: float = DEFAULT_LOOP_SINK_NGRAM_THRESHOLD
    entropy_floor_nats: float = DEFAULT_ENTROPY_FLOOR_NATS
    max_nll_drift_ratio: float = DOMAIN_GUARD_MAX_RATIO
    sparse_column_mode: bool = True

    def __post_init__(self) -> None:
        _validate_max_edited_param_ratio(self.max_edited_param_ratio)
        _validate_nonneg_finite(self.min_contrast_snr, "min_contrast_snr")
        _validate_nonneg_finite(self.damping_strength, "damping_strength")
        null_r = _validate_nonneg_finite(self.nullspace_eigen_ratio, "nullspace_eigen_ratio")
        if null_r >= 1.0:
            raise ValueError(f"nullspace_eigen_ratio must be < 1.0, got {null_r}")
        _validate_positive_int(self.top_k_channels, "top_k_channels")
        _validate_positive_int(self.top_k_rank1_directions, "top_k_rank1_directions")
        _validate_positive_int(self.ngram_size, "ngram_size")
        _validate_positive_int(self.min_sink_repeats, "min_sink_repeats")
        thresh = _validate_nonneg_finite(self.loop_sink_ngram_threshold, "loop_sink_ngram_threshold")
        if thresh > 1.0:
            raise ValueError(f"loop_sink_ngram_threshold must be <= 1.0, got {thresh}")
        _validate_nonneg_finite(self.entropy_floor_nats, "entropy_floor_nats")
        drift_r = _validate_positive_finite(self.max_nll_drift_ratio, "max_nll_drift_ratio")
        if drift_r < 1.0:
            raise ValueError(f"max_nll_drift_ratio must be >= 1.0, got {drift_r}")
        if not isinstance(self.sparse_column_mode, bool):
            raise TypeError(
                f"sparse_column_mode must be a bool, got {type(self.sparse_column_mode).__name__}"
            )


def _compute_nullspace_basis(
    m_prod: torch.Tensor,
    nullspace_eigen_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return ``(P_null, V_null, lam_max)`` for the kernel of ``m_prod`` in CPU ``float64``."""
    d_in = int(m_prod.shape[0])
    evals, evecs = torch.linalg.eigh(m_prod)
    lam_max = float(evals.max().item())
    if lam_max <= 1e-24:
        v_null = torch.eye(d_in, dtype=torch.float64)
        return v_null.clone(), v_null, 0.0

    tol = max(float(nullspace_eigen_ratio) * lam_max, 1e-15 * lam_max)
    null_mask = evals <= tol
    if not bool(null_mask.any().item()):
        return torch.zeros((d_in, d_in), dtype=torch.float64), torch.zeros((d_in, 0), dtype=torch.float64), lam_max

    v_null = evecs[:, null_mask].contiguous()
    p_null = (v_null @ v_null.T).contiguous()
    return p_null, v_null, lam_max


def _build_repetition_vocab_contrast(
    vocab_size: int,
    delta_mean_logits: torch.Tensor,
    repetition_token_ids: Sequence[int] | torch.Tensor | None,
) -> tuple[torch.Tensor, list[int]]:
    """Construct zero-sum repetition-token contrast weights over ``[0, vocab_size)``."""
    if repetition_token_ids is not None:
        if isinstance(repetition_token_ids, torch.Tensor):
            if repetition_token_ids.ndim != 1:
                raise ValueError(
                    f"repetition_token_ids tensor must be 1D, got shape {tuple(repetition_token_ids.shape)}"
                )
            if repetition_token_ids.is_floating_point() or repetition_token_ids.dtype == torch.bool:
                raise TypeError("repetition_token_ids tensor must have an integer dtype")
            raw_ids = [int(t) for t in repetition_token_ids.tolist()]
        elif isinstance(repetition_token_ids, Sequence) and not isinstance(
            repetition_token_ids, (str, bytes, bytearray)
        ):
            raw_ids = [_validate_nonneg_int(tok, "repetition_token_id") for tok in repetition_token_ids]
        else:
            raise TypeError("repetition_token_ids must be a sequence or 1D integer tensor")

        unique_ids = sorted(set(raw_ids))
        if not unique_ids:
            raise ValueError("repetition_token_ids must not be empty")
        if any(tok < 0 or tok >= vocab_size for tok in unique_ids):
            raise ValueError(f"repetition_token_ids must lie in [0, {vocab_size})")
    else:
        k_auto = min(4, max(1, vocab_size // 4))
        centered_delta = delta_mean_logits - delta_mean_logits.mean()
        pos_mask = centered_delta > 0.0
        if bool(pos_mask.any().item()):
            n_pos = int(pos_mask.sum().item())
            top_vals, top_idx = torch.topk(centered_delta, k=min(k_auto, n_pos))
            unique_ids = sorted(int(i) for i in top_idx[top_vals > 0.0].tolist())
        else:
            unique_ids = []

    contrast = torch.zeros(vocab_size, dtype=torch.float64)
    if not unique_ids:
        return contrast, []
    if len(unique_ids) == vocab_size:
        contrast.fill_(1.0 / float(vocab_size))
        return contrast, unique_ids

    pos_w = 1.0 / float(len(unique_ids))
    neg_w = -1.0 / float(vocab_size - len(unique_ids))
    contrast.fill_(neg_w)
    contrast[torch.tensor(unique_ids, dtype=torch.long)] = pos_w
    return contrast, unique_ids


def _fold_contrast_snr(z_loop: torch.Tensor, z_prod: torch.Tensor) -> tuple[float, float, float, float]:
    """Compute full-sample means, contrast, and 2-fold OOF-gated contrast SNR per channel."""
    mean_loop = float(z_loop.mean().item())
    mean_prod = float(z_prod.mean().item())
    contrast = mean_loop - mean_prod

    def _single_snr(a_loop: torch.Tensor, a_prod: torch.Tensor) -> float:
        diff = float(a_loop.mean().item()) - float(a_prod.mean().item())
        if diff <= 0.0:
            return 0.0
        v_l = float(a_loop.var(unbiased=False).item()) if a_loop.numel() > 1 else 0.0
        v_p = float(a_prod.var(unbiased=False).item()) if a_prod.numel() > 1 else 0.0
        pooled_std = math.sqrt(0.5 * (v_l + v_p) + 1e-10)
        return diff / pooled_std

    full_snr = _single_snr(z_loop, z_prod)
    if int(z_loop.shape[0]) >= 4 and int(z_prod.shape[0]) >= 4:
        snr_f0 = _single_snr(z_loop[0::2], z_prod[0::2])
        snr_f1 = _single_snr(z_loop[1::2], z_prod[1::2])
        gated_snr = min(full_snr, snr_f0, snr_f1)
    else:
        gated_snr = full_snr
    return mean_loop, mean_prod, contrast, max(0.0, gated_snr)


def _extract_rank1_loop_directions(
    w64: torch.Tensor,
    x_loop: torch.Tensor,
    x_prod: torch.Tensor,
    m_loop: torch.Tensor,
    m_prod: torch.Tensor,
    v_null: torch.Tensor,
    u_rep: torch.Tensor,
    top_k_rank1: int,
) -> list[dict[str, Any]]:
    """Extract top rank-1 null-space directions ``(u_out, v_in)`` driving repetition logits."""
    if v_null.shape[1] == 0:
        return []

    m_loop_null = v_null.T @ m_loop @ v_null
    m_loop_null = 0.5 * (m_loop_null + m_loop_null.T)
    evals_null, q_null = torch.linalg.eigh(m_loop_null)
    evals_desc = evals_null.flip(0)
    q_desc = q_null.flip(1)

    tr_prod = max(float(torch.diagonal(m_prod).mean().item()), 1e-8)
    lam_reg = max(1e-6, 0.05 * tr_prod)
    k_cand = min(int(top_k_rank1) * 2, int(q_desc.shape[1]))

    directions: list[dict[str, Any]] = []
    for k_idx in range(k_cand):
        if float(evals_desc[k_idx].item()) <= 1e-18:
            continue
        v_in = (v_null @ q_desc[:, k_idx]).contiguous()
        v_norm = float(v_in.norm().item())
        if v_norm <= 1e-12:
            continue
        v_in = v_in / v_norm

        raw_out = w64 @ v_in
        out_norm = float(raw_out.norm().item())
        if out_norm <= 1e-12:
            continue
        u_out = raw_out / out_norm
        rep_aff = float((u_rep @ u_out).item())
        if rep_aff < 0.0:
            v_in = -v_in
            u_out = -u_out
            rep_aff = -rep_aff

        proj_loop = (x_loop @ v_in) * (out_norm * rep_aff)
        proj_prod = (x_prod @ v_in) * (out_norm * rep_aff)
        _, _, logit_contrast, snr = _fold_contrast_snr(proj_loop, proj_prod)

        p_quad = max(0.0, float((v_in @ (m_loop @ v_in)).item()))
        s_quad = max(1e-10, float((v_in @ (m_prod @ v_in)).item()) + lam_reg)
        directions.append(
            {
                "direction_index": len(directions),
                "u_out": u_out.contiguous(),
                "v_in": v_in.contiguous(),
                "variance_ratio": p_quad / s_quad,
                "causal_logit_contrast": logit_contrast,
                "repetition_affinity": rep_aff,
                "contrast_snr": snr,
                "priority": max(0.0, logit_contrast) * snr,
            }
        )

    directions.sort(key=lambda item: (item["priority"], item["variance_ratio"]), reverse=True)
    trimmed = directions[: int(top_k_rank1)]
    for idx, item in enumerate(trimmed):
        item["direction_index"] = idx
    return trimmed


@torch.no_grad()
def localize_thinking_loop_channels(
    loop_hidden: torch.Tensor,
    productive_hidden: torch.Tensor,
    down_proj_or_lora_b: nn.Module | torch.Tensor,
    unembed_weight: nn.Module | torch.Tensor,
    *,
    repetition_token_ids: Sequence[int] | torch.Tensor | None = None,
    min_contrast_snr: float = DEFAULT_MIN_CONTRAST_SNR,
    top_k_channels: int = 8,
    top_k_rank1_directions: int = 4,
    nullspace_eigen_ratio: float = DEFAULT_NULLSPACE_EIGEN_RATIO,
    weight_attr: str | None = None,
) -> dict[str, Any]:
    """Rank late-layer output channels and rank-1 directions by loop-vs-productive causal logit contribution.

    Accepts ``loop_hidden`` and ``productive_hidden`` either in the input activation space
    of ``down_proj_or_lora_b`` (width ``d_in``) or in residual hidden space (width ``d_out == H``).
    Computes OOF-gated SNR against ``min_contrast_snr`` so clean models with indistinguishable
    hidden activations produce ``passes_snr_threshold=False`` and zero selected channels.
    """
    snr_thresh = _validate_nonneg_finite(min_contrast_snr, "min_contrast_snr")
    k_ch = _validate_positive_int(top_k_channels, "top_k_channels")
    k_r1 = _validate_positive_int(top_k_rank1_directions, "top_k_rank1_directions")
    null_ratio = _validate_nonneg_finite(nullspace_eigen_ratio, "nullspace_eigen_ratio")
    if null_ratio >= 1.0:
        raise ValueError(f"nullspace_eigen_ratio must be < 1.0, got {null_ratio}")

    loop_2d = _validate_hidden_states(loop_hidden, "loop_hidden")
    prod_2d = _validate_hidden_states(productive_hidden, "productive_hidden")
    if loop_2d.shape[1] != prod_2d.shape[1]:
        raise ValueError(
            f"loop_hidden width {loop_2d.shape[1]} must match productive_hidden width {prod_2d.shape[1]}"
        )

    w_proj = _resolve_weight_tensor(down_proj_or_lora_b, "down_proj_or_lora_b", weight_attr=weight_attr)
    w_unembed = _resolve_weight_tensor(unembed_weight, "unembed_weight")
    w64 = w_proj.detach().to(device="cpu", dtype=torch.float64)
    u64 = w_unembed.detach().to(device="cpu", dtype=torch.float64)

    d_out, d_in = int(w64.shape[0]), int(w64.shape[1])
    vocab_size, h_unembed = int(u64.shape[0]), int(u64.shape[1])
    if h_unembed != d_out:
        raise ValueError(
            f"unembed_weight hidden width {h_unembed} must match down_proj_or_lora_b output channels {d_out}"
        )

    feat_dim = int(loop_2d.shape[1])
    if feat_dim == d_in:
        input_space = "projection_input"
        x_loop, x_prod = loop_2d, prod_2d
        y_loop, y_prod = loop_2d @ w64.T, prod_2d @ w64.T
    elif feat_dim == d_out:
        input_space = "residual_hidden"
        w_pinv_t = torch.linalg.pinv(w64).T
        x_loop = loop_2d @ w_pinv_t
        x_prod = prod_2d @ w_pinv_t
        y_loop, y_prod = loop_2d, prod_2d
    else:
        raise ValueError(
            f"Hidden feature width {feat_dim} must match weight input width {d_in} or output width {d_out}"
        )

    m_prod, _ = second_moment_and_mean(x_prod)
    m_loop, _ = second_moment_and_mean(x_loop)
    p_null, v_null, _ = _compute_nullspace_basis(m_prod, null_ratio)

    delta_mean_y = y_loop.mean(dim=0) - y_prod.mean(dim=0)
    delta_mean_logits = u64 @ delta_mean_y
    vocab_contrast, resolved_tokens = _build_repetition_vocab_contrast(
        vocab_size, delta_mean_logits, repetition_token_ids
    )

    u_rep = u64.T @ vocab_contrast  # (d_out,) repetition affinity per output channel
    row_norms = w64.norm(dim=1)
    null_row_norms = (w64 @ p_null).norm(dim=1)

    z_loop_all = y_loop * u_rep.unsqueeze(0)
    z_prod_all = y_prod * u_rep.unsqueeze(0)

    ranked_channels: list[dict[str, Any]] = []
    for ch_idx in range(d_out):
        w_norm = float(row_norms[ch_idx].item())
        if w_norm <= 1e-12:
            continue
        mean_l, mean_p, contrast, snr = _fold_contrast_snr(z_loop_all[:, ch_idx], z_prod_all[:, ch_idx])
        null_frac = float(null_row_norms[ch_idx].item()) / w_norm
        priority = max(0.0, contrast) * snr * max(null_frac, 1e-4)
        ranked_channels.append(
            {
                "channel_index": ch_idx,
                "causal_logit_contrast": contrast,
                "repetition_affinity": float(u_rep[ch_idx].item()),
                "loop_logit_mean": mean_l,
                "prod_logit_mean": mean_p,
                "contrast_snr": snr,
                "nullspace_norm_fraction": null_frac,
                "row_weight_norm": w_norm,
                "priority": priority,
            }
        )

    ranked_channels.sort(
        key=lambda item: (item["priority"], item["contrast_snr"], item["causal_logit_contrast"]),
        reverse=True,
    )

    selected_channels = [
        int(entry["channel_index"])
        for entry in ranked_channels
        if entry["causal_logit_contrast"] > 0.0
        and entry["contrast_snr"] >= snr_thresh
        and entry["nullspace_norm_fraction"] > 1e-9
    ][:k_ch]

    max_contrast_snr = max((float(entry["contrast_snr"]) for entry in ranked_channels), default=0.0)
    passes_snr = bool(selected_channels) and (max_contrast_snr >= snr_thresh)

    rank1_dirs = _extract_rank1_loop_directions(
        w64, x_loop, x_prod, m_loop, m_prod, v_null, u_rep, k_r1
    )

    return {
        "input_space": input_space,
        "weight_shape": (d_out, d_in),
        "nullspace_dim": int(v_null.shape[1]),
        "repetition_token_ids": resolved_tokens,
        "min_contrast_snr": snr_thresh,
        "max_contrast_snr": max_contrast_snr,
        "passes_snr_threshold": passes_snr,
        "selected_channel_indices": selected_channels,
        "ranked_channels": ranked_channels[:k_ch],
        "ranked_rank1_directions": rank1_dirs,
        "m_prod": m_prod,
        "m_loop": m_loop,
    }


def _compute_strict_edit_cap(total_params: int, max_edited_param_ratio: float) -> int:
    """Return integer parameter cap strictly satisfying ``cap / total_params < max_edited_param_ratio``."""
    raw_cap = float(max_edited_param_ratio) * float(total_params)
    cap = max(0, math.ceil(raw_cap) - 1)
    while cap > 0 and (float(cap) / float(total_params)) >= float(max_edited_param_ratio):
        cap -= 1
    return cap


def _compute_nullspace_suppressor(
    m_prod: torch.Tensor,
    m_loop: torch.Tensor | None,
    nullspace_eigen_ratio: float,
) -> tuple[torch.Tensor, int]:
    """Build exact null-space projector ``P_suppress`` inside ``ker(m_prod)``."""
    p_null, v_null, _ = _compute_nullspace_basis(m_prod, nullspace_eigen_ratio)
    null_dim = int(v_null.shape[1])
    if null_dim == 0 or m_loop is None:
        return p_null, null_dim

    m_loop_null = v_null.T @ m_loop @ v_null
    m_loop_null = 0.5 * (m_loop_null + m_loop_null.T)
    evals, q_vecs = torch.linalg.eigh(m_loop_null)
    lam_max = float(evals.max().item())
    if lam_max <= 1e-24:
        return p_null, null_dim

    active_mask = evals > (1e-6 * lam_max)
    q_act = q_vecs[:, active_mask]
    v_act = v_null @ q_act
    return (v_act @ v_act.T).contiguous(), null_dim


def _solve_sparse_row_nullspace_delta(
    w_row_f64: torch.Tensor,
    p_suppress: torch.Tensor,
    mp_f64: torch.Tensor,
    ml_f64: torch.Tensor | None,
    *,
    damping_strength: float,
    nullspace_eigen_ratio: float,
    remaining_budget: int,
    sparse_column_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve a strictly budgeted row update lying in ``ker(M_prod)``."""
    d_in = int(w_row_f64.numel())
    if remaining_budget <= 0:
        return torch.zeros(d_in, dtype=torch.float64), torch.zeros(d_in, dtype=torch.bool)

    delta_full = -float(damping_strength) * (p_suppress @ w_row_f64)
    max_abs = float(delta_full.abs().max().item())
    if max_abs <= 1e-14:
        return torch.zeros(d_in, dtype=torch.float64), torch.zeros(d_in, dtype=torch.bool)

    active_mask = delta_full.abs() > (1e-12 * max_abs)
    n_active = int(active_mask.sum().item())
    if n_active <= remaining_budget:
        clean_delta = torch.where(active_mask, delta_full, torch.zeros_like(delta_full))
        return clean_delta, active_mask

    if not sparse_column_mode:
        return torch.zeros(d_in, dtype=torch.float64), torch.zeros(d_in, dtype=torch.bool)

    prod_diag = torch.diagonal(mp_f64).clamp_min(0.0)
    col_priority = delta_full.abs() / (prod_diag.sqrt() + 1e-6)
    k_cols = min(d_in, int(remaining_budget))
    _, top_cols = torch.topk(col_priority, k=k_cols)
    top_cols = torch.sort(top_cols).values

    mp_sub = mp_f64.index_select(0, top_cols).index_select(1, top_cols)
    ml_sub = (
        ml_f64.index_select(0, top_cols).index_select(1, top_cols)
        if ml_f64 is not None
        else None
    )
    p_sub, sub_null_dim = _compute_nullspace_suppressor(mp_sub, ml_sub, nullspace_eigen_ratio)
    if sub_null_dim == 0:
        return torch.zeros(d_in, dtype=torch.float64), torch.zeros(d_in, dtype=torch.bool)

    delta_sub = -float(damping_strength) * (p_sub @ w_row_f64.index_select(0, top_cols))
    sub_max = float(delta_sub.abs().max().item())
    if sub_max <= 1e-14:
        return torch.zeros(d_in, dtype=torch.float64), torch.zeros(d_in, dtype=torch.bool)

    out_delta = torch.zeros(d_in, dtype=torch.float64)
    out_mask = torch.zeros(d_in, dtype=torch.bool)
    keep_sub = delta_sub.abs() > (1e-12 * sub_max)
    selected_idx = top_cols[keep_sub]
    out_delta[selected_idx] = delta_sub[keep_sub]
    out_mask[selected_idx] = True
    return out_delta, out_mask


def _build_edit_noop_report(
    *,
    w_shape: tuple[int, int],
    denom: int,
    max_ratio: float,
    max_params_cap: int,
    nullspace_dim: int,
    contrast_snr: float,
    noop_reason: str,
) -> dict[str, Any]:
    """Assemble standardized telemetry when loop-suppression edit performs a safety no-op."""
    return {
        "weight_shape": w_shape,
        "edited_elements": 0,
        "total_model_params": denom,
        "edited_param_ratio": 0.0,
        "max_edited_param_ratio": max_ratio,
        "max_edited_params_cap": max_params_cap,
        "budget_respected": True,
        "edited_channels": [],
        "nullspace_dim": nullspace_dim,
        "max_contrast_snr": contrast_snr,
        "noop_reason": noop_reason,
    }


def _apply_nullspace_edit_with_cap(
    w_tensor: torch.Tensor,
    mp_f64: torch.Tensor,
    ml_f64: torch.Tensor | None,
    candidate_channels: Sequence[int],
    *,
    denom: int,
    max_ratio: float,
    max_params_cap: int,
    damping_f: float,
    null_ratio: float,
    sparse_column_mode: bool,
    contrast_snr: float,
) -> dict[str, Any]:
    """Execute budgeted row-wise sparse null-space damping on resolved tensor ``w_tensor``."""
    d_out, d_in = int(w_tensor.shape[0]), int(w_tensor.shape[1])
    p_suppress, null_dim = _compute_nullspace_suppressor(mp_f64, ml_f64, null_ratio)
    if null_dim == 0 or float(p_suppress.abs().max().item()) <= 1e-14:
        return _build_edit_noop_report(
            w_shape=(d_out, d_in),
            denom=denom,
            max_ratio=max_ratio,
            max_params_cap=max_params_cap,
            nullspace_dim=null_dim,
            contrast_snr=contrast_snr,
            noop_reason="empty_productive_nullspace",
        )

    before_w = w_tensor.detach().clone()
    remaining_budget = max_params_cap
    edited_channels: list[int] = []

    for ch_idx in candidate_channels:
        if remaining_budget <= 0:
            break
        row_before = w_tensor[ch_idx].detach()
        row_f64 = row_before.to(device="cpu", dtype=torch.float64)
        delta_row, support_mask = _solve_sparse_row_nullspace_delta(
            row_f64,
            p_suppress,
            mp_f64,
            ml_f64,
            damping_strength=damping_f,
            nullspace_eigen_ratio=null_ratio,
            remaining_budget=remaining_budget,
            sparse_column_mode=sparse_column_mode,
        )
        n_support = int(support_mask.sum().item())
        if n_support == 0:
            continue

        row_new = row_before.clone()
        mask_dev = support_mask.to(device=row_before.device)
        updated_vals = (row_f64[support_mask] + delta_row[support_mask]).to(
            device=row_before.device, dtype=row_before.dtype
        )
        row_new[mask_dev] = updated_vals
        actual_row_edits = int(torch.ne(row_new, row_before).sum().item())
        if actual_row_edits > 0:
            w_tensor[ch_idx].copy_(row_new)
            remaining_budget -= actual_row_edits
            edited_channels.append(ch_idx)

    edited_elements = int(torch.ne(w_tensor, before_w).sum().item())
    edited_ratio = float(edited_elements) / float(denom)
    noop_reason = None if edited_elements > 0 else "zero_nullspace_overlap"

    return {
        "weight_shape": (d_out, d_in),
        "edited_elements": edited_elements,
        "total_model_params": denom,
        "edited_param_ratio": edited_ratio,
        "max_edited_param_ratio": max_ratio,
        "max_edited_params_cap": max_params_cap,
        "budget_respected": edited_ratio < max_ratio and edited_elements <= max_params_cap,
        "edited_channels": edited_channels,
        "nullspace_dim": null_dim,
        "max_contrast_snr": contrast_snr,
        "noop_reason": noop_reason,
    }


@torch.no_grad()
def apply_nullspace_loop_suppression_edit_inplace(
    target_module_or_weight: nn.Module | torch.Tensor,
    m_prod: torch.Tensor,
    localization: Mapping[str, Any] | None = None,
    *,
    target_channels: Sequence[int] | None = None,
    m_loop: torch.Tensor | None = None,
    total_model_params: int | None = None,
    max_edited_param_ratio: float = DEFAULT_MAX_EDITED_PARAM_RATIO,
    damping_strength: float = DEFAULT_DAMPING_STRENGTH,
    min_contrast_snr: float = DEFAULT_MIN_CONTRAST_SNR,
    nullspace_eigen_ratio: float = DEFAULT_NULLSPACE_EIGEN_RATIO,
    weight_attr: str | None = None,
    sparse_column_mode: bool = True,
) -> dict[str, Any]:
    """Apply a strictly budgeted sparse damping update inside ``ker(M_prod)`` in-place.

    Modifications are strictly capped below ``max_edited_param_ratio <= 0.0002`` of
    ``total_model_params`` (defaulting to ``target_weight.numel()`` when omitted).
    When ``localization`` reports ``max_contrast_snr < min_contrast_snr``, performs a
    bitwise no-op to protect clean models.
    """
    w_tensor = _resolve_weight_tensor(
        target_module_or_weight, "target_module_or_weight", weight_attr=weight_attr
    )
    d_out, d_in = int(w_tensor.shape[0]), int(w_tensor.shape[1])

    mp_f64 = _validate_square_moment(m_prod, d_in, "m_prod")
    max_ratio = _validate_max_edited_param_ratio(max_edited_param_ratio)
    damping_f = _validate_nonneg_finite(damping_strength, "damping_strength")
    snr_thresh = _validate_nonneg_finite(min_contrast_snr, "min_contrast_snr")
    null_ratio = _validate_nonneg_finite(nullspace_eigen_ratio, "nullspace_eigen_ratio")
    if null_ratio >= 1.0:
        raise ValueError(f"nullspace_eigen_ratio must be < 1.0, got {null_ratio}")
    if not isinstance(sparse_column_mode, bool):
        raise TypeError(f"sparse_column_mode must be a bool, got {type(sparse_column_mode).__name__}")

    denom = (
        _validate_positive_int(total_model_params, "total_model_params")
        if total_model_params is not None
        else w_tensor.numel()
    )

    ml_src = m_loop
    if ml_src is None and localization is not None and "m_loop" in localization:
        ml_src = localization["m_loop"]
    ml_f64 = _validate_square_moment(ml_src, d_in, "m_loop") if ml_src is not None else None

    if localization is not None and not isinstance(localization, Mapping):
        raise TypeError(f"localization must be a mapping or None, got {type(localization).__name__}")
    if localization is None and target_channels is None:
        raise ValueError("Either localization or explicit target_channels must be provided")

    contrast_snr = float(localization.get("max_contrast_snr", 0.0)) if localization is not None else float("inf")
    max_params_cap = _compute_strict_edit_cap(denom, max_ratio)

    if localization is not None:
        passes_flag = localization.get("passes_snr_threshold")
        if contrast_snr < snr_thresh:
            return _build_edit_noop_report(
                w_shape=(d_out, d_in),
                denom=denom,
                max_ratio=max_ratio,
                max_params_cap=max_params_cap,
                nullspace_dim=int(localization.get("nullspace_dim", 0)),
                contrast_snr=contrast_snr,
                noop_reason="contrast_snr_below_threshold",
            )
        if passes_flag is False and target_channels is None:
            return _build_edit_noop_report(
                w_shape=(d_out, d_in),
                denom=denom,
                max_ratio=max_ratio,
                max_params_cap=max_params_cap,
                nullspace_dim=int(localization.get("nullspace_dim", 0)),
                contrast_snr=contrast_snr,
                noop_reason="zero_nullspace_overlap",
            )

    if damping_f == 0.0:
        return _build_edit_noop_report(
            w_shape=(d_out, d_in),
            denom=denom,
            max_ratio=max_ratio,
            max_params_cap=max_params_cap,
            nullspace_dim=0,
            contrast_snr=contrast_snr,
            noop_reason="zero_damping_strength",
        )

    if max_params_cap == 0:
        return _build_edit_noop_report(
            w_shape=(d_out, d_in),
            denom=denom,
            max_ratio=max_ratio,
            max_params_cap=0,
            nullspace_dim=0,
            contrast_snr=contrast_snr,
            noop_reason="budget_below_one_parameter",
        )

    raw_channels = (
        target_channels if target_channels is not None else localization.get("selected_channel_indices", [])
    )
    if isinstance(raw_channels, (str, bytes)) or not isinstance(raw_channels, Sequence):
        raise TypeError("target_channels must be a sequence of integer channel indices")

    seen_ch: set[int] = set()
    ordered_channels: list[int] = []
    for raw_idx in raw_channels:
        idx = _validate_nonneg_int(raw_idx, "target_channel")
        if idx >= d_out:
            raise ValueError(f"target_channel {idx} out of bounds for d_out={d_out}")
        if idx not in seen_ch:
            seen_ch.add(idx)
            ordered_channels.append(idx)

    if not ordered_channels:
        return _build_edit_noop_report(
            w_shape=(d_out, d_in),
            denom=denom,
            max_ratio=max_ratio,
            max_params_cap=max_params_cap,
            nullspace_dim=0,
            contrast_snr=contrast_snr,
            noop_reason="no_target_channels",
        )

    return _apply_nullspace_edit_with_cap(
        w_tensor,
        mp_f64,
        ml_f64,
        ordered_channels,
        denom=denom,
        max_ratio=max_ratio,
        max_params_cap=max_params_cap,
        damping_f=damping_f,
        null_ratio=null_ratio,
        sparse_column_mode=sparse_column_mode,
        contrast_snr=contrast_snr,
    )


def _normalize_token_sequences(token_seqs: Any, name: str) -> list[list[int]]:
    """Validate and convert token sequences (2D tensor or sequence of sequences) to int lists."""
    if isinstance(token_seqs, torch.Tensor):
        if token_seqs.ndim != 2 or token_seqs.shape[0] == 0 or token_seqs.shape[1] == 0:
            raise ValueError(f"{name} tensor must be non-empty 2D, got shape {tuple(token_seqs.shape)}")
        if token_seqs.is_floating_point() or token_seqs.dtype == torch.bool:
            raise TypeError(f"{name} tensor must have an integer dtype")
        seqs = [[int(t) for t in row] for row in token_seqs.tolist()]
    elif isinstance(token_seqs, Sequence) and not isinstance(token_seqs, (str, bytes, bytearray)):
        if not token_seqs:
            raise ValueError(f"{name} must be non-empty")
        seqs = []
        for idx, row in enumerate(token_seqs):
            if isinstance(row, torch.Tensor):
                if row.ndim != 1 or row.numel() == 0:
                    raise ValueError(f"{name}[{idx}] must be non-empty 1D")
                if row.is_floating_point() or row.dtype == torch.bool:
                    raise TypeError(f"{name}[{idx}] must have an integer dtype")
                seqs.append([int(t) for t in row.tolist()])
            elif isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
                if not row:
                    raise ValueError(f"{name}[{idx}] must be non-empty")
                seqs.append([_validate_nonneg_int(t, f"{name}[{idx}]") for t in row])
            else:
                raise TypeError(f"{name}[{idx}] must be a sequence or 1D integer tensor")
    else:
        raise TypeError(f"{name} must be a 2D integer tensor or sequence of token sequences")

    for idx, seq in enumerate(seqs):
        if any(tok < 0 for tok in seq):
            raise ValueError(f"{name}[{idx}] must contain only non-negative token IDs")
    return seqs


def _sequence_ngram_stats(
    seq: Sequence[int],
    *,
    ngram_size: int,
    min_sink_repeats: int,
    loop_sink_ngram_threshold: float,
) -> tuple[float, bool, int]:
    """Compute ``(ngram_repetition_rate, is_loop_sink, max_ngram_count)`` for one token sequence."""
    if len(seq) < ngram_size:
        raise ValueError(
            f"Token sequence length {len(seq)} must be >= ngram_size ({ngram_size})"
        )
    ngrams = [tuple(seq[i : i + ngram_size]) for i in range(len(seq) - ngram_size + 1)]
    counts = Counter(ngrams)
    total_ngrams = len(ngrams)
    unique_ngrams = len(counts)
    max_count = max(counts.values())
    rep_rate = 1.0 - (float(unique_ngrams) / float(total_ngrams))
    is_sink = (rep_rate >= loop_sink_ngram_threshold) or (max_count >= min_sink_repeats)
    return rep_rate, is_sink, max_count


def _compute_corpus_repetition_stats(
    seqs: Sequence[Sequence[int]],
    *,
    ngram_size: int,
    min_sink_repeats: int,
    loop_sink_ngram_threshold: float,
) -> dict[str, Any]:
    """Aggregate n-gram repetition rate and sequence loop-sink rate across ``seqs``."""
    rep_rates: list[float] = []
    sink_flags: list[bool] = []
    max_repeats: list[int] = []
    for seq in seqs:
        rr, sink, mc = _sequence_ngram_stats(
            seq,
            ngram_size=ngram_size,
            min_sink_repeats=min_sink_repeats,
            loop_sink_ngram_threshold=loop_sink_ngram_threshold,
        )
        rep_rates.append(rr)
        sink_flags.append(sink)
        max_repeats.append(mc)
    return {
        "sequence_count": len(seqs),
        "ngram_repetition_rate": float(np.mean(rep_rates)),
        "loop_sink_rate": float(np.mean([1.0 if f else 0.0 for f in sink_flags])),
        "max_ngram_repeat_count": max(max_repeats),
    }


def _shannon_entropy_stats(logits: Any, name: str) -> dict[str, float]:
    """Compute exact next-token Shannon entropy statistics (nats) in ``float64``."""
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(logits).__name__}")
    if logits.ndim not in (2, 3) or any(d == 0 for d in logits.shape) or logits.shape[-1] < 2:
        raise ValueError(
            f"{name} must be non-empty 2D/3D tensor with vocab >= 2, got shape {tuple(logits.shape)}"
        )
    lg = logits.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(lg).all().item()):
        raise ValueError(f"{name} must contain only finite values")

    log_probs = F.log_softmax(lg, dim=-1)
    probs = log_probs.exp()
    ent = (-(probs * log_probs).sum(dim=-1)).reshape(-1).clamp_min(0.0)
    ent_np = ent.numpy()
    return {
        "mean_entropy_nats": float(np.mean(ent_np)),
        "min_entropy_nats": float(np.min(ent_np)),
        "p10_entropy_nats": float(np.percentile(ent_np, 10.0)),
    }


def _nll_guard_stats(
    nll_before: Any,
    nll_after: Any,
    *,
    max_nll_drift_ratio: float,
    label: str,
) -> dict[str, Any]:
    """Evaluate paired NLL drift ratio and guard compliance (``<= max_nll_drift_ratio``)."""
    arr_b = np.asarray(
        nll_before.detach().cpu().numpy() if isinstance(nll_before, torch.Tensor) else nll_before,
        dtype=np.float64,
    ).reshape(-1)
    arr_a = np.asarray(
        nll_after.detach().cpu().numpy() if isinstance(nll_after, torch.Tensor) else nll_after,
        dtype=np.float64,
    ).reshape(-1)
    if arr_b.size == 0 or arr_b.shape != arr_a.shape:
        raise ValueError(
            f"{label} NLL before/after must be non-empty and match in shape; got {arr_b.shape} vs {arr_a.shape}"
        )
    if not (bool(np.isfinite(arr_b).all()) and bool(np.isfinite(arr_a).all())):
        raise ValueError(f"{label} NLL values must be finite")
    if bool((arr_b <= 0.0).any()) or bool((arr_a < 0.0).any()):
        raise ValueError(f"{label} NLL values must be non-negative with positive baseline")

    mean_b = float(arr_b.mean())
    mean_a = float(arr_a.mean())
    ratio = mean_a / max(mean_b, 1e-8)
    pct_drift = 100.0 * (ratio - 1.0)
    return {
        "n": int(arr_b.size),
        "nll_mean_before": mean_b,
        "nll_mean_after": mean_a,
        "nll_ratio": ratio,
        "pct_drift": pct_drift,
        "guard_ok": ratio <= max_nll_drift_ratio,
    }


def evaluate_repetition_and_guard_metrics(
    generated_ids_after: Sequence[Sequence[int]] | torch.Tensor,
    *,
    generated_ids_before: Sequence[Sequence[int]] | torch.Tensor | None = None,
    logits_after: torch.Tensor | None = None,
    logits_before: torch.Tensor | None = None,
    general_nll_before: Sequence[float] | np.ndarray | torch.Tensor | float | None = None,
    general_nll_after: Sequence[float] | np.ndarray | torch.Tensor | float | None = None,
    specialist_nll_before: Sequence[float] | np.ndarray | torch.Tensor | float | None = None,
    specialist_nll_after: Sequence[float] | np.ndarray | torch.Tensor | float | None = None,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
    min_sink_repeats: int = DEFAULT_MIN_SINK_REPEATS,
    loop_sink_ngram_threshold: float = DEFAULT_LOOP_SINK_NGRAM_THRESHOLD,
    entropy_floor_nats: float = DEFAULT_ENTROPY_FLOOR_NATS,
    max_nll_drift_ratio: float = DOMAIN_GUARD_MAX_RATIO,
    edited_elements: int = 0,
    total_model_params: int | None = None,
    max_edited_param_ratio: float = DEFAULT_MAX_EDITED_PARAM_RATIO,
) -> dict[str, Any]:
    """Compute n-gram repetition-sink rate, Shannon entropy recovery, and NLL guard compliance."""
    ng = _validate_positive_int(ngram_size, "ngram_size")
    min_rep = _validate_positive_int(min_sink_repeats, "min_sink_repeats")
    thresh = _validate_nonneg_finite(loop_sink_ngram_threshold, "loop_sink_ngram_threshold")
    if thresh > 1.0:
        raise ValueError(f"loop_sink_ngram_threshold must be <= 1.0, got {thresh}")
    ent_floor = _validate_nonneg_finite(entropy_floor_nats, "entropy_floor_nats")
    drift_cap = _validate_positive_finite(max_nll_drift_ratio, "max_nll_drift_ratio")
    if drift_cap < 1.0:
        raise ValueError(f"max_nll_drift_ratio must be >= 1.0, got {drift_cap}")
    ed_elems = _validate_nonneg_int(edited_elements, "edited_elements")
    max_ed_ratio = _validate_max_edited_param_ratio(max_edited_param_ratio)

    seqs_after = _normalize_token_sequences(generated_ids_after, "generated_ids_after")
    after_rep = _compute_corpus_repetition_stats(
        seqs_after, ngram_size=ng, min_sink_repeats=min_rep, loop_sink_ngram_threshold=thresh
    )

    before_rep: dict[str, Any] | None = None
    if generated_ids_before is not None:
        seqs_before = _normalize_token_sequences(generated_ids_before, "generated_ids_before")
        before_rep = _compute_corpus_repetition_stats(
            seqs_before, ngram_size=ng, min_sink_repeats=min_rep, loop_sink_ngram_threshold=thresh
        )

    if (logits_before is None) != (logits_after is None) and logits_after is None:
        raise ValueError("logits_after must be provided when logits_before is provided")

    ent_after = _shannon_entropy_stats(logits_after, "logits_after") if logits_after is not None else None
    ent_before = _shannon_entropy_stats(logits_before, "logits_before") if logits_before is not None else None

    if (general_nll_before is None) != (general_nll_after is None):
        raise ValueError("general_nll_before and general_nll_after must both be provided or both be None")
    if (specialist_nll_before is None) != (specialist_nll_after is None):
        raise ValueError("specialist_nll_before and specialist_nll_after must both be provided or both be None")

    gen_guard = (
        _nll_guard_stats(
            general_nll_before, general_nll_after, max_nll_drift_ratio=drift_cap, label="general"
        )
        if general_nll_before is not None
        else None
    )
    spec_guard = (
        _nll_guard_stats(
            specialist_nll_before, specialist_nll_after, max_nll_drift_ratio=drift_cap, label="specialist"
        )
        if specialist_nll_before is not None
        else None
    )

    if total_model_params is not None:
        tot_p = _validate_positive_int(total_model_params, "total_model_params")
        ed_ratio = float(ed_elems) / float(tot_p)
        budget_ok = ed_ratio < max_ed_ratio
    else:
        ed_ratio = 0.0
        budget_ok = ed_elems == 0

    nll_guard_ok = (gen_guard["guard_ok"] if gen_guard else True) and (
        spec_guard["guard_ok"] if spec_guard else True
    )
    loop_cured = after_rep["loop_sink_rate"] < 0.05 and after_rep["ngram_repetition_rate"] < 0.05

    ent_floor_recovered: bool | None = None
    ent_delta: float | None = None
    if ent_after is not None:
        ent_floor_recovered = ent_after["min_entropy_nats"] >= ent_floor
        if ent_before is not None:
            ent_delta = ent_after["mean_entropy_nats"] - ent_before["mean_entropy_nats"]
            ent_floor_recovered = ent_floor_recovered and (
                ent_after["min_entropy_nats"] >= ent_before["min_entropy_nats"]
            )

    guard_ok = bool(nll_guard_ok and budget_ok)
    rep_after_val = float(after_rep["loop_sink_rate"])
    rep_before_val = float(before_rep["loop_sink_rate"]) if before_rep is not None else rep_after_val
    anse_energy = (rep_after_val - rep_before_val) + (0.0 if (guard_ok and loop_cured) else ANSE_MAXIMUM_PAIN)

    return {
        "ngram_size": ng,
        "ngram_repetition_rate_before": before_rep["ngram_repetition_rate"] if before_rep else None,
        "ngram_repetition_rate_after": after_rep["ngram_repetition_rate"],
        "loop_sink_rate_before": before_rep["loop_sink_rate"] if before_rep else None,
        "loop_sink_rate_after": after_rep["loop_sink_rate"],
        "loop_rate_reduction": (
            (before_rep["loop_sink_rate"] - after_rep["loop_sink_rate"]) if before_rep else None
        ),
        "loop_cured": loop_cured,
        "entropy_before": ent_before,
        "entropy_after": ent_after,
        "entropy_recovery_delta": ent_delta,
        "entropy_floor_recovered": ent_floor_recovered,
        "general_nll_guard": gen_guard,
        "specialist_nll_guard": spec_guard,
        "nll_guard_ok": nll_guard_ok,
        "edited_elements": ed_elems,
        "edited_param_ratio": ed_ratio,
        "budget_ok": budget_ok,
        "guard_ok": guard_ok,
        "anse_energy": anse_energy,
    }


def _collect_moe_expert_latent_activations(
    moe_mlp: nn.Module,
    x_2d: torch.Tensor,
    exp_key: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Compute routing-weighted SwiGLU LoRA latent activations entering ``exp.w3_B``."""
    exp = moe_mlp.experts[exp_key]
    if float(exp.w3_B.detach().abs().max().item()) <= 1e-12:
        return None
    w_dt = moe_mlp.base_mlp.w1.weight.dtype
    x_base = x_2d.to(dtype=w_dt)
    h1 = moe_mlp.base_mlp.w1(x_base).to(torch.float32)
    h2 = moe_mlp.base_mlp.w2(x_base).to(torch.float32)
    x_fp32 = x_2d.to(torch.float32)

    logits = moe_mlp.router(x_fp32)
    if getattr(moe_mlp, "routing_mode", "sigma_topk") == "sigma_topk":
        probs = torch.sigmoid(logits)
        top_vals, top_idx = torch.topk(probs, k=int(moe_mlp.top_k), dim=-1)
        gates = top_vals
    else:
        probs = F.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(probs, k=int(moe_mlp.top_k), dim=-1)
        gates = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    e_idx = int(exp_key)
    hit = (top_idx == e_idx).to(torch.float32)
    w_tok = (gates * hit).sum(dim=-1, keepdim=True)
    dh1 = F.linear(F.linear(x_fp32, exp.w1_A), exp.w1_B) * exp.scaling
    dh2 = F.linear(F.linear(x_fp32, exp.w2_A), exp.w2_B) * exp.scaling
    act_full = F.silu(h1 + dh1) * (h2 + dh2)
    z_latent = F.linear(act_full, exp.w3_A) * (exp.scaling * w_tok)
    return z_latent, exp.w3_B


def _collect_dense_mlp_activations(
    mlp: nn.Module,
    x_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SwiGLU intermediate activations entering ``mlp.w3.weight``."""
    w_dt = mlp.w1.weight.dtype
    x_in = x_2d.to(dtype=w_dt)
    act = (F.silu(mlp.w1(x_in)) * mlp.w2(x_in)).to(torch.float32)
    return act, mlp.w3.weight


@torch.no_grad()
def apply_astra_loop_suppression_inplace(
    model: nn.Module,
    loop_ids: torch.Tensor,
    productive_ids: torch.Tensor,
    *,
    repetition_token_ids: Sequence[int] | torch.Tensor | None = None,
    layer_indices: Sequence[int] | None = None,
    config: AstraLoopSteeringConfig | None = None,
    max_edited_param_ratio: float = DEFAULT_MAX_EDITED_PARAM_RATIO,
    min_contrast_snr: float = DEFAULT_MIN_CONTRAST_SNR,
    damping_strength: float = DEFAULT_DAMPING_STRENGTH,
    nullspace_eigen_ratio: float = DEFAULT_NULLSPACE_EIGEN_RATIO,
    score_from: int = 0,
    batch_size: int = 16,
    max_channels_per_layer: int = 4,
) -> dict[str, Any]:
    """Localize and suppress thinking-loop channels across late layers of a multi-layer transformer or MoE.

    Runs strictly under ``@torch.no_grad()`` (zero gradient steps), enforces the global
    ``max_edited_param_ratio <= 0.0002`` parameter budget across all edited layers, and
    leaves clean models bitwise untouched when loop contrast SNR is below ``min_contrast_snr``.
    """
    if not isinstance(model, nn.Module) or not hasattr(model, "blocks") or len(model.blocks) == 0:
        raise ValueError("model must be an nn.Module exposing a non-empty 'blocks' list")
    for attr in ("wte", "ln", "head", "cos_cached", "sin_cached"):
        if not hasattr(model, attr):
            raise ValueError(f"model must expose '{attr}'")

    if not isinstance(loop_ids, torch.Tensor) or not isinstance(productive_ids, torch.Tensor):
        raise TypeError("loop_ids and productive_ids must be torch.Tensor instances")
    if loop_ids.ndim != 2 or productive_ids.ndim != 2 or loop_ids.shape[0] == 0 or productive_ids.shape[0] == 0:
        raise ValueError("loop_ids and productive_ids must be non-empty 2D tensors")

    if config is not None:
        if not isinstance(config, AstraLoopSteeringConfig):
            raise TypeError(f"config must be AstraLoopSteeringConfig or None, got {type(config).__name__}")
        cfg = config
    else:
        cfg = AstraLoopSteeringConfig(
            max_edited_param_ratio=max_edited_param_ratio,
            min_contrast_snr=min_contrast_snr,
            damping_strength=damping_strength,
            nullspace_eigen_ratio=nullspace_eigen_ratio,
            top_k_channels=_validate_positive_int(max_channels_per_layer, "max_channels_per_layer"),
        )

    offset = _validate_nonneg_int(score_from, "score_from")
    bsz = _validate_positive_int(batch_size, "batch_size")
    n_blocks = len(model.blocks)
    if layer_indices is None:
        target_layers = set(range(max(0, n_blocks - 2), n_blocks))
    else:
        target_layers = {_validate_nonneg_int(l_i, "layer_index") for l_i in layer_indices}
        if any(l_i >= n_blocks for l_i in target_layers):
            raise ValueError(f"layer_indices must lie in [0, {n_blocks})")

    total_model_params = sum(p.numel() for p in model.parameters())
    global_cap = _compute_strict_edit_cap(total_model_params, cfg.max_edited_param_ratio)
    remaining_budget = global_cap

    was_training = model.training
    model.eval()
    dev = model.wte.weight.device
    cos = model.cos_cached.to(dev)
    sin = model.sin_cached.to(dev)
    start_l = min(offset, max(0, int(loop_ids.shape[1]) - 1))
    start_p = min(offset, max(0, int(productive_ids.shape[1]) - 1))

    def _run_embeddings(ids_t: torch.Tensor) -> torch.Tensor:
        parts = [
            compute_embeddings_batch(model, ids_t[i : i + bsz].to(dev))
            for i in range(0, ids_t.shape[0], bsz)
        ]
        return torch.cat(parts, dim=0)

    def _pre_mlp_seq(blk: nn.Module, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xm_list, mi_list = [], []
        for i in range(0, x_seq.shape[0], bsz):
            xm, mi = forward_block_pre_mlp(blk, x_seq[i : i + bsz], cos, sin)
            xm_list.append(xm)
            mi_list.append(mi)
        return torch.cat(xm_list, dim=0), torch.cat(mi_list, dim=0)

    def _post_mlp_seq(blk: nn.Module, x_mid: torch.Tensor, mlp_in: torch.Tensor) -> torch.Tensor:
        blk_dt = blk.ln1.weight.dtype
        parts = [
            (x_mid[i : i + bsz] + blk.mlp(mlp_in[i : i + bsz].to(dtype=blk_dt)).to(torch.float32))
            for i in range(0, x_mid.shape[0], bsz)
        ]
        return torch.cat(parts, dim=0)

    layer_reports: list[dict[str, Any]] = []
    total_edited = 0
    try:
        x_loop_seq = _run_embeddings(loop_ids)
        x_prod_seq = _run_embeddings(productive_ids)

        for l_idx, blk in enumerate(model.blocks):
            xm_l, mi_l = _pre_mlp_seq(blk, x_loop_seq)
            xm_p, mi_p = _pre_mlp_seq(blk, x_prod_seq)

            if l_idx in target_layers and remaining_budget > 0:
                l_flat = mi_l[:, start_l:, :].reshape(-1, mi_l.shape[-1])
                p_flat = mi_p[:, start_p:, :].reshape(-1, mi_p.shape[-1])
                candidates: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []

                if hasattr(blk.mlp, "experts"):
                    for e_key in sorted(blk.mlp.experts.keys(), key=int):
                        res_l = _collect_moe_expert_latent_activations(blk.mlp, l_flat, e_key)
                        res_p = _collect_moe_expert_latent_activations(blk.mlp, p_flat, e_key)
                        if res_l is not None and res_p is not None:
                            candidates.append((f"expert_{e_key}.w3_B", res_l[0], res_p[0], res_l[1]))
                else:
                    act_l, w_l = _collect_dense_mlp_activations(blk.mlp, l_flat)
                    act_p, _ = _collect_dense_mlp_activations(blk.mlp, p_flat)
                    candidates.append(("dense.w3", act_l, act_p, w_l))

                evaluated: list[tuple[float, str, torch.Tensor, dict[str, Any]]] = []
                for label, act_l, act_p, w_proj in candidates:
                    loc = localize_thinking_loop_channels(
                        act_l,
                        act_p,
                        w_proj,
                        model.head.weight,
                        repetition_token_ids=repetition_token_ids,
                        min_contrast_snr=cfg.min_contrast_snr,
                        top_k_channels=cfg.top_k_channels,
                        top_k_rank1_directions=cfg.top_k_rank1_directions,
                        nullspace_eigen_ratio=cfg.nullspace_eigen_ratio,
                    )
                    top_prio = float(loc["ranked_channels"][0]["priority"]) if loc["ranked_channels"] else 0.0
                    evaluated.append((top_prio, label, w_proj, loc))

                evaluated.sort(key=lambda item: (item[3]["passes_snr_threshold"], item[0]), reverse=True)
                for _, label, w_proj, loc in evaluated:
                    if remaining_budget <= 0:
                        break
                    if not loc["passes_snr_threshold"]:
                        reason = (
                            "contrast_snr_below_threshold"
                            if float(loc["max_contrast_snr"]) < cfg.min_contrast_snr
                            else "zero_nullspace_overlap"
                        )
                        layer_reports.append(
                            {"layer": l_idx, "target": label, "edited_elements": 0, "noop_reason": reason}
                        )
                        continue
                    edit_rep = _apply_nullspace_edit_with_cap(
                        w_proj,
                        loc["m_prod"],
                        loc["m_loop"],
                        loc["selected_channel_indices"],
                        denom=total_model_params,
                        max_ratio=cfg.max_edited_param_ratio,
                        max_params_cap=remaining_budget,
                        damping_f=cfg.damping_strength,
                        null_ratio=cfg.nullspace_eigen_ratio,
                        sparse_column_mode=cfg.sparse_column_mode,
                        contrast_snr=float(loc["max_contrast_snr"]),
                    )
                    ed_count = int(edit_rep["edited_elements"])
                    remaining_budget -= ed_count
                    total_edited += ed_count
                    layer_reports.append({"layer": l_idx, "target": label, **edit_rep})

            x_loop_seq = _post_mlp_seq(blk, xm_l, mi_l)
            x_prod_seq = _post_mlp_seq(blk, xm_p, mi_p)
    finally:
        model.train(was_training)

    edited_ratio = float(total_edited) / float(total_model_params)
    return {
        "total_model_params": total_model_params,
        "edited_elements": total_edited,
        "edited_param_ratio": edited_ratio,
        "max_edited_param_ratio": cfg.max_edited_param_ratio,
        "budget_respected": edited_ratio < cfg.max_edited_param_ratio and total_edited <= global_cap,
        "target_layers": sorted(target_layers),
        "layer_reports": layer_reports,
        "noop_reason": None if total_edited > 0 else "contrast_snr_below_threshold",
    }


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "AstraLoopSteeringConfig",
    "DEFAULT_DAMPING_STRENGTH",
    "DEFAULT_ENTROPY_FLOOR_NATS",
    "DEFAULT_LOOP_SINK_NGRAM_THRESHOLD",
    "DEFAULT_MAX_EDITED_PARAM_RATIO",
    "DEFAULT_MIN_CONTRAST_SNR",
    "DEFAULT_MIN_SINK_REPEATS",
    "DEFAULT_NGRAM_SIZE",
    "DEFAULT_NULLSPACE_EIGEN_RATIO",
    "DOMAIN_GUARD_MAX_RATIO",
    "MAX_EDITED_PARAM_RATIO_CEILING",
    "apply_astra_loop_suppression_inplace",
    "apply_nullspace_loop_suppression_edit_inplace",
    "evaluate_repetition_and_guard_metrics",
    "localize_thinking_loop_channels",
]

# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Sub-0.02% Parameter-Budget Thinking-Loop & Repetition-Attractor Weight-Space Suppressor.

Implements training-free, zero-gradient Weight-Space Programming (WSP) surgery for
open-weight autoregressive transformers and Sparse MoEs (``CrowdV1WithEngram``,
``CrowdRuntimeMLP``, ``CrowdLoRAMoEMLP``, and generic ``down_proj`` / ``w3`` blocks),
inspired by MIT CSAIL <0.02% weight surgery for eliminating degenerate thinking loops:

1. ``diagnose_repetition_attractor_subspace(loop_hidden_states, concise_hidden_states, *, ngram_order=4, min_oof_snr=0.25)``:
   Computes exact n-gram token repetition ratios and extracts 2-fold out-of-fold (OOF)
   cross-validated state and state-transition attractor directions responsible for
   degenerate repetition cycles, gating out indistinguishable noise.
2. ``synthesize_sub002pct_antiloop_tail_surgery(tail_down_weights, final_norm_weight, attractor_dirs, p_null, *, damp_beta=0.35, tau_temp=1.0, total_model_params=None, max_budget_fraction=0.0002)``:
   Synthesizes strictly factorized rank-``k`` tail-layer down-projection updates inside
   the AlphaEdit general-prose null-space ``P_0`` plus optional folded norm temperature
   scaling ``1 / tau_temp``, rigorously obeying ``<= 0.02%`` total parameter budget.
3. ``execute_verified_antiloop_wsp_surgery(...)``:
   Executes verified tail-layer surgery in-place (or cloned) under ANSE thermodynamic
   gating, committing edits if and only if loop repetition drops, ``ANSE Delta E < 0``,
   and general/worst-domain NLL drift stays ``<= +1.0%``; otherwise restores exact
   parent weights atomically bit-for-bit and emits a SHA-256 provenance receipt.
"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.swarm.byzantine_wsp_delta_aggregator import (
    compute_payload_sha256,
    compute_state_sha256,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdLoRAExpertAdapter,
    CrowdLoRAMoEMLP,
)
from hivemind.swarm.logit_calibration_temperature_surgeon import (
    fold_temperature_into_norm_params,
)
from hivemind.swarm.tangent_task_vector_surgeon import (
    compute_prose_nullspace_projector_scaled,
)
from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    forward_model_pre_ln,
    per_chunk_nll_from_pre_ln,
    second_moment_and_mean,
)

SUPPRESSOR_SCHEMA: str = "hivemind.thinking_loop_wsp_suppressor/1"
SUPPRESSOR_RECEIPT_OBJECT: str = "hivemind.thinking_loop_wsp_suppressor_receipt"

MAX_BUDGET_FRACTION_CEILING: float = 0.0002
DEFAULT_NGRAM_ORDER: int = 4
DEFAULT_MIN_OOF_SNR: float = 0.25
DEFAULT_MAX_ATTRACTOR_RANK: int = 4
DEFAULT_DAMP_BETA: float = 0.35
DEFAULT_TAU_TEMP: float = 1.0
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_GENERAL_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_LOOP_ENERGY_WEIGHT: float = 1.0
DEFAULT_NULLSPACE_SV_TOL: float = 1e-8
DEFAULT_PROSE_SV_REL_TOL: float = 1e-3
DEFAULT_PROSE_VARIANCE_THRESHOLD: float = 0.9999
GENERAL_EXPERT_PREFIX: str = "E0_"
REASONING_EXPERT_PREFIX: str = "E2_"
VALID_PROJECTION_SIDES: frozenset[str] = frozenset({"auto", "input", "output"})

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


def _validate_finite_float(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number (excluding bool)."""
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


def _validate_max_budget_fraction(value: Any) -> float:
    """Enforce ``0.0 < max_budget_fraction <= 0.0002`` (sub-0.02% ceiling)."""
    frac = _validate_positive_finite(value, "max_budget_fraction")
    if frac > MAX_BUDGET_FRACTION_CEILING:
        raise ValueError(
            f"max_budget_fraction must be <= {MAX_BUDGET_FRACTION_CEILING}, got {frac}"
        )
    return frac


def _validate_unit_interval(value: Any, name: str, *, allow_zero: bool = True) -> float:
    """Validate a finite number inside ``[0.0, 1.0]`` (or ``(0.0, 1.0]``)."""
    out = _validate_nonneg_finite(value, name)
    if not allow_zero and out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    if out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    return out


def _validate_nonempty_str(value: Any, name: str) -> str:
    """Validate a non-empty printable string."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    if not value.strip() or any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{name} must be a non-empty printable string")
    return value


def _validate_finite_fp_tensor(
    tensor: Any,
    name: str,
    *,
    allowed_ndims: tuple[int, ...] = (2, 3),
) -> torch.Tensor:
    """Validate a non-empty finite floating-point tensor and convert to CPU ``float64``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim not in allowed_ndims or any(dim == 0 for dim in tensor.shape):
        raise ValueError(
            f"{name} must have ndim in {allowed_ndims} and be non-empty, got shape {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    out = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(out).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _validate_square_projector(
    proj: Any,
    name: str,
    *,
    expected_dim: int | None = None,
) -> torch.Tensor:
    """Validate a finite symmetric square 2D projector matrix on CPU ``float64``."""
    mat = _validate_finite_fp_tensor(proj, name, allowed_ndims=(2,))
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be square, got shape {tuple(mat.shape)}")
    if expected_dim is not None and int(mat.shape[0]) != expected_dim:
        raise ValueError(
            f"{name} shape {tuple(mat.shape)} does not match expected ({expected_dim}, {expected_dim})"
        )
    return (0.5 * (mat + mat.T)).contiguous()


def verify_antiloop_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != SUPPRESSOR_RECEIPT_OBJECT or receipt.get("schema") != SUPPRESSOR_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


def _restore_snapshot_bitwise(
    model: nn.Module,
    snapshot: Mapping[str, torch.Tensor],
    expected_sha256: str,
) -> str:
    """Restore ``model`` state atomically in-place from CPU snapshot and assert SHA-256."""
    cur_sd = model.state_dict()
    for key, snap_t in snapshot.items():
        if key in cur_sd:
            cur_sd[key].copy_(snap_t.to(device=cur_sd[key].device, dtype=cur_sd[key].dtype))
    restored_sha = compute_state_sha256(model)
    if restored_sha != expected_sha256:
        raise RuntimeError(
            f"Atomic bitwise rollback failed: expected {expected_sha256[:12]}, got {restored_sha[:12]}"
        )
    return restored_sha


# ---------------------------------------------------------------------------
# 2. N-Gram Repetition & 2-Fold OOF Attractor Subspace Diagnosis
# ---------------------------------------------------------------------------


def _normalize_token_sequences(token_seqs: Any, name: str) -> list[list[int]]:
    """Validate and convert token sequences (2D tensor or sequence of sequences) to int lists."""
    if isinstance(token_seqs, torch.Tensor):
        if token_seqs.ndim != 2 or token_seqs.shape[0] == 0 or token_seqs.shape[1] == 0:
            raise ValueError(f"{name} tensor must be non-empty 2D, got shape {tuple(token_seqs.shape)}")
        if token_seqs.is_floating_point() or token_seqs.dtype == torch.bool:
            raise TypeError(f"{name} tensor must have an integer dtype")
        seqs = [[int(tok) for tok in row] for row in token_seqs.tolist()]
    elif isinstance(token_seqs, Sequence) and not isinstance(token_seqs, (str, bytes, bytearray)):
        if not token_seqs:
            raise ValueError(f"{name} must not be empty")
        seqs = []
        for idx, row in enumerate(token_seqs):
            if isinstance(row, torch.Tensor):
                if row.ndim != 1 or row.numel() == 0:
                    raise ValueError(f"{name}[{idx}] must be non-empty 1D")
                if row.is_floating_point() or row.dtype == torch.bool:
                    raise TypeError(f"{name}[{idx}] must have an integer dtype")
                seqs.append([int(tok) for tok in row.tolist()])
            elif isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
                if not row:
                    raise ValueError(f"{name}[{idx}] must not be empty")
                seqs.append([_validate_nonneg_int(tok, f"{name}[{idx}]") for tok in row])
            else:
                raise TypeError(f"{name}[{idx}] must be a sequence or 1D integer tensor")
    else:
        raise TypeError(f"{name} must be a 2D integer tensor or sequence of token sequences")

    for idx, seq in enumerate(seqs):
        if any(tok < 0 for tok in seq):
            raise ValueError(f"{name}[{idx}] must contain only non-negative token IDs")
    return seqs


def compute_ngram_repetition_metrics(
    token_sequences: Sequence[Sequence[int]] | torch.Tensor,
    *,
    ngram_order: int = DEFAULT_NGRAM_ORDER,
) -> dict[str, Any]:
    """Measure n-gram token repetition rates and 4-gram loop-sink coverage across sequences."""
    ng = _validate_positive_int(ngram_order, "ngram_order")
    seqs = _normalize_token_sequences(token_sequences, "token_sequences")

    def _order_stats(order: int) -> tuple[float, float, float, int]:
        rates: list[float] = []
        coverages: list[float] = []
        sink_flags: list[float] = []
        max_c = 1
        for seq in seqs:
            if len(seq) < order:
                raise ValueError(
                    f"Sequence length {len(seq)} must be >= ngram_order ({order})"
                )
            grams = [tuple(seq[i : i + order]) for i in range(len(seq) - order + 1)]
            counts = Counter(grams)
            total = len(grams)
            unique = len(counts)
            seq_max = max(counts.values())
            max_c = max(max_c, seq_max)
            rep_rate = 1.0 - (float(unique) / float(total))
            rep_cov = float(sum(c for c in counts.values() if c >= 2)) / float(total)
            rates.append(rep_rate)
            coverages.append(rep_cov)
            sink_flags.append(1.0 if (rep_rate > 0.25 or seq_max >= 2) else 0.0)
        return (
            float(np.mean(rates)),
            float(np.mean(coverages)),
            float(np.mean(sink_flags)),
            max_c,
        )

    rep_rate, rep_cov, sink_rate, max_count = _order_stats(ng)
    if ng == 4:
        rep_4gram = rep_rate
    elif all(len(seq) >= 4 for seq in seqs):
        rep_4gram, _, _, _ = _order_stats(4)
    else:
        rep_4gram = rep_rate

    return {
        "ngram_order": ng,
        "sequence_count": len(seqs),
        "repeat_ngram_ratio": rep_rate,
        "repeat_4gram_ratio": rep_4gram,
        "repeated_ngram_coverage": rep_cov,
        "loop_sink_rate": sink_rate,
        "max_ngram_count": max_count,
    }


def _unpack_hidden_input(
    hidden_input: Any,
    token_ids_override: Any,
    name: str,
) -> tuple[list[torch.Tensor], Any]:
    """Normalize 2D/3D hidden states or mapping/sequence of trajectories into 2D/3D CPU float64."""
    toks = token_ids_override
    raw = hidden_input
    if isinstance(hidden_input, Mapping):
        if "hidden_states" not in hidden_input:
            raise ValueError(f"{name} mapping must contain 'hidden_states'")
        raw = hidden_input["hidden_states"]
        if toks is None and "token_ids" in hidden_input:
            toks = hidden_input["token_ids"]

    if isinstance(raw, torch.Tensor):
        t64 = _validate_finite_fp_tensor(raw, name, allowed_ndims=(2, 3))
        if t64.ndim == 3:
            return [t64[idx] for idx in range(t64.shape[0])], toks
        return [t64], toks

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        if not raw:
            raise ValueError(f"{name} sequence must not be empty")
        trajs: list[torch.Tensor] = []
        width: int | None = None
        for idx, item in enumerate(raw):
            t64 = _validate_finite_fp_tensor(item, f"{name}[{idx}]", allowed_ndims=(2,))
            if width is None:
                width = int(t64.shape[1])
            elif int(t64.shape[1]) != width:
                raise ValueError(f"{name}[{idx}] feature dim {t64.shape[1]} != {width}")
            trajs.append(t64)
        return trajs, toks

    raise TypeError(
        f"{name} must be a 2D/3D torch.Tensor, Sequence[Tensor], or Mapping with 'hidden_states'"
    )


def _split_trajectories_for_oof(
    trajs: list[torch.Tensor],
    *,
    is_sequence_batch: bool,
    transition_weight: float,
    ngram_order: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Split trajectories cleanly into fold0/fold1 state+transition features without leakage."""
    states_list: list[torch.Tensor] = []
    trans_list: list[torch.Tensor] = []
    periodicity_scores: list[float] = []

    for traj in trajs:
        states_list.append(traj)
        if is_sequence_batch and int(traj.shape[0]) >= 2:
            diffs = traj[1:] - traj[:-1]
            trans_list.append(math.sqrt(transition_weight) * diffs)
            if int(traj.shape[0]) > ngram_order:
                step_norm = float(diffs.norm(dim=1).mean().item())
                cyc_norm = float((traj[ngram_order:] - traj[:-ngram_order]).norm(dim=1).mean().item())
                ratio = max(0.0, 1.0 - (cyc_norm / max(step_norm * math.sqrt(float(ngram_order)), 1e-8)))
                periodicity_scores.append(min(1.0, ratio))

    if is_sequence_batch and len(trajs) >= 4:
        f0_states = torch.cat([trajs[i] for i in range(0, len(trajs), 2)], dim=0)
        f1_states = torch.cat([trajs[i] for i in range(1, len(trajs), 2)], dim=0)
        if trans_list:
            f0_tr = torch.cat([trans_list[i] for i in range(0, len(trans_list), 2)], dim=0)
            f1_tr = torch.cat([trans_list[i] for i in range(1, len(trans_list), 2)], dim=0)
            f0 = torch.cat([f0_states, f0_tr], dim=0)
            f1 = torch.cat([f1_states, f1_tr], dim=0)
        else:
            f0, f1 = f0_states, f1_states
    else:
        all_states = torch.cat(states_list, dim=0)
        n_rows = int(all_states.shape[0])
        if n_rows < 4:
            raise ValueError(
                f"At least 4 state vectors are required for 2-fold OOF diagnosis, got {n_rows}"
            )
        if is_sequence_batch:
            mid = n_rows // 2
            f0_s, f1_s = all_states[:mid], all_states[mid:]
        else:
            f0_s, f1_s = all_states[0::2], all_states[1::2]
        if trans_list:
            all_tr = torch.cat(trans_list, dim=0)
            mid_t = int(all_tr.shape[0]) // 2
            f0 = torch.cat([f0_s, all_tr[:mid_t]], dim=0) if mid_t > 0 else f0_s
            f1 = torch.cat([f1_s, all_tr[mid_t:]], dim=0) if int(all_tr.shape[0]) - mid_t > 0 else f1_s
        else:
            f0, f1 = f0_s, f1_s

    mean_periodicity = float(np.mean(periodicity_scores)) if periodicity_scores else 0.0
    return f0.contiguous(), f1.contiguous(), torch.cat([f0, f1], dim=0), (
        torch.cat(trans_list, dim=0) if trans_list else torch.empty((0, trajs[0].shape[1]), dtype=torch.float64)
    ), mean_periodicity


def _compute_snr_1d(z_loop: torch.Tensor, z_concise: torch.Tensor) -> float:
    """Compute directional standardized contrast ``(mean_L - mean_C) / pooled_std``."""
    diff = float(z_loop.mean().item()) - float(z_concise.mean().item())
    if diff <= 0.0:
        return 0.0
    v_l = float(z_loop.var(unbiased=False).item()) if z_loop.numel() > 1 else 0.0
    v_c = float(z_concise.var(unbiased=False).item()) if z_concise.numel() > 1 else 0.0
    pooled = math.sqrt(0.5 * (v_l + v_c) + 1e-8)
    return max(0.0, diff / pooled)


def _chance_corrected_cosine(cos_raw: float, dim: int, rank_k: int) -> float:
    """Discount random-subspace alignment expected by chance in ``R^dim``."""
    chance = min(0.65, 1.5 * math.sqrt(float(rank_k) / float(max(2, dim))))
    if cos_raw <= chance:
        return 0.0
    return min(1.0, (cos_raw - chance) / max(1e-6, 1.0 - chance))


def _fit_fold_attractor_probes(
    l_train: torch.Tensor,
    c_train: torch.Tensor,
    *,
    max_rank: int,
    ridge_scale: float,
    omega_concise: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit linear contrastive direction and generalized variance-ratio modes on a fold."""
    d = int(l_train.shape[1])
    eye_d = torch.eye(d, dtype=torch.float64)

    ml, mu_l = second_moment_and_mean(l_train)
    mc, mu_c = second_moment_and_mean(c_train)

    # 1. Linear mean-contrastive probe
    comb = ml + float(omega_concise) * mc
    tr_comb = max(float(torch.diagonal(comb).mean().item()), 1e-6)
    lam_lin = max(float(ridge_scale) * tr_comb, 1e-6)
    u_raw = torch.linalg.solve(comb + lam_lin * eye_d, mu_l - mu_c)
    u_norm = float(u_raw.norm().item())
    u_lin = (u_raw / u_norm) if u_norm > 1e-12 else torch.zeros(d, dtype=torch.float64)

    # 2. Generalized variance-ratio modes (catches zero-mean cyclic attractors as well)
    tr_c = max(float(torch.diagonal(mc).mean().item()), 1e-6)
    lam_c = max(float(ridge_scale) * tr_c, 1e-6)
    b_mat = mc + lam_c * eye_d
    ev_b, q_b = torch.linalg.eigh(b_mat)
    ev_b_c = ev_b.clamp_min(1e-8)
    b_inv_half = (q_b * ev_b_c.rsqrt().unsqueeze(0)) @ q_b.T
    whitened = 0.5 * (b_inv_half @ ml @ b_inv_half + (b_inv_half @ ml @ b_inv_half).T)
    mu_asc, q_asc = torch.linalg.eigh(whitened)
    mu_desc = mu_asc.flip(0)
    q_desc = q_asc.flip(1)

    v_raw = b_inv_half @ q_desc[:, :max_rank]
    v_norms = v_raw.norm(dim=0, keepdim=True).clamp_min(1e-12)
    v_unit = (v_raw / v_norms).T.contiguous()  # (k, D)
    return u_lin, v_unit, mu_desc[:max_rank]


@torch.no_grad()
def diagnose_repetition_attractor_subspace(
    loop_hidden_states: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, Any],
    concise_hidden_states: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, Any],
    *,
    ngram_order: int = DEFAULT_NGRAM_ORDER,
    min_oof_snr: float = DEFAULT_MIN_OOF_SNR,
    loop_token_ids: Sequence[Sequence[int]] | torch.Tensor | None = None,
    concise_token_ids: Sequence[Sequence[int]] | torch.Tensor | None = None,
    max_attractor_rank: int = DEFAULT_MAX_ATTRACTOR_RANK,
    ridge_scale: float = 0.5,
    omega_concise: float = 6.0,
    transition_weight: float = 0.5,
) -> dict[str, Any]:
    """Measure n-gram repetition and extract 2-fold OOF state-transition attractor directions."""
    ng = _validate_positive_int(ngram_order, "ngram_order")
    snr_thresh = _validate_nonneg_finite(min_oof_snr, "min_oof_snr")
    k_max = _validate_positive_int(max_attractor_rank, "max_attractor_rank")
    ridge_f = _validate_positive_finite(ridge_scale, "ridge_scale")
    omega_f = _validate_nonneg_finite(omega_concise, "omega_concise")
    trans_w = _validate_nonneg_finite(transition_weight, "transition_weight")

    loop_is_seq = isinstance(loop_hidden_states, torch.Tensor) and loop_hidden_states.ndim == 3
    if isinstance(loop_hidden_states, Sequence) and not isinstance(loop_hidden_states, (str, bytes, bytearray)):
        loop_is_seq = True
    if isinstance(loop_hidden_states, Mapping) and isinstance(loop_hidden_states.get("hidden_states"), torch.Tensor):
        loop_is_seq = loop_hidden_states["hidden_states"].ndim == 3

    concise_is_seq = isinstance(concise_hidden_states, torch.Tensor) and concise_hidden_states.ndim == 3
    if isinstance(concise_hidden_states, Sequence) and not isinstance(concise_hidden_states, (str, bytes, bytearray)):
        concise_is_seq = True
    if isinstance(concise_hidden_states, Mapping) and isinstance(concise_hidden_states.get("hidden_states"), torch.Tensor):
        concise_is_seq = concise_hidden_states["hidden_states"].ndim == 3

    l_trajs, resolved_l_toks = _unpack_hidden_input(loop_hidden_states, loop_token_ids, "loop_hidden_states")
    c_trajs, resolved_c_toks = _unpack_hidden_input(concise_hidden_states, concise_token_ids, "concise_hidden_states")
    d = int(l_trajs[0].shape[1])
    if int(c_trajs[0].shape[1]) != d:
        raise ValueError(
            f"loop_hidden_states width {d} must match concise_hidden_states width {c_trajs[0].shape[1]}"
        )
    k_eff = min(k_max, d)

    l_f0, l_f1, l_all, l_tr, l_period = _split_trajectories_for_oof(
        l_trajs, is_sequence_batch=loop_is_seq, transition_weight=trans_w, ngram_order=ng
    )
    c_f0, c_f1, c_all, _, c_period = _split_trajectories_for_oof(
        c_trajs, is_sequence_batch=concise_is_seq, transition_weight=trans_w, ngram_order=ng
    )

    if resolved_l_toks is not None:
        l_rep = compute_ngram_repetition_metrics(resolved_l_toks, ngram_order=ng)
        loop_repeat_ngram_ratio = float(l_rep["repeat_ngram_ratio"])
        repeat_4gram_ratio = float(l_rep["repeat_4gram_ratio"])
        loop_sink_rate = float(l_rep["loop_sink_rate"])
        token_ngram_measured = True
    else:
        loop_repeat_ngram_ratio = 0.0
        repeat_4gram_ratio = 0.0
        loop_sink_rate = 0.0
        token_ngram_measured = False

    if resolved_c_toks is not None:
        c_rep = compute_ngram_repetition_metrics(resolved_c_toks, ngram_order=ng)
        concise_repeat_ngram_ratio = float(c_rep["repeat_ngram_ratio"])
        concise_repeat_4gram_ratio = float(c_rep["repeat_4gram_ratio"])
    else:
        concise_repeat_ngram_ratio = 0.0
        concise_repeat_4gram_ratio = 0.0

    # Fit 2-fold cross-validated attractor probes
    u0, v0, _ = _fit_fold_attractor_probes(
        l_f0, c_f0, max_rank=k_eff, ridge_scale=ridge_f, omega_concise=omega_f
    )
    u1, v1, _ = _fit_fold_attractor_probes(
        l_f1, c_f1, max_rank=k_eff, ridge_scale=ridge_f, omega_concise=omega_f
    )

    # Out-of-fold linear directional SNR
    lin_snr_0 = _compute_snr_1d(l_f1 @ u0, c_f1 @ u0)
    lin_snr_1 = _compute_snr_1d(l_f0 @ u1, c_f0 @ u1)
    lin_cos = max(0.0, float((u0 @ u1).item()))
    lin_stab = _chance_corrected_cosine(lin_cos, d, 1)
    oof_linear_snr = min(lin_snr_0, lin_snr_1) * lin_stab

    # Out-of-fold quadratic subspace energy SNR (top-r modes)
    best_energy_snr = 0.0
    best_energy_stab = 0.0
    best_r = 1
    for r_chk in range(1, min(k_eff, 2) + 1):
        # Orthonormalize top-r bases for Grassmann overlap
        q0, _ = torch.linalg.qr(v0[:r_chk].T, mode="reduced")
        q1, _ = torch.linalg.qr(v1[:r_chk].T, mode="reduced")
        sub_cos = float(torch.linalg.vector_norm(q0.T @ q1).item()) / math.sqrt(float(r_chk))
        stab_r = _chance_corrected_cosine(sub_cos, d, r_chk)
        e_l1 = (l_f1 @ q0).pow(2).sum(dim=1)
        e_c1 = (c_f1 @ q0).pow(2).sum(dim=1)
        e_l0 = (l_f0 @ q1).pow(2).sum(dim=1)
        e_c0 = (c_f0 @ q1).pow(2).sum(dim=1)
        esnr = min(_compute_snr_1d(e_l1, e_c1), _compute_snr_1d(e_l0, e_c0)) * stab_r
        if esnr > best_energy_snr:
            best_energy_snr = esnr
            best_energy_stab = stab_r
            best_r = r_chk

    oof_snr = max(oof_linear_snr, best_energy_snr)
    fold_cos_stab = lin_stab if oof_linear_snr >= best_energy_snr else best_energy_stab
    passes_gate = bool(oof_snr >= snr_thresh)

    if not passes_gate:
        return {
            "ngram_order": ng,
            "min_oof_snr": snr_thresh,
            "oof_snr": float(oof_snr),
            "oof_linear_snr": float(oof_linear_snr),
            "oof_energy_snr": float(best_energy_snr),
            "fold_cosine_stability": float(fold_cos_stab),
            "passes_oof_gate": False,
            "attractor_rank": 0,
            "attractor_dirs": torch.zeros((0, d), dtype=torch.float64),
            "attractor_eigenvalues": [],
            "attractor_strengths": [],
            "repeat_ngram_ratio": loop_repeat_ngram_ratio,
            "repeat_4gram_ratio": repeat_4gram_ratio,
            "concise_repeat_ngram_ratio": concise_repeat_ngram_ratio,
            "concise_repeat_4gram_ratio": concise_repeat_4gram_ratio,
            "loop_sink_rate": loop_sink_rate,
            "state_periodicity_ratio": l_period,
            "concise_periodicity_ratio": c_period,
            "feature_dim": d,
            "token_ngram_measured": token_ngram_measured,
            "transition_mode_used": int(l_tr.shape[0]) > 0,
        }

    # Full-split orthonormalized attractor basis prioritizing validated direction(s)
    u_full, v_full, mu_full = _fit_fold_attractor_probes(
        l_all, c_all, max_rank=k_eff, ridge_scale=ridge_f, omega_concise=omega_f
    )

    cand_rows: list[torch.Tensor] = []
    cand_weights: list[float] = []
    if float(u_full.norm().item()) > 1e-12 and oof_linear_snr >= 0.25 * snr_thresh:
        cand_rows.append(u_full)
        cand_weights.append(max(1.0, oof_linear_snr))

    for idx in range(k_eff):
        eig_ratio = float(mu_full[idx].item())
        if idx < best_r or eig_ratio > 1.25:
            cand_rows.append(v_full[idx])
            cand_weights.append(max(0.25, eig_ratio - 1.0))

    if not cand_rows:
        cand_rows.append(v_full[0])
        cand_weights.append(1.0)

    # Orthonormalize via modified Gram-Schmidt preserving priority order
    ortho_rows: list[torch.Tensor] = []
    kept_eigs: list[float] = []
    for row_vec, w_val in zip(cand_rows, cand_weights, strict=True):
        if len(ortho_rows) >= k_eff:
            break
        v = row_vec.clone()
        for prev in ortho_rows:
            v = v - float((v @ prev).item()) * prev
        v_n = float(v.norm().item())
        if v_n > 1e-6:
            ortho_rows.append((v / v_n).contiguous())
            kept_eigs.append(float(w_val))

    dirs_tensor = (
        torch.stack(ortho_rows, dim=0).contiguous()
        if ortho_rows
        else torch.zeros((0, d), dtype=torch.float64)
    )
    max_eig = max(kept_eigs, default=1.0)
    strengths = [float(e / max_eig) for e in kept_eigs]

    return {
        "ngram_order": ng,
        "min_oof_snr": snr_thresh,
        "oof_snr": float(oof_snr),
        "oof_linear_snr": float(oof_linear_snr),
        "oof_energy_snr": float(best_energy_snr),
        "fold_cosine_stability": float(fold_cos_stab),
        "passes_oof_gate": True,
        "attractor_rank": int(dirs_tensor.shape[0]),
        "attractor_dirs": dirs_tensor,
        "attractor_eigenvalues": kept_eigs,
        "attractor_strengths": strengths,
        "repeat_ngram_ratio": loop_repeat_ngram_ratio,
        "repeat_4gram_ratio": repeat_4gram_ratio,
        "concise_repeat_ngram_ratio": concise_repeat_ngram_ratio,
        "concise_repeat_4gram_ratio": concise_repeat_4gram_ratio,
        "loop_sink_rate": loop_sink_rate,
        "state_periodicity_ratio": l_period,
        "concise_periodicity_ratio": c_period,
        "feature_dim": d,
        "token_ngram_measured": token_ngram_measured,
        "transition_mode_used": int(l_tr.shape[0]) > 0,
    }


# ---------------------------------------------------------------------------
# 3. Sub-0.02% Factorized AlphaEdit Null-Space Surgery & Norm Scaling
# ---------------------------------------------------------------------------


def _resolve_tail_weights(
    tail_down_weights: Any,
) -> tuple[str, list[str], list[torch.Tensor]]:
    """Normalize tail down-projection weight input into ``(kind, names, tensors)``."""
    if isinstance(tail_down_weights, nn.Module):
        if isinstance(tail_down_weights, nn.Linear):
            t = tail_down_weights.weight
            kind = "module"
        elif hasattr(tail_down_weights, "w3") and isinstance(tail_down_weights.w3, nn.Linear):
            t = tail_down_weights.w3.weight
            kind = "module"
        elif hasattr(tail_down_weights, "down_proj") and isinstance(tail_down_weights.down_proj, nn.Linear):
            t = tail_down_weights.down_proj.weight
            kind = "module"
        elif hasattr(tail_down_weights, "w3_B") and isinstance(tail_down_weights.w3_B, torch.Tensor):
            t = tail_down_weights.w3_B
            kind = "module"
        elif hasattr(tail_down_weights, "weight") and isinstance(tail_down_weights.weight, torch.Tensor):
            t = tail_down_weights.weight
            kind = "module"
        else:
            raise ValueError(
                f"Cannot resolve 2D down-projection weight from {type(tail_down_weights).__name__}"
            )
        _validate_finite_fp_tensor(t, "tail_down_weights", allowed_ndims=(2,))
        return kind, ["tail_down"], [t]

    if isinstance(tail_down_weights, torch.Tensor):
        _validate_finite_fp_tensor(tail_down_weights, "tail_down_weights", allowed_ndims=(2,))
        return "tensor", ["tail_down"], [tail_down_weights]

    if isinstance(tail_down_weights, Mapping):
        if not tail_down_weights:
            raise ValueError("tail_down_weights mapping must not be empty")
        names: list[str] = []
        tensors: list[torch.Tensor] = []
        for k, v in tail_down_weights.items():
            k_str = _validate_nonempty_str(k, "tail_down_weights key")
            _validate_finite_fp_tensor(v, f"tail_down_weights[{k_str!r}]", allowed_ndims=(2,))
            names.append(k_str)
            tensors.append(v)
        return "mapping", names, tensors

    if isinstance(tail_down_weights, Sequence) and not isinstance(tail_down_weights, (str, bytes, bytearray)):
        if not tail_down_weights:
            raise ValueError("tail_down_weights sequence must not be empty")
        names = [f"tail_{idx}" for idx in range(len(tail_down_weights))]
        tensors = []
        for idx, v in enumerate(tail_down_weights):
            _validate_finite_fp_tensor(v, f"tail_down_weights[{idx}]", allowed_ndims=(2,))
            tensors.append(v)
        return "sequence", names, tensors

    raise TypeError(
        "tail_down_weights must be a 2D Tensor, nn.Module, Sequence[Tensor], or Mapping[str, Tensor]"
    )


def _resolve_norm_params(
    final_norm_weight: Any,
    final_norm_bias: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Validate optional final norm weight and bias tensors."""
    if final_norm_weight is None:
        if final_norm_bias is not None:
            raise ValueError("final_norm_bias cannot be provided when final_norm_weight is None")
        return None, None
    if isinstance(final_norm_weight, nn.Module):
        w_t = getattr(final_norm_weight, "weight", None)
        b_t = getattr(final_norm_weight, "bias", None) if final_norm_bias is None else final_norm_bias
        if not isinstance(w_t, torch.Tensor):
            raise ValueError("final_norm_weight module must expose a 1D .weight tensor")
    elif isinstance(final_norm_weight, torch.Tensor):
        w_t = final_norm_weight
        b_t = final_norm_bias
    else:
        raise TypeError(
            f"final_norm_weight must be a 1D torch.Tensor, nn.Module, or None; got {type(final_norm_weight).__name__}"
        )

    _validate_finite_fp_tensor(w_t, "final_norm_weight", allowed_ndims=(1,))
    if b_t is not None:
        _validate_finite_fp_tensor(b_t, "final_norm_bias", allowed_ndims=(1,))
        if int(b_t.shape[0]) != int(w_t.shape[0]):
            raise ValueError(
                f"final_norm_bias length {b_t.shape[0]} != final_norm_weight length {w_t.shape[0]}"
            )
    return w_t, b_t


def _resolve_attractor_matrix(attractor_dirs: Any) -> tuple[torch.Tensor, bool]:
    """Extract normalized ``(k, D)`` attractor directions and whether OOF gate passed."""
    if isinstance(attractor_dirs, Mapping):
        if "attractor_dirs" not in attractor_dirs:
            raise ValueError("attractor_dirs mapping must contain 'attractor_dirs'")
        if attractor_dirs.get("passes_oof_gate") is False:
            raw = attractor_dirs["attractor_dirs"]
            if isinstance(raw, torch.Tensor) and raw.ndim == 2:
                return torch.zeros((0, int(raw.shape[1])), dtype=torch.float64), False
            return torch.zeros((0, 0), dtype=torch.float64), False
        dirs_raw = attractor_dirs["attractor_dirs"]
    else:
        dirs_raw = attractor_dirs

    if not isinstance(dirs_raw, torch.Tensor):
        raise TypeError(
            f"attractor_dirs must be a torch.Tensor or diagnosis dict, got {type(dirs_raw).__name__}"
        )
    if not dirs_raw.is_floating_point():
        raise TypeError(f"attractor_dirs must have a floating-point dtype, got {dirs_raw.dtype}")
    if dirs_raw.ndim == 1:
        if dirs_raw.numel() == 0:
            raise ValueError("1D attractor_dirs must be non-empty")
        mat = dirs_raw.detach().to(device="cpu", dtype=torch.float64).unsqueeze(0)
    elif dirs_raw.ndim == 2:
        if dirs_raw.shape[1] == 0:
            raise ValueError("attractor_dirs feature width must be > 0")
        mat = dirs_raw.detach().to(device="cpu", dtype=torch.float64)
    else:
        raise ValueError(f"attractor_dirs must be 1D or 2D, got shape {tuple(dirs_raw.shape)}")

    if not bool(torch.isfinite(mat).all().item()):
        raise ValueError("attractor_dirs must contain only finite values")
    if mat.shape[0] == 0:
        return mat, False

    # Orthonormalize non-zero rows via thin SVD so output directions are strictly orthonormal
    row_norms = mat.norm(dim=1)
    nonzero_mask = row_norms > 1e-12
    if not bool(nonzero_mask.any().item()):
        return torch.zeros((0, int(mat.shape[1])), dtype=torch.float64), False
    active = mat[nonzero_mask]
    _, s_vals, vh = torch.linalg.svd(active, full_matrices=False)
    top_s = float(s_vals[0].item())
    kept = s_vals > (1e-8 * top_s)
    return vh[kept].contiguous(), True


def _resolve_layer_projector(
    layer_name: str,
    d_out: int,
    d_in: int,
    p_null: torch.Tensor | Mapping[str, torch.Tensor],
    projection_side: str,
) -> tuple[torch.Tensor, str]:
    """Select and validate the AlphaEdit projector ``(P_0, side)`` for a ``(d_out, d_in)`` weight."""
    if projection_side not in VALID_PROJECTION_SIDES:
        raise ValueError(
            f"projection_side must be one of {sorted(VALID_PROJECTION_SIDES)}, got {projection_side!r}"
        )

    if isinstance(p_null, torch.Tensor):
        p64 = _validate_square_projector(p_null, "p_null")
        d_p = int(p64.shape[0])
        if projection_side == "input":
            if d_p != d_in:
                raise ValueError(f"p_null dim {d_p} != input dim {d_in}")
            return p64, "input"
        if projection_side == "output":
            if d_p != d_out:
                raise ValueError(f"p_null dim {d_p} != output dim {d_out}")
            return p64, "output"
        if d_p == d_in:
            return p64, "input"
        if d_p == d_out:
            return p64, "output"
        raise ValueError(
            f"p_null dim {d_p} matches neither input dim {d_in} nor output dim {d_out}"
        )

    if not isinstance(p_null, Mapping) or not p_null:
        raise TypeError("p_null must be a square 2D torch.Tensor or non-empty Mapping[str, Tensor]")

    # Priority lookup: exact layer key -> explicit side key -> generic keys
    candidates: list[tuple[str, str | None]] = [(layer_name, None)]
    if projection_side in ("auto", "input"):
        candidates.extend([(f"{layer_name}.in", "input"), ("in", "input"), ("act", "input"), ("x", "input")])
    if projection_side in ("auto", "output"):
        candidates.extend([(f"{layer_name}.out", "output"), ("out", "output")])
    candidates.append(("default", None))

    for key, hint_side in candidates:
        if key in p_null:
            p64 = _validate_square_projector(p_null[key], f"p_null[{key!r}]")
            d_p = int(p64.shape[0])
            eff_side = hint_side or projection_side
            if eff_side == "input" and d_p == d_in:
                return p64, "input"
            if eff_side == "output" and d_p == d_out:
                return p64, "output"
            if eff_side == "auto":
                if d_p == d_in:
                    return p64, "input"
                if d_p == d_out:
                    return p64, "output"

    raise ValueError(
        f"No compatible projector found in p_null for {layer_name!r} shape ({d_out}, {d_in})"
    )


def _orthonormalize_columns(mat: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal column basis for ``mat`` dropping near-zero singular modes."""
    if mat.numel() == 0 or float(mat.abs().max().item()) <= 1e-12:
        return torch.zeros((int(mat.shape[0]), 0), dtype=torch.float64)
    u_m, s_vals, _ = torch.linalg.svd(mat, full_matrices=False)
    top_s = float(s_vals[0].item())
    if top_s <= 1e-12:
        return torch.zeros((int(mat.shape[0]), 0), dtype=torch.float64)
    kept = s_vals > (1e-8 * top_s)
    return u_m[:, kept].contiguous()


def _build_sparse_nullspace_projector(
    q_prose: torch.Tensor,
    indices: torch.Tensor,
    nullspace_sv_tol: float,
) -> tuple[torch.Tensor, int]:
    """Compute exact projector onto ``ker(Q_prose[indices, indices])`` in ``R^len(indices)``."""
    q_sub = q_prose.index_select(0, indices).index_select(1, indices)
    q_sub = 0.5 * (q_sub + q_sub.T)
    evals, evecs = torch.linalg.eigh(q_sub)
    tol = max(float(nullspace_sv_tol), 1e-8 * max(1.0, float(evals.max().item())))
    null_mask = evals <= tol
    null_dim = int(null_mask.sum().item())
    if null_dim == 0:
        return torch.zeros((int(indices.numel()), int(indices.numel())), dtype=torch.float64), 0
    v_null = evecs[:, null_mask]
    return (v_null @ v_null.T).contiguous(), null_dim


def _synthesize_single_layer_factorized_update(
    w64: torch.Tensor,
    attractor_rows: torch.Tensor,
    p_null_mat: torch.Tensor,
    proj_side: str,
    *,
    layer_budget: int,
    damp_beta: float,
    nullspace_eig_threshold: float,
    nullspace_sv_tol: float,
) -> dict[str, Any]:
    """Synthesize a budgeted rank-``k`` update ``U @ V^T`` inside ``range(P_0)`` for one matrix."""
    d_out, d_in = int(w64.shape[0]), int(w64.shape[1])
    k_attr, attr_dim = int(attractor_rows.shape[0]), int(attractor_rows.shape[1])
    if layer_budget <= 0 or damp_beta == 0.0 or k_attr == 0:
        return {
            "delta_w": torch.zeros((d_out, d_in), dtype=torch.float64),
            "factor_u": torch.zeros((d_out, 0), dtype=torch.float64),
            "factor_v": torch.zeros((d_in, 0), dtype=torch.float64),
            "edited_elements": 0,
            "factorized_param_count": 0,
            "effective_rank": 0,
            "nullspace_leakage_norm": 0.0,
        }

    if attr_dim == d_out:
        attr_side = "output"
    elif attr_dim == d_in:
        attr_side = "input"
    else:
        raise ValueError(
            f"attractor_dirs width {attr_dim} matches neither d_out={d_out} nor d_in={d_in}"
        )

    # Symmetrize projector P_0 and construct orthogonal prose complement Q_prose = I - P_0
    p_evals, p_evecs = torch.linalg.eigh(0.5 * (p_null_mat + p_null_mat.T))
    keep_null = p_evals > float(nullspace_eig_threshold)
    if not bool(keep_null.any().item()):
        return {
            "delta_w": torch.zeros((d_out, d_in), dtype=torch.float64),
            "factor_u": torch.zeros((d_out, 0), dtype=torch.float64),
            "factor_v": torch.zeros((d_in, 0), dtype=torch.float64),
            "edited_elements": 0,
            "factorized_param_count": 0,
            "effective_rank": 0,
            "nullspace_leakage_norm": 0.0,
        }

    v_p0 = p_evecs[:, keep_null]
    p0_clean = (v_p0 @ v_p0.T).contiguous()
    q_prose = torch.eye(int(p0_clean.shape[0]), dtype=torch.float64) - p0_clean
    q_prose = 0.5 * (q_prose + q_prose.T)

    # Priority scores per row and per column combining attractor alignment and nullspace survival
    if attr_side == "output":
        u_base = attractor_rows.T.contiguous()  # (d_out, k_attr)
        w_eff = (w64 @ p0_clean) if proj_side == "input" else (p0_clean @ w64)
        row_raw = u_base.norm(dim=1) * w_eff.norm(dim=1).clamp_min(1e-8)
        col_proj = (u_base.T @ w64).norm(dim=0)
    else:
        v_base = attractor_rows.T.contiguous()  # (d_in, k_attr)
        w_eff = (w64 @ p0_clean) if proj_side == "input" else (p0_clean @ w64)
        row_raw = (w_eff @ v_base).norm(dim=1)
        col_proj = v_base.norm(dim=1) * w_eff.norm(dim=0).clamp_min(1e-8)

    if proj_side == "input":
        p_diag = torch.diagonal(p0_clean).clamp_min(0.0)
        col_scores = col_proj * p_diag.pow(2) + 1e-12 * p_diag
        row_scores = row_raw
    else:
        p_diag = torch.diagonal(p0_clean).clamp_min(0.0)
        row_scores = row_raw * p_diag.pow(2) + 1e-12 * p_diag
        col_scores = col_proj

    row_order = torch.argsort(row_scores, descending=True)
    col_order = torch.argsort(col_scores, descending=True)

    sig_rows = int((row_scores > 0.1 * max(float(row_scores.max().item()), 1e-12)).sum().item())
    sig_cols = int((col_scores > 0.1 * max(float(col_scores.max().item()), 1e-12)).sum().item())

    cand_r_set = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, d_out, sig_rows}
    if proj_side == "output":
        # Allow larger row support when output null-space needs > r_prose rows
        prose_rank = int(p0_clean.shape[0]) - int(keep_null.sum().item())
        cand_r_set.update({prose_rank + 1, prose_rank + 4, prose_rank + 8})
    cand_rs = sorted(r for r in cand_r_set if 1 <= r <= min(d_out, layer_budget))

    best_score = -1.0
    best_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    k_choices = [k_attr, 1] if k_attr > 1 else [1]
    for r_s in cand_rs:
        rows_idx = torch.sort(row_order[:r_s]).values
        max_c_mat = layer_budget // r_s
        if max_c_mat < 1:
            continue
        for k_use in k_choices:
            if k_use > r_s and attr_side == "output":
                continue
            max_c_fac = (layer_budget // k_use) - r_s
            c_s = min(d_in, max_c_mat, max_c_fac)
            if c_s < 1:
                continue
            if proj_side == "input" and sig_cols > 0 and sig_cols < c_s:
                # Also test tight column cap matching active attractor columns if smaller
                c_opts = [c_s]
                if sig_cols * r_s <= layer_budget and k_use * (r_s + sig_cols) <= layer_budget:
                    c_opts.append(sig_cols)
            else:
                c_opts = [c_s]

            for c_try in c_opts:
                cols_idx = torch.sort(col_order[:c_try]).values
                w_sub = w64.index_select(0, rows_idx).index_select(1, cols_idx)

                if proj_side == "input":
                    p_sub, sub_null_dim = _build_sparse_nullspace_projector(
                        q_prose, cols_idx, nullspace_sv_tol
                    )
                    if sub_null_dim == 0:
                        continue
                    if attr_side == "output":
                        u_q = _orthonormalize_columns(u_base[rows_idx, :k_use])
                        if u_q.shape[1] == 0:
                            continue
                        v_c = -float(damp_beta) * (p_sub @ (w_sub.T @ u_q))
                    else:
                        v_q = _orthonormalize_columns(p_sub @ v_base[cols_idx, :k_use])
                        if v_q.shape[1] == 0:
                            continue
                        u_r = -float(damp_beta) * (w_sub @ v_q)
                        u_q, v_c = u_r, v_q
                else:
                    p_sub, sub_null_dim = _build_sparse_nullspace_projector(
                        q_prose, rows_idx, nullspace_sv_tol
                    )
                    if sub_null_dim == 0:
                        continue
                    if attr_side == "output":
                        u_q = _orthonormalize_columns(p_sub @ u_base[rows_idx, :k_use])
                        if u_q.shape[1] == 0:
                            continue
                        v_c = -float(damp_beta) * (w_sub.T @ u_q)
                    else:
                        v_c = _orthonormalize_columns(v_base[cols_idx, :k_use])
                        if v_c.shape[1] == 0:
                            continue
                        u_q = -float(damp_beta) * (p_sub @ (w_sub @ v_c))

                k_eff = int(u_q.shape[1])
                if k_eff == 0 or k_eff * (r_s + c_try) > layer_budget:
                    continue
                delta_rc = u_q @ v_c.T
                energy = float((delta_rc * delta_rc).sum().item())
                if energy > 1e-24 and energy > best_score:
                    best_score = energy
                    best_tuple = (rows_idx, cols_idx, u_q, v_c, delta_rc)

    if best_tuple is None:
        return {
            "delta_w": torch.zeros((d_out, d_in), dtype=torch.float64),
            "factor_u": torch.zeros((d_out, 0), dtype=torch.float64),
            "factor_v": torch.zeros((d_in, 0), dtype=torch.float64),
            "edited_elements": 0,
            "factorized_param_count": 0,
            "effective_rank": 0,
            "nullspace_leakage_norm": 0.0,
        }

    rows_idx, cols_idx, u_q, v_c, delta_rc = best_tuple
    k_eff = int(u_q.shape[1])
    factor_u = torch.zeros((d_out, k_eff), dtype=torch.float64)
    factor_v = torch.zeros((d_in, k_eff), dtype=torch.float64)
    factor_u[rows_idx] = u_q
    factor_v[cols_idx] = v_c

    delta_w = torch.zeros((d_out, d_in), dtype=torch.float64)
    delta_w[rows_idx.unsqueeze(1), cols_idx.unsqueeze(0)] = delta_rc

    delta_norm = float(torch.linalg.matrix_norm(delta_w, ord="fro").item())
    if delta_norm > 1e-15:
        leak_mat = (delta_w @ q_prose) if proj_side == "input" else (q_prose @ delta_w)
        leak_norm = float(torch.linalg.matrix_norm(leak_mat, ord="fro").item()) / delta_norm
    else:
        leak_norm = 0.0

    edited_elements = int((delta_w.abs() > 0.0).sum().item())
    factorized_count = int((factor_u.abs() > 0.0).sum().item() + (factor_v.abs() > 0.0).sum().item())

    return {
        "delta_w": delta_w.contiguous(),
        "factor_u": factor_u.contiguous(),
        "factor_v": factor_v.contiguous(),
        "edited_elements": edited_elements,
        "factorized_param_count": factorized_count,
        "effective_rank": k_eff,
        "nullspace_leakage_norm": leak_norm,
    }


@torch.no_grad()
def synthesize_sub002pct_antiloop_tail_surgery(
    tail_down_weights: torch.Tensor | nn.Module | Sequence[torch.Tensor] | Mapping[str, torch.Tensor],
    final_norm_weight: torch.Tensor | nn.Module | None,
    attractor_dirs: torch.Tensor | Mapping[str, Any],
    p_null: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    damp_beta: float = DEFAULT_DAMP_BETA,
    tau_temp: float = DEFAULT_TAU_TEMP,
    total_model_params: int | None = None,
    max_budget_fraction: float = MAX_BUDGET_FRACTION_CEILING,
    final_norm_bias: torch.Tensor | None = None,
    projection_side: str = "auto",
    nullspace_eig_threshold: float = 0.5,
    nullspace_sv_tol: float = DEFAULT_NULLSPACE_SV_TOL,
) -> dict[str, Any]:
    """Synthesize strictly factorized rank-``k`` tail-layer null-space updates and norm scaling."""
    kind, names, raw_weights = _resolve_tail_weights(tail_down_weights)
    norm_w, norm_b = _resolve_norm_params(final_norm_weight, final_norm_bias)
    attr_rows, attr_ok = _resolve_attractor_matrix(attractor_dirs)

    beta_f = _validate_nonneg_finite(damp_beta, "damp_beta")
    tau_f = _validate_positive_finite(tau_temp, "tau_temp")
    max_frac = _validate_max_budget_fraction(max_budget_fraction)
    eig_thr = _validate_unit_interval(nullspace_eig_threshold, "nullspace_eig_threshold", allow_zero=True)
    if eig_thr >= 1.0:
        raise ValueError(f"nullspace_eig_threshold must be < 1.0, got {eig_thr}")
    sv_tol = _validate_nonneg_finite(nullspace_sv_tol, "nullspace_sv_tol")

    # Validate projectors up front even if attractor gate turns into a no-op
    resolved_projectors = [
        _resolve_layer_projector(name, int(w.shape[0]), int(w.shape[1]), p_null, projection_side)
        for name, w in zip(names, raw_weights, strict=True)
    ]

    local_params = sum(int(w.numel()) for w in raw_weights)
    if norm_w is not None:
        local_params += int(norm_w.numel())
    if norm_b is not None:
        local_params += int(norm_b.numel())

    denom = (
        _validate_positive_int(total_model_params, "total_model_params")
        if total_model_params is not None
        else local_params
    )
    max_budget_params = int(math.floor(max_frac * float(denom)))

    # Reserve norm-folding parameter cost first when tau_temp != 1.0 and budget allows
    norm_edits = 0
    tau_applied = 1.0
    norm_skipped_reason: str | None = None
    if norm_w is not None:
        norm_cost = int(norm_w.numel()) + (int(norm_b.numel()) if norm_b is not None else 0)
        if abs(tau_f - 1.0) > 1e-12:
            if norm_cost <= max_budget_params:
                norm_w_new, norm_b_new = fold_temperature_into_norm_params(norm_w, norm_b, tau_f)
                norm_edits = int((norm_w_new != norm_w).sum().item())
                if norm_b is not None and norm_b_new is not None:
                    norm_edits += int((norm_b_new != norm_b).sum().item())
                tau_applied = tau_f
            else:
                norm_w_new = norm_w.detach().clone()
                norm_b_new = norm_b.detach().clone() if norm_b is not None else None
                norm_skipped_reason = "norm_fold_exceeds_budget"
        else:
            norm_w_new = norm_w.detach().clone()
            norm_b_new = norm_b.detach().clone() if norm_b is not None else None
        norm_delta = norm_w_new - norm_w.detach()
    else:
        norm_w_new, norm_b_new, norm_delta = None, None, None

    rem_budget = max(0, max_budget_params - norm_edits)
    n_layers = len(raw_weights)
    layer_deltas: list[torch.Tensor] = []
    layer_edited_weights: list[torch.Tensor] = []
    factors_u: list[torch.Tensor] = []
    factors_v: list[torch.Tensor] = []
    total_w_edits = 0
    total_fac_params = 0
    max_leak = 0.0
    max_rank = 0

    if not attr_ok or beta_f == 0.0 or rem_budget == 0:
        for w in raw_weights:
            layer_deltas.append(torch.zeros_like(w))
            layer_edited_weights.append(w.detach().clone())
            factors_u.append(torch.zeros((int(w.shape[0]), 0), dtype=torch.float64))
            factors_v.append(torch.zeros((int(w.shape[1]), 0), dtype=torch.float64))
        noop_reason = (
            "empty_attractor_dirs_or_oof_rejected"
            if not attr_ok
            else ("zero_damp_beta" if beta_f == 0.0 else "budget_exhausted")
        )
    else:
        noop_reason = None
        for idx, (w, (p_mat, p_side)) in enumerate(zip(raw_weights, resolved_projectors, strict=True)):
            layers_left = n_layers - idx
            per_layer_cap = rem_budget // layers_left if layers_left > 1 else rem_budget
            w64 = w.detach().to(device="cpu", dtype=torch.float64)
            layer_res = _synthesize_single_layer_factorized_update(
                w64,
                attr_rows,
                p_mat,
                p_side,
                layer_budget=per_layer_cap,
                damp_beta=beta_f,
                nullspace_eig_threshold=eig_thr,
                nullspace_sv_tol=sv_tol,
            )
            delta_t = layer_res["delta_w"].to(device=w.device, dtype=w.dtype)
            w_new = (w.detach() + delta_t).contiguous()
            actual_delta = w_new - w.detach()
            actual_edits = int((actual_delta != 0).sum().item())
            fac_count = int(layer_res["factorized_param_count"])
            rem_budget = max(0, rem_budget - max(actual_edits, fac_count))
            total_w_edits += actual_edits
            total_fac_params += fac_count
            max_leak = max(max_leak, float(layer_res["nullspace_leakage_norm"]))
            max_rank = max(max_rank, int(layer_res["effective_rank"]))
            layer_deltas.append(actual_delta)
            layer_edited_weights.append(w_new)
            factors_u.append(layer_res["factor_u"])
            factors_v.append(layer_res["factor_v"])

        if total_w_edits == 0:
            noop_reason = "no_nullspace_within_budget"

    total_edits = total_w_edits + norm_edits
    total_fac = total_fac_params + norm_edits
    edited_frac = float(total_edits) / float(denom)
    fac_frac = float(total_fac) / float(denom)
    budget_ok = (edited_frac <= max_frac + 1e-15) and (fac_frac <= max_frac + 1e-15)

    def _pack(items: list[torch.Tensor]) -> Any:
        if kind in ("tensor", "module"):
            return items[0]
        if kind == "mapping":
            return dict(zip(names, items, strict=True))
        return items

    return {
        "tail_deltas": _pack(layer_deltas),
        "tail_weights_edited": _pack(layer_edited_weights),
        "factor_u": _pack(factors_u),
        "factor_v": _pack(factors_v),
        "final_norm_weight": norm_w_new,
        "final_norm_bias": norm_b_new,
        "final_norm_delta": norm_delta,
        "tau_temp_requested": tau_f,
        "tau_temp_applied": tau_applied,
        "norm_skipped_reason": norm_skipped_reason,
        "damp_beta": beta_f,
        "total_model_params": denom,
        "max_budget_fraction": max_frac,
        "max_budget_params": max_budget_params,
        "edited_elements": total_edits,
        "factorized_param_count": total_fac,
        "edited_param_fraction": edited_frac,
        "factorized_param_fraction": fac_frac,
        "budget_respected": budget_ok,
        "nullspace_leakage_norm": max_leak,
        "effective_rank": max_rank,
        "noop_reason": noop_reason,
    }


# ---------------------------------------------------------------------------
# 4. Verified End-to-End ANSE Anti-Loop Surgery with Atomic Rollback
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AntiLoopWSPSurgeryConfig:
    """Validated configuration governing sub-0.02% thinking-loop WSP surgery."""

    ngram_order: int = DEFAULT_NGRAM_ORDER
    min_oof_snr: float = DEFAULT_MIN_OOF_SNR
    max_attractor_rank: int = DEFAULT_MAX_ATTRACTOR_RANK
    damp_beta: float = DEFAULT_DAMP_BETA
    tau_temp: float = DEFAULT_TAU_TEMP
    max_budget_fraction: float = MAX_BUDGET_FRACTION_CEILING
    tail_layer_count: int = 1
    domain_guard_max_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    general_guard_max_ratio: float = DEFAULT_GENERAL_GUARD_MAX_RATIO
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    loop_energy_weight: float = DEFAULT_LOOP_ENERGY_WEIGHT
    prose_sv_rel_tol: float = DEFAULT_PROSE_SV_REL_TOL
    prose_variance_threshold: float | None = DEFAULT_PROSE_VARIANCE_THRESHOLD
    min_loop_reduction_rel: float = 0.0
    require_strict_energy_drop: bool = True
    general_prefix: str = GENERAL_EXPERT_PREFIX
    reasoning_prefix: str = REASONING_EXPERT_PREFIX
    max_new_tokens: int = 20
    score_from: int = 0
    batch_size: int = 16
    inplace: bool = True

    def __post_init__(self) -> None:
        _validate_positive_int(self.ngram_order, "ngram_order")
        _validate_nonneg_finite(self.min_oof_snr, "min_oof_snr")
        _validate_positive_int(self.max_attractor_rank, "max_attractor_rank")
        _validate_nonneg_finite(self.damp_beta, "damp_beta")
        _validate_positive_finite(self.tau_temp, "tau_temp")
        _validate_max_budget_fraction(self.max_budget_fraction)
        _validate_positive_int(self.tail_layer_count, "tail_layer_count")
        dom_r = _validate_positive_finite(self.domain_guard_max_ratio, "domain_guard_max_ratio")
        gen_r = _validate_positive_finite(self.general_guard_max_ratio, "general_guard_max_ratio")
        if dom_r < 1.0 or gen_r < 1.0:
            raise ValueError("domain_guard_max_ratio and general_guard_max_ratio must be >= 1.0")
        safe_r = _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        if safe_r < 1.0:
            raise ValueError("anse_safe_general_ratio must be >= 1.0")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_nonneg_finite(self.loop_energy_weight, "loop_energy_weight")
        _validate_nonneg_finite(self.prose_sv_rel_tol, "prose_sv_rel_tol")
        if self.prose_variance_threshold is not None:
            _validate_unit_interval(
                self.prose_variance_threshold, "prose_variance_threshold", allow_zero=False
            )
        _validate_unit_interval(self.min_loop_reduction_rel, "min_loop_reduction_rel")
        _validate_bool(self.require_strict_energy_drop, "require_strict_energy_drop")
        _validate_nonempty_str(self.general_prefix, "general_prefix")
        _validate_nonempty_str(self.reasoning_prefix, "reasoning_prefix")
        _validate_positive_int(self.max_new_tokens, "max_new_tokens")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        _validate_bool(self.inplace, "inplace")


def _resolve_surgery_config(
    config: AntiLoopWSPSurgeryConfig | None,
    *,
    damp_beta: float | None,
    tau_temp: float | None,
    max_budget_fraction: float | None,
    min_oof_snr: float | None,
    ngram_order: int | None,
    max_new_tokens: int | None,
    score_from: int | None,
    inplace: bool | None,
) -> AntiLoopWSPSurgeryConfig:
    """Merge optional keyword overrides into a validated ``AntiLoopWSPSurgeryConfig``."""
    if config is not None and not isinstance(config, AntiLoopWSPSurgeryConfig):
        raise TypeError(
            f"config must be AntiLoopWSPSurgeryConfig or None, got {type(config).__name__}"
        )
    base = config or AntiLoopWSPSurgeryConfig()
    return AntiLoopWSPSurgeryConfig(
        ngram_order=base.ngram_order if ngram_order is None else ngram_order,
        min_oof_snr=base.min_oof_snr if min_oof_snr is None else min_oof_snr,
        max_attractor_rank=base.max_attractor_rank,
        damp_beta=base.damp_beta if damp_beta is None else damp_beta,
        tau_temp=base.tau_temp if tau_temp is None else tau_temp,
        max_budget_fraction=(
            base.max_budget_fraction if max_budget_fraction is None else max_budget_fraction
        ),
        tail_layer_count=base.tail_layer_count,
        domain_guard_max_ratio=base.domain_guard_max_ratio,
        general_guard_max_ratio=base.general_guard_max_ratio,
        anse_safe_general_ratio=base.anse_safe_general_ratio,
        maximum_pain_energy=base.maximum_pain_energy,
        soft_margin_penalty=base.soft_margin_penalty,
        loop_energy_weight=base.loop_energy_weight,
        prose_sv_rel_tol=base.prose_sv_rel_tol,
        prose_variance_threshold=base.prose_variance_threshold,
        min_loop_reduction_rel=base.min_loop_reduction_rel,
        require_strict_energy_drop=base.require_strict_energy_drop,
        general_prefix=base.general_prefix,
        reasoning_prefix=base.reasoning_prefix,
        max_new_tokens=base.max_new_tokens if max_new_tokens is None else max_new_tokens,
        score_from=base.score_from if score_from is None else score_from,
        batch_size=base.batch_size,
        inplace=base.inplace if inplace is None else inplace,
    )


def _get_model_blocks_and_norm(model: nn.Module) -> tuple[Sequence[nn.Module], nn.Module | None, nn.Module]:
    """Validate and retrieve transformer block list, optional final norm, and unembedding head."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    blocks = getattr(model, "blocks", getattr(model, "layers", None))
    if blocks is None or len(blocks) == 0:
        raise ValueError("model must expose a non-empty 'blocks' or 'layers' sequence")
    norm_mod = getattr(model, "ln", getattr(model, "norm", None))
    head_mod = getattr(model, "head", getattr(model, "lm_head", None))
    if not isinstance(head_mod, nn.Module) or not hasattr(head_mod, "weight"):
        raise ValueError("model must expose a 'head' or 'lm_head' module with a .weight tensor")
    return blocks, norm_mod if isinstance(norm_mod, nn.Module) else None, head_mod


def _generate_greedy_sequences(
    model: nn.Module,
    prompts: list[list[int]],
    max_new_tokens: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Deterministically generate ``max_new_tokens`` greedily for every prompt."""
    dev = next(model.parameters()).device
    generated_only: list[list[int]] = []
    full_sequences: list[list[int]] = []

    if hasattr(model, "generate_ids"):
        for prompt in prompts:
            gen_ids = [
                int(t)
                for t in model.generate_ids(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    repetition_penalty=1.0,
                    eos_token_ids=(),
                )
            ]
            generated_only.append(gen_ids)
            full_sequences.append(list(prompt) + gen_ids)
        return generated_only, full_sequences

    for prompt in prompts:
        cur = list(prompt)
        gen_ids = []
        for _ in range(max_new_tokens):
            inp = torch.tensor([cur], dtype=torch.long, device=dev)
            out = model(inp)
            logits = out[0] if isinstance(out, tuple) else out
            next_tok = int(torch.argmax(logits[0, -1, :].float()).item())
            gen_ids.append(next_tok)
            cur.append(next_tok)
        generated_only.append(gen_ids)
        full_sequences.append(cur)
    return generated_only, full_sequences


def _resolve_block_down_targets(
    blk: nn.Module,
    layer_idx: int,
) -> list[tuple[str, nn.Module, torch.Tensor]]:
    """Return candidate ``(target_label, down_module, weight_tensor)`` in block ``blk``."""
    mlp = getattr(blk, "mlp", None)
    if not isinstance(mlp, nn.Module):
        raise ValueError(f"Block {layer_idx} must expose an nn.Module '.mlp'")

    if isinstance(mlp, CrowdLoRAMoEMLP):
        targets: list[tuple[str, nn.Module, torch.Tensor]] = []
        for e_key in sorted(mlp.experts.keys(), key=int):
            exp = mlp.experts[e_key]
            if isinstance(exp, CrowdLoRAExpertAdapter) and float(exp.w3_B.detach().abs().max().item()) > 1e-12:
                targets.append((f"blocks.{layer_idx}.mlp.experts.{e_key}.w3_B", exp, exp.w3_B))
        targets.append((f"blocks.{layer_idx}.mlp.base_mlp.w3", mlp.base_mlp.w3, mlp.base_mlp.w3.weight))
        return targets

    if hasattr(mlp, "w3") and isinstance(mlp.w3, nn.Linear):
        return [(f"blocks.{layer_idx}.mlp.w3", mlp.w3, mlp.w3.weight)]
    if hasattr(mlp, "down_proj") and isinstance(mlp.down_proj, nn.Linear):
        return [(f"blocks.{layer_idx}.mlp.down_proj", mlp.down_proj, mlp.down_proj.weight)]
    raise ValueError(f"Block {layer_idx} MLP must expose .w3 or .down_proj")


def _compute_crowd_moe_expert_latent(
    moe_mlp: CrowdLoRAMoEMLP,
    x_2d: torch.Tensor,
    e_key: str,
) -> torch.Tensor:
    """Compute routing-weighted latent vector entering ``experts[e_key].w3_B``."""
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

    hit = (top_idx == int(e_key)).to(torch.float32)
    w_tok = (gates * hit).sum(dim=-1, keepdim=True)
    exp = moe_mlp.experts[e_key]
    dh1 = F.linear(F.linear(x_fp32, exp.w1_A), exp.w1_B) * exp.scaling
    dh2 = F.linear(F.linear(x_fp32, exp.w2_A), exp.w2_B) * exp.scaling
    act_full = F.silu(h1 + dh1) * (h2 + dh2)
    return F.linear(act_full, exp.w3_A) * (exp.scaling * w_tok)


def _extract_block_states_crowd_fast(
    model: nn.Module,
    ids_2d: torch.Tensor,
    target_layer: int,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run ``CrowdV1WithEngram`` forward pass up to ``target_layer`` returning block out & down inputs."""
    dev = next(model.parameters()).device
    cos = model.cos_cached.to(dev)
    sin = model.sin_cached.to(dev)
    x_parts = [
        compute_embeddings_batch(model, ids_2d[b0 : b0 + batch_size].to(dev))
        for b0 in range(0, ids_2d.shape[0], batch_size)
    ]
    x_seq = torch.cat(x_parts, dim=0)

    down_inputs: dict[str, torch.Tensor] = {}
    blk_out_res = x_seq
    for l_idx, blk in enumerate(model.blocks):
        xm_l, mi_l = [], []
        for b0 in range(0, x_seq.shape[0], batch_size):
            xm, mi = forward_block_pre_mlp(blk, x_seq[b0 : b0 + batch_size], cos, sin)
            xm_l.append(xm)
            mi_l.append(mi)
        x_mid = torch.cat(xm_l, dim=0)
        mlp_in = torch.cat(mi_l, dim=0)

        if l_idx == target_layer:
            b_sz, t_len, h_dim = mlp_in.shape
            mi_flat = mlp_in.reshape(-1, h_dim)
            mlp = blk.mlp
            if isinstance(mlp, CrowdLoRAMoEMLP):
                w_dt = mlp.base_mlp.w1.weight.dtype
                act_base = (F.silu(mlp.base_mlp.w1(mi_flat.to(w_dt))) * mlp.base_mlp.w2(mi_flat.to(w_dt))).to(torch.float32)
                down_inputs[f"blocks.{l_idx}.mlp.base_mlp.w3"] = act_base.reshape(b_sz, t_len, -1)
                for e_key in sorted(mlp.experts.keys(), key=int):
                    exp = mlp.experts[e_key]
                    if float(exp.w3_B.detach().abs().max().item()) > 1e-12:
                        z_e = _compute_crowd_moe_expert_latent(mlp, mi_flat, e_key)
                        down_inputs[f"blocks.{l_idx}.mlp.experts.{e_key}.w3_B"] = z_e.reshape(b_sz, t_len, -1)
            elif hasattr(mlp, "w1") and hasattr(mlp, "w2"):
                w_dt = mlp.w1.weight.dtype
                act = (F.silu(mlp.w1(mi_flat.to(w_dt))) * mlp.w2(mi_flat.to(w_dt))).to(torch.float32)
                down_inputs[f"blocks.{l_idx}.mlp.w3"] = act.reshape(b_sz, t_len, -1)
            elif hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
                w_dt = mlp.gate_proj.weight.dtype
                act = (F.silu(mlp.gate_proj(mi_flat.to(w_dt))) * mlp.up_proj(mi_flat.to(w_dt))).to(torch.float32)
                down_inputs[f"blocks.{l_idx}.mlp.down_proj"] = act.reshape(b_sz, t_len, -1)

        out_parts = [
            forward_block_post_mlp(blk, x_mid[b0 : b0 + batch_size], mlp_in[b0 : b0 + batch_size])
            for b0 in range(0, x_mid.shape[0], batch_size)
        ]
        x_seq = torch.cat(out_parts, dim=0)
        if l_idx == target_layer:
            blk_out_res = x_seq
            break

    return blk_out_res.to(device="cpu", dtype=torch.float64), {
        k: v.to(device="cpu", dtype=torch.float64) for k, v in down_inputs.items()
    }


def _extract_block_states_generic(
    model: nn.Module,
    ids_2d: torch.Tensor,
    target_layer: int,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Capture block output hidden states and down-projection inputs via forward hooks."""
    blocks, _, _ = _get_model_blocks_and_norm(model)
    blk = blocks[target_layer]
    targets = _resolve_block_down_targets(blk, target_layer)
    dev = next(model.parameters()).device

    captured_in: dict[str, list[torch.Tensor]] = {label: [] for label, _, _ in targets}
    captured_blk: list[torch.Tensor] = []
    handles: list[Any] = []

    def _make_down_hook(lbl: str) -> Callable[[nn.Module, tuple[torch.Tensor, ...], Any], None]:
        def _hook(_mod: nn.Module, inputs: tuple[torch.Tensor, ...], _out: Any) -> None:
            captured_in[lbl].append(inputs[0].detach().to(device="cpu", dtype=torch.float64))

        return _hook

    def _blk_hook(_mod: nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
        out_t = output[0] if isinstance(output, tuple) else output
        captured_blk.append(out_t.detach().to(device="cpu", dtype=torch.float64))

    try:
        for label, mod, _ in targets:
            if not isinstance(mod, CrowdLoRAExpertAdapter):
                handles.append(mod.register_forward_hook(_make_down_hook(label)))
        handles.append(blk.register_forward_hook(_blk_hook))
        for b0 in range(0, ids_2d.shape[0], batch_size):
            model(ids_2d[b0 : b0 + batch_size].to(dev))
    finally:
        for h in handles:
            h.remove()

    down_map = {k: torch.cat(v, dim=0) for k, v in captured_in.items() if v}
    return torch.cat(captured_blk, dim=0), down_map


def _collect_target_states(
    model: nn.Module,
    ids_2d: torch.Tensor,
    target_layer: int,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch state extraction across native ``CrowdV1WithEngram`` and generic blocks."""
    if hasattr(model, "cos_cached") and hasattr(model, "sin_cached") and hasattr(model, "wte"):
        return _extract_block_states_crowd_fast(model, ids_2d, target_layer, batch_size=batch_size)
    return _extract_block_states_generic(model, ids_2d, target_layer, batch_size=batch_size)


def _normalize_eval_chunks(
    chunks: Sequence[Mapping[str, Any]] | torch.Tensor | None,
    fallback_seqs: list[list[int]],
    default_prefix: str,
    vocab_size: int,
) -> tuple[torch.Tensor, list[str]]:
    """Convert chunk mappings or 2D tensor into aligned ``(ids_2d, domain_labels)``."""
    if chunks is None:
        seqs = _normalize_token_sequences(fallback_seqs, "fallback_seqs")
        max_len = min(len(s) for s in seqs)
        ids_t = torch.tensor([s[:max_len] for s in seqs], dtype=torch.long)
        return ids_t, [default_prefix] * int(ids_t.shape[0])

    if isinstance(chunks, torch.Tensor):
        seqs = _normalize_token_sequences(chunks, "eval_chunks")
        ids_t = torch.tensor(seqs, dtype=torch.long)
        if bool((ids_t >= vocab_size).any().item()):
            raise ValueError(f"eval_chunks token IDs must lie in [0, {vocab_size})")
        return ids_t, [default_prefix] * int(ids_t.shape[0])

    if isinstance(chunks, Sequence) and not isinstance(chunks, (str, bytes, bytearray)) and chunks:
        rows: list[torch.Tensor] = []
        domains: list[str] = []
        seq_len: int | None = None
        for idx, ch in enumerate(chunks):
            if not isinstance(ch, Mapping) or "expert_id" not in ch or "ids" not in ch:
                raise ValueError(f"eval_chunks[{idx}] must map 'expert_id' and 'ids'")
            eid = _validate_nonempty_str(ch["expert_id"], f"eval_chunks[{idx}]['expert_id']")
            row_ids = _normalize_token_sequences([ch["ids"]], f"eval_chunks[{idx}]['ids']")[0]
            if len(row_ids) < 2:
                raise ValueError(f"eval_chunks[{idx}]['ids'] must have length >= 2")
            if any(tok >= vocab_size for tok in row_ids):
                raise ValueError(f"eval_chunks[{idx}]['ids'] token IDs must lie in [0, {vocab_size})")
            if seq_len is None:
                seq_len = len(row_ids)
            elif len(row_ids) != seq_len:
                raise ValueError("All eval_chunks must share identical sequence length")
            rows.append(torch.tensor(row_ids, dtype=torch.long))
            domains.append(eid)
        return torch.stack(rows, dim=0), domains

    raise ValueError("eval_chunks must be a non-empty sequence of chunk dicts, 2D tensor, or None")


def _compute_causal_nlls_by_domain(
    model: nn.Module,
    ids_2d: torch.Tensor,
    domains: list[str],
    *,
    score_from: int,
    batch_size: int,
) -> dict[str, float]:
    """Compute exact per-domain causal cross-entropy NLLs without gradients."""
    dev = next(model.parameters()).device
    start_pos = max(0, min(int(score_from) - 1, int(ids_2d.shape[1]) - 2))
    losses: list[float] = []

    if hasattr(model, "cos_cached") and hasattr(model, "sin_cached") and hasattr(model, "ln"):
        pre_ln = forward_model_pre_ln(model, ids_2d.to(dev), batch_size=batch_size)
        ln_b = (
            model.ln.bias
            if getattr(model, "ln", None) is not None and getattr(model.ln, "bias", None) is not None
            else torch.zeros_like(model.ln.weight)
        )
        chunk_nlls = per_chunk_nll_from_pre_ln(
            pre_ln,
            model.ln.weight,
            ln_b,
            float(getattr(model.ln, "eps", 1e-5)),
            model.head.weight,
            ids_2d.to(dev),
            score_from=start_pos + 1,
        )
        losses = [float(v) for v in chunk_nlls.tolist()]
    else:
        for b0 in range(0, ids_2d.shape[0], batch_size):
            batch_ids = ids_2d[b0 : b0 + batch_size].to(dev)
            out = model(batch_ids)
            logits = out[0] if isinstance(out, tuple) else out
            sub_lg = logits[:, start_pos:-1, :].float()
            sub_tg = batch_ids[:, start_pos + 1 :]
            tok_loss = F.cross_entropy(
                sub_lg.reshape(-1, sub_lg.shape[-1]),
                sub_tg.reshape(-1),
                reduction="none",
            ).view(batch_ids.shape[0], -1)
            losses.extend(float(v) for v in tok_loss.mean(dim=1).tolist())

    if not all(math.isfinite(v) for v in losses):
        raise ValueError("Model produced non-finite NLL values")

    grouped: dict[str, list[float]] = {}
    for dom, val in zip(domains, losses, strict=True):
        grouped.setdefault(dom, []).append(val)
    return {k: sum(v) / len(v) for k, v in sorted(grouped.items())}


def _evaluate_custom_eval_fn(eval_fn: Callable[[nn.Module], Mapping[str, Any]], model: nn.Module) -> tuple[dict[str, float], float | None]:
    """Invoke and validate a user-supplied ``eval_fn(model)``."""
    raw = eval_fn(model)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("eval_fn(model) must return a non-empty mapping")
    if bool(raw.get("non_finite_outputs", False)):
        raise ValueError("eval_fn reported non_finite_outputs")
    raw_doms = raw.get("domain_nlls", raw)
    if not isinstance(raw_doms, Mapping) or not raw_doms:
        raise ValueError("eval_fn(model) domain_nlls must be a non-empty mapping")
    skip_keys = {"nll_mean", "repeat_4gram_ratio", "non_finite_outputs", "anse_energy"}
    dom_nlls: dict[str, float] = {}
    for k, v in raw_doms.items():
        if k in skip_keys:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ValueError(f"eval_fn returned non-finite NLL for {k!r}: {v!r}")
        dom_nlls[str(k)] = float(v)
    if not dom_nlls:
        raise ValueError("eval_fn(model) must contain at least one finite domain NLL")
    rep_override = raw.get("repeat_4gram_ratio")
    rep_val = float(rep_override) if rep_override is not None else None
    return dom_nlls, rep_val


def _compute_anse_verdict(
    parent_dom_nll: Mapping[str, float],
    cand_dom_nll: Mapping[str, float],
    rep_ratio: float,
    *,
    budget_ok: bool,
    cfg: AntiLoopWSPSurgeryConfig,
) -> dict[str, Any]:
    """Compute ANSE thermodynamic energy, general-prose drift, and worst-domain guard verdict."""
    missing = set(parent_dom_nll.keys()) - set(cand_dom_nll.keys())
    if missing:
        return {
            "guard_ok": False,
            "general_guard_ok": False,
            "general_ratio": float("inf"),
            "general_pct_vs_base": float("inf"),
            "worst_domain_pct_vs_base": float("inf"),
            "nll_mean": float("inf"),
            "anse_energy": cfg.maximum_pain_energy,
        }

    ratios: dict[str, float] = {}
    for dom in sorted(parent_dom_nll.keys()):
        p_v = float(parent_dom_nll[dom])
        c_v = float(cand_dom_nll[dom])
        ratios[dom] = c_v / max(p_v, 1e-12)

    gen_keys = [d for d in sorted(ratios.keys()) if d.startswith(cfg.general_prefix)]
    if not gen_keys:
        gen_keys = [sorted(ratios.keys())[0]]
    general_ratio = max(ratios[d] for d in gen_keys)
    worst_ratio = max(ratios.values())

    mean_parent = sum(float(parent_dom_nll[d]) for d in parent_dom_nll) / float(len(parent_dom_nll))
    mean_cand = sum(float(cand_dom_nll[d]) for d in parent_dom_nll) / float(len(parent_dom_nll))
    nll_diff = mean_cand - mean_parent

    general_guard_ok = general_ratio <= cfg.general_guard_max_ratio + 1e-12
    worst_guard_ok = worst_ratio <= cfg.domain_guard_max_ratio + 1e-12
    guard_ok = bool(general_guard_ok and worst_guard_ok and budget_ok)

    if not guard_ok:
        anse_energy = cfg.maximum_pain_energy + max(0.0, nll_diff) + cfg.loop_energy_weight * rep_ratio
    else:
        soft_pen = cfg.soft_margin_penalty if general_ratio > cfg.anse_safe_general_ratio else 0.0
        anse_energy = nll_diff + cfg.loop_energy_weight * rep_ratio + soft_pen

    return {
        "guard_ok": guard_ok,
        "general_guard_ok": general_guard_ok,
        "general_ratio": general_ratio,
        "general_pct_vs_base": 100.0 * (general_ratio - 1.0),
        "worst_domain_pct_vs_base": 100.0 * (worst_ratio - 1.0),
        "nll_mean": mean_cand,
        "anse_energy": float(anse_energy),
    }


@torch.no_grad()
def execute_verified_antiloop_wsp_surgery(
    model: nn.Module,
    loop_prompts: Sequence[Sequence[int]] | torch.Tensor,
    concise_prompts: Sequence[Sequence[int]] | torch.Tensor,
    *,
    prose_eval_chunks: Sequence[Mapping[str, Any]] | torch.Tensor | None = None,
    reasoning_eval_chunks: Sequence[Mapping[str, Any]] | torch.Tensor | None = None,
    eval_fn: Callable[[nn.Module], Mapping[str, Any]] | None = None,
    p_null: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    target_layer_indices: Sequence[int] | None = None,
    config: AntiLoopWSPSurgeryConfig | None = None,
    damp_beta: float | None = None,
    tau_temp: float | None = None,
    max_budget_fraction: float | None = None,
    min_oof_snr: float | None = None,
    ngram_order: int | None = None,
    total_model_params: int | None = None,
    max_new_tokens: int | None = None,
    score_from: int | None = None,
    inplace: bool | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Run verified sub-0.02% thinking-loop tail surgery with ANSE gating and atomic rollback."""
    cfg = _resolve_surgery_config(
        config,
        damp_beta=damp_beta,
        tau_temp=tau_temp,
        max_budget_fraction=max_budget_fraction,
        min_oof_snr=min_oof_snr,
        ngram_order=ngram_order,
        max_new_tokens=max_new_tokens,
        score_from=score_from,
        inplace=inplace,
    )
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError(f"eval_fn must be callable or None, got {type(eval_fn).__name__}")

    blocks, norm_mod, head_mod = _get_model_blocks_and_norm(model)
    n_blocks = len(blocks)
    vocab_size = int(head_mod.weight.shape[0])

    loop_prompt_seqs = _normalize_token_sequences(loop_prompts, "loop_prompts")
    concise_prompt_seqs = _normalize_token_sequences(concise_prompts, "concise_prompts")
    for seq in loop_prompt_seqs + concise_prompt_seqs:
        if any(tok >= vocab_size for tok in seq):
            raise ValueError(f"Prompt token IDs must lie in [0, {vocab_size})")

    if target_layer_indices is None:
        layer_list = list(range(max(0, n_blocks - cfg.tail_layer_count), n_blocks))
    else:
        if isinstance(target_layer_indices, (str, bytes)) or not isinstance(target_layer_indices, Sequence):
            raise TypeError("target_layer_indices must be a non-empty sequence of layer indices")
        if not target_layer_indices:
            raise ValueError("target_layer_indices must not be empty")
        layer_list = sorted({_validate_nonneg_int(idx, "target_layer_index") for idx in target_layer_indices})
        if any(idx >= n_blocks for idx in layer_list):
            raise ValueError(f"target_layer_indices must lie in [0, {n_blocks})")

    work_model = model if cfg.inplace else copy.deepcopy(model)
    work_blocks, work_norm, _ = _get_model_blocks_and_norm(work_model)
    total_params = (
        _validate_positive_int(total_model_params, "total_model_params")
        if total_model_params is not None
        else sum(p.numel() for p in work_model.parameters())
    )

    was_training = work_model.training
    work_model.eval()
    try:
        parent_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in work_model.state_dict().items()}
        parent_sha = compute_state_sha256(work_model)

        loop_gen_before, loop_full_before = _generate_greedy_sequences(
            work_model, loop_prompt_seqs, cfg.max_new_tokens
        )
        concise_gen_before, concise_full_before = _generate_greedy_sequences(
            work_model, concise_prompt_seqs, cfg.max_new_tokens
        )

        # Pad/trim to common length for batched tensor forward passes
        min_l_len = min(len(s) for s in loop_full_before)
        min_c_len = min(len(s) for s in concise_full_before)
        loop_ids_before = torch.tensor([s[:min_l_len] for s in loop_full_before], dtype=torch.long)
        concise_ids_before = torch.tensor([s[:min_c_len] for s in concise_full_before], dtype=torch.long)

        loop_rep_stats_before = compute_ngram_repetition_metrics(
            loop_gen_before, ngram_order=cfg.ngram_order
        )
        rep_before = float(loop_rep_stats_before["repeat_4gram_ratio"])

        if eval_fn is not None:
            parent_dom_nll, rep_fn_before = _evaluate_custom_eval_fn(eval_fn, work_model)
            if rep_fn_before is not None:
                rep_before = rep_fn_before
            eval_mode = "eval_fn"
            prose_ids, prose_doms = None, []
            reason_ids, reason_doms = None, []
        else:
            eval_mode = "chunks"
            prose_ids, prose_doms = _normalize_eval_chunks(
                prose_eval_chunks, concise_prompt_seqs, f"{cfg.general_prefix}prose", vocab_size
            )
            reason_ids, reason_doms = _normalize_eval_chunks(
                reasoning_eval_chunks, loop_prompt_seqs, f"{cfg.reasoning_prefix}reasoning", vocab_size
            )
            parent_prose_nll = _compute_causal_nlls_by_domain(
                work_model, prose_ids, prose_doms, score_from=cfg.score_from, batch_size=cfg.batch_size
            )
            parent_reason_nll = _compute_causal_nlls_by_domain(
                work_model, reason_ids, reason_doms, score_from=cfg.score_from, batch_size=cfg.batch_size
            )
            parent_dom_nll = {**parent_prose_nll, **parent_reason_nll}

        gen_keys = [d for d in sorted(parent_dom_nll.keys()) if d.startswith(cfg.general_prefix)]
        reason_keys = [d for d in sorted(parent_dom_nll.keys()) if not d.startswith(cfg.general_prefix)]
        prose_nll_before = (
            sum(parent_dom_nll[d] for d in gen_keys) / float(len(gen_keys))
            if gen_keys
            else sum(parent_dom_nll.values()) / float(len(parent_dom_nll))
        )
        reason_nll_before = (
            sum(parent_dom_nll[d] for d in reason_keys) / float(len(reason_keys))
            if reason_keys
            else prose_nll_before
        )
        parent_nll_mean = sum(parent_dom_nll.values()) / float(len(parent_dom_nll))
        parent_anse_energy = cfg.loop_energy_weight * rep_before

        # Collect hidden states and diagnose OOF attractor on deepest target layer
        deepest_layer = layer_list[-1]
        h_loop, down_loop_map = _collect_target_states(
            work_model, loop_ids_before, deepest_layer, batch_size=cfg.batch_size
        )
        h_concise, down_concise_map = _collect_target_states(
            work_model, concise_ids_before, deepest_layer, batch_size=cfg.batch_size
        )

        # Slice out prompt prefix if trajectories are long enough so loop generation dominates
        prompt_l = min(len(s) for s in loop_prompt_seqs)
        prompt_c = min(len(s) for s in concise_prompt_seqs)
        h_loop_diag = h_loop[:, max(0, prompt_l - 1) :, :] if int(h_loop.shape[1]) > prompt_l + 2 else h_loop
        h_concise_diag = (
            h_concise[:, max(0, prompt_c - 1) :, :] if int(h_concise.shape[1]) > prompt_c + 2 else h_concise
        )

        diag = diagnose_repetition_attractor_subspace(
            h_loop_diag,
            h_concise_diag,
            loop_token_ids=loop_gen_before,
            concise_token_ids=concise_gen_before,
            ngram_order=cfg.ngram_order,
            min_oof_snr=cfg.min_oof_snr,
            max_attractor_rank=cfg.max_attractor_rank,
        )

        # Also inspect expert w3_B candidates in CrowdLoRAMoEMLP when compatible with p_null
        down_targets = _resolve_block_down_targets(work_blocks[deepest_layer], deepest_layer)
        selected_label, _, selected_weight = down_targets[-1]
        if len(down_targets) > 1:
            best_snr = float(diag["oof_snr"])
            for lbl, _, w_t in down_targets:
                if p_null is not None:
                    try:
                        _resolve_layer_projector(lbl, int(w_t.shape[0]), int(w_t.shape[1]), p_null, "auto")
                    except ValueError:
                        continue
                if lbl in down_loop_map and lbl in down_concise_map:
                    l_d = down_loop_map[lbl][:, max(0, prompt_l - 1) :, :] if int(down_loop_map[lbl].shape[1]) > prompt_l + 2 else down_loop_map[lbl]
                    c_d = down_concise_map[lbl][:, max(0, prompt_c - 1) :, :] if int(down_concise_map[lbl].shape[1]) > prompt_c + 2 else down_concise_map[lbl]
                    cand_diag = diagnose_repetition_attractor_subspace(
                        l_d, c_d, loop_token_ids=loop_gen_before, concise_token_ids=concise_gen_before,
                        ngram_order=cfg.ngram_order, min_oof_snr=cfg.min_oof_snr, max_attractor_rank=cfg.max_attractor_rank,
                    )
                    if cand_diag["passes_oof_gate"] and float(cand_diag["oof_snr"]) > best_snr and float(w_t.detach().abs().max().item()) > 1e-12:
                        best_snr = float(cand_diag["oof_snr"])
                        selected_label = lbl
                        selected_weight = w_t
                        diag = cand_diag

        if not diag["passes_oof_gate"]:
            receipt = _assemble_receipt(
                accepted=False,
                rolled_back=False,
                status="NOOP_OOF_SNR_REJECTED",
                nll_mean=parent_nll_mean,
                parent_nll_mean=parent_nll_mean,
                cand_nll_mean=parent_nll_mean,
                prose_nll_before=prose_nll_before,
                prose_nll_after=prose_nll_before,
                reason_nll_before=reason_nll_before,
                reason_nll_after=reason_nll_before,
                rep_before=rep_before,
                rep_after=rep_before,
                edited_elements=0,
                factorized_param_count=0,
                total_model_params=total_params,
                budget_respected=True,
                oof_snr=float(diag["oof_snr"]),
                passes_oof_gate=False,
                nullspace_leakage_norm=0.0,
                anse_energy=parent_anse_energy,
                parent_anse_energy=parent_anse_energy,
                cand_anse_energy=parent_anse_energy,
                delta_e=0.0,
                guard_ok=True,
                general_guard_ok=True,
                general_pct_vs_base=0.0,
                worst_domain_pct_vs_base=0.0,
                target_label=selected_label,
                parent_sha=parent_sha,
                cand_sha=parent_sha,
                final_sha=parent_sha,
                cfg=cfg,
            )
            return work_model, receipt

        # Build exact AlphaEdit prose null-space P_0 if not passed explicitly
        prose_p0_ids = concise_ids_before
        if p_null is None:
            if prose_ids is not None:
                e0_rows = [i for i, dom in enumerate(prose_doms) if dom.startswith(cfg.general_prefix)]
                prose_p0_ids = prose_ids[e0_rows] if e0_rows else prose_ids
                _, prose_down_map = _collect_target_states(
                    work_model, prose_p0_ids, deepest_layer, batch_size=cfg.batch_size
                )
                prose_act = prose_down_map.get(selected_label, down_concise_map[selected_label])
            else:
                prose_act = down_concise_map[selected_label]
            p_null_resolved, _ = compute_prose_nullspace_projector_scaled(
                prose_act,
                sv_rel_tol=cfg.prose_sv_rel_tol,
                variance_threshold=cfg.prose_variance_threshold,
            )
        else:
            p_null_resolved = p_null

        # If multiple tail layers requested, synthesize jointly across compatible target layers within shared budget
        if len(layer_list) > 1 and "." in selected_label:
            tail_suffix = selected_label.split(".", 2)[-1]
            attr_dim = int(diag["attractor_dirs"].shape[1])
            tail_weights_map: dict[str, torch.Tensor] = {}
            tail_pnull_map: dict[str, torch.Tensor] = {}
            for l_idx in layer_list:
                l_targets = _resolve_block_down_targets(work_blocks[l_idx], l_idx)
                matched = [
                    (lbl, w_t)
                    for lbl, _, w_t in l_targets
                    if lbl.endswith(tail_suffix) and (int(w_t.shape[0]) == attr_dim or int(w_t.shape[1]) == attr_dim)
                ]
                if not matched:
                    matched = [
                        (lbl, w_t)
                        for lbl, _, w_t in l_targets
                        if int(w_t.shape[0]) == attr_dim or int(w_t.shape[1]) == attr_dim
                    ]
                if not matched:
                    continue
                lbl_i, w_i = matched[0]
                tail_weights_map[lbl_i] = w_i
                if p_null is None:
                    p_ids_i = prose_p0_ids if prose_ids is not None else concise_ids_before
                    _, p_down_i = _collect_target_states(work_model, p_ids_i, l_idx, batch_size=cfg.batch_size)
                    p_act_i = p_down_i.get(lbl_i)
                    if p_act_i is not None:
                        tail_pnull_map[lbl_i], _ = compute_prose_nullspace_projector_scaled(
                            p_act_i,
                            sv_rel_tol=cfg.prose_sv_rel_tol,
                            variance_threshold=cfg.prose_variance_threshold,
                        )
                    else:
                        tail_pnull_map[lbl_i] = p_null_resolved
                elif isinstance(p_null_resolved, Mapping):
                    tail_pnull_map[lbl_i], _ = _resolve_layer_projector(
                        lbl_i, int(w_i.shape[0]), int(w_i.shape[1]), p_null_resolved, "auto"
                    )
                else:
                    tail_pnull_map[lbl_i] = p_null_resolved
            synth_target: Any = tail_weights_map
            synth_pnull: Any = tail_pnull_map
        else:
            synth_target = selected_weight
            synth_pnull = p_null_resolved

        norm_w_t = work_norm.weight if work_norm is not None else None
        norm_b_t = getattr(work_norm, "bias", None) if work_norm is not None else None
        synth = synthesize_sub002pct_antiloop_tail_surgery(
            synth_target,
            norm_w_t,
            diag,
            synth_pnull,
            damp_beta=cfg.damp_beta,
            tau_temp=cfg.tau_temp,
            total_model_params=total_params,
            max_budget_fraction=cfg.max_budget_fraction,
            final_norm_bias=norm_b_t,
        )

        if synth["edited_elements"] == 0:
            receipt = _assemble_receipt(
                accepted=False,
                rolled_back=False,
                status="NOOP_BUDGET_OR_NULLSPACE_EMPTY",
                nll_mean=parent_nll_mean,
                parent_nll_mean=parent_nll_mean,
                cand_nll_mean=parent_nll_mean,
                prose_nll_before=prose_nll_before,
                prose_nll_after=prose_nll_before,
                reason_nll_before=reason_nll_before,
                reason_nll_after=reason_nll_before,
                rep_before=rep_before,
                rep_after=rep_before,
                edited_elements=0,
                factorized_param_count=0,
                total_model_params=total_params,
                budget_respected=bool(synth["budget_respected"]),
                oof_snr=float(diag["oof_snr"]),
                passes_oof_gate=True,
                nullspace_leakage_norm=float(synth["nullspace_leakage_norm"]),
                anse_energy=parent_anse_energy,
                parent_anse_energy=parent_anse_energy,
                cand_anse_energy=parent_anse_energy,
                delta_e=0.0,
                guard_ok=True,
                general_guard_ok=True,
                general_pct_vs_base=0.0,
                worst_domain_pct_vs_base=0.0,
                target_label=selected_label,
                parent_sha=parent_sha,
                cand_sha=parent_sha,
                final_sha=parent_sha,
                cfg=cfg,
            )
            return work_model, receipt

        # Apply candidate mutation in-place
        if isinstance(synth["tail_weights_edited"], Mapping):
            for lbl_k, edited_t in synth["tail_weights_edited"].items():
                tail_weights_map[lbl_k].copy_(
                    edited_t.to(device=tail_weights_map[lbl_k].device, dtype=tail_weights_map[lbl_k].dtype)
                )
        else:
            selected_weight.copy_(
                synth["tail_weights_edited"].to(device=selected_weight.device, dtype=selected_weight.dtype)
            )
        if work_norm is not None and synth["final_norm_weight"] is not None:
            work_norm.weight.copy_(synth["final_norm_weight"].to(device=work_norm.weight.device, dtype=work_norm.weight.dtype))
            if norm_b_t is not None and synth["final_norm_bias"] is not None:
                norm_b_t.copy_(synth["final_norm_bias"].to(device=norm_b_t.device, dtype=norm_b_t.dtype))

        cand_sha = compute_state_sha256(work_model)

        # Post-surgery evaluation
        loop_gen_after, _ = _generate_greedy_sequences(work_model, loop_prompt_seqs, cfg.max_new_tokens)
        loop_rep_stats_after = compute_ngram_repetition_metrics(
            loop_gen_after, ngram_order=cfg.ngram_order
        )
        rep_after = float(loop_rep_stats_after["repeat_4gram_ratio"])

        non_finite_error = False
        try:
            if eval_mode == "eval_fn":
                assert eval_fn is not None
                cand_dom_nll, rep_fn_after = _evaluate_custom_eval_fn(eval_fn, work_model)
                if rep_fn_after is not None:
                    rep_after = rep_fn_after
            else:
                assert prose_ids is not None and reason_ids is not None
                cand_prose_nll = _compute_causal_nlls_by_domain(
                    work_model, prose_ids, prose_doms, score_from=cfg.score_from, batch_size=cfg.batch_size
                )
                cand_reason_nll = _compute_causal_nlls_by_domain(
                    work_model, reason_ids, reason_doms, score_from=cfg.score_from, batch_size=cfg.batch_size
                )
                cand_dom_nll = {**cand_prose_nll, **cand_reason_nll}
        except ValueError:
            non_finite_error = True
            cand_dom_nll = {}

        if non_finite_error:
            anse_summary = {
                "guard_ok": False,
                "general_guard_ok": False,
                "general_pct_vs_base": float("inf"),
                "worst_domain_pct_vs_base": float("inf"),
                "nll_mean": float("inf"),
                "anse_energy": cfg.maximum_pain_energy,
            }
            prose_nll_cand = float("inf")
            reason_nll_cand = float("inf")
        else:
            anse_summary = _compute_anse_verdict(
                parent_dom_nll,
                cand_dom_nll,
                rep_after,
                budget_ok=bool(synth["budget_respected"]),
                cfg=cfg,
            )
            prose_nll_cand = (
                sum(cand_dom_nll[d] for d in gen_keys if d in cand_dom_nll)
                / float(max(1, sum(1 for d in gen_keys if d in cand_dom_nll)))
                if any(d in cand_dom_nll for d in gen_keys)
                else sum(cand_dom_nll.values()) / float(len(cand_dom_nll))
            )
            reason_nll_cand = (
                sum(cand_dom_nll[d] for d in reason_keys if d in cand_dom_nll)
                / float(max(1, sum(1 for d in reason_keys if d in cand_dom_nll)))
                if any(d in cand_dom_nll for d in reason_keys)
                else prose_nll_cand
            )

        cand_anse_energy = float(anse_summary["anse_energy"])
        delta_e = cand_anse_energy - parent_anse_energy
        loop_dropped = (rep_after < rep_before) and (
            (rep_before - rep_after) / max(rep_before, 1e-12) >= cfg.min_loop_reduction_rel
        )
        energy_ok = (delta_e < 0.0) if cfg.require_strict_energy_drop else (delta_e <= 0.0)
        guard_ok = bool(anse_summary["guard_ok"])

        if not guard_ok or non_finite_error:
            status = "ROLLED_BACK_ANSE_MAXIMUM_PAIN"
            accepted = False
            rolled_back = True
            final_sha = _restore_snapshot_bitwise(work_model, parent_snapshot, parent_sha)
        elif not loop_dropped or not energy_ok:
            status = "ROLLED_BACK_LOOP_OR_ENERGY"
            accepted = False
            rolled_back = True
            final_sha = _restore_snapshot_bitwise(work_model, parent_snapshot, parent_sha)
        else:
            status = "ACCEPTED"
            accepted = True
            rolled_back = False
            final_sha = cand_sha

        suppression_frac = (
            (rep_before - rep_after) / rep_before if accepted and rep_before > 1e-12 else 0.0
        )
        active_nll_mean = float(anse_summary["nll_mean"]) if accepted else parent_nll_mean
        active_anse = cand_anse_energy if accepted else parent_anse_energy
        active_prose_nll = prose_nll_cand if accepted else prose_nll_before
        active_reason_nll = reason_nll_cand if accepted else reason_nll_before
        active_rep = rep_after if accepted else rep_before

        receipt = _assemble_receipt(
            accepted=accepted,
            rolled_back=rolled_back,
            status=status,
            nll_mean=active_nll_mean,
            parent_nll_mean=parent_nll_mean,
            cand_nll_mean=float(anse_summary["nll_mean"]) if math.isfinite(float(anse_summary["nll_mean"])) else None,
            prose_nll_before=prose_nll_before,
            prose_nll_after=active_prose_nll,
            reason_nll_before=reason_nll_before,
            reason_nll_after=active_reason_nll,
            rep_before=rep_before,
            rep_after=active_rep,
            edited_elements=int(synth["edited_elements"]) if accepted else 0,
            factorized_param_count=int(synth["factorized_param_count"]) if accepted else 0,
            total_model_params=total_params,
            budget_respected=bool(synth["budget_respected"]),
            oof_snr=float(diag["oof_snr"]),
            passes_oof_gate=True,
            nullspace_leakage_norm=float(synth["nullspace_leakage_norm"]),
            anse_energy=active_anse,
            parent_anse_energy=parent_anse_energy,
            cand_anse_energy=cand_anse_energy,
            delta_e=delta_e,
            guard_ok=guard_ok,
            general_guard_ok=bool(anse_summary["general_guard_ok"]),
            general_pct_vs_base=(
                float(anse_summary["general_pct_vs_base"])
                if math.isfinite(float(anse_summary["general_pct_vs_base"]))
                else None
            ),
            worst_domain_pct_vs_base=(
                float(anse_summary["worst_domain_pct_vs_base"])
                if math.isfinite(float(anse_summary["worst_domain_pct_vs_base"]))
                else None
            ),
            target_label=selected_label,
            parent_sha=parent_sha,
            cand_sha=cand_sha,
            final_sha=final_sha,
            cfg=cfg,
            cand_rep_after=rep_after,
            loop_suppression_fraction=suppression_frac,
        )
        return work_model, receipt
    finally:
        work_model.train(was_training)


def _assemble_receipt(
    *,
    accepted: bool,
    rolled_back: bool,
    status: str,
    nll_mean: float,
    parent_nll_mean: float,
    cand_nll_mean: float | None,
    prose_nll_before: float,
    prose_nll_after: float,
    reason_nll_before: float,
    reason_nll_after: float,
    rep_before: float,
    rep_after: float,
    edited_elements: int,
    factorized_param_count: int,
    total_model_params: int,
    budget_respected: bool,
    oof_snr: float,
    passes_oof_gate: bool,
    nullspace_leakage_norm: float,
    anse_energy: float,
    parent_anse_energy: float,
    cand_anse_energy: float,
    delta_e: float,
    guard_ok: bool,
    general_guard_ok: bool,
    general_pct_vs_base: float | None,
    worst_domain_pct_vs_base: float | None,
    target_label: str,
    parent_sha: str,
    cand_sha: str,
    final_sha: str,
    cfg: AntiLoopWSPSurgeryConfig,
    cand_rep_after: float | None = None,
    loop_suppression_fraction: float = 0.0,
) -> dict[str, Any]:
    """Assemble deterministic JSON-safe provenance receipt signed with SHA-256."""
    edited_frac = float(edited_elements) / float(total_model_params)
    fac_frac = float(factorized_param_count) / float(total_model_params)
    receipt: dict[str, Any] = {
        "object": SUPPRESSOR_RECEIPT_OBJECT,
        "schema": SUPPRESSOR_SCHEMA,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "status": status,
        "target_label": target_label,
        "nll_mean": float(nll_mean),
        "parent_nll_mean": float(parent_nll_mean),
        "cand_nll_mean": cand_nll_mean,
        "prose_nll_before": float(prose_nll_before),
        "prose_nll_after": float(prose_nll_after),
        "reasoning_nll_before": float(reason_nll_before),
        "reasoning_nll_after": float(reason_nll_after),
        "reasoning_nll_delta": float(reason_nll_after - reason_nll_before),
        "repeat_4gram_ratio": float(rep_after),
        "repeat_4gram_ratio_before": float(rep_before),
        "repeat_4gram_ratio_after": float(rep_after),
        "cand_repeat_4gram_ratio": float(cand_rep_after if cand_rep_after is not None else rep_after),
        "loop_suppression_fraction": float(loop_suppression_fraction),
        "edited_elements": edited_elements,
        "factorized_param_count": factorized_param_count,
        "total_model_params": total_model_params,
        "edited_param_fraction": edited_frac,
        "factorized_param_fraction": fac_frac,
        "budget_respected": budget_respected,
        "oof_snr": float(oof_snr),
        "passes_oof_gate": passes_oof_gate,
        "nullspace_leakage_norm": float(nullspace_leakage_norm),
        "anse_energy": float(anse_energy),
        "parent_anse_energy": float(parent_anse_energy),
        "cand_anse_energy": float(cand_anse_energy),
        "delta_e": float(delta_e),
        "guard_ok": guard_ok,
        "general_guard_ok": general_guard_ok,
        "general_pct_vs_base": general_pct_vs_base,
        "worst_domain_pct_vs_base": worst_domain_pct_vs_base,
        "parent_state_sha256": parent_sha,
        "candidate_state_sha256": cand_sha,
        "final_state_sha256": final_sha,
        "config": asdict(cfg),
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "AntiLoopWSPSurgeryConfig",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_DAMP_BETA",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_GENERAL_GUARD_MAX_RATIO",
    "DEFAULT_LOOP_ENERGY_WEIGHT",
    "DEFAULT_MAX_ATTRACTOR_RANK",
    "DEFAULT_MIN_OOF_SNR",
    "DEFAULT_NGRAM_ORDER",
    "DEFAULT_NULLSPACE_SV_TOL",
    "DEFAULT_PROSE_SV_REL_TOL",
    "DEFAULT_PROSE_VARIANCE_THRESHOLD",
    "DEFAULT_SOFT_MARGIN_PENALTY",
    "DEFAULT_TAU_TEMP",
    "GENERAL_EXPERT_PREFIX",
    "MAX_BUDGET_FRACTION_CEILING",
    "REASONING_EXPERT_PREFIX",
    "SUPPRESSOR_RECEIPT_OBJECT",
    "SUPPRESSOR_SCHEMA",
    "VALID_PROJECTION_SIDES",
    "compute_ngram_repetition_metrics",
    "diagnose_repetition_attractor_subspace",
    "execute_verified_antiloop_wsp_surgery",
    "synthesize_sub002pct_antiloop_tail_surgery",
    "verify_antiloop_receipt_provenance",
]

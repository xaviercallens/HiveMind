# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Reine-Cortex (QueenCortex) — Active Royal-Jelly MoE Campaign Governor & Pheromone Planner.

Implements the active closed-loop **La Ruche / Reine-Cortex / Les Abeilles** NeuroController
governing HiveMind, CrowdGPT, and OpenDiLoCo Sparse MoE campaigns on CPU (``0 MiB`` GPU VRAM,
``0`` gradient steps), grounded in ``PILOT1``..``PILOT10`` empirical findings across
``CrowdV1WithEngram`` (``C2``/``C3``), ``Granite-3.1-MoE``, and ``Qwen2.5-3B``:

1. ``distill_forager_nectar_to_royal_jelly(...)`` (**Les Abeilles Nourrices — Gelée Royale**):
   Purifies raw volunteer Forager Bee (``Abeilles Butineuses``) expert deltas via TIES
   sign-consensus election (``ties_density=0.25``), AlphaEdit ``P_0`` prose-nullspace
   projection, anchor-comb preservation / ghost-sink zeroing, and closed-form
   ``ker(G_D0)`` diagonal-Fisher RMSNorm/LayerNorm calibration so drifting/discarded
   outer rounds (``kept=[]``) are rescued in zero-VRAM closed form before rejection.
2. ``emit_queen_pheromone_campaign_directive(...)`` (**Phéromones Royales**):
   Issues a SHA-256-signed ``QueenPheromoneDirective`` for the upcoming campaign round
   specifying ``anchor_experts_to_freeze`` (``Alvéoles Ancres``),
   ``corruptor_experts_to_sink_or_reseed``, ``router_training_mode`` (``router_frozen``
   when co-adaptation KL is fragile vs ``router_unfrozen_coadapted``), and normalized
   Boltzmann/UCB ``bee_foraging_domain_weights`` steering bees toward highest-deficit domains.
3. ``execute_queen_cortex_campaign_round(...)`` (**Operculation d'Alvéole ANSE**):
   Orchestrates raw-round auditing, Royal-Jelly distillation, next-round pheromone emission,
   and thermodynamic gating under ``ANSE Delta E < 0`` and ``D0 / worst_domain <= +1.0%``
   with ANSE Maximum Pain (``1e6``) atomic bitwise rollback and SHA-256 provenance receipts.
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

from hivemind.swarm.byzantine_wsp_delta_aggregator import compute_payload_sha256
from hivemind.swarm.crowdgpt_moe_opendiloco import CrowdLoRAMoEMLP
from hivemind.swarm.folded_norm_ece_calibrator import compute_token_nll_and_ece
from hivemind.swarm.logit_calibration_temperature_surgeon import (
    fold_temperature_into_norm_params,
)
from hivemind.swarm.self_healing_checkpoint_rescuer import (
    apply_checkpoint_blocks_inplace,
    compute_checkpoint_sha256,
    extract_checkpoint_blocks,
)
from hivemind.swarm.tangent_task_vector_surgeon import (
    compute_prose_nullspace_projector_scaled,
)
from hivemind.swarm.ties_dare_moe_merger import trim_and_elect_sign_ties
from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    forward_block_post_mlp,
    forward_block_pre_mlp,
)

QUEEN_CORTEX_SCHEMA: str = "hivemind.queen_cortex_moe_campaign_governor/1"
QUEEN_PHEROMONE_OBJECT: str = "hivemind.queen_pheromone_directive"
QUEEN_CAMPAIGN_RECEIPT_OBJECT: str = "hivemind.queen_cortex_campaign_round_receipt"

ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_TIES_DENSITY: float = 0.25
DEFAULT_ANCHOR_GAIN_FLOOR_NATS: float = -0.01
DEFAULT_PROSE_HARM_SINK_FLOOR_NATS: float = 0.0015
DEFAULT_COADAPT_KL_CEILING: float = 0.05
DEFAULT_FORAGING_TEMPERATURE: float = 0.25
DEFAULT_UCB_EXPLORATION_COEFF: float = 0.05
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_D0_RATIO: float = 1.0085
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_ECE_ENERGY_WEIGHT: float = 0.25
DEFAULT_FISHER_DAMPING: float = 1e-3
DEFAULT_STEP_SCALES: tuple[float, ...] = (0.01, 0.025, 0.05, 0.10, 0.18)
DEFAULT_TAU_GRID: tuple[float, ...] = (0.96, 1.00, 1.03, 1.05, 1.08, 1.12)
DEFAULT_MAX_REL_DELTA: float = 0.08
DEFAULT_MAX_D0_REL_PROBE: float = 0.004
DEFAULT_GENERAL_PREFIXES: tuple[str, ...] = ("E0_", "D0_", "general")

VALID_ROUTER_TRAINING_MODES: frozenset[str] = frozenset(
    {"router_frozen", "router_unfrozen_coadapted"}
)
VALID_ROLLBACK_TARGETS: frozenset[str] = frozenset({"base", "parent"})

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPERT_TAIL_RE = re.compile(r"(?:\.experts\.|^expert_)(\d+)$")
_LAYER_ID_RE = re.compile(r"(?:^|\.)(?:blocks|layers|layer)[._]?(\d+)(?:\.|$)")
_SINGLE_TENSOR_KEY = "__tensor__"
_ROUTER_SUFFIX = ".router"


# ---------------------------------------------------------------------------
# 1. Validation, Configuration & Signed Receipts
# ---------------------------------------------------------------------------


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_int(value: Any, name: str, *, min_value: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return int(value)


def _require_float(
    value: Any,
    name: str,
    *,
    min_value: float | None = None,
    strict_min: bool = False,
    max_value: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if min_value is not None and (out <= min_value if strict_min else out < min_value):
        op = ">" if strict_min else ">="
        raise ValueError(f"{name} must be {op} {min_value}, got {out}")
    if max_value is not None and out > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {out}")
    return out


def _require_nonempty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_fp_tensor(
    tensor: Any,
    name: str,
    *,
    exact_ndim: int | None = None,
    min_ndim: int | None = None,
    require_finite: bool = True,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if exact_ndim is not None and tensor.ndim != exact_ndim:
        raise ValueError(f"{name} must have ndim == {exact_ndim}, got shape {tuple(tensor.shape)}")
    if min_ndim is not None and tensor.ndim < min_ndim:
        raise ValueError(f"{name} must have ndim >= {min_ndim}, got shape {tuple(tensor.shape)}")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    if require_finite and not bool(torch.isfinite(tensor.detach()).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _require_projector(proj: Any, name: str, *, eig_tol: float = 1e-2) -> torch.Tensor:
    mat = _require_fp_tensor(proj, name, exact_ndim=2).detach().to(device="cpu", dtype=torch.float64)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be square, got shape {tuple(mat.shape)}")
    asym = float((mat - mat.T).abs().max().item())
    scale = max(1.0, float(mat.abs().max().item()))
    if asym > max(1e-4, eig_tol) * scale:
        raise ValueError(f"{name} must be symmetric (max asymmetry {asym:.3e})")
    sym = (0.5 * (mat + mat.T)).contiguous()
    evals = torch.linalg.eigvalsh(sym)
    min_ev, max_ev = float(evals.min().item()), float(evals.max().item())
    if min_ev < -eig_tol or max_ev > 1.0 + eig_tol:
        raise ValueError(
            f"{name} eigenvalues [{min_ev:.3e}, {max_ev:.3e}] fall outside [-{eig_tol}, 1+{eig_tol}]"
        )
    return sym


def _require_float_sequence(values: Any, name: str, *, strict_pos: bool = False) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence) or not values:
        raise ValueError(f"{name} must be a non-empty sequence of finite floats")
    return tuple(
        _require_float(v, f"{name}[{i}]", min_value=0.0, strict_min=strict_pos)
        for i, v in enumerate(values)
    )


def _is_general_domain(domain_name: str, prefixes: Sequence[str] = DEFAULT_GENERAL_PREFIXES) -> bool:
    low = domain_name.strip().lower()
    return any(low.startswith(p.lower()) for p in prefixes)


def _parse_layer_idx(block_id: str) -> int | None:
    match = _LAYER_ID_RE.search(block_id.strip())
    return int(match.group(1)) if match is not None else None


def _parse_expert_idx(expert_key: str) -> int | None:
    stripped = expert_key.strip()
    if stripped.isdigit():
        return int(stripped)
    match = _EXPERT_TAIL_RE.search(stripped)
    return int(match.group(1)) if match is not None else None


def _is_final_ln_key(block_id: str) -> bool:
    low = block_id.strip().lower()
    return low in {"final_ln", "ln", "norm", "model.ln", "model.norm"} or low.endswith(
        (".ln", ".final_ln", ".norm")
    )


@dataclass(frozen=True)
class QueenCortexConfig:
    """Validated configuration governing Reine-Cortex distillation, pheromones, and ANSE gates."""

    ties_density: float = DEFAULT_TIES_DENSITY
    ties_sign_mode: str = "mass"
    max_peer_norm_ratio: float | None = 4.0
    anchor_gain_floor_nats: float = DEFAULT_ANCHOR_GAIN_FLOOR_NATS
    corruptor_prose_harm_floor_nats: float = DEFAULT_PROSE_HARM_SINK_FLOOR_NATS
    coadapt_kl_ceiling: float = DEFAULT_COADAPT_KL_CEILING
    foraging_temperature: float = DEFAULT_FORAGING_TEMPERATURE
    ucb_exploration_coeff: float = DEFAULT_UCB_EXPLORATION_COEFF
    max_domain_drift_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_D0_RATIO
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    ece_energy_weight: float = DEFAULT_ECE_ENERGY_WEIGHT
    min_energy_improvement: float = 1e-6
    fisher_damping: float = DEFAULT_FISHER_DAMPING
    fisher_step_scales: tuple[float, ...] = DEFAULT_STEP_SCALES
    tau_grid: tuple[float, ...] = DEFAULT_TAU_GRID
    max_norm_rel_delta: float = DEFAULT_MAX_REL_DELTA
    probe_max_d0_rel: float = DEFAULT_MAX_D0_REL_PROBE
    d0_grad_sv_rel_tol: float = 1e-6
    score_from: int = 32
    batch_size: int = 24
    general_prefixes: tuple[str, ...] = DEFAULT_GENERAL_PREFIXES
    rollback_target: str = "base"
    inplace: bool = True

    def __post_init__(self) -> None:
        _require_float(self.ties_density, "ties_density", min_value=0.0, strict_min=True, max_value=1.0)
        if self.ties_sign_mode not in {"mass", "majority"}:
            raise ValueError("ties_sign_mode must be 'mass' or 'majority'")
        if self.max_peer_norm_ratio is not None:
            _require_float(self.max_peer_norm_ratio, "max_peer_norm_ratio", min_value=0.0, strict_min=True)
        _require_float(self.anchor_gain_floor_nats, "anchor_gain_floor_nats", max_value=0.0)
        _require_float(self.corruptor_prose_harm_floor_nats, "corruptor_prose_harm_floor_nats", min_value=0.0)
        _require_float(self.coadapt_kl_ceiling, "coadapt_kl_ceiling", min_value=0.0)
        _require_float(self.foraging_temperature, "foraging_temperature", min_value=0.0, strict_min=True)
        _require_float(self.ucb_exploration_coeff, "ucb_exploration_coeff", min_value=0.0)
        _require_float(self.max_domain_drift_ratio, "max_domain_drift_ratio", min_value=1.0)
        _require_float(self.anse_safe_general_ratio, "anse_safe_general_ratio", min_value=1.0)
        _require_float(self.soft_margin_penalty, "soft_margin_penalty", min_value=0.0)
        _require_float(self.maximum_pain_energy, "maximum_pain_energy", min_value=0.0, strict_min=True)
        _require_float(self.ece_energy_weight, "ece_energy_weight", min_value=0.0)
        _require_float(self.min_energy_improvement, "min_energy_improvement", min_value=0.0)
        _require_float(self.fisher_damping, "fisher_damping", min_value=0.0, strict_min=True)
        object.__setattr__(
            self, "fisher_step_scales", _require_float_sequence(self.fisher_step_scales, "fisher_step_scales")
        )
        object.__setattr__(
            self, "tau_grid", _require_float_sequence(self.tau_grid, "tau_grid", strict_pos=True)
        )
        _require_float(self.max_norm_rel_delta, "max_norm_rel_delta", min_value=0.0, strict_min=True)
        _require_float(self.probe_max_d0_rel, "probe_max_d0_rel", min_value=0.0)
        _require_float(self.d0_grad_sv_rel_tol, "d0_grad_sv_rel_tol", min_value=0.0)
        _require_int(self.score_from, "score_from", min_value=0)
        _require_int(self.batch_size, "batch_size", min_value=1)
        if not isinstance(self.general_prefixes, tuple) or not self.general_prefixes:
            raise ValueError("general_prefixes must be a non-empty tuple of strings")
        for idx, pref in enumerate(self.general_prefixes):
            _require_nonempty_str(pref, f"general_prefixes[{idx}]")
        if self.rollback_target not in VALID_ROLLBACK_TARGETS:
            raise ValueError(f"rollback_target must be one of {sorted(VALID_ROLLBACK_TARGETS)}")
        _require_bool(self.inplace, "inplace")


@dataclass(frozen=True)
class QueenPheromoneDirective:
    """Signed Reine-Cortex directive governing expert freezes, router mode, and bee foraging."""

    object: str
    schema: str
    campaign_id: str
    round_index: int
    anchor_experts_to_freeze: tuple[str, ...]
    corruptor_experts_to_sink_or_reseed: tuple[str, ...]
    active_specialist_experts: tuple[str, ...]
    router_training_mode: str
    max_router_expert_kl_mismatch: float
    coadapt_kl_ceiling: float
    bee_foraging_domain_weights: dict[str, float]
    domain_priorities: tuple[dict[str, Any], ...]
    directive_sha256: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["anchor_experts_to_freeze"] = list(self.anchor_experts_to_freeze)
        data["corruptor_experts_to_sink_or_reseed"] = list(self.corruptor_experts_to_sink_or_reseed)
        data["active_specialist_experts"] = list(self.active_specialist_experts)
        data["domain_priorities"] = [dict(row) for row in self.domain_priorities]
        return data


def _verify_signed_payload(
    payload: Mapping[str, Any],
    *,
    expected_object: str,
    sig_field: str,
) -> bool:
    if not isinstance(payload, Mapping):
        raise TypeError(f"payload must be a mapping, got {type(payload).__name__}")
    claimed = payload.get(sig_field)
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if payload.get("object") != expected_object or payload.get("schema") != QUEEN_CORTEX_SCHEMA:
        return False
    core = {k: v for k, v in payload.items() if k != sig_field}
    try:
        return claimed == compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False


def verify_queen_pheromone_directive_signature(
    directive: QueenPheromoneDirective | Mapping[str, Any],
) -> bool:
    """Verify canonical SHA-256 signature on a ``QueenPheromoneDirective`` or dict."""
    if isinstance(directive, QueenPheromoneDirective):
        payload = directive.to_dict()
    elif isinstance(directive, Mapping):
        payload = directive
    else:
        raise TypeError(
            f"directive must be QueenPheromoneDirective or Mapping, got {type(directive).__name__}"
        )
    return _verify_signed_payload(
        payload, expected_object=QUEEN_PHEROMONE_OBJECT, sig_field="directive_sha256"
    )


def verify_queen_cortex_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Verify schema, object tag, signed pheromone directive, and canonical ``receipt_sha256``."""
    ok_receipt = _verify_signed_payload(
        receipt, expected_object=QUEEN_CAMPAIGN_RECEIPT_OBJECT, sig_field="receipt_sha256"
    )
    if not ok_receipt:
        return False
    pheromone = receipt.get("pheromone_directive")
    return isinstance(pheromone, Mapping) and verify_queen_pheromone_directive_signature(pheromone)


# ---------------------------------------------------------------------------
# 2. Component (a): Royal-Jelly Distillation & ker(G_D0) Fisher Norm Surgeon
# ---------------------------------------------------------------------------


def _normalize_probe_inputs(
    normed_unit_probe: torch.Tensor | Sequence[torch.Tensor],
    probe_targets: torch.Tensor | Sequence[torch.Tensor],
    is_general_mask: torch.Tensor | Sequence[bool] | np.ndarray,
    *,
    hidden_dim: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize probe unit activations, integer targets, and general mask to ``(C, T, D)``."""
    if isinstance(normed_unit_probe, torch.Tensor):
        u_t = _require_fp_tensor(normed_unit_probe, "normed_unit_probe", min_ndim=2)
        if u_t.ndim == 2:
            u_t = u_t.unsqueeze(1)
        elif u_t.ndim != 3:
            raise ValueError(f"normed_unit_probe must be 2D or 3D, got ndim={u_t.ndim}")
    elif isinstance(normed_unit_probe, Sequence) and not isinstance(normed_unit_probe, (str, bytes)):
        if not normed_unit_probe:
            raise ValueError("normed_unit_probe sequence must not be empty")
        chunks = [
            _require_fp_tensor(c, f"normed_unit_probe[{i}]", exact_ndim=2)
            for i, c in enumerate(normed_unit_probe)
        ]
        u_t = torch.stack(chunks, dim=0)
    else:
        raise TypeError("normed_unit_probe must be a Tensor or non-empty sequence of 2D Tensors")

    if int(u_t.shape[2]) != hidden_dim:
        raise ValueError(f"normed_unit_probe hidden dim {u_t.shape[2]} != w_norm_base dim {hidden_dim}")

    c_num, t_len, _ = u_t.shape
    if isinstance(probe_targets, torch.Tensor):
        t_t = probe_targets.detach()
    elif isinstance(probe_targets, Sequence) and not isinstance(probe_targets, (str, bytes)):
        if not probe_targets:
            raise ValueError("probe_targets sequence must not be empty")
        t_t = torch.stack([torch.as_tensor(x) for x in probe_targets], dim=0)
    else:
        raise TypeError("probe_targets must be a Tensor or non-empty sequence of target tensors")

    if t_t.is_floating_point() or t_t.dtype == torch.bool:
        raise TypeError(f"probe_targets must have an integer dtype, got {t_t.dtype}")
    if t_t.ndim == 1 and t_len == 1 and int(t_t.shape[0]) == c_num:
        t_t = t_t.unsqueeze(1)
    if tuple(t_t.shape) != (c_num, t_len):
        raise ValueError(f"probe_targets shape {tuple(t_t.shape)} != expected {(c_num, t_len)}")
    tgt_long = t_t.to(device="cpu", dtype=torch.long).contiguous()
    if bool((tgt_long < 0).any().item()) or bool((tgt_long >= vocab_size).any().item()):
        raise ValueError(f"probe_targets IDs must lie in [0, {vocab_size})")

    if isinstance(is_general_mask, torch.Tensor):
        if is_general_mask.dtype != torch.bool:
            raise TypeError(f"is_general_mask Tensor must be bool, got {is_general_mask.dtype}")
        m_arr = is_general_mask.detach().cpu()
    elif isinstance(is_general_mask, np.ndarray):
        if is_general_mask.dtype != np.bool_:
            raise TypeError(f"is_general_mask ndarray must be bool, got {is_general_mask.dtype}")
        m_arr = torch.from_numpy(is_general_mask.copy())
    elif isinstance(is_general_mask, Sequence) and not isinstance(is_general_mask, (str, bytes)):
        if not is_general_mask or not all(isinstance(v, bool) for v in is_general_mask):
            raise TypeError("is_general_mask sequence must be non-empty bool values")
        m_arr = torch.tensor(list(is_general_mask), dtype=torch.bool)
    else:
        raise TypeError("is_general_mask must be a bool Tensor, ndarray, or Sequence[bool]")

    if m_arr.ndim == 2 and tuple(m_arr.shape) == (c_num, t_len):
        chunk_mask = m_arr.all(dim=1)
    elif m_arr.ndim == 1 and int(m_arr.shape[0]) == c_num:
        chunk_mask = m_arr
    else:
        raise ValueError(f"is_general_mask shape {tuple(m_arr.shape)} incompatible with ({c_num}, {t_len})")
    if not bool(chunk_mask.any().item()):
        raise ValueError("is_general_mask must select at least one general-prose chunk")

    return u_t.detach().to(device="cpu", dtype=torch.float64).contiguous(), tgt_long, chunk_mask


@torch.no_grad()
def compute_ker_g_d0_fisher_norm_step(
    w_norm_base: torch.Tensor,
    normed_unit_probe: torch.Tensor | Sequence[torch.Tensor],
    head_weight: torch.Tensor,
    probe_targets: torch.Tensor | Sequence[torch.Tensor],
    is_general_mask: torch.Tensor | Sequence[bool] | np.ndarray,
    *,
    b_norm_base: torch.Tensor | None = None,
    damping: float = DEFAULT_FISHER_DAMPING,
    step_scales: Sequence[float] = DEFAULT_STEP_SCALES,
    tau_grid: Sequence[float] = DEFAULT_TAU_GRID,
    max_rel_delta: float = DEFAULT_MAX_REL_DELTA,
    max_d0_rel: float = DEFAULT_MAX_D0_REL_PROBE,
    ece_weight: float = DEFAULT_ECE_ENERGY_WEIGHT,
    sv_rel_tol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """Closed-form diagonal-Fisher natural-gradient step on ``w_norm`` inside ``ker(G_D0)``.

    Stacks exact per-chunk general-prose gradients ``G_D0`` wrt ``w_norm``, projects the
    Fisher-preconditioned target descent vector onto ``ker(G_D0)`` via
    ``P_ker = I - V_0^T V_0`` so first-order prose drift satisfies ``G_D0 @ delta_w == 0``,
    and folds optimal calibration temperature ``1 / tau_temp`` into final norm parameters.
    """
    w0_raw = _require_fp_tensor(w_norm_base, "w_norm_base", exact_ndim=1)
    head_raw = _require_fp_tensor(head_weight, "head_weight", exact_ndim=2)
    hidden_dim = int(w0_raw.shape[0])
    vocab_size, head_dim = int(head_raw.shape[0]), int(head_raw.shape[1])
    if head_dim != hidden_dim:
        raise ValueError(f"head_weight width {head_dim} != w_norm_base dim {hidden_dim}")

    b0_raw: torch.Tensor | None = None
    if b_norm_base is not None:
        b0_raw = _require_fp_tensor(b_norm_base, "b_norm_base", exact_ndim=1)
        if int(b0_raw.shape[0]) != hidden_dim:
            raise ValueError(f"b_norm_base dim {b0_raw.shape[0]} != w_norm_base dim {hidden_dim}")

    damping_f = _require_float(damping, "damping", min_value=0.0, strict_min=True)
    scales_t = _require_float_sequence(step_scales, "step_scales")
    taus_t = _require_float_sequence(tau_grid, "tau_grid", strict_pos=True)
    max_rel_f = _require_float(max_rel_delta, "max_rel_delta", min_value=0.0, strict_min=True)
    max_d0_f = _require_float(max_d0_rel, "max_d0_rel", min_value=0.0)
    ece_w_f = _require_float(ece_weight, "ece_weight", min_value=0.0)
    sv_tol_f = _require_float(sv_rel_tol, "sv_rel_tol", min_value=0.0)

    u_stack, tgt_stack, gen_chunk_mask = _normalize_probe_inputs(
        normed_unit_probe, probe_targets, is_general_mask, hidden_dim=hidden_dim, vocab_size=vocab_size
    )
    w0 = w0_raw.detach().to(device="cpu", dtype=torch.float64)
    b0 = b0_raw.detach().to(device="cpu", dtype=torch.float64) if b0_raw is not None else None
    head_f = head_raw.detach().to(device="cpu", dtype=torch.float64)
    c_num, t_len, _ = u_stack.shape

    x0 = u_stack * w0 + (b0 if b0 is not None else 0.0)
    lg0 = F.linear(x0.reshape(-1, hidden_dim), head_f)
    probs0 = F.softmax(lg0, dim=-1)
    probs0.scatter_add_(
        -1,
        tgt_stack.reshape(-1, 1),
        torch.full((c_num * t_len, 1), -1.0, dtype=torch.float64),
    )
    tok_grads = (probs0 @ head_f).reshape(c_num, t_len, hidden_dim) * u_stack
    chunk_grads = tok_grads.mean(dim=1)  # (C, D)
    f_diag = tok_grads.pow(2).mean(dim=(0, 1)) + damping_f

    g_d0_mat = chunk_grads[gen_chunk_mask]  # (|D0|, D)
    _, s0, vh0 = torch.linalg.svd(g_d0_mat, full_matrices=False)
    keep0 = (
        int((s0 > sv_tol_f * float(s0[0].item())).sum().item())
        if s0.numel() > 0 and float(s0[0].item()) > 1e-15
        else 0
    )
    v0 = vh0[:keep0, :]  # (keep0, D)

    spec_mask = ~gen_chunk_mask
    g_all = chunk_grads.mean(dim=0)
    g_spec = chunk_grads[spec_mask].mean(dim=0) if bool(spec_mask.any().item()) else g_all
    g_target = 0.35 * g_all + 0.65 * g_spec

    def _project_ker(vec: torch.Tensor) -> torch.Tensor:
        return vec - (v0.T @ (v0 @ vec)) if keep0 > 0 else vec

    # PSD projected Fisher preconditioning: P_ker @ F^{-1/2} @ P_ker guarantees descent in ker(G_D0)
    g_ker = _project_ker(g_target)
    proj_step = _project_ker(g_ker / f_diag.sqrt())
    rms_step = float(proj_step.pow(2).mean().sqrt().item())
    unit_dir = proj_step / rms_step if rms_step > 1e-12 else torch.zeros_like(proj_step)

    tgt_flat = tgt_stack.reshape(-1)
    gen_tok_mask = gen_chunk_mask.unsqueeze(1).expand(c_num, t_len).reshape(-1)

    def _eval_scaled_logits(lg_base: torch.Tensor, tau_val: float) -> tuple[float, float, float]:
        scaled_lg = lg_base * (1.0 / tau_val)
        metrics = compute_token_nll_and_ece(scaled_lg, tgt_flat)
        tok_nll = F.cross_entropy(scaled_lg, tgt_flat, reduction="none")
        d0_nll = float(tok_nll[gen_tok_mask].mean().item())
        return float(metrics["nll_mean"]), float(metrics["expected_calibration_error"]), d0_nll

    lg_raw = F.linear(x0.reshape(-1, hidden_dim), head_f)
    base_nll, base_ece, raw_base_d0 = _eval_scaled_logits(lg_raw, 1.0)
    best_obj = base_nll + ece_w_f * base_ece
    best_delta_w = torch.zeros_like(w0)
    best_eta = 0.0
    best_tau = 1.0
    best_nll, best_ece, best_d0 = base_nll, base_ece, raw_base_d0

    cand_etas: tuple[float, ...] = (0.0,) + tuple(e for e in scales_t if e > 0.0)
    for eta in cand_etas:
        if eta == 0.0:
            delta_w = torch.zeros_like(w0)
            lg_eta = lg_raw
        else:
            raw_delta = -eta * (w0.abs().clamp_min(0.1) * unit_dir)
            capped = torch.clamp(raw_delta, min=-max_rel_f * w0.abs(), max=max_rel_f * w0.abs())
            delta_w = _project_ker(capped)
            x_in = u_stack * (w0 + delta_w) + (b0 if b0 is not None else 0.0)
            lg_eta = F.linear(x_in.reshape(-1, hidden_dim), head_f)
        for tau_val in taus_t:
            m_nll, m_ece, d0_nll = _eval_scaled_logits(lg_eta, tau_val)
            d0_rel = (d0_nll - raw_base_d0) / max(1e-9, raw_base_d0)
            obj_val = m_nll + ece_w_f * m_ece
            if d0_rel <= max_d0_f and obj_val < best_obj - 1e-8:
                best_obj = obj_val
                best_delta_w = delta_w
                best_eta = eta
                best_tau = tau_val
                best_nll, best_ece, best_d0 = m_nll, m_ece, d0_nll

    delta_w_out = best_delta_w.to(device=w0_raw.device, dtype=w0_raw.dtype).contiguous()
    w_pre_tau = (w0 + delta_w_out.to(device="cpu", dtype=torch.float64)).to(
        device=w0_raw.device, dtype=w0_raw.dtype
    )
    w_cal, b_cal = fold_temperature_into_norm_params(w_pre_tau, b0_raw, best_tau)
    ortho_err = float((g_d0_mat @ delta_w_out.detach().cpu().to(torch.float64)).abs().max().item())

    return w_cal, b_cal, {
        "eta_fisher": float(best_eta),
        "tau_temp": float(best_tau),
        "delta_w_nullspace": delta_w_out,
        "w_norm_pre_tau": w_pre_tau,
        "g_d0_matrix": g_d0_mat.to(torch.float32),
        "g_d0_rank": int(keep0),
        "ker_g_d0_dim": int(hidden_dim - keep0),
        "first_order_d0_orthogonality": ortho_err,
        "probe_nll_before": base_nll,
        "probe_nll_after": best_nll,
        "probe_ece_before": base_ece,
        "probe_ece_after": best_ece,
        "probe_d0_rel_drift": (best_d0 - raw_base_d0) / max(1e-9, raw_base_d0),
    }


def _normalize_expert_store(
    experts: Any,
    name: str,
    *,
    require_finite: bool = True,
) -> tuple[dict[str, dict[str, torch.Tensor]], list[str], dict[str, Any], dict[str, bool]]:
    """Normalize ``{expert_key: Tensor | Mapping[str, Tensor]}`` preserving key metadata."""
    if not isinstance(experts, Mapping) or not experts:
        raise TypeError(f"{name} must be a non-empty mapping of expert weights")
    norm: dict[str, dict[str, torch.Tensor]] = {}
    orig_keys: dict[str, Any] = {}
    single_flags: dict[str, bool] = {}
    for raw_k, raw_v in experts.items():
        if not isinstance(raw_k, (str, int)) or isinstance(raw_k, bool):
            raise TypeError(f"{name} keys must be str or int, got {type(raw_k).__name__}")
        k_str = str(raw_k).strip()
        if not k_str:
            raise ValueError(f"{name} keys must be non-empty")
        if k_str in norm:
            raise ValueError(f"Duplicate canonical expert key {k_str!r} in {name}")
        orig_keys[k_str] = raw_k
        if isinstance(raw_v, torch.Tensor):
            _require_fp_tensor(raw_v, f"{name}[{raw_k!r}]", min_ndim=1, require_finite=require_finite)
            norm[k_str] = {_SINGLE_TENSOR_KEY: raw_v.detach().clone()}
            single_flags[k_str] = True
        elif isinstance(raw_v, Mapping) and raw_v:
            sub: dict[str, torch.Tensor] = {}
            for sub_k, sub_v in raw_v.items():
                sk = _require_nonempty_str(sub_k, f"{name}[{raw_k!r}] sub-key")
                _require_fp_tensor(
                    sub_v, f"{name}[{raw_k!r}][{sk!r}]", min_ndim=1, require_finite=require_finite
                )
                sub[sk] = sub_v.detach().clone()
            norm[k_str] = sub
            single_flags[k_str] = False
        else:
            raise TypeError(
                f"{name}[{raw_k!r}] must be a Tensor or non-empty Mapping[str, Tensor]"
            )
    return norm, list(norm.keys()), orig_keys, single_flags


def _normalize_bee_deltas(
    forager_bee_deltas: Any,
    base_norm: Mapping[str, Mapping[str, torch.Tensor]],
    single_flags: Mapping[str, bool],
    *,
    require_finite: bool = True,
) -> tuple[list[dict[str, dict[str, torch.Tensor]]], list[str]]:
    """Normalize a sequence or mapping of volunteer Forager Bee delta dicts."""
    if isinstance(forager_bee_deltas, Mapping):
        if not forager_bee_deltas:
            raise ValueError("forager_bee_deltas must not be empty")
        bee_ids = [_require_nonempty_str(str(k), "bee_id") for k in forager_bee_deltas.keys()]
        raw_bees = list(forager_bee_deltas.values())
    elif isinstance(forager_bee_deltas, Sequence) and not isinstance(forager_bee_deltas, (str, bytes)):
        if not forager_bee_deltas:
            raise ValueError("forager_bee_deltas must not be empty")
        bee_ids = [f"bee_{i}" for i in range(len(forager_bee_deltas))]
        raw_bees = list(forager_bee_deltas)
    else:
        raise TypeError("forager_bee_deltas must be a non-empty Sequence or Mapping")

    normalized_bees: list[dict[str, dict[str, torch.Tensor]]] = []
    for b_idx, (bid, bee_map) in enumerate(zip(bee_ids, raw_bees, strict=True)):
        if not isinstance(bee_map, Mapping):
            raise TypeError(f"forager_bee_deltas[{b_idx}] ({bid}) must be a Mapping")
        parsed: dict[str, dict[str, torch.Tensor]] = {}
        for raw_ek, raw_ev in bee_map.items():
            ek = str(raw_ek).strip()
            if ek not in base_norm:
                raise ValueError(f"Unknown expert key {ek!r} in bee {bid!r}")
            exp_base = base_norm[ek]
            if single_flags[ek]:
                t = _require_fp_tensor(
                    raw_ev, f"bee[{bid}][{ek}]", min_ndim=1, require_finite=require_finite
                )
                if tuple(t.shape) != tuple(exp_base[_SINGLE_TENSOR_KEY].shape):
                    raise ValueError(
                        f"Shape mismatch on bee[{bid}][{ek}]: {tuple(t.shape)} != "
                        f"{tuple(exp_base[_SINGLE_TENSOR_KEY].shape)}"
                    )
                parsed[ek] = {_SINGLE_TENSOR_KEY: t.detach().clone()}
            else:
                if not isinstance(raw_ev, Mapping) or not raw_ev:
                    raise TypeError(f"bee[{bid}][{ek}] must be a non-empty Mapping[str, Tensor]")
                sub_d: dict[str, torch.Tensor] = {}
                for sub_k, sub_v in raw_ev.items():
                    sk = _require_nonempty_str(sub_k, f"bee[{bid}][{ek}] key")
                    if sk not in exp_base:
                        raise ValueError(f"Unknown sub-parameter {sk!r} for expert {ek!r} in bee {bid!r}")
                    t = _require_fp_tensor(
                        sub_v, f"bee[{bid}][{ek}][{sk}]", min_ndim=1, require_finite=require_finite
                    )
                    if tuple(t.shape) != tuple(exp_base[sk].shape):
                        raise ValueError(
                            f"Shape mismatch on bee[{bid}][{ek}][{sk}]: {tuple(t.shape)} != "
                            f"{tuple(exp_base[sk].shape)}"
                        )
                    sub_d[sk] = t.detach().clone()
                parsed[ek] = sub_d
        normalized_bees.append(parsed)
    return normalized_bees, bee_ids


def _resolve_expert_projector(
    p_null_by_expert: Mapping[str | int, Any] | None,
    expert_key: str,
    sub_key: str,
    *,
    delta_shape: tuple[int, ...] | None = None,
    is_single_tensor: bool = True,
) -> torch.Tensor | None:
    if p_null_by_expert is None:
        return None
    if not isinstance(p_null_by_expert, Mapping):
        raise TypeError("p_null_by_expert must be a Mapping or None")

    tail_idx = _parse_expert_idx(expert_key)
    candidates: list[Any] = [expert_key]
    if tail_idx is not None:
        candidates.extend([str(tail_idx), tail_idx, f"expert_{tail_idx}"])

    entry: Any = None
    for cand in candidates:
        if cand in p_null_by_expert:
            entry = p_null_by_expert[cand]
            break
    if entry is None:
        return None
    if isinstance(entry, torch.Tensor):
        p_mat = _require_projector(entry, f"p_null_by_expert[{expert_key!r}]")
        if (
            not is_single_tensor
            and delta_shape is not None
            and len(delta_shape) == 2
            and sub_key in {"w1_B", "w2_B", "w3_A"}
            and int(p_mat.shape[0]) not in delta_shape
        ):
            return None
        return p_mat
    if isinstance(entry, Mapping):
        if sub_key in entry:
            val = entry[sub_key]
            return None if val is None else _require_projector(val, f"p_null[{expert_key!r}][{sub_key!r}]")
        role_keys = (
            ("act", "w3_A", "x", "input")
            if sub_key == "w3_A"
            else (("output", "y") if sub_key.endswith("_B") else ("x", "input"))
        )
        for rk in role_keys:
            if rk in entry and entry[rk] is not None:
                return _require_projector(entry[rk], f"p_null[{expert_key!r}][{rk!r}]")
        return None
    raise TypeError(f"p_null_by_expert[{expert_key!r}] must be Tensor, Mapping, or None")


def _project_delta_with_p0(
    delta: torch.Tensor,
    p_mat: torch.Tensor | None,
    sub_key: str,
    expert_key: str,
) -> torch.Tensor:
    delta_64 = delta.detach().to(device="cpu", dtype=torch.float64)
    if p_mat is None:
        return delta_64
    p_dim = int(p_mat.shape[0])
    if delta_64.ndim == 1:
        if int(delta_64.shape[0]) != p_dim:
            raise ValueError(f"1D delta dim {delta_64.shape[0]} != P_0 dim {p_dim} on {expert_key}.{sub_key}")
        return p_mat @ delta_64
    if delta_64.ndim != 2:
        raise ValueError(f"Expected 1D or 2D delta for P_0 projection, got ndim={delta_64.ndim}")

    d_out, d_in = int(delta_64.shape[0]), int(delta_64.shape[1])
    if sub_key.endswith("_B"):
        if d_out == p_dim:
            return p_mat @ delta_64
        if d_in == p_dim:
            return delta_64 @ p_mat
    else:
        if d_in == p_dim:
            return delta_64 @ p_mat
        if d_out == p_dim:
            return p_mat @ delta_64
    raise ValueError(
        f"P_0 shape {tuple(p_mat.shape)} incompatible with delta shape {tuple(delta_64.shape)} on {expert_key}.{sub_key}"
    )


@torch.no_grad()
def distill_forager_nectar_to_royal_jelly(
    base_experts: Mapping[str | int, Any],
    forager_bee_deltas: Sequence[Mapping[str | int, Any]] | Mapping[str | int, Any],
    p_null_by_expert: Mapping[str | int, Any] | None,
    *,
    w_norm_base: torch.Tensor,
    normed_unit_probe: torch.Tensor | Sequence[torch.Tensor],
    head_weight: torch.Tensor,
    probe_targets: torch.Tensor | Sequence[torch.Tensor],
    is_general_mask: torch.Tensor | Sequence[bool] | np.ndarray,
    ties_density: float = DEFAULT_TIES_DENSITY,
    ties_sign_mode: str = "mass",
    bee_weights: Sequence[float] | None = None,
    max_peer_norm_ratio: float | None = 4.0,
    sink_experts: Sequence[str | int] = (),
    anchor_experts: Sequence[str | int] = (),
    damping: float = DEFAULT_FISHER_DAMPING,
    step_scales: Sequence[float] = DEFAULT_STEP_SCALES,
    tau_grid: Sequence[float] = DEFAULT_TAU_GRID,
    max_rel_delta: float = DEFAULT_MAX_REL_DELTA,
    max_d0_rel: float = DEFAULT_MAX_D0_REL_PROBE,
    ece_weight: float = DEFAULT_ECE_ENERGY_WEIGHT,
    sv_rel_tol: float = 1e-6,
    b_norm_base: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Purify raw Forager Bee nectar into Royal Jelly via TIES + ``P_0`` + ``ker(G_D0)`` Fisher norm."""
    density_f = _require_float(ties_density, "ties_density", min_value=0.0, strict_min=True, max_value=1.0)
    if ties_sign_mode not in {"mass", "majority"}:
        raise ValueError("ties_sign_mode must be 'mass' or 'majority'")
    if isinstance(sink_experts, (str, bytes)) or not isinstance(sink_experts, (Sequence, set, frozenset)):
        raise TypeError("sink_experts must be a sequence or set")
    if isinstance(anchor_experts, (str, bytes)) or not isinstance(anchor_experts, (Sequence, set, frozenset)):
        raise TypeError("anchor_experts must be a sequence or set")

    base_norm, key_order, orig_keys, single_flags = _normalize_expert_store(base_experts, "base_experts")
    bees_norm, _ = _normalize_bee_deltas(forager_bee_deltas, base_norm, single_flags)

    if bee_weights is not None:
        if isinstance(bee_weights, (str, bytes)) or not isinstance(bee_weights, Sequence):
            raise TypeError("bee_weights must be a sequence of floats or None")
        if len(bee_weights) != len(bees_norm):
            raise ValueError(f"bee_weights length {len(bee_weights)} != {len(bees_norm)}")
        b_weights_norm = [_require_float(w, f"bee_weights[{i}]", min_value=0.0) for i, w in enumerate(bee_weights)]
        if sum(b_weights_norm) <= 0.0:
            raise ValueError("bee_weights must have a strictly positive sum")
    else:
        b_weights_norm = [1.0] * len(bees_norm)

    sink_tokens = {str(x).strip() for x in sink_experts}
    anchor_tokens = {str(x).strip() for x in anchor_experts}

    def _matches_role(ek: str, role_tokens: set[str]) -> bool:
        tail = _parse_expert_idx(ek)
        return ek in role_tokens or (tail is not None and (str(tail) in role_tokens or f"expert_{tail}" in role_tokens))

    royal_jelly_experts: dict[Any, Any] = {}
    expert_reports: dict[str, dict[str, Any]] = {}
    consensus_scores: list[float] = []

    for ek in key_order:
        exp_base = base_norm[ek]
        orig_k = orig_keys[ek]
        if _matches_role(ek, sink_tokens):
            updated = {
                sk: (torch.zeros_like(t) if sk.endswith("_B") else t.clone())
                for sk, t in exp_base.items()
            }
            expert_reports[ek] = {"action": "ghost_sunk", "sign_consensus_ratio": 1.0}
        elif _matches_role(ek, anchor_tokens):
            updated = {sk: t.clone() for sk, t in exp_base.items()}
            expert_reports[ek] = {"action": "anchor_frozen", "sign_consensus_ratio": 1.0}
        else:
            updated = {}
            sub_cons: list[float] = []
            norm_before_sq = 0.0
            norm_after_sq = 0.0
            for sk, base_t in exp_base.items():
                peer_tensors: list[torch.Tensor] = []
                peer_ws: list[float] = []
                for bee_map, bw in zip(bees_norm, b_weights_norm, strict=True):
                    if ek in bee_map and sk in bee_map[ek]:
                        peer_tensors.append(bee_map[ek][sk])
                        peer_ws.append(bw)
                if not peer_tensors:
                    updated[sk] = base_t.clone()
                    continue
                ties_delta, ties_meta = trim_and_elect_sign_ties(
                    peer_tensors,
                    density_top_p=density_f,
                    peer_weights=peer_ws,
                    sign_election_mode=ties_sign_mode,
                    max_peer_norm_ratio=max_peer_norm_ratio,
                )
                sub_cons.append(float(ties_meta["sign_consensus_ratio"]))
                p_mat = _resolve_expert_projector(
                    p_null_by_expert,
                    ek,
                    sk,
                    delta_shape=tuple(ties_delta.shape),
                    is_single_tensor=single_flags[ek],
                )
                proj_delta = _project_delta_with_p0(ties_delta, p_mat, sk, ek)
                norm_before_sq += float(ties_delta.to(torch.float64).pow(2).sum().item())
                norm_after_sq += float(proj_delta.pow(2).sum().item())
                updated[sk] = (base_t.detach().to(device="cpu", dtype=torch.float64) + proj_delta).to(
                    device=base_t.device, dtype=base_t.dtype
                )
            mean_c = float(np.mean(sub_cons)) if sub_cons else 1.0
            consensus_scores.extend(sub_cons)
            expert_reports[ek] = {
                "action": "royal_jelly_distilled" if sub_cons else "unchanged",
                "sign_consensus_ratio": mean_c,
                "delta_norm_before_p0": math.sqrt(norm_before_sq),
                "delta_norm_after_p0": math.sqrt(norm_after_sq),
            }
        royal_jelly_experts[orig_k] = updated[_SINGLE_TENSOR_KEY] if single_flags[ek] else updated

    w_cal, b_cal, fisher_meta = compute_ker_g_d0_fisher_norm_step(
        w_norm_base,
        normed_unit_probe,
        head_weight,
        probe_targets,
        is_general_mask,
        b_norm_base=b_norm_base,
        damping=damping,
        step_scales=step_scales,
        tau_grid=tau_grid,
        max_rel_delta=max_rel_delta,
        max_d0_rel=max_d0_rel,
        ece_weight=ece_weight,
        sv_rel_tol=sv_rel_tol,
    )

    return {
        "schema": QUEEN_CORTEX_SCHEMA,
        "royal_jelly_experts": royal_jelly_experts,
        "w_norm_calibrated": w_cal,
        "b_norm_calibrated": b_cal,
        "ties_density": density_f,
        "mean_sign_consensus_ratio": float(np.mean(consensus_scores)) if consensus_scores else 1.0,
        "expert_reports": expert_reports,
        **fisher_meta,
    }


# ---------------------------------------------------------------------------
# 3. Component (b): Signed QueenPheromoneDirective Planner
# ---------------------------------------------------------------------------


def emit_queen_pheromone_campaign_directive(
    expert_causal_gains_nats: Mapping[str | int, float],
    expert_prose_harm_nats: Mapping[str | int, float],
    router_expert_kl_mismatch: Mapping[str | int, float] | float,
    domain_nlls: Mapping[str, float],
    domain_eces: Mapping[str, float],
    *,
    anchor_gain_floor_nats: float = DEFAULT_ANCHOR_GAIN_FLOOR_NATS,
    coadapt_kl_ceiling: float = DEFAULT_COADAPT_KL_CEILING,
    temperature: float = DEFAULT_FORAGING_TEMPERATURE,
    prose_harm_sink_floor_nats: float = DEFAULT_PROSE_HARM_SINK_FLOOR_NATS,
    domain_round_counts: Mapping[str, int] | None = None,
    ucb_exploration_coeff: float = DEFAULT_UCB_EXPLORATION_COEFF,
    general_prefixes: Sequence[str] = DEFAULT_GENERAL_PREFIXES,
    campaign_id: str = "la-ruche-reine-cortex",
    round_index: int = 1,
) -> QueenPheromoneDirective:
    """Emit a signed ``QueenPheromoneDirective`` steering next-round expert & bee behavior."""
    if not isinstance(expert_causal_gains_nats, Mapping) or not expert_causal_gains_nats:
        raise TypeError("expert_causal_gains_nats must be a non-empty Mapping")
    if not isinstance(expert_prose_harm_nats, Mapping) or not expert_prose_harm_nats:
        raise TypeError("expert_prose_harm_nats must be a non-empty Mapping")
    if not isinstance(domain_nlls, Mapping) or not domain_nlls:
        raise TypeError("domain_nlls must be a non-empty Mapping")
    if not isinstance(domain_eces, Mapping) or not domain_eces:
        raise TypeError("domain_eces must be a non-empty Mapping")

    anchor_floor = _require_float(anchor_gain_floor_nats, "anchor_gain_floor_nats", max_value=0.0)
    kl_ceil = _require_float(coadapt_kl_ceiling, "coadapt_kl_ceiling", min_value=0.0)
    temp_f = _require_float(temperature, "temperature", min_value=0.0, strict_min=True)
    harm_floor = _require_float(prose_harm_sink_floor_nats, "prose_harm_sink_floor_nats", min_value=0.0)
    ucb_c = _require_float(ucb_exploration_coeff, "ucb_exploration_coeff", min_value=0.0)
    camp_s = _require_nonempty_str(campaign_id, "campaign_id")
    r_idx = _require_int(round_index, "round_index", min_value=1)

    gains_clean = {str(k).strip(): _require_float(v, f"gain[{k!r}]") for k, v in expert_causal_gains_nats.items()}
    harms_clean = {str(k).strip(): _require_float(v, f"harm[{k!r}]") for k, v in expert_prose_harm_nats.items()}
    if set(gains_clean.keys()) != set(harms_clean.keys()):
        raise ValueError("expert_causal_gains_nats and expert_prose_harm_nats must share identical keys")

    if isinstance(router_expert_kl_mismatch, Mapping):
        if not router_expert_kl_mismatch:
            raise ValueError("router_expert_kl_mismatch mapping must not be empty")
        kl_vals = [
            _require_float(v, f"kl[{k!r}]", min_value=0.0) for k, v in router_expert_kl_mismatch.items()
        ]
        max_kl = max(kl_vals)
    else:
        max_kl = _require_float(router_expert_kl_mismatch, "router_expert_kl_mismatch", min_value=0.0)

    nlls_clean = {
        _require_nonempty_str(k, "domain_nlls key"): _require_float(v, f"domain_nlls[{k!r}]", min_value=0.0)
        for k, v in domain_nlls.items()
    }
    eces_clean = {
        _require_nonempty_str(k, "domain_eces key"): _require_float(v, f"domain_eces[{k!r}]", min_value=0.0)
        for k, v in domain_eces.items()
    }
    if set(nlls_clean.keys()) != set(eces_clean.keys()):
        raise ValueError("domain_nlls and domain_eces must share identical domain keys")

    anchors: list[str] = []
    corruptors: list[str] = []
    active: list[str] = []
    for ek in sorted(gains_clean.keys()):
        gain_v = gains_clean[ek]
        harm_v = harms_clean[ek]
        if (harm_v > harm_floor and gain_v > anchor_floor) or (gain_v > harm_floor and harm_v >= 0.0):
            corruptors.append(ek)
        elif gain_v <= anchor_floor and harm_v <= harm_floor:
            anchors.append(ek)
        else:
            active.append(ek)

    router_mode = "router_frozen" if max_kl > kl_ceil else "router_unfrozen_coadapted"

    specialist_domains = [d for d in sorted(nlls_clean.keys()) if not _is_general_domain(d, general_prefixes)]
    target_domains = specialist_domains if specialist_domains else sorted(nlls_clean.keys())

    counts_map: dict[str, int] = {}
    if domain_round_counts is not None:
        if not isinstance(domain_round_counts, Mapping):
            raise TypeError("domain_round_counts must be a Mapping or None")
        counts_map = {
            str(d): _require_int(domain_round_counts.get(d, 0), f"domain_round_counts[{d!r}]", min_value=0)
            for d in target_domains
        }
    total_visits = max(1, sum(counts_map.values()))

    raw_deficits: dict[str, float] = {}
    ucb_bonuses: dict[str, float] = {}
    for dom in target_domains:
        bonus = (
            ucb_c * math.sqrt(math.log(float(total_visits) + 1.0) / float(counts_map[dom] + 1))
            if domain_round_counts is not None
            else 0.0
        )
        ucb_bonuses[dom] = bonus
        raw_deficits[dom] = nlls_clean[dom] + 0.25 * eces_clean[dom] + bonus

    max_def = max(raw_deficits.values())
    exp_logits = {d: math.exp((raw_deficits[d] - max_def) / temp_f) for d in target_domains}
    z_sum = sum(exp_logits.values())
    weights_by_domain = {d: (0.0 if d not in exp_logits else exp_logits[d] / z_sum) for d in sorted(nlls_clean)}

    priorities = tuple(
        {
            "domain_id": dom,
            "nll_mean": nlls_clean[dom],
            "ece_value": eces_clean[dom],
            "ucb_bonus": ucb_bonuses.get(dom, 0.0),
            "foraging_priority_weight": weights_by_domain[dom],
        }
        for dom in sorted(nlls_clean.keys(), key=lambda d: (-weights_by_domain[d], d))
    )

    core_payload = {
        "object": QUEEN_PHEROMONE_OBJECT,
        "schema": QUEEN_CORTEX_SCHEMA,
        "campaign_id": camp_s,
        "round_index": r_idx,
        "anchor_experts_to_freeze": anchors,
        "corruptor_experts_to_sink_or_reseed": corruptors,
        "active_specialist_experts": active,
        "router_training_mode": router_mode,
        "max_router_expert_kl_mismatch": max_kl,
        "coadapt_kl_ceiling": kl_ceil,
        "bee_foraging_domain_weights": weights_by_domain,
        "domain_priorities": [dict(row) for row in priorities],
    }
    sig = compute_payload_sha256(core_payload)
    return QueenPheromoneDirective(
        object=QUEEN_PHEROMONE_OBJECT,
        schema=QUEEN_CORTEX_SCHEMA,
        campaign_id=camp_s,
        round_index=r_idx,
        anchor_experts_to_freeze=tuple(anchors),
        corruptor_experts_to_sink_or_reseed=tuple(corruptors),
        active_specialist_experts=tuple(active),
        router_training_mode=router_mode,
        max_router_expert_kl_mismatch=max_kl,
        coadapt_kl_ceiling=kl_ceil,
        bee_foraging_domain_weights=weights_by_domain,
        domain_priorities=priorities,
        directive_sha256=sig,
    )


# ---------------------------------------------------------------------------
# 4. Component (c): Active Closed-Loop Campaign Governor & ANSE Gate
# ---------------------------------------------------------------------------


def _evaluate_model_on_chunks(
    model: nn.Module,
    chunks: Sequence[Mapping[str, Any]],
    *,
    score_from: int,
    general_prefixes: Sequence[str],
) -> dict[str, Any]:
    ids_all = torch.tensor([c["ids"] for c in chunks], dtype=torch.long)
    start_pos = max(0, min(int(ids_all.shape[1]) - 2, score_from - 1))
    cos = model.cos_cached.to(ids_all.device)
    sin = model.sin_cached.to(ids_all.device)

    xb = compute_embeddings_batch(model, ids_all)
    for blk in model.blocks:
        xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
        xb = forward_block_post_mlp(blk, xm, mi)

    h_scored = xb[:, start_pos:-1, :].to(torch.float32)
    ln_mod = getattr(model, "ln", None) or getattr(model, "norm", None)
    eps_attr = getattr(ln_mod, "eps", None)
    ln_eps = float(eps_attr if eps_attr is not None else getattr(ln_mod, "variance_epsilon", 1e-5))
    ln_w = ln_mod.weight.to(torch.float32)
    ln_b = ln_mod.bias.to(torch.float32) if getattr(ln_mod, "bias", None) is not None else None
    if ln_b is not None:
        h_norm = F.layer_norm(h_scored, (int(h_scored.shape[-1]),), ln_w, ln_b, eps=ln_eps)
    else:
        rms = torch.rsqrt(h_scored.pow(2).mean(dim=-1, keepdim=True) + ln_eps)
        h_norm = h_scored * rms * ln_w
    head_mod = getattr(model, "head", None) or getattr(model, "lm_head", None)
    logits = F.linear(h_norm.to(head_mod.weight.dtype), head_mod.weight).to(torch.float32)
    targets = ids_all[:, start_pos + 1 :]

    overall = compute_token_nll_and_ece(logits, targets)
    dom_indices: dict[str, list[int]] = {}
    for idx, ch in enumerate(chunks):
        dom = str(ch.get("expert_id", ch.get("domain", "unknown")))
        dom_indices.setdefault(dom, []).append(idx)

    domain_nlls: dict[str, float] = {}
    domain_eces: dict[str, float] = {}
    for dom, idxs in sorted(dom_indices.items()):
        sub = compute_token_nll_and_ece(logits[idxs], targets[idxs])
        domain_nlls[dom] = float(sub["nll_mean"])
        domain_eces[dom] = float(sub["expected_calibration_error"])

    gen_doms = [d for d in domain_nlls if _is_general_domain(d, general_prefixes)]
    gen_nll = float(np.mean([domain_nlls[d] for d in gen_doms])) if gen_doms else float(overall["nll_mean"])
    return {
        "all_finite": True,
        "nll_mean": float(overall["nll_mean"]),
        "ece_value": float(overall["expected_calibration_error"]),
        "general_nll": gen_nll,
        "domain_nlls": domain_nlls,
        "domain_eces": domain_eces,
    }


def _normalize_eval_result(
    raw: Any,
    general_prefixes: Sequence[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or not raw:
        raise TypeError(f"{label} evaluation result must be a non-empty Mapping")
    if "domain_nlls" in raw:
        dom_raw = raw["domain_nlls"]
        if not isinstance(dom_raw, Mapping) or not dom_raw:
            raise ValueError(f"{label}.domain_nlls must be a non-empty Mapping")
        domain_nlls = {str(k): _require_float(v, f"{label}.domain_nlls[{k!r}]") for k, v in dom_raw.items()}
        dom_ece_raw = raw.get("domain_eces", {})
        domain_eces = {
            d: _require_float(dom_ece_raw.get(d, raw.get("ece_value", 0.0)), f"{label}.domain_eces[{d!r}]", min_value=0.0)
            for d in domain_nlls
        }
        nll_mean = (
            _require_float(raw["nll_mean"], f"{label}.nll_mean")
            if "nll_mean" in raw
            else float(np.mean(list(domain_nlls.values())))
        )
        ece_val = _require_float(raw.get("ece_value", np.mean(list(domain_eces.values()))), f"{label}.ece_value", min_value=0.0)
    else:
        domain_nlls = {str(k): _require_float(v, f"{label}[{k!r}]") for k, v in raw.items()}
        domain_eces = {d: 0.0 for d in domain_nlls}
        nll_mean = float(np.mean(list(domain_nlls.values())))
        ece_val = 0.0

    gen_doms = [d for d in domain_nlls if _is_general_domain(d, general_prefixes)]
    if not gen_doms:
        raise ValueError(f"{label} must contain at least one general-prose domain matching {general_prefixes}")
    gen_nll = float(np.mean([domain_nlls[d] for d in gen_doms]))
    return {
        "all_finite": True,
        "nll_mean": nll_mean,
        "ece_value": ece_val,
        "general_nll": gen_nll,
        "domain_nlls": domain_nlls,
        "domain_eces": domain_eces,
    }


def _score_against_base(
    cand: Mapping[str, Any],
    base: Mapping[str, Any],
    cfg: QueenCortexConfig,
) -> dict[str, Any]:
    if not cand.get("all_finite", True):
        return {
            "general_drift_ratio": float("inf"),
            "worst_domain_drift_ratio": float("inf"),
            "guard_ok": False,
            "anse_energy": cfg.maximum_pain_energy,
        }
    gen_ratio = float(cand["general_nll"]) / max(1e-9, float(base["general_nll"]))
    worst_ratio = max(
        float(cand["domain_nlls"][d]) / max(1e-9, float(base["domain_nlls"][d]))
        for d in cand["domain_nlls"]
        if d in base["domain_nlls"]
    )
    guard_ok = bool(gen_ratio <= cfg.max_domain_drift_ratio and worst_ratio <= cfg.max_domain_drift_ratio)
    mean_diff = float(cand["nll_mean"]) - float(base["nll_mean"])
    ece_diff = float(cand["ece_value"]) - float(base["ece_value"])
    margin_pen = cfg.soft_margin_penalty if (guard_ok and gen_ratio > cfg.anse_safe_general_ratio) else 0.0
    anse = (mean_diff + cfg.ece_energy_weight * ece_diff + margin_pen) if guard_ok else cfg.maximum_pain_energy
    return {
        "general_drift_ratio": gen_ratio,
        "worst_domain_drift_ratio": worst_ratio,
        "guard_ok": guard_ok,
        "anse_energy": float(anse),
    }


def _validate_probe_chunks(
    chunks: Any,
    *,
    vocab_size: int,
    general_prefixes: Sequence[str],
    name: str,
) -> list[dict[str, Any]]:
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence) or not chunks:
        raise ValueError(f"{name} must be a non-empty sequence of chunk dicts")
    out: list[dict[str, Any]] = []
    has_gen = False
    seq_len: int | None = None
    for idx, ch in enumerate(chunks):
        if not isinstance(ch, Mapping):
            raise TypeError(f"{name}[{idx}] must be a Mapping")
        dom = _require_nonempty_str(ch.get("expert_id", ch.get("domain")), f"{name}[{idx}] domain")
        ids = ch.get("ids")
        if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence) or len(ids) < 2:
            raise ValueError(f"{name}[{idx}]['ids'] must be a sequence of >= 2 token IDs")
        for tid in ids:
            _require_int(tid, f"{name}[{idx}] token_id", min_value=0)
            if tid >= vocab_size:
                raise ValueError(f"{name}[{idx}] token_id {tid} >= vocab_size {vocab_size}")
        if seq_len is None:
            seq_len = len(ids)
        elif len(ids) != seq_len:
            raise ValueError(f"{name}[{idx}] sequence length {len(ids)} != {seq_len}")
        has_gen = has_gen or _is_general_domain(dom, general_prefixes)
        out.append({"expert_id": dom, "ids": [int(x) for x in ids]})
    if not has_gen:
        raise ValueError(f"{name} must contain at least one general-prose chunk")
    return out


def _build_model_p_null_and_probe(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    score_from: int,
    general_prefixes: Sequence[str],
    build_projectors: bool,
) -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, torch.Tensor]:
    ids_all = torch.tensor([c["ids"] for c in probe_chunks], dtype=torch.long)
    start_pos = max(0, min(int(ids_all.shape[1]) - 2, score_from - 1))
    is_gen = torch.tensor(
        [_is_general_domain(str(c["expert_id"]), general_prefixes) for c in probe_chunks],
        dtype=torch.bool,
    )
    cos = model.cos_cached.to(ids_all.device)
    sin = model.sin_cached.to(ids_all.device)
    xb = compute_embeddings_batch(model, ids_all)

    p_null: dict[str, dict[str, torch.Tensor]] = {}
    for l_idx, blk in enumerate(model.blocks):
        xm, mi = forward_block_pre_mlp(blk, xb, cos, sin)
        if build_projectors and isinstance(blk.mlp, CrowdLoRAMoEMLP):
            x_prose = mi[:, start_pos:-1, :][is_gen].reshape(-1, int(mi.shape[-1])).contiguous()
            p_x, _ = compute_prose_nullspace_projector_scaled(
                x_prose, sv_rel_tol=1e-3, variance_threshold=None
            )
            base_m = blk.mlp.base_mlp
            xp_cast = x_prose.to(dtype=base_m.w1.weight.dtype)
            act_p = F.silu(base_m.w1(xp_cast).float()) * base_m.w2(xp_cast).float()
            p_act, _ = compute_prose_nullspace_projector_scaled(
                act_p, sv_rel_tol=1e-3, variance_threshold=None
            )
            for e_key in sorted(blk.mlp.experts.keys(), key=int):
                p_null[f"blocks.{l_idx}.mlp.experts.{e_key}"] = {"x": p_x, "act": p_act}
        xb = forward_block_post_mlp(blk, xm, mi)

    h_scored = xb[:, start_pos:-1, :].to(torch.float32)
    ln_mod = getattr(model, "ln", None) or getattr(model, "norm", None)
    eps_attr = getattr(ln_mod, "eps", None)
    ln_eps = float(eps_attr if eps_attr is not None else getattr(ln_mod, "variance_epsilon", 1e-5))
    if getattr(ln_mod, "bias", None) is not None:
        unit_probe = F.layer_norm(h_scored, (int(h_scored.shape[-1]),), None, None, eps=ln_eps)
    else:
        unit_probe = h_scored * torch.rsqrt(h_scored.pow(2).mean(dim=-1, keepdim=True) + ln_eps)
    return p_null, unit_probe, ids_all[:, start_pos + 1 :], is_gen


def _compute_loeo_expert_metrics(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    score_from: int,
    general_prefixes: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    base_ev = _evaluate_model_on_chunks(
        model, probe_chunks, score_from=score_from, general_prefixes=general_prefixes
    )
    saved = copy.deepcopy(model.state_dict())
    expert_ids: set[int] = set()
    for blk in model.blocks:
        mlp = getattr(blk, "mlp", None)
        if isinstance(mlp, CrowdLoRAMoEMLP):
            expert_ids.update(int(k) for k in mlp.experts.keys())

    gains: dict[str, float] = {}
    harms: dict[str, float] = {}
    for e_idx in sorted(expert_ids):
        for blk in model.blocks:
            mlp = getattr(blk, "mlp", None)
            if isinstance(mlp, CrowdLoRAMoEMLP) and str(e_idx) in mlp.experts:
                exp = mlp.experts[str(e_idx)]
                exp.w1_B.zero_()
                exp.w2_B.zero_()
                exp.w3_B.zero_()
        ko_ev = _evaluate_model_on_chunks(
            model, probe_chunks, score_from=score_from, general_prefixes=general_prefixes
        )
        # Negative gain = expert lowers overall NLL; positive harm = expert raises D0 NLL
        gains[f"expert_{e_idx}"] = float(base_ev["nll_mean"]) - float(ko_ev["nll_mean"])
        harms[f"expert_{e_idx}"] = float(base_ev["general_nll"]) - float(ko_ev["general_nll"])
        model.load_state_dict(saved)
    return gains, harms


def _partition_checkpoint_blocks(
    blocks: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
    experts: dict[str, dict[str, torch.Tensor]] = {}
    routers: dict[str, dict[str, torch.Tensor]] = {}
    norms: dict[str, dict[str, torch.Tensor]] = {}
    for bid, sub in blocks.items():
        if _is_final_ln_key(bid):
            norms[bid] = dict(sub)
        elif bid.endswith(_ROUTER_SUFFIX):
            routers[bid] = dict(sub)
        else:
            experts[bid] = dict(sub)
    if not experts:
        raise ValueError("Checkpoint must contain at least one trainable expert block")
    return experts, routers, norms


def _partition_bee_deltas(
    forager_bee_deltas: Any,
    expert_keys: set[str],
    router_keys: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, dict[str, torch.Tensor]]], bool]:
    if isinstance(forager_bee_deltas, Mapping):
        raw_bees = list(forager_bee_deltas.values())
    elif isinstance(forager_bee_deltas, Sequence) and not isinstance(forager_bee_deltas, (str, bytes)):
        raw_bees = list(forager_bee_deltas)
    else:
        raise TypeError("forager_bee_deltas must be a non-empty Sequence or Mapping")
    if not raw_bees:
        raise ValueError("forager_bee_deltas must not be empty")

    exp_bees: list[dict[str, Any]] = []
    router_bees: list[dict[str, dict[str, torch.Tensor]]] = []
    all_finite = True
    for idx, bee in enumerate(raw_bees):
        if not isinstance(bee, Mapping) or not bee:
            raise TypeError(f"forager_bee_deltas[{idx}] must be a non-empty Mapping")
        eb: dict[str, Any] = {}
        rb: dict[str, dict[str, torch.Tensor]] = {}
        for k, v in bee.items():
            ks = str(k).strip()
            tensors_to_check = [v] if isinstance(v, torch.Tensor) else (list(v.values()) if isinstance(v, Mapping) else [])
            for t in tensors_to_check:
                if isinstance(t, torch.Tensor) and t.is_floating_point() and not bool(torch.isfinite(t).all().item()):
                    all_finite = False
            if ks in router_keys:
                rb[ks] = dict(v) if isinstance(v, Mapping) else {"weight": v}
            elif ks in expert_keys or not _is_final_ln_key(ks):
                eb[ks] = v
        exp_bees.append(eb)
        router_bees.append(rb)
    return exp_bees, router_bees, all_finite


@torch.no_grad()
def execute_queen_cortex_campaign_round(
    base_checkpoint: nn.Module | Mapping[str, Any],
    forager_bee_deltas: Sequence[Mapping[str | int, Any]] | Mapping[str | int, Any],
    *,
    p_null_by_expert: Mapping[str | int, Any] | None = None,
    parent_checkpoint: nn.Module | Mapping[str, Any] | None = None,
    probe_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_fn: Callable[[Any], Mapping[str, Any]] | None = None,
    w_norm_base: torch.Tensor | None = None,
    normed_unit_probe: torch.Tensor | Sequence[torch.Tensor] | None = None,
    head_weight: torch.Tensor | None = None,
    probe_targets: torch.Tensor | Sequence[torch.Tensor] | None = None,
    is_general_mask: torch.Tensor | Sequence[bool] | np.ndarray | None = None,
    expert_causal_gains_nats: Mapping[str | int, float] | None = None,
    expert_prose_harm_nats: Mapping[str | int, float] | None = None,
    router_expert_kl_mismatch: Mapping[str | int, float] | float = 0.0,
    previous_directive: QueenPheromoneDirective | Mapping[str, Any] | None = None,
    domain_round_counts: Mapping[str, int] | None = None,
    parent_anse_energy: float = 0.0,
    campaign_id: str = "la-ruche-reine-cortex",
    round_index: int = 1,
    config: QueenCortexConfig | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Execute one active Reine-Cortex MoE campaign round under ANSE & D0 guardrails."""
    cfg = config if config is not None else QueenCortexConfig()
    if not isinstance(cfg, QueenCortexConfig):
        raise TypeError(f"config must be QueenCortexConfig or None, got {type(cfg).__name__}")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError(f"eval_fn must be callable or None, got {type(eval_fn).__name__}")
    _require_float(parent_anse_energy, "parent_anse_energy")
    camp_s = _require_nonempty_str(campaign_id, "campaign_id")
    r_idx = _require_int(round_index, "round_index", min_value=1)

    is_model = isinstance(base_checkpoint, nn.Module)
    if is_model:
        work_model = base_checkpoint if cfg.inplace else copy.deepcopy(base_checkpoint)
        base_blocks = extract_checkpoint_blocks(base_checkpoint)
        base_sha = compute_checkpoint_sha256(base_checkpoint)
    elif isinstance(base_checkpoint, Mapping):
        work_model = None
        base_blocks, _, _, _ = _normalize_expert_store(base_checkpoint, "base_checkpoint")
        base_sha = compute_checkpoint_sha256(base_blocks)
    else:
        raise TypeError("base_checkpoint must be an nn.Module or non-empty Mapping")

    if parent_checkpoint is None:
        parent_blocks = {bid: {sk: t.clone() for sk, t in sub.items()} for bid, sub in base_blocks.items()}
        parent_sha = base_sha
    elif isinstance(parent_checkpoint, nn.Module):
        parent_blocks = extract_checkpoint_blocks(parent_checkpoint)
        parent_sha = compute_checkpoint_sha256(parent_checkpoint)
    elif isinstance(parent_checkpoint, Mapping):
        parent_blocks, _, _, _ = _normalize_expert_store(parent_checkpoint, "parent_checkpoint")
        parent_sha = compute_checkpoint_sha256(parent_blocks)
    else:
        raise TypeError("parent_checkpoint must be an nn.Module, Mapping, or None")

    exp_parent, rtr_parent, ln_parent = _partition_checkpoint_blocks(parent_blocks)
    _, _, ln_base = _partition_checkpoint_blocks(base_blocks)
    exp_bees, router_bees, bees_all_finite = _partition_bee_deltas(
        forager_bee_deltas, set(exp_parent.keys()), set(rtr_parent.keys())
    )

    head_mod = None
    if is_model:
        head_mod = getattr(work_model, "head", None) or getattr(work_model, "lm_head", None)
        ln_mod = getattr(work_model, "ln", None) or getattr(work_model, "norm", None)
        if head_mod is None or ln_mod is None:
            raise ValueError("Model must expose 'head'/'lm_head' and 'ln'/'norm'")
        vocab_sz = int(head_mod.weight.shape[0])
        p_chunks = (
            _validate_probe_chunks(
                probe_chunks, vocab_size=vocab_sz, general_prefixes=cfg.general_prefixes, name="probe_chunks"
            )
            if probe_chunks is not None
            else None
        )
        e_chunks = (
            _validate_probe_chunks(
                eval_chunks, vocab_size=vocab_sz, general_prefixes=cfg.general_prefixes, name="eval_chunks"
            )
            if eval_chunks is not None
            else p_chunks
        )
        if eval_fn is None and e_chunks is None:
            raise ValueError("Model mode requires probe_chunks/eval_chunks or eval_fn")
    else:
        p_chunks, e_chunks = None, None
        if eval_fn is None:
            raise ValueError("Dict checkpoint mode requires eval_fn")

    def _eval_blocks(blk_state: Mapping[str, Mapping[str, torch.Tensor]], label: str) -> dict[str, Any]:
        if eval_fn is not None:
            if work_model is not None:
                apply_checkpoint_blocks_inplace(work_model, blk_state)
                raw = eval_fn(work_model)
            else:
                raw = eval_fn(blk_state)
            return _normalize_eval_result(raw, cfg.general_prefixes, label)
        assert work_model is not None and e_chunks is not None
        apply_checkpoint_blocks_inplace(work_model, blk_state)
        return _evaluate_model_on_chunks(
            work_model, e_chunks, score_from=cfg.score_from, general_prefixes=cfg.general_prefixes
        )

    base_eval = _eval_blocks(base_blocks, "base_eval")
    rollback_target_blocks = base_blocks if cfg.rollback_target == "base" else parent_blocks
    rollback_expected_sha = base_sha if cfg.rollback_target == "base" else parent_sha

    # Evaluate raw unguarded mean bee delta on top of parent to expose raw drift
    if not bees_all_finite:
        raw_eval: dict[str, Any] = {"all_finite": False, "nll_mean": float("inf"), "ece_value": 1.0}
        raw_metrics = _score_against_base(raw_eval, base_eval, cfg)
    else:
        raw_blocks = {bid: {sk: t.clone() for sk, t in sub.items()} for bid, sub in parent_blocks.items()}
        for target_map in (exp_bees, router_bees):
            for bid in raw_blocks:
                for sk in raw_blocks[bid]:
                    acc = [b[bid][sk].detach().to(torch.float64) for b in target_map if bid in b and sk in b[bid]]
                    if acc:
                        mean_d = torch.stack(acc, dim=0).mean(dim=0)
                        raw_blocks[bid][sk] = (raw_blocks[bid][sk].to(torch.float64) + mean_d).to(
                            raw_blocks[bid][sk].dtype
                        )
        raw_eval = _eval_blocks(raw_blocks, "raw_unguarded_eval")
        raw_metrics = _score_against_base(raw_eval, base_eval, cfg)

    anchor_list: tuple[str, ...] = ()
    sink_list: tuple[str, ...] = ()
    freeze_router = False
    if previous_directive is not None:
        d_map = previous_directive.to_dict() if isinstance(previous_directive, QueenPheromoneDirective) else previous_directive
        anchor_list = tuple(str(x) for x in d_map.get("anchor_experts_to_freeze", ()))
        sink_list = tuple(str(x) for x in d_map.get("corruptor_experts_to_sink_or_reseed", ()))
        freeze_router = str(d_map.get("router_training_mode", "")) == "router_frozen"

    if not bees_all_finite:
        # Byzantine non-finite nectar triggers immediate ANSE Maximum Pain atomic rollback
        cand_ok = False
        cand_eval = {"all_finite": False, "nll_mean": float("inf"), "ece_value": 1.0, "domain_nlls": dict(base_eval["domain_nlls"]), "domain_eces": dict(base_eval["domain_eces"])}
        cand_metrics = _score_against_base(cand_eval, base_eval, cfg)
        distill_summary: dict[str, Any] = {"status": "QUARANTINED_NONFINITE_NECTAR"}
        cand_blocks = rollback_target_blocks
        cand_sha = rollback_expected_sha
    else:
        eff_p_null = p_null_by_expert
        if work_model is not None:
            apply_checkpoint_blocks_inplace(work_model, parent_blocks)
            if eff_p_null is None and p_chunks is not None:
                eff_p_null, _, _, _ = _build_model_p_null_and_probe(
                    work_model,
                    p_chunks,
                    score_from=cfg.score_from,
                    general_prefixes=cfg.general_prefixes,
                    build_projectors=True,
                )

        # Resolve norm/head/probe tensors for distillation
        w_n_b = w_norm_base
        b_n_b: torch.Tensor | None = None
        h_w = head_weight
        if work_model is not None:
            ln_key = next(iter(ln_base))
            if w_n_b is None:
                w_n_b = ln_base[ln_key]["weight"]
                b_n_b = ln_base[ln_key].get("bias")
            if h_w is None and head_mod is not None:
                h_w = head_mod.weight
        assert w_n_b is not None and h_w is not None, "w_norm_base and head_weight are required"

        # Pre-distill expert blocks via TIES + P_0, then refresh probe hidden states if in model mode
        if work_model is not None and p_chunks is not None and normed_unit_probe is None:
            # First run expert distillation using current parent unit probe, then refresh pre-LN activations!
            _, init_unit, init_tgt, init_gen = _build_model_p_null_and_probe(
                work_model, p_chunks, score_from=cfg.score_from, general_prefixes=cfg.general_prefixes, build_projectors=False
            )
            stage1 = distill_forager_nectar_to_royal_jelly(
                exp_parent,
                exp_bees,
                eff_p_null,
                w_norm_base=w_n_b,
                normed_unit_probe=init_unit,
                head_weight=h_w,
                probe_targets=init_tgt,
                is_general_mask=init_gen,
                ties_density=cfg.ties_density,
                ties_sign_mode=cfg.ties_sign_mode,
                max_peer_norm_ratio=cfg.max_peer_norm_ratio,
                sink_experts=sink_list,
                anchor_experts=anchor_list,
                damping=cfg.fisher_damping,
                step_scales=cfg.fisher_step_scales,
                tau_grid=cfg.tau_grid,
                max_rel_delta=cfg.max_norm_rel_delta,
                max_d0_rel=cfg.probe_max_d0_rel,
                ece_weight=cfg.ece_energy_weight,
                sv_rel_tol=cfg.d0_grad_sv_rel_tol,
                b_norm_base=b_n_b,
            )
            # Apply distilled experts (+ optional TIES-P0 router) and re-run Fisher norm on updated hidden states
            inter_blocks = {bid: {sk: t.clone() for sk, t in sub.items()} for bid, sub in parent_blocks.items()}
            for bid, sub in stage1["royal_jelly_experts"].items():
                inter_blocks[bid] = sub
            if not freeze_router:
                for r_bid in rtr_parent:
                    l_idx = _parse_layer_idx(r_bid)
                    p_rtr = (
                        _resolve_expert_projector(eff_p_null, f"blocks.{l_idx}.mlp.experts.0", "x")
                        if (eff_p_null is not None and l_idx is not None)
                        else None
                    )
                    for sk, r_base_t in rtr_parent[r_bid].items():
                        r_peers = [rb[r_bid][sk] for rb in router_bees if r_bid in rb and sk in rb[r_bid]]
                        if r_peers:
                            r_delta, _ = trim_and_elect_sign_ties(r_peers, density_top_p=cfg.ties_density)
                            r_proj = _project_delta_with_p0(r_delta, p_rtr, sk, r_bid) if p_rtr is not None else r_delta.to(torch.float64)
                            inter_blocks[r_bid][sk] = (r_base_t.to(torch.float64) + r_proj).to(r_base_t.dtype)
            apply_checkpoint_blocks_inplace(work_model, inter_blocks)
            _, upd_unit, upd_tgt, upd_gen = _build_model_p_null_and_probe(
                work_model, p_chunks, score_from=cfg.score_from, general_prefixes=cfg.general_prefixes, build_projectors=False
            )
            w_cal, b_cal, fisher_meta = compute_ker_g_d0_fisher_norm_step(
                w_n_b,
                upd_unit,
                h_w,
                upd_tgt,
                upd_gen,
                b_norm_base=b_n_b,
                damping=cfg.fisher_damping,
                step_scales=cfg.fisher_step_scales,
                tau_grid=cfg.tau_grid,
                max_rel_delta=cfg.max_norm_rel_delta,
                max_d0_rel=cfg.probe_max_d0_rel,
                ece_weight=cfg.ece_energy_weight,
                sv_rel_tol=cfg.d0_grad_sv_rel_tol,
            )
            distill_res = {**stage1, **fisher_meta, "w_norm_calibrated": w_cal, "b_norm_calibrated": b_cal}
            cand_blocks = inter_blocks
        else:
            assert normed_unit_probe is not None and probe_targets is not None and is_general_mask is not None
            distill_res = distill_forager_nectar_to_royal_jelly(
                exp_parent,
                exp_bees,
                eff_p_null,
                w_norm_base=w_n_b,
                normed_unit_probe=normed_unit_probe,
                head_weight=h_w,
                probe_targets=probe_targets,
                is_general_mask=is_general_mask,
                ties_density=cfg.ties_density,
                ties_sign_mode=cfg.ties_sign_mode,
                max_peer_norm_ratio=cfg.max_peer_norm_ratio,
                sink_experts=sink_list,
                anchor_experts=anchor_list,
                damping=cfg.fisher_damping,
                step_scales=cfg.fisher_step_scales,
                tau_grid=cfg.tau_grid,
                max_rel_delta=cfg.max_norm_rel_delta,
                max_d0_rel=cfg.probe_max_d0_rel,
                ece_weight=cfg.ece_energy_weight,
                sv_rel_tol=cfg.d0_grad_sv_rel_tol,
                b_norm_base=b_n_b,
            )
            cand_blocks = {bid: {sk: t.clone() for sk, t in sub.items()} for bid, sub in parent_blocks.items()}
            for bid, sub in distill_res["royal_jelly_experts"].items():
                cand_blocks[bid] = sub

        if ln_parent:
            ln_key = next(iter(ln_parent))
            cand_blocks[ln_key]["weight"] = distill_res["w_norm_calibrated"].to(cand_blocks[ln_key]["weight"].dtype)
            if distill_res["b_norm_calibrated"] is not None and "bias" in cand_blocks[ln_key]:
                cand_blocks[ln_key]["bias"] = distill_res["b_norm_calibrated"].to(cand_blocks[ln_key]["bias"].dtype)

        cand_sha = compute_checkpoint_sha256(cand_blocks)
        cand_eval = _eval_blocks(cand_blocks, "royal_jelly_eval")
        cand_metrics = _score_against_base(cand_eval, base_eval, cfg)
        distill_summary = {
            "ties_density": float(distill_res["ties_density"]),
            "mean_sign_consensus_ratio": float(distill_res["mean_sign_consensus_ratio"]),
            "eta_fisher": float(distill_res["eta_fisher"]),
            "tau_temp": float(distill_res["tau_temp"]),
            "g_d0_rank": int(distill_res["g_d0_rank"]),
            "ker_g_d0_dim": int(distill_res["ker_g_d0_dim"]),
            "first_order_d0_orthogonality": float(distill_res["first_order_d0_orthogonality"]),
        }
        delta_e = float(cand_metrics["anse_energy"]) - float(parent_anse_energy)
        cand_ok = bool(cand_metrics["guard_ok"] and delta_e < -cfg.min_energy_improvement)

    if cand_ok:
        accepted = True
        rolled_back = False
        final_blocks, final_eval, final_metrics = cand_blocks, cand_eval, cand_metrics
        status = "ALVEOLE_OPERCULATED_COMMITTED"
        final_delta_e = float(final_metrics["anse_energy"]) - float(parent_anse_energy)
    else:
        accepted = False
        rolled_back = True
        final_blocks = rollback_target_blocks
        final_eval = _eval_blocks(final_blocks, "rollback_eval")
        final_metrics = _score_against_base(final_eval, base_eval, cfg)
        status = "ROLLED_BACK_ANSE_MAXIMUM_PAIN"
        final_delta_e = cfg.maximum_pain_energy

    if work_model is not None:
        apply_checkpoint_blocks_inplace(work_model, final_blocks)
        final_sha = compute_checkpoint_sha256(work_model)
    else:
        final_sha = compute_checkpoint_sha256(final_blocks)

    rollback_verified = (final_sha == rollback_expected_sha) if rolled_back else False
    if rolled_back and not rollback_verified:
        raise RuntimeError(
            f"Atomic bitwise rollback SHA-256 mismatch: expected {rollback_expected_sha[:12]}, got {final_sha[:12]}"
        )

    # Emit next-round QueenPheromoneDirective
    if expert_causal_gains_nats is not None and expert_prose_harm_nats is not None:
        gains_in, harms_in = expert_causal_gains_nats, expert_prose_harm_nats
    elif work_model is not None and p_chunks is not None:
        gains_in, harms_in = _compute_loeo_expert_metrics(
            work_model, p_chunks, score_from=cfg.score_from, general_prefixes=cfg.general_prefixes
        )
    else:
        gains_in = {ek: 0.0 for ek in sorted(exp_parent.keys())}
        harms_in = {ek: 0.0 for ek in sorted(exp_parent.keys())}

    pheromone = emit_queen_pheromone_campaign_directive(
        gains_in,
        harms_in,
        router_expert_kl_mismatch,
        final_eval["domain_nlls"],
        final_eval["domain_eces"],
        anchor_gain_floor_nats=cfg.anchor_gain_floor_nats,
        coadapt_kl_ceiling=cfg.coadapt_kl_ceiling,
        temperature=cfg.foraging_temperature,
        prose_harm_sink_floor_nats=cfg.corruptor_prose_harm_floor_nats,
        domain_round_counts=domain_round_counts,
        ucb_exploration_coeff=cfg.ucb_exploration_coeff,
        general_prefixes=cfg.general_prefixes,
        campaign_id=camp_s,
        round_index=r_idx + 1,
    )

    def _clean_f(val: float) -> float | None:
        return float(val) if math.isfinite(float(val)) else None

    rescued = bool(accepted and (not raw_metrics["guard_ok"]))
    receipt: dict[str, Any] = {
        "object": QUEEN_CAMPAIGN_RECEIPT_OBJECT,
        "schema": QUEEN_CORTEX_SCHEMA,
        "campaign_id": camp_s,
        "round_index": r_idx,
        "status": status,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "rollback_target": cfg.rollback_target if rolled_back else "none",
        "rollback_bitwise_verified": rollback_verified,
        "rescued_from_discarded_round": rescued,
        "nll_mean": float(final_eval["nll_mean"]),
        "ece_value": float(final_eval["ece_value"]),
        "anse_energy": float(final_metrics["anse_energy"]) if accepted else cfg.maximum_pain_energy,
        "delta_e": float(final_delta_e),
        "general_drift_ratio": _clean_f(final_metrics["general_drift_ratio"]),
        "worst_domain_drift_ratio": _clean_f(final_metrics["worst_domain_drift_ratio"]),
        "guard_ok": bool(final_metrics["guard_ok"]) if accepted else False,
        "domain_nlls": dict(final_eval["domain_nlls"]),
        "domain_eces": dict(final_eval["domain_eces"]),
        "raw_unguarded_metrics": {
            "all_finite": bool(raw_eval.get("all_finite", True)),
            "nll_mean": _clean_f(raw_eval["nll_mean"]),
            "ece_value": _clean_f(raw_eval["ece_value"]),
            "general_drift_ratio": _clean_f(raw_metrics["general_drift_ratio"]),
            "worst_domain_drift_ratio": _clean_f(raw_metrics["worst_domain_drift_ratio"]),
            "guard_ok": bool(raw_metrics["guard_ok"]),
            "would_be_discarded": not bool(raw_metrics["guard_ok"]),
        },
        "candidate_metrics": {
            "all_finite": bool(cand_eval.get("all_finite", True)),
            "nll_mean": _clean_f(cand_eval["nll_mean"]),
            "ece_value": _clean_f(cand_eval["ece_value"]),
            "general_drift_ratio": _clean_f(cand_metrics["general_drift_ratio"]),
            "worst_domain_drift_ratio": _clean_f(cand_metrics["worst_domain_drift_ratio"]),
            "anse_energy": float(cand_metrics["anse_energy"]),
            "guard_ok": bool(cand_metrics["guard_ok"]),
        },
        "distillation_summary": distill_summary,
        "pheromone_directive": pheromone.to_dict(),
        "base_state_sha256": base_sha,
        "parent_state_sha256": parent_sha,
        "candidate_state_sha256": cand_sha,
        "final_state_sha256": final_sha,
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return (work_model if work_model is not None else final_blocks), receipt


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "QUEEN_CAMPAIGN_RECEIPT_OBJECT",
    "QUEEN_CORTEX_SCHEMA",
    "QUEEN_PHEROMONE_OBJECT",
    "QueenCortexConfig",
    "QueenPheromoneDirective",
    "compute_ker_g_d0_fisher_norm_step",
    "distill_forager_nectar_to_royal_jelly",
    "emit_queen_pheromone_campaign_directive",
    "execute_queen_cortex_campaign_round",
    "verify_queen_cortex_receipt_provenance",
    "verify_queen_pheromone_directive_signature",
]

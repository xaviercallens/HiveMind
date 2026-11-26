# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Closed-Loop Inter-Round Royal-Jelly DiLoCo Trophallaxis & Pheromone Propagation Controller.

Implements **Gen-3 Direction D1** (Zenodo ``10.5281/zenodo.24542441`` Section 6 and Lean 4
theorem ``ANSE.ReineCortexDirections.thm_rcd_d1_closed_loop_diloco_momentum_invariance``):
embeds **Reine-Cortex** (``QueenCortex``) directly inside the multi-round DiLoCo outer
barrier ``t -> t + 1`` on CPU (``0 MiB`` GPU VRAM, ``0`` gradient steps):

1. **Inter-Round Royal-Jelly Pre-Momentum Trophallaxis**
   (:func:`apply_inter_round_royal_jelly_momentum_step`):
   Purifies raw volunteer Forager Bee deltas via TIES sign-consensus election, Byzantine
   peer-norm capping, and AlphaEdit ``P_0 = I - V V^T`` general-prose null-space projection
   *before* outer Nesterov / heavy-ball DiLoCo momentum accumulation
   (``M_t = beta * P_0(M_{t-1}) + Delta_t^{RJ}``), preserving exact null-space membership
   ``||M_t V||_F == 0`` across arbitrarily many rounds while rescuing drifting outer rounds
   before rejection.
2. **Closed-Loop Pheromone Propagation & Boltzmann/UCB Foraging Steering**:
   Propagates signed ``QueenPheromoneDirective`` state across consecutive rounds:
   freezing converged Anchor Combs (``anchor_experts_to_freeze``) bitwise, isolating and
   resetting momentum on corrupting Ghost Sinks (``corruptor_experts_to_sink_or_reseed``),
   enforcing ``router_frozen`` vs ``router_unfrozen_coadapted`` co-adaptation modes, and
   steering volunteer bee weights via normalized Boltzmann/UCB domain foraging probabilities.
3. **Monotone ANSE Ratchet, Atomic Bitwise Rollback & Hash-Chained Campaign Receipts**
   (:class:`InterRoundRoyalJellyTrophallaxisController`, :func:`verify_inter_round_trophallaxis_chain`):
   Combines outer momentum steps with ``ker(G_D0)`` diagonal-Fisher norm + folded safe
   ``tau >= 1.0`` calibration, commits round ``t`` iff ``ANSE Delta E < 0`` and
   ``D0 / worst_domain <= +1.0%``, restores *both* model weights and outer momentum state
   bit-for-bit under ``ANSE_MAXIMUM_PAIN`` (``1e6``) on infeasible rounds, tracks parallel
   unguarded DiLoCo trajectories for paired proof of improvement, and signs a continuous
   SHA-256 hash chain across all rounds.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from hivemind.swarm.byzantine_wsp_delta_aggregator import compute_payload_sha256
from hivemind.swarm.queen_cortex_moe_campaign_governor import (
    ANSE_MAXIMUM_PAIN,
    DEFAULT_ANCHOR_GAIN_FLOOR_NATS,
    DEFAULT_ANSE_SAFE_D0_RATIO,
    DEFAULT_COADAPT_KL_CEILING,
    DEFAULT_DOMAIN_GUARD_MAX_RATIO,
    DEFAULT_ECE_ENERGY_WEIGHT,
    DEFAULT_FISHER_DAMPING,
    DEFAULT_FORAGING_TEMPERATURE,
    DEFAULT_GENERAL_PREFIXES,
    DEFAULT_MAX_D0_REL_PROBE,
    DEFAULT_MAX_REL_DELTA,
    DEFAULT_PROSE_HARM_SINK_FLOOR_NATS,
    DEFAULT_SOFT_MARGIN_PENALTY,
    DEFAULT_STEP_SCALES,
    DEFAULT_TAU_GRID,
    DEFAULT_TIES_DENSITY,
    DEFAULT_UCB_EXPLORATION_COEFF,
    QueenCortexConfig,
    QueenPheromoneDirective,
    _build_model_p_null_and_probe,
    _compute_loeo_expert_metrics,
    _evaluate_model_on_chunks,
    _is_final_ln_key,
    _is_general_domain,
    _normalize_bee_deltas,
    _normalize_eval_result,
    _normalize_expert_store,
    _parse_expert_idx,
    _parse_layer_idx,
    _partition_bee_deltas,
    _partition_checkpoint_blocks,
    _project_delta_with_p0,
    _require_bool,
    _require_float,
    _require_fp_tensor,
    _require_int,
    _require_nonempty_str,
    _require_projector,
    _resolve_expert_projector,
    _score_against_base,
    _validate_probe_chunks,
    compute_ker_g_d0_fisher_norm_step,
    emit_queen_pheromone_campaign_directive,
    verify_queen_pheromone_directive_signature,
)
from hivemind.swarm.self_healing_checkpoint_rescuer import (
    apply_checkpoint_blocks_inplace,
    compute_checkpoint_sha256,
    extract_checkpoint_blocks,
)
from hivemind.swarm.ties_dare_moe_merger import trim_and_elect_sign_ties

INTER_ROUND_TROPHALLAXIS_SCHEMA: str = "hivemind.inter_round_royal_jelly_trophallaxis/1"
ROUND_RECEIPT_OBJECT: str = "hivemind.inter_round_royal_jelly_trophallaxis_receipt"
CHAIN_SUMMARY_OBJECT: str = "hivemind.inter_round_royal_jelly_trophallaxis_chain"
GENESIS_RECEIPT_SHA256: str = "0" * 64

DEFAULT_OUTER_LR: float = 0.7
DEFAULT_OUTER_MOMENTUM_BETA: float = 0.9
DEFAULT_FORAGING_WEIGHT_FLOOR: float = 0.10

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_ROUTER_SUFFIX = ".router"


# ---------------------------------------------------------------------------
# 1. Configuration & Algebraic / Hash-Chain Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterRoundTrophallaxisConfig:
    """Validated configuration for closed-loop inter-round Royal-Jelly DiLoCo trophallaxis."""

    outer_lr: float = DEFAULT_OUTER_LR
    outer_momentum_beta: float = DEFAULT_OUTER_MOMENTUM_BETA
    nesterov: bool = True
    project_momentum_on_step: bool = True
    use_foraging_weights_for_bees: bool = True
    foraging_weight_floor: float = DEFAULT_FORAGING_WEIGHT_FLOOR
    reseed_corruptors_from_momentum: bool = True
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
    inplace: bool = False

    def __post_init__(self) -> None:
        _require_float(self.outer_lr, "outer_lr", min_value=0.0, strict_min=True, max_value=2.0)
        _require_float(self.outer_momentum_beta, "outer_momentum_beta", min_value=0.0, max_value=0.9999)
        _require_bool(self.nesterov, "nesterov")
        _require_bool(self.project_momentum_on_step, "project_momentum_on_step")
        _require_bool(self.use_foraging_weights_for_bees, "use_foraging_weights_for_bees")
        _require_float(self.foraging_weight_floor, "foraging_weight_floor", min_value=0.0, max_value=1.0)
        _require_bool(self.reseed_corruptors_from_momentum, "reseed_corruptors_from_momentum")
        # Validate QueenCortex-shared tolerances identically through QueenCortexConfig
        qc = self.to_queen_cortex_config()
        object.__setattr__(self, "fisher_step_scales", qc.fisher_step_scales)
        object.__setattr__(self, "tau_grid", qc.tau_grid)

    def to_queen_cortex_config(self) -> QueenCortexConfig:
        """Build an equivalent validated :class:`QueenCortexConfig` sharing ANSE and Fisher rules."""
        return QueenCortexConfig(
            ties_density=self.ties_density,
            ties_sign_mode=self.ties_sign_mode,
            max_peer_norm_ratio=self.max_peer_norm_ratio,
            anchor_gain_floor_nats=self.anchor_gain_floor_nats,
            corruptor_prose_harm_floor_nats=self.corruptor_prose_harm_floor_nats,
            coadapt_kl_ceiling=self.coadapt_kl_ceiling,
            foraging_temperature=self.foraging_temperature,
            ucb_exploration_coeff=self.ucb_exploration_coeff,
            max_domain_drift_ratio=self.max_domain_drift_ratio,
            anse_safe_general_ratio=self.anse_safe_general_ratio,
            soft_margin_penalty=self.soft_margin_penalty,
            maximum_pain_energy=self.maximum_pain_energy,
            ece_energy_weight=self.ece_energy_weight,
            min_energy_improvement=self.min_energy_improvement,
            fisher_damping=self.fisher_damping,
            fisher_step_scales=self.fisher_step_scales,
            tau_grid=self.tau_grid,
            max_norm_rel_delta=self.max_norm_rel_delta,
            probe_max_d0_rel=self.probe_max_d0_rel,
            d0_grad_sv_rel_tol=self.d0_grad_sv_rel_tol,
            score_from=self.score_from,
            batch_size=self.batch_size,
            general_prefixes=self.general_prefixes,
            rollback_target="parent",
            inplace=self.inplace,
        )


def _clone_blocks(blocks: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    """Return an isolated deep clone of normalized block mappings."""
    return {bid: {sk: t.detach().clone() for sk, t in sub.items()} for bid, sub in blocks.items()}


def _matches_role(block_id: str, role_tokens: set[str]) -> bool:
    """Match block IDs (including layer-prefixed expert paths) against directive expert tokens."""
    if block_id in role_tokens:
        return True
    tail = _parse_expert_idx(block_id)
    return tail is not None and (str(tail) in role_tokens or f"expert_{tail}" in role_tokens)


def initialize_zero_momentum_state(
    blocks: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Initialize zeroed CPU ``float64`` outer DiLoCo momentum buffers for trainable experts and routers."""
    if not isinstance(blocks, Mapping) or not blocks:
        raise ValueError("blocks must be a non-empty mapping of block tensors")
    momentum: dict[str, dict[str, torch.Tensor]] = {}
    for bid, sub in blocks.items():
        if _is_final_ln_key(bid):
            continue
        if not isinstance(sub, Mapping) or not sub:
            raise TypeError(f"blocks[{bid!r}] must be a non-empty Mapping[str, torch.Tensor]")
        momentum[bid] = {
            sk: torch.zeros(t.shape, dtype=torch.float64, device="cpu")
            for sk, t in sub.items()
        }
    if not momentum:
        raise ValueError("blocks must contain at least one expert or router block for momentum tracking")
    return momentum


def compute_momentum_state_sha256(
    momentum_state: Mapping[str, Mapping[str, torch.Tensor]],
) -> str:
    """Compute a deterministic bitwise SHA-256 digest over outer momentum buffers."""
    return compute_checkpoint_sha256(momentum_state)


def compute_prose_basis_from_projector(
    p_mat: torch.Tensor,
    *,
    eig_tol: float = 1e-2,
) -> torch.Tensor:
    """Extract an orthonormal basis ``V`` of ``ker(P_0)`` from ``P_0 = I - V V^T``."""
    p_clean = _require_projector(p_mat, "p_mat", eig_tol=eig_tol)
    dim = int(p_clean.shape[0])
    q_perp = 0.5 * ((torch.eye(dim, dtype=torch.float64) - p_clean) + (torch.eye(dim, dtype=torch.float64) - p_clean).T)
    evals, evecs = torch.linalg.eigh(q_perp)
    keep = evals > 0.5
    if not bool(keep.any().item()):
        return torch.zeros((dim, 0), dtype=torch.float64)
    return evecs[:, keep].contiguous()


def measure_momentum_prose_leakage(
    momentum_state: Mapping[str, Mapping[str, torch.Tensor]],
    p_null_by_expert: Mapping[str | int, Any] | None,
) -> dict[str, float]:
    """Measure residual projection ``M_t (I - P_0)`` along the preserved general-prose span."""
    if not isinstance(momentum_state, Mapping) or not momentum_state:
        raise ValueError("momentum_state must be a non-empty mapping")
    if p_null_by_expert is None:
        return {"max_abs_leakage": 0.0, "frobenius_leakage": 0.0, "relative_leakage": 0.0}

    max_abs = 0.0
    leak_sq = 0.0
    total_sq = 0.0
    for bid, sub in momentum_state.items():
        if _is_final_ln_key(bid):
            continue
        is_router = bid.endswith(_ROUTER_SUFFIX)
        l_idx = _parse_layer_idx(bid) if is_router else None
        for sk, m_t in sub.items():
            m64 = _require_fp_tensor(m_t, f"momentum_state[{bid!r}][{sk!r}]").detach().to("cpu", dtype=torch.float64)
            total_sq += float(m64.pow(2).sum().item())
            if is_router:
                p_mat = (
                    _resolve_expert_projector(
                        p_null_by_expert, f"blocks.{l_idx}.mlp.experts.0", "x", delta_shape=tuple(m64.shape), is_single_tensor=False
                    )
                    if l_idx is not None
                    else None
                )
            else:
                p_mat = _resolve_expert_projector(
                    p_null_by_expert,
                    bid,
                    sk,
                    delta_shape=tuple(m64.shape),
                    is_single_tensor=(sk == "__tensor__"),
                )
            if p_mat is None:
                continue
            projected = _project_delta_with_p0(m64, p_mat, sk, bid)
            leak = m64 - projected
            max_abs = max(max_abs, float(leak.abs().max().item()))
            leak_sq += float(leak.pow(2).sum().item())

    frob_leak = math.sqrt(leak_sq)
    total_norm = math.sqrt(total_sq)
    rel_leak = frob_leak / total_norm if total_norm > 1e-12 else 0.0
    return {
        "max_abs_leakage": max_abs,
        "frobenius_leakage": frob_leak,
        "relative_leakage": rel_leak,
    }


def resolve_pheromone_bee_weights(
    bee_ids: Sequence[str],
    *,
    bee_domains: Sequence[str] | Mapping[str, str] | None = None,
    directive: QueenPheromoneDirective | Mapping[str, Any] | None = None,
    explicit_bee_weights: Sequence[float] | None = None,
    weight_floor: float = DEFAULT_FORAGING_WEIGHT_FLOOR,
) -> list[float]:
    """Compute normalized peer weights steered by ``QueenPheromoneDirective`` foraging simplex."""
    n_bees = len(bee_ids)
    if n_bees == 0:
        raise ValueError("bee_ids must be non-empty")
    floor_f = _require_float(weight_floor, "weight_floor", min_value=0.0, max_value=1.0)

    if explicit_bee_weights is not None:
        if isinstance(explicit_bee_weights, (str, bytes)) or not isinstance(explicit_bee_weights, Sequence):
            raise TypeError("explicit_bee_weights must be a Sequence[float] or None")
        if len(explicit_bee_weights) != n_bees:
            raise ValueError(f"explicit_bee_weights length {len(explicit_bee_weights)} != {n_bees}")
        vals = [_require_float(w, f"explicit_bee_weights[{i}]", min_value=0.0) for i, w in enumerate(explicit_bee_weights)]
        total = sum(vals)
        if total <= 0.0:
            raise ValueError("explicit_bee_weights must have a strictly positive sum")
        return [v / total for v in vals]

    if directive is None or bee_domains is None:
        return [1.0 / float(n_bees)] * n_bees

    d_map = directive.to_dict() if isinstance(directive, QueenPheromoneDirective) else directive
    fw_raw = d_map.get("bee_foraging_domain_weights")
    if not isinstance(fw_raw, Mapping) or not fw_raw:
        return [1.0 / float(n_bees)] * n_bees

    if isinstance(bee_domains, Mapping):
        dom_list = [_require_nonempty_str(str(bee_domains.get(bid, "")), f"bee_domains[{bid!r}]") for bid in bee_ids]
    elif isinstance(bee_domains, Sequence) and not isinstance(bee_domains, (str, bytes)):
        if len(bee_domains) != n_bees:
            raise ValueError(f"bee_domains length {len(bee_domains)} != {n_bees}")
        dom_list = [_require_nonempty_str(str(d), f"bee_domains[{i}]") for i, d in enumerate(bee_domains)]
    else:
        raise TypeError("bee_domains must be a Sequence[str], Mapping[str, str], or None")

    uniform = 1.0 / float(n_bees)
    raw_ws: list[float] = []
    for dom in dom_list:
        w_dom = fw_raw.get(dom)
        if w_dom is None:
            low_dom = dom.lower()
            w_dom = next(
                (v for k, v in fw_raw.items() if str(k).lower().startswith(low_dom) or low_dom.startswith(str(k).lower())),
                uniform,
            )
        w_f = _require_float(w_dom, f"foraging_weight[{dom!r}]", min_value=0.0)
        raw_ws.append((1.0 - floor_f) * w_f + floor_f * uniform)

    total_w = sum(raw_ws)
    if total_w <= 0.0:
        return [uniform] * n_bees
    return [w / total_w for w in raw_ws]


# ---------------------------------------------------------------------------
# 2. Pre-Momentum Inter-Round Royal-Jelly Trophallaxis Step (Direction D1)
# ---------------------------------------------------------------------------


def _parse_and_verify_directive(
    directive: QueenPheromoneDirective | Mapping[str, Any] | None,
) -> tuple[set[str], set[str], bool, str | None]:
    """Validate optional incoming ``QueenPheromoneDirective`` and unpack control masks."""
    if directive is None:
        return set(), set(), False, None
    if isinstance(directive, QueenPheromoneDirective):
        d_map = directive.to_dict()
    elif isinstance(directive, Mapping):
        d_map = dict(directive)
    else:
        raise TypeError("directive must be a QueenPheromoneDirective, Mapping, or None")

    if not verify_queen_pheromone_directive_signature(d_map):
        raise ValueError("Incoming QueenPheromoneDirective failed SHA-256 signature verification")

    anchors = {str(x).strip() for x in d_map.get("anchor_experts_to_freeze", ()) if str(x).strip()}
    sinks = {str(x).strip() for x in d_map.get("corruptor_experts_to_sink_or_reseed", ()) if str(x).strip()}
    freeze_router = str(d_map.get("router_training_mode", "")) == "router_frozen"
    sig = str(d_map["directive_sha256"])
    return anchors, sinks, freeze_router, sig


def _step_tensor_momentum_with_p0(
    m_prev: torch.Tensor,
    delta_raw_peers: Sequence[torch.Tensor],
    peer_ws: Sequence[float],
    p_mat: torch.Tensor | None,
    sub_key: str,
    block_id: str,
    cfg: InterRoundTrophallaxisConfig,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Purify peer delta via TIES + ``P_0`` *before* outer DiLoCo momentum accumulation."""
    ties_delta, ties_meta = trim_and_elect_sign_ties(
        list(delta_raw_peers),
        density_top_p=cfg.ties_density,
        peer_weights=list(peer_ws),
        sign_election_mode=cfg.ties_sign_mode,
        max_peer_norm_ratio=cfg.max_peer_norm_ratio,
    )
    delta_rj = _project_delta_with_p0(ties_delta, p_mat, sub_key, block_id)
    m_base = (
        _project_delta_with_p0(m_prev, p_mat, sub_key, block_id)
        if (cfg.project_momentum_on_step and p_mat is not None)
        else m_prev.detach().to(device="cpu", dtype=torch.float64)
    )
    m_next = cfg.outer_momentum_beta * m_base + delta_rj
    if p_mat is not None:
        m_next = _project_delta_with_p0(m_next, p_mat, sub_key, block_id)
    outer_grad = (delta_rj + cfg.outer_momentum_beta * m_next) if cfg.nesterov else m_next
    outer_step = cfg.outer_lr * outer_grad
    return outer_step, m_next, float(ties_meta["sign_consensus_ratio"])


@torch.no_grad()
def apply_inter_round_royal_jelly_momentum_step(
    parent_blocks: Mapping[str, Mapping[str, torch.Tensor]],
    forager_bee_deltas: Sequence[Mapping[str | int, Any]] | Mapping[str | int, Any],
    momentum_state: Mapping[str, Mapping[str, torch.Tensor]],
    p_null_by_expert: Mapping[str | int, Any] | None,
    *,
    directive: QueenPheromoneDirective | Mapping[str, Any] | None = None,
    bee_domains: Sequence[str] | Mapping[str, str] | None = None,
    bee_weights: Sequence[float] | None = None,
    config: InterRoundTrophallaxisConfig | None = None,
) -> dict[str, Any]:
    """Execute one pre-momentum Royal-Jelly trophallaxis + DiLoCo outer update over blocks."""
    cfg = config if config is not None else InterRoundTrophallaxisConfig()
    if not isinstance(cfg, InterRoundTrophallaxisConfig):
        raise TypeError("config must be InterRoundTrophallaxisConfig or None")

    norm_parent, _, _, _ = _normalize_expert_store(parent_blocks, "parent_blocks")
    exp_parent, rtr_parent, ln_parent = _partition_checkpoint_blocks(norm_parent)
    exp_bees, router_bees, bees_all_finite = _partition_bee_deltas(
        forager_bee_deltas, set(exp_parent.keys()), set(rtr_parent.keys())
    )
    if not bees_all_finite:
        raise ValueError("forager_bee_deltas contains non-finite values")

    # Resolve canonical bee IDs matching _normalize_bee_deltas ordering
    if isinstance(forager_bee_deltas, Mapping):
        bee_ids = [str(k).strip() for k in forager_bee_deltas.keys()]
    else:
        bee_ids = [f"bee_{i}" for i in range(len(forager_bee_deltas))]

    norm_exp_parent, exp_key_order, _, exp_single_flags = _normalize_expert_store(exp_parent, "exp_parent")
    bees_norm, _ = _normalize_bee_deltas(exp_bees, norm_exp_parent, exp_single_flags)

    eff_bee_weights = resolve_pheromone_bee_weights(
        bee_ids,
        bee_domains=bee_domains if cfg.use_foraging_weights_for_bees else None,
        directive=directive if cfg.use_foraging_weights_for_bees else None,
        explicit_bee_weights=bee_weights,
        weight_floor=cfg.foraging_weight_floor,
    )

    anchors, sinks, freeze_router, directive_sha = _parse_and_verify_directive(directive)

    cand_blocks = _clone_blocks(norm_parent)
    cand_momentum: dict[str, dict[str, torch.Tensor]] = {}
    expert_reports: dict[str, dict[str, Any]] = {}
    consensus_scores: list[float] = []

    for ek in exp_key_order:
        sub_base = norm_exp_parent[ek]
        sub_m = momentum_state.get(ek, {})
        if _matches_role(ek, sinks):
            cand_blocks[ek] = {
                sk: (torch.zeros_like(t) if sk.endswith("_B") else t.clone())
                for sk, t in sub_base.items()
            }
            cand_momentum[ek] = {
                sk: (
                    torch.zeros(t.shape, dtype=torch.float64, device="cpu")
                    if (cfg.reseed_corruptors_from_momentum or sk.endswith("_B"))
                    else sub_m.get(sk, torch.zeros(t.shape, dtype=torch.float64, device="cpu")).detach().clone()
                )
                for sk, t in sub_base.items()
            }
            expert_reports[ek] = {"action": "corruptor_isolated", "sign_consensus_ratio": 1.0}
            continue

        if _matches_role(ek, anchors):
            cand_blocks[ek] = {sk: t.clone() for sk, t in sub_base.items()}
            cand_momentum[ek] = {
                sk: torch.zeros(t.shape, dtype=torch.float64, device="cpu")
                for sk, t in sub_base.items()
            }
            expert_reports[ek] = {"action": "anchor_frozen", "sign_consensus_ratio": 1.0}
            continue

        upd_sub: dict[str, torch.Tensor] = {}
        upd_m: dict[str, torch.Tensor] = {}
        sub_cons: list[float] = []
        for sk, base_t in sub_base.items():
            m_prev = sub_m.get(sk, torch.zeros(base_t.shape, dtype=torch.float64, device="cpu"))
            peers: list[torch.Tensor] = []
            peer_ws: list[float] = []
            for bee_map, bw in zip(bees_norm, eff_bee_weights, strict=True):
                if ek in bee_map and sk in bee_map[ek]:
                    peers.append(bee_map[ek][sk])
                    peer_ws.append(bw)
            p_mat = _resolve_expert_projector(
                p_null_by_expert,
                ek,
                sk,
                delta_shape=tuple(base_t.shape),
                is_single_tensor=exp_single_flags[ek],
            )
            if not peers:
                # Propagate existing null-space momentum only when non-zero, preserving P_0
                m_base = (
                    _project_delta_with_p0(m_prev, p_mat, sk, ek)
                    if (cfg.project_momentum_on_step and p_mat is not None)
                    else m_prev.detach().to(device="cpu", dtype=torch.float64)
                )
                m_next = cfg.outer_momentum_beta * m_base
                outer_step = cfg.outer_lr * ((cfg.outer_momentum_beta * m_next) if cfg.nesterov else m_next)
                upd_sub[sk] = (base_t.detach().to("cpu", dtype=torch.float64) + outer_step).to(
                    device=base_t.device, dtype=base_t.dtype
                )
                upd_m[sk] = m_next
                continue

            outer_step, m_next, cons = _step_tensor_momentum_with_p0(
                m_prev, peers, peer_ws, p_mat, sk, ek, cfg
            )
            sub_cons.append(cons)
            upd_sub[sk] = (base_t.detach().to("cpu", dtype=torch.float64) + outer_step).to(
                device=base_t.device, dtype=base_t.dtype
            )
            upd_m[sk] = m_next

        cand_blocks[ek] = upd_sub
        cand_momentum[ek] = upd_m
        consensus_scores.extend(sub_cons)
        expert_reports[ek] = {
            "action": "royal_jelly_trophallaxis" if sub_cons else "momentum_carry",
            "sign_consensus_ratio": float(np.mean(sub_cons)) if sub_cons else 1.0,
        }

    # Router update governed by QueenPheromoneDirective router_training_mode
    router_action = "router_frozen_bitwise" if freeze_router else "router_trophallaxis_p0"
    for r_bid, r_sub in rtr_parent.items():
        r_m_prev_map = momentum_state.get(r_bid, {})
        if freeze_router:
            cand_blocks[r_bid] = {sk: t.clone() for sk, t in r_sub.items()}
            cand_momentum[r_bid] = {
                sk: torch.zeros(t.shape, dtype=torch.float64, device="cpu")
                for sk, t in r_sub.items()
            }
            continue

        l_idx = _parse_layer_idx(r_bid)
        p_rtr = (
            _resolve_expert_projector(p_null_by_expert, f"blocks.{l_idx}.mlp.experts.0", "x")
            if (p_null_by_expert is not None and l_idx is not None)
            else None
        )
        upd_r: dict[str, torch.Tensor] = {}
        upd_rm: dict[str, torch.Tensor] = {}
        for sk, r_base_t in r_sub.items():
            m_prev = r_m_prev_map.get(sk, torch.zeros(r_base_t.shape, dtype=torch.float64, device="cpu"))
            r_peers: list[torch.Tensor] = []
            r_ws: list[float] = []
            for rb, bw in zip(router_bees, eff_bee_weights, strict=True):
                if r_bid in rb and sk in rb[r_bid]:
                    r_peers.append(rb[r_bid][sk])
                    r_ws.append(bw)
            if not r_peers:
                m_next = cfg.outer_momentum_beta * (
                    _project_delta_with_p0(m_prev, p_rtr, sk, r_bid)
                    if (cfg.project_momentum_on_step and p_rtr is not None)
                    else m_prev.detach().to("cpu", dtype=torch.float64)
                )
                outer_step = cfg.outer_lr * ((cfg.outer_momentum_beta * m_next) if cfg.nesterov else m_next)
            else:
                outer_step, m_next, _ = _step_tensor_momentum_with_p0(
                    m_prev, r_peers, r_ws, p_rtr, sk, r_bid, cfg
                )
            upd_r[sk] = (r_base_t.detach().to("cpu", dtype=torch.float64) + outer_step).to(
                device=r_base_t.device, dtype=r_base_t.dtype
            )
            upd_rm[sk] = m_next
        cand_blocks[r_bid] = upd_r
        cand_momentum[r_bid] = upd_rm

    for ln_k, ln_sub in ln_parent.items():
        cand_blocks[ln_k] = {sk: t.clone() for sk, t in ln_sub.items()}

    leakage = measure_momentum_prose_leakage(cand_momentum, p_null_by_expert)
    return {
        "candidate_blocks": cand_blocks,
        "candidate_momentum": cand_momentum,
        "expert_reports": expert_reports,
        "router_action": router_action,
        "applied_directive_sha256": directive_sha,
        "resolved_bee_weights": eff_bee_weights,
        "mean_sign_consensus_ratio": float(np.mean(consensus_scores)) if consensus_scores else 1.0,
        "momentum_prose_leakage": leakage,
    }


# ---------------------------------------------------------------------------
# 3. Closed-Loop Multi-Round Controller & SHA-256 Chain Auditor
# ---------------------------------------------------------------------------


def _build_round_receipt_dict(
    *,
    campaign_id: str,
    round_index: int,
    status: str,
    accepted: bool,
    rolled_back: bool,
    rollback_verified: bool,
    rescued: bool,
    final_eval: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
    anse_energy: float,
    delta_e: float,
    campaign_best_anse_energy: float,
    raw_unguarded_eval: Mapping[str, Any],
    raw_unguarded_metrics: Mapping[str, Any],
    unguarded_ratchet_eval: Mapping[str, Any],
    unguarded_ratchet_accepted: bool,
    candidate_eval: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    momentum_leakage: Mapping[str, float],
    fisher_summary: Mapping[str, Any],
    router_action: str,
    applied_directive_sha256: str | None,
    directive_overridden: bool,
    next_directive: QueenPheromoneDirective,
    pre_state_sha: str,
    cand_state_sha: str,
    final_state_sha: str,
    pre_momentum_sha: str,
    cand_momentum_sha: str,
    final_momentum_sha: str,
    prev_receipt_sha256: str,
) -> dict[str, Any]:
    """Assemble and SHA-256-sign one immutable inter-round trophallaxis receipt."""

    def _clean_f(val: float) -> float | None:
        return float(val) if math.isfinite(float(val)) else None

    ung_rat_nll = float(unguarded_ratchet_eval["nll_mean"])
    final_nll = float(final_eval["nll_mean"])
    receipt: dict[str, Any] = {
        "object": ROUND_RECEIPT_OBJECT,
        "schema": INTER_ROUND_TROPHALLAXIS_SCHEMA,
        "campaign_id": campaign_id,
        "round_index": int(round_index),
        "status": status,
        "accepted": bool(accepted),
        "rolled_back": bool(rolled_back),
        "rollback_bitwise_verified": bool(rollback_verified),
        "rescued_from_discarded_round": bool(rescued),
        "nll_mean": final_nll,
        "ece_value": float(final_eval["ece_value"]),
        "anse_energy": float(anse_energy),
        "delta_e": float(delta_e),
        "campaign_best_anse_energy": float(campaign_best_anse_energy),
        "general_drift_ratio": _clean_f(final_metrics["general_drift_ratio"]),
        "worst_domain_drift_ratio": _clean_f(final_metrics["worst_domain_drift_ratio"]),
        "guard_ok": bool(final_metrics["guard_ok"]) if accepted else False,
        "domain_nlls": dict(final_eval["domain_nlls"]),
        "domain_eces": dict(final_eval["domain_eces"]),
        "momentum_prose_leakage": {k: float(v) for k, v in momentum_leakage.items()},
        "fisher_summary": dict(fisher_summary),
        "router_action": router_action,
        "applied_directive_sha256": applied_directive_sha256,
        "directive_overridden": bool(directive_overridden),
        "next_pheromone_directive": next_directive.to_dict(),
        "unguarded_diloco_metrics": {
            "raw_all_finite": bool(raw_unguarded_eval.get("all_finite", True)),
            "raw_nll_mean": _clean_f(raw_unguarded_eval["nll_mean"]),
            "raw_general_drift_ratio": _clean_f(raw_unguarded_metrics["general_drift_ratio"]),
            "raw_worst_domain_drift_ratio": _clean_f(raw_unguarded_metrics["worst_domain_drift_ratio"]),
            "raw_guard_ok": bool(raw_unguarded_metrics["guard_ok"]),
            "would_be_rejected_by_guard": not bool(raw_unguarded_metrics["guard_ok"]),
            "ratchet_accepted": bool(unguarded_ratchet_accepted),
            "ratchet_nll_mean": ung_rat_nll,
            "cumulative_nll_delta_vs_unguarded_ratchet": final_nll - ung_rat_nll,
        },
        "candidate_metrics": {
            "all_finite": bool(candidate_eval.get("all_finite", True)),
            "nll_mean": _clean_f(candidate_eval["nll_mean"]),
            "ece_value": _clean_f(candidate_eval["ece_value"]),
            "general_drift_ratio": _clean_f(candidate_metrics["general_drift_ratio"]),
            "worst_domain_drift_ratio": _clean_f(candidate_metrics["worst_domain_drift_ratio"]),
            "anse_energy": float(candidate_metrics["anse_energy"]),
            "guard_ok": bool(candidate_metrics["guard_ok"]),
        },
        "pre_state_sha256": pre_state_sha,
        "candidate_state_sha256": cand_state_sha,
        "final_state_sha256": final_state_sha,
        "pre_momentum_sha256": pre_momentum_sha,
        "candidate_momentum_sha256": cand_momentum_sha,
        "final_momentum_sha256": final_momentum_sha,
        "prev_receipt_sha256": prev_receipt_sha256,
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


class InterRoundRoyalJellyTrophallaxisController:
    """Stateful closed-loop controller embedding Reine-Cortex inside multi-round DiLoCo barriers."""

    def __init__(
        self,
        base_checkpoint: nn.Module | Mapping[str, Any],
        *,
        config: InterRoundTrophallaxisConfig | None = None,
        eval_fn: Callable[[Any], Mapping[str, Any]] | None = None,
        probe_chunks: Sequence[Mapping[str, Any]] | None = None,
        eval_chunks: Sequence[Mapping[str, Any]] | None = None,
        p_null_by_expert: Mapping[str | int, Any] | None = None,
        w_norm_base: torch.Tensor | None = None,
        normed_unit_probe: torch.Tensor | Sequence[torch.Tensor] | None = None,
        head_weight: torch.Tensor | None = None,
        probe_targets: torch.Tensor | Sequence[torch.Tensor] | None = None,
        is_general_mask: torch.Tensor | Sequence[bool] | np.ndarray | None = None,
        campaign_id: str = "la-ruche-inter-round-trophallaxis",
        initial_directive: QueenPheromoneDirective | Mapping[str, Any] | None = None,
        initial_domain_round_counts: Mapping[str, int] | None = None,
    ) -> None:
        self._cfg = config if config is not None else InterRoundTrophallaxisConfig()
        if not isinstance(self._cfg, InterRoundTrophallaxisConfig):
            raise TypeError(
                f"config must be InterRoundTrophallaxisConfig or None, got {type(self._cfg).__name__}"
            )
        self._qc_cfg = self._cfg.to_queen_cortex_config()
        if eval_fn is not None and not callable(eval_fn):
            raise TypeError(f"eval_fn must be callable or None, got {type(eval_fn).__name__}")
        self._eval_fn = eval_fn
        self._campaign_id = _require_nonempty_str(campaign_id, "campaign_id")

        self._is_model = isinstance(base_checkpoint, nn.Module)
        if self._is_model:
            self._work_model: nn.Module | None = (
                base_checkpoint if self._cfg.inplace else copy.deepcopy(base_checkpoint)
            )
            self._base_blocks = extract_checkpoint_blocks(base_checkpoint)
            self._base_sha = compute_checkpoint_sha256(base_checkpoint)
            head_mod = getattr(self._work_model, "head", None) or getattr(self._work_model, "lm_head", None)
            ln_mod = getattr(self._work_model, "ln", None) or getattr(self._work_model, "norm", None)
            if head_mod is None or ln_mod is None:
                raise ValueError("Model must expose 'head'/'lm_head' and 'ln'/'norm'")
            vocab_sz = int(head_mod.weight.shape[0])
            self._p_chunks = (
                _validate_probe_chunks(
                    probe_chunks, vocab_size=vocab_sz, general_prefixes=self._cfg.general_prefixes, name="probe_chunks"
                )
                if probe_chunks is not None
                else None
            )
            self._e_chunks = (
                _validate_probe_chunks(
                    eval_chunks, vocab_size=vocab_sz, general_prefixes=self._cfg.general_prefixes, name="eval_chunks"
                )
                if eval_chunks is not None
                else self._p_chunks
            )
            if self._eval_fn is None and self._e_chunks is None:
                raise ValueError("Model mode requires probe_chunks/eval_chunks or eval_fn")
        elif isinstance(base_checkpoint, Mapping):
            self._work_model = None
            self._base_blocks, _, _, _ = _normalize_expert_store(base_checkpoint, "base_checkpoint")
            self._base_sha = compute_checkpoint_sha256(self._base_blocks)
            self._p_chunks = None
            self._e_chunks = None
            if self._eval_fn is None:
                raise ValueError("Dict checkpoint mode requires eval_fn")
        else:
            raise TypeError("base_checkpoint must be an nn.Module or non-empty Mapping")

        _partition_checkpoint_blocks(self._base_blocks)
        self._default_p_null = p_null_by_expert
        self._w_norm_base = w_norm_base
        self._normed_unit_probe = normed_unit_probe
        self._head_weight = head_weight
        self._probe_targets = probe_targets
        self._is_general_mask = is_general_mask

        self._current_blocks = _clone_blocks(self._base_blocks)
        self._current_sha = self._base_sha
        self._momentum_state = initialize_zero_momentum_state(self._base_blocks)
        self._momentum_sha = compute_momentum_state_sha256(self._momentum_state)

        self._unguarded_ratchet_blocks = _clone_blocks(self._base_blocks)
        self._unguarded_raw_blocks = _clone_blocks(self._base_blocks)
        self._unguarded_raw_momentum = initialize_zero_momentum_state(self._base_blocks)
        self._unguarded_ratchet_anse = 0.0

        self._base_eval = self._evaluate_blocks(self._base_blocks, "base_eval")
        self._unguarded_ratchet_eval = dict(self._base_eval)
        self._best_anse_energy = 0.0
        self._round_index = 0
        self._prev_receipt_sha256 = GENESIS_RECEIPT_SHA256
        self._receipts: list[dict[str, Any]] = []

        if initial_directive is not None:
            _parse_and_verify_directive(initial_directive)
            self._active_directive: QueenPheromoneDirective | dict[str, Any] | None = (
                initial_directive.to_dict() if isinstance(initial_directive, QueenPheromoneDirective) else dict(initial_directive)
            )
        else:
            self._active_directive = None

        if initial_domain_round_counts is not None:
            if not isinstance(initial_domain_round_counts, Mapping):
                raise TypeError("initial_domain_round_counts must be a Mapping or None")
            self._domain_round_counts: dict[str, int] = {
                _require_nonempty_str(str(k), "domain"): _require_int(v, f"domain_round_counts[{k!r}]", min_value=0)
                for k, v in initial_domain_round_counts.items()
            }
        else:
            self._domain_round_counts = {
                d: 0
                for d in self._base_eval["domain_nlls"]
                if not _is_general_domain(d, self._cfg.general_prefixes)
            }

    @property
    def campaign_id(self) -> str:
        return self._campaign_id

    @property
    def round_index(self) -> int:
        return self._round_index

    @property
    def base_sha256(self) -> str:
        return self._base_sha

    @property
    def current_sha256(self) -> str:
        return self._current_sha

    @property
    def momentum_sha256(self) -> str:
        return self._momentum_sha

    @property
    def best_anse_energy(self) -> float:
        return self._best_anse_energy

    @property
    def active_directive(self) -> dict[str, Any] | None:
        return dict(self._active_directive) if self._active_directive is not None else None

    @property
    def domain_round_counts(self) -> dict[str, int]:
        return dict(self._domain_round_counts)

    @property
    def receipts(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(r) for r in self._receipts)

    @property
    def head_receipt_sha256(self) -> str:
        return self._prev_receipt_sha256

    @property
    def current_checkpoint(self) -> Any:
        if self._work_model is not None:
            apply_checkpoint_blocks_inplace(self._work_model, self._current_blocks)
            return self._work_model
        return _clone_blocks(self._current_blocks)

    @property
    def momentum_state(self) -> dict[str, dict[str, torch.Tensor]]:
        return _clone_blocks(self._momentum_state)

    def _evaluate_blocks(
        self,
        blk_state: Mapping[str, Mapping[str, torch.Tensor]],
        label: str,
    ) -> dict[str, Any]:
        if self._eval_fn is not None:
            if self._work_model is not None:
                apply_checkpoint_blocks_inplace(self._work_model, blk_state)
                raw = self._eval_fn(self._work_model)
            else:
                raw = self._eval_fn(blk_state)
            return _normalize_eval_result(raw, self._cfg.general_prefixes, label)
        assert self._work_model is not None and self._e_chunks is not None
        apply_checkpoint_blocks_inplace(self._work_model, blk_state)
        return _evaluate_model_on_chunks(
            self._work_model,
            self._e_chunks,
            score_from=self._cfg.score_from,
            general_prefixes=self._cfg.general_prefixes,
        )

    def _step_unguarded_diloco(
        self,
        forager_bee_deltas: Sequence[Mapping[str | int, Any]] | Mapping[str | int, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]:
        """Advance parallel unguarded DiLoCo tracker (raw outer momentum + safety ratchet)."""
        exp_parent, rtr_parent, _ = _partition_checkpoint_blocks(self._unguarded_raw_blocks)
        exp_bees, rtr_bees, bees_finite = _partition_bee_deltas(
            forager_bee_deltas, set(exp_parent.keys()), set(rtr_parent.keys())
        )
        if not bees_finite:
            raw_eval: dict[str, Any] = {
                "all_finite": False,
                "nll_mean": float("inf"),
                "ece_value": 1.0,
                "domain_nlls": dict(self._base_eval["domain_nlls"]),
                "domain_eces": dict(self._base_eval["domain_eces"]),
            }
            raw_metrics = _score_against_base(raw_eval, self._base_eval, self._qc_cfg)
            return raw_eval, raw_metrics, dict(self._unguarded_ratchet_eval), False

        next_raw = _clone_blocks(self._unguarded_raw_blocks)
        next_m = _clone_blocks(self._unguarded_raw_momentum)
        for bid, sub_m in next_m.items():
            target_bees = rtr_bees if bid.endswith(_ROUTER_SUFFIX) else exp_bees
            for sk, m_prev in sub_m.items():
                peers = [
                    (b[bid][sk] if isinstance(b[bid], Mapping) else b[bid]).detach().to("cpu", dtype=torch.float64)
                    for b in target_bees
                    if bid in b and (not isinstance(b[bid], Mapping) or sk in b[bid])
                ]
                mean_d = (
                    torch.stack(peers, dim=0).mean(dim=0)
                    if peers
                    else torch.zeros_like(m_prev)
                )
                m_new = self._cfg.outer_momentum_beta * m_prev + mean_d
                outer_g = (mean_d + self._cfg.outer_momentum_beta * m_new) if self._cfg.nesterov else m_new
                step = self._cfg.outer_lr * outer_g
                next_m[bid][sk] = m_new
                base_t = next_raw[bid][sk]
                next_raw[bid][sk] = (base_t.detach().to("cpu", dtype=torch.float64) + step).to(
                    device=base_t.device, dtype=base_t.dtype
                )

        self._unguarded_raw_blocks = next_raw
        self._unguarded_raw_momentum = next_m
        raw_eval = self._evaluate_blocks(next_raw, "unguarded_raw_eval")
        raw_metrics = _score_against_base(raw_eval, self._base_eval, self._qc_cfg)

        # Unguarded safety ratchet accepts only if guard_ok and Delta E < 0 (else discards round)
        ung_accepted = bool(
            raw_metrics["guard_ok"]
            and float(raw_metrics["anse_energy"]) < self._unguarded_ratchet_anse - self._cfg.min_energy_improvement
        )
        if ung_accepted:
            self._unguarded_ratchet_blocks = _clone_blocks(next_raw)
            self._unguarded_ratchet_eval = dict(raw_eval)
            self._unguarded_ratchet_anse = float(raw_metrics["anse_energy"])
        return raw_eval, raw_metrics, dict(self._unguarded_ratchet_eval), ung_accepted

    @torch.no_grad()
    def step_outer_round(
        self,
        forager_bee_deltas: Sequence[Mapping[str | int, Any]] | Mapping[str | int, Any],
        *,
        p_null_by_expert: Mapping[str | int, Any] | None = None,
        probe_chunks: Sequence[Mapping[str, Any]] | None = None,
        w_norm_base: torch.Tensor | None = None,
        normed_unit_probe: torch.Tensor | Sequence[torch.Tensor] | None = None,
        head_weight: torch.Tensor | None = None,
        probe_targets: torch.Tensor | Sequence[torch.Tensor] | None = None,
        is_general_mask: torch.Tensor | Sequence[bool] | np.ndarray | None = None,
        expert_causal_gains_nats: Mapping[str | int, float] | None = None,
        expert_prose_harm_nats: Mapping[str | int, float] | None = None,
        router_expert_kl_mismatch: Mapping[str | int, float] | float = 0.0,
        bee_domains: Sequence[str] | Mapping[str, str] | None = None,
        bee_weights: Sequence[float] | None = None,
        override_directive: QueenPheromoneDirective | Mapping[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Execute one closed-loop DiLoCo outer round under Royal-Jelly trophallaxis & ANSE gate."""
        next_round_idx = self._round_index + 1
        pre_blocks = _clone_blocks(self._current_blocks)
        pre_sha = self._current_sha
        pre_momentum = _clone_blocks(self._momentum_state)
        pre_momentum_sha = self._momentum_sha
        pre_eval = self._evaluate_blocks(pre_blocks, "pre_round_eval")

        effective_directive = override_directive if override_directive is not None else self._active_directive
        if effective_directive is not None:
            _parse_and_verify_directive(effective_directive)

        # Resolve per-round probe chunks and finite check
        exp_parent, rtr_parent, ln_parent = _partition_checkpoint_blocks(pre_blocks)
        _, _, ln_base = _partition_checkpoint_blocks(self._base_blocks)
        _, _, bees_all_finite = _partition_bee_deltas(
            forager_bee_deltas, set(exp_parent.keys()), set(rtr_parent.keys())
        )

        # Advance parallel unguarded DiLoCo tracker first (restoring work_model afterward if needed)
        raw_ung_eval, raw_ung_metrics, ung_rat_eval, ung_rat_accepted = self._step_unguarded_diloco(
            forager_bee_deltas
        )

        eff_p_chunks = self._p_chunks
        if probe_chunks is not None and self._work_model is not None:
            head_mod = getattr(self._work_model, "head", None) or getattr(self._work_model, "lm_head", None)
            assert head_mod is not None
            eff_p_chunks = _validate_probe_chunks(
                probe_chunks,
                vocab_size=int(head_mod.weight.shape[0]),
                general_prefixes=self._cfg.general_prefixes,
                name="probe_chunks",
            )

        if not bees_all_finite:
            cand_ok = False
            cand_blocks = pre_blocks
            cand_momentum = pre_momentum
            cand_sha = pre_sha
            cand_momentum_sha = pre_momentum_sha
            cand_eval: dict[str, Any] = {
                "all_finite": False,
                "nll_mean": float("inf"),
                "ece_value": 1.0,
                "domain_nlls": dict(pre_eval["domain_nlls"]),
                "domain_eces": dict(pre_eval["domain_eces"]),
            }
            cand_metrics = _score_against_base(cand_eval, self._base_eval, self._qc_cfg)
            fisher_summary: dict[str, Any] = {"status": "QUARANTINED_NONFINITE_NECTAR"}
            momentum_leakage = measure_momentum_prose_leakage(pre_momentum, p_null_by_expert or self._default_p_null)
            router_action = "router_quarantined_rollback"
            applied_dir_sha = (
                str(
                    effective_directive.directive_sha256
                    if isinstance(effective_directive, QueenPheromoneDirective)
                    else effective_directive["directive_sha256"]
                )
                if effective_directive is not None
                else None
            )
        else:
            eff_p_null = p_null_by_expert if p_null_by_expert is not None else self._default_p_null
            if self._work_model is not None:
                apply_checkpoint_blocks_inplace(self._work_model, pre_blocks)
                if eff_p_null is None and eff_p_chunks is not None:
                    eff_p_null, _, _, _ = _build_model_p_null_and_probe(
                        self._work_model,
                        eff_p_chunks,
                        score_from=self._cfg.score_from,
                        general_prefixes=self._cfg.general_prefixes,
                        build_projectors=True,
                    )

            step_res = apply_inter_round_royal_jelly_momentum_step(
                pre_blocks,
                forager_bee_deltas,
                pre_momentum,
                eff_p_null,
                directive=effective_directive,
                bee_domains=bee_domains,
                bee_weights=bee_weights,
                config=self._cfg,
            )
            cand_blocks = step_res["candidate_blocks"]
            cand_momentum = step_res["candidate_momentum"]
            momentum_leakage = step_res["momentum_prose_leakage"]
            router_action = step_res["router_action"]
            applied_dir_sha = step_res["applied_directive_sha256"]

            # Resolve final norm calibration parameters (ker(G_D0) Fisher norm + safe tau)
            w_n_b = w_norm_base if w_norm_base is not None else self._w_norm_base
            b_n_b: torch.Tensor | None = None
            h_w = head_weight if head_weight is not None else self._head_weight
            if ln_base:
                ln_key = next(iter(ln_base))
                if w_n_b is None:
                    w_n_b = ln_base[ln_key]["weight"]
                    b_n_b = ln_base[ln_key].get("bias")
            if self._work_model is not None:
                head_mod = getattr(self._work_model, "head", None) or getattr(self._work_model, "lm_head", None)
                if h_w is None and head_mod is not None:
                    h_w = head_mod.weight
            assert w_n_b is not None and h_w is not None, "w_norm_base and head_weight are required"

            eff_unit = normed_unit_probe if normed_unit_probe is not None else self._normed_unit_probe
            eff_tgt = probe_targets if probe_targets is not None else self._probe_targets
            eff_mask = is_general_mask if is_general_mask is not None else self._is_general_mask
            if self._work_model is not None and eff_p_chunks is not None and eff_unit is None:
                apply_checkpoint_blocks_inplace(self._work_model, cand_blocks)
                _, eff_unit, eff_tgt, eff_mask = _build_model_p_null_and_probe(
                    self._work_model,
                    eff_p_chunks,
                    score_from=self._cfg.score_from,
                    general_prefixes=self._cfg.general_prefixes,
                    build_projectors=False,
                )
            assert eff_unit is not None and eff_tgt is not None and eff_mask is not None, (
                "normed_unit_probe, probe_targets, and is_general_mask are required"
            )

            w_cal, b_cal, fisher_meta = compute_ker_g_d0_fisher_norm_step(
                w_n_b,
                eff_unit,
                h_w,
                eff_tgt,
                eff_mask,
                b_norm_base=b_n_b,
                damping=self._cfg.fisher_damping,
                step_scales=self._cfg.fisher_step_scales,
                tau_grid=self._cfg.tau_grid,
                max_rel_delta=self._cfg.max_norm_rel_delta,
                max_d0_rel=self._cfg.probe_max_d0_rel,
                ece_weight=self._cfg.ece_energy_weight,
                sv_rel_tol=self._cfg.d0_grad_sv_rel_tol,
            )
            if ln_parent:
                ln_key = next(iter(ln_parent))
                cand_blocks[ln_key]["weight"] = w_cal.to(cand_blocks[ln_key]["weight"].dtype)
                if b_cal is not None and "bias" in cand_blocks[ln_key]:
                    cand_blocks[ln_key]["bias"] = b_cal.to(cand_blocks[ln_key]["bias"].dtype)

            cand_sha = compute_checkpoint_sha256(cand_blocks)
            cand_momentum_sha = compute_momentum_state_sha256(cand_momentum)
            cand_eval = self._evaluate_blocks(cand_blocks, "royal_jelly_trophallaxis_eval")
            cand_metrics = _score_against_base(cand_eval, self._base_eval, self._qc_cfg)
            fisher_summary = {
                "eta_fisher": float(fisher_meta["eta_fisher"]),
                "tau_temp": float(fisher_meta["tau_temp"]),
                "g_d0_rank": int(fisher_meta["g_d0_rank"]),
                "ker_g_d0_dim": int(fisher_meta["ker_g_d0_dim"]),
                "first_order_d0_orthogonality": float(fisher_meta["first_order_d0_orthogonality"]),
                "mean_sign_consensus_ratio": float(step_res["mean_sign_consensus_ratio"]),
            }
            delta_e = float(cand_metrics["anse_energy"]) - self._best_anse_energy
            cand_ok = bool(cand_metrics["guard_ok"] and delta_e < -self._cfg.min_energy_improvement)

        if cand_ok:
            accepted = True
            rolled_back = False
            rollback_verified = False
            final_blocks = cand_blocks
            final_momentum = cand_momentum
            final_eval = cand_eval
            final_metrics = cand_metrics
            anse_energy = float(cand_metrics["anse_energy"])
            delta_e = anse_energy - self._best_anse_energy
            self._best_anse_energy = anse_energy
            status = "TROPHALLAXIS_ROUND_COMMITTED"
            # Increment visit counts for non-general domains improved or actively foraged
            foraged_doms: set[str] = set()
            if isinstance(bee_domains, Mapping):
                foraged_doms.update(str(v) for v in bee_domains.values())
            elif isinstance(bee_domains, Sequence) and not isinstance(bee_domains, (str, bytes)):
                foraged_doms.update(str(v) for v in bee_domains)
            for dom in final_eval["domain_nlls"]:
                if _is_general_domain(dom, self._cfg.general_prefixes):
                    continue
                self._domain_round_counts.setdefault(dom, 0)
                if dom in foraged_doms or final_eval["domain_nlls"][dom] < pre_eval["domain_nlls"].get(dom, float("inf")):
                    self._domain_round_counts[dom] += 1
        else:
            accepted = False
            rolled_back = True
            final_blocks = pre_blocks
            final_momentum = pre_momentum
            final_eval = pre_eval
            final_metrics = _score_against_base(pre_eval, self._base_eval, self._qc_cfg)
            anse_energy = self._cfg.maximum_pain_energy
            delta_e = self._cfg.maximum_pain_energy
            status = "ROLLED_BACK_ANSE_MAXIMUM_PAIN"

        self._current_blocks = _clone_blocks(final_blocks)
        self._momentum_state = _clone_blocks(final_momentum)
        if self._work_model is not None:
            apply_checkpoint_blocks_inplace(self._work_model, self._current_blocks)
            final_sha = compute_checkpoint_sha256(self._work_model)
        else:
            final_sha = compute_checkpoint_sha256(self._current_blocks)
        final_momentum_sha = compute_momentum_state_sha256(self._momentum_state)

        if rolled_back:
            rollback_verified = (final_sha == pre_sha) and (final_momentum_sha == pre_momentum_sha)
            if not rollback_verified:
                raise RuntimeError(
                    "Atomic bitwise rollback failed SHA-256 verification across checkpoint/momentum"
                )

        self._current_sha = final_sha
        self._momentum_sha = final_momentum_sha

        # Compute LOEO metrics on intact/committed checkpoint and emit next-round QueenPheromoneDirective
        if expert_causal_gains_nats is not None and expert_prose_harm_nats is not None:
            gains_in, harms_in = expert_causal_gains_nats, expert_prose_harm_nats
        elif self._work_model is not None and eff_p_chunks is not None:
            gains_in, harms_in = _compute_loeo_expert_metrics(
                self._work_model,
                eff_p_chunks,
                score_from=self._cfg.score_from,
                general_prefixes=self._cfg.general_prefixes,
            )
        else:
            gains_in = {ek: 0.0 for ek in sorted(exp_parent.keys())}
            harms_in = {ek: 0.0 for ek in sorted(exp_parent.keys())}

        next_directive = emit_queen_pheromone_campaign_directive(
            gains_in,
            harms_in,
            router_expert_kl_mismatch,
            final_eval["domain_nlls"],
            final_eval["domain_eces"],
            anchor_gain_floor_nats=self._cfg.anchor_gain_floor_nats,
            coadapt_kl_ceiling=self._cfg.coadapt_kl_ceiling,
            temperature=self._cfg.foraging_temperature,
            prose_harm_sink_floor_nats=self._cfg.corruptor_prose_harm_floor_nats,
            domain_round_counts=self._domain_round_counts,
            ucb_exploration_coeff=self._cfg.ucb_exploration_coeff,
            general_prefixes=self._cfg.general_prefixes,
            campaign_id=self._campaign_id,
            round_index=next_round_idx + 1,
        )
        self._active_directive = next_directive.to_dict()

        rescued = bool(accepted and (not raw_ung_metrics["guard_ok"]))
        receipt = _build_round_receipt_dict(
            campaign_id=self._campaign_id,
            round_index=next_round_idx,
            status=status,
            accepted=accepted,
            rolled_back=rolled_back,
            rollback_verified=rollback_verified,
            rescued=rescued,
            final_eval=final_eval,
            final_metrics=final_metrics,
            anse_energy=anse_energy,
            delta_e=delta_e,
            campaign_best_anse_energy=self._best_anse_energy,
            raw_unguarded_eval=raw_ung_eval,
            raw_unguarded_metrics=raw_ung_metrics,
            unguarded_ratchet_eval=ung_rat_eval,
            unguarded_ratchet_accepted=ung_rat_accepted,
            candidate_eval=cand_eval,
            candidate_metrics=cand_metrics,
            momentum_leakage=momentum_leakage,
            fisher_summary=fisher_summary,
            router_action=router_action,
            applied_directive_sha256=applied_dir_sha,
            directive_overridden=(override_directive is not None),
            next_directive=next_directive,
            pre_state_sha=pre_sha,
            cand_state_sha=cand_sha,
            final_state_sha=final_sha,
            pre_momentum_sha=pre_momentum_sha,
            cand_momentum_sha=cand_momentum_sha,
            final_momentum_sha=final_momentum_sha,
            prev_receipt_sha256=self._prev_receipt_sha256,
        )
        self._round_index = next_round_idx
        self._prev_receipt_sha256 = str(receipt["receipt_sha256"])
        self._receipts.append(receipt)
        return self.current_checkpoint, receipt

    @torch.no_grad()
    def run_campaign(
        self,
        rounds: Sequence[Mapping[str, Any]],
    ) -> tuple[Any, dict[str, Any]]:
        """Execute a sequence of outer DiLoCo rounds and return ``(checkpoint, chain_summary)``."""
        if isinstance(rounds, (str, bytes)) or not isinstance(rounds, Sequence) or not rounds:
            raise ValueError("rounds must be a non-empty sequence of round mappings")

        round_receipts: list[dict[str, Any]] = []
        for idx, spec in enumerate(rounds):
            if not isinstance(spec, Mapping) or "forager_bee_deltas" not in spec:
                raise ValueError(f"rounds[{idx}] must be a mapping containing 'forager_bee_deltas'")
            _, rec = self.step_outer_round(
                spec["forager_bee_deltas"],
                p_null_by_expert=spec.get("p_null_by_expert"),
                probe_chunks=spec.get("probe_chunks"),
                w_norm_base=spec.get("w_norm_base"),
                normed_unit_probe=spec.get("normed_unit_probe"),
                head_weight=spec.get("head_weight"),
                probe_targets=spec.get("probe_targets"),
                is_general_mask=spec.get("is_general_mask"),
                expert_causal_gains_nats=spec.get("expert_causal_gains_nats"),
                expert_prose_harm_nats=spec.get("expert_prose_harm_nats"),
                router_expert_kl_mismatch=spec.get("router_expert_kl_mismatch", 0.0),
                bee_domains=spec.get("bee_domains"),
                bee_weights=spec.get("bee_weights"),
                override_directive=spec.get("override_directive"),
            )
            round_receipts.append(rec)

        chain_audit = verify_inter_round_trophallaxis_chain(
            self._receipts,
            campaign_id=self._campaign_id,
            min_energy_improvement=self._cfg.min_energy_improvement,
            max_domain_drift_ratio=self._cfg.max_domain_drift_ratio,
        )
        base_nll = float(self._base_eval["nll_mean"])
        final_nll = float(self._receipts[-1]["nll_mean"])
        ung_rat_nll = float(self._receipts[-1]["unguarded_diloco_metrics"]["ratchet_nll_mean"])
        max_d0_ratio = max(
            float(r["general_drift_ratio"])
            for r in self._receipts
            if r["general_drift_ratio"] is not None
        )
        max_leak = max(
            float(r["momentum_prose_leakage"]["frobenius_leakage"])
            for r in self._receipts
        )
        summary_core: dict[str, Any] = {
            "object": CHAIN_SUMMARY_OBJECT,
            "schema": INTER_ROUND_TROPHALLAXIS_SCHEMA,
            "campaign_id": self._campaign_id,
            "total_rounds": len(self._receipts),
            "accepted_rounds": sum(1 for r in self._receipts if r["accepted"]),
            "rolled_back_rounds": sum(1 for r in self._receipts if r["rolled_back"]),
            "rescued_rounds": sum(1 for r in self._receipts if r["rescued_from_discarded_round"]),
            "base_nll_mean": base_nll,
            "final_nll_mean": final_nll,
            "cumulative_nll_delta": final_nll - base_nll,
            "unguarded_ratchet_final_nll": ung_rat_nll,
            "cumulative_nll_delta_vs_unguarded_ratchet": final_nll - ung_rat_nll,
            "final_general_drift_ratio": self._receipts[-1]["general_drift_ratio"],
            "max_general_drift_ratio_observed": max_d0_ratio,
            "max_momentum_prose_leakage": max_leak,
            "final_anse_energy": self._best_anse_energy,
            "chain_verified": bool(chain_audit["valid"]),
            "chain_head_sha256": self._prev_receipt_sha256,
            "round_receipts": round_receipts,
        }
        summary_core["summary_sha256"] = compute_payload_sha256(summary_core)
        return self.current_checkpoint, summary_core


def verify_inter_round_trophallaxis_receipt_hash(receipt: Mapping[str, Any]) -> bool:
    """Verify object, schema, signed pheromone directive, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a Mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != ROUND_RECEIPT_OBJECT or receipt.get("schema") != INTER_ROUND_TROPHALLAXIS_SCHEMA:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        if claimed != compute_payload_sha256(core):
            return False
    except (TypeError, ValueError):
        return False
    next_dir = receipt.get("next_pheromone_directive")
    return isinstance(next_dir, Mapping) and verify_queen_pheromone_directive_signature(next_dir)


def verify_inter_round_trophallaxis_chain(
    receipts: Sequence[Mapping[str, Any]],
    *,
    campaign_id: str | None = None,
    min_energy_improvement: float = 0.0,
    max_domain_drift_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO,
    require_strict_accepted_descent: bool = True,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Audit SHA-256 hash chaining, state/momentum continuity, pheromone propagation, and ANSE descent."""
    if isinstance(receipts, (str, bytes)) or not isinstance(receipts, Sequence):
        raise TypeError("receipts must be a Sequence[Mapping]")
    min_imp = _require_float(min_energy_improvement, "min_energy_improvement", min_value=0.0)
    drift_cap = _require_float(max_domain_drift_ratio, "max_domain_drift_ratio", min_value=1.0)
    _require_bool(require_strict_accepted_descent, "require_strict_accepted_descent")
    _require_bool(raise_on_error, "raise_on_error")
    expected_campaign = _require_nonempty_str(campaign_id, "campaign_id") if campaign_id is not None else None

    violations: list[str] = []
    expected_prev_receipt = GENESIS_RECEIPT_SHA256
    prev_final_state: str | None = None
    prev_final_momentum: str | None = None
    prev_emitted_dir: str | None = None
    prev_accepted_energy = 0.0

    for idx, doc in enumerate(receipts):
        label = f"round[{idx}]"
        if not isinstance(doc, Mapping):
            violations.append(f"{label}: receipt must be a Mapping")
            continue

        if doc.get("campaign_id") != expected_campaign:
            if expected_campaign is None and isinstance(doc.get("campaign_id"), str) and doc["campaign_id"].strip():
                expected_campaign = doc["campaign_id"]
            else:
                violations.append(f"{label}: campaign_id mismatch")

        if doc.get("round_index") != idx + 1 or isinstance(doc.get("round_index"), bool):
            violations.append(f"{label}: expected round_index={idx + 1}, got {doc.get('round_index')!r}")

        if doc.get("prev_receipt_sha256") != expected_prev_receipt:
            violations.append(f"{label}: prev_receipt_sha256 mismatch")

        if not verify_inter_round_trophallaxis_receipt_hash(doc):
            violations.append(f"{label}: receipt_sha256 or pheromone signature invalid")

        for sha_field in (
            "pre_state_sha256",
            "candidate_state_sha256",
            "final_state_sha256",
            "pre_momentum_sha256",
            "candidate_momentum_sha256",
            "final_momentum_sha256",
        ):
            val = doc.get(sha_field)
            if not isinstance(val, str) or not _SHA256_HEX_RE.match(val):
                violations.append(f"{label}: malformed {sha_field}")

        if prev_final_state is not None and doc.get("pre_state_sha256") != prev_final_state:
            violations.append(f"{label}: state hash discontinuity across rounds")
        if prev_final_momentum is not None and doc.get("pre_momentum_sha256") != prev_final_momentum:
            violations.append(f"{label}: momentum hash discontinuity across rounds")
        if (
            not doc.get("directive_overridden", False)
            and prev_emitted_dir is not None
            and doc.get("applied_directive_sha256") != prev_emitted_dir
        ):
            violations.append(f"{label}: applied_directive_sha256 does not match prior emitted directive")

        if doc.get("accepted") is True:
            gen_r = doc.get("general_drift_ratio")
            worst_r = doc.get("worst_domain_drift_ratio")
            cur_e = doc.get("anse_energy")
            if doc.get("guard_ok") is not True:
                violations.append(f"{label}: accepted round must have guard_ok=True")
            if gen_r is None or float(gen_r) > drift_cap + 1e-12:
                violations.append(f"{label}: general_drift_ratio {gen_r} exceeds {drift_cap}")
            if worst_r is None or float(worst_r) > drift_cap + 1e-12:
                violations.append(f"{label}: worst_domain_drift_ratio {worst_r} exceeds {drift_cap}")
            if not isinstance(cur_e, (int, float)) or float(cur_e) >= ANSE_MAXIMUM_PAIN * 0.5:
                violations.append(f"{label}: accepted round has invalid or pain-level ANSE energy")
            elif require_strict_accepted_descent and float(cur_e) >= prev_accepted_energy - min_imp:
                violations.append(
                    f"{label}: accepted ANSE energy {float(cur_e):.6f} >= prior {prev_accepted_energy:.6f}"
                )
            else:
                prev_accepted_energy = float(cur_e)
        else:
            if doc.get("rolled_back") is not True or doc.get("rollback_bitwise_verified") is not True:
                violations.append(f"{label}: rejected round must verify atomic bitwise rollback")
            if doc.get("final_state_sha256") != doc.get("pre_state_sha256"):
                violations.append(f"{label}: rolled-back state SHA-256 differs from pre-round state")
            if doc.get("final_momentum_sha256") != doc.get("pre_momentum_sha256"):
                violations.append(f"{label}: rolled-back momentum SHA-256 differs from pre-round momentum")
            cur_e = doc.get("anse_energy")
            if not isinstance(cur_e, (int, float)) or float(cur_e) < ANSE_MAXIMUM_PAIN * 0.5:
                violations.append(f"{label}: rolled-back round must record ANSE Maximum Pain")

        prev_final_state = str(doc.get("final_state_sha256", ""))
        prev_final_momentum = str(doc.get("final_momentum_sha256", ""))
        next_dir = doc.get("next_pheromone_directive")
        prev_emitted_dir = (
            str(next_dir.get("directive_sha256"))
            if isinstance(next_dir, Mapping) and isinstance(next_dir.get("directive_sha256"), str)
            else None
        )
        expected_prev_receipt = str(doc.get("receipt_sha256", ""))

    valid = len(violations) == 0
    if raise_on_error and not valid:
        raise ValueError(f"Inter-round trophallaxis chain audit failed: {violations[0]}")

    return {
        "valid": valid,
        "entry_count": len(receipts),
        "chain_head_sha256": expected_prev_receipt,
        "violations": violations,
    }


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "CHAIN_SUMMARY_OBJECT",
    "DEFAULT_FORAGING_WEIGHT_FLOOR",
    "DEFAULT_OUTER_LR",
    "DEFAULT_OUTER_MOMENTUM_BETA",
    "GENESIS_RECEIPT_SHA256",
    "INTER_ROUND_TROPHALLAXIS_SCHEMA",
    "ROUND_RECEIPT_OBJECT",
    "InterRoundRoyalJellyTrophallaxisController",
    "InterRoundTrophallaxisConfig",
    "apply_inter_round_royal_jelly_momentum_step",
    "compute_momentum_state_sha256",
    "compute_prose_basis_from_projector",
    "initialize_zero_momentum_state",
    "measure_momentum_prose_leakage",
    "resolve_pheromone_bee_weights",
    "verify_inter_round_trophallaxis_chain",
    "verify_inter_round_trophallaxis_receipt_hash",
]

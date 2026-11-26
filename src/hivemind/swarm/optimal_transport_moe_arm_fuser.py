# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Optimal-Transport Hungarian & Orthogonal Procrustes Cross-Arm MoE Expert and Router Fuser.

Implements **Gen-3 Direction D3** (Zenodo ``10.5281/zenodo.24542441`` Section 6 /
Theorem 3.8 and Lean 4 theorem
``ANSE.ReineCortexDirections.thm_rcd_d3_permutation_transport_moe_invariance``), resolving
the ``PILOT5`` cross-arm router-expert co-adaptation mismatch (``RC_T8``) in CPU
``0 MiB`` VRAM / ``0`` gradient steps:

1. **Exact ``O(E^3)`` Hungarian Optimal-Transport Assignment**
   (:func:`solve_exact_hungarian_assignment`,
   :func:`compute_cross_arm_expert_similarity_matrix`):
   Recovers the globally optimal ``1``-to-``1`` expert slot bijection ``perm`` mapping
   ``Arm_B``'s permuted experts onto ``Arm_A``'s slots from combined probe router-logit
   correlation, Procrustes-aligned expert functional output cosine similarity, and
   optional aligned parameter similarity (zero permutation labels required).
2. **Orthogonal Procrustes Hidden & LoRA Rank Alignment + ``P_0`` Synergy**
   (:func:`match_moe_experts_hungarian_procrustes`,
   :func:`align_lora_rank_basis_procrustes`,
   :func:`fuse_aligned_moe_layer_weights`):
   Computes linear CKA + SVD Orthogonal Procrustes ``R_{B->A}`` aligning rotated hidden
   spaces before functional matching and weight transport (``router_B[perm] @ R``,
   ``W_in_B @ R``, ``R^T @ W_out_B``), resolves LoRA factor basis drift ``(B S, S^T A)``,
   projects deltas strictly into the general-prose AlphaEdit null-space ``P_0``, and
   performs orthogonal residual synergy across arms.
3. **ANSE-Gated Cross-Arm Fusion & Atomic Bitwise Rollback**
   (:func:`execute_verified_ot_moe_arm_fusion`):
   Supports ``CrowdV1WithEngram`` (``CrowdLoRAMoEMLP``, ``CrowdFineGrainedMoEMLP``) and
   HF-style MoE models, optionally performs 2-matmul affine LayerNorm + folded
   temperature calibration, and commits iff ``ANSE Delta E < 0`` against donor arms and
   ``D0 / worst_domain <= +1.0%``; otherwise enforces ``ANSE_MAXIMUM_PAIN`` (``1e6``)
   atomic bitwise rollback with SHA-256 provenance receipts.
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

from hivemind.swarm.crowdgpt_moe_opendiloco import CrowdFineGrainedMoEMLP, CrowdLoRAMoEMLP
from hivemind.swarm.domain_matched_moe_arm_composer import (
    _project_dense_tensor_in_nullspace,
    _project_lora_pair_in_nullspace,
    fast_affine_layernorm_temperature_grid,
)
from hivemind.swarm.logit_calibration_temperature_surgeon import (
    fold_temperature_into_norm_params,
)
from hivemind.swarm.lora_svd_mode_surgery import decompose_lora_qr_svd
from hivemind.swarm.natural_gradient_trust_wsp_surgeon import (
    ANSE_MAXIMUM_PAIN,
    _eval_crowd_model_domain_nlls,
    _parse_eval_domain_nll,
    _validate_bool,
    _validate_chunks,
    _validate_nonneg_finite,
    _validate_nonneg_int,
    _validate_positive_finite,
    _validate_positive_int,
    canonical_json_bytes,
    compute_payload_sha256,
    compute_state_sha256,
)
from hivemind.swarm.procrustes_wsp_skill_transfer import (
    _validate_activation_matrix,
    _validate_square_matrix,
    _validate_unit_interval,
    compute_linear_cka_and_orthogonal_procrustes,
)
from hivemind.swarm.tangent_task_vector_surgeon import (
    compute_prose_nullspace_projector_scaled,
)
from hivemind.swarm.wsp_alphaedit_control import forward_model_pre_ln

OT_MOE_FUSER_SCHEMA: str = "hivemind.optimal_transport_moe_arm_fuser/1"
OT_MOE_FUSION_RECEIPT_OBJECT: str = "hivemind.optimal_transport_moe_fusion_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_GENERAL_PREFIXES: tuple[str, ...] = ("E0_", "D0_", "general")
DEFAULT_ROUTER_SIM_WEIGHT: float = 0.65
DEFAULT_FUNC_SIM_WEIGHT: float = 0.35
DEFAULT_WEIGHT_SIM_WEIGHT: float = 0.0
DEFAULT_MIN_CKA_GATE: float = 0.65
DEFAULT_FUSION_ALPHA: float = 0.50
DEFAULT_ORTHOGONAL_SYNERGY_BETA: float = 0.50
DEFAULT_ORTHOGONAL_SYNERGY_SCALE: float = 0.65
DEFAULT_ORTHOGONAL_SYNERGY_RANK: int = 2
DEFAULT_NULLSPACE_SV_REL_TOL: float = 1e-3
DEFAULT_NULLSPACE_VARIANCE_THRESHOLD: float = 0.999
DEFAULT_NULLSPACE_LEAK_TOL: float = 1e-6
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_SCORE_FROM: int = 64
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_LAMBDA_GRID: tuple[float, ...] = tuple(round(0.05 * idx, 2) for idx in range(11, 21))
DEFAULT_TAU_GRID: tuple[float, ...] = (0.98, 1.00, 1.02, 1.04)

VALID_ALIGNMENT_MODES: frozenset[str] = frozenset(
    {"hungarian", "hungarian_procrustes", "hungarian_procrustes_p0_synergy"}
)
VALID_ROLLBACK_TARGETS: frozenset[str] = frozenset({"arm_a", "arm_b", "base"})

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# 1. Configuration, Validation & Provenance Verification
# ---------------------------------------------------------------------------


def verify_ot_moe_fusion_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a Mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if (
        receipt.get("object") != OT_MOE_FUSION_RECEIPT_OBJECT
        or receipt.get("schema") != OT_MOE_FUSER_SCHEMA
    ):
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


def _validate_float_tuple(
    values: Sequence[float],
    name: str,
    *,
    unit_interval: bool = False,
) -> tuple[float, ...]:
    """Validate a non-empty sequence of floats for calibration grids."""
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence) or not values:
        raise ValueError(f"{name} must be a non-empty sequence of floats")
    out: list[float] = []
    for idx, val in enumerate(values):
        out.append(
            _validate_unit_interval(val, f"{name}[{idx}]", allow_zero=True)
            if unit_interval
            else _validate_positive_finite(val, f"{name}[{idx}]")
        )
    return tuple(dict.fromkeys(out))


@dataclass(frozen=True)
class OptimalTransportMoEFuserConfig:
    """Validated configuration for Optimal-Transport Hungarian & Procrustes MoE arm fusion."""

    alignment_mode: str = "hungarian_procrustes_p0_synergy"
    router_sim_weight: float = DEFAULT_ROUTER_SIM_WEIGHT
    func_sim_weight: float = DEFAULT_FUNC_SIM_WEIGHT
    weight_sim_weight: float = DEFAULT_WEIGHT_SIM_WEIGHT
    min_cka_gate: float = DEFAULT_MIN_CKA_GATE
    proper_rotation: bool = False
    align_hidden_space_procrustes: bool = True
    align_lora_rank_procrustes: bool = True
    project_into_prose_nullspace: bool = True
    project_router_into_nullspace: bool = False
    fusion_alpha: float = DEFAULT_FUSION_ALPHA
    router_fusion_alpha: float = DEFAULT_FUSION_ALPHA
    orthogonal_synergy_beta: float = DEFAULT_ORTHOGONAL_SYNERGY_BETA
    orthogonal_synergy_scale: float = DEFAULT_ORTHOGONAL_SYNERGY_SCALE
    orthogonal_synergy_rank: int = DEFAULT_ORTHOGONAL_SYNERGY_RANK
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
    require_beat_both_arms: bool = True
    min_energy_improvement: float = 0.0
    rollback_target: str = "arm_a"
    general_prefixes: tuple[str, ...] = DEFAULT_GENERAL_PREFIXES
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    inplace: bool = False

    def __post_init__(self) -> None:
        if self.alignment_mode not in VALID_ALIGNMENT_MODES:
            raise ValueError(f"alignment_mode must be one of {sorted(VALID_ALIGNMENT_MODES)}")
        if self.rollback_target not in VALID_ROLLBACK_TARGETS:
            raise ValueError(f"rollback_target must be one of {sorted(VALID_ROLLBACK_TARGETS)}")
        w_r = _validate_nonneg_finite(self.router_sim_weight, "router_sim_weight")
        w_f = _validate_nonneg_finite(self.func_sim_weight, "func_sim_weight")
        w_w = _validate_nonneg_finite(self.weight_sim_weight, "weight_sim_weight")
        if (w_r + w_f + w_w) <= 0.0:
            raise ValueError("Sum of similarity weights (router + func + weight) must be > 0.0")
        _validate_unit_interval(self.min_cka_gate, "min_cka_gate", allow_zero=True)
        _validate_bool(self.proper_rotation, "proper_rotation")
        _validate_bool(self.align_hidden_space_procrustes, "align_hidden_space_procrustes")
        _validate_bool(self.align_lora_rank_procrustes, "align_lora_rank_procrustes")
        _validate_bool(self.project_into_prose_nullspace, "project_into_prose_nullspace")
        _validate_bool(self.project_router_into_nullspace, "project_router_into_nullspace")
        _validate_unit_interval(self.fusion_alpha, "fusion_alpha", allow_zero=True)
        _validate_unit_interval(self.router_fusion_alpha, "router_fusion_alpha", allow_zero=True)
        _validate_unit_interval(self.orthogonal_synergy_beta, "orthogonal_synergy_beta", allow_zero=True)
        _validate_positive_finite(self.orthogonal_synergy_scale, "orthogonal_synergy_scale")
        _validate_positive_int(self.orthogonal_synergy_rank, "orthogonal_synergy_rank")
        _validate_bool(self.calibrate_layernorm, "calibrate_layernorm")
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
        _validate_bool(self.enforce_all_domain_guard, "enforce_all_domain_guard")
        safe_r = _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        if safe_r < 1.0:
            raise ValueError(f"anse_safe_general_ratio must be >= 1.0, got {safe_r}")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_bool(self.require_beat_both_arms, "require_beat_both_arms")
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
        _validate_bool(self.inplace, "inplace")


# ---------------------------------------------------------------------------
# 2. Pure Optimal-Transport Hungarian Assignment & MoE Layer Abstraction
# ---------------------------------------------------------------------------


def solve_exact_hungarian_assignment(
    cost_matrix: torch.Tensor | Sequence[Sequence[float]],
) -> tuple[list[int], list[int], float, dict[str, Any]]:
    """Solve the exact ``O(E^3)`` Kuhn-Munkres Hungarian assignment on an ``(E, E)`` cost matrix.

    Returns ``(perm, inv_perm, total_cost, diagnostics)`` where ``perm[i] = j`` assigns
    column/donor slot ``j`` to row/target slot ``i`` minimizing ``sum_i cost[i, perm[i]]``.
    """
    if isinstance(cost_matrix, torch.Tensor):
        c_t = cost_matrix.detach().to(device="cpu", dtype=torch.float64)
    elif isinstance(cost_matrix, Sequence) and not isinstance(cost_matrix, (str, bytes)):
        c_t = torch.tensor(cost_matrix, dtype=torch.float64, device="cpu")
    else:
        raise TypeError(f"cost_matrix must be a 2D Tensor or Sequence, got {type(cost_matrix).__name__}")

    if c_t.ndim != 2 or c_t.shape[0] == 0 or c_t.shape[0] != c_t.shape[1]:
        raise ValueError(f"cost_matrix must be a non-empty square 2D matrix, got shape {tuple(c_t.shape)}")
    if not bool(torch.isfinite(c_t).all().item()):
        raise ValueError("cost_matrix must contain only finite values")

    n = int(c_t.shape[0])
    cost_rows = c_t.tolist()
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            row_vals = cost_rows[i0 - 1]
            for j in range(1, n + 1):
                if not used[j]:
                    cur = row_vals[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    perm = [0] * n
    inv_perm = [0] * n
    for j in range(1, n + 1):
        row_idx = p[j] - 1
        col_idx = j - 1
        perm[row_idx] = col_idx
        inv_perm[col_idx] = row_idx

    total_cost = float(sum(cost_rows[i][perm[i]] for i in range(n)))
    p_mat = torch.zeros((n, n), dtype=torch.float64)
    for i, col_idx in enumerate(perm):
        p_mat[i, col_idx] = 1.0
    birkhoff_err = float(
        (p_mat.sum(dim=0) - 1.0).abs().max().item() + (p_mat.sum(dim=1) - 1.0).abs().max().item()
    )
    return perm, inv_perm, total_cost, {
        "num_experts": n,
        "total_cost": total_cost,
        "mean_cost": total_cost / float(n),
        "is_bijection": sorted(perm) == list(range(n)),
        "birkhoff_marginal_error": birkhoff_err,
        "permutation_matrix": p_mat,
    }


def _resolve_router_linear(mlp: nn.Module) -> nn.Linear:
    """Resolve the router linear layer on ``Crowd*MoEMLP`` or HF-style MoE ``mlp``."""
    router_obj = getattr(mlp, "router", getattr(mlp, "gate", None))
    if isinstance(router_obj, nn.Linear):
        return router_obj
    if router_obj is not None and isinstance(getattr(router_obj, "wg", None), nn.Linear):
        return router_obj.wg
    raise TypeError(f"Unsupported MoE MLP module {type(mlp).__name__}: missing router/gate Linear")


_SWIGLU_ATTR_TRIPLETS: tuple[tuple[str, str, str], ...] = (
    ("gate_proj", "up_proj", "down_proj"),
    ("w1", "w2", "w3"),
)


def _extract_swiglu_descriptor(exp_mod: nn.Module) -> dict[str, Any] | None:
    """Extract uniform SwiGLU parameter mapping for HF/LLaMA/Qwen/Crowd-style expert modules."""
    for k_gate, k_up, k_down in _SWIGLU_ATTR_TRIPLETS:
        if hasattr(exp_mod, k_gate) and hasattr(exp_mod, k_up) and hasattr(exp_mod, k_down):
            return {
                "kind": "swiglu",
                "keys": (k_gate, k_up, k_down),
                "params": {
                    k_gate: getattr(exp_mod, k_gate).weight,
                    k_up: getattr(exp_mod, k_up).weight,
                    k_down: getattr(exp_mod, k_down).weight,
                },
                "module": exp_mod,
            }
    return None


def _resolve_expert_sequence(mlp: nn.Module, num_experts: int) -> list[dict[str, Any]]:
    """Build uniform per-expert descriptors across ``CrowdLoRA``, ``CrowdFineGrained``, and HF MoEs."""
    if isinstance(mlp, CrowdLoRAMoEMLP):
        out: list[dict[str, Any]] = []
        for e_idx in range(num_experts):
            exp = mlp.experts[str(e_idx)]
            out.append(
                {
                    "kind": "lora",
                    "params": {
                        "w1_A": exp.w1_A,
                        "w1_B": exp.w1_B,
                        "w2_A": exp.w2_A,
                        "w2_B": exp.w2_B,
                        "w3_A": exp.w3_A,
                        "w3_B": exp.w3_B,
                    },
                    "module": exp,
                    "base_mlp": mlp.base_mlp,
                }
            )
        return out

    if isinstance(mlp, CrowdFineGrainedMoEMLP):
        return [
            {
                "kind": "swiglu",
                "keys": ("w1", "w2", "w3"),
                "params": {
                    "w1": mlp.experts_w1[e_idx].weight,
                    "w2": mlp.experts_w2[e_idx].weight,
                    "w3": mlp.experts_w3[e_idx].weight,
                },
                "mlp": mlp,
                "expert_idx": e_idx,
            }
            for e_idx in range(num_experts)
        ]

    exps_raw = getattr(mlp, "experts", None)
    if isinstance(exps_raw, (nn.ModuleList, nn.ModuleDict)):
        out = []
        for e_idx in range(num_experts):
            exp_mod = exps_raw[str(e_idx)] if isinstance(exps_raw, nn.ModuleDict) else exps_raw[e_idx]
            desc = _extract_swiglu_descriptor(exp_mod)
            if desc is None:
                raise TypeError(f"Unsupported expert module {type(exp_mod).__name__}")
            out.append(desc)
        return out

    raise TypeError(f"Unsupported MoE MLP experts container on {type(mlp).__name__}")


def _eval_expert_descriptor(desc: Mapping[str, Any], x_flat: torch.Tensor) -> torch.Tensor:
    """Evaluate one expert slot forward pass in ``float32`` without modifying weights."""
    kind = desc["kind"]
    if kind == "lora":
        base_mlp = desc["base_mlp"]
        w_dt = base_mlp.w1.weight.dtype
        x_b = x_flat.to(dtype=w_dt)
        h1 = base_mlp.w1(x_b).to(torch.float32)
        h2 = base_mlp.w2(x_b).to(torch.float32)
        return desc["module"].forward_delta(x_flat.to(torch.float32), h1, h2, base_mlp.w3.weight).to(torch.float32)
    if "mlp" in desc:
        return desc["mlp"]._run_expert(int(desc["expert_idx"]), x_flat).to(torch.float32)
    mod = desc["module"]
    k_gate, k_up, k_down = desc["keys"]
    p = desc["params"]
    x_c = x_flat.to(dtype=p[k_gate].dtype)
    return getattr(mod, k_down)(F.silu(getattr(mod, k_gate)(x_c)) * getattr(mod, k_up)(x_c)).to(torch.float32)


def _normalize_experts_input(
    experts: Sequence[nn.Module] | nn.ModuleList | nn.ModuleDict | CrowdLoRAMoEMLP | CrowdFineGrainedMoEMLP | Mapping[Any, Any],
    num_experts: int,
    name: str,
) -> list[dict[str, Any]]:
    """Normalize ``experts`` into uniform descriptors for matching and weight transport."""
    if isinstance(experts, (CrowdLoRAMoEMLP, CrowdFineGrainedMoEMLP)):
        if experts.num_experts != num_experts:
            raise ValueError(f"{name}.num_experts={experts.num_experts} != router E={num_experts}")
        return _resolve_expert_sequence(experts, num_experts)

    if isinstance(experts, nn.Module) and hasattr(experts, "experts") and hasattr(experts, "router"):
        return _resolve_expert_sequence(experts, num_experts)

    if isinstance(experts, nn.ModuleDict):
        if len(experts) != num_experts:
            raise ValueError(f"{name} length {len(experts)} != router E={num_experts}")
        items = [experts[str(e)] for e in range(num_experts)]
    elif isinstance(experts, (nn.ModuleList, Sequence)) and not isinstance(experts, (str, bytes)):
        if len(experts) != num_experts:
            raise ValueError(f"{name} length {len(experts)} != router E={num_experts}")
        items = list(experts)
    else:
        raise TypeError(f"{name} must be an MoE MLP, ModuleList, ModuleDict, or Sequence[Module]")

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if isinstance(item, nn.Linear):
            out.append({"kind": "linear", "params": {"w3": item.weight}, "module": item})
        else:
            sw_desc = _extract_swiglu_descriptor(item) if isinstance(item, nn.Module) else None
            if sw_desc is not None:
                out.append(sw_desc)
            elif callable(item):
                out.append({"kind": "callable", "params": {}, "module": item})
            else:
                raise TypeError(f"Unsupported expert entry {name}[{idx}]: {type(item).__name__}")
    return out


def _eval_any_expert_descriptor(desc: Mapping[str, Any], x_flat: torch.Tensor) -> torch.Tensor:
    """Evaluate either a structured expert descriptor or a generic ``nn.Module``/linear expert."""
    if desc["kind"] in ("lora", "swiglu"):
        return _eval_expert_descriptor(desc, x_flat)
    mod = desc["module"]
    w = next(mod.parameters(), None) if isinstance(mod, nn.Module) else None
    x_in = x_flat.to(dtype=w.dtype) if w is not None else x_flat.to(torch.float32)
    return mod(x_in).to(torch.float32)


def _extract_aligned_weight_vector(
    desc: Mapping[str, Any],
    r_b2a: torch.Tensor | None,
) -> torch.Tensor:
    """Build a flattened parameter vector in ``Arm_A`` coordinates for optional weight cosine OT."""
    params: Mapping[str, torch.Tensor] = desc["params"]
    if not params:
        return torch.zeros(1, dtype=torch.float64)
    parts: list[torch.Tensor] = []
    r64 = r_b2a.to(torch.float64) if r_b2a is not None else None
    if desc["kind"] == "lora":
        for p_name in ("w1", "w2", "w3"):
            dw = params[f"{p_name}_B"].detach().to(torch.float64) @ params[f"{p_name}_A"].detach().to(torch.float64)
            if r64 is not None:
                dw = (dw @ r64) if p_name in ("w1", "w2") else (r64.T @ dw)
            parts.append(dw.reshape(-1))
    else:
        for k in sorted(params.keys()):
            w = params[k].detach().to(torch.float64)
            if r64 is not None:
                if k in ("w1", "w2", "gate_proj", "up_proj"):
                    w = w @ r64
                elif k in ("w3", "down_proj"):
                    w = r64.T @ w
            parts.append(w.reshape(-1))
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# 3. Cross-Arm Similarity, Procrustes Alignment & P_0 Fusion Primitives
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_cross_arm_expert_similarity_matrix(
    gate_alpha: torch.Tensor,
    gate_beta: torch.Tensor,
    acts_probe_alpha: torch.Tensor,
    experts_alpha: Any,
    experts_beta: Any,
    *,
    acts_probe_beta: torch.Tensor | None = None,
    r_b2a: torch.Tensor | None = None,
    router_sim_weight: float = DEFAULT_ROUTER_SIM_WEIGHT,
    func_sim_weight: float = DEFAULT_FUNC_SIM_WEIGHT,
    weight_sim_weight: float = DEFAULT_WEIGHT_SIM_WEIGHT,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute ``(E, E)`` cross-arm expert similarity combining router, functional, and weight alignment."""
    w_r = _validate_nonneg_finite(router_sim_weight, "router_sim_weight")
    w_f = _validate_nonneg_finite(func_sim_weight, "func_sim_weight")
    w_w = _validate_nonneg_finite(weight_sim_weight, "weight_sim_weight")
    total_w = w_r + w_f + w_w
    if total_w <= 0.0:
        raise ValueError("Sum of router_sim_weight, func_sim_weight, weight_sim_weight must be > 0.0")

    if not isinstance(gate_alpha, torch.Tensor) or not isinstance(gate_beta, torch.Tensor):
        raise TypeError("gate_alpha and gate_beta must be torch.Tensor instances")
    if gate_alpha.ndim != 2 or gate_beta.ndim != 2 or gate_alpha.shape != gate_beta.shape:
        raise ValueError(
            f"gate_alpha {tuple(gate_alpha.shape)} and gate_beta {tuple(gate_beta.shape)} must be matching 2D matrices"
        )
    num_e, hidden_dim = int(gate_alpha.shape[0]), int(gate_alpha.shape[1])
    if num_e < 1 or hidden_dim < 1:
        raise ValueError("Router gate matrices must be non-empty")

    x_a = _validate_activation_matrix(acts_probe_alpha, "acts_probe_alpha")
    if int(x_a.shape[1]) != hidden_dim:
        raise ValueError(f"acts_probe_alpha width {x_a.shape[1]} != gate hidden_dim {hidden_dim}")
    x_b = _validate_activation_matrix(acts_probe_beta, "acts_probe_beta") if acts_probe_beta is not None else x_a
    if x_b.shape != x_a.shape:
        raise ValueError(f"acts_probe_beta shape {tuple(x_b.shape)} != acts_probe_alpha shape {tuple(x_a.shape)}")

    r_rot = _validate_square_matrix(r_b2a, "r_b2a", expected_dim=hidden_dim, require_orthogonal=True) if r_b2a is not None else None
    exps_a = _normalize_experts_input(experts_alpha, num_e, "experts_alpha")
    exps_b = _normalize_experts_input(experts_beta, num_e, "experts_beta")

    x_a32 = x_a.to(torch.float32)
    x_b32 = x_b.to(torch.float32)
    g_a32 = gate_alpha.detach().to(device="cpu", dtype=torch.float32)
    g_b32 = gate_beta.detach().to(device="cpu", dtype=torch.float32)

    r_a = F.normalize(F.linear(x_a32, g_a32), dim=0, eps=1e-12).to(torch.float64)
    r_b = F.normalize(F.linear(x_b32, g_b32), dim=0, eps=1e-12).to(torch.float64)
    sim_router = (r_a.T @ r_b).contiguous()

    r_rot32 = r_rot.to(torch.float32) if r_rot is not None else None
    outs_a = torch.stack(
        [F.normalize(_eval_any_expert_descriptor(exps_a[e], x_a32).reshape(-1), dim=0, eps=1e-12) for e in range(num_e)],
        dim=0,
    ).to(torch.float64)
    b_list: list[torch.Tensor] = []
    for e in range(num_e):
        yb = _eval_any_expert_descriptor(exps_b[e], x_b32)
        if r_rot32 is not None and yb.shape[-1] == hidden_dim:
            yb = yb @ r_rot32
        b_list.append(F.normalize(yb.reshape(-1), dim=0, eps=1e-12))
    outs_b = torch.stack(b_list, dim=0).to(torch.float64)
    sim_func = (outs_a @ outs_b.T).contiguous()

    sim_total = (w_r * sim_router + w_f * sim_func) / total_w
    sim_weight: torch.Tensor | None = None
    if w_w > 0.0:
        vecs_a = [_extract_aligned_weight_vector(exps_a[e], None) for e in range(num_e)]
        vecs_b = [_extract_aligned_weight_vector(exps_b[e], r_rot) for e in range(num_e)]
        has_all = all(bool(exps_a[e]["params"]) and bool(exps_b[e]["params"]) for e in range(num_e))
        if has_all and len({tuple(v.shape) for v in vecs_a + vecs_b}) == 1:
            wa = torch.stack([F.normalize(v, dim=0, eps=1e-12) for v in vecs_a], dim=0)
            wb = torch.stack([F.normalize(v, dim=0, eps=1e-12) for v in vecs_b], dim=0)
            sim_weight = (wa @ wb.T).contiguous()
            sim_total = sim_total + (w_w / total_w) * sim_weight
        elif (w_r + w_f) > 0.0:
            sim_total = (w_r * sim_router + w_f * sim_func) / (w_r + w_f)

    return sim_total.contiguous(), {
        "sim_router": sim_router,
        "sim_func": sim_func,
        "sim_weight": sim_weight,
        "num_experts": num_e,
    }


@torch.no_grad()
def match_moe_experts_hungarian_procrustes(
    gate_alpha: torch.Tensor,
    gate_beta: torch.Tensor,
    acts_probe_alpha: torch.Tensor,
    experts_alpha: Any,
    experts_beta: Any,
    *,
    acts_probe_beta: torch.Tensor | None = None,
    align_hidden_space_procrustes: bool = True,
    proper_rotation: bool = False,
    min_cka_gate: float = DEFAULT_MIN_CKA_GATE,
    router_sim_weight: float = DEFAULT_ROUTER_SIM_WEIGHT,
    func_sim_weight: float = DEFAULT_FUNC_SIM_WEIGHT,
    weight_sim_weight: float = DEFAULT_WEIGHT_SIM_WEIGHT,
) -> dict[str, Any]:
    """Solve Optimal-Transport Hungarian assignment paired with Orthogonal Procrustes subspace alignment."""
    cka_gate = _validate_unit_interval(min_cka_gate, "min_cka_gate", allow_zero=True)
    use_proc = _validate_bool(align_hidden_space_procrustes, "align_hidden_space_procrustes")
    proper = _validate_bool(proper_rotation, "proper_rotation")

    hidden_dim = int(gate_alpha.shape[1]) if isinstance(gate_alpha, torch.Tensor) and gate_alpha.ndim == 2 else 0
    if acts_probe_beta is not None:
        cka_val, r_b2a, proc_metrics = compute_linear_cka_and_orthogonal_procrustes(
            acts_probe_beta,
            acts_probe_alpha,
            proper_rotation=proper,
        )
        r_effective = r_b2a if use_proc else torch.eye(hidden_dim, dtype=torch.float64)
        cka_ok = cka_val >= cka_gate
    else:
        cka_val = 1.0
        r_effective = torch.eye(hidden_dim, dtype=torch.float64)
        proc_metrics = {
            "cka_alignment": 1.0,
            "procrustes_residual_before": 0.0,
            "procrustes_residual_after": 0.0,
            "orthogonality_error": 0.0,
            "determinant": 1.0,
        }
        cka_ok = True

    sim_mat, sim_diag = compute_cross_arm_expert_similarity_matrix(
        gate_alpha,
        gate_beta,
        acts_probe_alpha,
        experts_alpha,
        experts_beta,
        acts_probe_beta=acts_probe_beta,
        r_b2a=r_effective if use_proc and acts_probe_beta is not None else None,
        router_sim_weight=router_sim_weight,
        func_sim_weight=func_sim_weight,
        weight_sim_weight=weight_sim_weight,
    )
    perm, inv_perm, total_cost, ot_diag = solve_exact_hungarian_assignment(-sim_mat)
    unaligned_sim = float(torch.diagonal(sim_mat).mean().item())
    matched_sim = -float(ot_diag["mean_cost"])

    return {
        "permutation": perm,
        "inverse_permutation": inv_perm,
        "r_b2a": r_effective,
        "cka_alignment": float(cka_val),
        "cka_gate_passed": cka_ok,
        "min_cka_gate": cka_gate,
        "procrustes_metrics": proc_metrics,
        "similarity_matrix": sim_mat,
        "sim_router": sim_diag["sim_router"],
        "sim_func": sim_diag["sim_func"],
        "mean_matched_similarity": matched_sim,
        "unaligned_identity_similarity": unaligned_sim,
        "similarity_gain": matched_sim - unaligned_sim,
        "total_transport_cost": total_cost,
        "is_bijection": ot_diag["is_bijection"],
        "birkhoff_marginal_error": ot_diag["birkhoff_marginal_error"],
    }


@torch.no_grad()
def align_lora_rank_basis_procrustes(
    b_alpha: torch.Tensor,
    a_alpha: torch.Tensor,
    b_beta: torch.Tensor,
    a_beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Align ``(B_beta, A_beta)`` LoRA rank basis onto ``(B_alpha, A_alpha)`` via ``S in O(r)``.

    Solves ``min_{S^T S = I_r} ||B_beta S - B_alpha||_F^2 + ||S^T A_beta - A_alpha||_F^2``
    so ``(B_beta S)(S^T A_beta) == B_beta A_beta`` identically while eliminating rank interference.
    """
    ba64 = b_alpha.detach().to(device="cpu", dtype=torch.float64)
    aa64 = a_alpha.detach().to(device="cpu", dtype=torch.float64)
    bb64 = b_beta.detach().to(device="cpu", dtype=torch.float64)
    ab64 = a_beta.detach().to(device="cpu", dtype=torch.float64)
    if ba64.shape != bb64.shape or aa64.shape != ab64.shape or ba64.ndim != 2 or aa64.ndim != 2:
        raise ValueError("LoRA factor shapes must match 2D (d_out, r) and (r, d_in)")
    rank = int(aa64.shape[0])
    if ba64.shape[1] != rank:
        raise ValueError(f"Rank mismatch: B has {ba64.shape[1]} vs A has {rank}")

    cross = (bb64.T @ ba64) + (ab64 @ aa64.T)
    if float(cross.abs().max().item()) <= 1e-18:
        s_mat = torch.eye(rank, dtype=torch.float64)
    else:
        u_m, _, vh_m = torch.linalg.svd(cross, full_matrices=False)
        s_mat = (u_m @ vh_m).contiguous()

    b_rot = (bb64 @ s_mat).contiguous()
    a_rot = (s_mat.T @ ab64).contiguous()
    prod_err = float(torch.linalg.matrix_norm((b_rot @ a_rot) - (bb64 @ ab64), ord="fro").item())
    factor_dist_before = float(
        torch.linalg.matrix_norm(bb64 - ba64, ord="fro").item()
        + torch.linalg.matrix_norm(ab64 - aa64, ord="fro").item()
    )
    factor_dist_after = float(
        torch.linalg.matrix_norm(b_rot - ba64, ord="fro").item()
        + torch.linalg.matrix_norm(a_rot - aa64, ord="fro").item()
    )
    return (
        b_rot.to(device=b_beta.device, dtype=b_beta.dtype),
        a_rot.to(device=a_beta.device, dtype=a_beta.dtype),
        {
            "product_invariance_error": prod_err,
            "factor_distance_before": factor_dist_before,
            "factor_distance_after": factor_dist_after,
        },
    )


def _synergy_merge_tensors(
    t_a: torch.Tensor,
    t_b_aligned: torch.Tensor,
    t_base: torch.Tensor | None,
    p_null: torch.Tensor | None,
    *,
    side: str,
    cfg: OptimalTransportMoEFuserConfig,
    is_output_proj: bool,
) -> tuple[torch.Tensor, float, float]:
    """Fuse aligned tensors via convex OT blend or ``P_0`` orthogonal residual synergy."""
    a64 = t_a.detach().to(device="cpu", dtype=torch.float64)
    b64 = t_b_aligned.detach().to(device="cpu", dtype=torch.float64)
    alpha = cfg.fusion_alpha
    apply_p0 = cfg.alignment_mode == "hungarian_procrustes_p0_synergy" and cfg.project_into_prose_nullspace and p_null is not None

    if t_base is not None and cfg.alignment_mode == "hungarian_procrustes_p0_synergy":
        w0 = t_base.detach().to(device="cpu", dtype=torch.float64)
        da = a64 - w0
        db = b64 - w0
        leak_before = 0.0
        leak_after = 0.0
        if apply_p0:
            da, lb_a, la_a, _ = _project_dense_tensor_in_nullspace(a64, p_null, side, cfg.nullspace_leak_tol, t_base=w0)
            db, lb_b, la_b, _ = _project_dense_tensor_in_nullspace(b64, p_null, side, cfg.nullspace_leak_tol, t_base=w0)
            da = da - w0
            db = db - w0
            leak_before = max(lb_a, lb_b)
            leak_after = max(la_a, la_b)
        if is_output_proj and cfg.orthogonal_synergy_beta > 0.0 and float(da.norm().item()) > 1e-12:
            _, _, vha = torch.linalg.svd(da, full_matrices=False)
            k_r = min(cfg.orthogonal_synergy_rank, int(vha.shape[0]))
            va = vha[:k_r, :].T
            db_orth = db - cfg.orthogonal_synergy_beta * ((db @ va) @ va.T)
            if apply_p0 and side == "input":
                db_orth = db_orth @ p_null
            elif apply_p0 and side == "output":
                db_orth = p_null @ db_orth
            fused = w0 + cfg.orthogonal_synergy_scale * (da + db_orth)
        else:
            fused = w0 + (1.0 - alpha) * da + alpha * db
        return fused.to(device=t_a.device, dtype=t_a.dtype).contiguous(), leak_before, leak_after

    if apply_p0:
        b_proj, leak_before, leak_after, _ = _project_dense_tensor_in_nullspace(
            b64, p_null, side, cfg.nullspace_leak_tol, t_base=a64
        )
        fused = (1.0 - alpha) * a64 + alpha * b_proj.to(torch.float64)
        return fused.to(device=t_a.device, dtype=t_a.dtype).contiguous(), leak_before, leak_after

    fused = (1.0 - alpha) * a64 + alpha * b64
    return fused.to(device=t_a.device, dtype=t_a.dtype).contiguous(), 0.0, 0.0


def _fuse_lora_factor_pair(
    b_a: torch.Tensor,
    a_a: torch.Tensor,
    b_b: torch.Tensor,
    a_b: torch.Tensor,
    *,
    cfg: OptimalTransportMoEFuserConfig,
    is_output_proj: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse aligned LoRA ``(B_a, A_a)`` and ``(B_b, A_b)`` without random-init cross-term loss."""
    ba64 = b_a.detach().to(device="cpu", dtype=torch.float64)
    aa64 = a_a.detach().to(device="cpu", dtype=torch.float64)
    bb64 = b_b.detach().to(device="cpu", dtype=torch.float64)
    ab64 = a_b.detach().to(device="cpu", dtype=torch.float64)
    rank = int(aa64.shape[0])

    norm_ba = float(torch.linalg.matrix_norm(ba64, ord="fro").item())
    norm_bb = float(torch.linalg.matrix_norm(bb64, ord="fro").item())
    if cfg.alignment_mode == "hungarian_procrustes_p0_synergy":
        w_a = cfg.orthogonal_synergy_scale
        w_b = cfg.orthogonal_synergy_scale
    else:
        w_a = 1.0 - cfg.fusion_alpha
        w_b = cfg.fusion_alpha

    if norm_ba <= 1e-12 and norm_bb <= 1e-12:
        return torch.zeros_like(ba64), aa64
    if norm_ba <= 1e-12:
        return w_b * bb64, ab64
    if norm_bb <= 1e-12:
        return w_a * ba64, aa64

    if is_output_proj and cfg.alignment_mode == "hungarian_procrustes_p0_synergy" and cfg.orthogonal_synergy_beta > 0.0:
        _, _, vha = decompose_lora_qr_svd(ba64, aa64)
        k_r = min(cfg.orthogonal_synergy_rank, int(vha.shape[1]))
        if k_r > 0:
            va = vha[:, :k_r]
            ab64 = ab64 - cfg.orthogonal_synergy_beta * ((ab64 @ va) @ va.T)

    b_cat = torch.cat([w_a * ba64, w_b * bb64], dim=1)
    a_cat = torch.cat([aa64, ab64], dim=0)
    u_r, s_r, v_r = decompose_lora_qr_svd(b_cat, a_cat)
    k_eff = min(rank, int(s_r.numel()))
    sq_s = s_r[:k_eff].clamp_min(0.0).sqrt()
    b_out = torch.zeros_like(ba64)
    a_out = torch.zeros_like(aa64)
    b_out[:, :k_eff] = u_r[:, :k_eff] * sq_s.unsqueeze(0)
    a_out[:k_eff, :] = sq_s.unsqueeze(1) * v_r[:, :k_eff].T
    return b_out, a_out


@torch.no_grad()
def fuse_aligned_moe_layer_weights(
    mlp_alpha: nn.Module,
    mlp_beta: nn.Module,
    alignment: Mapping[str, Any],
    *,
    mlp_base: nn.Module | None = None,
    p_null_x: torch.Tensor | None = None,
    p_null_act: torch.Tensor | None = None,
    config: OptimalTransportMoEFuserConfig | None = None,
) -> dict[str, Any]:
    """Permute, Procrustes-rotate, ``P_0``-project, and fuse one matched MoE layer into ``mlp_alpha``."""
    cfg = config or OptimalTransportMoEFuserConfig()
    perm: list[int] = list(alignment["permutation"])
    r_b2a = alignment["r_b2a"].detach().to(device="cpu", dtype=torch.float64)
    use_rot = cfg.alignment_mode != "hungarian" and cfg.align_hidden_space_procrustes

    router_a = _resolve_router_linear(mlp_alpha)
    router_b = _resolve_router_linear(mlp_beta)
    num_e = int(router_a.weight.shape[0])
    if len(perm) != num_e or int(router_b.weight.shape[0]) != num_e:
        raise ValueError("Router expert count mismatch during MoE layer fusion")

    exps_a = _resolve_expert_sequence(mlp_alpha, num_e)
    exps_b = _resolve_expert_sequence(mlp_beta, num_e)
    exps_base = _resolve_expert_sequence(mlp_base, num_e) if mlp_base is not None else None
    router_base_w = _resolve_router_linear(mlp_base).weight if mlp_base is not None else None

    ga64 = router_a.weight.detach().to(device="cpu", dtype=torch.float64)
    gb64 = router_b.weight.detach().to(device="cpu", dtype=torch.float64)[perm]
    if use_rot:
        gb64 = gb64 @ r_b2a

    r_alpha = cfg.router_fusion_alpha
    if cfg.project_router_into_nullspace and p_null_x is not None:
        r_ref = router_base_w.detach().to(device="cpu", dtype=torch.float64) if router_base_w is not None else ga64
        gb_p, _, _, _ = _project_dense_tensor_in_nullspace(gb64, p_null_x, "input", cfg.nullspace_leak_tol, t_base=r_ref)
        ga_p, _, _, _ = _project_dense_tensor_in_nullspace(ga64, p_null_x, "input", cfg.nullspace_leak_tol, t_base=r_ref)
        fused_router = (1.0 - r_alpha) * ga_p.to(torch.float64) + r_alpha * gb_p.to(torch.float64)
    else:
        fused_router = (1.0 - r_alpha) * ga64 + r_alpha * gb64
    router_a.weight.copy_(fused_router.to(device=router_a.weight.device, dtype=router_a.weight.dtype))

    max_leak_before = 0.0
    max_leak_after = 0.0
    rank_alignments = 0

    for e_i in range(num_e):
        b_idx = perm[e_i]
        pa = exps_a[e_i]["params"]
        pb = exps_b[b_idx]["params"]
        pbase = exps_base[e_i]["params"] if exps_base is not None else None
        kind = exps_a[e_i]["kind"]

        if kind == "lora":
            for proj in ("w1", "w2", "w3"):
                ak, bk = f"{proj}_A", f"{proj}_B"
                a_b_rot = pb[ak].detach().to(device="cpu", dtype=torch.float64)
                b_b_rot = pb[bk].detach().to(device="cpu", dtype=torch.float64)
                if use_rot:
                    if proj in ("w1", "w2"):
                        a_b_rot = a_b_rot @ r_b2a
                    else:
                        b_b_rot = r_b2a.T @ b_b_rot
                if cfg.align_lora_rank_procrustes and cfg.alignment_mode != "hungarian":
                    b_b_rot, a_b_rot, _ = align_lora_rank_basis_procrustes(pa[bk], pa[ak], b_b_rot, a_b_rot)
                    rank_alignments += 1

                p_proj = p_null_x if proj in ("w1", "w2") else (p_null_act if p_null_act is not None else p_null_x)
                side_proj = "input" if (proj in ("w1", "w2") or p_null_act is not None) else "output"
                if cfg.alignment_mode == "hungarian_procrustes_p0_synergy" and cfg.project_into_prose_nullspace and p_proj is not None:
                    ref_b = pbase[bk] if pbase is not None else torch.zeros_like(b_b_rot)
                    ref_a = pbase[ak] if pbase is not None else a_b_rot
                    b_b_p, a_b_p, lb, la, _ = _project_lora_pair_in_nullspace(
                        ref_b, ref_a, b_b_rot, a_b_rot, p_proj, side_proj, cfg.nullspace_leak_tol
                    )
                    max_leak_before = max(max_leak_before, lb)
                    max_leak_after = max(max_leak_after, la)
                    b_b_rot, a_b_rot = b_b_p.to(torch.float64), a_b_p.to(torch.float64)

                ba64 = pa[bk].detach().to(device="cpu", dtype=torch.float64)
                aa64 = pa[ak].detach().to(device="cpu", dtype=torch.float64)
                if cfg.alignment_mode == "hungarian_procrustes_p0_synergy" and cfg.project_into_prose_nullspace and p_proj is not None:
                    ref_ba = pbase[bk] if pbase is not None else torch.zeros_like(ba64)
                    ref_aa = pbase[ak] if pbase is not None else aa64
                    ba_p, aa_p, lba, laa, _ = _project_lora_pair_in_nullspace(
                        ref_ba, ref_aa, ba64, aa64, p_proj, side_proj, cfg.nullspace_leak_tol
                    )
                    max_leak_before = max(max_leak_before, lba)
                    max_leak_after = max(max_leak_after, laa)
                    ba64, aa64 = ba_p.to(torch.float64), aa_p.to(torch.float64)

                fused_b, fused_a = _fuse_lora_factor_pair(
                    ba64,
                    aa64,
                    b_b_rot,
                    a_b_rot,
                    cfg=cfg,
                    is_output_proj=(proj == "w3"),
                )
                pa[bk].copy_(fused_b.to(device=pa[bk].device, dtype=pa[bk].dtype))
                pa[ak].copy_(fused_a.to(device=pa[ak].device, dtype=pa[ak].dtype))
        else:
            for k_name in sorted(pa.keys()):
                is_in = k_name in ("w1", "w2", "gate_proj", "up_proj")
                wb_rot = pb[k_name].detach().to(device="cpu", dtype=torch.float64)
                if use_rot:
                    wb_rot = (wb_rot @ r_b2a) if is_in else (r_b2a.T @ wb_rot)
                p_mat = p_null_x if is_in else (p_null_act if p_null_act is not None else p_null_x)
                side = "input" if (is_in or p_null_act is not None) else "output"
                merged, lb, la = _synergy_merge_tensors(
                    pa[k_name],
                    wb_rot,
                    pbase[k_name] if pbase is not None else None,
                    p_mat,
                    side=side,
                    cfg=cfg,
                    is_output_proj=not is_in,
                )
                max_leak_before = max(max_leak_before, lb)
                max_leak_after = max(max_leak_after, la)
                pa[k_name].copy_(merged)

    return {
        "permutation": perm,
        "cka_alignment": float(alignment["cka_alignment"]),
        "procrustes_orth_err": float(alignment["procrustes_metrics"]["orthogonality_error"]),
        "mean_matched_similarity": float(alignment["mean_matched_similarity"]),
        "unaligned_identity_similarity": float(alignment["unaligned_identity_similarity"]),
        "similarity_gain": float(alignment["similarity_gain"]),
        "max_p0_leak_before": max_leak_before,
        "max_p0_leak_after": max_leak_after,
        "lora_rank_alignments": rank_alignments,
    }


# ---------------------------------------------------------------------------
# 4. Model Discovery, Calibration & ANSE-Gated End-to-End Fuser
# ---------------------------------------------------------------------------


def _discover_moe_mlp_layers(model: nn.Module) -> dict[int, nn.Module]:
    """Discover ordered MoE MLP modules on ``CrowdV1WithEngram`` or HF-style transformers."""
    if hasattr(model, "router") or hasattr(model, "gate"):
        return {0: model}

    blocks: Sequence[nn.Module] | None = None
    for chain in (("blocks",), ("backbone", "model", "layers"), ("model", "layers"), ("layers",)):
        cur: Any = model
        for attr in chain:
            cur = getattr(cur, attr, None)
            if cur is None:
                break
        if isinstance(cur, (nn.ModuleList, Sequence)):
            blocks = cur
            break

    if blocks is None:
        raise TypeError(f"Unsupported MoE model layout {type(model).__name__}")

    layers: dict[int, nn.Module] = {}
    for idx, blk in enumerate(blocks):
        mlp = getattr(blk, "mlp", None)
        if mlp is not None and (hasattr(mlp, "router") or hasattr(mlp, "gate")):
            layers[idx] = mlp
    if not layers:
        raise ValueError(f"No MoE MLP layers found on {type(model).__name__}")
    return layers


@torch.no_grad()
def _capture_mlp_inputs_and_pre_ln(
    model: nn.Module,
    moe_layers: Mapping[int, nn.Module],
    ids_2d: torch.Tensor,
    *,
    score_from: int,
    batch_size: int,
) -> tuple[dict[int, torch.Tensor], torch.Tensor | None]:
    """Capture scored MoE MLP input states across layers via forward pre-hooks."""
    start_pos = max(0, int(score_from) - 1)
    captured: dict[int, list[torch.Tensor]] = {l_idx: [] for l_idx in moe_layers}
    hooks: list[Any] = []
    for l_idx, mlp in moe_layers.items():
        def _hook(_mod: nn.Module, args: tuple[Any, ...], li: int = l_idx) -> None:
            x_in = args[0]
            if x_in.ndim == 3:
                x_s = x_in[:, start_pos:-1, :].reshape(-1, int(x_in.shape[-1]))
            else:
                x_s = x_in.reshape(-1, int(x_in.shape[-1]))
            captured[li].append(x_s.detach().to(device="cpu", dtype=torch.float32))

        hooks.append(mlp.register_forward_pre_hook(_hook))

    try:
        for b0 in range(0, int(ids_2d.shape[0]), batch_size):
            _ = model(input_ids=ids_2d[b0 : b0 + batch_size])
    finally:
        for h in hooks:
            h.remove()

    merged = {li: torch.cat(parts, dim=0).contiguous() for li, parts in captured.items() if parts}
    pre_ln: torch.Tensor | None = None
    if hasattr(model, "wte") and hasattr(model, "blocks") and hasattr(model, "ln"):
        pre_ln = forward_model_pre_ln(model, ids_2d, batch_size=batch_size)
    return merged, pre_ln


def _resolve_layer_p0(
    p_null_by_layer: Mapping[int, Any] | torch.Tensor | None,
    l_idx: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve optional ``(P_0_x, P_0_act)`` projectors for layer ``l_idx``."""
    if p_null_by_layer is None:
        return None, None
    if isinstance(p_null_by_layer, torch.Tensor):
        return _validate_square_matrix(p_null_by_layer, "p_null_by_layer"), None
    if l_idx not in p_null_by_layer:
        return None, None
    entry = p_null_by_layer[l_idx]
    if isinstance(entry, torch.Tensor):
        return _validate_square_matrix(entry, f"p_null_by_layer[{l_idx}]"), None
    if isinstance(entry, Mapping):
        px = _validate_square_matrix(entry["x"], f"p_null_by_layer[{l_idx}]['x']") if entry.get("x") is not None else None
        pact = _validate_square_matrix(entry["act"], f"p_null_by_layer[{l_idx}]['act']") if entry.get("act") is not None else None
        return px, pact
    raise TypeError(f"Unsupported p_null_by_layer[{l_idx}] type {type(entry).__name__}")


def _score_domain_nll_vs_ref(
    ref_nll: Mapping[str, float],
    cand_nll: Mapping[str, float],
    *,
    cfg: OptimalTransportMoEFuserConfig,
) -> dict[str, Any]:
    """Compute general/worst drift ratios and ANSE Energy relative to ``ref_nll``."""
    keys = sorted(ref_nll.keys())
    if set(cand_nll.keys()) != set(keys):
        raise ValueError("Candidate and reference domain NLL mappings must share identical keys")

    def _is_gen(dom: str) -> bool:
        low = dom.lower()
        return any(low.startswith(p.lower()) for p in cfg.general_prefixes)

    gen_keys = [k for k in keys if _is_gen(k)]
    if not gen_keys:
        raise ValueError(f"Missing required general-domain key matching prefixes {cfg.general_prefixes!r}")

    per_domain: dict[str, dict[str, float]] = {}
    diffs: list[float] = []
    worst_id = ""
    worst_ratio = 0.0
    for k in keys:
        b = float(ref_nll[k])
        c = float(cand_nll[k])
        diff = c - b
        ratio = c / max(b, 1e-8)
        diffs.append(diff)
        per_domain[k] = {"ref_nll": b, "cand_nll": c, "diff": diff, "ratio": ratio, "pct_vs_ref": 100.0 * (ratio - 1.0)}
        if ratio >= worst_ratio:
            worst_ratio = ratio
            worst_id = k

    gen_ratio = max(per_domain[k]["ratio"] for k in gen_keys)
    gen_ok = bool(gen_ratio <= cfg.max_general_drift_ratio + 1e-12)
    worst_ok = bool(worst_ratio <= cfg.max_worst_domain_drift_ratio + 1e-12)
    guard_ok = bool(gen_ok and (worst_ok if cfg.enforce_all_domain_guard else True))
    mean_diff = float(sum(diffs) / float(len(diffs)))
    nll_mean = float(sum(cand_nll[k] for k in keys) / float(len(keys)))
    base_nll_mean = float(sum(ref_nll[k] for k in keys) / float(len(keys)))
    soft_pen = cfg.soft_margin_penalty if (guard_ok and gen_ratio > cfg.anse_safe_general_ratio) else 0.0
    anse_energy = float(cfg.maximum_pain_energy if not guard_ok else (mean_diff + soft_pen))
    return {
        "nll_mean": nll_mean,
        "base_nll_mean": base_nll_mean,
        "mean_diff_vs_ref": mean_diff,
        "general_ratio": gen_ratio,
        "general_pct_vs_ref": 100.0 * (gen_ratio - 1.0),
        "worst_domain_id": worst_id,
        "worst_domain_ratio": worst_ratio,
        "worst_domain_pct_vs_ref": 100.0 * (worst_ratio - 1.0),
        "guard_ok": guard_ok,
        "anse_energy": anse_energy,
        "per_domain": per_domain,
    }


def _resolve_fuser_config(
    config: OptimalTransportMoEFuserConfig | None,
    overrides: Mapping[str, Any],
) -> OptimalTransportMoEFuserConfig:
    if config is not None and not isinstance(config, OptimalTransportMoEFuserConfig):
        raise TypeError(f"config must be OptimalTransportMoEFuserConfig or None, got {type(config).__name__}")
    raw = asdict(config) if config is not None else asdict(OptimalTransportMoEFuserConfig())
    for k, v in overrides.items():
        if v is not None:
            raw[k] = v
    return OptimalTransportMoEFuserConfig(**raw)


@torch.no_grad()
def execute_verified_ot_moe_arm_fusion(
    arm_a_model: nn.Module,
    arm_b_model: nn.Module,
    *,
    calibration_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    base_model: nn.Module | None = None,
    acts_probe_a: Mapping[int, torch.Tensor] | torch.Tensor | None = None,
    acts_probe_b: Mapping[int, torch.Tensor] | torch.Tensor | None = None,
    p_null_by_layer: Mapping[int, Any] | torch.Tensor | None = None,
    eval_fn: Callable[[nn.Module], Mapping[str, float] | tuple[float, Mapping[str, float]]] | None = None,
    baseline_domain_nll: Mapping[str, float] | None = None,
    layer_indices: Sequence[int] | None = None,
    config: OptimalTransportMoEFuserConfig | None = None,
    alignment_mode: str | None = None,
    calibrate_layernorm: bool | None = None,
    rollback_target: str | None = None,
    inplace: bool | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Execute OT Hungarian + Orthogonal Procrustes cross-arm MoE fusion under ANSE gating."""
    if not isinstance(arm_a_model, nn.Module) or not isinstance(arm_b_model, nn.Module):
        raise TypeError("arm_a_model and arm_b_model must be nn.Module instances")
    if base_model is not None and not isinstance(base_model, nn.Module):
        raise TypeError("base_model must be an nn.Module or None")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError("eval_fn must be callable or None")

    cfg = _resolve_fuser_config(
        config,
        {
            "alignment_mode": alignment_mode,
            "calibrate_layernorm": calibrate_layernorm,
            "rollback_target": rollback_target,
            "inplace": inplace,
        },
    )

    work_model = arm_a_model if cfg.inplace else copy.deepcopy(arm_a_model)
    layers_a = _discover_moe_mlp_layers(work_model)
    layers_b = _discover_moe_mlp_layers(arm_b_model)
    layers_base = _discover_moe_mlp_layers(base_model) if base_model is not None else {}
    if sorted(layers_a.keys()) != sorted(layers_b.keys()):
        raise ValueError("arm_a_model and arm_b_model must have matching MoE layer indices")

    target_layers = (
        sorted({_validate_nonneg_int(idx, "layer_index") for idx in layer_indices})
        if layer_indices is not None
        else sorted(layers_a.keys())
    )
    missing = [idx for idx in target_layers if idx not in layers_a]
    if missing:
        raise ValueError(f"Requested layer_indices not present in model: {missing}")

    ids_cal: torch.Tensor | None = None
    eids_cal: list[str] = []
    ids_eval: torch.Tensor | None = None
    eids_eval: list[str] = []
    if calibration_chunks is not None:
        ids_cal, eids_cal = _validate_chunks(
            calibration_chunks,
            "calibration_chunks",
            general_prefix=cfg.general_prefixes[0],
            score_from=cfg.score_from,
            require_both_groups=True,
        )
        if eval_chunks is not None:
            ids_eval, eids_eval = _validate_chunks(
                eval_chunks,
                "eval_chunks",
                general_prefix=cfg.general_prefixes[0],
                score_from=cfg.score_from,
                require_both_groups=False,
            )
        else:
            ids_eval, eids_eval = ids_cal, eids_cal
    elif acts_probe_a is None or eval_fn is None:
        raise ValueError("Provide calibration_chunks or both acts_probe_a and eval_fn")

    def _run_eval(target_m: nn.Module) -> dict[str, float]:
        if eval_fn is not None:
            return _parse_eval_domain_nll(eval_fn(target_m))
        assert ids_eval is not None
        return _eval_crowd_model_domain_nlls(
            target_m, ids_eval, eids_eval, score_from=cfg.score_from, batch_size=cfg.batch_size
        )

    was_training = work_model.training
    was_b_training = arm_b_model.training
    work_model.eval()
    arm_b_model.eval()
    if base_model is not None:
        base_model.eval()

    try:
        arm_a_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in work_model.state_dict().items()}
        arm_a_sha = compute_state_sha256(arm_a_snapshot)
        arm_b_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in arm_b_model.state_dict().items()}
        arm_b_sha = compute_state_sha256(arm_b_snapshot)
        base_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in base_model.state_dict().items()} if base_model is not None else None
        base_sha = compute_state_sha256(base_snapshot) if base_snapshot is not None else None

        nll_a = _run_eval(work_model)
        nll_b = _run_eval(arm_b_model)
        if baseline_domain_nll is not None:
            ref_nll = _parse_eval_domain_nll(baseline_domain_nll)
        elif base_model is not None:
            ref_nll = _run_eval(base_model)
        else:
            ref_nll = nll_a

        eval_a = _score_domain_nll_vs_ref(ref_nll, nll_a, cfg=cfg)
        eval_b = _score_domain_nll_vs_ref(ref_nll, nll_b, cfg=cfg)

        if ids_cal is not None:
            probe_a, _ = _capture_mlp_inputs_and_pre_ln(
                work_model, layers_a, ids_cal, score_from=cfg.score_from, batch_size=cfg.batch_size
            )
            probe_b, _ = _capture_mlp_inputs_and_pre_ln(
                arm_b_model, layers_b, ids_cal, score_from=cfg.score_from, batch_size=cfg.batch_size
            )
        else:
            assert acts_probe_a is not None
            if isinstance(acts_probe_a, torch.Tensor):
                pa_t = _validate_activation_matrix(acts_probe_a, "acts_probe_a").to(torch.float32)
                probe_a = {li: pa_t for li in target_layers}
            else:
                missing_pa = [li for li in target_layers if li not in acts_probe_a]
                if missing_pa:
                    raise ValueError(f"acts_probe_a missing required layers: {missing_pa}")
                probe_a = {
                    li: _validate_activation_matrix(acts_probe_a[li], f"acts_probe_a[{li}]").to(torch.float32)
                    for li in target_layers
                }
            if acts_probe_b is None:
                probe_b = dict(probe_a)
            elif isinstance(acts_probe_b, torch.Tensor):
                pb_t = _validate_activation_matrix(acts_probe_b, "acts_probe_b").to(torch.float32)
                probe_b = {li: pb_t for li in target_layers}
            else:
                missing_pb = [li for li in target_layers if li not in acts_probe_b]
                if missing_pb:
                    raise ValueError(f"acts_probe_b missing required layers: {missing_pb}")
                probe_b = {
                    li: _validate_activation_matrix(acts_probe_b[li], f"acts_probe_b[{li}]").to(torch.float32)
                    for li in target_layers
                }

        resolved_p0: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] = {}
        if p_null_by_layer is not None:
            for li in target_layers:
                resolved_p0[li] = _resolve_layer_p0(p_null_by_layer, li)
        elif cfg.alignment_mode == "hungarian_procrustes_p0_synergy" and cfg.project_into_prose_nullspace and ids_cal is not None:
            gen_flags = torch.tensor(
                [any(e.lower().startswith(p.lower()) for p in cfg.general_prefixes) for e in eids_cal],
                dtype=torch.bool,
            )
            n_ch = int(ids_cal.shape[0])
            p0_src_model = base_model if base_model is not None else work_model
            p0_src_layers = layers_base if base_model is not None else layers_a
            p0_acts, _ = _capture_mlp_inputs_and_pre_ln(
                p0_src_model, p0_src_layers, ids_cal, score_from=cfg.score_from, batch_size=cfg.batch_size
            )
            for li in target_layers:
                rows_p = p0_acts[li].reshape(n_ch, -1, int(p0_acts[li].shape[-1]))[gen_flags].reshape(-1, int(p0_acts[li].shape[-1]))
                px, _ = compute_prose_nullspace_projector_scaled(
                    rows_p,
                    sv_rel_tol=cfg.nullspace_sv_rel_tol,
                    variance_threshold=cfg.nullspace_variance_threshold,
                    max_active_modes=cfg.nullspace_max_modes,
                )
                resolved_p0[li] = (px, None)
        else:
            resolved_p0 = {li: (None, None) for li in target_layers}

        layer_reports: dict[str, Any] = {}
        cka_failed = False
        min_cka_seen = 1.0
        for li in target_layers:
            mlp_a = layers_a[li]
            mlp_b = layers_b[li]
            ga_w = _resolve_router_linear(mlp_a).weight
            gb_w = _resolve_router_linear(mlp_b).weight
            align_res = match_moe_experts_hungarian_procrustes(
                ga_w,
                gb_w,
                probe_a[li],
                mlp_a,
                mlp_b,
                acts_probe_beta=probe_b[li],
                align_hidden_space_procrustes=cfg.align_hidden_space_procrustes,
                proper_rotation=cfg.proper_rotation,
                min_cka_gate=cfg.min_cka_gate,
                router_sim_weight=cfg.router_sim_weight,
                func_sim_weight=cfg.func_sim_weight,
                weight_sim_weight=cfg.weight_sim_weight,
            )
            min_cka_seen = min(min_cka_seen, float(align_res["cka_alignment"]))
            if not bool(align_res["cka_gate_passed"]):
                cka_failed = True
                layer_reports[str(li)] = {
                    "cka_alignment": float(align_res["cka_alignment"]),
                    "cka_gate_passed": False,
                    "permutation": list(align_res["permutation"]),
                }
                break
            px, pact = resolved_p0.get(li, (None, None))
            fused_meta = fuse_aligned_moe_layer_weights(
                mlp_a,
                mlp_b,
                align_res,
                mlp_base=layers_base.get(li),
                p_null_x=px,
                p_null_act=pact,
                config=cfg,
            )
            layer_reports[str(li)] = {**fused_meta, "cka_gate_passed": True}

        ln_meta: dict[str, Any] | None = None
        if not cka_failed and cfg.calibrate_layernorm and ids_cal is not None and hasattr(work_model, "ln") and hasattr(work_model, "head"):
            h_pre_cal = forward_model_pre_ln(work_model, ids_cal, batch_size=cfg.batch_size)
            start_pos = max(0, cfg.score_from - 1)
            h_scored = h_pre_cal[:, start_pos:-1, :]
            t_scored = ids_cal[:, start_pos + 1 :]
            gen_mask = [any(e.lower().startswith(p.lower()) for p in cfg.general_prefixes) for e in eids_cal]
            anchor_ln_src = base_model.ln if base_model is not None else arm_a_model.ln
            w_cand = work_model.ln.weight.detach().clone()
            b_cand = getattr(work_model.ln, "bias", None)
            b_cand_c = b_cand.detach().clone() if b_cand is not None else None
            b_anc = getattr(anchor_ln_src, "bias", None)
            grid_res = fast_affine_layernorm_temperature_grid(
                h_scored,
                anchor_ln_src.weight.detach(),
                b_anc.detach() if b_anc is not None else None,
                w_cand,
                b_cand_c,
                work_model.head.weight.detach(),
                t_scored,
                gen_mask,
                eps=float(getattr(work_model.ln, "eps", 1e-5)),
                lambda_grid=cfg.lambda_grid,
                tau_grid=cfg.tau_grid,
                max_general_drift_rel=cfg.calibrator_max_general_drift_rel,
                seq_batch_size=cfg.batch_size,
            )
            lam_opt = float(grid_res["best_lambda_ln"])
            tau_opt = float(grid_res["best_tau_temp"])
            w_interp = anchor_ln_src.weight.detach().to(torch.float64) + lam_opt * (w_cand.to(torch.float64) - anchor_ln_src.weight.detach().to(torch.float64))
            b_interp = (
                b_anc.detach().to(torch.float64) + lam_opt * (b_cand_c.to(torch.float64) - b_anc.detach().to(torch.float64))
                if (b_anc is not None and b_cand_c is not None)
                else None
            )
            w_folded, b_folded = fold_temperature_into_norm_params(
                w_interp,
                b_interp,
                tau_opt,
            )
            work_model.ln.weight.copy_(w_folded.to(device=work_model.ln.weight.device, dtype=work_model.ln.weight.dtype))
            if b_cand is not None and b_folded is not None:
                b_cand.copy_(b_folded.to(device=b_cand.device, dtype=b_cand.dtype))
            ln_meta = {"best_lambda_ln": lam_opt, "best_tau_temp": tau_opt, "guard_feasible": bool(grid_res["guard_feasible"])}

        cand_sha = compute_state_sha256(work_model.state_dict())
        eval_error: str | None = None
        if cka_failed:
            cand_eval = {
                "nll_mean": None,
                "base_nll_mean": float(eval_a["base_nll_mean"]),
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
            energy_ok = False
        else:
            try:
                cand_nll = _run_eval(work_model)
                cand_eval = _score_domain_nll_vs_ref(ref_nll, cand_nll, cfg=cfg)
                cand_e = float(cand_eval["anse_energy"])
                beat_a = cand_e < float(eval_a["anse_energy"]) - cfg.min_energy_improvement
                beat_b = (not cfg.require_beat_both_arms) or (cand_e < float(eval_b["anse_energy"]) - cfg.min_energy_improvement)
                energy_ok = bool(beat_a and beat_b)
            except Exception as exc:
                eval_error = f"{type(exc).__name__}: {exc}"
                cand_eval = {
                    "nll_mean": None,
                    "base_nll_mean": float(eval_a["base_nll_mean"]),
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
                energy_ok = False

        if cka_failed:
            status = "ROLLED_BACK_CKA_GATE"
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

        expected_rb_sha = arm_a_sha
        if rolled_back:
            if cfg.rollback_target == "base" and base_snapshot is not None:
                rb_snap = base_snapshot
                expected_rb_sha = str(base_sha)
            elif cfg.rollback_target == "arm_b":
                rb_snap = arm_b_snapshot
                expected_rb_sha = arm_b_sha
            else:
                rb_snap = arm_a_snapshot
                expected_rb_sha = arm_a_sha
            cur_sd = work_model.state_dict()
            work_model.load_state_dict({k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in rb_snap.items()})

        final_sha = compute_state_sha256(work_model.state_dict())
        if rolled_back and final_sha != expected_rb_sha:
            raise RuntimeError(f"Atomic bitwise rollback failed: {final_sha[:12]} != {expected_rb_sha[:12]}")
    finally:
        work_model.train(was_training)
        arm_b_model.train(was_b_training)

    delta_e_a = float(cand_eval["anse_energy"]) - float(eval_a["anse_energy"])
    delta_e_b = float(cand_eval["anse_energy"]) - float(eval_b["anse_energy"])
    receipt: dict[str, Any] = {
        "object": OT_MOE_FUSION_RECEIPT_OBJECT,
        "schema": OT_MOE_FUSER_SCHEMA,
        "status": status,
        "accepted": accepted,
        "committed": accepted,
        "rolled_back": rolled_back,
        "rollback_target": cfg.rollback_target,
        "eval_error": eval_error,
        "guard_ok": bool(cand_eval["guard_ok"]),
        "min_cka_alignment": min_cka_seen,
        "nll_mean": float(cand_eval["nll_mean"]) if accepted else float(eval_a["nll_mean"]),
        "candidate_nll_mean": cand_eval["nll_mean"],
        "arm_a_nll_mean": float(eval_a["nll_mean"]),
        "arm_b_nll_mean": float(eval_b["nll_mean"]),
        "base_nll_mean": float(eval_a["base_nll_mean"]),
        "mean_diff_vs_ref": cand_eval["mean_diff_vs_ref"],
        "anse_energy": float(cand_eval["anse_energy"]),
        "arm_a_anse_energy": float(eval_a["anse_energy"]),
        "arm_b_anse_energy": float(eval_b["anse_energy"]),
        "delta_e_vs_arm_a": delta_e_a,
        "delta_e_vs_arm_b": delta_e_b,
        "general_ratio": cand_eval["general_ratio"],
        "general_pct_vs_ref": cand_eval["general_pct_vs_ref"],
        "worst_domain_id": cand_eval["worst_domain_id"],
        "worst_domain_ratio": cand_eval["worst_domain_ratio"],
        "worst_domain_pct_vs_ref": cand_eval["worst_domain_pct_vs_ref"],
        "per_domain": cand_eval["per_domain"],
        "layer_reports": layer_reports,
        "layernorm_calibration": ln_meta,
        "arm_a_state_sha256": arm_a_sha,
        "arm_b_state_sha256": arm_b_sha,
        "base_state_sha256": base_sha,
        "candidate_state_sha256": cand_sha,
        "final_state_sha256": final_sha,
        "config": asdict(cfg),
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return work_model, receipt


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_FUNC_SIM_WEIGHT",
    "DEFAULT_FUSION_ALPHA",
    "DEFAULT_GENERAL_PREFIXES",
    "DEFAULT_MIN_CKA_GATE",
    "DEFAULT_ORTHOGONAL_SYNERGY_BETA",
    "DEFAULT_ORTHOGONAL_SYNERGY_RANK",
    "DEFAULT_ORTHOGONAL_SYNERGY_SCALE",
    "DEFAULT_ROUTER_SIM_WEIGHT",
    "DEFAULT_WEIGHT_SIM_WEIGHT",
    "GENERAL_EXPERT_PREFIX",
    "OT_MOE_FUSER_SCHEMA",
    "OT_MOE_FUSION_RECEIPT_OBJECT",
    "OptimalTransportMoEFuserConfig",
    "VALID_ALIGNMENT_MODES",
    "VALID_ROLLBACK_TARGETS",
    "align_lora_rank_basis_procrustes",
    "canonical_json_bytes",
    "compute_cross_arm_expert_similarity_matrix",
    "compute_payload_sha256",
    "compute_state_sha256",
    "execute_verified_ot_moe_arm_fusion",
    "fuse_aligned_moe_layer_weights",
    "match_moe_experts_hungarian_procrustes",
    "solve_exact_hungarian_assignment",
    "verify_ot_moe_fusion_receipt_provenance",
]

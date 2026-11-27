# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Gen-3 Reine-Cortex Five-Directions (D1-D5) Unified Multi-Hive Campaign Orchestrator.

Composes all five merged Gen-3 Reine-Cortex neuro-control directions (Zenodo
``10.5281/zenodo.24542441`` Section 6 and Lean 4 ``ANSE.ReineCortexDirections``
formal invariants, empirically validated in ``PILOT11`` ``pilot11_la_ruche_vault.lock.json``
at ``-0.014684`` nats, PPL ``-1.458%``, ``D0 -0.248%``, ``PASS``) into a single
deterministic, stage-ratcheted CPU pipeline (:func:`execute_gen3_five_direction_campaign`):

1. **D5 — Proof-Carrying ZK-WSP Multi-Hive Intake** (:mod:`zk_wsp_federated_hive_verifier`):
   Verifies Fiat-Shamir ``P_0`` spot-checks and ``Lean4WSPAttestationReceipt`` bindings
   before Euclidean Multi-Krum aggregation inside ``P_0`` (`n >= 2f + 3`).
2. **D3 — Optimal-Transport Hungarian + Procrustes Cross-Arm Fusion** (:mod:`optimal_transport_moe_arm_fuser`):
   Recovers the ``O(E^3)`` expert permutation bijection and hidden/LoRA Procrustes rotation
   ``R_{B->A}`` to resolve router-expert co-adaptation mismatch inside ``P_0``.
3. **D2 — Kronecker-Factored Riemannian Null-Space Transport** (:mod:`kronecker_riemannian_nullspace_transport`):
   Executes damped K-FAC whitened ``R_step`` null-space transport under geodesic KL
   trust-region bounds, resolving curved deep-layer prose drift.
4. **D1 — Inter-Round Royal-Jelly Trophallaxis & Pheromone Emission** (:mod:`inter_round_royal_jelly_trophallaxis`):
   Purifies Forager Bee deltas via TIES sign-consensus + ``P_0`` before DiLoCo momentum
   accumulation, applies ``ker(G_D0)`` Fisher/folded-tau calibration, and emits signed
   ``QueenPheromoneDirective`` state for the next round.
5. **D4 — Dynamic Alveole Mitosis & Prose-Hessian 4-Bit NF4 Edge Export** (:mod:`alveole_mitosis_hessian_nf4_exporter`):
   Diagnoses and recycles causal-knockout corruptor cells inside ``P_0`` and exports
   eligible weights to diagonal prose-Hessian compensated 4-bit NF4 within the
   ``6,656 MiB`` 8 GB consumer GPU WDDM ceiling.

Every stage passes through a deterministic ANSE thermodynamic ratchet accepting iff
``Delta E < 0`` and general-prose drift ``D0 <= +1.0%`` (plus WDDM ceiling for D4),
enforcing atomic bitwise rollback under ``ANSE_MAXIMUM_PAIN`` (``1e6``) and chained
SHA-256 Merkle stage provenance.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from hivemind.swarm.alveole_mitosis_hessian_nf4_exporter import (
    STD_8GB_WDDM_CEILING_MIB,
    AlveoleMitosisNF4Config,
    execute_verified_alveole_mitosis_and_nf4_export,
)
from hivemind.swarm.inter_round_royal_jelly_trophallaxis import (
    InterRoundRoyalJellyTrophallaxisController,
    InterRoundTrophallaxisConfig,
)
from hivemind.swarm.kronecker_riemannian_nullspace_transport import (
    KroneckerRiemannianTransportConfig,
    execute_verified_kronecker_riemannian_surgery,
)
from hivemind.swarm.lean4_wsp_theorem_registry import Lean4WSPSpecAudit
from hivemind.swarm.optimal_transport_moe_arm_fuser import (
    OptimalTransportMoEFuserConfig,
    execute_verified_ot_moe_arm_fusion,
)
from hivemind.swarm.queen_cortex_moe_campaign_governor import (
    ANSE_MAXIMUM_PAIN,
    DEFAULT_ANSE_SAFE_D0_RATIO,
    DEFAULT_DOMAIN_GUARD_MAX_RATIO,
    DEFAULT_ECE_ENERGY_WEIGHT,
    DEFAULT_GENERAL_PREFIXES,
    DEFAULT_SOFT_MARGIN_PENALTY,
    QueenPheromoneDirective,
    _evaluate_model_on_chunks,
    _normalize_eval_result,
    _require_bool,
    _require_float,
    _require_int,
    _require_nonempty_str,
    _validate_probe_chunks,
    verify_queen_pheromone_directive_signature,
)
from hivemind.swarm.zk_wsp_federated_hive_verifier import (
    LEAN4_RCD_D5_THEOREM,
    ZKWSPFederatedHiveConfig,
    compute_payload_sha256,
    compute_state_sha256,
    execute_verified_zkwsp_federated_hive_round,
)

GEN3_ORCHESTRATOR_SCHEMA: str = "hivemind.reine_cortex_gen3_orchestrator/1"
GEN3_CAMPAIGN_RECEIPT_OBJECT: str = "hivemind.reine_cortex_gen3_campaign_receipt"
GEN3_STAGE_LEDGER_ENTRY_OBJECT: str = "hivemind.reine_cortex_gen3_stage_entry"
GENESIS_MERKLE_SHA256: str = "0" * 64

STAGE_D5_INTAKE: str = "D5_zkwsp_federated_intake"
STAGE_D3_FUSION: str = "D3_ot_cross_arm_fusion"
STAGE_D2_KRONECKER: str = "D2_kronecker_riemannian_transport"
STAGE_D1_TROPHALLAXIS: str = "D1_royal_jelly_trophallaxis"
STAGE_D4_MITOSIS_NF4: str = "D4_alveole_mitosis_hessian_nf4"

STAGE_ORDER: tuple[str, ...] = (
    STAGE_D5_INTAKE,
    STAGE_D3_FUSION,
    STAGE_D2_KRONECKER,
    STAGE_D1_TROPHALLAXIS,
    STAGE_D4_MITOSIS_NF4,
)

STAGE_DIRECTION_MAP: dict[str, str] = {
    STAGE_D5_INTAKE: "D5",
    STAGE_D3_FUSION: "D3",
    STAGE_D2_KRONECKER: "D2",
    STAGE_D1_TROPHALLAXIS: "D1",
    STAGE_D4_MITOSIS_NF4: "D4",
}

LEAN4_STAGE_THEOREM_MAP: dict[str, str] = {
    STAGE_D5_INTAKE: LEAN4_RCD_D5_THEOREM,
    STAGE_D3_FUSION: "ANSE.ReineCortexDirections.thm_rcd_d3_permutation_transport_moe_invariance",
    STAGE_D2_KRONECKER: "ANSE.ReineCortexDirections.thm_rcd_d2_kronecker_riemannian_nullspace_descent",
    STAGE_D1_TROPHALLAXIS: "ANSE.ReineCortexDirections.thm_rcd_d1_closed_loop_diloco_momentum_invariance",
    STAGE_D4_MITOSIS_NF4: "ANSE.ReineCortexDirections.thm_rcd_d4_alveole_mitosis_and_compensated_quant",
}

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# 1. Configuration, Merkle Helpers & Bitwise Snapshot Utilities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gen3ReineCortexCampaignConfig:
    """Validated configuration for the unified Gen-3 D1-D5 Reine-Cortex orchestrator."""

    campaign_id: str = "la-ruche-gen3-five-directions"
    round_id: str = "round_1"
    enable_d5_intake: bool = True
    enable_d3_fusion: bool = True
    enable_d2_kronecker: bool = True
    enable_d1_trophallaxis: bool = True
    enable_d4_mitosis_nf4: bool = True
    max_general_drift_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    max_worst_domain_drift_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    enforce_all_domain_guard: bool = False
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_D0_RATIO
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    ece_energy_weight: float = DEFAULT_ECE_ENERGY_WEIGHT
    min_energy_improvement: float = 1e-6
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    wddm_vram_ceiling_mib: float = float(STD_8GB_WDDM_CEILING_MIB)
    enforce_wddm_budget: bool = True
    projected_target_param_count: int | None = 3_300_000_000
    general_prefixes: tuple[str, ...] = DEFAULT_GENERAL_PREFIXES
    score_from: int = 4
    batch_size: int = 16
    inplace: bool = False

    def __post_init__(self) -> None:
        _require_nonempty_str(self.campaign_id, "campaign_id")
        _require_nonempty_str(self.round_id, "round_id")
        for flag in (
            "enable_d5_intake",
            "enable_d3_fusion",
            "enable_d2_kronecker",
            "enable_d1_trophallaxis",
            "enable_d4_mitosis_nf4",
            "enforce_all_domain_guard",
            "enforce_wddm_budget",
            "inplace",
        ):
            _require_bool(getattr(self, flag), flag)

        gen_cap = _require_float(self.max_general_drift_ratio, "max_general_drift_ratio", min_value=1.0)
        worst_cap = _require_float(self.max_worst_domain_drift_ratio, "max_worst_domain_drift_ratio", min_value=1.0)
        safe_gen = _require_float(self.anse_safe_general_ratio, "anse_safe_general_ratio", min_value=1.0)
        if safe_gen > gen_cap:
            raise ValueError(f"anse_safe_general_ratio ({safe_gen}) cannot exceed max_general_drift_ratio ({gen_cap})")
        _require_float(self.soft_margin_penalty, "soft_margin_penalty", min_value=0.0)
        _require_float(self.ece_energy_weight, "ece_energy_weight", min_value=0.0)
        _require_float(self.min_energy_improvement, "min_energy_improvement", min_value=0.0)
        _require_float(self.maximum_pain_energy, "maximum_pain_energy", min_value=1.0)
        _require_float(self.wddm_vram_ceiling_mib, "wddm_vram_ceiling_mib", min_value=0.0, strict_min=True)
        if worst_cap < 1.0:  # pragma: no cover - guarded by min_value=1.0 above
            raise ValueError("max_worst_domain_drift_ratio must be >= 1.0")
        if self.projected_target_param_count is not None:
            _require_int(self.projected_target_param_count, "projected_target_param_count", min_value=1)
        if (
            isinstance(self.general_prefixes, (str, bytes))
            or not isinstance(self.general_prefixes, Sequence)
            or not self.general_prefixes
        ):
            raise ValueError("general_prefixes must be a non-empty sequence of non-empty strings")
        cleaned_prefixes = tuple(_require_nonempty_str(p, "general_prefix") for p in self.general_prefixes)
        object.__setattr__(self, "general_prefixes", cleaned_prefixes)
        _require_int(self.score_from, "score_from", min_value=0)
        _require_int(self.batch_size, "batch_size", min_value=1)


def compute_merkle_root_sha256(digests: Sequence[str]) -> str:
    """Compute a deterministic binary SHA-256 Merkle root over ordered stage digests."""
    if isinstance(digests, (str, bytes)) or not isinstance(digests, Sequence) or not digests:
        raise ValueError("digests must be a non-empty sequence of 64-hex SHA-256 strings")
    level: list[str] = []
    for idx, val in enumerate(digests):
        if not isinstance(val, str) or not _SHA256_HEX_RE.match(val):
            raise ValueError(f"digests[{idx}] must be a 64-char lowercase SHA-256 hex digest")
        level.append(val)
    while len(level) > 1:
        nxt: list[str] = []
        for idx in range(0, len(level), 2):
            left = level[idx]
            right = level[idx + 1] if (idx + 1) < len(level) else left
            nxt.append(hashlib.sha256(f"{left}:{right}".encode()).hexdigest())
        level = nxt
    return level[0]


def _clone_state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot every parameter and buffer on CPU for atomic bitwise rollback."""
    return {k: v.detach().to(device="cpu").clone() for k, v in model.state_dict().items()}


def _restore_state_dict_bitwise(model: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
    """Restore model state bit-for-bit from a CPU snapshot."""
    cur_sd = model.state_dict()
    model.load_state_dict(
        {k: v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype) for k, v in snapshot.items()}
    )


# ---------------------------------------------------------------------------
# 2. Campaign Evaluation & Thermodynamic Stage Ratchet
# ---------------------------------------------------------------------------


def _evaluate_campaign_checkpoint(
    model: nn.Module,
    eval_chunks: Sequence[Mapping[str, Any]] | None,
    eval_fn: Callable[[nn.Module], Any] | None,
    *,
    score_from: int,
    general_prefixes: Sequence[str],
    label: str,
) -> dict[str, Any]:
    """Evaluate a checkpoint via ``eval_fn`` or ``eval_chunks`` into canonical ANSE metrics."""
    if eval_fn is not None:
        raw = eval_fn(model)
        if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], Mapping):
            raw = raw[1]
        return _normalize_eval_result(raw, general_prefixes, label)
    assert eval_chunks is not None
    return _evaluate_model_on_chunks(
        model,
        eval_chunks,
        score_from=score_from,
        general_prefixes=general_prefixes,
    )


def _score_stage_against_base(
    cand_eval: Mapping[str, Any],
    base_eval: Mapping[str, Any],
    parent_anse_energy: float,
    cfg: Gen3ReineCortexCampaignConfig,
) -> dict[str, Any]:
    """Score candidate stage vs campaign base (D0 <= +1.0%) and parent stage energy (Delta E < 0)."""
    if not cand_eval.get("all_finite", True):
        return {
            "all_finite": False,
            "general_drift_ratio": float("inf"),
            "general_drift_pct": None,
            "worst_domain_drift_ratio": float("inf"),
            "worst_domain_drift_pct": None,
            "guard_ok": False,
            "candidate_anse_energy": float(cfg.maximum_pain_energy),
            "delta_e": float(cfg.maximum_pain_energy),
            "energy_improved": False,
        }

    base_doms = base_eval["domain_nlls"]
    cand_doms = cand_eval["domain_nlls"]
    if set(cand_doms.keys()) != set(base_doms.keys()):
        raise ValueError("Candidate and baseline evaluations must share identical domain keys")

    gen_ratio = float(cand_eval["general_nll"]) / max(1e-9, float(base_eval["general_nll"]))
    worst_ratio = max(float(cand_doms[d]) / max(1e-9, float(base_doms[d])) for d in base_doms)
    gen_ok = gen_ratio <= cfg.max_general_drift_ratio + 1e-12
    worst_ok = (not cfg.enforce_all_domain_guard) or (worst_ratio <= cfg.max_worst_domain_drift_ratio + 1e-12)
    guard_ok = bool(gen_ok and worst_ok)

    if not guard_ok:
        cand_energy = float(cfg.maximum_pain_energy)
        delta_e = float(cfg.maximum_pain_energy)
    else:
        mean_diff = float(cand_eval["nll_mean"]) - float(base_eval["nll_mean"])
        ece_diff = float(cand_eval["ece_value"]) - float(base_eval["ece_value"])
        margin_pen = cfg.soft_margin_penalty if gen_ratio > cfg.anse_safe_general_ratio else 0.0
        cand_energy = float(mean_diff + cfg.ece_energy_weight * ece_diff + margin_pen)
        delta_e = float(cand_energy - parent_anse_energy)

    return {
        "all_finite": True,
        "general_drift_ratio": gen_ratio,
        "general_drift_pct": 100.0 * (gen_ratio - 1.0),
        "worst_domain_drift_ratio": worst_ratio,
        "worst_domain_drift_pct": 100.0 * (worst_ratio - 1.0),
        "guard_ok": guard_ok,
        "candidate_anse_energy": cand_energy,
        "delta_e": delta_e,
        "energy_improved": bool(guard_ok and (delta_e < -cfg.min_energy_improvement)),
    }


def _make_stage_merkle_entry(
    *,
    stage_index: int,
    stage_name: str,
    cfg: Gen3ReineCortexCampaignConfig,
    status: str,
    accepted: bool,
    skipped: bool,
    rolled_back: bool,
    pre_sha: str,
    cand_sha: str,
    final_sha: str,
    nll_mean: float,
    ece_value: float,
    general_drift_ratio: float | None,
    worst_domain_drift_ratio: float | None,
    guard_ok: bool,
    parent_energy: float,
    cand_energy: float,
    final_energy: float,
    delta_e: float,
    sub_receipt_sha: str | None,
    sub_summary: Mapping[str, Any],
    prev_merkle_sha: str,
) -> dict[str, Any]:
    """Assemble and SHA-256-seal one immutable stage entry in the Merkle chain."""
    entry_core: dict[str, Any] = {
        "object": GEN3_STAGE_LEDGER_ENTRY_OBJECT,
        "schema": GEN3_ORCHESTRATOR_SCHEMA,
        "campaign_id": cfg.campaign_id,
        "round_id": cfg.round_id,
        "stage_index": stage_index,
        "stage_name": stage_name,
        "direction": STAGE_DIRECTION_MAP[stage_name],
        "lean4_theorem_id": LEAN4_STAGE_THEOREM_MAP[stage_name],
        "status": status,
        "accepted": bool(accepted),
        "skipped": bool(skipped),
        "rolled_back": bool(rolled_back),
        "rollback_bitwise_verified": bool(rolled_back and (final_sha == pre_sha)),
        "pre_state_sha256": pre_sha,
        "candidate_state_sha256": cand_sha,
        "final_state_sha256": final_sha,
        "nll_mean": float(nll_mean),
        "ece_value": float(ece_value),
        "general_drift_ratio": general_drift_ratio,
        "worst_domain_drift_ratio": worst_domain_drift_ratio,
        "guard_ok": bool(guard_ok),
        "parent_anse_energy": float(parent_energy),
        "candidate_anse_energy": float(cand_energy),
        "anse_energy": float(cand_energy if not accepted else final_energy),
        "final_anse_energy": float(final_energy),
        "delta_e": float(delta_e),
        "substage_receipt_sha256": sub_receipt_sha,
        "substage_summary": dict(sub_summary),
        "prev_stage_merkle_sha256": prev_merkle_sha,
    }
    return {**entry_core, "stage_merkle_sha256": compute_payload_sha256(entry_core)}


# ---------------------------------------------------------------------------
# 3. Per-Direction Stage Executors (D5 -> D3 -> D2 -> D1 -> D4)
# ---------------------------------------------------------------------------


def _run_d5_stage(
    cand_model: nn.Module,
    *,
    cfg: Gen3ReineCortexCampaignConfig,
    hive_packets: Sequence[Mapping[str, Any]],
    subspace_commitment: Mapping[str, Any] | torch.Tensor | None,
    prose_keys: torch.Tensor | Mapping[str, torch.Tensor] | None,
    probe_chunks: Sequence[Mapping[str, Any]] | None,
    eval_chunks: Sequence[Mapping[str, Any]] | None,
    eval_fn: Callable[[nn.Module], Any] | None,
    lean_spec_audit: Lean4WSPSpecAudit | None,
    lean_spec_path: Path | str | None,
    d5_config: ZKWSPFederatedHiveConfig | None,
) -> dict[str, Any]:
    """Execute Gen-3 Direction D5: ZK-WSP + Lean 4 + Multi-Krum multi-hive intake."""
    if d5_config is not None and not isinstance(d5_config, ZKWSPFederatedHiveConfig):
        raise TypeError(f"d5_config must be ZKWSPFederatedHiveConfig or None, got {type(d5_config).__name__}")
    eff_cfg = (
        dataclasses.replace(d5_config, inplace=False)
        if d5_config is not None
        else ZKWSPFederatedHiveConfig(
            domain_guard_max_ratio=cfg.max_general_drift_ratio,
            enforce_all_domain_guard=cfg.enforce_all_domain_guard,
            general_prefixes=cfg.general_prefixes,
            score_from=cfg.score_from,
            batch_size=cfg.batch_size,
            inplace=False,
        )
    )
    out_model, receipt = execute_verified_zkwsp_federated_hive_round(
        cand_model,
        hive_packets,
        subspace_commitment=subspace_commitment,
        prose_keys=prose_keys,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        eval_fn=eval_fn,
        lean_spec_audit=lean_spec_audit,
        lean_spec_path=lean_spec_path,
        round_id=cfg.round_id,
        config=eff_cfg,
    )
    audit = receipt.get("zkwsp_audit", {})
    det_audit = {
        k: v for k, v in audit.items() if k not in ("verification_and_krum_ms", "per_peer_verify_ms")
    }
    det_receipt = {
        k: (det_audit if k == "zkwsp_audit" else v)
        for k, v in receipt.items()
        if k != "receipt_sha256"
    }
    sub_status = str(receipt.get("status", ""))
    return {
        "model": out_model,
        "sub_accepted": bool(receipt.get("accepted", False)),
        "sub_guard_failed": (not bool(receipt.get("guard_ok", False))) or ("QUORUM" in sub_status) or ("ALL_HIVES" in sub_status),
        "sub_status": sub_status,
        "sub_receipt_sha256": compute_payload_sha256(det_receipt),
        "wddm_footprint": None,
        "d1_pheromone_directive": None,
        "sub_summary": {
            "sub_status": sub_status,
            "num_submitted": int(audit.get("num_submitted", 0)),
            "num_verified": int(audit.get("num_verified", 0)),
            "num_rejected_by_verifier": int(audit.get("num_rejected_by_verifier", 0)),
            "multikrum_selected_hives": list(audit.get("multikrum_selected_hives", [])),
            "rejected_hive_ids": [r.get("hive_id") for r in audit.get("rejected_hives", [])],
        },
    }


def _run_d3_stage(
    cand_model: nn.Module,
    *,
    cfg: Gen3ReineCortexCampaignConfig,
    cross_arm_model: nn.Module,
    base_snapshot_model: nn.Module,
    base_domain_nlls: Mapping[str, float],
    probe_chunks: Sequence[Mapping[str, Any]] | None,
    eval_chunks: Sequence[Mapping[str, Any]] | None,
    eval_fn: Callable[[nn.Module], Any] | None,
    p_null_by_layer: Mapping[int, Any] | torch.Tensor | None,
    d3_config: OptimalTransportMoEFuserConfig | None,
) -> dict[str, Any]:
    """Execute Gen-3 Direction D3: OT Hungarian + Procrustes cross-arm fusion."""
    if d3_config is not None and not isinstance(d3_config, OptimalTransportMoEFuserConfig):
        raise TypeError(f"d3_config must be OptimalTransportMoEFuserConfig or None, got {type(d3_config).__name__}")
    eff_cfg = (
        dataclasses.replace(d3_config, inplace=False)
        if d3_config is not None
        else OptimalTransportMoEFuserConfig(
            max_general_drift_ratio=cfg.max_general_drift_ratio,
            max_worst_domain_drift_ratio=cfg.max_worst_domain_drift_ratio,
            enforce_all_domain_guard=cfg.enforce_all_domain_guard,
            general_prefixes=cfg.general_prefixes,
            score_from=cfg.score_from,
            batch_size=cfg.batch_size,
            inplace=False,
        )
    )
    out_model, receipt = execute_verified_ot_moe_arm_fusion(
        cand_model,
        cross_arm_model,
        calibration_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        base_model=base_snapshot_model,
        p_null_by_layer=p_null_by_layer,
        eval_fn=eval_fn,
        baseline_domain_nll=base_domain_nlls,
        config=eff_cfg,
    )
    sub_status = str(receipt.get("status", ""))
    perms = {
        str(li): list(rep.get("permutation", []))
        for li, rep in receipt.get("layer_reports", {}).items()
        if "permutation" in rep
    }
    return {
        "model": out_model,
        "sub_accepted": bool(receipt.get("accepted", False)),
        "sub_guard_failed": (not bool(receipt.get("guard_ok", False))) or ("CKA" in sub_status),
        "sub_status": sub_status,
        "sub_receipt_sha256": receipt.get("receipt_sha256"),
        "wddm_footprint": None,
        "d1_pheromone_directive": None,
        "sub_summary": {
            "sub_status": sub_status,
            "min_cka_alignment": float(receipt.get("min_cka_alignment", 0.0)),
            "layer_permutations": perms,
        },
    }


def _run_d2_stage(
    cand_model: nn.Module,
    *,
    cfg: Gen3ReineCortexCampaignConfig,
    base_domain_nlls: Mapping[str, float],
    probe_chunks: Sequence[Mapping[str, Any]] | None,
    eval_chunks: Sequence[Mapping[str, Any]] | None,
    eval_fn: Callable[[nn.Module], Any] | None,
    kronecker_target_deltas: Mapping[str, torch.Tensor] | None,
    kronecker_layer_indices: Sequence[int] | None,
    kronecker_expert_indices: Sequence[int] | None,
    d2_config: KroneckerRiemannianTransportConfig | None,
) -> dict[str, Any]:
    """Execute Gen-3 Direction D2: Kronecker-factored Riemannian null-space transport."""
    if d2_config is not None and not isinstance(d2_config, KroneckerRiemannianTransportConfig):
        raise TypeError(f"d2_config must be KroneckerRiemannianTransportConfig or None, got {type(d2_config).__name__}")
    # D2 returns only a receipt and mutates `cand_model` when inplace=True; `cand_model` is already a deep copy.
    eff_cfg = (
        dataclasses.replace(d2_config, inplace=True)
        if d2_config is not None
        else KroneckerRiemannianTransportConfig(
            domain_guard_max_ratio=cfg.max_general_drift_ratio,
            enforce_all_domain_guard=cfg.enforce_all_domain_guard,
            general_prefixes=cfg.general_prefixes,
            score_from=cfg.score_from,
            batch_size=cfg.batch_size,
            inplace=True,
        )
    )
    receipt = execute_verified_kronecker_riemannian_surgery(
        cand_model,
        probe_chunks,
        eval_chunks=eval_chunks,
        eval_fn=eval_fn,
        baseline_domain_nll=base_domain_nlls,
        target_deltas=kronecker_target_deltas,
        layer_indices=kronecker_layer_indices,
        expert_indices=kronecker_expert_indices,
        config=eff_cfg,
    )
    sub_status = str(receipt.get("status", ""))
    return {
        "model": cand_model,
        "sub_accepted": bool(receipt.get("accepted", False)),
        "sub_guard_failed": not bool(receipt.get("guard_ok", False)),
        "sub_status": sub_status,
        "sub_receipt_sha256": receipt.get("receipt_sha256"),
        "wddm_footprint": None,
        "d1_pheromone_directive": None,
        "sub_summary": {
            "sub_status": sub_status,
            "total_kl_nats": float(receipt.get("total_kl_nats", 0.0)),
            "kl_ratio_vs_euclidean_p0": float(receipt.get("kl_ratio_vs_euclidean_p0", 0.0)),
            "units_edited": int(receipt.get("units_edited", 0)),
        },
    }


def _run_d1_stage(
    cand_model: nn.Module,
    *,
    cfg: Gen3ReineCortexCampaignConfig,
    probe_chunks: Sequence[Mapping[str, Any]] | None,
    eval_chunks: Sequence[Mapping[str, Any]] | None,
    eval_fn: Callable[[nn.Module], Any] | None,
    trophallaxis_rounds: Sequence[Mapping[str, Any]] | None,
    forager_bee_deltas: Sequence[Mapping[str | int, Any]] | Mapping[str | int, Any] | None,
    p_null_by_expert: Mapping[str | int, Any] | None,
    bee_domains: Sequence[str] | Mapping[str, str] | None,
    bee_weights: Sequence[float] | None,
    initial_directive: QueenPheromoneDirective | Mapping[str, Any] | None,
    expert_causal_gains_nats: Mapping[str | int, float] | None,
    expert_prose_harm_nats: Mapping[str | int, float] | None,
    router_expert_kl_mismatch: Mapping[str | int, float] | float,
    d1_config: InterRoundTrophallaxisConfig | None,
) -> dict[str, Any]:
    """Execute Gen-3 Direction D1: Inter-round Royal-Jelly trophallaxis & pheromone emission."""
    if d1_config is not None and not isinstance(d1_config, InterRoundTrophallaxisConfig):
        raise TypeError(f"d1_config must be InterRoundTrophallaxisConfig or None, got {type(d1_config).__name__}")
    eff_cfg = (
        dataclasses.replace(d1_config, inplace=False)
        if d1_config is not None
        else InterRoundTrophallaxisConfig(
            max_domain_drift_ratio=cfg.max_general_drift_ratio,
            ece_energy_weight=cfg.ece_energy_weight,
            general_prefixes=cfg.general_prefixes,
            score_from=cfg.score_from,
            batch_size=cfg.batch_size,
            inplace=False,
        )
    )
    ctrl = InterRoundRoyalJellyTrophallaxisController(
        cand_model,
        config=eff_cfg,
        eval_fn=eval_fn,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        p_null_by_expert=p_null_by_expert,
        campaign_id=f"{cfg.campaign_id}::{cfg.round_id}::d1",
        initial_directive=initial_directive,
    )
    if trophallaxis_rounds is not None:
        out_model, d1_summary = ctrl.run_campaign(trophallaxis_rounds)
        accepted_cnt = int(d1_summary["accepted_rounds"])
        sub_accepted = accepted_cnt > 0
        sub_receipt_sha = str(d1_summary["summary_sha256"])
        sub_status = "CAMPAIGN_ACCEPTED" if sub_accepted else "ROLLED_BACK_ALL_ROUNDS"
        max_leak = float(d1_summary["max_momentum_prose_leakage"])
        rescued_cnt = int(d1_summary["rescued_rounds"])
        sub_guard_failed = not sub_accepted
    else:
        assert forager_bee_deltas is not None
        out_model, d1_rec = ctrl.step_outer_round(
            forager_bee_deltas,
            p_null_by_expert=p_null_by_expert,
            bee_domains=bee_domains,
            bee_weights=bee_weights,
            expert_causal_gains_nats=expert_causal_gains_nats,
            expert_prose_harm_nats=expert_prose_harm_nats,
            router_expert_kl_mismatch=router_expert_kl_mismatch,
        )
        sub_accepted = bool(d1_rec["accepted"])
        accepted_cnt = 1 if sub_accepted else 0
        sub_receipt_sha = str(d1_rec["receipt_sha256"])
        sub_status = str(d1_rec["status"])
        max_leak = float(d1_rec["momentum_prose_leakage"]["frobenius_leakage"])
        rescued_cnt = 1 if d1_rec["rescued_from_discarded_round"] else 0
        sub_guard_failed = not bool(d1_rec.get("guard_ok", False))

    directive = ctrl.active_directive if sub_accepted else None
    return {
        "model": out_model,
        "sub_accepted": sub_accepted,
        "sub_guard_failed": sub_guard_failed,
        "sub_status": sub_status,
        "sub_receipt_sha256": sub_receipt_sha,
        "wddm_footprint": None,
        "d1_pheromone_directive": directive,
        "sub_summary": {
            "sub_status": sub_status,
            "accepted_rounds": accepted_cnt,
            "rescued_rounds": rescued_cnt,
            "max_momentum_prose_leakage": max_leak,
            "momentum_sha256": ctrl.momentum_sha256,
            "emitted_directive_sha256": directive["directive_sha256"] if directive is not None else None,
        },
    }


def _run_d4_stage(
    cand_model: nn.Module,
    *,
    cfg: Gen3ReineCortexCampaignConfig,
    base_domain_nlls: Mapping[str, float],
    probe_chunks: Sequence[Mapping[str, Any]] | None,
    eval_chunks: Sequence[Mapping[str, Any]] | None,
    eval_fn: Callable[[nn.Module], Any] | None,
    p_null_by_layer: Mapping[int, Any] | torch.Tensor | None,
    mitosis_plan: Sequence[Mapping[str, Any]] | None,
    explicit_knockout_deltas: Mapping[int, float | Mapping[str, float]] | None,
    hessian_diag_by_layer: Mapping[str | int, torch.Tensor] | None,
    bos_sink_fn: Callable[[nn.Module], float] | None,
    d4_config: AlveoleMitosisNF4Config | None,
) -> dict[str, Any]:
    """Execute Gen-3 Direction D4: Dynamic Alveole mitosis + prose-Hessian 4-bit NF4 export."""
    if d4_config is not None and not isinstance(d4_config, AlveoleMitosisNF4Config):
        raise TypeError(f"d4_config must be AlveoleMitosisNF4Config or None, got {type(d4_config).__name__}")
    eff_cfg = (
        dataclasses.replace(d4_config, inplace=False)
        if d4_config is not None
        else AlveoleMitosisNF4Config(
            wddm_vram_ceiling_mib=cfg.wddm_vram_ceiling_mib,
            enforce_wddm_budget=cfg.enforce_wddm_budget,
            projected_target_param_count=cfg.projected_target_param_count,
            max_general_drift_ratio=cfg.max_general_drift_ratio,
            max_worst_domain_drift_ratio=cfg.max_worst_domain_drift_ratio,
            enforce_all_domain_guard=cfg.enforce_all_domain_guard,
            general_prefixes=cfg.general_prefixes,
            score_from=cfg.score_from,
            batch_size=cfg.batch_size,
            inplace=False,
        )
    )
    out_model, receipt = execute_verified_alveole_mitosis_and_nf4_export(
        cand_model,
        calibration_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        eval_fn=eval_fn,
        baseline_domain_nll=base_domain_nlls,
        p_null_by_layer=p_null_by_layer,
        hessian_diag_by_layer=hessian_diag_by_layer,
        explicit_knockout_deltas=explicit_knockout_deltas,
        mitosis_plan=mitosis_plan,
        bos_sink_fn=bos_sink_fn,
        config=eff_cfg,
    )
    nf4_exp = receipt.get("nf4_export")
    wddm = nf4_exp.get("wddm_footprint") if isinstance(nf4_exp, Mapping) else None
    sub_status = str(receipt.get("status", ""))
    cand_eval_d4 = receipt.get("candidate_eval", {})
    return {
        "model": out_model,
        "sub_accepted": bool(receipt.get("accepted", False)),
        "sub_guard_failed": (not bool(cand_eval_d4.get("guard_ok", False))) or ("WDDM" in sub_status) or ("BOS" in sub_status),
        "sub_status": sub_status,
        "sub_receipt_sha256": receipt.get("receipt_sha256"),
        "wddm_footprint": dict(wddm) if isinstance(wddm, Mapping) else None,
        "d1_pheromone_directive": None,
        "sub_summary": {
            "sub_status": sub_status,
            "num_recycled": int(receipt.get("mitosis", {}).get("num_recycled", 0)),
            "bos_sink_abs_drift": float(receipt.get("bos_sink_abs_drift", 0.0)),
            "wddm_footprint": dict(wddm) if isinstance(wddm, Mapping) else None,
        },
    }


# ---------------------------------------------------------------------------
# 4. Unified Five-Direction Campaign Orchestrator & Merkle Provenance Audit
# ---------------------------------------------------------------------------


@torch.no_grad()
def execute_gen3_five_direction_campaign(
    model: nn.Module,
    *,
    probe_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_fn: Callable[[nn.Module], Any] | None = None,
    config: Gen3ReineCortexCampaignConfig | None = None,
    hive_packets: Sequence[Mapping[str, Any]] | None = None,
    subspace_commitment: Mapping[str, Any] | torch.Tensor | None = None,
    prose_keys: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    lean_spec_audit: Lean4WSPSpecAudit | None = None,
    lean_spec_path: Path | str | None = None,
    d5_config: ZKWSPFederatedHiveConfig | None = None,
    cross_arm_model: nn.Module | None = None,
    p_null_by_layer: Mapping[int, Any] | torch.Tensor | None = None,
    d3_config: OptimalTransportMoEFuserConfig | None = None,
    kronecker_target_deltas: Mapping[str, torch.Tensor] | None = None,
    kronecker_layer_indices: Sequence[int] | None = None,
    kronecker_expert_indices: Sequence[int] | None = None,
    d2_config: KroneckerRiemannianTransportConfig | None = None,
    trophallaxis_rounds: Sequence[Mapping[str, Any]] | None = None,
    forager_bee_deltas: Sequence[Mapping[str | int, Any]] | Mapping[str | int, Any] | None = None,
    p_null_by_expert: Mapping[str | int, Any] | None = None,
    bee_domains: Sequence[str] | Mapping[str, str] | None = None,
    bee_weights: Sequence[float] | None = None,
    initial_directive: QueenPheromoneDirective | Mapping[str, Any] | None = None,
    expert_causal_gains_nats: Mapping[str | int, float] | None = None,
    expert_prose_harm_nats: Mapping[str | int, float] | None = None,
    router_expert_kl_mismatch: Mapping[str | int, float] | float = 0.0,
    d1_config: InterRoundTrophallaxisConfig | None = None,
    mitosis_plan: Sequence[Mapping[str, Any]] | None = None,
    explicit_knockout_deltas: Mapping[int, float | Mapping[str, float]] | None = None,
    hessian_diag_by_layer: Mapping[str | int, torch.Tensor] | None = None,
    bos_sink_fn: Callable[[nn.Module], float] | None = None,
    d4_config: AlveoleMitosisNF4Config | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Run stage-ratcheted Gen-3 D5 -> D3 -> D2 -> D1 -> D4 Reine-Cortex campaign on CPU."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if not model.state_dict():
        raise ValueError("model.state_dict() must be non-empty")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError("eval_fn must be callable or None")
    if probe_chunks is None and eval_chunks is None and eval_fn is None:
        raise ValueError("Provide at least one of probe_chunks, eval_chunks, or eval_fn")

    cfg = config if config is not None else Gen3ReineCortexCampaignConfig()
    if not isinstance(cfg, Gen3ReineCortexCampaignConfig):
        raise TypeError(f"config must be Gen3ReineCortexCampaignConfig or None, got {type(cfg).__name__}")

    head_mod = getattr(model, "head", None) or getattr(model, "lm_head", None)
    vocab_sz = int(head_mod.weight.shape[0]) if head_mod is not None and hasattr(head_mod, "weight") else None
    val_probe = (
        _validate_probe_chunks(probe_chunks, vocab_size=vocab_sz, general_prefixes=cfg.general_prefixes, name="probe_chunks")
        if (probe_chunks is not None and vocab_sz is not None)
        else probe_chunks
    )
    raw_eval_chunks = eval_chunks if eval_chunks is not None else val_probe
    val_eval = (
        _validate_probe_chunks(raw_eval_chunks, vocab_size=vocab_sz, general_prefixes=cfg.general_prefixes, name="eval_chunks")
        if (raw_eval_chunks is not None and vocab_sz is not None)
        else raw_eval_chunks
    )
    if eval_fn is None and (
        val_eval is None or vocab_sz is None or not hasattr(model, "blocks") or not hasattr(model, "cos_cached")
    ):
        raise ValueError("eval_fn or eval_chunks/probe_chunks with a compatible model head is required")

    ratchet_model = model if cfg.inplace else copy.deepcopy(model)
    base_snapshot_model = copy.deepcopy(ratchet_model)
    base_snapshot = _clone_state_dict_cpu(ratchet_model)
    base_sha = compute_state_sha256(ratchet_model)

    was_training = ratchet_model.training
    ratchet_model.eval()
    base_snapshot_model.eval()
    try:
        base_eval = _evaluate_campaign_checkpoint(
            ratchet_model,
            val_eval,
            eval_fn,
            score_from=cfg.score_from,
            general_prefixes=cfg.general_prefixes,
            label="campaign_base_eval",
        )
        base_domain_nlls = dict(base_eval["domain_nlls"])
        current_eval = dict(base_eval)
        current_anse_energy = 0.0
        prev_merkle_sha = GENESIS_MERKLE_SHA256

        stage_ledger: list[dict[str, Any]] = []
        d1_pheromone_directive: dict[str, Any] | None = None
        wddm_footprint: dict[str, Any] | None = None

        stage_specs: tuple[tuple[str, bool, Callable[[nn.Module], dict[str, Any]]], ...] = (
            (
                STAGE_D5_INTAKE,
                bool(cfg.enable_d5_intake and hive_packets is not None),
                lambda m: _run_d5_stage(
                    m,
                    cfg=cfg,
                    hive_packets=hive_packets or (),
                    subspace_commitment=subspace_commitment,
                    prose_keys=prose_keys,
                    probe_chunks=val_probe,
                    eval_chunks=val_eval,
                    eval_fn=eval_fn,
                    lean_spec_audit=lean_spec_audit,
                    lean_spec_path=lean_spec_path,
                    d5_config=d5_config,
                ),
            ),
            (
                STAGE_D3_FUSION,
                bool(cfg.enable_d3_fusion and cross_arm_model is not None),
                lambda m: _run_d3_stage(
                    m,
                    cfg=cfg,
                    cross_arm_model=cross_arm_model,  # type: ignore[arg-type]
                    base_snapshot_model=base_snapshot_model,
                    base_domain_nlls=base_domain_nlls,
                    probe_chunks=val_probe,
                    eval_chunks=val_eval,
                    eval_fn=eval_fn,
                    p_null_by_layer=p_null_by_layer,
                    d3_config=d3_config,
                ),
            ),
            (
                STAGE_D2_KRONECKER,
                bool(cfg.enable_d2_kronecker),
                lambda m: _run_d2_stage(
                    m,
                    cfg=cfg,
                    base_domain_nlls=base_domain_nlls,
                    probe_chunks=val_probe,
                    eval_chunks=val_eval,
                    eval_fn=eval_fn,
                    kronecker_target_deltas=kronecker_target_deltas,
                    kronecker_layer_indices=kronecker_layer_indices,
                    kronecker_expert_indices=kronecker_expert_indices,
                    d2_config=d2_config,
                ),
            ),
            (
                STAGE_D1_TROPHALLAXIS,
                bool(cfg.enable_d1_trophallaxis and (trophallaxis_rounds is not None or forager_bee_deltas is not None)),
                lambda m: _run_d1_stage(
                    m,
                    cfg=cfg,
                    probe_chunks=val_probe,
                    eval_chunks=val_eval,
                    eval_fn=eval_fn,
                    trophallaxis_rounds=trophallaxis_rounds,
                    forager_bee_deltas=forager_bee_deltas,
                    p_null_by_expert=p_null_by_expert,
                    bee_domains=bee_domains,
                    bee_weights=bee_weights,
                    initial_directive=initial_directive,
                    expert_causal_gains_nats=expert_causal_gains_nats,
                    expert_prose_harm_nats=expert_prose_harm_nats,
                    router_expert_kl_mismatch=router_expert_kl_mismatch,
                    d1_config=d1_config,
                ),
            ),
            (
                STAGE_D4_MITOSIS_NF4,
                bool(cfg.enable_d4_mitosis_nf4),
                lambda m: _run_d4_stage(
                    m,
                    cfg=cfg,
                    base_domain_nlls=base_domain_nlls,
                    probe_chunks=val_probe,
                    eval_chunks=val_eval,
                    eval_fn=eval_fn,
                    p_null_by_layer=p_null_by_layer,
                    mitosis_plan=mitosis_plan,
                    explicit_knockout_deltas=explicit_knockout_deltas,
                    hessian_diag_by_layer=hessian_diag_by_layer,
                    bos_sink_fn=bos_sink_fn,
                    d4_config=d4_config,
                ),
            ),
        )

        for idx, (stage_name, should_run, runner) in enumerate(stage_specs, start=1):
            pre_sha = compute_state_sha256(ratchet_model)
            pre_snap = _clone_state_dict_cpu(ratchet_model)
            parent_scored = _score_stage_against_base(current_eval, base_eval, current_anse_energy, cfg)

            if not should_run:
                entry = _make_stage_merkle_entry(
                    stage_index=idx,
                    stage_name=stage_name,
                    cfg=cfg,
                    status="SKIPPED_DISABLED_OR_NO_INPUT",
                    accepted=False,
                    skipped=True,
                    rolled_back=False,
                    pre_sha=pre_sha,
                    cand_sha=pre_sha,
                    final_sha=pre_sha,
                    nll_mean=float(current_eval["nll_mean"]),
                    ece_value=float(current_eval["ece_value"]),
                    general_drift_ratio=float(parent_scored["general_drift_ratio"]),
                    worst_domain_drift_ratio=float(parent_scored["worst_domain_drift_ratio"]),
                    guard_ok=bool(parent_scored["guard_ok"]),
                    parent_energy=current_anse_energy,
                    cand_energy=current_anse_energy,
                    final_energy=current_anse_energy,
                    delta_e=0.0,
                    sub_receipt_sha=None,
                    sub_summary={"reason": "disabled_or_missing_input"},
                    prev_merkle_sha=prev_merkle_sha,
                )
                stage_ledger.append(entry)
                prev_merkle_sha = str(entry["stage_merkle_sha256"])
                continue

            cand_input = copy.deepcopy(ratchet_model)
            stage_out = runner(cand_input)
            cand_model: nn.Module = stage_out["model"]
            cand_sha = compute_state_sha256(cand_model)
            sub_accepted = bool(stage_out["sub_accepted"])
            sub_guard_failed = bool(stage_out["sub_guard_failed"])
            stage_wddm = stage_out["wddm_footprint"]
            wddm_ok = (
                True
                if (not cfg.enforce_wddm_budget or stage_wddm is None)
                else bool(stage_wddm.get("fits_wddm_ceiling", False))
            )

            eval_err: str | None = None
            try:
                cand_eval = _evaluate_campaign_checkpoint(
                    cand_model,
                    val_eval,
                    eval_fn,
                    score_from=cfg.score_from,
                    general_prefixes=cfg.general_prefixes,
                    label=f"{stage_name}_eval",
                )
                scored = _score_stage_against_base(cand_eval, base_eval, current_anse_energy, cfg)
            except Exception as exc:
                eval_err = f"{type(exc).__name__}: {exc}"
                cand_eval = {
                    "all_finite": False,
                    "nll_mean": float("nan"),
                    "ece_value": 1.0,
                    "general_nll": float("nan"),
                    "domain_nlls": dict(current_eval["domain_nlls"]),
                    "domain_eces": dict(current_eval["domain_eces"]),
                }
                scored = _score_stage_against_base(cand_eval, base_eval, current_anse_energy, cfg)

            can_commit = bool(
                sub_accepted
                and eval_err is None
                and wddm_ok
                and scored["guard_ok"]
                and scored["energy_improved"]
                and cand_sha != pre_sha
            )
            pre_stage_energy = current_anse_energy

            if can_commit:
                _restore_state_dict_bitwise(ratchet_model, _clone_state_dict_cpu(cand_model))
                final_sha = compute_state_sha256(ratchet_model)
                current_eval = dict(cand_eval)
                cand_energy = float(scored["candidate_anse_energy"])
                delta_e = float(scored["delta_e"])
                current_anse_energy = cand_energy
                accepted = True
                rolled_back = False
                status = "ACCEPTED"
                out_nll = float(cand_eval["nll_mean"])
                out_ece = float(cand_eval["ece_value"])
                out_gen_r = float(scored["general_drift_ratio"])
                out_worst_r = float(scored["worst_domain_drift_ratio"])
                out_guard_ok = True
                if stage_out["d1_pheromone_directive"] is not None:
                    d1_pheromone_directive = dict(stage_out["d1_pheromone_directive"])
                if stage_wddm is not None:
                    wddm_footprint = dict(stage_wddm)
            else:
                _restore_state_dict_bitwise(ratchet_model, pre_snap)
                final_sha = compute_state_sha256(ratchet_model)
                if final_sha != pre_sha:  # pragma: no cover - defensive bitwise safety check
                    raise RuntimeError(f"Atomic bitwise rollback failed at {stage_name}")
                accepted = False
                rolled_back = True
                out_nll = float(current_eval["nll_mean"])
                out_ece = float(current_eval["ece_value"])
                out_gen_r = scored["general_drift_ratio"] if math.isfinite(float(scored["general_drift_ratio"])) else None
                out_worst_r = (
                    scored["worst_domain_drift_ratio"]
                    if math.isfinite(float(scored["worst_domain_drift_ratio"]))
                    else None
                )
                out_guard_ok = False
                if eval_err is not None:
                    status = "ROLLED_BACK_EVAL_FAILURE"
                    cand_energy = float(cfg.maximum_pain_energy)
                    delta_e = float(cfg.maximum_pain_energy)
                elif not wddm_ok:
                    status = "ROLLED_BACK_WDDM_VRAM_EXCEEDED"
                    cand_energy = float(cfg.maximum_pain_energy)
                    delta_e = float(cfg.maximum_pain_energy)
                elif not scored["guard_ok"] or sub_guard_failed:
                    status = "ROLLED_BACK_GUARD_VIOLATION"
                    cand_energy = float(cfg.maximum_pain_energy)
                    delta_e = float(cfg.maximum_pain_energy)
                elif not sub_accepted:
                    status = "ROLLED_BACK_SUBSTAGE_REJECTED"
                    cand_energy = float(scored["candidate_anse_energy"])
                    delta_e = float(scored["delta_e"])
                else:
                    status = "ROLLED_BACK_ENERGY_NON_IMPROVING"
                    cand_energy = float(scored["candidate_anse_energy"])
                    delta_e = float(scored["delta_e"])

            sub_sum = dict(stage_out["sub_summary"])
            if eval_err is not None:
                sub_sum["eval_error"] = eval_err
            entry = _make_stage_merkle_entry(
                stage_index=idx,
                stage_name=stage_name,
                cfg=cfg,
                status=status,
                accepted=accepted,
                skipped=False,
                rolled_back=rolled_back,
                pre_sha=pre_sha,
                cand_sha=cand_sha,
                final_sha=final_sha,
                nll_mean=out_nll,
                ece_value=out_ece,
                general_drift_ratio=out_gen_r,
                worst_domain_drift_ratio=out_worst_r,
                guard_ok=out_guard_ok,
                parent_energy=pre_stage_energy,
                cand_energy=cand_energy,
                final_energy=current_anse_energy,
                delta_e=delta_e,
                sub_receipt_sha=stage_out["sub_receipt_sha256"],
                sub_summary=sub_sum,
                prev_merkle_sha=prev_merkle_sha,
            )
            stage_ledger.append(entry)
            prev_merkle_sha = str(entry["stage_merkle_sha256"])

        final_sha = compute_state_sha256(ratchet_model)
        accepted_stages = [e["stage_name"] for e in stage_ledger if e["accepted"]]
        rejected_stages = [e["stage_name"] for e in stage_ledger if (not e["accepted"] and not e["skipped"])]
        skipped_stages = [e["stage_name"] for e in stage_ledger if e["skipped"]]

        if not accepted_stages:
            _restore_state_dict_bitwise(ratchet_model, base_snapshot)
            final_sha = compute_state_sha256(ratchet_model)
            if final_sha != base_sha:  # pragma: no cover - defensive bitwise safety check
                raise RuntimeError("Full campaign atomic rollback SHA-256 verification failed")

        final_scored = _score_stage_against_base(current_eval, base_eval, 0.0, cfg)
        stage_digests = [str(e["stage_merkle_sha256"]) for e in stage_ledger]
        merkle_root = compute_merkle_root_sha256(stage_digests)
        any_pain = any(float(e["anse_energy"]) >= cfg.maximum_pain_energy * 0.5 for e in stage_ledger)
        final_campaign_energy = (
            float(cfg.maximum_pain_energy)
            if (not accepted_stages and any_pain)
            else float(current_anse_energy)
        )

        receipt_core: dict[str, Any] = {
            "object": GEN3_CAMPAIGN_RECEIPT_OBJECT,
            "schema": GEN3_ORCHESTRATOR_SCHEMA,
            "campaign_id": cfg.campaign_id,
            "round_id": cfg.round_id,
            "lean4_theorems": dict(LEAN4_STAGE_THEOREM_MAP),
            "stage_order": list(STAGE_ORDER),
            "accepted": len(accepted_stages) > 0,
            "all_stages_rolled_back": len(accepted_stages) == 0,
            "full_rollback_verified": (len(accepted_stages) == 0) and (final_sha == base_sha),
            "accepted_stages": accepted_stages,
            "rejected_stages": rejected_stages,
            "skipped_stages": skipped_stages,
            "base_state_sha256": base_sha,
            "final_state_sha256": final_sha,
            "base_nll_mean": float(base_eval["nll_mean"]),
            "final_nll_mean": float(current_eval["nll_mean"]),
            "cumulative_nll_delta": float(current_eval["nll_mean"]) - float(base_eval["nll_mean"]),
            "base_ece_value": float(base_eval["ece_value"]),
            "final_ece_value": float(current_eval["ece_value"]),
            "general_drift_ratio": float(final_scored["general_drift_ratio"]),
            "general_drift_pct": float(final_scored["general_drift_pct"]),
            "worst_domain_drift_ratio": float(final_scored["worst_domain_drift_ratio"]),
            "worst_domain_drift_pct": float(final_scored["worst_domain_drift_pct"]),
            "guard_ok": bool(final_scored["guard_ok"]),
            "anse_energy": final_campaign_energy,
            "final_anse_energy": float(current_anse_energy),
            "delta_e": float(current_anse_energy),
            "base_domain_nlls": dict(base_domain_nlls),
            "final_domain_nlls": dict(current_eval["domain_nlls"]),
            "wddm_footprint": wddm_footprint,
            "d1_pheromone_directive": d1_pheromone_directive,
            "stage_ledger": stage_ledger,
            "stage_merkle_digests": stage_digests,
            "chain_head_sha256": prev_merkle_sha,
            "merkle_root_sha256": merkle_root,
            "config": asdict(cfg),
        }
        return ratchet_model, {**receipt_core, "receipt_sha256": compute_payload_sha256(receipt_core)}
    finally:
        ratchet_model.train(was_training)


def verify_gen3_campaign_merkle_chain(
    receipt: Mapping[str, Any],
    *,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Audit full SHA-256 Merkle chain, state continuity, rollback, and ANSE descent."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a Mapping, got {type(receipt).__name__}")
    _require_bool(raise_on_error, "raise_on_error")

    violations: list[str] = []
    if receipt.get("object") != GEN3_CAMPAIGN_RECEIPT_OBJECT:
        violations.append("invalid receipt object tag")
    if receipt.get("schema") != GEN3_ORCHESTRATOR_SCHEMA:
        violations.append("invalid receipt schema")

    claimed_receipt_sha = receipt.get("receipt_sha256")
    if not isinstance(claimed_receipt_sha, str) or not _SHA256_HEX_RE.match(claimed_receipt_sha):
        violations.append("malformed receipt_sha256")
    else:
        core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        try:
            if claimed_receipt_sha != compute_payload_sha256(core):
                violations.append("receipt_sha256 mismatch")
        except (TypeError, ValueError):
            violations.append("receipt payload non-serializable")

    ledger = receipt.get("stage_ledger")
    if not isinstance(ledger, Sequence) or isinstance(ledger, (str, bytes)) or not ledger:
        violations.append("stage_ledger must be a non-empty sequence")
        ledger_list: list[Any] = []
    else:
        ledger_list = list(ledger)

    prev_merkle = GENESIS_MERKLE_SHA256
    prev_state_sha = receipt.get("base_state_sha256")
    prev_energy = 0.0
    max_gen_cap = float(receipt.get("config", {}).get("max_general_drift_ratio", DEFAULT_DOMAIN_GUARD_MAX_RATIO))
    digests: list[str] = []

    for idx, entry in enumerate(ledger_list):
        lbl = f"stage[{idx}]"
        if not isinstance(entry, Mapping):
            violations.append(f"{lbl}: must be a Mapping")
            continue
        if entry.get("object") != GEN3_STAGE_LEDGER_ENTRY_OBJECT or entry.get("schema") != GEN3_ORCHESTRATOR_SCHEMA:
            violations.append(f"{lbl}: invalid object/schema")
        if entry.get("stage_index") != idx + 1:
            violations.append(f"{lbl}: stage_index mismatch")
        if entry.get("prev_stage_merkle_sha256") != prev_merkle:
            violations.append(f"{lbl}: prev_stage_merkle_sha256 discontinuity")
        if prev_state_sha is not None and entry.get("pre_state_sha256") != prev_state_sha:
            violations.append(f"{lbl}: pre_state_sha256 discontinuity")

        s_sha = entry.get("stage_merkle_sha256")
        if not isinstance(s_sha, str) or not _SHA256_HEX_RE.match(s_sha):
            violations.append(f"{lbl}: malformed stage_merkle_sha256")
        else:
            e_core = {k: v for k, v in entry.items() if k != "stage_merkle_sha256"}
            try:
                if s_sha != compute_payload_sha256(e_core):
                    violations.append(f"{lbl}: stage_merkle_sha256 mismatch")
            except (TypeError, ValueError):
                violations.append(f"{lbl}: stage entry non-serializable")
            digests.append(s_sha)
            prev_merkle = s_sha

        if entry.get("accepted") is True:
            fin_e = float(entry.get("final_anse_energy", 0.0))
            gen_r = entry.get("general_drift_ratio")
            if gen_r is None or float(gen_r) > max_gen_cap + 1e-12:
                violations.append(f"{lbl}: accepted stage exceeded general drift cap")
            if fin_e >= prev_energy:
                violations.append(f"{lbl}: accepted stage did not strictly lower ANSE energy")
            prev_energy = fin_e
        else:
            if entry.get("final_state_sha256") != entry.get("pre_state_sha256"):
                violations.append(f"{lbl}: unaccepted stage altered state SHA-256")
        prev_state_sha = entry.get("final_state_sha256")

    if ledger_list and prev_state_sha != receipt.get("final_state_sha256"):
        violations.append("final_state_sha256 does not match last stage final_state_sha256")

    if digests and len(digests) == len(ledger_list):
        expected_root = compute_merkle_root_sha256(digests)
        if receipt.get("merkle_root_sha256") != expected_root:
            violations.append("merkle_root_sha256 mismatch")
        if receipt.get("chain_head_sha256") != prev_merkle:
            violations.append("chain_head_sha256 mismatch")
        if receipt.get("stage_merkle_digests") != digests:
            violations.append("stage_merkle_digests mismatch")

    d1_dir = receipt.get("d1_pheromone_directive")
    if d1_dir is not None:
        if not isinstance(d1_dir, Mapping) or not verify_queen_pheromone_directive_signature(d1_dir):
            violations.append("d1_pheromone_directive signature invalid")

    valid = len(violations) == 0
    if raise_on_error and not valid:
        raise ValueError(f"Gen-3 Merkle provenance audit failed: {violations[0]}")
    return {
        "valid": valid,
        "violations": violations,
        "stage_count": len(ledger_list),
        "chain_head_sha256": prev_merkle,
        "merkle_root_sha256": receipt.get("merkle_root_sha256"),
    }


def verify_gen3_campaign_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Return True iff the Gen-3 campaign receipt, Merkle chain, and directives verify cleanly."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a Mapping, got {type(receipt).__name__}")
    return bool(verify_gen3_campaign_merkle_chain(receipt, raise_on_error=False)["valid"])


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "GEN3_CAMPAIGN_RECEIPT_OBJECT",
    "GEN3_ORCHESTRATOR_SCHEMA",
    "GEN3_STAGE_LEDGER_ENTRY_OBJECT",
    "GENESIS_MERKLE_SHA256",
    "LEAN4_STAGE_THEOREM_MAP",
    "STAGE_D1_TROPHALLAXIS",
    "STAGE_D2_KRONECKER",
    "STAGE_D3_FUSION",
    "STAGE_D4_MITOSIS_NF4",
    "STAGE_D5_INTAKE",
    "STAGE_DIRECTION_MAP",
    "STAGE_ORDER",
    "STD_8GB_WDDM_CEILING_MIB",
    "Gen3ReineCortexCampaignConfig",
    "compute_merkle_root_sha256",
    "execute_gen3_five_direction_campaign",
    "verify_gen3_campaign_merkle_chain",
    "verify_gen3_campaign_receipt_provenance",
]

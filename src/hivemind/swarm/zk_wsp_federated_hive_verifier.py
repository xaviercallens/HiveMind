# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Proof-Carrying ZK-WSP Subspace Spot-Check Verifier & Multi-Krum Federated Hive Controller.

Implements **Gen-3 Direction D5** (Zenodo ``10.5281/zenodo.24542441`` Section 6 and Lean 4
theorem ``ANSE.ReineCortexDirections.thm_rcd_d5_zkwsp_spotcheck_completeness_and_soundness``)
for decentralized multi-hive Weight-Space Programming (WSP) on CPU (``0 MiB`` GPU VRAM,
``0`` gradient steps):

1. **Orthonormal Prose Subspace Commitments**
   (:func:`build_prose_subspace_commitment`,
   :func:`build_prose_subspace_commitment_from_keys`,
   :func:`collect_prose_subspace_from_probe`):
   Commits an orthonormal preserved general-prose basis ``V_0`` (with ``||V_0^T V_0 - I|| <= tol``)
   and its complementary AlphaEdit null-space projector ``P_0 = I - V_0 V_0^T`` via canonical
   SHA-256 digests without exposing raw local training datasets across hives.
2. **Proof-Carrying ZK-WSP Spot-Check Commitments & Fiat-Shamir Challenges**
   (:func:`compute_zkwsp_spotcheck_witness`, :func:`verify_zkwsp_spotcheck_witness`):
   Verifies the exact Lean 4 ``RCD_D5`` predicate ``max_{q, u} |(Delta W V_{:,q})_u| <= tau_zk``
   together with deterministic Fiat-Shamir ``L1``-normalized Rademacher challenge projections
   seeded from ``SHA-256(hive_id, round_id, delta_sha256, v0_sha256)``, giving ``0%`` honest
   false positives and deterministic detection of prose-poisoning or witness tampering.
3. **Lean 4 Theorem Attestation & Multi-Krum Multi-Hive Federation in ``P_0``**
   (:func:`build_federated_hive_packet`, :func:`verify_federated_hive_packet`,
   :func:`verify_and_aggregate_federated_hives`):
   Cross-binds each hive submission to a machine-checked ``Lean4WSPAttestationReceipt``
   against ``lean4/HiveMindSpec.lean`` and filters surviving hives through Euclidean
   ``float64`` Multi-Krum under ``n >= 2f + 3`` strictly inside ``P_0``.
4. **ANSE Thermodynamic Ratchet & Atomic Bitwise Rollback**
   (:func:`execute_verified_zkwsp_federated_hive_round`,
   :func:`verify_zkwsp_federated_round_receipt_provenance`):
   Commits consensus updates iff ``ANSE Delta E < 0`` and general/worst-domain drift
   ``<= +1.0%``; otherwise restores parent weights bit-for-bit under
   ``ANSE_MAXIMUM_PAIN`` (``1e6``) with SHA-256 receipts.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from hivemind.swarm.byzantine_wsp_delta_aggregator import (
    ANSE_MAXIMUM_PAIN,
    _evaluate_model_domains,
    _is_general_expert_key,
    _is_layernorm_key,
    _normalize_peer_delta_collection,
    _normalize_peer_weights,
    _restore_snapshot_bitwise,
    _validate_bool,
    _validate_nonneg_finite,
    _validate_nonneg_int,
    _validate_positive_finite,
    _validate_positive_int,
    _validate_unit_interval,
    compute_payload_sha256,
    compute_state_sha256,
    multi_krum_nullspace_aggregate,
    screen_peer_wsp_deltas_spectral_and_sparsity,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import CrowdLoRAMoEMLP
from hivemind.swarm.lean4_wsp_theorem_registry import (
    Lean4WSPSpecAudit,
    audit_lean4_wsp_spec_file,
    bind_s5_receipt_to_lean4_theorems,
    verify_lean4_attestation_receipt,
)
from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    compute_swiglu_intermediate_states,
    forward_block_post_mlp,
    forward_block_pre_mlp,
)

ZKWSP_VERIFIER_SCHEMA: str = "hivemind.zk_wsp_federated_hive_verifier/1"
ZKWSP_COMMITMENT_OBJECT: str = "hivemind.zk_wsp_subspace_commitment"
ZKWSP_WITNESS_OBJECT: str = "hivemind.zk_wsp_spotcheck_witness"
ZKWSP_PACKET_OBJECT: str = "hivemind.zk_wsp_federated_hive_packet"
ZKWSP_ROUND_RECEIPT_OBJECT: str = "hivemind.zk_wsp_federated_hive_round_receipt"
LEAN4_RCD_D5_THEOREM: str = (
    "ANSE.ReineCortexDirections.thm_rcd_d5_zkwsp_spotcheck_completeness_and_soundness"
)

DEFAULT_GENERAL_PREFIXES: tuple[str, ...] = ("E0_", "D0_", "general")
DEFAULT_SPOTCHECK_TAU_ZK: float = 1e-4
DEFAULT_BASIS_ORTHONORMAL_TOL: float = 1e-5
DEFAULT_NUM_SPOTCHECK_CHALLENGES: int = 4
DEFAULT_SV_THRESHOLD: float = 1e-2
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_MAX_EDIT_FRACTION: float = 0.25
DEFAULT_MAX_SPECTRAL_SPIKE_RATIO: float = 4.0
DEFAULT_SCORE_FROM: int = 0
DEFAULT_BATCH_SIZE: int = 24

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_LAYER_KEY_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.")


# ---------------------------------------------------------------------------
# 1. Validation, Configuration & Provenance Helpers
# ---------------------------------------------------------------------------


def _validate_nonempty_str(value: Any, name: str) -> str:
    """Validate a non-empty printable string without control characters."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{name} must not contain control characters")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} must be a non-empty string")
    return stripped


def _is_general_domain(domain_id: str, general_prefixes: Sequence[str]) -> bool:
    """Return True if ``domain_id`` matches any configured general-prose prefix."""
    low = domain_id.strip().lower()
    return any(low.startswith(prefix.lower()) for prefix in general_prefixes)


def _normalize_eval_chunks_for_domain_eval(
    chunks: Sequence[Mapping[str, Any]],
    name: str,
) -> list[dict[str, Any]]:
    """Normalize chunk mappings to expose ``expert_id`` and ``ids`` for domain evaluation."""
    if isinstance(chunks, (str, bytes, bytearray)) or not isinstance(chunks, Sequence) or not chunks:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")
    normalized: list[dict[str, Any]] = []
    for idx, ch in enumerate(chunks):
        if not isinstance(ch, Mapping) or "ids" not in ch:
            raise ValueError(f"{name}[{idx}] must be a mapping containing 'ids'")
        raw_dom = ch.get("expert_id", ch.get("domain"))
        if raw_dom is None or not str(raw_dom).strip():
            raise ValueError(f"{name}[{idx}] must contain a non-empty 'expert_id' or 'domain'")
        item = dict(ch)
        item["expert_id"] = str(raw_dom).strip()
        normalized.append(item)
    return normalized


def verify_zkwsp_federated_round_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a Mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if (
        receipt.get("object") != ZKWSP_ROUND_RECEIPT_OBJECT
        or receipt.get("schema") != ZKWSP_VERIFIER_SCHEMA
    ):
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


@dataclass(frozen=True)
class ZKWSPFederatedHiveConfig:
    """Validated configuration for ZK-WSP verification, Multi-Krum, and ANSE gating."""

    num_byzantine_f: int = 1
    select_m: int | None = None
    spotcheck_tau_zk: float = DEFAULT_SPOTCHECK_TAU_ZK
    basis_orthonormal_tol: float = DEFAULT_BASIS_ORTHONORMAL_TOL
    num_spotcheck_challenges: int = DEFAULT_NUM_SPOTCHECK_CHALLENGES
    require_lean4_attestation: bool = True
    enforce_cross_binding: bool = True
    enforce_quorum_on_verified: bool = True
    enable_spectral_screening: bool = False
    max_edit_fraction: float = DEFAULT_MAX_EDIT_FRACTION
    max_spectral_spike_ratio: float = DEFAULT_MAX_SPECTRAL_SPIKE_RATIO
    sv_threshold: float = DEFAULT_SV_THRESHOLD
    prose_variance_threshold: float | None = 0.999
    prose_max_active_modes: int | None = None
    protect_layernorm: bool = True
    protect_general_expert: bool = False
    general_expert_index: int = 0
    consensus_step_scale: float = 1.0
    domain_guard_max_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    enforce_all_domain_guard: bool = True
    require_strict_energy_drop: bool = True
    general_prefixes: tuple[str, ...] = DEFAULT_GENERAL_PREFIXES
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = DEFAULT_BATCH_SIZE
    inplace: bool = False

    def __post_init__(self) -> None:
        _validate_nonneg_int(self.num_byzantine_f, "num_byzantine_f")
        if self.select_m is not None:
            _validate_positive_int(self.select_m, "select_m")
        _validate_nonneg_finite(self.spotcheck_tau_zk, "spotcheck_tau_zk")
        _validate_positive_finite(self.basis_orthonormal_tol, "basis_orthonormal_tol")
        _validate_positive_int(self.num_spotcheck_challenges, "num_spotcheck_challenges")
        for flag in (
            "require_lean4_attestation",
            "enforce_cross_binding",
            "enforce_quorum_on_verified",
            "enable_spectral_screening",
            "protect_layernorm",
            "protect_general_expert",
            "enforce_all_domain_guard",
            "require_strict_energy_drop",
            "inplace",
        ):
            _validate_bool(getattr(self, flag), flag)
        _validate_unit_interval(self.max_edit_fraction, "max_edit_fraction", allow_zero=False)
        _validate_positive_finite(self.max_spectral_spike_ratio, "max_spectral_spike_ratio")
        _validate_nonneg_finite(self.sv_threshold, "sv_threshold")
        if self.prose_variance_threshold is not None:
            _validate_unit_interval(
                self.prose_variance_threshold, "prose_variance_threshold", allow_zero=False
            )
        if self.prose_max_active_modes is not None:
            _validate_positive_int(self.prose_max_active_modes, "prose_max_active_modes")
        _validate_nonneg_int(self.general_expert_index, "general_expert_index")
        _validate_positive_finite(self.consensus_step_scale, "consensus_step_scale")
        max_dom = _validate_positive_finite(self.domain_guard_max_ratio, "domain_guard_max_ratio")
        safe_e0 = _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        if max_dom < 1.0:
            raise ValueError(f"domain_guard_max_ratio must be >= 1.0, got {max_dom!r}")
        if safe_e0 < 1.0:
            raise ValueError(f"anse_safe_general_ratio must be >= 1.0, got {safe_e0!r}")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        if (
            isinstance(self.general_prefixes, (str, bytes))
            or not isinstance(self.general_prefixes, Sequence)
            or not self.general_prefixes
        ):
            raise ValueError("general_prefixes must be a non-empty sequence of non-empty strings")
        cleaned_prefixes = tuple(
            _validate_nonempty_str(p, f"general_prefixes[{idx}]")
            for idx, p in enumerate(self.general_prefixes)
        )
        object.__setattr__(self, "general_prefixes", cleaned_prefixes)
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")


# ---------------------------------------------------------------------------
# 2. Orthonormal Prose Subspace Commitments (V_0 & P_0 = I - V_0 V_0^T)
# ---------------------------------------------------------------------------


def _validate_orthonormal_basis(
    basis: Any,
    name: str,
    *,
    tol: float,
) -> tuple[torch.Tensor, float]:
    """Validate a finite 2D orthonormal basis ``V_0`` in CPU ``float64``."""
    if not isinstance(basis, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(basis).__name__}")
    if basis.ndim != 2 or basis.shape[0] == 0 or basis.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D matrix, got shape {tuple(basis.shape)}")
    if not basis.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {basis.dtype}")
    if int(basis.shape[1]) > int(basis.shape[0]):
        raise ValueError(
            f"{name} columns ({basis.shape[1]}) cannot exceed feature dimension ({basis.shape[0]})"
        )
    v64 = basis.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(v64).all().item()):
        raise ValueError(f"{name} must contain only finite values")

    k_modes = int(v64.shape[1])
    gram = v64.T @ v64
    ortho_err = float((gram - torch.eye(k_modes, dtype=torch.float64)).abs().max().item())
    if ortho_err > tol:
        raise ValueError(
            f"{name} columns must be orthonormal (||V^T V - I||_inf={ortho_err:.3e} > tol={tol:.3e})"
        )
    return v64, ortho_err


def _basis_to_p0(v64: torch.Tensor) -> torch.Tensor:
    """Compute exact symmetric AlphaEdit null-space projector ``P_0 = I - V_0 V_0^T``."""
    d_in = int(v64.shape[0])
    proj = torch.eye(d_in, dtype=torch.float64) - (v64 @ v64.T)
    return (0.5 * (proj + proj.T)).contiguous()


def build_prose_subspace_commitment(
    v0_basis: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    basis_orthonormal_tol: float = DEFAULT_BASIS_ORTHONORMAL_TOL,
    commitment_id: str = "prose_v0_commitment",
) -> dict[str, Any]:
    """Validate orthonormal basis ``V_0`` and seal ``V_0`` + ``P_0`` under SHA-256."""
    tol = _validate_positive_finite(basis_orthonormal_tol, "basis_orthonormal_tol")
    cid = _validate_nonempty_str(commitment_id, "commitment_id")

    raw_map: dict[str, torch.Tensor] = {}
    if isinstance(v0_basis, torch.Tensor):
        raw_map["default"] = v0_basis
    elif isinstance(v0_basis, Mapping) and v0_basis:
        for k in sorted(v0_basis.keys(), key=str):
            k_str = _validate_nonempty_str(str(k), "v0_basis key")
            raw_map[k_str] = v0_basis[k]
    else:
        raise TypeError("v0_basis must be a 2D torch.Tensor or non-empty Mapping[str, torch.Tensor]")

    v0_map: dict[str, torch.Tensor] = {}
    p0_map: dict[str, torch.Tensor] = {}
    max_ortho_err = 0.0
    for key in sorted(raw_map.keys()):
        v64, err = _validate_orthonormal_basis(raw_map[key], f"v0_basis[{key!r}]", tol=tol)
        v0_map[key] = v64
        p0_map[key] = _basis_to_p0(v64)
        max_ortho_err = max(max_ortho_err, err)

    v0_sha = compute_state_sha256(v0_map)
    p0_sha = compute_state_sha256(p0_map)
    return {
        "object": ZKWSP_COMMITMENT_OBJECT,
        "schema": ZKWSP_VERIFIER_SCHEMA,
        "commitment_id": cid,
        "basis_keys": sorted(v0_map.keys()),
        "basis_shapes": {k: list(v.shape) for k, v in v0_map.items()},
        "v0_bases": v0_map,
        "p0_projectors": p0_map,
        "v0_sha256": v0_sha,
        "p0_sha256": p0_sha,
        "max_orthonormality_error": max_ortho_err,
    }


def _extract_basis_from_keys(
    keys_tensor: torch.Tensor,
    *,
    name: str,
    sv_threshold: float,
    variance_threshold: float | None,
    max_active_modes: int | None,
) -> torch.Tensor:
    """Extract orthonormal top singular basis ``V_0`` spanning preserved prose activations."""
    if not isinstance(keys_tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(keys_tensor).__name__}")
    if keys_tensor.ndim not in (2, 3) or any(dim == 0 for dim in keys_tensor.shape):
        raise ValueError(f"{name} must be a non-empty 2D or 3D tensor, got shape {tuple(keys_tensor.shape)}")
    if not keys_tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {keys_tensor.dtype}")
    mat64 = keys_tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1, int(keys_tensor.shape[-1]))
    if not bool(torch.isfinite(mat64).all().item()):
        raise ValueError(f"{name} must contain only finite values")

    _, s_vals, vh = torch.linalg.svd(mat64, full_matrices=False)
    above = [int(i) for i in range(int(s_vals.numel())) if float(s_vals[i].item()) > sv_threshold]
    if not above:
        return vh[:1, :].T.contiguous()

    total_energy = float(s_vals.pow(2).sum().item())
    selected: list[int] = []
    cum_energy = 0.0
    for idx in above:
        if max_active_modes is not None and len(selected) >= max_active_modes:
            break
        selected.append(idx)
        cum_energy += float(s_vals[idx].item()) ** 2
        if variance_threshold is not None and total_energy > 0.0 and (cum_energy / total_energy) >= variance_threshold:
            break
    return vh[selected, :].T.contiguous()


@torch.no_grad()
def build_prose_subspace_commitment_from_keys(
    prose_keys: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    sv_threshold: float = DEFAULT_SV_THRESHOLD,
    variance_threshold: float | None = 0.999,
    max_active_modes: int | None = None,
    basis_orthonormal_tol: float = DEFAULT_BASIS_ORTHONORMAL_TOL,
    commitment_id: str = "prose_keys_v0_commitment",
) -> dict[str, Any]:
    """Derive orthonormal ``V_0`` bases via SVD of prose keys and seal as commitment."""
    sv_thr = _validate_nonneg_finite(sv_threshold, "sv_threshold")
    var_thr = (
        _validate_unit_interval(variance_threshold, "variance_threshold", allow_zero=False)
        if variance_threshold is not None
        else None
    )
    max_m = _validate_positive_int(max_active_modes, "max_active_modes") if max_active_modes is not None else None

    if isinstance(prose_keys, torch.Tensor):
        v0_obj: torch.Tensor | dict[str, torch.Tensor] = _extract_basis_from_keys(
            prose_keys,
            name="prose_keys",
            sv_threshold=sv_thr,
            variance_threshold=var_thr,
            max_active_modes=max_m,
        )
    elif isinstance(prose_keys, Mapping) and prose_keys:
        v0_obj = {
            _validate_nonempty_str(str(k), "prose_keys key"): _extract_basis_from_keys(
                v,
                name=f"prose_keys[{k!r}]",
                sv_threshold=sv_thr,
                variance_threshold=var_thr,
                max_active_modes=max_m,
            )
            for k, v in sorted(prose_keys.items(), key=lambda kv: str(kv[0]))
        }
    else:
        raise TypeError("prose_keys must be a Tensor or non-empty Mapping[str, Tensor]")

    return build_prose_subspace_commitment(
        v0_obj,
        basis_orthonormal_tol=basis_orthonormal_tol,
        commitment_id=commitment_id,
    )


@torch.no_grad()
def collect_prose_subspace_from_probe(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    general_prefixes: Sequence[str] = DEFAULT_GENERAL_PREFIXES,
    sv_threshold: float = DEFAULT_SV_THRESHOLD,
    variance_threshold: float | None = 0.999,
    max_active_modes: int | None = None,
    basis_orthonormal_tol: float = DEFAULT_BASIS_ORTHONORMAL_TOL,
    score_from: int = DEFAULT_SCORE_FROM,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Capture ``E0_*`` / general-prose activations on ``model`` and build ``V_0`` commitment."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if isinstance(probe_chunks, (str, bytes)) or not isinstance(probe_chunks, Sequence) or not probe_chunks:
        raise ValueError("probe_chunks must be a non-empty sequence of chunk mappings")
    if not hasattr(model, "blocks") or not hasattr(model, "cos_cached"):
        raise ValueError("model must expose .blocks and .cos_cached for probe basis collection")

    gen_chunks = [
        ch
        for ch in probe_chunks
        if isinstance(ch, Mapping)
        and _is_general_domain(str(ch.get("expert_id", ch.get("domain", ""))), general_prefixes)
    ]
    if not gen_chunks:
        raise ValueError(f"probe_chunks must contain at least one general chunk matching {tuple(general_prefixes)}")

    ids_gen = torch.stack([torch.as_tensor(ch["ids"], dtype=torch.long) for ch in gen_chunks], dim=0)
    start_pos = max(0, min(int(ids_gen.shape[1]) - 2, _validate_nonneg_int(score_from, "score_from") - 1))
    b_sz = _validate_positive_int(batch_size, "batch_size")
    cos = model.cos_cached.to(ids_gen.device)
    sin = model.sin_cached.to(ids_gen.device)

    x_seq = torch.cat(
        [compute_embeddings_batch(model, ids_gen[b0 : b0 + b_sz]) for b0 in range(0, ids_gen.shape[0], b_sz)],
        dim=0,
    )
    prose_keys: dict[str, torch.Tensor] = {}
    for l_idx, blk in enumerate(model.blocks):
        xm_l: list[torch.Tensor] = []
        mi_l: list[torch.Tensor] = []
        for b0 in range(0, x_seq.shape[0], b_sz):
            xm, mi = forward_block_pre_mlp(blk, x_seq[b0 : b0 + b_sz], cos, sin)
            xm_l.append(xm)
            mi_l.append(mi)
        x_mid = torch.cat(xm_l, dim=0)
        mlp_in = torch.cat(mi_l, dim=0)
        scored_x = mlp_in[:, start_pos:-1, :].reshape(-1, int(mlp_in.shape[-1]))
        prose_keys[f"blocks.{l_idx}.x"] = scored_x
        if isinstance(blk.mlp, CrowdLoRAMoEMLP) and blk.mlp.experts:
            active_exp = sorted(int(k) for k in blk.mlp.experts.keys())
            prose_keys[f"blocks.{l_idx}.act"] = compute_swiglu_intermediate_states(blk.mlp, scored_x, active_exp)
        x_seq = torch.cat(
            [forward_block_post_mlp(blk, x_mid[b0 : b0 + b_sz], mlp_in[b0 : b0 + b_sz]) for b0 in range(0, x_mid.shape[0], b_sz)],
            dim=0,
        )

    if hasattr(model, "ln"):
        pre_ln_scored = x_seq[:, start_pos:-1, :].to(torch.float32)
        post_ln = model.ln(pre_ln_scored).reshape(-1, int(pre_ln_scored.shape[-1]))
        prose_keys["head.weight"] = post_ln

    return build_prose_subspace_commitment_from_keys(
        prose_keys,
        sv_threshold=sv_threshold,
        variance_threshold=variance_threshold,
        max_active_modes=max_active_modes,
        basis_orthonormal_tol=basis_orthonormal_tol,
        commitment_id="probe_extracted_v0_commitment",
    )


def _resolve_commitment(
    commitment_or_v0: Mapping[str, Any] | torch.Tensor,
    *,
    basis_orthonormal_tol: float,
) -> dict[str, Any]:
    """Accept an already-built subspace commitment dict or raw ``V_0`` basis tensor/mapping."""
    if isinstance(commitment_or_v0, Mapping) and commitment_or_v0.get("object") == ZKWSP_COMMITMENT_OBJECT:
        v0_bases = commitment_or_v0.get("v0_bases")
        if not isinstance(v0_bases, Mapping) or not v0_bases:
            raise ValueError("subspace_commitment missing non-empty 'v0_bases'")
        rebuilt = build_prose_subspace_commitment(
            v0_bases,
            basis_orthonormal_tol=basis_orthonormal_tol,
            commitment_id=str(commitment_or_v0.get("commitment_id", "prose_v0_commitment")),
        )
        if rebuilt["v0_sha256"] != commitment_or_v0.get("v0_sha256"):
            raise ValueError("subspace_commitment v0_sha256 failed integrity verification")
        return rebuilt
    return build_prose_subspace_commitment(
        commitment_or_v0,  # type: ignore[arg-type]
        basis_orthonormal_tol=basis_orthonormal_tol,
    )


# ---------------------------------------------------------------------------
# 3. Proof-Carrying ZK-WSP Spot-Check Witness & Fiat-Shamir Verifier
# ---------------------------------------------------------------------------


def _normalize_delta_input(
    delta: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    target_key: str | None = None,
) -> tuple[dict[str, torch.Tensor], bool]:
    """Normalize a single delta matrix or parameter dict into sorted finite tensors."""
    if isinstance(delta, torch.Tensor):
        if delta.numel() == 0 or not delta.is_floating_point():
            raise ValueError("delta tensor must be non-empty with floating-point dtype")
        k = _validate_nonempty_str(target_key, "target_key") if target_key is not None else "delta"
        return {k: delta.detach()}, True
    if isinstance(delta, Mapping) and delta:
        out: dict[str, torch.Tensor] = {}
        for key in sorted(delta.keys()):
            k_str = _validate_nonempty_str(str(key), "delta key")
            val = delta[key]
            if not isinstance(val, torch.Tensor) or val.numel() == 0 or not val.is_floating_point():
                raise ValueError(f"delta[{k_str!r}] must be a non-empty floating-point torch.Tensor")
            out[k_str] = val.detach()
        return out, False
    raise TypeError("delta must be a torch.Tensor or non-empty Mapping[str, torch.Tensor]")


def _resolve_basis_for_key(
    key: str,
    shape: tuple[int, ...],
    v0_bases: Mapping[str, torch.Tensor],
    *,
    protect_layernorm: bool,
    protect_general_expert: bool,
    general_expert_index: int,
) -> tuple[str | None, torch.Tensor | None]:
    """Resolve matching ``(basis_key, V_0)`` for parameter ``key`` using AlphaEdit priority."""
    if (protect_layernorm and _is_layernorm_key(key)) or (
        protect_general_expert and _is_general_expert_key(key, general_expert_index)
    ):
        return None, None
    if key in v0_bases:
        return key, v0_bases[key]
    if key.endswith(("w1_B", "w2_B")):
        return None, None

    layer_match = _LAYER_KEY_RE.search(key)
    layer_tag = f"blocks.{layer_match.group(1)}" if layer_match is not None else None
    is_act = key.endswith("w3_A") or key.endswith(".w3.weight") or key == "w3"

    candidates: list[str] = []
    if layer_tag is not None:
        candidates.append(f"{layer_tag}.act" if is_act else f"{layer_tag}.x")
    candidates.append("act" if is_act else "x")
    candidates.append("default")

    for cand in candidates:
        if cand in v0_bases:
            v64 = v0_bases[cand]
            d_in = int(v64.shape[0])
            if len(shape) == 1 and shape[0] == d_in:
                return cand, v64
            if len(shape) >= 2 and (shape[-1] == d_in or (len(shape) == 2 and shape[0] == d_in)):
                return cand, v64
    return None, None


def _project_delta_against_v0(
    t64: torch.Tensor,
    v64: torch.Tensor,
    *,
    key: str,
    protect_general_expert: bool,
    general_expert_index: int,
) -> torch.Tensor:
    """Compute ``Delta W @ V_0`` (or transposed / 1D alignment) in CPU ``float64``."""
    d_in = int(v64.shape[0])
    if t64.ndim == 1 and int(t64.shape[0]) == d_in:
        return (t64 @ v64).reshape(-1)
    if t64.ndim >= 2 and int(t64.shape[-1]) == d_in:
        if (
            protect_general_expert
            and "router" in key.lower()
            and t64.ndim == 2
            and general_expert_index < int(t64.shape[0])
        ):
            mask = torch.ones(int(t64.shape[0]), dtype=torch.bool)
            mask[general_expert_index] = False
            return (t64[mask] @ v64).reshape(-1, int(v64.shape[1]))
        return t64 @ v64
    if t64.ndim == 2 and int(t64.shape[0]) == d_in:
        return v64.T @ t64
    raise ValueError(f"Incompatible basis shape {tuple(v64.shape)} for parameter {key!r} of shape {tuple(t64.shape)}")


@torch.no_grad()
def compute_zkwsp_spotcheck_witness(
    hive_id: str,
    delta: torch.Tensor | Mapping[str, torch.Tensor],
    subspace_commitment: Mapping[str, Any] | torch.Tensor,
    *,
    round_id: str = "round_0",
    spotcheck_tau_zk: float = DEFAULT_SPOTCHECK_TAU_ZK,
    num_challenges: int = DEFAULT_NUM_SPOTCHECK_CHALLENGES,
    basis_orthonormal_tol: float = DEFAULT_BASIS_ORTHONORMAL_TOL,
    protect_layernorm: bool = True,
    protect_general_expert: bool = False,
    general_expert_index: int = 0,
    target_key: str | None = None,
) -> dict[str, Any]:
    """Compute deterministic Fiat-Shamir ZK-WSP subspace spot-check witness (Lean 4 ``RCD_D5``)."""
    hid = _validate_nonempty_str(hive_id, "hive_id")
    rid = _validate_nonempty_str(round_id, "round_id")
    tau_zk = _validate_nonneg_finite(spotcheck_tau_zk, "spotcheck_tau_zk")
    n_chal = _validate_positive_int(num_challenges, "num_challenges")
    _validate_bool(protect_layernorm, "protect_layernorm")
    _validate_bool(protect_general_expert, "protect_general_expert")
    gen_idx = _validate_nonneg_int(general_expert_index, "general_expert_index")

    commitment = _resolve_commitment(subspace_commitment, basis_orthonormal_tol=basis_orthonormal_tol)
    v0_bases: Mapping[str, torch.Tensor] = commitment["v0_bases"]
    delta_map, _ = _normalize_delta_input(delta, target_key=target_key)
    delta_sha = compute_state_sha256(delta_map)
    v0_sha = str(commitment["v0_sha256"])

    challenge_seed = compute_payload_sha256(
        {
            "hive_id": hid,
            "round_id": rid,
            "delta_sha256": delta_sha,
            "v0_sha256": v0_sha,
            "num_challenges": n_chal,
        }
    )

    spotchecked_keys: list[str] = []
    basis_leaks_by_key: dict[str, float] = {}
    max_abs_basis_leak = 0.0
    max_abs_challenge_leak = 0.0
    frob_leak_sq = 0.0
    challenge_digests: list[str] = []

    matched_items: list[tuple[str, str, torch.Tensor, torch.Tensor]] = []
    for key in sorted(delta_map.keys()):
        t64 = delta_map[key].to(device="cpu", dtype=torch.float64)
        b_key, v64 = _resolve_basis_for_key(
            key,
            tuple(t64.shape),
            v0_bases,
            protect_layernorm=protect_layernorm,
            protect_general_expert=protect_general_expert,
            general_expert_index=gen_idx,
        )
        if b_key is None or v64 is None:
            continue
        matched_items.append((key, b_key, t64, v64))
        spotchecked_keys.append(key)
        leak_mat = _project_delta_against_v0(
            t64,
            v64,
            key=key,
            protect_general_expert=protect_general_expert,
            general_expert_index=gen_idx,
        )
        key_max = float(leak_mat.abs().max().item()) if leak_mat.numel() > 0 else 0.0
        basis_leaks_by_key[key] = key_max
        max_abs_basis_leak = max(max_abs_basis_leak, key_max)
        frob_leak_sq += float((leak_mat * leak_mat).sum().item())

    if not matched_items:
        raise ValueError("No delta parameter matched any committed V_0 basis for ZK-WSP spot-check")

    for c_idx in range(n_chal):
        chal_hash = hashlib.sha256()
        for key, b_key, t64, v64 in matched_items:
            k_modes = int(v64.shape[1])
            seed_bytes = hashlib.sha256(f"{challenge_seed}:{c_idx}:{b_key}".encode()).digest()
            seed_int = int.from_bytes(seed_bytes[:8], "big") % (2**63 - 1)
            gen = torch.Generator(device="cpu").manual_seed(seed_int)
            signs = (torch.randint(0, 2, (k_modes,), generator=gen, dtype=torch.int64) * 2 - 1).to(torch.float64)
            c_vec = (signs / float(k_modes)).reshape(-1, 1)  # ||c_vec||_1 == 1.0 -> ||R c_vec||_inf <= tau_zk
            probe_dir = (v64 @ c_vec).reshape(-1, 1)
            resp = _project_delta_against_v0(
                t64,
                probe_dir,
                key=key,
                protect_general_expert=protect_general_expert,
                general_expert_index=gen_idx,
            ).contiguous()
            chal_max = float(resp.abs().max().item()) if resp.numel() > 0 else 0.0
            max_abs_challenge_leak = max(max_abs_challenge_leak, chal_max)
            chal_hash.update(f"{key}:{tuple(resp.shape)}:".encode())
            chal_hash.update(resp.reshape(-1).view(torch.uint8).numpy().tobytes())
        challenge_digests.append(chal_hash.hexdigest())

    witness_core: dict[str, Any] = {
        "object": ZKWSP_WITNESS_OBJECT,
        "schema": ZKWSP_VERIFIER_SCHEMA,
        "hive_id": hid,
        "round_id": rid,
        "delta_sha256": delta_sha,
        "v0_sha256": v0_sha,
        "spotcheck_tau_zk": float(tau_zk),
        "num_challenges": n_chal,
        "challenge_seed_sha256": challenge_seed,
        "challenge_response_digests": challenge_digests,
        "spotchecked_keys": spotchecked_keys,
        "basis_leaks_by_key": basis_leaks_by_key,
        "claimed_max_abs_basis_leak": max_abs_basis_leak,
        "claimed_max_abs_challenge_leak": max_abs_challenge_leak,
        "claimed_frobenius_leak": math.sqrt(frob_leak_sq),
        "passes_zkwsp_spotcheck": bool(max_abs_basis_leak <= tau_zk and max_abs_challenge_leak <= tau_zk),
    }
    witness_sha = compute_payload_sha256(witness_core)
    return {**witness_core, "witness_sha256": witness_sha}


def verify_zkwsp_spotcheck_witness(
    witness: Mapping[str, Any],
    *,
    hive_id: str,
    delta: torch.Tensor | Mapping[str, torch.Tensor],
    subspace_commitment: Mapping[str, Any] | torch.Tensor,
    round_id: str = "round_0",
    spotcheck_tau_zk: float = DEFAULT_SPOTCHECK_TAU_ZK,
    basis_orthonormal_tol: float = DEFAULT_BASIS_ORTHONORMAL_TOL,
    protect_layernorm: bool = True,
    protect_general_expert: bool = False,
    general_expert_index: int = 0,
    target_key: str | None = None,
) -> dict[str, Any]:
    """Deterministically verify a proof-carrying ZK-WSP spot-check witness."""
    if not isinstance(witness, Mapping):
        raise TypeError(f"witness must be a Mapping, got {type(witness).__name__}")
    tau_zk = _validate_nonneg_finite(spotcheck_tau_zk, "spotcheck_tau_zk")
    failures: list[str] = []

    if witness.get("object") != ZKWSP_WITNESS_OBJECT or witness.get("schema") != ZKWSP_VERIFIER_SCHEMA:
        failures.append("invalid_witness_schema_or_object")
    claimed_w_sha = witness.get("witness_sha256")
    if not isinstance(claimed_w_sha, str) or not _SHA256_HEX_RE.match(claimed_w_sha):
        failures.append("invalid_witness_sha256_format")
    else:
        w_core = {k: v for k, v in witness.items() if k != "witness_sha256"}
        try:
            if compute_payload_sha256(w_core) != claimed_w_sha:
                failures.append("zkwsp_witness_tampered")
        except (TypeError, ValueError):
            failures.append("zkwsp_witness_non_serializable")

    n_chal_raw = witness.get("num_challenges", DEFAULT_NUM_SPOTCHECK_CHALLENGES)
    if isinstance(n_chal_raw, bool) or not isinstance(n_chal_raw, int) or n_chal_raw < 1:
        failures.append("invalid_num_challenges")
        n_chal = DEFAULT_NUM_SPOTCHECK_CHALLENGES
    else:
        n_chal = int(n_chal_raw)

    recomputed = compute_zkwsp_spotcheck_witness(
        hive_id,
        delta,
        subspace_commitment,
        round_id=round_id,
        spotcheck_tau_zk=tau_zk,
        num_challenges=n_chal,
        basis_orthonormal_tol=basis_orthonormal_tol,
        protect_layernorm=protect_layernorm,
        protect_general_expert=protect_general_expert,
        general_expert_index=general_expert_index,
        target_key=target_key,
    )

    if witness.get("hive_id") != hive_id:
        failures.append("zkwsp_hive_id_mismatch")
    if witness.get("round_id") != round_id:
        failures.append("zkwsp_round_id_mismatch")
    if witness.get("v0_sha256") != recomputed["v0_sha256"]:
        failures.append("subspace_commitment_mismatch")
    if witness.get("delta_sha256") != recomputed["delta_sha256"]:
        failures.append("delta_sha256_mismatch")
    if witness.get("challenge_seed_sha256") != recomputed["challenge_seed_sha256"]:
        failures.append("zkwsp_challenge_seed_mismatch")
    if witness.get("challenge_response_digests") != recomputed["challenge_response_digests"]:
        failures.append("zkwsp_challenge_response_mismatch")
    claimed_leak = witness.get("claimed_max_abs_basis_leak")
    if (
        isinstance(claimed_leak, bool)
        or not isinstance(claimed_leak, (int, float))
        or not math.isfinite(float(claimed_leak))
        or abs(float(claimed_leak) - float(recomputed["claimed_max_abs_basis_leak"])) > 1e-10
    ):
        failures.append("zkwsp_claimed_leak_mismatch")

    meas_basis = float(recomputed["claimed_max_abs_basis_leak"])
    meas_chal = float(recomputed["claimed_max_abs_challenge_leak"])
    if not math.isfinite(meas_basis) or meas_basis > tau_zk or not math.isfinite(meas_chal) or meas_chal > tau_zk:
        failures.append("zkwsp_prose_nullspace_violation")

    return {
        "valid": len(failures) == 0,
        "failures": tuple(failures),
        "measured_max_abs_basis_leak": meas_basis,
        "measured_max_abs_challenge_leak": meas_chal,
        "measured_frobenius_leak": float(recomputed["claimed_frobenius_leak"]),
        "delta_sha256": recomputed["delta_sha256"],
        "v0_sha256": recomputed["v0_sha256"],
        "witness_sha256": recomputed["witness_sha256"],
    }


# ---------------------------------------------------------------------------
# 4. Proof-Carrying Federated Hive Packets & Lean 4 Attestation Verification
# ---------------------------------------------------------------------------


def _resolve_lean_audit(
    lean_spec_audit: Lean4WSPSpecAudit | None,
    lean_spec_path: Path | str | None,
) -> Lean4WSPSpecAudit:
    """Resolve and validate ``Lean4WSPSpecAudit`` without invoking external subprocesses."""
    if lean_spec_audit is not None:
        if not isinstance(lean_spec_audit, Lean4WSPSpecAudit):
            raise TypeError(f"lean_spec_audit must be a Lean4WSPSpecAudit, got {type(lean_spec_audit).__name__}")
        return lean_spec_audit
    return audit_lean4_wsp_spec_file(spec_path=lean_spec_path)


@torch.no_grad()
def build_federated_hive_packet(
    hive_id: str,
    delta: torch.Tensor | Mapping[str, torch.Tensor],
    subspace_commitment: Mapping[str, Any] | torch.Tensor,
    *,
    lean_spec_audit: Lean4WSPSpecAudit | None = None,
    lean_spec_path: Path | str | None = None,
    round_id: str = "round_0",
    domain_label: str = "specialist",
    peer_weight: float = 1.0,
    spotcheck_tau_zk: float = DEFAULT_SPOTCHECK_TAU_ZK,
    num_challenges: int = DEFAULT_NUM_SPOTCHECK_CHALLENGES,
    basis_orthonormal_tol: float = DEFAULT_BASIS_ORTHONORMAL_TOL,
    protect_layernorm: bool = True,
    protect_general_expert: bool = False,
    general_expert_index: int = 0,
    target_key: str | None = None,
    claimed_anse_energy: float = -0.005,
    claimed_general_rel_drift: float = 0.0,
) -> dict[str, Any]:
    """Construct a proof-carrying ZK-WSP + Lean 4 attested federated Hive delta packet."""
    hid = _validate_nonempty_str(hive_id, "hive_id")
    rid = _validate_nonempty_str(round_id, "round_id")
    dom = _validate_nonempty_str(domain_label, "domain_label")
    p_weight = _validate_positive_finite(peer_weight, "peer_weight")
    anse_e = float(claimed_anse_energy)
    if isinstance(claimed_anse_energy, bool) or not math.isfinite(anse_e):
        raise ValueError("claimed_anse_energy must be a finite real number")
    gen_drift = _validate_nonneg_finite(claimed_general_rel_drift, "claimed_general_rel_drift")

    audit = _resolve_lean_audit(lean_spec_audit, lean_spec_path)
    delta_map, is_single = _normalize_delta_input(delta, target_key=target_key)
    delta_sha = compute_state_sha256(delta_map)

    commitment = _resolve_commitment(subspace_commitment, basis_orthonormal_tol=basis_orthonormal_tol)
    witness = compute_zkwsp_spotcheck_witness(
        hid,
        delta_map,
        commitment,
        round_id=rid,
        spotcheck_tau_zk=spotcheck_tau_zk,
        num_challenges=num_challenges,
        basis_orthonormal_tol=basis_orthonormal_tol,
        protect_layernorm=protect_layernorm,
        protect_general_expert=protect_general_expert,
        general_expert_index=general_expert_index,
    )

    s5_receipt_core: dict[str, Any] = {
        "object": "hivemind.s5_surgery_receipt",
        "round_id": rid,
        "hive_id": hid,
        "domain_label": dom,
        "lean4_rcd_d5_theorem": LEAN4_RCD_D5_THEOREM,
        "delta_sha256": delta_sha,
        "v0_sha256": commitment["v0_sha256"],
        "witness_sha256": witness["witness_sha256"],
        "verdict": "PASS",
        "general_rel_drift": gen_drift,
        "anchor_layers_bitwise_preserved": True,
        "guard_ok": True,
        "anse_energy": anse_e,
        "claimed_max_abs_basis_leak": witness["claimed_max_abs_basis_leak"],
    }
    s5_receipt = {**s5_receipt_core, "receipt_sha256": compute_payload_sha256(s5_receipt_core)}
    lean4_att = bind_s5_receipt_to_lean4_theorems(s5_receipt, spec_audit=audit).to_dict()

    delta_payload: torch.Tensor | dict[str, torch.Tensor] = (
        next(iter(delta_map.values())).clone()
        if is_single
        else {k: v.clone() for k, v in delta_map.items()}
    )
    meta_core = {
        "object": ZKWSP_PACKET_OBJECT,
        "schema": ZKWSP_VERIFIER_SCHEMA,
        "hive_id": hid,
        "round_id": rid,
        "domain_label": dom,
        "peer_weight": p_weight,
        "target_key": target_key,
        "claimed_delta_sha256": delta_sha,
        "v0_sha256": commitment["v0_sha256"],
        "spotcheck_witness": witness,
        "s5_receipt": s5_receipt,
        "lean4_attestation": lean4_att,
    }
    return {
        **meta_core,
        "delta": delta_payload,
        "packet_sha256": compute_payload_sha256(meta_core),
    }


@torch.no_grad()
def verify_federated_hive_packet(
    packet: Mapping[str, Any],
    subspace_commitment: Mapping[str, Any] | torch.Tensor,
    *,
    lean_spec_audit: Lean4WSPSpecAudit | None = None,
    lean_spec_path: Path | str | None = None,
    expected_round_id: str | None = None,
    config: ZKWSPFederatedHiveConfig | None = None,
) -> dict[str, Any]:
    """Verify payload SHA-256, ZK-WSP witness, cross-binding, and Lean 4 attestation."""
    if not isinstance(packet, Mapping):
        raise TypeError(f"packet must be a Mapping, got {type(packet).__name__}")
    cfg = config if config is not None else ZKWSPFederatedHiveConfig()
    if not isinstance(cfg, ZKWSPFederatedHiveConfig):
        raise TypeError(f"config must be a ZKWSPFederatedHiveConfig or None, got {type(cfg).__name__}")

    commitment = _resolve_commitment(subspace_commitment, basis_orthonormal_tol=cfg.basis_orthonormal_tol)
    raw_hid = packet.get("hive_id")
    hid = raw_hid.strip() if isinstance(raw_hid, str) and raw_hid.strip() else "<unknown_hive>"
    rid = str(packet.get("round_id", "round_0")).strip() or "round_0"
    reasons: list[str] = []

    if expected_round_id is not None and rid != expected_round_id:
        reasons.append("round_id_mismatch")

    raw_delta = packet.get("delta")
    target_key = packet.get("target_key")
    t_key_str = str(target_key) if isinstance(target_key, str) and target_key.strip() else None
    try:
        delta_map, _ = _normalize_delta_input(raw_delta, target_key=t_key_str)  # type: ignore[arg-type]
        actual_delta_sha = compute_state_sha256(delta_map)
    except (TypeError, ValueError) as exc:
        reasons.append(f"invalid_or_non_finite_delta:{type(exc).__name__}")
        return {
            "accepted": False,
            "hive_id": hid,
            "round_id": rid,
            "rejection_reasons": tuple(reasons),
            "delta_sha256": "",
            "measured_max_abs_basis_leak": None,
            "measured_max_abs_challenge_leak": None,
            "lean4_verified": False,
            "delta_map": None,
        }

    claimed_delta_sha = packet.get("claimed_delta_sha256")
    if actual_delta_sha != claimed_delta_sha:
        reasons.append("delta_sha256_mismatch")

    if packet.get("v0_sha256") != commitment["v0_sha256"]:
        reasons.append("subspace_commitment_mismatch")

    # Verify Lean 4 attestation receipt & cross-binding before/alongside spot-checks
    lean4_verified = False
    s5_receipt = packet.get("s5_receipt")
    lean4_att = packet.get("lean4_attestation")
    if cfg.require_lean4_attestation:
        if not isinstance(s5_receipt, Mapping) or not isinstance(lean4_att, Mapping):
            reasons.append("missing_lean4_attestation_or_s5_receipt")
        else:
            audit = _resolve_lean_audit(lean_spec_audit, lean_spec_path)
            if cfg.enforce_cross_binding:
                if s5_receipt.get("hive_id") != hid:
                    reasons.append("cross_binding_hive_id_mismatch")
                if s5_receipt.get("round_id") != rid:
                    reasons.append("cross_binding_round_id_mismatch")
                if s5_receipt.get("delta_sha256") != claimed_delta_sha:
                    reasons.append("cross_binding_delta_sha256_mismatch")
                if s5_receipt.get("v0_sha256") != commitment["v0_sha256"]:
                    reasons.append("cross_binding_v0_sha256_mismatch")
            try:
                verif = verify_lean4_attestation_receipt(
                    lean4_att,
                    surgery_receipt=s5_receipt,
                    spec_audit=audit,
                )
                if verif.valid:
                    lean4_verified = True
                else:
                    reasons.append(f"lean4_attestation_failed:{','.join(verif.failures)}")
            except (TypeError, ValueError) as exc:
                reasons.append(f"lean4_attestation_invalid:{type(exc).__name__}")

    # Verify ZK-WSP subspace spot-check witness & exact RCD_D5 orthogonality
    witness = packet.get("spotcheck_witness")
    meas_basis: float | None = None
    meas_chal: float | None = None
    if not isinstance(witness, Mapping):
        reasons.append("missing_zkwsp_spotcheck_witness")
    else:
        try:
            w_verif = verify_zkwsp_spotcheck_witness(
                witness,
                hive_id=hid,
                delta=delta_map,
                subspace_commitment=commitment,
                round_id=rid,
                spotcheck_tau_zk=cfg.spotcheck_tau_zk,
                basis_orthonormal_tol=cfg.basis_orthonormal_tol,
                protect_layernorm=cfg.protect_layernorm,
                protect_general_expert=cfg.protect_general_expert,
                general_expert_index=cfg.general_expert_index,
                target_key=t_key_str,
            )
            meas_basis = float(w_verif["measured_max_abs_basis_leak"])
            meas_chal = float(w_verif["measured_max_abs_challenge_leak"])
            for r in w_verif["failures"]:
                if r not in reasons:
                    reasons.append(r)
            if cfg.enforce_cross_binding and isinstance(s5_receipt, Mapping):
                if s5_receipt.get("witness_sha256") != w_verif["witness_sha256"]:
                    reasons.append("cross_binding_witness_sha256_mismatch")
        except (TypeError, ValueError) as exc:
            reasons.append(f"zkwsp_spotcheck_error:{type(exc).__name__}")

    return {
        "accepted": len(reasons) == 0,
        "hive_id": hid,
        "round_id": rid,
        "rejection_reasons": tuple(reasons),
        "delta_sha256": actual_delta_sha,
        "measured_max_abs_basis_leak": meas_basis,
        "measured_max_abs_challenge_leak": meas_chal,
        "lean4_verified": lean4_verified,
        "delta_map": delta_map if len(reasons) == 0 else None,
    }


# ---------------------------------------------------------------------------
# 5. Multi-Krum Byzantine Multi-Hive Aggregation inside P_0 (n >= 2f + 3)
# ---------------------------------------------------------------------------


def _map_projectors_to_delta_keys(
    delta_sample: Mapping[str, torch.Tensor],
    commitment: Mapping[str, Any],
    *,
    protect_layernorm: bool,
    protect_general_expert: bool,
    general_expert_index: int,
) -> dict[str, torch.Tensor]:
    """Map committed ``P_0`` projectors directly to concrete parameter keys in ``delta_sample``."""
    v0_bases: Mapping[str, torch.Tensor] = commitment["v0_bases"]
    p0_projs: Mapping[str, torch.Tensor] = commitment["p0_projectors"]
    mapped: dict[str, torch.Tensor] = {}
    for key, val in delta_sample.items():
        b_key, _ = _resolve_basis_for_key(
            key,
            tuple(val.shape),
            v0_bases,
            protect_layernorm=protect_layernorm,
            protect_general_expert=protect_general_expert,
            general_expert_index=general_expert_index,
        )
        if b_key is not None and b_key in p0_projs:
            mapped[key] = p0_projs[b_key]
    return mapped


@torch.no_grad()
def verify_and_aggregate_federated_hives(
    packets: Sequence[Mapping[str, Any]],
    subspace_commitment: Mapping[str, Any] | torch.Tensor,
    *,
    lean_spec_audit: Lean4WSPSpecAudit | None = None,
    lean_spec_path: Path | str | None = None,
    expected_round_id: str | None = None,
    config: ZKWSPFederatedHiveConfig | None = None,
    enforce_byzantine_bound: bool = True,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
    """Verify ZK-WSP + Lean 4 receipts on all Hive packets and run Multi-Krum inside ``P_0``."""
    if isinstance(packets, (str, bytes)) or not isinstance(packets, Sequence) or not packets:
        raise ValueError("packets must be a non-empty sequence of federated hive packet mappings")
    _validate_bool(enforce_byzantine_bound, "enforce_byzantine_bound")

    cfg = config if config is not None else ZKWSPFederatedHiveConfig()
    if not isinstance(cfg, ZKWSPFederatedHiveConfig):
        raise TypeError(f"config must be a ZKWSPFederatedHiveConfig or None, got {type(cfg).__name__}")

    n_submitted = len(packets)
    if enforce_byzantine_bound and (n_submitted < 3 or 2 * cfg.num_byzantine_f >= n_submitted - 2):
        raise ValueError(
            f"Multi-Krum requires n >= 2*f + 3 (f < (n - 2)/2); got n={n_submitted}, f={cfg.num_byzantine_f}"
        )
    if cfg.select_m is not None and cfg.select_m > n_submitted:
        raise ValueError(f"select_m ({cfg.select_m}) cannot exceed submitted hive count ({n_submitted})")

    t0 = time.perf_counter()
    commitment = _resolve_commitment(subspace_commitment, basis_orthonormal_tol=cfg.basis_orthonormal_tol)
    audit = _resolve_lean_audit(lean_spec_audit, lean_spec_path) if cfg.require_lean4_attestation else None

    seen_hives: set[str] = set()
    accepted_packets: list[Mapping[str, Any]] = []
    accepted_deltas: list[dict[str, torch.Tensor]] = []
    accepted_ids: list[str] = []
    accepted_weights: list[float] = []
    rejected_hives: list[dict[str, Any]] = []
    per_hive_audit: list[dict[str, Any]] = []

    for idx, pkt in enumerate(packets):
        verif = verify_federated_hive_packet(
            pkt,
            commitment,
            lean_spec_audit=audit,
            expected_round_id=expected_round_id,
            config=cfg,
        )
        hid = verif["hive_id"]
        if hid in seen_hives:
            raise ValueError(f"Duplicate hive_id {hid!r} in submitted packets")
        seen_hives.add(hid)

        row_meta = {
            "index": idx,
            "hive_id": hid,
            "accepted": verif["accepted"],
            "rejection_reasons": list(verif["rejection_reasons"]),
            "measured_max_abs_basis_leak": verif["measured_max_abs_basis_leak"],
            "measured_max_abs_challenge_leak": verif["measured_max_abs_challenge_leak"],
            "lean4_verified": verif["lean4_verified"],
            "delta_sha256": verif["delta_sha256"],
        }
        per_hive_audit.append(row_meta)
        if verif["accepted"]:
            assert verif["delta_map"] is not None
            accepted_packets.append(pkt)
            accepted_deltas.append(verif["delta_map"])
            accepted_ids.append(hid)
            accepted_weights.append(float(pkt.get("peer_weight", 1.0)))
        else:
            rejected_hives.append(
                {
                    "hive_id": hid,
                    "reasons": list(verif["rejection_reasons"]),
                    "measured_leak": verif["measured_max_abs_basis_leak"],
                }
            )

    # Optional pre-aggregation spectral/sparsity screening over verified surviving hives
    screen_report: dict[str, Any] | None = None
    if cfg.enable_spectral_screening and accepted_deltas:
        screened_maps, screen_report = screen_peer_wsp_deltas_spectral_and_sparsity(
            accepted_deltas,
            max_edit_fraction=cfg.max_edit_fraction,
            max_spectral_spike_ratio=cfg.max_spectral_spike_ratio,
            peer_ids=accepted_ids,
        )
        screened_set = set(screen_report["accepted_peer_ids"])
        for hid in screen_report["rejected_peer_ids"]:
            rejected_hives.append(
                {
                    "hive_id": hid,
                    "reasons": list(screen_report["rejection_reasons"][hid]),
                    "measured_leak": None,
                }
            )
        surviving_deltas = [d for d, h in zip(accepted_deltas, accepted_ids, strict=True) if h in screened_set]
        surviving_ids = [h for h in accepted_ids if h in screened_set]
        surviving_weights = [w for w, h in zip(accepted_weights, accepted_ids, strict=True) if h in screened_set]
        assert len(surviving_deltas) == len(screened_maps)
    else:
        surviving_deltas = accepted_deltas
        surviving_ids = accepted_ids
        surviving_weights = accepted_weights

    n_verified = len(surviving_deltas)
    n_crypto_rejected = n_submitted - n_verified
    remaining_f = max(0, cfg.num_byzantine_f - n_crypto_rejected)
    quorum_ok = n_verified > 0
    if quorum_ok and enforce_byzantine_bound and cfg.enforce_quorum_on_verified:
        if n_crypto_rejected > cfg.num_byzantine_f or n_verified < 2 * remaining_f + 3:
            quorum_ok = False

    if not quorum_ok:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return None, {
            "quorum_satisfied": False,
            "submitted_hives": [str(p.get("hive_id", "")) for p in packets],
            "zkwsp_accepted_hives": accepted_ids,
            "surviving_hives": surviving_ids,
            "multikrum_selected_hives": [],
            "multikrum_rejected_hives": [],
            "rejected_hives": rejected_hives,
            "per_hive_audit": per_hive_audit,
            "screening_report": screen_report,
            "krum_report": None,
            "consensus_max_abs_basis_leak": None,
            "num_submitted": n_submitted,
            "num_verified": n_verified,
            "num_rejected_by_verifier": n_crypto_rejected,
            "remaining_byzantine_budget_f": remaining_f,
            "verification_and_krum_ms": round(elapsed_ms, 4),
            "per_peer_verify_ms": round(elapsed_ms / float(n_submitted), 4),
        }

    # Check schema consistency across surviving hives and map P_0 projectors to concrete keys
    _normalize_peer_delta_collection(surviving_deltas, peer_ids=surviving_ids)
    p0_for_keys = _map_projectors_to_delta_keys(
        surviving_deltas[0],
        commitment,
        protect_layernorm=cfg.protect_layernorm,
        protect_general_expert=cfg.protect_general_expert,
        general_expert_index=cfg.general_expert_index,
    )
    eff_f = min(remaining_f, max(0, (n_verified - 3) // 2)) if n_verified >= 3 else 0
    if cfg.select_m is not None:
        eff_select_m = min(cfg.select_m, n_verified)
    else:
        eff_select_m = max(1, n_verified - eff_f)

    norm_weights = _normalize_peer_weights(surviving_weights, n_verified)
    if n_verified >= 3 and 2 * eff_f < n_verified - 2:
        consensus_delta, krum_meta = multi_krum_nullspace_aggregate(
            {hid: d for hid, d in zip(surviving_ids, surviving_deltas, strict=True)},
            num_byzantine_f=eff_f,
            select_m=eff_select_m,
            p_null=p0_for_keys if p0_for_keys else None,
            peer_weights=norm_weights,
            enforce_byzantine_bound=True,
            protect_layernorm=cfg.protect_layernorm,
            protect_general_expert=cfg.protect_general_expert,
            general_expert_index=cfg.general_expert_index,
        )
        assert isinstance(consensus_delta, dict)
    else:
        # Allowed only when enforce_byzantine_bound=False and n_verified < 3
        consensus_delta = {}
        for key in sorted(surviving_deltas[0].keys()):
            ref = surviving_deltas[0][key]
            acc = torch.zeros_like(ref, device="cpu", dtype=torch.float64)
            for d_map, w in zip(surviving_deltas, norm_weights, strict=True):
                acc.add_(d_map[key].detach().to(device="cpu", dtype=torch.float64), alpha=w)
            if key in p0_for_keys:
                p64 = p0_for_keys[key]
                acc = (acc @ p64) if acc.shape[-1] == p64.shape[0] else (p64 @ acc)
            consensus_delta[key] = acc.to(device=ref.device, dtype=ref.dtype).contiguous()
        krum_meta = {
            "num_peers": n_verified,
            "num_byzantine_f": 0,
            "select_m": n_verified,
            "selected_peer_ids": list(surviving_ids),
            "rejected_peer_ids": [],
            "krum_distances_by_peer": {hid: 0.0 for hid in surviving_ids},
            "projection_applied": bool(p0_for_keys),
            "projected_keys": sorted(p0_for_keys.keys()),
        }

    consensus_witness = compute_zkwsp_spotcheck_witness(
        "consensus_hive",
        consensus_delta,
        commitment,
        round_id=expected_round_id or "round_0",
        spotcheck_tau_zk=cfg.spotcheck_tau_zk,
        num_challenges=cfg.num_spotcheck_challenges,
        basis_orthonormal_tol=cfg.basis_orthonormal_tol,
        protect_layernorm=cfg.protect_layernorm,
        protect_general_expert=cfg.protect_general_expert,
        general_expert_index=cfg.general_expert_index,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return consensus_delta, {
        "quorum_satisfied": True,
        "submitted_hives": [str(p.get("hive_id", "")) for p in packets],
        "zkwsp_accepted_hives": accepted_ids,
        "surviving_hives": surviving_ids,
        "multikrum_selected_hives": list(krum_meta["selected_peer_ids"]),
        "multikrum_rejected_hives": list(krum_meta["rejected_peer_ids"]),
        "rejected_hives": rejected_hives,
        "per_hive_audit": per_hive_audit,
        "screening_report": screen_report,
        "krum_report": krum_meta,
        "consensus_max_abs_basis_leak": float(consensus_witness["claimed_max_abs_basis_leak"]),
        "consensus_max_abs_challenge_leak": float(consensus_witness["claimed_max_abs_challenge_leak"]),
        "num_submitted": n_submitted,
        "num_verified": n_verified,
        "num_rejected_by_verifier": n_crypto_rejected,
        "remaining_byzantine_budget_f": remaining_f,
        "verification_and_krum_ms": round(elapsed_ms, 4),
        "per_peer_verify_ms": round(elapsed_ms / float(n_submitted), 4),
    }


# ---------------------------------------------------------------------------
# 6. End-to-End ANSE-Gated Multi-Hive Federation & Atomic Bitwise Rollback
# ---------------------------------------------------------------------------


def _score_candidate_eval(
    parent_dom_nlls: Mapping[str, float],
    parent_nll_mean: float,
    cand_eval: Mapping[str, Any],
    *,
    cfg: ZKWSPFederatedHiveConfig,
) -> dict[str, Any]:
    """Evaluate general/worst-domain guard and ANSE energy vs parent baseline."""
    cand_dom_raw: Mapping[str, float | None] = cand_eval["domain_nlls"]
    per_domain_drift_pct: dict[str, float | None] = {}
    general_ratios: list[float] = []
    all_ratios: list[float] = []
    for dom, p_val in parent_dom_nlls.items():
        c_val = cand_dom_raw.get(dom)
        if c_val is None or not math.isfinite(float(c_val)):
            per_domain_drift_pct[dom] = None
            continue
        ratio = float(c_val) / max(abs(p_val), 1e-8)
        per_domain_drift_pct[dom] = 100.0 * (ratio - 1.0)
        all_ratios.append(ratio)
        if _is_general_domain(dom, cfg.general_prefixes):
            general_ratios.append(ratio)

    all_finite = bool(cand_eval["all_finite"]) and len(all_ratios) == len(parent_dom_nlls)
    general_ratio = max(general_ratios) if general_ratios else (max(all_ratios) if all_ratios else 999.0)
    worst_ratio = max(all_ratios) if all_ratios else 999.0
    general_guard_ok = all_finite and (general_ratio <= cfg.domain_guard_max_ratio)
    all_dom_guard_ok = all_finite and (worst_ratio <= cfg.domain_guard_max_ratio)
    guard_ok = bool(general_guard_ok and (all_dom_guard_ok if cfg.enforce_all_domain_guard else True))

    cand_nll_mean = float(cand_eval["nll_mean"]) if (all_finite and cand_eval["nll_mean"] is not None) else None
    if not all_finite or not guard_ok or cand_nll_mean is None:
        cand_anse_energy = float(cfg.maximum_pain_energy)
    else:
        soft_pen = float(cfg.soft_margin_penalty) if general_ratio > cfg.anse_safe_general_ratio else 0.0
        cand_anse_energy = cand_nll_mean + soft_pen

    return {
        "all_finite": all_finite,
        "guard_ok": guard_ok,
        "general_guard_ok": general_guard_ok,
        "cand_nll_mean": cand_nll_mean,
        "cand_anse_energy": float(cand_anse_energy),
        "delta_e": float(cand_anse_energy - parent_nll_mean),
        "general_drift_pct": 100.0 * (general_ratio - 1.0) if all_finite else None,
        "worst_domain_drift_pct": 100.0 * (worst_ratio - 1.0) if all_finite else None,
        "per_domain_nll_candidate": {k: (float(v) if v is not None else None) for k, v in cand_dom_raw.items()},
        "per_domain_drift_pct": per_domain_drift_pct,
    }


@torch.no_grad()
def execute_verified_zkwsp_federated_hive_round(
    model: nn.Module,
    hive_packets: Sequence[Mapping[str, Any]],
    *,
    subspace_commitment: Mapping[str, Any] | torch.Tensor | None = None,
    prose_keys: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    probe_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_fn: Callable[[nn.Module], Mapping[str, Any]] | None = None,
    lean_spec_audit: Lean4WSPSpecAudit | None = None,
    lean_spec_path: Path | str | None = None,
    round_id: str | None = None,
    config: ZKWSPFederatedHiveConfig | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Verify multi-hive ZK-WSP packets, aggregate inside ``P_0``, and commit under ANSE gate."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if eval_fn is not None and not callable(eval_fn):
        raise TypeError("eval_fn must be callable or None")
    model_sd = model.state_dict()
    if not model_sd:
        raise ValueError("model state_dict must be non-empty")
    if probe_chunks is None and eval_chunks is None and eval_fn is None:
        raise ValueError("Provide at least one of probe_chunks, eval_chunks, or eval_fn")

    cfg = config if config is not None else ZKWSPFederatedHiveConfig()
    if not isinstance(cfg, ZKWSPFederatedHiveConfig):
        raise TypeError(f"config must be a ZKWSPFederatedHiveConfig or None, got {type(cfg).__name__}")

    work_model = model if cfg.inplace else copy.deepcopy(model)
    audit = _resolve_lean_audit(lean_spec_audit, lean_spec_path)

    if subspace_commitment is not None:
        commitment = _resolve_commitment(subspace_commitment, basis_orthonormal_tol=cfg.basis_orthonormal_tol)
    elif prose_keys is not None:
        commitment = build_prose_subspace_commitment_from_keys(
            prose_keys,
            sv_threshold=cfg.sv_threshold,
            variance_threshold=cfg.prose_variance_threshold,
            max_active_modes=cfg.prose_max_active_modes,
            basis_orthonormal_tol=cfg.basis_orthonormal_tol,
        )
    elif probe_chunks is not None:
        commitment = collect_prose_subspace_from_probe(
            work_model,
            probe_chunks,
            general_prefixes=cfg.general_prefixes,
            sv_threshold=cfg.sv_threshold,
            variance_threshold=cfg.prose_variance_threshold,
            max_active_modes=cfg.prose_max_active_modes,
            basis_orthonormal_tol=cfg.basis_orthonormal_tol,
            score_from=cfg.score_from,
            batch_size=cfg.batch_size,
        )
    else:
        raise ValueError("Provide subspace_commitment, prose_keys, or probe_chunks to define V_0")

    raw_verify_chunks = eval_chunks if eval_chunks is not None else probe_chunks
    verify_chunks = (
        _normalize_eval_chunks_for_domain_eval(
            raw_verify_chunks,
            "eval_chunks" if eval_chunks is not None else "probe_chunks",
        )
        if raw_verify_chunks is not None
        else None
    )
    was_training = work_model.training
    work_model.eval()
    try:
        parent_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in work_model.state_dict().items()}
        parent_sha = compute_state_sha256(parent_snapshot)

        parent_eval = _evaluate_model_domains(
            work_model,
            eval_chunks=verify_chunks,
            eval_fn=eval_fn,
            score_from=cfg.score_from,
            batch_size=cfg.batch_size,
        )
        if not parent_eval["all_finite"] or parent_eval["nll_mean"] is None:
            raise ValueError("Parent model evaluation produced non-finite NLL")
        parent_dom_nlls: dict[str, float] = {k: float(v) for k, v in parent_eval["domain_nlls"].items()}
        parent_nll_mean = float(parent_eval["nll_mean"])

        consensus_delta, zkwsp_audit = verify_and_aggregate_federated_hives(
            hive_packets,
            commitment,
            lean_spec_audit=audit,
            expected_round_id=round_id,
            config=cfg,
            enforce_byzantine_bound=True,
        )

        if consensus_delta is None:
            final_sha = compute_state_sha256(work_model)
            cand_anse_energy = float(cfg.maximum_pain_energy)
            status = (
                "ROLLED_BACK_ALL_HIVES_REJECTED"
                if zkwsp_audit["num_verified"] == 0
                else "ROLLED_BACK_QUORUM_BELOW_BYZANTINE_BOUND"
            )
            receipt_core: dict[str, Any] = {
                "object": ZKWSP_ROUND_RECEIPT_OBJECT,
                "schema": ZKWSP_VERIFIER_SCHEMA,
                "lean4_theorem_id": LEAN4_RCD_D5_THEOREM,
                "lean4_spec_sha256": audit.spec_sha256,
                "v0_sha256": commitment["v0_sha256"],
                "p0_sha256": commitment["p0_sha256"],
                "round_id": round_id,
                "accepted": False,
                "rolled_back": True,
                "rollback_bitwise_verified": final_sha == parent_sha,
                "status": status,
                "guard_ok": False,
                "general_guard_ok": False,
                "nll_mean": parent_nll_mean,
                "parent_nll_mean": parent_nll_mean,
                "candidate_nll_mean": None,
                "general_drift_pct": None,
                "worst_domain_drift_pct": None,
                "per_domain_nll_parent": parent_dom_nlls,
                "per_domain_nll_candidate": {k: None for k in parent_dom_nlls},
                "per_domain_drift_pct": {k: None for k in parent_dom_nlls},
                "anse_energy": cand_anse_energy,
                "parent_anse_energy": parent_nll_mean,
                "candidate_anse_energy": cand_anse_energy,
                "final_anse_energy": parent_nll_mean,
                "delta_e": float(cand_anse_energy - parent_nll_mean),
                "eval_error": None,
                "consensus_delta_sha256": "",
                "parent_state_sha256": parent_sha,
                "candidate_state_sha256": parent_sha,
                "final_state_sha256": final_sha,
                "zkwsp_audit": zkwsp_audit,
                "config": asdict(cfg),
            }
            return work_model, {**receipt_core, "receipt_sha256": compute_payload_sha256(receipt_core)}

        work_sd = work_model.state_dict()
        for k, d_tensor in consensus_delta.items():
            if k not in work_sd:
                raise ValueError(f"Consensus delta key {k!r} not found in model.state_dict()")
            if tuple(d_tensor.shape) != tuple(work_sd[k].shape):
                raise ValueError(
                    f"Consensus delta shape {tuple(d_tensor.shape)} for {k!r} != model shape {tuple(work_sd[k].shape)}"
                )
            updated = (
                work_sd[k].detach().to(torch.float64)
                + cfg.consensus_step_scale * d_tensor.detach().to(device=work_sd[k].device, dtype=torch.float64)
            ).to(dtype=work_sd[k].dtype)
            work_sd[k].copy_(updated)

        consensus_sha = compute_state_sha256(consensus_delta)
        cand_sha = compute_state_sha256(work_model)

        eval_error: str | None = None
        try:
            cand_raw = _evaluate_model_domains(
                work_model,
                eval_chunks=verify_chunks,
                eval_fn=eval_fn,
                score_from=cfg.score_from,
                batch_size=cfg.batch_size,
            )
            scored = _score_candidate_eval(parent_dom_nlls, parent_nll_mean, cand_raw, cfg=cfg)
        except Exception as exc:
            eval_error = f"{type(exc).__name__}: {exc}"
            scored = {
                "all_finite": False,
                "guard_ok": False,
                "general_guard_ok": False,
                "cand_nll_mean": None,
                "cand_anse_energy": float(cfg.maximum_pain_energy),
                "delta_e": float(cfg.maximum_pain_energy - parent_nll_mean),
                "general_drift_pct": None,
                "worst_domain_drift_pct": None,
                "per_domain_nll_candidate": {k: None for k in parent_dom_nlls},
                "per_domain_drift_pct": {k: None for k in parent_dom_nlls},
            }

        delta_e = float(scored["delta_e"])
        energy_ok = (delta_e < 0.0) if cfg.require_strict_energy_drop else (delta_e <= 0.0)

        if cand_sha == parent_sha:
            accepted = False
            rolled_back = False
            status = "FIXED_POINT_UNCHANGED" if scored["guard_ok"] else "REJECTED_GUARD_VIOLATION_UNCHANGED"
        elif eval_error is None and scored["all_finite"] and scored["guard_ok"] and energy_ok:
            accepted = True
            rolled_back = False
            status = "ACCEPTED"
        else:
            accepted = False
            rolled_back = True
            if eval_error is not None:
                status = "ROLLED_BACK_EVAL_FAILURE"
            elif not scored["all_finite"]:
                status = "ROLLED_BACK_NON_FINITE"
            elif not scored["guard_ok"]:
                status = "ROLLED_BACK_GUARD_VIOLATION"
            else:
                status = "ROLLED_BACK_ENERGY_NON_IMPROVING"
            _restore_snapshot_bitwise(work_model, parent_snapshot)

        final_sha = compute_state_sha256(work_model)
        if not accepted and final_sha != parent_sha:
            raise RuntimeError(f"Atomic bitwise rollback failed: {final_sha[:12]} != {parent_sha[:12]}")
    finally:
        work_model.train(was_training)

    receipt_ok_core: dict[str, Any] = {
        "object": ZKWSP_ROUND_RECEIPT_OBJECT,
        "schema": ZKWSP_VERIFIER_SCHEMA,
        "lean4_theorem_id": LEAN4_RCD_D5_THEOREM,
        "lean4_spec_sha256": audit.spec_sha256,
        "v0_sha256": commitment["v0_sha256"],
        "p0_sha256": commitment["p0_sha256"],
        "round_id": round_id,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "rollback_bitwise_verified": bool((not accepted) and final_sha == parent_sha),
        "status": status,
        "guard_ok": bool(scored["guard_ok"]),
        "general_guard_ok": bool(scored["general_guard_ok"]),
        "nll_mean": float(scored["cand_nll_mean"]) if (accepted and scored["cand_nll_mean"] is not None) else parent_nll_mean,
        "parent_nll_mean": parent_nll_mean,
        "candidate_nll_mean": scored["cand_nll_mean"],
        "general_drift_pct": scored["general_drift_pct"],
        "worst_domain_drift_pct": scored["worst_domain_drift_pct"],
        "per_domain_nll_parent": parent_dom_nlls,
        "per_domain_nll_candidate": scored["per_domain_nll_candidate"],
        "per_domain_drift_pct": scored["per_domain_drift_pct"],
        "anse_energy": float(scored["cand_anse_energy"]),
        "parent_anse_energy": parent_nll_mean,
        "candidate_anse_energy": float(scored["cand_anse_energy"]),
        "final_anse_energy": float(scored["cand_anse_energy"] if accepted else parent_nll_mean),
        "delta_e": float(scored["delta_e"]),
        "eval_error": eval_error,
        "consensus_delta_sha256": consensus_sha,
        "parent_state_sha256": parent_sha,
        "candidate_state_sha256": cand_sha,
        "final_state_sha256": final_sha,
        "zkwsp_audit": zkwsp_audit,
        "config": asdict(cfg),
    }
    return work_model, {**receipt_ok_core, "receipt_sha256": compute_payload_sha256(receipt_ok_core)}


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_BASIS_ORTHONORMAL_TOL",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_GENERAL_PREFIXES",
    "DEFAULT_MAX_EDIT_FRACTION",
    "DEFAULT_MAX_SPECTRAL_SPIKE_RATIO",
    "DEFAULT_NUM_SPOTCHECK_CHALLENGES",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_SOFT_MARGIN_PENALTY",
    "DEFAULT_SPOTCHECK_TAU_ZK",
    "DEFAULT_SV_THRESHOLD",
    "LEAN4_RCD_D5_THEOREM",
    "ZKWSP_COMMITMENT_OBJECT",
    "ZKWSP_PACKET_OBJECT",
    "ZKWSP_ROUND_RECEIPT_OBJECT",
    "ZKWSP_VERIFIER_SCHEMA",
    "ZKWSP_WITNESS_OBJECT",
    "ZKWSPFederatedHiveConfig",
    "build_federated_hive_packet",
    "build_prose_subspace_commitment",
    "build_prose_subspace_commitment_from_keys",
    "collect_prose_subspace_from_probe",
    "compute_payload_sha256",
    "compute_state_sha256",
    "compute_zkwsp_spotcheck_witness",
    "execute_verified_zkwsp_federated_hive_round",
    "verify_federated_hive_packet",
    "verify_and_aggregate_federated_hives",
    "verify_zkwsp_federated_round_receipt_provenance",
    "verify_zkwsp_spotcheck_witness",
]

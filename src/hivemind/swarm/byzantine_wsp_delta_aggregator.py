# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Byzantine-Resilient Multi-Krum & Spectral-Norm-Guarded WSP Delta Aggregator in Prose Null-Space.

Enables decentralized volunteer/peer LoRA and Weight-Space Programming (WSP) edits
to be safely merged into open-weight Sparse MoE models (``CrowdV1WithEngram``,
``CrowdLoRAMoEMLP``, ``CrowdFineGrainedMoEMLP``, and generic ``nn.Module`` layers)
under HiveMind Pillar 3 Byzantine resilience:

1. ``screen_peer_wsp_deltas_spectral_and_sparsity(...)``:
   Pre-screens peer delta submissions before aggregation, rejecting non-finite
   (``NaN``/``Inf``) tensors, dense flood edits exceeding ``max_edit_fraction``,
   and norm-matched rank-1 Trojan spectral spikes exceeding ``max_spectral_spike_ratio``.
2. ``multi_krum_nullspace_aggregate(...)``:
   Computes exact CPU ``float64`` pairwise Euclidean Multi-Krum distances to filter
   up to ``f < (n - 2) / 2`` Byzantine poisoned weight deltas, and projects the
   surviving consensus delta strictly into the AlphaEdit general-prose SVD null-space
   ``P_0 = I - V_active^T V_active`` so protected general-prose keys suffer zero drift.
3. ``apply_verified_byzantine_wsp_round(...)``:
   Orchestrates end-to-end screening, optional per-layer ``P_0`` construction from
   ``E0_*`` probe states, Multi-Krum + null-space aggregation, and ANSE verification
   (committing if and only if ``Delta E < 0`` and general-domain drift ``<= +1.0%``),
   enforcing atomic bitwise parent rollback and canonical SHA-256 receipts on any
   breach or ANSE Maximum Pain (``1e6``) rejection.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdFineGrainedMoEMLP,
    CrowdLoRAMoEMLP,
)
from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    compute_swiglu_intermediate_states,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    forward_model_pre_ln,
    per_chunk_nll_from_pre_ln,
)

SCHEMA_VERSION: str = "hivemind.byzantine_wsp_delta_aggregator/1"
RECEIPT_OBJECT: str = "hivemind.byzantine_wsp_round_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_MAX_EDIT_FRACTION: float = 0.05
DEFAULT_MAX_SPECTRAL_SPIKE_RATIO: float = 3.5
DEFAULT_EDIT_TOL: float = 1e-12
DEFAULT_SV_THRESHOLD: float = 1e-2
DEFAULT_DOMAIN_GUARD_MAX_RATIO: float = 1.01
DEFAULT_ANSE_SAFE_E0_RATIO: float = 1.0085
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_SOFT_MARGIN_PENALTY: float = 0.05
DEFAULT_SCORE_FROM: int = 0
VALID_DELTA_FORMATS: frozenset[str] = frozenset({"auto", "state_dict", "buckets"})

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_LAYER_KEY_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
_EXPERT_KEY_RE = re.compile(r"(?:^|\.)experts\.(\d+)\.|^expert_(\d+)$")
_LORA_FACTOR_PAIR_RE = re.compile(r"^(.*?\.(?:w1|w2|w3)|(?:w1|w2|w3))_(A|B)$")
_VECTOR_LIKE_MARKERS: tuple[str, ...] = ("router", "ln.", ".ln", "ln1", "ln2", "norm", "bias")
_LAYERNORM_MARKERS: tuple[str, ...] = ("ln.weight", "ln.bias", "ln1.", "ln2.", "layernorm", "rmsnorm")


# ---------------------------------------------------------------------------
# 1. Validation & Canonical Provenance Helpers
# ---------------------------------------------------------------------------


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_int(value: Any, name: str) -> int:
    """Validate that ``value`` is an integer (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return int(value)


def _validate_nonneg_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a non-negative integer."""
    out = _validate_int(value, name)
    if out < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return out


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a strictly positive integer."""
    out = _validate_int(value, name)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return out


def _validate_finite_float(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number (excluding booleans)."""
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


def _validate_unit_interval(
    value: Any,
    name: str,
    *,
    allow_zero: bool = True,
) -> float:
    """Validate a finite real number inside ``[0, 1]``."""
    out = _validate_nonneg_finite(value, name)
    if not allow_zero and out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    if out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    return out


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically with strict finite-number enforcement."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def compute_payload_sha256(payload: Any) -> str:
    """Return lowercase SHA-256 hex digest of ``canonical_json_bytes(payload)``."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_state_sha256(state_or_module: nn.Module | Mapping[str, torch.Tensor]) -> str:
    """Compute a deterministic bitwise SHA-256 digest over a model or tensor mapping."""
    if isinstance(state_or_module, nn.Module):
        state: Mapping[str, torch.Tensor] = state_or_module.state_dict()
    elif isinstance(state_or_module, Mapping):
        state = state_or_module
    else:
        raise TypeError(
            f"state_or_module must be an nn.Module or Mapping[str, Tensor], got {type(state_or_module).__name__}"
        )
    if not state:
        raise ValueError("state_dict must be non-empty")

    digest = hashlib.sha256()
    for key in sorted(state.keys()):
        if not isinstance(key, str) or not key:
            raise ValueError(f"state_dict keys must be non-empty strings, got {key!r}")
        val = state[key]
        if not isinstance(val, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] must be a torch.Tensor, got {type(val).__name__}")
        t_cpu = val.detach().to(device="cpu").contiguous()
        if t_cpu.is_floating_point() and not bool(torch.isfinite(t_cpu).all().item()):
            raise ValueError(f"state_dict[{key!r}] contains non-finite values")
        digest.update(f"{key}:{t_cpu.dtype}:{tuple(t_cpu.shape)}:".encode())
        elem_bytes = t_cpu.element_size()
        view_dt = (
            torch.uint8
            if elem_bytes == 1
            else (torch.int16 if elem_bytes == 2 else (torch.int32 if elem_bytes == 4 else torch.int64))
        )
        digest.update(t_cpu.view(view_dt).reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_byzantine_wsp_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify schema, object tag, and canonical ``receipt_sha256``."""
    if not isinstance(receipt, Mapping):
        raise TypeError(f"receipt must be a mapping, got {type(receipt).__name__}")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if receipt.get("object") != RECEIPT_OBJECT or receipt.get("schema") != SCHEMA_VERSION:
        return False
    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


@dataclass(frozen=True)
class ByzantineWSPConfig:
    """Validated configuration for Byzantine-resilient WSP screening, Multi-Krum, and ANSE verification."""

    num_byzantine_f: int = 1
    select_m: int | None = None
    max_edit_fraction: float = DEFAULT_MAX_EDIT_FRACTION
    max_spectral_spike_ratio: float = DEFAULT_MAX_SPECTRAL_SPIKE_RATIO
    edit_tol: float = DEFAULT_EDIT_TOL
    enable_screening: bool = True
    check_cohort_spectral_ratio: bool = True
    check_intra_matrix_spectral_ratio: bool = True
    sv_threshold: float = DEFAULT_SV_THRESHOLD
    prose_variance_threshold: float | None = None
    prose_max_active_modes: int | None = None
    protect_layernorm: bool = True
    protect_general_expert: bool = False
    general_expert_index: int = 0
    domain_guard_max_ratio: float = DEFAULT_DOMAIN_GUARD_MAX_RATIO
    anse_safe_general_ratio: float = DEFAULT_ANSE_SAFE_E0_RATIO
    maximum_pain_energy: float = ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_SOFT_MARGIN_PENALTY
    enforce_all_domain_guard: bool = False
    require_strict_energy_drop: bool = True
    general_prefix: str = GENERAL_EXPERT_PREFIX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = 24

    def __post_init__(self) -> None:
        _validate_nonneg_int(self.num_byzantine_f, "num_byzantine_f")
        if self.select_m is not None:
            _validate_positive_int(self.select_m, "select_m")
        _validate_unit_interval(self.max_edit_fraction, "max_edit_fraction", allow_zero=False)
        _validate_positive_finite(self.max_spectral_spike_ratio, "max_spectral_spike_ratio")
        _validate_nonneg_finite(self.edit_tol, "edit_tol")
        _validate_bool(self.enable_screening, "enable_screening")
        _validate_bool(self.check_cohort_spectral_ratio, "check_cohort_spectral_ratio")
        _validate_bool(self.check_intra_matrix_spectral_ratio, "check_intra_matrix_spectral_ratio")
        _validate_nonneg_finite(self.sv_threshold, "sv_threshold")
        if self.prose_variance_threshold is not None:
            _validate_unit_interval(
                self.prose_variance_threshold, "prose_variance_threshold", allow_zero=False
            )
        if self.prose_max_active_modes is not None:
            _validate_positive_int(self.prose_max_active_modes, "prose_max_active_modes")
        _validate_bool(self.protect_layernorm, "protect_layernorm")
        _validate_bool(self.protect_general_expert, "protect_general_expert")
        _validate_nonneg_int(self.general_expert_index, "general_expert_index")
        _validate_positive_finite(self.domain_guard_max_ratio, "domain_guard_max_ratio")
        _validate_positive_finite(self.anse_safe_general_ratio, "anse_safe_general_ratio")
        _validate_positive_finite(self.maximum_pain_energy, "maximum_pain_energy")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_bool(self.enforce_all_domain_guard, "enforce_all_domain_guard")
        _validate_bool(self.require_strict_energy_drop, "require_strict_energy_drop")
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")


# ---------------------------------------------------------------------------
# 2. Peer Delta Collection Normalization & Schema Validation
# ---------------------------------------------------------------------------


def _normalize_peer_delta_collection(
    peer_deltas: Sequence[Mapping[str, torch.Tensor] | torch.Tensor]
    | Mapping[str, Mapping[str, torch.Tensor]],
    *,
    peer_ids: Sequence[str] | None = None,
) -> tuple[list[dict[str, torch.Tensor]], list[str], tuple[str, ...], bool]:
    """Validate and normalize peer delta inputs into canonical per-peer tensor dictionaries.

    Returns ``(delta_dicts, resolved_peer_ids, sorted_keys, is_tensor_sequence)``.
    """
    is_tensor_sequence = False
    raw_items: list[dict[str, torch.Tensor]] = []
    default_ids: list[str] = []

    if isinstance(peer_deltas, Mapping):
        if not peer_deltas:
            raise ValueError("peer_deltas must not be empty")
        for pid, p_val in peer_deltas.items():
            if not isinstance(pid, str) or not pid.strip() or any(ord(c) < 32 or ord(c) == 127 for c in pid):
                raise ValueError(f"Invalid peer ID key {pid!r} in peer_deltas mapping")
            if not isinstance(p_val, Mapping) or not p_val:
                raise TypeError(
                    "peer_deltas mapping values must be non-empty Mapping[str, torch.Tensor] parameter delta dicts"
                )
            raw_items.append({str(k): v for k, v in p_val.items()})
            default_ids.append(pid)
    elif isinstance(peer_deltas, Sequence) and not isinstance(peer_deltas, (str, bytes, bytearray)):
        if not peer_deltas:
            raise ValueError("peer_deltas must not be empty")
        first = peer_deltas[0]
        if isinstance(first, torch.Tensor):
            is_tensor_sequence = True
            for idx, item in enumerate(peer_deltas):
                if not isinstance(item, torch.Tensor):
                    raise TypeError(
                        f"peer_deltas[{idx}] must be a torch.Tensor to match peer_deltas[0], got {type(item).__name__}"
                    )
                raw_items.append({"delta": item})
                default_ids.append(f"peer_{idx}")
        elif isinstance(first, Mapping):
            for idx, item in enumerate(peer_deltas):
                if not isinstance(item, Mapping) or not item:
                    raise ValueError(
                        f"peer_deltas[{idx}] must be a non-empty Mapping[str, torch.Tensor]"
                    )
                raw_items.append({str(k): v for k, v in item.items()})
                default_ids.append(f"peer_{idx}")
        else:
            raise TypeError(
                f"peer_deltas entries must be torch.Tensor or Mapping[str, torch.Tensor], got {type(first).__name__}"
            )
    else:
        raise TypeError(
            "peer_deltas must be a sequence of delta dicts/tensors or a mapping of peer_id -> delta dict"
        )

    if peer_ids is not None:
        if isinstance(peer_ids, (str, bytes, bytearray)) or not isinstance(peer_ids, Sequence):
            raise TypeError("peer_ids must be a sequence of strings or None")
        if len(peer_ids) != len(raw_items):
            raise ValueError(f"peer_ids length ({len(peer_ids)}) must match peer count ({len(raw_items)})")
        resolved_ids: list[str] = []
        seen: set[str] = set()
        for idx, pid in enumerate(peer_ids):
            if not isinstance(pid, str) or not pid.strip() or any(ord(c) < 32 or ord(c) == 127 for c in pid):
                raise ValueError(f"peer_ids[{idx}] must be a non-empty string without control characters")
            if pid in seen:
                raise ValueError(f"Duplicate peer_id {pid!r} in peer_ids")
            seen.add(pid)
            resolved_ids.append(pid)
    else:
        resolved_ids = default_ids

    ref_keys = tuple(sorted(raw_items[0].keys()))
    if any(not k for k in ref_keys):
        raise ValueError("Delta parameter keys must be non-empty strings")
    ref_shapes: dict[str, tuple[int, ...]] = {}
    for key in ref_keys:
        t0 = raw_items[0][key]
        if not isinstance(t0, torch.Tensor):
            raise TypeError(f"peer_deltas[0][{key!r}] must be a torch.Tensor, got {type(t0).__name__}")
        if t0.numel() == 0:
            raise ValueError(f"peer_deltas[0][{key!r}] must be non-empty, got shape {tuple(t0.shape)}")
        if not t0.is_floating_point():
            raise TypeError(f"peer_deltas[0][{key!r}] must have floating-point dtype, got {t0.dtype}")
        ref_shapes[key] = tuple(t0.shape)

    for idx, item in enumerate(raw_items[1:], start=1):
        keys_i = tuple(sorted(item.keys()))
        if keys_i != ref_keys:
            raise ValueError(f"peer_deltas[{idx}] keys do not match peer_deltas[0] schema")
        for key in ref_keys:
            ti = item[key]
            if not isinstance(ti, torch.Tensor):
                raise TypeError(f"peer_deltas[{idx}][{key!r}] must be a torch.Tensor, got {type(ti).__name__}")
            if not ti.is_floating_point():
                raise TypeError(f"peer_deltas[{idx}][{key!r}] must have floating-point dtype, got {ti.dtype}")
            if tuple(ti.shape) != ref_shapes[key]:
                raise ValueError(
                    f"Shape mismatch on peer_deltas[{idx}][{key!r}]: {tuple(ti.shape)} != {ref_shapes[key]}"
                )

    return raw_items, resolved_ids, ref_keys, is_tensor_sequence


# ---------------------------------------------------------------------------
# 3. Pre-Aggregation Spectral-Spike & Sparsity Screening
# ---------------------------------------------------------------------------


def _is_vector_like_key(key: str) -> bool:
    """Return True for router gate, bias, or norm parameters whose updates are naturally vector-aligned."""
    low = key.lower()
    return any(marker in low for marker in _VECTOR_LIKE_MARKERS)


def _intra_spike_from_sv(sv: torch.Tensor, *, frob_norm: float) -> float:
    """Compute singular-value concentration ratio ``sigma_1 / max(sigma_2, 1e-12 * frob)``."""
    if sv.numel() < 2 or frob_norm <= 0.0:
        return 1.0
    s1 = float(sv[0].item())
    s2 = float(sv[1].item())
    return s1 / max(s2, 1e-12 * frob_norm)


def _matrix_spectral_metrics(
    tensor: torch.Tensor,
    key: str,
    *,
    edit_tol: float,
    check_intra: bool,
) -> tuple[float, float, float]:
    """Return ``(max_singular_value, intra_spike_ratio, frob_sq)`` for a finite tensor."""
    t64 = tensor.detach().to(device="cpu", dtype=torch.float64)
    frob_sq = float((t64 * t64).sum().item())
    if frob_sq <= edit_tol * edit_tol:
        return 0.0, 1.0, 0.0
    frob_norm = math.sqrt(frob_sq)

    if t64.ndim == 1:
        return frob_norm, 1.0, frob_sq

    slices = t64.reshape(-1, int(t64.shape[-2]), int(t64.shape[-1]))
    max_sv = 0.0
    max_intra = 1.0
    is_vec_key = _is_vector_like_key(key)
    for idx in range(int(slices.shape[0])):
        mat = slices[idx]
        m_frob_sq = float((mat * mat).sum().item())
        if m_frob_sq <= edit_tol * edit_tol:
            continue
        sv = torch.linalg.svdvals(mat)
        s1 = float(sv[0].item())
        max_sv = max(max_sv, s1)
        if check_intra and not is_vec_key and min(mat.shape) >= 2:
            nnz = int((mat.abs() > edit_tol).sum().item())
            if nnz >= 2:
                max_intra = max(max_intra, _intra_spike_from_sv(sv, frob_norm=math.sqrt(m_frob_sq)))
    return max_sv, max_intra, frob_sq


def _lora_pair_spectral_metrics(
    peer_map: Mapping[str, torch.Tensor],
    sorted_keys: tuple[str, ...],
    *,
    edit_tol: float,
) -> tuple[float, list[str]]:
    """Detect rank-1 backdoor concentration in paired LoRA ``(B, A)`` delta factors via thin QR-SVD."""
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key in sorted_keys:
        m = _LORA_FACTOR_PAIR_RE.match(key)
        if m is not None:
            pairs.setdefault(m.group(1), {})[m.group(2)] = peer_map[key]

    max_intra = 1.0
    flagged: list[str] = []
    for prefix, ab in sorted(pairs.items()):
        if "A" not in ab or "B" not in ab:
            continue
        a64 = ab["A"].detach().to(device="cpu", dtype=torch.float64)
        b64 = ab["B"].detach().to(device="cpu", dtype=torch.float64)
        if a64.ndim != 2 or b64.ndim != 2 or int(a64.shape[0]) < 2 or int(b64.shape[1]) != int(a64.shape[0]):
            continue
        if float(a64.abs().max().item()) <= edit_tol or float(b64.abs().max().item()) <= edit_tol:
            continue
        _, rb = torch.linalg.qr(b64, mode="reduced")
        _, ra = torch.linalg.qr(a64.T, mode="reduced")
        sv = torch.linalg.svdvals(rb @ ra.T)
        frob = float(sv.norm().item())
        if frob <= edit_tol:
            continue
        ratio = _intra_spike_from_sv(sv, frob_norm=frob)
        if ratio > max_intra:
            max_intra = ratio
            flagged = [f"{prefix}_B", f"{prefix}_A"]
    return max_intra, flagged


@torch.no_grad()
def screen_peer_wsp_deltas_spectral_and_sparsity(
    peer_deltas: Sequence[Mapping[str, torch.Tensor] | torch.Tensor]
    | Mapping[str, Mapping[str, torch.Tensor]],
    *,
    max_edit_fraction: float = DEFAULT_MAX_EDIT_FRACTION,
    max_spectral_spike_ratio: float = DEFAULT_MAX_SPECTRAL_SPIKE_RATIO,
    edit_tol: float = DEFAULT_EDIT_TOL,
    reference_total_params: int | None = None,
    peer_ids: Sequence[str] | None = None,
    check_cohort_spectral_ratio: bool = True,
    check_intra_matrix_spectral_ratio: bool = True,
) -> tuple[list[dict[str, torch.Tensor]] | list[torch.Tensor], dict[str, Any]]:
    """Screen peer WSP deltas for NaN/Inf values, dense flood edits, and rank-1 Trojan spikes."""
    max_frac = _validate_unit_interval(max_edit_fraction, "max_edit_fraction", allow_zero=False)
    max_spike = _validate_positive_finite(max_spectral_spike_ratio, "max_spectral_spike_ratio")
    tol = _validate_nonneg_finite(edit_tol, "edit_tol")
    _validate_bool(check_cohort_spectral_ratio, "check_cohort_spectral_ratio")
    _validate_bool(check_intra_matrix_spectral_ratio, "check_intra_matrix_spectral_ratio")

    delta_maps, resolved_ids, ref_keys, is_tensor_seq = _normalize_peer_delta_collection(
        peer_deltas, peer_ids=peer_ids
    )
    total_elements = sum(int(delta_maps[0][k].numel()) for k in ref_keys)
    if reference_total_params is not None:
        denom_elements = _validate_positive_int(reference_total_params, "reference_total_params")
        if denom_elements < total_elements:
            raise ValueError(
                f"reference_total_params ({denom_elements}) must be >= delta element count ({total_elements})"
            )
    else:
        denom_elements = total_elements

    prelim: list[dict[str, Any]] = []
    for idx, (pid, p_map) in enumerate(zip(resolved_ids, delta_maps, strict=True)):
        non_finite_keys = [
            k for k in ref_keys if not bool(torch.isfinite(p_map[k].detach()).all().item())
        ]
        if non_finite_keys:
            prelim.append(
                {
                    "index": idx,
                    "peer_id": pid,
                    "all_finite": False,
                    "edit_fraction": None,
                    "edited_elements": None,
                    "total_elements": denom_elements,
                    "max_singular_value": None,
                    "intra_spectral_spike_ratio": None,
                    "l2_norm": None,
                    "sparsity_ok": False,
                    "offending_keys": non_finite_keys,
                }
            )
            continue

        edited = sum(int((p_map[k].detach().abs() > tol).sum().item()) for k in ref_keys)
        edit_fraction = float(edited) / float(denom_elements)
        sparsity_ok = edit_fraction <= max_frac

        max_sv = 0.0
        max_intra = 1.0
        frob_sq_total = 0.0
        offending: list[str] = []
        if not sparsity_ok:
            offending.extend(
                k for k in ref_keys if int((p_map[k].detach().abs() > tol).sum().item()) > 0
            )

        for k in ref_keys:
            k_sv, k_intra, k_frob_sq = _matrix_spectral_metrics(
                p_map[k], k, edit_tol=tol, check_intra=check_intra_matrix_spectral_ratio
            )
            max_sv = max(max_sv, k_sv)
            frob_sq_total += k_frob_sq
            if k_intra > max_intra:
                max_intra = k_intra
            if check_intra_matrix_spectral_ratio and k_intra > max_spike and k not in offending:
                offending.append(k)

        if check_intra_matrix_spectral_ratio:
            pair_intra, pair_keys = _lora_pair_spectral_metrics(p_map, ref_keys, edit_tol=tol)
            if pair_intra > max_intra:
                max_intra = pair_intra
            if pair_intra > max_spike:
                for pk in pair_keys:
                    if pk not in offending:
                        offending.append(pk)

        prelim.append(
            {
                "index": idx,
                "peer_id": pid,
                "all_finite": True,
                "edit_fraction": edit_fraction,
                "edited_elements": edited,
                "total_elements": denom_elements,
                "max_singular_value": max_sv,
                "intra_spectral_spike_ratio": max_intra,
                "l2_norm": math.sqrt(frob_sq_total),
                "sparsity_ok": sparsity_ok,
                "offending_keys": offending,
            }
        )

    cohort_pool = [
        float(entry["max_singular_value"])
        for entry in prelim
        if entry["all_finite"] and entry["sparsity_ok"] and float(entry["max_singular_value"]) > tol
    ]
    if len(cohort_pool) < 3:
        cohort_pool = [
            float(entry["max_singular_value"])
            for entry in prelim
            if entry["all_finite"] and float(entry["max_singular_value"]) > tol
        ]
    cohort_median_sv = float(np.median(cohort_pool)) if len(cohort_pool) >= 3 else 0.0

    accepted_indices: list[int] = []
    rejected_indices: list[int] = []
    accepted_peer_ids: list[str] = []
    rejected_peer_ids: list[str] = []
    rejection_reasons: dict[str, list[str]] = {}
    rejection_reasons_by_index: dict[int, list[str]] = {}
    per_peer_reports: list[dict[str, Any]] = []

    for entry in prelim:
        idx = int(entry["index"])
        pid = str(entry["peer_id"])
        reasons: list[str] = []
        if not entry["all_finite"]:
            reasons.append("non_finite_values")
            cohort_ratio = None
            spike_ratio = None
        else:
            if not entry["sparsity_ok"]:
                reasons.append("edit_fraction_exceeded")
            if check_cohort_spectral_ratio and cohort_median_sv > tol:
                cohort_ratio = float(entry["max_singular_value"]) / cohort_median_sv
            else:
                cohort_ratio = 1.0
            intra_val = float(entry["intra_spectral_spike_ratio"])
            spike_ratio = max(intra_val, cohort_ratio)
            if spike_ratio > max_spike:
                reasons.append("spectral_spike_exceeded")

        passed = len(reasons) == 0
        if passed:
            accepted_indices.append(idx)
            accepted_peer_ids.append(pid)
        else:
            rejected_indices.append(idx)
            rejected_peer_ids.append(pid)
            rejection_reasons[pid] = list(reasons)
            rejection_reasons_by_index[idx] = list(reasons)

        per_peer_reports.append(
            {
                "index": idx,
                "peer_id": pid,
                "passed": passed,
                "all_finite": entry["all_finite"],
                "edit_fraction": entry["edit_fraction"],
                "edited_elements": entry["edited_elements"],
                "total_elements": entry["total_elements"],
                "max_singular_value": entry["max_singular_value"],
                "intra_spectral_spike_ratio": entry["intra_spectral_spike_ratio"],
                "cohort_spectral_spike_ratio": cohort_ratio,
                "spectral_spike_ratio": spike_ratio,
                "l2_norm": entry["l2_norm"],
                "offending_keys": list(entry["offending_keys"]),
                "reasons": list(reasons),
            }
        )

    if is_tensor_seq:
        surviving: list[dict[str, torch.Tensor]] | list[torch.Tensor] = [
            delta_maps[i]["delta"] for i in accepted_indices
        ]
    else:
        surviving = [dict(delta_maps[i]) for i in accepted_indices]

    report: dict[str, Any] = {
        "num_peers_in": len(delta_maps),
        "num_accepted": len(accepted_indices),
        "num_rejected": len(rejected_indices),
        "accepted_indices": accepted_indices,
        "rejected_indices": rejected_indices,
        "accepted_peer_ids": accepted_peer_ids,
        "rejected_peer_ids": rejected_peer_ids,
        "rejection_reasons": rejection_reasons,
        "rejection_reasons_by_index": rejection_reasons_by_index,
        "cohort_median_singular_value": cohort_median_sv,
        "max_edit_fraction": max_frac,
        "max_spectral_spike_ratio": max_spike,
        "per_peer": per_peer_reports,
    }
    return surviving, report


# ---------------------------------------------------------------------------
# 4. AlphaEdit General-Prose Null-Space Projector & Delta Projection
# ---------------------------------------------------------------------------


def _normalize_prose_key_matrix(
    prose_keys: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Flatten 2D/3D general-prose activation tensor(s) into CPU ``float64`` 2D."""
    tensors: list[torch.Tensor] = []

    def _to_2d(item: Any, label: str) -> torch.Tensor:
        if not isinstance(item, torch.Tensor):
            raise TypeError(f"{label} must be a torch.Tensor, got {type(item).__name__}")
        if item.ndim not in (2, 3) or any(dim == 0 for dim in item.shape):
            raise ValueError(f"{label} must be a non-empty 2D or 3D tensor, got shape {tuple(item.shape)}")
        if not item.is_floating_point():
            raise TypeError(f"{label} must have a floating-point dtype, got {item.dtype}")
        flat = item.detach().to(device="cpu", dtype=torch.float64).reshape(-1, int(item.shape[-1]))
        if not bool(torch.isfinite(flat).all().item()):
            raise ValueError(f"{label} must contain only finite values")
        return flat

    if isinstance(prose_keys, torch.Tensor):
        return _to_2d(prose_keys, "prose_keys").contiguous()
    if isinstance(prose_keys, Mapping):
        if not prose_keys:
            raise ValueError("prose_keys mapping must not be empty")
        for key in sorted(prose_keys.keys(), key=str):
            tensors.append(_to_2d(prose_keys[key], f"prose_keys[{key!r}]"))
    elif isinstance(prose_keys, Sequence) and not isinstance(prose_keys, (str, bytes, bytearray)):
        if not prose_keys:
            raise ValueError("prose_keys sequence must not be empty")
        for idx, item in enumerate(prose_keys):
            tensors.append(_to_2d(item, f"prose_keys[{idx}]"))
    else:
        raise TypeError(
            f"prose_keys must be a Tensor, Sequence[Tensor], or Mapping[str, Tensor]; "
            f"got {type(prose_keys).__name__}"
        )

    width = int(tensors[0].shape[1])
    for idx, t in enumerate(tensors[1:], start=1):
        if int(t.shape[1]) != width:
            raise ValueError(f"prose_keys[{idx}] width {t.shape[1]} must match first width {width}")
    return torch.cat(tensors, dim=0).contiguous()


@torch.no_grad()
def compute_alphaedit_prose_nullspace(
    prose_keys: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor],
    *,
    sv_threshold: float = DEFAULT_SV_THRESHOLD,
    variance_threshold: float | None = None,
    max_active_modes: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the exact AlphaEdit general-prose SVD null-space projector ``P_0 = I - V_active^T V_active``."""
    sv_thr = _validate_nonneg_finite(sv_threshold, "sv_threshold")
    var_thr = (
        _validate_unit_interval(variance_threshold, "variance_threshold", allow_zero=False)
        if variance_threshold is not None
        else None
    )
    max_m = _validate_positive_int(max_active_modes, "max_active_modes") if max_active_modes is not None else None

    keys_64 = _normalize_prose_key_matrix(prose_keys)
    d_key = int(keys_64.shape[1])
    _, s_vals, vh = torch.linalg.svd(keys_64, full_matrices=False)
    above = [int(i) for i in range(int(s_vals.numel())) if float(s_vals[i].item()) > sv_thr]
    eye_d = torch.eye(d_key, dtype=torch.float64)

    if not above:
        return eye_d, {
            "dim": d_key,
            "active_modes": 0,
            "nullspace_dim": d_key,
            "variance_captured": 0.0,
            "top_sv": float(s_vals[0].item()) if s_vals.numel() > 0 else 0.0,
        }

    total_energy = float(s_vals.pow(2).sum().item())
    selected: list[int] = []
    cum_energy = 0.0
    for idx in above:
        if max_m is not None and len(selected) >= max_m:
            break
        selected.append(idx)
        cum_energy += float(s_vals[idx].item()) ** 2
        if var_thr is not None and total_energy > 0.0 and (cum_energy / total_energy) >= var_thr:
            break

    v_active = vh[selected, :]
    p_null = eye_d - (v_active.T @ v_active)
    p_null = (0.5 * (p_null + p_null.T)).contiguous()
    return p_null, {
        "dim": d_key,
        "active_modes": len(selected),
        "nullspace_dim": d_key - len(selected),
        "variance_captured": (cum_energy / total_energy) if total_energy > 0.0 else 1.0,
        "top_sv": float(s_vals[0].item()),
    }


def _validate_square_projector(proj: Any, name: str) -> torch.Tensor:
    """Validate a non-empty finite square 2D projector tensor in CPU ``float64``."""
    if not isinstance(proj, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(proj).__name__}")
    if proj.ndim != 2 or proj.shape[0] == 0 or proj.shape[0] != proj.shape[1]:
        raise ValueError(f"{name} must be a non-empty square 2D matrix, got shape {tuple(proj.shape)}")
    if not proj.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {proj.dtype}")
    p64 = proj.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(p64).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return p64


def _resolve_projector_for_key(
    key: str,
    shape: tuple[int, ...],
    p_null: torch.Tensor | Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor | None, bool]:
    """Resolve the matching projector matrix and whether an exact key match was requested."""
    if isinstance(p_null, torch.Tensor):
        return _validate_square_projector(p_null, "p_null"), False

    if key in p_null:
        return _validate_square_projector(p_null[key], f"p_null[{key!r}]"), True

    layer_match = _LAYER_KEY_RE.search(key)
    layer_tag = f"blocks.{layer_match.group(1)}" if layer_match is not None else None
    if key.endswith(("w1_B", "w2_B")):
        return None, False
    is_act = key.endswith("w3_A") or key.endswith(".w3.weight") or key == "w3"

    candidates: list[str] = []
    if layer_tag is not None:
        candidates.append(f"{layer_tag}.act" if is_act else f"{layer_tag}.x")
    candidates.append("act" if is_act else "x")
    candidates.append("default")

    for cand in candidates:
        if cand in p_null:
            p64 = _validate_square_projector(p_null[cand], f"p_null[{cand!r}]")
            d_p = int(p64.shape[0])
            if len(shape) == 1 and shape[0] == d_p:
                return p64, False
            if len(shape) >= 2 and (shape[-1] == d_p or (len(shape) == 2 and shape[0] == d_p)):
                return p64, False
    return None, False


def _is_layernorm_key(key: str) -> bool:
    low = key.lower()
    return any(marker in low for marker in _LAYERNORM_MARKERS)


def _is_general_expert_key(key: str, general_expert_index: int) -> bool:
    m = _EXPERT_KEY_RE.search(key)
    if m is None:
        return False
    idx_str = m.group(1) if m.group(1) is not None else m.group(2)
    return int(idx_str) == general_expert_index


@torch.no_grad()
def project_delta_into_nullspace(
    delta: Mapping[str, torch.Tensor] | torch.Tensor,
    p_null: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    protect_layernorm: bool = True,
    protect_general_expert: bool = False,
    general_expert_index: int = 0,
) -> tuple[dict[str, torch.Tensor] | torch.Tensor, dict[str, Any]]:
    """Project a consensus delta mapping or tensor strictly into AlphaEdit null-space ``P_0``."""
    _validate_bool(protect_layernorm, "protect_layernorm")
    _validate_bool(protect_general_expert, "protect_general_expert")
    gen_idx = _validate_nonneg_int(general_expert_index, "general_expert_index")

    is_tensor_input = isinstance(delta, torch.Tensor)
    if is_tensor_input:
        delta_map: Mapping[str, torch.Tensor] = {"delta": delta}
    elif isinstance(delta, Mapping) and delta:
        delta_map = delta
    else:
        raise TypeError("delta must be a torch.Tensor or non-empty Mapping[str, torch.Tensor]")

    if not isinstance(p_null, (torch.Tensor, Mapping)) or (isinstance(p_null, Mapping) and not p_null):
        raise TypeError("p_null must be a square 2D Tensor or non-empty Mapping[str, Tensor]")

    projected_out: dict[str, torch.Tensor] = {}
    projected_keys: list[str] = []
    skipped_keys: list[str] = []
    raw_norm_sq = 0.0
    proj_norm_sq = 0.0

    for key in sorted(delta_map.keys()):
        val = delta_map[key]
        if not isinstance(val, torch.Tensor) or not val.is_floating_point() or val.numel() == 0:
            raise ValueError(f"delta[{key!r}] must be a non-empty floating-point torch.Tensor")
        t64 = val.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(t64).all().item()):
            raise ValueError(f"delta[{key!r}] contains non-finite values")
        raw_norm_sq += float((t64 * t64).sum().item())

        if (protect_layernorm and _is_layernorm_key(key)) or (
            protect_general_expert and _is_general_expert_key(key, gen_idx)
        ):
            projected_out[key] = val.detach().clone()
            skipped_keys.append(key)
            proj_norm_sq += float((t64 * t64).sum().item())
            continue

        proj_mat, exact_match = _resolve_projector_for_key(key, tuple(t64.shape), p_null)
        if proj_mat is None:
            projected_out[key] = val.detach().clone()
            skipped_keys.append(key)
            proj_norm_sq += float((t64 * t64).sum().item())
            continue

        d_p = int(proj_mat.shape[0])
        if t64.ndim == 1 and int(t64.shape[0]) == d_p:
            res64 = t64 @ proj_mat
            projected_keys.append(key)
        elif t64.ndim >= 2 and int(t64.shape[-1]) == d_p:
            if protect_general_expert and "router" in key.lower() and t64.ndim == 2 and gen_idx < int(t64.shape[0]):
                res64 = t64.clone()
                mask = torch.ones(int(t64.shape[0]), dtype=torch.bool)
                mask[gen_idx] = False
                res64[mask] = t64[mask] @ proj_mat
            else:
                res64 = t64 @ proj_mat
            projected_keys.append(key)
        elif t64.ndim == 2 and int(t64.shape[0]) == d_p:
            res64 = proj_mat @ t64
            projected_keys.append(key)
        else:
            if exact_match:
                raise ValueError(
                    f"Projector shape {tuple(proj_mat.shape)} is incompatible with delta[{key!r}] shape {tuple(t64.shape)}"
                )
            res64 = t64
            skipped_keys.append(key)

        proj_norm_sq += float((res64 * res64).sum().item())
        projected_out[key] = res64.to(device=val.device, dtype=val.dtype).contiguous()

    raw_norm = math.sqrt(raw_norm_sq)
    proj_norm = math.sqrt(proj_norm_sq)
    meta = {
        "projection_applied": bool(projected_keys),
        "projected_keys": projected_keys,
        "skipped_keys": skipped_keys,
        "raw_l2_norm": raw_norm,
        "projected_l2_norm": proj_norm,
        "retained_norm_ratio": (proj_norm / raw_norm) if raw_norm > 1e-15 else 1.0,
    }
    if is_tensor_input:
        return projected_out["delta"], meta
    return projected_out, meta


# ---------------------------------------------------------------------------
# 5. Pairwise Euclidean Multi-Krum & Null-Space Consensus Aggregator
# ---------------------------------------------------------------------------


def _normalize_peer_weights(peer_weights: Sequence[float] | None, n_peers: int) -> list[float]:
    """Validate and normalize positive finite peer weights."""
    if peer_weights is None:
        return [1.0 / float(n_peers)] * n_peers
    if isinstance(peer_weights, (str, bytes, bytearray)) or not isinstance(peer_weights, Sequence):
        raise TypeError("peer_weights must be a sequence of positive numbers or None")
    if len(peer_weights) != n_peers:
        raise ValueError(f"peer_weights length ({len(peer_weights)}) must match peer count ({n_peers})")
    vals = [_validate_positive_finite(w, f"peer_weights[{idx}]") for idx, w in enumerate(peer_weights)]
    total = sum(vals)
    return [v / total for v in vals]


@torch.no_grad()
def multi_krum_nullspace_aggregate(
    peer_deltas: Sequence[Mapping[str, torch.Tensor] | torch.Tensor]
    | Mapping[str, Mapping[str, torch.Tensor]],
    *,
    num_byzantine_f: int,
    select_m: int | None = None,
    p_null: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    peer_weights: Sequence[float] | None = None,
    enforce_byzantine_bound: bool = True,
    protect_layernorm: bool = True,
    protect_general_expert: bool = False,
    general_expert_index: int = 0,
) -> tuple[dict[str, torch.Tensor] | torch.Tensor, dict[str, Any]]:
    """Filter up to ``f < (n - 2)/2`` Byzantine deltas via Multi-Krum and project into ``P_0``."""
    delta_maps, resolved_ids, ref_keys, is_tensor_seq = _normalize_peer_delta_collection(peer_deltas)
    n = len(delta_maps)
    f_val = _validate_nonneg_int(num_byzantine_f, "num_byzantine_f")
    _validate_bool(enforce_byzantine_bound, "enforce_byzantine_bound")

    if enforce_byzantine_bound:
        if n < 3 or 2 * f_val >= n - 2:
            raise ValueError(
                f"Multi-Krum requires f < (n - 2) / 2 (n >= 2*f + 3); got n={n}, num_byzantine_f={f_val}"
            )
    eff_f = min(f_val, max(0, (n - 3) // 2)) if n >= 3 else 0

    if select_m is None:
        m_val = max(1, n - eff_f)
    else:
        m_val = _validate_positive_int(select_m, "select_m")
        if m_val > n:
            raise ValueError(f"select_m ({m_val}) cannot exceed peer count ({n})")

    norm_weights = _normalize_peer_weights(peer_weights, n)

    flat_peers: list[torch.Tensor] = []
    for idx, p_map in enumerate(delta_maps):
        parts: list[torch.Tensor] = []
        for key in ref_keys:
            t64 = p_map[key].detach().to(device="cpu", dtype=torch.float64).reshape(-1)
            if not bool(torch.isfinite(t64).all().item()):
                raise ValueError(f"peer_deltas[{idx}][{key!r}] contains non-finite values")
            parts.append(t64)
        flat_peers.append(torch.cat(parts, dim=0))

    mat = torch.stack(flat_peers, dim=0)  # (n, D)
    dist_sq = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        diffs = mat - mat[i].unsqueeze(0)
        dist_sq[i] = (diffs * diffs).sum(dim=1)
        dist_sq[i, i] = float("inf")

    k_neighbors = max(1, min(n - 1, n - eff_f - 2)) if n > 1 else 0
    krum_distances: list[float] = []
    for i in range(n):
        if k_neighbors == 0:
            krum_distances.append(0.0)
        else:
            nearest, _ = torch.topk(dist_sq[i], k=k_neighbors, largest=False)
            krum_distances.append(float(nearest.sum().item()))

    ranked = sorted(range(n), key=lambda idx: (krum_distances[idx], idx))
    selected_indices = sorted(ranked[:m_val])
    rejected_indices = sorted(ranked[m_val:])

    sel_weights = [norm_weights[i] for i in selected_indices]
    sel_total = sum(sel_weights)
    consensus_map: dict[str, torch.Tensor] = {}
    for key in ref_keys:
        ref_t = delta_maps[0][key]
        acc = torch.zeros_like(ref_t, device="cpu", dtype=torch.float64)
        for idx, w in zip(selected_indices, sel_weights, strict=True):
            acc.add_(delta_maps[idx][key].detach().to(device="cpu", dtype=torch.float64), alpha=w / sel_total)
        consensus_map[key] = acc.to(device=ref_t.device, dtype=ref_t.dtype).contiguous()

    if p_null is not None:
        projected_map, proj_meta = project_delta_into_nullspace(
            consensus_map,
            p_null,
            protect_layernorm=protect_layernorm,
            protect_general_expert=protect_general_expert,
            general_expert_index=general_expert_index,
        )
        assert isinstance(projected_map, dict)
        consensus_map = projected_map
    else:
        proj_meta = {
            "projection_applied": False,
            "projected_keys": [],
            "skipped_keys": list(ref_keys),
            "raw_l2_norm": float(sum((t.to(torch.float64).pow(2).sum().item()) for t in consensus_map.values()) ** 0.5),
            "projected_l2_norm": float(
                sum((t.to(torch.float64).pow(2).sum().item()) for t in consensus_map.values()) ** 0.5
            ),
            "retained_norm_ratio": 1.0,
        }

    meta: dict[str, Any] = {
        "num_peers": n,
        "num_byzantine_f": eff_f,
        "select_m": m_val,
        "k_neighbors": k_neighbors,
        "selected_indices": selected_indices,
        "rejected_indices": rejected_indices,
        "selected_peer_ids": [resolved_ids[i] for i in selected_indices],
        "rejected_peer_ids": [resolved_ids[i] for i in rejected_indices],
        "krum_distances": krum_distances,
        "krum_distances_by_peer": {resolved_ids[i]: krum_distances[i] for i in range(n)},
        **proj_meta,
    }
    if is_tensor_seq:
        return consensus_map["delta"], meta
    return consensus_map, meta


# ---------------------------------------------------------------------------
# 6. End-to-End Verified Byzantine WSP Round (MoE Buckets + ANSE Guard)
# ---------------------------------------------------------------------------


def _unpack_bucket_deltas_to_param_map(
    model: nn.Module,
    bucket_delta: Mapping[str, torch.Tensor],
    *,
    num_router_fragments: int,
) -> dict[str, torch.Tensor]:
    """Unpack OpenDiLoCo flat bucket deltas into named ``state_dict`` parameter tensors."""
    blocks = getattr(model, "blocks", ())
    n_blocks = len(blocks)
    frag_span = max(1, math.ceil(n_blocks / max(1, num_router_fragments)))
    router_offsets: dict[str, int] = {k: 0 for k in bucket_delta if k.startswith("router_frag_")}
    expert_offsets: dict[str, int] = {k: 0 for k in bucket_delta if k.startswith("expert_")}
    param_map: dict[str, torch.Tensor] = {}

    for l_idx, blk in enumerate(blocks):
        f_idx = min(num_router_fragments - 1, l_idx // frag_span)
        r_key = f"router_frag_{f_idx}"
        mlp = getattr(blk, "mlp", None)
        if isinstance(mlp, (CrowdLoRAMoEMLP, CrowdFineGrainedMoEMLP)):
            if mlp.router.weight.requires_grad and r_key in bucket_delta:
                n = mlp.router.weight.numel()
                off = router_offsets[r_key]
                if off + n > bucket_delta[r_key].numel():
                    raise ValueError(
                        f"Bucket {r_key!r} length {bucket_delta[r_key].numel()} smaller than model layout ({off + n})"
                    )
                param_map[f"blocks.{l_idx}.mlp.router.weight"] = (
                    bucket_delta[r_key][off : off + n].view_as(mlp.router.weight).contiguous()
                )
                router_offsets[r_key] = off + n

        if isinstance(mlp, CrowdLoRAMoEMLP):
            for e_key, exp_mod in sorted(mlp.experts.items(), key=lambda kv: int(kv[0])):
                b_name = f"expert_{int(e_key)}"
                if b_name not in bucket_delta:
                    continue
                for p_name, p in exp_mod.named_parameters():
                    if not p.requires_grad:
                        continue
                    n = p.numel()
                    off = expert_offsets[b_name]
                    if off + n > bucket_delta[b_name].numel():
                        raise ValueError(
                            f"Bucket {b_name!r} length {bucket_delta[b_name].numel()} smaller than model layout ({off + n})"
                        )
                    param_map[f"blocks.{l_idx}.mlp.experts.{int(e_key)}.{p_name}"] = (
                        bucket_delta[b_name][off : off + n].view_as(p).contiguous()
                    )
                    expert_offsets[b_name] = off + n
        elif isinstance(mlp, CrowdFineGrainedMoEMLP):
            for e_idx in range(mlp.num_experts):
                b_name = f"expert_{e_idx}"
                if b_name not in bucket_delta:
                    continue
                for attr_name in ("experts_w1", "experts_w2", "experts_w3"):
                    mod_list = getattr(mlp, attr_name)
                    for p_name, p in mod_list[e_idx].named_parameters():
                        if not p.requires_grad:
                            continue
                        n = p.numel()
                        off = expert_offsets[b_name]
                        if off + n > bucket_delta[b_name].numel():
                            raise ValueError(
                                f"Bucket {b_name!r} length {bucket_delta[b_name].numel()} smaller than model layout ({off + n})"
                            )
                        param_map[f"blocks.{l_idx}.mlp.{attr_name}.{e_idx}.{p_name}"] = (
                            bucket_delta[b_name][off : off + n].view_as(p).contiguous()
                        )
                        expert_offsets[b_name] = off + n

    last_r_key = f"router_frag_{num_router_fragments - 1}"
    if last_r_key in bucket_delta:
        for ln_name, p in (("ln.weight", model.ln.weight), ("ln.bias", model.ln.bias)):
            if p.requires_grad:
                n = p.numel()
                off = router_offsets[last_r_key]
                if off + n > bucket_delta[last_r_key].numel():
                    raise ValueError(
                        f"Bucket {last_r_key!r} length {bucket_delta[last_r_key].numel()} smaller than model layout ({off + n})"
                    )
                param_map[ln_name] = bucket_delta[last_r_key][off : off + n].view_as(p).contiguous()
                router_offsets[last_r_key] = off + n

    for r_key, off in router_offsets.items():
        if off != bucket_delta[r_key].numel():
            raise ValueError(f"Bucket {r_key!r} length {bucket_delta[r_key].numel()} != model layout {off}")
    for e_key, off in expert_offsets.items():
        if off != bucket_delta[e_key].numel():
            raise ValueError(f"Bucket {e_key!r} length {bucket_delta[e_key].numel()} != model layout {off}")
    return param_map


@torch.no_grad()
def _collect_prose_projectors_from_probe(
    model: nn.Module,
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    general_prefix: str,
    sv_threshold: float,
    variance_threshold: float | None,
    max_active_modes: int | None,
    score_from: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Collect per-layer AlphaEdit ``P_0`` projectors from ``E0_*`` pre-MLP and SwiGLU states."""
    e0_chunks = [c for c in probe_chunks if str(c["expert_id"]).startswith(general_prefix)]
    if not e0_chunks:
        raise ValueError(f"probe_chunks must contain at least one {general_prefix!r} chunk")
    ids_e0 = torch.stack([torch.as_tensor(ch["ids"], dtype=torch.long) for ch in e0_chunks], dim=0)
    start_pos = max(0, min(int(ids_e0.shape[1]) - 2, score_from - 1))
    cos = model.cos_cached.to(ids_e0.device)
    sin = model.sin_cached.to(ids_e0.device)

    x_seq = torch.cat(
        [compute_embeddings_batch(model, ids_e0[b0 : b0 + batch_size]) for b0 in range(0, ids_e0.shape[0], batch_size)],
        dim=0,
    )
    projectors: dict[str, torch.Tensor] = {}
    for l_idx, blk in enumerate(model.blocks):
        xm_l: list[torch.Tensor] = []
        mi_l: list[torch.Tensor] = []
        for b0 in range(0, x_seq.shape[0], batch_size):
            xm, mi = forward_block_pre_mlp(blk, x_seq[b0 : b0 + batch_size], cos, sin)
            xm_l.append(xm)
            mi_l.append(mi)
        x_mid = torch.cat(xm_l, dim=0)
        mlp_in = torch.cat(mi_l, dim=0)
        scored_x = mlp_in[:, start_pos:-1, :].reshape(-1, int(mlp_in.shape[-1]))
        p_x, _ = compute_alphaedit_prose_nullspace(
            scored_x,
            sv_threshold=sv_threshold,
            variance_threshold=variance_threshold,
            max_active_modes=max_active_modes,
        )
        projectors[f"blocks.{l_idx}.x"] = p_x

        if isinstance(blk.mlp, CrowdLoRAMoEMLP) and blk.mlp.experts:
            active_exp = sorted(int(k) for k in blk.mlp.experts.keys())
            act_prose = compute_swiglu_intermediate_states(blk.mlp, scored_x, active_exp)
            p_act, _ = compute_alphaedit_prose_nullspace(
                act_prose,
                sv_threshold=sv_threshold,
                variance_threshold=variance_threshold,
                max_active_modes=max_active_modes,
            )
            projectors[f"blocks.{l_idx}.act"] = p_act

        x_seq = torch.cat(
            [
                forward_block_post_mlp(blk, x_mid[b0 : b0 + batch_size], mlp_in[b0 : b0 + batch_size])
                for b0 in range(0, x_mid.shape[0], batch_size)
            ],
            dim=0,
        )
    return projectors


@torch.no_grad()
def _evaluate_model_domains(
    model: nn.Module,
    *,
    eval_chunks: Sequence[Mapping[str, Any]] | None,
    eval_fn: Callable[[nn.Module], Mapping[str, Any]] | None,
    score_from: int,
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate per-domain finite NLLs via custom ``eval_fn`` or ``CrowdV1WithEngram`` forward pass."""
    if eval_fn is not None:
        raw = eval_fn(model)
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("eval_fn must return a non-empty mapping")
        dom_source = raw.get("domain_nlls", raw)
        if not isinstance(dom_source, Mapping) or not dom_source:
            raise ValueError("eval_fn result must include non-empty domain NLL entries")
        dom_map = {
            str(k): float(v)
            for k, v in dom_source.items()
            if str(k) not in {"non_finite_outputs", "nll_mean", "all_finite"}
        }
        if not dom_map:
            raise ValueError("eval_fn result must include at least one domain NLL")
        nf_cnt = int(raw.get("non_finite_outputs", 0))
        all_finite = nf_cnt == 0 and all(math.isfinite(v) for v in dom_map.values())
        nll_mean = (
            float(raw["nll_mean"])
            if ("nll_mean" in raw and math.isfinite(float(raw["nll_mean"])))
            else (float(np.mean(list(dom_map.values()))) if all_finite else None)
        )
        return {
            "domain_nlls": {k: (v if math.isfinite(v) else None) for k, v in dom_map.items()},
            "all_finite": all_finite,
            "nll_mean": nll_mean,
        }

    assert eval_chunks is not None
    ids_all = torch.stack([torch.as_tensor(ch["ids"], dtype=torch.long) for ch in eval_chunks], dim=0)
    domains = [str(ch["expert_id"]) for ch in eval_chunks]
    h_pre = forward_model_pre_ln(model, ids_all, batch_size=batch_size)
    if not bool(torch.isfinite(h_pre).all().item()):
        return {
            "domain_nlls": {d: None for d in sorted(set(domains))},
            "all_finite": False,
            "nll_mean": None,
        }
    ln_bias = getattr(model.ln, "bias", None)
    ln_b = ln_bias if isinstance(ln_bias, torch.Tensor) else torch.zeros_like(model.ln.weight)
    nlls = per_chunk_nll_from_pre_ln(
        h_pre,
        model.ln.weight,
        ln_b,
        float(getattr(model.ln, "eps", getattr(model.ln, "variance_epsilon", 1e-5))),
        model.head.weight,
        ids_all,
        score_from=score_from,
    )
    if not bool(torch.isfinite(nlls).all().item()):
        return {
            "domain_nlls": {d: None for d in sorted(set(domains))},
            "all_finite": False,
            "nll_mean": None,
        }
    nlls_np = nlls.detach().cpu().numpy().astype(np.float64)
    grouped: dict[str, list[float]] = {}
    for dom, val in zip(domains, nlls_np.tolist(), strict=True):
        grouped.setdefault(dom, []).append(float(val))
    dom_nlls = {dom: float(np.mean(vals)) for dom, vals in sorted(grouped.items())}
    return {
        "domain_nlls": dom_nlls,
        "all_finite": True,
        "nll_mean": float(nlls_np.mean()),
    }


def _resolve_round_config(
    config: ByzantineWSPConfig | None,
    *,
    num_byzantine_f: int | None,
    select_m: int | None,
    max_edit_fraction: float | None,
    max_spectral_spike_ratio: float | None,
) -> ByzantineWSPConfig:
    if config is not None and not isinstance(config, ByzantineWSPConfig):
        raise TypeError(f"config must be a ByzantineWSPConfig or None, got {type(config).__name__}")
    base = config or ByzantineWSPConfig()
    return ByzantineWSPConfig(
        num_byzantine_f=base.num_byzantine_f if num_byzantine_f is None else num_byzantine_f,
        select_m=base.select_m if select_m is None else select_m,
        max_edit_fraction=base.max_edit_fraction if max_edit_fraction is None else max_edit_fraction,
        max_spectral_spike_ratio=(
            base.max_spectral_spike_ratio if max_spectral_spike_ratio is None else max_spectral_spike_ratio
        ),
        edit_tol=base.edit_tol,
        enable_screening=base.enable_screening,
        check_cohort_spectral_ratio=base.check_cohort_spectral_ratio,
        check_intra_matrix_spectral_ratio=base.check_intra_matrix_spectral_ratio,
        sv_threshold=base.sv_threshold,
        prose_variance_threshold=base.prose_variance_threshold,
        prose_max_active_modes=base.prose_max_active_modes,
        protect_layernorm=base.protect_layernorm,
        protect_general_expert=base.protect_general_expert,
        general_expert_index=base.general_expert_index,
        domain_guard_max_ratio=base.domain_guard_max_ratio,
        anse_safe_general_ratio=base.anse_safe_general_ratio,
        maximum_pain_energy=base.maximum_pain_energy,
        soft_margin_penalty=base.soft_margin_penalty,
        enforce_all_domain_guard=base.enforce_all_domain_guard,
        require_strict_energy_drop=base.require_strict_energy_drop,
        general_prefix=base.general_prefix,
        score_from=base.score_from,
        batch_size=base.batch_size,
    )


def _restore_snapshot_bitwise(model: nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
    """Restore ``model.state_dict()`` in-place from CPU snapshot preserving devices, dtypes, and tying."""
    cur_sd = model.state_dict()
    for k, v in snapshot.items():
        cur_sd[k].copy_(v.to(device=cur_sd[k].device, dtype=cur_sd[k].dtype))


def _validate_eval_chunks_schema(chunks: Sequence[Mapping[str, Any]], name: str) -> None:
    if isinstance(chunks, (str, bytes, bytearray)) or not isinstance(chunks, Sequence) or not chunks:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")
    for idx, ch in enumerate(chunks):
        if not isinstance(ch, Mapping) or "expert_id" not in ch or "ids" not in ch:
            raise ValueError(f"{name}[{idx}] must be a mapping containing 'expert_id' and 'ids'")


@torch.no_grad()
def apply_verified_byzantine_wsp_round(
    model: nn.Module,
    peer_deltas: Sequence[Mapping[str, torch.Tensor]] | Mapping[str, Mapping[str, torch.Tensor]],
    *,
    probe_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    eval_fn: Callable[[nn.Module], Mapping[str, Any]] | None = None,
    p_null: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    prose_keys: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor] | None = None,
    num_byzantine_f: int | None = None,
    select_m: int | None = None,
    max_edit_fraction: float | None = None,
    max_spectral_spike_ratio: float | None = None,
    peer_weights: Sequence[float] | None = None,
    config: ByzantineWSPConfig | None = None,
    delta_format: str = "auto",
    num_router_fragments: int | None = None,
) -> dict[str, Any]:
    """Execute one Byzantine-resilient WSP aggregation round with ANSE gating and atomic rollback."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    model_sd = model.state_dict()
    if not model_sd:
        raise ValueError("model state_dict must be non-empty")
    if delta_format not in VALID_DELTA_FORMATS:
        raise ValueError(f"delta_format must be one of {sorted(VALID_DELTA_FORMATS)}, got {delta_format!r}")
    if probe_chunks is None and eval_chunks is None and eval_fn is None:
        raise ValueError("At least one of probe_chunks, eval_chunks, or eval_fn must be provided")
    if probe_chunks is not None:
        _validate_eval_chunks_schema(probe_chunks, "probe_chunks")
    if eval_chunks is not None:
        _validate_eval_chunks_schema(eval_chunks, "eval_chunks")

    cfg = _resolve_round_config(
        config,
        num_byzantine_f=num_byzantine_f,
        select_m=select_m,
        max_edit_fraction=max_edit_fraction,
        max_spectral_spike_ratio=max_spectral_spike_ratio,
    )

    delta_maps, resolved_ids, ref_keys, is_tensor_seq = _normalize_peer_delta_collection(peer_deltas)
    if is_tensor_seq:
        raise TypeError("apply_verified_byzantine_wsp_round requires named parameter or bucket delta mappings")
    n_initial = len(delta_maps)
    if n_initial < 3 or 2 * cfg.num_byzantine_f >= n_initial - 2:
        raise ValueError(
            f"Multi-Krum requires f < (n - 2) / 2 (n >= 2*f + 3); got n={n_initial}, num_byzantine_f={cfg.num_byzantine_f}"
        )
    if cfg.select_m is not None and cfg.select_m > n_initial:
        raise ValueError(f"select_m ({cfg.select_m}) cannot exceed peer count ({n_initial})")
    norm_weights_all = _normalize_peer_weights(peer_weights, n_initial)

    resolved_format = delta_format
    if resolved_format == "auto":
        is_bucket_like = all(
            k.startswith("router_frag_") or k.startswith("expert_") for k in ref_keys
        ) and any(k not in model_sd for k in ref_keys)
        resolved_format = "buckets" if is_bucket_like else "state_dict"

    if resolved_format == "buckets":
        nfrag = (
            _validate_positive_int(num_router_fragments, "num_router_fragments")
            if num_router_fragments is not None
            else max(1, len([k for k in ref_keys if k.startswith("router_frag_")]))
        )
        param_maps = [
            _unpack_bucket_deltas_to_param_map(model, dm, num_router_fragments=nfrag) for dm in delta_maps
        ]
    else:
        for k in ref_keys:
            if k not in model_sd:
                raise ValueError(f"Delta parameter key {k!r} not found in model.state_dict()")
            if tuple(delta_maps[0][k].shape) != tuple(model_sd[k].shape):
                raise ValueError(
                    f"Delta shape {tuple(delta_maps[0][k].shape)} for {k!r} != model shape {tuple(model_sd[k].shape)}"
                )
        param_maps = delta_maps

    verify_chunks = eval_chunks if eval_chunks is not None else probe_chunks
    was_training = model.training
    model.eval()
    try:
        parent_snapshot = {k: v.detach().to(device="cpu").clone() for k, v in model.state_dict().items()}
        parent_sha = compute_state_sha256(parent_snapshot)

        parent_eval = _evaluate_model_domains(
            model,
            eval_chunks=verify_chunks,
            eval_fn=eval_fn,
            score_from=cfg.score_from,
            batch_size=cfg.batch_size,
        )
        if not parent_eval["all_finite"] or parent_eval["nll_mean"] is None:
            raise ValueError("Parent model evaluation produced non-finite NLL")
        parent_dom_nlls: dict[str, float] = {k: float(v) for k, v in parent_eval["domain_nlls"].items()}
        parent_nll_mean = float(parent_eval["nll_mean"])
        parent_anse_energy = parent_nll_mean

        # Step 1: Pre-aggregation spectral spike & sparsity screening
        if cfg.enable_screening:
            surviving_maps, screen_report = screen_peer_wsp_deltas_spectral_and_sparsity(
                param_maps,
                max_edit_fraction=cfg.max_edit_fraction,
                max_spectral_spike_ratio=cfg.max_spectral_spike_ratio,
                edit_tol=cfg.edit_tol,
                peer_ids=resolved_ids,
                check_cohort_spectral_ratio=cfg.check_cohort_spectral_ratio,
                check_intra_matrix_spectral_ratio=cfg.check_intra_matrix_spectral_ratio,
            )
        else:
            surviving_maps = list(param_maps)
            screen_report = {
                "num_peers_in": n_initial,
                "num_accepted": n_initial,
                "num_rejected": 0,
                "accepted_indices": list(range(n_initial)),
                "rejected_indices": [],
                "accepted_peer_ids": list(resolved_ids),
                "rejected_peer_ids": [],
                "rejection_reasons": {},
                "rejection_reasons_by_index": {},
            }

        n_surviving = len(surviving_maps)
        n_screen_rejected = int(screen_report["num_rejected"])
        eff_f = max(0, cfg.num_byzantine_f - n_screen_rejected)
        quorum_ok = n_surviving >= 3 and (2 * eff_f < n_surviving - 2)

        if not quorum_ok:
            final_sha = compute_state_sha256(model)
            cand_anse_energy = float(cfg.maximum_pain_energy)
            delta_e = cand_anse_energy - parent_anse_energy
            receipt: dict[str, Any] = {
                "object": RECEIPT_OBJECT,
                "schema": SCHEMA_VERSION,
                "accepted": False,
                "rolled_back": True,
                "rollback_bitwise_verified": final_sha == parent_sha,
                "status": "ROLLED_BACK_QUORUM_BELOW_BYZANTINE_BOUND",
                "delta_format": resolved_format,
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
                "parent_anse_energy": parent_anse_energy,
                "candidate_anse_energy": cand_anse_energy,
                "final_anse_energy": parent_anse_energy,
                "delta_e": float(delta_e),
                "screening_report": screen_report,
                "krum_report": {"selected_indices": [], "rejected_indices": [], "krum_distances": []},
                "consensus_delta_sha256": "",
                "parent_state_sha256": parent_sha,
                "candidate_state_sha256": parent_sha,
                "final_state_sha256": final_sha,
                "config": asdict(cfg),
            }
            receipt["receipt_sha256"] = compute_payload_sha256(receipt)
            return receipt

        # Step 2: Resolve AlphaEdit general-prose null-space projector(s) P_0
        resolved_p_null: torch.Tensor | Mapping[str, torch.Tensor] | None = p_null
        if resolved_p_null is None and prose_keys is not None:
            resolved_p_null, _ = compute_alphaedit_prose_nullspace(
                prose_keys,
                sv_threshold=cfg.sv_threshold,
                variance_threshold=cfg.prose_variance_threshold,
                max_active_modes=cfg.prose_max_active_modes,
            )
        elif resolved_p_null is None and probe_chunks is not None and hasattr(model, "blocks") and hasattr(model, "cos_cached"):
            resolved_p_null = _collect_prose_projectors_from_probe(
                model,
                probe_chunks,
                general_prefix=cfg.general_prefix,
                sv_threshold=cfg.sv_threshold,
                variance_threshold=cfg.prose_variance_threshold,
                max_active_modes=cfg.prose_max_active_modes,
                score_from=cfg.score_from,
                batch_size=cfg.batch_size,
            )

        surviving_orig_indices: list[int] = list(screen_report["accepted_indices"])
        surviving_weights = [norm_weights_all[i] for i in surviving_orig_indices]
        eff_select_m = (
            min(cfg.select_m, n_surviving) if cfg.select_m is not None else max(1, n_surviving - eff_f)
        )

        # Step 3: Multi-Krum selection & projection into AlphaEdit P_0
        consensus_delta, krum_meta = multi_krum_nullspace_aggregate(
            surviving_maps,  # type: ignore[arg-type]
            num_byzantine_f=eff_f,
            select_m=eff_select_m,
            p_null=resolved_p_null,
            peer_weights=surviving_weights,
            enforce_byzantine_bound=True,
            protect_layernorm=cfg.protect_layernorm,
            protect_general_expert=cfg.protect_general_expert,
            general_expert_index=cfg.general_expert_index,
        )
        assert isinstance(consensus_delta, dict)
        # Map Multi-Krum indices back to original peer_deltas indices for full provenance
        krum_meta["selected_original_indices"] = [
            surviving_orig_indices[i] for i in krum_meta["selected_indices"]
        ]
        krum_meta["rejected_original_indices"] = sorted(
            set(range(n_initial)) - set(krum_meta["selected_original_indices"])
        )
        krum_meta["selected_peer_ids"] = [resolved_ids[i] for i in krum_meta["selected_original_indices"]]
        krum_meta["rejected_peer_ids"] = [resolved_ids[i] for i in krum_meta["rejected_original_indices"]]

        consensus_sha = compute_state_sha256(consensus_delta)

        # Step 4: Apply consensus edit in-place and evaluate ANSE guard + Delta E
        cur_sd = model.state_dict()
        for k, d_tensor in consensus_delta.items():
            new_val = (
                cur_sd[k].detach().to(torch.float64)
                + d_tensor.detach().to(device=cur_sd[k].device, dtype=torch.float64)
            ).to(dtype=cur_sd[k].dtype)
            cur_sd[k].copy_(new_val)

        candidate_sha = compute_state_sha256(model)
        cand_eval = _evaluate_model_domains(
            model,
            eval_chunks=verify_chunks,
            eval_fn=eval_fn,
            score_from=cfg.score_from,
            batch_size=cfg.batch_size,
        )

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
            if dom.startswith(cfg.general_prefix):
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

        delta_e = cand_anse_energy - parent_anse_energy
        energy_ok = (delta_e < 0.0) if cfg.require_strict_energy_drop else (delta_e <= 0.0)

        if candidate_sha == parent_sha:
            accepted = False
            rolled_back = False
            status = "FIXED_POINT_UNCHANGED" if guard_ok else "REJECTED_GUARD_VIOLATION_UNCHANGED"
        elif all_finite and guard_ok and energy_ok:
            accepted = True
            rolled_back = False
            status = "ACCEPTED"
        else:
            accepted = False
            rolled_back = True
            status = (
                "ROLLED_BACK_NON_FINITE"
                if not all_finite
                else ("ROLLED_BACK_GUARD_VIOLATION" if not guard_ok else "ROLLED_BACK_ENERGY_NON_IMPROVING")
            )
            _restore_snapshot_bitwise(model, parent_snapshot)

        final_sha = compute_state_sha256(model)
        if not accepted and final_sha != parent_sha:
            raise RuntimeError(f"Atomic bitwise rollback failed: {final_sha[:12]} != {parent_sha[:12]}")
    finally:
        model.train(was_training)

    receipt = {
        "object": RECEIPT_OBJECT,
        "schema": SCHEMA_VERSION,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "rollback_bitwise_verified": bool((not accepted) and final_sha == parent_sha),
        "status": status,
        "delta_format": resolved_format,
        "guard_ok": guard_ok,
        "general_guard_ok": general_guard_ok,
        "nll_mean": float(cand_nll_mean) if (accepted and cand_nll_mean is not None) else parent_nll_mean,
        "parent_nll_mean": parent_nll_mean,
        "candidate_nll_mean": cand_nll_mean,
        "general_drift_pct": 100.0 * (general_ratio - 1.0) if all_finite else None,
        "worst_domain_drift_pct": 100.0 * (worst_ratio - 1.0) if all_finite else None,
        "per_domain_nll_parent": parent_dom_nlls,
        "per_domain_nll_candidate": {k: (float(v) if v is not None else None) for k, v in cand_dom_raw.items()},
        "per_domain_drift_pct": per_domain_drift_pct,
        "anse_energy": float(cand_anse_energy),
        "parent_anse_energy": float(parent_anse_energy),
        "candidate_anse_energy": float(cand_anse_energy),
        "final_anse_energy": float(cand_anse_energy if accepted else parent_anse_energy),
        "delta_e": float(delta_e),
        "screening_report": screen_report,
        "krum_report": krum_meta,
        "consensus_delta_sha256": consensus_sha,
        "parent_state_sha256": parent_sha,
        "candidate_state_sha256": candidate_sha,
        "final_state_sha256": final_sha,
        "config": asdict(cfg),
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SAFE_E0_RATIO",
    "DEFAULT_DOMAIN_GUARD_MAX_RATIO",
    "DEFAULT_EDIT_TOL",
    "DEFAULT_MAX_EDIT_FRACTION",
    "DEFAULT_MAX_SPECTRAL_SPIKE_RATIO",
    "DEFAULT_SCORE_FROM",
    "DEFAULT_SOFT_MARGIN_PENALTY",
    "DEFAULT_SV_THRESHOLD",
    "GENERAL_EXPERT_PREFIX",
    "RECEIPT_OBJECT",
    "SCHEMA_VERSION",
    "VALID_DELTA_FORMATS",
    "ByzantineWSPConfig",
    "apply_verified_byzantine_wsp_round",
    "canonical_json_bytes",
    "compute_alphaedit_prose_nullspace",
    "compute_payload_sha256",
    "compute_state_sha256",
    "multi_krum_nullspace_aggregate",
    "project_delta_into_nullspace",
    "screen_peer_wsp_deltas_spectral_and_sparsity",
    "verify_byzantine_wsp_receipt_provenance",
]

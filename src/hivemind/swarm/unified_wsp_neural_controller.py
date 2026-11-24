# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unified 5-Stage Weight-Space Programming (WSP) Neural Controller for Open-Weight MoEs.

Composes HiveMind's cleanroom open-weight neural-control primitives into a deterministic,
zero-gradient 5-stage ``WeightSpaceController`` pipeline reproducing the T4 Pilot 1 and
overnight S2..S4 breakthroughs:

1. **Stage 1 — Finite & Spectral Preflight Sanitization**:
   Validates model/bucket structural integrity, repairs or rejects non-finite entries,
   and damps catastrophic LoRA singular-value / router-norm outliers via QR-SVD without
   disturbing legitimate specialist structure.
2. **Stage 2 — LOEO Partition & Per-Layer Causal E0 Harm Attribution**:
   Runs Leave-One-Expert-Out ablation to partition harmful/rogue experts into
   zero-specialist-cost core FoMoE ghost sinks (``delta_spec <= +0.010`` nats) vs
   specialist-bearing marginal experts while protecting helpful general anchors, and
   computes per-layer MoE bypass attribution to localize surgical edits to harmed layers.
3. **Stage 3 — Layer-Localized Sinking, Rank-r QR-SVD Surgery & Multi-Gate Control**:
   Applies zero-perturbation ghost sinking (``B = 0 => dy == 0``), optional
   Discriminative-AlphaEdit null-space projection, QR-SVD singular-mode attenuation on
   marginal/surviving specialists, and gentle per-domain orthogonalized ``sigma_topk``
   router multi-gate control with 2-fold out-of-fold SNR shutoff.
4. **Stage 4 — 21-Point Fine LayerNorm Guard Calibration**:
   Sweeps the 21-point fine grid ``{0.00, 0.05, ..., 1.00}`` on post-surgery pre-LN
   states to minimize overall NLL subject to strict general-domain drift clearance.
5. **Stage 5 — Thermodynamic ANSE Acceptance Gate & Atomic Bitwise Rollback**:
   Enforces ``Delta E = E_child - E_parent < 0`` and worst-domain drift ``<= +1.0%``
   across all domains, committing only verified improvements or atomically rolling back
   bitwise to the parent state, with canonical SHA-256 receipts at every stage.
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
    extract_moe_trainable_buckets,
    load_moe_trainable_buckets,
)
from hivemind.swarm.layer_localized_attribution import (
    compute_expert_weight_norms,
    compute_leave_one_expert_out_attribution,
    compute_per_layer_moe_bypass_attribution,
    per_chunk_causal_midrow_nll,
    select_surgical_layer_mask,
    zero_expert_lora_b_inplace,
)
from hivemind.swarm.lora_svd_mode_surgery import (
    SVDSurgeryConfig,
    apply_moe_layer_svd_surgery_inplace,
    decompose_lora_qr_svd,
)
from hivemind.swarm.sigma_router_gate_control import (
    apply_sigma_router_prose_control_inplace,
    evaluate_sigma_topk_routing_gates,
    fit_per_domain_orthogonal_gate_vectors,
    fit_uncentered_prose_probe_with_oof,
)
from hivemind.swarm.wsp_alphaedit_control import (
    compute_discriminative_alphaedit_filter,
    compute_embeddings_batch,
    compute_swiglu_intermediate_states,
    forward_block_post_mlp,
    forward_block_pre_mlp,
    forward_model_pre_ln,
    second_moment_and_mean,
    select_guard_optimal_lambda_ln,
)

CONTROLLER_SCHEMA: str = "hivemind.unified_wsp_neural_control/1"
CONTROLLER_RECEIPT_OBJECT: str = "hivemind.unified_wsp_neural_control_receipt"

GENERAL_EXPERT_PREFIX: str = "E0_"
DEFAULT_GENERAL_EXPERT_INDEX: int = 0
DEFAULT_SCORE_FROM: int = 64
CORE_SINK_MAX_SPEC_DELTA: float = 0.010
DOMAIN_GUARD_MAX_RATIO: float = 1.01
FINE_LN_GUARD_MAX_RATIO: float = 1.006
ANSE_SAFE_E0_RATIO: float = 1.0085
ANSE_MAXIMUM_PAIN: float = 1e6
ANSE_SOFT_MARGIN_PENALTY: float = 0.05
FINE_LAMBDA_LN_GRID: tuple[float, ...] = tuple(round(0.05 * idx, 2) for idx in range(21))
STAGE_ORDER: tuple[str, ...] = (
    "stage_1_preflight",
    "stage_2_attribution",
    "stage_3_surgery",
    "stage_4_layernorm",
    "stage_5_anse_gate",
)

_EXPERT_PREFIX_RE = re.compile(r"^E(\d+)(?:_|$)")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# 1. Validation & Provenance Helpers
# ---------------------------------------------------------------------------


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


def _validate_nonneg_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite non-negative real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0.0, got {value!r}")
    return out


def _validate_finite_float(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically with strict finite-number enforcement."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def compute_payload_sha256(payload: Any) -> str:
    """Return lowercase SHA-256 hex digest of ``canonical_json_bytes(payload)``."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_buckets_sha256(buckets: Mapping[str, torch.Tensor]) -> str:
    """Compute a deterministic bitwise SHA-256 digest over ordered bucket keys, shapes, dtypes, and bytes."""
    if not isinstance(buckets, Mapping) or not buckets:
        raise ValueError("buckets must be a non-empty mapping of name -> torch.Tensor")
    digest = hashlib.sha256()
    for key in sorted(buckets.keys()):
        val = buckets[key]
        if not isinstance(val, torch.Tensor):
            raise TypeError(f"Bucket {key!r} must be a torch.Tensor, got {type(val).__name__}")
        cpu_t = val.detach().to(device="cpu").contiguous()
        header = f"{key}:{cpu_t.dtype}:{tuple(cpu_t.shape)}:".encode()
        digest.update(header)
        digest.update(cpu_t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def attach_stage_receipt_sha256(stage_name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a canonical ``stage_receipt_sha256`` digest to one stage telemetry payload."""
    body = {k: v for k, v in payload.items() if k != "stage_receipt_sha256"}
    body["stage"] = stage_name
    body["stage_receipt_sha256"] = compute_payload_sha256(body)
    return body


def verify_controller_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Independently verify top-level ``receipt_sha256`` and all per-stage SHA-256 digests."""
    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a mapping")
    top_hash = receipt.get("receipt_sha256")
    if not isinstance(top_hash, str) or not _SHA256_HEX_RE.match(top_hash):
        return False

    stages = receipt.get("stages")
    stage_hashes = receipt.get("stage_hashes")
    if not isinstance(stages, Mapping) or not isinstance(stage_hashes, Mapping):
        return False
    if set(stages.keys()) != set(STAGE_ORDER) or set(stage_hashes.keys()) != set(STAGE_ORDER):
        return False

    for stage_name in STAGE_ORDER:
        stage_doc = stages[stage_name]
        if not isinstance(stage_doc, Mapping) or stage_doc.get("stage") != stage_name:
            return False
        claimed_stage = stage_doc.get("stage_receipt_sha256")
        if not isinstance(claimed_stage, str) or not _SHA256_HEX_RE.match(claimed_stage):
            return False
        if stage_hashes.get(stage_name) != claimed_stage:
            return False
        stage_core = {k: v for k, v in stage_doc.items() if k != "stage_receipt_sha256"}
        try:
            expected_stage = compute_payload_sha256(stage_core)
        except (TypeError, ValueError):
            return False
        if claimed_stage != expected_stage:
            return False

    top_core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        expected_top = compute_payload_sha256(top_core)
    except (TypeError, ValueError):
        return False
    return top_hash == expected_top


def _parse_expert_domain_index(expert_id: str) -> int | None:
    """Extract numeric expert index from domain identifiers such as ``E1_python_systems_code``."""
    match = _EXPERT_PREFIX_RE.match(expert_id.strip())
    return int(match.group(1)) if match is not None else None


def _validate_model_compatible(model: nn.Module) -> None:
    """Verify required CrowdV1-compatible transformer attributes on ``model``."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("model must expose a non-empty 'blocks' sequence")
    for attr in ("wte", "ln", "head", "cos_cached", "sin_cached"):
        if not hasattr(model, attr):
            raise ValueError(f"model must expose '{attr}'")


def _validate_bucket_pair(
    start_buckets: Mapping[str, torch.Tensor],
    tuned_buckets: Mapping[str, torch.Tensor],
) -> None:
    """Validate structural parity, non-emptiness, and finite ``start_buckets`` baseline."""
    if not isinstance(start_buckets, Mapping) or not start_buckets:
        raise ValueError("start_buckets must be a non-empty mapping of name -> torch.Tensor")
    if not isinstance(tuned_buckets, Mapping) or not tuned_buckets:
        raise ValueError("tuned_buckets must be a non-empty mapping of name -> torch.Tensor")
    if set(start_buckets.keys()) != set(tuned_buckets.keys()):
        raise ValueError("start_buckets and tuned_buckets must share identical bucket keys")
    if not any(k.startswith("router_frag_") for k in start_buckets):
        raise ValueError("Buckets must contain at least one 'router_frag_*' entry")
    if not any(k.startswith("expert_") for k in start_buckets):
        raise ValueError("Buckets must contain at least one 'expert_*' entry")

    for key in sorted(start_buckets.keys()):
        s_t = start_buckets[key]
        t_t = tuned_buckets[key]
        if not isinstance(s_t, torch.Tensor) or not isinstance(t_t, torch.Tensor):
            raise TypeError(f"Bucket {key!r} entries must be torch.Tensor instances")
        if s_t.shape != t_t.shape:
            raise ValueError(f"Shape mismatch on bucket {key!r}: {tuple(s_t.shape)} vs {tuple(t_t.shape)}")
        if s_t.numel() == 0:
            raise ValueError(f"Bucket {key!r} must be non-empty")
        if not bool(torch.isfinite(s_t).all().item()):
            raise ValueError(f"start_buckets[{key!r}] must contain only finite values")


def _resolve_bucket_io(
    buckets: Mapping[str, torch.Tensor],
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None,
    extract_buckets_fn: Callable[[nn.Module], dict[str, torch.Tensor]] | None,
    *,
    num_router_fragments: int | None = None,
) -> tuple[Callable[[nn.Module, Mapping[str, torch.Tensor]], None], Callable[[nn.Module], dict[str, torch.Tensor]]]:
    """Bind OpenDiLoCo bucket loader/extractor when custom callbacks are omitted."""
    inferred_nfrag = max(1, len([k for k in buckets if k.startswith("router_frag_")]))
    nfrag = inferred_nfrag if num_router_fragments is None else _validate_positive_int(num_router_fragments, "num_router_fragments")
    loader = load_buckets_fn or (lambda m, b: load_moe_trainable_buckets(m, b, num_router_fragments=nfrag))
    extractor = extract_buckets_fn or (lambda m: extract_moe_trainable_buckets(m, num_router_fragments=nfrag))
    return loader, extractor


def _extract_ln_params_from_buckets(buckets: Mapping[str, torch.Tensor], h_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice final ``(ln.weight, ln.bias)`` from the tail of the last ``router_frag_*`` bucket."""
    router_keys = sorted(k for k in buckets if k.startswith("router_frag_"))
    last_vec = buckets[router_keys[-1]]
    if last_vec.numel() < 2 * h_dim:
        raise ValueError(f"Final router fragment too short ({last_vec.numel()}) for h_dim={h_dim}")
    ln_w = last_vec[-2 * h_dim : -h_dim].detach().to("cpu", dtype=torch.float32).clone()
    ln_b = last_vec[-h_dim:].detach().to("cpu", dtype=torch.float32).clone()
    return ln_w, ln_b


def _clone_buckets(buckets: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a deep CPU clone of a bucket dictionary."""
    return {k: v.detach().to(device="cpu").clone() for k, v in buckets.items()}


def _validate_eval_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    general_prefix: str,
    name: str,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Validate chunk sequence and return ``(ids_2d, is_general_mask, expert_ids)``."""
    if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence) or len(chunks) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of chunk mappings")
    id_rows: list[torch.Tensor] = []
    expert_ids: list[str] = []
    seq_len: int | None = None
    for idx, ch in enumerate(chunks):
        if not isinstance(ch, Mapping) or "expert_id" not in ch or "ids" not in ch:
            raise ValueError(f"{name}[{idx}] must be a mapping containing 'expert_id' and 'ids'")
        eid = str(ch["expert_id"]).strip()
        if not eid:
            raise ValueError(f"{name}[{idx}]['expert_id'] must be a non-empty string")
        ids_t = torch.as_tensor(ch["ids"], dtype=torch.long)
        if ids_t.ndim != 1 or ids_t.numel() < 2:
            raise ValueError(f"{name}[{idx}]['ids'] must be 1D with length >= 2")
        if bool((ids_t < 0).any().item()):
            raise ValueError(f"{name}[{idx}]['ids'] must contain non-negative token IDs")
        if seq_len is None:
            seq_len = int(ids_t.numel())
        elif int(ids_t.numel()) != seq_len:
            raise ValueError(f"All {name} entries must share sequence length {seq_len}")
        id_rows.append(ids_t)
        expert_ids.append(eid)

    ids_2d = torch.stack(id_rows, dim=0)
    is_general = np.array([eid.startswith(general_prefix) for eid in expert_ids], dtype=bool)
    if not bool(np.any(is_general)):
        raise ValueError(f"{name} must include at least one general chunk starting with {general_prefix!r}")
    return ids_2d, is_general, expert_ids


# ---------------------------------------------------------------------------
# 2. Configuration & Execution Result Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightSpaceControllerConfig:
    """Configuration governing the 5-stage Weight-Space Programming pipeline."""

    general_prefix: str = GENERAL_EXPERT_PREFIX
    general_expert_index: int = DEFAULT_GENERAL_EXPERT_INDEX
    score_from: int = DEFAULT_SCORE_FROM
    batch_size: int = 24
    num_router_fragments: int | None = None

    # Stage 1: Finite & Spectral Preflight Sanitization
    sanitize_non_finite: bool = True
    max_lora_singular_value: float = 80.0
    max_router_row_norm: float = 50.0
    rogue_b_norm_threshold: float = 10.0
    rogue_norm_ratio_threshold: float = 4.0
    rogue_min_norm_floor: float = 1.0

    # Stage 2: LOEO Partition & Per-Layer Causal E0 Attribution
    loeo_chunks_per_domain: int = 8
    loeo_e0_harm_threshold: float = -0.0015
    loeo_general_help_threshold: float = 1e-4
    loeo_max_spec_loss_increase: float = 0.03
    core_sink_max_spec_delta: float = CORE_SINK_MAX_SPEC_DELTA
    max_general_probe_chunks: int = 24
    max_spec_probe_chunks: int = 24
    top_k_layers: int | None = None
    min_harm_nats: float = 0.002
    protect_early_layers_below: int = 0
    scored_expert_indices: tuple[int, ...] | None = None

    # Stage 3: Layer-Localized Sinking, QR-SVD, Multi-Gate, Optional AlphaEdit
    use_ghost_sink: bool = True
    force_zero_expert0: bool = False
    quarantine_rogue_all_layers: bool = True
    svd_tau: float = 1.20
    svd_eta: float = 3.0
    svd_max_modes_damped: int = 2
    svd_reg_spec: float = 0.10
    svd_truncate: bool = False
    svd_include_w3: bool = True
    svd_target_marginal_only: bool = False
    alphaedit_eta_x: float = 0.0
    alphaedit_eta_act: float = 0.0
    alphaedit_tau: float = 1.25
    alphaedit_max_rank_x: int = 96
    alphaedit_max_rank_act: int = 128
    alphaedit_reg_spec: float = 0.10
    router_beta_suppress: float = 1.25
    router_beta_sink: float = 0.50
    router_beta_general_boost: float = 0.0
    router_omega_spec: float = 6.0
    router_omega_own: float = 8.0
    router_omega_other: float = 2.0
    router_ridge_scale: float = 0.75
    router_min_domain_samples: int = 8

    # Stage 4: 21-Point Fine LayerNorm Guard Calibration
    use_ln_calibration: bool = True
    fine_ln_e0_max_ratio: float = FINE_LN_GUARD_MAX_RATIO
    forced_lambda_ln: float | None = None

    # Stage 5: Thermodynamic ANSE Acceptance Gate
    domain_guard_max_ratio: float = DOMAIN_GUARD_MAX_RATIO
    anse_safe_e0_ratio: float = ANSE_SAFE_E0_RATIO
    min_energy_improvement: float = 0.0
    min_chunks_per_expert: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.general_prefix, str) or not self.general_prefix.strip():
            raise ValueError("general_prefix must be a non-empty string")
        _validate_nonneg_int(self.general_expert_index, "general_expert_index")
        _validate_nonneg_int(self.score_from, "score_from")
        _validate_positive_int(self.batch_size, "batch_size")
        if self.num_router_fragments is not None:
            _validate_positive_int(self.num_router_fragments, "num_router_fragments")

        for flag_name in (
            "sanitize_non_finite",
            "use_ghost_sink",
            "force_zero_expert0",
            "quarantine_rogue_all_layers",
            "svd_truncate",
            "svd_include_w3",
            "svd_target_marginal_only",
            "use_ln_calibration",
        ):
            if not isinstance(getattr(self, flag_name), bool):
                raise TypeError(f"{flag_name} must be a bool")

        for pos_name in (
            "max_lora_singular_value",
            "max_router_row_norm",
            "fine_ln_e0_max_ratio",
            "domain_guard_max_ratio",
            "anse_safe_e0_ratio",
        ):
            val = _validate_nonneg_finite(getattr(self, pos_name), pos_name)
            if val <= 0.0:
                raise ValueError(f"{pos_name} must be > 0.0, got {val}")

        for nn_float in (
            "rogue_b_norm_threshold",
            "rogue_norm_ratio_threshold",
            "rogue_min_norm_floor",
            "loeo_general_help_threshold",
            "loeo_max_spec_loss_increase",
            "core_sink_max_spec_delta",
            "min_harm_nats",
            "svd_tau",
            "svd_eta",
            "svd_reg_spec",
            "alphaedit_eta_x",
            "alphaedit_eta_act",
            "alphaedit_tau",
            "alphaedit_reg_spec",
            "router_beta_suppress",
            "router_beta_sink",
            "router_beta_general_boost",
            "router_omega_spec",
            "router_omega_own",
            "router_omega_other",
            "router_ridge_scale",
            "min_energy_improvement",
        ):
            _validate_nonneg_finite(getattr(self, nn_float), nn_float)

        harm_thresh = _validate_finite_float(self.loeo_e0_harm_threshold, "loeo_e0_harm_threshold")
        if harm_thresh > 0.0:
            raise ValueError(f"loeo_e0_harm_threshold must be <= 0.0, got {harm_thresh}")

        _validate_positive_int(self.loeo_chunks_per_domain, "loeo_chunks_per_domain")
        _validate_positive_int(self.max_general_probe_chunks, "max_general_probe_chunks")
        _validate_positive_int(self.max_spec_probe_chunks, "max_spec_probe_chunks")
        _validate_nonneg_int(self.protect_early_layers_below, "protect_early_layers_below")
        _validate_nonneg_int(self.svd_max_modes_damped, "svd_max_modes_damped")
        _validate_nonneg_int(self.alphaedit_max_rank_x, "alphaedit_max_rank_x")
        _validate_nonneg_int(self.alphaedit_max_rank_act, "alphaedit_max_rank_act")
        _validate_nonneg_int(self.router_min_domain_samples, "router_min_domain_samples")
        _validate_positive_int(self.min_chunks_per_expert, "min_chunks_per_expert")

        if self.top_k_layers is not None:
            _validate_nonneg_int(self.top_k_layers, "top_k_layers")
        if self.scored_expert_indices is not None:
            for idx in self.scored_expert_indices:
                _validate_nonneg_int(idx, "scored_expert_index")
        if self.forced_lambda_ln is not None:
            _validate_finite_float(self.forced_lambda_ln, "forced_lambda_ln")

    def to_svd_config(self) -> SVDSurgeryConfig:
        """Convert SVD surgery parameters into a validated ``SVDSurgeryConfig``."""
        return SVDSurgeryConfig(
            tau_svd=self.svd_tau,
            eta_svd=self.svd_eta,
            max_modes_damped=self.svd_max_modes_damped,
            reg_spec=self.svd_reg_spec,
            truncate=self.svd_truncate,
        )


@dataclass(frozen=True)
class ControllerExecutionResult:
    """Verified outcome, final parameter buckets, and SHA-256 stage provenance of one pipeline run."""

    buckets: dict[str, torch.Tensor]
    receipt: dict[str, Any]
    accepted: bool
    rolled_back: bool
    status: str
    receipt_sha256: str


# ---------------------------------------------------------------------------
# 3. Stage 1 — Finite & Spectral Preflight Sanitization
# ---------------------------------------------------------------------------


@torch.no_grad()
def _clamp_adapter_singular_values_inplace(
    exp_mod: nn.Module,
    attr_prefix: str,
    max_singular_value: float,
) -> tuple[float, int]:
    """Clamp any LoRA singular value above ``max_singular_value`` via QR-SVD right projection."""
    w_b: torch.Tensor = getattr(exp_mod, f"{attr_prefix}_B")
    w_a: torch.Tensor = getattr(exp_mod, f"{attr_prefix}_A")
    _, s_vals, v_r = decompose_lora_qr_svd(w_b, w_a)
    max_sv = float(s_vals.max().item())
    if max_sv <= max_singular_value or max_sv <= 1e-12:
        return max_sv, 0

    over_idx = [idx for idx, val in enumerate(s_vals.tolist()) if val > max_singular_value]
    idx_t = torch.tensor(over_idx, dtype=torch.long)
    v_d = v_r[:, idx_t]
    scales_d = torch.tensor([max_singular_value / float(s_vals[i].item()) for i in over_idx], dtype=torch.float64)
    shrink_d = 1.0 - scales_d
    a64 = w_a.detach().to(device="cpu", dtype=torch.float64)
    delta_a = ((a64 @ v_d) * shrink_d.unsqueeze(0)) @ v_d.T
    a_new = (a64 - delta_a).to(device=w_a.device, dtype=w_a.dtype)
    w_a.copy_(a_new)
    return max_sv, len(over_idx)


@torch.no_grad()
def run_stage1_preflight_sanitization(
    model: nn.Module,
    start_buckets: Mapping[str, torch.Tensor],
    tuned_buckets: Mapping[str, torch.Tensor],
    *,
    config: WeightSpaceControllerConfig,
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
    extract_buckets_fn: Callable[[nn.Module], dict[str, torch.Tensor]] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Stage 1: validate structural integrity, sanitize non-finite values, and damp extreme spectral outliers."""
    _validate_model_compatible(model)
    _validate_bucket_pair(start_buckets, tuned_buckets)
    loader, extractor = _resolve_bucket_io(
        start_buckets,
        load_buckets_fn,
        extract_buckets_fn,
        num_router_fragments=config.num_router_fragments,
    )

    working = _clone_buckets(tuned_buckets)
    non_finite_repaired = 0
    non_finite_buckets: list[str] = []

    for key in sorted(working.keys()):
        tensor = working[key]
        finite_mask = torch.isfinite(tensor)
        bad_count = int((~finite_mask).sum().item())
        if bad_count > 0:
            if not config.sanitize_non_finite:
                raise ValueError(f"Non-finite entries ({bad_count}) detected in tuned_buckets[{key!r}]")
            if key.startswith("router_frag_"):
                fallback = start_buckets[key].detach().to(device="cpu", dtype=tensor.dtype)
                working[key] = torch.where(finite_mask, tensor, fallback)
            else:
                working[key] = torch.where(finite_mask, tensor, torch.zeros_like(tensor))
            non_finite_repaired += bad_count
            non_finite_buckets.append(key)

    loader(model, working)

    # Verify frozen trunk parameters remain finite
    for p_name, param in model.named_parameters():
        if not param.requires_grad and not bool(torch.isfinite(param).all().item()):
            raise ValueError(f"Frozen model parameter {p_name!r} contains non-finite values")

    router_rows_clamped = 0
    max_router_norm_before = 0.0
    spectral_clamped_adapters: list[dict[str, Any]] = []
    max_sv_before = 0.0

    for l_idx, blk in enumerate(model.blocks):
        mlp = getattr(blk, "mlp", None)
        router_w = getattr(getattr(mlp, "router", None), "weight", None)
        if isinstance(router_w, torch.Tensor):
            row_norms = router_w.detach().to(torch.float64).norm(dim=-1)
            max_router_norm_before = max(max_router_norm_before, float(row_norms.max().item()))
            for e_row in range(int(router_w.shape[0])):
                rn = float(row_norms[e_row].item())
                if rn > config.max_router_row_norm:
                    router_w[e_row].mul_(config.max_router_row_norm / rn)
                    router_rows_clamped += 1

        experts_dict = getattr(mlp, "experts", None)
        if experts_dict is not None and hasattr(experts_dict, "items"):
            for e_key, exp_mod in sorted(experts_dict.items(), key=lambda kv: int(kv[0])):
                for attr_prefix in ("w1", "w2", "w3"):
                    if not (hasattr(exp_mod, f"{attr_prefix}_B") and hasattr(exp_mod, f"{attr_prefix}_A")):
                        continue
                    sv_before, n_clamped = _clamp_adapter_singular_values_inplace(
                        exp_mod,
                        attr_prefix,
                        config.max_lora_singular_value,
                    )
                    max_sv_before = max(max_sv_before, sv_before)
                    if n_clamped > 0:
                        spectral_clamped_adapters.append(
                            {
                                "layer": l_idx,
                                "expert": int(e_key),
                                "projection": attr_prefix,
                                "max_singular_before": sv_before,
                                "clamped_modes": n_clamped,
                            }
                        )

    norm_inventory = compute_expert_weight_norms(model)
    high_norm_warnings = sorted(
        e_idx for e_idx, info in norm_inventory.items() if float(info["w1_b_norm"]) >= config.rogue_b_norm_threshold
    )
    sanitized_buckets = _clone_buckets(extractor(model))
    out_sha = compute_buckets_sha256(sanitized_buckets)

    receipt = attach_stage_receipt_sha256(
        "stage_1_preflight",
        {
            "output_buckets_sha256": out_sha,
            "non_finite_repaired_elements": non_finite_repaired,
            "non_finite_repaired_buckets": non_finite_buckets,
            "max_adapter_singular_value_before": max_sv_before,
            "spectral_clamped_adapters_count": len(spectral_clamped_adapters),
            "spectral_clamped_adapters": spectral_clamped_adapters,
            "max_router_row_norm_before": max_router_norm_before,
            "router_rows_clamped": router_rows_clamped,
            "high_norm_warning_experts": high_norm_warnings,
            "changed_from_input": bool(non_finite_repaired > 0 or router_rows_clamped > 0 or spectral_clamped_adapters),
        },
    )
    return sanitized_buckets, receipt


# ---------------------------------------------------------------------------
# 4. Stage 2 — LOEO Partition & Per-Layer Causal E0 Harm Attribution
# ---------------------------------------------------------------------------


def partition_loeo_experts(
    loeo_report: Mapping[str, Any],
    *,
    core_sink_max_spec_delta: float = CORE_SINK_MAX_SPEC_DELTA,
    force_zero_expert0: bool = False,
    use_ghost_sink: bool = True,
) -> dict[str, Any]:
    """Partition LOEO attribution into zero-spec-cost core ghost sinks vs specialist-bearing marginal experts."""
    max_spec_cap = _validate_nonneg_finite(core_sink_max_spec_delta, "core_sink_max_spec_delta")
    gen_idx = int(loeo_report.get("general_expert_index", DEFAULT_GENERAL_EXPERT_INDEX))
    protect_general = bool(loeo_report.get("protect_general_expert", False))
    expert_loeo = loeo_report.get("expert_loeo", {})

    harmful_set = {int(e) for e in loeo_report.get("harmful_sink_experts", [])}
    rogue_set = {int(e) for e in loeo_report.get("rogue_high_norm_experts", [])}
    recommended_set = {int(e) for e in loeo_report.get("recommended_sink_experts", [])}
    interfering_set = {int(e) for e in loeo_report.get("interfering_specialist_experts", [])}
    candidate_ids = sorted(harmful_set | rogue_set | recommended_set | interfering_set)

    core_sinks: list[int] = []
    marginal_specialists: list[int] = []
    for e_idx in candidate_ids:
        entry = expert_loeo.get(f"expert_{e_idx}", {})
        d_spec = float(entry.get("delta_spec_when_zeroed", 0.0))
        if (e_idx in recommended_set) and d_spec <= max_spec_cap:
            core_sinks.append(e_idx)
        else:
            marginal_specialists.append(e_idx)

    if use_ghost_sink and force_zero_expert0:
        if gen_idx not in core_sinks:
            core_sinks.append(gen_idx)
        marginal_specialists = [e for e in marginal_specialists if e != gen_idx]
        protect_general = False

    core_sinks = sorted(set(core_sinks))
    marginal_specialists = sorted(set(marginal_specialists))
    rogue_core_sinks = sorted(set(core_sinks) & rogue_set)
    non_rogue_core_sinks = sorted(set(core_sinks) - rogue_set)

    return {
        "general_expert_index": gen_idx,
        "expert0_regime": str(loeo_report.get("expert0_regime", "neutral")),
        "protect_general_expert": protect_general,
        "core_sink_experts": core_sinks,
        "rogue_core_sink_experts": rogue_core_sinks,
        "non_rogue_core_sink_experts": non_rogue_core_sinks,
        "marginal_specialist_experts": marginal_specialists,
        "helpful_general_experts": [int(e) for e in loeo_report.get("helpful_general_experts", [])],
        "helpful_specialist_experts": [int(e) for e in loeo_report.get("helpful_specialist_experts", [])],
        "neutral_experts": [int(e) for e in loeo_report.get("neutral_experts", [])],
        "core_sink_max_spec_delta": max_spec_cap,
    }


@torch.no_grad()
def run_stage2_loeo_and_layer_attribution(
    model: nn.Module,
    start_buckets: Mapping[str, torch.Tensor],
    sanitized_buckets: Mapping[str, torch.Tensor],
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    config: WeightSpaceControllerConfig,
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stage 2: run LOEO expert partitioning and per-layer causal E0 harm attribution."""
    _validate_model_compatible(model)
    loader, _ = _resolve_bucket_io(
        start_buckets,
        load_buckets_fn,
        None,
        num_router_fragments=config.num_router_fragments,
    )
    ids_all, is_e0, _ = _validate_eval_chunks(
        probe_chunks,
        general_prefix=config.general_prefix,
        name="probe_chunks",
    )

    h_dim = int(model.ln.weight.numel())
    ln_eps = float(getattr(model.ln, "eps", 1e-5))
    start_ln_w, start_ln_b = _extract_ln_params_from_buckets(start_buckets, h_dim)
    tuned_ln_w, tuned_ln_b = _extract_ln_params_from_buckets(sanitized_buckets, h_dim)

    loader(model, start_buckets)
    model.eval()
    h_base = forward_model_pre_ln(model, ids_all, batch_size=config.batch_size)
    base_probe_losses = (
        per_chunk_causal_midrow_nll(
            h_base,
            start_ln_w,
            start_ln_b,
            ln_eps,
            model.head.weight,
            ids_all,
            score_from=config.score_from,
            seq_batch_size=config.batch_size,
        )
        .cpu()
        .numpy()
    )

    if config.use_ln_calibration:
        loader(model, sanitized_buckets)
        model.eval()
        h_sanitized = forward_model_pre_ln(model, ids_all, batch_size=config.batch_size)
        init_lambda_ln, _ = select_guard_optimal_lambda_ln(
            h_sanitized,
            start_ln_w,
            start_ln_b,
            tuned_ln_w,
            tuned_ln_b,
            ln_eps,
            model.head.weight,
            ids_all,
            base_probe_losses,
            is_e0,
            score_from=config.score_from,
            target_e0_max_ratio=config.fine_ln_e0_max_ratio,
            lambda_grid=FINE_LAMBDA_LN_GRID,
        )
    else:
        init_lambda_ln = 1.0

    init_cal_ln_w = start_ln_w + init_lambda_ln * (tuned_ln_w - start_ln_w)
    init_cal_ln_b = start_ln_b + init_lambda_ln * (tuned_ln_b - start_ln_b)

    loeo_report = compute_leave_one_expert_out_attribution(
        model,
        probe_chunks,
        buckets=sanitized_buckets,
        load_buckets_fn=loader,
        cal_ln_w=init_cal_ln_w,
        cal_ln_b=init_cal_ln_b,
        score_from=config.score_from,
        loeo_chunks_per_domain=config.loeo_chunks_per_domain,
        batch_size=config.batch_size,
        general_prefix=config.general_prefix,
        general_expert_index=config.general_expert_index,
        scored_expert_indices=config.scored_expert_indices,
        loeo_e0_harm_threshold=config.loeo_e0_harm_threshold,
        loeo_general_help_threshold=config.loeo_general_help_threshold,
        loeo_max_spec_loss_increase=config.loeo_max_spec_loss_increase,
        rogue_b_norm_threshold=config.rogue_b_norm_threshold,
        rogue_norm_ratio_threshold=config.rogue_norm_ratio_threshold,
        rogue_min_norm_floor=config.rogue_min_norm_floor,
    )

    partition = partition_loeo_experts(
        loeo_report,
        core_sink_max_spec_delta=config.core_sink_max_spec_delta,
        force_zero_expert0=config.force_zero_expert0,
        use_ghost_sink=config.use_ghost_sink,
    )

    layer_attr = compute_per_layer_moe_bypass_attribution(
        model,
        probe_chunks,
        buckets=sanitized_buckets,
        load_buckets_fn=loader,
        cal_ln_w=init_cal_ln_w,
        cal_ln_b=init_cal_ln_b,
        score_from=config.score_from,
        max_general_probe_chunks=config.max_general_probe_chunks,
        max_spec_probe_chunks=config.max_spec_probe_chunks,
        batch_size=config.batch_size,
        general_prefix=config.general_prefix,
    )

    layer_mask = select_surgical_layer_mask(
        layer_attr,
        top_k_layers=config.top_k_layers,
        min_harm_nats=config.min_harm_nats,
        protect_early_layers_below=config.protect_early_layers_below,
    )

    context = {
        "base_probe_losses": base_probe_losses,
        "init_lambda_ln": init_lambda_ln,
        "loeo_report": loeo_report,
        "partition": partition,
        "layer_attribution": layer_attr,
        "layer_mask_summary": layer_mask,
    }
    receipt = attach_stage_receipt_sha256(
        "stage_2_attribution",
        {
            "input_buckets_sha256": compute_buckets_sha256(sanitized_buckets),
            "init_lambda_ln": init_lambda_ln,
            "expert0_regime": partition["expert0_regime"],
            "protect_general_expert": partition["protect_general_expert"],
            "core_sink_experts": partition["core_sink_experts"],
            "rogue_core_sink_experts": partition["rogue_core_sink_experts"],
            "marginal_specialist_experts": partition["marginal_specialist_experts"],
            "helpful_general_experts": partition["helpful_general_experts"],
            "helpful_specialist_experts": partition["helpful_specialist_experts"],
            "selected_layers": layer_mask["selected_layers"],
            "protected_layers": layer_mask["protected_layers"],
            "harm_coverage_fraction": layer_mask["harm_coverage_fraction"],
            "ranked_layers_by_general_harm": layer_attr["ranked_layers_by_general_harm"],
            "layer_general_harm_nats": [
                {"layer": int(row["layer"]), "general_harm_nats": float(row["general_harm_nats"])}
                for row in layer_attr["layers"]
            ],
        },
    )
    return context, receipt


# ---------------------------------------------------------------------------
# 5. Stage 3 — Layer-Localized Sinking, QR-SVD Surgery & Multi-Gate Control
# ---------------------------------------------------------------------------


def _build_probe_folds(
    probe_chunks: Sequence[Mapping[str, Any]],
    *,
    general_prefix: str,
    general_expert_index: int,
) -> tuple[list[int], list[int], list[int], list[int], dict[int, list[int]]]:
    """Split probe chunk indices into deterministic 2-fold prose/specialist folds and domain map."""
    prose_f0: list[int] = []
    prose_f1: list[int] = []
    spec_f0: list[int] = []
    spec_f1: list[int] = []
    dom_indices: dict[int, list[int]] = {}
    counts: dict[str, int] = {}

    for idx, ch in enumerate(probe_chunks):
        dom = str(ch["expert_id"]).strip()
        cnt = counts.get(dom, 0)
        counts[dom] = cnt + 1
        if dom.startswith(general_prefix):
            (prose_f0 if cnt % 2 == 0 else prose_f1).append(idx)
        else:
            (spec_f0 if cnt % 2 == 0 else spec_f1).append(idx)
            parsed = _parse_expert_domain_index(dom)
            if parsed is not None and parsed != general_expert_index:
                dom_indices.setdefault(parsed, []).append(idx)

    return prose_f0, prose_f1, spec_f0, spec_f1, dom_indices


@torch.no_grad()
def _expert_has_nonzero_lora_b(moe_mlp: nn.Module, expert_indices: Sequence[int]) -> list[int]:
    """Return subset of ``expert_indices`` whose LoRA B weights are currently non-zero on ``moe_mlp``."""
    experts_dict = getattr(moe_mlp, "experts", {})
    present: list[int] = []
    for e_idx in expert_indices:
        key = str(e_idx)
        if key not in experts_dict:
            continue
        exp = experts_dict[key]
        if any(
            isinstance(getattr(exp, attr, None), torch.Tensor) and bool(torch.any(getattr(exp, attr) != 0.0).item())
            for attr in ("w1_B", "w2_B", "w3_B")
        ):
            present.append(e_idx)
    return present


@torch.no_grad()
def _apply_optional_alphaedit_complement_inplace(
    moe_mlp: nn.Module,
    p_all: torch.Tensor,
    s_all: torch.Tensor,
    target_experts: Sequence[int],
    config: WeightSpaceControllerConfig,
) -> dict[str, Any]:
    """Optionally apply dual-space Discriminative-AlphaEdit right filter on ``(w1_A, w2_A, w3_A)``."""
    summary: dict[str, Any] = {"active_modes_x": 0, "active_modes_act": 0}
    if not target_experts:
        return summary

    if config.alphaedit_eta_x > 0.0:
        mp_x, _ = second_moment_and_mean(p_all)
        ms_x, _ = second_moment_and_mean(s_all)
        g_x, gx_stats = compute_discriminative_alphaedit_filter(
            mp_x,
            ms_x,
            eta=config.alphaedit_eta_x,
            tau=config.alphaedit_tau,
            max_rank=config.alphaedit_max_rank_x,
            reg_spec=config.alphaedit_reg_spec,
        )
        summary["active_modes_x"] = int(gx_stats["active_modes"])
        if gx_stats["active_modes"] > 0:
            gx_f32 = g_x.to(dtype=torch.float32)
            for e_idx in target_experts:
                exp = moe_mlp.experts[str(e_idx)]
                exp.w1_A.copy_((exp.w1_A.to(torch.float32) @ gx_f32.to(exp.w1_A.device)).to(exp.w1_A.dtype))
                exp.w2_A.copy_((exp.w2_A.to(torch.float32) @ gx_f32.to(exp.w2_A.device)).to(exp.w2_A.dtype))

    if config.alphaedit_eta_act > 0.0:
        act_p = compute_swiglu_intermediate_states(moe_mlp, p_all, target_experts)
        act_s = compute_swiglu_intermediate_states(moe_mlp, s_all, target_experts)
        mp_a, _ = second_moment_and_mean(act_p)
        ms_a, _ = second_moment_and_mean(act_s)
        g_a, ga_stats = compute_discriminative_alphaedit_filter(
            mp_a,
            ms_a,
            eta=config.alphaedit_eta_act,
            tau=config.alphaedit_tau,
            max_rank=config.alphaedit_max_rank_act,
            reg_spec=config.alphaedit_reg_spec,
        )
        summary["active_modes_act"] = int(ga_stats["active_modes"])
        if ga_stats["active_modes"] > 0:
            ga_f32 = g_a.to(dtype=torch.float32)
            for e_idx in target_experts:
                exp = moe_mlp.experts[str(e_idx)]
                exp.w3_A.copy_((exp.w3_A.to(torch.float32) @ ga_f32.to(exp.w3_A.device)).to(exp.w3_A.dtype))

    return summary


@torch.no_grad()
def run_stage3_wsp_surgery(
    model: nn.Module,
    sanitized_buckets: Mapping[str, torch.Tensor],
    probe_chunks: Sequence[Mapping[str, Any]],
    stage2_context: Mapping[str, Any],
    *,
    config: WeightSpaceControllerConfig,
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
    extract_buckets_fn: Callable[[nn.Module], dict[str, torch.Tensor]] | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    """Stage 3: execute layer-localized sinking, QR-SVD mode surgery, and gentle multi-gate control."""
    _validate_model_compatible(model)
    loader, extractor = _resolve_bucket_io(
        sanitized_buckets,
        load_buckets_fn,
        extract_buckets_fn,
        num_router_fragments=config.num_router_fragments,
    )
    ids_all, _, _ = _validate_eval_chunks(
        probe_chunks,
        general_prefix=config.general_prefix,
        name="probe_chunks",
    )

    loader(model, _clone_buckets(sanitized_buckets))
    model.eval()

    partition = stage2_context["partition"]
    layer_mask_summary = stage2_context["layer_mask_summary"]
    core_sinks = list(partition["core_sink_experts"]) if config.use_ghost_sink else []
    rogue_core = list(partition["rogue_core_sink_experts"]) if (config.use_ghost_sink and config.quarantine_rogue_all_layers) else []
    non_rogue_core = [e for e in core_sinks if e not in set(rogue_core)]
    marginal_experts = list(partition["marginal_specialist_experts"])
    protect_general = bool(partition["protect_general_expert"])
    selected_set = {int(l_idx) for l_idx in layer_mask_summary["selected_layers"]}

    prose_f0, prose_f1, spec_f0, spec_f1, dom_indices = _build_probe_folds(
        probe_chunks,
        general_prefix=config.general_prefix,
        general_expert_index=config.general_expert_index,
    )
    has_twofold = bool(prose_f0 and prose_f1 and spec_f0 and spec_f1)
    start_pos = max(0, config.score_from - 1)
    if start_pos >= int(ids_all.shape[1]) - 1:
        raise ValueError(f"Sequence length {ids_all.shape[1]} too short for score_from={config.score_from}")

    cos = model.cos_cached.to(ids_all.device)
    sin = model.sin_cached.to(ids_all.device)
    bsz = config.batch_size
    total_model_params = max(1, sum(p.numel() for p in model.parameters()))
    svd_cfg = config.to_svd_config()

    x_seq = torch.cat(
        [compute_embeddings_batch(model, ids_all[b0 : b0 + bsz]) for b0 in range(0, ids_all.shape[0], bsz)],
        dim=0,
    )

    layer_reports: list[dict[str, Any]] = []
    total_damped_modes = 0
    total_svd_modified_elements = 0
    router_layers_applied = 0
    quarantined_rogue_layers: list[int] = []

    for l_idx, blk in enumerate(model.blocks):
        mlp = getattr(blk, "mlp", None)
        has_moe = hasattr(mlp, "experts") and hasattr(mlp, "router")

        rogue_sunk_here: list[int] = []
        if has_moe and rogue_core:
            nonzero_rogue = _expert_has_nonzero_lora_b(mlp, rogue_core)
            if nonzero_rogue:
                zero_expert_lora_b_inplace(mlp, nonzero_rogue)
                rogue_sunk_here = nonzero_rogue
                quarantined_rogue_layers.append(l_idx)

        x_mid_parts: list[torch.Tensor] = []
        mlp_in_parts: list[torch.Tensor] = []
        for b0 in range(0, x_seq.shape[0], bsz):
            xm, mi = forward_block_pre_mlp(blk, x_seq[b0 : b0 + bsz], cos, sin)
            x_mid_parts.append(xm)
            mlp_in_parts.append(mi)
        x_mid = torch.cat(x_mid_parts, dim=0)
        mlp_in = torch.cat(mlp_in_parts, dim=0)

        if has_moe and l_idx in selected_set:
            sunk_here = sorted(set(rogue_sunk_here))
            if non_rogue_core:
                nonzero_core = _expert_has_nonzero_lora_b(mlp, non_rogue_core)
                if nonzero_core:
                    zero_expert_lora_b_inplace(mlp, nonzero_core)
                    sunk_here = sorted(set(sunk_here) | set(nonzero_core))

            sunk_set = set(sunk_here) | set(core_sinks)
            all_experts = sorted(int(k) for k in mlp.experts.keys())
            surviving_specialists = [
                e for e in all_experts if e not in sunk_set and not (protect_general and e == config.general_expert_index)
            ]

            svd_damped = 0
            svd_mod_elems = 0
            alpha_meta: dict[str, Any] = {"active_modes_x": 0, "active_modes_act": 0}
            router_meta: dict[str, Any] = {"applied": False, "rel_weight": 0.0}
            gate_eval: dict[str, Any] | None = None

            if has_twofold:
                scored = mlp_in[:, start_pos:-1, :]
                hd = int(scored.shape[-1])
                p0 = scored[prose_f0].reshape(-1, hd)
                p1 = scored[prose_f1].reshape(-1, hd)
                s0 = scored[spec_f0].reshape(-1, hd)
                s1 = scored[spec_f1].reshape(-1, hd)
                p_all = torch.cat([p0, p1], dim=0)
                s_all = torch.cat([s0, s1], dim=0)

                if config.svd_target_marginal_only:
                    svd_targets = [e for e in marginal_experts if e in surviving_specialists]
                else:
                    svd_targets = list(surviving_specialists)

                alpha_meta = _apply_optional_alphaedit_complement_inplace(
                    mlp,
                    p_all,
                    s_all,
                    svd_targets,
                    config,
                )

                if svd_targets and (svd_cfg.eta_svd > 0.0 or svd_cfg.truncate):
                    svd_rep = apply_moe_layer_svd_surgery_inplace(
                        mlp,
                        p_all,
                        s_all,
                        expert_indices=svd_targets,
                        config=svd_cfg,
                        include_w3=config.svd_include_w3,
                        total_model_params=total_model_params,
                    )
                    svd_damped = int(svd_rep["damped_modes"])
                    svd_mod_elems = int(svd_rep["modified_elements"])
                    total_damped_modes += svd_damped
                    total_svd_modified_elements += svd_mod_elems

                if (
                    config.router_beta_suppress > 0.0
                    or config.router_beta_sink > 0.0
                    or config.router_beta_general_boost > 0.0
                ):
                    u_global, probe_stats = fit_uncentered_prose_probe_with_oof(
                        p0,
                        p1,
                        s0,
                        s1,
                        omega_spec=config.router_omega_spec,
                        ridge_scale=config.router_ridge_scale,
                    )
                    rel = float(probe_stats["rel_weight"])
                    domain_x_map = {
                        e_idx: scored[idxs].reshape(-1, hd)
                        for e_idx, idxs in dom_indices.items()
                        if idxs and e_idx in surviving_specialists
                    }
                    if rel > 0.0:
                        if domain_x_map:
                            ctrl_vecs: torch.Tensor | dict[int, torch.Tensor] = fit_per_domain_orthogonal_gate_vectors(
                                p_all,
                                domain_x_map,
                                s_all,
                                omega_own=config.router_omega_own,
                                omega_other=config.router_omega_other,
                                ridge_scale=config.router_ridge_scale,
                                min_domain_samples=config.router_min_domain_samples,
                            )
                        else:
                            ctrl_vecs = u_global
                        num_router_experts = int(mlp.router.weight.shape[0])
                        sink_for_router = [e for e in sorted(sunk_set) if e < num_router_experts]
                        router_meta = apply_sigma_router_prose_control_inplace(
                            mlp.router,
                            ctrl_vecs,
                            rel_weight=rel,
                            beta_suppress=config.router_beta_suppress,
                            beta_sink=config.router_beta_sink,
                            beta_general_boost=config.router_beta_general_boost,
                            sink_experts=sink_for_router,
                            expert0_helps_prose=protect_general,
                            general_expert_index=config.general_expert_index,
                            sink_direction=u_global,
                            fallback_direction=u_global,
                        )
                        router_meta["oof_snr"] = float(probe_stats["oof_snr"])
                        if bool(router_meta.get("applied")):
                            router_layers_applied += 1
                        if domain_x_map:
                            gate_eval = evaluate_sigma_topk_routing_gates(
                                mlp.router,
                                p_all,
                                domain_x_map,
                                top_k=int(getattr(mlp, "top_k", 2)),
                                general_expert_index=config.general_expert_index,
                            )
                    else:
                        router_meta = {
                            "applied": False,
                            "rel_weight": 0.0,
                            "oof_snr": float(probe_stats["oof_snr"]),
                            "suppressed_experts": [],
                        }

            layer_reports.append(
                {
                    "layer": l_idx,
                    "sunk_experts": sunk_here,
                    "surviving_specialists": surviving_specialists,
                    "svd_damped_modes": svd_damped,
                    "svd_modified_elements": svd_mod_elems,
                    "alphaedit": alpha_meta,
                    "router_control": {
                        "applied": bool(router_meta.get("applied", False)),
                        "rel_weight": float(router_meta.get("rel_weight", 0.0)),
                        "suppressed_experts": router_meta.get("suppressed_experts", []),
                        "prose_offdomain_max_gate": (
                            float(gate_eval["prose_offdomain_max_gate"]) if gate_eval is not None else None
                        ),
                        "domain_min_own_gate": (
                            float(gate_eval["domain_min_own_gate"]) if gate_eval is not None else None
                        ),
                    },
                }
            )
        elif rogue_sunk_here:
            layer_reports.append(
                {
                    "layer": l_idx,
                    "sunk_experts": rogue_sunk_here,
                    "surviving_specialists": [],
                    "svd_damped_modes": 0,
                    "svd_modified_elements": 0,
                    "alphaedit": {"active_modes_x": 0, "active_modes_act": 0},
                    "router_control": {"applied": False, "rel_weight": 0.0, "suppressed_experts": []},
                }
            )

        x_seq = torch.cat(
            [forward_block_post_mlp(blk, x_mid[b0 : b0 + bsz], mlp_in[b0 : b0 + bsz]) for b0 in range(0, x_mid.shape[0], bsz)],
            dim=0,
        )

    post_stage3_buckets = _clone_buckets(extractor(model))
    receipt = attach_stage_receipt_sha256(
        "stage_3_surgery",
        {
            "input_buckets_sha256": compute_buckets_sha256(sanitized_buckets),
            "output_buckets_sha256": compute_buckets_sha256(post_stage3_buckets),
            "selected_layers": sorted(selected_set),
            "quarantined_rogue_layers": quarantined_rogue_layers,
            "total_svd_damped_modes": total_damped_modes,
            "total_svd_modified_elements": total_svd_modified_elements,
            "svd_parameter_fraction_modified": float(total_svd_modified_elements) / float(total_model_params),
            "router_layers_applied": router_layers_applied,
            "layers": layer_reports,
        },
    )
    return post_stage3_buckets, x_seq, receipt


# ---------------------------------------------------------------------------
# 6. Stage 4 — 21-Point Fine LayerNorm Guard Calibration
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_stage4_fine_layernorm_calibration(
    model: nn.Module,
    start_buckets: Mapping[str, torch.Tensor],
    tuned_reference_buckets: Mapping[str, torch.Tensor],
    post_stage3_buckets: Mapping[str, torch.Tensor],
    post_stage3_x_seq: torch.Tensor,
    probe_chunks: Sequence[Mapping[str, Any]],
    base_probe_losses: np.ndarray,
    *,
    config: WeightSpaceControllerConfig,
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
    extract_buckets_fn: Callable[[nn.Module], dict[str, torch.Tensor]] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Stage 4: run 21-point fine LayerNorm calibration on post-surgery pre-LN probe states."""
    _validate_model_compatible(model)
    loader, extractor = _resolve_bucket_io(
        post_stage3_buckets,
        load_buckets_fn,
        extract_buckets_fn,
        num_router_fragments=config.num_router_fragments,
    )
    ids_all, is_e0, _ = _validate_eval_chunks(
        probe_chunks,
        general_prefix=config.general_prefix,
        name="probe_chunks",
    )

    loader(model, _clone_buckets(post_stage3_buckets))
    model.eval()

    h_dim = int(model.ln.weight.numel())
    ln_eps = float(getattr(model.ln, "eps", 1e-5))
    start_ln_w, start_ln_b = _extract_ln_params_from_buckets(start_buckets, h_dim)
    tuned_ln_w, tuned_ln_b = _extract_ln_params_from_buckets(tuned_reference_buckets, h_dim)

    if config.forced_lambda_ln is not None:
        best_lambda_ln = float(config.forced_lambda_ln)
        _, ln_grid = select_guard_optimal_lambda_ln(
            post_stage3_x_seq,
            start_ln_w,
            start_ln_b,
            tuned_ln_w,
            tuned_ln_b,
            ln_eps,
            model.head.weight,
            ids_all,
            base_probe_losses,
            is_e0,
            score_from=config.score_from,
            target_e0_max_ratio=config.fine_ln_e0_max_ratio,
            lambda_grid=(best_lambda_ln,),
        )
    elif config.use_ln_calibration:
        best_lambda_ln, ln_grid = select_guard_optimal_lambda_ln(
            post_stage3_x_seq,
            start_ln_w,
            start_ln_b,
            tuned_ln_w,
            tuned_ln_b,
            ln_eps,
            model.head.weight,
            ids_all,
            base_probe_losses,
            is_e0,
            score_from=config.score_from,
            target_e0_max_ratio=config.fine_ln_e0_max_ratio,
            lambda_grid=FINE_LAMBDA_LN_GRID,
        )
    else:
        best_lambda_ln = 1.0
        ln_grid = {}

    model.ln.weight.copy_((start_ln_w + best_lambda_ln * (tuned_ln_w - start_ln_w)).to(model.ln.weight.dtype))
    model.ln.bias.copy_((start_ln_b + best_lambda_ln * (tuned_ln_b - start_ln_b)).to(model.ln.bias.dtype))

    post_stage4_buckets = _clone_buckets(extractor(model))
    selected_key = f"{best_lambda_ln:.2f}"
    selected_stats = ln_grid.get(selected_key, {})
    receipt = attach_stage_receipt_sha256(
        "stage_4_layernorm",
        {
            "input_buckets_sha256": compute_buckets_sha256(post_stage3_buckets),
            "output_buckets_sha256": compute_buckets_sha256(post_stage4_buckets),
            "grid_points": len(ln_grid),
            "best_lambda_ln": best_lambda_ln,
            "forced": config.forced_lambda_ln is not None,
            "selected_probe_e0_ratio": selected_stats.get("e0_ratio"),
            "selected_probe_e0_pct_vs_base": selected_stats.get("e0_pct_vs_base"),
            "selected_probe_nll_mean": selected_stats.get("nll_mean"),
        },
    )
    return post_stage4_buckets, receipt


# ---------------------------------------------------------------------------
# 7. Stage 5 — Thermodynamic ANSE Acceptance Gate & Atomic Bitwise Rollback
# ---------------------------------------------------------------------------


def evaluate_multidomain_anse_metrics(
    base_losses: np.ndarray | Sequence[float],
    cand_losses: np.ndarray | Sequence[float],
    expert_ids: Sequence[str],
    *,
    tuned_losses: np.ndarray | Sequence[float] | None = None,
    general_prefix: str = GENERAL_EXPERT_PREFIX,
    domain_guard_max_ratio: float = DOMAIN_GUARD_MAX_RATIO,
    anse_safe_e0_ratio: float = ANSE_SAFE_E0_RATIO,
    min_chunks_per_expert: int = 1,
) -> dict[str, Any]:
    """Compute per-domain NLL drifts, worst-domain guard compliance, specialist retention, and ANSE energy."""
    base_arr = np.asarray(base_losses, dtype=np.float64)
    cand_arr = np.asarray(cand_losses, dtype=np.float64)
    if base_arr.ndim != 1 or cand_arr.shape != base_arr.shape or len(expert_ids) != base_arr.shape[0]:
        raise ValueError("base_losses, cand_losses, and expert_ids must share the same 1D length")
    if base_arr.shape[0] == 0 or not bool(np.isfinite(base_arr).all()):
        raise ValueError("base_losses must be a non-empty finite 1D array")

    is_e0 = np.array([str(eid).startswith(general_prefix) for eid in expert_ids], dtype=bool)
    if not bool(np.any(is_e0)):
        raise ValueError(f"expert_ids must contain at least one general chunk starting with {general_prefix!r}")

    if not bool(np.isfinite(cand_arr).all()):
        return {
            "finite": False,
            "nll_mean": None,
            "base_nll_mean": float(base_arr.mean()),
            "mean_diff_vs_base": None,
            "e0_ratio": None,
            "e0_pct_vs_base": None,
            "worst_domain_id": None,
            "worst_domain_ratio": None,
            "worst_domain_pct_vs_base": None,
            "worst_domain_guard_ok": False,
            "spec_gain_nats": None,
            "spec_gain_retained_frac": None,
            "anse_energy": float(ANSE_MAXIMUM_PAIN),
            "per_domain": {},
        }

    order: list[str] = []
    idx_by_dom: dict[str, list[int]] = {}
    for idx, raw_eid in enumerate(expert_ids):
        eid = str(raw_eid)
        if eid not in idx_by_dom:
            order.append(eid)
            idx_by_dom[eid] = []
        idx_by_dom[eid].append(idx)

    per_domain: dict[str, dict[str, Any]] = {}
    for eid in order:
        ix = idx_by_dom[eid]
        b_m = float(base_arr[ix].mean())
        c_m = float(cand_arr[ix].mean())
        ratio = c_m / max(b_m, 1e-8)
        pct = 100.0 * (ratio - 1.0)
        qualifies = len(ix) >= min_chunks_per_expert
        per_domain[eid] = {
            "n": len(ix),
            "qualifies": qualifies,
            "base_nll": b_m,
            "cand_nll": c_m,
            "mean_diff": c_m - b_m,
            "ratio": ratio,
            "pct_vs_base": pct,
            "guard_ok": bool(ratio <= domain_guard_max_ratio),
        }

    qual_items = [(eid, info) for eid, info in per_domain.items() if info["qualifies"]]
    if not qual_items:
        qual_items = list(per_domain.items())

    worst_eid, worst_info = max(qual_items, key=lambda kv: float(kv[1]["ratio"]))
    worst_ratio = float(worst_info["ratio"])
    worst_pct = float(worst_info["pct_vs_base"])
    guard_ok = all(bool(info["guard_ok"]) for _, info in qual_items)

    base_e0 = float(base_arr[is_e0].mean())
    cand_e0 = float(cand_arr[is_e0].mean())
    e0_ratio = cand_e0 / max(base_e0, 1e-8)
    e0_pct = 100.0 * (e0_ratio - 1.0)

    has_spec = bool(np.any(~is_e0))
    spec_gain_nats = float(base_arr[~is_e0].mean() - cand_arr[~is_e0].mean()) if has_spec else 0.0
    spec_retained_frac: float | None = None
    if tuned_losses is not None and has_spec:
        tuned_arr = np.asarray(tuned_losses, dtype=np.float64)
        if tuned_arr.shape != base_arr.shape:
            raise ValueError("tuned_losses must match base_losses shape")
        if bool(np.isfinite(tuned_arr).all()):
            raw_gain = float(base_arr[~is_e0].mean() - tuned_arr[~is_e0].mean())
            spec_retained_frac = (spec_gain_nats / raw_gain) if abs(raw_gain) > 1e-8 else 0.0

    nll_mean = float(cand_arr.mean())
    base_mean = float(base_arr.mean())
    mean_diff = nll_mean - base_mean
    pain = ANSE_MAXIMUM_PAIN if not guard_ok else (ANSE_SOFT_MARGIN_PENALTY if e0_ratio > anse_safe_e0_ratio else 0.0)
    anse_energy = float(mean_diff + pain)

    return {
        "finite": True,
        "nll_mean": nll_mean,
        "base_nll_mean": base_mean,
        "mean_diff_vs_base": mean_diff,
        "e0_ratio": e0_ratio,
        "e0_pct_vs_base": e0_pct,
        "worst_domain_id": worst_eid,
        "worst_domain_ratio": worst_ratio,
        "worst_domain_pct_vs_base": worst_pct,
        "worst_domain_guard_ok": guard_ok,
        "spec_gain_nats": spec_gain_nats,
        "spec_gain_retained_frac": spec_retained_frac,
        "anse_energy": anse_energy,
        "per_domain": per_domain,
    }


@torch.no_grad()
def _eval_buckets_losses(
    model: nn.Module,
    buckets: Mapping[str, torch.Tensor],
    ids_all: torch.Tensor,
    loader: Callable[[nn.Module, Mapping[str, torch.Tensor]], None],
    *,
    score_from: int,
    batch_size: int,
) -> np.ndarray:
    """Evaluate exact per-chunk NLLs for ``buckets`` without failing on non-finite candidate states."""
    if not all(bool(torch.isfinite(v).all().item()) for v in buckets.values()):
        return np.full(int(ids_all.shape[0]), float("nan"), dtype=np.float64)
    loader(model, _clone_buckets(buckets))
    model.eval()
    h_pre = forward_model_pre_ln(model, ids_all, batch_size=batch_size)
    if not bool(torch.isfinite(h_pre).all().item()):
        return np.full(int(ids_all.shape[0]), float("nan"), dtype=np.float64)
    ln_eps = float(getattr(model.ln, "eps", 1e-5))
    return (
        per_chunk_causal_midrow_nll(
            h_pre,
            model.ln.weight,
            model.ln.bias,
            ln_eps,
            model.head.weight,
            ids_all,
            score_from=score_from,
            seq_batch_size=batch_size,
        )
        .cpu()
        .numpy()
        .astype(np.float64)
    )


@torch.no_grad()
def run_stage5_thermodynamic_anse_gate(
    model: nn.Module,
    start_buckets: Mapping[str, torch.Tensor],
    parent_buckets: Mapping[str, torch.Tensor],
    candidate_buckets: Mapping[str, torch.Tensor],
    eval_chunks: Sequence[Mapping[str, Any]],
    *,
    config: WeightSpaceControllerConfig,
    load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
    parent_state_dict: Mapping[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Stage 5: enforce thermodynamic ANSE gate (Delta E < 0 and worst-domain <= +1.0%) else atomic rollback."""
    _validate_model_compatible(model)
    loader, _ = _resolve_bucket_io(
        start_buckets,
        load_buckets_fn,
        None,
        num_router_fragments=config.num_router_fragments,
    )
    ids_eval, _, expert_ids = _validate_eval_chunks(
        eval_chunks,
        general_prefix=config.general_prefix,
        name="eval_chunks",
    )

    base_losses = _eval_buckets_losses(
        model, start_buckets, ids_eval, loader, score_from=config.score_from, batch_size=config.batch_size
    )
    parent_losses = _eval_buckets_losses(
        model, parent_buckets, ids_eval, loader, score_from=config.score_from, batch_size=config.batch_size
    )
    cand_losses = _eval_buckets_losses(
        model, candidate_buckets, ids_eval, loader, score_from=config.score_from, batch_size=config.batch_size
    )

    parent_eval = evaluate_multidomain_anse_metrics(
        base_losses,
        parent_losses,
        expert_ids,
        tuned_losses=parent_losses,
        general_prefix=config.general_prefix,
        domain_guard_max_ratio=config.domain_guard_max_ratio,
        anse_safe_e0_ratio=config.anse_safe_e0_ratio,
        min_chunks_per_expert=config.min_chunks_per_expert,
    )
    cand_eval = evaluate_multidomain_anse_metrics(
        base_losses,
        cand_losses,
        expert_ids,
        tuned_losses=parent_losses,
        general_prefix=config.general_prefix,
        domain_guard_max_ratio=config.domain_guard_max_ratio,
        anse_safe_e0_ratio=config.anse_safe_e0_ratio,
        min_chunks_per_expert=config.min_chunks_per_expert,
    )

    parent_sha = compute_buckets_sha256(parent_buckets)
    cand_sha = compute_buckets_sha256(candidate_buckets)
    delta_e = float(cand_eval["anse_energy"] - parent_eval["anse_energy"])

    if cand_sha == parent_sha:
        status = "FIXED_POINT_UNCHANGED"
        accepted = False
        rolled_back = False
        final_buckets = _clone_buckets(parent_buckets)
        if parent_state_dict is not None:
            model.load_state_dict(parent_state_dict)
        else:
            loader(model, final_buckets)
    elif bool(cand_eval["worst_domain_guard_ok"]) and (delta_e < -config.min_energy_improvement):
        status = "ACCEPTED"
        accepted = True
        rolled_back = False
        final_buckets = _clone_buckets(candidate_buckets)
        loader(model, final_buckets)
    else:
        status = (
            "ROLLED_BACK_GUARD_VIOLATION"
            if not bool(cand_eval["worst_domain_guard_ok"])
            else "ROLLED_BACK_ENERGY_NON_IMPROVING"
        )
        accepted = False
        rolled_back = True
        final_buckets = _clone_buckets(parent_buckets)
        if parent_state_dict is not None:
            model.load_state_dict(parent_state_dict)
        else:
            loader(model, final_buckets)

    final_sha = compute_buckets_sha256(final_buckets)
    receipt = attach_stage_receipt_sha256(
        "stage_5_anse_gate",
        {
            "parent_buckets_sha256": parent_sha,
            "candidate_buckets_sha256": cand_sha,
            "final_buckets_sha256": final_sha,
            "accepted": accepted,
            "rolled_back": rolled_back,
            "status": status,
            "parent_anse_energy": float(parent_eval["anse_energy"]),
            "candidate_anse_energy": float(cand_eval["anse_energy"]),
            "delta_e": delta_e,
            "domain_guard_max_ratio": config.domain_guard_max_ratio,
            "worst_domain_guard_ok": bool(cand_eval["worst_domain_guard_ok"]),
            "worst_domain_id": cand_eval["worst_domain_id"],
            "worst_domain_pct_vs_base": cand_eval["worst_domain_pct_vs_base"],
            "e0_pct_vs_base": cand_eval["e0_pct_vs_base"],
            "spec_gain_nats": cand_eval["spec_gain_nats"],
            "spec_gain_retained_frac": cand_eval["spec_gain_retained_frac"],
            "parent_metrics": parent_eval,
            "candidate_metrics": cand_eval,
        },
    )
    return final_buckets, receipt


# ---------------------------------------------------------------------------
# 8. Unified WeightSpaceController Orchestrator
# ---------------------------------------------------------------------------


class WeightSpaceController:
    """Deterministic zero-gradient 5-stage Weight-Space Programming controller for Sparse MoEs."""

    def __init__(
        self,
        config: WeightSpaceControllerConfig | None = None,
        *,
        load_buckets_fn: Callable[[nn.Module, Mapping[str, torch.Tensor]], None] | None = None,
        extract_buckets_fn: Callable[[nn.Module], dict[str, torch.Tensor]] | None = None,
    ) -> None:
        if config is not None and not isinstance(config, WeightSpaceControllerConfig):
            raise TypeError(f"config must be a WeightSpaceControllerConfig or None, got {type(config).__name__}")
        self.config = config or WeightSpaceControllerConfig()
        self._load_buckets_fn = load_buckets_fn
        self._extract_buckets_fn = extract_buckets_fn

    @torch.no_grad()
    def run(
        self,
        model: nn.Module,
        start_buckets: Mapping[str, torch.Tensor],
        probe_chunks: Sequence[Mapping[str, Any]],
        *,
        tuned_buckets: Mapping[str, torch.Tensor] | None = None,
        eval_chunks: Sequence[Mapping[str, Any]] | None = None,
    ) -> ControllerExecutionResult:
        """Execute all 5 WSP stages deterministically with atomic rollback on ANSE gate refusal."""
        _validate_model_compatible(model)
        loader, extractor = _resolve_bucket_io(
            start_buckets,
            self._load_buckets_fn,
            self._extract_buckets_fn,
            num_router_fragments=self.config.num_router_fragments,
        )

        was_training = model.training
        model.eval()
        try:
            parent_buckets = _clone_buckets(tuned_buckets if tuned_buckets is not None else extractor(model))
            _validate_bucket_pair(start_buckets, parent_buckets)
            loader(model, _clone_buckets(parent_buckets))
            parent_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

            # Stage 1: Finite & spectral preflight sanitization
            sanitized_buckets, stage1_receipt = run_stage1_preflight_sanitization(
                model,
                start_buckets,
                parent_buckets,
                config=self.config,
                load_buckets_fn=loader,
                extract_buckets_fn=extractor,
            )

            # Stage 2: LOEO partition + per-layer causal E0 harm attribution
            stage2_ctx, stage2_receipt = run_stage2_loeo_and_layer_attribution(
                model,
                start_buckets,
                sanitized_buckets,
                probe_chunks,
                config=self.config,
                load_buckets_fn=loader,
            )

            # Stage 3: Layer-localized sinking, QR-SVD mode surgery, and multi-gate control
            post_stage3_buckets, post_stage3_x_seq, stage3_receipt = run_stage3_wsp_surgery(
                model,
                sanitized_buckets,
                probe_chunks,
                stage2_ctx,
                config=self.config,
                load_buckets_fn=loader,
                extract_buckets_fn=extractor,
            )

            # Stage 4: 21-point fine LayerNorm guard calibration
            post_stage4_buckets, stage4_receipt = run_stage4_fine_layernorm_calibration(
                model,
                start_buckets,
                sanitized_buckets,
                post_stage3_buckets,
                post_stage3_x_seq,
                probe_chunks,
                stage2_ctx["base_probe_losses"],
                config=self.config,
                load_buckets_fn=loader,
                extract_buckets_fn=extractor,
            )

            # Stage 5: Thermodynamic ANSE acceptance gate & atomic bitwise rollback
            eval_split = probe_chunks if eval_chunks is None else eval_chunks
            final_buckets, stage5_receipt = run_stage5_thermodynamic_anse_gate(
                model,
                start_buckets,
                parent_buckets,
                post_stage4_buckets,
                eval_split,
                config=self.config,
                load_buckets_fn=loader,
                parent_state_dict=parent_state_dict,
            )
        finally:
            model.train(was_training)

        stages = {
            "stage_1_preflight": stage1_receipt,
            "stage_2_attribution": stage2_receipt,
            "stage_3_surgery": stage3_receipt,
            "stage_4_layernorm": stage4_receipt,
            "stage_5_anse_gate": stage5_receipt,
        }
        stage_hashes = {name: stages[name]["stage_receipt_sha256"] for name in STAGE_ORDER}
        receipt_body: dict[str, Any] = {
            "object": CONTROLLER_RECEIPT_OBJECT,
            "schema": CONTROLLER_SCHEMA,
            "eval_split_source": "probe_chunks" if eval_chunks is None else "eval_chunks",
            "config": asdict(self.config),
            "start_buckets_sha256": compute_buckets_sha256(start_buckets),
            "parent_buckets_sha256": compute_buckets_sha256(parent_buckets),
            "final_buckets_sha256": compute_buckets_sha256(final_buckets),
            "accepted": bool(stage5_receipt["accepted"]),
            "rolled_back": bool(stage5_receipt["rolled_back"]),
            "status": str(stage5_receipt["status"]),
            "stage_hashes": stage_hashes,
            "stages": stages,
        }
        receipt_sha = compute_payload_sha256(receipt_body)
        receipt_body["receipt_sha256"] = receipt_sha

        return ControllerExecutionResult(
            buckets=final_buckets,
            receipt=receipt_body,
            accepted=bool(stage5_receipt["accepted"]),
            rolled_back=bool(stage5_receipt["rolled_back"]),
            status=str(stage5_receipt["status"]),
            receipt_sha256=receipt_sha,
        )


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "ANSE_SAFE_E0_RATIO",
    "ANSE_SOFT_MARGIN_PENALTY",
    "CONTROLLER_RECEIPT_OBJECT",
    "CONTROLLER_SCHEMA",
    "CORE_SINK_MAX_SPEC_DELTA",
    "DEFAULT_GENERAL_EXPERT_INDEX",
    "DEFAULT_SCORE_FROM",
    "DOMAIN_GUARD_MAX_RATIO",
    "FINE_LAMBDA_LN_GRID",
    "FINE_LN_GUARD_MAX_RATIO",
    "GENERAL_EXPERT_PREFIX",
    "STAGE_ORDER",
    "ControllerExecutionResult",
    "WeightSpaceController",
    "WeightSpaceControllerConfig",
    "attach_stage_receipt_sha256",
    "canonical_json_bytes",
    "compute_buckets_sha256",
    "compute_payload_sha256",
    "evaluate_multidomain_anse_metrics",
    "partition_loeo_experts",
    "run_stage1_preflight_sanitization",
    "run_stage2_loeo_and_layer_attribution",
    "run_stage3_wsp_surgery",
    "run_stage4_fine_layernorm_calibration",
    "run_stage5_thermodynamic_anse_gate",
    "verify_controller_receipt_provenance",
]

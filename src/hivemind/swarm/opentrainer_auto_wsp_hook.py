# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Automatic Post-Outer-Round S5 WSP Surgery Hook & ExpertRegistry Canary Gate (HM-S35-04 / HM-S35-05).

Executes CPU-only training-free S5 PASS-Synergy Weight-Space Programming (WSP) surgery
automatically after every nocturnal DiLoCo / QLoRA outer round in OpenTrainer, gating
checkpoint lineage and ``ExpertRegistry`` ``candidate -> shadow -> live`` promotion under
strict ANSE thermodynamic monotonicity:

1. **HM-S35-04 — Automatic Post-Outer-Round S5 Surgery Hook**
   (``run_post_outer_round_auto_wsp_hook``):
   - Diagnoses beneficial-anchor vs interfering layers (``identify_beneficial_anchor_and_interfering_layers``)
     and separates pure prose-harming/rogue FoMoE ghost sinks from dual-active specialist
     experts (``partition_sink_vs_preserved_specialists``).
   - Applies four monotone S5 PASS-Synergy stages protecting beneficial anchor layers
     bitwise:
       (1) Selective FoMoE ghost-sink LoRA ``B`` zeroing (``w1_B = w2_B = w3_B = 0 => dy == 0``),
       (2) Cross-layer anchor residual null-space projection on downstream interfering layers,
       (3) 2-fold out-of-fold SNR-gated orthogonal multi-gate ``sigma_topk`` router surgery,
       (4) Guard-constrained final norm interpolation & folded ``1 / tau`` temperature calibration.
   - Promotes the sanitized S5 checkpoint to ``promoted_shadow`` if and only if
     ``s5_anse_energy < raw_anse_energy`` and ``general_rel_drift <= general_guard_rel``.
     Otherwise cleanly falls back bitwise to ``raw_outer_state`` (``rejected_fallback_raw``)
     on already-clean rounds with zero gain, or rolls back bitwise to ``parent_state``
     (``rejected_fallback_parent``) with ANSE Maximum Pain (``1e6``) on corrupted or
     infeasible rounds.

2. **HM-S35-05 — Shadow-to-Live Canary & ExpertRegistry Promotion Gate**
   (``evaluate_shadow_to_live_canary_gate``):
   - Requires a tamper-evident PASS WSP lineage receipt (``promoted_shadow``),
     ``shadow_general_drift_rel <= max_general_drift_rel``, and canary router probe pass
     ratio ``>= min_pass_ratio`` (default ``15 / 15 = 1.0``) before approving promotion
     to ``promoted_live`` and driving strict ``ExpertRegistry.promote_with_gate(...)``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch

from hivemind.core.expert_registry import (
    CANDIDATE as EXPERT_CANDIDATE,
)
from hivemind.core.expert_registry import (
    LORA_DOMAIN as EXPERT_LORA_DOMAIN,
)
from hivemind.core.expert_registry import (
    SHADOW as EXPERT_SHADOW,
)
from hivemind.core.expert_registry import (
    ExpertRecord,
    ExpertRegistry,
)
from hivemind.swarm.cross_layer_residual_nullspace import (
    identify_beneficial_anchor_and_interfering_layers,
)
from hivemind.swarm.logit_calibration_temperature_surgeon import (
    fold_temperature_into_norm_params,
)
from hivemind.swarm.monotone_wsp_stage_composer import (
    DEFAULT_ANSE_MAXIMUM_PAIN,
    DEFAULT_ANSE_SOFT_MARGIN_PENALTY,
    DEFAULT_GENERAL_GUARD_REL,
    GENERAL_EXPERT_PREFIX,
    WSPStageSpec,
    clone_state_dict,
    compute_state_dict_sha256,
    count_edited_parameters,
    evaluate_anse_domain_metrics,
    make_cross_layer_anchor_stage,
    make_ghost_sink_stage,
    parse_eval_domain_nll,
)
from hivemind.swarm.orthogonal_multigate_router_surgeon import (
    fit_oof_orthogonal_multigate_row_update,
    partition_sink_vs_preserved_specialists,
)
from hivemind.swarm.quantization_compensated_wsp import compute_payload_sha256

SCHEMA_VERSION: str = "hivemind.opentrainer_auto_wsp_hook/1"
LINEAGE_RECEIPT_OBJECT: str = "hivemind.post_round_wsp_lineage_receipt"
CANARY_GATE_OBJECT: str = "hivemind.expert_registry_shadow_live_canary_gate"

PROMOTION_STAGE_REJECTED_RAW: str = "rejected_fallback_raw"
PROMOTION_STAGE_REJECTED_PARENT: str = "rejected_fallback_parent"
PROMOTION_STAGE_PROMOTED_SHADOW: str = "promoted_shadow"
PROMOTION_STAGE_PROMOTED_LIVE: str = "promoted_live"
VALID_PROMOTION_STAGES: tuple[str, ...] = (
    PROMOTION_STAGE_REJECTED_RAW,
    PROMOTION_STAGE_REJECTED_PARENT,
    PROMOTION_STAGE_PROMOTED_SHADOW,
    PROMOTION_STAGE_PROMOTED_LIVE,
)
VALID_WSP_VERDICTS: tuple[str, ...] = ("PASS", "NO_GAIN", "FAIL")

DEFAULT_MAX_TOTAL_EDIT_FRACTION: float = 0.25
DEFAULT_CANARY_PROBE_TOTAL: int = 15
FINE_LAMBDA_LN_GRID: tuple[float, ...] = tuple(round(0.05 * idx, 2) for idx in range(21))
DEFAULT_TEMPERATURE_GRID: tuple[float, ...] = (0.80, 0.90, 1.0, 1.10, 1.25)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_EXPERT_W3_B_KEY_RE = re.compile(r"^blocks\.(\d+)\.mlp\.experts\.(\d+)\.w3_B$")
_ROUTER_KEY_RE = re.compile(r"^blocks\.(\d+)\.mlp\.router\.weight$")
_LN_WEIGHT_FALLBACK_KEYS: tuple[str, ...] = ("ln.weight", "model.ln.weight", "norm.weight", "ln_f.weight")
_LN_BIAS_FALLBACK_KEYS: tuple[str, ...] = ("ln.bias", "model.ln.bias", "norm.bias", "ln_f.bias")
_BARRIER_BASE_OFFSET: float = 10.0
_BARRIER_EXCESS_WEIGHT: float = 100.0


# ---------------------------------------------------------------------------
# 1. Validation & Canonical JSON/Tensor Helpers
# ---------------------------------------------------------------------------


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_nonempty_str(value: Any, name: str) -> str:
    """Validate that ``value`` is a non-empty string without control characters."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped or any(ord(ch) < 32 or ord(ch) == 127 for ch in stripped):
        raise ValueError(f"{name} must be a non-empty printable string")
    return stripped


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


def _validate_unit_interval(value: Any, name: str, *, allow_zero: bool = True) -> float:
    """Validate that ``value`` lies inside ``[0, 1]`` or ``(0, 1]``."""
    out = _validate_nonneg_finite(value, name)
    if not allow_zero and out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    if out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    return out


def _is_tensor_finite(tensor: torch.Tensor) -> bool:
    """Return True iff every floating/complex entry of ``tensor`` is finite."""
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return True
    return bool(torch.isfinite(tensor.detach()).all().item())


def _is_state_dict_finite(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Return True iff every tensor in ``state_dict`` is finite."""
    return all(_is_tensor_finite(t) for t in state_dict.values())


def _validate_state_mapping(
    state_dict: Any,
    name: str,
    *,
    require_finite: bool,
) -> tuple[dict[str, torch.Tensor], bool]:
    """Validate a non-empty tensor mapping and return ``(cloned_state, all_finite)``."""
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError(f"{name} must be a non-empty mapping of str -> torch.Tensor")
    out: dict[str, torch.Tensor] = {}
    total_numel = 0
    all_finite = True
    for key, val in state_dict.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings, got {key!r}")
        if not isinstance(val, torch.Tensor):
            raise TypeError(f"{name}[{key!r}] must be a torch.Tensor, got {type(val).__name__}")
        t_finite = _is_tensor_finite(val)
        if not t_finite:
            if require_finite:
                raise ValueError(f"{name}[{key!r}] contains non-finite values")
            all_finite = False
        total_numel += int(val.numel())
        out[key] = val.detach().clone()
    if total_numel <= 0:
        raise ValueError(f"{name} must contain at least one scalar parameter")
    return out, all_finite


def _validate_matching_state_shapes(
    reference: Mapping[str, torch.Tensor],
    other: Mapping[str, torch.Tensor],
    ref_name: str,
    other_name: str,
) -> None:
    """Verify two state dictionaries share identical keys and tensor shapes."""
    if set(reference.keys()) != set(other.keys()):
        missing = sorted(set(reference.keys()) - set(other.keys()))
        extra = sorted(set(other.keys()) - set(reference.keys()))
        raise ValueError(
            f"{other_name} keys do not match {ref_name} (missing={missing[:3]}, extra={extra[:3]})"
        )
    for key in reference:
        if reference[key].shape != other[key].shape:
            raise ValueError(
                f"Shape mismatch on {key!r} between {ref_name} {tuple(reference[key].shape)} "
                f"and {other_name} {tuple(other[key].shape)}"
            )


def _json_safe(value: Any) -> Any:
    """Recursively convert tuples into lists so payload JSON hashing is canonical."""
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def _parse_layer_index_key(raw_key: Any, name: str) -> int:
    """Normalize an int or numeric/block-prefixed string key into a non-negative layer index."""
    if isinstance(raw_key, bool):
        raise TypeError(f"{name} layer key must not be bool")
    if isinstance(raw_key, int):
        return _validate_nonneg_int(raw_key, f"{name} layer key")
    if isinstance(raw_key, str):
        text = raw_key.strip()
        if text.isdigit():
            return int(text)
        if text.startswith("blocks.") and text[7:].isdigit():
            return int(text[7:])
    raise ValueError(f"Invalid layer index key {raw_key!r} in {name}")


def _to_2d_cpu_float64(tensor: Any, name: str) -> torch.Tensor:
    """Validate a finite 2D/3D floating-point activation tensor and flatten to ``(N, D)`` float64."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim not in (2, 3) or any(dim == 0 for dim in tensor.shape):
        raise ValueError(f"{name} must be a non-empty 2D or 3D tensor, got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}")
    flat = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1, int(tensor.shape[-1]))
    if not bool(torch.isfinite(flat).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return flat.contiguous()


def _normalize_layer_mapping(raw_inputs: Any, name: str, value_fn: Callable[[Any, int], Any]) -> dict[int, Any]:
    """Normalize a layer-indexed Mapping or Sequence using ``value_fn(item, layer_idx)``."""
    out: dict[int, Any] = {}
    if isinstance(raw_inputs, Mapping):
        if not raw_inputs:
            raise ValueError(f"{name} mapping must not be empty")
        for raw_k, raw_v in raw_inputs.items():
            l_idx = _parse_layer_index_key(raw_k, name)
            if l_idx in out:
                raise ValueError(f"Duplicate layer index {l_idx} in {name}")
            out[l_idx] = value_fn(raw_v, l_idx)
        return out
    if isinstance(raw_inputs, (str, bytes, bytearray)) or not isinstance(raw_inputs, Sequence):
        raise TypeError(f"{name} must be a Mapping or Sequence")
    if len(raw_inputs) == 0:
        raise ValueError(f"{name} sequence must not be empty")
    for idx, raw_v in enumerate(raw_inputs):
        out[idx] = value_fn(raw_v, idx)
    return out


def _normalize_layer_prose_inputs(raw_inputs: Any) -> dict[int, torch.Tensor]:
    """Normalize per-layer general prose activations into ``{layer_idx: (N, H) float64}``."""
    return _normalize_layer_mapping(
        raw_inputs,
        "layer_prose_inputs",
        lambda val, idx: _to_2d_cpu_float64(val, f"layer_prose_inputs[{idx}]"),
    )


def _normalize_single_layer_spec(spec_val: Any, layer_idx: int) -> dict[str, torch.Tensor] | torch.Tensor:
    """Normalize one layer's specialist activations into 2D float64 tensor(s)."""
    if isinstance(spec_val, torch.Tensor):
        return _to_2d_cpu_float64(spec_val, f"layer_specialist_inputs[{layer_idx}]")
    if isinstance(spec_val, Mapping):
        if not spec_val:
            raise ValueError(f"layer_specialist_inputs[{layer_idx}] mapping must not be empty")
        normalized: dict[str, torch.Tensor] = {}
        for dom_k, dom_t in spec_val.items():
            if isinstance(dom_k, bool) or not isinstance(dom_k, (int, str)):
                raise TypeError(f"Domain key in layer_specialist_inputs[{layer_idx}] must be int or str")
            key_str = str(dom_k).strip()
            if not key_str:
                raise ValueError(f"Domain key in layer_specialist_inputs[{layer_idx}] must be non-empty")
            normalized[key_str] = _to_2d_cpu_float64(
                dom_t, f"layer_specialist_inputs[{layer_idx}][{dom_k!r}]"
            )
        return normalized
    if isinstance(spec_val, (str, bytes, bytearray)) or not isinstance(spec_val, Sequence):
        raise TypeError(
            f"layer_specialist_inputs[{layer_idx}] must be a Tensor, Mapping, or Sequence of Tensors"
        )
    if len(spec_val) == 0:
        raise ValueError(f"layer_specialist_inputs[{layer_idx}] sequence must not be empty")
    return {
        f"domain_{idx}": _to_2d_cpu_float64(dom_t, f"layer_specialist_inputs[{layer_idx}][{idx}]")
        for idx, dom_t in enumerate(spec_val)
    }


def _normalize_layer_specialist_inputs(raw_inputs: Any) -> dict[int, dict[str, torch.Tensor] | torch.Tensor]:
    """Normalize per-layer specialist activations keyed by layer index."""
    return _normalize_layer_mapping(raw_inputs, "layer_specialist_inputs", _normalize_single_layer_spec)


# ---------------------------------------------------------------------------
# 2. Configuration & Tamper-Evident Lineage Receipt Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PostRoundWSPHookConfig:
    """Configuration governing automatic post-round S5 surgery and ANSE ratchet gating."""

    general_prefix: str = GENERAL_EXPERT_PREFIX
    general_expert_index: int = 0
    general_guard_rel: float = DEFAULT_GENERAL_GUARD_REL
    max_total_edit_fraction: float = DEFAULT_MAX_TOTAL_EDIT_FRACTION
    min_energy_improvement: float = 0.0
    anchor_threshold_nats: float = -0.015
    harm_threshold_nats: float = 0.002
    protect_anchor_layers: bool = True
    enable_anchor_nullspace_projection: bool = True
    anchor_proj_strength: float = 1.0
    spec_preserve_floor_nats: float = 0.015
    prose_harm_floor_nats: float = 0.002
    rogue_expert_indices: tuple[int, ...] = ()
    router_beta: float = 2.0
    sink_attract_scale: float = 0.35
    min_oof_snr: float = 0.25
    n_folds: int = 2
    omega_spec: float = 6.0
    ridge_scale: float = 0.75
    sv_rel_tol: float = 1e-10
    gate_all_surviving_specialists: bool = False
    use_ln_calibration: bool = True
    lambda_ln_grid: tuple[float, ...] = FINE_LAMBDA_LN_GRID
    temperature_grid: tuple[float, ...] = DEFAULT_TEMPERATURE_GRID
    ln_weight_key: str = "ln.weight"
    ln_bias_key: str = "ln.bias"
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY
    maximum_pain: float = DEFAULT_ANSE_MAXIMUM_PAIN
    sanitize_non_finite_ghost_sinks: bool = False
    base_model_id: str = "hivemind/crowd-v1"

    def __post_init__(self) -> None:
        _validate_nonempty_str(self.general_prefix, "general_prefix")
        _validate_nonneg_int(self.general_expert_index, "general_expert_index")
        _validate_nonneg_finite(self.general_guard_rel, "general_guard_rel")
        _validate_unit_interval(self.max_total_edit_fraction, "max_total_edit_fraction", allow_zero=False)
        _validate_nonneg_finite(self.min_energy_improvement, "min_energy_improvement")

        anchor_t = _validate_finite_float(self.anchor_threshold_nats, "anchor_threshold_nats")
        harm_t = _validate_nonneg_finite(self.harm_threshold_nats, "harm_threshold_nats")
        if anchor_t > 0.0 or anchor_t >= harm_t:
            raise ValueError(
                f"anchor_threshold_nats ({anchor_t}) must be <= 0.0 and strictly < harm_threshold_nats ({harm_t})"
            )

        for flag_name in (
            "protect_anchor_layers",
            "enable_anchor_nullspace_projection",
            "gate_all_surviving_specialists",
            "use_ln_calibration",
            "sanitize_non_finite_ghost_sinks",
        ):
            _validate_bool(getattr(self, flag_name), flag_name)

        _validate_unit_interval(self.anchor_proj_strength, "anchor_proj_strength")
        _validate_nonneg_finite(self.spec_preserve_floor_nats, "spec_preserve_floor_nats")
        _validate_nonneg_finite(self.prose_harm_floor_nats, "prose_harm_floor_nats")

        if isinstance(self.rogue_expert_indices, (str, bytes)) or not isinstance(
            self.rogue_expert_indices, Sequence
        ):
            raise TypeError("rogue_expert_indices must be a sequence of non-negative integers")
        rogue_norm = tuple(
            sorted({_validate_nonneg_int(idx, "rogue_expert_index") for idx in self.rogue_expert_indices})
        )
        object.__setattr__(self, "rogue_expert_indices", rogue_norm)

        _validate_nonneg_finite(self.router_beta, "router_beta")
        _validate_nonneg_finite(self.sink_attract_scale, "sink_attract_scale")
        _validate_nonneg_finite(self.min_oof_snr, "min_oof_snr")
        folds = _validate_positive_int(self.n_folds, "n_folds")
        if folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {folds}")
        _validate_nonneg_finite(self.omega_spec, "omega_spec")
        _validate_nonneg_finite(self.ridge_scale, "ridge_scale")
        _validate_nonneg_finite(self.sv_rel_tol, "sv_rel_tol")

        if isinstance(self.lambda_ln_grid, (str, bytes)) or not isinstance(self.lambda_ln_grid, Sequence) or not self.lambda_ln_grid:
            raise ValueError("lambda_ln_grid must be a non-empty sequence of finite floats")
        lam_norm = tuple(_validate_finite_float(v, "lambda_ln_grid entry") for v in self.lambda_ln_grid)
        object.__setattr__(self, "lambda_ln_grid", lam_norm)

        if isinstance(self.temperature_grid, (str, bytes)) or not isinstance(self.temperature_grid, Sequence) or not self.temperature_grid:
            raise ValueError("temperature_grid must be a non-empty sequence of positive floats")
        temp_norm = tuple(_validate_positive_finite(v, "temperature_grid entry") for v in self.temperature_grid)
        object.__setattr__(self, "temperature_grid", temp_norm)

        _validate_nonempty_str(self.ln_weight_key, "ln_weight_key")
        _validate_nonempty_str(self.ln_bias_key, "ln_bias_key")
        _validate_nonempty_str(self.base_model_id, "base_model_id")
        _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty")
        _validate_positive_finite(self.maximum_pain, "maximum_pain")


@dataclass(frozen=True)
class PostRoundWSPLineageReceipt:
    """Tamper-evident A/B lineage & promotion receipt for one nocturnal outer round."""

    round_id: str
    parent_checkpoint_sha256: str
    raw_outer_sha256: str
    sanitized_s5_sha256: str
    raw_anse_energy: float
    s5_anse_energy: float
    general_rel_drift: float
    promotion_stage: str
    active_checkpoint_sha256: str
    has_explicit_parent: bool
    parent_anse_energy: float
    active_anse_energy: float
    anse_energy: float
    delta_anse_energy: float
    base_nll_mean: float
    raw_nll_mean: float
    s5_nll_mean: float
    nll_mean: float
    raw_general_rel_drift: float
    s5_general_rel_drift: float
    parent_general_rel_drift: float
    active_general_rel_drift: float
    worst_domain_rel_drift: float
    base_domain_nll: dict[str, float]
    raw_domain_nll: dict[str, float]
    s5_domain_nll: dict[str, float]
    active_domain_nll: dict[str, float]
    raw_guard_ok: bool
    s5_guard_ok: bool
    guard_ok: bool
    wsp_verdict: str
    verdict: str
    reason: str
    beneficial_anchor_layers: tuple[int, ...]
    interfering_layers: tuple[int, ...]
    editable_layers: tuple[int, ...]
    anchor_layers_bitwise_preserved: bool
    sink_experts: tuple[int, ...]
    preserved_specialists: tuple[int, ...]
    gated_targets: tuple[int, ...]
    committed_stages: tuple[str, ...]
    rolled_back_stages: tuple[str, ...]
    skipped_stages: tuple[str, ...]
    lambda_ln_applied: float | None
    temperature_applied: float | None
    total_params: int
    edited_params_vs_raw: int
    edit_fraction_vs_raw: float
    base_model_id: str
    object: str = LINEAGE_RECEIPT_OBJECT
    schema: str = SCHEMA_VERSION
    canary_passed: bool | None = None
    canary_gate_report: dict[str, Any] | None = None
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        _validate_nonempty_str(self.round_id, "round_id")
        if self.promotion_stage not in VALID_PROMOTION_STAGES:
            raise ValueError(
                f"promotion_stage must be one of {VALID_PROMOTION_STAGES}, got {self.promotion_stage!r}"
            )
        if self.wsp_verdict not in VALID_WSP_VERDICTS:
            raise ValueError(f"wsp_verdict must be one of {VALID_WSP_VERDICTS}, got {self.wsp_verdict!r}")
        for sha_field in (
            "parent_checkpoint_sha256",
            "raw_outer_sha256",
            "sanitized_s5_sha256",
            "active_checkpoint_sha256",
        ):
            val = getattr(self, sha_field)
            if not isinstance(val, str) or not _SHA256_HEX_RE.match(val):
                raise ValueError(f"{sha_field} must be a 64-hex SHA-256 string, got {val!r}")
        for float_field in (
            "raw_anse_energy",
            "s5_anse_energy",
            "general_rel_drift",
            "parent_anse_energy",
            "active_anse_energy",
            "anse_energy",
            "delta_anse_energy",
            "base_nll_mean",
            "raw_nll_mean",
            "s5_nll_mean",
            "nll_mean",
            "raw_general_rel_drift",
            "s5_general_rel_drift",
            "parent_general_rel_drift",
            "active_general_rel_drift",
            "worst_domain_rel_drift",
            "edit_fraction_vs_raw",
        ):
            _validate_finite_float(getattr(self, float_field), float_field)
        if self.receipt_sha256 != "" and not _SHA256_HEX_RE.match(self.receipt_sha256):
            raise ValueError("receipt_sha256 must be empty or a 64-hex SHA-256 string")

    @property
    def passed(self) -> bool:
        """Return True iff the WSP surgery verdict is ``'PASS'``."""
        return self.wsp_verdict == "PASS"

    @property
    def live_promoted(self) -> bool:
        """Return True iff the receipt has cleared the shadow-to-live canary gate."""
        return self.promotion_stage == PROMOTION_STAGE_PROMOTED_LIVE

    def to_dict(self) -> dict[str, Any]:
        """Serialize receipt to a canonical JSON-compatible dictionary."""
        return _json_safe(asdict(self))


def _seal_lineage_receipt(fields: Mapping[str, Any]) -> PostRoundWSPLineageReceipt:
    """Attach canonical ``receipt_sha256`` to a ``PostRoundWSPLineageReceipt``."""
    core = {k: v for k, v in fields.items() if k != "receipt_sha256"}
    unsealed = PostRoundWSPLineageReceipt(**core, receipt_sha256="")
    payload = {k: v for k, v in unsealed.to_dict().items() if k != "receipt_sha256"}
    sha = compute_payload_sha256(payload)
    return replace(unsealed, receipt_sha256=sha)


def verify_lineage_receipt_provenance(receipt: PostRoundWSPLineageReceipt | Mapping[str, Any]) -> bool:
    """Independently verify ``receipt_sha256`` over a ``PostRoundWSPLineageReceipt`` or dict."""
    if isinstance(receipt, PostRoundWSPLineageReceipt):
        payload: Mapping[str, Any] = receipt.to_dict()
    elif isinstance(receipt, Mapping):
        payload = receipt
    else:
        raise TypeError(
            f"receipt must be a PostRoundWSPLineageReceipt or Mapping, got {type(receipt).__name__}"
        )
    claimed = payload.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    core = {k: _json_safe(v) for k, v in payload.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


# ---------------------------------------------------------------------------
# 3. Internal Evaluation & S5 PASS-Synergy Surgery Stage Builders
# ---------------------------------------------------------------------------


def _anchor_layers_sha256(state_dict: Mapping[str, torch.Tensor], anchor_layers: Sequence[int]) -> str:
    """Compute a deterministic digest of all tensors belonging to protected anchor layers."""
    if not anchor_layers:
        return ""
    prefixes = tuple(f"blocks.{l_idx}." for l_idx in sorted(set(anchor_layers)))
    subset = {k: v for k, v in state_dict.items() if k.startswith(prefixes)}
    if not subset:
        return ""
    return compute_state_dict_sha256(subset)


def _compute_shaping_energy(metrics: Mapping[str, Any], *, guard_rel: float, maximum_pain: float, finite: bool) -> float:
    """Compute smooth out-of-guard barrier energy so partial E0 drift reduction progresses monotonically."""
    if not finite:
        return float(maximum_pain)
    if bool(metrics["guard_ok"]):
        return float(metrics["anse_energy"])
    worst_drift = float(metrics["worst_domain_drift"])
    excess = max(0.0, worst_drift - guard_rel)
    shaped = float(metrics["mean_diff_vs_base"]) + _BARRIER_BASE_OFFSET + _BARRIER_EXCESS_WEIGHT * excess
    return min(shaped, float(maximum_pain) - 1.0)


def _evaluate_state_metrics(
    eval_fn: Callable[[Mapping[str, torch.Tensor]], Any],
    state_dict: Mapping[str, torch.Tensor],
    base_domain_nll: Mapping[str, float],
    cfg: PostRoundWSPHookConfig,
    *,
    allow_runtime_eval_error: bool,
) -> tuple[dict[str, Any], dict[str, float], bool]:
    """Evaluate ``state_dict`` and compute strict ANSE metrics plus domain NLL map."""
    domain_order = tuple(sorted(base_domain_nll.keys()))
    if not _is_state_dict_finite(state_dict):
        pain_nll = {d: float(cfg.maximum_pain) for d in domain_order}
        metrics = evaluate_anse_domain_metrics(
            base_domain_nll,
            pain_nll,
            pre_domain_nll=base_domain_nll,
            general_guard_rel=cfg.general_guard_rel,
            general_prefix=cfg.general_prefix,
            soft_margin_penalty=cfg.soft_margin_penalty,
            maximum_pain=cfg.maximum_pain,
            cand_finite=False,
        )
        return metrics, pain_nll, False

    try:
        raw_out = eval_fn(clone_state_dict(state_dict))
    except (ArithmeticError, RuntimeError):
        if not allow_runtime_eval_error:
            raise
        pain_nll = {d: float(cfg.maximum_pain) for d in domain_order}
        metrics = evaluate_anse_domain_metrics(
            base_domain_nll,
            pain_nll,
            pre_domain_nll=base_domain_nll,
            general_guard_rel=cfg.general_guard_rel,
            general_prefix=cfg.general_prefix,
            soft_margin_penalty=cfg.soft_margin_penalty,
            maximum_pain=cfg.maximum_pain,
            cand_finite=False,
        )
        return metrics, pain_nll, False

    cand_nll, eval_finite = parse_eval_domain_nll(
        raw_out,
        allow_non_finite=True,
        maximum_pain=cfg.maximum_pain,
    )
    if set(cand_nll.keys()) != set(base_domain_nll.keys()):
        raise ValueError(
            f"eval_fn returned mismatched domain keys: {sorted(cand_nll.keys())} != {sorted(base_domain_nll.keys())}"
        )
    metrics = evaluate_anse_domain_metrics(
        base_domain_nll,
        cand_nll,
        pre_domain_nll=base_domain_nll,
        general_guard_rel=cfg.general_guard_rel,
        general_prefix=cfg.general_prefix,
        soft_margin_penalty=cfg.soft_margin_penalty,
        maximum_pain=cfg.maximum_pain,
        cand_finite=eval_finite,
    )
    return metrics, cand_nll, eval_finite


def _build_anchor_prose_subspace(
    state_dict: Mapping[str, torch.Tensor],
    anchor_layers: Sequence[int],
    anchor_experts: Sequence[int],
) -> torch.Tensor | None:
    """Collect finite LoRA ``w3_B`` output columns from anchor-layer general experts."""
    anchor_set = set(anchor_layers)
    exp_set = set(anchor_experts)
    cols: list[torch.Tensor] = []
    hidden_dim: int | None = None
    for key in sorted(state_dict.keys()):
        match = _EXPERT_W3_B_KEY_RE.match(key)
        if match is None:
            continue
        l_idx, e_idx = int(match.group(1)), int(match.group(2))
        if l_idx not in anchor_set or e_idx not in exp_set:
            continue
        tensor = state_dict[key]
        if tensor.ndim != 2 or tensor.numel() == 0 or not tensor.is_floating_point():
            continue
        t_64 = tensor.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(t_64).all().item()) or float(t_64.norm().item()) <= 1e-10:
            continue
        if hidden_dim is None:
            hidden_dim = int(t_64.shape[0])
        elif int(t_64.shape[0]) != hidden_dim:
            continue
        cols.append(t_64)
    if not cols:
        return None
    return torch.cat(cols, dim=1)


def _build_s5_orthogonal_router_stage(
    editable_layers: Sequence[int],
    gated_targets: Sequence[int],
    sink_experts: Sequence[int],
    prose_by_layer: Mapping[int, torch.Tensor],
    spec_by_layer: Mapping[int, Mapping[str, torch.Tensor] | torch.Tensor],
    cfg: PostRoundWSPHookConfig,
) -> WSPStageSpec:
    """Construct Stage 3: 2-fold OOF SNR-gated orthogonal multi-gate ``sigma_topk`` router surgery."""
    layer_set = set(editable_layers)
    gated_tuple = tuple(int(e) for e in gated_targets)
    sink_tuple = tuple(int(e) for e in sink_experts)

    def _transform(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = clone_state_dict(sd)
        for key in sorted(out.keys()):
            match = _ROUTER_KEY_RE.match(key)
            if match is None:
                continue
            l_idx = int(match.group(1))
            if l_idx not in layer_set or l_idx not in prose_by_layer or l_idx not in spec_by_layer:
                continue
            router_w = out[key]
            if router_w.ndim != 2:
                continue
            x_prose = prose_by_layer[l_idx]
            x_spec = spec_by_layer[l_idx]
            dim = int(router_w.shape[1])
            if int(x_prose.shape[1]) != dim:
                raise ValueError(
                    f"layer_prose_inputs[{l_idx}] width {x_prose.shape[1]} != router width {dim}"
                )
            updated_w = router_w.detach().clone()
            num_rows = int(updated_w.shape[0])
            rel_weights: list[float] = []
            for e_idx in gated_tuple:
                if e_idx >= num_rows:
                    continue
                new_row, stats = fit_oof_orthogonal_multigate_row_update(
                    updated_w[e_idx],
                    x_prose,
                    x_spec,
                    router_beta=cfg.router_beta,
                    min_oof_snr=cfg.min_oof_snr,
                    n_folds=cfg.n_folds,
                    omega_spec=cfg.omega_spec,
                    ridge_scale=cfg.ridge_scale,
                    sv_rel_tol=cfg.sv_rel_tol,
                    inplace=False,
                )
                updated_w[e_idx] = new_row
                rel_weights.append(float(stats["rel_weight"]))

            if sink_tuple and cfg.sink_attract_scale > 0.0 and cfg.router_beta > 0.0:
                sink_anchor = sink_tuple[0]
                if sink_anchor < num_rows and (not rel_weights or max(rel_weights) > 0.0):
                    shifted, sink_stats = fit_oof_orthogonal_multigate_row_update(
                        torch.zeros_like(updated_w[sink_anchor]),
                        x_prose,
                        x_spec,
                        router_beta=cfg.sink_attract_scale * cfg.router_beta,
                        min_oof_snr=cfg.min_oof_snr,
                        n_folds=cfg.n_folds,
                        omega_spec=cfg.omega_spec,
                        ridge_scale=cfg.ridge_scale,
                        sv_rel_tol=cfg.sv_rel_tol,
                        inplace=False,
                    )
                    if bool(sink_stats["applied"]):
                        updated_w[sink_anchor] = updated_w[sink_anchor] - shifted
            out[key] = updated_w
        return out

    return WSPStageSpec(
        name="s5_stage_3_orthogonal_multigate_router",
        transform_fn=_transform,
        stage_kind="oof_orthogonal_multigate_router",
        description="2-fold OOF SNR-gated orthogonal multi-gate sigma_topk router prose suppression.",
    )


def _resolve_final_norm_keys(
    state_dict: Mapping[str, torch.Tensor],
    cfg: PostRoundWSPHookConfig,
) -> tuple[str | None, str | None]:
    """Locate final norm weight and optional bias keys in ``state_dict``."""
    w_key: str | None = None
    if cfg.ln_weight_key in state_dict:
        w_key = cfg.ln_weight_key
    elif cfg.ln_weight_key == "ln.weight":
        for cand in _LN_WEIGHT_FALLBACK_KEYS:
            if cand in state_dict:
                w_key = cand
                break

    b_key: str | None = None
    if cfg.ln_bias_key in state_dict:
        b_key = cfg.ln_bias_key
    elif cfg.ln_bias_key == "ln.bias":
        for cand in _LN_BIAS_FALLBACK_KEYS:
            if cand in state_dict:
                b_key = cand
                break
    return w_key, b_key


def _search_folded_ln_temp_stage(
    current_state: Mapping[str, torch.Tensor],
    base_domain_nll: Mapping[str, float],
    eval_fn: Callable[[Mapping[str, torch.Tensor]], Any],
    base_final_ln_w: torch.Tensor | None,
    base_final_ln_b: torch.Tensor | None,
    cfg: PostRoundWSPHookConfig,
) -> tuple[WSPStageSpec | None, float | None, float | None]:
    """Coordinate-search ``(lambda_ln, tau_temp)`` and fold ``1 / tau_temp`` into final norm weights."""
    w_key, b_key = _resolve_final_norm_keys(current_state, cfg)
    if w_key is None:
        return None, None, None

    cur_w = current_state[w_key]
    if cur_w.ndim != 1 or not cur_w.is_floating_point():
        return None, None, None
    h_dim = int(cur_w.shape[0])
    cur_b = current_state[b_key] if b_key is not None else None
    if cur_b is not None and (cur_b.shape != (h_dim,) or not cur_b.is_floating_point()):
        return None, None, None

    target_w_64: torch.Tensor | None = None
    target_b_64: torch.Tensor | None = None
    if base_final_ln_w is not None:
        if not isinstance(base_final_ln_w, torch.Tensor) or base_final_ln_w.shape != (h_dim,) or not base_final_ln_w.is_floating_point():
            raise ValueError(f"base_final_ln_w must be a 1D floating-point tensor of shape ({h_dim},)")
        target_w_64 = base_final_ln_w.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(target_w_64).all().item()):
            raise ValueError("base_final_ln_w must contain only finite values")
        if base_final_ln_b is not None:
            if b_key is None:
                raise ValueError("base_final_ln_b provided but state_dict has no biasless/norm bias key")
            if not isinstance(base_final_ln_b, torch.Tensor) or base_final_ln_b.shape != (h_dim,) or not base_final_ln_b.is_floating_point():
                raise ValueError(f"base_final_ln_b must be a 1D floating-point tensor of shape ({h_dim},)")
            target_b_64 = base_final_ln_b.detach().to(device="cpu", dtype=torch.float64)
            if not bool(torch.isfinite(target_b_64).all().item()):
                raise ValueError("base_final_ln_b must contain only finite values")
    elif base_final_ln_b is not None:
        raise ValueError("base_final_ln_b cannot be provided when base_final_ln_w is None")

    cur_w64 = cur_w.detach().to(device="cpu", dtype=torch.float64)
    cur_b64 = cur_b.detach().to(device="cpu", dtype=torch.float64) if cur_b is not None else None

    if target_w_64 is not None:
        sorted_grid = sorted({float(x) for x in cfg.lambda_ln_grid} | {0.0})
        if len(sorted_grid) > 7:
            lam_candidates = sorted({sorted_grid[0], sorted_grid[len(sorted_grid) // 4], sorted_grid[len(sorted_grid) // 2], sorted_grid[(3 * len(sorted_grid)) // 4], sorted_grid[-1], 0.0})
        else:
            lam_candidates = sorted_grid
    else:
        lam_candidates = [0.0]

    def _build_norm_pair(lam: float, tau: float) -> tuple[torch.Tensor, torch.Tensor | None]:
        w_l = cur_w64 if target_w_64 is None else (cur_w64 + lam * (target_w_64 - cur_w64))
        b_l = cur_b64 if (cur_b64 is None or target_b_64 is None) else (cur_b64 + lam * (target_b_64 - cur_b64))
        w_cast = w_l.to(device=cur_w.device, dtype=cur_w.dtype)
        b_cast = b_l.to(device=cur_b.device, dtype=cur_b.dtype) if (b_l is not None and cur_b is not None) else None
        return fold_temperature_into_norm_params(w_cast, b_cast, tau)

    evaluated: dict[tuple[float, float], tuple[bool, float, float, float]] = {}

    def _eval_pair(lam: float, tau: float) -> tuple[bool, float, float, float]:
        key_pt = (round(lam, 6), round(tau, 6))
        if key_pt in evaluated:
            return evaluated[key_pt]
        w_cand, b_cand = _build_norm_pair(lam, tau)
        cand_sd = clone_state_dict(current_state)
        cand_sd[w_key] = w_cand
        if b_key is not None and b_cand is not None:
            cand_sd[b_key] = b_cand
        metrics, _, finite = _evaluate_state_metrics(
            eval_fn, cand_sd, base_domain_nll, cfg, allow_runtime_eval_error=True
        )
        shaping = _compute_shaping_energy(
            metrics, guard_rel=cfg.general_guard_rel, maximum_pain=cfg.maximum_pain, finite=finite
        )
        res = (bool(finite and metrics["guard_ok"]), shaping, float(metrics["anse_energy"]), float(metrics["nll_mean"]))
        evaluated[key_pt] = res
        return res

    # Phase 1: sweep coarse lambda_ln at tau=1.0
    best_lam = 0.0
    best_lam_key: tuple[int, float, float] | None = None
    for lam in lam_candidates:
        g_ok, shaping, _, nll_m = _eval_pair(lam, 1.0)
        rank_key = (0 if g_ok else 1, shaping, nll_m)
        if best_lam_key is None or rank_key < best_lam_key:
            best_lam_key = rank_key
            best_lam = lam

    # Optional fine local check around best_lam in cfg.lambda_ln_grid
    if target_w_64 is not None and len(cfg.lambda_ln_grid) > 7:
        full_grid = sorted({float(x) for x in cfg.lambda_ln_grid})
        idx_best = min(range(len(full_grid)), key=lambda i: abs(full_grid[i] - best_lam))
        neighbors = [full_grid[i] for i in range(max(0, idx_best - 1), min(len(full_grid), idx_best + 2))]
        for lam in neighbors:
            g_ok, shaping, _, nll_m = _eval_pair(lam, 1.0)
            rank_key = (0 if g_ok else 1, shaping, nll_m)
            if best_lam_key is None or rank_key < best_lam_key:
                best_lam_key = rank_key
                best_lam = lam

    # Phase 2: sweep folded scalar temperatures at best_lam (and 0.0 if distinct)
    best_pair = (best_lam, 1.0)
    best_pair_key: tuple[int, float, float, float] | None = None
    lam_sweep = [best_lam] if abs(best_lam) < 1e-9 else [best_lam, 0.0]
    temp_candidates = sorted({float(t) for t in cfg.temperature_grid} | {1.0})
    for lam in lam_sweep:
        for tau in temp_candidates:
            g_ok, shaping, _, nll_m = _eval_pair(lam, tau)
            rank_key = (0 if g_ok else 1, shaping, nll_m, abs(tau - 1.0))
            if best_pair_key is None or rank_key < best_pair_key:
                best_pair_key = rank_key
                best_pair = (lam, tau)

    chosen_lam, chosen_tau = best_pair
    if abs(chosen_lam) <= 1e-12 and abs(chosen_tau - 1.0) <= 1e-12:
        return None, 0.0, 1.0

    def _transform(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = clone_state_dict(sd)
        w_f, b_f = _build_norm_pair(chosen_lam, chosen_tau)
        out[w_key] = w_f
        if b_key is not None and b_f is not None:
            out[b_key] = b_f
        return out

    spec = WSPStageSpec(
        name="s5_stage_4_folded_ln_temperature",
        transform_fn=_transform,
        stage_kind="folded_layernorm_temperature",
        description="Guard-constrained final norm interpolation and zero-overhead 1/tau temperature fold.",
    )
    return spec, chosen_lam, chosen_tau


def _execute_progressive_stage(
    *,
    stage_spec: WSPStageSpec,
    raw_state: Mapping[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    current_sha: str,
    current_shaping_energy: float,
    current_metrics: dict[str, Any],
    current_nll: dict[str, float],
    base_domain_nll: Mapping[str, float],
    eval_fn: Callable[[Mapping[str, torch.Tensor]], Any],
    anchor_layers: Sequence[int],
    anchor_sha_ref: str,
    total_params: int,
    cfg: PostRoundWSPHookConfig,
    committed: list[str],
    rolled_back: list[str],
    skipped: list[str],
) -> tuple[dict[str, torch.Tensor], str, float, dict[str, Any], dict[str, float], bool]:
    """Run a single S5 stage under finiteness, anchor preservation, edit budget, and ANSE ratchet."""
    cand_raw = stage_spec.transform_fn(clone_state_dict(current_state))
    cand_state, cand_finite = _validate_state_mapping(cand_raw, stage_spec.name, require_finite=False)
    _validate_matching_state_shapes(current_state, cand_state, "current_state", stage_spec.name)
    cand_sha = compute_state_dict_sha256(cand_state)

    if cand_sha == current_sha:
        skipped.append(stage_spec.name)
        return current_state, current_sha, current_shaping_energy, current_metrics, current_nll, False

    anchor_ok = (
        not (cfg.protect_anchor_layers and anchor_layers)
        or _anchor_layers_sha256(cand_state, anchor_layers) == anchor_sha_ref
    )
    edited_vs_raw, _ = count_edited_parameters(raw_state, cand_state)
    budget_ok = (float(edited_vs_raw) / float(total_params)) <= cfg.max_total_edit_fraction + 1e-12

    if not cand_finite or not anchor_ok or not budget_ok:
        rolled_back.append(stage_spec.name)
        return current_state, current_sha, current_shaping_energy, current_metrics, current_nll, False

    cand_metrics, cand_nll, eval_finite = _evaluate_state_metrics(
        eval_fn, cand_state, base_domain_nll, cfg, allow_runtime_eval_error=True
    )
    cand_shaping = _compute_shaping_energy(
        cand_metrics, guard_rel=cfg.general_guard_rel, maximum_pain=cfg.maximum_pain, finite=eval_finite
    )
    if eval_finite and cand_shaping < current_shaping_energy - cfg.min_energy_improvement:
        committed.append(stage_spec.name)
        return cand_state, cand_sha, cand_shaping, cand_metrics, cand_nll, True

    rolled_back.append(stage_spec.name)
    return current_state, current_sha, current_shaping_energy, current_metrics, current_nll, False


# ---------------------------------------------------------------------------
# 4. HM-S35-04: Automatic Post-Outer-Round S5 Surgery Hook
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_post_outer_round_auto_wsp_hook(
    raw_outer_state: Mapping[str, torch.Tensor],
    base_domain_nll: Mapping[str, float],
    eval_fn: Callable[[Mapping[str, torch.Tensor]], Any],
    *,
    parent_state: Mapping[str, torch.Tensor] | None = None,
    layer_e0_harm_nats: Sequence[float] | Sequence[Mapping[str, Any]] | Mapping[Any, Any],
    expert_e0_harm_nats: Mapping[Any, float] | Sequence[float],
    expert_spec_loss_when_zeroed: Mapping[Any, float] | Sequence[float],
    layer_prose_inputs: Mapping[int, torch.Tensor] | Mapping[str, torch.Tensor] | Sequence[torch.Tensor] | None = None,
    layer_specialist_inputs: Mapping[int, Any] | Mapping[str, Any] | Sequence[Any] | None = None,
    base_final_ln_w: torch.Tensor | None = None,
    base_final_ln_b: torch.Tensor | None = None,
    round_id: str = "round_0",
    config: PostRoundWSPHookConfig | None = None,
) -> tuple[dict[str, torch.Tensor], PostRoundWSPLineageReceipt]:
    """Automatically diagnose and apply CPU-only S5 PASS-Synergy surgery after a nocturnal outer round.

    Promotes the sanitized S5 checkpoint to ``promoted_shadow`` iff ``s5_anse_energy < raw_anse_energy``
    and ``general_rel_drift <= general_guard_rel``. Cleanly keeps ``raw_outer_state`` bitwise when
    surgery yields zero gain (``rejected_fallback_raw``), or rolls back bitwise to ``parent_state``
    with an ANSE Maximum Pain receipt on corrupted/infeasible rounds (``rejected_fallback_parent``).
    """
    cfg = PostRoundWSPHookConfig() if config is None else config
    if not isinstance(cfg, PostRoundWSPHookConfig):
        raise TypeError(f"config must be a PostRoundWSPHookConfig or None, got {type(cfg).__name__}")
    round_name = _validate_nonempty_str(round_id, "round_id")
    if not callable(eval_fn):
        raise TypeError("eval_fn must be callable")

    base_nll, _ = parse_eval_domain_nll(base_domain_nll, allow_non_finite=False)
    base_metrics = evaluate_anse_domain_metrics(
        base_nll,
        base_nll,
        pre_domain_nll=base_nll,
        general_guard_rel=cfg.general_guard_rel,
        general_prefix=cfg.general_prefix,
        soft_margin_penalty=cfg.soft_margin_penalty,
        maximum_pain=cfg.maximum_pain,
        cand_finite=True,
    )
    base_nll_mean = float(base_metrics["nll_mean"])

    raw_state, raw_weights_finite = _validate_state_mapping(
        raw_outer_state, "raw_outer_state", require_finite=False
    )
    raw_sha = compute_state_dict_sha256(raw_state)
    _, total_params = count_edited_parameters(raw_state, raw_state)

    has_explicit_parent = parent_state is not None
    clean_parent: dict[str, torch.Tensor] | None = None
    if parent_state is not None:
        clean_parent, _ = _validate_state_mapping(parent_state, "parent_state", require_finite=True)
        _validate_matching_state_shapes(raw_state, clean_parent, "raw_outer_state", "parent_state")
        parent_sha = compute_state_dict_sha256(clean_parent)
    else:
        if not raw_weights_finite:
            raise ValueError(
                "raw_outer_state contains non-finite values and no parent_state was provided for rollback"
            )
        parent_sha = raw_sha

    if (layer_prose_inputs is None) != (layer_specialist_inputs is None):
        raise ValueError("layer_prose_inputs and layer_specialist_inputs must either both be provided or both be None")
    prose_by_layer = _normalize_layer_prose_inputs(layer_prose_inputs) if layer_prose_inputs is not None else {}
    spec_by_layer = (
        _normalize_layer_specialist_inputs(layer_specialist_inputs) if layer_specialist_inputs is not None else {}
    )

    layer_part = identify_beneficial_anchor_and_interfering_layers(
        layer_e0_harm_nats,
        anchor_threshold_nats=cfg.anchor_threshold_nats,
        harm_threshold_nats=cfg.harm_threshold_nats,
    )
    expert_part = partition_sink_vs_preserved_specialists(
        expert_e0_harm_nats,
        expert_spec_loss_when_zeroed,
        spec_preserve_floor_nats=cfg.spec_preserve_floor_nats,
        prose_harm_floor_nats=cfg.prose_harm_floor_nats,
        rogue_expert_indices=cfg.rogue_expert_indices or None,
        general_expert_index=cfg.general_expert_index,
    )

    anchor_layers = tuple(int(l_idx) for l_idx in layer_part["beneficial_anchor_layers"])
    interfering_layers = tuple(int(l_idx) for l_idx in layer_part["interfering_layers"])
    downstream_interfering = tuple(int(l_idx) for l_idx in layer_part["downstream_interfering_layers"])
    anchor_set = set(anchor_layers) if cfg.protect_anchor_layers else set()

    sink_experts = tuple(int(e) for e in expert_part["sink_experts"])
    preserved_specialists = tuple(int(e) for e in expert_part["preserved_specialists"])
    if cfg.gate_all_surviving_specialists:
        helpful_set = set(int(e) for e in expert_part["helpful_general_experts"])
        gated_targets = tuple(
            int(e)
            for e in expert_part["all_preserved_experts"]
            if int(e) != cfg.general_expert_index and int(e) not in helpful_set
        )
    else:
        gated_targets = preserved_specialists

    if interfering_layers:
        editable_layers = tuple(sorted(set(interfering_layers) - anchor_set))
    elif sink_experts or gated_targets:
        harm_map = layer_part["layer_e0_harm_nats"]
        editable_layers = tuple(
            sorted(int(k) for k, v in harm_map.items() if int(k) not in anchor_set and float(v) > 0.0)
        )
    else:
        editable_layers = ()

    # Evaluate parent (if provided) and raw outer checkpoint
    if clean_parent is not None:
        parent_metrics, _, parent_finite = _evaluate_state_metrics(
            eval_fn, clean_parent, base_nll, cfg, allow_runtime_eval_error=True
        )
    else:
        parent_metrics, parent_finite = {}, True

    raw_metrics, raw_nll, raw_eval_finite = _evaluate_state_metrics(
        eval_fn,
        raw_state,
        base_nll,
        cfg,
        allow_runtime_eval_error=(clean_parent is not None),
    )
    raw_ok_finite = bool(raw_weights_finite and raw_eval_finite)
    raw_guard_ok = bool(raw_ok_finite and raw_metrics["guard_ok"])
    raw_energy = float(raw_metrics["anse_energy"]) if raw_ok_finite else float(cfg.maximum_pain)
    raw_nll_mean = float(raw_metrics["nll_mean"]) if raw_ok_finite else float(cfg.maximum_pain)
    raw_gen_drift = float(raw_metrics["general_domain_drift"]) if raw_ok_finite else float(cfg.maximum_pain)

    if clean_parent is not None:
        parent_energy = float(parent_metrics["anse_energy"]) if parent_finite else float(cfg.maximum_pain)
        parent_gen_drift = float(parent_metrics["general_domain_drift"]) if parent_finite else float(cfg.maximum_pain)
    else:
        parent_energy = raw_energy
        parent_gen_drift = raw_gen_drift

    # Fast-path fail-closed rollback when raw_outer_state itself is non-finite or blew up eval
    if not raw_ok_finite:
        assert clean_parent is not None
        active_state = clone_state_dict(clean_parent)
        anchor_preserved = _anchor_layers_sha256(active_state, anchor_layers) == _anchor_layers_sha256(
            clean_parent, anchor_layers
        )
        parent_nll_map = (
            {d: float(parent_metrics["per_domain"][d]["cand_nll"]) for d in base_nll}
            if parent_finite
            else {d: float(cfg.maximum_pain) for d in base_nll}
        )
        parent_nll_mean = float(parent_metrics["nll_mean"]) if parent_finite else float(cfg.maximum_pain)
        receipt = _seal_lineage_receipt(
            {
                "round_id": round_name,
                "parent_checkpoint_sha256": parent_sha,
                "raw_outer_sha256": raw_sha,
                "sanitized_s5_sha256": raw_sha,
                "raw_anse_energy": float(cfg.maximum_pain),
                "s5_anse_energy": float(cfg.maximum_pain),
                "general_rel_drift": float(cfg.maximum_pain),
                "promotion_stage": PROMOTION_STAGE_REJECTED_PARENT,
                "active_checkpoint_sha256": parent_sha,
                "has_explicit_parent": has_explicit_parent,
                "parent_anse_energy": parent_energy,
                "active_anse_energy": parent_energy,
                "anse_energy": parent_energy,
                "delta_anse_energy": 0.0,
                "base_nll_mean": base_nll_mean,
                "raw_nll_mean": float(cfg.maximum_pain),
                "s5_nll_mean": float(cfg.maximum_pain),
                "nll_mean": parent_nll_mean,
                "raw_general_rel_drift": float(cfg.maximum_pain),
                "s5_general_rel_drift": float(cfg.maximum_pain),
                "parent_general_rel_drift": parent_gen_drift,
                "active_general_rel_drift": parent_gen_drift,
                "worst_domain_rel_drift": float(cfg.maximum_pain),
                "base_domain_nll": base_nll,
                "raw_domain_nll": raw_nll,
                "s5_domain_nll": raw_nll,
                "active_domain_nll": parent_nll_map,
                "raw_guard_ok": False,
                "s5_guard_ok": False,
                "guard_ok": bool(parent_finite and parent_metrics["guard_ok"]),
                "wsp_verdict": "FAIL",
                "verdict": "FAIL",
                "reason": "Corrupted/non-finite outer round weights or eval; rolled back bitwise to parent_state with ANSE Maximum Pain.",
                "beneficial_anchor_layers": anchor_layers,
                "interfering_layers": interfering_layers,
                "editable_layers": editable_layers,
                "anchor_layers_bitwise_preserved": anchor_preserved,
                "sink_experts": sink_experts,
                "preserved_specialists": preserved_specialists,
                "gated_targets": gated_targets,
                "committed_stages": (),
                "rolled_back_stages": ("s5_preflight_finiteness",),
                "skipped_stages": (),
                "lambda_ln_applied": None,
                "temperature_applied": None,
                "total_params": total_params,
                "edited_params_vs_raw": 0,
                "edit_fraction_vs_raw": 0.0,
                "base_model_id": cfg.base_model_id,
            }
        )
        return active_state, receipt

    # Build S5 PASS-Synergy surgical stages
    anchor_sha_ref = _anchor_layers_sha256(raw_state, anchor_layers)
    stages_1_to_3: list[WSPStageSpec] = []
    if sink_experts and editable_layers:
        stages_1_to_3.append(
            make_ghost_sink_stage(
                sink_experts,
                layer_indices=editable_layers,
                name="s5_stage_1_selective_ghost_sink",
                sanitize_non_finite=cfg.sanitize_non_finite_ghost_sinks,
            )
        )

    if cfg.enable_anchor_nullspace_projection and anchor_layers and downstream_interfering:
        anchor_experts = tuple(sorted(set(expert_part["helpful_general_experts"]) | {cfg.general_expert_index}))
        anchor_subspace = _build_anchor_prose_subspace(raw_state, anchor_layers, anchor_experts)
        proj_targets = tuple(sorted(set(downstream_interfering) - anchor_set))
        non_sink_experts = tuple(sorted(set(expert_part["all_preserved_experts"]) - {cfg.general_expert_index}))
        if anchor_subspace is not None and proj_targets and non_sink_experts:
            stages_1_to_3.append(
                make_cross_layer_anchor_stage(
                    anchor_subspace,
                    target_layers=proj_targets,
                    expert_indices=non_sink_experts,
                    factors=("w3_B",),
                    strength=cfg.anchor_proj_strength,
                    orthonormalize=True,
                    name="s5_stage_2_anchor_nullspace",
                )
            )

    if prose_by_layer and spec_by_layer and editable_layers and gated_targets:
        stages_1_to_3.append(
            _build_s5_orthogonal_router_stage(
                editable_layers,
                gated_targets,
                sink_experts,
                prose_by_layer,
                spec_by_layer,
                cfg,
            )
        )

    current_state = clone_state_dict(raw_state)
    current_sha = raw_sha
    current_metrics = dict(raw_metrics)
    current_nll = dict(raw_nll)
    current_shaping = _compute_shaping_energy(
        raw_metrics, guard_rel=cfg.general_guard_rel, maximum_pain=cfg.maximum_pain, finite=True
    )
    committed: list[str] = []
    rolled_back: list[str] = []
    skipped: list[str] = []

    for stage in stages_1_to_3:
        (
            current_state,
            current_sha,
            current_shaping,
            current_metrics,
            current_nll,
            _,
        ) = _execute_progressive_stage(
            stage_spec=stage,
            raw_state=raw_state,
            current_state=current_state,
            current_sha=current_sha,
            current_shaping_energy=current_shaping,
            current_metrics=current_metrics,
            current_nll=current_nll,
            base_domain_nll=base_nll,
            eval_fn=eval_fn,
            anchor_layers=anchor_layers,
            anchor_sha_ref=anchor_sha_ref,
            total_params=total_params,
            cfg=cfg,
            committed=committed,
            rolled_back=rolled_back,
            skipped=skipped,
        )

    lambda_ln_applied: float | None = None
    temp_applied: float | None = None
    if cfg.use_ln_calibration:
        stage4_spec, lambda_ln_applied, temp_applied = _search_folded_ln_temp_stage(
            current_state,
            base_nll,
            eval_fn,
            base_final_ln_w,
            base_final_ln_b,
            cfg,
        )
        if stage4_spec is not None:
            (
                current_state,
                current_sha,
                current_shaping,
                current_metrics,
                current_nll,
                stage4_committed,
            ) = _execute_progressive_stage(
                stage_spec=stage4_spec,
                raw_state=raw_state,
                current_state=current_state,
                current_sha=current_sha,
                current_shaping_energy=current_shaping,
                current_metrics=current_metrics,
                current_nll=current_nll,
                base_domain_nll=base_nll,
                eval_fn=eval_fn,
                anchor_layers=anchor_layers,
                anchor_sha_ref=anchor_sha_ref,
                total_params=total_params,
                cfg=cfg,
                committed=committed,
                rolled_back=rolled_back,
                skipped=skipped,
            )
            if not stage4_committed:
                lambda_ln_applied = 0.0
                temp_applied = 1.0

    # Strict final ANSE verification & shadow / fallback ratchet
    s5_sha = current_sha
    s5_energy = float(current_metrics["anse_energy"])
    s5_nll_mean = float(current_metrics["nll_mean"])
    s5_gen_drift = float(current_metrics["general_domain_drift"])
    s5_worst_drift = float(current_metrics["worst_domain_drift"])
    s5_guard_ok = bool(current_metrics["guard_ok"])

    improves_energy = s5_energy < (raw_energy - cfg.min_energy_improvement)
    clears_guard = s5_guard_ok and (s5_gen_drift <= cfg.general_guard_rel + 1e-12)

    if improves_energy and clears_guard:
        promotion_stage = PROMOTION_STAGE_PROMOTED_SHADOW
        wsp_verdict = "PASS"
        active_state = current_state
        active_sha = s5_sha
        active_energy = s5_energy
        active_nll_mean = s5_nll_mean
        active_gen_drift = s5_gen_drift
        active_nll = dict(current_nll)
        active_guard_ok = True
        reason = (
            f"S5 PASS-Synergy surgery lowered ANSE energy {raw_energy:+.6f} -> {s5_energy:+.6f} "
            f"with E0 drift {100.0 * s5_gen_drift:+.3f}% <= {100.0 * cfg.general_guard_rel:+.3f}%; promoted to shadow."
        )
    elif raw_guard_ok:
        promotion_stage = PROMOTION_STAGE_REJECTED_RAW
        wsp_verdict = "NO_GAIN"
        active_state = clone_state_dict(raw_state)
        active_sha = raw_sha
        active_energy = raw_energy
        active_nll_mean = raw_nll_mean
        active_gen_drift = raw_gen_drift
        active_nll = dict(raw_nll)
        active_guard_ok = True
        reason = (
            "Outer checkpoint already passed ANSE guard and S5 surgery yielded no energy improvement; "
            "kept raw_outer_state bitwise."
        )
    elif clean_parent is not None:
        promotion_stage = PROMOTION_STAGE_REJECTED_PARENT
        wsp_verdict = "FAIL"
        active_state = clone_state_dict(clean_parent)
        active_sha = parent_sha
        active_energy = parent_energy
        active_nll_mean = (
            float(parent_metrics["nll_mean"]) if parent_finite else float(cfg.maximum_pain)
        )
        active_gen_drift = parent_gen_drift
        active_nll = (
            {d: float(parent_metrics["per_domain"][d]["cand_nll"]) for d in base_nll}
            if parent_finite
            else {d: float(cfg.maximum_pain) for d in base_nll}
        )
        active_guard_ok = bool(parent_finite and parent_metrics["guard_ok"])
        reason = (
            f"Infeasible outer round (raw E={raw_energy:.4f}, S5 E={s5_energy:.4f}); "
            "rolled back bitwise to parent_state with ANSE Maximum Pain receipt."
        )
    else:
        promotion_stage = PROMOTION_STAGE_REJECTED_RAW
        wsp_verdict = "FAIL"
        active_state = clone_state_dict(raw_state)
        active_sha = raw_sha
        active_energy = raw_energy
        active_nll_mean = raw_nll_mean
        active_gen_drift = raw_gen_drift
        active_nll = dict(raw_nll)
        active_guard_ok = False
        reason = (
            f"Infeasible outer round without parent_state (raw E={raw_energy:.4f}, S5 E={s5_energy:.4f}); "
            "rejected shadow promotion."
        )

    edited_params_vs_raw, _ = count_edited_parameters(raw_state, active_state)
    edit_fraction_vs_raw = float(edited_params_vs_raw) / float(total_params)
    anchor_preserved = (
        _anchor_layers_sha256(active_state, anchor_layers) == anchor_sha_ref
        if (cfg.protect_anchor_layers and promotion_stage != PROMOTION_STAGE_REJECTED_PARENT)
        else True
    )

    receipt = _seal_lineage_receipt(
        {
            "round_id": round_name,
            "parent_checkpoint_sha256": parent_sha,
            "raw_outer_sha256": raw_sha,
            "sanitized_s5_sha256": s5_sha,
            "raw_anse_energy": raw_energy,
            "s5_anse_energy": s5_energy,
            "general_rel_drift": s5_gen_drift,
            "promotion_stage": promotion_stage,
            "active_checkpoint_sha256": active_sha,
            "has_explicit_parent": has_explicit_parent,
            "parent_anse_energy": parent_energy,
            "active_anse_energy": active_energy,
            "anse_energy": active_energy,
            "delta_anse_energy": s5_energy - raw_energy,
            "base_nll_mean": base_nll_mean,
            "raw_nll_mean": raw_nll_mean,
            "s5_nll_mean": s5_nll_mean,
            "nll_mean": active_nll_mean,
            "raw_general_rel_drift": raw_gen_drift,
            "s5_general_rel_drift": s5_gen_drift,
            "parent_general_rel_drift": parent_gen_drift,
            "active_general_rel_drift": active_gen_drift,
            "worst_domain_rel_drift": s5_worst_drift,
            "base_domain_nll": base_nll,
            "raw_domain_nll": raw_nll,
            "s5_domain_nll": current_nll,
            "active_domain_nll": active_nll,
            "raw_guard_ok": raw_guard_ok,
            "s5_guard_ok": s5_guard_ok,
            "guard_ok": active_guard_ok,
            "wsp_verdict": wsp_verdict,
            "verdict": wsp_verdict,
            "reason": reason,
            "beneficial_anchor_layers": anchor_layers,
            "interfering_layers": interfering_layers,
            "editable_layers": editable_layers,
            "anchor_layers_bitwise_preserved": anchor_preserved,
            "sink_experts": sink_experts,
            "preserved_specialists": preserved_specialists,
            "gated_targets": gated_targets,
            "committed_stages": tuple(committed),
            "rolled_back_stages": tuple(rolled_back),
            "skipped_stages": tuple(skipped),
            "lambda_ln_applied": lambda_ln_applied,
            "temperature_applied": temp_applied,
            "total_params": total_params,
            "edited_params_vs_raw": edited_params_vs_raw,
            "edit_fraction_vs_raw": edit_fraction_vs_raw,
            "base_model_id": cfg.base_model_id,
        }
    )
    return active_state, receipt


# ---------------------------------------------------------------------------
# 5. HM-S35-05: Shadow-to-Live Canary Gate & ExpertRegistry Promotion
# ---------------------------------------------------------------------------


def _coerce_lineage_receipt(
    lineage_receipt: PostRoundWSPLineageReceipt | Mapping[str, Any],
) -> tuple[PostRoundWSPLineageReceipt, bool]:
    """Normalize a ``PostRoundWSPLineageReceipt`` or serialized mapping and verify provenance."""
    if isinstance(lineage_receipt, PostRoundWSPLineageReceipt):
        prov_ok = True if lineage_receipt.receipt_sha256 == "" else verify_lineage_receipt_provenance(lineage_receipt)
        return lineage_receipt, prov_ok
    if isinstance(lineage_receipt, Mapping):
        prov_ok = True if not lineage_receipt.get("receipt_sha256") else verify_lineage_receipt_provenance(lineage_receipt)
        tuple_fields = {
            "beneficial_anchor_layers",
            "interfering_layers",
            "editable_layers",
            "sink_experts",
            "preserved_specialists",
            "gated_targets",
            "committed_stages",
            "rolled_back_stages",
            "skipped_stages",
        }
        kwargs: dict[str, Any] = {}
        for key, val in lineage_receipt.items():
            if key in tuple_fields and isinstance(val, list):
                kwargs[key] = tuple(val)
            else:
                kwargs[key] = val
        return PostRoundWSPLineageReceipt(**kwargs), prov_ok
    raise TypeError(
        f"lineage_receipt must be a PostRoundWSPLineageReceipt or Mapping, got {type(lineage_receipt).__name__}"
    )


def build_expert_registry_gate_payload(
    lineage_receipt: PostRoundWSPLineageReceipt,
    *,
    approved: bool,
    probe_pass_count: int,
    probe_total: int,
    min_pass_ratio: float,
    shadow_general_drift_rel: float,
    max_general_drift_rel: float,
    base_model_id: str | None = None,
) -> dict[str, Any]:
    """Build a gate payload compatible with ``ExpertRegistry.promote_with_gate``."""
    resolved_base = (
        _validate_nonempty_str(base_model_id, "base_model_id")
        if base_model_id is not None
        else lineage_receipt.base_model_id
    )
    gain_nats = round(float(lineage_receipt.raw_nll_mean - lineage_receipt.s5_nll_mean), 6)
    diff_nats = round(float(lineage_receipt.s5_nll_mean - lineage_receipt.raw_nll_mean), 6)
    worst_reg = round(max(float(lineage_receipt.worst_domain_rel_drift), float(shadow_general_drift_rel)), 6)
    return {
        "object": CANARY_GATE_OBJECT,
        "schema": SCHEMA_VERSION,
        "servable": bool(approved),
        "base": resolved_base,
        "adapter": {
            "lora_gguf_sha256": lineage_receipt.sanitized_s5_sha256,
        },
        "pooled": {
            "mean_nll_gain_nats": gain_nats,
            "bootstrap95": [diff_nats, 0.0],
        },
        "worst_domain_regression": worst_reg,
        "canary": {
            "round_id": lineage_receipt.round_id,
            "probe_pass_count": int(probe_pass_count),
            "probe_total": int(probe_total),
            "probe_pass_ratio": float(probe_pass_count) / float(probe_total),
            "min_pass_ratio": float(min_pass_ratio),
            "shadow_general_drift_rel": float(shadow_general_drift_rel),
            "max_general_drift_rel": float(max_general_drift_rel),
            "wsp_verdict": lineage_receipt.wsp_verdict,
        },
    }


def evaluate_shadow_to_live_canary_gate(
    lineage_receipt: PostRoundWSPLineageReceipt | Mapping[str, Any],
    probe_pass_count: int,
    probe_total: int,
    *,
    min_pass_ratio: float = 1.0,
    shadow_general_drift_rel: float = 0.0,
    max_general_drift_rel: float = 0.01,
    min_probe_total: int = 1,
    registry: ExpertRegistry | None = None,
    expert_id: str | None = None,
    cluster_id: str = "moe_swarm",
    base_model_id: str | None = None,
    adapter_size_mib: int = 48,
    keywords: Sequence[str] = (),
) -> PostRoundWSPLineageReceipt:
    """Gate ``candidate -> shadow -> live`` promotion on router canary probes and PASS WSP lineage.

    Approves promotion to ``promoted_live`` if and only if:
    - ``lineage_receipt`` has valid provenance and ``promotion_stage == 'promoted_shadow'``,
    - ``wsp_verdict == 'PASS'``, ``s5_guard_ok is True``, and ``s5_anse_energy < raw_anse_energy``,
    - ``probe_total >= min_probe_total`` and ``probe_pass_count / probe_total >= min_pass_ratio``,
    - ``lineage_receipt.general_rel_drift <= max_general_drift_rel`` and
      ``shadow_general_drift_rel <= max_general_drift_rel``.

    When ``registry`` (an :class:`ExpertRegistry`) and ``expert_id`` are supplied, the
    registry lifecycle is updated strictly via ``register -> transition(SHADOW) -> promote_with_gate``
    and left in ``SHADOW`` / ``CANDIDATE`` when blocked.
    """
    rec, prov_ok = _coerce_lineage_receipt(lineage_receipt)
    total_i = _validate_positive_int(probe_total, "probe_total")
    pass_i = _validate_nonneg_int(probe_pass_count, "probe_pass_count")
    if pass_i > total_i:
        raise ValueError(f"probe_pass_count ({pass_i}) cannot exceed probe_total ({total_i})")
    min_total_i = _validate_positive_int(min_probe_total, "min_probe_total")
    min_ratio_f = _validate_unit_interval(min_pass_ratio, "min_pass_ratio", allow_zero=False)
    shadow_drift_f = _validate_finite_float(shadow_general_drift_rel, "shadow_general_drift_rel")
    max_drift_f = _validate_nonneg_finite(max_general_drift_rel, "max_general_drift_rel")

    if registry is not None:
        if not isinstance(registry, ExpertRegistry):
            raise TypeError(f"registry must be an ExpertRegistry or None, got {type(registry).__name__}")
        eid_clean = _validate_nonempty_str(expert_id, "expert_id")
        cluster_clean = _validate_nonempty_str(cluster_id, "cluster_id")
        size_i = _validate_positive_int(adapter_size_mib, "adapter_size_mib")
    else:
        eid_clean = ""
        cluster_clean = ""
        size_i = 0

    pass_ratio = float(pass_i) / float(total_i)
    reasons: list[str] = []
    if not prov_ok:
        reasons.append("lineage_receipt SHA-256 provenance check failed")
    if rec.promotion_stage != PROMOTION_STAGE_PROMOTED_SHADOW:
        reasons.append(
            f"promotion_stage must be {PROMOTION_STAGE_PROMOTED_SHADOW!r}, got {rec.promotion_stage!r}"
        )
    if rec.wsp_verdict != "PASS" or not rec.s5_guard_ok or not rec.guard_ok:
        reasons.append(
            f"lineage receipt is not a PASS shadow candidate (wsp_verdict={rec.wsp_verdict!r}, guard_ok={rec.s5_guard_ok})"
        )
    if not (rec.s5_anse_energy < rec.raw_anse_energy):
        reasons.append(
            f"S5 energy {rec.s5_anse_energy:.6f} does not strictly beat raw energy {rec.raw_anse_energy:.6f}"
        )
    if rec.general_rel_drift > max_drift_f + 1e-12:
        reasons.append(
            f"lineage general_rel_drift={rec.general_rel_drift:.6f} > max_general_drift_rel={max_drift_f:.6f}"
        )
    if total_i < min_total_i:
        reasons.append(f"probe_total={total_i} < min_probe_total={min_total_i}")
    if pass_ratio < min_ratio_f - 1e-12:
        reasons.append(
            f"router canary pass ratio {pass_i}/{total_i} ({pass_ratio:.4f}) < min_pass_ratio={min_ratio_f:.4f}"
        )
    if shadow_drift_f > max_drift_f + 1e-12:
        reasons.append(
            f"shadow_general_drift_rel={shadow_drift_f:.6f} > max_general_drift_rel={max_drift_f:.6f}"
        )

    approved = len(reasons) == 0
    gate_payload = build_expert_registry_gate_payload(
        rec,
        approved=approved,
        probe_pass_count=pass_i,
        probe_total=total_i,
        min_pass_ratio=min_ratio_f,
        shadow_general_drift_rel=shadow_drift_f,
        max_general_drift_rel=max_drift_f,
        base_model_id=base_model_id,
    )

    if registry is not None:
        resolved_base = str(gate_payload["base"])
        if eid_clean not in registry:
            registry.register(
                ExpertRecord(
                    expert_id=eid_clean,
                    kind=EXPERT_LORA_DOMAIN,
                    cluster_id=cluster_clean,
                    base_model_id=resolved_base,
                    adapter_sha256=rec.sanitized_s5_sha256,
                    adapter_size_mib=size_i,
                    keywords=tuple(keywords),
                    source="opentrainer_auto_wsp_hook",
                )
            )
            if rec.promotion_stage == PROMOTION_STAGE_PROMOTED_SHADOW:
                registry.transition(
                    eid_clean,
                    EXPERT_SHADOW,
                    f"WSP S5 shadow receipt {rec.receipt_sha256[:16]} verified",
                )
        elif approved and registry.get(eid_clean).state == EXPERT_CANDIDATE:
            registry.transition(
                eid_clean,
                EXPERT_SHADOW,
                f"WSP S5 shadow receipt {rec.receipt_sha256[:16]} verified",
            )

        if approved:
            registry.promote_with_gate(eid_clean, gate_payload)

    next_stage = PROMOTION_STAGE_PROMOTED_LIVE if approved else rec.promotion_stage
    gate_report = {
        "object": CANARY_GATE_OBJECT,
        "schema": SCHEMA_VERSION,
        "approved": approved,
        "probe_pass_count": pass_i,
        "probe_total": total_i,
        "probe_pass_ratio": pass_ratio,
        "min_pass_ratio": min_ratio_f,
        "min_probe_total": min_total_i,
        "shadow_general_drift_rel": shadow_drift_f,
        "max_general_drift_rel": max_drift_f,
        "reasons": reasons if reasons else [f"Canary gate passed {pass_i}/{total_i} probes within drift guard."],
        "gate_payload": gate_payload,
    }
    updated_dict = rec.to_dict()
    updated_dict["promotion_stage"] = next_stage
    updated_dict["canary_passed"] = approved
    updated_dict["canary_gate_report"] = gate_report
    updated_dict["reason"] = (
        f"{rec.reason} | LIVE canary PASS ({pass_i}/{total_i})."
        if approved
        else f"{rec.reason} | LIVE canary BLOCKED: {'; '.join(reasons)}"
    )
    tuple_fields = {
        "beneficial_anchor_layers",
        "interfering_layers",
        "editable_layers",
        "sink_experts",
        "preserved_specialists",
        "gated_targets",
        "committed_stages",
        "rolled_back_stages",
        "skipped_stages",
    }
    for tf in tuple_fields:
        updated_dict[tf] = tuple(updated_dict[tf])
    return _seal_lineage_receipt(updated_dict)


__all__ = [
    "CANARY_GATE_OBJECT",
    "DEFAULT_ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SOFT_MARGIN_PENALTY",
    "DEFAULT_CANARY_PROBE_TOTAL",
    "DEFAULT_GENERAL_GUARD_REL",
    "DEFAULT_MAX_TOTAL_EDIT_FRACTION",
    "DEFAULT_TEMPERATURE_GRID",
    "FINE_LAMBDA_LN_GRID",
    "GENERAL_EXPERT_PREFIX",
    "LINEAGE_RECEIPT_OBJECT",
    "PROMOTION_STAGE_PROMOTED_LIVE",
    "PROMOTION_STAGE_PROMOTED_SHADOW",
    "PROMOTION_STAGE_REJECTED_PARENT",
    "PROMOTION_STAGE_REJECTED_RAW",
    "SCHEMA_VERSION",
    "VALID_PROMOTION_STAGES",
    "VALID_WSP_VERDICTS",
    "PostRoundWSPHookConfig",
    "PostRoundWSPLineageReceipt",
    "build_expert_registry_gate_payload",
    "evaluate_shadow_to_live_canary_gate",
    "run_post_outer_round_auto_wsp_hook",
    "verify_lineage_receipt_provenance",
]

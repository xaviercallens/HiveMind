# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Sequential SPRT & Confidence-Sequence Shadow-Canary Traffic Router for WSP Endpoints.

Implements an anytime-valid progressive traffic router (``0% -> 5% -> 25% -> 100%``)
for S5 Weight-Space Programming (WSP) open-weight producer checkpoints running
alongside pinned base checkpoints:

1. **Sequential Probability Ratio Test (SPRT) + Anytime-Valid Confidence Sequence**:
   ``update_sprt_canary_step`` tracks paired live/shadow token NLL deltas
   (``wsp_nll - base_nll``) with Welford online variance estimation, predictable-scale
   Wald log-likelihood ratio updates (``sprt_log_lr`` / ``stage_sprt_log_lr``), and
   Howard et al. normal-mixture confidence-sequence bounds to control Type-I false
   promotions across arbitrary stopping times.
2. **Instant 0-Traffic Atomic Rollback Guards**:
   Monitors general-prose drift (``E0_*`` cumulative, stage-local, sliding-window,
   and single-step spike caps against ``+1.0%``), repetition-entropy collapse loops,
   and p95 latency regressions, triggering an immediate fail-closed rollback to ``0.0``
   live traffic (``"rolled_back"``, ``ANSE_MAXIMUM_PAIN = 1e6``) on any violation.
3. **Stateless Deterministic SHA-256 Request Bucketing**:
   ``route_canary_request_deterministic`` maps ``(request_key, salt)`` to a monotone
   uniform bucket in ``[0.0, 1.0)`` via canonical SHA-256 hashing and seals each routing
   decision with an independently verifiable cryptographic audit receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION: str = "hivemind.shadow_canary_wsp_traffic_router/1"
STATE_RECEIPT_OBJECT: str = "hivemind.shadow_canary_state_receipt"
ROUTING_RECEIPT_OBJECT: str = "hivemind.shadow_canary_routing_receipt"

DEFAULT_RAMP_STAGES: tuple[float, ...] = (0.0, 0.05, 0.25, 1.0)
DEFAULT_ALPHA: float = 0.01
DEFAULT_BETA: float = 0.05
DEFAULT_NULL_GAIN_NATS: float = 0.0
DEFAULT_ALT_GAIN_NATS: float = 0.14
DEFAULT_SIGMA_FLOOR_NATS: float = 0.15
DEFAULT_CS_PRIOR_RHO: float = 0.10
DEFAULT_MIN_SAMPLES_PER_STAGE: int = 8
DEFAULT_MIN_GENERAL_SAMPLES_PER_STAGE: int = 0
DEFAULT_MAX_GENERAL_DRIFT_REL: float = 0.01
DEFAULT_SOFT_GENERAL_MARGIN_REL: float = 0.0085
DEFAULT_SINGLE_STEP_GENERAL_HARD_CAP_REL: float = 0.02
DEFAULT_GENERAL_GUARD_WINDOW: int = 4
DEFAULT_MAX_LATENCY_P95_RATIO: float = 1.25
DEFAULT_HARD_LATENCY_SPIKE_RATIO: float = 3.0
DEFAULT_MIN_LATENCY_P95_SAMPLES: int = 5
DEFAULT_LATENCY_WINDOW_SIZE: int = 256
DEFAULT_MAX_REPETITION_RATE: float = 0.30
DEFAULT_MIN_TOKEN_ENTROPY_NATS: float = 0.50
DEFAULT_ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_ANSE_SOFT_MARGIN_PENALTY: float = 0.05

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_UINT64_SCALE = float(1 << 64)


# ---------------------------------------------------------------------------
# 1. Validation & Canonical SHA-256 Provenance Helpers
# ---------------------------------------------------------------------------


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_nonempty_str(value: Any, name: str) -> str:
    """Validate that ``value`` is a non-empty printable string."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped or any(ord(ch) < 32 or ord(ch) == 127 for ch in stripped):
        raise ValueError(f"{name} must be a non-empty printable string")
    return stripped


def _validate_str_allow_empty(value: Any, name: str) -> str:
    """Validate that ``value`` is a printable string (allowing empty)."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{name} must not contain control characters")
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


def _validate_unit_interval(
    value: Any,
    name: str,
    *,
    allow_zero: bool = True,
    allow_one: bool = True,
) -> float:
    """Validate that ``value`` is finite and lies inside the configured unit interval."""
    out = _validate_nonneg_finite(value, name)
    if not allow_zero and out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    if allow_one and out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    if not allow_one and out >= 1.0:
        raise ValueError(f"{name} must be < 1.0, got {value!r}")
    return out


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically with strict finite-number enforcement."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def compute_payload_sha256(payload: Any) -> str:
    """Return lowercase SHA-256 hex digest of ``canonical_json_bytes(payload)``."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _compute_percentile(values: Sequence[float], q: float) -> float:
    """Compute linear-interpolated percentile ``q in [0, 1]`` over a non-empty sequence."""
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = float(q) * float(len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - float(lo)
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _stage_name_for_ramp(ramp_fraction: float, *, rolled_back: bool) -> str:
    """Return canonical lifecycle stage label for a ramp fraction."""
    if rolled_back:
        return "rolled_back"
    if ramp_fraction <= 0.0:
        return "shadow"
    if ramp_fraction >= 1.0:
        return "live_100"
    pct = ramp_fraction * 100.0
    if abs(pct - round(pct)) < 1e-6:
        return f"canary_{int(round(pct))}"
    return f"canary_{pct:.2f}".rstrip("0").rstrip(".")


def _active_endpoint_for_ramp(ramp_fraction: float) -> str:
    """Return active serving endpoint descriptor for a ramp fraction."""
    if ramp_fraction <= 0.0:
        return "base"
    if ramp_fraction >= 1.0:
        return "wsp"
    return "canary_split"


# ---------------------------------------------------------------------------
# 2. Configuration, Telemetry State & Routing Receipt Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowCanaryConfig:
    """Hyperparameters and safety guardrails for the SPRT shadow-canary traffic router."""

    ramp_stages: tuple[float, ...] = DEFAULT_RAMP_STAGES
    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA
    null_gain_nats: float = DEFAULT_NULL_GAIN_NATS
    alt_gain_nats: float = DEFAULT_ALT_GAIN_NATS
    sigma_floor_nats: float = DEFAULT_SIGMA_FLOOR_NATS
    cs_prior_rho: float = DEFAULT_CS_PRIOR_RHO
    min_samples_per_stage: int = DEFAULT_MIN_SAMPLES_PER_STAGE
    min_general_samples_per_stage: int = DEFAULT_MIN_GENERAL_SAMPLES_PER_STAGE
    max_general_drift_rel: float = DEFAULT_MAX_GENERAL_DRIFT_REL
    soft_general_margin_rel: float = DEFAULT_SOFT_GENERAL_MARGIN_REL
    single_step_general_hard_cap_rel: float = DEFAULT_SINGLE_STEP_GENERAL_HARD_CAP_REL
    general_guard_window: int = DEFAULT_GENERAL_GUARD_WINDOW
    max_latency_p95_ratio: float = DEFAULT_MAX_LATENCY_P95_RATIO
    hard_latency_spike_ratio: float = DEFAULT_HARD_LATENCY_SPIKE_RATIO
    min_latency_p95_samples: int = DEFAULT_MIN_LATENCY_P95_SAMPLES
    latency_window_size: int = DEFAULT_LATENCY_WINDOW_SIZE
    max_repetition_rate: float = DEFAULT_MAX_REPETITION_RATE
    min_token_entropy_nats: float = DEFAULT_MIN_TOKEN_ENTROPY_NATS
    rollback_on_latency_breach: bool = True
    lock_after_rollback: bool = True
    maximum_pain: float = DEFAULT_ANSE_MAXIMUM_PAIN
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY

    def __post_init__(self) -> None:
        if isinstance(self.ramp_stages, (str, bytes)) or not isinstance(self.ramp_stages, Sequence):
            raise TypeError(f"ramp_stages must be a Sequence of floats, got {type(self.ramp_stages).__name__}")
        if len(self.ramp_stages) < 2:
            raise ValueError("ramp_stages must contain at least 2 stages (0.0 and 1.0)")
        normalized_stages = tuple(_validate_unit_interval(v, "ramp_stages[]") for v in self.ramp_stages)
        if normalized_stages[0] != 0.0:
            raise ValueError(f"ramp_stages[0] must be 0.0, got {normalized_stages[0]}")
        if normalized_stages[-1] != 1.0:
            raise ValueError(f"ramp_stages[-1] must be 1.0, got {normalized_stages[-1]}")
        for idx in range(1, len(normalized_stages)):
            if normalized_stages[idx] <= normalized_stages[idx - 1]:
                raise ValueError(f"ramp_stages must be strictly increasing, got {normalized_stages}")
        object.__setattr__(self, "ramp_stages", normalized_stages)

        alpha_f = _validate_unit_interval(self.alpha, "alpha", allow_zero=False, allow_one=False)
        beta_f = _validate_unit_interval(self.beta, "beta", allow_zero=False, allow_one=False)
        if alpha_f >= 0.5 or beta_f >= 0.5 or (alpha_f + beta_f) >= 1.0:
            raise ValueError(
                f"alpha ({alpha_f}) and beta ({beta_f}) must each be < 0.5 with alpha + beta < 1.0"
            )
        object.__setattr__(self, "alpha", alpha_f)
        object.__setattr__(self, "beta", beta_f)

        null_f = _validate_finite_float(self.null_gain_nats, "null_gain_nats")
        alt_f = _validate_finite_float(self.alt_gain_nats, "alt_gain_nats")
        if alt_f <= null_f:
            raise ValueError(f"alt_gain_nats ({alt_f}) must exceed null_gain_nats ({null_f})")
        object.__setattr__(self, "null_gain_nats", null_f)
        object.__setattr__(self, "alt_gain_nats", alt_f)

        object.__setattr__(self, "sigma_floor_nats", _validate_positive_finite(self.sigma_floor_nats, "sigma_floor_nats"))
        object.__setattr__(self, "cs_prior_rho", _validate_positive_finite(self.cs_prior_rho, "cs_prior_rho"))
        object.__setattr__(
            self,
            "min_samples_per_stage",
            _validate_positive_int(self.min_samples_per_stage, "min_samples_per_stage"),
        )
        object.__setattr__(
            self,
            "min_general_samples_per_stage",
            _validate_nonneg_int(self.min_general_samples_per_stage, "min_general_samples_per_stage"),
        )

        max_gen = _validate_positive_finite(self.max_general_drift_rel, "max_general_drift_rel")
        soft_gen = _validate_nonneg_finite(self.soft_general_margin_rel, "soft_general_margin_rel")
        hard_cap = _validate_positive_finite(
            self.single_step_general_hard_cap_rel, "single_step_general_hard_cap_rel"
        )
        if soft_gen > max_gen:
            raise ValueError(f"soft_general_margin_rel ({soft_gen}) cannot exceed max_general_drift_rel ({max_gen})")
        if hard_cap < max_gen:
            raise ValueError(
                f"single_step_general_hard_cap_rel ({hard_cap}) must be >= max_general_drift_rel ({max_gen})"
            )
        object.__setattr__(self, "max_general_drift_rel", max_gen)
        object.__setattr__(self, "soft_general_margin_rel", soft_gen)
        object.__setattr__(self, "single_step_general_hard_cap_rel", hard_cap)
        object.__setattr__(
            self,
            "general_guard_window",
            _validate_positive_int(self.general_guard_window, "general_guard_window"),
        )

        max_lat = _validate_positive_finite(self.max_latency_p95_ratio, "max_latency_p95_ratio")
        if max_lat < 1.0:
            raise ValueError(f"max_latency_p95_ratio must be >= 1.0, got {max_lat}")
        hard_lat = _validate_positive_finite(self.hard_latency_spike_ratio, "hard_latency_spike_ratio")
        if hard_lat < max_lat:
            raise ValueError(
                f"hard_latency_spike_ratio ({hard_lat}) must be >= max_latency_p95_ratio ({max_lat})"
            )
        object.__setattr__(self, "max_latency_p95_ratio", max_lat)
        object.__setattr__(self, "hard_latency_spike_ratio", hard_lat)
        object.__setattr__(
            self,
            "min_latency_p95_samples",
            _validate_positive_int(self.min_latency_p95_samples, "min_latency_p95_samples"),
        )
        object.__setattr__(
            self,
            "latency_window_size",
            _validate_positive_int(self.latency_window_size, "latency_window_size"),
        )
        object.__setattr__(
            self,
            "max_repetition_rate",
            _validate_unit_interval(self.max_repetition_rate, "max_repetition_rate", allow_zero=False),
        )
        object.__setattr__(
            self,
            "min_token_entropy_nats",
            _validate_nonneg_finite(self.min_token_entropy_nats, "min_token_entropy_nats"),
        )
        object.__setattr__(
            self,
            "rollback_on_latency_breach",
            _validate_bool(self.rollback_on_latency_breach, "rollback_on_latency_breach"),
        )
        object.__setattr__(
            self,
            "lock_after_rollback",
            _validate_bool(self.lock_after_rollback, "lock_after_rollback"),
        )
        object.__setattr__(self, "maximum_pain", _validate_positive_finite(self.maximum_pain, "maximum_pain"))
        object.__setattr__(
            self,
            "soft_margin_penalty",
            _validate_nonneg_finite(self.soft_margin_penalty, "soft_margin_penalty"),
        )

    @property
    def sprt_upper_bound(self) -> float:
        """Wald SPRT upper log-likelihood ratio boundary ``log((1 - beta) / alpha)``."""
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def sprt_lower_bound(self) -> float:
        """Wald SPRT lower log-likelihood ratio boundary ``log(beta / (1 - alpha))``."""
        return math.log(self.beta / (1.0 - self.alpha))

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a deterministic JSON-safe dictionary."""
        data = asdict(self)
        data["ramp_stages"] = list(self.ramp_stages)
        data["sprt_upper_bound"] = self.sprt_upper_bound
        data["sprt_lower_bound"] = self.sprt_lower_bound
        return data


@dataclass(frozen=True)
class ShadowCanaryState:
    """Immutable snapshot of sequential SPRT, confidence sequence, and safety-guard telemetry."""

    config: ShadowCanaryConfig = ShadowCanaryConfig()
    base_checkpoint_id: str = "base_pinned"
    wsp_checkpoint_id: str = "wsp_s5_candidate"
    canary_salt: str = "hivemind-s5-canary"

    stage_index: int = 0
    ramp_fraction: float = 0.0
    stage_name: str = "shadow"
    active_endpoint: str = "base"
    promoted_to_live: bool = False
    rollback_triggered: bool = False
    rollback_reasons: tuple[str, ...] = ()
    last_transition: str = "initialized"
    transition_history: tuple[dict[str, Any], ...] = ()

    total_samples: int = 0
    stage_samples: int = 0
    base_nll_sum: float = 0.0
    wsp_nll_sum: float = 0.0
    nll_diff_sum: float = 0.0
    nll_diff_mean: float = 0.0
    gain_mean: float = 0.0
    gain_m2: float = 0.0
    gain_std: float = 0.0
    stage_gain_mean: float = 0.0
    stage_gain_m2: float = 0.0

    sprt_log_lr: float = 0.0
    stage_sprt_log_lr: float = 0.0
    live_regression_cusum: float = 0.0
    cs_variance_process: float = 0.0
    cs_half_width: float | None = None
    cs_gain_lower: float | None = None
    cs_gain_upper: float | None = None
    cs_nll_diff_lower: float | None = None
    cs_nll_diff_upper: float | None = None

    general_samples: int = 0
    stage_general_samples: int = 0
    general_base_nll_sum: float = 0.0
    general_wsp_nll_sum: float = 0.0
    stage_general_base_nll_sum: float = 0.0
    stage_general_wsp_nll_sum: float = 0.0
    general_recent_rel_drifts: tuple[float, ...] = ()
    general_rel_drift: float = 0.0
    stage_general_rel_drift: float = 0.0
    general_recent_rel_drift: float = 0.0
    worst_general_step_rel_drift: float = 0.0
    general_guard_violations: int = 0

    latency_window: tuple[float, ...] = ()
    latency_p95_ratio: float = 1.0
    latency_max_ratio: float = 1.0
    latency_mean_ratio: float = 1.0
    latency_ratio_sum: float = 0.0
    latency_violation_count: int = 0

    entropy_collapse_count: int = 0
    entropy_collapse_detected: bool = False
    repetition_loop_count: int = 0
    min_observed_token_entropy_nats: float | None = None
    max_observed_repetition_rate: float = 0.0

    anse_energy: float = 0.0
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.config, ShadowCanaryConfig):
            raise TypeError(f"config must be a ShadowCanaryConfig, got {type(self.config).__name__}")
        object.__setattr__(self, "base_checkpoint_id", _validate_nonempty_str(self.base_checkpoint_id, "base_checkpoint_id"))
        object.__setattr__(self, "wsp_checkpoint_id", _validate_nonempty_str(self.wsp_checkpoint_id, "wsp_checkpoint_id"))
        object.__setattr__(self, "canary_salt", _validate_str_allow_empty(self.canary_salt, "canary_salt"))

        stage_idx = _validate_nonneg_int(self.stage_index, "stage_index")
        if stage_idx >= len(self.config.ramp_stages):
            raise ValueError(f"stage_index ({stage_idx}) out of bounds for {len(self.config.ramp_stages)} stages")
        object.__setattr__(self, "stage_index", stage_idx)
        object.__setattr__(self, "ramp_fraction", _validate_unit_interval(self.ramp_fraction, "ramp_fraction"))
        object.__setattr__(self, "stage_name", _validate_nonempty_str(self.stage_name, "stage_name"))
        object.__setattr__(self, "active_endpoint", _validate_nonempty_str(self.active_endpoint, "active_endpoint"))
        object.__setattr__(self, "promoted_to_live", _validate_bool(self.promoted_to_live, "promoted_to_live"))
        object.__setattr__(self, "rollback_triggered", _validate_bool(self.rollback_triggered, "rollback_triggered"))
        object.__setattr__(self, "last_transition", _validate_nonempty_str(self.last_transition, "last_transition"))

        reasons_t = tuple(_validate_nonempty_str(r, "rollback_reasons[]") for r in self.rollback_reasons)
        history_t = tuple(dict(item) for item in self.transition_history)
        drifts_t = tuple(_validate_finite_float(v, "general_recent_rel_drifts[]") for v in self.general_recent_rel_drifts)
        lat_t = tuple(_validate_positive_finite(v, "latency_window[]") for v in self.latency_window)
        object.__setattr__(self, "rollback_reasons", reasons_t)
        object.__setattr__(self, "transition_history", history_t)
        object.__setattr__(self, "general_recent_rel_drifts", drifts_t)
        object.__setattr__(self, "latency_window", lat_t)

        object.__setattr__(self, "total_samples", _validate_nonneg_int(self.total_samples, "total_samples"))
        object.__setattr__(self, "stage_samples", _validate_nonneg_int(self.stage_samples, "stage_samples"))
        object.__setattr__(self, "general_samples", _validate_nonneg_int(self.general_samples, "general_samples"))
        object.__setattr__(self, "stage_general_samples", _validate_nonneg_int(self.stage_general_samples, "stage_general_samples"))
        object.__setattr__(
            self, "general_guard_violations", _validate_nonneg_int(self.general_guard_violations, "general_guard_violations")
        )
        object.__setattr__(
            self, "latency_violation_count", _validate_nonneg_int(self.latency_violation_count, "latency_violation_count")
        )
        object.__setattr__(
            self, "entropy_collapse_count", _validate_nonneg_int(self.entropy_collapse_count, "entropy_collapse_count")
        )
        object.__setattr__(
            self, "entropy_collapse_detected", _validate_bool(self.entropy_collapse_detected, "entropy_collapse_detected")
        )
        object.__setattr__(
            self, "repetition_loop_count", _validate_nonneg_int(self.repetition_loop_count, "repetition_loop_count")
        )

        for fname in (
            "base_nll_sum",
            "wsp_nll_sum",
            "nll_diff_sum",
            "nll_diff_mean",
            "gain_mean",
            "gain_m2",
            "gain_std",
            "stage_gain_mean",
            "stage_gain_m2",
            "sprt_log_lr",
            "stage_sprt_log_lr",
            "live_regression_cusum",
            "cs_variance_process",
            "general_base_nll_sum",
            "general_wsp_nll_sum",
            "stage_general_base_nll_sum",
            "stage_general_wsp_nll_sum",
            "general_rel_drift",
            "stage_general_rel_drift",
            "general_recent_rel_drift",
            "worst_general_step_rel_drift",
            "latency_p95_ratio",
            "latency_max_ratio",
            "latency_mean_ratio",
            "latency_ratio_sum",
            "max_observed_repetition_rate",
            "anse_energy",
        ):
            object.__setattr__(self, fname, _validate_finite_float(getattr(self, fname), fname))

        for opt_fname in (
            "cs_half_width",
            "cs_gain_lower",
            "cs_gain_upper",
            "cs_nll_diff_lower",
            "cs_nll_diff_upper",
            "min_observed_token_entropy_nats",
        ):
            val = getattr(self, opt_fname)
            if val is not None:
                object.__setattr__(self, opt_fname, _validate_finite_float(val, opt_fname))

        if not isinstance(self.receipt_sha256, str):
            raise TypeError("receipt_sha256 must be a str")
        if self.receipt_sha256 == "":
            sealed = compute_payload_sha256(self.core_dict())
            object.__setattr__(self, "receipt_sha256", sealed)

    def core_dict(self) -> dict[str, Any]:
        """Return canonical receipt fields prior to SHA-256 sealing."""
        return {
            "object": STATE_RECEIPT_OBJECT,
            "schema": SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "base_checkpoint_id": self.base_checkpoint_id,
            "wsp_checkpoint_id": self.wsp_checkpoint_id,
            "canary_salt": self.canary_salt,
            "stage_index": self.stage_index,
            "ramp_fraction": self.ramp_fraction,
            "stage_name": self.stage_name,
            "active_endpoint": self.active_endpoint,
            "promoted_to_live": self.promoted_to_live,
            "rollback_triggered": self.rollback_triggered,
            "rollback_reasons": list(self.rollback_reasons),
            "last_transition": self.last_transition,
            "transition_history": [dict(entry) for entry in self.transition_history],
            "total_samples": self.total_samples,
            "stage_samples": self.stage_samples,
            "base_nll_sum": self.base_nll_sum,
            "wsp_nll_sum": self.wsp_nll_sum,
            "nll_diff_sum": self.nll_diff_sum,
            "nll_diff_mean": self.nll_diff_mean,
            "gain_mean": self.gain_mean,
            "gain_m2": self.gain_m2,
            "gain_std": self.gain_std,
            "stage_gain_mean": self.stage_gain_mean,
            "stage_gain_m2": self.stage_gain_m2,
            "sprt_log_lr": self.sprt_log_lr,
            "stage_sprt_log_lr": self.stage_sprt_log_lr,
            "live_regression_cusum": self.live_regression_cusum,
            "cs_variance_process": self.cs_variance_process,
            "cs_half_width": self.cs_half_width,
            "cs_gain_lower": self.cs_gain_lower,
            "cs_gain_upper": self.cs_gain_upper,
            "cs_nll_diff_lower": self.cs_nll_diff_lower,
            "cs_nll_diff_upper": self.cs_nll_diff_upper,
            "general_samples": self.general_samples,
            "stage_general_samples": self.stage_general_samples,
            "general_base_nll_sum": self.general_base_nll_sum,
            "general_wsp_nll_sum": self.general_wsp_nll_sum,
            "stage_general_base_nll_sum": self.stage_general_base_nll_sum,
            "stage_general_wsp_nll_sum": self.stage_general_wsp_nll_sum,
            "general_recent_rel_drifts": list(self.general_recent_rel_drifts),
            "general_rel_drift": self.general_rel_drift,
            "stage_general_rel_drift": self.stage_general_rel_drift,
            "general_recent_rel_drift": self.general_recent_rel_drift,
            "worst_general_step_rel_drift": self.worst_general_step_rel_drift,
            "general_guard_violations": self.general_guard_violations,
            "latency_window": list(self.latency_window),
            "latency_p95_ratio": self.latency_p95_ratio,
            "latency_max_ratio": self.latency_max_ratio,
            "latency_mean_ratio": self.latency_mean_ratio,
            "latency_ratio_sum": self.latency_ratio_sum,
            "latency_violation_count": self.latency_violation_count,
            "entropy_collapse_count": self.entropy_collapse_count,
            "entropy_collapse_detected": self.entropy_collapse_detected,
            "repetition_loop_count": self.repetition_loop_count,
            "min_observed_token_entropy_nats": self.min_observed_token_entropy_nats,
            "max_observed_repetition_rate": self.max_observed_repetition_rate,
            "anse_energy": self.anse_energy,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize state telemetry and SHA-256 provenance to a JSON-safe dictionary."""
        payload = self.core_dict()
        payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True)
class CanaryRoutingDecision:
    """Stateless deterministic SHA-256 routing decision with cryptographic audit receipt."""

    request_key: str
    salt: str
    ramp_fraction: float
    bucket_digest_sha256: str
    bucket_unit: float
    bucket_bps: int
    routed_to_wsp: bool
    selected_endpoint: str
    selected_checkpoint_id: str
    base_checkpoint_id: str
    wsp_checkpoint_id: str
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize routing decision and cryptographic audit receipt to JSON-safe dict."""
        return {
            "object": ROUTING_RECEIPT_OBJECT,
            "schema": SCHEMA_VERSION,
            "request_key": self.request_key,
            "salt": self.salt,
            "ramp_fraction": self.ramp_fraction,
            "bucket_digest_sha256": self.bucket_digest_sha256,
            "bucket_unit": self.bucket_unit,
            "bucket_bps": self.bucket_bps,
            "routed_to_wsp": self.routed_to_wsp,
            "selected_endpoint": self.selected_endpoint,
            "selected_checkpoint_id": self.selected_checkpoint_id,
            "base_checkpoint_id": self.base_checkpoint_id,
            "wsp_checkpoint_id": self.wsp_checkpoint_id,
            "receipt_sha256": self.receipt_sha256,
        }


# ---------------------------------------------------------------------------
# 3. Sequential SPRT, Confidence Sequence & Atomic Rollback Engine
# ---------------------------------------------------------------------------


def create_shadow_canary_state(
    config: ShadowCanaryConfig | None = None,
    *,
    base_checkpoint_id: str = "base_pinned",
    wsp_checkpoint_id: str = "wsp_s5_candidate",
    canary_salt: str = "hivemind-s5-canary",
) -> ShadowCanaryState:
    """Create an initialized ``ShadowCanaryState`` pinned at 0% shadow traffic."""
    cfg = config if config is not None else ShadowCanaryConfig()
    return ShadowCanaryState(
        config=cfg,
        base_checkpoint_id=base_checkpoint_id,
        wsp_checkpoint_id=wsp_checkpoint_id,
        canary_salt=canary_salt,
    )


def _compute_sprt_and_cs_update(
    state: ShadowCanaryState,
    cfg: ShadowCanaryConfig,
    base_nll: float,
    wsp_nll: float,
) -> dict[str, Any]:
    """Update Welford online moments, self-stabilized Wald SPRT, CUSUM, and confidence sequence."""
    diff = wsp_nll - base_nll
    gain = -diff

    prev_n = state.total_samples
    prev_mean = state.gain_mean
    prev_m2 = state.gain_m2
    prev_var = (prev_m2 / float(prev_n - 1)) if prev_n >= 2 else 0.0
    step_dev2 = gain * gain if prev_n == 0 else (gain - prev_mean) ** 2

    # Self-stabilized variance denominator prevents single-step outlier LLR inflation under heavy noise.
    sigma_floor2 = cfg.sigma_floor_nats * cfg.sigma_floor_nats
    sigma2_sprt = max(sigma_floor2, prev_var, step_dev2)
    delta_eff = cfg.alt_gain_nats - cfg.null_gain_nats
    midpoint = 0.5 * (cfg.null_gain_nats + cfg.alt_gain_nats)
    step_llr = (delta_eff / sigma2_sprt) * (gain - midpoint)

    sprt_log_lr = state.sprt_log_lr + step_llr
    stage_sprt_log_lr = state.stage_sprt_log_lr + step_llr

    # Stage-local Page's CUSUM SPRT monitors active live stages for post-promotion degradation.
    s_prev_n = state.stage_samples
    s_prev_mean = state.stage_gain_mean
    s_prev_m2 = state.stage_gain_m2
    s_prev_var = (s_prev_m2 / float(s_prev_n - 1)) if s_prev_n >= 2 else 0.0
    sigma2_reg = max(sigma_floor2, s_prev_var)
    if state.stage_index > 0:
        step_reg = (delta_eff / sigma2_reg) * (-gain - 0.5 * delta_eff)
        live_regression_cusum = max(0.0, state.live_regression_cusum + step_reg)
    else:
        live_regression_cusum = 0.0

    cs_v = state.cs_variance_process + sigma2_sprt

    total_samples = prev_n + 1
    stage_samples = s_prev_n + 1
    delta1 = gain - prev_mean
    gain_mean = prev_mean + delta1 / float(total_samples)
    delta2 = gain - gain_mean
    gain_m2 = prev_m2 + delta1 * delta2
    gain_std = math.sqrt(max(0.0, gain_m2 / float(total_samples - 1))) if total_samples >= 2 else 0.0

    s_delta1 = gain - s_prev_mean
    stage_gain_mean = s_prev_mean + s_delta1 / float(stage_samples)
    s_delta2 = gain - stage_gain_mean
    stage_gain_m2 = s_prev_m2 + s_delta1 * s_delta2

    base_nll_sum = state.base_nll_sum + base_nll
    wsp_nll_sum = state.wsp_nll_sum + wsp_nll
    nll_diff_sum = state.nll_diff_sum + diff
    nll_diff_mean = -gain_mean

    v_plus_rho = cs_v + cfg.cs_prior_rho
    log_term = math.log(v_plus_rho / (cfg.alpha * cfg.alpha * cfg.cs_prior_rho))
    cs_half_width = math.sqrt(max(0.0, v_plus_rho * log_term)) / float(total_samples)
    cs_gain_lower = gain_mean - cs_half_width
    cs_gain_upper = gain_mean + cs_half_width
    cs_nll_diff_lower = nll_diff_mean - cs_half_width
    cs_nll_diff_upper = nll_diff_mean + cs_half_width

    return {
        "total_samples": total_samples,
        "stage_samples": stage_samples,
        "base_nll_sum": base_nll_sum,
        "wsp_nll_sum": wsp_nll_sum,
        "nll_diff_sum": nll_diff_sum,
        "nll_diff_mean": nll_diff_mean,
        "gain_mean": gain_mean,
        "gain_m2": gain_m2,
        "gain_std": gain_std,
        "stage_gain_mean": stage_gain_mean,
        "stage_gain_m2": stage_gain_m2,
        "sprt_log_lr": sprt_log_lr,
        "stage_sprt_log_lr": stage_sprt_log_lr,
        "live_regression_cusum": live_regression_cusum,
        "cs_variance_process": cs_v,
        "cs_half_width": cs_half_width,
        "cs_gain_lower": cs_gain_lower,
        "cs_gain_upper": cs_gain_upper,
        "cs_nll_diff_lower": cs_nll_diff_lower,
        "cs_nll_diff_upper": cs_nll_diff_upper,
    }


def _compute_general_guard_update(
    state: ShadowCanaryState,
    cfg: ShadowCanaryConfig,
    base_nll: float,
    wsp_nll: float,
    is_general_domain: bool,
) -> tuple[dict[str, Any], bool]:
    """Update general-prose (``E0_*``) drift trackers and return ``(fields, breached)``."""
    if not is_general_domain:
        return {
            "general_samples": state.general_samples,
            "stage_general_samples": state.stage_general_samples,
            "general_base_nll_sum": state.general_base_nll_sum,
            "general_wsp_nll_sum": state.general_wsp_nll_sum,
            "stage_general_base_nll_sum": state.stage_general_base_nll_sum,
            "stage_general_wsp_nll_sum": state.stage_general_wsp_nll_sum,
            "general_recent_rel_drifts": state.general_recent_rel_drifts,
            "general_rel_drift": state.general_rel_drift,
            "stage_general_rel_drift": state.stage_general_rel_drift,
            "general_recent_rel_drift": state.general_recent_rel_drift,
            "worst_general_step_rel_drift": state.worst_general_step_rel_drift,
            "general_guard_violations": state.general_guard_violations,
        }, False

    step_rel = (wsp_nll - base_nll) / base_nll
    gen_samples = state.general_samples + 1
    stage_gen_samples = state.stage_general_samples + 1
    gen_base_sum = state.general_base_nll_sum + base_nll
    gen_wsp_sum = state.general_wsp_nll_sum + wsp_nll
    stage_gen_base = state.stage_general_base_nll_sum + base_nll
    stage_gen_wsp = state.stage_general_wsp_nll_sum + wsp_nll

    recent = (*state.general_recent_rel_drifts, step_rel)[-cfg.general_guard_window :]
    gen_rel = (gen_wsp_sum - gen_base_sum) / gen_base_sum
    stage_gen_rel = (stage_gen_wsp - stage_gen_base) / stage_gen_base
    recent_rel = sum(recent) / float(len(recent))
    worst_step_rel = max(state.worst_general_step_rel_drift, step_rel)

    breached = (
        gen_rel > cfg.max_general_drift_rel
        or stage_gen_rel > cfg.max_general_drift_rel
        or recent_rel > cfg.max_general_drift_rel
        or step_rel > cfg.single_step_general_hard_cap_rel
    )
    violations = state.general_guard_violations + (1 if breached else 0)

    return {
        "general_samples": gen_samples,
        "stage_general_samples": stage_gen_samples,
        "general_base_nll_sum": gen_base_sum,
        "general_wsp_nll_sum": gen_wsp_sum,
        "stage_general_base_nll_sum": stage_gen_base,
        "stage_general_wsp_nll_sum": stage_gen_wsp,
        "general_recent_rel_drifts": recent,
        "general_rel_drift": gen_rel,
        "stage_general_rel_drift": stage_gen_rel,
        "general_recent_rel_drift": recent_rel,
        "worst_general_step_rel_drift": worst_step_rel,
        "general_guard_violations": violations,
    }, breached


def _compute_latency_update(
    state: ShadowCanaryState,
    cfg: ShadowCanaryConfig,
    latency_ratio: float,
    total_samples: int,
) -> tuple[dict[str, Any], bool]:
    """Update rolling latency p95 telemetry and evaluate latency guardrail breaches."""
    window = (*state.latency_window, latency_ratio)[-cfg.latency_window_size :]
    p95_ratio = _compute_percentile(window, 0.95)
    ratio_sum = state.latency_ratio_sum + latency_ratio
    mean_ratio = ratio_sum / float(total_samples)
    max_ratio = latency_ratio if state.total_samples == 0 else max(state.latency_max_ratio, latency_ratio)
    step_violation = latency_ratio > cfg.max_latency_p95_ratio
    violations = state.latency_violation_count + (1 if step_violation else 0)

    p95_breached = len(window) >= cfg.min_latency_p95_samples and p95_ratio > cfg.max_latency_p95_ratio
    spike_breached = latency_ratio > cfg.hard_latency_spike_ratio
    breached = cfg.rollback_on_latency_breach and (p95_breached or spike_breached)

    return {
        "latency_window": window,
        "latency_p95_ratio": p95_ratio,
        "latency_max_ratio": max_ratio,
        "latency_mean_ratio": mean_ratio,
        "latency_ratio_sum": ratio_sum,
        "latency_violation_count": violations,
    }, breached


def _compute_entropy_update(
    state: ShadowCanaryState,
    cfg: ShadowCanaryConfig,
    *,
    entropy_collapse: bool,
    repetition_rate: float | None,
    token_entropy_nats: float | None,
) -> tuple[dict[str, Any], bool]:
    """Update repetition-entropy loop telemetry and detect collapse conditions."""
    rep_loop = repetition_rate is not None and repetition_rate > cfg.max_repetition_rate
    low_entropy = token_entropy_nats is not None and token_entropy_nats < cfg.min_token_entropy_nats
    step_collapse = entropy_collapse or rep_loop or low_entropy

    collapse_count = state.entropy_collapse_count + (1 if step_collapse else 0)
    collapse_detected = state.entropy_collapse_detected or step_collapse
    rep_count = state.repetition_loop_count + (1 if rep_loop else 0)

    if token_entropy_nats is None:
        min_entropy = state.min_observed_token_entropy_nats
    elif state.min_observed_token_entropy_nats is None:
        min_entropy = token_entropy_nats
    else:
        min_entropy = min(state.min_observed_token_entropy_nats, token_entropy_nats)

    max_rep = state.max_observed_repetition_rate if repetition_rate is None else max(
        state.max_observed_repetition_rate, repetition_rate
    )

    return {
        "entropy_collapse_count": collapse_count,
        "entropy_collapse_detected": collapse_detected,
        "repetition_loop_count": rep_count,
        "min_observed_token_entropy_nats": min_entropy,
        "max_observed_repetition_rate": max_rep,
    }, step_collapse


def _compute_anse_energy(
    cfg: ShadowCanaryConfig,
    *,
    nll_diff_mean: float,
    general_rel_drift: float,
    guard_ok: bool,
) -> float:
    """Compute objective ANSE energy with Maximum Pain barrier on safety violations."""
    if not guard_ok:
        return cfg.maximum_pain + max(0.0, nll_diff_mean)
    soft_penalty = cfg.soft_margin_penalty if general_rel_drift > cfg.soft_general_margin_rel else 0.0
    return nll_diff_mean + soft_penalty


def update_sprt_canary_step(
    state: ShadowCanaryState,
    base_nll: float,
    wsp_nll: float,
    *,
    is_general_domain: bool,
    latency_ratio: float = 1.0,
    entropy_collapse: bool = False,
    repetition_rate: float | None = None,
    token_entropy_nats: float | None = None,
    config: ShadowCanaryConfig | None = None,
) -> ShadowCanaryState:
    """Advance the SPRT shadow-canary state by one paired live/shadow observation.

    Promotes through ``config.ramp_stages`` (``0% -> 5% -> 25% -> 100%``) only when
    Wald SPRT and the anytime-valid confidence sequence jointly prove ``wsp_nll < base_nll``
    under clean general-prose, latency p95, and entropy guards. Triggers instant 0-traffic
    atomic rollback on general-prose drift ``> +1.0%``, entropy collapse, or latency breach.
    """
    if not isinstance(state, ShadowCanaryState):
        raise TypeError(f"state must be a ShadowCanaryState, got {type(state).__name__}")
    cfg = state.config if config is None else config
    if not isinstance(cfg, ShadowCanaryConfig):
        raise TypeError(f"config must be a ShadowCanaryConfig or None, got {type(cfg).__name__}")

    base_f = _validate_positive_finite(base_nll, "base_nll")
    wsp_f = _validate_nonneg_finite(wsp_nll, "wsp_nll")
    is_gen = _validate_bool(is_general_domain, "is_general_domain")
    lat_f = _validate_positive_finite(latency_ratio, "latency_ratio")
    ent_flag = _validate_bool(entropy_collapse, "entropy_collapse")
    rep_f = (
        None
        if repetition_rate is None
        else _validate_unit_interval(repetition_rate, "repetition_rate")
    )
    tok_ent_f = (
        None
        if token_entropy_nats is None
        else _validate_nonneg_finite(token_entropy_nats, "token_entropy_nats")
    )

    sprt_fields = _compute_sprt_and_cs_update(state, cfg, base_f, wsp_f)
    gen_fields, gen_breached = _compute_general_guard_update(state, cfg, base_f, wsp_f, is_gen)
    lat_fields, lat_breached = _compute_latency_update(state, cfg, lat_f, sprt_fields["total_samples"])
    ent_fields, ent_breached = _compute_entropy_update(
        state,
        cfg,
        entropy_collapse=ent_flag,
        repetition_rate=rep_f,
        token_entropy_nats=tok_ent_f,
    )

    step_reasons: list[str] = []
    if gen_breached:
        step_reasons.append("general_prose_drift_guard_exceeded")
    if ent_breached:
        step_reasons.append("entropy_collapse_or_repetition_loop")
    if lat_breached:
        step_reasons.append("latency_p95_guard_exceeded")

    sprt_live_regression = state.stage_index > 0 and (
        sprt_fields["stage_sprt_log_lr"] <= cfg.sprt_lower_bound
        or sprt_fields["live_regression_cusum"] >= abs(cfg.sprt_lower_bound)
    )
    if sprt_live_regression:
        step_reasons.append("sprt_live_regression_below_lower_bound")

    was_locked_rollback = state.rollback_triggered and cfg.lock_after_rollback
    rollback_now = len(step_reasons) > 0
    combined_reasons = tuple(dict.fromkeys((*state.rollback_reasons, *step_reasons)))
    history = list(state.transition_history)

    if was_locked_rollback or rollback_now:
        next_stage_index = 0
        next_ramp = 0.0
        next_stage_name = _stage_name_for_ramp(0.0, rolled_back=True)
        next_endpoint = _active_endpoint_for_ramp(0.0)
        next_promoted = False
        next_rollback = True
        next_stage_samples = 0
        next_stage_sprt = 0.0
        next_stage_gain_mean = 0.0
        next_stage_gain_m2 = 0.0
        next_live_cusum = 0.0
        gen_fields["stage_general_samples"] = 0
        gen_fields["stage_general_base_nll_sum"] = 0.0
        gen_fields["stage_general_wsp_nll_sum"] = 0.0
        gen_fields["stage_general_rel_drift"] = 0.0

        if not was_locked_rollback:
            last_transition = f"rollback:{','.join(step_reasons)}"
            history.append(
                {
                    "step": sprt_fields["total_samples"],
                    "event": "atomic_rollback",
                    "from_ramp": state.ramp_fraction,
                    "to_ramp": 0.0,
                    "reasons": list(step_reasons),
                    "general_rel_drift": gen_fields["general_rel_drift"],
                    "sprt_log_lr": sprt_fields["sprt_log_lr"],
                }
            )
        else:
            last_transition = "locked_after_rollback"
    else:
        can_promote = (
            state.stage_index < len(cfg.ramp_stages) - 1
            and sprt_fields["stage_samples"] >= cfg.min_samples_per_stage
            and gen_fields["stage_general_samples"] >= cfg.min_general_samples_per_stage
            and sprt_fields["stage_sprt_log_lr"] >= cfg.sprt_upper_bound
            and sprt_fields["sprt_log_lr"] >= cfg.sprt_upper_bound
            and sprt_fields["cs_gain_lower"] is not None
            and sprt_fields["cs_gain_lower"] > cfg.null_gain_nats
            and gen_fields["general_rel_drift"] <= cfg.max_general_drift_rel
            and lat_fields["latency_p95_ratio"] <= cfg.max_latency_p95_ratio
            and not ent_fields["entropy_collapse_detected"]
        )
        if can_promote:
            next_stage_index = state.stage_index + 1
            next_ramp = cfg.ramp_stages[next_stage_index]
            next_stage_name = _stage_name_for_ramp(next_ramp, rolled_back=False)
            next_endpoint = _active_endpoint_for_ramp(next_ramp)
            next_promoted = next_stage_index == len(cfg.ramp_stages) - 1
            next_rollback = False
            next_stage_samples = 0
            next_stage_sprt = 0.0
            next_stage_gain_mean = 0.0
            next_stage_gain_m2 = 0.0
            next_live_cusum = 0.0
            gen_fields["stage_general_samples"] = 0
            gen_fields["stage_general_base_nll_sum"] = 0.0
            gen_fields["stage_general_wsp_nll_sum"] = 0.0
            gen_fields["stage_general_rel_drift"] = 0.0
            last_transition = f"promoted:{state.ramp_fraction}->{next_ramp}"
            history.append(
                {
                    "step": sprt_fields["total_samples"],
                    "event": "promote",
                    "from_ramp": state.ramp_fraction,
                    "to_ramp": next_ramp,
                    "sprt_log_lr": sprt_fields["sprt_log_lr"],
                    "stage_sprt_log_lr": sprt_fields["stage_sprt_log_lr"],
                    "cs_nll_diff_upper": sprt_fields["cs_nll_diff_upper"],
                }
            )
        else:
            next_stage_index = state.stage_index
            next_ramp = state.ramp_fraction
            next_stage_name = _stage_name_for_ramp(next_ramp, rolled_back=False)
            next_endpoint = _active_endpoint_for_ramp(next_ramp)
            next_promoted = state.promoted_to_live
            next_rollback = False
            next_stage_samples = sprt_fields["stage_samples"]
            next_stage_sprt = sprt_fields["stage_sprt_log_lr"]
            next_stage_gain_mean = sprt_fields["stage_gain_mean"]
            next_stage_gain_m2 = sprt_fields["stage_gain_m2"]
            next_live_cusum = sprt_fields["live_regression_cusum"]
            if state.stage_index == 0 and sprt_fields["stage_sprt_log_lr"] <= cfg.sprt_lower_bound:
                last_transition = "hold_shadow_h0"
            else:
                last_transition = "steady"

    lat_guard_ok = (
        not cfg.rollback_on_latency_breach
        or len(lat_fields["latency_window"]) < cfg.min_latency_p95_samples
        or lat_fields["latency_p95_ratio"] <= cfg.max_latency_p95_ratio
    )
    guard_ok = (
        not next_rollback
        and not ent_fields["entropy_collapse_detected"]
        and gen_fields["general_rel_drift"] <= cfg.max_general_drift_rel
        and lat_guard_ok
    )
    anse_energy = _compute_anse_energy(
        cfg,
        nll_diff_mean=sprt_fields["nll_diff_mean"],
        general_rel_drift=gen_fields["general_rel_drift"],
        guard_ok=guard_ok,
    )

    sprt_fields["stage_samples"] = next_stage_samples
    sprt_fields["stage_sprt_log_lr"] = next_stage_sprt
    sprt_fields["stage_gain_mean"] = next_stage_gain_mean
    sprt_fields["stage_gain_m2"] = next_stage_gain_m2
    sprt_fields["live_regression_cusum"] = next_live_cusum

    return ShadowCanaryState(
        config=cfg,
        base_checkpoint_id=state.base_checkpoint_id,
        wsp_checkpoint_id=state.wsp_checkpoint_id,
        canary_salt=state.canary_salt,
        stage_index=next_stage_index,
        ramp_fraction=next_ramp,
        stage_name=next_stage_name,
        active_endpoint=next_endpoint,
        promoted_to_live=next_promoted,
        rollback_triggered=next_rollback,
        rollback_reasons=combined_reasons,
        last_transition=last_transition,
        transition_history=tuple(history),
        **sprt_fields,
        **gen_fields,
        **lat_fields,
        **ent_fields,
        anse_energy=anse_energy,
        receipt_sha256="",
    )


# ---------------------------------------------------------------------------
# 4. Stateless Deterministic SHA-256 Request Bucketing & Audit Receipts
# ---------------------------------------------------------------------------


def route_canary_request_deterministic(
    request_key: str,
    ramp_fraction: float,
    *,
    salt: str = "",
    base_checkpoint_id: str = "base_pinned",
    wsp_checkpoint_id: str = "wsp_s5_candidate",
    state: ShadowCanaryState | None = None,
) -> CanaryRoutingDecision:
    """Deterministically bucket a request via SHA-256 with a canonical cryptographic receipt."""
    key_clean = _validate_nonempty_str(request_key, "request_key")
    ramp_f = _validate_unit_interval(ramp_fraction, "ramp_fraction")
    salt_clean = _validate_str_allow_empty(salt, "salt")
    base_id = _validate_nonempty_str(base_checkpoint_id, "base_checkpoint_id")
    wsp_id = _validate_nonempty_str(wsp_checkpoint_id, "wsp_checkpoint_id")

    if state is not None:
        if not isinstance(state, ShadowCanaryState):
            raise TypeError(f"state must be a ShadowCanaryState or None, got {type(state).__name__}")
        if state.rollback_triggered:
            ramp_f = 0.0
        elif abs(ramp_f - state.ramp_fraction) > 1e-12:
            raise ValueError(
                f"ramp_fraction ({ramp_f}) mismatches active state.ramp_fraction ({state.ramp_fraction})"
            )
        if salt_clean == "":
            salt_clean = state.canary_salt
        if base_checkpoint_id == "base_pinned":
            base_id = state.base_checkpoint_id
        if wsp_checkpoint_id == "wsp_s5_candidate":
            wsp_id = state.wsp_checkpoint_id

    bucket_digest = hashlib.sha256(
        canonical_json_bytes({"request_key": key_clean, "salt": salt_clean})
    ).hexdigest()
    bucket_uint64 = int(bucket_digest[:16], 16)
    bucket_unit = float(bucket_uint64) / _UINT64_SCALE
    bucket_bps = int(math.floor(bucket_unit * 10_000.0))

    if ramp_f <= 0.0:
        routed_to_wsp = False
    elif ramp_f >= 1.0:
        routed_to_wsp = True
    else:
        routed_to_wsp = bucket_unit < ramp_f

    selected_endpoint = "wsp" if routed_to_wsp else "base"
    selected_ckpt = wsp_id if routed_to_wsp else base_id

    core = {
        "object": ROUTING_RECEIPT_OBJECT,
        "schema": SCHEMA_VERSION,
        "request_key": key_clean,
        "salt": salt_clean,
        "ramp_fraction": ramp_f,
        "bucket_digest_sha256": bucket_digest,
        "bucket_unit": bucket_unit,
        "bucket_bps": bucket_bps,
        "routed_to_wsp": routed_to_wsp,
        "selected_endpoint": selected_endpoint,
        "selected_checkpoint_id": selected_ckpt,
        "base_checkpoint_id": base_id,
        "wsp_checkpoint_id": wsp_id,
    }
    receipt_sha = compute_payload_sha256(core)
    return CanaryRoutingDecision(
        request_key=key_clean,
        salt=salt_clean,
        ramp_fraction=ramp_f,
        bucket_digest_sha256=bucket_digest,
        bucket_unit=bucket_unit,
        bucket_bps=bucket_bps,
        routed_to_wsp=routed_to_wsp,
        selected_endpoint=selected_endpoint,
        selected_checkpoint_id=selected_ckpt,
        base_checkpoint_id=base_id,
        wsp_checkpoint_id=wsp_id,
        receipt_sha256=receipt_sha,
    )


def verify_shadow_canary_state_provenance(state: ShadowCanaryState | Mapping[str, Any]) -> bool:
    """Verify SHA-256 provenance and structural invariants of a ``ShadowCanaryState`` receipt."""
    if isinstance(state, ShadowCanaryState):
        data = state.to_dict()
    elif isinstance(state, Mapping):
        data = dict(state)
    else:
        raise TypeError(f"state must be a ShadowCanaryState or Mapping, got {type(state).__name__}")

    claimed = data.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    core = {k: v for k, v in data.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    if claimed != expected:
        return False

    if data.get("object") != STATE_RECEIPT_OBJECT or data.get("schema") != SCHEMA_VERSION:
        return False
    if bool(data.get("rollback_triggered")):
        if float(data.get("ramp_fraction", -1.0)) != 0.0 or data.get("stage_name") != "rolled_back":
            return False
    return True


def verify_routing_receipt_provenance(decision: CanaryRoutingDecision | Mapping[str, Any]) -> bool:
    """Verify SHA-256 provenance and deterministic bucketing consistency on a routing receipt."""
    if isinstance(decision, CanaryRoutingDecision):
        data = decision.to_dict()
    elif isinstance(decision, Mapping):
        data = dict(decision)
    else:
        raise TypeError(f"decision must be a CanaryRoutingDecision or Mapping, got {type(decision).__name__}")

    claimed = data.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    core = {k: v for k, v in data.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    if claimed != expected:
        return False

    try:
        recomputed = route_canary_request_deterministic(
            str(data["request_key"]),
            float(data["ramp_fraction"]),
            salt=str(data["salt"]),
            base_checkpoint_id=str(data["base_checkpoint_id"]),
            wsp_checkpoint_id=str(data["wsp_checkpoint_id"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return recomputed.to_dict() == data


def build_canary_rollout_summary(state: ShadowCanaryState) -> dict[str, Any]:
    """Build a concise JSON-safe rollout summary preserving canonical telemetry field names."""
    if not isinstance(state, ShadowCanaryState):
        raise TypeError(f"state must be a ShadowCanaryState, got {type(state).__name__}")
    return {
        "object": STATE_RECEIPT_OBJECT,
        "schema": SCHEMA_VERSION,
        "base_checkpoint_id": state.base_checkpoint_id,
        "wsp_checkpoint_id": state.wsp_checkpoint_id,
        "stage_index": state.stage_index,
        "stage_name": state.stage_name,
        "ramp_fraction": state.ramp_fraction,
        "active_endpoint": state.active_endpoint,
        "promoted_to_live": state.promoted_to_live,
        "rollback_triggered": state.rollback_triggered,
        "rollback_reasons": list(state.rollback_reasons),
        "last_transition": state.last_transition,
        "total_samples": state.total_samples,
        "nll_diff_mean": state.nll_diff_mean,
        "sprt_log_lr": state.sprt_log_lr,
        "stage_sprt_log_lr": state.stage_sprt_log_lr,
        "cs_nll_diff_lower": state.cs_nll_diff_lower,
        "cs_nll_diff_upper": state.cs_nll_diff_upper,
        "general_samples": state.general_samples,
        "general_rel_drift": state.general_rel_drift,
        "general_guard_violations": state.general_guard_violations,
        "latency_p95_ratio": state.latency_p95_ratio,
        "entropy_collapse_count": state.entropy_collapse_count,
        "entropy_collapse_detected": state.entropy_collapse_detected,
        "anse_energy": state.anse_energy,
        "receipt_sha256": state.receipt_sha256,
    }


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_ALT_GAIN_NATS",
    "DEFAULT_ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANSE_SOFT_MARGIN_PENALTY",
    "DEFAULT_BETA",
    "DEFAULT_CS_PRIOR_RHO",
    "DEFAULT_GENERAL_GUARD_WINDOW",
    "DEFAULT_HARD_LATENCY_SPIKE_RATIO",
    "DEFAULT_LATENCY_WINDOW_SIZE",
    "DEFAULT_MAX_GENERAL_DRIFT_REL",
    "DEFAULT_MAX_LATENCY_P95_RATIO",
    "DEFAULT_MAX_REPETITION_RATE",
    "DEFAULT_MIN_GENERAL_SAMPLES_PER_STAGE",
    "DEFAULT_MIN_LATENCY_P95_SAMPLES",
    "DEFAULT_MIN_SAMPLES_PER_STAGE",
    "DEFAULT_MIN_TOKEN_ENTROPY_NATS",
    "DEFAULT_NULL_GAIN_NATS",
    "DEFAULT_RAMP_STAGES",
    "DEFAULT_SIGMA_FLOOR_NATS",
    "DEFAULT_SINGLE_STEP_GENERAL_HARD_CAP_REL",
    "DEFAULT_SOFT_GENERAL_MARGIN_REL",
    "ROUTING_RECEIPT_OBJECT",
    "SCHEMA_VERSION",
    "STATE_RECEIPT_OBJECT",
    "CanaryRoutingDecision",
    "ShadowCanaryConfig",
    "ShadowCanaryState",
    "build_canary_rollout_summary",
    "canonical_json_bytes",
    "compute_payload_sha256",
    "create_shadow_canary_state",
    "route_canary_request_deterministic",
    "update_sprt_canary_step",
    "verify_routing_receipt_provenance",
    "verify_shadow_canary_state_provenance",
]

# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Frozen Latent JEPA World-Model Critic for Reine-Cortex Operculation and Pheromones Royales.

Implements Paper v4.4.0 Section 15.4 Opportunity J3 (``JEPA-WSP Critic``, DOI
``10.5281/zenodo.24553269``) strictly on CPU (``0 MiB`` GPU VRAM, ``0`` backpropagation
steps, ``0`` network or subprocess calls):

1. **Frozen Latent JEPA Predictor ``f_phi(z_ctx) -> z_hat_target``:**
   Provides deterministic CPU ``float64`` linear and two-layer MLP latent predictors
   (including closed-form zero-backprop ridge calibration via ``torch.linalg.solve``)
   whose parameters remain frozen with ``requires_grad=False``.
2. **Semantic Surprise & RankMe Isotropy Guard:**
   Computes stop-gradient semantic prediction surprise
   ``E_JEPA = ||f_phi(z_ctx) - sg(z_target)||_2^2`` alongside spectral ``RankMe``
   effective-rank and cosine-isotropy diagnostics matching ``representation_collapse_wsp_guard``.
3. **Opening-Template Cross-Entropy Gaming Rejection (Paper Section 14):**
   Detects superficial opening-template CE gaming (reproducing the Section 14 ``81%``
   full-chunk vs mid-row format artifact, stagnant/rising ``E_JEPA``, or RankMe
   collapse) inside CPU ``Operculation d'Alveole`` and enforces ``ANSE_MAXIMUM_PAIN``
   (``1e6``) atomic bitwise rollback.
4. **Semantic-Deficit Pheromones Royales & Simplex Conservation:**
   Converts per-domain semantic surprise and mid-row conceptual deficits into
   simplex-normalized foraging weights ``p_d`` (``sum(p_d) = 1.0``, strict order
   preservation) packaged inside a signed ``QueenPheromoneDirective``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hivemind.jepa.nullspace_jepa_rescuer import (
    ANSE_MAXIMUM_PAIN,
    NullspaceJepaRescueConfig,
    RankMeIsotropyReport,
    compute_payload_sha256,
    compute_state_sha256,
    measure_rankme_and_cosine_isotropy,
)

JEPA_WSP_CRITIC_SCHEMA: str = "hivemind.jepa.latent_jepa_wsp_critic/1"
JEPA_WSP_RECEIPT_OBJECT: str = "hivemind.jepa.jepa_wsp_operculation_receipt"

QUEEN_CORTEX_SCHEMA: str = "hivemind.queen_cortex_moe_campaign_governor/1"
QUEEN_PHEROMONE_OBJECT: str = "hivemind.queen_pheromone_directive"

VALID_PREDICTOR_MODES: tuple[str, ...] = ("linear", "mlp")
VALID_MLP_ACTIVATIONS: tuple[str, ...] = ("gelu", "silu", "tanh")

DEFAULT_GENERAL_DOMAIN: str = "D0_general_prose"
DEFAULT_GENERAL_PREFIXES: tuple[str, ...] = ("E0_", "D0_", "general")
DEFAULT_MAX_GENERAL_DRIFT_RATIO: float = 0.010
DEFAULT_MAX_DOMAIN_DRIFT_RATIO: float = 0.010
DEFAULT_JEPA_SURPRISE_WEIGHT: float = 0.25
DEFAULT_ECE_WEIGHT: float = 0.25
DEFAULT_MAX_FORMAT_ARTIFACT_SHARE: float = 0.50
DEFAULT_MIN_SURFACE_GAIN_FOR_GAMING_CHECK: float = 0.05
DEFAULT_MAX_ALLOWED_SURPRISE_REGRESSION: float = 0.0
DEFAULT_MIN_SURPRISE_IMPROVEMENT: float = 1e-4
DEFAULT_MIN_ENERGY_IMPROVEMENT: float = 1e-6
DEFAULT_FORAGING_TEMPERATURE: float = 0.25
DEFAULT_UCB_EXPLORATION_COEFF: float = 0.05
DEFAULT_ANCHOR_GAIN_FLOOR_NATS: float = -0.01
DEFAULT_PROSE_HARM_SINK_FLOOR_NATS: float = 0.0015
DEFAULT_COADAPT_KL_CEILING: float = 0.05
DEFAULT_MIN_NORMALIZED_RANKME: float = 0.15
DEFAULT_MIN_RANKME_RATIO: float = 0.85
DEFAULT_MAX_ANISOTROPY_DELTA: float = 0.05
DEFAULT_MAX_ANISOTROPY_ABS: float = 0.85
DEFAULT_EPS: float = 1e-8

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# 1. Validation, Configuration & Signed Pheromone Directive
# ---------------------------------------------------------------------------


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_int(value: Any, name: str, *, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return int(value)


def _require_finite_float(
    value: Any,
    name: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    strict_min: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if min_value is not None:
        if (strict_min and out <= min_value) or (not strict_min and out < min_value):
            op = ">" if strict_min else ">="
            raise ValueError(f"{name} must be {op} {min_value}, got {out}")
    if max_value is not None and out > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {out}")
    return out


def _require_nonempty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_fp_tensor(tensor: Any, name: str, *, ndim: int | None = None) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating-point, got {tensor.dtype}")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {tuple(tensor.shape)}")
    cpu_t = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(cpu_t).all().item()):
        raise ValueError(f"{name} contains non-finite values")
    return cpu_t


def _is_general_key(domain_name: str, cfg: JepaWspCriticConfig) -> bool:
    low = domain_name.strip().lower()
    return domain_name == cfg.general_domain_name or any(low.startswith(p.lower()) for p in cfg.general_prefixes)


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


def verify_pheromone_directive_signature(directive: QueenPheromoneDirective | Mapping[str, Any]) -> bool:
    """Verify canonical SHA-256 signature on a ``QueenPheromoneDirective`` or dict."""
    if isinstance(directive, QueenPheromoneDirective):
        payload: Mapping[str, Any] = directive.to_dict()
    elif isinstance(directive, Mapping):
        payload = directive
    else:
        raise TypeError(f"directive must be QueenPheromoneDirective or Mapping, got {type(directive).__name__}")
    claimed = payload.get("directive_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if payload.get("object") != QUEEN_PHEROMONE_OBJECT or payload.get("schema") != QUEEN_CORTEX_SCHEMA:
        return False
    core = {k: v for k, v in payload.items() if k != "directive_sha256"}
    try:
        return claimed == compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class JepaWspCriticConfig:
    """Validated configuration governing the frozen latent JEPA-WSP critic and Pheromones Royales."""

    general_domain_name: str = DEFAULT_GENERAL_DOMAIN
    general_prefixes: tuple[str, ...] = DEFAULT_GENERAL_PREFIXES
    max_general_drift_ratio: float = DEFAULT_MAX_GENERAL_DRIFT_RATIO
    max_domain_drift_ratio: float = DEFAULT_MAX_DOMAIN_DRIFT_RATIO
    enforce_all_domain_guard: bool = False
    jepa_surprise_weight: float = DEFAULT_JEPA_SURPRISE_WEIGHT
    ece_weight: float = DEFAULT_ECE_WEIGHT
    max_format_artifact_share: float = DEFAULT_MAX_FORMAT_ARTIFACT_SHARE
    min_surface_gain_for_gaming_check: float = DEFAULT_MIN_SURFACE_GAIN_FOR_GAMING_CHECK
    max_allowed_surprise_regression: float = DEFAULT_MAX_ALLOWED_SURPRISE_REGRESSION
    require_surprise_reduction_on_ce_gain: bool = True
    min_surprise_improvement: float = DEFAULT_MIN_SURPRISE_IMPROVEMENT
    min_energy_improvement: float = DEFAULT_MIN_ENERGY_IMPROVEMENT
    foraging_temperature: float = DEFAULT_FORAGING_TEMPERATURE
    ucb_exploration_coeff: float = DEFAULT_UCB_EXPLORATION_COEFF
    pheromone_jepa_weight: float = 1.0
    pheromone_midrow_nll_weight: float = 0.5
    pheromone_ece_weight: float = 0.25
    pheromone_surface_nll_weight: float = 0.0
    min_normalized_rankme: float = DEFAULT_MIN_NORMALIZED_RANKME
    min_rankme_ratio: float = DEFAULT_MIN_RANKME_RATIO
    max_anisotropy_delta: float = DEFAULT_MAX_ANISOTROPY_DELTA
    max_anisotropy_abs: float = DEFAULT_MAX_ANISOTROPY_ABS
    anchor_gain_floor_nats: float = DEFAULT_ANCHOR_GAIN_FLOOR_NATS
    prose_harm_sink_floor_nats: float = DEFAULT_PROSE_HARM_SINK_FLOOR_NATS
    coadapt_kl_ceiling: float = DEFAULT_COADAPT_KL_CEILING
    eps: float = DEFAULT_EPS

    def __post_init__(self) -> None:
        _require_nonempty_str(self.general_domain_name, "general_domain_name")
        if (
            isinstance(self.general_prefixes, (str, bytes))
            or not isinstance(self.general_prefixes, Sequence)
            or not self.general_prefixes
        ):
            raise ValueError("general_prefixes must be a non-empty sequence of strings")
        cleaned = tuple(_require_nonempty_str(p, f"general_prefixes[{i}]") for i, p in enumerate(self.general_prefixes))
        object.__setattr__(self, "general_prefixes", cleaned)

        _require_finite_float(self.max_general_drift_ratio, "max_general_drift_ratio", min_value=0.0)
        _require_finite_float(self.max_domain_drift_ratio, "max_domain_drift_ratio", min_value=0.0)
        _require_bool(self.enforce_all_domain_guard, "enforce_all_domain_guard")
        _require_finite_float(self.jepa_surprise_weight, "jepa_surprise_weight", min_value=0.0)
        _require_finite_float(self.ece_weight, "ece_weight", min_value=0.0)
        _require_finite_float(
            self.max_format_artifact_share, "max_format_artifact_share", min_value=0.0, max_value=1.0
        )
        _require_finite_float(
            self.min_surface_gain_for_gaming_check, "min_surface_gain_for_gaming_check", min_value=0.0
        )
        _require_finite_float(self.max_allowed_surprise_regression, "max_allowed_surprise_regression", min_value=0.0)
        _require_bool(self.require_surprise_reduction_on_ce_gain, "require_surprise_reduction_on_ce_gain")
        _require_finite_float(self.min_surprise_improvement, "min_surprise_improvement", min_value=0.0)
        _require_finite_float(self.min_energy_improvement, "min_energy_improvement", min_value=0.0)
        _require_finite_float(self.foraging_temperature, "foraging_temperature", min_value=0.0, strict_min=True)
        _require_finite_float(self.ucb_exploration_coeff, "ucb_exploration_coeff", min_value=0.0)
        _require_finite_float(self.pheromone_jepa_weight, "pheromone_jepa_weight", min_value=0.0)
        _require_finite_float(self.pheromone_midrow_nll_weight, "pheromone_midrow_nll_weight", min_value=0.0)
        _require_finite_float(self.pheromone_ece_weight, "pheromone_ece_weight", min_value=0.0)
        _require_finite_float(self.pheromone_surface_nll_weight, "pheromone_surface_nll_weight", min_value=0.0)
        _require_finite_float(
            self.min_normalized_rankme, "min_normalized_rankme", min_value=0.0, max_value=1.0, strict_min=True
        )
        _require_finite_float(
            self.min_rankme_ratio, "min_rankme_ratio", min_value=0.0, max_value=1.0, strict_min=True
        )
        _require_finite_float(self.max_anisotropy_delta, "max_anisotropy_delta", min_value=0.0)
        _require_finite_float(
            self.max_anisotropy_abs, "max_anisotropy_abs", min_value=0.0, max_value=1.0, strict_min=True
        )
        _require_finite_float(self.anchor_gain_floor_nats, "anchor_gain_floor_nats", max_value=0.0)
        _require_finite_float(self.prose_harm_sink_floor_nats, "prose_harm_sink_floor_nats", min_value=0.0)
        _require_finite_float(self.coadapt_kl_ceiling, "coadapt_kl_ceiling", min_value=0.0)
        _require_finite_float(self.eps, "eps", min_value=0.0, strict_min=True)


# ---------------------------------------------------------------------------
# 2. Frozen Latent JEPA Predictor f_phi(z_ctx) -> z_hat_target (0 Backprop)
# ---------------------------------------------------------------------------


def _apply_mlp_activation(tensor: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "gelu":
        return F.gelu(tensor)
    if activation == "silu":
        return F.silu(tensor)
    return torch.tanh(tensor)


@dataclass(frozen=True)
class FrozenLatentJepaPredictor:
    """Frozen CPU ``float64`` linear or two-layer MLP latent world-model predictor ``f_phi``."""

    mode: str
    w1: torch.Tensor
    b1: torch.Tensor | None = None
    w2: torch.Tensor | None = None
    b2: torch.Tensor | None = None
    activation: str = "gelu"
    residual: bool = False
    predictor_sha256: str = ""

    def __post_init__(self) -> None:
        if self.mode not in VALID_PREDICTOR_MODES:
            raise ValueError(f"mode must be one of {VALID_PREDICTOR_MODES}, got {self.mode!r}")
        if self.activation not in VALID_MLP_ACTIVATIONS:
            raise ValueError(f"activation must be one of {VALID_MLP_ACTIVATIONS}, got {self.activation!r}")
        _require_bool(self.residual, "residual")

        w1_c = _require_fp_tensor(self.w1, "w1", ndim=2)
        w1_c.requires_grad_(False)
        object.__setattr__(self, "w1", w1_c)
        hidden_or_out, ctx_dim = int(w1_c.shape[0]), int(w1_c.shape[1])

        b1_c: torch.Tensor | None = None
        if self.b1 is not None:
            b1_c = _require_fp_tensor(self.b1, "b1", ndim=1)
            if int(b1_c.shape[0]) != hidden_or_out:
                raise ValueError(f"b1 length {int(b1_c.shape[0])} != w1 out dim {hidden_or_out}")
            b1_c.requires_grad_(False)
        object.__setattr__(self, "b1", b1_c)

        state_map: dict[str, torch.Tensor] = {"w1": w1_c}
        if b1_c is not None:
            state_map["b1"] = b1_c

        if self.mode == "linear":
            if self.w2 is not None or self.b2 is not None:
                raise ValueError("linear predictor must not provide w2 or b2")
            target_dim = hidden_or_out
        else:
            if self.w2 is None:
                raise ValueError("mlp predictor requires w2")
            w2_c = _require_fp_tensor(self.w2, "w2", ndim=2)
            if int(w2_c.shape[1]) != hidden_or_out:
                raise ValueError(f"w2 width {int(w2_c.shape[1])} != w1 out dim {hidden_or_out}")
            w2_c.requires_grad_(False)
            object.__setattr__(self, "w2", w2_c)
            target_dim = int(w2_c.shape[0])
            state_map["w2"] = w2_c

            b2_c: torch.Tensor | None = None
            if self.b2 is not None:
                b2_c = _require_fp_tensor(self.b2, "b2", ndim=1)
                if int(b2_c.shape[0]) != target_dim:
                    raise ValueError(f"b2 length {int(b2_c.shape[0])} != w2 out dim {target_dim}")
                b2_c.requires_grad_(False)
            object.__setattr__(self, "b2", b2_c)
            if b2_c is not None:
                state_map["b2"] = b2_c

        if self.residual and ctx_dim != target_dim:
            raise ValueError(f"residual=True requires ctx_dim ({ctx_dim}) == target_dim ({target_dim})")

        digest = hashlib.sha256()
        digest.update(f"{self.mode}:{self.activation}:{self.residual}:".encode())
        digest.update(compute_state_sha256(state_map).encode("ascii"))
        object.__setattr__(self, "predictor_sha256", digest.hexdigest())

    @property
    def ctx_dim(self) -> int:
        return int(self.w1.shape[1])

    @property
    def target_dim(self) -> int:
        return int(self.w2.shape[0]) if self.w2 is not None else int(self.w1.shape[0])

    @property
    def is_frozen(self) -> bool:
        tensors = [t for t in (self.w1, self.b1, self.w2, self.b2) if t is not None]
        return all((not t.requires_grad) and t.device.type == "cpu" for t in tensors)

    @torch.no_grad()
    def predict(self, z_ctx: torch.Tensor) -> torch.Tensor:
        """Evaluate frozen ``f_phi(z_ctx) -> z_hat_target`` on CPU with zero autograd graph."""
        x = _require_fp_tensor(z_ctx, "z_ctx", ndim=2)
        if int(x.shape[1]) != self.ctx_dim:
            raise ValueError(f"z_ctx width {int(x.shape[1])} != predictor ctx_dim {self.ctx_dim}")
        hidden = x @ self.w1.T
        if self.b1 is not None:
            hidden = hidden + self.b1.unsqueeze(0)
        if self.mode == "mlp":
            assert self.w2 is not None
            act = _apply_mlp_activation(hidden, self.activation)
            out = act @ self.w2.T
            if self.b2 is not None:
                out = out + self.b2.unsqueeze(0)
        else:
            out = hidden
        if self.residual:
            out = out + x
        return out.detach()


def build_frozen_jepa_predictor(
    ctx_dim: int,
    target_dim: int | None = None,
    *,
    mode: str = "linear",
    hidden_dim: int | None = None,
    activation: str = "gelu",
    residual: bool = False,
    seed: int = 2026,
    w1: torch.Tensor | None = None,
    b1: torch.Tensor | None = None,
    w2: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
) -> FrozenLatentJepaPredictor:
    """Construct a deterministic frozen CPU linear or two-layer MLP JEPA predictor."""
    d_in = _require_int(ctx_dim, "ctx_dim", min_value=1)
    d_out = d_in if target_dim is None else _require_int(target_dim, "target_dim", min_value=1)
    if mode not in VALID_PREDICTOR_MODES:
        raise ValueError(f"mode must be one of {VALID_PREDICTOR_MODES}, got {mode!r}")
    if activation not in VALID_MLP_ACTIVATIONS:
        raise ValueError(f"activation must be one of {VALID_MLP_ACTIVATIONS}, got {activation!r}")
    seed_i = _require_int(seed, "seed", min_value=0)

    if w1 is not None:
        return FrozenLatentJepaPredictor(
            mode=mode,
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            activation=activation,
            residual=residual,
        )

    gen = torch.Generator(device="cpu").manual_seed(seed_i)
    if mode == "linear":
        w1_init = torch.randn(d_out, d_in, generator=gen, dtype=torch.float64) / math.sqrt(float(d_in))
        b1_init = torch.zeros(d_out, dtype=torch.float64)
        return FrozenLatentJepaPredictor(
            mode="linear",
            w1=w1_init,
            b1=b1_init,
            activation=activation,
            residual=residual,
        )

    d_hid = max(d_in, d_out) * 2 if hidden_dim is None else _require_int(hidden_dim, "hidden_dim", min_value=1)
    w1_init = torch.randn(d_hid, d_in, generator=gen, dtype=torch.float64) / math.sqrt(float(d_in))
    b1_init = torch.zeros(d_hid, dtype=torch.float64)
    w2_init = torch.randn(d_out, d_hid, generator=gen, dtype=torch.float64) / math.sqrt(float(d_hid))
    b2_init = torch.zeros(d_out, dtype=torch.float64)
    return FrozenLatentJepaPredictor(
        mode="mlp",
        w1=w1_init,
        b1=b1_init,
        w2=w2_init,
        b2=b2_init,
        activation=activation,
        residual=residual,
    )


@torch.no_grad()
def fit_frozen_jepa_predictor_closed_form(
    z_ctx: torch.Tensor,
    z_target: torch.Tensor,
    *,
    mode: str = "linear",
    hidden_dim: int | None = None,
    ridge_lambda: float = 1e-3,
    activation: str = "gelu",
    residual: bool = False,
    seed: int = 2026,
) -> FrozenLatentJepaPredictor:
    """Fit a frozen CPU JEPA predictor in closed form via ridge regression (0 backprop)."""
    x = _require_fp_tensor(z_ctx, "z_ctx", ndim=2)
    y = _require_fp_tensor(z_target, "z_target", ndim=2).detach()
    if int(x.shape[0]) != int(y.shape[0]):
        raise ValueError(f"z_ctx rows {int(x.shape[0])} != z_target rows {int(y.shape[0])}")
    if mode not in VALID_PREDICTOR_MODES:
        raise ValueError(f"mode must be one of {VALID_PREDICTOR_MODES}, got {mode!r}")
    if activation not in VALID_MLP_ACTIVATIONS:
        raise ValueError(f"activation must be one of {VALID_MLP_ACTIVATIONS}, got {activation!r}")
    lam = _require_finite_float(ridge_lambda, "ridge_lambda", min_value=0.0, strict_min=True)
    _require_bool(residual, "residual")
    seed_i = _require_int(seed, "seed", min_value=0)

    d_in, d_out = int(x.shape[1]), int(y.shape[1])
    if residual and d_in != d_out:
        raise ValueError(f"residual=True requires ctx_dim ({d_in}) == target_dim ({d_out})")
    y_fit = (y - x) if residual else y

    def _solve_ridge_affine(feat: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_rows, feat_dim = int(feat.shape[0]), int(feat.shape[1])
        design = torch.cat([feat, torch.ones((n_rows, 1), dtype=torch.float64)], dim=1)
        reg = torch.eye(feat_dim + 1, dtype=torch.float64) * lam
        reg[-1, -1] = 1e-8
        sol = torch.linalg.solve(design.T @ design + reg, design.T @ target)
        return sol[:-1, :].T.contiguous(), sol[-1, :].contiguous()

    if mode == "linear":
        w1_fit, b1_fit = _solve_ridge_affine(x, y_fit)
        return FrozenLatentJepaPredictor(
            mode="linear", w1=w1_fit, b1=b1_fit, activation=activation, residual=residual
        )

    d_hid = max(d_in, d_out) * 2 if hidden_dim is None else _require_int(hidden_dim, "hidden_dim", min_value=1)
    gen = torch.Generator(device="cpu").manual_seed(seed_i)
    w1_rand = torch.randn(d_hid, d_in, generator=gen, dtype=torch.float64) / math.sqrt(float(d_in))
    b1_rand = 0.1 * torch.randn(d_hid, generator=gen, dtype=torch.float64)
    h_act = _apply_mlp_activation(x @ w1_rand.T + b1_rand.unsqueeze(0), activation)
    w2_fit, b2_fit = _solve_ridge_affine(h_act, y_fit)
    return FrozenLatentJepaPredictor(
        mode="mlp",
        w1=w1_rand,
        b1=b1_rand,
        w2=w2_fit,
        b2=b2_fit,
        activation=activation,
        residual=residual,
    )


# ---------------------------------------------------------------------------
# 3. Stop-Gradient Semantic Surprise E_JEPA & RankMe Isotropy Evaluation
# ---------------------------------------------------------------------------


def _evaluate_rankme_isotropy(
    states: torch.Tensor,
    cfg: JepaWspCriticConfig,
    reference_report: RankMeIsotropyReport | None = None,
) -> tuple[RankMeIsotropyReport, dict[str, Any]]:
    """Measure RankMe effective rank and cosine anisotropy without heavy external imports."""
    rescue_cfg = NullspaceJepaRescueConfig(
        min_one_minus_cos=max(1e-6, 1.0 - cfg.max_anisotropy_abs),
        min_normalized_rankme=cfg.min_normalized_rankme,
        min_rankme_ratio=cfg.min_rankme_ratio,
    )
    rep = measure_rankme_and_cosine_isotropy(states, reference_report=reference_report, config=rescue_cfg)
    reasons = list(rep.collapse_reasons)
    aniso_delta: float | None = None
    if reference_report is not None:
        aniso_delta = float(rep.mean_pairwise_cosine - reference_report.mean_pairwise_cosine)
        if aniso_delta > cfg.max_anisotropy_delta and "anisotropy_spike_vs_reference" not in reasons:
            reasons.append("anisotropy_spike_vs_reference")

    metrics = {
        "n_tokens": rep.n_tokens,
        "hidden_dim": rep.hidden_dim,
        "rankme": float(rep.rankme),
        "normalized_rankme": float(rep.normalized_rankme),
        "centered_rankme": float(rep.centered_rankme),
        "participation_ratio": float(rep.participation_ratio),
        "anisotropy": float(rep.mean_pairwise_cosine),
        "isotropy_index": float(max(0.0, min(1.0, 1.0 - max(0.0, rep.mean_pairwise_cosine)))),
        "one_minus_cosine": float(rep.one_minus_cosine),
        "rankme_ratio_vs_ref": rep.rankme_ratio_vs_ref,
        "anisotropy_delta_vs_ref": aniso_delta,
        "collapse_warnings": reasons,
        "is_collapsed": len(reasons) > 0,
    }
    return rep, metrics


@dataclass(frozen=True)
class LatentSurpriseReport:
    """Semantic prediction surprise ``E_JEPA`` and RankMe isotropy report."""

    n_samples: int
    latent_dim: int
    mean_l2_sq_surprise: float
    rms_l2_surprise: float
    relative_l2_sq_surprise: float
    mean_cosine_similarity: float
    one_minus_cosine: float
    pred_rankme_report: RankMeIsotropyReport
    target_rankme_report: RankMeIsotropyReport
    pred_rankme_metrics: dict[str, Any]
    target_rankme_metrics: dict[str, Any]
    per_sample_surprise: tuple[float, ...]
    stopgrad_enforced: bool
    predictor_sha256: str

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "latent_dim": self.latent_dim,
            "mean_l2_sq_surprise": self.mean_l2_sq_surprise,
            "rms_l2_surprise": self.rms_l2_surprise,
            "relative_l2_sq_surprise": self.relative_l2_sq_surprise,
            "mean_cosine_similarity": self.mean_cosine_similarity,
            "one_minus_cosine": self.one_minus_cosine,
            "pred_rankme": float(self.pred_rankme_metrics["rankme"]),
            "pred_normalized_rankme": float(self.pred_rankme_metrics["normalized_rankme"]),
            "pred_anisotropy": float(self.pred_rankme_metrics["anisotropy"]),
            "pred_collapsed": bool(self.pred_rankme_metrics["is_collapsed"]),
            "target_rankme": float(self.target_rankme_metrics["rankme"]),
            "target_normalized_rankme": float(self.target_rankme_metrics["normalized_rankme"]),
            "target_anisotropy": float(self.target_rankme_metrics["anisotropy"]),
            "target_collapsed": bool(self.target_rankme_metrics["is_collapsed"]),
            "stopgrad_enforced": self.stopgrad_enforced,
            "predictor_sha256": self.predictor_sha256,
        }


@torch.no_grad()
def compute_latent_jepa_surprise(
    predictor: FrozenLatentJepaPredictor,
    z_ctx: torch.Tensor,
    z_target: torch.Tensor,
    *,
    config: JepaWspCriticConfig | None = None,
    reference_target_report: RankMeIsotropyReport | None = None,
    reference_pred_report: RankMeIsotropyReport | None = None,
) -> LatentSurpriseReport:
    """Compute ``E_JEPA = ||f_phi(z_ctx) - sg(z_target)||_2^2`` and RankMe isotropy on CPU."""
    if not isinstance(predictor, FrozenLatentJepaPredictor):
        raise TypeError(f"predictor must be FrozenLatentJepaPredictor, got {type(predictor).__name__}")
    cfg = config if config is not None else JepaWspCriticConfig()

    z_tgt_sg = _require_fp_tensor(z_target, "z_target", ndim=2).detach()
    z_hat = predictor.predict(z_ctx).detach()
    if z_hat.shape != z_tgt_sg.shape:
        raise ValueError(
            f"Predicted latent shape {tuple(z_hat.shape)} != target latent shape {tuple(z_tgt_sg.shape)}"
        )

    diff = z_hat - z_tgt_sg
    per_sample = diff.pow(2).sum(dim=-1)
    mean_sq = float(per_sample.mean().item())
    rms_l2 = float(math.sqrt(max(mean_sq, 0.0)))
    tgt_energy = float(z_tgt_sg.pow(2).sum(dim=-1).mean().item())
    rel_sq = float(mean_sq / max(tgt_energy, cfg.eps))

    cos_vals = F.cosine_similarity(z_hat, z_tgt_sg, dim=-1)
    mean_cos = max(-1.0, min(1.0, float(cos_vals.mean().item())))
    one_minus_cos = float(1.0 - mean_cos)

    pred_rep, pred_rm = _evaluate_rankme_isotropy(z_hat, cfg, reference_pred_report)
    tgt_rep, tgt_rm = _evaluate_rankme_isotropy(z_tgt_sg, cfg, reference_target_report)

    return LatentSurpriseReport(
        n_samples=int(z_hat.shape[0]),
        latent_dim=int(z_hat.shape[1]),
        mean_l2_sq_surprise=mean_sq,
        rms_l2_surprise=rms_l2,
        relative_l2_sq_surprise=rel_sq,
        mean_cosine_similarity=mean_cos,
        one_minus_cosine=one_minus_cos,
        pred_rankme_report=pred_rep,
        target_rankme_report=tgt_rep,
        pred_rankme_metrics=pred_rm,
        target_rankme_metrics=tgt_rm,
        per_sample_surprise=tuple(float(v) for v in per_sample.tolist()),
        stopgrad_enforced=(not z_tgt_sg.requires_grad) and (not z_hat.requires_grad),
        predictor_sha256=predictor.predictor_sha256,
    )


# ---------------------------------------------------------------------------
# 4. Opening-Template Cross-Entropy Gaming Detector (Paper §14 81% Artifact)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateGamingDiagnosis:
    """Diagnosis of superficial opening-template CE gaming vs genuine conceptual progress."""

    surface_nll_gain: float
    midrow_nll_gain: float | None
    format_artifact_share: float | None
    base_e_jepa: float
    cand_e_jepa: float
    delta_e_jepa: float
    target_collapsed: bool
    pred_collapsed: bool
    template_gaming_detected: bool
    gaming_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_nll_gain": self.surface_nll_gain,
            "midrow_nll_gain": self.midrow_nll_gain,
            "format_artifact_share": self.format_artifact_share,
            "base_e_jepa": self.base_e_jepa,
            "cand_e_jepa": self.cand_e_jepa,
            "delta_e_jepa": self.delta_e_jepa,
            "target_collapsed": self.target_collapsed,
            "pred_collapsed": self.pred_collapsed,
            "template_gaming_detected": self.template_gaming_detected,
            "gaming_reasons": list(self.gaming_reasons),
        }


@torch.no_grad()
def summarize_opening_vs_midrow_nll(
    base_token_nll: torch.Tensor,
    cand_token_nll: torch.Tensor,
    *,
    prefix_len: int = 64,
) -> dict[str, float]:
    """Summarize full-row, opening-prefix, and mid-row NLL gains across token-loss matrices."""
    base_t = _require_fp_tensor(base_token_nll, "base_token_nll", ndim=2)
    cand_t = _require_fp_tensor(cand_token_nll, "cand_token_nll", ndim=2)
    if base_t.shape != cand_t.shape:
        raise ValueError(f"base_token_nll shape {tuple(base_t.shape)} != cand_token_nll shape {tuple(cand_t.shape)}")
    pref = _require_int(prefix_len, "prefix_len", min_value=1)
    seq_len = int(base_t.shape[1])
    if pref >= seq_len:
        raise ValueError(f"prefix_len ({pref}) must be < sequence length ({seq_len}) to score mid-row tokens")

    base_full = float(base_t.mean().item())
    cand_full = float(cand_t.mean().item())
    base_open = float(base_t[:, :pref].mean().item())
    cand_open = float(cand_t[:, :pref].mean().item())
    base_mid = float(base_t[:, pref:].mean().item())
    cand_mid = float(cand_t[:, pref:].mean().item())

    full_gain = base_full - cand_full
    mid_gain = base_mid - cand_mid
    format_share = (
        max(0.0, min(1.0, (full_gain - max(0.0, mid_gain)) / full_gain)) if full_gain > 1e-12 else 0.0
    )
    return {
        "base_full_nll": base_full,
        "cand_full_nll": cand_full,
        "base_opening_nll": base_open,
        "cand_opening_nll": cand_open,
        "base_midrow_nll": base_mid,
        "cand_midrow_nll": cand_mid,
        "full_nll_gain": full_gain,
        "midrow_nll_gain": mid_gain,
        "format_artifact_share": format_share,
    }


def diagnose_template_gaming(
    *,
    base_surface_nll: float,
    cand_surface_nll: float,
    base_surprise: LatentSurpriseReport,
    cand_surprise: LatentSurpriseReport,
    base_midrow_nll: float | None = None,
    cand_midrow_nll: float | None = None,
    config: JepaWspCriticConfig | None = None,
) -> TemplateGamingDiagnosis:
    """Detect superficial opening-template CE gaming via format-share, ``E_JEPA``, and RankMe."""
    cfg = config if config is not None else JepaWspCriticConfig()
    b_surf = _require_finite_float(base_surface_nll, "base_surface_nll", min_value=0.0)
    c_surf = _require_finite_float(cand_surface_nll, "cand_surface_nll", min_value=0.0)
    if (base_midrow_nll is None) != (cand_midrow_nll is None):
        raise ValueError("base_midrow_nll and cand_midrow_nll must both be provided or both omitted")

    surface_gain = b_surf - c_surf
    midrow_gain: float | None = None
    format_share: float | None = None
    if base_midrow_nll is not None and cand_midrow_nll is not None:
        b_mid = _require_finite_float(base_midrow_nll, "base_midrow_nll", min_value=0.0)
        c_mid = _require_finite_float(cand_midrow_nll, "cand_midrow_nll", min_value=0.0)
        midrow_gain = b_mid - c_mid
        format_share = (
            max(0.0, min(1.0, (surface_gain - max(0.0, midrow_gain)) / surface_gain))
            if surface_gain > 1e-12
            else 0.0
        )

    delta_e = float(cand_surprise.mean_l2_sq_surprise - base_surprise.mean_l2_sq_surprise)
    target_col = bool(cand_surprise.target_rankme_metrics["is_collapsed"])
    pred_col = bool(cand_surprise.pred_rankme_metrics["is_collapsed"])

    reasons: list[str] = []
    if (
        format_share is not None
        and surface_gain >= cfg.min_surface_gain_for_gaming_check
        and format_share > cfg.max_format_artifact_share
    ):
        reasons.append("opening_template_format_artifact")

    if surface_gain >= cfg.min_surface_gain_for_gaming_check:
        if delta_e > cfg.max_allowed_surprise_regression or (
            cfg.require_surprise_reduction_on_ce_gain and delta_e > -cfg.min_surprise_improvement
        ):
            reasons.append("semantic_surprise_stagnation_or_increase")

    if target_col:
        reasons.append("target_rankme_or_isotropy_collapse")
    if pred_col:
        reasons.append("predictor_rankme_or_isotropy_collapse")

    return TemplateGamingDiagnosis(
        surface_nll_gain=surface_gain,
        midrow_nll_gain=midrow_gain,
        format_artifact_share=format_share,
        base_e_jepa=float(base_surprise.mean_l2_sq_surprise),
        cand_e_jepa=float(cand_surprise.mean_l2_sq_surprise),
        delta_e_jepa=delta_e,
        target_collapsed=target_col,
        pred_collapsed=pred_col,
        template_gaming_detected=(len(reasons) > 0),
        gaming_reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# 5. Semantic-Deficit Pheromones Royales & Simplex Conservation
# ---------------------------------------------------------------------------


def compute_semantic_foraging_weights(
    domain_jepa_surprise: Mapping[str, float],
    *,
    domain_midrow_nll: Mapping[str, float] | None = None,
    domain_surface_nll: Mapping[str, float] | None = None,
    domain_eces: Mapping[str, float] | None = None,
    domain_round_counts: Mapping[str, int] | None = None,
    config: JepaWspCriticConfig | None = None,
) -> tuple[dict[str, float], tuple[dict[str, Any], ...]]:
    """Convert domain semantic surprise ``E_JEPA`` into simplex-normalized foraging weights ``p_d``."""
    cfg = config if config is not None else JepaWspCriticConfig()
    if not isinstance(domain_jepa_surprise, Mapping) or not domain_jepa_surprise:
        raise ValueError("domain_jepa_surprise must be a non-empty Mapping")

    jepa_clean = {
        _require_nonempty_str(k, "domain_jepa_surprise key"): _require_finite_float(
            v, f"domain_jepa_surprise[{k!r}]", min_value=0.0
        )
        for k, v in domain_jepa_surprise.items()
    }
    domain_keys = sorted(jepa_clean.keys())

    def _clean_opt_map(raw_map: Mapping[str, float] | None, label: str) -> dict[str, float]:
        if raw_map is None:
            return {d: 0.0 for d in domain_keys}
        if not isinstance(raw_map, Mapping):
            raise TypeError(f"{label} must be a Mapping or None")
        if set(raw_map.keys()) != set(domain_keys):
            raise ValueError(f"{label} must share identical domain keys with domain_jepa_surprise")
        return {d: _require_finite_float(raw_map[d], f"{label}[{d!r}]", min_value=0.0) for d in domain_keys}

    mid_clean = _clean_opt_map(domain_midrow_nll, "domain_midrow_nll")
    surf_clean = _clean_opt_map(domain_surface_nll, "domain_surface_nll")
    ece_clean = _clean_opt_map(domain_eces, "domain_eces")

    specialist_domains = [d for d in domain_keys if not _is_general_key(d, cfg)]
    target_domains = specialist_domains if specialist_domains else domain_keys

    counts_map: dict[str, int] = {}
    if domain_round_counts is not None:
        if not isinstance(domain_round_counts, Mapping):
            raise TypeError("domain_round_counts must be a Mapping or None")
        counts_map = {
            d: _require_int(domain_round_counts.get(d, 0), f"domain_round_counts[{d!r}]", min_value=0)
            for d in target_domains
        }
    total_visits = max(1, sum(counts_map.values()))

    deficits: dict[str, float] = {}
    ucb_bonuses: dict[str, float] = {}
    for dom in target_domains:
        bonus = (
            cfg.ucb_exploration_coeff * math.sqrt(math.log(float(total_visits) + 1.0) / float(counts_map[dom] + 1))
            if domain_round_counts is not None
            else 0.0
        )
        ucb_bonuses[dom] = bonus
        deficits[dom] = (
            cfg.pheromone_jepa_weight * jepa_clean[dom]
            + cfg.pheromone_midrow_nll_weight * mid_clean[dom]
            + cfg.pheromone_surface_nll_weight * surf_clean[dom]
            + cfg.pheromone_ece_weight * ece_clean[dom]
            + bonus
        )

    max_def = max(deficits.values())
    exp_logits = {
        d: math.exp(max(-500.0, (deficits[d] - max_def) / cfg.foraging_temperature)) for d in target_domains
    }
    z_sum = sum(exp_logits.values())
    weights_by_domain = {d: (0.0 if d not in exp_logits else exp_logits[d] / z_sum) for d in domain_keys}

    priorities = tuple(
        {
            "domain_id": dom,
            "e_jepa_surprise": jepa_clean[dom],
            "midrow_nll": mid_clean[dom],
            "surface_nll": surf_clean[dom],
            "ece_value": ece_clean[dom],
            "ucb_bonus": ucb_bonuses.get(dom, 0.0),
            "conceptual_deficit": deficits.get(dom, 0.0),
            "foraging_priority_weight": weights_by_domain[dom],
        }
        for dom in sorted(domain_keys, key=lambda d: (-weights_by_domain[d], d))
    )
    return weights_by_domain, priorities


def emit_jepa_wsp_pheromone_directive(
    domain_jepa_surprise: Mapping[str, float],
    *,
    domain_midrow_nll: Mapping[str, float] | None = None,
    domain_surface_nll: Mapping[str, float] | None = None,
    domain_eces: Mapping[str, float] | None = None,
    domain_round_counts: Mapping[str, int] | None = None,
    expert_causal_gains_nats: Mapping[str | int, float] | None = None,
    expert_prose_harm_nats: Mapping[str | int, float] | None = None,
    router_expert_kl_mismatch: Mapping[str | int, float] | float = 0.0,
    campaign_id: str = "la-ruche-jepa-wsp",
    round_index: int = 1,
    config: JepaWspCriticConfig | None = None,
) -> QueenPheromoneDirective:
    """Emit a SHA-256-signed ``QueenPheromoneDirective`` steered by ``E_JEPA`` semantic surprise."""
    cfg = config if config is not None else JepaWspCriticConfig()
    camp_s = _require_nonempty_str(campaign_id, "campaign_id")
    r_idx = _require_int(round_index, "round_index", min_value=1)

    if isinstance(router_expert_kl_mismatch, Mapping):
        if not router_expert_kl_mismatch:
            raise ValueError("router_expert_kl_mismatch mapping must not be empty")
        max_kl = max(_require_finite_float(v, f"kl[{k!r}]", min_value=0.0) for k, v in router_expert_kl_mismatch.items())
    else:
        max_kl = _require_finite_float(router_expert_kl_mismatch, "router_expert_kl_mismatch", min_value=0.0)

    weights_by_domain, priorities = compute_semantic_foraging_weights(
        domain_jepa_surprise,
        domain_midrow_nll=domain_midrow_nll,
        domain_surface_nll=domain_surface_nll,
        domain_eces=domain_eces,
        domain_round_counts=domain_round_counts,
        config=cfg,
    )

    anchors: list[str] = []
    corruptors: list[str] = []
    active: list[str] = []
    if (expert_causal_gains_nats is None) != (expert_prose_harm_nats is None):
        raise ValueError("expert_causal_gains_nats and expert_prose_harm_nats must both be provided or omitted")
    if expert_causal_gains_nats is not None and expert_prose_harm_nats is not None:
        gains_clean = {
            str(k).strip(): _require_finite_float(v, f"gain[{k!r}]") for k, v in expert_causal_gains_nats.items()
        }
        harms_clean = {
            str(k).strip(): _require_finite_float(v, f"harm[{k!r}]") for k, v in expert_prose_harm_nats.items()
        }
        if not gains_clean or set(gains_clean.keys()) != set(harms_clean.keys()):
            raise ValueError("expert_causal_gains_nats and expert_prose_harm_nats must be non-empty with equal keys")
        for ek in sorted(gains_clean.keys()):
            g_v, h_v = gains_clean[ek], harms_clean[ek]
            if (h_v > cfg.prose_harm_sink_floor_nats and g_v > cfg.anchor_gain_floor_nats) or (
                g_v > cfg.prose_harm_sink_floor_nats and h_v >= 0.0
            ):
                corruptors.append(ek)
            elif g_v <= cfg.anchor_gain_floor_nats and h_v <= cfg.prose_harm_sink_floor_nats:
                anchors.append(ek)
            else:
                active.append(ek)
    else:
        active = [d for d in sorted(weights_by_domain.keys()) if not _is_general_key(d, cfg)]
        if not active:
            active = sorted(weights_by_domain.keys())

    router_mode = "router_frozen" if max_kl > cfg.coadapt_kl_ceiling else "router_unfrozen_coadapted"
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
        "coadapt_kl_ceiling": cfg.coadapt_kl_ceiling,
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
        coadapt_kl_ceiling=cfg.coadapt_kl_ceiling,
        bee_foraging_domain_weights=weights_by_domain,
        domain_priorities=priorities,
        directive_sha256=sig,
    )


# ---------------------------------------------------------------------------
# 6. CPU Operculation d'Alveole Gate & Atomic Bitwise Rollback
# ---------------------------------------------------------------------------


def _extract_state_tensors(
    state: dict[str, torch.Tensor] | nn.Module | Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if isinstance(state, nn.Module):
        return {k: v.detach().to(device="cpu").clone() for k, v in state.state_dict().items()}
    if isinstance(state, Mapping) and state:
        out: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            ks = _require_nonempty_str(k, "state key")
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"state[{k!r}] must be a torch.Tensor, got {type(v).__name__}")
            out[ks] = v.detach().to(device="cpu").clone()
        return out
    raise TypeError("state must be a non-empty tensor Mapping or nn.Module")


def _apply_state_bitwise(target: dict[str, torch.Tensor] | nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
    if isinstance(target, nn.Module):
        cur = target.state_dict()
        target.load_state_dict({k: v.to(device=cur[k].device, dtype=cur[k].dtype) for k, v in snapshot.items()})
        return
    for k, v in snapshot.items():
        cur_t = target[k]
        target[k] = v.to(device=cur_t.device, dtype=cur_t.dtype).clone()


def _normalize_domain_metrics(
    metrics: Mapping[str, Mapping[str, float]],
    label: str,
    cfg: JepaWspCriticConfig,
) -> dict[str, dict[str, float | None]]:
    if not isinstance(metrics, Mapping) or len(metrics) < 2:
        raise ValueError(f"{label} must be a mapping covering general prose and at least one specialist domain")
    out: dict[str, dict[str, float | None]] = {}
    for dom in sorted(metrics.keys()):
        dom_s = _require_nonempty_str(dom, f"{label} domain")
        row = metrics[dom]
        if not isinstance(row, Mapping):
            raise TypeError(f"{label}[{dom!r}] must be a mapping")
        nll_raw = row.get("nll", row.get("surface_nll", row.get("full_nll")))
        if nll_raw is None:
            raise ValueError(f"{label}[{dom!r}] must contain 'nll' or 'surface_nll'")
        nll_val = _require_finite_float(nll_raw, f"{label}[{dom!r}]['nll']", min_value=0.0)
        mid_raw = row.get("midrow_nll")
        mid_val = (
            _require_finite_float(mid_raw, f"{label}[{dom!r}]['midrow_nll']", min_value=0.0)
            if mid_raw is not None
            else None
        )
        ece_val = _require_finite_float(row.get("ece", 0.0), f"{label}[{dom!r}]['ece']", min_value=0.0)
        out[dom_s] = {"nll": nll_val, "midrow_nll": mid_val, "ece": ece_val}

    if not any(_is_general_key(d, cfg) for d in out):
        raise ValueError(f"{label} must contain a general-prose domain matching {cfg.general_domain_name!r}")
    if not any(not _is_general_key(d, cfg) for d in out):
        raise ValueError(f"{label} must contain at least one specialist domain")
    return out


def _normalize_domain_latents(
    latents: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    expected_domains: set[str],
    label: str,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if not isinstance(latents, Mapping) or set(latents.keys()) != expected_domains:
        raise ValueError(f"{label} must share identical domain keys with domain_metrics")
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for dom in sorted(expected_domains):
        pair = latents[dom]
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"{label}[{dom!r}] must be a (z_ctx, z_target) 2-tuple")
        z_ctx = _require_fp_tensor(pair[0], f"{label}[{dom!r}][0]", ndim=2)
        z_tgt = _require_fp_tensor(pair[1], f"{label}[{dom!r}][1]", ndim=2).detach()
        if int(z_ctx.shape[0]) != int(z_tgt.shape[0]):
            raise ValueError(f"{label}[{dom!r}] z_ctx and z_target sample counts must match")
        out[dom] = (z_ctx, z_tgt)
    return out


@torch.no_grad()
def evaluate_jepa_wsp_operculation(
    mutable_state: dict[str, torch.Tensor] | nn.Module,
    candidate_state: Mapping[str, torch.Tensor] | nn.Module,
    predictor: FrozenLatentJepaPredictor,
    domain_metrics_base: Mapping[str, Mapping[str, float]],
    domain_metrics_cand: Mapping[str, Mapping[str, float]],
    domain_latents_base: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    domain_latents_cand: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    expert_causal_gains_nats: Mapping[str | int, float] | None = None,
    expert_prose_harm_nats: Mapping[str | int, float] | None = None,
    router_expert_kl_mismatch: Mapping[str | int, float] | float = 0.0,
    domain_round_counts: Mapping[str, int] | None = None,
    campaign_id: str = "la-ruche-jepa-wsp",
    round_index: int = 1,
    config: JepaWspCriticConfig | None = None,
) -> dict[str, Any]:
    """Gate a candidate update inside CPU ``Operculation d'Alveole`` with atomic bitwise rollback."""
    cfg = config if config is not None else JepaWspCriticConfig()
    if not isinstance(predictor, FrozenLatentJepaPredictor):
        raise TypeError(f"predictor must be FrozenLatentJepaPredictor, got {type(predictor).__name__}")
    if not isinstance(mutable_state, (dict, nn.Module)):
        raise TypeError("mutable_state must be a mutable dict or nn.Module")

    base_snapshot = _extract_state_tensors(mutable_state)
    initial_state_sha256 = compute_state_sha256(base_snapshot)

    cand_invalid_values = False
    try:
        cand_snapshot = _extract_state_tensors(candidate_state)
        if set(cand_snapshot.keys()) != set(base_snapshot.keys()):
            raise ValueError("candidate_state and mutable_state must share identical tensor keys")
        for k in base_snapshot:
            if tuple(cand_snapshot[k].shape) != tuple(base_snapshot[k].shape):
                raise ValueError(
                    f"Shape mismatch on {k!r}: {tuple(cand_snapshot[k].shape)} != {tuple(base_snapshot[k].shape)}"
                )
        candidate_state_sha256 = compute_state_sha256(cand_snapshot)
    except ValueError as exc:
        if "non-finite" not in str(exc):
            raise
        cand_invalid_values = True
        cand_snapshot = base_snapshot
        candidate_state_sha256 = initial_state_sha256

    base_m = _normalize_domain_metrics(domain_metrics_base, "domain_metrics_base", cfg)
    cand_m = _normalize_domain_metrics(domain_metrics_cand, "domain_metrics_cand", cfg)
    if set(base_m.keys()) != set(cand_m.keys()):
        raise ValueError("domain_metrics_base and domain_metrics_cand must share identical domain keys")

    domains = sorted(base_m.keys())
    base_lat = _normalize_domain_latents(domain_latents_base, set(domains), "domain_latents_base")
    cand_lat = _normalize_domain_latents(domain_latents_cand, set(domains), "domain_latents_cand")

    base_dom_surprises: dict[str, LatentSurpriseReport] = {}
    cand_dom_surprises: dict[str, LatentSurpriseReport] = {}
    for dom in domains:
        b_rep = compute_latent_jepa_surprise(predictor, base_lat[dom][0], base_lat[dom][1], config=cfg)
        c_rep = compute_latent_jepa_surprise(
            predictor,
            cand_lat[dom][0],
            cand_lat[dom][1],
            config=cfg,
            reference_target_report=b_rep.target_rankme_report,
            reference_pred_report=b_rep.pred_rankme_report,
        )
        base_dom_surprises[dom] = b_rep
        cand_dom_surprises[dom] = c_rep

    base_pooled = compute_latent_jepa_surprise(
        predictor,
        torch.cat([base_lat[d][0] for d in domains], dim=0),
        torch.cat([base_lat[d][1] for d in domains], dim=0),
        config=cfg,
    )
    cand_pooled = compute_latent_jepa_surprise(
        predictor,
        torch.cat([cand_lat[d][0] for d in domains], dim=0),
        torch.cat([cand_lat[d][1] for d in domains], dim=0),
        config=cfg,
        reference_target_report=base_pooled.target_rankme_report,
        reference_pred_report=base_pooled.pred_rankme_report,
    )

    gen_domains = [d for d in domains if _is_general_key(d, cfg)]
    spec_domains = [d for d in domains if d not in gen_domains]

    base_gen_nll = float(sum(float(base_m[d]["nll"]) for d in gen_domains) / float(len(gen_domains)))
    cand_gen_nll = float(sum(float(cand_m[d]["nll"]) for d in gen_domains) / float(len(gen_domains)))
    gen_drift_nats = cand_gen_nll - base_gen_nll
    gen_drift_ratio = gen_drift_nats / max(base_gen_nll, cfg.eps)

    per_domain_drift_ratio = {
        d: (float(cand_m[d]["nll"]) - float(base_m[d]["nll"])) / max(float(base_m[d]["nll"]), cfg.eps)
        for d in domains
    }
    worst_domain_drift = max(per_domain_drift_ratio.values())
    guard_ok = (gen_drift_ratio <= cfg.max_general_drift_ratio + 1e-12) and (
        (not cfg.enforce_all_domain_guard) or (worst_domain_drift <= cfg.max_domain_drift_ratio + 1e-12)
    )

    per_domain_gaming: dict[str, dict[str, Any]] = {}
    gaming_reasons_all: list[str] = []
    any_collapse = bool(
        cand_pooled.target_rankme_metrics["is_collapsed"] or cand_pooled.pred_rankme_metrics["is_collapsed"]
    )

    for dom in spec_domains:
        diag = diagnose_template_gaming(
            base_surface_nll=float(base_m[dom]["nll"]),
            cand_surface_nll=float(cand_m[dom]["nll"]),
            base_surprise=base_dom_surprises[dom],
            cand_surprise=cand_dom_surprises[dom],
            base_midrow_nll=base_m[dom]["midrow_nll"],
            cand_midrow_nll=cand_m[dom]["midrow_nll"],
            config=cfg,
        )
        per_domain_gaming[dom] = diag.to_dict()
        if diag.target_collapsed or diag.pred_collapsed:
            any_collapse = True
        if diag.template_gaming_detected:
            gaming_reasons_all.extend(f"{dom}:{r}" for r in diag.gaming_reasons)

    base_spec_surf = float(sum(float(base_m[d]["nll"]) for d in spec_domains) / float(len(spec_domains)))
    cand_spec_surf = float(sum(float(cand_m[d]["nll"]) for d in spec_domains) / float(len(spec_domains)))
    all_mid = all(base_m[d]["midrow_nll"] is not None and cand_m[d]["midrow_nll"] is not None for d in spec_domains)
    base_spec_mid = (
        float(sum(float(base_m[d]["midrow_nll"]) for d in spec_domains) / float(len(spec_domains))) if all_mid else None
    )
    cand_spec_mid = (
        float(sum(float(cand_m[d]["midrow_nll"]) for d in spec_domains) / float(len(spec_domains))) if all_mid else None
    )
    pooled_diag = diagnose_template_gaming(
        base_surface_nll=base_spec_surf,
        cand_surface_nll=cand_spec_surf,
        base_surprise=base_pooled,
        cand_surprise=cand_pooled,
        base_midrow_nll=base_spec_mid,
        cand_midrow_nll=cand_spec_mid,
        config=cfg,
    )
    if pooled_diag.template_gaming_detected:
        gaming_reasons_all.extend(f"pooled:{r}" for r in pooled_diag.gaming_reasons)
    dedup_gaming_reasons = list(dict.fromkeys(gaming_reasons_all))
    template_gaming_detected = any(
        r.endswith("opening_template_format_artifact") or r.endswith("semantic_surprise_stagnation_or_increase")
        for r in dedup_gaming_reasons
    )

    base_nll_mean = float(sum(float(base_m[d]["nll"]) for d in domains) / float(len(domains)))
    cand_nll_mean = float(sum(float(cand_m[d]["nll"]) for d in domains) / float(len(domains)))
    base_ece_mean = float(sum(float(base_m[d]["ece"]) for d in domains) / float(len(domains)))
    cand_ece_mean = float(sum(float(cand_m[d]["ece"]) for d in domains) / float(len(domains)))
    nll_diff = cand_nll_mean - base_nll_mean
    ece_diff = cand_ece_mean - base_ece_mean
    jepa_surprise_diff = cand_pooled.mean_l2_sq_surprise - base_pooled.mean_l2_sq_surprise
    raw_energy = nll_diff + cfg.ece_weight * ece_diff + cfg.jepa_surprise_weight * jepa_surprise_diff

    if cand_invalid_values:
        rejection_reason = "non_finite_candidate_weights"
    elif any_collapse:
        rejection_reason = "rankme_or_isotropy_collapse_detected"
    elif template_gaming_detected:
        rejection_reason = "opening_template_ce_gaming_detected"
    elif not guard_ok:
        rejection_reason = "general_prose_or_domain_guard_exceeded"
    elif raw_energy >= -cfg.min_energy_improvement:
        rejection_reason = "anse_jepa_wsp_energy_non_improving"
    else:
        rejection_reason = ""

    accepted = rejection_reason == ""
    if accepted:
        _apply_state_bitwise(mutable_state, cand_snapshot)
        rolled_back = False
        final_anse_energy = float(raw_energy)
        active_m = cand_m
        active_surprises = cand_dom_surprises
        active_pooled = cand_pooled
    else:
        _apply_state_bitwise(mutable_state, base_snapshot)
        rolled_back = True
        final_anse_energy = ANSE_MAXIMUM_PAIN
        active_m = base_m
        active_surprises = base_dom_surprises
        active_pooled = base_pooled

    final_state_sha256 = compute_state_sha256(_extract_state_tensors(mutable_state))
    rollback_verified = (final_state_sha256 == initial_state_sha256) if rolled_back else True

    domain_surprise_map = {d: float(active_surprises[d].mean_l2_sq_surprise) for d in domains}
    domain_surface_map = {d: float(active_m[d]["nll"]) for d in domains}
    domain_ece_map = {d: float(active_m[d]["ece"]) for d in domains}
    domain_midrow_map = (
        {d: float(active_m[d]["midrow_nll"]) for d in domains}
        if all(active_m[d]["midrow_nll"] is not None for d in domains)
        else None
    )

    pheromone = emit_jepa_wsp_pheromone_directive(
        domain_surprise_map,
        domain_midrow_nll=domain_midrow_map,
        domain_surface_nll=domain_surface_map,
        domain_eces=domain_ece_map,
        domain_round_counts=domain_round_counts,
        expert_causal_gains_nats=expert_causal_gains_nats,
        expert_prose_harm_nats=expert_prose_harm_nats,
        router_expert_kl_mismatch=router_expert_kl_mismatch,
        campaign_id=campaign_id,
        round_index=round_index,
        config=cfg,
    )

    receipt: dict[str, Any] = {
        "schema": JEPA_WSP_CRITIC_SCHEMA,
        "object": JEPA_WSP_RECEIPT_OBJECT,
        "campaign_id": str(campaign_id),
        "round_index": int(round_index),
        "predictor_mode": predictor.mode,
        "predictor_sha256": predictor.predictor_sha256,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "rollback_verified": rollback_verified,
        "collapse_rejected": any_collapse,
        "template_gaming_rejected": template_gaming_detected,
        "guard_rejected": (
            (not guard_ok) and (not any_collapse) and (not template_gaming_detected) and (not cand_invalid_values)
        ),
        "rejection_reason": rejection_reason,
        "gaming_reasons": dedup_gaming_reasons,
        "initial_state_sha256": initial_state_sha256,
        "candidate_state_sha256": candidate_state_sha256,
        "final_state_sha256": final_state_sha256,
        "base_nll_mean": base_nll_mean,
        "cand_nll_mean": cand_nll_mean,
        "nll_diff": nll_diff,
        "ece_diff": ece_diff,
        "base_e_jepa_mean": float(base_pooled.mean_l2_sq_surprise),
        "cand_e_jepa_mean": float(cand_pooled.mean_l2_sq_surprise),
        "active_e_jepa_mean": float(active_pooled.mean_l2_sq_surprise),
        "jepa_surprise_diff": jepa_surprise_diff,
        "raw_anse_jepa_wsp_energy": raw_energy,
        "anse_energy": final_anse_energy,
        "general_drift_nats": gen_drift_nats,
        "general_drift_ratio": gen_drift_ratio,
        "worst_domain_drift_ratio": worst_domain_drift,
        "per_domain_drift_ratio": per_domain_drift_ratio,
        "guard_ok": guard_ok,
        "pooled_template_diagnosis": pooled_diag.to_dict(),
        "per_domain_template_diagnosis": per_domain_gaming,
        "base_surprise_summary": base_pooled.to_summary_dict(),
        "cand_surprise_summary": cand_pooled.to_summary_dict(),
        "per_domain_e_jepa": {d: float(cand_dom_surprises[d].mean_l2_sq_surprise) for d in domains},
        "pheromone_directive": pheromone.to_dict(),
        "bee_foraging_domain_weights": dict(pheromone.bee_foraging_domain_weights),
        "foraging_weights_sum": float(sum(pheromone.bee_foraging_domain_weights.values())),
        "config": dataclasses.asdict(cfg),
    }
    receipt["receipt_sha256"] = compute_payload_sha256(receipt)
    return receipt


def verify_jepa_wsp_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Verify canonical SHA-256 provenance, pheromone signature, simplex conservation, and ANSE gates."""
    if not isinstance(receipt, Mapping):
        return False
    if receipt.get("schema") != JEPA_WSP_CRITIC_SCHEMA or receipt.get("object") != JEPA_WSP_RECEIPT_OBJECT:
        return False
    for key in (
        "receipt_sha256",
        "predictor_sha256",
        "initial_state_sha256",
        "candidate_state_sha256",
        "final_state_sha256",
    ):
        val = receipt.get(key)
        if not isinstance(val, str) or not _SHA256_HEX_RE.match(val):
            return False

    core = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        if compute_payload_sha256(core) != receipt["receipt_sha256"]:
            return False
    except (TypeError, ValueError):
        return False

    pheromone = receipt.get("pheromone_directive")
    if not isinstance(pheromone, Mapping) or not verify_pheromone_directive_signature(pheromone):
        return False
    weights = pheromone.get("bee_foraging_domain_weights")
    if not isinstance(weights, Mapping) or not weights:
        return False
    w_vals = [float(v) for v in weights.values()]
    if any(not math.isfinite(v) or v < 0.0 for v in w_vals) or abs(sum(w_vals) - 1.0) > 1e-9:
        return False

    accepted = receipt.get("accepted")
    rolled_back = receipt.get("rolled_back")
    if not isinstance(accepted, bool) or not isinstance(rolled_back, bool) or accepted == rolled_back:
        return False
    if rolled_back:
        return (
            receipt["final_state_sha256"] == receipt["initial_state_sha256"]
            and float(receipt.get("anse_energy", 0.0)) == ANSE_MAXIMUM_PAIN
            and bool(receipt.get("rollback_verified")) is True
        )

    cfg = receipt.get("config")
    max_gen = (
        float(cfg.get("max_general_drift_ratio", DEFAULT_MAX_GENERAL_DRIFT_RATIO))
        if isinstance(cfg, Mapping)
        else DEFAULT_MAX_GENERAL_DRIFT_RATIO
    )
    return (
        float(receipt.get("anse_energy", 1.0)) < 0.0
        and float(receipt.get("general_drift_ratio", 1.0)) <= max_gen + 1e-12
        and receipt.get("collapse_rejected") is False
        and receipt.get("template_gaming_rejected") is False
        and receipt.get("guard_ok") is True
    )


__all__ = [
    "ANSE_MAXIMUM_PAIN",
    "DEFAULT_ANCHOR_GAIN_FLOOR_NATS",
    "DEFAULT_COADAPT_KL_CEILING",
    "DEFAULT_ECE_WEIGHT",
    "DEFAULT_EPS",
    "DEFAULT_FORAGING_TEMPERATURE",
    "DEFAULT_GENERAL_DOMAIN",
    "DEFAULT_GENERAL_PREFIXES",
    "DEFAULT_JEPA_SURPRISE_WEIGHT",
    "DEFAULT_MAX_ALLOWED_SURPRISE_REGRESSION",
    "DEFAULT_MAX_ANISOTROPY_ABS",
    "DEFAULT_MAX_ANISOTROPY_DELTA",
    "DEFAULT_MAX_DOMAIN_DRIFT_RATIO",
    "DEFAULT_MAX_FORMAT_ARTIFACT_SHARE",
    "DEFAULT_MAX_GENERAL_DRIFT_RATIO",
    "DEFAULT_MIN_ENERGY_IMPROVEMENT",
    "DEFAULT_MIN_NORMALIZED_RANKME",
    "DEFAULT_MIN_RANKME_RATIO",
    "DEFAULT_MIN_SURFACE_GAIN_FOR_GAMING_CHECK",
    "DEFAULT_MIN_SURPRISE_IMPROVEMENT",
    "DEFAULT_PROSE_HARM_SINK_FLOOR_NATS",
    "DEFAULT_UCB_EXPLORATION_COEFF",
    "JEPA_WSP_CRITIC_SCHEMA",
    "JEPA_WSP_RECEIPT_OBJECT",
    "QUEEN_CORTEX_SCHEMA",
    "QUEEN_PHEROMONE_OBJECT",
    "VALID_MLP_ACTIVATIONS",
    "VALID_PREDICTOR_MODES",
    "FrozenLatentJepaPredictor",
    "JepaWspCriticConfig",
    "LatentSurpriseReport",
    "QueenPheromoneDirective",
    "TemplateGamingDiagnosis",
    "build_frozen_jepa_predictor",
    "compute_latent_jepa_surprise",
    "compute_payload_sha256",
    "compute_semantic_foraging_weights",
    "compute_state_sha256",
    "diagnose_template_gaming",
    "emit_jepa_wsp_pheromone_directive",
    "evaluate_jepa_wsp_operculation",
    "fit_frozen_jepa_predictor_closed_form",
    "summarize_opening_vs_midrow_nll",
    "verify_jepa_wsp_receipt_provenance",
    "verify_pheromone_directive_signature",
]

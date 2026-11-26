# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Sequential QLoRA Chain Catastrophic-Forgetting Rescuer via TIES & Factorized Null-Space Projection.

Implements training-free, CPU-only (0 MiB GPU VRAM, 0 gradient steps) Weight-Space Programming
(WSP) grounded in ``PILOT7`` (IBM Granite-3.1-1B-A400M 32-Expert MoE) and ``PILOT8`` (IBM
Granite-3.1-3B-A800M 40-Expert MoE), where sequentially chained multi-domain QLoRA adapters
degraded general prose ``D0`` by ``+2.88%..+7.16%`` while overwriting earlier specialists.
Extracting consecutive chain increments ``delta_d = W_d - W_{d-1}``, resolving cross-step
sign interference via TIES top-k sign election, projecting into the orthonormal prose null-space
``P_0 = I - U_0 U_0^T`` via associative low-RSS factorization ``(scale * B) @ (A @ P_0)`` (avoiding
dense ``d_out x d_in`` intermediate materialization), and folding safe temperature calibration
``1 / tau_temp`` into final RMSNorm weights rescues forgotten specialists while holding general
prose drift within ``D0 <= +1.0%`` under the monotone ANSE thermodynamic ratchet.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from hivemind.swarm.byzantine_wsp_delta_aggregator import compute_payload_sha256

RESCUER_SCHEMA: str = "hivemind.sequential_chain_ties_nullspace_rescuer/1"
RESCUER_RECEIPT_OBJECT: str = "hivemind.sequential_chain_rescue_receipt"

ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_GENERAL_DOMAIN: str = "D0_general_prose"
DEFAULT_TIES_DENSITY: float = 0.25
DEFAULT_MAX_GENERAL_DRIFT_RATIO: float = 0.010
DEFAULT_MAX_DOMAIN_DRIFT_RATIO: float = 0.010
DEFAULT_ECE_WEIGHT: float = 0.25
DEFAULT_ECE_BINS: int = 10
DEFAULT_SAFE_TAU_GRID: tuple[float, ...] = (1.0, 1.03, 1.06, 1.10, 1.15, 1.22, 1.30)
DEFAULT_TIES_ALPHA_GRID: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0, 1.25)
DEFAULT_DONOR_ALPHA_GRID: tuple[float, ...] = (0.0, 0.5, 1.0)
DEFAULT_RMSNORM_EPS: float = 1e-6
DEFAULT_PROSE_INVARIANCE_EPS: float = 1e-4

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# 1. Validation & Configuration Helpers
# ---------------------------------------------------------------------------


def _validate_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value}")
    return out


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return int(value)


def _validate_2d_tensor(tensor: Any, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(tensor.shape)}")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    out = tensor.detach().to(dtype=torch.float32, device="cpu")
    if not torch.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values")
    return out


def _validate_1d_tensor(tensor: Any, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(tensor.shape)}")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    out = tensor.detach().to(dtype=torch.float32, device="cpu")
    if not torch.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values")
    return out


def _validate_square_projector(p_null: Any, expected_dim: int, name: str = "p_null") -> torch.Tensor:
    proj = _validate_2d_tensor(p_null, name)
    if proj.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"{name} must have shape ({expected_dim}, {expected_dim}), got {tuple(proj.shape)}"
        )
    return proj


def _validate_float_grid(values: Sequence[float], name: str, *, min_value: float) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of numbers")
    parsed: list[float] = []
    for idx, val in enumerate(values):
        fval = _validate_finite_float(val, f"{name}[{idx}]")
        if fval < min_value:
            raise ValueError(f"{name}[{idx}] must be >= {min_value}, got {fval}")
        parsed.append(fval)
    return tuple(parsed)


@dataclass(frozen=True)
class LoRAChainStep:
    """Single step in a sequential QLoRA fine-tuning chain."""

    step_name: str
    b_mat: torch.Tensor
    a_mat: torch.Tensor
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.step_name, str) or not self.step_name.strip():
            raise ValueError("step_name must be a non-empty string")
        b_t = _validate_2d_tensor(self.b_mat, f"{self.step_name}.b_mat")
        a_t = _validate_2d_tensor(self.a_mat, f"{self.step_name}.a_mat")
        if b_t.shape[1] != a_t.shape[0]:
            raise ValueError(
                f"{self.step_name} LoRA rank mismatch: b_mat {tuple(b_t.shape)} vs a_mat {tuple(a_t.shape)}"
            )
        _validate_finite_float(self.scale, f"{self.step_name}.scale")


@dataclass(frozen=True)
class IncrementalChainTaskVector:
    """Pure step-d domain increment extracted as delta_d = W_d - W_{d-1}."""

    step_index: int
    step_name: str
    delta_weight: torch.Tensor
    frobenius_norm: float
    cumulative_norm: float
    cosine_with_previous: float


@dataclass(frozen=True)
class TIESProjectionSummary:
    """Result and diagnostics of TIES sign election + P_0 prose-nullspace projection."""

    delta_elected: torch.Tensor
    delta_projected: torch.Tensor
    n_steps: int
    density: float
    trimmed_nonzero_fraction: float
    sign_conflict_cancelled_fraction: float
    elected_frobenius_norm: float
    projected_frobenius_norm: float


@dataclass(frozen=True)
class SequentialChainRescueConfig:
    """Configuration governing sequential QLoRA chain rescue and ANSE gating."""

    ties_density: float = DEFAULT_TIES_DENSITY
    ties_alpha_grid: tuple[float, ...] = DEFAULT_TIES_ALPHA_GRID
    donor_alpha_grid: tuple[float, ...] = DEFAULT_DONOR_ALPHA_GRID
    safe_tau_grid: tuple[float, ...] = DEFAULT_SAFE_TAU_GRID
    max_general_drift_ratio: float = DEFAULT_MAX_GENERAL_DRIFT_RATIO
    max_domain_drift_ratio: float = DEFAULT_MAX_DOMAIN_DRIFT_RATIO
    ece_weight: float = DEFAULT_ECE_WEIGHT
    ece_bins: int = DEFAULT_ECE_BINS
    rmsnorm_eps: float = DEFAULT_RMSNORM_EPS
    general_domain_name: str = DEFAULT_GENERAL_DOMAIN
    prose_invariance_eps: float = DEFAULT_PROSE_INVARIANCE_EPS

    def __post_init__(self) -> None:
        density = _validate_finite_float(self.ties_density, "ties_density")
        if not (0.0 < density <= 1.0):
            raise ValueError(f"ties_density must be in (0, 1], got {density}")
        _validate_float_grid(self.ties_alpha_grid, "ties_alpha_grid", min_value=0.0)
        _validate_float_grid(self.donor_alpha_grid, "donor_alpha_grid", min_value=0.0)
        _validate_float_grid(self.safe_tau_grid, "safe_tau_grid", min_value=1.0)
        max_gen = _validate_finite_float(self.max_general_drift_ratio, "max_general_drift_ratio")
        max_dom = _validate_finite_float(self.max_domain_drift_ratio, "max_domain_drift_ratio")
        if max_gen < 0.0 or max_dom < 0.0:
            raise ValueError("drift ratio ceilings must be >= 0.0")
        ece_w = _validate_finite_float(self.ece_weight, "ece_weight")
        if ece_w < 0.0:
            raise ValueError(f"ece_weight must be >= 0.0, got {ece_w}")
        _validate_positive_int(self.ece_bins, "ece_bins")
        eps = _validate_finite_float(self.rmsnorm_eps, "rmsnorm_eps")
        if eps <= 0.0:
            raise ValueError(f"rmsnorm_eps must be > 0, got {eps}")
        inv_eps = _validate_finite_float(self.prose_invariance_eps, "prose_invariance_eps")
        if inv_eps <= 0.0:
            raise ValueError(f"prose_invariance_eps must be > 0, got {inv_eps}")
        if not isinstance(self.general_domain_name, str) or not self.general_domain_name.strip():
            raise ValueError("general_domain_name must be a non-empty string")


# ---------------------------------------------------------------------------
# 2. Low-RSS Associative Factorized Projection & Incremental Chain Extraction
# ---------------------------------------------------------------------------


def estimate_factorized_vs_dense_flops(d_out: int, d_in: int, rank: int) -> dict[str, float]:
    """Compare FLOPs of associative ``(scale * B) @ (A @ P_0)`` vs dense ``((scale * B) @ A) @ P_0``."""
    out_dim = _validate_positive_int(d_out, "d_out")
    in_dim = _validate_positive_int(d_in, "d_in")
    r_dim = _validate_positive_int(rank, "rank")
    factorized_flops = float(2 * r_dim * in_dim * in_dim + 2 * out_dim * r_dim * in_dim)
    dense_flops = float(2 * out_dim * r_dim * in_dim + 2 * out_dim * in_dim * in_dim)
    speedup_ratio = dense_flops / factorized_flops
    return {
        "factorized_flops": factorized_flops,
        "dense_flops": dense_flops,
        "flop_speedup_ratio": speedup_ratio,
    }


def factorized_associative_nullspace_delta(
    b_scaled: torch.Tensor,
    a_mat: torch.Tensor,
    p_null: torch.Tensor,
) -> torch.Tensor:
    """Compute ``b_scaled @ (a_mat @ p_null)`` without materializing dense ``b_scaled @ a_mat`` first.

    Algebraically identical to ``(b_scaled @ a_mat) @ p_null`` while reducing peak intermediate
    memory and FLOPs from ``O(d_out * d_in^2)`` to ``O(r * d_in^2 + d_out * r * d_in)``.
    """
    b_t = _validate_2d_tensor(b_scaled, "b_scaled")
    a_t = _validate_2d_tensor(a_mat, "a_mat")
    if b_t.shape[1] != a_t.shape[0]:
        raise ValueError(
            f"Rank mismatch between b_scaled {tuple(b_t.shape)} and a_mat {tuple(a_t.shape)}"
        )
    proj = _validate_square_projector(p_null, int(a_t.shape[1]), "p_null")
    a_projected = a_t.to(dtype=torch.float64) @ proj.to(dtype=torch.float64)
    return (b_t.to(dtype=torch.float64) @ a_projected).to(dtype=torch.float32)


def _normalize_chain_step(item: Any, idx: int) -> LoRAChainStep:
    if isinstance(item, LoRAChainStep):
        return item
    if isinstance(item, Mapping):
        b_raw = item.get("b_mat", item.get("lora_B"))
        a_raw = item.get("a_mat", item.get("lora_A"))
        scale = float(item.get("scale", 1.0))
        step_name = str(item.get("step_name", f"step_{idx}"))
        return LoRAChainStep(step_name=step_name, b_mat=b_raw, a_mat=a_raw, scale=scale)
    if isinstance(item, tuple) and len(item) in (2, 3):
        b_raw = item[0]
        a_raw = item[1]
        scale = float(item[2]) if len(item) == 3 else 1.0
        return LoRAChainStep(step_name=f"step_{idx}", b_mat=b_raw, a_mat=a_raw, scale=scale)
    raise TypeError(
        f"chain_lora_factors[{idx}] must be LoRAChainStep, mapping, or (b_mat, a_mat[, scale]) tuple"
    )


def extract_incremental_chain_task_vectors(
    chain_lora_factors: Sequence[LoRAChainStep | Mapping[str, Any] | tuple[Any, ...]],
    *,
    p_null: torch.Tensor | None = None,
) -> list[IncrementalChainTaskVector]:
    """Extract pure step-``d`` increments ``delta_d = W_d - W_{d-1}`` from consecutive chain LoRAs.

    When ``p_null`` is supplied, each cumulative checkpoint ``W_d`` is evaluated directly inside
    the prose null-space via ``factorized_associative_nullspace_delta(scale * B_d, A_d, p_null)``,
    eliminating dense unprojected ``B @ A`` materialization.
    """
    if (
        not isinstance(chain_lora_factors, Sequence)
        or isinstance(chain_lora_factors, (str, bytes))
        or len(chain_lora_factors) == 0
    ):
        raise ValueError("chain_lora_factors must be a non-empty sequence")

    steps = [_normalize_chain_step(item, idx) for idx, item in enumerate(chain_lora_factors)]
    d_out = int(steps[0].b_mat.shape[0])
    d_in = int(steps[0].a_mat.shape[1])
    proj = _validate_square_projector(p_null, d_in, "p_null") if p_null is not None else None

    prev_cumulative: torch.Tensor | None = None
    prev_delta: torch.Tensor | None = None
    increments: list[IncrementalChainTaskVector] = []

    for idx, step in enumerate(steps):
        b_t = _validate_2d_tensor(step.b_mat, f"{step.step_name}.b_mat")
        a_t = _validate_2d_tensor(step.a_mat, f"{step.step_name}.a_mat")
        if b_t.shape[0] != d_out or a_t.shape[1] != d_in:
            raise ValueError(
                f"Inconsistent shape at {step.step_name}: expected ({d_out}, {d_in}), "
                f"got ({b_t.shape[0]}, {a_t.shape[1]})"
            )
        b_scaled = b_t * float(step.scale)
        if proj is not None:
            w_cumulative = factorized_associative_nullspace_delta(b_scaled, a_t, proj)
        else:
            w_cumulative = b_scaled @ a_t

        delta = w_cumulative.clone() if prev_cumulative is None else (w_cumulative - prev_cumulative)
        delta_norm = float(torch.linalg.norm(delta).item())
        cum_norm = float(torch.linalg.norm(w_cumulative).item())

        if prev_delta is None:
            cos_prev = 0.0
        else:
            prev_norm = float(torch.linalg.norm(prev_delta).item())
            denom = delta_norm * prev_norm
            cos_prev = float((delta * prev_delta).sum().item() / denom) if denom > 1e-12 else 0.0

        increments.append(
            IncrementalChainTaskVector(
                step_index=idx,
                step_name=step.step_name,
                delta_weight=delta,
                frobenius_norm=delta_norm,
                cumulative_norm=cum_norm,
                cosine_with_previous=cos_prev,
            )
        )
        prev_cumulative = w_cumulative
        prev_delta = delta

    return increments


# ---------------------------------------------------------------------------
# 3. TIES Sign Election & Null-Space Projection
# ---------------------------------------------------------------------------


def _trim_top_density(tensor: torch.Tensor, density: float) -> torch.Tensor:
    flat = tensor.reshape(-1)
    numel = flat.numel()
    if density >= 1.0:
        return tensor.clone()
    abs_flat = flat.abs()
    nonzero_abs = abs_flat[abs_flat > 0.0]
    if nonzero_abs.numel() == 0:
        return torch.zeros_like(tensor)
    k = max(1, int(math.ceil(numel * density)))
    if k >= nonzero_abs.numel():
        return tensor.clone()
    threshold = torch.topk(nonzero_abs, k, sorted=True).values[-1]
    mask = abs_flat >= threshold
    return (flat * mask.to(dtype=tensor.dtype)).reshape(tensor.shape)


def compute_prose_nullspace_projector(
    prose_activations: torch.Tensor,
    *,
    variance_threshold: float = 0.999,
) -> torch.Tensor:
    """Compute exact orthonormal prose null-space projector ``P_0 = I - U_0 U_0^T`` in float64."""
    x_prose = _validate_2d_tensor(prose_activations, "prose_activations").to(dtype=torch.float64)
    thresh = _validate_finite_float(variance_threshold, "variance_threshold")
    if not (0.0 < thresh <= 1.0):
        raise ValueError(f"variance_threshold must be in (0, 1], got {thresh}")
    d_in = int(x_prose.shape[1])
    _, s_vals, vh_mat = torch.linalg.svd(x_prose, full_matrices=False)
    energy = s_vals.pow(2)
    total_energy = float(energy.sum().item())
    if total_energy <= 1e-18:
        return torch.eye(d_in, dtype=torch.float32)
    cum_ratio = torch.cumsum(energy, dim=0) / total_energy
    keep = int(torch.searchsorted(cum_ratio, torch.tensor(thresh, dtype=torch.float64)).item()) + 1
    keep = min(keep, int(vh_mat.shape[0]))
    u_0 = vh_mat[:keep, :].T
    p_null = torch.eye(d_in, dtype=torch.float64) - (u_0 @ u_0.T)
    return p_null.to(dtype=torch.float32)


def ties_elect_and_project_increments(
    task_vectors: Sequence[IncrementalChainTaskVector | torch.Tensor],
    p_null: torch.Tensor,
    *,
    density: float = DEFAULT_TIES_DENSITY,
    step_weights: Sequence[float] | None = None,
) -> TIESProjectionSummary:
    """Trim incremental task vectors, elect consensus signs via TIES, and project into ``P_0``."""
    if (
        not isinstance(task_vectors, Sequence)
        or isinstance(task_vectors, (str, bytes))
        or len(task_vectors) == 0
    ):
        raise ValueError("task_vectors must be a non-empty sequence")
    dens = _validate_finite_float(density, "density")
    if not (0.0 < dens <= 1.0):
        raise ValueError(f"density must be in (0, 1], got {dens}")

    mats: list[torch.Tensor] = []
    for idx, item in enumerate(task_vectors):
        raw = item.delta_weight if isinstance(item, IncrementalChainTaskVector) else item
        mat = _validate_2d_tensor(raw, f"task_vectors[{idx}]")
        if mats and mat.shape != mats[0].shape:
            raise ValueError(
                f"Shape mismatch in task_vectors[{idx}]: expected {tuple(mats[0].shape)}, "
                f"got {tuple(mat.shape)}"
            )
        mats.append(mat)

    d_in = int(mats[0].shape[1])
    proj = _validate_square_projector(p_null, d_in, "p_null")

    n_steps = len(mats)
    if step_weights is None:
        weights = [1.0] * n_steps
    else:
        if len(step_weights) != n_steps:
            raise ValueError(
                f"step_weights length ({len(step_weights)}) must match task_vectors ({n_steps})"
            )
        weights = [_validate_finite_float(w, f"step_weights[{i}]") for i, w in enumerate(step_weights)]
        if any(w <= 0.0 for w in weights):
            raise ValueError("All step_weights must be strictly positive")

    trimmed_weighted = torch.stack(
        [_trim_top_density(mat, dens) * w for mat, w in zip(mats, weights, strict=True)],
        dim=0,
    )
    nonzero_mask = trimmed_weighted != 0.0
    total_nonzero = int(nonzero_mask.sum().item())
    trimmed_nonzero_fraction = float(total_nonzero) / float(trimmed_weighted.numel())

    # Elect coordinatewise sign by total signed magnitude; exact zero ties cancel cleanly.
    signed_mass = trimmed_weighted.sum(dim=0)
    elected_sign = torch.sign(signed_mass)

    aligned_mask = nonzero_mask & (torch.sign(trimmed_weighted) == elected_sign.unsqueeze(0)) & (
        elected_sign.unsqueeze(0) != 0.0
    )
    aligned_count = aligned_mask.sum(dim=0)
    aligned_sum = (trimmed_weighted * aligned_mask.to(dtype=torch.float32)).sum(dim=0)
    delta_elected = torch.where(
        aligned_count > 0,
        aligned_sum / aligned_count.clamp(min=1).to(dtype=torch.float32),
        torch.zeros_like(aligned_sum),
    )

    total_aligned = int(aligned_mask.sum().item())
    if total_nonzero > 0:
        conflict_cancelled_fraction = float(total_nonzero - total_aligned) / float(total_nonzero)
    else:
        conflict_cancelled_fraction = 0.0

    delta_projected = delta_elected @ proj
    return TIESProjectionSummary(
        delta_elected=delta_elected,
        delta_projected=delta_projected,
        n_steps=n_steps,
        density=dens,
        trimmed_nonzero_fraction=trimmed_nonzero_fraction,
        sign_conflict_cancelled_fraction=conflict_cancelled_fraction,
        elected_frobenius_norm=float(torch.linalg.norm(delta_elected).item()),
        projected_frobenius_norm=float(torch.linalg.norm(delta_projected).item()),
    )


# ---------------------------------------------------------------------------
# 4. RMSNorm + Token NLL / ECE Evaluation & State Provenance
# ---------------------------------------------------------------------------


def _apply_rmsnorm(hidden: torch.Tensor, norm_weight: torch.Tensor, eps: float) -> torch.Tensor:
    inv_rms = torch.rsqrt(hidden.pow(2).mean(dim=-1, keepdim=True) + eps)
    return hidden * inv_rms * norm_weight.unsqueeze(0)


def compute_nll_and_ece_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> tuple[float, float]:
    """Compute exact token cross-entropy NLL and top-1 Expected Calibration Error (ECE)."""
    log_probs = F.log_softmax(logits, dim=-1)
    nll_mean = float(F.nll_loss(log_probs, targets, reduction="mean").item())
    probs = log_probs.exp()
    confidences, predictions = probs.max(dim=-1)
    accuracies = (predictions == targets).to(dtype=torch.float32)

    edges = torch.linspace(0.0, 1.0, steps=ece_bins + 1, dtype=torch.float32)
    n_total = float(targets.numel())
    ece = 0.0
    for bin_idx in range(ece_bins):
        lo = edges[bin_idx]
        hi = edges[bin_idx + 1]
        if bin_idx == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        count = int(in_bin.sum().item())
        if count > 0:
            avg_conf = float(confidences[in_bin].mean().item())
            avg_acc = float(accuracies[in_bin].mean().item())
            ece += (float(count) / n_total) * abs(avg_acc - avg_conf)
    return nll_mean, float(ece)


def compute_rescue_state_sha256(
    layer_weight: torch.Tensor,
    final_norm_weight: torch.Tensor,
) -> str:
    """Compute deterministic bitwise SHA-256 fingerprint over the mutable weight tensors."""
    w_cpu = layer_weight.detach().contiguous().cpu().to(dtype=torch.float32)
    n_cpu = final_norm_weight.detach().contiguous().cpu().to(dtype=torch.float32)
    payload = {
        "layer_weight_shape": list(w_cpu.shape),
        "layer_weight_hex": w_cpu.numpy().tobytes().hex(),
        "final_norm_shape": list(n_cpu.shape),
        "final_norm_hex": n_cpu.numpy().tobytes().hex(),
    }
    return compute_payload_sha256(payload)


def _evaluate_multi_domain_metrics(
    layer_weight: torch.Tensor,
    final_norm_weight: torch.Tensor,
    lm_head_weight: torch.Tensor,
    domain_hidden_inputs: Mapping[str, torch.Tensor],
    domain_targets: Mapping[str, torch.Tensor],
    *,
    residual_scale: float,
    rmsnorm_eps: float,
    ece_bins: int,
) -> dict[str, Any]:
    per_domain_nll: dict[str, float] = {}
    per_domain_ece: dict[str, float] = {}
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for domain in sorted(domain_hidden_inputs.keys()):
        x_dom = domain_hidden_inputs[domain]
        y_dom = domain_targets[domain]
        h_out = x_dom + residual_scale * (x_dom @ layer_weight.T)
        h_norm = _apply_rmsnorm(h_out, final_norm_weight, rmsnorm_eps)
        logits = h_norm @ lm_head_weight.T
        nll_d, ece_d = compute_nll_and_ece_from_logits(logits, y_dom, ece_bins=ece_bins)
        per_domain_nll[domain] = nll_d
        per_domain_ece[domain] = ece_d
        all_logits.append(logits)
        all_targets.append(y_dom)

    cat_logits = torch.cat(all_logits, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)
    nll_mean, ece_value = compute_nll_and_ece_from_logits(cat_logits, cat_targets, ece_bins=ece_bins)
    return {
        "nll_mean": nll_mean,
        "ece_value": ece_value,
        "per_domain_nll": per_domain_nll,
        "per_domain_ece": per_domain_ece,
    }


def _evaluate_anse_energy(
    base_metrics: Mapping[str, Any],
    cand_metrics: Mapping[str, Any],
    config: SequentialChainRescueConfig,
) -> dict[str, Any]:
    gen_dom = config.general_domain_name
    base_gen = float(base_metrics["per_domain_nll"][gen_dom])
    cand_gen = float(cand_metrics["per_domain_nll"][gen_dom])
    general_drift_ratio = (cand_gen - base_gen) / max(base_gen, 1e-8)

    domain_drifts: dict[str, float] = {}
    for dom, base_val in base_metrics["per_domain_nll"].items():
        cand_val = float(cand_metrics["per_domain_nll"][dom])
        domain_drifts[dom] = (cand_val - float(base_val)) / max(float(base_val), 1e-8)

    worst_domain_drift_ratio = max(domain_drifts.values())
    nll_diff = float(cand_metrics["nll_mean"]) - float(base_metrics["nll_mean"])
    ece_diff = float(cand_metrics["ece_value"]) - float(base_metrics["ece_value"])
    raw_energy = nll_diff + config.ece_weight * ece_diff

    guard_ok = (
        general_drift_ratio <= config.max_general_drift_ratio
        and worst_domain_drift_ratio <= config.max_domain_drift_ratio
    )
    anse_energy = raw_energy if (guard_ok and raw_energy < 0.0) else ANSE_MAXIMUM_PAIN
    return {
        "nll_diff": nll_diff,
        "ece_diff": ece_diff,
        "general_drift_ratio": general_drift_ratio,
        "worst_domain_drift_ratio": worst_domain_drift_ratio,
        "per_domain_drift_ratio": domain_drifts,
        "guard_ok": guard_ok,
        "anse_energy": anse_energy,
    }


# ---------------------------------------------------------------------------
# 5. End-to-End Verified Sequential Chain Rescue with Atomic Rollback
# ---------------------------------------------------------------------------


def execute_verified_chain_rescue(
    mutable_state: dict[str, torch.Tensor],
    lm_head_weight: torch.Tensor,
    chain_lora_factors: Sequence[LoRAChainStep | Mapping[str, Any] | tuple[Any, ...]],
    domain_hidden_inputs: Mapping[str, torch.Tensor],
    domain_targets: Mapping[str, torch.Tensor],
    *,
    p_null: torch.Tensor | None = None,
    donor_lora: LoRAChainStep | Mapping[str, Any] | tuple[Any, ...] | None = None,
    residual_scale: float = 1.0,
    prose_nullspace_variance: float = 0.999,
    step_weights: Sequence[float] | None = None,
    config: SequentialChainRescueConfig | None = None,
) -> dict[str, Any]:
    """Execute verified sequential QLoRA chain rescue with ANSE gating and atomic bitwise rollback.

    Mutates ``mutable_state["layer_weight"]`` and ``mutable_state["final_norm_weight"]`` in-place
    when the rescued candidate strictly satisfies ``ANSE Delta E < 0`` and ``D0 / worst_domain <= guard``.
    Otherwise restores exact bitwise initial tensors (``final_state_sha256 == initial_state_sha256``)
    and assigns ``anse_energy = ANSE_MAXIMUM_PAIN (1e6)``.
    """
    cfg = config if config is not None else SequentialChainRescueConfig()
    if not isinstance(mutable_state, dict):
        raise TypeError("mutable_state must be a dict containing 'layer_weight' and 'final_norm_weight'")
    if "layer_weight" not in mutable_state or "final_norm_weight" not in mutable_state:
        raise ValueError("mutable_state must contain 'layer_weight' and 'final_norm_weight'")

    w_base = _validate_2d_tensor(mutable_state["layer_weight"], "mutable_state['layer_weight']")
    norm_base = _validate_1d_tensor(
        mutable_state["final_norm_weight"], "mutable_state['final_norm_weight']"
    )
    head_w = _validate_2d_tensor(lm_head_weight, "lm_head_weight")
    d_out, d_in = int(w_base.shape[0]), int(w_base.shape[1])
    if d_out != d_in:
        raise ValueError(f"layer_weight must be square residual projection, got {tuple(w_base.shape)}")
    if norm_base.shape[0] != d_out or head_w.shape[1] != d_out:
        raise ValueError(
            f"Dimension mismatch: layer_weight={tuple(w_base.shape)}, "
            f"final_norm={tuple(norm_base.shape)}, lm_head={tuple(head_w.shape)}"
        )
    res_scale = _validate_finite_float(residual_scale, "residual_scale")

    if not isinstance(domain_hidden_inputs, Mapping) or len(domain_hidden_inputs) == 0:
        raise ValueError("domain_hidden_inputs must be a non-empty mapping")
    if cfg.general_domain_name not in domain_hidden_inputs:
        raise ValueError(f"general_domain_name '{cfg.general_domain_name}' missing from domain_hidden_inputs")
    if set(domain_hidden_inputs.keys()) != set(domain_targets.keys()):
        raise ValueError("domain_hidden_inputs and domain_targets must have identical domain keys")

    clean_inputs: dict[str, torch.Tensor] = {}
    clean_targets: dict[str, torch.Tensor] = {}
    vocab_size = int(head_w.shape[0])
    for dom in sorted(domain_hidden_inputs.keys()):
        x_dom = _validate_2d_tensor(domain_hidden_inputs[dom], f"domain_hidden_inputs[{dom!r}]")
        if x_dom.shape[1] != d_in:
            raise ValueError(f"domain {dom!r} input dim {x_dom.shape[1]} != {d_in}")
        t_raw = domain_targets[dom]
        if not isinstance(t_raw, torch.Tensor) or t_raw.ndim != 1 or t_raw.shape[0] != x_dom.shape[0]:
            raise ValueError(f"domain {dom!r} targets must be 1D with length {x_dom.shape[0]}")
        t_long = t_raw.detach().to(dtype=torch.long, device="cpu")
        if int(t_long.min().item()) < 0 or int(t_long.max().item()) >= vocab_size:
            raise ValueError(f"domain {dom!r} target indices out of range [0, {vocab_size})")
        clean_inputs[dom] = x_dom
        clean_targets[dom] = t_long

    # Snapshot initial state for exact bitwise rollback guarantee.
    w_backup = mutable_state["layer_weight"].detach().clone()
    norm_backup = mutable_state["final_norm_weight"].detach().clone()
    initial_state_sha256 = compute_rescue_state_sha256(w_backup, norm_backup)

    # Resolve or synthesize orthonormal prose null-space projector P_0 from general prose inputs.
    if p_null is not None:
        proj = _validate_square_projector(p_null, d_in, "p_null")
    else:
        prose_x = clean_inputs[cfg.general_domain_name]
        proj = compute_prose_nullspace_projector(
            prose_x,
            variance_threshold=prose_nullspace_variance,
        )

    # 1. Baseline & unguarded raw final chain metrics.
    base_metrics = _evaluate_multi_domain_metrics(
        w_base,
        norm_base,
        head_w,
        clean_inputs,
        clean_targets,
        residual_scale=res_scale,
        rmsnorm_eps=cfg.rmsnorm_eps,
        ece_bins=cfg.ece_bins,
    )

    last_step = _normalize_chain_step(chain_lora_factors[-1], len(chain_lora_factors) - 1)
    raw_final_delta = (last_step.b_mat.detach().float() * last_step.scale) @ last_step.a_mat.detach().float()
    raw_chain_metrics = _evaluate_multi_domain_metrics(
        w_base + raw_final_delta,
        norm_base,
        head_w,
        clean_inputs,
        clean_targets,
        residual_scale=res_scale,
        rmsnorm_eps=cfg.rmsnorm_eps,
        ece_bins=cfg.ece_bins,
    )
    raw_chain_anse = _evaluate_anse_energy(base_metrics, raw_chain_metrics, cfg)

    # 2. Extract incremental chain task vectors directly via low-RSS associative P_0 projection.
    increments = extract_incremental_chain_task_vectors(chain_lora_factors, p_null=proj)
    ties_summary = ties_elect_and_project_increments(
        increments,
        proj,
        density=cfg.ties_density,
        step_weights=step_weights,
    )
    delta_ties_p0 = ties_summary.delta_projected

    # 3. Optional peak donor LoRA projected via associative factorized null-space product.
    if donor_lora is not None:
        donor_step = _normalize_chain_step(donor_lora, 0)
        donor_delta_p0 = factorized_associative_nullspace_delta(
            donor_step.b_mat.detach().float() * donor_step.scale,
            donor_step.a_mat.detach().float(),
            proj,
        )
        donor_alphas = cfg.donor_alpha_grid
    else:
        donor_delta_p0 = torch.zeros_like(w_base)
        donor_alphas = (0.0,)

    # 4. Deterministic grid search over (donor_alpha, ties_alpha, safe tau_temp >= 1.0).
    best_candidate: dict[str, Any] | None = None
    best_energy = ANSE_MAXIMUM_PAIN
    best_weights: tuple[torch.Tensor, torch.Tensor] | None = None

    for d_alpha in donor_alphas:
        for t_alpha in cfg.ties_alpha_grid:
            delta_total = d_alpha * donor_delta_p0 + t_alpha * delta_ties_p0
            w_cand = w_base + delta_total
            for tau_temp in cfg.safe_tau_grid:
                norm_cand = norm_base / tau_temp
                cand_metrics = _evaluate_multi_domain_metrics(
                    w_cand,
                    norm_cand,
                    head_w,
                    clean_inputs,
                    clean_targets,
                    residual_scale=res_scale,
                    rmsnorm_eps=cfg.rmsnorm_eps,
                    ece_bins=cfg.ece_bins,
                )
                anse_eval = _evaluate_anse_energy(base_metrics, cand_metrics, cfg)
                if anse_eval["guard_ok"] and anse_eval["anse_energy"] < best_energy:
                    prose_x = clean_inputs[cfg.general_domain_name]
                    prose_leak = float((prose_x @ delta_total.T).abs().max().item())
                    best_energy = float(anse_eval["anse_energy"])
                    best_weights = (w_cand, norm_cand)
                    best_candidate = {
                        "donor_alpha": float(d_alpha),
                        "ties_alpha": float(t_alpha),
                        "tau_temp": float(tau_temp),
                        "metrics": cand_metrics,
                        "anse": anse_eval,
                        "prose_invariance_max_diff": prose_leak,
                    }

    # 5. Monotone ANSE commit OR atomic bitwise rollback.
    if best_candidate is not None and best_weights is not None and best_energy < 0.0:
        w_best, norm_best = best_weights
        mutable_state["layer_weight"] = w_best.to(dtype=w_backup.dtype, device=w_backup.device)
        mutable_state["final_norm_weight"] = norm_best.to(
            dtype=norm_backup.dtype, device=norm_backup.device
        )
        accepted = True
        rolled_back = False
        final_metrics = best_candidate["metrics"]
        final_anse = best_candidate["anse"]
        selected_donor_alpha = best_candidate["donor_alpha"]
        selected_ties_alpha = best_candidate["ties_alpha"]
        selected_tau_temp = best_candidate["tau_temp"]
        prose_invariance_max_diff = best_candidate["prose_invariance_max_diff"]
    else:
        mutable_state["layer_weight"] = w_backup
        mutable_state["final_norm_weight"] = norm_backup
        accepted = False
        rolled_back = True
        final_metrics = base_metrics
        final_anse = {
            "nll_diff": 0.0,
            "ece_diff": 0.0,
            "general_drift_ratio": 0.0,
            "worst_domain_drift_ratio": 0.0,
            "per_domain_drift_ratio": {dom: 0.0 for dom in clean_inputs},
            "guard_ok": False,
            "anse_energy": ANSE_MAXIMUM_PAIN,
        }
        selected_donor_alpha = 0.0
        selected_ties_alpha = 0.0
        selected_tau_temp = 1.0
        prose_invariance_max_diff = 0.0

    final_state_sha256 = compute_rescue_state_sha256(
        mutable_state["layer_weight"], mutable_state["final_norm_weight"]
    )
    rollback_verified = (final_state_sha256 == initial_state_sha256) if rolled_back else True

    receipt_body: dict[str, Any] = {
        "schema": RESCUER_SCHEMA,
        "object": RESCUER_RECEIPT_OBJECT,
        "accepted": accepted,
        "rolled_back": rolled_back,
        "rollback_verified": rollback_verified,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
        "chain_length": len(increments),
        "ties_density": ties_summary.density,
        "sign_conflict_cancelled_fraction": ties_summary.sign_conflict_cancelled_fraction,
        "trimmed_nonzero_fraction": ties_summary.trimmed_nonzero_fraction,
        "selected_donor_alpha": selected_donor_alpha,
        "selected_ties_alpha": selected_ties_alpha,
        "selected_tau_temp": selected_tau_temp,
        "prose_invariance_max_diff": prose_invariance_max_diff,
        "base_nll_mean": float(base_metrics["nll_mean"]),
        "base_ece_value": float(base_metrics["ece_value"]),
        "raw_chain_nll_mean": float(raw_chain_metrics["nll_mean"]),
        "raw_chain_general_drift_ratio": float(raw_chain_anse["general_drift_ratio"]),
        "raw_chain_guard_ok": bool(raw_chain_anse["guard_ok"]),
        "nll_mean": float(final_metrics["nll_mean"]),
        "ece_value": float(final_metrics["ece_value"]),
        "nll_diff": float(final_anse["nll_diff"]),
        "ece_diff": float(final_anse["ece_diff"]),
        "general_drift_ratio": float(final_anse["general_drift_ratio"]),
        "worst_domain_drift_ratio": float(final_anse["worst_domain_drift_ratio"]),
        "per_domain_drift_ratio": {
            k: float(v) for k, v in sorted(final_anse["per_domain_drift_ratio"].items())
        },
        "anse_energy": float(final_anse["anse_energy"]),
        "config": asdict(cfg),
    }
    receipt_body["receipt_sha256"] = compute_payload_sha256(receipt_body)
    return receipt_body


def verify_chain_rescue_receipt_provenance(receipt: Mapping[str, Any]) -> bool:
    """Verify canonical SHA-256 provenance and structural invariants of a rescue receipt."""
    if not isinstance(receipt, Mapping):
        return False
    if receipt.get("schema") != RESCUER_SCHEMA or receipt.get("object") != RESCUER_RECEIPT_OBJECT:
        return False
    digest = receipt.get("receipt_sha256")
    if not isinstance(digest, str) or _SHA256_HEX_RE.fullmatch(digest) is None:
        return False
    init_sha = receipt.get("initial_state_sha256")
    final_sha = receipt.get("final_state_sha256")
    if not isinstance(init_sha, str) or _SHA256_HEX_RE.fullmatch(init_sha) is None:
        return False
    if not isinstance(final_sha, str) or _SHA256_HEX_RE.fullmatch(final_sha) is None:
        return False
    if receipt.get("rolled_back") is True and init_sha != final_sha:
        return False
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    return compute_payload_sha256(body) == digest

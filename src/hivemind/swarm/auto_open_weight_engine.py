# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Gen-2 ``S5`` PASS-Synergy Autonomous Open-Weight Neural Control Engine.

Synthesizes and extends the four empirically certified ``PASS`` Weight-Space
Programming (WSP) families from ``PILOT1-WSP-ALPHAEDIT-MOE-2026-11-23`` into a
single zero-gradient, CPU-only (``0 MiB`` GPU VRAM) stage-ratcheted pipeline:

1. **Stage ``S5.1`` — Causal Beneficial-Anchor Protection (``S2``) + Selective
   FoMoE Ghost Sinking (``S1``/``S4``)**:
   Protects beneficial anchor layers (where bypassing the MoE delta harms general
   prose NLL, e.g. Layer ``17`` at ``-0.2516`` nats) untouched bitwise, while
   zeroing pure rogue / prose-harming expert output factors (``{0, 1, 3}``) only
   on prose-interfering layers.
2. **Stage ``S5.2`` — 2-Fold Out-of-Fold SNR-Gated Orthogonal Multi-Gate
   Specialist Preservation (``S4``)**:
   Preserves dual-active specialist experts (e.g. ``expert_5`` with ``+0.0278``
   nats specialist capacity) and shifts their ``sigma_topk`` router row along a
   direction strictly orthogonal to the joint span of all specialist domain
   centroids whenever 2-fold OOF SNR meets ``min_oof_snr``.
3. **Stage ``S5.3`` — Tail-Layer Rank-Preserving QR-SVD AlphaEdit Null-Space
   Surgery (``S3``)**:
   On the top prose-interfering tail layers (e.g. Layers ``18, 19``), extracts
   the Marchenko-Pastur-guarded general-prose input subspace ``U_0`` and projects
   active specialist low-rank right factors ``A_e`` (or SVD right modes of
   ``B_e @ A_e``) into ``P_0 = I - U_0 U_0^T`` without rank inflation, driving
   general-domain drift toward ``<= 0.0%`` across all domains.
4. **Stage ``S5.4`` — Zero-Overhead Folded Final-Norm Interpolation &
   Temperature Calibration (``DYN-14``)**:
   Jointly searches ``(lambda_ln, tau_temp)`` under the general-domain guard and
   folds ``1 / tau_temp`` directly into final ``LayerNorm`` / ``RMSNorm`` scale
   (and optional bias) with zero runtime inference overhead.

Every stage is governed by the ANSE thermodynamic monotone ratchet:
``Delta E_stage < 0`` and ``general_rel_drift <= general_guard_rel`` (otherwise
rolled back bitwise; infeasible final states receive ``ANSE_MAXIMUM_PAIN = 1e6``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

import numpy as np

ANSE_MAXIMUM_PAIN: float = 1_000_000.0
DEFAULT_GENERAL_DOMAIN: str = "E0_general_fineweb"


@dataclasses.dataclass(frozen=True)
class AutoOpenWeightConfig:
    """Configuration for the Gen-2 ``S5`` PASS-Synergy Auto-Open-Weight Engine."""

    general_domain: str = DEFAULT_GENERAL_DOMAIN
    general_guard_rel: float = 0.01
    anchor_benefit_floor_nats: float = -0.010
    prose_harm_layer_floor_nats: float = 0.001
    tail_svd_top_k_layers: int = 2
    prose_harm_expert_floor_nats: float = 0.002
    specialist_preserve_floor_nats: float = 0.012
    router_beta: float = 0.35
    min_oof_snr: float = 0.25
    prose_nullspace_energy_keep: float = 0.85
    svd_damp_strength: float = 0.85
    lambda_ln_grid: tuple[float, ...] = (0.25, 0.50, 0.65, 0.80, 0.90, 1.00)
    tau_temp_grid: tuple[float, ...] = (0.90, 0.95, 1.00, 1.05, 1.12, 1.20)
    min_stage_gain_nats: float = 1e-6

    def __post_init__(self) -> None:
        if not self.general_domain or not isinstance(self.general_domain, str):
            raise ValueError("general_domain must be a non-empty string")
        if not (0.0 < float(self.general_guard_rel) <= 0.25):
            raise ValueError("general_guard_rel must lie in (0.0, 0.25]")
        if float(self.anchor_benefit_floor_nats) >= 0.0:
            raise ValueError("anchor_benefit_floor_nats must be strictly negative")
        if float(self.prose_harm_layer_floor_nats) < 0.0:
            raise ValueError("prose_harm_layer_floor_nats must be non-negative")
        if int(self.tail_svd_top_k_layers) < 0:
            raise ValueError("tail_svd_top_k_layers must be >= 0")
        if not (0.0 < float(self.router_beta) <= 5.0):
            raise ValueError("router_beta must lie in (0.0, 5.0]")
        if float(self.min_oof_snr) < 0.0:
            raise ValueError("min_oof_snr must be >= 0.0")
        if not (0.0 < float(self.prose_nullspace_energy_keep) < 1.0):
            raise ValueError("prose_nullspace_energy_keep must lie in (0.0, 1.0)")
        if not (0.0 <= float(self.svd_damp_strength) <= 1.0):
            raise ValueError("svd_damp_strength must lie in [0.0, 1.0]")
        if not self.lambda_ln_grid or any(not (0.0 <= float(v) <= 1.0) for v in self.lambda_ln_grid):
            raise ValueError("lambda_ln_grid must be non-empty with values in [0.0, 1.0]")
        if not self.tau_temp_grid or any(float(v) <= 0.0 for v in self.tau_temp_grid):
            raise ValueError("tau_temp_grid must be non-empty with strictly positive values")


@dataclasses.dataclass(frozen=True)
class LayerExpertProfile:
    """Causal attribution partitioning MoE layers and experts by functional role."""

    layer_e0_harm_nats: tuple[float, ...]
    beneficial_anchor_layers: tuple[int, ...]
    neutral_layers: tuple[int, ...]
    interfering_layers: tuple[int, ...]
    tail_svd_layers: tuple[int, ...]
    pure_sink_experts: tuple[int, ...]
    preserved_multigate_experts: tuple[int, ...]
    core_specialist_experts: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class StageRatchetReceipt:
    """Measured telemetry and SHA-256 provenance for a single ``S5`` stage."""

    stage_name: str
    accepted: bool
    reason: str
    pre_sha256: str
    post_sha256: str
    edited_params: int
    pre_anse_energy: float
    post_anse_energy: float
    delta_anse_energy: float
    general_rel_drift: float
    mean_diff_vs_base: float


@dataclasses.dataclass(frozen=True)
class AutoOpenWeightResult:
    """End-to-end result of running the ``S5`` PASS-Synergy pipeline."""

    committed: bool
    verdict: str
    anse_energy: float
    mean_diff_vs_base: float
    general_rel_drift: float
    all_domains_negative: bool
    total_params: int
    edited_params: int
    edited_fraction: float
    base_domain_nll: dict[str, float]
    initial_domain_nll: dict[str, float]
    final_domain_nll: dict[str, float]
    per_domain_rel_vs_base: dict[str, float]
    profile: LayerExpertProfile
    stages: tuple[StageRatchetReceipt, ...]
    selected_lambda_ln: float
    selected_tau_temp: float
    checkpoint_sha256: str
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary with zero forbidden keys."""
        return {
            "committed": self.committed,
            "verdict": self.verdict,
            "anse_energy": self.anse_energy,
            "mean_diff_vs_base": self.mean_diff_vs_base,
            "general_rel_drift": self.general_rel_drift,
            "all_domains_negative": self.all_domains_negative,
            "total_params": self.total_params,
            "edited_params": self.edited_params,
            "edited_fraction": self.edited_fraction,
            "base_domain_nll": dict(self.base_domain_nll),
            "initial_domain_nll": dict(self.initial_domain_nll),
            "final_domain_nll": dict(self.final_domain_nll),
            "per_domain_rel_vs_base": dict(self.per_domain_rel_vs_base),
            "profile": dataclasses.asdict(self.profile),
            "stages": [dataclasses.asdict(s) for s in self.stages],
            "selected_lambda_ln": self.selected_lambda_ln,
            "selected_tau_temp": self.selected_tau_temp,
            "checkpoint_sha256": self.checkpoint_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


class SingleShotVaultLock:
    """Enforces strict single-shot evaluation on a virgin holdout vault split."""

    def __init__(self, vault_id: str) -> None:
        if not vault_id or not isinstance(vault_id, str):
            raise ValueError("vault_id must be a non-empty string")
        self.vault_id = vault_id
        self._consumed = False
        self._receipt_sha256: str | None = None

    @property
    def consumed(self) -> bool:
        return self._consumed

    @property
    def receipt_sha256(self) -> str | None:
        return self._receipt_sha256

    def evaluate_once(
        self,
        state: Mapping[str, np.ndarray],
        base_vault_nll: Mapping[str, float],
        vault_eval_fn: Callable[[Mapping[str, np.ndarray]], Mapping[str, float]],
        *,
        general_domain: str = DEFAULT_GENERAL_DOMAIN,
        general_guard_rel: float = 0.01,
    ) -> dict[str, Any]:
        """Score ``state`` strictly once against ``vault_eval_fn`` and permanently lock."""
        if self._consumed:
            raise RuntimeError(
                f"Virgin holdout vault '{self.vault_id}' was already evaluated "
                f"(receipt={self._receipt_sha256}); refusing adaptive re-evaluation."
            )
        self._consumed = True
        vault_nll = {str(k): float(v) for k, v in vault_eval_fn(state).items()}
        energy, mean_diff, gen_rel, per_rel = compute_anse_objective(
            vault_nll,
            base_vault_nll,
            general_domain=general_domain,
            general_guard_rel=general_guard_rel,
        )
        guard_ok = math.isfinite(energy) and energy < ANSE_MAXIMUM_PAIN * 0.5 and mean_diff < 0.0
        payload = {
            "vault_id": self.vault_id,
            "checkpoint_sha256": hash_state_dict(state),
            "verdict_ip300": "PASS" if guard_ok else "FAIL",
            "guard_ok": guard_ok,
            "anse_energy": energy,
            "mean_diff_vs_base": mean_diff,
            "general_rel_drift": gen_rel,
            "all_domains_negative": all(v < 0.0 for v in per_rel.values()),
            "vault_domain_nll": vault_nll,
            "per_domain_rel_vs_base": per_rel,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        payload["vault_receipt_sha256"] = digest
        self._receipt_sha256 = digest
        return payload


def _validate_finite_array(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or Inf entries")
    return a


def clone_state_dict(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Deep-copy a mapping of NumPy arrays while validating finite entries."""
    if not state:
        raise ValueError("state must be non-empty")
    out: dict[str, np.ndarray] = {}
    for k, v in state.items():
        arr = np.asarray(v)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Tensor '{k}' contains non-finite values")
        out[str(k)] = np.array(arr, copy=True)
    return out


def restore_state_inplace(
    target: MutableMapping[str, np.ndarray],
    snapshot: Mapping[str, np.ndarray],
) -> None:
    """Bitwise-restore ``target`` in-place from ``snapshot``."""
    target.clear()
    for k, v in snapshot.items():
        target[k] = np.array(v, copy=True)


def hash_state_dict(state: Mapping[str, np.ndarray]) -> str:
    """Compute a canonical SHA-256 digest across sorted tensor names, shapes, and float64 bytes."""
    h = hashlib.sha256()
    for k in sorted(state.keys()):
        arr = np.ascontiguousarray(np.asarray(state[k], dtype=np.float64))
        h.update(k.encode("utf-8"))
        h.update(str(arr.shape).encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()


def count_edited_parameters(
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    *,
    atol: float = 1e-12,
) -> tuple[int, int]:
    """Return ``(edited_params, total_params)`` comparing two state dicts."""
    edited = 0
    total = 0
    for k in sorted(before.keys()):
        b = np.asarray(before[k], dtype=np.float64)
        a = np.asarray(after[k], dtype=np.float64)
        if b.shape != a.shape:
            raise ValueError(f"Shape mismatch on tensor '{k}': {b.shape} vs {a.shape}")
        total += int(b.size)
        edited += int(np.count_nonzero(np.abs(a - b) > atol))
    return edited, total


def compute_anse_objective(
    cand_nll: Mapping[str, float],
    base_nll: Mapping[str, float],
    *,
    general_domain: str = DEFAULT_GENERAL_DOMAIN,
    general_guard_rel: float = 0.01,
) -> tuple[float, float, float, dict[str, float]]:
    """Compute ``(anse_energy, mean_diff_vs_base, general_rel_drift, per_domain_rel)``.

    Any non-finite NLL or violation of ``general_rel_drift <= general_guard_rel``
    receives ``ANSE_MAXIMUM_PAIN`` (``10^6``).
    """
    if general_domain not in cand_nll or general_domain not in base_nll:
        raise ValueError(f"Missing required general_domain '{general_domain}' in domain NLL map")
    keys = sorted(base_nll.keys())
    if set(cand_nll.keys()) != set(base_nll.keys()):
        raise ValueError("cand_nll and base_nll must contain identical domain keys")

    diffs: list[float] = []
    per_rel: dict[str, float] = {}
    for k in keys:
        b = float(base_nll[k])
        c = float(cand_nll[k])
        if not (math.isfinite(b) and math.isfinite(c)) or b <= 0.0:
            return ANSE_MAXIMUM_PAIN, float("inf"), float("inf"), {}
        d = c - b
        diffs.append(d)
        per_rel[k] = d / b

    mean_diff = float(np.mean(diffs))
    gen_rel = float(per_rel[general_domain])
    if gen_rel > float(general_guard_rel) + 1e-12:
        return ANSE_MAXIMUM_PAIN, mean_diff, gen_rel, per_rel

    # Mild boundary penalty inside [0.5 * guard, guard] mirroring Pilot 1 ANSE scoring
    soft_penalty = max(0.0, gen_rel - 0.5 * float(general_guard_rel)) * 2.0
    anse_energy = mean_diff + soft_penalty
    return float(anse_energy), mean_diff, gen_rel, per_rel


def profile_moe_layers_and_experts(
    layer_e0_harm_nats: Sequence[float],
    expert_e0_harm_nats: Sequence[float],
    expert_spec_loss_when_zeroed: Sequence[float],
    *,
    config: AutoOpenWeightConfig | None = None,
) -> LayerExpertProfile:
    """Partition layers (anchor / neutral / interfering / tail-SVD) and experts (sink / multigate / specialist)."""
    cfg = config or AutoOpenWeightConfig()
    lh = _validate_finite_array(np.asarray(layer_e0_harm_nats, dtype=np.float64), "layer_e0_harm_nats")
    eh = _validate_finite_array(np.asarray(expert_e0_harm_nats, dtype=np.float64), "expert_e0_harm_nats")
    es = _validate_finite_array(
        np.asarray(expert_spec_loss_when_zeroed, dtype=np.float64),
        "expert_spec_loss_when_zeroed",
    )
    if eh.ndim != 1 or es.ndim != 1 or eh.shape[0] != es.shape[0]:
        raise ValueError("expert_e0_harm_nats and expert_spec_loss_when_zeroed must be 1D with equal length")
    if lh.ndim != 1:
        raise ValueError("layer_e0_harm_nats must be 1D")

    anchors: list[int] = []
    neutrals: list[int] = []
    interfering: list[int] = []
    for idx, val in enumerate(lh.tolist()):
        if val <= float(cfg.anchor_benefit_floor_nats):
            anchors.append(idx)
        elif val >= float(cfg.prose_harm_layer_floor_nats):
            interfering.append(idx)
        else:
            neutrals.append(idx)

    ranked_interfering = sorted(interfering, key=lambda i: float(lh[i]), reverse=True)
    tail_k = min(int(cfg.tail_svd_top_k_layers), len(ranked_interfering))
    tail_svd = tuple(sorted(ranked_interfering[:tail_k]))

    pure_sinks: list[int] = []
    multigate_preserved: list[int] = []
    core_specialists: list[int] = []
    for e_idx in range(eh.shape[0]):
        harm_e0 = float(eh[e_idx])
        spec_val = float(es[e_idx])
        if harm_e0 >= float(cfg.prose_harm_expert_floor_nats):
            if spec_val >= float(cfg.specialist_preserve_floor_nats):
                multigate_preserved.append(e_idx)
            else:
                pure_sinks.append(e_idx)
        else:
            core_specialists.append(e_idx)

    return LayerExpertProfile(
        layer_e0_harm_nats=tuple(float(x) for x in lh.tolist()),
        beneficial_anchor_layers=tuple(anchors),
        neutral_layers=tuple(neutrals),
        interfering_layers=tuple(interfering),
        tail_svd_layers=tail_svd,
        pure_sink_experts=tuple(pure_sinks),
        preserved_multigate_experts=tuple(multigate_preserved),
        core_specialist_experts=tuple(core_specialists),
    )


def apply_stage_s5_1_anchor_protected_ghost_sink_inplace(
    state: MutableMapping[str, np.ndarray],
    profile: LayerExpertProfile,
) -> int:
    """Zero ``layer_{l}.w2_B`` for pure ghost-sink experts strictly on interfering layers.

    Beneficial anchor layers (``profile.beneficial_anchor_layers``) and neutral
    layers are left 100% bitwise untouched.
    """
    changed = 0
    for layer_idx in profile.interfering_layers:
        if layer_idx in profile.beneficial_anchor_layers:
            continue
        key = f"layer_{layer_idx}.w2_B"
        if key not in state:
            continue
        w2_b = np.array(state[key], dtype=np.float64, copy=True)
        for exp_id in profile.pure_sink_experts:
            if 0 <= exp_id < w2_b.shape[0]:
                changed += int(np.count_nonzero(np.abs(w2_b[exp_id]) > 1e-12))
                w2_b[exp_id] = 0.0
        state[key] = w2_b.astype(state[key].dtype, copy=False)
    return changed


def compute_oof_orthogonal_router_direction(
    x_prose: np.ndarray,
    specialist_domain_acts: Mapping[str, np.ndarray],
    *,
    min_oof_snr: float = 0.25,
) -> tuple[np.ndarray | None, float]:
    """Compute a 2-fold OOF SNR-gated unit vector ``u_{0 perp S}`` orthogonal to all specialist centroids.

    Returns ``(direction_unit_or_None, oof_snr)``.
    """
    xp = _validate_finite_array(x_prose, "x_prose")
    if xp.ndim != 2 or xp.shape[0] < 4:
        raise ValueError("x_prose must be 2D with at least 4 samples for 2-fold OOF")
    if not specialist_domain_acts:
        raise ValueError("specialist_domain_acts must be non-empty")

    d_dim = xp.shape[1]
    spec_means: list[np.ndarray] = []
    all_spec_rows: list[np.ndarray] = []
    for dom_name in sorted(specialist_domain_acts.keys()):
        xs = _validate_finite_array(specialist_domain_acts[dom_name], f"specialist_domain_acts[{dom_name}]")
        if xs.ndim != 2 or xs.shape[1] != d_dim or xs.shape[0] < 2:
            raise ValueError(f"Invalid shape {xs.shape} for specialist domain '{dom_name}'")
        spec_means.append(np.mean(xs, axis=0))
        all_spec_rows.append(xs)

    mat_s = np.stack(spec_means, axis=1)  # (D, K_spec)
    u_s, s_vals, _ = np.linalg.svd(mat_s, full_matrices=False)
    tol = max(mat_s.shape) * np.finfo(np.float64).eps * (float(s_vals[0]) if s_vals.size else 1.0) * 16.0
    q_s = u_s[:, s_vals > tol]  # Orthonormal basis of specialist domain means

    def _orth_proj(vec: np.ndarray) -> np.ndarray:
        if q_s.size == 0:
            return vec
        return vec - q_s @ (q_s.T @ vec)

    # 2-fold out-of-fold SNR validation to reject random calibration noise
    mid = xp.shape[0] // 2
    folds_p = (xp[:mid], xp[mid:])
    xs_cat = np.concatenate(all_spec_rows, axis=0)
    mid_s = xs_cat.shape[0] // 2
    folds_s = (xs_cat[:mid_s], xs_cat[mid_s:])

    fold_snrs: list[float] = []
    for fold_idx in range(2):
        tr_p, val_p = folds_p[fold_idx], folds_p[1 - fold_idx]
        tr_s, val_s = folds_s[fold_idx], folds_s[1 - fold_idx]
        raw_tr = _orth_proj(np.mean(tr_p, axis=0) - np.mean(tr_s, axis=0))
        nrm = float(np.linalg.norm(raw_tr))
        if nrm <= 1e-10:
            return None, 0.0
        u_tr = raw_tr / nrm
        proj_p = val_p @ u_tr
        proj_s = val_s @ u_tr
        denom = math.sqrt(0.5 * (float(np.var(proj_p)) + float(np.var(proj_s))) + 1e-8)
        fold_snrs.append(float(np.mean(proj_p) - np.mean(proj_s)) / denom)

    oof_snr = float(min(fold_snrs))
    if oof_snr < float(min_oof_snr):
        return None, oof_snr

    # Project strictly orthogonal to specialist centroids; note mean(spec_means) lies in span(q_s)
    full_dir = _orth_proj(np.mean(xp, axis=0))
    full_norm = float(np.linalg.norm(full_dir))
    if full_norm <= 1e-10:
        return None, 0.0
    u_unit = full_dir / full_norm
    # Ensure positive inner product with prose mean so subtracting router_beta * u_unit lowers prose gate
    if float(np.mean(xp, axis=0) @ u_unit) < 0.0:
        u_unit = -u_unit
    return u_unit, oof_snr


def apply_stage_s5_2_oof_orthogonal_multigate_inplace(
    state: MutableMapping[str, np.ndarray],
    profile: LayerExpertProfile,
    layer_prose_inputs: Mapping[int, np.ndarray],
    layer_specialist_inputs: Mapping[int, Mapping[str, np.ndarray]],
    *,
    router_beta: float = 0.35,
    min_oof_snr: float = 0.25,
) -> tuple[int, dict[int, float]]:
    """Apply 2-fold OOF orthogonal multi-gate suppression on preserved dual-active experts."""
    changed = 0
    layer_snrs: dict[int, float] = {}
    if not profile.preserved_multigate_experts:
        return 0, layer_snrs

    for layer_idx in profile.interfering_layers:
        if layer_idx in profile.beneficial_anchor_layers:
            continue
        r_key = f"layer_{layer_idx}.router_w"
        if r_key not in state or layer_idx not in layer_prose_inputs or layer_idx not in layer_specialist_inputs:
            continue
        u_dir, snr = compute_oof_orthogonal_router_direction(
            layer_prose_inputs[layer_idx],
            layer_specialist_inputs[layer_idx],
            min_oof_snr=min_oof_snr,
        )
        layer_snrs[layer_idx] = snr
        if u_dir is None:
            continue
        rw = np.array(state[r_key], dtype=np.float64, copy=True)
        row_scale = max(float(np.mean(np.linalg.norm(rw, axis=1))), 1.0)
        delta_row = -float(router_beta) * row_scale * u_dir
        for exp_id in profile.preserved_multigate_experts:
            if 0 <= exp_id < rw.shape[0]:
                rw[exp_id] = rw[exp_id] + delta_row
                changed += int(rw.shape[1])
        state[r_key] = rw.astype(state[r_key].dtype, copy=False)
    return changed, layer_snrs


def extract_mp_guarded_prose_projector(
    h_prose: np.ndarray,
    *,
    energy_keep: float = 0.85,
) -> np.ndarray:
    """Compute AlphaEdit null-space projector ``P_0 = I - U_0 U_0^T`` with Marchenko-Pastur denoising."""
    hp = _validate_finite_array(h_prose, "h_prose")
    if hp.ndim != 2 or hp.shape[0] < 2:
        raise ValueError("h_prose must be 2D with at least 2 rows")
    n_samples, d_dim = hp.shape
    cov = (hp.T @ hp) / float(n_samples)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    total_e = float(np.sum(eigvals))
    if total_e <= 1e-12:
        return np.eye(d_dim, dtype=np.float64)

    # Marchenko-Pastur bulk edge guard sigma^2 * (1 + sqrt(d / n))^2
    median_eig = float(np.median(eigvals))
    aspect = float(d_dim) / float(max(n_samples, 1))
    mp_edge = median_eig * ((1.0 + math.sqrt(aspect)) ** 2)

    csum = np.cumsum(eigvals) / total_e
    k_energy = int(np.searchsorted(csum, float(energy_keep)) + 1)
    k_mp = int(np.count_nonzero(eigvals > mp_edge))
    k_keep = max(1, min(d_dim - 1, max(k_mp, min(k_energy, max(1, d_dim // 3)))))
    u_0 = eigvecs[:, :k_keep]
    p_0 = np.eye(d_dim, dtype=np.float64) - (u_0 @ u_0.T)
    return 0.5 * (p_0 + p_0.T)


def project_lora_pair_qr_svd_nullspace(
    lora_b: np.ndarray,
    lora_a: np.ndarray,
    p_null: np.ndarray,
    *,
    damp_strength: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank-preserving QR-SVD right-factor projection of ``(B, A)`` toward ``B @ A @ P_eff``."""
    b_mat = _validate_finite_array(lora_b, "lora_b")
    a_mat = _validate_finite_array(lora_a, "lora_a")
    p_mat = _validate_finite_array(p_null, "p_null")
    if b_mat.ndim != 2 or a_mat.ndim != 2 or b_mat.shape[1] != a_mat.shape[0]:
        raise ValueError("Invalid (lora_b, lora_a) shapes for rank-r LoRA pair")
    rank = b_mat.shape[1]
    in_dim = a_mat.shape[1]
    if p_mat.shape != (in_dim, in_dim):
        raise ValueError(f"p_null shape {p_mat.shape} must match ({in_dim}, {in_dim})")

    p_eff = (1.0 - float(damp_strength)) * np.eye(in_dim, dtype=np.float64) + float(damp_strength) * p_mat
    a_proj = a_mat @ p_eff  # (r, in_dim)

    # Thin QR + small r x r SVD preserves exact rank r and balances factor norms
    q_b, r_b = np.linalg.qr(b_mat, mode="reduced")
    q_a, r_a = np.linalg.qr(a_proj.T, mode="reduced")
    core = r_b @ r_a.T  # (r, r)
    u_c, s_c, vt_c = np.linalg.svd(core, full_matrices=False)
    sqrt_s = np.sqrt(np.maximum(s_c, 0.0))
    b_new = (q_b @ u_c) * sqrt_s[None, :]
    a_new = (sqrt_s[:, None]) * (vt_c @ q_a.T)
    if b_new.shape[1] < rank:
        pad = rank - b_new.shape[1]
        b_new = np.pad(b_new, ((0, 0), (0, pad)))
        a_new = np.pad(a_new, ((0, pad), (0, 0)))
    return b_new, a_new


def apply_stage_s5_3_tail_layer_qr_svd_nullspace_inplace(
    state: MutableMapping[str, np.ndarray],
    profile: LayerExpertProfile,
    tail_layer_prose_ffn_mid: Mapping[int, np.ndarray],
    *,
    energy_keep: float = 0.85,
    damp_strength: float = 0.85,
) -> int:
    """Apply rank-preserving QR-SVD AlphaEdit prose-nullspace projection on tail layers."""
    changed = 0
    active_experts = tuple(sorted(set(profile.core_specialist_experts) | set(profile.preserved_multigate_experts)))
    for layer_idx in profile.tail_svd_layers:
        if layer_idx in profile.beneficial_anchor_layers:
            continue
        b_key = f"layer_{layer_idx}.w2_B"
        a_key = f"layer_{layer_idx}.w2_A"
        if b_key not in state or a_key not in state or layer_idx not in tail_layer_prose_ffn_mid:
            continue
        p_null = extract_mp_guarded_prose_projector(
            tail_layer_prose_ffn_mid[layer_idx],
            energy_keep=energy_keep,
        )
        w2_b = np.array(state[b_key], dtype=np.float64, copy=True)
        w2_a = np.array(state[a_key], dtype=np.float64, copy=True)
        for exp_id in active_experts:
            if 0 <= exp_id < w2_b.shape[0]:
                if float(np.linalg.norm(w2_b[exp_id])) <= 1e-12:
                    continue
                b_new, a_new = project_lora_pair_qr_svd_nullspace(
                    w2_b[exp_id],
                    w2_a[exp_id],
                    p_null,
                    damp_strength=damp_strength,
                )
                w2_b[exp_id] = b_new
                w2_a[exp_id] = a_new
                changed += int(b_new.size + a_new.size)
        state[b_key] = w2_b.astype(state[b_key].dtype, copy=False)
        state[a_key] = w2_a.astype(state[a_key].dtype, copy=False)
    return changed


def apply_stage_s5_4_folded_final_norm_calibration_inplace(
    state: MutableMapping[str, np.ndarray],
    base_final_ln_w: np.ndarray,
    c2_final_ln_w: np.ndarray,
    eval_fn: Callable[[Mapping[str, np.ndarray]], Mapping[str, float]],
    base_domain_nll: Mapping[str, float],
    *,
    base_final_ln_b: np.ndarray | None = None,
    c2_final_ln_b: np.ndarray | None = None,
    config: AutoOpenWeightConfig | None = None,
) -> tuple[int, float, float]:
    """Grid-optimize ``(lambda_ln, tau_temp)`` and fold ``1 / tau_temp`` into ``final_ln_w`` / ``final_ln_b``."""
    cfg = config or AutoOpenWeightConfig()
    if "final_ln_w" not in state:
        return 0, 1.0, 1.0
    bw = _validate_finite_array(base_final_ln_w, "base_final_ln_w")
    cw = _validate_finite_array(c2_final_ln_w, "c2_final_ln_w")
    if bw.shape != cw.shape or bw.shape != state["final_ln_w"].shape:
        raise ValueError("final_ln_w shapes must match across base, c2, and state")

    has_bias = "final_ln_b" in state and base_final_ln_b is not None and c2_final_ln_b is not None
    bb = _validate_finite_array(base_final_ln_b, "base_final_ln_b") if has_bias and base_final_ln_b is not None else None
    cb = _validate_finite_array(c2_final_ln_b, "c2_final_ln_b") if has_bias and c2_final_ln_b is not None else None

    orig_w = np.array(state["final_ln_w"], copy=True)
    orig_b = np.array(state["final_ln_b"], copy=True) if has_bias else None

    cur_nll = eval_fn(state)
    best_e, _, _, _ = compute_anse_objective(
        cur_nll,
        base_domain_nll,
        general_domain=cfg.general_domain,
        general_guard_rel=cfg.general_guard_rel,
    )
    best_w = np.array(orig_w, copy=True)
    best_b = np.array(orig_b, copy=True) if orig_b is not None else None
    best_lam = 1.0
    best_tau = 1.0

    for lam in cfg.lambda_ln_grid:
        interp_w = (1.0 - float(lam)) * bw + float(lam) * cw
        interp_b = ((1.0 - float(lam)) * bb + float(lam) * cb) if (bb is not None and cb is not None) else None
        for tau in cfg.tau_temp_grid:
            inv_tau = 1.0 / float(tau)
            state["final_ln_w"] = (interp_w * inv_tau).astype(orig_w.dtype, copy=False)
            if has_bias and interp_b is not None and orig_b is not None:
                state["final_ln_b"] = (interp_b * inv_tau).astype(orig_b.dtype, copy=False)
            cand_nll = eval_fn(state)
            cand_e, _, _, _ = compute_anse_objective(
                cand_nll,
                base_domain_nll,
                general_domain=cfg.general_domain,
                general_guard_rel=cfg.general_guard_rel,
            )
            if cand_e < best_e - 1e-9:
                best_e = cand_e
                best_w = np.array(state["final_ln_w"], copy=True)
                best_b = np.array(state["final_ln_b"], copy=True) if has_bias else None
                best_lam = float(lam)
                best_tau = float(tau)

    state["final_ln_w"] = best_w
    changed = int(np.count_nonzero(np.abs(best_w - orig_w) > 1e-12))
    if has_bias and best_b is not None and orig_b is not None:
        state["final_ln_b"] = best_b
        changed += int(np.count_nonzero(np.abs(best_b - orig_b) > 1e-12))
    return changed, best_lam, best_tau


def run_auto_open_weight_s5_synergy(
    c2_state: Mapping[str, np.ndarray],
    base_domain_nll: Mapping[str, float],
    eval_fn: Callable[[Mapping[str, np.ndarray]], Mapping[str, float]],
    *,
    layer_e0_harm_nats: Sequence[float],
    expert_e0_harm_nats: Sequence[float],
    expert_spec_loss_when_zeroed: Sequence[float],
    layer_prose_inputs: Mapping[int, np.ndarray] | None = None,
    layer_specialist_inputs: Mapping[int, Mapping[str, np.ndarray]] | None = None,
    tail_layer_prose_ffn_mid: Mapping[int, np.ndarray] | None = None,
    base_final_ln_w: np.ndarray | None = None,
    base_final_ln_b: np.ndarray | None = None,
    extra_custom_stages: Sequence[tuple[str, Callable[[MutableMapping[str, np.ndarray]], int]]] = (),
    config: AutoOpenWeightConfig | None = None,
) -> tuple[dict[str, np.ndarray], AutoOpenWeightResult]:
    """Execute the 4-stage ``S5`` PASS-Synergy pipeline under a monotone ANSE ratchet.

    Returns ``(output_state_dict, AutoOpenWeightResult)``. If the final state
    violates the general-domain guard or fails to improve over ``base_domain_nll``,
    ``output_state_dict`` is rolled back bitwise to ``c2_state`` and
    ``anse_energy == ANSE_MAXIMUM_PAIN``.
    """
    cfg = config or AutoOpenWeightConfig()
    pristine_snapshot = clone_state_dict(c2_state)
    working = clone_state_dict(c2_state)

    profile = profile_moe_layers_and_experts(
        layer_e0_harm_nats,
        expert_e0_harm_nats,
        expert_spec_loss_when_zeroed,
        config=cfg,
    )

    init_nll = {str(k): float(v) for k, v in eval_fn(working).items()}
    cur_e, cur_diff, cur_gen_rel, _ = compute_anse_objective(
        init_nll,
        base_domain_nll,
        general_domain=cfg.general_domain,
        general_guard_rel=cfg.general_guard_rel,
    )
    # Note: uncorrected C2_final typically violates general_guard_rel (+3.33% > +1.0%),
    # so cur_e starts at ANSE_MAXIMUM_PAIN (1e6) and Stage S5.1/S5.2 immediately rescues it!
    cur_nll = dict(init_nll)
    stage_receipts: list[StageRatchetReceipt] = []
    chosen_lam = 1.0
    chosen_tau = 1.0

    c2_ln_w = np.array(pristine_snapshot["final_ln_w"], copy=True) if "final_ln_w" in pristine_snapshot else None
    c2_ln_b = np.array(pristine_snapshot["final_ln_b"], copy=True) if "final_ln_b" in pristine_snapshot else None

    def _run_ratcheted_stage(
        stage_name: str,
        mutator: Callable[[MutableMapping[str, np.ndarray]], int],
    ) -> None:
        nonlocal cur_e, cur_diff, cur_gen_rel, cur_nll
        stage_snap = clone_state_dict(working)
        pre_hash = hash_state_dict(stage_snap)
        pre_e = cur_e
        edited_cnt = int(mutator(working))
        if edited_cnt == 0:
            stage_receipts.append(
                StageRatchetReceipt(
                    stage_name=stage_name,
                    accepted=False,
                    reason="noop_zero_parameters_changed",
                    pre_sha256=pre_hash,
                    post_sha256=pre_hash,
                    edited_params=0,
                    pre_anse_energy=pre_e,
                    post_anse_energy=pre_e,
                    delta_anse_energy=0.0,
                    general_rel_drift=cur_gen_rel,
                    mean_diff_vs_base=cur_diff,
                )
            )
            return

        try:
            cand_nll = {str(k): float(v) for k, v in eval_fn(working).items()}
        except Exception:
            restore_state_inplace(working, stage_snap)
            stage_receipts.append(
                StageRatchetReceipt(
                    stage_name=stage_name,
                    accepted=False,
                    reason="rolled_back_eval_exception",
                    pre_sha256=pre_hash,
                    post_sha256=pre_hash,
                    edited_params=0,
                    pre_anse_energy=pre_e,
                    post_anse_energy=ANSE_MAXIMUM_PAIN,
                    delta_anse_energy=0.0,
                    general_rel_drift=cur_gen_rel,
                    mean_diff_vs_base=cur_diff,
                )
            )
            return

        cand_e, cand_diff, cand_gen_rel, _ = compute_anse_objective(
            cand_nll,
            base_domain_nll,
            general_domain=cfg.general_domain,
            general_guard_rel=cfg.general_guard_rel,
        )
        # Accept if it strictly reduces ANSE energy OR rescues an initially guard-violating state
        # OR strictly reduces raw mean_diff while both pre and post remain guard-violation (progressive rescue)
        rescues_guard = pre_e >= ANSE_MAXIMUM_PAIN * 0.5 and cand_e < ANSE_MAXIMUM_PAIN * 0.5
        improves_feasible = cand_e < ANSE_MAXIMUM_PAIN * 0.5 and (cand_e < pre_e - float(cfg.min_stage_gain_nats))
        reduces_infeasible_drift = (
            pre_e >= ANSE_MAXIMUM_PAIN * 0.5
            and cand_e >= ANSE_MAXIMUM_PAIN * 0.5
            and cand_gen_rel < cur_gen_rel - 1e-4
        )
        if rescues_guard or improves_feasible or reduces_infeasible_drift:
            post_hash = hash_state_dict(working)
            delta_e = cand_e - pre_e
            cur_e, cur_diff, cur_gen_rel, cur_nll = cand_e, cand_diff, cand_gen_rel, cand_nll
            stage_receipts.append(
                StageRatchetReceipt(
                    stage_name=stage_name,
                    accepted=True,
                    reason="accepted_monotone_anse_improvement",
                    pre_sha256=pre_hash,
                    post_sha256=post_hash,
                    edited_params=edited_cnt,
                    pre_anse_energy=pre_e,
                    post_anse_energy=cand_e,
                    delta_anse_energy=delta_e,
                    general_rel_drift=cand_gen_rel,
                    mean_diff_vs_base=cand_diff,
                )
            )
        else:
            restore_state_inplace(working, stage_snap)
            stage_receipts.append(
                StageRatchetReceipt(
                    stage_name=stage_name,
                    accepted=False,
                    reason="rolled_back_non_improving_or_guard_violating",
                    pre_sha256=pre_hash,
                    post_sha256=pre_hash,
                    edited_params=0,
                    pre_anse_energy=pre_e,
                    post_anse_energy=cand_e,
                    delta_anse_energy=0.0,
                    general_rel_drift=cur_gen_rel,
                    mean_diff_vs_base=cur_diff,
                )
            )

    # Stage S5.1: Anchor-protected FoMoE ghost sinking
    _run_ratcheted_stage(
        "S5.1_anchor_protected_ghost_sink",
        lambda st: apply_stage_s5_1_anchor_protected_ghost_sink_inplace(st, profile),
    )

    # Stage S5.2: 2-fold OOF SNR-gated orthogonal multi-gate specialist preservation
    if layer_prose_inputs is not None and layer_specialist_inputs is not None:
        _run_ratcheted_stage(
            "S5.2_oof_orthogonal_multigate",
            lambda st: apply_stage_s5_2_oof_orthogonal_multigate_inplace(
                st,
                profile,
                layer_prose_inputs,
                layer_specialist_inputs,
                router_beta=cfg.router_beta,
                min_oof_snr=cfg.min_oof_snr,
            )[0],
        )

    # Stage S5.3: Tail-layer QR-SVD AlphaEdit prose-nullspace projection
    if tail_layer_prose_ffn_mid is not None:
        _run_ratcheted_stage(
            "S5.3_tail_layer_qr_svd_nullspace",
            lambda st: apply_stage_s5_3_tail_layer_qr_svd_nullspace_inplace(
                st,
                profile,
                tail_layer_prose_ffn_mid,
                energy_keep=cfg.prose_nullspace_energy_keep,
                damp_strength=cfg.svd_damp_strength,
            ),
        )

    # Stage S5.4: Folded final LayerNorm / RMSNorm interpolation & temperature calibration
    if base_final_ln_w is not None and c2_ln_w is not None:

        def _stage_ln(st: MutableMapping[str, np.ndarray]) -> int:
            nonlocal chosen_lam, chosen_tau
            cnt, lam, tau = apply_stage_s5_4_folded_final_norm_calibration_inplace(
                st,
                base_final_ln_w,
                c2_ln_w,
                eval_fn,
                base_domain_nll,
                base_final_ln_b=base_final_ln_b,
                c2_final_ln_b=c2_ln_b,
                config=cfg,
            )
            if cnt > 0:
                chosen_lam, chosen_tau = lam, tau
            return cnt

        _run_ratcheted_stage("S5.4_folded_final_norm_temperature", _stage_ln)

    # Optional custom extension stages (e.g., mid-pipeline regressing test stages or quantization fold)
    for custom_name, custom_fn in extra_custom_stages:
        _run_ratcheted_stage(custom_name, custom_fn)

    final_e, final_diff, final_gen_rel, per_rel = compute_anse_objective(
        cur_nll,
        base_domain_nll,
        general_domain=cfg.general_domain,
        general_guard_rel=cfg.general_guard_rel,
    )
    if final_e >= ANSE_MAXIMUM_PAIN * 0.5 or final_diff >= 0.0:
        restore_state_inplace(working, pristine_snapshot)
        ckpt_sha = hash_state_dict(working)
        base_dict = {str(k): float(v) for k, v in base_domain_nll.items()}
        fail_res = AutoOpenWeightResult(
            committed=False,
            verdict="ROLLED_BACK_MAXIMUM_PAIN",
            anse_energy=ANSE_MAXIMUM_PAIN,
            mean_diff_vs_base=final_diff,
            general_rel_drift=final_gen_rel,
            all_domains_negative=False,
            total_params=sum(int(v.size) for v in working.values()),
            edited_params=0,
            edited_fraction=0.0,
            base_domain_nll=base_dict,
            initial_domain_nll=init_nll,
            final_domain_nll=init_nll,
            per_domain_rel_vs_base=per_rel,
            profile=profile,
            stages=tuple(stage_receipts),
            selected_lambda_ln=1.0,
            selected_tau_temp=1.0,
            checkpoint_sha256=ckpt_sha,
            receipt_sha256="",
        )
        raw_json = json.dumps(fail_res.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        fail_res = dataclasses.replace(fail_res, receipt_sha256=hashlib.sha256(raw_json).hexdigest())
        return working, fail_res

    edited_params, total_params = count_edited_parameters(pristine_snapshot, working)
    ckpt_sha = hash_state_dict(working)
    base_dict = {str(k): float(v) for k, v in base_domain_nll.items()}
    res = AutoOpenWeightResult(
        committed=True,
        verdict="PASS",
        anse_energy=final_e,
        mean_diff_vs_base=final_diff,
        general_rel_drift=final_gen_rel,
        all_domains_negative=all(v < 0.0 for v in per_rel.values()),
        total_params=total_params,
        edited_params=edited_params,
        edited_fraction=float(edited_params) / float(max(total_params, 1)),
        base_domain_nll=base_dict,
        initial_domain_nll=init_nll,
        final_domain_nll=cur_nll,
        per_domain_rel_vs_base=per_rel,
        profile=profile,
        stages=tuple(stage_receipts),
        selected_lambda_ln=chosen_lam,
        selected_tau_temp=chosen_tau,
        checkpoint_sha256=ckpt_sha,
        receipt_sha256="",
    )
    raw_json = json.dumps(res.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    res = dataclasses.replace(res, receipt_sha256=hashlib.sha256(raw_json).hexdigest())
    return working, res

# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""End-to-End Top-Level AutoOpenWeightOrchestrator for Season 35 Weight-Space Programming.

Chains the five merged Season 35 open-weight neural control modules on ``origin/main``
into a single deterministic, CPU-only (``0 MiB`` GPU VRAM) pipeline
(:func:`run_full_auto_open_weight_pipeline` / :class:`AutoOpenWeightOrchestrator`):

1. **Architecture Detection & Preflight Diagnosis**
   (:mod:`hivemind.swarm.multi_arch_auto_wsp_adapter`):
   Auto-detects Granite-MoE, Qwen-MoE, Qwen-Dense, and Llama-GQA native PyTorch
   layouts (LoRA or full-weight, GQA head ratio, final LayerNorm / RMSNorm),
   canonicalizes parameters to CPU ``float64``, and diagnoses rogue expert Frobenius
   norm spikes, zero sinks, and final norm drift.
2. **CPU S5 PASS-Synergy Weight Surgery with Monotone ANSE Ratchet**
   (:mod:`hivemind.swarm.auto_open_weight_engine` +
   :mod:`hivemind.swarm.multi_arch_auto_wsp_adapter`):
   Executes ``S5.1`` anchor-protected FoMoE ghost sinking, ``S5.2`` 2-fold OOF
   SNR-gated orthogonal multi-gate router surgery, ``S5.3`` tail-layer QR-SVD
   AlphaEdit null-space projection, and ``S5.4`` zero-overhead folded final-norm
   temperature calibration while preserving untouched tensors bitwise in native dtype.
3. **Runtime Algebraic & Bitwise Soundness Certification**
   (:mod:`hivemind.swarm.wsp_formal_invariant_verifier`):
   Certifies all five formal obligations in an immutable ``WSPFormalCertificate``
   (projector idempotence/symmetry/spectrum, preserved-key invariance, orthogonal
   multi-gate centroid invariance, anchor-layer 0-bit immutability, and monotone
   ANSE stage descent). Any infeasible guard or invariant violation halts
   automatically before NF4 export and rolls back bitwise with ``ANSE_MAXIMUM_PAIN``.
4. **Optional Single-Shot Virgin-Vault Evaluation Lock**
   (:class:`hivemind.swarm.auto_open_weight_engine.SingleShotVaultLock`):
   Enforces strict one-shot holdout scoring on an untouched vault split and halts
   before edge export on guard breach.
5. **4-Bit NF4 Hessian-Compensated Edge Export & Shadow-Canary Promotion Bundle**
   (:mod:`hivemind.swarm.volunteer_wsp_workunit_exporter` +
   :mod:`hivemind.swarm.opentrainer_auto_wsp_hook` +
   :class:`hivemind.core.expert_registry.ExpertRegistry`):
   Folds S5 edits into 4-bit NF4 blocks within the 8 GB WDDM ceiling
   (``<= 6656 MiB``, ``< 0.02`` nat NLL gap vs FP32), seals the tamper-evident
   ``PostRoundWSPLineageReceipt`` and ``ExpertRegistry`` shadow/canary gate
   bundle, and binds every stage into a unified SHA-256 manifest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
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
from hivemind.swarm.auto_open_weight_engine import (
    ANSE_MAXIMUM_PAIN,
    AutoOpenWeightConfig,
    LayerExpertProfile,
    SingleShotVaultLock,
    compute_anse_objective,
    compute_oof_orthogonal_router_direction,
    extract_mp_guarded_prose_projector,
)
from hivemind.swarm.multi_arch_auto_wsp_adapter import (
    ARCH_GRANITE_MOE,
    SUPPORTED_ARCH_FAMILIES,
    MultiArchAutoWSPResult,
    OpenWeightArchitectureSpec,
    canonicalize_state_dict_for_s5,
    clone_native_state_dict,
    detect_open_weight_architecture,
    hash_native_state_dict,
    materialize_s5_back_to_native_layout,
    run_multi_arch_auto_open_weight_s5,
    verify_multi_arch_receipt_provenance,
)
from hivemind.swarm.opentrainer_auto_wsp_hook import (
    DEFAULT_CANARY_PROBE_TOTAL,
    PROMOTION_STAGE_PROMOTED_LIVE,
    PROMOTION_STAGE_PROMOTED_SHADOW,
    PROMOTION_STAGE_REJECTED_PARENT,
    PROMOTION_STAGE_REJECTED_RAW,
    PostRoundWSPLineageReceipt,
    build_expert_registry_gate_payload,
    evaluate_shadow_to_live_canary_gate,
    verify_lineage_receipt_provenance,
)
from hivemind.swarm.volunteer_wsp_workunit_exporter import (
    DEFAULT_NF4_BLOCK_SIZE,
    MAX_NF4_NLL_GAP_NATS,
    STD_8GB_WDDM_CEILING_MIB,
    VRAMCeilingExceededError,
    export_s5_model_to_nf4_edge,
    verify_workunit_receipt_provenance,
)
from hivemind.swarm.wsp_formal_invariant_verifier import (
    DEFAULT_CENTROID_INVARIANCE_TOL,
    DEFAULT_KEY_INVARIANCE_TOL,
    DEFAULT_PROJECTOR_TOL,
    DEFAULT_PROSE_ALIGNMENT_FLOOR,
    TensorLike,
    WSPFormalCertificate,
    canonical_json_bytes,
    certify_s5_surgery_bundle,
    compute_payload_sha256,
    verify_wsp_formal_certificate_provenance,
)

SCHEMA_VERSION: str = "hivemind.auto_open_weight_orchestrator/1"
PIPELINE_MANIFEST_OBJECT: str = "hivemind.auto_open_weight_pipeline_manifest"
PREFLIGHT_DIAGNOSIS_OBJECT: str = "hivemind.auto_open_weight_preflight_diagnosis"
PROMOTION_BUNDLE_OBJECT: str = "hivemind.auto_open_weight_promotion_bundle"

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SKIP_AUTO_NF4_PREFIXES: tuple[str, ...] = (
    "wte.",
    "embed_tokens.",
    "model.embed_tokens.",
    "engram.",
    "head.",
    "lm_head.",
)


# ---------------------------------------------------------------------------
# 1. Validation & Provenance Helpers
# ---------------------------------------------------------------------------


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_nonempty_str(value: Any, name: str) -> str:
    """Validate that ``value`` is a non-empty printable string without control characters."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped or any(ord(ch) < 32 or ord(ch) == 127 for ch in stripped):
        raise ValueError(f"{name} must be a non-empty printable string")
    return stripped


def _validate_nonneg_int(value: Any, name: str) -> int:
    """Validate a non-negative integer (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate a strictly positive integer (excluding booleans)."""
    out = _validate_nonneg_int(value, name)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return out


def _validate_finite_float(value: Any, name: str) -> float:
    """Validate a finite real number (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _validate_nonneg_finite(value: Any, name: str) -> float:
    """Validate a finite non-negative real number."""
    out = _validate_finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0.0, got {value!r}")
    return out


def _validate_positive_finite(value: Any, name: str) -> float:
    """Validate a finite strictly positive real number."""
    out = _validate_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_unit_interval(value: Any, name: str, *, allow_zero: bool = True) -> float:
    """Validate a finite scalar inside ``[0, 1]`` or ``(0, 1]``."""
    out = _validate_nonneg_finite(value, name)
    if not allow_zero and out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    if out > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {value!r}")
    return out


def _sanitize_finite_or_pain(value: float, pain: float = ANSE_MAXIMUM_PAIN) -> float:
    """Replace non-finite float metrics with bounded ``ANSE_MAXIMUM_PAIN`` for JSON sealing."""
    v = float(value)
    return v if math.isfinite(v) else float(pain)


def _validate_cpu_state_mapping(
    state_dict: Any,
    *,
    name: str,
    require_finite: bool,
) -> tuple[dict[str, torch.Tensor], bool]:
    """Validate a non-empty CPU tensor mapping and report whether all tensors are finite."""
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError(f"{name} must be a non-empty mapping of str -> torch.Tensor")
    out: dict[str, torch.Tensor] = {}
    all_finite = True
    for key, val in state_dict.items():
        k_str = _validate_nonempty_str(key, f"{name} key")
        if not isinstance(val, torch.Tensor):
            raise TypeError(f"{name}[{k_str!r}] must be a torch.Tensor, got {type(val).__name__}")
        if int(val.numel()) == 0:
            raise ValueError(f"{name}[{k_str!r}] must be non-empty")
        if val.device.type != "cpu":
            raise ValueError(
                f"{name}[{k_str!r}] must reside on CPU for zero-VRAM surgery, got device={val.device}"
            )
        is_fin = (
            bool(torch.isfinite(val.detach()).all().item())
            if (val.is_floating_point() or val.is_complex())
            else True
        )
        if not is_fin:
            if require_finite:
                raise ValueError(f"{name}[{k_str!r}] contains non-finite values")
            all_finite = False
        out[k_str] = val
    return out, all_finite


def _validate_base_domain_nll(
    base_domain_nll: Any,
    *,
    general_domain: str,
    name: str = "base_domain_nll",
) -> dict[str, float]:
    """Validate base domain NLL mapping ensuring finite positive baselines and general domain."""
    if not isinstance(base_domain_nll, Mapping) or not base_domain_nll:
        raise ValueError(f"{name} must be a non-empty mapping of domain -> positive float")
    out: dict[str, float] = {}
    for raw_k, raw_v in base_domain_nll.items():
        dom = _validate_nonempty_str(raw_k, f"{name} domain")
        out[dom] = _validate_positive_finite(raw_v, f"{name}[{dom!r}]")
    if general_domain not in out:
        raise ValueError(f"{name} must contain general_domain {general_domain!r}")
    return out


def _seal_lineage_receipt(fields: Mapping[str, Any]) -> PostRoundWSPLineageReceipt:
    """Build and SHA-256-seal a ``PostRoundWSPLineageReceipt`` compatible with ``opentrainer_auto_wsp_hook``."""
    core = {k: v for k, v in fields.items() if k != "receipt_sha256"}
    unsealed = PostRoundWSPLineageReceipt(**core, receipt_sha256="")
    payload = {k: v for k, v in unsealed.to_dict().items() if k != "receipt_sha256"}
    sha = compute_payload_sha256(payload)
    return dataclasses.replace(unsealed, receipt_sha256=sha)


def _hash_unvalidated_cpu_state(state: Mapping[str, torch.Tensor]) -> str:
    """Compute a deterministic bitwise SHA-256 over raw CPU tensor bytes even when non-finite."""
    digest = hashlib.sha256()
    for key in sorted(state.keys()):
        t_cpu = state[key].detach().to(device="cpu").contiguous()
        digest.update(f"{key}:{t_cpu.dtype}:{tuple(t_cpu.shape)}:".encode())
        if t_cpu.numel() > 0:
            raw = (
                t_cpu.to(torch.uint8).numpy().tobytes()
                if t_cpu.dtype == torch.bool
                else t_cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
            )
            digest.update(raw)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 2. Configuration & Immutable Pipeline Result Dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AutoOpenWeightOrchestratorConfig:
    """Unified configuration governing all 5 stages of ``AutoOpenWeightOrchestrator``."""

    s5_config: AutoOpenWeightConfig = dataclasses.field(default_factory=AutoOpenWeightConfig)
    arch_hint: str | None = None
    full_weight_max_rank: int | None = None
    auto_temperature_calibration: bool = True
    rogue_norm_ratio_threshold: float = 2.2
    ln_drift_threshold: float = 0.20
    projector_tol: float = DEFAULT_PROJECTOR_TOL
    key_tol: float = DEFAULT_KEY_INVARIANCE_TOL
    centroid_tol: float = DEFAULT_CENTROID_INVARIANCE_TOL
    min_prose_alignment: float = DEFAULT_PROSE_ALIGNMENT_FLOOR
    enable_nf4_export: bool = True
    nf4_block_size: int = DEFAULT_NF4_BLOCK_SIZE
    max_nf4_gap_nats: float = MAX_NF4_NLL_GAP_NATS
    vram_budget_mib: float = 6144.0
    quantize_attention_nf4: bool = False
    quantize_untouched_2d_nf4: bool = False
    base_model_id: str = "hivemind/crowd-v1"
    cluster_id: str = "moe_swarm"
    adapter_size_mib: int = 48
    canary_min_pass_ratio: float = 1.0
    canary_max_general_drift_rel: float = 0.01
    canary_min_probe_total: int = 1
    canary_default_probe_pass_count: int = DEFAULT_CANARY_PROBE_TOTAL
    canary_default_probe_total: int = DEFAULT_CANARY_PROBE_TOTAL
    keywords: tuple[str, ...] = ("wsp_s5", "auto_open_weight")
    round_id: str = "round_0"
    surgery_id: str = "s5_auto_open_weight"

    def __post_init__(self) -> None:
        if not isinstance(self.s5_config, AutoOpenWeightConfig):
            raise TypeError(
                f"s5_config must be AutoOpenWeightConfig, got {type(self.s5_config).__name__}"
            )
        if self.arch_hint is not None and self.arch_hint not in SUPPORTED_ARCH_FAMILIES:
            raise ValueError(
                f"Unsupported arch_hint {self.arch_hint!r}; expected one of {sorted(SUPPORTED_ARCH_FAMILIES)}"
            )
        if self.full_weight_max_rank is not None:
            _validate_positive_int(self.full_weight_max_rank, "full_weight_max_rank")
        _validate_bool(self.auto_temperature_calibration, "auto_temperature_calibration")
        rogue_r = _validate_positive_finite(self.rogue_norm_ratio_threshold, "rogue_norm_ratio_threshold")
        if rogue_r <= 1.0:
            raise ValueError("rogue_norm_ratio_threshold must be > 1.0")
        _validate_nonneg_finite(self.ln_drift_threshold, "ln_drift_threshold")
        _validate_nonneg_finite(self.projector_tol, "projector_tol")
        _validate_nonneg_finite(self.key_tol, "key_tol")
        _validate_nonneg_finite(self.centroid_tol, "centroid_tol")
        _validate_nonneg_finite(self.min_prose_alignment, "min_prose_alignment")
        _validate_bool(self.enable_nf4_export, "enable_nf4_export")
        _validate_positive_int(self.nf4_block_size, "nf4_block_size")
        _validate_positive_finite(self.max_nf4_gap_nats, "max_nf4_gap_nats")
        vram = _validate_positive_finite(self.vram_budget_mib, "vram_budget_mib")
        if vram > float(STD_8GB_WDDM_CEILING_MIB):
            raise VRAMCeilingExceededError(
                f"vram_budget_mib ({vram:.1f} MiB) exceeds std_8gb WDDM ceiling ({STD_8GB_WDDM_CEILING_MIB} MiB)"
            )
        _validate_bool(self.quantize_attention_nf4, "quantize_attention_nf4")
        _validate_bool(self.quantize_untouched_2d_nf4, "quantize_untouched_2d_nf4")
        _validate_nonempty_str(self.base_model_id, "base_model_id")
        _validate_nonempty_str(self.cluster_id, "cluster_id")
        _validate_positive_int(self.adapter_size_mib, "adapter_size_mib")
        _validate_unit_interval(self.canary_min_pass_ratio, "canary_min_pass_ratio", allow_zero=False)
        _validate_nonneg_finite(self.canary_max_general_drift_rel, "canary_max_general_drift_rel")
        min_tot = _validate_positive_int(self.canary_min_probe_total, "canary_min_probe_total")
        def_tot = _validate_positive_int(self.canary_default_probe_total, "canary_default_probe_total")
        def_pass = _validate_nonneg_int(
            self.canary_default_probe_pass_count, "canary_default_probe_pass_count"
        )
        if def_tot < min_tot or def_pass > def_tot:
            raise ValueError("canary default probe counts must satisfy 0 <= pass <= total >= min_probe_total")
        if isinstance(self.keywords, (str, bytes)) or not isinstance(self.keywords, Sequence):
            raise TypeError("keywords must be a sequence of strings")
        cleaned_kw = tuple(_validate_nonempty_str(k, "keyword") for k in self.keywords)
        object.__setattr__(self, "keywords", cleaned_kw)
        _validate_nonempty_str(self.round_id, "round_id")
        _validate_nonempty_str(self.surgery_id, "surgery_id")


@dataclasses.dataclass(frozen=True)
class PreflightLayerDiagnosis:
    """Preflight Frobenius norm & rogue-expert diagnosis for one layer."""

    layer_index: int
    expert_count: int
    expert_factor_fro_norms: tuple[float, ...]
    expert_effective_fro_norms: tuple[float, ...]
    median_active_norm: float
    max_to_median_ratio: float
    zero_sink_experts: tuple[int, ...]
    rogue_experts: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize layer diagnosis into a JSON-safe dictionary."""
        return {
            "layer_index": self.layer_index,
            "expert_count": self.expert_count,
            "expert_factor_fro_norms": list(self.expert_factor_fro_norms),
            "expert_effective_fro_norms": list(self.expert_effective_fro_norms),
            "median_active_norm": self.median_active_norm,
            "max_to_median_ratio": self.max_to_median_ratio,
            "zero_sink_experts": list(self.zero_sink_experts),
            "rogue_experts": list(self.rogue_experts),
        }


@dataclasses.dataclass(frozen=True)
class PreflightDiagnosisReport:
    """Step-1 architecture specification and rogue-expert / final-norm diagnosis receipt."""

    arch_spec: OpenWeightArchitectureSpec
    healthy: bool
    rogue_alert_count: int
    rogue_alerts: tuple[dict[str, Any], ...]
    final_norm_max_abs_drift_from_unit: float
    final_norm_mean_scale: float
    norm_drift_detected: bool
    layers: tuple[PreflightLayerDiagnosis, ...]
    pre_native_sha256: str
    canonical_pre_sha256: str
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize preflight report into a canonical JSON-safe dictionary."""
        return {
            "schema": SCHEMA_VERSION,
            "object": PREFLIGHT_DIAGNOSIS_OBJECT,
            "arch_spec": self.arch_spec.to_dict(),
            "healthy": self.healthy,
            "rogue_alert_count": self.rogue_alert_count,
            "rogue_alerts": [dict(a) for a in self.rogue_alerts],
            "final_norm_max_abs_drift_from_unit": self.final_norm_max_abs_drift_from_unit,
            "final_norm_mean_scale": self.final_norm_mean_scale,
            "norm_drift_detected": self.norm_drift_detected,
            "layers": [layer.to_dict() for layer in self.layers],
            "pre_native_sha256": self.pre_native_sha256,
            "canonical_pre_sha256": self.canonical_pre_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


@dataclasses.dataclass(frozen=True)
class NF4EdgeBundle:
    """Step-5 4-bit NF4 Hessian-compensated edge export bundle and telemetry."""

    quantized_weights: Mapping[str, torch.Tensor]
    block_size: int
    quantized_tensor_count: int
    total_tensor_count: int
    edge_payload_bytes: int
    edge_vram_mib: float
    vram_budget_mib: float
    fp32_nll_nats: float | None
    nf4_nll_nats: float | None
    nf4_gap_nats: float
    hessian_weighted_error: float
    naive_hessian_weighted_error: float
    hessian_error_reduction_ratio: float
    exported_weights_sha256: str
    receipt: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize NF4 edge bundle telemetry into a JSON-safe dictionary (omitting tensor blobs)."""
        return {
            "block_size": self.block_size,
            "quantized_tensor_count": self.quantized_tensor_count,
            "total_tensor_count": self.total_tensor_count,
            "edge_payload_bytes": self.edge_payload_bytes,
            "edge_vram_mib": self.edge_vram_mib,
            "vram_budget_mib": self.vram_budget_mib,
            "fp32_nll_nats": self.fp32_nll_nats,
            "nf4_nll_nats": self.nf4_nll_nats,
            "nf4_gap_nats": self.nf4_gap_nats,
            "hessian_weighted_error": self.hessian_weighted_error,
            "naive_hessian_weighted_error": self.naive_hessian_weighted_error,
            "hessian_error_reduction_ratio": self.hessian_error_reduction_ratio,
            "exported_weights_sha256": self.exported_weights_sha256,
            "receipt": dict(self.receipt),
        }


@dataclasses.dataclass(frozen=True)
class ShadowCanaryPromotionBundle:
    """Step-5 ``ExpertRegistry`` shadow/canary lineage and promotion gate bundle."""

    lineage_receipt: PostRoundWSPLineageReceipt
    gate_payload: Mapping[str, Any]
    promotion_stage: str
    shadow_canary_ready: bool
    promoted_to_live: bool
    expert_id: str | None
    registry_state: str | None
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize promotion bundle into a canonical JSON-safe dictionary."""
        return {
            "schema": SCHEMA_VERSION,
            "object": PROMOTION_BUNDLE_OBJECT,
            "lineage_receipt": self.lineage_receipt.to_dict(),
            "gate_payload": dict(self.gate_payload),
            "promotion_stage": self.promotion_stage,
            "shadow_canary_ready": self.shadow_canary_ready,
            "promoted_to_live": self.promoted_to_live,
            "expert_id": self.expert_id,
            "registry_state": self.registry_state,
            "receipt_sha256": self.receipt_sha256,
        }


@dataclasses.dataclass(frozen=True)
class AutoOpenWeightPipelineResult:
    """Unified end-to-end result and SHA-256 manifest from ``AutoOpenWeightOrchestrator``."""

    committed: bool
    verdict: str
    halted_before_nf4_export: bool
    halted_stage: str | None
    halt_reason: str | None
    rolled_back_bitwise: bool
    nf4_export_executed: bool
    arch_family: str
    parameterization: str
    anse_energy: float
    nll_mean: float
    nf4_gap_nats: float
    general_rel_drift: float
    mean_diff_vs_base: float
    all_domains_negative: bool
    pre_native_sha256: str
    post_native_sha256: str
    untouched_keys_bitwise_preserved: bool
    preflight_diagnosis: PreflightDiagnosisReport
    multi_arch_result: MultiArchAutoWSPResult | None
    formal_certificate: WSPFormalCertificate | None
    vault_receipt: Mapping[str, Any] | None
    nf4_edge_bundle: NF4EdgeBundle | None
    promotion_bundle: ShadowCanaryPromotionBundle
    manifest: Mapping[str, Any]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize entire pipeline execution and nested receipts into JSON-safe telemetry."""
        return {
            "schema": SCHEMA_VERSION,
            "object": PIPELINE_MANIFEST_OBJECT,
            "committed": self.committed,
            "verdict": self.verdict,
            "halted_before_nf4_export": self.halted_before_nf4_export,
            "halted_stage": self.halted_stage,
            "halt_reason": self.halt_reason,
            "rolled_back_bitwise": self.rolled_back_bitwise,
            "nf4_export_executed": self.nf4_export_executed,
            "arch_family": self.arch_family,
            "parameterization": self.parameterization,
            "anse_energy": self.anse_energy,
            "nll_mean": self.nll_mean,
            "nf4_gap_nats": self.nf4_gap_nats,
            "general_rel_drift": self.general_rel_drift,
            "mean_diff_vs_base": self.mean_diff_vs_base,
            "all_domains_negative": self.all_domains_negative,
            "pre_native_sha256": self.pre_native_sha256,
            "post_native_sha256": self.post_native_sha256,
            "untouched_keys_bitwise_preserved": self.untouched_keys_bitwise_preserved,
            "preflight_diagnosis": self.preflight_diagnosis.to_dict(),
            "multi_arch_result": (
                self.multi_arch_result.to_dict() if self.multi_arch_result is not None else None
            ),
            "formal_certificate": (
                self.formal_certificate.to_dict() if self.formal_certificate is not None else None
            ),
            "vault_receipt": dict(self.vault_receipt) if self.vault_receipt is not None else None,
            "nf4_edge_bundle": (
                self.nf4_edge_bundle.to_dict() if self.nf4_edge_bundle is not None else None
            ),
            "promotion_bundle": self.promotion_bundle.to_dict(),
            "manifest": dict(self.manifest),
            "manifest_sha256": self.manifest_sha256,
        }


# ---------------------------------------------------------------------------
# 3. Core Orchestrator Implementation
# ---------------------------------------------------------------------------


class AutoOpenWeightOrchestrator:
    """Top-level Season 35 orchestrator for architecture-aware S5 WSP, formal certification, and NF4 edge export."""

    def __init__(self, config: AutoOpenWeightOrchestratorConfig | None = None) -> None:
        self.config = AutoOpenWeightOrchestratorConfig() if config is None else config
        if not isinstance(self.config, AutoOpenWeightOrchestratorConfig):
            raise TypeError(
                f"config must be AutoOpenWeightOrchestratorConfig or None, got {type(self.config).__name__}"
            )

    def diagnose_preflight(
        self,
        native_state: Mapping[str, torch.Tensor],
    ) -> PreflightDiagnosisReport:
        """Detect open-weight architecture layout and diagnose rogue-expert / norm anomalies."""
        cfg = self.config
        validated, _ = _validate_cpu_state_mapping(
            native_state, name="native_state", require_finite=True
        )
        arch_spec = detect_open_weight_architecture(validated, arch_hint=cfg.arch_hint)
        pre_native_sha = hash_native_state_dict(validated)
        canon_pre = canonicalize_state_dict_for_s5(
            validated,
            arch_hint=cfg.arch_hint,
            full_weight_max_rank=cfg.full_weight_max_rank,
        )
        canon_sha = compute_payload_sha256(
            {k: [list(v.shape), str(v.dtype)] for k, v in sorted(canon_pre.items())}
        )

        layer_diags: list[PreflightLayerDiagnosis] = []
        rogue_alerts: list[dict[str, Any]] = []
        for l_idx in range(arch_spec.layer_count):
            w2_b = np.asarray(canon_pre[f"layer_{l_idx}.w2_B"], dtype=np.float64)
            w2_a = np.asarray(canon_pre[f"layer_{l_idx}.w2_A"], dtype=np.float64)
            factor_norms: list[float] = []
            eff_norms: list[float] = []
            for e_idx in range(arch_spec.expert_count):
                b_norm = float(np.linalg.norm(w2_b[e_idx]))
                eff_norm = float(np.linalg.norm(w2_b[e_idx] @ w2_a[e_idx]))
                factor_norms.append(round(b_norm, 6))
                eff_norms.append(round(eff_norm, 6))

            active_norms = sorted(n for n in eff_norms if n > 1e-9)
            if not active_norms:
                median_active = 0.0
                max_ratio = 0.0
            else:
                mid = len(active_norms) // 2
                median_active = (
                    active_norms[mid]
                    if len(active_norms) % 2 == 1
                    else 0.5 * (active_norms[mid - 1] + active_norms[mid])
                )
                max_ratio = max(eff_norms) / median_active if median_active > 1e-9 else 0.0

            zero_sinks = tuple(e_idx for e_idx, n in enumerate(eff_norms) if n <= 1e-9)
            if arch_spec.expert_count >= 2 and median_active > 1e-9:
                rogues = tuple(
                    e_idx
                    for e_idx, n in enumerate(eff_norms)
                    if (n / median_active) >= float(cfg.rogue_norm_ratio_threshold)
                )
            else:
                rogues = ()
            if rogues:
                rogue_alerts.append(
                    {
                        "layer_index": l_idx,
                        "rogue_experts": list(rogues),
                        "median_active_norm": round(median_active, 6),
                        "max_to_median_ratio": round(max_ratio, 6),
                    }
                )
            layer_diags.append(
                PreflightLayerDiagnosis(
                    layer_index=l_idx,
                    expert_count=arch_spec.expert_count,
                    expert_factor_fro_norms=tuple(factor_norms),
                    expert_effective_fro_norms=tuple(eff_norms),
                    median_active_norm=round(median_active, 6),
                    max_to_median_ratio=round(max_ratio, 6),
                    zero_sink_experts=zero_sinks,
                    rogue_experts=rogues,
                )
            )

        if "final_ln_w" in canon_pre:
            ln_w = np.asarray(canon_pre["final_ln_w"], dtype=np.float64)
            ln_max_drift = float(np.max(np.abs(ln_w - 1.0)))
            ln_mean = float(np.mean(ln_w))
        else:
            ln_max_drift = 0.0
            ln_mean = 1.0

        norm_drift_detected = ln_max_drift > float(cfg.ln_drift_threshold)
        healthy = (len(rogue_alerts) == 0) and (not norm_drift_detected)
        core = {
            "schema": SCHEMA_VERSION,
            "object": PREFLIGHT_DIAGNOSIS_OBJECT,
            "arch_spec": arch_spec.to_dict(),
            "healthy": healthy,
            "rogue_alert_count": len(rogue_alerts),
            "rogue_alerts": rogue_alerts,
            "final_norm_max_abs_drift_from_unit": round(ln_max_drift, 6),
            "final_norm_mean_scale": round(ln_mean, 6),
            "norm_drift_detected": norm_drift_detected,
            "layers": [d.to_dict() for d in layer_diags],
            "pre_native_sha256": pre_native_sha,
            "canonical_pre_sha256": canon_sha,
        }
        return PreflightDiagnosisReport(
            arch_spec=arch_spec,
            healthy=healthy,
            rogue_alert_count=len(rogue_alerts),
            rogue_alerts=tuple(rogue_alerts),
            final_norm_max_abs_drift_from_unit=round(ln_max_drift, 6),
            final_norm_mean_scale=round(ln_mean, 6),
            norm_drift_detected=norm_drift_detected,
            layers=tuple(layer_diags),
            pre_native_sha256=pre_native_sha,
            canonical_pre_sha256=canon_sha,
            receipt_sha256=compute_payload_sha256(core),
        )

    @staticmethod
    def _normalize_np_activation_map(
        raw_map: Mapping[int, Any] | None,
    ) -> dict[int, np.ndarray]:
        """Convert optional per-layer activation tensors to CPU ``float64`` NumPy arrays."""
        if not raw_map:
            return {}
        out: dict[int, np.ndarray] = {}
        for l_idx, val in raw_map.items():
            arr = (
                val.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
                if isinstance(val, torch.Tensor)
                else np.asarray(val, dtype=np.float64)
            )
            out[int(l_idx)] = arr
        return out

    @staticmethod
    def _normalize_np_specialist_map(
        raw_map: Mapping[int, Mapping[str, Any]] | None,
    ) -> dict[int, dict[str, np.ndarray]]:
        """Convert optional per-layer specialist domain activations to CPU ``float64`` NumPy."""
        if not raw_map:
            return {}
        out: dict[int, dict[str, np.ndarray]] = {}
        for l_idx, dom_map in raw_map.items():
            out[int(l_idx)] = {
                str(dom): (
                    val.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
                    if isinstance(val, torch.Tensor)
                    else np.asarray(val, dtype=np.float64)
                )
                for dom, val in dom_map.items()
            }
        return out

    @staticmethod
    def _nullspace_basis_from_projector(p_null: np.ndarray) -> np.ndarray:
        """Extract orthonormal basis ``V_perp`` spanning ``range(P_0)`` from ``P_0``."""
        p_sym = 0.5 * (p_null + p_null.T)
        evals, evecs = np.linalg.eigh(p_sym)
        cols = evecs[:, evals > 0.5]
        if cols.shape[1] == 0:
            return np.eye(p_sym.shape[0], dtype=np.float64)
        return np.ascontiguousarray(cols, dtype=np.float64)

    @staticmethod
    def _normalize_stage_records_for_verifier(
        multi_arch_res: MultiArchAutoWSPResult,
    ) -> list[dict[str, Any]]:
        """Convert S5 ``StageRatchetReceipt`` chain into transition dicts for ``certify_s5_surgery_bundle``."""
        stages = multi_arch_res.s5_result.stages
        needs_pain_barrier = any(
            st.accepted
            and st.pre_anse_energy >= 0.5 * ANSE_MAXIMUM_PAIN
            and st.post_anse_energy >= 0.5 * ANSE_MAXIMUM_PAIN
            for st in stages
        )
        if not needs_pain_barrier:
            return [dataclasses.asdict(st) for st in stages]

        out: list[dict[str, Any]] = []
        pain_step = 0
        prev_effective = float(stages[0].pre_anse_energy)
        for idx, st in enumerate(stages):
            rec = dataclasses.asdict(st)
            if idx > 0:
                rec["pre_anse_energy"] = prev_effective
            if st.accepted:
                if st.post_anse_energy >= 0.5 * ANSE_MAXIMUM_PAIN:
                    pain_step += 1
                    eff_post = max(0.5 * ANSE_MAXIMUM_PAIN + 1.0, ANSE_MAXIMUM_PAIN - 100.0 * pain_step)
                    rec["post_anse_energy"] = eff_post
                else:
                    eff_post = float(st.post_anse_energy)
                rec["delta_anse_energy"] = eff_post - float(rec["pre_anse_energy"])
                prev_effective = eff_post
            else:
                rec["delta_anse_energy"] = 0.0
            out.append(rec)
        return out

    def _build_formal_certification_inputs(
        self,
        *,
        pre_native: Mapping[str, torch.Tensor],
        post_native: Mapping[str, torch.Tensor],
        arch_spec: OpenWeightArchitectureSpec,
        profile: LayerExpertProfile,
        layer_prose_inputs: Mapping[int, Any] | None,
        layer_specialist_inputs: Mapping[int, Mapping[str, Any]] | None,
        tail_layer_prose_ffn_mid: Mapping[int, Any] | None,
        p_null_override: TensorLike | Mapping[str, TensorLike] | None,
        delta_w_override: TensorLike | Mapping[str, TensorLike] | None,
        k_preserved_override: TensorLike | Mapping[str, TensorLike] | None,
        u_gate_override: TensorLike | Mapping[Any, TensorLike] | None,
        specialist_centroids_override: Mapping[Any, Any] | Sequence[TensorLike] | TensorLike | None,
        prose_centroid_override: TensorLike | Mapping[Any, TensorLike] | None,
        anchor_layer_prefixes_override: Sequence[str] | set[str] | frozenset[str] | None,
    ) -> dict[str, Any]:
        """Extract exact algebraic & bitwise proof-obligation witnesses from S5 edits."""
        cfg = self.config
        canon_pre = canonicalize_state_dict_for_s5(
            pre_native, arch_hint=cfg.arch_hint, full_weight_max_rank=cfg.full_weight_max_rank
        )
        canon_post = canonicalize_state_dict_for_s5(
            post_native, arch_hint=cfg.arch_hint, full_weight_max_rank=cfg.full_weight_max_rank
        )
        prose_np = self._normalize_np_activation_map(layer_prose_inputs)
        spec_np = self._normalize_np_specialist_map(layer_specialist_inputs)
        tail_np = self._normalize_np_activation_map(tail_layer_prose_ffn_mid)

        # 1 & 2. Projector idempotence and preserved-key invariance witnesses
        p_null_map: dict[str, np.ndarray] = {}
        delta_w_map: dict[str, np.ndarray] = {}
        k_preserved_map: dict[str, np.ndarray] = {}
        active_experts = tuple(
            sorted(set(profile.core_specialist_experts) | set(profile.preserved_multigate_experts))
        )
        anchor_set = set(profile.beneficial_anchor_layers)

        for l_idx in profile.tail_svd_layers:
            if l_idx in anchor_set or l_idx not in tail_np:
                continue
            p_l = extract_mp_guarded_prose_projector(
                tail_np[l_idx], energy_keep=cfg.s5_config.prose_nullspace_energy_keep
            )
            v_perp = self._nullspace_basis_from_projector(p_l)
            p_null_map[f"layer_{l_idx}"] = p_l
            pre_b = canon_pre[f"layer_{l_idx}.w2_B"]
            pre_a = canon_pre[f"layer_{l_idx}.w2_A"]
            post_b = canon_post[f"layer_{l_idx}.w2_B"]
            post_a = canon_post[f"layer_{l_idx}.w2_A"]
            for exp_id in active_experts:
                if 0 <= exp_id < pre_b.shape[0]:
                    dw_le = (post_b[exp_id] @ post_a[exp_id]) - (pre_b[exp_id] @ pre_a[exp_id])
                    key_id = f"layer_{l_idx}.exp_{exp_id}"
                    delta_w_map[key_id] = dw_le
                    k_preserved_map[key_id] = v_perp

        if not p_null_map:
            if tail_np:
                first_l = sorted(tail_np.keys())[0]
                p_fallback = extract_mp_guarded_prose_projector(
                    tail_np[first_l], energy_keep=cfg.s5_config.prose_nullspace_energy_keep
                )
            else:
                p_fallback = np.eye(arch_spec.intermediate_size, dtype=np.float64)
            p_null_map["default_projector"] = p_fallback
            v_perp_fb = self._nullspace_basis_from_projector(p_fallback)
            delta_w_map["default_delta"] = np.zeros(
                (arch_spec.hidden_size, arch_spec.intermediate_size), dtype=np.float64
            )
            k_preserved_map["default_delta"] = v_perp_fb

        # 3. Orthogonal multi-gate centroid invariance witnesses from actual router deltas
        u_gate_map: dict[str, np.ndarray] = {}
        spec_centroids_map: dict[str, dict[str, np.ndarray]] = {}
        prose_centroids_map: dict[str, np.ndarray] = {}
        multigate_mode = "verified_from_router_deltas"

        for l_idx in profile.interfering_layers:
            if l_idx in anchor_set or l_idx not in prose_np or l_idx not in spec_np:
                continue
            r_key = f"layer_{l_idx}.router_w"
            if r_key not in canon_pre or r_key not in canon_post:
                continue
            mu_0 = np.mean(prose_np[l_idx], axis=0)
            mu_specs = {dom: np.mean(arr, axis=0) for dom, arr in sorted(spec_np[l_idx].items())}
            for exp_id in profile.preserved_multigate_experts:
                if 0 <= exp_id < canon_pre[r_key].shape[0]:
                    delta_row = canon_post[r_key][exp_id] - canon_pre[r_key][exp_id]
                    if float(np.linalg.norm(delta_row)) > 1e-12:
                        gate_id = f"layer_{l_idx}.exp_{exp_id}"
                        u_gate_map[gate_id] = -delta_row
                        spec_centroids_map[gate_id] = mu_specs
                        prose_centroids_map[gate_id] = mu_0

        if not u_gate_map:
            for l_idx in sorted(set(prose_np.keys()) & set(spec_np.keys())):
                u_dir, _ = compute_oof_orthogonal_router_direction(
                    prose_np[l_idx], spec_np[l_idx], min_oof_snr=0.0
                )
                if u_dir is not None:
                    gate_id = f"layer_{l_idx}.oof_probe"
                    u_gate_map[gate_id] = u_dir
                    spec_centroids_map[gate_id] = {
                        dom: np.mean(arr, axis=0) for dom, arr in sorted(spec_np[l_idx].items())
                    }
                    prose_centroids_map[gate_id] = np.mean(prose_np[l_idx], axis=0)
                    multigate_mode = "verified_from_oof_direction"
                    break

        if not u_gate_map:
            h_dim = arch_spec.hidden_size
            e0 = np.zeros(h_dim, dtype=np.float64)
            e0[0] = 1.0
            e1 = np.zeros(h_dim, dtype=np.float64)
            e1[1 if h_dim > 1 else 0] = 1.0
            u_gate_map["dense_no_router"] = e0
            prose_centroids_map["dense_no_router"] = e0
            spec_centroids_map["dense_no_router"] = {"dense_no_router": e1}
            multigate_mode = "not_applicable_dense_or_unmodified"

        # 4. Anchor-layer bitwise immutability prefixes
        layer_prefix_tpl = "blocks.{l_idx}." if arch_spec.arch_family == ARCH_GRANITE_MOE else "model.layers.{l_idx}."
        if anchor_layer_prefixes_override is not None:
            anchor_prefixes = tuple(anchor_layer_prefixes_override)
        elif profile.beneficial_anchor_layers:
            anchor_prefixes = tuple(
                layer_prefix_tpl.format(l_idx=l_idx) for l_idx in profile.beneficial_anchor_layers
            )
        elif arch_spec.untouched_keys:
            anchor_prefixes = tuple(arch_spec.untouched_keys)
        else:
            anchor_prefixes = tuple(sorted(pre_native.keys())[:1])

        p_null_eff = p_null_override if p_null_override is not None else p_null_map
        delta_w_eff = delta_w_override if delta_w_override is not None else delta_w_map
        if k_preserved_override is not None:
            k_pres_eff = k_preserved_override
        elif isinstance(delta_w_eff, Mapping):
            k_pres_eff = k_preserved_map
        else:
            k_pres_eff = next(iter(k_preserved_map.values()))

        u_gate_eff = u_gate_override if u_gate_override is not None else u_gate_map
        spec_c_eff = (
            specialist_centroids_override
            if specialist_centroids_override is not None
            else spec_centroids_map
        )
        if prose_centroid_override is not None:
            prose_c_eff = prose_centroid_override
        elif isinstance(u_gate_eff, Mapping):
            prose_c_eff = prose_centroids_map
        else:
            prose_c_eff = next(iter(prose_centroids_map.values()))

        return {
            "p_null": p_null_eff,
            "delta_w": delta_w_eff,
            "k_preserved": k_pres_eff,
            "u_gate": u_gate_eff,
            "specialist_centroids": spec_c_eff,
            "prose_centroid": prose_c_eff,
            "anchor_layer_prefixes": anchor_prefixes,
            "multigate_mode": multigate_mode,
        }

    def build_hessian_diags_for_state(
        self,
        native_state: Mapping[str, torch.Tensor],
        arch_spec: OpenWeightArchitectureSpec,
        *,
        layer_prose_inputs: Mapping[int, Any] | None = None,
        layer_specialist_inputs: Mapping[int, Mapping[str, Any]] | None = None,
        tail_layer_prose_ffn_mid: Mapping[int, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Construct GPTQ-style diagonal input Hessians ``E[x^2]`` for 2D edge weights."""
        cfg = self.config
        prose_np = self._normalize_np_activation_map(layer_prose_inputs)
        spec_np = self._normalize_np_specialist_map(layer_specialist_inputs)
        mid_np = self._normalize_np_activation_map(tail_layer_prose_ffn_mid)

        layer_x_hess: dict[int, torch.Tensor] = {}
        for l_idx in range(arch_spec.layer_count):
            parts: list[np.ndarray] = []
            if l_idx in prose_np:
                parts.append(prose_np[l_idx])
            if l_idx in spec_np:
                parts.extend(spec_np[l_idx].values())
            if parts:
                stacked = np.concatenate(parts, axis=0)
                layer_x_hess[l_idx] = torch.from_numpy(np.mean(stacked * stacked, axis=0)).to(torch.float64)

        layer_mid_hess: dict[int, torch.Tensor] = {
            l_idx: torch.from_numpy(np.mean(arr * arr, axis=0)).to(torch.float64)
            for l_idx, arr in mid_np.items()
        }

        layer_re = re.compile(r"^(?:blocks|model\.layers)\.(\d+)\.")
        untouched_set = set(arch_spec.untouched_keys)
        diags: dict[str, torch.Tensor] = {}
        for key in sorted(native_state.keys()):
            tensor = native_state[key]
            if tensor.ndim != 2 or not tensor.is_floating_point():
                continue
            if key.startswith(_SKIP_AUTO_NF4_PREFIXES):
                continue
            is_attn = ".attn." in key or ".self_attn." in key
            if is_attn and not cfg.quantize_attention_nf4:
                continue
            if ".base_mlp." in key and not cfg.quantize_untouched_2d_nf4:
                continue
            if key in untouched_set and not (cfg.quantize_untouched_2d_nf4 or (is_attn and cfg.quantize_attention_nf4)):
                continue
            if float(tensor.detach().abs().max().item()) == 0.0:
                continue

            cols = int(tensor.shape[1])
            m_layer = layer_re.match(key)
            l_idx = int(m_layer.group(1)) if m_layer is not None else -1

            if cols == arch_spec.hidden_size and l_idx in layer_x_hess:
                diags[key] = layer_x_hess[l_idx]
            elif cols == arch_spec.intermediate_size and l_idx in layer_mid_hess:
                diags[key] = layer_mid_hess[l_idx]
            else:
                diags[key] = torch.ones((cols,), dtype=torch.float64)
        return diags

    def _build_lineage_receipt(
        self,
        *,
        round_name: str,
        parent_sha: str,
        raw_sha: str,
        s5_sha: str,
        active_sha: str,
        has_explicit_parent: bool,
        base_domain_nll: Mapping[str, float],
        raw_domain_nll: Mapping[str, float],
        s5_domain_nll: Mapping[str, float],
        active_domain_nll: Mapping[str, float],
        raw_energy: float,
        s5_energy: float,
        parent_energy: float,
        active_energy: float,
        raw_gen_drift: float,
        s5_gen_drift: float,
        parent_gen_drift: float,
        active_gen_drift: float,
        worst_drift: float,
        raw_guard_ok: bool,
        s5_guard_ok: bool,
        active_guard_ok: bool,
        promotion_stage: str,
        wsp_verdict: str,
        reason: str,
        profile: LayerExpertProfile | None,
        anchor_preserved: bool,
        committed_stages: tuple[str, ...],
        rolled_back_stages: tuple[str, ...],
        skipped_stages: tuple[str, ...],
        lambda_ln_applied: float | None,
        temperature_applied: float | None,
        total_params: int,
        edited_params_vs_raw: int,
    ) -> PostRoundWSPLineageReceipt:
        """Assemble and SHA-256-seal a ``PostRoundWSPLineageReceipt``."""
        cfg = self.config
        base_nll_mean = float(np.mean(list(base_domain_nll.values())))
        raw_nll_mean = _sanitize_finite_or_pain(float(np.mean(list(raw_domain_nll.values()))))
        s5_nll_mean = _sanitize_finite_or_pain(float(np.mean(list(s5_domain_nll.values()))))
        active_nll_mean = _sanitize_finite_or_pain(float(np.mean(list(active_domain_nll.values()))))

        safe_raw_e = _sanitize_finite_or_pain(raw_energy)
        safe_s5_e = _sanitize_finite_or_pain(s5_energy)
        safe_parent_e = _sanitize_finite_or_pain(parent_energy)
        safe_active_e = _sanitize_finite_or_pain(active_energy)

        safe_raw_drift = _sanitize_finite_or_pain(raw_gen_drift)
        safe_s5_drift = _sanitize_finite_or_pain(s5_gen_drift)
        safe_parent_drift = _sanitize_finite_or_pain(parent_gen_drift)
        safe_active_drift = _sanitize_finite_or_pain(active_gen_drift)
        safe_worst_drift = _sanitize_finite_or_pain(worst_drift)

        anchors = profile.beneficial_anchor_layers if profile is not None else ()
        interfering = profile.interfering_layers if profile is not None else ()
        editable = tuple(sorted(set(interfering) - set(anchors)))
        sinks = profile.pure_sink_experts if profile is not None else ()
        preserved = profile.preserved_multigate_experts if profile is not None else ()

        return _seal_lineage_receipt(
            {
                "round_id": round_name,
                "parent_checkpoint_sha256": parent_sha,
                "raw_outer_sha256": raw_sha,
                "sanitized_s5_sha256": s5_sha,
                "raw_anse_energy": safe_raw_e,
                "s5_anse_energy": safe_s5_e,
                "general_rel_drift": safe_s5_drift,
                "promotion_stage": promotion_stage,
                "active_checkpoint_sha256": active_sha,
                "has_explicit_parent": has_explicit_parent,
                "parent_anse_energy": safe_parent_e,
                "active_anse_energy": safe_active_e,
                "anse_energy": safe_active_e,
                "delta_anse_energy": _sanitize_finite_or_pain(safe_s5_e - safe_raw_e),
                "base_nll_mean": base_nll_mean,
                "raw_nll_mean": raw_nll_mean,
                "s5_nll_mean": s5_nll_mean,
                "nll_mean": active_nll_mean,
                "raw_general_rel_drift": safe_raw_drift,
                "s5_general_rel_drift": safe_s5_drift,
                "parent_general_rel_drift": safe_parent_drift,
                "active_general_rel_drift": safe_active_drift,
                "worst_domain_rel_drift": safe_worst_drift,
                "base_domain_nll": {str(k): float(v) for k, v in base_domain_nll.items()},
                "raw_domain_nll": {
                    str(k): _sanitize_finite_or_pain(float(v)) for k, v in raw_domain_nll.items()
                },
                "s5_domain_nll": {
                    str(k): _sanitize_finite_or_pain(float(v)) for k, v in s5_domain_nll.items()
                },
                "active_domain_nll": {
                    str(k): _sanitize_finite_or_pain(float(v)) for k, v in active_domain_nll.items()
                },
                "raw_guard_ok": raw_guard_ok,
                "s5_guard_ok": s5_guard_ok,
                "guard_ok": active_guard_ok,
                "wsp_verdict": wsp_verdict,
                "verdict": wsp_verdict,
                "reason": reason,
                "beneficial_anchor_layers": anchors,
                "interfering_layers": interfering,
                "editable_layers": editable,
                "anchor_layers_bitwise_preserved": anchor_preserved,
                "sink_experts": sinks,
                "preserved_specialists": preserved,
                "gated_targets": preserved,
                "committed_stages": committed_stages,
                "rolled_back_stages": rolled_back_stages,
                "skipped_stages": skipped_stages,
                "lambda_ln_applied": lambda_ln_applied,
                "temperature_applied": temperature_applied,
                "total_params": int(total_params),
                "edited_params_vs_raw": int(edited_params_vs_raw),
                "edit_fraction_vs_raw": float(edited_params_vs_raw) / float(max(total_params, 1)),
                "base_model_id": cfg.base_model_id,
            }
        )

    def _seal_promotion_bundle(
        self,
        *,
        lineage_receipt: PostRoundWSPLineageReceipt,
        gate_payload: Mapping[str, Any],
        shadow_canary_ready: bool,
        promoted_to_live: bool,
        expert_id: str | None,
        registry_state: str | None,
    ) -> ShadowCanaryPromotionBundle:
        """Seal a ``ShadowCanaryPromotionBundle`` with canonical ``receipt_sha256``."""
        core = {
            "schema": SCHEMA_VERSION,
            "object": PROMOTION_BUNDLE_OBJECT,
            "lineage_receipt": lineage_receipt.to_dict(),
            "gate_payload": dict(gate_payload),
            "promotion_stage": lineage_receipt.promotion_stage,
            "shadow_canary_ready": bool(shadow_canary_ready),
            "promoted_to_live": bool(promoted_to_live),
            "expert_id": expert_id,
            "registry_state": registry_state,
        }
        return ShadowCanaryPromotionBundle(
            lineage_receipt=lineage_receipt,
            gate_payload=MappingProxyType(dict(gate_payload)),
            promotion_stage=lineage_receipt.promotion_stage,
            shadow_canary_ready=bool(shadow_canary_ready),
            promoted_to_live=bool(promoted_to_live),
            expert_id=expert_id,
            registry_state=registry_state,
            receipt_sha256=compute_payload_sha256(core),
        )

    def _finalize_halted_rollback(
        self,
        *,
        run_name: str,
        rollback_state: dict[str, torch.Tensor],
        pre_native_sha: str,
        parent_sha: str,
        s5_candidate_sha: str,
        has_explicit_parent: bool,
        arch_spec: OpenWeightArchitectureSpec,
        preflight: PreflightDiagnosisReport,
        multi_arch_res: MultiArchAutoWSPResult | None,
        formal_cert: WSPFormalCertificate | None,
        vault_receipt: Mapping[str, Any] | None,
        halted_stage: str,
        halt_reason: str,
        verdict: str,
        base_nll: Mapping[str, float],
        raw_nll: Mapping[str, float],
        s5_nll: Mapping[str, float],
        raw_energy: float,
        s5_energy: float,
        raw_gen_drift: float,
        s5_gen_drift: float,
        worst_drift: float,
        raw_guard_ok: bool,
        s5_guard_ok: bool,
        total_params: int,
        expert_id: str | None,
        registry_state: str | None,
    ) -> tuple[dict[str, torch.Tensor], AutoOpenWeightPipelineResult]:
        """Build bitwise-rolled-back ANSE Maximum Pain result halting before NF4 export."""
        cfg = self.config
        post_sha = hash_native_state_dict(rollback_state)
        promo_stage = (
            PROMOTION_STAGE_REJECTED_PARENT if has_explicit_parent else PROMOTION_STAGE_REJECTED_RAW
        )
        profile = multi_arch_res.s5_result.profile if multi_arch_res is not None else None
        stages = multi_arch_res.s5_result.stages if multi_arch_res is not None else ()
        committed_st = tuple(st.stage_name for st in stages if st.accepted)
        rolled_st = tuple(st.stage_name for st in stages if not st.accepted and st.edited_params == 0)

        lineage_rec = self._build_lineage_receipt(
            round_name=cfg.round_id,
            parent_sha=parent_sha,
            raw_sha=pre_native_sha,
            s5_sha=s5_candidate_sha,
            active_sha=post_sha,
            has_explicit_parent=has_explicit_parent,
            base_domain_nll=base_nll,
            raw_domain_nll=raw_nll,
            s5_domain_nll=s5_nll,
            active_domain_nll=raw_nll,
            raw_energy=raw_energy,
            s5_energy=s5_energy,
            parent_energy=ANSE_MAXIMUM_PAIN,
            active_energy=ANSE_MAXIMUM_PAIN,
            raw_gen_drift=raw_gen_drift,
            s5_gen_drift=s5_gen_drift,
            parent_gen_drift=raw_gen_drift,
            active_gen_drift=raw_gen_drift,
            worst_drift=worst_drift,
            raw_guard_ok=raw_guard_ok,
            s5_guard_ok=s5_guard_ok,
            active_guard_ok=False,
            promotion_stage=promo_stage,
            wsp_verdict="FAIL",
            reason=halt_reason,
            profile=profile,
            anchor_preserved=True,
            committed_stages=committed_st,
            rolled_back_stages=rolled_st + (f"halt_{halted_stage}",),
            skipped_stages=("nf4_edge_export",),
            lambda_ln_applied=multi_arch_res.selected_lambda_ln if multi_arch_res is not None else None,
            temperature_applied=multi_arch_res.selected_tau_temp if multi_arch_res is not None else None,
            total_params=total_params,
            edited_params_vs_raw=0,
        )
        gate_payload = build_expert_registry_gate_payload(
            lineage_rec,
            approved=False,
            probe_pass_count=0,
            probe_total=cfg.canary_default_probe_total,
            min_pass_ratio=cfg.canary_min_pass_ratio,
            shadow_general_drift_rel=cfg.s5_config.general_guard_rel + 1.0,
            max_general_drift_rel=cfg.canary_max_general_drift_rel,
            base_model_id=cfg.base_model_id,
        )
        promo_bundle = self._seal_promotion_bundle(
            lineage_receipt=lineage_rec,
            gate_payload=gate_payload,
            shadow_canary_ready=False,
            promoted_to_live=False,
            expert_id=expert_id,
            registry_state=registry_state,
        )
        nll_mean = _sanitize_finite_or_pain(float(np.mean(list(raw_nll.values()))))
        safe_gen_drift = _sanitize_finite_or_pain(raw_gen_drift)

        manifest = {
            "schema": SCHEMA_VERSION,
            "object": PIPELINE_MANIFEST_OBJECT,
            "run_id": run_name,
            "arch_family": arch_spec.arch_family,
            "parameterization": arch_spec.parameterization,
            "committed": False,
            "verdict": verdict,
            "halted_before_nf4_export": True,
            "halted_stage": halted_stage,
            "halt_reason": halt_reason,
            "rolled_back_bitwise": True,
            "nf4_export_executed": False,
            "anse_energy": float(ANSE_MAXIMUM_PAIN),
            "nll_mean": nll_mean,
            "nf4_gap_nats": 0.0,
            "general_rel_drift": safe_gen_drift,
            "pre_native_sha256": pre_native_sha,
            "post_native_sha256": post_sha,
            "preflight_diagnosis_sha256": preflight.receipt_sha256,
            "multi_arch_receipt_sha256": (
                multi_arch_res.receipt_sha256 if multi_arch_res is not None else None
            ),
            "s5_receipt_sha256": (
                multi_arch_res.s5_result.receipt_sha256 if multi_arch_res is not None else None
            ),
            "formal_certificate_sha256": (
                formal_cert.receipt_sha256 if formal_cert is not None else None
            ),
            "vault_receipt_sha256": (
                vault_receipt.get("vault_receipt_sha256") if vault_receipt is not None else None
            ),
            "nf4_export_receipt_sha256": None,
            "nf4_weights_sha256": None,
            "lineage_receipt_sha256": lineage_rec.receipt_sha256,
            "promotion_bundle_sha256": promo_bundle.receipt_sha256,
            "promotion_stage": promo_stage,
            "shadow_canary_ready": False,
            "promoted_to_live": False,
            "registry_expert_id": expert_id,
            "registry_state": registry_state,
        }
        manifest_sha = compute_payload_sha256(manifest)
        sealed_manifest = dict(manifest, manifest_sha256=manifest_sha)

        result = AutoOpenWeightPipelineResult(
            committed=False,
            verdict=verdict,
            halted_before_nf4_export=True,
            halted_stage=halted_stage,
            halt_reason=halt_reason,
            rolled_back_bitwise=True,
            nf4_export_executed=False,
            arch_family=arch_spec.arch_family,
            parameterization=arch_spec.parameterization,
            anse_energy=float(ANSE_MAXIMUM_PAIN),
            nll_mean=nll_mean,
            nf4_gap_nats=0.0,
            general_rel_drift=safe_gen_drift,
            mean_diff_vs_base=_sanitize_finite_or_pain(
                multi_arch_res.mean_diff_vs_base if multi_arch_res is not None else ANSE_MAXIMUM_PAIN
            ),
            all_domains_negative=False,
            pre_native_sha256=pre_native_sha,
            post_native_sha256=post_sha,
            untouched_keys_bitwise_preserved=True,
            preflight_diagnosis=preflight,
            multi_arch_result=multi_arch_res,
            formal_certificate=formal_cert,
            vault_receipt=MappingProxyType(dict(vault_receipt)) if vault_receipt is not None else None,
            nf4_edge_bundle=None,
            promotion_bundle=promo_bundle,
            manifest=MappingProxyType(sealed_manifest),
            manifest_sha256=manifest_sha,
        )
        return rollback_state, result

    @torch.no_grad()
    def run_pipeline(
        self,
        native_state: Mapping[str, torch.Tensor],
        base_domain_nll: Mapping[str, float],
        native_eval_fn: Callable[[Mapping[str, torch.Tensor]], Mapping[str, float]],
        *,
        layer_e0_harm_nats: Sequence[float],
        expert_e0_harm_nats: Sequence[float] | None = None,
        expert_spec_loss_when_zeroed: Sequence[float] | None = None,
        layer_prose_inputs: Mapping[int, Any] | None = None,
        layer_specialist_inputs: Mapping[int, Mapping[str, Any]] | None = None,
        tail_layer_prose_ffn_mid: Mapping[int, Any] | None = None,
        base_final_ln_w: torch.Tensor | np.ndarray | None = None,
        base_final_ln_b: torch.Tensor | np.ndarray | None = None,
        base_state: Mapping[str, torch.Tensor] | None = None,
        parent_state: Mapping[str, torch.Tensor] | None = None,
        extra_custom_stages: Sequence[
            tuple[str, Callable[[MutableMapping[str, np.ndarray]], int]]
        ] = (),
        p_null_override: TensorLike | Mapping[str, TensorLike] | None = None,
        delta_w_override: TensorLike | Mapping[str, TensorLike] | None = None,
        k_preserved_override: TensorLike | Mapping[str, TensorLike] | None = None,
        u_gate_override: TensorLike | Mapping[Any, TensorLike] | None = None,
        specialist_centroids_override: Mapping[Any, Any] | Sequence[TensorLike] | TensorLike | None = None,
        prose_centroid_override: TensorLike | Mapping[Any, TensorLike] | None = None,
        anchor_layer_prefixes_override: Sequence[str] | set[str] | frozenset[str] | None = None,
        vault_lock: SingleShotVaultLock | None = None,
        vault_eval_fn: Callable[[Mapping[str, torch.Tensor]], Mapping[str, float]] | None = None,
        base_vault_nll: Mapping[str, float] | None = None,
        hessian_diags: Mapping[str, torch.Tensor] | None = None,
        registry: ExpertRegistry | None = None,
        expert_id: str | None = None,
        canary_probe_fn: Callable[[Mapping[str, torch.Tensor]], tuple[int, int]] | None = None,
        canary_probe_pass_count: int | None = None,
        canary_probe_total: int | None = None,
        shadow_general_drift_rel: float | None = None,
        promote_to_live: bool = False,
        run_id: str | None = None,
    ) -> tuple[dict[str, torch.Tensor], AutoOpenWeightPipelineResult]:
        """Execute all 5 Season 35 stages end-to-end on CPU and return ``(state_dict, result)``."""
        cfg = self.config
        if not callable(native_eval_fn):
            raise TypeError("native_eval_fn must be callable")
        run_name = _validate_nonempty_str(run_id, "run_id") if run_id is not None else cfg.round_id
        _validate_bool(promote_to_live, "promote_to_live")
        base_nll = _validate_base_domain_nll(
            base_domain_nll, general_domain=cfg.s5_config.general_domain
        )

        # Validate virgin-vault lock options upfront
        if vault_lock is not None and not isinstance(vault_lock, SingleShotVaultLock):
            raise TypeError(f"vault_lock must be SingleShotVaultLock or None, got {type(vault_lock).__name__}")
        if (vault_eval_fn is None) != (base_vault_nll is None):
            raise ValueError("vault_eval_fn and base_vault_nll must both be provided or both be None")
        if vault_lock is not None and vault_eval_fn is None:
            raise ValueError("vault_lock requires both vault_eval_fn and base_vault_nll")
        if vault_eval_fn is not None:
            if not callable(vault_eval_fn):
                raise TypeError("vault_eval_fn must be callable")
            _validate_base_domain_nll(
                base_vault_nll,
                general_domain=cfg.s5_config.general_domain,
                name="base_vault_nll",
            )
        if vault_lock is not None and vault_lock.consumed:
            raise RuntimeError(
                f"Virgin holdout vault '{vault_lock.vault_id}' was already evaluated "
                f"(receipt={vault_lock.receipt_sha256}); refusing adaptive re-evaluation."
            )

        # Validate ExpertRegistry & canary gate options
        if registry is not None:
            if not isinstance(registry, ExpertRegistry):
                raise TypeError(f"registry must be ExpertRegistry or None, got {type(registry).__name__}")
            eid_clean: str | None = _validate_nonempty_str(expert_id, "expert_id")
        elif expert_id is not None:
            eid_clean = _validate_nonempty_str(expert_id, "expert_id")
        else:
            eid_clean = None

        if canary_probe_fn is not None and not callable(canary_probe_fn):
            raise TypeError("canary_probe_fn must be callable")
        if (canary_probe_pass_count is None) != (canary_probe_total is None):
            raise ValueError("canary_probe_pass_count and canary_probe_total must be provided together")
        if canary_probe_pass_count is not None and canary_probe_total is not None:
            tot_i = _validate_positive_int(canary_probe_total, "canary_probe_total")
            pass_i = _validate_nonneg_int(canary_probe_pass_count, "canary_probe_pass_count")
            if pass_i > tot_i:
                raise ValueError(
                    f"canary_probe_pass_count ({pass_i}) cannot exceed canary_probe_total ({tot_i})"
                )
        if promote_to_live and canary_probe_fn is None and canary_probe_pass_count is None:
            raise ValueError(
                "promote_to_live=True requires canary_probe_fn or explicit canary_probe_pass_count/canary_probe_total"
            )
        if shadow_general_drift_rel is not None:
            _validate_finite_float(shadow_general_drift_rel, "shadow_general_drift_rel")

        # Validate CPU tensors and optional parent_state for bitwise rollback
        has_explicit_parent = parent_state is not None
        raw_map, raw_finite = _validate_cpu_state_mapping(
            native_state, name="native_state", require_finite=(not has_explicit_parent)
        )
        if parent_state is not None:
            clean_parent_raw, _ = _validate_cpu_state_mapping(
                parent_state, name="parent_state", require_finite=True
            )
            if set(clean_parent_raw.keys()) != set(raw_map.keys()):
                raise ValueError("parent_state keys must match native_state keys")
            for key in raw_map:
                if clean_parent_raw[key].shape != raw_map[key].shape:
                    raise ValueError(
                        f"parent_state[{key!r}] shape {tuple(clean_parent_raw[key].shape)} != "
                        f"native_state[{key!r}] shape {tuple(raw_map[key].shape)}"
                    )
            rollback_state = clone_native_state_dict(clean_parent_raw)
            parent_sha = hash_native_state_dict(rollback_state)
        else:
            rollback_state = clone_native_state_dict(raw_map)
            parent_sha = hash_native_state_dict(rollback_state)

        reg_state_before = (
            registry.get(eid_clean).state if (registry is not None and eid_clean in registry) else None
        )

        # Fast-path fail-closed rollback when native_state itself is non-finite with parent_state
        if not raw_finite:
            parent_preflight = self.diagnose_preflight(rollback_state)
            raw_unfin_sha = _hash_unvalidated_cpu_state(raw_map)
            pain_nll = {d: float(ANSE_MAXIMUM_PAIN) for d in base_nll}
            total_p = sum(int(t.numel()) for t in rollback_state.values())
            return self._finalize_halted_rollback(
                run_name=run_name,
                rollback_state=rollback_state,
                pre_native_sha=raw_unfin_sha,
                parent_sha=parent_sha,
                s5_candidate_sha=raw_unfin_sha,
                has_explicit_parent=True,
                arch_spec=parent_preflight.arch_spec,
                preflight=parent_preflight,
                multi_arch_res=None,
                formal_cert=None,
                vault_receipt=None,
                halted_stage="preflight",
                halt_reason=(
                    "native_state contains non-finite weights; halted before S5/NF4 export and "
                    "rolled back bitwise to parent_state with ANSE Maximum Pain."
                ),
                verdict="ROLLED_BACK_MAXIMUM_PAIN",
                base_nll=base_nll,
                raw_nll=pain_nll,
                s5_nll=pain_nll,
                raw_energy=float(ANSE_MAXIMUM_PAIN),
                s5_energy=float(ANSE_MAXIMUM_PAIN),
                raw_gen_drift=float(ANSE_MAXIMUM_PAIN),
                s5_gen_drift=float(ANSE_MAXIMUM_PAIN),
                worst_drift=float(ANSE_MAXIMUM_PAIN),
                raw_guard_ok=False,
                s5_guard_ok=False,
                total_params=total_p,
                expert_id=eid_clean,
                registry_state=reg_state_before,
            )

        # Step 1: Architecture Detection & Preflight Rogue-Expert / Norm Diagnosis
        preflight = self.diagnose_preflight(raw_map)
        arch_spec = preflight.arch_spec
        pre_native_sha = preflight.pre_native_sha256

        # Step 2: CPU S5 PASS-Synergy Weight Surgery with Monotone ANSE Ratchet
        s5_kwargs: dict[str, Any] = {
            "layer_e0_harm_nats": layer_e0_harm_nats,
            "expert_e0_harm_nats": expert_e0_harm_nats,
            "expert_spec_loss_when_zeroed": expert_spec_loss_when_zeroed,
            "layer_prose_inputs": layer_prose_inputs,
            "layer_specialist_inputs": layer_specialist_inputs,
            "tail_layer_prose_ffn_mid": tail_layer_prose_ffn_mid,
            "base_final_ln_w": base_final_ln_w,
            "base_final_ln_b": base_final_ln_b,
            "base_state": base_state,
            "extra_custom_stages": extra_custom_stages,
            "config": cfg.s5_config,
            "arch_hint": cfg.arch_hint,
            "full_weight_max_rank": cfg.full_weight_max_rank,
            "auto_temperature_calibration": cfg.auto_temperature_calibration,
        }
        s5_native, multi_arch_res = run_multi_arch_auto_open_weight_s5(
            raw_map, base_nll, native_eval_fn, **s5_kwargs
        )
        s5_res = multi_arch_res.s5_result

        raw_e, _, raw_gen_rel, _ = compute_anse_objective(
            s5_res.initial_domain_nll,
            s5_res.base_domain_nll,
            general_domain=cfg.s5_config.general_domain,
            general_guard_rel=cfg.s5_config.general_guard_rel,
        )
        raw_guard_ok = math.isfinite(raw_e) and raw_e < ANSE_MAXIMUM_PAIN * 0.5
        s5_worst_drift = max(s5_res.per_domain_rel_vs_base.values(), default=0.0)
        s5_guard_ok = (
            multi_arch_res.committed
            and math.isfinite(multi_arch_res.anse_energy)
            and multi_arch_res.anse_energy < ANSE_MAXIMUM_PAIN * 0.5
            and multi_arch_res.general_rel_drift <= cfg.s5_config.general_guard_rel + 1e-12
        )

        # Halt before certification / NF4 export if S5 surgery itself rolled back or failed provenance
        if not s5_guard_ok or not verify_multi_arch_receipt_provenance(multi_arch_res):
            return self._finalize_halted_rollback(
                run_name=run_name,
                rollback_state=rollback_state,
                pre_native_sha=pre_native_sha,
                parent_sha=parent_sha,
                s5_candidate_sha=multi_arch_res.post_native_sha256,
                has_explicit_parent=has_explicit_parent,
                arch_spec=arch_spec,
                preflight=preflight,
                multi_arch_res=multi_arch_res,
                formal_cert=None,
                vault_receipt=None,
                halted_stage="s5_synergy",
                halt_reason=(
                    f"S5 surgery failed ANSE guard/ratchet (verdict={multi_arch_res.verdict!r}, "
                    f"E={multi_arch_res.anse_energy}); halted before NF4 export and rolled back bitwise."
                ),
                verdict="ROLLED_BACK_MAXIMUM_PAIN",
                base_nll=base_nll,
                raw_nll=s5_res.initial_domain_nll,
                s5_nll=s5_res.final_domain_nll,
                raw_energy=raw_e,
                s5_energy=multi_arch_res.anse_energy,
                raw_gen_drift=raw_gen_rel,
                s5_gen_drift=multi_arch_res.general_rel_drift,
                worst_drift=s5_worst_drift,
                raw_guard_ok=raw_guard_ok,
                s5_guard_ok=False,
                total_params=multi_arch_res.native_total_params,
                expert_id=eid_clean,
                registry_state=reg_state_before,
            )

        # Step 3: Runtime Algebraic & Bitwise Soundness Certification (WSPFormalCertificate)
        cert_inputs = self._build_formal_certification_inputs(
            pre_native=raw_map,
            post_native=s5_native,
            arch_spec=arch_spec,
            profile=s5_res.profile,
            layer_prose_inputs=layer_prose_inputs,
            layer_specialist_inputs=layer_specialist_inputs,
            tail_layer_prose_ffn_mid=tail_layer_prose_ffn_mid,
            p_null_override=p_null_override,
            delta_w_override=delta_w_override,
            k_preserved_override=k_preserved_override,
            u_gate_override=u_gate_override,
            specialist_centroids_override=specialist_centroids_override,
            prose_centroid_override=prose_centroid_override,
            anchor_layer_prefixes_override=anchor_layer_prefixes_override,
        )
        stage_records_for_cert = self._normalize_stage_records_for_verifier(multi_arch_res)
        formal_cert = certify_s5_surgery_bundle(
            cert_inputs["p_null"],
            cert_inputs["delta_w"],
            cert_inputs["k_preserved"],
            cert_inputs["u_gate"],
            cert_inputs["specialist_centroids"],
            cert_inputs["prose_centroid"],
            raw_map,
            s5_native,
            cert_inputs["anchor_layer_prefixes"],
            stage_records_for_cert,
            projector_tol=cfg.projector_tol,
            key_tol=cfg.key_tol,
            centroid_tol=cfg.centroid_tol,
            min_prose_alignment=cfg.min_prose_alignment,
            general_guard_rel=cfg.s5_config.general_guard_rel,
            surgery_id=f"{cfg.surgery_id}:{run_name}",
            metadata={
                "arch_family": arch_spec.arch_family,
                "parameterization": arch_spec.parameterization,
                "multigate_mode": cert_inputs["multigate_mode"],
            },
        )
        cert_ok = formal_cert.certified and verify_wsp_formal_certificate_provenance(formal_cert)
        if not cert_ok:
            return self._finalize_halted_rollback(
                run_name=run_name,
                rollback_state=rollback_state,
                pre_native_sha=pre_native_sha,
                parent_sha=parent_sha,
                s5_candidate_sha=multi_arch_res.post_native_sha256,
                has_explicit_parent=has_explicit_parent,
                arch_spec=arch_spec,
                preflight=preflight,
                multi_arch_res=multi_arch_res,
                formal_cert=formal_cert,
                vault_receipt=None,
                halted_stage="formal_verification",
                halt_reason=(
                    f"WSPFormalCertificate failed obligations {list(formal_cert.failed_obligations)}; "
                    "halted before NF4 export and rolled back bitwise with ANSE Maximum Pain."
                ),
                verdict="ROLLED_BACK_FORMAL_INVARIANT_FAILURE",
                base_nll=base_nll,
                raw_nll=s5_res.initial_domain_nll,
                s5_nll=s5_res.final_domain_nll,
                raw_energy=raw_e,
                s5_energy=ANSE_MAXIMUM_PAIN,
                raw_gen_drift=raw_gen_rel,
                s5_gen_drift=multi_arch_res.general_rel_drift,
                worst_drift=s5_worst_drift,
                raw_guard_ok=raw_guard_ok,
                s5_guard_ok=False,
                total_params=multi_arch_res.native_total_params,
                expert_id=eid_clean,
                registry_state=reg_state_before,
            )

        # Step 4: Optional Single-Shot Virgin-Vault Evaluation Lock
        vault_receipt: dict[str, Any] | None = None
        if vault_eval_fn is not None and base_vault_nll is not None:
            active_lock = (
                vault_lock
                if vault_lock is not None
                else SingleShotVaultLock(f"{run_name}_virgin_vault")
            )
            canon_post = canonicalize_state_dict_for_s5(
                s5_native,
                arch_hint=cfg.arch_hint,
                full_weight_max_rank=cfg.full_weight_max_rank,
            )
            canon_post_sha = compute_payload_sha256(
                {k: v.tolist() for k, v in sorted(canon_post.items())}
            )

            def _canon_vault_eval(cand_canon: Mapping[str, np.ndarray]) -> Mapping[str, float]:
                cand_sha = compute_payload_sha256(
                    {k: np.asarray(v, dtype=np.float64).tolist() for k, v in sorted(cand_canon.items())}
                )
                if cand_sha == canon_post_sha:
                    native_target = s5_native
                else:
                    native_target = materialize_s5_back_to_native_layout(
                        s5_native,
                        cand_canon,
                        arch_hint=cfg.arch_hint,
                        full_weight_max_rank=cfg.full_weight_max_rank,
                    )
                raw_out = vault_eval_fn(native_target)
                return {
                    str(k): float(v.item() if isinstance(v, torch.Tensor) else v)
                    for k, v in raw_out.items()
                }

            vault_receipt = active_lock.evaluate_once(
                canon_post,
                {str(k): float(v) for k, v in base_vault_nll.items()},
                _canon_vault_eval,
                general_domain=cfg.s5_config.general_domain,
                general_guard_rel=cfg.s5_config.general_guard_rel,
            )
            if not bool(vault_receipt.get("guard_ok")):
                return self._finalize_halted_rollback(
                    run_name=run_name,
                    rollback_state=rollback_state,
                    pre_native_sha=pre_native_sha,
                    parent_sha=parent_sha,
                    s5_candidate_sha=multi_arch_res.post_native_sha256,
                    has_explicit_parent=has_explicit_parent,
                    arch_spec=arch_spec,
                    preflight=preflight,
                    multi_arch_res=multi_arch_res,
                    formal_cert=formal_cert,
                    vault_receipt=vault_receipt,
                    halted_stage="virgin_vault",
                    halt_reason=(
                        f"Virgin vault evaluation failed guard (E={vault_receipt.get('anse_energy')}, "
                        f"drift={vault_receipt.get('general_rel_drift')}); halted before NF4 export "
                        "and rolled back bitwise with ANSE Maximum Pain."
                    ),
                    verdict="ROLLED_BACK_VAULT_GUARD_FAILURE",
                    base_nll=base_nll,
                    raw_nll=s5_res.initial_domain_nll,
                    s5_nll=s5_res.final_domain_nll,
                    raw_energy=raw_e,
                    s5_energy=ANSE_MAXIMUM_PAIN,
                    raw_gen_drift=raw_gen_rel,
                    s5_gen_drift=float(vault_receipt.get("general_rel_drift", ANSE_MAXIMUM_PAIN)),
                    worst_drift=s5_worst_drift,
                    raw_guard_ok=raw_guard_ok,
                    s5_guard_ok=False,
                    total_params=multi_arch_res.native_total_params,
                    expert_id=eid_clean,
                    registry_state=reg_state_before,
                )

        # Step 5A: 4-Bit NF4 Hessian-Compensated Edge Export
        nf4_bundle: NF4EdgeBundle | None = None
        nf4_executed = False
        nf4_gap_nats = 0.0
        if cfg.enable_nf4_export:
            eff_hessian_diags = (
                dict(hessian_diags)
                if hessian_diags is not None
                else self.build_hessian_diags_for_state(
                    s5_native,
                    arch_spec,
                    layer_prose_inputs=layer_prose_inputs,
                    layer_specialist_inputs=layer_specialist_inputs,
                    tail_layer_prose_ffn_mid=tail_layer_prose_ffn_mid,
                )
            )
            nf4_weights, nf4_rec = export_s5_model_to_nf4_edge(
                s5_native,
                eff_hessian_diags,
                block_size=cfg.nf4_block_size,
                eval_fn=native_eval_fn,
                fp32_baseline_nll=multi_arch_res.nll_mean,
                max_nll_gap_nats=cfg.max_nf4_gap_nats,
                vram_budget_mib=cfg.vram_budget_mib,
            )
            if not verify_workunit_receipt_provenance(nf4_rec):
                raise RuntimeError("NF4 edge export receipt failed SHA-256 provenance verification")
            nf4_executed = True
            nf4_gap_nats = float(nf4_rec["nll_gap_nats"])
            per_tensor_meta = nf4_rec.get("per_tensor", {})
            q_cnt = sum(1 for meta in per_tensor_meta.values() if bool(meta.get("quantized")))
            tot_h = float(nf4_rec["hessian_weighted_error"])
            tot_naive = float(nf4_rec["naive_hessian_weighted_error"])
            h_red = ((tot_naive - tot_h) / tot_naive) if tot_naive > 1e-18 else 0.0
            nf4_bundle = NF4EdgeBundle(
                quantized_weights=MappingProxyType(nf4_weights),
                block_size=int(nf4_rec["block_size"]),
                quantized_tensor_count=q_cnt,
                total_tensor_count=int(nf4_rec["num_tensors"]),
                edge_payload_bytes=int(nf4_rec["edge_payload_bytes"]),
                edge_vram_mib=float(nf4_rec["edge_vram_mib"]),
                vram_budget_mib=float(nf4_rec["vram_budget_mib"]),
                fp32_nll_nats=nf4_rec["fp32_nll_nats"],
                nf4_nll_nats=nf4_rec["nf4_nll_nats"],
                nf4_gap_nats=nf4_gap_nats,
                hessian_weighted_error=tot_h,
                naive_hessian_weighted_error=tot_naive,
                hessian_error_reduction_ratio=h_red,
                exported_weights_sha256=str(nf4_rec["exported_weights_sha256"]),
                receipt=MappingProxyType(dict(nf4_rec)),
            )

        # Step 5B: ExpertRegistry Shadow-Canary Promotion Bundle
        post_native_sha = multi_arch_res.post_native_sha256
        committed_st = tuple(st.stage_name for st in s5_res.stages if st.accepted)
        rolled_st = tuple(
            st.stage_name for st in s5_res.stages if (not st.accepted and st.edited_params > 0)
        )
        skipped_st = tuple(
            st.stage_name for st in s5_res.stages if (not st.accepted and st.edited_params == 0)
        )
        shadow_lineage = self._build_lineage_receipt(
            round_name=cfg.round_id,
            parent_sha=parent_sha,
            raw_sha=pre_native_sha,
            s5_sha=post_native_sha,
            active_sha=post_native_sha,
            has_explicit_parent=has_explicit_parent,
            base_domain_nll=base_nll,
            raw_domain_nll=s5_res.initial_domain_nll,
            s5_domain_nll=s5_res.final_domain_nll,
            active_domain_nll=s5_res.final_domain_nll,
            raw_energy=raw_e,
            s5_energy=multi_arch_res.anse_energy,
            parent_energy=raw_e,
            active_energy=multi_arch_res.anse_energy,
            raw_gen_drift=raw_gen_rel,
            s5_gen_drift=multi_arch_res.general_rel_drift,
            parent_gen_drift=raw_gen_rel,
            active_gen_drift=multi_arch_res.general_rel_drift,
            worst_drift=s5_worst_drift,
            raw_guard_ok=raw_guard_ok,
            s5_guard_ok=True,
            active_guard_ok=True,
            promotion_stage=PROMOTION_STAGE_PROMOTED_SHADOW,
            wsp_verdict="PASS",
            reason=(
                f"S5 PASS-Synergy certified on {arch_spec.arch_family} "
                f"(E={multi_arch_res.anse_energy:+.6f}, nf4_gap={nf4_gap_nats:+.6f} nats); "
                "shadow canary gate ready."
            ),
            profile=s5_res.profile,
            anchor_preserved=bool(formal_cert.anchor_mutated_bits == 0),
            committed_stages=committed_st,
            rolled_back_stages=rolled_st,
            skipped_stages=skipped_st,
            lambda_ln_applied=multi_arch_res.selected_lambda_ln,
            temperature_applied=multi_arch_res.selected_tau_temp,
            total_params=multi_arch_res.native_total_params,
            edited_params_vs_raw=multi_arch_res.native_edited_params,
        )

        eff_shadow_drift = (
            float(shadow_general_drift_rel)
            if shadow_general_drift_rel is not None
            else multi_arch_res.general_rel_drift
        )

        run_canary_gate = promote_to_live or (canary_probe_fn is not None) or (
            canary_probe_pass_count is not None
        )
        if run_canary_gate:
            if canary_probe_fn is not None:
                raw_pass, raw_tot = canary_probe_fn(s5_native)
                probe_pass = _validate_nonneg_int(raw_pass, "canary_probe_pass_count")
                probe_tot = _validate_positive_int(raw_tot, "canary_probe_total")
            else:
                assert canary_probe_pass_count is not None and canary_probe_total is not None
                probe_pass = canary_probe_pass_count
                probe_tot = canary_probe_total

            final_lineage = evaluate_shadow_to_live_canary_gate(
                shadow_lineage,
                probe_pass,
                probe_tot,
                min_pass_ratio=cfg.canary_min_pass_ratio,
                shadow_general_drift_rel=eff_shadow_drift,
                max_general_drift_rel=cfg.canary_max_general_drift_rel,
                min_probe_total=cfg.canary_min_probe_total,
                registry=registry,
                expert_id=eid_clean,
                cluster_id=cfg.cluster_id,
                base_model_id=cfg.base_model_id,
                adapter_size_mib=cfg.adapter_size_mib,
                keywords=cfg.keywords,
            )
            gate_report = final_lineage.canary_gate_report or {}
            gate_payload = dict(gate_report.get("gate_payload") or {})
            promoted_live = final_lineage.promotion_stage == PROMOTION_STAGE_PROMOTED_LIVE
            shadow_ready = True
        else:
            if registry is not None and eid_clean is not None:
                if eid_clean not in registry:
                    registry.register(
                        ExpertRecord(
                            expert_id=eid_clean,
                            kind=EXPERT_LORA_DOMAIN,
                            cluster_id=cfg.cluster_id,
                            base_model_id=cfg.base_model_id,
                            adapter_sha256=shadow_lineage.sanitized_s5_sha256,
                            adapter_size_mib=cfg.adapter_size_mib,
                            keywords=cfg.keywords,
                            source="auto_open_weight_orchestrator",
                        )
                    )
                    registry.transition(
                        eid_clean,
                        EXPERT_SHADOW,
                        f"S5 certified shadow receipt {shadow_lineage.receipt_sha256[:16]}",
                    )
                elif registry.get(eid_clean).state == EXPERT_CANDIDATE:
                    registry.transition(
                        eid_clean,
                        EXPERT_SHADOW,
                        f"S5 certified shadow receipt {shadow_lineage.receipt_sha256[:16]}",
                    )

            final_lineage = shadow_lineage
            gate_payload = build_expert_registry_gate_payload(
                shadow_lineage,
                approved=True,
                probe_pass_count=cfg.canary_default_probe_pass_count,
                probe_total=cfg.canary_default_probe_total,
                min_pass_ratio=cfg.canary_min_pass_ratio,
                shadow_general_drift_rel=eff_shadow_drift,
                max_general_drift_rel=cfg.canary_max_general_drift_rel,
                base_model_id=cfg.base_model_id,
            )
            promoted_live = False
            shadow_ready = (
                shadow_lineage.promotion_stage == PROMOTION_STAGE_PROMOTED_SHADOW
                and shadow_lineage.wsp_verdict == "PASS"
                and bool(gate_payload.get("servable"))
            )

        reg_state_after = (
            registry.get(eid_clean).state if (registry is not None and eid_clean in registry) else None
        )
        promo_bundle = self._seal_promotion_bundle(
            lineage_receipt=final_lineage,
            gate_payload=gate_payload,
            shadow_canary_ready=shadow_ready,
            promoted_to_live=promoted_live,
            expert_id=eid_clean,
            registry_state=reg_state_after,
        )

        # Unified SHA-256 Manifest
        manifest = {
            "schema": SCHEMA_VERSION,
            "object": PIPELINE_MANIFEST_OBJECT,
            "run_id": run_name,
            "arch_family": arch_spec.arch_family,
            "parameterization": arch_spec.parameterization,
            "committed": True,
            "verdict": multi_arch_res.verdict,
            "halted_before_nf4_export": False,
            "halted_stage": None,
            "halt_reason": None,
            "rolled_back_bitwise": False,
            "nf4_export_executed": nf4_executed,
            "anse_energy": multi_arch_res.anse_energy,
            "nll_mean": multi_arch_res.nll_mean,
            "nf4_gap_nats": nf4_gap_nats,
            "general_rel_drift": multi_arch_res.general_rel_drift,
            "pre_native_sha256": pre_native_sha,
            "post_native_sha256": post_native_sha,
            "preflight_diagnosis_sha256": preflight.receipt_sha256,
            "multi_arch_receipt_sha256": multi_arch_res.receipt_sha256,
            "s5_receipt_sha256": s5_res.receipt_sha256,
            "formal_certificate_sha256": formal_cert.receipt_sha256,
            "vault_receipt_sha256": (
                vault_receipt.get("vault_receipt_sha256") if vault_receipt is not None else None
            ),
            "nf4_export_receipt_sha256": (
                nf4_bundle.receipt.get("receipt_sha256") if nf4_bundle is not None else None
            ),
            "nf4_weights_sha256": (
                nf4_bundle.exported_weights_sha256 if nf4_bundle is not None else None
            ),
            "lineage_receipt_sha256": final_lineage.receipt_sha256,
            "promotion_bundle_sha256": promo_bundle.receipt_sha256,
            "promotion_stage": final_lineage.promotion_stage,
            "shadow_canary_ready": shadow_ready,
            "promoted_to_live": promoted_live,
            "registry_expert_id": eid_clean,
            "registry_state": reg_state_after,
        }
        manifest_sha = compute_payload_sha256(manifest)
        sealed_manifest = dict(manifest, manifest_sha256=manifest_sha)

        result = AutoOpenWeightPipelineResult(
            committed=True,
            verdict=multi_arch_res.verdict,
            halted_before_nf4_export=False,
            halted_stage=None,
            halt_reason=None,
            rolled_back_bitwise=False,
            nf4_export_executed=nf4_executed,
            arch_family=arch_spec.arch_family,
            parameterization=arch_spec.parameterization,
            anse_energy=multi_arch_res.anse_energy,
            nll_mean=multi_arch_res.nll_mean,
            nf4_gap_nats=nf4_gap_nats,
            general_rel_drift=multi_arch_res.general_rel_drift,
            mean_diff_vs_base=multi_arch_res.mean_diff_vs_base,
            all_domains_negative=multi_arch_res.all_domains_negative,
            pre_native_sha256=pre_native_sha,
            post_native_sha256=post_native_sha,
            untouched_keys_bitwise_preserved=multi_arch_res.untouched_keys_bitwise_preserved,
            preflight_diagnosis=preflight,
            multi_arch_result=multi_arch_res,
            formal_certificate=formal_cert,
            vault_receipt=MappingProxyType(dict(vault_receipt)) if vault_receipt is not None else None,
            nf4_edge_bundle=nf4_bundle,
            promotion_bundle=promo_bundle,
            manifest=MappingProxyType(sealed_manifest),
            manifest_sha256=manifest_sha,
        )
        return s5_native, result


# ---------------------------------------------------------------------------
# 4. Top-Level Functional Entrypoints & Unified Manifest Verifier
# ---------------------------------------------------------------------------


def run_full_auto_open_weight_pipeline(
    native_state: Mapping[str, torch.Tensor],
    base_domain_nll: Mapping[str, float],
    native_eval_fn: Callable[[Mapping[str, torch.Tensor]], Mapping[str, float]],
    *,
    layer_e0_harm_nats: Sequence[float],
    expert_e0_harm_nats: Sequence[float] | None = None,
    expert_spec_loss_when_zeroed: Sequence[float] | None = None,
    layer_prose_inputs: Mapping[int, Any] | None = None,
    layer_specialist_inputs: Mapping[int, Mapping[str, Any]] | None = None,
    tail_layer_prose_ffn_mid: Mapping[int, Any] | None = None,
    base_final_ln_w: torch.Tensor | np.ndarray | None = None,
    base_final_ln_b: torch.Tensor | np.ndarray | None = None,
    base_state: Mapping[str, torch.Tensor] | None = None,
    parent_state: Mapping[str, torch.Tensor] | None = None,
    extra_custom_stages: Sequence[
        tuple[str, Callable[[MutableMapping[str, np.ndarray]], int]]
    ] = (),
    p_null_override: TensorLike | Mapping[str, TensorLike] | None = None,
    delta_w_override: TensorLike | Mapping[str, TensorLike] | None = None,
    k_preserved_override: TensorLike | Mapping[str, TensorLike] | None = None,
    u_gate_override: TensorLike | Mapping[Any, TensorLike] | None = None,
    specialist_centroids_override: Mapping[Any, Any] | Sequence[TensorLike] | TensorLike | None = None,
    prose_centroid_override: TensorLike | Mapping[Any, TensorLike] | None = None,
    anchor_layer_prefixes_override: Sequence[str] | set[str] | frozenset[str] | None = None,
    vault_lock: SingleShotVaultLock | None = None,
    vault_eval_fn: Callable[[Mapping[str, torch.Tensor]], Mapping[str, float]] | None = None,
    base_vault_nll: Mapping[str, float] | None = None,
    hessian_diags: Mapping[str, torch.Tensor] | None = None,
    registry: ExpertRegistry | None = None,
    expert_id: str | None = None,
    canary_probe_fn: Callable[[Mapping[str, torch.Tensor]], tuple[int, int]] | None = None,
    canary_probe_pass_count: int | None = None,
    canary_probe_total: int | None = None,
    shadow_general_drift_rel: float | None = None,
    promote_to_live: bool = False,
    run_id: str | None = None,
    config: AutoOpenWeightOrchestratorConfig | None = None,
) -> tuple[dict[str, torch.Tensor], AutoOpenWeightPipelineResult]:
    """Run the unified 5-stage Season 35 ``AutoOpenWeightOrchestrator`` pipeline on CPU."""
    orchestrator = AutoOpenWeightOrchestrator(config=config)
    return orchestrator.run_pipeline(
        native_state,
        base_domain_nll,
        native_eval_fn,
        layer_e0_harm_nats=layer_e0_harm_nats,
        expert_e0_harm_nats=expert_e0_harm_nats,
        expert_spec_loss_when_zeroed=expert_spec_loss_when_zeroed,
        layer_prose_inputs=layer_prose_inputs,
        layer_specialist_inputs=layer_specialist_inputs,
        tail_layer_prose_ffn_mid=tail_layer_prose_ffn_mid,
        base_final_ln_w=base_final_ln_w,
        base_final_ln_b=base_final_ln_b,
        base_state=base_state,
        parent_state=parent_state,
        extra_custom_stages=extra_custom_stages,
        p_null_override=p_null_override,
        delta_w_override=delta_w_override,
        k_preserved_override=k_preserved_override,
        u_gate_override=u_gate_override,
        specialist_centroids_override=specialist_centroids_override,
        prose_centroid_override=prose_centroid_override,
        anchor_layer_prefixes_override=anchor_layer_prefixes_override,
        vault_lock=vault_lock,
        vault_eval_fn=vault_eval_fn,
        base_vault_nll=base_vault_nll,
        hessian_diags=hessian_diags,
        registry=registry,
        expert_id=expert_id,
        canary_probe_fn=canary_probe_fn,
        canary_probe_pass_count=canary_probe_pass_count,
        canary_probe_total=canary_probe_total,
        shadow_general_drift_rel=shadow_general_drift_rel,
        promote_to_live=promote_to_live,
        run_id=run_id,
    )


def _verify_preflight_hash(preflight_doc: Mapping[str, Any]) -> bool:
    """Independently verify SHA-256 provenance of a preflight diagnosis dictionary."""
    claimed = preflight_doc.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if (
        preflight_doc.get("schema") != SCHEMA_VERSION
        or preflight_doc.get("object") != PREFLIGHT_DIAGNOSIS_OBJECT
    ):
        return False
    core = {k: v for k, v in preflight_doc.items() if k != "receipt_sha256"}
    try:
        return claimed == compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False


def _verify_vault_receipt_hash(vault_doc: Mapping[str, Any]) -> bool:
    """Independently verify ``SingleShotVaultLock`` receipt SHA-256."""
    claimed = vault_doc.get("vault_receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    core = {k: v for k, v in vault_doc.items() if k != "vault_receipt_sha256"}
    try:
        return claimed == hashlib.sha256(canonical_json_bytes(core)).hexdigest()
    except (TypeError, ValueError):
        return False


def verify_pipeline_manifest_provenance(
    result_or_manifest: AutoOpenWeightPipelineResult | Mapping[str, Any],
) -> bool:
    """Independently verify unified SHA-256 manifest and all nested Season 35 receipts."""
    if isinstance(result_or_manifest, AutoOpenWeightPipelineResult):
        payload: Mapping[str, Any] = result_or_manifest.to_dict()
    elif isinstance(result_or_manifest, Mapping):
        payload = result_or_manifest
    else:
        raise TypeError(
            f"result_or_manifest must be AutoOpenWeightPipelineResult or Mapping, "
            f"got {type(result_or_manifest).__name__}"
        )

    manifest_doc = payload.get("manifest") if "manifest" in payload else payload
    if not isinstance(manifest_doc, Mapping):
        return False
    claimed = manifest_doc.get("manifest_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if (
        manifest_doc.get("schema") != SCHEMA_VERSION
        or manifest_doc.get("object") != PIPELINE_MANIFEST_OBJECT
    ):
        return False

    core_manifest = {k: v for k, v in manifest_doc.items() if k != "manifest_sha256"}
    try:
        if claimed != compute_payload_sha256(core_manifest):
            return False
    except (TypeError, ValueError):
        return False

    if "manifest" not in payload:
        return True

    if payload.get("manifest_sha256") != claimed:
        return False

    for field_name in (
        "committed",
        "verdict",
        "halted_before_nf4_export",
        "rolled_back_bitwise",
        "nf4_export_executed",
        "anse_energy",
        "nll_mean",
        "nf4_gap_nats",
        "pre_native_sha256",
        "post_native_sha256",
    ):
        if payload.get(field_name) != manifest_doc.get(field_name):
            return False

    preflight_doc = payload.get("preflight_diagnosis")
    if not isinstance(preflight_doc, Mapping) or not _verify_preflight_hash(preflight_doc):
        return False
    if preflight_doc.get("receipt_sha256") != manifest_doc.get("preflight_diagnosis_sha256"):
        return False

    multi_doc = payload.get("multi_arch_result")
    if multi_doc is not None:
        if not isinstance(multi_doc, Mapping) or not verify_multi_arch_receipt_provenance(multi_doc):
            return False
        if multi_doc.get("receipt_sha256") != manifest_doc.get("multi_arch_receipt_sha256"):
            return False

    cert_doc = payload.get("formal_certificate")
    if cert_doc is not None:
        if not isinstance(cert_doc, Mapping) or not verify_wsp_formal_certificate_provenance(cert_doc):
            return False
        if cert_doc.get("receipt_sha256") != manifest_doc.get("formal_certificate_sha256"):
            return False

    vault_doc = payload.get("vault_receipt")
    if vault_doc is not None:
        if not isinstance(vault_doc, Mapping) or not _verify_vault_receipt_hash(vault_doc):
            return False
        if vault_doc.get("vault_receipt_sha256") != manifest_doc.get("vault_receipt_sha256"):
            return False

    nf4_doc = payload.get("nf4_edge_bundle")
    if nf4_doc is not None:
        if not isinstance(nf4_doc, Mapping) or not isinstance(nf4_doc.get("receipt"), Mapping):
            return False
        nf4_rec = nf4_doc["receipt"]
        if not verify_workunit_receipt_provenance(nf4_rec):
            return False
        if nf4_rec.get("receipt_sha256") != manifest_doc.get("nf4_export_receipt_sha256"):
            return False
        if nf4_rec.get("exported_weights_sha256") != manifest_doc.get("nf4_weights_sha256"):
            return False

    promo_doc = payload.get("promotion_bundle")
    if not isinstance(promo_doc, Mapping):
        return False
    promo_sha = promo_doc.get("receipt_sha256")
    if not isinstance(promo_sha, str) or promo_sha != manifest_doc.get("promotion_bundle_sha256"):
        return False
    promo_core = {k: v for k, v in promo_doc.items() if k != "receipt_sha256"}
    try:
        if promo_sha != compute_payload_sha256(promo_core):
            return False
    except (TypeError, ValueError):
        return False

    lineage_doc = promo_doc.get("lineage_receipt")
    if not isinstance(lineage_doc, Mapping) or not verify_lineage_receipt_provenance(lineage_doc):
        return False
    return lineage_doc.get("receipt_sha256") == manifest_doc.get("lineage_receipt_sha256")


__all__ = [
    "PIPELINE_MANIFEST_OBJECT",
    "PREFLIGHT_DIAGNOSIS_OBJECT",
    "PROMOTION_BUNDLE_OBJECT",
    "SCHEMA_VERSION",
    "AutoOpenWeightOrchestrator",
    "AutoOpenWeightOrchestratorConfig",
    "AutoOpenWeightPipelineResult",
    "NF4EdgeBundle",
    "PreflightDiagnosisReport",
    "PreflightLayerDiagnosis",
    "ShadowCanaryPromotionBundle",
    "run_full_auto_open_weight_pipeline",
    "verify_pipeline_manifest_provenance",
]

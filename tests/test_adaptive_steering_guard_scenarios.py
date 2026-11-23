# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-06: runtime thermodynamic dense-fallback guard."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.adaptive_steering_guard import (
    DEFAULT_ANSE_MAXIMUM_PAIN,
    DEFAULT_MAX_DELTA_BASE_RATIO,
    DEFAULT_MAX_DOMAIN_DRIFT,
    AdaptiveSteeredMoELayer,
    FallbackControllerConfig,
    HotSwapGateConfig,
    ProseConfidenceFallbackController,
    ThermodynamicHotSwapGate,
    collect_adaptive_steering_telemetry,
    compute_buckets_sha256,
    compute_thermodynamic_energy,
    extract_dense_fallback_buckets,
    extract_guarded_moe_buckets,
    install_adaptive_steering_guard,
    load_guarded_moe_buckets,
    remove_adaptive_steering_guard,
    set_adaptive_steering_enabled,
    unwrap_moe_mlp,
)
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdRuntimeMLP,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdFineGrainedMoEMLP,
    CrowdLoRAMoEMLP,
    upcycle_crowd_v1_to_sparse_moe,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "adaptive_steering_guard.py"
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."



class _FallbackZeroStubAudit:
    def check_file(self, _path: str) -> list[str]:
        return []


try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover
    ZeroStubAudit = _FallbackZeroStubAudit  # type: ignore[assignment,misc]

def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_subspace_batch(
    n_samples: int,
    hidden_size: int,
    carrier_dim: int,
    *,
    carrier_strength: float = 2.5,
    noise_slice: slice,
    noise_scale: float = 0.20,
    seed: int,
) -> torch.Tensor:
    """Create deterministic activations localized to a subspace with a carrier signal."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.zeros((n_samples, hidden_size), dtype=torch.float32)
    width = noise_slice.stop - noise_slice.start
    x[:, noise_slice] = noise_scale * torch.randn((n_samples, width), generator=gen, dtype=torch.float32)
    x[:, carrier_dim] = carrier_strength
    return x


def _build_upcycled_crowd_model(
    *,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    num_layers: int = 2,
    num_experts: int = 4,
    top_k: int = 2,
    routing_mode: str = "sigma_topk",
    split_mode: str = "lora_domain_experts",
    resident_expert_indices: tuple[int, ...] | None = None,
    seed: int = 20261123,
) -> CrowdV1WithEngram:
    """Build a small deterministic CPU CrowdV1WithEngram upcycled to Sparse MoE."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cfg = CrowdV1RuntimeConfig(
        vocab_size=32,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        intermediate_size=intermediate_size,
        max_position_embeddings=32,
        engram_buckets=0,
        tie_word_embeddings=True,
    )
    model = CrowdV1WithEngram(cfg)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=gen, dtype=torch.float32) * 0.12)
        if cfg.tie_word_embeddings:
            model.head.weight = model.wte.weight
    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode=split_mode,
        num_experts=num_experts,
        top_k=top_k,
        lora_rank=4,
        lora_alpha=8.0,
        routing_mode=routing_mode,
        resident_expert_indices=resident_expert_indices,
        seed=seed,
    )
    return model


def _build_lora_moe_layer(
    *,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    num_experts: int = 4,
    top_k: int = 2,
    routing_mode: str = "sigma_topk",
    resident_expert_indices: tuple[int, ...] | None = None,
    seed: int = 1001,
) -> CrowdLoRAMoEMLP:
    """Build a standalone deterministic CrowdLoRAMoEMLP layer."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cfg = CrowdV1RuntimeConfig(hidden_size=hidden_size, intermediate_size=intermediate_size)
    dense = CrowdRuntimeMLP(cfg)
    with torch.no_grad():
        for param in dense.parameters():
            param.copy_(torch.randn(param.shape, generator=gen, dtype=torch.float32) * 0.18)
    return CrowdLoRAMoEMLP(
        dense,
        num_experts=num_experts,
        top_k=top_k,
        rank=4,
        alpha=8.0,
        routing_mode=routing_mode,
        resident_expert_indices=resident_expert_indices,
        seed=seed,
    )


def _forward_hidden_sequence(model: CrowdV1WithEngram, x_seq: torch.Tensor) -> torch.Tensor:
    """Propagate sequence hidden states through all transformer blocks and final LayerNorm."""
    cos = model.cos_cached.to(x_seq.device)
    sin = model.sin_cached.to(x_seq.device)
    h = x_seq
    for blk in model.blocks:
        h, _ = blk(h, cos, sin, past_kv=None, use_cache=False)
    return model.ln(h)


# ---------------------------------------------------------------------------
# S1: ThermodynamicHotSwapGate Acceptance, Rejection & Atomic Rollback
# ---------------------------------------------------------------------------


def test_s1_thermodynamic_hotswap_acceptance_rejection_and_atomic_rollback() -> None:
    """Verify Delta E < 0 and <= +1.0% drift gating, non-finite rejection, and atomic parent/dense rollback."""
    model = _build_upcycled_crowd_model(num_layers=2, num_experts=4, seed=4101)
    dense_buckets = extract_dense_fallback_buckets(model, num_router_fragments=2)
    assert dense_buckets is not None
    load_guarded_moe_buckets(model, dense_buckets, num_router_fragments=2)

    base_domain_nll = {
        "E0_general_fineweb": 3.000,
        "E1_python_systems_code": 2.800,
        "E2_math_reasoning": 2.900,
        "E5_exoplanet_transit": 2.700,
    }
    gate = ThermodynamicHotSwapGate(
        HotSwapGateConfig(max_domain_drift=DEFAULT_MAX_DOMAIN_DRIFT),
        dense_buckets=dense_buckets,
        baseline_domain_nll=base_domain_nll,
    )

    # Craft 4 distinct bucket sets identified by expert_1 scaling offsets
    def _make_buckets(tag: float) -> dict[str, torch.Tensor]:
        out = {k: v.clone() for k, v in dense_buckets.items()}
        out["expert_1"] = out["expert_1"] + tag
        out["router_frag_0"] = out["router_frag_0"] + 0.1 * tag
        return out

    buckets_c1 = _make_buckets(0.05)  # valid improvement: Delta E < 0, E0 drift +0.20%
    buckets_drift = _make_buckets(0.22)  # E0 drift +1.25% > +1.0% -> must reject & rollback
    buckets_regress = _make_buckets(0.09)  # drift OK (+0.30%), but E_child > E_parent -> reject & rollback
    buckets_c2 = _make_buckets(0.11)  # further valid drop with soft margin penalty check (+0.90% E0)

    def _eval_by_tag(m: nn.Module) -> dict[str, Any]:
        snap = extract_guarded_moe_buckets(m, num_router_fragments=2)
        tag = float((snap["expert_1"] - dense_buckets["expert_1"]).mean().item())
        if abs(tag) < 1e-6:
            return {"domain_nll": dict(base_domain_nll)}
        if abs(tag - 0.05) < 1e-4:
            return {
                "domain_nll": {
                    "E0_general_fineweb": 3.006,  # +0.20%
                    "E1_python_systems_code": 2.400,
                    "E2_math_reasoning": 2.650,
                    "E5_exoplanet_transit": 2.550,
                }
            }
        if abs(tag - 0.22) < 1e-4:
            return {
                "domain_nll": {
                    "E0_general_fineweb": 3.0375,  # +1.25% -> violates +1.0% guard!
                    "E1_python_systems_code": 2.000,
                    "E2_math_reasoning": 2.200,
                    "E5_exoplanet_transit": 2.100,
                }
            }
        if abs(tag - 0.09) < 1e-4:
            return {
                "domain_nll": {
                    "E0_general_fineweb": 3.009,  # +0.30% (guard OK)
                    "E1_python_systems_code": 2.550,  # worse than c1 (2.400) -> Delta E > 0!
                    "E2_math_reasoning": 2.720,
                    "E5_exoplanet_transit": 2.620,
                }
            }
        return {
            "domain_nll": {
                "E0_general_fineweb": 3.027,  # +0.90% (guard OK, > +0.85% triggers soft penalty +0.05)
                "E1_python_systems_code": 2.100,
                "E2_math_reasoning": 2.300,
                "E5_exoplanet_transit": 2.200,
            }
        }

    # 1. Accept child 1 (from dense parent E=0.0)
    r1 = gate.try_hot_swap(model, buckets_c1, eval_fn=_eval_by_tag)
    assert r1.accepted is True
    assert r1.rolled_back is False
    assert r1.rollback_target is None
    assert r1.delta_energy < 0.0
    assert r1.domain_guard_ok is True
    assert r1.worst_domain_drift_pct == pytest.approx(0.2, abs=1e-4)
    assert gate.parent_energy == pytest.approx(r1.energy_child)
    assert gate.accepted_count == 1
    c1_sha = compute_buckets_sha256(buckets_c1)
    assert compute_buckets_sha256(extract_guarded_moe_buckets(model, num_router_fragments=2)) == c1_sha

    # 2. Reject child violating <= +1.0% worst-domain drift and verify atomic rollback to c1 parent
    r2 = gate.try_hot_swap(model, buckets_drift, eval_fn=_eval_by_tag)
    assert r2.accepted is False
    assert r2.rolled_back is True
    assert r2.rollback_target == "parent"
    assert r2.domain_guard_ok is False
    assert r2.energy_child >= DEFAULT_ANSE_MAXIMUM_PAIN - 5.0
    assert r2.worst_domain == "E0_general_fineweb"
    assert r2.worst_domain_drift_pct == pytest.approx(1.25, abs=1e-4)
    assert r2.active_state_sha256 == c1_sha
    assert compute_buckets_sha256(extract_guarded_moe_buckets(model, num_router_fragments=2)) == c1_sha
    assert gate.parent_energy == pytest.approx(r1.energy_child)

    # 3. Reject child with Delta E >= 0 even though worst-domain drift <= +1.0%
    r3 = gate.try_hot_swap(model, buckets_regress, eval_fn=_eval_by_tag)
    assert r3.accepted is False
    assert r3.rolled_back is True
    assert r3.rollback_target == "parent"
    assert r3.domain_guard_ok is True
    assert r3.thermodynamic_ok is False
    assert r3.delta_energy > 0.0
    assert compute_buckets_sha256(extract_guarded_moe_buckets(model, num_router_fragments=2)) == c1_sha

    # 4. Reject non-finite (NaN), shape mismatch, missing key, and runtime exception during eval_fn
    buckets_nan = _make_buckets(0.05)
    buckets_nan["expert_1"][0] = float("nan")
    r_nan = gate.try_hot_swap(model, buckets_nan, eval_fn=_eval_by_tag)
    assert r_nan.accepted is False and r_nan.rolled_back is True
    assert "NON_FINITE_CANDIDATE_WEIGHTS" in r_nan.reasons[0]
    assert compute_buckets_sha256(extract_guarded_moe_buckets(model, num_router_fragments=2)) == c1_sha

    buckets_shape = _make_buckets(0.05)
    buckets_shape["expert_1"] = buckets_shape["expert_1"][:-1]
    r_shape = gate.try_hot_swap(model, buckets_shape, eval_fn=_eval_by_tag)
    assert r_shape.accepted is False and "BUCKET_SHAPE_MISMATCH" in r_shape.reasons[0]

    buckets_key = _make_buckets(0.05)
    del buckets_key["expert_2"]
    r_key = gate.try_hot_swap(model, buckets_key, eval_fn=_eval_by_tag)
    assert r_key.accepted is False and "BUCKET_KEY_MISMATCH" in r_key.reasons[0]

    buckets_bad_type: dict[str, Any] = dict(_make_buckets(0.05))
    buckets_bad_type["expert_1"] = [1.0, 2.0]
    r_type = gate.try_hot_swap(model, buckets_bad_type, eval_fn=_eval_by_tag)  # type: ignore[arg-type]
    assert r_type.accepted is False and "INVALID_BUCKET_TYPE" in r_type.reasons[0]

    def _boom_eval(_m: nn.Module) -> dict[str, Any]:
        raise RuntimeError("injected evaluator fault")

    r_exc = gate.try_hot_swap(model, buckets_c2, eval_fn=_boom_eval)
    assert r_exc.accepted is False and r_exc.rolled_back is True
    assert "CANDIDATE_EVAL_FAILURE" in r_exc.reasons[0]
    assert compute_buckets_sha256(extract_guarded_moe_buckets(model, num_router_fragments=2)) == c1_sha

    # 5. Accept c2 (valid Delta E < 0 even after +0.05 soft margin penalty)
    r4 = gate.try_hot_swap(model, buckets_c2, eval_fn=_eval_by_tag)
    assert r4.accepted is True
    assert gate.accepted_count == 2
    assert gate.active_buckets is not None
    assert gate.dense_buckets is not None
    assert gate.baseline_domain_nll == base_domain_nll

    # 6. Violation with rollback_to_dense=True must revert atomically to dense base (B == 0)
    r5 = gate.try_hot_swap(model, buckets_drift, eval_fn=_eval_by_tag, rollback_to_dense=True)
    assert r5.accepted is False
    assert r5.rolled_back is True
    assert r5.rollback_target == "dense"
    dense_sha = compute_buckets_sha256(dense_buckets)
    assert r5.active_state_sha256 == dense_sha
    assert compute_buckets_sha256(extract_guarded_moe_buckets(model, num_router_fragments=2)) == dense_sha

    # Also test inferred dense fallback without pre-registered dense_buckets + force_dense_fallback
    gate_inferred = ThermodynamicHotSwapGate(
        HotSwapGateConfig(rollback_to_dense_on_violation=True, verify_rollback_hash=False),
        baseline_domain_nll=base_domain_nll,
    )
    assert gate_inferred.active_buckets is None
    assert gate_inferred.dense_buckets is None
    load_guarded_moe_buckets(model, buckets_c1, num_router_fragments=2)
    r_inf = gate_inferred.try_hot_swap(model, buckets_drift, eval_fn=_eval_by_tag)
    assert r_inf.accepted is False and r_inf.rollback_target == "dense"
    r_force = gate_inferred.force_dense_fallback(model)
    assert r_force.rollback_target == "dense"
    assert r_force.to_dict()["rolled_back"] is True

    # Test tuple & domain_drifts evaluator formats and non-strict energy drop
    gate_nonstrict = ThermodynamicHotSwapGate(
        HotSwapGateConfig(require_strict_energy_drop=False),
        parent_energy=-0.10,
        baseline_domain_nll={"general": 2.5, "code": 2.0},
    )
    r_tuple = gate_nonstrict.try_hot_swap(
        model,
        dense_buckets,
        eval_fn=lambda _m: (-0.10, {"general": 2.50, "code": 1.90}),
    )
    assert r_tuple.accepted is True
    r_drifts = gate_nonstrict.evaluate_candidate_metrics(
        -0.10,
        -0.15,
        domain_drifts={"general": 0.004, "code": -0.05},
    )
    assert r_drifts.accepted is True
    r_child_nll = gate_nonstrict.evaluate_candidate_metrics(
        0.0,
        -0.20,
        domain_nll_child={"general": 2.50, "code": 1.80},
        domain_nll_baseline={"general": 2.50, "code": 2.00},
    )
    assert r_child_nll.accepted is True

    # Initialize baseline_domain_nll lazily on try_hot_swap
    gate_lazy_base = ThermodynamicHotSwapGate(parent_energy=0.0)
    r_lazy = gate_lazy_base.try_hot_swap(
        model,
        dense_buckets,
        eval_fn=lambda _m: {"domain_nll": {"general": 2.50, "code": 1.85}},
        baseline_domain_nll={"general": 2.50, "code": 2.00},
    )
    assert r_lazy.accepted is True

    # Weighted energy check with "general" fallback prefix
    thermo_w = compute_thermodynamic_energy(
        {"general": 2.0, "code": 2.0},
        {"general": 2.01, "code": 1.60},
        domain_weights={"general": 0.25, "code": 0.75},
    )
    assert thermo_w["guard_ok"] is True
    assert thermo_w["general_ratio"] == pytest.approx(1.005)
    assert thermo_w["mean_diff_vs_base"] == pytest.approx(-0.2975)

    # Single-module state_dict hot-swap + verify_rollback_hash negative control
    single_moe = _build_lora_moe_layer(seed=4199)
    single_dense = extract_dense_fallback_buckets(single_moe)
    assert single_dense is not None
    single_gate = ThermodynamicHotSwapGate(
        HotSwapGateConfig(verify_rollback_hash=True),
        parent_energy=0.0,
        dense_buckets=single_dense,
        baseline_domain_nll={"E0_prose": 3.0, "E1_code": 2.5},
    )
    snap_single = single_gate.snapshot_model_state(single_moe)
    bad_single = {k: v + 1.0 for k, v in snap_single.items()}
    r_single = single_gate.try_hot_swap(
        single_moe,
        bad_single,
        eval_fn=lambda _m: {"energy": 100.0, "domain_drifts": {"E0_prose": 0.02, "E1_code": 0.0}},
    )
    assert r_single.accepted is False and r_single.rolled_back is True
    assert compute_buckets_sha256(single_gate.snapshot_model_state(single_moe)) == compute_buckets_sha256(snap_single)


# ---------------------------------------------------------------------------
# S2: Uncentered Linear Probe Calibration & Out-of-Distribution Spike Clamping
# ---------------------------------------------------------------------------


def test_s2_ood_prose_confidence_and_norm_ratio_spike_clamping() -> None:
    """Verify calibrated prose-confidence damping, norm-ratio spike clamping, OOF shutoff, and NaN safety."""
    hidden_size = 32
    prose_f0 = _make_subspace_batch(64, hidden_size, 0, carrier_strength=2.5, noise_slice=slice(0, 8), seed=2101)
    prose_f1 = _make_subspace_batch(64, hidden_size, 0, carrier_strength=2.5, noise_slice=slice(0, 8), seed=2102)
    spec_f0 = _make_subspace_batch(64, hidden_size, 12, carrier_strength=2.5, noise_slice=slice(8, 32), seed=2103)
    spec_f1 = _make_subspace_batch(64, hidden_size, 12, carrier_strength=2.5, noise_slice=slice(8, 32), seed=2104)

    ctrl = ProseConfidenceFallbackController.from_probe_folds(
        prose_f0,
        prose_f1,
        spec_f0,
        spec_f1,
        config=FallbackControllerConfig(
            prose_confidence_threshold=0.55,
            prose_transition_width=0.30,
            max_delta_base_ratio=0.50,
            spike_decay=1.0,
            min_delta_scale=0.0,
        ),
    )
    assert ctrl.config.rel_weight > 0.95
    assert ctrl.probe_direction is not None

    # 1. Calibrated prose vs specialist separation
    _, conf_prose = ctrl.compute_prose_confidence(prose_f0)
    _, conf_spec = ctrl.compute_prose_confidence(spec_f0)
    assert float(conf_prose.mean().item()) > 0.90
    assert float(conf_spec.mean().item()) < 0.10

    # 2. Smooth monotonic decay across confidence ramp when varying prose carrier magnitude
    assert ctrl.probe_direction is not None
    u_unit = ctrl.probe_direction / ctrl.probe_direction.norm()
    sweep_vals = torch.linspace(-0.5, 2.0, 41, dtype=torch.float32)
    x_sweep = sweep_vals.unsqueeze(-1) * u_unit.unsqueeze(0) / ctrl.probe_direction.norm()
    base_sweep = torch.ones((41, hidden_size), dtype=torch.float32)
    delta_sweep = 0.05 * torch.ones((41, hidden_size), dtype=torch.float32)
    gammas, _, _ = ctrl.compute_damping_factors(x_sweep, base_sweep, delta_sweep)
    assert float(gammas[0].item()) == pytest.approx(1.0)
    assert float(gammas[-1].item()) == pytest.approx(0.0, abs=1e-5)
    assert bool((gammas[1:] <= gammas[:-1] + 1e-6).all().item())

    # 3. Abnormal OOD delta/base norm-ratio spike clamping on specialist tokens (where prose conf is ~0)
    n_spec = 16
    x_spec = spec_f0[:n_spec]
    base_spec = torch.ones((n_spec, hidden_size), dtype=torch.float32)
    # Normal specialist delta (ratio 0.25 < 0.50) vs escalating spikes (ratio 1.0, 5.0, 50.0)
    scales = torch.tensor([0.25] * 8 + [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.float32)
    delta_spike = torch.ones((n_spec, hidden_size), dtype=torch.float32) * scales.unsqueeze(-1)

    guarded_spike, tel_spike = ctrl.apply_damping(x_spec, base_spec, base_spec + delta_spike)
    eff_ratios = (guarded_spike - base_spec).norm(dim=-1) / base_spec.norm(dim=-1)
    assert torch.allclose(guarded_spike[:8], base_spec[:8] + delta_spike[:8], atol=1e-6)
    assert bool((eff_ratios[8:] <= DEFAULT_MAX_DELTA_BASE_RATIO + 1e-5).all().item())
    assert float(eff_ratios[-1].item()) < 0.05
    assert tel_spike.ratio_trigger_fraction == pytest.approx(0.5)
    assert tel_spike.post_damping_max_ratio <= DEFAULT_MAX_DELTA_BASE_RATIO + 1e-5
    assert tel_spike.to_dict()["num_tokens"] == n_spec

    # 4. Non-finite (NaN / Inf) delta tokens must sanitize cleanly back to base_out
    delta_corrupt = delta_spike[:4].clone()
    delta_corrupt[1, 3] = float("nan")
    delta_corrupt[3, 7] = float("inf")
    guarded_clean, tel_clean = ctrl.apply_damping_to_delta(x_spec[:4], base_spec[:4], delta_corrupt)
    assert bool(torch.isfinite(guarded_clean).all().item())
    assert torch.equal(guarded_clean[1], base_spec[1])
    assert torch.equal(guarded_clean[3], base_spec[3])
    assert tel_clean.ratio_trigger_fraction == pytest.approx(0.5)

    # 5. Pure Gaussian noise OOF shutoff: prose trigger shuts off cleanly while norm-ratio guard stays live
    gen_n = torch.Generator(device="cpu").manual_seed(2199)
    n0 = torch.randn((48, hidden_size), generator=gen_n)
    n1 = torch.randn((48, hidden_size), generator=gen_n)
    ctrl_noise = ProseConfidenceFallbackController.from_probe_folds(n0, n1, n0.clone(), n1.clone())
    assert ctrl_noise.config.rel_weight == 0.0
    _, noise_conf = ctrl_noise.compute_prose_confidence(prose_f0)
    assert torch.equal(noise_conf, torch.zeros(prose_f0.shape[0]))
    guarded_n, tel_n = ctrl_noise.apply_damping(prose_f0[:4], base_spec[:4], base_spec[:4] + 10.0 * base_spec[:4])
    assert tel_n.prose_trigger_fraction == 0.0
    assert tel_n.ratio_trigger_fraction == 1.0
    assert bool(torch.isfinite(guarded_n).all().item())

    # 6. update_probe, zero-probe, and 0-token batch handling
    ctrl_empty = ProseConfidenceFallbackController()
    z_proj, z_conf = ctrl_empty.compute_prose_confidence(prose_f0[:4])
    assert torch.equal(z_proj, torch.zeros(4)) and torch.equal(z_conf, torch.zeros(4))
    ctrl_empty.update_probe(torch.zeros(hidden_size))
    z_proj2, _ = ctrl_empty.compute_prose_confidence(prose_f0[:4])
    assert torch.equal(z_proj2, torch.zeros(4))
    ctrl_empty.update_probe(ctrl.probe_direction, probe_midpoint=0.5, probe_logit_scale=8.0, rel_weight=1.0)
    g_zero, _, tel_zero = ctrl_empty.compute_damping_factors(
        torch.empty((0, hidden_size)),
        torch.empty((0, hidden_size)),
        torch.empty((0, hidden_size)),
    )
    assert g_zero.shape == (0,) and tel_zero.num_tokens == 0


# ---------------------------------------------------------------------------
# S3: Mixed-Domain Token Batches Across Guarded CrowdV1WithEngram & MoE Layers
# ---------------------------------------------------------------------------


def test_s3_mixed_domain_batches_and_guarded_model_integration() -> None:
    """Verify per-token selectivity in mixed-domain [B, T, H] batches and bucket round-trips."""
    hidden_size = 32
    model = _build_upcycled_crowd_model(
        hidden_size=hidden_size,
        intermediate_size=64,
        num_layers=2,
        num_experts=4,
        top_k=2,
        routing_mode="sigma_topk",
        resident_expert_indices=(0, 1, 2),
        seed=3101,
    )

    prose_f0 = _make_subspace_batch(48, hidden_size, 0, carrier_strength=2.5, noise_slice=slice(0, 8), seed=3110)
    prose_f1 = _make_subspace_batch(48, hidden_size, 0, carrier_strength=2.5, noise_slice=slice(0, 8), seed=3111)
    spec_f0 = _make_subspace_batch(48, hidden_size, 14, carrier_strength=2.5, noise_slice=slice(8, 32), seed=3112)
    spec_f1 = _make_subspace_batch(48, hidden_size, 14, carrier_strength=2.5, noise_slice=slice(8, 32), seed=3113)

    gen = torch.Generator(device="cpu").manual_seed(3120)
    # Plant helpful specialist expert 1 on dim 14 and harmful prose leakage on expert 2 (dim 0)
    with torch.no_grad():
        for blk in model.blocks:
            moe = blk.mlp
            assert isinstance(moe, CrowdLoRAMoEMLP)
            moe.router.weight.zero_()
            moe.router.weight[1, 14] = 2.0
            moe.router.weight[2, 0] = 2.0
            moe.router.weight[0, 0] = 1.0
            for e_idx in (1, 2):
                exp = moe.experts[str(e_idx)]
                for attr in ("w1_B", "w2_B", "w3_B"):
                    param = getattr(exp, attr)
                    param.copy_(torch.randn(param.shape, generator=gen) * 0.04)

    ctrl = ProseConfidenceFallbackController.from_probe_folds(
        prose_f0,
        prose_f1,
        spec_f0,
        spec_f1,
        config=FallbackControllerConfig(
            prose_confidence_threshold=0.55,
            prose_transition_width=0.25,
            max_delta_base_ratio=0.50,
            min_delta_scale=0.0,
        ),
    )

    # Mixed-domain sequence batch [B=2, T=8, H=32]: tokens 0..3 prose, tokens 4..7 specialist
    x_mixed = torch.cat([prose_f0[:8].view(2, 4, hidden_size), spec_f0[:8].view(2, 4, hidden_size)], dim=1)

    # Capture unguarded MoE output & pure dense baseline output at layer 0 before wrapping
    raw_layer0 = model.blocks[0].mlp
    assert isinstance(raw_layer0, CrowdLoRAMoEMLP)
    with torch.no_grad():
        raw_moe_out = raw_layer0(x_mixed)
        dense_out = raw_layer0.base_mlp(x_mixed)

    wrapped = install_adaptive_steering_guard(model, {0: ctrl, 1: ctrl})
    assert set(wrapped.keys()) == {0, 1}
    assert isinstance(model.blocks[0].mlp, AdaptiveSteeredMoELayer)

    with torch.no_grad():
        guarded_l0 = model.blocks[0].mlp(x_mixed)

    # Prose tokens [:, :4] fall back cleanly to dense_out; specialist tokens [:, 4:] match raw MoE
    prose_err_vs_dense = float((guarded_l0[:, :4] - dense_out[:, :4]).abs().max().item())
    prose_err_raw = float((raw_moe_out[:, :4] - dense_out[:, :4]).abs().max().item())
    assert prose_err_raw > 0.005
    assert prose_err_vs_dense < 0.02 * prose_err_raw

    spec_diff = float((guarded_l0[:, 4:] - raw_moe_out[:, 4:]).abs().max().item())
    assert spec_diff < 1e-5
    assert wrapped[0].last_telemetry is not None
    assert wrapped[0].last_telemetry.damped_token_fraction == pytest.approx(0.5)

    # Fit per-layer controllers on real post-attention ln2 states so full multi-layer sequence damping holds
    cos = model.cos_cached
    sin = model.sin_cached
    with torch.no_grad():
        p_in_seq = torch.cat([prose_f0[:8], prose_f1[:8]], dim=0).view(4, 4, hidden_size)
        s_in_seq = torch.cat([spec_f0[:8], spec_f1[:8]], dim=0).view(4, 4, hidden_size)
        # Keep attn near-identity on subspace signals by using layer 0 & 1 ln2 outputs
        layer_ctrls: dict[int, ProseConfidenceFallbackController] = {}
        hp_cur, hs_cur = p_in_seq, s_in_seq
        for l_idx, blk in enumerate(model.blocks):
            hp_mid = hp_cur + blk.attn(blk.ln1(hp_cur), cos, sin, past_kv=None, use_cache=False)[0]
            hs_mid = hs_cur + blk.attn(blk.ln1(hs_cur), cos, sin, past_kv=None, use_cache=False)[0]
            mp_in = blk.ln2(hp_mid).reshape(-1, hidden_size)
            ms_in = blk.ln2(hs_mid).reshape(-1, hidden_size)
            half_m = mp_in.shape[0] // 2
            layer_ctrls[l_idx] = ProseConfidenceFallbackController.from_probe_folds(
                mp_in[:half_m],
                mp_in[half_m:],
                ms_in[:half_m],
                ms_in[half_m:],
                config=ctrl.config,
            )
            hp_cur = hp_mid + unwrap_moe_mlp(blk.mlp)(blk.ln2(hp_mid))
            hs_cur = hs_mid + unwrap_moe_mlp(blk.mlp)(blk.ln2(hs_mid))

    # Re-install with per-layer controllers and forward sequence without cross-token mixing (T=1 per stream or ln2 input)
    remove_adaptive_steering_guard(model)
    install_adaptive_steering_guard(model, layer_ctrls)
    with torch.no_grad():
        # Evaluate non-mixing causal step (2 prose batches + 2 specialist batches in one batch [B=4, T=4, H])
        x_seq_batch = torch.cat([p_in_seq[:2], s_in_seq[:2]], dim=0)
        full_guarded_out = _forward_hidden_sequence(model, x_seq_batch)
    assert bool(torch.isfinite(full_guarded_out).all().item())

    tel_summary = collect_adaptive_steering_telemetry(model)
    assert tel_summary["guarded_layers_with_telemetry"] == 2
    assert 0.2 <= tel_summary["mean_damped_token_fraction"] <= 0.8

    # Verify bucket extraction & hot-swap compatibility while layers are guarded
    buckets_before = extract_guarded_moe_buckets(model, num_router_fragments=2)
    gate = ThermodynamicHotSwapGate(baseline_domain_nll={"E0_prose": 3.0, "E1_code": 2.5})
    buckets_next = {k: (v + 0.01 if k == "expert_1" else v.clone()) for k, v in buckets_before.items()}
    eval_calls = [0]

    def _step_eval(_m: nn.Module) -> dict[str, Any]:
        eval_calls[0] += 1
        if eval_calls[0] == 1:
            return {"domain_nll": {"E0_prose": 3.0, "E1_code": 2.3}}
        return {"domain_nll": {"E0_prose": 3.005, "E1_code": 2.0}}

    receipt = gate.try_hot_swap(model, buckets_next, eval_fn=_step_eval)
    assert receipt.accepted is True
    assert isinstance(model.blocks[0].mlp, AdaptiveSteeredMoELayer)

    # Also test softmax_topk mode on CrowdLoRAMoEMLP and CrowdFineGrainedMoEMLP (both auto-sum & explicit dense_mlp)
    moe_soft = _build_lora_moe_layer(routing_mode="softmax_topk", resident_expert_indices=(0, 1, 2), seed=3150)
    with torch.no_grad():
        moe_soft.experts["1"].w3_B.fill_(0.15)
    wrap_soft = AdaptiveSteeredMoELayer(moe_soft, ctrl)
    y_soft = wrap_soft(x_mixed)
    assert y_soft.shape == x_mixed.shape

    model_fg = _build_upcycled_crowd_model(
        num_layers=1, num_experts=4, routing_mode="sigma_topk", split_mode="fine_grained_width_split", seed=3160
    )
    fg_buckets = extract_guarded_moe_buckets(model_fg, num_router_fragments=1, only_trainable=True)
    load_guarded_moe_buckets(model_fg, fg_buckets, num_router_fragments=1)
    assert extract_dense_fallback_buckets(model_fg) is None
    install_adaptive_steering_guard(model_fg, ctrl, layer_indices=(0,))
    assert isinstance(unwrap_moe_mlp(model_fg.blocks[0].mlp), CrowdFineGrainedMoEMLP)
    y_fg = model_fg.blocks[0].mlp(x_mixed)
    assert y_fg.shape == x_mixed.shape
    assert remove_adaptive_steering_guard(model_fg) == 1

    # Explicit external dense_mlp wrapper on a custom MoE module and internal .base_mlp wrapper
    dense_ext = CrowdRuntimeMLP(CrowdV1RuntimeConfig(hidden_size=hidden_size, intermediate_size=64))
    wrap_ext = AdaptiveSteeredMoELayer(moe_soft, ctrl, dense_mlp=dense_ext)
    assert wrap_ext(x_mixed).shape == x_mixed.shape

    class _CustomMoEWithBase(nn.Module):
        def __init__(self, base: nn.Module) -> None:
            super().__init__()
            self.base_mlp = base

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base_mlp(x) * 1.1

    wrap_custom_base = AdaptiveSteeredMoELayer(_CustomMoEWithBase(dense_ext), ctrl)
    assert wrap_custom_base(x_mixed).shape == x_mixed.shape


# ---------------------------------------------------------------------------
# S4: Zero-Overhead Bypass When Disabled
# ---------------------------------------------------------------------------


def test_s4_zero_overhead_bypass_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify bitwise identity and zero extra dense/probe calls when guard is disabled."""
    model = _build_upcycled_crowd_model(num_layers=2, num_experts=4, seed=4401)
    prose_f0 = _make_subspace_batch(32, 32, 0, carrier_strength=2.5, noise_slice=slice(0, 8), seed=4402)
    prose_f1 = _make_subspace_batch(32, 32, 0, carrier_strength=2.5, noise_slice=slice(0, 8), seed=4403)
    spec_f0 = _make_subspace_batch(32, 32, 12, carrier_strength=2.5, noise_slice=slice(8, 32), seed=4404)
    spec_f1 = _make_subspace_batch(32, 32, 12, carrier_strength=2.5, noise_slice=slice(8, 32), seed=4405)
    ctrl = ProseConfidenceFallbackController.from_probe_folds(prose_f0, prose_f1, spec_f0, spec_f1)

    gen = torch.Generator(device="cpu").manual_seed(4406)
    with torch.no_grad():
        for blk in model.blocks:
            for exp in blk.mlp.experts.values():  # type: ignore[union-attr]
                exp.w1_B.copy_(torch.randn(exp.w1_B.shape, generator=gen) * 0.2)
                exp.w3_B.copy_(torch.randn(exp.w3_B.shape, generator=gen) * 0.2)

    x_in = prose_f0[:8].view(2, 4, 32)
    with torch.no_grad():
        raw_out = _forward_hidden_sequence(model, x_in)

    install_adaptive_steering_guard(model, ctrl, guard_enabled=False)
    base_calls = {"count": 0}
    raw_l0 = unwrap_moe_mlp(model.blocks[0].mlp)
    hook_handle = raw_l0.base_mlp.w1.register_forward_hook(  # type: ignore[union-attr]
        lambda _m, _inp, _out: base_calls.__setitem__("count", base_calls["count"] + 1)
    )
    damping_calls = {"count": 0}
    orig_compute = ctrl.compute_damping_factors

    def _spy_compute(x: torch.Tensor, base_out: torch.Tensor, delta: torch.Tensor) -> Any:
        damping_calls["count"] += 1
        return orig_compute(x, base_out, delta)

    monkeypatch.setattr(ctrl, "compute_damping_factors", _spy_compute)
    try:
        with torch.no_grad():
            bypassed_out = _forward_hidden_sequence(model, x_in)
        assert torch.equal(bypassed_out, raw_out)
        assert base_calls["count"] == 1
        assert damping_calls["count"] == 0
        assert model.blocks[0].mlp.last_telemetry is None  # type: ignore[union-attr]

        # Re-enable via set_adaptive_steering_enabled, then disable via controller.set_enabled(False)
        assert set_adaptive_steering_enabled(model, True) == 2
        ctrl.set_enabled(False)
        with torch.no_grad():
            bypassed_ctrl_out = _forward_hidden_sequence(model, x_in)
        assert torch.equal(bypassed_ctrl_out, raw_out)
        assert damping_calls["count"] == 0

        # Direct controller.apply_damping & apply_damping_to_delta also bypass without calling compute_damping_factors
        d_out, d_tel = ctrl.apply_damping(x_in, x_in, x_in * 2.0)
        assert d_tel.bypassed is True and torch.equal(d_out, x_in * 2.0)
        dd_out, dd_tel = ctrl.apply_damping_to_delta(x_in, x_in, x_in)
        assert dd_tel.bypassed is True and torch.equal(dd_out, x_in * 2.0)
        assert damping_calls["count"] == 0
    finally:
        hook_handle.remove()

    assert remove_adaptive_steering_guard(model) == 2


# ---------------------------------------------------------------------------
# S5: Multi-Seed Stress, Error Branches & Repository Security / Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_multiseed_stress_validation_and_compliance_gates() -> None:
    """Run multi-seed stress invariants, validation branches, and security/compliance audits."""
    # 1. Multi-seed mixed-domain + hot-swap stress invariants
    for seed_offset in range(6):
        seed = 202600 + seed_offset
        gen = torch.Generator(device="cpu").manual_seed(seed)
        hidden_size = 24
        prose_f0 = _make_subspace_batch(32, hidden_size, 0, carrier_strength=2.5, noise_slice=slice(0, 6), seed=seed + 1)
        prose_f1 = _make_subspace_batch(32, hidden_size, 0, carrier_strength=2.5, noise_slice=slice(0, 6), seed=seed + 2)
        spec_f0 = _make_subspace_batch(32, hidden_size, 10, carrier_strength=2.5, noise_slice=slice(6, 24), seed=seed + 3)
        spec_f1 = _make_subspace_batch(32, hidden_size, 10, carrier_strength=2.5, noise_slice=slice(6, 24), seed=seed + 4)

        ctrl = ProseConfidenceFallbackController.from_probe_folds(
            prose_f0,
            prose_f1,
            spec_f0,
            spec_f1,
            config=FallbackControllerConfig(max_delta_base_ratio=0.45, spike_decay=1.0),
        )
        x_mix = torch.cat([prose_f0[:8], spec_f0[:8]], dim=0)
        base = torch.randn((16, hidden_size), generator=gen) * 0.5 + 1.0
        delta = torch.randn((16, hidden_size), generator=gen) * 0.1
        delta[-4:] *= 25.0  # inject OOD spikes on last 4 specialist tokens

        guarded, tel = ctrl.apply_damping_to_delta(x_mix, base, delta)
        assert bool(torch.isfinite(guarded).all().item())
        assert tel.post_damping_max_ratio <= 0.45 + 1e-5
        assert tel.min_gamma < 0.25

        # Alternating hot-swap ratchet stream
        model = _build_upcycled_crowd_model(
            hidden_size=hidden_size,
            intermediate_size=48,
            num_layers=2,
            num_experts=4,
            routing_mode="sigma_topk" if seed_offset % 2 == 0 else "softmax_topk",
            seed=seed,
        )
        base_nll = {"E0_general": 3.0, "E1_spec": 2.5}
        gate = ThermodynamicHotSwapGate(parent_energy=0.0, baseline_domain_nll=base_nll)
        start_b = extract_guarded_moe_buckets(model, num_router_fragments=2)
        cand_good = {k: v + (0.02 if k == "expert_1" else 0.0) for k, v in start_b.items()}
        cand_bad = {k: v + (0.50 if k == "expert_1" else 0.0) for k, v in start_b.items()}

        r_good = gate.try_hot_swap(
            model, cand_good, eval_fn=lambda _m: {"domain_nll": {"E0_general": 3.003, "E1_spec": 2.2}}
        )
        assert r_good.accepted is True
        r_bad = gate.try_hot_swap(
            model, cand_bad, eval_fn=lambda _m: {"domain_nll": {"E0_general": 3.050, "E1_spec": 1.8}}
        )
        assert r_bad.accepted is False and r_bad.rolled_back is True
        assert compute_buckets_sha256(extract_guarded_moe_buckets(model, num_router_fragments=2)) == compute_buckets_sha256(cand_good)

    # 2. Validation branches across configs, helpers, gate, controller, and wrappers
    with pytest.raises(TypeError, match="bool"):
        FallbackControllerConfig(enabled="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="< 1.0"):
        FallbackControllerConfig(prose_confidence_threshold=1.0)
    with pytest.raises(ValueError, match="> 0.0"):
        FallbackControllerConfig(max_delta_base_ratio=0.0)
    with pytest.raises(ValueError, match=">= 0.0"):
        FallbackControllerConfig(spike_decay=-0.1)
    with pytest.raises(ValueError, match="in \\[0.0, 1.0\\]"):
        FallbackControllerConfig(min_delta_scale=1.5)
    with pytest.raises(ValueError, match="finite"):
        FallbackControllerConfig(probe_midpoint=float("nan"))

    with pytest.raises(ValueError, match="non-empty string"):
        HotSwapGateConfig(general_domain_prefix="")
    with pytest.raises(ValueError, match="non-empty Mapping"):
        compute_buckets_sha256({})
    with pytest.raises(ValueError, match="non-empty string"):
        compute_buckets_sha256({"": torch.ones(2)})
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_buckets_sha256({"a": [1.0]})  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="non-empty mapping"):
        compute_thermodynamic_energy({}, {"a": 1.0})
    with pytest.raises(ValueError, match="non-empty mapping"):
        compute_thermodynamic_energy({"a": 1.0}, {})
    with pytest.raises(ValueError, match="identical domain keys"):
        compute_thermodynamic_energy({"a": 1.0}, {"b": 1.0})
    with pytest.raises(ValueError, match="identical domain keys"):
        compute_thermodynamic_energy({"a": 1.0}, {"a": 1.0}, domain_weights={"b": 1.0})
    with pytest.raises(ValueError, match="positive total weight"):
        compute_thermodynamic_energy({"a": 1.0}, {"a": 1.0}, domain_weights={"a": 0.0})

    gate_val = ThermodynamicHotSwapGate()
    with pytest.raises(ValueError, match="non-empty mapping"):
        gate_val.set_baseline_domain_nll({})
    with pytest.raises(TypeError, match="nn.Module"):
        gate_val.snapshot_model_state("not_a_module")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="domain_nll_baseline"):
        gate_val.evaluate_candidate_metrics(0.0, -0.1, domain_nll_child={"a": 1.0})
    with pytest.raises(ValueError, match="domain_drifts"):
        gate_val.evaluate_candidate_metrics(0.0, -0.1)
    with pytest.raises(TypeError, match="nn.Module"):
        gate_val.try_hot_swap("bad", {"a": torch.ones(1)}, eval_fn=lambda _m: {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty Mapping"):
        gate_val.try_hot_swap(nn.Linear(2, 2), {}, eval_fn=lambda _m: {})
    with pytest.raises(TypeError, match="callable"):
        gate_val.try_hot_swap(nn.Linear(2, 2), {"weight": torch.ones(2, 2)}, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="nn.Module"):
        gate_val.force_dense_fallback("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="No dense_buckets"):
        gate_val.force_dense_fallback(nn.Linear(2, 2))
    with pytest.raises(TypeError, match="mapping"):
        gate_val._resolve_domain_drifts_and_energy(123, baseline_domain_nll=None)
    with pytest.raises(ValueError, match="baseline_domain_nll"):
        gate_val._resolve_domain_drifts_and_energy({"domain_nll": {"a": 1.0}}, baseline_domain_nll=None)
    with pytest.raises(ValueError, match="energy"):
        gate_val._resolve_domain_drifts_and_energy({"domain_drifts": {"a": 0.0}}, baseline_domain_nll=None)
    with pytest.raises(ValueError, match="verify <= \\+1.0% guard"):
        gate_val._resolve_domain_drifts_and_energy({"energy": 0.0}, baseline_domain_nll=None)

    # Bucket underflow & rollback hash mismatch negative control
    m_small = _build_upcycled_crowd_model(hidden_size=16, intermediate_size=32, num_layers=1, seed=5100)
    with pytest.raises(ValueError, match="underflow"):
        load_guarded_moe_buckets(m_small, {"expert_0": torch.ones(1)}, num_router_fragments=1)
    with pytest.raises(ValueError, match="non-empty Mapping"):
        load_guarded_moe_buckets(m_small, {})
    with pytest.raises(TypeError, match="int"):
        extract_guarded_moe_buckets(m_small, num_router_fragments=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        extract_guarded_moe_buckets(m_small, num_router_fragments=0)

    # Corrupt snapshot mismatch simulation on _perform_atomic_rollback
    gate_corrupt = ThermodynamicHotSwapGate(
        HotSwapGateConfig(verify_rollback_hash=True),
        parent_energy=0.0,
        baseline_domain_nll={"E0_general": 3.0},
    )
    snap_small = extract_guarded_moe_buckets(m_small, num_router_fragments=1)
    with pytest.raises(RuntimeError, match="Atomic rollback verification failed"):
        gate_corrupt._perform_atomic_rollback(
            m_small,
            snap_small,
            prefer_dense=False,
            load_buckets_fn=lambda _m, _b: None,
            extract_buckets_fn=lambda _m: {"expert_0": torch.zeros_like(snap_small["expert_0"])},
            num_router_fragments=1,
        )

    # Controller, layer wrapper & installer validation branches
    with pytest.raises(TypeError, match="torch.Tensor"):
        ProseConfidenceFallbackController(probe_direction=[1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty 1D"):
        ProseConfidenceFallbackController(probe_direction=torch.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        ProseConfidenceFallbackController(probe_direction=torch.tensor([1.0, float("nan")]))

    ctrl_bad = ProseConfidenceFallbackController(probe_direction=torch.ones(8))
    with pytest.raises(ValueError, match="ndim >= 1"):
        ctrl_bad.compute_prose_confidence(torch.tensor(1.0))
    with pytest.raises(ValueError, match="does not match probe_direction"):
        ctrl_bad.compute_prose_confidence(torch.ones((2, 4)))
    with pytest.raises(TypeError, match="torch.Tensor"):
        ctrl_bad.compute_damping_factors([1.0], torch.ones(1), torch.ones(1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Incompatible tensor shapes"):
        ctrl_bad.compute_damping_factors(torch.ones((2, 8)), torch.ones((3, 8)), torch.ones((3, 8)))

    with pytest.raises(TypeError, match="nn.Module"):
        AdaptiveSteeredMoELayer("bad", ctrl_bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ProseConfidenceFallbackController"):
        AdaptiveSteeredMoELayer(nn.Linear(8, 8), "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dense_mlp must be provided"):
        AdaptiveSteeredMoELayer(nn.Linear(8, 8), ctrl_bad)
    with pytest.raises(TypeError, match="nn.Module"):
        AdaptiveSteeredMoELayer(nn.Linear(8, 8), ctrl_bad, dense_mlp="bad")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="nn.Module"):
        install_adaptive_steering_guard("bad", ctrl_bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="blocks"):
        install_adaptive_steering_guard(nn.Linear(4, 4), ctrl_bad)
    with pytest.raises(ValueError, match="empty target set"):
        install_adaptive_steering_guard(m_small, ctrl_bad, layer_indices=())
    with pytest.raises(ValueError, match="out of range"):
        install_adaptive_steering_guard(m_small, ctrl_bad, layer_indices=(99,))
    with pytest.raises(KeyError, match="Missing ProseConfidenceFallbackController"):
        install_adaptive_steering_guard(m_small, {0: ctrl_bad}, layer_indices=(0, 1)) if len(m_small.blocks) > 1 else (
            install_adaptive_steering_guard(m_small, {}, layer_indices=(0,))
        )

    # 3. Repository AGPL-3.0, ZeroStubAudit, PRSecurityScanner, lateral movement, and gatekeeper scans
    lateral_guard = _load_module_from_path("test_no_lateral_movement_ctrl06", REPO_ROOT / "tests" / "test_no_lateral_movement.py")
    gatekeeper = _load_module_from_path("pr_gatekeeper_ctrl06", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/adaptive_steering_guard.py", TARGET_MODULE_PATH),
        ("tests/test_adaptive_steering_guard_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

        stub_res = ZeroStubAudit().audit_python_code(src_text)
        assert stub_res.is_clean, f"ZeroStubAudit violations in {rel}: {stub_res.violations}"
        assert stub_res.penalty_energy == 0.0

        assert scanner.scan_python_ast_security(rel, src_text) == []
        assert scanner.scan_text_for_payment_solicitation(src_text, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(src_text, rel) == []
        assert scanner.scan_unicode_trojan_source(src_text, rel) == []
        assert lateral_guard.audit_source_for_lateral_movement(src_text, rel) == []

        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in src_text.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []
        changed_files.append(
            {"filename": rel, "status": "added", "additions": len(src_text.splitlines()), "patch": "", "content": src_text}
        )

    telemetry_json = json.dumps({"delta_energy": -0.18, "worst_domain_drift_pct": 0.25, "mean_gamma": 0.91})
    assert gatekeeper.forbidden_scan(f"diff --git a/x.py b/x.py\n+++ b/x.py\n+PAYLOAD = {telemetry_json!r}") == []

    pr_audit = scanner.audit_pull_request(
        pr_number=1106,
        pr_title="feat(swarm): runtime thermodynamic dense-fallback guard for open-weight MoE neural control",
        pr_body="Implements NEURAL-CTRL-06 runtime thermodynamic dense-fallback guard and scenario tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): runtime thermodynamic dense-fallback guard for open-weight MoE neural control"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []

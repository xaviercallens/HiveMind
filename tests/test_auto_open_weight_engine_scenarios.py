# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Six deterministic CPU verification scenarios for ``HM-S35-02`` Auto Open-Weight Engine.

Covers ``src/hivemind/swarm/auto_open_weight_engine.py``, ``scripts/hivemind_auto_open_weight.py``,
``docs/AUTO_OPEN_WEIGHT_CAPABILITY_PLAN.md``, and ``ROADMAP.md`` Section 3.25.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hivemind.swarm.auto_open_weight_engine import (
    ANSE_MAXIMUM_PAIN,
    DEFAULT_GENERAL_DOMAIN,
    AutoOpenWeightConfig,
    SingleShotVaultLock,
    apply_stage_s5_2_oof_orthogonal_multigate_inplace,
    apply_stage_s5_4_folded_final_norm_calibration_inplace,
    clone_state_dict,
    compute_anse_objective,
    compute_oof_orthogonal_router_direction,
    count_edited_parameters,
    extract_mp_guarded_prose_projector,
    hash_state_dict,
    profile_moe_layers_and_experts,
    project_lora_pair_qr_svd_nullspace,
    run_auto_open_weight_s5_synergy,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:

    class _FallbackAuditResult:
        def __init__(self, violations: list[str]) -> None:
            self.violations = violations
            self.is_clean = len(violations) == 0

    class ZeroStubAudit:  # type: ignore[no-redef]
        """CI-safe AST stub verifier when external gwaya package is absent."""

        def audit_python_code(self, code: str) -> _FallbackAuditResult:
            tree = ast.parse(code)
            bad: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    body = [
                        s
                        for s in node.body
                        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))
                    ]
                    if not body or (len(body) == 1 and isinstance(body[0], ast.Pass)):
                        bad.append(f"stub function: {node.name}")
                if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                    if isinstance(node.exc.func, ast.Name) and node.exc.func.id == "NotImplementedError":
                        bad.append("NotImplementedError")
            return _FallbackAuditResult(bad)

        def check_file(self, path: str | Path) -> list[str]:
            return self.audit_python_code(Path(path).read_text(encoding="utf-8")).violations


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "auto_open_weight_engine.py"
CLI_PATH = REPO_ROOT / "scripts" / "hivemind_auto_open_weight.py"
DOC_PATH = REPO_ROOT / "docs" / "AUTO_OPEN_WEIGHT_CAPABILITY_PLAN.md"


def _load_cli_module() -> Any:
    spec = importlib.util.spec_from_file_location("hivemind_auto_open_weight", CLI_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _build_synthetic_c2_moe_world(seed: int = 20261124) -> dict[str, Any]:
    """Construct a deterministic 6-layer, 8-expert sigma_topk MoE mirroring C2_final.

    - Layer 3 is a beneficial anchor layer (improves E0 prose by -0.08 nats AND specialists).
    - Layer 0 is neutral.
    - Layers 1, 2, 4, 5 are prose-interfering layers, with tail Layers 4, 5 carrying
      additional specialist-to-prose subspace leakage cured by Stage S5.3 QR-SVD.
    - Experts {0, 1, 3} are pure rogue/prose-harming experts.
    - Expert 5 is dual-active: fires on both prose (harming E0) and E5_exoplanet (helping E5).
    - Experts {2, 4, 6, 7} are core specialists for E1_code, E2_math, E6_stars, E7_cyber.
    - Final LayerNorm is over-amplified by 1.15x (cured by Stage S5.4 folded temperature).
    """
    rng = np.random.default_rng(seed)
    n_layers = 6
    n_experts = 8
    d_model = 16
    d_mid = 12
    rank = 4
    n_tok = 32

    domains = [
        DEFAULT_GENERAL_DOMAIN,
        "E1_python_systems_code",
        "E2_math_reasoning",
        "E5_exoplanet_transit",
        "E6_stellar_spectroscopy",
        "E7_defensive_cyber",
    ]
    spec_domains = domains[1:]

    # Orthonormal basis vectors in d_model and d_mid so subspaces are exact & clean
    q_model, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
    q_mid, _ = np.linalg.qr(rng.standard_normal((d_mid, d_mid)))

    # Domain centroids in d_model: specialist domains live in dims 0..4, prose lives in dim 5 + shared dim 3
    spec_dirs = {dom: q_model[:, idx] for idx, dom in enumerate(spec_domains)}
    prose_orth_dir = q_model[:, 6]  # strictly orthogonal to all 5 specialist centroids!
    prose_center = 1.6 * prose_orth_dir + 0.55 * spec_dirs["E5_exoplanet_transit"]

    # Inputs per domain (n_tok, d_model) and FFN mid activations (n_tok, d_mid)
    domain_x: dict[str, np.ndarray] = {}
    domain_mid: dict[str, np.ndarray] = {}
    domain_target_vec: dict[str, np.ndarray] = {}

    prose_mid_subspace = q_mid[:, :3]  # 3D prose FFN-mid subspace
    spec_mid_dirs = {dom: q_mid[:, 3 + idx] for idx, dom in enumerate(spec_domains)}

    domain_x[DEFAULT_GENERAL_DOMAIN] = prose_center[None, :] + 0.08 * rng.standard_normal((n_tok, d_model))
    prose_mid_mean = np.sum(prose_mid_subspace, axis=1)
    domain_mid[DEFAULT_GENERAL_DOMAIN] = (
        1.4 * prose_mid_mean[None, :]
        + 0.25 * (rng.standard_normal((n_tok, 3)) @ prose_mid_subspace.T)
        + 0.02 * rng.standard_normal((n_tok, d_mid))
    )
    domain_target_vec[DEFAULT_GENERAL_DOMAIN] = q_model[:, 7]

    for idx, dom in enumerate(spec_domains):
        domain_x[dom] = 1.8 * spec_dirs[dom][None, :] + 0.08 * rng.standard_normal((n_tok, d_model))
        domain_mid[dom] = (
            1.5 * spec_mid_dirs[dom][None, :]
            + 0.25 * (rng.standard_normal((n_tok, 3)) @ prose_mid_subspace.T)
            + 0.03 * rng.standard_normal((n_tok, d_mid))
        )
        domain_target_vec[dom] = q_model[:, 8 + idx]

    state: dict[str, np.ndarray] = {}
    interfering_layers = {1, 2, 4, 5}
    tail_layers = {4, 5}
    anchor_layer = 3

    spec_expert_map = {
        2: "E1_python_systems_code",
        4: "E2_math_reasoning",
        5: "E5_exoplanet_transit",
        6: "E6_stellar_spectroscopy",
        7: "E7_defensive_cyber",
    }

    for l_idx in range(n_layers):
        rw = np.zeros((n_experts, d_model), dtype=np.float32)
        w2_b = np.zeros((n_experts, d_model, rank), dtype=np.float32)
        w2_a = np.zeros((n_experts, rank, d_mid), dtype=np.float32)

        if l_idx == anchor_layer:
            # Beneficial anchor layer: experts {0, 1, 3} help general prose! Zeroing them HURTS E0!
            for sink_e in (0, 1, 3):
                rw[sink_e] = (2.0 * prose_orth_dir).astype(np.float32)
                w2_b[sink_e, :, 0] = (0.38 * domain_target_vec[DEFAULT_GENERAL_DOMAIN]).astype(np.float32)
                w2_a[sink_e, 0, :] = prose_mid_subspace[:, sink_e % 3].astype(np.float32)
            for exp_e, dom in spec_expert_map.items():
                rw[exp_e] = (2.2 * spec_dirs[dom]).astype(np.float32)
                w2_b[exp_e, :, 0] = (0.35 * domain_target_vec[dom]).astype(np.float32)
                w2_a[exp_e, 0, :] = spec_mid_dirs[dom].astype(np.float32)
        elif l_idx in interfering_layers:
            # Rogue / prose-harming experts {0, 1, 3} push prose AWAY from target
            for idx_s, sink_e in enumerate((0, 1, 3)):
                scale = 2.2 if sink_e == 3 else 1.15
                rw[sink_e] = (2.0 * prose_orth_dir).astype(np.float32)
                w2_b[sink_e, :, 0] = (-0.30 * scale * domain_target_vec[DEFAULT_GENERAL_DOMAIN]).astype(np.float32)
                w2_a[sink_e, 0, :] = prose_mid_subspace[:, idx_s].astype(np.float32)

            # Dual-active expert 5: helps E5 strongly, but its router also fires on prose_orth_dir!
            rw[5] = (2.2 * spec_dirs["E5_exoplanet_transit"] + 1.2 * prose_orth_dir).astype(np.float32)
            w2_b[5, :, 0] = (
                0.52 * domain_target_vec["E5_exoplanet_transit"]
                - 0.14 * domain_target_vec[DEFAULT_GENERAL_DOMAIN]
            ).astype(np.float32)
            w2_a[5, 0, :] = (spec_mid_dirs["E5_exoplanet_transit"] + 0.35 * prose_mid_subspace[:, 0]).astype(np.float32)

            # Core specialists {2, 4, 6, 7}
            for exp_e in (2, 4, 6, 7):
                dom = spec_expert_map[exp_e]
                rw[exp_e] = (2.2 * spec_dirs[dom] + (0.20 * prose_orth_dir if l_idx in tail_layers else -0.8 * prose_orth_dir)).astype(
                    np.float32
                )
                w2_b[exp_e, :, 0] = (
                    0.42 * domain_target_vec[dom]
                    - (0.04 * domain_target_vec[DEFAULT_GENERAL_DOMAIN] if l_idx in tail_layers else 0.0)
                ).astype(np.float32)
                w2_a[exp_e, 0, :] = (
                    spec_mid_dirs[dom] + (0.30 * prose_mid_subspace[:, 1] if l_idx in tail_layers else 0.0)
                ).astype(np.float32)

        state[f"layer_{l_idx}.router_w"] = rw
        state[f"layer_{l_idx}.w2_B"] = w2_b
        state[f"layer_{l_idx}.w2_A"] = w2_a

    base_ln_w = np.ones((d_model,), dtype=np.float32)
    base_ln_b = np.zeros((d_model,), dtype=np.float32)
    # C2 final LayerNorm has slight over-confident scale drift (1.18x)
    state["final_ln_w"] = (1.18 * np.ones((d_model,), dtype=np.float32)).astype(np.float32)
    state["final_ln_b"] = (0.04 * domain_target_vec[DEFAULT_GENERAL_DOMAIN] * (-1.0)).astype(np.float32)

    base_nll = {dom: 4.20 for dom in domains}

    def eval_moe(st: Mapping[str, np.ndarray], *, noise_seed_offset: int = 0) -> dict[str, float]:
        if "trigger_eval_crash" in st:
            raise RuntimeError("Simulated evaluator failure")
        out: dict[str, float] = {}
        ln_w = np.asarray(st["final_ln_w"], dtype=np.float64)
        ln_b = np.asarray(st.get("final_ln_b", np.zeros_like(ln_w)), dtype=np.float64)
        # Over-scaling penalty around optimal norm scale ~1.02 (models calibration ECE/NLL)
        scale_mean = float(np.mean(ln_w))
        cal_pen = 0.22 * ((scale_mean - 1.02) ** 2)

        for d_idx, dom in enumerate(domains):
            x = domain_x[dom]
            mid = domain_mid[dom]
            tgt = domain_target_vec[dom]
            residual = np.zeros_like(x, dtype=np.float64)

            for l_idx in range(n_layers):
                rw = np.asarray(st[f"layer_{l_idx}.router_w"], dtype=np.float64)
                wb = np.asarray(st[f"layer_{l_idx}.w2_B"], dtype=np.float64)
                wa = np.asarray(st[f"layer_{l_idx}.w2_A"], dtype=np.float64)
                gates = _sigmoid(x @ rw.T)  # (n_tok, n_experts) sigma_topk
                # Top-4 active experts per token
                top_idx = np.argsort(gates, axis=1)[:, -4:]
                mask = np.zeros_like(gates)
                np.put_along_axis(mask, top_idx, 1.0, axis=1)
                active_g = gates * mask
                for e_id in range(n_experts):
                    delta_mat = wb[e_id] @ wa[e_id]  # (d_model, d_mid)
                    exp_out = mid @ delta_mat.T  # (n_tok, d_model)
                    residual = residual + active_g[:, e_id : e_id + 1] * exp_out

            normed = residual * ln_w[None, :] + ln_b[None, :]
            align = float(np.mean(normed @ tgt))
            quad_reg = 0.03 * float(np.mean(np.sum(normed * normed, axis=1)))
            tiny_shift = 0.002 * float(noise_seed_offset) * (1.0 if d_idx == 0 else -0.5)
            out[dom] = float(base_nll[dom] - 0.55 * align + quad_reg + cal_pen + tiny_shift)
        return out

    # Compute genuine causal layer & expert attribution on this synthetic world!
    c2_raw_nll = eval_moe(state)
    layer_e0_harm: list[float] = []
    for l_idx in range(n_layers):
        bypass = clone_state_dict(state)
        bypass[f"layer_{l_idx}.w2_B"] = np.zeros_like(bypass[f"layer_{l_idx}.w2_B"])
        byp_nll = eval_moe(bypass)
        # Positive means layer l_idx was HARMING E0 (bypassing lowered E0 NLL)
        layer_e0_harm.append(float(c2_raw_nll[DEFAULT_GENERAL_DOMAIN] - byp_nll[DEFAULT_GENERAL_DOMAIN]))

    expert_e0_harm: list[float] = []
    expert_spec_loss: list[float] = []
    for e_id in range(n_experts):
        zero_e = clone_state_dict(state)
        for l_idx in interfering_layers:
            zero_e[f"layer_{l_idx}.w2_B"][e_id] = 0.0
        z_nll = eval_moe(zero_e)
        expert_e0_harm.append(float(c2_raw_nll[DEFAULT_GENERAL_DOMAIN] - z_nll[DEFAULT_GENERAL_DOMAIN]))
        spec_loss = float(np.mean([z_nll[d] - c2_raw_nll[d] for d in spec_domains]))
        expert_spec_loss.append(spec_loss)

    layer_prose_inputs = {l_idx: domain_x[DEFAULT_GENERAL_DOMAIN] for l_idx in range(n_layers)}
    layer_spec_inputs = {
        l_idx: {dom: domain_x[dom] for dom in spec_domains}
        for l_idx in range(n_layers)
    }
    tail_prose_mid = {l_idx: domain_mid[DEFAULT_GENERAL_DOMAIN] for l_idx in range(n_layers)}

    return {
        "state": state,
        "base_nll": base_nll,
        "eval_fn": eval_moe,
        "layer_e0_harm": layer_e0_harm,
        "expert_e0_harm": expert_e0_harm,
        "expert_spec_loss": expert_spec_loss,
        "layer_prose_inputs": layer_prose_inputs,
        "layer_spec_inputs": layer_spec_inputs,
        "tail_prose_mid": tail_prose_mid,
        "base_ln_w": base_ln_w,
        "base_ln_b": base_ln_b,
        "spec_dirs": spec_dirs,
    }


def test_s1_s5_pass_synergy_strictly_outperforms_s1_s2_and_s4_alone() -> None:
    """Scenario 1: S5 PASS-Synergy beats S1 (uniform sink), S2 (anchor sink), and S4 (uniform multigate)."""
    world = _build_synthetic_c2_moe_world()
    c2 = world["state"]
    base_nll = world["base_nll"]
    eval_fn = world["eval_fn"]

    # Raw uncorrected C2 violates the +1.0% general prose guard -> Maximum Pain
    raw_e, raw_diff, raw_gen_rel, _ = compute_anse_objective(eval_fn(c2), base_nll)
    assert raw_gen_rel > 0.01
    assert raw_e == ANSE_MAXIMUM_PAIN
    assert raw_diff < 0.0

    # Baseline S1: uniform {0, 1, 3, 5} sinking across ALL layers (including anchor Layer 3!)
    s1_state = clone_state_dict(c2)
    for l_idx in range(6):
        for e_id in (0, 1, 3, 5):
            s1_state[f"layer_{l_idx}.w2_B"][e_id] = 0.0
    s1_state["final_ln_w"] = (0.25 * s1_state["final_ln_w"] + 0.75 * world["base_ln_w"]).astype(np.float32)
    e_s1, _, gen_s1, _ = compute_anse_objective(eval_fn(s1_state), base_nll)
    assert gen_s1 <= 0.01

    # Baseline S2: protect beneficial anchor Layer 3, sink {0, 1, 3, 5} only on interfering layers
    s2_state = clone_state_dict(c2)
    for l_idx in (1, 2, 4, 5):
        for e_id in (0, 1, 3, 5):
            s2_state[f"layer_{l_idx}.w2_B"][e_id] = 0.0
    s2_state["final_ln_w"] = (0.65 * s2_state["final_ln_w"] + 0.35 * world["base_ln_w"]).astype(np.float32)
    e_s2, _, gen_s2, _ = compute_anse_objective(eval_fn(s2_state), base_nll)
    assert gen_s2 <= 0.01
    assert e_s2 < e_s1  # Protecting beneficial anchor Layer 3 beats uniform S1!

    # Baseline S4: uniform {0, 1, 3} sinking across ALL layers (still zeroing anchor Layer 3!) + multigate on 5
    s4_state = clone_state_dict(c2)
    for l_idx in range(6):
        for e_id in (0, 1, 3):
            s4_state[f"layer_{l_idx}.w2_B"][e_id] = 0.0
    prof_uniform = profile_moe_layers_and_experts(
        [0.05] * 6,  # pretend all layers are interfering (no anchor protection)
        world["expert_e0_harm"],
        world["expert_spec_loss"],
    )
    apply_stage_s5_2_oof_orthogonal_multigate_inplace(
        s4_state,
        prof_uniform,
        world["layer_prose_inputs"],
        world["layer_spec_inputs"],
        router_beta=1.40,
    )
    s4_state["final_ln_w"] = (0.80 * s4_state["final_ln_w"] + 0.20 * world["base_ln_w"]).astype(np.float32)
    e_s4, _, gen_s4, _ = compute_anse_objective(eval_fn(s4_state), base_nll)
    assert gen_s4 <= 0.01

    # Gen-2 S5 PASS-Synergy: Anchor protection (S2) + Selective sink & OOF Multi-Gate (S4) + Tail QR-SVD (S3) + Folded LN (DYN-14)
    cfg = AutoOpenWeightConfig(router_beta=1.40, svd_damp_strength=0.90)
    s5_state, s5_res = run_auto_open_weight_s5_synergy(
        c2,
        base_nll,
        eval_fn,
        layer_e0_harm_nats=world["layer_e0_harm"],
        expert_e0_harm_nats=world["expert_e0_harm"],
        expert_spec_loss_when_zeroed=world["expert_spec_loss"],
        layer_prose_inputs=world["layer_prose_inputs"],
        layer_specialist_inputs=world["layer_spec_inputs"],
        tail_layer_prose_ffn_mid=world["tail_prose_mid"],
        base_final_ln_w=world["base_ln_w"],
        base_final_ln_b=world["base_ln_b"],
        config=cfg,
    )

    assert s5_res.committed is True
    assert s5_res.verdict == "PASS"
    assert s5_res.general_rel_drift <= 0.005
    assert s5_res.all_domains_negative is True  # All 6/6 domains strictly improved vs base!
    assert s5_res.anse_energy < e_s4 < 0.0
    assert s5_res.anse_energy < e_s2 < e_s1 < 0.0
    assert len(s5_res.receipt_sha256) == 64
    assert hash_state_dict(s5_state) == s5_res.checkpoint_sha256


def test_s2_bitwise_anchor_layer_preservation_and_specialist_multigate_orthogonality() -> None:
    """Scenario 2: Anchor Layer 3 is 100% bitwise untouched and specialist router drift is < 1e-5."""
    world = _build_synthetic_c2_moe_world()
    c2 = world["state"]
    cfg = AutoOpenWeightConfig(router_beta=0.80)
    s5_state, s5_res = run_auto_open_weight_s5_synergy(
        c2,
        world["base_nll"],
        world["eval_fn"],
        layer_e0_harm_nats=world["layer_e0_harm"],
        expert_e0_harm_nats=world["expert_e0_harm"],
        expert_spec_loss_when_zeroed=world["expert_spec_loss"],
        layer_prose_inputs=world["layer_prose_inputs"],
        layer_specialist_inputs=world["layer_spec_inputs"],
        tail_layer_prose_ffn_mid=world["tail_prose_mid"],
        base_final_ln_w=world["base_ln_w"],
        base_final_ln_b=world["base_ln_b"],
        config=cfg,
    )

    assert 3 in s5_res.profile.beneficial_anchor_layers
    assert 0 in s5_res.profile.neutral_layers
    assert s5_res.profile.pure_sink_experts == (0, 1, 3)
    assert s5_res.profile.preserved_multigate_experts == (5,)

    # Anchor Layer 3 and Neutral Layer 0 must be 100% bitwise identical!
    for protected_l in (0, 3):
        for suffix in ("router_w", "w2_B", "w2_A"):
            key = f"layer_{protected_l}.{suffix}"
            assert np.array_equal(c2[key], s5_state[key]), f"Protected tensor {key} was modified!"

    # On interfering layers, router row 5 shift must have < 1e-5 inner-product drift on ALL calibrated specialist centroids!
    for l_idx in s5_res.profile.interfering_layers:
        delta_row_5 = s5_state[f"layer_{l_idx}.router_w"][5].astype(np.float64) - c2[
            f"layer_{l_idx}.router_w"
        ][5].astype(np.float64)
        for dom_name, dom_acts in world["layer_spec_inputs"][l_idx].items():
            centroid_vec = np.mean(dom_acts, axis=0)
            logit_shift = abs(float(delta_row_5 @ centroid_vec))
            assert logit_shift < 1e-5, f"Specialist centroid {dom_name} shifted by {logit_shift}"


def test_s3_oof_snr_noise_gate_and_mid_pipeline_regressing_stage_rollback() -> None:
    """Scenario 3: Pure Gaussian calibration noise is rejected by OOF SNR, and regressing stages roll back bitwise."""
    rng = np.random.default_rng(42)
    # Indistinguishable zero-mean Gaussian prose vs specialist activations -> OOF SNR < 0.25 -> None
    noisy_prose = rng.standard_normal((24, 16))
    noisy_specs = {
        "E1_code": rng.standard_normal((24, 16)),
        "E2_math": rng.standard_normal((24, 16)),
    }
    u_dir, oof_snr = compute_oof_orthogonal_router_direction(noisy_prose, noisy_specs, min_oof_snr=0.40)
    assert u_dir is None
    assert oof_snr < 0.40

    # Inject a corrupting custom stage AND an exception-throwing custom stage after S5.1..S5.4:
    # S5.1..S5.4 gains must be preserved while both bad stages roll back cleanly!
    world = _build_synthetic_c2_moe_world()

    def _corrupting_stage(st: Any) -> int:
        st["final_ln_w"] = (st["final_ln_w"] * 3.5).astype(np.float32)
        return int(st["final_ln_w"].size)

    def _crashing_stage(st: Any) -> int:
        st["trigger_eval_crash"] = np.ones((1,), dtype=np.float32)
        return 1

    def _nan_stage(st: Any) -> int:
        st["final_ln_w"] = np.full_like(st["final_ln_w"], np.nan)
        return int(st["final_ln_w"].size)

    def _noop_stage(_st: Any) -> int:
        return 0

    _, res = run_auto_open_weight_s5_synergy(
        world["state"],
        world["base_nll"],
        world["eval_fn"],
        layer_e0_harm_nats=world["layer_e0_harm"],
        expert_e0_harm_nats=world["expert_e0_harm"],
        expert_spec_loss_when_zeroed=world["expert_spec_loss"],
        layer_prose_inputs=world["layer_prose_inputs"],
        layer_specialist_inputs=world["layer_spec_inputs"],
        tail_layer_prose_ffn_mid=world["tail_prose_mid"],
        base_final_ln_w=world["base_ln_w"],
        base_final_ln_b=world["base_ln_b"],
        extra_custom_stages=[
            ("S5.5_corrupting_probe", _corrupting_stage),
            ("S5.6_crashing_probe", _crashing_stage),
            ("S5.6b_nan_probe", _nan_stage),
            ("S5.7_noop_probe", _noop_stage),
        ],
        config=AutoOpenWeightConfig(router_beta=0.80),
    )
    assert res.committed is True
    by_name = {s.stage_name: s for s in res.stages}
    assert by_name["S5.5_corrupting_probe"].accepted is False
    assert by_name["S5.5_corrupting_probe"].pre_sha256 == by_name["S5.5_corrupting_probe"].post_sha256
    assert by_name["S5.6_crashing_probe"].accepted is False
    assert by_name["S5.6_crashing_probe"].reason == "rolled_back_eval_exception"
    assert by_name["S5.6b_nan_probe"].accepted is False
    assert by_name["S5.7_noop_probe"].accepted is False
    assert by_name["S5.7_noop_probe"].reason == "noop_zero_parameters_changed"


def test_s4_single_shot_virgin_vault_lock_and_anse_maximum_pain_rollback() -> None:
    """Scenario 4: Infeasible guard triggers bitwise rollback (E=1e6), and SingleShotVaultLock blocks re-use."""
    world = _build_synthetic_c2_moe_world()
    pristine_hash = hash_state_dict(world["state"])

    # Adversarial evaluator where every state violates the general guard (+25% E0 drift)
    def _infeasible_eval(_st: Any) -> dict[str, float]:
        bad = dict(world["base_nll"])
        bad[DEFAULT_GENERAL_DOMAIN] = bad[DEFAULT_GENERAL_DOMAIN] * 1.25
        return bad

    rolled_state, fail_res = run_auto_open_weight_s5_synergy(
        world["state"],
        world["base_nll"],
        _infeasible_eval,
        layer_e0_harm_nats=world["layer_e0_harm"],
        expert_e0_harm_nats=world["expert_e0_harm"],
        expert_spec_loss_when_zeroed=world["expert_spec_loss"],
    )
    assert fail_res.committed is False
    assert fail_res.anse_energy == ANSE_MAXIMUM_PAIN
    assert fail_res.verdict == "ROLLED_BACK_MAXIMUM_PAIN"
    assert hash_state_dict(rolled_state) == pristine_hash

    # Single-shot virgin holdout vault evaluation succeeds once and refuses any second access!
    s5_state, _ = run_auto_open_weight_s5_synergy(
        world["state"],
        world["base_nll"],
        world["eval_fn"],
        layer_e0_harm_nats=world["layer_e0_harm"],
        expert_e0_harm_nats=world["expert_e0_harm"],
        expert_spec_loss_when_zeroed=world["expert_spec_loss"],
        layer_prose_inputs=world["layer_prose_inputs"],
        layer_specialist_inputs=world["layer_spec_inputs"],
        tail_layer_prose_ffn_mid=world["tail_prose_mid"],
        base_final_ln_w=world["base_ln_w"],
        base_final_ln_b=world["base_ln_b"],
        config=AutoOpenWeightConfig(router_beta=0.80),
    )
    vault_lock = SingleShotVaultLock("vault_skip320")
    assert vault_lock.consumed is False
    vault_receipt = vault_lock.evaluate_once(
        s5_state,
        world["base_nll"],
        lambda st: world["eval_fn"](st, noise_seed_offset=1),
    )
    assert vault_lock.consumed is True
    assert vault_receipt["verdict_ip300"] == "PASS"
    assert vault_receipt["guard_ok"] is True
    assert vault_lock.receipt_sha256 == vault_receipt["vault_receipt_sha256"]

    with pytest.raises(RuntimeError, match="already evaluated"):
        vault_lock.evaluate_once(s5_state, world["base_nll"], world["eval_fn"])


def test_s5_cli_diagnose_and_s5_surgery_roundtrip_npz_and_safetensors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Scenario 5: CLI diagnose + run-s5-synergy across .npz and .safetensors with exit codes 0 and 2."""
    cli = _load_cli_module()
    world = _build_synthetic_c2_moe_world()

    npz_in = tmp_path / "c2_raw.npz"
    cli.save_checkpoint_arrays(npz_in, world["state"])

    # Unhealthy raw checkpoint (rogue expert_3 norm spike) -> exit code 2
    rc_diag_bad = cli.main(["diagnose", "--checkpoint", str(npz_in), "--rogue-ratio", "1.5", "--json"])
    out_bad = json.loads(capsys.readouterr().out)
    assert rc_diag_bad == 2
    assert out_bad["healthy"] is False
    assert out_bad["rogue_alert_count"] >= 1

    # Markdown output branch on diagnose
    rc_md = cli.main(["diagnose", "--checkpoint", str(npz_in), "--rogue-ratio", "100.0", "--ln-drift-max", "0.5"])
    md_text = capsys.readouterr().out
    assert rc_md == 0
    assert "# HiveMind Auto Open-Weight Report" in md_text

    attr_path = tmp_path / "attribution.json"
    attr_path.write_text(
        json.dumps(
            {
                "layer_e0_harm_nats": world["layer_e0_harm"],
                "expert_e0_harm_nats": world["expert_e0_harm"],
                "expert_spec_loss_when_zeroed": world["expert_spec_loss"],
            }
        ),
        encoding="utf-8",
    )

    npz_out = tmp_path / "c2_s5.npz"
    rc_surg = cli.main(
        [
            "run-s5-synergy",
            "--checkpoint",
            str(npz_in),
            "--attribution-json",
            str(attr_path),
            "--output",
            str(npz_out),
            "--lambda-ln",
            "0.80",
            "--tau-temp",
            "1.02",
            "--json",
        ]
    )
    surg_summary = json.loads(capsys.readouterr().out)
    assert rc_surg == 0
    assert surg_summary["beneficial_anchor_layers"] == [3]
    assert surg_summary["pure_sink_experts"] == [0, 1, 3]
    assert surg_summary["preserved_multigate_experts"] == [5]

    # Markdown branch on run-s5-synergy + .safetensors round-trip if safetensors is available
    st_in = tmp_path / "c2_raw.safetensors"
    st_out = tmp_path / "c2_s5.safetensors"
    cli.save_checkpoint_arrays(st_in, world["state"])
    rc_st = cli.main(
        [
            "run-s5-synergy",
            "--checkpoint",
            str(st_in),
            "--attribution-json",
            str(attr_path),
            "--output",
            str(st_out),
        ]
    )
    assert rc_st == 0
    _ = capsys.readouterr().out
    reloaded = cli.load_checkpoint_arrays(st_out)
    assert np.array_equal(reloaded["layer_3.w2_B"], world["state"]["layer_3.w2_B"])
    for e_sink in (0, 1, 3):
        assert float(np.linalg.norm(reloaded["layer_1.w2_B"][e_sink])) == 0.0


def test_s6_validation_errors_roadmap_and_security_stub_compliance_gates(tmp_path: Path) -> None:
    """Scenario 6: Input validation errors, ROADMAP.md §3.25 presence, PRSecurityScanner, ZeroStubAudit."""
    cli = _load_cli_module()

    # Config & helper validation error branches
    with pytest.raises(ValueError, match="general_domain"):
        AutoOpenWeightConfig(general_domain="")
    with pytest.raises(ValueError, match="general_guard_rel"):
        AutoOpenWeightConfig(general_guard_rel=0.9)
    with pytest.raises(ValueError, match="anchor_benefit_floor_nats"):
        AutoOpenWeightConfig(anchor_benefit_floor_nats=0.1)
    with pytest.raises(ValueError, match="prose_harm_layer_floor_nats"):
        AutoOpenWeightConfig(prose_harm_layer_floor_nats=-0.1)
    with pytest.raises(ValueError, match="tail_svd_top_k_layers"):
        AutoOpenWeightConfig(tail_svd_top_k_layers=-1)
    with pytest.raises(ValueError, match="router_beta"):
        AutoOpenWeightConfig(router_beta=0.0)
    with pytest.raises(ValueError, match="min_oof_snr"):
        AutoOpenWeightConfig(min_oof_snr=-1.0)
    with pytest.raises(ValueError, match="prose_nullspace_energy_keep"):
        AutoOpenWeightConfig(prose_nullspace_energy_keep=1.0)
    with pytest.raises(ValueError, match="svd_damp_strength"):
        AutoOpenWeightConfig(svd_damp_strength=1.5)
    with pytest.raises(ValueError, match="lambda_ln_grid"):
        AutoOpenWeightConfig(lambda_ln_grid=())
    with pytest.raises(ValueError, match="tau_temp_grid"):
        AutoOpenWeightConfig(tau_temp_grid=(0.0,))
    with pytest.raises(ValueError, match="vault_id"):
        SingleShotVaultLock("")
    with pytest.raises(ValueError, match="non-empty"):
        clone_state_dict({})
    with pytest.raises(ValueError, match="non-finite"):
        clone_state_dict({"bad": np.array([np.nan])})
    with pytest.raises(ValueError, match="Shape mismatch"):
        count_edited_parameters({"a": np.zeros((2,))}, {"a": np.zeros((3,))})
    with pytest.raises(ValueError, match="Missing required general_domain"):
        compute_anse_objective({"x": 1.0}, {"x": 1.0})
    with pytest.raises(ValueError, match="identical domain keys"):
        compute_anse_objective({DEFAULT_GENERAL_DOMAIN: 1.0, "a": 1.0}, {DEFAULT_GENERAL_DOMAIN: 1.0})
    e_bad, _, _, _ = compute_anse_objective({DEFAULT_GENERAL_DOMAIN: 1.0}, {DEFAULT_GENERAL_DOMAIN: 0.0})
    assert e_bad == ANSE_MAXIMUM_PAIN

    with pytest.raises(ValueError, match="equal length"):
        profile_moe_layers_and_experts([0.1], [0.1, 0.2], [0.1])
    with pytest.raises(ValueError, match="1D"):
        profile_moe_layers_and_experts([[0.1]], [0.1], [0.1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 4 samples"):
        compute_oof_orthogonal_router_direction(np.ones((2, 4)), {"s": np.ones((4, 4))})
    with pytest.raises(ValueError, match="non-empty"):
        compute_oof_orthogonal_router_direction(np.ones((6, 4)), {})
    with pytest.raises(ValueError, match="Invalid shape"):
        compute_oof_orthogonal_router_direction(np.ones((6, 4)), {"s": np.ones((6, 5))})

    # Degenerate zero-variance projector & QR-SVD shape validation
    p_eye = extract_mp_guarded_prose_projector(np.zeros((4, 6)))
    assert np.allclose(p_eye, np.eye(6))
    with pytest.raises(ValueError, match="at least 2 rows"):
        extract_mp_guarded_prose_projector(np.ones((1, 4)))
    with pytest.raises(ValueError, match="Invalid"):
        project_lora_pair_qr_svd_nullspace(np.ones((4, 2)), np.ones((3, 5)), np.eye(5))
    with pytest.raises(ValueError, match="p_null shape"):
        project_lora_pair_qr_svd_nullspace(np.ones((4, 2)), np.ones((2, 5)), np.eye(4))
    with pytest.raises(ValueError, match="final_ln_w shapes"):
        apply_stage_s5_4_folded_final_norm_calibration_inplace(
            {"final_ln_w": np.ones((4,))},
            np.ones((3,)),
            np.ones((4,)),
            lambda _: {DEFAULT_GENERAL_DOMAIN: 1.0},
            {DEFAULT_GENERAL_DOMAIN: 1.0},
        )
    assert (
        apply_stage_s5_4_folded_final_norm_calibration_inplace(
            {"other": np.ones((2,))},
            np.ones((2,)),
            np.ones((2,)),
            lambda _: {DEFAULT_GENERAL_DOMAIN: 1.0},
            {DEFAULT_GENERAL_DOMAIN: 1.0},
        )
        == (0, 1.0, 1.0)
    )

    with pytest.raises(FileNotFoundError):
        cli.load_checkpoint_arrays(tmp_path / "missing.npz")
    bad_ext = tmp_path / "bad.bin"
    bad_ext.write_bytes(b"123")
    with pytest.raises(ValueError, match="Unsupported checkpoint suffix"):
        cli.load_checkpoint_arrays(bad_ext)
    with pytest.raises(ValueError, match="Unsupported output suffix"):
        cli.save_checkpoint_arrays(bad_ext, {"w": np.ones((2,), dtype=np.float32)})
    with pytest.raises(ValueError, match="lambda_ln"):
        cli.run_offline_s5_checkpoint_surgery({"w": np.ones((2,))}, {}, lambda_ln=1.5)
    with pytest.raises(ValueError, match="tau_temp"):
        cli.run_offline_s5_checkpoint_surgery({"w": np.ones((2,))}, {}, tau_temp=0.0)

    # docs/AUTO_OPEN_WEIGHT_CAPABILITY_PLAN.md Season 35 specification verification
    assert DOC_PATH.exists()
    doc_text = DOC_PATH.read_text(encoding="utf-8")
    assert "Season 35" in doc_text
    for task_id in ("HM-S35-01", "HM-S35-02", "HM-S35-03", "HM-S35-04", "HM-S35-05", "HM-S35-06", "HM-S35-07", "HM-S35-08"):
        assert task_id in doc_text, f"Missing {task_id} in docs/AUTO_OPEN_WEIGHT_CAPABILITY_PLAN.md"

    # AGPL-3.0 header, ZeroStubAudit, Lateral-Movement AST, and PRSecurityScanner gates
    auditor = ZeroStubAudit()
    from hivemind.security.pr_security_scanner import PRSecurityScanner
    from tests.test_no_lateral_movement import audit_source_for_lateral_movement

    scanner = PRSecurityScanner(trusted_maintainers={"xaviercallens"})
    for target_file in (ENGINE_PATH, CLI_PATH, Path(__file__).resolve()):
        src = target_file.read_text(encoding="utf-8")
        assert "Licensed under the AGPL-3.0." in src[:300]
        if target_file != Path(__file__).resolve():
            assert auditor.audit_python_code(src).is_clean
            assert audit_source_for_lateral_movement(src, str(target_file)) == []

    rel_files = [
        "src/hivemind/swarm/auto_open_weight_engine.py",
        "scripts/hivemind_auto_open_weight.py",
        "docs/AUTO_OPEN_WEIGHT_CAPABILITY_PLAN.md",
        "tests/test_auto_open_weight_engine_scenarios.py",
    ]
    changed_payloads = [
        {
            "filename": rf,
            "patch": "\n".join("+" + ln for ln in (REPO_ROOT / rf).read_text(encoding="utf-8").splitlines()[:80]),
        }
        for rf in rel_files[:-1]
    ]
    sec_report = scanner.audit_pull_request(
        pr_number=1008,
        pr_title="feat(swarm): Season 35 Auto Open-Weight S5 PASS-Synergy Engine, CLI & Capability Spec",
        pr_body="Implements HM-S35-02 S5 PASS-Synergy engine, CLI, and Season 35 capability spec.",
        author="xaviercallens",
        changed_files=changed_payloads,
        author_email=None,
    )
    assert len(sec_report.findings) == 0

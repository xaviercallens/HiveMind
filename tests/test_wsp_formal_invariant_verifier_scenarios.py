# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-22: HM-S35-08 S5 WSP formal invariant verifier."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.auto_open_weight_engine import (
    ANSE_MAXIMUM_PAIN,
    DEFAULT_GENERAL_DOMAIN,
    AutoOpenWeightConfig,
    clone_state_dict,
    compute_oof_orthogonal_router_direction,
    extract_mp_guarded_prose_projector,
    run_auto_open_weight_s5_synergy,
)
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import upcycle_crowd_v1_to_sparse_moe
from hivemind.swarm.wsp_formal_invariant_verifier import (
    CERTIFICATE_OBJECT,
    OBLIGATION_NAMES,
    SCHEMA_VERSION,
    WSPFormalCertificate,
    canonical_json_bytes,
    certify_s5_surgery_bundle,
    compute_payload_sha256,
    compute_state_sha256,
    verify_anchor_layer_bitwise_immutability,
    verify_monotone_anse_stage_descent,
    verify_orthogonal_multigate_centroid_invariance,
    verify_preserved_key_invariance,
    verify_projector_idempotence_and_symmetry,
    verify_wsp_formal_certificate_provenance,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackStubResult:
    """CI-safe stub audit result when optional ``gwaya`` is absent."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = tuple(violations)
        self.is_clean = len(violations) == 0
        self.penalty_energy = 0.0 if self.is_clean else 1e6


class _FallbackZeroStubAudit:
    """CI-safe fallback implementing both ``check_file`` and ``audit_python_code``."""

    def check_file(self, path: str | Path) -> list[str]:
        text = Path(path).read_text(encoding="utf-8")
        return list(self.audit_python_code(text).violations)

    def audit_python_code(self, code: str) -> _FallbackStubResult:
        tree = ast.parse(code)
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stmts = [
                    s
                    for s in node.body
                    if not (
                        isinstance(s, ast.Expr)
                        and isinstance(s.value, ast.Constant)
                        and isinstance(s.value.value, str)
                    )
                ]
                if not stmts or (len(stmts) == 1 and isinstance(stmts[0], ast.Pass)):
                    violations.append(f"stub function: {node.name}")
                if (
                    len(stmts) == 1
                    and isinstance(stmts[0], ast.Raise)
                    and isinstance(stmts[0].exc, ast.Call)
                    and isinstance(stmts[0].exc.func, ast.Name)
                    and stmts[0].exc.func.id == "NotImplementedError"
                ):
                    violations.append(f"NotImplementedError in {node.name}")
        return _FallbackStubResult(violations)


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = (
    REPO_ROOT / "src" / "hivemind" / "swarm" / "wsp_formal_invariant_verifier.py"
)
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _build_s5_moe_testbed(seed: int = 20261124) -> dict[str, Any]:
    """Construct a deterministic 4-layer 6-expert sigma_topk MoE world for S5 certification."""
    rng = np.random.default_rng(seed)
    n_layers = 4
    n_experts = 6
    d_model = 16
    d_mid = 12
    rank = 4
    n_tok = 28

    domains = [
        DEFAULT_GENERAL_DOMAIN,
        "E1_python_systems_code",
        "E2_math_reasoning",
        "E5_exoplanet_transit",
    ]
    spec_domains = domains[1:]

    q_model, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
    q_mid, _ = np.linalg.qr(rng.standard_normal((d_mid, d_mid)))

    spec_dirs = {dom: q_model[:, idx] for idx, dom in enumerate(spec_domains)}
    prose_orth_dir = q_model[:, 4]
    prose_center = 1.6 * prose_orth_dir + 0.50 * spec_dirs["E5_exoplanet_transit"]

    prose_mid_subspace = q_mid[:, :3]
    spec_mid_dirs = {dom: q_mid[:, 3 + idx] for idx, dom in enumerate(spec_domains)}

    domain_x: dict[str, np.ndarray] = {
        DEFAULT_GENERAL_DOMAIN: prose_center[None, :] + 0.06 * rng.standard_normal((n_tok, d_model))
    }
    prose_mid_mean = np.sum(prose_mid_subspace, axis=1)
    domain_mid: dict[str, np.ndarray] = {
        DEFAULT_GENERAL_DOMAIN: (
            1.4 * prose_mid_mean[None, :]
            + 0.20 * (rng.standard_normal((n_tok, 3)) @ prose_mid_subspace.T)
            + 0.02 * rng.standard_normal((n_tok, d_mid))
        )
    }
    domain_target: dict[str, np.ndarray] = {DEFAULT_GENERAL_DOMAIN: q_model[:, 6]}

    for idx, dom in enumerate(spec_domains):
        domain_x[dom] = 1.8 * spec_dirs[dom][None, :] + 0.06 * rng.standard_normal((n_tok, d_model))
        domain_mid[dom] = (
            1.5 * spec_mid_dirs[dom][None, :]
            + 0.20 * (rng.standard_normal((n_tok, 3)) @ prose_mid_subspace.T)
            + 0.02 * rng.standard_normal((n_tok, d_mid))
        )
        domain_target[dom] = q_model[:, 7 + idx]

    state: dict[str, np.ndarray] = {}
    anchor_layer = 1
    interfering_layers = {2, 3}
    spec_expert_map = {
        2: "E1_python_systems_code",
        3: "E5_exoplanet_transit",
        4: "E2_math_reasoning",
    }

    for l_idx in range(n_layers):
        rw = np.zeros((n_experts, d_model), dtype=np.float32)
        w2_b = np.zeros((n_experts, d_model, rank), dtype=np.float32)
        w2_a = np.zeros((n_experts, rank, d_mid), dtype=np.float32)

        if l_idx == anchor_layer:
            for sink_e in (0, 1):
                rw[sink_e] = (2.0 * prose_orth_dir).astype(np.float32)
                w2_b[sink_e, :, 0] = (0.40 * domain_target[DEFAULT_GENERAL_DOMAIN]).astype(np.float32)
                w2_a[sink_e, 0, :] = prose_mid_subspace[:, sink_e].astype(np.float32)
            for exp_e, dom in spec_expert_map.items():
                rw[exp_e] = (2.2 * spec_dirs[dom]).astype(np.float32)
                w2_b[exp_e, :, 0] = (0.35 * domain_target[dom]).astype(np.float32)
                w2_a[exp_e, 0, :] = spec_mid_dirs[dom].astype(np.float32)
        elif l_idx in interfering_layers:
            for sink_e in (0, 1):
                rw[sink_e] = (2.0 * prose_orth_dir).astype(np.float32)
                w2_b[sink_e, :, 0] = (-0.42 * domain_target[DEFAULT_GENERAL_DOMAIN]).astype(np.float32)
                w2_a[sink_e, 0, :] = prose_mid_subspace[:, sink_e].astype(np.float32)

            rw[3] = (2.2 * spec_dirs["E5_exoplanet_transit"] + 1.2 * prose_orth_dir).astype(np.float32)
            w2_b[3, :, 0] = (
                0.52 * domain_target["E5_exoplanet_transit"]
                - 0.16 * domain_target[DEFAULT_GENERAL_DOMAIN]
            ).astype(np.float32)
            w2_a[3, 0, :] = (
                spec_mid_dirs["E5_exoplanet_transit"] + 0.35 * prose_mid_subspace[:, 0]
            ).astype(np.float32)

            for exp_e in (2, 4):
                dom = spec_expert_map[exp_e]
                rw[exp_e] = (2.2 * spec_dirs[dom] + 0.25 * prose_orth_dir).astype(np.float32)
                w2_b[exp_e, :, 0] = (
                    0.44 * domain_target[dom] - 0.05 * domain_target[DEFAULT_GENERAL_DOMAIN]
                ).astype(np.float32)
                w2_a[exp_e, 0, :] = (spec_mid_dirs[dom] + 0.30 * prose_mid_subspace[:, 2]).astype(
                    np.float32
                )

        state[f"layer_{l_idx}.router_w"] = rw
        state[f"layer_{l_idx}.w2_B"] = w2_b
        state[f"layer_{l_idx}.w2_A"] = w2_a

    base_ln_w = np.ones((d_model,), dtype=np.float32)
    base_ln_b = np.zeros((d_model,), dtype=np.float32)
    state["final_ln_w"] = (1.18 * np.ones((d_model,), dtype=np.float32)).astype(np.float32)
    state["final_ln_b"] = (-0.04 * domain_target[DEFAULT_GENERAL_DOMAIN]).astype(np.float32)

    base_nll = {dom: 4.10 for dom in domains}

    def eval_moe(st: dict[str, np.ndarray] | Any) -> dict[str, float]:
        out: dict[str, float] = {}
        ln_w = np.asarray(st["final_ln_w"], dtype=np.float64)
        ln_b = np.asarray(st["final_ln_b"], dtype=np.float64)
        cal_pen = 0.22 * ((float(np.mean(ln_w)) - 1.02) ** 2)
        for dom in domains:
            x = domain_x[dom]
            mid = domain_mid[dom]
            tgt = domain_target[dom]
            residual = np.zeros_like(x, dtype=np.float64)
            for l_idx in range(n_layers):
                rw = np.asarray(st[f"layer_{l_idx}.router_w"], dtype=np.float64)
                wb = np.asarray(st[f"layer_{l_idx}.w2_B"], dtype=np.float64)
                wa = np.asarray(st[f"layer_{l_idx}.w2_A"], dtype=np.float64)
                gates = _sigmoid(x @ rw.T)
                top_idx = np.argsort(gates, axis=1)[:, -3:]
                mask = np.zeros_like(gates)
                np.put_along_axis(mask, top_idx, 1.0, axis=1)
                active_g = gates * mask
                for e_id in range(n_experts):
                    delta_mat = wb[e_id] @ wa[e_id]
                    exp_out = mid @ delta_mat.T
                    residual = residual + active_g[:, e_id : e_id + 1] * exp_out
            normed = residual * ln_w[None, :] + ln_b[None, :]
            align = float(np.mean(normed @ tgt))
            quad_reg = 0.03 * float(np.mean(np.sum(normed * normed, axis=1)))
            out[dom] = float(base_nll[dom] - 0.55 * align + quad_reg + cal_pen)
        return out

    c2_raw_nll = eval_moe(state)
    layer_e0_harm: list[float] = []
    for l_idx in range(n_layers):
        bypass = clone_state_dict(state)
        bypass[f"layer_{l_idx}.w2_B"] = np.zeros_like(bypass[f"layer_{l_idx}.w2_B"])
        layer_e0_harm.append(
            float(c2_raw_nll[DEFAULT_GENERAL_DOMAIN] - eval_moe(bypass)[DEFAULT_GENERAL_DOMAIN])
        )

    expert_e0_harm: list[float] = []
    expert_spec_loss: list[float] = []
    for e_id in range(n_experts):
        zero_e = clone_state_dict(state)
        for l_idx in interfering_layers:
            zero_e[f"layer_{l_idx}.w2_B"][e_id] = 0.0
        z_nll = eval_moe(zero_e)
        expert_e0_harm.append(
            float(c2_raw_nll[DEFAULT_GENERAL_DOMAIN] - z_nll[DEFAULT_GENERAL_DOMAIN])
        )
        expert_spec_loss.append(
            float(np.mean([z_nll[d] - c2_raw_nll[d] for d in spec_domains]))
        )

    layer_prose_inputs = {l_idx: domain_x[DEFAULT_GENERAL_DOMAIN] for l_idx in range(n_layers)}
    layer_spec_inputs = {
        l_idx: {dom: domain_x[dom] for dom in spec_domains} for l_idx in range(n_layers)
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
        "domain_x": domain_x,
        "domain_mid": domain_mid,
        "spec_domains": spec_domains,
    }


def _build_valid_torch_bundle_inputs(seed: int = 20261124) -> dict[str, Any]:
    """Build deterministic CPU torch tensors satisfying all 5 S5 proof obligations."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d_in = 16
    d_out = 12
    q_mat, _ = torch.linalg.qr(torch.randn((d_in, d_in), generator=gen, dtype=torch.float64))

    # Preserved keys K_0 live in span(q_mat[:, :4]); P_0 = I - Q_0 Q_0^T projects to orthogonal space
    q_0 = q_mat[:, :4]
    p_null = torch.eye(d_in, dtype=torch.float64) - (q_0 @ q_0.T)
    k_preserved = q_0 @ torch.randn((4, 10), generator=gen, dtype=torch.float64)
    raw_update = torch.randn((d_out, d_in), generator=gen, dtype=torch.float64)
    delta_w = raw_update @ p_null

    # Orthogonal multi-gate vector u_gate: orthogonal to 3 specialist centroids, positive on prose
    spec_centroids = {
        "E1_code": (2.0 * q_mat[:, 4] + 0.3 * q_mat[:, 5]).to(torch.float64),
        "E2_math": (1.8 * q_mat[:, 5] - 0.2 * q_mat[:, 6]).to(torch.float64),
        "E5_exoplanet": (2.2 * q_mat[:, 6] + 0.1 * q_mat[:, 4]).to(torch.float64),
    }
    prose_centroid = (1.5 * q_mat[:, 8] + 0.4 * q_mat[:, 6]).to(torch.float64)
    u_gate = q_mat[:, 8].to(torch.float64)

    cfg = CrowdV1RuntimeConfig(
        vocab_size=32,
        hidden_size=d_in,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=24,
        max_position_embeddings=32,
        engram_buckets=0,
        tie_word_embeddings=True,
    )
    model_pre = CrowdV1WithEngram(cfg)
    upcycle_crowd_v1_to_sparse_moe(
        model_pre,
        split_mode="lora_domain_experts",
        num_experts=4,
        top_k=2,
        lora_rank=2,
        lora_alpha=4.0,
        routing_mode="sigma_topk",
        seed=seed + 7,
    )
    post_state = {k: v.detach().clone() for k, v in model_pre.state_dict().items()}
    with torch.no_grad():
        post_state["blocks.1.mlp.router.weight"] = (
            post_state["blocks.1.mlp.router.weight"] - 0.25 * u_gate.to(torch.float32).unsqueeze(0)
        )

    stage_records = [
        {
            "stage_name": "S5.1_anchor_sink",
            "accepted": True,
            "pre_anse_energy": 0.15,
            "post_anse_energy": -0.18,
            "delta_anse_energy": -0.33,
            "general_rel_drift": 0.004,
            "pre_sha256": "a" * 64,
            "post_sha256": "b" * 64,
        },
        {
            "stage_name": "S5.2_oof_multigate",
            "accepted": True,
            "pre_anse_energy": -0.18,
            "post_anse_energy": -0.42,
            "delta_anse_energy": -0.24,
            "general_rel_drift": 0.001,
            "pre_sha256": "b" * 64,
            "post_sha256": "c" * 64,
        },
    ]

    return {
        "p_null": p_null,
        "delta_w": delta_w,
        "k_preserved": k_preserved,
        "u_gate": u_gate,
        "spec_centroids": spec_centroids,
        "prose_centroid": prose_centroid,
        "model_pre": model_pre,
        "pre_state": {k: v.detach().clone() for k, v in model_pre.state_dict().items()},
        "post_state": post_state,
        "anchor_prefixes": ("blocks.0.",),
        "stage_records": stage_records,
    }


def test_s1_end_to_end_s5_pass_synergy_certified_sha256_receipt() -> None:
    """S1: End-to-end S5 PASS-Synergy edit passes all 5 algebraic obligations with SHA-256 receipt."""
    world = _build_s5_moe_testbed(seed=20261124)
    c2_state = world["state"]
    cfg = AutoOpenWeightConfig(router_beta=1.10, svd_damp_strength=0.90)
    s5_state, s5_res = run_auto_open_weight_s5_synergy(
        c2_state,
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
    assert s5_res.committed is True
    assert s5_res.verdict == "PASS"

    tail_layers = list(s5_res.profile.tail_svd_layers)
    assert len(tail_layers) >= 1
    p_null_map: dict[str, np.ndarray] = {}
    delta_w_map: dict[str, np.ndarray] = {}
    k_pres_map: dict[str, np.ndarray] = {}
    u_gate_map: dict[str, np.ndarray] = {}

    rng = np.random.default_rng(777)
    for l_idx in tail_layers:
        p_0 = extract_mp_guarded_prose_projector(
            world["tail_prose_mid"][l_idx],
            energy_keep=cfg.prose_nullspace_energy_keep,
        )
        p_null_map[f"layer_{l_idx}"] = p_0
        # Preserved prose key subspace U_0 lies in ker(P_0); construct K_0 from eigvecs with eigenvalue ~0
        eigvals, eigvecs = np.linalg.eigh(p_0)
        u_0 = eigvecs[:, eigvals < 0.5]
        assert u_0.shape[1] >= 1
        k_0 = u_0 @ rng.standard_normal((u_0.shape[1], 6))
        delta_w_map[f"layer_{l_idx}"] = rng.standard_normal((16, p_0.shape[0])) @ p_0
        k_pres_map[f"layer_{l_idx}"] = k_0

    for l_idx in s5_res.profile.interfering_layers:
        u_dir, snr = compute_oof_orthogonal_router_direction(
            world["layer_prose_inputs"][l_idx],
            world["layer_spec_inputs"][l_idx],
            min_oof_snr=cfg.min_oof_snr,
        )
        assert u_dir is not None and snr >= cfg.min_oof_snr
        u_gate_map[f"layer_{l_idx}"] = u_dir

    spec_centroids = {
        dom: np.mean(acts, axis=0) for dom, acts in world["layer_spec_inputs"][2].items()
    }
    prose_centroid = np.mean(world["layer_prose_inputs"][2], axis=0)

    cert = certify_s5_surgery_bundle(
        p_null=p_null_map,
        delta_w=delta_w_map,
        k_preserved=k_pres_map,
        u_gate=u_gate_map,
        specialist_centroids=spec_centroids,
        prose_centroid=prose_centroid,
        pre_state=c2_state,
        post_state=s5_state,
        anchor_layer_prefixes=("layer_1.",),
        stage_records=s5_res,
        surgery_id="hm_s35_08_pilot2_s5_pass_synergy",
        metadata={
            "tags": {"s5", "cpu"},
            "tail_layers": tail_layers,
            "n_experts": np.int64(6),
            "tol_ref": np.float64(1e-5),
        },
    )

    assert isinstance(cert, WSPFormalCertificate)
    assert cert.schema == SCHEMA_VERSION
    assert cert.object == CERTIFICATE_OBJECT
    assert cert.certified is True
    assert cert.passed_obligations == 5
    assert cert.total_obligations == 5
    assert cert.failed_obligations == ()
    assert tuple(cert.obligations.keys()) == OBLIGATION_NAMES
    assert cert.anchor_mutated_bits == 0
    assert cert.anchor_pre_sha256 == cert.anchor_post_sha256
    assert cert.energy_delta < 0.0
    assert cert.anse_energy < 0.0
    assert cert.max_residual_norm <= 1e-5
    assert len(cert.receipt_sha256) == 64

    assert verify_wsp_formal_certificate_provenance(cert) is True
    assert len(canonical_json_bytes(cert.obligations)) > 10
    cert_dict = cert.to_dict()
    assert verify_wsp_formal_certificate_provenance(cert_dict) is True
    roundtrip = json.loads(canonical_json_bytes(cert_dict).decode("utf-8"))
    assert verify_wsp_formal_certificate_provenance(roundtrip) is True

    # Immutability guarantees on certificate and nested mappings
    with pytest.raises(AttributeError):
        cert.certified = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        cert.obligations["extra"] = cert.obligations["preserved_key_invariance"]  # type: ignore[index]
    with pytest.raises(TypeError):
        cert.metadata["tamper"] = 1  # type: ignore[index]

    # Provenance tampering detection
    for tamper_fn in (
        lambda d: d.update({"certified": False}),
        lambda d: d.update({"anse_energy": 0.99}),
        lambda d: d.update({"receipt_sha256": "bad_hex"}),
        lambda d: d.update({"schema": "wrong_schema"}),
        lambda d: d.update({"total_obligations": 99}),
        lambda d: d.update({"passed_obligations": 4}),
        lambda d: d.update({"failed_obligations": ["x"]}),
        lambda d: d.update({"failed_obligations": 123}),
        lambda d: d["obligations"]["preserved_key_invariance"].update({"passed": False}),
        lambda d: d["obligations"]["preserved_key_invariance"].update({"object": "wrong"}),
        lambda d: d["obligations"]["preserved_key_invariance"].update({"receipt_sha256": "0" * 64}),
        lambda d: d["obligations"]["preserved_key_invariance"].update({"receipt_sha256": "short"}),
        lambda d: d["obligations"].pop("monotone_anse_stage_descent"),
        lambda d: d["obligations"].update({"monotone_anse_stage_descent": "not_a_dict"}),
        lambda d: d["obligations"]["preserved_key_invariance"].update({"max_residual_norm": float("nan")}),
        lambda d: d.update({"max_residual_norm": float("nan")}),
    ):
        tampered = copy.deepcopy(cert_dict)
        tamper_fn(tampered)
        assert verify_wsp_formal_certificate_provenance(tampered) is False

    # Also certify a single-tensor CrowdV1WithEngram PyTorch module bundle (passing nn.Module directly)
    # and verify shared k_preserved across a Mapping of delta_w tensors + stage_receipts container
    torch_bundle = _build_valid_torch_bundle_inputs()
    torch_cert = certify_s5_surgery_bundle(
        torch_bundle["p_null"],
        {"tail_a": torch_bundle["delta_w"]},
        torch_bundle["k_preserved"],
        torch_bundle["u_gate"],
        torch_bundle["spec_centroids"],
        torch_bundle["prose_centroid"],
        torch_bundle["model_pre"],
        torch_bundle["post_state"],
        torch_bundle["anchor_prefixes"],
        torch_bundle["stage_records"],
        surgery_id="crowd_v1_torch_s5",
    )
    assert torch_cert.certified is True
    assert torch_cert.anchor_mutated_bits == 0
    assert verify_wsp_formal_certificate_provenance(torch_cert) is True


def test_s2_non_idempotent_and_asymmetric_projector_rejection() -> None:
    """S2: Non-idempotent, asymmetric, and out-of-spectrum projectors are strictly rejected."""
    # Valid edge projectors: I, 0, and rank-2 orthogonal complement
    r_eye = verify_projector_idempotence_and_symmetry(torch.eye(6, dtype=torch.float64))
    r_zero = verify_projector_idempotence_and_symmetry(torch.zeros((6, 6), dtype=torch.float64))
    assert r_eye.passed is True and r_eye.metrics["projected_rank"] == 6
    assert r_zero.passed is True and r_zero.metrics["projected_rank"] == 0

    # Non-idempotent symmetric matrix: 0.5 * I (symmetric, eigvals=0.5 in [0, 1], but P^2 != P)
    p_non_idem = 0.5 * torch.eye(4, dtype=torch.float64)
    r_non_idem = verify_projector_idempotence_and_symmetry(p_non_idem, tol=1e-6)
    assert r_non_idem.passed is False
    assert r_non_idem.metrics["idempotence_ok"] is False
    assert r_non_idem.metrics["symmetry_ok"] is True
    assert r_non_idem.metrics["spectrum_ok"] is True
    assert r_non_idem.max_residual_norm == pytest.approx(0.5)

    # Asymmetric oblique projector: P = [[1, 0.8], [0, 0]] satisfies P^2 == P but fails symmetry
    p_asym = torch.tensor([[1.0, 0.8], [0.0, 0.0]], dtype=torch.float64)
    r_asym = verify_projector_idempotence_and_symmetry(p_asym, tol=1e-6)
    assert r_asym.passed is False
    assert r_asym.metrics["idempotence_ok"] is True
    assert r_asym.metrics["symmetry_ok"] is False
    assert r_asym.metrics["symmetry_residual"] == pytest.approx(math.sqrt(2.0) * 0.8)

    # Out-of-bounds eigenvalue spectrum: > 1 and < 0
    r_over = verify_projector_idempotence_and_symmetry(torch.diag(torch.tensor([1.25, 0.0])))
    r_under = verify_projector_idempotence_and_symmetry(torch.diag(torch.tensor([-0.25, 1.0])))
    assert r_over.passed is False and r_over.metrics["spectrum_ok"] is False
    assert r_under.passed is False and r_under.metrics["spectrum_ok"] is False

    # Bundle rejection when non-idempotent / asymmetric projector is supplied
    tb = _build_valid_torch_bundle_inputs()
    bad_cert_idem = certify_s5_surgery_bundle(
        0.5 * torch.eye(16, dtype=torch.float64),
        tb["delta_w"],
        tb["k_preserved"],
        tb["u_gate"],
        tb["spec_centroids"],
        tb["prose_centroid"],
        tb["pre_state"],
        tb["post_state"],
        tb["anchor_prefixes"],
        tb["stage_records"],
    )
    assert bad_cert_idem.certified is False
    assert bad_cert_idem.passed_obligations == 4
    assert bad_cert_idem.failed_obligations == ("projector_idempotence_and_symmetry",)
    assert verify_wsp_formal_certificate_provenance(bad_cert_idem) is True

    bad_cert_asym = certify_s5_surgery_bundle(
        {"tail_a": torch.tensor([[1.0, 0.5], [0.0, 0.0]], dtype=torch.float64)},
        tb["delta_w"],
        tb["k_preserved"],
        tb["u_gate"],
        tb["spec_centroids"],
        tb["prose_centroid"],
        tb["pre_state"],
        tb["post_state"],
        tb["anchor_prefixes"],
        tb["stage_records"],
    )
    assert bad_cert_asym.certified is False
    assert "projector_idempotence_and_symmetry" in bad_cert_asym.failed_obligations


def test_s3_preserved_key_leakage_and_specialist_centroid_interference_detection() -> None:
    """S3: Preserved-key leakage and specialist centroid / prose-sign interference are caught."""
    tb = _build_valid_torch_bundle_inputs()

    # Valid preserved-key invariance + zero-key floor check
    r_key_ok = verify_preserved_key_invariance(tb["delta_w"], tb["k_preserved"], tol=1e-5)
    assert r_key_ok.passed is True
    assert r_key_ok.max_residual_norm < 1e-12
    r_key_zero = verify_preserved_key_invariance(
        tb["delta_w"], torch.zeros((16, 4), dtype=torch.float64), tol=1e-5
    )
    assert r_key_zero.passed is True and r_key_zero.metrics["denominator"] == 1e-8

    # Inject small key leakage along preserved keys
    k0_unit = tb["k_preserved"][:, 0] / tb["k_preserved"][:, 0].norm()
    leaked_dw = tb["delta_w"] + 2e-3 * torch.outer(
        torch.ones(12, dtype=torch.float64), k0_unit
    )
    r_key_bad = verify_preserved_key_invariance(leaked_dw, tb["k_preserved"], tol=1e-5)
    assert r_key_bad.passed is False
    assert r_key_bad.max_residual_norm > 1e-5

    # Valid orthogonal multigate across Mapping, Sequence, and 2D Tensor representations
    r_gate_map = verify_orthogonal_multigate_centroid_invariance(
        tb["u_gate"], tb["spec_centroids"], tb["prose_centroid"], tol=1e-5
    )
    r_gate_seq = verify_orthogonal_multigate_centroid_invariance(
        tb["u_gate"], list(tb["spec_centroids"].values()), tb["prose_centroid"], tol=1e-5
    )
    r_gate_mat = verify_orthogonal_multigate_centroid_invariance(
        tb["u_gate"], torch.stack(list(tb["spec_centroids"].values()), dim=0), tb["prose_centroid"], tol=1e-5
    )
    assert r_gate_map.passed is True
    assert r_gate_seq.passed is True
    assert r_gate_mat.passed is True

    # Specialist centroid interference detection
    interfering_u = tb["u_gate"] + 1e-3 * tb["spec_centroids"]["E1_code"]
    r_gate_leak = verify_orthogonal_multigate_centroid_invariance(
        interfering_u, tb["spec_centroids"], tb["prose_centroid"], tol=1e-5
    )
    assert r_gate_leak.passed is False
    assert r_gate_leak.metrics["centroids_ok"] is False
    assert "E1_code" in r_gate_leak.metrics["violating_centroids"]

    # Reversed / zero / insufficient prose alignment detection
    r_gate_rev = verify_orthogonal_multigate_centroid_invariance(
        -tb["u_gate"], tb["spec_centroids"], tb["prose_centroid"], tol=1e-5
    )
    assert r_gate_rev.passed is False and r_gate_rev.metrics["prose_alignment_ok"] is False
    r_gate_zero = verify_orthogonal_multigate_centroid_invariance(
        torch.zeros_like(tb["u_gate"]), tb["spec_centroids"], tb["prose_centroid"], tol=1e-5
    )
    assert r_gate_zero.passed is False and r_gate_zero.metrics["prose_alignment_ok"] is False
    r_gate_floor = verify_orthogonal_multigate_centroid_invariance(
        tb["u_gate"],
        tb["spec_centroids"],
        tb["prose_centroid"],
        tol=1e-5,
        min_prose_alignment=10.0,
    )
    assert r_gate_floor.passed is False and r_gate_floor.metrics["prose_alignment_ok"] is False

    # Bundle rejection on key leakage + centroid interference simultaneously
    bad_cert = certify_s5_surgery_bundle(
        tb["p_null"],
        {"e1": leaked_dw},
        {"e1": tb["k_preserved"]},
        {"e1": interfering_u},
        tb["spec_centroids"],
        tb["prose_centroid"],
        tb["pre_state"],
        tb["post_state"],
        tb["anchor_prefixes"],
        tb["stage_records"],
    )
    assert bad_cert.certified is False
    assert bad_cert.passed_obligations == 3
    assert bad_cert.failed_obligations == (
        "preserved_key_invariance",
        "orthogonal_multigate_centroid_invariance",
    )


def test_s4_anchor_single_ulp_bitflip_tamper_and_non_monotone_anse_rejection() -> None:
    """S4: Single-ULP / sign-bit anchor tamper and non-monotone ANSE stages are rejected."""
    tb = _build_valid_torch_bundle_inputs()
    anchor_key = "blocks.0.mlp.router.weight"

    # 1a. Single-ULP bit flip in float32
    post_ulp_f32 = {k: v.clone() for k, v in tb["post_state"].items()}
    orig_val = float(post_ulp_f32[anchor_key][0, 0].item())
    post_ulp_f32[anchor_key][0, 0] = float(np.nextafter(np.float32(orig_val), np.float32(orig_val + 10.0)))
    r_ulp = verify_anchor_layer_bitwise_immutability(
        tb["pre_state"], post_ulp_f32, tb["anchor_prefixes"]
    )
    assert r_ulp.passed is False
    assert r_ulp.metrics["mutated_bits"] >= 1
    assert r_ulp.metrics["mutated_elements"] == 1
    assert anchor_key in r_ulp.metrics["mutated_keys"]

    # 1b. Exact 1-bit XOR ULP tamper on bfloat16 & +0.0 -> -0.0 sign-bit flip
    pre_bf16 = {"blocks.0.w": torch.tensor([1.0, 0.0], dtype=torch.bfloat16), "blocks.1.w": torch.ones(2)}
    post_bf16 = {"blocks.0.w": torch.tensor([1.0, 0.0], dtype=torch.bfloat16), "blocks.1.w": torch.zeros(2)}
    u16_view = post_bf16["blocks.0.w"].view(torch.uint16)
    u16_view[0] = u16_view[0] ^ 1  # flip 1 LSB ULP on element 0
    r_bf16 = verify_anchor_layer_bitwise_immutability(pre_bf16, post_bf16, ("blocks.0.",))
    assert r_bf16.passed is False
    assert r_bf16.metrics["mutated_bits"] == 1
    assert r_bf16.metrics["mutated_elements"] == 1

    # Sign-bit flip on zero (+0.0 -> -0.0): torch.equal says True, bitwise verifier catches 1-bit flip!
    pre_zero = {"blocks.0.z": torch.tensor([0.0, 2.0], dtype=torch.float32)}
    post_neg_zero = {"blocks.0.z": torch.tensor([-0.0, 2.0], dtype=torch.float32)}
    assert torch.equal(pre_zero["blocks.0.z"], post_neg_zero["blocks.0.z"]) is True
    r_sign_zero = verify_anchor_layer_bitwise_immutability(pre_zero, post_neg_zero, ("blocks.0.",))
    assert r_sign_zero.passed is False
    assert r_sign_zero.metrics["mutated_bits"] == 1
    assert r_sign_zero.metrics["mutated_elements"] == 1

    # 1c. Missing anchor key, extra anchor key, shape/dtype change, non-finite post anchor, empty tensor
    pre_extra = {"blocks.0.a": torch.ones(2, dtype=torch.float32), "blocks.0.empty": torch.empty(0)}
    post_missing = {"blocks.0.empty": torch.empty(0), "blocks.0.b": torch.ones(2, dtype=torch.float32)}
    r_struct = verify_anchor_layer_bitwise_immutability(pre_extra, post_missing, ("blocks.0.",))
    assert r_struct.passed is False
    assert r_struct.metrics["missing_post_keys"] == ("blocks.0.a",)
    assert r_struct.metrics["extra_post_keys"] == ("blocks.0.b",)

    r_shape = verify_anchor_layer_bitwise_immutability(
        {"blocks.0.a": torch.ones(2, dtype=torch.float32)},
        {"blocks.0.a": torch.ones(3, dtype=torch.float32)},
        ("blocks.0.",),
    )
    assert r_shape.passed is False

    r_nan_post = verify_anchor_layer_bitwise_immutability(
        {"blocks.0.a": np.ones(2, dtype=np.float32)},
        {"blocks.0.a": np.array([1.0, float("nan")], dtype=np.float32)},
        ("blocks.0.",),
    )
    assert r_nan_post.passed is False
    assert r_nan_post.metrics["non_finite_post_keys"] == ("blocks.0.a",)

    # NumPy single-ULP tamper + extra non-finite anchor key
    pre_np_ulp = {"blocks.0.np": np.ones((2, 2), dtype=np.float64)}
    post_np_ulp = {"blocks.0.np": np.ones((2, 2), dtype=np.float64)}
    post_np_ulp["blocks.0.np"][0, 1] = float(np.nextafter(1.0, 2.0))
    post_np_ulp["blocks.0.extra_nan"] = np.array([float("nan")], dtype=np.float32)
    r_np_ulp = verify_anchor_layer_bitwise_immutability(pre_np_ulp, post_np_ulp, ("blocks.0.",))
    assert r_np_ulp.passed is False
    assert r_np_ulp.metrics["mutated_bits"] >= 1
    assert "blocks.0.extra_nan" in r_np_ulp.metrics["non_finite_post_keys"]

    # 2. Non-monotone ANSE stage rejection across trajectory, stage_receipts container, & receipt formats
    class _StageReceiptsContainer:
        def __init__(self, receipts: list[dict[str, Any]]) -> None:
            self.stage_receipts = receipts

    class _StageObj:
        def __init__(self, pre_e: float, post_e: float) -> None:
            self.pre_anse_energy = pre_e
            self.post_anse_energy = post_e

    assert verify_monotone_anse_stage_descent([0.1, -0.1, -0.3]).passed is True
    assert verify_monotone_anse_stage_descent(
        _StageReceiptsContainer([{"stage_name": "s1", "pre_anse_energy": 0.1, "post_anse_energy": -0.2}])
    ).passed is True
    assert verify_monotone_anse_stage_descent([_StageObj(0.2, -0.1)]).passed is True
    assert verify_monotone_anse_stage_descent([("base", 0.2), ("s1", -0.1), ("s2", -0.25)]).passed is True
    assert verify_monotone_anse_stage_descent(
        [{"stage": "base", "anse_energy": 0.2}, {"stage": "s1", "anse_energy": -0.2}]
    ).passed is True

    # Non-monotone step (-0.1 -> -0.05) and flat step (-0.1 -> -0.1)
    r_up = verify_monotone_anse_stage_descent([0.2, -0.1, -0.05])
    r_flat = verify_monotone_anse_stage_descent([0.2, -0.1, -0.1])
    assert r_up.passed is False and r_flat.passed is False

    # Min-descent margin violation, Maximum Pain entry, guard excess, broken chain/rollback, delta forgery
    r_margin = verify_monotone_anse_stage_descent([0.0, -0.01], min_descent=0.05)
    assert r_margin.passed is False
    r_pain = verify_monotone_anse_stage_descent([0.0, -0.1, ANSE_MAXIMUM_PAIN])
    assert r_pain.passed is False

    r_guard = verify_monotone_anse_stage_descent(
        [
            {
                "stage_name": "s1",
                "accepted": True,
                "pre_anse_energy": 0.0,
                "post_anse_energy": -0.2,
                "general_rel_drift": 0.025,
            }
        ],
        general_guard_rel=0.01,
    )
    assert r_guard.passed is False

    r_bad_chain = verify_monotone_anse_stage_descent(
        [
            {
                "stage_name": "s1",
                "accepted": True,
                "pre_anse_energy": 0.1,
                "post_anse_energy": -0.1,
                "pre_sha256": "a" * 64,
                "post_sha256": "b" * 64,
            },
            {
                "stage_name": "s2",
                "accepted": True,
                "pre_anse_energy": -0.5,
                "post_anse_energy": -0.8,
                "pre_sha256": "z" * 64,
                "post_sha256": "c" * 64,
                "delta_anse_energy": -0.9,
            },
        ]
    )
    assert r_bad_chain.passed is False

    r_bad_rollback = verify_monotone_anse_stage_descent(
        [
            {
                "stage_name": "s1",
                "accepted": False,
                "pre_anse_energy": -0.1,
                "post_anse_energy": -0.1,
                "general_rel_drift": 0.002,
                "pre_sha256": "a" * 64,
                "post_sha256": "b" * 64,
                "edited_params": 4,
            }
        ],
        allow_rolled_back_stages=False,
    )
    assert r_bad_rollback.passed is False

    # Bundle rejection for ULP anchor tamper and non-monotone ANSE
    bad_cert_s4 = certify_s5_surgery_bundle(
        tb["p_null"],
        tb["delta_w"],
        tb["k_preserved"],
        tb["u_gate"],
        tb["spec_centroids"],
        tb["prose_centroid"],
        tb["pre_state"],
        post_ulp_f32,
        tb["anchor_prefixes"],
        [0.1, -0.2, -0.05],
    )
    assert bad_cert_s4.certified is False
    assert bad_cert_s4.anchor_mutated_bits >= 1
    assert bad_cert_s4.failed_obligations == (
        "anchor_layer_bitwise_immutability",
        "monotone_anse_stage_descent",
    )


def test_s5_validation_errors_and_security_stub_compliance_gates() -> None:
    """S5: Input validation contracts and AGPL/AST/security/gatekeeper compliance gates."""
    tb = _build_valid_torch_bundle_inputs()

    # 1. Projector validation errors
    with pytest.raises(TypeError, match="torch.Tensor or numpy.ndarray"):
        verify_projector_idempotence_and_symmetry([[1.0, 0.0], [0.0, 1.0]])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating-point"):
        verify_projector_idempotence_and_symmetry(torch.eye(2, dtype=torch.int64))
    with pytest.raises(TypeError, match="floating-point"):
        verify_projector_idempotence_and_symmetry(np.eye(2, dtype=np.int32))
    with pytest.raises(ValueError, match="2D"):
        verify_projector_idempotence_and_symmetry(torch.ones(4))
    with pytest.raises(ValueError, match="non-empty"):
        verify_projector_idempotence_and_symmetry(torch.empty((0, 0)))
    with pytest.raises(ValueError, match="square"):
        verify_projector_idempotence_and_symmetry(torch.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        verify_projector_idempotence_and_symmetry(torch.tensor([[1.0, float("nan")], [0.0, 1.0]]))
    with pytest.raises(TypeError, match="real number"):
        verify_projector_idempotence_and_symmetry(torch.eye(2), tol=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        verify_projector_idempotence_and_symmetry(torch.eye(2), tol=float("nan"))
    with pytest.raises(ValueError, match=">= 0.0"):
        verify_projector_idempotence_and_symmetry(torch.eye(2), tol=-1e-3)

    # 2. Preserved-key & multigate validation errors
    with pytest.raises(ValueError, match="Incompatible matrix shapes"):
        verify_preserved_key_invariance(torch.zeros((3, 4)), torch.ones((5, 2)))
    with pytest.raises(ValueError, match="prose_centroid dim"):
        verify_orthogonal_multigate_centroid_invariance(
            torch.ones(4), {"E1": torch.ones(4)}, torch.ones(5)
        )
    with pytest.raises(ValueError, match="width"):
        verify_orthogonal_multigate_centroid_invariance(
            torch.ones(4), torch.ones((2, 5)), torch.ones(4)
        )
    with pytest.raises(ValueError, match="dim"):
        verify_orthogonal_multigate_centroid_invariance(
            torch.ones(4), {"E1": torch.ones(5)}, torch.ones(4)
        )
    with pytest.raises(ValueError, match="dim"):
        verify_orthogonal_multigate_centroid_invariance(
            torch.ones(4), [torch.ones(5)], torch.ones(4)
        )
    with pytest.raises(ValueError, match="not be empty"):
        verify_orthogonal_multigate_centroid_invariance(torch.ones(4), {}, torch.ones(4))
    with pytest.raises(ValueError, match="not be empty"):
        verify_orthogonal_multigate_centroid_invariance(torch.ones(4), [], torch.ones(4))
    with pytest.raises(TypeError, match="keys must be int or str"):
        verify_orthogonal_multigate_centroid_invariance(
            torch.ones(4), {True: torch.ones(4)}, torch.ones(4)
        )
    with pytest.raises(ValueError, match="non-empty"):
        verify_orthogonal_multigate_centroid_invariance(
            torch.ones(4), {" ": torch.ones(4)}, torch.ones(4)
        )
    with pytest.raises(TypeError, match="Mapping, Sequence"):
        verify_orthogonal_multigate_centroid_invariance(
            torch.ones(4), "bad_centroids", torch.ones(4)
        )

    # 3. Anchor immutability, state SHA-256, ANSE & bundle validation errors
    with pytest.raises(TypeError, match="Mapping"):
        compute_state_sha256("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        compute_state_sha256({})
    with pytest.raises(TypeError, match="torch.Tensor or numpy.ndarray"):
        compute_state_sha256({"a": [1.0]})
    with pytest.raises(KeyError, match="missing"):
        compute_state_sha256({"a": torch.ones(2)}, keys=["missing"])
    with pytest.raises(ValueError, match="non-finite"):
        compute_state_sha256({"a": torch.tensor([float("nan")])})
    with pytest.raises(ValueError, match="non-finite"):
        verify_anchor_layer_bitwise_immutability(
            {"blocks.0.w": torch.tensor([float("inf")])},
            {"blocks.0.w": torch.ones(1)},
            ("blocks.0.",),
        )
    with pytest.raises(TypeError, match="Sequence or set"):
        verify_anchor_layer_bitwise_immutability(tb["pre_state"], tb["post_state"], "blocks.0.")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not be empty"):
        verify_anchor_layer_bitwise_immutability(tb["pre_state"], tb["post_state"], [])
    with pytest.raises(ValueError, match="matched 0 keys"):
        verify_anchor_layer_bitwise_immutability(tb["pre_state"], tb["post_state"], ["nonexistent."])

    with pytest.raises(TypeError, match="non-empty sequence"):
        verify_monotone_anse_stage_descent("bad")
    with pytest.raises(ValueError, match="not be empty"):
        verify_monotone_anse_stage_descent([])
    with pytest.raises(TypeError, match="cannot be bool"):
        verify_monotone_anse_stage_descent([True, False])
    with pytest.raises(ValueError, match="at least 2"):
        verify_monotone_anse_stage_descent([0.1])
    with pytest.raises(ValueError, match="at least 2"):
        verify_monotone_anse_stage_descent([("s0", 0.1)])
    with pytest.raises(TypeError, match="pair"):
        verify_monotone_anse_stage_descent([("s0", 0.1), ("s1",)])
    with pytest.raises(ValueError, match="at least 2"):
        verify_monotone_anse_stage_descent([{"stage": "s0", "anse_energy": 0.1}])
    with pytest.raises(ValueError, match="missing energy"):
        verify_monotone_anse_stage_descent([{"stage": "s0"}, {"stage": "s1"}])
    with pytest.raises(ValueError, match="both pre and post"):
        verify_monotone_anse_stage_descent([{"pre_anse_energy": 0.1}])
    with pytest.raises(ValueError, match="> 0.0"):
        verify_monotone_anse_stage_descent([0.1, -0.1], maximum_pain=0.0)
    with pytest.raises(TypeError, match="bool"):
        verify_monotone_anse_stage_descent([0.1, -0.1], allow_rolled_back_stages="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        verify_monotone_anse_stage_descent([{"pre_anse_energy": 0.1, "post_anse_energy": -0.1, "edited_params": True}])
    with pytest.raises(ValueError, match=">= 0"):
        verify_monotone_anse_stage_descent([{"pre_anse_energy": 0.1, "post_anse_energy": -0.1, "edited_params": -1}])
    with pytest.raises(TypeError, match="str"):
        certify_s5_surgery_bundle(
            tb["p_null"],
            tb["delta_w"],
            tb["k_preserved"],
            tb["u_gate"],
            tb["spec_centroids"],
            tb["prose_centroid"],
            tb["pre_state"],
            tb["post_state"],
            tb["anchor_prefixes"],
            tb["stage_records"],
            surgery_id=123,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="non-empty string"):
        certify_s5_surgery_bundle(
            tb["p_null"],
            tb["delta_w"],
            tb["k_preserved"],
            tb["u_gate"],
            tb["spec_centroids"],
            tb["prose_centroid"],
            tb["pre_state"],
            tb["post_state"],
            tb["anchor_prefixes"],
            tb["stage_records"],
            surgery_id="   ",
        )

    with pytest.raises(ValueError, match="must not be empty"):
        certify_s5_surgery_bundle(
            {},
            tb["delta_w"],
            tb["k_preserved"],
            tb["u_gate"],
            tb["spec_centroids"],
            tb["prose_centroid"],
            tb["pre_state"],
            tb["post_state"],
            tb["anchor_prefixes"],
            tb["stage_records"],
        )
    with pytest.raises(TypeError, match="Mapping"):
        certify_s5_surgery_bundle(
            tb["p_null"],
            tb["delta_w"],
            {"a": tb["k_preserved"]},
            tb["u_gate"],
            tb["spec_centroids"],
            tb["prose_centroid"],
            tb["pre_state"],
            tb["post_state"],
            tb["anchor_prefixes"],
            tb["stage_records"],
        )
    with pytest.raises(ValueError, match="identical keys"):
        certify_s5_surgery_bundle(
            tb["p_null"],
            {"a": tb["delta_w"]},
            {"b": tb["k_preserved"]},
            tb["u_gate"],
            tb["spec_centroids"],
            tb["prose_centroid"],
            tb["pre_state"],
            tb["post_state"],
            tb["anchor_prefixes"],
            tb["stage_records"],
        )
    with pytest.raises(ValueError, match="must not be empty"):
        certify_s5_surgery_bundle(
            tb["p_null"],
            {},
            tb["k_preserved"],
            tb["u_gate"],
            tb["spec_centroids"],
            tb["prose_centroid"],
            tb["pre_state"],
            tb["post_state"],
            tb["anchor_prefixes"],
            tb["stage_records"],
        )
    with pytest.raises(TypeError, match="Mapping"):
        certify_s5_surgery_bundle(
            tb["p_null"],
            tb["delta_w"],
            tb["k_preserved"],
            tb["u_gate"],
            tb["spec_centroids"],
            {"a": tb["prose_centroid"]},
            tb["pre_state"],
            tb["post_state"],
            tb["anchor_prefixes"],
            tb["stage_records"],
        )
    with pytest.raises(ValueError, match="must not be empty"):
        certify_s5_surgery_bundle(
            tb["p_null"],
            tb["delta_w"],
            tb["k_preserved"],
            {},
            tb["spec_centroids"],
            tb["prose_centroid"],
            tb["pre_state"],
            tb["post_state"],
            tb["anchor_prefixes"],
            tb["stage_records"],
        )
    with pytest.raises(TypeError, match="metadata"):
        certify_s5_surgery_bundle(
            tb["p_null"],
            tb["delta_w"],
            tb["k_preserved"],
            tb["u_gate"],
            tb["spec_centroids"],
            tb["prose_centroid"],
            tb["pre_state"],
            tb["post_state"],
            tb["anchor_prefixes"],
            tb["stage_records"],
            metadata="bad_meta",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="Unsupported value type"):
        certify_s5_surgery_bundle(
            tb["p_null"],
            tb["delta_w"],
            tb["k_preserved"],
            tb["u_gate"],
            tb["spec_centroids"],
            tb["prose_centroid"],
            tb["pre_state"],
            tb["post_state"],
            tb["anchor_prefixes"],
            tb["stage_records"],
            metadata={"bad_obj": object()},
        )
    with pytest.raises(ValueError, match="finite"):
        certify_s5_surgery_bundle(
            tb["p_null"],
            tb["delta_w"],
            tb["k_preserved"],
            tb["u_gate"],
            tb["spec_centroids"],
            tb["prose_centroid"],
            tb["pre_state"],
            tb["post_state"],
            tb["anchor_prefixes"],
            tb["stage_records"],
            metadata={"bad_num": float("nan")},
        )
    with pytest.raises(TypeError, match="WSPFormalCertificate or Mapping"):
        verify_wsp_formal_certificate_provenance("not_a_cert")  # type: ignore[arg-type]
    assert len(compute_payload_sha256({"max_residual_norm": 1e-12, "energy_delta": -0.4, "anse_energy": -0.4})) == 64

    # 4. AGPL-3.0 header, CI-safe fallback + optional ZeroStubAudit, PRSecurityScanner, lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn22", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn22", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/wsp_formal_invariant_verifier.py", TARGET_MODULE_PATH),
        ("tests/test_wsp_formal_invariant_verifier_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

        assert fallback_auditor.check_file(path_obj) == []
        assert fallback_auditor.audit_python_code(src_text).is_clean is True

        if ZeroStubAudit is not None:
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
            {
                "filename": rel,
                "status": "added",
                "additions": len(src_text.splitlines()),
                "patch": "",
                "content": src_text,
            }
        )

    sample_cert_json = json.dumps(
        {
            "object": CERTIFICATE_OBJECT,
            "max_residual_norm": 1.2e-14,
            "energy_delta": -0.42,
            "anse_energy": -0.42,
            "certified": True,
        },
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+CERT = {sample_cert_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1122,
        pr_title="feat(swarm): HM-S35-08 certified algebraic soundness verifier for S5 weight-space programming invariants",
        pr_body="Implements NEURAL-CTRL-DYN-22 formal algebraic verifier and deterministic CPU scenario tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): HM-S35-08 certified algebraic soundness verifier for S5 weight-space programming invariants"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []

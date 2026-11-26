# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Deterministic CPU tests for DAILY-C1-S1: null-space protected post-JEPA Gelee Royale rescuer."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch
from hivemind.jepa.nullspace_jepa_rescuer import (
    ANSE_MAXIMUM_PAIN,
    RESCUER_RECEIPT_OBJECT,
    RESCUER_SCHEMA,
    JepaLoRADelta,
    NullspaceJepaRescueConfig,
    build_prose_nullspace_projector,
    compute_nll_and_ece,
    compute_state_sha256,
    measure_rankme_and_cosine_isotropy,
    project_jepa_lora_associative,
    rescue_post_jepa_lora,
    verify_nullspace_jepa_receipt_provenance,
)
from hivemind.security.owner_order_guardrail import (
    RequestIntentCategory,
    classify_request_intent,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:
    ZeroStubAudit = None


class _FallbackAuditResult:
    """CI-safe AST stub audit outcome when optional ``gwaya`` is absent."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = tuple(violations)
        self.is_clean = len(violations) == 0


class _FallbackZeroStubAudit:
    """CI-safe AST stub checker mirroring ``ZeroStubAudit.audit_python_code``."""

    def audit_python_code(self, code: str) -> _FallbackAuditResult:
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
        return _FallbackAuditResult(violations)


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = REPO_ROOT / "src" / "hivemind" / "jepa" / "nullspace_jepa_rescuer.py"
TEST_PATH = Path(__file__).resolve()
LEDGER_DOC_PATH = REPO_ROOT / "docs" / "DAILY_ROADMAP_PAPER_LEDGER.md"
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."


@pytest.fixture(autouse=True, scope="module")
def _cpu_single_thread() -> Any:
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


def _load_gatekeeper() -> Any:
    spec = importlib.util.spec_from_file_location("pr_gatekeeper", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_jepa_benchmark_fixture(*, n_per_dom: int = 48) -> dict[str, Any]:
    """Build deterministic CPU fixture reproducing P1 (+0.90 nat) / E-STP-1b (+1.37 nat) prose drift.

    - Coordinates ``0..7``: ``D0_general_prose`` (classes ``0..3``).
    - Coordinates ``8..15``: ``D1_jepa_code`` (classes ``4..7``, shifted features fixed by specialist LoRA).
    - Coordinates ``16..23``: ``D2_jepa_lean4`` (classes ``8..11``, shifted features fixed by specialist LoRA).
    """
    gen = torch.Generator().manual_seed(20261126)
    d_model = 24
    vocab = 12

    lm_head = torch.zeros(vocab, d_model, dtype=torch.float32)
    for cls_idx in range(vocab):
        if cls_idx < 4:
            lm_head[cls_idx, cls_idx] = 2.2
        elif cls_idx < 8:
            lm_head[cls_idx, 8 + (cls_idx - 4)] = 2.2
        else:
            lm_head[cls_idx, 16 + (cls_idx - 8)] = 2.2
    lm_head = lm_head + 0.02 * torch.randn(vocab, d_model, generator=gen)

    base_weight = torch.zeros(d_model, d_model, dtype=torch.float32)
    final_norm = torch.full((d_model,), 1.5, dtype=torch.float32)

    x_d0 = torch.zeros(n_per_dom, d_model, dtype=torch.float32)
    y_d0 = torch.arange(n_per_dom, dtype=torch.long) % 4
    for i in range(n_per_dom):
        c = int(y_d0[i].item())
        x_d0[i, c] = 1.0
        x_d0[i, 4 + ((c + i // 4) % 4)] = 0.25

    x_d1 = torch.zeros(n_per_dom, d_model, dtype=torch.float32)
    y_d1 = 4 + (torch.arange(n_per_dom, dtype=torch.long) % 4)
    for i in range(n_per_dom):
        sub = int((y_d1[i] - 4).item())
        x_d1[i, 8 + ((sub + 1) % 4)] = 1.0
        x_d1[i, 12 + ((sub + i // 4) % 4)] = 0.25

    x_d2 = torch.zeros(n_per_dom, d_model, dtype=torch.float32)
    y_d2 = 8 + (torch.arange(n_per_dom, dtype=torch.long) % 4)
    for i in range(n_per_dom):
        sub = int((y_d2[i] - 8).item())
        x_d2[i, 16 + ((sub + 1) % 4)] = 1.0
        x_d2[i, 20 + ((sub + i // 4) % 4)] = 0.25

    # Construct rank-12 LoRA: ranks 0..7 heal D1/D2; ranks 8..11 corrupt D0 by > +1.1 nat!
    rank = 12
    b_mat = torch.zeros(d_model, rank, dtype=torch.float32)
    a_mat = torch.zeros(rank, d_model, dtype=torch.float32)
    for sub in range(4):
        b_mat[8 + sub, sub] = 2.0
        a_mat[sub, 8 + ((sub + 1) % 4)] = 1.0
        b_mat[16 + sub, 4 + sub] = 2.0
        a_mat[4 + sub, 16 + ((sub + 1) % 4)] = 1.0
        # Prose corruption shortcut component inside coordinates 0..3
        b_mat[sub, 8 + sub] = -2.2
        b_mat[(sub + 2) % 4, 8 + sub] = 2.2
        a_mat[8 + sub, sub] = 1.0

    v_prose = torch.zeros(8, d_model, dtype=torch.float64)
    for idx in range(8):
        v_prose[idx, idx] = 1.0

    return {
        "mutable_state": {"layer_weight": base_weight.clone(), "final_norm_weight": final_norm.clone()},
        "lm_head_weight": lm_head,
        "jepa_lora": JepaLoRADelta(b_mat=b_mat, a_mat=a_mat, scale=1.0, objective_mode="pairs", adapter_name="p1_jepa"),
        "domain_hidden_inputs": {"D0_general_prose": x_d0, "D1_jepa_code": x_d1, "D2_jepa_lean4": x_d2},
        "domain_targets": {"D0_general_prose": y_d0, "D1_jepa_code": y_d1, "D2_jepa_lean4": y_d2},
        "v_prose": v_prose,
    }


def test_s1_exact_algebraic_annihilation_and_specialist_alignment() -> None:
    """S1: Exact algebraic annihilation ||((sB)(AP_0)) V_prose^T|| < 1e-6 & specialist alignment."""
    fx = _build_jepa_benchmark_fixture()
    v_prose, p_0 = build_prose_nullspace_projector(v_prose=fx["v_prose"])
    assert p_0.shape == (24, 24)
    assert float(torch.linalg.matrix_norm(p_0 @ v_prose.T, ord="fro").item()) < 1e-12

    spec_x = torch.cat([fx["domain_hidden_inputs"]["D1_jepa_code"], fx["domain_hidden_inputs"]["D2_jepa_lean4"]], dim=0)
    proj = project_jepa_lora_associative(
        fx["jepa_lora"],
        p_0,
        v_prose,
        specialist_activations=spec_x,
        alpha=1.0,
    )
    # Exact algebraic annihilation & associative identity Theorem 15.1(2)
    assert proj.annihilation_norm < 1e-6
    assert proj.associative_diff < 1e-12
    assert proj.rank_bound == 12
    assert proj.buffer_compression_ratio == pytest.approx(24.0 / 12.0)
    # Specialist alignment is cleanly preserved in the orthogonal complement
    assert proj.specialist_alignment_cosine > 0.999
    assert proj.specialist_retained_norm_ratio > 0.999

    # Also verify SVD synthesis from prose_activations and fallback basis extraction without v_prose
    v_svd, p0_svd = build_prose_nullspace_projector(
        prose_activations=fx["domain_hidden_inputs"]["D0_general_prose"],
        variance_threshold=1.0,
    )
    assert int(v_svd.shape[0]) == 7
    proj_no_v = project_jepa_lora_associative(
        {"lora_B": fx["jepa_lora"].b_mat, "lora_A": fx["jepa_lora"].a_mat, "scale": 1.0},
        p0_svd,
    )
    assert proj_no_v.annihilation_norm < 1e-6
    assert (
        float(
            torch.linalg.matrix_norm(
                fx["domain_hidden_inputs"]["D0_general_prose"].double() @ proj_no_v.delta_projected.T,
                ord="fro",
            ).item()
        )
        < 1e-10
    )
    assert proj_no_v.specialist_alignment_cosine > 0.5

    # Zero-energy prose activations yield identity P_0 and empty V_prose
    v_empty, p0_empty = build_prose_nullspace_projector(prose_activations=torch.zeros(4, 8))
    assert v_empty.shape == (0, 8)
    assert torch.allclose(p0_empty, torch.eye(8, dtype=torch.float64))


def test_s2_rescue_guard_failing_jepa_delta_into_anse_pass() -> None:
    """S2: Guard-failing post-JEPA delta (+0.90 / +1.37 nat prose drift) is rescued into ANSE PASS."""
    fx = _build_jepa_benchmark_fixture()
    state = fx["mutable_state"]

    receipt = rescue_post_jepa_lora(
        state,
        fx["lm_head_weight"],
        fx["jepa_lora"],
        fx["domain_hidden_inputs"],
        fx["domain_targets"],
        v_prose=fx["v_prose"],
    )

    assert receipt["raw_jepa_guard_ok"] is False
    assert receipt["raw_jepa_general_drift_nats"] > 0.90
    assert receipt["raw_jepa_anse_energy"] == ANSE_MAXIMUM_PAIN

    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["collapse_rejected"] is False
    assert receipt["guard_rejected"] is False
    assert receipt["annihilation_norm"] < 1e-6
    assert receipt["general_drift_ratio"] <= 0.010
    assert abs(receipt["general_drift_nats"]) < 1e-5
    assert receipt["worst_domain_drift_ratio"] <= 0.010
    assert receipt["anse_energy"] < 0.0
    assert receipt["nll_diff"] < 0.0
    assert receipt["one_minus_cosine_post"] > 0.05
    assert receipt["rankme_post"] > 5.0
    assert receipt["final_state_sha256"] != receipt["initial_state_sha256"]
    assert verify_nullspace_jepa_receipt_provenance(receipt) is True

    # Also verify STP objective mode, rectangular (d_out != d_in) non-residual layer without final_norm,
    # and automatic prose_activations synthesis when neither v_prose nor p_0 is passed.
    rect_b = torch.zeros(12, 4, dtype=torch.float32)
    rect_a = torch.zeros(4, 24, dtype=torch.float32)
    for sub in range(4):
        rect_b[4 + sub, sub] = 2.0
        rect_a[sub, 8 + ((sub + 1) % 4)] = 1.0
        rect_b[sub, sub] = -2.2
        rect_a[sub, sub] = 1.0
    rect_state = {"layer_weight": torch.zeros(12, 24, dtype=torch.float32)}
    rect_head = torch.eye(12, dtype=torch.float32) * 2.0
    rect_receipt = rescue_post_jepa_lora(
        rect_state,
        rect_head,
        JepaLoRADelta(b_mat=rect_b, a_mat=rect_a, scale=1.0, objective_mode="stp", adapter_name="e_stp_1b"),
        {"D0_general_prose": fx["domain_hidden_inputs"]["D0_general_prose"], "D1_jepa_code": fx["domain_hidden_inputs"]["D1_jepa_code"]},
        {"D0_general_prose": fx["domain_targets"]["D0_general_prose"], "D1_jepa_code": fx["domain_targets"]["D1_jepa_code"]},
    )
    assert rect_receipt["accepted"] is True
    assert rect_receipt["objective_mode"] == "stp"
    assert verify_nullspace_jepa_receipt_provenance(rect_receipt) is True


def test_s3_reject_cosine_collapsed_and_low_rankme_representation_collapse() -> None:
    """S3: Reject cosine-collapsed / low-RankMe representation collapse with atomic rollback."""
    fx = _build_jepa_benchmark_fixture()
    d_model = 24
    # Build a rank-1 LoRA inside P_0 (coords 8..23) whose massive rank-1 cone swamps hidden states
    # so 1 - cos <= 5e-4 and RankMe collapses near 1.0 despite zero D0 drift!
    b_collapse = torch.zeros(d_model, 1, dtype=torch.float32)
    a_collapse = torch.zeros(1, d_model, dtype=torch.float32)
    b_collapse[8, 0] = 500.0
    a_collapse[0, 8:24] = 1.0

    state = fx["mutable_state"]
    w_before = state["layer_weight"].clone()
    n_before = state["final_norm_weight"].clone()
    init_sha = compute_state_sha256(state)

    receipt = rescue_post_jepa_lora(
        state,
        fx["lm_head_weight"],
        JepaLoRADelta(b_mat=b_collapse, a_mat=a_collapse, scale=1.0, adapter_name="collapsed_jepa"),
        fx["domain_hidden_inputs"],
        fx["domain_targets"],
        v_prose=fx["v_prose"],
    )

    assert receipt["accepted"] is False
    assert receipt["rolled_back"] is True
    assert receipt["rollback_verified"] is True
    assert receipt["collapse_rejected"] is True
    assert receipt["rejection_reason"] == "rankme_or_cosine_collapse_detected"
    assert receipt["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert receipt["specialist_one_minus_cosine_post"] < 0.05
    assert "cosine_cone_collapse" in receipt["collapse_reasons"]
    assert receipt["final_state_sha256"] == init_sha
    assert torch.equal(state["layer_weight"], w_before)
    assert torch.equal(state["final_norm_weight"], n_before)
    assert verify_nullspace_jepa_receipt_provenance(receipt) is True

    # Also verify direct RankMe & paired-view / STP collapse diagnostics (P1 1 - cos <= 5e-4)
    gen = torch.Generator().manual_seed(7)
    iso_h = torch.randn(32, 16, generator=gen)
    iso_rep = measure_rankme_and_cosine_isotropy(iso_h)
    assert iso_rep.passed is True
    assert iso_rep.one_minus_cosine > 0.5

    collapsed_h = torch.ones(32, 16) + 1e-4 * torch.randn(32, 16, generator=gen)
    col_rep = measure_rankme_and_cosine_isotropy(collapsed_h, reference_report=iso_rep)
    assert col_rep.passed is False
    assert col_rep.one_minus_cosine <= 5e-4
    assert "cosine_cone_collapse" in col_rep.collapse_reasons
    assert "rankme_drop_vs_reference" in col_rep.collapse_reasons

    # Zero-state & paired-view / STP collapse flags
    zero_rep = measure_rankme_and_cosine_isotropy(torch.zeros(8, 8))
    assert zero_rep.passed is False and "zero_or_degenerate_states" in zero_rep.collapse_reasons

    v1 = torch.ones(8, 12)
    v2 = torch.ones(8, 12) + 1e-4 * torch.randn(8, 12, generator=gen)
    pv_rep = measure_rankme_and_cosine_isotropy(iso_h, paired_views=(v1, v2))
    assert pv_rep.passed is False and "jepa_objective_cosine_collapse" in pv_rep.collapse_reasons

    straight_seq = torch.arange(6, dtype=torch.float32).view(1, 6, 1).expand(4, 6, 8).clone()
    stp_rep = measure_rankme_and_cosine_isotropy(iso_h, stp_hidden_seq=straight_seq)
    assert stp_rep.passed is False and "jepa_objective_cosine_collapse" in stp_rep.collapse_reasons

    # Reject via paired_views_fn inside rescue_post_jepa_lora
    state2 = {"layer_weight": w_before.clone(), "final_norm_weight": n_before.clone()}
    pv_receipt = rescue_post_jepa_lora(
        state2,
        fx["lm_head_weight"],
        fx["jepa_lora"],
        fx["domain_hidden_inputs"],
        fx["domain_targets"],
        v_prose=fx["v_prose"],
        paired_views_fn=lambda _w, _n: (v1, v2),
    )
    assert pv_receipt["accepted"] is False and pv_receipt["collapse_rejected"] is True
    assert pv_receipt["jepa_one_minus_cos"] is not None and pv_receipt["jepa_one_minus_cos"] <= 5e-4


def test_s4_atomic_bitwise_rollback_on_prose_drift_or_non_improving_anse() -> None:
    """S4: Atomic bitwise rollback when D0 drift > +1.0% or ANSE Delta E >= 0 or P_0 invalid."""
    fx = _build_jepa_benchmark_fixture()

    # 4a. Incomplete prose basis (covers only coord 0 while D0 lives in 0..3):
    #     A rank-preserving cyclic permutation on coords 1..3 preserves RankMe/isotropy
    #     yet scrambles D0 predictions (> +1.0% prose drift) -> ANSE guard triggers rollback.
    partial_v = torch.zeros(1, 24, dtype=torch.float64)
    partial_v[0, 0] = 1.0
    b_drift = torch.zeros(24, 3, dtype=torch.float32)
    a_drift = torch.zeros(3, 24, dtype=torch.float32)
    perm_targets = {1: 2, 2: 3, 3: 1}
    for idx, c in enumerate((1, 2, 3)):
        b_drift[c, idx] = -1.0
        b_drift[perm_targets[c], idx] = 1.0
        a_drift[idx, c] = 1.0

    state_a = fx["mutable_state"]
    w_orig_a = state_a["layer_weight"].clone()
    n_orig_a = state_a["final_norm_weight"].clone()
    sha_a = compute_state_sha256(state_a)

    rec_drift = rescue_post_jepa_lora(
        state_a,
        fx["lm_head_weight"],
        JepaLoRADelta(b_mat=b_drift, a_mat=a_drift, scale=1.0),
        fx["domain_hidden_inputs"],
        fx["domain_targets"],
        v_prose=partial_v,
        config=NullspaceJepaRescueConfig(safe_tau_grid=(1.0,)),
    )
    assert rec_drift["accepted"] is False
    assert rec_drift["rolled_back"] is True
    assert rec_drift["rollback_verified"] is True
    assert rec_drift["guard_rejected"] is True
    assert rec_drift["rejection_reason"] == "general_prose_or_domain_guard_exceeded"
    assert rec_drift["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert rec_drift["final_state_sha256"] == sha_a
    assert torch.equal(state_a["layer_weight"], w_orig_a)
    assert torch.equal(state_a["final_norm_weight"], n_orig_a)
    assert verify_nullspace_jepa_receipt_provenance(rec_drift) is True

    # 4b. ANSE Delta E >= 0 (specialist corruption inside P_0, D0 == 0%) -> rollback
    b_harm_spec = torch.zeros(24, 4, dtype=torch.float32)
    a_harm_spec = torch.zeros(4, 24, dtype=torch.float32)
    for sub in range(4):
        b_harm_spec[8 + sub, sub] = -2.0
        a_harm_spec[sub, 8 + ((sub + 1) % 4)] = 1.0

    state_b = {"layer_weight": w_orig_a.clone(), "final_norm_weight": n_orig_a.clone()}
    sha_b = compute_state_sha256(state_b)
    rec_energy = rescue_post_jepa_lora(
        state_b,
        fx["lm_head_weight"],
        JepaLoRADelta(b_mat=b_harm_spec, a_mat=a_harm_spec, scale=1.0),
        fx["domain_hidden_inputs"],
        fx["domain_targets"],
        v_prose=fx["v_prose"],
        config=NullspaceJepaRescueConfig(safe_tau_grid=(1.0,), max_domain_drift_ratio=5.0),
    )
    assert rec_energy["accepted"] is False
    assert rec_energy["rolled_back"] is True
    assert rec_energy["rollback_verified"] is True
    assert rec_energy["rejection_reason"] == "anse_energy_non_improving"
    assert rec_energy["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert rec_energy["final_state_sha256"] == sha_b
    assert verify_nullspace_jepa_receipt_provenance(rec_energy) is True

    # 4c. Inconsistent (p_0 = I, v_prose) fails algebraic annihilation tolerance -> rollback
    state_c = {"layer_weight": w_orig_a.clone(), "final_norm_weight": n_orig_a.clone()}
    rec_annih = rescue_post_jepa_lora(
        state_c,
        fx["lm_head_weight"],
        fx["jepa_lora"],
        fx["domain_hidden_inputs"],
        fx["domain_targets"],
        v_prose=fx["v_prose"],
        p_0=torch.eye(24, dtype=torch.float64),
    )
    assert rec_annih["accepted"] is False
    assert rec_annih["rolled_back"] is True
    assert rec_annih["rejection_reason"] == "prose_nullspace_annihilation_breach"
    assert rec_annih["final_state_sha256"] == compute_state_sha256(state_c)


def test_s5_deterministic_receipt_parity_validation_and_compliance_gates() -> None:
    """S5: Deterministic SHA-256 receipt parity, input validation, and repository compliance gates."""
    fx1 = _build_jepa_benchmark_fixture()
    fx2 = _build_jepa_benchmark_fixture()

    r1 = rescue_post_jepa_lora(
        fx1["mutable_state"],
        fx1["lm_head_weight"],
        fx1["jepa_lora"],
        fx1["domain_hidden_inputs"],
        fx1["domain_targets"],
        v_prose=fx1["v_prose"],
    )
    r2 = rescue_post_jepa_lora(
        fx2["mutable_state"],
        fx2["lm_head_weight"],
        fx2["jepa_lora"],
        fx2["domain_hidden_inputs"],
        fx2["domain_targets"],
        v_prose=fx2["v_prose"],
    )
    assert r1["schema"] == RESCUER_SCHEMA
    assert r1["object"] == RESCUER_RECEIPT_OBJECT
    assert r1["receipt_sha256"] == r2["receipt_sha256"]
    assert r1["final_state_sha256"] == r2["final_state_sha256"]
    assert r1["projected_factor_sha256"] == r2["projected_factor_sha256"]

    # Tampered receipt provenance checks
    assert verify_nullspace_jepa_receipt_provenance("not-a-dict") is False  # type: ignore[arg-type]
    bad_schema = dict(r1, schema="wrong")
    assert verify_nullspace_jepa_receipt_provenance(bad_schema) is False
    bad_hash = dict(r1, receipt_sha256="not-hex")
    assert verify_nullspace_jepa_receipt_provenance(bad_hash) is False
    tampered = dict(r1, nll_diff=-99.0)
    assert verify_nullspace_jepa_receipt_provenance(tampered) is False
    non_json = dict(r1, anse_energy=float("nan"))
    assert verify_nullspace_jepa_receipt_provenance(non_json) is False
    bad_bool = dict(r1, accepted=True, rolled_back=True)
    bad_bool["receipt_sha256"] = ""
    assert verify_nullspace_jepa_receipt_provenance(bad_bool) is False
    # Re-sign invalid rolled_back receipt with mismatched SHA to trigger rollback invariant branch
    from hivemind.jepa.nullspace_jepa_rescuer import compute_payload_sha256

    bad_rb = {k: v for k, v in r1.items() if k != "receipt_sha256"}
    bad_rb["accepted"] = False
    bad_rb["rolled_back"] = True
    bad_rb["receipt_sha256"] = compute_payload_sha256(bad_rb)
    assert verify_nullspace_jepa_receipt_provenance(bad_rb) is False
    bad_rb2 = dict(bad_rb, final_state_sha256=bad_rb["initial_state_sha256"])
    del bad_rb2["receipt_sha256"]
    bad_rb2["receipt_sha256"] = compute_payload_sha256(bad_rb2)
    assert verify_nullspace_jepa_receipt_provenance(bad_rb2) is False
    bad_acc = {k: v for k, v in r1.items() if k != "receipt_sha256"}
    bad_acc["accepted"] = "yes"
    bad_acc["receipt_sha256"] = compute_payload_sha256(bad_acc)
    assert verify_nullspace_jepa_receipt_provenance(bad_acc) is False

    # Input validation boundary checks
    with pytest.raises(TypeError):
        NullspaceJepaRescueConfig(ece_bins=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NullspaceJepaRescueConfig(max_general_drift_ratio=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NullspaceJepaRescueConfig(residual_connection="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        NullspaceJepaRescueConfig(general_domain_name=" ")
    with pytest.raises(ValueError):
        NullspaceJepaRescueConfig(ece_bins=0)
    with pytest.raises(ValueError):
        NullspaceJepaRescueConfig(max_general_drift_ratio=-0.1)
    with pytest.raises(ValueError):
        NullspaceJepaRescueConfig(annihilation_tol=0.0)
    with pytest.raises(ValueError):
        NullspaceJepaRescueConfig(min_one_minus_cos=3.0)
    with pytest.raises(ValueError):
        NullspaceJepaRescueConfig(min_one_minus_cos=float("nan"))
    with pytest.raises(ValueError):
        NullspaceJepaRescueConfig(alpha_grid=())
    with pytest.raises(ValueError):
        JepaLoRADelta(b_mat=torch.zeros(4, 2), a_mat=torch.zeros(2, 4), adapter_name="")
    with pytest.raises(ValueError):
        JepaLoRADelta(b_mat=torch.zeros(4, 2), a_mat=torch.zeros(2, 4), objective_mode="unknown")
    with pytest.raises(ValueError):
        JepaLoRADelta(b_mat=torch.zeros(4, 2), a_mat=torch.zeros(3, 4))
    with pytest.raises(TypeError):
        JepaLoRADelta(b_mat=[[1.0]], a_mat=torch.zeros(1, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        JepaLoRADelta(b_mat=torch.ones(2, 1, dtype=torch.long), a_mat=torch.ones(1, 2))
    with pytest.raises(ValueError):
        JepaLoRADelta(b_mat=torch.zeros(0, 2), a_mat=torch.zeros(2, 2))
    with pytest.raises(ValueError):
        JepaLoRADelta(b_mat=torch.tensor([[float("nan")]]), a_mat=torch.ones(1, 1))
    with pytest.raises(TypeError):
        compute_state_sha256({"bad": "not-a-tensor"})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        compute_state_sha256({})
    with pytest.raises(ValueError):
        compute_state_sha256({"": torch.ones(2)})
    with pytest.raises(ValueError):
        compute_state_sha256({"nan": torch.tensor([float("inf")])})
    with pytest.raises(ValueError):
        build_prose_nullspace_projector()
    with pytest.raises(ValueError):
        build_prose_nullspace_projector(prose_activations=torch.ones(2, 2), v_prose=torch.eye(2))
    with pytest.raises(ValueError):
        build_prose_nullspace_projector(v_prose=torch.zeros(2, 0))
    with pytest.raises(ValueError):
        build_prose_nullspace_projector(v_prose=torch.ones(3, 2))
    with pytest.raises(ValueError):
        build_prose_nullspace_projector(v_prose=torch.tensor([[2.0, 0.0]]))
    with pytest.raises(ValueError):
        build_prose_nullspace_projector(v_prose=torch.tensor([[float("nan"), 0.0]]))
    with pytest.raises(ValueError):
        build_prose_nullspace_projector(v_prose=torch.zeros(2, 2, 2))
    with pytest.raises(TypeError):
        project_jepa_lora_associative("bad", torch.eye(4))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        project_jepa_lora_associative(
            JepaLoRADelta(b_mat=torch.ones(4, 2), a_mat=torch.ones(2, 4)), torch.eye(3)
        )
    with pytest.raises(ValueError):
        project_jepa_lora_associative(
            JepaLoRADelta(b_mat=torch.ones(4, 2), a_mat=torch.ones(2, 4)), torch.ones(4, 4)
        )
    with pytest.raises(ValueError):
        project_jepa_lora_associative(
            JepaLoRADelta(b_mat=torch.ones(4, 2), a_mat=torch.ones(2, 4)),
            torch.eye(4),
            v_prose=torch.zeros(1, 3),
        )
    with pytest.raises(ValueError):
        project_jepa_lora_associative(
            JepaLoRADelta(b_mat=torch.ones(4, 2), a_mat=torch.ones(2, 4)),
            torch.eye(4),
            v_prose=torch.zeros(1, 1, 4),
        )
    with pytest.raises(ValueError):
        project_jepa_lora_associative(
            JepaLoRADelta(b_mat=torch.ones(4, 2), a_mat=torch.ones(2, 4)),
            torch.tensor([[1.0, 2.0], [0.0, 1.0]]),
        )
    with pytest.raises(ValueError):
        project_jepa_lora_associative(
            JepaLoRADelta(b_mat=torch.ones(4, 2), a_mat=torch.ones(2, 4)),
            torch.eye(4),
            specialist_activations=torch.ones(2, 3),
        )
    with pytest.raises(ValueError):
        measure_rankme_and_cosine_isotropy(torch.ones(2))
    with pytest.raises(ValueError):
        measure_rankme_and_cosine_isotropy(torch.ones(2, 2), paired_views=(torch.ones(2, 2), torch.ones(2, 3)))
    with pytest.raises(ValueError):
        measure_rankme_and_cosine_isotropy(torch.ones(2, 2), stp_hidden_seq=torch.ones(2, 2, 2))
    with pytest.raises(ValueError):
        rescue_post_jepa_lora({}, fx1["lm_head_weight"], fx1["jepa_lora"], fx1["domain_hidden_inputs"], fx1["domain_targets"])
    with pytest.raises(ValueError):
        rescue_post_jepa_lora(
            {"layer_weight": torch.zeros(24, 24), "final_norm_weight": torch.ones(4)},
            fx1["lm_head_weight"],
            fx1["jepa_lora"],
            fx1["domain_hidden_inputs"],
            fx1["domain_targets"],
        )
    with pytest.raises(ValueError):
        rescue_post_jepa_lora(
            {"layer_weight": torch.zeros(24, 24)},
            torch.zeros(12, 4),
            fx1["jepa_lora"],
            fx1["domain_hidden_inputs"],
            fx1["domain_targets"],
        )
    with pytest.raises(ValueError):
        rescue_post_jepa_lora(
            {"layer_weight": torch.zeros(12, 24)},
            torch.zeros(12, 12),
            fx1["jepa_lora"],
            fx1["domain_hidden_inputs"],
            fx1["domain_targets"],
        )
    with pytest.raises(ValueError):
        rescue_post_jepa_lora(
            {"layer_weight": torch.zeros(12, 24)},
            torch.zeros(12, 12),
            JepaLoRADelta(b_mat=torch.zeros(12, 2), a_mat=torch.zeros(2, 24)),
            fx1["domain_hidden_inputs"],
            fx1["domain_targets"],
            config=NullspaceJepaRescueConfig(residual_connection=True),
        )
    with pytest.raises(ValueError):
        rescue_post_jepa_lora(
            {"layer_weight": torch.zeros(24, 24)},
            fx1["lm_head_weight"],
            fx1["jepa_lora"],
            {"D0_general_prose": fx1["domain_hidden_inputs"]["D0_general_prose"]},
            {"D0_general_prose": fx1["domain_targets"]["D0_general_prose"]},
        )
    with pytest.raises(ValueError):
        rescue_post_jepa_lora(
            {"layer_weight": torch.zeros(24, 24)},
            fx1["lm_head_weight"],
            fx1["jepa_lora"],
            {"D1_a": fx1["domain_hidden_inputs"]["D1_jepa_code"], "D2_b": fx1["domain_hidden_inputs"]["D2_jepa_lean4"]},
            {"D1_a": fx1["domain_targets"]["D1_jepa_code"], "D2_b": fx1["domain_targets"]["D2_jepa_lean4"]},
        )
    with pytest.raises(ValueError):
        rescue_post_jepa_lora(
            {"layer_weight": torch.zeros(24, 24)},
            fx1["lm_head_weight"],
            fx1["jepa_lora"],
            fx1["domain_hidden_inputs"],
            {"D0_general_prose": fx1["domain_targets"]["D0_general_prose"]},
        )
    with pytest.raises(ValueError):
        rescue_post_jepa_lora(
            {"layer_weight": torch.zeros(24, 24)},
            fx1["lm_head_weight"],
            fx1["jepa_lora"],
            {"D0_general_prose": torch.zeros(4, 8), "D1_jepa_code": fx1["domain_hidden_inputs"]["D1_jepa_code"]},
            {"D0_general_prose": torch.zeros(4, dtype=torch.long), "D1_jepa_code": fx1["domain_targets"]["D1_jepa_code"]},
        )
    with pytest.raises(ValueError):
        rescue_post_jepa_lora(
            {"layer_weight": torch.zeros(24, 24)},
            fx1["lm_head_weight"],
            fx1["jepa_lora"],
            {"D0_general_prose": fx1["domain_hidden_inputs"]["D0_general_prose"], "D1_jepa_code": fx1["domain_hidden_inputs"]["D1_jepa_code"]},
            {"D0_general_prose": torch.zeros(2, dtype=torch.long), "D1_jepa_code": fx1["domain_targets"]["D1_jepa_code"]},
        )
    with pytest.raises(ValueError):
        rescue_post_jepa_lora(
            {"layer_weight": torch.zeros(24, 24)},
            fx1["lm_head_weight"],
            fx1["jepa_lora"],
            {"D0_general_prose": fx1["domain_hidden_inputs"]["D0_general_prose"], "D1_jepa_code": fx1["domain_hidden_inputs"]["D1_jepa_code"]},
            {"D0_general_prose": torch.full((48,), 99, dtype=torch.long), "D1_jepa_code": fx1["domain_targets"]["D1_jepa_code"]},
        )
    nll_v, ece_v = compute_nll_and_ece(torch.tensor([[2.0, 0.0], [0.0, 2.0]]), torch.tensor([0, 1]), ece_bins=5)
    assert nll_v > 0.0 and ece_v >= 0.0

    # Repository compliance checks: AGPL-3.0, ZeroStubAudit, PRSecurityScanner, FORBIDDEN_ADDED, OwnerOrderGuardrail
    scanner = PRSecurityScanner()
    gk = _load_gatekeeper()
    auditor = ZeroStubAudit() if ZeroStubAudit is not None else _FallbackZeroStubAudit()

    changed_files: list[dict[str, Any]] = []
    for rel, path in (
        ("src/hivemind/jepa/nullspace_jepa_rescuer.py", TARGET_PATH),
        ("tests/test_nullspace_jepa_rescuer.py", TEST_PATH),
    ):
        src = path.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src[:300]
        audit_res = auditor.audit_python_code(src)
        assert audit_res.is_clean is True, f"Stub audit failed on {rel}: {audit_res.violations}"
        assert scanner.scan_python_ast_security(rel, src) == []
        added_diff = "\n".join("+" + ln for ln in src.splitlines())
        assert gk.forbidden_scan(added_diff) == []
        changed_files.append({"filename": rel, "status": "added", "additions": len(src.splitlines()), "patch": "", "content": src})

    doc_text = LEDGER_DOC_PATH.read_text(encoding="utf-8")
    assert "10.5281/zenodo.24553269" in doc_text
    assert "0.00 EUR / 0.00 USD" in doc_text
    assert "AGPL-3.0" in doc_text
    assert "DAILY-C1-S1" in doc_text
    assert "src/hivemind/jepa/nullspace_jepa_rescuer.py" in doc_text
    assert "tests/test_nullspace_jepa_rescuer.py" in doc_text
    assert scanner.scan_text_for_payment_solicitation(doc_text, "docs/DAILY_ROADMAP_PAPER_LEDGER.md") == []
    assert scanner.scan_text_for_secrets_and_exfiltration(doc_text, "docs/DAILY_ROADMAP_PAPER_LEDGER.md") == []
    assert scanner.scan_unicode_trojan_source(doc_text, "docs/DAILY_ROADMAP_PAPER_LEDGER.md") == []
    assert gk.forbidden_scan("\n".join("+" + ln for ln in doc_text.splitlines())) == []
    doc_intent = classify_request_intent(
        doc_text,
        source_channel="pull_request",
        changed_files=("docs/DAILY_ROADMAP_PAPER_LEDGER.md",),
    )
    assert doc_intent.intent_category == RequestIntentCategory.STANDARD_CONTRIBUTION
    assert doc_intent.is_directive_order is False
    assert doc_intent.protected_paths_touched == ()

    changed_files.append(
        {
            "filename": "docs/DAILY_ROADMAP_PAPER_LEDGER.md",
            "status": "added",
            "additions": len(doc_text.splitlines()),
            "patch": "",
            "content": doc_text,
        }
    )
    scan = scanner.audit_pull_request(
        pr_number=1061,
        pr_title="feat(jepa): null-space protected post-JEPA Gelee Royale LoRA rescuer with RankMe isotropy guard",
        pr_body="Implements DAILY-C1-S1 (Paper v4.4.0 section 15.4 Opportunity J1, Theorem 15.1(2)) under 0.00 EUR / 0.00 USD AGPL-3.0.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(jepa): null-space protected post-JEPA Gelee Royale LoRA rescuer"],
        author_email=None,
    )
    assert scan.passed is True
    assert scan.blocked_merge is False
    assert scan.findings == []

# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-24: Lean 4 Season 35 WSP theorems & formal spec bridge."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.auto_open_weight_engine import (
    DEFAULT_GENERAL_DOMAIN,
    AutoOpenWeightConfig,
    SingleShotVaultLock,
    run_auto_open_weight_s5_synergy,
)
from hivemind.swarm.lean4_wsp_theorem_registry import (
    ATTESTATION_RECEIPT_OBJECT,
    EXPECTED_SEASON35_THEOREMS,
    SCHEMA_VERSION,
    SEASON35_NAMESPACE,
    SEASON35_QUALIFIED_PREFIX,
    SEASON35_THEOREM_BY_NAME,
    SEASON35_THEOREM_CATALOG,
    SPEC_AUDIT_OBJECT,
    Lean4SpecAuditError,
    audit_lean4_wsp_spec_file,
    bind_s5_receipt_to_lean4_theorems,
    canonical_json_bytes,
    compute_payload_sha256,
    parse_lean4_wsp_spec,
    resolve_default_lean4_spec_path,
    resolve_s5_theorem_obligations,
    strip_lean4_comments_and_strings,
    verify_lean4_attestation_receipt,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackAuditResult:
    """Minimal AST stub audit result used when optional ``gwaya`` is absent in CI."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = list(violations)
        self.is_clean = len(violations) == 0
        self.penalty_energy = 0.0 if self.is_clean else 1_000_000.0


class _FallbackZeroStubAudit:
    """CI-safe AST stub verifier implementing both ``check_file`` and ``audit_python_code``."""

    def audit_python_code(self, code: str) -> _FallbackAuditResult:
        """Scan Python source for placeholder ``pass`` / ``...`` / ``NotImplementedError`` stubs."""
        tree = ast.parse(code)
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stmts = [
                    stmt
                    for stmt in node.body
                    if not (
                        isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str)
                    )
                ]
                if not stmts or (len(stmts) == 1 and isinstance(stmts[0], ast.Pass)):
                    violations.append(f"empty_or_pass_function:{node.name}")
                elif (
                    len(stmts) == 1
                    and isinstance(stmts[0], ast.Expr)
                    and isinstance(stmts[0].value, ast.Constant)
                    and stmts[0].value.value is Ellipsis
                ):
                    violations.append(f"ellipsis_function:{node.name}")
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                if isinstance(node.exc.func, ast.Name) and node.exc.func.id == "NotImplementedError":
                    violations.append("not_implemented_raise")
        return _FallbackAuditResult(violations)

    def check_file(self, path: str | Path) -> list[str]:
        """Inspect a Python source file on disk and return detected stub violation strings."""
        return self.audit_python_code(Path(path).read_text(encoding="utf-8")).violations


REPO_ROOT = Path(__file__).resolve().parents[1]
LEAN_SPEC_PATH = REPO_ROOT / "lean4" / "HiveMindSpec.lean"
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "lean4_wsp_theorem_registry.py"
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


def _make_clean_synthetic_lean_spec(*, include_comments: bool = True) -> str:
    """Build a minimal synthetic Lean 4 specification containing all 5 Season 35 theorems."""
    comment_header = (
        "/- outer comment /- nested comment containing sorry admit axiom -/ -/\n"
        "-- line comment mentioning sorry and #print axioms fake\n"
        if include_comments
        else ""
    )
    lines = [
        comment_header,
        "namespace HiveMindSpec",
        f"namespace {SEASON35_NAMESPACE}",
    ]
    for thm in EXPECTED_SEASON35_THEOREMS:
        lines.append(f"theorem {thm} : 1 = 1 := by")
        lines.append("  rfl")
    lines.extend(
        [
            f"end {SEASON35_NAMESPACE}",
            "end HiveMindSpec",
            "",
        ]
    )
    for thm in EXPECTED_SEASON35_THEOREMS:
        lines.append(f"#print axioms {SEASON35_QUALIFIED_PREFIX}.{thm}")
    return "\n".join(lines) + "\n"


def _build_deterministic_cpu_s5_world(seed: int = 20261124) -> dict[str, Any]:
    """Construct a small deterministic CPU MoE state using ``torch.Generator`` for S5 surgery."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_layers = 3
    n_experts = 4
    d_model = 10
    d_mid = 8
    rank = 2
    n_tok = 8

    q_model_raw = torch.randn((d_model, d_model), generator=gen, dtype=torch.float64)
    q_model_t, _ = torch.linalg.qr(q_model_raw)
    q_model = q_model_t.numpy()

    q_mid_raw = torch.randn((d_mid, d_mid), generator=gen, dtype=torch.float64)
    q_mid_t, _ = torch.linalg.qr(q_mid_raw)
    q_mid = q_mid_t.numpy()

    domains = (DEFAULT_GENERAL_DOMAIN, "E1_python_code", "E5_exoplanet")
    spec_domains = domains[1:]

    spec_dirs = {dom: q_model[:, idx] for idx, dom in enumerate(spec_domains)}
    prose_orth = q_model[:, 3]
    prose_center = 1.5 * prose_orth + 0.5 * spec_dirs["E5_exoplanet"]

    noise_x_p = (0.04 * torch.randn((n_tok, d_model), generator=gen, dtype=torch.float64)).numpy()
    noise_m_p = (0.03 * torch.randn((n_tok, 2), generator=gen, dtype=torch.float64)).numpy()

    prose_mid_sub = q_mid[:, :2]
    domain_x: dict[str, np.ndarray] = {DEFAULT_GENERAL_DOMAIN: prose_center[None, :] + noise_x_p}
    domain_mid: dict[str, np.ndarray] = {
        DEFAULT_GENERAL_DOMAIN: 1.4 * np.sum(prose_mid_sub, axis=1)[None, :] + noise_m_p @ prose_mid_sub.T
    }
    domain_tgt: dict[str, np.ndarray] = {DEFAULT_GENERAL_DOMAIN: q_model[:, 4]}

    for idx, dom in enumerate(spec_domains):
        nx = (0.04 * torch.randn((n_tok, d_model), generator=gen, dtype=torch.float64)).numpy()
        nm = (0.03 * torch.randn((n_tok, 2), generator=gen, dtype=torch.float64)).numpy()
        domain_x[dom] = 1.7 * spec_dirs[dom][None, :] + nx
        domain_mid[dom] = 1.5 * q_mid[:, 2 + idx][None, :] + nm @ prose_mid_sub.T
        domain_tgt[dom] = q_model[:, 5 + idx]

    state: dict[str, np.ndarray] = {}
    anchor_layer = 0
    interfering_layers = {1, 2}
    spec_expert_map = {1: "E1_python_code", 2: "E5_exoplanet"}

    for l_idx in range(n_layers):
        rw = np.zeros((n_experts, d_model), dtype=np.float32)
        w2_b = np.zeros((n_experts, d_model, rank), dtype=np.float32)
        w2_a = np.zeros((n_experts, rank, d_mid), dtype=np.float32)
        if l_idx == anchor_layer:
            rw[0] = (2.0 * prose_orth).astype(np.float32)
            w2_b[0, :, 0] = (0.35 * domain_tgt[DEFAULT_GENERAL_DOMAIN]).astype(np.float32)
            w2_a[0, 0, :] = prose_mid_sub[:, 0].astype(np.float32)
            for exp_e, dom in spec_expert_map.items():
                rw[exp_e] = (2.2 * spec_dirs[dom]).astype(np.float32)
                w2_b[exp_e, :, 0] = (0.35 * domain_tgt[dom]).astype(np.float32)
                w2_a[exp_e, 0, :] = q_mid[:, 2 + (exp_e - 1)].astype(np.float32)
        elif l_idx in interfering_layers:
            for sink_e in (0, 3):
                scale = 2.0 if sink_e == 3 else 1.2
                rw[sink_e] = (2.0 * prose_orth).astype(np.float32)
                w2_b[sink_e, :, 0] = (-0.32 * scale * domain_tgt[DEFAULT_GENERAL_DOMAIN]).astype(np.float32)
                w2_a[sink_e, 0, :] = prose_mid_sub[:, 0 if sink_e == 0 else 1].astype(np.float32)
            rw[2] = (2.2 * spec_dirs["E5_exoplanet"] + 1.2 * prose_orth).astype(np.float32)
            w2_b[2, :, 0] = (0.50 * domain_tgt["E5_exoplanet"] - 0.14 * domain_tgt[DEFAULT_GENERAL_DOMAIN]).astype(
                np.float32
            )
            w2_a[2, 0, :] = (q_mid[:, 3] + 0.35 * prose_mid_sub[:, 0]).astype(np.float32)
            rw[1] = (2.2 * spec_dirs["E1_python_code"] + 0.25 * prose_orth).astype(np.float32)
            w2_b[1, :, 0] = (0.42 * domain_tgt["E1_python_code"] - 0.04 * domain_tgt[DEFAULT_GENERAL_DOMAIN]).astype(
                np.float32
            )
            w2_a[1, 0, :] = (q_mid[:, 2] + 0.30 * prose_mid_sub[:, 1]).astype(np.float32)
        state[f"layer_{l_idx}.router_w"] = rw
        state[f"layer_{l_idx}.w2_B"] = w2_b
        state[f"layer_{l_idx}.w2_A"] = w2_a

    base_ln_w = np.ones((d_model,), dtype=np.float32)
    base_ln_b = np.zeros((d_model,), dtype=np.float32)
    state["final_ln_w"] = (1.16 * np.ones((d_model,), dtype=np.float32)).astype(np.float32)
    state["final_ln_b"] = np.zeros((d_model,), dtype=np.float32)
    base_nll = {dom: 4.15 for dom in domains}

    def eval_moe(st: dict[str, np.ndarray] | Any) -> dict[str, float]:
        ln_w = np.asarray(st["final_ln_w"], dtype=np.float64)
        ln_b = np.asarray(st.get("final_ln_b", np.zeros_like(ln_w)), dtype=np.float64)
        cal_pen = 0.22 * ((float(np.mean(ln_w)) - 1.02) ** 2)
        out: dict[str, float] = {}
        for dom in domains:
            x = domain_x[dom]
            mid = domain_mid[dom]
            tgt = domain_tgt[dom]
            residual = np.zeros_like(x, dtype=np.float64)
            for l_idx in range(n_layers):
                rw = np.asarray(st[f"layer_{l_idx}.router_w"], dtype=np.float64)
                wb = np.asarray(st[f"layer_{l_idx}.w2_B"], dtype=np.float64)
                wa = np.asarray(st[f"layer_{l_idx}.w2_A"], dtype=np.float64)
                logits = np.clip(x @ rw.T, -30.0, 30.0)
                gates = 1.0 / (1.0 + np.exp(-logits))
                top_idx = np.argsort(gates, axis=1)[:, -2:]
                mask = np.zeros_like(gates)
                np.put_along_axis(mask, top_idx, 1.0, axis=1)
                active_g = gates * mask
                for e_id in range(n_experts):
                    delta = wb[e_id] @ wa[e_id]
                    residual = residual + active_g[:, e_id : e_id + 1] * (mid @ delta.T)
            normed = residual * ln_w[None, :] + ln_b[None, :]
            align = float(np.mean(normed @ tgt))
            quad_reg = 0.03 * float(np.mean(np.sum(normed * normed, axis=1)))
            out[dom] = float(base_nll[dom] - 0.55 * align + quad_reg + cal_pen)
        return out

    raw_nll = eval_moe(state)
    layer_harm: list[float] = []
    for l_idx in range(n_layers):
        byp = {k: np.array(v, copy=True) for k, v in state.items()}
        byp[f"layer_{l_idx}.w2_B"] = np.zeros_like(byp[f"layer_{l_idx}.w2_B"])
        layer_harm.append(float(raw_nll[DEFAULT_GENERAL_DOMAIN] - eval_moe(byp)[DEFAULT_GENERAL_DOMAIN]))

    exp_harm: list[float] = []
    exp_spec: list[float] = []
    for e_id in range(n_experts):
        zero_e = {k: np.array(v, copy=True) for k, v in state.items()}
        for l_idx in interfering_layers:
            zero_e[f"layer_{l_idx}.w2_B"][e_id] = 0.0
        zn = eval_moe(zero_e)
        exp_harm.append(float(raw_nll[DEFAULT_GENERAL_DOMAIN] - zn[DEFAULT_GENERAL_DOMAIN]))
        exp_spec.append(float(np.mean([zn[d] - raw_nll[d] for d in spec_domains])))

    return {
        "state": state,
        "base_nll": base_nll,
        "eval_fn": eval_moe,
        "layer_harm": layer_harm,
        "exp_harm": exp_harm,
        "exp_spec": exp_spec,
        "layer_prose": {l_idx: domain_x[DEFAULT_GENERAL_DOMAIN] for l_idx in range(n_layers)},
        "layer_spec": {l_idx: {d: domain_x[d] for d in spec_domains} for l_idx in range(n_layers)},
        "tail_mid": {l_idx: domain_mid[DEFAULT_GENERAL_DOMAIN] for l_idx in range(n_layers)},
        "base_ln_w": base_ln_w,
        "base_ln_b": base_ln_b,
    }


# ---------------------------------------------------------------------------
# S1: Real lean4/HiveMindSpec.lean Season 35 Theorem & #print axioms Audit
# ---------------------------------------------------------------------------


def test_s1_real_hivemind_spec_season35_audit_and_print_axioms() -> None:
    """Scenario 1: Audit real lean4/HiveMindSpec.lean for 5 Season 35 theorems and top-level #print axioms."""
    default_path = resolve_default_lean4_spec_path()
    assert default_path == LEAN_SPEC_PATH
    assert resolve_default_lean4_spec_path(REPO_ROOT) == LEAN_SPEC_PATH

    audit = audit_lean4_wsp_spec_file(repo_root=REPO_ROOT)
    assert audit.passed is True
    assert audit.violations == ()
    assert audit.season35_namespace_present is True
    assert audit.namespaces_balanced is True
    assert audit.season35_theorems_declared == EXPECTED_SEASON35_THEOREMS
    assert len(audit.season35_qualified_theorems) == 5
    assert audit.total_spec_theorems >= 171
    assert audit.expected_theorems_missing == ()
    assert audit.duplicate_season35_theorems == ()
    assert audit.missing_print_axioms == ()
    assert audit.nested_print_axioms == ()
    assert audit.season35_sorry_count == 0
    assert audit.season35_admit_count == 0
    assert audit.season35_custom_axiom_count == 0
    assert audit.season35_native_decide_count == 0
    assert audit.season35_sorry_ax_count == 0
    assert audit.global_sorry_count == 0
    assert audit.global_admit_count == 0
    assert audit.global_custom_axiom_count == 0

    for thm in EXPECTED_SEASON35_THEOREMS:
        qual = f"{SEASON35_QUALIFIED_PREFIX}.{thm}"
        assert qual in audit.season35_qualified_theorems
        assert qual in audit.top_level_print_axioms
        assert thm in SEASON35_THEOREM_BY_NAME
        assert SEASON35_THEOREM_BY_NAME[thm].qualified_name == qual

    assert len(SEASON35_THEOREM_CATALOG) == 5
    cat_dicts = [entry.to_dict() for entry in SEASON35_THEOREM_CATALOG]
    assert all(d["qualified_name"].startswith(SEASON35_QUALIFIED_PREFIX) for d in cat_dicts)

    audit_dict = audit.to_dict()
    assert audit_dict["object"] == SPEC_AUDIT_OBJECT
    assert audit_dict["schema"] == SCHEMA_VERSION
    assert audit_dict["passed"] is True
    assert len(audit_dict["spec_sha256"]) == 64
    assert len(audit_dict["season35_section_sha256"]) == 64

    # Default path lookup without explicit repo_root also resolves identically
    audit_default = audit_lean4_wsp_spec_file()
    assert audit_default.spec_sha256 == audit.spec_sha256
    assert audit_default.season35_section_sha256 == audit.season35_section_sha256

    # Nested block comments and string literals must be stripped without breaking line numbers
    raw_nested = '/- outer /- inner sorry -/ still comment -/\n#eval "escaped \\" sorry"\ntheorem ok : 1 = 1 := rfl\n'
    stripped, balanced = strip_lean4_comments_and_strings(raw_nested)
    assert balanced is True
    assert "sorry" not in stripped
    assert stripped.count("\n") == raw_nested.count("\n")


# ---------------------------------------------------------------------------
# S2: Adversarial Rejection on Synthetic Lean 4 Sources
# ---------------------------------------------------------------------------


def test_s2_adversarial_lean_source_rejection() -> None:
    """Scenario 2: Reject sorry, admit, custom axiom, native_decide, missing/nested #print axioms, and scope bugs."""
    clean_src = _make_clean_synthetic_lean_spec(include_comments=True)
    clean_audit = parse_lean4_wsp_spec(clean_src, spec_path="synthetic/CleanSpec.lean")
    assert clean_audit.passed is True
    assert clean_audit.violations == ()

    # 1. Planted sorry inside Season 35 theorem
    sorry_src = clean_src.replace(
        "theorem nullspace_projector_idempotent_scalar : 1 = 1 := by\n  rfl",
        "theorem nullspace_projector_idempotent_scalar : 1 = 1 := by\n  sorry",
    )
    sorry_audit = parse_lean4_wsp_spec(sorry_src)
    assert sorry_audit.passed is False
    assert sorry_audit.season35_sorry_count == 1
    assert any("sorry_detected" in v for v in sorry_audit.violations)

    # 2. Planted admit inside Season 35 theorem
    admit_src = clean_src.replace(
        "theorem preserved_key_zero_drift_under_nullspace_edit : 1 = 1 := by\n  rfl",
        "theorem preserved_key_zero_drift_under_nullspace_edit : 1 = 1 := by\n  admit",
    )
    admit_audit = parse_lean4_wsp_spec(admit_src)
    assert admit_audit.passed is False
    assert admit_audit.season35_admit_count == 1
    assert any("admit_detected" in v for v in admit_audit.violations)

    # 3. Planted custom axiom declaration inside Season 35
    axiom_src = clean_src.replace(
        f"namespace {SEASON35_NAMESPACE}",
        f"namespace {SEASON35_NAMESPACE}\naxiom forged_wsp_postulate : False",
    )
    axiom_audit = parse_lean4_wsp_spec(axiom_src)
    assert axiom_audit.passed is False
    assert axiom_audit.season35_custom_axiom_count == 1
    assert any("custom_axiom_detected" in v for v in axiom_audit.violations)

    # 4. Planted native_decide & sorryAx
    native_src = clean_src.replace("  rfl", "  native_decide", 1)
    native_audit = parse_lean4_wsp_spec(native_src)
    assert native_audit.passed is False
    assert native_audit.season35_native_decide_count == 1
    assert any("native_decide_detected" in v for v in native_audit.violations)

    sorry_ax_src = clean_src.replace("  rfl", "  exact sorryAx _", 1)
    sorry_ax_audit = parse_lean4_wsp_spec(sorry_ax_src)
    assert sorry_ax_audit.passed is False
    assert sorry_ax_audit.season35_sorry_ax_count == 1
    assert any("sorry_ax_detected" in v for v in sorry_ax_audit.violations)

    # 5. Missing top-level #print axioms directive
    target_print = f"#print axioms {SEASON35_QUALIFIED_PREFIX}.single_shot_vault_lock_blocks_second_eval"
    missing_print_src = clean_src.replace(target_print + "\n", "")
    missing_print_audit = parse_lean4_wsp_spec(missing_print_src)
    assert missing_print_audit.passed is False
    assert f"{SEASON35_QUALIFIED_PREFIX}.single_shot_vault_lock_blocks_second_eval" in missing_print_audit.missing_print_axioms
    assert any("missing_top_level_print_axioms" in v for v in missing_print_audit.violations)

    # 6. #print axioms placed nested inside HiveMindSpec instead of top level
    nested_print_src = clean_src.replace(target_print + "\n", "").replace(
        "end HiveMindSpec",
        f"{target_print}\nend HiveMindSpec",
    )
    nested_print_audit = parse_lean4_wsp_spec(nested_print_src)
    assert nested_print_audit.passed is False
    assert f"{SEASON35_QUALIFIED_PREFIX}.single_shot_vault_lock_blocks_second_eval" in nested_print_audit.nested_print_axioms
    assert any("nested_print_axioms_not_top_level" in v for v in nested_print_audit.violations)

    # 7. Missing Season35 namespace
    no_ns_src = "namespace HiveMindSpec\ntheorem other : 1 = 1 := rfl\nend HiveMindSpec\n"
    no_ns_audit = parse_lean4_wsp_spec(no_ns_src)
    assert no_ns_audit.passed is False
    assert no_ns_audit.season35_namespace_present is False
    assert any("missing_namespace" in v for v in no_ns_audit.violations)

    # 8. Duplicate theorem declaration inside Season 35
    dup_src = clean_src.replace(
        f"end {SEASON35_NAMESPACE}",
        "theorem nullspace_projector_idempotent_scalar : 1 = 1 := rfl\n" f"end {SEASON35_NAMESPACE}",
    )
    dup_audit = parse_lean4_wsp_spec(dup_src)
    assert dup_audit.passed is False
    assert "nullspace_projector_idempotent_scalar" in dup_audit.duplicate_season35_theorems

    # 9. Unbalanced namespace, unexpected end, and unterminated block comment
    unclosed_ns = clean_src + "namespace Orphan\n"
    assert parse_lean4_wsp_spec(unclosed_ns).namespaces_balanced is False
    bad_close = clean_src.replace("end HiveMindSpec", "end WrongName")
    assert parse_lean4_wsp_spec(bad_close).namespaces_balanced is False
    unclosed_comment = clean_src + "/- never closed\n"
    assert any("unterminated" in v for v in parse_lean4_wsp_spec(unclosed_comment).violations)
    unclosed_str = clean_src + '#eval "unterminated\n'
    assert any("unterminated" in v for v in parse_lean4_wsp_spec(unclosed_str).violations)

    # 10. Unsupported private theorem modifier inside Season 35
    priv_thm = clean_src.replace(
        "theorem nullspace_projector_idempotent_scalar",
        "private theorem nullspace_projector_idempotent_scalar",
    )
    assert any("unsupported_theorem_declaration_syntax" in v for v in parse_lean4_wsp_spec(priv_thm).violations)


# ---------------------------------------------------------------------------
# S3: Runtime S5 Surgery Receipt -> Lean 4 Theorem Binding & SHA-256 Attestation
# ---------------------------------------------------------------------------


def test_s3_runtime_s5_surgery_receipt_to_lean4_binding_and_attestation() -> None:
    """Scenario 3: Bind CPU S5 AutoOpenWeightResult + SingleShotVaultLock and lineage/composer receipts."""
    world = _build_deterministic_cpu_s5_world(seed=20261124)
    s5_state, s5_res = run_auto_open_weight_s5_synergy(
        world["state"],
        world["base_nll"],
        world["eval_fn"],
        layer_e0_harm_nats=world["layer_harm"],
        expert_e0_harm_nats=world["exp_harm"],
        expert_spec_loss_when_zeroed=world["exp_spec"],
        layer_prose_inputs=world["layer_prose"],
        layer_specialist_inputs=world["layer_spec"],
        tail_layer_prose_ffn_mid=world["tail_mid"],
        base_final_ln_w=world["base_ln_w"],
        base_final_ln_b=world["base_ln_b"],
        config=AutoOpenWeightConfig(
            router_beta=0.85,
            svd_damp_strength=0.85,
            lambda_ln_grid=(0.50, 0.80, 1.00),
            tau_temp_grid=(0.95, 1.00, 1.12),
        ),
    )
    assert s5_res.committed is True
    assert s5_res.verdict == "PASS"

    vault_lock = SingleShotVaultLock("vault_s35_lean4_bridge")
    vault_receipt = vault_lock.evaluate_once(s5_state, world["base_nll"], world["eval_fn"])
    assert vault_receipt["verdict_ip300"] == "PASS"
    with pytest.raises(RuntimeError, match="already evaluated"):
        vault_lock.evaluate_once(s5_state, world["base_nll"], world["eval_fn"])

    attestation = bind_s5_receipt_to_lean4_theorems(
        s5_res,
        repo_root=REPO_ROOT,
        vault_receipt=vault_receipt,
    )
    assert attestation.attestation_valid is True
    assert attestation.spec_audit_passed is True
    assert attestation.theorem_count_verified == 5
    assert attestation.all_lean_theorems_verified is True
    assert attestation.all_runtime_obligations_satisfied is True
    assert attestation.failures == ()
    assert len(attestation.attestation_sha256) == 64
    assert attestation.vault_receipt_sha256 == vault_receipt["vault_receipt_sha256"]

    by_thm = {b.theorem_short_name: b for b in attestation.theorem_bindings}
    assert set(by_thm.keys()) == set(EXPECTED_SEASON35_THEOREMS)
    for thm in EXPECTED_SEASON35_THEOREMS:
        assert by_thm[thm].lean_verified is True
        assert by_thm[thm].runtime_satisfied is True
        assert by_thm[thm].exercised is True

    att_dict = attestation.to_dict()
    assert att_dict["object"] == ATTESTATION_RECEIPT_OBJECT
    assert att_dict["schema"] == SCHEMA_VERSION
    assert att_dict["surgery_receipt_verdict"] == "PASS"

    verified = verify_lean4_attestation_receipt(
        attestation,
        spec_path=LEAN_SPEC_PATH,
        surgery_receipt=s5_res,
        vault_receipt=vault_receipt,
    )
    assert verified.valid is True
    assert verified.attestation_sha256_ok is True
    assert verified.spec_sha256_ok is True
    assert verified.season35_section_sha256_ok is True
    assert verified.spec_audit_ok is True
    assert verified.surgery_receipt_ok is True
    assert verified.vault_receipt_ok is True
    assert verified.to_dict()["valid"] is True

    # Also support binding without vault receipt, lineage-style receipt, and monotone composer wrapper
    att_no_vault = bind_s5_receipt_to_lean4_theorems(s5_res, repo_root=REPO_ROOT)
    assert att_no_vault.attestation_valid is True
    no_vault_by_thm = {b.theorem_short_name: b for b in att_no_vault.theorem_bindings}
    assert no_vault_by_thm["single_shot_vault_lock_blocks_second_eval"].exercised is False
    assert no_vault_by_thm["single_shot_vault_lock_blocks_second_eval"].runtime_satisfied is True

    lineage_core = {
        "object": "hivemind.post_round_wsp_lineage_receipt",
        "wsp_verdict": "PASS",
        "raw_anse_energy": 1000000.0,
        "s5_anse_energy": -0.412,
        "general_rel_drift": 0.001,
        "anchor_layers_bitwise_preserved": True,
        "preserved_specialists": [2],
        "committed_stages": ["S5.2_oof_orthogonal_multigate", "S5.3_tail_layer_qr_svd_nullspace"],
        "active_checkpoint_sha256": "a" * 64,
    }
    lineage_core["receipt_sha256"] = compute_payload_sha256(lineage_core)
    att_lineage = bind_s5_receipt_to_lean4_theorems(lineage_core, repo_root=REPO_ROOT)
    assert att_lineage.attestation_valid is True
    assert verify_lean4_attestation_receipt(att_lineage.to_dict(), surgery_receipt=lineage_core).valid is True

    class _ComposerReceiptContainer:
        def __init__(self, inner: dict[str, Any]) -> None:
            self.receipt = inner

    composer_core = {
        "object": "hivemind.monotone_wsp_pipeline_receipt",
        "passed": True,
        "status": "COMMITTED_IMPROVED",
        "general_domain_drift": 0.002,
        "champion_sha256": "b" * 64,
        "stages": [
            {
                "stage_name": "stage_2_cross_layer_anchor",
                "stage_kind": "cross_layer_anchor_protection",
                "committed": True,
                "pre_anse_energy": -0.10,
                "post_anse_energy": -0.25,
                "delta_anse_energy": -0.15,
                "pre_sha256": "c" * 64,
                "post_sha256": "d" * 64,
            }
        ],
    }
    composer_core["receipt_sha256"] = compute_payload_sha256(composer_core)
    att_comp = bind_s5_receipt_to_lean4_theorems(_ComposerReceiptContainer(composer_core), repo_root=REPO_ROOT)
    assert att_comp.attestation_valid is True

    # Non-PASS receipt rejected when require_pass=True, accepted with failure note if checked under require_pass=False
    fail_receipt = {"committed": False, "anse_energy": 1000000.0, "general_rel_drift": 0.05}
    att_fail = bind_s5_receipt_to_lean4_theorems(fail_receipt, repo_root=REPO_ROOT, require_pass=True)
    assert att_fail.attestation_valid is False
    assert any("surgery_receipt_not_pass" in f for f in att_fail.failures)


# ---------------------------------------------------------------------------
# S4: Tamper Detection on Modified Lean Spec Hash & Surgery Receipts
# ---------------------------------------------------------------------------


def test_s4_tamper_detection_on_modified_lean_spec_and_receipt(tmp_path: Path) -> None:
    """Scenario 4: Detect single-byte Lean spec edits, Season 35 tampering, and receipt alterations."""
    real_spec_text = LEAN_SPEC_PATH.read_text(encoding="utf-8")
    tmp_spec = tmp_path / "HiveMindSpec.lean"
    tmp_spec.write_text(real_spec_text, encoding="utf-8")

    s5_receipt = {
        "verdict": "PASS",
        "anse_energy": -0.35,
        "general_rel_drift": 0.001,
        "checkpoint_sha256": "e" * 64,
        "stages": [
            {
                "stage_name": "S5.3_tail_layer_qr_svd_nullspace",
                "accepted": True,
                "pre_anse_energy": -0.10,
                "post_anse_energy": -0.35,
                "delta_anse_energy": -0.25,
                "pre_sha256": "1" * 64,
                "post_sha256": "e" * 64,
            }
        ],
    }
    s5_receipt["receipt_sha256"] = compute_payload_sha256(s5_receipt)

    audit = audit_lean4_wsp_spec_file(spec_path=tmp_spec)
    att = bind_s5_receipt_to_lean4_theorems(s5_receipt, spec_audit=audit)
    assert att.attestation_valid is True
    assert verify_lean4_attestation_receipt(att, spec_audit=audit, surgery_receipt=s5_receipt).valid is True

    # 1. Tamper 1 comment line outside Season 35 -> spec_sha256 mismatch
    tampered_outside = real_spec_text.replace(
        "Project HiveMind (v2.1)",
        "Project HiveMind (v2.1-tampered)",
        1,
    )
    tmp_spec.write_text(tampered_outside, encoding="utf-8")
    ver_outside = verify_lean4_attestation_receipt(att, spec_path=tmp_spec)
    assert ver_outside.valid is False
    assert ver_outside.spec_sha256_ok is False
    assert ver_outside.season35_section_sha256_ok is True
    assert "spec_sha256_mismatch" in ver_outside.failures

    # 2. Tamper inside Season 35 section -> BOTH spec_sha256 and season35_section_sha256 fail
    tampered_s35 = real_spec_text.replace(
        "isScalarProjector",
        "isScalarProjectorTampered",
    )
    ver_s35 = verify_lean4_attestation_receipt(att, spec_source=tampered_s35)
    assert ver_s35.valid is False
    assert ver_s35.spec_sha256_ok is False
    assert ver_s35.season35_section_sha256_ok is False
    assert "season35_section_sha256_mismatch" in ver_s35.failures

    # 3. Inject sorry into spec -> current_spec_audit fails
    dirty_src = _make_clean_synthetic_lean_spec().replace("  rfl", "  sorry", 1)
    dirty_audit = parse_lean4_wsp_spec(dirty_src)
    with pytest.raises(Lean4SpecAuditError, match="sorry_detected"):
        bind_s5_receipt_to_lean4_theorems(s5_receipt, spec_audit=dirty_audit, strict=True)

    att_non_strict = bind_s5_receipt_to_lean4_theorems(s5_receipt, spec_audit=dirty_audit, strict=False)
    assert att_non_strict.attestation_valid is False
    assert att_non_strict.spec_audit_passed is False
    ver_dirty = verify_lean4_attestation_receipt(att, spec_audit=dirty_audit)
    assert ver_dirty.valid is False
    assert ver_dirty.spec_audit_ok is False

    # 4. Tamper with attestation dict fields directly
    tampered_att = att.to_dict()
    tampered_att["spec_sha256"] = "0" * 64
    ver_att_hash = verify_lean4_attestation_receipt(tampered_att)
    assert ver_att_hash.valid is False
    assert ver_att_hash.attestation_sha256_ok is False
    assert "attestation_sha256_mismatch" in ver_att_hash.failures

    bad_format_att = att.to_dict()
    bad_format_att["attestation_sha256"] = "not-hex"
    assert verify_lean4_attestation_receipt(bad_format_att).attestation_sha256_ok is False

    bad_meta_att = att.to_dict()
    bad_meta_att["object"] = "wrong.object"
    bad_meta_att["schema"] = "wrong.schema"
    bad_meta_att["theorem_bindings"] = "not_a_list"
    ver_meta = verify_lean4_attestation_receipt(bad_meta_att)
    assert ver_meta.valid is False
    assert "invalid_attestation_object" in ver_meta.failures
    assert "invalid_attestation_schema" in ver_meta.failures
    assert "missing_theorem_bindings" in ver_meta.failures

    unverified_bindings_att = att.to_dict()
    unverified_bindings_att["theorem_bindings"] = []
    unverified_bindings_att["attestation_valid"] = False
    ver_unv = verify_lean4_attestation_receipt(unverified_bindings_att)
    assert "attestation_marked_invalid" in ver_unv.failures
    assert any("unverified_bound_theorems" in f for f in ver_unv.failures)

    # 5. Tamper with bound surgery receipt or vault receipt after attestation
    tampered_surgery = dict(s5_receipt)
    tampered_surgery["general_rel_drift"] = 0.009
    ver_surg = verify_lean4_attestation_receipt(att, surgery_receipt=tampered_surgery)
    assert ver_surg.valid is False
    assert ver_surg.surgery_receipt_ok is False
    assert "surgery_receipt_sha256_mismatch" in ver_surg.failures

    vault_ok = {"vault_id": "v1", "guard_ok": True, "vault_locked": True, "vault_access_count": 1}
    vault_ok["vault_receipt_sha256"] = compute_payload_sha256(vault_ok)
    att_with_vault = bind_s5_receipt_to_lean4_theorems(
        s5_receipt, spec_audit=parse_lean4_wsp_spec(_make_clean_synthetic_lean_spec()), vault_receipt=vault_ok
    )
    tampered_vault = dict(vault_ok)
    tampered_vault["vault_access_count"] = 2
    ver_vault = verify_lean4_attestation_receipt(att_with_vault, vault_receipt=tampered_vault)
    assert ver_vault.valid is False
    assert ver_vault.vault_receipt_ok is False


# ---------------------------------------------------------------------------
# S5: Validation Errors, Edge Branches & Security / Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_compliance_gates(tmp_path: Path) -> None:
    """Scenario 5: Input validation errors, edge cases, AGPL-3.0, ZeroStubAudit, PRSecurityScanner, gatekeeper."""
    clean_src = _make_clean_synthetic_lean_spec()

    # 1. Lexer & parser input validation errors
    with pytest.raises(TypeError, match="must be a str"):
        strip_lean4_comments_and_strings(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="NUL bytes"):
        strip_lean4_comments_and_strings("theorem t : True := rfl\x00")
    with pytest.raises(ValueError, match="must not be empty"):
        parse_lean4_wsp_spec("   ")
    with pytest.raises(TypeError, match="must be a str"):
        parse_lean4_wsp_spec(clean_src, spec_path=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="NUL bytes"):
        parse_lean4_wsp_spec(clean_src, spec_path="a\x00b")
    with pytest.raises(ValueError, match="non-empty string"):
        parse_lean4_wsp_spec(clean_src, spec_path="   ")
    with pytest.raises(TypeError, match="non-empty sequence"):
        parse_lean4_wsp_spec(clean_src, expected_theorems="nullspace")
    with pytest.raises(ValueError, match="must not be empty"):
        parse_lean4_wsp_spec(clean_src, expected_theorems=())

    with pytest.raises(FileNotFoundError):
        audit_lean4_wsp_spec_file(spec_path=tmp_path / "missing.lean")
    with pytest.raises(ValueError, match="regular file"):
        audit_lean4_wsp_spec_file(spec_path=tmp_path)

    # 2. Receipt binding & JSON serialization validation errors
    with pytest.raises(TypeError, match="must be a Mapping"):
        bind_s5_receipt_to_lean4_theorems("bad_receipt")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        bind_s5_receipt_to_lean4_theorems({})
    with pytest.raises(TypeError, match="must be a bool"):
        bind_s5_receipt_to_lean4_theorems({"verdict": "PASS"}, require_pass="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a real number"):
        bind_s5_receipt_to_lean4_theorems({"verdict": "PASS"}, general_guard_rel=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite and >= 0.0"):
        bind_s5_receipt_to_lean4_theorems({"verdict": "PASS"}, general_guard_rel=-0.1)
    with pytest.raises(TypeError, match="Lean4WSPSpecAudit"):
        bind_s5_receipt_to_lean4_theorems({"verdict": "PASS"}, spec_audit="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Non-finite float"):
        canonical_json_bytes({"bad": float("nan")})

    # Embedded receipt hash mismatch, non-monotone stage energy, and missing rollback checks
    clean_audit = parse_lean4_wsp_spec(clean_src)
    bad_hash_rec = {"verdict": "PASS", "anse_energy": -0.2, "general_rel_drift": 0.0, "receipt_sha256": "0" * 64}
    att_bad_hash = bind_s5_receipt_to_lean4_theorems(bad_hash_rec, spec_audit=clean_audit)
    assert att_bad_hash.attestation_valid is False
    assert "surgery_receipt:receipt_sha256_mismatch" in att_bad_hash.failures

    bad_fmt_rec = {"verdict": "PASS", "anse_energy": -0.2, "general_rel_drift": 0.0, "receipt_sha256": "short"}
    assert "surgery_receipt:invalid_receipt_sha256_format" in (
        bind_s5_receipt_to_lean4_theorems(bad_fmt_rec, spec_audit=clean_audit).failures
    )

    bad_vault = {"vault_locked": True, "vault_access_count": 2, "vault_receipt_sha256": "0" * 64}
    att_bad_v = bind_s5_receipt_to_lean4_theorems(
        {"verdict": "PASS", "anse_energy": -0.2, "general_rel_drift": 0.0},
        spec_audit=clean_audit,
        vault_receipt=bad_vault,
    )
    assert att_bad_v.attestation_valid is False
    assert "vault_receipt:vault_receipt_sha256_mismatch" in att_bad_v.failures

    # Monotone violation checks via resolve_s5_theorem_obligations
    non_monotone_rec = {
        "verdict": "PASS",
        "general_rel_drift": 0.0,
        "stages": [
            {
                "stage_name": "S5.1",
                "accepted": True,
                "pre_anse_energy": -0.5,
                "post_anse_energy": -0.1,
                "delta_anse_energy": 0.4,
            }
        ],
    }
    obs_nm = resolve_s5_theorem_obligations(non_monotone_rec, spec_audit=clean_audit)
    assert not any(b.runtime_satisfied for b in obs_nm if b.theorem_short_name == "monotone_anse_stage_ratchet_strictly_descends")

    no_rollback_rec = {
        "verdict": "PASS",
        "general_rel_drift": 0.0,
        "stages": [
            {
                "stage_name": "S5.1",
                "accepted": False,
                "pre_sha256": "a" * 64,
                "post_sha256": "b" * 64,
            }
        ],
    }
    obs_nr = resolve_s5_theorem_obligations(no_rollback_rec, spec_audit=clean_audit)
    assert not any(b.runtime_satisfied for b in obs_nr if b.theorem_short_name == "monotone_anse_stage_ratchet_strictly_descends")

    # Missing energy telemetry and status fallback checks
    obs_missing_e = resolve_s5_theorem_obligations({"status": "PENDING"}, spec_audit=clean_audit)
    assert not any(b.runtime_satisfied for b in obs_missing_e if b.theorem_short_name == "monotone_anse_stage_ratchet_strictly_descends")
    att_status = bind_s5_receipt_to_lean4_theorems({"status": "PASS", "anse_energy": -0.1, "general_rel_drift": 0.0}, spec_audit=clean_audit)
    assert att_status.surgery_receipt_verdict == "PASS"

    # Verify TypeError when verify_lean4_attestation_receipt receives invalid spec_audit
    valid_att = bind_s5_receipt_to_lean4_theorems(
        {"verdict": "PASS", "anse_energy": -0.2, "general_rel_drift": 0.0},
        spec_audit=clean_audit,
    )
    with pytest.raises(TypeError, match="Lean4WSPSpecAudit"):
        verify_lean4_attestation_receipt(valid_att, spec_audit="bad")  # type: ignore[arg-type]

    # 3. Fallback stub auditor check_file & audit_python_code exercise
    fallback_auditor = _FallbackZeroStubAudit()
    bad_stub_file = tmp_path / "bad_stub.py"
    bad_stub_file.write_text("def incomplete():\n    pass\n", encoding="utf-8")
    assert len(fallback_auditor.check_file(bad_stub_file)) == 1
    assert fallback_auditor.audit_python_code("def ell():\n    ...\n").is_clean is False
    assert fallback_auditor.audit_python_code("def ni():\n    raise NotImplementedError()\n").is_clean is False

    # 4. AGPL-3.0 header, optional ZeroStubAudit, PRSecurityScanner, lateral movement, pr_gatekeeper FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn24", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn24", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/lean4_wsp_theorem_registry.py", TARGET_MODULE_PATH),
        ("tests/test_lean4_wsp_theorem_registry_scenarios.py", TEST_FILE_PATH),
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

    lean_text = LEAN_SPEC_PATH.read_text(encoding="utf-8")
    changed_files.append(
        {
            "filename": "lean4/HiveMindSpec.lean",
            "status": "modified",
            "additions": 200,
            "patch": "",
            "content": lean_text,
        }
    )

    sample_att_json = json.dumps(valid_att.to_dict(), sort_keys=True)
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+ATTESTATION = {sample_att_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1124,
        pr_title="feat(lean4,swarm): machine-checked Lean 4 Season 35 WSP soundness theorems (0 sorry) and formal spec bridge",
        pr_body="Implements NEURAL-CTRL-DYN-24 Season 35 Lean 4 WSP soundness theorems and Python attestation registry.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(lean4,swarm): machine-checked Lean 4 Season 35 WSP soundness theorems (0 sorry) and formal spec bridge"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []

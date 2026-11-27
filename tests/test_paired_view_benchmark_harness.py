# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Deterministic CPU tests for DAILY-C1-S2: E15-J and P2 paired-view benchmark harness."""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hivemind.jepa.paired_view_benchmark_harness import (
    DEFAULT_MBPP_EVAL_N,
    E15J_RECEIPT_OBJECT,
    E15J_SCHEMA,
    PairedViewHarnessConfig,
    PairedViewRecord,
    build_lean4_paired_record,
    build_paired_view_records,
    build_python_paired_record,
    build_shuffled_pair_derangement,
    build_three_arm_schedule,
    compare_paired_arms,
    compute_exact_mcnemar_p,
    compute_paired_view_alignment_diagnostics,
    compute_payload_sha256,
    compute_step_dropout_mask,
    decontaminate_paired_records,
    dropout_draw,
    evaluate_e15j_p2_promotion_gate,
    extract_lean4_paired_view,
    extract_python_paired_view,
    should_apply_jepa_loss,
    verify_e15j_receipt_provenance,
    verify_zero_overlap_splits,
)
from hivemind.jepa.spec import JepaSpec, apply_aux, step_uniform
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
TARGET_PATH = REPO_ROOT / "src" / "hivemind" / "jepa" / "paired_view_benchmark_harness.py"
TEST_PATH = Path(__file__).resolve()
LEDGER_DOC_PATH = REPO_ROOT / "docs" / "DAILY_ROADMAP_PAPER_LEDGER.md"
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."


def _load_gatekeeper() -> Any:
    spec = importlib.util.spec_from_file_location("pr_gatekeeper", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_sample_records() -> tuple[tuple[PairedViewRecord, ...], tuple[PairedViewRecord, ...]]:
    train_items = [
        {
            "item_id": "mbpp-511",
            "domain": "python_code",
            "split": "train",
            "source_code": (
                "def sum_pair(a: int, b: int) -> int:\n"
                '    """Return the integer sum of two inputs a and b."""\n'
                "    return a + b\n"
            ),
        },
        {
            "item_id": "mbpp-512",
            "domain": "python_code",
            "split": "train",
            "source_code": (
                "async def async_scale(x: float, factor: float) -> float:\n"
                "    return x * factor\n"
            ),
            "docstring_or_spec": "Multiply x by factor asynchronously.",
        },
        {
            "item_id": "lean-train-1",
            "domain": "lean4_proof",
            "split": "train",
            "lean_source": (
                "-- Commutativity helper\n"
                "theorem add_zero_right (n : Nat) : n + 0 = n :=\n"
                "  Nat.add_zero n\n"
            ),
        },
        {
            "item_id": "lean-train-2",
            "domain": "lean4_proof",
            "split": "train",
            "lean_source": (
                "/- Nested /- comment -/ preserved -/\n"
                'lemma id_eq_self (s : String := "a:=b") : s = s := by\n'
                "  rfl\n"
            ),
        },
    ]
    eval_items = [
        {
            "item_id": "mbpp-11",
            "domain": "python_code",
            "split": "eval",
            "source_code": (
                "def square_diff(u: int, v: int) -> int:\n"
                '    """Compute the difference of squares u^2 - v^2."""\n'
                "    return (u - v) * (u + v)\n"
            ),
            "source": "mbpp_sanitized",
        },
        {
            "item_id": "lean-eval-1",
            "domain": "lean4_proof",
            "split": "eval",
            "theorem_signature": "theorem mul_one_left (n : Nat) : 1 * n = n",
            "proof_term": "by simp [Nat.one_mul]",
            "source": "granite_lean4_eval",
        },
    ]
    return build_paired_view_records(train_items), build_paired_view_records(eval_items)


def test_s1_paired_view_builders_and_zero_overlap_splits() -> None:
    """S1: Python/Lean 4 paired-view builders, AST/comment checks, and zero-overlap verification."""
    train_recs, eval_recs = _make_sample_records()
    assert len(train_recs) == 4 and len(eval_recs) == 2
    assert train_recs[0].domain == "python_code"
    assert "docstring" not in train_recs[0].view_b_target
    assert '"""' not in train_recs[0].view_b_target
    assert train_recs[0].view_a_spec == "Return the integer sum of two inputs a and b."
    assert train_recs[2].domain == "lean4_proof"
    assert train_recs[2].view_a_spec == "theorem add_zero_right (n : Nat) : n + 0 = n"
    assert train_recs[2].view_b_target == "Nat.add_zero n"
    # ':=' inside string literal in binder must not split theorem signature prematurely
    assert '"a:=b"' in train_recs[3].view_a_spec
    assert train_recs[3].view_b_target == "by\n  rfl"
    assert isinstance(train_recs[0].to_dict()["pair_sha256"], str)

    # Disjoint split passes cleanly and deterministically
    rep1 = verify_zero_overlap_splits(train_recs, eval_recs)
    rep2 = verify_zero_overlap_splits(train_recs, eval_recs)
    assert rep1.is_disjoint is True
    assert rep1.item_id_overlap == ()
    assert rep1.view_a_overlap_count == 0
    assert rep1.view_b_overlap_count == 0
    assert rep1.pair_sha256_overlap_count == 0
    assert rep1.report_sha256 == rep2.report_sha256

    # Explicit overlap detection (ID, view_a, view_b) & decontamination helper
    leaked_train = (
        *train_recs,
        build_python_paired_record(
            "mbpp-999-dup-spec",
            docstring_or_spec="Compute the difference of squares u^2 - v^2.",
            function_body="def alt(u, v):\n    return u * u - v * v",
            split="train",
        ),
        build_lean4_paired_record(
            "lean-eval-1",
            theorem_signature="theorem other_thm : True",
            proof_term="trivial",
            split="train",
        ),
    )
    leak_report = verify_zero_overlap_splits(leaked_train, eval_recs, raise_on_overlap=False)
    assert leak_report.is_disjoint is False
    assert leak_report.item_id_overlap == ("lean-eval-1",)
    assert leak_report.view_a_overlap_count == 1
    with pytest.raises(ValueError, match="Zero-overlap split violation"):
        verify_zero_overlap_splits(leaked_train, eval_recs, raise_on_overlap=True)

    kept, dropped = decontaminate_paired_records(leaked_train, eval_recs)
    assert dropped == ("mbpp-999-dup-spec", "lean-eval-1")
    assert verify_zero_overlap_splits(kept, eval_recs).is_disjoint is True

    # Python & Lean 4 extractor failure cases (syntax, stubs, sorry holes, missing separator)
    spec_named, body_named = extract_python_paired_view(
        "def first():\n    '''a'''\n    return 1\n\ndef second(x):\n    '''b'''\n    return x + 1\n",
        function_name="second",
    )
    assert spec_named == "b" and "x + 1" in body_named
    for bad_py in (
        "def broken(:",
        "x = 1",
        "def no_doc(x):\n    return x + 1\n",
        "def only_pass(x):\n    '''doc'''\n    pass\n",
        "def only_dots(x):\n    '''doc'''\n    ...\n",
        "def only_nie(x):\n    '''doc'''\n    raise NotImplementedError()\n",
        "def only_nie2(x):\n    '''doc'''\n    raise NotImplementedError\n",
        "def ret_none(x) -> int:\n    '''doc'''\n    return None\n",
        "def only_doc(x):\n    '''doc'''\n",
    ):
        with pytest.raises(ValueError):
            extract_python_paired_view(bad_py)
    with pytest.raises(ValueError):
        extract_python_paired_view("def f():\n    '''d'''\n    return 1\n", function_name="missing")

    sig_l, prf_l = extract_lean4_paired_view('theorem ok (s := "hi") : True := by trivial')
    assert sig_l == 'theorem ok (s := "hi") : True' and prf_l == "by trivial"
    for bad_lean in (
        "theorem unclosed : True := /- never closed",
        "theorem hole : False := by sorry",
        "lemma hole2 : False := by admit",
        "def hole3 : Nat := sorryAx Nat",
        "not_a_decl x := 1",
        "theorem no_sep : True",
        "theorem empty_proof : True :=   ",
    ):
        with pytest.raises(ValueError):
            extract_lean4_paired_view(bad_lean)


def test_s2_hash_deterministic_dropout_and_derangement_control() -> None:
    """S2: sha256(seed, step, row, k) loss dropout, Arm S derangement guarantee, and CPU alignment."""
    spec_drop = JepaSpec(mode="pairs", lam=0.5, dropout=0.5, seed=17)
    spec_off = JepaSpec(mode="none", lam=0.0)
    spec_full = JepaSpec(mode="pairs", lam=1.0, dropout=0.0, seed=17)

    # Exact replay parity with hivemind.jepa.spec.step_uniform & apply_aux
    assert dropout_draw(17, 4, 2, 1) == step_uniform(17, 4, 2, 1, salt="dropout")
    assert [should_apply_jepa_loss(spec_drop, s) for s in range(25)] == [apply_aux(spec_drop, s) for s in range(25)]
    assert should_apply_jepa_loss(spec_off, 0) is False
    assert all(should_apply_jepa_loss(spec_full, 3, row=r, k=1) for r in range(16))

    draws = [should_apply_jepa_loss(spec_drop, step=s // 16, row=s % 16, k=1) for s in range(1600)]
    assert draws == [should_apply_jepa_loss(spec_drop, step=s // 16, row=s % 16, k=1) for s in range(1600)]
    assert abs(sum(draws) / len(draws) - 0.5) < 0.04
    mask_a = compute_step_dropout_mask(spec_drop, step=9, n_rows=12, k_predictor_tokens=1)
    mask_b = compute_step_dropout_mask(spec_drop, step=9, n_rows=12, k_predictor_tokens=1)
    assert mask_a == mask_b and len(mask_a) == 12

    # Derangement guarantee across odd/even sizes, duplicate target repair, and multiple seeds
    train_recs, _ = _make_sample_records()
    # Add two more records where one shares identical target body with another to exercise collision repair
    augmented = (
        *train_recs,
        build_python_paired_record(
            "mbpp-513-same-body",
            docstring_or_spec="Alternative phrasing for summing two integers.",
            function_body=train_recs[0].view_b_target,
            split="train",
        ),
        build_lean4_paired_record(
            "lean-train-3",
            theorem_signature="theorem zero_add_left (n : Nat) : 0 + n = n",
            proof_term="Nat.zero_add n",
            split="train",
        ),
    )
    for seed in (0, 7, 42):
        for step in (0, 3):
            shuf = build_shuffled_pair_derangement(augmented, seed=seed, step=step, k_predictor_tokens=1)
            assert shuf == build_shuffled_pair_derangement(augmented, seed=seed, step=step, k_predictor_tokens=1)
            assert len(shuf) == len(augmented)
            # Exact permutation with zero fixed points and zero identical-target pairings
            assert sorted(s.donor_index for s in shuf) == list(range(len(augmented)))
            for idx, (orig, sh) in enumerate(zip(augmented, shuf, strict=True)):
                assert sh.donor_index != idx
                assert sh.source_target_item_id != orig.item_id
                assert sh.shuffled_view_b_target != orig.view_b_target

    # Different seeds produce distinct shuffled schedules
    shuf_s0 = build_shuffled_pair_derangement(augmented, seed=0, step=0, k_predictor_tokens=1)
    shuf_s99 = build_shuffled_pair_derangement(augmented, seed=99, step=0, k_predictor_tokens=1)
    assert [s.donor_index for s in shuf_s0] != [s.donor_index for s in shuf_s99]

    # Three-arm schedule & CPU NumPy alignment diagnostics (matched vs shuffled Arm S)
    cfg = PairedViewHarnessConfig(jepa_spec=spec_drop, k_predictor_tokens=1)
    sched = build_three_arm_schedule(augmented, config=cfg, step=5)
    assert sched["schema"] == E15J_SCHEMA
    assert set(sched["arms"].keys()) == {"A", "J", "S"}
    assert sched["arms"]["A"]["k_predictor_tokens"] == 0
    assert sched["arms"]["J"]["k_predictor_tokens"] == 1
    assert sched["arms"]["S"]["k_predictor_tokens"] == 1
    assert sched == build_three_arm_schedule(augmented, config=cfg, step=5)

    rng = np.random.default_rng(20261127)
    emb_a = rng.normal(size=(16, 12))
    emb_b = emb_a + 0.05 * rng.normal(size=(16, 12))
    diag = compute_paired_view_alignment_diagnostics(emb_a, emb_b, spec=JepaSpec(mode="pairs", lam=1.0, seed=5))
    assert diag["matched_one_minus_cos"] < 0.05 or diag["alignment_margin"] > 0.5
    assert diag["shuffled_one_minus_cos"] > diag["matched_one_minus_cos"]
    assert diag["alignment_margin"] > 0.4
    assert diag["matched_infonce_loss"] < diag["shuffled_infonce_loss"]

    diag_center_infonce = compute_paired_view_alignment_diagnostics(
        emb_a,
        emb_b,
        spec=JepaSpec(mode="pairs", lam=0.5, center=True, dropout=0.0, seed=5),
        infonce=True,
    )
    assert diag_center_infonce["centered"] is True
    assert diag_center_infonce["infonce"] is True
    assert diag_center_infonce["weighted_aux_loss"] > 0.0

    # Collapse detection & all-dropped step behavior
    collapsed = np.ones((8, 6), dtype=np.float64)
    diag_col = compute_paired_view_alignment_diagnostics(
        collapsed, collapsed, spec=JepaSpec(mode="pairs", lam=1.0, dropout=1.0, seed=1)
    )
    assert diag_col["collapse_detected"] is True
    assert diag_col["active_pairs"] == 0 and diag_col["weighted_aux_loss"] == 0.0


def test_s3_mcnemar_exact_gate_pass_and_boundary_rejections() -> None:
    """S3: Exact McNemar & >= +3.0 pts promotion gate pass and boundary cases (incl. n=257)."""
    # 3a. MBPP-sanitized scale (n = 257): Arm J gains +19 / loses 1 vs Arm A (+7.00 pts, p < 1e-4); Arm S neutral
    ids_257 = [f"mbpp-{11 + i}" for i in range(DEFAULT_MBPP_EVAL_N)]
    arm_a = [i < 169 for i in range(DEFAULT_MBPP_EVAL_N)]  # 169 / 257 = 65.76%
    arm_j = list(arm_a)
    for idx in range(169, 188):
        arm_j[idx] = True  # +19 gained
    arm_j[0] = False  # -1 lost -> net +18 (+7.0039 pts)
    arm_s = list(arm_a)
    arm_s[169] = True
    arm_s[1] = False  # net 0 vs A

    rec_pass = evaluate_e15j_p2_promotion_gate(
        ids_257,
        arm_a,
        arm_j,
        arm_s,
        arm_a_general_nll=3.10,
        arm_j_general_nll=3.12,
        arm_s_general_nll=3.11,
        arm_j_one_minus_cos=0.22,
        domain="python_code",
        experiment_id="E15-J",
    )
    assert rec_pass["verdict"] == "PROMOTED"
    assert rec_pass["promoted"] is True
    assert rec_pass["p2_gate_open"] is True
    assert rec_pass["j_vs_a"]["delta_pts"] > 7.0
    assert rec_pass["j_vs_a"]["mcnemar_p"] == pytest.approx(compute_exact_mcnemar_p(19, 1))
    assert rec_pass["j_vs_a"]["mcnemar_p"] < 1e-4
    assert rec_pass["shuffled_control_gained"] is False
    assert rec_pass["j_beats_shuffled_control"] is True
    assert rec_pass["general_guard_ok"] is True
    assert rec_pass["collapse_guard_ok"] is True
    assert verify_e15j_receipt_provenance(rec_pass) is True

    # 3b. Exact +3.0 pts & McNemar p < 0.05 boundary (n = 200):
    #     6 gained / 0 lost => +3.0 pts, p = 2 / 64 = 0.03125 < 0.05 -> PROMOTED
    ids_200 = [f"lean-{i}" for i in range(200)]
    a_200 = [i < 100 for i in range(200)]
    s_200 = list(a_200)
    j_200_pass = list(a_200)
    for i in range(100, 106):
        j_200_pass[i] = True
    rec_bound_pass = evaluate_e15j_p2_promotion_gate(
        ids_200, a_200, j_200_pass, s_200, domain="lean4_proof", experiment_id="P2-Lean4"
    )
    assert rec_bound_pass["j_vs_a"]["delta_pts"] == pytest.approx(3.0)
    assert rec_bound_pass["j_vs_a"]["mcnemar_p"] == pytest.approx(0.03125)
    assert rec_bound_pass["promoted"] is True

    # 3c. Boundary fail 1: +3.0 pts net (7 gained / 1 lost on n = 200) yet p = 18/256 = 0.0703125 >= 0.05 -> INCONCLUSIVE
    j_200_pfail = list(a_200)
    for i in range(100, 107):
        j_200_pfail[i] = True
    j_200_pfail[0] = False
    rec_pfail = evaluate_e15j_p2_promotion_gate(ids_200, a_200, j_200_pfail, s_200)
    assert rec_pfail["j_vs_a"]["delta_pts"] == pytest.approx(3.0)
    assert rec_pfail["j_vs_a"]["mcnemar_p"] == pytest.approx(0.0703125)
    assert rec_pfail["promoted"] is False
    assert rec_pfail["verdict"] == "INCONCLUSIVE"
    assert "mcnemar_j_vs_a_not_significant" in rec_pfail["rejection_reasons"]

    # 3d. Boundary fail 2: significant p < 0.05 (7 gained / 0 lost on n = 257, p = 0.015625) yet +2.72 pts < +3.0 -> INCONCLUSIVE
    j_257_ptsfail = list(arm_a)
    for i in range(169, 176):
        j_257_ptsfail[i] = True
    rec_ptsfail = evaluate_e15j_p2_promotion_gate(ids_257, arm_a, j_257_ptsfail, arm_a)
    assert rec_ptsfail["j_vs_a"]["delta_pts"] < 3.0
    assert rec_ptsfail["j_vs_a"]["mcnemar_p"] < 0.05
    assert rec_ptsfail["promoted"] is False
    assert rec_ptsfail["verdict"] == "INCONCLUSIVE"
    assert "gain_below_min_pts_threshold" in rec_ptsfail["rejection_reasons"]

    # 3e. Refuted regression (Arm J significantly worse than Arm A) & large-N exact McNemar branch
    j_reg = list(arm_a)
    for i in range(12):
        j_reg[i] = False
    rec_reg = evaluate_e15j_p2_promotion_gate(ids_257, arm_a, j_reg, arm_a)
    assert rec_reg["verdict"] == "REFUTED_REGRESSION"
    assert rec_reg["promoted"] is False
    assert compute_exact_mcnemar_p(0, 0) == 1.0
    assert compute_exact_mcnemar_p(600, 500) < 0.01


def test_s4_shuffled_control_invalidation_guard_and_leakage_gates() -> None:
    """S4: Shuffled-pair control invalidation, general NLL guard, collapse guard, and split leakage."""
    ids = [f"item-{i}" for i in range(100)]
    arm_a = [i < 50 for i in range(100)]
    arm_j = [i < 64 for i in range(100)]  # +14 pts vs A

    # 4a. Arm S (shuffled control) ALSO gains significantly over A (+10 pts, p < 0.05) -> INVALID_SHUFFLED_CONTROL
    arm_s_confounded = [i < 60 for i in range(100)]
    rec_shuf = evaluate_e15j_p2_promotion_gate(ids, arm_a, arm_j, arm_s_confounded)
    assert rec_shuf["verdict"] == "INVALID_SHUFFLED_CONTROL"
    assert rec_shuf["promoted"] is False
    assert rec_shuf["p2_gate_open"] is False
    assert rec_shuf["shuffled_control_gained"] is True
    assert "shuffled_control_gained_significantly" in rec_shuf["rejection_reasons"]
    assert verify_e15j_receipt_provenance(rec_shuf) is True

    # 4b. Arm S gains +3 pts (not significant vs A, p = 0.25), Arm J gains +7 pts (p = 0.0156 vs A),
    #     but J vs S is +4 pts (p = 0.125 >= 0.05) -> fails require_j_beats_s attribution gate
    arm_s_weak = [i < 53 for i in range(100)]
    arm_j_mod = [i < 57 for i in range(100)]
    rec_attr = evaluate_e15j_p2_promotion_gate(ids, arm_a, arm_j_mod, arm_s_weak)
    assert rec_attr["shuffled_control_gained"] is False
    assert rec_attr["j_beats_shuffled_control"] is False
    assert rec_attr["promoted"] is False
    assert "jepa_did_not_beat_shuffled_control" in rec_attr["rejection_reasons"]

    # 4c. General-prose NLL guard exceeded (+0.90 nat) & JEPA cosine collapse (5e-4 < 0.05)
    rec_guard = evaluate_e15j_p2_promotion_gate(
        ids,
        arm_a,
        arm_j,
        arm_a,
        arm_a_general_nll=3.779,
        arm_j_general_nll=4.676,
        arm_j_one_minus_cos=0.0005,
    )
    assert rec_guard["verdict"] == "REJECTED_GUARD_OR_COLLAPSE"
    assert rec_guard["promoted"] is False
    assert rec_guard["general_guard_ok"] is False
    assert rec_guard["collapse_guard_ok"] is False
    assert "general_nll_guard_exceeded" in rec_guard["rejection_reasons"]
    assert "jepa_cosine_collapse_detected" in rec_guard["rejection_reasons"]

    # 4d. Split leakage rejection via SplitIntegrityReport
    train_recs, eval_recs = _make_sample_records()
    leaky_rep = verify_zero_overlap_splits((*train_recs, eval_recs[0]), eval_recs, raise_on_overlap=False)
    rec_leak = evaluate_e15j_p2_promotion_gate(ids, arm_a, arm_j, arm_a, split_integrity=leaky_rep)
    assert rec_leak["verdict"] == "INVALID_SPLIT_LEAKAGE"
    assert rec_leak["promoted"] is False
    assert rec_leak["split_disjoint_ok"] is False


def test_s5_provenance_validation_errors_and_compliance_audits() -> None:
    """S5: Deterministic SHA-256 provenance, boundary error handling, and repo compliance gates."""
    ids = [f"item-{i}" for i in range(60)]
    arm_a = [i < 30 for i in range(60)]
    arm_j = [i < 42 for i in range(60)]
    rec1 = evaluate_e15j_p2_promotion_gate(ids, arm_a, arm_j, arm_a)
    rec2 = evaluate_e15j_p2_promotion_gate(ids, arm_a, arm_j, arm_a)
    assert rec1["schema"] == E15J_SCHEMA
    assert rec1["object"] == E15J_RECEIPT_OBJECT
    assert rec1["receipt_sha256"] == rec2["receipt_sha256"]
    assert verify_e15j_receipt_provenance(rec1) is True

    # Tampered receipts & structural provenance checks
    assert verify_e15j_receipt_provenance("invalid") is False  # type: ignore[arg-type]
    assert verify_e15j_receipt_provenance(dict(rec1, schema="wrong")) is False
    assert verify_e15j_receipt_provenance(dict(rec1, receipt_sha256="badhex")) is False
    assert verify_e15j_receipt_provenance(dict(rec1, n_items=999)) is False
    assert verify_e15j_receipt_provenance(dict(rec1, general_nll_delta=float("nan"))) is False
    mismatch = {k: v for k, v in rec1.items() if k != "receipt_sha256"}
    mismatch["p2_gate_open"] = False
    mismatch["receipt_sha256"] = compute_payload_sha256(mismatch)
    assert verify_e15j_receipt_provenance(mismatch) is False
    bad_prom = {k: v for k, v in rec1.items() if k != "receipt_sha256"}
    bad_prom["shuffled_control_gained"] = True
    bad_prom["receipt_sha256"] = compute_payload_sha256(bad_prom)
    assert verify_e15j_receipt_provenance(bad_prom) is False

    # Input validation paths
    with pytest.raises(TypeError):
        PairedViewHarnessConfig(jepa_spec="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PairedViewHarnessConfig(jepa_spec=JepaSpec(mode="stp", lam=0.1))
    with pytest.raises(TypeError):
        PairedViewHarnessConfig(k_predictor_tokens=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PairedViewHarnessConfig(k_predictor_tokens=-1)
    with pytest.raises(TypeError):
        PairedViewHarnessConfig(min_gain_pts=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PairedViewHarnessConfig(min_gain_pts=float("inf"))
    with pytest.raises(ValueError):
        PairedViewHarnessConfig(mcnemar_alpha=1.0)
    with pytest.raises(TypeError):
        PairedViewHarnessConfig(require_j_beats_s="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PairedViewHarnessConfig(infonce_temperature=0.0)

    with pytest.raises(ValueError):
        PairedViewRecord("id1", "bad_domain", "train", "a", "b")
    with pytest.raises(ValueError):
        PairedViewRecord("id1", "python_code", "bad_split", "a", "b")
    with pytest.raises(ValueError):
        PairedViewRecord("id1", "python_code", "train", "  ", "b")
    with pytest.raises(ValueError):
        build_python_paired_record("id1", source_code="def f():\n    '''d'''\n    return 1", function_body="return 1")
    with pytest.raises(ValueError):
        build_lean4_paired_record("id1", lean_source="theorem t : True := trivial", theorem_signature="sig")
    with pytest.raises(ValueError):
        build_lean4_paired_record("id1", theorem_signature="theorem t : False", proof_term="by sorry")
    with pytest.raises(ValueError):
        build_paired_view_records([])
    with pytest.raises(TypeError):
        build_paired_view_records(["not_mapping"])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        build_paired_view_records(
            [
                {"item_id": "dup", "domain": "python_code", "docstring_or_spec": "a", "function_body": "return 1"},
                {"item_id": "dup", "domain": "python_code", "docstring_or_spec": "b", "function_body": "return 2"},
            ]
        )
    with pytest.raises(ValueError):
        build_paired_view_records([{"item_id": "bad", "domain": "rust"}])

    train_recs, eval_recs = _make_sample_records()
    with pytest.raises(ValueError):
        verify_zero_overlap_splits([], eval_recs)
    with pytest.raises(TypeError):
        verify_zero_overlap_splits(["bad"], eval_recs)  # type: ignore[list-item]
    with pytest.raises(ValueError):
        verify_zero_overlap_splits((train_recs[0], train_recs[0]), eval_recs)

    with pytest.raises(TypeError):
        should_apply_jepa_loss("bad", 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compute_step_dropout_mask("bad", 0, 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_shuffled_pair_derangement(train_recs[:1], seed=0)
    with pytest.raises(TypeError):
        build_shuffled_pair_derangement(["a", "b"], seed=0)  # type: ignore[list-item]
    identical_recs = (
        build_python_paired_record("a", docstring_or_spec="s1", function_body="return 1"),
        build_python_paired_record("b", docstring_or_spec="s2", function_body="return 1"),
    )
    with pytest.raises(ValueError):
        build_shuffled_pair_derangement(identical_recs, seed=0)
    with pytest.raises(ValueError):
        build_three_arm_schedule([])

    arr = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(TypeError):
        compute_paired_view_alignment_diagnostics([[1.0, 1.0], [1.0, 1.0]], arr)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        compute_paired_view_alignment_diagnostics(arr, arr, spec="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compute_paired_view_alignment_diagnostics(np.ones((1, 2)), np.ones((1, 2)))
    with pytest.raises(ValueError):
        compute_paired_view_alignment_diagnostics(np.array([[1.0, math.nan], [1.0, 1.0]]), arr)
    with pytest.raises(ValueError):
        compute_paired_view_alignment_diagnostics(arr, np.ones((2, 3)))
    with pytest.raises(ValueError):
        compute_paired_view_alignment_diagnostics(np.zeros((2, 2)), arr)

    with pytest.raises(TypeError):
        compare_paired_arms("ab", [True, False])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compare_paired_arms([], [])
    with pytest.raises(TypeError):
        compare_paired_arms([1, 0], [True, False])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        compare_paired_arms([True], [True, False])

    with pytest.raises(ValueError):
        evaluate_e15j_p2_promotion_gate(ids, arm_a, arm_j, arm_a, domain="rust")
    with pytest.raises(ValueError):
        evaluate_e15j_p2_promotion_gate([], arm_a, arm_j, arm_a)
    with pytest.raises(ValueError):
        evaluate_e15j_p2_promotion_gate(["dup", "dup"], [True, False], [True, True], [False, False])
    with pytest.raises(ValueError):
        evaluate_e15j_p2_promotion_gate(["a", "b"], [True], [True, True], [False, False])
    with pytest.raises(ValueError):
        evaluate_e15j_p2_promotion_gate(ids, arm_a, arm_j, arm_a, arm_a_general_nll=1.0)
    with pytest.raises(TypeError):
        evaluate_e15j_p2_promotion_gate(ids, arm_a, arm_j, arm_a, split_integrity="bad")  # type: ignore[arg-type]

    # Repository compliance checks: AGPL-3.0, ZeroStubAudit, PRSecurityScanner, FORBIDDEN_ADDED, OwnerOrderGuardrail
    scanner = PRSecurityScanner()
    gk = _load_gatekeeper()
    auditor = ZeroStubAudit() if ZeroStubAudit is not None else _FallbackZeroStubAudit()

    changed_files: list[dict[str, Any]] = []
    for rel, path in (
        ("src/hivemind/jepa/paired_view_benchmark_harness.py", TARGET_PATH),
        ("tests/test_paired_view_benchmark_harness.py", TEST_PATH),
    ):
        src = path.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src[:300]
        audit_res = auditor.audit_python_code(src)
        assert audit_res.is_clean is True, f"Stub audit failed on {rel}: {audit_res.violations}"
        assert scanner.scan_python_ast_security(rel, src) == []
        added_diff = "\n".join("+" + ln for ln in src.splitlines())
        assert gk.forbidden_scan(added_diff) == []
        changed_files.append(
            {"filename": rel, "status": "added", "additions": len(src.splitlines()), "patch": "", "content": src}
        )

    doc_text = LEDGER_DOC_PATH.read_text(encoding="utf-8")
    assert "10.5281/zenodo.24553269" in doc_text
    assert "0.00 EUR / 0.00 USD" in doc_text
    assert "AGPL-3.0" in doc_text
    assert "DAILY-C1-S2" in doc_text
    assert "src/hivemind/jepa/paired_view_benchmark_harness.py" in doc_text
    assert "tests/test_paired_view_benchmark_harness.py" in doc_text
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
            "status": "modified",
            "additions": len(doc_text.splitlines()),
            "patch": "",
            "content": doc_text,
        }
    )
    scan = scanner.audit_pull_request(
        pr_number=1071,
        pr_title="feat(jepa): pre-registered E15-J and P2 paired-view benchmark builder with shuffled-pair control and McNemar gate",
        pr_body="Implements DAILY-C1-S2 (Paper v4.4.0 section 15.4 Opportunity J2, E15-J & P2) under 0.00 EUR / 0.00 USD AGPL-3.0.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(jepa): pre-registered E15-J and P2 paired-view benchmark builder with shuffled-pair control and McNemar gate"],
        author_email=None,
    )
    assert scan.passed is True
    assert scan.blocked_merge is False
    assert scan.findings == []

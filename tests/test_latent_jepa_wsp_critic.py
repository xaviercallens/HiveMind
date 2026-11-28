# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Deterministic CPU tests for DAILY-C1-S3: frozen latent JEPA-WSP critic."""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from hivemind.jepa.latent_jepa_wsp_critic import (
    ANSE_MAXIMUM_PAIN,
    JEPA_WSP_CRITIC_SCHEMA,
    JEPA_WSP_RECEIPT_OBJECT,
    QUEEN_CORTEX_SCHEMA,
    QUEEN_PHEROMONE_OBJECT,
    FrozenLatentJepaPredictor,
    JepaWspCriticConfig,
    build_frozen_jepa_predictor,
    compute_latent_jepa_surprise,
    compute_payload_sha256,
    compute_semantic_foraging_weights,
    compute_state_sha256,
    diagnose_template_gaming,
    emit_jepa_wsp_pheromone_directive,
    evaluate_jepa_wsp_operculation,
    fit_frozen_jepa_predictor_closed_form,
    summarize_opening_vs_midrow_nll,
    verify_jepa_wsp_receipt_provenance,
    verify_pheromone_directive_signature,
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
TARGET_PATH = REPO_ROOT / "src" / "hivemind" / "jepa" / "latent_jepa_wsp_critic.py"
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


def _make_isotropic_pairs(
    seed: int,
    n_samples: int = 32,
    dim: int = 12,
    *,
    noise_scale: float = 0.05,
    mode: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z_ctx = torch.randn(n_samples, dim, generator=gen, dtype=torch.float64)
    q_mat, _ = torch.linalg.qr(torch.randn(dim, dim, generator=gen, dtype=torch.float64))
    if mode == "mlp":
        base_tgt = torch.tanh(z_ctx @ q_mat) + 0.35 * z_ctx
    else:
        base_tgt = z_ctx @ q_mat + 0.15
    z_noisy = base_tgt + noise_scale * torch.randn(n_samples, dim, generator=gen, dtype=torch.float64)
    return z_ctx, base_tgt, z_noisy


def _make_operculation_fixture() -> dict[str, Any]:
    d_lat = 12
    z_ctx_d0, z_tgt_d0, _ = _make_isotropic_pairs(101, 24, d_lat, noise_scale=0.02)
    z_ctx_d1, z_tgt_d1_clean, z_tgt_d1_base = _make_isotropic_pairs(202, 24, d_lat, noise_scale=0.70)
    z_ctx_d2, z_tgt_d2_clean, z_tgt_d2_base = _make_isotropic_pairs(303, 24, d_lat, noise_scale=0.55)

    train_ctx = torch.cat([z_ctx_d0, z_ctx_d1, z_ctx_d2], dim=0)
    train_tgt = torch.cat([z_tgt_d0, z_tgt_d1_clean, z_tgt_d2_clean], dim=0)
    predictor = fit_frozen_jepa_predictor_closed_form(train_ctx, train_tgt, mode="linear", ridge_lambda=1e-4)

    gen = torch.Generator(device="cpu").manual_seed(404)
    z_tgt_d1_improved = z_tgt_d1_clean + 0.10 * torch.randn(24, d_lat, generator=gen, dtype=torch.float64)
    z_tgt_d2_improved = z_tgt_d2_clean + 0.08 * torch.randn(24, d_lat, generator=gen, dtype=torch.float64)

    base_state = {
        "expert_weight": torch.zeros(d_lat, d_lat, dtype=torch.float32),
        "norm_weight": torch.ones(d_lat, dtype=torch.float32),
    }
    good_cand = {
        "expert_weight": 0.15 * torch.eye(d_lat, dtype=torch.float32),
        "norm_weight": torch.full((d_lat,), 1.02, dtype=torch.float32),
    }
    gaming_cand = {
        "expert_weight": -0.25 * torch.eye(d_lat, dtype=torch.float32),
        "norm_weight": torch.full((d_lat,), 0.99, dtype=torch.float32),
    }

    metrics_base = {
        "D0_general_prose": {"nll": 2.400, "midrow_nll": 2.420, "ece": 0.030},
        "D1_code_python": {"nll": 2.950, "midrow_nll": 3.010, "ece": 0.055},
        "D2_formal_lean4": {"nll": 3.120, "midrow_nll": 3.180, "ece": 0.060},
    }
    metrics_good = {
        "D0_general_prose": {"nll": 2.395, "midrow_nll": 2.415, "ece": 0.029},
        "D1_code_python": {"nll": 2.680, "midrow_nll": 2.750, "ece": 0.042},
        "D2_formal_lean4": {"nll": 2.860, "midrow_nll": 2.930, "ece": 0.046},
    }
    # Paper section 14 81% format artifact: full drops -0.331 while midrow drops only -0.062
    metrics_gaming = {
        "D0_general_prose": {"nll": 2.400, "midrow_nll": 2.420, "ece": 0.030},
        "D1_code_python": {"nll": 2.619, "midrow_nll": 2.948, "ece": 0.052},
        "D2_formal_lean4": {"nll": 2.789, "midrow_nll": 3.118, "ece": 0.058},
    }
    latents_base = {
        "D0_general_prose": (z_ctx_d0, z_tgt_d0),
        "D1_code_python": (z_ctx_d1, z_tgt_d1_base),
        "D2_formal_lean4": (z_ctx_d2, z_tgt_d2_base),
    }
    latents_good = {
        "D0_general_prose": (z_ctx_d0, z_tgt_d0),
        "D1_code_python": (z_ctx_d1, z_tgt_d1_improved),
        "D2_formal_lean4": (z_ctx_d2, z_tgt_d2_improved),
    }
    gen_game = torch.Generator(device="cpu").manual_seed(707)
    latents_gaming = {
        "D0_general_prose": (z_ctx_d0, z_tgt_d0),
        "D1_code_python": (
            z_ctx_d1,
            z_tgt_d1_base + 0.35 * torch.randn(24, d_lat, generator=gen_game, dtype=torch.float64),
        ),
        "D2_formal_lean4": (
            z_ctx_d2,
            z_tgt_d2_base + 0.30 * torch.randn(24, d_lat, generator=gen_game, dtype=torch.float64),
        ),
    }
    return {
        "predictor": predictor,
        "base_state": base_state,
        "good_cand": good_cand,
        "gaming_cand": gaming_cand,
        "metrics_base": metrics_base,
        "metrics_good": metrics_good,
        "metrics_gaming": metrics_gaming,
        "latents_base": latents_base,
        "latents_good": latents_good,
        "latents_gaming": latents_gaming,
    }


def test_s1_frozen_predictor_semantic_surprise_and_rankme_isotropy() -> None:
    """S1: Frozen linear/MLP predictor, stop-grad E_JEPA surprise, and RankMe isotropy."""
    z_ctx_lin, z_tgt_lin, z_tgt_noisy = _make_isotropic_pairs(11, 48, 10, noise_scale=0.45, mode="linear")
    lin_pred = fit_frozen_jepa_predictor_closed_form(z_ctx_lin, z_tgt_lin, mode="linear", ridge_lambda=1e-6)
    assert lin_pred.ctx_dim == 10 and lin_pred.target_dim == 10
    assert lin_pred.is_frozen is True

    # Seeded constructors + all MLP activations + residual path
    seeded_lin = build_frozen_jepa_predictor(10, 10, mode="linear", seed=7)
    seeded_mlp = build_frozen_jepa_predictor(10, 10, mode="mlp", hidden_dim=20, activation="silu", residual=True, seed=7)
    assert seeded_lin.is_frozen is True and seeded_mlp.is_frozen is True
    assert seeded_lin.predictor_sha256 == build_frozen_jepa_predictor(10, 10, mode="linear", seed=7).predictor_sha256

    z_ctx_mlp, z_tgt_mlp, _ = _make_isotropic_pairs(17, 64, 8, noise_scale=0.01, mode="mlp")
    for act in ("gelu", "silu", "tanh"):
        mlp_pred = fit_frozen_jepa_predictor_closed_form(
            z_ctx_mlp, z_tgt_mlp, mode="mlp", hidden_dim=32, activation=act, residual=True, ridge_lambda=1e-4, seed=9
        )
        assert mlp_pred.is_frozen is True
        rep_mlp = compute_latent_jepa_surprise(mlp_pred, z_ctx_mlp, z_tgt_mlp)
        assert rep_mlp.mean_l2_sq_surprise < 0.20

    # Stop-gradient sg(z_target) & zero backprop through frozen predictor
    z_ctx_req = z_ctx_lin.clone().requires_grad_(True)
    z_tgt_req = z_tgt_lin.clone().requires_grad_(True)
    rep_clean = compute_latent_jepa_surprise(lin_pred, z_ctx_req, z_tgt_req)
    rep_noisy = compute_latent_jepa_surprise(lin_pred, z_ctx_lin, z_tgt_noisy)
    assert rep_clean.stopgrad_enforced is True
    assert z_ctx_req.grad is None and z_tgt_req.grad is None
    assert rep_clean.mean_l2_sq_surprise < 1e-4
    assert rep_noisy.mean_l2_sq_surprise > 1.0
    assert rep_clean.rms_l2_surprise == pytest.approx(math.sqrt(rep_clean.mean_l2_sq_surprise))
    assert rep_clean.pred_rankme_metrics["is_collapsed"] is False
    assert rep_clean.target_rankme_metrics["is_collapsed"] is False
    summary = rep_clean.to_summary_dict()
    assert summary["stopgrad_enforced"] is True and summary["target_collapsed"] is False

    # Rank-1 cosine cone collapse and anisotropy spike detection (plus predictor collapse branch)
    gen = torch.Generator(device="cpu").manual_seed(29)
    cone_tgt = torch.ones((24, 10), dtype=torch.float64) + 1e-4 * torch.randn(24, 10, generator=gen, dtype=torch.float64)
    rep_cone = compute_latent_jepa_surprise(
        lin_pred,
        z_ctx_lin[:24],
        cone_tgt,
        reference_target_report=rep_clean.target_rankme_report,
    )
    assert rep_cone.target_rankme_metrics["is_collapsed"] is True
    assert "anisotropy_spike_vs_reference" in rep_cone.target_rankme_metrics["collapse_warnings"]
    assert "rankme_drop_vs_reference" in rep_cone.target_rankme_metrics["collapse_warnings"]
    collapsed_pred = build_frozen_jepa_predictor(
        10, 10, w1=torch.zeros(10, 10, dtype=torch.float64), b1=torch.ones(10, dtype=torch.float64)
    )
    rep_pred_col = compute_latent_jepa_surprise(collapsed_pred, z_ctx_lin[:24], z_tgt_lin[:24])
    diag_pred_col = diagnose_template_gaming(
        base_surface_nll=2.0,
        cand_surface_nll=2.0,
        base_surprise=rep_clean,
        cand_surprise=rep_pred_col,
    )
    assert diag_pred_col.pred_collapsed is True
    assert "predictor_rankme_or_isotropy_collapse" in diag_pred_col.gaming_reasons


def test_s2_reject_opening_template_ce_gaming_81pct_format_artifact() -> None:
    """S2: Reject Paper section 14 81% opening-template CE gaming with ANSE_MAXIMUM_PAIN rollback."""
    # Exact Paper §14 reproduction: full chunk improves -0.331 nat while token 64+ mid-row improves -0.062 nat
    # Construct deterministic token NLL matrices (N=16, seq=128, prefix_len=64) matching -0.331 full vs -0.062 mid-row
    base_tok_nll = torch.full((16, 128), 3.0, dtype=torch.float64)
    cand_tok_nll = torch.full((16, 128), 3.0, dtype=torch.float64)
    cand_tok_nll[:, :64] = 3.0 - 0.600
    cand_tok_nll[:, 64:] = 3.0 - 0.062
    tok_summary = summarize_opening_vs_midrow_nll(base_tok_nll, cand_tok_nll, prefix_len=64)
    assert tok_summary["full_nll_gain"] == pytest.approx(0.331, abs=1e-9)
    assert tok_summary["midrow_nll_gain"] == pytest.approx(0.062, abs=1e-9)
    assert tok_summary["format_artifact_share"] == pytest.approx((0.331 - 0.062) / 0.331, abs=1e-6)
    assert tok_summary["format_artifact_share"] > 0.80

    fx = _make_operculation_fixture()
    state = {k: v.clone() for k, v in fx["base_state"].items()}
    init_sha = compute_state_sha256(state)

    receipt = evaluate_jepa_wsp_operculation(
        state,
        fx["gaming_cand"],
        fx["predictor"],
        fx["metrics_base"],
        fx["metrics_gaming"],
        fx["latents_base"],
        fx["latents_gaming"],
        campaign_id="jepa-wsp-s2-gaming",
        round_index=2,
    )
    assert receipt["accepted"] is False
    assert receipt["rolled_back"] is True
    assert receipt["rollback_verified"] is True
    assert receipt["template_gaming_rejected"] is True
    assert receipt["rejection_reason"] == "opening_template_ce_gaming_detected"
    assert receipt["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert receipt["final_state_sha256"] == init_sha
    assert receipt["pooled_template_diagnosis"]["format_artifact_share"] > 0.80
    assert any("opening_template_format_artifact" in r for r in receipt["gaming_reasons"])
    assert any("semantic_surprise_stagnation_or_increase" in r for r in receipt["gaming_reasons"])
    assert verify_jepa_wsp_receipt_provenance(receipt) is True

    # Even without explicit midrow_nll, JEPA surprise stagnation catches CE gaming!
    no_mid_base = {
        d: {"nll": row["nll"], "ece": row["ece"]} for d, row in fx["metrics_base"].items()
    }
    no_mid_gaming = {
        d: {"nll": row["nll"], "ece": row["ece"]} for d, row in fx["metrics_gaming"].items()
    }
    state_nomid = {k: v.clone() for k, v in fx["base_state"].items()}
    rec_nomid = evaluate_jepa_wsp_operculation(
        state_nomid,
        fx["gaming_cand"],
        fx["predictor"],
        no_mid_base,
        no_mid_gaming,
        fx["latents_base"],
        fx["latents_gaming"],
    )
    assert rec_nomid["accepted"] is False
    assert rec_nomid["template_gaming_rejected"] is True
    assert rec_nomid["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert rec_nomid["final_state_sha256"] == init_sha


def test_s3_semantic_deficit_pheromone_steering_and_simplex_conservation() -> None:
    """S3: Semantic-deficit Pheromones Royales steering and simplex conservation sum(p_d) = 1."""
    # Domain 'code_python' has lower surface NLL (gamed template) yet highest semantic surprise E_JEPA
    # Domain 'format_boilerplate' has higher surface NLL yet lowest semantic surprise E_JEPA
    domain_surprise = {
        "D0_general_prose": 0.15,
        "code_python": 4.20,
        "formal_lean4": 3.10,
        "math_reasoning": 1.80,
        "format_boilerplate": 0.45,
    }
    domain_midrow = {
        "D0_general_prose": 2.30,
        "code_python": 3.40,
        "formal_lean4": 3.10,
        "math_reasoning": 2.50,
        "format_boilerplate": 1.20,
    }
    domain_surface = {
        "D0_general_prose": 2.30,
        "code_python": 1.10,
        "formal_lean4": 2.40,
        "math_reasoning": 2.20,
        "format_boilerplate": 3.90,
    }
    domain_eces = {
        "D0_general_prose": 0.02,
        "code_python": 0.06,
        "formal_lean4": 0.05,
        "math_reasoning": 0.04,
        "format_boilerplate": 0.01,
    }

    weights, priorities = compute_semantic_foraging_weights(
        domain_surprise,
        domain_midrow_nll=domain_midrow,
        domain_surface_nll=domain_surface,
        domain_eces=domain_eces,
    )
    assert weights["D0_general_prose"] == 0.0
    for dom in ("code_python", "formal_lean4", "math_reasoning", "format_boilerplate"):
        assert weights[dom] > 0.0
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)
    # Order preservation: highest semantic surprise outranks superficial high-surface-NLL boilerplate
    assert weights["code_python"] > weights["formal_lean4"] > weights["math_reasoning"] > weights["format_boilerplate"]
    assert priorities[0]["domain_id"] == "code_python"

    # Full QueenPheromoneDirective with UCB counts, anchor/corruptor classification, and frozen router mode
    directive = emit_jepa_wsp_pheromone_directive(
        domain_surprise,
        domain_midrow_nll=domain_midrow,
        domain_surface_nll=domain_surface,
        domain_eces=domain_eces,
        domain_round_counts={"code_python": 1, "formal_lean4": 0, "math_reasoning": 4, "format_boilerplate": 10},
        expert_causal_gains_nats={"expert_0": -0.02, "expert_1": 0.01, "expert_2": -0.004},
        expert_prose_harm_nats={"expert_0": 0.0001, "expert_1": 0.005, "expert_2": 0.0002},
        router_expert_kl_mismatch={"layer_0": 0.08, "layer_1": 0.02},
        campaign_id="laruche-j3-s3",
        round_index=4,
    )
    assert directive.object == QUEEN_PHEROMONE_OBJECT
    assert directive.schema == QUEEN_CORTEX_SCHEMA
    assert directive.anchor_experts_to_freeze == ("expert_0",)
    assert directive.corruptor_experts_to_sink_or_reseed == ("expert_1",)
    assert directive.active_specialist_experts == ("expert_2",)
    assert directive.router_training_mode == "router_frozen"
    assert sum(directive.bee_foraging_domain_weights.values()) == pytest.approx(1.0, abs=1e-12)
    assert verify_pheromone_directive_signature(directive) is True
    assert verify_pheromone_directive_signature(directive.to_dict()) is True

    # Single-group fallback (when no specialist prefix is omitted, all domains share simplex)
    only_gen_weights, _ = compute_semantic_foraging_weights({"D0_a": 1.0, "D0_b": 2.0})
    assert sum(only_gen_weights.values()) == pytest.approx(1.0, abs=1e-12)
    assert only_gen_weights["D0_b"] > only_gen_weights["D0_a"] > 0.0


def test_s4_operculation_commit_vs_atomic_bitwise_rollback_gates() -> None:
    """S4: Genuine conceptual improvement commits vs RankMe, D0 guard, non-improving, and NaN rollbacks."""
    fx = _make_operculation_fixture()

    # 4a. Genuine improvement on nn.Module -> accepted & committed bitwise
    mod = nn.Linear(12, 12)
    cand_mod = nn.Linear(12, 12)
    with torch.no_grad():
        mod.weight.zero_()
        mod.bias.zero_()
        cand_mod.weight.copy_(0.25 * torch.eye(12))
        cand_mod.bias.fill_(0.01)
    init_mod_sha = compute_state_sha256(mod.state_dict())
    expected_cand_sha = compute_state_sha256(cand_mod.state_dict())

    rec_pass = evaluate_jepa_wsp_operculation(
        mod,
        cand_mod,
        fx["predictor"],
        fx["metrics_base"],
        fx["metrics_good"],
        fx["latents_base"],
        fx["latents_good"],
        campaign_id="jepa-wsp-s4-pass",
        round_index=1,
    )
    assert rec_pass["accepted"] is True
    assert rec_pass["rolled_back"] is False
    assert rec_pass["rollback_verified"] is True
    assert rec_pass["template_gaming_rejected"] is False
    assert rec_pass["collapse_rejected"] is False
    assert rec_pass["guard_rejected"] is False
    assert rec_pass["anse_energy"] < 0.0
    assert rec_pass["jepa_surprise_diff"] < 0.0
    assert rec_pass["final_state_sha256"] == expected_cand_sha
    assert rec_pass["final_state_sha256"] != init_mod_sha
    assert rec_pass["foraging_weights_sum"] == pytest.approx(1.0, abs=1e-12)
    assert verify_jepa_wsp_receipt_provenance(rec_pass) is True

    # 4b. RankMe / anisotropy collapse -> atomic bitwise rollback
    state_col = {k: v.clone() for k, v in fx["base_state"].items()}
    sha_col = compute_state_sha256(state_col)
    gen = torch.Generator(device="cpu").manual_seed(55)
    collapsed_tgt = torch.ones(24, 12, dtype=torch.float64) + 1e-4 * torch.randn(24, 12, generator=gen, dtype=torch.float64)
    lat_col = {
        "D0_general_prose": fx["latents_good"]["D0_general_prose"],
        "D1_code_python": (fx["latents_good"]["D1_code_python"][0], collapsed_tgt),
        "D2_formal_lean4": fx["latents_good"]["D2_formal_lean4"],
    }
    rec_col = evaluate_jepa_wsp_operculation(
        state_col,
        fx["good_cand"],
        fx["predictor"],
        fx["metrics_base"],
        fx["metrics_good"],
        fx["latents_base"],
        lat_col,
    )
    assert rec_col["accepted"] is False
    assert rec_col["rolled_back"] is True
    assert rec_col["collapse_rejected"] is True
    assert rec_col["rejection_reason"] == "rankme_or_isotropy_collapse_detected"
    assert rec_col["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert rec_col["final_state_sha256"] == sha_col

    # 4c. General-prose drift > +1.0% -> atomic bitwise rollback
    state_d0 = {k: v.clone() for k, v in fx["base_state"].items()}
    sha_d0 = compute_state_sha256(state_d0)
    metrics_d0_breach = {
        "D0_general_prose": {"nll": 2.480, "midrow_nll": 2.490, "ece": 0.030},
        "D1_code_python": fx["metrics_good"]["D1_code_python"],
        "D2_formal_lean4": fx["metrics_good"]["D2_formal_lean4"],
    }
    rec_d0 = evaluate_jepa_wsp_operculation(
        state_d0,
        fx["good_cand"],
        fx["predictor"],
        fx["metrics_base"],
        metrics_d0_breach,
        fx["latents_base"],
        fx["latents_good"],
    )
    assert rec_d0["accepted"] is False
    assert rec_d0["guard_rejected"] is True
    assert rec_d0["rejection_reason"] == "general_prose_or_domain_guard_exceeded"
    assert rec_d0["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert rec_d0["final_state_sha256"] == sha_d0

    # 4d. Non-improving energy -> atomic bitwise rollback
    state_flat = {k: v.clone() for k, v in fx["base_state"].items()}
    sha_flat = compute_state_sha256(state_flat)
    rec_flat = evaluate_jepa_wsp_operculation(
        state_flat,
        fx["good_cand"],
        fx["predictor"],
        fx["metrics_base"],
        fx["metrics_base"],
        fx["latents_base"],
        fx["latents_base"],
    )
    assert rec_flat["accepted"] is False
    assert rec_flat["rejection_reason"] == "anse_jepa_wsp_energy_non_improving"
    assert rec_flat["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert rec_flat["final_state_sha256"] == sha_flat

    # 4e. Non-finite candidate weights -> atomic bitwise rollback
    state_nan = {k: v.clone() for k, v in fx["base_state"].items()}
    sha_nan = compute_state_sha256(state_nan)
    bad_cand = {
        "expert_weight": torch.full_like(state_nan["expert_weight"], float("nan")),
        "norm_weight": state_nan["norm_weight"].clone(),
    }
    rec_nan = evaluate_jepa_wsp_operculation(
        state_nan,
        bad_cand,
        fx["predictor"],
        fx["metrics_base"],
        fx["metrics_good"],
        fx["latents_base"],
        fx["latents_good"],
    )
    assert rec_nan["accepted"] is False
    assert rec_nan["rejection_reason"] == "non_finite_candidate_weights"
    assert rec_nan["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert rec_nan["final_state_sha256"] == sha_nan


def test_s5_provenance_validation_boundaries_and_compliance_gates() -> None:
    """S5: Deterministic SHA-256 provenance, boundary validation, and repository compliance gates."""
    fx = _make_operculation_fixture()
    s1 = {k: v.clone() for k, v in fx["base_state"].items()}
    s2 = {k: v.clone() for k, v in fx["base_state"].items()}

    r1 = evaluate_jepa_wsp_operculation(
        s1,
        fx["good_cand"],
        fx["predictor"],
        fx["metrics_base"],
        fx["metrics_good"],
        fx["latents_base"],
        fx["latents_good"],
    )
    r2 = evaluate_jepa_wsp_operculation(
        s2,
        fx["good_cand"],
        fx["predictor"],
        fx["metrics_base"],
        fx["metrics_good"],
        fx["latents_base"],
        fx["latents_good"],
    )
    assert r1["schema"] == JEPA_WSP_CRITIC_SCHEMA
    assert r1["object"] == JEPA_WSP_RECEIPT_OBJECT
    assert r1["receipt_sha256"] == r2["receipt_sha256"]
    assert r1["final_state_sha256"] == r2["final_state_sha256"]

    # Provenance verification failure modes
    assert verify_jepa_wsp_receipt_provenance("bad") is False  # type: ignore[arg-type]
    assert verify_jepa_wsp_receipt_provenance(dict(r1, schema="wrong")) is False
    assert verify_jepa_wsp_receipt_provenance(dict(r1, receipt_sha256="short")) is False
    assert verify_jepa_wsp_receipt_provenance(dict(r1, nll_diff=-99.0)) is False
    assert verify_jepa_wsp_receipt_provenance(dict(r1, anse_energy=float("nan"))) is False

    def _resign(patch: dict[str, Any]) -> dict[str, Any]:
        obj = {k: v for k, v in r1.items() if k != "receipt_sha256"}
        obj.update(patch)
        obj["receipt_sha256"] = compute_payload_sha256(obj)
        return obj

    assert verify_jepa_wsp_receipt_provenance(_resign({"pheromone_directive": {"bad": 1}})) is False
    bad_w = dict(r1["pheromone_directive"])
    bad_w["bee_foraging_domain_weights"] = {"D1_code_python": 0.4}
    del bad_w["directive_sha256"]
    bad_w["directive_sha256"] = compute_payload_sha256(bad_w)
    assert verify_jepa_wsp_receipt_provenance(_resign({"pheromone_directive": bad_w})) is False
    assert verify_jepa_wsp_receipt_provenance(_resign({"accepted": True, "rolled_back": True})) is False
    assert verify_jepa_wsp_receipt_provenance(_resign({"accepted": False, "rolled_back": True})) is False
    assert verify_pheromone_directive_signature({"directive_sha256": "0" * 64, "object": "bad"}) is False
    assert verify_pheromone_directive_signature({"directive_sha256": "bad"}) is False
    assert verify_pheromone_directive_signature({"directive_sha256": "0" * 64, "object": QUEEN_PHEROMONE_OBJECT, "schema": QUEEN_CORTEX_SCHEMA, "nan": float("nan")}) is False
    with pytest.raises(TypeError):
        verify_pheromone_directive_signature("bad")  # type: ignore[arg-type]

    # Boundary & type validation across public API
    with pytest.raises(TypeError):
        JepaWspCriticConfig(enforce_all_domain_guard="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        JepaWspCriticConfig(max_general_drift_ratio=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        JepaWspCriticConfig(general_prefixes=())
    with pytest.raises(ValueError):
        JepaWspCriticConfig(foraging_temperature=0.0)
    with pytest.raises(ValueError):
        JepaWspCriticConfig(max_format_artifact_share=1.5)
    with pytest.raises(ValueError):
        JepaWspCriticConfig(max_general_drift_ratio=float("nan"))

    with pytest.raises(ValueError):
        FrozenLatentJepaPredictor(mode="bad", w1=torch.eye(2))
    with pytest.raises(ValueError):
        FrozenLatentJepaPredictor(mode="linear", w1=torch.eye(2), activation="bad")
    with pytest.raises(ValueError):
        FrozenLatentJepaPredictor(mode="linear", w1=torch.eye(2), b1=torch.zeros(3))
    with pytest.raises(ValueError):
        FrozenLatentJepaPredictor(mode="linear", w1=torch.eye(2), w2=torch.eye(2))
    with pytest.raises(ValueError):
        FrozenLatentJepaPredictor(mode="mlp", w1=torch.eye(2))
    with pytest.raises(ValueError):
        FrozenLatentJepaPredictor(mode="mlp", w1=torch.eye(2), w2=torch.eye(3))
    with pytest.raises(ValueError):
        FrozenLatentJepaPredictor(mode="mlp", w1=torch.eye(2), w2=torch.eye(2), b2=torch.zeros(3))
    with pytest.raises(ValueError):
        FrozenLatentJepaPredictor(mode="linear", w1=torch.ones(2, 3), residual=True)

    with pytest.raises(TypeError):
        build_frozen_jepa_predictor(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_frozen_jepa_predictor(0)
    with pytest.raises(ValueError):
        build_frozen_jepa_predictor(4, mode="bad")
    with pytest.raises(ValueError):
        build_frozen_jepa_predictor(4, activation="bad")
    with pytest.raises(ValueError):
        build_frozen_jepa_predictor(4, w1=torch.eye(4)).predict(torch.ones(2, 3))
    with pytest.raises(TypeError):
        build_frozen_jepa_predictor(4, w1="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_frozen_jepa_predictor(4, w1=torch.ones(4, 4, dtype=torch.long))
    with pytest.raises(ValueError):
        build_frozen_jepa_predictor(4, w1=torch.empty(0, 4))
    with pytest.raises(ValueError):
        build_frozen_jepa_predictor(4, w1=torch.tensor([[float("nan"), 0.0]]))

    with pytest.raises(ValueError):
        fit_frozen_jepa_predictor_closed_form(torch.ones(4, 2), torch.ones(3, 2))
    with pytest.raises(ValueError):
        fit_frozen_jepa_predictor_closed_form(torch.ones(4, 2), torch.ones(4, 2), mode="bad")
    with pytest.raises(ValueError):
        fit_frozen_jepa_predictor_closed_form(torch.ones(4, 2), torch.ones(4, 2), activation="bad")
    with pytest.raises(ValueError):
        fit_frozen_jepa_predictor_closed_form(torch.ones(4, 2), torch.ones(4, 3), residual=True)

    with pytest.raises(TypeError):
        compute_latent_jepa_surprise("bad", torch.ones(2, 4), torch.ones(2, 4))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compute_latent_jepa_surprise(fx["predictor"], torch.ones(2, 12), torch.ones(2, 6))
    with pytest.raises(ValueError):
        summarize_opening_vs_midrow_nll(torch.ones(2, 8), torch.ones(2, 9))
    with pytest.raises(ValueError):
        summarize_opening_vs_midrow_nll(torch.ones(2, 8), torch.ones(2, 8), prefix_len=8)
    with pytest.raises(ValueError):
        diagnose_template_gaming(
            base_surface_nll=1.0,
            cand_surface_nll=0.9,
            base_surprise=compute_latent_jepa_surprise(fx["predictor"], *fx["latents_base"]["D1_code_python"]),
            cand_surprise=compute_latent_jepa_surprise(fx["predictor"], *fx["latents_good"]["D1_code_python"]),
            base_midrow_nll=1.0,
        )
    with pytest.raises(ValueError):
        compute_semantic_foraging_weights({})
    with pytest.raises(TypeError):
        compute_semantic_foraging_weights({"a": 1.0}, domain_midrow_nll="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compute_semantic_foraging_weights({"a": 1.0}, domain_midrow_nll={"b": 1.0})
    with pytest.raises(TypeError):
        compute_semantic_foraging_weights({"a": 1.0}, domain_round_counts="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        emit_jepa_wsp_pheromone_directive({"a": 1.0}, router_expert_kl_mismatch={})
    with pytest.raises(ValueError):
        emit_jepa_wsp_pheromone_directive({"a": 1.0}, expert_causal_gains_nats={"e0": -0.1})
    with pytest.raises(ValueError):
        emit_jepa_wsp_pheromone_directive({"a": 1.0}, expert_causal_gains_nats={}, expert_prose_harm_nats={})

    with pytest.raises(TypeError):
        evaluate_jepa_wsp_operculation(
            s1, fx["good_cand"], "bad", fx["metrics_base"], fx["metrics_good"], fx["latents_base"], fx["latents_good"]  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        evaluate_jepa_wsp_operculation(
            "bad", fx["good_cand"], fx["predictor"], fx["metrics_base"], fx["metrics_good"], fx["latents_base"], fx["latents_good"]  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        evaluate_jepa_wsp_operculation(
            {"a": "bad"}, fx["good_cand"], fx["predictor"], fx["metrics_base"], fx["metrics_good"], fx["latents_base"], fx["latents_good"]  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError):
        evaluate_jepa_wsp_operculation(
            s1, {"wrong_key": torch.zeros(1)}, fx["predictor"], fx["metrics_base"], fx["metrics_good"], fx["latents_base"], fx["latents_good"]
        )
    with pytest.raises(ValueError):
        evaluate_jepa_wsp_operculation(
            s1,
            {"expert_weight": torch.zeros(2, 2), "norm_weight": s1["norm_weight"].clone()},
            fx["predictor"],
            fx["metrics_base"],
            fx["metrics_good"],
            fx["latents_base"],
            fx["latents_good"],
        )
    with pytest.raises(ValueError):
        evaluate_jepa_wsp_operculation(
            s1, fx["good_cand"], fx["predictor"], {"D0_general_prose": {"nll": 1.0}}, fx["metrics_good"], fx["latents_base"], fx["latents_good"]
        )
    with pytest.raises(TypeError):
        evaluate_jepa_wsp_operculation(
            s1, fx["good_cand"], fx["predictor"], {"D0_general_prose": "bad", "D1_x": {"nll": 1.0}}, fx["metrics_good"], fx["latents_base"], fx["latents_good"]  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError):
        evaluate_jepa_wsp_operculation(
            s1, fx["good_cand"], fx["predictor"], {"D0_general_prose": {}, "D1_x": {"nll": 1.0}}, fx["metrics_good"], fx["latents_base"], fx["latents_good"]
        )
    with pytest.raises(ValueError):
        evaluate_jepa_wsp_operculation(
            s1, fx["good_cand"], fx["predictor"], {"x": {"nll": 1.0}, "y": {"nll": 1.0}}, fx["metrics_good"], fx["latents_base"], fx["latents_good"]
        )
    with pytest.raises(ValueError):
        evaluate_jepa_wsp_operculation(
            s1, fx["good_cand"], fx["predictor"], {"D0_a": {"nll": 1.0}, "D0_b": {"nll": 1.0}}, fx["metrics_good"], fx["latents_base"], fx["latents_good"]
        )
    with pytest.raises(ValueError):
        evaluate_jepa_wsp_operculation(
            s1, fx["good_cand"], fx["predictor"], fx["metrics_base"], {"D0_general_prose": {"nll": 1.0}, "D1_other": {"nll": 1.0}}, fx["latents_base"], fx["latents_good"]
        )
    with pytest.raises(ValueError):
        evaluate_jepa_wsp_operculation(
            s1, fx["good_cand"], fx["predictor"], fx["metrics_base"], fx["metrics_good"], {"D0_general_prose": (torch.ones(2, 12), torch.ones(2, 12))}, fx["latents_good"]
        )
    with pytest.raises(ValueError):
        bad_lat = dict(fx["latents_base"], D1_code_python=(torch.ones(2, 12),))
        evaluate_jepa_wsp_operculation(
            s1, fx["good_cand"], fx["predictor"], fx["metrics_base"], fx["metrics_good"], bad_lat, fx["latents_good"]  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        bad_lat2 = dict(fx["latents_base"], D1_code_python=(torch.ones(2, 12), torch.ones(3, 12)))
        evaluate_jepa_wsp_operculation(
            s1, fx["good_cand"], fx["predictor"], fx["metrics_base"], fx["metrics_good"], bad_lat2, fx["latents_good"]
        )

    # Repository compliance checks: AGPL-3.0, ZeroStubAudit, PRSecurityScanner, FORBIDDEN_ADDED, OwnerOrderGuardrail
    scanner = PRSecurityScanner()
    gk = _load_gatekeeper()
    auditor = ZeroStubAudit() if ZeroStubAudit is not None else _FallbackZeroStubAudit()

    changed_files: list[dict[str, Any]] = []
    for rel, path in (
        ("src/hivemind/jepa/latent_jepa_wsp_critic.py", TARGET_PATH),
        ("tests/test_latent_jepa_wsp_critic.py", TEST_PATH),
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
    assert "DAILY-C1-S3" in doc_text
    assert "src/hivemind/jepa/latent_jepa_wsp_critic.py" in doc_text
    assert "tests/test_latent_jepa_wsp_critic.py" in doc_text
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
        pr_number=1075,
        pr_title="feat(jepa): frozen latent JEPA world-model critic for Reine-Cortex Operculation and Pheromones Royales",
        pr_body="Implements DAILY-C1-S3 (Paper v4.4.0 section 15.4 Opportunity J3, JEPA-WSP Critic) under 0.00 EUR / 0.00 USD AGPL-3.0.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(jepa): frozen latent JEPA world-model critic for Reine-Cortex Operculation and Pheromones Royales"],
        author_email=None,
    )
    assert scan.passed is True
    assert scan.blocked_merge is False
    assert scan.findings == []

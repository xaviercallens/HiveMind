# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-37: folded final RMSNorm/LayerNorm temperature & ECE surgeon."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.folded_norm_ece_calibrator import (
    ANSE_MAXIMUM_PAIN,
    CALIBRATOR_RECEIPT_OBJECT,
    CALIBRATOR_SCHEMA,
    FoldedNormCalibrationConfig,
    apply_norm_weights,
    compute_module_state_sha256,
    compute_payload_sha256,
    compute_token_nll_and_ece,
    execute_verified_folded_norm_calibration,
    fit_folded_norm_temperature_and_interpolation,
    fold_temperature_into_norm_params,
    verify_folded_norm_receipt_provenance,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover - optional in lightweight CI runners
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackStubResult:
    """CI-safe stub audit result used when optional ``gwaya`` is absent."""

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
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "folded_norm_ece_calibrator.py"
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


class _VarianceEpsRMSNorm(nn.Module):
    """Qwen2/Qwen3.5-style biasless RMSNorm exposing ``variance_epsilon``."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.variance_epsilon = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (int(self.weight.shape[0]),), self.weight, self.variance_epsilon)


class _QwenStyleRMSNormLM(nn.Module):
    """Compact deterministic causal LM with tail ``down_proj``, final RMSNorm, and biasless ``lm_head``."""

    def __init__(
        self,
        *,
        vocab_size: int = 24,
        hidden_size: int = 32,
        use_variance_eps_norm: bool = False,
        seed: int = 20261237,
    ) -> None:
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm: nn.Module = (
            _VarianceEpsRMSNorm(hidden_size, eps=1e-6)
            if use_variance_eps_norm
            else nn.RMSNorm(hidden_size, eps=1e-6)
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Build 4 orthogonal domain axes: D0 (0..5), D1 (6..11), D2 (12..17), D3 (18..23)
        q_mat, _ = torch.linalg.qr(torch.randn((hidden_size, hidden_size), generator=gen, dtype=torch.float64))
        axes = q_mat[:, :4].T.to(torch.float32)  # (4, H)

        with torch.no_grad():
            self.embed.weight.zero_()
            self.down_proj.weight.zero_()
            self.lm_head.weight.zero_()
            self.norm.weight.fill_(1.0)

            # Input embeddings determine domain identity cleanly, scaled to induce sharp overconfidence
            for d_idx in range(4):
                lo, hi = 6 * d_idx, 6 * (d_idx + 1)
                ax = axes[d_idx]
                for tok in range(lo, hi):
                    self.embed.weight[tok] = 2.5 * ax
                # Modal token vs secondary token in each domain
                main_tok = lo + 1
                alt_tok = lo + 2
                # High confidence logit gap (8.0 vs 4.5 vs -4.0) -> ~96.5% confidence on 75% modal data
                scale_main = 8.0 / math.sqrt(float(hidden_size))
                scale_alt = 4.5 / math.sqrt(float(hidden_size))
                scale_other = -4.0 / math.sqrt(float(hidden_size))
                for tok in range(lo, hi):
                    if tok == main_tok:
                        self.lm_head.weight[tok] = scale_main * ax
                    elif tok == alt_tok:
                        self.lm_head.weight[tok] = scale_alt * ax
                    else:
                        self.lm_head.weight[tok] = scale_other * ax

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        h = x + self.down_proj(x)
        h_norm = self.norm(h)
        return self.lm_head(h_norm)


def _build_overconfident_crowd_layernorm_lm(*, seed: int = 20261238) -> CrowdV1WithEngram:
    """Build deterministic ``CrowdV1WithEngram`` (with biased final ``LayerNorm``) exhibiting sharp logit overconfidence."""
    hidden_size = 32
    vocab_size = 24
    cfg = CrowdV1RuntimeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        intermediate_size=64,
        max_position_embeddings=32,
        engram_buckets=0,
        tie_word_embeddings=False,
    )
    model = CrowdV1WithEngram(cfg)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q_mat, _ = torch.linalg.qr(torch.randn((hidden_size, hidden_size), generator=gen, dtype=torch.float64))
    # Ensure zero-mean orthogonal directions so LayerNorm leaves direction intact
    ones = torch.ones((hidden_size, 1), dtype=torch.float64)
    q_zero, _ = torch.linalg.qr(torch.cat([ones, q_mat[:, :5]], dim=1))
    axes = q_zero[:, 1:5].T.to(torch.float32)

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()
        model.blocks[0].ln1.weight.fill_(1.0)
        model.blocks[0].ln2.weight.fill_(1.0)

        for d_idx in range(4):
            lo, hi = 6 * d_idx, 6 * (d_idx + 1)
            ax = axes[d_idx]
            for tok in range(lo, hi):
                model.wte.weight[tok] = 2.0 * ax
            main_tok = lo + 1
            alt_tok = lo + 2
            scale_main = 8.0 / math.sqrt(float(hidden_size))
            scale_alt = 4.5 / math.sqrt(float(hidden_size))
            scale_other = -4.0 / math.sqrt(float(hidden_size))
            for tok in range(lo, hi):
                if tok == main_tok:
                    model.head.weight[tok] = scale_main * ax
                elif tok == alt_tok:
                    model.head.weight[tok] = scale_alt * ax
                else:
                    model.head.weight[tok] = scale_other * ax
    return model


def _make_multidomain_chunks(
    *,
    prefix_style: str = "D",
    seq_len: int = 16,
    score_from: int = 4,
    chunks_per_domain: int = 8,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    """Build deterministic 4-domain chunks with ~72% modal / ~28% runner-up ambiguity."""
    gen = torch.Generator(device="cpu").manual_seed(7700 + seed_offset)
    if prefix_style == "D":
        domains = (
            ("D0_general_prose", 0, 1, 2),
            ("D1_python_systems_code", 6, 7, 8),
            ("D2_formal_math_lean4", 12, 13, 14),
            ("D3_scientific_reasoning", 18, 19, 20),
        )
    else:
        domains = (
            ("E0_general_fineweb", 0, 1, 2),
            ("E1_python_systems_code", 6, 7, 8),
            ("E2_formal_math_lean4", 12, 13, 14),
            ("E3_scientific_reasoning", 18, 19, 20),
        )

    chunks: list[dict[str, Any]] = []
    for d_idx, (dom_name, lo, main_tok, alt_tok) in enumerate(domains):
        for r_idx in range(chunks_per_domain):
            ids = torch.randint(lo, lo + 6, (seq_len,), generator=gen).tolist()
            for pos in range(score_from, seq_len):
                ids[pos] = alt_tok if (pos + r_idx + d_idx) % 4 == 0 else main_tok
            key_name = "domain" if prefix_style == "D" else "expert_id"
            chunks.append({key_name: dom_name, "ids": ids})
    return chunks


# ---------------------------------------------------------------------------
# Scenario S1: <0.001% Folded Norm Calibration Reduces ECE >40% & Lowers NLL
# ---------------------------------------------------------------------------


def test_scenario_s1_overconfident_rmsnorm_and_layernorm_folded_ece_and_nll_calibration() -> None:
    """S1: Calibrate overconfident RMSNorm & LayerNorm LMs via <0.001% folded params (ECE -40%+, NLL down)."""
    probe_d = _make_multidomain_chunks(prefix_style="D", seed_offset=1)
    eval_d = _make_multidomain_chunks(prefix_style="D", seed_offset=2)

    rms_model = _QwenStyleRMSNormLM(use_variance_eps_norm=True)
    _, rms_receipt = execute_verified_folded_norm_calibration(
        rms_model,
        probe_d,
        eval_chunks=eval_d,
        score_from=4,
        max_general_drift_rel=0.002,
        max_edited_param_fraction=1e-5,
        total_model_params=1_543_714_304,
    )

    assert rms_receipt["accepted"] is True
    assert rms_receipt["rolled_back"] is False
    assert rms_receipt["status"] == "ACCEPTED"
    assert rms_receipt["norm_kind"] == "rmsnorm"
    assert rms_receipt["tau_temp"] > 1.25
    assert rms_receipt["edited_non_norm_params"] == 0
    assert rms_receipt["edited_norm_params"] == 32
    assert rms_receipt["edited_param_fraction"] < 1e-5  # < 0.001% parameter budget
    assert rms_receipt["ece_reduction_rel"] > 0.40
    assert rms_receipt["expected_calibration_error"] < rms_receipt["base_ece"] * 0.60
    assert rms_receipt["nll_mean"] < rms_receipt["base_nll_mean"]
    assert rms_receipt["general_drift_rel"] <= 0.002
    assert rms_receipt["delta_e"] < 0.0
    assert verify_folded_norm_receipt_provenance(rms_receipt) is True

    # LayerNorm (with bias) on CrowdV1WithEngram
    probe_e = _make_multidomain_chunks(prefix_style="E", seed_offset=11)
    eval_e = _make_multidomain_chunks(prefix_style="E", seed_offset=12)
    ln_model = _build_overconfident_crowd_layernorm_lm()
    _, ln_receipt = execute_verified_folded_norm_calibration(
        ln_model,
        probe_e,
        eval_chunks=eval_e,
        norm_kind="layernorm",
        score_from=4,
        max_general_drift_rel=0.002,
        total_model_params=1_543_714_304,
    )
    assert ln_receipt["accepted"] is True
    assert ln_receipt["norm_kind"] == "layernorm"
    assert ln_receipt["has_bias"] is True
    assert ln_receipt["edited_param_fraction"] < 1e-5
    assert ln_receipt["ece_reduction_rel"] > 0.40
    assert ln_receipt["nll_mean"] < ln_receipt["base_nll_mean"]
    assert ln_receipt["general_drift_rel"] <= 0.002
    assert verify_folded_norm_receipt_provenance(ln_receipt) is True

    # Verify compute_token_nll_and_ece ignore_index masking & bounded metrics directly
    logits_chk = torch.tensor(
        [
            [[4.0, 0.0, -3.0], [4.0, 0.0, -3.0], [10.0, -10.0, -10.0]],
            [[0.5, 0.4, 0.1], [2.0, 1.0, -1.0], [-5.0, 5.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    targets_chk = torch.tensor([[0, 1, -100], [0, 0, -100]], dtype=torch.long)
    tok_metrics = compute_token_nll_and_ece(logits_chk, targets_chk, num_bins=10, ignore_index=-100)
    assert tok_metrics["num_tokens"] == 4
    assert tok_metrics["num_bins"] == 10
    assert math.isclose(tok_metrics["top1_accuracy"], 0.75, rel_tol=1e-9)
    assert tok_metrics["expected_calibration_error"] > 0.0
    assert tok_metrics["mce"] >= tok_metrics["expected_calibration_error"]
    assert tok_metrics["brier_value"] >= 0.0
    assert tok_metrics["entropy_mean"] > 0.0
    assert len(tok_metrics["bins"]) == 10


# ---------------------------------------------------------------------------
# Scenario S2: Exact Equivalence (<1e-6 Max Logit Diff) Across RMSNorm & LayerNorm
# ---------------------------------------------------------------------------


def test_scenario_s2_exact_logit_equivalence_rmsnorm_and_layernorm_with_interpolation() -> None:
    """S2: Folded norm weights w/tau & b/tau match explicit logits/tau to <1e-6 on RMSNorm & LayerNorm."""
    gen = torch.Generator(device="cpu").manual_seed(20263702)
    n_seq, seq_len, hidden_dim, vocab_size = 12, 8, 24, 16
    pre_h = torch.randn((n_seq, seq_len, hidden_dim), generator=gen, dtype=torch.float32)
    lm_head_w = 0.4 * torch.randn((vocab_size, hidden_dim), generator=gen, dtype=torch.float32)
    targets = torch.randint(0, vocab_size, (n_seq, seq_len), generator=gen, dtype=torch.long)
    targets[:, 0] = -100  # Exercise ignore_index handling in fit_folded_norm_temperature_and_interpolation
    is_gen_mask = torch.tensor([i < 4 for i in range(n_seq)], dtype=torch.bool)

    # 1. RMSNorm equivalence (with interpolation cand_w != base_w)
    base_rms_w = 0.8 + 0.3 * torch.rand(hidden_dim, generator=gen, dtype=torch.float32)
    cand_rms_w = base_rms_w + 0.25 * torch.randn(hidden_dim, generator=gen, dtype=torch.float32)
    rms_fit = fit_folded_norm_temperature_and_interpolation(
        pre_h,
        base_rms_w,
        None,
        cand_rms_w,
        None,
        lm_head_w,
        targets,
        is_gen_mask,
        norm_kind="rmsnorm",
        eps=1e-6,
        max_general_drift_rel=0.01,
        total_model_params=1_543_714_304,
    )
    assert rms_fit["norm_kind"] == "rmsnorm"
    assert rms_fit["folded_norm_bias"] is None
    assert rms_fit["max_logit_equiv_diff"] < 1e-6
    assert rms_fit["lm_head_matmuls_used"] < rms_fit["grid_evaluations_count"]
    assert rms_fit["edited_param_fraction"] < 1e-5

    lam_r = float(rms_fit["lambda_norm"])
    tau_r = float(rms_fit["tau_temp"])
    w_interp_r = (base_rms_w.double() + lam_r * (cand_rms_w.double() - base_rms_w.double())).float()
    logits_folded_r = F.linear(
        apply_norm_weights(pre_h, rms_fit["folded_norm_weight"], None, eps=1e-6, norm_type="rmsnorm"),
        lm_head_w,
    )
    logits_explicit_r = (
        F.linear(apply_norm_weights(pre_h, w_interp_r, None, eps=1e-6, norm_type="rmsnorm"), lm_head_w) / tau_r
    )
    assert float((logits_folded_r - logits_explicit_r).abs().max().item()) < 1e-6

    # 2. LayerNorm (with non-zero bias) + auto norm_kind + sequence bool mask
    base_ln_w = 0.9 + 0.2 * torch.rand(hidden_dim, generator=gen, dtype=torch.float32)
    base_ln_b = 0.2 * torch.randn(hidden_dim, generator=gen, dtype=torch.float32)
    cand_ln_w = base_ln_w + 0.3 * torch.randn(hidden_dim, generator=gen, dtype=torch.float32)
    cand_ln_b = base_ln_b + 0.2 * torch.randn(hidden_dim, generator=gen, dtype=torch.float32)

    ln_fit = fit_folded_norm_temperature_and_interpolation(
        pre_h,
        base_ln_w,
        base_ln_b,
        cand_ln_w,
        cand_ln_b,
        lm_head_w,
        targets,
        [i < 4 for i in range(n_seq)],
        norm_kind="auto",
        eps=1e-5,
        max_general_drift_rel=0.01,
        tau_grid=(0.8, 1.0, 1.25, 1.6),
        refine_tau=False,
    )
    assert ln_fit["norm_kind"] == "layernorm"
    assert ln_fit["folded_norm_bias"] is not None
    assert ln_fit["max_logit_equiv_diff"] < 1e-6
    assert ln_fit["lm_head_matmuls_used"] <= 4

    lam_l = float(ln_fit["lambda_norm"])
    tau_l = float(ln_fit["tau_temp"])
    w_interp_l = (base_ln_w.double() + lam_l * (cand_ln_w.double() - base_ln_w.double())).float()
    b_interp_l = (base_ln_b.double() + lam_l * (cand_ln_b.double() - base_ln_b.double())).float()
    folded_w_chk, folded_b_chk = fold_temperature_into_norm_params(w_interp_l, b_interp_l, tau_l)
    assert torch.equal(folded_w_chk, ln_fit["folded_norm_weight"])
    assert folded_b_chk is not None and torch.equal(folded_b_chk, ln_fit["folded_norm_bias"])
    logits_folded_l = F.linear(
        apply_norm_weights(pre_h, folded_w_chk, folded_b_chk, eps=1e-5, norm_type="layernorm"),
        lm_head_w,
    )
    logits_explicit_l = (
        F.linear(apply_norm_weights(pre_h, w_interp_l, b_interp_l, eps=1e-5, norm_type="layernorm"), lm_head_w)
        / tau_l
    )
    assert float((logits_folded_l - logits_explicit_l).abs().max().item()) < 1e-6

    # 3. Biasless LayerNorm explicit norm_kind='layernorm' + per-token numpy bool mask
    ln_nobias_fit = fit_folded_norm_temperature_and_interpolation(
        pre_h,
        base_rms_w,
        None,
        base_rms_w,
        None,
        lm_head_w,
        targets,
        np.broadcast_to((np.arange(n_seq) < 4)[:, None], (n_seq, seq_len)),
        norm_kind="layernorm",
        eps=1e-5,
    )
    assert ln_nobias_fit["norm_kind"] == "layernorm"
    assert ln_nobias_fit["folded_norm_bias"] is None
    assert ln_nobias_fit["max_logit_equiv_diff"] < 1e-6


# ---------------------------------------------------------------------------
# Scenario S3: ANSE Stage Ratchet Replicating PILOT3B (Rolls Back Harmful SVD)
# ---------------------------------------------------------------------------


def test_scenario_s3_anse_stage_ratchet_rolls_back_harmful_svd_and_keeps_folded_norm() -> None:
    """S3: Stage ratchet honestly rolls back unnecessary tail-SVD perturbation while keeping folded norm."""
    probe = _make_multidomain_chunks(prefix_style="D", seed_offset=21)
    eval_c = _make_multidomain_chunks(prefix_style="D", seed_offset=22)

    # Part A: upstream_stage_fn plants a tail-layer down_proj perturbation that inflates NLL (PILOT3B replication)
    model_a = _QwenStyleRMSNormLM(use_variance_eps_norm=False, seed=20261239)
    clean_down_a = model_a.down_proj.weight.detach().clone()

    def _harmful_tail_esd_stage(m: nn.Module) -> dict[str, Any]:
        with torch.no_grad():
            d = m.down_proj.weight.shape[0]
            skew = torch.roll(torch.eye(d), shifts=1, dims=0)
            m.down_proj.weight.add_(0.75 * skew)
        return {"stage": "tail_esd_rank1_probe"}

    _, rec_a = execute_verified_folded_norm_calibration(
        model_a,
        probe,
        eval_chunks=eval_c,
        upstream_stage_fn=_harmful_tail_esd_stage,
        score_from=4,
        total_model_params=1_543_714_304,
    )
    assert rec_a["accepted"] is True
    assert rec_a["upstream_stage_ratchet"] == "rolled_back_to_clean_base"
    assert rec_a["upstream_stage_meta"] == {"stage": "tail_esd_rank1_probe"}
    assert torch.equal(model_a.down_proj.weight.detach(), clean_down_a)
    assert rec_a["edited_non_norm_params"] == 0
    assert rec_a["edited_param_fraction"] < 1e-5
    assert rec_a["ece_reduction_rel"] > 0.40
    assert rec_a["delta_e"] < 0.0
    assert verify_folded_norm_receipt_provenance(rec_a) is True

    # Part B: pre-edited model + clean base_state_dict rolls back harmful upstream edit too
    model_b = _QwenStyleRMSNormLM(use_variance_eps_norm=False, seed=20261239)
    clean_sd_b = {k: v.detach().clone() for k, v in model_b.state_dict().items()}
    _harmful_tail_esd_stage(model_b)
    _, rec_b = execute_verified_folded_norm_calibration(
        model_b,
        probe,
        eval_chunks=eval_c,
        base_state_dict=clean_sd_b,
        score_from=4,
        total_model_params=1_543_714_304,
    )
    assert rec_b["accepted"] is True
    assert rec_b["upstream_stage_ratchet"] == "rolled_back_to_clean_base"
    assert torch.equal(model_b.down_proj.weight.detach(), clean_sd_b["down_proj.weight"])
    assert rec_b["edited_non_norm_params"] == 0

    # Part C: Positive control — genuinely beneficial upstream edit is accepted AND inplace=False keeps caller untouched
    model_c = _QwenStyleRMSNormLM(use_variance_eps_norm=False, seed=20261240)
    with torch.no_grad():
        # Corrupt down_proj by inverting hidden representation (-1.8 * I => h = -0.8 * x), ruining NLL
        d_c = model_c.down_proj.weight.shape[0]
        model_c.down_proj.weight.copy_(-1.8 * torch.eye(d_c))
    orig_sha_c = compute_module_state_sha256(model_c)

    def _beneficial_upstream_repair(m: nn.Module) -> dict[str, Any]:
        with torch.no_grad():
            m.down_proj.weight.zero_()
        return {"stage": "beneficial_repair"}

    out_c, rec_c = execute_verified_folded_norm_calibration(
        model_c,
        probe,
        eval_chunks=eval_c,
        upstream_stage_fn=_beneficial_upstream_repair,
        score_from=4,
        inplace=False,
    )
    assert out_c is not model_c
    assert compute_module_state_sha256(model_c) == orig_sha_c  # caller untouched when inplace=False
    assert rec_c["accepted"] is True
    assert rec_c["upstream_stage_ratchet"] == "accepted"
    assert float(out_c.down_proj.weight.abs().max().item()) == 0.0


# ---------------------------------------------------------------------------
# Scenario S4: ANSE Maximum Pain Atomic Rollback & Fixed-Point Verification
# ---------------------------------------------------------------------------


def test_scenario_s4_anse_maximum_pain_atomic_rollback_and_provenance() -> None:
    """S4: Infeasible guard or budget breach triggers ANSE Maximum Pain and atomic bitwise rollback."""
    probe = _make_multidomain_chunks(prefix_style="D", seed_offset=31)
    eval_c = _make_multidomain_chunks(prefix_style="D", seed_offset=32)

    # 1. Infeasible general-domain NLL baseline ceiling triggers ANSE Maximum Pain atomic rollback
    model_guard = _QwenStyleRMSNormLM(seed=20261241)
    model_guard.train(True)
    init_sha = compute_module_state_sha256(model_guard)

    impossible_base_nll = {
        "D0_general_prose": 0.05,
        "D1_python_systems_code": 0.05,
        "D2_formal_math_lean4": 0.05,
        "D3_scientific_reasoning": 0.05,
    }
    _, rec_guard = execute_verified_folded_norm_calibration(
        model_guard,
        probe,
        eval_chunks=eval_c,
        base_domain_nll=impossible_base_nll,
        base_ece=0.001,
        score_from=4,
        max_general_drift_rel=0.005,
        max_domain_drift_rel=0.005,
    )
    assert rec_guard["accepted"] is False
    assert rec_guard["rolled_back"] is True
    assert rec_guard["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rec_guard["guard_ok"] is False
    assert rec_guard["cand_anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_guard["final_state_sha256"] == init_sha
    assert compute_module_state_sha256(model_guard) == init_sha
    assert model_guard.training is True  # training flag preserved
    assert verify_folded_norm_receipt_provenance(rec_guard) is True

    # 2. Parameter budget violation rolls back atomically
    model_budget = _QwenStyleRMSNormLM(seed=20261242)
    init_budget_sha = compute_module_state_sha256(model_budget)
    _, rec_budget = execute_verified_folded_norm_calibration(
        model_budget,
        probe,
        eval_chunks=eval_c,
        score_from=4,
        max_edited_param_fraction=1e-7,  # 32 / ~2048 > 1e-7
    )
    assert rec_budget["accepted"] is False
    assert rec_budget["rolled_back"] is True
    assert rec_budget["status"] == "ROLLED_BACK_BUDGET_VIOLATION"
    assert rec_budget["budget_ok"] is False
    assert rec_budget["cand_anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert compute_module_state_sha256(model_budget) == init_budget_sha

    # 3. Fixed-point unchanged when tau=1.0 is already optimal (plus non-improving rollback branch)
    model_fp = _QwenStyleRMSNormLM(seed=20261243)
    cfg_fp = FoldedNormCalibrationConfig(tau_grid=(1.0,), refine_tau=False, score_from=4)
    _, rec_fp = execute_verified_folded_norm_calibration(model_fp, probe, config=cfg_fp)
    assert rec_fp["accepted"] is False
    assert rec_fp["rolled_back"] is False
    assert rec_fp["status"] == "FIXED_POINT_UNCHANGED"

    # 4. Also cover fit_folded_norm_temperature_and_interpolation infeasible guard fallback
    gen = torch.Generator(device="cpu").manual_seed(999)
    h_tiny = torch.randn((4, 4, 8), generator=gen)
    w_tiny = torch.ones(8)
    head_tiny = torch.randn((6, 8), generator=gen)
    t_tiny = torch.randint(0, 6, (4, 4), generator=gen)
    infeasible_fit = fit_folded_norm_temperature_and_interpolation(
        h_tiny,
        w_tiny,
        None,
        w_tiny,
        None,
        head_tiny,
        t_tiny,
        [True, True, False, False],
        base_general_nll_override=0.01,
        base_total_nll_override=0.01,
        base_ece_override=0.0,
        max_general_drift_rel=0.001,
    )
    assert infeasible_fit["guard_feasible"] is False
    assert infeasible_fit["anse_energy"] >= ANSE_MAXIMUM_PAIN

    # 5. Tamper detection on receipt provenance
    tampered = dict(rec_guard)
    tampered["tau_temp"] = 9.99
    assert verify_folded_norm_receipt_provenance(tampered) is False
    bad_hash = dict(rec_guard, receipt_sha256="not-a-sha")
    assert verify_folded_norm_receipt_provenance(bad_hash) is False
    wrong_schema = dict(rec_guard, schema="wrong/1")
    wrong_schema["receipt_sha256"] = compute_payload_sha256(
        {k: v for k, v in wrong_schema.items() if k != "receipt_sha256"}
    )
    assert verify_folded_norm_receipt_provenance(wrong_schema) is False
    bad_nan = dict(rec_guard, tau_temp=float("nan"))
    assert verify_folded_norm_receipt_provenance(bad_nan) is False
    assert CALIBRATOR_SCHEMA == rec_guard["schema"]
    assert CALIBRATOR_RECEIPT_OBJECT == rec_guard["object"]


# ---------------------------------------------------------------------------
# Scenario S5: Validation Errors & Repository Security/Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_scenario_s5_validation_errors_and_security_stub_compliance_gates() -> None:
    """S5: Input validation edge cases plus ZeroStubAudit, PRSecurityScanner, and lateral-movement checks."""
    # 1. FoldedNormCalibrationConfig validation
    with pytest.raises(TypeError, match="bool"):
        FoldedNormCalibrationConfig(refine_tau="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        FoldedNormCalibrationConfig(num_bins=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_bins"):
        FoldedNormCalibrationConfig(num_bins=0)
    with pytest.raises(ValueError, match="score_from"):
        FoldedNormCalibrationConfig(score_from=-1)
    with pytest.raises(TypeError, match="real number"):
        FoldedNormCalibrationConfig(max_general_drift_rel="0.01")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        FoldedNormCalibrationConfig(max_general_drift_rel=float("nan"))
    with pytest.raises(ValueError, match=">= 0.0"):
        FoldedNormCalibrationConfig(max_general_drift_rel=-0.01)
    with pytest.raises(ValueError, match="> 0.0"):
        FoldedNormCalibrationConfig(anse_safe_general_ratio=0.0)
    with pytest.raises(ValueError, match="max_edited_param_fraction"):
        FoldedNormCalibrationConfig(max_edited_param_fraction=1.5)
    with pytest.raises(TypeError, match="lambda_grid"):
        FoldedNormCalibrationConfig(lambda_grid="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lambda_grid"):
        FoldedNormCalibrationConfig(lambda_grid=())
    with pytest.raises(ValueError, match="lambda_grid"):
        FoldedNormCalibrationConfig(lambda_grid=(1.2,))
    with pytest.raises(TypeError, match="tau_grid"):
        FoldedNormCalibrationConfig(tau_grid="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tau_grid"):
        FoldedNormCalibrationConfig(tau_grid=())
    with pytest.raises(TypeError, match="general_prefixes"):
        FoldedNormCalibrationConfig(general_prefixes="E0_")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="general_prefixes"):
        FoldedNormCalibrationConfig(general_prefixes=())
    with pytest.raises(ValueError, match="general_prefixes"):
        FoldedNormCalibrationConfig(general_prefixes=("",))

    # 2. compute_token_nll_and_ece validation
    ok_logits = torch.zeros((2, 3, 5), dtype=torch.float32)
    ok_targets = torch.zeros((2, 3), dtype=torch.long)
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_token_nll_and_ece([[1.0]], ok_targets)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ndim >= 2"):
        compute_token_nll_and_ece(torch.ones(5), torch.zeros((), dtype=torch.long))
    with pytest.raises(ValueError, match="non-empty"):
        compute_token_nll_and_ece(torch.ones((0, 5)), torch.zeros((0,), dtype=torch.long))
    with pytest.raises(TypeError, match="floating-point"):
        compute_token_nll_and_ece(torch.ones((2, 5), dtype=torch.long), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        bad_lg = ok_logits.clone()
        bad_lg[0, 0, 0] = float("inf")
        compute_token_nll_and_ece(bad_lg, ok_targets)
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_token_nll_and_ece(ok_logits, [[0, 1, 2], [0, 1, 2]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        compute_token_nll_and_ece(ok_logits, torch.zeros((2, 2), dtype=torch.long))
    with pytest.raises(TypeError, match="integer dtype"):
        compute_token_nll_and_ece(ok_logits, torch.zeros((2, 3), dtype=torch.float32))
    with pytest.raises(ValueError, match="non-ignored"):
        compute_token_nll_and_ece(ok_logits, torch.full((2, 3), -100, dtype=torch.long))
    with pytest.raises(ValueError, match="token IDs"):
        compute_token_nll_and_ece(ok_logits, torch.full((2, 3), 9, dtype=torch.long))

    # 3. fit_folded_norm_temperature_and_interpolation validation
    h_ok = torch.ones((4, 3, 6), dtype=torch.float32)
    w_ok = torch.ones(6, dtype=torch.float32)
    b_ok = torch.zeros(6, dtype=torch.float32)
    head_ok = torch.ones((5, 6), dtype=torch.float32)
    t_ok = torch.zeros((4, 3), dtype=torch.long)
    mask_ok = [True, True, False, False]

    with pytest.raises(ValueError, match="norm_kind"):
        fit_folded_norm_temperature_and_interpolation(
            h_ok, w_ok, None, w_ok, None, head_ok, t_ok, mask_ok, norm_kind="batchnorm"
        )
    with pytest.raises(ValueError, match="both be provided"):
        fit_folded_norm_temperature_and_interpolation(
            h_ok, w_ok, b_ok, w_ok, None, head_ok, t_ok, mask_ok, norm_kind="layernorm"
        )
    with pytest.raises(ValueError, match="bias"):
        fit_folded_norm_temperature_and_interpolation(
            h_ok, w_ok, b_ok, w_ok, b_ok, head_ok, t_ok, mask_ok, norm_kind="rmsnorm"
        )
    with pytest.raises(ValueError, match="ndim == 2"):
        fit_folded_norm_temperature_and_interpolation(
            h_ok, w_ok, None, w_ok, None, torch.ones(6), t_ok, mask_ok
        )
    with pytest.raises(ValueError, match="hidden dim"):
        fit_folded_norm_temperature_and_interpolation(
            h_ok, w_ok, None, w_ok, None, torch.ones((5, 7)), t_ok, mask_ok
        )
    with pytest.raises(ValueError, match="hidden dim"):
        fit_folded_norm_temperature_and_interpolation(
            h_ok, torch.ones(7), None, w_ok, None, head_ok, t_ok, mask_ok
        )
    with pytest.raises(TypeError, match="bool"):
        fit_folded_norm_temperature_and_interpolation(
            h_ok, w_ok, None, w_ok, None, head_ok, t_ok, torch.ones(4, dtype=torch.long)
        )
    with pytest.raises(TypeError, match="bool"):
        fit_folded_norm_temperature_and_interpolation(
            h_ok, w_ok, None, w_ok, None, head_ok, t_ok, np.ones(4, dtype=np.int64)
        )
    with pytest.raises(ValueError, match="empty"):
        fit_folded_norm_temperature_and_interpolation(
            h_ok, w_ok, None, w_ok, None, head_ok, t_ok, []
        )
    with pytest.raises(TypeError, match="bool"):
        fit_folded_norm_temperature_and_interpolation(
            h_ok, w_ok, None, w_ok, None, head_ok, t_ok, [1, 0, 1, 0]  # type: ignore[list-item]
        )
    with pytest.raises(TypeError, match="bool Tensor"):
        fit_folded_norm_temperature_and_interpolation(
            h_ok, w_ok, None, w_ok, None, head_ok, t_ok, "bad"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="must match"):
        fit_folded_norm_temperature_and_interpolation(
            h_ok, w_ok, None, w_ok, None, head_ok, t_ok, [True, False]
        )
    with pytest.raises(ValueError, match="at least one non-ignored token"):
        fit_folded_norm_temperature_and_interpolation(
            h_ok, w_ok, None, w_ok, None, head_ok, torch.full_like(t_ok, -100), mask_ok
        )
    with pytest.raises(ValueError, match="token IDs"):
        fit_folded_norm_temperature_and_interpolation(
            h_ok, w_ok, None, w_ok, None, head_ok, torch.full_like(t_ok, 99), mask_ok
        )
    with pytest.raises(ValueError, match="at least one non-ignored general"):
        fit_folded_norm_temperature_and_interpolation(
            h_ok, w_ok, None, w_ok, None, head_ok, t_ok, [False, False, False, False]
        )

    # 4. execute_verified_folded_norm_calibration & receipt validation
    m_ok = _QwenStyleRMSNormLM(seed=20261244)
    chunks_ok = _make_multidomain_chunks(prefix_style="D", seed_offset=41)
    with pytest.raises(TypeError, match="FoldedNormCalibrationConfig"):
        execute_verified_folded_norm_calibration(m_ok, chunks_ok, config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="upstream_stage_fn"):
        execute_verified_folded_norm_calibration(m_ok, chunks_ok, upstream_stage_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not both"):
        execute_verified_folded_norm_calibration(
            m_ok, chunks_ok, upstream_stage_fn=lambda _: None, base_state_dict=m_ok.state_dict()
        )
    with pytest.raises(TypeError, match="nn.Module"):
        execute_verified_folded_norm_calibration("not_a_model", chunks_ok)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="norm_kind"):
        execute_verified_folded_norm_calibration(m_ok, chunks_ok, norm_kind="bad")
    with pytest.raises(ValueError, match="final normalization"):
        execute_verified_folded_norm_calibration(nn.Linear(4, 4), chunks_ok)
    with pytest.raises(ValueError, match="unembedding head"):
        class _NoHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.RMSNorm(4)
        execute_verified_folded_norm_calibration(_NoHead(), chunks_ok)
    with pytest.raises(ValueError, match="must match norm dim"):
        class _BadHeadWidth(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.RMSNorm(4)
                self.head = nn.Linear(5, 8, bias=False)
        execute_verified_folded_norm_calibration(_BadHeadWidth(), chunks_ok)
    with pytest.raises(ValueError, match="non-zero bias"):
        class _BiasedHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.RMSNorm(4)
                self.head = nn.Linear(4, 8, bias=True)
                with torch.no_grad():
                    self.head.bias.fill_(0.5)
        execute_verified_folded_norm_calibration(_BiasedHead(), chunks_ok)
    with pytest.raises(ValueError, match="non-empty sequence"):
        execute_verified_folded_norm_calibration(m_ok, [], score_from=4)
    with pytest.raises(ValueError, match="ids"):
        execute_verified_folded_norm_calibration(m_ok, [{"domain": "D0_g"}], score_from=4)
    with pytest.raises(TypeError, match="integer dtype"):
        execute_verified_folded_norm_calibration(
            m_ok, [{"domain": "D0_g", "ids": torch.ones(8)}], score_from=4
        )
    with pytest.raises(TypeError, match="sequence or 1D"):
        execute_verified_folded_norm_calibration(
            m_ok, [{"domain": "D0_g", "ids": "bad"}], score_from=4
        )
    with pytest.raises(ValueError, match="length >= 2"):
        execute_verified_folded_norm_calibration(m_ok, [{"domain": "D0_g", "ids": [1]}], score_from=1)
    with pytest.raises(ValueError, match="token IDs"):
        execute_verified_folded_norm_calibration(
            m_ok, [{"domain": "D0_g", "ids": [0, 1, 2, 99]}], score_from=1
        )
    with pytest.raises(ValueError, match="too short"):
        execute_verified_folded_norm_calibration(
            m_ok, [{"domain": "D0_g", "ids": [0, 1, 2, 3]}], score_from=10
        )
    with pytest.raises(ValueError, match="sequence length"):
        execute_verified_folded_norm_calibration(
            m_ok,
            [{"domain": "D0_g", "ids": [0, 1, 2, 3, 4]}, {"domain": "D1_s", "ids": [0, 1, 2, 3]}],
            score_from=2,
        )
    with pytest.raises(ValueError, match="both general"):
        execute_verified_folded_norm_calibration(
            m_ok,
            [{"domain": "D0_g", "ids": [0, 1, 2, 3, 4]}, {"domain": "D0_g2", "ids": [0, 1, 2, 3, 4]}],
            score_from=2,
        )
    with pytest.raises(ValueError, match="base_state_dict"):
        execute_verified_folded_norm_calibration(m_ok, chunks_ok, base_state_dict={}, score_from=4)
    with pytest.raises(ValueError, match="exactly match"):
        execute_verified_folded_norm_calibration(
            m_ok, chunks_ok, base_state_dict={"x": torch.ones(2)}, score_from=4
        )
    with pytest.raises(TypeError, match="torch.Tensor"):
        bad_sd = dict(m_ok.state_dict())
        bad_sd["norm.weight"] = "bad"  # type: ignore[assignment]
        execute_verified_folded_norm_calibration(m_ok, chunks_ok, base_state_dict=bad_sd, score_from=4)
    with pytest.raises(ValueError, match="shape"):
        bad_sd2 = dict(m_ok.state_dict())
        bad_sd2["norm.weight"] = torch.ones(99)
        execute_verified_folded_norm_calibration(m_ok, chunks_ok, base_state_dict=bad_sd2, score_from=4)
    with pytest.raises(ValueError, match="non-finite"):
        bad_sd3 = dict(m_ok.state_dict())
        bad_sd3["norm.weight"] = torch.full_like(bad_sd3["norm.weight"], float("nan"))
        execute_verified_folded_norm_calibration(m_ok, chunks_ok, base_state_dict=bad_sd3, score_from=4)
    with pytest.raises(ValueError, match="base_domain_nll"):
        execute_verified_folded_norm_calibration(m_ok, chunks_ok, base_domain_nll={}, score_from=4)
    with pytest.raises(ValueError, match="missing verification domains"):
        execute_verified_folded_norm_calibration(
            m_ok, chunks_ok, base_domain_nll={"D0_general_prose": 1.0}, score_from=4
        )
    with pytest.raises(TypeError, match="mapping"):
        verify_folded_norm_receipt_provenance("bad")  # type: ignore[arg-type]

    # Pre-hook errors: norm never called or mismatched shape
    class _SkipNormModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.RMSNorm(6)
            self.lm_head = nn.Linear(6, 8, bias=False)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return torch.zeros((ids.shape[0], ids.shape[1], 8))

    skip_chunks = [
        {"domain": "D0_g", "ids": [0, 1, 2, 3, 4]},
        {"domain": "D1_s", "ids": [0, 1, 2, 3, 4]},
    ]
    with pytest.raises(RuntimeError, match="not invoked"):
        execute_verified_folded_norm_calibration(_SkipNormModel(), skip_chunks, score_from=2)

    class _BadShapeNormModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.RMSNorm(6)
            self.lm_head = nn.Linear(6, 8, bias=False)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return self.lm_head(self.norm(torch.ones((ids.shape[0], 6))))

    with pytest.raises(ValueError, match="incompatible"):
        execute_verified_folded_norm_calibration(_BadShapeNormModel(), skip_chunks, score_from=2)

    # 5. AGPL-3.0, ZeroStubAudit (CI-safe), PRSecurityScanner (author_email=None), lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn37", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path(
        "pr_gatekeeper_dyn37", REPO_ROOT / "scripts" / "pr_gatekeeper.py"
    )
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/folded_norm_ece_calibrator.py", TARGET_MODULE_PATH),
        ("tests/test_folded_norm_ece_calibrator_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"
        assert fallback_auditor.check_file(str(path_obj)) == []
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

    telemetry_json = json.dumps(
        {
            "nll_mean": 2.2565,
            "expected_calibration_error": 0.0138,
            "edited_param_fraction": 9.95e-7,
            "anse_energy": -0.0090,
        },
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+TELEMETRY = {telemetry_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1237,
        pr_title="feat(swarm): zero-overhead folded final RMSNorm/LayerNorm temperature and ECE calibration surgeon for open-weight models",
        pr_body="Implements NEURAL-CTRL-DYN-37 folded final norm temperature and ECE calibration surgeon with exact logit equivalence, ANSE stage ratchet, and SHA-256 provenance.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): zero-overhead folded final RMSNorm/LayerNorm temperature and ECE calibration surgeon for open-weight models"
        ],
        author_email=None,
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []

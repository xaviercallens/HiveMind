# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-52: Effective-DOF BIC/MDL Arm Selector & Two-Stage Promoter."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.security.owner_order_guardrail import (
    RequestIntentCategory,
    classify_request_intent,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.effective_dof_bic_arm_selector import (
    ANSE_MAXIMUM_PAIN,
    SELECTOR_RECEIPT_OBJECT,
    SELECTOR_SCHEMA,
    EffectiveDOFBICSelectorConfig,
    compute_effective_dof_and_bic_penalty,
    execute_verified_bic_promoted_wsp,
    select_variance_guarded_champion,
    verify_bic_arm_selector_receipt_provenance,
)
from hivemind.swarm.logit_calibration_temperature_surgeon import compute_module_state_sha256

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover - optional in lightweight CI runners
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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "effective_dof_bic_arm_selector.py"
)
TEST_FILE_PATH = Path(__file__).resolve()
ROADMAP_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_ROADMAP.md"
PLAN_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md"
EXPECTED_AGPL_HEADER = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)


@pytest.fixture(autouse=True, scope="module")
def _single_thread_cpu_torch() -> Any:
    """Eliminate OpenMP thread-pool spin on small deterministic CPU tensors."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev_threads)


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _VarianceEpsRMSNorm(nn.Module):
    """Qwen2.5-style biasless RMSNorm exposing ``variance_epsilon``."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.variance_epsilon = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (int(self.weight.shape[0]),), self.weight, self.variance_epsilon)


class _BICPromotionTestLM(nn.Module):
    """Deterministic causal LM with orthogonal D0 prose, low-rank specialist axes, and high-DOF nuisance dims.

    Supports both ``RMSNorm`` and ``LayerNorm`` + ``lm_head.bias``, plus controllable
    split-dependent nuisance sign (``probe/selection`` spurious positive correlation vs
    ``confirm/eval`` reversal) or pure parity zero-SNR nuisance.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 80,
        vocab_size: int = 40,
        d0_logit_scale: float = 0.72,
        spec_norm_init: float = 0.12,
        spec_head_gain: float = 0.65,
        nuisance_amp: float = 0.0,
        nuisance_mode: str = "spurious_shift",
        use_layernorm_with_bias: bool = False,
        seed: int = 20265201,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.nuisance_amp = float(nuisance_amp)
        self.nuisance_mode = nuisance_mode
        gen = torch.Generator(device="cpu").manual_seed(seed)

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.norm: nn.Module = (
            nn.LayerNorm(hidden_size, eps=1e-5)
            if use_layernorm_with_bias
            else _VarianceEpsRMSNorm(hidden_size, eps=1e-6)
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=use_layernorm_with_bias)

        with torch.no_grad():
            self.embed.weight.zero_()
            self.lm_head.weight.zero_()
            self.norm.weight.fill_(1.0)
            if isinstance(self.norm, nn.LayerNorm):
                self.norm.bias.fill_(0.02)
            if self.lm_head.bias is not None:
                self.lm_head.bias.copy_(0.03 * torch.randn((vocab_size,), generator=gen))

            # D0 general prose occupies dims 0..3 (tokens 0..9)
            for tok in range(0, 10):
                self.embed.weight[tok, 0:4] = torch.tensor([1.5, 1.0, 0.6, 0.3], dtype=torch.float32)
                if tok == 1:
                    self.lm_head.weight[tok, 0:4] = float(d0_logit_scale) * torch.tensor(
                        [1.1, 0.7, 0.4, 0.2], dtype=torch.float32
                    )
                elif tok == 2:
                    self.lm_head.weight[tok, 0:4] = float(d0_logit_scale) * torch.tensor(
                        [0.35, 0.25, 0.15, 0.05], dtype=torch.float32
                    )
                else:
                    self.lm_head.weight[tok, 0:4] = float(d0_logit_scale) * torch.tensor(
                        [-0.45, -0.30, -0.20, -0.10], dtype=torch.float32
                    )

            # Specialists D1..D3 occupy low-rank subspace dims 8..11 inside ker(G_D0)
            # OFC-protected channels 5, 6 remain disjoint from active axes
            if spec_norm_init > 0.0:
                self.norm.weight[8:12].fill_(float(spec_norm_init))
            for d_idx in range(1, 4):
                tok_lo = 10 * d_idx
                for tok in range(tok_lo, tok_lo + 10):
                    self.embed.weight[tok, 8:12] = torch.tensor([2.2, 1.8, 1.4, 1.0], dtype=torch.float32)
                    if tok == tok_lo + 1:
                        self.lm_head.weight[tok, 8:12] = float(spec_head_gain) * torch.tensor(
                            [0.85, 0.65, 0.45, 0.25], dtype=torch.float32
                        )
                    elif tok == tok_lo + 2:
                        self.lm_head.weight[tok, 8:12] = float(spec_head_gain) * torch.tensor(
                            [0.25, 0.18, 0.12, 0.06], dtype=torch.float32
                        )
                    else:
                        self.lm_head.weight[tok, 8:12] = float(spec_head_gain) * torch.tensor(
                            [-0.35, -0.25, -0.18, -0.10], dtype=torch.float32
                        )

            # High-DOF nuisance head weights on dims 16..hidden_size
            if hidden_size > 16 and nuisance_amp > 0.0:
                if nuisance_mode == "spurious_shift":
                    for d_idx in range(1, 4):
                        tok_lo = 10 * d_idx
                        self.lm_head.weight[tok_lo + 1, 16:] = 0.04
                        self.lm_head.weight[tok_lo + 2, 16:] = 0.01
                        self.lm_head.weight[tok_lo + 3 : tok_lo + 10, 16:] = -0.02
                else:
                    nuis_w = 0.03 * torch.randn((vocab_size, hidden_size - 16), generator=gen, dtype=torch.float32)
                    self.lm_head.weight[10:, 16:] = nuis_w[10:]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids).clone()
        if self.hidden_size > 16 and self.nuisance_amp > 0.0:
            is_spec = (input_ids[:, :1] >= 10).to(dtype=h.dtype).unsqueeze(-1)
            if self.nuisance_mode == "spurious_shift":
                # Token at position 0 encodes split polarity: even -> +1 (probe/selection), odd -> -1 (confirm/eval)
                split_sign = (1.0 - 2.0 * (input_ids[:, :1] % 2).to(dtype=h.dtype)).unsqueeze(-1)
                h[..., 16:] = h[..., 16:] + self.nuisance_amp * split_sign * is_spec
            elif self.nuisance_mode == "zero_snr_parity":
                # Deterministic per-chunk sign parity cancelling exactly across probe chunks
                chunk_par = (1.0 - 2.0 * (input_ids[:, :1] % 2).to(dtype=h.dtype)).unsqueeze(-1)
                h[..., 16:] = h[..., 16:] + self.nuisance_amp * chunk_par * is_spec
        return self.lm_head(self.norm(h))


def _make_split_chunks(
    *,
    n_per_domain: int,
    seq_len: int = 18,
    seed: int = 20265210,
    d0_modal_prob: float = 0.72,
    spec_modal_prob: float = 0.80,
    split_parity_bit: int = 0,
    alternate_parity: bool = False,
) -> list[dict[str, Any]]:
    """Generate deterministic 4-domain next-token chunks."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    specs = (
        ("D0_general_prose", 0, d0_modal_prob),
        ("D1_python_systems_code", 10, spec_modal_prob),
        ("D2_formal_math_lean4", 20, spec_modal_prob),
        ("D3_computational_physics", 30, spec_modal_prob),
    )
    chunks: list[dict[str, Any]] = []
    for dom, lo, p_modal in specs:
        for c_idx in range(n_per_domain):
            p_bit = (c_idx % 2) if alternate_parity else split_parity_bit
            prefix_tok = lo + (2 if p_bit == 0 else 3)
            seq: list[int] = [prefix_tok]
            for pos in range(1, seq_len):
                u = float(torch.rand((), generator=gen).item())
                if u < p_modal:
                    seq.append(lo + 1)
                elif u < p_modal + 0.12:
                    seq.append(lo + 2)
                else:
                    seq.append(lo + 3 + (pos % 6))
            chunks.append({"domain": dom, "ids": seq})
    return chunks


# ---------------------------------------------------------------------------
# Scenario 1: PILOT16 Replay & Live End-to-End BIC Promotion vs Full-DOF Overfit
# ---------------------------------------------------------------------------


def test_s1_pilot16_replay_blocks_noisy_full_dof_and_promotes_js_rank4() -> None:
    """S1: PILOT16 replay where Full-DOF edges JS-Rank-4 by 0.000287 nats -> BIC promotes JS-Rank-4."""
    # Exact PILOT16 telemetry (Qwen/Qwen2.5-1.5B, 5,080 selection tokens, H=1536, mean_shrinkage=0.2989)
    js4_shrinkage = [0.2989, 0.2989, 0.2989, 0.2989]
    js8_shrinkage = [0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28]

    dof_js4 = compute_effective_dof_and_bic_penalty(
        "james_stein_rank_k", rank_k=4, full_dim=1536, n_sel_tokens=5080, shrinkage_factors=js4_shrinkage
    )
    dof_full = compute_effective_dof_and_bic_penalty(
        "full_dof", rank_k=4, full_dim=1536, n_sel_tokens=5080
    )
    assert dof_js4["effective_dof"] == pytest.approx(sum(js4_shrinkage), rel=1e-9)
    assert dof_full["effective_dof"] == 1536.0
    assert dof_full["bic_penalty"] == pytest.approx(
        1536.0 * math.log(5080.0) / (2.0 * 5080.0), rel=1e-9
    )

    pilot16_selection = {
        "ARM3_CROSSFIT_JS_RANK4_OFC_15B": {
            "arm_kind": "james_stein_rank_k",
            "rank_k": 4,
            "full_dim": 1536,
            "n_sel_tokens": 5080,
            "shrinkage_factors": js4_shrinkage,
            "nll_diff": -0.007977,
            "ece_diff": -0.0170,
            "general_drift_ratio": -0.0038,
            "paired_se": 0.00147,
        },
        "ARM4_CROSSFIT_JS_RANK8_OFC_15B": {
            "arm_kind": "james_stein_rank_k",
            "rank_k": 8,
            "full_dim": 1536,
            "n_sel_tokens": 5080,
            "shrinkage_factors": js8_shrinkage,
            "nll_diff": -0.008137,
            "ece_diff": -0.0160,
            "general_drift_ratio": -0.0040,
            "paired_se": 0.00148,
        },
        "ARM2_UNREG_FULLDOF_1536_FISHER_15B": {
            "arm_kind": "full_dof",
            "full_dim": 1536,
            "n_sel_tokens": 5080,
            "nll_diff": -0.008264,  # Narrowly edges JS-Rank-4 by 0.000287 nats on selection
            "ece_diff": -0.0165,
            "general_drift_ratio": -0.00491,
            "paired_se": 0.00150,
        },
    }
    pilot16_confirm = {
        "ARM2_UNREG_FULLDOF_1536_FISHER_15B": {"nll_diff": -0.006150, "ece_rel_delta": -0.3179},
        "ARM1_SCALAR_FOLDED_RMSNORM_15B": {"nll_diff": -0.006789, "ece_rel_delta": -0.3407},
        "ARM4_CROSSFIT_JS_RANK8_OFC_15B": {"nll_diff": -0.007301, "ece_rel_delta": -0.3600},
        "ARM3_CROSSFIT_JS_RANK4_OFC_15B": {"nll_diff": -0.007602, "ece_rel_delta": -0.3921},
    }

    # Unpenalized selection picks noisy ARM2_UNREG_FULLDOF_1536_FISHER_15B
    unpenalized_winner = min(pilot16_selection, key=lambda k: pilot16_selection[k]["nll_diff"])
    assert unpenalized_winner == "ARM2_UNREG_FULLDOF_1536_FISHER_15B"
    assert abs(pilot16_selection[unpenalized_winner]["nll_diff"] - (-0.007977)) == pytest.approx(0.000287, abs=1e-9)

    decision = select_variance_guarded_champion(
        pilot16_selection, ece_weight=0.05, general_guard_rel=0.010, bootstrap_margin_z=1.0
    )
    assert decision["selected_arm"] == "ARM3_CROSSFIT_JS_RANK4_OFC_15B"
    assert decision["status"] == "RETAINED_LOW_DOF_ANCHOR"
    assert decision["fallback_to_scalar"] is False
    assert decision["rejected_arms"]["ARM2_UNREG_FULLDOF_1536_FISHER_15B"] == "rejected_by_bic_complexity_penalty"
    assert decision["rejected_arms"]["ARM4_CROSSFIT_JS_RANK8_OFC_15B"] in (
        "rejected_by_bic_complexity_penalty",
        "rejected_by_paired_selection_variance_guard",
    )

    # Confirm comparison: BIC-promoted JS-Rank-4 beats unpenalized Full-DOF on BOTH NLL and ECE
    bic_conf = pilot16_confirm[decision["selected_arm"]]
    raw_conf = pilot16_confirm[unpenalized_winner]
    assert bic_conf["nll_diff"] < raw_conf["nll_diff"]
    assert bic_conf["ece_rel_delta"] < raw_conf["ece_rel_delta"]

    # Including 1-DOF scalar also cleanly blocks Full-DOF and beats Full-DOF on confirm
    with_scalar = {
        **pilot16_selection,
        "ARM1_SCALAR_FOLDED_RMSNORM_15B": {
            "arm_kind": "scalar",
            "full_dim": 1536,
            "n_sel_tokens": 5080,
            "nll_diff": -0.008175,
            "ece_diff": -0.0110,
            "general_drift_ratio": -0.0041,
            "paired_se": 0.00145,
        },
    }
    dec_with_scalar = select_variance_guarded_champion(
        with_scalar, ece_weight=0.05, general_guard_rel=0.010, bootstrap_margin_z=1.0
    )
    assert dec_with_scalar["selected_arm"] == "ARM1_SCALAR_FOLDED_RMSNORM_15B"
    assert pilot16_confirm[dec_with_scalar["selected_arm"]]["nll_diff"] < raw_conf["nll_diff"]
    assert pilot16_confirm[dec_with_scalar["selected_arm"]]["ece_rel_delta"] < raw_conf["ece_rel_delta"]

    # Live PyTorch CPU end-to-end replay where unreg_full_dof overfits spurious selection dims and drops on confirm
    model = _BICPromotionTestLM(
        hidden_size=80,
        vocab_size=40,
        d0_logit_scale=0.72,
        spec_norm_init=0.05,
        spec_head_gain=0.80,
        nuisance_amp=0.14,
        nuisance_mode="spurious_shift",
    )
    base_sha = compute_module_state_sha256(model)
    probe_ch = _make_split_chunks(n_per_domain=8, seq_len=18, seed=5211, split_parity_bit=0)
    sel_ch = _make_split_chunks(n_per_domain=8, seq_len=18, seed=5212, split_parity_bit=0)
    eval_ch = _make_split_chunks(n_per_domain=8, seq_len=18, seed=5213, split_parity_bit=1)

    cfg = EffectiveDOFBICSelectorConfig(
        js_rank_primary=4,
        js_rank_secondary=8,
        include_secondary_js_rank=True,
        include_full_dof_control=True,
        bootstrap_margin_z=1.0,
        ece_energy_weight=0.05,
        max_step_norm=0.30,
    )
    edited, receipt = execute_verified_bic_promoted_wsp(
        model,
        probe_ch,
        selection_chunks=sel_ch,
        eval_chunks=eval_ch,
        protected_indices=[5, 6],
        config=cfg,
        inplace=False,
    )
    assert compute_module_state_sha256(model) == base_sha
    assert compute_module_state_sha256(edited) != base_sha
    assert receipt["object"] == SELECTOR_RECEIPT_OBJECT
    assert receipt["schema"] == SELECTOR_SCHEMA
    assert verify_bic_arm_selector_receipt_provenance(receipt) is True
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["selected_arm"] == "crossfit_js_rank_4"
    assert receipt["ofc_protected_max_delta"] == 0.0
    assert receipt["d0_mean_grad_ortho_err"] < 1e-6
    assert receipt["general_drift_ratio"] <= 0.010
    assert receipt["delta_e"] < 0.0
    assert receipt["selection_decision"]["rejected_arms"]["unreg_full_dof"] == "rejected_by_bic_complexity_penalty"
    js4_eval = receipt["eval_arm_comparison"]["crossfit_js_rank_4"]
    full_eval = receipt["eval_arm_comparison"]["unreg_full_dof"]
    assert js4_eval["nll_mean"] < full_eval["nll_mean"]
    assert js4_eval["ece_value"] < full_eval["ece_value"]


# ---------------------------------------------------------------------------
# Scenario 2: High-SNR Regime Lawfully Promotes Higher-DOF Arm
# ---------------------------------------------------------------------------


def test_s2_high_snr_regime_lawfully_promotes_higher_dof_arm() -> None:
    """S2: When held-out improvement clears both BIC penalty and paired SE, higher-DOF arm is promoted."""
    # Direct paired per-chunk selection verification
    gen = torch.Generator(device="cpu").manual_seed(5220)
    noise = 0.002 * torch.randn((40,), generator=gen, dtype=torch.float64)
    scalar_chunks = -0.004 + noise
    js4_chunks = -0.036 + noise * 0.8

    paired_metrics = {
        "scalar_1dof": {
            "arm_kind": "scalar",
            "full_dim": 768,
            "n_sel_tokens": 2048,
            "per_chunk_nll_diffs": scalar_chunks,
            "ece_diff": -0.005,
            "general_drift_ratio": -0.002,
        },
        "js_rank4": {
            "arm_kind": "james_stein",
            "rank_k": 4,
            "full_dim": 768,
            "n_sel_tokens": 2048,
            "shrinkage_factors": [0.85, 0.80, 0.72, 0.65],
            "per_chunk_nll_diffs": js4_chunks,
            "ece_diff": -0.024,
            "general_drift_ratio": -0.004,
        },
    }
    dec = select_variance_guarded_champion(paired_metrics, ece_weight=0.05, bootstrap_margin_z=1.0)
    assert dec["selected_arm"] == "js_rank4"
    assert dec["status"] == "PROMOTED_HIGHER_DOF"
    assert dec["fallback_to_scalar"] is False
    assert len(dec["promotion_trace"]) == 1
    assert dec["promotion_trace"][0]["promoted"] is True
    assert dec["promotion_trace"][0]["reason"] == "cleared_bic_and_paired_variance_guard"

    # End-to-end LayerNorm + head bias model with strong multi-axis ker(G_D0) specialist signal
    model = _BICPromotionTestLM(
        hidden_size=48,
        vocab_size=40,
        d0_logit_scale=0.72,
        spec_norm_init=0.04,
        spec_head_gain=0.95,
        nuisance_amp=0.0,
        use_layernorm_with_bias=True,
        seed=5221,
    )
    probe_ch = _make_split_chunks(n_per_domain=8, seq_len=16, seed=5222)
    sel_ch = _make_split_chunks(n_per_domain=8, seq_len=16, seed=5223)
    eval_ch = _make_split_chunks(n_per_domain=8, seq_len=16, seed=5224)

    cfg = EffectiveDOFBICSelectorConfig(
        js_rank_primary=4,
        include_secondary_js_rank=False,
        include_full_dof_control=False,
        max_step_norm=0.35,
        bootstrap_margin_z=1.0,
    )
    edited, receipt = execute_verified_bic_promoted_wsp(
        model,
        probe_ch,
        selection_chunks=sel_ch,
        eval_chunks=eval_ch,
        protected_indices=torch.tensor([5, 6], dtype=torch.long),
        config=cfg,
        norm_kind="layernorm",
        inplace=True,
    )
    assert verify_bic_arm_selector_receipt_provenance(receipt) is True
    assert receipt["accepted"] is True
    assert receipt["norm_kind"] == "layernorm"
    assert receipt["selected_arm"] == "crossfit_js_rank_4"
    assert receipt["selection_decision"]["status"] == "PROMOTED_HIGHER_DOF"
    assert receipt["effective_dof"] >= 1.0
    assert receipt["bic_penalty"] > 0.0
    assert receipt["nll_mean"] < receipt["base_nll_mean"]
    assert receipt["delta_e"] < 0.0
    assert receipt["ofc_protected_max_delta"] == 0.0
    assert torch.allclose(
        edited.norm.weight[5:7].detach(),
        torch.full((2,), float(receipt["tau_temp"]), dtype=edited.norm.weight.dtype),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Scenario 3: Effective-DOF Monotonicity & Zero-SNR Scalar Fallback
# ---------------------------------------------------------------------------


def test_s3_effective_dof_monotonicity_and_zero_snr_contraction_to_scalar_fallback() -> None:
    """S3: Enforce 1-DOF scalar <= JS-Rank-4 sum(c_j) <= JS-Rank-8 <= Full-DOF & zero-SNR scalar fallback."""
    d_base = compute_effective_dof_and_bic_penalty("base", full_dim=1536, n_sel_tokens=5080)
    d_scalar = compute_effective_dof_and_bic_penalty("scalar", full_dim=1536, n_sel_tokens=5080)
    d_js4 = compute_effective_dof_and_bic_penalty(
        "james_stein_rank_k",
        rank_k=4,
        full_dim=1536,
        n_sel_tokens=5080,
        shrinkage_factors=[0.55, 0.45, 0.35, 0.25],
    )
    d_js8 = compute_effective_dof_and_bic_penalty(
        "james_stein_rank_k",
        rank_k=8,
        full_dim=1536,
        n_sel_tokens=5080,
        shrinkage_factors=None,
    )
    d_full = compute_effective_dof_and_bic_penalty("full_dof", full_dim=1536, n_sel_tokens=5080)

    assert d_base["effective_dof"] == 0.0
    assert d_scalar["effective_dof"] == 1.0
    assert d_js4["effective_dof"] == pytest.approx(1.60, rel=1e-9)
    assert d_js8["effective_dof"] == 8.0
    assert d_full["effective_dof"] == 1536.0
    assert (
        d_base["effective_dof"]
        < d_scalar["effective_dof"]
        <= d_js4["effective_dof"]
        <= d_js8["effective_dof"]
        <= d_full["effective_dof"]
    )
    assert d_scalar["bic_penalty"] <= d_js4["bic_penalty"] <= d_js8["bic_penalty"] <= d_full["bic_penalty"]

    # Zero-SNR James-Stein shrinkage contracts k_eff to 0.0 -> zero_snr_contracted=True -> scalar fallback
    d_zero = compute_effective_dof_and_bic_penalty(
        "james_stein_rank_k",
        rank_k=4,
        full_dim=1536,
        n_sel_tokens=5080,
        shrinkage_factors=torch.zeros(4, dtype=torch.float32),
    )
    assert d_zero["effective_dof"] == 0.0
    assert d_zero["zero_snr_contracted"] is True

    zero_snr_sel = select_variance_guarded_champion(
        {
            "scalar_1dof": {
                "arm_kind": "scalar",
                "full_dim": 1536,
                "n_sel_tokens": 5080,
                "nll_diff": -0.0063,
                "ece_diff": -0.012,
                "general_drift_ratio": -0.004,
                "paired_se": 0.001,
            },
            "js_rank4_zero_snr": {
                "arm_kind": "james_stein_rank_k",
                "rank_k": 4,
                "full_dim": 1536,
                "n_sel_tokens": 5080,
                "shrinkage_factors": [0.0, 0.0, 0.0, 0.0],
                "nll_diff": -0.0063,
                "ece_diff": -0.012,
                "general_drift_ratio": -0.004,
                "paired_se": 0.001,
            },
            "js_rank4_noisy_edge": {
                "arm_kind": "james_stein_rank_k",
                "rank_k": 4,
                "full_dim": 1536,
                "n_sel_tokens": 5080,
                "shrinkage_factors": [0.60, 0.50, 0.40, 0.30],
                "nll_diff": -0.0075,
                "ece_diff": -0.012,
                "general_drift_ratio": -0.004,
                "paired_se": 0.002,
            },
        },
        bootstrap_margin_z=1.0,
    )
    assert zero_snr_sel["selected_arm"] == "scalar_1dof"
    assert zero_snr_sel["fallback_to_scalar"] is True
    assert zero_snr_sel["status"] == "FELL_BACK_TO_SCALAR"
    assert zero_snr_sel["rejected_arms"]["js_rank4_zero_snr"] == "zero_snr_contracted_to_scalar"
    assert zero_snr_sel["rejected_arms"]["js_rank4_noisy_edge"] == "rejected_by_paired_selection_variance_guard"

    # End-to-end execution on under-confident Base model with zero specialist ker(G_D0) SNR -> scalar fallback
    model = _BICPromotionTestLM(
        hidden_size=48,
        vocab_size=40,
        d0_logit_scale=0.35,  # Under-confident Base prose -> tau > 1.0 scalar helps cleanly
        spec_norm_init=1.0,
        spec_head_gain=0.0,  # Zero specialist signal inside dims 8..11
        nuisance_amp=0.0,
        nuisance_mode="zero_snr_parity",
        seed=5230,
    )
    probe_ch = _make_split_chunks(
        n_per_domain=8, seq_len=14, seed=5231, d0_modal_prob=0.80, spec_modal_prob=0.60, alternate_parity=True
    )
    sel_ch = _make_split_chunks(
        n_per_domain=8, seq_len=14, seed=5232, d0_modal_prob=0.80, spec_modal_prob=0.60, alternate_parity=True
    )
    eval_ch = _make_split_chunks(
        n_per_domain=8, seq_len=14, seed=5233, d0_modal_prob=0.80, spec_modal_prob=0.60, alternate_parity=True
    )

    _, receipt = execute_verified_bic_promoted_wsp(
        model,
        probe_ch,
        selection_chunks=sel_ch,
        eval_chunks=eval_ch,
        config=EffectiveDOFBICSelectorConfig(
            tau_grid=(0.90, 1.00, 1.15, 1.35, 1.60),
            include_secondary_js_rank=False,
            include_full_dof_control=False,
        ),
        inplace=False,
    )
    assert verify_bic_arm_selector_receipt_provenance(receipt) is True
    assert receipt["accepted"] is True
    assert receipt["selected_arm"] == "scalar_folded_tau"
    assert receipt["effective_dof"] == 1.0
    assert receipt["tau_temp"] > 1.0
    assert receipt["alpha_scale"] == 0.0


# ---------------------------------------------------------------------------
# Scenario 4: ANSE Maximum Pain Atomic Bitwise Rollback on Infeasible Guard
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_bitwise_rollback_and_tamper_detection() -> None:
    """S4: Guard violation or non-improving eval triggers ANSE Maximum Pain and bitwise rollback."""
    # Case A: All selection arms violate D0 guard -> NO_FEASIBLE_ARM with ANSE_MAXIMUM_PAIN
    infeasible_sel = select_variance_guarded_champion(
        {
            "arm_bad_d0": {
                "arm_kind": "scalar",
                "n_sel_tokens": 1024,
                "nll_diff": -0.020,
                "general_drift_ratio": 0.025,
            },
            "arm_bad_risk": {
                "arm_kind": "james_stein",
                "n_sel_tokens": 1024,
                "nll_diff": 0.005,
                "general_drift_ratio": 0.0,
            },
        },
        general_guard_rel=0.010,
    )
    assert infeasible_sel["selected_arm"] is None
    assert infeasible_sel["status"] == "NO_FEASIBLE_ARM"
    assert infeasible_sel["anse_energy"] >= ANSE_MAXIMUM_PAIN

    # Case B: Only high-DOF arm fails its own BIC penalty vs base -> NO_COMPLEXITY_JUSTIFIED_ARM
    high_dof_only = select_variance_guarded_champion(
        {
            "base_ref": {"arm_kind": "base", "nll_diff": 0.0, "n_sel_tokens": 1024},
            "arm_full_overfit": {
                "arm_kind": "full_dof",
                "full_dim": 1536,
                "n_sel_tokens": 1024,
                "nll_diff": -0.014,
                "general_drift_ratio": 0.004,
            },
        }
    )
    assert high_dof_only["selected_arm"] is None
    assert high_dof_only["status"] == "NO_COMPLEXITY_JUSTIFIED_ARM"

    # Case C: Live model hits D0 guard violation on shifted eval -> ANSE Maximum Pain + bitwise rollback
    model = _BICPromotionTestLM(
        hidden_size=48,
        vocab_size=40,
        d0_logit_scale=0.35,
        spec_norm_init=0.05,
        spec_head_gain=0.80,
        seed=5240,
    )
    init_sha = compute_module_state_sha256(model)
    init_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    p_ch = _make_split_chunks(n_per_domain=6, seq_len=12, seed=5241, d0_modal_prob=0.82, spec_modal_prob=0.82)
    s_ch = _make_split_chunks(n_per_domain=6, seq_len=12, seed=5242, d0_modal_prob=0.82, spec_modal_prob=0.82)
    e_ch = _make_split_chunks(n_per_domain=6, seq_len=12, seed=5243, d0_modal_prob=0.30, spec_modal_prob=0.30)

    _, rec_guard = execute_verified_bic_promoted_wsp(
        model,
        p_ch,
        selection_chunks=s_ch,
        eval_chunks=e_ch,
        config=EffectiveDOFBICSelectorConfig(
            tau_grid=(1.0, 1.4, 2.0),
            max_general_drift_rel=0.001,
            max_domain_drift_rel=0.05,
            include_secondary_js_rank=False,
        ),
        inplace=True,
    )
    assert verify_bic_arm_selector_receipt_provenance(rec_guard) is True
    assert rec_guard["accepted"] is False
    assert rec_guard["rolled_back"] is True
    assert rec_guard["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rec_guard["cand_anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_guard["final_state_sha256"] == init_sha
    for k, orig in init_sd.items():
        assert torch.equal(model.state_dict()[k], orig)

    # Case D: Relaxed guard on non-improving eval -> ROLLED_BACK_ENERGY_NON_IMPROVING
    _, rec_nonimp = execute_verified_bic_promoted_wsp(
        model,
        p_ch,
        selection_chunks=s_ch,
        eval_chunks=e_ch,
        config=EffectiveDOFBICSelectorConfig(
            tau_grid=(1.0, 1.4, 2.0),
            max_general_drift_rel=5.0,
            max_domain_drift_rel=5.0,
            include_secondary_js_rank=False,
        ),
        inplace=True,
    )
    assert rec_nonimp["accepted"] is False
    assert rec_nonimp["rolled_back"] is True
    assert rec_nonimp["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert rec_nonimp["final_state_sha256"] == init_sha

    # Case E: Fixed point unchanged (tau_grid=(1.0,), alpha_grid=(0.0,), automatic probe split fallback)
    auto_ch = _make_split_chunks(n_per_domain=4, seq_len=12, seed=5244)
    _, rec_fixed = execute_verified_bic_promoted_wsp(
        model,
        auto_ch,
        config=EffectiveDOFBICSelectorConfig(
            tau_grid=(1.0,),
            alpha_grid=(0.0,),
            include_secondary_js_rank=False,
            include_full_dof_control=False,
            require_strict_energy_drop=False,
        ),
        inplace=True,
    )
    assert rec_fixed["accepted"] is False
    assert rec_fixed["rolled_back"] is False
    assert rec_fixed["status"].startswith(("FIXED_POINT_UNCHANGED", "REJECTED_"))
    assert rec_fixed["final_state_sha256"] == init_sha

    # Case F: Tampered receipt fails provenance verification
    tampered = dict(rec_guard)
    tampered["nll_mean"] = float(rec_guard["nll_mean"]) + 0.1
    assert verify_bic_arm_selector_receipt_provenance(tampered) is False


# ---------------------------------------------------------------------------
# Scenario 5: Validation Errors & Security / Stub / OwnerOrderGuardrail Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_guardrail_compliance() -> None:
    """S5: Exhaustive validation checks plus AGPL-3.0, ZeroStubAudit, PRSecurityScanner & OwnerOrderGuardrail."""
    # 1. Config & effective-DOF validation
    with pytest.raises(ValueError, match="js_rank_secondary"):
        EffectiveDOFBICSelectorConfig(js_rank_primary=4, js_rank_secondary=4)
    with pytest.raises(ValueError, match="< 1.0"):
        EffectiveDOFBICSelectorConfig(nullspace_sv_tol=1.0)
    with pytest.raises(ValueError, match="<= 1.0"):
        EffectiveDOFBICSelectorConfig(min_shrinkage_to_activate=1.5)
    with pytest.raises(TypeError, match="non-empty sequence"):
        EffectiveDOFBICSelectorConfig(tau_grid="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        EffectiveDOFBICSelectorConfig(alpha_grid=())
    with pytest.raises(TypeError, match="non-empty sequence"):
        EffectiveDOFBICSelectorConfig(general_prefixes="D0_")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        EffectiveDOFBICSelectorConfig(general_prefixes=())
    with pytest.raises(TypeError, match="bool"):
        EffectiveDOFBICSelectorConfig(include_full_dof_control="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        EffectiveDOFBICSelectorConfig(js_rank_primary=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        EffectiveDOFBICSelectorConfig(js_rank_primary=0)
    with pytest.raises(ValueError, match=">= 0"):
        EffectiveDOFBICSelectorConfig(score_from=-1)
    with pytest.raises(ValueError, match="finite"):
        EffectiveDOFBICSelectorConfig(bootstrap_margin_z=float("nan"))

    cfg_norm = EffectiveDOFBICSelectorConfig(tau_grid=(1.1,), alpha_grid=(0.5,))
    assert 1.0 in cfg_norm.tau_grid and 0.0 in cfg_norm.alpha_grid

    with pytest.raises(ValueError, match="Unsupported arm_kind"):
        compute_effective_dof_and_bic_penalty("unknown_arm")
    with pytest.raises(ValueError, match="<= full_dim"):
        compute_effective_dof_and_bic_penalty("james_stein", rank_k=16, full_dim=8)
    with pytest.raises(ValueError, match=">= 2"):
        compute_effective_dof_and_bic_penalty("scalar", n_sel_tokens=1)
    with pytest.raises(TypeError, match="shrinkage_factors"):
        compute_effective_dof_and_bic_penalty("james_stein", shrinkage_factors="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1D floating-point"):
        compute_effective_dof_and_bic_penalty("james_stein", shrinkage_factors=torch.ones((2, 2)))
    with pytest.raises(ValueError, match="exceeds full_dim"):
        compute_effective_dof_and_bic_penalty("james_stein", rank_k=2, full_dim=2, shrinkage_factors=[0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="lie in"):
        compute_effective_dof_and_bic_penalty("james_stein", shrinkage_factors=[1.5])

    # 2. select_variance_guarded_champion validation & flexible telemetry formats
    with pytest.raises(ValueError, match="non-empty mapping"):
        select_variance_guarded_champion({})
    with pytest.raises(TypeError, match="Mapping"):
        select_variance_guarded_champion({"a": "bad"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="nll_diff"):
        select_variance_guarded_champion({"a": {"arm_kind": "scalar", "general_drift_ratio": 0.0}})
    with pytest.raises(ValueError, match="arm_kind"):
        select_variance_guarded_champion({"a": {"nll_diff": -0.01, "general_drift_ratio": 0.0}})
    with pytest.raises(ValueError, match="general_drift_ratio"):
        select_variance_guarded_champion({"a": {"arm_kind": "scalar", "nll_diff": -0.01}})
    with pytest.raises(ValueError, match="ci95_hi"):
        select_variance_guarded_champion(
            {"a": {"arm_kind": "scalar", "nll_diff": -0.01, "general_drift_ratio": 0.0, "ci95_lo": 0.1, "ci95_hi": -0.1}}
        )
    with pytest.raises(ValueError, match="both per_chunk"):
        select_variance_guarded_champion(
            {"a": {"arm_kind": "scalar", "per_chunk_cand_nll": [1.0, 1.0], "general_drift_ratio": 0.0}}
        )
    with pytest.raises(ValueError, match="share length"):
        select_variance_guarded_champion(
            {
                "a": {
                    "arm_kind": "scalar",
                    "per_chunk_cand_nll": [1.0, 1.0],
                    "per_chunk_base_nll": [1.0],
                    "general_drift_ratio": 0.0,
                }
            }
        )
    with pytest.raises(ValueError, match="non-empty"):
        select_variance_guarded_champion(
            {"a": {"arm_kind": "scalar", "per_chunk_nll_diffs": [], "general_drift_ratio": 0.0}}
        )
    with pytest.raises(TypeError, match="per_chunk_nll_diffs"):
        select_variance_guarded_champion(
            {"a": {"arm_kind": "scalar", "per_chunk_nll_diffs": "bad", "general_drift_ratio": 0.0}}
        )
    with pytest.raises(ValueError, match=">= 2"):
        select_variance_guarded_champion(
            {"a": {"effective_dof": 1.0, "n_sel_tokens": 1, "nll_diff": -0.01, "general_drift_ratio": 0.0}}
        )

    flex_sel = select_variance_guarded_champion(
        {
            "cand_a": {
                "effective_dof": 1.0,
                "n_sel_tokens": 1000,
                "per_chunk_cand_nll": [1.9, 1.9, 1.9],
                "per_chunk_base_nll": [2.0, 2.0, 2.0],
                "ece_value": 0.02,
                "base_ece_value": 0.03,
                "general_pct_vs_base": -0.5,
            },
            "cand_b": {
                "effective_dof": 1.0,
                "n_sel_tokens": 1000,
                "nll_mean": 1.85,
                "base_nll_mean": 2.00,
                "ece_diff": -0.01,
                "general_drift_ratio": -0.002,
                "ci95_lo": -0.16,
                "ci95_hi": -0.14,
            },
        }
    )
    assert flex_sel["selected_arm"] == "cand_b"

    # 3. Executor & provenance validation
    m_ok = _BICPromotionTestLM(hidden_size=24, vocab_size=40, seed=5250)
    good_ch = _make_split_chunks(n_per_domain=4, seq_len=8, seed=5251)
    with pytest.raises(TypeError, match="EffectiveDOFBICSelectorConfig"):
        execute_verified_bic_promoted_wsp(m_ok, good_ch, config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="nn.Module"):
        execute_verified_bic_promoted_wsp("bad", good_ch)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="norm_kind"):
        execute_verified_bic_promoted_wsp(m_ok, good_ch, norm_kind="bad")
    with pytest.raises(ValueError, match="final normalization"):
        execute_verified_bic_promoted_wsp(nn.Linear(4, 4), good_ch)
    with pytest.raises(ValueError, match="unembedding head"):
        no_h = nn.Module()
        no_h.norm = nn.RMSNorm(4)
        execute_verified_bic_promoted_wsp(no_h, good_ch)
    with pytest.raises(ValueError, match="must match norm dim"):
        bad_hw = nn.Module()
        bad_hw.norm = nn.RMSNorm(4)
        bad_hw.head = nn.Linear(5, 8, bias=False)
        execute_verified_bic_promoted_wsp(bad_hw, good_ch)
    with pytest.raises(ValueError, match="out of bounds"):
        execute_verified_bic_promoted_wsp(m_ok, good_ch, protected_indices=[999])
    with pytest.raises(ValueError, match="1D integer"):
        execute_verified_bic_promoted_wsp(m_ok, good_ch, protected_indices=torch.ones((2, 2), dtype=torch.long))
    with pytest.raises(TypeError, match="sequence or 1D"):
        execute_verified_bic_promoted_wsp(m_ok, good_ch, protected_indices="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty sequence"):
        execute_verified_bic_promoted_wsp(m_ok, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 4"):
        execute_verified_bic_promoted_wsp(m_ok, good_ch[:2])
    with pytest.raises(ValueError, match="Automatic"):
        execute_verified_bic_promoted_wsp(
            m_ok,
            [
                {"domain": "D0_g", "ids": [0, 1, 2, 3]},
                {"domain": "D1_s", "ids": [10, 11, 12, 13]},
                {"domain": "D0_g", "ids": [0, 1, 2, 3]},
                {"domain": "D0_g", "ids": [0, 1, 2, 3]},
            ],
        )
    with pytest.raises(ValueError, match="both general"):
        execute_verified_bic_promoted_wsp(
            m_ok,
            [{"domain": "D0_a", "ids": [0, 1, 2, 3]}, {"domain": "D0_b", "ids": [0, 1, 2, 3]}],
            selection_chunks=good_ch,
        )
    with pytest.raises(TypeError, match="integer dtype"):
        execute_verified_bic_promoted_wsp(m_ok, [{"domain": "D0_g", "ids": torch.ones(4)}], selection_chunks=good_ch)
    with pytest.raises(TypeError, match="sequence or 1D"):
        execute_verified_bic_promoted_wsp(m_ok, [{"domain": "D0_g", "ids": "bad"}], selection_chunks=good_ch)
    with pytest.raises(ValueError, match="1D"):
        execute_verified_bic_promoted_wsp(
            m_ok, [{"domain": "D0_g", "ids": torch.ones((2, 2), dtype=torch.long)}], selection_chunks=good_ch
        )
    with pytest.raises(ValueError, match="length >= 2"):
        execute_verified_bic_promoted_wsp(m_ok, [{"domain": "D0_g", "ids": [1]}], selection_chunks=good_ch)
    with pytest.raises(ValueError, match="token IDs"):
        execute_verified_bic_promoted_wsp(m_ok, [{"domain": "D0_g", "ids": [0, 999]}], selection_chunks=good_ch)
    with pytest.raises(ValueError, match="too short"):
        execute_verified_bic_promoted_wsp(
            m_ok,
            good_ch,
            selection_chunks=good_ch,
            config=EffectiveDOFBICSelectorConfig(score_from=20),
        )
    with pytest.raises(ValueError, match="share sequence length"):
        execute_verified_bic_promoted_wsp(
            m_ok,
            [{"domain": "D0_g", "ids": [0, 1, 2, 3]}, {"domain": "D1_s", "ids": [10, 11, 12]}],
            selection_chunks=good_ch,
        )
    with pytest.raises(TypeError, match="Mapping"):
        verify_bic_arm_selector_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_bic_arm_selector_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_bic_arm_selector_receipt_provenance({"receipt_sha256": "0" * 64, "object": "bad"}) is False
    assert (
        verify_bic_arm_selector_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": SELECTOR_RECEIPT_OBJECT,
                "schema": SELECTOR_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )

    # Custom norm_attr / head_attr resolution check
    _, rec_attr = execute_verified_bic_promoted_wsp(
        m_ok,
        good_ch,
        selection_chunks=good_ch,
        norm_attr="norm",
        head_attr="lm_head",
        config=EffectiveDOFBICSelectorConfig(tau_grid=(1.0,), alpha_grid=(0.0,), include_secondary_js_rank=False),
        inplace=False,
    )
    assert rec_attr["norm_attr"] == "norm" and rec_attr["head_attr"] == "lm_head"

    # 4. AGPL-3.0, ZeroStubAudit (CI-safe), PRSecurityScanner, lateral movement, FORBIDDEN_ADDED & OwnerOrderGuardrail
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn52", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn52", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/effective_dof_bic_arm_selector.py", TARGET_MODULE_PATH),
        ("tests/test_effective_dof_bic_arm_selector_scenarios.py", TEST_FILE_PATH),
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

    for doc_rel, doc_path in (
        ("docs/NEURO_CONTROLLER_ROADMAP.md", ROADMAP_DOC_PATH),
        ("docs/NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md", PLAN_DOC_PATH),
    ):
        doc_text = doc_path.read_text(encoding="utf-8")
        assert "0.00 EUR / 0.00 USD" in doc_text
        assert "NEURAL-CTRL-DYN-52" in doc_text
        assert "NEURAL-CTRL-DYN-51" in doc_text
        assert "PILOT16" in doc_text
        assert "PILOT17" in doc_text
        assert scanner.scan_text_for_payment_solicitation(doc_text, doc_rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(doc_text, doc_rel) == []
        assert scanner.scan_unicode_trojan_source(doc_text, doc_rel) == []
        intent = classify_request_intent(doc_text, source_channel="pull_request", changed_files=(doc_rel,))
        assert intent.intent_category == RequestIntentCategory.STANDARD_CONTRIBUTION
        assert intent.is_directive_order is False
        assert intent.protected_paths_touched == ()
        changed_files.append(
            {
                "filename": doc_rel,
                "status": "modified",
                "additions": len(doc_text.splitlines()),
                "patch": "",
                "content": doc_text,
            }
        )

    telemetry_json = json.dumps(
        {
            "nll_mean": -0.007602,
            "ece_value": 0.016648,
            "effective_dof": 1.1956,
            "bic_penalty": 0.001004,
            "general_drift_ratio": -0.00363,
            "anse_energy": -0.0086,
        },
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+TELEMETRY = {telemetry_json!r}"
        )
        == []
    )

    pr_title = (
        "feat(swarm): effective-DOF BIC/MDL complexity-penalized arm selector and "
        "two-stage variance-guarded James-Stein promoter (PILOT12 & PILOT16)"
    )
    pr_body = (
        "Implements NEURAL-CTRL-DYN-52 effective-DOF MDL/BIC complexity penalization "
        "and paired selection-variance guard grounded in PILOT12 and PILOT16 under "
        "ANSE Delta E < 0 and D0 <= +1.0%."
    )
    pr_intent = classify_request_intent(
        f"{pr_title}\n{pr_body}",
        source_channel="pull_request",
        changed_files=[c["filename"] for c in changed_files],
    )
    assert pr_intent.intent_category == RequestIntentCategory.STANDARD_CONTRIBUTION
    assert pr_intent.is_directive_order is False
    assert pr_intent.protected_paths_touched == ()

    pr_audit = scanner.audit_pull_request(
        pr_number=1252,
        pr_title=pr_title,
        pr_body=pr_body,
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[pr_title],
        author_email=None,
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []

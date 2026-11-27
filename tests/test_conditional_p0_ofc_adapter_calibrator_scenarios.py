# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-51: Conditional P_0/OFC Multi-Adapter Blender & Signed-Tau Norm Calibrator."""

from __future__ import annotations

import ast
import importlib.util
import json
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
from hivemind.swarm.conditional_p0_ofc_adapter_calibrator import (
    ANSE_MAXIMUM_PAIN,
    CALIBRATOR_RECEIPT_OBJECT,
    CALIBRATOR_SCHEMA,
    ConditionalP0OFCCalibratorConfig,
    calibrate_signed_tau_and_crossfit_js_norm,
    compute_conditional_convex_p0_ofc_blend,
    compute_ledoit_wolf_nullspace_projector,
    diagnose_adapter_d0_alignment_and_ofc_overlap,
    discover_ofc_value_neurons,
    execute_verified_conditional_p0_ofc_adapter_calibration,
    fit_crossfitted_james_stein_ofc_step,
    verify_conditional_calibrator_receipt_provenance,
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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "conditional_p0_ofc_adapter_calibrator.py"
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


class _MultiAdapterCausalLM(nn.Module):
    """Deterministic causal LM with orthogonal domain subspaces, OFC difficulty channel, and massive outliers."""

    def __init__(
        self,
        *,
        hidden_size: int = 100,
        vocab_size: int = 40,
        d0_base_gain: float = 0.65,
        spec_base_gain: float = 0.55,
        massive_amp: float = 18.0,
        ofc_col: int = 12,
        massive_cols: tuple[int, ...] = (60, 61, 62),
        seed: int = 20265101,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.ofc_col = ofc_col
        self.massive_cols = massive_cols
        _ = torch.Generator(device="cpu").manual_seed(seed)

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.early_layer = nn.Identity()
        self.adapter_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = _VarianceEpsRMSNorm(hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        with torch.no_grad():
            self.embed.weight.zero_()
            self.adapter_proj.weight.zero_()
            self.lm_head.weight.zero_()
            self.norm.weight.fill_(1.0)

            # Massive-activation outlier channels unrelated to chunk difficulty (OFC IoU == 0)
            for idx_m, m_col in enumerate(massive_cols):
                self.embed.weight[:, m_col] = float(massive_amp) * (1.0 if idx_m % 2 == 0 else -1.0)
                self.norm.weight[m_col] = 0.01

            # D0 general prose occupies dims 0..5 (tokens 0..9)
            for tok in range(0, 10):
                jitter = 0.08 * ((tok % 3) - 1)
                self.embed.weight[tok, 0:4] = torch.tensor(
                    [1.4 + jitter, 0.8 - 0.5 * jitter, 0.5 + 0.3 * jitter, 0.3], dtype=torch.float32
                )
                self.embed.weight[tok, 4:6] = torch.tensor([0.2, 0.1], dtype=torch.float32)
                if tok == 1:
                    self.lm_head.weight[tok, 0:4] = float(d0_base_gain) * torch.tensor(
                        [1.2, 0.7, 0.4, 0.2], dtype=torch.float32
                    )
                elif tok == 2:
                    self.lm_head.weight[tok, 0:4] = float(d0_base_gain) * torch.tensor(
                        [0.4, 0.2, 0.1, 0.1], dtype=torch.float32
                    )
                else:
                    self.lm_head.weight[tok, 0:4] = float(d0_base_gain) * torch.tensor(
                        [-0.5, -0.3, -0.2, -0.1], dtype=torch.float32
                    )

            # Non-OFC background variance across passive dims 48..98 & 6..19 (keeps median RMS ~0.45)
            gen_bg = torch.Generator(device="cpu").manual_seed(seed + 17)
            self.embed.weight[:, 48:98] = 0.45 * torch.randn((vocab_size, 50), generator=gen_bg)
            self.embed.weight[:, 6:20] = 0.45 * torch.randn((vocab_size, 14), generator=gen_bg)
            self.embed.weight[:, ofc_col] = 0.0
            for m_col in massive_cols:
                self.embed.weight[:, m_col] = float(massive_amp) * (1.0 if m_col % 2 == 0 else -1.0)

            # OFC value neuron `ofc_col` drives prediction confidence cleanly across easy vs hard chunks
            self.lm_head.weight[1, ofc_col] = 0.9
            self.lm_head.weight[11, ofc_col] = 0.9
            self.lm_head.weight[21, ofc_col] = 0.9
            self.lm_head.weight[31, ofc_col] = 0.9

            # Specialist domains D1, D2, D3 live in mutually orthogonal blocks (20..25, 30..35, 40..45)
            for d_idx in range(1, 4):
                tok_lo = 10 * d_idx
                dim_lo = 10 * (d_idx + 1)
                for tok in range(tok_lo, tok_lo + 10):
                    self.embed.weight[tok, dim_lo : dim_lo + 4] = torch.tensor(
                        [1.3, 0.9, 0.6, 0.4], dtype=torch.float32
                    )
                    if tok == tok_lo + 1:
                        self.lm_head.weight[tok, dim_lo : dim_lo + 4] = float(spec_base_gain) * torch.tensor(
                            [1.1, 0.8, 0.5, 0.3], dtype=torch.float32
                        )
                    elif tok == tok_lo + 2:
                        self.lm_head.weight[tok, dim_lo : dim_lo + 4] = float(spec_base_gain) * torch.tensor(
                            [0.3, 0.2, 0.1, 0.1], dtype=torch.float32
                        )
                    else:
                        self.lm_head.weight[tok, dim_lo : dim_lo + 4] = float(spec_base_gain) * torch.tensor(
                            [-0.4, -0.3, -0.2, -0.1], dtype=torch.float32
                        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        # Encode chunk difficulty signal in `ofc_col` (easy chunks start with tok % 10 == 1)
        easy_flag = ((input_ids[:, :1] % 10) == 1).to(dtype=h.dtype) * 2.0 - 1.0
        h = h.clone()
        h[..., self.ofc_col] = h[..., self.ofc_col] + 0.95 * easy_flag.expand_as(h[..., self.ofc_col])
        h_early = self.early_layer(h)
        h_post = h_early + self.adapter_proj(h_early)
        return self.lm_head(self.norm(h_post))


class _SignedTauTestLM(nn.Module):
    """Lightweight causal LM supporting both RMSNorm (no bias) and LayerNorm + head bias."""

    def __init__(
        self,
        *,
        hidden_size: int = 48,
        vocab_size: int = 40,
        logit_scale: float = 1.0,
        use_layernorm_with_bias: bool = False,
        spec_nullspace_signal: float = 0.0,
        seed: int = 20265102,
    ) -> None:
        super().__init__()
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
                self.norm.bias.fill_(0.05)
            if self.lm_head.bias is not None:
                self.lm_head.bias.copy_(0.04 * torch.randn((vocab_size,), generator=gen))

            # D0 prose uses dims 0..3
            for tok in range(0, 10):
                self.embed.weight[tok, 0:4] = torch.tensor([1.6, 1.0, 0.7, 0.4], dtype=torch.float32)
                if tok == 1:
                    self.lm_head.weight[tok, 0:4] = float(logit_scale) * torch.tensor(
                        [0.9, 0.6, 0.4, 0.2], dtype=torch.float32
                    )
                elif tok == 2:
                    self.lm_head.weight[tok, 0:4] = float(logit_scale) * torch.tensor(
                        [0.3, 0.2, 0.1, 0.05], dtype=torch.float32
                    )
                else:
                    self.lm_head.weight[tok, 0:4] = float(logit_scale) * torch.tensor(
                        [-0.35, -0.25, -0.15, -0.05], dtype=torch.float32
                    )

            # Specialists D1..D3 occupy dims 10..15 inside ker(G_D0)
            if spec_nullspace_signal > 0.0:
                self.norm.weight[10:14].fill_(0.20)
            for d_idx in range(1, 4):
                tok_lo = 10 * d_idx
                for tok in range(tok_lo, tok_lo + 10):
                    self.embed.weight[tok, 10:14] = torch.tensor([1.5, 1.1, 0.8, 0.5], dtype=torch.float32)
                    mult = float(logit_scale) * (1.0 + float(spec_nullspace_signal))
                    if tok == tok_lo + 1:
                        self.lm_head.weight[tok, 10:14] = mult * torch.tensor(
                            [0.85, 0.55, 0.35, 0.20], dtype=torch.float32
                        )
                    elif tok == tok_lo + 2:
                        self.lm_head.weight[tok, 10:14] = mult * torch.tensor(
                            [0.25, 0.15, 0.10, 0.05], dtype=torch.float32
                        )
                    else:
                        self.lm_head.weight[tok, 10:14] = mult * torch.tensor(
                            [-0.35, -0.25, -0.15, -0.05], dtype=torch.float32
                        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(self.embed(input_ids)))


def _make_domain_chunks(
    *,
    n_per_domain: int,
    seq_len: int = 16,
    seed: int = 20265110,
    d0_modal_prob: float = 0.72,
    spec_modal_prob: float = 0.82,
    include_difficulty_prefix: bool = True,
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
    for dom, lo, p_base in specs:
        for c_idx in range(n_per_domain):
            is_easy = (c_idx % 2 == 0) if include_difficulty_prefix else False
            p_modal = min(0.95, p_base + 0.18) if is_easy else max(0.35, p_base - 0.18)
            seq: list[int] = [lo + (1 if is_easy else 3)]
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
# Scenario 1: Benign Synergistic Multi-Adapter Regime (PILOT15 -> lambda = 0.0)
# ---------------------------------------------------------------------------


def test_s1_benign_synergistic_multi_adapter_selects_lambda_zero_no_stripping() -> None:
    """S1: Benign multi-adapter transfer selects lambda=0.0 retaining full gain without P_0 stripping."""
    model = _MultiAdapterCausalLM(hidden_size=100, vocab_size=40, d0_base_gain=0.45, spec_base_gain=0.45)
    probe_chunks = _make_domain_chunks(n_per_domain=8, seq_len=14, seed=5101, d0_modal_prob=0.72, spec_modal_prob=0.76)
    sel_chunks = _make_domain_chunks(n_per_domain=8, seq_len=14, seed=5102, d0_modal_prob=0.72, spec_modal_prob=0.76)
    eval_chunks = _make_domain_chunks(n_per_domain=8, seq_len=14, seed=5103, d0_modal_prob=0.72, spec_modal_prob=0.76)

    # 3 constructive adapters reinforcing D0 (dims 0..3) and specialists D1..D3 (dims 20..43)
    d_ltm = torch.zeros((100, 100), dtype=torch.float32)
    d_ltm[0:4, 0:4] = 0.30 * torch.eye(4)
    d_ltm[20:24, 20:24] = 0.35 * torch.eye(4)

    d_voice = torch.zeros((100, 100), dtype=torch.float32)
    d_voice[0:4, 0:4] = 0.25 * torch.eye(4)
    d_voice[30:34, 30:34] = 0.35 * torch.eye(4)

    d_simpo = torch.zeros((100, 100), dtype=torch.float32)
    d_simpo[0:4, 0:4] = 0.25 * torch.eye(4)
    d_simpo[40:44, 40:44] = 0.35 * torch.eye(4)

    delta_sum = d_ltm + d_voice + d_simpo

    # Direct check of (a) alignment + OFC IoU confound and (b) conditional blend lambda=0.0
    gen = torch.Generator(device="cpu").manual_seed(5111)
    d0_acts = torch.randn((48, 100), generator=gen, dtype=torch.float32) * 0.05
    d0_acts[:, 0:4] += 1.5
    p0_direct, sp_stats = compute_ledoit_wolf_nullspace_projector(d0_acts, energy_keep=0.985)
    assert sp_stats["nullspace_dim"] > 0

    early_h = torch.randn((32, 100), generator=gen, dtype=torch.float32) * 1.0
    early_h[:, 60:63] = 20.0  # Massive activations unrelated to chunk NLLs
    nll_probe = torch.linspace(1.0, 3.0, steps=32)
    early_h[:, 12] = -1.8 * (nll_probe - 2.0)  # <1% OFC value neuron strictly encoding value/NLL at normal RMS

    diag = diagnose_adapter_d0_alignment_and_ofc_overlap(
        early_h,
        nll_probe,
        base_d0_nll=2.50,
        merged_d0_nll=2.45,
        delta_w=delta_sum,
        d0_token_activations=d0_acts,
        adapter_deltas={"ltm": d_ltm, "voice": d_voice, "simpo": d_simpo},
        trigger_threshold=0.005,
    )
    assert diag["d0_drift_probe"] == pytest.approx(-0.02, abs=1e-9)
    assert diag["transfer_regime"] == "benign_synergistic"
    assert diag["is_synergistic"] is True
    assert diag["requires_p0_ofc_protection"] is False
    assert diag["oof_auroc"] >= 0.85
    assert diag["ofc_iou"] == 0.0
    assert diag["ofc_disjoint_from_massive"] is True
    assert diag["adapter_alignment"]["num_adapters"] == 3
    assert diag["delta_alignment"]["prose_subspace_energy_frac"] > 0.10

    blended_benign, blend_meta = compute_conditional_convex_p0_ofc_blend(
        delta_sum,
        p0_direct,
        diag["ofc_indices"],
        d0_drift_probe=diag["d0_drift_probe"],
        trigger_threshold=0.005,
    )
    assert blend_meta["blend_lambda"] == 0.0
    assert blend_meta["stripping_applied"] is False
    assert blend_meta["protection_mode"] == "unstripped_benign"
    assert blend_meta["retained_frob_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert torch.equal(blended_benign, delta_sum)

    # Compare against unconditional P_0 stripping (which destroys the D0 synergy)
    stripped_uncond = delta_sum.to(torch.float64) @ p0_direct
    assert float(torch.linalg.norm(stripped_uncond).item()) < 0.85 * float(torch.linalg.norm(delta_sum).item())

    # End-to-end verified pipeline on _MultiAdapterCausalLM (inplace=False leaves parent pristine)
    base_sha = compute_module_state_sha256(model)
    cfg = ConditionalP0OFCCalibratorConfig(
        trigger_threshold=0.005,
        full_protection_threshold=0.010,
        enable_norm_calibration=False,
    )
    edited, receipt = execute_verified_conditional_p0_ofc_adapter_calibration(
        model,
        {"adapter_proj": {"ltm_local": d_ltm, "personal_voice": d_voice, "simpo_audience": d_simpo}},
        probe_chunks,
        selection_chunks=sel_chunks,
        eval_chunks=eval_chunks,
        early_layer_attr="early_layer",
        config=cfg,
        inplace=False,
    )
    assert compute_module_state_sha256(model) == base_sha
    assert receipt["object"] == CALIBRATOR_RECEIPT_OBJECT
    assert receipt["schema"] == CALIBRATOR_SCHEMA
    assert verify_conditional_calibrator_receipt_provenance(receipt) is True
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["blend_lambda"] == 0.0
    assert receipt["protection_mode"] == "unstripped_benign"
    assert receipt["alignment_diagnosis"]["transfer_regime"] == "benign_synergistic"
    assert receipt["ofc_iou"] == 0.0
    assert receipt["oof_auroc"] >= 0.80
    assert receipt["general_drift_ratio"] < 0.0
    assert receipt["delta_e"] < 0.0
    assert receipt["nll_mean"] < receipt["base_nll_mean"]
    assert torch.allclose(edited.adapter_proj.weight, delta_sum, atol=1e-6)


# ---------------------------------------------------------------------------
# Scenario 2: Hostile/Forgetting Adapter Activates Convex P_0 + OFC Protection
# ---------------------------------------------------------------------------


def test_s2_hostile_forgetting_adapter_activates_convex_p0_ofc_zero_leak() -> None:
    """S2: Hostile adapter breaching D0 > +1.0% activates P_0 + OFC protection with 0.0 OFC neuron leak."""
    # Smooth convex ramp verification across drift thresholds
    p0_eye = torch.eye(100, dtype=torch.float64)
    p0_eye[0:4, 0:4] = 0.0  # D0 prose subspace dims 0..3 projected out
    dw_test = torch.ones((100, 100), dtype=torch.float32)

    _, m_soft = compute_conditional_convex_p0_ofc_blend(
        dw_test, p0_eye, [12], d0_drift_probe=0.0075, trigger_threshold=0.005
    )
    assert 0.0 < m_soft["blend_lambda"] < 1.0
    assert m_soft["protection_mode"] == "convex_ramp_p0_ofc"
    assert m_soft["stripping_applied"] is True

    blended_full, m_full = compute_conditional_convex_p0_ofc_blend(
        dw_test, p0_eye, [12], d0_drift_probe=0.025, trigger_threshold=0.005
    )
    assert m_full["blend_lambda"] == 1.0
    assert m_full["protection_mode"] == "full_p0_ofc_protected"
    assert m_full["ofc_input_leak"] == 0.0
    assert float(blended_full[:, 12].abs().max().item()) == 0.0
    assert float(blended_full[:, 0:4].abs().max().item()) < 1e-6

    # End-to-end model rescue: hostile adapters poison D0 (dims 0..3) AND OFC dim 12 while helping D1..D3
    model = _MultiAdapterCausalLM(hidden_size=100, vocab_size=40, d0_base_gain=0.70, spec_base_gain=0.40)
    probe_chunks = _make_domain_chunks(n_per_domain=8, seq_len=14, seed=5201, d0_modal_prob=0.80, spec_modal_prob=0.74)
    sel_chunks = _make_domain_chunks(n_per_domain=8, seq_len=14, seed=5202, d0_modal_prob=0.80, spec_modal_prob=0.74)
    eval_chunks = _make_domain_chunks(n_per_domain=8, seq_len=14, seed=5203, d0_modal_prob=0.80, spec_modal_prob=0.74)

    d_hostile_1 = torch.zeros((100, 100), dtype=torch.float32)
    # Invert D0 prose subspace -> catastrophic forgetting if left unguarded
    d_hostile_1[0:4, 0:4] = -2.2 * torch.eye(4)
    # Corrupt OFC value neuron column 12 on D0 output row
    d_hostile_1[0:4, 12] = -1.5
    d_hostile_1[20:24, 20:24] = 0.75 * torch.eye(4)

    d_hostile_2 = torch.zeros((100, 100), dtype=torch.float32)
    d_hostile_2[0:4, 0:4] = -1.8 * torch.eye(4)
    d_hostile_2[30:34, 30:34] = 0.75 * torch.eye(4)
    d_hostile_2[40:44, 40:44] = 0.75 * torch.eye(4)

    cfg = ConditionalP0OFCCalibratorConfig(
        trigger_threshold=0.005,
        full_protection_threshold=0.010,
        energy_keep=0.95,
        enable_norm_calibration=False,
    )
    edited, receipt = execute_verified_conditional_p0_ofc_adapter_calibration(
        model,
        {"adapter_proj": [d_hostile_1, d_hostile_2]},
        probe_chunks,
        selection_chunks=sel_chunks,
        eval_chunks=eval_chunks,
        config=cfg,
        inplace=True,
    )
    assert verify_conditional_calibrator_receipt_provenance(receipt) is True
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["alignment_diagnosis"]["d0_drift_probe"] > 0.01
    assert receipt["alignment_diagnosis"]["transfer_regime"] == "hostile_forgetting"
    assert receipt["alignment_diagnosis"]["requires_p0_ofc_protection"] is True
    assert receipt["blend_lambda"] == 1.0
    assert receipt["protection_mode"] == "full_p0_ofc_protected"
    assert receipt["ofc_input_leak"] == 0.0
    assert receipt["general_drift_ratio"] <= 0.01
    assert receipt["delta_e"] < 0.0
    assert receipt["nll_mean"] < receipt["base_nll_mean"]
    ofc_idx = receipt["alignment_diagnosis"]["ofc_indices"]
    assert 12 in ofc_idx
    assert float(edited.adapter_proj.weight[:, ofc_idx].abs().max().item()) == 0.0


# ---------------------------------------------------------------------------
# Scenario 3: Signed-Tau RLHF Instruct (<1.0) vs Base (>1.0) & Linearity
# ---------------------------------------------------------------------------


def test_s3_signed_tau_distinguishes_instruct_overconfident_vs_base_underconfident_and_linearity() -> None:
    """S3: Signed-tau calibration separates Instruct (tau < 1.0) from Base (tau > 1.0) at <1e-6 linearity."""
    # Part A: Over-confident RLHF / Instruct logits (tau < 1.0 softens logits and cuts ECE ~50%+)
    instruct_model = _SignedTauTestLM(
        hidden_size=48, vocab_size=40, logit_scale=2.5, use_layernorm_with_bias=False, spec_nullspace_signal=0.15
    )
    inst_probe = _make_domain_chunks(
        n_per_domain=8, seq_len=14, seed=5301, d0_modal_prob=0.60, spec_modal_prob=0.62, include_difficulty_prefix=False
    )
    inst_sel = _make_domain_chunks(
        n_per_domain=8, seq_len=14, seed=5302, d0_modal_prob=0.60, spec_modal_prob=0.62, include_difficulty_prefix=False
    )
    inst_eval = _make_domain_chunks(
        n_per_domain=8, seq_len=14, seed=5303, d0_modal_prob=0.60, spec_modal_prob=0.62, include_difficulty_prefix=False
    )

    cfg_signed = ConditionalP0OFCCalibratorConfig(
        tau_grid=(0.40, 0.55, 0.70, 0.82, 0.94, 1.00, 1.05, 1.25, 1.60, 2.00),
        alpha_grid=(0.0, 0.25, 0.50, 1.0),
        js_max_rank=4,
        js_max_step_norm=0.08,
        ece_energy_weight=0.25,
    )
    _, rec_inst = calibrate_signed_tau_and_crossfit_js_norm(
        instruct_model,
        inst_probe,
        selection_chunks=inst_sel,
        eval_chunks=inst_eval,
        protected_indices=[5, 7],
        config=cfg_signed,
        inplace=False,
    )
    assert verify_conditional_calibrator_receipt_provenance(rec_inst) is True
    assert rec_inst["accepted"] is True
    assert rec_inst["tau_temp"] < 1.0
    assert rec_inst["calibration_regime"] == "overconfident_rlhf_instruct"
    assert rec_inst["ece_rel_delta"] <= -0.35
    assert rec_inst["general_drift_ratio"] <= 0.01
    assert rec_inst["delta_e"] < 0.0
    assert rec_inst["ofc_protected_max_delta"] == 0.0
    assert rec_inst["d0_mean_grad_ortho_err"] < 1e-6
    assert rec_inst["logit_linearity_max_abs_diff"] < 1e-6

    # Part B: Under-confident Base logits with LayerNorm + head bias (tau > 1.0 sharpens logits)
    base_model = _SignedTauTestLM(
        hidden_size=48, vocab_size=40, logit_scale=0.35, use_layernorm_with_bias=True, spec_nullspace_signal=0.30
    )
    base_probe = _make_domain_chunks(
        n_per_domain=8, seq_len=14, seed=5311, d0_modal_prob=0.76, spec_modal_prob=0.80, include_difficulty_prefix=False
    )
    base_sel = _make_domain_chunks(
        n_per_domain=8, seq_len=14, seed=5312, d0_modal_prob=0.76, spec_modal_prob=0.80, include_difficulty_prefix=False
    )
    base_eval = _make_domain_chunks(
        n_per_domain=8, seq_len=14, seed=5313, d0_modal_prob=0.76, spec_modal_prob=0.80, include_difficulty_prefix=False
    )

    _, rec_base = calibrate_signed_tau_and_crossfit_js_norm(
        base_model,
        base_probe,
        selection_chunks=base_sel,
        eval_chunks=base_eval,
        protected_indices=[6, 8],
        config=cfg_signed,
        norm_kind="layernorm",
        inplace=True,
    )
    assert verify_conditional_calibrator_receipt_provenance(rec_base) is True
    assert rec_base["accepted"] is True
    assert rec_base["tau_temp"] > 1.0
    assert rec_base["calibration_regime"] == "underconfident_base"
    assert rec_base["norm_kind"] == "layernorm"
    assert rec_base["general_drift_ratio"] <= 0.01
    assert rec_base["delta_e"] < 0.0
    assert rec_base["ofc_protected_max_delta"] == 0.0
    assert rec_base["d0_mean_grad_ortho_err"] < 1e-6
    assert rec_base["logit_linearity_max_abs_diff"] < 1e-6

    # Verify crossfit JS step directly preserves ker(G_D0) intersect S_val^perp
    gen = torch.Generator(device="cpu").manual_seed(5320)
    g_d0 = torch.randn((8, 32), generator=gen, dtype=torch.float64)
    g_sp = torch.randn((16, 32), generator=gen, dtype=torch.float64) * 0.1
    g_sp[:, 12] += 1.5
    g_sp[:, 14] -= 1.2
    js_res = fit_crossfitted_james_stein_ofc_step(
        g_sp, g_d0, protected_indices=[12, 19], max_rank=4, max_step_norm=0.08
    )
    assert js_res["zero_step"] is False
    assert js_res["ofc_protected_max_delta"] == 0.0
    assert js_res["d0_mean_grad_ortho_err"] < 1e-6


# ---------------------------------------------------------------------------
# Scenario 4: ANSE Maximum Pain Atomic Bitwise Rollback on Infeasible Guard
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_bitwise_rollback() -> None:
    """S4: Guard breach or non-improving eval triggers ANSE Maximum Pain and exact bitwise rollback."""
    # Case A: Signed-tau calibration breaches tight guard on shifted eval -> ANSE Maximum Pain rollback
    m_guard = _SignedTauTestLM(hidden_size=32, vocab_size=40, logit_scale=0.40, spec_nullspace_signal=0.2)
    init_sha = compute_module_state_sha256(m_guard)
    init_sd = {k: v.detach().clone() for k, v in m_guard.state_dict().items()}

    p_ch = _make_domain_chunks(n_per_domain=6, seq_len=12, seed=5401, d0_modal_prob=0.80, spec_modal_prob=0.80)
    s_ch = _make_domain_chunks(n_per_domain=6, seq_len=12, seed=5402, d0_modal_prob=0.80, spec_modal_prob=0.80)
    e_ch = _make_domain_chunks(n_per_domain=6, seq_len=12, seed=5403, d0_modal_prob=0.35, spec_modal_prob=0.35)

    _, rec_guard = calibrate_signed_tau_and_crossfit_js_norm(
        m_guard,
        p_ch,
        selection_chunks=s_ch,
        eval_chunks=e_ch,
        config=ConditionalP0OFCCalibratorConfig(
            tau_grid=(1.0, 1.5, 2.2),
            max_general_drift_rel=0.001,
            max_domain_drift_rel=0.05,
        ),
        inplace=True,
    )
    assert verify_conditional_calibrator_receipt_provenance(rec_guard) is True
    assert rec_guard["accepted"] is False
    assert rec_guard["rolled_back"] is True
    assert rec_guard["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rec_guard["cand_anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_guard["final_state_sha256"] == init_sha
    for key, orig in init_sd.items():
        assert torch.equal(m_guard.state_dict()[key], orig)

    # Case B: Relaxed guard on non-improving eval -> ROLLED_BACK_ENERGY_NON_IMPROVING
    _, rec_worse = calibrate_signed_tau_and_crossfit_js_norm(
        m_guard,
        p_ch,
        selection_chunks=s_ch,
        eval_chunks=e_ch,
        config=ConditionalP0OFCCalibratorConfig(
            tau_grid=(1.0, 1.5, 2.2),
            max_general_drift_rel=5.0,
            max_domain_drift_rel=5.0,
        ),
        inplace=True,
    )
    assert rec_worse["accepted"] is False
    assert rec_worse["rolled_back"] is True
    assert rec_worse["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert rec_worse["final_state_sha256"] == init_sha

    # Case C: Fixed point unchanged (tau_grid=(1.0,), alpha_grid=(0.0,))
    _, rec_fixed = calibrate_signed_tau_and_crossfit_js_norm(
        m_guard,
        p_ch,
        config=ConditionalP0OFCCalibratorConfig(tau_grid=(1.0,), alpha_grid=(0.0,)),
        inplace=True,
    )
    assert rec_fixed["accepted"] is False
    assert rec_fixed["rolled_back"] is False
    assert rec_fixed["status"] == "FIXED_POINT_UNCHANGED"
    assert rec_fixed["final_state_sha256"] == init_sha

    # Case D: End-to-end adapter calibration triggers atomic rollback on infeasible adapter
    m_ad = _MultiAdapterCausalLM(hidden_size=100, vocab_size=40, d0_base_gain=0.70, spec_base_gain=0.70)
    ad_sha = compute_module_state_sha256(m_ad)
    ad_probe = _make_domain_chunks(n_per_domain=6, seq_len=12, seed=5411, d0_modal_prob=0.80, spec_modal_prob=0.80)
    ad_bad_delta = -0.35 * torch.eye(100, dtype=torch.float32)  # Uniformly degrades predictions
    _, rec_ad_rb = execute_verified_conditional_p0_ofc_adapter_calibration(
        m_ad,
        {"adapter_proj": ad_bad_delta},
        ad_probe,
        config=ConditionalP0OFCCalibratorConfig(enable_norm_calibration=False),
        inplace=True,
    )
    assert verify_conditional_calibrator_receipt_provenance(rec_ad_rb) is True
    assert rec_ad_rb["accepted"] is False
    assert rec_ad_rb["rolled_back"] is True
    assert rec_ad_rb["final_state_sha256"] == ad_sha

    # Case E: Tampered receipt fails provenance verification
    tampered = dict(rec_ad_rb)
    tampered["nll_mean"] = float(rec_ad_rb["nll_mean"]) + 0.25
    assert verify_conditional_calibrator_receipt_provenance(tampered) is False


# ---------------------------------------------------------------------------
# Scenario 5: Validation Errors & Security / Stub / OwnerOrderGuardrail Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_guardrail_compliance() -> None:
    """S5: Exhaustive validation edges plus AGPL-3.0, ZeroStubAudit, PRSecurityScanner, and OwnerOrderGuardrail."""
    # 1. Config & helper validation
    with pytest.raises(ValueError, match="> trigger_threshold"):
        ConditionalP0OFCCalibratorConfig(trigger_threshold=0.01, full_protection_threshold=0.01)
    with pytest.raises(ValueError, match="< 1.0"):
        ConditionalP0OFCCalibratorConfig(energy_keep=1.0)
    with pytest.raises(ValueError, match="<= 1.0"):
        ConditionalP0OFCCalibratorConfig(ofc_fraction=1.5)
    with pytest.raises(ValueError, match="> 1.0"):
        ConditionalP0OFCCalibratorConfig(massive_ratio_threshold=1.0)
    with pytest.raises(ValueError, match="< 1.0"):
        ConditionalP0OFCCalibratorConfig(nullspace_sv_tol=1.0)
    with pytest.raises(TypeError, match="non-empty sequence"):
        ConditionalP0OFCCalibratorConfig(tau_grid="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        ConditionalP0OFCCalibratorConfig(tau_grid=())
    with pytest.raises(ValueError, match=">= 0.0"):
        ConditionalP0OFCCalibratorConfig(alpha_grid=(-0.1,))
    with pytest.raises(TypeError, match="non-empty sequence"):
        ConditionalP0OFCCalibratorConfig(general_prefixes="D0_")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        ConditionalP0OFCCalibratorConfig(general_prefixes=())
    with pytest.raises(TypeError, match="bool"):
        ConditionalP0OFCCalibratorConfig(enable_norm_calibration="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        ConditionalP0OFCCalibratorConfig(js_max_rank=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        ConditionalP0OFCCalibratorConfig(js_max_rank=0)
    with pytest.raises(ValueError, match=">= 0"):
        ConditionalP0OFCCalibratorConfig(score_from=-1)
    with pytest.raises(ValueError, match="finite"):
        ConditionalP0OFCCalibratorConfig(js_max_step_norm=float("nan"))

    # 2. Ledoit-Wolf, OFC discovery, diagnosis & blend validation
    with pytest.raises(ValueError, match="< 1.0"):
        compute_ledoit_wolf_nullspace_projector(torch.ones((4, 4)), energy_keep=1.0)
    p0_1d, st_1d = compute_ledoit_wolf_nullspace_projector(torch.ones((1, 1)))
    assert p0_1d.shape == (1, 1) and st_1d["nullspace_dim"] == 1

    with pytest.raises(ValueError, match="must match"):
        discover_ofc_value_neurons(torch.ones((4, 4)), torch.ones(3))
    with pytest.raises(ValueError, match="at least 4"):
        discover_ofc_value_neurons(torch.ones((3, 4)), torch.ones(3))
    with pytest.raises(ValueError, match="<= 1.0"):
        discover_ofc_value_neurons(torch.ones((4, 4)), torch.ones(4), ofc_fraction=1.5)
    with pytest.raises(ValueError, match="> 1.0"):
        discover_ofc_value_neurons(torch.ones((4, 4)), torch.ones(4), massive_ratio_threshold=1.0)
    # Single-class NLL fallback -> oof_auroc == 0.5
    ofc_const = discover_ofc_value_neurons(torch.ones((4, 8)), torch.ones(4))
    assert ofc_const["oof_auroc"] == 0.5

    with pytest.raises(ValueError, match="square"):
        diagnose_adapter_d0_alignment_and_ofc_overlap(
            torch.ones((4, 8)), torch.linspace(1.0, 2.0, 4), base_d0_nll=2.0, merged_d0_nll=2.0, p0=torch.ones((4, 8))
        )
    with pytest.raises(ValueError, match="must match delta_w"):
        diagnose_adapter_d0_alignment_and_ofc_overlap(
            torch.ones((4, 8)),
            torch.linspace(1.0, 2.0, 4),
            base_d0_nll=2.0,
            merged_d0_nll=2.0,
            delta_w=torch.ones((4, 6)),
            p0=torch.eye(8),
        )
    with pytest.raises(TypeError, match="mapping or sequence"):
        diagnose_adapter_d0_alignment_and_ofc_overlap(
            torch.ones((4, 8)),
            torch.linspace(1.0, 2.0, 4),
            base_d0_nll=2.0,
            merged_d0_nll=2.0,
            adapter_deltas="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="at least 2"):
        diagnose_adapter_d0_alignment_and_ofc_overlap(
            torch.ones((4, 8)),
            torch.linspace(1.0, 2.0, 4),
            base_d0_nll=2.0,
            merged_d0_nll=2.0,
            adapter_deltas=[torch.ones((4, 8))],
        )
    with pytest.raises(ValueError, match="must match"):
        diagnose_adapter_d0_alignment_and_ofc_overlap(
            torch.ones((4, 8)),
            torch.linspace(1.0, 2.0, 4),
            base_d0_nll=2.0,
            merged_d0_nll=2.0,
            adapter_deltas=[torch.ones((4, 8)), torch.ones((4, 6))],
        )
    d_within = diagnose_adapter_d0_alignment_and_ofc_overlap(
        torch.ones((4, 8)),
        torch.linspace(1.0, 2.0, 4),
        base_d0_nll=1.0,
        merged_d0_nll=1.003,
        delta_w=torch.zeros((4, 4)),
        adapter_deltas=[torch.zeros((4, 4)), torch.zeros((4, 4))],
    )
    assert d_within["transfer_regime"] == "benign_within_trigger"

    with pytest.raises(ValueError, match="must match"):
        compute_conditional_convex_p0_ofc_blend(
            torch.ones((4, 6)), torch.eye(4), [], d0_drift_probe=0.0
        )
    with pytest.raises(ValueError, match="out of bounds"):
        compute_conditional_convex_p0_ofc_blend(
            torch.ones((4, 6)), torch.eye(6), [9], d0_drift_probe=0.0
        )
    with pytest.raises(ValueError, match="> trigger_threshold"):
        compute_conditional_convex_p0_ofc_blend(
            torch.ones((4, 6)),
            torch.eye(6),
            [],
            d0_drift_probe=0.0,
            trigger_threshold=0.01,
            full_protection_threshold=0.005,
        )
    with pytest.raises(ValueError, match="1D integer tensor"):
        compute_conditional_convex_p0_ofc_blend(
            torch.ones((4, 6)), torch.eye(6), torch.ones((2, 2), dtype=torch.long), d0_drift_probe=0.0
        )
    with pytest.raises(TypeError, match="sequence or 1D"):
        compute_conditional_convex_p0_ofc_blend(
            torch.ones((4, 6)), torch.eye(6), "bad", d0_drift_probe=0.0  # type: ignore[arg-type]
        )

    # 3. James-Stein & model calibration validation
    with pytest.raises(ValueError, match="must match"):
        fit_crossfitted_james_stein_ofc_step(torch.ones((4, 6)), torch.ones((4, 5)))
    with pytest.raises(ValueError, match="< 1.0"):
        fit_crossfitted_james_stein_ofc_step(torch.ones((4, 6)), torch.ones((4, 6)), sv_tol=1.0)
    assert fit_crossfitted_james_stein_ofc_step(torch.ones((4, 4)), torch.eye(4))["zero_step"] is True
    assert fit_crossfitted_james_stein_ofc_step(torch.ones((2, 4)), torch.zeros((2, 4)))["zero_step"] is True
    assert fit_crossfitted_james_stein_ofc_step(torch.zeros((6, 4)), torch.zeros((2, 4)))["zero_step"] is True

    m_ok = _SignedTauTestLM(hidden_size=16, vocab_size=40, seed=5501)
    good_ch = _make_domain_chunks(n_per_domain=4, seq_len=8, seed=5502)
    with pytest.raises(TypeError, match="ConditionalP0OFCCalibratorConfig"):
        calibrate_signed_tau_and_crossfit_js_norm(m_ok, good_ch, config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="nn.Module"):
        calibrate_signed_tau_and_crossfit_js_norm("bad", good_ch)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="norm_kind"):
        calibrate_signed_tau_and_crossfit_js_norm(m_ok, good_ch, norm_kind="bad")
    with pytest.raises(ValueError, match="final normalization"):
        calibrate_signed_tau_and_crossfit_js_norm(nn.Linear(4, 4), good_ch)
    with pytest.raises(ValueError, match="unembedding head"):
        no_h = nn.Module()
        no_h.norm = nn.RMSNorm(4)
        calibrate_signed_tau_and_crossfit_js_norm(no_h, good_ch)
    with pytest.raises(ValueError, match="must match norm dim"):
        bad_hw = nn.Module()
        bad_hw.norm = nn.RMSNorm(4)
        bad_hw.head = nn.Linear(5, 8, bias=False)
        calibrate_signed_tau_and_crossfit_js_norm(bad_hw, good_ch)
    with pytest.raises(ValueError, match="non-empty sequence"):
        calibrate_signed_tau_and_crossfit_js_norm(m_ok, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 4"):
        calibrate_signed_tau_and_crossfit_js_norm(m_ok, good_ch[:2])
    with pytest.raises(ValueError, match="Automatic"):
        calibrate_signed_tau_and_crossfit_js_norm(
            m_ok,
            [
                {"domain": "D0_g", "ids": [0, 1, 2, 3]},
                {"domain": "D1_s", "ids": [10, 11, 12, 13]},
                {"domain": "D0_g", "ids": [0, 1, 2, 3]},
                {"domain": "D0_g", "ids": [0, 1, 2, 3]},
            ],
        )
    with pytest.raises(ValueError, match="both general"):
        calibrate_signed_tau_and_crossfit_js_norm(
            m_ok, [{"domain": "D0_a", "ids": [0, 1, 2, 3]}, {"domain": "D0_b", "ids": [0, 1, 2, 3]}], selection_chunks=good_ch
        )
    with pytest.raises(TypeError, match="integer dtype"):
        calibrate_signed_tau_and_crossfit_js_norm(
            m_ok, [{"domain": "D0_g", "ids": torch.ones(4)}], selection_chunks=good_ch
        )
    with pytest.raises(TypeError, match="sequence or 1D"):
        calibrate_signed_tau_and_crossfit_js_norm(
            m_ok, [{"domain": "D0_g", "ids": "bad"}], selection_chunks=good_ch
        )
    with pytest.raises(ValueError, match="1D"):
        calibrate_signed_tau_and_crossfit_js_norm(
            m_ok, [{"domain": "D0_g", "ids": torch.ones((2, 2), dtype=torch.long)}], selection_chunks=good_ch
        )
    with pytest.raises(ValueError, match="length >= 2"):
        calibrate_signed_tau_and_crossfit_js_norm(m_ok, [{"domain": "D0_g", "ids": [1]}], selection_chunks=good_ch)
    with pytest.raises(ValueError, match="token IDs"):
        calibrate_signed_tau_and_crossfit_js_norm(m_ok, [{"domain": "D0_g", "ids": [0, 999]}], selection_chunks=good_ch)
    with pytest.raises(ValueError, match="too short"):
        calibrate_signed_tau_and_crossfit_js_norm(
            m_ok,
            good_ch,
            selection_chunks=good_ch,
            config=ConditionalP0OFCCalibratorConfig(score_from=20),
        )
    with pytest.raises(ValueError, match="share sequence length"):
        calibrate_signed_tau_and_crossfit_js_norm(
            m_ok,
            [{"domain": "D0_g", "ids": [0, 1, 2, 3]}, {"domain": "D1_s", "ids": [10, 11, 12]}],
            selection_chunks=good_ch,
        )
    with pytest.raises(ValueError, match="non-empty mapping"):
        execute_verified_conditional_p0_ofc_adapter_calibration(m_ok, {}, good_ch)
    with pytest.raises(TypeError, match="ConditionalP0OFCCalibratorConfig"):
        execute_verified_conditional_p0_ofc_adapter_calibration(m_ok, {"embed": torch.zeros((40, 16))}, good_ch, config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="does not exist"):
        execute_verified_conditional_p0_ofc_adapter_calibration(m_ok, {"missing": torch.zeros((4, 4))}, good_ch)
    with pytest.raises(ValueError, match="2D weight"):
        execute_verified_conditional_p0_ofc_adapter_calibration(m_ok, {"norm": torch.zeros((4, 4))}, good_ch)
    with pytest.raises(ValueError, match="must match"):
        execute_verified_conditional_p0_ofc_adapter_calibration(m_ok, {"lm_head": torch.zeros((4, 4))}, good_ch)
    with pytest.raises(ValueError, match="early_layer_attr"):
        execute_verified_conditional_p0_ofc_adapter_calibration(
            _MultiAdapterCausalLM(), {"adapter_proj": torch.zeros((100, 100))}, good_ch, early_layer_attr="missing"
        )
    with pytest.raises(TypeError, match="Mapping"):
        verify_conditional_calibrator_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_conditional_calibrator_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_conditional_calibrator_receipt_provenance({"receipt_sha256": "0" * 64, "object": "bad"}) is False
    assert (
        verify_conditional_calibrator_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": CALIBRATOR_RECEIPT_OBJECT,
                "schema": CALIBRATOR_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )

    # 4. AGPL-3.0, ZeroStubAudit (CI-safe), PRSecurityScanner, lateral movement, FORBIDDEN_ADDED & OwnerOrderGuardrail
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn51", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn51", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/conditional_p0_ofc_adapter_calibrator.py", TARGET_MODULE_PATH),
        ("tests/test_conditional_p0_ofc_adapter_calibrator_scenarios.py", TEST_FILE_PATH),
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
        assert "NEURAL-CTRL-DYN-51" in doc_text
        assert "PILOT15" in doc_text
        assert "PILOT16" in doc_text
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
            "nll_mean": 2.92628,
            "ece_value": 0.01631,
            "general_drift_ratio": -0.01670,
            "ofc_iou": 0.0,
            "anse_energy": -0.03826,
        },
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+TELEMETRY = {telemetry_json!r}"
        )
        == []
    )

    pr_title = "feat(swarm): conditional convex P_0/OFC multi-adapter blender and signed-tau Base-vs-Instruct RLHF norm calibrator (PILOT15)"
    pr_body = (
        "Implements NEURAL-CTRL-DYN-51 conditional convex P_0/OFC multi-adapter blender and "
        "signed-tau cross-fitted James-Stein norm calibrator inside ker(G_D0) intersect S_val^perp "
        "grounded in PILOT15 and PILOT16 under ANSE Delta E < 0 and D0 <= +1.0%."
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
        pr_number=1251,
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
